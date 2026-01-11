import os
import math
import tempfile
import requests
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional auto-refresh
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

# DhanHQ (NO DhanContext)
from dhanhq import dhanhq
import importlib.metadata as ilm

IST = ZoneInfo("Asia/Kolkata")
APP_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="F&O Intraday Scanner", layout="wide")


# -------------------- Helpers: secrets --------------------
def sget(path, default=None):
    cur = st.secrets
    try:
        for k in path:
            cur = cur[k]
        return cur
    except Exception:
        return default


# -------------------- Dhan client --------------------
@st.cache_resource
def get_dhan_client(client_id: str, access_token: str):
    return dhanhq(client_id, access_token)


# -------------------- Indicators --------------------
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def dema(series: pd.Series, period: int) -> pd.Series:
    e1 = ema(series, period)
    e2 = ema(e1, period)
    return 2 * e1 - e2

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1
    ).max(axis=1)

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range(high, low, close)
    atr_ = tr.ewm(alpha=1/period, adjust=False).mean()

    plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(alpha=1/period, adjust=False).mean() / atr_
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(alpha=1/period, adjust=False).mean() / atr_

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_ = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx_, plus_di, minus_di

def vwap_intraday(df_today: pd.DataFrame) -> pd.Series:
    tp = (df_today["high"] + df_today["low"] + df_today["close"]) / 3.0
    return (tp * df_today["volume"]).cumsum() / df_today["volume"].cumsum()


# -------------------- Score --------------------
def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))

def score_signal(side: str, adx_v: float, rsi_v: float, rel_vol: float, atr_pct: float, di_gap: float) -> float:
    adx_s = clamp((adx_v - 20.0) / 30.0)
    rv_s  = clamp((rel_vol - 1.0) / 2.0)
    atr_s = clamp((atr_pct - 0.8) / 2.2)
    di_s  = clamp(di_gap / 25.0)

    if side == "BULL":
        rsi_s = clamp((rsi_v - 50.0) / 30.0)
    else:
        rsi_s = clamp((50.0 - rsi_v) / 30.0)

    return float(100.0 * (0.30 * adx_s + 0.25 * rsi_s + 0.25 * rv_s + 0.10 * di_s + 0.10 * atr_s))


# -------------------- Normalize Dhan response (FIXED) --------------------
def epoch_to_dt(epoch_val: int) -> datetime:
    if epoch_val > 10_000_000_000:  # ms
        epoch_val = int(epoch_val / 1000)
    return datetime.fromtimestamp(int(epoch_val), tz=IST)

def normalize_intraday(resp) -> pd.DataFrame:
    """
    FIXED: Never throws ValueError even if Dhan returns error payloads.
    Supports:
    - dict with "data": list[dict]
    - dict with "data": dict of arrays (open/high/low/close/volume)
    - list[dict]
    - error dicts (returns empty)
    """
    if resp is None or not isinstance(resp, (dict, list)):
        return pd.DataFrame()

    # list of candles
    if isinstance(resp, list):
        df = pd.DataFrame(resp)
        return df

    # dict response
    if "data" not in resp:
        return pd.DataFrame()

    data = resp["data"]

    # data: list[dict]
    if isinstance(data, list):
        return pd.DataFrame(data)

    # data: dict (arrays or error scalars)
    if isinstance(data, dict):
        if len(data) == 0:
            return pd.DataFrame()

        # If all values are scalars -> error payload, NOT candle data
        if all(not isinstance(v, (list, tuple, np.ndarray, pd.Series)) for v in data.values()):
            return pd.DataFrame()

        # Build from arrays
        key_map = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
        cols = {}
        for k, v in data.items():
            kk = key_map.get(k, k)
            if kk in ["open", "high", "low", "close", "volume"] and isinstance(v, (list, tuple, np.ndarray)):
                cols[kk] = v

        if "close" not in cols:
            return pd.DataFrame()

        df = pd.DataFrame(cols)

        # time column if present
        ts_key = next((k for k in ["start_Time", "startTime", "ts", "timestamp", "time"] if k in data), None)
        if ts_key and isinstance(data[ts_key], (list, tuple, np.ndarray)):
            # Try ms epoch first; if wrong, values become NaT (safe)
            df["datetime"] = pd.to_datetime(data[ts_key], unit="ms", errors="coerce")
        else:
            df["datetime"] = pd.NaT

        return df

    return pd.DataFrame()


# -------------------- Cached fetch --------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_intraday_cached(
    client_id: str,
    access_token: str,
    security_id: int,
    exchange_segment: str,
    instrument_type: str,
    interval_min: int,
    from_date: str,
    to_date: str
):
    dhan = get_dhan_client(client_id, access_token)
    return dhan.intraday_minute_data(
        security_id=str(security_id),
        exchange_segment=str(exchange_segment),
        instrument_type=str(instrument_type),
        from_date=from_date,
        to_date=to_date,
        interval=int(interval_min),
    )


# -------------------- Scan logic --------------------
def scan_symbol(
    client_id: str,
    access_token: str,
    symbol: str,
    security_id: int,
    exchange_segment: str,
    instrument_type: str,
    interval_min: int,
    from_date: str,
    to_date: str,
    adx_min: float,
    rsi_bull: float,
    rsi_bear: float,
    min_price: float,
    min_atr_pct: float,
    min_rel_vol: float,
):
    resp = fetch_intraday_cached(
        client_id, access_token,
        security_id, exchange_segment, instrument_type,
        interval_min, from_date, to_date
    )

    df = normalize_intraday(resp)
    if df.empty or len(df) < 120:
        return None

    # Ensure required columns exist
    needed_cols = {"open", "high", "low", "close", "volume"}
    if not needed_cols.issubset(df.columns):
        return None

    # Indicators
    df["dema100"] = dema(df["close"], 100)
    df["rsi14"] = rsi(df["close"], 14)
    df["atr14"] = atr(df["high"], df["low"], df["close"], 14)
    df["adx14"], df["+di"], df["-di"] = adx(df["high"], df["low"], df["close"], 14)

    last = df.iloc[-1]
    ltp = float(last["close"])
    if ltp < float(min_price):
        return None

    atr_pct = float(last["atr14"] / ltp * 100.0) if ltp > 0 else 0.0
    if atr_pct < float(min_atr_pct):
        return None

    # Today slice for VWAP & volume checks
    if "datetime" in df.columns and df["datetime"].notna().any():
        today = datetime.now(IST).date()
        df_today = df[df["datetime"].dt.date == today].copy()
    else:
        df_today = df.copy()

    if len(df_today) < 25:
        return None

    df_today["vwap"] = vwap_intraday(df_today)
    vwap_last = float(df_today["vwap"].iloc[-1])

    vol = df_today["volume"]
    vol_sma20 = float(vol.rolling(20).mean().iloc[-1])
    rel_vol = float(vol.iloc[-1] / vol_sma20) if vol_sma20 > 0 else 0.0

    volume_rising = (vol.iloc[-1] > vol.iloc[-2]) and (vol.iloc[-1] > vol_sma20) and (rel_vol >= float(min_rel_vol))
    if not volume_rising:
        return None

    rsi_v = float(last["rsi14"])
    adx_v = float(last["adx14"])
    plus_di = float(last["+di"])
    minus_di = float(last["-di"])
    di_gap = abs(plus_di - minus_di)

    if math.isnan(rsi_v) or math.isnan(adx_v) or math.isnan(plus_di) or math.isnan(minus_di):
        return None

    dema100 = float(last["dema100"])

    bull = (
        (rsi_v > float(rsi_bull)) and
        (adx_v >= float(adx_min)) and
        (plus_di > minus_di) and
        (ltp > dema100) and
        (ltp >= vwap_last)
    )

    bear = (
        (rsi_v < float(rsi_bear)) and
        (adx_v >= float(adx_min)) and
        (minus_di > plus_di) and
        (ltp < dema100) and
        (ltp <= vwap_last)
    )

    if not (bull or bear):
        return None

    side = "BULL" if bull else "BEAR"
    score = score_signal(side, adx_v, rsi_v, rel_vol, atr_pct, di_gap)

    return {
        "symbol": symbol,
        "security_id": int(security_id),
        "side": side,
        "close": float(ltp),
        "rsi14": float(rsi_v),
        "adx14": float(adx_v),
        "+di": float(plus_di),
        "-di": float(minus_di),
        "rel_vol": float(rel_vol),
        "atr_pct": float(atr_pct),
        "vwap": float(vwap_last),
        "dema100": float(dema100),
        "score": float(score),
    }


# -------------------- Notifications --------------------
def send_telegram(text: str):
    enabled = bool(sget(("telegram", "enabled"), False))
    token = sget(("telegram", "bot_token"), "")
    chat_id = sget(("telegram", "chat_id"), "")
    if not enabled or not token or not chat_id:
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "disable_web_page_preview": True}
    r = requests.post(url, json=payload, timeout=20)
    r.raise_for_status()


def format_message(bulls: pd.DataFrame, bears: pd.DataFrame, interval_min: int, top_n: int) -> str:
    now_ist = datetime.now(IST).strftime("%Y-%m-%d %H:%M IST")
    lines = [f"📈 Intraday Scanner ({now_ist}) | TF: {interval_min}m", ""]

    def section(title: str, df: pd.DataFrame):
        if df.empty:
            return [f"{title}: (none)", ""]
        out = [f"{title} (Top {min(top_n, len(df))}):"]
        for _, r in df.head(top_n).iterrows():
            out.append(
                f"• {r['symbol']} | Score {r['score']:.1f} | RSI {r['rsi14']:.1f} | ADX {r['adx14']:.1f} | RV {r['rel_vol']:.2f} | ATR% {r['atr_pct']:.2f}"
            )
        out.append("")
        return out

    lines += section("🟢 BULL", bulls)
    lines += section("🔴 BEAR", bears)
    return "\n".join(lines).strip()


# -------------------- Output directory (writable) --------------------
def get_out_dir():
    # Try repo folder first
    candidate = os.path.join(APP_DIR, "outputs")
    try:
        os.makedirs(candidate, exist_ok=True)
        testfile = os.path.join(candidate, ".write_test")
        with open(testfile, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(testfile)
        return candidate
    except Exception:
        # Fallback to /tmp
        candidate = os.path.join(tempfile.gettempdir(), "scanner_outputs")
        os.makedirs(candidate, exist_ok=True)
        return candidate


# -------------------- UI --------------------
st.title("F&O Intraday Scanner (DhanHQ + Streamlit)")

with st.sidebar:
    st.header("Scanner settings")

    # show installed version (useful on Streamlit Cloud)
    try:
        st.caption(f"dhanhq version: {ilm.version('dhanhq')}")
    except Exception:
        pass

    interval_min = int(sget(("scanner", "interval_min"), 5))
    interval_min = st.selectbox(
        "Timeframe (minutes)",
        [1, 5, 15, 25, 60],
        index=[1, 5, 15, 25, 60].index(interval_min) if interval_min in [1, 5, 15, 25, 60] else 1
    )

    adx_min = float(st.number_input("ADX minimum", value=float(sget(("scanner", "adx_min"), 25)), step=1.0))
    rsi_bull = float(st.number_input("RSI for BULL (>)", value=float(sget(("scanner", "rsi_bull"), 60)), step=1.0))
    rsi_bear = float(st.number_input("RSI for BEAR (<)", value=float(sget(("scanner", "rsi_bear"), 40)), step=1.0))

    min_price = float(st.number_input("Min price (₹)", value=float(sget(("scanner", "min_price"), 80)), step=10.0))
    min_atr_pct = float(st.number_input("Min ATR% (14)", value=float(sget(("scanner", "min_atr_pct"), 1.0)), step=0.1))
    min_rel_vol = float(st.number_input("Min Relative Volume (vs SMA20)", value=float(sget(("scanner", "min_rel_vol"), 1.15)), step=0.05))

    top_n = int(st.number_input("Top N in message", value=int(sget(("scanner", "top_n"), 12)), step=1))
    max_symbols = int(st.number_input("Max symbols to scan", value=int(sget(("scanner", "max_symbols"), 207)), step=10))

    max_workers = int(sget(("scanner", "max_workers"), 6))
    max_workers = st.slider("Max workers", 1, 12, max_workers)

    st.divider()
    st.subheader("Auto-run (only if app is open)")
    auto_run = st.checkbox("Auto-run once after 10:15 IST", value=False)
    auto_refresh = st.checkbox("Auto-refresh page (every 60s)", value=False)
    if auto_refresh and st_autorefresh:
        st_autorefresh(interval=60_000, key="refresh_60s")

# Credentials
client_id = sget(("dhan", "client_id"), "")
access_token = sget(("dhan", "access_token"), "")
if not client_id or not access_token:
    st.error("Missing Dhan credentials in secrets. Add [dhan].client_id and [dhan].access_token.")
    st.stop()

# Watchlist input: default repo CSV + optional upload
uploaded = st.file_uploader("Optional: upload a watchlist CSV (otherwise uses repo stock_watchlist.csv)", type=["csv"])
if uploaded:
    watch = pd.read_csv(uploaded)
else:
    watch_path = os.path.join(APP_DIR, "stock_watchlist.csv")
    watch = pd.read_csv(watch_path)

needed = {"SEM_TRADING_SYMBOL", "SEM_SMST_SECURITY_ID"}
if not needed.issubset(set(watch.columns)):
    st.error(f"Watchlist CSV must contain at least columns: {sorted(list(needed))}")
    st.stop()

watch = watch.head(max_symbols).copy()

# Exchange mapping (can be set in secrets.toml if you want)
EX_EQ = sget(("dhan_map", "exchange_segment_equity"), "NSE_EQ")
IN_EQ = sget(("dhan_map", "instrument_type_equity"), "EQUITY")

# Date range (intraday endpoint usually returns limited recent days)
today = datetime.now(IST).date()
from_date = (today - timedelta(days=7)).strftime("%Y-%m-%d")
to_date = (today + timedelta(days=1)).strftime("%Y-%m-%d")


def run_scan():
    progress = st.progress(0)
    status = st.empty()

    rows = watch.to_dict("records")
    results = []
    total = len(rows)

    errors = 0

    def work(r):
        symbol = str(r["SEM_TRADING_SYMBOL"]).strip()
        sec_id = int(r["SEM_SMST_SECURITY_ID"])
        return scan_symbol(
            client_id, access_token,
            symbol, sec_id, EX_EQ, IN_EQ,
            interval_min, from_date, to_date,
            adx_min, rsi_bull, rsi_bear,
            min_price, min_atr_pct, min_rel_vol
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(work, r) for r in rows]
        done = 0
        for fut in as_completed(futs):
            done += 1
            if done % 5 == 0 or done == total:
                progress.progress(done / total)
                status.write(f"Scanned {done}/{total} ...")

            try:
                out = fut.result()
                if out:
                    results.append(out)
            except Exception:
                errors += 1

    progress.empty()
    status.empty()
    return pd.DataFrame(results), errors


# Auto-run logic (only works if app is open)
if auto_run:
    now = datetime.now(IST)
    key = "last_auto_run_date"
    last_date = st.session_state.get(key)
    if now.time() >= dtime(10, 15) and last_date != str(today):
        st.session_state[key] = str(today)
        st.info("Auto-running scan (after 10:15 IST)...")
        with st.spinner("Fetching candles + computing signals..."):
            st.session_state["results_df"], st.session_state["errors"] = run_scan()

colA, colB, colC = st.columns([1, 1, 2])
with colA:
    run_btn = st.button("🚀 Run scan now", type="primary")
with colB:
    clear_btn = st.button("🧹 Clear cache/results")
with colC:
    st.caption("No auto-orders. Message-only notifications.")

if clear_btn:
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state.pop("results_df", None)
    st.session_state.pop("errors", None)
    st.success("Cleared cache and results. Reload if needed.")

if run_btn:
    with st.spinner("Fetching candles + computing signals..."):
        st.session_state["results_df"], st.session_state["errors"] = run_scan()

res = st.session_state.get("results_df")
if res is None:
    st.stop()

errors = st.session_state.get("errors", 0)
if errors:
    st.caption(f"Skipped {errors} symbols due to API / data issues (normal).")

if res.empty:
    st.warning("No matches found with current filters.")
    st.stop()

# Split lists
bulls = res[res["side"] == "BULL"].sort_values("score", ascending=False)
bears = res[res["side"] == "BEAR"].sort_values("score", ascending=False)

# Save results_YYYYMMDD.csv
out_dir = get_out_dir()
out_file = os.path.join(out_dir, f"results_{today.strftime('%Y%m%d')}.csv")
res.sort_values(["side", "score"], ascending=[True, False]).to_csv(out_file, index=False)

st.success(f"Saved: {out_file}")

# Display
m1, m2, m3 = st.columns(3)
m1.metric("BULL matches", len(bulls))
m2.metric("BEAR matches", len(bears))
m3.metric("Total matches", len(res))

tab1, tab2, tab3 = st.tabs(["🟢 BULL", "🔴 BEAR", "📋 All"])
with tab1:
    st.dataframe(bulls.reset_index(drop=True), use_container_width=True)
with tab2:
    st.dataframe(bears.reset_index(drop=True), use_container_width=True)
with tab3:
    st.dataframe(res.sort_values("score", ascending=False).reset_index(drop=True), use_container_width=True)

# Download button
csv_bytes = res.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download CSV",
    data=csv_bytes,
    file_name=f"results_{today.strftime('%Y%m%d')}.csv",
    mime="text/csv"
)

# Message preview + send
st.subheader("📣 Notification")
msg = format_message(bulls, bears, interval_min, top_n)
st.text_area("Message preview", value=msg, height=220)

send_col1, send_col2 = st.columns([1, 2])
with send_col1:
    send_now = st.button("Send to Telegram")
with send_col2:
    test_short = st.button("Send TEST (Telegram)")

if test_short:
    test_msg = f"✅ Test message ({datetime.now(IST).strftime('%Y-%m-%d %H:%M IST')})"
    try:
        send_telegram(test_msg)
        st.success("Test message sent.")
    except Exception as e:
        st.error(f"Send failed: {e}")

if send_now:
    try:
        send_telegram(msg)
        st.success("Sent Telegram message.")
    except Exception as e:
        st.error(f"Send failed: {e}")
