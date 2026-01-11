import os
import io
import math
import requests
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional auto-refresh (helps if you keep the tab open)
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

# DhanHQ v2 style (DhanContext + dhanhq(ctx))
from dhanhq import dhanhq
try:
    from dhanhq.dhan_context import DhanContext
except Exception:
    # fallback (some versions may export differently)
    from dhanhq import DhanContext  # type: ignore

IST = ZoneInfo("Asia/Kolkata")
APP_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="F&O Intraday Scanner", layout="wide")

# -------------------- Helpers: secrets --------------------
def sget(path, default=None):
    """
    Safe getter for nested st.secrets keys.
    Example: sget(("scanner","interval_min"), 5)
    """
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
    ctx = DhanContext(client_id, access_token)
    return dhanhq(ctx)

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

def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))

def score_signal(side: str, adx_v: float, rsi_v: float, rel_vol: float, atr_pct: float, di_gap: float) -> float:
    # 0..100 “neat” score (higher = better setup)
    adx_s = clamp((adx_v - 20.0) / 30.0)          # ADX 20..50
    rv_s  = clamp((rel_vol - 1.0) / 2.0)          # RVOL 1..3
    atr_s = clamp((atr_pct - 0.8) / 2.2)          # ATR% 0.8..3.0-ish
    di_s  = clamp(di_gap / 25.0)                  # DI gap 0..25

    if side == "BULL":
        rsi_s = clamp((rsi_v - 50.0) / 30.0)      # RSI 50..80
    else:
        rsi_s = clamp((50.0 - rsi_v) / 30.0)      # RSI 50..20

    score = 100.0 * (0.30 * adx_s + 0.25 * rsi_s + 0.25 * rv_s + 0.10 * di_s + 0.10 * atr_s)
    return float(score)

# -------------------- Dhan candles normalization --------------------
def epoch_to_dt(epoch_val: int) -> datetime:
    # Handle seconds vs ms
    if epoch_val > 10_000_000_000:  # likely ms
        epoch_val = int(epoch_val / 1000)
    return datetime.fromtimestamp(int(epoch_val), tz=IST)

def normalize_intraday_response(resp) -> pd.DataFrame:
    """
    Supports common shapes:
    - dict with arrays: open/high/low/close/volume/start_Time
    - dict with "data": list[dict]
    - already list[dict]
    """
    if resp is None:
        return pd.DataFrame()

    if isinstance(resp, dict) and "data" in resp and isinstance(resp["data"], list):
        df = pd.DataFrame(resp["data"])
    elif isinstance(resp, list):
        df = pd.DataFrame(resp)
    elif isinstance(resp, dict) and all(k in resp for k in ["open", "high", "low", "close", "volume"]):
        # arrays style
        time_key = "start_Time" if "start_Time" in resp else ("startTime" if "startTime" in resp else None)
        n = len(resp["close"])
        df = pd.DataFrame({
            "open": resp["open"][:n],
            "high": resp["high"][:n],
            "low": resp["low"][:n],
            "close": resp["close"][:n],
            "volume": resp["volume"][:n],
            "ts": resp[time_key][:n] if time_key else [None]*n
        })
    else:
        return pd.DataFrame()

    # Try to map columns
    col_map = {}
    for a, b in [("o","open"),("h","high"),("l","low"),("c","close"),("v","volume")]:
        if a in df.columns and b not in df.columns:
            col_map[a] = b
    df = df.rename(columns=col_map)

    # Timestamp column detection
    ts_col = None
    for c in ["ts","start_Time","startTime","timestamp","time","t"]:
        if c in df.columns:
            ts_col = c
            break

    if ts_col:
        df["datetime"] = df[ts_col].apply(lambda x: epoch_to_dt(int(x)) if pd.notna(x) else pd.NaT)
    else:
        df["datetime"] = pd.NaT

    # Ensure numeric
    for c in ["open","high","low","close","volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["open","high","low","close","volume"]).copy()
    df = df.sort_values("datetime" if df["datetime"].notna().any() else df.index).reset_index(drop=True)
    return df

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
    # intraday_minute_data supports last 5 trading days (with intervals 1/5/15/25/60) :contentReference[oaicite:3]{index=3}
    resp = dhan.intraday_minute_data(
        security_id=str(security_id),
        exchange_segment=str(exchange_segment),
        instrument_type=str(instrument_type),
        from_date=from_date,
        to_date=to_date,
        interval=int(interval_min),
    )
    return resp

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
    df = normalize_intraday_response(resp)
    if df.empty or len(df) < 120:
        return None

    # Indicators on full series
    df["dema100"] = dema(df["close"], 100)
    df["rsi14"] = rsi(df["close"], 14)
    df["atr14"] = atr(df["high"], df["low"], df["close"], 14)
    df["adx14"], df["+di"], df["-di"] = adx(df["high"], df["low"], df["close"], 14)

    last = df.iloc[-1]
    ltp = float(last["close"])
    if ltp < float(min_price):
        return None

    atr_pct = float(last["atr14"] / last["close"] * 100.0) if float(last["close"]) > 0 else 0.0
    if atr_pct < float(min_atr_pct):
        return None

    # Intraday-only VWAP and volume rising (today IST)
    if df["datetime"].notna().any():
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
        "close": ltp,
        "rsi14": rsi_v,
        "adx14": adx_v,
        "+di": plus_di,
        "-di": minus_di,
        "rel_vol": rel_vol,
        "atr_pct": atr_pct,
        "vwap": vwap_last,
        "dema100": dema100,
        "score": score,
    }

# -------------------- Notifications --------------------
def send_telegram(text: str):
    enabled = bool(sget(("telegram","enabled"), False))
    token = sget(("telegram","bot_token"), "")
    chat_id = sget(("telegram","chat_id"), "")
    if not enabled or not token or not chat_id:
        return

    # Bot API is HTTP-based; sendMessage is the standard method :contentReference[oaicite:4]{index=4}
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "disable_web_page_preview": True}
    r = requests.post(url, json=payload, timeout=20)
    r.raise_for_status()

def send_whatsapp_twilio(text: str):
    enabled = bool(sget(("twilio","enabled"), False))
    if not enabled:
        return
    sid = sget(("twilio","account_sid"), "")
    auth = sget(("twilio","auth_token"), "")
    w_from = sget(("twilio","whatsapp_from"), "")
    w_to = sget(("twilio","whatsapp_to"), "")
    if not sid or not auth or not w_from or not w_to:
        return

    # Twilio Message resource supports WhatsApp using "whatsapp:<E.164>" addressing :contentReference[oaicite:5]{index=5}
    url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
    from requests.auth import HTTPBasicAuth
    data = {"From": w_from, "To": w_to, "Body": text}
    r = requests.post(url, data=data, auth=HTTPBasicAuth(sid, auth), timeout=20)
    r.raise_for_status()

def send_whatsapp_cloud(text: str):
    enabled = bool(sget(("wa_cloud","enabled"), False))
    if not enabled:
        return
    phone_id = sget(("wa_cloud","phone_number_id"), "")
    token = sget(("wa_cloud","access_token"), "")
    to_phone = sget(("wa_cloud","to_phone"), "")
    version = sget(("wa_cloud","version"), "v19.0")
    if not phone_id or not token or not to_phone:
        return

    # WhatsApp Cloud API uses POST /{Phone-Number-ID}/messages :contentReference[oaicite:6]{index=6}
    url = f"https://graph.facebook.com/{version}/{phone_id}/messages"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "messaging_product": "whatsapp",
        "to": str(to_phone),
        "type": "text",
        "text": {"body": text},
    }
    r = requests.post(url, headers=headers, json=payload, timeout=20)
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

# -------------------- UI --------------------
st.title("F&O Intraday Scanner (DhanHQ + Streamlit)")

with st.sidebar:
    st.header("Scanner settings")

    interval_min = int(sget(("scanner","interval_min"), 5))
    interval_min = st.selectbox("Timeframe (minutes)", [1, 5, 15, 25, 60], index=[1,5,15,25,60].index(interval_min) if interval_min in [1,5,15,25,60] else 1)

    adx_min = float(st.number_input("ADX minimum", value=float(sget(("scanner","adx_min"), 25)), step=1.0))
    rsi_bull = float(st.number_input("RSI for BULL (>)", value=float(sget(("scanner","rsi_bull"), 60)), step=1.0))
    rsi_bear = float(st.number_input("RSI for BEAR (<)", value=float(sget(("scanner","rsi_bear"), 40)), step=1.0))

    min_price = float(st.number_input("Min price (₹)", value=float(sget(("scanner","min_price"), 80)), step=10.0))
    min_atr_pct = float(st.number_input("Min ATR% (14)", value=float(sget(("scanner","min_atr_pct"), 1.0)), step=0.1))
    min_rel_vol = float(st.number_input("Min Relative Volume (vs SMA20)", value=float(sget(("scanner","min_rel_vol"), 1.15)), step=0.05))

    top_n = int(st.number_input("Top N in message", value=int(sget(("scanner","top_n"), 12)), step=1))
    max_symbols = int(st.number_input("Max symbols to scan", value=int(sget(("scanner","max_symbols"), 207)), step=10))

    use_threads = bool(sget(("scanner","use_threads"), True))
    use_threads = st.checkbox("Use threading (faster)", value=use_threads)
    max_workers = int(sget(("scanner","max_workers"), 6))
    max_workers = st.slider("Max workers", 1, 12, max_workers)

    st.divider()
    st.subheader("Auto-run (only if app is open)")
    auto_run = st.checkbox("Auto-run once after 10:15 IST", value=False)
    auto_refresh = st.checkbox("Auto-refresh page (every 60s)", value=False)
    if auto_refresh and st_autorefresh:
        st_autorefresh(interval=60_000, key="refresh_60s")

# Credentials
client_id = sget(("dhan","client_id"), "")
access_token = sget(("dhan","access_token"), "")
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

# Basic validation (expected columns from your file)
needed = {"SEM_TRADING_SYMBOL", "SEM_SMST_SECURITY_ID", "SEM_EXM_EXCH_ID", "SEM_SEGMENT"}
if not needed.issubset(set(watch.columns)):
    st.error(f"Watchlist CSV must contain columns: {sorted(list(needed))}")
    st.stop()

watch = watch.head(max_symbols).copy()

# Exchange mapping (your file uses NSE + EQUITY)
EX_EQ = sget(("dhan_map","exchange_segment_equity"), "NSE_EQ")
IN_EQ = sget(("dhan_map","instrument_type_equity"), "EQUITY")

def map_exchange(row):
    # For your current CSV: NSE + EQUITY -> NSE_EQ + EQUITY
    return EX_EQ, IN_EQ

# Date range (intraday endpoint only provides last ~5 trading days) :contentReference[oaicite:7]{index=7}
today = datetime.now(IST).date()
from_date = (today - timedelta(days=7)).strftime("%Y-%m-%d")
to_date = (today + timedelta(days=1)).strftime("%Y-%m-%d")  # non-inclusive in some APIs

def run_scan():
    progress = st.progress(0)
    status = st.empty()

    rows = watch.to_dict("records")
    results = []
    total = len(rows)

    def work(r):
        symbol = str(r["SEM_TRADING_SYMBOL"]).strip()
        sec_id = int(r["SEM_SMST_SECURITY_ID"])
        ex_seg, inst_type = map_exchange(r)
        return scan_symbol(
            client_id, access_token,
            symbol, sec_id, ex_seg, inst_type,
            interval_min, from_date, to_date,
            adx_min, rsi_bull, rsi_bear,
            min_price, min_atr_pct, min_rel_vol
        )

    if use_threads:
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
                    pass
    else:
        for i, r in enumerate(rows, start=1):
            status.write(f"Scanning {i}/{total}: {r['SEM_TRADING_SYMBOL']}")
            try:
                out = work(r)
                if out:
                    results.append(out)
            except Exception:
                pass
            progress.progress(i / total)

    progress.empty()
    status.empty()
    return pd.DataFrame(results)

# Auto-run logic (only works if app is open)
if auto_run:
    now = datetime.now(IST)
    key = "last_auto_run_date"
    last_date = st.session_state.get(key)
    if now.time() >= dtime(10, 15) and last_date != str(today):
        st.session_state[key] = str(today)
        st.info("Auto-running scan (after 10:15 IST)...")
        st.session_state["results_df"] = run_scan()

colA, colB, colC = st.columns([1,1,2])
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
    st.success("Cleared cache and results. Reload if needed.")

if run_btn:
    with st.spinner("Fetching candles + computing signals..."):
        st.session_state["results_df"] = run_scan()

res = st.session_state.get("results_df")
if res is None:
    st.stop()

if res.empty:
    st.warning("No matches found with current filters.")
    st.stop()

# Split lists
bulls = res[res["side"] == "BULL"].sort_values("score", ascending=False)
bears = res[res["side"] == "BEAR"].sort_values("score", ascending=False)

# Save results_YYYYMMDD.csv
os.makedirs(os.path.join(APP_DIR, "outputs"), exist_ok=True)
out_file = os.path.join(APP_DIR, "outputs", f"results_{today.strftime('%Y%m%d')}.csv")
res.sort_values(["side","score"], ascending=[True, False]).to_csv(out_file, index=False)

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

send_col1, send_col2, send_col3 = st.columns([1,1,2])
with send_col1:
    send_now = st.button("Send to Telegram/WhatsApp")
with send_col2:
    test_short = st.button("Send TEST (short)")

if test_short:
    test_msg = f"✅ Test message ({datetime.now(IST).strftime('%Y-%m-%d %H:%M IST')})"
    try:
        send_telegram(test_msg)
        send_whatsapp_twilio(test_msg)
        send_whatsapp_cloud(test_msg)
        st.success("Test message sent (where enabled).")
    except Exception as e:
        st.error(f"Send failed: {e}")

if send_now:
    try:
        send_telegram(msg)
        send_whatsapp_twilio(msg)
        send_whatsapp_cloud(msg)
        st.success("Sent notifications (where enabled).")
    except Exception as e:
        st.error(f"Send failed: {e}")
