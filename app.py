# =========================
# app.py – Streamlit Cloud
# =========================

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

# DhanHQ (NO DhanContext – works on Streamlit Cloud)
from dhanhq import dhanhq
import importlib.metadata as ilm

# -------------------- Constants --------------------
IST = ZoneInfo("Asia/Kolkata")
APP_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="F&O Intraday Scanner", layout="wide")

# -------------------- Secrets helper --------------------
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
    # IMPORTANT: Do NOT use DhanContext
    return dhanhq(client_id, access_token)

# -------------------- Indicators --------------------
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def dema(series, period):
    e1 = ema(series, period)
    e2 = ema(e1, period)
    return 2 * e1 - e2

def rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def true_range(high, low, close):
    prev_close = close.shift(1)
    return pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1
    ).max(axis=1)

def atr(high, low, close, period=14):
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def adx(high, low, close, period=14):
    up = high.diff()
    down = -low.diff()

    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    tr = true_range(high, low, close)
    atr_ = tr.ewm(alpha=1/period, adjust=False).mean()

    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean() / atr_
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean() / atr_

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_ = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx_, plus_di, minus_di

def vwap(df):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    return (tp * df["volume"]).cumsum() / df["volume"].cumsum()

# -------------------- Scoring --------------------
def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))

def neat_score(side, adx_v, rsi_v, rel_vol, atr_pct, di_gap):
    adx_s = clamp((adx_v - 20) / 30)
    rv_s  = clamp((rel_vol - 1) / 2)
    atr_s = clamp((atr_pct - 0.8) / 2.2)
    di_s  = clamp(di_gap / 25)

    if side == "BULL":
        rsi_s = clamp((rsi_v - 50) / 30)
    else:
        rsi_s = clamp((50 - rsi_v) / 30)

    return 100 * (0.30*adx_s + 0.25*rsi_s + 0.25*rv_s + 0.10*di_s + 0.10*atr_s)

# -------------------- Normalize Dhan candles --------------------
def epoch_to_dt(x):
    if x > 10_000_000_000:
        x //= 1000
    return datetime.fromtimestamp(int(x), tz=IST)

def normalize_intraday(resp):
    if not resp:
        return pd.DataFrame()

    if isinstance(resp, dict) and "data" in resp:
        df = pd.DataFrame(resp["data"])
    elif isinstance(resp, list):
        df = pd.DataFrame(resp)
    else:
        return pd.DataFrame()

    rename = {"o":"open","h":"high","l":"low","c":"close","v":"volume"}
    df = df.rename(columns=rename)

    ts_col = next((c for c in ["ts","timestamp","time","start_Time","startTime"] if c in df.columns), None)
    if ts_col:
        df["datetime"] = df[ts_col].apply(lambda x: epoch_to_dt(x) if pd.notna(x) else pd.NaT)
    else:
        df["datetime"] = pd.NaT

    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["open","high","low","close","volume"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df

# -------------------- Cached fetch --------------------
@st.cache_data(ttl=300)
def fetch_intraday(client_id, access_token, security_id, exchange_segment, instrument_type,
                   interval_min, from_date, to_date):
    dhan = get_dhan_client(client_id, access_token)
    return dhan.intraday_minute_data(
        security_id=str(security_id),
        exchange_segment=exchange_segment,
        instrument_type=instrument_type,
        from_date=from_date,
        to_date=to_date,
        interval=int(interval_min),
    )

# -------------------- Scan one symbol --------------------
def scan_symbol(params):
    (client_id, access_token, row, cfg, from_date, to_date) = params
    symbol = row["SEM_TRADING_SYMBOL"]
    sec_id = int(row["SEM_SMST_SECURITY_ID"])

    resp = fetch_intraday(
        client_id, access_token, sec_id,
        cfg["exchange_segment"], cfg["instrument_type"],
        cfg["interval"], from_date, to_date
    )
    df = normalize_intraday(resp)
    if len(df) < 120:
        return None

    df["dema100"] = dema(df["close"], 100)
    df["rsi14"] = rsi(df["close"], 14)
    df["atr14"] = atr(df["high"], df["low"], df["close"], 14)
    df["adx14"], df["+di"], df["-di"] = adx(df["high"], df["low"], df["close"], 14)

    last = df.iloc[-1]
    ltp = float(last["close"])
    atr_pct = float(last["atr14"] / ltp * 100)

    if ltp < cfg["min_price"] or atr_pct < cfg["min_atr_pct"]:
        return None

    today_df = df[df["datetime"].dt.date == datetime.now(IST).date()]
    if len(today_df) < 25:
        return None

    today_df["vwap"] = vwap(today_df)
    vol = today_df["volume"]
    rel_vol = vol.iloc[-1] / vol.rolling(20).mean().iloc[-1]

    if rel_vol < cfg["min_rel_vol"] or vol.iloc[-1] <= vol.iloc[-2]:
        return None

    rsi_v = float(last["rsi14"])
    adx_v = float(last["adx14"])
    pdi = float(last["+di"])
    mdi = float(last["-di"])

    bull = (rsi_v > cfg["rsi_bull"] and adx_v >= cfg["adx_min"] and pdi > mdi and
            ltp > last["dema100"] and ltp >= today_df["vwap"].iloc[-1])

    bear = (rsi_v < cfg["rsi_bear"] and adx_v >= cfg["adx_min"] and mdi > pdi and
            ltp < last["dema100"] and ltp <= today_df["vwap"].iloc[-1])

    if not (bull or bear):
        return None

    side = "BULL" if bull else "BEAR"
    score = neat_score(side, adx_v, rsi_v, rel_vol, atr_pct, abs(pdi-mdi))

    return {
        "symbol": symbol,
        "side": side,
        "close": round(ltp,2),
        "rsi": round(rsi_v,1),
        "adx": round(adx_v,1),
        "rel_vol": round(rel_vol,2),
        "atr_pct": round(atr_pct,2),
        "score": round(score,1),
    }

# -------------------- Messaging --------------------
def send_telegram(text):
    if not sget(("telegram","enabled"), False):
        return
    token = sget(("telegram","bot_token"))
    chat_id = sget(("telegram","chat_id"))
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=20)

# -------------------- UI --------------------
st.title("📈 F&O Intraday Momentum Scanner")

with st.sidebar:
    st.caption(f"dhanhq version: {ilm.version('dhanhq')}")
    interval = st.selectbox("Timeframe (min)", [5,15], index=0)
    adx_min = st.number_input("ADX ≥", 10, 50, 25)
    rsi_bull = st.number_input("RSI Bull >", 50, 80, 60)
    rsi_bear = st.number_input("RSI Bear <", 20, 50, 40)
    min_price = st.number_input("Min Price", 10, 500, 80)
    min_atr_pct = st.number_input("Min ATR %", 0.5, 5.0, 1.0)
    min_rel_vol = st.number_input("Min Relative Volume", 1.0, 3.0, 1.15)
    max_workers = st.slider("Threads", 1, 10, 5)

# Secrets
client_id = sget(("dhan","client_id"))
access_token = sget(("dhan","access_token"))
if not client_id or not access_token:
    st.error("Missing Dhan credentials in secrets")
    st.stop()

# Watchlist
watch = pd.read_csv("stock_watchlist.csv")

cfg = {
    "interval": interval,
    "adx_min": adx_min,
    "rsi_bull": rsi_bull,
    "rsi_bear": rsi_bear,
    "min_price": min_price,
    "min_atr_pct": min_atr_pct,
    "min_rel_vol": min_rel_vol,
    "exchange_segment": "NSE_EQ",
    "instrument_type": "EQUITY",
}

today = datetime.now(IST).date()
from_date = (today - timedelta(days=5)).strftime("%Y-%m-%d")
to_date = today.strftime("%Y-%m-%d")

if st.button("🚀 Run Scanner"):
    params = [(client_id, access_token, r, cfg, from_date, to_date) for _, r in watch.iterrows()]
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for f in as_completed([ex.submit(scan_symbol, p) for p in params]):
            out = f.result()
            if out:
                results.append(out)

    df = pd.DataFrame(results)
    if df.empty:
        st.warning("No matches today")
        st.stop()

    bulls = df[df.side=="BULL"].sort_values("score", ascending=False)
    bears = df[df.side=="BEAR"].sort_values("score", ascending=False)

    st.subheader("🟢 BULL")
    st.dataframe(bulls, use_container_width=True)

    st.subheader("🔴 BEAR")
    st.dataframe(bears, use_container_width=True)

    # Save CSV (ephemeral storage)
    out_dir = os.path.join(tempfile.gettempdir(), "scanner_outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"results_{today.strftime('%Y%m%d')}.csv")
    df.to_csv(out_file, index=False)

    st.success(f"Saved: {out_file}")
    st.download_button("⬇️ Download CSV", df.to_csv(index=False), "results.csv")

    msg = f"Intraday Scanner {today}\n\nBULL:\n" + \
          "\n".join([f"{r.symbol} ({r.score})" for r in bulls.head(10).itertuples()]) + \
          "\n\nBEAR:\n" + \
          "\n".join([f"{r.symbol} ({r.score})" for r in bears.head(10).itertuples()])

    if st.button("Send Telegram Alert"):
        send_telegram(msg)
        st.success("Telegram sent")
