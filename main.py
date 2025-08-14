# main.py  —  Binance Futures Autotrade (EMA+RSI+MACD+Volume+KDJ) + Telegram control
# Replit/VPS ready — single file
#
# Env vars (.env):
# BINANCE_API_KEY=...
# BINANCE_API_SECRET=...
# TELEGRAM_BOT_TOKEN=...
# TELEGRAM_CHAT_ID=...
# TRADING_ENABLED=true
# INTERVAL=5m
# SCAN_SECONDS=30
# LEVERAGE=5
# POSITION_SIZE_PCT=3.0
# MAX_OPEN_POSITIONS=4
# MIN_QUOTE_VOL_24H=5000000
# TP_PCT=1.5
# SL_PCT=0.8
# USE_TRAILING=true
# TRAILING_PCT=1.0
# RSI_PERIOD=14
# RSI_BUY_MAX=65
# RSI_SELL_MIN=35
# EMA_FAST=20
# EMA_MID=50
# EMA_SLOW=200
# MACD_FAST=12
# MACD_SLOW=26
# MACD_SIGNAL=9
# KDJ_PERIOD=9
# KDJ_SIGNAL=3
# VOL_MULTIPLIER=1.8
#
# Run:
#   pip install requests pandas numpy python-dotenv
#   python main.py

import os
import time
import hmac
import math
import hashlib
import urllib.parse
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# ==========================
# Env & Config
# ==========================
load_dotenv()

API_KEY  = os.getenv("BINANCE_API_KEY", "")
API_SEC  = os.getenv("BINANCE_API_SECRET", "")
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT  = os.getenv("TELEGRAM_CHAT_ID", "")
BASE_URL = "https://fapi.binance.com"

TRADING_ENABLED = os.getenv("TRADING_ENABLED", "true").lower() == "true"
INTERVAL        = os.getenv("INTERVAL", "5m")
SCAN_SECONDS    = int(float(os.getenv("SCAN_SECONDS", "30")))
LEVERAGE        = int(float(os.getenv("LEVERAGE", "5")))
POSITION_SIZE_PCT = float(os.getenv("POSITION_SIZE_PCT", "3.0"))
MAX_OPEN_POSITIONS = int(float(os.getenv("MAX_OPEN_POSITIONS", "4")))
MIN_QUOTE_VOL_24H  = float(os.getenv("MIN_QUOTE_VOL_24H", "5000000"))

TP_PCT        = float(os.getenv("TP_PCT", "1.5"))
SL_PCT        = float(os.getenv("SL_PCT", "0.8"))
USE_TRAILING  = os.getenv("USE_TRAILING", "true").lower() == "true"
TRAILING_PCT  = float(os.getenv("TRAILING_PCT", "1.0"))

RSI_PERIOD    = int(float(os.getenv("RSI_PERIOD", "14")))
RSI_BUY_MAX   = float(os.getenv("RSI_BUY_MAX", "65"))
RSI_SELL_MIN  = float(os.getenv("RSI_SELL_MIN", "35"))

EMA_FAST      = int(float(os.getenv("EMA_FAST", "20")))
EMA_MID       = int(float(os.getenv("EMA_MID", "50")))
EMA_SLOW      = int(float(os.getenv("EMA_SLOW", "200")))

MACD_FAST     = int(float(os.getenv("MACD_FAST", "12")))
MACD_SLOW     = int(float(os.getenv("MACD_SLOW", "26")))
MACD_SIGNAL   = int(float(os.getenv("MACD_SIGNAL", "9")))

KDJ_PERIOD    = int(float(os.getenv("KDJ_PERIOD", "9")))
KDJ_SIGNAL    = int(float(os.getenv("KDJ_SIGNAL", "3")))

VOL_MULTIPLIER = float(os.getenv("VOL_MULTIPLIER", "1.8"))

# ==========================
# Telegram
# ==========================
class Telegram:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self._offset = None

    def _url(self, method: str):
        return f"https://api.telegram.org/bot{self.token}/{method}"

    def send(self, text: str):
        if not (self.token and self.chat_id):
            return
        try:
            requests.post(self._url("sendMessage"), data={
                "chat_id": self.chat_id,
                "text": text,
                "disable_web_page_preview": "true"
            }, timeout=10)
        except Exception as e:
            print("Telegram send error:", e)

    def poll(self):
        if not self.token:
            return []
        try:
            params = {"timeout": 0}
            if self._offset is not None:
                params["offset"] = self._offset + 1
            r = requests.get(self._url("getUpdates"), params=params, timeout=10)
            r.raise_for_status()
            out = []
            for upd in r.json().get("result", []):
                self._offset = upd["update_id"]
                msg = upd.get("message") or {}
                text = (msg.get("text") or "").strip()
                if text.startswith("/"):
                    parts = text.split(maxsplit=1)
                    cmd = parts[0].lower()
                    args = parts[1] if len(parts) > 1 else ""
                    out.append((cmd, args, str(msg.get("chat", {}).get("id"))))
            return out
        except Exception as e:
            return []

tg = Telegram(TG_TOKEN, TG_CHAT)

def log(msg: str):
    print(msg, flush=True)
    tg.send(msg)

# ==========================
# Binance REST helper
# ==========================
def headers():
    return {"X-MBX-APIKEY": API_KEY}

def sign_qs(params: dict):
    query = urllib.parse.urlencode(params, doseq=True)
    sig = hmac.new(API_SEC.encode(), query.encode(), hashlib.sha256).hexdigest()
    return query + f"&signature={sig}"

def b_get(path, params=None, signed=False):
    url = BASE_URL + path
    params = params or {}
    if signed:
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = 5000
        url = url + "?" + sign_qs(params)
        params = None
    r = requests.get(url, headers=headers(), params=params, timeout=15)
    r.raise_for_status()
    return r.json()

def b_post(path, params=None, signed=True):
    url = BASE_URL + path
    params = params or {}
    if signed:
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = 5000
        body = sign_qs(params)
        r = requests.post(url, headers=headers(), data=body, timeout=15)
    else:
        r = requests.post(url, headers=headers(), data=params, timeout=15)
    r.raise_for_status()
    return r.json()

def b_delete(path, params=None, signed=True):
    url = BASE_URL + path
    params = params or {}
    if signed:
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = 5000
        body = sign_qs(params)
        r = requests.delete(url, headers=headers(), data=body, timeout=15)
    else:
        r = requests.delete(url, headers=headers(), data=params, timeout=15)
    r.raise_for_status()
    return r.json()

# ==========================
# Market data
# ==========================
def klines(symbol, interval=INTERVAL, limit=400):
    data = b_get("/fapi/v1/klines", {"symbol": symbol, "interval": interval, "limit": limit})
    df = pd.DataFrame(data, columns=[
        "open_time","open","high","low","close","volume","close_time",
        "qav","num_trades","taker_base","taker_quote","ignore"
    ])
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["open","high","low","close","volume"]]

def ticker_24h(symbol=None):
    params = {"symbol": symbol} if symbol else None
    return b_get("/fapi/v1/ticker/24hr", params=params)

def exchange_info():
    return b_get("/fapi/v1/exchangeInfo")

# ==========================
# Account / Trading
# ==========================
def balance_usdt():
    try:
        bals = b_get("/fapi/v2/balance", signed=True)
        for b in bals:
            if b.get("asset") == "USDT":
                return float(b.get("availableBalance", b.get("balance", 0.0)))
    except Exception as e:
        pass
    return 0.0

def positions_open_map():
    mp = {}
    try:
        positions = b_get("/fapi/v2/positionRisk", signed=True)
        for p in positions:
            amt = float(p.get("positionAmt", "0"))
            if amt != 0.0:
                mp[p["symbol"]] = p
    except Exception as e:
        pass
    return mp

def change_leverage(symbol, lev):
    try:
        b_post("/fapi/v1/leverage", {"symbol": symbol, "leverage": lev})
    except Exception:
        pass

def order_market(symbol, side, qty):
    return b_post("/fapi/v1/order", {"symbol": symbol, "side": side, "type": "MARKET", "quantity": qty})

def cancel_all(symbol):
    try:
        b_delete("/fapi/v1/allOpenOrders", {"symbol": symbol})
    except Exception:
        pass

def place_exits(symbol, side, entry_price, qty):
    # Percent → decimal
    tp = TP_PCT / 100.0
    sl = SL_PCT / 100.0
    exit_side = "SELL" if side == "BUY" else "BUY"

    if side == "BUY":
        tp_price = round(entry_price * (1 + tp), 6)
        sl_price = round(entry_price * (1 - sl), 6)
    else:
        tp_price = round(entry_price * (1 - tp), 6)
        sl_price = round(entry_price * (1 + sl), 6)

    # TP
    try:
        b_post("/fapi/v1/order", {
            "symbol": symbol,
            "side": exit_side,
            "type": "TAKE_PROFIT_MARKET",
            "stopPrice": tp_price,
            "closePosition": True,
            "reduceOnly": True,
            "workingType": "MARK_PRICE"
        })
    except Exception as e:
        log(f"TP error {symbol}: {e}")

    # SL
    try:
        b_post("/fapi/v1/order", {
            "symbol": symbol,
            "side": exit_side,
            "type": "STOP_MARKET",
            "stopPrice": sl_price,
            "closePosition": True,
            "reduceOnly": True,
            "workingType": "MARK_PRICE"
        })
    except Exception as e:
        log(f"SL error {symbol}: {e}")

    # Trailing (optional)
    if USE_TRAILING:
        cb = max(0.1, min(5.0, float(TRAILING_PCT)))
        try:
            b_post("/fapi/v1/order", {
                "symbol": symbol,
                "side": exit_side,
                "type": "TRAILING_STOP_MARKET",
                "callbackRate": cb,
                "reduceOnly": True,
                "quantity": qty
            })
        except Exception as e:
            log(f"Trailing error {symbol}: {e}")

# ==========================
# Filters & symbol meta
# ==========================
def fetch_symbols_and_meta():
    info = exchange_info()
    out = []
    for s in info["symbols"]:
        if s.get("contractType") != "PERPETUAL":
            continue
        sym = s["symbol"]
        if not sym.endswith("USDT"):
            continue
        try:
            t = ticker_24h(sym)
            qv = float(t["quoteVolume"])
            if qv < MIN_QUOTE_VOL_24H:
                continue
        except Exception:
            continue
        step = 0.001
        minq = 0.001
        for f in s.get("filters", []):
            if f.get("filterType") == "LOT_SIZE":
                step = float(f["stepSize"])
                minq = float(f["minQty"])
        out.append({"symbol": sym, "step": step, "min_qty": minq})
    return out

# ==========================
# Indicators
# ==========================
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-9)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ef = series.ewm(span=fast, adjust=False).mean()
    es = series.ewm(span=slow, adjust=False).mean()
    line = ef - es
    sig = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig
    return line, sig, hist

def kdj(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 9, signal: int = 3):
    low_min = low.rolling(window=period).min()
    high_max = high.rolling(window=period).max()
    rsv = (close - low_min) / (high_max - low_min + 1e-9) * 100
    k = rsv.ewm(com=signal-1, adjust=False).mean()
    d = k.ewm(com=signal-1, adjust=False).mean()
    j = 3*k - 2*d
    return k, d, j

def vol_avg(volume: pd.Series, period: int = 20) -> pd.Series:
    return volume.rolling(window=period).mean()

# ==========================
# Strategy (uses previous closed candle)
# ==========================
def evaluate_signal(df: pd.DataFrame):
    if len(df) < max(EMA_SLOW, RSI_PERIOD) + 5:
        return None

    close = df["close"]; high = df["high"]; low = df["low"]; volume = df["volume"]

    ef = ema(close, EMA_FAST)
    em = ema(close, EMA_MID)
    es = ema(close, EMA_SLOW)
    r  = rsi(close, RSI_PERIOD)
    m_line, m_sig, m_hist = macd(close, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    k, d, j = kdj(high, low, close, KDJ_PERIOD, KDJ_SIGNAL)
    vavg = vol_avg(volume, 20)

    # last closed candle (avoid repaint)
    c    = close.iloc[-2]
    rsi_ = r.iloc[-2]
    vf   = volume.iloc[-2]
    vma  = vavg.iloc[-2]
    ef_, em_, es_ = ef.iloc[-2], em.iloc[-2], es.iloc[-2]
    mline, msig, mh = m_line.iloc[-2], m_sig.iloc[-2], m_hist.iloc[-2]
    k1, d1, j1 = k.iloc[-2], d.iloc[-2], j.iloc[-2]

    vol_ok = vf > vma * VOL_MULTIPLIER
    macd_bull = (mline > msig) and (mh > 0)
    macd_bear = (mline < msig) and (mh < 0)

    long_setup  = (ef_ > em_ > es_) and (rsi_ < RSI_BUY_MAX) and vol_ok and macd_bull and (k1 >= d1)
    short_setup = (ef_ < em_ < es_) and (rsi_ > RSI_SELL_MIN) and vol_ok and macd_bear and (k1 <= d1)

    if long_setup:
        conf = 70.0
        if abs(ef_ - em_) / max(c, 1e-9) > 0.002: conf += 10
        if mh > 0 and abs(mh) > 0.1: conf += 5
        if vf > vma * (VOL_MULTIPLIER + 0.5): conf += 10
        return {"side": "BUY", "reason": "EMA uptrend + RSI OK + MACD bull + Vol surge (+KDJ)", "confidence": round(min(conf, 98.0),2)}

    if short_setup:
        conf = 70.0
        if abs(ef_ - em_) / max(c, 1e-9) > 0.002: conf += 10
        if mh < 0 and abs(mh) > 0.1: conf += 5
        if vf > vma * (VOL_MULTIPLIER + 0.5): conf += 10
        return {"side": "SELL", "reason": "EMA downtrend + RSI OK + MACD bear + Vol surge (+KDJ)", "confidence": round(min(conf, 98.0),2)}

    return None

# ==========================
# Risk helpers
# ==========================
def pos_size_usdt(balance_usdt: float, pct: float) -> float:
    return max(0.0, balance_usdt * (pct / 100.0))

def round_step(x: float, step: float) -> float:
    if step <= 0: return x
    return math.floor(x / step) * step

def compute_qty(entry_price: float, usdt_amount: float, step: float, min_qty: float) -> float:
    qty = usdt_amount / max(entry_price, 1e-9)
    qty = round_step(qty, step)
    if qty < min_qty: return 0.0
    return qty

# ==========================
# Main loop
# ==========================
def run():
    global TRADING_ENABLED

    log("🚀 Bot started (Binance Futures)")
    info = exchange_info()
    symbols_meta = []
    for sym in fetch_symbols_and_meta():
        symbols_meta.append(sym)
    log(f"Scanning {len(symbols_meta)} USDT perpetual symbols.")

    while True:
        try:
            # Telegram commands
            for cmd, args, from_id in tg.poll():
                if TG_CHAT and from_id != str(TG_CHAT):
                    continue
                if cmd == "/start":
                    TRADING_ENABLED = True
                    log("✅ Trading ENABLED")
                elif cmd == "/stop":
                    TRADING_ENABLED = False
                    log("🛑 Trading DISABLED")
                elif cmd == "/status":
                    bal = balance_usdt()
                    open_map = positions_open_map()
                    log(f"ℹ️ Status — trading={'ON' if TRADING_ENABLED else 'OFF'}, balance≈{bal:.2f} USDT, open_positions={len(open_map)}")

            open_map = positions_open_map()
            open_count = len(open_map)
            usdt = balance_usdt()

            for meta in symbols_meta:
                if not TRADING_ENABLED:
                    break

                symbol = meta["symbol"]
                if symbol in open_map:
                    # already have a position; skip new entry
                    continue
                if open_count >= MAX_OPEN_POSITIONS:
                    break

                # get klines
                df = klines(symbol, INTERVAL, 400)
                if df.empty:
                    continue

                sig = evaluate_signal(df)
                if not sig:
                    continue

                last_close = float(df["close"].iloc[-1])

                # risk sizing
                pos_usdt = pos_size_usdt(usdt, POSITION_SIZE_PCT)
                qty = compute_qty(last_close, pos_usdt, meta["step"], meta["min_qty"])
                if qty <= 0:
                    continue

                # set leverage & enter
                try:
                    change_leverage(symbol, LEVERAGE)
                except Exception:
                    pass

                side = sig["side"]
                log(f"📈 Signal {symbol}: {sig} | qty≈{qty}")
                try:
                    res = order_market(symbol, side, qty)
                except Exception as e:
                    log(f"❌ Entry failed {symbol}: {e}")
                    continue

                # try derive avg price
                entry_price = last_close
                try:
                    if isinstance(res, dict):
                        if float(res.get("avgPrice", 0) or 0) > 0:
                            entry_price = float(res["avgPrice"])
                        elif float(res.get("price", 0) or 0) > 0:
                            entry_price = float(res["price"])
                except Exception:
                    pass

                log(f"✅ Entered {symbol} {side} @ ~{entry_price}")
                place_exits(symbol, side, entry_price, qty)
                open_count += 1

            time.sleep(SCAN_SECONDS)

        except KeyboardInterrupt:
            log("👋 Stopped by user.")
            break
        except Exception as e:
            log(f"⚠️ Loop error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    run()