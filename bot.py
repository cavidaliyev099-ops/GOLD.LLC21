"""
XAU/USD Telegram siqnal botu (Twelve Data + Forex Factory xəbərləri + GitHub Actions)

Qaydalar:
  - 4H və 1H trend eyni istiqamətdə olmalıdır (məcburi)
  - HAHO (VWMA 20): şam VWMA-dan yuxarıdırsa yalnız BUY, aşağıdırsa yalnız SELL (məcburi)
  - Score >= MIN_SCORE (default 65) olanda siqnal + şəkil göndərilir
  - SL/TP 1:2 risk/mükafat
  - Yüksək təsirli USD xəbərindən 30 dəq əvvəl və 30 dəq sonra siqnal verilmir
  - Hər siqnal sonra izlənir, TP/SL vurulanda Telegram-a yazılır

Bu maliyyə məsləhəti deyil. Əvvəlcə DRY_RUN=1 ilə və demo hesabda yoxla.
"""
import json
import os
import sys
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

# ---------------------------------------------------------------- AYARLAR
TD_KEY = os.getenv("TWELVEDATA_API_KEY", "")
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "")

SYMBOL = os.getenv("SYMBOL", "XAU/USD")
DXY_PROXY_SYMBOL = os.getenv("DXY_PROXY_SYMBOL", "EUR/USD")  # DXY = EUR/USD-nin tərsi
MIN_SCORE = int(os.getenv("MIN_SCORE", "65"))
RR = float(os.getenv("RR", "2"))  # 1:2
COOLDOWN_MIN = int(os.getenv("COOLDOWN_MIN", "60"))
MAX_SIGNALS_PER_DAY = int(os.getenv("MAX_SIGNALS_PER_DAY", "3"))
MAX_RISK_ATR = float(os.getenv("MAX_RISK_ATR", "3.5"))
DRY_RUN = os.getenv("DRY_RUN", "0") == "1"
TEST = os.getenv("TEST", "0") == "1"
SESSION_FILTER = os.getenv("SESSION_FILTER", "1") == "1"
ACTIVE_SESSIONS = os.getenv("ACTIVE_SESSIONS", "LNDN,NYAM,NYPM").split(",")
STATE_FILE = os.getenv("STATE_FILE", "state.json")
CHART_FILE = os.getenv("CHART_FILE", "signal.png")

NEWS_URL = os.getenv("NEWS_URL", "https://nfs.faireconomy.media/ff_calendar_thisweek.json")
NEWS_IMPACT = os.getenv("NEWS_IMPACT", "High").split(",")
BLOCK_BEFORE = int(os.getenv("NEWS_BLOCK_BEFORE_MIN", "30"))
BLOCK_AFTER = int(os.getenv("NEWS_BLOCK_AFTER_MIN", "30"))
NEWS_REFRESH_MIN = int(os.getenv("NEWS_REFRESH_MIN", "180"))

NY = ZoneInfo("America/New_York")
BAKU = ZoneInfo("Asia/Baku")

SESSIONS = {  # KZP indikatorundakı sessiyalar (Nyu-York vaxtı)
    "ASIA": ((20, 0), (0, 0)),
    "LNDN": ((2, 0), (5, 0)),
    "NYAM": ((9, 30), (11, 0)),
    "NYL": ((12, 0), (13, 0)),
    "NYPM": ((13, 30), (16, 0)),
}
INVERSE_NEWS = ("unemployment rate", "jobless claims", "unemployment claims")


# ---------------------------------------------------------------- DATA
def fetch(symbol, interval, size):
    r = requests.get(
        "https://api.twelvedata.com/time_series",
        params={
            "symbol": symbol,
            "interval": interval,
            "outputsize": size,
            "timezone": "UTC",
            "apikey": TD_KEY,
        },
        timeout=30,
    )
    j = r.json()
    if j.get("status") == "error" or "values" not in j:
        raise RuntimeError(f"Twelve Data xətası ({symbol} {interval}): {j.get('message', j)}")
    df = pd.DataFrame(j["values"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce") if c in df else 0.0
    return df.sort_values("datetime").reset_index(drop=True)


def drop_forming(df, minutes):
    now = pd.Timestamp.now(tz="UTC")
    if len(df) and df["datetime"].iloc[-1] + pd.Timedelta(minutes=minutes) > now:
        return df.iloc[:-1].reset_index(drop=True)
    return df


# ---------------------------------------------------------------- İNDİKATORLAR
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def sma(s, n):
    return s.rolling(n).mean()


def rsi(s, n=14):
    d = s.diff()
    up, dn = d.clip(lower=0), -d.clip(upper=0)
    ru = up.ewm(alpha=1 / n, adjust=False).mean()
    rd = dn.ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + ru / rd.replace(0, np.nan))


def atr(df, n=14):
    pc = df["close"].shift()
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def vwma(df, n=20):
    """HAHO = VWMA 20. Həcm yoxdursa SMA-ya keçir."""
    base = sma(df["close"], n)
    if df["volume"].fillna(0).sum() <= 0:
        return base, False
    v = (df["close"] * df["volume"]).rolling(n).sum() / df["volume"].rolling(n).sum()
    return v.fillna(base), True


def trend(df):
    c = df["close"]
    e20, e50 = ema(c, 20), ema(c, 50)
    if c.iloc[-1] > e50.iloc[-1] and e20.iloc[-1] > e50.iloc[-1]:
        return 1
    if c.iloc[-1] < e50.iloc[-1] and e20.iloc[-1] < e50.iloc[-1]:
        return -1
    return 0


def dxy_bias(df):
    """(qısa, uzun): +1 DXY qalxır, -1 enir. EMA 20/50/100 əsasında."""
    c = df["close"]
    e20, e50, e100 = ema(c, 20), ema(c, 50), ema(c, 100)
    if c.iloc[-1] > e20.iloc[-1] and c.iloc[-1] > e50.iloc[-1]:
        short = 1
    elif c.iloc[-1] < e20.iloc[-1] and c.iloc[-1] < e50.iloc[-1]:
        short = -1
    else:
        short = 0
    longt = 1 if e50.iloc[-1] > e100.iloc[-1] else -1
    return short, longt


# ---------------------------------------------------------------- LİKVİDLİK
def session_levels(df5):
    now_ny = datetime.now(NY)
    d = df5.copy()
    d["t"] = d["datetime"].dt.tz_convert(NY)
    out = []
    for name, ((sh, sm), (eh, em)) in SESSIONS.items():
        for k in range(0, 4):
            day = (now_ny - timedelta(days=k)).date()
            if name == "ASIA":
                start = datetime.combine(day - timedelta(days=1), time(sh, sm), tzinfo=NY)
                end = datetime.combine(day, time(0, 0), tzinfo=NY)
            else:
                start = datetime.combine(day, time(sh, sm), tzinfo=NY)
                end = datetime.combine(day, time(eh, em), tzinfo=NY)
            if end > now_ny:
                continue
            seg = d[(d["t"] >= start) & (d["t"] < end)]
            if len(seg) >= 3:
                out.append((f"{name} High", float(seg["high"].max()), "H", end))
                out.append((f"{name} Low", float(seg["low"].min()), "L", end))
                break
    return out


def prev_day_levels(df1h):
    now_ny = datetime.now(NY)
    roll = now_ny.replace(hour=17, minute=0, second=0, microsecond=0)
    if now_ny < roll:
        roll -= timedelta(days=1)
    start = roll - timedelta(days=1)
    seg = df1h[(df1h["datetime"] >= pd.Timestamp(start)) & (df1h["datetime"] < pd.Timestamp(roll))]
    if len(seg) < 3:
        return []
    return [
        ("PDH", float(seg["high"].max()), "H", roll),
        ("PDL", float(seg["low"].min()), "L", roll),
    ]


def find_sweep(df5, levels, direction, lookback=12):
    last = df5.iloc[-lookback:]
    best = None
    for name, price, typ, end in levels:
        for _, r in last.iterrows():
            if r["datetime"] < pd.Timestamp(end):
                continue
            if direction == 1 and typ == "L" and r["low"] < price < r["close"]:
                if df5["close"].iloc[-1] > price:
                    best = (name, price, float(r["low"]), r["datetime"])
            if direction == -1 and typ == "H" and r["high"] > price > r["close"]:
                if df5["close"].iloc[-1] < price:
                    best = (name, price, float(r["high"]), r["datetime"])
    return best


def bos(df5, direction, n=3, within=3):
    h, l, c = df5["high"].values, df5["low"].values, df5["close"].values
    L = len(df5)
    idx = [
        i
        for i in range(n, L - n)
        if (direction == 1 and h[i] == max(h[i - n : i + n + 1]))
        or (direction == -1 and l[i] == min(l[i - n : i + n + 1]))
    ]
    if not idx:
        return False
    i = idx[-1]
    lvl = h[i] if direction == 1 else l[i]
    for k in range(1, within + 1):
        j = L - k
        if j <= i or j < 1:
            continue
        if direction == 1 and c[j] > lvl and c[j - 1] <= lvl:
            return True
        if direction == -1 and c[j] < lvl and c[j - 1] >= lvl:
            return True
    return False


# ---------------------------------------------------------------- TBT-ə bənzər trend xətti qırılması
def pivots(df, n=13):
    h, l = df["high"].values, df["low"].values
    L = len(df)
    ph = [(i, h[i]) for i in range(n, L - n) if h[i] == max(h[i - n : i + n + 1])]
    pl = [(i, l[i]) for i in range(n, L - n) if l[i] == min(l[i - n : i + n + 1])]
    return ph, pl


def tbt_break(df5, direction, n=13, within=3):
    """
    BUY : son iki pivot high enən müqavimət xətti əmələ gətirir, qiymət onu yuxarı qırıb.
    SELL: son iki pivot low qalxan dəstək xətti əmələ gətirir, qiymət onu aşağı qırıb.
    Period 13 (ChartPrime TBT ayarı ilə eyni). Alqoritm ChartPrime-ın eyni deyil, ona yaxın variantdır.
    """
    ph, pl = pivots(df5, n)
    c = df5["close"].values
    L = len(c)
    if direction == 1:
        if len(ph) < 2:
            return None
        (x1, y1), (x2, y2) = ph[-2], ph[-1]
        if y2 >= y1:
            return None
    else:
        if len(pl) < 2:
            return None
        (x1, y1), (x2, y2) = pl[-2], pl[-1]
        if y2 <= y1:
            return None
    slope = (y2 - y1) / (x2 - x1)
    for k in range(1, within + 1):
        j = L - k
        if j - 1 <= x2:
            continue
        line_j = y1 + slope * (j - x1)
        line_p = y1 + slope * (j - 1 - x1)
        if direction == 1 and c[j] > line_j and c[j - 1] <= line_p:
            return {"x1": x1, "y1": y1, "slope": slope, "level": line_j}
        if direction == -1 and c[j] < line_j and c[j - 1] >= line_p:
            return {"x1": x1, "y1": y1, "slope": slope, "level": line_j}
    return None


# ---------------------------------------------------------------- Volume profile (qiymət səviyyəsində keçirilən vaxt)
def vol_profile(df, bars=288, bins=30):
    d = df.iloc[-bars:]
    lo, hi = float(d["low"].min()), float(d["high"].max())
    if hi <= lo:
        return None
    edges = np.linspace(lo, hi, bins + 1)
    dens = np.zeros(bins)
    for lo_i, hi_i in zip(d["low"].values, d["high"].values):
        a = max(np.searchsorted(edges, lo_i, side="right") - 1, 0)
        b = min(np.searchsorted(edges, hi_i, side="right") - 1, bins - 1)
        dens[a : b + 1] += 1
    poc = int(dens.argmax())
    thr = 0.7 * dens.max()
    hvn = [(float(edges[i]), float(edges[i + 1])) for i in range(bins) if dens[i] >= thr]
    return {"poc": (float(edges[poc]) + float(edges[poc + 1])) / 2, "hvn": hvn}


def vp_support(direction, price, a, vp):
    """BUY: qiymət sıx zonanın içində və ya 1 ATR üstündə (dəstək). SELL: eksinə."""
    if not vp:
        return None
    for lo, hi in vp["hvn"]:
        if direction == 1 and lo <= price and price - hi <= a:
            return (lo, hi)
        if direction == -1 and price <= hi and lo - price <= a:
            return (lo, hi)
    return None


# ---------------------------------------------------------------- XƏBƏR (Forex Factory)
def parse_num(s):
    if s is None:
        return None
    s = str(s).replace(",", "").replace("%", "").strip()
    mult = 1.0
    for suf, m in (("K", 1e3), ("M", 1e6), ("B", 1e9), ("T", 1e12)):
        if s.upper().endswith(suf):
            mult, s = m, s[:-1]
            break
    try:
        return float(s) * mult
    except ValueError:
        return None


def gold_bias(e):
    """Proqnoz əvvəlkindən yüksəkdirsə (adətən) dollar üçün müsbət, qızıl üçün mənfi.
    Qaytarır: qızıl üçün istiqamət (+1 yuxarı, -1 aşağı, 0 naməlum). Zəif gözləntidir."""
    f, p = parse_num(e.get("forecast")), parse_num(e.get("previous"))
    if f is None or p is None or f == p:
        return 0
    usd_up = f > p
    if any(k in e["title"].lower() for k in INVERSE_NEWS):
        usd_up = not usd_up
    return -1 if usd_up else 1


def get_news(state):
    now = pd.Timestamp.now(tz="UTC")
    cache = state.get("news")
    if cache and (now - pd.Timestamp(cache["ts"])).total_seconds() < NEWS_REFRESH_MIN * 60:
        return cache["events"]
    try:
        r = requests.get(NEWS_URL, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"Xəbər təqvimi alınmadı: {e}")
        return cache["events"] if cache else None
    events = []
    for e in data:
        if e.get("country") != "USD" or e.get("impact") not in NEWS_IMPACT:
            continue
        try:
            ts = pd.Timestamp(e["date"]).tz_convert("UTC")
        except Exception:
            continue
        events.append(
            {
                "title": e.get("title", ""),
                "ts": ts.isoformat(),
                "impact": e.get("impact"),
                "forecast": e.get("forecast"),
                "previous": e.get("previous"),
            }
        )
    state["news"] = {"ts": now.isoformat(), "events": events}
    return events


def news_context(events):
    """blocked: indi pəncərədə olan xəbər; upcoming: növbəti 4 saatdakı xəbərlər."""
    now = pd.Timestamp.now(tz="UTC")
    blocked, upcoming, alerts = None, [], []
    for e in events or []:
        mins = (pd.Timestamp(e["ts"]) - now).total_seconds() / 60
        e = dict(e, mins=mins)
        if -BLOCK_AFTER <= mins <= BLOCK_BEFORE:
            blocked = e
        elif BLOCK_BEFORE < mins <= 240:
            upcoming.append(e)
        if -BLOCK_AFTER <= mins <= 60:
            alerts.append(e)
    return blocked, upcoming, alerts


def news_line(e):
    t = pd.Timestamp(e["ts"]).tz_convert(BAKU).strftime("%H:%M")
    b = gold_bias(e)
    exp = {
        1: "gözlənti: dollar zəif → qızıl üçün yuxarı təzyiq",
        -1: "gözlənti: dollar güclü → qızıl üçün aşağı təzyiq",
        0: "gözlənti: aydın deyil",
    }[b]
    return (
        f"{e['title']} (USD, {e['impact']}) — Bakı vaxtı {t}\n"
        f"Proqnoz: {e.get('forecast') or '-'} | Əvvəlki: {e.get('previous') or '-'} | {exp}"
    )


# ---------------------------------------------------------------- SCORE
def score(d, ctx):
    df5 = ctx["df5"]
    close = df5["close"].iloc[-1]
    a = ctx["atr"]
    pts = {}

    trend_ok = ctx["t4"] == d and ctx["t1"] == d
    pts["Trend 4H+1H"] = 20 if trend_ok else 0

    e20, e50, e100 = ctx["e20"], ctx["e50"], ctx["e100"]
    if d == 1:
        full = close > e20 > e50 > e100
        part = close > e50 and e20 > e50
    else:
        full = close < e20 < e50 < e100
        part = close < e50 and e20 < e50
    pts["EMA 20/50/100"] = 10 if full else (5 if part else 0)

    hv_ok = close > ctx["vwma"] if d == 1 else close < ctx["vwma"]
    pts["HAHO (VWMA 20)"] = 10 if hv_ok else 0

    r = ctx["rsi"]
    r_now, r_prev = r.iloc[-1], r.iloc[-2]
    window = r.iloc[-24:]
    if d == 1:
        reversal = window.min() <= 30 and r_now > r_prev and r_now > 30
        cont = 50 < r_now < 70 and r_now > r_prev
    else:
        reversal = window.max() >= 70 and r_now < r_prev and r_now < 70
        cont = 30 < r_now < 50 and r_now < r_prev
    pts["RSI"] = 10 if reversal else (5 if cont else 0)

    sw = find_sweep(df5, ctx["levels"], d)
    pts["Liquidity sweep"] = 15 if sw else 0

    pts["BOS"] = 5 if bos(df5, d) else 0

    tbt = tbt_break(df5, d)
    pts["Trend xətti qırılması"] = 10 if tbt else 0

    vp = vp_support(d, close, a, ctx["vp"])
    pts["Volume profile"] = 5 if vp else 0

    ds, dl = ctx["dxy"]
    if ds == -d and dl == -d:
        pts["DXY"] = 10
    elif ds == -d or dl == -d or (ds == 0 and dl == 0):
        pts["DXY"] = 5
    else:
        pts["DXY"] = 0

    conflict = any(gold_bias(e) == -d for e in ctx["upcoming"])
    pts["Xəbər"] = 0 if conflict else 5

    return {
        "total": sum(pts.values()),
        "pts": pts,
        "required_ok": trend_ok and hv_ok,
        "trend_ok": trend_ok,
        "hv_ok": hv_ok,
        "sweep": sw,
        "tbt": tbt,
        "vp": vp,
    }


# ---------------------------------------------------------------- ŞƏKİL
def make_chart(ctx, d, entry, sl, tp, res, total, path, bars=90):
    df5 = ctx["df5"]
    start = max(len(df5) - bars, 0)
    w = df5.iloc[start:].reset_index(drop=True)
    n = len(w)

    bg, fg = "#131722", "#d1d4dc"
    fig, ax = plt.subplots(figsize=(10, 6.2), dpi=110)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    # mənfəət / zərər zonaları
    x0, x1 = n - 14, n + 16
    ax.fill_between([x0, x1], entry, tp, color="#26a69a", alpha=0.18, lw=0)
    ax.fill_between([x0, x1], entry, sl, color="#ef5350", alpha=0.18, lw=0)

    # şamlar
    for i, r in w.iterrows():
        col = "#26a69a" if r["close"] >= r["open"] else "#ef5350"
        ax.plot([i, i], [r["low"], r["high"]], color=col, lw=1, zorder=2)
        ax.add_patch(
            Rectangle(
                (i - 0.32, min(r["open"], r["close"])),
                0.64,
                max(abs(r["close"] - r["open"]), 1e-9),
                color=col,
                zorder=3,
            )
        )

    # hərəkətli ortalar və VWMA
    full = df5["close"]
    for series, color, lab in (
        (ema(full, 20), "#fdd835", "EMA20"),
        (ema(full, 50), "#fb8c00", "EMA50"),
        (ema(full, 100), "#ab47bc", "EMA100"),
        (ctx["vwma_series"], "#42a5f5", "VWMA20"),
    ):
        ax.plot(range(n), series.iloc[start:].values, color=color, lw=1.1, label=lab, zorder=1)

    # sweep səviyyəsi
    if res["sweep"]:
        name, lvl = res["sweep"][0], res["sweep"][1]
        ax.axhline(lvl, color="#90a4ae", ls=":", lw=1)
        ax.text(1, lvl, f" {name} (sweep)", color="#90a4ae", fontsize=8, va="bottom")

    # trend xətti
    if res["tbt"]:
        t = res["tbt"]
        xs = np.array([max(t["x1"], start), len(df5) - 1])
        ys = t["y1"] + t["slope"] * (xs - t["x1"])
        ax.plot(xs - start, ys, color="#e0e0e0", lw=1.4, ls="--", zorder=4)
        ax.text(xs[0] - start, ys[0], " trend xətti", color="#e0e0e0", fontsize=8)

    # giriş / SL / TP xətləri
    for y, col, lab in (
        (entry, "#ffffff", f"GİRİŞ {entry:.2f}"),
        (sl, "#ef5350", f"SL {sl:.2f}"),
        (tp, "#26a69a", f"TP {tp:.2f}  (1:{RR:g})"),
    ):
        ax.hlines(y, x0, x1, colors=col, lw=1.6, zorder=5)
        ax.text(x1 + 0.3, y, lab, color=col, fontsize=9, va="center", fontweight="bold")

    # ox
    ax.annotate(
        "BUY" if d == 1 else "SELL",
        xy=(n - 1, entry),
        xytext=(n - 12, entry + (tp - entry) * 0.55),
        color="white",
        fontsize=11,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="white", lw=1.6),
        zorder=6,
    )

    ymin = min(w["low"].min(), sl) - abs(entry - sl) * 0.3
    ymax = max(w["high"].max(), tp) + abs(tp - entry) * 0.15
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(-1, n + 14)
    ax.tick_params(colors=fg, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#2a2e39")
    ax.grid(color="#1e222d", lw=0.6)
    ax.yaxis.tick_right()
    xt = list(range(0, n, 15))
    ax.set_xticks(xt)
    ax.set_xticklabels([w["datetime"].iloc[i].tz_convert(BAKU).strftime("%H:%M") for i in xt])
    ax.legend(loc="upper left", fontsize=7, facecolor=bg, edgecolor="#2a2e39", labelcolor=fg, ncol=4)
    ax.set_title(
        f"{'BUY' if d == 1 else 'SELL'} {SYMBOL}  |  5 dəq  |  Score {total}/100  |  vaxt: Bakı",
        color=fg,
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, facecolor=bg)
    plt.close(fig)


# ---------------------------------------------------------------- TELEGRAM / STATE
def tg_post(method, **kw):
    if not TG_TOKEN or not TG_CHAT:
        raise RuntimeError("TELEGRAM_BOT_TOKEN və TELEGRAM_CHAT_ID lazımdır")
    r = requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/{method}", timeout=60, **kw)
    r.raise_for_status()


def send(text):
    print(text)
    if not DRY_RUN:
        tg_post("sendMessage", json={"chat_id": TG_CHAT, "text": text})


def send_photo(path, caption):
    print(f"[ŞƏKİL: {path}] {caption}")
    if DRY_RUN:
        return
    with open(path, "rb") as f:
        tg_post("sendPhoto", data={"chat_id": TG_CHAT, "caption": caption[:1000]}, files={"photo": f})


def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(st):
    with open(STATE_FILE, "w") as f:
        json.dump(st, f)


def in_session():
    now_ny = datetime.now(NY).time()
    for name in ACTIVE_SESSIONS:
        name = name.strip()
        if name not in SESSIONS:
            continue
        (sh, sm), (eh, em) = SESSIONS[name]
        s, e = time(sh, sm), time(eh, em)
        if (s <= now_ny < e) if s < e else (now_ny >= s or now_ny < e):
            return name
    return None


def check_open(state, df5):
    """Açıq siqnalları izlə: TP və ya SL vurulubsa bildir."""
    now = pd.Timestamp.now(tz="UTC")
    stats = state.setdefault("stats", {"tp": 0, "sl": 0, "expired": 0})
    keep = []
    for s in state.get("open", []):
        t0 = pd.Timestamp(s["ts"])
        res = None
        for _, r in df5[df5["datetime"] > t0].iterrows():
            if s["dir"] == 1:
                sl_hit, tp_hit = r["low"] <= s["sl"], r["high"] >= s["tp"]
            else:
                sl_hit, tp_hit = r["high"] >= s["sl"], r["low"] <= s["tp"]
            if sl_hit:  # eyni şamda ikisi də olsa konservativ: SL
                res = "SL"
                break
            if tp_hit:
                res = "TP"
                break
        side = "BUY" if s["dir"] == 1 else "SELL"
        if res:
            stats["tp" if res == "TP" else "sl"] += 1
            icon = "✅ TP vuruldu" if res == "TP" else "❌ SL vuruldu"
            send(
                f"{icon}: {side} {SYMBOL} (giriş {s['entry']:.2f})\n"
                f"Statistika: TP {stats['tp']} | SL {stats['sl']} | vaxtı bitən {stats['expired']}"
            )
        elif (now - t0) > pd.Timedelta(hours=12):
            stats["expired"] += 1
            send(f"⌛ {side} {SYMBOL} siqnalı 12 saat ərzində nə TP, nə SL vurdu (bağlandı).")
        else:
            keep.append(s)
    state["open"] = keep


# ---------------------------------------------------------------- MAIN
def main():
    if not TD_KEY:
        sys.exit("TWELVEDATA_API_KEY yoxdur")

    state = load_state()
    df5 = drop_forming(fetch(SYMBOL, "5min", 1000), 5)
    age_min = (pd.Timestamp.now(tz="UTC") - df5["datetime"].iloc[-1]).total_seconds() / 60
    if age_min > 20 and not TEST:
        print(f"Bazar bağlıdır (son şam {age_min:.0f} dəq əvvəl).")
        return

    check_open(state, df5)

    # xəbər
    events = get_news(state)
    blocked, upcoming, alerts = news_context(events)
    alerted = state.setdefault("news_alerted", [])
    for e in alerts:
        key = f"{e['ts']}|{e['title']}"
        if key not in alerted:
            where = "başlayır" if e["mins"] > 0 else "açıqlandı"
            send(
                f"⚠️ XƏBƏR VAXTI ({where}): {news_line(e)}\n"
                f"Xəbərdən {BLOCK_BEFORE} dəq əvvəl və {BLOCK_AFTER} dəq sonra bot siqnal vermir. "
                f"Açıqlanan rəqəm (actual) pulsuz mənbədə yoxdur, nəticəni özün yoxla."
            )
            alerted.append(key)
    state["news_alerted"] = alerted[-50:]

    sess = in_session()
    if (SESSION_FILTER and not sess and not TEST) or (blocked and not TEST):
        print("Aktiv sessiya yoxdur və ya xəbər pəncərəsidir.")
        save_state(state)
        return

    df4 = drop_forming(fetch(SYMBOL, "4h", 250), 240)
    df1 = drop_forming(fetch(SYMBOL, "1h", 300), 60)
    try:
        eu = drop_forming(fetch(DXY_PROXY_SYMBOL, "1h", 300), 60)
        for c in ["open", "high", "low", "close"]:
            eu[c] = 1.0 / eu[c]  # DXY proksisi: EUR/USD-nin tərsi
        dxy = dxy_bias(eu)
    except Exception as e:
        print(f"DXY proksisi alınmadı: {e}")
        dxy = (0, 0)

    vw, has_vol = vwma(df5)
    c5 = df5["close"]
    a = float(atr(df5).iloc[-1])
    ctx = {
        "df5": df5,
        "atr": a,
        "t4": trend(df4),
        "t1": trend(df1),
        "e20": ema(c5, 20).iloc[-1],
        "e50": ema(c5, 50).iloc[-1],
        "e100": ema(c5, 100).iloc[-1],
        "vwma": vw.iloc[-1],
        "vwma_series": vw,
        "rsi": rsi(c5),
        "levels": session_levels(df5) + prev_day_levels(df1),
        "dxy": dxy,
        "vp": vol_profile(df5),
        "upcoming": upcoming,
    }
    price = float(c5.iloc[-1])
    results = {d: score(d, ctx) for d in (1, -1)}

    tt = {1: "yüksəliş", -1: "enmə", 0: "neytral"}
    status = (
        f"{SYMBOL} {price:.2f}\n"
        f"4H: {tt[ctx['t4']]} | 1H: {tt[ctx['t1']]} | DXY (EUR/USD proksi): {tt[dxy[0]]}/{tt[dxy[1]]}\n"
        f"BUY score: {results[1]['total']} | SELL score: {results[-1]['total']} (minimum {MIN_SCORE})\n"
        f"Sessiya: {sess or '-'} | VWMA həcmlə: {'bəli' if has_vol else 'xeyr (SMA ilə əvəz)'}"
    )
    if blocked:
        status += f"\nXəbər pəncərəsi: {blocked['title']}"
    if upcoming:
        status += "\nNövbəti xəbər: " + "; ".join(
            f"{e['title']} ({pd.Timestamp(e['ts']).tz_convert(BAKU).strftime('%H:%M')} Bakı)"
            for e in upcoming[:2]
        )

    today = datetime.now(NY).date().isoformat()
    if state.get("day") != today:
        state["day"], state["count"] = today, 0

    sent = False
    for d in (1, -1):
        res = results[d]
        if not (res["required_ok"] and res["total"] >= MIN_SCORE) or blocked:
            continue
        if state["count"] >= MAX_SIGNALS_PER_DAY:
            print("Gündəlik siqnal limiti dolub.")
            continue
        last = state.get("last")
        if last and last.get("dir") == d:
            mins = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(last["ts"])).total_seconds() / 60
            if mins < COOLDOWN_MIN:
                print(f"Cooldown: {mins:.0f} dəq keçib.")
                continue

        sw = res["sweep"]
        if d == 1:
            sl = (min(sw[2], price - a) if sw else price - 1.5 * a) - 0.1 * a
            risk = price - sl
            tp = price + RR * risk
        else:
            sl = (max(sw[2], price + a) if sw else price + 1.5 * a) + 0.1 * a
            risk = sl - price
            tp = price - RR * risk
        if risk > MAX_RISK_ATR * a:
            print(f"Risk çox böyükdür ({risk / a:.1f} ATR), siqnal atlanır.")
            continue

        side = "BUY" if d == 1 else "SELL"
        extras = []
        if sw:
            extras.append(f"Sweep: {sw[0]} ({sw[1]:.2f})")
        if res["tbt"]:
            extras.append(f"Trend xətti qırılıb ({res['tbt']['level']:.2f})")
        if res["vp"]:
            extras.append(f"Sıx volume zonası: {res['vp'][0]:.2f}–{res['vp'][1]:.2f}")
        breakdown = "\n".join(f"  {k}: {v}" for k, v in res["pts"].items())
        news_txt = ("\n\nXəbər: " + news_line(upcoming[0])) if upcoming else ""
        caption = (
            f"{'🟢' if d == 1 else '🔴'} {side} {SYMBOL} | Score {res['total']}/100\n"
            f"Giriş: {price:.2f}\nSL: {sl:.2f}\nTP: {tp:.2f} (1:{RR:g})"
        )
        msg = (
            f"{caption}\n\n"
            + ("\n".join(x for x in extras if x) + "\n\n" if extras else "")
            + f"Score təfsilatı:\n{breakdown}\n\n{status}{news_txt}\n\n"
            "Bu maliyyə məsləhəti deyil. Lotu və riski özün idarə et."
        )
        try:
            make_chart(ctx, d, price, sl, tp, res, res["total"], CHART_FILE)
            send_photo(CHART_FILE, caption)
        except Exception as e:
            print(f"Şəkil göndərilmədi: {e}")
        send(msg)

        now_iso = pd.Timestamp.now(tz="UTC").isoformat()
        state["last"] = {"dir": d, "ts": now_iso}
        state["count"] += 1
        state.setdefault("open", []).append(
            {
                "dir": d,
                "entry": price,
                "sl": float(sl),
                "tp": float(tp),
                "ts": df5["datetime"].iloc[-1].isoformat(),
            }
        )
        sent = True

    save_state(state)
    if TEST and not sent:
        send("Bot işləyir (siqnal yoxdur).\n" + status)
    elif not sent:
        print("Siqnal yoxdur.\n" + status)


if __name__ == "__main__":
    main()
