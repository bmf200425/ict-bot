#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BOT DE SEÑALES ICT + SMART MONEY CONCEPTS  —  BTC / ETH
=======================================================
Analiza el mercado y AVISA. No ejecuta operaciones, no conecta cuentas
ni coloca órdenes. Si no hay una oportunidad clara responde: NO OPERAR.

Uso rápido:
    python ict_signal_bot.py --once          # un análisis y termina
    python ict_signal_bot.py                 # vigila cada 5 min
    python ict_signal_bot.py --demo          # prueba con datos simulados

Telegram (opcional): define las variables de entorno
    TELEGRAM_TOKEN=xxxxx   TELEGRAM_CHAT_ID=yyyyy

Aviso: herramienta educativa/analítica. No es asesoramiento financiero.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

NY = ZoneInfo("America/New_York")

# ─────────────────────────────── CONFIGURACIÓN ───────────────────────────────
CONFIG = {
    "symbols": ["BTCUSDT", "ETHUSDT"],
    # temporalidad: nº de velas a descargar
    "timeframes": {"1d": 400, "4h": 500, "1h": 500, "15m": 500, "5m": 500},
    # fuerza del fractal (velas a cada lado) para definir un swing
    "swing_len": {"1d": 2, "4h": 2, "1h": 3, "15m": 3, "5m": 3},
    "min_rr": 2.0,             # R:B mínimo NETO (después de costes)
    "costs_pct": 0.10,         # comisión + slippage ida y vuelta, en %
    "sweep_lookback": 32,      # velas de 15m donde buscar barridos (~8 h)
    "mss_window": 20,          # velas de 15m tras el barrido para ver el MSS
    "eq_tol_atr": 0.15,        # tolerancia de Equal Highs/Lows (x ATR)
    "sl_buffer_atr": 0.25,     # colchón del stop más allá de la invalidación (x ATR 15m)
    "fvg_min_atr": 0.10,       # tamaño mínimo de un FVG (x ATR)
    "max_target_atr_1d": 3.0,  # objetivos no más lejos de 3 ATR diarios
    "min_confluence": 3,       # confluencias extra mínimas para emitir señal
    "loop_minutes": 5,
    "state_file": "ict_bot_state.json",
}

EMA_PERIODS = (9, 20, 50, 100, 200)
TF_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000,
         "4h": 14_400_000, "1d": 86_400_000, "1w": 604_800_000}

# ─────────────────────────────── DATOS DE MERCADO ────────────────────────────
BINANCE_HOSTS = ["https://api.binance.com", "https://data-api.binance.vision"]
BYBIT_TF = {"1m": "1", "5m": "5", "15m": "15", "1h": "60", "4h": "240", "1d": "D"}


def _get_json(url: str, timeout: int = 15):
    req = urllib.request.Request(url, headers={"User-Agent": "ict-signal-bot/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _finish(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    df["t"] = df["t"].astype("int64")
    now_ms = int(time.time() * 1000)
    df = df[df["t"] + TF_MS[tf] <= now_ms]          # solo velas CERRADAS
    df.index = pd.to_datetime(df["t"], unit="ms", utc=True)
    return df[["open", "high", "low", "close", "volume"]].copy()


COLS = ["t", "open", "high", "low", "close", "volume"]


def _binance(host: str):
    def fetch(symbol: str, tf: str, limit: int) -> pd.DataFrame:
        q = urllib.parse.urlencode({"symbol": symbol, "interval": tf, "limit": min(limit, 1000)})
        data = _get_json(f"{host}/api/v3/klines?{q}")
        return _finish(pd.DataFrame([r[:6] for r in data], columns=COLS), tf)
    return fetch


def _bybit(symbol: str, tf: str, limit: int) -> pd.DataFrame:
    q = urllib.parse.urlencode({"category": "spot", "symbol": symbol,
                                "interval": BYBIT_TF[tf], "limit": min(limit, 1000)})
    data = _get_json(f"https://api.bybit.com/v5/market/kline?{q}")["result"]["list"][::-1]
    return _finish(pd.DataFrame([r[:6] for r in data], columns=COLS), tf)


def _kraken(symbol: str, tf: str, limit: int) -> pd.DataFrame:
    pair = {"BTCUSDT": "XBTUSDT"}.get(symbol, symbol)
    iv = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}[tf]
    js = _get_json(f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval={iv}")
    if js.get("error"):
        raise RuntimeError(js["error"])
    rows = next(v for k, v in js["result"].items() if k != "last")[-limit:]
    df = pd.DataFrame([[int(r[0]) * 1000, r[1], r[2], r[3], r[4], r[6]] for r in rows], columns=COLS)
    return _finish(df, tf)


# (nombre, función, prefijo del gráfico en TradingView) — se prueban en orden.
# Las tres primeras devuelven precios de Binance; desde servidores de EE. UU.
# (como GitHub) suele responder Binance.US o Kraken.
SOURCES = [
    ("Binance", _binance("https://api.binance.com"), "BINANCE"),
    ("Binance data-api", _binance("https://data-api.binance.vision"), "BINANCE"),
    ("Binance.US", _binance("https://api.binance.us"), "BINANCEUS"),
    ("Bybit", _bybit, "BYBIT"),
    ("Kraken", _kraken, "KRAKEN"),
]


def fetch_klines(symbol: str, tf: str, limit: int) -> pd.DataFrame:
    """Compatibilidad: descarga una temporalidad de la primera fuente que responda."""
    errors = []
    for name, fn, _ in SOURCES:
        try:
            return fn(symbol, tf, limit)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{name}: {e}")
    raise RuntimeError(f"No se pudieron descargar {symbol} {tf}: " + " | ".join(errors))


# ─────────────────────────────── INDICADORES ─────────────────────────────────
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    for p in EMA_PERIODS:
        df[f"ema{p}"] = c.ewm(span=p, adjust=False).mean()
    return df


# ─────────────────────────────── ESTRUCTURAS DE DATOS ────────────────────────
@dataclass
class Swing:
    idx: int
    price: float
    kind: str        # "high" | "low"
    label: str = ""  # HH, LH, HL, LL
    time: pd.Timestamp | None = None


@dataclass
class StructEvent:
    kind: str        # BOS | CHoCH
    direction: str   # bull | bear
    idx: int         # vela que rompe (cierre)
    level: float
    swing_idx: int
    displacement: bool
    time: pd.Timestamp | None = None


@dataclass
class Zone:
    kind: str        # FVG | IFVG | OB
    direction: str   # bull | bear
    top: float
    bottom: float
    idx: int
    time: pd.Timestamp
    tf: str = ""
    mitigated: bool = False
    invalidated: bool = False

    @property
    def mid(self) -> float:
        return (self.top + self.bottom) / 2

    def name(self) -> str:
        d = "alcista" if self.direction == "bull" else "bajista"
        return f"{self.kind} {d} {self.tf} [{fmt(self.bottom)} – {fmt(self.top)}]"


@dataclass
class Level:
    price: float
    name: str
    side: str            # BSL (arriba) | SSL (abajo)
    time: pd.Timestamp   # cuándo se formó


@dataclass
class TFAnalysis:
    tf: str
    df: pd.DataFrame
    swings_h: list
    swings_l: list
    events: list
    trend: str | None
    fvgs: list
    obs: list


# ─────────────────────────────── UTILIDADES ──────────────────────────────────
def fmt(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x:,.2f}" if x >= 100 else f"{x:,.4f}"


def tf_label(d: str | None) -> str:
    return {"bull": "ALCISTA", "bear": "BAJISTA"}.get(d or "", "NEUTRAL")


# ─────────────────────────────── A. ESTRUCTURA ───────────────────────────────
def find_swings(df: pd.DataFrame, n: int):
    hi, lo = df["high"].values, df["low"].values
    sh, sl = [], []
    for i in range(n, len(df) - n):
        if hi[i] >= hi[i - n:i + n + 1].max() and hi[i] > hi[i - n:i].max():
            sh.append(Swing(i, hi[i], "high", time=df.index[i]))
        if lo[i] <= lo[i - n:i + n + 1].min() and lo[i] < lo[i - n:i].min():
            sl.append(Swing(i, lo[i], "low", time=df.index[i]))
    for seq, up, dn in ((sh, "HH", "LH"), (sl, "HL", "LL")):
        for a, b in zip(seq, seq[1:]):
            b.label = up if b.price > a.price else dn
    return sh, sl


def is_displacement(df: pd.DataFrame, j: int, direction: str, k: int = 3) -> bool:
    """Vela(s) impulsiva(s): cuerpo grande vs ATR y poca mecha."""
    o, c, h, l, a = (df[x].values for x in ("open", "close", "high", "low", "atr"))
    for i in range(max(1, j - k + 1), j + 1):
        body = c[i] - o[i]
        rng = max(h[i] - l[i], 1e-12)
        if direction == "bull" and body > 0 and body >= 1.0 * a[i - 1] and body / rng >= 0.55:
            return True
        if direction == "bear" and body < 0 and -body >= 1.0 * a[i - 1] and -body / rng >= 0.55:
            return True
    return False


def analyze_structure(df: pd.DataFrame, n: int):
    """BOS / CHoCH confirmados SOLO al cierre de vela."""
    sh, sl = find_swings(df, n)
    conf_h = {s.idx + n: s for s in sh}
    conf_l = {s.idx + n: s for s in sl}
    c = df["close"].values
    events, trend = [], None
    last_h = last_l = None
    h_broken = l_broken = True
    for j in range(len(df)):
        if j in conf_h:
            last_h, h_broken = conf_h[j], False
        if j in conf_l:
            last_l, l_broken = conf_l[j], False
        if last_h is not None and not h_broken and c[j] > last_h.price:
            kind = "CHoCH" if trend == "bear" else "BOS"
            events.append(StructEvent(kind, "bull", j, last_h.price, last_h.idx,
                                      is_displacement(df, j, "bull"), df.index[j]))
            h_broken, trend = True, "bull"
        if last_l is not None and not l_broken and c[j] < last_l.price:
            kind = "CHoCH" if trend == "bull" else "BOS"
            events.append(StructEvent(kind, "bear", j, last_l.price, last_l.idx,
                                      is_displacement(df, j, "bear"), df.index[j]))
            l_broken, trend = True, "bear"
    return sh, sl, events, trend


# ─────────────────────────────── C. FVG / IFVG / OB ──────────────────────────
def _track(zone: Zone, df: pd.DataFrame, start: int) -> Zone | None:
    """Marca mitigación/invalidación. Devuelve un IFVG si un FVG se invalida."""
    hi, lo, c = df["high"].values, df["low"].values, df["close"].values
    for k in range(start, len(df)):
        if zone.direction == "bull":
            if lo[k] <= zone.top:
                zone.mitigated = True
            if c[k] < zone.bottom:
                zone.invalidated = True
                break
        else:
            if hi[k] >= zone.bottom:
                zone.mitigated = True
            if c[k] > zone.top:
                zone.invalidated = True
                break
    else:
        return None
    if zone.kind == "FVG":
        inv = Zone("IFVG", "bear" if zone.direction == "bull" else "bull",
                   zone.top, zone.bottom, k, df.index[k], zone.tf)
        _track(inv, df, k + 1)
        return inv
    return None


def find_fvgs(df: pd.DataFrame, tf: str):
    hi, lo, a = df["high"].values, df["low"].values, df["atr"].values
    out = []
    mn = CONFIG["fvg_min_atr"]
    for i in range(2, len(df)):
        if lo[i] > hi[i - 2] and lo[i] - hi[i - 2] >= mn * a[i]:
            out.append(Zone("FVG", "bull", lo[i], hi[i - 2], i, df.index[i], tf))
        elif hi[i] < lo[i - 2] and lo[i - 2] - hi[i] >= mn * a[i]:
            out.append(Zone("FVG", "bear", lo[i - 2], hi[i], i, df.index[i], tf))
    extra = []
    for z in out:
        inv = _track(z, df, z.idx + 1)
        if inv:
            extra.append(inv)
    return out + extra


def find_order_blocks(df: pd.DataFrame, events: list, tf: str):
    """OB = última vela contraria antes del impulso que rompió estructura."""
    o, c, hi, lo = (df[x].values for x in ("open", "close", "high", "low"))
    obs = []
    for ev in events:
        a, b = ev.swing_idx, ev.idx
        if b <= a:
            continue
        if ev.direction == "bull":
            origin = a + int(np.argmin(lo[a:b + 1]))
            k = next((x for x in range(origin, max(origin - 6, -1), -1) if c[x] < o[x]), origin)
            z = Zone("OB", "bull", hi[k], lo[k], k, df.index[k], tf)
        else:
            origin = a + int(np.argmax(hi[a:b + 1]))
            k = next((x for x in range(origin, max(origin - 6, -1), -1) if c[x] > o[x]), origin)
            z = Zone("OB", "bear", hi[k], lo[k], k, df.index[k], tf)
        _track(z, df, b + 1)
        obs.append(z)
    return obs


def analyze_tf(df: pd.DataFrame, tf: str) -> TFAnalysis:
    df = add_indicators(df)
    sh, sl, events, trend = analyze_structure(df, CONFIG["swing_len"][tf])
    fvgs = find_fvgs(df, tf)
    obs = find_order_blocks(df, events, tf)
    return TFAnalysis(tf, df, sh, sl, events, trend, fvgs, obs)


# ─────────────────────────────── B. LIQUIDEZ ─────────────────────────────────
def prev_day_week(d1: pd.DataFrame):
    pdh, pdl = d1["high"].iloc[-1], d1["low"].iloc[-1]
    wk = d1.index.normalize() - pd.to_timedelta(d1.index.weekday, unit="D")
    g = d1.groupby(wk)
    weeks = g.agg({"high": "max", "low": "min"})
    cur_week = wk[-1]
    weeks = weeks[weeks.index < cur_week]
    pwh, pwl = (weeks["high"].iloc[-1], weeks["low"].iloc[-1]) if len(weeks) else (np.nan, np.nan)
    return pdh, pdl, pwh, pwl, d1.index[-1]


def session_levels(m15: pd.DataFrame):
    """Máximo/mínimo de Asia (20:00–00:00 NY) y Londres (02:00–05:00 NY) más recientes."""
    loc = m15.index.tz_convert(NY)
    out = {}
    now = loc[-1]
    for name, (h0, h1) in {"Asia": (20, 24), "Londres": (2, 5)}.items():
        mask = (loc.hour >= h0) & (loc.hour < h1)
        sub = m15[mask & (loc >= now - pd.Timedelta(hours=24))]
        if len(sub):
            out[f"{name} High"] = (sub["high"].max(), sub["high"].idxmax())
            out[f"{name} Low"] = (sub["low"].min(), sub["low"].idxmin())
    return out


def equal_levels(swings: list, atr_v: float, kind: str, tf: str):
    tol = CONFIG["eq_tol_atr"] * atr_v
    out = []
    recent = swings[-12:]
    for i in range(len(recent)):
        for j in range(i + 1, len(recent)):
            if abs(recent[i].price - recent[j].price) <= tol:
                p = max(recent[i].price, recent[j].price) if kind == "high" else min(recent[i].price, recent[j].price)
                out.append(Level(p, f"Equal {'Highs' if kind == 'high' else 'Lows'} {tf}",
                                 "BSL" if kind == "high" else "SSL", recent[j].time))
    return out


def liquidity_pools(A: dict):
    d1, m15 = A["1d"].df, A["15m"].df
    pdh, pdl, pwh, pwl, pd_time = prev_day_week(d1)
    lv = [Level(pdh, "PDH (máx. día anterior)", "BSL", pd_time),
          Level(pdl, "PDL (mín. día anterior)", "SSL", pd_time)]
    if not np.isnan(pwh):
        lv += [Level(pwh, "PWH (máx. semana anterior)", "BSL", pd_time - pd.Timedelta(days=1)),
               Level(pwl, "PWL (mín. semana anterior)", "SSL", pd_time - pd.Timedelta(days=1))]
    for name, (p, t) in session_levels(m15).items():
        lv.append(Level(p, name, "BSL" if "High" in name else "SSL", t))
    for tf in ("4h", "1h", "15m"):
        T = A[tf]
        atr_v = T.df["atr"].iloc[-1]
        for s in T.swings_h[-8:]:
            lv.append(Level(s.price, f"Swing High {tf}", "BSL", s.time))
        for s in T.swings_l[-8:]:
            lv.append(Level(s.price, f"Swing Low {tf}", "SSL", s.time))
        if tf in ("1h", "15m"):
            lv += equal_levels(T.swings_h, atr_v, "high", tf)
            lv += equal_levels(T.swings_l, atr_v, "low", tf)
    return lv


def untouched(level: Level, m15: pd.DataFrame, upto: pd.Timestamp) -> bool:
    """¿La liquidez seguía intacta entre su formación y 'upto'?"""
    sub = m15[(m15.index > level.time) & (m15.index < upto)]
    if level.side == "SSL":
        return len(sub) == 0 or sub["low"].min() > level.price
    return len(sub) == 0 or sub["high"].max() < level.price


def find_sweep(levels: list, m15: pd.DataFrame, side: str):
    """Barrido: mecha más allá del nivel y CIERRE de vuelta dentro (misma vela o la siguiente)."""
    hi, lo, c = m15["high"].values, m15["low"].values, m15["close"].values
    n = len(m15)
    best = None
    for k in range(max(0, n - CONFIG["sweep_lookback"]), n):
        t = m15.index[k]
        for L in levels:
            if L.side != side or L.time >= t - pd.Timedelta(minutes=45):
                continue
            nxt = c[k + 1] if k + 1 < n else c[k]
            if side == "SSL" and lo[k] < L.price and (c[k] > L.price or nxt > L.price):
                if untouched(L, m15, t):
                    cand = dict(level=L, idx=k, time=t, extreme=lo[k:min(k + 3, n)].min())
                    best = cand if best is None or k >= best["idx"] else best
            if side == "BSL" and hi[k] > L.price and (c[k] < L.price or nxt < L.price):
                if untouched(L, m15, t):
                    cand = dict(level=L, idx=k, time=t, extreme=hi[k:min(k + 3, n)].max())
                    best = cand if best is None or k >= best["idx"] else best
    return best


# ─────────────────────────────── E. EMAs ─────────────────────────────────────
def ema_read(df: pd.DataFrame, direction: str):
    last = df.iloc[-1]
    px = last["close"]
    above = [p for p in EMA_PERIODS if px > last[f"ema{p}"]]
    stacked_bull = all(last[f"ema{a}"] > last[f"ema{b}"] for a, b in zip(EMA_PERIODS, EMA_PERIODS[1:]))
    stacked_bear = all(last[f"ema{a}"] < last[f"ema{b}"] for a, b in zip(EMA_PERIODS, EMA_PERIODS[1:]))
    slope50 = last["ema50"] - df["ema50"].iloc[-6]
    slope200 = last["ema200"] - df["ema200"].iloc[-6]
    if direction == "bull":
        ok = px > last["ema50"] and px > last["ema200"] and slope50 > 0
        contra = px < last["ema200"] and slope200 < 0
    else:
        ok = px < last["ema50"] and px < last["ema200"] and slope50 < 0
        contra = px > last["ema200"] and slope200 > 0
    txt = (f"precio sobre {len(above)}/5 EMAs; EMA50 {'↗' if slope50 > 0 else '↘'}, "
           f"EMA200 {'↗' if slope200 > 0 else '↘'}"
           + ("; EMAs alineadas alcistas" if stacked_bull else "; EMAs alineadas bajistas" if stacked_bear else "; EMAs entrelazadas"))
    return ok, contra, txt


# ─────────────────────────────── KILL ZONES / PO3 / SMT ──────────────────────
def kill_zone(ts: pd.Timestamp):
    t = ts.tz_convert(NY)
    m = t.hour * 60 + t.minute
    zones = [("Asia", 20 * 60, 24 * 60), ("London Kill Zone", 2 * 60, 5 * 60),
             ("New York AM Kill Zone", 7 * 60, 10 * 60), ("London Close", 10 * 60, 12 * 60),
             ("New York PM", 13 * 60 + 30, 16 * 60)]
    for name, a, b in zones:
        if a <= m < b:
            return name, name in ("London Kill Zone", "New York AM Kill Zone")
    return "Fuera de kill zone", False


def midnight_open(m15: pd.DataFrame):
    loc = m15.index.tz_convert(NY)
    today = loc[-1].normalize()
    sub = m15[loc >= today]
    return (sub["open"].iloc[0], sub) if len(sub) else (np.nan, sub)


def smt_divergence(a15: TFAnalysis, b15: TFAnalysis | None, direction: str):
    """BTC vs ETH: uno barre su extremo previo y el otro no."""
    if b15 is None:
        return None
    lb = CONFIG["sweep_lookback"]

    def took(T: TFAnalysis):
        df = T.df
        cut = df.index[-lb]
        if direction == "bull":
            prev = [s for s in T.swings_l if s.time < cut]
            return bool(prev) and df["low"].iloc[-lb:].min() < prev[-1].price
        prev = [s for s in T.swings_h if s.time < cut]
        return bool(prev) and df["high"].iloc[-lb:].max() > prev[-1].price

    ta, tb = took(a15), took(b15)
    return ta != tb


# ─────────────────────────────── MOTOR DE SEÑALES ────────────────────────────
@dataclass
class Signal:
    symbol: str
    direction: str
    valid: bool
    checklist: list = field(default_factory=list)      # (ok, texto)
    confluences: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    entry: float | None = None
    entry_zone: Zone | None = None
    sl: float | None = None
    tp1: tuple | None = None
    tp2: tuple | None = None
    rr: float | None = None
    score: int = 0
    key: str = ""


def dealing_range(T4: TFAnalysis):
    if not T4.swings_h or not T4.swings_l:
        return None
    hi, lo = T4.swings_h[-1].price, T4.swings_l[-1].price
    if hi <= lo:
        hi = max(s.price for s in T4.swings_h[-3:])
        lo = min(s.price for s in T4.swings_l[-3:])
    return lo, hi, (lo + hi) / 2


def evaluate(symbol: str, A: dict, direction: str, peer15: TFAnalysis | None) -> Signal:
    bull = direction == "bull"
    S = Signal(symbol, direction, False)
    m15, m5 = A["15m"].df, A["5m"].df
    px = m5["close"].iloc[-1]
    atr15 = m15["atr"].iloc[-1]
    atr1d = A["1d"].df["atr"].iloc[-1]
    chk = S.checklist

    # 1. Contexto HTF ---------------------------------------------------------
    t1d, t4h, t1h = A["1d"].trend, A["4h"].trend, A["1h"].trend
    dr = dealing_range(A["4h"])
    in_discount = dr is not None and (px < dr[2] if bull else px > dr[2])
    htf_zones = [z for z in A["4h"].fvgs + A["4h"].obs + A["1d"].fvgs + A["1d"].obs
                 if z.direction == direction and not z.invalidated
                 and z.bottom - 0.5 * atr15 <= px <= z.top + 0.5 * atr15]
    trend_ok = t4h == direction
    zone_ok = in_discount and bool(htf_zones)
    opp = "bear" if bull else "bull"
    if t1d == opp and t4h == opp:
        chk.append((False, f"Contexto: 1D y 4H en contra ({tf_label(t1d)}) — no se opera contra la estructura principal"))
        return S
    ctx_ok = trend_ok or zone_ok
    ctx_txt = f"1D {tf_label(t1d)} · 4H {tf_label(t4h)} · 1H {tf_label(t1h)}"
    if dr:
        ctx_txt += f" · rango 4H {fmt(dr[0])}–{fmt(dr[1])} (EQ {fmt(dr[2])}) → precio en {'DISCOUNT' if px < dr[2] else 'PREMIUM'}"
    chk.append((ctx_ok, "Contexto HTF: " + ctx_txt))
    if htf_zones:
        S.confluences.append("Precio dentro de " + htf_zones[-1].name())
    if t1d and t4h and t1d != t4h:
        S.warnings.append(f"Conflicto de temporalidades: 1D {tf_label(t1d)} vs 4H {tf_label(t4h)}. Exigir confirmación completa.")
    if in_discount:
        S.confluences.append(f"Precio en zona {'discount' if bull else 'premium'} del rango 4H")
    elif dr:
        S.warnings.append(f"Precio en {'premium' if bull else 'discount'} del rango 4H: zona menos favorable para {'comprar' if bull else 'vender'}.")

    # 2. Barrido de liquidez ---------------------------------------------------
    levels = liquidity_pools(A)
    sweep = find_sweep(levels, m15, "SSL" if bull else "BSL")
    if sweep:
        chk.append((True, f"Barrido de {'SSL' if bull else 'BSL'}: {sweep['level'].name} en {fmt(sweep['level'].price)} "
                          f"({sweep['time'].tz_convert(NY):%d/%m %H:%M} NY), extremo {fmt(sweep['extreme'])}"))
    else:
        support = [z for z in htf_zones]
        chk.append((bool(support), "Barrido de liquidez: no detectado" +
                    (" — pero hay reacción en zona HTF" if support else "")))
    anchor_t = sweep["time"] if sweep else m15.index[-CONFIG["sweep_lookback"]]

    # 3 + 4. Desplazamiento y BOS/MSS en 15M (o 5M) ---------------------------
    mss = None
    for T in (A["15m"], A["5m"]):
        evs = [e for e in T.events if e.direction == direction and e.time > anchor_t
               and e.time <= anchor_t + pd.Timedelta(minutes=15 * CONFIG["mss_window"])]
        if evs:
            disp = [e for e in evs if e.displacement]
            mss = (T.tf, (disp or evs)[0])
            break
    if mss:
        tf, ev = mss
        name = "MSS" if (sweep and ev.kind == "CHoCH") else ev.kind
        chk.append((ev.displacement, f"Desplazamiento {'alcista' if bull else 'bajista'}: "
                    + ("vela(s) impulsiva(s) confirmadas" if ev.displacement else "ruptura sin impulso claro")))
        chk.append((True, f"{name} {'alcista' if bull else 'bajista'} en {tf} al cierre ({fmt(ev.level)}, "
                          f"{ev.time.tz_convert(NY):%d/%m %H:%M} NY)"))
    else:
        chk.append((False, "Desplazamiento: sin impulso posterior"))
        chk.append((False, f"BOS/MSS {'alcista' if bull else 'bajista'} en 15M/5M: no confirmado"))
        return _finish_signal(S)

    # 5. Zona de entrada (FVG / OB / IFVG) ------------------------------------
    leg_start = sweep["time"] if sweep else anchor_t
    cands = []
    for T in (A["15m"], A["5m"]):
        for z in T.fvgs + T.obs:
            if z.direction != direction or z.invalidated or z.time < leg_start - pd.Timedelta(minutes=30):
                continue
            if bull and z.bottom > px:          # zona por encima del precio: ya no sirve para comprar
                continue
            if not bull and z.top < px:
                continue
            cands.append(z)
    pref = {"FVG": 0, "OB": 1, "IFVG": 2}
    cands.sort(key=lambda z: (z.mitigated, pref[z.kind], 0 if z.tf == "15m" else 1, -z.idx))
    zone = cands[0] if cands else None
    if not zone:
        chk.append((False, "Zona de entrada (FVG/OB): ninguna válida tras el desplazamiento"))
        return _finish_signal(S)
    chk.append((True, f"Zona de entrada: {zone.name()}" + (" (ya tocada una vez)" if zone.mitigated else " (sin mitigar)")))
    S.entry_zone = zone
    S.entry = zone.mid if zone.kind != "OB" else (zone.top + zone.mid) / 2
    if bull:
        S.entry = min(S.entry, px)
    else:
        S.entry = max(S.entry, px)

    # 6. Invalidación -----------------------------------------------------------
    buf = CONFIG["sl_buffer_atr"] * atr15
    if bull:
        ref = min(zone.bottom, sweep["extreme"] if sweep else zone.bottom)
        S.sl = ref - buf
    else:
        ref = max(zone.top, sweep["extreme"] if sweep else zone.top)
        S.sl = ref + buf
    chk.append((True, f"Invalidación: {'cierre por debajo' if bull else 'cierre por encima'} de {fmt(ref)} → SL {fmt(S.sl)}"))

    # 7 + 8. Objetivo en liquidez y R:B neto ----------------------------------
    cost = S.entry * CONFIG["costs_pct"] / 100
    risk = abs(S.entry - S.sl) + cost
    max_d = CONFIG["max_target_atr_1d"] * atr1d
    tside = "BSL" if bull else "SSL"
    tgts = {}
    for L in levels:
        if L.side != tside:
            continue
        d = (L.price - S.entry) if bull else (S.entry - L.price)
        if d <= 0 or d > max_d or not untouched(L, m15, m15.index[-1] + pd.Timedelta(minutes=1)):
            continue
        key = round(L.price, 2)
        tgts.setdefault(key, L)
    tg = []
    for L in sorted(tgts.values(), key=lambda L: L.price, reverse=not bull):
        if tg and abs(L.price - tg[-1].price) < 0.5 * atr15:   # niveles casi iguales = un solo pool
            continue
        tg.append(L)
    rr_of = lambda L: (abs(L.price - S.entry) - cost) / risk  # noqa: E731
    main = next((L for L in tg if rr_of(L) >= CONFIG["min_rr"]), None)
    if not tg:
        chk.append((False, f"Objetivo de liquidez {'arriba' if bull else 'abajo'}: no hay pools intactos cercanos"))
        return _finish_signal(S)
    first = tg[0]
    if main is None:
        chk.append((True, f"Objetivo lógico: {first.name} {fmt(first.price)}"))
        chk.append((False, f"R:B neto insuficiente ({rr_of(first):.2f} < {CONFIG['min_rr']}) tras costes de {CONFIG['costs_pct']}%"))
        return _finish_signal(S)
    if first is not main and rr_of(first) >= 1.0:
        S.tp1, S.tp2 = (first.price, first.name), (main.price, main.name)
    else:
        S.tp1 = (main.price, main.name)
        nxt = next((L for L in tg if (L.price > main.price if bull else L.price < main.price)), None)
        S.tp2 = (nxt.price, nxt.name) if nxt else None
    S.rr = rr_of(main)
    chk.append((True, f"Objetivo lógico: {main.name} {fmt(main.price)}"))
    chk.append((True, f"R:B neto {S.rr:.2f} (≥ {CONFIG['min_rr']}) incluyendo costes {CONFIG['costs_pct']}%"))

    # Confluencias adicionales --------------------------------------------------
    lo_leg = sweep["extreme"] if sweep else m15[m15.index >= leg_start]["low"].min()
    hi_leg = sweep["extreme"] if sweep else m15[m15.index >= leg_start]["high"].max()
    leg = m15[m15.index >= leg_start]
    if bull:
        top = leg["high"].max()
        o62, o79 = top - 0.62 * (top - lo_leg), top - 0.79 * (top - lo_leg)
        if o79 <= S.entry <= o62 or (zone.bottom <= o62 and zone.top >= o79):
            S.confluences.append(f"Entrada en OTE 62–79% ({fmt(o79)}–{fmt(o62)})")
    else:
        bot = leg["low"].min()
        o62, o79 = bot + 0.62 * (hi_leg - bot), bot + 0.79 * (hi_leg - bot)
        if o62 <= S.entry <= o79 or (zone.bottom <= o79 and zone.top >= o62):
            S.confluences.append(f"Entrada en OTE 62–79% ({fmt(o62)}–{fmt(o79)})")

    for tf in ("4h", "1h"):
        ok, contra, txt = ema_read(A[tf].df, direction)
        if ok:
            S.confluences.append(f"EMAs {tf} a favor: {txt}")
        elif contra:
            S.warnings.append(f"EMAs {tf} en contra: {txt}")
    last15 = A["15m"].df.iloc[-1]
    near = [p for p in EMA_PERIODS if abs(last15[f"ema{p}"] - S.entry) <= 0.4 * atr15]
    if near:
        S.confluences.append("EMA " + "/".join(map(str, near)) + " 15M en la zona de entrada")

    kz, good = kill_zone(m15.index[-1] + pd.Timedelta(minutes=15))
    if good:
        S.confluences.append(f"Dentro de {kz}")
    else:
        S.warnings.append(f"{kz}: menor probabilidad de desplazamiento institucional.")

    mo, today = midnight_open(m15)
    if not np.isnan(mo) and len(today):
        if bull and S.entry < mo:
            S.confluences.append(f"PO3: entrada por debajo del Midnight Open ({fmt(mo)})")
        if not bull and S.entry > mo:
            S.confluences.append(f"PO3: entrada por encima del Midnight Open ({fmt(mo)})")
        if bull and today["low"].min() < mo and px > mo:
            S.confluences.append("Judas Swing: manipulación bajo la apertura y recuperación")
        if not bull and today["high"].max() > mo and px < mo:
            S.confluences.append("Judas Swing: manipulación sobre la apertura y rechazo")

    smt = smt_divergence(A["15m"], peer15, direction)
    if smt:
        S.confluences.append(f"SMT Divergence {'alcista' if bull else 'bajista'} BTC/ETH en 15M")

    if zone.kind == "IFVG":
        S.confluences.append("IFVG: FVG contrario invalidado y ahora actuando como soporte/resistencia")

    return _finish_signal(S)


def _finish_signal(S: Signal) -> Signal:
    required_ok = len(S.checklist) >= 8 and all(ok for ok, _ in S.checklist)
    conflict = any("Conflicto" in w for w in S.warnings)
    need = CONFIG["min_confluence"] + (1 if conflict else 0)
    S.score = sum(ok for ok, _ in S.checklist) + len(S.confluences)
    net = len(S.confluences) - len(S.warnings)
    S.valid = required_ok and len(S.confluences) >= need and net >= 2
    if required_ok and len(S.confluences) < need:
        S.warnings.append(f"Solo {len(S.confluences)} confluencias extra (mínimo {need}). Calidad insuficiente.")
    elif required_ok and net < 2:
        S.warnings.append("Demasiadas advertencias frente a las confluencias: confianza BAJA, no se emite señal.")
    if S.entry_zone is not None:
        S.key = f"{S.symbol}:{S.direction}:{S.entry_zone.tf}:{S.entry_zone.kind}:{S.entry_zone.time.isoformat()}"
    return S


# ─────────────────────────────── INFORME ─────────────────────────────────────
def confidence(S: Signal) -> str:
    n = len(S.confluences) - len(S.warnings)
    return "ALTA" if n >= 4 else "MEDIA" if n >= 2 else "BAJA"


def report(symbol: str, A: dict, sigs: list) -> str:
    px = A["5m"].df["close"].iloc[-1]
    now = A["5m"].df.index[-1].tz_convert(NY)
    head = (f"📊 {symbol} · {fmt(px)} · {now:%d/%m/%Y %H:%M} NY\n"
            f"Estructura: 1D {tf_label(A['1d'].trend)} | 4H {tf_label(A['4h'].trend)} | "
            f"1H {tf_label(A['1h'].trend)} | 15M {tf_label(A['15m'].trend)}\n")
    valid = [s for s in sigs if s.valid]
    if len(valid) > 1:
        return head + "\n⛔ NO OPERAR — señales LONG y SHORT simultáneas: el mercado está indeciso.\n"
    if not valid:
        best = max(sigs, key=lambda s: (sum(ok for ok, _ in s.checklist), s.score))
        lines = [head, "⛔ NO OPERAR — no hay una oportunidad suficientemente clara.\n",
                 f"Escenario más cercano: {'LONG' if best.direction == 'bull' else 'SHORT'}"]
        lines += [f"  {'✅' if ok else '❌'} {t}" for ok, t in best.checklist]
        if best.warnings:
            lines += ["⚠️ " + w for w in best.warnings]
        lines.append("\nQué esperar: que se cumplan los puntos ❌ antes de considerar una entrada.")
        return "\n".join(lines)

    S = valid[0]
    side = "🟢 LONG (COMPRA)" if S.direction == "bull" else "🔴 SHORT (VENTA)"
    risk_pct = abs(S.entry - S.sl) / S.entry * 100
    lines = [head, f"{side} — confianza {confidence(S)}",
             f"Entrada (límite): {fmt(S.entry)}   zona {fmt(S.entry_zone.bottom)} – {fmt(S.entry_zone.top)}",
             f"Stop Loss: {fmt(S.sl)}  ({risk_pct:.2f}% de distancia)",
             f"TP1: {fmt(S.tp1[0])}  ({S.tp1[1]})"]
    if S.tp2:
        lines.append(f"TP2: {fmt(S.tp2[0])}  ({S.tp2[1]})")
    lines.append(f"R:B neto al objetivo principal: 1:{S.rr:.2f}\n")
    lines.append("Checklist:")
    lines += [f"  ✅ {t}" for _, t in S.checklist]
    lines.append("Confluencias:")
    lines += [f"  ➕ {c}" for c in S.confluences]
    if S.warnings:
        lines.append("Advertencias:")
        lines += [f"  ⚠️ {w}" for w in S.warnings]
    lines += ["",
              "Cómo interpretarla: espera a que el precio regrese a la zona de entrada; "
              "si cierra más allá del SL la idea queda invalidada. En TP1 se puede asegurar parcial "
              "y mover el stop a break-even. Arriesga solo un % pequeño de tu cuenta (p. ej. 0,5–1%).",
              "⚠️ Análisis automático, no asesoramiento financiero."]
    return "\n".join(lines)


# ─────────────────────────────── ORQUESTACIÓN ────────────────────────────────
def load_all(symbol: str, source=None):
    """Descarga TODAS las temporalidades de una misma fuente (coherencia de precios)."""
    tfs = CONFIG["timeframes"]
    if source is not None:
        return {tf: analyze_tf(source(symbol, tf, n), tf) for tf, n in tfs.items()}, ("Datos simulados", "DEMO")
    forced = (os.getenv("DATA_SOURCE") or "").lower()
    errors = []
    for name, fn, tv in SOURCES:
        if forced and forced not in name.lower():
            continue
        try:
            raw = {tf: fn(symbol, tf, n) for tf, n in tfs.items()}
            short = [tf for tf, d in raw.items() if len(d) < 210]
            if short:
                raise RuntimeError(f"pocas velas en {short}")
            return {tf: analyze_tf(d, tf) for tf, d in raw.items()}, (name, tv)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{name}: {str(e)[:120]}")
    raise RuntimeError(" | ".join(errors))


def run_once(source=None):
    data, src, out = {}, {}, []
    for sym in CONFIG["symbols"]:
        try:
            data[sym], src[sym] = load_all(sym, source)
        except Exception as e:  # noqa: BLE001
            out.append((sym, f"❗ {sym}: error obteniendo datos → {e}", []))
    for sym, A in data.items():
        peer = next((data[p]["15m"] for p in data if p != sym), None)
        sigs = [evaluate(sym, A, d, peer) for d in ("bull", "bear")]
        name, tv = src[sym]
        text = report(sym, A, sigs) + f"\n\nDatos: {name} · gráfico en TradingView: {tv}:{sym}"
        out.append((sym, text, [s for s in sigs if s.valid]))
    return out


def send_telegram(text: str) -> None:
    tok, chat = os.getenv("TELEGRAM_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not tok or not chat:
        return
    body = urllib.parse.urlencode({"chat_id": chat, "text": text[:4000]}).encode()
    try:
        urllib.request.urlopen(f"https://api.telegram.org/bot{tok}/sendMessage", data=body, timeout=15)
    except Exception as e:  # noqa: BLE001
        print(f"[telegram] error: {e}", file=sys.stderr)


def load_state() -> dict:
    try:
        with open(CONFIG["state_file"], encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return {"sent": []}


def save_state(st: dict) -> None:
    st["sent"] = st["sent"][-200:]
    with open(CONFIG["state_file"], "w", encoding="utf-8") as f:
        json.dump(st, f)


def main():
    ap = argparse.ArgumentParser(description="Bot de señales ICT/SMC para BTC y ETH (solo avisos).")
    ap.add_argument("--once", action="store_true", help="un análisis y salir")
    ap.add_argument("--demo", action="store_true", help="usar datos simulados (sin internet)")
    ap.add_argument("--notify-all", action="store_true", help="enviar también los 'NO OPERAR' a Telegram")
    args = ap.parse_args()

    source = None
    if args.demo:
        from demo_data import synthetic_source  # noqa: PLC0415
        source = synthetic_source

    state = load_state()
    while True:
        results = run_once(source)
        if all(t.startswith("❗") for _, t, _ in results):
            for _, t, _ in results:
                print(t, file=sys.stderr)
            if args.notify_all:
                send_telegram("\n\n".join(t for _, t, _ in results))
            if args.once:
                sys.exit(1)          # en GitHub Actions marca la ejecución como fallida
        for sym, text, valid in results:
            print("\n" + "═" * 60 + "\n" + text)
            new = [s for s in valid if s.key not in state["sent"]]
            if new:
                send_telegram(text)
                state["sent"] += [s.key for s in new]
            elif args.notify_all:
                send_telegram(text)
        save_state(state)
        if args.once or args.demo:
            break
        wait = CONFIG["loop_minutes"] * 60
        nxt = (int(time.time()) // wait + 1) * wait + 5     # justo tras el cierre de vela
        time.sleep(max(5, nxt - time.time()))


if __name__ == "__main__":
    main()
