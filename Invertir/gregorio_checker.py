#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 18:55:26 2025

@author: alex
"""

# gregorio_checker.py
# Dashboard "estilo Gregorio" para evaluar un ticker por dividendos crecientes

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import os
import json
import hashlib
import requests
import streamlit as st
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
import plotly.graph_objects as go
import plotly.express as px
import re, urllib.request


def get_groq_api_key() -> str:
    """
    Obtiene la clave de Groq desde:
    1) .streamlit/secrets.toml
    2) Variable de entorno GROQ_API_KEY
    """
    try:
        return st.secrets["groq_api_key"]
        
    except Exception:
        return os.getenv("groq_api_key", "").strip()

groq_api_key = get_groq_api_key()

# --- Estado de sesión para no re-ejecutar fetch al cambiar controles de gráfico ---
if "metrics" not in st.session_state:
    st.session_state.metrics = None
if "divs_year" not in st.session_state:
    st.session_state.divs_year = None
if "price_series" not in st.session_state:
    st.session_state.price_series = None
if "ccy_symbol" not in st.session_state:
    st.session_state.ccy_symbol = "$"

@st.cache_data(ttl=3600)
def _cached_fetch(ticker: str):
    return fetch_metrics(ticker)



@st.cache_data(ttl=24*3600, show_spinner=False)
def load_sp500_tickers() -> list[str]:
    url = "https://en.wikipedia.org/w/index.php?title=List_of_S%26P_500_companies&action=raw"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"}  # importante para evitar bloqueos
    )
    raw = urllib.request.urlopen(req, timeout=20).read().decode("utf-8", errors="ignore")

    # En el raw hay una tabla wiki; los tickers suelen aparecer como '|| AAPL' etc.
    # Capturamos tokens tipo BRK.B, BF.B, AAPL, etc.
    tickers = sorted(set(re.findall(r"\|\s*([A-Z]{1,5}(?:\.[A-Z])?)\s*\n", raw)))

    # Ajuste yfinance: BRK.B -> BRK-B
    tickers = [t.replace(".", "-") for t in tickers]

    # Filtra basura (por si aparecen siglas extrañas)
    tickers = [t for t in tickers if 1 <= len(t) <= 6 and t.replace("-", "").isalpha()]
    return tickers

@st.cache_data(ttl=24*3600, show_spinner=False)
def load_dividend_aristocrats_tickers() -> list[str]:
    """
    Carga los tickers de los S&P 500 Dividend Aristocrats desde Wikipedia (RAW),
    evitando errores 403.
    """
    url = "https://en.wikipedia.org/w/index.php?title=S%26P_500_Dividend_Aristocrats&action=raw"

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    raw = urllib.request.urlopen(req, timeout=20).read().decode("utf-8", errors="ignore")

    # En el raw, los tickers aparecen normalmente en tablas wiki como:
    # || Company || Ticker || Sector || ...
    # Capturamos patrones tipo AAPL, JNJ, BRK.B, etc.
    tickers = re.findall(r"\|\s*([A-Z]{1,5}(?:\.[A-Z])?)\s*\n", raw)

    # Normaliza para yfinance: BRK.B -> BRK-B
    tickers = [t.replace(".", "-") for t in tickers]

    # Limpieza defensiva
    tickers = sorted(set(
        t for t in tickers
        if 1 <= len(t) <= 6 and t.replace("-", "").isalpha()
    ))

    return tickers

@st.cache_data(ttl=6*3600, show_spinner=False)
def get_basic_yf_snapshot(ticker: str) -> dict:
    """Devuelve marketCap y dividendRate de yfinance.info con cache."""
    try:
        info = yf.Ticker(ticker).info or {}
        return {
            "ticker": ticker,
            "marketCap": info.get("marketCap", np.nan),
            "dividendRate": info.get("dividendRate", np.nan),  # >0 => dividend payer
        }
    except Exception:
        return {"ticker": ticker, "marketCap": np.nan, "dividendRate": np.nan}

def filter_top_dividend_payers(tickers: list[str], top_n: int = 200, max_workers: int = 12) -> list[str]:
    """
    Filtra tickers:
      1) dividendRate > 0 (dividend payer)
      2) Top N por marketCap
    Nota: marketCap viene de yfinance y puede faltar en algunos tickers.
    """
    rows = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(get_basic_yf_snapshot, tk): tk for tk in tickers}
        for fut in as_completed(futures):
            rows.append(fut.result())

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["marketCap", "dividendRate"])
    df = df[(df["marketCap"] > 0) & (df["dividendRate"] > 0)]
    df = df.sort_values("marketCap", ascending=False).head(top_n)
    return df["ticker"].tolist()


GROQ_BASE_URL = "https://api.groq.com/openai/v1"  # OpenAI-compatible :contentReference[oaicite:3]{index=3}

def _hash_key(*parts) -> str:
    raw = "||".join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

@st.cache_data(ttl=7 * 24 * 3600, show_spinner=False)
def groq_chat_cached(cache_key: str, api_key: str, model: str, messages: list, temperature: float = 0.2, max_tokens: int = 700) -> str:
    """
    Caché por clave: si el prompt/model no cambia, no vuelve a llamar a Groq.
    OJO: cache_key ya incluye el contenido. api_key no se usa dentro del hash, solo para ejecutar.
    """
    url = f"{GROQ_BASE_URL}/chat/completions"  # :contentReference[oaicite:4]{index=4}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }

    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

def build_prompt_individual(metrics, rules, total_score, breakdown, flags) -> str:
    # Importante: el prompt prohíbe inventar datos
    return f"""
Eres un analista MUY conservador de inversión a largo plazo por dividendos crecientes.
Solo puedes usar los datos que te doy. NO inventes cifras, NO asumas nada que no esté explícito.
Si faltan datos, dilo como "dato no disponible".

Datos de la empresa:
- Ticker: {metrics.ticker}
- Nombre: {metrics.name}
- Sector: {metrics.sector}
- Bolsa/plaza: {getattr(metrics,'exchange_name','')} ({getattr(metrics,'exchange_code','')})
- País (sede): {getattr(metrics,'country','')}
- Divisa: {getattr(metrics,'currency','')}

Ratios calculados (pueden ser NaN si no disponibles):
- RPD TTM (%): {metrics.rpd_ttm}
- DGR 5a (%): {metrics.dgr5}
- DGR 10a (%): {metrics.dgr10}
- Payout (%): {metrics.payout}
- PER TTM: {metrics.per_ttm}
- Deuda/Patrimonio (x): {metrics.de_ratio}
- ROE (%): {metrics.roe}
- Racha dividendos (años): {metrics.streak_years}
- EV/EBITDA (x): {getattr(metrics,'ev_ebitda', np.nan)}
- FCF Yield (%): {getattr(metrics,'fcf_yield', np.nan)}
- RPD Forward (%): {getattr(metrics,'rpd_forward', np.nan)}

Umbrales del sector (siempre referéncialos al evaluar):
- rpd_min={rules.rpd_min}, rpd_max={rules.rpd_max}
- dgr5_min={rules.dgr5_min}
- payout_min={rules.payout_min}, payout_max={rules.payout_max}
- per_min={rules.per_min}, per_max={rules.per_max}
- de_max={rules.de_max}
- roe_min={rules.roe_min}
- streak_min={rules.streak_min}

Score estilo Gregorio (0-100):
- Total: {total_score}
- Desglose: {breakdown}

Banderas rojas detectadas:
{flags if flags else "Ninguna"}

Tarea:
1) Resumen ejecutivo (5-7 líneas).
2) Diagnóstico del dividendo (RPD + crecimiento + racha + payout): qué está bien y qué vigilar.
3) Solidez financiera (deuda + ROE + FCF yield si existe).
4) Valoración (PER y EV/EBITDA si existe) frente a umbrales.
5) Conclusión: "Apta / Dudosa / No apta" para dividendos crecientes + 3 acciones prácticas (p.ej. esperar mejor precio, comprar DCA, vigilar payout).
Usa viñetas. Si algún dato es NaN, dilo y no concluyas con ese dato.
""".strip()

def build_prompt_ranking(df_sorted: pd.DataFrame) -> str:
    # Reducimos tokens: enviamos solo top 25 y columnas clave
    cols = [c for c in ["Ticker","Nombre","Sector","Score total","RPD (%)","DGR 5a (%)","Payout (%)","PER","Deuda/Patrimonio","ROE (%)"] if c in df_sorted.columns]
    top = df_sorted[cols].head(25).copy()
    table_txt = top.to_csv(index=False)

    return f"""
Eres un analista conservador de inversión a largo plazo por dividendos crecientes.
Solo puedes usar la tabla que te doy. NO inventes datos ni empresas. Si algo no está, dilo.

Esta es la tabla TOP 25 del ranking (CSV):
{table_txt}

Tarea:
1) Resume en 8-12 líneas qué patrones explican las mejores puntuaciones.
2) Señala los 5 tickers más interesantes y por qué (solo usando columnas).
3) Señala 5 riesgos típicos observables en la tabla (payout alto, deuda alta, etc.).
4) Propón un “plan de trabajo” para el inversor: qué revisar a mano antes de comprar.
""".strip()

def build_prompt_sector_summary(df_sorted: pd.DataFrame) -> str:
    cols = [c for c in ["Ticker","Sector","Score total","RPD (%)","DGR 5a (%)","Payout (%)","PER","Deuda/Patrimonio","ROE (%)"] if c in df_sorted.columns]
    # top 3 por sector para limitar tokens
    g = df_sorted[cols].copy()
    by_sector = g.sort_values("Score total", ascending=False).groupby("Sector", as_index=False).head(3)
    table_txt = by_sector.to_csv(index=False)

    return f"""
Eres un analista conservador de dividendos crecientes.
Solo usa la tabla. NO inventes datos.

Top 3 por sector (CSV):
{table_txt}

Tarea:
- Para cada sector, resume en 2-3 viñetas por qué esas empresas salen arriba y qué métrica vigilar.
- Cierra con 5 “reglas rápidas” (heurísticas) para filtrar oportunidades por sector sin caer en trampas.
""".strip()



# ========= Estilo Plotly "finance clean" =========
FINANCE_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, Segoe UI, Roboto, Helvetica, Arial, sans-serif", size=13),
    hovermode="x unified",
    xaxis=dict(
        showspikes=True, spikemode="across", spikesnap="cursor", spikedash="solid", spikethickness=1,
        showgrid=True, gridcolor="rgba(0,0,0,0.06)", zeroline=False
    ),
    yaxis=dict(
        showspikes=True, spikemode="toaxis+across", spikedash="solid", spikethickness=1,
        showgrid=True, gridcolor="rgba(0,0,0,0.06)", zeroline=False
    ),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=50, r=25, t=60, b=40),
)

_CCY_SYMBOL = {
    "USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", "CHF": "Fr", "CAD": "$", "AUD": "$"
}

# === Bolsa / exchange human-friendly ===
YF_EXCHANGE_MAP = {
    "NYQ": "NYSE", "NMS": "NASDAQ", "NGM": "NASDAQ GM", "NCM": "NASDAQ CM",
    "BATS": "Cboe BZX", "ASE": "NYSE American", "PNK": "OTC",
    "TOR": "TSX", "TSX": "TSX", "VAN": "TSX Venture",
    "LSE": "London Stock Exchange", "LIS": "Euronext Lisbon",
    "EPA": "Euronext Paris", "AMS": "Euronext Amsterdam", "BRU": "Euronext Brussels",
    "FRA": "Frankfurt", "GER": "Xetra (Deutsche Börse)", "VIE": "Vienna",
    "SWX": "SIX Swiss", "SAO": "B3 (São Paulo)", "ASX": "ASX",
    "HKSE": "Hong Kong", "MCE": "BME (Madrid)", "MIL": "Borsa Italiana",
    "TSE": "Tokyo", "NSE": "NSE India", "BSE": "BSE India",
}


def _ccy_symbol_from_info(info: dict, fallback: str = "$") -> str:
    code = (info or {}).get("currency", None)
    return _CCY_SYMBOL.get(code, fallback)

def _fmt_hover_money(symbol: str) -> str:
    # Devuelve un template para Plotly con el valor Y formateado con miles y 2 decimales.
    # Ojo: se escapan llaves con {{ }} para que Python no intente interpolar 'y'.
    return f"{symbol} %{{y:,.2f}}"


def dgr_not_reliable(by_year_clean: pd.Series, by_year_raw: pd.Series) -> bool:
    # Muy pocos años completos limpios
    if by_year_clean is None or len(by_year_clean) < 5:
        return True

    # Demasiada diferencia entre crudo y limpio → dividendos irregulares / scrip
    try:
        dgr_clean_5 = cagr_from_annual(by_year_clean, 5)
        dgr_raw_5   = cagr_from_annual(by_year_raw, 5)
        if pd.notna(dgr_clean_5) and pd.notna(dgr_raw_5):
            if abs(dgr_clean_5 - dgr_raw_5) > 0.05:  # >5 puntos porcentuales
                return True
    except Exception:
        pass

    return False



# =============== Utilidades de tiempo y series ===============

def _slice_price_series(s: pd.Series, rango: str, agg: str) -> pd.Series:
    if s is None or s.empty:
        return s
    s = s.dropna()
    # Filtra por rango
    end = s.index.max()
    if rango == "YTD":
        start = pd.Timestamp(year=end.year, month=1, day=1)
    elif rango == "1 año":
        start = end - pd.Timedelta(days=365)
    elif rango == "3 años":
        start = end - pd.DateOffset(years=3)
    elif rango == "5 años":
        start = end - pd.DateOffset(years=5)
    elif rango == "10 años":
        start = end - pd.DateOffset(years=10)
    elif rango == "20 años":
        start = end - pd.DateOffset(years=20)
    else:  # "Todo"
        start = s.index.min()
    s = s[(s.index >= start) & (s.index <= end)]
    # Agregación para que el gráfico vaya fluido
    if agg == "Semanal":
        s = s.resample("W-FRI").last()
    elif agg == "Mensual":
        s = s.resample("M").last()
    return s





def to_naive_utc_index(idx) -> pd.DatetimeIndex:
    idx = pd.to_datetime(idx, utc=True, errors="coerce")
    return idx.tz_localize(None)

def dividends_ttm(divs: pd.Series, ref_date: Optional[pd.Timestamp]=None) -> float:
    if divs is None or divs.empty:
        return 0.0
    divs = divs.copy()
    divs.index = to_naive_utc_index(divs.index)
    ref = pd.to_datetime(ref_date) if ref_date is not None else pd.to_datetime("today").tz_localize(None)
    cutoff = ref - pd.Timedelta(days=365)
    return float(divs[divs.index > cutoff].sum())

def dividends_by_year(divs: pd.Series) -> pd.Series:
    """
    Suma anual de dividendos TOTALES (crudo). Conservamos esta versión para diagnóstico.
    """
    if divs is None or divs.empty:
        return pd.Series(dtype="float64")
    s = divs.copy()
    s.index = to_naive_utc_index(s.index)
    return s.groupby(s.index.year).sum()

def dividends_by_year_clean(divs: pd.Series, expected_payments: int = None) -> pd.Series:
    """
    Suma anual 'limpia':
      - Trabaja por año natural.
      - Descarta 'dividendos especiales' (pagos individuales > 1.5x de la mediana del año).
      - Mantiene solo AÑOS COMPLETOS: años con el número esperado de pagos 'regulares'.
    expected_payments:
      - Si None, lo infiere como la moda del conteo anual de pagos (≈4 en USA trimestrales).
    """
    if divs is None or divs.empty:
        return pd.Series(dtype="float64")

    s = divs.copy()
    s.index = to_naive_utc_index(s.index)
    by_year = s.groupby(s.index.year)

    # Conteos por año para inferir periodicidad
    counts = by_year.size()
    if expected_payments is None:
        if len(counts) == 0:
            expected_payments = 4
        else:
            expected_payments = int(counts.mode().iloc[0])

    cleaned = {}
    for y, grp in by_year:
        vals = grp.sort_index().values.astype(float)
        if len(vals) == 0:
            continue
        # Detecta 'especiales': > 1.5 * mediana
        med = float(np.median(vals))
        regulars = [v for v in vals if v <= 1.5 * med + 1e-12]

        # Si todavía hay más de expected_payments, coge los expected_payments más cercanos a la mediana
        if len(regulars) > expected_payments:
            # ordena por |v - mediana| y toma los expected_payments
            regulars = sorted(regulars, key=lambda v: abs(v - med))[:expected_payments]

        # Consideramos 'año completo' si alcanzamos exactamente expected_payments regulares
        if len(regulars) == expected_payments:
            cleaned[y] = float(np.sum(regulars))
        # si no, lo descartamos como año incompleto

    if not cleaned:
        return pd.Series(dtype="float64")

    ser = pd.Series(cleaned).sort_index()
    return ser


def cagr_from_annual(series_by_year: pd.Series, years: int) -> float:
    """
    CAGR sobre años COMPLETOS y ventana EXACTA:
      - Usa el último AÑO COMPLETO disponible como 'end' (Y-1 si estás a mitad de año).
      - start_year = end_year - years
      - n = years
    Requiere que existan ambos años en la serie; si falta, devuelve NaN.
    """
    s = series_by_year.dropna().sort_index()
    if len(s) < 2:
        return np.nan

    end_year = int(s.index.max())  # último año disponible en la serie
    start_year = end_year - years
    if start_year not in s.index or end_year not in s.index:
        return np.nan

    start_val = float(s.loc[start_year])
    end_val   = float(s.loc[end_year])
    if start_val <= 0 or end_val <= 0:
        return np.nan

    n = years  # exponente correcto
    return (end_val / start_val) ** (1.0 / n) - 1.0


def dividend_streak_years(series_by_year: pd.Series) -> int:
    """Años consecutivos (hasta el último) pagando > 0 (no exige crecimiento, solo pago)."""
    if series_by_year is None or series_by_year.empty:
        return 0
    s = series_by_year.copy()
    years_sorted = sorted(s.index.tolist())
    # contamos desde el final hacia atrás mientras s[year] > 0
    cnt = 0
    for y in reversed(years_sorted):
        if s.loc[y] and s.loc[y] > 0:
            cnt += 1
        else:
            break
    return cnt

# --- Scoring helpers ---

def _band_score(val, min_v=None, max_v=None, prefer_mid=False):
    """
    Normaliza a 0-100.
    - Dos cotas: 100 dentro del rango; si prefer_mid=True, pico en el centro del rango.
    - Solo mínima: 0 si val < min, 100 si val >= min (maneja min_v==0 sin divisiones).
    - Solo máxima: 100 si val <= max, 0 si val > max (maneja max_v==0 sin divisiones).
    - NaN -> 0.
    """
    if pd.isna(val):
        return 0.0

    # Dos cotas
    if min_v is not None and max_v is not None:
        if val < min_v:
            # caída lineal hacia 0 (evita divisiones por 0)
            if min_v == 0:
                return 0.0
            return max(0.0, 100.0 * (val / min_v))
        if val > max_v:
            if val == 0:
                return 0.0
            return max(0.0, 100.0 * (max_v / val))
        # Dentro del rango
        if prefer_mid:
            mid = (min_v + max_v) / 2.0
            half = (max_v - min_v) / 2.0 if (max_v - min_v) != 0 else 1.0
            return max(0.0, 100.0 - 100.0 * abs(val - mid) / half)
        return 100.0

    # Solo mínima
    if min_v is not None:
        return 100.0 if val >= min_v else 0.0

    # Solo máxima
    if max_v is not None:
        return 100.0 if val <= max_v else 0.0

    # Sin cotas
    return 50.0


def red_flags(metrics) -> list:
    flags = []
    # DGR negativo + payout alto
    if pd.notna(metrics.dgr5) and metrics.dgr5 < 0 and pd.notna(metrics.payout) and metrics.payout > 80:
        flags.append("DGR 5a negativo con payout alto")
    # Payout sobre FCF
    if hasattr(metrics, "payout_fcf") and pd.notna(metrics.payout_fcf) and metrics.payout_fcf > 100:
        flags.append("Payout FCF > 100%")
    # FCF negativo 2/3 últimos años (si lo calculas más adelante)
    # Puedes almacenar metrics.fcf_last3 y contar <0
    # Dividendo recortado: si racha de crecimiento = 0 y DGR 5a < 0:
    if hasattr(metrics, "streak_growth") and metrics.streak_growth == 0 and pd.notna(metrics.dgr5) and metrics.dgr5 < 0:
        flags.append("Recortes/ausencia de crecimiento del dividendo")
    return flags

def _rng(a, b, units=""):
    if a is None and b is None:
        return "—"
    if a is None:  # solo máx
        return f"≤ {b}{units}".strip()
    if b is None:  # solo mín
        return f"≥ {a}{units}".strip()
    return f"{a}–{b}{units}".strip()

def score_company(metrics, rules):
    """
    Devuelve:
      - total (0-100)
      - breakdown (dict con 4 bloques)
      - details (dict con tablas por bloque para UI)
    """
    details = {"Dividendo": [], "Solidez": [], "Valoración": [], "Historial": []}

    # ---------- DIVIDENDO (40%) ----------
    # pesos internos del bloque Dividendo
    w_rpd, w_dgr5, w_dgr10, w_payout = 0.35, 0.30, 0.15, 0.20

    rpd_val = metrics.rpd_forward if pd.notna(getattr(metrics, "rpd_forward", np.nan)) else metrics.rpd_ttm
    sc_rpd = _band_score(rpd_val, rules.rpd_min, rules.rpd_max, prefer_mid=True)
    details["Dividendo"].append({
        "Métrica": "RPD (usa forward si hay)",
        "Valor": None if pd.isna(rpd_val) else f"{rpd_val:.2f} %",
        "Rango": _rng(rules.rpd_min, rules.rpd_max, "%"),
        "Peso bloq.": f"{int(w_rpd*100)}%",
        "Sub-score": f"{sc_rpd:.0f}"
    })

    if not getattr(metrics, "dgr_reliable", True):
        sc_dgr5 = 50.0   # crecimiento bajo/irregular, pero no 0
    else:
        sc_dgr5 = _band_score(metrics.dgr5, rules.dgr5_min, None)

    details["Dividendo"].append({
        "Métrica": "DGR 5 años",
        "Valor": "s/d" if pd.isna(metrics.dgr5) else f"{metrics.dgr5:.2f} %",
        "Rango": _rng(rules.dgr5_min, None, "%"),
        "Peso bloq.": f"{int(w_dgr5*100)}%",
        "Sub-score": f"{sc_dgr5:.0f}"
    })

    # Solo exigimos no negativo en DGR10
    sc_dgr10 = _band_score(metrics.dgr10, 0.0, None)
    details["Dividendo"].append({
        "Métrica": "DGR 10 años",
        "Valor": "s/d" if pd.isna(metrics.dgr10) else f"{metrics.dgr10:.2f} %",
        "Rango": "≥ 0 %",
        "Peso bloq.": f"{int(w_dgr10*100)}%",
        "Sub-score": f"{sc_dgr10:.0f}"
    })

    payout_pref = getattr(metrics, "payout_fcf", np.nan)
    use_payout_fcf = pd.notna(payout_pref)
    sc_payout = _band_score(payout_pref if use_payout_fcf else metrics.payout,
                            40.0 if use_payout_fcf else rules.payout_min,
                            70.0 if use_payout_fcf else rules.payout_max,
                            prefer_mid=True)
    details["Dividendo"].append({
        "Métrica": "Payout (FCF si hay, si no contable)",
        "Valor": ("s/d" if (pd.isna(payout_pref) and pd.isna(metrics.payout))
                  else f"{(payout_pref if use_payout_fcf else metrics.payout):.2f} %"),
        "Rango": _rng(40.0 if use_payout_fcf else rules.payout_min,
                      70.0 if use_payout_fcf else rules.payout_max, "%"),
        "Peso bloq.": f"{int(w_payout*100)}%",
        "Sub-score": f"{sc_payout:.0f}"
    })

    score_div = 0.4 * (w_rpd*sc_rpd + w_dgr5*sc_dgr5 + w_dgr10*sc_dgr10 + w_payout*sc_payout)

    # ---------- SOLIDEZ (25%) ----------
    w_debt, w_roe, w_streak_pay, w_streak_grow = 0.35, 0.30, 0.20, 0.15

    sc_debt = _band_score(metrics.de_ratio, None, rules.de_max)
    details["Solidez"].append({
        "Métrica": "Deuda/Patrimonio",
        "Valor": "s/d" if pd.isna(metrics.de_ratio) else f"{metrics.de_ratio:.2f} x",
        "Rango": _rng(None, rules.de_max, "x"),
        "Peso bloq.": f"{int(w_debt*100)}%",
        "Sub-score": f"{sc_debt:.0f}"
    })

    sc_roe = _band_score(metrics.roe, rules.roe_min, None)
    details["Solidez"].append({
        "Métrica": "ROE",
        "Valor": "s/d" if pd.isna(metrics.roe) else f"{metrics.roe:.2f} %",
        "Rango": _rng(rules.roe_min, None, "%"),
        "Peso bloq.": f"{int(w_roe*100)}%",
        "Sub-score": f"{sc_roe:.0f}"
    })

    sc_streak_pay = _band_score(metrics.streak_years, rules.streak_min, None)
    details["Solidez"].append({
        "Métrica": "Racha de pagos",
        "Valor": f"{metrics.streak_years} a" if pd.notna(metrics.streak_years) else "s/d",
        "Rango": _rng(rules.streak_min, None, "a"),
        "Peso bloq.": f"{int(w_streak_pay*100)}%",
        "Sub-score": f"{sc_streak_pay:.0f}"
    })

    sc_streak_g = _band_score(getattr(metrics, "streak_growth", np.nan), 5.0, None)
    details["Solidez"].append({
        "Métrica": "Racha de crecimiento",
        "Valor": ("s/d" if pd.isna(getattr(metrics, "streak_growth", np.nan))
                  else f"{metrics.streak_growth} a"),
        "Rango": "≥ 5 a",
        "Peso bloq.": f"{int(w_streak_grow*100)}%",
        "Sub-score": f"{sc_streak_g:.0f}"
    })

    score_solid = 0.25 * (w_debt*sc_debt + w_roe*sc_roe + w_streak_pay*sc_streak_pay + w_streak_grow*sc_streak_g)

    # ---------- VALORACIÓN (15%) ----------
    if pd.notna(metrics.per_ttm):
        w_per = 1.0
        sc_per = _band_score(metrics.per_ttm, rules.per_min, rules.per_max, prefer_mid=True)
        details["Valoración"].append({
            "Métrica": "PER (ttm)",
            "Valor": f"{metrics.per_ttm:.2f}",
            "Rango": _rng(rules.per_min, rules.per_max, ""),
            "Peso bloq.": "100%",
            "Sub-score": f"{sc_per:.0f}"
        })
        score_val = 0.15 * (w_per*sc_per)
    else:
        w_ev, w_fcfy = 0.6, 0.4
        ev_eb = getattr(metrics, "ev_ebitda", np.nan)
        fcf_y = getattr(metrics, "fcf_yield", np.nan)
        sc_ev  = _band_score(ev_eb, None, 6.0)      # razonable ≤6x (telecos/utilities)
        sc_fcf = _band_score(fcf_y, 8.0, None)      # deseable ≥8–10%
        details["Valoración"].append({
            "Métrica": "EV/EBITDA",
            "Valor": "s/d" if pd.isna(ev_eb) else f"{ev_eb:.2f} x",
            "Rango": "≤ 6.0 x",
            "Peso bloq.": f"{int(w_ev*100)}%",
            "Sub-score": f"{sc_ev:.0f}"
        })
        details["Valoración"].append({
            "Métrica": "FCF Yield",
            "Valor": "s/d" if pd.isna(fcf_y) else f"{fcf_y:.2f} %",
            "Rango": "≥ 8.0 %",
            "Peso bloq.": f"{int(w_fcfy*100)}%",
            "Sub-score": f"{sc_fcf:.0f}"
        })
        score_val = 0.15 * (w_ev*sc_ev + w_fcfy*sc_fcf)

    # ---------- HISTORIAL (20%) ----------
    w_hist_grow, w_hist_fcf = 0.5, 0.5
    sc_hist_grow = _band_score(getattr(metrics, "streak_growth", np.nan), 5.0, None)
    sc_hist_fcf  = _band_score(metrics.fcf_pos_years, 3.0, None)
    details["Historial"].append({
        "Métrica": "Racha de crecimiento",
        "Valor": "s/d" if pd.isna(getattr(metrics, "streak_growth", np.nan)) else f"{metrics.streak_growth} a",
        "Rango": "≥ 5 a",
        "Peso bloq.": f"{int(w_hist_grow*100)}%",
        "Sub-score": f"{sc_hist_grow:.0f}"
    })
    details["Historial"].append({
        "Métrica": "Años con FCF positivo",
        "Valor": "s/d" if pd.isna(metrics.fcf_pos_years) else f"{metrics.fcf_pos_years}",
        "Rango": "≥ 3",
        "Peso bloq.": f"{int(w_hist_fcf*100)}%",
        "Sub-score": f"{sc_hist_fcf:.0f}"
    })
    score_hist = 0.20 * (w_hist_grow*sc_hist_grow + w_hist_fcf*sc_hist_fcf)

    total = score_div + score_solid + score_val + score_hist
    total = max(0.0, min(100.0, total))
    breakdown = {"Dividendo": score_div, "Solidez": score_solid, "Valoración": score_val, "Historial": score_hist}
    return total, breakdown, details


def recommendation(total_score, flags):
    if flags:
        return "Pausa / Revisar banderas rojas", "❌", flags
    if total_score >= 75:
        return "Comprar / Añadir (DCA normal)", "✅", []
    if total_score >= 60:
        return "Vigilar o DCA prudente", "⚠️", []
    return "Mantenerse al margen", "❌", []


def chart_price_line(price_series: pd.Series, title: str, symbol: str):
    s = price_series.dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=s.index, y=s.values, mode="lines", name="Precio",
        hovertemplate=_fmt_hover_money(symbol) + "<extra></extra>"
    ))
    # MM adaptativas
    win_short, win_long = (20, 50)
    if s.index.freqstr in ("M", "MS") or (len(s) > 0 and s.index.inferred_freq == "M"):
        win_short, win_long = (3, 6)      # ~3 y 6 meses
    elif s.index.freqstr and s.index.freqstr.startswith("W"):
        win_short, win_long = (10, 30)    # ~10 y 30 semanas

    for win, name in [(win_short, f"MM{win_short}"), (win_long, f"MM{win_long}")]:
        if len(s) >= win:
            ma = s.rolling(win).mean()
            fig.add_trace(go.Scatter(
                x=ma.index, y=ma.values, mode="lines", name=name, opacity=0.65,
                hovertemplate=_fmt_hover_money(symbol) + "<extra></extra>"
            ))
    apply_finance_layout(fig, title=title, rangeslider=True)
    return fig



def chart_dividends_bar(divs_year: pd.Series, title: str, symbol: str):
    s = divs_year.sort_index()
    fig = go.Figure(go.Bar(
        x=s.index.astype(str), y=s.values, name="Dividendo anual",
        hovertemplate=_fmt_hover_money(symbol) + "<extra></extra>"
    ))
    apply_finance_layout(fig, title=title, rangeslider=False)
    return fig



def apply_finance_layout(fig, title=None, rangeslider=False):
    # Copia superficial del layout base
    layout = dict(FINANCE_LAYOUT)
    if title:
        layout["title"] = title
    if rangeslider:
        # añade rangeslider sin duplicar claves
        xa = dict(layout.get("xaxis", {}))
        xa["rangeslider"] = dict(visible=True)
        layout["xaxis"] = xa
    fig.update_layout(**layout)


# =============== Reglas por sector (umbrales de Gregorio) ===============

@dataclass
class SectorRules:
    rpd_min: float
    rpd_max: float
    dgr5_min: float
    payout_min: float
    payout_max: float
    per_min: float
    per_max: float
    de_max: float
    roe_min: float
    streak_min: int

# Baseline y ajustes por sector (puedes afinar estos umbrales a tu gusto)
BASELINE = SectorRules(
    rpd_min=3.0, rpd_max=6.0,
    dgr5_min=5.0,
    payout_min=30.0, payout_max=70.0,
    per_min=8.0, per_max=18.0,
    de_max=1.0,
    roe_min=10.0,
    streak_min=10
)

SECTOR_TWEAKS: Dict[str, SectorRules] = {
    "Utilities":      SectorRules(3.5, 7.0, 3.0, 40.0, 80.0, 8.0, 20.0, 2.0, 8.0, 8),
    "Communication Services": SectorRules(4.0, 8.0, 2.0, 40.0, 85.0, 7.0, 16.0, 1.5, 8.0, 5),
    "Consumer Defensive": SectorRules(3.0, 6.0, 5.0, 30.0, 70.0, 10.0, 20.0, 1.0, 10.0, 10),
    "Healthcare":     SectorRules(2.5, 5.5, 5.0, 30.0, 65.0, 10.0, 22.0, 1.0, 10.0, 8),
    "Industrials":    SectorRules(2.5, 5.5, 5.0, 25.0, 60.0, 9.0, 18.0, 1.0, 10.0, 7),
    "Technology":     SectorRules(1.5, 4.0, 7.0, 20.0, 55.0, 12.0, 25.0, 0.8, 12.0, 7),
    "Energy":         SectorRules(4.0, 9.0, 0.0, 30.0, 70.0, 6.0, 12.0, 1.0, 8.0, 5),
    "Financial Services": SectorRules(3.0, 7.0, 3.0, 30.0, 60.0, 7.0, 12.0, 2.0, 10.0, 7),
    "Consumer Cyclical": SectorRules(2.5, 5.5, 5.0, 25.0, 60.0, 8.0, 18.0, 1.0, 10.0, 7),
    # Para sectores no listados se usa BASELINE
}

def rules_for_sector(sector_name: Optional[str]) -> SectorRules:
    if sector_name in SECTOR_TWEAKS:
        return SECTOR_TWEAKS[sector_name]
    return BASELINE

# =============== Cálculo de métricas del ticker ===============

@dataclass
class TickerMetrics:
    ticker: str
    name: str
    sector: str
    price: float
    rpd_ttm: float       # %
    dgr5: float          # %
    dgr10: float         # %
    payout: float        # %
    per_ttm: float
    de_ratio: float
    roe: float           # %
    streak_years: int
    fcf_pos_years: int   # cuenta de años con FCF positivo (hasta 4-5 máx según datos)
    # --- NUEVOS ---
    exchange_code: str = ""
    exchange_name: str = ""
    country: str = ""
    currency: str = ""

def fetch_metrics(ticker: str) -> Tuple[TickerMetrics, pd.Series, pd.Series]:
    t = yf.Ticker(ticker)
    info = {}
    try:
        info = t.info or {}
    except Exception:
        info = {}

    name = info.get("shortName", ticker)
    sector = info.get("sector", "Unknown")

    # --- NUEVO: datos de cotización ---
    ex_code  = (info.get("exchange") or "").strip()          # p.ej. 'NYQ'
    ex_name  = YF_EXCHANGE_MAP.get(ex_code, ex_code or "Unknown")
    country  = (info.get("country") or "").strip()            # país de la compañía (no siempre = plaza)
    currency = (info.get("currency") or "").strip()           # divisa de cotización


    # Todo el histórico (daily). Si está vacío, intenta 20y; si falla, usa currentPrice
    hist = t.history(period="max", interval="1d")
    if hist is None or hist.empty:
        hist = t.history(period="20y", interval="1d")
    price = float(hist["Close"].dropna().iloc[-1]) if (hist is not None and not hist.empty) else float(info.get("currentPrice", np.nan))

    # Dividendos crudos y limpios (años completos)
    divs = t.dividends
    if divs is None or divs.empty:
        divs = pd.Series(dtype="float64")

    by_year_raw = dividends_by_year(divs)              # para diagnóstico
    by_year_clean = dividends_by_year_clean(divs)      # para DGR 'bueno'

    # TTM real
    ttm = dividends_ttm(divs)
    rpd = (ttm / price * 100.0) if price and price > 0 else np.nan

    # CAGR 5/10 años (ajustados y crudos)
    dgr5_raw  = cagr_from_annual(by_year_raw, years=5)
    dgr10_raw = cagr_from_annual(by_year_raw, years=10)
    dgr5_adj  = cagr_from_annual(by_year_clean, years=5)
    dgr10_adj = cagr_from_annual(by_year_clean, years=10)

    # Ratios contables (pueden faltar)
    payout = info.get("payoutRatio", np.nan)
    if isinstance(payout, (int, float)) and pd.notna(payout):
        payout *= 100.0

    per_ttm = info.get("trailingPE", np.nan)

    # Deuda/Patrimonio: robustez sobre % vs ratio
    de_ratio = info.get("debtToEquity", np.nan)
    if isinstance(de_ratio, (int, float)) and pd.notna(de_ratio):
        # Muchos tickers devuelven % (p.ej., 58 -> 0.58x). Si es claramente porcentaje, divide por 100.
        if de_ratio > 10 and de_ratio < 10000:
            de_ratio = float(de_ratio) / 100.0
        else:
            de_ratio = float(de_ratio)

    roe = info.get("returnOnEquity", np.nan)
    if isinstance(roe, (int, float)) and pd.notna(roe) and roe <= 1:
        roe *= 100.0  # yfinance suele darlo en fracción

    # Cash-flow anual (orientación columnas=fecha). Aseguramos última columna (más reciente)
    payout_fcf = np.nan
    fcf_pos_years = 0
    fcf_series = pd.Series(dtype="float64")
    try:
        cf = t.cashflow  # anual, index con 'Free Cash Flow', 'Dividends Paid', ...
        if cf is not None and not cf.empty:
            cf_t = cf.copy()
            # Si columnas son fechas, ordenar por columna ascendente y transponer serie
            cf_t = cf_t.loc[:, sorted(cf_t.columns)]
            if "Free Cash Flow" in cf_t.index:
                fcf_series = cf_t.loc["Free Cash Flow"].dropna()
                # contar años con FCF positivo
                fcf_pos_years = int((fcf_series > 0).sum())
            if "Dividends Paid" in cf_t.index and not fcf_series.empty:
                div_paid_last = abs(float(cf_t.loc["Dividends Paid"].dropna().iloc[-1]))
                fcf_last = abs(float(fcf_series.iloc[-1]))
                if fcf_last > 0:
                    payout_fcf = (div_paid_last / fcf_last) * 100.0
    except Exception:
        pass

    # Rachas (pagos y aumentos)
    streak_pay = dividend_streak_years(by_year_raw)
    streak_growth = dividend_growth_streak(by_year_clean)

    # RPD forward
    forward_div = info.get("forwardAnnualDividendRate") or info.get("dividendRate")
    rpd_forward = np.nan
    if forward_div and price:
        rpd_forward = (forward_div / price) * 100.0

    # EV/EBITDA y FCF yield
    ev = info.get("enterpriseValue", np.nan)
    ebitda = info.get("ebitda", np.nan)
    ev_ebitda = ev / ebitda if ev and ebitda and ebitda > 0 else np.nan
    marketcap = info.get("marketCap", np.nan)
    fcf_yield = (fcf_series.iloc[-1] / marketcap * 100.0) if (marketcap and not fcf_series.empty) else np.nan

    metrics = TickerMetrics(
        ticker=ticker.upper(),
        name=name,
        sector=sector,
        price=price,
        rpd_ttm=float(rpd) if pd.notna(rpd) else np.nan,
        dgr5=float(dgr5_adj * 100.0) if pd.notna(dgr5_adj) else np.nan,
        dgr10=float(dgr10_adj * 100.0) if pd.notna(dgr10_adj) else np.nan,
        payout=float(payout) if pd.notna(payout) else np.nan,
        per_ttm=float(per_ttm) if pd.notna(per_ttm) else np.nan,
        de_ratio=float(de_ratio) if pd.notna(de_ratio) else np.nan,
        roe=float(roe) if pd.notna(roe) else np.nan,
        streak_years=streak_pay,
        fcf_pos_years=fcf_pos_years,
        # --- NUEVOS ---
        exchange_code=ex_code,
        exchange_name=ex_name,
        country=country,
        currency=currency,
    )
    # extras para diagnóstico
    metrics.rpd_forward = rpd_forward
    metrics.payout_fcf = payout_fcf
    metrics.ev_ebitda = ev_ebitda
    metrics.fcf_yield = fcf_yield
    metrics.streak_growth = streak_growth
    metrics.dgr5_raw  = float(dgr5_raw * 100.0) if pd.notna(dgr5_raw) else np.nan
    metrics.dgr10_raw = float(dgr10_raw * 100.0) if pd.notna(dgr10_raw) else np.nan
    metrics.dgr_reliable = not dgr_not_reliable(by_year_clean, by_year_raw)


    price_series = hist["Close"] if (hist is not None and "Close" in hist) else pd.Series(dtype="float64")
    return metrics, by_year_raw, price_series



def dividend_growth_streak(series_by_year: pd.Series) -> int:
    """
    Años consecutivos (hasta el último) con AUMENTO del DPS anual.
    """
    if series_by_year is None or series_by_year.empty:
        return 0
    s = series_by_year.dropna().sort_index()
    years = list(s.index)
    if len(years) < 2:
        return 0

    streak = 0
    # contamos desde el final hacia atrás: cada paso exige s[y] > s[y-1]
    for i in range(len(years)-1, 0, -1):
        if s.iloc[i] > s.iloc[i-1]:
            streak += 1
        else:
            break
    return streak


def near_limit(value, min_v, max_v, tol=0.1) -> bool:
    """Devuelve True si el valor está dentro del ±10% del límite permitido."""
    if pd.isna(value):
        return False
    if min_v is not None and value < min_v and value >= min_v * (1 - tol):
        return True
    if max_v is not None and value > max_v and value <= max_v * (1 + tol):
        return True
    return False





# =============== Evaluación vs reglas ===============

@dataclass
class CheckResult:
    ok: bool
    borderline: bool
    msg: str

def check_range(value: float, min_v: Optional[float], max_v: Optional[float], units: str="") -> CheckResult:
    if pd.isna(value):
        return CheckResult(False, False, f"Sin datos")
    if min_v is not None and value < min_v:
        return CheckResult(False, True, f"< {min_v}{units}")
    if max_v is not None and value > max_v:
        return CheckResult(False, True, f"> {max_v}{units}")
    return CheckResult(True, False, f"OK")

def evaluate(metrics: TickerMetrics, rules: SectorRules) -> Dict[str, Dict]:
    results = {}
    ranges = {
        "RPD TTM (%)": f"{rules.rpd_min}–{rules.rpd_max} %",
        "DGR 5a (%)": f"≥ {rules.dgr5_min} %",
        "Payout (%)": f"{rules.payout_min}–{rules.payout_max} %",
        "PER (ttm)": f"{rules.per_min}–{rules.per_max}",
        "Deuda/Patrimonio": f"≤ {rules.de_max} x",
        "ROE (%)": f"≥ {rules.roe_min} %",
        "Racha dividendos (años)": f"≥ {rules.streak_min} a"
    }

    for k, (val, min_v, max_v, units) in {
        "RPD TTM (%)": (metrics.rpd_ttm, rules.rpd_min, rules.rpd_max, "%"),
        "DGR 5a (%)": (metrics.dgr5, rules.dgr5_min, None, "%"),
        "Payout (%)": (metrics.payout, rules.payout_min, rules.payout_max, "%"),
        "PER (ttm)": (metrics.per_ttm, rules.per_min, rules.per_max, ""),
        "Deuda/Patrimonio": (metrics.de_ratio, None, rules.de_max, "x"),
        "ROE (%)": (metrics.roe, rules.roe_min, None, "%"),
        "Racha dividendos (años)": (metrics.streak_years, rules.streak_min, None, "a"),
    }.items():
        if k.startswith("DGR") and not metrics.dgr_reliable:
            results[k] = {
                "icon": "⚠️",
                "msg": "No fiable (estructura de dividendo)",
                "range": ranges[k]
            }
            continue
        if pd.isna(val):
            results[k] = {"icon": "❌", "msg": "Sin datos", "range": ranges[k]}
        elif k.startswith("DGR") and val < 0:
            results[k] = {"icon": "❌", "msg": "Negativo", "range": ranges[k]}
        elif (min_v is not None and val < min_v) or (max_v is not None and val > max_v):
            icon = "⚠️" if near_limit(val, min_v, max_v) else "❌"
            results[k] = {"icon": icon, "msg": f"{val:.2f}{units}", "range": ranges[k]}
        else:
            results[k] = {"icon": "✅", "msg": f"{val:.2f}{units}", "range": ranges[k]}

    return results

def chart_score_breakdown(breakdown: dict):
    cats = list(breakdown.keys())
    vals = [breakdown[k] for k in cats]

    fig = go.Figure(go.Bar(
        x=vals, y=cats, orientation="h",
        hovertemplate="%{x:.0f} / 100<extra></extra>"
    ))

    # Aplica layout financiero base (una sola vez)
    apply_finance_layout(fig, title="Desglose del score", rangeslider=False)

    # Ajustes específicos de ejes SIN duplicar claves
    fig.update_xaxes(range=[0, 100])  # rango 0–100 para el score
    # (opcional) quitar decimales en ticks
    fig.update_xaxes(tickmode="linear", dtick=10)

    # Altura compacta
    fig.update_layout(height=280)

    return fig



# =============== ANEXO: explicación de ratios y rangos ===============

def rules_to_range_str(r: SectorRules) -> Dict[str, str]:
    return {
        "RPD TTM (%)": f"{r.rpd_min}–{r.rpd_max} %",
        "DGR 5a (%)": f"≥ {r.dgr5_min} %",
        "Payout (%)": f"{r.payout_min}–{r.payout_max} %",
        "PER (ttm)": f"{r.per_min}–{r.per_max}",
        "Deuda/Patrimonio (x)": f"≤ {r.de_max} x",
        "ROE (%)": f"≥ {r.roe_min} %",
        "Racha dividendos (años)": f"≥ {r.streak_min} a",
    }

def render_anexo():
    st.title("Anexo de ratios y rangos recomendados")
    st.write(
        "Aquí tienes una explicación breve de cada ratio **estilo Gregorio**: "
        "qué mide, por qué importa, cómo se calcula y en qué **rangos** es razonable moverse. "
        "Los rangos pueden variar por **sector**."
    )

    # Definiciones y por qué importan
    definiciones = [
        {
            "Ratio": "RPD TTM (%)",
            "Qué mide": "Rentabilidad por dividendo actual con base en TTM (últimos 12 meses).",
            "Fórmula": "RPD = (Dividendo anual TTM / Precio) × 100",
            "Por qué importa": "Te dice cuánta renta anual genera hoy cada euro invertido."
        },
        {
            "Ratio": "DGR 5a (%)",
            "Qué mide": "Crecimiento anual compuesto del dividendo a 5 años.",
            "Fórmula": "CAGR = (Div5 / Div0)^(1/5) − 1",
            "Por qué importa": "Indica si la empresa aumenta el dividendo de forma sostenida."
        },
        {
            "Ratio": "Payout (%)",
            "Qué mide": "% del beneficio destinado a dividendos.",
            "Fórmula": "Payout = (Dividendo / BPA) × 100",
            "Por qué importa": "Un payout moderado sugiere dividendo sostenible y margen para crecer."
        },
        {
            "Ratio": "PER (ttm)",
            "Qué mide": "Años de beneficios que pagas al precio actual.",
            "Fórmula": "PER = Precio / BPA (ttm)",
            "Por qué importa": "Da una idea de si pagas un precio razonable por la calidad."
        },
        {
            "Ratio": "Deuda/Patrimonio (x)",
            "Qué mide": "Nivel de apalancamiento financiero.",
            "Fórmula": "D/E = Deuda total / Patrimonio neto",
            "Por qué importa": "Menos deuda = más margen en crisis y dividendo más defendible."
        },
        {
            "Ratio": "ROE (%)",
            "Qué mide": "Rentabilidad del patrimonio de los accionistas.",
            "Fórmula": "ROE = (Beneficio neto / Patrimonio) × 100",
            "Por qué importa": "Empresas con ROE alto suelen tener ventajas competitivas."
        },
        {
            "Ratio": "Racha dividendos (años)",
            "Qué mide": "Años consecutivos pagando dividendo (> 0).",
            "Fórmula": "Conteo de años con pago positivo",
            "Por qué importa": "Historial de pagos consistente = mayor fiabilidad."
        },
    ]
    st.subheader("Definiciones rápidas")
    st.table(pd.DataFrame(definiciones))

    # Rangos baseline
    st.subheader("Rangos recomendados (baseline)")
    base_ranges = rules_to_range_str(BASELINE)
    st.table(pd.DataFrame([base_ranges]))

    # Rangos por sector
    st.subheader("Rangos por sector (overrides)")
    if SECTOR_TWEAKS:
        rows = []
        for sec, r in SECTOR_TWEAKS.items():
            d = rules_to_range_str(r)
            d["Sector"] = sec
            rows.append(d)
        df = pd.DataFrame(rows).set_index("Sector")
        st.table(df)
    else:
        st.info("No hay ajustes sectoriales definidos; se usa solo el baseline.")
    
    
    # Descarga como Markdown
    md = ["# Anexo de ratios (estilo Gregorio)\n"]
    for d in definiciones:
        md.append(f"## {d['Ratio']}\n")
        md.append(f"- **Qué mide:** {d['Qué mide']}\n")
        md.append(f"- **Fórmula:** {d['Fórmula']}\n")
        md.append(f"- **Por qué importa:** {d['Por qué importa']}\n")
    md.append("\n## Rangos baseline\n")
    for k, v in base_ranges.items():
        md.append(f"- **{k}:** {v}")
    md.append("\n## Rangos por sector\n")
    for sec, r in SECTOR_TWEAKS.items():
        rr = rules_to_range_str(r)
        md.append(f"### {sec}")
        for k, v in rr.items():
            md.append(f"- **{k}:** {v}")
    md_text = "\n".join(md)
    
    st.download_button(
        label="Descargar anexo (Markdown)",
        data=md_text.encode("utf-8"),
        file_name="anexo_ratios_gregorio.md",
        mime="text/markdown"
    )


def render_anexo_scoring():
    st.title("Anexo: Metodología de Scoring")
    st.write("""
El objetivo del *score* es reflejar el enfoque de Gregorio: **rentas crecientes y sostenibles**,
con **negocios estables** y **precio razonable**. No exige perfección; pondera el **conjunto**.
    """)

    # Ponderaciones
    weights = pd.DataFrame([
        {"Bloque": "Dividendo", "Peso": "40%"},
        {"Bloque": "Solidez", "Peso": "25%"},
        {"Bloque": "Valoración", "Peso": "15%"},
        {"Bloque": "Historial", "Peso": "20%"},
    ])
    st.subheader("Ponderaciones del score")
    st.table(weights)

    st.subheader("Componentes y reglas")
    st.markdown("""
**Dividendo (40%)**
- RPD (usa *forward* si existe, si no TTM). Premia estar en el **rango del sector** y cerca del centro.
- DGR 5 años: cuanto más alto, mejor; si es **negativo** puntúa 0.
- DGR 10 años: solo pedimos **≥ 0%**.
- Payout: usamos **Payout FCF** si hay datos (preferible), si no payout contable. Rango objetivo baseline 40–70%.

**Solidez (25%)**
- Deuda/Patrimonio (o ND/EBITDA si lo sustituyes): menos es mejor.
- ROE: deseable ≥ 8–10%.
- Racha de pagos: años consecutivos pagando.
- Racha de crecimiento: años consecutivos **aumentando** el dividendo.

**Valoración (15%)**
- Si hay PER: se pondera dentro del rango del sector (preferencia por el punto medio).
- Si **no hay PER**: usamos **EV/EBITDA** (≈ 5–6x razonable en telecos/utilities) y **FCF Yield** (deseable ≥ 8–10%).

**Historial (20%)**
- FCF positivo: al menos 3 años recientes con FCF > 0.
- Racha de crecimiento también pondera aquí (media con FCF positivo).
    """)

    st.subheader("Banderas rojas (pausa automática)")
    st.markdown("""
- **Payout FCF > 100%** o **FCF negativo 2 de 3 años**.
- **DGR 5a < 0%** *y* payout alto.
- **Recorte reciente** del dividendo (racha de crecimiento 0 y DGR 5a negativa).
Si hay banderas, la recomendación sugiere **pausar** o revisar con detalle.
    """)

    st.subheader("Cómo puntúa cada métrica (0–100)")
    st.markdown("""
- **Dos cotas (mín–máx)**: 100 si está dentro del rango; si se activa *prefer_mid*, máximo en el centro del rango.
- **Solo mínima**: 100 si `valor ≥ min`, 0 si `valor < min`.
- **Solo máxima**: 100 si `valor ≤ max`, 0 si `valor > max`.
- **NaN**: 0.
    """)

    # Descarga
    md = """
# Anexo: Metodología de Scoring

## Ponderaciones
- Dividendo: 40%
- Solidez: 25%
- Valoración: 15%
- Historial: 20%

## Dividendo
- RPD (forward preferible), DGR 5a/10a, Payout (FCF > contable)
- DGR5 < 0% → puntuación 0
- DGR10: solo se exige ≥ 0%

## Solidez
- Deuda/Patrimonio (o ND/EBITDA), ROE, racha de pagos, racha de crecimiento

## Valoración
- Con PER: rango sectorial
- Sin PER: EV/EBITDA (~5–6x) y FCF Yield (≥ 8–10%)

## Historial
- Años con FCF positivo (≥3), racha de crecimiento

## Banderas rojas
- Payout FCF > 100% o FCF negativo 2/3
- DGR5 < 0% + payout alto
- Recorte reciente del dividendo

## Reglas de puntuación (0–100)
- Dos cotas: 100 en el rango (pico en el centro si corresponde)
- Solo mínima: 100 si valor ≥ min; 0 si < min
- Solo máxima: 100 si valor ≤ max; 0 si > max
- NaN: 0
"""
    st.download_button(
        "Descargar anexo (Markdown)",
        data=md.encode("utf-8"),
        file_name="anexo_scoring_gregorio.md",
        mime="text/markdown"
    )




@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all_metrics(tickers_tuple):
    tickers = list(tickers_tuple)
    results = []
    for tk in tickers:
        try:
            m, divs, _ = fetch_metrics(tk)
            rules = rules_for_sector(m.sector)
            total_score, breakdown, details = score_company(m, rules)
            results.append({
                "Ticker": m.ticker,
                "Nombre": m.name,
                "Sector": m.sector,
                "RPD (%)": m.rpd_ttm,
                "DGR 5a (%)": m.dgr5,
                "Payout (%)": m.payout,
                "PER": m.per_ttm,
                "Deuda/Patrimonio": m.de_ratio,
                "ROE (%)": m.roe,
                "Score total": total_score,
            })
        except Exception:
            results.append({"Ticker": tk, "Nombre": "ERROR", "Sector": "", "Score total": np.nan})
    return pd.DataFrame(results)




# =============== UI Streamlit ===============

st.set_page_config(page_title="Chequeo Gregorio por Ticker", layout="centered")

# ========= Estado global =========
if "mode" not in st.session_state:
    st.session_state.mode = "individual"
if "active_ticker" not in st.session_state:
    st.session_state.active_ticker = "JNJ"
if "autoload" not in st.session_state:
    st.session_state.autoload = False
if "last_loaded_ticker" not in st.session_state:
    st.session_state.last_loaded_ticker = None

# Datos cacheados en sesión
if "metrics" not in st.session_state:
    st.session_state.metrics = None
if "divs_year" not in st.session_state:
    st.session_state.divs_year = None
if "price_series" not in st.session_state:
    st.session_state.price_series = None
if "rules" not in st.session_state:
    st.session_state.rules = None
if "checks" not in st.session_state:
    st.session_state.checks = None
if "ccy_symbol" not in st.session_state:
    st.session_state.ccy_symbol = "$"

# ========= Navegación =========
st.title("Chequeo estilo Gregorio")
mode_label = st.radio(
    "Modo",
    ["🔍 Análisis individual", "📊 Ranking / Screening"],
    horizontal=True,
    index=0 if st.session_state.mode == "individual" else 1,
)
st.session_state.mode = "individual" if mode_label.startswith("🔍") else "ranking"
st.markdown("---")


# =========================
# 🔍 ANÁLISIS INDIVIDUAL
# =========================
if st.session_state.mode == "individual":

    st.sidebar.title("Menú")
    vista = st.sidebar.radio("Vista", ["Evaluación", "Anexo de ratios", "Anexo: Scoring"])

    if vista == "Anexo de ratios":
        render_anexo()
        st.stop()
    if vista == "Anexo: Scoring":
        render_anexo_scoring()
        st.stop()

    with st.sidebar:
        st.header("Parámetros de evaluación")
        ticker_input = st.text_input("Ticker", value=st.session_state.active_ticker, key="ticker_input").strip().upper()

        sector_override = st.selectbox(
            "Forzar sector (opcional)",
            options=["(auto)", "Consumer Defensive", "Healthcare", "Utilities", "Communication Services",
                     "Industrials", "Technology", "Energy", "Financial Services", "Consumer Cyclical"],
            index=0,
            key="sector_override"
        )

        run_btn = st.button("Evaluar", key="btn_eval")

    # ¿Hay que cargar datos?
    need_load = False
    if run_btn and ticker_input:
        st.session_state.active_ticker = ticker_input
        need_load = True

    if st.session_state.autoload and st.session_state.active_ticker:
        if st.session_state.metrics is None or st.session_state.last_loaded_ticker != st.session_state.active_ticker:
            need_load = True

    if need_load:
        try:
            metrics, divs_year, price_series = fetch_metrics(st.session_state.active_ticker)

            if sector_override != "(auto)":
                metrics.sector = sector_override

            st.session_state.metrics = metrics
            st.session_state.divs_year = divs_year
            st.session_state.price_series = price_series

            st.session_state.rules = rules_for_sector(metrics.sector)
            st.session_state.checks = evaluate(metrics, st.session_state.rules)

            try:
                info_local = yf.Ticker(metrics.ticker).info or {}
            except Exception:
                info_local = {}
            st.session_state.ccy_symbol = _ccy_symbol_from_info(info_local, "$")

            st.session_state.last_loaded_ticker = st.session_state.active_ticker
            st.session_state.autoload = False

        except Exception as e:
            st.error(f"Error al evaluar {st.session_state.active_ticker}: {e}")
            st.exception(e)
            st.stop()

    if st.session_state.metrics is None:
        st.info("Introduce un ticker y pulsa **Evaluar**.")
        st.stop()

    metrics      = st.session_state.metrics
    divs_year    = st.session_state.divs_year
    price_series = st.session_state.price_series
    ccy_symbol   = st.session_state.ccy_symbol
    rules        = st.session_state.rules or rules_for_sector(metrics.sector)
    checks       = st.session_state.checks or evaluate(metrics, rules)

    st.subheader(f"{metrics.ticker} — {metrics.name}")
    cotiza_txt = f"Cotiza en: **{metrics.exchange_name or '—'}**"
    if metrics.country:
        cotiza_txt += f" ({metrics.country})"
    if metrics.currency:
        cotiza_txt += f" · {metrics.currency}"
    st.caption(f"Sector: **{metrics.sector}** | {cotiza_txt} | Precio: **{metrics.price:.2f}**")

    col1, col2, col3 = st.columns(3)
    col1.metric("RPD TTM", f"{metrics.rpd_ttm:,.2f} %")
    if not metrics.dgr_reliable:
        col2.metric("DGR 5 años", "No fiable")
    else:
        col2.metric("DGR 5 años", "s/d" if pd.isna(metrics.dgr5) else f"{metrics.dgr5:,.2f} %")

    #col2.metric("DGR 5 años", "s/d" if pd.isna(metrics.dgr5) else f"{metrics.dgr5:,.2f} %")
    col3.metric("DGR 10 años", "s/d" if pd.isna(metrics.dgr10) else f"{metrics.dgr10:,.2f} %")

    col4, col5, col6 = st.columns(3)
    col4.metric("Payout", "s/d" if pd.isna(metrics.payout) else f"{metrics.payout:,.2f} %")
    col5.metric("PER (ttm)", "s/d" if pd.isna(metrics.per_ttm) else f"{metrics.per_ttm:,.2f}")
    col6.metric("Deuda/Patrimonio", "s/d" if pd.isna(metrics.de_ratio) else f"{metrics.de_ratio:,.2f} x")

    col7, col8 = st.columns(2)
    col7.metric("ROE", "s/d" if pd.isna(metrics.roe) else f"{metrics.roe:,.2f} %")
    col8.metric("Racha dividendos", f"{metrics.streak_years} años")

    ev_eb = getattr(metrics, "ev_ebitda", np.nan)
    fcf_y = getattr(metrics, "fcf_yield", np.nan)
    col9, col10 = st.columns(2)
    col9.metric("EV/EBITDA", "s/d" if pd.isna(ev_eb) else f"{ev_eb:,.2f} x")
    col10.metric("FCF Yield", "s/d" if pd.isna(fcf_y) else f"{fcf_y:,.2f} %")

    if pd.notna(getattr(metrics, "rpd_forward", np.nan)):
        st.metric("RPD Forward", f"{metrics.rpd_forward:,.2f} %")

    with st.expander("Detalle DGR (crudo vs ajustado)"):
        st.write(f"DGR 5a (ajustado): **{metrics.dgr5:.2f}%** | DGR 5a (crudo): {getattr(metrics,'dgr5_raw', np.nan):.2f}%")
        st.write(f"DGR 10a (ajustado): **{metrics.dgr10:.2f}%** | DGR 10a (crudo): {getattr(metrics,'dgr10_raw', np.nan):.2f}%")

    st.markdown("---")
    st.subheader("Veredicto por criterios (según sector)")
    table = []
    for k, v in checks.items():
        table.append({
            "Criterio": k,
            "Resultado": v["icon"],
            "Valor": v["msg"],
            "Intervalo recomendado": v["range"]
        })
    st.table(pd.DataFrame(table))

    st.subheader("Histórico de dividendos anuales")
    if divs_year is not None and not divs_year.empty:
        st.plotly_chart(chart_dividends_bar(divs_year, "Dividendos por año", ccy_symbol),
                        use_container_width=True, config={"displaylogo": False})
    else:
        st.info("Sin histórico de dividendos para mostrar.")

    st.subheader("Precio")
    if price_series is not None and not price_series.empty:
        c1, c2 = st.columns(2)
        rango = c1.selectbox("Rango", ["Todo", "20 años", "10 años", "5 años", "3 años", "1 año", "YTD"], index=0, key="rango_price")
        agg = c2.selectbox("Agrupar", ["Diario", "Semanal", "Mensual"], index=0, key="agg_price")
        ps = _slice_price_series(price_series, rango, agg)
        st.plotly_chart(chart_price_line(ps, f"Precio ({rango.lower()}, {agg.lower()})", ccy_symbol),
                        use_container_width=True, config={"displaylogo": False})
    else:
        st.info("Sin serie de precios para mostrar.")

    st.markdown("---")
    st.subheader("Umbrales usados (según sector)")
    rules_df = pd.DataFrame([{
        "RPD min %": rules.rpd_min,
        "RPD max %": rules.rpd_max,
        "DGR 5a min %": rules.dgr5_min,
        "Payout % (min-máx)": f"{rules.payout_min} - {rules.payout_max}",
        "PER (mín-máx)": f"{rules.per_min} - {rules.per_max}",
        "Deuda/Patrimonio máx (x)": rules.de_max,
        "ROE min %": rules.roe_min,
        "Racha dividendos min (años)": rules.streak_min,
    }])
    st.table(rules_df)

    flags = red_flags(metrics)
    total_score, breakdown, details = score_company(metrics, rules)
    st.plotly_chart(chart_score_breakdown(breakdown), use_container_width=True, config={"displaylogo": False})

    rec_text, rec_icon, rec_flags = recommendation(total_score, flags)
    st.markdown("---")
    st.subheader("Puntuación compuesta (estilo Gregorio)")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Score total", f"{total_score:.0f}/100")
    c2.metric("Dividendo", f"{breakdown['Dividendo']:.0f}")
    c3.metric("Solidez", f"{breakdown['Solidez']:.0f}")
    c4.metric("Valoración", f"{breakdown['Valoración']:.0f}")
    c5.metric("Historial", f"{breakdown['Historial']:.0f}")

    with st.expander("Ver detalle del cálculo (componentes y pesos)"):
        for bloque in ["Dividendo", "Solidez", "Valoración", "Historial"]:
            st.markdown(f"**{bloque}**")
            st.table(pd.DataFrame(details[bloque])[["Métrica","Valor","Rango","Peso bloq.","Sub-score"]])

    if rec_icon == "✅":
        st.success(f"{rec_icon} {rec_text}")
    elif rec_icon == "⚠️":
        st.warning(f"{rec_icon} {rec_text}")
    else:
        st.error(f"{rec_icon} {rec_text}")

    if flags:
        st.write("**Banderas rojas detectadas:**")
        for f in flags:
            st.write(f"- {f}")
    
    if not metrics.dgr_reliable:
        st.info(
            "El crecimiento del dividendo no es fiable por estructura irregular "
            "(scrip dividend, ajustes o años incompletos). "
            "Se trata como dividendo estable, no como crecimiento negativo."
        )

    st.markdown("---")
    st.subheader("Informe IA (Groq)")
    model = st.selectbox("Modelo Groq", ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"], index=0, key="groq_model_ind")

    if st.button("Generar informe IA (empresa)", key="btn_ai_individual"):
        prompt = build_prompt_individual(metrics, rules, total_score, breakdown, flags)
        messages = [
            {"role": "system", "content": "Responde en español. Sé conciso y extremadamente fiel a los datos."},
            {"role": "user", "content": prompt},
        ]
        cache_key = _hash_key("individual", model, prompt)
        with st.spinner("Generando informe con Groq..."):
            ai_text = groq_chat_cached(cache_key, groq_api_key, model, messages, temperature=0.2, max_tokens=900)
        st.session_state["ai_report_individual"] = ai_text
    
    
    st.text_area("Informe IA", value=st.session_state.get("ai_report_individual", ""), height=320)
    st.info("Consejo Gregorio: prioriza negocios estables, con dividendo creciente, payout razonable y balance sano. Compra periódica (DCA) y paciencia a 10–30 años.")


# =========================
# 📊 RANKING / SCREENING
# =========================
else:
    st.header("📊 Ranking de empresas (por puntuación Gregorio)")

    bolsa_opcion = st.selectbox(
        "Selecciona una bolsa o conjunto",
        ["S&P 500 (real)", "Dividend Aristocrats (real)", "Personalizado"],
        index=0,
        key="bolsa_opcion"
    )

    top_n = st.slider("Top N por capitalización (solo dividend payers)", 50, 300, 200, step=25, key="top_n")

    tickers_input = ""
    if bolsa_opcion == "Personalizado":
        tickers_input = st.text_area("Tickers separados por comas (ej: JNJ, PG, KO)", value="JNJ,PG,KO", key="tickers_input")

    if bolsa_opcion == "S&P 500 (real)":
        base = load_sp500_tickers()
    elif bolsa_opcion == "Dividend Aristocrats (real)":
        base = load_dividend_aristocrats_tickers()
    else:
        base = [t.strip().upper().replace(".", "-") for t in tickers_input.split(",") if t.strip()]

    st.caption(f"Universo base: {len(base)} tickers")

    with st.spinner("Aplicando filtro: dividend payers + top por capitalización..."):
        tickers = filter_top_dividend_payers(base, top_n=top_n)

    st.success(f"Universo filtrado: {len(tickers)} tickers")

    universe_key = (bolsa_opcion, top_n, ",".join(tickers))
    if st.session_state.get("ranking_universe_key") != universe_key:
        st.session_state.ranking_df = None
        st.session_state.ranking_universe_key = universe_key

    run_rank = st.button("Calcular ranking", key="btn_rank")

    if run_rank:
        with st.spinner("Calculando ranking... (la primera vez puede tardar)"):
            df = fetch_all_metrics(tuple(tickers))
        st.session_state.ranking_df = df

    df = st.session_state.get("ranking_df", None)
    if df is None or df.empty:
        st.info("Configura el universo y pulsa **Calcular ranking**.")
        st.stop()

    df_sorted = df.sort_values("Score total", ascending=False)
    st.subheader("🏆 Ranking general")
    st.dataframe(df_sorted, use_container_width=True, height=520)

    st.markdown("---")
    st.subheader("Resúmenes IA (Groq)")
    model = st.selectbox("Modelo Groq", ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"], index=0, key="groq_model_rank")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Generar resumen del ranking (Top 25)", key="btn_ai_rank"):
            prompt = build_prompt_ranking(df_sorted)
            messages = [
                {"role": "system", "content": "Responde en español. No inventes datos ni empresas."},
                {"role": "user", "content": prompt},
            ]
            cache_key = _hash_key("ranking", model, prompt)
            with st.spinner("Generando resumen del ranking..."):
                st.session_state["ai_rank_summary"] = groq_chat_cached(cache_key, groq_api_key, model, messages, temperature=0.2, max_tokens=900)

    with c2:
        if st.button("Generar resumen por sectores (Top 3/sector)", key="btn_ai_sector"):
            prompt = build_prompt_sector_summary(df_sorted)
            messages = [
                {"role": "system", "content": "Responde en español. No inventes datos ni empresas."},
                {"role": "user", "content": prompt},
            ]
            cache_key = _hash_key("sectors", model, prompt)
            with st.spinner("Generando resumen por sectores..."):
                st.session_state["ai_sector_summary"] = groq_chat_cached(cache_key, groq_api_key, model, messages, temperature=0.2, max_tokens=900)

    st.text_area("Resumen IA del ranking", st.session_state.get("ai_rank_summary", ""), height=240)
    st.text_area("Resumen IA por sectores", st.session_state.get("ai_sector_summary", ""), height=240)

    selected_ticker = st.selectbox(
        "Selecciona una empresa para ver su análisis detallado:",
        df_sorted["Ticker"].tolist(),
        key="rank_select"
    )

    if st.button("🔍 Ver análisis detallado", key="btn_go_detail"):
        st.session_state.active_ticker = selected_ticker
        st.session_state.mode = "individual"
        st.session_state.autoload = True
        st.session_state.metrics = None
        st.rerun()
