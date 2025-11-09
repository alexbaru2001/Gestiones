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

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# =============== Utilidades de tiempo y series ===============

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
    if divs is None or divs.empty:
        return pd.Series(dtype="float64")
    s = divs.copy()
    s.index = to_naive_utc_index(s.index)
    return s.groupby(s.index.year).sum()

def cagr_from_annual(series_by_year: pd.Series, years: int) -> float:
    s = series_by_year.dropna()
    if len(s) < 2:
        return np.nan
    last_year = int(s.index.max())
    first_target = last_year - (years - 1)
    if first_target in s.index:
        start_val = s.loc[first_target]
    else:
        # toma el valor más antiguo disponible dentro de la ventana; si no hay, el primero
        window = s[s.index <= first_target]
        start_val = window.iloc[-1] if not window.empty else s.iloc[0]
    end_val = s.iloc[-1]
    if start_val <= 0 or end_val <= 0:
        return np.nan
    n = max(1, years - 1)
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
    """Normaliza 0-100. Si prefer_mid=True, premia estar en el rango medio."""
    if pd.isna(val): 
        return 0.0
    if min_v is None and max_v is None:
        return 50.0
    if min_v is not None and max_v is not None:
        if val < min_v:  # por debajo del rango
            return max(0.0, 100.0*(val/min_v))  # cae lineal
        if val > max_v:  # por encima del rango
            return max(0.0, 100.0*(max_v/val))
        # dentro del rango
        if prefer_mid:
            mid = (min_v + max_v)/2
            # campana simple: pico en el centro
            return 100.0 - 100.0*abs(val-mid)/((max_v-min_v)/2)
        return 100.0
    if min_v is not None:  # solo cota mínima
        return max(0.0, min(100.0, 100.0*(val/min_v)))
    if max_v is not None:  # solo cota máxima
        return max(0.0, min(100.0, 100.0*(max_v/val)))
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

def score_company(metrics, rules):
    """Devuelve (score_total, breakdown_dict) en escala 0-100."""
    # 1) Dividendo (40%)
    s_rpd   = _band_score(metrics.rpd_forward if pd.notna(getattr(metrics, "rpd_forward", np.nan)) else metrics.rpd_ttm,
                          rules.rpd_min, rules.rpd_max, prefer_mid=True)
    s_dgr5  = _band_score(metrics.dgr5, rules.dgr5_min, None)
    s_dgr10 = _band_score(metrics.dgr10, 0.0, None)  # que no sea negativo
    # payout: queremos dentro de banda
    s_payout = _band_score(metrics.payout, rules.payout_min, rules.payout_max, prefer_mid=True)
    # payout FCF pesa más si existe
    s_payout_fcf = _band_score(getattr(metrics, "payout_fcf", np.nan), 40.0, 70.0, prefer_mid=True) if pd.notna(getattr(metrics, "payout_fcf", np.nan)) else s_payout
    score_div = 0.4*(0.35*s_rpd + 0.30*s_dgr5 + 0.15*s_dgr10 + 0.20*s_payout_fcf)

    # 2) Solidez (25%)
    s_debt  = _band_score(metrics.de_ratio, None, rules.de_max)
    s_roe   = _band_score(metrics.roe, rules.roe_min, None)
    s_streak_pay = _band_score(metrics.streak_years, rules.streak_min, None)
    s_streak_growth = _band_score(getattr(metrics, "streak_growth", np.nan), 5.0, None)  # p.ej. al menos 5 años creciendo
    score_solid = 0.25*(0.35*s_debt + 0.30*s_roe + 0.20*s_streak_pay + 0.15*s_streak_growth)

    # 3) Valoración (15%) -> usa PER o EV/EBITDA + FCF yield
    if pd.notna(metrics.per_ttm):
        s_per = _band_score(metrics.per_ttm, rules.per_min, rules.per_max, prefer_mid=True)
        score_val = 0.15*s_per
    else:
        s_ev_ebitda = _band_score(getattr(metrics, "ev_ebitda", np.nan), None, 6.0)  # telecos/utilities 5-6x razonable
        s_fcf_yield = _band_score(getattr(metrics, "fcf_yield", np.nan), 8.0, None)   # deseable ≥8-10%
        score_val = 0.15*(0.6*s_ev_ebitda + 0.4*s_fcf_yield)

    # 4) Calidad/Historial (20%)
    # Usa mezcla de DGR10 (ya incluido), FCF positivo años, rachas
    s_fcf_pos = _band_score(metrics.fcf_pos_years, 3.0, None)  # al menos 3 años recientes positivos
    score_hist = 0.20*(0.5*s_streak_growth + 0.5*s_fcf_pos)

    total = score_div + score_solid + score_val + score_hist
    breakdown = {
        "Dividendo": score_div, "Solidez": score_solid, "Valoración": score_val, "Historial": score_hist
    }
    return max(0.0, min(100.0, total)), breakdown

def recommendation(total_score, flags):
    if flags:
        return "Pausa / Revisar banderas rojas", "❌", flags
    if total_score >= 75:
        return "Comprar / Añadir (DCA normal)", "✅", []
    if total_score >= 60:
        return "Vigilar o DCA prudente", "⚠️", []
    return "Mantenerse al margen", "❌", []


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

def fetch_metrics(ticker: str) -> Tuple[TickerMetrics, pd.Series, pd.Series]:
    t = yf.Ticker(ticker)
    info = {}
    try:
        info = t.info or {}
    except Exception:
        info = {}

    name = info.get("shortName", ticker)
    sector = info.get("sector", "Unknown")

    # Precio
    hist = t.history(period="1y")
    price = float(hist["Close"].dropna().iloc[-1]) if not hist.empty else float(info.get("currentPrice", np.nan))

    # Dividendos
    divs = t.dividends
    if divs is None or divs.empty:
        divs = pd.Series(dtype="float64")
    by_year = dividends_by_year(divs)
    ttm = dividends_ttm(divs)

    rpd = (ttm / price * 100.0) if price and price > 0 else np.nan
    dgr5 = cagr_from_annual(by_year, years=5)
    dgr10 = cagr_from_annual(by_year, years=10)

    # Ratios contables (pueden faltar en info)
    payout = info.get("payoutRatio", np.nan)
    if isinstance(payout, (int, float)) and pd.notna(payout):
        payout *= 100.0

    per_ttm = info.get("trailingPE", np.nan)
    de_ratio = info.get("debtToEquity", np.nan)  # suele venir como número (p.ej. 80 => 0.8x patrimonio)
    if isinstance(de_ratio, (int, float)) and pd.notna(de_ratio):
        de_ratio = float(de_ratio) / 100.0 if de_ratio > 10 else float(de_ratio)  # heurística ligera

    roe = info.get("returnOnEquity", np.nan)
    if isinstance(roe, (int, float)) and pd.notna(roe) and roe <= 1:
        roe *= 100.0  # yfinance a menudo da ROE en fracción

    # FCF y rachas
    cf = pd.DataFrame()
    payout_fcf = np.nan
    try:
        cf = t.cashflow  # anual
    except Exception:
        pass

    fcf_pos_years = 0
    if cf is not None and not cf.empty:
        cf = cf.copy()
        if "Free Cash Flow" in cf.index:
            fcf_series = cf.loc["Free Cash Flow"].dropna()
            fcf_pos_years = int((fcf_series > 0).sum())
            if not fcf_series.empty and len(fcf_series) > 0 and "Dividends Paid" in cf.index:
                div_paid = abs(cf.loc["Dividends Paid"].dropna().iloc[0])
                fcf_last = abs(fcf_series.iloc[0])
                if fcf_last > 0:
                    payout_fcf = (div_paid / fcf_last) * 100

    streak_pay = dividend_streak_years(by_year)
    streak_growth = dividend_growth_streak(by_year)

    # RPD forward
    forward_div = info.get("forwardAnnualDividendRate") or info.get("dividendRate")
    rpd_forward = np.nan
    if forward_div and price:
        rpd_forward = (forward_div / price) * 100

    # EV/EBITDA y FCF yield
    ev = info.get("enterpriseValue", np.nan)
    ebitda = info.get("ebitda", np.nan)
    ev_ebitda = ev / ebitda if ev and ebitda and ebitda > 0 else np.nan
    marketcap = info.get("marketCap", np.nan)
    fcf_yield = (fcf_series.iloc[0] / marketcap * 100) if marketcap and not cf.empty else np.nan

    metrics = TickerMetrics(
        ticker=ticker.upper(),
        name=name,
        sector=sector,
        price=price,
        rpd_ttm=float(rpd) if pd.notna(rpd) else np.nan,
        dgr5=float(dgr5 * 100.0) if pd.notna(dgr5) else np.nan,
        dgr10=float(dgr10 * 100.0) if pd.notna(dgr10) else np.nan,
        payout=float(payout) if pd.notna(payout) else np.nan,
        per_ttm=float(per_ttm) if pd.notna(per_ttm) else np.nan,
        de_ratio=float(de_ratio) if pd.notna(de_ratio) else np.nan,
        roe=float(roe) if pd.notna(roe) else np.nan,
        streak_years=streak_pay,
        fcf_pos_years=fcf_pos_years
    )
    metrics.rpd_forward = rpd_forward
    metrics.payout_fcf = payout_fcf
    metrics.ev_ebitda = ev_ebitda
    metrics.fcf_yield = fcf_yield
    metrics.streak_growth = streak_growth
    return metrics, by_year, hist["Close"] if "Close" in hist else pd.Series(dtype="float64")


def dividend_growth_streak(series_by_year: pd.Series) -> int:
    """Años consecutivos con crecimiento del dividendo."""
    if series_by_year is None or series_by_year.empty:
        return 0
    s = series_by_year.dropna().sort_index()
    streak = 0
    prev = None
    for _, val in s.items():
        if prev is not None:
            if val > prev:
                streak += 1
            else:
                streak = 0
        prev = val
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












# =============== UI Streamlit ===============

st.set_page_config(page_title="Chequeo Gregorio por Ticker", layout="centered")

st.sidebar.title("Menú")
vista = st.sidebar.radio("Vista", ["Evaluación", "Anexo de ratios"])

if vista == "Anexo de ratios":
    render_anexo()
    st.stop()

# === Vista Evaluación ===
st.title("Chequeo estilo Gregorio 🧱📈")
st.write("Introduce un *ticker* y comprueba si cumple los criterios de **dividendos crecientes** ajustados por sector.")

with st.sidebar:
    st.header("Parámetros de evaluación")
    default_ticker = st.text_input("Ticker", value="JNJ").strip()
    sector_override = st.selectbox(
        "Forzar sector (opcional)",
        options=["(auto)", "Consumer Defensive", "Healthcare", "Utilities", "Communication Services",
                 "Industrials", "Technology", "Energy", "Financial Services", "Consumer Cyclical"],
        index=0
    )
    run_btn = st.button("Evaluar")


if run_btn and default_ticker:
    try:
        metrics, divs_year, price_series = fetch_metrics(default_ticker)
        if sector_override != "(auto)":
            metrics.sector = sector_override

        rules = rules_for_sector(metrics.sector)
        checks = evaluate(metrics, rules)

        st.subheader(f"{metrics.ticker} — {metrics.name}")
        st.caption(f"Sector: **{metrics.sector}** | Precio: **{metrics.price:.2f}**")

        # Tarjetas de métricas
        col1, col2, col3 = st.columns(3)
        col1.metric("RPD TTM", f"{metrics.rpd_ttm:,.2f} %")
        col2.metric("DGR 5 años", f"{metrics.dgr5:,.2f} %")
        col3.metric("DGR 10 años", f"{metrics.dgr10:,.2f} %")

        col4, col5, col6 = st.columns(3)
        col4.metric("Payout", f"{metrics.payout:,.2f} %" if not pd.isna(metrics.payout) else "s/d")
        col5.metric("PER (ttm)", f"{metrics.per_ttm:,.2f}" if not pd.isna(metrics.per_ttm) else "s/d")
        col6.metric("Deuda/Patrimonio", f"{metrics.de_ratio:,.2f} x" if not pd.isna(metrics.de_ratio) else "s/d")

        col7, col8 = st.columns(2)
        col7.metric("ROE", f"{metrics.roe:,.2f} %" if not pd.isna(metrics.roe) else "s/d")
        col8.metric("Racha dividendos", f"{metrics.streak_years} años")

        st.markdown("---")
        st.subheader("Veredicto por criterios (según sector)")

        def verdict_icon(cr: CheckResult) -> str:
            if not cr.ok and cr.borderline:
                return "⚠️"
            return "✅" if cr.ok else "❌"

        table = []
        for k, v in checks.items():
            table.append({
                "Criterio": k,
                "Resultado": v["icon"],
                "Valor": v["msg"],
                "Intervalo recomendado": v["range"]
            })
        st.table(pd.DataFrame(table))

        # Gráfico: dividendos anuales y precio (escala simple)
        st.subheader("Histórico de dividendos anuales y precio")
        fig1 = plt.figure()
        ax = fig1.add_subplot(111)
        if not divs_year.empty:
            divs_year.sort_index().plot(ax=ax)
        ax.set_title("Dividendos por año")
        ax.set_xlabel("Año")
        ax.set_ylabel("Dividendo total por acción")
        st.pyplot(fig1)

        if price_series is not None and not price_series.empty:
            fig2 = plt.figure()
            ax2 = fig2.add_subplot(111)
            price_series.dropna().plot(ax=ax2)
            ax2.set_title("Precio (últimos 12 meses)")
            ax2.set_xlabel("Fecha")
            ax2.set_ylabel("Precio")
            st.pyplot(fig2)

        # Ayuda: mostrar reglas actuales
        st.markdown("---")
        st.subheader("Umbrales usados (según sector)")
        rules_df = pd.DataFrame({
            "RPD min %": [rules.rpd_min],
            "RPD max %": [rules.rpd_max],
            "DGR 5a min %": [rules.dgr5_min],
            "Payout % (min-máx)": [f"{rules.payout_min} - {rules.payout_max}"],
            "PER (mín-máx)": [f"{rules.per_min} - {rules.per_max}"],
            "Deuda/Patrimonio máx (x)": [rules.de_max],
            "ROE min %": [rules.roe_min],
            "Racha dividendos min (años)": [rules.streak_min],
        })
        st.table(rules_df)
        
        flags = red_flags(metrics)
        
        total_score, breakdown = score_company(metrics, rules)
        rec_text, rec_icon, rec_flags = recommendation(total_score, flags)
        
        st.markdown("---")
        st.subheader("Puntuación compuesta (estilo Gregorio)")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Score total", f"{total_score:.0f}/100")
        c2.metric("Dividendo", f"{breakdown['Dividendo']:.0f}")
        c3.metric("Solidez", f"{breakdown['Solidez']:.0f}")
        c4.metric("Valoración", f"{breakdown['Valoración']:.0f}")
        c5.metric("Historial", f"{breakdown['Historial']:.0f}")
        
        st.success(f"{rec_icon} {rec_text}") if rec_icon=="✅" else st.warning(f"{rec_icon} {rec_text}") if rec_icon=="⚠️" else st.error(f"{rec_icon} {rec_text}")
        if flags:
            st.write("**Banderas rojas detectadas:**")
            for f in flags:
                st.write(f"- {f}")



        st.info("Consejo Gregorio: prioriza negocios estables, con dividendo creciente, payout razonable y balance sano. Compra periódica (DCA) y paciencia a 10–30 años.")

    except Exception as e:
        st.error(f"Error al evaluar {default_ticker}: {e}")
        st.exception(e)
else:
    st.write("Introduce un ticker y pulsa **Evaluar**.")
