#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 00:20:22 2025

@author: alex
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import time

# ============
# 1. Sectores recomendados según Gregorio
# ============
sectores_recomendados = {
    "Utilities": {"PER": (10, 20), "Payout": (0.5, 0.8), "DeudaEBITDA": 4},
    "Consumer Defensive": {"PER": (15, 25), "Payout": (0.4, 0.7), "DeudaEBITDA": 3},
    "Healthcare": {"PER": (15, 25), "Payout": (0.3, 0.6), "DeudaEBITDA": 3},
    "Financial Services": {"PER": (8, 15), "Payout": (0.3, 0.6), "DeudaEBITDA": 15}, # banca especial
    "Communication Services": {"PER": (10, 20), "Payout": (0.4, 0.7), "DeudaEBITDA": 3},
    "Industrials": {"PER": (10, 20), "Payout": (0.4, 0.7), "DeudaEBITDA": 4}, # infraestructuras
}

# ============
# 2. Funciones auxiliares
# ============

def estadisticas_ratio(serie):
    if len(serie) == 0:
        return None, None, None, None, None
    serie = [x for x in serie if x is not None]
    if len(serie) == 0:
        return None, None, None, None, None
    return (np.mean(serie), np.median(serie), np.std(serie),
            np.percentile(serie, 25), np.percentile(serie, 75))

def calcular_ratios(info):
    per = info.get("trailingPE")
    payout = info.get("payoutRatio")
    deuda = info.get("totalDebt")
    ebitda = info.get("ebitda")
    deuda_ebitda = deuda / ebitda if deuda and ebitda and ebitda > 0 else None
    return per, payout, deuda_ebitda

def cumple_ratios(sector, per, payout, deuda_ebitda):
    if sector not in sectores_recomendados:
        return False
    reglas = sectores_recomendados[sector]
    ok_per = per and reglas["PER"][0] <= per <= reglas["PER"][1]
    ok_payout = payout and reglas["Payout"][0] <= payout <= reglas["Payout"][1]
    ok_deuda = deuda_ebitda and deuda_ebitda <= reglas["DeudaEBITDA"]
    return ok_per and ok_payout and ok_deuda

# ============
# 3. Descargar listas de tickers
# ============

def obtener_sp500():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    html = requests.get(url, headers=headers).text
    tabla = pd.read_html(html, header=0)[0]
    return tabla["Symbol"].tolist()

def obtener_ibex35():
    url = "https://en.wikipedia.org/wiki/IBEX_35"
    headers = {"User-Agent": "Mozilla/5.0"}
    html = requests.get(url, headers=headers).text
    tablas = pd.read_html(html, header=0)
    return tablas[2].iloc[:, 0].tolist()  # tabla con tickers

# ============
# 4. Análisis de empresa básico
# ============

def analizar_empresa(ticker, años_hist=5):
    try:
        empresa = yf.Ticker(ticker)
        info = empresa.info

        eps = info.get("trailingEps")
        dividend_yield = info.get("dividendYield")
        dividendos = empresa.dividends

        crecimiento_div = False
        if not dividendos.empty:
            dividendos = dividendos.resample("YE").sum()
            if len(dividendos) >= años_hist:
                crecimiento_div = dividendos.iloc[-1] > dividendos.iloc[-años_hist]

        sector = info.get("sector")
        industria = info.get("industry")

        per, payout, deuda_ebitda = calcular_ratios(info)
        ratios_ok = cumple_ratios(sector, per, payout, deuda_ebitda)

        return {
            "Ticker": ticker,
            "Sector": sector,
            "Industria": industria,
            "EPS": eps,
            "Dividend Yield": dividend_yield,
            "Rentable": eps is not None and eps > 0,
            "Paga Dividendos": dividend_yield is not None and dividend_yield > 0,
            "Historial Dividendos": not dividendos.empty,
            "Dividendo Creciente": crecimiento_div,
            "PER": per,
            "Payout": payout,
            "Deuda/EBITDA": deuda_ebitda,
            "Ratios OK": ratios_ok
        }
    except Exception:
        return None

def filtrar_empresas(lista_tickers, años_hist=5):
    resultados = [analizar_empresa(t, años_hist) for t in lista_tickers]
    df = pd.DataFrame([r for r in resultados if r is not None])
    df_filtrado = df[df["Rentable"] & df["Paga Dividendos"] &
                     df["Historial Dividendos"] & df["Dividendo Creciente"] &
                     df["Ratios OK"]]
    return df, df_filtrado

# ============
# 5. Análisis avanzado con históricos y estadísticas
# ============

def analizar_empresa_avanzado(ticker, años_hist=5):
    empresa = yf.Ticker(ticker)
    info = empresa.info

    per = info.get("trailingPE")
    payout = info.get("payoutRatio")
    dividend_yield = info.get("dividendYield")
    deuda = info.get("totalDebt")
    ebitda = info.get("ebitda")
    deuda_ebitda = deuda / ebitda if deuda and ebitda and ebitda > 0 else None
    eps_actual = info.get("trailingEps")

    hist = empresa.financials.T
    hist_ratios = {k: {} for k in ["PER", "Payout", "Deuda/EBITDA", "Dividend Yield", "EPS"]}

    precio = info.get("currentPrice")
    acciones = info.get("sharesOutstanding")

    try:
        for fecha in hist.index[:años_hist]:
            beneficios = hist.loc[fecha, "Net Income"] if "Net Income" in hist.columns else None
            eps = beneficios/acciones if beneficios and acciones else None
            if eps and eps > 0:
                hist_ratios["EPS"][fecha.year] = eps
                hist_ratios["PER"][fecha.year] = precio/eps

            dividendos = empresa.dividends[empresa.dividends.index.year == fecha.year].sum()
            if eps and eps > 0 and dividendos > 0:
                hist_ratios["Payout"][fecha.year] = dividendos/eps
            if precio and dividendos > 0:
                hist_ratios["Dividend Yield"][fecha.year] = dividendos/precio

            deuda_hist = info.get("totalDebt")
            ebitda_hist = hist.loc[fecha, "EBITDA"] if "EBITDA" in hist.columns else None
            if deuda_hist and ebitda_hist and ebitda_hist > 0:
                hist_ratios["Deuda/EBITDA"][fecha.year] = deuda_hist/ebitda_hist
    except Exception:
        pass

    return {
        "Ticker": ticker,
        "Sector": info.get("sector"),
        "Industria": info.get("industry"),
        "PER actual": per,
        "Payout actual": payout,
        "Deuda/EBITDA actual": deuda_ebitda,
        "Dividend Yield actual": dividend_yield,
        "EPS actual": eps_actual,
        "PER (stats)": estadisticas_ratio(list(hist_ratios["PER"].values())),
        "Payout (stats)": estadisticas_ratio(list(hist_ratios["Payout"].values())),
        "Deuda/EBITDA (stats)": estadisticas_ratio(list(hist_ratios["Deuda/EBITDA"].values())),
        "Dividend Yield (stats)": estadisticas_ratio(list(hist_ratios["Dividend Yield"].values())),
        "EPS (stats)": estadisticas_ratio(list(hist_ratios["EPS"].values())),
        "Históricos": hist_ratios
    }

# ============
# 6. Dashboard gráfico
# ============

def graficar_dashboard_ratios(ticker, analisis):
    ratios = {
        "PER": (analisis["PER actual"], analisis["PER (stats)"], analisis["Históricos"]["PER"]),
        "Payout": (analisis["Payout actual"], analisis["Payout (stats)"], analisis["Históricos"]["Payout"]),
        "Deuda/EBITDA": (analisis["Deuda/EBITDA actual"], analisis["Deuda/EBITDA (stats)"], analisis["Históricos"]["Deuda/EBITDA"]),
        "Dividend Yield": (analisis["Dividend Yield actual"], analisis["Dividend Yield (stats)"], analisis["Históricos"]["Dividend Yield"]),
        "EPS": (analisis["EPS actual"], analisis["EPS (stats)"], analisis["Históricos"]["EPS"])
    }

    n = len(ratios)
    fig, axes = plt.subplots(n, 2, figsize=(12, n*3))

    for i, (nombre, (actual, stats, serie)) in enumerate(ratios.items()):
        # Boxplot simplificado
        ax_box = axes[i, 0]
        if stats and stats[0] is not None:
            _, mediana, _, p25, p75 = stats
            ax_box.fill_between([0.8, 1.2], p25, p75, color="green", alpha=0.2, label="Rango 25-75%")
            ax_box.scatter(1, mediana, color="blue", marker="|", s=200, label="Mediana histórica")
            if actual is not None:
                ax_box.scatter(1, actual, color="red", s=100, zorder=5, label="Valor actual")
        ax_box.set_xticks([1])
        ax_box.set_xticklabels([nombre])
        ax_box.set_title(f"{nombre} - Boxplot")
        ax_box.grid(True, linestyle="--", alpha=0.5)
        ax_box.legend()

        # Evolución temporal
        ax_evo = axes[i, 1]
        if serie:
            años = list(serie.keys())
            valores = list(serie.values())
            ax_evo.plot(años, valores, marker="o", color="blue", label="Histórico")
            if actual is not None:
                ax_evo.axhline(actual, color="red", linestyle="--", label="Actual")
        ax_evo.set_title(f"{nombre} - Evolución")
        ax_evo.set_xlabel("Año")
        ax_evo.grid(True, linestyle="--", alpha=0.5)
        ax_evo.legend()

    plt.suptitle(f"Dashboard de Ratios - {ticker}", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()