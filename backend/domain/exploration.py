"""Exploración de un universo de empresas en dos fases.

Analizar a fondo las 500 del S&P son entre 20 y 60 minutos de descargas y acabaría con Yahoo
cortando las peticiones, así que primero se hace una criba barata y solo las finalistas pasan por
el análisis completo:

- Fase 1 (criba): una descarga en bloque de 12 años de precios y dividendos de todo el universo. Con eso se
  calculan RPD, RPD media de 5 años, crecimiento del dividendo y racha, que son las métricas del
  bloque Dividendo e Historial (60 de los 100 puntos del score). Cuesta segundos, no minutos.
- Fase 2: las mejores de la criba pasan por `analyze_ticker`, que da el score real con payout,
  ROE, deuda, valoración y banderas rojas.

El resultado es "las mejores N de las finalistas que mejor pintaban", no un ranking garantizado
del universo entero: una empresa con RPD baja hoy pero cuentas impecables puede caerse en la criba.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from backend.domain.investments import (
    average_of_last,
    cagr_from_annual,
    dividend_growth_streak,
    dividend_yield_history,
    dividends_by_year,
    dividends_by_year_clean,
    dividends_ttm,
    dividend_streak_years,
)
from backend.domain.universes import get_universe, get_members

DOWNLOAD_CHUNK = 100
SCREEN_PAGE = 250


def collect_candidates(universe_key: str) -> list[str]:
    """Los tickers a cribar: la lista del índice, o lo que devuelve el buscador por región."""
    universe = get_universe(universe_key)
    if universe.kind == "index":
        return get_members(universe_key)
    return _screen_candidates(universe)


def _screen_candidates(universe) -> list[str]:
    """Las mayores de cada país del universo, una consulta por región.

    Se pregunta país a país en vez de todo junto porque la capitalización que devuelve Yahoo está
    en moneda local: un único ranking global mezclaría rupias con dólares y lo ordenaría mal.
    """
    import yfinance as yf
    from yfinance import EquityQuery

    symbols: list[str] = []
    for region in universe.regions:
        query = EquityQuery("and", [EquityQuery("eq", ["region", region]), EquityQuery("gt", ["intradaymarketcap", 0])])
        try:
            page = yf.screen(query, size=universe.per_region, sortField="intradaymarketcap", sortAsc=False)
        except Exception as exc:
            if symbols:
                continue  # un país que falla no debe tumbar la exploración entera
            raise RuntimeError(f"No se pudo consultar el buscador de Yahoo: {exc}") from exc
        for quote in page.get("quotes") or []:
            symbol = str(quote.get("symbol") or "").strip()
            if symbol and symbol not in symbols and quote.get("quoteType") == "EQUITY":
                symbols.append(symbol)
    return symbols


def screen_chunk(tickers: list[str]) -> list[dict[str, Any]]:
    """Criba de un grupo de tickers con una sola descarga en bloque."""
    if not tickers:
        return []

    import yfinance as yf

    try:
        data = yf.download(
            tickers,
            # 12 años: con 6 no había suficientes ejercicios cerrados para el crecimiento a 5 años
            # (salía siempre 0) ni para distinguir rachas de pago, que se clavaban en la ventana.
            period="12y",
            actions=True,
            group_by="ticker",
            progress=False,
            auto_adjust=False,
            threads=True,
        )
    except Exception as exc:
        raise RuntimeError(f"No se pudieron descargar los datos de la criba: {exc}") from exc

    rows = []
    for ticker in tickers:
        try:
            frame = data[ticker] if len(tickers) > 1 else data
        except (KeyError, TypeError):
            continue
        row = _screen_row(ticker, frame)
        if row is not None:
            rows.append(row)
    return rows


def _screen_row(ticker: str, frame: pd.DataFrame) -> dict[str, Any] | None:
    if frame is None or frame.empty or "Close" not in frame:
        return None
    closes = frame["Close"].dropna()
    if closes.empty:
        return None
    price = float(closes.iloc[-1])
    if price <= 0:
        return None

    dividends = frame["Dividends"].dropna() if "Dividends" in frame else pd.Series(dtype="float64")
    dividends = dividends[dividends > 0]
    by_year_raw = dividends_by_year(dividends)
    by_year_clean = dividends_by_year_clean(dividends)
    ttm = dividends_ttm(dividends)
    rpd = (ttm / price * 100.0) if ttm > 0 else 0.0
    rpd_avg5 = average_of_last(dividend_yield_history(by_year_raw, frame), 5)
    dgr5 = cagr_from_annual(by_year_clean, 5)

    return {
        "ticker": ticker,
        "price": price,
        "rpd_ttm": rpd,
        "rpd_avg5": None if math.isnan(rpd_avg5) else rpd_avg5,
        "dgr5": None if math.isnan(dgr5) else dgr5 * 100.0,
        "streak_years": dividend_streak_years(by_year_raw),
        "streak_growth": dividend_growth_streak(by_year_clean),
        "prescore": None,
    }


def prescore(row: dict[str, Any]) -> float:
    """Nota de la criba (0-100) con lo único que se puede calcular barato.

    No es el score real: le faltan payout, ROE, deuda y valoración. Solo sirve para decidir a
    quién merece la pena dedicarle una descarga completa.
    """
    rpd = _number(row.get("rpd_ttm"))
    dgr5 = _number(row.get("dgr5"))
    streak = _number(row.get("streak_years"))
    growth = _number(row.get("streak_growth"))
    rpd_avg5 = _number(row.get("rpd_avg5"))

    if rpd <= 0:
        return 0.0  # sin dividendo no encaja en una cartera de dividendos crecientes

    # RPD: 40 puntos, con techo para no premiar las trampas de rentabilidad altísima.
    rpd_points = 40.0 * min(rpd, 6.0) / 6.0 if rpd <= 8.0 else 20.0
    # Crecimiento del dividendo: 25 puntos.
    growth_points = 25.0 * max(0.0, min(dgr5, 12.0)) / 12.0
    # Historial: 25 puntos entre racha de pago y racha de subidas. Los topes son 12 y 10 porque la
    # criba solo mira 12 años: pedir más sería castigar a todas por igual por una ventana corta.
    streak_points = 15.0 * min(streak, 12.0) / 12.0 + 10.0 * min(growth, 10.0) / 10.0
    # Valoración frente a su propia media: 10 puntos si cotiza por encima de su RPD histórica.
    value_points = 0.0
    if rpd_avg5 > 0:
        value_points = 10.0 * max(0.0, min((rpd - rpd_avg5) / rpd_avg5, 1.0))

    return round(rpd_points + growth_points + streak_points + value_points, 2)


def rank_candidates(rows: list[dict[str, Any]], finalists: int) -> list[dict[str, Any]]:
    scored = []
    for row in rows:
        row = dict(row)
        row["prescore"] = prescore(row)
        scored.append(row)
    scored.sort(key=lambda item: item["prescore"], reverse=True)
    return scored[:finalists]


def _number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(number) else number
