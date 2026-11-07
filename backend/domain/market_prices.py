"""Precio de cierre de mes de un activo, para valorar lo que el archivo no trae.

El reparto de papeles es: el archivo del usuario dice **qué tiene y a cuánto lo compró**; el valor
de mercado se busca fuera. Esto último es lo que hace este módulo.

Sirve igual para acciones y para fondos. Los fondos no cotizan en continuo —la gestora publica un
único valor liquidativo al día— y Yahoo los identifica con tickers propios (`0P…`), que el catálogo
de activos resuelve a partir del ISIN.

Igual que con los tipos de cambio, el precio de un mes cerrado no vuelve a cambiar, así que se
cachea en disco: sin caché, cada lectura del archivo dispararía una consulta por activo y mes.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

from backend.config import get_investments_path

CACHE_NAME = "precios_mensuales.json"


def cache_path():
    return get_investments_path() / CACHE_NAME


def load_cache() -> dict[str, float]:
    path = cache_path()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {str(key): float(value) for key, value in payload.items() if _is_number(value)}


def save_cache(prices: dict[str, float]) -> None:
    path = cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(prices, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    except OSError:
        return


def key(isin: str, month: str) -> str:
    return f"{(isin or '').upper()}:{month}"


def get_prices(pairs: Iterable[tuple[str, str]], allow_network: bool = True) -> dict[str, float]:
    """Cierre de cada (ISIN, mes) pedido. Lo que no se sepa simplemente no aparece."""
    pairs = {(str(isin).upper(), str(month)) for isin, month in pairs if isin and month}
    prices = load_cache()
    missing = {pair for pair in pairs if key(*pair) not in prices}
    if not missing or not allow_network:
        return prices

    by_isin: dict[str, list[str]] = {}
    for isin, month in missing:
        by_isin.setdefault(isin, []).append(month)

    tickers = _resolve_tickers(sorted(by_isin))
    for isin, months in by_isin.items():
        ticker = tickers.get(isin)
        if not ticker:
            continue
        prices.update(_fetch(isin, ticker, sorted(months)))
    save_cache(prices)
    return prices


def _resolve_tickers(isins: list[str]) -> dict[str, str]:
    """El catálogo ya sabe pasar de ISIN a ticker y cachea también los que no encuentra."""
    try:
        from backend.domain.asset_catalog import resolve_tickers
    except ImportError:  # pragma: no cover
        return {}
    try:
        return resolve_tickers(isins)
    except Exception:
        return {}


def _fetch(isin: str, ticker: str, months: list[str]) -> dict[str, float]:
    try:
        import pandas as pd
        import yfinance as yf
    except ImportError:  # pragma: no cover - dependencia del contenedor
        return {}

    try:
        history = yf.Ticker(ticker).history(period="5y", interval="1d")["Close"].dropna()
    except Exception:
        return {}
    if history.empty:
        return {}

    history.index = pd.to_datetime(history.index, utc=True).tz_localize(None)
    monthly = history.resample("ME").last()

    prices: dict[str, float] = {}
    for month in months:
        try:
            cutoff = pd.Period(month, freq="M").to_timestamp(how="end").normalize()
        except Exception:
            continue
        window = monthly[monthly.index <= cutoff]
        if window.empty:
            continue
        close = float(window.iloc[-1])
        if close > 0:
            prices[key(isin, month)] = round(close, 6)
    return prices


def _is_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number
