"""Tipos de cambio a euros, mes a mes.

Una posición puede venir en divisa y sin el importe en euros: el extracto de DeGiro cotiza PG en
dólares, y multiplicar cantidad por precio daría dólares, no euros. Aquí se busca el cambio del
cierre de cada mes para poder valorarla.

Se cachea en disco porque es un dato histórico que no cambia: el euro-dólar del 31 de marzo de 2026
será siempre el mismo, así que pedirlo una vez basta. Sin caché, cada carga del archivo de
posiciones dispararía una consulta por mes y divisa.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any

from backend.config import get_investments_path

BASE = "EUR"
CACHE_NAME = "tipos_cambio.json"


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


def save_cache(rates: dict[str, float]) -> None:
    path = cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rates, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    except OSError:
        return


def get_rates(pairs: set[tuple[str, str]], allow_network: bool = True) -> dict[str, float]:
    """Cambio de una divisa a euros al cierre de cada mes pedido.

    La clave es `<MONEDA>:<AAAA-MM>` y el valor, cuántos euros vale una unidad de esa divisa.
    """
    rates = load_cache()
    missing = {(currency, month) for currency, month in pairs if _key(currency, month) not in rates}
    if not missing or not allow_network:
        return rates

    for currency in sorted({currency for currency, _ in missing}):
        months = sorted(month for other, month in missing if other == currency)
        fetched = _fetch_currency(currency, months)
        rates.update(fetched)
    save_cache(rates)
    return rates


def convert(amount: float, currency: str, month: str, rates: dict[str, float]) -> float | None:
    """Pasa un importe a euros. Devuelve None si no se conoce el cambio, en vez de inventarlo."""
    if not _is_number(amount):
        return None
    currency = (currency or BASE).upper()
    if currency == BASE:
        return round(float(amount), 2)
    rate = rates.get(_key(currency, month))
    if not rate:
        return None
    return round(float(amount) * rate, 2)


def _fetch_currency(currency: str, months: list[str]) -> dict[str, float]:
    if currency.upper() == BASE or not months:
        return {}
    try:
        import pandas as pd
        import yfinance as yf
    except ImportError:  # pragma: no cover - dependencia del contenedor
        return {}

    # Yahoo cotiza el par como EURUSD=X, es decir cuántos dólares vale un euro: para pasar de
    # dólares a euros hay que invertirlo.
    try:
        history = yf.Ticker(f"{BASE}{currency.upper()}=X").history(period="10y", interval="1d")["Close"].dropna()
    except Exception:
        return {}
    if history.empty:
        return {}

    history.index = pd.to_datetime(history.index, utc=True).tz_localize(None)
    monthly = history.resample("ME").last()

    rates: dict[str, float] = {}
    for month in months:
        try:
            cutoff = pd.Period(month, freq="M").to_timestamp(how="end").normalize()
        except Exception:
            continue
        window = monthly[monthly.index <= cutoff]
        if window.empty:
            continue
        quote = float(window.iloc[-1])
        if quote > 0:
            rates[_key(currency, month)] = round(1.0 / quote, 8)
    return rates


def _key(currency: str, month: str) -> str:
    return f"{(currency or BASE).upper()}:{month}"


def _is_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number
