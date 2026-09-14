"""Catálogo de activos por ISIN: ticker, región y sector.

Antes ambos mapeos (ISIN→ticker y ISIN→región/sector) eran diccionarios fijos en el código, así que
cada valor nuevo que se comprara se quedaba sin ticker (fuera del análisis por acción) y sin
clasificar. Aquí se combinan tres fuentes, por orden de prioridad:

1. El fichero de caché en Data, editable a mano: si la resolución automática se equivoca de listado
   o de sector, se corrige ahí y esa corrección manda.
2. Los mapeos base incluidos en el código, como semilla y red de seguridad sin conexión.
3. Yahoo Finance, solo para lo que no esté en ninguno de los dos anteriores. El resultado se guarda
   en la caché —incluidas las búsquedas sin resultado— para consultar la red una sola vez por ISIN.

El "foco" (Dividendos, Acciones calidad, Indexados...) no se deduce: es una clasificación propia de
la estrategia de cada uno, así que se deja en blanco para que se rellene a mano si se quiere.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.config import get_asset_catalog_path

# Semilla histórica: se mantiene para que todo siga funcionando sin conexión ni caché.
BUILTIN_ISIN_TICKERS = {
    "DE0007074007": "KWS.DE",
    "ES0144580Y14": "IBE.MC",
    "ES0173516115": "REP.MC",
    "US91324P1021": "UNH",
    "FR0000121014": "MC.PA",
    "US7427181091": "PG",
    "US9311421039": "WMT",
    "ES0157261019": "ROVI.MC",
    "XFC000A2YY6Q": "BTC-EUR",
}

BUILTIN_ASSET_METADATA = {
    "DE0007074007": {"region": "Europa", "sector": "Consumo defensivo", "focus": "Acciones calidad"},
    "ES0144580Y14": {"region": "Europa", "sector": "Utilities", "focus": "Dividendos"},
    "ES0173516115": {"region": "Europa", "sector": "Energia", "focus": "Dividendos"},
    "US91324P1021": {"region": "Norteamerica", "sector": "Salud", "focus": "Salud defensiva"},
    "FR0000121014": {"region": "Europa", "sector": "Lujo", "focus": "Acciones calidad"},
    "US7427181091": {"region": "Norteamerica", "sector": "Consumo defensivo", "focus": "Dividendos"},
    "US9311421039": {"region": "Norteamerica", "sector": "Consumo defensivo", "focus": "Dividendos"},
    "ES0157261019": {"region": "Europa", "sector": "Salud", "focus": "Acciones calidad"},
    "XFC000A2YY6Q": {"region": "Global", "sector": "Criptoactivo", "focus": "Alternativos"},
    "LU0389811539": {"region": "Europa", "sector": "Renta variable diversificada", "focus": "Indexados"},
    "LU0996175948": {"region": "Emergentes", "sector": "Renta variable diversificada", "focus": "Indexados"},
    "LU0968301142": {"region": "Frontera", "sector": "Renta variable diversificada", "focus": "Mercados frontera"},
    "IE00B6RVWW34": {"region": "Japon", "sector": "Renta variable diversificada", "focus": "Indexados"},
    "IE00B83YJG36": {"region": "Global", "sector": "Inmobiliario", "focus": "Real estate"},
    "IE00B42W4L06": {"region": "Global", "sector": "Small caps", "focus": "Indexados"},
    "IE0032126645": {"region": "Norteamerica", "sector": "Renta variable diversificada", "focus": "Indexados"},
}

# Sectores de Yahoo (conjunto cerrado) traducidos al vocabulario que ya usa la app.
YAHOO_SECTORS = {
    "Basic Materials": "Materiales basicos",
    "Communication Services": "Comunicacion",
    "Consumer Cyclical": "Consumo ciclico",
    "Consumer Defensive": "Consumo defensivo",
    "Energy": "Energia",
    "Financial Services": "Financiero",
    "Healthcare": "Salud",
    "Industrials": "Industrial",
    "Real Estate": "Inmobiliario",
    "Technology": "Tecnologia",
    "Utilities": "Utilities",
}

# Solo se traducen países de los que se puede afirmar la región con seguridad; lo demás se deja sin
# clasificar antes que arriesgarse a etiquetarlo mal.
YAHOO_REGIONS = {
    "United States": "Norteamerica",
    "Canada": "Norteamerica",
    "Mexico": "Norteamerica",
    "Japan": "Japon",
    "Spain": "Europa",
    "France": "Europa",
    "Germany": "Europa",
    "Italy": "Europa",
    "Netherlands": "Europa",
    "Portugal": "Europa",
    "Belgium": "Europa",
    "Ireland": "Europa",
    "Switzerland": "Europa",
    "Austria": "Europa",
    "Denmark": "Europa",
    "Sweden": "Europa",
    "Norway": "Europa",
    "Finland": "Europa",
    "United Kingdom": "Europa",
    "Luxembourg": "Europa",
}

METADATA_FIELDS = ("region", "sector", "focus")


def load_cache(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Caché completa por ISIN. Un ticker null significa "ya se buscó y Yahoo no lo encuentra", para
    no repetir la consulta en cada importación."""
    path = path or get_asset_catalog_path()
    raw: Any = None
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            raw = None
    else:
        # Compatibilidad con el formato anterior, que solo guardaba {isin: ticker}.
        legacy = path.with_name("isin_tickers.json")
        if legacy.exists():
            try:
                raw = json.loads(legacy.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                raw = None
    if not isinstance(raw, dict):
        return {}

    cache: dict[str, dict[str, Any]] = {}
    for isin, value in raw.items():
        if not isin:
            continue
        if isinstance(value, dict):
            cache[str(isin)] = dict(value)
        else:
            cache[str(isin)] = {"ticker": str(value) if value else None}
    return cache


def save_cache(cache: dict[str, dict[str, Any]], path: Path | None = None) -> None:
    path = path or get_asset_catalog_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def known_tickers(path: Path | None = None) -> dict[str, str]:
    """Tickers utilizables sin tocar la red: semilla + caché (la caché manda para poder corregir)."""
    cached = {isin: record.get("ticker") for isin, record in load_cache(path).items()}
    return {**BUILTIN_ISIN_TICKERS, **{isin: ticker for isin, ticker in cached.items() if ticker}}


def known_metadata(path: Path | None = None) -> dict[str, dict[str, str]]:
    """Región/sector/foco conocidos sin tocar la red: semilla + caché (la caché manda)."""
    catalog = {isin: dict(meta) for isin, meta in BUILTIN_ASSET_METADATA.items()}
    for isin, record in load_cache(path).items():
        fields = {field: record[field] for field in METADATA_FIELDS if record.get(field)}
        if fields:
            catalog[isin] = {**catalog.get(isin, {}), **fields}
    return catalog


def _update_cache(path: Path | None, updates: dict[str, dict[str, Any]]) -> None:
    if not updates:
        return
    cache = load_cache(path)
    for isin, fields in updates.items():
        cache[isin] = {**cache.get(isin, {}), **fields}
    save_cache(cache, path)


def lookup_ticker_online(isin: str) -> str | None:
    try:
        from yfinance.utils import get_ticker_by_isin  # type: ignore

        ticker = get_ticker_by_isin(isin)
    except Exception:
        return None
    if not ticker or not isinstance(ticker, str):
        return None
    ticker = ticker.strip()
    # Yahoo devuelve cadena vacía o el propio ISIN cuando no encuentra nada.
    if not ticker or ticker.upper() == isin.upper():
        return None
    return ticker


def lookup_metadata_online(ticker: str) -> dict[str, str]:
    """Región y sector según Yahoo. Solo se devuelve lo que se puede traducir con seguridad."""
    try:
        from yfinance import Ticker  # type: ignore

        info = Ticker(ticker).info or {}
    except Exception:
        return {}
    metadata = {}
    sector = YAHOO_SECTORS.get(str(info.get("sector") or "").strip())
    if sector:
        metadata["sector"] = sector
    region = YAHOO_REGIONS.get(str(info.get("country") or "").strip())
    if region:
        metadata["region"] = region
    return metadata


def resolve_tickers(isins: list[str], path: Path | None = None, allow_network: bool = True) -> dict[str, str]:
    """Devuelve los tickers de los ISIN pedidos, buscando en la red solo los desconocidos."""
    mapping = known_tickers(path)
    cache = load_cache(path)
    pending = sorted({isin for isin in isins if isin and isin not in mapping and "ticker" not in cache.get(isin, {})})
    if not pending or not allow_network:
        return mapping

    updates = {isin: {"ticker": lookup_ticker_online(isin)} for isin in pending}
    _update_cache(path, updates)
    mapping.update({isin: fields["ticker"] for isin, fields in updates.items() if fields["ticker"]})
    return mapping


def resolve_metadata(
    isin_tickers: dict[str, str], path: Path | None = None, allow_network: bool = True
) -> dict[str, dict[str, str]]:
    """Completa región/sector de los ISIN que no los tengan, a partir de su ticker."""
    catalog = known_metadata(path)
    cache = load_cache(path)
    pending = sorted(
        isin
        for isin, ticker in isin_tickers.items()
        if ticker and not catalog.get(isin, {}).get("sector") and "sector" not in cache.get(isin, {})
    )
    if not pending or not allow_network:
        return catalog

    updates: dict[str, dict[str, Any]] = {}
    for isin in pending:
        found = lookup_metadata_online(isin_tickers[isin])
        # Se guarda el sector aunque venga vacío: marca que ya se intentó y evita reconsultar.
        updates[isin] = {"sector": found.get("sector"), **({"region": found["region"]} if found.get("region") else {})}
    _update_cache(path, updates)

    for isin, fields in updates.items():
        clean = {field: value for field, value in fields.items() if value}
        if clean:
            catalog[isin] = {**catalog.get(isin, {}), **clean}
    return catalog


def fill_missing_tickers(positions: list[dict[str, Any]], path: Path | None = None, allow_network: bool = False) -> None:
    """Completa in-place el ticker de las posiciones que no lo tengan.

    Por defecto no usa la red: al cargar fotos ya guardadas solo se rellena con lo ya conocido, para
    que abrir Cartera no dependa de Yahoo. La resolución online se hace al importar/reconstruir.
    """
    missing = [str(position.get("isin")) for position in positions if not position.get("ticker") and position.get("isin")]
    if not missing:
        return
    mapping = resolve_tickers(missing, path, allow_network=allow_network)
    for position in positions:
        if position.get("ticker") or not position.get("isin"):
            continue
        ticker = mapping.get(str(position["isin"]))
        if ticker:
            position["ticker"] = ticker


def fill_missing_metadata(positions: list[dict[str, Any]], path: Path | None = None, allow_network: bool = False) -> None:
    """Completa in-place la región/sector de las posiciones sin clasificar (mismo criterio de red)."""
    pending = {
        str(position["isin"]): str(position["ticker"])
        for position in positions
        if position.get("isin") and position.get("ticker") and position.get("sector") in (None, "", "Sin clasificar")
    }
    if not pending:
        return
    catalog = resolve_metadata(pending, path, allow_network=allow_network)
    for position in positions:
        metadata = catalog.get(str(position.get("isin") or ""))
        if not metadata:
            continue
        for field in METADATA_FIELDS:
            if metadata.get(field) and position.get(field) in (None, "", "Sin clasificar"):
                position[field] = metadata[field]
