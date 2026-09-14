import json

from backend.domain import asset_catalog


def test_known_tickers_merges_builtin_with_cache(tmp_path):
    path = tmp_path / "asset_catalog.json"
    path.write_text(json.dumps({"US67066G1040": {"ticker": "NVDA"}}), encoding="utf-8")

    mapping = asset_catalog.known_tickers(path)

    assert mapping["US67066G1040"] == "NVDA"  # viene de la caché
    assert mapping["US7427181091"] == "PG"  # sigue estando la semilla del código


def test_cache_overrides_builtin_so_a_wrong_lookup_can_be_corrected_by_hand(tmp_path):
    path = tmp_path / "asset_catalog.json"
    path.write_text(
        json.dumps({"US7427181091": {"ticker": "PG.MI", "sector": "Otro sector"}}), encoding="utf-8"
    )

    assert asset_catalog.known_tickers(path)["US7427181091"] == "PG.MI"
    assert asset_catalog.known_metadata(path)["US7427181091"]["sector"] == "Otro sector"
    # Lo que no se corrige a mano conserva el valor de la semilla.
    assert asset_catalog.known_metadata(path)["US7427181091"]["focus"] == "Dividendos"


def test_legacy_ticker_only_cache_is_still_readable(tmp_path):
    """El primer formato guardaba {isin: ticker}; no debe perderse al pasar al catálogo."""
    (tmp_path / "isin_tickers.json").write_text(json.dumps({"FR0000120073": "AI.PA"}), encoding="utf-8")

    assert asset_catalog.known_tickers(tmp_path / "asset_catalog.json")["FR0000120073"] == "AI.PA"


def test_resolve_tickers_only_queries_unknown_isins_and_persists_them(tmp_path, monkeypatch):
    path = tmp_path / "asset_catalog.json"
    consultados = []

    def fake_lookup(isin):
        consultados.append(isin)
        return {"FR0000120073": "AI.PA", "NL0011585146": "RACE.MI"}.get(isin)

    monkeypatch.setattr(asset_catalog, "lookup_ticker_online", fake_lookup)

    mapping = asset_catalog.resolve_tickers(["US7427181091", "FR0000120073", "NL0011585146"], path)

    # El ISIN que ya estaba en la semilla no se consulta por red.
    assert consultados == ["FR0000120073", "NL0011585146"]
    assert mapping["FR0000120073"] == "AI.PA"
    assert mapping["NL0011585146"] == "RACE.MI"
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "FR0000120073": {"ticker": "AI.PA"},
        "NL0011585146": {"ticker": "RACE.MI"},
    }


def test_isins_without_result_are_cached_so_they_are_not_queried_again(tmp_path, monkeypatch):
    """Los fondos y demás ISIN que Yahoo no encuentra no deben reconsultarse en cada importación."""
    path = tmp_path / "asset_catalog.json"
    consultados = []

    monkeypatch.setattr(asset_catalog, "lookup_ticker_online", lambda isin: consultados.append(isin))

    asset_catalog.resolve_tickers(["LU0996175948"], path)
    assert consultados == ["LU0996175948"]
    assert json.loads(path.read_text(encoding="utf-8")) == {"LU0996175948": {"ticker": None}}

    mapping = asset_catalog.resolve_tickers(["LU0996175948"], path)
    assert consultados == ["LU0996175948"]  # no se vuelve a preguntar
    assert "LU0996175948" not in mapping


def test_resolve_metadata_translates_yahoo_sector_and_country(tmp_path, monkeypatch):
    path = tmp_path / "asset_catalog.json"
    monkeypatch.setattr(
        asset_catalog,
        "lookup_metadata_online",
        lambda ticker: {"AI.PA": {"sector": "Materiales basicos", "region": "Europa"}}.get(ticker, {}),
    )

    catalog = asset_catalog.resolve_metadata({"FR0000120073": "AI.PA"}, path)

    assert catalog["FR0000120073"] == {"sector": "Materiales basicos", "region": "Europa"}
    assert json.loads(path.read_text(encoding="utf-8"))["FR0000120073"]["sector"] == "Materiales basicos"


def test_resolve_metadata_does_not_overwrite_the_builtin_classification(tmp_path, monkeypatch):
    """LVMH está clasificado a mano como 'Lujo'; Yahoo diría 'Consumo ciclico' y no debe pisarlo."""

    def explode(ticker):  # pragma: no cover - no debería llamarse
        raise AssertionError("no debe consultarse un ISIN ya clasificado")

    monkeypatch.setattr(asset_catalog, "lookup_metadata_online", explode)

    catalog = asset_catalog.resolve_metadata({"FR0000121014": "MC.PA"}, tmp_path / "asset_catalog.json")

    assert catalog["FR0000121014"]["sector"] == "Lujo"


def test_metadata_without_result_is_cached_too(tmp_path, monkeypatch):
    path = tmp_path / "asset_catalog.json"
    consultados = []

    def fake_lookup(ticker):
        consultados.append(ticker)
        return {}

    monkeypatch.setattr(asset_catalog, "lookup_metadata_online", fake_lookup)

    asset_catalog.resolve_metadata({"US67066G1040": "NVDA"}, path)
    asset_catalog.resolve_metadata({"US67066G1040": "NVDA"}, path)

    assert consultados == ["NVDA"]


def test_fill_missing_metadata_completes_positions_in_place(tmp_path, monkeypatch):
    path = tmp_path / "asset_catalog.json"
    path.write_text(
        json.dumps({"US67066G1040": {"ticker": "NVDA", "sector": "Tecnologia", "region": "Norteamerica"}}),
        encoding="utf-8",
    )

    positions = [
        {"isin": "US67066G1040", "ticker": "NVDA", "sector": "Sin clasificar", "region": "Sin clasificar"},
        {"isin": "FR0000121014", "ticker": "MC.PA", "sector": "Lujo", "region": "Europa"},
    ]

    asset_catalog.fill_missing_metadata(positions, path, allow_network=False)

    assert positions[0]["sector"] == "Tecnologia"
    assert positions[0]["region"] == "Norteamerica"
    assert positions[1]["sector"] == "Lujo"  # no se toca lo ya clasificado


def test_lookup_metadata_ignores_sectors_and_countries_it_cannot_map(monkeypatch):
    import yfinance

    class FakeTicker:
        def __init__(self, symbol):
            self.info = {"sector": "Sector Inventado", "country": "Narnia"}

    monkeypatch.setattr(yfinance, "Ticker", FakeTicker)
    assert asset_catalog.lookup_metadata_online("XXX") == {}


def test_lookup_ignores_yahoo_non_answers(monkeypatch):
    import yfinance.utils as yf_utils

    monkeypatch.setattr(yf_utils, "get_ticker_by_isin", lambda isin: "")
    assert asset_catalog.lookup_ticker_online("FR0000120073") is None

    # Yahoo a veces devuelve el propio ISIN cuando no encuentra el valor.
    monkeypatch.setattr(yf_utils, "get_ticker_by_isin", lambda isin: isin)
    assert asset_catalog.lookup_ticker_online("FR0000120073") is None
