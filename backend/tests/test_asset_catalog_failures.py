import json

import pytest

from backend.domain import asset_catalog


def test_un_fallo_de_red_no_se_cachea_como_isin_irresoluble(tmp_path, monkeypatch):
    """Pasó de verdad: el Vanguard US 500 quedó grabado como sin ticker por un corte de Yahoo, y
    el cacheo negativo impedía reintentarlo nunca más."""
    path = tmp_path / "asset_catalog.json"

    def falla(isin):
        raise asset_catalog.LookupFailed("timeout")

    monkeypatch.setattr(asset_catalog, "lookup_ticker_online", falla)
    asset_catalog.resolve_tickers(["IE0032126645"], path)

    assert not path.exists() or "IE0032126645" not in json.loads(path.read_text(encoding="utf-8"))


def test_un_isin_que_de_verdad_no_existe_si_se_cachea(tmp_path, monkeypatch):
    path = tmp_path / "asset_catalog.json"
    monkeypatch.setattr(asset_catalog, "lookup_ticker_online", lambda isin: None)

    asset_catalog.resolve_tickers(["XX0000000000"], path)

    assert json.loads(path.read_text(encoding="utf-8"))["XX0000000000"] == {"ticker": None}


def test_tras_un_fallo_de_red_el_isin_se_reintenta(tmp_path, monkeypatch):
    path = tmp_path / "asset_catalog.json"
    intentos = []

    def primero_falla(isin):
        intentos.append(isin)
        if len(intentos) == 1:
            raise asset_catalog.LookupFailed("timeout")
        return "0P00000SUJ.F"

    monkeypatch.setattr(asset_catalog, "lookup_ticker_online", primero_falla)
    asset_catalog.resolve_tickers(["IE0032126645"], path)
    mapping = asset_catalog.resolve_tickers(["IE0032126645"], path)

    assert mapping["IE0032126645"] == "0P00000SUJ.F"
    assert len(intentos) == 2
