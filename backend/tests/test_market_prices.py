import pytest

from backend.domain import market_prices as mp


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(mp, "get_investments_path", lambda: tmp_path)
    return tmp_path


def test_la_cache_se_guarda_y_se_relee():
    mp.save_cache({"IE0032126645:2026-09": 83.1477})

    assert mp.load_cache()["IE0032126645:2026-09"] == 83.1477


def test_lo_que_ya_esta_en_cache_no_se_vuelve_a_pedir(monkeypatch):
    """El cierre de un mes pasado no cambia: consultarlo una vez basta."""
    mp.save_cache({"IE0032126645:2026-09": 83.1477})
    monkeypatch.setattr(mp, "_resolve_tickers", lambda *_: pytest.fail("no debía resolverse ningún ticker"))

    assert mp.get_prices([("IE0032126645", "2026-09")])["IE0032126645:2026-09"] == 83.1477


def test_solo_se_consulta_lo_que_falta(monkeypatch):
    mp.save_cache({"A:2026-01": 10.0})
    pedidos = []
    monkeypatch.setattr(mp, "_resolve_tickers", lambda isins: pedidos.append(isins) or {})

    mp.get_prices([("A", "2026-01"), ("B", "2026-01")])

    assert pedidos == [["B"]], "el ISIN ya cacheado no debe resolverse otra vez"


def test_un_isin_que_no_resuelve_no_rompe_el_resto(monkeypatch):
    monkeypatch.setattr(mp, "_resolve_tickers", lambda isins: {"B": "BBB"})
    monkeypatch.setattr(mp, "_fetch", lambda isin, ticker, months: {mp.key(isin, months[0]): 25.0})

    prices = mp.get_prices([("A", "2026-01"), ("B", "2026-01")])

    assert prices == {"B:2026-01": 25.0}


def test_sin_red_se_devuelve_lo_que_haya_sin_fallar():
    mp.save_cache({"A:2026-01": 10.0})

    assert mp.get_prices([("A", "2026-01"), ("A", "2026-02")], allow_network=False) == {"A:2026-01": 10.0}


def test_una_cache_corrupta_no_tumba_la_valoracion(isolated_cache):
    (isolated_cache / mp.CACHE_NAME).write_text("{roto", encoding="utf-8")

    assert mp.load_cache() == {}


def test_las_parejas_sin_isin_o_sin_mes_se_ignoran(monkeypatch):
    monkeypatch.setattr(mp, "_resolve_tickers", lambda isins: {})

    assert mp.get_prices([("", "2026-01"), ("A", "")]) == {}
