import json

import pytest

from backend.domain import fx


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(fx, "get_investments_path", lambda: tmp_path)
    return tmp_path


def test_un_importe_en_euros_no_necesita_conversion():
    assert fx.convert(100.0, "EUR", "2026-01", {}) == 100.0


def test_un_importe_en_divisa_se_pasa_a_euros_con_el_cambio_del_mes():
    # 1 dólar = 0,85 €, así que 714,43 $ son 607,27 €.
    rates = {"USD:2025-12": 0.85}

    assert fx.convert(714.43, "USD", "2025-12", rates) == 607.27


def test_sin_cambio_conocido_no_se_inventa_un_importe():
    assert fx.convert(100.0, "USD", "2026-01", {}) is None


def test_cada_mes_usa_su_propio_cambio():
    rates = {"USD:2026-01": 0.80, "USD:2026-02": 0.90}

    assert fx.convert(100.0, "USD", "2026-01", rates) == 80.0
    assert fx.convert(100.0, "USD", "2026-02", rates) == 90.0


def test_un_importe_que_no_es_numero_no_rompe_la_conversion():
    assert fx.convert(None, "USD", "2026-01", {"USD:2026-01": 0.85}) is None


def test_la_cache_se_guarda_y_se_relee(isolated_cache):
    fx.save_cache({"USD:2026-01": 0.85})

    assert fx.load_cache() == {"USD:2026-01": 0.85}
    assert (isolated_cache / fx.CACHE_NAME).exists()


def test_una_cache_corrupta_no_tumba_el_calculo(isolated_cache):
    (isolated_cache / fx.CACHE_NAME).write_text("{roto", encoding="utf-8")

    assert fx.load_cache() == {}


def test_lo_que_ya_esta_en_cache_no_se_vuelve_a_pedir(monkeypatch):
    """El cambio de un mes cerrado no cambia: pedirlo una vez basta."""
    fx.save_cache({"USD:2026-01": 0.85})
    monkeypatch.setattr(fx, "_fetch_currency", lambda *_: pytest.fail("no debía consultarse la red"))

    assert fx.get_rates({("USD", "2026-01")})["USD:2026-01"] == 0.85


def test_sin_red_se_devuelve_lo_que_haya_en_cache_sin_fallar():
    fx.save_cache({"USD:2026-01": 0.85})

    rates = fx.get_rates({("USD", "2026-01"), ("USD", "2026-02")}, allow_network=False)

    assert rates == {"USD:2026-01": 0.85}
