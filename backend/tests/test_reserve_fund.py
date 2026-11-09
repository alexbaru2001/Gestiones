import math

from backend.domain.reserve_fund import ReserveFundStatus


def test_porcentaje_y_pendiente():
    status = ReserveFundStatus(objetivo=4000.0, cargado=1000.0)
    assert status.porcentaje == 0.25
    assert status.pendiente == 3000.0


def test_porcentaje_sin_objetivo_es_cero_y_no_revienta():
    status = ReserveFundStatus(objetivo=0.0, cargado=1000.0)
    assert status.porcentaje == 0.0
    assert status.pendiente == 0.0


def test_un_fondo_por_encima_del_objetivo_se_enseña_al_cien_no_al_ciento_veinte():
    # Pasa de verdad: si el gasto medio baja, el objetivo baja y el fondo se queda por encima.
    status = ReserveFundStatus(objetivo=1000.0, cargado=1180.0)
    assert status.porcentaje == 1.0
    assert status.pendiente == 0.0


def test_fondo_vacio():
    status = ReserveFundStatus(objetivo=3000.0, cargado=0.0)
    assert status.porcentaje == 0.0
    assert status.pendiente == 3000.0


def test_to_dict_redondea_y_arrastra_la_regla_usada():
    data = ReserveFundStatus(
        objetivo=3718.0333, cargado=1198.6940, meses_de_cobertura=6, cuota_mensual=0.20
    ).to_dict()
    assert data["objetivo"] == 3718.03
    assert data["cargado"] == 1198.69
    assert math.isclose(data["porcentaje"], 0.3224, abs_tol=1e-4)
    assert data["pendiente"] == 2519.34
    assert data["meses_de_cobertura"] == 6
    assert data["cuota_mensual"] == 0.20
