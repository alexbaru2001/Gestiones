"""El fondo de reserva de punta a punta: desde el Excel hasta lo que ve el frontend."""

from io import BytesIO

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import backend.main as backend_main
from backend.infrastructure.finance_checkpoint_repository import JsonFinanceCheckpointRepository
from backend.infrastructure.finance_history_repository import CsvFinanceHistoryRepository
from backend.infrastructure.transaction_history_repository import CsvTransactionHistoryRepository
from backend.main import app


@pytest.fixture(autouse=True)
def _isolated_finance_repositories(tmp_path, monkeypatch):
    monkeypatch.setattr(backend_main, "finance_history_repository", CsvFinanceHistoryRepository(tmp_path / "historial.csv"))
    monkeypatch.setattr(backend_main, "finance_checkpoint_repository", JsonFinanceCheckpointRepository(tmp_path / "checkpoint.json"))
    monkeypatch.setattr(
        backend_main,
        "transaction_history_repository",
        CsvTransactionHistoryRepository(gastos_path=tmp_path / "gastos.csv", ingresos_path=tmp_path / "ingresos.csv"),
    )


def _workbook(meses: int = 12, gasto: float = 300.0, sueldo: float = 2000.0) -> BytesIO:
    """Un Excel con el mismo sueldo y el mismo gasto todos los meses desde 2024-10."""
    workbook = BytesIO()
    periodos = pd.period_range("2024-10", periods=meses, freq="M")

    gastos = pd.DataFrame(
        [
            {
                "Fecha": periodo.to_timestamp().strftime("%Y-%m-10"),
                "Categoria": "Alimentacion",
                "Cuenta": "Principal",
                "Cantidad": gasto,
                "Etiquetas": "",
                "Comentario": "Compra",
            }
            for periodo in periodos
        ]
    )
    ingresos = pd.DataFrame(
        [
            {
                "Fecha": periodo.to_timestamp().strftime("%Y-%m-01"),
                "Categoria": "Salario",
                "Cuenta": "Principal",
                "Cantidad": sueldo,
                "Etiquetas": "",
                "Comentario": "Nomina",
            }
            for periodo in periodos
        ]
    )
    transferencias = pd.DataFrame(columns=["Fecha", "Saliente", "Entrante", "Cantidad", "Comentario"])

    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        gastos.to_excel(writer, sheet_name="Gastos", index=False)
        ingresos.to_excel(writer, sheet_name="Ingresos", index=False)
        transferencias.to_excel(writer, sheet_name="Transferencias", index=False)

    workbook.seek(0)
    return workbook


def _procesar(client: TestClient, workbook: BytesIO, modo: str = "visualizar") -> dict:
    response = client.post(
        "/api/v1/process",
        data={"modo": modo},
        files={
            "file": (
                "Gestiones.xlsx",
                workbook.getvalue(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_el_fondo_tiene_objetivo_desde_el_primer_calculo():
    """Antes el objetivo era siempre 0: el fondo acumulaba sin meta y sin porcentaje."""
    data = _procesar(TestClient(app), _workbook())

    fondo = data["result"]["analisis"]["fondo_reserva"]
    filas = data["result"]["historial"]["resumen"]
    gasto_medio = sum(fila["💳 Gasto del mes"] for fila in filas) / len(filas)

    assert fondo["objetivo"] == pytest.approx(gasto_medio * 6, abs=0.02)
    assert fondo["objetivo"] > 0
    assert fondo["meses_de_cobertura"] == 6


def test_el_porcentaje_cuadra_con_lo_cargado_y_el_objetivo():
    fondo = _procesar(TestClient(app), _workbook())["result"]["analisis"]["fondo_reserva"]

    assert fondo["porcentaje"] == pytest.approx(fondo["cargado"] / fondo["objetivo"], abs=1e-4)
    assert fondo["pendiente"] == pytest.approx(fondo["objetivo"] - fondo["cargado"], abs=0.02)


def test_se_aparta_el_veinte_por_ciento_del_ahorro_del_mes():
    """El primer mes fija el punto de partida; el segundo ya aporta."""
    filas = _procesar(TestClient(app), _workbook())["result"]["historial"]["resumen"]

    aporte = filas[1]["Fondo de reserva cargado"] - filas[0]["Fondo de reserva cargado"]
    ahorro_del_mes = (filas[1]["💰 Ahorros"] - filas[0]["💰 Ahorros"]) + aporte

    assert aporte == pytest.approx(ahorro_del_mes * 0.20, abs=0.02)


def test_el_fondo_nunca_pasa_del_objetivo():
    """Con 60 meses de sueldo alto y gasto bajo el fondo se llena de sobra: debe quedarse clavado."""
    data = _procesar(TestClient(app), _workbook(meses=24, gasto=200.0, sueldo=3000.0))
    filas = data["result"]["historial"]["resumen"]
    fondo = data["result"]["analisis"]["fondo_reserva"]

    cargados = [fila["Fondo de reserva cargado"] for fila in filas]
    assert max(cargados) <= fondo["objetivo"] + 0.02
    assert cargados[-1] == pytest.approx(fondo["objetivo"], abs=0.02)
    assert fondo["porcentaje"] == pytest.approx(1.0, abs=1e-3)
    # Y una vez lleno deja de crecer: los últimos meses son todos la misma cifra.
    assert cargados[-1] == pytest.approx(cargados[-2], abs=0.02)


def test_un_fondo_lleno_no_se_rellena_solo_cada_mes():
    """La rama vieja reasignaba el fondo al objetivo cada mes, así que lo gastado se reponía gratis.

    Aquí se comprueba el efecto observable: una vez lleno, el ahorro del mes se queda entero en
    Ahorros en vez de seguir alimentando al fondo.
    """
    filas = _procesar(TestClient(app), _workbook(meses=24, gasto=200.0, sueldo=3000.0))["result"]["historial"]["resumen"]

    aporte_ultimo_mes = filas[-1]["Fondo de reserva cargado"] - filas[-2]["Fondo de reserva cargado"]
    ahorro_ultimo_mes = filas[-1]["💰 Ahorros"] - filas[-2]["💰 Ahorros"]

    assert aporte_ultimo_mes == pytest.approx(0.0, abs=0.02)
    assert ahorro_ultimo_mes > 0


def test_el_objetivo_se_guarda_en_el_checkpoint_para_el_siguiente_tramo():
    client = TestClient(app)
    data = _procesar(client, _workbook(), modo="historico")

    checkpoint = client.get("/api/v1/process/checkpoint").json()["result"]
    snapshot = checkpoint["fondo_reserva_snapshot"]

    assert snapshot["Cantidad del fondo"] > 0
    assert snapshot["Cantidad cargada"] == pytest.approx(
        data["result"]["historial"]["resumen"][-1]["Fondo de reserva cargado"], abs=0.02
    )
    assert snapshot["Porcentaje"] == pytest.approx(
        snapshot["Cantidad cargada"] / snapshot["Cantidad del fondo"], abs=1e-4
    )


def test_el_fondo_sale_de_ahorros_y_no_es_dinero_nuevo():
    """El fondo es una etiqueta sobre dinero que ya estaba ahorrado, no dinero nuevo.

    Mes a mes: lo que entra en el fondo es el 20 % del ahorro del mes (o lo que falte por cubrir,
    si es menos), y el resto se queda en Ahorros. Las dos cifras tienen que sumar exactamente el
    ahorro del mes; si el fondo fuese dinero nuevo, sumarían de más.
    """
    data = _procesar(TestClient(app), _workbook())
    filas = data["result"]["historial"]["resumen"]
    objetivo = data["result"]["analisis"]["fondo_reserva"]["objetivo"]

    comprobados = 0
    for anterior, actual in zip(filas, filas[1:]):
        aporte = actual["Fondo de reserva cargado"] - anterior["Fondo de reserva cargado"]
        subida_de_ahorros = actual["💰 Ahorros"] - anterior["💰 Ahorros"]
        ahorro_del_mes = subida_de_ahorros + aporte
        if ahorro_del_mes <= 0:
            continue
        pendiente = max(0.0, objetivo - anterior["Fondo de reserva cargado"])
        assert aporte == pytest.approx(min(ahorro_del_mes * 0.20, pendiente), abs=0.02)
        comprobados += 1

    assert comprobados >= 5


def test_un_checkpoint_viejo_recupera_la_media_de_gasto_del_historico():
    """Migración: los checkpoints escritos antes de este cambio no traen la media de gasto.

    Sin reconstruirla, el objetivo del siguiente tramo se calcularía solo con los meses del Excel
    recién subido y olvidaría todo lo vivido antes.
    """
    from backend.main import _seed_reserve_average

    viejo = {
        "as_of_month": "2025-12",
        "fondo_reserva_snapshot": {"Cantidad cargada": 1652.34, "Cantidad del fondo": 0.0, "Porcentaje": 0.0},
    }
    historico = [
        {"Mes": "2025-10", "💳 Gasto del mes": 600.0},
        {"Mes": "2025-11", "💳 Gasto del mes": 800.0},
        {"Mes": "2025-12", "💳 Gasto del mes": 700.0},
        # Un mes posterior al cierre del checkpoint no cuenta: todavía no se había vivido.
        {"Mes": "2026-01", "💳 Gasto del mes": 9000.0},
    ]

    migrado = _seed_reserve_average(viejo, historico)
    snapshot = migrado["fondo_reserva_snapshot"]

    assert snapshot["Meses contados"] == 3
    assert snapshot["Gasto acumulado"] == pytest.approx(2100.0)
    # Lo que ya traía se conserva.
    assert snapshot["Cantidad cargada"] == 1652.34


def test_un_checkpoint_que_ya_trae_la_media_no_se_toca():
    from backend.main import _seed_reserve_average

    actual = {
        "as_of_month": "2025-12",
        "fondo_reserva_snapshot": {"Cantidad cargada": 10.0, "Gasto acumulado": 50.0, "Meses contados": 2},
    }
    assert _seed_reserve_average(actual, [{"Mes": "2025-01", "💳 Gasto del mes": 999.0}]) is actual


def test_sin_checkpoint_no_hay_nada_que_migrar():
    from backend.main import _seed_reserve_average

    assert _seed_reserve_average(None, [{"Mes": "2025-01", "💳 Gasto del mes": 600.0}]) is None
