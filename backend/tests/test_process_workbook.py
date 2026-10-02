from io import BytesIO

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import backend.main as backend_main
from backend.infrastructure.finance_checkpoint_repository import JsonFinanceCheckpointRepository
from backend.infrastructure.finance_history_repository import CsvFinanceHistoryRepository
from backend.infrastructure.objectives_repository import JsonObjectivesRepository
from backend.infrastructure.transaction_history_repository import CsvTransactionHistoryRepository
from backend.main import app


@pytest.fixture(autouse=True)
def _isolated_finance_repositories(tmp_path, monkeypatch):
    """Cada test de este archivo llama a /api/v1/process directamente sobre las instancias por
    defecto de backend.main. Sin aislar los repositorios aquí, un test acabaría leyendo/escribiendo
    el historial.csv y checkpoint reales del repo (que además ya tienen datos del usuario committeados),
    contaminando los resultados con meses que el propio test nunca subió."""
    monkeypatch.setattr(backend_main, "finance_history_repository", CsvFinanceHistoryRepository(tmp_path / "historial.csv"))
    monkeypatch.setattr(backend_main, "finance_checkpoint_repository", JsonFinanceCheckpointRepository(tmp_path / "checkpoint.json"))
    monkeypatch.setattr(
        backend_main,
        "transaction_history_repository",
        CsvTransactionHistoryRepository(gastos_path=tmp_path / "gastos.csv", ingresos_path=tmp_path / "ingresos.csv"),
    )


def _sample_workbook() -> BytesIO:
    workbook = BytesIO()

    gastos = pd.DataFrame(
        [
            {
                "Fecha": "2024-10-10",
                "Categoria": "Alimentacion",
                "Cuenta": "Principal",
                "Cantidad": 100.0,
                "Etiquetas": "",
                "Comentario": "Compra semanal",
            }
        ]
    )
    ingresos = pd.DataFrame(
        [
            {
                "Fecha": "2024-10-01",
                "Categoria": "Salario",
                "Cuenta": "Principal",
                "Cantidad": 2000.0,
                "Etiquetas": "",
                "Comentario": "Nomina",
            }
        ]
    )
    transferencias = pd.DataFrame(columns=["Fecha", "Saliente", "Entrante", "Cantidad", "Comentario"])

    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        gastos.to_excel(writer, sheet_name="Gastos", index=False)
        ingresos.to_excel(writer, sheet_name="Ingresos", index=False)
        transferencias.to_excel(writer, sheet_name="Transferencias", index=False)

    workbook.seek(0)
    return workbook


def _sample_workbook_with_income_history() -> BytesIO:
    workbook = BytesIO()
    months = pd.period_range("2023-12", "2024-10", freq="M")

    gastos = pd.DataFrame(
        [
            {
                "Fecha": "2024-10-10",
                "Categoria": "Alimentacion",
                "Cuenta": "Principal",
                "Cantidad": 100.0,
                "Etiquetas": "",
                "Comentario": "Compra semanal",
            }
        ]
    )
    ingresos = pd.DataFrame(
        [
            {
                "Fecha": month.to_timestamp().strftime("%Y-%m-%d"),
                "Categoria": "Salario",
                "Cuenta": "Principal",
                "Cantidad": 2000.0,
                "Etiquetas": "",
                "Comentario": "Nomina",
            }
            for month in months
        ]
    )
    transferencias = pd.DataFrame(columns=["Fecha", "Saliente", "Entrante", "Cantidad", "Comentario"])

    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        gastos.to_excel(writer, sheet_name="Gastos", index=False)
        ingresos.to_excel(writer, sheet_name="Ingresos", index=False)
        transferencias.to_excel(writer, sheet_name="Transferencias", index=False)

    workbook.seek(0)
    return workbook


def _sample_workbook_with_dividends() -> BytesIO:
    workbook = BytesIO()

    gastos = pd.DataFrame(
        [
            {
                "Fecha": "2024-10-10",
                "Categoria": "Alimentacion",
                "Cuenta": "Principal",
                "Cantidad": 100.0,
                "Etiquetas": "",
                "Comentario": "Compra semanal",
            }
        ]
    )
    ingresos = pd.DataFrame(
        [
            {
                "Fecha": "2024-10-01",
                "Categoria": "Salario",
                "Cuenta": "Principal",
                "Cantidad": 2000.0,
                "Etiquetas": "",
                "Comentario": "Nomina",
            },
            {
                "Fecha": "2024-10-15",
                "Categoria": "Interés",
                "Cuenta": "Principal",
                "Cantidad": 4.5,
                "Etiquetas": "Dividendos",
                "Comentario": "Primer dividendo",
            },
            {
                "Fecha": "2024-11-15",
                "Categoria": "Interés",
                "Cuenta": "Principal",
                "Cantidad": 7.25,
                "Etiquetas": "Dividendos, acciones",
                "Comentario": "Segundo dividendo",
            },
        ]
    )
    transferencias = pd.DataFrame(columns=["Fecha", "Saliente", "Entrante", "Cantidad", "Comentario"])

    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        gastos.to_excel(writer, sheet_name="Gastos", index=False)
        ingresos.to_excel(writer, sheet_name="Ingresos", index=False)
        transferencias.to_excel(writer, sheet_name="Transferencias", index=False)

    workbook.seek(0)
    return workbook


def _sample_workbook_with_fees() -> BytesIO:
    workbook = BytesIO()

    gastos = pd.DataFrame(
        [
            {
                "Fecha": "2024-10-10",
                "Categoria": "Alimentacion",
                "Cuenta": "Principal",
                "Cantidad": 100.0,
                "Etiquetas": "",
                "Comentario": "Compra semanal",
            },
            {
                "Fecha": "2024-10-20",
                "Categoria": "Otros",
                "Cuenta": "Principal",
                "Cantidad": 3.5,
                "Etiquetas": "Comision",
                "Comentario": "Comisión de compra",
            },
            {
                "Fecha": "2024-11-05",
                "Categoria": "Otros",
                "Cuenta": "Principal",
                "Cantidad": 2.25,
                "Etiquetas": "Comision, broker",
                "Comentario": "Comisión de venta",
            },
        ]
    )
    ingresos = pd.DataFrame(
        [
            {
                "Fecha": "2024-10-01",
                "Categoria": "Salario",
                "Cuenta": "Principal",
                "Cantidad": 2000.0,
                "Etiquetas": "",
                "Comentario": "Nomina",
            }
        ]
    )
    transferencias = pd.DataFrame(columns=["Fecha", "Saliente", "Entrante", "Cantidad", "Comentario"])

    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        gastos.to_excel(writer, sheet_name="Gastos", index=False)
        ingresos.to_excel(writer, sheet_name="Ingresos", index=False)
        transferencias.to_excel(writer, sheet_name="Transferencias", index=False)

    workbook.seek(0)
    return workbook


def _sample_workbook_with_dividend_companies() -> BytesIO:
    workbook = BytesIO()

    gastos = pd.DataFrame(
        [
            {
                "Fecha": "2024-10-10",
                "Categoria": "Alimentacion",
                "Cuenta": "Principal",
                "Cantidad": 100.0,
                "Etiquetas": "",
                "Comentario": "Compra semanal",
            }
        ]
    )
    ingresos = pd.DataFrame(
        [
            {
                "Fecha": "2024-10-01",
                "Categoria": "Salario",
                "Cuenta": "Principal",
                "Cantidad": 2000.0,
                "Etiquetas": "",
                "Comentario": "Nomina",
            },
            {
                "Fecha": "2024-10-15",
                "Categoria": "Interés",
                "Cuenta": "Trade Republic",
                "Cantidad": 0.28,
                "Etiquetas": "Dividendos, PyG",
                "Comentario": "Procter and Gamber",
            },
            {
                "Fecha": "2024-11-05",
                "Categoria": "Interés",
                "Cuenta": "Degiro",
                "Cantidad": 3.98,
                "Etiquetas": "Dividendos, PyG",
                "Comentario": "Procter and Gamber",
            },
            {
                "Fecha": "2024-11-15",
                "Categoria": "Interés",
                "Cuenta": "Trade Republic",
                "Cantidad": 7.57,
                "Etiquetas": "Dividendos, IB",
                "Comentario": "Iberdrola",
            },
            {
                "Fecha": "2024-11-20",
                "Categoria": "Interés",
                "Cuenta": "Principal",
                "Cantidad": 2.05,
                "Etiquetas": "Interes",
                "Comentario": "Saveback",
            },
        ]
    )
    transferencias = pd.DataFrame(columns=["Fecha", "Saliente", "Entrante", "Cantidad", "Comentario"])

    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        gastos.to_excel(writer, sheet_name="Gastos", index=False)
        ingresos.to_excel(writer, sheet_name="Ingresos", index=False)
        transferencias.to_excel(writer, sheet_name="Transferencias", index=False)

    workbook.seek(0)
    return workbook


def test_process_workbook_returns_serializable_summary():
    client = TestClient(app)
    workbook = _sample_workbook()

    response = client.post(
        "/api/v1/process",
        files={
            "file": (
                "Inicio.xlsx",
                workbook.getvalue(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["result"]["movimientos"]["gastos"] == 1
    assert data["result"]["movimientos"]["ingresos"] == 1
    assert data["result"]["params"]["fecha_inicio"] == "2024-10-01"
    assert data["result"]["historial"]["meses"] == 1
    assert data["result"]["historial"]["ultimo_mes"]["Mes"] == "2024-10"
    assert data["result"]["analisis"]["gastos"]["totales_categoria"][0] == {
        "categoria": "alimentacion",
        "total": 100.0,
    }
    assert data["result"]["analisis"]["gastos"]["mensual"][0]["balance"] == 1900.0
    assert data["result"]["analisis"]["ahorro"]["ultimo_mes"]["porcentaje_ahorro"] == 95.0


def test_process_workbook_adds_accumulated_dividends_to_history():
    client = TestClient(app)
    workbook = _sample_workbook_with_dividends()

    response = client.post(
        "/api/v1/process",
        files={
            "file": (
                "Inicio.xlsx",
                workbook.getvalue(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )

    assert response.status_code == 200
    rows = response.json()["result"]["historial"]["resumen"]
    dividends_by_month = {row["Mes"]: row["Dividendos"] for row in rows}
    assert dividends_by_month["2024-10"] == 4.5
    assert dividends_by_month["2024-11"] == 11.75


def test_process_workbook_adds_accumulated_fees_to_history():
    client = TestClient(app)
    workbook = _sample_workbook_with_fees()

    response = client.post(
        "/api/v1/process",
        files={
            "file": (
                "Inicio.xlsx",
                workbook.getvalue(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )

    assert response.status_code == 200
    rows = response.json()["result"]["historial"]["resumen"]
    fees_by_month = {row["Mes"]: row["Comisiones"] for row in rows}
    assert fees_by_month["2024-10"] == 3.5
    assert fees_by_month["2024-11"] == 5.75


def test_process_workbook_returns_dividend_payments():
    client = TestClient(app)
    workbook = _sample_workbook_with_dividend_companies()

    response = client.post(
        "/api/v1/process",
        files={
            "file": (
                "Inicio.xlsx",
                workbook.getvalue(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )

    assert response.status_code == 200
    payments = response.json()["result"]["analisis"]["dividendos_pagos"]
    assert len(payments) == 3
    assert payments[0] == {"fecha": "2024-10-15", "codigo": "PyG", "comentario": "Procter and Gamber", "cantidad": 0.28}
    assert payments[1] == {"fecha": "2024-11-05", "codigo": "PyG", "comentario": "Procter and Gamber", "cantidad": 3.98}
    assert payments[2] == {"fecha": "2024-11-15", "codigo": "IB", "comentario": "Iberdrola", "cantidad": 7.57}
    assert not any(payment["comentario"] == "Saveback" for payment in payments)


def test_process_workbook_rejects_invalid_objectives_json():
    client = TestClient(app)
    workbook = _sample_workbook()

    response = client.post(
        "/api/v1/process",
        data={"objetivos_json": "{nope"},
        files={
            "file": (
                "Inicio.xlsx",
                workbook.getvalue(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "objetivos_json debe ser JSON válido"


def test_process_workbook_rejects_invalid_pipeline_config():
    client = TestClient(app)
    workbook = _sample_workbook()

    response = client.post(
        "/api/v1/process?porcentaje_gasto=1.4",
        files={
            "file": (
                "Inicio.xlsx",
                workbook.getvalue(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "porcentaje_gasto debe estar entre 0 y 1"


def test_process_workbook_accepts_objectives_payload():
    client = TestClient(app)
    workbook = _sample_workbook_with_income_history()

    response = client.post(
        "/api/v1/process",
        data={
            "objetivos_json": (
                '[{"nombre":"Coche","etiquetas":["coche"],'
                '"fraccion_presupuesto":0.1,"duracion_meses":1,"mes_inicio":"2024-10"}]'
            )
        },
        files={
            "file": (
                "Inicio.xlsx",
                workbook.getvalue(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )

    assert response.status_code == 200
    data = response.json()
    objetivos = data["result"]["historial"]["objetivos"]
    assert len(objetivos) == 1
    assert objetivos[0]["Objetivo"] == "Coche"


def test_objectives_can_be_saved_and_loaded(tmp_path, monkeypatch):
    objectives_path = tmp_path / "objetivos_vista.json"
    monkeypatch.setattr(backend_main, "objectives_repository", JsonObjectivesRepository(objectives_path))
    client = TestClient(app)

    payload = {
        "objetivos": [
            {
                "nombre": "Coche",
                "etiquetas": "coche, taller",
                "fraccion_presupuesto": 0.2,
                "duracion_meses": 6,
                "mes_inicio": "2024-10",
            }
        ]
    }

    save_response = client.put("/api/v1/objectives", json=payload)
    assert save_response.status_code == 200
    assert save_response.json()["objetivos"][0]["etiquetas"] == ["coche", "taller"]

    load_response = client.get("/api/v1/objectives")
    assert load_response.status_code == 200
    assert load_response.json()["objetivos"][0]["nombre"] == "Coche"


def test_objectives_reject_duplicate_names(tmp_path, monkeypatch):
    monkeypatch.setattr(
        backend_main,
        "objectives_repository",
        JsonObjectivesRepository(tmp_path / "objetivos_vista.json"),
    )
    client = TestClient(app)

    response = client.put(
        "/api/v1/objectives",
        json={
            "objetivos": [
                {
                    "nombre": "Coche",
                    "etiquetas": ["coche"],
                    "fraccion_presupuesto": 0.1,
                    "duracion_meses": 6,
                    "mes_inicio": "2024-10",
                },
                {
                    "nombre": "Coche",
                    "etiquetas": ["vehiculo"],
                    "fraccion_presupuesto": 0.1,
                    "duracion_meses": 6,
                    "mes_inicio": "2024-11",
                },
            ]
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Los objetivos deben tener nombres únicos"


def _workbook_with_refund_month(gasto: float, reembolso: float, ingreso_mes2: float) -> BytesIO:
    """Dos meses: el primero se pasa de presupuesto y deja deuda; el segundo recibe un reembolso.

    El segundo mes es el caso que importa: si el reembolso supera al gasto, el gasto neto queda en
    negativo y la fórmula antigua lo trataba como presupuesto liberado.
    """
    workbook = BytesIO()
    gastos = pd.DataFrame(
        [
            # Mes 1: gasto muy por encima del presupuesto, para arrastrar deuda al mes 2.
            {"Fecha": "2024-10-10", "Categoria": "Otros", "Cuenta": "Principal", "Cantidad": 2500.0, "Etiquetas": "", "Comentario": ""},
            {"Fecha": "2024-11-10", "Categoria": "Otros", "Cuenta": "Principal", "Cantidad": gasto, "Etiquetas": "", "Comentario": ""},
        ]
    )
    ingresos = pd.DataFrame(
        [
            {"Fecha": "2024-10-01", "Categoria": "Salario", "Cuenta": "Principal", "Cantidad": 2000.0, "Etiquetas": "", "Comentario": "Nomina"},
            {"Fecha": "2024-11-01", "Categoria": "Salario", "Cuenta": "Principal", "Cantidad": ingreso_mes2, "Etiquetas": "", "Comentario": "Nomina"},
            # Un ingreso en categoría de gasto es un reembolso: se resta del gasto del mes.
            {"Fecha": "2024-11-20", "Categoria": "Otros", "Cuenta": "Principal", "Cantidad": reembolso, "Etiquetas": "", "Comentario": "Devolucion"},
        ]
    )
    transferencias = pd.DataFrame(columns=["Fecha", "Saliente", "Entrante", "Cantidad", "Comentario"])

    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        gastos.to_excel(writer, sheet_name="Gastos", index=False)
        ingresos.to_excel(writer, sheet_name="Ingresos", index=False)
        transferencias.to_excel(writer, sheet_name="Transferencias", index=False)

    workbook.seek(0)
    return workbook


def _process(workbook: BytesIO) -> list[dict]:
    client = TestClient(app)
    response = client.post(
        "/api/v1/process",
        files={"file": ("registro.xlsx", workbook, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        data={"modo": "visualizar"},
    )
    assert response.status_code == 200, response.text
    return response.json()["result"]["historial"]["resumen"]


def test_un_mes_con_mas_reembolsos_que_gasto_no_libera_presupuesto_si_queda_deuda():
    """El caso real de junio de 2026: el gasto neto era -762,83 € y aparecían 762,83 € de sobrante
    mientras seguía habiendo deuda acumulada."""
    filas = _process(_workbook_with_refund_month(gasto=100.0, reembolso=900.0, ingreso_mes2=1000.0))
    mes = filas[1]

    assert mes["💳 Gasto del mes"] < 0, "el mes debe cerrar con gasto neto negativo"
    assert mes["📉 Deuda Presupuestaria acumulada"] > 0, "la deuda no se salda del todo"
    assert mes["🧾 Presupuesto Disponible"] == 0.0


def test_el_reembolso_cuenta_para_saldar_la_deuda_y_solo_sobra_lo_que_la_supera():
    """Regla acordada: presupuesto + reembolso pagan la deuda, y sobra únicamente el exceso."""
    filas = _process(_workbook_with_refund_month(gasto=0.0, reembolso=4000.0, ingreso_mes2=1000.0))
    mes = filas[1]

    assert mes["📉 Deuda Presupuestaria acumulada"] == 0.0
    assert mes["🧾 Presupuesto Disponible"] > 0.0


def test_sobrante_y_deuda_acumulada_nunca_conviven_en_el_mismo_mes():
    """Son la parte negativa y la positiva de la misma resta, así que es imposible por construcción."""
    for reembolso in (0.0, 500.0, 900.0, 4000.0):
        for filas in (_process(_workbook_with_refund_month(gasto=200.0, reembolso=reembolso, ingreso_mes2=1000.0)),):
            for mes in filas:
                sobrante = float(mes["🧾 Presupuesto Disponible"] or 0.0)
                deuda = float(mes["📉 Deuda Presupuestaria acumulada"] or 0.0)
                assert not (sobrante > 0 and deuda > 0), f"{mes['Mes']}: sobrante {sobrante} con deuda {deuda}"


def test_un_mes_con_gasto_positivo_y_deuda_sin_cubrir_sigue_sin_sobrante():
    """Con gasto positivo la fórmula nueva y la antigua coinciden: si el presupuesto del mes no da
    para saldar la deuda arrastrada, no sobra nada. Es el comportamiento de siempre."""
    filas = _process(_workbook_with_refund_month(gasto=100.0, reembolso=0.0, ingreso_mes2=20000.0))
    mes = filas[1]

    assert mes["💳 Gasto del mes"] > 0
    assert mes["📉 Deuda Presupuestaria acumulada"] > 0
    assert mes["🧾 Presupuesto Disponible"] == 0.0
