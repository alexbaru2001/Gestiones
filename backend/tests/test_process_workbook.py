from io import BytesIO

import pandas as pd
from fastapi.testclient import TestClient

import backend.main as backend_main
from backend.infrastructure.objectives_repository import JsonObjectivesRepository
from backend.main import app


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
