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
    assert data["result"]["historial"]["meses"] == 1
    assert data["result"]["historial"]["ultimo_mes"]["Mes"] == "2024-10"


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
