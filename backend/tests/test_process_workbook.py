from io import BytesIO

import pandas as pd
from fastapi.testclient import TestClient

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
