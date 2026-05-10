from io import BytesIO

import openpyxl
from fastapi.testclient import TestClient

from backend import main
from backend.domain.portfolio import UploadedInvestmentFile, build_snapshot_from_files


def make_degiro_portfolio_workbook() -> bytes:
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Resumen de la Cartera"
    sheet.append(("Producto", "Symbol/ISIN", "Cantidad", "Precio de ", "Valor local", None, "Valor en EUR"))
    sheet.append(("CASH & CASH FUND & FTX CASH (EUR)", None, None, None, "EUR", 48.87, 48.87))
    sheet.append(("LABORATORIOS FARMACEUTICOS ROVI SA", "ES0157261019", 3, 62.85, "EUR", 188.55, 188.55))
    output = BytesIO()
    workbook.save(output)
    return output.getvalue()


def make_degiro_account_workbook() -> bytes:
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Estado de cuenta"
    sheet.append(("Fecha", "Hora", "Fecha valor", "Producto", "ISIN", "Descripción", "Tipo", "Variación", None, "Saldo", None, "ID Orden"))
    sheet.append(
        (
            "15-12-2025",
            "09:00",
            "15-12-2025",
            "LABORATORIOS FARMACEUTICOS ROVI SA",
            "ES0157261019",
            "Compra 3 Laboratorios Farmaceuticos ROVI SA@62,25 EUR (ES0157261019)",
            None,
            "EUR",
            -186.75,
            "EUR",
            13.82,
            "orden",
        )
    )
    output = BytesIO()
    workbook.save(output)
    return output.getvalue()


def test_build_snapshot_combines_positions_and_costs():
    snapshot = build_snapshot_from_files(
        [
            UploadedInvestmentFile("Portfolio.xlsx", make_degiro_portfolio_workbook()),
            UploadedInvestmentFile("Account.xlsx", make_degiro_account_workbook()),
        ]
    )

    rovi = next(position for position in snapshot["positions"] if position["isin"] == "ES0157261019")
    assert rovi["current_value"] == 188.55
    assert rovi["cost"] == 186.75
    assert rovi["unrealized_gain"] == 1.8
    assert snapshot["summary"]["cash"] == 48.87


def test_import_portfolio_endpoint_persists_local_snapshot(tmp_path, monkeypatch):
    repository = main.LocalPortfolioRepository(tmp_path)
    monkeypatch.setattr(main, "portfolio_repository", repository)
    client = TestClient(main.app)

    response = client.post(
        "/api/v1/portfolio/import",
        files=[
            (
                "files",
                ("Portfolio.xlsx", make_degiro_portfolio_workbook(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
            )
        ],
    )

    assert response.status_code == 200
    assert response.json()["result"]["summary"]["positions"] == 2
    assert (tmp_path / "processed" / "portfolio_snapshot.json").exists()
