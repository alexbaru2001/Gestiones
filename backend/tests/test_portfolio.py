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
    assert next(document for document in snapshot["documents"]["expected"] if document["kind"] == "degiro_portfolio")["uploaded"] is True
    assert "trade_republic_net_worth" in {document["kind"] for document in snapshot["documents"]["missing"]}
    assert "degiro_account" not in {document["kind"] for document in snapshot["documents"]["missing"]}


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
            ),
            (
                "files",
                ("Portfolio2.xlsx", make_degiro_portfolio_workbook(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
            ),
            (
                "files",
                ("Portfolio3.xlsx", make_degiro_portfolio_workbook(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
            ),
        ],
    )

    assert response.status_code == 200
    assert response.json()["result"]["summary"]["positions"] == 2
    assert (tmp_path / "processed" / "portfolio_snapshot.json").exists()
    assert (tmp_path / "snapshots" / "sin-fecha.json").exists()


def test_portfolio_repository_saves_snapshots_by_reference_date(tmp_path):
    repository = main.LocalPortfolioRepository(tmp_path)
    snapshot = {
        "snapshot_key": "2026-07-04",
        "snapshot_date": "2026-07-04",
        "snapshot_month": "2026-07",
        "summary": {},
    }

    repository.save_current_and_snapshot(snapshot)

    assert (tmp_path / "processed" / "portfolio_snapshot.json").exists()
    assert (tmp_path / "snapshots" / "2026-07-04.json").exists()


def test_portfolio_repository_updates_snapshot_date(tmp_path):
    repository = main.LocalPortfolioRepository(tmp_path)
    repository.save_current_and_snapshot(
        {
            "snapshot_key": "2026-07-30",
            "snapshot_date": "2026-07-30",
            "snapshot_month": "2026-07",
            "summary": {},
        }
    )

    snapshot = repository.update_snapshot_date("2026-07-30", "2026-06-30")

    assert snapshot["snapshot_key"] == "2026-06-30"
    assert snapshot["snapshot_date"] == "2026-06-30"
    assert snapshot["snapshot_month"] == "2026-06"
    assert not (tmp_path / "snapshots" / "2026-07-30.json").exists()
    assert (tmp_path / "snapshots" / "2026-06-30.json").exists()
    assert repository.load()["snapshot_date"] == "2026-06-30"


def test_portfolio_snapshot_date_endpoint_updates_snapshot(tmp_path, monkeypatch):
    repository = main.LocalPortfolioRepository(tmp_path)
    repository.save_current_and_snapshot(
        {
            "snapshot_key": "2026-07-30",
            "snapshot_date": "2026-07-30",
            "snapshot_month": "2026-07",
            "summary": {},
        }
    )
    monkeypatch.setattr(main, "portfolio_repository", repository)
    client = TestClient(main.app)

    response = client.put("/api/v1/portfolio/snapshots/2026-07-30/date", json={"snapshot_date": "2026-06-30"})

    assert response.status_code == 200
    assert response.json()["result"]["snapshot_date"] == "2026-06-30"
    assert response.json()["snapshots"][0]["snapshot_month"] == "2026-06"


def test_portfolio_snapshot_uses_finance_invested_for_matching_month(tmp_path):
    history_path = tmp_path / "historial.csv"
    history_path.write_text(
        "Mes,Inversiones,Dinero Invertido\n"
        "2025-12,-100.50,6689.94\n",
        encoding="utf-8",
    )
    repository = main.LocalPortfolioRepository(tmp_path / "portfolio", finance_history_path=history_path)

    snapshot = repository.enrich_with_finance_history(
        build_snapshot_from_files(
            [
                UploadedInvestmentFile("Portfolio.xlsx", make_degiro_portfolio_workbook()),
                UploadedInvestmentFile("Account.xlsx", make_degiro_account_workbook()),
            ]
        )
    )

    assert snapshot["snapshot_month"] == "2025-12"
    assert snapshot["summary"]["known_cost"] == 6689.94
    assert snapshot["summary"]["finance_invested"] == 6689.94
    assert snapshot["summary"]["investment_bucket"] == -100.5


def test_portfolio_snapshot_uses_latest_finance_month_for_newer_photo(tmp_path):
    history_path = tmp_path / "historial.csv"
    history_path.write_text(
        "Mes,Inversiones,Dinero Invertido,Dividendos\n"
        "2026-05,-1000.00,7000.00,4.00\n"
        "2026-06,-1839.55,7586.52,11.77\n",
        encoding="utf-8",
    )
    repository = main.LocalPortfolioRepository(tmp_path / "portfolio", finance_history_path=history_path)
    repository.save_current_and_snapshot(
        {
            "snapshot_key": "2026-05-11",
            "snapshot_date": "2026-05-11",
            "snapshot_month": "2026-05",
            "summary": {"dividends": 12.34, "fees": 1.5},
        }
    )

    snapshot = repository.enrich_with_finance_history(
        {
            "snapshot_date": "2026-07-04",
            "snapshot_month": "2026-07",
            "summary": {
                "invested": 8341.11,
                "known_cost": 0,
                "known_unrealized_gain": 0,
                "known_unrealized_gain_pct": None,
                "dividends": 0,
                "fees": 0,
            },
        }
    )

    assert snapshot["summary"]["finance_source_month"] == "2026-06"
    assert snapshot["summary"]["finance_invested"] == 7586.52
    assert snapshot["summary"]["known_cost"] == 7586.52
    assert snapshot["summary"]["known_unrealized_gain"] == 753.09
    assert snapshot["summary"]["known_unrealized_gain_pct"] == 9.93
    assert snapshot["summary"]["dividends"] == 11.77
    assert snapshot["summary"]["fees"] == 1.5
