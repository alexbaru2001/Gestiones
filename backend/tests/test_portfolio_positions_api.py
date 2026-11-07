from io import BytesIO

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend import main
from backend.infrastructure.portfolio_positions_repository import ExcelPortfolioPositionsRepository


@pytest.fixture(autouse=True)
def isolated_repository(tmp_path, monkeypatch):
    """La imagen de test lleva datos reales dentro; las cargas van a un tmp_path."""
    repository = ExcelPortfolioPositionsRepository(
        positions_path=tmp_path / "posiciones.xlsx", movements_path=tmp_path / "movimientos.xlsx"
    )
    monkeypatch.setattr(main, "portfolio_positions_repository", repository)
    return repository


def positions_file(rows):
    buffer = BytesIO()
    pd.DataFrame(rows).to_excel(buffer, index=False)
    buffer.seek(0)
    return buffer


def movements_file(rows):
    """Reproduce el extracto de MyInvestor: nueve filas de cabecera antes de los títulos."""
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, index=False, startrow=9)
    buffer.seek(0)
    return buffer


def row(mes, broker, isin, nombre, cantidad, precio=0.0, valor=None, moneda="EUR"):
    return {"Mes": mes, "Broker": broker, "ISIN": isin, "Nombre": nombre, "Cantidad": cantidad,
            "Precio cierre": precio, "Moneda": moneda, "Valor EUR": valor}


def upload(client, posiciones, movimientos=None):
    files = {"posiciones": ("Cartera.xlsx", posiciones, "application/vnd.ms-excel")}
    if movimientos is not None:
        files["movimientos"] = ("Movimientos.xlsx", movimientos, "application/vnd.ms-excel")
    response = client.post("/api/v1/portfolio/positions", files=files)
    assert response.status_code == 200, response.text
    return response.json()


def test_una_carga_deja_las_posiciones_disponibles_para_la_pantalla():
    client = TestClient(main.app)

    payload = upload(client, positions_file([row("2026-01", "Trade Republic", "DE0007074007", "KWS", 3, 74.4, 223.2)]))

    assert payload["resumen"]["meses_nuevos"] == ["2026-01"]
    assert payload["result"]["posiciones"][0]["valor"] == 223.2
    assert client.get("/api/v1/portfolio/positions").json()["result"]["meses"] == ["2026-01"]


def test_subir_un_mes_nuevo_conserva_los_anteriores():
    client = TestClient(main.app)
    upload(client, positions_file([row("2026-01", "TR", "A", "Alfa", 3, 10.0, 30.0)]))

    payload = upload(client, positions_file([row("2026-02", "TR", "A", "Alfa", 5, 10.0, 50.0)]))

    assert payload["result"]["meses"] == ["2026-01", "2026-02"]
    assert payload["resumen"]["meses_nuevos"] == ["2026-02"]


def test_subir_dos_veces_el_mismo_archivo_no_duplica_posiciones():
    client = TestClient(main.app)
    filas = [row("2026-01", "TR", "A", "Alfa", 3, 10.0, 30.0)]
    upload(client, positions_file(filas))

    payload = upload(client, positions_file(filas))

    assert payload["resumen"]["posiciones_nuevas"] == 0
    assert len(payload["result"]["posiciones"]) == 1


def test_los_movimientos_se_deducen_comparando_meses():
    client = TestClient(main.app)
    upload(client, positions_file([
        row("2026-01", "TR", "A", "Alfa", 3, 10.0, 30.0),
        row("2026-02", "TR", "A", "Alfa", 5, 10.0, 50.0),
        row("2026-02", "TR", "B", "Beta", 1, 20.0, 20.0),
    ]))

    movimientos = client.get("/api/v1/portfolio/positions").json()["result"]["movimientos"]

    assert sorted((m["Nombre"], m["tipo"]) for m in movimientos) == [("Alfa", "compra"), ("Beta", "alta")]


def test_un_mes_que_solo_trae_un_broker_no_vende_los_otros():
    client = TestClient(main.app)
    upload(client, positions_file([
        row("2026-01", "Degiro", "D", "Delta", 2, 10.0, 20.0),
        row("2026-01", "TR", "A", "Alfa", 3, 10.0, 30.0),
        row("2026-02", "TR", "A", "Alfa", 3, 10.0, 30.0),
    ]))

    movimientos = client.get("/api/v1/portfolio/positions").json()["result"]["movimientos"]

    assert movimientos == [], "DeGiro no viene en 2026-02; callar no es vender"


def test_la_comision_de_myinvestor_se_reparte_entre_sus_fondos():
    client = TestClient(main.app)
    movimientos = movements_file([
        {"Fecha Operación": "29/04/2026", "Fecha Valor": "29/04/2026",
         "Movimiento": "COM. GESTION 1T 2026", "Importe": -2.17, "Saldo": 10.0},
        {"Fecha Operación": "29/04/2026", "Fecha Valor": "29/04/2026",
         "Movimiento": "IVA COM. GESTION 1T 2026", "Importe": -0.46, "Saldo": 9.54},
    ])
    posiciones = positions_file([
        row("2026-04", "MyInvestor", "IE0032126645", "Vanguard US 500", 10, 100.0, 750.0),
        row("2026-04", "MyInvestor", "LU0996175948", "Index MSCI EM", 5, 50.0, 250.0),
        row("2026-04", "Trade Republic", "A", "Alfa", 1, 10.0, 1000.0),
    ])

    payload = upload(client, posiciones, movimientos)

    imputadas = {p["Nombre"]: p["comision_imputada"] for p in payload["result"]["posiciones"]}
    assert imputadas["Vanguard US 500"] == pytest.approx(2.63 * 0.75, abs=0.01)
    assert imputadas["Index MSCI EM"] == pytest.approx(2.63 * 0.25, abs=0.01)
    assert imputadas["Alfa"] is None, "la comisión es de MyInvestor, no toca a Trade Republic"


def test_los_movimientos_repetidos_entre_exportaciones_no_se_duplican():
    client = TestClient(main.app)
    filas = [{"Fecha Operación": "13/10/2025", "Fecha Valor": "13/10/2025",
              "Movimiento": "Aportacion automatica cartera", "Importe": 150, "Saldo": 159.29}]
    posiciones = [row("2026-01", "TR", "A", "Alfa", 1, 10.0, 10.0)]
    upload(client, positions_file(posiciones), movements_file(filas))

    payload = upload(client, positions_file(posiciones), movements_file(filas))

    assert payload["resumen"]["movimientos_nuevos"] == 0


def test_un_archivo_sin_las_columnas_obligatorias_se_rechaza_con_un_motivo():
    client = TestClient(main.app)

    response = client.post(
        "/api/v1/portfolio/positions",
        files={"posiciones": ("Cartera.xlsx", positions_file([{"Mes": "2026-01", "Cosa": 1}]), "application/vnd.ms-excel")},
    )

    assert response.status_code == 400
    assert "faltan columnas" in response.json()["detail"]


def test_solo_se_aceptan_excel():
    client = TestClient(main.app)

    response = client.post(
        "/api/v1/portfolio/positions",
        files={"posiciones": ("Cartera.csv", BytesIO(b"Mes,Broker,ISIN\n"), "text/csv")},
    )

    assert response.status_code == 400


def test_la_version_anterior_se_archiva_antes_de_sobrescribir(isolated_repository):
    client = TestClient(main.app)
    upload(client, positions_file([row("2026-01", "TR", "A", "Alfa", 3, 10.0, 30.0)]))

    upload(client, positions_file([row("2026-02", "TR", "A", "Alfa", 5, 10.0, 50.0)]))

    copias = list(isolated_repository.versions_dir.glob("*/posiciones.xlsx"))
    assert len(copias) == 1, "la primera carga no tenía nada que archivar; la segunda sí"
