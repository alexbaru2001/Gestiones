import json
from io import BytesIO

import pandas as pd
from fastapi.testclient import TestClient

import backend.main as backend_main
from backend.infrastructure.finance_checkpoint_repository import JsonFinanceCheckpointRepository
from backend.infrastructure.finance_history_repository import CsvFinanceHistoryRepository
from backend.infrastructure.transaction_history_repository import CsvTransactionHistoryRepository
from backend.main import app


def _workbook(gastos_rows: list[dict], ingresos_rows: list[dict]) -> BytesIO:
    workbook = BytesIO()
    gastos = pd.DataFrame(gastos_rows)
    ingresos = pd.DataFrame(ingresos_rows)
    transferencias = pd.DataFrame(columns=["Fecha", "Saliente", "Entrante", "Cantidad", "Comentario"])

    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        gastos.to_excel(writer, sheet_name="Gastos", index=False)
        ingresos.to_excel(writer, sheet_name="Ingresos", index=False)
        transferencias.to_excel(writer, sheet_name="Transferencias", index=False)

    workbook.seek(0)
    return workbook


def _gasto(fecha, cantidad, categoria="Alimentacion", etiquetas="", comentario=""):
    return {"Fecha": fecha, "Categoria": categoria, "Cuenta": "Principal", "Cantidad": cantidad, "Etiquetas": etiquetas, "Comentario": comentario}


def _ingreso(fecha, cantidad, categoria="Salario", cuenta="Principal", etiquetas="", comentario=""):
    return {"Fecha": fecha, "Categoria": categoria, "Cuenta": cuenta, "Cantidad": cantidad, "Etiquetas": etiquetas, "Comentario": comentario}


# 5 meses (2024-10 a 2025-02): salario + gasto todos los meses, un dividendo y una comisión antes del
# corte (nov/dic) y otro después (ene/feb), para ejercitar todos los acumuladores del checkpoint.
FULL_GASTOS = [
    _gasto("2024-10-05", 100.0),
    _gasto("2024-11-05", 100.0),
    _gasto("2024-12-05", 100.0),
    _gasto("2024-12-20", 2.0, categoria="Otros", etiquetas="Comision", comentario="Comisión compra"),
    _gasto("2025-01-05", 100.0),
    _gasto("2025-02-05", 100.0),
    _gasto("2025-02-15", 3.0, categoria="Otros", etiquetas="Comision", comentario="Comisión venta"),
]
FULL_INGRESOS = [
    _ingreso("2024-10-01", 2000.0),
    _ingreso("2024-11-01", 2000.0),
    _ingreso("2024-11-15", 5.0, categoria="Interes", cuenta="Degiro", etiquetas="Dividendos, ACME", comentario="ACME Corp"),
    _ingreso("2024-12-01", 2000.0),
    _ingreso("2025-01-01", 2000.0),
    _ingreso("2025-01-15", 7.0, categoria="Interes", cuenta="Degiro", etiquetas="Dividendos, ACME", comentario="ACME Corp"),
    _ingreso("2025-02-01", 2000.0),
]

CHUNK1_GASTOS = [row for row in FULL_GASTOS if row["Fecha"] < "2025-01-01"]
CHUNK1_INGRESOS = [row for row in FULL_INGRESOS if row["Fecha"] < "2025-01-01"]
CHUNK2_GASTOS = [row for row in FULL_GASTOS if row["Fecha"] >= "2025-01-01"]
CHUNK2_INGRESOS = [row for row in FULL_INGRESOS if row["Fecha"] >= "2025-01-01"]


def _numeric_rows(rows: list[dict]) -> list[dict]:
    """Normaliza los valores de las filas de historial (vengan de la API en JSON o del CSV en texto)
    a float redondeado, para poder comparar sin que el tipo de dato distorsione la igualdad."""
    numeric_fields = [
        "🎁 Regalos", "💼 Vacaciones", "📈 Inversiones", "Dinero Invertido", "💰 Ahorros",
        "Fondo de reserva cargado", "total", "💳 Gasto del mes", "💸 Presupuesto Mes",
        "🧾 Presupuesto Disponible", "📉 Deuda Presupuestaria mensual", "📉 Deuda Presupuestaria acumulada",
        "Dividendos", "Comisiones",
    ]
    normalized = []
    for row in rows:
        clean = {"Mes": row.get("Mes")}
        for field in numeric_fields:
            value = row.get(field)
            clean[field] = round(float(value), 2) if value not in (None, "") else None
        normalized.append(clean)
    return normalized


def test_split_commit_matches_one_shot_processing(tmp_path, monkeypatch):
    client = TestClient(app)

    # --- referencia: todo de una vez, sin checkpoint ---
    one_shot_repo = CsvFinanceHistoryRepository(tmp_path / "one_shot_historial.csv")
    one_shot_checkpoint_repo = JsonFinanceCheckpointRepository(tmp_path / "one_shot_checkpoint.json")
    monkeypatch.setattr(backend_main, "finance_history_repository", one_shot_repo)
    monkeypatch.setattr(backend_main, "finance_checkpoint_repository", one_shot_checkpoint_repo)
    one_shot_txn_repo = CsvTransactionHistoryRepository(
        gastos_path=tmp_path / "one_shot_historial_gastos.csv", ingresos_path=tmp_path / "one_shot_historial_ingresos.csv"
    )
    monkeypatch.setattr(backend_main, "transaction_history_repository", one_shot_txn_repo)

    workbook = _workbook(FULL_GASTOS, FULL_INGRESOS)
    response = client.post(
        f"/api/v1/process?fecha_inicio=2024-10-01",
        data={"modo": "visualizar"},
        files={"file": ("Inicio.xlsx", workbook.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
    )
    assert response.status_code == 200
    one_shot_rows = _numeric_rows(response.json()["result"]["historial"]["resumen"])
    assert [row["Mes"] for row in one_shot_rows] == ["2024-10", "2024-11", "2024-12", "2025-01", "2025-02"]

    # --- en dos tramos: primero 2024-10→2024-12, luego 2025-01→2025-02 con el checkpoint ---
    split_repo = CsvFinanceHistoryRepository(tmp_path / "split_historial.csv")
    split_checkpoint_repo = JsonFinanceCheckpointRepository(tmp_path / "split_checkpoint.json")
    monkeypatch.setattr(backend_main, "finance_history_repository", split_repo)
    monkeypatch.setattr(backend_main, "finance_checkpoint_repository", split_checkpoint_repo)
    split_txn_repo = CsvTransactionHistoryRepository(
        gastos_path=tmp_path / "split_historial_gastos.csv", ingresos_path=tmp_path / "split_historial_ingresos.csv"
    )
    monkeypatch.setattr(backend_main, "transaction_history_repository", split_txn_repo)

    chunk1 = _workbook(CHUNK1_GASTOS, CHUNK1_INGRESOS)
    response1 = client.post(
        f"/api/v1/process?fecha_inicio=2024-10-01",
        data={"modo": "historico"},
        files={"file": ("Inicio.xlsx", chunk1.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
    )
    assert response1.status_code == 200
    assert split_checkpoint_repo.load()["as_of_month"] == "2024-12"

    chunk2 = _workbook(CHUNK2_GASTOS, CHUNK2_INGRESOS)
    response2 = client.post(
        f"/api/v1/process?fecha_inicio=2025-01-01",
        data={"modo": "historico"},
        files={"file": ("Inicio.xlsx", chunk2.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
    )
    assert response2.status_code == 200
    assert split_checkpoint_repo.load()["as_of_month"] == "2025-02"

    split_rows = _numeric_rows(split_repo.load())
    assert [row["Mes"] for row in split_rows] == ["2024-10", "2024-11", "2024-12", "2025-01", "2025-02"]

    assert split_rows == one_shot_rows


def test_historico_repeated_commit_skips_existing_months_without_error(tmp_path, monkeypatch):
    """Reenviar un tramo que ya está guardado no debe rechazarse: simplemente no añade nada nuevo
    (los meses repetidos nunca se sobrescriben) y deja el histórico intacto."""
    client = TestClient(app)
    repo = CsvFinanceHistoryRepository(tmp_path / "historial.csv")
    checkpoint_repo = JsonFinanceCheckpointRepository(tmp_path / "checkpoint.json")
    monkeypatch.setattr(backend_main, "finance_history_repository", repo)
    monkeypatch.setattr(backend_main, "finance_checkpoint_repository", checkpoint_repo)
    txn_txn_repo = CsvTransactionHistoryRepository(
        gastos_path=tmp_path / "historial_gastos.csv", ingresos_path=tmp_path / "historial_ingresos.csv"
    )
    monkeypatch.setattr(backend_main, "transaction_history_repository", txn_txn_repo)

    chunk1 = _workbook(CHUNK1_GASTOS, CHUNK1_INGRESOS)
    response1 = client.post(
        f"/api/v1/process?fecha_inicio=2024-10-01",
        data={"modo": "historico"},
        files={"file": ("Inicio.xlsx", chunk1.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
    )
    assert response1.status_code == 200
    rows_after_first_commit = repo.load()

    # Reenviar el mismo tramo (p.ej. un histórico creado antes de que existiera el checkpoint, que
    # se sincroniza por primera vez) no debe fallar ni duplicar filas.
    response_repeat = client.post(
        f"/api/v1/process?fecha_inicio=2024-10-01",
        data={"modo": "historico"},
        files={"file": ("Inicio.xlsx", chunk1.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
    )
    assert response_repeat.status_code == 200
    body = response_repeat.json()
    assert body["meses_nuevos"] == []
    assert body["meses_ya_guardados"] == ["2024-10", "2024-11", "2024-12"]

    rows_after_repeat = repo.load()
    assert rows_after_repeat == rows_after_first_commit
    assert [row["Mes"] for row in rows_after_repeat] == ["2024-10", "2024-11", "2024-12"]


def test_historico_bootstraps_checkpoint_from_preexisting_history(tmp_path, monkeypatch):
    """Si el histórico ya tenía datos guardados desde antes de que existiera el checkpoint (el caso
    de un usuario que ya usaba /api/v1/process cuando esta función no existía), reenviar el Excel
    completo debe crear el checkpoint sin duplicar ni rechazar nada, permitiendo que a partir de ahí
    los tramos incrementales funcionen."""
    client = TestClient(app)
    repo = CsvFinanceHistoryRepository(tmp_path / "historial.csv")
    checkpoint_repo = JsonFinanceCheckpointRepository(tmp_path / "checkpoint.json")
    monkeypatch.setattr(backend_main, "finance_history_repository", repo)
    monkeypatch.setattr(backend_main, "finance_checkpoint_repository", checkpoint_repo)
    txn_txn_repo = CsvTransactionHistoryRepository(
        gastos_path=tmp_path / "historial_gastos.csv", ingresos_path=tmp_path / "historial_ingresos.csv"
    )
    monkeypatch.setattr(backend_main, "transaction_history_repository", txn_txn_repo)

    # Histórico "preexistente": guardado directamente, como si viniera del flujo antiguo (sin checkpoint).
    workbook = _workbook(FULL_GASTOS, FULL_INGRESOS)
    preexisting = client.post(
        f"/api/v1/process?fecha_inicio=2024-10-01",
        data={"modo": "visualizar"},
        files={"file": ("Inicio.xlsx", workbook.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
    ).json()["result"]["historial"]["resumen"]
    repo.save(preexisting)
    assert checkpoint_repo.load() is None

    # Reenviar el mismo Excel completo en modo histórico: no hay meses nuevos, pero debe sincronizar
    # el checkpoint para que futuros tramos incrementales puedan continuar.
    response = client.post(
        f"/api/v1/process?fecha_inicio=2024-10-01",
        data={"modo": "historico"},
        files={"file": ("Inicio.xlsx", workbook.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["meses_nuevos"] == []
    assert body["meses_ya_guardados"] == ["2024-10", "2024-11", "2024-12", "2025-01", "2025-02"]

    checkpoint = checkpoint_repo.load()
    assert checkpoint is not None
    assert checkpoint["as_of_month"] == "2025-02"
    assert _numeric_rows(repo.load()) == _numeric_rows(preexisting)


def test_historico_merges_without_overwriting_existing_rows(tmp_path, monkeypatch):
    client = TestClient(app)
    repo = CsvFinanceHistoryRepository(tmp_path / "historial.csv")
    checkpoint_repo = JsonFinanceCheckpointRepository(tmp_path / "checkpoint.json")
    monkeypatch.setattr(backend_main, "finance_history_repository", repo)
    monkeypatch.setattr(backend_main, "finance_checkpoint_repository", checkpoint_repo)
    txn_txn_repo = CsvTransactionHistoryRepository(
        gastos_path=tmp_path / "historial_gastos.csv", ingresos_path=tmp_path / "historial_ingresos.csv"
    )
    monkeypatch.setattr(backend_main, "transaction_history_repository", txn_txn_repo)

    chunk1 = _workbook(CHUNK1_GASTOS, CHUNK1_INGRESOS)
    client.post(
        f"/api/v1/process?fecha_inicio=2024-10-01",
        data={"modo": "historico"},
        files={"file": ("Inicio.xlsx", chunk1.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
    )
    chunk2 = _workbook(CHUNK2_GASTOS, CHUNK2_INGRESOS)
    client.post(
        f"/api/v1/process?fecha_inicio=2025-01-01",
        data={"modo": "historico"},
        files={"file": ("Inicio.xlsx", chunk2.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
    )

    rows = repo.load()
    assert [row["Mes"] for row in rows] == ["2024-10", "2024-11", "2024-12", "2025-01", "2025-02"]


def test_visualizar_mode_never_persists(tmp_path, monkeypatch):
    client = TestClient(app)
    repo = CsvFinanceHistoryRepository(tmp_path / "historial.csv")
    checkpoint_repo = JsonFinanceCheckpointRepository(tmp_path / "checkpoint.json")
    monkeypatch.setattr(backend_main, "finance_history_repository", repo)
    monkeypatch.setattr(backend_main, "finance_checkpoint_repository", checkpoint_repo)
    txn_txn_repo = CsvTransactionHistoryRepository(
        gastos_path=tmp_path / "historial_gastos.csv", ingresos_path=tmp_path / "historial_ingresos.csv"
    )
    monkeypatch.setattr(backend_main, "transaction_history_repository", txn_txn_repo)

    workbook = _workbook(FULL_GASTOS, FULL_INGRESOS)
    response = client.post(
        f"/api/v1/process?fecha_inicio=2024-10-01",
        data={"modo": "visualizar"},
        files={"file": ("Inicio.xlsx", workbook.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
    )

    assert response.status_code == 200
    assert response.json()["result"]["historial"]["resumen"]
    assert repo.load() == []
    assert checkpoint_repo.load() is None


OBJETIVO_COCHE = [
    {
        "nombre": "Coche",
        "etiquetas": ["coche"],
        "fraccion_presupuesto": 0.1,
        "duracion_meses": 6,
        "mes_inicio": "2024-10",
    }
]


def test_objetivo_saldo_continues_across_split_commit(tmp_path, monkeypatch):
    """Un objetivo que sigue abierto en el segundo tramo debe continuar con el saldo exacto que
    tenía al cerrar el primero, no reiniciar desde 0 ni desde el saldo_inicial fijo de su config."""
    client = TestClient(app)

    # --- referencia: todo de una vez ---
    one_shot_repo = CsvFinanceHistoryRepository(tmp_path / "one_shot_historial.csv")
    one_shot_checkpoint_repo = JsonFinanceCheckpointRepository(tmp_path / "one_shot_checkpoint.json")
    monkeypatch.setattr(backend_main, "finance_history_repository", one_shot_repo)
    monkeypatch.setattr(backend_main, "finance_checkpoint_repository", one_shot_checkpoint_repo)
    one_shot_txn_repo = CsvTransactionHistoryRepository(
        gastos_path=tmp_path / "one_shot_historial_gastos.csv", ingresos_path=tmp_path / "one_shot_historial_ingresos.csv"
    )
    monkeypatch.setattr(backend_main, "transaction_history_repository", one_shot_txn_repo)

    workbook = _workbook(FULL_GASTOS, FULL_INGRESOS)
    response = client.post(
        f"/api/v1/process?fecha_inicio=2024-10-01",
        data={"modo": "visualizar", "objetivos_json": json.dumps(OBJETIVO_COCHE)},
        files={"file": ("Inicio.xlsx", workbook.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
    )
    assert response.status_code == 200
    one_shot_objetivos = response.json()["result"]["historial"]["objetivos"]
    one_shot_enero = [row for row in one_shot_objetivos if row["Mes"] == "2025-01"][0]

    # --- en dos tramos, reenviando la misma config de objetivo (sin tocar saldo_inicial a mano) ---
    split_repo = CsvFinanceHistoryRepository(tmp_path / "split_historial.csv")
    split_checkpoint_repo = JsonFinanceCheckpointRepository(tmp_path / "split_checkpoint.json")
    monkeypatch.setattr(backend_main, "finance_history_repository", split_repo)
    monkeypatch.setattr(backend_main, "finance_checkpoint_repository", split_checkpoint_repo)
    split_txn_repo = CsvTransactionHistoryRepository(
        gastos_path=tmp_path / "split_historial_gastos.csv", ingresos_path=tmp_path / "split_historial_ingresos.csv"
    )
    monkeypatch.setattr(backend_main, "transaction_history_repository", split_txn_repo)

    chunk1 = _workbook(CHUNK1_GASTOS, CHUNK1_INGRESOS)
    response1 = client.post(
        f"/api/v1/process?fecha_inicio=2024-10-01",
        data={"modo": "historico", "objetivos_json": json.dumps(OBJETIVO_COCHE)},
        files={"file": ("Inicio.xlsx", chunk1.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
    )
    assert response1.status_code == 200
    checkpoint_after_chunk1 = split_checkpoint_repo.load()
    assert "Coche" in checkpoint_after_chunk1["objetivos_saldos"]

    chunk2 = _workbook(CHUNK2_GASTOS, CHUNK2_INGRESOS)
    response2 = client.post(
        f"/api/v1/process?fecha_inicio=2025-01-01",
        data={"modo": "historico", "objetivos_json": json.dumps(OBJETIVO_COCHE)},
        files={"file": ("Inicio.xlsx", chunk2.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
    )
    assert response2.status_code == 200
    split_objetivos = response2.json()["result"]["historial"]["objetivos"]
    split_enero = [row for row in split_objetivos if row["Mes"] == "2025-01"][0]

    assert round(split_enero["saldo_fin_mes"], 2) == round(one_shot_enero["saldo_fin_mes"], 2)


def test_recalculado_desde_cero_flag_reflects_whether_checkpoint_was_used(tmp_path, monkeypatch):
    client = TestClient(app)
    repo = CsvFinanceHistoryRepository(tmp_path / "historial.csv")
    checkpoint_repo = JsonFinanceCheckpointRepository(tmp_path / "checkpoint.json")
    monkeypatch.setattr(backend_main, "finance_history_repository", repo)
    monkeypatch.setattr(backend_main, "finance_checkpoint_repository", checkpoint_repo)
    txn_txn_repo = CsvTransactionHistoryRepository(
        gastos_path=tmp_path / "historial_gastos.csv", ingresos_path=tmp_path / "historial_ingresos.csv"
    )
    monkeypatch.setattr(backend_main, "transaction_history_repository", txn_txn_repo)

    chunk1 = _workbook(CHUNK1_GASTOS, CHUNK1_INGRESOS)
    response1 = client.post(
        f"/api/v1/process?fecha_inicio=2024-10-01",
        data={"modo": "historico"},
        files={"file": ("Inicio.xlsx", chunk1.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
    )
    assert response1.status_code == 200
    assert response1.json()["recalculado_desde_cero"] is False

    # fecha_inicio no avanza respecto al checkpoint (2024-12): se recalcula desde cero, y el flag debe avisarlo.
    response_repeat = client.post(
        f"/api/v1/process?fecha_inicio=2024-10-01",
        data={"modo": "visualizar"},
        files={"file": ("Inicio.xlsx", chunk1.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
    )
    assert response_repeat.status_code == 200
    assert response_repeat.json()["recalculado_desde_cero"] is True

    # fecha_inicio avanza más allá del checkpoint: se usa la continuidad con normalidad.
    chunk2 = _workbook(CHUNK2_GASTOS, CHUNK2_INGRESOS)
    response2 = client.post(
        f"/api/v1/process?fecha_inicio=2025-01-01",
        data={"modo": "visualizar"},
        files={"file": ("Inicio.xlsx", chunk2.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
    )
    assert response2.status_code == 200
    assert response2.json()["recalculado_desde_cero"] is False


def test_analisis_desglose_incluye_todo_el_historico_no_solo_el_ultimo_tramo(tmp_path, monkeypatch):
    """Bug real reportado: tras subir solo el tramo más reciente, "Intereses" (y cualquier desglose
    por categoría) salía calculado solo con ese tramo, sin los meses ya guardados de antes. El
    detalle de transacciones debe persistirse también, para que estos desgloses usen todo el
    histórico igual que si se hubiera procesado de una sola vez."""
    client = TestClient(app)

    # --- referencia: todo de una vez ---
    one_shot_repo = CsvFinanceHistoryRepository(tmp_path / "one_shot_historial.csv")
    one_shot_checkpoint_repo = JsonFinanceCheckpointRepository(tmp_path / "one_shot_checkpoint.json")
    one_shot_txn_repo = CsvTransactionHistoryRepository(
        gastos_path=tmp_path / "one_shot_gastos.csv", ingresos_path=tmp_path / "one_shot_ingresos.csv"
    )
    monkeypatch.setattr(backend_main, "finance_history_repository", one_shot_repo)
    monkeypatch.setattr(backend_main, "finance_checkpoint_repository", one_shot_checkpoint_repo)
    monkeypatch.setattr(backend_main, "transaction_history_repository", one_shot_txn_repo)

    workbook = _workbook(FULL_GASTOS, FULL_INGRESOS)
    one_shot_response = client.post(
        f"/api/v1/process?fecha_inicio=2024-10-01",
        data={"modo": "visualizar"},
        files={"file": ("Inicio.xlsx", workbook.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
    ).json()
    one_shot_analisis = one_shot_response["result"]["analisis"]
    one_shot_interes_total = sum(
        row.get("Interes", 0) for row in one_shot_analisis["ingresos"]["categorias"]
    )
    one_shot_dividendo_pagos = len(one_shot_analisis["dividendos_pagos"])

    # --- en dos tramos: el segundo solo sube su propio Excel (enero-febrero), como hace el usuario real ---
    split_repo = CsvFinanceHistoryRepository(tmp_path / "split_historial.csv")
    split_checkpoint_repo = JsonFinanceCheckpointRepository(tmp_path / "split_checkpoint.json")
    split_txn_repo = CsvTransactionHistoryRepository(
        gastos_path=tmp_path / "split_gastos.csv", ingresos_path=tmp_path / "split_ingresos.csv"
    )
    monkeypatch.setattr(backend_main, "finance_history_repository", split_repo)
    monkeypatch.setattr(backend_main, "finance_checkpoint_repository", split_checkpoint_repo)
    monkeypatch.setattr(backend_main, "transaction_history_repository", split_txn_repo)

    chunk1 = _workbook(CHUNK1_GASTOS, CHUNK1_INGRESOS)
    client.post(
        f"/api/v1/process?fecha_inicio=2024-10-01",
        data={"modo": "historico"},
        files={"file": ("Inicio.xlsx", chunk1.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
    )

    # El segundo tramo sube SOLO enero-febrero (nunca repite 2024-10/11/12), tal como hace el usuario
    # real con su Excel incremental.
    chunk2 = _workbook(CHUNK2_GASTOS, CHUNK2_INGRESOS)
    response2 = client.post(
        f"/api/v1/process?fecha_inicio=2025-01-01",
        data={"modo": "historico"},
        files={"file": ("Inicio.xlsx", chunk2.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
    )
    assert response2.status_code == 200
    split_analisis = response2.json()["result"]["analisis"]
    split_interes_total = sum(row.get("Interes", 0) for row in split_analisis["ingresos"]["categorias"])
    split_dividendo_pagos = len(split_analisis["dividendos_pagos"])

    assert round(split_interes_total, 2) == round(one_shot_interes_total, 2)
    assert split_dividendo_pagos == one_shot_dividendo_pagos


def test_delete_historico_clears_summary_checkpoint_and_transaction_detail(tmp_path, monkeypatch):
    client = TestClient(app)
    repo = CsvFinanceHistoryRepository(tmp_path / "historial.csv")
    checkpoint_repo = JsonFinanceCheckpointRepository(tmp_path / "checkpoint.json")
    txn_repo = CsvTransactionHistoryRepository(
        gastos_path=tmp_path / "gastos.csv", ingresos_path=tmp_path / "ingresos.csv"
    )
    monkeypatch.setattr(backend_main, "finance_history_repository", repo)
    monkeypatch.setattr(backend_main, "finance_checkpoint_repository", checkpoint_repo)
    monkeypatch.setattr(backend_main, "transaction_history_repository", txn_repo)

    chunk1 = _workbook(CHUNK1_GASTOS, CHUNK1_INGRESOS)
    commit_response = client.post(
        f"/api/v1/process?fecha_inicio=2024-10-01",
        data={"modo": "historico"},
        files={"file": ("Inicio.xlsx", chunk1.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
    )
    assert commit_response.status_code == 200
    assert repo.load()
    assert checkpoint_repo.load() is not None
    assert txn_repo.load_gastos()
    assert txn_repo.load_ingresos()

    delete_response = client.delete("/api/v1/process/historico")
    assert delete_response.status_code == 200
    assert delete_response.json()["ok"] is True

    assert repo.load() == []
    assert checkpoint_repo.load() is None
    assert txn_repo.load_gastos() == []
    assert txn_repo.load_ingresos() == []

    # Borrar cuando ya no hay nada guardado no debe fallar.
    second_delete = client.delete("/api/v1/process/historico")
    assert second_delete.status_code == 200
