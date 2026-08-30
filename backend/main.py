from typing import Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.config import get_cors_origins
from backend.domain.investments import analyze_ticker
from backend.domain.models import PipelineConfig
from backend.domain.portfolio import UploadedInvestmentFile
from backend.infrastructure.container import build_process_finance_workbook_use_case
from backend.infrastructure.finance_checkpoint_repository import JsonFinanceCheckpointRepository
from backend.infrastructure.finance_history_repository import CsvFinanceHistoryRepository
from backend.infrastructure.objectives_repository import (
    JsonObjectivesRepository,
    ObjectivesStorageError,
    ObjectivesValidationError,
    parse_objectives_json,
)
from backend.infrastructure.portfolio_repository import LocalPortfolioRepository
from backend.infrastructure.transaction_history_repository import CsvTransactionHistoryRepository

app = FastAPI(title="Gestiones Backend", version="0.1.0")
objectives_repository = JsonObjectivesRepository()
portfolio_repository = LocalPortfolioRepository()
finance_history_repository = CsvFinanceHistoryRepository()
finance_checkpoint_repository = JsonFinanceCheckpointRepository()
transaction_history_repository = CsvTransactionHistoryRepository()

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v1/objectives")
def get_objectives() -> dict[str, list[dict[str, Any]]]:
    try:
        return {"objetivos": objectives_repository.load()}
    except ObjectivesStorageError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.put("/api/v1/objectives")
def save_objectives(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    try:
        return {"objetivos": objectives_repository.save(payload)}
    except ObjectivesValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/v1/investments/analyze")
def analyze_investment(ticker: str) -> dict[str, Any]:
    try:
        return {"ok": True, "result": analyze_ticker(ticker)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/v1/portfolio")
def get_portfolio() -> dict[str, Any]:
    snapshot = portfolio_repository.load()
    return {"ok": True, "result": snapshot}


@app.get("/api/v1/portfolio/snapshots")
def get_portfolio_snapshots() -> dict[str, Any]:
    snapshots = portfolio_repository.load_snapshots()
    return {"ok": True, "result": snapshots}


@app.put("/api/v1/portfolio/snapshots/{snapshot_key}/date")
def update_portfolio_snapshot_date(snapshot_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    snapshot_date = payload.get("snapshot_date")
    if not isinstance(snapshot_date, str) or not snapshot_date.strip():
        raise HTTPException(status_code=400, detail="Indica una fecha válida para la foto.")
    try:
        snapshot = portfolio_repository.update_snapshot_date(snapshot_key, snapshot_date.strip())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"ok": True, "result": snapshot, "snapshots": portfolio_repository.load_snapshots()}


@app.post("/api/v1/portfolio/import")
async def import_portfolio(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    if not files:
        raise HTTPException(status_code=400, detail="Selecciona al menos un PDF o Excel de cartera.")
    if len(files) != 3:
        raise HTTPException(status_code=400, detail="Selecciona los 3 archivos de foto: Trade Republic, MyInvestor y DeGiro.")

    uploaded_files = []
    for file in files:
        filename = file.filename or "documento"
        if not filename.lower().endswith((".pdf", ".xlsx")):
            raise HTTPException(status_code=400, detail="Solo se aceptan archivos PDF o XLSX de cartera.")
        uploaded_files.append(UploadedInvestmentFile(filename=filename, content=await file.read()))

    snapshot = portfolio_repository.save_uploads_and_rebuild(uploaded_files)
    return {"ok": True, "result": snapshot, "snapshots": portfolio_repository.load_snapshots()}


@app.post("/api/v1/process")
async def process_workbook(
    file: UploadFile = File(...),
    fecha_inicio: str = "2024-10-01",
    porcentaje_gasto: float = 0.3,
    porcentaje_inversion: float = 0.1,
    porcentaje_vacaciones: float = 0.05,
    objetivos_json: str | None = Form(default=None),
    modo: str = Form(default="visualizar"),
):
    if modo not in ("visualizar", "historico"):
        raise HTTPException(status_code=400, detail="modo debe ser 'visualizar' o 'historico'.")
    if not file.filename.lower().endswith((".xlsx", ".xlsm", ".xls")):
        raise HTTPException(status_code=400, detail="El archivo debe ser Excel (.xlsx/.xlsm/.xls)")

    content = await file.read()
    try:
        objetivos = parse_objectives_json(objetivos_json)
        config = PipelineConfig(
            fecha_inicio=fecha_inicio,
            porcentaje_gasto=porcentaje_gasto,
            porcentaje_inversion=porcentaje_inversion,
            porcentaje_vacaciones=porcentaje_vacaciones,
        )
    except ObjectivesValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    checkpoint = finance_checkpoint_repository.load()
    checkpoint_as_of = checkpoint.get("as_of_month") if checkpoint else None
    fecha_inicio_month = fecha_inicio[:7]

    recalculado_desde_cero = bool(checkpoint_as_of and fecha_inicio_month <= checkpoint_as_of)
    if recalculado_desde_cero:
        # Repetir o solapar el tramo ya cerrado no necesita continuar los acumuladores desde el
        # checkpoint: se recalcula igual que si nunca hubiera existido. El paso de guardado más abajo
        # ya se encarga de no duplicar los meses que ya estén en el histórico (o de sincronizar el
        # checkpoint si el Excel completo coincide con lo ya guardado, sin escribir nada nuevo).
        checkpoint = None

    historical_transactions = {
        "gastos": transaction_history_repository.load_gastos(),
        "ingresos": transaction_history_repository.load_ingresos(),
    }

    use_case = build_process_finance_workbook_use_case()
    try:
        result = use_case.execute(
            excel_bytes=content,
            params=config,
            objetivos=objetivos,
            checkpoint=checkpoint,
            historical_transactions=historical_transactions,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # El resumen mensual que devuelve el pipeline en ESTE cálculo solo cubre el tramo del Excel
    # subido (p.ej. solo enero-agosto si es lo único que se sube). Para que el frontend (gráficas de
    # Tipologías, Presupuesto, Ahorro...) siempre vea el histórico completo, se combina aquí con lo
    # ya guardado — tanto en "visualizar" (sin persistir nada) como en "historico" (que además persiste).
    computed_rows = result.historial.resumen
    if modo == "historico" and not computed_rows:
        raise HTTPException(status_code=400, detail="No hay ningún mes que procesar en ese rango.")
    existing_rows = finance_history_repository.load()
    existing_months = {row.get("Mes") for row in existing_rows}
    new_rows = [row for row in computed_rows if row.get("Mes") not in existing_months]
    meses_nuevos = sorted(row.get("Mes") for row in new_rows)
    meses_ya_guardados = sorted({row.get("Mes") for row in computed_rows if row.get("Mes") in existing_months})
    display_rows = sorted(existing_rows + new_rows, key=lambda row: row.get("Mes") or "")

    if modo == "historico":
        if new_rows:
            finance_history_repository.save(display_rows)
        # El checkpoint se sincroniza siempre con el cierre de este cálculo, tanto si había meses
        # nuevos que añadir como si el Excel solo confirmaba lo que ya estaba guardado (por ejemplo,
        # la primera vez que se usa este flujo con un histórico que ya existía de antes).
        if result.checkpoint:
            finance_checkpoint_repository.save(result.checkpoint)
        # Guardamos también el detalle de gastos/ingresos de los meses nuevos, para que los
        # desgloses por categoría (Intereses, Gastos por tipo, Dividendos por empresa...) puedan
        # calcularse sobre todo el histórico y no solo sobre el Excel de este tramo.
        meses_nuevos_set = set(meses_nuevos)
        transacciones = result.transacciones or {}
        transaction_history_repository.append_gastos(
            [row for row in transacciones.get("gastos", []) if row.get("Mes") in meses_nuevos_set]
        )
        transaction_history_repository.append_ingresos(
            [row for row in transacciones.get("ingresos", []) if row.get("Mes") in meses_nuevos_set]
        )

    result_dict = result.to_dict()
    result_dict["historial"]["resumen"] = display_rows
    result_dict["historial"]["meses"] = len(display_rows)
    result_dict["historial"]["ultimo_mes"] = display_rows[-1] if display_rows else None

    return {
        "ok": True,
        "result": result_dict,
        "modo": modo,
        "meses_nuevos": meses_nuevos,
        "meses_ya_guardados": meses_ya_guardados,
        "recalculado_desde_cero": recalculado_desde_cero,
    }


@app.get("/api/v1/process/checkpoint")
def get_finance_checkpoint() -> dict[str, Any]:
    return {"ok": True, "result": finance_checkpoint_repository.load()}


@app.delete("/api/v1/process/historico")
def delete_historico() -> dict[str, Any]:
    """Borra por completo el histórico guardado (resumen mensual, checkpoint y detalle de
    gastos/ingresos), para volver a empezar desde cero subiendo el Excel completo de nuevo."""
    finance_history_repository.delete()
    finance_checkpoint_repository.delete()
    transaction_history_repository.delete()
    return {"ok": True}
