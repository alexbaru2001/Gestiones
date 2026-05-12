from typing import Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.config import get_cors_origins
from backend.domain.investments import analyze_ticker
from backend.domain.models import PipelineConfig
from backend.domain.portfolio import UploadedInvestmentFile
from backend.infrastructure.container import build_process_finance_workbook_use_case
from backend.infrastructure.finance_history_repository import CsvFinanceHistoryRepository
from backend.infrastructure.objectives_repository import (
    JsonObjectivesRepository,
    ObjectivesStorageError,
    ObjectivesValidationError,
    parse_objectives_json,
)
from backend.infrastructure.portfolio_repository import LocalPortfolioRepository

app = FastAPI(title="Gestiones Backend", version="0.1.0")
objectives_repository = JsonObjectivesRepository()
portfolio_repository = LocalPortfolioRepository()
finance_history_repository = CsvFinanceHistoryRepository()

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


@app.post("/api/v1/portfolio/import")
async def import_portfolio(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    if not files:
        raise HTTPException(status_code=400, detail="Selecciona al menos un PDF o Excel de cartera.")

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
):
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

    use_case = build_process_finance_workbook_use_case()
    try:
        result = use_case.execute(
            excel_bytes=content,
            params=config,
            objetivos=objetivos,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    finance_history_repository.save(result.historial.resumen)
    return {"ok": True, "result": result.to_dict()}
