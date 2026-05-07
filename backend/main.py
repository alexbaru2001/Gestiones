from typing import Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.config import get_cors_origins
from backend.domain.models import PipelineConfig
from backend.infrastructure.container import build_process_finance_workbook_use_case
from backend.infrastructure.objectives_repository import (
    JsonObjectivesRepository,
    ObjectivesStorageError,
    ObjectivesValidationError,
    parse_objectives_json,
)

app = FastAPI(title="Gestiones Backend", version="0.1.0")
objectives_repository = JsonObjectivesRepository()

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

    return {"ok": True, "result": result.to_dict()}
