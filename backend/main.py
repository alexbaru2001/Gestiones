import json
from typing import Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.infrastructure.container import build_process_finance_workbook_use_case

app = FastAPI(title="Gestiones Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _parse_objetivos_json(objetivos_json: str | None) -> list[dict[str, Any]]:
    if not objetivos_json:
        return []

    try:
        data = json.loads(objetivos_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="objetivos_json debe ser JSON válido") from exc

    if isinstance(data, dict):
        data = data.get("objetivos", [])

    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise HTTPException(status_code=400, detail="objetivos_json debe ser una lista de objetivos")

    return data


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
    objetivos = _parse_objetivos_json(objetivos_json)
    use_case = build_process_finance_workbook_use_case()
    try:
        result = use_case.execute(
            excel_bytes=content,
            params={
                "fecha_inicio": fecha_inicio,
                "porcentaje_gasto": porcentaje_gasto,
                "porcentaje_inversion": porcentaje_inversion,
                "porcentaje_vacaciones": porcentaje_vacaciones,
            },
            objetivos=objetivos,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"ok": True, "result": result}
