import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.infrastructure.container import build_process_finance_workbook_use_case

app = FastAPI(title="Gestiones Backend", version="0.1.0")
OBJECTIVES_PATH = Path(__file__).resolve().parents[1] / "Personal_finanzas" / "Data" / "objetivos_vista.json"

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

    return _normalize_objectives_payload(data, detail_prefix="objetivos_json")


def _normalize_objectives_payload(data: Any, detail_prefix: str = "objetivos") -> list[dict[str, Any]]:
    if isinstance(data, dict):
        data = data.get("objetivos", [])

    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise HTTPException(status_code=400, detail=f"{detail_prefix} debe ser una lista de objetivos")

    objectives: list[dict[str, Any]] = []
    names: set[str] = set()
    total_fraction = 0.0

    for raw in data:
        name = str(raw.get("nombre", "")).strip()
        if not name:
            raise HTTPException(status_code=400, detail="Cada objetivo necesita nombre")
        if name in names:
            raise HTTPException(status_code=400, detail="Los objetivos deben tener nombres únicos")
        names.add(name)

        tags_raw = raw.get("etiquetas", [])
        if isinstance(tags_raw, str):
            tags = [tag.strip().lower() for tag in tags_raw.split(",") if tag.strip()]
        elif isinstance(tags_raw, list):
            tags = [str(tag).strip().lower() for tag in tags_raw if str(tag).strip()]
        else:
            raise HTTPException(status_code=400, detail=f"Objetivo '{name}': etiquetas debe ser texto o lista")

        fraction = float(raw.get("fraccion_presupuesto", 0.0))
        if fraction < 0 or fraction > 1:
            raise HTTPException(status_code=400, detail=f"Objetivo '{name}': fraccion_presupuesto fuera de [0,1]")
        total_fraction += fraction

        duration = int(raw.get("duracion_meses", 0))
        if duration <= 0:
            raise HTTPException(status_code=400, detail=f"Objetivo '{name}': duracion_meses debe ser > 0")

        start_month = str(raw.get("mes_inicio", "")).strip()[:7]
        if len(start_month) != 7 or start_month[4] != "-":
            raise HTTPException(status_code=400, detail=f"Objetivo '{name}': mes_inicio debe tener formato YYYY-MM")

        objectives.append(
            {
                "nombre": name,
                "etiquetas": tags,
                "fraccion_presupuesto": fraction,
                "duracion_meses": duration,
                "mes_inicio": start_month,
                "saldo_inicial": float(raw.get("saldo_inicial", 0.0)),
            }
        )

    if total_fraction > 1 + 1e-9:
        raise HTTPException(status_code=400, detail="La suma de fraccion_presupuesto de los objetivos supera 1")

    return objectives


@app.get("/api/v1/objectives")
def get_objectives() -> dict[str, list[dict[str, Any]]]:
    if not OBJECTIVES_PATH.exists():
        return {"objetivos": []}

    try:
        data = json.loads(OBJECTIVES_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="El archivo de objetivos no contiene JSON válido") from exc

    return {"objetivos": _normalize_objectives_payload(data)}


@app.put("/api/v1/objectives")
def save_objectives(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    objectives = _normalize_objectives_payload(payload)
    OBJECTIVES_PATH.parent.mkdir(parents=True, exist_ok=True)
    OBJECTIVES_PATH.write_text(
        json.dumps({"objetivos": objectives}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {"objetivos": objectives}


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
