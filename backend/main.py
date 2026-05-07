from fastapi import FastAPI, UploadFile, File, HTTPException
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


@app.post("/api/v1/process")
async def process_workbook(
    file: UploadFile = File(...),
    fecha_inicio: str = "2024-10-01",
    porcentaje_gasto: float = 0.3,
    porcentaje_inversion: float = 0.1,
    porcentaje_vacaciones: float = 0.05,
):
    if not file.filename.lower().endswith((".xlsx", ".xlsm", ".xls")):
        raise HTTPException(status_code=400, detail="El archivo debe ser Excel (.xlsx/.xlsm/.xls)")

    content = await file.read()
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
            objetivos=[],
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"ok": True, "result": result}
