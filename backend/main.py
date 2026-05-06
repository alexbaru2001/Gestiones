from fastapi import FastAPI, UploadFile, File, HTTPException
from backend.infrastructure.container import build_process_finance_workbook_use_case

app = FastAPI(title="Gestiones Backend", version="0.1.0")


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
    return {"ok": True, "result": result}
