from typing import Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.config import get_cors_origins
from backend.domain.investments import analyze_ticker, compare_tickers, search_tickers
from backend.domain.exploration import DOWNLOAD_CHUNK, collect_candidates, prescore, screen_chunk
from backend.domain.investment_contributions import build_investment_contributions, merge_income_rows
from backend.domain.fx import convert as convert_to_eur, get_rates
from backend.domain.market_prices import get_prices, key as price_key
from backend.domain.models import PipelineConfig
from backend.domain.myinvestor_movements import extract_contributions, extract_fees
from backend.domain.asset_catalog import known_metadata, known_tickers
from backend.domain.portfolio_positions import (
    accumulate_costs,
    allocate_fees,
    classify,
    derive_movements,
    months_of,
    position_value,
    summarize_load,
)
from backend.adapters.legacy_pipeline_adapter import read_workbook_first_month
from backend.domain.universes import list_universes, refresh_members
from backend.domain.portfolio import FUND_NAME_MAP, UploadedInvestmentFile
from backend.infrastructure.container import build_process_finance_workbook_use_case
from backend.infrastructure.finance_checkpoint_repository import JsonFinanceCheckpointRepository
from backend.infrastructure.finance_history_repository import CsvFinanceHistoryRepository
from backend.infrastructure.objectives_repository import (
    JsonObjectivesRepository,
    ObjectivesStorageError,
    ObjectivesValidationError,
    parse_objectives_json,
)
from backend.infrastructure.portfolio_positions_repository import (
    ExcelPortfolioPositionsRepository,
    PositionsStorageError,
)
from backend.infrastructure.portfolio_repository import LocalPortfolioRepository
from backend.infrastructure.transaction_history_repository import CsvTransactionHistoryRepository
from backend.infrastructure.exploration_repository import JsonExplorationRepository
from backend.infrastructure.portfolio_review_repository import JsonPortfolioReviewRepository
from backend.infrastructure.watchlist_repository import JsonWatchlistRepository, WatchlistStorageError

app = FastAPI(title="Gestiones Backend", version="0.1.0")
objectives_repository = JsonObjectivesRepository()
portfolio_repository = LocalPortfolioRepository()
portfolio_positions_repository = ExcelPortfolioPositionsRepository()
finance_history_repository = CsvFinanceHistoryRepository()
finance_checkpoint_repository = JsonFinanceCheckpointRepository()
transaction_history_repository = CsvTransactionHistoryRepository()
watchlist_repository = JsonWatchlistRepository()
portfolio_review_repository = JsonPortfolioReviewRepository()
exploration_repository = JsonExplorationRepository()

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
def analyze_investment(ticker: str, refresh: bool = False) -> dict[str, Any]:
    try:
        result = analyze_ticker(ticker, refresh=refresh)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    # Si el ticker está en seguimiento, cada análisis deja su punto en el histórico de puntuación.
    try:
        watchlist_repository.record_analysis(result)
    except WatchlistStorageError:
        pass
    return {"ok": True, "result": result}


@app.get("/api/v1/investments/search")
def search_investments(q: str) -> dict[str, Any]:
    try:
        return {"ok": True, "result": search_tickers(q)}
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/api/v1/investments/compare")
def compare_investments(tickers: str, refresh: bool = False) -> dict[str, Any]:
    requested = [item for item in tickers.split(",") if item.strip()]
    try:
        return {"ok": True, "result": compare_tickers(requested, refresh=refresh)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/v1/investments/watchlist")
def get_watchlist() -> dict[str, Any]:
    try:
        return {"ok": True, "result": watchlist_repository.load()}
    except WatchlistStorageError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/v1/investments/watchlist")
def add_to_watchlist(payload: dict[str, Any]) -> dict[str, Any]:
    ticker = payload.get("ticker")
    if not isinstance(ticker, str) or not ticker.strip():
        raise HTTPException(status_code=400, detail="Indica un ticker para seguir.")
    analysis = payload.get("analysis") if isinstance(payload.get("analysis"), dict) else None
    try:
        return {"ok": True, "result": watchlist_repository.add(ticker, analysis)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except WatchlistStorageError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.delete("/api/v1/investments/watchlist/{ticker}")
def remove_from_watchlist(ticker: str) -> dict[str, Any]:
    try:
        return {"ok": True, "result": watchlist_repository.remove(ticker)}
    except WatchlistStorageError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/v1/investments/portfolio-review")
def get_portfolio_review() -> dict[str, Any]:
    try:
        return {"ok": True, "result": portfolio_review_repository.load()}
    except WatchlistStorageError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/v1/investments/portfolio-review")
def record_portfolio_review(payload: dict[str, Any]) -> dict[str, Any]:
    """Anota una posición revisada. El frontend llama una vez por ticker para poder ir informando
    del avance: una sola petición para toda la cartera dejaría la pantalla muerta medio minuto."""
    ticker = payload.get("ticker")
    analysis = payload.get("analysis")
    if not isinstance(ticker, str) or not ticker.strip():
        raise HTTPException(status_code=400, detail="Indica un ticker para revisar.")
    if not isinstance(analysis, dict):
        raise HTTPException(status_code=400, detail="Falta el análisis de la posición.")
    position = payload.get("position") if isinstance(payload.get("position"), dict) else None
    try:
        return {"ok": True, "result": portfolio_review_repository.record(ticker, analysis, position)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except WatchlistStorageError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/v1/investments/portfolio-review/prune")
def prune_portfolio_review(payload: dict[str, Any]) -> dict[str, Any]:
    tickers = payload.get("tickers")
    if not isinstance(tickers, list):
        raise HTTPException(status_code=400, detail="Indica los tickers que siguen en cartera.")
    try:
        return {"ok": True, "result": portfolio_review_repository.forget_missing([str(item) for item in tickers])}
    except WatchlistStorageError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/v1/investments/universes")
def get_universes() -> dict[str, Any]:
    try:
        saved = exploration_repository.load_all()
    except WatchlistStorageError:
        saved = {}
    return {"ok": True, "result": {"universes": list_universes(), "explorations": saved, "chunk": DOWNLOAD_CHUNK}}


@app.post("/api/v1/investments/universes/{universe_key}/refresh")
def refresh_universe(universe_key: str) -> dict[str, Any]:
    try:
        members = refresh_members(universe_key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"ok": True, "result": {"universe": universe_key, "size": len(members)}}


@app.post("/api/v1/investments/explore/candidates")
def explore_candidates(payload: dict[str, Any]) -> dict[str, Any]:
    universe_key = payload.get("universe")
    if not isinstance(universe_key, str) or not universe_key.strip():
        raise HTTPException(status_code=400, detail="Indica qué universo quieres explorar.")
    try:
        candidates = collect_candidates(universe_key.strip())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"ok": True, "result": {"candidates": candidates, "chunk": DOWNLOAD_CHUNK}}


@app.post("/api/v1/investments/explore/screen")
def explore_screen(payload: dict[str, Any]) -> dict[str, Any]:
    """Criba de un grupo de tickers. El frontend va llamando por tandas para poder ir informando
    del avance en vez de dejar la pantalla muerta durante media descarga del índice entero."""
    tickers = payload.get("tickers")
    if not isinstance(tickers, list) or not tickers:
        raise HTTPException(status_code=400, detail="Indica los tickers de esta tanda.")
    if len(tickers) > DOWNLOAD_CHUNK:
        raise HTTPException(status_code=400, detail=f"Como mucho {DOWNLOAD_CHUNK} tickers por tanda.")
    try:
        rows = screen_chunk([str(item).strip().upper() for item in tickers if str(item).strip()])
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"ok": True, "result": [{**row, "prescore": prescore(row)} for row in rows]}


@app.post("/api/v1/investments/explore/save")
def save_exploration(payload: dict[str, Any]) -> dict[str, Any]:
    universe_key = payload.get("universe")
    results = payload.get("results")
    if not isinstance(universe_key, str) or not universe_key.strip():
        raise HTTPException(status_code=400, detail="Indica qué universo se ha explorado.")
    if not isinstance(results, list):
        raise HTTPException(status_code=400, detail="Faltan los resultados de la exploración.")
    meta = {
        "screened": payload.get("screened"),
        "analyzed": payload.get("analyzed"),
        "approximate": bool(payload.get("approximate")),
        "label": payload.get("label"),
    }
    try:
        return {"ok": True, "result": exploration_repository.save(universe_key.strip(), results, meta)}
    except WatchlistStorageError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/v1/portfolio")
def get_portfolio() -> dict[str, Any]:
    snapshot = portfolio_repository.load()
    return {"ok": True, "result": snapshot}


@app.get("/api/v1/portfolio/positions")
def get_portfolio_positions() -> dict[str, Any]:
    try:
        positions = portfolio_positions_repository.load_positions()
        movements_rows = portfolio_positions_repository.load_movements()
    except PositionsStorageError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"ok": True, "result": _build_positions_result(positions, movements_rows)}


@app.post("/api/v1/portfolio/positions")
async def upload_portfolio_positions(
    posiciones: UploadFile = File(...),
    movimientos: UploadFile | None = File(default=None),
) -> dict[str, Any]:
    """Carga incremental: lo que llega se combina con lo guardado, no lo sustituye."""
    if not (posiciones.filename or "").lower().endswith((".xlsx", ".xlsm")):
        raise HTTPException(status_code=400, detail="El archivo de posiciones debe ser Excel (.xlsx).")

    try:
        stored_positions = portfolio_positions_repository.load_positions()
        stored_movements = portfolio_positions_repository.load_movements()
        incoming = ExcelPortfolioPositionsRepository.parse_positions_upload(await posiciones.read())
        merged_positions = portfolio_positions_repository.save_positions(incoming)

        merged_movements = stored_movements
        if movimientos is not None and movimientos.filename:
            incoming_movements = ExcelPortfolioPositionsRepository.parse_movements_upload(await movimientos.read())
            merged_movements = portfolio_positions_repository.save_movements(incoming_movements)
    except PositionsStorageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "ok": True,
        "resumen": summarize_load(stored_positions, merged_positions, stored_movements, merged_movements, incoming),
        "result": _build_positions_result(merged_positions, merged_movements),
    }


def _build_positions_result(
    positions: list[dict[str, Any]], movements_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Lo que la pantalla necesita: posiciones valoradas, movimientos deducidos y comisiones imputadas."""
    fees = extract_fees(movements_rows)

    # Solo se busca fuera lo que el archivo no resuelve: el resto no gasta ninguna consulta.
    unresolved = [position for position in positions if position_value(position) is None]
    rates = get_rates(
        {
            (position.get("Moneda", "EUR"), position["Mes"])
            for position in unresolved
            if str(position.get("Moneda", "EUR")).upper() != "EUR"
        }
    )
    # Las que además no traen precio (los fondos) necesitan la cotización del mes.
    market = get_prices(
        {(position["ISIN"], position["Mes"]) for position in unresolved if not position.get("Precio cierre")}
    )

    # La clasificación sale del mismo catálogo que la vista anterior, sin red: abrir Cartera no
    # puede depender de que Yahoo responda.
    catalog = known_metadata()
    tickers = known_tickers()
    fund_isins = set(FUND_NAME_MAP.values())

    valued = []
    for position in positions:
        value = position_value(position)
        source = "archivo"
        if value is None:
            price = position.get("Precio cierre") or market.get(price_key(position["ISIN"], position["Mes"]))
            if price:
                amount = float(position.get("Cantidad") or 0) * float(price)
                value = convert_to_eur(amount, position.get("Moneda"), position["Mes"], rates)
                source = "mercado" if not position.get("Precio cierre") else "cambio"
        valued.append(
            {
                **position,
                **classify(position, catalog, fund_isins),
                "ticker": tickers.get(position["ISIN"]),
                "valor": value,
                "origen_valor": source if value is not None else "sin valorar",
            }
        )

    # El coste necesita un precio para cada compra: el del archivo cuando está, y si no el de
    # mercado, que es el caso de los fondos.
    costs = accumulate_costs(positions, price_of=lambda isin, month: market.get(price_key(isin, month)))
    for position in valued:
        position["coste"] = costs.get((position["Mes"], position["Broker"], position["ISIN"]))

    # El reparto a prorrata se hace sobre el valor ya convertido: si no, una posición en divisa
    # quedaría fuera del reparto y las demás cargarían con su parte de la comisión.
    allocated = allocate_fees(
        [{**position, "Valor EUR": position["valor"]} for position in valued], fees
    )
    for position in valued:
        position["comision_imputada"] = allocated.get((position["Mes"], position["Broker"], position["ISIN"]))
    return {
        "posiciones": valued,
        "meses": months_of(positions),
        "movimientos": derive_movements(positions),
        "comisiones": fees,
        "aportaciones": extract_contributions(movements_rows),
    }


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

    # El tramo que se va a calcular empieza donde empiecen a la vez el parámetro y los datos: pedir
    # desde 2024-10 un Excel que solo trae 2026 no procesa 2024.
    workbook_start = read_workbook_first_month(content)
    tramo_inicio = max(fecha_inicio_month, workbook_start) if workbook_start else fecha_inicio_month

    recalculado_desde_cero = bool(checkpoint_as_of and tramo_inicio <= checkpoint_as_of)
    if recalculado_desde_cero:
        # Repetir o solapar el tramo ya cerrado no necesita continuar los acumuladores desde el
        # checkpoint: se recalcula igual que si nunca hubiera existido. El paso de guardado más abajo
        # ya se encarga de no duplicar los meses que ya estén en el histórico (o de sincronizar el
        # checkpoint si el Excel completo coincide con lo ya guardado, sin escribir nada nuevo).
        #
        # Lo que NO puede pasar es llegar aquí porque `fecha_inicio` venga con su valor por defecto:
        # entonces se tiraba el checkpoint y se recalculaba desde cero con un Excel que solo tiene
        # el año en curso, y los acumulados se desplomaban en su primer mes.
        checkpoint = None

    historical_transactions = {
        "gastos": transaction_history_repository.load_gastos(),
        "ingresos": transaction_history_repository.load_ingresos(),
    }

    checkpoint = _seed_reserve_average(checkpoint, finance_history_repository.load())

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

    # El desglose de la bolsa de inversión se calcula aquí y no en el adaptador porque necesita el
    # histórico completo (display_rows), que solo existe una vez combinado lo guardado con el tramo
    # recién calculado; en el adaptador solo se ve el tramo y la gráfica saldría con un único mes.
    result_dict.setdefault("analisis", {})["aporte_inversion"] = build_investment_contributions(
        display_rows,
        merge_income_rows(historical_transactions.get("ingresos"), (result.transacciones or {}).get("ingresos")),
        config.porcentaje_inversion,
    )

    return {
        "ok": True,
        "result": result_dict,
        "modo": modo,
        "meses_nuevos": meses_nuevos,
        "meses_ya_guardados": meses_ya_guardados,
        "recalculado_desde_cero": recalculado_desde_cero,
    }


def _seed_reserve_average(checkpoint: dict[str, Any] | None, stored_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Rellena la media de gasto del fondo en checkpoints guardados antes de que existiera.

    El objetivo del fondo son seis meses de gasto medio, y esa media se arrastra de tramo en tramo
    dentro del checkpoint. Un checkpoint escrito por una versión anterior no la trae, y sin ella el
    siguiente tramo calcularía la media solo con los meses del Excel que se suba, olvidando todo lo
    vivido antes. Aquí se reconstruye una única vez a partir del histórico ya guardado.
    """
    if not checkpoint:
        return checkpoint
    snapshot = dict(checkpoint.get("fondo_reserva_snapshot") or {})
    if snapshot.get("Meses contados"):
        return checkpoint
    hasta = checkpoint.get("as_of_month")
    gastos = [
        float(row.get("💳 Gasto del mes") or 0.0)
        for row in stored_rows
        if row.get("Mes") and (not hasta or str(row["Mes"]) <= str(hasta))
    ]
    if not gastos:
        return checkpoint
    snapshot["Gasto acumulado"] = sum(gastos)
    snapshot["Meses contados"] = len(gastos)
    return {**checkpoint, "fondo_reserva_snapshot": snapshot}


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
