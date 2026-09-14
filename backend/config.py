from __future__ import annotations

import os
from pathlib import Path

DEFAULT_CORS_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")
DEFAULT_OBJECTIVES_PATH = Path(__file__).resolve().parents[1] / "Personal_finanzas" / "Data" / "objetivos_vista.json"
DEFAULT_INVESTMENTS_PATH = Path(__file__).resolve().parents[1] / "Personal_finanzas" / "Data" / "Inversiones"
DEFAULT_FINANCE_HISTORY_PATH = Path(__file__).resolve().parents[1] / "Personal_finanzas" / "Data" / "historial.csv"
DEFAULT_FINANCE_CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[1] / "Personal_finanzas" / "Data" / "finance_checkpoint.json"
)
DEFAULT_GASTOS_HISTORY_PATH = Path(__file__).resolve().parents[1] / "Personal_finanzas" / "Data" / "gastos_historico.csv"
DEFAULT_INGRESOS_HISTORY_PATH = (
    Path(__file__).resolve().parents[1] / "Personal_finanzas" / "Data" / "ingresos_historico.csv"
)
DEFAULT_INVESTMENT_KNOWLEDGE_PATH = DEFAULT_INVESTMENTS_PATH / "knowledge"
DEFAULT_ASSET_CATALOG_PATH = DEFAULT_INVESTMENTS_PATH / "asset_catalog.json"
DEFAULT_WATCHLIST_PATH = DEFAULT_INVESTMENTS_PATH / "watchlist.json"
DEFAULT_PORTFOLIO_REVIEW_PATH = DEFAULT_INVESTMENTS_PATH / "portfolio_review.json"
DEFAULT_UNIVERSES_PATH = DEFAULT_INVESTMENTS_PATH / "universes.json"
DEFAULT_EXPLORATIONS_PATH = DEFAULT_INVESTMENTS_PATH / "exploraciones.json"
DEFAULT_ANALYSIS_CACHE_PATH = DEFAULT_INVESTMENTS_PATH / "analysis_cache"


def get_cors_origins() -> list[str]:
    origins = os.getenv("GESTIONES_CORS_ORIGINS", "")
    if not origins.strip():
        return list(DEFAULT_CORS_ORIGINS)
    return [origin.strip() for origin in origins.split(",") if origin.strip()]


def get_objectives_path() -> Path:
    configured_path = os.getenv("GESTIONES_OBJECTIVES_PATH", "")
    if not configured_path.strip():
        return DEFAULT_OBJECTIVES_PATH
    return Path(configured_path).expanduser()


def get_investments_path() -> Path:
    configured_path = os.getenv("GESTIONES_INVESTMENTS_PATH", "")
    if not configured_path.strip():
        return DEFAULT_INVESTMENTS_PATH
    return Path(configured_path).expanduser()


def get_finance_history_path() -> Path:
    configured_path = os.getenv("GESTIONES_FINANCE_HISTORY_PATH", "")
    if not configured_path.strip():
        return DEFAULT_FINANCE_HISTORY_PATH
    return Path(configured_path).expanduser()


def get_finance_checkpoint_path() -> Path:
    configured_path = os.getenv("GESTIONES_FINANCE_CHECKPOINT_PATH", "")
    if not configured_path.strip():
        return DEFAULT_FINANCE_CHECKPOINT_PATH
    return Path(configured_path).expanduser()


def get_gastos_history_path() -> Path:
    configured_path = os.getenv("GESTIONES_GASTOS_HISTORY_PATH", "")
    if not configured_path.strip():
        return DEFAULT_GASTOS_HISTORY_PATH
    return Path(configured_path).expanduser()


def get_ingresos_history_path() -> Path:
    configured_path = os.getenv("GESTIONES_INGRESOS_HISTORY_PATH", "")
    if not configured_path.strip():
        return DEFAULT_INGRESOS_HISTORY_PATH
    return Path(configured_path).expanduser()


def get_asset_catalog_path() -> Path:
    configured_path = os.getenv("GESTIONES_ASSET_CATALOG_PATH", "")
    if not configured_path.strip():
        return DEFAULT_ASSET_CATALOG_PATH
    return Path(configured_path).expanduser()


def get_watchlist_path() -> Path:
    configured_path = os.getenv("GESTIONES_WATCHLIST_PATH", "")
    if not configured_path.strip():
        return DEFAULT_WATCHLIST_PATH
    return Path(configured_path).expanduser()


def get_portfolio_review_path() -> Path:
    configured_path = os.getenv("GESTIONES_PORTFOLIO_REVIEW_PATH", "")
    if not configured_path.strip():
        return DEFAULT_PORTFOLIO_REVIEW_PATH
    return Path(configured_path).expanduser()


def get_universes_path() -> Path:
    configured_path = os.getenv("GESTIONES_UNIVERSES_PATH", "")
    if not configured_path.strip():
        return DEFAULT_UNIVERSES_PATH
    return Path(configured_path).expanduser()


def get_explorations_path() -> Path:
    configured_path = os.getenv("GESTIONES_EXPLORATIONS_PATH", "")
    if not configured_path.strip():
        return DEFAULT_EXPLORATIONS_PATH
    return Path(configured_path).expanduser()


def get_analysis_cache_path() -> Path:
    configured_path = os.getenv("GESTIONES_ANALYSIS_CACHE_PATH", "")
    if not configured_path.strip():
        return DEFAULT_ANALYSIS_CACHE_PATH
    return Path(configured_path).expanduser()


def get_investment_knowledge_path() -> Path:
    configured_path = os.getenv("GESTIONES_INVESTMENT_KNOWLEDGE_PATH", "")
    if not configured_path.strip():
        return DEFAULT_INVESTMENT_KNOWLEDGE_PATH
    return Path(configured_path).expanduser()
