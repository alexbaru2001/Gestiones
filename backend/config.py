from __future__ import annotations

import os
from pathlib import Path

DEFAULT_CORS_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")
DEFAULT_OBJECTIVES_PATH = Path(__file__).resolve().parents[1] / "Personal_finanzas" / "Data" / "objetivos_vista.json"
DEFAULT_INVESTMENTS_PATH = Path(__file__).resolve().parents[1] / "Personal_finanzas" / "Data" / "Inversiones"
DEFAULT_FINANCE_HISTORY_PATH = Path(__file__).resolve().parents[1] / "Personal_finanzas" / "Data" / "historial.csv"
DEFAULT_INVESTMENT_KNOWLEDGE_PATH = DEFAULT_INVESTMENTS_PATH / "knowledge"


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


def get_investment_knowledge_path() -> Path:
    configured_path = os.getenv("GESTIONES_INVESTMENT_KNOWLEDGE_PATH", "")
    if not configured_path.strip():
        return DEFAULT_INVESTMENT_KNOWLEDGE_PATH
    return Path(configured_path).expanduser()
