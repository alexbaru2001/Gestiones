from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.config import get_objectives_path
from backend.domain.models import ObjectiveConfig


class ObjectivesValidationError(ValueError):
    pass


class ObjectivesStorageError(RuntimeError):
    pass


def parse_objectives_json(objectives_json: str | None) -> list[dict[str, Any]]:
    if not objectives_json:
        return []

    try:
        data = json.loads(objectives_json)
    except json.JSONDecodeError as exc:
        raise ObjectivesValidationError("objetivos_json debe ser JSON válido") from exc

    return normalize_objectives_payload(data, detail_prefix="objetivos_json")


def normalize_objectives_payload(data: Any, detail_prefix: str = "objetivos") -> list[dict[str, Any]]:
    if isinstance(data, dict):
        data = data.get("objetivos", [])

    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ObjectivesValidationError(f"{detail_prefix} debe ser una lista de objetivos")

    objectives: list[ObjectiveConfig] = []
    names: set[str] = set()
    total_fraction = 0.0

    for raw in data:
        name = str(raw.get("nombre", "")).strip()
        if not name:
            raise ObjectivesValidationError("Cada objetivo necesita nombre")
        if name in names:
            raise ObjectivesValidationError("Los objetivos deben tener nombres únicos")
        names.add(name)

        tags_raw = raw.get("etiquetas", [])
        if isinstance(tags_raw, str):
            tags = [tag.strip().lower() for tag in tags_raw.split(",") if tag.strip()]
        elif isinstance(tags_raw, list):
            tags = [str(tag).strip().lower() for tag in tags_raw if str(tag).strip()]
        else:
            raise ObjectivesValidationError(f"Objetivo '{name}': etiquetas debe ser texto o lista")

        fraction = float(raw.get("fraccion_presupuesto", 0.0))
        if fraction < 0 or fraction > 1:
            raise ObjectivesValidationError(f"Objetivo '{name}': fraccion_presupuesto fuera de [0,1]")
        total_fraction += fraction

        duration = int(raw.get("duracion_meses", 0))
        if duration <= 0:
            raise ObjectivesValidationError(f"Objetivo '{name}': duracion_meses debe ser > 0")

        start_month = str(raw.get("mes_inicio", "")).strip()[:7]
        if len(start_month) != 7 or start_month[4] != "-":
            raise ObjectivesValidationError(f"Objetivo '{name}': mes_inicio debe tener formato YYYY-MM")

        objectives.append(
            ObjectiveConfig(
                nombre=name,
                etiquetas=tags,
                fraccion_presupuesto=fraction,
                duracion_meses=duration,
                mes_inicio=start_month,
                saldo_inicial=float(raw.get("saldo_inicial", 0.0)),
            )
        )

    if total_fraction > 1 + 1e-9:
        raise ObjectivesValidationError("La suma de fraccion_presupuesto de los objetivos supera 1")

    return [objective.to_dict() for objective in objectives]


class JsonObjectivesRepository:
    def __init__(self, path: Path | None = None) -> None:
        self._path = path or get_objectives_path()

    def load(self) -> list[dict[str, Any]]:
        if not self._path.exists():
            return []

        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ObjectivesStorageError("El archivo de objetivos no contiene JSON válido") from exc

        return normalize_objectives_payload(data)

    def save(self, payload: Any) -> list[dict[str, Any]]:
        objectives = normalize_objectives_payload(payload)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps({"objetivos": objectives}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return objectives
