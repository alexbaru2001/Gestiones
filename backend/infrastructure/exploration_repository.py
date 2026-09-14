from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from backend.config import get_explorations_path
from backend.infrastructure.watchlist_repository import WatchlistStorageError

# Exploraciones guardadas. Una por universo: interesa la última foto de cada índice, no un archivo
# histórico de cada vez que se pulsó el botón.
MAX_RESULTS_STORED = 40


class JsonExplorationRepository:
    """Guarda el resultado de cada exploración para poder volver a abrirlo sin reanalizar."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        return self._path or get_explorations_path()

    def load_all(self) -> dict[str, Any]:
        return self._read()

    def load(self, universe_key: str) -> dict[str, Any] | None:
        return self._read().get(universe_key)

    def save(self, universe_key: str, results: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
        payload = self._read()
        payload[universe_key] = {
            "universe": universe_key,
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "results": results[:MAX_RESULTS_STORED],
            **meta,
        }
        self._write(payload)
        return payload[universe_key]

    def _read(self) -> dict[str, Any]:
        path = self.path
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise WatchlistStorageError(f"No se pudo leer las exploraciones guardadas: {exc}") from exc
        return payload if isinstance(payload, dict) else {}

    def _write(self, payload: dict[str, Any]) -> None:
        path = self.path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        except OSError as exc:
            raise WatchlistStorageError(f"No se pudo guardar la exploración: {exc}") from exc
