from __future__ import annotations

from datetime import date
import json
from pathlib import Path
from typing import Any

from backend.config import get_portfolio_review_path
from backend.infrastructure.watchlist_repository import WatchlistStorageError, with_analysis


class JsonPortfolioReviewRepository:
    """Revisión periódica de las posiciones reales de la cartera.

    Vive en su propio archivo, separada del seguimiento manual: una cosa son las empresas que se
    vigilan por curiosidad y otra las que ya se tienen compradas, que se revisan en bloque.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        return self._path or get_portfolio_review_path()

    def load(self) -> dict[str, Any]:
        payload = self._read()
        positions = sorted(payload["positions"].values(), key=_sort_key)
        return {"positions": positions, "last_review": payload.get("last_review")}

    def record(self, ticker: str, analysis: dict[str, Any], position: dict[str, Any] | None = None) -> dict[str, Any]:
        key = str(ticker or "").strip().upper()
        if not key:
            raise ValueError("Indica un ticker para revisar.")
        payload = self._read()
        entry = payload["positions"].get(key) or {"ticker": key, "name": key, "history": []}
        updated = with_analysis(entry, analysis)
        if position:
            # El peso viene de la cartera, no del análisis: hace falta para ponderar la nota media.
            updated["weight"] = _number(position.get("weight"))
            updated["value"] = _number(position.get("value"))
            updated["broker"] = position.get("broker")
            updated["portfolio_name"] = position.get("name") or updated.get("name")
        payload["positions"][key] = updated
        payload["last_review"] = date.today().isoformat()
        self._write(payload)
        return self.load()

    def forget_missing(self, tickers: list[str]) -> dict[str, Any]:
        """Quita de la revisión lo que ya no está en la cartera (una posición vendida)."""
        keep = {str(ticker).strip().upper() for ticker in tickers if str(ticker).strip()}
        payload = self._read()
        payload["positions"] = {key: value for key, value in payload["positions"].items() if key in keep}
        self._write(payload)
        return self.load()

    def _read(self) -> dict[str, Any]:
        path = self.path
        if not path.exists():
            return {"positions": {}, "last_review": None}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise WatchlistStorageError(f"No se pudo leer la revisión de cartera: {exc}") from exc
        if not isinstance(payload, dict):
            return {"positions": {}, "last_review": None}
        positions = payload.get("positions")
        if not isinstance(positions, dict):
            positions = {}
        return {
            "positions": {str(key).upper(): value for key, value in positions.items() if isinstance(value, dict)},
            "last_review": payload.get("last_review"),
        }

    def _write(self, payload: dict[str, Any]) -> None:
        path = self.path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as exc:
            raise WatchlistStorageError(f"No se pudo guardar la revisión de cartera: {exc}") from exc


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def _sort_key(entry: dict[str, Any]) -> tuple[int, float, str]:
    """Lo peor primero: la revisión sirve para mirar lo que flojea, no para felicitarse."""
    score = entry.get("last_score")
    if isinstance(score, (int, float)):
        return (0, float(score), str(entry.get("ticker", "")))
    return (1, 0.0, str(entry.get("ticker", "")))
