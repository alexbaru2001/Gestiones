from __future__ import annotations

from datetime import date
import json
from pathlib import Path
from typing import Any

from backend.config import get_watchlist_path

# Puntos de histórico que se guardan por ticker. Con una toma al día, dos años de seguimiento.
MAX_HISTORY_POINTS = 730


class WatchlistStorageError(RuntimeError):
    """El archivo de seguimiento existe pero no se puede leer o escribir."""


class JsonWatchlistRepository:
    """Lista de seguimiento persistida en disco.

    Antes cada análisis se perdía al recargar la página: no quedaba ni la lista de candidatos ni
    rastro de cómo evolucionaba su puntuación, que es justo la señal que hace revisar una tesis.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        return self._path or get_watchlist_path()

    def load(self) -> list[dict[str, Any]]:
        entries = self._read()
        return sorted(entries.values(), key=_sort_key)

    def add(self, ticker: str, analysis: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        key = ticker.strip().upper()
        if not key:
            raise ValueError("Indica un ticker para seguir.")
        entries = self._read()
        entry = entries.get(key) or {
            "ticker": key,
            "name": key,
            "added_at": date.today().isoformat(),
            "history": [],
        }
        entries[key] = with_analysis(entry, analysis)
        self._write(entries)
        return sorted(entries.values(), key=_sort_key)

    def remove(self, ticker: str) -> list[dict[str, Any]]:
        entries = self._read()
        entries.pop(ticker.strip().upper(), None)
        self._write(entries)
        return sorted(entries.values(), key=_sort_key)

    def record_analysis(self, analysis: dict[str, Any]) -> None:
        """Anota la puntuación del día, pero solo para tickers que ya se siguen."""
        key = str(analysis.get("ticker") or "").strip().upper()
        entries = self._read()
        if not key or key not in entries:
            return
        entries[key] = with_analysis(entries[key], analysis)
        self._write(entries)

    def _read(self) -> dict[str, dict[str, Any]]:
        path = self.path
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise WatchlistStorageError(f"No se pudo leer la lista de seguimiento: {exc}") from exc
        if not isinstance(payload, dict):
            return {}
        return {str(key).upper(): value for key, value in payload.items() if isinstance(value, dict)}

    def _write(self, entries: dict[str, dict[str, Any]]) -> None:
        path = self.path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as exc:
            raise WatchlistStorageError(f"No se pudo guardar la lista de seguimiento: {exc}") from exc


def with_analysis(entry: dict[str, Any], analysis: dict[str, Any] | None) -> dict[str, Any]:
    """Añade al registro la foto de hoy (score, precio, RPD) sin perder el histórico anterior."""
    updated = dict(entry)
    updated.setdefault("history", [])
    if not analysis:
        return updated

    metrics = analysis.get("metrics") if isinstance(analysis.get("metrics"), dict) else {}
    updated["name"] = str(analysis.get("name") or updated.get("name") or updated["ticker"])
    updated["sector"] = analysis.get("sector")
    updated["currency"] = (analysis.get("exchange") or {}).get("currency")
    updated["recommendation"] = analysis.get("recommendation")
    updated["flags"] = analysis.get("flags") or []

    point = {
        "date": date.today().isoformat(),
        "score": analysis.get("score"),
        "price": analysis.get("price"),
        "rpd_ttm": metrics.get("rpd_ttm"),
        "rpd_avg5": metrics.get("rpd_avg5"),
        "payout": metrics.get("payout"),
        "flags": len(analysis.get("flags") or []),
    }
    # Una sola toma por día: analizar tres veces la misma tarde no debe llenar el histórico.
    history = [item for item in updated["history"] if isinstance(item, dict) and item.get("date") != point["date"]]
    history.append(point)
    updated["history"] = history[-MAX_HISTORY_POINTS:]
    updated["last_score"] = point["score"]
    updated["last_price"] = point["price"]
    updated["last_rpd"] = point["rpd_ttm"]
    updated["last_checked"] = point["date"]
    return updated


def _sort_key(entry: dict[str, Any]) -> tuple[int, float, str]:
    score = entry.get("last_score")
    if isinstance(score, (int, float)):
        return (0, -float(score), str(entry.get("ticker", "")))
    return (1, 0.0, str(entry.get("ticker", "")))
