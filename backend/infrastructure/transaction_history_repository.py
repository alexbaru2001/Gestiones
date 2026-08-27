from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from backend.config import get_gastos_history_path, get_ingresos_history_path


class CsvTransactionHistoryRepository:
    """Guarda, tramo a tramo, las filas de Gastos/Ingresos ya clasificadas (con tipo_logico) que se
    han añadido al histórico. A diferencia de historial.csv (un resumen mensual), esto conserva el
    detalle de cada movimiento para que los desgloses por categoría (Intereses, Gastos por tipo,
    Dividendos por empresa...) puedan calcularse sobre todo el histórico y no solo sobre el Excel que
    se subió en la última vez, que puede traer únicamente el tramo más reciente."""

    def __init__(self, gastos_path: Path | None = None, ingresos_path: Path | None = None):
        self.gastos_path = gastos_path or get_gastos_history_path()
        self.ingresos_path = ingresos_path or get_ingresos_history_path()

    def load_gastos(self) -> list[dict[str, Any]]:
        return _load(self.gastos_path)

    def load_ingresos(self) -> list[dict[str, Any]]:
        return _load(self.ingresos_path)

    def append_gastos(self, rows: list[dict[str, Any]]) -> None:
        _append(self.gastos_path, rows)

    def append_ingresos(self, rows: list[dict[str, Any]]) -> None:
        _append(self.ingresos_path, rows)

    def delete(self) -> None:
        self.gastos_path.unlink(missing_ok=True)
        self.ingresos_path.unlink(missing_ok=True)


def _load(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _append(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    existing = _load(path)
    columns = _ordered_columns(existing + rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows({column: record.get(column, "") for column in columns} for record in existing + rows)


def _ordered_columns(records: list[dict[str, Any]]) -> list[str]:
    preferred = ["Mes", "fecha", "categoria", "cuenta", "cantidad", "etiquetas", "comentario", "tipo_logico"]
    seen = set()
    columns = []
    for column in preferred:
        if any(column in record for record in records):
            columns.append(column)
            seen.add(column)
    for record in records:
        for column in record:
            if column not in seen:
                columns.append(column)
                seen.add(column)
    return columns
