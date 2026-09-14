from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from backend.config import get_finance_history_path


class CsvFinanceHistoryRepository:
    def __init__(self, path: Path | None = None):
        self.path = path or get_finance_history_path()

    def load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8-sig", newline="") as file:
            return [_with_numeric_values(record) for record in csv.DictReader(file)]

    def save(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        columns = ordered_columns(records)
        with self.path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows({column: record.get(column, "") for column in columns} for record in records)

    def delete(self) -> None:
        self.path.unlink(missing_ok=True)


def _with_numeric_values(record: dict[str, Any]) -> dict[str, Any]:
    """El CSV devuelve todo como texto ('6205.88'), pero el resto de la app espera números nativos:
    el frontend formatea con formatMoney, que muestra '-' ante cualquier valor que no sea number, así
    que un mes leído del histórico saldría vacío en pantalla. "Mes" se mantiene como texto."""
    converted: dict[str, Any] = {}
    for key, value in record.items():
        if key == "Mes" or not isinstance(value, str):
            converted[key] = value
            continue
        text = value.strip()
        if not text:
            converted[key] = None
            continue
        try:
            converted[key] = float(text)
        except ValueError:
            converted[key] = value
    return converted


def ordered_columns(records: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "Mes",
        "Regalos",
        "Vacaciones",
        "Inversiones",
        "Dinero Invertido",
        "Ahorros",
        "Fondo de reserva cargado",
        "total",
    ]
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
