from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil
from typing import Any

import pandas as pd

from backend.config import get_movements_path, get_positions_path
from backend.domain.myinvestor_movements import HEADER_ROW as MOVEMENTS_HEADER_ROW
from backend.domain.portfolio_positions import COLUMNS, merge_movements, merge_positions, read_positions


class PositionsStorageError(RuntimeError):
    """El archivo de posiciones existe pero no se puede leer o escribir."""


class ExcelPortfolioPositionsRepository:
    """Guarda el archivo de posiciones que mantiene el usuario y el extracto de movimientos.

    Cada carga es incremental: lo que llega se combina con lo guardado en vez de sustituirlo. Antes
    de escribir se aparta una copia de la versión vigente, igual que se hace con los originales de
    bróker en `raw/<fecha>/`, para poder volver atrás si una carga mete datos malos.
    """

    def __init__(self, positions_path: Path | None = None, movements_path: Path | None = None) -> None:
        self._positions_path = positions_path
        self._movements_path = movements_path

    @property
    def positions_path(self) -> Path:
        return self._positions_path or get_positions_path()

    @property
    def movements_path(self) -> Path:
        return self._movements_path or get_movements_path()

    @property
    def versions_dir(self) -> Path:
        return self.positions_path.parent / "versiones"

    # --- Lectura ---------------------------------------------------------------------------------

    def load_positions(self) -> list[dict[str, Any]]:
        return read_positions(self._read_excel(self.positions_path))

    def load_movements(self) -> list[dict[str, Any]]:
        return self._read_excel(self.movements_path)

    # --- Escritura -------------------------------------------------------------------------------

    def save_positions(self, incoming: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged = merge_positions(self.load_positions(), incoming)
        self._archive(self.positions_path)
        self._write_excel(self.positions_path, merged, list(COLUMNS))
        return merged

    def save_movements(self, incoming: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged = merge_movements(self.load_movements(), incoming)
        self._archive(self.movements_path)
        self._write_excel(self.movements_path, merged, None)
        return merged

    # --- Lectura de los archivos que sube el usuario ----------------------------------------------

    @staticmethod
    def parse_positions_upload(content: bytes) -> list[dict[str, Any]]:
        from io import BytesIO

        try:
            frame = pd.read_excel(BytesIO(content))
        except Exception as exc:
            raise PositionsStorageError(f"No se pudo leer el archivo de posiciones: {exc}") from exc
        missing = {"Mes", "Broker", "ISIN"} - set(frame.columns)
        if missing:
            raise PositionsStorageError(
                f"Al archivo de posiciones le faltan columnas: {', '.join(sorted(missing))}."
            )
        return frame.to_dict("records")

    @staticmethod
    def parse_movements_upload(content: bytes) -> list[dict[str, Any]]:
        """El extracto de MyInvestor lleva la cabecera en la fila 10; las de arriba son titular y saldo."""
        from io import BytesIO

        try:
            frame = pd.read_excel(BytesIO(content), header=MOVEMENTS_HEADER_ROW)
        except Exception as exc:
            raise PositionsStorageError(f"No se pudo leer el archivo de movimientos: {exc}") from exc
        if "Movimiento" not in frame.columns:
            # Puede venir ya recortado, sin las filas de cabecera del extracto original.
            try:
                frame = pd.read_excel(BytesIO(content))
            except Exception as exc:
                raise PositionsStorageError(f"No se pudo leer el archivo de movimientos: {exc}") from exc
        if "Movimiento" not in frame.columns:
            raise PositionsStorageError("El archivo de movimientos no tiene una columna 'Movimiento'.")
        frame = frame.dropna(subset=["Movimiento"])
        # Las fechas se guardan como texto para que la clave de deduplicación sea estable entre
        # cargas: pandas las convierte a Timestamp y su representación cambia según el archivo.
        for column in frame.columns:
            if "Fecha" in str(column):
                frame[column] = frame[column].astype(str).str.slice(0, 10)
        return frame.to_dict("records")

    # --- Interno ---------------------------------------------------------------------------------

    def _read_excel(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        try:
            return pd.read_excel(path).to_dict("records")
        except Exception as exc:
            raise PositionsStorageError(f"No se pudo leer {path.name}: {exc}") from exc

    def _write_excel(self, path: Path, rows: list[dict[str, Any]], columns: list[str] | None) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            frame = pd.DataFrame(rows, columns=columns) if columns else pd.DataFrame(rows)
            frame.to_excel(path, index=False)
        except Exception as exc:
            raise PositionsStorageError(f"No se pudo guardar {path.name}: {exc}") from exc

    def _archive(self, path: Path) -> None:
        if not path.exists():
            return
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        destination = self.versions_dir / stamp
        try:
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination / path.name)
        except OSError:
            # Perder la copia de respaldo no debe impedir guardar la carga nueva.
            return
