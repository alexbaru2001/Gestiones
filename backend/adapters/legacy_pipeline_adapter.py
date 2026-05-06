from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any


class LegacyPipelineAdapter:
    """Adapter para reutilizar el pipeline actual sin romper comportamiento."""

    def __init__(self) -> None:
        self._legacy_path = Path(__file__).resolve().parents[2] / "Personal_finanzas"

    def run_pipeline_from_bytes(self, excel_bytes: bytes, params: dict, objetivos: list[dict] | None = None) -> dict[str, Any]:
        legacy_path_str = str(self._legacy_path)
        if legacy_path_str not in sys.path:
            sys.path.append(legacy_path_str)

        from pipeline import run_pipeline, PipelineParams  # type: ignore

        parsed = PipelineParams(**params)
        result = run_pipeline(
            excel_source=io.BytesIO(excel_bytes),
            params=parsed,
            objetivos_vista=objetivos or [],
            fondo_reserva_snapshot=None,
            output_path=None,
        )
        return result
