from typing import Protocol, Any
from backend.domain.models import PipelineConfig


class FinanceProcessingPort(Protocol):
    def run_pipeline_from_bytes(
        self,
        excel_bytes: bytes,
        params: PipelineConfig,
        objetivos: list[dict] | None = None,
    ) -> dict[str, Any]:
        ...
