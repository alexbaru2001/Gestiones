from typing import Protocol
from backend.domain.models import PipelineConfig
from backend.domain.results import PipelineResult


class FinanceProcessingPort(Protocol):
    def run_pipeline_from_bytes(
        self,
        excel_bytes: bytes,
        params: PipelineConfig,
        objetivos: list[dict] | None = None,
        checkpoint: dict | None = None,
        historical_transactions: dict | None = None,
    ) -> PipelineResult:
        ...
