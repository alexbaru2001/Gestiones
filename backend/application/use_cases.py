from backend.domain.models import PipelineConfig
from backend.domain.results import PipelineResult
from backend.ports.finance_port import FinanceProcessingPort


class ProcessFinanceWorkbookUseCase:
    def __init__(self, port: FinanceProcessingPort):
        self._port = port

    def execute(
        self,
        excel_bytes: bytes,
        params: PipelineConfig,
        objetivos: list[dict] | None = None,
        checkpoint: dict | None = None,
    ) -> PipelineResult:
        return self._port.run_pipeline_from_bytes(
            excel_bytes=excel_bytes, params=params, objetivos=objetivos or [], checkpoint=checkpoint
        )
