from typing import Any
from backend.ports.finance_port import FinanceProcessingPort


class ProcessFinanceWorkbookUseCase:
    def __init__(self, port: FinanceProcessingPort):
        self._port = port

    def execute(self, excel_bytes: bytes, params: dict, objetivos: list[dict] | None = None) -> dict[str, Any]:
        return self._port.run_pipeline_from_bytes(excel_bytes=excel_bytes, params=params, objetivos=objetivos or [])
