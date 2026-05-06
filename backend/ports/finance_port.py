from typing import Protocol, Any


class FinanceProcessingPort(Protocol):
    def run_pipeline_from_bytes(self, excel_bytes: bytes, params: dict, objetivos: list[dict] | None = None) -> dict[str, Any]:
        ...
