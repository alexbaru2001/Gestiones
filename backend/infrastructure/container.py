from backend.adapters.legacy_pipeline_adapter import LegacyPipelineAdapter
from backend.application.use_cases import ProcessFinanceWorkbookUseCase


def build_process_finance_workbook_use_case() -> ProcessFinanceWorkbookUseCase:
    return ProcessFinanceWorkbookUseCase(port=LegacyPipelineAdapter())
