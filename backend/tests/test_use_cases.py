from backend.application.use_cases import ProcessFinanceWorkbookUseCase
from backend.domain.models import PipelineConfig
from backend.domain.results import HistoryResult, MovementSummary, PipelineResult


class FakeFinancePort:
    def run_pipeline_from_bytes(
        self,
        excel_bytes: bytes,
        params: PipelineConfig,
        objetivos: list[dict] | None = None,
        checkpoint: dict | None = None,
    ):
        return PipelineResult(
            params=params.to_dict(),
            movimientos=MovementSummary(gastos=1, ingresos=2, transferencias=3, cuentas=4),
            historial=HistoryResult(meses=1, ultimo_mes={"Mes": "2024-10"}, resumen=[], objetivos=objetivos or []),
        )


def test_process_finance_workbook_use_case_returns_domain_result():
    use_case = ProcessFinanceWorkbookUseCase(port=FakeFinancePort())

    result = use_case.execute(
        excel_bytes=b"excel",
        params=PipelineConfig(),
        objetivos=[{"nombre": "Coche"}],
    )

    assert isinstance(result, PipelineResult)
    assert result.movimientos.gastos == 1
    assert result.historial.objetivos == [{"nombre": "Coche"}]
