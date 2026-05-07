from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from backend.domain.models import PipelineConfig
from backend.domain.results import HistoryResult, MovementSummary, PipelineResult


class LegacyPipelineAdapter:
    """Adapter para reutilizar el pipeline actual sin romper comportamiento."""

    def __init__(self) -> None:
        self._legacy_path = Path(__file__).resolve().parents[2] / "Personal_finanzas"

    def run_pipeline_from_bytes(
        self,
        excel_bytes: bytes,
        params: PipelineConfig,
        objetivos: list[dict] | None = None,
    ) -> PipelineResult:
        legacy_path_str = str(self._legacy_path)
        if legacy_path_str not in sys.path:
            sys.path.append(legacy_path_str)

        from pipeline import run_pipeline, PipelineParams  # type: ignore

        parsed = PipelineParams(**params.to_dict())
        result = run_pipeline(
            excel=io.BytesIO(excel_bytes),
            params=parsed,
            objetivos=objetivos or [],
            fondo_reserva_snapshot=None,
            output_path=None,
        )
        return self._to_response(result)

    def _to_response(self, result: dict[str, Any]) -> PipelineResult:
        historial = result.get("historial")
        resumen_df, objetivos_df = self._split_historial(historial)
        ultimo_mes = self._last_record(resumen_df)

        presupuesto = result.get("presupuesto")
        cuentas = getattr(presupuesto, "accounts", {}) or {}

        return PipelineResult(
            params=self._params_to_dict(result.get("params")),
            movimientos=MovementSummary(
                gastos=self._dataframe_len(result.get("gastos")),
                ingresos=self._dataframe_len(result.get("ingresos")),
                transferencias=self._dataframe_len(result.get("transferencias")),
                cuentas=len(cuentas),
            ),
            historial=HistoryResult(
                meses=self._dataframe_len(resumen_df),
                ultimo_mes=ultimo_mes,
                resumen=self._dataframe_to_records(resumen_df),
                objetivos=self._dataframe_to_records(objetivos_df),
            ),
            analisis=self._build_analysis(result),
        )

    def _split_historial(self, historial: Any) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        if isinstance(historial, tuple) and len(historial) == 2:
            return historial
        return None, None

    def _dataframe_len(self, value: Any) -> int:
        if isinstance(value, pd.DataFrame):
            return int(len(value))
        return 0

    def _last_record(self, value: Any) -> dict[str, Any] | None:
        if not isinstance(value, pd.DataFrame) or value.empty:
            return None
        return self._clean_record(value.tail(1).to_dict(orient="records")[0])

    def _dataframe_to_records(self, value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, pd.DataFrame) or value.empty:
            return []
        return [self._clean_record(record) for record in value.to_dict(orient="records")]

    def _params_to_dict(self, value: Any) -> dict[str, Any]:
        if hasattr(value, "__dataclass_fields__"):
            return self._clean_record({field: getattr(value, field) for field in value.__dataclass_fields__})
        return self._clean_record(value if isinstance(value, dict) else {})

    def _build_analysis(self, result: dict[str, Any]) -> dict[str, Any]:
        gastos = result.get("gastos")
        ingresos = result.get("ingresos")
        analysis: dict[str, Any] = {}
        if isinstance(gastos, pd.DataFrame) and isinstance(ingresos, pd.DataFrame):
            analysis["gastos"] = self._build_expense_analysis(gastos, ingresos)
            analysis["ahorro"] = self._build_savings_analysis(gastos, ingresos)
        return analysis

    def _build_expense_analysis(self, gastos: pd.DataFrame, ingresos: pd.DataFrame) -> dict[str, Any]:
        from logic import resumen_gastos, resumen_mensual  # type: ignore

        categorias = resumen_gastos(gastos, ingresos)
        mensual = resumen_mensual(gastos, ingresos)
        if categorias.empty:
            return {
                "categorias": [],
                "mensual": [],
                "totales_categoria": [],
                "ultimo_mes": None,
            }

        categorias_records = self._dataframe_to_records(categorias.reset_index().rename(columns={"mes": "Mes"}))
        monthly_records = self._dataframe_to_records(mensual.reset_index().rename(columns={"mes": "Mes"}))

        totals = categorias.sum().sort_values(ascending=False)
        totals_records = [
            self._clean_record({"categoria": category, "total": amount})
            for category, amount in totals.items()
            if float(amount) != 0
        ]

        latest_month = str(categorias.index.max())
        latest_values = categorias.loc[latest_month].sort_values(ascending=False)
        latest_records = [
            self._clean_record({"categoria": category, "total": amount})
            for category, amount in latest_values.items()
            if float(amount) != 0
        ]

        return {
            "categorias": categorias_records,
            "mensual": monthly_records,
            "totales_categoria": totals_records,
            "ultimo_mes": {
                "Mes": latest_month,
                "categorias": latest_records,
            },
        }

    def _build_savings_analysis(self, gastos: pd.DataFrame, ingresos: pd.DataFrame) -> dict[str, Any]:
        from logic import resumen_mensual  # type: ignore

        mensual = resumen_mensual(gastos, ingresos).reset_index().rename(columns={"mes": "Mes"})
        if mensual.empty:
            return {"mensual": [], "ultimo_mes": None}

        mensual["porcentaje_ahorro"] = mensual.apply(
            lambda row: ((row["ingresos"] - row["gastos"]) / row["ingresos"] * 100) if row["ingresos"] else 0,
            axis=1,
        )
        records = self._dataframe_to_records(mensual)
        return {
            "mensual": records,
            "ultimo_mes": records[-1] if records else None,
        }

    def _clean_record(self, record: dict[str, Any]) -> dict[str, Any]:
        return {str(key): self._clean_value(value) for key, value in record.items()}

    def _clean_value(self, value: Any) -> Any:
        if pd.isna(value) if not isinstance(value, (list, tuple, dict)) else False:
            return None
        if isinstance(value, pd.Period):
            return value.strftime("%Y-%m")
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if hasattr(value, "item"):
            return value.item()
        return value
