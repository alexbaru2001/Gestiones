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
        resumen_df = self._add_dividend_history_to_summary(resumen_df, result.get("ingresos"))
        resumen_df = self._add_fees_history_to_summary(resumen_df, result.get("gastos"))
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
            analisis=self._build_analysis(result, resumen_df),
        )

    def _add_dividend_history_to_summary(self, resumen: Any, ingresos: Any) -> Any:
        required = {"fecha", "categoria", "cantidad", "etiquetas"}
        if not isinstance(resumen, pd.DataFrame) or resumen.empty:
            return resumen
        if not isinstance(ingresos, pd.DataFrame) or ingresos.empty or not required.issubset(ingresos.columns):
            return resumen

        data = ingresos.copy()
        data["fecha"] = pd.to_datetime(data["fecha"], errors="coerce")
        data["cantidad"] = pd.to_numeric(data["cantidad"].astype(str).str.replace(",", ".", regex=False), errors="coerce").fillna(0.0)
        data = data.dropna(subset=["fecha"])
        if data.empty:
            return resumen

        categories = data["categoria"].map(normalize_text)
        tags = data["etiquetas"].map(normalize_text)
        dividends = data[(categories == "interes") & tags.str.contains("dividendos", regex=False)].copy()

        output = resumen.copy()
        month_column = "Mes" if "Mes" in output.columns else "mes" if "mes" in output.columns else None
        if month_column is None:
            return output
        output_months = output[month_column].astype(str)

        if dividends.empty:
            output["Dividendos"] = 0.0
            return output

        dividends["Mes"] = dividends["fecha"].dt.to_period("M").astype(str)
        monthly = dividends.groupby("Mes")["cantidad"].sum().sort_index()
        accumulated = monthly.cumsum()
        output["Dividendos"] = output_months.map(lambda month: accumulated_value_until(accumulated, month))
        return output

    def _add_fees_history_to_summary(self, resumen: Any, gastos: Any) -> Any:
        required = {"fecha", "categoria", "cantidad", "etiquetas"}
        if not isinstance(resumen, pd.DataFrame) or resumen.empty:
            return resumen
        if not isinstance(gastos, pd.DataFrame) or gastos.empty or not required.issubset(gastos.columns):
            return resumen

        data = gastos.copy()
        data["fecha"] = pd.to_datetime(data["fecha"], errors="coerce")
        data["cantidad"] = pd.to_numeric(data["cantidad"].astype(str).str.replace(",", ".", regex=False), errors="coerce").fillna(0.0)
        data = data.dropna(subset=["fecha"])
        if data.empty:
            return resumen

        categories = data["categoria"].map(normalize_text)
        tags = data["etiquetas"].map(normalize_text)
        fees = data[(categories == "otros") & tags.str.contains("comision", regex=False)].copy()

        output = resumen.copy()
        month_column = "Mes" if "Mes" in output.columns else "mes" if "mes" in output.columns else None
        if month_column is None:
            return output
        output_months = output[month_column].astype(str)

        if fees.empty:
            output["Comisiones"] = 0.0
            return output

        fees["Mes"] = fees["fecha"].dt.to_period("M").astype(str)
        monthly = fees.groupby("Mes")["cantidad"].sum().sort_index()
        accumulated = monthly.cumsum()
        output["Comisiones"] = output_months.map(lambda month: accumulated_value_until(accumulated, month))
        return output

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

    def _build_analysis(self, result: dict[str, Any], resumen: pd.DataFrame | None = None) -> dict[str, Any]:
        gastos = result.get("gastos")
        ingresos = result.get("ingresos")
        analysis: dict[str, Any] = {}
        if isinstance(gastos, pd.DataFrame) and isinstance(ingresos, pd.DataFrame):
            analysis["gastos"] = self._build_expense_analysis(gastos, ingresos)
            analysis["ahorro"] = self._build_savings_analysis(gastos, ingresos, resumen)
            analysis["ingresos"] = self._build_income_analysis(ingresos)
        return analysis

    def _build_income_analysis(self, ingresos: pd.DataFrame) -> dict[str, Any]:
        required = {"fecha", "categoria", "cantidad"}
        if ingresos.empty or not required.issubset(ingresos.columns):
            return {"categorias": [], "totales_categoria": [], "ultimo_mes": None}

        data = ingresos.copy()
        data["fecha"] = pd.to_datetime(data["fecha"], errors="coerce")
        data["cantidad"] = pd.to_numeric(data["cantidad"].astype(str).str.replace(",", ".", regex=False), errors="coerce").fillna(0.0)
        data = data.dropna(subset=["fecha"])
        if data.empty:
            return {"categorias": [], "totales_categoria": [], "ultimo_mes": None}

        data["Mes"] = data["fecha"].dt.to_period("M").astype(str)
        categorias = data.pivot_table(index="Mes", columns="categoria", values="cantidad", aggfunc="sum", fill_value=0)
        categorias_records = self._dataframe_to_records(categorias.reset_index())
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
            "totales_categoria": totals_records,
            "ultimo_mes": {
                "Mes": latest_month,
                "categorias": latest_records,
            },
        }

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

    def _build_savings_analysis(self, gastos: pd.DataFrame, ingresos: pd.DataFrame, resumen: pd.DataFrame | None = None) -> dict[str, Any]:
        from logic import resumen_mensual  # type: ignore

        mensual = resumen_mensual(gastos, ingresos).reset_index().rename(columns={"mes": "Mes"})
        if "Mes" not in mensual.columns and "index" in mensual.columns:
            mensual = mensual.rename(columns={"index": "Mes"})
        if mensual.empty:
            return {"mensual": [], "ultimo_mes": None}

        if isinstance(resumen, pd.DataFrame) and not resumen.empty and {"Mes", "💳 Gasto del mes"}.issubset(resumen.columns):
            gastos_historial = resumen[["Mes", "💳 Gasto del mes"]].copy()
            gastos_historial["Mes"] = gastos_historial["Mes"].astype(str)
            gastos_historial["gastos_historial"] = pd.to_numeric(gastos_historial["💳 Gasto del mes"], errors="coerce").fillna(0.0)
            mensual = mensual.merge(gastos_historial[["Mes", "gastos_historial"]], on="Mes", how="left")
            mensual["gastos"] = mensual["gastos_historial"].fillna(mensual["gastos"])
            mensual = mensual.drop(columns=["gastos_historial"])
            mensual["balance"] = mensual["ingresos"] - mensual["gastos"]

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


def normalize_text(value: Any) -> str:
    import unicodedata

    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFD", text)
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    return text.strip().lower()


def accumulated_value_until(series: pd.Series, month: str) -> float:
    values = series[series.index <= month]
    if values.empty:
        return 0.0
    return round(float(values.iloc[-1]), 2)
