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
        checkpoint: dict[str, Any] | None = None,
        historical_transactions: dict[str, list[dict[str, Any]]] | None = None,
    ) -> PipelineResult:
        legacy_path_str = str(self._legacy_path)
        if legacy_path_str not in sys.path:
            sys.path.append(legacy_path_str)

        from pipeline import run_pipeline, PipelineParams  # type: ignore

        parsed = PipelineParams(**params.to_dict())
        objetivos_con_checkpoint = self._apply_objetivo_checkpoint(objetivos or [], checkpoint or {})
        result = run_pipeline(
            excel=io.BytesIO(excel_bytes),
            params=parsed,
            objetivos=objetivos_con_checkpoint,
            fondo_reserva_snapshot=None,
            output_path=None,
            checkpoint=checkpoint,
        )
        return self._to_response(result, checkpoint or {}, historical_transactions or {})

    def _apply_objetivo_checkpoint(self, objetivos: list[dict], checkpoint: dict[str, Any]) -> list[dict]:
        """Sustituye el saldo_inicial de cada objetivo todavía abierto por el saldo con el que cerró
        el tramo anterior, para que un objetivo a caballo entre dos tramos no pierda su progreso."""
        objetivos_saldos = checkpoint.get("objetivos_saldos") or {}
        if not objetivos_saldos:
            return objetivos
        actualizados = []
        for objetivo in objetivos:
            nombre = objetivo.get("nombre")
            if nombre in objetivos_saldos:
                objetivo = {**objetivo, "saldo_inicial": objetivos_saldos[nombre]}
            actualizados.append(objetivo)
        return actualizados

    def _to_response(
        self, result: dict[str, Any], checkpoint: dict[str, Any], historical_transactions: dict[str, list[dict[str, Any]]]
    ) -> PipelineResult:
        historial = result.get("historial")
        resumen_df, objetivos_df = self._split_historial(historial)
        # Dividendos/Comisiones acumulados usan solo las filas de ESTE tramo: ya son continuos por su
        # propio checkpoint (dividendos_acumulado/comisiones_acumulado), y sumarles además el histórico
        # completo los duplicaría.
        resumen_df = self._add_dividend_history_to_summary(
            resumen_df, result.get("ingresos"), float(checkpoint.get("dividendos_acumulado", 0.0))
        )
        resumen_df = self._add_fees_history_to_summary(
            resumen_df, result.get("gastos"), float(checkpoint.get("comisiones_acumulado", 0.0))
        )
        ultimo_mes = self._last_record(resumen_df)

        presupuesto = result.get("presupuesto")
        cuentas = getattr(presupuesto, "accounts", {}) or {}

        # Los desgloses (categorías de gasto/ingreso, dividendos por empresa...) sí necesitan el
        # histórico completo: a diferencia del resumen mensual, no tienen un acumulador propio, así
        # que se recalculan cada vez a partir de las filas de detalle (histórico guardado + este tramo).
        analysis_result = {
            **result,
            "gastos": self._merge_transaction_rows(result.get("gastos"), historical_transactions.get("gastos")),
            "ingresos": self._merge_transaction_rows(result.get("ingresos"), historical_transactions.get("ingresos")),
        }

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
            analisis=self._build_analysis(analysis_result, resumen_df),
            checkpoint=self._derive_checkpoint(result, resumen_df, objetivos_df),
            transacciones={
                "gastos": self._transactions_for_persistence(result.get("gastos")),
                "ingresos": self._transactions_for_persistence(result.get("ingresos")),
            },
        )

    def _merge_transaction_rows(self, current_df: Any, historical_rows: list[dict[str, Any]] | None) -> Any:
        """Combina las filas de este tramo con el histórico ya guardado, sin duplicar: para un mes
        presente en ambos (p.ej. al resincronizar), gana la versión de este tramo."""
        if not historical_rows:
            return current_df
        historical_df = pd.DataFrame(historical_rows)
        if historical_df.empty or "fecha" not in historical_df.columns:
            return current_df
        historical_df["fecha"] = pd.to_datetime(historical_df["fecha"], errors="coerce")
        historical_df = historical_df.dropna(subset=["fecha"])
        # Lo que viene del CSV persistido llega como texto (incluida "cantidad"): sin convertirlo a
        # numérico aquí, concatenar con las filas de este tramo (ya numéricas) deja la columna con
        # tipos mixtos, y groupby/sum aguas abajo concatena los números como si fueran texto.
        if "cantidad" in historical_df.columns:
            historical_df["cantidad"] = pd.to_numeric(historical_df["cantidad"], errors="coerce").fillna(0.0)
        if not isinstance(current_df, pd.DataFrame) or current_df.empty:
            return historical_df.drop(columns=["Mes"], errors="ignore")
        current_months = set(pd.to_datetime(current_df["fecha"], errors="coerce").dt.to_period("M").astype(str))
        historical_df["_mes"] = historical_df["fecha"].dt.to_period("M").astype(str)
        historical_only = historical_df[~historical_df["_mes"].isin(current_months)].drop(columns=["_mes", "Mes"], errors="ignore")
        if historical_only.empty:
            return current_df
        return pd.concat([historical_only, current_df], ignore_index=True)

    def _transactions_for_persistence(self, df: Any) -> list[dict[str, Any]]:
        if not isinstance(df, pd.DataFrame) or df.empty or "fecha" not in df.columns:
            return []
        output = df.copy()
        output["fecha"] = pd.to_datetime(output["fecha"], errors="coerce")
        output = output.dropna(subset=["fecha"])
        output["Mes"] = output["fecha"].dt.to_period("M").astype(str)
        output["fecha"] = output["fecha"].dt.strftime("%Y-%m-%d")
        return self._dataframe_to_records(output)

    def _derive_checkpoint(self, result: dict[str, Any], resumen_df: Any, objetivos_df: Any = None) -> dict[str, Any] | None:
        """Calcula el estado de cierre (saldos, acumulados) del último mes procesado, para que un
        futuro tramo pueda continuar desde aquí en vez de reiniciar los acumuladores desde cero."""
        if not isinstance(resumen_df, pd.DataFrame) or resumen_df.empty:
            return None

        last = resumen_df.iloc[-1]
        mes = str(last.get("Mes"))

        # Estado sin redondear del cierre del pipeline: usarlo (en vez de releer las columnas ya
        # redondeadas a 2 decimales de resumen_df) evita que un futuro tramo arrastre un error de
        # 1 céntimo por cada redondeo intermedio.
        estado_cierre = result.get("estado_cierre") or {}

        saldos_iniciales = None
        presupuesto = result.get("presupuesto")
        if presupuesto is not None:
            try:
                fecha_corte = pd.Period(mes, freq="M").to_timestamp(how="end")
                balances = presupuesto.balances_a_fecha(fecha_corte)
                saldos_iniciales = {str(nombre): float(valor) for nombre, valor in balances.items()}
            except Exception:
                saldos_iniciales = None

        return {
            "as_of_month": mes,
            "saldos_iniciales": saldos_iniciales,
            "deuda_acumulada": float(estado_cierre.get("deuda_acumulada", 0.0) or 0.0),
            "regalos": float(estado_cierre.get("regalos", 0.0) or 0.0),
            "vacaciones": float(estado_cierre.get("vacaciones", 0.0) or 0.0),
            "inversiones": float(estado_cierre.get("inversiones", 0.0) or 0.0),
            "ahorro": float(estado_cierre.get("ahorro", 0.0) or 0.0),
            "fondo_reserva_snapshot": {
                "Cantidad cargada": float(estado_cierre.get("fondo_cargado", 0.0) or 0.0),
                "Cantidad del fondo": 0.0,
                "Porcentaje": 0.0,
            },
            "dividendos_acumulado": float(last.get("Dividendos", 0.0) or 0.0),
            "comisiones_acumulado": float(last.get("Comisiones", 0.0) or 0.0),
            "ingreso_mes_anterior": self._real_income_for_month(result.get("ingresos"), mes),
            "objetivos_saldos": self._objetivos_saldos_abiertos(objetivos_df, mes),
        }

    def _objetivos_saldos_abiertos(self, objetivos_df: Any, mes: str) -> dict[str, float]:
        """Saldo de cierre de cada objetivo que sigue abierto (no ha vencido) al terminar el tramo,
        para que el siguiente tramo continúe su progreso en vez de reiniciarlo a 0. Los objetivos que
        vencen dentro de este mismo tramo ya quedan liquidados y no se arrastran."""
        if not isinstance(objetivos_df, pd.DataFrame) or objetivos_df.empty:
            return {}
        if not {"Mes", "Objetivo", "saldo_fin_mes", "vence_en_mes"}.issubset(objetivos_df.columns):
            return {}
        hasta_mes = objetivos_df[objetivos_df["Mes"] <= mes]
        if hasta_mes.empty:
            return {}
        ultima_fila_por_objetivo = hasta_mes.sort_values("Mes").groupby("Objetivo").tail(1)
        abiertos = ultima_fila_por_objetivo[~ultima_fila_por_objetivo["vence_en_mes"].astype(bool)]
        return {str(row["Objetivo"]): round(float(row["saldo_fin_mes"]), 2) for _, row in abiertos.iterrows()}

    def _real_income_for_month(self, ingresos: Any, mes: str) -> float:
        """Suma de 'Ingreso Real' del mes indicado, para que el primer mes de un tramo que continúa
        un histórico ya cerrado calcule su presupuesto bruto igual que si nunca se hubiera cortado."""
        if not isinstance(ingresos, pd.DataFrame) or ingresos.empty:
            return 0.0
        required = {"fecha", "cantidad", "tipo_logico"}
        if not required.issubset(ingresos.columns):
            return 0.0
        data = ingresos.copy()
        data["fecha"] = pd.to_datetime(data["fecha"], errors="coerce")
        data["cantidad"] = pd.to_numeric(data["cantidad"].astype(str).str.replace(",", ".", regex=False), errors="coerce").fillna(0.0)
        data = data.dropna(subset=["fecha"])
        if data.empty:
            return 0.0
        reales = data[data["tipo_logico"] == "Ingreso Real"].copy()
        if reales.empty:
            return 0.0
        reales["mes"] = reales["fecha"].dt.to_period("M").astype(str)
        return round(float(reales.groupby("mes")["cantidad"].sum().get(mes, 0.0)), 2)

    def _add_dividend_history_to_summary(self, resumen: Any, ingresos: Any, initial_value: float = 0.0) -> Any:
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
            output["Dividendos"] = round(initial_value, 2)
            return output

        dividends["Mes"] = dividends["fecha"].dt.to_period("M").astype(str)
        monthly = dividends.groupby("Mes")["cantidad"].sum().sort_index()
        accumulated = monthly.cumsum() + initial_value
        output["Dividendos"] = output_months.map(lambda month: accumulated_value_until(accumulated, month, initial_value))
        return output

    def _add_fees_history_to_summary(self, resumen: Any, gastos: Any, initial_value: float = 0.0) -> Any:
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
            output["Comisiones"] = round(initial_value, 2)
            return output

        fees["Mes"] = fees["fecha"].dt.to_period("M").astype(str)
        monthly = fees.groupby("Mes")["cantidad"].sum().sort_index()
        accumulated = monthly.cumsum() + initial_value
        output["Comisiones"] = output_months.map(lambda month: accumulated_value_until(accumulated, month, initial_value))
        return output

    def _build_dividend_payments(self, ingresos: Any) -> list[dict[str, Any]]:
        """Extrae cada pago de dividendo (fecha, empresa) a partir de la etiqueta secundaria
        (p.ej. "Dividendos, IB") y del comentario (p.ej. "Iberdrola"). Devuelve la lista de pagos
        individuales sin agregar, para que cada pantalla pueda acumularlos hasta la fecha que le
        interese (mes seleccionado en Finanzas, fecha de la foto en Cartera). No depende de una
        lista fija de empresas: cualquier etiqueta nueva que aparezca en el Excel se recoge igual."""
        required = {"fecha", "categoria", "cantidad", "etiquetas"}
        if not isinstance(ingresos, pd.DataFrame) or ingresos.empty or not required.issubset(ingresos.columns):
            return []

        data = ingresos.copy()
        data["fecha"] = pd.to_datetime(data["fecha"], errors="coerce")
        data["cantidad"] = pd.to_numeric(data["cantidad"].astype(str).str.replace(",", ".", regex=False), errors="coerce").fillna(0.0)
        data["comentario"] = data.get("comentario", "").fillna("").astype(str)
        data = data.dropna(subset=["fecha"])
        if data.empty:
            return []

        categories = data["categoria"].map(normalize_text)
        dividends = data[categories == "interes"].copy()
        dividends["tag_list"] = dividends["etiquetas"].map(lambda value: [tag.strip() for tag in str(value).split(",")])
        dividends = dividends[dividends["tag_list"].map(lambda tags: any(normalize_text(tag) == "dividendos" for tag in tags))]
        if dividends.empty:
            return []

        def company_tag(tags: list[str]) -> str | None:
            for tag in tags:
                if tag and normalize_text(tag) != "dividendos":
                    return tag
            return None

        dividends["codigo"] = dividends["tag_list"].map(company_tag)
        dividends = dividends[dividends["codigo"].notna()]
        if dividends.empty:
            return []

        payments = []
        for row in dividends.sort_values("fecha").itertuples():
            payments.append(
                {
                    "fecha": row.fecha.strftime("%Y-%m-%d"),
                    "codigo": row.codigo,
                    "comentario": row.comentario.strip(),
                    "cantidad": round(float(row.cantidad), 2),
                }
            )
        return payments

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
            analysis["dividendos_pagos"] = self._build_dividend_payments(ingresos)
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


def accumulated_value_until(series: pd.Series, month: str, default: float = 0.0) -> float:
    values = series[series.index <= month]
    if values.empty:
        return round(float(default), 2)
    return round(float(values.iloc[-1]), 2)
