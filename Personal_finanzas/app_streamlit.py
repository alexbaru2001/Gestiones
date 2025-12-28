#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 13:18:26 2025

@author: alex
"""

# app_streamlit.py
from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from io_data import fusionar_archivos_registro_seguro
from logic import resumen_mensual, resumen_gastos
from plots import (
    plot_balance_mensual,
    plot_explicacion_gastos,
    plot_porcentaje_ahorro,
    grafico_presupuesto_1,
)
from pipeline import run_pipeline, PipelineParams


st.set_page_config(page_title="Finanzas personales", layout="wide")


def _float_pct(x: float) -> float:
    return float(x)


@st.cache_data(show_spinner=False)
def _run_cached_pipeline(
    excel_bytes: Optional[bytes],
    excel_name: str,
    fecha_inicio: str,
    porcentaje_gasto: float,
    porcentaje_inversion: float,
    porcentaje_vacaciones: float,
):
    """
    Cache por bytes del Excel + parámetros. Si no hay excel_bytes,
    usamos el Inicio.xlsx local vía pipeline.
    """
    params = PipelineParams(
        fecha_inicio=fecha_inicio,
        porcentaje_gasto=porcentaje_gasto,
        porcentaje_inversion=porcentaje_inversion,
        porcentaje_vacaciones=porcentaje_vacaciones,
    )

    if excel_bytes is None:
        return run_pipeline(excel_path=None, params=params)

    # Guardamos a un buffer temporal en memoria para pandas.
    # pd.read_excel acepta file-like.
    excel_file = io.BytesIO(excel_bytes)
    # Pasamos el buffer directamente al pipeline:
    # (pipeline._leer_inicio_xlsx usa Path, así que aquí hacemos un fallback:
    #  guardamos en disco temporal dentro de la app)
    tmp_dir = Path(".streamlit_tmp")
    tmp_dir.mkdir(exist_ok=True)
    tmp_path = tmp_dir / excel_name
    tmp_path.write_bytes(excel_bytes)

    return run_pipeline(excel_path=tmp_path, params=params)


def main():
    st.title("Gestión de finanzas personales")

    with st.sidebar:
        st.header("Datos")
        uploaded = st.file_uploader("Subir Excel (Inicio.xlsx)", type=["xlsx"])

        st.divider()
        st.header("Parámetros")
        fecha_inicio = st.text_input("Fecha inicio (YYYY-MM-DD)", value="2024-10-01")

        porcentaje_gasto = st.slider("Porcentaje gasto", 0.0, 1.0, 0.30, 0.01)
        porcentaje_inversion = st.slider("Porcentaje inversión", 0.0, 1.0, 0.10, 0.01)
        porcentaje_vacaciones = st.slider("Porcentaje vacaciones", 0.0, 1.0, 0.05, 0.01)

        total_pct = porcentaje_gasto + porcentaje_inversion + porcentaje_vacaciones
        st.caption(f"Suma parcial (sin objetivos): {total_pct:.2%}")

        st.divider()
        st.header("Mantenimiento")
        confirmar_fusion = st.checkbox("Confirmo fusionar registros descargados", value=False)
        if st.button("Fusionar registros"):
            fusionar_archivos_registro_seguro(confirmar=confirmar_fusion)
            st.success("Fusión ejecutada (si estaba confirmada).")

        st.divider()
        recalcular = st.button("Recalcular", type="primary")

    # Disparador: si no pulsa recalcular, igualmente cargamos una vez
    if uploaded is None:
        excel_bytes = None
        excel_name = "Inicio.xlsx"
    else:
        excel_bytes = uploaded.getvalue()
        excel_name = uploaded.name

    if recalcular:
        _run_cached_pipeline.clear()

    with st.spinner("Calculando..."):
        data = _run_cached_pipeline(
            excel_bytes=excel_bytes,
            excel_name=excel_name,
            fecha_inicio=fecha_inicio,
            porcentaje_gasto=_float_pct(porcentaje_gasto),
            porcentaje_inversion=_float_pct(porcentaje_inversion),
            porcentaje_vacaciones=_float_pct(porcentaje_vacaciones),
        )

    gastos = data["gastos"]
    ingresos = data["ingresos"]
    historial = data["historial"]

    # ---- Tabs UI ----
    tabs = st.tabs(["Resumen", "Historial", "Gastos", "Ahorro", "Presupuesto"])

    with tabs[0]:
        st.subheader("Resumen rápido")

        c1, c2, c3, c4 = st.columns(4)
        last = historial.iloc[-1]

        c1.metric("Total", f"{last['total']:.2f} €" if "total" in last else "—")
        c2.metric("Ahorros", f"{last['Ahorros']:.2f} €" if "Ahorros" in last else "—")
        c3.metric("Inversiones", f"{last['Inversiones']:.2f} €" if "Inversiones" in last else "—")
        c4.metric("Fondo reserva", f"{last['Fondo de reserva cargado']:.2f} €" if "Fondo de reserva cargado" in last else "—")

        st.caption(f"Fuente: {data['excel_path']}")

        # Balance mensual
        try:
            resumen = resumen_mensual(gastos, ingresos)
            fig_balance = plot_balance_mensual(resumen)
            st.plotly_chart(fig_balance, use_container_width=True)
        except Exception as e:
            st.warning(f"No se pudo generar balance mensual: {e}")

    with tabs[1]:
        st.subheader("Historial cuentas virtuales")
        st.dataframe(historial, use_container_width=True)

        # Descarga CSV
        csv_bytes = historial.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Descargar historial.csv",
            data=csv_bytes,
            file_name="historial.csv",
            mime="text/csv",
        )

    with tabs[2]:
        st.subheader("Gastos por categoría")

        try:
            pivot, stats, figs = plot_explicacion_gastos(
                df_gastos=gastos,
                df_ingresos=ingresos,
                categorias_fijas=["alimentacion", "entretenimiento", "hosteleria", "otros", "transporte"],
            )

            # Stats (tabla)
            st.markdown("**Promedio mensual por categoría**")
            st.dataframe(stats["promedio_mensual_por_categoria"], use_container_width=True)

            st.markdown("**Volatilidad (std) por categoría**")
            st.dataframe(stats["volatilidad_por_categoria_std"], use_container_width=True)

            st.plotly_chart(figs["barras_apiladas"], use_container_width=True)
            st.plotly_chart(figs["porcentajes"], use_container_width=True)
            st.plotly_chart(figs["lineas"], use_container_width=True)
            st.plotly_chart(figs["boxplot"], use_container_width=True)
            st.plotly_chart(figs["heatmap"], use_container_width=True)

        except Exception as e:
            st.warning(f"No se pudo generar explicación de gastos: {e}")

    with tabs[3]:
        st.subheader("Porcentaje de ahorro mensual")
        try:
            df_ahorro, fig_ahorro = plot_porcentaje_ahorro(gastos, ingresos)
            st.plotly_chart(fig_ahorro, use_container_width=True)
            st.dataframe(df_ahorro, use_container_width=True)
        except Exception as e:
            st.warning(f"No se pudo calcular el porcentaje de ahorro: {e}")

    with tabs[4]:
        st.subheader("Presupuesto")
        try:
            fig_pres, metrics = grafico_presupuesto_1(historial)
            st.plotly_chart(fig_pres, use_container_width=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Restante", f"{metrics.get('dinero_restante', 0):.2f} €")
            c2.metric("Gasto semanal aprox.", f"{metrics.get('gasto_semanal', 0):.2f} €")
            c3.metric("Semanas restantes", f"{int(metrics.get('semanas_restantes', 0))}")
        except Exception as e:
            st.warning(f"No se pudo generar el gráfico de presupuesto: {e}")


if __name__ == "__main__":
    main()
