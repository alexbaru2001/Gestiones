#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import io
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
import contextlib
import io
from pipeline import PipelineParams, run_pipeline

# Ajusta imports a tus módulos reales
from plots import plot_balance_mensual, plot_explicacion_gastos, plot_porcentaje_ahorro
from io_data import leer_fondo_reserva_snapshot
from logic import resumen_global


def _objetivos_default() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "nombre": "Coche",
                "etiquetas": "coche",
                "fraccion_presupuesto": 0.33,
                "duracion_meses": 10,
                "mes_inicio": "2025-01",   # o None
            }
        ]
    )


def _df_objetivos_a_lista(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    out: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        out.append(
            {
                "nombre": str(row.get("nombre", "")).strip(),
                "etiquetas": str(row.get("etiquetas", "")).strip(),
                "fraccion_presupuesto": float(row.get("fraccion_presupuesto", 0) or 0),
                "duracion_meses": int(row.get("duracion_meses", 0) or 0),
                "mes_inicio": str(row.get("mes_inicio", "")).strip() or None,
            }
        )
    return out


def main():
    st.set_page_config(page_title="Finanzas personales", layout="wide")
    st.title("Finanzas personales - Streamlit")
    
    if "out" not in st.session_state:
        st.session_state.out = None

    if "saldo_result" not in st.session_state:
        st.session_state.saldo_result = None
    
    with st.sidebar:
        st.header("Datos")
        excel_file = st.file_uploader("Sube Inicio.xlsx", type=["xlsx"])

        st.header("Parámetros")
        fecha_inicio = st.text_input("fecha_inicio (YYYY-MM-DD)", value="2024-10-01")
        porcentaje_gasto = st.number_input("porcentaje_gasto", min_value=0.0, max_value=1.0, value=0.30, step=0.01)
        porcentaje_inversion = st.number_input("porcentaje_inversion", min_value=0.0, max_value=1.0, value=0.10, step=0.01)
        porcentaje_vacaciones = st.number_input("porcentaje_vacaciones", min_value=0.0, max_value=1.0, value=0.05, step=0.01)

        st.header("Objetivos (presupuesto)")
        if "df_objetivos" not in st.session_state:
            st.session_state.df_objetivos = _objetivos_default()

        st.caption("Campos: nombre, etiquetas (coma-separadas), fraccion_presupuesto, duracion_meses")
        df_edit = st.data_editor(
            st.session_state.df_objetivos,
            num_rows="dynamic",
            use_container_width=True,
            key="editor_objetivos",
        )
        st.session_state.df_objetivos = df_edit
        
        st.divider()
        st.header("Saldo a una fecha")

        fecha_saldo = st.date_input("Fecha", value=pd.to_datetime("2025-09-30"))
        if st.button("Calcular saldo"):
            if st.session_state.out is None:
                st.warning("Primero ejecuta el pipeline para construir el presupuesto.")
            else:
                presupuesto = st.session_state.out["presupuesto"]
                saldo_total, saldos_dict = resumen_global(presupuesto, fecha_saldo)

                st.success(f"Saldo total a {fecha_saldo}: {saldo_total:,.2f} €")

                df_saldos = (
                    pd.DataFrame(list(saldos_dict.items()), columns=["Cuenta", "Saldo"])
                    .sort_values("Saldo", ascending=False)
                    .reset_index(drop=True)
                )
                st.dataframe(df_saldos, use_container_width=True)
        
        
        # Validación simple en UI
        try:
            objetivos_lista = _df_objetivos_a_lista(df_edit)
            total_fr = sum(float(o.get("fraccion_presupuesto", 0) or 0) for o in objetivos_lista if o.get("nombre"))
            if total_fr > 1 + 1e-9:
                st.error(f"Suma fraccion_presupuesto = {total_fr:.2f} > 1. Reduce fracciones.")
        except Exception as e:
            st.error(f"Error en objetivos: {e}")

        ejecutar = st.button("Ejecutar pipeline")

    if not ejecutar:
        st.info("Sube el excel, ajusta parámetros y pulsa 'Ejecutar pipeline'.")
        return

    if excel_file is None:
        st.error("Necesitas subir Inicio.xlsx.")
        return

    params = PipelineParams(
        fecha_inicio=fecha_inicio,
        porcentaje_gasto=porcentaje_gasto,
        porcentaje_inversion=porcentaje_inversion,
        porcentaje_vacaciones=porcentaje_vacaciones,
    )

    try:
        excel_bytes = excel_file.read()
        excel_buf = io.BytesIO(excel_bytes)

        # Si tienes snapshot del fondo, úsalo; si no, pasa None
        try:
            fondo_snapshot = leer_fondo_reserva_snapshot()
        except Exception:
            fondo_snapshot = None

        objetivos_lista = _df_objetivos_a_lista(st.session_state.df_objetivos)

        out = run_pipeline(
            excel=excel_buf,
            params=params,
            objetivos=objetivos_lista,
            output_path=None,
            fondo_reserva_snapshot=fondo_snapshot,
        )
        st.session_state.out = out
    except Exception as e:
        st.exception(e)
        return

    out = st.session_state.out
    if out is None:
        st.info("Sube el excel, ajusta parámetros y pulsa 'Ejecutar pipeline'.")
        return
    historial, objetivos_df = out["historial"]
    gastos: pd.DataFrame = out["gastos"]
    ingresos: pd.DataFrame = out["ingresos"]
    presupuesto = out["presupuesto"]

    st.subheader("Historial (cuentas virtuales)")
    st.dataframe(historial, use_container_width=True)
    
    st.subheader("Objetivos (detalle mensual)")
    st.dataframe(objetivos_df, use_container_width=True)
    
    st.subheader("Saldo a una fecha")

    fecha_saldo = st.text_input("Fecha (YYYY-MM-DD)", value="2025-09-30", key="fecha_saldo")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        btn_saldo = st.button("Calcular saldo", key="btn_saldo")
    
    with col2:
        if btn_saldo:
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    resumen_global(presupuesto, fecha_saldo)
                salida = buf.getvalue().strip()
                if not salida:
                    st.info("No hubo salida.")
                else:
                    st.code(salida)
            except Exception as e:
                st.exception(e)
    st.subheader("Gráficos")

    # 1) Balance mensual: requiere un resumen con columnas ingresos/gastos/balance
    try:
        # si tú ya tienes una función resumen_mensual en logic, úsala aquí.
        # mientras tanto, intento construirlo desde historial si existe:
        if {"Presupuesto Mes", "Gasto del mes"}.issubset(historial.columns):
            resumen = pd.DataFrame(index=historial["Mes"].astype(str))
            resumen["ingresos"] = historial["Presupuesto Mes"].values
            resumen["gastos"] = historial["Gasto del mes"].values
            resumen["balance"] = resumen["ingresos"] - resumen["gastos"]
            fig_bal = plot_balance_mensual(resumen)
            st.plotly_chart(fig_bal, use_container_width=True)
        else:
            st.warning("No encuentro columnas para construir resumen de balance (Presupuesto Mes / Gasto del mes).")
    except Exception as e:
        st.warning(f"No se pudo generar Balance mensual: {e}")

    # 2) Explicación de gastos
    try:
        pivot, stats, figs = plot_explicacion_gastos(gastos, ingresos, categorias_fijas=None)
        st.write("KPIs")
        st.write(stats)

        st.plotly_chart(figs["barras_apiladas"], use_container_width=True)
        st.plotly_chart(figs["porcentajes"], use_container_width=True)
        st.plotly_chart(figs["lineas"], use_container_width=True)
        st.plotly_chart(figs["boxplot"], use_container_width=True)
        st.plotly_chart(figs["heatmap"], use_container_width=True)
    except Exception as e:
        st.warning(f"No se pudo generar explicación de gastos: {e}")

    # 3) % ahorro
    try:
        df_ahorro, fig_ahorro = plot_porcentaje_ahorro(gastos, ingresos)
        st.plotly_chart(fig_ahorro, use_container_width=True)
        st.dataframe(df_ahorro, use_container_width=True)
    except Exception as e:
        st.warning(f"No se pudo generar % ahorro: {e}")


if __name__ == "__main__":
    main()
