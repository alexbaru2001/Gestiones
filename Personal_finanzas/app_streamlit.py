

from __future__ import annotations

import io
import re
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from pipeline import PipelineParams, run_pipeline

# Ajusta imports a tus módulos reales
from plots import plot_balance_mensual, plot_balance_mensual_con_ahorro, plot_explicacion_gastos, plot_porcentaje_ahorro
from io_data import leer_fondo_reserva_snapshot
from logic import resumen_global, resumen_mensual


# =====================================================
# Helpers UI / datos
# =====================================================

def _objetivos_default() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "nombre": "Coche",
                "etiquetas": "coche",
                "fraccion_presupuesto": 0.33,
                "duracion_meses": 10,
                "mes_inicio": "2025-01",  # o None
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


def _validar_objetivos(df: pd.DataFrame) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Valida objetivos en UI y devuelve (errores, objetivos_lista_normalizada)."""
    errores: List[str] = []
    objetivos = _df_objetivos_a_lista(df)

    # Filtra filas vacías (sin nombre)
    objetivos = [o for o in objetivos if o.get("nombre")]

    # Validaciones por objetivo
    for i, o in enumerate(objetivos, start=1):
        nombre = o.get("nombre") or f"(fila {i})"

        fr = float(o.get("fraccion_presupuesto", 0) or 0)
        if fr < 0:
            errores.append(f"Objetivo '{nombre}': fraccion_presupuesto no puede ser negativa.")

        dur = int(o.get("duracion_meses", 0) or 0)
        if dur <= 0:
            errores.append(f"Objetivo '{nombre}': duracion_meses debe ser >= 1.")

        mes_inicio = o.get("mes_inicio")
        if mes_inicio is not None and not re.fullmatch(r"\d{4}-\d{2}", str(mes_inicio)):
            errores.append(f"Objetivo '{nombre}': mes_inicio debe ser 'YYYY-MM' o vacío.")

    # Validación global: suma fracciones
    total_fr = sum(float(o.get("fraccion_presupuesto", 0) or 0) for o in objetivos)
    if total_fr > 1 + 1e-9:
        errores.append(f"La suma de fraccion_presupuesto es {total_fr:.2f} y no puede ser > 1.")

    return errores, objetivos


def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _df_to_excel_bytes(dfs: Dict[str, pd.DataFrame]) -> bytes:
    """Genera un xlsx en memoria con varias hojas."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, df in dfs.items():
            safe = re.sub(r"[^A-Za-z0-9 _-]", "", name)[:31] or "Sheet"
            df.to_excel(writer, sheet_name=safe, index=False)
    return output.getvalue()


def _presupuesto_balances_df(presupuesto) -> pd.DataFrame:
    """DataFrame con saldos por cuenta."""
    saldos = []
    for nombre, cuenta in getattr(presupuesto, "accounts", {}).items():
        saldos.append({"Cuenta": nombre, "Saldo": float(getattr(cuenta, "balance", 0.0))})
    df = pd.DataFrame(saldos)
    if df.empty:
        return df
    return df.sort_values("Saldo", ascending=False).reset_index(drop=True)


# =====================================================
# Cache del pipeline
# =====================================================

@st.cache_resource(show_spinner=False)
def _run_pipeline_cached(
    excel_bytes: bytes,
    params_dict: Dict[str, Any],
    objetivos: List[Dict[str, Any]],
    fondo_reserva_snapshot: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    params = PipelineParams(**params_dict)
    excel_buf = io.BytesIO(excel_bytes)
    return run_pipeline(
        excel=excel_buf,
        params=params,
        objetivos=objetivos,
        fondo_reserva_snapshot=fondo_reserva_snapshot,
    )

def _gastos_mes_por_categoria(df_gastos: pd.DataFrame, mes_str: str) -> pd.DataFrame:
    """
    mes_str: 'YYYY-MM'
    Devuelve df con columnas: categoria, gasto
    """
    if df_gastos is None or not isinstance(df_gastos, pd.DataFrame) or df_gastos.empty:
        return pd.DataFrame(columns=["categoria", "gasto"])

    dfg = df_gastos.copy()
    if "fecha" not in dfg.columns or "categoria" not in dfg.columns or "cantidad" not in dfg.columns:
        return pd.DataFrame(columns=["categoria", "gasto"])

    dfg["mes"] = dfg["fecha"].dt.to_period("M").astype(str)
    dfg_mes = dfg.loc[dfg["mes"] == mes_str].copy()

    if dfg_mes.empty:
        return pd.DataFrame(columns=["categoria", "gasto"])

    out = (
        dfg_mes.groupby("categoria", as_index=False)["cantidad"]
        .sum()
        .rename(columns={"cantidad": "gasto"})
        .sort_values("gasto", ascending=False)
        .reset_index(drop=True)
    )
    return out



# =====================================================
# App
# =====================================================

def main():
    st.set_page_config(page_title="Finanzas personales", layout="wide")
    st.title("Finanzas personales")

    if "out" not in st.session_state:
        st.session_state.out = None
    if "df_objetivos" not in st.session_state:
        st.session_state.df_objetivos = _objetivos_default()

    # -------------------------
    # Sidebar: entradas + run
    # -------------------------
    with st.sidebar:
        st.header("1) Datos")
        excel_file = st.file_uploader("Sube Inicio.xlsx", type=["xlsx"])

        st.header("2) Parámetros")
        fecha_inicio = st.text_input("fecha_inicio (YYYY-MM-DD)", value="2024-10-01")
        porcentaje_gasto = st.number_input("porcentaje_gasto", min_value=0.0, max_value=1.0, value=0.30, step=0.01)
        porcentaje_inversion = st.number_input("porcentaje_inversion", min_value=0.0, max_value=1.0, value=0.10, step=0.01)
        porcentaje_vacaciones = st.number_input("porcentaje_vacaciones", min_value=0.0, max_value=1.0, value=0.05, step=0.01)

        st.header("3) Objetivos (presupuesto)")
        df_obj = st.session_state.df_objetivos.copy()

        # --- métricas asignado/disponible ---
        total_fr = 0.0
        if df_obj is not None and not df_obj.empty and "fraccion_presupuesto" in df_obj.columns:
            total_fr = float(pd.to_numeric(df_obj["fraccion_presupuesto"], errors="coerce").fillna(0).sum())
        
        disp_fr = max(0.0, 1.0 - total_fr)
        
        c1, c2, c3 = st.columns([1,1,2])
        c1.metric("Asignado", f"{total_fr*100:.1f}%")
        c2.metric("Disponible", f"{disp_fr*100:.1f}%")
        c3.progress(min(max(total_fr, 0.0), 1.0))
        
        st.caption("Crea objetivos con formulario. La suma de fracciones no puede superar 100%.")
        
        # --- Formulario guiado ---
        with st.form("form_objetivo", clear_on_submit=True):
            nombre = st.text_input("Nombre del objetivo", placeholder="Ej: Coche")
            etiquetas = st.text_input("Etiquetas (coma-separadas)", placeholder="ej: coche,capricho")
            colA, colB = st.columns(2)
            with colA:
                fraccion_pct = st.number_input("Asignación (% del presupuesto mensual)", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
            with colB:
                duracion = st.number_input("Duración (meses)", min_value=1, max_value=600, value=12, step=1)
        
            mes_inicio = st.text_input("Mes inicio (YYYY-MM)", value=pd.Timestamp.today().strftime("%Y-%m"))
        
            # Preview: cómo quedaría el total
            fr = fraccion_pct / 100.0
            total_preview = total_fr + fr
            st.info(f"Total tras añadir: {total_preview*100:.1f}% (límite 100%)")
        
            guardar = st.form_submit_button("Añadir objetivo", type="primary")
        
        if guardar:
            # Validación rápida antes de tocar df
            if not nombre.strip():
                st.error("El nombre no puede estar vacío.")
            elif total_fr + (fraccion_pct/100.0) > 1.0 + 1e-9:
                st.error("No se puede añadir: superarías el 100% asignado.")
            elif not re.fullmatch(r"\d{4}-\d{2}", str(mes_inicio).strip()):
                st.error("Mes inicio debe ser 'YYYY-MM'.")
            else:
                nueva = {
                    "nombre": nombre.strip(),
                    "etiquetas": etiquetas.strip(),
                    "fraccion_presupuesto": float(fraccion_pct/100.0),
                    "duracion_meses": int(duracion),
                    "mes_inicio": str(mes_inicio).strip(),
                }
                st.session_state.df_objetivos = pd.concat([df_obj, pd.DataFrame([nueva])], ignore_index=True)
                st.success("Objetivo añadido.")
        
        # --- modo avanzado: tabla ---
        with st.expander("Modo avanzado: editar tabla", expanded=False):
            st.caption("Campos: nombre, etiquetas, fraccion_presupuesto, duracion_meses, mes_inicio (YYYY-MM)")
            df_edit = st.data_editor(
                st.session_state.df_objetivos,
                num_rows="dynamic",
                use_container_width=True,
                key="editor_objetivos",
            )
            st.session_state.df_objetivos = df_edit
        
        # Validación final (tu función actual)
        errores_obj, objetivos_lista = _validar_objetivos(st.session_state.df_objetivos)
        if errores_obj:
            st.error("\n".join(errores_obj))
            
            
            
            
            

        st.divider()

        # Botón de ejecución (deshabilitado si falta excel o hay errores)
        can_run = (excel_file is not None) and (not errores_obj)
        ejecutar = st.button("Ejecutar pipeline", type="primary", disabled=not can_run)

        if not can_run and excel_file is None:
            st.info("Sube Inicio.xlsx para poder ejecutar el pipeline.")
        elif not can_run and errores_obj:
            st.info("Corrige los errores de objetivos para poder ejecutar el pipeline.")

    # -------------------------
    # Ejecutar pipeline (cacheado)
    # -------------------------
    if ejecutar:
        params = PipelineParams(
            fecha_inicio=fecha_inicio,
            porcentaje_gasto=porcentaje_gasto,
            porcentaje_inversion=porcentaje_inversion,
            porcentaje_vacaciones=porcentaje_vacaciones,
        )
        params_dict = asdict(params)

        try:
            excel_bytes = excel_file.read() if excel_file is not None else b""
            fondo_reserva_snapshot = leer_fondo_reserva_snapshot()

            with st.spinner("Procesando…"):
                out = _run_pipeline_cached(
                    excel_bytes=excel_bytes,
                    params_dict=params_dict,
                    objetivos=objetivos_lista,
                    fondo_reserva_snapshot=fondo_reserva_snapshot,
                )
            st.session_state.out = out
            st.success("Pipeline ejecutado correctamente.")
        except Exception as e:
            st.session_state.out = None
            st.exception(e)

    out = st.session_state.out
    if out is None:
        st.info("Configura datos y pulsa 'Ejecutar pipeline'.")
        return

    # -------------------------
    # Extraer outputs estándar
    # -------------------------
    presupuesto = out["presupuesto"]
    gastos = out.get("gastos")
    ingresos = out.get("ingresos")
    transferencias = out.get("transferencias")
    historial = out.get("historial")
    df_resumen, objetivos_df = (historial if isinstance(historial, tuple) and len(historial) == 2 else (None, None))

    balances_df = _presupuesto_balances_df(presupuesto)
    saldo_total_actual = float(getattr(presupuesto, "get_balance", lambda: 0.0)())

    # Métricas principales (con tolerancia si faltan cuentas)
    def _get_saldo_cuenta(nombre: str) -> Optional[float]:
        if balances_df is None or balances_df.empty:
            return None
        row = balances_df.loc[balances_df["Cuenta"].astype(str) == nombre]
        if row.empty:
            return None
        return float(row.iloc[0]["Saldo"])

    # --- métricas desde historial (última fila) ---
    inv_hist = None
    fr_hist = None
    ah_hist = None
    
    if isinstance(df_resumen, pd.DataFrame) and not df_resumen.empty:
        last = df_resumen.iloc[-1]
    
        # Nombres de columnas tal y como los construye tu lógica
        inv_hist = float(last.get("Dinero Invertido")) if "Dinero Invertido" in df_resumen.columns else None
        fr_hist = float(last.get("Fondo de reserva cargado")) if "Fondo de reserva cargado" in df_resumen.columns else None
    
        # Ojo: esta columna lleva emoji en tu df_resumen
        ah_col = "💰 Ahorros"
        ah_hist = float(last.get(ah_col)) if ah_col in df_resumen.columns else None

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Saldo total (actual)", f"{saldo_total_actual:,.2f} €")
    
    col2.metric("Invertido", f"{inv_hist:,.2f} €" if inv_hist is not None else "—")
    col3.metric("Fondo reserva", f"{fr_hist:,.2f} €" if fr_hist is not None else "—")
    col4.metric("Ahorro", f"{ah_hist:,.2f} €" if ah_hist is not None else "—")




    # -------------------------
    # Tabs
    # -------------------------
    tab_resumen, tab_graficos, tab_objetivos, tab_datos, tab_herr, tab_export = st.tabs(
        ["Resumen", "Gráficos", "Objetivos", "Datos", "Herramientas", "Exportar"]
    )

    with tab_resumen:
        st.subheader("Resumen del mes (presupuesto)")
    
        if not isinstance(df_resumen, pd.DataFrame) or df_resumen.empty:
            st.info("No hay historial (df_resumen) disponible para construir el resumen.")
        else:
            last = df_resumen.iloc[-1]
            mes = str(last.get("Mes", ""))
    
            gasto_mes = float(last.get("💳 Gasto del mes", 0.0))
            presupuesto_mes = float(last.get("💸 Presupuesto Mes", 0.0))
            presupuesto_disp = float(last.get("🧾 Presupuesto Disponible", presupuesto_mes))
    
            restante = max(0.0, presupuesto_disp - gasto_mes)
            deuda_mes = max(0.0, presupuesto_mes - presupuesto_disp)*0.4 #Se pone esto para ir poco a poco bajandolo

            # Para el donut 3-partes:
            gasto_variable = max(0.0, gasto_mes)
            restante_total = max(0.0, presupuesto_mes - deuda_mes - gasto_variable)
            pct_ejec = (gasto_mes / presupuesto_disp * 100) if presupuesto_disp else 0.0
            pct_base = (gasto_mes / presupuesto_mes * 100) if presupuesto_mes else 0.0
            pct_disp = (gasto_mes / presupuesto_disp * 100) if presupuesto_disp else 0.0

    
            # --- Layout elegante ---
            c1, c2 = st.columns([1.2, 1])
    
            with c1:
                # Donut Gastado vs Restante (no rompe si todo 0)
                import plotly.graph_objects as go
    
                fig = go.Figure(
                    data=[
                        go.Pie(
                            labels=["Compromisos/Deuda", "Gasto del mes", "Restante"],
                            values=[deuda_mes, gasto_variable, restante_total],
                            hole=0.55,
                        )
                    ]
                )
                
                fig.update_layout(
                    title=f"Presupuesto {mes} (base vs disponible)",
                    showlegend=True,
                    margin=dict(l=10, r=10, t=50, b=10),
                    annotations=[
                        dict(
                            text=f"Disponible<br>{presupuesto_disp:,.2f}€",
                            x=0.5,
                            y=0.5,
                            font_size=16,
                            showarrow=False,
                        )
                    ],
                )
                st.plotly_chart(fig, use_container_width=True)
    
            with c2:
                st.metric("🧾 Presupuesto disponible", f"{presupuesto_disp:,.2f} €")
                st.metric("💳 Gastado este mes", f"{gasto_mes:,.2f} €")
                st.metric("✅ Restante", f"{restante:,.2f} €")
                st.metric("% ejecutado", f"{pct_ejec:,.1f} %")
    
                # Opcional: mostrar también el presupuesto “bruto”
                if abs(presupuesto_mes - presupuesto_disp) > 1e-6:
                    st.caption(f"Presupuesto mes (bruto): {presupuesto_mes:,.2f} €")
    
            st.divider()
    
            st.subheader("Gastos del mes por categoría")
    
            df_cat = _gastos_mes_por_categoria(gastos, mes)
            if df_cat.empty:
                st.info("No hay gastos registrados para este mes (o no hay datos).")
            else:
                import plotly.express as px
    
                fig_cat = px.bar(df_cat, x="categoria", y="gasto", title=f"Gastos por categoría ({mes})")
                fig_cat.update_layout(margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig_cat, use_container_width=True)
    
                with st.expander("Ver tabla", expanded=False):
                    st.dataframe(df_cat, use_container_width=True)

        st.divider()
        st.subheader("Inversiones")
        
        if not isinstance(df_resumen, pd.DataFrame) or df_resumen.empty:
            st.info("No hay historial (df_resumen) para mostrar inversiones.")
        else:
            last = df_resumen.iloc[-1]
        
            invertido = float(last.get("Dinero Invertido", 0.0))
            pendiente = float(last.get("📈 Inversiones", 0.0))  # marcado para invertir
            patrimonio = float(last.get("total", 0.0))  # patrimonio según historial
        
            # % sobre patrimonio (evitar div/0)
            pct_invertido = (invertido / patrimonio * 100) if patrimonio else 0.0
            pct_pendiente = (pendiente / patrimonio * 100) if patrimonio else 0.0
        
            c1, c2 = st.columns([1.2, 1])
        
            with c1:
                import plotly.graph_objects as go
        
                fig_inv = go.Figure(
                    data=[
                        go.Pie(
                            labels=["Invertido", "Pendiente de invertir"],
                            values=[max(0.0, invertido), max(0.0, pendiente)],
                            hole=0.55,
                        )
                    ]
                )
                fig_inv.update_layout(
                    title="Distribución: invertido vs pendiente",
                    margin=dict(l=10, r=10, t=50, b=10),
                    annotations=[
                        dict(
                            text=f"Invertido<br>{pct_invertido:,.1f}%",
                            x=0.5,
                            y=0.5,
                            font_size=16,
                            showarrow=False,
                        )
                    ],
                )
                st.plotly_chart(fig_inv, use_container_width=True)
        
            with c2:
                st.metric("📌 Patrimonio (historial)", f"{patrimonio:,.2f} €")
                st.metric("✅ Invertido", f"{invertido:,.2f} €")
                st.metric("🕒 Pendiente de invertir", f"{pendiente:,.2f} €")
                st.metric("% invertido del patrimonio", f"{pct_invertido:,.1f} %")
                st.metric("% pendiente del patrimonio", f"{pct_pendiente:,.1f} %")
        
            # -------------------------
            # Intereses por año
            # -------------------------
            st.subheader("Intereses generados por año")
        
            if not isinstance(ingresos, pd.DataFrame) or ingresos.empty:
                st.info("No hay ingresos para calcular intereses.")
            else:
                df_int = ingresos.copy()
                if "fecha" not in df_int.columns or "cantidad" not in df_int.columns:
                    st.info("No encuentro columnas necesarias en ingresos (fecha/cantidad).")
                else:
                    # Usar tipo_logico si existe (tu pipeline lo crea)
                    if "tipo_logico" in df_int.columns:
                        mask = df_int["tipo_logico"].astype(str) == "Rendimiento Financiero"
                    else:
                        # fallback por si algún día no está tipo_logico:
                        mask = df_int.get("categoria", "").astype(str).str.lower() == "interes"
        
                    df_int = df_int.loc[mask].copy()
                    if df_int.empty:
                        st.info("No hay intereses registrados en los datos.")
                    else:
                        df_int["anio"] = pd.to_datetime(df_int["fecha"]).dt.year
                        por_anio = (
                            df_int.groupby("anio")["cantidad"]
                            .sum()
                            .reset_index()
                            .sort_values("anio")
                        )
        
                        import plotly.express as px
                        fig_int = px.bar(por_anio, x="anio", y="cantidad", title="Intereses por año")
                        fig_int.update_layout(yaxis_title="€", xaxis_title="Año", margin=dict(l=10, r=10, t=50, b=10))
                        st.plotly_chart(fig_int, use_container_width=True)
        
                        with st.expander("Ver tabla", expanded=False):
                            st.dataframe(por_anio, use_container_width=True)



        if df_resumen is not None and isinstance(df_resumen, pd.DataFrame) and not df_resumen.empty:
            st.subheader("Resumen mensual (df_resumen)")
            st.dataframe(df_resumen, use_container_width=True)
        else:
            st.info("No hay df_resumen disponible en 'historial'.")

    with tab_graficos:
        if not (isinstance(gastos, pd.DataFrame) and isinstance(ingresos, pd.DataFrame)) or gastos.empty or ingresos.empty:
            st.info("No hay datos suficientes (gastos/ingresos) para generar gráficos.")
        else:
            # 1) Balance mensual
            try:
                resumen = resumen_mensual(gastos, ingresos)
                fig_balance = plot_balance_mensual(resumen)
                st.plotly_chart(fig_balance, use_container_width=True)
            except Exception as e:
                st.warning(f"No se pudo generar 'Balance mensual': {e}")

            # 2) Porcentaje de ahorro mensual
            try:
                _, fig_ahorro = plot_porcentaje_ahorro(gastos, ingresos)
                st.plotly_chart(fig_ahorro, use_container_width=True)
            except Exception as e:
                st.warning(f"No se pudo generar 'Porcentaje de ahorro': {e}")

            # 3) Explicación de gastos
            try:
                _, stats, figs = plot_explicacion_gastos(
                    gastos,
                    ingresos,
                    categorias_fijas=["alimentacion", "transporte", "hosteleria", "entretenimiento", "otros"],
                )
                st.subheader("Explicación de gastos")
                if isinstance(figs, dict):
                    for _, fig in figs.items():
                        st.plotly_chart(fig, use_container_width=True)
                with st.expander("Estadísticas", expanded=False):
                    st.json({k: (v.to_dict() if hasattr(v, "to_dict") else v) for k, v in stats.items()})
            except Exception as e:
                st.warning(f"No se pudo generar 'Explicación de gastos': {e}")

    with tab_objetivos:
        st.subheader("Objetivos: panel visual")

        cfg = pd.DataFrame(out.get("objetivos", [])) if out else pd.DataFrame()
        if cfg.empty:
            st.info("No hay objetivos configurados.")
        else:
            # total asignado / disponible (también aquí, por claridad)
            total_fr = float(pd.to_numeric(cfg["fraccion_presupuesto"], errors="coerce").fillna(0).sum())
            st.write(f"Asignado: **{total_fr*100:.1f}%** | Disponible: **{max(0,1-total_fr)*100:.1f}%**")
            st.progress(min(max(total_fr,0.0),1.0))
        
            # Si hay objetivos_df, calcular "saldo actual" por objetivo
            saldo_actual = {}
            if isinstance(objetivos_df, pd.DataFrame) and not objetivos_df.empty:
                tmp = objetivos_df.copy()
                tmp["Mes"] = tmp["Mes"].astype(str)
                # última fila por objetivo (ordenando por Mes)
                tmp = tmp.sort_values(["Objetivo","Mes"])
                last_rows = tmp.groupby("Objetivo", as_index=False).tail(1)
                saldo_actual = dict(zip(last_rows["Objetivo"], last_rows["saldo_fin_mes"]))
        
            # Render de tarjetas
            for _, row in cfg.iterrows():
                nombre = str(row.get("nombre",""))
                et = str(row.get("etiquetas",""))
                fr = float(row.get("fraccion_presupuesto",0) or 0)
                dur = int(row.get("duracion_meses",0) or 0)
                mi = row.get("mes_inicio")
        
                with st.container(border=True):
                    c1, c2, c3 = st.columns([2,1,1])
                    c1.markdown(f"**{nombre}**")
                    c1.caption(et if et else "Sin etiquetas")
                    c2.metric("Asignación", f"{fr*100:.1f}%")
                    c3.metric("Duración", f"{dur} meses")
        
                    if mi:
                        st.caption(f"Inicio: {mi}")
        
                    if nombre in saldo_actual:
                        st.metric("Saldo acumulado (según movimientos)", f"{float(saldo_actual[nombre]):,.2f} €")
                        
                        
        st.subheader("Objetivos configurados")
        st.dataframe(pd.DataFrame(out.get("objetivos", [])), use_container_width=True)

        st.subheader("Evolución / resumen de objetivos (objetivos_df)")
        if objetivos_df is None or not isinstance(objetivos_df, pd.DataFrame) or objetivos_df.empty:
            st.info("No hay 'objetivos_df' disponible.")
        else:
            st.dataframe(objetivos_df, use_container_width=True)

    with tab_datos:
        st.subheader("Datos procesados del Excel")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Gastos**")
            if isinstance(gastos, pd.DataFrame):
                st.dataframe(gastos, use_container_width=True, height=400)
        with c2:
            st.markdown("**Ingresos**")
            if isinstance(ingresos, pd.DataFrame):
                st.dataframe(ingresos, use_container_width=True, height=400)
        with c3:
            st.markdown("**Transferencias**")
            if isinstance(transferencias, pd.DataFrame):
                st.dataframe(transferencias, use_container_width=True, height=400)

    with tab_herr:
        st.subheader("Saldo a una fecha")
        fecha_saldo = st.date_input("Fecha", value=pd.Timestamp.today().normalize())
        if st.button("Calcular saldo", key="btn_saldo_fecha"):
            try:
                saldo_total, saldos_dict = resumen_global(presupuesto, fecha_saldo)
                st.success(f"Saldo total a {pd.to_datetime(fecha_saldo).date()}: {saldo_total:,.2f} €")
                df_saldos = (
                    pd.DataFrame(list(saldos_dict.items()), columns=["Cuenta", "Saldo"])
                    .sort_values("Saldo", ascending=False)
                    .reset_index(drop=True)
                )
                st.dataframe(df_saldos, use_container_width=True)
            except Exception as e:
                st.exception(e)

    with tab_export:
        st.subheader("Exportar")
        st.caption("Descargas rápidas (CSV) y un Excel con varias hojas.")

        # CSVs
        c1, c2, c3 = st.columns(3)
        with c1:
            if isinstance(df_resumen, pd.DataFrame) and not df_resumen.empty:
                st.download_button(
                    "Descargar df_resumen (CSV)",
                    data=_df_to_csv_bytes(df_resumen),
                    file_name="df_resumen.csv",
                    mime="text/csv",
                )
        with c2:
            if isinstance(objetivos_df, pd.DataFrame) and not objetivos_df.empty:
                st.download_button(
                    "Descargar objetivos_df (CSV)",
                    data=_df_to_csv_bytes(objetivos_df),
                    file_name="objetivos_df.csv",
                    mime="text/csv",
                )
        with c3:
            if isinstance(balances_df, pd.DataFrame) and not balances_df.empty:
                st.download_button(
                    "Descargar saldos por cuenta (CSV)",
                    data=_df_to_csv_bytes(balances_df),
                    file_name="saldos_cuentas.csv",
                    mime="text/csv",
                )

        st.divider()

        # Excel multi-hoja
        hojas: Dict[str, pd.DataFrame] = {}
        if isinstance(df_resumen, pd.DataFrame) and not df_resumen.empty:
            hojas["df_resumen"] = df_resumen.copy()
        if isinstance(objetivos_df, pd.DataFrame) and not objetivos_df.empty:
            hojas["objetivos_df"] = objetivos_df.copy()
        if isinstance(balances_df, pd.DataFrame) and not balances_df.empty:
            hojas["saldos_cuentas"] = balances_df.copy()
        if isinstance(gastos, pd.DataFrame) and not gastos.empty:
            hojas["gastos"] = gastos.copy()
        if isinstance(ingresos, pd.DataFrame) and not ingresos.empty:
            hojas["ingresos"] = ingresos.copy()
        if isinstance(transferencias, pd.DataFrame) and not transferencias.empty:
            hojas["transferencias"] = transferencias.copy()

        if hojas:
            st.download_button(
                "Descargar todo (Excel)",
                data=_df_to_excel_bytes(hojas),
                file_name="finanzas_personales_export.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        else:
            st.info("No hay datos disponibles para exportar.")


if __name__ == "__main__":
    main()
