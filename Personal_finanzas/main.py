import streamlit as st
import pandas as pd
from pathlib import Path

# Importa tus funciones (cuando las muevas a módulos)
# from io_data import fusionar_archivos_registro, cargar_objetivos_vista, guardar_objetivos_vista
# from logic import (
#   procesar_ingresos, calcular_historial_cuentas_virtuales,
#   plot_balance_mensual, plot_porcentaje_ahorro, grafico_presupuesto_1
# )

st.set_page_config(page_title="Finanzas personales", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
HIST_PATH = DATA_DIR / "historial.csv"
OBJ_PATH = DATA_DIR / "objetivos_vista.json"

st.title("📊 Finanzas personales (Streamlit)")

# -------- Helpers de caché --------
@st.cache_data
def load_historial_csv(path: Path):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

# -------- Sidebar --------
st.sidebar.header("Configuración")

porc_gasto = st.sidebar.slider("Porcentaje gasto", 0.0, 1.0, 0.30, 0.01)
porc_inversion = st.sidebar.slider("Porcentaje inversión", 0.0, 1.0, 0.10, 0.01)
porc_vacaciones = st.sidebar.slider("Porcentaje vacaciones", 0.0, 1.0, 0.05, 0.01)

st.sidebar.divider()

confirm_fusion = st.sidebar.checkbox("Confirmo que quiero fusionar registros")
btn_fusionar = st.sidebar.button("Fusionar excels de registros")

btn_recalcular = st.sidebar.button("Recalcular historial")

# -------- Acciones (aquí llamarías a tus funciones reales) --------
if btn_fusionar:
    if not confirm_fusion:
        st.error("Marca la confirmación antes de fusionar.")
    else:
        # fusionar_archivos_registro()  # <- cuando lo tengas en io_data.py
        st.success("Fusión realizada (aquí llamas a fusionar_archivos_registro).")
        st.cache_data.clear()

if btn_recalcular:
    # df_resumen = calcular_historial_cuentas_virtuales(..., output_path=None)
    # guardar_historial(df_resumen, HIST_PATH)
    st.success("Historial recalculado (aquí llamas a tu cálculo).")
    st.cache_data.clear()

df_hist = load_historial_csv(HIST_PATH)

if df_hist.empty:
    st.info("No hay historial aún. Fusiona y/o recalcula para generar datos.")
    st.stop()

# -------- Tabs --------
tab1, tab2, tab3, tab4 = st.tabs(["Resumen", "Presupuesto mes", "Gastos", "Objetivos"])

with tab1:
    st.subheader("Resumen")
    st.dataframe(df_hist.tail(24), use_container_width=True)

    # Ejemplo: plot balance mensual (cuando tu función devuelva fig)
    # fig = plot_balance_mensual(...)
    # st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Presupuesto del mes")
    # fig = grafico_presupuesto_1(df_hist)
    # if fig:
    #     st.plotly_chart(fig, use_container_width=True)
    st.write("Aquí va tu donut de presupuesto (grafico_presupuesto_1).")

with tab3:
    st.subheader("Gastos")
    st.write("Aquí pondrías tus gráficos de explicación de gastos (robustos) devolviendo figs.")

with tab4:
    st.subheader("Objetivos")

    # --- Gestor de objetivos en Streamlit ---
    # objetivos = cargar_objetivos_vista(OBJ_PATH)
    objetivos = []  # <- sustituye por tu loader real

    if "objetivos" not in st.session_state:
        st.session_state.objetivos = objetivos

    colA, colB = st.columns([1, 2], gap="large")

    with colA:
        st.caption("Lista de objetivos")
        nombres = [o["nombre"] for o in st.session_state.objetivos] if st.session_state.objetivos else []
        seleccionado = st.selectbox("Selecciona", ["(nuevo)"] + nombres)

    with colB:
        st.caption("Editar / crear")
        if seleccionado == "(nuevo)":
            obj = {"nombre": "", "etiquetas": [], "porcentaje_ingreso": 0.0, "saldo_inicial": 0.0,
                   "objetivo_total": None, "horizonte_meses": None, "mes_inicio": None}
        else:
            obj = next(o for o in st.session_state.objetivos if o["nombre"] == seleccionado)

        with st.form("form_objetivo", clear_on_submit=False):
            nombre = st.text_input("Nombre", value=obj.get("nombre",""))
            etiquetas = st.text_input("Etiquetas (coma)", value=", ".join(obj.get("etiquetas", [])))
            porcentaje = st.number_input("% ingreso (0-1)", min_value=0.0, max_value=1.0, step=0.01, value=float(obj.get("porcentaje_ingreso",0.0)))
            saldo_inicial = st.number_input("Saldo inicial", value=float(obj.get("saldo_inicial",0.0)))
            objetivo_total = st.text_input("Meta (opcional)", value="" if obj.get("objetivo_total") is None else str(obj["objetivo_total"]))
            horizonte = st.text_input("Horizonte meses (opcional)", value="" if obj.get("horizonte_meses") is None else str(obj["horizonte_meses"]))
            mes_inicio = st.text_input("Mes inicio AAAA-MM (opcional)", value="" if obj.get("mes_inicio") is None else str(obj["mes_inicio"]))

            guardar = st.form_submit_button("Guardar en sesión")

        if guardar:
            et = [e.strip().lower() for e in etiquetas.split(",") if e.strip()]
            nuevo = {
                "nombre": nombre.strip(),
                "etiquetas": et,
                "porcentaje_ingreso": float(porcentaje),
                "saldo_inicial": float(saldo_inicial),
                "objetivo_total": float(objetivo_total) if objetivo_total.strip() else None,
                "horizonte_meses": int(horizonte) if horizonte.strip() else None,
                "mes_inicio": mes_inicio.strip() or None,
            }
            if not nuevo["nombre"]:
                st.error("El objetivo necesita nombre.")
            else:
                # reemplaza o añade
                st.session_state.objetivos = [o for o in st.session_state.objetivos if o["nombre"] != nuevo["nombre"]] + [nuevo]
                st.success("Guardado en sesión.")

    st.divider()
    if st.button("Guardar objetivos en disco"):
        # guardar_objetivos_vista(st.session_state.objetivos, OBJ_PATH)
        st.success("Guardados en disco (aquí llamas a guardar_objetivos_vista).")