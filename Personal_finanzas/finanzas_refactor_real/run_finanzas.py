# -*- coding: utf-8 -*-
"""Paquete de finanzas personales (refactor).
Separación de responsabilidades: IO, lógica (engine), gráficos.
Preparado para una futura capa Streamlit.
"""

from pathlib import Path
import pandas as pd

from finanzas.io import fusionar_archivos_registro
from finanzas.budget_io import cargar_saldos_iniciales, cargar_transacciones
from finanzas.domain import Budget
from finanzas.engine import crear_historial_cuentas_virtuales_puro

def main():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "Data"
    excel_path = data_dir / "Inicio.xlsx"
    historial_path = data_dir / "historial.csv"
    fondo_reserva_path = data_dir / "fondo_reserva.csv"
    objetivos_path = data_dir / "objetivos_vista.json"

    # 1) Fusiona descargas en el excel base (si procede)
    # fusionar_archivos_registro()  # descomenta si lo necesitas

    # 2) Carga datos desde el Excel
    xls = pd.ExcelFile(excel_path)
    saldos_df = pd.read_excel(xls, "Saldos_iniciales")
    trans_df = pd.read_excel(xls, "Registros")

    budget = Budget()
    cargar_saldos_iniciales(budget, saldos_df)
    cargar_transacciones(budget, trans_df)

    # 3) Calcula histórico (modo puro: no escribe ficheros)
    df_hist = crear_historial_cuentas_virtuales_puro(
        budget=budget,
        archivo_fondo_reserva=fondo_reserva_path,
        ruta_objetivos=objetivos_path,
        output_path=None
    )

    # 4) Guardado explícito
    df_hist.to_csv(historial_path, index=False)
    print(f"Historial guardado en: {historial_path}")

if __name__ == "__main__":
    main()
