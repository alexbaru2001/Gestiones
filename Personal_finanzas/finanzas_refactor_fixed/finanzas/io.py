# -*- coding: utf-8 -*-
"""Paquete de finanzas personales (refactor).
Separación de responsabilidades: IO, lógica (engine), gráficos.
Preparado para una futura capa Streamlit.
"""
import json
import os
import re
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import unicodedata
import plotly.express as px
# from financier import Account, Transaction, Budget
import warnings
warnings.filterwarnings('ignore')
import plotly.graph_objects as go
from datetime import datetime
from collections import defaultdict
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from pathlib import Path

from dataclasses import dataclass

def get_project_root() -> Path:
    # finanzas/ -> project root
    return Path(__file__).resolve().parents[1]

def get_data_dir() -> Path:
    # Allows overriding data folder (useful for future Streamlit and different machines)
    env = os.getenv("FINANZAS_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return get_project_root() / "Data"

def resolve_existing_path(*candidates: Path) -> Path:
    for p in candidates:
        if p is None:
            continue
        try:
            if p.exists():
                return p
        except OSError:
            continue
    # Return first candidate as default (even if missing) for clearer errors upstream
    return candidates[0]

from typing import Union

hoy = str(pd.Timestamp.today().to_period('M'))

REGISTROS_DIR = Path(__file__).resolve().parent / "Data" / "Registros"
ARCHIVO_BASE_REGISTROS = REGISTROS_DIR / "Inicio.xlsx"
HOJAS_REGISTRO = ["Gastos", "Ingresos", "Transferencias"]
OBJETIVOS_VISTA_CONFIG_PATH = Path(__file__).resolve().parent / "Data" / "objetivos_vista.json"

def _limpiar_dataframe_registro(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza un DataFrame de registros para facilitar la fusión."""
    df = df.dropna(how="all")

    if "Fecha y hora" in df.columns:
        df["Fecha y hora"] = pd.to_datetime(df["Fecha y hora"], errors="coerce")
        df = df.dropna(subset=["Fecha y hora"])
        df = df.sort_values("Fecha y hora", ascending=False)

    # No usamos ``drop_duplicates`` porque hay movimientos reales (p.ej. traspasos fraccionados)
    # que comparten exactamente la misma información en todas las columnas.
    return df.reset_index(drop=True)

def fusionar_archivos_registro() -> None:
    """Fusiona los registros descargados con el archivo base ``Inicio.xlsx``."""
    if not ARCHIVO_BASE_REGISTROS.exists():
        raise FileNotFoundError(
            "No se encontró el archivo base 'Inicio.xlsx'.\n"
            f"Probé: {ARCHIVO_BASE_REGISTROS}\n"
            "Soluciones: (1) crea una carpeta 'Data' junto al proyecto y pon ahí Inicio.xlsx; "
            "(2) o exporta FINANZAS_DATA_DIR apuntando a tu carpeta de datos."
        )

    dataframes = {}
    columnas_por_hoja = {}

    for hoja in HOJAS_REGISTRO:
        df_base = pd.read_excel(ARCHIVO_BASE_REGISTROS, sheet_name=hoja)
        columnas_por_hoja[hoja] = df_base.columns
        dataframes[hoja] = _limpiar_dataframe_registro(df_base)

    if not REGISTROS_DIR.exists():
        # No hay carpeta de registros descargados; nada que fusionar.
        return

    archivos_a_fusionar = sorted(
        archivo
        for archivo in REGISTROS_DIR.glob("*.xlsx")
        if archivo != ARCHIVO_BASE_REGISTROS
    )

    if not archivos_a_fusionar:
        return

    for archivo in archivos_a_fusionar:
        for hoja in HOJAS_REGISTRO:
            df_temp = pd.read_excel(archivo, sheet_name=hoja, skiprows=1)
            df_temp = df_temp.reindex(columns=columnas_por_hoja[hoja])
            df_temp = _limpiar_dataframe_registro(df_temp)

            if df_temp.empty:
                continue

            combinado = pd.concat([dataframes[hoja], df_temp], ignore_index=True)
            dataframes[hoja] = _limpiar_dataframe_registro(combinado)

    with pd.ExcelWriter(ARCHIVO_BASE_REGISTROS, engine="openpyxl") as writer:
        for hoja, df in dataframes.items():
            df.to_excel(writer, sheet_name=hoja, index=False)

    for archivo in archivos_a_fusionar:
        archivo.unlink()
