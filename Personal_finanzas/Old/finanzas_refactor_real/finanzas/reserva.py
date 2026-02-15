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
from typing import Union

hoy = str(pd.Timestamp.today().to_period('M'))

REGISTROS_DIR = Path(__file__).resolve().parent / "Data" / "Registros"
ARCHIVO_BASE_REGISTROS = REGISTROS_DIR / "Inicio.xlsx"
HOJAS_REGISTRO = ["Gastos", "Ingresos", "Transferencias"]
OBJETIVOS_VISTA_CONFIG_PATH = Path(__file__).resolve().parent / "Data" / "objetivos_vista.json"

def fondo_reserva_general(carpeta_data='Data', lectura=True, historial=None, fondo_cargado_actual=None):
    """Calcula la situación del fondo de reserva y actualiza su histórico.

    La función garantiza la existencia de la carpeta destino, lee (o crea) el
    fichero ``fondo_reserva.csv`` y devuelve la última instantánea disponible.
    Además corrige el flujo de actualización cuando no han transcurrido tres
    meses desde el último registro (antes duplicaba filas en cada llamada).
    """

    base_dir = Path(__file__).resolve().parent

    carpeta = Path(carpeta_data)
    if not carpeta.is_absolute():
        carpeta = base_dir / carpeta
    carpeta.mkdir(parents=True, exist_ok=True)

    archivo_csv = carpeta / 'fondo_reserva.csv'
    historial_path = base_dir / 'Data' / 'historial.csv'

    if historial is not None:
        historial_df = historial.copy()
    elif historial_path.exists():
        historial_df = pd.read_csv(historial_path)
    else:
        historial_df = None

    if historial_df is None:
        raise FileNotFoundError(
            "No se encontró un historial válido de cuentas virtuales para calcular el fondo de reserva."
        )

    if historial_df.empty:
        raise ValueError("El historial de cuentas virtuales está vacío; no se puede calcular el fondo de reserva.")

    if 'Mes' in historial_df.columns:
        gastos_disponibles = historial_df['Gasto del mes']
        fondo_historial = historial_df['Fondo de reserva cargado']
    else:
        gastos_disponibles = historial_df['Gasto del mes']
        fondo_historial = historial_df['Fondo de reserva cargado']
        
    media_gastos_mensuales = gastos_disponibles.mean()
    fondo_reserva = media_gastos_mensuales * 6
    
    if fondo_cargado_actual is not None:
        fondo_cargado = float(fondo_cargado_actual)
    elif not fondo_historial.empty:
        fondo_cargado = float(fondo_historial.iloc[-1])
    elif archivo_csv.exists():
        df_existente = pd.read_csv(archivo_csv)
        if df_existente.empty:
            fondo_cargado = 0.0
        else:
            fondo_cargado = float(df_existente.iloc[-1]['Cantidad cargada'])
    else:
        fondo_cargado = 0.0

    columnas_fondo = ['fecha de creacion', 'Cantidad del fondo', 'Cantidad cargada', 'Porcentaje']

    if archivo_csv.exists():
        df_fondo = pd.read_csv(archivo_csv)
        if list(df_fondo.columns) != columnas_fondo:
            df_fondo.columns = columnas_fondo
        
        if not df_fondo.empty:
            df_fondo['fecha de creacion'] = pd.to_datetime(df_fondo['fecha de creacion'])
        else:
            df_fondo = pd.DataFrame(
                {
                    'fecha de creacion': pd.to_datetime([datetime.now()]),
                    'Cantidad del fondo': [fondo_reserva],
                    'Cantidad cargada': [fondo_cargado],
                    'Porcentaje': [0.0],
                }
            )
    else:
        df_fondo = pd.DataFrame(
            {
                'fecha de creacion': pd.to_datetime([datetime.now()]),
                'Cantidad del fondo': [fondo_reserva],
                'Cantidad cargada': [fondo_cargado],
                'Porcentaje': [0.0],
            }
        )

    hoy = pd.Timestamp(datetime.now())
    if df_fondo.empty:
        ultima_fecha = None
        fondo_reserva_ultima_fecha = 0.0
    else:
        ultima_fecha = df_fondo['fecha de creacion'].max()
        fondo_reserva_ultima_fecha = df_fondo.loc[
            df_fondo['fecha de creacion'].idxmax(), 'Cantidad del fondo'
        ]

    porcentaje = 0.0 if fondo_reserva == 0 else fondo_cargado / fondo_reserva

    if ultima_fecha is None or hoy - ultima_fecha >= timedelta(days=90):
        nuevo_registro = pd.DataFrame(
            {
                'fecha de creacion': [hoy],
                'Cantidad del fondo': [fondo_reserva],
                'Cantidad cargada': [fondo_cargado],
                'Porcentaje': [porcentaje],
            }
        )
        df_fondo = pd.concat([df_fondo, nuevo_registro], ignore_index=True)
    else:
        idx_ultima = df_fondo['fecha de creacion'].idxmax()
        df_fondo.loc[idx_ultima, 'Cantidad del fondo'] = fondo_reserva
        df_fondo.loc[idx_ultima, 'Cantidad cargada'] = fondo_cargado
        df_fondo.loc[idx_ultima, 'Porcentaje'] = porcentaje

    df_fondo = df_fondo.sort_values('fecha de creacion').reset_index(drop=True)
    df_fondo.to_csv(archivo_csv, index=False)

    registro_actual = df_fondo.iloc[-1]
    if lectura:
        print('Última actualización:', registro_actual['fecha de creacion'].strftime('%d-%m-%Y'))
        print(f"Cantidad del fondo requerida: {round(registro_actual['Cantidad del fondo'])}€")
    return registro_actual
