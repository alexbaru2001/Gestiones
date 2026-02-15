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

def predecir_total(df_resumen, variable):
    # Asegurar índice de fechas correcto
    df_resumen['Mes'] = pd.to_datetime(df_resumen['Mes'])
    df_resumen = df_resumen.set_index('Mes').sort_index()
   
    # Serie temporal de interés
    serie_total = df_resumen[variable]
   
    # Ajustar modelo ARIMA(p,d,q)
    modelo = ARIMA(serie_total, order=(1,1,1))  # parámetros a ajustar
    resultado = modelo.fit()
   
    # Predecir próximos 6 meses
    forecast = resultado.get_forecast(steps=6)
    predicciones = forecast.predicted_mean
    conf_int = forecast.conf_int()
   
    # Mostrar
    plt.figure(figsize=(10,5))
    plt.plot(serie_total, label='Histórico')
    plt.plot(predicciones.index, predicciones, label='Predicción', color='orange')
    plt.fill_between(predicciones.index,
                     conf_int.iloc[:,0],
                     conf_int.iloc[:,1], color='orange', alpha=0.2)
    plt.legend()
    plt.title("Predicción de 'total' para próximos meses")
    print(predicciones)
