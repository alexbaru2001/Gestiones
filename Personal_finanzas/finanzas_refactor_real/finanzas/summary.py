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

def resumen_global(presupuesto, fecha):
    fecha_objetivo = pd.to_datetime(fecha)
    saldos_a_fecha = presupuesto.balances_a_fecha(fecha_objetivo)
    saldo_total = presupuesto.balance_total_a_fecha(fecha_objetivo)
    print(f"💰 Saldo total al {fecha_objetivo.date()}: {round(saldo_total, 2)}")
    print(f"Saldos al {fecha_objetivo.date()}:")
    for nombre, saldo in saldos_a_fecha.items():
        print(f"Cuenta: {nombre}, Saldo: {round(saldo, 2)}")
