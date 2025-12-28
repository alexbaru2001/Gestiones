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

def cargar_saldos_iniciales(presupuesto, cuentas_obj, saldos_iniciales):
    for nombre_cuenta, saldo in saldos_iniciales.items():
        cuenta = cuentas_obj.get(nombre_cuenta)
        if not cuenta:
            raise ValueError(f"La cuenta '{nombre_cuenta}' no existe en el presupuesto.")
       
        transaccion_inicial = Transaction(
            amount=saldo,
            description="Saldo inicial",
            category="Ajuste inicial",
            account=cuenta,
            timestamp=pd.to_datetime("2000-01-01")  # Fecha muy antigua para que quede al principio
        )
        presupuesto.add_transaction(transaccion_inicial)

def cargar_transacciones(df,cuentas_obj, presupuesto,signo=1):
    for _, fila in df.iterrows():
        monto = signo * fila['cantidad']
        cuenta = cuentas_obj[fila['cuenta']]
        transaccion = Transaction(
            amount=monto,
            description=str(fila['comentario']),
            category=str(fila['categoria']),
            account=cuenta,
            timestamp=fila['fecha']
        )
        presupuesto.add_transaction(transaccion)
