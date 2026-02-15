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

def clasificar_ingreso(fila):
    categoria = fila['categoria']
    etiquetas = str(fila.get('etiquetas', '')).lower()

    if categoria == 'Salario':
        if 'abuelos' in etiquetas:
            return 'Vacaciones'
        if 'mi regalo' in etiquetas:
            return 'Regalos'
        return 'Ingreso Real'
   
   
    elif categoria == 'Interes':
        return 'Rendimiento Financiero'
   
    elif categoria in ['Transporte', 'Hosteleria', 'Entretenimiento', 'Alimentacion', 'Otros']:
        # Esto es una devolución por gastos compartidos
        if 'mi regalo' in etiquetas:
            return 'Regalos'
        elif 'fondo reserva' in etiquetas:
            return 'Fondo reserva'
        elif 'vacaciones' in etiquetas:
            return 'Vacaciones'
        return f"Reembolso {categoria}"
   
    else:
        return 'Ingreso No Clasificado'

def procesar_ingresos(df_ingresos):
    df_ingresos = df_ingresos.copy()
    df_ingresos['tipo_logico'] = df_ingresos.apply(clasificar_ingreso, axis=1)
    return df_ingresos

def calcular_gastos_netos(df_gastos, df_ingresos):
    gastos_por_cat = df_gastos.groupby('categoria')['cantidad'].sum()
    reembolsos = df_ingresos[df_ingresos['tipo_logico'].str.startswith('Reembolso')]
    reembolsos['categoria_reembolso'] = reembolsos['tipo_logico'].str.replace('Reembolso ', '', regex=False)
    reembolsos_por_cat = reembolsos.groupby('categoria_reembolso')['cantidad'].sum()
    gastos_netos = gastos_por_cat.subtract(reembolsos_por_cat, fill_value=0)
    return gastos_netos

def resumen_ingresos(df_ingresos):
    return df_ingresos[df_ingresos['tipo_logico'] == 'Ingreso Real']['cantidad'].sum()

def resumen_gastos(df_gastos, df_ingresos):
    df_gastos_explicacion = df_gastos.copy()
    df_ingresos_explicacion = df_ingresos.copy()
   
    df_gastos_explicacion['mes'] = df_gastos_explicacion['fecha'].dt.to_period('M').astype(str)
    df_ingresos_explicacion['mes'] = df_ingresos_explicacion['fecha'].dt.to_period('M').astype(str)
   
    gastos_mensuales_explicacion = df_gastos_explicacion.groupby(['mes', 'tipo_logico'])['cantidad'].sum().rename('gastos').reset_index()
    ingresos_mensuales_explicacion = df_ingresos_explicacion.groupby(['mes', 'tipo_logico'])['cantidad'].sum().rename('ingresos').reset_index()
   
    df_explicacion = pd.merge(ingresos_mensuales_explicacion, gastos_mensuales_explicacion, on=['mes', 'tipo_logico'], how='outer')
    df_explicacion = df_explicacion.fillna(0)
   
    df_explicacion['grupo'] = df_explicacion['tipo_logico'].apply(
        lambda x: 'Ingreso Real' if x == 'Ingreso Real' else ('Reembolso' if str(x).startswith('Reembolso') else 'Otro')
    )
   
    df_explicacion = df_explicacion.loc[df_explicacion['grupo'].isin([ 'Reembolso'])]
   
    df_explicacion['balance'] = df_explicacion['gastos'] - df_explicacion['ingresos']
    # Elimina las columnas de ingresos y gastos
    df_explicacion = df_explicacion.drop(['ingresos', 'gastos', 'grupo'], axis=1)
   
    pivot_explicacion = df_explicacion.pivot_table(index='mes', columns='tipo_logico', values='balance')
    pivot_explicacion.columns = ['alimentacion', 'entretenimiento', 'hosteleria', 'otros', 'transporte']
    pivot_explicacion = pivot_explicacion.fillna(0)
    return pivot_explicacion

def resumen_mensual(df_gastos, df_ingresos):
    df_gastos = df_gastos.copy()
    df_ingresos = df_ingresos.copy()
   
    df_gastos['mes'] = df_gastos['fecha'].dt.to_period('M').astype(str)
    df_ingresos['mes'] = df_ingresos['fecha'].dt.to_period('M').astype(str)
   
    gastos_mensuales = df_gastos.groupby(['mes', 'tipo_logico'])['cantidad'].sum().rename('gastos').reset_index()
    ingresos_mensuales = df_ingresos.groupby(['mes', 'tipo_logico'])['cantidad'].sum().rename('ingresos').reset_index()
   

    df_resumen = pd.merge(ingresos_mensuales, gastos_mensuales, on=['mes', 'tipo_logico'], how='outer')
    df_resumen = df_resumen.fillna(0)
   
    df_resumen['grupo'] = df_resumen['tipo_logico'].apply(
        lambda x: 'Ingreso Real' if x == 'Ingreso Real' else ('Reembolso' if str(x).startswith('Reembolso') else 'Otro')
    )
   
    df_resumen = df_resumen.loc[df_resumen['grupo'].isin(['Ingreso Real', 'Reembolso'])]
    df_resumen = df_resumen.groupby(['mes', 'grupo'])[['ingresos', 'gastos']].sum().reset_index()
    # Calcula el balance
    df_resumen['balance'] = abs(df_resumen['ingresos'] - df_resumen['gastos'])
   
    # Elimina las columnas de ingresos y gastos
    df_resumen = df_resumen.drop(['ingresos', 'gastos'], axis=1)
   
    # Crea la tabla dinámica
    pivot = df_resumen.pivot_table(index='mes', columns='grupo', values='balance', aggfunc='sum')
   
    # Opcional: renombra las columnas si lo deseas
    pivot = pivot.rename(columns={'Ingreso Real': 'ingresos', 'Reembolso': 'gastos'})
   
    pivot['balance'] = pivot['ingresos'] - pivot['gastos']
    return pivot

def eliminar_tildes(texto):
    if isinstance(texto, str):  
        return unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('utf-8')
    return texto

def agregar_fila_ingresos(dataframe, val1, val2, val3, val4, val5, val6):
    nueva_fila = pd.DataFrame([{'fecha': val1, 'categoria': val2, 'cuenta': val3, 'cantidad': val4, 'etiquetas': val5, 'comentario': val6}])
    return pd.concat([dataframe, nueva_fila], ignore_index=True)
