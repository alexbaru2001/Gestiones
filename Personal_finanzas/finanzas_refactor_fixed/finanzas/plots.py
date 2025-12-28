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

def plot_balance_mensual(resumen):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=resumen.index, y=resumen['ingresos'], name='Ingresos', marker_color='green'))
    fig.add_trace(go.Bar(x=resumen.index, y=-resumen['gastos'], name='Gastos', marker_color='red'))
    fig.add_trace(go.Scatter(x=resumen.index, y=resumen['balance'], name='Balance Neto', line=dict(color='blue', width=3)))

    fig.update_layout(
        title="💵 Balance mensual",
        barmode='relative',
        xaxis_title='Mes',
        yaxis_title='Cantidad (€)',
        legend_title='Tipo',
        template='plotly_white'
    )
    return fig

def plot_explicacion_gastos(df_gastos, df_ingresos):
   
    pivot_explicacion = resumen_gastos(df_gastos, df_ingresos)


    # =====================
    # 2. KPIs y estadísticas
    # =====================
    print("Promedio mensual por categoría:")
    print(pivot_explicacion.mean())
   
    print("\nCategoría más volátil (desviación estándar):")
    print(pivot_explicacion.std())
   
    print("\nMes con mayor gasto total:")
    print(pivot_explicacion.sum(axis=1).idxmax())
   
    # =====================
    # 3. Gráfico de barras apiladas (valores absolutos)
    # =====================
    df_reset = pivot_explicacion.reset_index().rename(columns={"index": "Mes"})
    fig1 = px.bar(
        df_reset,
        x="mes",
        y=pivot_explicacion.columns,
        title="Gastos mensuales por categoría",
        labels={"value": "Gasto (€)", "variable": "Categoría"},
    )
    # ======================
    # Añadimos línea discontinua de la media
    # ======================
    # Calcular gasto total mensual
    df_reset["Total"] = df_reset[pivot_explicacion.columns].sum(axis=1)
   
    # Media de gasto mensual
    media_gasto = df_reset["Total"].mean()
   
    # Añadir línea de puntos
    fig1.add_trace(
        go.Scatter(
            x=df_reset["mes"],
            y=[media_gasto]*len(df_reset),
            mode="lines+markers",
            name="Media mensual",
            line=dict(dash="dot", color="black"),
            marker=dict(symbol="circle", size=6)
        )
    )
    fig1.show()
   
    # =====================
    # 4. Gráfico de porcentajes (distribución relativa)
    # =====================
    fig2 = px.bar(
        df_reset,
        x="mes",
        y=pivot_explicacion.columns,
        title="Distribución porcentual de gastos",
        labels={"value": "Proporción", "variable": "Categoría"},
        barmode="relative"
    )
    fig2.show()
   
    # =====================
    # 5. Evolución temporal por categoría (línea)
    # =====================
    fig3 = px.line(
        df_reset,
        x="mes",
        y=pivot_explicacion.columns,
        title="Evolución de gastos por categoría"
    )
    fig3.show()
   
    # =====================
    # 6. Boxplot (dispersión por categoría)
    # =====================
    df_melt = df_reset.melt(id_vars="mes", var_name="Categoría", value_name="Gasto")
    fig4 = px.box(df_melt, x="Categoría", y="Gasto", title="Distribución de gastos por categoría")
    fig4.show()
   
    # =====================
    # 7. Heatmap (intensidad de gasto)
    # =====================
    fig5 = go.Figure(data=go.Heatmap(
        z=pivot_explicacion.values,
        x=pivot_explicacion.columns,
        y=pivot_explicacion.index,
        colorscale="Blues"
    ))
    fig5.update_layout(title="Mapa de calor de gastos")
    fig5.show()
   
    # =====================
    # 8. Insights básicos
    # =====================
    total_mes = pivot_explicacion.sum(axis=1)
    print("\nResumen rápido:")
    print(f"Mes con mayor gasto total: {total_mes.idxmax()} ({total_mes.max()} €)")
    print(f"Mes con menor gasto total: {total_mes.idxmin()} ({total_mes.min()} €)")
   
    prop_fijo = pivot_explicacion[["alimentacion", "transporte"]].sum(axis=1) / total_mes * 100
    print("\n% promedio de gasto fijo (alimentación + transporte):", round(prop_fijo.mean(),2), "%")

def plot_porcentaje_ahorro(gastos, ingresos):
    resumen = resumen_mensual(gastos, ingresos)
    resumen = resumen[resumen['ingresos'] > 0]
    resumen['porcentaje_ahorro'] = ((resumen['ingresos'] - resumen['gastos']) / resumen['ingresos']) * 100

    fig = px.line(
        resumen,
        x=resumen.index,
        y='porcentaje_ahorro',
        title='📊 Porcentaje de ahorro mensual',
        labels={'porcentaje_ahorro': '% Ahorro', 'index': 'Mes'},
        markers=True
    )
    fig.update_layout(template='plotly_white', yaxis_ticksuffix="%")
    return fig

def grafico_presupuesto(df_resumen: pd.DataFrame):
    # Convertir columna Mes a datetime si no lo está
    df_resumen = df_resumen.copy()
    df_resumen['Mes'] = pd.to_datetime(df_resumen['Mes'])

    # Último mes disponible en el dataframe
    ultimo_mes = df_resumen['Mes'].max().to_period('M')
    mes_actual = pd.Timestamp.today().to_period('M')

    if ultimo_mes != mes_actual:
        print(f"⚠️ El último mes en df_resumen es {ultimo_mes}, pero el mes actual es {mes_actual}. No se genera gráfico.")
        return None

    # Seleccionar fila del último mes
    fila = df_resumen.loc[df_resumen['Mes'].dt.to_period('M') == ultimo_mes].iloc[0]

    gasto_mes = fila['Gasto del mes']
    presupuesto_mes = fila['Presupuesto Mes']
    dinero_restante = presupuesto_mes - gasto_mes

    if dinero_restante < 0:
        dinero_restante = 0  # no puede haber "dinero negativo" disponible

    # Calcular semanas restantes en el mes
    hoy = pd.Timestamp.today()
    fin_mes = hoy + pd.offsets.MonthEnd(0)
    dias_restantes = (fin_mes - hoy).days + 1
    semanas_restantes = max(1, dias_restantes // 7)  # al menos 1 semana
    gasto_semanal = dinero_restante / semanas_restantes

    print(f"💡 Dinero restante este mes: {dinero_restante:.2f}")
    print(f"👉 Puedes gastar aproximadamente {gasto_semanal:.2f} por semana durante las {semanas_restantes} semanas restantes de {ultimo_mes}.")

    # Gráfico circular con Plotly
    fig = go.Figure(data=[
        go.Pie(
            labels=['Gastado', 'Disponible'],
            values=[gasto_mes, max(0, presupuesto_mes - gasto_mes)],
            hole=0.4,
            marker=dict(colors=['#EF553B', '#00CC96'])
        )
    ])
    fig.update_layout(
        title=f"Presupuesto de {ultimo_mes} - Gastado vs Disponible",
        annotations=[dict(text=f"Restante\n{dinero_restante:.2f}€", x=0.5, y=-0.2, font_size=14, showarrow=False)]
    )
    return fig

def graficar_gasto_vs_media(gastos,ingresos):
    df_gastos = resumen_gastos(gastos,ingresos)
   
    # Asegurarnos que el índice es periodo mensual
    if not isinstance(df_gastos.index, pd.PeriodIndex):
        df_gastos.index = pd.to_datetime(df_gastos.index).to_period('M')

    # Último mes disponible en el dataframe
    ultimo_mes = df_gastos.index.max()

    # Mes actual en formato periodo mensual
    mes_actual = pd.Timestamp.today().to_period('M')

    # Verificar si coincide
    if ultimo_mes != mes_actual:
        print(f"⚠️ El último mes en los datos es {ultimo_mes}, no coincide con el mes actual {mes_actual}. No se genera gráfico.")
        return None

    # Datos del último mes
    gastos_ultimo_mes = df_gastos.loc[ultimo_mes]

    # Media histórica de cada categoría
    media_por_categoria = df_gastos.mean()

    # Crear dataframe para comparación
    df_plot = pd.DataFrame({
        "Categoria": gastos_ultimo_mes.index,
        "Gasto_ultimo_mes": gastos_ultimo_mes.values,
        "Media_historica": media_por_categoria.values
    })

    # Cálculo de diferencia
    df_plot["Diferencia"] = df_plot["Media_historica"] - df_plot["Gasto_ultimo_mes"]

    # Gráfico de barras
    fig = px.bar(
        df_plot,
        x="Categoria",
        y=["Gasto_ultimo_mes", "Media_historica"],
        barmode="group",
        title=f"Gastos de {ultimo_mes.strftime('%Y-%m')} vs Media histórica"
    )

    return fig

    return df_plot

def grafico_presupuesto_1(df_resumen: pd.DataFrame):
    # Copia del DataFrame
    df_resumen = df_resumen.copy()
    df_resumen['Mes'] = pd.to_datetime(df_resumen['Mes'])

    # Fechas clave
    ultimo_mes = df_resumen['Mes'].max().to_period('M')
    mes_actual = pd.Timestamp.today().to_period('M')

    # Si el mes actual no está en los datos, usamos el último disponible
    if mes_actual not in df_resumen['Mes'].dt.to_period('M').unique():
        print(f"⚠️ No hay datos para {mes_actual}. Se usará el último mes disponible: {ultimo_mes}.")
        mes_para_grafico = ultimo_mes
    else:
        mes_para_grafico = mes_actual

    # Seleccionar la fila del mes elegido
    fila = df_resumen.loc[df_resumen['Mes'].dt.to_period('M') == mes_para_grafico].iloc[0]

    gasto_mes = fila['Gasto del mes']
    presupuesto_mes = fila['Presupuesto Mes']
    dinero_restante = max(0, presupuesto_mes - gasto_mes)

    # Calcular semanas restantes del mes actual (no del último mes de datos)
    hoy = pd.Timestamp.today()
    fin_mes = hoy + pd.offsets.MonthEnd(0)
    dias_restantes = (fin_mes - hoy).days + 1
    semanas_restantes = max(1, dias_restantes // 7)
    gasto_semanal = dinero_restante / semanas_restantes if semanas_restantes else 0

    print(f"💡 Dinero restante este mes ({mes_actual}): {dinero_restante:.2f}")
    print(f"👉 Puedes gastar aprox. {gasto_semanal:.2f} €/semana durante {semanas_restantes} semanas restantes de {mes_actual}.")

    # Gráfico circular
    fig = go.Figure(data=[
        go.Pie(
            labels=['Gastado', 'Disponible'],
            values=[gasto_mes, max(0, presupuesto_mes - gasto_mes)],
            hole=0.4,
            marker=dict(colors=['#EF553B', '#00CC96'])
        )
    ])
    fig.update_layout(
        title=f"Presupuesto de {mes_actual} (datos hasta {mes_para_grafico})",
        annotations=[dict(text=f"Restante\n{dinero_restante:.2f}€", x=0.5, y=-0.2, font_size=14, showarrow=False)]
    )
    return fig

def graficar_gasto_vs_media_1(gastos, ingresos):
    df_gastos = resumen_gastos(gastos, ingresos)

    # Asegurar índice mensual
    if not isinstance(df_gastos.index, pd.PeriodIndex):
        df_gastos.index = pd.to_datetime(df_gastos.index).to_period('M')

    ultimo_mes = df_gastos.index.max()
    mes_actual = pd.Timestamp.today().to_period('M')

    # Usar último mes si el actual no está disponible
    if mes_actual not in df_gastos.index:
        print(f"⚠️ No hay datos para {mes_actual}. Se usará el último mes disponible: {ultimo_mes}.")
        mes_para_grafico = ultimo_mes
    else:
        mes_para_grafico = mes_actual

    # Datos del mes elegido
    gastos_ultimo_mes = df_gastos.loc[mes_para_grafico]
    media_por_categoria = df_gastos.mean()

    # DataFrame para graficar
    df_plot = pd.DataFrame({
        "Categoria": gastos_ultimo_mes.index,
        "Gasto_ultimo_mes": gastos_ultimo_mes.values,
        "Media_historica": media_por_categoria.values
    })
    df_plot["Diferencia"] = df_plot["Media_historica"] - df_plot["Gasto_ultimo_mes"]

    # Gráfico
    fig = px.bar(
        df_plot,
        x="Categoria",
        y=["Gasto_ultimo_mes", "Media_historica"],
        barmode="group",
        title=f"Gastos de {mes_actual.strftime('%Y-%m')} (datos hasta {mes_para_grafico}) vs Media histórica"
    )

    return fig
    return df_plot
