#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 27 17:30:36 2025

@author: alex
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from logic import resumen_gastos, resumen_mensual


# =====================================================
# 1) Balance mensual (ingresos / gastos / balance)
# =====================================================

def plot_balance_mensual(resumen: pd.DataFrame) -> go.Figure:
    """
    Espera un DataFrame con columnas: ingresos, gastos, balance
    e índice con el mes (str o PeriodIndex).
    Devuelve: plotly.graph_objects.Figure
    """
    if resumen is None or len(resumen) == 0:
        raise ValueError("El DataFrame 'resumen' está vacío.")

    required = {"ingresos", "gastos", "balance"}
    missing = required - set(resumen.columns)
    if missing:
        raise ValueError(f"Faltan columnas en 'resumen': {missing}")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=resumen.index, y=resumen["ingresos"], name="Ingresos"))
    fig.add_trace(go.Bar(x=resumen.index, y=-resumen["gastos"], name="Gastos"))
    fig.add_trace(go.Scatter(x=resumen.index, y=resumen["balance"], name="Balance Neto"))

    fig.update_layout(
        title="Balance mensual",
        barmode="relative",
        xaxis_title="Mes",
        yaxis_title="Cantidad (€)",
        legend_title="Tipo",
        template="plotly_white",
    )
    return fig


# =====================================================
# 2) Explicación de gastos (figs + stats, sin prints)
# =====================================================

def plot_explicacion_gastos(df_gastos: pd.DataFrame, df_ingresos: pd.DataFrame, categorias_fijas=None):
    """
    Genera explicación de gastos (figuras + estadísticas) SIN imprimir ni mostrar.

    Devuelve:
      - pivot_explicacion: DataFrame (mes x categorías)
      - stats: dict (series y valores)
      - figs: dict[str, plotly Figure]
    """
    pivot_explicacion = resumen_gastos(df_gastos, df_ingresos)

    if pivot_explicacion is None or pivot_explicacion.empty:
        raise ValueError("pivot_explicacion está vacío. Revisa df_gastos/df_ingresos.")

    pivot_explicacion = pivot_explicacion.copy()
    pivot_explicacion.index = pivot_explicacion.index.astype(str)

    if categorias_fijas is not None:
        pivot_explicacion = pivot_explicacion.reindex(columns=categorias_fijas, fill_value=0)

    total_mes = pivot_explicacion.sum(axis=1)

    stats = {
        "promedio_mensual_por_categoria": pivot_explicacion.mean().sort_values(ascending=False),
        "volatilidad_por_categoria_std": pivot_explicacion.std().sort_values(ascending=False),
        "mes_mayor_gasto_total": total_mes.idxmax(),
        "mes_menor_gasto_total": total_mes.idxmin(),
        "gasto_total_max": float(total_mes.max()) if len(total_mes) else 0.0,
        "gasto_total_min": float(total_mes.min()) if len(total_mes) else 0.0,
    }

    if set(["alimentacion", "transporte"]).issubset(set(pivot_explicacion.columns)) and (total_mes != 0).any():
        prop_fijo = (pivot_explicacion[["alimentacion", "transporte"]].sum(axis=1) / total_mes.replace(0, np.nan)) * 100
        stats["porcentaje_promedio_gasto_fijo_alim_transp"] = float(np.nanmean(prop_fijo.values))
    else:
        stats["porcentaje_promedio_gasto_fijo_alim_transp"] = None

    df_reset = pivot_explicacion.reset_index().rename(columns={"index": "mes"})
    cols = list(pivot_explicacion.columns)

    df_reset["Total"] = df_reset[cols].sum(axis=1)
    media_gasto = df_reset["Total"].mean() if len(df_reset) else 0.0

    figs = {}

    # 2.1 Barras apiladas + línea media
    fig1 = px.bar(
        df_reset,
        x="mes",
        y=cols,
        title="Gastos mensuales por categoría",
        labels={"value": "Gasto (€)", "variable": "Categoría"},
    )
    fig1.add_trace(
        go.Scatter(
            x=df_reset["mes"],
            y=[media_gasto] * len(df_reset),
            mode="lines+markers",
            name="Media mensual",
            line=dict(dash="dot"),
        )
    )
    figs["barras_apiladas"] = fig1

    # 2.2 Distribución porcentual (normalizada a % real)
    df_pct = df_reset.copy()
    denom = df_pct[cols].sum(axis=1).replace(0, np.nan)
    for c in cols:
        df_pct[c] = (df_pct[c] / denom) * 100

    fig2 = px.bar(
        df_pct,
        x="mes",
        y=cols,
        title="Distribución porcentual de gastos",
        labels={"value": "%", "variable": "Categoría"},
    )
    figs["porcentajes"] = fig2

    # 2.3 Evolución temporal
    fig3 = px.line(
        df_reset,
        x="mes",
        y=cols,
        title="Evolución de gastos por categoría",
    )
    figs["lineas"] = fig3

    # 2.4 Boxplot
    df_melt = df_reset.melt(id_vars="mes", value_vars=cols, var_name="Categoría", value_name="Gasto")
    fig4 = px.box(df_melt, x="Categoría", y="Gasto", title="Distribución de gastos por categoría")
    figs["boxplot"] = fig4

    # 2.5 Heatmap
    fig5 = go.Figure(
        data=go.Heatmap(
            z=pivot_explicacion.values,
            x=pivot_explicacion.columns,
            y=pivot_explicacion.index,
        )
    )
    fig5.update_layout(title="Mapa de calor de gastos")
    figs["heatmap"] = fig5

    return pivot_explicacion, stats, figs


# =====================================================
# 3) Porcentaje de ahorro mensual (df + fig)
# =====================================================

def plot_porcentaje_ahorro(df_gastos: pd.DataFrame, df_ingresos: pd.DataFrame):
    """
    Devuelve:
      - resumen: DataFrame con 'porcentaje_ahorro'
      - fig: plotly Figure
    """
    resumen = resumen_mensual(df_gastos, df_ingresos)
    if resumen is None or resumen.empty:
        raise ValueError("resumen_mensual devolvió vacío. Revisa gastos/ingresos.")

    resumen = resumen.copy()
    resumen = resumen[resumen["ingresos"] > 0]
    if resumen.empty:
        raise ValueError("No hay meses con ingresos > 0 para calcular porcentaje de ahorro.")

    resumen["porcentaje_ahorro"] = ((resumen["ingresos"] - resumen["gastos"]) / resumen["ingresos"]) * 100

    fig = px.line(
        resumen,
        x=resumen.index,
        y="porcentaje_ahorro",
        title="Porcentaje de ahorro mensual",
        labels={"porcentaje_ahorro": "% Ahorro", "index": "Mes"},
        markers=True,
    )
    fig.update_layout(template="plotly_white", yaxis_ticksuffix="%")
    return resumen, fig


# =====================================================
# 4) Presupuesto del mes (pie) + métricas (sin prints)
# =====================================================

def grafico_presupuesto(df_resumen: pd.DataFrame):
    """
    Versión 'estricta' como tu original:
    solo genera figura si el último mes del df coincide con el mes actual.
    Devuelve:
      - fig (o None)
      - info dict con dinero_restante, gasto_semanal, semanas_restantes, etc. (o None)
    """
    df_resumen = df_resumen.copy()
    df_resumen["Mes"] = pd.to_datetime(df_resumen["Mes"])

    ultimo_mes = df_resumen["Mes"].max().to_period("M")
    mes_actual = pd.Timestamp.today().to_period("M")

    if ultimo_mes != mes_actual:
        return None, None

    fila = df_resumen.loc[df_resumen["Mes"].dt.to_period("M") == ultimo_mes].iloc[0]

    gasto_mes = float(fila["Gasto del mes"])
    presupuesto_mes = float(fila["Presupuesto Mes"])
    dinero_restante = max(0.0, presupuesto_mes - gasto_mes)

    hoy = pd.Timestamp.today()
    fin_mes = hoy + pd.offsets.MonthEnd(0)
    dias_restantes = (fin_mes - hoy).days + 1
    semanas_restantes = max(1, dias_restantes // 7)
    gasto_semanal = dinero_restante / semanas_restantes if semanas_restantes else 0.0

    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Gastado", "Disponible"],
                values=[gasto_mes, max(0.0, presupuesto_mes - gasto_mes)],
                hole=0.4,
            )
        ]
    )
    fig.update_layout(
        title=f"Presupuesto de {ultimo_mes} - Gastado vs Disponible",
        annotations=[
            dict(
                text=f"Restante\n{dinero_restante:.2f}€",
                x=0.5,
                y=-0.2,
                font_size=14,
                showarrow=False,
            )
        ],
    )

    info = {
        "mes": str(ultimo_mes),
        "gasto_mes": gasto_mes,
        "presupuesto_mes": presupuesto_mes,
        "dinero_restante": dinero_restante,
        "semanas_restantes": semanas_restantes,
        "gasto_semanal_recomendado": gasto_semanal,
    }
    return fig, info


def grafico_presupuesto_1(df_resumen: pd.DataFrame):
    """
    Versión flexible como tu '_1':
    si no hay datos del mes actual, usa el último disponible.
    Devuelve (fig, info).
    """
    df_resumen = df_resumen.copy()
    df_resumen["Mes"] = pd.to_datetime(df_resumen["Mes"])

    ultimo_mes = df_resumen["Mes"].max().to_period("M")
    mes_actual = pd.Timestamp.today().to_period("M")

    if mes_actual not in df_resumen["Mes"].dt.to_period("M").unique():
        mes_para_grafico = ultimo_mes
    else:
        mes_para_grafico = mes_actual

    fila = df_resumen.loc[df_resumen["Mes"].dt.to_period("M") == mes_para_grafico].iloc[0]

    gasto_mes = float(fila["Gasto del mes"])
    presupuesto_mes = float(fila["Presupuesto Mes"])
    dinero_restante = max(0.0, presupuesto_mes - gasto_mes)

    hoy = pd.Timestamp.today()
    fin_mes = hoy + pd.offsets.MonthEnd(0)
    dias_restantes = (fin_mes - hoy).days + 1
    semanas_restantes = max(1, dias_restantes // 7)
    gasto_semanal = dinero_restante / semanas_restantes if semanas_restantes else 0.0

    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Gastado", "Disponible"],
                values=[gasto_mes, max(0.0, presupuesto_mes - gasto_mes)],
                hole=0.4,
            )
        ]
    )
    fig.update_layout(
        title=f"Presupuesto de {mes_actual} (datos hasta {mes_para_grafico})",
        annotations=[
            dict(
                text=f"Restante\n{dinero_restante:.2f}€",
                x=0.5,
                y=-0.2,
                font_size=14,
                showarrow=False,
            )
        ],
    )

    info = {
        "mes_actual": str(mes_actual),
        "mes_datos": str(mes_para_grafico),
        "gasto_mes": gasto_mes,
        "presupuesto_mes": presupuesto_mes,
        "dinero_restante": dinero_restante,
        "semanas_restantes": semanas_restantes,
        "gasto_semanal_recomendado": gasto_semanal,
    }
    return fig, info


# =====================================================
# 5) Gasto vs media (sin show)
# =====================================================

def graficar_gasto_vs_media(df_gastos: pd.DataFrame, df_ingresos: pd.DataFrame):
    """
    Versión estricta: solo genera si el último mes coincide con el mes actual.
    Devuelve (df_plot, fig) o (None, None)
    """
    df_g = resumen_gastos(df_gastos, df_ingresos)

    if not isinstance(df_g.index, pd.PeriodIndex):
        df_g.index = pd.to_datetime(df_g.index).to_period("M")

    ultimo_mes = df_g.index.max()
    mes_actual = pd.Timestamp.today().to_period("M")

    if ultimo_mes != mes_actual:
        return None, None

    gastos_ultimo_mes = df_g.loc[ultimo_mes]
    media_por_categoria = df_g.mean()

    df_plot = pd.DataFrame(
        {
            "Categoria": gastos_ultimo_mes.index,
            "Gasto_ultimo_mes": gastos_ultimo_mes.values,
            "Media_historica": media_por_categoria.values,
        }
    )
    df_plot["Diferencia"] = df_plot["Media_historica"] - df_plot["Gasto_ultimo_mes"]

    fig = px.bar(
        df_plot,
        x="Categoria",
        y=["Gasto_ultimo_mes", "Media_historica"],
        barmode="group",
        title=f"Gastos de {ultimo_mes.strftime('%Y-%m')} vs Media histórica",
    )
    return df_plot, fig


def graficar_gasto_vs_media_1(df_gastos: pd.DataFrame, df_ingresos: pd.DataFrame):
    """
    Versión flexible: si no hay datos del mes actual, usa el último mes disponible.
    Devuelve (df_plot, fig)
    """
    df_g = resumen_gastos(df_gastos, df_ingresos)

    if not isinstance(df_g.index, pd.PeriodIndex):
        df_g.index = pd.to_datetime(df_g.index).to_period("M")

    ultimo_mes = df_g.index.max()
    mes_actual = pd.Timestamp.today().to_period("M")

    if mes_actual not in df_g.index:
        mes_para_grafico = ultimo_mes
    else:
        mes_para_grafico = mes_actual

    gastos_mes = df_g.loc[mes_para_grafico]
    media_por_categoria = df_g.mean()

    df_plot = pd.DataFrame(
        {
            "Categoria": gastos_mes.index,
            "Gasto_ultimo_mes": gastos_mes.values,
            "Media_historica": media_por_categoria.values,
        }
    )
    df_plot["Diferencia"] = df_plot["Media_historica"] - df_plot["Gasto_ultimo_mes"]

    fig = px.bar(
        df_plot,
        x="Categoria",
        y=["Gasto_ultimo_mes", "Media_historica"],
        barmode="group",
        title=f"Gastos de {mes_actual.strftime('%Y-%m')} (datos hasta {mes_para_grafico}) vs Media histórica",
    )
    return df_plot, fig
