#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 01:13:27 2025

@author: alex
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 12:02:06 2025

@author: abarragan1
"""
import os
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


hoy = str(pd.Timestamp.today().to_period('M'))

REGISTROS_DIR = Path(__file__).resolve().parent / "Data" / "Registros"
ARCHIVO_BASE_REGISTROS = REGISTROS_DIR / "Inicio.xlsx"
HOJAS_REGISTRO = ["Gastos", "Ingresos", "Transferencias"]


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
            f"No se encontró el archivo base requerido: {ARCHIVO_BASE_REGISTROS}"
        )

    dataframes = {}
    columnas_por_hoja = {}

    for hoja in HOJAS_REGISTRO:
        df_base = pd.read_excel(ARCHIVO_BASE_REGISTROS, sheet_name=hoja)
        columnas_por_hoja[hoja] = df_base.columns
        dataframes[hoja] = _limpiar_dataframe_registro(df_base)

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


fusionar_archivos_registro()



class Account:
    def __init__(self, name):
        self.name = name
        self.transactions = []

    @property
    def balance(self):
        return sum(t.amount for t in self.transactions)
   
    def balance_a_fecha(self, fecha):
        return sum(t.amount for t in self.transactions if t.timestamp <= fecha)


    def __repr__(self):
        return f"<Account: {self.name}, Balance: {self.balance:.2f}>"

class Transaction:
    def __init__(self, amount, description, category, account, timestamp=None):
        self.amount = amount
        self.description = description
        self.category = category
        self.account = account
        self.timestamp = timestamp if timestamp else datetime.now()

    def __repr__(self):
        return f"<Transaction {self.timestamp.date()} | {self.amount:.2f} | {self.category} | {self.account.name}>"

class Budget:
    def __init__(self):
        self.accounts = {}
        self.transactions = []

    def add_account(self, account):
        self.accounts[account.name] = account

    def get_account(self, name):
        return self.accounts.get(name)

    def add_transaction(self, transaction):
        if transaction.account.name not in self.accounts:
            raise ValueError(f"La cuenta '{transaction.account.name}' no está en el presupuesto.")
        self.accounts[transaction.account.name].transactions.append(transaction)
        self.transactions.append(transaction)

    def get_balance(self):
        return sum(account.balance for account in self.accounts.values())
   
    def balances_a_fecha(self, fecha):
        return {nombre: cuenta.balance_a_fecha(fecha) for nombre, cuenta in self.accounts.items()}

    def balance_total_a_fecha(self, fecha):
        return sum(cuenta.balance_a_fecha(fecha) for cuenta in self.accounts.values())

    def get_transactions(self, category=None, account=None):
        result = self.transactions
        if category:
            result = [t for t in result if t.category == category]
        if account:
            result = [t for t in result if t.account.name == account]
        return result

    def get_transactions_by_month(self):
        months = defaultdict(list)
        for t in self.transactions:
            key = t.timestamp.strftime("%Y-%m")
            months[key].append(t)
        return months

    def __repr__(self):
        return f"<Budget: {len(self.accounts)} cuentas, {len(self.transactions)} transacciones>"


saldos_iniciales = {
    'Principal': 482.99000000000007,
    'Cobee': 0,#-5.684341886080802e-14,
    'Metálico': 54.999999999999986,
    'Revolut': 0.0,
    'Ahorro': 4680.999999999999
}


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



# Función para agregar transacciones desde un dataframe
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
    fig.show()

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
    fig.show()
   
   
from dateutil.relativedelta import relativedelta

def invertido_en_mes(presupuesto, mes_periodo: pd.Period):
    """
    Devuelve el dinero invertido en un mes determinado.
   
    - Si mes_periodo es pasado: toma el último día del mes.
    - Si mes_periodo es el actual: toma el día actual.
    """

    # Fecha de hoy
    hoy = pd.Timestamp(datetime.today())
    mes_actual = hoy.to_period('M')

    if mes_periodo < mes_actual:
        # Último día del mes pasado
        fecha_objetivo = mes_periodo.to_timestamp(how="end")
    elif mes_periodo == mes_actual:
        # Día actual
        fecha_objetivo = hoy
    else:
        raise ValueError("El mes indicado es en el futuro y no puede calcularse.")

    # Consultar el balance en la fecha
    invertido = presupuesto.balances_a_fecha(fecha_objetivo)["Invertido"]

    return invertido


def crear_historial_cuentas_virtuales(
    df_ingresos,
    df_gastos,
    presupuesto,
    fecha_inicio='2024-10-01',
    porcentaje_gasto=0.3,
    porcentaje_inversion=0.1,
    porcentaje_vacaciones=0.05,
    saldos_iniciales=None,
    output_path="Data/historial.csv"
):
    """Genera el histórico mensual de cuentas virtuales optimizando su cálculo.

    La aritmética original se conserva, pero se precalculan todas las series
    mensuales necesarias para reducir filtros reiterativos sobre los
    ``DataFrame`` de gastos e ingresos.
    """

    # Compatibilidad con tu firma original si existía una variable global del mismo nombre
    if saldos_iniciales is None:
        saldos_iniciales = globals().get('saldos_iniciales', {}) or {}

    # Copias defensivas
    df_gastos = df_gastos.copy()
    df_ingresos = df_ingresos.copy()

    # Procesamiento según tu pipeline
    df_ingresos = procesar_ingresos(df_ingresos)

    # Mes como string YYYY-MM
    df_gastos['mes'] = df_gastos['fecha'].dt.to_period('M').astype(str)
    df_ingresos['mes'] = df_ingresos['fecha'].dt.to_period('M').astype(str)

    #Agrupaciones recurrentes para evitar filtros repetitivos en cada iteración
    gastos_por_mes = {mes: grupo.copy() for mes, grupo in df_gastos.groupby('mes')}
    ingresos_por_mes = {mes: grupo.copy() for mes, grupo in df_ingresos.groupby('mes')}
    meses_disponibles = sorted(set(gastos_por_mes) | set(ingresos_por_mes))

    # Series auxiliares agregadas
    regalos_gasto = (
        df_gastos[df_gastos['tipo_logico'] == 'Regalos']
        .groupby('mes')['cantidad']
        .sum()
    )
    regalos_ingreso = (
        df_ingresos[df_ingresos['tipo_logico'] == 'Regalos']
        .groupby('mes')['cantidad']
        .sum()
    )
    vacaciones_gasto = (
        df_gastos[df_gastos['etiquetas'] == 'Vacaciones']
        .groupby('mes')['cantidad']
        .sum()
    )
    vacaciones_ingreso = (
        df_ingresos[df_ingresos['tipo_logico'] == 'Vacaciones']
        .groupby('mes')['cantidad']
        .sum()
    )
    fondo_reserva_gasto = (
        df_gastos[df_gastos['categoria'] == 'FondoReserva']
        .groupby('mes')['cantidad']
        .sum()
    )
    intereses_ingreso = (
        df_ingresos[df_ingresos['tipo_logico'] == 'Rendimiento Financiero']
        .groupby('mes')['cantidad']
        .sum()
    )

    empty_gastos = df_gastos.head(0)
    empty_ingresos = df_ingresos.head(0)
    gastos_netos_por_mes = {}
    for mes in meses_disponibles:
        gastos_mes = gastos_por_mes.get(mes, empty_gastos)
        ingresos_mes = ingresos_por_mes.get(mes, empty_ingresos)
        gastos_netos_por_mes[mes] = calcular_gastos_netos(gastos_mes, ingresos_mes)





    # Fechas
    fecha_actual = max(df_ingresos['fecha'].max(), df_gastos['fecha'].max())
    fecha_inicio = pd.Timestamp(fecha_inicio)
    fecha_fin = pd.Timestamp(fecha_actual)

    # Inicializaciones (idénticas a tu lógica)
    deuda_acumulada = 0.0
    regalos = 0.0

    vacaciones_inicial = saldos_iniciales.get('Metálico', 0.0)
    vacaciones = vacaciones_inicial

    ahorro = 0.0
    ahorro_inicial = sum(v for k, v in saldos_iniciales.items() if k != 'Metálico')
    ahorro += ahorro_inicial

    inversiones = 0.0
    resumenes = []

    # Solo si arranca en 2024-10, igual que tu if
    if fecha_inicio.to_period('M') == pd.Timestamp('2024-10-01').to_period('M'):
        ahorro_emergencia = 0
        fondo_cargado = 0

    # Fondo de reserva (idéntico estilo de acceso)
    df_Fondo_reserva = fondo_reserva_general()
    fondo_cargado_ini = df_Fondo_reserva['Cantidad cargada']
    fondo_reserva = df_Fondo_reserva['Cantidad del fondo']
    porcentaje = df_Fondo_reserva['Porcentaje']  # aunque luego se sobreescribe, lo conservo por fidelidad

    # Precalculo (no altera resultados)
    ingresos_reales = df_ingresos[df_ingresos['tipo_logico'] == 'Ingreso Real']
    ingresos_reales_mensual = ingresos_reales.groupby('mes')['cantidad'].sum()

    # Iteración mensual
    mes_actual = fecha_inicio.to_period('M')
    while mes_actual <= fecha_fin.to_period('M'):
        mes_actual_str = mes_actual.strftime('%Y-%m')
        mes_anterior = (mes_actual - 1).strftime('%Y-%m')

        # REGALOS
        gasto_regalos = regalos_gasto.get(mes_actual_str, 0.0)
        ingreso_regalos = regalos_ingreso.get(mes_actual_str, 0.0)
        regalos_mes = ingreso_regalos - gasto_regalos
        regalos += regalos_mes

        # INGRESOS (como en tu versión)
        # Vacaciones
        gasto_vacaciones = vacaciones_gasto.get(mes_actual_str, 0.0)
       
        if mes_actual >= pd.Timestamp("2025-09-01").to_period('M'):
            vacaciones_mensual = ingresos_reales_mensual.get(mes_actual_str, 0) * porcentaje_vacaciones
        else:
            vacaciones_mensual = 0
        
        vacaciones_ingresos = vacaciones_ingreso.get(mes_actual_str, 0.0)
        vacaciones += vacaciones_ingresos + vacaciones_mensual - gasto_vacaciones

        # FONDO DE RESERVA
        gasto_fondo_reserva = fondo_reserva_gasto.get(mes_actual_str, 0.0)
       
        # GASTOS netos (igual a tu cálculo restando vacaciones/regalos)
        gasto_total_mes_actual_aux = gastos_netos_por_mes.get(
            mes_actual_str, pd.Series(dtype=float)
        ).sum()
        gasto_total_mes_actual = gasto_total_mes_actual_aux - gasto_vacaciones - gasto_regalos - gasto_fondo_reserva

        # Presupuesto desde el mes anterior
        presupuesto_teorico = ingresos_reales_mensual.get(mes_anterior, 0) * porcentaje_gasto

        # Ajuste con deuda
        presupuesto_disponible = max(0, presupuesto_teorico - deuda_acumulada)

        # Deuda mensual y acumulada (idéntico)
        deuda_mensual = gasto_total_mes_actual - presupuesto_teorico
        exceso_gasto = deuda_mensual + deuda_acumulada
        nueva_deuda = max(0, exceso_gasto)
        deuda_acumulada = nueva_deuda

        # Inversiones (desde 2025-05 como en tu código)
        interes = intereses_ingreso.get(mes_actual_str, 0.0)
        if mes_actual >= pd.Timestamp("2025-05-01").to_period('M'):
            inv_mensual = ingresos_reales_mensual.get(mes_actual_str, 0) * porcentaje_inversion
        else:
            inv_mensual = 0
        inversion_mes = inv_mensual + interes
        dinero_invertido_entre_mes = invertido_en_mes(presupuesto, mes_actual) - invertido_en_mes(presupuesto, mes_actual - 1)

        inversiones += inversion_mes - dinero_invertido_entre_mes
        dinero_invertido = invertido_en_mes(presupuesto, mes_actual)
       

        # Ahorro (misma fórmula)
        ingreso_total = ingresos_reales_mensual.get(mes_actual_str, 0.0)
        gastos_netos_total = gastos_netos_por_mes.get(
            mes_actual_str, pd.Series(dtype=float)).sum()

        presupuesto_efectivo = max(0, presupuesto_disponible - gasto_total_mes_actual)

        if presupuesto_efectivo > 0:
            ahorro_transaccional = (
                ingreso_total - gastos_netos_total - (presupuesto_efectivo * 2 / 3) - inv_mensual - vacaciones_mensual
                + gasto_vacaciones + gasto_regalos
            )
            vacaciones += presupuesto_efectivo / 3
            inversiones += presupuesto_efectivo / 3
        else:
            ahorro_transaccional = (
                ingreso_total - gastos_netos_total - presupuesto_efectivo - inv_mensual - vacaciones_mensual
                + gasto_vacaciones + gasto_regalos
            )

        # Fondo de reserva (MISMA lógica: sin tope explícito, suma mensual de fondo_cargado_ini + ahorro_emergencia)
        if mes_actual == pd.Timestamp('2024-10-01').to_period('M'):
            porcentaje = 0.0 if fondo_reserva == 0 else min(fondo_cargado / fondo_reserva, 1.0)
        else:
            porcentaje = 0.0 if fondo_reserva == 0 else min(fondo_cargado / fondo_reserva, 1.0)
            if porcentaje < 1.0:
                ahorro_emergencia = max(0.0, ahorro_transaccional * 0.1)
                fondo_cargado += ahorro_emergencia - gasto_fondo_reserva
                fondo_cargado = max(0.0, fondo_cargado)
            else:
                fondo_cargado = max(0.0, fondo_reserva - gasto_fondo_reserva)
                ahorro_emergencia = 0.0

            porcentaje = 0.0 if fondo_reserva == 0 else min(fondo_cargado / fondo_reserva, 1.0)
       
        ahorro += ahorro_transaccional - ahorro_emergencia

        total = regalos + vacaciones + inversiones + ahorro + fondo_cargado - dinero_invertido
        # total = regalos + vacaciones + inversiones + ahorro + fondo_cargado
        resumen = {
            'mes': mes_actual_str,
            '🎁 Regalos': round(regalos, 2),
            '💼 Vacaciones': round(vacaciones, 2),
            '📈 Inversiones': round(inversiones, 2),
            'Dinero Invertido': round(dinero_invertido, 2),
            '💰 Ahorros': round(ahorro, 2),
            'Fondo de reserva cargado': round(fondo_cargado, 4),
            'total': round(total, 2),
            '💳 Gasto del mes': round(gasto_total_mes_actual, 2),
            '💸 Presupuesto Mes': round(presupuesto_teorico, 2),
            '🧾 Presupuesto Disponible': round(presupuesto_efectivo, 2),
            '📉 Deuda Presupuestaria mensual': round(deuda_mensual, 2) if deuda_mensual > 0 else 0.0,
            '📉 Deuda Presupuestaria acumulada': round(deuda_acumulada, 2) if deuda_acumulada > 0 else 0.0
        }

        if mes_actual == pd.Timestamp('2024-10-01').to_period('M'):
            df_resumen_sep = pd.DataFrame([resumen]).copy(deep=True)
            df_resumen_sep.columns = [
                'Mes', 'Regalos', 'Vacaciones', 'Inversiones', 'Dinero Invertido', 'Ahorros',
                'Fondo de reserva cargado', 'total', 'Gasto del mes',
                'Presupuesto Mes', 'Presupuesto Disponible',
                'Deuda Presupuestaria mensual', 'Deuda Presupuestaria acumulada'
            ]
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                df_resumen_sep.to_csv(output_path, index=False)

        resumenes.append(resumen)
        mes_actual += 1

    df_resumen = pd.DataFrame(resumenes).copy(deep=True)
    df_resumen.columns = [
        'Mes', 'Regalos', 'Vacaciones', 'Inversiones', 'Dinero Invertido', 'Ahorros',
        'Fondo de reserva cargado', 'total', 'Gasto del mes',
        'Presupuesto Mes', 'Presupuesto Disponible',
        'Deuda Presupuestaria mensual', 'Deuda Presupuestaria acumulada'
    ]
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_resumen.to_csv(output_path, index=False)
        
   # Actualizar el histórico del fondo de reserva con la instantánea recién calculada
    fondo_reserva_general(
        lectura=False,
        historial=df_resumen,
        fondo_cargado_actual=float(df_resumen.iloc[-1]['Fondo de reserva cargado'])
    )
    print("Es posible que los valores no sean precisos para el fondo de reserva, se debe observar cuando se ha obtenido el ingreso de fin de mes.")
    return df_resumen




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

def resumen_global(presupuesto, fecha):
    fecha_objetivo = pd.to_datetime(fecha)
    saldos_a_fecha = presupuesto.balances_a_fecha(fecha_objetivo)
    saldo_total = presupuesto.balance_total_a_fecha(fecha_objetivo)
    print(f"💰 Saldo total al {fecha_objetivo.date()}: {round(saldo_total, 2)}")
    print(f"Saldos al {fecha_objetivo.date()}:")
    for nombre, saldo in saldos_a_fecha.items():
        print(f"Cuenta: {nombre}, Saldo: {round(saldo, 2)}")

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
    plt.show()
   
    print(predicciones)


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

    fig.show()

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

    fig.show()
    return df_plot
