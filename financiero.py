# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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


def resumen_mensual(df_gastos, df_ingresos):
    df_gastos = df_gastos.copy()
    df_ingresos = df_ingresos.copy()

    df_gastos['mes'] = df_gastos['fecha'].dt.to_period('M').astype(str)
    df_ingresos['mes'] = df_ingresos['fecha'].dt.to_period('M').astype(str)

    gastos_mensuales = df_gastos.groupby('mes')['cantidad'].sum().rename('gastos')
    ingresos_mensuales = df_ingresos[df_ingresos['tipo_logico'] == 'Ingreso Real'].groupby('mes')['cantidad'].sum().rename('ingresos')

    resumen = pd.concat([ingresos_mensuales, gastos_mensuales], axis=1).fillna(0)
    resumen['balance'] = resumen['ingresos'] - resumen['gastos']
    return resumen

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

def plot_ingresos_por_tipo(df_ingresos):
    resumen = df_ingresos.groupby('tipo_logico')['cantidad'].sum()
    fig = px.pie(
        names=resumen.index,
        values=resumen.values,
        title="🍰 Distribución de ingresos por tipo lógico",
        hole=0.4
    )
    fig.update_traces(textinfo='percent+label')
    fig.show()

def plot_gastos_netos(gastos_netos):
    gastos_filtrados = gastos_netos[gastos_netos > 0].sort_values(ascending=True)
    fig = px.bar(
        gastos_filtrados,
        orientation='h',
        title="📉 Gastos personales netos por categoría",
        labels={'value': 'Cantidad (€)', 'index': 'Categoría'},
        color_discrete_sequence=['crimson']
    )
    fig.update_layout(template='plotly_white')
    fig.show()

def plot_porcentaje_ahorro(df_gastos, df_ingresos):
    df_ingresos = procesar_ingresos(df_ingresos)
    df_gastos['mes'] = df_gastos['fecha'].dt.to_period('M').astype(str)
    df_ingresos['mes'] = df_ingresos['fecha'].dt.to_period('M').astype(str)

    ingresos_real = df_ingresos[df_ingresos['tipo_logico'] == 'Ingreso Real']
    reembolsos = df_ingresos[df_ingresos['tipo_logico'].str.startswith('Reembolso')]
    reembolsos['categoria_reembolso'] = reembolsos['tipo_logico'].str.replace('Reembolso ', '', regex=False)
    reembolsos['mes'] = reembolsos['fecha'].dt.to_period('M').astype(str)
   
    ingresos_mensual = ingresos_real.groupby('mes')['cantidad'].sum()
    gastos_mensual = df_gastos.groupby('mes')['cantidad'].sum() - reembolsos.groupby('mes')['cantidad'].sum()

    resumen = pd.concat([ingresos_mensual.rename('ingresos'), gastos_mensual.rename('gastos')], axis=1).fillna(0)
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

def crear_historial_cuentas_virtuales(df_ingresos, df_gastos, fecha_inicio='2024-10-01',
                                     porcentaje_gasto=0.3, porcentaje_inversion=0.1, saldos_iniciales=saldos_iniciales):
    df_ingresos = procesar_ingresos(df_ingresos)
    df_gastos['mes'] = df_gastos['fecha'].dt.to_period('M').astype(str)
    df_ingresos['mes'] = df_ingresos['fecha'].dt.to_period('M').astype(str)

    fecha_actual = max(df_ingresos['fecha'].max(), df_gastos['fecha'].max())
   
    # Convertir fechas a pd.Timestamp
    fecha_inicio = pd.Timestamp(fecha_inicio)
    fecha_fin = pd.Timestamp(fecha_actual)

    # Inicializar deuda acumulada
    deuda_acumulada = 0.0
   
    # Inicializar regalos acumulada
    regalos = 0
   
    #Vacaciones
    vacaciones_inicial = saldos_iniciales.get('Metálico', 0.0)
    vacaciones = vacaciones_inicial
    print(vacaciones_inicial)
   
    #Ahorros
    ahorro = 0
   
    # Inicializar regalos acumulada
   
    ahorro_inicial = sum(v for k, v in saldos_iniciales.items() if k != 'Metálico')
    ahorro += ahorro_inicial

    #Inicializar las inversiones
    inversiones = 0
    resumenes = []

    # Iterar mes a mes
    mes_actual = fecha_inicio.to_period('M')
    while mes_actual <= fecha_fin.to_period('M'):
        mes_actual_str = mes_actual.strftime('%Y-%m')
        mes_anterior = (mes_actual - 1).strftime('%Y-%m')
       
        regalos_mes = df_ingresos.loc[(df_ingresos['tipo_logico'] == 'Regalos') & (df_ingresos['mes'] == mes_actual_str),'cantidad'].sum()-df_gastos.loc[(df_gastos['tipo_logico'] == 'Regalos') & (df_gastos['mes'] == mes_actual_str),'cantidad'].sum()
        regalos+= regalos_mes
                                                                                                                                                                                                               
        # Filtrar ingresos y gastos para los meses correspondientes
        ingresos_reales = df_ingresos[df_ingresos['tipo_logico'] == 'Ingreso Real']
        ingresos_reales_mensual = ingresos_reales.groupby('mes')['cantidad'].sum()

        gasto_total_mes_actual = calcular_gastos_netos(df_gastos[df_gastos['mes'] == mes_actual_str], df_ingresos[df_ingresos['mes'] == mes_actual_str]).sum()
       
        gasto_vacaciones = df_gastos.loc[(df_gastos['etiquetas'] == 'Vacaciones') & (df_gastos['mes'] == mes_actual_str),'cantidad'].sum()
        # Presupuesto teórico para el mes actual (30% de ingresos mes anterior)
       
        # if mes_actual == fecha_inicio:
        #     presupuesto_teorico =
        presupuesto_teorico = ingresos_reales_mensual.get(mes_anterior, 0) * porcentaje_gasto

        # Ajustar presupuesto con la deuda acumulada
        presupuesto_disponible = max(0, presupuesto_teorico - deuda_acumulada)

        # Calcular nueva deuda si gasto supera el presupuesto disponible
        deuda_mensual = gasto_total_mes_actual - presupuesto_teorico
        exceso_gasto = deuda_mensual + deuda_acumulada
        nueva_deuda = max(0, exceso_gasto)

        # Actualizar deuda acumulada para el siguiente mes
        deuda_acumulada = nueva_deuda

        # Vacaciones
        vacaciones_ingresos = df_ingresos[(df_ingresos['tipo_logico'] == 'Vacaciones') & (df_ingresos['mes'] == mes_actual_str)]['cantidad'].sum()
        vacaciones += vacaciones_ingresos - gasto_vacaciones

        # Inversiones (solo desde julio 2025)
       
        interes = df_ingresos[(df_ingresos['tipo_logico'] == 'Rendimiento Financiero') & (df_ingresos['mes'] == mes_actual_str)]['cantidad'].sum()
        if mes_actual >= pd.Timestamp("2025-05-01").to_period('M'):
            inv_mensual = ingresos_reales_mensual.get(mes_actual_str, 0) * porcentaje_inversion
        else:
            inv_mensual = 0
           
        inversion_mes = inv_mensual + interes
        inversiones += inversion_mes

        # Cálculo ahorro
        ingreso_total = ingresos_reales[ingresos_reales['mes'] == mes_actual_str]['cantidad'].sum()
        gastos_netos_total = calcular_gastos_netos(df_gastos[df_gastos['mes'] == mes_actual_str], df_ingresos[df_ingresos['mes'] == mes_actual_str]).sum()

        # Presupuesto efectivo (lo que queda tras gastos y deuda)
        presupuesto_efectivo = max(0, presupuesto_disponible - gasto_total_mes_actual)
       
        if presupuesto_efectivo >0:
            #Se añade gasto vacaciones para no duplicar los gastos
            ahorro_transaccional = ingreso_total - gastos_netos_total - inversion_mes - vacaciones_ingresos + gasto_vacaciones + presupuesto_efectivo/3
            vacaciones += presupuesto_efectivo/3
            inversiones += presupuesto_efectivo/3
        else:
            ahorro_transaccional = ingreso_total - gastos_netos_total - inversion_mes - vacaciones_ingresos - presupuesto_efectivo
        ahorro += ahorro_transaccional


        #Fondo de reserva
       
       


        resumen = {
            'mes': mes_actual_str,
            '🎁 Regalos': round(regalos,2),
            '💼 Vacaciones': round(vacaciones, 2),
            '📈 Inversiones': round(inversiones, 2),
            '💳 Gasto del mes': round(gasto_total_mes_actual,2),
            '💸 Presupuesto Mes': round(presupuesto_teorico, 2),
            '🧾 Presupuesto Disponible': round(presupuesto_efectivo, 2),
            '💰 Ahorros': round(ahorro, 2),
            '📉 Deuda Presupuestaria mensual': round(deuda_mensual, 2) if deuda_mensual > 0 else 0.0,
            '📉 Deuda Presupuestaria acumulada': round(deuda_acumulada, 2) if deuda_acumulada > 0 else 0.0
        }

        resumenes.append(resumen)

        mes_actual += 1  # siguiente mes
    df_resumen = pd.DataFrame(resumenes).copy(deep = True)
    df_resumen.columns= ['Mes', 'Regalos', 'Vacaciones','Inversiones','Gasto del mes','Presupuesto Mes','Presupuesto Disponible','Ahorros','Deuda Presupuestaria mensual', 'Deuda Presupuestaria acumulada']
    df_resumen.to_csv(os.path.join('Data', 'historial.csv'), index=False)
    return pd.DataFrame(resumenes)




def fondo_reserva_general(carpeta_data='Data'):
    # Verifica si la carpeta existe, si no, créala
    if not os.path.exists(carpeta_data):
        os.makedirs(carpeta_data)

    # Ruta al archivo CSV
    archivo_csv = os.path.join(carpeta_data, 'fondo_reserva.csv')
   
    historial = pd.read_csv(os.path.join('Data', 'historial.csv'))
    # Calcula el fondo de reserva
    gastos_disponibles = historial['Gasto del mes']
    media_gastos_mensuales = gastos_disponibles.mean()
    fondo_reserva = media_gastos_mensuales * 6

    # Comprueba si el archivo CSV existe
    if os.path.exists(archivo_csv):
        df_fondo = pd.read_csv(archivo_csv)

        # Convertir fechas a datetime
        df_fondo['fecha de creacion'] = pd.to_datetime(df_fondo['fecha de creacion'])

        ultima_fecha = df_fondo['fecha de creacion'].max()
        hoy = datetime.now()

        # Comparar si la última entrada fue hace más de 3 meses
        if hoy - ultima_fecha >= timedelta(days=90):
            # Si pasos más de 3 meses, añadir un nuevo registro
            nuevo_registro = pd.DataFrame({
                'fecha de creacion': [hoy],
                'Cantidad del fondo': [fondo_reserva]
            })
            df_fondo = pd.concat([df_fondo, nuevo_registro])
            df_fondo.to_csv(archivo_csv, index=False)

    else:
        # Si el archivo no existe, créalo
        nuevo_registro = pd.DataFrame({
            'fecha de creacion': [datetime.now()],
            'Cantidad del fondo': [fondo_reserva]
        })
        nuevo_registro.to_csv(archivo_csv, index=False)

        # En caso de ser la primera creacion
        df_fondo = nuevo_registro

    # Devolver el registro con la fecha más actual
    registro_actual = df_fondo.sort_values('fecha de creacion', ascending=False).iloc[0]
   
    print('Última actualización:', registro_actual.reset_index().iloc[0,1].strftime('%d-%m-%Y'))
    print(f'Cantidad del fondo requerida: {round(registro_actual.reset_index().iloc[1,1])}€')
    return registro_actual
