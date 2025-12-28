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


from .reserva import fondo_reserva_general

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
    output_path="Data/historial.csv",
    objetivos_config=None,
    actualizar_fondo_reserva: bool = True):
    """Genera el histórico mensual de cuentas virtuales optimizando su cálculo.

    La aritmética original se conserva, pero se precalculan todas las series
    mensuales necesarias para reducir filtros reiterativos sobre los
    ``DataFrame`` de gastos e ingresos. Además incorpora un motor de
    *objetivos vista* para repartir parte de los ingresos en sobres
    etiquetados ("Coche", "Boda", etc.) y descontar automáticamente los
    gastos asociados a esas etiquetas.
    """

    # Compatibilidad con tu firma original si existía una variable global del mismo nombre
    if saldos_iniciales is None:
        saldos_iniciales = globals().get('saldos_iniciales', {}) or {}

    objetivos_config = cargar_objetivos_vista(objetivos_config)
    objetivos_nombres = [cfg['nombre'] for cfg in objetivos_config]
    porcentaje_objetivos = sum(cfg['porcentaje_ingreso'] for cfg in objetivos_config)

    porcentaje_total = porcentaje_gasto + porcentaje_inversion + porcentaje_vacaciones + porcentaje_objetivos
    if porcentaje_total > 1 + 1e-6:
        raise ValueError(
            "La suma de porcentajes (gasto, inversiones, vacaciones y objetivos vista) supera el 100%."
        )

    objetivos_por_nombre = {cfg['nombre']: cfg for cfg in objetivos_config}
    if len(objetivos_por_nombre) != len(objetivos_config):
        raise ValueError("Los objetivos vista deben tener nombres únicos.")
    objetivos_meta = {
        nombre: {
            'objetivo_total': cfg.get('objetivo_total'),
            'horizonte_meses': cfg.get('horizonte_meses'),
            'mes_inicio': cfg.get('mes_inicio'),
        }
        for nombre, cfg in objetivos_por_nombre.items()
    }
    objetivos_saldos = {nombre: 0.0 for nombre in objetivos_por_nombre}
    objetivos_saldo_inicial = {nombre: cfg['saldo_inicial'] for nombre, cfg in objetivos_por_nombre.items()}
    objetivos_mes_inicio = {nombre: cfg.get('mes_inicio') for nombre, cfg in objetivos_por_nombre.items()}
    objetivos_inicial_aplicado = {nombre: False for nombre in objetivos_por_nombre}

    def _renombrar_columnas_historial(df: pd.DataFrame) -> pd.DataFrame:
        columnas_base = {
            'Mes': 'Mes',
            '🎁 Regalos': 'Regalos',
            '💼 Vacaciones': 'Vacaciones',
            '📈 Inversiones': 'Inversiones',
            'Dinero Invertido': 'Dinero Invertido',
            '💰 Ahorros': 'Ahorros',
            'Fondo de reserva cargado': 'Fondo de reserva cargado',
            'total': 'total',
            '💳 Gasto del mes': 'Gasto del mes',
            '💸 Presupuesto Mes': 'Presupuesto Mes',
            '🧾 Presupuesto Disponible': 'Presupuesto Disponible',
            '📉 Deuda Presupuestaria mensual': 'Deuda Presupuestaria mensual',
            '📉 Deuda Presupuestaria acumulada': 'Deuda Presupuestaria acumulada',
        }

        renombradas = {}
        for columna in df.columns:
            if columna in columnas_base:
                renombradas[columna] = columnas_base[columna]
            elif columna.startswith('🎯 '):
                nombre_objetivo = columna.split(' ', 1)[1] if ' ' in columna else columna.replace('🎯', '').strip()
                renombradas[columna] = f"Objetivo {nombre_objetivo}"

        return df.rename(columns=renombradas)

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

    objetivos_gastos_por_mes = {}
    objetivos_ingresos_por_mes = {}
    objetivos_ingresos_reales_por_mes = {}

    if objetivos_config:
        etiquetas_gastos = df_gastos['etiquetas'].fillna('').str.lower()
        etiquetas_ingresos = df_ingresos['etiquetas'].fillna('').str.lower()

    for cfg in objetivos_config:
        etiquetas = cfg['etiquetas']
        if etiquetas:
            patron = "|".join(re.escape(et) for et in etiquetas if et)
        else:
            patron = ""

        if patron:
            mask_gastos = etiquetas_gastos.str.contains(patron, regex=True)
            mask_ingresos = etiquetas_ingresos.str.contains(patron, regex=True)
        else:
            mask_gastos = pd.Series(False, index=df_gastos.index)
            mask_ingresos = pd.Series(False, index=df_ingresos.index)

        gastos_obj = (
            df_gastos.loc[mask_gastos].groupby('mes')['cantidad'].sum()
            if mask_gastos.any()
            else pd.Series(dtype=float)
        )
        ingresos_obj = (
            df_ingresos.loc[mask_ingresos].groupby('mes')['cantidad'].sum()
            if mask_ingresos.any()
            else pd.Series(dtype=float)
        )

        if mask_ingresos.any():
            mask_ingresos_reales = mask_ingresos & (df_ingresos['tipo_logico'] == 'Ingreso Real')
            ingresos_reales_obj = (
                df_ingresos.loc[mask_ingresos_reales].groupby('mes')['cantidad'].sum()
                if mask_ingresos_reales.any()
                else pd.Series(dtype=float)
            )
        else:
            ingresos_reales_obj = pd.Series(dtype=float)

        objetivos_gastos_por_mes[cfg['nombre']] = gastos_obj
        objetivos_ingresos_por_mes[cfg['nombre']] = ingresos_obj
        objetivos_ingresos_reales_por_mes[cfg['nombre']] = ingresos_reales_obj

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
    
    if objetivos_saldos:
        ahorro -= sum(objetivos_saldos.values())

    inversiones = 0.0
    resumenes = []

    ahorro_emergencia = 0.0

    df_Fondo_reserva = fondo_reserva_general(lectura=False)
    fondo_cargado_inicial = float(df_Fondo_reserva.get('Cantidad cargada', 0.0))
    fondo_reserva = float(df_Fondo_reserva.get('Cantidad del fondo', 0.0))
    porcentaje = df_Fondo_reserva.get('Porcentaje', 0.0)

    # Solo si arranca en 2024-10, igual que tu if
    if fecha_inicio.to_period('M') == pd.Timestamp('2024-10-01').to_period('M'):
        fondo_cargado = 0.0
    else:
        fondo_cargado = fondo_cargado_inicial    


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
        if mes_actual >= pd.Timestamp("2024-10-01").to_period('M'):
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
            
            
        # Objetivos vista: aportaciones y gastos etiquetados
        total_aporte_salario = 0.0
        total_aporte_ingresos_reales = 0.0
        total_gasto_objetivos = 0.0
        resumen_objetivos_mes = {}

        for nombre in objetivos_nombres:
            cfg = objetivos_por_nombre[nombre]
            mes_inicio_objetivo = objetivos_mes_inicio.get(nombre)
            objetivo_activo = mes_inicio_objetivo is None or mes_actual >= mes_inicio_objetivo

            if not objetivo_activo:
                resumen_objetivos_mes[nombre] = 0.0
                continue

            if not objetivos_inicial_aplicado[nombre]:
                saldo_inicial_obj = objetivos_saldo_inicial[nombre]
                if saldo_inicial_obj:
                    ahorro_transaccional -= saldo_inicial_obj
                objetivos_saldos[nombre] += saldo_inicial_obj
                objetivos_inicial_aplicado[nombre] = True

            aporte_salario_teorico = ingreso_total * cfg['porcentaje_ingreso']

            ingresos_etiquetados = objetivos_ingresos_por_mes.get(nombre, pd.Series(dtype=float))
            ingresos_etiquetados_mes = float(ingresos_etiquetados.get(mes_actual_str, 0.0))

            ingresos_etiquetados_reales = objetivos_ingresos_reales_por_mes.get(nombre, pd.Series(dtype=float))
            ingresos_etiquetados_reales_mes = float(ingresos_etiquetados_reales.get(mes_actual_str, 0.0))

            gastos_etiquetados = objetivos_gastos_por_mes.get(nombre, pd.Series(dtype=float))
            gastos_etiquetados_mes = float(gastos_etiquetados.get(mes_actual_str, 0.0))

            objetivo_total_meta = objetivos_meta.get(nombre, {}).get('objetivo_total')
            saldo_actual_obj = objetivos_saldos[nombre]

            capacidad_restante = None
            if objetivo_total_meta is not None and objetivo_total_meta > 0:
                capacidad_restante = max(0.0, float(objetivo_total_meta) - saldo_actual_obj)

            if capacidad_restante is not None and capacidad_restante <= 1e-9:
                capacidad_restante = 0.0
                aporte_salario_real = 0.0
                aporte_ingresos_real = 0.0
            else:
                aporte_salario_real = aporte_salario_teorico
                if capacidad_restante is not None:
                    aporte_salario_real = min(aporte_salario_real, capacidad_restante)
                    capacidad_restante -= aporte_salario_real
                    capacidad_restante = max(0.0, capacidad_restante)

                aporte_ingresos_real = ingresos_etiquetados_mes
                if capacidad_restante is not None:
                    aporte_ingresos_real = min(aporte_ingresos_real, capacidad_restante)

            objetivos_saldos[nombre] += aporte_salario_real + aporte_ingresos_real - gastos_etiquetados_mes
            resumen_objetivos_mes[nombre] = round(objetivos_saldos[nombre], 2)

            total_aporte_salario += aporte_salario_real

            if ingresos_etiquetados_mes > 0 and aporte_ingresos_real > 0:
                proporcion_reales = aporte_ingresos_real / ingresos_etiquetados_mes
                proporcion_reales = min(max(proporcion_reales, 0.0), 1.0)
                aporte_ingresos_reales_real = ingresos_etiquetados_reales_mes * proporcion_reales
            else:
                aporte_ingresos_reales_real = 0.0

            total_aporte_ingresos_reales += aporte_ingresos_reales_real
            total_gasto_objetivos += gastos_etiquetados_mes

        if objetivos_nombres:
            ahorro_transaccional -= total_aporte_salario + total_aporte_ingresos_reales
            ahorro_transaccional += total_gasto_objetivos
        

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

        # ``inversiones`` refleja únicamente el dinero líquido reservado para
        # futuras aportaciones. El capital que ya salió de las cuentas reales y
        # vive en la cuenta "Invertido" se reporta en ``dinero_invertido`` pero
        # no forma parte de los saldos disponibles. Por eso el total debe
        # sumar directamente todas las bolsas de efectivo sin volver a restar
        # ``dinero_invertido`` (hacerlo descontaba dos veces las compras
        # efectivas de inversión).
        total_objetivos = sum(objetivos_saldos.values()) if objetivos_saldos else 0.0
        total = regalos + vacaciones + inversiones + ahorro + fondo_cargado + total_objetivos
        
        resumen = {
            'Mes': mes_actual_str,
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
        
        for nombre in objetivos_nombres:
            saldo_objetivo = resumen_objetivos_mes.get(nombre, round(objetivos_saldos.get(nombre, 0.0), 2))
            resumen[f"🎯 {nombre}"] = saldo_objetivo

            meta = objetivos_meta.get(nombre)
            objetivo_total_meta = meta.get('objetivo_total') if meta else None
            mes_inicio_objetivo = objetivos_mes_inicio.get(nombre)
            objetivo_activo = mes_inicio_objetivo is None or mes_actual >= mes_inicio_objetivo
            if objetivo_total_meta and objetivo_total_meta > 0 and objetivo_activo:
                progreso = max(0.0, min(1.0, objetivos_saldos[nombre] / objetivo_total_meta))
                resumen[f"% Objetivo {nombre}"] = round(progreso * 100, 2)
            elif objetivo_total_meta and objetivo_total_meta > 0:
                resumen[f"% Objetivo {nombre}"] = 0.0

        resumen.update(
            {
                'total': round(total, 2),
                '💳 Gasto del mes': round(gasto_total_mes_actual, 2),
                '💸 Presupuesto Mes': round(presupuesto_teorico, 2),
                '🧾 Presupuesto Disponible': round(presupuesto_efectivo, 2),
                '📉 Deuda Presupuestaria mensual': round(deuda_mensual, 2) if deuda_mensual > 0 else 0.0,
                '📉 Deuda Presupuestaria acumulada': round(deuda_acumulada, 2) if deuda_acumulada > 0 else 0.0,
            }
        )

        if mes_actual == pd.Timestamp('2024-10-01').to_period('M'):
            df_resumen_sep = pd.DataFrame([resumen]).copy(deep=True)
            df_resumen_sep = _renombrar_columnas_historial(df_resumen_sep)
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                df_resumen_sep.to_csv(output_path, index=False)

        resumenes.append(resumen)
        mes_actual += 1

    df_resumen = pd.DataFrame(resumenes).copy(deep=True)
    df_resumen = _renombrar_columnas_historial(df_resumen)
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_resumen.to_csv(output_path, index=False)
        
   # Actualizar el histórico del fondo de reserva con la instantánea recién calculada
    if actualizar_fondo_reserva:
        fondo_reserva_general(
            lectura=False,
            historial=df_resumen,
            fondo_cargado_actual=float(df_resumen.iloc[-1]['Fondo de reserva cargado'])
        )
        print("Es posible que los valores no sean precisos para el fondo de reserva, se debe observar cuando se ha obtenido el ingreso de fin de mes.")
    return df_resumen

def crear_historial_cuentas_virtuales_puro(*args, **kwargs):
    """Versión 'pura' del engine: no escribe archivos ni actualiza fondo de reserva.
    Devuelve el DataFrame calculado.
    """
    kwargs.setdefault("output_path", None)
    # evitar que el engine toque el fondo de reserva al final si existiera esa lógica
    kwargs.setdefault("actualizar_fondo_reserva", False)
    return crear_historial_cuentas_virtuales(*args, **kwargs)
