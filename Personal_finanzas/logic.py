#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 27 16:33:39 2025

@author: alex
"""

import re
import unicodedata
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from domain import Transaction
from config import saldos_iniciales as saldos_iniciales_config
from io_data import fondo_reserva_general
# =====================================================
# CLASIFICACIÓN Y PREPROCESADO
# =====================================================

def clasificar_ingreso(fila):
    categoria = fila["categoria"]
    etiquetas = str(fila.get("etiquetas", "")).lower()

    if categoria == "Salario":
        if "abuelos" in etiquetas:
            return "Vacaciones"
        if "mi regalo" in etiquetas:
            return "Regalos"
        return "Ingreso Real"

    elif categoria == "Interes":
        return "Rendimiento Financiero"

    elif categoria in ["Transporte", "Hosteleria", "Entretenimiento", "Alimentacion", "Otros"]:
        # Esto es una devolución por gastos compartidos
        if "mi regalo" in etiquetas:
            return "Regalos"
        elif "fondo reserva" in etiquetas:
            return "Fondo reserva"
        elif "vacaciones" in etiquetas:
            return "Vacaciones"
        return f"Reembolso {categoria}"

    else:
        return "Ingreso No Clasificado"


def procesar_ingresos(df_ingresos):
    df_ingresos = df_ingresos.copy()
    df_ingresos["tipo_logico"] = df_ingresos.apply(clasificar_ingreso, axis=1)
    return df_ingresos


def calcular_gastos_netos(df_gastos, df_ingresos):
    gastos_por_cat = df_gastos.groupby("categoria")["cantidad"].sum()
    reembolsos = df_ingresos[df_ingresos["tipo_logico"].str.startswith("Reembolso")].copy()
    reembolsos["categoria_reembolso"] = reembolsos["tipo_logico"].str.replace("Reembolso ", "", regex=False)
    reembolsos_por_cat = reembolsos.groupby("categoria_reembolso")["cantidad"].sum()
    gastos_netos = gastos_por_cat.subtract(reembolsos_por_cat, fill_value=0)
    return gastos_netos


def resumen_ingresos(df_ingresos):
    return df_ingresos[df_ingresos["tipo_logico"] == "Ingreso Real"]["cantidad"].sum()


def resumen_gastos(df_gastos, df_ingresos):
    df_gastos_explicacion = df_gastos.copy()
    df_ingresos_explicacion = df_ingresos.copy()

    df_gastos_explicacion["mes"] = df_gastos_explicacion["fecha"].dt.to_period("M").astype(str)
    df_ingresos_explicacion["mes"] = df_ingresos_explicacion["fecha"].dt.to_period("M").astype(str)

    gastos_mensuales_explicacion = (
        df_gastos_explicacion.groupby(["mes", "tipo_logico"])["cantidad"].sum().rename("gastos").reset_index()
    )
    ingresos_mensuales_explicacion = (
        df_ingresos_explicacion.groupby(["mes", "tipo_logico"])["cantidad"].sum().rename("ingresos").reset_index()
    )

    df_explicacion = pd.merge(
        ingresos_mensuales_explicacion,
        gastos_mensuales_explicacion,
        on=["mes", "tipo_logico"],
        how="outer",
    ).fillna(0)

    df_explicacion["grupo"] = df_explicacion["tipo_logico"].apply(
        lambda x: "Ingreso Real" if x == "Ingreso Real" else ("Reembolso" if str(x).startswith("Reembolso") else "Otro")
    )

    df_explicacion = df_explicacion.loc[df_explicacion["grupo"].isin(["Reembolso"])]

    df_explicacion["balance"] = df_explicacion["gastos"] - df_explicacion["ingresos"]
    df_explicacion = df_explicacion.drop(["ingresos", "gastos", "grupo"], axis=1)

    pivot_explicacion = df_explicacion.pivot_table(index="mes", columns="tipo_logico", values="balance")

    cols = ["alimentacion", "entretenimiento", "hosteleria", "otros", "transporte"]
    pivot_explicacion = pivot_explicacion.rename(columns=str.lower)
    pivot_explicacion = pivot_explicacion.reindex(columns=cols, fill_value=0)
    pivot_explicacion = pivot_explicacion.fillna(0)

    return pivot_explicacion


def resumen_mensual(df_gastos, df_ingresos):
    df_gastos = df_gastos.copy()
    df_ingresos = df_ingresos.copy()

    df_gastos["mes"] = df_gastos["fecha"].dt.to_period("M").astype(str)
    df_ingresos["mes"] = df_ingresos["fecha"].dt.to_period("M").astype(str)

    gastos_mensuales = (
        df_gastos.groupby(["mes", "tipo_logico"])["cantidad"].sum().rename("gastos").reset_index()
    )
    ingresos_mensuales = (
        df_ingresos.groupby(["mes", "tipo_logico"])["cantidad"].sum().rename("ingresos").reset_index()
    )

    df_resumen = pd.merge(ingresos_mensuales, gastos_mensuales, on=["mes", "tipo_logico"], how="outer").fillna(0)

    df_resumen["grupo"] = df_resumen["tipo_logico"].apply(
        lambda x: "Ingreso Real" if x == "Ingreso Real" else ("Reembolso" if str(x).startswith("Reembolso") else "Otro")
    )

    df_resumen = df_resumen.loc[df_resumen["grupo"].isin(["Ingreso Real", "Reembolso"])]
    df_resumen = df_resumen.groupby(["mes", "grupo"])[["ingresos", "gastos"]].sum().reset_index()

    df_resumen["balance"] = abs(df_resumen["ingresos"] - df_resumen["gastos"])
    df_resumen = df_resumen.drop(["ingresos", "gastos"], axis=1)

    pivot = df_resumen.pivot_table(index="mes", columns="grupo", values="balance", aggfunc="sum")
    pivot = pivot.rename(columns={"Ingreso Real": "ingresos", "Reembolso": "gastos"})
    pivot["balance"] = pivot["ingresos"] - pivot["gastos"]

    return pivot


# =====================================================
# UTILIDADES
# =====================================================

def eliminar_tildes(texto):
    if isinstance(texto, str):
        return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("utf-8")
    return texto


def agregar_fila_ingresos(dataframe, val1, val2, val3, val4, val5, val6):
    nueva_fila = pd.DataFrame(
        [{
            "fecha": val1,
            "categoria": val2,
            "cuenta": val3,
            "cantidad": val4,
            "etiquetas": val5,
            "comentario": val6
        }]
    )
    return pd.concat([dataframe, nueva_fila], ignore_index=True)


# =====================================================
# CARGA A PRESUPUESTO (SIN IO)
# =====================================================

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
            timestamp=pd.to_datetime("2000-01-01"),
        )
        presupuesto.add_transaction(transaccion_inicial)


def cargar_transacciones(df, cuentas_obj, presupuesto, signo=1):
    for _, fila in df.iterrows():
        monto = signo * fila["cantidad"]
        cuenta = cuentas_obj[fila["cuenta"]]
        transaccion = Transaction(
            amount=monto,
            description=str(fila["comentario"]),
            category=str(fila["categoria"]),
            account=cuenta,
            timestamp=fila["fecha"],
        )
        presupuesto.add_transaction(transaccion)


# =====================================================
# CÁLCULOS DE INVERSIÓN / HISTÓRICO
# =====================================================

def invertido_en_mes(presupuesto, mes_periodo: pd.Period):
    """
    Devuelve el dinero invertido en un mes determinado.

    - Si mes_periodo es pasado: toma el último día del mes.
    - Si mes_periodo es el actual: toma el día actual.
    """
    hoy = pd.Timestamp(datetime.today())
    mes_actual = hoy.to_period("M")

    if mes_periodo < mes_actual:
        fecha_objetivo = mes_periodo.to_timestamp(how="end")
    elif mes_periodo == mes_actual:
        fecha_objetivo = hoy
    else:
        raise ValueError("El mes indicado es en el futuro y no puede calcularse.")

    invertido = presupuesto.balances_a_fecha(fecha_objetivo)["Invertido"]
    return invertido


def crear_historial_cuentas_virtuales(
    df_ingresos,
    df_gastos,
    presupuesto,
    fecha_inicio="2024-10-01",
    porcentaje_gasto=0.3,
    porcentaje_inversion=0.1,
    porcentaje_vacaciones=0.05,
    saldos_iniciales=None,
    output_path=None,  # se mantiene por compatibilidad, pero aquí no se usa
    objetivos_config=None,  # debe ser lista de dicts YA NORMALIZADA o None
    fondo_reserva_snapshot=None,  # dict/Series con claves: 'Cantidad cargada', 'Cantidad del fondo', 'Porcentaje'
):
    """
    Lógica pura: calcula df_resumen y lo devuelve.
    NO guarda CSV, NO imprime, NO llama a IO.

    - objetivos_config: lista de objetivos (como devuelve io_data.cargar_objetivos_vista)
    - fondo_reserva_snapshot: última instantánea del fondo (si no se pasa, asume 0)
    """

    # Mantener compatibilidad con tu comportamiento
    if saldos_iniciales is None:
        saldos_iniciales = saldos_iniciales_config
        
        


    # Objetivos ya vienen cargados desde IO/UI; si no, lista vacía
    if objetivos_config is None:
        objetivos_config = []
    objetivos_nombres = [cfg["nombre"] for cfg in objetivos_config]
    porcentaje_objetivos = sum(cfg["porcentaje_ingreso"] for cfg in objetivos_config)

    porcentaje_total = porcentaje_gasto + porcentaje_inversion + porcentaje_vacaciones + porcentaje_objetivos
    if porcentaje_total > 1 + 1e-6:
        raise ValueError(
            "La suma de porcentajes (gasto, inversiones, vacaciones y objetivos vista) supera el 100%."
        )

    objetivos_por_nombre = {cfg["nombre"]: cfg for cfg in objetivos_config}
    if len(objetivos_por_nombre) != len(objetivos_config):
        raise ValueError("Los objetivos vista deben tener nombres únicos.")

    objetivos_meta = {
        nombre: {
            "objetivo_total": cfg.get("objetivo_total"),
            "horizonte_meses": cfg.get("horizonte_meses"),
            "mes_inicio": cfg.get("mes_inicio"),
        }
        for nombre, cfg in objetivos_por_nombre.items()
    }

    objetivos_saldos = {nombre: 0.0 for nombre in objetivos_por_nombre}
    objetivos_saldo_inicial = {nombre: cfg["saldo_inicial"] for nombre, cfg in objetivos_por_nombre.items()}
    objetivos_mes_inicio = {nombre: cfg.get("mes_inicio") for nombre, cfg in objetivos_por_nombre.items()}
    objetivos_inicial_aplicado = {nombre: False for nombre in objetivos_por_nombre}

    def _renombrar_columnas_historial(df: pd.DataFrame) -> pd.DataFrame:
        columnas_base = {
            "Mes": "Mes",
            "🎁 Regalos": "Regalos",
            "💼 Vacaciones": "Vacaciones",
            "📈 Inversiones": "Inversiones",
            "Dinero Invertido": "Dinero Invertido",
            "💰 Ahorros": "Ahorros",
            "Fondo de reserva cargado": "Fondo de reserva cargado",
            "total": "total",
            "💳 Gasto del mes": "Gasto del mes",
            "💸 Presupuesto Mes": "Presupuesto Mes",
            "🧾 Presupuesto Disponible": "Presupuesto Disponible",
            "📉 Deuda Presupuestaria mensual": "Deuda Presupuestaria mensual",
            "📉 Deuda Presupuestaria acumulada": "Deuda Presupuestaria acumulada",
        }

        renombradas = {}
        for columna in df.columns:
            if columna in columnas_base:
                renombradas[columna] = columnas_base[columna]
            elif columna.startswith("🎯 "):
                nombre_objetivo = columna.split(" ", 1)[1] if " " in columna else columna.replace("🎯", "").strip()
                renombradas[columna] = f"Objetivo {nombre_objetivo}"

        return df.rename(columns=renombradas)

    # Copias defensivas
    df_gastos = df_gastos.copy()
    df_ingresos = df_ingresos.copy()

    # Pipeline original
    df_ingresos = procesar_ingresos(df_ingresos)

    # Mes como string
    df_gastos["mes"] = df_gastos["fecha"].dt.to_period("M").astype(str)
    df_ingresos["mes"] = df_ingresos["fecha"].dt.to_period("M").astype(str)

    gastos_por_mes = {mes: grupo.copy() for mes, grupo in df_gastos.groupby("mes")}
    ingresos_por_mes = {mes: grupo.copy() for mes, grupo in df_ingresos.groupby("mes")}
    meses_disponibles = sorted(set(gastos_por_mes) | set(ingresos_por_mes))

    regalos_gasto = df_gastos[df_gastos["tipo_logico"] == "Regalos"].groupby("mes")["cantidad"].sum()
    regalos_ingreso = df_ingresos[df_ingresos["tipo_logico"] == "Regalos"].groupby("mes")["cantidad"].sum()

    vacaciones_gasto = df_gastos[df_gastos["etiquetas"] == "Vacaciones"].groupby("mes")["cantidad"].sum()
    vacaciones_ingreso = df_ingresos[df_ingresos["tipo_logico"] == "Vacaciones"].groupby("mes")["cantidad"].sum()

    fondo_reserva_gasto = df_gastos[df_gastos["categoria"] == "FondoReserva"].groupby("mes")["cantidad"].sum()

    intereses_ingreso = df_ingresos[df_ingresos["tipo_logico"] == "Rendimiento Financiero"].groupby("mes")["cantidad"].sum()

    objetivos_gastos_por_mes = {}
    objetivos_ingresos_por_mes = {}
    objetivos_ingresos_reales_por_mes = {}

    if objetivos_config:
        etiquetas_gastos = df_gastos["etiquetas"].fillna("").str.lower()
        etiquetas_ingresos = df_ingresos["etiquetas"].fillna("").str.lower()

        for cfg in objetivos_config:
            etiquetas = cfg["etiquetas"]
            patron = "|".join(re.escape(et) for et in etiquetas if et) if etiquetas else ""

            if patron:
                mask_gastos = etiquetas_gastos.str.contains(patron, regex=True)
                mask_ingresos = etiquetas_ingresos.str.contains(patron, regex=True)
            else:
                mask_gastos = pd.Series(False, index=df_gastos.index)
                mask_ingresos = pd.Series(False, index=df_ingresos.index)

            gastos_obj = (
                df_gastos.loc[mask_gastos].groupby("mes")["cantidad"].sum()
                if mask_gastos.any()
                else pd.Series(dtype=float)
            )
            ingresos_obj = (
                df_ingresos.loc[mask_ingresos].groupby("mes")["cantidad"].sum()
                if mask_ingresos.any()
                else pd.Series(dtype=float)
            )

            if mask_ingresos.any():
                mask_ingresos_reales = mask_ingresos & (df_ingresos["tipo_logico"] == "Ingreso Real")
                ingresos_reales_obj = (
                    df_ingresos.loc[mask_ingresos_reales].groupby("mes")["cantidad"].sum()
                    if mask_ingresos_reales.any()
                    else pd.Series(dtype=float)
                )
            else:
                ingresos_reales_obj = pd.Series(dtype=float)

            objetivos_gastos_por_mes[cfg["nombre"]] = gastos_obj
            objetivos_ingresos_por_mes[cfg["nombre"]] = ingresos_obj
            objetivos_ingresos_reales_por_mes[cfg["nombre"]] = ingresos_reales_obj
    else:
        etiquetas_gastos = None
        etiquetas_ingresos = None

    empty_gastos = df_gastos.head(0)
    empty_ingresos = df_ingresos.head(0)
    gastos_netos_por_mes = {}
    for mes in meses_disponibles:
        gastos_mes = gastos_por_mes.get(mes, empty_gastos)
        ingresos_mes = ingresos_por_mes.get(mes, empty_ingresos)
        gastos_netos_por_mes[mes] = calcular_gastos_netos(gastos_mes, ingresos_mes)

    # Fechas
    fecha_actual = max(df_ingresos["fecha"].max(), df_gastos["fecha"].max())
    fecha_inicio = pd.Timestamp(fecha_inicio)
    fecha_fin = pd.Timestamp(fecha_actual)

    deuda_acumulada = 0.0
    regalos = 0.0

    vacaciones_inicial = saldos_iniciales.get("Metálico", 0.0)
    vacaciones = vacaciones_inicial

    ahorro = 0.0
    ahorro_inicial = sum(v for k, v in saldos_iniciales.items() if k != "Metálico")
    ahorro += ahorro_inicial

    if objetivos_saldos:
        ahorro -= sum(objetivos_saldos.values())

    inversiones = 0.0
    resumenes = []
    ahorro_emergencia = 0.0

    # Fondo reserva: en lógica pura NO leemos CSV.
    # Si no te pasan snapshot, asumimos 0.
    if fondo_reserva_snapshot is None:
        fondo_cargado_inicial = 0.0
        fondo_reserva = 0.0
        porcentaje = 0.0
    else:
        fondo_cargado_inicial = float(fondo_reserva_snapshot.get("Cantidad cargada", 0.0))
        fondo_reserva = float(fondo_reserva_snapshot.get("Cantidad del fondo", 0.0))
        porcentaje = float(fondo_reserva_snapshot.get("Porcentaje", 0.0))

    if fecha_inicio.to_period("M") == pd.Timestamp("2024-10-01").to_period("M"):
        fondo_cargado = 0.0
    else:
        fondo_cargado = fondo_cargado_inicial

    ingresos_reales = df_ingresos[df_ingresos["tipo_logico"] == "Ingreso Real"]
    ingresos_reales_mensual = ingresos_reales.groupby("mes")["cantidad"].sum()

    mes_actual = fecha_inicio.to_period("M")
    while mes_actual <= fecha_fin.to_period("M"):
        mes_actual_str = mes_actual.strftime("%Y-%m")
        mes_anterior = (mes_actual - 1).strftime("%Y-%m")

        # REGALOS
        gasto_regalos = float(regalos_gasto.get(mes_actual_str, 0.0))
        ingreso_regalos = float(regalos_ingreso.get(mes_actual_str, 0.0))
        regalos_mes = ingreso_regalos - gasto_regalos
        regalos += regalos_mes

        # VACACIONES
        gasto_vacaciones = float(vacaciones_gasto.get(mes_actual_str, 0.0))

        if mes_actual >= pd.Timestamp("2025-09-01").to_period("M"):
            vacaciones_mensual = float(ingresos_reales_mensual.get(mes_actual_str, 0.0)) * porcentaje_vacaciones
        else:
            vacaciones_mensual = 0.0

        vacaciones_ingresos = float(vacaciones_ingreso.get(mes_actual_str, 0.0))
        vacaciones += vacaciones_ingresos + vacaciones_mensual - gasto_vacaciones

        # FONDO RESERVA GASTO
        gasto_fondo_reserva = float(fondo_reserva_gasto.get(mes_actual_str, 0.0))

        # GASTOS NETOS
        gasto_total_mes_actual_aux = float(
            gastos_netos_por_mes.get(mes_actual_str, pd.Series(dtype=float)).sum()
        )
        gasto_total_mes_actual = gasto_total_mes_actual_aux - gasto_vacaciones - gasto_regalos - gasto_fondo_reserva

        # PRESUPUESTO
        presupuesto_teorico = float(ingresos_reales_mensual.get(mes_anterior, 0.0)) * porcentaje_gasto
        presupuesto_disponible = max(0.0, presupuesto_teorico - deuda_acumulada)

        deuda_mensual = gasto_total_mes_actual - presupuesto_teorico
        exceso_gasto = deuda_mensual + deuda_acumulada
        deuda_acumulada = max(0.0, exceso_gasto)

        # INVERSIONES
        interes = float(intereses_ingreso.get(mes_actual_str, 0.0))
        if mes_actual >= pd.Timestamp("2024-10-01").to_period("M"):
            inv_mensual = float(ingresos_reales_mensual.get(mes_actual_str, 0.0)) * porcentaje_inversion
        else:
            inv_mensual = 0.0

        inversion_mes = inv_mensual + interes
        dinero_invertido_entre_mes = invertido_en_mes(presupuesto, mes_actual) - invertido_en_mes(presupuesto, mes_actual - 1)

        inversiones += inversion_mes - dinero_invertido_entre_mes
        dinero_invertido = invertido_en_mes(presupuesto, mes_actual)

        # AHORRO
        ingreso_total = float(ingresos_reales_mensual.get(mes_actual_str, 0.0))
        gastos_netos_total = float(gastos_netos_por_mes.get(mes_actual_str, pd.Series(dtype=float)).sum())

        presupuesto_efectivo = max(0.0, presupuesto_disponible - gasto_total_mes_actual)

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

        # OBJETIVOS VISTA
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

            aporte_salario_teorico = ingreso_total * cfg["porcentaje_ingreso"]

            ingresos_etiquetados = objetivos_ingresos_por_mes.get(nombre, pd.Series(dtype=float))
            ingresos_etiquetados_mes = float(ingresos_etiquetados.get(mes_actual_str, 0.0))

            ingresos_etiquetados_reales = objetivos_ingresos_reales_por_mes.get(nombre, pd.Series(dtype=float))
            ingresos_etiquetados_reales_mes = float(ingresos_etiquetados_reales.get(mes_actual_str, 0.0))

            gastos_etiquetados = objetivos_gastos_por_mes.get(nombre, pd.Series(dtype=float))
            gastos_etiquetados_mes = float(gastos_etiquetados.get(mes_actual_str, 0.0))

            objetivo_total_meta = objetivos_meta.get(nombre, {}).get("objetivo_total")
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
                    capacidad_restante = max(0.0, capacidad_restante - aporte_salario_real)

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

        # FONDO RESERVA (misma lógica, pero sin IO)
        if mes_actual == pd.Timestamp("2024-10-01").to_period("M"):
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

        total_objetivos = sum(objetivos_saldos.values()) if objetivos_saldos else 0.0
        total = regalos + vacaciones + inversiones + ahorro + fondo_cargado + total_objetivos

        resumen = {
            "Mes": mes_actual_str,
            "🎁 Regalos": round(regalos, 2),
            "💼 Vacaciones": round(vacaciones, 2),
            "📈 Inversiones": round(inversiones, 2),
            "Dinero Invertido": round(dinero_invertido, 2),
            "💰 Ahorros": round(ahorro, 2),
            "Fondo de reserva cargado": round(fondo_cargado, 4),
            "total": round(total, 2),
            "💳 Gasto del mes": round(gasto_total_mes_actual, 2),
            "💸 Presupuesto Mes": round(presupuesto_teorico, 2),
            "🧾 Presupuesto Disponible": round(presupuesto_efectivo, 2),
            "📉 Deuda Presupuestaria mensual": round(deuda_mensual, 2) if deuda_mensual > 0 else 0.0,
            "📉 Deuda Presupuestaria acumulada": round(deuda_acumulada, 2) if deuda_acumulada > 0 else 0.0,
        }

        for nombre in objetivos_nombres:
            saldo_objetivo = resumen_objetivos_mes.get(nombre, round(objetivos_saldos.get(nombre, 0.0), 2))
            resumen[f"🎯 {nombre}"] = saldo_objetivo

            meta = objetivos_meta.get(nombre)
            objetivo_total_meta = meta.get("objetivo_total") if meta else None
            mes_inicio_objetivo = objetivos_mes_inicio.get(nombre)
            objetivo_activo = mes_inicio_objetivo is None or mes_actual >= mes_inicio_objetivo

            if objetivo_total_meta and objetivo_total_meta > 0 and objetivo_activo:
                progreso = max(0.0, min(1.0, objetivos_saldos[nombre] / objetivo_total_meta))
                resumen[f"% Objetivo {nombre}"] = round(progreso * 100, 2)
            elif objetivo_total_meta and objetivo_total_meta > 0:
                resumen[f"% Objetivo {nombre}"] = 0.0

        resumenes.append(resumen)
        mes_actual += 1

    df_resumen = pd.DataFrame(resumenes).copy(deep=True)
    df_resumen = _renombrar_columnas_historial(df_resumen)

    return df_resumen


# =====================================================
# RESUMEN GLOBAL (mantiene prints fuera: aquí devolvemos datos)
# =====================================================

def resumen_global(presupuesto, fecha):
    fecha_objetivo = pd.to_datetime(fecha)
    saldos_a_fecha = presupuesto.balances_a_fecha(fecha_objetivo)
    saldo_total = presupuesto.balance_total_a_fecha(fecha_objetivo)
    return saldo_total, saldos_a_fecha
