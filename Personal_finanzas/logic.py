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
    """
    Devuelve un pivot (mes x categorias) con el gasto neto:
      gasto_neto_categoria = gastos_categoria - reembolsos_categoria

    Categorías (columnas) en minúsculas sin tildes como venías usando en plots.
    """
    df_g = df_gastos.copy()
    df_i = df_ingresos.copy()

    df_g["mes"] = df_g["fecha"].dt.to_period("M").astype(str)
    df_i["mes"] = df_i["fecha"].dt.to_period("M").astype(str)

    # 1) Gastos brutos por mes y categoría (usa 'categoria' del Excel)
    g = (
        df_g.groupby(["mes", "categoria"])["cantidad"]
        .sum()
        .rename("gasto")
        .reset_index()
    )

    # Normalización de nombre de categoría (como en tus plots)
    g["categoria"] = g["categoria"].astype(str).str.lower()

    # 2) Reembolsos por mes y categoría (vienen desde ingresos tipo_logico="Reembolso X")
    reemb = df_i[df_i["tipo_logico"].astype(str).str.startswith("Reembolso")].copy()
    if not reemb.empty:
        reemb["categoria"] = reemb["tipo_logico"].astype(str).str.replace("Reembolso ", "", regex=False).str.lower()
        r = (
            reemb.groupby(["mes", "categoria"])["cantidad"]
            .sum()
            .rename("reembolso")
            .reset_index()
        )
    else:
        r = pd.DataFrame(columns=["mes", "categoria", "reembolso"])

    # 3) Neto = gasto - reembolso (si no hay reembolso, 0)
    merged = g.merge(r, on=["mes", "categoria"], how="left").fillna({"reembolso": 0})
    merged["neto"] = merged["gasto"] - merged["reembolso"]

    # 4) Pivot final (mes x categoria)
    pivot = merged.pivot_table(index="mes", columns="categoria", values="neto", aggfunc="sum").fillna(0)

    # Orden de columnas fijo si existe
    cols = ["alimentacion", "transporte", "hosteleria", "entretenimiento", "otros"]
    pivot = pivot.reindex(columns=cols, fill_value=0)

    return pivot



def _split_tags(s: str):
    if s is None:
        return []
    s = str(s).strip().lower()
    if not s:
        return []
    # etiquetas coma-separadas (como usas en Streamlit)
    return [t.strip() for t in s.split(",") if t.strip()]


def resumen_mensual(df_gastos, df_ingresos):
    df_gastos = df_gastos.copy()
    df_ingresos = df_ingresos.copy()

    df_gastos["mes"] = df_gastos["fecha"].dt.to_period("M").astype(str)
    df_ingresos["mes"] = df_ingresos["fecha"].dt.to_period("M").astype(str)

    # 1) ingresos reales
    ingresos_real = (
        df_ingresos.loc[df_ingresos["tipo_logico"] == "Ingreso Real"]
        .groupby("mes")["cantidad"]
        .sum()
        .rename("ingresos")
    )

    # 2) gastos totales
    gastos_totales = (
        df_gastos.groupby("mes")["cantidad"]
        .sum()
        .rename("gastos_totales")
    )

    # 3) reembolsos clásicos (lo que ya tenías como "Reembolso ...")
    reembolsos = (
        df_ingresos.loc[df_ingresos["tipo_logico"].astype(str).str.startswith("Reembolso")]
        .groupby("mes")["cantidad"]
        .sum()
        .rename("reembolsos")
    )

    # 4) compensaciones por etiqueta:
    #    - un ingreso con etiqueta X compensa gastos con etiqueta X en el mismo mes
    #    - no toca "Ingreso Real" (sueldo), para evitar que el sueldo se interprete como compensación
    g = df_gastos[["mes", "cantidad", "etiquetas"]].copy()
    i = df_ingresos[["mes", "cantidad", "etiquetas", "tipo_logico"]].copy()

    g["tags"] = g["etiquetas"].apply(_split_tags)
    i["tags"] = i["etiquetas"].apply(_split_tags)

    # quedarnos solo con ingresos NO salariales con alguna etiqueta
    i = i[(i["tipo_logico"] != "Ingreso Real") & (i["tags"].apply(len) > 0)].copy()
    g = g[g["tags"].apply(len) > 0].copy()

    if len(g) and len(i):
        g_exp = g.explode("tags")
        i_exp = i.explode("tags")

        # tags existentes en gastos por mes
        tags_gastos = g_exp[["mes", "tags"]].drop_duplicates()

        # ingresos cuya etiqueta existe en gastos del mismo mes
        i_match = i_exp.merge(tags_gastos, on=["mes", "tags"], how="inner")

        compensaciones_etiqueta = (
            i_match.groupby("mes")["cantidad"]
            .sum()
            .rename("comp_etiqueta")
        )
    else:
        compensaciones_etiqueta = pd.Series(dtype=float, name="comp_etiqueta")

    # 5) unir y calcular netos
    out = pd.concat([ingresos_real, gastos_totales, reembolsos, compensaciones_etiqueta], axis=1).fillna(0)

    # OJO: compensaciones = reembolsos + ingresos con etiqueta coincidente
    out["compensaciones"] = out["reembolsos"] + out["comp_etiqueta"]

    # gasto neto (no permitir que baje de 0 por seguridad)
    out["gastos"] = (out["gastos_totales"] - out["compensaciones"]).clip(lower=0)

    out["balance"] = out["ingresos"] - out["gastos"]

    return out[["ingresos", "gastos", "balance"]]



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
    output_path=None,  # compatibilidad, no se usa
    objetivos_config=None,  # lista de dicts YA NORMALIZADA o None
    fondo_reserva_snapshot=None,  # dict/Series: 'Cantidad cargada', 'Cantidad del fondo', 'Porcentaje'
):
    """
    Lógica pura: calcula df_resumen y objetivos_df y los devuelve.
    NO guarda CSV, NO imprime, NO llama a IO.

    OBJETIVOS (NUEVO):
      - Campos: nombre, etiquetas, fraccion_presupuesto, duracion_meses, mes_inicio, saldo_inicial(opc)
      - Aporte mensual = fraccion_presupuesto * (porcentaje_gasto * ingreso_real_mes_anterior)
      - Movimientos con etiqueta del objetivo se excluyen del gasto corriente y se imputan al objetivo.
      - Saldo objetivo puede ser negativo durante su vida.
      - Liquidación al final de mes_fin:
          * saldo > 0 -> suma a ahorro
          * saldo < 0 -> suma a deuda_acumulada y resta a ahorro
    """

    import numpy as np
    import pandas as pd
    import re

    # -------------------------
    # Helpers (locales)
    # -------------------------
    def _mes_fin_objetivo(mes_inicio: pd.Period, duracion_meses: int) -> pd.Period:
        return mes_inicio + (int(duracion_meses) - 1)

    def _to_period_m(x):
        """Convierte 'YYYY-MM' / Timestamp / Period a pd.Period('M')."""
        if x is None or x == "":
            return None
        if isinstance(x, pd.Period):
            return x.asfreq("M")
        if isinstance(x, pd.Timestamp):
            return x.to_period("M")
        # string
        return pd.Period(str(x)[:7], freq="M")
    
    
    def _objetivo_activo(cfg, mes: pd.Period) -> bool:
        mes_inicio = _to_period_m(cfg.get("mes_inicio"))
        if mes_inicio is None:
            return False  # o True si quieres que por defecto empiece "ya", pero tú estás exigiendo mes_inicio
    
        duracion = int(cfg.get("duracion_meses", 0))
        if duracion <= 0:
            return False
    
        mes_fin = _mes_fin_objetivo(mes_inicio, duracion)
        return mes_inicio <= mes <= mes_fin

    def _objetivo_vence(cfg, mes: pd.Period) -> bool:
        mes_inicio = _to_period_m(cfg.get("mes_inicio"))
        if mes_inicio is None:
            return False
        return mes == _mes_fin_objetivo(mes_inicio, int(cfg.get("duracion_meses", 0)))


    def _validar_historial_10m(ingresos_reales_mensual: pd.Series, mes_inicio: pd.Period, nombre: str):
        """
        Exige 10 meses completos ANTERIORES a mes_inicio.
        ingresos_reales_mensual index: 'YYYY-MM'
        """
        meses_previos = [(mes_inicio - i).strftime("%Y-%m") for i in range(1, 11)]
        faltan = [m for m in meses_previos if m not in ingresos_reales_mensual.index]
        if faltan:
            raise ValueError(
                f"Objetivo '{nombre}': se necesitan 10 meses de histórico de ingresos reales "
                f"antes de {mes_inicio.strftime('%Y-%m')}. Faltan: {faltan}"
            )

    def _media_ingresos_10m(ingresos_reales_mensual, mes_inicio):
        mes_inicio = _to_period_m(mes_inicio)  # <-- CLAVE
        if mes_inicio is None:
            raise ValueError("mes_inicio es obligatorio para calcular la media de 10 meses (formato 'YYYY-MM').")
    
        meses_previos = [(mes_inicio - i).strftime("%Y-%m") for i in range(1, 11)]
        valores = [float(ingresos_reales_mensual.get(m, 0.0)) for m in meses_previos]
        return float(np.mean(valores))

    # -------------------------
    # Compatibilidad saldos iniciales
    # -------------------------
    if saldos_iniciales is None:
        saldos_iniciales = saldos_iniciales_config

    # -------------------------
    # Objetivos: ya vienen cargados/normalizados fuera.
    # Si no, lista vacía.
    # -------------------------
    if objetivos_config is None:
        objetivos_config = []

    # --- VALIDACIÓN NUEVOS OBJETIVOS (presupuesto objetivo) ---
    # Campos obligatorios nuevos
    required = ("nombre", "etiquetas", "fraccion_presupuesto", "duracion_meses")
    
    for cfg in objetivos_config:
        for k in required:
            if k not in cfg:
                raise ValueError(f"Objetivo inválido, falta '{k}': {cfg}")
    
        # nombre
        if not str(cfg["nombre"]).strip():
            raise ValueError(f"Objetivo inválido, nombre vacío: {cfg}")
    
        # etiquetas -> lista de strings minúsculas
        if isinstance(cfg["etiquetas"], str):
            etiquetas = [e.strip().lower() for e in cfg["etiquetas"].split(",") if e.strip()]
        else:
            etiquetas = [str(e).strip().lower() for e in cfg["etiquetas"] if str(e).strip()]
        cfg["etiquetas"] = etiquetas
    
        # fraccion_presupuesto
        fr = float(cfg["fraccion_presupuesto"])
        if fr < 0 or fr > 1:
            raise ValueError(f"Objetivo '{cfg['nombre']}': fraccion_presupuesto fuera de [0,1].")
        cfg["fraccion_presupuesto"] = fr
    
        # duracion_meses
        dm = int(cfg["duracion_meses"])
        if dm <= 0:
            raise ValueError(f"Objetivo '{cfg['nombre']}': duracion_meses debe ser > 0.")
        cfg["duracion_meses"] = dm
    
    # suma fracciones <= 1
    total_fr = sum(float(o["fraccion_presupuesto"]) for o in objetivos_config)
    if total_fr > 1 + 1e-9:
        raise ValueError("La suma de fraccion_presupuesto de los objetivos supera 1.")
    # --- FIN VALIDACIÓN ---

    # nombres únicos
    nombres = [c["nombre"] for c in objetivos_config]
    if len(nombres) != len(set(nombres)):
        raise ValueError("Los objetivos deben tener nombres únicos.")

    # -------------------------
    # Copias defensivas y pipeline original
    # -------------------------
    df_gastos = df_gastos.copy()
    df_ingresos = df_ingresos.copy()

    df_ingresos = procesar_ingresos(df_ingresos)

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

    # Gastos netos por mes (tu lógica: gasto bruto - reembolsos)
    empty_gastos = df_gastos.head(0)
    empty_ingresos = df_ingresos.head(0)
    gastos_netos_por_mes = {}
    for mes in meses_disponibles:
        gastos_mes = gastos_por_mes.get(mes, empty_gastos)
        ingresos_mes = ingresos_por_mes.get(mes, empty_ingresos)
        gastos_netos_por_mes[mes] = calcular_gastos_netos(gastos_mes, ingresos_mes)

    # -------------------------
    # Fechas
    # -------------------------
    fecha_actual = max(df_ingresos["fecha"].max(), df_gastos["fecha"].max())
    fecha_inicio = pd.Timestamp(fecha_inicio)
    fecha_fin = pd.Timestamp(fecha_actual)

    # -------------------------
    # Inicializaciones (tu lógica)
    # -------------------------
    deuda_acumulada = 0.0
    regalos = 0.0

    vacaciones_inicial = saldos_iniciales.get("Metálico", 0.0)
    vacaciones = vacaciones_inicial

    ahorro = 0.0
    ahorro_inicial = sum(v for k, v in saldos_iniciales.items() if k != "Metálico")
    ahorro += ahorro_inicial

    inversiones = 0.0
    resumenes = []
    ahorro_emergencia = 0.0

    # Fondo reserva snapshot (lógica pura)
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

    # -------------------------
    # Objetivos: preparar separación por etiquetas
    # -------------------------
    objetivos_saldos = {cfg["nombre"]: float(cfg.get("saldo_inicial", 0.0)) for cfg in objetivos_config}

    objetivos_gastos_por_mes = {}
    objetivos_ingresos_por_mes = {}

    # --- Normalización/validación robusta de objetivos ---
    if objetivos_config is None:
        objetivos_config = []
    
    objetivos_norm = []
    for cfg in objetivos_config:
        # Asegura dict (por si viene como Series)
        if not isinstance(cfg, dict):
            cfg = dict(cfg)
    
        nombre = str(cfg.get("nombre", "")).strip()
        if not nombre:
            raise ValueError(f"Objetivo inválido: nombre vacío: {cfg}")
    
        etiquetas = cfg.get("etiquetas", [])
        if isinstance(etiquetas, str):
            etiquetas = [e.strip().lower() for e in etiquetas.split(",") if e.strip()]
        else:
            etiquetas = [str(e).strip().lower() for e in etiquetas if str(e).strip()]
    
        fraccion = float(cfg.get("fraccion_presupuesto", 0.0))
        if fraccion < 0 or fraccion > 1:
            raise ValueError(f"Objetivo '{nombre}': fraccion_presupuesto fuera de [0,1].")
    
        duracion = int(cfg.get("duracion_meses", 0))
        if duracion <= 0:
            raise ValueError(f"Objetivo '{nombre}': duracion_meses debe ser > 0.")
    
        mes_inicio = cfg.get("mes_inicio", None)
        if mes_inicio in ("", None):
            raise ValueError(f"Objetivo '{nombre}': mes_inicio es obligatorio (formato 'YYYY-MM').")
    
        # Normaliza formato a YYYY-MM
        mes_inicio = str(mes_inicio).strip()[:7]
    
        objetivos_norm.append(
            {
                "nombre": nombre,
                "etiquetas": etiquetas,
                "fraccion_presupuesto": fraccion,
                "duracion_meses": duracion,
                "mes_inicio": mes_inicio,   # <-- SIEMPRE presente
            }
        )
    
    # Valida suma fracciones <= 1
    total_fr = sum(o["fraccion_presupuesto"] for o in objetivos_norm)
    if total_fr > 1 + 1e-9:
        raise ValueError("La suma de fraccion_presupuesto de los objetivos supera 1.")

    # Sustituye a partir de aquí
    objetivos_config = objetivos_norm
    # Tabla larga de objetivos (para UI/IO posterior)
    objetivos_rows = []

    # -------------------------
    # Iteración mensual
    # -------------------------
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

        # GASTOS NETOS (tu base)
        gasto_total_mes_actual_aux = float(gastos_netos_por_mes.get(mes_actual_str, pd.Series(dtype=float)).sum())
        gasto_total_mes_actual = gasto_total_mes_actual_aux - gasto_vacaciones - gasto_regalos - gasto_fondo_reserva

        # -------------------------
        # PRESUPUESTO BRUTO y OBJETIVOS (NUEVO)
        # -------------------------
        presupuesto_bruto = float(ingresos_reales_mensual.get(mes_anterior, 0.0)) * porcentaje_gasto

        # objetivos activos en este mes
        objetivos_activos = [cfg for cfg in objetivos_config if _objetivo_activo(cfg, mes_actual)]
        suma_fracciones = float(sum(cfg["fraccion_presupuesto"] for cfg in objetivos_activos)) if objetivos_activos else 0.0
        if suma_fracciones > 1.0 + 1e-9:
            raise ValueError(
                f"En {mes_actual_str} la suma de fraccion_presupuesto de objetivos activos supera 1.0: {suma_fracciones:.4f}"
            )

        # Aporte total reservado a objetivos desde presupuesto (antes de deuda)
        aporte_obj_total = presupuesto_bruto * suma_fracciones

        # Presupuesto corriente (vida normal)
        presupuesto_teorico = presupuesto_bruto  # mantenemos el nombre para no reventar tu output
        presupuesto_corriente = presupuesto_bruto - aporte_obj_total

        # Excluir del gasto corriente todos los gastos etiquetados de objetivos
        gastos_objetivos_mes = 0.0
        ingresos_objetivos_mes = 0.0
        for cfg in objetivos_activos:
            nombre = cfg["nombre"]
            gastos_objetivos_mes += float(objetivos_gastos_por_mes.get(nombre, pd.Series(dtype=float)).get(mes_actual_str, 0.0))
            ingresos_objetivos_mes += float(objetivos_ingresos_por_mes.get(nombre, pd.Series(dtype=float)).get(mes_actual_str, 0.0))

        gasto_total_mes_actual -= gastos_objetivos_mes  # clave: ya no cuenta como gasto corriente

        # Presupuesto disponible (tu lógica con deuda, pero usando presupuesto_corriente)
        presupuesto_disponible = max(0.0, presupuesto_corriente - deuda_acumulada)

        deuda_mensual = gasto_total_mes_actual - presupuesto_corriente
        exceso_gasto = deuda_mensual + deuda_acumulada
        deuda_acumulada = max(0.0, exceso_gasto)

        # -------------------------
        # INVERSIONES (igual)
        # -------------------------
        interes = float(intereses_ingreso.get(mes_actual_str, 0.0))
        if mes_actual >= pd.Timestamp("2024-10-01").to_period("M"):
            inv_mensual = float(ingresos_reales_mensual.get(mes_actual_str, 0.0)) * porcentaje_inversion
        else:
            inv_mensual = 0.0

        inversion_mes = inv_mensual + interes
        dinero_invertido_entre_mes = invertido_en_mes(presupuesto, mes_actual) - invertido_en_mes(presupuesto, mes_actual - 1)

        inversiones += inversion_mes - dinero_invertido_entre_mes
        dinero_invertido = invertido_en_mes(presupuesto, mes_actual)

        # -------------------------
        # AHORRO (tu lógica existente, sin tocar)
        # Nota: esta parte sigue igual; el objetivo nuevo NO se gestiona vía ahorro_transaccional,
        # se gestiona por presupuesto_corriente y ledger objetivo.
        # -------------------------
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

        # -------------------------
        # LEDGER OBJETIVOS (NUEVO) + LIQUIDACIÓN EN mes_fin
        # -------------------------
        resumen_objetivos_mes = {}

        for cfg in objetivos_activos:
            nombre = cfg["nombre"]
            fraccion = float(cfg["fraccion_presupuesto"])

            aporte_mes = presupuesto_bruto * fraccion
            ingresos_et = float(objetivos_ingresos_por_mes.get(nombre, pd.Series(dtype=float)).get(mes_actual_str, 0.0))
            gastos_et = float(objetivos_gastos_por_mes.get(nombre, pd.Series(dtype=float)).get(mes_actual_str, 0.0))

            saldo_ini = float(objetivos_saldos.get(nombre, 0.0))
            saldo_fin = saldo_ini + aporte_mes + ingresos_et - gastos_et
            objetivos_saldos[nombre] = saldo_fin
            resumen_objetivos_mes[nombre] = round(saldo_fin, 2)

            vence = _objetivo_vence(cfg, mes_actual)
            liquidacion = 0.0
            
            ahorro -= aporte_mes
            
            if vence:
                
                if saldo_fin > 0:
                    ahorro += saldo_fin
                    liquidacion = saldo_fin
                elif saldo_fin < 0:
                    deuda_acumulada += abs(saldo_fin)
                    ahorro -= abs(saldo_fin)
                    liquidacion = saldo_fin  # negativo
                objetivos_saldos[nombre] = 0.0  # cerrado
            
                

            # predicción informativa (media 10m antes del mes_inicio del objetivo)
            media10 = _media_ingresos_10m(ingresos_reales_mensual, cfg.get("mes_inicio"))
            pred_pres_bruto = media10 * porcentaje_gasto
            pred_aporte_mes = pred_pres_bruto * fraccion
            pred_total = pred_aporte_mes * int(cfg["duracion_meses"])
            
            mes_inicio_p = _to_period_m(cfg.get("mes_inicio"))
            
            mes_inicio_p = _to_period_m(cfg.get("mes_inicio"))
            if mes_inicio_p is None:
                raise ValueError(f"Objetivo '{cfg.get('nombre','(sin nombre)')}': mes_inicio es obligatorio (YYYY-MM).")
            
            mes_fin_p = _mes_fin_objetivo(mes_inicio_p, int(cfg.get("duracion_meses", 0)))
            if mes_inicio_p is None:
                raise ValueError(f"Objetivo '{cfg.get('nombre','(sin nombre)')}': mes_inicio es obligatorio (YYYY-MM).")
            objetivos_rows.append(
                {
                    "Mes": mes_actual_str,
                    "Objetivo": nombre,
                    "fraccion_presupuesto": fraccion,
                    "duracion_meses": int(cfg["duracion_meses"]),
                    "mes_inicio": mes_inicio_p.strftime("%Y-%m"),
                    "mes_fin": mes_fin_p.strftime("%Y-%m"),
                    "aporte_mes": round(aporte_mes, 2),
                    "ingresos_etiquetados_mes": round(ingresos_et, 2),
                    "gastos_etiquetados_mes": round(gastos_et, 2),
                    "saldo_fin_mes": round(saldo_fin, 2),
                    "vence_en_mes": bool(vence),
                    "liquidacion": round(liquidacion, 2),
                    "pred_media_ingresos_10m": round(media10, 2),
                    "pred_aporte_mes_medio": round(pred_aporte_mes, 2),
                    "pred_total_duracion": round(pred_total, 2),
                }
            )

        # -------------------------
        # FONDO RESERVA (igual, sin IO)
        # -------------------------
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

        # Total: OJO -> como ya no hay columnas dinámicas, el total incluye saldo vivo de objetivos todavía abiertos.
        total_objetivos = float(sum(objetivos_saldos.values())) if objetivos_saldos else 0.0
        
        total = regalos + vacaciones + inversiones + ahorro + fondo_cargado + total_objetivos + dinero_invertido
        
        
        
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
            # mantenemos este nombre para compatibilidad visual:
            "💸 Presupuesto Mes": round(presupuesto_corriente, 2),
            "🧾 Presupuesto Disponible": round(presupuesto_efectivo, 2),
            "📉 Deuda Presupuestaria mensual": round(deuda_mensual, 2) if deuda_mensual > 0 else 0.0,
            "📉 Deuda Presupuestaria acumulada": round(deuda_acumulada, 2) if deuda_acumulada > 0 else 0.0,
            # informativo (fijo)
            "Aporte objetivos desde presupuesto": round(aporte_obj_total, 2),
            "Presupuesto bruto": round(presupuesto_bruto, 2),
        }

        resumenes.append(resumen)
        mes_actual += 1

    df_resumen = pd.DataFrame(resumenes).copy(deep=True)

    # Si quieres mantener tu renombrado (aquí ya casi no afecta)
    # df_resumen = _renombrar_columnas_historial(df_resumen)

    objetivos_df = pd.DataFrame(objetivos_rows) if objetivos_rows else pd.DataFrame(
        columns=[
            "Mes","Objetivo","fraccion_presupuesto","duracion_meses","mes_inicio","mes_fin",
            "aporte_mes","ingresos_etiquetados_mes","gastos_etiquetados_mes","saldo_fin_mes",
            "vence_en_mes","liquidacion",
            "pred_media_ingresos_10m","pred_aporte_mes_medio","pred_total_duracion"
        ]
    )

    return df_resumen, objetivos_df



# =====================================================
# RESUMEN GLOBAL (mantiene prints fuera: aquí devolvemos datos)
# =====================================================

def resumen_global(presupuesto, fecha):
    fecha_objetivo = pd.to_datetime(fecha)
    saldos_a_fecha = presupuesto.balances_a_fecha(fecha_objetivo)
    saldo_total = presupuesto.balance_total_a_fecha(fecha_objetivo)
    return saldo_total, saldos_a_fecha
