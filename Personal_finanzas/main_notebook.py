#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 27 18:01:54 2025

@author: alex
"""

# main_notebook.py
from __future__ import annotations

import os
from pathlib import Path
import pandas as pd

from config import REGISTROS_DIR, ARCHIVO_BASE_REGISTROS, saldos_iniciales
from domain import Account, Budget
from io_data import (
    fusionar_archivos_registro_seguro,
    cargar_objetivos_vista,
    guardar_historial,
)
from logic import (
    eliminar_tildes,
    clasificar_ingreso,
    cargar_saldos_iniciales,
    cargar_transacciones,
    crear_historial_cuentas_virtuales,
    resumen_mensual,
)
from plots import (
    plot_balance_mensual,
    plot_explicacion_gastos,
    plot_porcentaje_ahorro,
)

# -----------------------------
# 1) Lectura de Inicio.xlsx
# -----------------------------
def cargar_datos_inicio_xlsx(excel_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Lee Inicio.xlsx (Gastos/Ingresos/Transferencias) y devuelve 3 dataframes
    con las columnas estándar que usabas:
      - gastos: ["fecha","categoria","cuenta","cantidad","etiquetas","comentario"]
      - ingresos: igual + luego tipo_logico
      - transferencias: ["fecha","saliente","entrante","cantidad","comentario"]
    """
    # Gastos e ingresos: mismo patrón que tu notebook
    gastos = pd.read_excel(excel_path, sheet_name=0).iloc[:, [0, 1, 2, 3] + [-2, -1]]
    ingresos = pd.read_excel(excel_path, sheet_name=1).iloc[:, [0, 1, 2, 3] + [-2, -1]]

    gastos.columns = ["fecha", "categoria", "cuenta", "cantidad", "etiquetas", "comentario"]
    ingresos.columns = ["fecha", "categoria", "cuenta", "cantidad", "etiquetas", "comentario"]

    # Transferencias: aquí NO invento columnas raras.
    # En tu notebook estabas haciendo un .drop(columns=...) porque el extract trae columnas de divisa.
    # Para hacerlo robusto: nos quedamos con las 5 primeras columnas útiles por posición y renombramos.
    transferencias_raw = pd.read_excel(excel_path, sheet_name=2)
    transferencias = transferencias_raw.iloc[:, :5].copy()
    transferencias.columns = ["fecha", "saliente", "entrante", "cantidad", "comentario"]

    # Fechas
    gastos["fecha"] = pd.to_datetime(gastos["fecha"], errors="coerce")
    ingresos["fecha"] = pd.to_datetime(ingresos["fecha"], errors="coerce")
    transferencias["fecha"] = pd.to_datetime(transferencias["fecha"], errors="coerce")

    gastos = gastos.dropna(subset=["fecha"])
    ingresos = ingresos.dropna(subset=["fecha"])
    transferencias = transferencias.dropna(subset=["fecha"])

    # Normalización de categorías (lo que ya hacías)
    gastos["categoria"] = gastos["categoria"].apply(eliminar_tildes)
    ingresos["categoria"] = ingresos["categoria"].apply(eliminar_tildes)

    return gastos, ingresos, transferencias


# -----------------------------
# 2) Construir Budget
# -----------------------------
def construir_presupuesto(gastos: pd.DataFrame, ingresos: pd.DataFrame, transferencias: pd.DataFrame) -> Budget:
    """
    Crea Budget + Account(s) a partir de las cuentas que aparecen en los excels,
    carga saldos iniciales y vuelca todas las transacciones.
    """
    presupuesto = Budget()

    # Cuentas detectadas en datos
    cuentas = set()
    if "cuenta" in gastos.columns:
        cuentas |= set(gastos["cuenta"].dropna().unique())
    if "cuenta" in ingresos.columns:
        cuentas |= set(ingresos["cuenta"].dropna().unique())
    if "saliente" in transferencias.columns:
        cuentas |= set(transferencias["saliente"].dropna().unique())
    if "entrante" in transferencias.columns:
        cuentas |= set(transferencias["entrante"].dropna().unique())

    # Crear objetos Account
    cuentas_obj = {}
    for nombre in sorted(cuentas):
        acc = Account(nombre)
        presupuesto.add_account(acc)
        cuentas_obj[nombre] = acc

    # Cargar saldos iniciales (tu dict 'saldos_iniciales' de config)
    cargar_saldos_iniciales(presupuesto, cuentas_obj, saldos_iniciales)

    # Cargar ingresos / gastos en presupuesto
    # NOTA: tu cargar_transacciones usa 'cantidad' y asigna signo.
    cargar_transacciones(ingresos, cuentas_obj, presupuesto, signo=+1)
    cargar_transacciones(gastos, cuentas_obj, presupuesto, signo=-1)

    # Transferencias: convierto a dos transacciones (sale de una cuenta y entra en otra)
    # Esto respeta tu modelo (Transaction) sin inventar una estructura nueva.
    for _, row in transferencias.iterrows():
        fecha = row["fecha"]
        saliente = row["saliente"]
        entrante = row["entrante"]
        cantidad = row["cantidad"]
        comentario = row.get("comentario", "")

        # Sale
        df_sal = pd.DataFrame([{
            "fecha": fecha,
            "categoria": "Transferencia",
            "cuenta": saliente,
            "cantidad": float(cantidad),
            "etiquetas": "",
            "comentario": f"Transf a {entrante} | {comentario}",
        }])
        cargar_transacciones(df_sal, cuentas_obj, presupuesto, signo=-1)

        # Entra
        df_ent = pd.DataFrame([{
            "fecha": fecha,
            "categoria": "Transferencia",
            "cuenta": entrante,
            "cantidad": float(cantidad),
            "etiquetas": "",
            "comentario": f"Transf desde {saliente} | {comentario}",
        }])
        cargar_transacciones(df_ent, cuentas_obj, presupuesto, signo=+1)

    return presupuesto


# -----------------------------
# 3) MAIN de notebook
# -----------------------------
def main():
    # (Opcional) fusionar registros descargados si quieres
    # Por seguridad lo dejo apagado por defecto (como ya hiciste con *_seguro)
    fusionar_archivos_registro_seguro(confirmar=False)

    excel_path = ARCHIVO_BASE_REGISTROS
    gastos, ingresos, transferencias = cargar_datos_inicio_xlsx(excel_path)

    # Tu “tipo_logico” de ingresos lo calculabas en notebook
    ingresos = ingresos.copy()
    ingresos["tipo_logico"] = ingresos.apply(clasificar_ingreso, axis=1)
    
    # Gastos: tipo_logico = categoria (como en el notebook)    
    gastos = gastos.copy()
    gastos["tipo_logico"] = gastos["categoria"]  # base
    
    etq = gastos["etiquetas"].fillna("").astype(str).str.lower()
    com = gastos["comentario"].fillna("").astype(str).str.lower()
    
    # si detectas regalos por etiqueta o comentario, márcalo como Regalos
    mask_regalos = etq.str.contains("mi regalo") | com.str.contains("mi regalo")
    gastos.loc[mask_regalos, "tipo_logico"] = "Regalos"

    
    presupuesto = construir_presupuesto(gastos, ingresos, transferencias)
    
    print("Gastos marcados como Regalos:", (gastos["tipo_logico"] == "Regalos").sum())
    print(gastos.loc[gastos["tipo_logico"] == "Regalos", ["fecha","categoria","cantidad","etiquetas","comentario"]].head(20))

    # Objetivos vista (lee Data/objetivos_vista.json)
    objetivos = cargar_objetivos_vista()
    print(saldos_iniciales)
    # Historial (si quieres guardarlo, pásale output_path)
    historial = crear_historial_cuentas_virtuales(
        df_ingresos=ingresos,
        df_gastos=gastos,
        presupuesto=presupuesto,
        fecha_inicio="2024-10-01",
        porcentaje_gasto=0.3,
        porcentaje_inversion=0.1,
        porcentaje_vacaciones=0.05,
        saldos_iniciales=saldos_iniciales,
        output_path=None,
        objetivos_config=objetivos,
    )
    cols = [
    "Mes", "Regalos", "Vacaciones", "Inversiones", "Dinero Invertido", "Ahorros",
    "Fondo de reserva cargado", "total", "Gasto del mes", "Presupuesto Mes",
    "Presupuesto Disponible", "Deuda Presupuestaria mensual", "Deuda Presupuestaria acumulada"
]
    print(historial[cols].head(15).to_string(index=False))

    # Si quieres exportar desde aquí:
    out_csv = Path("Data") / "historial.csv"
    guardar_historial(historial, out_csv)

    # -----------------------------
    # 4) Gráficos para notebook
    # -----------------------------
    # Balance mensual (usa resumen_mensual)
    resumen = resumen_mensual(gastos, ingresos)
    fig_balance = plot_balance_mensual(resumen)
    fig_balance.show()

    # Explicación de gastos: devuelve pivot + stats + figs (sin prints)
    pivot, stats, figs = plot_explicacion_gastos(
        df_gastos=gastos,
        df_ingresos=ingresos,
        categorias_fijas=["alimentacion", "entretenimiento", "hosteleria", "otros", "transporte"],
    )
    # muestras los gráficos que quieras
    # figs["barras_apiladas"].show()
    # figs["porcentajes"].show()
    # figs["lineas"].show()
    # figs["boxplot"].show()
    # figs["heatmap"].show()

    # Porcentaje de ahorro
    df_ahorro, fig_ahorro = plot_porcentaje_ahorro(gastos, ingresos)
    # fig_ahorro.show()

    return {
        "gastos": gastos,
        "ingresos": ingresos,
        "transferencias": transferencias,
        "presupuesto": presupuesto,
        "historial": historial,
        "resumen": resumen,
        "pivot_explicacion": pivot,
        "stats_explicacion": stats,
        "df_ahorro": df_ahorro,
    }


if __name__ == "__main__":
    objetos = main()
