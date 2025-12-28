#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 13:16:47 2025

@author: alex
"""

# pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union

import pandas as pd

from config import ARCHIVO_BASE_REGISTROS, saldos_iniciales
from domain import Account, Budget
from io_data import cargar_objetivos_vista
from logic import (
    eliminar_tildes,
    clasificar_ingreso,
    cargar_saldos_iniciales,
    cargar_transacciones,
    crear_historial_cuentas_virtuales,
)


@dataclass(frozen=True)
class PipelineParams:
    fecha_inicio: str = "2024-10-01"
    porcentaje_gasto: float = 0.3
    porcentaje_inversion: float = 0.1
    porcentaje_vacaciones: float = 0.05


def _leer_inicio_xlsx(excel_path: Union[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Lee Inicio.xlsx (Gastos/Ingresos/Transferencias) y devuelve 3 dataframes
    con columnas estándar:
      - gastos: ["fecha","categoria","cuenta","cantidad","etiquetas","comentario"]
      - ingresos: ["fecha","categoria","cuenta","cantidad","etiquetas","comentario"]
      - transferencias: ["fecha","saliente","entrante","cantidad","comentario"]
    """
    excel_path = Path(excel_path)

    # Gastos e ingresos (mismo patrón que usabas)
    gastos = pd.read_excel(excel_path, sheet_name=0).iloc[:, [0, 1, 2, 3] + [-2, -1]]
    ingresos = pd.read_excel(excel_path, sheet_name=1).iloc[:, [0, 1, 2, 3] + [-2, -1]]

    gastos.columns = ["fecha", "categoria", "cuenta", "cantidad", "etiquetas", "comentario"]
    ingresos.columns = ["fecha", "categoria", "cuenta", "cantidad", "etiquetas", "comentario"]

    # Transferencias: robusto por posición (tu notebook tenía drop de columnas de divisa)
    transferencias_raw = pd.read_excel(excel_path, sheet_name=2)
    transferencias = transferencias_raw.iloc[:, :5].copy()
    transferencias.columns = ["fecha", "saliente", "entrante", "cantidad", "comentario"]

    # Fechas
    gastos["fecha"] = pd.to_datetime(gastos["fecha"], errors="coerce")
    ingresos["fecha"] = pd.to_datetime(ingresos["fecha"], errors="coerce")
    transferencias["fecha"] = pd.to_datetime(transferencias["fecha"], errors="coerce")

    gastos = gastos.dropna(subset=["fecha"]).copy()
    ingresos = ingresos.dropna(subset=["fecha"]).copy()
    transferencias = transferencias.dropna(subset=["fecha"]).copy()

    # Normalización base (como hacías)
    gastos["categoria"] = gastos["categoria"].apply(eliminar_tildes)
    ingresos["categoria"] = ingresos["categoria"].apply(eliminar_tildes)

    # Asegurar strings en etiquetas/comentario (evita NaN raros)
    gastos["etiquetas"] = gastos["etiquetas"].fillna("").astype(str)
    ingresos["etiquetas"] = ingresos["etiquetas"].fillna("").astype(str)
    gastos["comentario"] = gastos["comentario"].fillna("").astype(str)
    ingresos["comentario"] = ingresos["comentario"].fillna("").astype(str)

    return gastos, ingresos, transferencias


def _preprocesar_tipo_logico(gastos: pd.DataFrame, ingresos: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    - ingresos['tipo_logico'] usando clasificar_ingreso
    - gastos['tipo_logico'] = categoria (y fix de regalos por 'mi regalo')
    """
    ingresos = ingresos.copy()
    ingresos["tipo_logico"] = ingresos.apply(clasificar_ingreso, axis=1)

    gastos = gastos.copy()
    # Importante: logic.py espera exactamente 'Regalos' en gastos para descontarlo.
    gastos["tipo_logico"] = gastos["categoria"]

    # Fix que ya comprobaste: marcar gastos de regalos por etiqueta o comentario
    etq = gastos["etiquetas"].fillna("").astype(str).str.lower()
    com = gastos["comentario"].fillna("").astype(str).str.lower()
    mask_regalos = etq.str.contains("mi regalo") | com.str.contains("mi regalo")
    gastos.loc[mask_regalos, "tipo_logico"] = "Regalos"

    return gastos, ingresos


def _construir_presupuesto(
    gastos: pd.DataFrame,
    ingresos: pd.DataFrame,
    transferencias: pd.DataFrame,
    saldos_iniciales_dict: dict,
) -> Budget:
    """
    Crea Budget + Account(s), carga saldos iniciales y vuelca transacciones.
    """
    presupuesto = Budget()

    cuentas = set()
    cuentas |= set(gastos["cuenta"].dropna().unique())
    cuentas |= set(ingresos["cuenta"].dropna().unique())
    cuentas |= set(transferencias["saliente"].dropna().unique())
    cuentas |= set(transferencias["entrante"].dropna().unique())

    cuentas_obj: Dict[str, Account] = {}
    for nombre in sorted(cuentas):
        acc = Account(nombre)
        presupuesto.add_account(acc)
        cuentas_obj[nombre] = acc

    # Saldos iniciales (tu dict real)
    cargar_saldos_iniciales(presupuesto, cuentas_obj, saldos_iniciales_dict)

    # Ingresos + gastos
    cargar_transacciones(ingresos, cuentas_obj, presupuesto, signo=+1)
    cargar_transacciones(gastos, cuentas_obj, presupuesto, signo=-1)

    # Transferencias: dos movimientos (sale y entra)
    for _, row in transferencias.iterrows():
        fecha = row["fecha"]
        saliente = row["saliente"]
        entrante = row["entrante"]
        cantidad = float(row["cantidad"])
        comentario = row.get("comentario", "")

        df_sal = pd.DataFrame([{
            "fecha": fecha,
            "categoria": "Transferencia",
            "cuenta": saliente,
            "cantidad": cantidad,
            "etiquetas": "",
            "comentario": f"Transf a {entrante} | {comentario}",
        }])
        cargar_transacciones(df_sal, cuentas_obj, presupuesto, signo=-1)

        df_ent = pd.DataFrame([{
            "fecha": fecha,
            "categoria": "Transferencia",
            "cuenta": entrante,
            "cantidad": cantidad,
            "etiquetas": "",
            "comentario": f"Transf desde {saliente} | {comentario}",
        }])
        cargar_transacciones(df_ent, cuentas_obj, presupuesto, signo=+1)

    return presupuesto


def run_pipeline(
    excel_path: Optional[Union[str, Path]] = None,
    params: Optional[PipelineParams] = None,
    objetivos_config: Optional[object] = None,
    output_path: Optional[Union[str, Path]] = None,
    saldos_iniciales_override: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Pipeline “UI-agnóstico”:
      - Lee excel (por defecto ARCHIVO_BASE_REGISTROS)
      - Preprocesa tipo_logico (incluye fix regalos)
      - Crea Budget
      - Carga objetivos
      - Crea historial de cuentas virtuales
      - (Opcional) guarda historial si output_path no es None

    Devuelve dict con objetos clave para la UI.
    """
    if params is None:
        params = PipelineParams()

    if excel_path is None:
        excel_path = ARCHIVO_BASE_REGISTROS

    saldos_iniciales_dict = saldos_iniciales_override if saldos_iniciales_override is not None else saldos_iniciales

    gastos, ingresos, transferencias = _leer_inicio_xlsx(excel_path)
    gastos, ingresos = _preprocesar_tipo_logico(gastos, ingresos)

    presupuesto = _construir_presupuesto(gastos, ingresos, transferencias, saldos_iniciales_dict)

    objetivos = cargar_objetivos_vista(objetivos_config)

    historial = crear_historial_cuentas_virtuales(
        df_ingresos=ingresos,
        df_gastos=gastos,
        presupuesto=presupuesto,
        fecha_inicio=params.fecha_inicio,
        porcentaje_gasto=params.porcentaje_gasto,
        porcentaje_inversion=params.porcentaje_inversion,
        porcentaje_vacaciones=params.porcentaje_vacaciones,
        saldos_iniciales=saldos_iniciales_dict,
        output_path=str(output_path) if output_path else None,
        objetivos_config=objetivos,
    )

    return {
        "excel_path": str(excel_path),
        "params": params,
        "gastos": gastos,
        "ingresos": ingresos,
        "transferencias": transferencias,
        "presupuesto": presupuesto,
        "objetivos": objetivos,
        "historial": historial,
    }
