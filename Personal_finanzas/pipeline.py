#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

# Ajusta estos imports a tu estructura real
from config import saldos_iniciales, ARCHIVO_BASE_REGISTROS  # si no existe ARCHIVO_BASE_REGISTROS, quítalo
from domain import Account, Budget, Transaction
from logic import (
    eliminar_tildes,
    procesar_ingresos,
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


def _leer_inicio_xlsx(excel: Union[str, Path, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Lee Inicio.xlsx (Gastos/Ingresos/Transferencias) desde:
      - ruta (str/Path) o
      - file-like (BytesIO) que viene de Streamlit

    Devuelve:
      gastos: ["fecha","categoria","cuenta","cantidad","etiquetas","comentario", ...]
      ingresos: ["fecha","categoria","cuenta","cantidad","etiquetas","comentario", ...]
      transferencias: ["fecha","saliente","entrante","cantidad","comentario"]
    """
    gastos = pd.read_excel(excel, sheet_name=0).iloc[:, [0, 1, 2, 3] + [-2, -1]]
    ingresos = pd.read_excel(excel, sheet_name=1).iloc[:, [0, 1, 2, 3] + [-2, -1]]

    gastos.columns = ["fecha", "categoria", "cuenta", "cantidad", "etiquetas", "comentario"]
    ingresos.columns = ["fecha", "categoria", "cuenta", "cantidad", "etiquetas", "comentario"]

    transferencias_raw = pd.read_excel(excel, sheet_name=2)
    transferencias = transferencias_raw.iloc[:, :5].copy()
    transferencias.columns = ["fecha", "saliente", "entrante", "cantidad", "comentario"]

    gastos["fecha"] = pd.to_datetime(gastos["fecha"], errors="coerce")
    ingresos["fecha"] = pd.to_datetime(ingresos["fecha"], errors="coerce")
    transferencias["fecha"] = pd.to_datetime(transferencias["fecha"], errors="coerce")

    gastos = gastos.dropna(subset=["fecha"]).copy()
    ingresos = ingresos.dropna(subset=["fecha"]).copy()
    transferencias = transferencias.dropna(subset=["fecha"]).copy()

    # Normalización (como hacías)
    gastos["categoria"] = gastos["categoria"].apply(eliminar_tildes)
    ingresos["categoria"] = ingresos["categoria"].apply(eliminar_tildes)

    gastos["etiquetas"] = gastos["etiquetas"].fillna("").astype(str)
    ingresos["etiquetas"] = ingresos["etiquetas"].fillna("").astype(str)
    gastos["comentario"] = gastos["comentario"].fillna("").astype(str)
    ingresos["comentario"] = ingresos["comentario"].fillna("").astype(str)

    return gastos, ingresos, transferencias


def _preprocesar_tipo_logico(gastos: pd.DataFrame, ingresos: pd.DataFrame):
    gastos = gastos.copy()
    ingresos = ingresos.copy()

    # --- ingresos: tu pipeline ya usa procesar_ingresos ---
    ingresos = procesar_ingresos(ingresos)

    # --- gastos: asegurar tipo_logico sin romper lo que venga del Excel ---
    if "tipo_logico" not in gastos.columns:
        gastos["tipo_logico"] = gastos["categoria"].astype(str)
    else:
        # si ya existe, solo rellenar NaNs/vacíos con categoria
        base = gastos["categoria"].astype(str)
        tl = gastos["tipo_logico"].astype(str)
        tl = tl.replace({"nan": ""})
        gastos["tipo_logico"] = tl.where(tl.str.strip() != "", base)

    etiquetas = gastos.get("etiquetas", "").fillna("").astype(str).str.lower()
    comentario = gastos.get("comentario", "").fillna("").astype(str).str.lower()
    categoria = gastos.get("categoria", "").fillna("").astype(str).str.lower()

    # --- FIX regalos en gastos (para que se resten) ---
    mask_regalos = (categoria == "regalos") | etiquetas.str.contains("mi regalo", regex=False) | comentario.str.contains("mi regalo", regex=False)
    gastos.loc[mask_regalos, "tipo_logico"] = "Regalos"

    return gastos, ingresos


def _crear_budget_desde_dfs(
    gastos: pd.DataFrame,
    ingresos: pd.DataFrame,
    transferencias: pd.DataFrame,
    saldos_iniciales_dict: Dict[str, float],
) -> Budget:
    """
    Construye Budget con cuentas, saldo inicial y transacciones.
    """
    presupuesto = Budget()

    # cuentas que aparecen en el excel + saldos iniciales
    cuentas = set()
    cuentas |= set(gastos["cuenta"].dropna().astype(str).unique())
    cuentas |= set(ingresos["cuenta"].dropna().astype(str).unique())
    if not transferencias.empty:
        cuentas |= set(transferencias["saliente"].dropna().astype(str).unique())
        cuentas |= set(transferencias["entrante"].dropna().astype(str).unique())
    cuentas |= set(saldos_iniciales_dict.keys())

    cuentas_obj = {}
    for nombre in sorted(cuentas):
        acc = Account(nombre)
        presupuesto.add_account(acc)
        cuentas_obj[nombre] = acc

    cargar_saldos_iniciales(presupuesto, cuentas_obj, saldos_iniciales_dict)
    cargar_transacciones(ingresos, cuentas_obj, presupuesto, signo=+1)
    cargar_transacciones(gastos, cuentas_obj, presupuesto, signo=-1)

    # Transferencias: doble apunte
    for _, row in transferencias.iterrows():
        fecha = row["fecha"]
        saliente = str(row["saliente"])
        entrante = str(row["entrante"])
        cantidad = float(row["cantidad"])
        comentario = str(row.get("comentario", ""))

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


def _normalizar_objetivos_presupuesto(objetivos_raw: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    objetivos_raw: lista de dicts con:
      - nombre
      - etiquetas
      - fraccion_presupuesto
      - duracion_meses
    """
    if not objetivos_raw:
        return []

    objetivos: List[Dict[str, Any]] = []
    for obj in objetivos_raw:
        nombre = str(obj.get("nombre", "")).strip()
        if not nombre:
            continue

        etiquetas = obj.get("etiquetas", [])
        if isinstance(etiquetas, str):
            etiquetas_list = [e.strip().lower() for e in etiquetas.split(",") if e.strip()]
        else:
            etiquetas_list = [str(e).strip().lower() for e in etiquetas if str(e).strip()]

        fr = float(obj.get("fraccion_presupuesto", 0.0))
        dm = int(obj.get("duracion_meses", 0))

        if fr < 0 or fr > 1:
            raise ValueError(f"Objetivo '{nombre}': fraccion_presupuesto fuera de [0,1].")
        if dm <= 0:
            raise ValueError(f"Objetivo '{nombre}': duracion_meses debe ser > 0.")

        mes_inicio = obj.get("mes_inicio", None)
        if mes_inicio in ("", None):
            mes_inicio_norm = None
        else:
            mes_inicio_norm = str(mes_inicio).strip()[:7]  # deja YYYY-MM
        
        objetivos.append(
            {
                "nombre": nombre,
                "etiquetas": etiquetas_list,
                "fraccion_presupuesto": fr,
                "duracion_meses": dm,
                "mes_inicio": mes_inicio_norm,   # <-- clave siempre presente
            }
        )

    total_fr = sum(o["fraccion_presupuesto"] for o in objetivos)
    if total_fr > 1 + 1e-9:
        raise ValueError("La suma de fraccion_presupuesto de los objetivos supera 1. Reduce fracciones.")

    # nombres únicos
    nombres = [o["nombre"] for o in objetivos]
    if len(set(nombres)) != len(nombres):
        raise ValueError("Los objetivos deben tener nombres únicos.")

    return objetivos


def run_pipeline(
    excel: Optional[Union[str, Path, Any]] = None,
    params: Optional[PipelineParams] = None,
    objetivos: Optional[List[Dict[str, Any]]] = None,
    output_path: Optional[Union[str, Path]] = None,
    fondo_reserva_snapshot: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Ejecuta el pipeline completo:
      - lee excel
      - prepara tipo_logico
      - crea Budget (incluye transferencias)
      - llama a crear_historial_cuentas_virtuales (lógica pura)
    """
    if params is None:
        params = PipelineParams()

    if excel is None:
        # Si no pasas excel, intenta la ruta por defecto si existe
        if "ARCHIVO_BASE_REGISTROS" in globals():
            excel = ARCHIVO_BASE_REGISTROS
        else:
            raise ValueError("No se ha proporcionado excel y no hay ruta por defecto configurada.")

    gastos, ingresos, transferencias = _leer_inicio_xlsx(excel)
    gastos, ingresos = _preprocesar_tipo_logico(gastos, ingresos)

    presupuesto = _crear_budget_desde_dfs(gastos, ingresos, transferencias, saldos_iniciales)

    objetivos_norm = _normalizar_objetivos_presupuesto(objetivos)

    historial = crear_historial_cuentas_virtuales(
        df_ingresos=ingresos,
        df_gastos=gastos,
        presupuesto=presupuesto,
        fecha_inicio=params.fecha_inicio,
        porcentaje_gasto=params.porcentaje_gasto,
        porcentaje_inversion=params.porcentaje_inversion,
        porcentaje_vacaciones=params.porcentaje_vacaciones,
        saldos_iniciales=saldos_iniciales,
        output_path=str(output_path) if output_path else None,  # se mantiene por compatibilidad, lógica no debería escribir
        objetivos_config=objetivos_norm,
        fondo_reserva_snapshot=fondo_reserva_snapshot,
    )

    return {
        "params": params,
        "gastos": gastos,
        "ingresos": ingresos,
        "transferencias": transferencias,
        "presupuesto": presupuesto,
        "objetivos": objetivos_norm,
        "historial": historial,
    }
