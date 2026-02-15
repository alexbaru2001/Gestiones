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

def _normalizar_lista(valor):
    if valor is None:
        return []
    if isinstance(valor, (list, tuple, set)):
        return [str(v).strip() for v in valor if str(v).strip()]
    return [str(valor).strip()]

def _normalizar_objetivos(datos):
    if isinstance(datos, dict):
        datos = datos.get("objetivos", [])
    objetivos = []
    for objetivo in datos:
        nombre = str(objetivo.get("nombre", "")).strip()
        if not nombre:
            continue

        etiquetas = [e.lower() for e in _normalizar_lista(objetivo.get("etiquetas"))]
        porcentaje = float(objetivo.get("porcentaje_ingreso", 0.0))
        if porcentaje < 0 or porcentaje > 1:
            raise ValueError(
                f"El objetivo '{nombre}' tiene un porcentaje fuera del rango [0, 1]."
            )

        saldo_inicial = float(objetivo.get("saldo_inicial", 0.0))
        objetivo_total = objetivo.get("objetivo_total")
        horizonte_meses = objetivo.get("horizonte_meses")

        mes_inicio_raw = objetivo.get("mes_inicio")
        mes_inicio = None
        if mes_inicio_raw not in (None, ""):
            try:
                if isinstance(mes_inicio_raw, pd.Period):
                    mes_inicio = mes_inicio_raw
                elif isinstance(mes_inicio_raw, pd.Timestamp):
                    mes_inicio = mes_inicio_raw.to_period("M")
                else:
                    mes_inicio = pd.Period(str(mes_inicio_raw), freq="M")
            except Exception as exc:  # noqa: BLE001 (queremos informar del valor inválido)
                raise ValueError(
                    f"El objetivo '{nombre}' tiene un mes_inicio inválido: {mes_inicio_raw}"
                ) from exc

        objetivos.append(
            {
                "nombre": nombre,
                "etiquetas": etiquetas,
                "porcentaje_ingreso": porcentaje,
                "saldo_inicial": saldo_inicial,
                "objetivo_total": float(objetivo_total) if objetivo_total is not None else None,
                "horizonte_meses": int(horizonte_meses) if horizonte_meses else None,
                "mes_inicio": mes_inicio,
            }
        )

    return objetivos

def cargar_objetivos_vista(config_path: Union[Path, str] = OBJETIVOS_VISTA_CONFIG_PATH):
    """Lee la configuración de objetivos vista desde disco."""

    if config_path is None:
        return []

    if isinstance(config_path, (list, tuple, set, dict)):
        return _normalizar_objetivos(config_path)

    ruta = Path(config_path)
    if not ruta.is_absolute():
        ruta = Path(__file__).resolve().parent / ruta

    if not ruta.exists():
        plantilla = {
            "objetivos": [
                {
                    "nombre": "Coche",
                    "etiquetas": ["coche"],
                    "porcentaje_ingreso": 0.0,
                    "saldo_inicial": 0.0,
                    "objetivo_total": 12000.0,
                    "horizonte_meses": 24,
                    "mes_inicio": str(pd.Timestamp.today().to_period("M")),
                }
            ]
        }
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(json.dumps(plantilla, indent=2, ensure_ascii=False), encoding="utf-8")
        datos = plantilla["objetivos"]
    else:
        try:
            contenido = ruta.read_text(encoding="utf-8")
            datos = json.loads(contenido)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"El archivo de objetivos vista {ruta} no contiene JSON válido: {exc}"
            ) from exc

    return _normalizar_objetivos(datos)

def _objetivo_a_serializable(objetivo: dict) -> dict:
    """Convierte un objetivo normalizado en un diccionario apto para JSON."""

    mes_inicio = objetivo.get("mes_inicio")
    if isinstance(mes_inicio, pd.Period):
        mes_inicio = mes_inicio.strftime("%Y-%m")
    elif isinstance(mes_inicio, pd.Timestamp):
        mes_inicio = mes_inicio.to_period("M").strftime("%Y-%m")
    elif mes_inicio in ("", None):
        mes_inicio = None
    else:
        mes_inicio = str(mes_inicio)

    return {
        "nombre": objetivo.get("nombre", ""),
        "etiquetas": list(objetivo.get("etiquetas", [])),
        "porcentaje_ingreso": float(objetivo.get("porcentaje_ingreso", 0.0)),
        "saldo_inicial": float(objetivo.get("saldo_inicial", 0.0)),
        "objetivo_total": (
            float(objetivo.get("objetivo_total"))
            if objetivo.get("objetivo_total") is not None
            else None
        ),
        "horizonte_meses": (
            int(objetivo.get("horizonte_meses"))
            if objetivo.get("horizonte_meses") not in (None, "")
            else None
        ),
        "mes_inicio": mes_inicio,
    }

def guardar_objetivos_vista(
    objetivos: Union[list[dict], dict],
    config_path: Union[Path, str] = OBJETIVOS_VISTA_CONFIG_PATH,
) -> list[dict]:
    """Sobrescribe la configuración de objetivos vista tras validar su contenido."""

    if config_path is None:
        raise ValueError("Se requiere una ruta válida para guardar los objetivos vista.")

    objetivos_normalizados = _normalizar_objetivos(objetivos)
    objetivos_serializables = [
        _objetivo_a_serializable(obj) for obj in objetivos_normalizados
    ]

    ruta = Path(config_path)
    if not ruta.is_absolute():
        ruta = Path(__file__).resolve().parent / ruta

    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(
        json.dumps({"objetivos": objetivos_serializables}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return objetivos_serializables
