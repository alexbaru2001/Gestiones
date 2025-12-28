#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 27 16:25:25 2025

@author: alex
"""

import json
import pandas as pd
from pathlib import Path
from typing import Union
from datetime import datetime, timedelta

from config import (
    REGISTROS_DIR,
    ARCHIVO_BASE_REGISTROS,
    HOJAS_REGISTRO,
    OBJETIVOS_VISTA_CONFIG_PATH,
)


# =====================================================
# OBJETIVOS VISTA (JSON)
# =====================================================

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
            except Exception as exc:
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
    objetivos_serializables = [_objetivo_a_serializable(obj) for obj in objetivos_normalizados]

    ruta = Path(config_path)
    if not ruta.is_absolute():
        ruta = Path(__file__).resolve().parent / ruta

    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(
        json.dumps({"objetivos": objetivos_serializables}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return objetivos_serializables


# =====================================================
# GUARDADOS GENÉRICOS (CSV)
# =====================================================

def guardar_historial(df: pd.DataFrame, output_path: Union[str, Path]) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


# =====================================================
# REGISTROS (EXCEL) - FUSIÓN
# =====================================================

def _limpiar_dataframe_registro(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza un DataFrame de registros para facilitar la fusión."""
    df = df.dropna(how="all")

    if "Fecha y hora" in df.columns:
        df["Fecha y hora"] = pd.to_datetime(df["Fecha y hora"], errors="coerce")
        df = df.dropna(subset=["Fecha y hora"])
        df = df.sort_values("Fecha y hora", ascending=False)

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

    # En tu código actual: mover a processed (no borrar)
    processed_dir = REGISTROS_DIR / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    for archivo in archivos_a_fusionar:
        destino = processed_dir / archivo.name
        if destino.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            destino = processed_dir / f"{archivo.stem}_{timestamp}{archivo.suffix}"
        archivo.rename(destino)


def fusionar_archivos_registro_seguro(confirmar: bool = False) -> None:
    if not confirmar:
        return
    fusionar_archivos_registro()


def fondo_reserva_general(carpeta_data='Data', lectura=True, historial=None, fondo_cargado_actual=None, escribir: bool = True):
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
    
    if escribir:
        df_fondo.to_csv(archivo_csv, index=False)

    registro_actual = df_fondo.iloc[-1]
    if lectura:
        print('Última actualización:', registro_actual['fecha de creacion'].strftime('%d-%m-%Y'))
        print(f"Cantidad del fondo requerida: {round(registro_actual['Cantidad del fondo'])}€")
    return registro_actual