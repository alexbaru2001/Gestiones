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

def crear_widget_gestor_objetivos(
    config_path: Union[Path, str] = OBJETIVOS_VISTA_CONFIG_PATH,
):
    """Crea un widget interactivo para gestionar objetivos vista desde un notebook."""

    try:
        import ipywidgets as widgets
        from traitlets import TraitError
    except ImportError as exc:  # pragma: no cover - dependemos del entorno del cuaderno
        raise ImportError(
            "ipywidgets es necesario para usar el gestor interactivo de objetivos vista."
        ) from exc

    objetivos_config = [
        _objetivo_a_serializable(obj) for obj in cargar_objetivos_vista(config_path)
    ]

    ruta = Path(config_path)
    if not ruta.is_absolute():
        ruta = Path(__file__).resolve().parent / ruta

    status = widgets.HTML()
    porcentaje_total_label = widgets.HTML()

    lista_objetivos = widgets.Select(
        options=[obj["nombre"] for obj in objetivos_config],
        description="Objetivos",
        rows=min(10, max(4, len(objetivos_config) or 4)),
        layout=widgets.Layout(width="30%"),
    )

    nombre_input = widgets.Text(description="Nombre", placeholder="Nuevo objetivo")
    etiquetas_input = widgets.Text(
        description="Etiquetas",
        placeholder="Separadas por comas",
    )
    porcentaje_input = widgets.BoundedFloatText(
        description="% ingreso",
        min=0.0,
        max=1.0,
        step=0.01,
        value=0.0,
    )
    saldo_inicial_input = widgets.FloatText(description="Saldo inicial", value=0.0)
    objetivo_total_input = widgets.Text(
        description="Meta",
        placeholder="Importe objetivo (opcional)",
    )
    horizonte_input = widgets.Text(
        description="Horizonte",
        placeholder="Meses objetivo (opcional)",
    )
    mes_inicio_input = widgets.Text(
        description="Mes inicio",
        placeholder="AAAA-MM (opcional)",
    )

    guardar_memoria_btn = widgets.Button(
        description="Guardar cambios",
        button_style="success",
        icon="save",
    )
    guardar_archivo_btn = widgets.Button(
        description="Guardar en disco",
        button_style="info",
        icon="cloud-upload",
    )
    nuevo_btn = widgets.Button(description="Nuevo", icon="plus", button_style="primary")
    eliminar_btn = widgets.Button(description="Eliminar", icon="trash", button_style="danger")

    selected_name = {"valor": None}

    def actualizar_porcentaje_total():
        total = sum(obj.get("porcentaje_ingreso", 0.0) for obj in objetivos_config)
        total_real = total + 0.45
        color = "green" if total_real <= 1.0 else "red"
        porcentaje_total_label.value = (
            f"<b>Porcentaje acumulado:</b> <span style='color:{color}'>"
            f"{total_real:.1%}</span>"
        )

    def mostrar_mensaje(texto: str, exito: bool = True):
        color = "green" if exito else "red"
        status.value = f"<span style='color:{color}'>{texto}</span>"

    def rellenar_formulario(nombre: str | None):
        if not nombre:
            nombre_input.value = ""
            etiquetas_input.value = ""
            porcentaje_input.value = 0.0
            saldo_inicial_input.value = 0.0
            objetivo_total_input.value = ""
            horizonte_input.value = ""
            mes_inicio_input.value = ""
            selected_name["valor"] = None
            return

        for obj in objetivos_config:
            if obj["nombre"] == nombre:
                nombre_input.value = obj["nombre"]
                etiquetas_input.value = ", ".join(obj.get("etiquetas", []))
                porcentaje_input.value = obj.get("porcentaje_ingreso", 0.0)
                saldo_inicial_input.value = obj.get("saldo_inicial", 0.0)
                objetivo_total_input.value = (
                    "" if obj.get("objetivo_total") is None else str(obj["objetivo_total"])
                )
                horizonte_input.value = (
                    "" if obj.get("horizonte_meses") is None else str(obj["horizonte_meses"])
                )
                mes_inicio_input.value = obj.get("mes_inicio") or ""
                selected_name["valor"] = nombre
                return

        mostrar_mensaje("No se encontró el objetivo seleccionado.", exito=False)

    def refrescar_lista(nombre_a_seleccionar: str | None = None):
        opciones = [obj["nombre"] for obj in objetivos_config]
        lista_objetivos.options = opciones
        try:
            if nombre_a_seleccionar and nombre_a_seleccionar in opciones:
                lista_objetivos.value = nombre_a_seleccionar
            elif opciones:
                lista_objetivos.value = opciones[0]
            else:
                lista_objetivos.value = None
        except TraitError:
            lista_objetivos.value = None
        rellenar_formulario(lista_objetivos.value)
        actualizar_porcentaje_total()

    def obtener_datos_formulario() -> dict:
        etiquetas = [e.strip().lower() for e in etiquetas_input.value.split(",") if e.strip()]
        objetivo_total_txt = objetivo_total_input.value.strip()
        horizonte_txt = horizonte_input.value.strip()
        mes_inicio_txt = mes_inicio_input.value.strip()
        return {
            "nombre": nombre_input.value.strip(),
            "etiquetas": etiquetas,
            "porcentaje_ingreso": porcentaje_input.value or 0.0,
            "saldo_inicial": saldo_inicial_input.value or 0.0,
            "objetivo_total": float(objetivo_total_txt) if objetivo_total_txt else None,
            "horizonte_meses": int(horizonte_txt) if horizonte_txt else None,
            "mes_inicio": mes_inicio_txt or None,
        }

    def on_guardar_memoria(_):
        datos = obtener_datos_formulario()
        if not datos["nombre"]:
            mostrar_mensaje("El objetivo necesita un nombre.", exito=False)
            return

        try:
            objetivo_normalizado = _normalizar_objetivos([datos])[0]
        except ValueError as exc:
            mostrar_mensaje(str(exc), exito=False)
            return

        objetivo_serial = _objetivo_a_serializable(objetivo_normalizado)

        nombres_existentes = {obj["nombre"] for obj in objetivos_config}
        nombre_actual = selected_name["valor"]

        if objetivo_serial["nombre"] != nombre_actual and objetivo_serial["nombre"] in nombres_existentes:
            mostrar_mensaje(
                f"Ya existe un objetivo llamado '{objetivo_serial['nombre']}'.",
                exito=False,
            )
            return

        total_propuesto = objetivo_serial["porcentaje_ingreso"]
        for obj in objetivos_config:
            if obj["nombre"] == nombre_actual:
                continue
            total_propuesto += obj.get("porcentaje_ingreso", 0.0)

        if total_propuesto > 1.0 + 1e-6:
            mostrar_mensaje(
                "La suma de porcentajes de ingreso supera el 100%. Ajusta los valores.",
                exito=False,
            )
            return

        actualizado = False
        for idx, obj in enumerate(objetivos_config):
            if obj["nombre"] == nombre_actual:
                objetivos_config[idx] = objetivo_serial
                actualizado = True
                break

        if not actualizado:
            objetivos_config.append(objetivo_serial)

        mostrar_mensaje("Objetivo actualizado en la sesión.")
        refrescar_lista(objetivo_serial["nombre"])

    def on_guardar_archivo(_):
        try:
            guardados = guardar_objetivos_vista(
                objetivos_config,
                config_path=ruta,
            )
        except Exception as exc:  # noqa: BLE001 - queremos mostrar el error al usuario
            mostrar_mensaje(f"Error al guardar: {exc}", exito=False)
            return

        objetivos_config[:] = guardados
        mostrar_mensaje(f"Objetivos guardados en {ruta}.")
        refrescar_lista(lista_objetivos.value)

    def on_nuevo(_):
        lista_objetivos.value = None
        rellenar_formulario(None)
        mostrar_mensaje("Introduce los datos del nuevo objetivo y pulsa 'Guardar cambios'.", exito=False)

    def on_eliminar(_):
        nombre = lista_objetivos.value
        if not nombre:
            mostrar_mensaje("Selecciona un objetivo para eliminarlo.", exito=False)
            return

        objetivos_config[:] = [obj for obj in objetivos_config if obj["nombre"] != nombre]
        mostrar_mensaje(f"Objetivo '{nombre}' eliminado de la sesión.")
        refrescar_lista(None)

    def on_seleccion(change):
        if change.get("name") == "value":
            rellenar_formulario(change.get("new"))

    lista_objetivos.observe(on_seleccion, names="value")
    guardar_memoria_btn.on_click(on_guardar_memoria)
    guardar_archivo_btn.on_click(on_guardar_archivo)
    nuevo_btn.on_click(on_nuevo)
    eliminar_btn.on_click(on_eliminar)

    formulario = widgets.VBox(
        [
            nombre_input,
            etiquetas_input,
            porcentaje_input,
            saldo_inicial_input,
            objetivo_total_input,
            horizonte_input,
            mes_inicio_input,
            widgets.HBox([guardar_memoria_btn, guardar_archivo_btn]),
        ],
        layout=widgets.Layout(width="65%"),
    )

    acciones_lista = widgets.VBox([lista_objetivos, widgets.HBox([nuevo_btn, eliminar_btn])])

    contenedor = widgets.VBox(
        [
            widgets.HBox([acciones_lista, formulario]),
            porcentaje_total_label,
            status,
        ]
    )

    refrescar_lista(lista_objetivos.options[0] if lista_objetivos.options else None)

    return contenedor
