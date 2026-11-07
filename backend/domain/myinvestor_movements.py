"""Lectura del Excel de movimientos de MyInvestor.

El archivo trae la cabecera real en la fila 10 (las anteriores son titular, cuenta y saldo), y la
columna `Movimiento` mezcla cosas muy distintas: aportaciones del usuario, compras de fondo
—escritas como `NOMBRE DEL FONDO @ participaciones`—, la comisión de gestión trimestral con su IVA
e intereses del efectivo.

De aquí sale lo único que no está en ningún otro sitio: **las comisiones de gestión de MyInvestor**.
No aparecen en Finanzas porque se cobran contra el efectivo del propio bróker, así que sin leerlas
quedarían fuera del cálculo de rentabilidad.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable

HEADER_ROW = 9
COLUMNS = ("Fecha Operación", "Fecha Valor", "Movimiento", "Importe", "Saldo")

BROKER = "MyInvestor"

CONTRIBUTION = "aportacion"
FEE = "com. gestion"
FEE_VAT = "iva com. gestion"
CASH_INTEREST = "periodo"


def classify(movement: str) -> str:
    text = _normalize(movement)
    if text.startswith(_normalize(FEE_VAT)) or text.startswith(_normalize(FEE)):
        return "comision"
    if text.startswith(CONTRIBUTION):
        return "aportacion"
    if text.startswith(CASH_INTEREST):
        return "interes"
    if text.startswith("efectivo-eur"):
        return "efectivo"
    return "fondo"


def extract_fees(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Comisiones de gestión e IVA, con el mes en el que se cobraron.

    Se devuelven en positivo: en el extracto son cargos (importe negativo), pero aguas abajo se
    reparten como coste, y mezclar los dos signos en el mismo concepto invita a errores.
    """
    fees = []
    for row in rows or []:
        if classify(str(row.get("Movimiento", ""))) != "comision":
            continue
        month = _month(row.get("Fecha Operación"))
        amount = _number(row.get("Importe"))
        if not month or not amount:
            continue
        fees.append(
            {
                "Mes": month,
                "Broker": BROKER,
                "Importe": round(abs(amount), 2),
                "Concepto": str(row.get("Movimiento", "")).strip(),
            }
        )
    return fees


def extract_contributions(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Dinero que el usuario mete en la cuenta del bróker, mes a mes."""
    contributions: dict[str, float] = {}
    for row in rows or []:
        if classify(str(row.get("Movimiento", ""))) != "aportacion":
            continue
        month = _month(row.get("Fecha Operación"))
        if month:
            contributions[month] = round(contributions.get(month, 0.0) + _number(row.get("Importe")), 2)
    return [{"Mes": month, "Broker": BROKER, "Importe": amount} for month, amount in sorted(contributions.items())]


def extract_fund_operations(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compras de fondo, con las participaciones que vienen tras la arroba cuando el texto las trae.

    No todas las líneas la incluyen: el extracto a veces escribe el nombre largo del fondo y se
    queda sin sitio para el `@`. Esas se devuelven con `participaciones` a None en vez de
    descartarlas, porque el importe sí es bueno.
    """
    operations = []
    for row in rows or []:
        text = str(row.get("Movimiento", "")).strip()
        if classify(text) != "fondo":
            continue
        month = _month(row.get("Fecha Operación"))
        amount = _number(row.get("Importe"))
        if not month or not amount:
            continue
        match = re.search(r"@\s*([\d.,]+)\s*$", text)
        operations.append(
            {
                "Mes": month,
                "Broker": BROKER,
                "Nombre": re.sub(r"\s*@.*$", "", text).strip(),
                # El cargo sale de la cuenta, así que el importe invertido es su valor absoluto.
                "Importe": round(abs(amount), 2),
                "participaciones": _number(match.group(1)) if match else None,
            }
        )
    return operations


def _normalize(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    return "".join(char for char in text if not unicodedata.combining(char)).strip().lower()


def _month(value: Any) -> str:
    """El extracto escribe las fechas como dd/mm/aaaa."""
    text = str(value or "").strip()
    match = re.match(r"(\d{2})/(\d{2})/(\d{4})", text)
    if match:
        return f"{match.group(3)}-{match.group(2)}"
    # Si pandas ya la convirtió a fecha, llega como aaaa-mm-dd.
    match = re.match(r"(\d{4})-(\d{2})", text)
    return f"{match.group(1)}-{match.group(2)}" if match else ""


def _number(value: Any) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0
    try:
        return float(text.replace(".", "").replace(",", ".")) if text.count(",") == 1 else float(text)
    except ValueError:
        return 0.0
