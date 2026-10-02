"""Desglose mensual de la bolsa de inversión: de dónde entra el dinero y cuánto sale invertido.

La bolsa (`📈 Inversiones`) se alimenta de cuatro sitios, según `logic.py` (líneas 672-700):

- el porcentaje configurado del sueldo del mes (los ingresos de tipo "Ingreso Real"),
- el rendimiento financiero, del que los **dividendos** se separan por su etiqueta porque el
  usuario los sigue aparte —son el mismo tipo lógico, así que mezclarlos los contaría dos veces—,
- y un tercio del presupuesto que sobra al cerrar el mes, el "extra por ahorrar". Ese sobrante solo
  existe cuando el presupuesto del mes cubre el gasto **después** de descontar la deuda arrastrada,
  así que un mes con extra cierra por fuerza con deuda acumulada cero: son excluyentes.

De ella sale el dinero que se invierte de verdad, que es la subida del acumulado "Dinero Invertido".
Se reconstruye aquí a partir del resumen mensual y de las filas de ingresos, en vez de tocar el
pipeline heredado: así cubre todo el histórico y no cambia el formato de `historial.csv`.

La comprobación que valida el cálculo es que, para cada mes,
`sueldo + intereses + dividendos + extra - invertido` sea igual a lo que sube o baja la bolsa.
"""

from __future__ import annotations

from typing import Any, Iterable

SALARY_KIND = "ingreso real"
FINANCIAL_KIND = "rendimiento financiero"
DIVIDEND_TAG = "dividendos"

# Un tercio del presupuesto sobrante va a la bolsa (los otros dos tercios, a vacaciones y ahorro).
LEFTOVER_SHARE = 1 / 3

CONTRIBUTION_KEYS = ("sueldo", "intereses", "dividendos", "extra")


def build_investment_contributions(
    summary_rows: list[dict[str, Any]],
    income_rows: Iterable[dict[str, Any]],
    porcentaje_inversion: float = 0.1,
) -> list[dict[str, Any]]:
    if not summary_rows:
        return []

    salary: dict[str, float] = {}
    interest: dict[str, float] = {}
    dividends: dict[str, float] = {}

    for row in income_rows or []:
        month = _month_of(row)
        if not month:
            continue
        amount = _as_float(row.get("cantidad"))
        kind = _normalize(row.get("tipo_logico"))
        if kind == SALARY_KIND:
            salary[month] = salary.get(month, 0.0) + amount
        elif kind == FINANCIAL_KIND:
            bucket = dividends if _is_dividend(row.get("etiquetas")) else interest
            bucket[month] = bucket.get(month, 0.0) + amount

    rows: list[dict[str, Any]] = []
    previous_invested: float | None = None
    for record in summary_rows:
        month = str(record.get("Mes") or "")
        invested_total = _as_float(record.get("Dinero Invertido"))
        # Lo invertido en el mes es cuánto sube el acumulado. El primer mes del histórico no tiene
        # anterior con el que compararse, así que vale lo que marque el acumulado.
        invested = invested_total if previous_invested is None else invested_total - previous_invested
        previous_invested = invested_total

        # Ojo con el nombre de la columna: "🧾 Presupuesto Disponible" NO guarda el presupuesto
        # disponible, sino `presupuesto_efectivo` (logic.py:814), que ya es lo que sobra DESPUÉS de
        # restar el gasto del mes. Volver a restarlo aquí hundía el extra a cero casi siempre.
        leftover = max(0.0, _as_float(record.get("🧾 Presupuesto Disponible")))

        rows.append(
            {
                "Mes": month,
                "sueldo": round(salary.get(month, 0.0) * float(porcentaje_inversion or 0.0), 2),
                "intereses": round(interest.get(month, 0.0), 2),
                "dividendos": round(dividends.get(month, 0.0), 2),
                "extra": round(leftover * LEFTOVER_SHARE, 2),
                "invertido": round(invested, 2),
            }
        )
    return rows


def merge_income_rows(
    historical_rows: Iterable[dict[str, Any]] | None, current_rows: Iterable[dict[str, Any]] | None
) -> list[dict[str, Any]]:
    """Une el detalle guardado con el del tramo recién calculado sin duplicar.

    Un mes puede estar en los dos sitios (al reprocesar un Excel que ya incluía meses guardados).
    Concatenarlos sin más contaba cada ingreso dos veces y dejaba el desglose al doble, así que
    para un mes presente en ambos manda la versión recién calculada.
    """
    current = list(current_rows or [])
    current_months = {_month_of(row) for row in current}
    historical_only = [row for row in (historical_rows or []) if _month_of(row) not in current_months]
    return historical_only + current


def _month_of(row: dict[str, Any]) -> str:
    month = str(row.get("Mes") or "").strip()
    if month:
        return month[:7]
    date = str(row.get("fecha") or "").strip()
    return date[:7] if len(date) >= 7 else ""


def _is_dividend(tags: Any) -> bool:
    return any(_normalize(tag) == DIVIDEND_TAG for tag in str(tags or "").split(","))


def _normalize(value: Any) -> str:
    import unicodedata

    text = unicodedata.normalize("NFKD", str(value or ""))
    return "".join(char for char in text if not unicodedata.combining(char)).strip().lower()


def _as_float(value: Any) -> float:
    try:
        number = float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if number != number else number  # descarta NaN
