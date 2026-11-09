"""Cartera a partir del archivo de posiciones que mantiene el usuario.

Los brókeres solo entregan **fotos a cierre de mes**, no movimientos. Así que aquí se hace el
camino inverso: comparando la foto de un mes con la del siguiente se deduce qué pasó —si una
posición sube de cantidad hubo una compra, si baja una venta, y si desaparece se vendió entera.

Dos reglas que no son obvias y que hay que respetar:

- La regla de "lo que no aparece se ha vendido" se aplica **por bróker y solo a los brókeres que
  vengan en ese mes**. Un mes que solo traiga el extracto de Trade Republic no puede dar por
  vendida la cartera de DeGiro.
- Una subida de cantidad no siempre es una compra: un *scrip dividend* (los derechos de Iberdrola)
  entrega acciones nuevas sin poner dinero. Aquí se marca el movimiento y se deja que quien lo
  consuma decida; el criterio del usuario es tratarlo como entrada sin coste.
"""

from __future__ import annotations

from typing import Any, Iterable

COLUMNS = (
    "Mes",
    "Broker",
    "ISIN",
    "Nombre",
    "Cantidad",
    "Precio cierre",
    "Moneda",
    "Valor EUR",
    "Aportado EUR",
)

# Clave de una posición dentro de una foto mensual.
POSITION_KEY = ("Mes", "Broker", "ISIN")

# El saldo corriente distingue dos movimientos idénticos del mismo día, que es justo donde una
# clave sin él colapsaría dos operaciones reales en una.
MOVEMENT_KEY = ("Fecha Operación", "Movimiento", "Importe", "Saldo")

# Por debajo de esto una diferencia de cantidad es ruido de redondeo del extracto, no una operación.
QUANTITY_EPSILON = 1e-6


def read_positions(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normaliza las filas del archivo y descarta las que no identifican una posición."""
    positions = []
    for row in rows or []:
        month = _text(row.get("Mes"))[:7]
        broker = _text(row.get("Broker"))
        isin = _text(row.get("ISIN")).upper()
        if not month or not broker or not isin:
            continue
        positions.append(
            {
                "Mes": month,
                "Broker": broker,
                "ISIN": isin,
                "Nombre": _text(row.get("Nombre")) or isin,
                "Cantidad": _number(row.get("Cantidad")),
                "Precio cierre": _number(row.get("Precio cierre")),
                "Moneda": _text(row.get("Moneda")).upper() or "EUR",
                "Valor EUR": _optional_number(row.get("Valor EUR")),
                "Aportado EUR": _optional_number(row.get("Aportado EUR")),
            }
        )
    positions.sort(key=lambda item: (item["Mes"], item["Broker"], item["Nombre"]))
    return positions


def position_value(position: dict[str, Any]) -> float | None:
    """Valor en euros de una posición.

    Manda el importe que da el bróker: los extractos redondean el precio por título y recalcular
    acumula desviación (Iberdrola en agosto de 2026, 26,994544 × 19,81 = 534,76 € frente a los
    534,63 € del extracto). Solo cuando no viene se deriva, y únicamente si está en euros, porque
    convertir divisa exige un tipo de cambio que no está en el archivo.
    """
    value = position.get("Valor EUR")
    if value is not None:
        return float(value)
    price = position.get("Precio cierre")
    # Sin precio no se puede derivar: devolver cero lo haría pasar por una posición que no vale
    # nada, cuando lo que ocurre es que hay que buscar su cotización fuera.
    if price is None or _number(price) == 0:
        return None
    if _text(position.get("Moneda")).upper() != "EUR":
        return None
    return round(_number(position.get("Cantidad")) * _number(price), 2)


def months_of(positions: Iterable[dict[str, Any]]) -> list[str]:
    return sorted({position["Mes"] for position in positions})


def brokers_in_month(positions: Iterable[dict[str, Any]], month: str) -> set[str]:
    return {position["Broker"] for position in positions if position["Mes"] == month}


def snapshot_by_month(positions: Iterable[dict[str, Any]]) -> dict[str, dict[tuple[str, str], dict[str, Any]]]:
    """Posiciones indexadas por mes y, dentro de cada mes, por (bróker, ISIN)."""
    snapshots: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for position in positions:
        snapshots.setdefault(position["Mes"], {})[(position["Broker"], position["ISIN"])] = position
    return snapshots


def derive_movements(positions: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Movimientos deducidos comparando cada mes con el anterior, bróker a bróker."""
    positions = list(positions)
    snapshots = snapshot_by_month(positions)
    months = sorted(snapshots)

    movements: list[dict[str, Any]] = []
    for previous_month, month in zip(months, months[1:]):
        current = snapshots[month]
        previous = snapshots[previous_month]
        # Solo se comparan los brókeres presentes en los dos meses: si una carga no trae un bróker,
        # callar es "no tengo datos", nunca "lo he vendido todo".
        comparable = brokers_in_month(positions, month) & brokers_in_month(positions, previous_month)

        for (broker, isin), position in current.items():
            if broker not in comparable:
                continue
            before = previous.get((broker, isin))
            delta = position["Cantidad"] - (before["Cantidad"] if before else 0.0)
            if abs(delta) <= QUANTITY_EPSILON:
                continue
            movements.append(_movement(month, position, delta, "alta" if before is None else ("compra" if delta > 0 else "venta")))

        for (broker, isin), position in previous.items():
            if broker not in comparable or (broker, isin) in current:
                continue
            movements.append(_movement(month, position, -position["Cantidad"], "cierre"))

    movements.sort(key=lambda item: (item["Mes"], item["Broker"], item["Nombre"]))
    return movements


def _movement(month: str, position: dict[str, Any], delta: float, kind: str) -> dict[str, Any]:
    price = _number(position.get("Precio cierre"))
    return {
        "Mes": month,
        "Broker": position["Broker"],
        "ISIN": position["ISIN"],
        "Nombre": position["Nombre"],
        "tipo": kind,
        "cantidad": round(delta, 8),
        # Aproximado a propósito: el precio disponible es el del cierre del mes, no el del día de la
        # operación. El importe exacto de lo aportado lo sabe Finanzas.
        "importe_estimado": round(delta * price, 2) if price else None,
    }


def allocate_fees(
    positions: Iterable[dict[str, Any]], fees: Iterable[dict[str, Any]]
) -> dict[tuple[str, str, str], float]:
    """Reparte cada comisión entre las posiciones de su bróker y mes, a prorrata del valor.

    Es una convención: una comisión de compra pertenece a una operación concreta, no al conjunto.
    Repartirla por valor es lo razonable cuando no se sabe de qué operación venía, y deja la
    rentabilidad de cada activo ya neta de gastos.
    """
    snapshots = snapshot_by_month(positions)
    allocated: dict[tuple[str, str, str], float] = {}

    for fee in fees or []:
        month = _text(fee.get("Mes"))[:7]
        broker = _text(fee.get("Broker"))
        amount = _number(fee.get("Importe"))
        if not month or not broker or not amount:
            continue

        same_broker = [
            position
            for position in snapshots.get(month, {}).values()
            if position["Broker"] == broker and (position_value(position) or 0.0) > 0
        ]
        total = sum(position_value(position) or 0.0 for position in same_broker)
        if not total:
            continue

        for position in same_broker:
            share = (position_value(position) or 0.0) / total
            key = (month, broker, position["ISIN"])
            allocated[key] = round(allocated.get(key, 0.0) + amount * share, 4)
    return allocated


# --- Carga incremental ---------------------------------------------------------------------------


def merge_positions(
    stored: Iterable[dict[str, Any]], incoming: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Combina lo guardado con el archivo recién subido.

    Un mes que venga en el archivo nuevo **manda entero sobre el guardado**, para que corregir o
    completar un mes ya cargado funcione; reemplazar fila a fila dejaría colgando las posiciones que
    se hayan quitado de ese mes. Los meses que solo estén en lo guardado se conservan.
    """
    incoming = read_positions(incoming)
    stored = read_positions(stored)
    replaced = {(position["Mes"], position["Broker"]) for position in incoming}
    kept = [position for position in stored if (position["Mes"], position["Broker"]) not in replaced]
    merged = kept + incoming
    merged.sort(key=lambda item: (item["Mes"], item["Broker"], item["Nombre"]))
    return merged


def merge_movements(
    stored: Iterable[dict[str, Any]], incoming: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Une los movimientos guardados con los del archivo nuevo, sin duplicar.

    Las exportaciones del bróker se solapan por fechas, así que la misma operación llega una y otra
    vez. Se deduplica por `MOVEMENT_KEY`, que incluye el saldo corriente precisamente para no
    confundir dos operaciones idénticas del mismo día con una repetida.
    """
    merged: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for row in list(stored or []) + list(incoming or []):
        key = tuple(_text(row.get(field)) for field in MOVEMENT_KEY)
        if key in seen:
            continue
        seen.add(key)
        merged.append(dict(row))
    return merged


def summarize_load(
    stored_positions: list[dict[str, Any]],
    merged_positions: list[dict[str, Any]],
    stored_movements: list[dict[str, Any]],
    merged_movements: list[dict[str, Any]],
    incoming_positions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Qué ha cambiado en esta carga, para no tener que fiarse a ciegas de que entró lo que se quería.

    Actualizado es lo que venía en el archivo y además ya estaba guardado. Antes se cruzaba todo lo
    guardado con todo lo resultante, así que una carga de un solo mes decía haber actualizado el
    histórico entero: justo lo contrario de lo que este resumen sirve para comprobar.
    """
    before = {(position["Mes"], position["Broker"]) for position in read_positions(stored_positions)}
    after = {(position["Mes"], position["Broker"]) for position in merged_positions}
    llegan = (
        {(position["Mes"], position["Broker"]) for position in read_positions(incoming_positions)}
        if incoming_positions is not None
        else after
    )
    new_months = sorted({month for month, _ in llegan - before})
    updated = sorted({f"{month} ({broker})" for month, broker in llegan & before})
    return {
        "posiciones": len(merged_positions),
        "posiciones_nuevas": len(merged_positions) - len(read_positions(stored_positions)),
        "meses_nuevos": new_months,
        "meses_actualizados": updated,
        "movimientos": len(merged_movements),
        "movimientos_nuevos": len(merged_movements) - len(stored_movements or []),
    }


def _text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "nat"} else text


def _number(value: Any) -> float:
    number = _optional_number(value)
    return 0.0 if number is None else number


def _optional_number(value: Any) -> float | None:
    text = _text(value)
    if not text:
        return None
    try:
        number = float(text.replace(",", "."))
    except ValueError:
        return None
    return None if number != number else number  # descarta NaN


UNCLASSIFIED = "Sin clasificar"

# Los fondos no se distinguen por el ISIN: se reconocen porque están en el mapa de fondos conocidos
# o porque su ISIN de Yahoo es uno de los `0P…`. Todo lo demás que no sea cripto es una acción.
CRYPTO_SECTOR = "Criptoactivo"


def classify(position: dict[str, Any], catalog: dict[str, dict[str, str]], fund_isins=()) -> dict[str, str]:
    """Región, sector, foco y tipo de una posición.

    El archivo del usuario solo dice qué tiene y cuánto; la clasificación vive en el catálogo de
    activos, que es el mismo que usa la vista anterior. Así un activo cae en el mismo grupo en las
    dos y los desgloses se pueden comparar.
    """
    isin = (position.get("ISIN") or "").upper()
    metadata = catalog.get(isin, {})
    # La cripto no siempre llega con un ISIN: en el archivo del usuario Bitcoin se apunta como
    # "BTC". Se reconoce por el nombre, igual que hace la vista anterior con el producto.
    if not metadata and "BITCOIN" in (position.get("Nombre") or "").upper():
        metadata = {"region": "Global", "sector": CRYPTO_SECTOR, "focus": "Alternativos"}

    sector = metadata.get("sector") or UNCLASSIFIED
    if isin in set(fund_isins):
        kind = "fund"
    elif sector == CRYPTO_SECTOR:
        kind = "crypto"
    else:
        kind = "stock"
    return {
        "region": metadata.get("region") or UNCLASSIFIED,
        "sector": sector,
        "foco": metadata.get("focus") or UNCLASSIFIED,
        "tipo": kind,
    }


def accumulate_costs(
    positions: Iterable[dict[str, Any]], price_of=None
) -> dict[tuple[str, str, str], float]:
    """Coste acumulado de cada posición, mes a mes.

    Hay dos caminos, y el primero manda:

    1. Si la fila trae **Aportado EUR**, ese es el dinero que se puso en esa posición ese mes, según
       el detalle de órdenes del bróker. Es el dato bueno: recoge el precio real de cada compra.
    2. Si no, el coste se deduce comparando meses —cuando sube la cantidad se valora la compra al
       cierre del mes—, y cuando baja se descuenta la parte proporcional del coste medio, que es
       como se calcula una venta parcial.

    La diferencia entre los dos caminos no es menor. Sin `Aportado EUR`, una posición que ya existía
    en el primer mes del archivo arrastra como "coste" su valor de mercado de aquel mes, no lo que
    se pagó por ella: la rentabilidad que sale entonces es la del archivo, no la del inversor.
    """
    positions = list(positions)
    snapshots = snapshot_by_month(positions)
    months = sorted(snapshots)

    costs: dict[tuple[str, str, str], float] = {}
    state: dict[tuple[str, str], tuple[float, float]] = {}  # (bróker, ISIN) -> (cantidad, coste)

    for month in months:
        current = snapshots[month]
        comparable = brokers_in_month(positions, month)

        for (broker, isin), position in current.items():
            quantity = _number(position.get("Cantidad"))
            previous_quantity, cost = state.get((broker, isin), (0.0, 0.0))
            delta = quantity - previous_quantity

            contribution = position.get("Aportado EUR")
            if contribution is not None:
                cost += float(contribution)
            elif delta > QUANTITY_EPSILON:
                price = _resolve_price(position, price_of)
                if price:
                    cost += delta * price

            if delta < -QUANTITY_EPSILON and previous_quantity > 0:
                # Venta parcial: se retira el coste medio de lo vendido, no su valor de mercado.
                cost *= max(0.0, quantity) / previous_quantity

            state[(broker, isin)] = (quantity, cost)
            costs[(month, broker, isin)] = round(cost, 2)

        # Una posición que desaparece de un bróker presente se ha vendido: su coste deja de contar.
        for (broker, isin) in list(state):
            if broker in comparable and (broker, isin) not in current:
                state.pop((broker, isin), None)

    return costs


def _resolve_price(position: dict[str, Any], price_of) -> float:
    price = _number(position.get("Precio cierre"))
    if price:
        return price
    if price_of is None:
        return 0.0
    return _number(price_of(position["ISIN"], position["Mes"]))
