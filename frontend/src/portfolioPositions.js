// Lógica pura de la vista del archivo de posiciones: agrupar por mes, detectar meses incompletos
// y resumir los movimientos deducidos.

function isNumber(value) {
  return value !== null && value !== undefined && value !== '' && Number.isFinite(Number(value))
}

export function groupByMonth(positions = []) {
  const months = new Map()
  for (const position of positions) {
    const month = position?.Mes
    if (!month) continue
    if (!months.has(month)) months.set(month, { month, positions: [], brokers: new Set(), value: 0, fees: 0 })
    const entry = months.get(month)
    entry.positions.push(position)
    entry.brokers.add(position.Broker)
    if (isNumber(position.valor)) entry.value += Number(position.valor)
    if (isNumber(position.comision_imputada)) entry.fees += Number(position.comision_imputada)
  }
  return [...months.values()]
    .map((entry) => ({ ...entry, brokers: [...entry.brokers].sort(), value: Math.round(entry.value * 100) / 100 }))
    .sort((left, right) => left.month.localeCompare(right.month))
}

/**
 * Meses a los que les falta algún bróker que sí aparece en otros meses.
 *
 * Importa avisarlo porque la comparación entre meses solo mira los brókeres presentes: un mes
 * incompleto no inventa ventas, pero su valor total sí se queda corto y se leería como una caída.
 */
export function findIncompleteMonths(positions = []) {
  const grouped = groupByMonth(positions)
  const allBrokers = new Set(positions.map((position) => position?.Broker).filter(Boolean))
  return grouped
    .map((entry) => ({ month: entry.month, missing: [...allBrokers].filter((broker) => !entry.brokers.includes(broker)).sort() }))
    .filter((entry) => entry.missing.length > 0)
}

export function summarizeMovements(movements = []) {
  const counts = { alta: 0, compra: 0, venta: 0, cierre: 0 }
  for (const movement of movements) {
    if (movement?.tipo in counts) counts[movement.tipo] += 1
  }
  return counts
}

export function movementsOfMonth(movements = [], month) {
  return movements.filter((movement) => movement?.Mes === month)
}

/** Las posiciones sin valor son las que están en divisa y no traen el importe en euros del bróker. */
export function findPositionsWithoutValue(positions = []) {
  return positions.filter((position) => !isNumber(position?.valor))
}
