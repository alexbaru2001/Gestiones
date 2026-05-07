export const comparisonRows = [
  { label: 'Total', field: 'total' },
  { label: 'Ahorros', field: '💰 Ahorros' },
  { label: 'Gasto', field: '💳 Gasto del mes' },
  { label: 'Presupuesto', field: '💸 Presupuesto Mes' },
]

export function getHistoryRows(result) {
  return result?.historial?.resumen ?? []
}

export function getSelectedRow(rows, latest, selectedMonth) {
  return rows.find((row) => row.Mes === selectedMonth) ?? latest ?? null
}

export function getPreviousRow(rows, selectedRow) {
  if (!selectedRow) return null
  const index = rows.findIndex((row) => row.Mes === selectedRow.Mes)
  return index > 0 ? rows[index - 1] : null
}

export function getRecentRows(rows, limit = 8) {
  return rows.slice(-limit)
}

export function getTrendMax(rows) {
  return Math.max(
    1,
    ...rows.flatMap((row) => [
      Math.abs(Number(row.total) || 0),
      Math.abs(Number(row['💳 Gasto del mes']) || 0),
      Math.abs(Number(row['💸 Presupuesto Mes']) || 0),
    ]),
  )
}

export function getObjectiveRows(result) {
  return result?.historial?.objetivos ?? []
}

export function getObjectiveNames(objectiveRows) {
  return Array.from(new Set(objectiveRows.map((row) => row.Objetivo))).sort()
}

export function filterObjectiveRows(objectiveRows, selectedObjective) {
  return selectedObjective === 'all'
    ? objectiveRows
    : objectiveRows.filter((row) => row.Objetivo === selectedObjective)
}

export function getObjectiveTotals(objectiveRows) {
  return objectiveRows.reduce(
    (totals, row) => ({
      aporte: totals.aporte + (Number(row.aporte_mes) || 0),
      gasto: totals.gasto + (Number(row.gastos_etiquetados_mes) || 0),
      liquidacion: totals.liquidacion + (Number(row.liquidacion) || 0),
    }),
    { aporte: 0, gasto: 0, liquidacion: 0 },
  )
}
