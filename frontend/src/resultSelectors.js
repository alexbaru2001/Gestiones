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

export function getDividendPayments(result) {
  return result?.analisis?.dividendos_pagos ?? []
}

// Acumula los pagos de dividendos por empresa hasta una fecha de corte (inclusive), para que cada
// pantalla pueda mostrar "lo cobrado hasta la foto/mes seleccionado" en vez del total histórico.
// Pasa `untilDate` (YYYY-MM-DD) para un corte exacto de fecha, o `untilMonth` (YYYY-MM) para un
// corte por mes; sin ninguno de los dos, se acumulan todos los pagos.
export function aggregateDividendsByCompany(payments, { untilDate, untilMonth } = {}) {
  const scoped = payments.filter((payment) => {
    if (untilDate) return payment.fecha <= untilDate
    if (untilMonth) return payment.fecha.slice(0, 7) <= untilMonth
    return true
  })
  const groups = new Map()
  const commentCounts = new Map()
  for (const payment of [...scoped].sort((a, b) => (a.fecha < b.fecha ? -1 : 1))) {
    const key = String(payment.codigo).toLowerCase()
    if (!groups.has(key)) {
      groups.set(key, { codigo: payment.codigo, empresa: payment.codigo, total: 0, pagos: 0, ultimo_pago: null })
    }
    const group = groups.get(key)
    group.total += Number(payment.cantidad) || 0
    group.pagos += 1
    group.ultimo_pago = payment.fecha
    const comment = String(payment.comentario ?? '').trim()
    if (comment) {
      if (!commentCounts.has(key)) commentCounts.set(key, new Map())
      const counts = commentCounts.get(key)
      counts.set(comment, (counts.get(comment) ?? 0) + 1)
    }
  }
  for (const [key, group] of groups) {
    const counts = commentCounts.get(key)
    if (counts) {
      let bestComment = null
      let bestCount = 0
      for (const [comment, count] of counts) {
        if (count > bestCount) {
          bestComment = comment
          bestCount = count
        }
      }
      if (bestComment) group.empresa = bestComment
    }
    group.total = Math.round(group.total * 100) / 100
  }
  return Array.from(groups.values()).sort((a, b) => b.total - a.total)
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
