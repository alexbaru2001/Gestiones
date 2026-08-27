import assert from 'node:assert/strict'
import test from 'node:test'

import {
  aggregateDividendsByCompany,
  filterObjectiveRows,
  getDividendPayments,
  getHistoryRows,
  getObjectiveNames,
  getObjectiveRows,
  getObjectiveTotals,
  getPreviousRow,
  getRecentRows,
  getSelectedRow,
  getTrendMax,
} from './resultSelectors.js'

const rows = [
  { Mes: '2024-09', total: 100, '💳 Gasto del mes': 20, '💸 Presupuesto Mes': 50 },
  { Mes: '2024-10', total: 200, '💳 Gasto del mes': 30, '💸 Presupuesto Mes': 60 },
  { Mes: '2024-11', total: -300, '💳 Gasto del mes': 40, '💸 Presupuesto Mes': 70 },
]

test('history selectors handle rows and month selection', () => {
  const result = { historial: { resumen: rows, ultimo_mes: rows[2], objetivos: [] } }

  assert.equal(getHistoryRows(result), rows)
  assert.equal(getSelectedRow(rows, rows[2], '2024-10'), rows[1])
  assert.equal(getSelectedRow(rows, rows[2], ''), rows[2])
  assert.equal(getPreviousRow(rows, rows[1]), rows[0])
  assert.equal(getPreviousRow(rows, rows[0]), null)
  assert.deepEqual(getRecentRows(rows, 2), rows.slice(1))
  assert.equal(getTrendMax(rows), 300)
})

test('objective selectors filter names and totals', () => {
  const objectives = [
    { Objetivo: 'Coche', aporte_mes: 10, gastos_etiquetados_mes: 2, liquidacion: 1 },
    { Objetivo: 'Viaje', aporte_mes: 20, gastos_etiquetados_mes: 3, liquidacion: -4 },
    { Objetivo: 'Coche', aporte_mes: 30, gastos_etiquetados_mes: 5, liquidacion: 6 },
  ]
  const result = { historial: { objetivos: objectives } }

  assert.equal(getObjectiveRows(result), objectives)
  assert.deepEqual(getObjectiveNames(objectives), ['Coche', 'Viaje'])
  assert.deepEqual(filterObjectiveRows(objectives, 'Viaje'), [objectives[1]])
  assert.deepEqual(getObjectiveTotals(filterObjectiveRows(objectives, 'Coche')), {
    aporte: 40,
    gasto: 7,
    liquidacion: 7,
  })
})

test('getDividendPayments reads the raw payment list and defaults to empty', () => {
  const payments = [{ fecha: '2024-11-15', codigo: 'IB', comentario: 'Iberdrola', cantidad: 7.57 }]
  assert.equal(getDividendPayments({ analisis: { dividendos_pagos: payments } }), payments)
  assert.deepEqual(getDividendPayments({}), [])
  assert.deepEqual(getDividendPayments(null), [])
})

test('aggregateDividendsByCompany accumulates only up to the given cutoff', () => {
  const payments = [
    { fecha: '2024-10-15', codigo: 'PyG', comentario: 'Procter and Gamber', cantidad: 0.28 },
    { fecha: '2024-11-05', codigo: 'PyG', comentario: 'Procter and Gamber', cantidad: 3.98 },
    { fecha: '2024-11-15', codigo: 'IB', comentario: 'Iberdrola', cantidad: 7.57 },
  ]

  const all = aggregateDividendsByCompany(payments)
  assert.deepEqual(all.map((row) => [row.empresa, row.total]), [
    ['Iberdrola', 7.57],
    ['Procter and Gamber', 4.26],
  ])

  const untilOctober = aggregateDividendsByCompany(payments, { untilMonth: '2024-10' })
  assert.deepEqual(untilOctober.map((row) => [row.empresa, row.total, row.pagos]), [['Procter and Gamber', 0.28, 1]])

  const untilNov10 = aggregateDividendsByCompany(payments, { untilDate: '2024-11-10' })
  assert.deepEqual(untilNov10.map((row) => [row.empresa, row.total]), [['Procter and Gamber', 4.26]])

  assert.deepEqual(aggregateDividendsByCompany(payments, { untilDate: '2024-10-01' }), [])
})
