import assert from 'node:assert/strict'
import test from 'node:test'

import {
  filterObjectiveRows,
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
