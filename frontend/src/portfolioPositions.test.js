import assert from 'node:assert/strict'
import test from 'node:test'

import {
  findIncompleteMonths,
  findPositionsWithoutValue,
  groupByMonth,
  movementsOfMonth,
  summarizeMovements,
} from './portfolioPositions.js'

const posiciones = [
  { Mes: '2026-01', Broker: 'Trade Republic', ISIN: 'A', Nombre: 'Alfa', valor: 100, comision_imputada: 0.5 },
  { Mes: '2026-01', Broker: 'Degiro', ISIN: 'B', Nombre: 'Beta', valor: 50, comision_imputada: null },
  { Mes: '2026-02', Broker: 'Trade Republic', ISIN: 'A', Nombre: 'Alfa', valor: 120, comision_imputada: null },
]

test('las posiciones se agrupan por mes con su valor y sus comisiones', () => {
  const meses = groupByMonth(posiciones)

  assert.deepEqual(meses.map((m) => m.month), ['2026-01', '2026-02'])
  assert.equal(meses[0].value, 150)
  assert.equal(meses[0].fees, 0.5)
  assert.deepEqual(meses[0].brokers, ['Degiro', 'Trade Republic'])
})

test('un mes al que le falta un bróker se señala', () => {
  // 2026-02 solo trae Trade Republic: su total se quedaría corto y parecería una caída.
  assert.deepEqual(findIncompleteMonths(posiciones), [{ month: '2026-02', missing: ['Degiro'] }])
})

test('si todos los meses traen los mismos brókeres no se avisa de nada', () => {
  const completos = [
    { Mes: '2026-01', Broker: 'TR', valor: 10 },
    { Mes: '2026-02', Broker: 'TR', valor: 12 },
  ]

  assert.deepEqual(findIncompleteMonths(completos), [])
})

test('los movimientos se cuentan por tipo', () => {
  const movimientos = [
    { Mes: '2026-02', tipo: 'compra' }, { Mes: '2026-02', tipo: 'alta' },
    { Mes: '2026-03', tipo: 'venta' }, { Mes: '2026-03', tipo: 'cierre' }, { Mes: '2026-03', tipo: 'compra' },
  ]

  assert.deepEqual(summarizeMovements(movimientos), { alta: 1, compra: 2, venta: 1, cierre: 1 })
  assert.equal(movementsOfMonth(movimientos, '2026-03').length, 3)
})

test('sin movimientos el recuento es todo ceros, no undefined', () => {
  assert.deepEqual(summarizeMovements([]), { alta: 0, compra: 0, venta: 0, cierre: 0 })
})

test('las posiciones en divisa sin importe en euros se listan aparte', () => {
  const conHueco = [...posiciones, { Mes: '2025-12', Broker: 'Degiro', Nombre: 'PG', Moneda: 'USD', valor: null }]

  assert.deepEqual(findPositionsWithoutValue(conHueco).map((p) => p.Nombre), ['PG'])
})

test('un valor de 0 es un dato real y no se confunde con un hueco', () => {
  const cero = [{ Mes: '2026-01', Broker: 'TR', Nombre: 'Derechos', valor: 0 }]

  assert.deepEqual(findPositionsWithoutValue(cero), [])
})
