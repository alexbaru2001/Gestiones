import assert from 'node:assert/strict'
import test from 'node:test'

import {
  buildReviewRows,
  detectReviewChanges,
  getValuationVerdict,
  splitAnalyzablePositions,
  summarizeReview,
} from './portfolioReview.js'

const snapshot = {
  positions: [
    { ticker: 'UNH', name: 'UnitedHealth', current_value: 1198.37, broker: 'Trade Republic' },
    { ticker: 'PG', name: 'Procter', current_value: 620.55, broker: 'DeGiro' },
    { ticker: 'pg', name: 'Procter (TR)', current_value: 54.89, broker: 'Trade Republic' },
    { ticker: null, name: 'VANGUARD US 500', current_value: 1257.02, asset_type: 'fondo' },
    { ticker: '', name: 'BITCOIN', current_value: 345.59, asset_type: 'cripto' },
  ],
}

test('la misma empresa en dos brókeres se revisa una vez con el valor sumado', () => {
  const { analyzable } = splitAnalyzablePositions(snapshot)

  const procter = analyzable.find((row) => row.ticker === 'PG')
  assert.equal(analyzable.length, 2)
  assert.ok(Math.abs(procter.value - 675.44) < 0.001)
  assert.equal(procter.broker, 'DeGiro · Trade Republic')
})

test('los pesos se calculan sobre el total de la cartera, no solo sobre lo analizable', () => {
  const { analyzable, total } = splitAnalyzablePositions(snapshot)

  assert.ok(Math.abs(total - 3476.42) < 0.001)
  assert.equal(analyzable[0].ticker, 'UNH')
  assert.ok(Math.abs(analyzable[0].weight - 34.47) < 0.01)
})

test('lo que no tiene ticker se aparta y se cuantifica en vez de ignorarlo', () => {
  const { skipped, skippedWeight } = splitAnalyzablePositions(snapshot)

  assert.deepEqual(skipped.map((item) => item.name), ['VANGUARD US 500', 'BITCOIN'])
  assert.ok(Math.abs(skippedWeight - 46.1) < 0.1)
})

test('sin cartera cargada no hay nada que revisar', () => {
  assert.deepEqual(splitAnalyzablePositions(null).analyzable, [])
})

test('la nota de la cartera se pondera por valor', () => {
  const summary = summarizeReview([
    { ticker: 'UNH', last_score: 80, value: 1000, flags: [] },
    { ticker: 'MC', last_score: 40, value: 10, flags: ['payout alto'] },
  ])

  // La media simple sería 60; ponderada apenas baja de 80 porque MC pesa 10 € de 1010 €.
  assert.equal(summary.simpleScore, 60)
  assert.ok(Math.abs(summary.weightedScore - 79.6) < 0.1)
  assert.equal(summary.flaggedCount, 1)
  assert.ok(Math.abs(summary.flaggedWeight - 0.99) < 0.01)
})

test('sin posiciones puntuadas no se inventa un resumen', () => {
  assert.equal(summarizeReview([{ ticker: 'KO', last_score: null, value: 100 }]), null)
})

test('el veredicto compara la RPD actual con su media de 5 años', () => {
  assert.equal(getValuationVerdict({ rpd_ttm: 4.5, rpd_avg5: 3.0 }), 'barata')
  assert.equal(getValuationVerdict({ rpd_ttm: 2.36, rpd_avg5: 3.29 }), 'cara')
  assert.equal(getValuationVerdict({ rpd_ttm: 3.05, rpd_avg5: 3.0 }), 'en su media')
  assert.equal(getValuationVerdict({ rpd_ttm: 3.0 }), null)
})

test('cada fila lleva su variación desde la revisión anterior', () => {
  const rows = buildReviewRows([
    {
      ticker: 'KO',
      flags: ['payout alto'],
      history: [
        { date: '2026-03-01', score: 78, rpd_ttm: 3.4, rpd_avg5: 3.3 },
        { date: '2026-09-01', score: 71, rpd_ttm: 2.4, rpd_avg5: 3.3 },
      ],
    },
  ])

  assert.equal(rows[0].delta, -7)
  assert.equal(rows[0].previousDate, '2026-03-01')
  assert.equal(rows[0].valuation, 'cara')
  assert.equal(rows[0].flagCount, 1)
})

test('una caída de score relevante se avisa y una pequeña no', () => {
  const caida = detectReviewChanges([
    { ticker: 'KO', history: [{ date: '1', score: 78 }, { date: '2', score: 71 }] },
  ])
  const ruido = detectReviewChanges([
    { ticker: 'KO', history: [{ date: '1', score: 78 }, { date: '2', score: 76 }] },
  ])

  assert.equal(caida[0].kind, 'score')
  assert.equal(caida[0].severity, 'bad')
  assert.deepEqual(ruido, [])
})

test('las banderas rojas nuevas se avisan, las que ya estaban no', () => {
  const nuevas = detectReviewChanges([
    { ticker: 'IBE.MC', history: [{ date: '1', score: 70, flags: 0 }, { date: '2', score: 70, flags: 1 }] },
  ])
  const mismas = detectReviewChanges([
    { ticker: 'IBE.MC', history: [{ date: '1', score: 70, flags: 1 }, { date: '2', score: 70, flags: 1 }] },
  ])

  assert.equal(nuevas[0].kind, 'flags')
  assert.deepEqual(mismas, [])
})

test('pasar de cara a barata frente a su media se avisa como algo bueno', () => {
  const changes = detectReviewChanges([
    {
      ticker: 'KO',
      history: [
        { date: '1', score: 70, rpd_ttm: 2.4, rpd_avg5: 3.3 },
        { date: '2', score: 70, rpd_ttm: 3.8, rpd_avg5: 3.3 },
      ],
    },
  ])

  const valoracion = changes.find((change) => change.kind === 'valoracion')
  assert.equal(valoracion.severity, 'good')
  assert.match(valoracion.text, /de cara a barata/)
})

test('con una sola revisión todavía no hay cambios que contar', () => {
  assert.deepEqual(detectReviewChanges([{ ticker: 'KO', history: [{ date: '1', score: 70 }] }]), [])
})
