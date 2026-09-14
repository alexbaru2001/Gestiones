import assert from 'node:assert/strict'
import test from 'node:test'

import {
  chunkTickers,
  describeProgress,
  getValuationLabel,
  rankByPrescore,
  rankByScore,
  toExplorationRow,
} from './exploration.js'

test('el universo se reparte en tandas del tamaño que admite el backend', () => {
  const chunks = chunkTickers(Array.from({ length: 250 }, (_, index) => `T${index}`), 100)

  assert.equal(chunks.length, 3)
  assert.equal(chunks[0].length, 100)
  assert.equal(chunks[2].length, 50)
})

test('un universo vacío no genera tandas', () => {
  assert.deepEqual(chunkTickers([], 100), [])
})

test('la criba se ordena de mayor a menor y se corta en el número de finalistas', () => {
  const rows = [
    { ticker: 'A', prescore: 30 },
    { ticker: 'B', prescore: 70 },
    { ticker: 'C', prescore: 50 },
  ]

  assert.deepEqual(rankByPrescore(rows, 2).map((row) => row.ticker), ['B', 'C'])
})

test('las empresas sin dividendo se descartan en la criba', () => {
  const rows = [{ ticker: 'NODIV', prescore: 0 }, { ticker: 'KO', prescore: 45 }]

  assert.deepEqual(rankByPrescore(rows).map((row) => row.ticker), ['KO'])
})

test('de cada análisis solo se guarda lo que se enseña en la tabla', () => {
  const row = toExplorationRow(
    {
      ticker: 'KO',
      name: 'Coca-Cola',
      sector: 'Consumer Defensive',
      score: 73.8,
      price: 88.29,
      flags: [],
      exchange: { currency: 'USD' },
      metrics: { rpd_ttm: 2.36, rpd_avg5: 3.29, payout: 62.4, per_ttm: 26.5, streak_years: 65 },
      price_history: [{ date: '2026-01-01', close: 1 }],
      ai_analysis: { text: 'largo' },
    },
    51.2,
  )

  assert.equal(row.ticker, 'KO')
  assert.equal(row.prescore, 51.2)
  assert.equal(row.rpd_avg5, 3.29)
  assert.equal(row.price_history, undefined)
  assert.equal(row.ai_analysis, undefined)
})

test('el ranking final usa el score real, no el de la criba', () => {
  const rows = [
    { ticker: 'A', prescore: 90, score: 55 },
    { ticker: 'B', prescore: 40, score: 82 },
  ]

  assert.deepEqual(rankByScore(rows, 2).map((row) => row.ticker), ['B', 'A'])
})

test('una empresa que falló el análisis no entra en el ranking final', () => {
  assert.deepEqual(rankByScore([{ ticker: 'A', score: null }, { ticker: 'B', score: 60 }]).map((r) => r.ticker), ['B'])
})

test('la valoración compara la RPD con su media de 5 años', () => {
  assert.equal(getValuationLabel({ rpd_ttm: 4.5, rpd_avg5: 3.0 }), 'barata')
  assert.equal(getValuationLabel({ rpd_ttm: 2.36, rpd_avg5: 3.29 }), 'cara')
  assert.equal(getValuationLabel({ rpd_ttm: 3.0 }), null)
})

test('el contador no puede pasarse del total en la última tanda', () => {
  assert.equal(describeProgress({ phase: 'criba', done: 503, total: 503 }), 'Criba 503 de 503 empresas')
})

test('el avance se describe distinto en cada fase', () => {
  assert.equal(describeProgress({ phase: 'criba', done: 200, total: 503 }), 'Criba 201 de 503 empresas')
  assert.equal(
    describeProgress({ phase: 'analisis', done: 5, total: 40, ticker: 'KO' }),
    'Análisis a fondo 6 de 40 · KO',
  )
  assert.equal(describeProgress(null), '')
})
