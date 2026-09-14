import assert from 'node:assert/strict'
import test from 'node:test'

import {
  evaluateMetric,
  extractPortfolioTickers,
  formatBand,
  getBestTickerForMetric,
  getMetricBand,
  getScoreTrend,
  getValuation,
  getWeakestBlock,
  markDividendCuts,
} from './investmentMetrics.js'

const rules = {
  rpd_min: 3,
  rpd_max: 6,
  dgr5_min: 5,
  payout_min: 30,
  payout_max: 70,
  per_min: 10,
  per_max: 20,
  de_max: 1,
  roe_min: 10,
  streak_min: 10,
}

test('cada métrica se evalúa contra el umbral de su sector', () => {
  assert.deepEqual(getMetricBand('rpd_ttm', rules), { min: 3, max: 6 })
  assert.deepEqual(getMetricBand('de_ratio', rules), { min: null, max: 1 })
  assert.deepEqual(getMetricBand('roe', rules), { min: 10, max: null })
  assert.equal(getMetricBand('ev_ebitda', rules), null)
})

test('un valor dentro de la banda se marca bien y uno fuera se marca mal', () => {
  assert.equal(evaluateMetric(4, getMetricBand('rpd_ttm', rules)), 'on')
  assert.equal(evaluateMetric(1.5, getMetricBand('rpd_ttm', rules)), 'off')
  assert.equal(evaluateMetric(8, getMetricBand('rpd_ttm', rules)), 'off')
})

test('sin dato o sin banda no se pinta nada, para no fingir un juicio', () => {
  assert.equal(evaluateMetric(null, getMetricBand('rpd_ttm', rules)), null)
  assert.equal(evaluateMetric(4, null), null)
})

test('la banda se escribe según tenga mínimo, máximo o ambos', () => {
  assert.equal(formatBand({ min: 3, max: 6 }, '%'), 'objetivo 3–6 %')
  assert.equal(formatBand({ min: 10, max: null }, '%'), 'mín. 10 %')
  assert.equal(formatBand({ min: null, max: 1 }, ''), 'máx. 1')
})

test('el bloque más débil es el que más puntos pierde sobre su peso', () => {
  const weakest = getWeakestBlock({ Dividendo: 38, Solidez: 12, Valoración: 14, Historial: 19 })

  assert.equal(weakest.block, 'Solidez')
  assert.equal(weakest.lost, 13)
  assert.equal(weakest.weight, 25)
})

test('más RPD que su media de 5 años significa que cotiza barata', () => {
  const valuation = getValuation({ rpd_ttm: 4.5, rpd_avg5: 3.0, reference_price: 60 })

  assert.equal(valuation.verdict, 'barata')
  assert.equal(valuation.difference, 1.5)
  assert.equal(valuation.referencePrice, 60)
})

test('menos RPD que su media significa que cotiza cara', () => {
  assert.equal(getValuation({ rpd_ttm: 2.36, rpd_avg5: 3.29 }).verdict, 'cara')
})

test('una diferencia pequeña se considera estar en su media', () => {
  assert.equal(getValuation({ rpd_ttm: 3.1, rpd_avg5: 3.0 }).verdict, 'en su media')
})

test('sin media histórica no se inventa una valoración', () => {
  assert.equal(getValuation({ rpd_ttm: 3.1 }), null)
  assert.equal(getValuation({ rpd_ttm: 3.1, rpd_avg5: 0 }), null)
})

test('los recortes de dividendo se marcan para poder pintarlos en rojo', () => {
  const rows = markDividendCuts([
    { year: 2022, amount: 1.0 },
    { year: 2023, amount: 1.2 },
    { year: 2024, amount: 0.8 },
    { year: 2025, amount: 0.8 },
  ])

  assert.deepEqual(rows.map((row) => row.trend), ['flat', 'up', 'down', 'flat'])
})

test('la tendencia del score compara las dos últimas tomas', () => {
  const trend = getScoreTrend([
    { date: '2026-01-01', score: 71 },
    { date: '2026-06-01', score: 68 },
  ])

  assert.equal(trend.delta, -3)
  assert.equal(trend.since, '2026-01-01')
})

test('con una sola toma todavía no hay tendencia', () => {
  assert.equal(getScoreTrend([{ date: '2026-01-01', score: 71 }]), null)
})

test('los tickers de la cartera salen sin duplicados y ordenados por valor', () => {
  const tickers = extractPortfolioTickers({
    positions: [
      { ticker: 'pg', name: 'Procter', current_value: 100 },
      { ticker: 'NVDA', name: 'NVIDIA', current_value: 400 },
      { ticker: 'PG', name: 'Procter (otro bróker)', current_value: 50 },
      { ticker: null, name: 'Fondo sin ticker', current_value: 900 },
    ],
  })

  assert.deepEqual(tickers.map((item) => item.ticker), ['NVDA', 'PG'])
})

test('sin cartera cargada la lista de atajos queda vacía', () => {
  assert.deepEqual(extractPortfolioTickers(null), [])
})

test('la mejor celda depende de si conviene que la métrica sea alta o baja', () => {
  const columns = [
    { ticker: 'KO', metrics: { rpd_ttm: 2.4, payout: 75 } },
    { ticker: 'JNJ', metrics: { rpd_ttm: 3.3, payout: 48 } },
  ]

  assert.equal(getBestTickerForMetric(columns, { key: 'rpd_ttm', better: 'high' }), 'JNJ')
  assert.equal(getBestTickerForMetric(columns, { key: 'payout', better: 'low' }), 'JNJ')
  assert.equal(getBestTickerForMetric(columns, { key: 'rpd_avg5', better: null }), null)
})
