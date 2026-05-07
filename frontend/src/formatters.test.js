import assert from 'node:assert/strict'
import test from 'node:test'

import { formatDelta, formatMoney, formatPercent } from './formatters.js'

test('formatMoney formats euro amounts for Spanish locale', () => {
  assert.equal(formatMoney(1234.5), '1234,50\u00a0€')
})

test('formatMoney returns dash for non numeric values', () => {
  assert.equal(formatMoney(null), '-')
  assert.equal(formatMoney('100'), '-')
})

test('formatPercent rounds decimal fractions', () => {
  assert.equal(formatPercent(0.305), '31%')
})

test('formatDelta prefixes positive values', () => {
  assert.equal(formatDelta(12), '+12,00\u00a0€')
  assert.equal(formatDelta(-12), '-12,00\u00a0€')
  assert.equal(formatDelta(undefined), '-')
})
