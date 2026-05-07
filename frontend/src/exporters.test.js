import assert from 'node:assert/strict'
import test from 'node:test'

import { toCsv } from './exporters.js'

test('toCsv returns empty text for empty records', () => {
  assert.equal(toCsv([]), '')
})

test('toCsv preserves discovered columns and escapes csv values', () => {
  const csv = toCsv([
    { Mes: '2024-10', Total: 100, Nota: 'normal' },
    { Mes: '2024-11', Total: 200, Nota: 'texto, con "comillas"' },
  ])

  assert.equal(csv, 'Mes,Total,Nota\n2024-10,100,normal\n2024-11,200,"texto, con ""comillas"""')
})
