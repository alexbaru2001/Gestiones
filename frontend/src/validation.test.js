import assert from 'node:assert/strict'
import test from 'node:test'

import { validateFinanceInput } from './validation.js'

const validParams = {
  fecha_inicio: '2024-10-01',
  porcentaje_gasto: 0.3,
  porcentaje_inversion: 0.1,
  porcentaje_vacaciones: 0.05,
}

test('validateFinanceInput accepts valid params without objectives', () => {
  assert.deepEqual(validateFinanceInput(validParams, []), [])
})

test('validateFinanceInput rejects invalid fractions and missing date', () => {
  const messages = validateFinanceInput(
    {
      fecha_inicio: '',
      porcentaje_gasto: 1.2,
      porcentaje_inversion: -0.1,
      porcentaje_vacaciones: Number.NaN,
    },
    [],
  )

  assert.deepEqual(messages, [
    'La fecha de inicio es obligatoria.',
    'Gasto debe estar entre 0 y 1.',
    'Inversión debe estar entre 0 y 1.',
    'Vacaciones debe estar entre 0 y 1.',
  ])
})

test('validateFinanceInput validates objective names, duration, month and total fraction', () => {
  const messages = validateFinanceInput(validParams, [
    {
      nombre: 'Coche',
      fraccion_presupuesto: 0.7,
      duracion_meses: 12,
      mes_inicio: '2024-10',
    },
    {
      nombre: 'Coche',
      fraccion_presupuesto: 0.4,
      duracion_meses: 0,
      mes_inicio: '202410',
    },
  ])

  assert.deepEqual(messages, [
    'Objetivo "Coche": los meses deben ser un entero mayor que 0.',
    'Objetivo "Coche": el inicio debe tener formato YYYY-MM.',
    'Hay objetivos repetidos: Coche.',
    'La suma de fracciones de objetivos no puede superar 1.',
  ])
})
