import assert from 'node:assert/strict'
import test from 'node:test'

import { createObjective, objectiveFromApi, objectiveToPayload } from './objectives.js'

test('createObjective returns editable defaults', () => {
  assert.deepEqual(createObjective('id-1'), {
    id: 'id-1',
    nombre: '',
    etiquetas: '',
    fraccion_presupuesto: 0.1,
    duracion_meses: 1,
    mes_inicio: '2024-10',
    saldo_inicial: 0,
  })
})

test('objectiveToPayload normalizes comma separated tags', () => {
  const payload = objectiveToPayload({
    id: 'local-id',
    nombre: 'Coche',
    etiquetas: ' Coche, taller, ',
    fraccion_presupuesto: 0.2,
    duracion_meses: 6,
    mes_inicio: '2024-10',
    saldo_inicial: 100,
  })

  assert.deepEqual(payload, {
    nombre: 'Coche',
    etiquetas: ['coche', 'taller'],
    fraccion_presupuesto: 0.2,
    duracion_meses: 6,
    mes_inicio: '2024-10',
    saldo_inicial: 100,
  })
})

test('objectiveFromApi maps persisted objectives to editable rows', () => {
  const objective = objectiveFromApi(
    {
      nombre: 'Viaje',
      etiquetas: ['hotel', 'tren'],
      fraccion_presupuesto: '0.15',
      duracion_meses: '3',
      mes_inicio: '2024-12-01',
      saldo_inicial: '50',
    },
    'id-2',
  )

  assert.deepEqual(objective, {
    id: 'id-2',
    nombre: 'Viaje',
    etiquetas: 'hotel, tren',
    fraccion_presupuesto: 0.15,
    duracion_meses: 3,
    mes_inicio: '2024-12',
    saldo_inicial: 50,
  })
})
