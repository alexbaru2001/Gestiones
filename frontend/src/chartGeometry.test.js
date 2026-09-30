import assert from 'node:assert/strict'
import test from 'node:test'

import {
  clampIndex,
  getSeriesChange,
  getSvgCoordinates,
  getValueDomain,
  isNumber,
  linePath,
  nearestIndexFromX,
  polylinePoints,
} from './chartGeometry.js'

test('el trazo es recto: solo comandos M y L, sin curvas', () => {
  const path = linePath([
    { x: 0, y: 10 },
    { x: 10, y: 5 },
    { x: 20, y: 8 },
  ])

  assert.equal(path, 'M 0.0,10.0 L 10.0,5.0 L 20.0,8.0')
  assert.ok(!path.includes('C'))
  assert.ok(!path.includes('Q'))
})

test('sin puntos no se dibuja nada en vez de un trazo roto', () => {
  assert.equal(linePath([]), '')
  assert.equal(linePath(undefined), '')
  assert.equal(polylinePoints([]), '')
})

test('el cruce se engancha al punto más cercano en horizontal', () => {
  const xs = [0, 50, 100, 150]

  assert.equal(nearestIndexFromX(0, xs), 0)
  assert.equal(nearestIndexFromX(48, xs), 1)
  assert.equal(nearestIndexFromX(74, xs), 1)
  assert.equal(nearestIndexFromX(76, xs), 2)
  assert.equal(nearestIndexFromX(999, xs), 3)
})

test('un ratón fuera del área por la izquierda se queda en el primer dato', () => {
  assert.equal(nearestIndexFromX(-200, [10, 20, 30]), 0)
})

test('sin serie no hay punto que enganchar', () => {
  assert.equal(nearestIndexFromX(10, []), null)
})

test('la variación se mide desde el primer valor del periodo hasta el punto señalado', () => {
  const change = getSeriesChange([100, 120, 180], 1)

  assert.equal(change.delta, 20)
  assert.equal(change.percent, 20)
  assert.equal(change.from, 100)
  assert.equal(change.to, 120)
})

test('sin índice, la variación es la del último valor de la serie', () => {
  assert.equal(getSeriesChange([100, 120, 180]).delta, 80)
})

test('una variación negativa conserva su signo, no se muestra en valor absoluto', () => {
  const change = getSeriesChange([200, 150])

  assert.equal(change.delta, -50)
  assert.equal(change.percent, -25)
})

test('con un primer valor de 0 se da el importe pero no un porcentaje infinito', () => {
  const change = getSeriesChange([0, 50])

  assert.equal(change.delta, 50)
  assert.equal(change.percent, null)
})

test('partiendo de un saldo negativo el porcentaje usa la magnitud, para que subir sea positivo', () => {
  const change = getSeriesChange([-200, -100])

  assert.equal(change.delta, 100)
  assert.equal(change.percent, 50)
})

test('un hueco de datos no se cuela como un cero en la variación', () => {
  assert.equal(getSeriesChange([null, 50]), null)
  assert.equal(getSeriesChange([100, null], 1), null)
  assert.equal(getSeriesChange([]), null)
})

test('isNumber distingue el cero real del dato ausente', () => {
  assert.equal(isNumber(0), true)
  assert.equal(isNumber(null), false)
  assert.equal(isNumber(''), false)
  assert.equal(isNumber(undefined), false)
  assert.equal(isNumber(Number.NaN), false)
})

test('las coordenadas reparten los puntos entre los márgenes y respetan el rango', () => {
  const coordinates = getSvgCoordinates([0, 5, 10], 0, 10, {
    width: 120,
    height: 100,
    paddingLeft: 10,
    paddingRight: 10,
    paddingTop: 0,
    paddingBottom: 0,
  })

  assert.equal(coordinates[0].x, 10)
  assert.equal(coordinates[2].x, 110)
  assert.equal(coordinates[0].y, 100) // el mínimo abajo
  assert.equal(coordinates[2].y, 0) // el máximo arriba
  assert.equal(coordinates[1].y, 50)
})

test('una serie plana no divide por cero', () => {
  const coordinates = getSvgCoordinates([7, 7], 7, 7, { width: 100, height: 50 })

  assert.ok(Number.isFinite(coordinates[0].y))
  assert.ok(Number.isFinite(coordinates[1].y))
})

test('el índice activo nunca se sale de la serie', () => {
  assert.equal(clampIndex(-3, 5), 0)
  assert.equal(clampIndex(99, 5), 4)
  assert.equal(clampIndex(2, 5), 2)
  assert.equal(clampIndex(0, 0), null)
})

test('el eje se ajusta al recorrido de los datos en vez de aplastarlos contra el cero', () => {
  const { min, max } = getValueDomain([6600, 8000, 9900])

  // Sin forzar el 0: el recorrido real ocupa el alto de la gráfica.
  assert.ok(min > 6000 && min < 6600)
  assert.ok(max > 9900 && max < 10500)
})

test('una serie que cruza el cero lo conserva dentro del eje', () => {
  const { min, max } = getValueDomain([-500, 200])

  assert.ok(min < 0)
  assert.ok(max > 0)
})

test('las gráficas de barras sí pueden pedir que el cero entre en el eje', () => {
  assert.equal(getValueDomain([300, 900], { includeZero: true }).min, 0)
})

test('una serie completamente plana no colapsa el eje en una línea', () => {
  const { min, max } = getValueDomain([500, 500, 500])

  assert.ok(min < 500)
  assert.ok(max > 500)
})

test('una serie plana en cero tampoco colapsa', () => {
  const { min, max } = getValueDomain([0, 0])

  assert.ok(max > min)
})

test('sin datos se devuelve un rango utilizable en vez de infinitos', () => {
  assert.deepEqual(getValueDomain([]), { min: 0, max: 1 })
  assert.deepEqual(getValueDomain([null, undefined]), { min: 0, max: 1 })
})
