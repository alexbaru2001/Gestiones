import assert from 'node:assert/strict'
import test from 'node:test'

import {
  clampIndex,
  getSeriesChange,
  getChartHeight,
  getNiceTicks,
  pickLabelIndices,
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

test('el alto sale del ancho para mantener la proporción en cualquier pantalla', () => {
  assert.equal(getChartHeight(450, 4 / 3), 338)
  assert.equal(getChartHeight(700, 4 / 3), 520) // topado: 525 pasaría del máximo
})

test('la proporción se respeta mientras quepa entre los topes', () => {
  const alto = getChartHeight(600, 4 / 3)

  assert.ok(Math.abs(600 / alto - 4 / 3) < 0.05)
})

test('una gráfica muy estrecha no se queda sin alto utilizable', () => {
  assert.equal(getChartHeight(120, 4 / 3), 200)
})

test('un ancho aún sin medir devuelve el mínimo en vez de NaN', () => {
  assert.equal(getChartHeight(0), 200)
  assert.equal(getChartHeight(null), 200)
  assert.equal(getChartHeight(undefined), 200)
})

test('una proporción inválida no rompe el cálculo', () => {
  assert.equal(getChartHeight(600, 0), 200)
})

test('los topes se pueden ajustar por gráfica', () => {
  assert.equal(getChartHeight(450, 4 / 3, 120, 260), 260)
})

test('las marcas del eje caen en valores redondos, no en fracciones del rango', () => {
  const ticks = getNiceTicks(0, 1000, 5)

  assert.deepEqual(ticks, [0, 250, 500, 750, 1000])
})

test('las marcas se adaptan a la escala de los datos', () => {
  assert.deepEqual(getNiceTicks(0, 10, 5), [0, 2.5, 5, 7.5, 10])
  assert.deepEqual(getNiceTicks(0, 4, 5), [0, 1, 2, 3, 4])
})

test('un rango con negativos reparte marcas a ambos lados del cero', () => {
  const ticks = getNiceTicks(-300, 500, 5)

  assert.ok(ticks.some((tick) => tick < 0))
  assert.ok(ticks.some((tick) => tick > 0))
  assert.ok(ticks.includes(0))
})

test('las marcas nunca se salen del rango de la gráfica', () => {
  const ticks = getNiceTicks(8773, 21745, 5)

  assert.ok(ticks[0] >= 8773)
  assert.ok(ticks.at(-1) <= 21745)
})

test('las etiquetas decimales no arrastran basura de coma flotante', () => {
  for (const tick of getNiceTicks(0, 0.5, 5)) {
    assert.ok(String(tick).length < 8, `etiqueta ilegible: ${tick}`)
  }
})

test('una serie plana no genera marcas duplicadas', () => {
  assert.deepEqual(getNiceTicks(500, 500), [500])
})

test('con pocos meses se etiquetan todos', () => {
  assert.deepEqual(pickLabelIndices(4, 5), [0, 1, 2, 3])
})

test('con muchos meses se reparten las etiquetas incluyendo los extremos', () => {
  const indices = pickLabelIndices(16, 5)

  assert.equal(indices[0], 0)
  assert.equal(indices.at(-1), 15)
  assert.ok(indices.length <= 5)
})

test('sin datos no se etiqueta nada', () => {
  assert.deepEqual(pickLabelIndices(0), [])
})

test('un rango irregular no se queda con dos marcas sueltas', () => {
  // Caso real de la tarjeta de Vacaciones, que daba solo [0, 500].
  assert.ok(getNiceTicks(-402, 666, 4).length >= 3)
})

test('el paso elegido es el que más se acerca al número de marcas pedido', () => {
  assert.equal(getNiceTicks(0, 1000, 5).length, 5)
  assert.ok(Math.abs(getNiceTicks(310, 1752, 4).length - 4) <= 1)
})
