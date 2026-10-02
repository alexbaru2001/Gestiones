// Geometría común de todas las gráficas: escalas, trazo y la aritmética del cruce con el ratón.
// Vive aparte de los componentes para poder probarla con `npm test` y para que Finanzas, Cartera e
// Invertir dibujen exactamente igual en vez de reimplementar cada una lo suyo.

// Number(null) y Number('') valen 0: sin esta guarda, un hueco de datos se dibujaría como un cero
// real y hundiría la línea hasta el suelo.
export function isNumber(value) {
  return value !== null && value !== undefined && value !== '' && Number.isFinite(Number(value))
}

export function getSvgCoordinates(values, min, max, options = {}) {
  const width = options.width ?? 640
  const height = options.height ?? 220
  const padding = options.padding ?? 18
  const paddingLeft = options.paddingLeft ?? padding
  const paddingRight = options.paddingRight ?? padding
  const paddingTop = options.paddingTop ?? padding
  const paddingBottom = options.paddingBottom ?? padding
  const range = max - min || 1
  return values.map((value, index) => ({
    x: paddingLeft + (index / Math.max(1, values.length - 1)) * (width - paddingLeft - paddingRight),
    y: height - paddingBottom - ((Number(value) - min) / range) * (height - paddingTop - paddingBottom),
  }))
}

// Trazo recto entre puntos reales. Sustituye a la curva Catmull-Rom que redondeaba los vértices:
// un mes no transiciona suavemente al siguiente, y la curva sugería valores intermedios que no existen.
export function linePath(points) {
  if (!points?.length) return ''
  return points
    .map((point, index) => `${index === 0 ? 'M' : 'L'} ${point.x.toFixed(1)},${point.y.toFixed(1)}`)
    .join(' ')
}

export function polylinePoints(points) {
  if (!points?.length) return ''
  return points.map((point) => `${point.x.toFixed(1)},${point.y.toFixed(1)}`).join(' ')
}

// El cruce se engancha al dato más cercano en horizontal: la vertical del ratón no cae entre dos
// meses, sino sobre el que se está señalando.
export function nearestIndexFromX(x, xs) {
  if (!xs?.length) return null
  let best = 0
  let bestDistance = Infinity
  for (let index = 0; index < xs.length; index += 1) {
    const distance = Math.abs(Number(xs[index]) - x)
    if (distance < bestDistance) {
      bestDistance = distance
      best = index
    }
  }
  return best
}

// Convierte la posición del puntero en el sistema de coordenadas del SVG. Hace falta porque el SVG
// se estira al ancho del contenedor: usar el X de pantalla directamente desalinea la cruz.
export function pointerToSvgX(event, element, viewBoxWidth) {
  if (!element) return null
  const rect = element.getBoundingClientRect()
  if (!rect.width) return null
  return ((event.clientX - rect.left) / rect.width) * viewBoxWidth
}

// Variación entre el primer valor del periodo y el punto señalado, que es lo que responde a
// "¿cuánto llevo desde que empezó el periodo?".
export function getSeriesChange(values, index) {
  if (!Array.isArray(values) || !values.length) return null
  const target = index ?? values.length - 1
  const first = values[0]
  const current = values[target]
  if (!isNumber(first) || !isNumber(current)) return null
  const delta = Number(current) - Number(first)
  // Con un primer valor de 0 el porcentaje sería infinito: se da el importe y se omite el %.
  const percent = Number(first) === 0 ? null : (delta / Math.abs(Number(first))) * 100
  return { delta, percent, from: Number(first), to: Number(current) }
}

export function clampIndex(index, length) {
  if (!length) return null
  return Math.max(0, Math.min(length - 1, index))
}

/**
 * Rango vertical de una gráfica de nivel (saldos, precios, valor de cartera).
 *
 * Forzar el 0 en el eje aplasta la curva: una serie que va de 6.600 € a 9.900 € ocupa un tercio
 * del alto y parece una recta. El criterio habitual en gráficas financieras es ajustar el eje al
 * recorrido real de los datos con un margen, y meter el 0 solo cuando la serie lo cruza, porque
 * ahí el cambio de signo sí es información. Las gráficas de barras son la excepción y siempre
 * arrancan en 0: la longitud de la barra representa la magnitud y recortarla engaña.
 */
export function getValueDomain(values, options = {}) {
  const padding = options.padding ?? 0.08
  const numbers = (values ?? []).filter(isNumber).map(Number)
  if (!numbers.length) return { min: 0, max: 1 }

  let min = Math.min(...numbers)
  let max = Math.max(...numbers)

  // Si la serie cambia de signo, el cero es una referencia real y se queda dentro.
  if (min < 0 && max > 0) {
    // ya está contenido
  } else if (options.includeZero) {
    min = Math.min(min, 0)
    max = Math.max(max, 0)
  }

  if (min === max) {
    // Serie plana de verdad: se abre un margen alrededor para que no quede pegada a un borde.
    const margin = Math.abs(min) * padding || 1
    return { min: min - margin, max: max + margin }
  }

  const margin = (max - min) * padding
  // El margen no debe cruzar el cero cuando el cero es la base de las barras: una barra que
  // arranca por debajo de su propia línea de cero no representa nada.
  const floor = options.includeZero && Math.min(...numbers) >= 0 ? 0 : min - margin
  const ceiling = options.includeZero && Math.max(...numbers) <= 0 ? 0 : max + margin
  return { min: floor, max: ceiling }
}

export const CHART_ASPECT = 4 / 3

/**
 * Alto de una gráfica a partir de su ancho.
 *
 * Las gráficas recibían un alto fijo y se estiraban a lo ancho, así que acababan con proporciones
 * de hasta 7:1 y cualquier subida se leía como una recta. El ojo juzga bien una pendiente cuando
 * ronda los 45°, y para eso la forma tiene que depender del ancho, no del hueco disponible.
 * Los topes evitan que en una pantalla muy ancha una tarjeta se convierta en una columna gigante.
 */
export function getChartHeight(width, aspect = CHART_ASPECT, min = 200, max = 520) {
  if (!isNumber(width) || Number(width) <= 0 || !isNumber(aspect) || Number(aspect) <= 0) return min
  return Math.round(Math.max(min, Math.min(max, Number(width) / Number(aspect))))
}

/**
 * Marcas del eje vertical en valores redondos.
 *
 * Antes solo se escribían el mínimo y el máximo, y las líneas de rejilla caían en fracciones fijas
 * (un cuarto, la mitad...) que no correspondían a ninguna cifra legible. Con pasos de 1, 2, 2,5 o 5
 * por década, las marcas caen en números que se leen de un vistazo.
 */
export function getNiceTicks(min, max, count = 5) {
  if (!isNumber(min) || !isNumber(max) || count < 2) return []
  if (min === max) return [Number(min)]

  const low = Number(min)
  const high = Number(max)
  const rawStep = (high - low) / (count - 1)
  const magnitude = 10 ** Math.floor(Math.log10(Math.abs(rawStep)))

  // Se prueban los pasos legibles de la década y se elige el que deja un número de marcas más
  // cercano al pedido. Redondear siempre hacia arriba (3,5 → 5) dejaba ejes con solo dos marcas.
  let best = null
  for (const factor of [1, 2, 2.5, 5, 10]) {
    const step = factor * magnitude
    const ticks = buildTicks(low, high, step)
    if (ticks.length < 2) continue
    const distance = Math.abs(ticks.length - count)
    if (!best || distance < best.distance) best = { distance, ticks }
  }
  return best ? best.ticks : []
}

function buildTicks(min, max, step) {
  const ticks = []
  const first = Math.ceil(min / step) * step
  for (let value = first; value <= max + step * 0.001; value += step) {
    // El redondeo evita los 0.30000000000000004 que ensucian las etiquetas.
    ticks.push(Number(value.toFixed(10)))
  }
  return ticks
}

/**
 * Qué posiciones del eje horizontal llevan etiqueta.
 *
 * Se escribían solo la primera y la última, así que no se sabía dónde caía el resto del recorrido.
 * Se reparten hasta `max` etiquetas, siempre con los extremos incluidos.
 */
export function pickLabelIndices(count, max = 5) {
  if (!count || count < 1) return []
  if (count <= max) return Array.from({ length: count }, (_, index) => index)
  const step = (count - 1) / (max - 1)
  const indices = new Set()
  for (let position = 0; position < max; position += 1) {
    indices.add(Math.round(position * step))
  }
  return [...indices].sort((left, right) => left - right)
}
