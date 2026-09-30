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
