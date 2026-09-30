import { useCallback, useMemo, useRef, useState } from 'react'

import { clampIndex, nearestIndexFromX, pointerToSvgX } from './chartGeometry'

/**
 * Cruce vertical que sigue al puntero por toda el área de trazado.
 *
 * Antes cada gráfica ponía un círculo invisible sobre cada dato y solo reaccionaba si acertabas a
 * pasar justo por encima. Aquí se escucha el área entera y se engancha al dato más cercano, que es
 * como se comportan las gráficas de las aplicaciones de bolsa: da igual la altura del ratón, manda
 * su posición horizontal.
 *
 * `activeIndex` nunca es nulo mientras haya serie: cuando el puntero sale, cae al último dato, para
 * que la cifra de cabecera enseñe siempre el valor más reciente en reposo.
 */
export function useChartCrosshair(xs, viewBoxWidth) {
  const svgRef = useRef(null)
  const [hoverIndex, setHoverIndex] = useState(null)
  const length = xs?.length ?? 0

  const updateFromEvent = useCallback(
    (event) => {
      if (!length) return
      const x = pointerToSvgX(event, svgRef.current, viewBoxWidth)
      if (x === null) return
      setHoverIndex(nearestIndexFromX(x, xs))
    },
    [length, viewBoxWidth, xs],
  )

  const onKeyDown = useCallback(
    (event) => {
      if (!length) return
      const step = event.key === 'ArrowRight' ? 1 : event.key === 'ArrowLeft' ? -1 : 0
      if (step === 0) {
        if (event.key === 'Home') setHoverIndex(0)
        else if (event.key === 'End') setHoverIndex(length - 1)
        else return
      } else {
        setHoverIndex((previous) => clampIndex((previous ?? length - 1) + step, length))
      }
      event.preventDefault()
    },
    [length],
  )

  const handlers = useMemo(
    () => ({
      onPointerMove: updateFromEvent,
      onPointerDown: updateFromEvent,
      onPointerLeave: () => setHoverIndex(null),
      onBlur: () => setHoverIndex(null),
      onKeyDown,
    }),
    [onKeyDown, updateFromEvent],
  )

  const activeIndex = hoverIndex ?? (length ? length - 1 : null)

  return { activeIndex, handlers, isHovering: hoverIndex !== null, svgRef }
}
