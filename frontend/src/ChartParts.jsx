import { getSeriesChange } from './chartGeometry'

/**
 * Capa de interacción común a todas las gráficas: la horizontal punteada del valor de partida, la
 * vertical que sigue al ratón, el punto del cruce con la curva y el rectángulo que captura el
 * puntero. Se monta al final del SVG para que quede por encima del trazo.
 */
export function ChartCrosshair({
  activeIndex,
  baselineY = null,
  handlers,
  isHovering,
  label = 'Recorrer la gráfica',
  plot,
  points = [],
  pointColor = 'currentColor',
}) {
  const active = activeIndex === null ? null : points[activeIndex]
  const last = points.at(-1)

  return (
    <>
      {baselineY === null ? null : (
        <line className="chart-baseline" x1={plot.left} x2={plot.right} y1={baselineY} y2={baselineY} />
      )}
      {last ? <circle className="chart-last-point" cx={last.x} cy={last.y} fill={pointColor} r="3.6" /> : null}
      {isHovering && active ? (
        <>
          <line className="chart-crosshair" x1={active.x} x2={active.x} y1={plot.top} y2={plot.bottom} />
          <circle className="chart-active-point" cx={active.x} cy={active.y} fill={pointColor} r="5" />
        </>
      ) : null}
      <rect
        aria-label={label}
        className="chart-capture"
        height={Math.max(0, plot.bottom - plot.top)}
        role="application"
        tabIndex={0}
        width={Math.max(0, plot.right - plot.left)}
        x={plot.left}
        y={plot.top}
        {...handlers}
      />
    </>
  )
}

/**
 * Cifra de cabecera gobernada por el cruce: enseña el valor del punto señalado y, en reposo, el
 * último de la serie. La variación se mide siempre contra el primer valor del periodo elegido.
 */
export function ChartReadout({
  activeIndex,
  className = '',
  formatDelta,
  formatValue,
  label,
  labels = [],
  periodLabel,
  values = [],
}) {
  const index = activeIndex ?? values.length - 1
  const value = values[index]
  const change = getSeriesChange(values, index)
  const stamp = labels[index] ?? null

  return (
    <div className={`chart-readout ${className}`.trim()}>
      {label ? <span className="chart-readout-label">{label}</span> : null}
      <strong className="chart-readout-value">{formatValue(value)}</strong>
      {change ? (
        <span
          className={`chart-readout-change ${change.delta > 0 ? 'is-up' : change.delta < 0 ? 'is-down' : ''}`.trim()}
        >
          {/* En el primer punto la variación contra sí mismo es siempre 0: decirlo así informa más
              que enseñar un "+0,00 € (0,00 %)" que parece un dato. */}
          {index === 0 ? (
            'inicio del periodo'
          ) : (
            <>
              {formatDelta(change.delta)}
              {change.percent === null ? '' : ` (${change.percent > 0 ? '+' : ''}${change.percent.toFixed(2).replace('.', ',')} %)`}
              {periodLabel ? ` · ${periodLabel}` : ''}
            </>
          )}
        </span>
      ) : null}
      {stamp ? <span className="chart-readout-stamp">{stamp}</span> : null}
    </div>
  )
}

/** Fila de periodos en horizontal sobre la gráfica, en vez de un desplegable. */
export function PeriodTabs({ ariaLabel = 'Periodo', onChange, options, value }) {
  return (
    <div aria-label={ariaLabel} className="period-tabs" role="tablist">
      {options.map((option) => (
        <button
          aria-selected={value === option.value}
          className={value === option.value ? 'active' : ''}
          key={option.value}
          onClick={() => onChange(option.value)}
          role="tab"
          type="button"
        >
          {option.label}
        </button>
      ))}
    </div>
  )
}
