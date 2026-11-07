import { useEffect, useMemo, useRef, useState } from 'react'

import { ChartCrosshair, ChartReadout } from './ChartParts'
import {
  CHART_ASPECT,
  getChartHeight,
  getNiceTicks,
  getSvgCoordinates,
  getValueDomain,
  linePath,
  pickLabelIndices,
} from './chartGeometry'
import { useChartCrosshair } from './useChartCrosshair'

/**
 * Vista de la cartera construida desde el archivo de posiciones del usuario.
 *
 * Va aparte y al final de la pantalla a propósito: convive con la vista que ya existe (la que se
 * monta desde los tres archivos de bróker) hasta comprobar que da lo mismo. Cuando lo haga, será
 * esta la que se quede.
 */

const BROKER_COLORS = { 'Trade Republic': '#b08d57', MyInvestor: '#6a9ab5', Degiro: '#7fa06f' }
const FALLBACK_COLORS = ['#1f4d3d', '#7a2e2e', '#1f5c6b', '#c0804a']

function formatMoney(value) {
  if (!Number.isFinite(Number(value))) return '—'
  return `${new Intl.NumberFormat('es-ES', { minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(Number(value))} €`
}

function formatCompact(value) {
  if (!Number.isFinite(Number(value))) return ''
  const number = Number(value)
  return Math.abs(number) >= 1000
    ? `${new Intl.NumberFormat('es-ES', { maximumFractionDigits: 1 }).format(number / 1000)}k €`
    : `${new Intl.NumberFormat('es-ES', { maximumFractionDigits: 0 }).format(number)} €`
}

function formatMonth(month) {
  if (!month) return ''
  const [year, value] = String(month).split('-')
  return `${value}/${String(year).slice(2)}`
}

function useWidth(fallback) {
  const ref = useRef(null)
  const [width, setWidth] = useState(fallback)
  useEffect(() => {
    const element = ref.current
    if (!element) return undefined
    const measure = () => setWidth(Math.round(element.getBoundingClientRect().width || fallback))
    measure()
    const observer = new ResizeObserver(measure)
    observer.observe(element)
    return () => observer.disconnect()
  }, [fallback])
  return [ref, width]
}

/** Serie temporal con el estilo de la casa: trazo recto, recuadro, cruce y valor en cabecera. */
function formatPercent(value) {
  if (!Number.isFinite(Number(value))) return '—'
  return `${Number(value) >= 0 ? '' : ''}${Number(value).toFixed(2).replace('.', ',')} %`
}

function ValueChart({ format = 'money', label, months, onActiveIndex, series, readoutValues }) {
  const formatValue = format === 'percent' ? formatPercent : formatMoney
  const formatTick = format === 'percent' ? (value) => `${Math.round(value)} %` : formatCompact
  const [containerRef, width] = useWidth(520)
  // Tope más alto que el del resto de tarjetas: aquí la gráfica comparte fila con la asignación,
  // así que le queda ancho para crecer sin pasarse y mantener la proporción casi cuadrada.
  const height = getChartHeight(width, CHART_ASPECT, 260, 700)
  // Los márgenes crecen con la letra de los ejes: con 12px, los 58 de antes cortaban "10,5k €".
  const padding = { left: 70, right: 18, top: 18, bottom: 36 }
  const plotBottom = height - padding.bottom

  const all = series.flatMap((item) => item.values)
  const { min, max } = getValueDomain(all)
  const options = {
    width,
    height,
    paddingLeft: padding.left,
    paddingRight: padding.right,
    paddingTop: padding.top,
    paddingBottom: padding.bottom,
  }
  const coordinates = series.map((item) => getSvgCoordinates(item.values, min, max, options))
  const lead = coordinates[0] ?? []
  const plot = { left: padding.left, right: width - padding.right, top: padding.top, bottom: plotBottom }
  const yTicks = getNiceTicks(min, max, height > 260 ? 5 : 4)
  const labelIndices = pickLabelIndices(months.length, width > 520 ? 5 : 4)

  const { activeIndex, handlers, isHovering, svgRef } = useChartCrosshair(lead.map((point) => point.x), width)

  // La asignación de la derecha sigue al ratón sobre el eje X. Sin ratón encima no hay mes que
  // destacar y se deja decidir a quien escuche: así vuelve solo al último mes.
  useEffect(() => {
    onActiveIndex?.(isHovering ? activeIndex : null)
  }, [activeIndex, isHovering, onActiveIndex])

  return (
    <div ref={containerRef}>
      <ChartReadout
        activeIndex={isHovering ? activeIndex : null}
        formatDelta={(value) => `${value >= 0 ? '+' : ''}${formatValue(value)}`}
        formatValue={formatValue}
        label={label}
        labels={months.map(formatMonth)}
        values={readoutValues ?? series[0]?.values ?? []}
      />
      <svg
        aria-label={label}
        className="mini-trend-svg preview-chart-svg"
        height={height}
        ref={svgRef}
        viewBox={`0 0 ${width} ${height}`}
      >
        {yTicks.map((tick) => {
          const y = getSvgCoordinates([tick], min, max, options)[0].y
          return (
            <g key={tick}>
              <line className="mini-grid-line" x1={padding.left} x2={width - padding.right} y1={y} y2={y} />
              <text className="mini-axis-label" textAnchor="end" x={padding.left - 10} y={y + 4}>
                {formatTick(tick)}
              </text>
            </g>
          )
        })}
        {series.map((item, index) => (
          <path
            d={linePath(coordinates[index])}
            fill="none"
            key={item.label}
            stroke={item.color}
            strokeDasharray={item.dashed ? '6 4' : undefined}
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth="2.6"
          />
        ))}
        {labelIndices.map((index) => (
          <text
            className="mini-axis-month"
            key={months[index] ?? index}
            textAnchor={index === 0 ? 'start' : index === months.length - 1 ? 'end' : 'middle'}
            x={lead[index]?.x}
            y={height - 8}
          >
            {formatMonth(months[index])}
          </text>
        ))}
        <rect
          className="chart-frame"
          height={plotBottom - padding.top}
          width={width - padding.left - padding.right}
          x={padding.left}
          y={padding.top}
        />
        <ChartCrosshair
          activeIndex={activeIndex}
          baselineY={lead[0]?.y ?? null}
          handlers={handlers}
          isHovering={isHovering}
          label={`Recorrer ${label}`}
          plot={plot}
          pointColor={series[0]?.color}
          points={lead}
        />
      </svg>
      {series.length > 1 ? (
        <div className="portfolio-line-legend">
          {series.map((item) => (
            <span className={item.dashed ? 'is-dashed' : undefined} key={item.label} style={{ '--legend-color': item.color }}>
              {item.label}
              {activeIndex !== null ? ` ${formatValue(item.values[activeIndex])}` : ''}
            </span>
          ))}
        </div>
      ) : null}
    </div>
  )
}

// Misma paleta que la asignación de arriba, para que un activo no cambie de color entre vistas.
const ALLOCATION_COLORS = ['#B08D57', '#7C93A6', '#8B7BA0', '#5F8F7C', '#A67C6D', '#C9A876', '#6B7A8C', '#9A6B4F']
const ALLOCATION_PREVIEW_SIZE = 7

function allocationColor(index) {
  return ALLOCATION_COLORS[index % ALLOCATION_COLORS.length]
}

function donutBackground(items) {
  if (!items.length) return 'rgba(255, 255, 255, 0.08)'
  let start = 0
  const segments = items.map((item, index) => {
    const end = start + item.weight
    const segment = `${allocationColor(index)} ${start}% ${end}%`
    start = end
    return segment
  })
  return `conic-gradient(${segments.join(', ')})`
}

/**
 * Reparto de la cartera en un mes concreto, al lado de la gráfica.
 *
 * El mes no se elige: es aquel sobre el que está el ratón en la serie temporal. Así se ve cómo se
 * recompone la cartera a lo largo del recorrido en vez de solo su foto final.
 */
function MonthAllocation({ isLive, items, month, total }) {
  const [showAll, setShowAll] = useState(false)
  const hidden = Math.max(items.length - ALLOCATION_PREVIEW_SIZE, 0)
  const visible = showAll ? items : items.slice(0, ALLOCATION_PREVIEW_SIZE)

  return (
    <aside className="preview-allocation">
      <div className="preview-allocation-head">
        <h4>Asignación</h4>
        <span className={isLive ? 'preview-allocation-month is-live' : 'preview-allocation-month'}>
          {formatMonth(month)}
        </span>
      </div>
      <div className="portfolio-donut" aria-label={`Reparto de la cartera en ${formatMonth(month)}`} style={{ background: donutBackground(items) }}>
        <strong>{formatMoney(total)}</strong>
        <span>{isLive ? 'valor del mes' : 'valor actual'}</span>
      </div>
      <p className="muted-text allocation-hint">Pasa el ratón por la gráfica para ver otro mes.</p>
      <div className="allocation-list preview-allocation-list">
        {visible.map((item, index) => (
          <article key={item.label}>
            <div className="preview-allocation-row">
              <i style={{ background: allocationColor(index) }} />
              <span className="allocation-name">{item.label}</span>
              <span className="allocation-value">{formatMoney(item.value)}</span>
              <span className="allocation-weight">{item.weight.toFixed(2).replace('.', ',')} %</span>
            </div>
            <div className="portfolio-bar">
              <span style={{ background: allocationColor(index), width: `${item.weight}%` }} />
            </div>
          </article>
        ))}
      </div>
      {hidden > 0 ? (
        <button aria-expanded={showAll} className="allocation-more" onClick={() => setShowAll((previous) => !previous)} type="button">
          {showAll ? 'Ver menos' : `Ver más (${hidden})`}
        </button>
      ) : null}
    </aside>
  )
}

export function PortfolioPreview({ data }) {
  const [scope, setScope] = useState('global')
  const [mode, setMode] = useState('absoluto')
  const [hoverIndex, setHoverIndex] = useState(null)
  const positions = data?.posiciones ?? []
  const months = data?.meses ?? []

  const sumBy = useMemo(() => {
    const totals = new Map()
    for (const position of positions) {
      for (const [field, key] of [['valor', 'valor'], ['coste', 'coste']]) {
        const amount = Number(position[field])
        if (!Number.isFinite(amount)) continue
        for (const broker of [position.Broker, '*']) {
          const mapKey = `${key}|${broker}|${position.Mes}`
          totals.set(mapKey, (totals.get(mapKey) ?? 0) + amount)
        }
      }
    }
    return totals
  }, [positions])

  const serieOf = (field, broker) => months.map((month) => sumBy.get(`${field}|${broker}|${month}`) ?? 0)
  // Rentabilidad: lo que ha subido sobre lo que costó. Sin coste no hay porcentaje que calcular.
  const returnOf = (broker) => {
    const value = serieOf('valor', broker)
    const cost = serieOf('coste', broker)
    return value.map((amount, index) => (cost[index] ? ((amount - cost[index]) / cost[index]) * 100 : 0))
  }

  const brokers = useMemo(() => [...new Set(positions.map((p) => p.Broker))].sort(), [positions])
  const colorOf = (broker, index) => BROKER_COLORS[broker] ?? FALLBACK_COLORS[index % FALLBACK_COLORS.length]

  const totals = serieOf('valor', '*')
  const costs = serieOf('coste', '*')

  const series = useMemo(() => {
    if (scope === 'global') {
      return mode === 'rentabilidad'
        ? [{ label: 'Rentabilidad', color: '#b08d57', values: returnOf('*') }]
        : [
            { label: 'Valor', color: '#b08d57', values: totals },
            // Discontinua, como la serie secundaria del resto de comparativas de la app.
            { label: 'Coste', color: '#7e8b83', values: costs, dashed: true },
          ]
    }
    if (mode === 'rentabilidad') {
      return brokers.map((broker, index) => ({ label: broker, color: colorOf(broker, index), values: returnOf(broker) }))
    }
    return brokers.flatMap((broker, index) => [
      { label: broker, color: colorOf(broker, index), values: serieOf('valor', broker) },
      { label: `${broker} (coste)`, color: colorOf(broker, index), values: serieOf('coste', broker), dashed: true },
    ])
  }, [brokers, costs, mode, scope, totals])

  const readoutValues = mode === 'rentabilidad' ? returnOf('*') : totals

  // Las comisiones se cobran cuatro veces al año, así que en un mes concreto casi nunca hay
  // ninguna: lo que informa es cuánto lleva pagado cada posición desde el principio.
  const feesByPosition = useMemo(() => {
    const totalsByKey = new Map()
    for (const position of positions) {
      const fee = Number(position.comision_imputada)
      if (!Number.isFinite(fee) || !fee) continue
      const key = `${position.Broker}-${position.ISIN}`
      totalsByKey.set(key, (totalsByKey.get(key) ?? 0) + fee)
    }
    return totalsByKey
  }, [positions])

  // Sin el ratón encima, la asignación enseña el último mes: es la foto que se espera ver al llegar.
  const allocationIndex = hoverIndex === null ? months.length - 1 : hoverIndex
  const allocationMonth = months[allocationIndex]
  const allocation = useMemo(() => {
    const byName = new Map()
    for (const position of positions) {
      if (position.Mes !== allocationMonth) continue
      const amount = Number(position.valor)
      if (!Number.isFinite(amount) || amount <= 0) continue
      // Se agrupa por nombre: el mismo activo en dos brókeres es una sola apuesta de la cartera.
      const label = position.Nombre || position.ISIN
      byName.set(label, (byName.get(label) ?? 0) + amount)
    }
    const total = [...byName.values()].reduce((sum, amount) => sum + amount, 0)
    const items = [...byName]
      .map(([label, value]) => ({ label, value, weight: total ? (value / total) * 100 : 0 }))
      .sort((left, right) => right.value - left.value)
    return { items, total }
  }, [allocationMonth, positions])

  const lastMonth = months.at(-1)
  const lastPositions = useMemo(
    () =>
      positions
        .filter((p) => p.Mes === lastMonth && Number(p.valor) > 0)
        .sort((left, right) => Number(right.valor) - Number(left.valor)),
    [lastMonth, positions],
  )
  const lastTotal = totals.at(-1) ?? 0

  if (!months.length) return null

  return (
    <section className="portfolio-preview">
      <div className="portfolio-chart-header">
        <div>
          <h3>Vista nueva · desde tu archivo de posiciones</h3>
          <span className="muted-text">
            Convive con la de arriba hasta comprobar que cuadran. {positions.length} posiciones en {months.length} meses.
          </span>
        </div>
      </div>

      <div className="preview-controls">
        <div className="portfolio-chart-tabs" role="tablist" aria-label="Alcance de la gráfica">
          {[{ id: 'global', label: 'Global' }, { id: 'broker', label: 'Por bróker' }].map((item) => (
            <button
              aria-selected={scope === item.id}
              className={scope === item.id ? 'active' : ''}
              key={item.id}
              onClick={() => setScope(item.id)}
              role="tab"
              type="button"
            >
              {item.label}
            </button>
          ))}
        </div>
        <div className="portfolio-chart-tabs" role="tablist" aria-label="Unidad de la gráfica">
          {[{ id: 'absoluto', label: 'Absoluto' }, { id: 'rentabilidad', label: 'Rentabilidad' }].map((item) => (
            <button
              aria-selected={mode === item.id}
              className={mode === item.id ? 'active' : ''}
              key={item.id}
              onClick={() => setMode(item.id)}
              role="tab"
              type="button"
            >
              {item.label}
            </button>
          ))}
        </div>
      </div>

      <article className="chart-card chart-card-wide">
        <ValueChart
          format={mode === 'rentabilidad' ? 'percent' : 'money'}
          label={mode === 'rentabilidad' ? 'Rentabilidad de la cartera' : 'Valor y coste de la cartera'}
          months={months}
          onActiveIndex={setHoverIndex}
          readoutValues={readoutValues}
          series={series}
        />
        <MonthAllocation
          isLive={hoverIndex !== null}
          items={allocation.items}
          month={allocationMonth}
          total={allocation.total}
        />
      </article>

      <div className="table-wrap">
        <table className="positions-table">
          <thead>
            <tr>
              <th>Activo</th>
              <th>Bróker</th>
              <th className="num">Cantidad</th>
              <th className="num">Valor</th>
              <th className="num">Peso</th>
              <th className="num">Comisiones pagadas</th>
              <th>Valor tomado de</th>
            </tr>
          </thead>
          <tbody>
            {lastPositions.map((position) => (
              <tr key={`${position.Broker}-${position.ISIN}`}>
                <td>
                  <strong>{position.Nombre}</strong>
                  <small className="review-name">{position.ISIN}</small>
                </td>
                <td>{position.Broker}</td>
                <td className="num">{Number(position.Cantidad).toLocaleString('es-ES', { maximumFractionDigits: 6 })}</td>
                <td className="num">{formatMoney(position.valor)}</td>
                <td className="num">
                  {lastTotal ? `${((Number(position.valor) / lastTotal) * 100).toFixed(2).replace('.', ',')} %` : '—'}
                </td>
                <td className="num">
                  {feesByPosition.get(`${position.Broker}-${position.ISIN}`)
                    ? formatMoney(feesByPosition.get(`${position.Broker}-${position.ISIN}`))
                    : '—'}
                </td>
                <td>
                  {/* Interesa distinguirlo: el importe del bróker es exacto y el de mercado una
                      estimación con medio punto de error en los fondos. */}
                  <span className={`origen-tag origen-${position.origen_valor}`}>
                    {position.origen_valor === 'archivo'
                      ? 'tu archivo'
                      : position.origen_valor === 'cambio'
                        ? 'tu archivo + cambio'
                        : 'mercado'}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
          <tfoot>
            <tr>
              <td colSpan={3}>Total a {formatMonth(lastMonth)}</td>
              <td className="num">
                <strong>{formatMoney(lastTotal)}</strong>
              </td>
              <td className="num">100 %</td>
              <td className="num" colSpan={2} />
            </tr>
          </tfoot>
        </table>
      </div>
    </section>
  )
}
