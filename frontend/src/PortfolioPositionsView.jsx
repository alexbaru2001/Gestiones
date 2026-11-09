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
 * Cartera construida desde el archivo de posiciones que mantiene el usuario.
 *
 * Sustituye a la vista que se montaba desde los tres archivos de bróker: aquella solo conocía la
 * última foto, así que la evolución salía de las fotos sueltas que hubiera guardadas y el coste
 * venía de Finanzas. Aquí cada mes es un dato del archivo, de modo que valor, coste y reparto se
 * pueden seguir mes a mes y la rentabilidad sale de lo que de verdad se pagó.
 *
 * Exporta tres bloques, pensados para montarse por separado en la pantalla de Cartera:
 * `PortfolioValueBoard` (evolución + desglose del mes) y `PortfolioPositionsTable` (el listado).
 */

const BROKER_COLORS = { 'Trade Republic': '#1d3f9a', MyInvestor: '#7a9cc6', Degiro: '#0b6b63' }
const FALLBACK_COLORS = ['#1d3f9a', '#0b6b63', '#b45309', '#7a9cc6']

function formatMoney(value) {
  if (!Number.isFinite(Number(value))) return '—'
  return `${new Intl.NumberFormat('es-ES', { minimumFractionDigits: 2, maximumFractionDigits: 2, useGrouping: 'always' })
    .format(Number(value))
    .replace('-', '\u2212')} €`
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
  return `${Number(value).toFixed(2).replace('.', ',').replace('-', '\u2212')} %`
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
const ALLOCATION_COLORS = ['#1d3f9a', '#0b6b63', '#b45309', '#7a9cc6', '#8a93a3', '#14213d', '#b3261e', '#bfd9d5']
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

// Las mismas dimensiones que ofrecía la asignación anterior, sobre los campos que el backend
// añade a cada posición desde el catálogo de activos.
const ALLOCATION_DIMENSIONS = [
  { id: 'Nombre', label: 'Valores' },
  { id: 'region', label: 'Continente' },
  { id: 'sector', label: 'Sector' },
  { id: 'foco', label: 'Foco' },
  { id: 'Broker', label: 'Bróker' },
  { id: 'tipo', label: 'Tipo' },
]

const ASSET_TYPE_LABELS = { stock: 'Acciones', fund: 'Fondos', crypto: 'Cripto', cash: 'Liquidez' }

function groupLabel(position, dimension) {
  const raw = position[dimension]
  if (dimension === 'tipo') return ASSET_TYPE_LABELS[raw] ?? 'Sin clasificar'
  return raw || position.ISIN || 'Sin clasificar'
}

/**
 * Reparto de la cartera en un mes concreto, al lado de la gráfica.
 *
 * El mes no se elige: es aquel sobre el que está el ratón en la serie temporal. Así se ve cómo se
 * recompone la cartera a lo largo del recorrido en vez de solo su foto final.
 */
function MonthAllocation({ dimension, isLive, items, month, onDimensionChange, total }) {
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
      <div className="portfolio-chart-tabs preview-allocation-tabs" role="tablist" aria-label="Vista de distribución de cartera">
        {ALLOCATION_DIMENSIONS.map((item) => (
          <button
            aria-selected={dimension === item.id}
            className={dimension === item.id ? 'active' : ''}
            key={item.id}
            onClick={() => {
              onDimensionChange(item.id)
              setShowAll(false)
            }}
            role="tab"
            type="button"
          >
            {item.label}
          </button>
        ))}
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
              <span className="allocation-weight">{formatPercent(item.weight)}</span>
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

/** Series de valor y coste por bróker, que alimentan tanto la gráfica como el desglose. */
function useSeries(positions, months) {
  const sumBy = useMemo(() => {
    const totals = new Map()
    for (const position of positions) {
      for (const field of ['valor', 'coste']) {
        const amount = Number(position[field])
        if (!Number.isFinite(amount)) continue
        for (const broker of [position.Broker, '*']) {
          const mapKey = `${field}|${broker}|${position.Mes}`
          totals.set(mapKey, (totals.get(mapKey) ?? 0) + amount)
        }
      }
    }
    return totals
  }, [positions])

  return useMemo(() => {
    const serieOf = (field, broker) => months.map((month) => sumBy.get(`${field}|${broker}|${month}`) ?? 0)
    // Rentabilidad: lo que ha subido sobre lo que costó. Sin coste no hay porcentaje que calcular.
    const returnOf = (broker) => {
      const value = serieOf('valor', broker)
      const cost = serieOf('coste', broker)
      return value.map((amount, index) => (cost[index] ? ((amount - cost[index]) / cost[index]) * 100 : 0))
    }
    return { returnOf, serieOf }
  }, [months, sumBy])
}

/**
 * Evolución de la cartera y reparto del mes bajo el cursor, en una sola fila.
 *
 * Reemplaza a las tarjetas de Evolución y Asignación anteriores. Van juntas porque comparten el
 * mes: mover el ratón por la gráfica recompone el desglose, que era la pregunta que antes obligaba
 * a cambiar de foto guardada.
 */
export function PortfolioValueBoard({ data }) {
  const [scope, setScope] = useState('global')
  const [mode, setMode] = useState('absoluto')
  const [dimension, setDimension] = useState('Nombre')
  const [hoverIndex, setHoverIndex] = useState(null)

  const positions = data?.posiciones ?? []
  const months = data?.meses ?? []
  const { returnOf, serieOf } = useSeries(positions, months)

  const brokers = useMemo(() => [...new Set(positions.map((item) => item.Broker))].sort(), [positions])
  const colorOf = (broker, index) => BROKER_COLORS[broker] ?? FALLBACK_COLORS[index % FALLBACK_COLORS.length]

  const totals = serieOf('valor', '*')
  const costs = serieOf('coste', '*')

  const series = useMemo(() => {
    if (scope === 'global') {
      return mode === 'rentabilidad'
        ? [{ label: 'Rentabilidad', color: '#1d3f9a', values: returnOf('*') }]
        : [
            { label: 'Valor', color: '#1d3f9a', values: totals },
            // Discontinua, como la serie secundaria del resto de comparativas de la app.
            { label: 'Coste', color: '#8a93a3', values: costs, dashed: true },
          ]
    }
    if (mode === 'rentabilidad') {
      return brokers.map((broker, index) => ({ label: broker, color: colorOf(broker, index), values: returnOf(broker) }))
    }
    return brokers.flatMap((broker, index) => [
      { label: broker, color: colorOf(broker, index), values: serieOf('valor', broker) },
      { label: `${broker} (coste)`, color: colorOf(broker, index), values: serieOf('coste', broker), dashed: true },
    ])
  }, [brokers, costs, mode, returnOf, scope, serieOf, totals])

  // Sin el ratón encima, el desglose enseña el último mes: es la foto que se espera ver al llegar.
  const allocationIndex = hoverIndex === null ? months.length - 1 : hoverIndex
  const allocationMonth = months[allocationIndex]
  const allocation = useMemo(() => {
    const groups = new Map()
    for (const position of positions) {
      if (position.Mes !== allocationMonth) continue
      const amount = Number(position.valor)
      if (!Number.isFinite(amount) || amount <= 0) continue
      const label = groupLabel(position, dimension)
      groups.set(label, (groups.get(label) ?? 0) + amount)
    }
    const total = [...groups.values()].reduce((sum, amount) => sum + amount, 0)
    const items = [...groups]
      .map(([label, value]) => ({ label, value, weight: total ? (value / total) * 100 : 0 }))
      .sort((left, right) => right.value - left.value)
    return { items, total }
  }, [allocationMonth, dimension, positions])

  if (!months.length) {
    return (
      <section className="portfolio-card">
        <h3>Evolución</h3>
        <p className="muted-text">
          Aún no hay archivo de posiciones cargado. Súbelo desde «Cargar posiciones» para ver la cartera.
        </p>
      </section>
    )
  }

  return (
    <section className="portfolio-card portfolio-value-board">
      <div className="portfolio-chart-header">
        <div>
          <h3>Evolución</h3>
          <span className="muted-text">
            {months.length} meses · {new Set(positions.map((item) => item.ISIN)).size} activos
          </span>
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
      </div>

      <div className="chart-card-wide">
        <ValueChart
          format={mode === 'rentabilidad' ? 'percent' : 'money'}
          label={mode === 'rentabilidad' ? 'Rentabilidad de la cartera' : 'Valor y coste de la cartera'}
          months={months}
          onActiveIndex={setHoverIndex}
          readoutValues={mode === 'rentabilidad' ? returnOf('*') : totals}
          series={series}
        />
        <MonthAllocation
          dimension={dimension}
          isLive={hoverIndex !== null}
          items={allocation.items}
          month={allocationMonth}
          onDimensionChange={setDimension}
          total={allocation.total}
        />
      </div>
    </section>
  )
}

const POSITION_COLUMNS = [
  { key: 'label', label: 'Activo' },
  { key: 'broker', label: 'Bróker' },
  { key: 'quantity', label: 'Cantidad', numeric: true },
  { key: 'cost', label: 'Coste', numeric: true },
  { key: 'value', label: 'Valor', numeric: true },
  { key: 'gain', label: 'P/L', numeric: true },
  { key: 'weight', label: 'Peso', numeric: true },
  { key: 'fees', label: 'Comisiones', numeric: true },
]

const ORIGIN_LABELS = { archivo: 'bróker', cambio: 'bróker + FX', mercado: 'mercado' }

/**
 * Listado de posiciones del último mes del archivo.
 *
 * Frente al anterior gana coste y plusvalía por posición, que antes no se podían dar: el coste solo
 * se conocía del conjunto, desde Finanzas. También dice de dónde sale cada valoración, porque el
 * importe del bróker es exacto y el de mercado una estimación.
 */
export function PortfolioPositionsTable({ action = null, data, onSelect, selectedIsin }) {
  const [sortKey, setSortKey] = useState('value')
  const [sortDirection, setSortDirection] = useState(-1)

  const positions = data?.posiciones ?? []
  const months = data?.meses ?? []
  const lastMonth = months.at(-1)

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

  const rows = useMemo(() => {
    const current = positions.filter((item) => item.Mes === lastMonth && Number(item.valor) > 0)
    const total = current.reduce((sum, item) => sum + Number(item.valor), 0)
    return current.map((position) => {
      const value = Number(position.valor)
      const cost = Number.isFinite(Number(position.coste)) ? Number(position.coste) : null
      return {
        broker: position.Broker,
        cost,
        // Sin coste no hay plusvalía; un 0 se leería como "ni gana ni pierde", que es otra cosa.
        gain: cost === null ? null : value - cost,
        fees: feesByPosition.get(`${position.Broker}-${position.ISIN}`) ?? null,
        id: `${position.Broker}-${position.ISIN}`,
        isin: position.ISIN,
        label: position.Nombre,
        origin: position.origen_valor,
        quantity: Number(position.Cantidad),
        value,
        weight: total ? (value / total) * 100 : 0,
      }
    })
  }, [feesByPosition, lastMonth, positions])

  const sortedRows = useMemo(() => {
    const column = POSITION_COLUMNS.find((item) => item.key === sortKey)
    return [...rows].sort((left, right) => {
      const a = left[sortKey]
      const b = right[sortKey]
      if (column?.numeric) {
        // Lo que no tiene dato se va al final en cualquier sentido: no es ni el mayor ni el menor.
        if (a === null) return 1
        if (b === null) return -1
        return (a - b) * sortDirection
      }
      return String(a ?? '').localeCompare(String(b ?? ''), 'es') * sortDirection
    })
  }, [rows, sortDirection, sortKey])

  const total = rows.reduce((sum, row) => sum + row.value, 0)
  const totalCost = rows.reduce((sum, row) => sum + (row.cost ?? 0), 0)

  if (!rows.length) {
    return (
      <section className="portfolio-card">
        <h3>Posiciones</h3>
        <p className="muted-text">Aún no hay posiciones cargadas.</p>
      </section>
    )
  }

  return (
    <section className="portfolio-card">
      <div className="portfolio-chart-header">
        <div>
          <h3>Posiciones</h3>
          <span className="muted-text">A cierre de {formatMonth(lastMonth)}</span>
        </div>
        <div className="positions-header-actions">
          <span className="muted-text">Pulsa una fila para ver el detalle</span>
          {action}
        </div>
      </div>
      <div className="table-wrap compact-table-wrap">
        <table className="positions-table">
          <thead>
            <tr>
              {POSITION_COLUMNS.map((column) => (
                <th className={column.numeric ? 'num' : undefined} key={column.key}>
                  <button
                    className={sortKey === column.key ? 'sortable is-active' : 'sortable'}
                    onClick={() => {
                      setSortDirection((current) => (sortKey === column.key ? -current : column.numeric ? -1 : 1))
                      setSortKey(column.key)
                    }}
                    type="button"
                  >
                    {column.label}
                    <span aria-hidden="true">{sortKey === column.key ? (sortDirection === 1 ? ' ▴' : ' ▾') : ''}</span>
                  </button>
                </th>
              ))}
              <th>Origen</th>
            </tr>
          </thead>
          <tbody>
            {sortedRows.map((row) => (
              <tr
                className={selectedIsin === row.isin ? 'is-selected' : undefined}
                key={row.id}
                onClick={() => onSelect?.(row.isin)}
              >
                <td className="position-name-cell">
                  <strong>{row.label}</strong>
                  <span className="row-sub">{row.isin}</span>
                </td>
                <td>{row.broker}</td>
                <td className="num">{row.quantity.toLocaleString('es-ES', { maximumFractionDigits: 6 })}</td>
                <td className="num">{row.cost === null ? '—' : formatMoney(row.cost)}</td>
                <td className="num">{formatMoney(row.value)}</td>
                <td className={row.gain === null ? 'num' : `num ${row.gain >= 0 ? 'is-up' : 'is-down'}`}>
                  {row.gain === null ? '—' : `${row.gain >= 0 ? '+' : ''}${formatMoney(row.gain)}`}
                </td>
                <td className="num">
                  <span className="weight-cell">
                    <span className="weight-bar"><i style={{ width: `${row.weight}%` }} /></span>
                    {formatPercent(row.weight)}
                  </span>
                </td>
                <td className="num">{row.fees === null ? '—' : formatMoney(row.fees)}</td>
                <td>
                  {/* Interesa distinguirlo: el importe del bróker es exacto y el de mercado una
                      estimación con medio punto de error en los fondos. */}
                  <span className={`origen-tag origen-${row.origin}`}>{ORIGIN_LABELS[row.origin] ?? row.origin}</span>
                </td>
              </tr>
            ))}
          </tbody>
          <tfoot>
            <tr>
              <td colSpan={3}>Total a {formatMonth(lastMonth)}</td>
              <td className="num">{formatMoney(totalCost)}</td>
              <td className="num"><strong>{formatMoney(total)}</strong></td>
              <td className={`num ${total - totalCost >= 0 ? 'is-up' : 'is-down'}`}>
                {`${total - totalCost >= 0 ? '+' : ''}${formatMoney(total - totalCost)}`}
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
