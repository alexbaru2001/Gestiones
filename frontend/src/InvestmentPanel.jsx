import { useMemo, useState } from 'react'
import { Search } from 'lucide-react'
import { requestJson } from './api'

const metricGroups = [
  ['rpd_ttm', 'RPD TTM', '%'],
  ['rpd_forward', 'RPD forward', '%'],
  ['dgr5', 'DGR 5 años', '%'],
  ['dgr10', 'DGR 10 años', '%'],
  ['payout', 'Payout', '%'],
  ['per_ttm', 'PER', ''],
  ['de_ratio', 'Deuda/Patrimonio', 'x'],
  ['roe', 'ROE', '%'],
  ['ev_ebitda', 'EV/EBITDA', 'x'],
  ['fcf_yield', 'FCF yield', '%'],
  ['streak_years', 'Racha pagos', 'años'],
  ['streak_growth', 'Racha crecimiento', 'años'],
]

const ratioExplanations = [
  ['RPD TTM', 'Rentabilidad por dividendo pagada durante los últimos 12 meses. Sirve para medir renta actual.'],
  ['RPD forward', 'Rentabilidad esperada usando el dividendo anual previsto. Es útil, pero depende de estimaciones.'],
  ['DGR 5/10 años', 'Crecimiento anual compuesto del dividendo. Para dividendos crecientes interesa que sea positivo y estable.'],
  ['Payout', 'Porcentaje del beneficio destinado a dividendos. Si es demasiado alto, el dividendo tiene menos margen.'],
  ['PER', 'Precio dividido entre beneficio por acción. Ayuda a valorar si el precio parece exigente frente a beneficios.'],
  ['Deuda/Patrimonio', 'Relación entre deuda y fondos propios. Menor deuda suele dar más margen en crisis.'],
  ['ROE', 'Rentabilidad sobre fondos propios. Mide la calidad con la que la empresa convierte capital en beneficios.'],
  ['EV/EBITDA', 'Valor de empresa frente al EBITDA. Complementa al PER, especialmente si hay deuda relevante.'],
  ['FCF yield', 'Flujo de caja libre frente a capitalización. Indica cuánto efectivo genera el negocio respecto al precio.'],
  ['Rachas', 'Años consecutivos pagando o aumentando dividendo. Dan contexto sobre disciplina y estabilidad histórica.'],
]

const scoreBlocks = ['Dividendo', 'Solidez', 'Valoración', 'Historial']
const periodOptions = [
  ['6m', '6M', 183],
  ['1y', '1A', 365],
  ['3y', '3A', 1095],
  ['5y', '5A', 1825],
  ['all', 'Todo', null],
]

function formatNumber(value, suffix = '') {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return 's/d'
  const formatted = new Intl.NumberFormat('es-ES', {
    maximumFractionDigits: Math.abs(Number(value)) >= 100 ? 0 : 2,
  }).format(Number(value))
  return suffix ? `${formatted} ${suffix}` : formatted
}

function formatDate(value) {
  if (!value) return ''
  return new Intl.DateTimeFormat('es-ES', { month: 'short', year: 'numeric' }).format(new Date(value))
}

function getBarWidth(value, max) {
  if (!max || !Number.isFinite(max)) return '0%'
  return `${Math.max(0, Math.min(100, (Number(value) / max) * 100))}%`
}

function getScoreClass(score) {
  if (score >= 75) return 'good'
  if (score >= 60) return 'watch'
  return 'bad'
}

export function InvestmentPanel() {
  const [ticker, setTicker] = useState('KO')
  const [result, setResult] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [period, setPeriod] = useState('1y')
  const [error, setError] = useState('')

  const dividendMax = useMemo(() => {
    const values = result?.dividends_by_year?.map((item) => Number(item.amount)).filter(Number.isFinite) ?? []
    return Math.max(...values, 0)
  }, [result])

  const priceRows = useMemo(() => filterPriceRows(result?.price_history ?? [], period), [period, result])

  const analyzeTicker = async (event) => {
    event.preventDefault()
    const cleanTicker = ticker.trim().toUpperCase()
    if (!cleanTicker) {
      setError('Indica un ticker para analizar.')
      return
    }

    setIsLoading(true)
    setError('')
    try {
      const data = await requestJson(
        `/api/v1/investments/analyze?ticker=${encodeURIComponent(cleanTicker)}`,
        {},
        'No se pudo analizar el ticker',
      )
      setResult(data.result)
      setTicker(data.result?.ticker ?? cleanTicker)
    } catch (err) {
      setError(err.message)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <section className="panel investment-panel">
      <div className="panel-header dashboard-header">
        <div>
          <span className="section-kicker">Análisis de activos</span>
          <h2>Invertir</h2>
          <p>Dividendos, valoración y solidez por ticker</p>
        </div>
      </div>

      <form className="ticker-form" onSubmit={analyzeTicker}>
        <label>
          Ticker
          <input value={ticker} onChange={(event) => setTicker(event.target.value)} placeholder="KO, JNJ, ROVI.MC..." />
        </label>
        <button className="primary-button inline-primary" type="submit" disabled={isLoading}>
          <Search aria-hidden="true" size={18} />
          {isLoading ? 'Analizando...' : 'Analizar'}
        </button>
      </form>

      {error ? <p className="error-message">{error}</p> : null}

      {!result && !isLoading ? (
        <div className="empty-state compact-empty">
          <strong>Sin ticker cargado</strong>
          <span>Busca una empresa para ver métricas, score, dividendos y precio reciente.</span>
        </div>
      ) : null}

      {result ? (
        <div className="investment-dashboard">
          <header className="investment-hero">
            <div>
              <span>{result.ticker}</span>
              <h3>{result.name}</h3>
              <p>
                {result.sector || 'Sector no disponible'} · {result.exchange?.name || 'Bolsa no disponible'} ·{' '}
                {result.exchange?.currency || 'Divisa no disponible'}
              </p>
            </div>
            <article className={`score-card ${getScoreClass(result.score)}`}>
              <span>Score</span>
              <strong>{formatNumber(result.score)}/100</strong>
              <small>{result.recommendation}</small>
            </article>
          </header>

          <div className="investment-kpis">
            <article>
              <span>Precio</span>
              <strong>{formatNumber(result.price, result.exchange?.currency)}</strong>
            </article>
            {metricGroups.map(([key, label, suffix]) => (
              <article key={key}>
                <span>{label}</span>
                <strong>{formatNumber(result.metrics?.[key], suffix)}</strong>
              </article>
            ))}
          </div>

          {result.flags?.length ? (
            <section className="warning-panel">
              <strong>Banderas rojas</strong>
              {result.flags.map((flag) => (
                <span key={flag}>{flag}</span>
              ))}
            </section>
          ) : null}

          <section className="investment-card price-card">
            <div className="chart-header">
              <div>
                <h3>Precio histórico</h3>
                <span>{priceRows.length ? `${formatDate(priceRows[0].date)} - ${formatDate(priceRows.at(-1).date)}` : 'Sin datos'}</span>
              </div>
              <div className="period-tabs">
                {periodOptions.map(([key, label]) => (
                  <button className={period === key ? 'active' : ''} key={key} type="button" onClick={() => setPeriod(key)}>
                    {label}
                  </button>
                ))}
              </div>
            </div>
            <EconomicPriceChart rows={priceRows} currency={result.exchange?.currency} />
          </section>

          <div className="investment-analysis-grid">
            <section className="investment-card">
              <h3>Desglose del score</h3>
              <div className="score-breakdown">
                {scoreBlocks.map((block) => (
                  <div className="score-row" key={block}>
                    <span>{block}</span>
                    <div className="score-bar">
                      <span style={{ width: getBarWidth(result.breakdown?.[block], 40) }} />
                    </div>
                    <strong>{formatNumber(result.breakdown?.[block])}</strong>
                  </div>
                ))}
              </div>
            </section>

            <section className="investment-card">
              <h3>Análisis Groq</h3>
              <AiAnalysis analysis={result.ai_analysis} />
            </section>
          </div>

          <section className="investment-card">
            <h3>Dividendos anuales</h3>
            {result.dividends_by_year?.length ? (
              <div className="dividend-bars">
                {result.dividends_by_year.slice(-12).map((item) => (
                  <article key={item.year}>
                    <span>{item.year}</span>
                    <div>
                      <span style={{ width: getBarWidth(item.amount, dividendMax) }} />
                    </div>
                    <strong>{formatNumber(item.amount, result.exchange?.currency)}</strong>
                  </article>
                ))}
              </div>
            ) : (
              <p className="muted-text">Sin histórico de dividendos disponible.</p>
            )}
          </section>

          <section className="investment-card">
            <h3>Criterios por bloque</h3>
            <div className="criteria-grid">
              {scoreBlocks.map((block) => (
                <article key={block}>
                  <strong>{block}</strong>
                  {(result.details?.[block] ?? []).map((item) => (
                    <span key={`${block}-${item.metric}`}>
                      {item.metric}: {formatNumber(item.value)} · {item.range} · score {formatNumber(item.subscore)}
                    </span>
                  ))}
                </article>
              ))}
            </div>
          </section>

          <section className="investment-card">
            <h3>Guía de ratios</h3>
            <div className="ratio-guide-grid">
              {ratioExplanations.map(([label, text]) => (
                <article key={label}>
                  <strong>{label}</strong>
                  <span>{text}</span>
                </article>
              ))}
            </div>
          </section>
        </div>
      ) : null}
    </section>
  )
}

function AiAnalysis({ analysis }) {
  if (!analysis?.configured) {
    return <p className="muted-text">{analysis?.error ?? 'Configura GROQ_API_KEY para activar el análisis automático.'}</p>
  }
  if (analysis.error) return <p className="muted-text">{analysis.error}</p>
  if (!analysis.text) return <p className="muted-text">Sin análisis disponible.</p>

  return (
    <div className="ai-analysis">
      <span>Modelo: {analysis.model}</span>
      {analysis.profile ? <span>Perfil: largo plazo por dividendos crecientes</span> : null}
      {analysis.knowledge_sources?.length ? (
        <span>Conocimiento local: {analysis.knowledge_sources.join(', ')}</span>
      ) : null}
      {analysis.text.split('\n').map((line, index) => (
        <p key={`${index}-${line}`}>{line || '\u00a0'}</p>
      ))}
    </div>
  )
}

function EconomicPriceChart({ rows, currency }) {
  const [tooltip, setTooltip] = useState(null)
  const chart = useMemo(() => buildChart(rows), [rows])

  if (!chart.points.length) return <p className="muted-text">Sin histórico de precio disponible.</p>

  return (
    <div className="economic-chart-wrap">
      <div className="chart-scale">
        <span>{formatNumber(chart.max, currency)}</span>
        <span>{formatNumber(chart.min, currency)}</span>
      </div>
      <svg className="economic-chart" viewBox="0 0 720 320" role="img" aria-label="Precio histórico">
        <defs>
          <linearGradient id="priceArea" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor="#286b57" stopOpacity="0.24" />
            <stop offset="100%" stopColor="#286b57" stopOpacity="0.02" />
          </linearGradient>
        </defs>
        {[70, 130, 190, 250].map((y) => (
          <line className="economic-grid-line" key={y} x1="54" x2="690" y1={y} y2={y} />
        ))}
        <path className="economic-area" d={chart.areaPath} />
        <polyline className="economic-price-line" points={chart.points.join(' ')} />
        <polyline className="economic-average-line" points={chart.averagePoints.join(' ')} />
        {tooltip ? (
          <>
            <line className="economic-crosshair" x1={tooltip.x} x2={tooltip.x} y1="42" y2="270" />
            <circle className="economic-active-point" cx={tooltip.x} cy={tooltip.y} r="5.5" />
          </>
        ) : null}
        {chart.markers.map((marker) => (
          <g key={marker.label}>
            <line className="economic-axis-tick" x1={marker.x} x2={marker.x} y1="270" y2="276" />
            <text className="economic-axis-label" x={marker.x} y="296" textAnchor="middle">
              {marker.label}
            </text>
          </g>
        ))}
        {chart.pointData.map((point) => (
          <circle
            className="economic-hover-point"
            cx={point.x}
            cy={point.y}
            key={`${point.date}-${point.close}`}
            onMouseEnter={() => setTooltip(point)}
            onMouseLeave={() => setTooltip(null)}
            r="8"
          />
        ))}
      </svg>
      <div className="economic-legend">
        <span>Precio</span>
        <span>Media móvil 30 sesiones</span>
      </div>
      {tooltip ? (
        <div className="chart-tooltip investment-tooltip" style={{ left: `${(tooltip.x / 720) * 100}%`, top: `${(tooltip.y / 320) * 100}%` }}>
          <strong>{new Intl.DateTimeFormat('es-ES').format(new Date(tooltip.date))}</strong>
          <span>{formatNumber(tooltip.close, currency)}</span>
        </div>
      ) : null}
    </div>
  )
}

function filterPriceRows(rows, period) {
  const cleanRows = rows.filter((row) => Number.isFinite(Number(row.close)) && row.date)
  const option = periodOptions.find(([key]) => key === period)
  const days = option?.[2]
  if (!days || cleanRows.length < 2) return cleanRows

  const lastDate = new Date(cleanRows.at(-1).date)
  const cutoff = new Date(lastDate)
  cutoff.setDate(lastDate.getDate() - days)
  return cleanRows.filter((row) => new Date(row.date) >= cutoff)
}

function buildChart(rows) {
  const sampled = sampleRows(rows, 120)
  const values = sampled.map((row) => Number(row.close))
  const min = Math.min(...values)
  const max = Math.max(...values)
  const padding = (max - min) * 0.08 || 1
  const lower = min - padding
  const upper = max + padding
  const range = upper - lower || 1

  const pointData = sampled.map((row, index) => {
    const x = sampled.length === 1 ? 372 : 54 + (index / (sampled.length - 1)) * 636
    const y = 270 - ((Number(row.close) - lower) / range) * 220
    return { ...row, x, y }
  })
  const points = pointData.map((point) => `${point.x.toFixed(1)},${point.y.toFixed(1)}`)
  const averages = movingAverage(sampled, 30)
  const averagePoints = averages
    .map((value, index) => {
      if (value === null) return null
      const x = sampled.length === 1 ? 372 : 54 + (index / (sampled.length - 1)) * 636
      const y = 270 - ((value - lower) / range) * 220
      return `${x.toFixed(1)},${y.toFixed(1)}`
    })
    .filter(Boolean)
  const areaPath = points.length ? `M ${points[0]} L ${points.slice(1).join(' L ')} L 690 270 L 54 270 Z` : ''

  return {
    min,
    max,
    points,
    pointData: pointData.filter((_, index) => index % Math.max(1, Math.ceil(pointData.length / 48)) === 0),
    averagePoints,
    areaPath,
    markers: buildMarkers(sampled),
  }
}

function sampleRows(rows, maxPoints) {
  if (rows.length <= maxPoints) return rows
  const step = Math.ceil(rows.length / maxPoints)
  return rows.filter((_, index) => index % step === 0 || index === rows.length - 1)
}

function movingAverage(rows, windowSize) {
  return rows.map((_, index) => {
    if (index < windowSize - 1) return null
    const window = rows.slice(index - windowSize + 1, index + 1).map((row) => Number(row.close))
    return window.reduce((total, value) => total + value, 0) / window.length
  })
}

function buildMarkers(rows) {
  if (!rows.length) return []
  const count = Math.min(5, rows.length)
  return Array.from({ length: count }, (_, index) => {
    const rowIndex = count === 1 ? 0 : Math.round((index / (count - 1)) * (rows.length - 1))
    const x = count === 1 ? 372 : 54 + (rowIndex / (rows.length - 1)) * 636
    return { x, label: formatDate(rows[rowIndex].date) }
  })
}
