import { useMemo, useState } from 'react'
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
  ['fcf_yield', 'FCF yield', '%'],
  ['streak_years', 'Racha pagos', 'años'],
  ['streak_growth', 'Racha crecimiento', 'años'],
  ['fcf_pos_years', 'Años FCF positivo', ''],
]

const scoreBlocks = ['Dividendo', 'Solidez', 'Valoración', 'Historial']

function formatNumber(value, suffix = '') {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return 's/d'
  const formatted = new Intl.NumberFormat('es-ES', {
    maximumFractionDigits: Math.abs(Number(value)) >= 100 ? 0 : 2,
  }).format(Number(value))
  return suffix ? `${formatted} ${suffix}` : formatted
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
  const [error, setError] = useState('')

  const dividendMax = useMemo(() => {
    const values = result?.dividends_by_year?.map((item) => Number(item.amount)).filter(Number.isFinite) ?? []
    return Math.max(...values, 0)
  }, [result])

  const pricePoints = useMemo(() => getPricePoints(result?.price_history ?? []), [result])

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
      <div className="panel-header">
        <div>
          <h2>Invertir</h2>
          <span>Análisis de dividendos crecientes por ticker</span>
        </div>
      </div>

      <form className="ticker-form" onSubmit={analyzeTicker}>
        <label>
          Ticker
          <input value={ticker} onChange={(event) => setTicker(event.target.value)} placeholder="KO, JNJ, PG..." />
        </label>
        <button className="primary-button inline-primary" type="submit" disabled={isLoading}>
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
              <h3>Precio reciente</h3>
              {pricePoints.length ? (
                <svg className="investment-line-chart" viewBox="0 0 600 220" role="img" aria-label="Precio reciente">
                  <polyline points={pricePoints.join(' ')} fill="none" stroke="#286b57" strokeWidth="4" />
                </svg>
              ) : (
                <p className="muted-text">Sin histórico de precio disponible.</p>
              )}
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
        </div>
      ) : null}
    </section>
  )
}

function getPricePoints(rows) {
  if (!rows.length) return []
  const sample = rows.filter((_, index) => index % Math.ceil(rows.length / 80) === 0 || index === rows.length - 1)
  const values = sample.map((item) => Number(item.close)).filter(Number.isFinite)
  const min = Math.min(...values)
  const max = Math.max(...values)
  const range = max - min || 1

  return sample.map((item, index) => {
    const x = sample.length === 1 ? 300 : (index / (sample.length - 1)) * 560 + 20
    const y = 200 - ((Number(item.close) - min) / range) * 170
    return `${x.toFixed(1)},${y.toFixed(1)}`
  })
}
