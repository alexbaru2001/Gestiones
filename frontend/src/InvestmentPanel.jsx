import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { ListChecks, RefreshCw, Search, Star, Trash2 } from 'lucide-react'
import { requestJson } from './api'
import {
  evaluateMetric,
  extractPortfolioTickers,
  formatBand,
  getBestTickerForMetric,
  getMetricBand,
  getScoreTrend,
  getValuation,
  getWeakestBlock,
  markDividendCuts,
  metricGroups,
  metricHelp,
  ratioExplanations,
  scoreBlockWeights,
} from './investmentMetrics'
import {
  buildReviewRows,
  detectReviewChanges,
  splitAnalyzablePositions,
  summarizeReview,
} from './portfolioReview'
import {
  DEFAULT_FINALISTS,
  DEFAULT_TOP,
  chunkTickers,
  describeProgress,
  getValuationLabel,
  rankByPrescore,
  rankByScore,
  toExplorationRow,
} from './exploration'

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

function formatDateTime(value) {
  if (!value) return ''
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return ''
  return new Intl.DateTimeFormat('es-ES', { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' }).format(date)
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

export function InvestmentPanel({ autoStartReview = false, onReviewStarted }) {
  const [ticker, setTicker] = useState('KO')
  const [result, setResult] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [period, setPeriod] = useState('1y')
  const [error, setError] = useState('')

  const [mode, setMode] = useState('analizar')
  const [suggestions, setSuggestions] = useState([])
  const [areSuggestionsOpen, setAreSuggestionsOpen] = useState(false)
  const [watchlist, setWatchlist] = useState([])
  const [portfolioTickers, setPortfolioTickers] = useState([])
  const [compareSelection, setCompareSelection] = useState([])
  const [comparison, setComparison] = useState(null)
  const [isComparing, setIsComparing] = useState(false)
  const [portfolioSnapshot, setPortfolioSnapshot] = useState(null)
  const [review, setReview] = useState(null)
  const [reviewProgress, setReviewProgress] = useState(null)
  const [universes, setUniverses] = useState([])
  const [savedExplorations, setSavedExplorations] = useState({})
  const [universeKey, setUniverseKey] = useState('sp500')
  const [exploration, setExploration] = useState(null)
  const [exploreProgress, setExploreProgress] = useState(null)
  const searchBoxRef = useRef(null)

  const dividendRows = useMemo(() => markDividendCuts(result?.dividends_by_year ?? []), [result])
  const dividendMax = useMemo(() => {
    const values = dividendRows.map((item) => Number(item.amount)).filter(Number.isFinite)
    return Math.max(...values, 0)
  }, [dividendRows])
  const priceRows = useMemo(() => filterPriceRows(result?.price_history ?? [], period), [period, result])
  const valuation = useMemo(() => getValuation(result?.metrics), [result])
  const weakestBlock = useMemo(() => getWeakestBlock(result?.breakdown), [result])
  const portfolioSplit = useMemo(() => splitAnalyzablePositions(portfolioSnapshot), [portfolioSnapshot])
  const reviewRows = useMemo(() => buildReviewRows(review?.positions ?? []), [review])
  const reviewSummary = useMemo(() => summarizeReview(review?.positions ?? []), [review])
  const reviewChanges = useMemo(() => detectReviewChanges(review?.positions ?? []), [review])
  const watchedTicker = useMemo(
    () => watchlist.find((entry) => entry.ticker === result?.ticker) ?? null,
    [result, watchlist],
  )

  const loadWatchlist = useCallback(async () => {
    try {
      const data = await requestJson('/api/v1/investments/watchlist', {}, 'No se pudo cargar el seguimiento')
      setWatchlist(data.result ?? [])
    } catch {
      setWatchlist([])
    }
  }, [])

  useEffect(() => {
    loadWatchlist()
    requestJson('/api/v1/portfolio', {}, 'No se pudo cargar la cartera')
      .then((data) => {
        setPortfolioSnapshot(data.result)
        setPortfolioTickers(extractPortfolioTickers(data.result))
      })
      .catch(() => setPortfolioTickers([]))
    requestJson('/api/v1/investments/portfolio-review', {}, 'No se pudo cargar la revisión')
      .then((data) => setReview(data.result))
      .catch(() => setReview(null))
    requestJson('/api/v1/investments/universes', {}, 'No se pudieron cargar los universos')
      .then((data) => {
        setUniverses(data.result?.universes ?? [])
        setSavedExplorations(data.result?.explorations ?? {})
      })
      .catch(() => setUniverses([]))
  }, [loadWatchlist])

  // Búsqueda por nombre con retardo: antes había que saberse el ticker exacto ("ROVI.MC").
  useEffect(() => {
    const query = ticker.trim()
    if (query.length < 2 || !areSuggestionsOpen) {
      setSuggestions([])
      return undefined
    }
    const timer = setTimeout(() => {
      requestJson(`/api/v1/investments/search?q=${encodeURIComponent(query)}`, {}, 'No se pudo buscar')
        .then((data) => setSuggestions(data.result ?? []))
        .catch(() => setSuggestions([]))
    }, 350)
    return () => clearTimeout(timer)
  }, [areSuggestionsOpen, ticker])

  useEffect(() => {
    const closeOnOutsideClick = (event) => {
      if (searchBoxRef.current && !searchBoxRef.current.contains(event.target)) setAreSuggestionsOpen(false)
    }
    document.addEventListener('mousedown', closeOnOutsideClick)
    return () => document.removeEventListener('mousedown', closeOnOutsideClick)
  }, [])

  const analyze = useCallback(
    async (requestedTicker, { refresh = false } = {}) => {
      const cleanTicker = String(requestedTicker ?? '').trim().toUpperCase()
      if (!cleanTicker) {
        setError('Indica un ticker para analizar.')
        return
      }
      setMode('analizar')
      setAreSuggestionsOpen(false)
      setIsLoading(true)
      setError('')
      try {
        const data = await requestJson(
          `/api/v1/investments/analyze?ticker=${encodeURIComponent(cleanTicker)}${refresh ? '&refresh=true' : ''}`,
          {},
          'No se pudo analizar el ticker',
        )
        setResult(data.result)
        setTicker(data.result?.ticker ?? cleanTicker)
        loadWatchlist()
      } catch (err) {
        setError(err.message)
      } finally {
        setIsLoading(false)
      }
    },
    [loadWatchlist],
  )

  const toggleWatch = async (entryTicker, analysis = null) => {
    const key = String(entryTicker ?? '').trim().toUpperCase()
    if (!key) return
    const isWatched = watchlist.some((entry) => entry.ticker === key)
    try {
      const data = isWatched
        ? await requestJson(`/api/v1/investments/watchlist/${encodeURIComponent(key)}`, { method: 'DELETE' }, 'No se pudo quitar del seguimiento')
        : await requestJson(
            '/api/v1/investments/watchlist',
            {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ ticker: key, analysis }),
            },
            'No se pudo guardar en el seguimiento',
          )
      setWatchlist(data.result ?? [])
      setCompareSelection((previous) => previous.filter((item) => item !== key || !isWatched))
    } catch (err) {
      setError(err.message)
    }
  }

  const runPortfolioReview = useCallback(
    async ({ refresh = false } = {}) => {
      const positions = portfolioSplit.analyzable
      if (!positions.length) {
        setError('No hay posiciones con ticker que analizar. Importa una foto de cartera primero.')
        return
      }
      setMode('cartera')
      setError('')
      setReviewProgress({ done: 0, total: positions.length, ticker: positions[0].ticker, failed: [] })

      const failed = []
      let latest = review
      // Una posición cada vez: así se puede ir informando del avance y un fallo suelto no tumba
      // la revisión entera, como pasaría con una única petición de medio minuto.
      for (const [index, position] of positions.entries()) {
        setReviewProgress({ done: index, total: positions.length, ticker: position.ticker, failed: [...failed] })
        try {
          const analysis = await requestJson(
            `/api/v1/investments/analyze?ticker=${encodeURIComponent(position.ticker)}${refresh ? '&refresh=true' : ''}`,
            {},
            `No se pudo analizar ${position.ticker}`,
          )
          const saved = await requestJson(
            '/api/v1/investments/portfolio-review',
            {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ ticker: position.ticker, analysis: analysis.result, position }),
            },
            `No se pudo guardar la revisión de ${position.ticker}`,
          )
          latest = saved.result
          setReview(saved.result)
        } catch (err) {
          failed.push({ ticker: position.ticker, error: err.message })
        }
      }

      // Lo vendido deja de revisarse: si no, la nota de la cartera arrastraría posiciones que ya no están.
      try {
        const pruned = await requestJson(
          '/api/v1/investments/portfolio-review/prune',
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ tickers: positions.map((position) => position.ticker) }),
          },
          'No se pudo limpiar la revisión',
        )
        latest = pruned.result
        setReview(pruned.result)
      } catch {
        setReview(latest)
      }

      setReviewProgress({ done: positions.length, total: positions.length, ticker: '', failed })
    },
    [portfolioSplit, review],
  )

  useEffect(() => {
    if (!autoStartReview || !portfolioSplit.analyzable.length) return
    onReviewStarted?.()
    runPortfolioReview()
    // Solo debe dispararse cuando se llega desde el botón de Cartera y ya hay posiciones cargadas.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoStartReview, portfolioSplit.analyzable.length])

  const selectedUniverse = universes.find((item) => item.key === universeKey) ?? null

  const runExploration = useCallback(async () => {
    const universe = universes.find((item) => item.key === universeKey)
    if (!universe) return
    setMode('explorar')
    setError('')
    setExploration(null)
    setExploreProgress({ phase: 'criba', done: 0, total: universe.size, ticker: '' })

    try {
      const candidatesResponse = await requestJson(
        '/api/v1/investments/explore/candidates',
        { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ universe: universeKey }) },
        'No se pudo preparar el universo',
      )
      const candidates = candidatesResponse.result?.candidates ?? []
      const chunkSize = candidatesResponse.result?.chunk ?? 100

      // Fase 1: criba barata por tandas, para poder ir contando el avance.
      const screened = []
      const chunks = chunkTickers(candidates, chunkSize)
      for (const chunk of chunks) {
        setExploreProgress({ phase: 'criba', done: screened.length, total: candidates.length, ticker: '' })
        const response = await requestJson(
          '/api/v1/investments/explore/screen',
          { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ tickers: chunk }) },
          'Falló la criba',
        )
        screened.push(...(response.result ?? []))
      }

      // Fase 2: solo las finalistas pasan por el análisis completo, que es lo caro.
      const finalists = rankByPrescore(screened, DEFAULT_FINALISTS)
      const analyzed = []
      const failed = []
      for (const [index, finalist] of finalists.entries()) {
        setExploreProgress({ phase: 'analisis', done: index, total: finalists.length, ticker: finalist.ticker })
        try {
          const response = await requestJson(
            `/api/v1/investments/analyze?ticker=${encodeURIComponent(finalist.ticker)}`,
            {},
            `No se pudo analizar ${finalist.ticker}`,
          )
          analyzed.push(toExplorationRow(response.result, finalist.prescore))
        } catch (err) {
          failed.push({ ticker: finalist.ticker, error: err.message })
        }
      }

      const results = rankByScore(analyzed, DEFAULT_TOP)
      const saved = await requestJson(
        '/api/v1/investments/explore/save',
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            universe: universeKey,
            label: universe.label,
            approximate: universe.approximate,
            screened: screened.length,
            analyzed: analyzed.length,
            results,
          }),
        },
        'No se pudo guardar la exploración',
      )
      setExploration({ ...saved.result, failed })
      setSavedExplorations((previous) => ({ ...previous, [universeKey]: saved.result }))
      setExploreProgress(null)
    } catch (err) {
      setError(err.message)
      setExploreProgress(null)
    }
  }, [universeKey, universes])

  const refreshUniverse = async () => {
    try {
      await requestJson(
        `/api/v1/investments/universes/${encodeURIComponent(universeKey)}/refresh`,
        { method: 'POST' },
        'No se pudo refrescar la lista',
      )
      const data = await requestJson('/api/v1/investments/universes', {}, 'No se pudieron cargar los universos')
      setUniverses(data.result?.universes ?? [])
    } catch (err) {
      setError(err.message)
    }
  }

  const toggleCompare = (entryTicker) => {
    setCompareSelection((previous) =>
      previous.includes(entryTicker)
        ? previous.filter((item) => item !== entryTicker)
        : previous.length >= 4
          ? previous
          : [...previous, entryTicker],
    )
  }

  const runComparison = async (refresh = false) => {
    if (compareSelection.length < 2) {
      setError('Elige al menos dos tickers para comparar.')
      return
    }
    setMode('comparar')
    setIsComparing(true)
    setError('')
    try {
      const data = await requestJson(
        `/api/v1/investments/compare?tickers=${encodeURIComponent(compareSelection.join(','))}${refresh ? '&refresh=true' : ''}`,
        {},
        'No se pudo comparar',
      )
      setComparison(data.result)
    } catch (err) {
      setError(err.message)
    } finally {
      setIsComparing(false)
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
        <div className="investment-mode-tabs" role="tablist" aria-label="Modo de análisis">
          <button
            className={mode === 'analizar' ? 'active' : ''}
            role="tab"
            aria-selected={mode === 'analizar'}
            type="button"
            onClick={() => setMode('analizar')}
          >
            Analizar
          </button>
          <button
            className={mode === 'comparar' ? 'active' : ''}
            role="tab"
            aria-selected={mode === 'comparar'}
            type="button"
            onClick={() => setMode('comparar')}
          >
            Comparar {compareSelection.length ? `(${compareSelection.length})` : ''}
          </button>
          <button
            className={mode === 'explorar' ? 'active' : ''}
            role="tab"
            aria-selected={mode === 'explorar'}
            type="button"
            onClick={() => setMode('explorar')}
          >
            Explorar
          </button>
          <button
            className={mode === 'cartera' ? 'active' : ''}
            role="tab"
            aria-selected={mode === 'cartera'}
            type="button"
            onClick={() => setMode('cartera')}
          >
            Mi cartera
          </button>
        </div>
      </div>

      <form
        className="ticker-form"
        onSubmit={(event) => {
          event.preventDefault()
          analyze(ticker)
        }}
      >
        <label className="ticker-search" ref={searchBoxRef}>
          Empresa o ticker
          <input
            value={ticker}
            onChange={(event) => {
              setTicker(event.target.value)
              setAreSuggestionsOpen(true)
            }}
            onFocus={() => setAreSuggestionsOpen(true)}
            placeholder="Iberdrola, Coca-Cola, ROVI.MC..."
            autoComplete="off"
          />
          {areSuggestionsOpen && suggestions.length ? (
            <ul className="ticker-suggestions">
              {suggestions.map((item) => (
                <li key={item.ticker}>
                  <button type="button" onClick={() => analyze(item.ticker)}>
                    <strong>{item.ticker}</strong>
                    <span>{item.name}</span>
                    <small>{[item.exchange, item.sector].filter(Boolean).join(' · ')}</small>
                  </button>
                </li>
              ))}
            </ul>
          ) : null}
        </label>
        <button className="primary-button inline-primary" type="submit" disabled={isLoading}>
          <Search aria-hidden="true" size={18} />
          {isLoading ? 'Analizando...' : 'Analizar'}
        </button>
      </form>

      {portfolioSplit.analyzable.length ? (
        <div className="portfolio-shortcuts">
          <button
            className="primary-button inline-primary review-button"
            type="button"
            disabled={Boolean(reviewProgress && reviewProgress.done < reviewProgress.total)}
            onClick={() => runPortfolioReview()}
          >
            <ListChecks aria-hidden="true" size={17} />
            Analizar toda la cartera ({portfolioSplit.analyzable.length})
          </button>
          <span className="muted-text">En tu cartera:</span>
          {portfolioTickers.map((item) => (
            <button key={item.ticker} type="button" title={item.name} onClick={() => analyze(item.ticker)}>
              {item.ticker}
            </button>
          ))}
        </div>
      ) : null}

      {error ? <p className="error-message">{error}</p> : null}

      {mode !== 'cartera' && mode !== 'explorar' ? (
      <WatchlistTable
        entries={watchlist}
        selection={compareSelection}
        onAnalyze={analyze}
        onRemove={(entryTicker) => toggleWatch(entryTicker)}
        onToggleCompare={toggleCompare}
        onCompare={() => runComparison(false)}
        isComparing={isComparing}
      />
      ) : null}

      {mode === 'comparar' ? (
        <ComparisonTable comparison={comparison} onAnalyze={analyze} isLoading={isComparing} />
      ) : null}

      {mode === 'explorar' ? (
        <UniverseExplorer
          exploration={exploration ?? savedExplorations[universeKey] ?? null}
          onAnalyze={analyze}
          onExplore={runExploration}
          onRefreshMembers={refreshUniverse}
          onSelect={(key) => {
            setUniverseKey(key)
            setExploration(null)
          }}
          progress={exploreProgress}
          selected={selectedUniverse}
          universes={universes}
        />
      ) : null}

      {mode === 'cartera' ? (
        <PortfolioReview
          changes={reviewChanges}
          lastReview={review?.last_review}
          onAnalyze={analyze}
          onRefresh={() => runPortfolioReview({ refresh: true })}
          progress={reviewProgress}
          rows={reviewRows}
          split={portfolioSplit}
          summary={reviewSummary}
        />
      ) : null}

      {mode === 'analizar' && !result && !isLoading ? (
        <div className="empty-state compact-empty">
          <strong>Sin ticker cargado</strong>
          <span>Busca una empresa por nombre o pulsa una de tu cartera para ver métricas, score y dividendos.</span>
        </div>
      ) : null}

      {mode === 'analizar' && result ? (
        <div className="investment-dashboard">
          <header className="investment-hero">
            <div>
              <span>{result.ticker}</span>
              <h3>{result.name}</h3>
              <p>
                {result.sector || 'Sector no disponible'} · {result.exchange?.name || 'Bolsa no disponible'} ·{' '}
                {result.exchange?.currency || 'Divisa no disponible'}
              </p>
              <div className="investment-hero-actions">
                <button
                  className={watchedTicker ? 'watch-button is-watched' : 'watch-button'}
                  type="button"
                  onClick={() => toggleWatch(result.ticker, result)}
                >
                  <Star aria-hidden="true" size={15} />
                  {watchedTicker ? 'En seguimiento' : 'Seguir'}
                </button>
                <button className="watch-button" type="button" onClick={() => analyze(result.ticker, { refresh: true })}>
                  <RefreshCw aria-hidden="true" size={15} />
                  Actualizar
                </button>
                {result.cached_at ? <span className="muted-text">Datos de {formatDateTime(result.cached_at)}</span> : null}
              </div>
            </div>
            <div className="investment-hero-score">
              <article className={`score-card ${getScoreClass(result.score)}`}>
                <span>Score</span>
                <strong>{formatNumber(result.score)}/100</strong>
                <small>{result.recommendation}</small>
              </article>
              {result.flags?.length ? (
                <section className="warning-panel hero-flags">
                  <strong>Banderas rojas</strong>
                  {result.flags.map((flag) => (
                    <span key={flag}>{flag}</span>
                  ))}
                </section>
              ) : null}
              <ScoreTrend entry={watchedTicker} />
            </div>
          </header>

          <div className="investment-kpis">
            <article>
              <span>Precio</span>
              <strong>{formatNumber(result.price, result.exchange?.currency)}</strong>
            </article>
            {metricGroups.map(([key, label, suffix]) => {
              const band = getMetricBand(key, result.rules)
              const state = evaluateMetric(result.metrics?.[key], band)
              return (
                <article className={state ? `kpi-${state}` : undefined} key={key} title={metricHelp[key]}>
                  <span>{label}</span>
                  <strong>{formatNumber(result.metrics?.[key], suffix)}</strong>
                  {band ? <small>{formatBand(band, suffix === 'x' ? '' : suffix)}</small> : null}
                </article>
              )
            })}
          </div>

          {valuation ? (
            <section className="investment-card valuation-card">
              <h3>Valoración por dividendo</h3>
              <div className="valuation-grid">
                <article>
                  <span>RPD actual</span>
                  <strong>{formatNumber(valuation.current, '%')}</strong>
                </article>
                <article>
                  <span>Su media de 5 años</span>
                  <strong>{formatNumber(valuation.average, '%')}</strong>
                </article>
                <article className={valuation.verdict === 'barata' ? 'kpi-on' : valuation.verdict === 'cara' ? 'kpi-off' : undefined}>
                  <span>Frente a su media</span>
                  <strong>
                    {valuation.difference >= 0 ? '+' : ''}
                    {formatNumber(valuation.difference, 'pp')}
                  </strong>
                  <small>cotiza {valuation.verdict}</small>
                </article>
                <article>
                  <span>Precio de referencia</span>
                  <strong>{formatNumber(valuation.referencePrice, result.exchange?.currency)}</strong>
                  <small>al que daría su RPD media</small>
                </article>
              </div>
              <p className="muted-text">
                El precio de referencia es orientativo: solo dice a qué precio el dividendo actual rendiría lo que ha
                rendido de media estos cinco años. No incorpora crecimiento ni riesgo del negocio.
              </p>
            </section>
          ) : null}

          <section className="investment-card price-card">
            <div className="chart-header">
              <div>
                <h3>Precio histórico</h3>
                <span>
                  {priceRows.length ? `${formatDate(priceRows[0].date)} - ${formatDate(priceRows.at(-1).date)}` : 'Sin datos'}
                </span>
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
              {weakestBlock && weakestBlock.lost > 0 ? (
                <p className="muted-text">
                  Donde más puntos pierde: <strong>{weakestBlock.block}</strong> ({formatNumber(weakestBlock.lost)} de{' '}
                  {weakestBlock.weight} posibles).
                </p>
              ) : null}
              <ScoreBreakdown breakdown={result.breakdown} details={result.details} rules={result.rules} />
            </section>

            <section className="investment-card">
              <h3>Análisis Groq</h3>
              <AiAnalysis analysis={result.ai_analysis} />
            </section>
          </div>

          <section className="investment-card">
            <h3>Dividendos anuales</h3>
            {dividendRows.length ? (
              <div className="dividend-bars">
                {dividendRows.slice(-12).map((item) => (
                  <article className={`trend-${item.trend}`} key={item.year}>
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
        </div>
      ) : null}

      <RatioGuide />
    </section>
  )
}

function RatioGuide() {
  const [isOpen, setIsOpen] = useState(true)

  return (
    <section className="investment-card ratio-guide-card">
      <div className="chart-header">
        <div>
          <h3>Guía de ratios</h3>
          <span>Qué mide cada cifra del panel</span>
        </div>
        <button className="watch-button" type="button" aria-expanded={isOpen} onClick={() => setIsOpen(!isOpen)}>
          {isOpen ? 'Ocultar' : 'Mostrar'}
        </button>
      </div>
      {isOpen ? (
        <div className="ratio-guide-grid">
          {ratioExplanations.map(([label, text]) => (
            <article key={label}>
              <strong>{label}</strong>
              <span>{text}</span>
            </article>
          ))}
        </div>
      ) : null}
    </section>
  )
}

function ScoreTrend({ entry }) {
  const trend = useMemo(() => getScoreTrend(entry?.history ?? []), [entry])
  if (!trend) return null
  const direction = trend.delta > 0 ? 'up' : trend.delta < 0 ? 'down' : 'flat'
  return (
    <p className={`score-trend trend-${direction}`}>
      {trend.delta >= 0 ? '+' : ''}
      {formatNumber(trend.delta)} puntos desde {trend.since}
    </p>
  )
}

function ScoreBreakdown({ breakdown, details, rules }) {
  const [openBlock, setOpenBlock] = useState('')
  return (
    <div className="score-breakdown">
      {scoreBlocks.map((block) => {
        const isOpen = openBlock === block
        const rows = details?.[block] ?? []
        return (
          <div className={isOpen ? 'score-row is-open' : 'score-row'} key={block}>
            <button type="button" aria-expanded={isOpen} onClick={() => setOpenBlock(isOpen ? '' : block)}>
              <span>{block}</span>
              <div className="score-bar">
                <span style={{ width: getBarWidth(breakdown?.[block], scoreBlockWeights[block]) }} />
              </div>
              <strong>
                {formatNumber(breakdown?.[block])}/{scoreBlockWeights[block]}
              </strong>
            </button>
            {isOpen ? (
              <ul className="score-criteria">
                {rows.length ? (
                  rows.map((item) => (
                    <li key={`${block}-${item.metric}`}>
                      <span className="criteria-name">{item.metric}</span>
                      <span className="criteria-value">{formatNumber(item.value)}</span>
                      <span className="criteria-range">{item.range}</span>
                      <span className="criteria-score">{formatNumber(item.subscore)}</span>
                    </li>
                  ))
                ) : (
                  <li className="muted-text">Sin criterios disponibles para este bloque.</li>
                )}
              </ul>
            ) : null}
          </div>
        )
      })}
      {rules ? <p className="muted-text">Umbrales aplicados según el sector de la empresa.</p> : null}
    </div>
  )
}

function WatchlistTable({ entries, selection, onAnalyze, onRemove, onToggleCompare, onCompare, isComparing }) {
  if (!entries.length) {
    return (
      <section className="investment-card watchlist-card">
        <div className="chart-header">
          <div>
            <h3>Seguimiento</h3>
            <span>Guarda una empresa con "Seguir" y quedará aquí con su puntuación</span>
          </div>
        </div>
        <p className="muted-text">Todavía no sigues ninguna empresa.</p>
      </section>
    )
  }

  return (
    <section className="investment-card watchlist-card">
      <div className="chart-header">
        <div>
          <h3>Seguimiento</h3>
          <span>{entries.length} empresas guardadas · marca 2 a 4 para compararlas</span>
        </div>
        <button
          className="watch-button"
          type="button"
          disabled={selection.length < 2 || isComparing}
          onClick={onCompare}
        >
          {isComparing ? 'Comparando...' : `Comparar (${selection.length})`}
        </button>
      </div>
      <div className="table-wrap">
        <table className="watchlist-table">
          <thead>
            <tr>
              <th aria-label="Comparar" />
              <th>Ticker</th>
              <th>Empresa</th>
              <th className="num">Score</th>
              <th className="num">RPD</th>
              <th className="num">Precio</th>
              <th>Revisado</th>
              <th aria-label="Quitar" />
            </tr>
          </thead>
          <tbody>
            {entries.map((entry) => {
              const trend = getScoreTrend(entry.history ?? [])
              return (
                <tr key={entry.ticker}>
                  <td>
                    <input
                      type="checkbox"
                      checked={selection.includes(entry.ticker)}
                      onChange={() => onToggleCompare(entry.ticker)}
                      aria-label={`Comparar ${entry.ticker}`}
                    />
                  </td>
                  <td>
                    <button className="link-button" type="button" onClick={() => onAnalyze(entry.ticker)}>
                      {entry.ticker}
                    </button>
                  </td>
                  <td>{entry.name}</td>
                  <td className="num">
                    {Number.isFinite(Number(entry.last_score)) ? (
                      <span className={`score-pill ${getScoreClass(Number(entry.last_score))}`}>
                        {formatNumber(entry.last_score)}
                      </span>
                    ) : (
                      's/d'
                    )}
                    {trend && trend.delta !== 0 ? (
                      <small className={trend.delta > 0 ? 'trend-up' : 'trend-down'}>
                        {trend.delta > 0 ? '+' : ''}
                        {formatNumber(trend.delta)}
                      </small>
                    ) : null}
                  </td>
                  <td className="num">{formatNumber(entry.last_rpd, '%')}</td>
                  <td className="num">{formatNumber(entry.last_price, entry.currency)}</td>
                  <td>{entry.last_checked ?? 'nunca'}</td>
                  <td>
                    <button
                      className="icon-button"
                      type="button"
                      aria-label={`Quitar ${entry.ticker} del seguimiento`}
                      onClick={() => onRemove(entry.ticker)}
                    >
                      <Trash2 aria-hidden="true" size={15} />
                    </button>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </section>
  )
}



function UniverseExplorer({ exploration, onAnalyze, onExplore, onRefreshMembers, onSelect, progress, selected, universes }) {
  const isRunning = Boolean(progress)
  const results = exploration?.results ?? []
  const failed = exploration?.failed ?? []

  return (
    <div className="portfolio-review">
      <section className="investment-card">
        <div className="chart-header">
          <div>
            <h3>Explorar un universo</h3>
            <span>Criba rápida de todo el índice y análisis a fondo de las {DEFAULT_FINALISTS} mejores</span>
          </div>
        </div>

        <div className="universe-picker" role="radiogroup" aria-label="Universo a explorar">
          {universes.map((universe) => (
            <button
              aria-checked={selected?.key === universe.key}
              className={selected?.key === universe.key ? 'is-selected' : undefined}
              key={universe.key}
              role="radio"
              type="button"
              onClick={() => onSelect(universe.key)}
            >
              <strong>{universe.label}</strong>
              <small>{universe.size} empresas{universe.approximate ? ' · aproximado' : ''}</small>
            </button>
          ))}
        </div>

        {selected ? <p className="muted-text">{selected.description}</p> : null}

        <div className="investment-hero-actions">
          <button className="primary-button inline-primary review-button" type="button" disabled={isRunning} onClick={onExplore}>
            <ListChecks aria-hidden="true" size={17} />
            {isRunning ? 'Explorando...' : `Analizar ${selected?.label ?? 'universo'}`}
          </button>
          {selected?.source_url ? (
            <button className="watch-button" type="button" disabled={isRunning} onClick={onRefreshMembers}>
              <RefreshCw aria-hidden="true" size={15} />
              Refrescar lista de miembros
            </button>
          ) : null}
          {exploration?.finished_at ? (
            <span className="muted-text">Última exploración: {formatDateTime(exploration.finished_at)}</span>
          ) : null}
        </div>

        {isRunning ? (
          <div className="review-progress">
            <div className="review-progress-bar">
              <span style={{ width: `${progress.total ? (progress.done / progress.total) * 100 : 0}%` }} />
            </div>
            <span className="muted-text">{describeProgress(progress)}</span>
          </div>
        ) : null}

        {failed.length ? (
          <p className="warning-text">Sin datos para: {failed.map((item) => item.ticker).join(', ')}</p>
        ) : null}
      </section>

      {results.length ? (
        <section className="investment-card">
          <div className="chart-header">
            <div>
              <h3>Las {results.length} mejores de {exploration.label ?? selected?.label}</h3>
              <span>
                Cribadas {exploration.screened ?? '?'} empresas · analizadas a fondo {exploration.analyzed ?? '?'}
              </span>
            </div>
          </div>
          <p className="muted-text">
            Son las mejores de las {DEFAULT_FINALISTS} finalistas que mejor pintaban en la criba, no un top
            {' '}{DEFAULT_TOP} garantizado del índice entero: la criba solo mira dividendo e historial, así que una
            empresa con poca RPD hoy y cuentas impecables puede quedarse fuera.
            {exploration.approximate
              ? ' Además, este universo es una aproximación por región y tamaño, no la lista oficial del índice.'
              : ''}
          </p>
          <div className="table-wrap">
            <table className="watchlist-table review-table">
              <thead>
                <tr>
                  <th>Empresa</th>
                  <th>Sector</th>
                  <th className="num">Score</th>
                  <th className="num">RPD</th>
                  <th className="num">DGR 5a</th>
                  <th className="num">Payout</th>
                  <th className="num">PER</th>
                  <th>Frente a su media</th>
                  <th className="num">Banderas</th>
                </tr>
              </thead>
              <tbody>
                {results.map((row) => {
                  const valuation = getValuationLabel(row)
                  return (
                    <tr key={row.ticker}>
                      <td>
                        <button className="link-button" type="button" onClick={() => onAnalyze(row.ticker)}>
                          {row.ticker}
                        </button>
                        <small className="review-name">{row.name}</small>
                      </td>
                      <td>{row.sector || '—'}</td>
                      <td className="num">
                        <span className={`score-pill ${getScoreClass(Number(row.score))}`}>{formatNumber(row.score)}</span>
                      </td>
                      <td className="num">{formatNumber(row.rpd_ttm, '%')}</td>
                      <td className="num">{formatNumber(row.dgr5, '%')}</td>
                      <td className="num">{formatNumber(row.payout, '%')}</td>
                      <td className="num">{formatNumber(row.per_ttm)}</td>
                      <td className={valuation === 'barata' ? 'trend-up' : valuation === 'cara' ? 'trend-down' : undefined}>
                        {valuation ?? 's/d'}
                      </td>
                      <td className="num">{row.flags?.length ? <span className="flag-count">{row.flags.length}</span> : '—'}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </section>
      ) : !isRunning ? (
        <div className="empty-state compact-empty">
          <strong>Sin exploración</strong>
          <span>Elige un universo y pulsa Analizar. La criba tarda segundos; el análisis a fondo, un par de minutos.</span>
        </div>
      ) : null}
    </div>
  )
}

function PortfolioReview({ changes, lastReview, onAnalyze, onRefresh, progress, rows, split, summary }) {
  const isRunning = Boolean(progress && progress.done < progress.total)
  const failed = progress?.failed ?? []

  return (
    <div className="portfolio-review">
      <section className="investment-card">
        <div className="chart-header">
          <div>
            <h3>Revisión de cartera</h3>
            <span>
              {lastReview ? `Última revisión completa: ${lastReview}` : 'Todavía no has revisado la cartera'}
            </span>
          </div>
          <button className="watch-button" type="button" disabled={isRunning} onClick={onRefresh}>
            <RefreshCw aria-hidden="true" size={15} />
            Forzar actualización
          </button>
        </div>

        {isRunning ? (
          <div className="review-progress">
            <div className="review-progress-bar">
              <span style={{ width: `${(progress.done / progress.total) * 100}%` }} />
            </div>
            <span className="muted-text">
              Analizando {progress.done + 1} de {progress.total} · {progress.ticker}
            </span>
          </div>
        ) : null}

        {summary ? (
          <div className="review-summary">
            <article>
              <span>Nota media ponderada</span>
              <strong>{formatNumber(summary.weightedScore)}/100</strong>
              <small>media simple: {formatNumber(summary.simpleScore)}</small>
            </article>
            <article className={summary.flaggedCount ? 'kpi-off' : 'kpi-on'}>
              <span>Con banderas rojas</span>
              <strong>{summary.flaggedCount} de {summary.analyzed}</strong>
              <small>{formatNumber(summary.flaggedWeight, '%')} de lo analizado</small>
            </article>
            <article>
              <span>Analizado</span>
              <strong>{formatNumber(summary.reviewedValue, '€')}</strong>
              <small>{summary.analyzed} posiciones con ticker</small>
            </article>
            <article>
              <span>Sin analizar</span>
              <strong>{split.skipped.length} posiciones</strong>
              <small>{formatNumber(split.skippedWeight, '%')} de la cartera</small>
            </article>
          </div>
        ) : (
          <p className="muted-text">
            Pulsa "Analizar toda la cartera" para puntuar tus {split.analyzable.length} posiciones con ticker.
          </p>
        )}

        {failed.length ? (
          <p className="warning-text">
            Sin datos para: {failed.map((item) => `${item.ticker} (${item.error})`).join(', ')}
          </p>
        ) : null}
      </section>

      {changes.length ? (
        <section className="investment-card">
          <h3>Cambios desde la revisión anterior</h3>
          <ul className="review-changes">
            {changes.map((change) => (
              <li className={`severity-${change.severity}`} key={`${change.ticker}-${change.kind}`}>
                <strong>{change.ticker}</strong>
                <span>{change.text}</span>
              </li>
            ))}
          </ul>
        </section>
      ) : rows.length ? (
        <section className="investment-card">
          <h3>Cambios desde la revisión anterior</h3>
          <p className="muted-text">Nada relevante ha cambiado desde la revisión anterior.</p>
        </section>
      ) : null}

      {rows.length ? (
        <section className="investment-card">
          <div className="chart-header">
            <div>
              <h3>Posiciones revisadas</h3>
              <span>Ordenadas de peor a mejor puntuación</span>
            </div>
          </div>
          <div className="table-wrap">
            <table className="watchlist-table review-table">
              <thead>
                <tr>
                  <th>Activo</th>
                  <th className="num">Peso</th>
                  <th className="num">Score</th>
                  <th className="num">Δ</th>
                  <th className="num">RPD</th>
                  <th>Frente a su media</th>
                  <th className="num">Banderas</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={row.ticker}>
                    <td>
                      <button className="link-button" type="button" onClick={() => onAnalyze(row.ticker)}>
                        {row.ticker}
                      </button>
                      <small className="review-name">{row.portfolio_name ?? row.name}</small>
                    </td>
                    <td className="num">{formatNumber(row.weight, '%')}</td>
                    <td className="num">
                      {Number.isFinite(Number(row.last_score)) ? (
                        <span className={`score-pill ${getScoreClass(Number(row.last_score))}`}>
                          {formatNumber(row.last_score)}
                        </span>
                      ) : (
                        's/d'
                      )}
                    </td>
                    <td className="num">
                      {row.delta === null ? (
                        '—'
                      ) : (
                        <span className={row.delta > 0 ? 'trend-up' : row.delta < 0 ? 'trend-down' : undefined}>
                          {row.delta > 0 ? '+' : ''}
                          {formatNumber(row.delta)}
                        </span>
                      )}
                    </td>
                    <td className="num">{formatNumber(row.last_rpd, '%')}</td>
                    <td className={row.valuation === 'barata' ? 'trend-up' : row.valuation === 'cara' ? 'trend-down' : undefined}>
                      {row.valuation ?? 's/d'}
                    </td>
                    <td className="num">{row.flagCount ? <span className="flag-count">{row.flagCount}</span> : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      {split.skipped.length ? (
        <section className="investment-card">
          <h3>No analizables</h3>
          <p className="muted-text">
            Fondos indexados, efectivo y criptoactivos no tienen ticker de Yahoo ni métricas de dividendo, así que la
            revisión no los cubre: son {split.skipped.length} posiciones y {formatNumber(split.skippedWeight, '%')} de
            tu cartera.
          </p>
          <ul className="skipped-list">
            {split.skipped.map((item) => (
              <li key={item.name}>
                <span>{item.name}</span>
                <strong>{formatNumber(item.value, '€')}</strong>
              </li>
            ))}
          </ul>
        </section>
      ) : null}
    </div>
  )
}

function ComparisonTable({ comparison, onAnalyze, isLoading }) {
  if (isLoading) return <p className="muted-text">Comparando empresas...</p>
  if (!comparison) {
    return (
      <div className="empty-state compact-empty">
        <strong>Sin comparación</strong>
        <span>Marca entre 2 y 4 empresas de tu seguimiento y pulsa "Comparar".</span>
      </div>
    )
  }

  const { columns = [], metrics = [], errors = [] } = comparison
  if (!columns.length) return <p className="muted-text">No se pudo analizar ninguno de los tickers elegidos.</p>

  return (
    <section className="investment-card comparison-card">
      <div className="chart-header">
        <div>
          <h3>Comparativa</h3>
          <span>La mejor cifra de cada fila va destacada</span>
        </div>
      </div>
      <div className="table-wrap">
        <table className="comparison-table">
          <thead>
            <tr>
              <th>Métrica</th>
              {columns.map((column) => (
                <th className="num" key={column.ticker}>
                  <button className="link-button" type="button" onClick={() => onAnalyze(column.ticker)}>
                    {column.ticker}
                  </button>
                  <small>{column.name}</small>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            <tr className="comparison-score-row">
              <th scope="row">Score</th>
              {columns.map((column) => (
                <td className="num" key={column.ticker}>
                  <span className={`score-pill ${getScoreClass(Number(column.score))}`}>{formatNumber(column.score)}</span>
                </td>
              ))}
            </tr>
            <tr>
              <th scope="row">Precio</th>
              {columns.map((column) => (
                <td className="num" key={column.ticker}>
                  {formatNumber(column.price, column.exchange?.currency)}
                </td>
              ))}
            </tr>
            {metrics.map((metric) => {
              const best = getBestTickerForMetric(columns, metric)
              return (
                <tr key={metric.key}>
                  <th scope="row" title={metricHelp[metric.key]}>
                    {metric.label}
                  </th>
                  {columns.map((column) => {
                    const band = getMetricBand(metric.key, column.rules)
                    const state = evaluateMetric(column.metrics?.[metric.key], band)
                    return (
                      <td
                        className={`num${column.ticker === best ? ' is-best' : ''}${state === 'off' ? ' is-off' : ''}`}
                        key={column.ticker}
                      >
                        {formatNumber(column.metrics?.[metric.key], metric.suffix)}
                      </td>
                    )
                  })}
                </tr>
              )
            })}
            <tr>
              <th scope="row">Banderas rojas</th>
              {columns.map((column) => (
                <td className="num" key={column.ticker}>
                  {column.flags?.length ? <span className="flag-count">{column.flags.length}</span> : '—'}
                </td>
              ))}
            </tr>
            <tr>
              <th scope="row">Veredicto</th>
              {columns.map((column) => (
                <td key={column.ticker}>{column.recommendation}</td>
              ))}
            </tr>
          </tbody>
        </table>
      </div>
      {errors.length ? (
        <p className="warning-text">
          Sin datos para: {errors.map((item) => `${item.ticker} (${item.error})`).join(', ')}
        </p>
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
