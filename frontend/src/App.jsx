import { useEffect, useMemo, useRef, useState } from 'react'
import { BriefcaseBusiness, ChartNoAxesCombined, PanelLeftClose, PanelLeftOpen, Server, WalletCards } from 'lucide-react'
import { requestJson } from './api'
import { downloadJson, downloadText, toCsv } from './exporters'
import { InputPanel } from './InputPanel'
import { InvestmentPanel } from './InvestmentPanel'
import { PortfolioPanel } from './PortfolioPanel'
import { createObjective, objectiveFromApi, objectiveToPayload } from './objectives'
import {
  filterObjectiveRows,
  getDividendPayments,
  getHistoryRows,
  getObjectiveNames,
  getObjectiveRows,
  getObjectiveTotals,
  getPreviousRow,
  getRecentRows,
  getSelectedRow,
  getTrendMax,
} from './resultSelectors'
import { ResultsPanel } from './ResultsPanel'
import { validateFinanceInput } from './validation'

function nextMonthStart(yyyyMm) {
  const [year, month] = yyyyMm.split('-').map(Number)
  return new Date(Date.UTC(year, month, 1)).toISOString().slice(0, 10)
}

const initialParams = {
  fecha_inicio: '2024-10-01',
  porcentaje_gasto: 0.3,
  porcentaje_inversion: 0.1,
  porcentaje_vacaciones: 0.05,
}

export function App() {
  const errorRef = useRef(null)
  const [file, setFile] = useState(null)
  const [params, setParams] = useState(initialParams)
  const [objectives, setObjectives] = useState([])
  const [health, setHealth] = useState('pendiente')
  const [isChecking, setIsChecking] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [isLoadingObjectives, setIsLoadingObjectives] = useState(false)
  const [isSavingObjectives, setIsSavingObjectives] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [objectivesStatus, setObjectivesStatus] = useState('')
  const [selectedObjective, setSelectedObjective] = useState('all')
  const [selectedMonth, setSelectedMonth] = useState('')
  const [activeArea, setActiveArea] = useState('finanzas')
  const [isInputOpen, setIsInputOpen] = useState(true)
  const [processStatus, setProcessStatus] = useState('')
  const [checkpoint, setCheckpoint] = useState(null)

  const latest = result?.historial?.ultimo_mes
  const rows = useMemo(() => getHistoryRows(result), [result])
  const dividendPayments = useMemo(() => getDividendPayments(result), [result])
  const selectedRow = useMemo(() => getSelectedRow(rows, latest, selectedMonth), [latest, rows, selectedMonth])
  const previousRow = useMemo(() => getPreviousRow(rows, selectedRow), [rows, selectedRow])
  const recentRows = useMemo(() => getRecentRows(rows), [rows])
  const trendMax = useMemo(() => getTrendMax(recentRows), [recentRows])
  const objectiveRows = useMemo(() => getObjectiveRows(result), [result])
  const objectiveNames = useMemo(() => getObjectiveNames(objectiveRows), [objectiveRows])
  const filteredObjectiveRows = useMemo(
    () => filterObjectiveRows(objectiveRows, selectedObjective),
    [objectiveRows, selectedObjective],
  )
  const objectiveTotals = useMemo(() => getObjectiveTotals(filteredObjectiveRows), [filteredObjectiveRows])
  const activeObjectives = useMemo(() => objectives.filter((objective) => objective.nombre.trim()), [objectives])
  const validationMessages = useMemo(() => validateFinanceInput(params, activeObjectives), [activeObjectives, params])
  const hasValidationErrors = validationMessages.length > 0
  const areaSubtitle =
    activeArea === 'finanzas'
      ? 'Procesado de finanzas personales'
      : activeArea === 'invertir'
        ? 'Análisis de inversión por dividendos'
        : 'Cartera local de inversión'

  useEffect(() => {
    loadObjectives({ silent: true })
    requestJson('/api/v1/process/checkpoint', {}, '')
      .then((data) => setCheckpoint(data.result ?? null))
      .catch(() => {})
  }, [])

  // Cuando hay un checkpoint guardado, la fecha de inicio debe apuntar siempre al mes siguiente:
  // si se deja en una fecha anterior o igual, el backend deja de usar el checkpoint (para permitir
  // reprocesar/resincronizar) y el resultado se calcula desde cero, lo que da cifras muy distintas
  // si el Excel subido ya no trae el histórico completo.
  useEffect(() => {
    if (!checkpoint?.as_of_month) return
    setParams((current) => {
      if (current.fecha_inicio.slice(0, 7) > checkpoint.as_of_month) return current
      return { ...current, fecha_inicio: nextMonthStart(checkpoint.as_of_month) }
    })
  }, [checkpoint])

  useEffect(() => {
    if (error) errorRef.current?.focus()
  }, [error])

  const checkHealth = async () => {
    setIsChecking(true)
    setError('')
    try {
      const data = await requestJson('/health', {}, 'Backend no disponible')
      setHealth(data.status)
    } catch (err) {
      setHealth('error')
      setError(err.message)
    } finally {
      setIsChecking(false)
    }
  }

  const processWorkbook = async (event, modo = 'visualizar') => {
    event.preventDefault()
    if (!file) {
      setError('Selecciona un Excel antes de procesar.')
      return
    }
    if (hasValidationErrors) {
      setError(validationMessages[0])
      return
    }

    setIsProcessing(true)
    setError('')
    setProcessStatus('')
    setResult(null)

    const formData = new FormData()
    formData.append('file', file)
    formData.append('modo', modo)
    const objetivos = activeObjectives.map(objectiveToPayload)
    if (objetivos.length > 0) {
      formData.append('objetivos_json', JSON.stringify(objetivos))
    }

    const query = new URLSearchParams({
      fecha_inicio: params.fecha_inicio,
      porcentaje_gasto: String(params.porcentaje_gasto),
      porcentaje_inversion: String(params.porcentaje_inversion),
      porcentaje_vacaciones: String(params.porcentaje_vacaciones),
    })

    try {
      const data = await requestJson(
        `/api/v1/process?${query.toString()}`,
        {
          method: 'POST',
          body: formData,
        },
        'No se pudo procesar el Excel',
      )
      setResult(data.result)
      setSelectedObjective('all')
      setSelectedMonth(data.result?.historial?.ultimo_mes?.Mes ?? '')
      setIsInputOpen(false)
      const recalculadoDesdeCero = Boolean(data.recalculado_desde_cero)
      const aviso = recalculadoDesdeCero
        ? ' Se ha recalculado todo desde cero (sin continuar el histórico guardado) porque la fecha de inicio no es posterior a él: si el Excel subido no trae el histórico completo, las cifras acumuladas saldrán incompletas.'
        : ''
      if (modo === 'historico') {
        setCheckpoint(data.result?.checkpoint ?? null)
        const nuevos = data.meses_nuevos ?? []
        const yaGuardados = data.meses_ya_guardados ?? []
        const hasta = data.result?.checkpoint?.as_of_month ?? data.result?.historial?.ultimo_mes?.Mes ?? ''
        if (nuevos.length > 0) {
          setProcessStatus(`Añadidos ${nuevos.length} ${nuevos.length === 1 ? 'mes nuevo' : 'meses nuevos'} (${nuevos.join(', ')}). Histórico hasta ${hasta}.${aviso}`)
        } else if (yaGuardados.length > 0) {
          setProcessStatus(`Esos meses ya estaban guardados, no se ha duplicado nada. Histórico sincronizado hasta ${hasta}.${aviso}`)
        } else {
          setProcessStatus(`Añadido al histórico hasta ${hasta}.${aviso}`)
        }
      } else {
        setProcessStatus(`Vista previa (no se ha guardado en el histórico).${aviso}`)
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setIsProcessing(false)
    }
  }

  const updateParam = (key, value) => {
    setError('')
    setParams((current) => ({
      ...current,
      [key]: key === 'fecha_inicio' ? value : Number(value),
    }))
  }

  const addObjective = () => {
    setError('')
    setObjectives((current) => [...current, createObjective()])
  }

  const updateObjective = (id, key, value) => {
    setError('')
    setObjectives((current) =>
      current.map((objective) =>
        objective.id === id
          ? {
              ...objective,
              [key]: ['fraccion_presupuesto', 'duracion_meses', 'saldo_inicial'].includes(key) ? Number(value) : value,
            }
          : objective,
      ),
    )
  }

  const removeObjective = (id) => {
    setError('')
    setObjectives((current) => current.filter((objective) => objective.id !== id))
  }

  const loadObjectives = async ({ silent = false } = {}) => {
    setIsLoadingObjectives(true)
    setError('')
    if (!silent) setObjectivesStatus('Cargando...')
    try {
      const data = await requestJson('/api/v1/objectives', {}, 'No se pudieron cargar los objetivos')
      setObjectives((data.objetivos ?? []).map(objectiveFromApi))
      setObjectivesStatus(`${data.objetivos?.length ?? 0} objetivos cargados`)
    } catch (err) {
      setObjectivesStatus(silent ? '' : err.message)
    } finally {
      setIsLoadingObjectives(false)
    }
  }

  const saveObjectives = async () => {
    if (hasValidationErrors) {
      setError(validationMessages[0])
      setObjectivesStatus('')
      return
    }

    setIsSavingObjectives(true)
    setError('')
    setObjectivesStatus('Guardando...')

    try {
      const data = await requestJson(
        '/api/v1/objectives',
        {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ objetivos: activeObjectives.map(objectiveToPayload) }),
        },
        'No se pudieron guardar los objetivos',
      )
      setObjectives((data.objetivos ?? []).map(objectiveFromApi))
      setObjectivesStatus(`${data.objetivos?.length ?? 0} objetivos guardados`)
    } catch (err) {
      setError(err.message)
      setObjectivesStatus('')
    } finally {
      setIsSavingObjectives(false)
    }
  }

  const exportResult = () => {
    if (!result) return
    const month = selectedRow?.Mes ?? latest?.Mes ?? 'resultado'
    downloadJson(`gestiones-${month}.json`, {
      exported_at: new Date().toISOString(),
      source_file: file?.name ?? null,
      params,
      objectives: objectives.map(objectiveToPayload),
      result,
    })
  }

  const exportHistoryCsv = () => {
    if (!rows.length) return
    const month = selectedRow?.Mes ?? latest?.Mes ?? 'resultado'
    downloadText(`gestiones-historial-${month}.csv`, toCsv(rows), 'text/csv;charset=utf-8')
  }

  const exportObjectivesCsv = () => {
    if (!filteredObjectiveRows.length) return
    const label =
      selectedObjective === 'all'
        ? 'objetivos'
        : selectedObjective
            .toLowerCase()
            .replaceAll(' ', '-')
            .replaceAll('/', '-')
    downloadText(`gestiones-${label}.csv`, toCsv(filteredObjectiveRows), 'text/csv;charset=utf-8')
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-block">
          <span className="brand-mark" aria-hidden="true">
            <WalletCards size={22} strokeWidth={1.9} />
          </span>
          <div>
            <span className="brand-eyebrow">Panel personal</span>
            <h1>Gestiones</h1>
            <p>{areaSubtitle}</p>
          </div>
        </div>
        <div className="topbar-actions">
          <nav className="area-switch" aria-label="Área de trabajo">
            <button
              className={activeArea === 'finanzas' ? 'active' : ''}
              type="button"
              onClick={() => setActiveArea('finanzas')}
            >
              <ChartNoAxesCombined aria-hidden="true" size={17} />
              Finanzas
            </button>
            <button
              className={activeArea === 'invertir' ? 'active' : ''}
              type="button"
              onClick={() => setActiveArea('invertir')}
            >
              <BriefcaseBusiness aria-hidden="true" size={17} />
              Invertir
            </button>
            <button
              className={activeArea === 'cartera' ? 'active' : ''}
              type="button"
              onClick={() => setActiveArea('cartera')}
            >
              <WalletCards aria-hidden="true" size={17} />
              Cartera
            </button>
          </nav>
          {activeArea === 'finanzas' && (
            <button
              aria-expanded={isInputOpen}
              className="icon-text-button"
              onClick={() => setIsInputOpen((current) => !current)}
              title={isInputOpen ? 'Ocultar configuración del Excel' : 'Mostrar configuración del Excel'}
              type="button"
            >
              {isInputOpen ? <PanelLeftClose aria-hidden="true" size={18} /> : <PanelLeftOpen aria-hidden="true" size={18} />}
              <span>Datos Excel</span>
            </button>
          )}
          <button
            className={`backend-status status-${health}`}
            type="button"
            onClick={checkHealth}
            disabled={isChecking}
            title="Comprobar conexión con el backend"
          >
            <Server aria-hidden="true" size={16} />
            <span>{isChecking ? 'Comprobando' : health}</span>
          </button>
        </div>
      </header>

      {activeArea === 'finanzas' ? (
        <section className={isInputOpen ? 'workspace' : 'workspace input-collapsed'}>
          {isInputOpen && (
            <aside className="input-column">
              <InputPanel
                file={file}
                params={params}
                objectives={objectives}
                isLoadingObjectives={isLoadingObjectives}
                isSavingObjectives={isSavingObjectives}
                isProcessing={isProcessing}
                hasValidationErrors={hasValidationErrors}
                validationMessages={validationMessages}
                objectivesStatus={objectivesStatus}
                processStatus={processStatus}
                checkpoint={checkpoint}
                error={error}
                errorRef={errorRef}
                onFileChange={setFile}
                onParamChange={updateParam}
                onObjectiveAdd={addObjective}
                onObjectiveChange={updateObjective}
                onObjectiveRemove={removeObjective}
                onObjectivesLoad={() => loadObjectives()}
                onObjectivesSave={saveObjectives}
                onVisualize={(event) => processWorkbook(event, 'visualizar')}
                onAddToHistory={(event) => processWorkbook(event, 'historico')}
              />
            </aside>
          )}

          <ResultsPanel
            isProcessing={isProcessing}
            rows={rows}
            selectedRow={selectedRow}
            selectedObjective={selectedObjective}
            previousRow={previousRow}
            recentRows={recentRows}
            trendMax={trendMax}
            result={result}
            objectiveRows={objectiveRows}
            objectiveNames={objectiveNames}
            filteredObjectiveRows={filteredObjectiveRows}
            objectiveTotals={objectiveTotals}
            onMonthChange={setSelectedMonth}
            onObjectiveFilterChange={setSelectedObjective}
            onExportResult={exportResult}
            onExportHistoryCsv={exportHistoryCsv}
            onExportObjectivesCsv={exportObjectivesCsv}
          />
        </section>
      ) : activeArea === 'invertir' ? (
        <InvestmentPanel />
      ) : (
        <PortfolioPanel dividendPayments={dividendPayments} financeRows={rows} />
      )}
    </main>
  )
}
