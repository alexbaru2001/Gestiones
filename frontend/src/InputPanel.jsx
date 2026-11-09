import { useState } from 'react'
import { Eye, FileSpreadsheet, History, MoreHorizontal, Plus, RefreshCw, Save, Trash2 } from 'lucide-react'

export function InputPanel({
  file,
  params,
  objectives,
  isLoadingObjectives,
  isSavingObjectives,
  isProcessing,
  isDeletingHistorico,
  hasValidationErrors,
  validationMessages,
  objectivesStatus,
  processStatus,
  checkpoint,
  error,
  errorRef,
  onFileChange,
  onParamChange,
  onObjectiveAdd,
  onObjectiveChange,
  onObjectiveRemove,
  onObjectivesLoad,
  onObjectivesSave,
  onVisualize,
  onAddToHistory,
  onDeleteHistorico,
}) {
  const [isConfirmingDelete, setIsConfirmingDelete] = useState(false)

  // El reparto se escribe en enteros (30, 10, 5); por dentro sigue siendo una fracción.
  const toPercentField = (fraction) => Math.round(Number(fraction) * 1000) / 10
  const assigned =
    Number(params.porcentaje_gasto) + Number(params.porcentaje_inversion) + Number(params.porcentaje_vacaciones)
  const assignedPercent = Math.round(assigned * 1000) / 10
  const unassignedPercent = Math.round((1 - assigned) * 1000) / 10

  return (
    <form className="panel" onSubmit={onVisualize}>
      <ol className="input-steps">
        <li className="input-step">
          <div className="input-step-head">
            <span className="input-step-number" aria-hidden="true">1</span>
            <h2>Excel del mes</h2>
          </div>
          <label className="file-input">
            <input type="file" accept=".xlsx,.xlsm,.xls" onChange={(event) => onFileChange(event.target.files?.[0] ?? null)} />
            <span>
              <FileSpreadsheet aria-hidden="true" size={18} />
              {file ? 'Cambiar Excel' : 'Seleccionar Excel'}
            </span>
          </label>
          {file ? <p className="file-status selected">{file.name}</p> : null}
          <div className="input-step-note">
            <p className="checkpoint-status">
              {checkpoint?.as_of_month
                ? `Histórico guardado hasta ${checkpoint.as_of_month}. La fecha de inicio debe ser posterior.`
                : 'Todavía no hay histórico guardado: la primera vez que añadas al histórico se registra desde la fecha de inicio.'}
            </p>
            <details className="danger-menu">
              <summary aria-label="Más acciones del histórico">
                <MoreHorizontal aria-hidden="true" size={16} />
              </summary>
              <div className="danger-menu-body">
                {isConfirmingDelete ? (
                  <div className="delete-historico-confirm" role="alert">
                    <p>
                      Esto borra <strong>todo</strong> el histórico guardado (resumen mensual, checkpoint y el detalle de
                      gastos/ingresos). No se puede deshacer. Tendrás que volver a subir el Excel completo desde el principio.
                    </p>
                    <div className="process-actions">
                      <button className="secondary-button" type="button" onClick={() => setIsConfirmingDelete(false)}>
                        Cancelar
                      </button>
                      <button
                        className="danger-button"
                        type="button"
                        disabled={isDeletingHistorico}
                        onClick={() => {
                          setIsConfirmingDelete(false)
                          onDeleteHistorico()
                        }}
                      >
                        <Trash2 aria-hidden="true" size={16} />
                        {isDeletingHistorico ? 'Borrando...' : 'Sí, borrar todo'}
                      </button>
                    </div>
                  </div>
                ) : (
                  <button className="text-button danger" type="button" onClick={() => setIsConfirmingDelete(true)}>
                    <Trash2 aria-hidden="true" size={15} />
                    Borrar histórico completo
                  </button>
                )}
              </div>
            </details>
          </div>
        </li>

        <li className="input-step">
          <div className="input-step-head">
            <span className="input-step-number" aria-hidden="true">2</span>
            <h2>Periodo y reparto</h2>
          </div>
          <label className="field-row">
            <span>Fecha de inicio</span>
            <input type="date" value={params.fecha_inicio} onChange={(event) => onParamChange('fecha_inicio', event.target.value)} />
          </label>
          <label className="field-row split-row">
            <span>
              <i className="split-dot split-gasto" aria-hidden="true" />
              Gasto
            </span>
            <span className="percent-field">
              <input
                type="number"
                min="0"
                max="100"
                step="1"
                value={toPercentField(params.porcentaje_gasto)}
                onChange={(event) => onParamChange('porcentaje_gasto', Number(event.target.value) / 100)}
              />
              <em>%</em>
            </span>
          </label>
          <label className="field-row split-row">
            <span>
              <i className="split-dot split-inversion" aria-hidden="true" />
              Inversión
            </span>
            <span className="percent-field">
              <input
                type="number"
                min="0"
                max="100"
                step="1"
                value={toPercentField(params.porcentaje_inversion)}
                onChange={(event) => onParamChange('porcentaje_inversion', Number(event.target.value) / 100)}
              />
              <em>%</em>
            </span>
          </label>
          <label className="field-row split-row">
            <span>
              <i className="split-dot split-vacaciones" aria-hidden="true" />
              Vacaciones
            </span>
            <span className="percent-field">
              <input
                type="number"
                min="0"
                max="100"
                step="1"
                value={toPercentField(params.porcentaje_vacaciones)}
                onChange={(event) => onParamChange('porcentaje_vacaciones', Number(event.target.value) / 100)}
              />
              <em>%</em>
            </span>
          </label>
          <p className={unassignedPercent < 0 ? 'split-summary is-over' : 'split-summary'}>
            {unassignedPercent < 0
              ? `Asignado ${assignedPercent} %: te pasas del 100 %.`
              : `Asignado ${assignedPercent} %. Sin asignar ${unassignedPercent} %.`}
          </p>
        </li>

        <li className="input-step objectives-section">
        <div className="section-heading">
          <div className="input-step-head">
            <span className="input-step-number" aria-hidden="true">3</span>
            <h2>Objetivos</h2>
          </div>
          <div className="button-row">
            <button className="text-button" type="button" onClick={onObjectivesLoad} disabled={isLoadingObjectives}>
              <RefreshCw aria-hidden="true" size={15} />
              {isLoadingObjectives ? 'Cargando...' : 'Cargar'}
            </button>
            <button className="text-button" type="button" onClick={onObjectivesSave} disabled={isSavingObjectives || hasValidationErrors}>
              <Save aria-hidden="true" size={15} />
              {isSavingObjectives ? 'Guardando...' : 'Guardar'}
            </button>
            <button className="text-button" type="button" onClick={onObjectiveAdd}>
              <Plus aria-hidden="true" size={15} />
              Añadir
            </button>
          </div>
        </div>

        {objectives.length === 0 ? (
          <p className="muted-text">Sin objetivos configurados para este cálculo.</p>
        ) : (
          <div className="objective-list">
            {objectives.map((objective) => (
              <div className="objective-row" key={objective.id}>
                <label>
                  Nombre
                  <input type="text" value={objective.nombre} onChange={(event) => onObjectiveChange(objective.id, 'nombre', event.target.value)} />
                </label>
                <label>
                  Etiquetas
                  <input type="text" value={objective.etiquetas} onChange={(event) => onObjectiveChange(objective.id, 'etiquetas', event.target.value)} />
                </label>
                <label>
                  Fracción
                  <input
                    type="number"
                    min="0"
                    max="1"
                    step="0.01"
                    value={objective.fraccion_presupuesto}
                    onChange={(event) => onObjectiveChange(objective.id, 'fraccion_presupuesto', event.target.value)}
                  />
                </label>
                <label>
                  Meses
                  <input
                    type="number"
                    min="1"
                    step="1"
                    value={objective.duracion_meses}
                    onChange={(event) => onObjectiveChange(objective.id, 'duracion_meses', event.target.value)}
                  />
                </label>
                <label>
                  Inicio
                  <input type="month" value={objective.mes_inicio} onChange={(event) => onObjectiveChange(objective.id, 'mes_inicio', event.target.value)} />
                </label>
                <label>
                  Saldo inicial
                  <input
                    type="number"
                    step="0.01"
                    value={objective.saldo_inicial}
                    onChange={(event) => onObjectiveChange(objective.id, 'saldo_inicial', event.target.value)}
                  />
                </label>
                <button className="text-button danger" type="button" onClick={() => onObjectiveRemove(objective.id)}>
                  <Trash2 aria-hidden="true" size={15} />
                  Quitar
                </button>
              </div>
            ))}
          </div>
        )}
        {validationMessages.length > 0 && (
          <div className="validation-box" role="alert" aria-live="polite">
            {validationMessages.slice(0, 4).map((message) => (
              <p key={message}>{message}</p>
            ))}
          </div>
        )}
        {objectivesStatus && (
          <p className="status-message" aria-live="polite">
            {objectivesStatus}
          </p>
        )}
        </li>
      </ol>

      <div className="process-actions">
        <button className="primary-button" type="submit" disabled={isProcessing || hasValidationErrors}>
          <Eye aria-hidden="true" size={18} />
          {isProcessing ? 'Procesando...' : 'Visualizar'}
        </button>
        <button className="secondary-button" type="button" onClick={onAddToHistory} disabled={isProcessing || hasValidationErrors}>
          <History aria-hidden="true" size={18} />
          {isProcessing ? 'Procesando...' : 'Añadir al histórico'}
        </button>
      </div>
      <p className="process-actions-hint muted-text">
Visualizar solo muestra el resultado. Añadir al histórico lo guarda de forma permanente y no admite meses ya guardados.
      </p>

      {processStatus && !error && (
        <p className="status-message" aria-live="polite">
          {processStatus}
        </p>
      )}

      {error && (
        <p className="error-message" role="alert" tabIndex="-1" ref={errorRef}>
          {error}
        </p>
      )}
    </form>
  )
}
