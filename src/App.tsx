import React, { useEffect, useState, useCallback } from 'react'
import { invoke } from '@tauri-apps/api/core'
import { listen, Event } from '@tauri-apps/api/event'
import { open as openDialog, save as saveDialog } from '@tauri-apps/plugin-dialog'
import {
  Mic2, Image, Activity, FileText, Stethoscope,
  Layers, Settings2, PanelLeftOpen, PanelLeftClose,
} from 'lucide-react'
import { ModalityPanel } from './components/ModalityPanel'
import { FusionPanel } from './components/FusionPanel'
import { PatientContextPanel } from './components/PatientContextPanel'
import { ReportPanel } from './components/ReportPanel'
import { ModelManager } from './components/ModelManager'
import { StatusBar } from './components/StatusBar'

interface ModalityResult {
  modality: string
  success: boolean
  results?: Record<string, any>
  metadata?: Record<string, any>
  error?: string
  inference_time_ms?: number
}

interface FusionResult {
  success: boolean
  fused_results?: Record<string, any>
  modality_contributions?: Record<string, number>
  confidence?: number
  explanation?: string
  error?: string
}

interface PatientContext {
  patient_id: string
  age?: number
  sex?: string
  height_cm?: number
  weight_kg?: number
  bmi?: number
  medications?: string[]
  allergies?: string[]
  comorbidities?: string[]
}

type View = 'dashboard' | 'breath' | 'cxr' | 'spirometry' | 'fusion' | 'report' | 'models'

const VIEW_TITLES: Record<View, string> = {
  dashboard: 'Session',
  breath: 'Breath Sounds',
  cxr: 'Chest X-Ray',
  spirometry: 'Spirometry',
  fusion: 'Fusion',
  report: 'Report',
  models: 'Models',
}

const MODALITIES = [
  { id: 'breath', label: 'Breath Sounds', icon: Mic2, hint: 'wav · mp3 · flac' },
  { id: 'cxr', label: 'Chest X-Ray', icon: Image, hint: 'dcm · png · jpg' },
  { id: 'spirometry', label: 'Spirometry', icon: Activity, hint: 'json · csv' },
] as const

// every view except the three modality workspaces
const OTHER_VIEWS: { id: View; label: string; icon: React.ComponentType<{ className?: string }> }[] = [
  { id: 'fusion', label: 'Fusion', icon: Layers },
  { id: 'report', label: 'Report', icon: FileText },
  { id: 'models', label: 'Models', icon: Settings2 },
]

export function App() {
  const [view, setView] = useState<View>('dashboard')
  const [patientContext, setPatientContext] = useState<PatientContext>({
    patient_id: '',
    age: undefined,
    sex: '',
  })
  const [modalityResults, setModalityResults] = useState<Record<string, ModalityResult>>({})
  const [fusionResult, setFusionResult] = useState<FusionResult | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [gpuInfo, setGpuInfo] = useState<any>(null)
  const [deviceBusy, setDeviceBusy] = useState(false)
  const [modelsLoaded, setModelsLoaded] = useState<Record<string, boolean>>({})
  const [sidebarOpen, setSidebarOpen] = useState(true)

  const checkHealth = useCallback(async () => {
    try {
      const insideTauri = typeof window !== 'undefined' && ('__TAURI_INTERNALS__' in window || '__TAURI__' in window)
      if (!insideTauri) {
        // browser preview (vite dev server): no tauri shell, so the sidecar
        // bridge doesn't exist. show the state the app has on this machine
        // (rtx 5070 sidecar verified) so the ui is fully browsable. the
        // packaged app always uses the real health data below.
        setGpuInfo({
          status: 'healthy',
          device: 'cuda',
          cuda_available: true,
          gpu_name: 'NVIDIA GeForce RTX 5070',
          gpu_memory_total_mb: 11843,
        })
        setModelsLoaded({ breath: true, cxr: true, spirometry: true, fusion: true })
        return
      }
      const health = await invoke('python_health_check')
      setGpuInfo(health)
      const models = (await invoke('list_models')) as { plugins: { name: string; loaded: boolean }[] }
      const loaded: Record<string, boolean> = {}
      models.plugins.forEach((p) => { loaded[p.name] = p.loaded })
      setModelsLoaded(loaded)
    } catch (e) {
      console.error('Health check failed:', e)
    }
  }, [])

  useEffect(() => {
    checkHealth()
    const interval = setInterval(checkHealth, 30000)
    return () => clearInterval(interval)
  }, [checkHealth])

  // the cpu|gpu segmented control is a real switch: it pins the sidecar to
  // a device (cpu) or back to auto (its best accelerator) and re-loads the
  // plugins there. browser preview can't reach the bridge, so it stays a
  // read-only indicator there.
  const switchDevice = useCallback(async (want: 'cpu' | 'gpu') => {
    const inside = insideTauri
    if (!inside || deviceBusy) return
    setDeviceBusy(true)
    try {
      const health = await invoke('set_device', { device: want === 'gpu' ? 'auto' : 'cpu' })
      setGpuInfo(health)
    } catch (e) {
      console.error('Device switch failed:', e)
      checkHealth()
    } finally {
      setDeviceBusy(false)
    }
  }, [deviceBusy, checkHealth])

  useEffect(() => {
    const unlisten = listen('python-output', (event: Event<string>) => {
      console.log('[Python]', event.payload)
    })
    return () => {
      unlisten.then(f => f())
    }
  }, [])

  const runModalityInference = async (modality: string, filePath: string, options: Record<string, any> = {}) => {
    setIsProcessing(true)
    try {
      const result = await invoke<ModalityResult>('run_inference', {
        modality,
        input_path: filePath,
        options,
        patient_id: patientContext.patient_id || undefined,
      })
      const newer = { ...modalityResults, [modality]: result }
      setModalityResults(newer)
      // Auto-run fusion once two modalities have results
      const successes = Object.values(newer).filter(r => r?.success).length
      if (successes >= 2) {
        await runFusion(modality, result)
      }
      return result
    } catch (e) {
      console.error('Inference failed:', e)
      throw e
    } finally {
      setIsProcessing(false)
    }
  }

  const runFusion = async (newModality?: string, newResult?: ModalityResult) => {
    const merged = newModality && newResult
      ? { ...modalityResults, [newModality]: newResult }
      : modalityResults
    const successfulModalities = Object.entries(merged)
      .filter(([, r]) => r?.success)
      .reduce<Record<string, ModalityResult>>((acc, [k, v]) => ({ ...acc, [k]: v as ModalityResult }), {})

    if (Object.keys(successfulModalities).length < 2) return

    setIsProcessing(true)
    try {
      const result = await invoke<FusionResult>('run_fusion_inference', {
        modalities: successfulModalities,
        patient_context: patientContext,
        fusion_options: {}
      })
      setFusionResult(result)
    } catch (e) {
      console.error('Fusion failed:', e)
    } finally {
      setIsProcessing(false)
    }
  }

  const handleFileSelect = async (modality: string) => {
    const filters = {
      breath: [{ name: 'Audio', extensions: ['wav', 'mp3', 'flac'] }],
      cxr: [{ name: 'Images', extensions: ['dcm', 'dicom', 'png', 'jpg', 'jpeg'] }],
      spirometry: [{ name: 'Data', extensions: ['json', 'csv'] }],
    }
    const file = await openDialog({ filters: filters[modality as keyof typeof filters], multiple: false })
    if (file) {
      await runModalityInference(modality, Array.isArray(file) ? file[0] : file)
    }
  }

  const exportReport = async (format: 'pdf' | 'fhir' | 'dicom-sr' | 'json') => {
    const filePath = await saveDialog({
      filters: [{ name: format.toUpperCase(), extensions: [format === 'dicom-sr' ? 'dcm' : format] }],
      defaultPath: `stethos-report-${Date.now()}.${format === 'dicom-sr' ? 'dcm' : format}`
    })
    if (filePath) {
      const data = {
        patient_context: patientContext,
        breath: modalityResults.breath,
        cxr: modalityResults.cxr,
        spirometry: modalityResults.spirometry,
        fusion: fusionResult,
      }
      await invoke('export_report', { format, data, output_path: filePath })
    }
  }

  // the sidecar reports its torch device; anything beyond cpu is acceleration
  const deviceStr = gpuInfo?.device ?? (gpuInfo?.cuda_available ? 'cuda' : 'cpu')
  const gpuOn = deviceStr !== 'cpu'
  const insideTauri = typeof window !== 'undefined' && ('__TAURI_INTERNALS__' in window || '__TAURI__' in window)

  return (
    <div className="h-screen flex bg-[var(--bg)] text-[var(--text)] overflow-hidden">
      {/* Workbench rail */}
      <aside className={`${sidebarOpen ? 'w-64' : 'w-11'} flex-shrink-0 bg-[var(--chrome)] flex flex-col transition-all duration-150`} style={{ borderRight: '1px solid var(--border)' }}>
        <div className="h-10 flex items-center justify-between px-2.5 pr-1" style={{ borderBottom: '1px solid var(--border-soft)' }}>
          {sidebarOpen ? (
            <div className="flex items-center gap-2 min-w-0 overflow-hidden">
              <Stethoscope className="w-3.5 h-3.5 text-[var(--accent)] flex-shrink-0" strokeWidth={2.2} />
              <div className="min-w-0">
                <div className="text-[13px] font-semibold leading-tight">Stethos</div>
                <div className="text-[8.5px] uppercase tracking-[0.14em] text-[var(--text-faint)] leading-tight truncate">Respiratory ML Workstation</div>
              </div>
            </div>
          ) : (
            <Stethoscope className="w-4 h-4 text-[var(--accent)] mx-auto" strokeWidth={2.2} />
          )}
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-1 rounded-[3px] hover:bg-[var(--chrome-2)] text-[var(--text-faint)] hover:text-[var(--text)] transition-colors flex-shrink-0"
            aria-label={sidebarOpen ? 'Collapse sidebar' : 'Expand sidebar'}
          >
            {sidebarOpen
              ? <PanelLeftClose className="w-3.5 h-3.5" />
              : <PanelLeftOpen className="w-3.5 h-3.5" />}
          </button>
        </div>

        <nav className="flex-1 px-1.5 py-2 space-y-0.5 overflow-y-auto">
          {sidebarOpen && <p className="px-2 pt-1 pb-1 text-[9.5px] uppercase tracking-widest text-[var(--text-faint)]">Modalities</p>}
          {MODALITIES.map(mod => (
            <RailItem
              key={mod.id}
              icon={<mod.icon className="w-3.5 h-3.5" />}
              label={mod.label}
              hint={mod.hint}
              active={view === mod.id}
              expanded={sidebarOpen}
              onClick={() => setView(mod.id as View)}
            />
          ))}

          {sidebarOpen && <p className="px-2 pt-3 pb-1 text-[9.5px] uppercase tracking-widest text-[var(--text-faint)]">Analysis</p>}
          {OTHER_VIEWS.map(v => (
            <RailItem
              key={v.id}
              icon={<v.icon className="w-3.5 h-3.5" />}
              label={v.label}
              active={view === v.id}
              expanded={sidebarOpen}
              onClick={() => setView(v.id)}
            />
          ))}
        </nav>

        {sidebarOpen && (
          <div className="p-3" style={{ borderTop: '1px solid var(--border-soft)' }}>
            <PatientContextPanel
              context={patientContext}
              onChange={setPatientContext}
              compact={!sidebarOpen}
            />
          </div>
        )}
      </aside>

      {/* Workspace */}
      <main className="flex-1 flex flex-col overflow-hidden">
        <header className="h-10 px-3 flex items-center justify-between flex-shrink-0" style={{ borderBottom: '1px solid var(--border)', background: 'var(--chrome)' }}>
          <div className="flex items-center gap-3">
            <h2 className="text-[12px] font-medium text-[var(--text)]">{VIEW_TITLES[view]}</h2>
            {isProcessing && (
              <span className="text-[10.5px] text-[var(--text-faint)] font-mono">working…</span>
            )}
          </div>
          <div className="flex items-center gap-2.5">
            {gpuOn && (
              <span className="text-[10px] font-mono text-[var(--text-faint)] hidden md:inline">
                {gpuInfo?.gpu_name}
              </span>
            )}
            <div
              className="seg"
              title={insideTauri
                ? (gpuOn ? `Running on ${gpuInfo?.device}` : 'Running on CPU — click gpu to use the accelerator')
                : 'Device is read-only in browser preview'}
            >
              <button
                type="button"
                className={`seg-item ${!gpuOn ? 'is-active' : ''}`}
                disabled={deviceBusy}
                onClick={() => switchDevice('cpu')}
              >cpu</button>
              <button
                type="button"
                className={`seg-item ${gpuOn ? 'is-active' : ''}`}
                disabled={deviceBusy}
                onClick={() => switchDevice('gpu')}
              >gpu</button>
            </div>
          </div>
        </header>

        <div className="flex-1 overflow-auto p-4">
          {view === 'dashboard' && (
            <SessionView
              modalityResults={modalityResults}
              fusionResult={fusionResult}
              patientContext={patientContext}
              modelsLoaded={modelsLoaded}
              gpuInfo={gpuInfo}
              onQuickUpload={handleFileSelect}
              onOpenView={setView}
            />
          )}

          {view === 'breath' && (
            <ModalityPanel
              modality="breath"
              icon={Mic2}
              title="Breath Sound Analysis"
              description="Lung sound recording → normal, crackle or wheeze."
              result={modalityResults.breath}
              onRun={handleFileSelect}
              isProcessing={isProcessing}
            />
          )}

          {view === 'cxr' && (
            <ModalityPanel
              modality="cxr"
              icon={Image}
              title="Chest X-Ray Analysis"
              description="Radiograph → pathology classification, segmentation and measurements."
              result={modalityResults.cxr}
              onRun={handleFileSelect}
              isProcessing={isProcessing}
            />
          )}

          {view === 'spirometry' && (
            <ModalityPanel
              modality="spirometry"
              icon={Activity}
              title="Spirometry Analysis"
              description="Flow-volume data → pattern classification, GOLD staging, trajectory."
              result={modalityResults.spirometry}
              onRun={handleFileSelect}
              isProcessing={isProcessing}
            />
          )}

          {view === 'fusion' && (
            <FusionPanel
              results={modalityResults}
              fusionResult={fusionResult}
              patientContext={patientContext}
              onRunFusion={runFusion}
              isProcessing={isProcessing}
            />
          )}

          {view === 'report' && (
            <ReportPanel
              patientContext={patientContext}
              modalityResults={modalityResults}
              fusionResult={fusionResult}
              onExport={exportReport}
            />
          )}

          {view === 'models' && <ModelManager />}
        </div>

        <StatusBar gpuInfo={gpuInfo} modelsLoaded={modelsLoaded} />
      </main>
    </div>
  )
}

function RailItem({ icon, label, hint, active, expanded, onClick }: {
  icon: React.ReactNode
  label: string
  hint?: string
  active: boolean
  expanded: boolean
  onClick: () => void
}) {
  return (
    <button
      onClick={onClick}
      title={expanded ? undefined : label}
      className={`w-full flex items-center gap-2.5 px-2 py-[5px] rounded-[3px] text-[12px] transition-colors duration-100 text-left ${
        active
          ? 'bg-[var(--chrome-2)] text-white'
          : 'text-[var(--text-muted)] hover:bg-[var(--chrome-2)] hover:text-[var(--text)]'
      }`}
      style={active ? { boxShadow: 'inset 2px 0 0 var(--accent)' } : undefined}
    >
      <span className={`flex-shrink-0 ${active ? 'text-[var(--text)]' : 'text-[var(--text-faint)]'}`}>{icon}</span>
      {expanded && (
        <>
          <span className="truncate flex-1">{label}</span>
          {hint && <span className="text-[9.5px] font-mono text-[var(--text-faint)]">{hint}</span>}
        </>
      )}
    </button>
  )
}

function SessionView({
  modalityResults,
  fusionResult,
  patientContext,
  modelsLoaded,
  gpuInfo,
  onQuickUpload,
  onOpenView,
}: {
  modalityResults: Record<string, ModalityResult>
  fusionResult: FusionResult | null
  patientContext: PatientContext
  modelsLoaded: Record<string, boolean>
  gpuInfo: { device?: string; cuda_available?: boolean } | null
  onQuickUpload: (mod: string) => void
  onOpenView: (v: View) => void
}) {
  const completedCount = Object.values(modalityResults).filter(r => r.success).length
  const hasFusion = fusionResult?.success
  const loadedCount = Object.values(modelsLoaded).filter(Boolean).length

  return (
    <div className="max-w-4xl mx-auto space-y-3 animate-fade-in">
      <div className="panel">
        <div className="panel-header" style={{ background: 'var(--panel)' }}>
          <span className="text-[12px] text-[var(--text)]">
            {patientContext.patient_id || <span className="text-[var(--text-faint)]">no patient id set</span>}
            {patientContext.age && <span className="text-[var(--text-faint)]"> · {patientContext.age} yr</span>}
            {patientContext.sex && <span className="text-[var(--text-faint)]"> · {patientContext.sex}</span>}
          </span>
          <div className="flex items-center gap-2 font-mono text-[10.5px] text-[var(--text-faint)]">
            <span>models {loadedCount}/{Object.keys(modelsLoaded).length}</span>
            <span>·</span>
            <span>{gpuInfo?.device ?? (gpuInfo?.cuda_available ? 'cuda' : 'cpu')}</span>
          </div>
        </div>

        {/* modality rows — the workbench table */}
        <div>
          {MODALITIES.map((mod, i) => {
            const r = modalityResults[mod.id]
            return (
              <div
                key={mod.id}
                className="table-row-hover flex items-center gap-3 px-3.5 py-2.5"
                style={i > 0 ? { borderTop: '1px solid var(--border-soft)' } : undefined}
              >
                <mod.icon className="w-4 h-4 text-[var(--text-faint)] flex-shrink-0" />
                <div className="w-36 flex-shrink-0">
                  <p className="text-[12px]">{mod.label}</p>
                  <p className="text-[9.5px] font-mono text-[var(--text-faint)]">{mod.hint}</p>
                </div>
                <div className="flex-1 min-w-0 text-[11px] text-[var(--text-muted)] truncate">
                  {r?.error ? (
                    <span className="text-[var(--bad)]">{r.error}</span>
                  ) : r?.success ? (
                    <ModalityPreview modality={mod.id} result={r} />
                  ) : isRowPending(mod.id) ? null : (
                    <span className="text-[var(--text-faint)]">—</span>
                  )}
                </div>
                {r?.success && r?.inference_time_ms && (
                  <span className="font-mono text-[10px] text-[var(--text-faint)]">{r.inference_time_ms.toFixed(0)} ms</span>
                )}
                {r?.success && <span className="badge badge-success flex-shrink-0">done</span>}
                <button
                  onClick={() => onQuickUpload(mod.id)}
                  className="btn-secondary flex-shrink-0"
                >
                  {r?.success ? 're-run' : 'analyze'}
                </button>
              </div>
            )
          })}
        </div>

        {/* fusion row */}
        <div className="flex items-center gap-3 px-3.5 py-2.5" style={{ borderTop: '1px solid var(--border-soft)' }}>
          <Layers className="w-4 h-4 text-[var(--text-faint)] flex-shrink-0" />
          <div className="w-36 flex-shrink-0">
            <p className="text-[12px]">Fusion</p>
            <p className="text-[9.5px] font-mono text-[var(--text-faint)]">≥2 modalities</p>
          </div>
          <div className="flex-1 min-w-0 text-[11px] text-[var(--text-muted)]">
            {hasFusion ? (
              <span>
                {fusionResult.fused_results?.risk_level ?? '—'} risk ·{' '}
                {Object.entries(fusionResult.modality_contributions || {})
                  .map(([m, v]) => `${m.replace('_', ' ')} ${Math.round(v * 100)}%`)
                  .join(' · ')}
              </span>
            ) : (
              <span className="text-[var(--text-faint)]">
                {completedCount >= 2 ? 'ready — open the fusion view' : `${completedCount}/2 modalities to enable`}
              </span>
            )}
          </div>
          {hasFusion && <span className="badge badge-info flex-shrink-0">fused</span>}
          <button
            onClick={() => onOpenView('fusion')}
            className="btn-ghost flex-shrink-0"
          >
            open
          </button>
        </div>
      </div>

      {hasFusion && fusionResult.explanation && (
        <div className="panel">
          <div className="panel-header">
            <span className="panel-title">Fusion Assessment</span>
            <span className={`badge ${
              ['Very High', 'High'].includes(fusionResult.fused_results?.risk_level) ? 'badge-danger'
              : fusionResult.fused_results?.risk_level === 'Moderate' ? 'badge-warning'
              : 'badge-success'
            }`}>{fusionResult.fused_results?.risk_level} risk</span>
          </div>
          <div className="p-3.5">
            <p className="text-[12px] text-[var(--text-muted)] leading-relaxed">{fusionResult.explanation}</p>
          </div>
        </div>
      )}

      <p className="text-[10.5px] text-[var(--text-faint)] px-1">
        research tool — not a medical device. all inference runs locally.
      </p>
    </div>
  )
}

function isRowPending(_modality: string): boolean {
  return false
}

function ModalityPreview({ modality, result }: { modality: string; result: ModalityResult }) {
  const r = result.results
  if (!r) return null

  switch (modality) {
    case 'breath':
      return (
        <span className="font-mono">
          <span>{r.prediction}</span>
          {r.probabilities?.[r.prediction] && (
            <span className="text-[var(--text-faint)]"> {Math.round(r.probabilities[r.prediction] * 100)}%</span>
          )}
        </span>
      )
    case 'cxr':
      return (
        <span>
          {r.positive_findings?.length
            ? `${r.positive_findings.length} finding${r.positive_findings.length > 1 ? 's' : ''}: ${r.positive_findings.join(', ')}`
            : 'no positive findings'}
          {r.measurements?.cardiothoracic_ratio && (
            <span className="font-mono text-[var(--text-faint)]"> · ctr {r.measurements.cardiothoracic_ratio.toFixed(2)}</span>
          )}
        </span>
      )
    case 'spirometry':
      return (
        <span>
          {r.pattern}
          {r.gold_stage && <span className="font-mono text-[var(--text-faint)]"> · {r.gold_stage}</span>}
          <span className="font-mono text-[var(--text-faint)]"> · fev1/fvc {r.fev1_fvc_ratio?.toFixed(2)}</span>
        </span>
      )
    default:
      return null
  }
}

export { isRowPending }
