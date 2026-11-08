import React, { useEffect, useState, useCallback, useRef } from 'react'
import { invoke } from '@tauri-apps/api/core'
import { listen, Event } from '@tauri-apps/api/event'
import { open as openDialog, save as saveDialog } from '@tauri-apps/plugin-dialog'
import {
  Mic2, Image, Activity, FileText,
  Layers, Settings2, PanelLeftOpen, PanelLeftClose, History,
  Users,
} from 'lucide-react'
import { ModalityPanel } from './components/ModalityPanel'
import { FusionPanel } from './components/FusionPanel'
import { PatientContextPanel } from './components/PatientContextPanel'
import { ReportPanel } from './components/ReportPanel'
import { ModelManager } from './components/ModelManager'
import { StatusBar } from './components/StatusBar'
import { HistoryPanel } from './components/HistoryPanel'
import { StreamingInference } from './components/StreamingInference'
import { addStudy, basename, summarizeResult } from './lib/studies'
import { extractProbabilities } from './lib/confidence'

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

type View = 'dashboard' | 'breath' | 'cxr' | 'spirometry' | 'fusion' | 'history' | 'report' | 'models'

const VIEW_TITLES: Record<View, string> = {
  dashboard: 'Session',
  breath: 'Breath Sounds',
  cxr: 'Chest X-Ray',
  spirometry: 'Spirometry',
  fusion: 'Fusion',
  history: 'History & Compare',
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
  { id: 'history', label: 'History', icon: History },
  { id: 'report', label: 'Report', icon: FileText },
  { id: 'models', label: 'Models', icon: Settings2 },
]

export function App() {
  const [view, setView] = useState<View>('dashboard')

  // Load patient registry and current patient from localStorage
  const [patientRegistry, setPatientRegistry] = useState<Record<string, PatientContext>>(() => {
    try { return JSON.parse(localStorage.getItem('stethos.patients') || '{}') } catch { return {} }
  })
  const [patientContext, setPatientContext] = useState<PatientContext>(() => {
    try {
      const currentId = localStorage.getItem('stethos.currentPatient')
      const registry = JSON.parse(localStorage.getItem('stethos.patients') || '{}')
      if (currentId && registry[currentId]) return registry[currentId]
    } catch {}
    return { patient_id: '', age: undefined, sex: '' }
  })

  // Track whether a patient has been selected (for first-launch welcome screen)
  const [patientSelected, setPatientSelected] = useState(() => {
    try {
      return !!localStorage.getItem('stethos.currentPatient')
    } catch { return false }
  })

  // Session data is now per-patient: load from stethos.session.{patient_id}
  const [modalityResults, setModalityResults] = useState<Record<string, ModalityResult>>(() => {
    if (!patientContext.patient_id) return {}
    try {
      return JSON.parse(localStorage.getItem(`stethos.session.${patientContext.patient_id}`) || '{}').results || {}
    } catch { return {} }
  })
  const [fusionResult, setFusionResult] = useState<FusionResult | null>(() => {
    if (!patientContext.patient_id) return null
    try {
      return JSON.parse(localStorage.getItem(`stethos.session.${patientContext.patient_id}`) || '{}').fusion || null
    } catch { return null }
  })

  // Persist patient registry
  useEffect(() => {
    try { localStorage.setItem('stethos.patients', JSON.stringify(patientRegistry)) } catch {}
  }, [patientRegistry])

  // Persist current patient selection
  useEffect(() => {
    localStorage.setItem('stethos.currentPatient', patientContext.patient_id || '')
    if (patientContext.patient_id) setPatientSelected(true)
  }, [patientContext.patient_id])

  // Persist session data per patient
  useEffect(() => {
    if (!patientContext.patient_id) return
    try {
      localStorage.setItem(`stethos.session.${patientContext.patient_id}`, JSON.stringify({
        at: Date.now(),
        results: modalityResults,
        fusion: fusionResult,
      }))
    } catch { /* storage full — session persistence is best-effort */ }
  }, [patientContext.patient_id, modalityResults, fusionResult])

  // When patientContext changes (switch patient), reload that patient's session
  useEffect(() => {
    if (!patientContext.patient_id) {
      setModalityResults({})
      setFusionResult(null)
      return
    }
    try {
      const session = JSON.parse(localStorage.getItem(`stethos.session.${patientContext.patient_id}`) || '{}')
      setModalityResults(session.results || {})
      setFusionResult(session.fusion || null)
    } catch {
      setModalityResults({})
      setFusionResult(null)
    }
  }, [patientContext.patient_id])

  // capture every js error into a local diagnostics ring so crashes that
  // reset ui state leave a trace we can inspect
  useEffect(() => {
    const record = (msg: string, stack: string) => {
      try {
        const list: any[] = JSON.parse(localStorage.getItem('stethos.errors') || '[]')
        list.push({ at: Date.now(), msg, stack: stack.slice(0, 400) })
        localStorage.setItem('stethos.errors', JSON.stringify(list.slice(-20)))
      } catch { /* ignore */ }
      // eslint-disable-next-line no-console
      console.error('[stethos diagnostics]', msg)
    }
    const onErr = (e: ErrorEvent) => record(e.message || 'error', ((e as any).error?.stack as string) || '')
    const onRej = (e: PromiseRejectionEvent) => record('unhandled rejection: ' + String(e.reason).slice(0, 300), '')
    window.addEventListener('error', onErr)
    window.addEventListener('unhandledrejection', onRej)
    return () => {
      window.removeEventListener('error', onErr)
      window.removeEventListener('unhandledrejection', onRej)
    }
  }, [])
  const [isProcessing, setIsProcessing] = useState(false)
  const [gpuInfo, setGpuInfo] = useState<any>(null)
  const [deviceBusy, setDeviceBusy] = useState(false)
  const [modelsLoaded, setModelsLoaded] = useState<Record<string, boolean>>({})
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [backendNote, setBackendNote] = useState<string | null>(null)
  // always-current mirror of modalityResults for callbacks that fire long
  // after their render (slow inference resolving over a newer run otherwise
  // clobbers the other modality's result)
  const modalityResultsRef = useRef<Record<string, ModalityResult>>({})

  // returns true when the sidecar bridge is healthy. with recover=true a failed
  // probe triggers an actual restart: SidecarManager adopts an already-healthy
  // sidecar or spawns a fresh one, so a killed/crashed backend self-heals.
  const checkHealth = useCallback(async (recover = false): Promise<boolean> => {
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
        return true
      }
      let health: any
      try {
        health = await invoke('python_health_check')
      } catch (probeErr) {
        if (!recover) throw probeErr
        console.warn('sidecar probe failed, attempting restart:', probeErr)
        await invoke('start_sidecar')
        health = await invoke('python_health_check')
      }
      setGpuInfo(health)
      const models = (await invoke('list_models')) as { plugins: { name: string; loaded: boolean }[] }
      const loaded: Record<string, boolean> = {}
      models.plugins.forEach((p) => { loaded[p.name] = p.loaded })
      setModelsLoaded(loaded)
      setBackendNote(null)

      // sticky device mode: a sidecar restart/reset drops its device override,
      // re-pin gpu whenever the sidecar reports cpu while the user chose gpu
      const pinned = localStorage.getItem('stethos.device')
      if (pinned === 'gpu' && health.device === 'cpu' && health.cuda_available) {
        try {
          const h2 = await invoke('set_device', { device: 'auto' })
          setGpuInfo(h2)
        } catch (e) { console.warn('device re-pin failed (retries next poll):', e) }
      }
      return true
    } catch (e) {
      console.error('Health check failed:', e)
      return false
    }
  }, [])

  useEffect(() => {
    let cancelled = false
    const boot = async () => {
      // first-run: the sidecar spawns and loads weights, which can take a
      // while. retry instead of leaving the ui in a dead-end state.
      for (let i = 0; i < 8 && !cancelled; i++) {
        if (await checkHealth(true)) return
        await new Promise((r) => setTimeout(r, 2000))
      }
      if (!cancelled) setBackendNote('unreachable')
    }
    boot()
    // keep-healing: if the sidecar dies later, the next poll restarts it
    const interval = setInterval(() => { checkHealth(true) }, 30000)
    return () => { cancelled = true; clearInterval(interval) }
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
      localStorage.setItem('stethos.device', want) // survives app restarts
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
    // the tauri event bridge only exists inside the app shell; calling listen()
    // in browser preview throws an unhandled rejection at startup
    const insideTauri = typeof window !== 'undefined' && ('__TAURI_INTERNALS__' in window || '__TAURI__' in window)
    if (!insideTauri) return
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
      // Tauri v2 matches invoke args by camelCase of the rust parameter name
      // (input_path -> inputPath); snake_case keys silently fail with
      // "missing required key ..." from inside the dialog/drop flow.
      const result = await invoke<ModalityResult>('run_inference', {
        modality,
        inputPath: filePath,
        options,
      })
      // functional update: never writes from a stale snapshot, so a slow run
      // finishing late can't wipe another modality's result
      setModalityResults(prev => {
        const next = { ...prev, [modality]: result }
        modalityResultsRef.current = next
        return next
      })

      // record the run in the local history timeline
      if (result?.success) {
        try {
          addStudy({
            id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
            at: Date.now(),
            modality: modality as any,
            patient_id: patientContext.patient_id || 'unassigned',
            file_name: basename(filePath),
            ok: true,
            summary: summarizeResult(modality, result),
            probabilities: ((): Record<string, number> | undefined => {
              const p = extractProbabilities(modality, result.results)
              return p ?? undefined
            })(),
            detail: result.results,
            duration_ms: result.inference_time_ms,
          })
        } catch (e) { console.error('history record failed', e) }
      }

      // Auto-run fusion once two modalities have results
      const newer = modalityResultsRef.current
      const successes = Object.values(newer).filter(r => r?.success).length
      if (successes >= 2) {
        await runFusion(modality, result)
      }
      return result
    } catch (e) {
      console.error('Inference failed:', e)
      // surface the failure for this modality instead of silently doing
      // nothing — the panel renders result.error
      const err = e instanceof Error ? e.message : String(e)
      setModalityResults(prev => {
        const next = { ...prev, [modality]: { modality, success: false, error: err } as ModalityResult }
        modalityResultsRef.current = next
        return next
      })
      throw e
    } finally {
      setIsProcessing(false)
    }
  }

  const runFusion = async (newModality?: string, newResult?: ModalityResult) => {
    // read from the always-current ref, not a possibly-stale closure
    const merged = newModality && newResult
      ? { ...modalityResultsRef.current, [newModality]: newResult }
      : modalityResultsRef.current
    const successfulModalities = Object.entries(merged)
      .filter(([, r]) => r?.success)
      .reduce<Record<string, ModalityResult>>((acc, [k, v]) => ({ ...acc, [k]: v as ModalityResult }), {})

    if (Object.keys(successfulModalities).length < 2) return

    setIsProcessing(true)
    try {
      // run_fusion_inference's rust signature is a single `inputs` object;
      // Tauri v2 arg casing also requires camelCase nested keys stay as the
      // python FusionRequest expects (snake_case fields inside the payload).
      const result = await invoke<FusionResult>('run_fusion_inference', {
        inputs: {
          modalities: successfulModalities,
          patient_context: patientContext,
          fusion_options: {},
        },
      })
      setFusionResult(result)

      // fusion runs are timeline entries too
      if (result?.success) {
        try {
          addStudy({
            id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
            at: Date.now(),
            modality: 'fusion',
            patient_id: patientContext.patient_id || 'unassigned',
            ok: true,
            summary: summarizeResult('fusion', result),
            detail: { ...(result.fused_results || {}), explanation: result.explanation },
            duration_ms: undefined,
          })
        } catch (e) { console.error('history record failed', e) }
      }
    } catch (e) {
      console.error('Fusion failed:', e)
    } finally {
      setIsProcessing(false)
    }
  }

  const handleFileSelect = async (modality: string, filePath?: string) => {
    console.log('[handleFileSelect] called with modality:', modality, 'filePath:', filePath)
    // dropped files arrive with an absolute path (tauri injects it) — skip the
    // dialog and run directly on them
    if (filePath) {
      console.log('[handleFileSelect] using dropped file path')
      await runModalityInference(modality, filePath)
      return
    }
    const filters = {
      breath: [{ name: 'Audio', extensions: ['wav', 'mp3', 'flac'] }],
      cxr: [{ name: 'Images', extensions: ['dcm', 'dicom', 'png', 'jpg', 'jpeg'] }],
      spirometry: [{ name: 'Data', extensions: ['json', 'csv'] }],
    }
    try {
      const file = await openDialog({ filters: filters[modality as keyof typeof filters], multiple: false })
      console.log('[handleFileSelect] openDialog returned:', file)
      if (file) {
        await runModalityInference(modality, Array.isArray(file) ? file[0] : file)
      }
    } catch (e) {
      console.error('[handleFileSelect] openDialog error:', e)
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
      await invoke('export_report', { format, data, outputPath: filePath })
    }
  }

  // the sidecar reports its torch device; anything beyond cpu is acceleration
  const deviceStr = gpuInfo?.device ?? (gpuInfo?.cuda_available ? 'cuda' : 'cpu')
  const gpuOn = deviceStr !== 'cpu'
  const insideTauri = typeof window !== 'undefined' && ('__TAURI_INTERNALS__' in window || '__TAURI__' in window)

  // Welcome screen — shown on first launch before any patient is selected
  if (!patientSelected) {
    return (
      <div className="h-screen flex bg-[var(--bg)] text-[var(--text)] overflow-hidden">
        <main className="flex-1 flex flex-col items-center justify-center p-8">
          <div className="max-w-md w-full text-center animate-fade-in">
            <img
              src="/logo-light.png"
              alt="Stethos"
              className="w-20 h-20 mx-auto mb-5 opacity-95"
              draggable={false}
            />
            <h1 className="text-3xl font-semibold text-white mb-2">Welcome to Stethos</h1>
            <p className="text-[var(--text-muted)] mb-8">
              Local ML diagnostics for respiratory analysis. Start by creating or selecting a patient.
            </p>

            {/* Existing patients */}
            {Object.keys(patientRegistry).length > 0 && (
              <div className="mb-6 space-y-3">
                <p className="text-sm text-[var(--text-faint)] text-left font-medium">Existing Patients</p>
                {Object.entries(patientRegistry).map(([id, p]) => (
                  <button
                    key={id}
                    onClick={() => { setPatientContext(p); setPatientSelected(true); }}
                    className="w-full text-left p-4 rounded-[4px] border border-[var(--border)] bg-[var(--chrome)] hover:bg-[var(--chrome-2)] hover:border-[var(--accent)] transition-all"
                  >
                    <div className="font-mono text-lg">{id}</div>
                    <div className="text-sm text-[var(--text-faint)] mt-1">
                      {p.age && `${p.age}yr · `}{p.sex && `${p.sex} · `}
                      {p.bmi && `bmi ${p.bmi}`}
                    </div>
                  </button>
                ))}
              </div>
            )}

            {/* Create new patient */}
            <div className="border-t border-[var(--border-soft)] pt-6">
              <p className="text-sm text-[var(--text-faint)] text-left font-medium mb-3">Create New Patient</p>
              <NewPatientForm
                registry={patientRegistry}
                onCreate={(id, ctx) => {
                  setPatientRegistry(prev => ({ ...prev, [id]: ctx }))
                  setPatientContext(ctx)
                  setPatientSelected(true)
                }}
              />
            </div>

            <p className="mt-8 text-xs text-[var(--text-faint)]">
              research tool — not a medical device. all inference runs locally.
            </p>
          </div>
        </main>
      </div>
    )
  }

  return (
    <div className="h-screen flex bg-[var(--bg)] text-[var(--text)] overflow-hidden">
      {/* Workbench rail */}
      <aside className={`${sidebarOpen ? 'w-64' : 'w-11'} flex-shrink-0 bg-[var(--chrome)] flex flex-col transition-all duration-150`} style={{ borderRight: '1px solid var(--border)' }}>
        <div className="h-10 flex items-center justify-between px-2.5 pr-1" style={{ borderBottom: '1px solid var(--border-soft)' }}>
          {sidebarOpen ? (
            <div className="flex items-center gap-2 min-w-0 overflow-hidden">
              <img src="/logo-light.png" alt="" className="w-5 h-5 flex-shrink-0" draggable={false} />
              <div className="min-w-0">
                <div className="text-[13px] font-semibold leading-tight">Stethos</div>
                <div className="text-[8.5px] uppercase tracking-[0.14em] text-[var(--text-faint)] leading-tight truncate">Local ML Diagnostics</div>
              </div>
            </div>
          ) : (
            <img src="/logo-light.png" alt="Stethos" className="w-6 h-6 mx-auto" draggable={false} />
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
              registry={patientRegistry}
              currentPatientId={patientContext.patient_id}
              onSwitchPatient={(id) => {
                const p = patientRegistry[id]
                if (p) setPatientContext(p)
              }}
              onCreatePatient={(id, ctx) => {
                setPatientRegistry(prev => ({ ...prev, [id]: ctx }))
                setPatientContext(ctx)
              }}
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
            {/* Switch Patient Button — go back to welcome screen to select/create patients */}
            <button
              onClick={() => { setPatientSelected(false); setPatientContext({ patient_id: '', age: undefined, sex: '' }); }}
              className="px-2 py-1 text-[10.5px] font-mono bg-[var(--inset)] border border-[var(--border)] rounded-[3px] text-[var(--text-faint)] hover:text-[var(--text)] hover:border-[var(--accent)] transition-colors"
              title="Switch patient"
            >
              <Users className="w-3.5 h-3.5 mr-1.5" strokeWidth={1.8} />
              Switch Patient
            </button>
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
            <div className="flex-1 flex flex-col">
              <ModalityPanel
                modality="breath"
                icon={Mic2}
                title="Breath Sound Analysis"
                description="Lung sound recording → normal, crackle or wheeze."
                result={modalityResults.breath}
                patientContext={patientContext}
                onRun={handleFileSelect}
                isProcessing={isProcessing}
              />
              <div className="border-t border-[var(--border-soft)] p-4">
                <StreamingInference
                  modality="breath"
                  onResult={(result) => {
                    // Could auto-trigger inference on streaming results
                    console.log('Streaming result:', result)
                  }}
                  autoStart={false}
                />
              </div>
            </div>
          )}

          {view === 'cxr' && (
            <ModalityPanel
              modality="cxr"
              icon={Image}
              title="Chest X-Ray Analysis"
              description="Radiograph → pathology classification, segmentation and measurements."
              result={modalityResults.cxr}
              patientContext={patientContext}
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
              patientContext={patientContext}
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

          {view === 'history' && (
            <HistoryPanel patientId={patientContext.patient_id} />
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

        <StatusBar gpuInfo={gpuInfo} modelsLoaded={modelsLoaded} note={backendNote} />
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

function NewPatientForm({ registry, onCreate }: { registry: Record<string, PatientContext>; onCreate: (id: string, ctx: PatientContext) => void }) {
  const [id, setId] = useState('')
  const [age, setAge] = useState('')
  const [sex, setSex] = useState('')
  const [height, setHeight] = useState('')
  const [weight, setWeight] = useState('')
  const [error, setError] = useState('')

  const nextId = `PAT-${String(Object.keys(registry).length + 1).padStart(3, '0')}`

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    if (!id.trim()) { setError('Patient ID is required'); return }
    if (registry[id.trim()]) { setError('Patient ID already exists'); return }

    const ageNum = parseInt(age) || undefined
    const heightNum = parseInt(height) || undefined
    const weightNum = parseInt(weight) || undefined
    const bmi = heightNum && weightNum ? parseFloat((weightNum / ((heightNum / 100) ** 2)).toFixed(1)) : undefined

    onCreate(id.trim(), {
      patient_id: id.trim(),
      age: ageNum,
      sex: sex.toUpperCase(),
      height_cm: heightNum,
      weight_kg: weightNum,
      bmi,
    })
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-3 text-left">
      <div className="grid grid-cols-2 gap-3">
        <input
          type="text"
          value={id}
          onChange={e => setId(e.target.value)}
          placeholder={nextId}
          className="input text-sm"
        />
        <input
          type="number"
          value={age}
          onChange={e => setAge(e.target.value)}
          placeholder="Age"
          className="input text-sm"
        />
      </div>
      <div className="grid grid-cols-2 gap-3">
        <select value={sex} onChange={e => setSex(e.target.value)} className="input text-sm">
          <option value="">Sex</option>
          <option value="M">Male</option>
          <option value="F">Female</option>
          <option value="O">Other</option>
        </select>
        <input
          type="number"
          value={height}
          onChange={e => setHeight(e.target.value)}
          placeholder="Height (cm)"
          className="input text-sm"
        />
      </div>
      <div className="grid grid-cols-2 gap-3">
        <input
          type="number"
          value={weight}
          onChange={e => setWeight(e.target.value)}
          placeholder="Weight (kg)"
          className="input text-sm"
        />
        <div></div>
      </div>
      {error && <p className="text-xs text-red-400">{error}</p>}
      <button type="submit" className="btn-primary w-full py-2 mt-2">
        Create Patient
      </button>
    </form>
  )
}
