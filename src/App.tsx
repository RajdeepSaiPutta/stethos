import React, { useEffect, useState, useCallback } from 'react'
import { invoke } from '@tauri-apps/api/core'
import { listen, Event } from '@tauri-apps/api/event'
import { open as openDialog, save as saveDialog } from '@tauri-apps/plugin-dialog'
import {
  LayoutDashboard, Mic2, Image, Activity, Brain, FileText,
  CheckCircle, ChevronRight, ChevronLeft, Stethoscope,
  BrainCircuit, Zap, Settings2,
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

const MODALITIES = [
  { id: 'breath', label: 'Breath Sounds', icon: Mic2, color: 'text-emerald-400', bg: 'bg-emerald-500/10' },
  { id: 'cxr', label: 'Chest X-Ray', icon: Image, color: 'text-blue-400', bg: 'bg-blue-500/10' },
  { id: 'spirometry', label: 'Spirometry', icon: Activity, color: 'text-amber-400', bg: 'bg-amber-500/10' },
] as const

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
  const [modelsLoaded, setModelsLoaded] = useState<Record<string, boolean>>({})
  const [sidebarOpen, setSidebarOpen] = useState(true)

  const checkHealth = useCallback(async () => {
    try {
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
      defaultPath: `respiratory-report-${Date.now()}.${format === 'dicom-sr' ? 'dcm' : format}`
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

  return (
    <div className="h-screen flex bg-gray-950 text-gray-100 overflow-hidden">
      {/* Sidebar */}
      <aside className={`${sidebarOpen ? 'w-72' : 'w-20'} flex-shrink-0 bg-gray-900 border-r border-gray-800 flex flex-col transition-all duration-200`}>
        <div className="p-4 border-b border-gray-800 flex items-center justify-between">
          {sidebarOpen && (
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-lg bg-emerald-500/20 flex items-center justify-center">
                <Stethoscope className="w-5 h-5 text-emerald-400" />
              </div>
              <div>
                <h1 className="font-bold text-white text-lg">Respiratory AI</h1>
                <p className="text-xs text-gray-500">Workstation v0.1</p>
              </div>
            </div>
          )}
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-1.5 rounded-lg hover:bg-gray-800 transition-colors"
            aria-label={sidebarOpen ? 'Collapse sidebar' : 'Expand sidebar'}
          >
            {sidebarOpen ? <ChevronLeft className="w-5 h-5 text-gray-400" /> : <ChevronRight className="w-5 h-5 text-gray-400" />}
          </button>
        </div>

        <nav className="flex-1 p-3 space-y-1 overflow-y-auto">
          <button
            onClick={() => setView('dashboard')}
            className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-150 ${
              view === 'dashboard'
                ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30'
                : 'text-gray-400 hover:bg-gray-800 hover:text-white'
            }`}
          >
            <LayoutDashboard className="w-5 h-5 flex-shrink-0" />
            {sidebarOpen && <span>Dashboard</span>}
          </button>

          {MODALITIES.map(mod => (
            <button
              key={mod.id}
              onClick={() => setView(mod.id as View)}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-150 ${
                view === mod.id
                  ? `${mod.bg} ${mod.color} border border-current/30`
                  : 'text-gray-400 hover:bg-gray-800 hover:text-white'
              }`}
            >
              <mod.icon className={`w-5 h-5 flex-shrink-0 ${mod.color}`} />
              {sidebarOpen && <span>{mod.label}</span>}
            </button>
          ))}

          <button
            onClick={() => setView('fusion')}
            className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-150 ${
              view === 'fusion'
                ? 'bg-purple-500/10 text-purple-400 border border-purple-500/30'
                : 'text-gray-400 hover:bg-gray-800 hover:text-white'
            }`}
          >
            <BrainCircuit className="w-5 h-5 flex-shrink-0 text-purple-400" />
            {sidebarOpen && <span>Fusion</span>}
          </button>

          <button
            onClick={() => setView('report')}
            className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-150 ${
              view === 'report'
                ? 'bg-blue-500/10 text-blue-400 border border-blue-500/30'
                : 'text-gray-400 hover:bg-gray-800 hover:text-white'
            }`}
          >
            <FileText className="w-5 h-5 flex-shrink-0 text-blue-400" />
            {sidebarOpen && <span>Report</span>}
          </button>

          <button
            onClick={() => setView('models')}
            className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-150 ${
              view === 'models'
                ? 'bg-gray-700 text-gray-200 border border-gray-600'
                : 'text-gray-400 hover:bg-gray-800 hover:text-white'
            }`}
          >
            <Settings2 className="w-5 h-5 flex-shrink-0 text-gray-400" />
            {sidebarOpen && <span>Models</span>}
          </button>
        </nav>

        <div className="p-3 border-t border-gray-800">
          <PatientContextPanel
            context={patientContext}
            onChange={setPatientContext}
            compact={!sidebarOpen}
          />
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col overflow-hidden">
        <header className="h-14 px-6 border-b border-gray-800 flex items-center justify-between bg-gray-900/50 backdrop-blur-sm">
          <div className="flex items-center gap-4">
            <h2 className="text-xl font-semibold text-white capitalize">{view}</h2>
            {isProcessing && (
              <div className="flex items-center gap-2 text-emerald-400">
                <div className="w-4 h-4 border-2 border-emerald-400 border-t-transparent rounded-full animate-spin" />
                <span className="text-sm">Processing...</span>
              </div>
            )}
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 text-xs text-gray-500">
              <span className={`w-2 h-2 rounded-full ${gpuInfo?.cuda_available ? 'bg-green-400' : 'bg-gray-500'}`} />
              {gpuInfo?.cuda_available ? (
                <>
                  {gpuInfo?.gpu_name} · {Math.round(gpuInfo?.gpu_memory_total_mb || 0)} MB
                </>
              ) : (
                'CPU Mode'
              )}
            </div>
          </div>
        </header>

        <div className="flex-1 overflow-auto p-6">
          {view === 'dashboard' && (
            <DashboardView
              modalityResults={modalityResults}
              fusionResult={fusionResult}
              patientContext={patientContext}
              modelsLoaded={modelsLoaded}
              gpuInfo={gpuInfo}
              onQuickUpload={handleFileSelect}
            />
          )}

          {view === 'breath' && (
            <ModalityPanel
              modality="breath"
              icon={Mic2}
              color="emerald"
              title="Breath Sound Analysis"
              description="Upload a WAV/MP3/FLAC recording of breath sounds for classification"
              result={modalityResults.breath}
              onRun={handleFileSelect}
              isProcessing={isProcessing}
            />
          )}

          {view === 'cxr' && (
            <ModalityPanel
              modality="cxr"
              icon={Image}
              color="blue"
              title="Chest X-Ray Analysis"
              description="Upload a DICOM/PNG/JPEG chest X-ray for pathology classification, segmentation, and measurements"
              result={modalityResults.cxr}
              onRun={handleFileSelect}
              isProcessing={isProcessing}
            />
          )}

          {view === 'spirometry' && (
            <ModalityPanel
              modality="spirometry"
              icon={Activity}
              color="amber"
              title="Spirometry Analysis"
              description="Upload JSON or CSV spirometry data for pattern classification, GOLD staging, and trajectory prediction"
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

function DashboardView({
  modalityResults,
  fusionResult,
  patientContext,
  modelsLoaded,
  gpuInfo,
  onQuickUpload,
}: {
  modalityResults: Record<string, ModalityResult>
  fusionResult: FusionResult | null
  patientContext: PatientContext
  modelsLoaded: Record<string, boolean>
  gpuInfo: { cuda_available?: boolean } | null
  onQuickUpload: (mod: string) => void
}) {
  const completedCount = Object.values(modalityResults).filter(r => r.success).length
  const hasFusion = fusionResult?.success

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Patient Header */}
      <div className="card-elevated">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <h3 className="text-lg font-semibold text-white">Patient Context</h3>
            <p className="text-sm text-gray-400 mt-1">
              {patientContext.patient_id ? `ID: ${patientContext.patient_id}` : 'No patient ID set'}
              {patientContext.age && ` · Age: ${patientContext.age}`}
              {patientContext.sex && ` · Sex: ${patientContext.sex}`}
            </p>
          </div>
          <div className="flex items-center gap-3">
            <PatientContextPanel context={patientContext} onChange={() => {}} compact={false} />
          </div>
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Modalities Complete"
          value={`${completedCount}/3`}
          trend={completedCount > 0 ? 'up' : 'neutral'}
          icon={<CheckCircle className="w-6 h-6 text-emerald-400" />}
        />
        <MetricCard
          title="Fusion Status"
          value={hasFusion ? 'Ready' : 'Pending'}
          trend={hasFusion ? 'up' : 'neutral'}
          icon={<BrainCircuit className="w-6 h-6 text-purple-400" />}
        />
        <MetricCard
          title="Models Loaded"
          value={`${Object.values(modelsLoaded).filter(Boolean).length}/${Object.keys(modelsLoaded).length}`}
          trend="neutral"
          icon={<Brain className="w-6 h-6 text-blue-400" />}
        />
        <MetricCard
          title="GPU"
          value={gpuInfo?.cuda_available ? 'Active' : 'CPU Only'}
          trend={gpuInfo?.cuda_available ? 'up' : 'neutral'}
          icon={<Zap className="w-6 h-6 text-amber-400" />}
        />
      </div>

      {/* Modality Quick Actions */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {MODALITIES.map(mod => (
          <QuickActionCard
            key={mod.id}
            modality={mod}
            result={modalityResults[mod.id]}
            onClick={() => onQuickUpload(mod.id)}
          />
        ))}
      </div>

      {/* Fusion Summary */}
      {fusionResult && (
        <FusionSummaryCard fusion={fusionResult} />
      )}

      {/* Recent Results */}
      {completedCount > 0 && (
        <RecentResultsPanel results={modalityResults} />
      )}
    </div>
  )
}

function MetricCard({ title, value, trend, icon }: { title: string; value: string; trend: 'up' | 'down' | 'neutral'; icon: React.ReactNode }) {
  return (
    <div className="metric-card">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-lg bg-white/5 flex items-center justify-center">{icon}</div>
        <div>
          <p className="metric-label">{title}</p>
          <p className="metric-value">{value}</p>
        </div>
      </div>
      <span className={`metric-trend ${trend === 'up' ? 'trend-up' : trend === 'down' ? 'trend-down' : 'trend-stable'}`}>
        {trend === 'up' ? '✓' : trend === 'down' ? '✗' : '—'}
      </span>
    </div>
  )
}

function QuickActionCard({
  modality,
  result,
  onClick,
}: {
  modality: (typeof MODALITIES)[number]
  result?: ModalityResult
  onClick: () => void
}) {
  const hasResult = result?.success
  return (
    <div className="panel group">
      <div className="panel-header">
        <div className="flex items-center gap-3">
          <div className={`w-10 h-10 rounded-lg ${modality.bg} flex items-center justify-center`}>
            <modality.icon className={`w-5 h-5 ${modality.color}`} />
          </div>
          <div>
            <p className="panel-title">{modality.label}</p>
            <p className="text-xs text-gray-500">Click to upload</p>
          </div>
        </div>
        {hasResult && (
          <span className="badge badge-success">Done</span>
        )}
      </div>
      <button
        onClick={onClick}
        className="w-full p-5 hover:bg-gray-900 transition-colors flex flex-col items-center justify-center gap-3"
      >
        <div className={`w-16 h-16 rounded-xl flex items-center justify-center ${modality.bg}`}>
          <modality.icon className={`w-8 h-8 ${modality.color}`} />
        </div>
        {hasResult ? (
          <>
            <p className="text-sm font-medium text-white">Result ready</p>
            <p className="text-xs text-gray-500">Click to re-run</p>
          </>
        ) : (
          <>
            <p className="text-sm font-medium text-gray-300">Upload file to analyze</p>
            <p className="text-xs text-gray-500">Drag & drop or click</p>
          </>
        )}
      </button>
      {hasResult && result?.results && (
        <div className="p-4 border-t border-gray-800 bg-gray-950/50">
          <p className="text-xs text-gray-400 mb-2">Preview:</p>
          <ModalityPreview modality={modality.id} result={result} />
        </div>
      )}
    </div>
  )
}

function ModalityPreview({ modality, result }: { modality: string; result: ModalityResult }) {
  const r = result.results
  if (!r) return null

  switch (modality) {
    case 'breath':
      return (
        <div className="flex items-center gap-3 text-sm">
          <span className="badge badge-success">{r.prediction?.toUpperCase()}</span>
          <span className="text-gray-400">{r.interpretation?.slice(0, 80)}...</span>
        </div>
      )
    case 'cxr':
      return (
        <div className="flex items-center gap-3 text-sm">
          {r.positive_findings?.slice(0, 3).map((f: string, i: number) => (
            <span key={i} className="badge badge-warning">{f}</span>
          ))}
          {r.measurements?.cardiothoracic_ratio && (
            <span className="text-gray-400">CTR: {r.measurements.cardiothoracic_ratio.toFixed(2)}</span>
          )}
        </div>
      )
    case 'spirometry':
      return (
        <div className="flex items-center gap-3 text-sm">
          <span className={`badge ${r.pattern === 'Normal' ? 'badge-success' : 'badge-warning'}`}>{r.pattern}</span>
          {r.gold_stage && <span className="badge badge-info">{r.gold_stage}</span>}
          <span className="text-gray-400">FEV1/FVC: {r.fev1_fvc_ratio?.toFixed(2)}</span>
        </div>
      )
    default:
      return null
  }
}

function FusionSummaryCard({ fusion }: { fusion: FusionResult }) {
  const risk = fusion.fused_results?.risk_level || 'Unknown'
  const riskColors: Record<string, string> = {
    'Very High': 'text-red-400 bg-red-500/10 border-red-500/30',
    'High': 'text-red-400 bg-red-500/10 border-red-500/30',
    'Moderate': 'text-amber-400 bg-amber-500/10 border-amber-500/30',
    'Low': 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30',
    'Very Low': 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30',
  }

  return (
    <div className={`panel border-${riskColors[risk]?.split(' ')[2] || 'gray-700'}`}>
      <div className="panel-header">
        <div className="flex items-center gap-3">
          <BrainCircuit className="w-5 h-5 text-purple-400" />
          <h3 className="panel-title">Multimodal Fusion Assessment</h3>
        </div>
        <span className={`badge ${riskColors[risk] ? 'bg-emerald-500/20 text-emerald-400' : 'badge-neutral'}`}>
          {risk} Risk
        </span>
      </div>
      <div className="p-5">
        <p className="text-gray-300 mb-4">{fusion.explanation}</p>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {Object.entries(fusion.modality_contributions || {}).map(([mod, val]) => (
            <div key={mod} className="text-center">
              <p className="text-2xl font-bold text-white">{Math.round(val * 100)}%</p>
              <p className="text-xs text-gray-500 capitalize">{mod.replace('_', ' ')}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function RecentResultsPanel({ results }: { results: Record<string, ModalityResult> }) {
  const recent = Object.entries(results)
    .filter(([_, r]) => r.success)
    .slice(-3)

  return (
    <div className="panel">
      <div className="panel-header">
        <h3 className="panel-title">Recent Results</h3>
      </div>
      <div className="divide-y divide-gray-800">
        {recent.map(([mod, r]) => (
          <div key={mod} className="p-4 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-white/5 flex items-center justify-center">
                {mod === 'breath' && <Mic2 className="w-4 h-4 text-emerald-400" />}
                {mod === 'cxr' && <Image className="w-4 h-4 text-blue-400" />}
                {mod === 'spirometry' && <Activity className="w-4 h-4 text-amber-400" />}
              </div>
              <div>
                <p className="font-medium text-white capitalize">{mod}</p>
                <p className="text-xs text-gray-500">{r.inference_time_ms?.toFixed(0)}ms</p>
              </div>
            </div>
            <ModalityPreview modality={mod} result={r} />
          </div>
        ))}
      </div>
    </div>
  )
}