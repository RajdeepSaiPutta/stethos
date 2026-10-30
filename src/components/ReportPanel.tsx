import { useState } from 'react'
import { FileText, Download, Info, AlertCircle, CheckCircle, ClipboardList } from 'lucide-react'
import type { ModalityResult, FusionResult, PatientContext } from '../lib/types'

type ExportFormat = 'pdf' | 'fhir' | 'dicom-sr' | 'json'

interface ReportPanelProps {
  patientContext: PatientContext
  modalityResults: Record<string, ModalityResult>
  fusionResult: FusionResult | null
  onExport: (format: ExportFormat) => Promise<void>
}

const FORMAT_LABELS: Record<ExportFormat, string> = {
  pdf: 'Clinical PDF',
  fhir: 'FHIR Bundle',
  'dicom-sr': 'DICOM SR',
  json: 'Raw JSON',
}

export function ReportPanel({ patientContext, modalityResults, fusionResult, onExport }: ReportPanelProps) {
  const completed = Object.entries(modalityResults).filter(([, r]) => r.success)
  const failed = Object.entries(modalityResults).filter(([, r]) => !r.success)
  const [exporting, setExportingName] = useState<ExportFormat | null>(null)

  const handleExport = async (format: ExportFormat) => {
    setExportingName(format)
    try {
      await onExport(format)
    } finally {
      setExportingName(null)
    }
  }

  const breath = modalityResults.breath?.results
  const cxr = modalityResults.cxr?.results
  const spiro = modalityResults.spirometry?.results
  const fusion = fusionResult?.fused_results

  return (
    <div className="space-y-6 animate-fade-in max-w-4xl">
      <div className="panel">
        <div className="panel-header">
          <div className="flex items-center gap-3">
            <FileText className="w-5 h-5 text-blue-400" />
            <h3 className="panel-title">Report Summary</h3>
          </div>
          <span className="badge badge-neutral">
            {completed.length}/3 modalities complete
          </span>
        </div>

        <div className="p-5 space-y-5">
          {/* patient block */}
          <div className="bg-gray-900/50 rounded-lg p-4">
            <p className="text-xs uppercase tracking-wide text-gray-500 mb-2">Patient</p>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
              <div><span className="text-gray-500">ID:</span> <span className="text-white">{patientContext.patient_id || '—'}</span></div>
              <div><span className="text-gray-500">Age:</span> <span className="text-white">{patientContext.age ?? '—'}</span></div>
              <div><span className="text-gray-500">Sex:</span> <span className="text-white">{patientContext.sex || '—'}</span></div>
              <div><span className="text-gray-500">BMI:</span> <span className="text-white">{patientContext.bmi ?? '—'}</span></div>
            </div>
            {patientContext.medications && patientContext.medications.length > 0 && (
              <div className="mt-3 flex flex-wrap gap-2">
                {patientContext.medications.map((m) => (
                  <span key={m} className="badge badge-info">{m}</span>
                ))}
              </div>
            )}
          </div>

          {/* findings per modality */}
          <div className="space-y-3">
            {completed.map(([mod, r]) => (
              <div key={mod} className="flex items-start gap-3 bg-gray-900/50 rounded-lg p-4">
                <CheckCircle className="w-4 h-4 text-emerald-400 mt-0.5 flex-shrink-0" />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-white capitalize">{mod.replace('_', ' ')}</p>
                  <p className="text-xs text-gray-400 mt-1 break-words">
                    {mod === 'breath' && `Prediction: ${breath?.prediction ?? '—'}${typeof breath?.confidence === 'number' ? ` · ${Math.round(breath.confidence)}% confidence` : ''}`}
                    {mod === 'cxr' && `Findings: ${cxr?.positive_findings?.length ? cxr.positive_findings.join(', ') : 'none significant'}`}
                    {mod === 'spirometry' && `Pattern: ${spiro?.pattern ?? '—'}${spiro?.gold_stage ? ` · ${spiro.gold_stage}` : ''}`}
                    {r.inference_time_ms ? ` · ${r.inference_time_ms.toFixed(0)} ms` : ''}
                  </p>
                </div>
              </div>
            ))}

            {fusionResult?.success && fusion && (
              <div className="flex items-start gap-3 bg-purple-500/5 border border-purple-500/20 rounded-lg p-4">
                <CheckCircle className="w-4 h-4 text-purple-400 mt-0.5 flex-shrink-0" />
                <div>
                  <p className="text-sm font-medium text-white">Multimodal fusion</p>
                  <p className="text-xs text-gray-400 mt-1">{fusionResult.explanation || `Risk level: ${fusion.risk_level ?? '—'}`}</p>
                </div>
              </div>
            )}

            {failed.map(([mod, r]) => (
              <div key={mod} className="flex items-start gap-3 bg-red-500/5 border border-red-500/20 rounded-lg p-4">
                <AlertCircle className="w-4 h-4 text-red-400 mt-0.5 flex-shrink-0" />
                <div>
                  <p className="text-sm font-medium text-white capitalize">{mod.replace('_', ' ')} failed</p>
                  <p className="text-xs text-gray-500 mt-1">{r.error || 'Unknown error'}</p>
                </div>
              </div>
            ))}

            {completed.length === 0 && failed.length === 0 && (
              <p className="text-sm text-gray-500">No results yet — run a modality analysis first. The report fills itself as results arrive.</p>
            )}
          </div>

          {/* disclaimer */}
          <div className="flex items-start gap-2 text-xs text-gray-500">
            <Info className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
            <p>research tool, not a medical device — confirm findings with a clinician.</p>
          </div>
        </div>

        <div className="px-5 py-4 border-t border-gray-800 flex flex-wrap items-center gap-3 bg-gray-950">
          <ClipboardList className="w-4 h-4 text-gray-500" />
          <span className="text-xs text-gray-500 mr-2">Export as:</span>
          {(Object.keys(FORMAT_LABELS) as ExportFormat[]).map((fmt) => (
            <button
              key={fmt}
              onClick={() => handleExport(fmt)}
              disabled={exporting !== null}
              className="btn-secondary text-xs px-3 py-1.5 flex items-center gap-1.5 disabled:opacity-50"
            >
              <Download className="w-3.5 h-3.5" />
              {exporting === fmt ? 'Saving…' : FORMAT_LABELS[fmt]}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
