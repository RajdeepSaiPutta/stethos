import { useState } from 'react'
import { Download, Info, AlertCircle, CheckCircle } from 'lucide-react'
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
          <h3 className="panel-title">Report Summary</h3>
          <span className="badge">
            {completed.length}/3 modalities complete
          </span>
        </div>

        <div className="p-3.5 space-y-4">
          {/* patient block */}
          <div className="inset-well p-3">
            <p className="text-[9.5px] uppercase tracking-widest text-[var(--text-faint)] mb-2">Patient</p>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-[12px]">
              <div><span className="text-[var(--text-faint)]">id</span> <span className="mono text-[var(--text)]">{patientContext.patient_id || '—'}</span></div>
              <div><span className="text-[var(--text-faint)]">age</span> <span className="mono text-[var(--text)]">{patientContext.age ?? '—'}</span></div>
              <div><span className="text-[var(--text-faint)]">sex</span> <span className="mono text-[var(--text)]">{patientContext.sex || '—'}</span></div>
              <div><span className="text-[var(--text-faint)]">bmi</span> <span className="mono text-[var(--text)]">{patientContext.bmi ?? '—'}</span></div>
            </div>
            {patientContext.medications && patientContext.medications.length > 0 && (
              <div className="mt-2.5 flex flex-wrap gap-1.5">
                {patientContext.medications.map((m) => (
                  <span key={m} className="badge">{m}</span>
                ))}
              </div>
            )}
          </div>

          {/* findings per modality */}
          <div className="space-y-2">
            {completed.map(([mod, r]) => (
              <div key={mod} className="flex items-start gap-2.5 inset-well p-3">
                <CheckCircle className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" style={{ color: 'var(--good)' }} />
                <div className="flex-1 min-w-0">
                  <p className="text-[12px] text-[var(--text)] capitalize">{mod.replace('_', ' ')}</p>
                  <p className="text-[11px] text-[var(--text-muted)] mt-0.5 break-words">
                    {mod === 'breath' && `Prediction: ${breath?.prediction ?? '—'}${typeof breath?.confidence === 'number' ? ` · ${Math.round(breath.confidence)}% confidence` : ''}`}
                    {mod === 'cxr' && `Findings: ${cxr?.positive_findings?.length ? cxr.positive_findings.join(', ') : 'none significant'}`}
                    {mod === 'spirometry' && `Pattern: ${spiro?.pattern ?? '—'}${spiro?.gold_stage ? ` · ${spiro.gold_stage}` : ''}`}
                    {r.inference_time_ms ? ` · ${r.inference_time_ms.toFixed(0)} ms` : ''}
                  </p>
                </div>
              </div>
            ))}

            {fusionResult?.success && fusion && (
              <div className="flex items-start gap-2.5 inset-well p-3">
                <CheckCircle className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" style={{ color: 'var(--accent)' }} />
                <div>
                  <p className="text-[12px] text-[var(--text)]">Multimodal fusion</p>
                  <p className="text-[11px] text-[var(--text-muted)] mt-0.5">{fusionResult.explanation || `Risk level: ${fusion.risk_level ?? '—'}`}</p>
                </div>
              </div>
            )}

            {failed.map(([mod, r]) => (
              <div key={mod} className="flex items-start gap-2.5 p-3" style={{ border: '1px solid rgba(209,85,85,0.3)', borderRadius: 3, background: 'rgba(209,85,85,0.06)' }}>
                <AlertCircle className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" style={{ color: 'var(--bad)' }} />
                <div>
                  <p className="text-[12px] text-[var(--text)] capitalize">{mod.replace('_', ' ')} failed</p>
                  <p className="text-[11px] text-[var(--text-faint)] mt-0.5">{r.error || 'Unknown error'}</p>
                </div>
              </div>
            ))}

            {completed.length === 0 && failed.length === 0 && (
              <p className="text-[12px] text-[var(--text-faint)]">No results yet — run a modality analysis first. The report fills itself as results arrive.</p>
            )}
          </div>

          {/* disclaimer */}
          <div className="flex items-start gap-2 text-[10.5px] text-[var(--text-faint)]">
            <Info className="w-3 h-3 flex-shrink-0 mt-0.5" />
            <p>research tool, not a medical device — confirm findings with a clinician.</p>
          </div>
        </div>

        <div className="px-3.5 py-3 flex flex-wrap items-center gap-2" style={{ borderTop: '1px solid var(--border-soft)', background: 'var(--chrome)' }}>
          <span className="text-[10.5px] text-[var(--text-faint)] mr-1 font-mono">export</span>
          {(Object.keys(FORMAT_LABELS) as ExportFormat[]).map((fmt) => (
            <button
              key={fmt}
              onClick={() => handleExport(fmt)}
              disabled={exporting !== null}
              className="btn-secondary px-2.5 disabled:opacity-40"
            >
              <Download className="w-3 h-3" />
              {exporting === fmt ? 'Saving…' : FORMAT_LABELS[fmt]}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
