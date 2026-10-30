import { Layers, Loader2, AlertCircle } from 'lucide-react'

interface FusionPanelProps {
  results: Record<string, any>
  fusionResult: any
  patientContext: unknown
  onRunFusion: () => void
  isProcessing: boolean
}

const MOD_LABELS: Record<string, string> = {
  breath: 'Breath Sounds',
  cxr: 'Chest X-Ray',
  spirometry: 'Spirometry',
}

export function FusionPanel({ results, fusionResult, patientContext: _patientContext, onRunFusion, isProcessing }: FusionPanelProps) {

  const successfulModalities = Object.entries(results).filter(([_, r]: any) => r?.success)
  const canRunFusion = successfulModalities.length >= 2

  const riskTone = (level?: string) =>
    ['Very High', 'High'].includes(level ?? '') ? 'var(--bad)'
    : level === 'Moderate' ? 'var(--warn)'
    : 'var(--good)'

  if (!fusionResult && !canRunFusion) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center text-center max-w-xl mx-auto">
        <Layers className="w-5 h-5 text-[var(--text-faint)] mb-4" strokeWidth={1.6} />
        <h2 className="text-[13px] font-medium text-white mb-1.5">Multimodal Fusion</h2>
        <p className="text-[12px] text-[var(--text-faint)] mb-8">
          Fusion combines breath sounds, chest X-ray and spirometry into a single risk
          assessment with per-modality contributions. Complete at least two
          modalities to enable it.
        </p>
        <div className="w-full panel">
          {Object.entries(MOD_LABELS).map(([id, label], i) => {
            const done = !!(results as any)[id]?.success
            return (
              <div key={id} className="table-row-hover flex items-center justify-between px-3.5 py-2.5" style={i > 0 ? { borderTop: '1px solid var(--border-soft)' } : undefined}>
                <span className="text-[12px] text-[var(--text-muted)]">{label}</span>
                {done
                  ? <span className="badge badge-success">complete</span>
                  : <span className="badge badge-neutral">pending</span>}
              </div>
            )
          })}
        </div>
      </div>
    )
  }

  if (!fusionResult && canRunFusion) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center text-center max-w-lg mx-auto">
        <Layers className="w-5 h-5 text-[var(--text-faint)] mb-4" strokeWidth={1.6} />
        <h2 className="text-[13px] font-medium text-white mb-1.5">Ready for Fusion</h2>
        <p className="text-[12px] text-[var(--text-faint)] mb-6">
          {successfulModalities.length} modalities complete.
        </p>
        <button
          onClick={onRunFusion}
          disabled={isProcessing}
          className="btn-primary px-5"
        >
          <Layers className="w-3.5 h-3.5" /> Run Fusion
        </button>
        <div className="mt-8 w-full panel">
          {successfulModalities.map(([mod, r]: any, i) => (
            <div key={mod} className="flex items-center justify-between px-3.5 py-2.5" style={i > 0 ? { borderTop: '1px solid var(--border-soft)' } : undefined}>
              <span className="text-[12px] text-[var(--text-muted)]">{MOD_LABELS[mod] ?? mod}</span>
              <span className="mono text-[11px] text-[var(--text)]">
                {mod === 'breath' ? r.results?.prediction : mod === 'cxr' ? `${r.results?.positive_findings?.length || 0} findings` : r.results?.pattern}
              </span>
            </div>
          ))}
        </div>
      </div>
    )
  }

  if (isProcessing && !fusionResult) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center text-center">
        <Loader2 className="w-5 h-5 animate-spin text-[var(--text-faint)] mb-4" />
        <h2 className="text-[13px] font-medium text-white mb-1.5">Running Fusion</h2>
        <p className="mono text-[10.5px] text-[var(--text-faint)]">cross-attention in progress</p>
      </div>
    )
  }

  if (!fusionResult?.success) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center text-center max-w-md mx-auto">
        <AlertCircle className="w-5 h-5 mb-4" style={{ color: 'var(--bad)' }} />
        <h2 className="text-[13px] font-medium text-white mb-1.5">Fusion Failed</h2>
        <p className="text-[12px] text-[var(--text-faint)] mb-5">{fusionResult?.error || 'Unknown error'}</p>
        <button onClick={onRunFusion} className="btn-secondary">Retry</button>
      </div>
    )
  }

  const fusion = fusionResult
  const risk = fusion.fused_results?.risk_level || 'Unknown'
  const tone = riskTone(risk)

  return (
    <div className="flex-1 max-w-4xl space-y-3">
      {/* Risk header */}
      <div className="panel">
        <div className="panel-header">
          <div>
            <h3 className="text-[13px] font-medium text-white">Multimodal Risk Assessment</h3>
            <p className="text-[10.5px] text-[var(--text-faint)] mt-0.5">
              cross-attention fusion of {successfulModalities.length} modalities
            </p>
          </div>
          <span className="mono text-sm font-medium" style={{ color: tone }}>{risk} risk</span>
        </div>
        <div className="p-3.5">
          <p className="text-[12px] text-[var(--text-muted)] leading-relaxed">{fusion.explanation}</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        {/* Modality Contributions */}
        <div className="panel">
          <div className="panel-header">
            <h3 className="panel-title">Modality Contributions</h3>
          </div>
          <div className="p-3.5 space-y-3">
            {Object.entries(fusion.modality_contributions || {}).map(([mod, val]) => (
              <div key={mod} className="space-y-1.5">
                <div className="flex justify-between text-[11.5px]">
                  <span className="text-[var(--text-muted)]">{MOD_LABELS[mod] ?? mod.replace('_', ' ')}</span>
                  <span className="mono text-[11px]">{Math.round(Number(val) * 100)}%</span>
                </div>
                <div className="h-1 bg-[var(--inset)] overflow-hidden" style={{ borderRadius: 1 }}>
                  <div className="h-full transition-all duration-500" style={{ width: `${Number(val) * 100}%`, background: tone }} />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Risk Probabilities */}
        <div className="panel">
          <div className="panel-header">
            <h3 className="panel-title">Risk Probabilities</h3>
          </div>
          <div className="p-3.5 space-y-2.5">
            {Object.entries(fusion.fused_results?.risk_probabilities || {}).map(([level, prob]) => {
              const isCurrent = level === risk
              return (
                <div key={level} className="flex items-center gap-3">
                  <span className={`w-28 text-[11.5px] flex-shrink-0 ${isCurrent ? 'text-white' : 'text-[var(--text-muted)]'}`}>{level}</span>
                  <div className="flex-1 h-1 bg-[var(--inset)] overflow-hidden" style={{ borderRadius: 1 }}>
                    <div className="h-full transition-all duration-500" style={{ width: `${Number(prob) * 100}%`, background: isCurrent ? tone : 'var(--border-bright)' }} />
                  </div>
                  <span className={`mono text-[11px] w-14 text-right ${isCurrent ? 'text-white' : 'text-[var(--text-faint)]'}`}>{(Number(prob) * 100).toFixed(1)}%</span>
                </div>
              )
            })}
          </div>
        </div>
      </div>

      {/* Input Modality Details */}
      <div className="panel">
        <div className="panel-header">
          <h3 className="panel-title">Input Modality Details</h3>
        </div>
        <div className="p-3.5 grid grid-cols-1 md:grid-cols-3 gap-px" style={{ background: 'var(--border-soft)' }}>
          {successfulModalities.map(([mod, r]: any) => (
            <div key={mod} className="bg-[var(--panel)] p-3">
              <h4 className="text-[11px] uppercase tracking-wider text-[var(--text-faint)] mb-2">{MOD_LABELS[mod] ?? mod}</h4>
              <div className="text-[11.5px] text-[var(--text-muted)] space-y-1">
                {mod === 'breath' && r.results && (
                  <>
                    <p><span className="text-[var(--text-faint)]">prediction</span> <span className="mono text-[var(--text)]">{r.results.prediction}</span></p>
                    <p><span className="text-[var(--text-faint)]">confidence</span> <span className="mono">{Math.round((r.results.probabilities?.[r.results.prediction] || 0) * 100)}%</span></p>
                  </>
                )}
                {mod === 'cxr' && r.results && (
                  <>
                    <p><span className="text-[var(--text-faint)]">findings</span> <span>{r.results.positive_findings?.length || 0}</span></p>
                    {r.results.measurements?.cardiothoracic_ratio && (
                      <p><span className="text-[var(--text-faint)]">ctr</span> <span className="mono">{r.results.measurements.cardiothoracic_ratio.toFixed(2)}</span></p>
                    )}
                  </>
                )}
                {mod === 'spirometry' && r.results && (
                  <>
                    <p><span className="text-[var(--text-faint)]">pattern</span> <span>{r.results.pattern}</span></p>
                    {r.results.gold_stage && <p><span className="text-[var(--text-faint)]">gold</span> <span className="mono">{r.results.gold_stage}</span></p>}
                  </>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="flex gap-2 justify-end">
        <button onClick={onRunFusion} className="btn-secondary">
          <Loader2 className="w-3.5 h-3.5" /> Re-run Fusion
        </button>
      </div>
    </div>
  )
}
