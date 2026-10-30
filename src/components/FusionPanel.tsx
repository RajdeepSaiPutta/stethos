import { BrainCircuit, Zap, AlertCircle, Loader2, Brain } from 'lucide-react'

interface FusionPanelProps {
  results: Record<string, any>
  fusionResult: any
  patientContext: unknown
  onRunFusion: () => void
  isProcessing: boolean
}

export function FusionPanel({ results, fusionResult, patientContext: _patientContext, onRunFusion, isProcessing }: FusionPanelProps) {

  const successfulModalities = Object.entries(results).filter(([_, r]: any) => r?.success)
  const canRunFusion = successfulModalities.length >= 2

  if (!fusionResult && !canRunFusion) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center text-center">
        <div className="w-20 h-20 rounded-xl bg-purple-500/10 flex items-center justify-center mb-6">
          <BrainCircuit className="w-10 h-10 text-purple-400" />
        </div>
        <h2 className="text-2xl font-semibold text-white mb-2">Multimodal Fusion</h2>
        <p className="text-gray-400 max-w-md mb-8">
          Fusion combines breath sounds, chest X-ray, and spirometry into a unified risk assessment.
          Complete at least 2 modalities to enable fusion.
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 max-w-2xl w-full">
          {[
            { id: 'breath', label: 'Breath Sounds', icon: '🎙️', done: !!results.breath?.success },
            { id: 'cxr', label: 'Chest X-Ray', icon: '🩻', done: !!results.cxr?.success },
            { id: 'spirometry', label: 'Spirometry', icon: '📊', done: !!results.spirometry?.success },
          ].map(m => (
            <div key={m.id} className={`p-6 rounded-xl border-2 transition-all ${
              m.done ? 'border-emerald-500/30 bg-emerald-500/5' : 'border-gray-700 bg-gray-900/50'
            }`}>
              <div className="text-3xl mb-2">{m.icon}</div>
              <p className="font-medium text-white">{m.label}</p>
              <p className="text-sm text-gray-500 mt-1">
                {m.done ? '✓ Complete' : 'Pending'}
              </p>
            </div>
          ))}
        </div>
      </div>
    )
  }

  if (!fusionResult && canRunFusion) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center text-center">
        <div className="w-20 h-20 rounded-xl bg-purple-500/10 flex items-center justify-center mb-6">
          <BrainCircuit className="w-10 h-10 text-purple-400" />
        </div>
        <h2 className="text-2xl font-semibold text-white mb-2">Ready for Fusion</h2>
        <p className="text-gray-400 mb-8">
          {successfulModalities.length} modalities completed. Run fusion to get unified risk assessment.
        </p>
        <button
          onClick={onRunFusion}
          disabled={isProcessing}
          className="btn-primary px-8 py-3 text-lg gap-2"
        >
          <Brain className="w-5 h-5" /> Run Fusion Analysis
        </button>
        <div className="mt-6 grid grid-cols-3 gap-4 max-w-xl w-full text-left">
          {successfulModalities.map(([mod, r]: any) => (
            <div key={mod} className="p-4 bg-gray-900/50 rounded-lg border border-gray-700">
              <p className="font-medium text-white capitalize">{mod}</p>
              <p className="text-sm text-gray-500 mt-1">{mod === 'breath' ? r.results?.prediction : mod === 'cxr' ? `${r.results?.positive_findings?.length || 0} findings` : r.results?.pattern} pattern</p>
            </div>
          ))}
        </div>
      </div>
    )
  }

  if (isProcessing && !fusionResult) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center text-center">
        <div className="w-20 h-20 rounded-xl bg-purple-500/10 flex items-center justify-center mb-6">
          <Loader2 className="w-10 h-10 text-purple-400 animate-spin" />
        </div>
        <h2 className="text-2xl font-semibold text-white mb-2">Running Fusion...</h2>
        <p className="text-gray-400">Cross-attention fusion in progress</p>
      </div>
    )
  }

  if (!fusionResult?.success) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center text-center">
        <AlertCircle className="w-16 h-16 text-red-400 mb-4" />
        <h2 className="text-2xl font-semibold text-white mb-2">Fusion Failed</h2>
        <p className="text-gray-400 mb-4">{fusionResult?.error || 'Unknown error'}</p>
        <button onClick={onRunFusion} className="btn-primary">Retry</button>
      </div>
    )
  }

  const fusion = fusionResult
  const risk = fusion.fused_results?.risk_level || 'Unknown'
  const riskColors = {
    'Very High': { bg: 'bg-red-500/10', border: 'border-red-500/30', text: 'text-red-400', badge: 'badge-danger' },
    'High': { bg: 'bg-red-500/10', border: 'border-red-500/30', text: 'text-red-400', badge: 'badge-danger' },
    'Moderate': { bg: 'bg-amber-500/10', border: 'border-amber-500/30', text: 'text-amber-400', badge: 'badge-warning' },
    'Low': { bg: 'bg-emerald-500/10', border: 'border-emerald-500/30', text: 'text-emerald-400', badge: 'badge-success' },
    'Very Low': { bg: 'bg-emerald-500/10', border: 'border-emerald-500/30', text: 'text-emerald-400', badge: 'badge-success' },
  }
  const rc = riskColors[risk as keyof typeof riskColors] || riskColors.Moderate

  return (
    <div className="flex-1 space-y-6">
      {/* Risk Header */}
      <div className={`panel ${rc.border}`}>
        <div className="panel-header">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-purple-500/20 flex items-center justify-center">
              <BrainCircuit className="w-5 h-5 text-purple-400" />
            </div>
            <div>
              <h3 className="panel-title">Multimodal Risk Assessment</h3>
              <p className="text-sm text-gray-400">Cross-attention fusion of {successfulModalities.length} modalities</p>
            </div>
          </div>
          <span className={`badge ${rc.badge} text-lg px-4 py-2`}>
            {risk} Risk
          </span>
        </div>
        <div className="p-5">
          <p className="text-gray-300 mb-4">{fusion.explanation}</p>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {Object.entries(fusion.modality_contributions || {}).map(([mod, val]) => (
              <div key={mod} className="text-center p-4 bg-gray-900/50 rounded-lg">
                <p className="text-3xl font-bold text-white">{Math.round(Number(val) * 100)}%</p>
                <p className="text-xs text-gray-500 capitalize">{mod.replace('_', ' ')}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Detailed Breakdown */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Modality Contributions */}
        <div className="panel">
          <div className="panel-header">
            <h3 className="panel-title">Modality Contributions</h3>
          </div>
          <div className="p-5 space-y-4">
            {Object.entries(fusion.modality_contributions || {}).map(([mod, val]) => (
              <div key={mod} className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="capitalize text-gray-300">{mod.replace('_', ' ')}</span>
                  <span className="font-mono text-white">{Math.round(Number(val) * 100)}%</span>
                </div>
                <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                  <div className="h-full bg-purple-500 transition-all duration-500" style={{ width: `${Number(val) * 100}%` }} />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Confidence & Risk Probabilities */}
        <div className="panel">
          <div className="panel-header">
            <h3 className="panel-title">Risk Probabilities</h3>
          </div>
          <div className="p-5 space-y-3">
            {Object.entries(fusion.fused_results?.risk_probabilities || {}).map(([level, prob]) => {
              const isCurrent = level === risk
              return (
                <div key={level} className={`flex items-center gap-3 p-3 rounded-lg ${isCurrent ? rc.bg + ' ' + rc.border : 'bg-gray-900/50'}`}>
                  <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${isCurrent ? rc.bg : 'bg-gray-800'}`}>
                    {isCurrent ? <Zap className={`w-4 h-4 ${rc.text}`} /> : <Brain className="w-4 h-4 text-gray-500" />}
                  </div>
                  <div className="flex-1">
                    <p className="font-medium text-white">{level}</p>
                    <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden mt-1">
                      <div className="h-full transition-all duration-500" style={{ width: `${Number(prob) * 100}%` }} />
                    </div>
                  </div>
                  <span className={`font-mono text-lg ${isCurrent ? rc.text : 'text-gray-400'}`}>{(Number(prob) * 100).toFixed(1)}%</span>
                </div>
              )
            })}
          </div>
        </div>
      </div>

      {/* Individual Modality Details */}
      <div className="panel">
        <div className="panel-header">
          <h3 className="panel-title">Input Modality Details</h3>
        </div>
        <div className="p-5">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {successfulModalities.map(([mod, r]: any) => (
              <div key={mod} className="bg-gray-900/50 rounded-lg p-4 border border-gray-700">
                <div className="flex items-center gap-2 mb-3">
                  <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${mod === 'breath' ? 'bg-emerald-500/10' : mod === 'cxr' ? 'bg-blue-500/10' : 'bg-amber-500/10'}`}>
                    {mod === 'breath' && <span className="text-emerald-400">🎙️</span>}
                    {mod === 'cxr' && <span className="text-blue-400">🩻</span>}
                    {mod === 'spirometry' && <span className="text-amber-400">📊</span>}
                  </div>
                  <h4 className="font-medium text-white capitalize">{mod}</h4>
                </div>
                <div className="text-sm text-gray-400 space-y-1">
                  {mod === 'breath' && r.results && (
                    <>
                      <p>Prediction: <span className="text-white font-mono">{r.results.prediction}</span></p>
                      <p>Confidence: <span className="font-mono">{(r.results.probabilities?.[r.results.prediction] || 0) * 100}%</span></p>
                    </>
                  )}
                  {mod === 'cxr' && r.results && (
                    <>
                      <p>Findings: <span className="text-white">{r.results.positive_findings?.length || 0}</span></p>
                      {r.results.measurements?.cardiothoracic_ratio && (
                        <p>CTR: <span className="font-mono">{r.results.measurements.cardiothoracic_ratio.toFixed(2)}</span></p>
                      )}
                    </>
                  )}
                  {mod === 'spirometry' && r.results && (
                    <>
                      <p>Pattern: <span className="text-white">{r.results.pattern}</span></p>
                      {r.results.gold_stage && <p>GOLD: <span className="text-white">{r.results.gold_stage}</span></p>}
                    </>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Actions */}
      <div className="flex gap-3 justify-end">
        <button onClick={onRunFusion} className="btn-secondary">
          <Loader2 className="w-4 h-4" /> Re-run Fusion
        </button>
      </div>
    </div>
  )
}