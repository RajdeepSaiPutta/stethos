import React, { useState, useRef } from 'react'
import { Upload, Loader2, Download } from 'lucide-react'

interface ModalityPanelProps {
  modality: string
  icon: React.ComponentType<{ className?: string; strokeWidth?: number | string }>
  color?: string
  title: string
  description: string
  result?: any
  onRun: (modality: string) => void
  isProcessing: boolean
}

export function ModalityPanel({ modality, icon: Icon, title, description, result, onRun, isProcessing }: ModalityPanelProps) {
  const [dragActive, setDragActive] = useState(false)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') setDragActive(true)
    else if (e.type === 'dragleave') setDragActive(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0])
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) handleFile(e.target.files[0])
  }

  const handleFile = (file: File) => {
    const validTypes = {
      breath: ['audio/wav', 'audio/mp3', 'audio/mpeg', 'audio/flac', 'audio/x-wav'],
      cxr: ['image/dicom', 'application/dicom', 'image/png', 'image/jpeg'],
      spirometry: ['application/json', 'text/csv'],
    }
    if (!validTypes[modality as keyof typeof validTypes]?.some(t => file.type === t || file.name.endsWith(t.replace('audio/', '').replace('image/', '').replace('application/', '').replace('text/', '.')))) {
      alert(`Invalid file type for ${title}`)
      return
    }
    setPreviewUrl(URL.createObjectURL(file))
    onRun(modality)
  }

  const formatTime = (ms: number) => `${ms.toFixed(0)} ms`

  const renderResult = () => {
    if (!result?.success) return null
    const r = result.results
    if (!r) return null

    switch (modality) {
      case 'breath':
        return (
          <div className="space-y-4 animate-fade-in">
            <div className="flex items-baseline gap-3">
              <span className="mono text-sm text-white">{r.prediction}</span>
              <span className="badge">{formatTime(result.inference_time_ms || 0)}</span>
            </div>
            <div className="space-y-2.5">
              {Object.entries(r.probabilities || {}).map(([cls, prob]) => (
                <div key={cls} className="flex items-center gap-3">
                  <span className="w-24 text-[11px] text-[var(--text-muted)] capitalize">{cls}</span>
                  <div className="flex-1 h-1 bg-[var(--inset)] overflow-hidden" style={{ borderRadius: 1 }}>
                    <div className="h-full" style={{ width: `${(prob as number) * 100}%`, background: cls === r.prediction ? 'var(--accent)' : 'var(--border-bright)' }} />
                  </div>
                  <span className="mono text-right text-[11px] text-[var(--text)]">{((prob as number) * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
            <div className="p-3 inset-well">
              <p className="text-[12px] text-[var(--text-muted)]">{r.interpretation}</p>
            </div>
          </div>
        )

      case 'cxr':
        return (
          <div className="space-y-4 animate-fade-in">
            <div className="flex items-baseline gap-3">
              <span className="mono text-sm text-white">{r.positive_findings?.length ? `${r.positive_findings.length} findings` : 'no positive findings'}</span>
              <span className="badge">{formatTime(result.inference_time_ms || 0)}</span>
            </div>
            {r.positive_findings?.length && (
              <div className="flex flex-wrap gap-1.5">
                {r.positive_findings.map((f: string, i: number) => (
                  <span key={i} className="badge">{f}</span>
                ))}
              </div>
            )}
            {r.measurements && (
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-px" style={{ background: 'var(--border-soft)', border: '1px solid var(--border-soft)' }}>
                {Object.entries(r.measurements).map(([k, v]) => (
                  <div key={k} className="bg-[var(--panel)] p-2.5">
                    <p className="text-[9.5px] uppercase tracking-wider text-[var(--text-faint)]">{k.replace(/_/g, ' ')}</p>
                    <p className="mono text-[13px] text-white mt-0.5">{typeof v === 'number' ? v.toFixed(3) : String(v)}</p>
                  </div>
                ))}
              </div>
            )}
            {r.segmentation_overlay && (
              <div className="relative aspect-square overflow-hidden" style={{ border: '1px solid var(--border)', borderRadius: 4, maxWidth: 420 }}>
                <img src={r.segmentation_overlay} alt="Segmentation" className="w-full h-full object-cover" />
                <div className="absolute bottom-2 left-2 right-2 bg-black/75 px-2.5 py-1 text-[10px] font-mono text-gray-300">
                  green lungs · red heart · yellow trachea
                </div>
              </div>
            )}
          </div>
        )

      case 'spirometry':
        return (
          <div className="space-y-4 animate-fade-in">
            <div className="flex items-baseline gap-3">
              <span className="mono text-sm text-white">{r.pattern}</span>
              {r.gold_stage && <span className="badge">{r.gold_stage}</span>}
              <span className="badge">{formatTime(result.inference_time_ms || 0)}</span>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-px" style={{ background: 'var(--border-soft)', border: '1px solid var(--border-soft)' }}>
              {Object.entries(r.predicted_values || {}).map(([k, v]) => (
                <div key={k} className="bg-[var(--panel)] p-2.5">
                  <p className="text-[9.5px] uppercase tracking-wider text-[var(--text-faint)]">{k}</p>
                  <p className="mono text-[13px] text-white mt-0.5">{typeof v === 'number' ? v.toFixed(2) : String(v)}</p>
                </div>
              ))}
            </div>
            <div className="p-3 inset-well">
              <p className="text-[12px] text-[var(--text-muted)]">
                <span className="text-[var(--text-faint)]">trend</span> {r.trend}
                <span className="text-[var(--text-faint)]"> · fev1/fvc</span> <span className="mono">{r.fev1_fvc_ratio?.toFixed(2)}</span>
              </p>
            </div>
          </div>
        )

      default:
        return null
    }
  }

  return (
    <div className="flex-1 flex flex-col max-w-3xl">
      <div className="mb-4">
        <h2 className="text-[13px] font-medium text-white">{title}</h2>
        <p className="text-[11.5px] text-[var(--text-faint)] mt-0.5">{description}</p>
      </div>

      {/* Drop zone / result */}
      <div className={`flex-1 ${result?.success ? '' : 'flex flex-col items-center justify-center'}`}>
        {result?.success ? (
          <div className="w-full">
            <div className="panel p-4">{renderResult()}</div>
            <div className="flex gap-2 mt-4">
              <button onClick={() => fileInputRef.current?.click()} className="btn-secondary">
                <Upload className="w-3.5 h-3.5" /> New file
              </button>
              <button className="btn-ghost">
                <Download className="w-3.5 h-3.5" /> Export
              </button>
            </div>
          </div>
        ) : (
          <div
            className={`w-full border border-dashed p-10 transition-colors duration-150 ${
              dragActive ? '' : ''
            }`}
            style={{
              borderRadius: 4,
              borderColor: dragActive ? 'var(--accent)' : 'var(--border-bright)',
              background: dragActive ? 'var(--accent-dim)' : 'transparent',
            }}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            <input
              ref={fileInputRef}
              type="file"
              className="hidden"
              onChange={handleFileSelect}
              accept={
                modality === 'breath' ? '.wav,.mp3,.flac' :
                modality === 'cxr' ? '.dcm,.dicom,.png,.jpg,.jpeg' :
                '.json,.csv'
              }
            />
            <div className="flex flex-col items-center gap-3 text-center">
              <Icon className="w-6 h-6 text-[var(--text-faint)]" strokeWidth={1.6} />
              <div>
                <p className="text-[12.5px] text-[var(--text-muted)]">Drop a file here, or</p>
                <button onClick={() => fileInputRef.current?.click()} className="text-[12.5px] text-[var(--accent)] hover:underline mt-0.5">
                  browse
                </button>
                <p className="mono text-[10px] text-[var(--text-faint)] mt-2">
                  {modality === 'breath' ? 'wav · mp3 · flac' :
                   modality === 'cxr' ? 'dcm · png · jpg' :
                   'json · csv'}
                </p>
              </div>
              {previewUrl && (
                <div className="inset-well px-3 py-2 text-left max-w-sm w-full" style={{ borderRadius: 3 }}>
                  <p className="mono text-[10px] text-[var(--text-faint)]">selected</p>
                  <p className="mono text-[11px] text-[var(--text)] truncate">{previewUrl.split('/').pop()}</p>
                </div>
              )}
            </div>
          </div>
        )}

        {result?.error && (
          <div className="w-full mt-4 p-3" style={{ border: '1px solid rgba(209, 85, 85, 0.4)', borderRadius: 4, background: 'rgba(209, 85, 85, 0.08)' }}>
            <p className="text-[12px]" style={{ color: 'var(--bad)' }}>{result.error}</p>
          </div>
        )}

        {isProcessing && !result && (
          <div className="w-full text-center py-10">
            <Loader2 className="w-5 h-5 mx-auto animate-spin text-[var(--text-faint)] mb-3" />
            <p className="mono text-[11px] text-[var(--text-faint)]">processing…</p>
          </div>
        )}
      </div>
    </div>
  )
}
