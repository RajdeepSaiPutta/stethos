import React, { useState, useRef } from 'react'
import { Upload, AlertCircle, Loader2, Download } from 'lucide-react'

interface ModalityPanelProps {
  modality: string
  icon: React.ComponentType<{ className?: string }>
  color: string
  title: string
  description: string
  result?: any
  onRun: (modality: string) => void
  isProcessing: boolean
}

const COLOR_CLASSES = {
  emerald: { bg: 'bg-emerald-500/10', border: 'border-emerald-500/30', text: 'text-emerald-400', badge: 'badge-success' },
  blue: { bg: 'bg-blue-500/10', border: 'border-blue-500/30', text: 'text-blue-400', badge: 'badge-info' },
  amber: { bg: 'bg-amber-500/10', border: 'border-amber-500/30', text: 'text-amber-400', badge: 'badge-warning' },
}

export function ModalityPanel({ modality, icon: Icon, color, title, description, result, onRun, isProcessing }: ModalityPanelProps) {
  const [dragActive, setDragActive] = useState(false)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const colors = COLOR_CLASSES[color as keyof typeof COLOR_CLASSES] || COLOR_CLASSES.emerald

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
            <div className="flex items-center gap-3">
              <span className={`badge ${colors.badge} text-lg px-4 py-2`}>{r.prediction?.toUpperCase()}</span>
              <span className="text-sm text-gray-400">{formatTime(result.inference_time_ms || 0)}</span>
            </div>
            <div className="grid grid-cols-2 gap-3">
              {Object.entries(r.probabilities || {}).map(([cls, prob]) => (
                <div key={cls} className="bg-gray-900/50 rounded-lg p-3">
                  <p className="text-xs text-gray-500 capitalize">{cls}</p>
                  <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden mt-1">
                    <div className="h-full bg-emerald-500 transition-all duration-500" style={{ width: `${(prob as number) * 100}%` }} />
                  </div>
                  <p className="text-right text-sm font-mono text-emerald-400 mt-1">{(prob as number * 100).toFixed(1)}%</p>
                </div>
              ))}
            </div>
            <div className="p-4 bg-gray-900/50 rounded-lg border border-gray-800">
              <p className="text-sm text-gray-300">{r.interpretation}</p>
            </div>
          </div>
        )

      case 'cxr':
        return (
          <div className="space-y-4 animate-fade-in">
            <div className="flex items-center gap-3">
              <span className={`badge ${r.positive_findings?.length ? 'badge-warning' : 'badge-success'} text-lg px-4 py-2`}>
                {r.positive_findings?.length ? `${r.positive_findings.length} Findings` : 'Normal'}
              </span>
              <span className="text-sm text-gray-400">{formatTime(result.inference_time_ms || 0)}</span>
            </div>
            {r.positive_findings?.length && (
              <div className="flex flex-wrap gap-2">
                {r.positive_findings.map((f: string, i: number) => (
                  <span key={i} className="badge badge-warning">{f}</span>
                ))}
              </div>
            )}
            {r.measurements && (
              <div className="grid grid-cols-2 gap-3">
                {Object.entries(r.measurements).map(([k, v]) => (
                  <div key={k} className="bg-gray-900/50 rounded-lg p-3">
                    <p className="text-xs text-gray-500">{k.replace(/_/g, ' ')}</p>
                    <p className="font-mono text-lg text-white">{typeof v === 'number' ? v.toFixed(3) : String(v)}</p>
                  </div>
                ))}
              </div>
            )}
            {r.segmentation_overlay && (
              <div className="relative aspect-square rounded-lg overflow-hidden border border-gray-700">
                <img src={r.segmentation_overlay} alt="Segmentation" className="w-full h-full object-cover" />
                <div className="absolute bottom-2 left-2 right-2 bg-black/70 px-3 py-1 text-xs text-gray-300">
                  Segmentation overlay (Green=Lungs, Red=Heart, Yellow=Trachea)
                </div>
              </div>
            )}
          </div>
        )

      case 'spirometry':
        return (
          <div className="space-y-4 animate-fade-in">
            <div className="flex items-center gap-3">
              <span className={`badge ${r.pattern === 'Normal' ? 'badge-success' : 'badge-warning'} text-lg px-4 py-2`}>{r.pattern}</span>
              {r.gold_stage && <span className="badge badge-info">{r.gold_stage}</span>}
              <span className="text-sm text-gray-400">{formatTime(result.inference_time_ms || 0)}</span>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {Object.entries(r.predicted_values || {}).map(([k, v]) => (
                <div key={k} className="bg-gray-900/50 rounded-lg p-3">
                  <p className="text-xs text-gray-500">{k}</p>
                  <p className="font-mono text-lg text-white">{typeof v === 'number' ? v.toFixed(2) : String(v)}</p>
                </div>
              ))}
            </div>
            <div className="p-4 bg-gray-900/50 rounded-lg border border-gray-800">
              <p className="text-sm text-gray-300">
                <strong>Trend:</strong> {r.trend} · FEV1/FVC: {r.fev1_fvc_ratio?.toFixed(2)}
              </p>
            </div>
          </div>
        )

      default:
        return null
    }
  }

  return (
    <div className="flex-1 flex flex-col">
      <div className="mb-6">
        <div className="flex items-center gap-3 mb-2">
          <div className={`w-10 h-10 rounded-lg ${colors.bg} flex items-center justify-center`}>
            <Icon className={`w-6 h-6 ${colors.text}`} />
          </div>
          <div>
            <h2 className="text-2xl font-semibold text-white">{title}</h2>
            <p className="text-gray-400">{description}</p>
          </div>
        </div>
      </div>

      {/* Drop Zone / Result */}
      <div className={`flex-1 ${result?.success ? '' : 'flex flex-col items-center justify-center'}`}>
        {result?.success ? (
          <div className="w-full max-w-3xl mx-auto">
            {renderResult()}
            <div className="flex gap-3 mt-6 pt-4 border-t border-gray-800">
              <button onClick={() => fileInputRef.current?.click()} className="btn-secondary flex-1">
                <Upload className="w-4 h-4" /> New File
              </button>
              <button className="btn-ghost">
                <Download className="w-4 h-4" /> Export
              </button>
            </div>
          </div>
        ) : (
          <div
            className={`w-full max-w-2xl mx-auto border-2 border-dashed rounded-2xl p-12 transition-all duration-200 ${
              dragActive ? 'border-emerald-500 bg-emerald-500/5' : 'border-gray-700 hover:border-gray-600'
            }`}
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
            <div className="flex flex-col items-center gap-4 text-center">
              <div className={`w-16 h-16 rounded-xl flex items-center justify-center ${colors.bg}`}>
                <Icon className={`w-8 h-8 ${colors.text}`} />
              </div>
              <div>
                <p className="text-lg font-medium text-white">Drop file here or click to browse</p>
                <p className="text-sm text-gray-500 mt-1">
                  {modality === 'breath' ? 'WAV, MP3, FLAC' :
                   modality === 'cxr' ? 'DICOM, PNG, JPG' :
                   'JSON, CSV'}
                </p>
              </div>
              {previewUrl && (
                <div className="mt-4 p-3 bg-gray-900/50 rounded-lg border border-gray-700 text-left max-w-md">
                  <p className="text-xs text-gray-400">Selected:</p>
                  <p className="font-mono text-sm text-white truncate">{previewUrl.split('/').pop()}</p>
                </div>
              )}
            </div>
          </div>
        )}

        {result?.error && (
          <div className="w-full max-w-2xl mx-auto mt-6 p-4 bg-red-500/10 border border-red-500/30 rounded-lg">
            <div className="flex items-center gap-3 text-red-400">
              <AlertCircle className="w-5 h-5 flex-shrink-0" />
              <p className="text-sm">{result.error}</p>
            </div>
          </div>
        )}

        {isProcessing && modality === 'breath' && !result && (
          <div className="w-full max-w-2xl mx-auto text-center py-12">
            <Loader2 className="w-10 h-10 mx-auto animate-spin text-emerald-400 mb-4" />
            <p className="text-gray-400">Processing audio...</p>
          </div>
        )}
      </div>
    </div>
  )
}