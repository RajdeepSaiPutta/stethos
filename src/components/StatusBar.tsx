import type { HealthInfo } from '../lib/types'

interface StatusBarProps {
  gpuInfo: HealthInfo | null
  modelsLoaded: Record<string, boolean>
}

export function StatusBar({ gpuInfo, modelsLoaded }: StatusBarProps) {
  const total = Object.keys(modelsLoaded).length
  const loaded = Object.values(modelsLoaded).filter(Boolean).length
  const backendUp = gpuInfo !== null

  return (
    <footer className="h-6 px-3 flex items-center justify-between text-[10px] font-mono gap-4" style={{ borderTop: '1px solid var(--border)', background: 'var(--chrome)', color: 'var(--text-faint)' }}>
      <div className="flex items-center gap-3">
        <span className="flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 rounded-full" style={{ background: backendUp ? 'var(--good)' : 'var(--warn)' }} />
          sidecar {backendUp ? 'connected' : 'starting…'}
        </span>
        <span>models {loaded}/{total}</span>
      </div>
      <div className="flex items-center gap-3 min-w-0">
        <span className="truncate">{gpuInfo?.cuda_available ? `${gpuInfo.gpu_name ?? 'GPU'} · ${Math.round(gpuInfo.gpu_memory_total_mb ?? 0)} MB` : 'cpu'}</span>
        <span className="hidden sm:inline">research use only</span>
      </div>
    </footer>
  )
}
