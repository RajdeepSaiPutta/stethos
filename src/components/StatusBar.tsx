import { Activity, CircleDot } from 'lucide-react'
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
    <footer className="h-8 px-6 border-t border-gray-800 flex items-center justify-between text-xs text-gray-500 bg-gray-950">
      <div className="flex items-center gap-4">
        <span className="flex items-center gap-1.5">
          <CircleDot className={`w-2.5 h-2.5 ${backendUp ? 'text-green-400' : 'text-red-400'}`} />
          Sidecar {backendUp ? 'connected' : 'starting…'}
        </span>
        <span className="flex items-center gap-1.5">
          <Activity className="w-3 h-3" />
          Models {loaded}/{total}
        </span>
      </div>
      <div className="flex items-center gap-4">
        <span>{gpuInfo?.cuda_available ? `${gpuInfo.gpu_name ?? 'CUDA'} · ${Math.round(gpuInfo.gpu_memory_total_mb ?? 0)} MB` : 'CPU mode'}</span>
        <span className="text-gray-700">|</span>
        <span>research tool — not a medical device</span>
      </div>
    </footer>
  )
}
