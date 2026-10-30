import { useEffect, useState, useCallback } from 'react'
import { invoke } from '@tauri-apps/api/core'
import {
  Brain, RefreshCw, CheckCircle, XCircle, Package, HardDrive, Cpu,
} from 'lucide-react'
import type { PluginInfo, HealthInfo } from '../lib/types'

export function ModelManager() {
  const [health, setHealth] = useState<HealthInfo | null>(null)
  const [refreshing, setRefreshing] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    setRefreshing(true)
    setError(null)
    try {
      const [h, m] = await Promise.all([
        invoke<HealthInfo>('python_health_check'),
        invoke<{ plugins: PluginInfo[] }>('list_models'),
      ])
      setHealth({ ...h, plugins: m.plugins ?? [] })
    } catch (e) {
      setError(typeof e === 'string' ? e : JSON.stringify(e))
    } finally {
      setRefreshing(false)
    }
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh])

  const plugins: PluginInfo[] = health?.plugins ?? []
  const loadedCount = plugins.filter((p) => p.loaded).length

  return (
    <div className="space-y-6 animate-fade-in max-w-4xl">
      <div className="panel">
        <div className="panel-header">
          <div className="flex items-center gap-3">
            <Brain className="w-5 h-5 text-gray-300" />
            <h3 className="panel-title">Installed Models</h3>
          </div>
          <div className="flex items-center gap-3">
            <span className="badge badge-neutral">{loadedCount}/{plugins.length} loaded</span>
            <button onClick={refresh} disabled={refreshing} className="btn-ghost p-2" aria-label="Refresh model status">
              <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
            </button>
          </div>
        </div>

        <div className="divide-y divide-gray-800">
          {error && (
            <div className="px-5 py-3 flex items-center gap-2 bg-red-500/5 text-red-400 text-sm">
              <XCircle className="w-4 h-4" />
              Health check failed: {error}
            </div>
          )}

          {plugins.map((p) => (
            <div key={p.name} className="px-5 py-4 flex items-center justify-between hover:bg-gray-900/40 transition-colors">
              <div className="flex items-center gap-3">
                <div className="w-9 h-9 rounded-lg bg-white/5 flex items-center justify-center">
                  <Package className="w-4 h-4 text-gray-400" />
                </div>
                <div>
                  <p className="text-sm font-medium text-white">{p.name}</p>
                  <p className="text-xs text-gray-500">{p.description || 'Inference plugin'}</p>
                </div>
              </div>
              {p.loaded ? (
                <span className="badge badge-success flex items-center gap-1"><CheckCircle className="w-3 h-3" /> loaded</span>
              ) : (
                <span className="badge badge-neutral flex items-center gap-1"><XCircle className="w-3 h-3" /> load on demand</span>
              )}
            </div>
          ))}

          {plugins.length === 0 && !refreshing && (
            <p className="px-5 py-6 text-sm text-gray-500">
              No plugins reported. The python sidecar may still be starting up — hit refresh.
            </p>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="card">
          <div className="flex items-center gap-3 mb-3">
            <Cpu className="w-5 h-5 text-amber-400" />
            <p className="font-semibold text-white">Runtime</p>
          </div>
          <dl className="space-y-2 text-sm">
            <div className="flex justify-between"><dt className="text-gray-500">CUDA</dt><dd className="text-gray-200">{health?.cuda_available ? 'available' : 'not available (cpu mode)'}</dd></div>
            <div className="flex justify-between"><dt className="text-gray-500">Device</dt><dd className="text-gray-200">{health?.cuda_available ? health.gpu_name || 'cuda gpu' : 'cpu'}</dd></div>
            {health?.cuda_available && !!health.gpu_memory_total_mb && (
              <div className="flex justify-between"><dt className="text-gray-500">VRAM</dt><dd className="text-gray-200">{health.gpu_memory_total_mb} MB</dd></div>
            )}
          </dl>
        </div>

        <div className="card">
        <div className="flex items-center gap-3 mb-3">
          <HardDrive className="w-5 h-5 text-blue-400" />
          <p className="font-semibold text-white">Plugin Notes</p>
        </div>
        <ul className="space-y-1.5 text-sm text-gray-400">
          <li>• Plugins load on demand — first inference in a modality is slower.</li>
          <li>• Each modality model lives under <span className="font-mono text-xs text-gray-500">python/plugins</span>.</li>
          <li>• No data leaves this machine; inference runs in the local sidecar.</li>
        </ul>
      </div>
      </div>
    </div>
  )
}
