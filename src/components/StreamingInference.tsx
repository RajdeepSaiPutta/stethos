import { useEffect, useRef, useState, useCallback } from 'react'
import { Mic, MicOff, AlertTriangle } from 'lucide-react'

interface StreamingConfig {
  sampleRate: number
  windowDuration: number
  hopDuration: number
  nMels: number
}

interface StreamingResult {
  timestamp: number
  prediction: string
  probabilities: Record<string, number>
  confidence: number
  windowStart: number
  windowEnd: number
}

interface StreamingInferenceProps {
  modality?: 'breath'
  onResult?: (result: StreamingResult) => void
  config?: Partial<StreamingConfig>
  autoStart?: boolean
}

export function StreamingInference({ onResult, config, autoStart = false }: StreamingInferenceProps) {
  // Simple translation fallback - in production would use useI18n()
  const t = (key: string) => {
    const dict: Record<string, string> = {
      'breath_sounds': 'Breath Sounds',
      'streaming': 'Streaming',
      'recording': 'Recording',
      'stopped': 'Stopped',
      'microphone_error': 'Microphone access denied',
      'latest_result': 'Latest Result',
      'start_recording': 'Start Recording',
      'stop_recording': 'Stop Recording'
    }
    return dict[key] || key
  }
  const [isRecording, setIsRecording] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [lastResult, setLastResult] = useState<StreamingResult | null>(null)
  const [audioContext, setAudioContext] = useState<AudioContext | null>(null)
  const [mediaStream, setMediaStream] = useState<MediaStream | null>(null)
  const [audioWorklet, setAudioWorklet] = useState<AudioWorkletNode | null>(null)

  const configRef = useRef<StreamingConfig>({
    sampleRate: 16000,
    windowDuration: 2.0,
    hopDuration: 1.0,
    nMels: 64,
    ...config
  })

  const resultCallbackRef = useRef(onResult)
  resultCallbackRef.current = onResult

  // Initialize audio context
  useEffect(() => {
    const ctx = new AudioContext({ sampleRate: configRef.current.sampleRate })
    setAudioContext(ctx)

    // Load AudioWorklet
    ctx.audioWorklet.addModule('/streaming-worker.js').then(() => {
      console.log('Streaming worklet loaded')
    }).catch(err => {
      console.warn('AudioWorklet not available:', err)
    })

    return () => {
      ctx.close()
    }
  }, [])

  const startRecording = useCallback(async () => {
    if (!audioContext) return

    try {
      setError(null)
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: configRef.current.sampleRate,
          channelCount: 1,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false
        }
      })

      setMediaStream(stream)

      // Create AudioWorkletNode
      const worklet = new AudioWorkletNode(audioContext, 'streaming-processor', {
        processorOptions: {
          windowSize: Math.round(configRef.current.sampleRate * configRef.current.windowDuration),
          hopSize: Math.round(configRef.current.sampleRate * configRef.current.hopDuration),
          sampleRate: configRef.current.sampleRate
        }
      })

      // Connect: microphone -> worklet -> destination (for monitoring)
      const source = audioContext.createMediaStreamSource(stream)
      source.connect(worklet).connect(audioContext.destination)

      worklet.port.onmessage = (e) => {
        if (e.data.type === 'window') {
          // In a real implementation, this would send to WASM model or backend
          // For now, simulate inference
          simulateInference(e.data.data).then(result => {
            setLastResult(result)
            resultCallbackRef.current?.(result)
          })
        }
      }

      setAudioWorklet(worklet)
      setIsRecording(true)
    } catch (err) {
      setError(t('microphone_error'))
      console.error('Recording error:', err)
    }
  }, [audioContext, t])

  const stopRecording = useCallback(() => {
    if (audioWorklet) {
      audioWorklet.disconnect()
      setAudioWorklet(null)
    }
    if (mediaStream) {
      mediaStream.getTracks().forEach(track => track.stop())
      setMediaStream(null)
    }
    setIsRecording(false)
  }, [audioWorklet, mediaStream])

  const toggleRecording = () => {
    if (isRecording) stopRecording()
    else startRecording()
  }

  // Simulate inference (replace with real WASM model or backend call)
  const simulateInference = async (_audioWindow: Float32Array): Promise<StreamingResult> => {
    // Simulate network delay
    await new Promise(r => setTimeout(r, 100))

    // In reality, this would run the BreathCNN model on the audio window
    // For now, return mock data
    const predictions = ['normal', 'crackle', 'wheeze', 'both']
    const pred = predictions[Math.floor(Math.random() * predictions.length)]
    const probs = predictions.map(p => p === pred ? 0.6 + Math.random() * 0.3 : Math.random() * 0.1)

    return {
      timestamp: Date.now(),
      prediction: pred,
      probabilities: Object.fromEntries(predictions.map((p, i) => [p, probs[i]])),
      confidence: probs[0],
      windowStart: Date.now() / 1000 - 2,
      windowEnd: Date.now() / 1000
    }
  }

  if (autoStart) {
    useEffect(() => {
      startRecording()
      return () => stopRecording()
    }, [])
  }

  return (
    <div className="p-4 border border-[var(--border)] rounded-[4px] bg-[var(--chrome)]">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-medium text-white">{t('breath_sounds')} {t('streaming')}</h3>
        <span className={`px-2 py-1 text-xs rounded ${isRecording ? 'bg-green-500/20 text-green-400' : 'bg-gray-500/20 text-gray-400'}`}>
          {isRecording ? t('recording') : t('stopped')}
        </span>
      </div>

      {error && (
        <div className="mb-3 p-2 bg-red-500/10 border border-red-500/30 rounded text-red-400 text-sm flex items-center gap-2">
          <AlertTriangle className="w-4 h-4" />
          {error}
        </div>
      )}

      <div className="flex gap-2 mb-4">
        <button
          onClick={toggleRecording}
          disabled={!audioContext}
          className={`flex-1 px-4 py-2 rounded transition-colors ${isRecording
            ? 'bg-red-500 hover:bg-red-600 text-white'
            : 'bg-[var(--accent)] hover:bg-[var(--accent)]/80 text-white'
          }`}
        >
          <div className="flex items-center justify-center gap-2">
            {isRecording ? <MicOff className="w-4 h-4" /> : <Mic className="w-4 h-4" />}
            {isRecording ? t('stop_recording') : t('start_recording')}
          </div>
        </button>
      </div>

      {lastResult && (
        <div className="space-y-3">
          <h4 className="font-medium text-sm text-[var(--text-muted)]">{t('latest_result')}</h4>
          <div className="p-3 bg-[var(--inset)] rounded">
            <div className="flex items-center justify-between">
              <span className="font-mono text-lg capitalize">{lastResult.prediction}</span>
              <span className="text-sm text-[var(--text-faint)]">
                {Math.round(lastResult.confidence * 100)}%
              </span>
            </div>
            <div className="mt-2 space-y-1">
              {Object.entries(lastResult.probabilities).map(([cls, prob]) => (
                <div key={cls} className="flex items-center gap-2 text-xs">
                  <span className="w-16 capitalize text-[var(--text-muted)]">{cls}</span>
                  <div className="flex-1 h-1.5 bg-[var(--border)] rounded">
                    <div
                      className="h-full rounded"
                      style={{
                        width: `${prob * 100}%`,
                        background: cls === lastResult.prediction ? 'var(--accent)' : 'var(--border-bright)'
                      }}
                    />
                  </div>
                  <span className="w-10 text-right font-mono">{Math.round(prob * 100)}%</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}