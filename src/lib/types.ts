export interface PatientContext {
  patient_id: string
  age?: number
  sex?: string
  height_cm?: number
  weight_kg?: number
  bmi?: number
  medications?: string[]
  allergies?: string[]
  comorbidities?: string[]
}

export interface ModalityResult {
  modality: string
  success: boolean
  results?: Record<string, any>
  metadata?: Record<string, any>
  error?: string
  inference_time_ms?: number
}

export interface FusionResult {
  success: boolean
  fused_results?: Record<string, any>
  modality_contributions?: Record<string, number>
  confidence?: number
  explanation?: string
  error?: string
}

export interface PluginInfo {
  name: string
  loaded: boolean
  description?: string
}

export interface HealthInfo {
  device?: string
  cuda_available: boolean
  gpu_name?: string
  gpu_memory_total_mb?: number
  python_version?: string
  plugins: PluginInfo[]
}
