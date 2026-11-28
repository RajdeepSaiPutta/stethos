import { useState, useCallback } from 'react'
import { Save, RotateCcw, AlertCircle } from 'lucide-react'

interface SpirometryData {
  // Patient info
  age: string
  sex: 'M' | 'F' | 'O'
  height_cm: string
  weight_kg: string
  
  // Core PFT parameters
  FVC: string
  FEV1: string
  FEV1_FVC: string
  PEF: string
  FEF25_75: string
  FEF25: string
  FEF50: string
  FEF75: string
  PIF: string
  FIVC: string
  TLC: string
  RV: string
  DLCO: string
}

const PARAMETER_DEFAULTS: Partial<SpirometryData> = {
  FVC: '4.0',
  FEV1: '3.2',
  FEV1_FVC: '0.80',
  PEF: '8.5',
  FEF25_75: '4.5',
  FEF25: '7.5',
  FEF50: '5.0',
  FEF75: '3.5',
  PIF: '7.0',
  FIVC: '4.0',
  TLC: '6.0',
  RV: '1.5',
  DLCO: '25.0',
}

const PARAMETER_UNITS: Record<string, string> = {
  FVC: 'L',
  FEV1: 'L',
  FEV1_FVC: 'ratio',
  PEF: 'L/s',
  FEF25_75: 'L/s',
  FEF25: 'L/s',
  FEF50: 'L/s',
  FEF75: 'L/s',
  PIF: 'L/s',
  FIVC: 'L',
  TLC: 'L',
  RV: 'L',
  DLCO: 'mL/min/mmHg',
}

const PARAMETER_LABELS: Record<string, string> = {
  FVC: 'FVC (Forced Vital Capacity)',
  FEV1: 'FEV₁ (Forced Exp. Volume in 1s)',
  FEV1_FVC: 'FEV₁/FVC Ratio',
  PEF: 'PEF (Peak Expiratory Flow)',
  FEF25_75: 'FEF₂₅₋₇₅ (Mid-Exp. Flow)',
  FEF25: 'FEF₂₅',
  FEF50: 'FEF₅₀',
  FEF75: 'FEF₇₅',
  PIF: 'PIF (Peak Inspiratory Flow)',
  FIVC: 'FIVC (Forced Inspiratory Vital Cap.)',
  TLC: 'TLC (Total Lung Capacity)',
  RV: 'RV (Residual Volume)',
  DLCO: 'DLCO (Diffusing Capacity)',
}

const PARAMETER_GROUPS = [
  { key: 'core', label: 'Core Spirometry', params: ['FVC', 'FEV1', 'FEV1_FVC', 'PEF', 'FEF25_75'] },
  { key: 'flows', label: 'Flow Parameters', params: ['FEF25', 'FEF50', 'FEF75', 'PIF'] },
  { key: 'volumes', label: 'Lung Volumes', params: ['FIVC', 'TLC', 'RV'] },
  { key: 'gas', label: 'Gas Exchange', params: ['DLCO'] },
]

export function SpirometryRecorder({
  onSave,
  onCancel,
  initialData,
}: {
  onSave: (data: SpirometryData, format: 'json' | 'csv') => void
  onCancel: () => void
  initialData?: Partial<SpirometryData>
}) {
  const [data, setData] = useState<SpirometryData>({
    age: '',
    sex: 'M',
    height_cm: '',
    weight_kg: '',
    ...PARAMETER_DEFAULTS,
    ...initialData,
  } as SpirometryData)
  
  const [activeTab, setActiveTab] = useState<'patient' | 'core' | 'flows' | 'volumes' | 'gas'>('patient')
  const [validationErrors, setValidationErrors] = useState<Record<string, string>>({})

  const validate = useCallback(() => {
    const errors: Record<string, string> = {}
    
    if (!data.age || parseInt(data.age) < 1 || parseInt(data.age) > 120) {
      errors.age = 'Age must be between 1 and 120'
    }
    if (!data.height_cm || parseFloat(data.height_cm) < 50 || parseFloat(data.height_cm) > 250) {
      errors.height_cm = 'Height must be between 50-250 cm'
    }
    if (!data.weight_kg || parseFloat(data.weight_kg) < 20 || parseFloat(data.weight_kg) > 300) {
      errors.weight_kg = 'Weight must be between 20-300 kg'
    }
    
    const coreParams = ['FVC', 'FEV1', 'FEV1_FVC', 'PEF', 'FEF25_75']
    coreParams.forEach(param => {
      const val = parseFloat(data[param as keyof SpirometryData])
      if (isNaN(val) || val < 0) {
        errors[param] = `${PARAMETER_LABELS[param]} is required`
      }
    })
    
    if (parseFloat(data.FEV1) > parseFloat(data.FVC)) {
      errors.FEV1 = 'FEV₁ cannot exceed FVC'
    }
    if (parseFloat(data.FEV1_FVC) > 1 || parseFloat(data.FEV1_FVC) < 0) {
      errors.FEV1_FVC = 'FEV₁/FVC ratio must be between 0 and 1'
    }
    
    setValidationErrors(errors)
    return Object.keys(errors).length === 0
  }, [data])

  const handleChange = (field: keyof SpirometryData, value: string) => {
    setData(prev => ({ ...prev, [field]: value }))
    if (validationErrors[field]) {
      setValidationErrors(prev => { const n = { ...prev }; delete n[field]; return n })
    }
  }

  const handleSave = (format: 'json' | 'csv') => {
    if (!validate()) return
    onSave(data, format)
  }

  const handleReset = () => {
    setData({
      age: '',
      sex: 'M',
      height_cm: '',
      weight_kg: '',
      ...PARAMETER_DEFAULTS,
    } as SpirometryData)
    setValidationErrors({})
  }

  const tabs = [
    { id: 'patient', label: 'Patient', icon: null },
    { id: 'core', label: 'Core', icon: null },
    { id: 'flows', label: 'Flows', icon: null },
    { id: 'volumes', label: 'Volumes', icon: null },
    { id: 'gas', label: 'Gas', icon: null },
  ] as const

  return (
    <div className="flex-1 flex flex-col max-w-4xl mx-auto p-4">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <h2 className="text-[13px] font-medium text-white">Record Spirometry Test</h2>
          <p className="text-[11.5px] text-[var(--text-faint)] mt-0.5">
            Enter pulmonary function test values manually
          </p>
        </div>
        <div className="flex gap-2">
          <button onClick={onCancel} className="btn-ghost text-[11px]">
            Cancel
          </button>
        </div>
      </div>

      <div className="flex-1 flex flex-col overflow-hidden panel" style={{ minHeight: 0 }}>
        {/* Tab bar */}
        <div className="flex gap-1 p-1 bg-[var(--inset)]" style={{ borderRadius: 4 }}>
          {tabs.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as typeof activeTab)}
              className={`px-3 py-1.5 text-[11px] font-mono rounded-[3px] transition-colors ${
                activeTab === tab.id
                  ? 'bg-[var(--panel)] text-white shadow-sm'
                  : 'text-[var(--text-muted)] hover:text-[var(--text)]'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Tab content */}
        <div className="flex-1 overflow-y-auto p-4">
          {activeTab === 'patient' && (
            <PatientInfoForm data={data} onChange={handleChange} errors={validationErrors} />
          )}
          {activeTab === 'core' && (
            <ParameterGroupForm
              title="Core Spirometry"
              params={PARAMETER_GROUPS[0].params}
              data={data}
              onChange={handleChange}
              errors={validationErrors}
            />
          )}
          {activeTab === 'flows' && (
            <ParameterGroupForm
              title="Flow Parameters"
              params={PARAMETER_GROUPS[1].params}
              data={data}
              onChange={handleChange}
              errors={validationErrors}
            />
          )}
          {activeTab === 'volumes' && (
            <ParameterGroupForm
              title="Lung Volumes"
              params={PARAMETER_GROUPS[2].params}
              data={data}
              onChange={handleChange}
              errors={validationErrors}
            />
          )}
          {activeTab === 'gas' && (
            <ParameterGroupForm
              title="Gas Exchange"
              params={PARAMETER_GROUPS[3].params}
              data={data}
              onChange={handleChange}
              errors={validationErrors}
            />
          )}
        </div>

        {/* Action bar */}
        <div className="p-4 border-t border-[var(--border-soft)] flex items-center justify-between">
          <div className="flex gap-2">
            <button onClick={handleReset} className="btn-ghost text-[11px]">
              <RotateCcw className="w-3.5 h-3.5 mr-1.5" />
              Reset to defaults
            </button>
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => handleSave('json')}
              disabled={Object.keys(validationErrors).length > 0}
              className="btn-primary text-[11px]"
            >
              <Save className="w-3.5 h-3.5 mr-1.5" />
              Save as JSON
            </button>
            <button
              onClick={() => handleSave('csv')}
              disabled={Object.keys(validationErrors).length > 0}
              className="btn-secondary text-[11px]"
            >
              <Save className="w-3.5 h-3.5 mr-1.5" />
              Save as CSV
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

function PatientInfoForm({
  data,
  onChange,
  errors,
}: {
  data: SpirometryData
  onChange: (field: keyof SpirometryData, value: string) => void
  errors: Record<string, string>
}) {
  const bmi = data.height_cm && data.weight_kg
    ? (parseFloat(data.weight_kg) / ((parseFloat(data.height_cm) / 100) ** 2)).toFixed(1)
    : null

  return (
    <div className="space-y-4 max-w-md">
      <div className="grid grid-cols-2 gap-3">
        <InputField
          label="Age"
          value={data.age}
          onChange={v => onChange('age', v)}
          type="number"
          placeholder="e.g., 45"
          error={errors.age}
          unit="years"
        />
        <SelectField
          label="Sex"
          value={data.sex}
          onChange={v => onChange('sex', v as 'M' | 'F' | 'O')}
          options={[
            { value: 'M', label: 'Male' },
            { value: 'F', label: 'Female' },
            { value: 'O', label: 'Other' },
          ]}
          error={errors.sex}
        />
      </div>
      <div className="grid grid-cols-2 gap-3">
        <InputField
          label="Height"
          value={data.height_cm}
          onChange={v => onChange('height_cm', v)}
          type="number"
          placeholder="e.g., 175"
          error={errors.height_cm}
          unit="cm"
        />
        <InputField
          label="Weight"
          value={data.weight_kg}
          onChange={v => onChange('weight_kg', v)}
          type="number"
          placeholder="e.g., 70"
          error={errors.weight_kg}
          unit="kg"
        />
      </div>
      {bmi && (
        <div className="p-3 inset-well" style={{ borderRadius: 3 }}>
          <p className="text-[12px] text-[var(--text-muted)]">
            Calculated BMI: <span className="mono text-white font-medium">{bmi}</span>
          </p>
        </div>
      )}
      <div className="p-3" style={{ background: 'var(--accent-dim)', borderRadius: 4, border: '1px solid var(--accent)' }}>
        <p className="text-[11px] text-[var(--accent)]">
          <strong>Tip:</strong> Enter core spirometry values (FVC, FEV₁, FEV₁/FVC, PEF, FEF₂₅₋₇₅) in the <strong>Core</strong> tab.
          Additional flow parameters, lung volumes, and DLCO are optional but improve analysis accuracy.
        </p>
      </div>
    </div>
  )
}

function ParameterGroupForm({
  title,
  params,
  data,
  onChange,
  errors,
}: {
  title: string
  params: string[]
  data: SpirometryData
  onChange: (field: keyof SpirometryData, value: string) => void
  errors: Record<string, string>
}) {
  return (
    <div className="space-y-3 max-w-xl">
      <p className="text-[11px] text-[var(--text-faint)] uppercase tracking-wider">{title}</p>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {params.map(param => {
            const key = param as keyof SpirometryData
            return (
              <InputField
                key={param}
                label={PARAMETER_LABELS[param]}
                value={data[key] ?? ''}
                onChange={v => onChange(key, v)}
                type="number"
                step="0.01"
                placeholder={`e.g., ${PARAMETER_DEFAULTS[key] || '0.0'}`}
                error={errors[param]}
                unit={PARAMETER_UNITS[param]}
                required={['FVC', 'FEV1', 'FEV1_FVC', 'PEF', 'FEF25_75'].includes(param)}
              />
            )
          })}
      </div>
    </div>
  )
}

function InputField({
  label,
  value,
  onChange,
  type = 'text',
  step,
  placeholder,
  error,
  unit,
  required = false,
}: {
  label: string
  value: string
  onChange: (value: string) => void
  type?: string
  step?: string
  placeholder?: string
  error?: string
  unit?: string
  required?: boolean
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <label className="flex items-center gap-1.5 text-[11px] text-[var(--text-muted)]">
        <span>{label}</span>
        {required && <span className="text-[var(--bad)]" aria-hidden="true">*</span>}
        {unit && <span className="font-mono text-[9.5px] text-[var(--text-faint)]">({unit})</span>}
      </label>
      <div className="relative">
        <input
          type={type}
          step={step}
          value={value}
          onChange={e => onChange(e.target.value)}
          placeholder={placeholder}
          className={`input text-sm w-full ${
            error ? 'border-[var(--bad)] focus:border-[var(--bad)]' : ''
          }`}
          aria-invalid={!!error}
          aria-describedby={error ? `${label}-error` : undefined}
        />
        {error && (
          <p id={`${label}-error`} className="absolute bottom-full left-0 mb-1 text-[10px]" style={{ color: 'var(--bad)' }}>
            <AlertCircle className="w-3 h-3 inline-block mr-1" /> {error}
          </p>
        )}
      </div>
    </div>
  )
}

function SelectField({
  label,
  value,
  onChange,
  options,
  error,
}: {
  label: string
  value: string
  onChange: (value: string) => void
  options: { value: string; label: string }[]
  error?: string
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <label className="text-[11px] text-[var(--text-muted)]">{label}</label>
      <div className="relative">
        <select
          value={value}
          onChange={e => onChange(e.target.value)}
          className={`input text-sm w-full ${error ? 'border-[var(--bad)] focus:border-[var(--bad)]' : ''}`}
          aria-invalid={!!error}
        >
          {options.map(opt => (
            <option key={opt.value} value={opt.value}>{opt.label}</option>
          ))}
        </select>
        {error && (
          <p className="absolute bottom-full left-0 mb-1 text-[10px]" style={{ color: 'var(--bad)' }}>
            <AlertCircle className="w-3 h-3 inline-block mr-1" /> {error}
          </p>
        )}
      </div>
    </div>
  )
}