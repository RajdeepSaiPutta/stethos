import React, { useState } from 'react'
import { User, Calendar, Ruler, Scale, Edit2, Save, X, Plus, Users } from 'lucide-react'

interface PatientContextPanelProps {
  context: any
  onChange: (ctx: any) => void
  compact?: boolean
  // New props for patient switching
  registry?: Record<string, any>
  currentPatientId?: string
  onSwitchPatient?: (id: string) => void
  onCreatePatient?: (id: string, ctx: any) => void
}

export function PatientContextPanel({ 
  context, 
  onChange, 
  compact = false,
  registry = {},
  currentPatientId = '',
  onSwitchPatient,
  onCreatePatient,
}: PatientContextPanelProps) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(context)
  const [showSwitcher, setShowSwitcher] = useState(false)
  const [newPatientId, setNewPatientId] = useState('')

  const handleSave = () => {
    const bmi = draft.height_cm && draft.weight_kg
      ? (draft.weight_kg / ((draft.height_cm / 100) ** 2)).toFixed(1)
      : undefined
    onChange({ ...draft, bmi: bmi ? parseFloat(bmi) : undefined })
    setEditing(false)
  }

  const handleCancel = () => {
    setDraft(context)
    setEditing(false)
  }

  const handleCreatePatient = () => {
    if (!newPatientId.trim()) return
    onCreatePatient?.(newPatientId.trim(), { ...draft, patient_id: newPatientId.trim() })
    setNewPatientId('')
    setShowSwitcher(false)
  }

  if (compact) {
    return (
      <div className="flex items-center gap-2 relative">
        <Users className="w-3.5 h-3.5 text-[var(--text-faint)]" strokeWidth={1.8} />
        <div className="text-right flex-1 min-w-0">
          <p className="text-[11px] text-[var(--text)] truncate max-w-[100px] font-mono">
            {context.patient_id || 'no patient'}
          </p>
          <p className="text-[9.5px] font-mono text-[var(--text-faint)]">
            {context.age && `${context.age}yr `}{context.sex && `${context.sex} `}
            {context.bmi && `bmi ${context.bmi}`}
          </p>
        </div>
        {onSwitchPatient && (
          <button 
            onClick={() => setShowSwitcher(!showSwitcher)}
            className="btn-ghost p-1.5 hover:bg-[var(--chrome-2)]"
            title="Switch patient"
          >
            <User className="w-3.5 h-3.5" strokeWidth={1.8} />
          </button>
        )}
        {showSwitcher && onSwitchPatient && (
          <div className="absolute bottom-full left-0 mb-1 w-48 bg-[var(--chrome)] border border-[var(--border)] rounded-[4px] p-2 shadow-lg z-20 animate-fade-in">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-xs font-medium text-[var(--text)]">Patients</span>
              <button onClick={() => setShowSwitcher(false)} className="ml-auto btn-ghost p-0.5">
                <X className="w-3 h-3" />
              </button>
            </div>
            {Object.entries(registry).map(([id, p]) => (
              <button
                key={id}
                onClick={() => { onSwitchPatient(id); setShowSwitcher(false); }}
                className={`w-full text-left px-2 py-1.5 text-xs rounded-[3px] transition-colors ${
                  id === currentPatientId ? 'bg-[var(--accent-dim)] text-[var(--accent)]' : 'hover:bg-[var(--chrome-2)]'
                }`}
              >
                <div className="font-mono">{id}</div>
                <div className="text-[9.5px] text-[var(--text-faint)]">
                  {p.age && `${p.age}yr `}{p.sex && `${p.sex} `}{p.bmi && `bmi ${p.bmi}`}
                </div>
              </button>
            ))}
            {!Object.keys(registry).length && (
              <p className="text-xs text-[var(--text-faint)] px-2 py-2">No patients yet</p>
            )}
            <hr className="my-2 border-[var(--border-soft)]" />
            <div className="space-y-1">
              <input
                type="text"
                value={newPatientId}
                onChange={e => setNewPatientId(e.target.value)}
                placeholder="New patient ID (e.g. PAT-002)"
                className="input text-xs w-full"
                onKeyDown={e => e.key === 'Enter' && handleCreatePatient()}
                autoFocus
              />
              <button onClick={handleCreatePatient} className="btn-primary text-xs w-full py-1">
                <Plus className="w-3 h-3 mr-1" /> Create
              </button>
            </div>
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h4 className="font-medium text-white flex items-center gap-2">
          <Users className="w-3.5 h-3.5 text-[var(--text-faint)]" strokeWidth={1.8} />
          Patient Context
        </h4>
        <div className="flex items-center gap-2">
          {editing ? (
            <div className="flex gap-2">
              <button onClick={handleCancel} className="btn-ghost p-1.5" title="Cancel">
                <X className="w-4 h-4" />
              </button>
              <button onClick={handleSave} className="btn-primary p-1.5" title="Save">
                <Save className="w-4 h-4" />
              </button>
            </div>
          ) : (
            <button onClick={() => setEditing(true)} className="btn-ghost p-1.5" title="Edit">
              <Edit2 className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      {onSwitchPatient && Object.keys(registry).length > 1 && (
        <div className="mb-2">
          <label className="text-xs text-gray-500 mb-1 block">Switch Patient</label>
          <select
            value={currentPatientId}
            onChange={e => onSwitchPatient(e.target.value)}
            className="input text-xs w-full"
            disabled={editing}
          >
            {Object.entries(registry).map(([id, p]) => (
              <option key={id} value={id}>
                {id} {p.age && `(${p.age}yr)`} {p.sex && p.sex} {p.bmi && `bmi ${p.bmi}`}
              </option>
            ))}
          </select>
        </div>
      )}

      <div className="grid grid-cols-2 gap-3">
        <Field
          label="Patient ID"
          value={draft.patient_id}
          onChange={e => setDraft({ ...draft, patient_id: e.target.value })}
          disabled={!editing}
          placeholder="PAT-001"
          icon={<User className="w-4 h-4" />}
        />
        <Field
          label="Age"
          type="number"
          value={draft.age || ''}
          onChange={e => setDraft({ ...draft, age: parseInt(e.target.value) || undefined })}
          disabled={!editing}
          placeholder="45"
          icon={<Calendar className="w-4 h-4" />}
        />
        <Field
          label="Sex"
          value={draft.sex}
          onChange={e => setDraft({ ...draft, sex: e.target.value })}
          disabled={!editing}
          placeholder="M / F / Other"
          icon={<User className="w-4 h-4" />}
        />
        <Field
          label="Height (cm)"
          type="number"
          value={draft.height_cm || ''}
          onChange={e => setDraft({ ...draft, height_cm: parseInt(e.target.value) || undefined })}
          disabled={!editing}
          placeholder="175"
          icon={<Ruler className="w-4 h-4" />}
        />
        <Field
          label="Weight (kg)"
          type="number"
          value={draft.weight_kg || ''}
          onChange={e => setDraft({ ...draft, weight_kg: parseInt(e.target.value) || undefined })}
          disabled={!editing}
          placeholder="70"
          icon={<Scale className="w-4 h-4" />}
        />
        <Field
          label="BMI"
          value={draft.bmi?.toFixed(1) || ''}
          disabled={true}
          placeholder="Auto"
          icon={<Scale className="w-4 h-4" />}
        />
      </div>

      {editing && (
        <div className="pt-2 border-t border-gray-800">
          <p className="text-xs text-gray-500">
            BMI auto-calculates from height/weight. Click Save to update.
          </p>
        </div>
      )}
    </div>
  )
}

interface FieldProps {
  label: string
  value: string | number | undefined
  onChange?: (e: React.ChangeEvent<HTMLInputElement>) => void
  type?: string
  disabled?: boolean
  placeholder?: string
  icon?: React.ReactNode
}

function Field({ label, value, onChange, type = 'text', disabled, placeholder, icon }: FieldProps) {
  return (
    <div className="space-y-1">
      <label className="text-xs text-gray-500 flex items-center gap-1">{label} {icon && <span className="text-gray-600">{icon}</span>}</label>
      <input
        type={type}
        value={value}
        onChange={onChange}
        disabled={disabled}
        placeholder={placeholder}
        className={`input ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
      />
    </div>
  )
}