import React, { useState } from 'react'
import { User, Calendar, Ruler, Scale, Edit2, Save, X } from 'lucide-react'

interface PatientContextPanelProps {
  context: any
  onChange: (ctx: any) => void
  compact?: boolean
}

export function PatientContextPanel({ context, onChange, compact = false }: PatientContextPanelProps) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(context)

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

  if (compact) {
    return (
      <div className="flex items-center gap-2">
        <User className="w-3.5 h-3.5 text-[var(--text-faint)]" strokeWidth={1.8} />
        <div className="text-right">
          <p className="text-[11px] text-[var(--text)] truncate max-w-[100px] font-mono">
            {context.patient_id || 'no patient'}
          </p>
          <p className="text-[9.5px] font-mono text-[var(--text-faint)]">
            {context.age && `${context.age}yr `}{context.sex && `${context.sex} `}
            {context.bmi && `bmi ${context.bmi}`}
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h4 className="font-medium text-white flex items-center gap-2">
          <User className="w-3.5 h-3.5 text-[var(--text-faint)]" strokeWidth={1.8} />
          Patient Context
        </h4>
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