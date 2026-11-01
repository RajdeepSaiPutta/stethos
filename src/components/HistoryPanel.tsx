import React, { useMemo, useState } from 'react'
import { History, Trash2, ChevronDown, ChevronRight, GitCompare } from 'lucide-react'
import {
  loadStudies,
  deleteStudy,
  clearStudies,
  type StudyRecord,
} from '../lib/studies'
import { assessConfidence } from '../lib/confidence'

const MODALITY_LABEL: Record<string, string> = {
  breath: 'breath',
  cxr: 'x-ray',
  spirometry: 'spirometry',
  fusion: 'fusion',
}

// chronological local history of every inference run recorded in this
// browser profile, with same-modality comparison between selected studies.
export function HistoryPanel({ patientId }: { patientId: string }) {
  const [studies, setStudies] = useState<StudyRecord[]>(() => loadStudies())
  const [scope, setScope] = useState<'patient' | 'all'>('patient')
  const [expanded, setExpanded] = useState<Set<string>>(new Set())
  const [selected, setSelected] = useState<Set<string>>(new Set())

  const visible = useMemo(() => {
    return scope === 'patient' && patientId
      ? studies.filter((s) => s.patient_id === patientId)
      : studies
  }, [studies, scope, patientId])

  const groups = useMemo(() => {
    const byDay = new Map<string, StudyRecord[]>()
    for (const s of visible) {
      const day = new Date(s.at).toLocaleDateString(undefined, {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
      })
      const arr = byDay.get(day) ?? []
      arr.push(s)
      byDay.set(day, arr)
    }
    return [...byDay.entries()]
  }, [visible])

  // comparison is same-modality only: the probability axes have to match
  const selectedStudies = useMemo(
    () => selected.size ? [...selected].map((id) => studies.find((s) => s.id === id)).filter(Boolean) as StudyRecord[] : [],
    [selected, studies]
  )
  const compareModality = selectedStudies[0]?.modality
  const comparable = compareModality
    ? selectedStudies.filter((s) => s.modality === compareModality).slice(0, 4)
    : []

  const toggle = (id: string) => {
    setStudies(loadStudies()) // keep in sync with deletions below
    setExpanded((prev) => {
      const next = new Set(prev)
      next.has(id) ? next.delete(id) : next.add(id)
      return next
    })
  }

  const toggleSelect = (id: string, modality: string) => {
    setSelected((prev) => {
      const next = new Set(prev)
      if (next.has(id)) {
        next.delete(id)
      } else {
        // switching modality group resets older picks
        const firstMod = studies.find((s) => next.has(s.id))?.modality
        if (next.size >= 4 || (firstMod && firstMod !== modality)) next.clear()
        next.add(id)
      }
      return next
    })
  }

  const remove = (id: string) => {
    deleteStudy(id)
    setSelected((prev) => { const n = new Set(prev); n.delete(id); return n })
    setStudies(loadStudies())
  }

  const clearAll = () => {
    clearStudies()
    setStudies([])
    setSelected(new Set())
  }

  if (!studies.length) {
    return (
      <div className="max-w-4xl mx-auto">
        <EmptyState patientId={patientId} />
      </div>
    )
  }

  return (
    <div className="max-w-4xl mx-auto space-y-3 animate-fade-in">
      {comparable.length >= 2 && (
        <CompareTable studies={comparable} />
      )}

      <div className="panel">
        <div className="panel-header" style={{ background: 'var(--panel)' }}>
          <span className="flex items-center gap-2">
            <History className="w-3.5 h-3.5 text-[var(--accent)]" />
            <span className="panel-title">Study History</span>
          </span>
          <div className="flex items-center gap-2">
            <div className="seg">
              <button
                type="button"
                className={`seg-item ${scope === 'patient' ? 'is-active' : ''}`}
                disabled={!patientId}
                onClick={() => setScope('patient')}
              >this patient</button>
              <button
                type="button"
                className={`seg-item ${scope === 'all' ? 'is-active' : ''}`}
                onClick={() => setScope('all')}
              >all</button>
            </div>
            <button
              onClick={clearAll}
              className="btn-ghost"
              title="Delete the entire local history"
            >
              <Trash2 className="w-3.5 h-3.5" /> clear
            </button>
          </div>
        </div>

        {scope === 'patient' && !patientId && (
          <div className="px-3.5 py-3 text-[11.5px] text-[var(--text-faint)]">
            set a patient id in the sidebar to filter history to one patient — showing all patients meanwhile.
          </div>
        )}

        {groups.map(([day, recs]) => (
          <div key={day}>
            <div className="px-3.5 py-1.5 text-[10px] uppercase tracking-widest text-[var(--text-faint)]" style={{ borderTop: '1px solid var(--border-soft)', background: 'var(--chrome)' }}>
              {day}
            </div>
            {recs.map((s) => <HistoryRow
              key={s.id}
              study={s}
              expanded={expanded.has(s.id)}
              selected={selected.has(s.id)}
              compareDimmed={!!compareModality && s.modality !== compareModality}
              onToggle={() => toggle(s.id)}
              onSelect={() => toggleSelect(s.id, s.modality)}
              onRemove={() => remove(s.id)}
            />)}
          </div>
        ))}
      </div>

      <p className="text-[10.5px] text-[var(--text-faint)] px-1">
        history is stored locally in this browser profile — nothing leaves the machine. select 2–4 studies of the same modality to compare.
      </p>
    </div>
  )
}

function EmptyState({ patientId }: { patientId: string }) {
  return (
    <div className="panel p-10 text-center animate-fade-in">
      <div className="flex flex-col items-center gap-3">
        <History className="w-6 h-6 text-[var(--text-faint)]" strokeWidth={1.6} />
        <div>
          <p className="text-[12.5px] text-[var(--text-muted)]">no studies recorded yet</p>
          <p className="text-[11.5px] text-[var(--text-faint)] mt-1">
            every inference you run gets a timeline entry here{patientId ? `, tagged ${patientId}` : ''}. run a breath, x-ray or spirometry file to start.
          </p>
        </div>
      </div>
    </div>
  )
}

function HistoryRow({
  study,
  expanded,
  selected,
  compareDimmed,
  onToggle,
  onSelect,
  onRemove,
}: {
  study: StudyRecord
  expanded: boolean
  selected: boolean
  compareDimmed: boolean
  onToggle: () => void
  onSelect: () => void
  onRemove: () => void
}) {
  const probs = study.probabilities
  const rep = probs ? assessConfidence(probs) : null

  return (
    <div style={{ borderTop: '1px solid var(--border-soft)', opacity: compareDimmed && !selected ? 0.45 : undefined }}>
      <div
        className={`table-row-hover flex items-center gap-3 px-3.5 py-2.5 cursor-pointer ${selected ? 'bg-[var(--chrome-2)]' : ''}`}
        style={selected ? { boxShadow: 'inset 2px 0 0 var(--accent)' } : undefined}
        onClick={onToggle}
      >
        {expanded
          ? <ChevronDown className="w-3.5 h-3.5 text-[var(--text-faint)] flex-shrink-0" />
          : <ChevronRight className="w-3.5 h-3.5 text-[var(--text-faint)] flex-shrink-0" />}
        <span className="mono text-[10.5px] text-[var(--text-faint)] flex-shrink-0 w-14">
          {new Date(study.at).toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })}
        </span>
        <span className="badge badge-neutral flex-shrink-0">{MODALITY_LABEL[study.modality] ?? study.modality}</span>
        <span className="flex-1 min-w-0 text-[11.5px] text-[var(--text-muted)] truncate">{study.summary}</span>
        {patientTag(study)}
        {rep && (
          <span
            className={`badge flex-shrink-0 ${
              rep.level === 'strong' ? 'badge-success' : rep.level === 'fair' ? 'badge-warning' : 'badge-danger'
            }`}
            title={rep.detail}
          >
            {rep.level === 'strong' ? 'strong' : rep.level === 'fair' ? 'fair' : 'low'}
          </span>
        )}
        {study.duration_ms != null && (
          <span className="font-mono text-[10px] text-[var(--text-faint)] flex-shrink-0">{study.duration_ms.toFixed(0)} ms</span>
        )}
        <button
          onClick={(e) => { e.stopPropagation(); onSelect() }}
          title="add to comparison (same-modality studies only)"
          className="btn-ghost flex-shrink-0"
        >
          <GitCompare className={`w-3.5 h-3.5 ${selected ? 'text-[var(--accent)]' : ''}`} />
        </button>
        <button
          onClick={(e) => { e.stopPropagation(); onRemove() }}
          title="delete this entry"
          className="btn-ghost flex-shrink-0"
        >
          <Trash2 className="w-3.5 h-3.5" />
        </button>
      </div>

      {expanded && (
        <div className="px-3.5 pb-3.5 pt-1">
          <StudyDetail study={study} />
        </div>
      )}
    </div>
  )
}

function patientTag(study: StudyRecord) {
  if (!study.patient_id) return null
  return (
    <span className="font-mono text-[10px] text-[var(--text-faint)] flex-shrink-0 truncate max-w-28">
      {study.patient_id}
    </span>
  )
}

export function StudyDetail({ study }: { study: StudyRecord }) {
  const d = study.detail
  const well: React.CSSProperties = { borderRadius: 3 }

  return (
    <div className="space-y-2">
      {study.file_name && (
        <p className="mono text-[10px] text-[var(--text-faint)]">
          source <span className="text-[var(--text-muted)]">{study.file_name}</span>
        </p>
      )}

      {study.probabilities && Object.keys(study.probabilities).length > 0 && (
        <div className="space-y-1.5">
          {Object.entries(study.probabilities)
            .slice()
            .sort((a, b) => b[1] - a[1])
            .map(([cls, prob]) => (
              <div key={cls} className="flex items-center gap-3">
                <span className="w-24 text-[11px] text-[var(--text-muted)] capitalize">{cls.replace(/_/g, ' ')}</span>
                <div className="flex-1 h-1 bg-[var(--inset)] overflow-hidden" style={{ borderRadius: 1 }}>
                  <div className="h-full" style={{ width: `${prob * 100}%`, background: 'var(--border-bright)' }} />
                </div>
                <span className="mono text-[11px] text-[var(--text)] w-12 text-right">{(prob * 100).toFixed(1)}%</span>
              </div>
            ))}
        </div>
      )}

      {study.modality === 'cxr' && d?.measurements && typeof d.measurements === 'object' && Object.keys(d.measurements).length > 0 && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-px" style={{ background: 'var(--border-soft)', border: '1px solid var(--border-soft)' }}>
          {Object.entries(d.measurements).map(([k, v]) => (
            <div key={k} className="bg-[var(--panel)] p-2.5">
              <p className="text-[9.5px] uppercase tracking-wider text-[var(--text-faint)]">{k.replace(/_/g, ' ')}</p>
              <p className="mono text-[12px] text-white mt-0.5">{typeof v === 'number' ? v.toFixed(3) : String(v)}</p>
            </div>
          ))}
        </div>
      )}

      {study.modality === 'spirometry' && d?.predicted_values && typeof d.predicted_values === 'object' && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-px" style={{ background: 'var(--border-soft)', border: '1px solid var(--border-soft)' }}>
          {Object.entries(d.predicted_values).map(([k, v]) => (
            <div key={k} className="bg-[var(--panel)] p-2.5">
              <p className="text-[9.5px] uppercase tracking-wider text-[var(--text-faint)]">{k}</p>
              <p className="mono text-[12px] text-white mt-0.5">{typeof v === 'number' ? v.toFixed(2) : String(v)}</p>
            </div>
          ))}
        </div>
      )}

      {study.modality === 'fusion' && typeof d?.explanation === 'string' && (
        <div className="p-3 inset-well" style={well}>
          <p className="text-[12px] text-[var(--text-muted)] leading-relaxed">{d.explanation}</p>
        </div>
      )}

      <p className="mono text-[10px] text-[var(--text-faint)]">
        recorded {new Date(study.at).toLocaleString()}
      </p>
    </div>
  )
}

// side-by-side probability columns for 2-4 same-modality studies
export function CompareTable({ studies }: { studies: StudyRecord[] }) {
  const first = studies[0]

  const classes = useMemo(() => {
    const union = new Set<string>()
    studies.forEach((s) => Object.keys(s.probabilities ?? {}).forEach((k) => union.add(k)))
    return [...union].sort((a, b) => {
      const av = Math.max(...studies.map((s) => s.probabilities?.[a] ?? 0))
      const bv = Math.max(...studies.map((s) => s.probabilities?.[b] ?? 0))
      return bv - av
    })
  }, [studies])

  return (
    <div className="panel animate-fade-in">
      <div className="panel-header" style={{ background: 'var(--panel)' }}>
        <span className="flex items-center gap-2">
          <GitCompare className="w-3.5 h-3.5 text-[var(--accent)]" />
          <span className="panel-title">Comparison · {MODALITY_LABEL[first.modality]}</span>
        </span>
        <span className="mono text-[10px] text-[var(--text-faint)]">
          {studies.length} studies · baseline = leftmost
        </span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full border-collapse">
          <thead>
            <tr>
              <th className="text-left px-3.5 py-2 text-[10px] uppercase tracking-wider text-[var(--text-faint)] font-normal">study</th>
              {studies.map((s, i) => (
                <th key={s.id} className="px-3.5 py-2 text-right text-[10px] uppercase tracking-wider text-[var(--text-faint)] font-normal">
                  <span className="block mono text-[10px] text-[var(--text)]">
                    {new Date(s.at).toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })}
                    {i > 0 && <span className="text-[var(--text-faint)]"> · Δ</span>}
                  </span>
                  {s.patient_id && <span className="block mono text-[9.5px]">{s.patient_id}</span>}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="px-3.5 py-1.5 text-[10px] uppercase tracking-wider text-[var(--text-faint)]">summary</td>
              {studies.map((s) => (
                <td key={`${s.id}-sum`} className="px-3.5 py-1.5 text-right text-[11px] text-[var(--text-muted)]">{s.summary}</td>
              ))}
            </tr>
            {classes.map((cls) => (
              <tr key={cls} className="table-row-hover">
                <td className="px-3.5 py-1.5 text-[11px] text-[var(--text-muted)] capitalize">{cls.replace(/_/g, ' ')}</td>
                {studies.map((s, i) => {
                  const v = s.probabilities?.[cls]
                  const base = i > 0 ? studies[0].probabilities?.[cls] : undefined
                  const delta = i > 0 && v != null && base != null ? v - base : null
                  const bigger = delta != null && Math.abs(delta) > 0.02
                  return (
                    <td key={`${s.id}-${cls}`} className="px-3.5 py-1.5 text-right mono text-[11.5px] text-[var(--text)]">
                      {v != null ? (v * 100).toFixed(1) + '%' : '—'}
                      {delta != null && (
                        <span
                          className="ml-1.5 text-[10px]"
                          style={{ color: bigger ? (delta > 0 ? 'var(--good)' : 'var(--bad)') : undefined }}
                        >
                          {delta > 0 ? '+' : ''}{bigger ? (delta * 100).toFixed(1) : ''}
                        </span>
                      )}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
