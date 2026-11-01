// local study history. kept in localStorage — nothing here touches the
// network, matching the local-first identity of the app. capped so a long
// testing session can't bloat the profile.

export interface StudyRecord {
  id: string
  at: number // epoch ms
  modality: 'breath' | 'cxr' | 'spirometry' | 'fusion'
  patient_id: string
  file_name?: string // source file (modality runs)
  ok: boolean // false only for invoke-level failures we choose to keep
  summary: string // one-liner for the timeline row
  probabilities?: Record<string, number> // class -> prob (if the modality has any)
  detail?: Record<string, any> // findings / measurements / explanation for expanded view
  duration_ms?: number
}

const KEY = 'stethos.studies.v1'
const CAP = 1000

export function summarizeResult(modality: string, result: any): string {
  const r = result?.results
  if (!r) return '—'
  switch (modality) {
    case 'breath': {
      const p = r.probabilities?.[r.prediction]
      return `${r.prediction}${p != null ? ` ${Math.round(p * 100)}%` : ''}`
    }
    case 'cxr': {
      const findings = r.positive_findings?.length
        ? `${r.positive_findings.length} finding${r.positive_findings.length > 1 ? 's' : ''}: ${r.positive_findings.slice(0, 3).join(', ')}`
        : 'no positive findings'
      const ctr = r.measurements?.cardiothoracic_ratio
      return ctr != null ? `${findings} · ctr ${ctr.toFixed(2)}` : findings
    }
    case 'spirometry':
      return `${r.pattern}${r.gold_stage ? ` · ${r.gold_stage}` : ''}${r.fev1_fvc_ratio != null ? ` · fev1/fvc ${r.fev1_fvc_ratio.toFixed(2)}` : ''}`
    case 'fusion': {
      const risk = r.risk_level ?? r.fused_results?.risk_level ?? '—'
      const contrib = result.modality_contributions
        ? ' · ' +
          Object.entries(result.modality_contributions)
            .map(([m, v]) => `${m.replace('_', ' ')} ${Math.round((v as number) * 100)}%`)
            .join(', ')
        : ''
      return `${risk} risk${contrib}`
    }
    default:
      return '—'
  }
}

export function loadStudies(): StudyRecord[] {
  try {
    const raw = localStorage.getItem(KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    if (!Array.isArray(parsed)) return []
    return parsed as StudyRecord[]
  } catch {
    return []
  }
}

export function addStudy(rec: StudyRecord): void {
  try {
    const all = [rec, ...loadStudies()].slice(0, CAP)
    localStorage.setItem(KEY, JSON.stringify(all))
  } catch {
    // storage full or unavailable — history is a nicety, never a blocker
  }
}

export function deleteStudy(id: string): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(loadStudies().filter((s) => s.id !== id)))
  } catch {
    /* ignore */
  }
}

export function clearStudies(): void {
  try {
    localStorage.removeItem(KEY)
  } catch {
    /* ignore */
  }
}

export function basename(path: string): string {
  return path.split(/[\\/]/).pop() || path
}
