// confidence quality assessment. honest about what the numbers are: entropy
// and margin of the returned distribution. no invented calibration constants —
// with synthetic-trained checkpoints the important job is flagging when the
// label is uninformative rather than dressing it up as a real confidence.

export interface ConfidenceReport {
  level: 'strong' | 'fair' | 'low-signal'
  top1: number
  margin: number
  entropy: number // normalized 0..1 (1 = uniform)
  detail: string
}

// pull the per-class probability map out of a modality result, or null when
// the modality inherently has none (cxr findings list, for example)
export function extractProbabilities(
  modality: string,
  results: any
): Record<string, number> | null {
  if (!results) return null
  const raw =
    modality === 'breath'
      ? results.probabilities
      : modality === 'spirometry'
        ? results.pattern_probabilities
        : modality === 'fusion'
          ? results.pathology_probabilities
          : null
  if (!raw || typeof raw !== 'object') return null
  const probs: Record<string, number> = {}
  let valid = false
  for (const [k, v] of Object.entries(raw)) {
    if (typeof v === 'number' && isFinite(v)) {
      probs[k] = v
      if (v > 0) valid = true
    }
  }
  return valid ? probs : null
}

export function assessConfidence(probs: Record<string, number>): ConfidenceReport | null {
  // normalize defensively (softmax outputs should already sum to 1)
  const entries: [string, number][] = Object.entries(probs).filter(
    ([, v]) => typeof v === 'number' && isFinite(v) && v >= 0
  )
  if (entries.length < 2) return null
  const total = entries.reduce((s, [, v]) => s + v, 0)
  if (total <= 0) return null
  const p = entries.map(([, v]) => v / total)

  const top1 = Math.max(...p)
  const second = p.slice().sort((a, b) => b - a)[1] ?? 0
  const margin = top1 - second

  const n = p.length
  let h = 0
  for (const v of p) if (v > 0) h -= v * Math.log(v)
  const entropy = h / Math.log(n) // 1.0 when uniform

  const level =
    top1 >= 0.55 && margin >= 0.2
      ? 'strong'
      : margin >= 0.08 || entropy <= 0.9
        ? 'fair'
        : 'low-signal'

  const pct = (x: number) => `${Math.round(x * 100)}%`
  const detail =
    `top class ${pct(top1)}, runner-up gap ${pct(margin)}, distribution entropy ${entropy.toFixed(2)}/1.` +
    (level === 'low-signal'
      ? ' the model is not meaningfully confident in this label — treat it as uninformative.'
      : '')

  return { level, top1, margin, entropy, detail }
}
