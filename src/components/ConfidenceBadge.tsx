import { extractProbabilities, assessConfidence } from '../lib/confidence'

// small signal-strength badge for a modality result. renders nothing when the
// modality has no per-class distribution to assess.
export function ConfidenceBadge({ modality, result }: { modality: string; result?: any }) {
  const probs = extractProbabilities(modality, result?.results)
  if (!probs) return null
  const rep = assessConfidence(probs)
  if (!rep) return null

  const cls =
    rep.level === 'strong'
      ? 'badge badge-success'
      : rep.level === 'fair'
        ? 'badge badge-warning'
        : 'badge badge-danger'

  const label =
    rep.level === 'strong' ? 'signal strong' : rep.level === 'fair' ? 'signal fair' : 'low signal'

  return (
    <span className={`${cls} flex-shrink-0`} title={rep.detail}>
      {label}
    </span>
  )
}
