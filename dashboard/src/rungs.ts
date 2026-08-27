export const RUNG_DESCRIPTIONS: Record<string, string> = {
  FULL: 'Answered directly. Every claim in the answer is backed by a cited source.',
  PARTIAL:
    'Answered partially. Some claims could not be verified against a source, and are listed below.',
  ACTION_DECLINED:
    'The requested action was drafted but not carried out, because this tenant does not permit it.',
  PROVIDER_FALLBACK:
    'The primary model failed or was too slow, so a backup model produced this answer.',
  REFUSED: 'Declined to answer. Not enough evidence was found to answer confidently.',
}
