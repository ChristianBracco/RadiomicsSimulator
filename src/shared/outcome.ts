/**
 * Outcome mapping unico per tutto il backend.
 *
 * Regola attuale:
 * - Label 1 o 3 => responder = 1
 * - tutto il resto => non responder = 0
 *
 * Usare questa funzione in:
 * - excelLoader.ts
 * - profiler.ts
 * - auc.ts
 * - patient-level analysis
 */
export function mapOutcome(label: unknown): 0 | 1 {
  const v = Number(label);
  return v === 1 || v === 3 ? 1 : 0;
}

export function outcomeLabel(binaryOutcome: unknown): "Responder" | "Non responder" {
  return Number(binaryOutcome) === 1 ? "Responder" : "Non responder";
}
