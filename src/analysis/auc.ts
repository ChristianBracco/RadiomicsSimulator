import { mapOutcome } from "../shared/outcome";
import { featureColumns, mannWhitneyAuc, toFiniteNumber } from "../shared/numeric";

export type FeatureAuc = {
  feature: string;
  auc: number;
  aucDirectionCorrected: number;
  nValid: number;
  nResponders: number;
  nNonResponders: number;
};

export function computeUnivariateAuc(
  rows: Record<string, unknown>[],
  features = featureColumns(rows),
): FeatureAuc[] {
  const out: FeatureAuc[] = [];

  for (const feature of features) {
    const values: Array<{ x: number; y: 0 | 1 }> = [];

    for (const row of rows) {
      const x = toFiniteNumber(row[feature]);
      if (x === null) continue;

      values.push({
        x,
        y: mapOutcome(row.Label),
      });
    }

    const auc = mannWhitneyAuc(values);
    if (auc === null || !Number.isFinite(auc)) continue;

    const nResponders = values.filter((v) => v.y === 1).length;
    const nNonResponders = values.filter((v) => v.y === 0).length;

    out.push({
      feature,
      auc,
      aucDirectionCorrected: Math.max(auc, 1 - auc),
      nValid: values.length,
      nResponders,
      nNonResponders,
    });
  }

  return out.sort((a, b) => b.aucDirectionCorrected - a.aucDirectionCorrected);
}
