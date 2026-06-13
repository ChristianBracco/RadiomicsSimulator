import { featureColumns, pearsonFinite } from "../shared/numeric";

export type CorrelationItem = {
  featureA: string;
  featureB: string;
  r: number;
  absR: number;
  nValidPairs: number;
};

export function computeTopCorrelations(
  rows: Record<string, unknown>[],
  topN = 20,
): CorrelationItem[] {
  const features = featureColumns(rows);
  const out: CorrelationItem[] = [];

  for (let i = 0; i < features.length; i++) {
    for (let j = i + 1; j < features.length; j++) {
      const a = features[i];
      const b = features[j];
      const r = pearsonFinite(rows, a, b);

      if (r === null || !Number.isFinite(r)) continue;

      out.push({
        featureA: a,
        featureB: b,
        r,
        absR: Math.abs(r),
        nValidPairs: rows.filter((row) => Number.isFinite(Number(row[a])) && Number.isFinite(Number(row[b]))).length,
      });
    }
  }

  return out.sort((x, y) => y.absR - x.absR).slice(0, topN);
}
