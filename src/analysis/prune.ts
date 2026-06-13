import { featureColumns, pearsonFinite } from "../shared/numeric";
import { computeUnivariateAuc } from "./auc";

export type PruneOptions = {
  threshold?: number;
  priority?: "column-order" | "auc";
};

export type PruneResult = {
  threshold: number;
  priority: "column-order" | "auc";
  keptFeatures: string[];
  removedFeatures: string[];
  removedMap: Array<{
    kept: string;
    removed: string;
    r: number;
  }>;
};

export function correlationPrune(
  rows: Record<string, unknown>[],
  options: PruneOptions = {},
): PruneResult {
  const threshold = options.threshold ?? 0.95;
  const priority = options.priority ?? "column-order";

  let features = featureColumns(rows);

  if (priority === "auc") {
    const aucs = computeUnivariateAuc(rows, features);
    const score = new Map(aucs.map((x) => [x.feature, x.aucDirectionCorrected]));
    features = [...features].sort((a, b) => (score.get(b) ?? 0) - (score.get(a) ?? 0));
  }

  const removed = new Set<string>();
  const keptFeatures: string[] = [];
  const removedMap: PruneResult["removedMap"] = [];

  for (const f of features) {
    if (removed.has(f)) continue;

    keptFeatures.push(f);

    for (const g of features) {
      if (f === g) continue;
      if (removed.has(g)) continue;
      if (keptFeatures.includes(g)) continue;

      const r = pearsonFinite(rows, f, g);
      if (r === null) continue;

      if (Math.abs(r) >= threshold) {
        removed.add(g);
        removedMap.push({
          kept: f,
          removed: g,
          r,
        });
      }
    }
  }

  return {
    threshold,
    priority,
    keptFeatures,
    removedFeatures: [...removed],
    removedMap,
  };
}
