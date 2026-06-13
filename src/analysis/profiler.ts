import { mapOutcome } from "../shared/outcome";
import { featureColumns } from "../shared/numeric";

export type DatasetProfile = {
  rows: number;
  patients: number;
  responders: number;
  nonResponders: number;
  features: number;
  featureNames: string[];
};

export function profileDataset(rows: Record<string, unknown>[]): DatasetProfile {
  const patients = new Set(rows.map((r) => String(r.Patient ?? "")).filter(Boolean));
  const featureNames = featureColumns(rows);

  let responders = 0;
  let nonResponders = 0;

  for (const row of rows) {
    const y = mapOutcome(row.Label);
    if (y === 1) responders += 1;
    else nonResponders += 1;
  }

  return {
    rows: rows.length,
    patients: patients.size,
    responders,
    nonResponders,
    features: featureNames.length,
    featureNames,
  };
}
