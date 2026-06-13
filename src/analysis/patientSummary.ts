import { featureColumns, toFiniteNumber } from "../shared/numeric";
import { mapOutcome } from "../shared/outcome";

export function aggregateByPatient(rows: Record<string, unknown>[]): Record<string, unknown>[] {
  const features = featureColumns(rows);
  const byPatient = new Map<string, Record<string, unknown>[]>();

  for (const row of rows) {
    const patient = String(row.Patient ?? "");
    if (!patient) continue;

    if (!byPatient.has(patient)) byPatient.set(patient, []);
    byPatient.get(patient)!.push(row);
  }

  const out: Record<string, unknown>[] = [];

  for (const [patient, patientRows] of byPatient.entries()) {
    const item: Record<string, unknown> = {
      Patient: patient,
      BinaryOutcome: mapOutcome(patientRows[0].Label),
      nLesions: patientRows.length,
    };

    for (const feature of features) {
      const values = patientRows
        .map((r) => toFiniteNumber(r[feature]))
        .filter((x): x is number => x !== null);

      if (values.length === 0) continue;

      item[`${feature}_mean`] = values.reduce((a, b) => a + b, 0) / values.length;
      item[`${feature}_max`] = Math.max(...values);
    }

    out.push(item);
  }

  return out;
}
