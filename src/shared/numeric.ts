export function toFiniteNumber(value: unknown): number | null {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

export function finitePairs(
  rows: Record<string, unknown>[],
  a: string,
  b: string,
): Array<[number, number]> {
  const out: Array<[number, number]> = [];

  for (const row of rows) {
    const x = toFiniteNumber(row[a]);
    const y = toFiniteNumber(row[b]);

    if (x !== null && y !== null) {
      out.push([x, y]);
    }
  }

  return out;
}

export function pearsonFromPairs(pairs: Array<[number, number]>): number | null {
  const n = pairs.length;
  if (n < 3) return null;

  let sx = 0;
  let sy = 0;
  let sxx = 0;
  let syy = 0;
  let sxy = 0;

  for (const [x, y] of pairs) {
    sx += x;
    sy += y;
    sxx += x * x;
    syy += y * y;
    sxy += x * y;
  }

  const num = n * sxy - sx * sy;
  const denX = n * sxx - sx * sx;
  const denY = n * syy - sy * sy;

  if (denX <= 0 || denY <= 0) return null;

  return num / Math.sqrt(denX * denY);
}

export function pearsonFinite(
  rows: Record<string, unknown>[],
  a: string,
  b: string,
): number | null {
  return pearsonFromPairs(finitePairs(rows, a, b));
}

export function mannWhitneyAuc(
  values: Array<{ x: number; y: 0 | 1 }>,
): number | null {
  const pos = values.filter((v) => v.y === 1).map((v) => v.x);
  const neg = values.filter((v) => v.y === 0).map((v) => v.x);

  if (pos.length === 0 || neg.length === 0) return null;

  let wins = 0;
  let ties = 0;

  for (const p of pos) {
    for (const n of neg) {
      if (p > n) wins += 1;
      else if (p === n) ties += 1;
    }
  }

  return (wins + 0.5 * ties) / (pos.length * neg.length);
}

export function featureColumns(
  rows: Record<string, unknown>[],
  excluded = new Set(["Patient", "Label", "Organ", "BinaryOutcome"]),
): string[] {
  const first = rows[0] ?? {};

  return Object.keys(first).filter((k) => {
    if (excluded.has(k)) return false;

    return rows.some((row) => toFiniteNumber(row[k]) !== null);
  });
}
