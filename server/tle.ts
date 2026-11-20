/**
 * Two-/three-line element set (TLE / 3LE) parser.
 *
 * Pure, allocation-light and fast enough for the full Starlink catalogue
 * (~12k records parse in a few milliseconds). Used by the dev server
 * (server/elementsService.ts); runs fine in a browser or worker too since it
 * only depends on contract types.
 */
import type { ElementRecord } from '../src/shared/contracts.ts';

export interface ParseTleResult {
  records: ElementRecord[];
  /**
   * Number of element sets that were dropped: bad checksum, malformed or
   * mismatched lines, orphan line 1 / line 2, and a truncated final record
   * (line 1 without line 2). Lone name lines without element lines are not
   * counted (they are indistinguishable from junk text).
   */
  rejected: number;
}

const TLE_LINE_LENGTH = 69;

/** Modulo-10 checksum over columns 1-68: digits count their value, '-' counts 1. */
export function tleChecksumOk(line: string): boolean {
  if (line.length < TLE_LINE_LENGTH) return false;
  let sum = 0;
  for (let i = 0; i < TLE_LINE_LENGTH - 1; i++) {
    const c = line.charCodeAt(i);
    if (c >= 48 && c <= 57) sum += c - 48;
    else if (c === 45) sum += 1;
  }
  const expected = line.charCodeAt(TLE_LINE_LENGTH - 1) - 48;
  return expected >= 0 && expected <= 9 && sum % 10 === expected;
}

/**
 * Decode a 5-character TLE catalog-number field. Supports plain numbers
 * (optionally space padded) and Alpha-5 ("A0001" => 100001; letters I and O
 * are not used, so J => 18, P => 23, Z => 33). Returns NaN when invalid.
 */
export function decodeCatalogNumber(field: string): number {
  const f = field.trim();
  if (f.length === 0 || f.length > 5) return NaN;
  const first = f.charCodeAt(0);
  let value = 0;
  let start = 0;
  if (first >= 65 && first <= 90) {
    // Alpha-5: letter + exactly four digits.
    if (f.length !== 5 || first === 73 /* I */ || first === 79 /* O */) return NaN;
    let letter = first - 65 + 10; // A=10
    if (first > 73) letter--; // skip I
    if (first > 79) letter--; // skip O
    value = letter;
    start = 1;
  }
  for (let i = start; i < f.length; i++) {
    const c = f.charCodeAt(i);
    if (c < 48 || c > 57) return NaN;
    value = value * 10 + (c - 48);
  }
  return value;
}

/**
 * Element-set epoch of a TLE line 1 in ms since the Unix epoch, or NaN.
 * Columns 19-32: two-digit year (57-99 => 19xx) + fractional day of year.
 */
export function tleEpochMs(line1: string): number {
  const yy = Number(line1.slice(18, 20));
  const doy = Number(line1.slice(20, 32));
  if (!Number.isFinite(yy) || !Number.isFinite(doy) || line1.length < 32) return NaN;
  const year = yy < 57 ? 2000 + yy : 1900 + yy;
  return Date.UTC(year, 0, 1) + (doy - 1) * 86_400_000;
}

function isBlank(line: string): boolean {
  for (let i = 0; i < line.length; i++) {
    const c = line.charCodeAt(i);
    if (c !== 32 && c !== 9 && c !== 13) return false;
  }
  return true;
}

/** Validate a line-1 / line-2 pair and build the record, or return null. */
function buildRecord(line1: string, line2: string, name: string | null): ElementRecord | null {
  if (line1.length !== TLE_LINE_LENGTH || line2.length !== TLE_LINE_LENGTH) return null;
  if (!tleChecksumOk(line1) || !tleChecksumOk(line2)) return null;
  const cat1 = line1.slice(2, 7);
  const cat2 = line2.slice(2, 7);
  if (cat1 !== cat2) return null;
  const noradId = decodeCatalogNumber(cat1);
  if (!(noradId > 0)) return null;
  return {
    name: name !== null && name.length > 0 ? name : `NORAD ${noradId}`,
    noradId,
    intlDesignator: line1.slice(9, 17).trim(),
    line1,
    line2,
  };
}

function cleanName(line: string): string {
  let name = line.trim();
  // 3LE (Space-Track style) prefixes the name line with "0 ".
  if (name.length > 2 && name.charCodeAt(0) === 48 && name.charCodeAt(1) === 32) name = name.slice(2).trim();
  return name;
}

/**
 * Parse TLE text containing 3-line (name + 2 lines) and/or bare 2-line sets.
 * Tolerates CRLF, trailing whitespace, blank lines and a UTF-8 BOM.
 */
export function parseTle(text: string): ParseTleResult {
  const records: ElementRecord[] = [];
  let rejected = 0;
  if (text.length > 0 && text.charCodeAt(0) === 0xfeff) text = text.slice(1);
  const lines = text.split('\n');
  const n = lines.length;
  let pendingName: string | null = null;

  for (let i = 0; i < n; i++) {
    const raw = lines[i]!;
    if (isBlank(raw)) continue;
    const line = raw.trimEnd();
    const c0 = line.charCodeAt(0);
    const isElementLine = line.length > 1 && line.charCodeAt(1) === 32;

    if (c0 === 49 /* '1' */ && isElementLine) {
      // Find the next non-blank line; it must be line 2.
      let j = i + 1;
      while (j < n && isBlank(lines[j]!)) j++;
      if (j >= n) {
        rejected++; // truncated final record
        pendingName = null;
        break;
      }
      const next = lines[j]!.trimEnd();
      if (next.charCodeAt(0) === 50 /* '2' */ && next.length > 1 && next.charCodeAt(1) === 32) {
        const rec = buildRecord(line, next, pendingName);
        if (rec !== null) records.push(rec);
        else rejected++;
        i = j;
      } else {
        rejected++; // line 1 without line 2; re-examine `next` on the next iteration
      }
      pendingName = null;
    } else if (c0 === 50 /* '2' */ && isElementLine && line.length === TLE_LINE_LENGTH) {
      rejected++; // orphan line 2
      pendingName = null;
    } else {
      pendingName = cleanName(line);
    }
  }
  return { records, rejected };
}
