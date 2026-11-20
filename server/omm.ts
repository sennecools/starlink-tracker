/**
 * OMM (CCSDS Orbit Mean-Elements Message) CSV parser for CelesTrak GP / SupGP
 * data (`FORMAT=csv`).
 *
 * Why not TLE: the TLE catalog-number field has five characters, so objects
 * with NORAD ids above 99,999 (all new launches since late 2025, incl. every
 * Starlink V3) are not published as TLE at all. OMM carries the full id, a
 * 4-digit-year epoch with microseconds and a proper COSPAR id.
 *
 * Each row becomes an ElementRecord whose `omm` member is a plain object in
 * exactly the shape satellite.js `json2satrec` expects (CelesTrak's OMM JSON:
 * numbers as numbers, EPOCH as an ISO string without zone, implied UTC).
 *
 * Pure (no Node APIs) and fast: ~10k rows parse in a few milliseconds.
 */
import type { ElementRecord, OmmElements } from '../src/shared/contracts.ts';
import { parseTle, tleEpochMs } from './tle.ts';

export interface ParseOmmResult {
  records: ElementRecord[];
  /** Data rows that failed validation (wrong column count, bad numbers, out-of-range elements, bad epoch). */
  rejected: number;
  /** False when the text has no recognisable OMM CSV header (nothing was parsed). */
  headerOk: boolean;
  /** True when an unterminated (cut-off) last line was ignored. */
  truncated: boolean;
}

export interface ParseOmmOptions {
  /**
   * The text is known to be a cut-off download: an unterminated last line is
   * always ignored (it could be cut inside its last column and still parse).
   * Default false: an unterminated last line is kept when it has every column
   * and validates (a complete file without a trailing newline).
   */
  truncated?: boolean;
}

/** Columns a row needs to be propagated. */
const REQUIRED_COLUMNS = [
  'OBJECT_NAME',
  'EPOCH',
  'MEAN_MOTION',
  'ECCENTRICITY',
  'INCLINATION',
  'RA_OF_ASC_NODE',
  'ARG_OF_PERICENTER',
  'MEAN_ANOMALY',
  'NORAD_CAT_ID',
] as const;

// ---------------------------------------------------------------------------
// Field helpers
// ---------------------------------------------------------------------------

const EPOCH_RE = /^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2}):(\d{2}(?:\.\d*)?)Z?$/;

/**
 * OMM EPOCH ("2026-11-20T05:40:30.660672", optional "Z") -> ms since the Unix
 * epoch (fraction below the ms kept as a fractional ms), or NaN.
 */
export function ommEpochMs(epoch: string): number {
  const m = EPOCH_RE.exec(epoch.trim());
  if (!m) return NaN;
  const mo = Number(m[2]);
  const d = Number(m[3]);
  const h = Number(m[4]);
  const mi = Number(m[5]);
  const s = Number(m[6]);
  if (mo < 1 || mo > 12 || d < 1 || d > 31 || h > 23 || mi > 59 || !(s < 61)) return NaN;
  return Date.UTC(Number(m[1]), mo - 1, d, h, mi, 0) + s * 1000;
}

/** Normalise an epoch string to the CelesTrak JSON form "YYYY-MM-DDTHH:MM:SS.ffffff" (no zone). */
function normaliseEpoch(epoch: string): string {
  let e = epoch.trim().replace(' ', 'T');
  if (e.endsWith('Z')) e = e.slice(0, -1);
  return e;
}

/**
 * COSPAR id "2026-256A" -> TLE-style international designator "26256A".
 * An id already in TLE style is returned trimmed; anything else (blank,
 * "UNKNOWN", analyst placeholders) gives "".
 */
export function intlDesignatorFromObjectId(objectId: string): string {
  const id = objectId.trim().toUpperCase();
  const m = /^(\d{4})-(\d{3})([A-Z]{0,3})$/.exec(id);
  if (m) return `${m[1]!.slice(2)}${m[2]}${m[3]}`;
  if (/^\d{5}[A-Z]{0,3}$/.test(id)) return id;
  return '';
}

/** Strict number: blank / non-numeric => NaN (Number('') would be 0). Accepts ".25205656E-3", "+1.5", "-.1". */
function num(field: string | undefined): number {
  if (field === undefined) return NaN;
  const f = field.trim();
  if (f.length === 0) return NaN;
  const v = Number(f);
  return Number.isFinite(v) ? v : NaN;
}

/** Optional number: blank => fallback. */
function numOr(field: string | undefined, fallback: number): number {
  if (field === undefined || field.trim().length === 0) return fallback;
  return num(field);
}

/** Split one CSV line; supports RFC 4180 quoting ("a,b", "say ""hi"""). */
export function splitCsvLine(line: string): string[] {
  if (line.indexOf('"') === -1) return line.split(',');
  const out: string[] = [];
  let cur = '';
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i]!;
    if (inQuotes) {
      if (c === '"') {
        if (line[i + 1] === '"') {
          cur += '"';
          i++;
        } else inQuotes = false;
      } else cur += c;
    } else if (c === '"') inQuotes = true;
    else if (c === ',') {
      out.push(cur);
      cur = '';
    } else cur += c;
  }
  out.push(cur);
  return out;
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/** True when the text starts (after BOM / blank lines) with an OMM CSV header line. */
export function looksLikeOmmCsv(text: string): boolean {
  let i = 0;
  if (text.charCodeAt(0) === 0xfeff) i = 1;
  while (i < text.length && (text[i] === '\r' || text[i] === '\n' || text[i] === ' ' || text[i] === '\t')) i++;
  const end = text.indexOf('\n', i);
  const first = (end === -1 ? text.slice(i) : text.slice(i, end)).toUpperCase();
  return first.includes(',') && first.includes('NORAD_CAT_ID') && first.includes('MEAN_MOTION');
}

/**
 * Parse CelesTrak OMM CSV. Columns are located by the header (any order,
 * case-insensitive, quoted or not); CRLF, a BOM and blank lines are tolerated.
 * Sanity checks per row: mean motion in (0, 20] rev/day, eccentricity in
 * [0, 1), inclination in [0, 180], a parseable epoch and a positive integer
 * NORAD id. BSTAR / MEAN_MOTION_DOT / MEAN_MOTION_DDOT default to 0 when blank.
 */
export function parseOmmCsv(text: string, opts: ParseOmmOptions = {}): ParseOmmResult {
  const records: ElementRecord[] = [];
  let rejected = 0;
  let truncated = false;
  if (text.charCodeAt(0) === 0xfeff) text = text.slice(1);
  const lines = text.split('\n');
  // An unterminated last line is (potentially) cut off.
  const last = lines.length - 1;
  const lastUnterminated = lines[last]!.length > 0;

  // Header: first non-blank line.
  let li = 0;
  while (li < lines.length && lines[li]!.trim().length === 0) li++;
  if (li >= lines.length) return { records, rejected, headerOk: false, truncated };
  const header = splitCsvLine(lines[li]!.replace(/\r$/, '')).map((h) => h.trim().toUpperCase());
  const col = (name: string): number => header.indexOf(name);
  for (const c of REQUIRED_COLUMNS) {
    if (col(c) === -1) return { records, rejected, headerOk: false, truncated };
  }
  if (li === last && lastUnterminated) {
    // Only a header line without a newline: nothing to parse.
    return { records, rejected, headerOk: true, truncated };
  }
  const cName = col('OBJECT_NAME');
  const cId = col('OBJECT_ID');
  const cEpoch = col('EPOCH');
  const cMm = col('MEAN_MOTION');
  const cEcc = col('ECCENTRICITY');
  const cInc = col('INCLINATION');
  const cRaan = col('RA_OF_ASC_NODE');
  const cArgp = col('ARG_OF_PERICENTER');
  const cMa = col('MEAN_ANOMALY');
  const cEph = col('EPHEMERIS_TYPE');
  const cCls = col('CLASSIFICATION_TYPE');
  const cNorad = col('NORAD_CAT_ID');
  const cSet = col('ELEMENT_SET_NO');
  const cRev = col('REV_AT_EPOCH');
  const cBstar = col('BSTAR');
  const cDot = col('MEAN_MOTION_DOT');
  const cDdot = col('MEAN_MOTION_DDOT');
  const nCols = header.length;

  for (let i = li + 1; i <= last; i++) {
    let line = lines[i]!;
    if (line.length > 0 && line.charCodeAt(line.length - 1) === 13) line = line.slice(0, -1);
    if (line.trim().length === 0) continue;
    const f = splitCsvLine(line);
    if (i === last && lastUnterminated && (opts.truncated || f.length !== nCols)) {
      truncated = true; // cut-off final line: ignored, not counted as rejected
      continue;
    }
    if (f.length !== nCols) {
      rejected++;
      continue;
    }
    const noradId = num(f[cNorad]);
    const mm = num(f[cMm]);
    const ecc = num(f[cEcc]);
    const inc = num(f[cInc]);
    const raan = num(f[cRaan]);
    const argp = num(f[cArgp]);
    const ma = num(f[cMa]);
    const bstar = cBstar === -1 ? 0 : numOr(f[cBstar], 0);
    const dot = cDot === -1 ? 0 : numOr(f[cDot], 0);
    const ddot = cDdot === -1 ? 0 : numOr(f[cDdot], 0);
    const setNo = cSet === -1 ? 999 : numOr(f[cSet], 999);
    const rev = cRev === -1 ? NaN : num(f[cRev]);
    const eph = cEph === -1 ? 0 : numOr(f[cEph], 0);
    const epochRaw = f[cEpoch]!.trim();
    const epochMs = ommEpochMs(epochRaw);
    if (
      !(Number.isInteger(noradId) && noradId > 0) ||
      !(mm > 0 && mm <= 20) ||
      !(ecc >= 0 && ecc < 1) ||
      !(inc >= 0 && inc <= 180) ||
      !Number.isFinite(raan) ||
      !Number.isFinite(argp) ||
      !Number.isFinite(ma) ||
      !Number.isFinite(bstar) ||
      !Number.isFinite(dot) ||
      !Number.isFinite(ddot) ||
      !Number.isFinite(setNo) ||
      eph !== 0 || // only SGP4 (type 0) elements can be propagated with SGP4
      !Number.isFinite(epochMs)
    ) {
      rejected++;
      continue;
    }
    const name = f[cName]!.trim();
    const objectId = cId === -1 ? '' : f[cId]!.trim();
    const cls = cCls === -1 ? '' : f[cCls]!.trim().toUpperCase();
    const omm: OmmElements = {
      OBJECT_NAME: name,
      OBJECT_ID: objectId,
      EPOCH: normaliseEpoch(epochRaw),
      MEAN_MOTION: mm,
      ECCENTRICITY: ecc,
      INCLINATION: inc,
      RA_OF_ASC_NODE: raan,
      ARG_OF_PERICENTER: argp,
      MEAN_ANOMALY: ma,
      EPHEMERIS_TYPE: 0,
      CLASSIFICATION_TYPE: cls === 'C' ? 'C' : 'U',
      NORAD_CAT_ID: noradId,
      ELEMENT_SET_NO: setNo,
      BSTAR: bstar,
      MEAN_MOTION_DOT: dot,
      MEAN_MOTION_DDOT: ddot,
    };
    if (Number.isFinite(rev)) omm.REV_AT_EPOCH = rev;
    records.push({
      name: name.length > 0 ? name : `NORAD ${noradId}`,
      noradId,
      intlDesignator: intlDesignatorFromObjectId(objectId),
      omm,
    });
  }
  return { records, rejected, headerOk: true, truncated };
}

// ---------------------------------------------------------------------------
// Format-agnostic helpers (used by the elements service)
// ---------------------------------------------------------------------------

export interface ParseElementsResult {
  records: ElementRecord[];
  rejected: number;
  format: 'omm-csv' | 'tle';
  /** OMM CSV header found, or at least one TLE record parsed. */
  recognised: boolean;
}

/** Parse OMM CSV or TLE / 3LE text, detected from the content. */
export function parseElementsText(text: string, opts: ParseOmmOptions = {}): ParseElementsResult {
  if (looksLikeOmmCsv(text)) {
    const r = parseOmmCsv(text, opts);
    return { records: r.records, rejected: r.rejected, format: 'omm-csv', recognised: r.headerOk };
  }
  const r = parseTle(text);
  return { records: r.records, rejected: r.rejected, format: 'tle', recognised: r.records.length > 0 };
}

/**
 * CelesTrak's answer to a query that matches nothing ("No GP data found").
 * A valid, empty result (e.g. a name partition without any objects), not an error.
 */
export function isNoDataResponse(text: string): boolean {
  return /^\s*No GP data found\b/i.test(text);
}

/** Element-set epoch of any record (OMM or TLE), ms since the Unix epoch, or NaN. */
export function recordEpochMs(record: ElementRecord): number {
  if (record.omm) return ommEpochMs(record.omm.EPOCH);
  if (record.line1 !== undefined) return tleEpochMs(record.line1);
  return NaN;
}

/** Starlink satellite names ("STARLINK-40096", "STARLINK-1008 [DTC]"...). */
export function isStarlinkName(name: string): boolean {
  return /^\s*STARLINK\b/i.test(name);
}

export function filterStarlink(records: readonly ElementRecord[]): ElementRecord[] {
  return records.filter((r) => isStarlinkName(r.name));
}

