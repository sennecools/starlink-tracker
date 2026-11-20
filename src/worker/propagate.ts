/**
 * SGP4 propagation core. Pure and environment-agnostic (no DOM / worker
 * globals) so it runs in the Web Worker, in Node tests and anywhere else.
 *
 * Hot path design (`propagateInto`):
 *   - The Julian date, GMST (`gstime`) and its sin/cos are computed ONCE per call.
 *   - Per satellite we call satellite.js' `sgp4(satrec, minutesSinceEpoch)`
 *     directly instead of `propagate(satrec, date)`: identical arithmetic
 *     (propagate does `(jday(date) - jdsatepoch) * 1440` then `sgp4`) but
 *     without a per-satellite `jday`, `args.slice` and options parsing.
 *   - The ECI -> ECEF rotation is inlined (same formula as `eciToEcf`), so the
 *     only allocations are the result objects sgp4 itself returns.
 * Plain JS was chosen over the WASM BulkPropagator: ~5 ms for 10k satellites
 * once per second in a worker is far below any budget, and the JS path has no
 * async WASM instantiation, no extra asset to serve and results identical to
 * the reference implementation.
 */
import { type SatRec, gstime, jday, json2satrec, sgp4, twoline2satrec } from 'satellite.js';
import type { ElementRecord, SatCatalog } from '../shared/contracts.ts';
import { estimateHardware } from '../shared/hardware.ts';
import { classifyShell } from '../shared/shells.ts';

/** Earth gravitational parameter, km^3/s^2 (WGS-84 / EGM-96). */
const MU_KM3_S2 = 398600.4418;
/** WGS-84 equatorial radius, km. */
const EARTH_RADIUS_KM = 6378.137;
/** Julian date of the Unix epoch (1970-01-01T00:00:00Z). */
const JD_UNIX_EPOCH = 2440587.5;
const MS_PER_DAY = 86_400_000;
const MINUTES_PER_DAY = 1440;
const TWO_PI = 2 * Math.PI;
const RAD2DEG = 180 / Math.PI;

export interface BuiltCatalog {
  /** One initialised SGP4 record per accepted satellite, same index as the catalog. */
  satrecs: SatRec[];
  catalog: SatCatalog;
  /** Number of input records dropped (unparseable, SGP4 init error, or failing at epoch). */
  rejected: number;
}

/**
 * Julian date (UTC) for a Unix time in ms, via satellite.js' own `jday` so the
 * result matches `propagate(satrec, new Date(timeMs))` bit for bit.
 */
export function julianDateFromMs(timeMs: number): number {
  return jday(new Date(timeMs));
}

/** Element-set epoch of an initialised satrec, ms since Unix epoch (rounded to the ms). */
export function satrecEpochMs(satrec: SatRec): number {
  return Math.round((satrec.jdsatepoch - JD_UNIX_EPOCH) * MS_PER_DAY);
}

/**
 * Initialise the SGP4 record of one element record: OMM via json2satrec
 * (preferred, the only format with NORAD ids > 99,999), else the TLE lines via
 * twoline2satrec. Throws when the record has neither. Callers check
 * `satrec.error` themselves (buildCatalog / satDetails.makeSatrec).
 */
export function recordToSatrec(record: ElementRecord): SatRec {
  if (record.omm) return json2satrec(record.omm);
  if (typeof record.line1 === 'string' && typeof record.line2 === 'string') return twoline2satrec(record.line1, record.line2);
  throw new Error(`No orbital elements (neither OMM nor TLE) for ${record.name} (${record.noradId})`);
}

/**
 * Parse and initialise every record and derive the static catalog.
 * Accepted satellites keep their input order; index i of `satrecs` is index i
 * of every catalog array (and the GPU index).
 */
export function buildCatalog(records: ElementRecord[]): BuiltCatalog {
  const accepted: SatRec[] = [];
  const acceptedRecords: ElementRecord[] = [];

  for (let i = 0; i < records.length; i++) {
    const rec = records[i]!;
    let satrec: SatRec;
    try {
      satrec = recordToSatrec(rec);
    } catch {
      continue;
    }
    if (satrec.error !== 0) continue;
    if (!(satrec.no > 0) || !Number.isFinite(satrec.jdsatepoch)) continue;
    // Must propagate cleanly at its own epoch.
    const pv = sgp4(satrec, 0);
    if (
      pv === null ||
      satrec.error !== 0 ||
      !Number.isFinite(pv.position.x) ||
      !Number.isFinite(pv.position.y) ||
      !Number.isFinite(pv.position.z)
    ) {
      continue;
    }
    accepted.push(satrec);
    acceptedRecords.push(rec);
  }

  const count = accepted.length;
  const names: string[] = new Array<string>(count);
  const intlDesignators: string[] = new Array<string>(count);
  const noradIds = new Int32Array(count);
  const epochMs = new Float64Array(count);
  const inclinationDeg = new Float32Array(count);
  const meanAltitudeKm = new Float32Array(count);
  const periodMin = new Float32Array(count);
  const shell = new Uint8Array(count);
  const bus = new Uint8Array(count);
  const lasers = new Uint8Array(count);

  for (let i = 0; i < count; i++) {
    const satrec = accepted[i]!;
    const rec = acceptedRecords[i]!;
    names[i] = rec.name;
    intlDesignators[i] = rec.intlDesignator;
    noradIds[i] = rec.noradId;
    epochMs[i] = satrecEpochMs(satrec);

    const incDeg = satrec.inclo * RAD2DEG;
    // satrec.no: (Brouwer, un-Kozai'd) mean motion in rad/min -> rad/s.
    const nRadPerSec = satrec.no / 60;
    const aKm = Math.cbrt(MU_KM3_S2 / (nRadPerSec * nRadPerSec));
    const altKm = aKm - EARTH_RADIUS_KM;

    inclinationDeg[i] = incDeg;
    meanAltitudeKm[i] = altKm;
    periodMin[i] = TWO_PI / satrec.no;
    const [busIndex, laserTerminals] = estimateHardware(rec.intlDesignator, incDeg, rec.name);
    bus[i] = busIndex;
    lasers[i] = laserTerminals;
    shell[i] = classifyShell(incDeg, altKm, busIndex);
  }

  return {
    satrecs: accepted,
    catalog: {
      count,
      names,
      noradIds,
      intlDesignators,
      epochMs,
      inclinationDeg,
      meanAltitudeKm,
      periodMin,
      shell,
      bus,
      lasers,
    },
    rejected: records.length - count,
  };
}

/**
 * Propagate every satellite to `timeMs` and write Earth-fixed (ECEF) X,Y,Z in
 * metres into `out` (`out.length >= satrecs.length * 3`). A satellite whose
 * propagation fails (SGP4 error or decayed) gets a NaN triple.
 * Returns the number of failures.
 *
 * `outNormals` (optional, same length rule): unit orbit normals h = r x v of
 * the INERTIAL (TEME) state, rotated into the same Earth-fixed axes as `out`
 * so they can be combined with the ECEF positions (NaN triple on failure).
 * The TEME velocity matters: an ECEF velocity includes Earth rotation and
 * tilts the apparent orbit plane by ~3.5 degrees.
 */
export function propagateInto(
  satrecs: readonly SatRec[],
  timeMs: number,
  out: Float64Array,
  outNormals?: Float64Array,
): number {
  const n = satrecs.length;
  if (out.length < n * 3) {
    throw new RangeError(`propagateInto: out has ${out.length} slots, need ${n * 3}`);
  }
  if (outNormals !== undefined && outNormals.length < n * 3) {
    throw new RangeError(`propagateInto: outNormals has ${outNormals.length} slots, need ${n * 3}`);
  }
  const jd = julianDateFromMs(timeMs);
  const gmst = gstime(jd);
  const c = Math.cos(gmst);
  const s = Math.sin(gmst);
  let failures = 0;

  for (let i = 0, o = 0; i < n; i++, o += 3) {
    const satrec = satrecs[i]!;
    const pv = sgp4(satrec, (jd - satrec.jdsatepoch) * MINUTES_PER_DAY);
    // tempa <= 0 is satellite.js' community decay check (checkForDecay):
    // long-decayed elements otherwise yield meaningless positions.
    if (pv === null || satrec.tempa <= 0) {
      out[o] = NaN;
      out[o + 1] = NaN;
      out[o + 2] = NaN;
      if (outNormals !== undefined) {
        outNormals[o] = NaN;
        outNormals[o + 1] = NaN;
        outNormals[o + 2] = NaN;
      }
      failures++;
      continue;
    }
    const p = pv.position;
    // ECI (TEME) -> ECEF, identical to satellite.js eciToEcf; km -> m.
    out[o] = (p.x * c + p.y * s) * 1000;
    out[o + 1] = (p.x * -s + p.y * c) * 1000;
    out[o + 2] = p.z * 1000;
    if (outNormals !== undefined) {
      const v = pv.velocity;
      const hx = p.y * v.z - p.z * v.y;
      const hy = p.z * v.x - p.x * v.z;
      const hz = p.x * v.y - p.y * v.x;
      const inv = 1 / Math.sqrt(hx * hx + hy * hy + hz * hz);
      outNormals[o] = (hx * c + hy * s) * inv;
      outNormals[o + 1] = (hx * -s + hy * c) * inv;
      outNormals[o + 2] = hz * inv;
    }
  }
  return failures;
}
