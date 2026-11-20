/**
 * Details of ONE selected satellite, computed on the main thread from its
 * satrec with satellite.js (pure: no DOM, no Cesium, runs in Node tests).
 *
 *   satStatic       facts that only change with the element set (panel header / static rows)
 *   satLive         live state at one instant (refreshed ~4x per second, reuses `out`)
 *   orbitEci        one closed orbit ring (TEME-aligned, metres) for the orbit path
 *   groundTrackEcef the sub-satellite track on the ground (ECEF, metres)
 *
 * Frames: satellite.js returns TEME positions; Earth-fixed coordinates use the
 * same GMST rotation (`gstime`) as the propagation worker, so everything lines
 * up with the GPU satellites to well under a metre.
 */
import { type SatRec, eciToGeodetic, gstime, sgp4, sunPos } from 'satellite.js';
import type { ElementRecord, SatCatalog } from '../shared/contracts.ts';
import { MIN_ELEVATION_DEG } from '../shared/phase2.ts';
import type { SatLive, SatStatic } from '../shared/phase2.ts';
import { SHELLS, SHELL_OTHER } from '../shared/shells.ts';
import { julianDateFromMs, recordToSatrec, satrecEpochMs } from '../worker/propagate.ts';

/** WGS-84 equatorial radius, km. */
export const EARTH_RADIUS_KM = 6378.137;
/** Earth gravitational parameter, km^3/s^2. */
const MU_KM3_S2 = 398600.4418;
/** Earth rotation rate, rad/s (IERS). */
const OMEGA_EARTH_RAD_S = 7.2921150e-5;
/** WGS-84 ellipsoid (same numbers satellite.js uses in eciToGeodetic). */
const WGS84_A_KM = 6378.137;
const WGS84_B_KM = 6356.7523142;
const WGS84_E2 = 1 - (WGS84_B_KM * WGS84_B_KM) / (WGS84_A_KM * WGS84_A_KM);

const TWO_PI = 2 * Math.PI;
const RAD2DEG = 180 / Math.PI;
const DEG2RAD = Math.PI / 180;
const MINUTES_PER_DAY = 1440;
const MS_PER_MIN = 60_000;
const COS_MIN_ELEV = Math.cos(MIN_ELEVATION_DEG * DEG2RAD);
const MIN_ELEV_RAD = MIN_ELEVATION_DEG * DEG2RAD;

// ---------------------------------------------------------------------------
// Element-set level
// ---------------------------------------------------------------------------

/** Parses and initialises the SGP4 record (OMM or TLE). Throws when the element set is invalid. */
export function makeSatrec(record: ElementRecord): SatRec {
  const satrec = recordToSatrec(record);
  if (satrec.error !== 0 || !(satrec.no > 0) || !Number.isFinite(satrec.jdsatepoch)) {
    throw new Error(`Invalid element set for ${record.name} (${record.noradId}), SGP4 error ${satrec.error}`);
  }
  return satrec;
}

export interface CosparParts {
  /** "2019-074B", or the trimmed raw designator when it cannot be parsed. */
  formatted: string;
  /** 4-digit year, NaN when unknown. */
  year: number;
  /** Launch number of that year, NaN when unknown. */
  launchNumber: number;
}

/**
 * TLE international designator ("19074B", "98067A  ") -> COSPAR "2019-074B".
 * Two-digit years 57-99 are 19xx (Sputnik was 1957), 00-56 are 20xx.
 */
export function parseCospar(intlDesignator: string): CosparParts {
  const raw = intlDesignator.trim();
  const m = /^(\d{2})(\d{3})([A-Z]{0,3})$/.exec(raw.toUpperCase());
  if (!m) return { formatted: raw, year: Number.NaN, launchNumber: Number.NaN };
  const yy = Number(m[1]);
  const year = yy >= 57 ? 1900 + yy : 2000 + yy;
  return { formatted: `${year}-${m[2]}${m[3]}`, year, launchNumber: Number(m[2]) };
}

/** Semi-major axis (km) from the satrec's (Brouwer) mean motion, as the catalog does. */
export function semiMajorAxisKm(satrec: SatRec): number {
  const n = satrec.no / 60; // rad/min -> rad/s
  return Math.cbrt(MU_KM3_S2 / (n * n));
}

/**
 * Draconic (node-to-node) period in ms from SGP4's secular rates: the time the
 * satellite needs to return to the same argument of latitude. A ring sampled
 * over this period closes on itself much better than with 2*pi/no.
 */
export function orbitPeriodMs(satrec: SatRec): number {
  const uDot = satrec.mdot + satrec.argpdot; // rad/min
  const rate = uDot > 0 && Number.isFinite(uDot) ? uDot : satrec.no;
  return (TWO_PI / rate) * MS_PER_MIN;
}

/** Secular regression rate of the ascending node, rad per ms (negative for prograde LEO). */
export function nodeRateRadPerMs(satrec: SatRec): number {
  const r = satrec.nodedot;
  return Number.isFinite(r) ? r / MS_PER_MIN : 0;
}

export function satStatic(index: number, record: ElementRecord, catalog: SatCatalog, satrec: SatRec): SatStatic {
  const cospar = parseCospar(record.intlDesignator || (catalog.intlDesignators[index] ?? ''));
  const shellIndex = index >= 0 && index < catalog.count ? catalog.shell[index]! : SHELL_OTHER;
  const shell = SHELLS[shellIndex] ?? SHELLS[SHELL_OTHER]!;
  const e = satrec.ecco;
  const a = semiMajorAxisKm(satrec);
  return {
    index,
    name: record.name,
    noradId: record.noradId,
    cosparId: cospar.formatted,
    launchYear: cospar.year,
    launchNumber: cospar.launchNumber,
    shellIndex,
    shellLabel: shell.label,
    inclinationDeg: satrec.inclo * RAD2DEG,
    eccentricity: e,
    periodMin: TWO_PI / satrec.no,
    apogeeKm: a * (1 + e) - EARTH_RADIUS_KM,
    perigeeKm: a * (1 - e) - EARTH_RADIUS_KM,
    epochMs: satrecEpochMs(satrec),
  };
}

// ---------------------------------------------------------------------------
// Live state
// ---------------------------------------------------------------------------

/** A SatLive with every number NaN (use as the reusable `out` object). */
export function createSatLive(): SatLive {
  return {
    timeMs: Number.NaN,
    latDeg: Number.NaN,
    lonDeg: Number.NaN,
    altitudeKm: Number.NaN,
    speedKmS: Number.NaN,
    groundSpeedKmH: Number.NaN,
    inShadow: false,
    footprintRadiusKm: Number.NaN,
  };
}

/**
 * Radius (km, along the ground) of the area that sees the satellite at least
 * `minElevationDeg` above the horizon: lambda = acos(R cos e / (R + h)) - e.
 */
export function footprintRadiusKm(altitudeKm: number, minElevationDeg = MIN_ELEVATION_DEG): number {
  if (!(altitudeKm > 0)) return 0;
  const R = EARTH_RADIUS_KM;
  const eRad = minElevationDeg === MIN_ELEVATION_DEG ? MIN_ELEV_RAD : minElevationDeg * DEG2RAD;
  const cosE = minElevationDeg === MIN_ELEVATION_DEG ? COS_MIN_ELEV : Math.cos(eRad);
  const lambda = Math.acos((R * cosE) / (R + altitudeKm)) - eRad;
  return lambda > 0 ? R * lambda : 0;
}

/**
 * Cylindrical Earth-shadow test: the satellite (ECI km) is in shadow when it
 * is on the night side and within one Earth radius of the Sun-Earth axis.
 * `sun` is any vector pointing to the Sun (only its direction is used).
 */
export function inCylindricalShadow(rx: number, ry: number, rz: number, sx: number, sy: number, sz: number): boolean {
  const sLen = Math.sqrt(sx * sx + sy * sy + sz * sz);
  if (!(sLen > 0)) return false;
  const ux = sx / sLen;
  const uy = sy / sLen;
  const uz = sz / sLen;
  const along = rx * ux + ry * uy + rz * uz;
  if (along >= 0) return false; // Sun side
  const px = rx - along * ux;
  const py = ry - along * uy;
  const pz = rz - along * uz;
  return px * px + py * py + pz * pz < EARTH_RADIUS_KM * EARTH_RADIUS_KM;
}

function fillInvalid(out: SatLive, timeMs: number): SatLive {
  out.timeMs = timeMs;
  out.latDeg = Number.NaN;
  out.lonDeg = Number.NaN;
  out.altitudeKm = Number.NaN;
  out.speedKmS = Number.NaN;
  out.groundSpeedKmH = Number.NaN;
  out.inShadow = false;
  out.footprintRadiusKm = Number.NaN;
  return out;
}

/** Live state at `timeMs` written into `out` (fields NaN when propagation fails). */
export function satLive(satrec: SatRec, timeMs: number, out: SatLive): SatLive {
  const jd = julianDateFromMs(timeMs);
  const pv = sgp4(satrec, (jd - satrec.jdsatepoch) * MINUTES_PER_DAY);
  if (pv === null || satrec.error !== 0) return fillInvalid(out, timeMs);
  const r = pv.position;
  const v = pv.velocity;
  if (!Number.isFinite(r.x) || !Number.isFinite(v.x)) return fillInvalid(out, timeMs);

  const gmst = gstime(jd);
  const geo = eciToGeodetic(r, gmst);
  let lonDeg = geo.longitude * RAD2DEG;
  if (lonDeg > 180) lonDeg -= 360;
  else if (lonDeg < -180) lonDeg += 360;

  const speed = Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z);

  // Ground speed: Earth-relative velocity (v - w x r), horizontal part,
  // scaled from the satellite's radius down to the sub-satellite point.
  const rLen = Math.sqrt(r.x * r.x + r.y * r.y + r.z * r.z);
  const vrx = v.x + OMEGA_EARTH_RAD_S * r.y;
  const vry = v.y - OMEGA_EARTH_RAD_S * r.x;
  const vrz = v.z;
  const radial = (vrx * r.x + vry * r.y + vrz * r.z) / rLen;
  const hx = vrx - (radial * r.x) / rLen;
  const hy = vry - (radial * r.y) / rLen;
  const hz = vrz - (radial * r.z) / rLen;
  const groundRadius = Math.max(1, rLen - geo.height);
  const groundSpeedKmS = Math.sqrt(hx * hx + hy * hy + hz * hz) * (groundRadius / rLen);

  const sun = sunPos(jd).rsun;

  out.timeMs = timeMs;
  out.latDeg = geo.latitude * RAD2DEG;
  out.lonDeg = lonDeg;
  out.altitudeKm = geo.height;
  out.speedKmS = speed;
  out.groundSpeedKmH = groundSpeedKmS * 3600;
  out.inShadow = inCylindricalShadow(r.x, r.y, r.z, sun.x, sun.y, sun.z);
  out.footprintRadiusKm = footprintRadiusKm(geo.height);
  return out;
}

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

/**
 * One full (draconic) period of positions in METRES, starting at `timeMs`:
 * `samples + 1` points (x,y,z interleaved) so the ring closes.
 *
 * With `closed` (default true) the ring is made to close exactly:
 *  1. sample k is rotated about Z by -nodedot * dt_k, i.e. expressed in a frame
 *     that coincides with TEME at `timeMs` and turns with the regressing node
 *     (removes the ~0.3 deg/orbit nodal drift, typically ~26 km);
 *  2. the remaining residual (drag shortening the period, short-period terms;
 *     median ~3 km, up to ~165 km for a fast-decaying satellite) is
 *     distributed linearly along the ring, so sample 0 is untouched (the
 *     satellite at `timeMs`) and sample n lands exactly on sample 0.
 * To place the ring at time t, rotate it about Z by nodedot * (t - timeMs) on
 * top of TEME->Earth-fixed (OrbitPath does this when given the node rate); for
 * t within a minute of `timeMs` the difference to plain TEME is under 1 km.
 * With `closed = false` the samples are plain TEME.
 * Samples where propagation fails are NaN.
 */
export function orbitEci(satrec: SatRec, timeMs: number, samples: number, out?: Float64Array, closed = true): Float64Array {
  const n = Math.max(3, Math.floor(samples));
  const len = (n + 1) * 3;
  const buf = out && out.length >= len ? out : new Float64Array(len);
  const periodMs = orbitPeriodMs(satrec);
  const nodeRate = closed ? satrec.nodedot : 0; // rad/min
  const jd0 = julianDateFromMs(timeMs);
  const tsince0 = (jd0 - satrec.jdsatepoch) * MINUTES_PER_DAY;
  const stepMin = periodMs / MS_PER_MIN / n;
  for (let k = 0; k <= n; k++) {
    const dtMin = k * stepMin;
    const pv = sgp4(satrec, tsince0 + dtMin);
    const k3 = k * 3;
    if (pv === null || satrec.error !== 0 || !Number.isFinite(pv.position.x)) {
      buf[k3] = buf[k3 + 1] = buf[k3 + 2] = Number.NaN;
      continue;
    }
    const p = pv.position;
    let x = p.x;
    let y = p.y;
    if (nodeRate !== 0 && Number.isFinite(nodeRate)) {
      const ang = -nodeRate * dtMin;
      const c = Math.cos(ang);
      const s = Math.sin(ang);
      const xr = x * c - y * s;
      y = x * s + y * c;
      x = xr;
    }
    buf[k3] = x * 1000;
    buf[k3 + 1] = y * 1000;
    buf[k3 + 2] = p.z * 1000;
  }
  if (closed) {
    const n3 = n * 3;
    const ex = buf[n3]! - buf[0]!;
    const ey = buf[n3 + 1]! - buf[1]!;
    const ez = buf[n3 + 2]! - buf[2]!;
    if (Number.isFinite(ex) && Number.isFinite(ey) && Number.isFinite(ez)) {
      for (let k = 1; k <= n; k++) {
        const f = k / n;
        const k3 = k * 3;
        buf[k3] = buf[k3]! - ex * f;
        buf[k3 + 1] = buf[k3 + 1]! - ey * f;
        buf[k3 + 2] = buf[k3 + 2]! - ez * f;
      }
    }
  }
  return buf;
}

/**
 * Sub-satellite point (on the WGS-84 ellipsoid, height 0) over the next
 * `durationMin` minutes, ECEF metres: `samples + 1` points from `timeMs` to
 * `timeMs + durationMin`. Samples where propagation fails are NaN.
 */
export function groundTrackEcef(
  satrec: SatRec,
  timeMs: number,
  durationMin: number,
  samples: number,
  out?: Float64Array,
): Float64Array {
  const n = Math.max(1, Math.floor(samples));
  const len = (n + 1) * 3;
  const buf = out && out.length >= len ? out : new Float64Array(len);
  const stepMs = (durationMin * MS_PER_MIN) / n;
  for (let k = 0; k <= n; k++) {
    const t = timeMs + k * stepMs;
    const jd = julianDateFromMs(t);
    const pv = sgp4(satrec, (jd - satrec.jdsatepoch) * MINUTES_PER_DAY);
    const k3 = k * 3;
    if (pv === null || satrec.error !== 0 || !Number.isFinite(pv.position.x)) {
      buf[k3] = buf[k3 + 1] = buf[k3 + 2] = Number.NaN;
      continue;
    }
    const geo = eciToGeodetic(pv.position, gstime(jd));
    const sinLat = Math.sin(geo.latitude);
    const cosLat = Math.cos(geo.latitude);
    const N = WGS84_A_KM / Math.sqrt(1 - WGS84_E2 * sinLat * sinLat);
    buf[k3] = N * cosLat * Math.cos(geo.longitude) * 1000;
    buf[k3 + 1] = N * cosLat * Math.sin(geo.longitude) * 1000;
    buf[k3 + 2] = N * (1 - WGS84_E2) * sinLat * 1000;
  }
  return buf;
}
