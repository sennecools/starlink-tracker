import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { eciToGeodetic, gstime, sgp4 } from 'satellite.js';
import { parseOmmCsv } from '../server/omm.ts';
import { parseTle } from '../server/tle.ts';
import {
  EARTH_RADIUS_KM,
  createSatLive,
  footprintRadiusKm,
  groundTrackEcef,
  inCylindricalShadow,
  makeSatrec,
  orbitEci,
  orbitPeriodMs,
  parseCospar,
  satLive,
  satStatic,
} from '../src/selection/satDetails.ts';
import { SHELLS, SHELL_V3 } from '../src/shared/shells.ts';
import { buildCatalog, julianDateFromMs } from '../src/worker/propagate.ts';

const FIXTURE = fileURLToPath(new URL('../fixtures/sup-gp-starlink-partial.tle', import.meta.url));
const { records } = parseTle(readFileSync(FIXTURE, 'utf8'));
const { catalog } = buildCatalog(records);

/** Index of the first catalog satellite in shell 0 (53 deg) at an operational altitude. */
const IDX = (() => {
  for (let i = 0; i < catalog.count; i++) if (catalog.shell[i] === 0) return i;
  return 0;
})();
const RECORD = records.find((r) => r.noradId === catalog.noradIds[IDX])!;
const SATREC = makeSatrec(RECORD);
const T0 = catalog.epochMs[IDX]! + 3_600_000; // one hour after epoch

function norm3(a: Float64Array, i: number): number {
  return Math.hypot(a[i * 3]!, a[i * 3 + 1]!, a[i * 3 + 2]!);
}

test('COSPAR formatting and launch year rule', () => {
  assert.deepEqual(parseCospar('19074B'), { formatted: '2019-074B', year: 2019, launchNumber: 74 });
  assert.deepEqual(parseCospar('98067A  '), { formatted: '1998-067A', year: 1998, launchNumber: 67 });
  assert.deepEqual(parseCospar('57001A'), { formatted: '1957-001A', year: 1957, launchNumber: 1 });
  assert.deepEqual(parseCospar('56123ABC'), { formatted: '2056-123ABC', year: 2056, launchNumber: 123 });
  const bad = parseCospar(' XYZ ');
  assert.equal(bad.formatted, 'XYZ');
  assert.ok(Number.isNaN(bad.year));
});

test('makeSatrec rejects garbage', () => {
  assert.throws(() => makeSatrec({ ...RECORD, line1: 'garbage', line2: 'more garbage' }));
  const l2 = RECORD.line2!;
  assert.throws(() => makeSatrec({ ...RECORD, line2: l2.slice(0, 52) + ' 0.00000000' + l2.slice(63) }));
  // Neither OMM nor TLE.
  assert.throws(() => makeSatrec({ name: 'EMPTY', noradId: 1, intlDesignator: '' }));
});

test('satStatic: STARLINK-1008 fixture facts', () => {
  const idx = catalog.names.indexOf('STARLINK-1008');
  assert.ok(idx >= 0);
  const rec = records.find((r) => r.name === 'STARLINK-1008')!;
  const s = satStatic(idx, rec, catalog, makeSatrec(rec));
  assert.equal(s.index, idx);
  assert.equal(s.noradId, 44714);
  assert.equal(s.cosparId, '2019-074B');
  assert.equal(s.launchYear, 2019);
  assert.equal(s.launchNumber, 74);
  assert.ok(Math.abs(s.inclinationDeg - 53.144) < 1e-6);
  assert.ok(Math.abs(s.eccentricity - 0.0002857) < 1e-9);
  assert.equal(s.shellIndex, catalog.shell[idx]);
  assert.equal(s.shellLabel, SHELLS[catalog.shell[idx]!]!.label);
  assert.ok(Math.abs(s.periodMin - catalog.periodMin[idx]!) < 1e-3);
  // Apogee/perigee bracket the mean altitude and differ by 2ae.
  const meanAlt = catalog.meanAltitudeKm[idx]!;
  assert.ok(s.apogeeKm >= meanAlt - 1e-3 && s.perigeeKm <= meanAlt + 1e-3, `${s.perigeeKm} ${meanAlt} ${s.apogeeKm}`);
  const a = meanAlt + EARTH_RADIUS_KM;
  assert.ok(Math.abs(s.apogeeKm - s.perigeeKm - 2 * a * s.eccentricity) < 0.01);
  assert.ok(Math.abs(s.epochMs - catalog.epochMs[idx]!) <= 1);
});

test('satStatic: apogee/perigee sane for the whole fixture', () => {
  for (let i = 0; i < catalog.count; i += 7) {
    const rec = records.find((r) => r.noradId === catalog.noradIds[i])!;
    const s = satStatic(i, rec, catalog, makeSatrec(rec));
    assert.ok(s.perigeeKm > 100 && s.perigeeKm <= s.apogeeKm && s.apogeeKm < 2000, `${rec.name}: ${s.perigeeKm}..${s.apogeeKm}`);
    assert.match(s.cosparId, /^(19|20)\d{2}-\d{3}[A-Z]{1,3}$/);
  }
});

test('satLive matches satellite.js eciToGeodetic', () => {
  const out = createSatLive();
  for (let k = 0; k < 10; k++) {
    const t = T0 + k * 517_000;
    const ret = satLive(SATREC, t, out);
    assert.equal(ret, out, 'returns the out object');
    const jd = julianDateFromMs(t);
    const pv = sgp4(SATREC, (jd - SATREC.jdsatepoch) * 1440)!;
    const geo = eciToGeodetic(pv.position, gstime(jd));
    let lon = (geo.longitude * 180) / Math.PI;
    if (lon > 180) lon -= 360;
    assert.equal(out.timeMs, t);
    assert.ok(Math.abs(out.latDeg - (geo.latitude * 180) / Math.PI) < 1e-9);
    assert.ok(Math.abs(out.lonDeg - lon) < 1e-9);
    assert.ok(out.lonDeg >= -180 && out.lonDeg <= 180);
    assert.ok(Math.abs(out.altitudeKm - geo.height) < 1e-9);
    assert.ok(Math.abs(out.footprintRadiusKm - footprintRadiusKm(geo.height)) < 1e-9);
  }
});

test('satLive speeds are orbital', () => {
  const out = satLive(SATREC, T0, createSatLive());
  assert.ok(out.altitudeKm > 300 && out.altitudeKm < 700, `alt ${out.altitudeKm}`);
  assert.ok(out.speedKmS > 7.5 && out.speedKmS < 7.8, `speed ${out.speedKmS}`);
  // Vis-viva-ish check against the current radius.
  const r = out.altitudeKm + EARTH_RADIUS_KM;
  assert.ok(Math.abs(out.speedKmS - Math.sqrt(398600.4418 / r)) < 0.05);
  // Ground speed: ~v * R/(R+h) minus/plus Earth rotation, i.e. roughly 24,500-27,500 km/h.
  assert.ok(out.groundSpeedKmH > 24_000 && out.groundSpeedKmH < 28_000, `ground ${out.groundSpeedKmH}`);
});

test('satLive ground speed agrees with successive sub-points', () => {
  const a = satLive(SATREC, T0, createSatLive());
  const b = satLive(SATREC, T0 + 1000, createSatLive());
  const toRad = Math.PI / 180;
  const dLat = (b.latDeg - a.latDeg) * toRad;
  const dLon = (b.lonDeg - a.lonDeg) * toRad;
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(a.latDeg * toRad) * Math.cos(b.latDeg * toRad) * Math.sin(dLon / 2) ** 2;
  const distKm = 2 * 6371 * Math.asin(Math.sqrt(h));
  const kmh = distKm * 3600;
  assert.ok(Math.abs(kmh - a.groundSpeedKmH) / kmh < 0.02, `${kmh} vs ${a.groundSpeedKmH}`);
});

test('footprint radius at 550 km and 25 deg is about 940 km', () => {
  const r = footprintRadiusKm(550);
  assert.ok(Math.abs(r - 940) < 10, `footprint ${r}`);
  assert.ok(footprintRadiusKm(340) < r && footprintRadiusKm(1200) > r);
  assert.equal(footprintRadiusKm(0), 0);
  assert.ok(Math.abs(footprintRadiusKm(550, 0) - EARTH_RADIUS_KM * Math.acos(EARTH_RADIUS_KM / (EARTH_RADIUS_KM + 550))) < 1e-9);
});

test('orbitEci: closed ring at orbital radius, starts at the satellite', () => {
  const samples = 360;
  const ring = orbitEci(SATREC, T0, samples);
  assert.equal(ring.length, (samples + 1) * 3);
  for (let k = 0; k <= samples; k++) {
    const rKm = norm3(ring, k) / 1000;
    assert.ok(rKm > EARTH_RADIUS_KM + 250 && rKm < EARTH_RADIUS_KM + 700, `r ${rKm}`);
  }
  const gapKm = Math.hypot(ring[0]! - ring[samples * 3]!, ring[1]! - ring[samples * 3 + 1]!, ring[2]! - ring[samples * 3 + 2]!) / 1000;
  assert.ok(gapKm < 0.001, `ring gap ${gapKm} km`);
  // First sample is the TEME position at T0 (metres).
  const jd = julianDateFromMs(T0);
  const p = sgp4(SATREC, (jd - SATREC.jdsatepoch) * 1440)!.position;
  assert.ok(Math.hypot(ring[0]! / 1000 - p.x, ring[1]! / 1000 - p.y, ring[2]! / 1000 - p.z) < 1e-6);
  // Reuses a big-enough out buffer.
  const buf = new Float64Array((samples + 1) * 3);
  assert.equal(orbitEci(SATREC, T0, samples, buf), buf);
  // Plain TEME samples: the raw period gap is a few tens of km (nodal drift),
  // and closing the ring never moves a sample further than that.
  const teme = orbitEci(SATREC, T0, samples, undefined, false);
  const gapTeme = Math.hypot(teme[0]! - teme[samples * 3]!, teme[1]! - teme[samples * 3 + 1]!, teme[2]! - teme[samples * 3 + 2]!) / 1000;
  assert.ok(gapTeme > 1 && gapTeme < 80, `TEME gap ${gapTeme} km`);
  for (let k = 0; k <= samples; k++) {
    const d = Math.hypot(ring[k * 3]! - teme[k * 3]!, ring[k * 3 + 1]! - teme[k * 3 + 1]!, ring[k * 3 + 2]! - teme[k * 3 + 2]!) / 1000;
    assert.ok(d <= gapTeme + 1, `sample ${k} moved ${d} km`);
  }
  assert.ok(Math.hypot(teme[0]! - ring[0]!, teme[1]! - ring[1]!, teme[2]! - ring[2]!) === 0);
  assert.ok(orbitPeriodMs(SATREC) > 85 * 60_000 && orbitPeriodMs(SATREC) < 100 * 60_000);
});

test('orbitEci at the ring resolution (2048): straight chords stay within ~10 m of the orbit', () => {
  const samples = 2048; // SelectionController ORBIT_SAMPLES
  const teme = orbitEci(SATREC, T0, samples, undefined, false);
  const stepMin = orbitPeriodMs(SATREC) / 60_000 / samples;
  const tsince0 = (julianDateFromMs(T0) - SATREC.jdsatepoch) * 1440;
  let worst = 0;
  for (let k = 0; k < samples; k += 7) {
    const mid = sgp4(SATREC, tsince0 + (k + 0.5) * stepMin)!.position;
    const mx = (teme[k * 3]! + teme[k * 3 + 3]!) / 2 / 1000;
    const my = (teme[k * 3 + 1]! + teme[k * 3 + 4]!) / 2 / 1000;
    const mz = (teme[k * 3 + 2]! + teme[k * 3 + 5]!) / 2 / 1000;
    worst = Math.max(worst, Math.hypot(mx - mid.x, my - mid.y, mz - mid.z) * 1000);
  }
  assert.ok(worst < 12, `chord midpoint ${worst} m off the orbit`);
});

test('groundTrackEcef: points on the ellipsoid under the satellite', () => {
  const track = groundTrackEcef(SATREC, T0, 95, 190);
  assert.equal(track.length, 191 * 3);
  const live = satLive(SATREC, T0, createSatLive());
  for (let k = 0; k <= 190; k++) {
    const rKm = norm3(track, k) / 1000;
    assert.ok(rKm > 6356 && rKm < 6378.2, `r ${rKm}`);
  }
  // First point is the sub-satellite point at T0.
  const lat = Math.asin(track[2]! / norm3(track, 0)); // geocentric, close enough for a 0.3 deg check
  const lon = Math.atan2(track[1]!, track[0]!);
  assert.ok(Math.abs((lon * 180) / Math.PI - live.lonDeg) < 1e-6);
  assert.ok(Math.abs((lat * 180) / Math.PI - live.latDeg) < 0.3);
});

test('shadow flag flips over an orbit', () => {
  const out = createSatLive();
  let lit = 0;
  let dark = 0;
  let flips = 0;
  let prev: boolean | null = null;
  const period = orbitPeriodMs(SATREC);
  for (let k = 0; k < 200; k++) {
    satLive(SATREC, T0 + (k * period) / 200, out);
    if (out.inShadow) dark++;
    else lit++;
    if (prev !== null && prev !== out.inShadow) flips++;
    prev = out.inShadow;
  }
  assert.ok(lit > 0 && dark > 0, `lit ${lit} dark ${dark}`);
  assert.ok(flips >= 1 && flips <= 2, `flips ${flips}`);
  // Eclipse is less than half an orbit for LEO.
  assert.ok(dark < 100);
});

test('cylindrical shadow geometry', () => {
  const R = EARTH_RADIUS_KM;
  assert.equal(inCylindricalShadow(-(R + 500), 0, 0, 1, 0, 0), true);
  assert.equal(inCylindricalShadow(R + 500, 0, 0, 1, 0, 0), false);
  assert.equal(inCylindricalShadow(-(R + 500), R + 10, 0, 1, 0, 0), false);
  assert.equal(inCylindricalShadow(0, R + 500, 0, 1, 0, 0), false);
});

test('makeSatrec / satStatic for an OMM record (NORAD id > 99,999, Starship V3)', () => {
  const csv =
    'OBJECT_NAME,OBJECT_ID,EPOCH,MEAN_MOTION,ECCENTRICITY,INCLINATION,RA_OF_ASC_NODE,ARG_OF_PERICENTER,MEAN_ANOMALY,' +
    'EPHEMERIS_TYPE,CLASSIFICATION_TYPE,NORAD_CAT_ID,ELEMENT_SET_NO,REV_AT_EPOCH,BSTAR,MEAN_MOTION_DOT,MEAN_MOTION_DDOT\n' +
    'STARLINK-40096,2026-256A,2026-11-20T03:12:00.000000,15.94000000,.00080000,30.5000,120.0000,90.0000,270.0000,0,U,101415,999,105,.50000000E-3,.10000000E-3,0\n';
  const [rec] = parseOmmCsv(csv).records;
  assert.ok(rec);
  const satrec = makeSatrec(rec);
  const cat = buildCatalog([rec]).catalog;
  const s = satStatic(0, rec, cat, satrec);
  assert.equal(s.noradId, 101415);
  assert.equal(s.cosparId, '2026-256A');
  assert.equal(s.launchYear, 2026);
  assert.equal(s.shellLabel, SHELLS[SHELL_V3]!.label);
  assert.equal(s.epochMs, Date.UTC(2026, 10, 20, 3, 12));
  assert.ok(Math.abs(s.inclinationDeg - 30.5) < 1e-6);
  assert.ok(s.perigeeKm > 250 && s.apogeeKm < 340, `${s.perigeeKm}..${s.apogeeKm} km`);
  // An OMM record with an unusable element set is rejected like a bad TLE.
  assert.throws(() => makeSatrec({ ...rec, omm: { ...rec.omm!, MEAN_MOTION: 0 } }));
});
