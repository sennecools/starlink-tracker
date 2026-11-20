import { strict as assert } from 'node:assert';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { eciToEcf, gstime, propagate, twoline2satrec } from 'satellite.js';
import type { ElementRecord } from '../src/shared/contracts.ts';
import { parseOmmCsv } from '../server/omm.ts';
import { BUS_V10, BUS_V15, BUS_V2MINI, BUS_V3, BUS_VERSIONS, estimateHardware } from '../src/shared/hardware.ts';
import { SHELLS, SHELL_TRANSIT, SHELL_V3 } from '../src/shared/shells.ts';
import { buildCatalog, propagateInto, recordToSatrec } from '../src/worker/propagate.ts';

const FIXTURE = new URL('../fixtures/sup-gp-starlink-partial.tle', import.meta.url);

/** A TLE-sourced record (line1/line2 present). */
type TleRecord = ElementRecord & { line1: string; line2: string };

/** Minimal 3-line TLE splitter (name / line 1 / line 2). */
function parseTle(text: string): TleRecord[] {
  const lines = text.split(/\r?\n/).map((l) => l.trimEnd()).filter((l) => l.length > 0);
  const out: TleRecord[] = [];
  for (let i = 0; i + 2 < lines.length; i++) {
    const name = lines[i]!;
    const l1 = lines[i + 1];
    const l2 = lines[i + 2];
    if (l1?.startsWith('1 ') && l2?.startsWith('2 ') && !name.startsWith('1 ') && !name.startsWith('2 ')) {
      out.push({
        name: name.trim(),
        noradId: Number(l1.substring(2, 7)),
        intlDesignator: l1.substring(9, 17).trim(),
        line1: l1,
        line2: l2,
      });
      i += 2;
    }
  }
  return out;
}

/** TLE checksum (mod 10 of digits, '-' counts 1), so edited lines stay valid. */
function withChecksum(line68: string): string {
  let sum = 0;
  for (const ch of line68) {
    if (ch >= '0' && ch <= '9') sum += ch.charCodeAt(0) - 48;
    else if (ch === '-') sum += 1;
  }
  return line68 + String(sum % 10);
}

const records = parseTle(readFileSync(FIXTURE, 'utf8'));
const built = buildCatalog(records);
const { catalog, satrecs } = built;

test('fixture parses to 853 records and all are accepted', () => {
  assert.equal(records.length, 853);
  assert.equal(built.rejected, 0);
  assert.equal(catalog.count, 853);
  assert.equal(satrecs.length, 853);
  for (const arr of [
    catalog.names,
    catalog.intlDesignators,
    catalog.noradIds,
    catalog.epochMs,
    catalog.inclinationDeg,
    catalog.meanAltitudeKm,
    catalog.periodMin,
    catalog.shell,
    catalog.bus,
    catalog.lasers,
  ]) {
    assert.equal(arr.length, 853);
  }
  assert.equal(catalog.names[0], 'STARLINK-1008');
  assert.equal(catalog.noradIds[0], 44714);
  assert.equal(catalog.intlDesignators[0], '19074B');
});

test('epochs are decoded correctly', () => {
  // STARLINK-1008 epoch 26324.14631944 -> 2026 day 324 (Nov 20) + 0.14631944 d
  const expected = Date.UTC(2026, 0, 1) + (324 - 1 + 0.14631944) * 86_400_000;
  assert.ok(Math.abs(catalog.epochMs[0]! - expected) < 2, `epoch ${catalog.epochMs[0]} vs ${expected}`);
  for (let i = 0; i < catalog.count; i++) {
    const t = catalog.epochMs[i]!;
    assert.ok(t > Date.UTC(2026, 0, 1) && t < Date.UTC(2027, 0, 1), `epoch out of range at ${i}`);
  }
});

test('inclination, altitude, period and shells are physically plausible', () => {
  const s1008 = 0;
  assert.ok(Math.abs(catalog.inclinationDeg[s1008]! - 53.144) < 1e-3);
  // 15.7079 rev/day -> ~91.67 min, ~ 380 km (being lowered / in transit)
  assert.ok(Math.abs(catalog.periodMin[s1008]! - 1440 / 15.70790074) < 0.1);

  const shellCounts = new Array<number>(SHELLS.length).fill(0);
  for (let i = 0; i < catalog.count; i++) {
    const inc = catalog.inclinationDeg[i]!;
    const alt = catalog.meanAltitudeKm[i]!;
    const per = catalog.periodMin[i]!;
    const sh = catalog.shell[i]!;
    assert.ok(inc > 30 && inc < 100, `inclination ${inc} at ${i}`);
    assert.ok(alt > 150 && alt < 700, `altitude ${alt} at ${i}`);
    assert.ok(per > 86 && per < 99, `period ${per} at ${i}`);
    assert.ok(sh < SHELLS.length);
    shellCounts[sh]!++;
    // Kepler consistency: T = 2*pi*sqrt(a^3/mu)
    const a = alt + 6378.137;
    const tMin = (2 * Math.PI * Math.sqrt((a * a * a) / 398600.4418)) / 60;
    assert.ok(Math.abs(tMin - per) < 0.01, `period/altitude mismatch at ${i}`);
    if (sh === 0) {
      assert.ok(inc > 52.5 && inc < 53.6);
      assert.ok(alt > 420 && alt < 650, `53-deg shell alt ${alt}`);
      assert.ok(per > 92 && per < 98, `53-deg shell period ${per}`);
    }
    if (sh === SHELL_TRANSIT) assert.ok(alt < 420);
  }
  console.log('shell counts:', SHELLS.map((s, i) => `${s.id}=${shellCounts[i]}`).join(' '));
  assert.ok(shellCounts[0]! > 0, 'expected some 53-degree shell satellites');
});

test('propagateInto matches satellite.js propagate + eciToEcf to < 1 mm', () => {
  const out = new Float64Array(catalog.count * 3);
  for (const offsetMs of [0, 37_123, 3_600_000, -5_400_000, 86_400_000 * 2.5]) {
    const timeMs = catalog.epochMs[0]! + offsetMs;
    const failures = propagateInto(satrecs, timeMs, out);
    if (offsetMs <= 3_600_000) assert.equal(failures, 0);
    const date = new Date(timeMs);
    const gmst = gstime(date);
    let maxErr = 0;
    for (let i = 0; i < catalog.count; i++) {
      const ref = twoline2satrec(records[i]!.line1, records[i]!.line2);
      const pv = propagate(ref, date);
      if (!pv) {
        // Re-entering satellite: reference fails too, we must report NaN.
        assert.ok(Number.isNaN(out[i * 3]), `expected NaN at ${i}`);
        continue;
      }
      const ecf = eciToEcf(pv.position, gmst);
      const dx = out[i * 3]! - ecf.x * 1000;
      const dy = out[i * 3 + 1]! - ecf.y * 1000;
      const dz = out[i * 3 + 2]! - ecf.z * 1000;
      maxErr = Math.max(maxErr, Math.hypot(dx, dy, dz));
      const r = Math.hypot(out[i * 3]!, out[i * 3 + 1]!, out[i * 3 + 2]!);
      assert.ok(r > 6.3e6 && r < 7.1e6, `radius ${r} at ${i}`);
    }
    assert.ok(maxErr < 1e-3, `max error ${maxErr} m at offset ${offsetMs}`);
  }
});

test('operational satellites sit at ECEF radius 6.6e6..7.1e6 m', () => {
  const out = new Float64Array(catalog.count * 3);
  propagateInto(satrecs, Date.UTC(2026, 10, 20, 12), out);
  let checked = 0;
  for (let i = 0; i < catalog.count; i++) {
    if (catalog.meanAltitudeKm[i]! < 300) continue; // very low ones are de-orbiting
    const r = Math.hypot(out[i * 3]!, out[i * 3 + 1]!, out[i * 3 + 2]!);
    assert.ok(r > 6.6e6 && r < 7.1e6, `radius ${r} at ${i}`);
    checked++;
  }
  assert.ok(checked > 800);
});

test('propagateInto throws on an undersized buffer', () => {
  assert.throws(() => propagateInto(satrecs, Date.now(), new Float64Array(3)), RangeError);
  const out = new Float64Array(catalog.count * 3);
  assert.throws(() => propagateInto(satrecs, Date.now(), out, new Float64Array(3)), RangeError);
});

test('catalog bus / lasers follow estimateHardware for every record', () => {
  const counts = [0, 0, 0];
  for (let i = 0; i < catalog.count; i++) {
    const [bus, lasers] = estimateHardware(catalog.intlDesignators[i]!, catalog.inclinationDeg[i]!, catalog.names[i]!);
    assert.equal(catalog.bus[i], bus, `bus at ${i}`);
    assert.equal(catalog.lasers[i], lasers, `lasers at ${i}`);
    assert.ok(catalog.bus[i]! < BUS_VERSIONS.length);
    counts[catalog.bus[i]!]!++;
  }
  // STARLINK-1008 (2019-074B) is a v1.0 satellite without laser terminals.
  assert.equal(catalog.bus[0], BUS_V10);
  assert.equal(catalog.lasers[0], 0);
  // The fixture spans 2019..2022 launches: both v1.0 and v1.5 must appear.
  assert.ok(counts[BUS_V10]! > 0 && counts[BUS_V15]! > 0, `bus counts ${counts.join('/')}`);
  for (let i = 0; i < catalog.count; i++) {
    if (catalog.bus[i] === BUS_V15) assert.equal(catalog.lasers[i], 4);
  }
});

test('propagateInto outNormals: unit, inertial orbit normals in Earth-fixed axes', () => {
  const timeMs = Date.UTC(2026, 10, 20, 12);
  const pos = new Float64Array(catalog.count * 3);
  const nrm = new Float64Array(catalog.count * 3);
  const posOnly = new Float64Array(catalog.count * 3);
  propagateInto(satrecs, timeMs, pos, nrm);
  propagateInto(satrecs, timeMs, posOnly);
  assert.deepEqual(pos, posOnly, 'normals output must not change positions');
  for (let i = 0; i < catalog.count; i++) {
    const hx = nrm[i * 3]!;
    const hy = nrm[i * 3 + 1]!;
    const hz = nrm[i * 3 + 2]!;
    assert.ok(Math.abs(Math.hypot(hx, hy, hz) - 1) < 1e-12, `normal not unit at ${i}`);
    // Perpendicular to the position vector.
    const r = Math.hypot(pos[i * 3]!, pos[i * 3 + 1]!, pos[i * 3 + 2]!);
    const dot = (hx * pos[i * 3]! + hy * pos[i * 3 + 1]! + hz * pos[i * 3 + 2]!) / r;
    assert.ok(Math.abs(dot) < 1e-9, `normal not perpendicular to r at ${i}`);
    // Inertial: angle to the pole equals the inclination (an Earth-fixed
    // velocity would be off by up to ~3.5 deg).
    const inc = (Math.acos(hz) * 180) / Math.PI;
    assert.ok(Math.abs(inc - catalog.inclinationDeg[i]!) < 0.2, `normal inclination ${inc} at ${i}`);
  }
});

test('corrupted records are rejected, decayed ones yield NaN', () => {
  // Highest satellite in the fixture (lowest mean motion): survives a year of propagation.
  let hi = 0;
  for (let i = 1; i < catalog.count; i++) if (catalog.meanAltitudeKm[i]! > catalog.meanAltitudeKm[hi]!) hi = i;
  const good = records[hi]!;
  const garbage: ElementRecord = { ...good, name: 'GARBAGE', line1: 'nonsense', line2: 'more nonsense' };
  // Eccentricity 0.9999999 with a perigee far below the surface -> SGP4 error at epoch.
  const badEcc: ElementRecord = {
    ...good,
    name: 'BAD-ECC',
    line2: withChecksum(good.line2.substring(0, 26) + '9999999' + good.line2.substring(33, 68)),
  };
  // Very low orbit (16.45 rev/day ~ 180 km) with huge drag: valid at epoch, decays within weeks.
  const decaying: ElementRecord = {
    ...good,
    name: 'DECAYING',
    line1: withChecksum(good.line1.substring(0, 53) + ' 99999-1' + good.line1.substring(61, 68)),
    line2: withChecksum(good.line2.substring(0, 52) + '16.45000000' + good.line2.substring(63, 68)),
  };

  const res = buildCatalog([good, garbage, badEcc, decaying]);
  assert.equal(res.catalog.count + res.rejected, 4);
  assert.ok(res.rejected >= 2, `expected garbage + bad eccentricity rejected, got ${res.rejected}`);
  assert.equal(res.catalog.names[0], good.name);
  assert.ok(!res.catalog.names.includes('GARBAGE'));
  assert.ok(!res.catalog.names.includes('BAD-ECC'));

  const idx = res.catalog.names.indexOf('DECAYING');
  assert.ok(idx >= 0, 'decaying record should be valid at epoch');
  const out = new Float64Array(res.catalog.count * 3);
  const failures = propagateInto(res.satrecs, res.catalog.epochMs[idx]! + 365 * 86_400_000, out);
  assert.ok(failures >= 1);
  assert.ok(Number.isNaN(out[idx * 3]) && Number.isNaN(out[idx * 3 + 1]) && Number.isNaN(out[idx * 3 + 2]));
  assert.ok(Number.isFinite(out[0]), 'healthy satellite must still propagate');
});

test('performance: ~10k satellites per keyframe', () => {
  const big: ElementRecord[] = [];
  while (big.length < 10_000) big.push(...records);
  big.length = 10_000;
  const t0 = performance.now();
  const b = buildCatalog(big);
  const parseMs = performance.now() - t0;
  const out = new Float64Array(b.catalog.count * 3);
  const base = Date.UTC(2026, 10, 20, 12);
  for (let k = 0; k < 10; k++) propagateInto(b.satrecs, base + k * 1000, out); // warm-up
  const runs = 30;
  const times: number[] = [];
  for (let k = 0; k < runs; k++) {
    const t = performance.now();
    propagateInto(b.satrecs, base + 60_000 + k * 1000, out);
    times.push(performance.now() - t);
  }
  times.sort((x, y) => x - y);
  const median = times[runs >> 1]!;
  console.log(
    `10k sats: buildCatalog ${parseMs.toFixed(1)} ms, propagateInto median ${median.toFixed(2)} ms ` +
      `(min ${times[0]!.toFixed(2)}, max ${times[runs - 1]!.toFixed(2)})`,
  );
  assert.ok(median < 50, `propagation unexpectedly slow: ${median} ms`);
});

test('OMM records (NORAD id > 99,999) build into the catalog; V3 gets its bus, lasers and shell', () => {
  const header =
    'OBJECT_NAME,OBJECT_ID,EPOCH,MEAN_MOTION,ECCENTRICITY,INCLINATION,RA_OF_ASC_NODE,ARG_OF_PERICENTER,MEAN_ANOMALY,' +
    'EPHEMERIS_TYPE,CLASSIFICATION_TYPE,NORAD_CAT_ID,ELEMENT_SET_NO,REV_AT_EPOCH,BSTAR,MEAN_MOTION_DOT,MEAN_MOTION_DDOT';
  const csv = [
    header,
    // Starship Flight 15 V3 in its ~290 km insertion orbit.
    'STARLINK-40096,2026-256A,2026-11-20T03:12:00.000000,15.94000000,.00080000,30.5000,120.0000,90.0000,270.0000,0,U,101415,999,105,.50000000E-3,.10000000E-3,0',
    // Not a V3: Falcon 9 launch in 2026 at 53 deg -> V2 Mini.
    'STARLINK-39001,2026-201C,2026-11-20T04:00:00.000000,15.30000000,.00010000,53.1600,10.0000,90.0000,270.0000,0,U,100200,999,700,.20000000E-3,.10000000E-4,0',
    '',
  ].join('\n');
  const omm = parseOmmCsv(csv).records;
  assert.equal(omm.length, 2);
  const mixed: ElementRecord[] = [records[0]!, ...omm];
  const res = buildCatalog(mixed);
  assert.equal(res.rejected, 0);
  assert.equal(res.catalog.count, 3);
  assert.deepEqual([...res.catalog.noradIds], [records[0]!.noradId, 101415, 100200]);
  assert.deepEqual(res.catalog.intlDesignators, [records[0]!.intlDesignator, '26256A', '26201C']);
  // V3 (launch year 2026, inclination < 35 deg).
  assert.equal(res.catalog.bus[1], BUS_V3);
  assert.equal(res.catalog.lasers[1], 6);
  assert.equal(res.catalog.shell[1], SHELL_V3);
  assert.ok(Math.abs(res.catalog.inclinationDeg[1]! - 30.5) < 1e-4);
  assert.ok(res.catalog.meanAltitudeKm[1]! > 250 && res.catalog.meanAltitudeKm[1]! < 330, `alt ${res.catalog.meanAltitudeKm[1]}`);
  assert.equal(res.catalog.epochMs[1], Date.UTC(2026, 10, 20, 3, 12));
  // V2 Mini at 53 deg stays in the 53-deg shell.
  assert.equal(res.catalog.bus[2], BUS_V2MINI);
  assert.equal(res.catalog.shell[2], 0);
  // "V3" in the name is enough, whatever the designator.
  const named = buildCatalog([{ ...omm[1]!, name: 'STARLINK V3 TEST' }]);
  assert.equal(named.catalog.bus[0], BUS_V3);
  assert.equal(named.catalog.shell[0], SHELL_V3);

  const out = new Float64Array(res.catalog.count * 3);
  assert.equal(propagateInto(res.satrecs, Date.UTC(2026, 10, 20, 6), out), 0);
  const r = Math.hypot(out[3]!, out[4]!, out[5]!);
  assert.ok(r > 6.6e6 && r < 6.75e6, `V3 radius ${r}`);
});

test('recordToSatrec: OMM preferred over TLE, error without elements', () => {
  const rec = records[0]!;
  assert.equal(recordToSatrec(rec).satnum, rec.line1.substring(2, 7));
  const csv =
    'OBJECT_NAME,OBJECT_ID,EPOCH,MEAN_MOTION,ECCENTRICITY,INCLINATION,RA_OF_ASC_NODE,ARG_OF_PERICENTER,MEAN_ANOMALY,NORAD_CAT_ID\n' +
    'STARLINK-40096,2026-256A,2026-11-20T03:12:00.000000,15.94,.0008,30.5,120,90,270,101415\n';
  const both: ElementRecord = { ...rec, omm: parseOmmCsv(csv).records[0]!.omm! };
  assert.equal(recordToSatrec(both).satnum, '101415');
  assert.throws(() => recordToSatrec({ name: 'EMPTY', noradId: 1, intlDesignator: '' }), /No orbital elements/);
  const empty = buildCatalog([{ name: 'EMPTY', noradId: 1, intlDesignator: '' }, rec]);
  assert.equal(empty.rejected, 1);
  assert.equal(empty.catalog.count, 1);
});
