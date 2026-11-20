import { strict as assert } from 'node:assert';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import type { ElementRecord } from '../src/shared/contracts.ts';
import { SHELL_TRANSIT } from '../src/shared/shells.ts';
import {
  DEFAULT_LASER_TOPOLOGY_OPTIONS,
  LINK_CROSS_PLANE,
  LINK_IN_PLANE,
  type LaserTopologyResult,
  type TopologyCatalog,
  computeLaserLinks,
} from '../src/worker/laserTopology.ts';
import { buildCatalog, propagateInto } from '../src/worker/propagate.ts';

const R_EARTH_KM = 6378.137;
const MU = 398600.4418;
const DEG = Math.PI / 180;

// ---------------------------------------------------------------------------
// Synthetic constellations (circular Walker-delta shells, inertial = ECEF at t)
// ---------------------------------------------------------------------------

interface ShellSpec {
  planes: number;
  perPlane: number;
  incDeg: number;
  altKm: number;
  /** Walker phasing factor F. */
  phasing: number;
  shell: number;
  lasers: number;
  raan0Deg?: number;
}

interface Synthetic {
  positions: Float64Array;
  velocities: Float64Array;
  catalog: TopologyCatalog;
  plane: Int32Array;
  slot: Int32Array;
  spec: Int32Array;
}

function synth(specs: ShellSpec[]): Synthetic {
  const n = specs.reduce((s, sp) => s + sp.planes * sp.perPlane, 0);
  const positions = new Float64Array(n * 3);
  const velocities = new Float64Array(n * 3);
  const shell = new Uint8Array(n);
  const lasers = new Uint8Array(n);
  const inclinationDeg = new Float32Array(n);
  const meanAltitudeKm = new Float32Array(n);
  const plane = new Int32Array(n);
  const slot = new Int32Array(n);
  const spec = new Int32Array(n);
  let k = 0;
  specs.forEach((sp, si) => {
    const r = (R_EARTH_KM + sp.altKm) * 1000;
    const v = Math.sqrt(MU / (R_EARTH_KM + sp.altKm)) * 1000;
    const inc = sp.incDeg * DEG;
    for (let p = 0; p < sp.planes; p++) {
      const raan = (sp.raan0Deg ?? 0) * DEG + (2 * Math.PI * p) / sp.planes;
      for (let s = 0; s < sp.perPlane; s++, k++) {
        const u = (2 * Math.PI * s) / sp.perPlane + (2 * Math.PI * sp.phasing * p) / (sp.planes * sp.perPlane);
        const cO = Math.cos(raan);
        const sO = Math.sin(raan);
        const cu = Math.cos(u);
        const su = Math.sin(u);
        const ci = Math.cos(inc);
        const sI = Math.sin(inc);
        positions[k * 3] = r * (cO * cu - sO * su * ci);
        positions[k * 3 + 1] = r * (sO * cu + cO * su * ci);
        positions[k * 3 + 2] = r * su * sI;
        velocities[k * 3] = v * (-cO * su - sO * cu * ci);
        velocities[k * 3 + 1] = v * (-sO * su + cO * cu * ci);
        velocities[k * 3 + 2] = v * cu * sI;
        shell[k] = sp.shell;
        lasers[k] = sp.lasers;
        inclinationDeg[k] = sp.incDeg;
        meanAltitudeKm[k] = sp.altKm;
        plane[k] = p;
        slot[k] = s;
        spec[k] = si;
      }
    }
  });
  return { positions, velocities, catalog: { count: n, shell, lasers, inclinationDeg, meanAltitudeKm }, plane, slot, spec };
}

function run(s: Synthetic, options?: Parameters<typeof computeLaserLinks>[0]['options']): LaserTopologyResult {
  return computeLaserLinks({ positions: s.positions, velocities: s.velocities, catalog: s.catalog, timeMs: 0, options });
}

function degrees(n: number, pairs: Uint32Array): Uint8Array {
  const d = new Uint8Array(n);
  for (let k = 0; k < pairs.length; k++) d[pairs[k]!]!++;
  return d;
}

function distKm(pos: Float64Array, a: number, b: number): number {
  return Math.hypot(pos[a * 3]! - pos[b * 3]!, pos[a * 3 + 1]! - pos[b * 3 + 1]!, pos[a * 3 + 2]! - pos[b * 3 + 2]!) / 1000;
}

function latDeg(pos: Float64Array, i: number): number {
  return Math.asin(pos[i * 3 + 2]! / Math.hypot(pos[i * 3]!, pos[i * 3 + 1]!, pos[i * 3 + 2]!)) / DEG;
}

/** Minimum altitude (km) of the straight segment between two satellites. */
function chordMinAltKm(pos: Float64Array, a: number, b: number): number {
  const ax = pos[a * 3]! / 1000, ay = pos[a * 3 + 1]! / 1000, az = pos[a * 3 + 2]! / 1000;
  const dx = pos[b * 3]! / 1000 - ax, dy = pos[b * 3 + 1]! / 1000 - ay, dz = pos[b * 3 + 2]! / 1000 - az;
  const dd = dx * dx + dy * dy + dz * dz;
  const t = Math.min(1, Math.max(0, -(ax * dx + ay * dy + az * dz) / dd));
  return Math.hypot(ax + t * dx, ay + t * dy, az + t * dz) - R_EARTH_KM;
}

/** Structural invariants every result must satisfy. */
function checkInvariants(res: LaserTopologyResult, n: number, lasers: Uint8Array, pos: Float64Array): void {
  const { pairs, kinds } = res;
  assert.equal(pairs.length, kinds.length * 2);
  const seen = new Set<number>();
  for (let k = 0; k < kinds.length; k++) {
    const a = pairs[k * 2]!;
    const b = pairs[k * 2 + 1]!;
    assert.ok(a < b && b < n, `pair ${a}-${b} not ordered / out of range`);
    const key = a * n + b;
    assert.ok(!seen.has(key), `duplicate link ${a}-${b}`);
    seen.add(key);
    assert.ok(kinds[k] === LINK_IN_PLANE || kinds[k] === LINK_CROSS_PLANE);
    assert.ok(lasers[a]! > 0 && lasers[b]! > 0, 'link to a satellite without lasers');
    const clearance = chordMinAltKm(pos, a, b);
    assert.ok(clearance >= DEFAULT_LASER_TOPOLOGY_OPTIONS.minClearanceAltKm - 1e-6, `link ${a}-${b} grazes/crosses Earth (${clearance.toFixed(0)} km)`);
    const d = distKm(pos, a, b);
    const max = kinds[k] === LINK_IN_PLANE ? DEFAULT_LASER_TOPOLOGY_OPTIONS.inPlaneMaxKm : DEFAULT_LASER_TOPOLOGY_OPTIONS.crossPlaneMaxKm;
    assert.ok(d <= max + 1e-6, `link ${a}-${b} too long: ${d.toFixed(0)} km`);
  }
  const deg = degrees(n, pairs);
  for (let i = 0; i < n; i++) assert.ok(deg[i]! <= lasers[i]!, `degree ${deg[i]} > cap ${lasers[i]} at ${i}`);
}

function wrapDiff(a: number, b: number, m: number): number {
  const d = (((a - b) % m) + m) % m;
  return Math.min(d, m - d);
}

// ---------------------------------------------------------------------------

test('Walker-delta 72x22 @53 deg, 550 km, 4 terminals: a regular +Grid', () => {
  const s = synth([{ planes: 72, perPlane: 22, incDeg: 53, altKm: 550, phasing: 17, shell: 0, lasers: 4 }]);
  const n = s.catalog.count;
  const res = run(s);
  checkInvariants(res, n, s.catalog.lasers, s.positions);

  let inPlane = 0;
  let cross = 0;
  for (let k = 0; k < res.kinds.length; k++) {
    const a = res.pairs[k * 2]!;
    const b = res.pairs[k * 2 + 1]!;
    if (res.kinds[k] === LINK_IN_PLANE) {
      inPlane++;
      assert.equal(s.plane[a], s.plane[b], 'in-plane link across planes');
      assert.equal(wrapDiff(s.slot[a]!, s.slot[b]!, 22), 1, `in-plane link skips a satellite (${s.slot[a]} -> ${s.slot[b]})`);
    } else {
      cross++;
      assert.equal(wrapDiff(s.plane[a]!, s.plane[b]!, 72), 1, `cross-plane link to a non-adjacent plane (${s.plane[a]} -> ${s.plane[b]})`);
    }
  }
  // Every plane is a closed ring of 22 in-plane links.
  assert.equal(inPlane, 72 * 22);
  const deg = degrees(n, res.pairs);
  const mean = (2 * res.kinds.length) / n;
  let four = 0;
  for (let i = 0; i < n; i++) if (deg[i] === 4) four++;
  console.log(`walker 53: ${res.kinds.length} links (${inPlane} in-plane, ${cross} cross), mean degree ${mean.toFixed(2)}, ${four}/${n} sats with 4`);
  assert.ok(mean > 3.7 && mean <= 4, `mean degree ${mean}`);
});

test('V2 Mini (3 terminals): in-plane links come first, cap respected', () => {
  const s = synth([{ planes: 36, perPlane: 20, incDeg: 43, altKm: 530, phasing: 5, shell: 1, lasers: 3 }]);
  const n = s.catalog.count;
  const res = run(s);
  checkInvariants(res, n, s.catalog.lasers, s.positions);
  let inPlane = 0;
  for (let k = 0; k < res.kinds.length; k++) if (res.kinds[k] === LINK_IN_PLANE) inPlane++;
  assert.equal(inPlane, 36 * 20, 'every in-plane ring must be complete before cross links');
  const deg = degrees(n, res.pairs);
  let three = 0;
  for (let i = 0; i < n; i++) if (deg[i] === 3) three++;
  console.log(`v2mini shell: ${res.kinds.length} links, ${three}/${n} sats use all 3 terminals`);
  assert.ok(three > n * 0.8);
});

test('satellites without terminals or in transit get no links', () => {
  const s = synth([{ planes: 24, perPlane: 22, incDeg: 53, altKm: 550, phasing: 1, shell: 0, lasers: 4 }]);
  const n = s.catalog.count;
  const lasers = s.catalog.lasers;
  const shell = s.catalog.shell;
  for (let i = 0; i < n; i++) {
    if (s.plane[i]! % 4 === 1) lasers[i] = 0; // v1.0 plane: no terminals
    if (i % 7 === 3) shell[i] = SHELL_TRANSIT;
  }
  // A satellite with an invalid position.
  s.positions[5 * 3] = NaN;
  const res = run(s);
  const deg = degrees(n, res.pairs);
  for (let i = 0; i < n; i++) {
    if (lasers[i] === 0 || shell[i] === SHELL_TRANSIT || i === 5) assert.equal(deg[i], 0, `ineligible sat ${i} linked`);
  }
  assert.ok(res.kinds.length > 0);
  // Check the remaining invariants on valid positions only.
  s.positions[5 * 3] = s.positions[5 * 3 + 1]!;
  for (let k = 0; k < res.kinds.length; k++) {
    const a = res.pairs[k * 2]!;
    const b = res.pairs[k * 2 + 1]!;
    assert.ok(a < b && lasers[a]! > 0 && lasers[b]! > 0);
  }
});

test('no link passes through (or grazes) the Earth', () => {
  // 3 satellites per plane at 120 deg: the chord (11,900 km) passes through
  // the Earth. Even with the length limit lifted, line of sight must block it.
  const s = synth([{ planes: 4, perPlane: 3, incDeg: 53, altKm: 550, phasing: 0, shell: 0, lasers: 4 }]);
  const res = run(s, { inPlaneMaxKm: 20_000 });
  for (let k = 0; k < res.kinds.length; k++) assert.notEqual(res.kinds[k], LINK_IN_PLANE);
  // 10 per plane at 36 deg spacing: chord 4,300 km, min altitude ~ 320 km -> allowed.
  const s2 = synth([{ planes: 1, perPlane: 10, incDeg: 53, altKm: 550, phasing: 0, shell: 0, lasers: 4 }]);
  const res2 = run(s2);
  assert.equal(res2.kinds.length, 10);
  const s3 = synth([{ planes: 1, perPlane: 6, incDeg: 53, altKm: 550, phasing: 0, shell: 0, lasers: 4 }]);
  const res3 = run(s3, { inPlaneMaxKm: 20_000 });
  // 60 deg spacing: chord 6,928 km, min radius 6,000 km < Earth -> blocked.
  assert.equal(res3.kinds.length, 0);
});

test('cross-plane links above |lat| 70 deg are dropped, except (capped) on polar shells', () => {
  const s = synth([{ planes: 30, perPlane: 30, incDeg: 75, altKm: 560, phasing: 1, shell: 6, lasers: 4 }]);
  const res = run(s);
  checkInvariants(res, s.catalog.count, s.catalog.lasers, s.positions);
  let crossHigh = 0;
  let cross = 0;
  for (let k = 0; k < res.kinds.length; k++) {
    if (res.kinds[k] !== LINK_CROSS_PLANE) continue;
    cross++;
    const a = res.pairs[k * 2]!;
    const b = res.pairs[k * 2 + 1]!;
    if (Math.abs(latDeg(s.positions, a)) > 70 || Math.abs(latDeg(s.positions, b)) > 70) crossHigh++;
  }
  assert.ok(cross > 0);
  assert.equal(crossHigh, 0);

  // 24 planes over 360 deg: adjacent planes 15 deg apart in RAAN.
  const p = synth([{ planes: 24, perPlane: 30, incDeg: 97.6, altKm: 560, phasing: 1, shell: 3, lasers: 4 }]);
  const resP = run(p);
  checkInvariants(resP, p.catalog.count, p.catalog.lasers, p.positions);
  let polarHigh = 0;
  for (let k = 0; k < resP.kinds.length; k++) {
    if (resP.kinds[k] !== LINK_CROSS_PLANE) continue;
    const a = resP.pairs[k * 2]!;
    const b = resP.pairs[k * 2 + 1]!;
    if (Math.abs(latDeg(p.positions, a)) > 70 || Math.abs(latDeg(p.positions, b)) > 70) {
      polarHigh++;
      assert.ok(distKm(p.positions, a, b) <= DEFAULT_LASER_TOPOLOGY_OPTIONS.polarHighLatCrossMaxKm + 1e-6);
    }
  }
  console.log(`polar shell: ${resP.kinds.length} links, ${polarHigh} cross-plane links above 70 deg (capped)`);
  assert.ok(polarHigh > 0, 'polar exception should keep some high-latitude cross links');
});

test('different shells (altitude / inclination) never link to each other', () => {
  const s = synth([
    { planes: 72, perPlane: 22, incDeg: 53.0, altKm: 550, phasing: 17, shell: 0, lasers: 4 },
    { planes: 72, perPlane: 22, incDeg: 53.2, altKm: 540, phasing: 17, shell: 0, lasers: 4, raan0Deg: 2.5 },
    { planes: 36, perPlane: 20, incDeg: 43, altKm: 550, phasing: 5, shell: 1, lasers: 3 },
  ]);
  const res = run(s);
  checkInvariants(res, s.catalog.count, s.catalog.lasers, s.positions);
  for (let k = 0; k < res.kinds.length; k++) {
    assert.equal(s.spec[res.pairs[k * 2]!], s.spec[res.pairs[k * 2 + 1]!], 'link between shells');
  }
});

test('polar plane pair 0.9 deg apart: in-plane links stay in their plane, the pair is cross-linked', () => {
  // The real 97.3 deg / ~462 km shell has plane pairs ~0.85-0.95 deg apart in
  // RAAN (launch pairs such as 2025-179 / 2025-211). `spec` is the plane here.
  const s = synth([
    { planes: 1, perPlane: 24, incDeg: 97.3, altKm: 462, phasing: 0, shell: 3, lasers: 4 },
    { planes: 1, perPlane: 24, incDeg: 97.3, altKm: 462, phasing: 0, shell: 3, lasers: 4, raan0Deg: 0.9 },
  ]);
  const res = run(s);
  checkInvariants(res, s.catalog.count, s.catalog.lasers, s.positions);
  let inPlane = 0;
  let crossPair = 0;
  for (let k = 0; k < res.kinds.length; k++) {
    const pa = s.spec[res.pairs[k * 2]!];
    const pb = s.spec[res.pairs[k * 2 + 1]!];
    if (res.kinds[k] === LINK_IN_PLANE) {
      inPlane++;
      assert.equal(pa, pb, 'in-plane link joins two distinct planes');
    } else if (pa !== pb) {
      crossPair++;
    }
  }
  assert.equal(inPlane, 48, 'each plane should be a closed 24-satellite ring');
  assert.ok(crossPair > 0, 'the two planes should be cross-linked');
});

test('a small coplanarMaxDeg keeps polar normals inside the normal grid', () => {
  // Below ~0.46 deg the unclamped normal-grid scale exceeds the 8-bit cell
  // range and polar normals (components near +-1) used to be dropped.
  const s = synth([{ planes: 24, perPlane: 30, incDeg: 97.6, altKm: 560, phasing: 1, shell: 3, lasers: 4 }]);
  const res = run(s, { coplanarMaxDeg: 0.3 });
  checkInvariants(res, s.catalog.count, s.catalog.lasers, s.positions);
  let inPlane = 0;
  for (let k = 0; k < res.kinds.length; k++) if (res.kinds[k] === LINK_IN_PLANE) inPlane++;
  assert.equal(inPlane, 24 * 30, 'every plane should be a closed ring');
});

// ---------------------------------------------------------------------------
// Real element sets
// ---------------------------------------------------------------------------

function parseTle(text: string): ElementRecord[] {
  const lines = text.split(/\r?\n/).map((l) => l.trimEnd()).filter((l) => l.length > 0);
  const out: ElementRecord[] = [];
  for (let i = 0; i + 2 < lines.length; i += 3) {
    const l1 = lines[i + 1]!;
    out.push({
      name: lines[i]!.trim(),
      noradId: Number(l1.substring(2, 7)),
      intlDesignator: l1.substring(9, 17).trim(),
      line1: l1,
      line2: lines[i + 2]!,
    });
  }
  return out;
}

test('fixture (real SupGP elements): invariants, plausible mesh, satrec path == normals path', () => {
  const records = parseTle(readFileSync(new URL('../fixtures/sup-gp-starlink-partial.tle', import.meta.url), 'utf8'));
  const { catalog, satrecs } = buildCatalog(records);
  const n = catalog.count;
  const timeMs = Date.UTC(2026, 10, 20, 12);
  const pos = new Float64Array(n * 3);
  const nrm = new Float64Array(n * 3);
  propagateInto(satrecs, timeMs, pos, nrm);

  const res = computeLaserLinks({ positions: pos, normals: nrm, catalog, timeMs });
  checkInvariants(res, n, catalog.lasers, pos);
  const viaSatrecs = computeLaserLinks({ positions: pos, satrecs, catalog, timeMs });
  assert.deepEqual(viaSatrecs.pairs, res.pairs);
  assert.deepEqual(viaSatrecs.kinds, res.kinds);

  let equipped = 0;
  for (let i = 0; i < n; i++) if (catalog.lasers[i]! > 0 && catalog.shell[i] !== SHELL_TRANSIT) equipped++;
  const deg = degrees(n, res.pairs);
  let linked = 0;
  for (let i = 0; i < n; i++) {
    if (deg[i]! > 0) linked++;
    if (catalog.lasers[i] === 0) assert.equal(deg[i], 0);
    if (catalog.shell[i] === SHELL_TRANSIT) assert.equal(deg[i], 0);
  }
  let inPlane = 0;
  for (let k = 0; k < res.kinds.length; k++) if (res.kinds[k] === LINK_IN_PLANE) inPlane++;
  console.log(
    `fixture: ${n} sats, ${equipped} equipped+operational, ${linked} linked, ` +
      `${res.kinds.length} links (${inPlane} in-plane, ${res.kinds.length - inPlane} cross)`,
  );
  // The partial fixture holds only a few launches per plane, but plenty of
  // equipped satellites should still find neighbours.
  assert.ok(res.kinds.length > 50, 'expected a non-trivial mesh on real data');

  // An Earth-fixed velocity would tilt planes by ~3.5 deg: the normals from
  // propagateInto must be inertial (inclination == angle(h, z)).
  for (let i = 0; i < n; i++) {
    const inc = Math.acos(nrm[i * 3 + 2]!) / DEG;
    assert.ok(Math.abs(inc - catalog.inclinationDeg[i]!) < 0.2, `normal inclination ${inc} vs ${catalog.inclinationDeg[i]}`);
  }
});

test('performance: ~10k satellites across 7 shells', () => {
  const s = synth([
    { planes: 72, perPlane: 22, incDeg: 53.0, altKm: 550, phasing: 17, shell: 0, lasers: 4 },
    { planes: 72, perPlane: 22, incDeg: 53.2, altKm: 540, phasing: 17, shell: 0, lasers: 4, raan0Deg: 2.5 },
    { planes: 72, perPlane: 44, incDeg: 53.1, altKm: 480, phasing: 11, shell: 0, lasers: 3, raan0Deg: 1.2 },
    { planes: 48, perPlane: 40, incDeg: 43, altKm: 530, phasing: 7, shell: 1, lasers: 3 },
    { planes: 36, perPlane: 20, incDeg: 70, altKm: 570, phasing: 5, shell: 2, lasers: 4 },
    { planes: 24, perPlane: 29, incDeg: 97.6, altKm: 560, phasing: 3, shell: 3, lasers: 4 },
    { planes: 28, perPlane: 20, incDeg: 33, altKm: 535, phasing: 3, shell: 4, lasers: 3 },
  ]);
  const n = s.catalog.count;
  let res = run(s);
  for (let k = 0; k < 4; k++) res = run(s); // warm-up
  const times: number[] = [];
  for (let k = 0; k < 15; k++) {
    const t = performance.now();
    res = run(s);
    times.push(performance.now() - t);
  }
  times.sort((a, b) => a - b);
  const median = times[times.length >> 1]!;
  checkInvariants(res, n, s.catalog.lasers, s.positions);
  console.log(
    `${n} sats -> ${res.kinds.length} links: computeLaserLinks median ${median.toFixed(2)} ms ` +
      `(min ${times[0]!.toFixed(2)}, max ${times[times.length - 1]!.toFixed(2)})`,
  );
  assert.ok(median < 100, `topology unexpectedly slow: ${median} ms`);
});
