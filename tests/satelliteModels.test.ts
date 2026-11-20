/**
 * Pure math behind the close-up satellite models (satelliteModelMath.ts):
 * attitude frame, sun-tracking wing angle and node matrix, cylindrical
 * shadow, and the candidate search / selection used by SatellitePoints +
 * SatelliteModels. No Cesium runtime.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  EARTH_ROTATION_RAD_S,
  EARTH_SHADOW_RADIUS_M,
  attitudeMatrix,
  insertNearest,
  pickEvictableSlot,
  scanNearest,
  selectCandidates,
  sunInGltfBody,
  sunInGltfBodyXForward,
  sunlitFactor,
  wingNodeMatrix,
  wingSunAngle,
} from '../src/render/satelliteModelMath.ts';

const EPS = 1e-9;

function col(m: Float64Array, c: number): [number, number, number] {
  return [m[c * 4], m[c * 4 + 1], m[c * 4 + 2]];
}
function dot(a: ArrayLike<number>, b: ArrayLike<number>): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
function cross(a: readonly number[], b: readonly number[]): [number, number, number] {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
function norm(a: readonly number[]): number {
  return Math.sqrt(dot(a, a));
}
function close(a: number, b: number, eps = 1e-9, msg?: string): void {
  assert.ok(Math.abs(a - b) <= eps, `${msg ?? ''} expected ${b}, got ${a}`);
}

/** glTF -> Cesium model frame (ModelUtility.getAxisCorrectionMatrix(Y, Z)): +Z->+X, +X->+Y, +Y->+Z. */
function gltfToModel(v: readonly number[]): [number, number, number] {
  return [v[2], v[0], v[1]];
}

test('attitude: orthonormal right-handed frame, -Z to nadir, +X along the horizontal velocity', () => {
  const m = new Float64Array(16);
  // Inclined LEO state (ECEF), velocity with a radial component.
  const p = [4_000_000, 3_500_000, 4_200_000];
  const v = [-4_500, 1_200, 5_500];
  assert.ok(attitudeMatrix(p[0], p[1], p[2], v[0], v[1], v[2], m));
  const x = col(m, 0);
  const y = col(m, 1);
  const z = col(m, 2);
  close(norm(x), 1, EPS, '|x|');
  close(norm(y), 1, EPS, '|y|');
  close(norm(z), 1, EPS, '|z|');
  close(dot(x, y), 0, EPS, 'x.y');
  close(dot(y, z), 0, EPS, 'y.z');
  close(dot(z, x), 0, EPS, 'z.x');
  const xy = cross(x, y);
  close(dot(xy, z), 1, EPS, 'right handed');
  // -Z points at the Earth's centre (nadir).
  const r = norm(p);
  close(dot([-z[0], -z[1], -z[2]], [-p[0] / r, -p[1] / r, -p[2] / r]), 1, EPS, 'nadir');
  // Translation column.
  assert.deepEqual([m[12], m[13], m[14], m[15]], [p[0], p[1], p[2], 1]);
  assert.deepEqual([m[3], m[7], m[11]], [0, 0, 0]);
  // +X is the inertial velocity (v + ω×r) with the radial part removed.
  const vi = [v[0] - EARTH_ROTATION_RAD_S * p[1], v[1] + EARTH_ROTATION_RAD_S * p[0], v[2]];
  const zr = [p[0] / r, p[1] / r, p[2] / r];
  const d = dot(vi, zr);
  const h = [vi[0] - d * zr[0], vi[1] - d * zr[1], vi[2] - d * zr[2]];
  const hn = norm(h);
  close(dot(x, [h[0] / hn, h[1] / hn, h[2] / hn]), 1, EPS, 'forward');
});

test("glTF -> model mapping equals Cesium's Axis.Y_UP_TO_Z_UP * Axis.Z_UP_TO_X_UP", () => {
  // Column-major 3x3 literals copied from @cesium/engine Source/Scene/Axis.js (1.147).
  const yUpToZUp = [1, 0, 0, 0, 0, 1, 0, -1, 0];
  const zUpToXUp = [0, 0, -1, 0, 1, 0, 1, 0, 0];
  const mul = (m: number[], v: readonly number[]): number[] => [
    m[0] * v[0] + m[3] * v[1] + m[6] * v[2],
    m[1] * v[0] + m[4] * v[1] + m[7] * v[2],
    m[2] * v[0] + m[5] * v[1] + m[8] * v[2],
  ];
  // ModelUtility.getAxisCorrectionMatrix: result = YtoZ; result = result * ZtoX.
  for (const v of [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ]) {
    assert.deepEqual(mul(yUpToZUp, mul(zUpToXUp, v)).map((c) => c + 0), gltfToModel(v));
  }
});

test('attitude: glTF +Z (forward) maps to the flight direction, glTF +Y (up) to zenith', () => {
  const m = new Float64Array(16);
  // Equatorial prograde orbit at the +X axis: flies towards +Y, zenith +X.
  attitudeMatrix(6_928_137, 0, 0, 0, 7_600, 0, m);
  const apply = (vModel: readonly number[]): number[] => [
    m[0] * vModel[0] + m[4] * vModel[1] + m[8] * vModel[2],
    m[1] * vModel[0] + m[5] * vModel[1] + m[9] * vModel[2],
    m[2] * vModel[0] + m[6] * vModel[1] + m[10] * vModel[2],
  ];
  const fwd = apply(gltfToModel([0, 0, 1]));
  const up = apply(gltfToModel([0, 1, 0]));
  close(fwd[1], 1, EPS, 'glTF +Z -> +Y ECEF (velocity)');
  close(up[0], 1, EPS, 'glTF +Y -> +X ECEF (zenith)');
  const side = apply(gltfToModel([1, 0, 0]));
  // ŷ = ẑ × x̂ = X × Y = +Z (orbit normal for a prograde equatorial orbit).
  close(side[2], 1, EPS, 'glTF +X -> orbit normal');
});

test('attitude: degenerate velocity still gives an orthonormal frame; invalid position rejected', () => {
  const m = new Float64Array(16);
  // At the pole with zero ECEF velocity (ω×r = 0 there too).
  assert.ok(attitudeMatrix(0, 0, 6_900_000, 0, 0, 0, m));
  const x = col(m, 0);
  const z = col(m, 2);
  close(norm(x), 1, EPS);
  close(dot(x, z), 0, EPS);
  close(z[2], 1, EPS);
  const before = Float64Array.from(m);
  assert.equal(attitudeMatrix(Number.NaN, 0, 0, 1, 0, 0, m), false);
  assert.deepEqual(Array.from(m), Array.from(before));
});

test('wing angle: rotated rest normal points at the sun projected on the plane ⟂ wing axis', () => {
  const m = new Float64Array(16);
  attitudeMatrix(6_928_137, 0, 0, 0, 7_600, 0, m);
  const sb = new Float64Array(3);
  // A few sun directions (ECEF, unit).
  const suns = [
    [1, 0, 0], // zenith -> angle 0
    [0, 1, 0], // along track ahead -> +90°
    [0, -1, 0],
    [-1, 0, 0], // behind the Earth -> 180°
    [0.3, 0.5, 0.81],
  ];
  for (const s0 of suns) {
    const n = norm(s0);
    const s = [s0[0] / n, s0[1] / n, s0[2] / n];
    sunInGltfBody(m, s[0], s[1], s[2], sb);
    const theta = wingSunAngle(sb, [1, 0, 0], [0, 1, 0]);
    // Rest normal glTF +Y rotated about glTF +X by theta: (0, cos, sin) in glTF.
    const nG = [0, Math.cos(theta), Math.sin(theta)];
    // Best possible: |projection of s onto the glTF YZ plane|.
    const best = Math.hypot(sb[1], sb[2]);
    close(dot(nG, sb), best, 1e-9, `sun ${s0}`);
  }
  // Specific angles.
  sunInGltfBody(m, 1, 0, 0, sb);
  close(wingSunAngle(sb, [1, 0, 0], [0, 1, 0]), 0, EPS, 'zenith sun');
  sunInGltfBody(m, 0, 1, 0, sb);
  close(wingSunAngle(sb, [1, 0, 0], [0, 1, 0]), Math.PI / 2, EPS, 'sun ahead');
});

test('wing node matrix: rotation about the parent axis through the pivot, translation kept', () => {
  // Pivot at (2, 0.5, -0.3) mirrored by 180° about Y (a "left" wing).
  const orig = new Float64Array([-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 2, 0.5, -0.3, 1]);
  const out = new Float64Array(16);
  const theta = 0.7;
  wingNodeMatrix(orig, theta, out, [1, 0, 0]);
  assert.deepEqual([out[12], out[13], out[14], out[15]], [2, 0.5, -0.3, 1]);
  // Columns are Rx(theta) * original columns.
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  const rx = (v: number[]): number[] => [v[0], c * v[1] - s * v[2], s * v[1] + c * v[2]];
  for (let k = 0; k < 3; k++) {
    const expected = rx([orig[k * 4], orig[k * 4 + 1], orig[k * 4 + 2]]);
    for (let i = 0; i < 3; i++) close(out[k * 4 + i], expected[i], 1e-12);
  }
  // theta = 0 reproduces the original.
  wingNodeMatrix(orig, 0, out, [1, 0, 0]);
  assert.deepEqual(Array.from(out), Array.from(orig));
  // Generic axis: rotating about Z by 90° maps local X column to +Y.
  const ident = new Float64Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
  wingNodeMatrix(ident, Math.PI / 2, out, [0, 0, 1]);
  close(out[0], 0, 1e-12);
  close(out[1], 1, 1e-12);
});

test('cylindrical shadow: lit on the day side, dark in the umbra, smooth edge', () => {
  const R = EARTH_SHADOW_RADIUS_M;
  const r = 6_928_137;
  // Sun along +X.
  assert.equal(sunlitFactor(r, 0, 0, 1, 0, 0), 1, 'sub-solar');
  assert.equal(sunlitFactor(-r, 0, 0, 1, 0, 0), 0, 'anti-solar (umbra)');
  assert.equal(sunlitFactor(0, r, 0, 1, 0, 0), 1, 'terminator, outside cylinder');
  assert.equal(sunlitFactor(-1_000_000, R + 100_000, 0, 1, 0, 0), 1, 'night side but outside the cylinder');
  const edge = sunlitFactor(-1_000_000, R, 0, 1, 0, 0);
  close(edge, 0.5, 1e-9, 'shadow edge');
  const a = sunlitFactor(-1_000_000, R - 10_000, 0, 1, 0, 0);
  const b = sunlitFactor(-1_000_000, R + 10_000, 0, 1, 0, 0);
  assert.ok(a > 0 && a < 0.5 && b > 0.5 && b < 1, 'monotonic soft edge');
});

test('insertNearest keeps the nearest N in ascending order', () => {
  const idx = new Int32Array(3);
  const d = new Float64Array(3);
  let n = 0;
  const items: [number, number][] = [
    [10, 50],
    [11, 20],
    [12, 70],
    [13, 5],
    [14, 30],
    [15, 90],
  ];
  for (const [i, d2] of items) n = insertNearest(i, d2, idx, d, n, 3);
  assert.equal(n, 3);
  assert.deepEqual(Array.from(idx), [13, 11, 14]);
  assert.deepEqual(Array.from(d), [5, 20, 30]);
  assert.equal(insertNearest(1, 1, idx, d, 0, 0), 0, 'zero capacity');
});

test('scanNearest: radius, interpolation, NaN, hidden shells and the always-index', () => {
  const count = 6;
  const a = new Float64Array(count * 3);
  const b = new Float64Array(count * 3);
  const put = (arr: Float64Array, i: number, x: number, y: number, z: number): void => {
    arr[i * 3] = x;
    arr[i * 3 + 1] = y;
    arr[i * 3 + 2] = z;
  };
  const C = 7_000_000;
  // 0: 1 km away in A, 3 km in B -> 2 km at t = 0.5
  put(a, 0, C + 1_000, 0, 0);
  put(b, 0, C + 3_000, 0, 0);
  // 1: 10 km
  put(a, 1, C, 10_000, 0);
  put(b, 1, C, 10_000, 0);
  // 2: 60 km (outside)
  put(a, 2, C, 0, 60_000);
  put(b, 2, C, 0, 60_000);
  // 3: NaN
  put(a, 3, Number.NaN, Number.NaN, Number.NaN);
  put(b, 3, Number.NaN, Number.NaN, Number.NaN);
  // 4: 5 km but hidden shell
  put(a, 4, C - 5_000, 0, 0);
  put(b, 4, C - 5_000, 0, 0);
  // 5: 7 km, hidden shell but "always" (selected)
  put(a, 5, C, -7_000, 0);
  put(b, 5, C, -7_000, 0);
  const shell = new Uint8Array([0, 0, 0, 0, 1, 1]);
  const vis = new Uint8Array([1, 0]);
  const idx = new Int32Array(8);
  const d2 = new Float64Array(8);
  let n = scanNearest(a, b, 0.5, 0.5, count, C, 0, 0, 40_000, shell, vis, 5, idx, d2);
  assert.equal(n, 3);
  assert.deepEqual(Array.from(idx.subarray(0, n)), [0, 5, 1]);
  close(Math.sqrt(d2[0]), 2_000, 1e-6, 'interpolated distance');
  // Single slot (wb = 0): uses A only.
  n = scanNearest(a, b, 1, 0, count, C, 0, 0, 40_000, shell, vis, -1, idx, d2);
  assert.deepEqual(Array.from(idx.subarray(0, n)), [0, 1]);
  close(Math.sqrt(d2[0]), 1_000, 1e-6);
  // Capacity limit keeps the nearest.
  n = scanNearest(a, b, 1, 0, count, C, 0, 0, 40_000, shell, vis, 5, idx.subarray(0, 2), d2.subarray(0, 2));
  assert.equal(n, 2);
  assert.deepEqual(Array.from(idx.subarray(0, 2)), [0, 5]);
});

test('selectCandidates: forced first, show radius for new ones, hysteresis for shown ones, max', () => {
  const foundIdx = new Int32Array([7, 3, 9, 4, 8]);
  const foundD2 = new Float64Array([1, 4, 9, 16, 25]); // distances 1..5
  const shown = new Uint8Array(16);
  const out = new Int32Array(8);
  // show radius 3 (d2 <= 9): 7, 3, 9
  let k = selectCandidates(foundIdx, foundD2, 5, shown, -1, 9, 8, out);
  assert.deepEqual(Array.from(out.subarray(0, k)), [7, 3, 9]);
  // 8 is already shown -> kept although beyond the show radius (within hide radius).
  shown[8] = 1;
  k = selectCandidates(foundIdx, foundD2, 5, shown, -1, 9, 8, out);
  assert.deepEqual(Array.from(out.subarray(0, k)), [7, 3, 9, 8]);
  // Forced goes first and is not duplicated.
  k = selectCandidates(foundIdx, foundD2, 5, shown, 9, 9, 8, out);
  assert.deepEqual(Array.from(out.subarray(0, k)), [9, 7, 3, 8]);
  // Forced outside the found list still included; max respected.
  k = selectCandidates(foundIdx, foundD2, 5, shown, 12, 9, 2, out);
  assert.deepEqual(Array.from(out.subarray(0, k)), [12, 7]);
});

// ---------------------------------------------------------------------------
// X-forward convention of the shipped .glb files (forwardAxis: Axis.X)
// ---------------------------------------------------------------------------

/** glTF -> model with forwardAxis X (Axis.Y_UP_TO_Z_UP only): (x, y, z) -> (x, -z, y). */
function gltfToModelXFwd(v: readonly number[]): [number, number, number] {
  return [v[0], -v[2], v[1]];
}

test("X-forward: glTF -> model mapping equals Cesium's Axis.Y_UP_TO_Z_UP alone", () => {
  const yUpToZUp = [1, 0, 0, 0, 0, 1, 0, -1, 0];
  for (const v of [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ]) {
    const r = [
      yUpToZUp[0] * v[0] + yUpToZUp[3] * v[1] + yUpToZUp[6] * v[2],
      yUpToZUp[1] * v[0] + yUpToZUp[4] * v[1] + yUpToZUp[7] * v[2],
      yUpToZUp[2] * v[0] + yUpToZUp[5] * v[1] + yUpToZUp[8] * v[2],
    ].map((c) => c + 0);
    assert.deepEqual(r, gltfToModelXFwd(v).map((c) => c + 0));
  }
});

test('X-forward: glTF +X flies forward, +Y is zenith, +Z (R wing) on the -orbit-normal side; sun mapping', () => {
  const m = new Float64Array(16);
  attitudeMatrix(6_928_137, 0, 0, 0, 7_600, 0, m);
  const apply = (g: ArrayLike<number>): number[] => {
    const v = gltfToModelXFwd([g[0], g[1], g[2]]);
    return [
      m[0] * v[0] + m[4] * v[1] + m[8] * v[2],
      m[1] * v[0] + m[5] * v[1] + m[9] * v[2],
      m[2] * v[0] + m[6] * v[1] + m[10] * v[2],
    ];
  };
  close(apply([1, 0, 0])[1], 1, EPS, 'glTF +X -> velocity (+Y ECEF)');
  close(apply([0, 1, 0])[0], 1, EPS, 'glTF +Y -> zenith (+X ECEF)');
  close(apply([0, 0, 1])[2], -1, EPS, 'glTF +Z -> -orbit normal');
  // sunInGltfBodyXForward is the inverse rotation: mapping the body vector back gives s.
  const sb = new Float64Array(3);
  for (const s0 of [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0.3, -0.5, 0.81],
  ]) {
    const n = norm(s0);
    const s = [s0[0] / n, s0[1] / n, s0[2] / n];
    sunInGltfBodyXForward(m, s[0], s[1], s[2], sb);
    const back = apply(sb);
    for (let k = 0; k < 3; k++) close(back[k], s[k], 1e-12, `sun ${s0}[${k}]`);
    // Wings about glTF +Z, cells +Y at rest: theta = atan2(-s.x, s.y) (public/models/README.md).
    const theta = wingSunAngle(sb, [0, 0, 1], [0, 1, 0]);
    close(theta, Math.atan2(-sb[0], sb[1]), 1e-12, 'README formula');
    const nG = [-Math.sin(theta), Math.cos(theta), 0];
    close(dot(nG, sb), Math.hypot(sb[0], sb[1]), 1e-9, 'cells face the sun');
  }
});

test('pickEvictableSlot: recycles an idle warmed-up instance of another file, never the last one of a file', () => {
  const slot = (file: string, sat = -1, warm = 0, model: unknown = {}) => ({ file, sat, warm, model });
  // Pool after a dense V2 Mini train: preloads + 10 v2mini, all idle now.
  const pool = [slot('v15'), slot('v2mini'), slot('v3', 5), ...Array.from({ length: 9 }, () => slot('v2mini'))];
  const k = pickEvictableSlot(pool, 'v3');
  assert.equal(pool[k]!.file, 'v2mini');
  assert.equal(k, pool.length - 1); // newest first
  // Busy, warming up or not-yet-created instances are never taken.
  const busy = [slot('v15'), slot('v2mini', 7), slot('v2mini', -1, 3), slot('v2mini', -1, 0, null), slot('v3', 1)];
  assert.equal(pickEvictableSlot(busy, 'v3'), -1);
  // A file's only instance (its warmed-up preload) is kept.
  assert.equal(pickEvictableSlot([slot('v15'), slot('v2mini'), slot('v3', 2)], 'v3'), -1);
  // Never the file that needs the room.
  assert.equal(pickEvictableSlot([slot('v3'), slot('v3'), slot('v2mini', 1)], 'v3'), -1);
  // The other way round: idle V3s make room for V2 Minis.
  const v3heavy = [slot('v15'), slot('v2mini', 4), slot('v3'), slot('v3', 9), slot('v3')];
  assert.equal(pickEvictableSlot(v3heavy, 'v2mini'), 4);
});
