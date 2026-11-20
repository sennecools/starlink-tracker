/**
 * Pure math behind the coverage hologram (src/render/coverageMath.ts): cap
 * angle / ground radius for the 25° minimum elevation, footprint frame, the
 * cancellation-free cap point offset used by the vertex shader, and the
 * static meshes. No Cesium runtime.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  COVERAGE_LIFT_MAX_M,
  COVERAGE_LIFT_MIN_M,
  DEG2RAD,
  MEAN_EARTH_RADIUS_KM,
  WGS84_A_M,
  WGS84_B_M,
  buildConeMesh,
  buildDiscMesh,
  capPointOffset,
  coverageCapAngle,
  coverageConeHalfAngle,
  coverageGroundRadiusKm,
  coverageLiftM,
  coverageSlantRange,
  ellipsoidRadiusAlong,
  footprintBasis,
} from '../src/render/coverageMath.ts';
import { MIN_ELEVATION_DEG } from '../src/shared/phase2.ts';

const R = MEAN_EARTH_RADIUS_KM;

function close(actual: number, expected: number, tol: number, msg?: string): void {
  assert.ok(Math.abs(actual - expected) <= tol, `${msg ?? ''} expected ${expected} ± ${tol}, got ${actual}`);
}

test('550 km at 25° covers a ~940 km ground radius', () => {
  close(coverageGroundRadiusKm(550, MIN_ELEVATION_DEG), 940, 5);
  close(coverageCapAngle(R, R + 550, 25) / DEG2RAD, 8.46, 0.02, 'cap angle (deg)');
});

test('low V3 orbits have a much smaller footprint', () => {
  // 275 km (Starship insertion orbit): lambda = acos(R cos25 / (R + 275)) - 25° ≈ 4.68°.
  const r275 = coverageGroundRadiusKm(275, 25);
  close(r275, 521, 5);
  close(coverageGroundRadiusKm(316, 25), 589, 5);
  close(coverageGroundRadiusKm(480, 25), 841, 5);
  assert.ok(r275 < coverageGroundRadiusKm(550, 25));
});

test('cap angle degenerate cases', () => {
  assert.equal(coverageCapAngle(R, R, 25), 0, 'on the surface');
  assert.equal(coverageCapAngle(R, R - 10, 25), 0, 'below the surface');
  assert.equal(coverageCapAngle(R, R + 550, 90), 0, 'zenith only');
  assert.equal(coverageCapAngle(0, 100, 25), 0);
  assert.equal(coverageCapAngle(R, Number.NaN, 25), 0);
  assert.equal(coverageGroundRadiusKm(-5, 25), 0);
  // 0° elevation: the geometric horizon, acos(R / D).
  close(coverageCapAngle(R, R + 550, 0), Math.acos(R / (R + 550)), 1e-12);
  // Monotonic: lower minimum elevation -> larger cap.
  assert.ok(coverageCapAngle(R, R + 550, 10) > coverageCapAngle(R, R + 550, 25));
});

test('cap edge sees the satellite at exactly the minimum elevation (sphere)', () => {
  for (const h of [275, 550, 1200]) {
    const D = R + h;
    const lam = coverageCapAngle(R, D, 25);
    // Ground point at the cap edge and the satellite on +Z.
    const gx = R * Math.sin(lam);
    const gz = R * Math.cos(lam);
    const lx = -gx;
    const lz = D - gz; // line of sight
    const len = Math.hypot(lx, lz);
    const elev = Math.asin((lx * Math.sin(lam) + lz * Math.cos(lam)) / len) / DEG2RAD;
    close(elev, 25, 1e-9, `elevation at ${h} km`);
    close(coverageSlantRange(R, D, lam), len, 1e-9, 'slant range');
    // Cone half-angle + cap angle + elevation = 90°.
    close((coverageConeHalfAngle(lam, 25) + lam) / DEG2RAD + 25, 90, 1e-9);
  }
});

test('footprintBasis is a right-handed orthonormal up/east/north frame', () => {
  const b = new Float64Array(9);
  const cases: [number, number, number][] = [
    [7e6, 0, 0],
    [3e6, 4e6, 5e6],
    [-2e6, -6e6, -1e6],
    [0, 0, 6.9e6], // north pole (fallback east)
    [1, 0, -6.9e6], // near the south pole
  ];
  for (const [x, y, z] of cases) {
    assert.equal(footprintBasis(x, y, z, b), true);
    const up = [b[0], b[1], b[2]];
    const east = [b[3], b[4], b[5]];
    const north = [b[6], b[7], b[8]];
    const r = Math.hypot(x, y, z);
    close(up[0], x / r, 1e-12);
    close(up[1], y / r, 1e-12);
    close(up[2], z / r, 1e-12);
    for (const v of [up, east, north]) close(Math.hypot(v[0], v[1], v[2]), 1, 1e-12);
    const dot = (a: number[], c: number[]) => a[0] * c[0] + a[1] * c[1] + a[2] * c[2];
    close(dot(up, east), 0, 1e-12);
    close(dot(up, north), 0, 1e-12);
    close(dot(east, north), 0, 1e-12);
    // east × north = up
    close(east[1] * north[2] - east[2] * north[1], up[0], 1e-12);
    close(east[2] * north[0] - east[0] * north[2], up[1], 1e-12);
    close(east[0] * north[1] - east[1] * north[0], up[2], 1e-12);
    if (Math.abs(up[2]) < 0.99) {
      close(east[2], 0, 1e-12, 'east is horizontal (no z)');
      assert.ok(north[2] > 0, 'north points to +Z');
    }
  }
  assert.equal(footprintBasis(0, 0, 0, b), false);
  assert.equal(footprintBasis(Number.NaN, 0, 1, b), false);
});

test('capPointOffset matches a direct double-precision construction on WGS-84', () => {
  const b = new Float64Array(9);
  const out = [0, 0, 0];
  const lift = 2500;
  const sat: [number, number, number] = [4.1e6, -3.2e6, 4.4e6];
  footprintBasis(sat[0], sat[1], sat[2], b);
  const C = [b[0], b[1], b[2]];
  const rC = ellipsoidRadiusAlong(C[0], C[1], C[2]);
  const cap = coverageCapAngle(rC, Math.hypot(...sat), 25);
  for (const u of [0, 0.01, 0.3, 1, 1.08]) {
    for (const phi of [0, 1, 2.5, 4, 6]) {
      capPointOffset(b, cap, u, phi, lift, out);
      const a = u * cap;
      const w = [0, 1, 2].map((i) => Math.cos(phi) * b[3 + i] + Math.sin(phi) * b[6 + i]);
      const d = [0, 1, 2].map((i) => Math.cos(a) * C[i] + Math.sin(a) * w[i]);
      const rD = ellipsoidRadiusAlong(d[0], d[1], d[2]);
      for (let i = 0; i < 3; i++) {
        const expected = (rD + lift) * d[i] - (rC + lift) * C[i];
        close(out[i], expected, 1e-6, `u=${u} phi=${phi} axis ${i}`);
      }
      // The point is on the ellipsoid + lift along its geocentric direction.
      const P = [0, 1, 2].map((i) => (rC + lift) * C[i] + out[i]);
      const len = Math.hypot(P[0], P[1], P[2]);
      close(len - ellipsoidRadiusAlong(P[0] / len, P[1] / len, P[2] / len), lift, 1e-6);
    }
  }
  // u = 0 is the centre itself.
  capPointOffset(b, cap, 0, 1.2, lift, out);
  for (const v of out) close(v, 0, 1e-9);
});

test('cap point offsets are precise for tiny angles (no cancellation)', () => {
  const b = new Float64Array(9);
  footprintBasis(WGS84_A_M + 550e3, 0, 0, b);
  const out = [0, 0, 0];
  // 1e-9 of a 0.15 rad cap ≈ 1 mm on the ground.
  capPointOffset(b, 0.15, 1e-9, 0, 0, out);
  const arc = WGS84_A_M * 0.15e-9;
  close(Math.hypot(out[0], out[1], out[2]), arc, arc * 1e-6);
});

test('ellipsoid radius: equator a, pole b', () => {
  close(ellipsoidRadiusAlong(1, 0, 0), WGS84_A_M, 1e-6);
  close(ellipsoidRadiusAlong(0, 0, 1), WGS84_B_M, 1e-6);
  const s = Math.SQRT1_2;
  const r45 = ellipsoidRadiusAlong(s, 0, s);
  assert.ok(r45 < WGS84_A_M && r45 > WGS84_B_M);
});

test('disc mesh: rings include u = 1 exactly, valid indices, closed in phi', () => {
  const { vertices, indices } = buildDiscMesh(128, 0.02, 1.08);
  const cols = 129;
  const rings = vertices.length / 2 / cols;
  assert.equal(rings, 55);
  assert.ok(vertices.length / 2 < 65536, 'fits 16-bit indices');
  const us = new Set<number>();
  for (let i = 0; i < vertices.length; i += 2) us.add(Math.round(vertices[i] * 1e5) / 1e5);
  assert.ok(us.has(1), 'a ring sits exactly on the rim');
  assert.ok(us.has(0) && us.has(1.08));
  let max = 0;
  for (const i of indices) if (i > max) max = i;
  assert.ok(max < vertices.length / 2);
  assert.equal(indices.length, (rings - 1) * 128 * 6);
  close(vertices[(cols - 1) * 2 + 1], Math.PI * 2, 1e-6, 'seam column at 2π');
});

test('cone mesh: apex + rim triangles cover the full circle', () => {
  const v = buildConeMesh(96);
  assert.equal(v.length, 96 * 3 * 2);
  for (let t = 0; t < 96; t++) {
    const o = t * 6;
    assert.equal(v[o], 0, 'apex s = 0');
    assert.equal(v[o + 2], 1);
    assert.equal(v[o + 4], 1);
    close(v[o + 5] - v[o + 3], (Math.PI * 2) / 96, 1e-6);
  }
  close(v[v.length - 1], Math.PI * 2, 1e-5);
});

test('coverage lift: 2.5 km from 10 km camera height up, always below a low camera', () => {
  assert.equal(COVERAGE_LIFT_MAX_M, 2_500);
  assert.equal(coverageLiftM(10_000), COVERAGE_LIFT_MAX_M);
  assert.equal(coverageLiftM(500_000), COVERAGE_LIFT_MAX_M);
  assert.equal(coverageLiftM(20_000_000), COVERAGE_LIFT_MAX_M);
  // Camera at 1 km (zoomed to a town): the layer sits 250 m up, below the camera.
  assert.equal(coverageLiftM(1_000), 250);
  for (const h of [250, 400, 1_000, 2_500, 5_000, 9_999]) assert.ok(coverageLiftM(h) < h, `lift below the camera at ${h} m`);
  // Floor (CameraRig minimum is 250 m above terrain; below the ellipsoid over deep valleys).
  assert.equal(coverageLiftM(50), COVERAGE_LIFT_MIN_M);
  assert.equal(coverageLiftM(-100), COVERAGE_LIFT_MIN_M);
  assert.equal(coverageLiftM(Number.NaN), COVERAGE_LIFT_MAX_M);
});
