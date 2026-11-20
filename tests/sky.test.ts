import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  CUBE_FACES,
  GALACTIC_CENTER_J2000,
  GALACTIC_NORTH_POLE_J2000,
  POLARIS_J2000,
  STAR_STRIDE,
  angleBetweenDeg,
  blackbodyRgb,
  bvToTemperature,
  defaultEpoch,
  dirToFaceUv,
  equirectToRaDec,
  faceTexelToDir,
  faceUvToDir,
  j2000ToSkyVec,
  julianYearToJd,
  mulMat3,
  mulMat3T,
  precessionMatrix,
  raDecToEquirect,
  raDecToVec,
  starDisplayColor,
  vecToRaDec,
} from '../src/render/sky/skyMath.ts';
import { STAR_VERTEX_FLOATS, buildStarVertices, decodeStarsBin, deviceAllows4k } from '../src/render/sky/skyAssets.ts';
import type { SkyManifest } from '../src/render/sky/skyAssets.ts';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const SKY = join(ROOT, 'public', 'sky');

/** Deterministic PRNG (mulberry32). */
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randomUnit(r: () => number): { x: number; y: number; z: number } {
  const z = 2 * r() - 1;
  const phi = 2 * Math.PI * r();
  const s = Math.sqrt(1 - z * z);
  return { x: s * Math.cos(phi), y: s * Math.sin(phi), z };
}

// ---------------------------------------------------------------------------
// RA/Dec <-> vector
// ---------------------------------------------------------------------------

test('raDecToVec / vecToRaDec round-trip', () => {
  const r = rng(1);
  for (let i = 0; i < 2000; i++) {
    const ra = r() * 360;
    const dec = Math.asin(2 * r() - 1) * (180 / Math.PI);
    const v = raDecToVec(ra, dec);
    assert.ok(Math.abs(Math.hypot(v.x, v.y, v.z) - 1) < 1e-12);
    const back = vecToRaDec(v.x, v.y, v.z);
    assert.ok(Math.abs(back.decDeg - dec) < 1e-9, 'dec');
    if (Math.abs(dec) < 89.999) {
      let d = Math.abs(back.raDeg - ra);
      d = Math.min(d, 360 - d);
      assert.ok(d < 1e-8, `ra ${ra} -> ${back.raDeg}`);
    }
  }
  const x = raDecToVec(0, 0);
  assert.ok(Math.abs(x.x - 1) < 1e-15 && Math.abs(x.y) < 1e-15);
  const y = raDecToVec(90, 0);
  assert.ok(Math.abs(y.y - 1) < 1e-15);
  const z = raDecToVec(123, 90);
  assert.ok(Math.abs(z.z - 1) < 1e-15);
});

// ---------------------------------------------------------------------------
// Precession
// ---------------------------------------------------------------------------

test('precession matrix is a rotation and identity at J2000', () => {
  const I = precessionMatrix(2000);
  for (let i = 0; i < 9; i++) assert.ok(Math.abs(I[i] - (i % 4 === 0 ? 1 : 0)) < 1e-15);
  const P = precessionMatrix(2027);
  // P Pᵀ = I
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      let s = 0;
      for (let k = 0; k < 3; k++) s += P[r * 3 + k] * P[c * 3 + k];
      assert.ok(Math.abs(s - (r === c ? 1 : 0)) < 1e-14);
    }
  }
  const det =
    P[0] * (P[4] * P[8] - P[5] * P[7]) - P[1] * (P[3] * P[8] - P[5] * P[6]) + P[2] * (P[3] * P[7] - P[4] * P[6]);
  assert.ok(Math.abs(det - 1) < 1e-14);
  // forward / inverse round-trip
  const r = rng(2);
  for (let i = 0; i < 200; i++) {
    const v = randomUnit(r);
    const w = mulMat3T(P, mulMat3(P, v, { x: 0, y: 0, z: 0 }), { x: 0, y: 0, z: 0 });
    assert.ok(Math.hypot(w.x - v.x, w.y - v.y, w.z - v.z) < 1e-14);
  }
});

test('precession matches Meeus example 21.b (θ Persei to 2028 Nov 13.19)', () => {
  // J2000 position with the proper motion over t = 28.86705 yr already applied
  const ra0 = 41.04994 + (0.03425 * 15 * 28.86705) / 3600;
  const dec0 = 49.22847 - (0.0895 * 28.86705) / 3600;
  const epoch = 2000 + (2462088.69 - 2451545.0) / 365.25;
  assert.ok(Math.abs(julianYearToJd(epoch) - 2462088.69) < 1e-6);
  const v = j2000ToSkyVec(ra0, dec0, epoch);
  const { raDeg, decDeg } = vecToRaDec(v.x, v.y, v.z);
  // Meeus: α = 2h46m11.331s = 41.547213°, δ = +49°20'54.54" = 49.348483°
  assert.ok(Math.abs(raDeg - 41.547213) < 3e-5, `ra ${raDeg}`);
  assert.ok(Math.abs(decDeg - 49.348483) < 3e-5, `dec ${decDeg}`);
});

test('precession J2000 -> 2027: Polaris moves north, ecliptic pole stays put, ~0.37° total', () => {
  const p = j2000ToSkyVec(POLARIS_J2000.raDeg, POLARIS_J2000.decDeg, 2027);
  const pd = vecToRaDec(p.x, p.y, p.z).decDeg;
  assert.ok(pd > 89.37 && pd < 89.39, `Polaris dec ${pd}`);
  // ecliptic north pole (RA 18h, Dec 90° - ε) is (almost) invariant under precession
  const ep = raDecToVec(270, 66.560708);
  const ep27 = j2000ToSkyVec(270, 66.560708, 2027);
  assert.ok(angleBetweenDeg(ep, ep27) < 0.01);
  // a point on the ecliptic moves by the general precession ≈ 50.29"/yr * 27 yr ≈ 0.377°
  const eq = raDecToVec(0, 0);
  const eq27 = j2000ToSkyVec(0, 0, 2027);
  const moved = angleBetweenDeg(eq, eq27);
  assert.ok(moved > 0.37 && moved < 0.385, `moved ${moved}`);
});

test('defaultEpoch rounds to the nearest Julian year', () => {
  assert.equal(defaultEpoch(new Date('2026-11-20T00:00:00Z')), 2027);
  assert.equal(defaultEpoch(new Date('2026-03-01T00:00:00Z')), 2026);
});

// ---------------------------------------------------------------------------
// Cube face addressing (as used by build-sky)
// ---------------------------------------------------------------------------

test('face centres and the Cesium face orientation', () => {
  const out = { x: 0, y: 0, z: 0 };
  const centres = [
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
  ];
  for (let f = 0; f < 6; f++) {
    faceUvToDir(f, 0.5, 0.5, out);
    assert.deepEqual([out.x, out.y, out.z].map((c) => c + 0), centres[f], CUBE_FACES[f]);
  }
  // GL table 3.21 + Cesium's single vertical flip: top-centre of +X looks along -Y,
  // right-centre of +X along -Z; top-centre of +Z looks along -Y.
  faceUvToDir(0, 0.5, 0, out);
  assert.deepEqual([out.x, out.y + 0, out.z + 0], [1, -1, 0]);
  faceUvToDir(0, 1, 0.5, out);
  assert.deepEqual([out.x, out.y + 0, out.z + 0], [1, 0, -1]);
  faceUvToDir(4, 0.5, 0, out);
  assert.deepEqual([out.x + 0, out.y, out.z], [0, -1, 1]);
});

test('faceUvToDir / dirToFaceUv round-trip and texel addressing', () => {
  const r = rng(3);
  const d = { x: 0, y: 0, z: 0 };
  const fu = { face: 0, s: 0, v: 0 };
  for (let i = 0; i < 5000; i++) {
    const v = randomUnit(r);
    dirToFaceUv(v.x, v.y, v.z, fu);
    assert.ok(fu.s >= 0 && fu.s <= 1 && fu.v >= 0 && fu.v <= 1);
    faceUvToDir(fu.face, fu.s, fu.v, d);
    const n = Math.hypot(d.x, d.y, d.z);
    assert.ok(Math.hypot(d.x / n - v.x, d.y / n - v.y, d.z / n - v.z) < 1e-12);
  }
  for (let f = 0; f < 6; f++) {
    for (let k = 0; k < 50; k++) {
      const s = r(), v = r();
      faceUvToDir(f, s, v, d);
      dirToFaceUv(d.x, d.y, d.z, fu);
      assert.equal(fu.face, f);
      assert.ok(Math.abs(fu.s - s) < 1e-12 && Math.abs(fu.v - v) < 1e-12);
    }
    // texel centres map back into their own texel
    const N = 64;
    for (let k = 0; k < 40; k++) {
      const col = Math.floor(r() * N), row = Math.floor(r() * N);
      faceTexelToDir(f, col, row, N, d);
      assert.ok(Math.abs(Math.hypot(d.x, d.y, d.z) - 1) < 1e-12);
      dirToFaceUv(d.x, d.y, d.z, fu);
      assert.equal(fu.face, f);
      assert.equal(Math.floor(fu.s * N), col);
      assert.equal(Math.floor(fu.v * N), row);
    }
  }
});

test('NASA equirect convention: RA 0 at centre, increasing to the left', () => {
  assert.ok(Math.abs(raDecToEquirect(0, 0).u - 0.5) < 1e-12);
  assert.ok(Math.abs(raDecToEquirect(80.9, -69.75).u - 0.2753) < 1e-3); // LMC
  assert.ok(Math.abs(raDecToEquirect(13.2, -72.8).u - 0.4633) < 1e-3); // SMC
  const gc = raDecToEquirect(GALACTIC_CENTER_J2000.raDeg, GALACTIC_CENTER_J2000.decDeg);
  assert.ok(Math.abs(gc.u - 0.76) < 0.01 && Math.abs(gc.v - 0.66) < 0.01);
  const r = rng(4);
  for (let i = 0; i < 500; i++) {
    const ra = r() * 360, dec = r() * 180 - 90;
    const e = raDecToEquirect(ra, dec);
    assert.ok(e.u >= 0 && e.u < 1);
    const b = equirectToRaDec(e.u, e.v);
    let d = Math.abs(b.raDeg - ra);
    d = Math.min(d, 360 - d);
    assert.ok(d < 1e-9 && Math.abs(b.decDeg - dec) < 1e-9);
  }
});

// ---------------------------------------------------------------------------
// Star colours
// ---------------------------------------------------------------------------

test('B-V colour: temperatures and blackbody chromaticity', () => {
  assert.ok(Math.abs(bvToTemperature(0.65) - 5800) < 150, 'sun-like');
  assert.ok(bvToTemperature(-0.3) > 15000);
  assert.ok(bvToTemperature(1.6) < 3700);
  const hot = blackbodyRgb(20000);
  const cool = blackbodyRgb(3200);
  const d65 = blackbodyRgb(6504);
  assert.ok(hot[2] === 1 && hot[0] < 0.8, `hot ${Array.from(hot)}`);
  assert.ok(cool[0] === 1 && cool[2] < 0.5, `cool ${Array.from(cool)}`);
  assert.ok(Math.min(d65[0], d65[1], d65[2]) > 0.85, `6500K ~ white ${Array.from(d65)}`);
  const grey = starDisplayColor(1.5, 0);
  assert.deepEqual(Array.from(grey), [1, 1, 1]);
  const c = starDisplayColor(-0.2, 0.55);
  assert.ok(c[2] === 1 && c[0] < 1 && c[0] > 0.6);
});

test('deviceAllows4k policy', () => {
  assert.equal(deviceAllows4k(16384, undefined), true);
  assert.equal(deviceAllows4k(2048, undefined), false);
  assert.equal(deviceAllows4k(16384, { deviceMemory: 4 }), false);
  assert.equal(deviceAllows4k(16384, { deviceMemory: 8 }), true);
  assert.equal(deviceAllows4k(16384, { connection: { saveData: true } }), false);
  assert.equal(deviceAllows4k(16384, { connection: { effectiveType: '3g' } }), false);
  assert.equal(deviceAllows4k(16384, { connection: { effectiveType: '4g' } }), true);
  // phones / tablets: no memory hint (iOS Safari, Android Firefox) or capped at 8 (Android Chrome)
  assert.equal(deviceAllows4k(16384, { userAgentData: { mobile: true }, deviceMemory: 8 }), false);
  assert.equal(deviceAllows4k(16384, { maxTouchPoints: 5 }, true), false);
  // touch laptop with a mouse/trackpad as the primary pointer, and plain desktops
  assert.equal(deviceAllows4k(16384, { maxTouchPoints: 10, userAgentData: { mobile: false } }, false), true);
  assert.equal(deviceAllows4k(16384, { maxTouchPoints: 0 }, true), true);
});

// ---------------------------------------------------------------------------
// Generated assets
// ---------------------------------------------------------------------------

const haveAssets = existsSync(join(SKY, 'manifest.json'));

function jpegSize(buf: Uint8Array): { width: number; height: number } {
  assert.equal(buf[0], 0xff);
  assert.equal(buf[1], 0xd8);
  let i = 2;
  while (i < buf.length) {
    if (buf[i] !== 0xff) throw new Error('bad JPEG marker');
    const marker = buf[i + 1];
    const len = (buf[i + 2] << 8) | buf[i + 3];
    if (marker >= 0xc0 && marker <= 0xc3) {
      return { height: (buf[i + 5] << 8) | buf[i + 6], width: (buf[i + 7] << 8) | buf[i + 8] };
    }
    i += 2 + len;
  }
  throw new Error('no SOF marker');
}

test('manifest and cube faces', { skip: !haveAssets && 'public/sky not built' }, () => {
  const m = JSON.parse(readFileSync(join(SKY, 'manifest.json'), 'utf8')) as SkyManifest;
  assert.equal(m.version, 1);
  assert.match(m.credit ?? '', /NASA\/Goddard Space Flight Center Scientific Visualization Studio/);
  const c2 = m.cube['2k'];
  assert.ok(c2, '2k cube present');
  for (const [q, size] of [['2k', 2048], ['4k', 4096]] as const) {
    const e = m.cube[q];
    if (!e) continue;
    assert.equal(e.size, size);
    assert.deepEqual(e.faces, [...CUBE_FACES]);
    let total = 0;
    for (const f of CUBE_FACES) {
      const buf: Buffer = readFileSync(join(SKY, e.path, `${f}.${e.ext}`));
      total += buf.length;
      assert.deepEqual(jpegSize(buf), { width: size, height: size }, `${q}/${f}`);
    }
    assert.equal(total, e.bytes);
    assert.equal(e.epoch, c2.epoch);
  }
  assert.equal(m.stars?.epoch, c2.epoch, 'stars and cube share one sky frame');
});

test('stars.bin: count, order, magnitudes, unit vectors', { skip: !haveAssets && 'public/sky not built' }, () => {
  const m = JSON.parse(readFileSync(join(SKY, 'manifest.json'), 'utf8')) as SkyManifest;
  const entry = m.stars;
  assert.ok(entry);
  const raw = readFileSync(join(SKY, entry.file));
  const data = decodeStarsBin(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength), entry);
  const n = data.length / STAR_STRIDE;
  assert.equal(n, entry.count);
  assert.ok(n > 41000 && n < 42000, `count ${n}`);
  let prev = -Infinity;
  for (let i = 0; i < n; i++) {
    const o = i * STAR_STRIDE;
    const len = Math.hypot(data[o], data[o + 1], data[o + 2]);
    assert.ok(Math.abs(len - 1) < 2e-6, `star ${i} |v| = ${len}`);
    const mag = data[o + 3];
    assert.ok(mag >= prev, 'sorted bright -> faint');
    assert.ok(mag >= -1.5 && mag <= 8.0001);
    assert.ok(Number.isFinite(data[o + 4]) && data[o + 4] > -0.6 && data[o + 4] < 6) // a few carbon stars reach B-V 5.5;
    prev = mag;
  }
  // Sirius first, at its precessed position
  assert.ok(Math.abs(data[3] + 1.44) < 0.01);
  const sirius = j2000ToSkyVec(101.2872, -16.7161, entry.epoch);
  assert.ok(angleBetweenDeg({ x: data[0], y: data[1], z: data[2] }, sirius) < 1e-3);
  // vertices for the GPU
  const verts = buildStarVertices(data.subarray(0, 100 * STAR_STRIDE), 0.55);
  assert.equal(verts.length, 100 * STAR_VERTEX_FLOATS);
  assert.equal(verts[6], data[3]);
  assert.throws(() => decodeStarsBin(new ArrayBuffer(20), entry));
});

test('cube orientation: galactic centre bright, galactic pole dark', { skip: !haveAssets && 'public/sky not built' }, async () => {
  let sharp: typeof import('sharp').default;
  try {
    sharp = (await import('sharp')).default;
  } catch {
    return; // native module unavailable on this machine
  }
  const m = JSON.parse(readFileSync(join(SKY, 'manifest.json'), 'utf8')) as SkyManifest;
  const e = m.cube['2k'];
  assert.ok(e);
  const faces: Buffer[] = [];
  for (const f of CUBE_FACES) faces.push(await sharp(join(SKY, e.path, `${f}.${e.ext}`)).greyscale().raw().toBuffer());
  const N = e.size;
  const fu = { face: 0, s: 0, v: 0 };
  const lumAround = (raDeg: number, decDeg: number, radiusDeg: number): number => {
    let sum = 0, cnt = 0;
    for (let dy = -radiusDeg; dy <= radiusDeg; dy += 0.25) {
      for (let dx = -radiusDeg; dx <= radiusDeg; dx += 0.25) {
        if (dx * dx + dy * dy > radiusDeg * radiusDeg) continue;
        const dec = decDeg + dy;
        const v = j2000ToSkyVec(raDeg + dx / Math.cos((dec * Math.PI) / 180), dec, e.epoch);
        dirToFaceUv(v.x, v.y, v.z, fu);
        const col = Math.min(N - 1, Math.floor(fu.s * N)), row = Math.min(N - 1, Math.floor(fu.v * N));
        sum += faces[fu.face][row * N + col];
        cnt++;
      }
    }
    return sum / cnt;
  };
  const gc = lumAround(GALACTIC_CENTER_J2000.raDeg + 3, GALACTIC_CENTER_J2000.decDeg + 2, 3); // just off the dust lane
  const ngp = lumAround(GALACTIC_NORTH_POLE_J2000.raDeg, GALACTIC_NORTH_POLE_J2000.decDeg, 5);
  const mirrored = lumAround(360 - GALACTIC_CENTER_J2000.raDeg - 3, GALACTIC_CENTER_J2000.decDeg + 2, 3);
  assert.ok(gc > 60, `galactic centre region ${gc.toFixed(1)}`);
  assert.ok(ngp < 3, `galactic north pole ${ngp.toFixed(1)}`);
  assert.ok(gc > 5 * mirrored, `mirror check ${gc.toFixed(1)} vs ${mirrored.toFixed(1)}`);
});
