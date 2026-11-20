#!/usr/bin/env node
/**
 * build-stars — converts d3-celestial's Hipparcos-based star list
 * (node_modules/d3-celestial/data/stars.8.json: GeoJSON points, coordinates
 * [RA° in -180..180, Dec°] J2000, properties.mag, properties.bv as string)
 * into public/sky/stars.bin for the GPU star field:
 *
 *   little-endian Float32 × 5 per star: x, y, z (unit vector, same sky frame
 *   as the cube faces: J2000 precessed to the mean equator/equinox of
 *   `epoch`), V magnitude, B-V; sorted bright -> faint.
 *
 * and merges a "stars" section (count, stride, bytes, epoch, hash) into
 * public/sky/manifest.json.
 *
 *   node scripts/build-stars.mjs [--epoch=2027] [--out=public/sky] [--maglimit=8]
 *
 * The epoch defaults to the cube's epoch from the manifest (so stars and
 * Milky Way always share one frame), else to skyMath.defaultEpoch().
 * Output is deterministic (idempotent).
 */
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { defaultEpoch, mulMat3, precessionMatrix, raDecToVec, STAR_STRIDE } from '../src/render/sky/skyMath.ts';

/** Colour index used when the catalogue has none (≈ G2V, solar). */
const DEFAULT_BV = 0.65;
const CREDIT = 'Stars: ESA Hipparcos catalogue via d3-celestial (BSD-3-Clause)';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const args = {};
for (const a of process.argv.slice(2)) {
  const m = /^--([^=]+)(?:=(.*))?$/.exec(a);
  if (m) args[m[1]] = m[2] ?? true;
}
const outDir = resolve(root, typeof args.out === 'string' ? args.out : 'public/sky');
const magLimit = args.maglimit ? Number(args.maglimit) : 8;
const manifestPath = join(outDir, 'manifest.json');
let manifest = {};
try {
  manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
} catch {
  /* first run */
}
const cubeEpoch = manifest.cube?.['2k']?.epoch ?? manifest.cube?.['4k']?.epoch;
const epoch = args.epoch ? Number(args.epoch) : (cubeEpoch ?? defaultEpoch());

const src = join(root, 'node_modules/d3-celestial/data/stars.8.json');
const geo = JSON.parse(readFileSync(src, 'utf8'));
const stars = [];
let missingBv = 0;
for (const f of geo.features) {
  const [ra180, dec] = f.geometry.coordinates;
  const mag = Number(f.properties.mag);
  if (!Number.isFinite(mag) || mag > magLimit) continue;
  let bv = Number.parseFloat(f.properties.bv);
  if (!Number.isFinite(bv)) {
    bv = DEFAULT_BV;
    missingBv++;
  }
  const ra = ra180 < 0 ? ra180 + 360 : ra180; // d3-celestial stores RA in -180..180
  stars.push({ id: f.id, ra, dec, mag, bv });
}
// bright -> faint; ties broken by id for determinism
stars.sort((a, b) => a.mag - b.mag || a.id - b.id);

const P = precessionMatrix(epoch);
const buf = new ArrayBuffer(stars.length * STAR_STRIDE * 4);
const dv = new DataView(buf);
const v = { x: 0, y: 0, z: 0 };
for (let i = 0; i < stars.length; i++) {
  const s = stars[i];
  mulMat3(P, raDecToVec(s.ra, s.dec, v), v);
  const o = i * STAR_STRIDE * 4;
  dv.setFloat32(o, v.x, true);
  dv.setFloat32(o + 4, v.y, true);
  dv.setFloat32(o + 8, v.z, true);
  dv.setFloat32(o + 12, s.mag, true);
  dv.setFloat32(o + 16, s.bv, true);
}
mkdirSync(outDir, { recursive: true });
const bytes = new Uint8Array(buf);
writeFileSync(join(outDir, 'stars.bin'), bytes);

// re-read right before writing to keep sections other scripts own
try {
  manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
} catch {
  manifest = {};
}
manifest.version = 1;
manifest.stars = {
  file: 'stars.bin',
  count: stars.length,
  stride: STAR_STRIDE,
  layout: 'float32le x,y,z,mag,bv',
  bytes: bytes.length,
  epoch,
  magLimit,
  magRange: [stars[0]?.mag ?? 0, stars[stars.length - 1]?.mag ?? 0],
  missingBv,
  hash: createHash('sha1').update(bytes).digest('hex').slice(0, 12),
  credit: CREDIT,
};
const order = ['version', 'credit', 'frame', 'cube', 'stars'];
const sorted = {};
for (const k of order) if (k in manifest) sorted[k] = manifest[k];
for (const k of Object.keys(manifest)) if (!(k in sorted)) sorted[k] = manifest[k];
writeFileSync(manifestPath, JSON.stringify(sorted, null, 2) + '\n');
console.log(
  `stars.bin: ${stars.length} stars (mag ${manifest.stars.magRange[0]}..${manifest.stars.magRange[1]}, ${missingBv} without B-V), ` +
    `${(bytes.length / 1024).toFixed(0)} KiB, epoch ${epoch}`,
);
