#!/usr/bin/env node
/**
 * build-sky — bakes the NASA SVS "Deep Star Maps 2020" Milky Way (OpenEXR,
 * half float, plate carrée, ICRF/J2000) into the six JPEG faces of a Cesium
 * sky-box cube map, precessed to the mean equator/equinox of the baking epoch.
 *
 *   node scripts/build-sky.mjs --src8k=<milkyway_2020_8k.exr> [--src16k=<milkyway_2020_16k.exr>]
 *                              [--only=2k|4k] [--epoch=2027] [--force] [--out=public/sky]
 *
 * Paths may also come from SKY_SRC_8K / SKY_SRC_16K. Outputs:
 *   public/sky/2k/{px,nx,py,ny,pz,nz}.jpg   2048² faces from the 8k map
 *   public/sky/4k/{px,nx,py,ny,pz,nz}.jpg   4096² faces from the 16k map
 *   public/sky/manifest.json                 (merged; the "stars" section belongs to build-stars)
 *
 * Idempotent: a quality is skipped when its input key (source size/mtime,
 * tone-map constants, epoch, script version) matches the manifest and all
 * faces exist. --force rebuilds.
 *
 * Pipeline per output texel: centre direction (sky frame of date) -> inverse
 * precession -> J2000 RA/Dec -> equirect position; area filter = a grid of
 * bilinear taps spanning the texel's footprint on the source (the grid widens
 * in RA by 1/cos(dec), so the poles are properly averaged), in linear light;
 * then the tone map below; sRGB encode; ±0.5 LSB triangular dither on
 * non-black texels (no banding in the faint glow); mozjpeg.
 *
 * Faces are computed by 6 worker_threads sharing the decoded map through a
 * SharedArrayBuffer (RGB half floats, ~805 MB for the 16k map).
 *
 * Credit: NASA/Goddard Space Flight Center Scientific Visualization Studio;
 * Gaia DR2: ESA/Gaia/DPAC.
 */
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { availableParallelism } from 'node:os';
import { basename, dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads';
import {
  CUBE_FACES,
  defaultEpoch,
  faceUvToDir,
  linearToSrgb,
  mulMat3T,
  precessionMatrix,
} from '../src/render/sky/skyMath.ts';

// ---------------------------------------------------------------------------
// Tone map ("deep space": black sky, restrained Milky Way, neutral colour).
// All luminances are in the EXR's linear units (high-latitude sky ≈ 2.7e-3,
// Milky Way band ≈ 1–3e-2, galactic centre ≈ 0.1).
// ---------------------------------------------------------------------------
export const TONE = Object.freeze({
  /**
   * Mean source luminance the constants below were tuned for (the 8k map).
   * Every map is first scaled to this mean: the 16k file stores ~1/4 of the
   * 8k values per pixel (same energy spread over 4x the pixels).
   */
  referenceMeanLum: 0.006714,
  /** Partial white balance towards the map's mean colour (0 = none, 1 = fully neutral mean). */
  whiteBalanceStrength: 0.7,
  /** Map mean linear RGB (measured from the 8k map), used for the white balance. */
  sourceMean: [0.007422, 0.006582, 0.005864],
  /** Luminance treated as pure black (noise floor + unresolved high-latitude glow). */
  blackPoint: 0.0042,
  /** Softness of the black-point knee: f(x) = x² / (x + toe), x = L - blackPoint. */
  toe: 0.006,
  /** Linear gain applied after the black point. */
  exposure: 4.2,
  /** Asymptotic maximum display luminance (linear) — keeps the core from blowing out. */
  ceiling: 0.34,
  /** Chroma kept (0 = greyscale, 1 = source colours). */
  saturation: 0.55,
  /** Dither amplitude in 8-bit LSB (triangular), only where the output is at least ditherMin LSB. */
  dither: 1.0,
  ditherMin: 3,
  /** Area-filter footprint in output texels (>1 softens the source's per-pixel grain). */
  filterWidth: 1.7,
  /** Share of a pixel's excess over its 3x3 luminance median that is kept (point-source suppression). */
  pointSourceKeep: 0.1,
});

const JPEG = Object.freeze({ quality: 90, mozjpeg: true, progressive: true, chromaSubsampling: '4:2:0' });
const SCRIPT_VERSION = 6;
const CREDIT = 'NASA/Goddard Space Flight Center Scientific Visualization Studio; Gaia DR2: ESA/Gaia/DPAC';
const QUALITIES = {
  // matchGain: extra input gain so the 4k cube (from the spikier 16k map, which loses more
  // energy at the black point) has the same overall brightness as the 2k cube; the swap
  // 2k -> 4k must be invisible except for sharpness.
  '2k': { size: 2048, srcKey: 'src8k', envKey: 'SKY_SRC_8K', matchGain: 1 },
  '4k': { size: 4096, srcKey: 'src16k', envKey: 'SKY_SRC_16K', matchGain: 1.22 },
};

// ---------------------------------------------------------------------------
// Half float LUT
// ---------------------------------------------------------------------------
function makeHalfLut() {
  const lut = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h >> 15, e = (h >> 10) & 31, m = h & 1023;
    let v;
    if (e === 0) v = m * 2 ** -24;
    else if (e === 31) v = 0; // inf / NaN -> 0 (none expected)
    else v = (1 + m / 1024) * 2 ** (e - 15);
    lut[h] = s ? 0 : v; // negative radiance -> 0
  }
  return lut;
}

// ---------------------------------------------------------------------------
// Worker: one cube face
// ---------------------------------------------------------------------------
function renderFace({ sab, W, H, face, size, epoch, tone, seed, gain }) {
  const src = new Uint16Array(sab);
  const lut = makeHalfLut();
  const P = precessionMatrix(epoch);
  const out = new Uint8Array(size * size * 3);
  const d = { x: 0, y: 0, z: 0 };
  const wbR = Math.pow(tone.sourceMean[1] / tone.sourceMean[0], tone.whiteBalanceStrength);
  const wbB = Math.pow(tone.sourceMean[1] / tone.sourceMean[2], tone.whiteBalanceStrength);
  const invPi = 1 / Math.PI;
  const pxPerRadV = H * invPi;
  const pxPerRadU = W * 0.5 * invPi;
  let rng = seed >>> 0 || 1;
  const rand = () => {
    rng ^= rng << 13; rng >>>= 0;
    rng ^= rng >>> 17;
    rng ^= rng << 5; rng >>>= 0;
    return rng / 4294967296;
  };

  // bilinear sample (x, y in source pixel units, pixel centres at +0.5) accumulating into acc[]
  const acc = new Float64Array(3);
  function tap(x, y) {
    x -= 0.5; y -= 0.5;
    let x0 = Math.floor(x), y0 = Math.floor(y);
    const fx = x - x0, fy = y - y0;
    let y1 = y0 + 1;
    if (y0 < 0) y0 = 0;
    if (y1 < 0) y1 = 0;
    if (y0 > H - 1) y0 = H - 1;
    if (y1 > H - 1) y1 = H - 1;
    x0 %= W; if (x0 < 0) x0 += W;
    let x1 = x0 + 1; if (x1 >= W) x1 = 0;
    const w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy), w01 = (1 - fx) * fy, w11 = fx * fy;
    const i00 = (y0 * W + x0) * 3, i10 = (y0 * W + x1) * 3, i01 = (y1 * W + x0) * 3, i11 = (y1 * W + x1) * 3;
    acc[0] += w00 * lut[src[i00]] + w10 * lut[src[i10]] + w01 * lut[src[i01]] + w11 * lut[src[i11]];
    acc[1] += w00 * lut[src[i00 + 1]] + w10 * lut[src[i10 + 1]] + w01 * lut[src[i01 + 1]] + w11 * lut[src[i11 + 1]];
    acc[2] += w00 * lut[src[i00 + 2]] + w10 * lut[src[i10 + 2]] + w01 * lut[src[i01 + 2]] + w11 * lut[src[i11 + 2]];
  }

  const texel = 2 / size; // face-plane units per texel
  for (let row = 0; row < size; row++) {
    const v = (row + 0.5) / size;
    for (let col = 0; col < size; col++) {
      const s = (col + 0.5) / size;
      faceUvToDir(face, s, v, d);
      const a = 2 * s - 1, b = 1 - 2 * v;
      const r2 = 1 + a * a + b * b;
      // angular texel size (radians), larger of the two face axes
      const ang = (tone.filterWidth * texel * Math.sqrt(1 + Math.min(a * a, b * b))) / r2;
      const inv = 1 / Math.sqrt(r2);
      d.x *= inv; d.y *= inv; d.z *= inv;
      mulMat3T(P, d, d); // sky frame of date -> J2000
      const dec = Math.asin(Math.max(-1, Math.min(1, d.z)));
      let ra = Math.atan2(d.y, d.x);
      // NASA map: u = 0.5 - ra/2π (RA 0 at centre, increasing to the left), v = (π/2 - dec)/π
      let u = 0.5 - ra / (2 * Math.PI);
      u -= Math.floor(u);
      const cx = u * W;
      const cy = (Math.PI / 2 - dec) * pxPerRadV;
      const hv = ang * pxPerRadV;
      const cosd = Math.max(Math.cos(dec), 1e-4);
      const hu = Math.min(W, (ang * pxPerRadU) / cosd);
      const nv = Math.min(64, Math.max(2, Math.ceil(hv * 1.25)));
      const nu = Math.min(1024, Math.max(2, Math.ceil(hu * 1.25)));
      acc[0] = 0; acc[1] = 0; acc[2] = 0;
      for (let j = 0; j < nv; j++) {
        const y = cy + ((j + 0.5) / nv - 0.5) * hv;
        for (let i = 0; i < nu; i++) tap(cx + ((i + 0.5) / nu - 0.5) * hu, y);
      }
      const n = gain / (nu * nv);
      let r = acc[0] * n * wbR, g = acc[1] * n, bl = acc[2] * n * wbB;

      // --- tone map ---
      const L = 0.2126 * r + 0.7152 * g + 0.0722 * bl;
      const o = (row * size + col) * 3;
      if (!(L > tone.blackPoint)) { out[o] = 0; out[o + 1] = 0; out[o + 2] = 0; continue; }
      let x = L - tone.blackPoint;
      x = (x * x) / (x + tone.toe);
      const Yd = tone.ceiling * (1 - Math.exp((-tone.exposure * x) / tone.ceiling));
      const k = Yd / L;
      const sat = tone.saturation;
      r = (L + (r - L) * sat) * k;
      g = (L + (g - L) * sat) * k;
      bl = (L + (bl - L) * sat) * k;
      const m = Math.max(r, g, bl);
      if (m > 1) { r /= m; g /= m; bl /= m; }
      const sr = linearToSrgb(r) * 255, sg = linearToSrgb(g) * 255, sb = linearToSrgb(bl) * 255;
      const dith = Math.max(sr, sg, sb) >= tone.ditherMin ? tone.dither * (rand() - rand()) : 0;
      out[o] = q8(sr + dith);
      out[o + 1] = q8(sg + dith);
      out[o + 2] = q8(sb + dith);
    }
  }
  return out;
}

/** Median of 9 (Paeth/Devillard sorting network); permutes `p`. */
function median9(p) {
  let t;
  const s = (a, b) => {
    if (p[a] > p[b]) { t = p[a]; p[a] = p[b]; p[b] = t; }
  };
  s(1, 2); s(4, 5); s(7, 8); s(0, 1); s(3, 4); s(6, 7); s(1, 2); s(4, 5); s(7, 8);
  s(0, 3); s(5, 8); s(4, 7); s(3, 6); s(1, 4); s(2, 5); s(4, 7); s(4, 2); s(6, 4); s(4, 2);
  return p[4];
}

const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);
/** float -> IEEE half (round to nearest), non-negative finite inputs. */
function toHalf(v) {
  f32[0] = v;
  const x = u32[0];
  const e = ((x >>> 23) & 0xff) - 127 + 15;
  const m = x & 0x7fffff;
  if (e <= 0) {
    if (e < -10) return 0;
    const mm = (m | 0x800000) >> (1 - e);
    return (mm + 0x1000) >> 13;
  }
  if (e >= 31) return 0x7bff;
  const h = (e << 10) | (m >> 13);
  return h + ((m >> 12) & 1); // carry into the exponent is correct rounding
}

function q8(v) {
  const r = Math.round(v);
  return r < 0 ? 0 : r > 255 ? 255 : r;
}

if (!isMainThread) {
  const res = renderFace(workerData);
  parentPort.postMessage(res, [res.buffer]);
} else {
  await main();
}

// ---------------------------------------------------------------------------
// Main thread
// ---------------------------------------------------------------------------
function parseArgs(argv) {
  const args = {};
  for (const a of argv) {
    const m = /^--([^=]+)(?:=(.*))?$/.exec(a);
    if (m) args[m[1]] = m[2] ?? true;
  }
  return args;
}

function readManifest(path) {
  try {
    return JSON.parse(readFileSync(path, 'utf8'));
  } catch {
    return {};
  }
}

async function decodeExr(path) {
  const { HalfFloatType } = await import('three');
  const { EXRLoader } = await import('three/examples/jsm/loaders/EXRLoader.js');
  const t0 = performance.now();
  const buf = readFileSync(path);
  const ab =
    buf.byteOffset === 0 && buf.byteLength === buf.buffer.byteLength
      ? buf.buffer
      : buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  const loader = new EXRLoader();
  loader.setDataType(HalfFloatType);
  const tex = loader.parse(ab);
  const { width: W, height: H, data } = tex;
  const C = data.length / (W * H);
  if (!(C === 4 || C === 3)) throw new Error(`unexpected channel count ${C}`);
  const lut = makeHalfLut();
  // mean luminance (every 61st pixel) for the exposure normalization
  let sum = 0;
  let n = 0;
  for (let i = 0; i < W * H; i += 61, n++) {
    sum += 0.2126 * lut[data[i * C]] + 0.7152 * lut[data[i * C + 1]] + 0.0722 * lut[data[i * C + 2]];
  }
  const meanLum = sum / n;

  // Repack to RGB half in a SharedArrayBuffer, rows TOP -> BOTTOM (three's
  // EXRLoader stores the bottom row first), suppressing unresolved point
  // sources on the way: a pixel brighter than the 3x3 median of luminance is
  // pulled towards it (keeps TONE.pointSourceKeep of the excess). The map's
  // single-pixel Gaia stars otherwise turn into grain/aliasing; the real stars
  // come from the GPU star field.
  const keep = TONE.pointSourceKeep;
  const sab = new SharedArrayBuffer(W * H * 3 * 2);
  const dst = new Uint16Array(sab);
  const lumRow = (y, out) => {
    const srow = (H - 1 - Math.min(H - 1, Math.max(0, y))) * W * C;
    for (let x = 0; x < W; x++) {
      const si = srow + x * C;
      out[x] = 0.2126 * lut[data[si]] + 0.7152 * lut[data[si + 1]] + 0.0722 * lut[data[si + 2]];
    }
  };
  let rPrev = new Float32Array(W), rCur = new Float32Array(W), rNext = new Float32Array(W);
  lumRow(-1, rPrev);
  lumRow(0, rCur);
  const win = new Float32Array(9);
  for (let y = 0; y < H; y++) {
    lumRow(y + 1, rNext);
    const srow = (H - 1 - y) * W * C;
    const drow = y * W * 3;
    for (let x = 0; x < W; x++) {
      const si = srow + x * C, di = drow + x * 3;
      const L = rCur[x];
      const xl = x === 0 ? W - 1 : x - 1, xr = x === W - 1 ? 0 : x + 1;
      win[0] = rPrev[xl]; win[1] = rPrev[x]; win[2] = rPrev[xr];
      win[3] = rCur[xl]; win[4] = L; win[5] = rCur[xr];
      win[6] = rNext[xl]; win[7] = rNext[x]; win[8] = rNext[xr];
      const med = median9(win);
      if (L > med && L > 0) {
        const f = (med + (L - med) * keep) / L;
        dst[di] = toHalf(lut[data[si]] * f);
        dst[di + 1] = toHalf(lut[data[si + 1]] * f);
        dst[di + 2] = toHalf(lut[data[si + 2]] * f);
      } else {
        dst[di] = data[si];
        dst[di + 1] = data[si + 1];
        dst[di + 2] = data[si + 2];
      }
    }
    const t = rPrev; rPrev = rCur; rCur = rNext; rNext = t;
  }
  console.log(`  decoded ${basename(path)}: ${W}x${H}, mean luminance ${meanLum.toExponential(3)} (${((performance.now() - t0) / 1000).toFixed(1)} s)`);
  return { sab, W, H, meanLum };
}

function runWorker(data) {
  return new Promise((res, rej) => {
    const w = new Worker(fileURLToPath(import.meta.url), { workerData: data });
    w.once('message', (m) => res(m));
    w.once('error', rej);
    w.once('exit', (code) => {
      if (code !== 0) rej(new Error(`worker exited with ${code}`));
    });
  });
}

async function main() {
  const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
  const args = parseArgs(process.argv.slice(2));
  const outDir = resolve(root, typeof args.out === 'string' ? args.out : 'public/sky');
  const epoch = args.epoch ? Number(args.epoch) : defaultEpoch();
  const force = !!args.force;
  const only = typeof args.only === 'string' ? args.only : null;
  const sharp = (await import('sharp')).default;
  const manifestPath = join(outDir, 'manifest.json');
  mkdirSync(outDir, { recursive: true });

  for (const [q, cfg] of Object.entries(QUALITIES)) {
    if (only && only !== q) continue;
    const src = args[cfg.srcKey] || process.env[cfg.envKey];
    if (!src || typeof src !== 'string') {
      console.log(`[${q}] no --${cfg.srcKey} given — skipped`);
      continue;
    }
    if (!existsSync(src)) throw new Error(`[${q}] source not found: ${src}`);
    const st = statSync(src);
    const inputKey = createHash('sha1')
      .update(JSON.stringify({ SCRIPT_VERSION, TONE, JPEG, size: cfg.size, matchGain: cfg.matchGain, epoch, src: basename(src), bytes: st.size, mtime: Math.round(st.mtimeMs) }))
      .digest('hex')
      .slice(0, 16);
    const faceDir = join(outDir, q);
    const manifest = readManifest(manifestPath);
    const prev = manifest.cube?.[q];
    if (!force && prev?.inputKey === inputKey && CUBE_FACES.every((f) => existsSync(join(faceDir, `${f}.jpg`)))) {
      console.log(`[${q}] up to date (${inputKey}) — skipped`);
      continue;
    }

    console.log(`[${q}] ${cfg.size}² faces from ${basename(src)} (epoch ${epoch})`);
    const t0 = performance.now();
    const { sab, W, H, meanLum } = await decodeExr(src);
    const gain = (TONE.referenceMeanLum / meanLum) * cfg.matchGain;
    const threads = Math.max(1, Math.min(6, availableParallelism()));
    const faces = new Array(6);
    let next = 0;
    const pump = async () => {
      while (next < 6) {
        const f = next++;
        faces[f] = await runWorker({ sab, W, H, face: f, size: cfg.size, epoch, tone: { ...TONE }, gain, seed: 0x9e3779b9 ^ (f * 7919 + cfg.size) });
      }
    };
    await Promise.all(Array.from({ length: threads }, pump));
    console.log(`  resampled 6 faces in ${((performance.now() - t0) / 1000).toFixed(1)} s`);

    mkdirSync(faceDir, { recursive: true });
    const files = {};
    let total = 0;
    const hash = createHash('sha1');
    for (let f = 0; f < 6; f++) {
      const file = join(faceDir, `${CUBE_FACES[f]}.jpg`);
      const buf = await sharp(Buffer.from(faces[f].buffer), { raw: { width: cfg.size, height: cfg.size, channels: 3 } })
        .jpeg(JPEG)
        .toBuffer();
      writeFileSync(file, buf);
      hash.update(buf);
      files[CUBE_FACES[f]] = buf.length;
      total += buf.length;
    }
    const m2 = readManifest(manifestPath);
    m2.version = 1;
    m2.credit = CREDIT;
    m2.frame = 'TEME axes ~ mean equator & equinox of `epoch` (IAU 1976 precession from ICRF/J2000); x->RA 0h, z->north pole';
    m2.cube ??= {};
    m2.cube[q] = {
      size: cfg.size,
      epoch,
      faces: CUBE_FACES,
      path: `${q}/`,
      ext: 'jpg',
      bytes: total,
      faceBytes: files,
      hash: hash.digest('hex').slice(0, 12),
      inputKey,
      source: basename(src),
      sourceMeanLum: Number(meanLum.toExponential(4)),
      gain: Number(gain.toFixed(4)),
      tone: TONE,
    };
    writeFileSync(manifestPath, JSON.stringify(sortKeys(m2), null, 2) + '\n');
    console.log(`  wrote ${q}: ${(total / 1048576).toFixed(2)} MiB (${((performance.now() - t0) / 1000).toFixed(1)} s total)`);
  }
}

function sortKeys(m) {
  const order = ['version', 'credit', 'frame', 'cube', 'stars'];
  const o = {};
  for (const k of order) if (k in m) o[k] = m[k];
  for (const k of Object.keys(m)) if (!(k in o)) o[k] = m[k];
  return o;
}
