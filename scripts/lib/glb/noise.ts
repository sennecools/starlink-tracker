/**
 * Deterministic, optionally periodic (tileable) noise for procedural
 * textures: integer hash, value noise, fBm and Worley (cellular) noise.
 * Periods are in lattice cells; pass 0 for non-periodic.
 */

/** 32-bit integer hash of (x, y, seed) -> [0, 1). */
export function hash2(x: number, y: number, seed: number): number {
  let h = (Math.imul(x | 0, 0x27d4eb2d) ^ Math.imul(y | 0, 0x165667b1) ^ Math.imul(seed | 0, 0x9e3779b9)) >>> 0;
  h = Math.imul(h ^ (h >>> 15), 0x85ebca6b);
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35);
  h ^= h >>> 16;
  return (h >>> 0) / 4294967296;
}

/** mulberry32 PRNG. */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const wrap = (i: number, p: number): number => (p > 0 ? ((i % p) + p) % p : i);
const quintic = (t: number): number => t * t * t * (t * (t * 6 - 15) + 10);

/** Value noise in [0, 1). */
export function valueNoise(x: number, y: number, px: number, py: number, seed: number): number {
  const xi = Math.floor(x);
  const yi = Math.floor(y);
  const fx = quintic(x - xi);
  const fy = quintic(y - yi);
  const x0 = wrap(xi, px);
  const x1 = wrap(xi + 1, px);
  const y0 = wrap(yi, py);
  const y1 = wrap(yi + 1, py);
  const a = hash2(x0, y0, seed);
  const b = hash2(x1, y0, seed);
  const c = hash2(x0, y1, seed);
  const d = hash2(x1, y1, seed);
  return a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy;
}

/** Fractal sum of value noise, normalised to roughly [0, 1). Periods double per octave. */
export function fbm(x: number, y: number, octaves: number, px: number, py: number, seed: number, gain = 0.5): number {
  let sum = 0;
  let amp = 1;
  let norm = 0;
  let f = 1;
  for (let o = 0; o < octaves; o++) {
    sum += amp * valueNoise(x * f, y * f, px * f, py * f, seed + o * 1013);
    norm += amp;
    amp *= gain;
    f *= 2;
  }
  return sum / norm;
}

export interface WorleyResult {
  f1: number;
  f2: number;
  /** Random id of the nearest cell in [0, 1). */
  id: number;
  /** Second random of the nearest cell in [0, 1). */
  id2: number;
  /** Offset from the nearest feature point to the sample. */
  dx: number;
  dy: number;
}

/** Worley noise (one jittered feature point per cell); writes into `out` (no allocation). */
export function worley(x: number, y: number, px: number, py: number, seed: number, out: WorleyResult, jitter = 0.9): void {
  const xi = Math.floor(x);
  const yi = Math.floor(y);
  let f1 = Infinity;
  let f2 = Infinity;
  for (let j = -1; j <= 1; j++) {
    for (let i = -1; i <= 1; i++) {
      const cx = xi + i;
      const cy = yi + j;
      const hx = wrap(cx, px);
      const hy = wrap(cy, py);
      const fx = cx + 0.5 + (hash2(hx, hy, seed) - 0.5) * jitter;
      const fy = cy + 0.5 + (hash2(hx, hy, seed + 7919) - 0.5) * jitter;
      const dx = x - fx;
      const dy = y - fy;
      const d = Math.sqrt(dx * dx + dy * dy);
      if (d < f1) {
        f2 = f1;
        f1 = d;
        out.id = hash2(hx, hy, seed + 104729);
        out.id2 = hash2(hx, hy, seed + 1299709);
        out.dx = dx;
        out.dy = dy;
      } else if (d < f2) {
        f2 = d;
      }
    }
  }
  out.f1 = f1;
  out.f2 = f2;
}

export function newWorley(): WorleyResult {
  return { f1: 0, f2: 0, id: 0, id2: 0, dx: 0, dy: 0 };
}

export const clamp01 = (v: number): number => (v < 0 ? 0 : v > 1 ? 1 : v);
export const lerp = (a: number, b: number, t: number): number => a + (b - a) * t;
export function smoothstep(e0: number, e1: number, x: number): number {
  const t = clamp01((x - e0) / (e1 - e0));
  return t * t * (3 - 2 * t);
}
