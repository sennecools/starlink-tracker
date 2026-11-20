/**
 * Cesium-free helpers for the sky assets in public/sky (manifest format,
 * stars.bin decoding, star vertex building, the 4k device policy). Kept free
 * of Cesium and the DOM so node --test can cover them.
 */
import type { SkyQuality } from '../../shared/phase2.ts';
import { STAR_STRIDE, starDisplayColor } from './skyMath.ts';

export interface CubeEntry {
  size: number;
  epoch: number;
  faces: string[];
  /** Folder relative to the sky base URL, with trailing slash ("2k/"). */
  path: string;
  ext: string;
  bytes: number;
  faceBytes?: Record<string, number>;
  /** Content hash, used as a cache-busting query parameter. */
  hash: string;
}

export interface StarsEntry {
  file: string;
  count: number;
  stride: number;
  bytes: number;
  epoch: number;
  magLimit?: number;
  magRange?: [number, number];
  hash: string;
}

export interface SkyManifest {
  version: number;
  credit?: string;
  frame?: string;
  cube: Partial<Record<SkyQuality, CubeEntry>>;
  stars?: StarsEntry;
}

export interface NavigatorHints {
  deviceMemory?: number;
  connection?: { saveData?: boolean; effectiveType?: string };
  userAgentData?: { mobile?: boolean };
  maxTouchPoints?: number;
}

/**
 * Device/network policy for the 4k cube (~10 MB download, ~540 MB of VRAM
 * with mipmaps, ~0.9 GB transient while it replaces the 2k cube):
 * MAX_CUBE_MAP_TEXTURE_SIZE >= 4096, not a phone/tablet (UA-CH mobile, or a
 * touch device whose primary pointer is coarse: iOS/iPadOS Safari, Android
 * Firefox and 8 GB-capped Android Chrome hide or cap the memory hint),
 * deviceMemory >= 8 GB or unknown, no Save-Data, effectiveType '4g' or unknown.
 */
export function deviceAllows4k(maxCubeMapSize: number, nav: NavigatorHints | undefined, coarsePointer = false): boolean {
  if (!(maxCubeMapSize >= 4096)) return false;
  if (nav?.userAgentData?.mobile === true) return false;
  if (coarsePointer && (nav?.maxTouchPoints ?? 0) > 0) return false;
  const mem = nav?.deviceMemory;
  if (typeof mem === 'number' && mem < 8) return false;
  const conn = nav?.connection;
  if (conn) {
    if (conn.saveData) return false;
    if (conn.effectiveType && conn.effectiveType !== '4g') return false;
  }
  return true;
}

/** stars.bin -> Float32Array (validates against the manifest; handles big-endian hosts). */
export function decodeStarsBin(buf: ArrayBuffer, entry: StarsEntry): Float32Array {
  if (entry.stride !== STAR_STRIDE) throw new Error(`stars.bin stride ${entry.stride}, expected ${STAR_STRIDE}`);
  if (buf.byteLength !== entry.count * STAR_STRIDE * 4) throw new Error('stars.bin size does not match the manifest');
  const littleEndian = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;
  if (littleEndian) return new Float32Array(buf);
  const dv = new DataView(buf);
  const out = new Float32Array(buf.byteLength / 4);
  for (let i = 0; i < out.length; i++) out[i] = dv.getFloat32(i * 4, true);
  return out;
}

const BV_LUT_SIZE = 256;
const BV_MIN = -0.4;
const BV_MAX = 2.0;
/** Floats per star vertex: dir xyz, colour rgb, mag. */
export const STAR_VERTEX_FLOATS = 7;

function buildColorLut(saturation: number): Float32Array {
  const lut = new Float32Array(BV_LUT_SIZE * 3);
  const c = new Float32Array(3);
  for (let i = 0; i < BV_LUT_SIZE; i++) {
    starDisplayColor(BV_MIN + ((BV_MAX - BV_MIN) * i) / (BV_LUT_SIZE - 1), saturation, c);
    lut[i * 3] = c[0];
    lut[i * 3 + 1] = c[1];
    lut[i * 3 + 2] = c[2];
  }
  return lut;
}

/** Interleaved GPU vertices (dir xyz, display rgb, mag) from stars.bin values. */
export function buildStarVertices(data: Float32Array, saturation: number): Float32Array {
  const n = Math.floor(data.length / STAR_STRIDE);
  const lut = buildColorLut(saturation);
  const out = new Float32Array(n * STAR_VERTEX_FLOATS);
  for (let i = 0; i < n; i++) {
    const s = i * STAR_STRIDE;
    const o = i * STAR_VERTEX_FLOATS;
    out[o] = data[s];
    out[o + 1] = data[s + 1];
    out[o + 2] = data[s + 2];
    let li = Math.round(((data[s + 4] - BV_MIN) / (BV_MAX - BV_MIN)) * (BV_LUT_SIZE - 1));
    if (!(li >= 0)) li = 0;
    if (li > BV_LUT_SIZE - 1) li = BV_LUT_SIZE - 1;
    out[o + 3] = lut[li * 3];
    out[o + 4] = lut[li * 3 + 1];
    out[o + 5] = lut[li * 3 + 2];
    out[o + 6] = data[s + 3];
  }
  return out;
}
