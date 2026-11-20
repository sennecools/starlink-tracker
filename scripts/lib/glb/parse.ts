/**
 * Minimal GLB reader used by the build summary and tests: header/chunk
 * parsing, node lookup and the scene's world-space bounding box (from the
 * POSITION accessor min/max, transformed through the node hierarchy at rest).
 */
import { CHUNK_BIN, CHUNK_JSON, GLB_MAGIC } from './gltf.ts';
import { identity, mul, transformPoint, trs } from './math.ts';
import type { Mat4, Quat, Vec3 } from './math.ts';

export interface GltfNodeJson {
  name?: string;
  mesh?: number;
  children?: number[];
  translation?: Vec3;
  rotation?: Quat;
  scale?: Vec3;
  matrix?: number[];
  extras?: Record<string, unknown>;
}

export interface GltfJson {
  asset: { version: string; generator?: string };
  scene?: number;
  scenes: Array<{ nodes: number[] }>;
  nodes: GltfNodeJson[];
  meshes: Array<{ name?: string; primitives: Array<{ attributes: Record<string, number>; indices?: number; material?: number }> }>;
  accessors: Array<{ min?: number[]; max?: number[]; count: number; type: string; componentType: number; bufferView?: number }>;
  bufferViews: Array<{ buffer: number; byteOffset?: number; byteLength: number }>;
  images?: Array<{ name?: string; bufferView?: number; mimeType?: string; uri?: string }>;
  textures?: Array<{ source?: number; sampler?: number }>;
  materials?: Array<Record<string, unknown> & { name?: string; doubleSided?: boolean }>;
  extensionsUsed?: string[];
  extensionsRequired?: string[];
  extras?: Record<string, unknown>;
}

export interface Glb {
  version: number;
  length: number;
  json: GltfJson;
  bin: Uint8Array;
}

export function parseGlb(bytes: Uint8Array): Glb {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (bytes.byteLength < 20) throw new Error('GLB too short');
  if (dv.getUint32(0, true) !== GLB_MAGIC) throw new Error('bad GLB magic');
  const version = dv.getUint32(4, true);
  const length = dv.getUint32(8, true);
  if (length !== bytes.byteLength) throw new Error(`GLB length ${length} != file size ${bytes.byteLength}`);
  let o = 12;
  let json: GltfJson | null = null;
  let bin: Uint8Array = new Uint8Array(0);
  while (o < length) {
    const chunkLen = dv.getUint32(o, true);
    const chunkType = dv.getUint32(o + 4, true);
    if (chunkLen % 4 !== 0) throw new Error('chunk not 4-byte aligned');
    const data = bytes.subarray(o + 8, o + 8 + chunkLen);
    if (chunkType === CHUNK_JSON) json = JSON.parse(new TextDecoder().decode(data)) as GltfJson;
    else if (chunkType === CHUNK_BIN) bin = data;
    o += 8 + chunkLen;
  }
  if (!json) throw new Error('GLB without JSON chunk');
  return { version, length, json, bin };
}

export function nodeLocalMatrix(n: GltfNodeJson): Mat4 {
  if (n.matrix) return n.matrix.slice();
  return trs(n.translation, n.rotation, n.scale);
}

/** World matrices of all nodes reachable from the default scene. */
export function worldMatrices(json: GltfJson): Map<number, Mat4> {
  const out = new Map<number, Mat4>();
  const visit = (i: number, parent: Mat4): void => {
    const m = mul(parent, nodeLocalMatrix(json.nodes[i]));
    out.set(i, m);
    for (const c of json.nodes[i].children ?? []) visit(c, m);
  };
  for (const r of json.scenes[json.scene ?? 0].nodes) visit(r, identity());
  return out;
}

/** World-space AABB of all mesh primitives (rest pose). */
export function sceneBounds(json: GltfJson): { min: Vec3; max: Vec3; size: Vec3 } {
  const min: Vec3 = [Infinity, Infinity, Infinity];
  const max: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (const [i, m] of worldMatrices(json)) {
    const n = json.nodes[i];
    if (n.mesh === undefined) continue;
    for (const p of json.meshes[n.mesh].primitives) {
      const acc = json.accessors[p.attributes.POSITION];
      if (!acc.min || !acc.max) throw new Error('POSITION accessor without bounds');
      for (let c = 0; c < 8; c++) {
        const corner: Vec3 = [
          c & 1 ? acc.max[0] : acc.min[0],
          c & 2 ? acc.max[1] : acc.min[1],
          c & 4 ? acc.max[2] : acc.min[2],
        ];
        const w = transformPoint(m, corner);
        for (let k = 0; k < 3; k++) {
          if (w[k] < min[k]) min[k] = w[k];
          if (w[k] > max[k]) max[k] = w[k];
        }
      }
    }
  }
  return { min, max, size: [max[0] - min[0], max[1] - min[1], max[2] - min[2]] };
}

export function findNode(json: GltfJson, name: string): number {
  return json.nodes.findIndex((n) => n.name === name);
}

/** Bytes of an embedded image. */
export function imageBytes(glb: Glb, imageIndex: number): Uint8Array {
  const img = glb.json.images?.[imageIndex];
  if (!img || img.bufferView === undefined) throw new Error(`image ${imageIndex} is not embedded`);
  const bv = glb.json.bufferViews[img.bufferView];
  return glb.bin.subarray(bv.byteOffset ?? 0, (bv.byteOffset ?? 0) + bv.byteLength);
}

/** Pixel size from a PNG (IHDR) or baseline/progressive JPEG (SOFn) header. */
export function imageSize(bytes: Uint8Array): { width: number; height: number } {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (bytes[0] === 0x89 && bytes[1] === 0x50) return { width: dv.getUint32(16), height: dv.getUint32(20) };
  if (bytes[0] === 0xff && bytes[1] === 0xd8) {
    let o = 2;
    while (o < bytes.length) {
      if (bytes[o] !== 0xff) throw new Error('bad JPEG marker');
      const marker = bytes[o + 1];
      const len = dv.getUint16(o + 2);
      if (marker >= 0xc0 && marker <= 0xcf && marker !== 0xc4 && marker !== 0xc8 && marker !== 0xcc) {
        return { height: dv.getUint16(o + 5), width: dv.getUint16(o + 7) };
      }
      o += 2 + len;
    }
  }
  throw new Error('unknown image format');
}
