/**
 * Tiny glTF 2.0 document builder + binary (.glb) writer. Core spec only:
 * one buffer (the GLB BIN chunk), non-interleaved vertex attributes, images
 * embedded as buffer views, PBR metallic-roughness materials.
 *
 * GLB layout (spec §4.4): 12-byte header (magic "glTF", version 2, total
 * length), JSON chunk (padded with spaces to 4 bytes), BIN chunk (padded with
 * zeros to 4 bytes).
 */
import type { MeshBuilder } from './mesh.ts';
import type { Quat, Vec3 } from './math.ts';

export const GLB_MAGIC = 0x46546c67; // "glTF"
export const CHUNK_JSON = 0x4e4f534a; // "JSON"
export const CHUNK_BIN = 0x004e4942; // "BIN\0"

const ARRAY_BUFFER = 34962;
const ELEMENT_ARRAY_BUFFER = 34963;
const FLOAT = 5126;
const UNSIGNED_SHORT = 5123;
const UNSIGNED_INT = 5125;

export const SAMPLER_REPEAT = { magFilter: 9729, minFilter: 9987, wrapS: 10497, wrapT: 10497 };
export const SAMPLER_CLAMP = { magFilter: 9729, minFilter: 9987, wrapS: 33071, wrapT: 33071 };
export const SAMPLER_MIRROR = { magFilter: 9729, minFilter: 9987, wrapS: 33648, wrapT: 33648 };

export interface TextureRef {
  index: number;
  texCoord?: number;
  scale?: number;
  strength?: number;
}

export interface MaterialDef {
  name: string;
  baseColorFactor?: [number, number, number, number];
  baseColorTexture?: number;
  metallicFactor?: number;
  roughnessFactor?: number;
  /** ORM texture: R = occlusion, G = roughness, B = metallic. */
  ormTexture?: number;
  /** Also bind the ORM texture's R channel as occlusionTexture. */
  useOcclusion?: boolean;
  occlusionStrength?: number;
  normalTexture?: number;
  normalScale?: number;
  doubleSided?: boolean;
}

export interface NodeDef {
  name: string;
  mesh?: number;
  translation?: Vec3;
  rotation?: Quat;
  scale?: Vec3;
  children?: number[];
  extras?: Record<string, unknown>;
}

type Json = Record<string, unknown>;

export class GltfDoc {
  private readonly bin: Uint8Array[] = [];
  private binLength = 0;
  readonly json: {
    asset: Json;
    scene: number;
    scenes: Json[];
    nodes: Json[];
    meshes: Json[];
    materials: Json[];
    textures: Json[];
    images: Json[];
    samplers: Json[];
    accessors: Json[];
    bufferViews: Json[];
    buffers: Json[];
    extras?: Json;
  };
  private readonly materialNeedsTangents: boolean[] = [];
  private readonly materialTextured: boolean[] = [];

  constructor(generator: string, copyright?: string) {
    this.json = {
      asset: { version: '2.0', generator, ...(copyright ? { copyright } : {}) },
      scene: 0,
      scenes: [],
      nodes: [],
      meshes: [],
      materials: [],
      textures: [],
      images: [],
      samplers: [],
      accessors: [],
      bufferViews: [],
      buffers: [],
    };
  }

  /** Appends bytes to the BIN chunk (4-byte aligned) and returns a buffer view index. */
  addBufferView(data: Uint8Array, target?: number): number {
    const pad = (4 - (this.binLength % 4)) % 4;
    if (pad) {
      this.bin.push(new Uint8Array(pad));
      this.binLength += pad;
    }
    const view: Json = { buffer: 0, byteOffset: this.binLength, byteLength: data.byteLength };
    if (target !== undefined) view.target = target;
    this.bin.push(data);
    this.binLength += data.byteLength;
    this.json.bufferViews.push(view);
    return this.json.bufferViews.length - 1;
  }

  private addAccessor(
    data: Float32Array | Uint16Array | Uint32Array,
    type: 'SCALAR' | 'VEC2' | 'VEC3' | 'VEC4',
    target: number,
    withBounds = false,
  ): number {
    const comps = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 }[type];
    const componentType = data instanceof Float32Array ? FLOAT : data instanceof Uint16Array ? UNSIGNED_SHORT : UNSIGNED_INT;
    const bufferView = this.addBufferView(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), target);
    const acc: Json = { bufferView, componentType, count: data.length / comps, type };
    if (withBounds) {
      const min = new Array<number>(comps).fill(Infinity);
      const max = new Array<number>(comps).fill(-Infinity);
      for (let i = 0; i < data.length; i++) {
        const k = i % comps;
        if (data[i] < min[k]) min[k] = data[i];
        if (data[i] > max[k]) max[k] = data[i];
      }
      // Bounds must match the stored float32 values exactly.
      acc.min = min.map((v) => Math.fround(v));
      acc.max = max.map((v) => Math.fround(v));
    }
    this.json.accessors.push(acc);
    return this.json.accessors.length - 1;
  }

  addSampler(s: Json): number {
    this.json.samplers.push({ ...s });
    return this.json.samplers.length - 1;
  }

  /** Embeds an encoded PNG/JPEG and returns a texture index. */
  addTexture(name: string, bytes: Uint8Array, mimeType: 'image/png' | 'image/jpeg', sampler: number): number {
    const bufferView = this.addBufferView(bytes);
    this.json.images.push({ name, bufferView, mimeType });
    this.json.textures.push({ name, source: this.json.images.length - 1, sampler });
    return this.json.textures.length - 1;
  }

  addMaterial(d: MaterialDef): number {
    const pbr: Json = {};
    if (d.baseColorFactor) pbr.baseColorFactor = d.baseColorFactor;
    if (d.baseColorTexture !== undefined) pbr.baseColorTexture = { index: d.baseColorTexture };
    pbr.metallicFactor = d.metallicFactor ?? (d.ormTexture !== undefined ? 1 : 0);
    pbr.roughnessFactor = d.roughnessFactor ?? 1;
    if (d.ormTexture !== undefined) pbr.metallicRoughnessTexture = { index: d.ormTexture };
    const mat: Json = { name: d.name, pbrMetallicRoughness: pbr };
    if (d.normalTexture !== undefined) {
      mat.normalTexture = d.normalScale !== undefined ? { index: d.normalTexture, scale: d.normalScale } : { index: d.normalTexture };
    }
    if (d.useOcclusion && d.ormTexture !== undefined) {
      mat.occlusionTexture = { index: d.ormTexture, strength: d.occlusionStrength ?? 1 };
    }
    if (d.doubleSided) mat.doubleSided = true;
    this.json.materials.push(mat);
    this.materialNeedsTangents.push(d.normalTexture !== undefined);
    this.materialTextured.push(
      d.baseColorTexture !== undefined || d.ormTexture !== undefined || d.normalTexture !== undefined,
    );
    return this.json.materials.length - 1;
  }

  /** Mesh from (builder, material) pairs; empty builders are skipped. */
  addMesh(name: string, parts: Array<[MeshBuilder, number]>): number {
    const primitives: Json[] = [];
    for (const [mb, material] of parts) {
      if (mb.triangleCount === 0) continue;
      const attributes: Json = {
        POSITION: this.addAccessor(new Float32Array(mb.positions), 'VEC3', ARRAY_BUFFER, true),
        NORMAL: this.addAccessor(new Float32Array(mb.normals), 'VEC3', ARRAY_BUFFER),
      };
      if (this.materialNeedsTangents[material]) {
        attributes.TANGENT = this.addAccessor(mb.computeTangents(), 'VEC4', ARRAY_BUFFER);
      }
      if (this.materialTextured[material]) {
        attributes.TEXCOORD_0 = this.addAccessor(new Float32Array(mb.uvs), 'VEC2', ARRAY_BUFFER);
      }
      const idx = mb.vertexCount <= 65535 ? new Uint16Array(mb.indices) : new Uint32Array(mb.indices);
      primitives.push({ attributes, indices: this.addAccessor(idx, 'SCALAR', ELEMENT_ARRAY_BUFFER), material, mode: 4 });
    }
    if (primitives.length === 0) throw new Error(`mesh ${name} has no geometry`);
    this.json.meshes.push({ name, primitives });
    return this.json.meshes.length - 1;
  }

  addNode(d: NodeDef): number {
    const n: Json = { name: d.name };
    if (d.mesh !== undefined) n.mesh = d.mesh;
    // Round away float noise (1.4300000000000002) for readable JSON; 1 µm / 1e-9.
    const r6 = (v: number): number => Math.round(v * 1e6) / 1e6 + 0;
    const r9 = (v: number): number => Math.round(v * 1e9) / 1e9 + 0;
    if (d.translation && d.translation.some((v) => r6(v) !== 0)) n.translation = d.translation.map(r6);
    if (d.rotation && !(d.rotation[0] === 0 && d.rotation[1] === 0 && d.rotation[2] === 0 && d.rotation[3] === 1)) {
      n.rotation = d.rotation.map(r9);
    }
    if (d.scale) n.scale = d.scale;
    if (d.children && d.children.length) n.children = d.children;
    if (d.extras) n.extras = d.extras;
    this.json.nodes.push(n);
    return this.json.nodes.length - 1;
  }

  addScene(name: string, nodes: number[]): number {
    this.json.scenes.push({ name, nodes });
    return this.json.scenes.length - 1;
  }

  /** Serialises to a GLB byte array. */
  toGlb(): Uint8Array {
    const binPad = (4 - (this.binLength % 4)) % 4;
    const binLen = this.binLength + binPad;
    this.json.buffers = [{ byteLength: binLen }];
    const json: Json = { ...this.json };
    for (const k of ['textures', 'images', 'samplers', 'materials']) {
      if ((json[k] as unknown[]).length === 0) delete json[k];
    }
    let jsonBytes = new TextEncoder().encode(JSON.stringify(json));
    const jsonPad = (4 - (jsonBytes.length % 4)) % 4;
    if (jsonPad) {
      const padded = new Uint8Array(jsonBytes.length + jsonPad).fill(0x20);
      padded.set(jsonBytes);
      jsonBytes = padded;
    }
    const total = 12 + 8 + jsonBytes.length + 8 + binLen;
    const out = new Uint8Array(total);
    const dv = new DataView(out.buffer);
    dv.setUint32(0, GLB_MAGIC, true);
    dv.setUint32(4, 2, true);
    dv.setUint32(8, total, true);
    dv.setUint32(12, jsonBytes.length, true);
    dv.setUint32(16, CHUNK_JSON, true);
    out.set(jsonBytes, 20);
    let o = 20 + jsonBytes.length;
    dv.setUint32(o, binLen, true);
    dv.setUint32(o + 4, CHUNK_BIN, true);
    o += 8;
    for (const part of this.bin) {
      out.set(part, o);
      o += part.byteLength;
    }
    return out;
  }
}
