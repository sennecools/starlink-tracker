import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, statSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { GltfDoc } from '../scripts/lib/glb/gltf.ts';
import { MeshBuilder, box } from '../scripts/lib/glb/mesh.ts';
import { cross, identity, transformPoint } from '../scripts/lib/glb/math.ts';
import type { Vec3 } from '../scripts/lib/glb/math.ts';
import { findNode, imageBytes, imageSize, parseGlb, sceneBounds, worldMatrices } from '../scripts/lib/glb/parse.ts';
import type { Glb, GltfJson } from '../scripts/lib/glb/parse.ts';
import { rasterize } from '../scripts/lib/glb/texture.ts';
import { V3, buildV3 } from '../scripts/lib/glb/starlink.ts';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const MODELS = join(ROOT, 'public', 'models');
const MAX_BYTES = 6 * 1024 * 1024;

interface Expect {
  file: string;
  /** Overall scene AABB (metres): X (velocity), Z (wing axis); ±10 %. */
  sizeX: number;
  sizeZ: number;
  /** Allowed overall height range (bus + antennas + terminals). */
  sizeY: [number, number];
  bus: Vec3;
  wings: Array<'L' | 'R'>;
  wingWidth: number;
  wingLength: number;
  totalArrayAreaM2: number;
  lasers: number;
}

const SPECS: Expect[] = [
  {
    file: 'starlink-v2mini.glb',
    sizeX: 4.1,
    sizeZ: 30,
    sizeY: [0.25, 1.0],
    bus: [4.1, 0.25, 2.7],
    wings: ['L', 'R'],
    wingWidth: 4.1,
    wingLength: 12.8,
    totalArrayAreaM2: 105,
    lasers: 3,
  },
  {
    file: 'starlink-v15.glb',
    sizeX: 2.8,
    sizeZ: 2.8 + 0.6 + 8.1,
    sizeY: [0.2, 1.0],
    bus: [1.4, 0.2, 2.8],
    wings: ['R'],
    wingWidth: 2.8,
    wingLength: 8.1,
    totalArrayAreaM2: 2.8 * 8.1,
    lasers: 4,
  },
  {
    // Artist's approximation (no official drawings): ~60 m span, flat-pack bus.
    file: 'starlink-v3.glb',
    sizeX: 7.5,
    sizeZ: 60,
    sizeY: [0.4, 1.4],
    bus: [7.5, 0.4, 3.2],
    wings: ['L', 'R'],
    wingWidth: 4.9,
    wingLength: 27,
    // 2 wings x 2 columns of 2.35 m x 22 panels (25.95 m of cells per column).
    totalArrayAreaM2: 2 * 4.7 * 25.95,
    lasers: 6,
  },
];

const near = (actual: number, expected: number, tol = 0.1, what = ''): void => {
  assert.ok(
    Math.abs(actual - expected) <= Math.abs(expected) * tol,
    `${what}: ${actual.toFixed(3)} not within ±${tol * 100}% of ${expected}`,
  );
};

const load = (file: string): Glb => parseGlb(new Uint8Array(readFileSync(join(MODELS, file))));

function primitiveBounds(json: GltfJson, meshIndex: number, materialName: string): { min: number[]; max: number[] } {
  const mats = json.materials ?? [];
  const p = json.meshes[meshIndex].primitives.find((q) => mats[q.material ?? -1]?.name === materialName);
  assert.ok(p, `mesh ${meshIndex} has a ${materialName} primitive`);
  const acc = json.accessors[p.attributes.POSITION];
  assert.ok(acc.min && acc.max);
  return { min: acc.min, max: acc.max };
}

function floatView(glb: Glb, accessorIndex: number): Float32Array {
  const acc = glb.json.accessors[accessorIndex];
  assert.equal(acc.componentType, 5126);
  const bv = glb.json.bufferViews[acc.bufferView ?? -1];
  const comps = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 }[acc.type] ?? 1;
  const start = glb.bin.byteOffset + (bv.byteOffset ?? 0);
  return new Float32Array(glb.bin.buffer.slice(start, start + acc.count * comps * 4));
}

for (const spec of SPECS) {
  test(`${spec.file}: GLB container, size limit, core-only glTF`, () => {
    const size = statSync(join(MODELS, spec.file)).size;
    assert.ok(size < MAX_BYTES, `${spec.file} is ${size} bytes`);
    assert.ok(size > 200 * 1024, 'has embedded textures');
    const glb = load(spec.file);
    assert.equal(glb.version, 2);
    assert.equal(glb.length, size);
    assert.equal(glb.json.asset.version, '2.0');
    assert.ok(glb.bin.byteLength > 0, 'BIN chunk present');
    const allowed = new Set(['KHR_texture_transform']);
    for (const e of glb.json.extensionsUsed ?? []) assert.ok(allowed.has(e), `extension ${e} not allowed`);
    assert.equal((glb.json.extensionsRequired ?? []).length, 0);
    // Every image is embedded (no external URIs).
    for (const img of glb.json.images ?? []) {
      assert.equal(img.uri, undefined);
      assert.ok(img.mimeType === 'image/jpeg' || img.mimeType === 'image/png');
    }
  });

  test(`${spec.file}: required node names and pivot setup`, () => {
    const { json } = load(spec.file);
    assert.ok(findNode(json, 'Bus') >= 0, 'Bus node');
    assert.notEqual(json.nodes[findNode(json, 'Bus')].mesh, undefined);
    for (const side of ['L', 'R'] as const) {
      const i = findNode(json, `ArrayPivot_${side}`);
      if (!spec.wings.includes(side)) {
        assert.equal(i, -1, `no ArrayPivot_${side}`);
        continue;
      }
      assert.ok(i >= 0, `ArrayPivot_${side}`);
      const pivot = json.nodes[i];
      // Identity rotation so the renderer can set a pure Z rotation.
      assert.equal(pivot.rotation, undefined);
      assert.equal(pivot.matrix, undefined);
      assert.ok(pivot.translation, 'pivot sits at the wing root');
      const z = pivot.translation[2];
      assert.equal(Math.sign(z), side === 'R' ? 1 : -1, 'R wing at +Z, L wing at -Z');
      assert.ok(Math.abs(z) > spec.bus[2] / 2 && Math.abs(z) < spec.bus[2] / 2 + 0.3, 'pivot at the bus edge');
      assert.equal(pivot.translation[0], 0);
      assert.equal(pivot.translation[1], 0);
      assert.ok((pivot.children ?? []).length > 0, 'wing geometry is a child of the pivot');
    }
    const lasers = json.nodes.filter((n) => /^LaserTerminal_\d+$/.test(n.name ?? ''));
    assert.equal(lasers.length, spec.lasers);
    for (let k = 0; k < spec.lasers; k++) {
      const n = json.nodes[findNode(json, `LaserTerminal_${k}`)];
      assert.notEqual(n.mesh, undefined);
      // On top of the bus (zenith side).
      assert.ok((n.translation?.[1] ?? 0) > spec.bus[1] / 2);
    }
  });

  test(`${spec.file}: bounding box and array dimensions match the spec (±10%)`, () => {
    const glb = load(spec.file);
    const b = sceneBounds(glb.json);
    near(b.size[0], spec.sizeX, 0.1, 'size X');
    near(b.size[2], spec.sizeZ, 0.1, 'size Z');
    assert.ok(b.size[1] >= spec.sizeY[0] && b.size[1] <= spec.sizeY[1], `size Y ${b.size[1]}`);
    // Bus core.
    const bus = primitiveBounds(glb.json, glb.json.nodes[findNode(glb.json, 'Bus')].mesh ?? -1, 'BusGraphite');
    near(bus.max[0] - bus.min[0], spec.bus[0], 0.1, 'bus X');
    near(bus.max[1] - bus.min[1], spec.bus[1], 0.1, 'bus Y');
    near(bus.max[2] - bus.min[2], spec.bus[2], 0.1, 'bus Z');
    // Wing (cells only): width along X, length along Z.
    const wingNode = glb.json.nodes[findNode(glb.json, 'Wing_R')];
    const cells = primitiveBounds(glb.json, wingNode.mesh ?? -1, 'SolarCells');
    near(cells.max[0] - cells.min[0], spec.wingWidth, 0.1, 'wing width');
    near(cells.max[2] - cells.min[2], spec.wingLength, 0.1, 'wing length');
    const extras = (glb.json.nodes[glb.json.scenes[0].nodes[0]].extras ?? {}) as { wing?: { areaM2?: number } };
    near(extras.wing?.areaM2 ?? 0, spec.totalArrayAreaM2, 0.1, 'array area');
    // Mirrored wings: L spans -Z, R spans +Z.
    if (spec.wings.length === 2) {
      const W = worldMatrices(glb.json);
      const l = W.get(findNode(glb.json, 'Wing_L'));
      assert.ok(l);
      const tip = transformPoint(l, [0, 0, cells.max[2]]);
      assert.ok(tip[2] < -spec.wingLength, 'left wing tip on -Z');
    }
  });

  test(`${spec.file}: textures are power-of-two and within size budgets`, () => {
    const glb = load(spec.file);
    const images = glb.json.images ?? [];
    assert.ok(images.length >= 8);
    images.forEach((img, i) => {
      const { width, height } = imageSize(imageBytes(glb, i));
      const pot = (n: number): boolean => (n & (n - 1)) === 0;
      assert.ok(pot(width) && pot(height), `${img.name} ${width}x${height} power of two`);
      const max = (img.name ?? '').startsWith('solar_cells') ? 2048 : 1024;
      assert.ok(width <= max && height <= max, `${img.name} ${width}x${height} <= ${max}`);
    });
  });

  test(`${spec.file}: materials, normals and tangents`, () => {
    const glb = load(spec.file);
    const mats = glb.json.materials ?? [];
    for (const m of mats) {
      if (m.doubleSided) assert.equal(m.name, 'DishReflector', 'double-sided only for thin shells');
    }
    for (const mesh of glb.json.meshes) {
      for (const p of mesh.primitives) {
        const mat = mats[p.material ?? -1];
        assert.ok(mat, 'primitive has a material');
        const n = floatView(glb, p.attributes.NORMAL);
        for (let i = 0; i < n.length; i += 3) {
          assert.ok(Math.abs(Math.hypot(n[i], n[i + 1], n[i + 2]) - 1) < 1e-3, 'unit normals');
        }
        if (mat.normalTexture) {
          assert.notEqual(p.attributes.TANGENT, undefined, `${mat.name} has tangents`);
          const t = floatView(glb, p.attributes.TANGENT);
          for (let i = 0; i < t.length; i += 4) assert.ok(Math.abs(t[i + 3]) === 1, 'tangent w = ±1');
        }
      }
    }
  });

  test(`${spec.file}: Khronos glTF-Validator reports no errors`, async () => {
    const require = createRequire(import.meta.url);
    let validator: { validateBytes(b: Uint8Array, o?: object): Promise<{ issues: { numErrors: number; messages: Array<{ code: string; message: string; severity: number }> } }> };
    try {
      validator = require('gltf-validator');
    } catch {
      return; // dev dependency not installed: skip
    }
    const bytes = new Uint8Array(readFileSync(join(MODELS, spec.file)));
    const report = await validator.validateBytes(bytes, { format: 'glb', maxIssues: 50 });
    const errors = report.issues.messages.filter((m) => m.severity === 0);
    assert.equal(report.issues.numErrors, 0, JSON.stringify(errors.slice(0, 5)));
  });
}

function primitivePositions(glb: Glb, meshIndex: number, materialName: string): Float32Array {
  const mats = glb.json.materials ?? [];
  const p = glb.json.meshes[meshIndex].primitives.find((q) => mats[q.material ?? -1]?.name === materialName);
  assert.ok(p, `mesh ${meshIndex} has a ${materialName} primitive`);
  return floatView(glb, p.attributes.POSITION);
}

test('starlink-v3.glb: six laser terminals spread fore/aft/sides on the zenith face', () => {
  const { json } = load('starlink-v3.glb');
  const [bx, by, bz] = V3.bus;
  const pos: Vec3[] = [];
  for (let k = 0; k < 6; k++) {
    const n = json.nodes[findNode(json, `LaserTerminal_${k}`)];
    const t = n.translation;
    assert.ok(t, `LaserTerminal_${k} has a translation`);
    assert.ok(Math.abs(t[0]) < bx / 2 && Math.abs(t[2]) < bz / 2, `LaserTerminal_${k} inside the bus footprint`);
    assert.ok(t[1] > by / 2, `LaserTerminal_${k} above the zenith plate`);
    pos.push([t[0], t[1], t[2]]);
  }
  assert.equal(findNode(json, 'LaserTerminal_6'), -1, 'exactly six');
  for (let i = 0; i < 6; i++) {
    for (let j = i + 1; j < 6; j++) {
      assert.ok(Math.hypot(pos[i][0] - pos[j][0], pos[i][2] - pos[j][2]) > 0.5, `terminals ${i} and ${j} are apart`);
    }
  }
  assert.equal(pos.filter((p) => p[0] > bx / 4).length, 2, 'fore pair');
  assert.equal(pos.filter((p) => p[0] < -bx / 4).length, 2, 'aft pair');
  assert.ok(pos.some((p) => Math.abs(p[0]) < bx / 4 && p[2] > bz / 4), 'right-side terminal');
  assert.ok(pos.some((p) => Math.abs(p[0]) < bx / 4 && p[2] < -bz / 4), 'left-side terminal');
});

test('starlink-v3.glb: two-column wings, dense nadir arrays, four gateway dishes', () => {
  const glb = load('starlink-v3.glb');
  const { json } = glb;
  // Wing cells: two columns separated by the spine gap at x = 0.
  const cells = primitivePositions(glb, json.nodes[findNode(json, 'Wing_R')].mesh ?? -1, 'SolarCells');
  let left = 0;
  let right = 0;
  for (let i = 0; i < cells.length; i += 3) {
    assert.ok(Math.abs(cells[i]) >= (V3.columnGap ?? 0) / 2 - 1e-4, 'no cells in the spine gap');
    if (cells[i] < 0) left++;
    else right++;
  }
  assert.ok(left > 0 && left === right, 'both columns populated symmetrically');
  // Phased arrays: one antenna face (4 vertices) per panel; more than V2 Mini's 4.
  const bus = json.nodes[findNode(json, 'Bus')].mesh ?? -1;
  assert.ok(primitivePositions(glb, bus, 'PhasedArray').length / 12 >= 8, '>= 8 phased-array panels');
  // Gateway dishes hang below the bus, one per quadrant of the nadir face.
  const dishes = primitivePositions(glb, bus, 'DishReflector');
  const quadrants = new Set<string>();
  for (let i = 0; i < dishes.length; i += 3) {
    assert.ok(dishes[i + 1] < -V3.bus[1] / 2, 'dish below the nadir plate');
    quadrants.add(`${Math.sign(dishes[i])}${Math.sign(dishes[i + 2])}`);
  }
  assert.equal(quadrants.size, 4, 'four dishes, one per quadrant');
});

test('starlink-v3.glb: labelled as an artist\'s approximation, larger than V2 Mini', () => {
  const v3 = load('starlink-v3.glb').json;
  const mini = load('starlink-v2mini.glb').json;
  const copyright = (v3.asset as { copyright?: string }).copyright ?? '';
  assert.match(copyright, /artist's approximation/i);
  assert.match(copyright, /not an official SpaceX model/i);
  type RootExtras = { model?: string; spanM?: number; provenance?: string; wing?: { areaM2?: number } };
  const ex = (j: GltfJson): RootExtras => (j.nodes[j.scenes[0].nodes[0]].extras ?? {}) as RootExtras;
  assert.equal(ex(v3).model, 'V3');
  assert.match(ex(v3).provenance ?? '', /artist's approximation/i);
  near(ex(v3).spanM ?? 0, 60, 0.05, 'span');
  assert.ok((ex(v3).wing?.areaM2 ?? 0) > 2 * (ex(mini).wing?.areaM2 ?? Infinity), 'more than twice the V2 Mini array area');
});

test('buildV3() is deterministic and matches the committed starlink-v3.glb', async () => {
  const { glb } = await buildV3();
  const committed = new Uint8Array(readFileSync(join(MODELS, 'starlink-v3.glb')));
  assert.equal(glb.byteLength, committed.byteLength, 'same size (run `node scripts/build-models.mjs --only=v3` after changing the generator)');
  assert.ok(Buffer.from(glb).equals(Buffer.from(committed)), 'byte-identical output');
});

test('box faces read un-mirrored and tangents follow the glTF (+Y up) convention', () => {
  const mb = new MeshBuilder('t');
  box(identity(), [0, 0, 0], [2, 1, 1], { py: { mb, uv: { kind: 'fit' } } });
  const t = mb.computeTangents();
  // Top face: u runs along +X, v along +Z (image down) => T = +X and the
  // shader bitangent cross(N, T) * w must point along -Z (image up).
  for (let v = 0; v < mb.vertexCount; v++) {
    assert.deepEqual([t[v * 4], t[v * 4 + 1], t[v * 4 + 2]].map((x) => Math.round(x)), [1, 0, 0]);
    const n: Vec3 = [mb.normals[v * 3], mb.normals[v * 3 + 1], mb.normals[v * 3 + 2]];
    const b = cross(n, [t[v * 4], t[v * 4 + 1], t[v * 4 + 2]]).map((x) => x * t[v * 4 + 3]);
    assert.deepEqual(b.map((x) => Math.round(x)), [0, 0, -1]);
  }
  // Mirrored UVs flip the handedness.
  const mm = new MeshBuilder('m');
  box(identity(), [0, 0, 0], [2, 1, 1], { py: { mb: mm, uv: { kind: 'fit', flipV: true } } });
  assert.equal(mm.computeTangents()[3], -1);
  // Winding: counter-clockwise around +Y.
  const [a, b2, c] = mb.indices;
  const P = (i: number): Vec3 => [mb.positions[i * 3], mb.positions[i * 3 + 1], mb.positions[i * 3 + 2]];
  const e1 = P(b2).map((x, k) => x - P(a)[k]) as Vec3;
  const e2 = P(c).map((x, k) => x - P(a)[k]) as Vec3;
  assert.ok(cross(e1, e2)[1] > 0);
});

test('height-field normal maps use +X = +u and +Y = image up', () => {
  const maps = rasterize({
    name: 'bump',
    width: 16,
    height: 16,
    sizeU: 1,
    sizeV: 1,
    tileable: false,
    ss: 1,
    sample(u, v, s) {
      s.h = Math.max(0, 0.2 - Math.hypot(u - 0.5, v - 0.5));
    },
  });
  const px = (x: number, y: number): number[] => [...maps.normal.subarray((y * 16 + x) * 3, (y * 16 + x) * 3 + 3)];
  assert.ok(px(10, 8)[0] > 140, 'right of the bump tilts toward +u');
  assert.ok(px(5, 8)[0] < 116, 'left of the bump tilts toward -u');
  assert.ok(px(8, 5)[1] > 140, 'above the bump (image up) tilts toward +Y');
  assert.ok(px(8, 10)[1] < 116, 'below the bump tilts toward -Y');
});

test('GltfDoc writes a GLB that parses back with exact bounds', () => {
  const doc = new GltfDoc('test');
  const mat = doc.addMaterial({ name: 'm', baseColorFactor: [1, 0, 0, 1] });
  const mb = new MeshBuilder('b');
  const spec = { mb };
  box(identity(), [1, 2, 3], [2, 4, 6], { px: spec, nx: spec, py: spec, ny: spec, pz: spec, nz: spec });
  const mesh = doc.addMesh('box', [[mb, mat]]);
  const node = doc.addNode({ name: 'Box', mesh, translation: [10, 0, 0] });
  doc.addScene('s', [node]);
  const bytes = doc.toGlb();
  assert.equal(bytes.byteLength % 4, 0);
  const glb = parseGlb(bytes);
  const b = sceneBounds(glb.json);
  assert.deepEqual(b.min, [10, 0, 0]);
  assert.deepEqual(b.max, [12, 4, 6]);
  assert.equal(glb.json.meshes[0].primitives[0].attributes.TEXCOORD_0, undefined, 'no UVs for untextured materials');
});
