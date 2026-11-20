/**
 * Indexed triangle-mesh builder plus a few parametric primitives (box,
 * cylinder/frustum, spherical cap, paraboloid dish, rod). One MeshBuilder =
 * one glTF primitive (one material). Primitives take an explicit placement
 * matrix `m` (rigid, column-major) so faces of one box can go to different
 * builders/materials.
 *
 * Conventions:
 *  - Triangles are counter-clockwise seen from the side the normal points to
 *    (glTF front face); `quad()`/`tri()` fix the winding from the normal.
 *  - UVs use the glTF convention: (0,0) is the top-left texel of the image,
 *    v grows downwards.
 *  - Tangents (MikkTSpace-compatible orientation): xyz = dP/du, w = handedness
 *    such that cross(N, T) * w = dP/d(-v), i.e. the "+Y up" (OpenGL)
 *    normal-map convention required by glTF.
 */
import {
  add,
  cross,
  dot,
  length,
  mul,
  normalize,
  scale,
  sub,
  transformDir,
  transformPoint,
} from './math.ts';
import type { Mat4, Vec3 } from './math.ts';

export type Vec2 = [number, number];

export class MeshBuilder {
  readonly name: string;
  readonly positions: number[] = [];
  readonly normals: number[] = [];
  readonly uvs: number[] = [];
  readonly indices: number[] = [];

  constructor(name: string) {
    this.name = name;
  }

  get vertexCount(): number {
    return this.positions.length / 3;
  }

  get triangleCount(): number {
    return this.indices.length / 3;
  }

  /** Adds a vertex given in local coordinates of placement `m`. */
  vertex(m: Mat4, p: Vec3, n: Vec3, uv: Vec2): number {
    const wp = transformPoint(m, p);
    const wn = normalize(transformDir(m, n));
    this.positions.push(wp[0], wp[1], wp[2]);
    this.normals.push(wn[0], wn[1], wn[2]);
    this.uvs.push(uv[0], uv[1]);
    return this.vertexCount - 1;
  }

  /** Triangle by vertex index; winding is corrected to agree with the vertex normals. */
  tri(a: number, b: number, c: number): void {
    const P = this.positions;
    const N = this.normals;
    const pa: Vec3 = [P[a * 3], P[a * 3 + 1], P[a * 3 + 2]];
    const pb: Vec3 = [P[b * 3], P[b * 3 + 1], P[b * 3 + 2]];
    const pc: Vec3 = [P[c * 3], P[c * 3 + 1], P[c * 3 + 2]];
    const fn = cross(sub(pb, pa), sub(pc, pa));
    if (length(fn) < 1e-12) return; // degenerate: drop
    const vn: Vec3 = [
      N[a * 3] + N[b * 3] + N[c * 3],
      N[a * 3 + 1] + N[b * 3 + 1] + N[c * 3 + 1],
      N[a * 3 + 2] + N[b * 3 + 2] + N[c * 3 + 2],
    ];
    if (dot(fn, vn) >= 0) this.indices.push(a, b, c);
    else this.indices.push(a, c, b);
  }

  /** Planar quad p0..p3 (in order around the quad) with a shared normal. */
  quad(m: Mat4, p: [Vec3, Vec3, Vec3, Vec3], n: Vec3, uv: [Vec2, Vec2, Vec2, Vec2]): void {
    const i0 = this.vertex(m, p[0], n, uv[0]);
    const i1 = this.vertex(m, p[1], n, uv[1]);
    const i2 = this.vertex(m, p[2], n, uv[2]);
    const i3 = this.vertex(m, p[3], n, uv[3]);
    this.tri(i0, i1, i2);
    this.tri(i0, i2, i3);
  }

  /** Axis-aligned bounds of the positions. */
  bounds(): { min: Vec3; max: Vec3 } {
    const min: Vec3 = [Infinity, Infinity, Infinity];
    const max: Vec3 = [-Infinity, -Infinity, -Infinity];
    const P = this.positions;
    for (let i = 0; i < P.length; i += 3) {
      for (let k = 0; k < 3; k++) {
        if (P[i + k] < min[k]) min[k] = P[i + k];
        if (P[i + k] > max[k]) max[k] = P[i + k];
      }
    }
    return { min, max };
  }

  /**
   * Per-vertex tangents (vec4) from positions/normals/uvs (Lengyel's method,
   * accumulated per vertex, Gram-Schmidt orthogonalised). See the header for
   * the handedness convention.
   */
  computeTangents(): Float32Array {
    const P = this.positions;
    const N = this.normals;
    const T = this.uvs;
    const nv = this.vertexCount;
    const tan = new Float64Array(nv * 3);
    const bit = new Float64Array(nv * 3);
    const I = this.indices;
    for (let f = 0; f < I.length; f += 3) {
      const a = I[f];
      const b = I[f + 1];
      const c = I[f + 2];
      const e1: Vec3 = [P[b * 3] - P[a * 3], P[b * 3 + 1] - P[a * 3 + 1], P[b * 3 + 2] - P[a * 3 + 2]];
      const e2: Vec3 = [P[c * 3] - P[a * 3], P[c * 3 + 1] - P[a * 3 + 1], P[c * 3 + 2] - P[a * 3 + 2]];
      // Use v' = -v so the bitangent is dP/d(-v) (glTF "+Y up").
      const du1 = T[b * 2] - T[a * 2];
      const dv1 = -(T[b * 2 + 1] - T[a * 2 + 1]);
      const du2 = T[c * 2] - T[a * 2];
      const dv2 = -(T[c * 2 + 1] - T[a * 2 + 1]);
      const det = du1 * dv2 - du2 * dv1;
      if (Math.abs(det) < 1e-20) continue;
      const r = 1 / det;
      const sdir = scale(sub(scale(e1, dv2), scale(e2, dv1)), r);
      const tdir = scale(sub(scale(e2, du1), scale(e1, du2)), r);
      for (const v of [a, b, c]) {
        for (let k = 0; k < 3; k++) {
          tan[v * 3 + k] += sdir[k];
          bit[v * 3 + k] += tdir[k];
        }
      }
    }
    const out = new Float32Array(nv * 4);
    for (let v = 0; v < nv; v++) {
      const n: Vec3 = [N[v * 3], N[v * 3 + 1], N[v * 3 + 2]];
      let t: Vec3 = [tan[v * 3], tan[v * 3 + 1], tan[v * 3 + 2]];
      t = sub(t, scale(n, dot(n, t)));
      if (length(t) < 1e-9) {
        // No usable UV gradient: any unit vector perpendicular to n.
        const ref: Vec3 = Math.abs(n[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
        t = cross(ref, n);
      }
      t = normalize(t);
      const b: Vec3 = [bit[v * 3], bit[v * 3 + 1], bit[v * 3 + 2]];
      const w = dot(cross(n, t), b) < 0 ? -1 : 1;
      out[v * 4] = t[0];
      out[v * 4 + 1] = t[1];
      out[v * 4 + 2] = t[2];
      out[v * 4 + 3] = w;
    }
    return out;
  }
}

// ---------------------------------------------------------------------------
// UV mapping
// ---------------------------------------------------------------------------

/**
 * 'fit': the face spans the texture rectangle `rect` = [u0, v0, u1, v1]
 *        (default the whole texture); `flipU`/`flipV` mirror it.
 * 'world': planar mapping, `tile` metres per texture repeat (REPEAT sampler).
 */
export type UvMap =
  | { kind: 'fit'; rect?: [number, number, number, number]; flipU?: boolean; flipV?: boolean }
  | { kind: 'world'; tile: number; offset?: Vec2 };

export interface FaceSpec {
  mb: MeshBuilder;
  uv?: UvMap;
}

export type FaceName = 'px' | 'nx' | 'py' | 'ny' | 'pz' | 'nz';

/**
 * Box face frames: outward normal, image-right (u) and image-down (v)
 * directions, chosen so that every face reads un-mirrored from outside
 * (u x v = -n).
 */
export const BOX_FACES: Record<FaceName, { n: Vec3; u: Vec3; v: Vec3 }> = {
  px: { n: [1, 0, 0], u: [0, 0, -1], v: [0, -1, 0] },
  nx: { n: [-1, 0, 0], u: [0, 0, 1], v: [0, -1, 0] },
  py: { n: [0, 1, 0], u: [1, 0, 0], v: [0, 0, 1] },
  ny: { n: [0, -1, 0], u: [1, 0, 0], v: [0, 0, -1] },
  pz: { n: [0, 0, 1], u: [1, 0, 0], v: [0, -1, 0] },
  nz: { n: [0, 0, -1], u: [-1, 0, 0], v: [0, -1, 0] },
};

function mapUv(map: UvMap | undefined, s: number, t: number, local: Vec3, fu: Vec3, fv: Vec3): Vec2 {
  if (!map || map.kind === 'fit') {
    let a = s;
    let b = t;
    if (map?.flipU) a = 1 - a;
    if (map?.flipV) b = 1 - b;
    const r = map?.rect ?? [0, 0, 1, 1];
    return [r[0] + a * (r[2] - r[0]), r[1] + b * (r[3] - r[1])];
  }
  const o = map.offset ?? [0, 0];
  return [dot(local, fu) / map.tile + o[0], dot(local, fv) / map.tile + o[1]];
}

// ---------------------------------------------------------------------------
// Primitives
// ---------------------------------------------------------------------------

/** Axis-aligned (in placement space) box; only the faces listed are emitted. */
export function box(
  m: Mat4,
  center: Vec3,
  size: Vec3,
  faces: Partial<Record<FaceName, FaceSpec>>,
): void {
  const h: Vec3 = [size[0] / 2, size[1] / 2, size[2] / 2];
  for (const name of Object.keys(faces) as FaceName[]) {
    const spec = faces[name];
    if (!spec) continue;
    const f = BOX_FACES[name];
    const axisN = f.n.findIndex((x) => x !== 0);
    const axisU = f.u.findIndex((x) => x !== 0);
    const axisV = f.v.findIndex((x) => x !== 0);
    const fc = add(center, scale(f.n, h[axisN]));
    const hu = scale(f.u, h[axisU]);
    const hv = scale(f.v, h[axisV]);
    const corners: [Vec3, Vec3, Vec3, Vec3] = [
      sub(sub(fc, hu), hv), // s=0,t=0
      sub(add(fc, hu), hv), // s=1,t=0
      add(add(fc, hu), hv), // s=1,t=1
      add(sub(fc, hu), hv), // s=0,t=1
    ];
    const st: Vec2[] = [[0, 0], [1, 0], [1, 1], [0, 1]];
    const uv = corners.map((c, i) => mapUv(spec.uv, st[i][0], st[i][1], c, f.u, f.v)) as [Vec2, Vec2, Vec2, Vec2];
    spec.mb.quad(m, corners, f.n, uv);
  }
}

/** Convenience: every face of the box into one builder with world-scale UVs. */
export function solidBox(m: Mat4, mb: MeshBuilder, center: Vec3, size: Vec3, tile = 1): void {
  const spec: FaceSpec = { mb, uv: { kind: 'world', tile } };
  box(m, center, size, { px: spec, nx: spec, py: spec, ny: spec, pz: spec, nz: spec });
}

export interface CylinderOpts {
  /** Side surface. */
  side?: MeshBuilder;
  /** Cap at y = y0 + height (normal +Y). */
  top?: MeshBuilder;
  /** Cap at y = y0 (normal -Y). */
  bottom?: MeshBuilder;
  radius: number;
  /** Frustum: radius at the top (defaults to `radius`). */
  radiusTop?: number;
  height: number;
  y0?: number;
  segments?: number;
  /** Side UV: u wraps once around (or `uRepeat` times), v spans the height. */
  uRepeat?: number;
  /** Render the side with outward normals flipped inwards (e.g. a baffle interior). */
  inward?: boolean;
}

/** Cylinder / frustum along local +Y from y0 to y0 + height. */
export function cylinder(m: Mat4, o: CylinderOpts): void {
  const seg = o.segments ?? 32;
  const r0 = o.radius;
  const r1 = o.radiusTop ?? o.radius;
  const y0 = o.y0 ?? 0;
  const y1 = y0 + o.height;
  const ur = o.uRepeat ?? 1;
  if (o.side) {
    const mb = o.side;
    // Slope of the frustum side: normal = (cos, (r0 - r1)/h, sin) normalised.
    const ny = (r0 - r1) / o.height;
    const base = mb.vertexCount;
    for (let i = 0; i <= seg; i++) {
      const a = (i / seg) * Math.PI * 2;
      const c = Math.cos(a);
      const s = Math.sin(a);
      let n = normalize([c, ny, s]);
      if (o.inward) n = scale(n, -1);
      mb.vertex(m, [r0 * c, y0, r0 * s], n, [(i / seg) * ur, 1]);
      mb.vertex(m, [r1 * c, y1, r1 * s], n, [(i / seg) * ur, 0]);
    }
    for (let i = 0; i < seg; i++) {
      const a = base + i * 2;
      mb.tri(a, a + 2, a + 3);
      mb.tri(a, a + 3, a + 1);
    }
  }
  const cap = (mb: MeshBuilder, y: number, r: number, ny: number): void => {
    if (r <= 0) return;
    const c0 = mb.vertex(m, [0, y, 0], [0, ny, 0], [0.5, 0.5]);
    const ring: number[] = [];
    for (let i = 0; i <= seg; i++) {
      const a = (i / seg) * Math.PI * 2;
      const c = Math.cos(a);
      const s = Math.sin(a);
      ring.push(mb.vertex(m, [r * c, y, r * s], [0, ny, 0], [0.5 + 0.5 * c, 0.5 + 0.5 * s * ny]));
    }
    for (let i = 0; i < seg; i++) mb.tri(c0, ring[i], ring[i + 1]);
  };
  if (o.top) cap(o.top, y1, r1, 1);
  if (o.bottom) cap(o.bottom, y0, r0, -1);
}

/** Flat annulus in the local XZ plane at height y, facing +Y (or -Y). */
export function annulus(m: Mat4, mb: MeshBuilder, rIn: number, rOut: number, y: number, up = true, segments = 32): void {
  const n: Vec3 = [0, up ? 1 : -1, 0];
  const base = mb.vertexCount;
  for (let i = 0; i <= segments; i++) {
    const a = (i / segments) * Math.PI * 2;
    const c = Math.cos(a);
    const s = Math.sin(a);
    mb.vertex(m, [rIn * c, y, rIn * s], n, [0.5 + 0.5 * (rIn / rOut) * c, 0.5 + 0.5 * (rIn / rOut) * s]);
    mb.vertex(m, [rOut * c, y, rOut * s], n, [0.5 + 0.5 * c, 0.5 + 0.5 * s]);
  }
  for (let i = 0; i < segments; i++) {
    const a = base + i * 2;
    mb.tri(a, a + 1, a + 3);
    mb.tri(a, a + 3, a + 2);
  }
}

/**
 * Spherical (optionally flattened) cap centred on the local origin, pole at
 * +Y: polar angle 0..thetaMax. `heightScale` squashes it into a spheroid.
 */
export function dome(
  m: Mat4,
  mb: MeshBuilder,
  radius: number,
  thetaMax = Math.PI / 2,
  heightScale = 1,
  segments = 32,
  rings = 12,
): void {
  const ry = radius * heightScale;
  const grid: number[][] = [];
  for (let j = 0; j <= rings; j++) {
    const th = (j / rings) * thetaMax;
    const row: number[] = [];
    for (let i = 0; i <= segments; i++) {
      const ph = (i / segments) * Math.PI * 2;
      const x = radius * Math.sin(th) * Math.cos(ph);
      const z = radius * Math.sin(th) * Math.sin(ph);
      const y = ry * Math.cos(th);
      // Ellipsoid normal: gradient of x²/r² + y²/ry² + z²/r².
      const n = j === 0 ? ([0, 1, 0] as Vec3) : normalize([x / (radius * radius), y / (ry * ry), z / (radius * radius)]);
      const uvR = 0.5 * (th / thetaMax);
      row.push(mb.vertex(m, [x, y, z], n, [0.5 + uvR * Math.cos(ph), 0.5 + uvR * Math.sin(ph)]));
    }
    grid.push(row);
  }
  for (let j = 0; j < rings; j++) {
    for (let i = 0; i < segments; i++) {
      const a = grid[j][i];
      const b = grid[j][i + 1];
      const c = grid[j + 1][i + 1];
      const d = grid[j + 1][i];
      if (j > 0) mb.tri(a, b, c);
      mb.tri(a, c, d);
    }
  }
}

/**
 * Paraboloid dish, vertex at the local origin, opening towards +Y:
 * y = r² / (4 f) for r <= radius. Single surface with normals on the concave
 * side; use a double-sided material.
 */
export function dish(m: Mat4, mb: MeshBuilder, radius: number, depth: number, segments = 40, rings = 10): void {
  const f = (radius * radius) / (4 * depth);
  const grid: number[][] = [];
  for (let j = 0; j <= rings; j++) {
    const r = (j / rings) * radius;
    const row: number[] = [];
    for (let i = 0; i <= segments; i++) {
      const ph = (i / segments) * Math.PI * 2;
      const x = r * Math.cos(ph);
      const z = r * Math.sin(ph);
      const y = (r * r) / (4 * f);
      const n = normalize([-x / (2 * f), 1, -z / (2 * f)]);
      row.push(mb.vertex(m, [x, y, z], n, [0.5 + 0.5 * (r / radius) * Math.cos(ph), 0.5 + 0.5 * (r / radius) * Math.sin(ph)]));
    }
    grid.push(row);
  }
  for (let j = 0; j < rings; j++) {
    for (let i = 0; i < segments; i++) {
      const a = grid[j][i];
      const b = grid[j][i + 1];
      const c = grid[j + 1][i + 1];
      const d = grid[j + 1][i];
      if (j > 0) mb.tri(a, b, c);
      mb.tri(a, c, d);
    }
  }
}

/** Placement for a primitive built along +Y that should run from a to b. */
export function rodPlacement(base: Mat4, a: Vec3, b: Vec3): { m: Mat4; length: number } {
  const d = sub(b, a);
  const len = length(d);
  const y = normalize(d);
  const ref: Vec3 = Math.abs(y[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
  const z = normalize(cross(ref, y));
  const x = cross(y, z);
  const local = [x[0], x[1], x[2], 0, y[0], y[1], y[2], 0, z[0], z[1], z[2], 0, a[0], a[1], a[2], 1];
  return { m: mul(base, local), length: len };
}

/** Round rod (with end caps) from a to b. */
export function rod(base: Mat4, mb: MeshBuilder, a: Vec3, b: Vec3, radius: number, segments = 12): void {
  const { m, length: len } = rodPlacement(base, a, b);
  cylinder(m, { side: mb, top: mb, bottom: mb, radius, height: len, segments, uRepeat: 1 });
}
