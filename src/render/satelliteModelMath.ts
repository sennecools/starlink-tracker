/**
 * Pure math for the close-up satellite models (SatelliteModels.ts) and the
 * GPU candidate search in SatellitePoints.ts. No Cesium import, so it runs in
 * Node tests.
 *
 * Frames
 * ------
 * Cesium converts glTF (Y-up, +Z forward) to its model frame (Z-up, +X
 * forward) with ModelUtility.getAxisCorrectionMatrix(upAxis = Y, forwardAxis
 * = Z) = Axis.Y_UP_TO_Z_UP * Axis.Z_UP_TO_X_UP (Model defaults). Net mapping:
 *
 *      glTF +Z (forward) -> model +X
 *      glTF +X           -> model +Y
 *      glTF +Y (up)      -> model +Z
 *
 * The satellite's model matrix built here (column-major, like Cesium Matrix4):
 *
 *      col0 = x̂ = flight direction (inertial velocity, made perpendicular to the radial)
 *      col1 = ŷ = ẑ × x̂ (orbit-normal side)
 *      col2 = ẑ = radial up (zenith), so model -Z points to nadir
 *      col3 = ECEF position (metres)
 *
 * so in glTF terms: +Z flies forward, +Y points to zenith (phased-array /
 * antenna face on -Y looks at the Earth), ±X is the solar-wing span axis.
 *
 * The shipped models (public/models/README.md) are authored X-forward instead
 * (glTF +X = flight, +Y = zenith, wings ±Z) and loaded with
 * `forwardAxis: Axis.X`, which drops Z_UP_TO_X_UP: glTF +X -> model +X,
 * +Y -> +Z, +Z -> -Y. The same attitude matrix then flies them nose-first;
 * use sunInGltfBodyXForward and wing axis glTF +Z for those.
 */

/** Earth rotation rate (rad/s, IERS). */
export const EARTH_ROTATION_RAD_S = 7.292115e-5;
/** Radius of the cylindrical Earth shadow (mean Earth radius, metres). */
export const EARTH_SHADOW_RADIUS_M = 6_371_000;

/** Anything indexable like a Cesium Matrix4 / Float64Array / number[]. */
export interface Mat4Like {
  [index: number]: number;
}

/**
 * Writes the attitude + position model matrix (column-major 4x4) for a
 * satellite at ECEF position p with ECEF velocity v (m/s).
 *
 * The ECEF velocity is turned into the inertial velocity expressed in ECEF
 * axes (v + ω × r) first: Starlink flies along its inertial velocity, which
 * differs ~4° from the Earth-relative one. Returns false (out untouched) when
 * the position is invalid. A degenerate velocity falls back to an arbitrary
 * horizontal forward axis.
 */
export function attitudeMatrix(
  px: number,
  py: number,
  pz: number,
  vx: number,
  vy: number,
  vz: number,
  out: Mat4Like,
): boolean {
  const r = Math.sqrt(px * px + py * py + pz * pz);
  if (!(r > 1)) return false; // also NaN
  const zx = px / r;
  const zy = py / r;
  const zz = pz / r;

  // Inertial velocity in ECEF axes: v + ω × r, ω = (0, 0, Ω).
  let fx = vx - EARTH_ROTATION_RAD_S * py;
  let fy = vy + EARTH_ROTATION_RAD_S * px;
  let fz = vz;
  // Remove the radial component.
  const d = fx * zx + fy * zy + fz * zz;
  fx -= d * zx;
  fy -= d * zy;
  fz -= d * zz;
  let fl = Math.sqrt(fx * fx + fy * fy + fz * fz);
  if (!(fl > 1e-6)) {
    // Degenerate: pick the horizontal direction towards the least aligned axis.
    const ax = Math.abs(zx);
    const ay = Math.abs(zy);
    const az = Math.abs(zz);
    let ux = 0;
    let uy = 0;
    let uz = 0;
    if (az <= ax && az <= ay) uz = 1;
    else if (ay <= ax) uy = 1;
    else ux = 1;
    // f = u × z ... any perpendicular works.
    fx = uy * zz - uz * zy;
    fy = uz * zx - ux * zz;
    fz = ux * zy - uy * zx;
    fl = Math.sqrt(fx * fx + fy * fy + fz * fz);
  }
  const xx = fx / fl;
  const xy = fy / fl;
  const xz = fz / fl;
  // ŷ = ẑ × x̂ (unit, since ẑ ⟂ x̂).
  const yx = zy * xz - zz * xy;
  const yy = zz * xx - zx * xz;
  const yz = zx * xy - zy * xx;

  out[0] = xx;
  out[1] = xy;
  out[2] = xz;
  out[3] = 0;
  out[4] = yx;
  out[5] = yy;
  out[6] = yz;
  out[7] = 0;
  out[8] = zx;
  out[9] = zy;
  out[10] = zz;
  out[11] = 0;
  out[12] = px;
  out[13] = py;
  out[14] = pz;
  out[15] = 1;
  return true;
}

/**
 * Writes a uniformly scaled translation-only matrix (used to "warm up" a model
 * invisibly in front of the camera so its shaders compile before first use).
 */
export function scaledTranslation(px: number, py: number, pz: number, scale: number, out: Mat4Like): void {
  for (let i = 0; i < 16; i++) out[i] = 0;
  out[0] = scale;
  out[5] = scale;
  out[10] = scale;
  out[12] = px;
  out[13] = py;
  out[14] = pz;
  out[15] = 1;
}

/** A 3-vector as a plain tuple (glTF body frame). */
export type Vec3 = readonly [number, number, number];

/**
 * Sun direction expressed in the glTF body frame of a satellite whose model
 * matrix is `m` (attitudeMatrix()): glTF (X, Y, Z) = model (ŷ, ẑ, x̂).
 * Writes 3 numbers into `out`.
 */
export function sunInGltfBody(m: Mat4Like, sx: number, sy: number, sz: number, out: Mat4Like): void {
  out[0] = sx * m[4] + sy * m[5] + sz * m[6]; // s·ŷ
  out[1] = sx * m[8] + sy * m[9] + sz * m[10]; // s·ẑ
  out[2] = sx * m[0] + sy * m[1] + sz * m[2]; // s·x̂
}

/**
 * Same as sunInGltfBody, for models loaded with `forwardAxis: Axis.X` (the
 * convention of public/models/*.glb: glTF +X = flight direction, +Y = zenith,
 * solar wings along ±Z). Cesium then only applies Axis.Y_UP_TO_Z_UP:
 * glTF (x, y, z) -> model (x, -z, y), i.e. glTF (X, Y, Z) = model (x̂, ẑ, -ŷ).
 */
export function sunInGltfBodyXForward(m: Mat4Like, sx: number, sy: number, sz: number, out: Mat4Like): void {
  out[0] = sx * m[0] + sy * m[1] + sz * m[2]; // s·x̂
  out[1] = sx * m[8] + sy * m[9] + sz * m[10]; // s·ẑ
  out[2] = -(sx * m[4] + sy * m[5] + sz * m[6]); // -s·ŷ
}

/**
 * One-degree-of-freedom sun tracking of a solar wing.
 *
 * The wing rotates about the unit `axis` (glTF body frame, default +X = the
 * span axis); at angle 0 its active cell face points along `restNormal`
 * (⟂ axis, default glTF +Y = zenith). Rotating by θ (right hand) turns the
 * normal into n0·cos θ + (axis × n0)·sin θ, so the angle that maximises
 * normal·sun is atan2(s·(axis × n0), s·n0). `s` = sun in the glTF body frame.
 */
export function wingSunAngle(s: ArrayLike<number>, axis: Vec3 = [1, 0, 0], restNormal: Vec3 = [0, 1, 0]): number {
  const [ax, ay, az] = axis;
  const [nx, ny, nz] = restNormal;
  const bx = ay * nz - az * ny;
  const by = az * nx - ax * nz;
  const bz = ax * ny - ay * nx;
  return Math.atan2(s[0] * bx + s[1] * by + s[2] * bz, s[0] * nx + s[1] * ny + s[2] * nz);
}

/**
 * Wing pivot node matrix: the node's original local transform with an extra
 * rotation of θ about the unit `axis` of the PARENT frame, through the
 * pivot's origin:
 *     out = T(orig) · R(axis, θ) · RS(orig)
 * (rotation/scale columns pre-multiplied by R, translation kept). Works for
 * pivots with any own rotation (e.g. a left wing mirrored by 180°), as long
 * as the parent frame is the unrotated glTF body.
 */
export function wingNodeMatrix(orig: ArrayLike<number>, theta: number, out: Mat4Like, axis: Vec3 = [1, 0, 0]): void {
  const [x, y, z] = axis;
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  const t = 1 - c;
  // Rodrigues rotation matrix (row-major r_ij).
  const r00 = t * x * x + c;
  const r01 = t * x * y - s * z;
  const r02 = t * x * z + s * y;
  const r10 = t * x * y + s * z;
  const r11 = t * y * y + c;
  const r12 = t * y * z - s * x;
  const r20 = t * x * z - s * y;
  const r21 = t * y * z + s * x;
  const r22 = t * z * z + c;
  for (let col = 0; col < 3; col++) {
    const o = col * 4;
    const a = orig[o];
    const b = orig[o + 1];
    const d = orig[o + 2];
    out[o] = r00 * a + r01 * b + r02 * d;
    out[o + 1] = r10 * a + r11 * b + r12 * d;
    out[o + 2] = r20 * a + r21 * b + r22 * d;
    out[o + 3] = orig[o + 3];
  }
  out[12] = orig[12];
  out[13] = orig[13];
  out[14] = orig[14];
  out[15] = orig[15];
}

/**
 * Cylindrical Earth-shadow model with a soft edge: 1 = full sun, 0 = umbra.
 * (px,py,pz) ECEF position in metres, (sx,sy,sz) unit sun direction.
 * `softM` is the half-width of the smooth transition around the shadow radius.
 */
export function sunlitFactor(
  px: number,
  py: number,
  pz: number,
  sx: number,
  sy: number,
  sz: number,
  radiusM = EARTH_SHADOW_RADIUS_M,
  softM = 25_000,
): number {
  const along = px * sx + py * sy + pz * sz;
  if (!(along < 0)) return 1; // sun side (or NaN -> lit)
  const r2 = px * px + py * py + pz * pz;
  const perp = Math.sqrt(Math.max(0, r2 - along * along));
  const lo = radiusM - softM;
  const hi = radiusM + softM;
  if (perp <= lo) return 0;
  if (perp >= hi) return 1;
  const f = (perp - lo) / (hi - lo);
  return f * f * (3 - 2 * f);
}

// ---------------------------------------------------------------------------
// Candidate search
// ---------------------------------------------------------------------------

/**
 * Inserts (index, d2) into the ascending-by-d2 list held in idx/dist[0..n),
 * keeping at most `cap` entries. Returns the new length. Allocation-free.
 */
export function insertNearest(
  index: number,
  d2: number,
  idx: Int32Array,
  dist: Float64Array,
  n: number,
  cap: number,
): number {
  if (cap <= 0) return 0;
  if (n >= cap && !(d2 < dist[n - 1])) return n;
  let j = n < cap ? n : cap - 1;
  while (j > 0 && dist[j - 1] > d2) {
    idx[j] = idx[j - 1];
    dist[j] = dist[j - 1];
    j--;
  }
  idx[j] = index;
  dist[j] = d2;
  return n < cap ? n + 1 : n;
}

/**
 * Linear scan over interpolated positions (a·wa + b·wb, ECEF metres, 3 per
 * satellite, NaN = invalid) for the satellites within `radius` of (cx,cy,cz).
 * Satellites of hidden shells (shellVisible[shell[i]] !== 1) are skipped
 * unless i === always. The nearest `outIdx.length` results are written
 * nearest-first (squared distances in outD2); the count is returned.
 * ~10k satellites cost ~20-40 µs (per-axis early reject, no sqrt).
 */
export function scanNearest(
  a: Float64Array,
  b: Float64Array,
  wa: number,
  wb: number,
  count: number,
  cx: number,
  cy: number,
  cz: number,
  radius: number,
  shell: Uint8Array,
  shellVisible: Uint8Array,
  always: number,
  outIdx: Int32Array,
  outD2: Float64Array,
): number {
  const cap = Math.min(outIdx.length, outD2.length);
  const r2 = radius * radius;
  let n = 0;
  const blend = wb !== 0;
  for (let i = 0; i < count; i++) {
    const i3 = i * 3;
    let x = a[i3];
    if (blend) x = x * wa + b[i3] * wb;
    const dx = x - cx;
    if (!(dx <= radius && dx >= -radius)) continue; // also NaN
    let y = a[i3 + 1];
    if (blend) y = y * wa + b[i3 + 1] * wb;
    const dy = y - cy;
    if (!(dy <= radius && dy >= -radius)) continue;
    let z = a[i3 + 2];
    if (blend) z = z * wa + b[i3 + 2] * wb;
    const dz = z - cz;
    if (!(dz <= radius && dz >= -radius)) continue;
    const d2 = dx * dx + dy * dy + dz * dz;
    if (!(d2 <= r2)) continue;
    if (shellVisible[shell[i]] !== 1 && i !== always) continue;
    n = insertNearest(i, d2, outIdx, outD2, n, cap);
  }
  return n;
}

/**
 * Picks which satellites get a model, with hysteresis: `found` holds the
 * candidates within the (larger) hide radius, nearest first. A satellite is
 * taken if it is within the show radius, or already shown (shown[i] === 1).
 * `forced` (>= 0) is always taken first (caller checks its range). At most
 * `max` entries are written to `out`; returns the count.
 */
export function selectCandidates(
  foundIdx: Int32Array,
  foundD2: Float64Array,
  n: number,
  shown: Uint8Array,
  forced: number,
  showR2: number,
  max: number,
  out: Int32Array,
): number {
  const cap = Math.min(max, out.length);
  let k = 0;
  if (forced >= 0 && cap > 0) out[k++] = forced;
  for (let j = 0; j < n && k < cap; j++) {
    const i = foundIdx[j];
    if (i === forced) continue;
    if (foundD2[j] <= showR2 || shown[i] === 1) out[k++] = i;
  }
  return k;
}

/** The parts of a model-pool slot that eviction looks at. */
export interface PoolSlotState {
  file: string;
  /** Satellite shown, -1 = idle. */
  sat: number;
  /** Warm-up frames left (0 = ready and warmed up). */
  warm: number;
  /** Null while the instance is not created yet / was dropped. */
  model: unknown;
}

/**
 * Index of an idle, warmed-up slot of a file other than `keepFile` that may be
 * destroyed to make room in a pool that reached its global cap (so a dense
 * cluster of one bus seen earlier cannot starve close-ups of another bus for
 * the rest of the session), or -1. Never a file's last instance: every file
 * keeps its warmed-up preload. Newest slots first.
 */
export function pickEvictableSlot(slots: readonly PoolSlotState[], keepFile: string): number {
  for (let k = slots.length - 1; k >= 0; k--) {
    const s = slots[k]!;
    if (s.file === keepFile || s.sat >= 0 || s.warm !== 0 || !s.model) continue;
    let same = 0;
    for (const o of slots) if (o.file === s.file) same++;
    if (same >= 2) return k;
  }
  return -1;
}
