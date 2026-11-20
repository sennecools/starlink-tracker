/**
 * Pure math for the camera rig (no Cesium, no DOM): vectors, quaternions,
 * WGS-84 helpers, rig pose, grab / zoom-to-cursor, smoothing and limits.
 *
 * Conventions
 * - ECEF metres, doubles everywhere.
 * - A rig "frame" is a unit quaternion q mapping local axes to ECEF:
 *     q·x̂ = right, q·ŷ = forward (horizontal look direction), q·ẑ = up.
 *   In globe mode up is the geodetic surface normal at the look target, so
 *   the frame IS the "unit target vector + parallel-transported heading
 *   reference" pair: rotating it by world rotations (grab, inertia) is
 *   parallel transport along the rotation, with no lat/lon anywhere — dragging
 *   straight over a pole is as smooth as anywhere else.
 * - pitch: radians, -π/2 = looking straight down, 0 = horizontal.
 *   dir = cos(p)·forward + sin(p)·up;  camUp = -sin(p)·forward + cos(p)·up.
 * - heading: increasing heading turns `forward` clockwise seen from above
 *   (north -> east), implemented as q <- q·Rz(-Δ).
 * - Screen positions are NDC: x right, y up, both in [-1, 1].
 *
 * All functions write into caller-provided `out` objects; module-level
 * scratch vectors are used internally (single-threaded, not re-entrant).
 */

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export interface Quat {
  x: number;
  y: number;
  z: number;
  w: number;
}

/** Ellipsoid of revolution: equatorial radius a, polar radius b (metres). */
export interface Ellipsoid {
  a: number;
  b: number;
}

export const WGS84: Readonly<Ellipsoid> = { a: 6_378_137, b: 6_356_752.314245179 };
export const DEG = Math.PI / 180;
export const HALF_PI = Math.PI / 2;

/** Camera pose (unit dir/up/right, orthonormal; right = dir × up). */
export interface Pose {
  pos: Vec3;
  dir: Vec3;
  up: Vec3;
  right: Vec3;
}

/** Tangents of the half field of view (x: horizontal, y: vertical). */
export interface Lens {
  tanX: number;
  tanY: number;
}

/** Globe-mode controller state. */
export interface RigState {
  q: Quat;
  /** Distance camera -> look point, metres. */
  range: number;
  pitch: number;
  /** Height of the look point above the ellipsoid (terrain under the target), metres. */
  h: number;
}

/** A resolved view: frame + explicit look point (used for flights and track mode). */
export interface RigView {
  q: Quat;
  look: Vec3;
  range: number;
  pitch: number;
}

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

export function vec3(x = 0, y = 0, z = 0): Vec3 {
  return { x, y, z };
}

export function quat(): Quat {
  return { x: 0, y: 0, z: 0, w: 1 };
}

export function pose(): Pose {
  return { pos: vec3(), dir: vec3(0, 0, -1), up: vec3(0, 1, 0), right: vec3(1, 0, 0) };
}

export function rigState(): RigState {
  return { q: quat(), range: 1e7, pitch: -HALF_PI, h: 0 };
}

export function rigView(): RigView {
  return { q: quat(), look: vec3(), range: 1e7, pitch: -HALF_PI };
}

export function copyRig(src: RigState, dst: RigState): RigState {
  copyQuat(src.q, dst.q);
  dst.range = src.range;
  dst.pitch = src.pitch;
  dst.h = src.h;
  return dst;
}

export function copyView(src: RigView, dst: RigView): RigView {
  copyQuat(src.q, dst.q);
  copyVec(src.look, dst.look);
  dst.range = src.range;
  dst.pitch = src.pitch;
  return dst;
}

// ---------------------------------------------------------------------------
// Vectors
// ---------------------------------------------------------------------------

export function setVec(out: Vec3, x: number, y: number, z: number): Vec3 {
  out.x = x;
  out.y = y;
  out.z = z;
  return out;
}

export function copyVec(a: Vec3, out: Vec3): Vec3 {
  out.x = a.x;
  out.y = a.y;
  out.z = a.z;
  return out;
}

export function dot(a: Vec3, b: Vec3): number {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

export function len(a: Vec3): number {
  return Math.sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}

export function cross(a: Vec3, b: Vec3, out: Vec3): Vec3 {
  const x = a.y * b.z - a.z * b.y;
  const y = a.z * b.x - a.x * b.z;
  const z = a.x * b.y - a.y * b.x;
  out.x = x;
  out.y = y;
  out.z = z;
  return out;
}

export function normalize(a: Vec3, out: Vec3): Vec3 {
  const l = len(a);
  if (l > 0) {
    out.x = a.x / l;
    out.y = a.y / l;
    out.z = a.z / l;
  } else {
    out.x = 0;
    out.y = 0;
    out.z = 1;
  }
  return out;
}

/** out = a + b·s */
export function addScaled(a: Vec3, b: Vec3, s: number, out: Vec3): Vec3 {
  out.x = a.x + b.x * s;
  out.y = a.y + b.y * s;
  out.z = a.z + b.z * s;
  return out;
}

export function scale(a: Vec3, s: number, out: Vec3): Vec3 {
  out.x = a.x * s;
  out.y = a.y * s;
  out.z = a.z * s;
  return out;
}

export function distance(a: Vec3, b: Vec3): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  const dz = a.z - b.z;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/** Angle between two (not necessarily unit) vectors, radians, robust near 0 and π. */
export function angleBetween(a: Vec3, b: Vec3): number {
  const cx = a.y * b.z - a.z * b.y;
  const cy = a.z * b.x - a.x * b.z;
  const cz = a.x * b.y - a.y * b.x;
  return Math.atan2(Math.sqrt(cx * cx + cy * cy + cz * cz), dot(a, b));
}

/** Removes the component of `v` along unit `n` and normalises; false if degenerate. */
export function projectOnPlane(v: Vec3, n: Vec3, out: Vec3): boolean {
  const d = dot(v, n);
  const x = v.x - n.x * d;
  const y = v.y - n.y * d;
  const z = v.z - n.z * d;
  const l = Math.sqrt(x * x + y * y + z * z);
  if (!(l > 1e-9)) return false;
  out.x = x / l;
  out.y = y / l;
  out.z = z / l;
  return true;
}

/** Spherical interpolation of the directions of a and b, with linearly interpolated length. */
export function slerpPoint(a: Vec3, b: Vec3, t: number, out: Vec3): Vec3 {
  const la = len(a);
  const lb = len(b);
  const r = la + (lb - la) * t;
  const theta = angleBetween(a, b);
  let wa: number;
  let wb: number;
  if (theta < 1e-6) {
    wa = (1 - t) / la;
    wb = t / lb;
  } else {
    const s = Math.sin(theta);
    wa = Math.sin((1 - t) * theta) / s / la;
    wb = Math.sin(t * theta) / s / lb;
  }
  const x = a.x * wa + b.x * wb;
  const y = a.y * wa + b.y * wb;
  const z = a.z * wa + b.z * wb;
  const l = Math.sqrt(x * x + y * y + z * z) || 1;
  out.x = (x / l) * r;
  out.y = (y / l) * r;
  out.z = (z / l) * r;
  return out;
}

// ---------------------------------------------------------------------------
// Quaternions
// ---------------------------------------------------------------------------

export function copyQuat(a: Quat, out: Quat): Quat {
  out.x = a.x;
  out.y = a.y;
  out.z = a.z;
  out.w = a.w;
  return out;
}

export function quatNormalize(q: Quat, out: Quat): Quat {
  const l = Math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
  if (l > 0) {
    out.x = q.x / l;
    out.y = q.y / l;
    out.z = q.z / l;
    out.w = q.w / l;
  } else {
    out.x = 0;
    out.y = 0;
    out.z = 0;
    out.w = 1;
  }
  return out;
}

/** out = a·b (apply b first, then a). Safe when out aliases a or b. */
export function quatMul(a: Quat, b: Quat, out: Quat): Quat {
  const x = a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y;
  const y = a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x;
  const z = a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w;
  const w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z;
  out.x = x;
  out.y = y;
  out.z = z;
  out.w = w;
  return out;
}

/** Rotates v by unit quaternion q. Safe when out aliases v. */
export function quatRotate(q: Quat, v: Vec3, out: Vec3): Vec3 {
  // t = 2·(q.xyz × v); v' = v + w·t + q.xyz × t
  const tx = 2 * (q.y * v.z - q.z * v.y);
  const ty = 2 * (q.z * v.x - q.x * v.z);
  const tz = 2 * (q.x * v.y - q.y * v.x);
  const x = v.x + q.w * tx + (q.y * tz - q.z * ty);
  const y = v.y + q.w * ty + (q.z * tx - q.x * tz);
  const z = v.z + q.w * tz + (q.x * ty - q.y * tx);
  out.x = x;
  out.y = y;
  out.z = z;
  return out;
}

export function quatFromAxisAngle(ax: number, ay: number, az: number, angle: number, out: Quat): Quat {
  const l = Math.sqrt(ax * ax + ay * ay + az * az);
  if (!(l > 0) || angle === 0) {
    out.x = 0;
    out.y = 0;
    out.z = 0;
    out.w = 1;
    return out;
  }
  const s = Math.sin(angle / 2) / l;
  out.x = ax * s;
  out.y = ay * s;
  out.z = az * s;
  out.w = Math.cos(angle / 2);
  return out;
}

/** Rotation vector (axis · angle, radians) -> quaternion. */
export function quatFromRotationVector(v: Vec3, out: Quat): Quat {
  const angle = len(v);
  if (angle < 1e-12) {
    out.x = v.x / 2;
    out.y = v.y / 2;
    out.z = v.z / 2;
    out.w = 1;
    return quatNormalize(out, out);
  }
  return quatFromAxisAngle(v.x, v.y, v.z, angle, out);
}

/** Quaternion -> rotation vector (axis · angle) with angle in [0, π]. */
export function quatToRotationVector(q: Quat, out: Vec3): Vec3 {
  let { x, y, z, w } = q;
  if (w < 0) {
    x = -x;
    y = -y;
    z = -z;
    w = -w;
  }
  const s = Math.sqrt(x * x + y * y + z * z);
  if (s < 1e-12) {
    out.x = 2 * x;
    out.y = 2 * y;
    out.z = 2 * z;
    return out;
  }
  const angle = 2 * Math.atan2(s, w);
  out.x = (x / s) * angle;
  out.y = (y / s) * angle;
  out.z = (z / s) * angle;
  return out;
}

/** Shortest rotation taking unit vector a to unit vector b. */
export function quatFromUnitVectors(a: Vec3, b: Vec3, out: Quat): Quat {
  const r = dot(a, b) + 1;
  if (r < 1e-12) {
    // 180°: any axis perpendicular to a.
    if (Math.abs(a.x) > Math.abs(a.z)) {
      out.x = -a.y;
      out.y = a.x;
      out.z = 0;
    } else {
      out.x = 0;
      out.y = -a.z;
      out.z = a.y;
    }
    out.w = 0;
  } else {
    out.x = a.y * b.z - a.z * b.y;
    out.y = a.z * b.x - a.x * b.z;
    out.z = a.x * b.y - a.y * b.x;
    out.w = r;
  }
  return quatNormalize(out, out);
}

/** Geodesic interpolation of unit quaternions (shortest path). Safe when out aliases a. */
export function quatSlerp(a: Quat, b: Quat, t: number, out: Quat): Quat {
  let bx = b.x;
  let by = b.y;
  let bz = b.z;
  let bw = b.w;
  let c = a.x * bx + a.y * by + a.z * bz + a.w * bw;
  if (c < 0) {
    c = -c;
    bx = -bx;
    by = -by;
    bz = -bz;
    bw = -bw;
  }
  let wa: number;
  let wb: number;
  if (c > 0.9999995) {
    wa = 1 - t;
    wb = t;
  } else {
    const theta = Math.acos(c);
    const s = Math.sin(theta);
    wa = Math.sin((1 - t) * theta) / s;
    wb = Math.sin(t * theta) / s;
  }
  out.x = a.x * wa + bx * wb;
  out.y = a.y * wa + by * wb;
  out.z = a.z * wa + bz * wb;
  out.w = a.w * wa + bw * wb;
  return quatNormalize(out, out);
}

/** Quaternion whose rotation matrix has columns (right, forward, up). Inputs must be orthonormal. */
export function quatFromBasis(r: Vec3, f: Vec3, u: Vec3, out: Quat): Quat {
  const m00 = r.x;
  const m10 = r.y;
  const m20 = r.z;
  const m01 = f.x;
  const m11 = f.y;
  const m21 = f.z;
  const m02 = u.x;
  const m12 = u.y;
  const m22 = u.z;
  const trace = m00 + m11 + m22;
  if (trace > 0) {
    const s = 0.5 / Math.sqrt(trace + 1);
    out.w = 0.25 / s;
    out.x = (m21 - m12) * s;
    out.y = (m02 - m20) * s;
    out.z = (m10 - m01) * s;
  } else if (m00 > m11 && m00 > m22) {
    const s = 2 * Math.sqrt(1 + m00 - m11 - m22);
    out.w = (m21 - m12) / s;
    out.x = 0.25 * s;
    out.y = (m01 + m10) / s;
    out.z = (m02 + m20) / s;
  } else if (m11 > m22) {
    const s = 2 * Math.sqrt(1 + m11 - m00 - m22);
    out.w = (m02 - m20) / s;
    out.x = (m01 + m10) / s;
    out.y = 0.25 * s;
    out.z = (m12 + m21) / s;
  } else {
    const s = 2 * Math.sqrt(1 + m22 - m00 - m11);
    out.w = (m10 - m01) / s;
    out.x = (m02 + m20) / s;
    out.y = (m12 + m21) / s;
    out.z = 0.25 * s;
  }
  return quatNormalize(out, out);
}

/** Frame axes of q: right = q·x̂, forward = q·ŷ, up = q·ẑ (any may be null). */
export function frameAxes(q: Quat, right: Vec3 | null, forward: Vec3 | null, up: Vec3 | null): void {
  const { x, y, z, w } = q;
  if (right) {
    right.x = 1 - 2 * (y * y + z * z);
    right.y = 2 * (x * y + w * z);
    right.z = 2 * (x * z - w * y);
  }
  if (forward) {
    forward.x = 2 * (x * y - w * z);
    forward.y = 1 - 2 * (x * x + z * z);
    forward.z = 2 * (y * z + w * x);
  }
  if (up) {
    up.x = 2 * (x * z + w * y);
    up.y = 2 * (y * z - w * x);
    up.z = 1 - 2 * (x * x + y * y);
  }
}

const sqA = quat();

/** World rotation (pre-multiply): q <- R·q. */
export function rotateFrameWorld(q: Quat, r: Quat): Quat {
  quatMul(r, q, q);
  return quatNormalize(q, q);
}

/** Heading change: q <- q·Rz(-Δ) (positive Δ turns forward clockwise seen from above). */
export function addHeading(q: Quat, delta: number): Quat {
  quatFromAxisAngle(0, 0, 1, -delta, sqA);
  quatMul(q, sqA, q);
  return quatNormalize(q, q);
}

const sfUp = vec3();
const sfTarget = vec3();

/**
 * Parallel-transports the frame so its up axis points along `upTarget`
 * (any length): minimal rotation from the current up to the new one.
 */
export function transportFrameTo(q: Quat, upTarget: Vec3): Quat {
  frameAxes(q, null, null, sfUp);
  normalize(upTarget, sfTarget);
  quatFromUnitVectors(sfUp, sfTarget, sqA);
  return rotateFrameWorld(q, sqA);
}

const sbR = vec3();
const sbF = vec3();

/**
 * Frame with the given up (unit) whose forward is `forwardHint` projected
 * onto the tangent plane (falls back to `fallbackHint`, then to north).
 */
export function frameFromUpForward(up: Vec3, forwardHint: Vec3, fallbackHint: Vec3 | null, out: Quat): Quat {
  if (!projectOnPlane(forwardHint, up, sbF)) {
    if (!fallbackHint || !projectOnPlane(fallbackHint, up, sbF)) {
      setVec(sbR, 0, 0, 1);
      if (!projectOnPlane(sbR, up, sbF)) setVec(sbF, 0, 1, 0);
    }
  }
  cross(sbF, up, sbR);
  return quatFromBasis(sbR, sbF, up, out);
}

// ---------------------------------------------------------------------------
// Ellipsoid
// ---------------------------------------------------------------------------

/** Point on the ellipsoid whose geodetic normal is the unit vector n. */
export function surfaceFromNormal(n: Vec3, e: Ellipsoid, out: Vec3): Vec3 {
  const a2 = e.a * e.a;
  const b2 = e.b * e.b;
  const k = Math.sqrt(a2 * n.x * n.x + a2 * n.y * n.y + b2 * n.z * n.z);
  out.x = (a2 * n.x) / k;
  out.y = (a2 * n.y) / k;
  out.z = (b2 * n.z) / k;
  return out;
}

/** Unit geodetic normal from geodetic latitude / longitude (radians). */
export function normalFromLatLon(lat: number, lon: number, out: Vec3): Vec3 {
  const cl = Math.cos(lat);
  out.x = cl * Math.cos(lon);
  out.y = cl * Math.sin(lon);
  out.z = Math.sin(lat);
  return out;
}

/** Geodetic coordinates of an ECEF point. */
export interface Geodetic {
  lat: number;
  lon: number;
  height: number;
}

export function geodetic(): Geodetic {
  return { lat: 0, lon: 0, height: 0 };
}

/**
 * ECEF -> geodetic (latitude, longitude radians; height metres) with two
 * Bowring iterations: sub-millimetre from the Earth's centre region out to
 * lunar distance, smooth across the poles, no allocation.
 */
export function toGeodetic(p: Vec3, e: Ellipsoid, out: Geodetic): Geodetic {
  const a = e.a;
  const b = e.b;
  const e2 = 1 - (b * b) / (a * a);
  const ep2 = (a * a - b * b) / (b * b);
  const r = Math.sqrt(p.x * p.x + p.y * p.y);
  out.lon = Math.atan2(p.y, p.x);
  let beta = Math.atan2(p.z * a, r * b);
  let lat = 0;
  for (let i = 0; i < 2; i++) {
    const sb = Math.sin(beta);
    const cb = Math.cos(beta);
    lat = Math.atan2(p.z + ep2 * b * sb * sb * sb, r - e2 * a * cb * cb * cb);
    beta = Math.atan2(b * Math.sin(lat), a * Math.cos(lat));
  }
  const sl = Math.sin(lat);
  out.lat = lat;
  out.height = r * Math.cos(lat) + p.z * sl - a * Math.sqrt(1 - e2 * sl * sl);
  return out;
}

const sgGeo = geodetic();

/** Geodetic normal at the foot point of p; returns p's geodetic height. */
export function geodeticNormalOf(p: Vec3, e: Ellipsoid, outNormal: Vec3): number {
  toGeodetic(p, e, sgGeo);
  normalFromLatLon(sgGeo.lat, sgGeo.lon, outNormal);
  return sgGeo.height;
}

/** Height of p above the ellipsoid (geodetic), metres. */
export function geodeticHeight(p: Vec3, e: Ellipsoid): number {
  return toGeodetic(p, e, sgGeo).height;
}

/**
 * Nearest positive intersection distance of the ray (o + t·d, d unit) with the
 * ellipsoid (a, a, b); NaN if missed. If the origin is inside, the exit point.
 */
export function rayEllipsoid(o: Vec3, d: Vec3, a: number, b: number): number {
  const ox = o.x / a;
  const oy = o.y / a;
  const oz = o.z / b;
  const dx = d.x / a;
  const dy = d.y / a;
  const dz = d.z / b;
  const A = dx * dx + dy * dy + dz * dz;
  const B = 2 * (ox * dx + oy * dy + oz * dz);
  const C = ox * ox + oy * oy + oz * oz - 1;
  const disc = B * B - 4 * A * C;
  if (!(disc >= 0)) return Number.NaN;
  const sq = Math.sqrt(disc);
  const qq = B > 0 ? -0.5 * (B + sq) : -0.5 * (B - sq);
  let t0 = qq / A;
  let t1 = qq !== 0 ? C / qq : t0;
  if (t0 > t1) {
    const tmp = t0;
    t0 = t1;
    t1 = tmp;
  }
  if (t0 > 0) return t0;
  if (t1 > 0) return t1;
  return Number.NaN;
}

/**
 * Point where the ray meets the sphere of radius rho (nearest positive hit);
 * if it misses, the sphere point closest to the ray (a virtual trackball that
 * is continuous across the silhouette). Returns true on a real hit.
 */
export function raySphereOrClosest(o: Vec3, d: Vec3, rho: number, out: Vec3): boolean {
  const t = rayEllipsoid(o, d, rho, rho);
  if (t > 0) {
    addScaled(o, d, t, out);
    return true;
  }
  const tc = Math.max(0, -dot(o, d));
  addScaled(o, d, tc, out);
  const l = len(out);
  if (l > 0) scale(out, rho / l, out);
  else normalize(o, out);
  return false;
}

// ---------------------------------------------------------------------------
// Pose
// ---------------------------------------------------------------------------

const spF = vec3();
const spU = vec3();
const spR = vec3();

/** Camera pose from a frame, look point, range and pitch. */
export function poseFromLook(q: Quat, look: Vec3, range: number, pitch: number, out: Pose): Pose {
  frameAxes(q, spR, spF, spU);
  const cp = Math.cos(pitch);
  const sp = Math.sin(pitch);
  const d = out.dir;
  d.x = cp * spF.x + sp * spU.x;
  d.y = cp * spF.y + sp * spU.y;
  d.z = cp * spF.z + sp * spU.z;
  const u = out.up;
  u.x = -sp * spF.x + cp * spU.x;
  u.y = -sp * spF.y + cp * spU.y;
  u.z = -sp * spF.z + cp * spU.z;
  copyVec(spR, out.right);
  out.pos.x = look.x - range * d.x;
  out.pos.y = look.y - range * d.y;
  out.pos.z = look.z - range * d.z;
  return out;
}

export function poseFromView(v: RigView, out: Pose): Pose {
  return poseFromLook(v.q, v.look, v.range, v.pitch, out);
}

const slUp = vec3();

/** Globe-mode look point: ellipsoid point under the frame's up axis, lifted by h. */
export function globeLookPoint(s: RigState, e: Ellipsoid, out: Vec3): Vec3 {
  frameAxes(s.q, null, null, slUp);
  surfaceFromNormal(slUp, e, out);
  return addScaled(out, slUp, s.h, out);
}

const sgLook = vec3();

export function globePose(s: RigState, e: Ellipsoid, out: Pose): Pose {
  globeLookPoint(s, e, sgLook);
  return poseFromLook(s.q, sgLook, s.range, s.pitch, out);
}

export function globeView(s: RigState, e: Ellipsoid, out: RigView): RigView {
  copyQuat(s.q, out.q);
  globeLookPoint(s, e, out.look);
  out.range = s.range;
  out.pitch = s.pitch;
  return out;
}

/** Unit ray direction through NDC (nx, ny) for the pose. */
export function pixelRay(p: Pose, lens: Lens, nx: number, ny: number, out: Vec3): Vec3 {
  const sx = nx * lens.tanX;
  const sy = ny * lens.tanY;
  out.x = p.dir.x + sx * p.right.x + sy * p.up.x;
  out.y = p.dir.y + sx * p.right.y + sy * p.up.y;
  out.z = p.dir.z + sx * p.right.z + sy * p.up.z;
  return normalize(out, out);
}

/** Globe frame at geodetic lat/lon (radians) with the given heading (radians, 0 = north up). */
export function frameAtLatLon(lat: number, lon: number, heading: number, out: Quat): Quat {
  const up = normalFromLatLon(lat, lon, spU);
  const sl = Math.sin(lat);
  const cl = Math.cos(lat);
  const so = Math.sin(lon);
  const co = Math.cos(lon);
  // north and east unit tangents
  const nx = -sl * co;
  const ny = -sl * so;
  const nz = cl;
  const ex = -so;
  const ey = co;
  const ch = Math.cos(heading);
  const sh = Math.sin(heading);
  setVec(spF, ch * nx + sh * ex, ch * ny + sh * ey, ch * nz);
  cross(spF, up, spR);
  return quatFromBasis(spR, spF, up, out);
}

/**
 * Heading of the frame relative to true north (radians, clockwise from north),
 * NaN when the up axis is (nearly) a pole.
 */
export function headingFromNorth(q: Quat): number {
  frameAxes(q, null, spF, spU);
  // north = ẑ projected on the tangent plane
  const d = spU.z;
  const nx = -spU.x * d;
  const ny = -spU.y * d;
  const nz = 1 - spU.z * d;
  const nl = Math.sqrt(nx * nx + ny * ny + nz * nz);
  if (nl < 1e-9) return Number.NaN;
  // east = north × up
  const ex = (ny * spU.z - nz * spU.y) / nl;
  const ey = (nz * spU.x - nx * spU.z) / nl;
  const ez = (nx * spU.y - ny * spU.x) / nl;
  const fn = (spF.x * nx + spF.y * ny + spF.z * nz) / nl;
  const fe = spF.x * ex + spF.y * ey + spF.z * ez;
  return Math.atan2(fe, fn);
}

// ---------------------------------------------------------------------------
// Anchoring (camera pose -> globe state)
// ---------------------------------------------------------------------------

const saHit = vec3();
const saN = vec3();
const saF = vec3();

/**
 * Re-expresses a camera pose as a globe state that reproduces it: the look
 * target is where the view ray meets the ellipsoid (or, if it misses, the
 * point `fallbackRange` along the ray). Returns false if the pose is
 * degenerate. The resulting camera equals the input pose up to rounding.
 */
export function anchorFromPose(p: Pose, e: Ellipsoid, fallbackRange: number, out: RigState): boolean {
  let t = rayEllipsoid(p.pos, p.dir, e.a, e.b);
  let h = 0;
  if (t > 0) {
    addScaled(p.pos, p.dir, t, saHit);
    geodeticNormalOf(saHit, e, saN);
  } else {
    t = fallbackRange > 0 ? fallbackRange : 1;
    addScaled(p.pos, p.dir, t, saHit);
    h = geodeticNormalOf(saHit, e, saN);
  }
  const sp = Math.max(-1, Math.min(1, dot(p.dir, saN)));
  const cp = Math.sqrt(1 - sp * sp);
  // forward = cos(p)·dir - sin(p)·camUp (exact when the pose has no roll w.r.t. the normal)
  setVec(saF, cp * p.dir.x - sp * p.up.x, cp * p.dir.y - sp * p.up.y, cp * p.dir.z - sp * p.up.z);
  if (!Number.isFinite(saF.x)) return false;
  frameFromUpForward(saN, saF, p.up, out.q);
  out.range = t;
  out.pitch = Math.asin(sp);
  out.h = h;
  return true;
}

// ---------------------------------------------------------------------------
// Grab and zoom-to-cursor (globe mode, operate on a target state)
// ---------------------------------------------------------------------------

const sgPose = pose();
const sgRay = vec3();
const sgA = vec3();
const sgB = vec3();
const sgUA = vec3();
const sgUB = vec3();
const sgQ = quat();
const sgTotal = quat();

/**
 * "Grab the globe": rotates the state about the Earth's centre so the surface
 * point under NDC (ax, ay) ends up under (bx, by). `rho` is the radius of the
 * grab sphere (|grabbed point|). The rotation is exact (correction passes
 * absorb the ellipsoid's non-sphericity) but limited to `maxFactor` x
 * the flat-pan angle, so grabbing near the horizon cannot fling the globe.
 * Writes the applied world rotation (rotation vector) to outRotVec.
 */
export function applyGrab(
  s: RigState,
  e: Ellipsoid,
  lens: Lens,
  ax: number,
  ay: number,
  bx: number,
  by: number,
  rho: number,
  outRotVec: Vec3,
  maxFactor = 4,
): void {
  globePose(s, e, sgPose);
  raySphereOrClosest(sgPose.pos, pixelRay(sgPose, lens, ax, ay, sgRay), rho, sgA);
  normalize(sgA, sgUA);
  sgTotal.x = 0;
  sgTotal.y = 0;
  sgTotal.z = 0;
  sgTotal.w = 1;
  // Angle of a flat pan of the same screen distance at the look point.
  const flat = (Math.hypot((bx - ax) * lens.tanX, (by - ay) * lens.tanY) * s.range) / rho;
  const maxAngle = Math.min(Math.PI / 3, Math.max(flat * maxFactor, 1e-12));
  // First step (possibly limited), then exact refinement.
  raySphereOrClosest(sgPose.pos, pixelRay(sgPose, lens, bx, by, sgRay), rho, sgB);
  normalize(sgB, sgUB);
  quatFromUnitVectors(sgUB, sgUA, sgQ);
  const angle = angleBetween(sgUB, sgUA);
  let clamped = false;
  if (angle > maxAngle) {
    quatToRotationVector(sgQ, sgRay);
    scale(sgRay, maxAngle / angle, sgRay);
    quatFromRotationVector(sgRay, sgQ);
    clamped = true;
  }
  rotateFrameWorld(s.q, sgQ);
  quatMul(sgQ, sgTotal, sgTotal);
  if (!clamped) {
    refineAlign(s, e, lens, bx, by, rho, ALIGN_MAX_ITER, sgQ);
    quatMul(sgQ, sgTotal, sgTotal);
  }
  quatToRotationVector(sgTotal, outRotVec);
}

const ALIGN_MAX_ITER = 12;
/** Stop refining once the correction is below this angle (≈ 0.06 mm on the Earth). */
const ALIGN_EPS_RAD = 1e-11;
const saTotal = quat();

/**
 * Rotates the state about the Earth's centre until the ray through NDC
 * (nx, ny) meets the sphere of radius `rho` at the direction sgUA (set by the
 * caller). On a sphere one pass is exact; on an ellipsoid the look point
 * shifts slightly with each rotation and a few passes converge. Writes the
 * total rotation applied to `out`. Returns false if the ray misses.
 */
function refineAlign(s: RigState, e: Ellipsoid, lens: Lens, nx: number, ny: number, rho: number, maxIter: number, out: Quat): boolean {
  saTotal.x = 0;
  saTotal.y = 0;
  saTotal.z = 0;
  saTotal.w = 1;
  let ok = true;
  for (let iter = 0; iter < maxIter; iter++) {
    globePose(s, e, sgPose);
    if (!raySphereOrClosest(sgPose.pos, pixelRay(sgPose, lens, nx, ny, sgRay), rho, sgB)) {
      ok = false;
      break;
    }
    normalize(sgB, sgUB);
    const angle = angleBetween(sgUB, sgUA);
    if (angle < ALIGN_EPS_RAD) break;
    quatFromUnitVectors(sgUB, sgUA, sgQ);
    rotateFrameWorld(s.q, sgQ);
    quatMul(sgQ, saTotal, saTotal);
  }
  copyQuat(saTotal, out);
  return ok;
}

/**
 * Keeps a world point under the cursor: rotates the state about the Earth's
 * centre so the ray through NDC (nx, ny) passes through `point`. Used every
 * frame while an eased zoom runs, so the point under the cursor stays locked
 * during the whole animation, not just at its end. Writes the applied world
 * rotation to `out` (apply the same rotation to other states that should
 * move along). Returns false if the cursor ray misses the point's sphere.
 */
export function alignPointToCursor(s: RigState, e: Ellipsoid, lens: Lens, nx: number, ny: number, point: Vec3, out: Quat): boolean {
  normalize(point, sgUA);
  return refineAlign(s, e, lens, nx, ny, len(point), ALIGN_MAX_ITER, out);
}

/**
 * Point of the globe (ellipsoid lifted by the state's look height) under NDC
 * (nx, ny) for the state's camera; false if the ray misses.
 */
export function globePointUnderCursor(s: RigState, e: Ellipsoid, lens: Lens, nx: number, ny: number, out: Vec3): boolean {
  globePose(s, e, sgPose);
  pixelRay(sgPose, lens, nx, ny, sgRay);
  const t = rayEllipsoid(sgPose.pos, sgRay, e.a + s.h, e.b + s.h);
  if (!(t > 0)) return false;
  addScaled(sgPose.pos, sgRay, t, out);
  return true;
}

/**
 * Zoom towards NDC (nx, ny) by `factor` (< 1 = in): range is scaled and
 * clamped to [rMin, rMax], then the frame is rotated so the surface point that
 * was under the cursor stays there. Returns the applied range factor.
 */
export function applyZoomAt(
  s: RigState,
  e: Ellipsoid,
  lens: Lens,
  nx: number,
  ny: number,
  factor: number,
  rMin: number,
  rMax: number,
): number {
  const r0 = s.range;
  const r1 = Math.min(rMax, Math.max(rMin, r0 * factor));
  if (r1 === r0) return 1;
  globePose(s, e, sgPose);
  pixelRay(sgPose, lens, nx, ny, sgRay);
  const t = rayEllipsoid(sgPose.pos, sgRay, e.a + s.h, e.b + s.h);
  s.range = r1;
  if (!(t > 0)) return r1 / r0;
  addScaled(sgPose.pos, sgRay, t, sgA);
  const rho = len(sgA);
  normalize(sgA, sgUA);
  refineAlign(s, e, lens, nx, ny, rho, ALIGN_MAX_ITER, sgQ);
  return r1 / r0;
}

/** World rotation that pans the look target by (forward, right) metres on a sphere of radius rho. */
export function panRotation(q: Quat, forwardM: number, rightM: number, rho: number, out: Quat): Quat {
  frameAxes(q, spR, spF, null);
  // forward: axis = up × forward = -right; rightward: axis = forward
  const ax = (-spR.x * forwardM + spF.x * rightM) / rho;
  const ay = (-spR.y * forwardM + spF.y * rightM) / rho;
  const az = (-spR.z * forwardM + spF.z * rightM) / rho;
  setVec(sgRay, ax, ay, az);
  return quatFromRotationVector(sgRay, out);
}

// ---------------------------------------------------------------------------
// Limits
// ---------------------------------------------------------------------------

function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}

export function smoothstep(t: number): number {
  const x = clamp(t, 0, 1);
  return x * x * (3 - 2 * x);
}

export function smootherstep(t: number): number {
  const x = clamp(t, 0, 1);
  return x * x * x * (x * (x * 6 - 15) + 10);
}

const PITCH_NEAR_R = 20_000;
const PITCH_MID_R = 2_000_000;
const PITCH_FAR_R = 12_000_000;
const PITCH_NEAR_DEG = -1.5;
const PITCH_MID_DEG = -20;
const PITCH_FAR_DEG = -80;

/**
 * Highest allowed pitch (closest to the horizon) for a globe view at `range`:
 * -80° (nearly straight down) from ~12,000 km out, opening up to -20° at
 * 2,000 km and to -1.5° (just above the horizon) at 20 km and below.
 */
export function pitchLimitForRange(range: number): number {
  if (!(range > PITCH_MID_R)) {
    const t = clamp(Math.log(Math.max(range, 1) / PITCH_NEAR_R) / Math.log(PITCH_MID_R / PITCH_NEAR_R), 0, 1);
    return (PITCH_NEAR_DEG + (PITCH_MID_DEG - PITCH_NEAR_DEG) * t) * DEG;
  }
  const t = smoothstep(Math.log(range / PITCH_MID_R) / Math.log(PITCH_FAR_R / PITCH_MID_R));
  return (PITCH_MID_DEG + (PITCH_FAR_DEG - PITCH_MID_DEG) * t) * DEG;
}

/**
 * Minimum range for a globe view so the camera stays `need` metres above the
 * look point (need = minAltitude + terrain under camera - look height).
 */
export function minRangeForNeed(need: number): number {
  return Math.max(1, need);
}

/**
 * Applies globe-mode limits in place: range within [need, maxRange], pitch in
 * [-90°, min(pitchLimitForRange, pitch that keeps the camera `need` above the
 * look point)].
 */
export function constrainGlobe(s: RigState, need: number, maxRange: number): RigState {
  s.range = clamp(s.range, minRangeForNeed(need), maxRange);
  let pmax = pitchLimitForRange(s.range);
  if (need > 0) pmax = Math.min(pmax, -Math.asin(Math.min(1, need / s.range)));
  s.pitch = clamp(s.pitch, -HALF_PI, pmax);
  return s;
}

/** Highest pitch allowed in track mode. */
export const TRACK_PITCH_MAX = 80 * DEG;

/**
 * Track-mode limits: range in [minR, maxR]; pitch in [-90°, 80°] and low
 * enough that the camera (orbiting a target at distance `targetR` from the
 * Earth's centre) stays at least `groundR` from the centre.
 */
export function constrainTrack(s: RigState, targetR: number, groundR: number, minR: number, maxR: number): RigState {
  s.range = clamp(s.range, minR, maxR);
  const r = s.range;
  const rhs = (targetR * targetR + r * r - groundR * groundR) / (2 * r * targetR);
  const pmax = Math.min(TRACK_PITCH_MAX, Math.asin(clamp(rhs, -1, 1)));
  s.pitch = clamp(s.pitch, -HALF_PI, pmax);
  return s;
}

// ---------------------------------------------------------------------------
// Smoothing (frame-rate independent)
// ---------------------------------------------------------------------------

/** Fraction of the remaining distance covered in dt by an exponential follower with time constant tau. */
export function smoothAlpha(dt: number, tau: number): number {
  if (!(tau > 0)) return 1;
  if (!(dt > 0)) return 0;
  return 1 - Math.exp(-dt / tau);
}

export function smoothScalar(cur: number, target: number, dt: number, tau: number): number {
  return cur + (target - cur) * smoothAlpha(dt, tau);
}

/** Exponential follow in log space (constant feel at every scale). */
export function smoothRange(cur: number, target: number, dt: number, tau: number): number {
  if (!(cur > 0) || !(target > 0)) return target;
  return cur * Math.pow(target / cur, smoothAlpha(dt, tau));
}

/** Geodesic exponential follow for orientations. Safe when out aliases cur. */
export function smoothQuat(cur: Quat, target: Quat, dt: number, tau: number, out: Quat): Quat {
  return quatSlerp(cur, target, smoothAlpha(dt, tau), out);
}

/** Exponential decay factor for an inertial velocity over dt. */
export function decay(dt: number, tau: number): number {
  return Math.exp(-dt / tau);
}

// ---------------------------------------------------------------------------
// Flights
// ---------------------------------------------------------------------------

/** Range ratios (|ln rb/ra|) up to here move the look point with e itself... */
const LOOK_COUPLE_LN0 = 1.5;
/** ...from here on fully coupled to the range (blended in between). */
const LOOK_COUPLE_LN1 = 4.5;

/**
 * Look-point progress (0 = a.look, 1 = b.look) of a flight at eased parameter
 * e whose range goes log-linearly from ra to rb. For big zooms (e.g. the
 * 16,000 km overview -> a 120 m close-up) moving the look point with e would
 * leave the target kilometres off-axis while the camera is already metres
 * from it (and pop it into view at the very end). Instead the look error
 * (distance of the look point from the near-end look) is tied to the range:
 * err = D · g², g = (r - rNear) / (rFar - rNear), so err / r shrinks with the
 * range and the target slides smoothly to the screen centre. Small range
 * changes keep plain e. Endpoints are exact (0 -> 0, 1 -> 1), monotonic in e.
 */
export function flightLookParam(ra: number, rb: number, e: number): number {
  const la = Math.log(ra);
  const lb = Math.log(rb);
  const d = Math.abs(lb - la);
  if (!(d > LOOK_COUPLE_LN0) || e <= 0 || e >= 1) return e;
  const r = Math.exp(la + (lb - la) * e);
  const lo = Math.min(ra, rb);
  const g = Math.min(1, Math.max(0, (r - lo) / (Math.max(ra, rb) - lo)));
  const coupled = ra > rb ? 1 - g * g : g * g;
  const x = Math.min(1, (d - LOOK_COUPLE_LN0) / (LOOK_COUPLE_LN1 - LOOK_COUPLE_LN0));
  const w = x * x * (3 - 2 * x);
  return e + (coupled - e) * w;
}

/**
 * Interpolates two views at eased parameter e: frame by slerp, look point by
 * spherical interpolation around the Earth's centre (progress coupled to the
 * range for big zooms, see flightLookParam), range in log space with an
 * optional "hop" (extra range at mid-flight, for long arcs), pitch linearly.
 */
export function interpolateView(a: RigView, b: RigView, e: number, hop: number, out: RigView): RigView {
  const ra = Math.max(a.range, 1e-3);
  const rb = Math.max(b.range, 1e-3);
  quatSlerp(a.q, b.q, e, out.q);
  slerpPoint(a.look, b.look, flightLookParam(ra, rb, e), out.look);
  out.range = Math.exp(Math.log(ra) + (Math.log(rb) - Math.log(ra)) * e) + hop * Math.sin(Math.PI * e);
  out.pitch = a.pitch + (b.pitch - a.pitch) * e;
  return out;
}

/** Extra mid-flight range for a flight between two look points (0 when not needed). */
export function flightHop(a: RigView, b: RigView, earthRadius: number): number {
  const arc = angleBetween(a.look, b.look) * earthRadius;
  return Math.max(0, 0.6 * arc - Math.max(a.range, b.range));
}
