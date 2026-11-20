/**
 * Pure sky math shared by the asset build scripts (scripts/build-sky.mjs,
 * scripts/build-stars.mjs — Node imports this .ts file directly via type
 * stripping) and the runtime (StarField / createSky). No Cesium, no DOM.
 *
 * Frames
 * ------
 * - Source data (NASA SVS Deep Star Maps 2020, Hipparcos via d3-celestial) is
 *   in ICRF ≈ J2000 mean equator & equinox.
 * - Cesium samples its sky box with a direction in TEME axes (true equator,
 *   mean equinox of date; SkyBoxVS: czm_temeToPseudoFixed * position). We bake
 *   J2000 -> mean-of-date precession (IAU 1976, Lieske) for a fixed epoch into
 *   both the cube faces and the star vectors. Nutation (<= ~17" in longitude,
 *   ~9" in obliquity) and the drift between the baked epoch and "now"
 *   (~50"/year) are far below a texel (2k cube: ~160"/texel), so they are
 *   ignored.
 * - Cartesian convention: x -> equinox (RA 0h), z -> celestial north pole,
 *   y = z × x (RA 6h). A unit vector is (cosδ cosα, cosδ sinα, sinδ).
 *
 * Cube addressing (verified against Cesium 1.147)
 * -----------------------------------------------
 * WebGL cube-map lookup (GLES 3.0 §3.8.10, table 3.21) of direction r:
 *   +X: sc=-rz tc=-ry   -X: sc=+rz tc=-ry
 *   +Y: sc=+rx tc=+rz   -Y: sc=+rx tc=-rz
 *   +Z: sc=+rx tc=-ry   -Z: sc=-rx tc=-ry
 *   s = (sc/|ma| + 1)/2, t = (tc/|ma| + 1)/2; t = 0 is the FIRST uploaded row.
 * Cesium flips every sky-box face vertically exactly once (loadCubeMap
 * fetches with flipY -> ImageBitmap 'flipY', or UNPACK_FLIP_Y_WEBGL for an
 * HTMLImageElement), so the TOP image row is t = 1. Checked empirically: the
 * bright stars of Cesium's own tycho2t3_80 faces only line up with this
 * convention (score 155 vs ~31-105 for the flipped/mirrored alternatives).
 *
 * Face order (index): 0 px (positiveX), 1 nx, 2 py, 3 ny, 4 pz, 5 nz.
 */

export const DEG = Math.PI / 180;
export const ARCSEC = DEG / 3600;

/** Face file stems in Cesium's positiveX, negativeX, ... order. */
export const CUBE_FACES = ['px', 'nx', 'py', 'ny', 'pz', 'nz'] as const;
export type CubeFace = (typeof CUBE_FACES)[number];

/** Cesium SkyBox `sources` key for each face index. */
export const CUBE_FACE_SOURCE_KEYS = [
  'positiveX',
  'negativeX',
  'positiveY',
  'negativeY',
  'positiveZ',
  'negativeZ',
] as const;

/** Writable 3-vector. */
export interface Vec3Like {
  x: number;
  y: number;
  z: number;
}

// ---------------------------------------------------------------------------
// RA/Dec <-> unit vector
// ---------------------------------------------------------------------------

export function raDecToVec(raDeg: number, decDeg: number, out: Vec3Like = { x: 0, y: 0, z: 0 }): Vec3Like {
  const ra = raDeg * DEG;
  const dec = decDeg * DEG;
  const c = Math.cos(dec);
  out.x = c * Math.cos(ra);
  out.y = c * Math.sin(ra);
  out.z = Math.sin(dec);
  return out;
}

/** RA in [0, 360), Dec in [-90, 90]. The vector need not be normalized. */
export function vecToRaDec(x: number, y: number, z: number): { raDeg: number; decDeg: number } {
  const r = Math.hypot(x, y, z);
  let ra = Math.atan2(y, x) / DEG;
  if (ra < 0) ra += 360;
  if (ra >= 360) ra -= 360;
  const dec = Math.asin(Math.max(-1, Math.min(1, z / r))) / DEG;
  return { raDeg: ra, decDeg: dec };
}

/** Great-circle angle between two vectors, degrees. */
export function angleBetweenDeg(a: Vec3Like, b: Vec3Like): number {
  const cx = a.y * b.z - a.z * b.y;
  const cy = a.z * b.x - a.x * b.z;
  const cz = a.x * b.y - a.y * b.x;
  return Math.atan2(Math.hypot(cx, cy, cz), a.x * b.x + a.y * b.y + a.z * b.z) / DEG;
}

// ---------------------------------------------------------------------------
// Precession (IAU 1976 / Lieske 1977), J2000 mean -> mean of date
// ---------------------------------------------------------------------------

export function julianYearToJd(year: number): number {
  return 2451545.0 + (year - 2000) * 365.25;
}

export function dateToJulianYear(date: Date): number {
  return 2000 + (date.getTime() / 86400000 + 2440587.5 - 2451545.0) / 365.25;
}

/**
 * Default baking epoch: the current Julian year rounded to the nearest whole
 * year, so re-running the build on another day gives byte-identical output
 * (worst-case error vs. "now" is half a year of precession ≈ 25").
 */
export function defaultEpoch(date: Date = new Date()): number {
  return Math.round(dateToJulianYear(date));
}

/**
 * Row-major 3x3 precession matrix P with r_date = P · r_J2000 for the given
 * Julian epoch (e.g. 2027.0).
 */
export function precessionMatrix(epochYear: number, out: Float64Array = new Float64Array(9)): Float64Array {
  const T = (julianYearToJd(epochYear) - 2451545.0) / 36525;
  const zeta = (2306.2181 * T + 0.30188 * T * T + 0.017998 * T * T * T) * ARCSEC;
  const z = (2306.2181 * T + 1.09468 * T * T + 0.018203 * T * T * T) * ARCSEC;
  const theta = (2004.3109 * T - 0.42665 * T * T - 0.041833 * T * T * T) * ARCSEC;
  const cz = Math.cos(zeta), sz = Math.sin(zeta);
  const cZ = Math.cos(z), sZ = Math.sin(z);
  const ct = Math.cos(theta), st = Math.sin(theta);
  out[0] = cz * ct * cZ - sz * sZ;
  out[1] = -sz * ct * cZ - cz * sZ;
  out[2] = -st * cZ;
  out[3] = cz * ct * sZ + sz * cZ;
  out[4] = -sz * ct * sZ + cz * cZ;
  out[5] = -st * sZ;
  out[6] = cz * st;
  out[7] = -sz * st;
  out[8] = ct;
  return out;
}

/** out = M · v (row-major 3x3). `out` may alias `v`. */
export function mulMat3(m: ArrayLike<number>, v: Vec3Like, out: Vec3Like): Vec3Like {
  const x = v.x, y = v.y, z = v.z;
  out.x = m[0] * x + m[1] * y + m[2] * z;
  out.y = m[3] * x + m[4] * y + m[5] * z;
  out.z = m[6] * x + m[7] * y + m[8] * z;
  return out;
}

/** out = Mᵀ · v (inverse rotation). `out` may alias `v`. */
export function mulMat3T(m: ArrayLike<number>, v: Vec3Like, out: Vec3Like): Vec3Like {
  const x = v.x, y = v.y, z = v.z;
  out.x = m[0] * x + m[3] * y + m[6] * z;
  out.y = m[1] * x + m[4] * y + m[7] * z;
  out.z = m[2] * x + m[5] * y + m[8] * z;
  return out;
}

/** J2000 RA/Dec (deg) -> unit vector in the sky frame of `epochYear`. */
export function j2000ToSkyVec(raDeg: number, decDeg: number, epochYear: number, out: Vec3Like = { x: 0, y: 0, z: 0 }): Vec3Like {
  return mulMat3(precessionMatrix(epochYear), raDecToVec(raDeg, decDeg, out), out);
}

// ---------------------------------------------------------------------------
// Cube faces
// ---------------------------------------------------------------------------

/**
 * Direction (not normalized; the major axis component is ±1) for face `face`
 * at image coordinates s (0 = left edge, 1 = right edge) and v (0 = TOP edge,
 * 1 = bottom edge) of the JPEG as written by build-sky.
 */
export function faceUvToDir(face: number, s: number, v: number, out: Vec3Like): Vec3Like {
  const sc = 2 * s - 1;
  const tc = 1 - 2 * v; // top row is t = 1 in Cesium (see header)
  switch (face) {
    case 0: out.x = 1; out.y = -tc; out.z = -sc; break; // +X
    case 1: out.x = -1; out.y = -tc; out.z = sc; break; // -X
    case 2: out.x = sc; out.y = 1; out.z = tc; break; // +Y
    case 3: out.x = sc; out.y = -1; out.z = -tc; break; // -Y
    case 4: out.x = sc; out.y = -tc; out.z = 1; break; // +Z
    default: out.x = -sc; out.y = -tc; out.z = -1; break; // -Z
  }
  return out;
}

/** Unit direction through the centre of texel (col, row) of a size×size face (row 0 = top). */
export function faceTexelToDir(face: number, col: number, row: number, size: number, out: Vec3Like): Vec3Like {
  faceUvToDir(face, (col + 0.5) / size, (row + 0.5) / size, out);
  const inv = 1 / Math.hypot(out.x, out.y, out.z);
  out.x *= inv;
  out.y *= inv;
  out.z *= inv;
  return out;
}

export interface FaceUv {
  face: number;
  /** 0 = left, 1 = right. */
  s: number;
  /** 0 = top, 1 = bottom (image rows). */
  v: number;
}

/** Inverse of faceUvToDir: which face / image position a direction lands on (GL major-axis rule). */
export function dirToFaceUv(x: number, y: number, z: number, out: FaceUv = { face: 0, s: 0, v: 0 }): FaceUv {
  const ax = Math.abs(x), ay = Math.abs(y), az = Math.abs(z);
  let sc: number, tc: number, ma: number;
  if (ax >= ay && ax >= az) {
    ma = ax;
    if (x > 0) { out.face = 0; sc = -z; tc = -y; } else { out.face = 1; sc = z; tc = -y; }
  } else if (ay >= az) {
    ma = ay;
    if (y > 0) { out.face = 2; sc = x; tc = z; } else { out.face = 3; sc = x; tc = -z; }
  } else {
    ma = az;
    if (z > 0) { out.face = 4; sc = x; tc = -y; } else { out.face = 5; sc = -x; tc = -y; }
  }
  const t = 0.5 * (tc / ma + 1);
  out.s = 0.5 * (sc / ma + 1);
  out.v = 1 - t;
  return out;
}

// ---------------------------------------------------------------------------
// NASA SVS Deep Star Maps equirectangular convention
// ---------------------------------------------------------------------------

/**
 * Equirectangular (plate carrée) position of J2000 RA/Dec in the NASA SVS
 * Deep Star Maps 2020: RA 0h at the horizontal CENTRE, RA increasing to the
 * LEFT (sky seen from inside), north at the top. Verified from the 8k map:
 * the LMC (RA 80.9°) sits at u≈0.275 and the SMC (RA 13.2°) at u≈0.463, the
 * galactic centre (266.4°, -28.9°) at u≈0.76, v≈0.66.
 * Returns u in [0,1) (left→right) and v in [0,1] (top→bottom).
 */
export function raDecToEquirect(raDeg: number, decDeg: number, out: { u: number; v: number } = { u: 0, v: 0 }): { u: number; v: number } {
  let u = 0.5 - raDeg / 360;
  u -= Math.floor(u);
  out.u = u;
  out.v = (90 - decDeg) / 180;
  return out;
}

export function equirectToRaDec(u: number, v: number): { raDeg: number; decDeg: number } {
  let ra = (0.5 - u) * 360;
  ra -= 360 * Math.floor(ra / 360);
  return { raDeg: ra, decDeg: 90 - v * 180 };
}

// ---------------------------------------------------------------------------
// Star colour: B-V -> temperature -> blackbody linear sRGB
// ---------------------------------------------------------------------------

/** Ballesteros (2012) effective temperature from B-V colour index, kelvin. */
export function bvToTemperature(bv: number): number {
  const b = Math.max(-0.4, Math.min(2.0, bv));
  return 4600 * (1 / (0.92 * b + 1.7) + 1 / (0.92 * b + 0.62));
}

// Wyman, Sloan & Shirley (2013) multi-lobe analytic fit of the CIE 1931 2° CMFs.
function g(x: number, mu: number, s1: number, s2: number): number {
  const t = (x - mu) / (x < mu ? s1 : s2);
  return Math.exp(-0.5 * t * t);
}
function cieX(l: number): number {
  return 1.056 * g(l, 599.8, 37.9, 31.0) + 0.362 * g(l, 442.0, 16.0, 26.7) - 0.065 * g(l, 501.1, 20.4, 26.2);
}
function cieY(l: number): number {
  return 0.821 * g(l, 568.8, 46.9, 40.5) + 0.286 * g(l, 530.9, 16.3, 31.1);
}
function cieZ(l: number): number {
  return 1.217 * g(l, 437.0, 11.8, 36.0) + 0.681 * g(l, 459.0, 26.0, 13.8);
}

/**
 * Chromaticity of a blackbody at `kelvin` as LINEAR sRGB (D65), normalized so
 * the largest channel is 1. `out` receives [r, g, b].
 */
export function blackbodyRgb(kelvin: number, out: number[] | Float32Array = [0, 0, 0]): number[] | Float32Array {
  const c2 = 1.4388e-2; // m·K
  let X = 0, Y = 0, Z = 0;
  for (let nm = 380; nm <= 780; nm += 5) {
    const lm = nm * 1e-9;
    const b = 1 / (lm ** 5 * (Math.exp(c2 / (lm * kelvin)) - 1));
    X += b * cieX(nm);
    Y += b * cieY(nm);
    Z += b * cieZ(nm);
  }
  let r = 3.2406 * X - 1.5372 * Y - 0.4986 * Z;
  let gg = -0.9689 * X + 1.8758 * Y + 0.0415 * Z;
  let bb = 0.0557 * X - 0.204 * Y + 1.057 * Z;
  r = Math.max(r, 0);
  gg = Math.max(gg, 0);
  bb = Math.max(bb, 0);
  const m = Math.max(r, gg, bb) || 1;
  out[0] = r / m;
  out[1] = gg / m;
  out[2] = bb / m;
  return out;
}

/** Linear sRGB → sRGB-encoded (0..1). */
export function linearToSrgb(x: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  return x <= 0.0031308 ? 12.92 * x : 1.055 * x ** (1 / 2.4) - 0.055;
}

/**
 * Display colour of a star with colour index `bv`: blackbody chromaticity,
 * desaturated towards white by (1 - saturation), sRGB-encoded, max channel 1.
 */
export function starDisplayColor(bv: number, saturation: number, out: number[] | Float32Array = [0, 0, 0]): number[] | Float32Array {
  blackbodyRgb(bvToTemperature(bv), out);
  let m = 0;
  for (let i = 0; i < 3; i++) {
    const lin = 1 + (out[i] - 1) * saturation;
    out[i] = linearToSrgb(lin);
    if (out[i] > m) m = out[i];
  }
  if (m > 0) for (let i = 0; i < 3; i++) out[i] /= m;
  return out;
}

// ---------------------------------------------------------------------------
// stars.bin layout (written by build-stars, read by StarField)
// ---------------------------------------------------------------------------

/** Float32 values per star: x, y, z (unit, sky frame of the manifest epoch), mag, bv. */
export const STAR_STRIDE = 5;

/** Reference points for orientation checks (J2000). */
export const GALACTIC_CENTER_J2000 = { raDeg: 266.405, decDeg: -28.936 } as const;
export const GALACTIC_NORTH_POLE_J2000 = { raDeg: 192.8595, decDeg: 27.1283 } as const;
export const POLARIS_J2000 = { raDeg: 37.9546, decDeg: 89.2641 } as const;
