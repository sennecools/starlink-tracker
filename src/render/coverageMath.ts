/**
 * Pure math behind the coverage hologram (CoverageFootprint.ts): the
 * spherical cap on the Earth from which a satellite is seen above the minimum
 * elevation angle, and the local frame the footprint mesh is laid out in.
 * No Cesium import, so it runs in Node tests.
 *
 * Geometry (sphere of local radius R around the Earth's centre, satellite at
 * distance D = R + h from the centre, minimum elevation ε):
 *
 *     λ = acos(R · cos ε / D) − ε          Earth central angle of the cap edge
 *     ground radius = R · λ                (arc length along the surface)
 *     slant range at the edge = sqrt(D² + R² − 2 D R cos λ)
 *
 * 550 km / 25°: λ ≈ 8.46°, ground radius ≈ 940 km.
 * 275 km / 25° (V3 insertion orbit): λ ≈ 4.68°, ground radius ≈ 520 km.
 *
 * The renderer uses R = the WGS-84 radius along the satellite's geocentric
 * direction, which is within 0.4 % of the geodetic picture and keeps the GPU
 * math exact relative to the cap centre.
 */

export const DEG2RAD = Math.PI / 180;
/** Mean Earth radius (km), as used by satDetails.footprintRadiusKm. */
export const MEAN_EARTH_RADIUS_KM = 6371.0088;
/** WGS-84 semi-axes (m). */
export const WGS84_A_M = 6_378_137;
export const WGS84_B_M = 6_356_752.314245179;

/** Footprint height above the ellipsoid from 10 km camera height up (clears the depth plane). */
export const COVERAGE_LIFT_MAX_M = 2_500;
/** Lowest footprint height (camera at its 250 m minimum altitude and below). */
export const COVERAGE_LIFT_MIN_M = 30;

/**
 * Height of the coverage footprint above the ellipsoid for a camera at
 * `cameraHeightM` (ellipsoid height): a quarter of it, clamped to
 * [COVERAGE_LIFT_MIN_M, COVERAGE_LIFT_MAX_M]. A fixed 2.5 km lift would put
 * the layer ABOVE a low camera (a ceiling outside the view, and limb-discarded
 * nearby); the lift only has to clear the depth plane, which lies below the
 * surface everywhere except at the limb, where the footprint fades anyway.
 */
export function coverageLiftM(cameraHeightM: number): number {
  if (!Number.isFinite(cameraHeightM)) return COVERAGE_LIFT_MAX_M;
  return Math.min(COVERAGE_LIFT_MAX_M, Math.max(COVERAGE_LIFT_MIN_M, 0.25 * cameraHeightM));
}

/**
 * Earth central angle (radians) between the sub-satellite point and the edge
 * of the coverage cap. `earthRadius` and `satDistance` (from the Earth's
 * centre) in the same unit. 0 when the satellite is below/at the surface or
 * cannot be seen above `minElevationDeg` from anywhere.
 */
export function coverageCapAngle(earthRadius: number, satDistance: number, minElevationDeg: number): number {
  if (!(earthRadius > 0) || !(satDistance > earthRadius)) return 0;
  const e = minElevationDeg * DEG2RAD;
  if (!(e >= 0 && e < Math.PI / 2)) return 0;
  const c = (earthRadius * Math.cos(e)) / satDistance;
  const lambda = Math.acos(Math.min(1, Math.max(-1, c))) - e;
  return lambda > 0 ? lambda : 0;
}

/** Ground radius (km, arc length) of the coverage cap of a satellite at `altitudeKm`. */
export function coverageGroundRadiusKm(altitudeKm: number, minElevationDeg: number, earthRadiusKm = MEAN_EARTH_RADIUS_KM): number {
  if (!(altitudeKm > 0)) return 0;
  return earthRadiusKm * coverageCapAngle(earthRadiusKm, earthRadiusKm + altitudeKm, minElevationDeg);
}

/** Slant range (same unit as the inputs) from the satellite to the cap edge. */
export function coverageSlantRange(earthRadius: number, satDistance: number, capAngle: number): number {
  const d2 = satDistance * satDistance + earthRadius * earthRadius - 2 * satDistance * earthRadius * Math.cos(capAngle);
  return Math.sqrt(Math.max(0, d2));
}

/** Half-angle (radians) of the beam cone at the satellite (nadir to cap edge): 90° − ε − λ. */
export function coverageConeHalfAngle(capAngle: number, minElevationDeg: number): number {
  return Math.max(0, Math.PI / 2 - minElevationDeg * DEG2RAD - capAngle);
}

/**
 * Orthonormal footprint frame at the satellite's geocentric direction:
 * out[0..2] = up (unit, from the Earth's centre through the satellite),
 * out[3..5] = east, out[6..8] = north (right-handed: east × north = up).
 * Near the poles (|up·ẑ| > 0.9999) east falls back to a stable +Y-derived
 * axis. Returns false (out untouched) for a zero / non-finite position.
 * Allocation-free.
 */
export function footprintBasis(x: number, y: number, z: number, out: Float64Array | number[]): boolean {
  const r = Math.sqrt(x * x + y * y + z * z);
  if (!(r > 0) || !Number.isFinite(r)) return false;
  const ux = x / r;
  const uy = y / r;
  const uz = z / r;
  // east = ẑ × up (normalised); at the poles use ŷ × up instead.
  let ex: number, ey: number, ez: number;
  if (Math.abs(uz) < 0.9999) {
    ex = -uy;
    ey = ux;
    ez = 0;
  } else {
    // ŷ × up = (uz, 0, -ux)
    ex = uz;
    ey = 0;
    ez = -ux;
  }
  const el = Math.sqrt(ex * ex + ey * ey + ez * ez);
  ex /= el;
  ey /= el;
  ez /= el;
  // north = up × east
  const nx = uy * ez - uz * ey;
  const ny = uz * ex - ux * ez;
  const nz = ux * ey - uy * ex;
  out[0] = ux;
  out[1] = uy;
  out[2] = uz;
  out[3] = ex;
  out[4] = ey;
  out[5] = ez;
  out[6] = nx;
  out[7] = ny;
  out[8] = nz;
  return true;
}

/**
 * Radius of the WGS-84 ellipsoid along the unit direction (ux, uy, uz) from
 * the centre (geocentric radius), metres.
 */
export function ellipsoidRadiusAlong(ux: number, uy: number, uz: number, a = WGS84_A_M, b = WGS84_B_M): number {
  const q = (ux * ux + uy * uy) / (a * a) + (uz * uz) / (b * b);
  return 1 / Math.sqrt(q);
}

/**
 * Offset (metres) from the cap centre C·(rC + lift) to the cap point at
 * fraction `u` of the cap angle and azimuth `phi` (0 = east, π/2 = north),
 * both lifted `lift` metres along their geocentric direction above the
 * ellipsoid. Exactly what the footprint vertex shader computes (CPU
 * reference; written to be precise for small offsets: no large-number
 * cancellation). `basis` from footprintBasis.
 */
export function capPointOffset(
  basis: ArrayLike<number>,
  capAngle: number,
  u: number,
  phi: number,
  lift: number,
  out: Float64Array | number[],
  a = WGS84_A_M,
  b = WGS84_B_M,
): void {
  const ang = u * capAngle;
  const s = Math.sin(ang);
  const h = Math.sin(ang * 0.5);
  const omc = -2 * h * h; // cos(ang) - 1
  const cp = Math.cos(phi);
  const sp = Math.sin(phi);
  const cx = basis[0], cy = basis[1], cz = basis[2];
  // d - C
  const mx = omc * cx + s * (cp * basis[3] + sp * basis[6]);
  const my = omc * cy + s * (cp * basis[4] + sp * basis[7]);
  const mz = omc * cz + s * (cp * basis[5] + sp * basis[8]);
  const ia2 = 1 / (a * a);
  const ib2 = 1 / (b * b);
  const qC = (cx * cx + cy * cy) * ia2 + cz * cz * ib2;
  const rC = 1 / Math.sqrt(qC);
  // q(d) - q(C) = (d - C) · M · (d + C)
  const dq = (mx * (2 * cx + mx) + my * (2 * cy + my)) * ia2 + mz * (2 * cz + mz) * ib2;
  const x = dq / qC;
  const sq = Math.sqrt(1 + x);
  const dr = (-rC * x) / (sq * (1 + sq)); // r(d) - r(C)
  const k = rC + lift;
  out[0] = k * mx + dr * (cx + mx);
  out[1] = k * my + dr * (cy + my);
  out[2] = k * mz + dr * (cz + mz);
}

// ---------------------------------------------------------------------------
// Static meshes of the hologram (attribute = (u or s, phi))
// ---------------------------------------------------------------------------

/** Disc vertices (u, phi) ring-major and TRIANGLES indices. */
export function buildDiscMesh(
  segments = 128,
  ringStep = 0.02,
  uMax = 1.08,
): { vertices: Float32Array; indices: Uint16Array } {
  const rings = Math.round(uMax / ringStep) + 1;
  const cols = segments + 1;
  const vertices = new Float32Array(rings * cols * 2);
  let v = 0;
  for (let r = 0; r < rings; r++) {
    const u = r * ringStep;
    for (let c = 0; c < cols; c++) {
      vertices[v++] = u;
      vertices[v++] = (c / segments) * Math.PI * 2;
    }
  }
  const indices = new Uint16Array((rings - 1) * segments * 6);
  let k = 0;
  for (let r = 0; r < rings - 1; r++) {
    for (let c = 0; c < segments; c++) {
      const a = r * cols + c;
      const b = a + cols;
      indices[k++] = a;
      indices[k++] = b;
      indices[k++] = b + 1;
      indices[k++] = a;
      indices[k++] = b + 1;
      indices[k++] = a + 1;
    }
  }
  return { vertices, indices };
}

/** Cone triangles (s, phi): apex (s = 0) + two rim vertices (s = 1) per segment. */
export function buildConeMesh(segments = 96): Float32Array {
  const out = new Float32Array(segments * 3 * 2);
  const step = (Math.PI * 2) / segments;
  let v = 0;
  for (let i = 0; i < segments; i++) {
    out[v++] = 0;
    out[v++] = (i + 0.5) * step;
    out[v++] = 1;
    out[v++] = i * step;
    out[v++] = 1;
    out[v++] = (i + 1) * step;
  }
  return out;
}
