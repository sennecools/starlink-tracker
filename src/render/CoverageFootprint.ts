/**
 * CoverageFootprint — the "coverage hologram" of the selected satellite: where
 * on Earth it can provide internet, i.e. the spherical cap from which it is
 * seen at least MIN_ELEVATION_DEG (25°) above the horizon, plus a translucent
 * beam cone from the satellite down to the edge of that cap.
 *
 *   λ = acos(R · cos ε / (R + h)) − ε      (coverageMath.ts)
 *   550 km → ~940 km ground radius, 275 km (V3 insertion orbit) → ~520 km.
 *
 * Two DrawCommands of one custom primitive (same low-level pattern as
 * SatellitePoints / sky/*), both STATIC meshes; everything moving is uniforms:
 *
 *  FOOTPRINT  polar-grid disc (u = fraction of the cap angle 0..1.08, φ =
 *             azimuth), 55 rings × 128 segments, one indexed TRIANGLES draw.
 *             The vertex shader places each vertex on the WGS-84 surface
 *             (+ LIFT_M) from the cap centre C, local east/north and the cap
 *             angle, computing the OFFSET from the cap centre with a
 *             cancellation-free formula (coverageMath.capPointOffset), and adds
 *             the eye-relative centre from a high/low (EncodedCartesian3-style)
 *             pair, so it is jitter-free at any zoom. Fragment: SpaceX-blue
 *             hologram — soft radial fill, crisp bright rim (+ outer glow), a
 *             faint hexagonal cell grid (~45 km cells, anchored to the GROUND
 *             through an atan-warped cube-face projection of the fragment's
 *             ECEF position, so cells do not slide as the satellite moves), a
 *             handful of brighter "active beam" cells that slowly cycle, a
 *             scan ring sweeping outward every 3 s, a nadir marker, and a
 *             screen-space scanline / flicker texture. Fades at the limb.
 *  CONE       96 apex→rim triangles (positions relative to the satellite, which
 *             is high/low encoded). Fresnel-like silhouette brightening, very
 *             low alpha in the middle, fine scanlines drifting down to Earth,
 *             12 rim lines from the satellite fading out halfway. Faded near
 *             the apex (never hides the dot / 3D model), per fragment near
 *             the camera, and strongly when the camera is within ~50 km of the
 *             satellite (close-up / track mode).
 *
 * Both: Pass.OPAQUE + additive blend, no depth writes, depth TEST on — like
 * the points (TRANSLUCENT would get OIT derived commands that replace the
 * additive blend). With the default globe.depthTestAgainstTerrain = false the
 * depth buffer holds Cesium's ellipsoid depth plane at this point, so the
 * footprint (lifted LIFT_M above the ellipsoid) is never cut by terrain, and
 * the far side of the Earth hides it and the cone. The footprint additionally
 * fades analytically at the limb.
 *
 * Per frame (CPU): one SatellitePoints.getPosition at the frame's sim time
 * (same interpolation as the GPU points), a basis, a cap angle and a few
 * in-place uniform writes. Zero allocations. GPU: ~10k triangles + ~200 for
 * the cone, two cheap fragment shaders (no textures).
 *
 * The shader programs are compiled during the first WARMUP_FRAMES frames
 * (drawn degenerate), so the first selection does not hitch.
 */
import {
  BlendEquation,
  BlendFunction,
  BoundingSphere,
  Buffer,
  BufferUsage,
  Cartesian3,
  Cartesian4,
  ComponentDatatype,
  DrawCommand,
  IndexDatatype,
  Pass,
  PrimitiveType,
  RenderState,
  SceneMode,
  ShaderProgram,
  ShaderSource,
  VertexArray,
} from 'cesium';
import type { Context, FrameState, Scene } from 'cesium';
import type { SatCatalog } from '../shared/contracts.ts';
import { MIN_ELEVATION_DEG } from '../shared/phase2.ts';
import type { CoverageApi } from '../shared/phase2.ts';
import { julianToUnixMs } from './SatellitePoints.ts';
import type { SatellitePoints } from './SatellitePoints.ts';
import {
  COVERAGE_LIFT_MAX_M,
  WGS84_A_M,
  WGS84_B_M,
  buildConeMesh,
  buildDiscMesh,
  coverageCapAngle,
  coverageLiftM,
  footprintBasis,
} from './coverageMath.ts';

export {
  buildConeMesh,
  buildDiscMesh,
  capPointOffset,
  coverageCapAngle,
  coverageConeHalfAngle,
  coverageGroundRadiusKm,
  coverageSlantRange,
  footprintBasis,
} from './coverageMath.ts';

/** Show / hide fade, wall ms. */
export const COVERAGE_FADE_MS = 400;
/** Fade-out when switching directly from one satellite to another. */
const SWITCH_FADE_MS = 180;
/** Footprint height above the WGS-84 ellipsoid (clears the depth plane); lower under a low camera (coverageLiftM). */
const LIFT_M = COVERAGE_LIFT_MAX_M;
/** Hexagonal cell width (flat to flat), metres. */
export const HEX_CELL_M = 45_000;
/** Mesh resolution. */
const DISC_SEGMENTS = 128;
const DISC_RING_STEP = 0.02;
const DISC_U_MAX = 1.08;
const CONE_SEGMENTS = 96;
/** Camera within this distance of the satellite: cone strongly faded ... */
const CONE_NEAR_M = 30_000;
/** ... back to full strength from here. */
const CONE_FAR_M = 90_000;
/** Cone strength left when the camera is right at the satellite. */
const CONE_NEAR_STRENGTH = 0.12;
/** Animation clock wraps here (s): all periodic terms are integer cycles of it. */
const TIME_WRAP_S = 600;
/** Frames the (invisible) warm-up draw runs to compile the shaders. */
const WARMUP_FRAMES = 3;
/** Below this many metres above the surface no footprint is drawn. */
const MIN_ALTITUDE_M = 80_000;

const ELLIPSOID_Z_SCALE = (WGS84_A_M * WGS84_A_M) / (WGS84_B_M * WGS84_B_M);

// ---------------------------------------------------------------------------
// GLSL
// ---------------------------------------------------------------------------

const ATTR_GRID = { a_grid: 0 } as const;

/** Shared uniforms + the cap point offset (CPU reference: coverageMath.capPointOffset). */
const COMMON_GLSL = /* glsl */ `
// Cap frame at the sub-satellite point (geocentric up, east, north).
uniform vec3 u_up;
uniform vec3 u_east;
uniform vec3 u_north;
// x: cap angle (rad, incl. reveal scale), y: ellipsoid radius along up (m),
// z: q(up) with the (1, 1, a^2/b^2) metric, w: lift (m)
uniform vec4 u_geo;
// (1, 1, a^2/b^2)
uniform vec3 u_ellip;
// x: time (s, wrapped), y: footprint strength 0..1, z: cone strength 0..1, w: hex cell width (m)
uniform vec4 u_anim;

const vec3 HOLO_BLUE = vec3(0.431, 0.706, 1.0);
const vec3 HOLO_ICE = vec3(0.78, 0.93, 1.0);

// Offset (m) from the cap centre (up * (r + lift)) to the cap point at
// fraction u of the cap angle and azimuth phi; precise for small offsets.
vec3 capOffset(float u, float phi)
{
    float ang = u * u_geo.x;
    float s = sin(ang);
    float h = sin(0.5 * ang);
    vec3 m = (-2.0 * h * h) * u_up + s * (cos(phi) * u_east + sin(phi) * u_north);
    float dq = dot(m * (2.0 * u_up + m), u_ellip);
    float x = dq / u_geo.z;
    float sq = sqrt(max(1.0 + x, 1e-6));
    float dr = -u_geo.y * x / (sq * (1.0 + sq));
    return (u_geo.y + u_geo.w) * m + dr * (u_up + m);
}
`;

/** Fragment-only helpers (gl_FragCoord is not available in vertex shaders). */
const HOLO_FS_GLSL = /* glsl */ `
float holoFlicker(float t)
{
    return 0.95 + 0.05 * sin(t * 31.0) * sin(t * 17.3 + 1.3);
}

// Fine horizontal screen-space scanlines (CSS-pixel based).
float holoScanlines(float t)
{
    return 0.84 + 0.16 * sin(gl_FragCoord.y / czm_pixelRatio * 2.1 - t * 6.0);
}
`;

const FOOTPRINT_VS = /* glsl */ `
in vec2 a_grid; // x: u (fraction of the cap angle), y: phi (rad)

uniform vec3 u_centreHigh;
uniform vec3 u_centreLow;

out vec3 v_rel;
out vec3 v_world;
out float v_u;

void main()
{
    vec3 off = capOffset(a_grid.x, a_grid.y);
    vec3 rel = (u_centreHigh - czm_encodedCameraPositionMCHigh) + (u_centreLow - czm_encodedCameraPositionMCLow) + off;
    v_rel = rel;
    v_world = (u_centreHigh + u_centreLow) + off;
    v_u = a_grid.x;
    gl_Position = czm_modelViewProjectionRelativeToEye * vec4(rel, 1.0);
    czm_vertexLogDepth();
    if (u_anim.y <= 0.0) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    }
}
`;

const FOOTPRINT_FS = /* glsl */ `
in vec3 v_rel;
in vec3 v_world;
in float v_u;

float hash12(vec2 p)
{
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Ground-fixed 2D coordinates (m) on an atan-warped cube face (+ face id in z).
vec3 groundCoords(vec3 p)
{
    vec3 n = normalize(p);
    vec3 a = abs(n);
    vec2 uv;
    float face;
    if (a.x >= a.y && a.x >= a.z) {
        uv = n.yz / a.x;
        face = n.x > 0.0 ? 0.0 : 1.0;
    } else if (a.y >= a.z) {
        uv = n.xz / a.y;
        face = n.y > 0.0 ? 2.0 : 3.0;
    } else {
        uv = n.xy / a.z;
        face = n.z > 0.0 ? 4.0 : 5.0;
    }
    return vec3(atan(uv) * 6371000.0, face);
}

// Pointy-top hex tiling with unit flat-to-flat width: xy = offset from the
// cell centre, zw = cell id.
vec4 hexCell(vec2 p)
{
    const vec2 s = vec2(1.0, 1.7320508);
    vec4 hc = floor(vec4(p, p - vec2(0.5, 0.8660254)) / s.xyxy) + 0.5;
    vec4 h = vec4(p - hc.xy * s, p - (hc.zw + 0.5) * s);
    return dot(h.xy, h.xy) < dot(h.zw, h.zw) ? vec4(h.xy, hc.xy) : vec4(h.zw, hc.zw + 0.5);
}

// 0 at the cell centre, 0.5 on its edge.
float hexRadius(vec2 h)
{
    h = abs(h);
    return max(dot(h, vec2(0.5, 0.8660254)), h.x);
}

void main()
{
    float strength = u_anim.y;
    float t = u_anim.x;
    float u = v_u;
    if (strength <= 0.0 || u > 1.075) {
        discard;
    }

    // Limb: fade where the surface turns away from the viewer.
    vec3 nrm = normalize(v_world * u_ellip);
    float facing = dot(normalize(-v_rel), nrm);
    float horizon = smoothstep(-0.03, 0.07, facing);
    if (horizon <= 0.0) {
        discard;
    }

    float fu = max(fwidth(u), 1e-5);
    float inside = 1.0 - smoothstep(1.0 - fu, 1.0 + fu, u);

    // Soft radial fill, slightly brighter towards the edge.
    float fill = 0.03 + 0.07 * u * u * u;

    // Crisp rim + glow (stronger outside).
    float d = abs(u - 1.0);
    float rimLine = 1.0 - smoothstep(0.7 * fu, 1.9 * fu, d);
    float rimGlow = exp(-d * (u < 1.0 ? 22.0 : 70.0));

    // Ground-anchored hex cells.
    vec3 g = groundCoords(v_world);
    vec2 p = g.xy / u_anim.w;
    vec4 cell = hexCell(p);
    float hr = hexRadius(cell.xy);
    float fp = max(length(fwidth(p)), 1e-5);
    float gridVis = 1.0 - smoothstep(0.12, 0.34, fp);
    float grid = (1.0 - smoothstep(0.0, 1.4 * fp, 0.5 - hr)) * gridVis;

    // A few "active beam" cells, each with its own slow cycle.
    vec2 id = cell.zw + vec2(g.z * 137.0, g.z * 59.0);
    float ph = t / 5.0 + hash12(id) * 7.0;
    float beamOn = step(0.972, hash12(id + floor(ph) * 17.31));
    float env = sin(3.14159265 * fract(ph));
    beamOn *= env * env;
    float activeFill = beamOn * (1.0 - smoothstep(0.08, 0.5, hr));
    float activeEdge = beamOn * (1.0 - smoothstep(0.0, 2.0 * fp, 0.5 - hr));

    // Scan ring sweeping outward every 3 s (+ a short trail).
    float r = fract(t / 3.0);
    float rd = (u - r) * 24.0;
    float ring = exp(-rd * rd);
    float trail = u < r ? exp(-(r - u) * 9.0) * 0.35 : 0.0;
    float scan = (ring + trail) * (1.0 - 0.6 * r) * smoothstep(0.0, 0.06, r);

    // Nadir marker.
    float nadir = exp(-u * u * 2500.0);

    float a = (fill + 0.11 * grid + 0.30 * activeFill + 0.35 * activeEdge + 0.12 * scan) * inside
        + 0.95 * rimLine + 0.28 * rimGlow + 0.55 * nadir;
    a *= holoScanlines(t) * holoFlicker(t) * horizon * strength;
    vec3 col = mix(HOLO_BLUE, HOLO_ICE, clamp(rimLine + 0.6 * activeEdge + nadir + 0.4 * scan * inside, 0.0, 1.0));
    out_FragColor = vec4(col * a, 0.0);
    czm_writeLogDepth();
}
`;

const CONE_VS = /* glsl */ `
in vec2 a_grid; // x: s (0 = satellite, 1 = footprint rim), y: phi (rad)

uniform vec3 u_satHigh;
uniform vec3 u_satLow;
// cap centre - satellite (m)
uniform vec3 u_toCentre;

out vec3 v_rel;
out vec3 v_p;
out float v_s;

void main()
{
    vec3 p = a_grid.x * (u_toCentre + capOffset(1.0, a_grid.y));
    vec3 rel = (u_satHigh - czm_encodedCameraPositionMCHigh) + (u_satLow - czm_encodedCameraPositionMCLow) + p;
    v_rel = rel;
    v_p = p;
    v_s = a_grid.x;
    gl_Position = czm_modelViewProjectionRelativeToEye * vec4(rel, 1.0);
    czm_vertexLogDepth();
    if (u_anim.z <= 0.0) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    }
}
`;

const CONE_FS = /* glsl */ `
in vec3 v_rel;
in vec3 v_p;
in float v_s;

void main()
{
    float s = v_s;
    float strength = u_anim.z;
    if (strength <= 0.0 || s < 0.02) {
        discard;
    }
    float t = u_anim.x;

    // Azimuth around the cone axis (the axis is exactly -u_up).
    vec2 en = vec2(dot(v_p, u_east), dot(v_p, u_north));
    float rad = max(length(en), 1e-2);
    vec2 cs = en / rad;
    float phi = atan(cs.y, cs.x);

    // Fresnel-like silhouette brightening (two-sided).
    vec3 tangent = -cs.y * u_east + cs.x * u_north;
    vec3 nrm = normalize(cross(tangent, v_p));
    float fres = 1.0 - abs(dot(nrm, normalize(v_rel)));
    float edge = fres * fres * fres * fres;

    // Fine scanlines drifting down towards the Earth.
    float axial = dot(v_p, -u_up);
    float x = axial / 9000.0 - t * 0.6;
    float fx = max(fwidth(x), 1e-5);
    float dl = min(fract(x), 1.0 - fract(x));
    float scan = (1.0 - smoothstep(0.0, 1.5 * fx, dl)) * (1.0 - smoothstep(0.18, 0.45, fx));

    // 12 rim lines from the satellite, fading out halfway down.
    const float STEP = 6.2831853 / 12.0;
    float dm = abs(fract(phi / STEP + 0.5) - 0.5) * STEP * rad;
    float fl = max(fwidth(dm), 1e-3);
    float line = (1.0 - smoothstep(0.0, 1.6 * fl, dm)) * (1.0 - smoothstep(0.08, 0.5, s));

    float apex = smoothstep(0.02, 0.16, s);
    float foot = smoothstep(0.8, 1.0, s);
    float camFade = smoothstep(1500.0, 30000.0, length(v_rel));

    float a = (0.016 + 0.2 * edge + 0.05 * scan + 0.45 * line + 0.06 * foot) * apex * camFade * strength;
    a *= holoFlicker(t) * holoScanlines(t);
    vec3 col = mix(HOLO_BLUE, HOLO_ICE, clamp(line + 0.5 * edge, 0.0, 1.0));
    out_FragColor = vec4(col * a, 0.0);
    czm_writeLogDepth();
}
`;

// ---------------------------------------------------------------------------
// Primitive
// ---------------------------------------------------------------------------

/** EncodedCartesian3 semantics: high = multiple of 65536, low = remainder. */
function encodeHigh(value: number): number {
  if (value >= 0) return Math.floor(value / 65536) * 65536;
  return -Math.floor(-value / 65536) * 65536;
}

function encodeInto(x: number, y: number, z: number, high: Cartesian3, low: Cartesian3): void {
  high.x = encodeHigh(x);
  high.y = encodeHigh(y);
  high.z = encodeHigh(z);
  low.x = x - high.x;
  low.y = y - high.y;
  low.z = z - high.z;
}

const RENDER_STATE_DESC = {
  depthTest: { enabled: true },
  depthMask: false,
  blending: {
    enabled: true,
    equationRgb: BlendEquation.ADD,
    equationAlpha: BlendEquation.ADD,
    functionSourceRgb: BlendFunction.ONE,
    functionDestinationRgb: BlendFunction.ONE,
    functionSourceAlpha: BlendFunction.ZERO,
    functionDestinationAlpha: BlendFunction.ONE,
  },
} as const;

/** Private Cesium API (Renderer/Buffer.js), not in the shared typings. */
const BufferWithIndex = Buffer as unknown as {
  createIndexBuffer(options: { context: Context; typedArray: ArrayBufferView; usage: number; indexDatatype: number }): unknown;
};

export interface CoverageFootprintOptions {
  scene: Scene;
  points: () => SatellitePoints | null;
  catalog: () => SatCatalog | null;
}

const scratchSat = new Cartesian3();
const scratchBasis = new Float64Array(9);

export class CoverageFootprint implements CoverageApi {
  private readonly scene: Scene;
  private readonly getPoints: () => SatellitePoints | null;
  private readonly getCatalog: () => SatCatalog | null;
  private readonly primitive: { update(fs: FrameState): void; isDestroyed(): boolean; destroy(): undefined };
  private destroyed = false;
  private gpuReleased = false;

  // --- state ---
  private target = -1;
  private shown = -1;
  private visibleFlag = true;
  private fade = 0;
  private lastWall = -1;
  private boundPoints: SatellitePoints | null = null;
  private warm = WARMUP_FRAMES;

  // --- uniforms (mutated in place) ---
  private readonly uUp = new Cartesian3(0, 0, 1);
  private readonly uEast = new Cartesian3(1, 0, 0);
  private readonly uNorth = new Cartesian3(0, 1, 0);
  private readonly uGeo = new Cartesian4(0, WGS84_A_M, 1, LIFT_M);
  private readonly uEllip = new Cartesian3(1, 1, ELLIPSOID_Z_SCALE);
  private readonly uAnim = new Cartesian4(0, 0, 0, HEX_CELL_M);
  private readonly uCentreHigh = new Cartesian3();
  private readonly uCentreLow = new Cartesian3();
  private readonly uSatHigh = new Cartesian3();
  private readonly uSatLow = new Cartesian3();
  private readonly uToCentre = new Cartesian3();
  private readonly boundingSphere = new BoundingSphere(new Cartesian3(), 7_000_000);

  // --- GPU ---
  private discVA: VertexArray | undefined;
  private coneVA: VertexArray | undefined;
  private discProgram: ShaderProgram | undefined;
  private coneProgram: ShaderProgram | undefined;
  private renderState: RenderState | undefined;
  private discCommand: DrawCommand | undefined;
  private coneCommand: DrawCommand | undefined;
  private discIndexCount = 0;
  private coneVertexCount = 0;

  constructor(opts: CoverageFootprintOptions) {
    this.scene = opts.scene;
    this.getPoints = opts.points;
    this.getCatalog = opts.catalog;
    this.primitive = {
      update: (fs: FrameState) => this.update(fs),
      isDestroyed: () => this.gpuReleased,
      destroy: () => {
        this.releaseGpu();
        return undefined;
      },
    };
    this.scene.primitives.add(this.primitive);
  }

  // -------------------------------------------------------------------------
  // CoverageApi
  // -------------------------------------------------------------------------

  setSatellite(index: number, sameSatellite = false): void {
    if (this.destroyed) return;
    const prevTarget = this.target;
    this.target = Number.isInteger(index) && index >= 0 ? index : -1;
    // The index refers to the CURRENT catalogue: bind to it now (update()
    // clears the target only on a swap that happens after this call).
    const pts = this.livePoints();
    if (pts !== this.boundPoints) {
      this.boundPoints = pts;
      if (sameSatellite && this.shown >= 0 && this.shown === prevTarget && this.target >= 0) {
        // Catalogue hot swap re-binding the same satellite: keep the fade (and
        // so the reveal) and just move to its index in the new catalogue.
        this.shown = this.target;
      } else {
        this.shown = -1;
        this.fade = 0;
      }
    }
  }

  setVisible(visible: boolean): void {
    this.visibleFlag = !!visible;
  }

  get visible(): boolean {
    return this.visibleFlag;
  }

  /** Console diagnostics. */
  debugInfo(): { target: number; shown: number; fade: number; capDeg: number; groundRadiusKm: number } {
    return {
      target: this.target,
      shown: this.shown,
      fade: this.fade,
      capDeg: (this.uGeo.x * 180) / Math.PI,
      groundRadiusKm: (this.uGeo.x * this.uGeo.y) / 1000,
    };
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    const scene = this.scene;
    if (!scene.isDestroyed() && scene.primitives.contains(this.primitive)) {
      scene.primitives.remove(this.primitive); // calls primitive.destroy() -> releaseGpu
    }
    this.releaseGpu();
  }

  // -------------------------------------------------------------------------
  // Per frame
  // -------------------------------------------------------------------------

  private livePoints(): SatellitePoints | null {
    const p = this.getPoints();
    return p && !p.isDestroyed() ? p : null;
  }

  private update(frameState: FrameState): void {
    if (this.destroyed || this.gpuReleased) return;
    if (!frameState.passes.render || frameState.mode !== SceneMode.SCENE3D) return;
    if (!this.discCommand) this.createResources(frameState.context);
    const disc = this.discCommand;
    const cone = this.coneCommand;
    if (!disc || !cone) return;

    const wall = performance.now();
    const dt = this.lastWall < 0 ? 0 : Math.min(wall - this.lastWall, 250);
    this.lastWall = wall;
    this.uAnim.x = (wall / 1000) % TIME_WRAP_S;

    // Shader warm-up: degenerate draws, nothing reaches the screen.
    if (this.warm > 0) {
      this.warm--;
      this.uAnim.y = 0;
      this.uAnim.z = 0;
      disc.cull = false;
      cone.cull = false;
      frameState.commandList.push(disc, cone);
      return;
    }
    disc.cull = true;
    cone.cull = true;

    // Catalogue swapped: indices no longer mean the same satellites.
    const pts = this.livePoints();
    const catalog = this.getCatalog();
    if (pts !== this.boundPoints) {
      this.boundPoints = pts;
      this.target = -1;
      this.shown = -1;
      this.fade = 0;
    }
    const count = pts && catalog && catalog.count === pts.count ? pts.count : 0;
    if (this.target >= count) this.target = -1;
    if (this.shown >= count) {
      this.shown = -1;
      this.fade = 0;
    }

    // Fade state machine: fade out the shown satellite before switching.
    const desired = this.visibleFlag ? this.target : -1;
    if (this.shown !== desired) {
      const ms = desired >= 0 && this.shown >= 0 ? SWITCH_FADE_MS : COVERAGE_FADE_MS;
      this.fade = this.shown >= 0 ? Math.max(0, this.fade - dt / ms) : 0;
      if (this.fade <= 0) this.shown = desired;
    } else if (this.shown >= 0) {
      this.fade = Math.min(1, this.fade + dt / COVERAGE_FADE_MS);
    }
    if (this.shown < 0 || this.fade <= 0 || !pts || !pts.show) return;

    const sat = pts.getPosition(this.shown, scratchSat, julianToUnixMs(frameState.time));
    if (!sat || !this.updateGeometry(sat.x, sat.y, sat.z, frameState)) return;

    frameState.commandList.push(disc);
    if (this.uAnim.z > 0) frameState.commandList.push(cone);
  }

  /** Writes all per-frame uniforms + the bounding sphere. False = nothing to draw. */
  private updateGeometry(x: number, y: number, z: number, frameState: FrameState): boolean {
    const b = scratchBasis;
    if (!footprintBasis(x, y, z, b)) return false;
    const dist = Math.sqrt(x * x + y * y + z * z);
    const ux = b[0], uy = b[1], uz = b[2];
    const q = ux * ux + uy * uy + uz * uz * ELLIPSOID_Z_SCALE; // (1,1,a²/b²) metric
    const rC = WGS84_A_M / Math.sqrt(q);
    if (!(dist - rC > MIN_ALTITUDE_M)) return false;
    const cap = coverageCapAngle(rC, dist, MIN_ELEVATION_DEG);
    if (!(cap > 0)) return false;

    // The layer stays below a low camera (a fixed LIFT_M would make it a ceiling).
    const cam = frameState.camera.positionWC;
    const lift = coverageLiftM(frameState.camera.positionCartographic.height);

    // Ease-out reveal: the cap grows from 82 % to full size while fading in.
    const f = this.fade;
    const ease = 1 - (1 - f) * (1 - f) * (1 - f);
    this.uGeo.x = cap * (0.82 + 0.18 * ease);
    this.uGeo.y = rC;
    this.uGeo.z = q;
    this.uGeo.w = lift;
    this.uUp.x = ux;
    this.uUp.y = uy;
    this.uUp.z = uz;
    this.uEast.x = b[3];
    this.uEast.y = b[4];
    this.uEast.z = b[5];
    this.uNorth.x = b[6];
    this.uNorth.y = b[7];
    this.uNorth.z = b[8];

    const k = rC + lift;
    const cx = ux * k;
    const cy = uy * k;
    const cz = uz * k;
    encodeInto(cx, cy, cz, this.uCentreHigh, this.uCentreLow);
    encodeInto(x, y, z, this.uSatHigh, this.uSatLow);
    this.uToCentre.x = cx - x;
    this.uToCentre.y = cy - y;
    this.uToCentre.z = cz - z;

    // Bounding sphere around satellite + footprint disc.
    const h = dist - k; // satellite height above the cap centre
    const discR = k * Math.sin(cap * DISC_U_MAX);
    const bs = this.boundingSphere;
    bs.center.x = (x + cx) * 0.5;
    bs.center.y = (y + cy) * 0.5;
    bs.center.z = (z + cz) * 0.5;
    bs.radius = Math.sqrt(discR * discR + 0.25 * h * h) + k * (1 - Math.cos(cap * DISC_U_MAX)) + 20_000;

    // Strengths: smoothstep'd fade; the cone gives way to the close-up camera.
    const s = f * f * (3 - 2 * f);
    this.uAnim.y = s;
    const dx = cam.x - x;
    const dy = cam.y - y;
    const dz = cam.z - z;
    const camDist = Math.sqrt(dx * dx + dy * dy + dz * dz);
    let c = (camDist - CONE_NEAR_M) / (CONE_FAR_M - CONE_NEAR_M);
    c = c < 0 ? 0 : c > 1 ? 1 : c;
    c = c * c * (3 - 2 * c);
    this.uAnim.z = s * (CONE_NEAR_STRENGTH + (1 - CONE_NEAR_STRENGTH) * c);
    return true;
  }

  // -------------------------------------------------------------------------
  // GPU resources
  // -------------------------------------------------------------------------

  private createResources(context: Context): void {
    const { vertices, indices } = buildDiscMesh(DISC_SEGMENTS, DISC_RING_STEP, DISC_U_MAX);
    const discBuffer = Buffer.createVertexBuffer({ context, typedArray: vertices, usage: BufferUsage.STATIC_DRAW });
    const discIndex = BufferWithIndex.createIndexBuffer({
      context,
      typedArray: indices,
      usage: BufferUsage.STATIC_DRAW,
      indexDatatype: IndexDatatype.UNSIGNED_SHORT,
    });
    this.discVA = new VertexArray({
      context,
      attributes: [{ index: ATTR_GRID.a_grid, vertexBuffer: discBuffer, componentsPerAttribute: 2, componentDatatype: ComponentDatatype.FLOAT }],
      indexBuffer: discIndex,
    });
    this.discIndexCount = indices.length;

    const coneVerts = buildConeMesh(CONE_SEGMENTS);
    const coneBuffer = Buffer.createVertexBuffer({ context, typedArray: coneVerts, usage: BufferUsage.STATIC_DRAW });
    this.coneVA = new VertexArray({
      context,
      attributes: [{ index: ATTR_GRID.a_grid, vertexBuffer: coneBuffer, componentsPerAttribute: 2, componentDatatype: ComponentDatatype.FLOAT }],
    });
    this.coneVertexCount = coneVerts.length / 2;

    this.discProgram = ShaderProgram.fromCache({
      context,
      vertexShaderSource: new ShaderSource({ sources: [COMMON_GLSL, FOOTPRINT_VS] }),
      fragmentShaderSource: new ShaderSource({ sources: [COMMON_GLSL, HOLO_FS_GLSL, FOOTPRINT_FS] }),
      attributeLocations: { ...ATTR_GRID },
    });
    this.coneProgram = ShaderProgram.fromCache({
      context,
      vertexShaderSource: new ShaderSource({ sources: [COMMON_GLSL, CONE_VS] }),
      fragmentShaderSource: new ShaderSource({ sources: [COMMON_GLSL, HOLO_FS_GLSL, CONE_FS] }),
      attributeLocations: { ...ATTR_GRID },
    });
    this.renderState = RenderState.fromCache(RENDER_STATE_DESC);

    const common = {
      u_up: () => this.uUp,
      u_east: () => this.uEast,
      u_north: () => this.uNorth,
      u_geo: () => this.uGeo,
      u_ellip: () => this.uEllip,
      u_anim: () => this.uAnim,
    };
    this.discCommand = new DrawCommand({
      boundingVolume: this.boundingSphere,
      primitiveType: PrimitiveType.TRIANGLES,
      vertexArray: this.discVA,
      count: this.discIndexCount,
      shaderProgram: this.discProgram,
      uniformMap: {
        ...common,
        u_centreHigh: () => this.uCentreHigh,
        u_centreLow: () => this.uCentreLow,
      },
      renderState: this.renderState,
      pass: Pass.OPAQUE,
      owner: this,
      cull: true,
      occlude: false,
      castShadows: false,
      receiveShadows: false,
    });
    this.coneCommand = new DrawCommand({
      boundingVolume: this.boundingSphere,
      primitiveType: PrimitiveType.TRIANGLES,
      vertexArray: this.coneVA,
      count: this.coneVertexCount,
      shaderProgram: this.coneProgram,
      uniformMap: {
        ...common,
        u_satHigh: () => this.uSatHigh,
        u_satLow: () => this.uSatLow,
        u_toCentre: () => this.uToCentre,
      },
      renderState: this.renderState,
      pass: Pass.OPAQUE,
      owner: this,
      cull: true,
      occlude: false,
      castShadows: false,
      receiveShadows: false,
    });
  }

  private releaseGpu(): void {
    if (this.gpuReleased) return;
    this.gpuReleased = true;
    // VertexArray.destroy() also destroys its vertex/index buffers.
    if (this.discVA && !this.discVA.isDestroyed()) this.discVA.destroy();
    if (this.coneVA && !this.coneVA.isDestroyed()) this.coneVA.destroy();
    if (this.discProgram && !this.discProgram.isDestroyed()) this.discProgram.destroy();
    if (this.coneProgram && !this.coneProgram.isDestroyed()) this.coneProgram.destroy();
    if (this.renderState) RenderState.removeFromCache(RENDER_STATE_DESC);
    this.discVA = this.coneVA = undefined;
    this.discProgram = this.coneProgram = undefined;
    this.renderState = undefined;
    this.discCommand = this.coneCommand = undefined;
  }
}
