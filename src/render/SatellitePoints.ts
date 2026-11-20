/**
 * SatellitePoints — a custom Cesium primitive that draws every satellite in a
 * single GL_POINTS draw call.
 *
 * GPU layout (one VertexArray, `count` vertices):
 *   loc 0/1  a_posHighA / a_posLowA   keyframe slot A, EncodedCartesian3 high/low (float32 x3 each,
 *   loc 2/3  a_posHighB / a_posLowB   keyframe slot B   interleaved in one DYNAMIC_DRAW buffer per slot)
 *   loc 4    a_colorShell             sRGB colour + shell index (4 x uint8, STATIC_DRAW)
 *   loc 5    a_index                  satellite index (float32, STATIC_DRAW)
 *
 * Per frame the CPU only computes a handful of uniforms (u_t = interpolation
 * factor between slot A and slot B, visibility / selection state). A slot's
 * position buffer is re-uploaded (bufferSubData) only when a new keyframe is
 * promoted into it, i.e. about once per simulated second.
 *
 * Occlusion ("behind the Earth") is handled twice, on purpose:
 *  1. Depth test (no depth writes) against whatever Cesium has in the depth
 *     buffer when the OPAQUE pass runs: the globe's depth, or — with the
 *     default globe.depthTestAgainstTerrain = false — Cesium's ellipsoid
 *     DepthPlane. This also gives correct occlusion against terrain and future
 *     3D models. Log depth works because the shaders call czm_vertexLogDepth /
 *     czm_writeLogDepth, which DerivedCommand.createLogDepthCommand detects.
 *  2. An analytic ray/ellipsoid horizon test in the vertex shader against a
 *     slightly shrunk WGS84 ellipsoid, so points behind the planet are hidden
 *     even while globe tiles / the depth plane are not available.
 *
 * Pass choice: Pass.OPAQUE with a custom additive blend state. Commands in
 * Pass.TRANSLUCENT get Cesium's order-independent-translucency derived
 * commands, which replace the blend state (no true additive glow). The OPAQUE
 * pass runs after the GLOBE pass (+ depth plane), so the depth test above is
 * valid, and since this command writes no depth it does not disturb anything
 * drawn later.
 *
 * Laser links (MODELLED topology, see contracts.ts LaserLinks) are drawn by
 * the same primitive so they are in perfect sync with the points: a second
 * VertexArray shares the four position slots + colour/shell buffer and adds
 * an index buffer of GL_LINES pairs, sorted [in-plane..., cross-plane...].
 * Two DrawCommands (offset/count ranges of that one index buffer) with their
 * own colour uniform draw the two kinds; the same u_t interpolation runs in
 * the line vertex shader. The index buffer is only rewritten when the
 * topology changes (~every 10 s). Same pass/blend choice as the points
 * (OPAQUE pass + additive blend, no depth writes): TRANSLUCENT commands get
 * OIT derived commands that would replace the additive blend.
 *
 * Near-camera fade (setModelFade): points closer than `nearM` disappear and
 * fade back in by `farM`, so the dot does not sit on top of a close-up 3D
 * model. Injected into the points vertex shader at load (one uniform).
 */
import {
  BlendEquation,
  BlendFunction,
  BoundingSphere,
  Buffer,
  BufferUsage,
  Cartesian2,
  Cartesian3,
  Cartesian4,
  ComponentDatatype,
  ContextLimits,
  DrawCommand,
  Ellipsoid,
  IndexDatatype,
  JulianDate,
  Matrix4,
  Pass,
  PrimitiveType,
  RenderState,
  SceneMode,
  ShaderProgram,
  ShaderSource,
  VertexArray,
} from 'cesium';
import type { Context, FrameState, Scene } from 'cesium';
import type { Keyframe, LaserLinks, SatCatalog } from '../shared/contracts.ts';
import { SHELLS, SHELL_OTHER } from '../shared/shells.ts';
import { ATTRIBUTE_LOCATIONS, SATELLITE_POINTS_FS, SATELLITE_POINTS_VS } from './shaders/satellitePointsGlsl.ts';
import { scanNearest } from './satelliteModelMath.ts';

/** Simulated milliseconds between two keyframes. */
export const KEYFRAME_STEP_MS = 1000;
/** Keep keyframes available up to this far ahead of the current sim time. */
const LOOKAHEAD_MS = 2 * KEYFRAME_STEP_MS;
/** nextKeyframeTime() stops asking once this many keyframes wait in the queue. */
const MAX_PENDING = 3;
/** How far past the newest keyframe positions are linearly extrapolated. */
const MAX_EXTRAPOLATION_MS = KEYFRAME_STEP_MS;
/**
 * Shells supported by the GPU visibility mask (two vec4 uniforms: indices
 * 0..7). SHELLS has exactly 8 entries (incl. SHELL_V3 = 7); a larger table
 * needs a third uniform in the points and link vertex shaders.
 */
export const MAX_SHELLS = 8;
if (SHELLS.length > MAX_SHELLS) {
  console.warn(`[SatellitePoints] ${SHELLS.length} shells defined, only ${MAX_SHELLS} supported; extra shells draw as OTHER`);
}
/** Bounding-sphere margin over the farthest satellite. */
const BOUNDS_MARGIN_M = 100_000;
/** GPU horizon test uses the ellipsoid shrunk by this factor (~19 km lower). */
const GPU_HORIZON_SHRINK = 0.997;
/** Points fade in over this many ms after the first keyframe is shown. */
const FADE_IN_MS = 1200;
/** Laser links fade in/out over this many ms when toggled / first received. */
const LINK_FADE_MS = 600;

// ---------------------------------------------------------------------------
// Laser link + near-fade shader code (lives here: the points GLSL module is
// shared, this primitive owns the link program).
// ---------------------------------------------------------------------------

/** Attribute subset used by the link program (same locations as the points VA). */
const LINK_ATTRIBUTE_LOCATIONS = {
  a_posHighA: ATTRIBUTE_LOCATIONS.a_posHighA,
  a_posLowA: ATTRIBUTE_LOCATIONS.a_posLowA,
  a_posHighB: ATTRIBUTE_LOCATIONS.a_posHighB,
  a_posLowB: ATTRIBUTE_LOCATIONS.a_posLowB,
  a_colorShell: ATTRIBUTE_LOCATIONS.a_colorShell,
} as const;

const LASER_LINKS_VS = /* glsl */ `
in vec3 a_posHighA;
in vec3 a_posLowA;
in vec3 a_posHighB;
in vec3 a_posLowB;
in vec4 a_colorShell;

uniform float u_t;
uniform vec4 u_shellVis0;
uniform vec4 u_shellVis1;
// w: horizon ellipsoid inverse-scale (shared with the points)
uniform vec4 u_state;
// rgb: colour (0..1, gamma-corrected in the FS), a: base alpha
uniform vec4 u_linkColor;
// x: distance (m) with full alpha, y: distance with z * alpha, w: global intensity
uniform vec4 u_linkFade;

out float v_vis;
out float v_occ;
out float v_alpha;
out vec3 v_rgb;

void main()
{
    float wA = 1.0 - u_t;
    float wB = u_t;
    bool validA = dot(a_posHighA, a_posHighA) > 0.0;
    bool validB = dot(a_posHighB, a_posHighB) > 0.0;
    bool valid = (validA || wA == 0.0) && (validB || wB == 0.0);

    float shell = a_colorShell.w;
    vec4 m0 = step(abs(vec4(0.0, 1.0, 2.0, 3.0) - shell), vec4(0.5));
    vec4 m1 = step(abs(vec4(4.0, 5.0, 6.0, 7.0) - shell), vec4(0.5));
    float vis = dot(m0, u_shellVis0) + dot(m1, u_shellVis1);

    vec3 relA = (a_posHighA - czm_encodedCameraPositionMCHigh) + (a_posLowA - czm_encodedCameraPositionMCLow);
    vec3 relB = (a_posHighB - czm_encodedCameraPositionMCHigh) + (a_posLowB - czm_encodedCameraPositionMCLow);
    vec3 rel = relA * wA + relB * wB;

    // Same analytic horizon test as the points (the depth test does the exact
    // per-fragment occlusion; this removes the hidden half of limb-crossing
    // links while globe tiles / the depth plane are not there yet).
    bool occluded = false;
    vec3 invRadii = czm_ellipsoidInverseRadii * u_state.w;
    vec3 e = czm_viewerPositionWC * invRadii;
    vec3 d = rel * invRadii;
    float c = dot(e, e) - 1.0;
    if (c > 0.0) {
        float a = dot(d, d);
        float b = dot(e, d);
        float disc = b * b - a * c;
        if (disc > 0.0 && a > 0.0) {
            float tHit = (-b - sqrt(disc)) / a;
            occluded = tHit > 0.0 && tHit < 1.0;
        }
    }

    float dist = length(rel);
    float f = clamp((dist - u_linkFade.x) / max(u_linkFade.y - u_linkFade.x, 1.0), 0.0, 1.0);
    f = f * f * (3.0 - 2.0 * f);
    v_alpha = u_linkColor.a * mix(1.0, u_linkFade.z, f) * u_linkFade.w;
    v_vis = (valid && vis > 0.5) ? 1.0 : 0.0;
    v_occ = occluded ? 1.0 : 0.0;
    v_rgb = czm_gammaCorrect(u_linkColor.rgb);

    gl_Position = czm_modelViewProjectionRelativeToEye * vec4(rel, 1.0);
    czm_vertexLogDepth();
}
`;

const LASER_LINKS_FS = /* glsl */ `
in float v_vis;
in float v_occ;
in float v_alpha;
in vec3 v_rgb;

void main()
{
    // v_vis is 1 only if BOTH endpoints are valid and in a visible shell.
    if (v_vis < 0.999 || v_occ > 0.5 || v_alpha <= 0.0) {
        discard;
    }
    // Additive (ONE, ONE): premultiplied colour.
    out_FragColor = vec4(v_rgb * v_alpha, v_alpha);
    czm_writeLogDepth();
}
`;

/** Adds the u_modelFade near-camera fade to the shared points vertex shader. */
function injectModelFade(src: string): string {
  const decl = 'uniform float u_maxPointSize;';
  const assign = 'v_intensity = u_state.z;';
  if (!src.includes(decl) || !src.includes(assign)) {
    console.warn('[SatellitePoints] points shader changed: near-camera model fade disabled');
    return src;
  }
  return src
    .replace(decl, `${decl}\n// x: near distance (fully faded), y: far distance (fully visible); y <= x disables\nuniform vec2 u_modelFade;`)
    .replace(
      assign,
      'v_intensity = u_state.z * (u_modelFade.y > u_modelFade.x ? clamp((dist - u_modelFade.x) / (u_modelFade.y - u_modelFade.x), 0.0, 1.0) : 1.0);',
    );
}

const SATELLITE_POINTS_VS_FADE = injectModelFade(SATELLITE_POINTS_VS);

/** Visual tuning of the laser-link web (uniform-driven). */
export interface LaserLinkStyle {
  /** In-plane link colour, 0..1 sRGB-ish (gamma corrected like the points). */
  color: readonly [number, number, number];
  /** In-plane base alpha (additive intensity). */
  alpha: number;
  /** Cross-plane link colour. */
  crossColor: readonly [number, number, number];
  /** Cross-plane base alpha. */
  crossAlpha: number;
  /** Camera distance (m) up to which links have full alpha. */
  nearM: number;
  /** Camera distance (m) from which links have alpha * farAlpha. */
  farM: number;
  /** Alpha multiplier at/after farM (overview reads as a fine web). */
  farAlpha: number;
}

/**
 * Laser palette: a restrained SpaceX-style blue. In-plane links rgb(110,180,255),
 * cross-plane links slightly cooler and dimmer rgb(80,150,255). Exported so the
 * HUD swatch / info panel can match the web exactly.
 */
export const LASER_COLOR_RGB: readonly [number, number, number] = [110 / 255, 180 / 255, 255 / 255];
export const LASER_CROSS_COLOR_RGB: readonly [number, number, number] = [80 / 255, 150 / 255, 255 / 255];
export const LASER_COLOR_CSS = 'rgb(110, 180, 255)';
export const LASER_CROSS_COLOR_CSS = 'rgb(80, 150, 255)';

export const DEFAULT_LASER_LINK_STYLE: Readonly<LaserLinkStyle> = {
  color: LASER_COLOR_RGB,
  alpha: 0.34,
  crossColor: LASER_CROSS_COLOR_RGB,
  crossAlpha: 0.2,
  nearM: 1_500_000,
  farM: 12_000_000,
  farAlpha: 0.42,
};

/** Private Cesium API (Renderer/Buffer.js), not in the shared typings. */
interface IndexBufferLike {
  readonly sizeInBytes: number;
  copyFromArrayView(arrayView: ArrayBufferView, offsetInBytes?: number): void;
  isDestroyed(): boolean;
  destroy(): undefined;
}
const BufferWithIndex = Buffer as unknown as {
  createIndexBuffer(options: { context: Context; sizeInBytes: number; usage: number; indexDatatype: number }): IndexBufferLike;
};
type RangeCommand = DrawCommand & { offset: number };

/** Visual tuning knobs (all uniform-driven, no buffer re-upload). */
export interface SatellitePointsStyle {
  /** Core size constant: corePx = k / sqrt(distanceMetres), in CSS px. */
  sizeK: number;
  /** Clamp of the core diameter in CSS px. */
  minSizePx: number;
  maxSizePx: number;
  /** Sprite diameter / core diameter (extent of the glow halo). */
  glowScale: number;
  /** Overall intensity multiplier (additive blending). */
  brightness: number;
  /** Linear 0..1 RGB of the selected satellite. */
  selectedColor: readonly [number, number, number];
  /** Linear 0..1 RGB tint of the hovered satellite. */
  hoveredColor: readonly [number, number, number];
}

export const DEFAULT_SATELLITE_STYLE: Readonly<SatellitePointsStyle> = {
  sizeK: 9500,
  minSizePx: 1.7,
  maxSizePx: 9,
  glowScale: 4,
  brightness: 1.15,
  selectedColor: [0.35, 0.93, 1.0],
  hoveredColor: [1.0, 1.0, 1.0],
};

export interface SatellitePointsOptions {
  catalog: SatCatalog;
  /**
   * Extra size multiplier (default 1). devicePixelRatio and
   * viewer.resolutionScale are already compensated automatically through
   * czm_pixelRatio, so points keep the same CSS-pixel size on screen.
   */
  pixelRatio?: number;
  /** Depth test against the globe / depth plane (default true). */
  depthTest?: boolean;
  style?: Partial<SatellitePointsStyle>;
  /**
   * Fade the points and the link web in when first shown (default true). A
   * catalogue hot swap passes false: the replacement shows at full strength
   * straight away instead of blanking the sky for FADE_IN_MS.
   */
  fadeIn?: boolean;
}

// ---------------------------------------------------------------------------
// Time helper
// ---------------------------------------------------------------------------

let cachedLeapDay = Number.NaN;
let cachedLeapOffsetMs = 0;

/**
 * Converts a Cesium JulianDate (stored internally as TAI) to Unix epoch
 * milliseconds (UTC), without allocating. Leap seconds are looked up only when
 * the Julian day number changes.
 */
export function julianToUnixMs(jd: JulianDate): number {
  if (jd.dayNumber !== cachedLeapDay) {
    cachedLeapDay = jd.dayNumber;
    cachedLeapOffsetMs = -JulianDate.computeTaiMinusUtc(jd) * 1000;
  }
  return (jd.dayNumber - 2440587.5) * 86_400_000 + jd.secondsOfDay * 1000 + cachedLeapOffsetMs;
}

// ---------------------------------------------------------------------------
// Scratch (module level: never allocate on hot paths)
// ---------------------------------------------------------------------------

const scratchViewProjection = new Matrix4();
const WGS84_RADII = Ellipsoid.WGS84.radii;
const INV_RX = 1 / WGS84_RADII.x;
const INV_RY = 1 / WGS84_RADII.y;
const INV_RZ = 1 / WGS84_RADII.z;

/** EncodedCartesian3.encode semantics: high = multiple of 65536, low = remainder. */
function encodeHigh(value: number): number {
  if (value >= 0) return Math.floor(value / 65536) * 65536;
  return -Math.floor(-value / 65536) * 65536;
}

/**
 * Packs ECEF metres into interleaved float32 [hx,hy,hz,lx,ly,lz] per satellite.
 * Invalid (NaN) satellites are written as all zeros, which the vertex shader
 * treats as "hidden" (no real satellite sits within 65 km of Earth's centre).
 * Returns the largest valid radius (for the bounding sphere).
 */
export function encodePositions(src: Float64Array, dst: Float32Array, count: number): number {
  let maxR2 = 0;
  for (let i = 0; i < count; i++) {
    const s = i * 3;
    const d = i * 6;
    const x = src[s];
    const y = src[s + 1];
    const z = src[s + 2];
    // NaN fails every comparison, so this also filters NaN/Infinity.
    if (!(Math.abs(x) < 1e12 && Math.abs(y) < 1e12 && Math.abs(z) < 1e12)) {
      dst[d] = 0;
      dst[d + 1] = 0;
      dst[d + 2] = 0;
      dst[d + 3] = 0;
      dst[d + 4] = 0;
      dst[d + 5] = 0;
      continue;
    }
    const hx = encodeHigh(x);
    const hy = encodeHigh(y);
    const hz = encodeHigh(z);
    dst[d] = hx;
    dst[d + 1] = hy;
    dst[d + 2] = hz;
    dst[d + 3] = x - hx;
    dst[d + 4] = y - hy;
    dst[d + 5] = z - hz;
    const r2 = x * x + y * y + z * z;
    if (r2 > maxR2) maxR2 = r2;
  }
  return Math.sqrt(maxR2);
}

// ---------------------------------------------------------------------------
// Primitive
// ---------------------------------------------------------------------------

export class SatellitePoints {
  /** Number of satellites (vertices). */
  readonly count: number;
  /** Hide/show the whole primitive. */
  show = true;
  /** Simulated ms between keyframes (see nextKeyframeTime). */
  readonly keyframeStepMs = KEYFRAME_STEP_MS;
  /**
   * Receives position buffers this primitive no longer references, so the
   * integrator can hand them back to the worker (transfer) for reuse.
   */
  onRecycle?: (buf: Float64Array) => void;

  private readonly catalog: SatCatalog;
  private readonly sizeMultiplier: number;
  private depthTestEnabled: boolean;

  // --- keyframe timeline (CPU) ---
  private readonly slotTime = new Float64Array([Number.NaN, Number.NaN]);
  private readonly slotPos: [Float64Array | null, Float64Array | null] = [null, null];
  private readonly slotRadius = new Float64Array(2);
  /** Number of slots holding a keyframe of the current timeline (0..2). Slot 0 fills first. */
  private filled = 0;
  /** Slot holding the newest keyframe. */
  private newest = 0;
  private readonly pending: Keyframe[] = [];
  /** Most recent "now" seen via update() or nextKeyframeTime(). */
  private refNowMs = Number.NaN;
  /** Interpolation factor slot A -> slot B used for the last frame. */
  private t = 0;
  private fadeStart = -1;

  // --- uniforms (mutated in place, never reallocated) ---
  private readonly uShellVis0 = new Cartesian4(1, 1, 1, 1);
  private readonly uShellVis1 = new Cartesian4(1, 1, 1, 1);
  private readonly uSize = new Cartesian4();
  private readonly uState = new Cartesian4(-1, -1, 1, 1 / GPU_HORIZON_SHRINK);
  private readonly uSelectedColor = new Cartesian4(0, 0, 0, 1);
  private readonly uHoveredColor = new Cartesian4(0, 0, 0, 1);
  private uMaxPointSize = 64;
  private brightness = 1;
  private readonly shellVisible = new Uint8Array(MAX_SHELLS).fill(1);
  private selected = -1;
  private hovered = -1;

  // --- GPU resources (created lazily on first update) ---
  private vertexArray: VertexArray | undefined;
  private posBuffers: [Buffer | undefined, Buffer | undefined] = [undefined, undefined];
  private shaderProgram: ShaderProgram | undefined;
  private renderStateDesc: object | undefined;
  private renderState: RenderState | undefined;
  private command: DrawCommand | undefined;
  private readonly staging: Float32Array;
  private readonly boundingSphere = new BoundingSphere(Cartesian3.ZERO, 6_378_137 + 2_000_000);
  private readonly uniformMap: Record<string, () => unknown>;
  private destroyed = false;
  /** Shared vertex buffers (not destroyed by either VertexArray; see destroy()). */
  private sharedBuffers: Buffer[] = [];

  // --- near-camera fade (3D model replaces the dot) ---
  private readonly uModelFade = new Cartesian2(-2, -1);

  // --- laser links ---
  private linkStaging: Uint16Array | Uint32Array | null = null;
  /** Index counts (2 per link) of the in-plane / cross-plane ranges. */
  private linkInIdx = 0;
  private linkCrossIdx = 0;
  private linksDirty = false;
  private linksVisible = true;
  private linkIntensity = 0;
  /** fadeIn: false -> the first link frame jumps to its target intensity instead of fading. */
  private linkSnap = false;
  private linkLastWall = -1;
  private linkVertexArray: VertexArray | undefined;
  private linkIndexBuffer: IndexBufferLike | undefined;
  private linkIndexCapacity = 0;
  private linkProgram: ShaderProgram | undefined;
  private linkInCommand: RangeCommand | undefined;
  private linkCrossCommand: RangeCommand | undefined;
  private readonly uLinkColorIn = new Cartesian4();
  private readonly uLinkColorCross = new Cartesian4();
  private readonly uLinkFade = new Cartesian4(0, 1, 1, 0);

  constructor(opts: SatellitePointsOptions) {
    this.catalog = opts.catalog;
    this.count = opts.catalog.count;
    this.sizeMultiplier = opts.pixelRatio && opts.pixelRatio > 0 ? opts.pixelRatio : 1;
    this.depthTestEnabled = opts.depthTest ?? true;
    this.staging = new Float32Array(Math.max(1, this.count) * 6);
    this.setStyle({ ...DEFAULT_SATELLITE_STYLE, ...opts.style });
    this.setLinkStyle(DEFAULT_LASER_LINK_STYLE);
    if (opts.fadeIn === false) {
      // age = now - 0 is far past FADE_IN_MS: fade = 1 from the first frame.
      this.fadeStart = 0;
      this.linkSnap = true;
    }

    this.uniformMap = {
      u_t: () => this.t,
      u_shellVis0: () => this.uShellVis0,
      u_shellVis1: () => this.uShellVis1,
      u_size: () => this.uSize,
      u_state: () => this.uState,
      u_selectedColor: () => this.uSelectedColor,
      u_hoveredColor: () => this.uHoveredColor,
      u_maxPointSize: () => this.uMaxPointSize,
      u_modelFade: () => this.uModelFade,
    };
  }

  // -------------------------------------------------------------------------
  // Laser links (MODELLED topology)
  // -------------------------------------------------------------------------

  /**
   * Sets the (modelled) laser-link topology drawn as GL_LINES between the
   * satellites' interpolated positions, or null to remove it. Pairs with an
   * out-of-range or identical index are skipped. The GPU index buffer is
   * rewritten on the next frame (no per-frame cost while links are static).
   */
  setLinks(links: LaserLinks | null): void {
    if (this.destroyed) return;
    if (!links) {
      this.linkInIdx = 0;
      this.linkCrossIdx = 0;
      this.linksDirty = false;
      return;
    }
    const pairs = links.pairs;
    const kinds = links.kinds;
    const n = Math.min(kinds.length, pairs.length >> 1);
    const needed = n * 2;
    let staging = this.linkStaging;
    if (!staging || staging.length < needed) {
      const cap = Math.max(256, Math.ceil(needed * 1.25));
      staging = this.count <= 65535 ? new Uint16Array(cap) : new Uint32Array(cap);
      this.linkStaging = staging;
    }
    const count = this.count;
    let w = 0;
    for (let pass = 0; pass < 2; pass++) {
      for (let k = 0; k < n; k++) {
        if ((kinds[k] === 1 ? 1 : 0) !== pass) continue;
        const a = pairs[k * 2];
        const b = pairs[k * 2 + 1];
        if (a >= count || b >= count || a === b) continue;
        staging[w++] = a;
        staging[w++] = b;
      }
      if (pass === 0) this.linkInIdx = w;
    }
    this.linkCrossIdx = w - this.linkInIdx;
    this.linksDirty = true;
  }

  /** Number of links currently uploaded/drawable (in-plane + cross-plane). */
  get linkCount(): number {
    return (this.linkInIdx + this.linkCrossIdx) >> 1;
  }

  /** Show/hide the laser-link web (fades over ~0.6 s). */
  setLinksVisible(on: boolean): void {
    this.linksVisible = on;
  }

  get linksShown(): boolean {
    return this.linksVisible;
  }

  setLinkStyle(style: Partial<LaserLinkStyle>): void {
    if (style.color) {
      this.uLinkColorIn.x = style.color[0];
      this.uLinkColorIn.y = style.color[1];
      this.uLinkColorIn.z = style.color[2];
    }
    if (style.alpha !== undefined) this.uLinkColorIn.w = Math.max(0, style.alpha);
    if (style.crossColor) {
      this.uLinkColorCross.x = style.crossColor[0];
      this.uLinkColorCross.y = style.crossColor[1];
      this.uLinkColorCross.z = style.crossColor[2];
    }
    if (style.crossAlpha !== undefined) this.uLinkColorCross.w = Math.max(0, style.crossAlpha);
    if (style.nearM !== undefined) this.uLinkFade.x = style.nearM;
    if (style.farM !== undefined) this.uLinkFade.y = style.farM;
    if (style.farAlpha !== undefined) this.uLinkFade.z = Math.max(0, style.farAlpha);
  }

  /**
   * Near-camera point fade: points closer to the camera than `nearM` are
   * invisible, fully visible again from `farM`. farM <= nearM disables it.
   */
  setModelFade(nearM: number, farM: number): void {
    if (Number.isFinite(nearM) && Number.isFinite(farM) && farM > nearM) {
      this.uModelFade.x = nearM;
      this.uModelFade.y = farM;
    } else {
      this.uModelFade.x = -2;
      this.uModelFade.y = -1;
    }
  }

  // -------------------------------------------------------------------------
  // Keyframe management
  // -------------------------------------------------------------------------

  /**
   * Adds a keyframe. Keyframes are expected in increasing time, one
   * KEYFRAME_STEP_MS apart (as returned by nextKeyframeTime). A keyframe that
   * does not fit the current timeline is either used to restart it (when the
   * clock jumped and the keyframe is near the current time) or recycled.
   * Ownership of `kf.positions` passes to this object; it is handed back via
   * onRecycle once no longer needed.
   */
  addKeyframe(kf: Keyframe): void {
    if (this.destroyed) return;
    if (kf.positions.length !== this.count * 3 || !Number.isFinite(kf.timeMs)) {
      console.warn('[SatellitePoints] ignoring keyframe with wrong size/time', kf.timeMs, kf.positions.length);
      this.recycle(kf.positions);
      return;
    }
    const latest = this.latestTime();
    if (latest === null) {
      this.pending.push(kf);
      return;
    }
    const step = this.keyframeStepMs;
    if (kf.timeMs > latest && kf.timeMs - latest <= step * 1.5) {
      this.pending.push(kf);
      return;
    }
    // Out of sequence. Restart the timeline with it if the clock left the
    // covered interval and this keyframe is close to "now".
    const ref = this.refNowMs;
    const earliest = this.earliestTime() as number;
    if (Number.isFinite(ref) && Math.abs(kf.timeMs - ref) <= step * 1.5 && this.isJumped(ref, earliest, latest)) {
      this.resetTimeline();
      this.pending.push(kf);
      return;
    }
    this.recycle(kf.positions);
  }

  /**
   * Which keyframe time (Unix ms, sim time) the integrator should request
   * next, or null when the timeline is covered far enough ahead. Side-effect
   * free apart from remembering `nowMs`; call it while no request is in
   * flight (it returns the same time again until that keyframe is added).
   * If the clock jumped outside the covered interval it returns the step-
   * aligned time at/just before `nowMs` (the timeline restarts there).
   */
  nextKeyframeTime(nowMs: number): number | null {
    if (!Number.isFinite(nowMs)) return null;
    this.refNowMs = nowMs;
    const step = this.keyframeStepMs;
    const earliest = this.earliestTime();
    const latest = this.latestTime();
    if (earliest === null || latest === null || this.isJumped(nowMs, earliest, latest)) {
      return Math.floor(nowMs / step) * step;
    }
    if (this.pending.length >= MAX_PENDING) return null;
    if (latest < nowMs + LOOKAHEAD_MS) return latest + step;
    return null;
  }

  /** Time span [earliest, latest] currently held (slots + queue), or null. */
  coveredRange(): { startMs: number; endMs: number } | null {
    const a = this.earliestTime();
    const b = this.latestTime();
    return a === null || b === null ? null : { startMs: a, endMs: b };
  }

  /**
   * True when the held keyframes can render `nowMs` (inside [earliest,
   * latest + max extrapolation]): false before the first keyframe and after
   * a clock jump until the restart keyframe is in. Allocation-free.
   */
  coversTime(nowMs: number): boolean {
    const a = this.earliestTime();
    const b = this.latestTime();
    return a !== null && b !== null && !this.isJumped(nowMs, a, b);
  }

  private earliestTime(): number | null {
    if (this.filled === 0) return this.pending.length > 0 ? this.pending[0].timeMs : null;
    if (this.filled === 1) return this.slotTime[0];
    return Math.min(this.slotTime[0], this.slotTime[1]);
  }

  private latestTime(): number | null {
    if (this.pending.length > 0) return this.pending[this.pending.length - 1].timeMs;
    if (this.filled === 0) return null;
    return this.slotTime[this.newest];
  }

  private isJumped(nowMs: number, earliest: number, latest: number): boolean {
    return nowMs < earliest || nowMs > latest + MAX_EXTRAPOLATION_MS;
  }

  private resetTimeline(): void {
    for (let s = 0; s < 2; s++) {
      const buf = this.slotPos[s];
      this.slotPos[s] = null;
      this.slotTime[s] = Number.NaN;
      if (buf) this.recycle(buf);
    }
    this.filled = 0;
    this.newest = 0;
    while (this.pending.length > 0) {
      this.recycle((this.pending.shift() as Keyframe).positions);
    }
  }

  private recycle(buf: Float64Array): void {
    const cb = this.onRecycle;
    if (!cb) return;
    try {
      cb(buf);
    } catch (err) {
      console.warn('[SatellitePoints] onRecycle threw', err);
    }
  }

  /** Promotes queued keyframes into GPU slots as sim time advances. */
  private advance(nowMs: number): void {
    let uploads = 0;
    while (this.pending.length > 0 && uploads < 4) {
      let slot: number;
      if (this.filled === 0) slot = 0;
      else if (this.filled === 1) slot = 1;
      else {
        // Interval [older, newest] still current: keep it.
        if (!(nowMs >= this.slotTime[this.newest])) break;
        slot = 1 - this.newest;
      }
      this.loadSlot(slot, this.pending.shift() as Keyframe);
      uploads++;
    }
  }

  private loadSlot(slot: number, kf: Keyframe): void {
    const old = this.slotPos[slot];
    this.slotPos[slot] = kf.positions;
    this.slotTime[slot] = kf.timeMs;
    this.newest = slot;
    if (this.filled < slot + 1) this.filled = slot + 1;
    if (old && old !== kf.positions) this.recycle(old);

    this.slotRadius[slot] = encodePositions(kf.positions, this.staging, this.count);
    const buffer = this.posBuffers[slot];
    if (buffer) buffer.copyFromArrayView(this.staging, 0);
    const r = Math.max(this.slotRadius[0], this.filled > 1 ? this.slotRadius[1] : 0, 6_378_137);
    this.boundingSphere.radius = r + BOUNDS_MARGIN_M;
  }

  /** Interpolation factor (slot 0 -> slot 1) for a given sim time. */
  private computeT(nowMs: number): number {
    if (this.filled < 2) return 0;
    const t0 = this.slotTime[0];
    const t1 = this.slotTime[1];
    const lo = t0 < t1 ? t0 : t1;
    const hi = t0 < t1 ? t1 : t0;
    let tc = nowMs;
    if (!(tc >= lo)) tc = lo; // also catches NaN
    if (tc > hi + MAX_EXTRAPOLATION_MS) tc = hi + MAX_EXTRAPOLATION_MS;
    return (tc - t0) / (t1 - t0);
  }

  // -------------------------------------------------------------------------
  // Uniform-driven state
  // -------------------------------------------------------------------------

  setShellVisible(shellIndex: number, visible: boolean): void {
    if (!(shellIndex >= 0 && shellIndex < MAX_SHELLS)) return;
    const v = visible ? 1 : 0;
    this.shellVisible[shellIndex] = v;
    const target = shellIndex < 4 ? this.uShellVis0 : this.uShellVis1;
    switch (shellIndex & 3) {
      case 0:
        target.x = v;
        break;
      case 1:
        target.y = v;
        break;
      case 2:
        target.z = v;
        break;
      default:
        target.w = v;
    }
  }

  isShellVisible(shellIndex: number): boolean {
    return shellIndex >= 0 && shellIndex < MAX_SHELLS && this.shellVisible[shellIndex] === 1;
  }

  /** Selected satellite (-1 = none): drawn larger in the accent colour, always visible. */
  setSelected(index: number): void {
    this.selected = Number.isInteger(index) && index >= 0 && index < this.count ? index : -1;
    this.uState.x = this.selected;
  }

  get selectedIndex(): number {
    return this.selected;
  }

  /** Hovered satellite (-1 = none): drawn slightly larger and brighter. */
  setHovered(index: number): void {
    this.hovered = Number.isInteger(index) && index >= 0 && index < this.count ? index : -1;
    this.uState.y = this.hovered;
  }

  get hoveredIndex(): number {
    return this.hovered;
  }

  setStyle(style: Partial<SatellitePointsStyle>): void {
    const m = this.sizeMultiplier;
    if (style.sizeK !== undefined) this.uSize.x = style.sizeK * m;
    if (style.minSizePx !== undefined) this.uSize.y = style.minSizePx * m;
    if (style.maxSizePx !== undefined) this.uSize.z = style.maxSizePx * m;
    if (style.glowScale !== undefined) this.uSize.w = Math.max(1, style.glowScale);
    if (style.brightness !== undefined) this.brightness = Math.max(0, style.brightness);
    if (style.selectedColor) {
      this.uSelectedColor.x = style.selectedColor[0];
      this.uSelectedColor.y = style.selectedColor[1];
      this.uSelectedColor.z = style.selectedColor[2];
    }
    if (style.hoveredColor) {
      this.uHoveredColor.x = style.hoveredColor[0];
      this.uHoveredColor.y = style.hoveredColor[1];
      this.uHoveredColor.z = style.hoveredColor[2];
    }
  }

  /** Enables/disables depth testing against the globe (rebuilds the cached render state). */
  setDepthTest(enabled: boolean): void {
    if (enabled === this.depthTestEnabled) return;
    this.depthTestEnabled = enabled;
    if (this.command) {
      this.releaseRenderState();
      this.createRenderState();
      this.command.renderState = this.renderState;
      if (this.linkInCommand) this.linkInCommand.renderState = this.renderState;
      if (this.linkCrossCommand) this.linkCrossCommand.renderState = this.renderState;
    }
  }

  // -------------------------------------------------------------------------
  // CPU-side queries (phase 2)
  // -------------------------------------------------------------------------

  /**
   * ECEF position (metres) of satellite `index`, interpolated exactly like the
   * GPU does. `timeMs` defaults to the time of the last rendered frame.
   * Returns undefined when unknown (no keyframe, failed propagation, bad index).
   */
  getPosition(index: number, out: Cartesian3, timeMs?: number): Cartesian3 | undefined {
    if (!(index >= 0 && index < this.count) || this.filled === 0) return undefined;
    const t = timeMs === undefined ? this.t : this.computeT(timeMs);
    const i3 = index * 3;
    const a = this.slotPos[0] as Float64Array;
    if (this.filled === 1 || t === 0) {
      const x = a[i3];
      const y = a[i3 + 1];
      const z = a[i3 + 2];
      if (Number.isNaN(x) || Number.isNaN(y) || Number.isNaN(z)) return undefined;
      out.x = x;
      out.y = y;
      out.z = z;
      return out;
    }
    const b = this.slotPos[1] as Float64Array;
    const bx = b[i3];
    const by = b[i3 + 1];
    const bz = b[i3 + 2];
    if (Number.isNaN(bx) || Number.isNaN(by) || Number.isNaN(bz)) return undefined;
    if (t === 1) {
      out.x = bx;
      out.y = by;
      out.z = bz;
      return out;
    }
    const ax = a[i3];
    const ay = a[i3 + 1];
    const az = a[i3 + 2];
    if (Number.isNaN(ax) || Number.isNaN(ay) || Number.isNaN(az)) return undefined;
    const wa = 1 - t;
    out.x = ax * wa + bx * t;
    out.y = ay * wa + by * t;
    out.z = az * wa + bz * t;
    return out;
  }

  /**
   * ECEF velocity (m/s) of satellite `index`: the exact derivative of the
   * GPU's linear interpolation between the two keyframe slots (constant
   * within one keyframe interval). Undefined until two keyframes are loaded
   * or when either sample is invalid.
   */
  getVelocity(index: number, out: Cartesian3): Cartesian3 | undefined {
    if (!(index >= 0 && index < this.count) || this.filled < 2) return undefined;
    const dt = (this.slotTime[1] - this.slotTime[0]) / 1000;
    if (!(dt !== 0)) return undefined;
    const a = this.slotPos[0] as Float64Array;
    const b = this.slotPos[1] as Float64Array;
    const i3 = index * 3;
    const vx = (b[i3] - a[i3]) / dt;
    const vy = (b[i3 + 1] - a[i3 + 1]) / dt;
    const vz = (b[i3 + 2] - a[i3 + 2]) / dt;
    if (Number.isNaN(vx) || Number.isNaN(vy) || Number.isNaN(vz)) return undefined;
    out.x = vx;
    out.y = vy;
    out.z = vz;
    return out;
  }

  /**
   * The satellites within `radiusM` of `center` (interpolated at `timeMs`,
   * default: last rendered frame), nearest first: indices in outIndex and
   * squared distances in outDist2, at most outIndex.length of them. Hidden
   * shells are skipped (except the selected satellite). Returns the count.
   * Allocation-free; an early-out skips the scan when the camera is far
   * outside the constellation.
   */
  findNearest(center: Cartesian3, radiusM: number, outIndex: Int32Array, outDist2: Float64Array, timeMs?: number): number {
    if (this.filled === 0 || !(radiusM > 0)) return 0;
    const maxR = Math.max(this.slotRadius[0], this.filled > 1 ? this.slotRadius[1] : 0);
    const cr = Math.sqrt(center.x * center.x + center.y * center.y + center.z * center.z);
    if (cr - maxR > radiusM) return 0;
    const t = timeMs === undefined ? this.t : this.computeT(timeMs);
    const single = this.filled === 1 || t === 0;
    const a = this.slotPos[0] as Float64Array;
    const b = (single ? a : this.slotPos[1]) as Float64Array;
    return scanNearest(
      a,
      b,
      single ? 1 : 1 - t,
      single ? 0 : t,
      this.count,
      center.x,
      center.y,
      center.z,
      radiusM,
      this.catalog.shell,
      this.shellVisible,
      this.selected,
      outIndex,
      outDist2,
    );
  }

  /**
   * Index of the visible satellite nearest to `windowPosition` (CSS pixels,
   * as delivered by ScreenSpaceEventHandler) within `radiusPx`, or -1.
   * Skips hidden shells, invalid satellites and satellites behind the Earth.
   * Allocation-free; ~10k points take well under 1 ms. 3D scene mode only.
   */
  pick(scene: Scene, windowPosition: Cartesian2, radiusPx = 12): number {
    if (this.filled === 0 || !this.show) return -1;
    const camera = scene.camera;
    const m = Matrix4.multiply(camera.frustum.projectionMatrix, camera.viewMatrix, scratchViewProjection);
    const canvas = scene.canvas;
    const halfW = canvas.clientWidth * 0.5;
    const halfH = canvas.clientHeight * 0.5;
    const px = windowPosition.x;
    const py = windowPosition.y;
    const eye = camera.positionWC;
    const ex = eye.x * INV_RX;
    const ey = eye.y * INV_RY;
    const ez = eye.z * INV_RZ;
    const c = ex * ex + ey * ey + ez * ez - 1;

    const m0 = m[0], m1 = m[1], m3 = m[3];
    const m4 = m[4], m5 = m[5], m7 = m[7];
    const m8 = m[8], m9 = m[9], m11 = m[11];
    const m12 = m[12], m13 = m[13], m15 = m[15];

    const t = this.t;
    const single = this.filled === 1 || t === 0;
    const a = this.slotPos[0] as Float64Array;
    const b = (single ? a : this.slotPos[1]) as Float64Array;
    const wa = single ? 1 : 1 - t;
    const wb = single ? 0 : t;
    const shells = this.catalog.shell;
    const vis = this.shellVisible;
    const selected = this.selected;

    let best = radiusPx * radiusPx;
    let bestIndex = -1;
    for (let i = 0, n = this.count; i < n; i++) {
      if (vis[shells[i]] !== 1 && i !== selected) continue;
      const i3 = i * 3;
      let x: number, y: number, z: number;
      if (wb === 0) {
        x = a[i3];
        y = a[i3 + 1];
        z = a[i3 + 2];
      } else if (wa === 0) {
        x = b[i3];
        y = b[i3 + 1];
        z = b[i3 + 2];
      } else {
        x = a[i3] * wa + b[i3] * wb;
        y = a[i3 + 1] * wa + b[i3 + 1] * wb;
        z = a[i3 + 2] * wa + b[i3 + 2] * wb;
      }
      const cw = m3 * x + m7 * y + m11 * z + m15;
      if (!(cw > 0)) continue; // behind the camera, or NaN
      const inv = 1 / cw;
      const sx = ((m0 * x + m4 * y + m8 * z + m12) * inv + 1) * halfW;
      const sy = (1 - (m1 * x + m5 * y + m9 * z + m13) * inv) * halfH;
      const dx = sx - px;
      const dy = sy - py;
      const d2 = dx * dx + dy * dy;
      if (!(d2 < best)) continue;
      // Horizon test (exact WGS84 ellipsoid, unit-sphere space).
      if (c > 0) {
        const qx = (x - eye.x) * INV_RX;
        const qy = (y - eye.y) * INV_RY;
        const qz = (z - eye.z) * INV_RZ;
        const qa = qx * qx + qy * qy + qz * qz;
        const qb = ex * qx + ey * qy + ez * qz;
        const disc = qb * qb - qa * c;
        if (disc > 0 && qa > 0) {
          const tHit = (-qb - Math.sqrt(disc)) / qa;
          if (tHit > 0 && tHit < 1) continue;
        }
      }
      best = d2;
      bestIndex = i;
    }
    return bestIndex;
  }

  // -------------------------------------------------------------------------
  // Cesium primitive interface
  // -------------------------------------------------------------------------

  update(frameState: FrameState): void {
    if (this.destroyed || this.count === 0) return;
    const nowMs = julianToUnixMs(frameState.time);
    this.refNowMs = nowMs;

    if (!this.command) this.createResources(frameState.context);
    this.advance(nowMs);
    this.t = this.computeT(nowMs);

    if (!this.show || this.filled === 0) return;
    if (!frameState.passes.render || frameState.mode !== SceneMode.SCENE3D) return;

    let fade = 1;
    const wall = performance.now();
    if (this.fadeStart < 0) this.fadeStart = wall;
    const age = wall - this.fadeStart;
    if (age < FADE_IN_MS) {
      const f = age / FADE_IN_MS;
      fade = f * f * (3 - 2 * f);
    }
    this.uState.z = this.brightness * fade;

    frameState.commandList.push(this.command as DrawCommand);
    this.updateLinks(frameState, wall, fade);
  }

  /** Uploads a changed topology, advances the link fade and queues the two link commands. */
  private updateLinks(frameState: FrameState, wall: number, pointsFade: number): void {
    const dtMs = this.linkLastWall < 0 ? 0 : Math.min(wall - this.linkLastWall, 250);
    this.linkLastWall = wall;
    const total = this.linkInIdx + this.linkCrossIdx;
    const target = this.linksVisible && total > 0 ? 1 : 0;
    if (this.linkSnap && total > 0) {
      this.linkSnap = false;
      this.linkIntensity = target;
    }
    if (this.linkIntensity < target) this.linkIntensity = Math.min(target, this.linkIntensity + dtMs / LINK_FADE_MS);
    else if (this.linkIntensity > target) this.linkIntensity = Math.max(target, this.linkIntensity - dtMs / LINK_FADE_MS);
    if (total === 0) return;

    if (this.linksDirty) {
      this.linksDirty = false;
      this.uploadLinks(frameState.context, total);
    }
    if (this.linkIntensity <= 0 || !this.linkInCommand || !this.linkCrossCommand) return;
    const f = this.linkIntensity;
    this.uLinkFade.w = f * f * (3 - 2 * f) * pointsFade;
    if (this.linkInIdx > 0) frameState.commandList.push(this.linkInCommand);
    if (this.linkCrossIdx > 0) frameState.commandList.push(this.linkCrossCommand);
  }

  private uploadLinks(context: Context, total: number): void {
    const staging = this.linkStaging;
    if (!staging || !this.vertexArray) return;
    if (!this.linkIndexBuffer || this.linkIndexCapacity < total) this.createLinkResources(context, staging.length);
    const ib = this.linkIndexBuffer;
    if (!ib) return;
    ib.copyFromArrayView(staging.subarray(0, total), 0);
    const inCmd = this.linkInCommand as RangeCommand;
    const crossCmd = this.linkCrossCommand as RangeCommand;
    inCmd.offset = 0;
    inCmd.count = this.linkInIdx;
    crossCmd.offset = this.linkInIdx;
    crossCmd.count = this.linkCrossIdx;
  }

  /** (Re)creates the link index buffer (capacity in indices), its VertexArray and the two commands. */
  private createLinkResources(context: Context, capacity: number): void {
    this.destroyLinkVertexArray();
    const [posA, posB, colorBuffer] = this.sharedBuffers;
    if (!posA || !posB || !colorBuffer) return;
    const wide = this.count > 65535;
    const indexBuffer = BufferWithIndex.createIndexBuffer({
      context,
      sizeInBytes: capacity * (wide ? 4 : 2),
      usage: BufferUsage.DYNAMIC_DRAW,
      indexDatatype: wide ? IndexDatatype.UNSIGNED_INT : IndexDatatype.UNSIGNED_SHORT,
    });
    this.linkIndexBuffer = indexBuffer;
    this.linkIndexCapacity = capacity;
    const L = LINK_ATTRIBUTE_LOCATIONS;
    // Shares the points' vertex buffers (vertexArrayDestroyable = false on
    // those); the index buffer is owned by (destroyed with) this VertexArray.
    this.linkVertexArray = new VertexArray({
      context,
      attributes: [
        { index: L.a_posHighA, vertexBuffer: posA, componentsPerAttribute: 3, componentDatatype: ComponentDatatype.FLOAT, offsetInBytes: 0, strideInBytes: 24 },
        { index: L.a_posLowA, vertexBuffer: posA, componentsPerAttribute: 3, componentDatatype: ComponentDatatype.FLOAT, offsetInBytes: 12, strideInBytes: 24 },
        { index: L.a_posHighB, vertexBuffer: posB, componentsPerAttribute: 3, componentDatatype: ComponentDatatype.FLOAT, offsetInBytes: 0, strideInBytes: 24 },
        { index: L.a_posLowB, vertexBuffer: posB, componentsPerAttribute: 3, componentDatatype: ComponentDatatype.FLOAT, offsetInBytes: 12, strideInBytes: 24 },
        { index: L.a_colorShell, vertexBuffer: colorBuffer, componentsPerAttribute: 4, componentDatatype: ComponentDatatype.UNSIGNED_BYTE, normalize: false },
      ],
      indexBuffer,
    });
    if (!this.linkProgram) {
      this.linkProgram = ShaderProgram.fromCache({
        context,
        vertexShaderSource: new ShaderSource({ sources: [LASER_LINKS_VS] }),
        fragmentShaderSource: new ShaderSource({ sources: [LASER_LINKS_FS] }),
        attributeLocations: { ...LINK_ATTRIBUTE_LOCATIONS },
      });
    }
    const makeCommand = (color: Cartesian4): RangeCommand =>
      new DrawCommand({
        boundingVolume: this.boundingSphere,
        primitiveType: PrimitiveType.LINES,
        vertexArray: this.linkVertexArray,
        count: 0,
        offset: 0,
        shaderProgram: this.linkProgram,
        uniformMap: {
          u_t: () => this.t,
          u_shellVis0: () => this.uShellVis0,
          u_shellVis1: () => this.uShellVis1,
          u_state: () => this.uState,
          u_linkColor: () => color,
          u_linkFade: () => this.uLinkFade,
        },
        renderState: this.renderState,
        pass: Pass.OPAQUE,
        owner: this,
        cull: true,
        occlude: false,
        castShadows: false,
        receiveShadows: false,
      }) as RangeCommand;
    this.linkInCommand = makeCommand(this.uLinkColorIn);
    this.linkCrossCommand = makeCommand(this.uLinkColorCross);
  }

  private destroyLinkVertexArray(): void {
    if (this.linkVertexArray && !this.linkVertexArray.isDestroyed()) this.linkVertexArray.destroy();
    else if (this.linkIndexBuffer && !this.linkIndexBuffer.isDestroyed()) this.linkIndexBuffer.destroy();
    this.linkVertexArray = undefined;
    this.linkIndexBuffer = undefined;
    this.linkIndexCapacity = 0;
    this.linkInCommand = undefined;
    this.linkCrossCommand = undefined;
  }

  private createResources(context: Context): void {
    const n = this.count;

    const colorShell = new Uint8Array(n * 4);
    const indices = new Float32Array(n);
    const shells = this.catalog.shell;
    for (let i = 0; i < n; i++) {
      let s = shells[i];
      if (!(s < SHELLS.length) || s >= MAX_SHELLS) s = SHELL_OTHER;
      const col = SHELLS[s].color;
      colorShell[i * 4] = Math.round(col[0] * 255);
      colorShell[i * 4 + 1] = Math.round(col[1] * 255);
      colorShell[i * 4 + 2] = Math.round(col[2] * 255);
      colorShell[i * 4 + 3] = s;
      indices[i] = i;
    }

    const posA = Buffer.createVertexBuffer({ context, sizeInBytes: n * 24, usage: BufferUsage.DYNAMIC_DRAW });
    const posB = Buffer.createVertexBuffer({ context, sizeInBytes: n * 24, usage: BufferUsage.DYNAMIC_DRAW });
    this.posBuffers = [posA, posB];
    const colorBuffer = Buffer.createVertexBuffer({ context, typedArray: colorShell, usage: BufferUsage.STATIC_DRAW });
    const indexBuffer = Buffer.createVertexBuffer({ context, typedArray: indices, usage: BufferUsage.STATIC_DRAW });
    // Shared with the link VertexArray: destroyed explicitly in destroy().
    this.sharedBuffers = [posA, posB, colorBuffer, indexBuffer];
    for (const buf of this.sharedBuffers) buf.vertexArrayDestroyable = false;

    const L = ATTRIBUTE_LOCATIONS;
    this.vertexArray = new VertexArray({
      context,
      attributes: [
        { index: L.a_posHighA, vertexBuffer: posA, componentsPerAttribute: 3, componentDatatype: ComponentDatatype.FLOAT, offsetInBytes: 0, strideInBytes: 24 },
        { index: L.a_posLowA, vertexBuffer: posA, componentsPerAttribute: 3, componentDatatype: ComponentDatatype.FLOAT, offsetInBytes: 12, strideInBytes: 24 },
        { index: L.a_posHighB, vertexBuffer: posB, componentsPerAttribute: 3, componentDatatype: ComponentDatatype.FLOAT, offsetInBytes: 0, strideInBytes: 24 },
        { index: L.a_posLowB, vertexBuffer: posB, componentsPerAttribute: 3, componentDatatype: ComponentDatatype.FLOAT, offsetInBytes: 12, strideInBytes: 24 },
        { index: L.a_colorShell, vertexBuffer: colorBuffer, componentsPerAttribute: 4, componentDatatype: ComponentDatatype.UNSIGNED_BYTE, normalize: false },
        { index: L.a_index, vertexBuffer: indexBuffer, componentsPerAttribute: 1, componentDatatype: ComponentDatatype.FLOAT },
      ],
    });

    this.shaderProgram = ShaderProgram.fromCache({
      context,
      vertexShaderSource: new ShaderSource({ sources: [SATELLITE_POINTS_VS_FADE] }),
      fragmentShaderSource: new ShaderSource({ sources: [SATELLITE_POINTS_FS] }),
      attributeLocations: { ...ATTRIBUTE_LOCATIONS },
    });

    this.createRenderState();

    const maxPoint = ContextLimits.maximumAliasedPointSize;
    this.uMaxPointSize = maxPoint > 1 ? maxPoint : 64;

    this.command = new DrawCommand({
      boundingVolume: this.boundingSphere,
      primitiveType: PrimitiveType.POINTS,
      vertexArray: this.vertexArray,
      count: n,
      shaderProgram: this.shaderProgram,
      uniformMap: this.uniformMap,
      renderState: this.renderState,
      pass: Pass.OPAQUE,
      owner: this,
      cull: true,
      occlude: false,
      castShadows: false,
      receiveShadows: false,
    });
  }

  private createRenderState(): void {
    const desc = {
      depthTest: { enabled: this.depthTestEnabled },
      depthMask: false,
      blending: {
        enabled: true,
        equationRgb: BlendEquation.ADD,
        equationAlpha: BlendEquation.ADD,
        functionSourceRgb: BlendFunction.ONE,
        functionSourceAlpha: BlendFunction.ONE,
        functionDestinationRgb: BlendFunction.ONE,
        functionDestinationAlpha: BlendFunction.ONE,
      },
    };
    this.renderStateDesc = desc;
    this.renderState = RenderState.fromCache(desc);
  }

  private releaseRenderState(): void {
    if (this.renderStateDesc) RenderState.removeFromCache(this.renderStateDesc);
    this.renderStateDesc = undefined;
    this.renderState = undefined;
  }

  isDestroyed(): boolean {
    return this.destroyed;
  }

  destroy(): undefined {
    if (this.destroyed) return undefined;
    this.destroyed = true;
    // The vertex buffers are shared by both VertexArrays and flagged
    // vertexArrayDestroyable = false, so they are destroyed explicitly.
    this.destroyLinkVertexArray();
    if (this.vertexArray && !this.vertexArray.isDestroyed()) this.vertexArray.destroy();
    this.vertexArray = undefined;
    for (const buf of this.sharedBuffers) if (!buf.isDestroyed()) buf.destroy();
    this.sharedBuffers = [];
    this.posBuffers = [undefined, undefined];
    if (this.shaderProgram && !this.shaderProgram.isDestroyed()) this.shaderProgram.destroy();
    this.shaderProgram = undefined;
    if (this.linkProgram && !this.linkProgram.isDestroyed()) this.linkProgram.destroy();
    this.linkProgram = undefined;
    this.linkStaging = null;
    this.linkInIdx = 0;
    this.linkCrossIdx = 0;
    this.releaseRenderState();
    this.command = undefined;
    this.resetTimeline();
    return undefined;
  }
}
