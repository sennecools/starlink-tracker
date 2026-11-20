/**
 * OrbitPath — the selected satellite's orbit ring and ground track.
 *
 * Ring
 *   One closed orbit sampled in TEME (see satDetails.orbitEci), drawn as a
 *   ~2 px screen-space polyline (Primitive + PolylineGeometry +
 *   PolylineMaterialAppearance with a custom fabric). Its modelMatrix is a pure
 *   rotation about Z, rewritten in place every frame in scene.preUpdate:
 *
 *       theta = nodeRate * (now - phase0) - GMST(now)        (TEME -> Earth-fixed)
 *
 *   so the ring stays fixed in inertial space while the globe turns under it
 *   and always passes through the moving satellite. GMST comes from
 *   satellite.js' `gstime`, the exact rotation the propagation worker uses for
 *   the GPU points (Cesium's computeTemeToPseudoFixedMatrix uses a slightly
 *   different GMST polynomial). A Primitive (rather than a PolylineCollection)
 *   is used on purpose: PolylineCollection bakes its modelMatrix into the
 *   vertex buffers and rebuilds them whenever it changes; a Primitive only
 *   transforms its bounding sphere. Per frame: one Z rotation written into the
 *   existing Matrix4 and one float uniform -> no allocations.
 *
 *   Brightness is a gradient by orbital phase: the fabric gets
 *   phase = fract((now - phase0) / period) and, with materialInput.s running
 *   0..1 along the samples (PolylineGeometry assigns s = index / (n - 1), i.e.
 *   uniform in time), d = fract(s - phase) is "how far ahead of the
 *   satellite"; alpha is highest just ahead (d -> 0) and fades to a faint tail
 *   just behind it (d -> 1).
 *
 * Ground track
 *   ECEF sub-satellite points lifted 15 km above the ellipsoid, thin dashed
 *   line that fades out towards the end of the track (custom fabric, dash in
 *   screen space like Cesium's PolylineDash).
 *
 * Depth: both are translucent primitives with the default depth test, so the
 * globe / ellipsoid depth plane hides them behind the Earth like the points.
 *
 * Double buffering: every setOrbit / setGroundTrack builds a NEW primitive
 * (synchronously, hidden); the old one keeps drawing until the new one is
 * ready (one frame later), then they swap in preUpdate. No flicker, no empty
 * frame, and refreshing every ~30 s (elements are perturbed) is cheap.
 */
import {
  ArcType,
  Cartesian3,
  Color,
  GeometryInstance,
  Material,
  Matrix3,
  Matrix4,
  PolylineGeometry,
  PolylineMaterialAppearance,
  Primitive,
  PrimitiveCollection,
} from 'cesium';
import type { JulianDate, Scene } from 'cesium';
import { gstime } from 'satellite.js';
import { julianToUnixMs } from './SatellitePoints.ts';

/** Refresh the ring/ground track this often (sim ms): elements drift. */
export const ORBIT_REFRESH_MS = 30_000;
/** Ground track height above the ellipsoid, metres. */
const GROUND_TRACK_LIFT_M = 15_000;
const RING_WIDTH_PX = 2.5;
const GROUND_TRACK_WIDTH_PX = 1.2;
/** Ground track opacity (subtle: the coverage footprint is the main ground cue now). */
const GROUND_TRACK_ALPHA = 0.22;
const JD_UNIX_EPOCH = 2440587.5;
const MS_PER_DAY = 86_400_000;

/**
 * True when an orbit computed at `orbitTimeMs` should be recomputed at
 * `nowMs` (older than `intervalMs`, from the future, or never computed).
 */
export function orbitNeedsRefresh(orbitTimeMs: number, nowMs: number, intervalMs = ORBIT_REFRESH_MS): boolean {
  if (!Number.isFinite(orbitTimeMs)) return true;
  const age = nowMs - orbitTimeMs;
  return !(age >= 0 && age < intervalMs);
}

/** Ring head colour for a shell: the sRGB shell colour mixed towards white, alpha ~0.9. */
export function orbitColorForShell(rgb: readonly [number, number, number], result = new Color()): Color {
  const w = 0.45;
  result.red = rgb[0] + (1 - rgb[0]) * w;
  result.green = rgb[1] + (1 - rgb[1]) * w;
  result.blue = rgb[2] + (1 - rgb[2]) * w;
  result.alpha = 0.9;
  return result;
}

const RING_FABRIC_TYPE = 'StarlinkOrbitRing';
const RING_SOURCE = /* glsl */ `
uniform vec4 color;
uniform float phase;
uniform float tailAlpha;
uniform float gap;

czm_material czm_getMaterial(czm_materialInput materialInput)
{
    czm_material material = czm_getDefaultMaterial(materialInput);
    // 0 just ahead of the satellite ... 1 just behind it.
    float d = fract(materialInput.s - phase);
    float head = 1.0 - d;
    float a = mix(tailAlpha, 1.0, pow(head, 1.6));
    // Tiny gap around the satellite itself so the line does not cut its sprite.
    a *= smoothstep(0.0, gap, d) * smoothstep(0.0, gap * 0.5, head);
    // Soft cross-section: fade the outer edge of the line for a smooth look.
    float edge = 1.0 - abs(materialInput.st.t - 0.5) * 2.0;
    a *= smoothstep(0.0, 0.5, edge);
    vec4 c = czm_gammaCorrect(color);
    material.emission = c.rgb;
    material.alpha = c.a * a;
    return material;
}
`;

const TRACK_FABRIC_TYPE = 'StarlinkGroundTrack';
const TRACK_SOURCE = /* glsl */ `
uniform vec4 color;
uniform float dashLength;
uniform float endAlpha;
in float v_polylineAngle;

czm_material czm_getMaterial(czm_materialInput materialInput)
{
    czm_material material = czm_getDefaultMaterial(materialInput);
    float c = cos(v_polylineAngle);
    float s = sin(v_polylineAngle);
    vec2 pos = mat2(c, s, -s, c) * gl_FragCoord.xy;
    float dash = fract(pos.x / (dashLength * czm_pixelRatio));
    if (dash > 0.55) {
        discard;
    }
    float a = mix(1.0, endAlpha, materialInput.s);
    vec4 col = czm_gammaCorrect(color);
    material.emission = col.rgb;
    material.alpha = col.a * a;
    return material;
}
`;

interface Layer {
  primitive: Primitive;
  material: Material;
  /** Ring only: time of sample 0, period, node rate (rad/ms). */
  phase0: number;
  period: number;
  nodeRate: number;
}

const scratchColor = new Color();

const scratchRotation = new Matrix3();

/** Writes a rotation about Z by `theta` (radians, counter-clockwise) into `m` in place. */
function setRotationZ(m: Matrix4, theta: number): void {
  Matrix4.fromRotation(Matrix3.fromRotationZ(theta, scratchRotation), m);
}

function toPositions(xyz: Float64Array, liftM: number): Cartesian3[] {
  const out: Cartesian3[] = [];
  for (let i = 0; i + 2 < xyz.length; i += 3) {
    const x = xyz[i]!;
    const y = xyz[i + 1]!;
    const z = xyz[i + 2]!;
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
    if (liftM !== 0) {
      const r = Math.sqrt(x * x + y * y + z * z);
      const k = r > 0 ? (r + liftM) / r : 1;
      out.push(new Cartesian3(x * k, y * k, z * k));
    } else {
      out.push(new Cartesian3(x, y, z));
    }
  }
  return out;
}

export class OrbitPath {
  private readonly scene: Scene;
  private readonly collection: PrimitiveCollection;
  private ring: Layer | null = null;
  private ringPending: Layer | null = null;
  private track: Layer | null = null;
  private trackPending: Layer | null = null;
  private readonly trackColor = new Color(1, 1, 1, GROUND_TRACK_ALPHA);
  private readonly removePreUpdate: () => void;
  private destroyed = false;
  private visible = true;
  /** Shader warm-up layers (see constructor) and frames left to show them. */
  private warmRing: Layer | null = null;
  private warmTrack: Layer | null = null;
  private warmFrames = 0;

  constructor(scene: Scene) {
    this.scene = scene;
    this.collection = new PrimitiveCollection({ destroyPrimitives: true });
    scene.primitives.add(this.collection);
    this.removePreUpdate = scene.preUpdate.addEventListener(this.onPreUpdate, this);
    this.prewarm();
  }

  /**
   * Draws a fully transparent ring + track for a few frames at start-up so
   * both shader programs are compiled before the first real selection
   * (avoids a hitch on the first click).
   */
  private prewarm(): void {
    const n = 64;
    const circle = new Float64Array((n + 1) * 3);
    const r = 6_378_137 + 550_000;
    for (let k = 0; k <= n; k++) {
      const a = (k / n) * 2 * Math.PI;
      circle[k * 3] = r * Math.cos(a);
      circle[k * 3 + 1] = r * Math.sin(a);
    }
    const alpha = this.trackColor.alpha;
    this.trackColor.alpha = 0;
    this.setOrbit(circle, Color.TRANSPARENT, 0, 1);
    this.setGroundTrack(circle);
    this.trackColor.alpha = alpha;
    this.trackColor.red = this.trackColor.green = this.trackColor.blue = 1;
    this.warmRing = this.ringPending;
    this.warmTrack = this.trackPending;
    this.warmFrames = 3;
  }

  /** Time (ms) of sample 0 of the most recently set ring, NaN when none. */
  get orbitTimeMs(): number {
    const l = this.ringPending ?? this.ring;
    return l ? l.phase0 : Number.NaN;
  }

  /** Hide/show everything without dropping the geometry. */
  setVisible(visible: boolean): void {
    this.visible = visible;
    this.collection.show = visible;
  }

  /**
   * Sets (or clears, with null) the orbit ring.
   * @param eciMetres closed ring from satDetails.orbitEci (x,y,z metres, TEME at phase0)
   * @param color head colour (see orbitColorForShell); copied
   * @param phase0TimeMs time of sample 0 (ms since Unix epoch)
   * @param periodMs time spanned by the samples (satDetails.orbitPeriodMs)
   * @param nodeRateRadPerMs nodal regression rate (satDetails.nodeRateRadPerMs) when the ring
   *        was sampled node-fixed (orbitEci's default); 0 for a plain TEME ring
   */
  setOrbit(eciMetres: Float64Array | null, color: Color, phase0TimeMs: number, periodMs: number, nodeRateRadPerMs = 0): void {
    if (this.destroyed) return;
    this.dropLayer('ringPending');
    const positions = eciMetres ? toPositions(eciMetres, 0) : [];
    if (positions.length < 2 || !(periodMs > 0) || !Number.isFinite(phase0TimeMs)) {
      this.dropLayer('ring');
      return;
    }

    // Ground track takes a faint, whiter version of the ring colour.
    Color.lerp(color, Color.WHITE, 0.5, scratchColor);
    this.trackColor.red = scratchColor.red;
    this.trackColor.green = scratchColor.green;
    this.trackColor.blue = scratchColor.blue;

    const material = new Material({
      fabric: {
        type: RING_FABRIC_TYPE,
        uniforms: {
          color: Color.clone(color),
          phase: 0,
          tailAlpha: 0.1,
          gap: 0.0012,
        },
        source: RING_SOURCE,
      },
      translucent: true,
    });
    const nodeRate = Number.isFinite(nodeRateRadPerMs) ? nodeRateRadPerMs : 0;
    const layer: Layer = {
      primitive: this.buildPrimitive(positions, RING_WIDTH_PX, material),
      material,
      phase0: phase0TimeMs,
      period: periodMs,
      nodeRate,
    };
    // Oriented in every preUpdate (also while pending), shown once built.
    this.ringPending = layer;
  }

  /** Sets (or clears, with null) the ground track (ECEF metres on the ellipsoid; lifted 15 km). */
  setGroundTrack(ecefMetres: Float64Array | null): void {
    if (this.destroyed) return;
    this.dropLayer('trackPending');
    const positions = ecefMetres ? toPositions(ecefMetres, GROUND_TRACK_LIFT_M) : [];
    if (positions.length < 2) {
      this.dropLayer('track');
      return;
    }
    const material = new Material({
      fabric: {
        type: TRACK_FABRIC_TYPE,
        uniforms: {
          color: Color.clone(this.trackColor),
          dashLength: 14,
          endAlpha: 0.1,
        },
        source: TRACK_SOURCE,
      },
      translucent: true,
    });
    this.trackPending = {
      primitive: this.buildPrimitive(positions, GROUND_TRACK_WIDTH_PX, material),
      material,
      phase0: Number.NaN,
      period: 0,
      nodeRate: 0,
    };
  }

  /** Removes ring and ground track. */
  clear(): void {
    this.setOrbit(null, Color.WHITE, 0, 0);
    this.setGroundTrack(null);
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.removePreUpdate();
    this.ring = this.ringPending = this.track = this.trackPending = null;
    this.warmRing = this.warmTrack = null;
    // Destroys the collection and (destroyPrimitives) everything in it.
    this.scene.primitives.remove(this.collection);
  }

  isDestroyed(): boolean {
    return this.destroyed;
  }

  // -------------------------------------------------------------------------

  private buildPrimitive(positions: Cartesian3[], width: number, material: Material): Primitive {
    const primitive = new Primitive({
      geometryInstances: new GeometryInstance({
        geometry: new PolylineGeometry({
          positions,
          width,
          arcType: ArcType.NONE,
          vertexFormat: PolylineMaterialAppearance.VERTEX_FORMAT,
        }),
      }),
      appearance: new PolylineMaterialAppearance({ material, translucent: true }),
      asynchronous: false,
      allowPicking: false,
      show: false,
    });
    this.collection.add(primitive);
    return primitive;
  }

  private dropLayer(key: 'ring' | 'ringPending' | 'track' | 'trackPending'): void {
    const layer = this[key];
    if (!layer) return;
    this[key] = null;
    this.collection.remove(layer.primitive); // destroys it
  }

  private orientRing(layer: Layer, nowMs: number): void {
    if (!Number.isFinite(nowMs)) return;
    const gmst = gstime(JD_UNIX_EPOCH + nowMs / MS_PER_DAY);
    const dt = nowMs - layer.phase0;
    setRotationZ(layer.primitive.modelMatrix, layer.nodeRate * dt - gmst);
    let p = dt / layer.period;
    p -= Math.floor(p);
    layer.material.uniforms.phase = p;
  }

  private onPreUpdate(_scene: Scene, time: JulianDate): void {
    if (this.destroyed) return;

    // Swap in freshly built primitives once they are ready (built hidden last frame).
    const rp = this.ringPending;
    if (rp && rp.primitive.ready) {
      this.ringPending = null;
      this.dropLayer('ring');
      this.ring = rp;
      rp.primitive.show = true;
    }
    const tp = this.trackPending;
    if (tp && tp.primitive.ready) {
      this.trackPending = null;
      this.dropLayer('track');
      this.track = tp;
      tp.primitive.show = true;
    }

    if (this.warmFrames > 0 && !this.ringPending && !this.trackPending) {
      if (--this.warmFrames === 0) {
        if (this.ring && this.ring === this.warmRing) this.dropLayer('ring');
        if (this.track && this.track === this.warmTrack) this.dropLayer('track');
        this.warmRing = this.warmTrack = null;
      }
    }

    if (!this.visible) return;
    const ring = this.ring;
    const pending = this.ringPending;
    if (!ring && !pending) return;
    const nowMs = julianToUnixMs(time);
    if (ring) this.orientRing(ring, nowMs);
    if (pending) this.orientRing(pending, nowMs);
  }
}
