/**
 * LaserHighlight — bright beams from the SELECTED satellite to its (modelled)
 * laser-link neighbours, with an animated "data pulse" running from the
 * selected satellite towards each neighbour.
 *
 * - One PolylineCollection with a pool of MAX_BEAMS (4) polylines, created
 *   once; each has its own Material instance of the custom fabric below
 *   (colour per link kind, pulse density per beam length, shared time).
 * - Endpoints are recomputed every frame in scene.preUpdate from
 *   SatellitePoints.getPosition(i, out, nowMs) — the same interpolation and
 *   sim time the GPU points (and the camera rig) use — written into reused
 *   Cartesian3s, and the SAME positions array is re-assigned to the polyline
 *   (Polyline's setter needs the assignment to mark the buffer dirty; with
 *   distinct endpoints it neither copies nor allocates).
 * - Polylines are depth tested, so beams are hidden behind the Earth.
 *
 * The link topology is a MODEL (SpaceX does not publish it); the UI states so.
 */
import { Cartesian3, Color, Material, PolylineCollection } from 'cesium';
import type { JulianDate, Polyline, Scene } from 'cesium';
import { julianToUnixMs } from './SatellitePoints.ts';
import type { SatellitePoints } from './SatellitePoints.ts';

/** At most this many beams (V1.5: 4, V2 Mini: 3, V3: 6 laser terminals). */
export const MAX_BEAMS = 6;
/** Beam width in CSS pixels. */
const BEAM_WIDTH_PX = 2;
/** Roughly one travelling pulse per this many metres of beam (1..6 per beam). */
const PULSE_SPACING_M = 700_000;

/**
 * In-plane / cross-plane beam colours: the link web's blue
 * (SatellitePoints LASER_COLOR_RGB / LASER_CROSS_COLOR_RGB), lifted towards
 * a brighter ice blue so the selected satellite's beams stand out.
 */
export const BEAM_COLOR_CSS = 'rgb(140, 205, 255)';
const IN_PLANE_COLOR = new Color(0.55, 0.8, 1.0, 0.95);
const CROSS_PLANE_COLOR = new Color(0.42, 0.68, 1.0, 0.95);

const LASER_BEAM_TYPE = 'StarlinkLaserBeam';

/**
 * Fabric source. st.s runs 0 (selected satellite) -> 1 (neighbour) along the
 * beam, st.t 0..1 across its width. A thin steady core plus comet-shaped
 * pulses (bright head, fading tail) moving towards the neighbour.
 */
const LASER_BEAM_SOURCE = /* glsl */ `
czm_material czm_getMaterial(czm_materialInput materialInput)
{
    czm_material material = czm_getDefaultMaterial(materialInput);
    vec2 st = materialInput.st;
    float across = abs(st.t - 0.5) * 2.0;
    float core = 1.0 - smoothstep(0.25, 1.0, across);
    float phase = fract(st.s * pulses - time * speed);
    float tail = pow(phase, 7.0);
    float head = 1.0 - smoothstep(0.965, 1.0, phase);
    float pulse = tail * head;
    // Soften both ends so the beam does not start inside the sprite as a hard bar.
    float ends = smoothstep(0.0, 0.015, st.s) * (1.0 - smoothstep(0.985, 1.0, st.s));
    material.diffuse = color.rgb;
    // Blue beam, the pulse head burns to a cyan-white core.
    material.emission = color.rgb * (0.35 + 1.6 * pulse) + vec3(0.72, 0.93, 1.0) * (0.9 * pulse * pulse);
    material.alpha = clamp(color.a * core * ends * (0.38 + 0.9 * pulse), 0.0, 1.0);
    return material;
}
`;

interface Beam {
  polyline: Polyline;
  material: Material;
  positions: [Cartesian3, Cartesian3];
  target: number;
  kind: number;
}

export class LaserHighlight {
  private readonly scene: Scene;
  private readonly points: () => SatellitePoints | null;
  private readonly collection: PolylineCollection;
  private readonly beams: Beam[] = [];
  private source = -1;
  private active = 0;
  private boundPoints: SatellitePoints | null = null;
  private readonly srcPos = new Cartesian3();
  private readonly removePreUpdate: () => void;
  private destroyed = false;

  constructor(scene: Scene, points: () => SatellitePoints | null) {
    this.scene = scene;
    this.points = points;
    this.collection = scene.primitives.add(new PolylineCollection()) as PolylineCollection;
    for (let i = 0; i < MAX_BEAMS; i++) {
      const material = new Material({
        translucent: true,
        fabric: {
          type: LASER_BEAM_TYPE,
          uniforms: {
            color: Color.clone(IN_PLANE_COLOR),
            time: 0,
            speed: 0.55,
            pulses: 1,
          },
          source: LASER_BEAM_SOURCE,
        },
      });
      const positions: [Cartesian3, Cartesian3] = [new Cartesian3(0, 0, 7e6), new Cartesian3(0, 1, 7e6)];
      const polyline = this.collection.add({
        show: false,
        width: BEAM_WIDTH_PX,
        positions,
        material,
      });
      this.beams.push({ polyline, material, positions, target: -1, kind: 0 });
    }
    this.removePreUpdate = scene.preUpdate.addEventListener(this.onPreUpdate, this);
  }

  /**
   * Shows beams from satellite `index` to up to MAX_BEAMS `neighbours`
   * (catalogue indices) with link `kinds` (0 = in-plane, 1 = cross-plane).
   */
  setSource(index: number, neighbours: readonly number[], kinds: readonly number[]): void {
    if (this.destroyed) return;
    const pts = this.points();
    const count = pts && !pts.isDestroyed() ? pts.count : 0;
    if (!(Number.isInteger(index) && index >= 0 && index < count)) {
      this.clear();
      return;
    }
    this.source = index;
    this.boundPoints = pts;
    let k = 0;
    for (let j = 0; j < neighbours.length && k < MAX_BEAMS; j++) {
      const n = neighbours[j];
      if (!(Number.isInteger(n) && n >= 0 && n < count) || n === index) continue;
      const beam = this.beams[k++];
      beam.target = n;
      beam.kind = kinds[j] === 1 ? 1 : 0;
      Color.clone(beam.kind === 1 ? CROSS_PLANE_COLOR : IN_PLANE_COLOR, beam.material.uniforms.color as Color);
    }
    for (let j = k; j < MAX_BEAMS; j++) {
      this.beams[j].target = -1;
      this.beams[j].polyline.show = false;
    }
    this.active = k;
  }

  clear(): void {
    this.source = -1;
    this.active = 0;
    this.boundPoints = null;
    for (const b of this.beams) {
      b.target = -1;
      b.polyline.show = false;
    }
  }

  private hideAll(): void {
    for (let j = 0; j < this.active; j++) this.beams[j].polyline.show = false;
  }

  private onPreUpdate(_scene: Scene, time: JulianDate): void {
    if (this.destroyed || this.source < 0) return;
    const pts = this.points();
    if (!pts || pts.isDestroyed() || pts !== this.boundPoints) {
      // Catalogue swapped: indices no longer mean the same satellites.
      this.clear();
      return;
    }
    if (!pts.show) {
      this.hideAll();
      return;
    }
    const nowMs = julianToUnixMs(time);
    const src = pts.getPosition(this.source, this.srcPos, nowMs);
    if (!src) {
      this.hideAll();
      return;
    }
    const t = (performance.now() / 1000) % 3600;
    for (let j = 0; j < this.active; j++) {
      const beam = this.beams[j];
      const pos = beam.positions;
      if (!pts.getPosition(beam.target, pos[1], nowMs)) {
        beam.polyline.show = false;
        continue;
      }
      pos[0].x = src.x;
      pos[0].y = src.y;
      pos[0].z = src.z;
      const dx = pos[1].x - src.x;
      const dy = pos[1].y - src.y;
      const dz = pos[1].z - src.z;
      const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (!(len > 1)) {
        beam.polyline.show = false;
        continue;
      }
      beam.polyline.positions = pos;
      const u = beam.material.uniforms as { time: number; pulses: number };
      u.time = t;
      u.pulses = Math.min(6, Math.max(1, Math.round(len / PULSE_SPACING_M)));
      beam.polyline.show = true;
    }
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.removePreUpdate();
    if (!this.scene.isDestroyed() && this.scene.primitives.contains(this.collection)) {
      this.scene.primitives.remove(this.collection); // destroys it (destroyPrimitives = true)
    } else if (!this.collection.isDestroyed()) {
      this.collection.destroy();
    }
    this.beams.length = 0;
  }
}
