/**
 * SatelliteModels — close-up 3D models of the satellites near the camera.
 *
 * Pipeline (scene.preUpdate, before any primitive updates):
 *  1. Candidate search, at most every SEARCH_INTERVAL_MS: SatellitePoints.
 *     findNearest() scans the interpolated positions (per-axis early reject,
 *     no allocation, skipped entirely when the camera is far outside the
 *     constellation) for satellites within HIDE_RANGE_M of the camera; a
 *     satellite gets a model when within SHOW_RANGE_M (or already has one:
 *     hysteresis), nearest first, max `maxModels`, the forced (selected /
 *     tracked) one first.
 *  2. Every frame, every assigned model gets its satellite's exact position
 *     (SatellitePoints.getPosition at the frame's sim time, in doubles — the
 *     same values the GPU points and the camera rig use, so no jitter even at
 *     20-50 m) and attitude (satelliteModelMath.attitudeMatrix): model +X
 *     along the inertial velocity (made perpendicular to the radial), +Z
 *     zenith, -Z nadir. The model matrix is written in place (Model compares
 *     it every update, no setter needed).
 *  3. At <= 10 Hz: sun direction (Simon 1994 + ICRF->fixed, exactly what
 *     Cesium's own lighting uses), solar wings rotated to face the sun about
 *     their span axis, and eclipse dimming (cylindrical shadow: lightColor +
 *     image-based-lighting factor scaled down in the Earth's shadow).
 *
 * Frames / model convention (see satelliteModelMath.ts for the derivation):
 * the shipped .glb files (public/models/README.md) are authored X-forward:
 * glTF +X = flight direction, +Y = zenith (phased arrays / dishes on -Y face
 * the Earth), solar wings spanning ±Z. They are loaded with
 * `forwardAxis: Axis.X`, so Cesium maps glTF -> model frame as +X -> +X,
 * +Y -> +Z, +Z -> -Y. Final matrix columns: [x̂ = flight dir, ŷ = ẑ × x̂,
 * ẑ = zenith, position]. The wing pivots "ArrayPivot_L" / "ArrayPivot_R"
 * are children of the unrotated body, rotate about glTF Z and their cell
 * face points to glTF +Y at rest (WING_AXIS / WING_REST_NORMAL below), so
 * θ = atan2(-s.x, s.y) with s = sun in glTF body axes.
 * "LaserTerminal_0..n" nodes: the first catalog.lasers[sat] of them are shown,
 * the rest hidden (v1.0 buses reuse the v1.5 model and show none; the
 * Transporter-1 test sats show 2 of its 4; V3 shows its 6).
 *
 * Every bus in BUS_VERSIONS is supported generically (V1.0 / V1.5 / V2 Mini /
 * V3 today): one warm-up instance per distinct file. A file that fails to load
 * (e.g. starlink-v3.glb not built yet) is retried every FAILED_RETRY_MS while
 * a satellite of that bus is close, so a model generated while the app runs
 * appears without a reload.
 *
 * Models are pooled per file (BUS_VERSIONS[bus].model), one Model instance
 * per visible satellite (Cesium's ResourceCache shares the glTF, geometry and
 * textures of instances of the same URL). One instance per file is created at
 * construction and "warmed up": drawn for a few frames at 1e-4 scale right in
 * front of the camera (sub-pixel), so its shaders are compiled and textures
 * uploaded before the first real use. Missing files (404) are tolerated: the
 * points then simply stay visible.
 *
 * The points fade out near the camera (SatellitePoints.setModelFade) only
 * while every satellite within POINT_FADE_FAR_M actually has a ready model.
 */
import {
  Axis,
  Cartesian2,
  Cartesian3,
  Matrix3,
  Matrix4,
  Model,
  PrimitiveCollection,
  ShadowMode,
  Simon1994PlanetaryPositions,
  Transforms,
} from 'cesium';
import type { JulianDate, ModelNode, Scene } from 'cesium';
import type { SatCatalog } from '../shared/contracts.ts';
import { BUS_V2MINI, BUS_VERSIONS } from '../shared/hardware.ts';
import { julianToUnixMs } from './SatellitePoints.ts';
import type { SatellitePoints } from './SatellitePoints.ts';
import {
  attitudeMatrix,
  pickEvictableSlot,
  scaledTranslation,
  selectCandidates,
  sunInGltfBodyXForward,
  sunlitFactor,
  wingNodeMatrix,
  wingSunAngle,
} from './satelliteModelMath.ts';
import type { Vec3 } from './satelliteModelMath.ts';

/** A model is shown for satellites within this distance of the camera. */
export const SHOW_RANGE_M = 40_000;
/** ...and kept until it is farther than this (hysteresis). */
export const HIDE_RANGE_M = 46_000;
/** Candidate search period (positions themselves are updated every frame). */
const SEARCH_INTERVAL_MS = 100;
/** Sun / wing / eclipse update period (<= 10 Hz). */
const SUN_INTERVAL_MS = 100;
/** Points are invisible closer than this to the camera when models cover them... */
export const POINT_FADE_NEAR_M = 1_200;
/** ...and fully visible again from here (a ~30 m satellite is ~9 px at 3.5 km). */
export const POINT_FADE_FAR_M = 3_500;
/** Frames a freshly loaded model is drawn (invisibly) to compile its shaders. */
const WARMUP_FRAMES = 4;
const WARMUP_SCALE = 1e-4;
const WARMUP_DISTANCE_M = 200;
/** Direct light / IBL remaining in the umbra. */
const ECLIPSE_LIGHT = 0.035;
const ECLIPSE_IBL = 0.12;
/** The dynamic environment map is regenerated after this much motion (default 1 km = every 0.13 s in LEO). */
const ENV_MAP_EPSILON_M = 300_000;
/** Search buffer size (nearest N within HIDE_RANGE_M). */
const FOUND_CAP = 48;
/** A model file that failed to load is retried after this long (wall ms), on demand. */
const FAILED_RETRY_MS = 30_000;
/** Upper bound of LaserTerminal_<n> nodes looked up per model. */
const MAX_TERMINAL_NODES = 12;

/** Wing rotation axis and rest normal of the active cell face, glTF body frame. */
const WING_AXIS: Vec3 = [0, 0, 1];
const WING_REST_NORMAL: Vec3 = [0, 1, 0];
const WING_NODE_NAMES = ['ArrayPivot_L', 'ArrayPivot_R'] as const;

export interface SatelliteModelsOptions {
  scene: Scene;
  points: () => SatellitePoints | null;
  catalog: () => SatCatalog | null;
  /** Base URL of the .glb files (default '/models/'). */
  baseUrl?: string;
  /** Maximum number of simultaneously shown models (default 12). */
  maxModels?: number;
}

interface Wing {
  node: ModelNode;
  orig: Matrix4;
}

interface Slot {
  file: string;
  model: Model | null;
  /** Catalogue index shown by this slot, -1 = free. */
  sat: number;
  /** Warm-up frames left (counted once ready). */
  warm: number;
  wings: Wing[];
  /** LaserTerminal_0..n nodes, in index order. */
  terminals: ModelNode[];
  /** Number of leading `terminals` currently shown (-1 = unknown). */
  terminalsShown: number;
  wingAngle: number;
  /** Sunlit factor last applied (NaN = never). */
  lit: number;
  readonly lightColor: Cartesian3;
}

const scratchPos = new Cartesian3();
const scratchVel = new Cartesian3();
const scratchCam = new Cartesian3();
const scratchNodeMatrix = new Matrix4();
const scratchIbl = new Cartesian2();
const scratchSunBody = new Float64Array(3);

export class SatelliteModels {
  private readonly scene: Scene;
  private readonly getPoints: () => SatellitePoints | null;
  private readonly getCatalog: () => SatCatalog | null;
  private readonly baseUrl: string;
  private readonly maxModels: number;
  private readonly collection: PrimitiveCollection;
  private readonly slots: Slot[] = [];
  private readonly loadingFiles = new Set<string>();
  private readonly failedFiles = new Set<string>();
  /** Wall time (performance.now) of the last failure per file. */
  private readonly failedAt = new Map<string, number>();
  private readonly warnedFiles = new Set<string>();
  private readonly removePreUpdate: () => void;
  private destroyed = false;

  private boundPoints: SatellitePoints | null = null;
  private shownMark = new Uint8Array(0);
  private forced = -1;
  private searchDue = true;
  private lastSearch = -Infinity;
  private lastSun = -Infinity;
  private fadeOn = false;

  private readonly foundIdx = new Int32Array(FOUND_CAP);
  private readonly foundD2 = new Float64Array(FOUND_CAP);
  private readonly chosen: Int32Array;
  private readonly icrfToFixed = new Matrix3();
  private readonly sunDir = new Cartesian3(1, 0, 0);
  private sunValid = false;

  constructor(opts: SatelliteModelsOptions) {
    this.scene = opts.scene;
    this.getPoints = opts.points;
    this.getCatalog = opts.catalog;
    const base = opts.baseUrl ?? '/models/';
    this.baseUrl = base.endsWith('/') ? base : `${base}/`;
    this.maxModels = Math.max(1, Math.floor(opts.maxModels ?? 12));
    this.chosen = new Int32Array(this.maxModels);
    this.collection = this.scene.primitives.add(new PrimitiveCollection({ destroyPrimitives: true })) as PrimitiveCollection;

    // Preload (and warm up) one instance of every distinct model file.
    const files = new Set(BUS_VERSIONS.map((b) => b.model));
    for (const file of files) this.createSlot(file);

    this.removePreUpdate = this.scene.preUpdate.addEventListener(this.onPreUpdate, this);
  }

  /** Always give this satellite a model (selected / tracked) while within range; -1 = none. */
  setForced(index: number): void {
    const i = Number.isInteger(index) && index >= 0 ? index : -1;
    if (i === this.forced) return;
    this.forced = i;
    this.searchDue = true;
  }

  /** Console / HUD diagnostics. */
  debugInfo(): { slots: number; shown: number[]; loading: string[]; failed: string[]; fade: boolean } {
    return {
      slots: this.slots.length,
      shown: this.slots.filter((s) => s.sat >= 0).map((s) => s.sat),
      loading: [...this.loadingFiles],
      failed: [...this.failedFiles],
      fade: this.fadeOn,
    };
  }

  // -------------------------------------------------------------------------
  // Pool
  // -------------------------------------------------------------------------

  private createSlot(file: string): void {
    if (this.destroyed || this.failedFiles.has(file) || this.loadingFiles.has(file)) return;
    if (this.slots.length >= this.maxModels + 1) return;
    const slot: Slot = {
      file,
      model: null,
      sat: -1,
      warm: WARMUP_FRAMES,
      wings: [],
      terminals: [],
      terminalsShown: -1,
      wingAngle: Number.NaN,
      lit: Number.NaN,
      lightColor: new Cartesian3(2, 2, 2),
    };
    this.slots.push(slot);
    this.loadingFiles.add(file);
    const matrix = new Matrix4();
    this.warmMatrix(matrix);
    Model.fromGltfAsync({
      url: this.baseUrl + file,
      // The .glb files are authored glTF +X = flight direction (see header).
      forwardAxis: Axis.X,
      show: true, // warm-up: drawn sub-pixel in front of the camera until ready + WARMUP_FRAMES
      modelMatrix: matrix,
      shadows: ShadowMode.DISABLED,
      allowPicking: false,
      incrementallyLoadTextures: false,
      // Defined from the start: switching lightColor between undefined and
      // defined would rebuild the draw commands.
      lightColor: slot.lightColor,
      environmentMapOptions: { maximumPositionEpsilon: ENV_MAP_EPSILON_M, maximumSecondsDifference: 600 },
    })
      .then((model) => {
        if (this.destroyed) {
          model.destroy();
          return;
        }
        slot.model = model;
        this.collection.add(model);
        model.readyEvent.addEventListener(() => this.onModelReady(slot));
        model.errorEvent.addEventListener((err: unknown) => this.onModelFailed(slot, err));
      })
      .catch((err: unknown) => this.onModelFailed(slot, err));
  }

  private onModelReady(slot: Slot): void {
    this.loadingFiles.delete(slot.file);
    const model = slot.model;
    if (!model || this.destroyed) return;
    for (const name of WING_NODE_NAMES) {
      const node = model.getNode(name);
      if (node) slot.wings.push({ node, orig: Matrix4.clone(node.originalMatrix) });
    }
    for (let t = 0; t < MAX_TERMINAL_NODES; t++) {
      const node = model.getNode(`LaserTerminal_${t}`);
      if (!node) break;
      slot.terminals.push(node);
    }
    slot.terminalsShown = -1;
    this.failedAt.delete(slot.file);
  }

  private onModelFailed(slot: Slot, err: unknown): void {
    this.loadingFiles.delete(slot.file);
    this.failedFiles.add(slot.file);
    this.failedAt.set(slot.file, performance.now());
    if (!this.warnedFiles.has(slot.file)) {
      this.warnedFiles.add(slot.file);
      console.warn(
        `[SatelliteModels] could not load ${this.baseUrl}${slot.file}; close-ups fall back to points (retried every ${FAILED_RETRY_MS / 1000} s)`,
        err,
      );
    }
    const i = this.slots.indexOf(slot);
    if (i >= 0) this.slots.splice(i, 1);
    if (slot.sat >= 0 && slot.sat < this.shownMark.length) this.shownMark[slot.sat] = 0;
    slot.sat = -1;
    const model = slot.model;
    slot.model = null;
    if (model && !this.destroyed && this.collection.contains(model)) this.collection.remove(model);
    this.searchDue = true;
  }

  /** Destroys one idle, ready instance of a file other than `keepFile` (never a file's last one). */
  private evictIdle(keepFile: string): boolean {
    const k = pickEvictableSlot(this.slots, keepFile);
    if (k < 0) return false;
    const slot = this.slots[k];
    this.slots.splice(k, 1);
    const model = slot.model;
    slot.model = null;
    // sat < 0: no shownMark entry to clear. destroyPrimitives: remove() destroys it.
    if (model && this.collection.contains(model)) this.collection.remove(model);
    return true;
  }

  private releaseSlot(slot: Slot): void {
    if (slot.sat >= 0 && slot.sat < this.shownMark.length) this.shownMark[slot.sat] = 0;
    slot.sat = -1;
    if (slot.model && slot.warm === 0) slot.model.show = false;
  }

  private releaseAll(): void {
    for (const slot of this.slots) this.releaseSlot(slot);
  }

  private warmMatrix(out: Matrix4): void {
    const cam = this.scene.camera;
    const p = cam.positionWC;
    const d = cam.directionWC;
    scaledTranslation(
      p.x + d.x * WARMUP_DISTANCE_M,
      p.y + d.y * WARMUP_DISTANCE_M,
      p.z + d.z * WARMUP_DISTANCE_M,
      WARMUP_SCALE,
      out,
    );
  }

  // -------------------------------------------------------------------------
  // Per frame
  // -------------------------------------------------------------------------

  private onPreUpdate(_scene: Scene, time: JulianDate): void {
    if (this.destroyed) return;
    // Warm-ups run regardless of the points state.
    for (let k = 0; k < this.slots.length; k++) {
      const slot = this.slots[k];
      const model = slot.model;
      if (slot.warm === 0 || !model) continue;
      if (model.ready) slot.warm--;
      if (slot.warm > 0) this.warmMatrix(model.modelMatrix);
      else model.show = false;
    }

    let pts = this.getPoints();
    if (pts && pts.isDestroyed()) pts = null;
    if (pts !== this.boundPoints) {
      this.releaseAll();
      const old = this.boundPoints;
      if (old && !old.isDestroyed() && this.fadeOn) old.setModelFade(0, 0);
      this.fadeOn = false;
      this.boundPoints = pts;
      this.shownMark = new Uint8Array(pts ? pts.count : 0); // only on catalogue swaps
      this.searchDue = true;
    }
    const catalog = this.getCatalog();
    if (!pts || !pts.show || !catalog || catalog.count !== pts.count) {
      this.releaseAll();
      this.setFade(pts, false);
      return;
    }

    const nowMs = julianToUnixMs(time);
    const wall = performance.now();
    const cam = Cartesian3.clone(this.scene.camera.positionWC, scratchCam);

    if (this.searchDue || wall - this.lastSearch >= SEARCH_INTERVAL_MS) {
      this.searchDue = false;
      this.lastSearch = wall;
      this.search(pts, catalog, cam, nowMs);
    }

    let sunTick = false;
    if (wall - this.lastSun >= SUN_INTERVAL_MS) {
      this.lastSun = wall;
      this.updateSun(time);
      sunTick = true;
    }

    for (let k = 0; k < this.slots.length; k++) {
      const slot = this.slots[k];
      const model = slot.model;
      if (slot.sat < 0 || !model) continue;
      const p = pts.getPosition(slot.sat, scratchPos, nowMs);
      if (!p) {
        model.show = false;
        continue;
      }
      const v = pts.getVelocity(slot.sat, scratchVel);
      const m = model.modelMatrix;
      if (v) attitudeMatrix(p.x, p.y, p.z, v.x, v.y, v.z, m);
      else attitudeMatrix(p.x, p.y, p.z, 0, 0, 0, m);
      model.show = true;
      if (sunTick || Number.isNaN(slot.lit)) this.updateSunDependent(slot, model, m);
    }
  }

  private search(pts: SatellitePoints, catalog: SatCatalog, cam: Cartesian3, nowMs: number): void {
    const n = pts.findNearest(cam, HIDE_RANGE_M, this.foundIdx, this.foundD2, nowMs);
    const shown = this.shownMark;

    // Forced satellite (may fall outside the nearest-N list).
    let forced = -1;
    if (this.forced >= 0 && this.forced < pts.count && pts.getPosition(this.forced, scratchPos, nowMs)) {
      const dx = scratchPos.x - cam.x;
      const dy = scratchPos.y - cam.y;
      const dz = scratchPos.z - cam.z;
      const d2 = dx * dx + dy * dy + dz * dz;
      const r = shown[this.forced] === 1 ? HIDE_RANGE_M : SHOW_RANGE_M;
      if (d2 <= r * r) forced = this.forced;
    }

    const count = selectCandidates(
      this.foundIdx,
      this.foundD2,
      n,
      shown,
      forced,
      SHOW_RANGE_M * SHOW_RANGE_M,
      this.maxModels,
      this.chosen,
    );

    // Release slots whose satellite is no longer chosen.
    for (let k = 0; k < this.slots.length; k++) {
      const slot = this.slots[k];
      if (slot.sat < 0) continue;
      let keep = false;
      for (let c = 0; c < count; c++) {
        if (this.chosen[c] === slot.sat) {
          keep = true;
          break;
        }
      }
      if (!keep) this.releaseSlot(slot);
    }

    // Assign free ready slots (or grow the pool) for newly chosen satellites.
    for (let c = 0; c < count; c++) {
      const sat = this.chosen[c];
      if (shown[sat] === 1) continue;
      const file = (BUS_VERSIONS[catalog.bus[sat]] ?? BUS_VERSIONS[BUS_V2MINI]).model;
      if (this.failedFiles.has(file)) {
        // Retry on demand (the file may have been generated since).
        const at = this.failedAt.get(file) ?? -Infinity;
        if (performance.now() - at >= FAILED_RETRY_MS && !this.loadingFiles.has(file)) {
          this.failedFiles.delete(file);
          this.failedAt.set(file, performance.now()); // throttles the next retry too
          this.createSlot(file);
        }
        continue;
      }
      let slot: Slot | null = null;
      for (let k = 0; k < this.slots.length; k++) {
        const s = this.slots[k];
        if (s.sat < 0 && s.file === file && s.warm === 0 && s.model && s.model.ready) {
          slot = s;
          break;
        }
      }
      if (!slot) {
        // Another instance of the same file: resources are cached, so it is
        // ready within a few frames (picked up by the next search).
        // At the global cap, recycle an idle instance of another file first:
        // a dense train of one bus must not starve another bus's close-ups.
        let usable = 0;
        for (const s of this.slots) if (!this.failedFiles.has(s.file)) usable++;
        if (usable >= this.maxModels && !this.loadingFiles.has(file) && this.evictIdle(file)) usable--;
        if (usable < this.maxModels) this.createSlot(file);
        continue;
      }
      slot.sat = sat;
      slot.lit = Number.NaN;
      slot.wingAngle = Number.NaN;
      shown[sat] = 1;
      const lasers = Math.min(Math.max(0, catalog.lasers[sat] | 0), slot.terminals.length);
      if (lasers !== slot.terminalsShown) {
        slot.terminalsShown = lasers;
        for (let t = 0; t < slot.terminals.length; t++) slot.terminals[t].show = t < lasers;
      }
    }

    // Fade the points near the camera only if every one of them has a model.
    const far2 = POINT_FADE_FAR_M * POINT_FADE_FAR_M;
    let covered = true;
    for (let j = 0; j < n; j++) {
      if (this.foundD2[j] > far2) break; // nearest first
      if (shown[this.foundIdx[j]] !== 1) {
        covered = false;
        break;
      }
    }
    this.setFade(pts, covered);
  }

  private setFade(pts: SatellitePoints | null, on: boolean): void {
    if (on === this.fadeOn) return;
    this.fadeOn = on;
    if (!pts) return;
    if (on) pts.setModelFade(POINT_FADE_NEAR_M, POINT_FADE_FAR_M);
    else pts.setModelFade(0, 0);
  }

  private updateSun(time: JulianDate): void {
    let m = Transforms.computeIcrfToCentralBodyFixedMatrix(time, this.icrfToFixed);
    if (!m) m = Transforms.computeTemeToPseudoFixedMatrix(time, this.icrfToFixed);
    const s = Simon1994PlanetaryPositions.computeSunPositionInEarthInertialFrame(time, this.sunDir);
    Matrix3.multiplyByVector(m, s, s);
    Cartesian3.normalize(s, s);
    this.sunValid = Number.isFinite(s.x) && Number.isFinite(s.y) && Number.isFinite(s.z);
  }

  /** Wings to the sun + eclipse dimming (<= 10 Hz per model). */
  private updateSunDependent(slot: Slot, model: Model, m: Matrix4): void {
    if (!this.sunValid) return;
    const s = this.sunDir;

    if (slot.wings.length > 0) {
      sunInGltfBodyXForward(m, s.x, s.y, s.z, scratchSunBody);
      const theta = wingSunAngle(scratchSunBody, WING_AXIS, WING_REST_NORMAL);
      if (!(Math.abs(theta - slot.wingAngle) < 0.002)) {
        slot.wingAngle = theta;
        for (const w of slot.wings) {
          wingNodeMatrix(w.orig, theta, scratchNodeMatrix, WING_AXIS);
          w.node.matrix = scratchNodeMatrix; // setter clones
        }
      }
    }

    const lit = sunlitFactor(m[12], m[13], m[14], s.x, s.y, s.z);
    if (Math.abs(lit - slot.lit) < 0.005) return;
    slot.lit = lit;
    const light = this.scene.light;
    const k = (ECLIPSE_LIGHT + (1 - ECLIPSE_LIGHT) * lit) * light.intensity;
    const lc = slot.lightColor;
    lc.x = light.color.red * k;
    lc.y = light.color.green * k;
    lc.z = light.color.blue * k;
    model.lightColor = lc; // setter clones; stays defined -> no command rebuild
    const ibl = ECLIPSE_IBL + (1 - ECLIPSE_IBL) * lit; // never 0 -> no shader regeneration
    scratchIbl.x = ibl;
    scratchIbl.y = ibl;
    model.imageBasedLighting.imageBasedLightingFactor = scratchIbl;
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.removePreUpdate();
    const pts = this.boundPoints;
    if (pts && !pts.isDestroyed()) pts.setModelFade(0, 0);
    this.slots.length = 0;
    this.loadingFiles.clear();
    if (!this.scene.isDestroyed() && this.scene.primitives.contains(this.collection)) {
      this.scene.primitives.remove(this.collection); // destroys the models (destroyPrimitives)
    } else if (!this.collection.isDestroyed()) {
      this.collection.destroy();
    }
  }
}
