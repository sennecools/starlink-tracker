/**
 * CameraRig — replaces Cesium's ScreenSpaceCameraController with a custom,
 * frame-rate independent camera.
 *
 * Why: Cesium's controller keeps the camera's up constrained to the Earth's
 * axis (gimbal flips / spins around the poles), converts every view through
 * heading/pitch/roll (singular at the poles and at pitch -90°), and zooms
 * linearly with lurching terrain collisions. This rig instead:
 *
 * - GLOBE mode: state = a frame quaternion (its up axis = geodetic normal at
 *   the look target, its forward axis = the parallel-transported heading
 *   reference) + range + pitch + look height. Grabs, pans and inertia are
 *   world rotations of that frame, so the poles are not special at all.
 * - TRACK mode: the same frame model centred on a moving target; the frame is
 *   transported along with the target every frame (zero lag).
 * - Every control writes a TARGET state; the displayed state follows it with
 *   exponential smoothing on real dt (geodesic for orientation, log for
 *   range); drags leave an exponentially decaying inertia.
 * - The camera is written directly in ECEF doubles each scene.preUpdate
 *   (position / direction / up / right) — no heading/pitch/roll round-trip.
 *
 * Input (Pointer Events on the canvas; keyboard on the window):
 *   left drag             grab the globe (track: orbit the target)
 *   right/middle/ctrl drag heading (x) + pitch (y) around the look point
 *   wheel / pinch         zoom to the cursor (track: range)
 *   double-click          smooth 2.5x zoom towards the point
 *   1 finger              grab (track: orbit); 2 fingers: pinch + twist + pan,
 *                         or a parallel vertical swipe = pitch
 *   arrows / + - / Q E / R F / N   pan / zoom / heading / pitch / north up
 */
import type { Cartographic, Viewer } from 'cesium';
import type { CameraMode, CameraRigApi } from '../shared/phase2.ts';
import {
  DEG,
  HALF_PI,
  WGS84,
  addHeading,
  alignPointToCursor,
  anchorFromPose,
  applyGrab,
  applyZoomAt,
  constrainGlobe,
  constrainTrack,
  copyRig,
  copyVec,
  copyView,
  decay,
  flightHop,
  frameAtLatLon,
  frameAxes,
  frameFromUpForward,
  geodetic,
  globeLookPoint,
  globePointUnderCursor,
  globePose,
  globeView,
  headingFromNorth,
  interpolateView,
  len,
  minRangeForNeed,
  normalize,
  panRotation,
  pixelRay,
  pose,
  poseFromView,
  quat,
  quatFromRotationVector,
  rayEllipsoid,
  rigState,
  rigView,
  rotateFrameWorld,
  scale,
  setVec,
  smoothAlpha,
  smoothQuat,
  smoothRange,
  smootherstep,
  toGeodetic,
  transportFrameTo,
  vec3,
} from './cameraMath.ts';
import type { Ellipsoid, Lens, Pose, RigState, RigView, Vec3 } from './cameraMath.ts';

export interface CameraRigOptions {
  /** Minimum camera height above the terrain/ellipsoid under it, metres (default 250). */
  minAltitudeM?: number;
  /** Maximum globe-mode range, metres (default 150,000 km). */
  maxRangeM?: number;
}

type PositionGetter = (out: { x: number; y: number; z: number }) => boolean;
type InputKind = 'drag' | 'zoom' | 'tilt';
type Gesture = 'none' | 'pending' | 'grab' | 'orbit' | 'two-pending' | 'pinch' | 'tilt2';

// ---------------------------------------------------------------------------
// Tunables
// ---------------------------------------------------------------------------

const DEFAULT_MIN_ALTITUDE_M = 250;
const DEFAULT_MAX_RANGE_M = 150_000_000;
const TRACK_MIN_RANGE_M = 30;
const TRACK_MAX_RANGE_M = 20_000_000;
const TRACK_DEFAULT_RANGE_M = 2_000_000;
const TRACK_DEFAULT_PITCH = -30 * DEG;
const TRACK_DEFAULT_DURATION_S = 2.5;
const RELEASE_RANGE_M = 6_000_000;
const RELEASE_DEFAULT_DURATION_S = 2.2;
const OVERVIEW = { lat: 25 * DEG, lon: 5 * DEG, range: 16_000_000 } as const;

/** Smoothing time constants (s). */
const TAU_GRAB = 0.04;
const TAU_ORBIT_DRAG = 0.05;
const TAU_WHEEL = 0.15;
const TAU_KEYS = 0.12;
const TAU_DBLCLICK = 0.3;
const TAU_NORTH = 0.35;
const TAU_LOOK_HEIGHT = 0.35;
/** Inertia decay after releasing a drag (s). */
const TAU_INERTIA = 0.35;
/** Velocity estimator time constant (s) and staleness cut-off (ms). */
const TAU_VELOCITY = 0.05;
const VELOCITY_STALE_MS = 80;

const DRAG_THRESHOLD_MOUSE_PX = 3;
const DRAG_THRESHOLD_TOUCH_PX = 6;
const WHEEL_GESTURE_GAP_MS = 300;

/** Orbit drag: radians per CSS px (scaled by canvas size). */
const ORBIT_HEADING_PER_WIDTH = Math.PI;
const ORBIT_PITCH_PER_HEIGHT = 0.8 * Math.PI;

/** Keyboard rates. */
const KEY_PAN_RANGE_PER_S = 0.9;
const KEY_ZOOM_PER_S = 1.6;
const KEY_HEADING_PER_S = 70 * DEG;
const KEY_PITCH_PER_S = 45 * DEG;

/** Terrain sampling. */
const TERRAIN_CAMERA_MAX_HEIGHT_M = 40_000;
const TERRAIN_LOOK_MAX_RANGE_M = 60_000;
const TERRAIN_SAMPLE_MS = 110;

const MAX_FRAME_DT_S = 0.1;
/** Settle flight after interrupting a globe flight whose pose is outside the globe limits (s). */
const SETTLE_DURATION_S = 0.35;

// Keyboard bits
const K_UP = 1;
const K_DOWN = 2;
const K_LEFT = 4;
const K_RIGHT = 8;
const K_ZOOM_IN = 16;
const K_ZOOM_OUT = 32;
const K_ROT_L = 64;
const K_ROT_R = 128;
const K_TILT_UP = 256;
const K_TILT_DOWN = 512;

function keyBit(e: KeyboardEvent): number {
  switch (e.key) {
    case 'ArrowUp':
      return K_UP;
    case 'ArrowDown':
      return K_DOWN;
    case 'ArrowLeft':
      return K_LEFT;
    case 'ArrowRight':
      return K_RIGHT;
    case '+':
    case '=':
      return K_ZOOM_IN;
    case '-':
    case '_':
      return K_ZOOM_OUT;
  }
  if (e.code === 'NumpadAdd') return K_ZOOM_IN;
  if (e.code === 'NumpadSubtract') return K_ZOOM_OUT;
  switch (e.key.length === 1 ? e.key.toLowerCase() : '') {
    case 'q':
      return K_ROT_L;
    case 'e':
      return K_ROT_R;
    case 'r':
      return K_TILT_UP;
    case 'f':
      return K_TILT_DOWN;
  }
  return 0;
}

function keyKind(bit: number): InputKind {
  if (bit & (K_ZOOM_IN | K_ZOOM_OUT)) return 'zoom';
  if (bit & (K_UP | K_DOWN | K_LEFT | K_RIGHT)) return 'drag';
  return 'tilt';
}

function isTypingTarget(target: EventTarget | null): boolean {
  const t = target as { tagName?: string; isContentEditable?: boolean } | null;
  if (!t) return false;
  const tag = t.tagName;
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || t.isContentEditable === true;
}

/** HUD scroll containers: arrow keys there scroll the content, not the camera. */
const KEY_SCROLLER_SELECTOR = '.sx-panel__scroll, .sx-legend__list';

function isScrollerTarget(target: EventTarget | null): boolean {
  const t = target as { closest?: (selector: string) => unknown } | null;
  return typeof t?.closest === 'function' && t.closest(KEY_SCROLLER_SELECTOR) != null;
}

interface Slot {
  id: number;
  touch: boolean;
  x: number;
  y: number;
  downX: number;
  downY: number;
}

function slot(): Slot {
  return { id: -1, touch: false, x: 0, y: 0, downX: 0, downY: 0 };
}

interface EventHost {
  addEventListener(type: string, fn: (e: never) => void, opts?: AddEventListenerOptions | boolean): void;
  removeEventListener(type: string, fn: (e: never) => void, opts?: EventListenerOptions | boolean): void;
}

// ---------------------------------------------------------------------------

export class CameraRig implements CameraRigApi {
  onUserInput: ((kind: InputKind) => void) | null = null;

  private readonly viewer: Viewer;
  private readonly canvas: HTMLCanvasElement;
  private readonly win: EventHost;
  private readonly ell: Ellipsoid;
  private readonly minAlt: number;
  private readonly maxRange: number;

  private _mode: CameraMode = 'globe';
  private destroyed = false;
  private lastTick = Number.NaN;
  private removePreUpdate: (() => void) | null = null;
  private readonly prevController: { enableInputs: boolean; enableCollisionDetection: boolean };
  private readonly prevTouchAction: string;

  // --- states ---
  private readonly globeCur: RigState = rigState();
  private readonly globeTgt: RigState = rigState();
  private readonly trackCur: RigState = rigState();
  private readonly trackTgt: RigState = rigState();
  private trackGet: PositionGetter | null = null;
  private readonly sPos: Vec3 = vec3();
  /** What was displayed last frame (view + resulting pose). */
  private readonly disp: RigView = rigView();
  private readonly pose: Pose = pose();
  /** Scratch pose for input hit-testing against the target state. */
  private readonly pose2: Pose = pose();

  // --- smoothing time constants (set by the latest input) ---
  private tauRot = TAU_WHEEL;
  private tauRange = TAU_WHEEL;
  private tauPitch = TAU_WHEEL;

  // --- flight ---
  private flightActive = false;
  private flightT = 0;
  private flightDur = 1;
  private flightHop = 0;
  private flightStartGet: PositionGetter | null = null;
  private readonly f0: RigView = rigView();
  private readonly fEnd: RigView = rigView();
  private flightResolve: (() => void) | null = null;
  /** A short settle flight after cancelling a globe flight (user input does not cancel it). */
  private settling = false;

  // --- inertia ---
  private readonly grabVel: Vec3 = vec3();
  private headingVel = 0;
  private pitchVel = 0;
  private lastMoveT = 0;

  // --- pointers ---
  private gesture: Gesture = 'none';
  private pendingKind: 'grab' | 'orbit' = 'grab';
  private readonly s0: Slot = slot();
  private readonly s1: Slot = slot();
  private lastX = 0;
  private lastY = 0;
  private grabRho = WGS84.a;
  private tfDist = 0;
  private tfAngle = 0;
  private tfMidX = 0;
  private tfMidY = 0;
  private tfStartDist = 0;
  private tfStartAngle = 0;
  private tfStartMidX = 0;
  private tfStartMidY = 0;
  private savedCursor: string | null = null;
  private lastWheelT = -1e9;
  private moved = false;
  private rectLeft = 0;
  private rectTop = 0;
  private rectW = 1;
  private rectH = 1;
  private readonly lens: Lens = { tanX: 0.6, tanY: 0.577 };

  // --- zoom-to-cursor anchor (wheel / double-click) ---
  private zoomAnchor = false;
  private zoomNx = 0;
  private zoomNy = 0;
  private readonly zoomPoint: Vec3 = vec3();
  private readonly zoomLens: Lens = { tanX: 0.6, tanY: 0.577 };

  // --- keyboard ---
  private keyMask = 0;
  private readonly keyCodes: Record<string, number> = {};

  // --- terrain cache ---
  private camTerrain = 0;
  private camSampleT = -1e9;
  private lookSampleT = -1e9;
  private readonly carto = { longitude: 0, latitude: 0, height: 0 };
  private readonly geo = geodetic();

  // --- scratch ---
  private readonly tmpV: Vec3 = vec3();
  private readonly tmpV2: Vec3 = vec3();
  private readonly tmpQ = quat();
  private readonly tmpState: RigState = rigState();

  constructor(viewer: Viewer, opts: CameraRigOptions = {}) {
    this.viewer = viewer;
    const scene = viewer.scene;
    this.canvas = scene.canvas;
    this.win = (this.canvas.ownerDocument?.defaultView ?? globalThis) as unknown as EventHost;
    const radii = scene.globe?.ellipsoid?.radii;
    this.ell = radii ? { a: radii.x, b: radii.z } : { a: WGS84.a, b: WGS84.b };
    this.minAlt = Math.max(1, opts.minAltitudeM ?? DEFAULT_MIN_ALTITUDE_M);
    this.maxRange = Math.max(1_000, opts.maxRangeM ?? DEFAULT_MAX_RANGE_M);

    const ctl = scene.screenSpaceCameraController;
    this.prevController = { enableInputs: ctl.enableInputs, enableCollisionDetection: ctl.enableCollisionDetection };
    ctl.enableInputs = false;
    ctl.enableCollisionDetection = false;

    this.ensureIdentityTransform();
    this.initFromCamera();

    this.prevTouchAction = this.canvas.style.touchAction;
    this.canvas.style.touchAction = 'none';

    const c = this.canvas as unknown as EventHost;
    c.addEventListener('pointerdown', this.onPointerDown);
    c.addEventListener('pointermove', this.onPointerMove, { passive: true });
    c.addEventListener('pointerup', this.onPointerUp, { passive: true });
    c.addEventListener('pointercancel', this.onPointerCancel, { passive: true });
    c.addEventListener('wheel', this.onWheel, { passive: false });
    c.addEventListener('dblclick', this.onDblClick);
    c.addEventListener('contextmenu', this.onContextMenu);
    this.win.addEventListener('keydown', this.onKeyDown);
    this.win.addEventListener('keyup', this.onKeyUp, { passive: true });
    this.win.addEventListener('blur', this.onBlur, { passive: true });

    this.removePreUpdate = scene.preUpdate.addEventListener(this.onPreUpdate);
  }

  get mode(): CameraMode {
    return this._mode;
  }

  /** True while a scripted flight (overview / track fly-in / release) is running. */
  get flying(): boolean {
    return this.flightActive;
  }

  /**
   * True if the most recent pointer press turned into a drag (moved past the
   * click threshold). Lets click handlers ignore the click that ends a drag.
   */
  get lastPressDragged(): boolean {
    return this.moved;
  }

  /**
   * Target range (metres) of the current track, NaN in globe mode. Lets a
   * re-target (e.g. hopping to a laser-link neighbour) keep the user's zoom.
   */
  get trackRangeM(): number {
    return this._mode === 'track' ? this.trackTgt.range : Number.NaN;
  }

  /** True while a pointer drag / pinch is in progress (e.g. to pause hover picking). */
  get dragging(): boolean {
    const g = this.gesture;
    return g === 'grab' || g === 'orbit' || g === 'pinch' || g === 'tilt2';
  }

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

  flyToOverview(durationS = 3.5): Promise<void> {
    if (this.destroyed) return Promise.resolve();
    const liveStart = this._mode === 'track' && !this.flightActive ? this.trackGet : null;
    this.resolveFlight();
    this.stopInertia();
    const g = this.globeTgt;
    frameAtLatLon(OVERVIEW.lat, OVERVIEW.lon, 0, g.q);
    g.range = OVERVIEW.range;
    g.pitch = -HALF_PI;
    g.h = 0;
    copyRig(g, this.globeCur);
    this._mode = 'globe';
    this.trackGet = null;
    this.startFlight(durationS, liveStart);
    return new Promise<void>((resolve) => {
      if (this.flightActive) this.flightResolve = resolve;
      else resolve();
    });
  }

  track(getPosition: PositionGetter, opts: { rangeM?: number; durationS?: number } = {}): void {
    if (this.destroyed) return;
    const s = this.tmpV2;
    if (!getPosition(s) || !(len(s) > 0)) return;
    // Live start only while still on the same target (a re-target starts from the frozen view).
    const liveStart = this._mode === 'track' && !this.flightActive && this.trackGet !== getPosition ? this.trackGet : null;
    this.resolveFlight();
    this.stopInertia();
    copyVec(s, this.sPos);

    // Track frame: up = target direction, forward = current view direction
    // projected on the tangent plane (so the fly-in does not spin).
    const t = this.trackTgt;
    normalize(this.sPos, this.tmpV);
    frameFromUpForward(this.tmpV, this.pose.dir, this.pose.up, t.q);
    t.range = Math.min(TRACK_MAX_RANGE_M, Math.max(TRACK_MIN_RANGE_M, opts.rangeM ?? TRACK_DEFAULT_RANGE_M));
    t.pitch = TRACK_DEFAULT_PITCH;
    t.h = 0;
    this.constrainTrackState(t);
    copyRig(t, this.trackCur);

    this._mode = 'track';
    this.trackGet = getPosition;
    this.tauRot = TAU_KEYS;
    this.tauRange = TAU_WHEEL;
    this.tauPitch = TAU_KEYS;
    this.startFlight(opts.durationS ?? TRACK_DEFAULT_DURATION_S, liveStart);
  }

  release(durationS = RELEASE_DEFAULT_DURATION_S): void {
    if (this.destroyed || this._mode !== 'track') return;
    const liveStart = this.flightActive ? null : this.trackGet;
    this.resolveFlight();
    this.stopInertia();
    // Globe view centred under the (last known) target position.
    const n = this.tmpV;
    toGeodetic(this.sPos, this.ell, this.geo);
    setVec(n, Math.cos(this.geo.lat) * Math.cos(this.geo.lon), Math.cos(this.geo.lat) * Math.sin(this.geo.lon), Math.sin(this.geo.lat));
    frameAxes(this.trackCur.q, null, this.tmpV2, null);
    const g = this.globeTgt;
    frameFromUpForward(n, this.tmpV2, this.pose.up, g.q);
    g.range = RELEASE_RANGE_M;
    g.pitch = -HALF_PI;
    g.h = 0;
    copyRig(g, this.globeCur);
    this._mode = 'globe';
    this.trackGet = null;
    this.startFlight(durationS, liveStart);
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.resolveFlight();
    this.removePreUpdate?.();
    this.removePreUpdate = null;
    const c = this.canvas as unknown as EventHost;
    c.removeEventListener('pointerdown', this.onPointerDown);
    c.removeEventListener('pointermove', this.onPointerMove);
    c.removeEventListener('pointerup', this.onPointerUp);
    c.removeEventListener('pointercancel', this.onPointerCancel);
    c.removeEventListener('wheel', this.onWheel);
    c.removeEventListener('dblclick', this.onDblClick);
    c.removeEventListener('contextmenu', this.onContextMenu);
    this.win.removeEventListener('keydown', this.onKeyDown);
    this.win.removeEventListener('keyup', this.onKeyUp);
    this.win.removeEventListener('blur', this.onBlur);
    this.endPointerGesture(false);
    this.canvas.style.touchAction = this.prevTouchAction;
    const scene = this.viewer.scene;
    if (!scene.isDestroyed()) {
      const ctl = scene.screenSpaceCameraController;
      ctl.enableInputs = this.prevController.enableInputs;
      ctl.enableCollisionDetection = this.prevController.enableCollisionDetection;
    }
    this.onUserInput = null;
    this.trackGet = null;
    this.flightStartGet = null;
  }

  // -------------------------------------------------------------------------
  // Setup helpers
  // -------------------------------------------------------------------------

  private ensureIdentityTransform(): void {
    const camera = this.viewer.scene.camera;
    const m = camera.transform as unknown as ArrayLike<number>;
    let identity = true;
    for (let i = 0; i < 16; i++) {
      if (m[i] !== (i % 5 === 0 ? 1 : 0)) {
        identity = false;
        break;
      }
    }
    if (identity) return;
    const Ctor = (camera.transform as object).constructor as new (...v: number[]) => typeof camera.transform;
    camera.lookAtTransform(new Ctor(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1));
  }

  /** Initial globe state reproducing Cesium's current camera (falls back to the overview). */
  private initFromCamera(): void {
    const camera = this.viewer.scene.camera;
    const p = this.pose;
    copyVec(camera.positionWC, p.pos);
    copyVec(camera.directionWC, p.dir);
    copyVec(camera.upWC, p.up);
    copyVec(camera.rightWC, p.right);
    const ok =
      Number.isFinite(p.pos.x) &&
      len(p.pos) > this.ell.b &&
      anchorFromPose(p, this.ell, Math.max(1, len(p.pos) - this.ell.a), this.globeCur);
    if (!ok) {
      frameAtLatLon(OVERVIEW.lat, OVERVIEW.lon, 0, this.globeCur.q);
      this.globeCur.range = OVERVIEW.range;
      this.globeCur.pitch = -HALF_PI;
      this.globeCur.h = 0;
    }
    this.globeCur.h = 0;
    constrainGlobe(this.globeCur, this.minAlt - this.globeCur.h, this.maxRange);
    copyRig(this.globeCur, this.globeTgt);
    globeView(this.globeCur, this.ell, this.disp);
    poseFromView(this.disp, this.pose);
  }

  // -------------------------------------------------------------------------
  // Flights
  // -------------------------------------------------------------------------

  private startFlight(durationS: number, liveStart: PositionGetter | null): void {
    this.zoomAnchor = false;
    copyView(this.disp, this.f0);
    this.flightStartGet = liveStart;
    this.flightT = 0;
    // prefers-reduced-motion: every scripted flight becomes a near-instant cut (read per flight, so a
    // change made while the page is open applies without a listener).
    this.flightDur = this.prefersReducedMotion() ? 0.05 : Math.max(0.05, Number.isFinite(durationS) ? durationS : 1);
    this.flightActive = true;
    this.computeModeView(this.fEnd);
    this.flightHop = flightHop(this.f0, this.fEnd, this.ell.a);
  }

  private prefersReducedMotion(): boolean {
    const mm = (this.win as unknown as { matchMedia?: (q: string) => { matches: boolean } }).matchMedia;
    try {
      return typeof mm === 'function' && mm.call(this.win, '(prefers-reduced-motion: reduce)').matches === true;
    } catch {
      return false;
    }
  }

  private resolveFlight(): void {
    this.flightActive = false;
    this.settling = false;
    this.flightStartGet = null;
    const r = this.flightResolve;
    this.flightResolve = null;
    r?.();
  }

  /**
   * User input while a globe flight runs: continue from where the camera is, as a globe state.
   * When that pose is outside the globe limits (e.g. early in a release flight the view still
   * looks at the satellite, above the horizon), the limits are eased in with a short settle
   * flight from the displayed view instead of snapping the camera in one frame.
   */
  private cancelGlobeFlight(): void {
    if (!this.flightActive || this._mode !== 'globe' || this.settling) return;
    const s = this.tmpState;
    if (!anchorFromPose(this.pose, this.ell, this.disp.range, s)) {
      this.resolveFlight();
      return;
    }
    const p0 = s.pitch;
    const r0 = s.range;
    const h0 = s.h;
    s.h = 0; // a ray miss anchors in space; the look height is re-sampled from the terrain anyway
    constrainGlobe(s, this.needFor(s), this.maxRange);
    copyRig(s, this.globeCur);
    copyRig(s, this.globeTgt);
    this.resolveFlight();
    if (s.pitch !== p0 || s.range !== r0 || h0 !== 0) {
      this.settling = true;
      // f0 = the displayed view; the end follows globeCur (and so the user's input) every frame.
      this.startFlight(SETTLE_DURATION_S, null);
    }
  }

  // -------------------------------------------------------------------------
  // Frame update
  // -------------------------------------------------------------------------

  private readonly onPreUpdate = (): void => {
    if (this.destroyed) return;
    const now = performance.now();
    let dt = Number.isNaN(this.lastTick) ? 0 : (now - this.lastTick) / 1000;
    this.lastTick = now;
    if (!(dt > 0)) dt = 0;
    else if (dt > MAX_FRAME_DT_S) dt = MAX_FRAME_DT_S;
    this.tick(dt, now);
  };

  /** One frame: inputs -> targets -> constraints -> smoothing -> flight -> camera. */
  private tick(dt: number, now: number): void {
    // 1. Moving targets, read for THIS frame (zero lag).
    if (this._mode === 'track' && this.trackGet) {
      if (this.trackGet(this.sPos) && len(this.sPos) > 0) {
        transportFrameTo(this.trackTgt.q, this.sPos);
        transportFrameTo(this.trackCur.q, this.sPos);
      } else {
        this.trackGet = null;
        this.release();
      }
    }
    if (this.flightActive && this.flightStartGet) {
      const p = this.tmpV;
      if (this.flightStartGet(p) && len(p) > 0) {
        transportFrameTo(this.f0.q, p);
        copyVec(p, this.f0.look);
      } else {
        this.flightStartGet = null;
      }
    }

    // 2. Continuous inputs.
    if (this.keyMask !== 0) this.applyKeys(dt);
    this.applyInertia(dt);

    // 3. Constraints + smoothing of the active mode.
    if (this._mode === 'globe') {
      const tgt = this.globeTgt;
      const cur = this.globeCur;
      constrainGlobe(tgt, this.needFor(tgt), this.maxRange);
      smoothQuat(cur.q, tgt.q, dt, this.tauRot, cur.q);
      cur.range = smoothRange(cur.range, tgt.range, dt, this.tauRange);
      cur.pitch += (tgt.pitch - cur.pitch) * smoothAlpha(dt, this.tauPitch);
      cur.h += (tgt.h - cur.h) * smoothAlpha(dt, TAU_LOOK_HEIGHT);
      constrainGlobe(cur, this.needFor(cur), this.maxRange);
      if (this.zoomAnchor) {
        // Keep the zoom point locked under the cursor while the range eases;
        // the target turns along so both stay consistent.
        if (alignPointToCursor(cur, this.ell, this.zoomLens, this.zoomNx, this.zoomNy, this.zoomPoint, this.tmpQ)) {
          rotateFrameWorld(tgt.q, this.tmpQ);
        } else {
          this.zoomAnchor = false;
        }
        if (Math.abs(Math.log(cur.range / tgt.range)) < 1e-6) this.zoomAnchor = false;
      }
    } else {
      const tgt = this.trackTgt;
      const cur = this.trackCur;
      this.constrainTrackState(tgt);
      smoothQuat(cur.q, tgt.q, dt, this.tauRot, cur.q);
      cur.range = smoothRange(cur.range, tgt.range, dt, this.tauRange);
      cur.pitch += (tgt.pitch - cur.pitch) * smoothAlpha(dt, this.tauPitch);
      this.constrainTrackState(cur);
    }

    // 4. View: the mode's view, or a flight towards it.
    if (this.flightActive) {
      this.flightT += dt;
      const u = Math.min(1, this.flightT / this.flightDur);
      this.computeModeView(this.fEnd);
      interpolateView(this.f0, this.fEnd, smootherstep(u), this.flightHop, this.disp);
      if (u >= 1) this.resolveFlight();
    } else {
      this.computeModeView(this.disp);
    }

    // 5. Pose, safety floor, write the camera.
    poseFromView(this.disp, this.pose);
    const geo = toGeodetic(this.pose.pos, this.ell, this.geo);
    const floor = this.camTerrain + this.minAlt * 0.5;
    if (geo.height < floor) {
      const up = this.tmpV;
      const cl = Math.cos(geo.lat);
      setVec(up, cl * Math.cos(geo.lon), cl * Math.sin(geo.lon), Math.sin(geo.lat));
      const p = this.pose.pos;
      const d = floor - geo.height;
      setVec(p, p.x + up.x * d, p.y + up.y * d, p.z + up.z * d);
    }
    this.writeCamera();

    // 6. Terrain samples for the next frames (cheap, throttled, only when low).
    this.sampleTerrain(now, geo.lat, geo.lon, geo.height);
  }

  private computeModeView(out: RigView): void {
    if (this._mode === 'globe') {
      globeView(this.globeCur, this.ell, out);
    } else {
      const c = this.trackCur;
      copyVec(this.sPos, out.look);
      out.q.x = c.q.x;
      out.q.y = c.q.y;
      out.q.z = c.q.z;
      out.q.w = c.q.w;
      out.range = c.range;
      out.pitch = c.pitch;
    }
  }

  private writeCamera(): void {
    const cam = this.viewer.scene.camera;
    const p = this.pose;
    copyVec(p.pos, cam.position);
    copyVec(p.dir, cam.direction);
    copyVec(p.up, cam.up);
    copyVec(p.right, cam.right);
  }

  /** Metres the camera must stay above the look point (min altitude + terrain under camera - look height). */
  private needFor(s: RigState): number {
    return this.minAlt + Math.max(0, this.camTerrain) - s.h;
  }

  private constrainTrackState(s: RigState): void {
    const groundR = this.ell.a + this.minAlt + Math.max(0, this.camTerrain);
    constrainTrack(s, len(this.sPos), groundR, TRACK_MIN_RANGE_M, TRACK_MAX_RANGE_M);
  }

  private sampleTerrain(now: number, lat: number, lon: number, camHeight: number): void {
    const globe = this.viewer.scene.globe;
    if (camHeight < TERRAIN_CAMERA_MAX_HEIGHT_M && globe) {
      if (now - this.camSampleT > TERRAIN_SAMPLE_MS) {
        this.camSampleT = now;
        const v = this.heightAt(lat, lon);
        if (v !== undefined) this.camTerrain = v;
      }
    } else {
      this.camTerrain = 0;
    }
    if (this._mode !== 'globe') return;
    const tgt = this.globeTgt;
    if (tgt.range < TERRAIN_LOOK_MAX_RANGE_M && globe) {
      if (now - this.lookSampleT > TERRAIN_SAMPLE_MS) {
        this.lookSampleT = now;
        const up = this.tmpV;
        frameAxes(tgt.q, null, null, up);
        const v = this.heightAt(Math.asin(Math.max(-1, Math.min(1, up.z))), Math.atan2(up.y, up.x));
        if (v !== undefined) tgt.h = v;
      }
    } else {
      tgt.h = 0;
    }
  }

  private heightAt(lat: number, lon: number): number | undefined {
    const globe = this.viewer.scene.globe;
    if (!globe) return undefined;
    this.carto.latitude = lat;
    this.carto.longitude = lon;
    this.carto.height = 0;
    const v = globe.getHeight(this.carto as unknown as Cartographic);
    return v !== undefined && Number.isFinite(v) ? v : undefined;
  }

  // -------------------------------------------------------------------------
  // Continuous inputs
  // -------------------------------------------------------------------------

  private applyKeys(dt: number): void {
    this.cancelGlobeFlight();
    const m = this.keyMask;
    const fwd = (m & K_UP ? 1 : 0) - (m & K_DOWN ? 1 : 0);
    const side = (m & K_RIGHT ? 1 : 0) - (m & K_LEFT ? 1 : 0);
    const zoom = (m & K_ZOOM_OUT ? 1 : 0) - (m & K_ZOOM_IN ? 1 : 0);
    const rot = (m & K_ROT_R ? 1 : 0) - (m & K_ROT_L ? 1 : 0);
    const tilt = (m & K_TILT_UP ? 1 : 0) - (m & K_TILT_DOWN ? 1 : 0);
    const tgt = this._mode === 'globe' ? this.globeTgt : this.trackTgt;
    if (this._mode === 'globe') {
      if (fwd !== 0 || side !== 0) {
        const k = (side !== 0 && fwd !== 0 ? Math.SQRT1_2 : 1) * KEY_PAN_RANGE_PER_S * tgt.range * dt;
        globeLookPoint(tgt, this.ell, this.tmpV);
        panRotation(tgt.q, fwd * k, side * k, Math.max(1, len(this.tmpV)), this.tmpQ);
        rotateFrameWorld(tgt.q, this.tmpQ);
      }
    } else {
      // Track: arrows orbit.
      if (side !== 0) addHeading(tgt.q, side * KEY_HEADING_PER_S * dt);
      if (fwd !== 0) tgt.pitch -= fwd * KEY_PITCH_PER_S * dt;
    }
    if (zoom !== 0) tgt.range *= Math.exp(zoom * KEY_ZOOM_PER_S * dt);
    if (rot !== 0) addHeading(tgt.q, rot * KEY_HEADING_PER_S * dt);
    if (tilt !== 0) tgt.pitch += tilt * KEY_PITCH_PER_S * dt;
  }

  private applyInertia(dt: number): void {
    if (dt <= 0) return;
    const active = this.gesture === 'grab' || this.gesture === 'orbit' || this.gesture === 'pinch' || this.gesture === 'tilt2';
    if (active) return;
    const v = this.grabVel;
    if (v.x !== 0 || v.y !== 0 || v.z !== 0) {
      if (this._mode !== 'globe') {
        setVec(v, 0, 0, 0);
      } else {
        scale(v, dt, this.tmpV);
        quatFromRotationVector(this.tmpV, this.tmpQ);
        rotateFrameWorld(this.globeTgt.q, this.tmpQ);
        scale(v, decay(dt, TAU_INERTIA), v);
        // Stop once the surface drifts less than 0.2% of the view per second.
        if (len(v) * this.ell.a < this.globeTgt.range * 0.002) setVec(v, 0, 0, 0);
      }
    }
    if (this.headingVel !== 0 || this.pitchVel !== 0) {
      const tgt = this._mode === 'globe' ? this.globeTgt : this.trackTgt;
      addHeading(tgt.q, this.headingVel * dt);
      tgt.pitch += this.pitchVel * dt;
      const k = decay(dt, TAU_INERTIA);
      this.headingVel *= k;
      this.pitchVel *= k;
      if (Math.abs(this.headingVel) < 0.002) this.headingVel = 0;
      if (Math.abs(this.pitchVel) < 0.002) this.pitchVel = 0;
    }
  }

  private stopInertia(): void {
    setVec(this.grabVel, 0, 0, 0);
    this.headingVel = 0;
    this.pitchVel = 0;
  }

  // -------------------------------------------------------------------------
  // Gesture plumbing
  // -------------------------------------------------------------------------

  private beginGesture(kind: InputKind): void {
    this.cancelGlobeFlight();
    this.zoomAnchor = false;
    this.onUserInput?.(kind);
  }

  private refreshRect(): void {
    const r = this.canvas.getBoundingClientRect();
    this.rectLeft = r.left;
    this.rectTop = r.top;
    this.rectW = r.width > 0 ? r.width : 1;
    this.rectH = r.height > 0 ? r.height : 1;
  }

  private refreshLens(): void {
    const fr = this.viewer.scene.camera.frustum as { fovy?: number };
    const fovy = fr.fovy !== undefined && fr.fovy > 0 ? fr.fovy : 60 * DEG;
    this.lens.tanY = Math.tan(fovy / 2);
    this.lens.tanX = this.lens.tanY * (this.rectW / this.rectH);
  }

  private ndcX(x: number): number {
    return (x / this.rectW) * 2 - 1;
  }

  private ndcY(y: number): number {
    return 1 - (y / this.rectH) * 2;
  }

  /** Radius of the grab sphere: distance from the Earth's centre to the surface under (x, y). */
  private computeGrabRho(x: number, y: number): void {
    this.refreshLens();
    const tgt = this.globeTgt;
    globePose(tgt, this.ell, this.pose2);
    pixelRay(this.pose2, this.lens, this.ndcX(x), this.ndcY(y), this.tmpV);
    const t = rayEllipsoid(this.pose2.pos, this.tmpV, this.ell.a + tgt.h, this.ell.b + tgt.h);
    if (t > 0) {
      const p = this.tmpV2;
      setVec(p, this.pose2.pos.x + this.tmpV.x * t, this.pose2.pos.y + this.tmpV.y * t, this.pose2.pos.z + this.tmpV.z * t);
      this.grabRho = len(p);
    } else {
      globeLookPoint(tgt, this.ell, this.tmpV2);
      this.grabRho = len(this.tmpV2);
    }
  }

  private setCursor(c: string | null): void {
    const style = this.canvas.style;
    if (c !== null) {
      if (this.savedCursor === null) this.savedCursor = style.cursor;
      style.cursor = c;
    } else if (this.savedCursor !== null) {
      if (style.cursor === 'grabbing' || style.cursor === 'move') style.cursor = this.savedCursor;
      this.savedCursor = null;
    }
  }

  private slotFor(id: number): Slot | null {
    if (this.s0.id === id) return this.s0;
    if (this.s1.id === id) return this.s1;
    return null;
  }

  private localX(e: PointerEvent | MouseEvent): number {
    return e.clientX - this.rectLeft;
  }

  private localY(e: PointerEvent | MouseEvent): number {
    return e.clientY - this.rectTop;
  }

  // -------------------------------------------------------------------------
  // Pointer events
  // -------------------------------------------------------------------------

  private readonly onPointerDown = (e: PointerEvent): void => {
    if (this.destroyed) return;
    const touch = e.pointerType === 'touch';
    if (!touch) {
      if (this.gesture !== 'none' || e.button > 2) return;
      if (e.button === 1) e.preventDefault(); // no middle-click autoscroll
    }
    if (this.s0.id < 0) {
      this.refreshRect();
      const s = this.s0;
      s.id = e.pointerId;
      s.touch = touch;
      s.x = s.downX = this.localX(e);
      s.y = s.downY = this.localY(e);
      const orbitButton = !touch && (e.button === 1 || e.button === 2 || e.ctrlKey || e.metaKey);
      this.pendingKind = orbitButton || this._mode === 'track' ? 'orbit' : 'grab';
      this.gesture = 'pending';
      this.moved = false;
      this.stopInertia();
      try {
        this.canvas.setPointerCapture(e.pointerId);
      } catch {
        /* capture is best-effort */
      }
      return;
    }
    if (touch && this.s1.id < 0 && this.s0.touch) {
      const s = this.s1;
      s.id = e.pointerId;
      s.touch = true;
      s.x = s.downX = this.localX(e);
      s.y = s.downY = this.localY(e);
      this.stopInertia();
      this.startTwoFinger();
    }
  };

  private startTwoFinger(): void {
    this.measureTwoFinger();
    this.tfStartDist = this.tfDist;
    this.tfStartAngle = this.tfAngle;
    this.tfStartMidX = this.tfMidX;
    this.tfStartMidY = this.tfMidY;
    this.gesture = 'two-pending';
    if (this._mode === 'globe') this.computeGrabRho(this.tfMidX, this.tfMidY);
  }

  private measureTwoFinger(): void {
    const dx = this.s1.x - this.s0.x;
    const dy = this.s1.y - this.s0.y;
    this.tfDist = Math.max(1, Math.hypot(dx, dy));
    this.tfAngle = Math.atan2(dy, dx);
    this.tfMidX = (this.s0.x + this.s1.x) / 2;
    this.tfMidY = (this.s0.y + this.s1.y) / 2;
  }

  private readonly onPointerMove = (e: PointerEvent): void => {
    if (this.destroyed || this.gesture === 'none') return;
    const s = this.slotFor(e.pointerId);
    if (!s) return;
    if (!s.touch && e.buttons === 0) {
      // The mouse button went up where we never saw it (e.g. released outside the window
      // after the OS dropped the capture): end the gesture instead of grabbing a bare cursor.
      this.pointerEnded(e, false);
      return;
    }
    s.x = this.localX(e);
    s.y = this.localY(e);
    const t = e.timeStamp;
    switch (this.gesture) {
      case 'pending': {
        if (s !== this.s0) return;
        const thr = s.touch ? DRAG_THRESHOLD_TOUCH_PX : DRAG_THRESHOLD_MOUSE_PX;
        if (Math.hypot(s.x - s.downX, s.y - s.downY) <= thr) return;
        this.moved = true;
        this.lastX = s.downX;
        this.lastY = s.downY;
        this.lastMoveT = t;
        if (this.pendingKind === 'grab' && this._mode === 'globe') {
          this.beginGesture('drag');
          this.gesture = 'grab';
          this.setCursor('grabbing');
          this.computeGrabRho(s.downX, s.downY);
        } else {
          this.beginGesture('tilt');
          this.gesture = 'orbit';
          this.setCursor('move');
        }
        this.dragTo(s.x, s.y, t);
        return;
      }
      case 'grab':
      case 'orbit':
        if (s === this.s0) this.dragTo(s.x, s.y, t);
        return;
      case 'two-pending': {
        const prevDist = this.tfStartDist;
        const prevAngle = this.tfStartAngle;
        this.measureTwoFinger();
        const dDist = this.tfDist - prevDist;
        const dAng = Math.abs(wrapAngle(this.tfAngle - prevAngle));
        const dmx = this.tfMidX - this.tfStartMidX;
        const dmy = this.tfMidY - this.tfStartMidY;
        if (Math.abs(dDist) < 8 && dAng < 4 * DEG && Math.hypot(dmx, dmy) < 8) return;
        this.moved = true;
        // Parallel vertical swipe with the fingers side by side -> pitch.
        const fingersLevel = Math.abs(Math.sin(this.tfStartAngle)) < 0.5;
        const d0 = this.s0.y - this.s0.downY;
        const d1 = this.s1.y - this.s1.downY;
        const parallel = d0 * d1 > 0;
        if (fingersLevel && parallel && Math.abs(dmy) > 2 * Math.abs(dmx) && Math.abs(dDist) < 0.5 * Math.abs(dmy)) {
          this.beginGesture('tilt');
          this.gesture = 'tilt2';
        } else {
          this.beginGesture('zoom');
          this.gesture = 'pinch';
        }
        // Apply the movement since the second finger went down.
        const nd = this.tfDist;
        const na = this.tfAngle;
        const nx = this.tfMidX;
        const ny = this.tfMidY;
        this.tfDist = this.tfStartDist;
        this.tfAngle = this.tfStartAngle;
        this.tfMidX = this.tfStartMidX;
        this.tfMidY = this.tfStartMidY;
        this.applyTwoFinger(nd, na, nx, ny);
        return;
      }
      case 'pinch':
      case 'tilt2': {
        const pd = this.tfDist;
        const pa = this.tfAngle;
        const px = this.tfMidX;
        const py = this.tfMidY;
        this.measureTwoFinger();
        const nd = this.tfDist;
        const na = this.tfAngle;
        const nx = this.tfMidX;
        const ny = this.tfMidY;
        this.tfDist = pd;
        this.tfAngle = pa;
        this.tfMidX = px;
        this.tfMidY = py;
        this.applyTwoFinger(nd, na, nx, ny);
        return;
      }
    }
  };

  /** Applies a two-finger step from the stored (tfDist, tfAngle, tfMid) to the new values, then stores them. */
  private applyTwoFinger(dist: number, angle: number, midX: number, midY: number): void {
    this.cancelGlobeFlight();
    const tgt = this._mode === 'globe' ? this.globeTgt : this.trackTgt;
    this.zoomAnchor = false;
    if (this.gesture === 'tilt2') {
      tgt.pitch -= ((midY - this.tfMidY) / this.rectH) * ORBIT_PITCH_PER_HEIGHT;
      this.tauPitch = TAU_ORBIT_DRAG;
    } else {
      const factor = this.tfDist / dist;
      const dAngle = wrapAngle(angle - this.tfAngle);
      if (this._mode === 'globe') {
        // pan with the midpoint, zoom about it, twist around the look point
        this.refreshLens();
        if (midX !== this.tfMidX || midY !== this.tfMidY) {
          applyGrab(tgt, this.ell, this.lens, this.ndcX(this.tfMidX), this.ndcY(this.tfMidY), this.ndcX(midX), this.ndcY(midY), this.grabRho, this.tmpV);
        }
        if (factor !== 1) {
          applyZoomAt(tgt, this.ell, this.lens, this.ndcX(midX), this.ndcY(midY), factor, minRangeForNeed(this.needFor(tgt)), this.maxRange);
        }
      } else if (factor !== 1) {
        tgt.range *= factor;
      }
      if (dAngle !== 0) addHeading(tgt.q, -dAngle);
      this.tauRot = TAU_GRAB;
      this.tauRange = TAU_GRAB;
      this.tauPitch = TAU_ORBIT_DRAG;
    }
    this.tfDist = dist;
    this.tfAngle = angle;
    this.tfMidX = midX;
    this.tfMidY = midY;
  }

  /** Single-pointer drag step from (lastX, lastY) to (x, y). */
  private dragTo(x: number, y: number, t: number): void {
    this.cancelGlobeFlight();
    const dtEv = Math.max(4, t - this.lastMoveT) / 1000;
    const a = 1 - Math.exp(-dtEv / TAU_VELOCITY);
    if (this.gesture === 'grab' && this._mode === 'globe') {
      this.refreshLens();
      const rv = this.tmpV;
      applyGrab(this.globeTgt, this.ell, this.lens, this.ndcX(this.lastX), this.ndcY(this.lastY), this.ndcX(x), this.ndcY(y), this.grabRho, rv);
      const v = this.grabVel;
      setVec(v, v.x + (rv.x / dtEv - v.x) * a, v.y + (rv.y / dtEv - v.y) * a, v.z + (rv.z / dtEv - v.z) * a);
      this.tauRot = TAU_GRAB;
    } else {
      const dH = ((x - this.lastX) / this.rectW) * ORBIT_HEADING_PER_WIDTH;
      const dP = -((y - this.lastY) / this.rectH) * ORBIT_PITCH_PER_HEIGHT;
      const tgt = this._mode === 'globe' ? this.globeTgt : this.trackTgt;
      addHeading(tgt.q, dH);
      tgt.pitch += dP;
      this.headingVel += (dH / dtEv - this.headingVel) * a;
      this.pitchVel += (dP / dtEv - this.pitchVel) * a;
      this.tauRot = TAU_ORBIT_DRAG;
      this.tauPitch = TAU_ORBIT_DRAG;
    }
    this.lastX = x;
    this.lastY = y;
    this.lastMoveT = t;
  }

  private readonly onPointerUp = (e: PointerEvent): void => {
    this.pointerEnded(e, true);
  };

  private readonly onPointerCancel = (e: PointerEvent): void => {
    this.pointerEnded(e, false);
  };

  private pointerEnded(e: PointerEvent, keepInertia: boolean): void {
    if (this.destroyed) return;
    const s = this.slotFor(e.pointerId);
    if (!s) return;
    if (this.s1.id >= 0) {
      // Two -> one finger: continue with the remaining finger, no jump.
      if (s === this.s0) {
        this.s0.id = this.s1.id;
        this.s0.touch = this.s1.touch;
        this.s0.x = this.s1.x;
        this.s0.y = this.s1.y;
        this.s0.downX = this.s1.downX;
        this.s0.downY = this.s1.downY;
      }
      this.s1.id = -1;
      this.stopInertia();
      if (this.gesture === 'two-pending' && !this.moved) {
        this.gesture = 'pending';
        this.s0.downX = this.s0.x;
        this.s0.downY = this.s0.y;
        return;
      }
      this.lastX = this.s0.x;
      this.lastY = this.s0.y;
      this.lastMoveT = e.timeStamp;
      if (this._mode === 'globe' && this.pendingKind === 'grab') {
        this.gesture = 'grab';
        this.computeGrabRho(this.lastX, this.lastY);
      } else {
        this.gesture = 'orbit';
      }
      return;
    }
    // Last pointer up.
    if (keepInertia && (this.gesture === 'grab' || this.gesture === 'orbit')) {
      const idle = e.timeStamp - this.lastMoveT;
      if (idle > VELOCITY_STALE_MS) this.stopInertia();
      else {
        const k = decay(idle / 1000, TAU_VELOCITY);
        scale(this.grabVel, k, this.grabVel);
        this.headingVel *= k;
        this.pitchVel *= k;
      }
      // A grab leaves the global rotation inertia only; an orbit only heading/pitch.
      if (this.gesture === 'grab') {
        this.headingVel = 0;
        this.pitchVel = 0;
      } else {
        setVec(this.grabVel, 0, 0, 0);
      }
    } else {
      this.stopInertia();
    }
    this.endPointerGesture(true, e.pointerId);
  }

  private endPointerGesture(releaseCapture: boolean, pointerId?: number): void {
    if (releaseCapture && pointerId !== undefined) {
      try {
        if (this.canvas.hasPointerCapture(pointerId)) this.canvas.releasePointerCapture(pointerId);
      } catch {
        /* ignore */
      }
    }
    this.s0.id = -1;
    this.s1.id = -1;
    this.gesture = 'none';
    this.setCursor(null);
  }

  private readonly onWheel = (e: WheelEvent): void => {
    if (this.destroyed) return;
    e.preventDefault();
    let dy = e.deltaY;
    if (e.deltaMode === 1) dy *= 33;
    else if (e.deltaMode === 2) dy *= 800;
    if (dy === 0) return;
    // ctrl+wheel = trackpad pinch: small deltas, stronger gain.
    const gain = e.ctrlKey ? 0.01 : 0.0022;
    const k = Math.max(-0.7, Math.min(0.7, dy * gain));
    if (e.timeStamp - this.lastWheelT > WHEEL_GESTURE_GAP_MS) {
      this.refreshRect();
      this.beginGesture('zoom');
    }
    this.lastWheelT = e.timeStamp;
    this.zoomAtCss(this.localX(e), this.localY(e), Math.exp(k), TAU_WHEEL);
  };

  private readonly onDblClick = (e: MouseEvent): void => {
    // defaultPrevented: the double-click focused a satellite (SelectionController), it is not a zoom.
    if (this.destroyed || e.button !== 0 || e.defaultPrevented) return;
    this.refreshRect();
    this.beginGesture('zoom');
    this.stopInertia();
    this.zoomAtCss(this.localX(e), this.localY(e), 1 / 2.5, TAU_DBLCLICK);
  };

  private readonly onContextMenu = (e: MouseEvent): void => {
    e.preventDefault();
  };

  private zoomAtCss(x: number, y: number, factor: number, tau: number): void {
    this.cancelGlobeFlight();
    if (this._mode === 'track') {
      this.trackTgt.range = Math.min(TRACK_MAX_RANGE_M, Math.max(TRACK_MIN_RANGE_M, this.trackTgt.range * factor));
      this.tauRange = tau;
      return;
    }
    this.refreshLens();
    const tgt = this.globeTgt;
    const nx = this.ndcX(x);
    const ny = this.ndcY(y);
    this.tauRange = tau;
    // Lock the point that is under the cursor NOW (on the displayed state):
    // every frame of the eased zoom re-aligns it under the cursor.
    if (!this.flightActive && globePointUnderCursor(this.globeCur, this.ell, this.lens, nx, ny, this.zoomPoint)) {
      this.zoomAnchor = true;
      this.zoomNx = nx;
      this.zoomNy = ny;
      this.zoomLens.tanX = this.lens.tanX;
      this.zoomLens.tanY = this.lens.tanY;
      tgt.range = Math.min(this.maxRange, Math.max(minRangeForNeed(this.needFor(tgt)), tgt.range * factor));
      return;
    }
    // Cursor in space: zoom about the target (and towards the cursor if it hits later).
    this.zoomAnchor = false;
    applyZoomAt(tgt, this.ell, this.lens, nx, ny, factor, minRangeForNeed(this.needFor(tgt)), this.maxRange);
    this.tauRot = tau;
  }

  // -------------------------------------------------------------------------
  // Keyboard
  // -------------------------------------------------------------------------

  private readonly onKeyDown = (e: KeyboardEvent): void => {
    if (this.destroyed || e.ctrlKey || e.metaKey || e.altKey || isTypingTarget(e.target)) return;
    if (e.key === 'n' || e.key === 'N') {
      if (!e.repeat) this.northUp();
      return;
    }
    const bit = keyBit(e);
    if (bit === 0) return;
    // Arrows in a scrolling HUD list (info panel, legend) scroll it; other camera keys stay global.
    if ((bit & (K_UP | K_DOWN | K_LEFT | K_RIGHT)) !== 0 && isScrollerTarget(e.target)) return;
    e.preventDefault();
    if ((this.keyMask & bit) === 0) {
      if (!e.repeat) this.beginGesture(keyKind(bit));
      this.keyMask |= bit;
      this.keyCodes[e.code] = bit;
      this.stopInertia();
      this.tauRot = TAU_KEYS;
      this.tauRange = TAU_KEYS;
      this.tauPitch = TAU_KEYS;
    }
  };

  private readonly onKeyUp = (e: KeyboardEvent): void => {
    const bit = this.keyCodes[e.code] ?? keyBit(e);
    if (bit) this.keyMask &= ~bit;
    delete this.keyCodes[e.code];
  };

  private readonly onBlur = (): void => {
    this.keyMask = 0;
    for (const k of Object.keys(this.keyCodes)) delete this.keyCodes[k];
    // A pointer gesture cannot survive losing the window (its pointerup may never arrive).
    if (this.gesture !== 'none') {
      this.stopInertia();
      this.endPointerGesture(true, this.s0.id >= 0 ? this.s0.id : undefined);
    }
  };

  /** Smoothly turns a globe view north-up (no-op over the exact pole or in track mode). */
  private northUp(): void {
    if (this._mode !== 'globe') return;
    this.beginGesture('tilt');
    const h = headingFromNorth(this.globeTgt.q);
    if (!Number.isFinite(h)) return;
    addHeading(this.globeTgt.q, -h);
    this.tauRot = TAU_NORTH;
  }
}

function wrapAngle(a: number): number {
  return a - 2 * Math.PI * Math.round(a / (2 * Math.PI));
}
