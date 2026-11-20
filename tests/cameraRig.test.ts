/**
 * Camera rig tests: the pure math in cameraMath.ts (pole crossing, grab,
 * zoom-to-cursor, smoothing, limits) and the CameraRig class driven through a
 * fake Viewer (no WebGL, no Cesium runtime) on a fake performance.now().
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import type { Viewer } from 'cesium';
import {
  DEG,
  HALF_PI,
  WGS84,
  addHeading,
  alignPointToCursor,
  anchorFromPose,
  angleBetween,
  applyGrab,
  applyZoomAt,
  constrainGlobe,
  constrainTrack,
  copyRig,
  copyVec,
  distance,
  dot,
  frameAtLatLon,
  frameAxes,
  geodetic,
  globeLookPoint,
  globePose,
  headingFromNorth,
  len,
  normalFromLatLon,
  normalize,
  pitchLimitForRange,
  pixelRay,
  pose,
  quat,
  quatFromAxisAngle,
  rayEllipsoid,
  raySphereOrClosest,
  rigState,
  rotateFrameWorld,
  smoothQuat,
  smoothRange,
  smoothScalar,
  surfaceFromNormal,
  toGeodetic,
  vec3,
} from '../src/render/cameraMath.ts';
import type { Ellipsoid, Lens, Pose, RigState, Vec3 } from '../src/render/cameraMath.ts';
import { CameraRig } from '../src/render/CameraRig.ts';

const SPHERE: Ellipsoid = { a: 6_371_000, b: 6_371_000 };
const SQUARE_LENS: Lens = { tanX: Math.tan(30 * DEG), tanY: Math.tan(30 * DEG) };

function state(lat: number, lon: number, heading: number, range: number, pitch: number): RigState {
  const s = rigState();
  frameAtLatLon(lat * DEG, lon * DEG, heading * DEG, s.q);
  s.range = range;
  s.pitch = pitch * DEG;
  s.h = 0;
  return s;
}

function hitEllipsoid(p: Pose, lens: Lens, nx: number, ny: number, e: Ellipsoid, out: Vec3): boolean {
  const ray = pixelRay(p, lens, nx, ny, vec3());
  const t = rayEllipsoid(p.pos, ray, e.a, e.b);
  if (!(t > 0)) return false;
  out.x = p.pos.x + ray.x * t;
  out.y = p.pos.y + ray.y * t;
  out.z = p.pos.z + ray.z * t;
  return true;
}

// ---------------------------------------------------------------------------
// Pure math
// ---------------------------------------------------------------------------

test('dragging straight across the north pole keeps the view continuous (no flip, no spin)', () => {
  for (const e of [SPHERE, WGS84]) {
    const s = state(80, 0, 0, 2_000_000, -90);
    const p = pose();
    const rot = vec3();
    let prevUp = vec3();
    let prevDir = vec3();
    let prevLook = vec3();
    globePose(s, e, p);
    copyVec(p.up, prevUp);
    copyVec(p.dir, prevDir);
    globeLookPoint(s, e, prevLook);
    const startN = toGeodetic(prevLook, e, geodetic());
    let maxUpStep = 0;
    let minAlign = 1;
    let crossed = false;
    for (let i = 0; i < 20; i++) {
      // pull the content from above the centre to below it -> camera moves "up" the screen
      applyGrab(s, e, SQUARE_LENS, 0, 0.1, 0, -0.1, len(prevLook), rot);
      globePose(s, e, p);
      const look = globeLookPoint(s, e, vec3());
      const motion = normalize({ x: look.x - prevLook.x, y: look.y - prevLook.y, z: look.z - prevLook.z }, vec3());
      // screen-up stays aligned with the direction of travel the whole way
      minAlign = Math.min(minAlign, dot(motion, p.up));
      maxUpStep = Math.max(maxUpStep, angleBetween(prevUp, p.up), angleBetween(prevDir, p.dir));
      if (toGeodetic(look, e, geodetic()).lat < toGeodetic(prevLook, e, geodetic()).lat) crossed = true;
      copyVec(p.up, prevUp);
      copyVec(p.dir, prevDir);
      prevLook = look;
    }
    const end = toGeodetic(prevLook, e, geodetic());
    assert.ok(crossed, 'the look target went over the pole');
    assert.ok(Math.abs(Math.abs(end.lon - startN.lon) - Math.PI) < 1e-6, `ended on the far meridian (lon ${end.lon / DEG})`);
    assert.ok(end.lat < 75 * DEG, `travelled well past the pole (lat ${end.lat / DEG})`);
    assert.ok(minAlign > 0.999, `screen-up follows the travel direction (min cos ${minAlign})`);
    assert.ok(maxUpStep < 4 * DEG, `largest per-step orientation change ${maxUpStep / DEG}°`);
    // Heading relative to north is now ~180°: that is the geometry, reached continuously.
    assert.ok(Math.abs(Math.abs(headingFromNorth(s.q)) - Math.PI) < 1e-6);
  }
});

test('grab keeps the grabbed surface point under the cursor ray (sphere and WGS-84, tilted)', () => {
  for (const e of [SPHERE, WGS84]) {
    const s = state(30, 10, 25, 5_000_000, -55);
    const p = globePose(s, e, pose());
    const a = { x: 0.2, y: -0.1 };
    const b = { x: -0.1, y: 0.25 };
    const g = vec3();
    assert.ok(hitEllipsoid(p, SQUARE_LENS, a.x, a.y, e, g));
    const rho = len(g);
    applyGrab(s, e, SQUARE_LENS, a.x, a.y, b.x, b.y, rho, vec3());
    const p2 = globePose(s, e, pose());
    const hit = vec3();
    assert.ok(raySphereOrClosest(p2.pos, pixelRay(p2, SQUARE_LENS, b.x, b.y, vec3()), rho, hit));
    assert.ok(distance(hit, g) < 1, `grabbed point off the cursor by ${distance(hit, g)} m`);
  }
});

test('grab near the horizon is limited instead of flinging the globe', () => {
  const s = state(45, 0, 0, 30_000, -3);
  const before = rigState();
  copyRig(s, before);
  const rot = vec3();
  applyGrab(s, WGS84, SQUARE_LENS, 0, -0.02, 0, -0.06, WGS84.a, rot);
  const flat = (0.04 * SQUARE_LENS.tanY * 30_000) / WGS84.a;
  assert.ok(len(rot) <= 4 * flat + 1e-12, `rotation ${len(rot)} rad vs flat ${flat}`);
});

test('zoom-to-cursor converges to the point under the cursor', () => {
  const e = WGS84;
  const tgt = state(50, 5, 40, 3_000_000, -70);
  const cur = rigState();
  copyRig(tgt, cur);
  const cursor = { x: 0.4, y: -0.3 };
  const h = vec3();
  assert.ok(hitEllipsoid(globePose(tgt, e, pose()), SQUARE_LENS, cursor.x, cursor.y, e, h));
  for (let i = 0; i < 3; i++) applyZoomAt(tgt, e, SQUARE_LENS, cursor.x, cursor.y, 0.5, 1, 1e9);
  assert.ok(Math.abs(tgt.range - 375_000) < 1e-6);
  // The target state already has the point under the cursor...
  const ht = vec3();
  assert.ok(hitEllipsoid(globePose(tgt, e, pose()), SQUARE_LENS, cursor.x, cursor.y, e, ht));
  assert.ok(distance(ht, h) < 1, `target off by ${distance(ht, h)} m`);
  // ...and the smoothed state converges onto it.
  const tau = 0.15;
  let maxDrift = 0;
  for (let i = 0; i < 180; i++) {
    smoothQuat(cur.q, tgt.q, 1 / 60, tau, cur.q);
    cur.range = smoothRange(cur.range, tgt.range, 1 / 60, tau);
    const hc = vec3();
    if (hitEllipsoid(globePose(cur, e, pose()), SQUARE_LENS, cursor.x, cursor.y, e, hc)) {
      maxDrift = Math.max(maxDrift, distance(hc, h) / cur.range);
    }
  }
  const hc = vec3();
  assert.ok(hitEllipsoid(globePose(cur, e, pose()), SQUARE_LENS, cursor.x, cursor.y, e, hc));
  assert.ok(distance(hc, h) < 1, `converged point off by ${distance(hc, h)} m`);
  // Independent q/range easing alone would let the point wander while easing...
  assert.ok(maxDrift > 0.01);
});

test('zoom-to-cursor with a locked anchor keeps the point under the cursor on EVERY frame', () => {
  const e = WGS84;
  const tgt = state(50, 5, 40, 3_000_000, -70);
  const cur = rigState();
  copyRig(tgt, cur);
  const cursor = { x: 0.4, y: -0.3 };
  const h = vec3();
  assert.ok(hitEllipsoid(globePose(cur, e, pose()), SQUARE_LENS, cursor.x, cursor.y, e, h));
  tgt.range = 375_000;
  const rot = quat();
  let maxDrift = 0;
  for (let i = 0; i < 180; i++) {
    smoothQuat(cur.q, tgt.q, 1 / 60, 0.15, cur.q);
    cur.range = smoothRange(cur.range, tgt.range, 1 / 60, 0.15);
    assert.ok(alignPointToCursor(cur, e, SQUARE_LENS, cursor.x, cursor.y, h, rot));
    rotateFrameWorld(tgt.q, rot);
    const hc = vec3();
    assert.ok(hitEllipsoid(globePose(cur, e, pose()), SQUARE_LENS, cursor.x, cursor.y, e, hc));
    maxDrift = Math.max(maxDrift, distance(hc, h));
  }
  assert.ok(maxDrift < 0.01, `point drifted ${maxDrift} m while easing`);
  assert.ok(Math.abs(cur.range / 375_000 - 1) < 1e-4);
});

test('zoom-to-cursor respects the range clamp', () => {
  const s = state(10, 10, 0, 2_000, -90);
  const f = applyZoomAt(s, WGS84, SQUARE_LENS, 0.3, 0.3, 0.01, 500, 1e9);
  assert.equal(s.range, 500);
  assert.ok(Math.abs(f - 0.25) < 1e-12);
  const s2 = state(10, 10, 0, 1e8, -90);
  applyZoomAt(s2, WGS84, SQUARE_LENS, 0, 0, 10, 1, 1.5e8);
  assert.equal(s2.range, 1.5e8);
});

test('smoothing is frame-rate independent (2 steps at 60 fps == 1 step at 30 fps)', () => {
  const tau = 0.12;
  const a1 = smoothScalar(smoothScalar(0, 1, 1 / 60, tau), 1, 1 / 60, tau);
  const a2 = smoothScalar(0, 1, 1 / 30, tau);
  assert.ok(Math.abs(a1 - a2) < 1e-12);

  const r1 = smoothRange(smoothRange(1e7, 1e3, 1 / 60, tau), 1e3, 1 / 60, tau);
  const r2 = smoothRange(1e7, 1e3, 1 / 30, tau);
  assert.ok(Math.abs(r1 / r2 - 1) < 1e-12);

  const q0 = quat();
  const qt = quatFromAxisAngle(0.3, -1, 0.5, 2.1, quat());
  const qa = smoothQuat(smoothQuat(q0, qt, 1 / 60, tau, quat()), qt, 1 / 60, tau, quat());
  const qb = smoothQuat(q0, qt, 1 / 30, tau, quat());
  const qdot = Math.abs(qa.x * qb.x + qa.y * qb.y + qa.z * qb.z + qa.w * qb.w);
  assert.ok(1 - qdot < 1e-12, `quaternion mismatch ${1 - qdot}`);
});

test('range clamps and pitch limits', () => {
  // min altitude: never closer than `need`, pitch forced down when close
  const s = state(0, 0, 0, 100, -10);
  constrainGlobe(s, 250, 1.5e8);
  assert.equal(s.range, 250);
  assert.ok(Math.abs(s.pitch + HALF_PI) < 1e-12);
  const s2 = state(0, 0, 0, 1_000, -1);
  constrainGlobe(s2, 250, 1.5e8);
  assert.ok(s2.range * Math.sin(-s2.pitch) >= 250 - 1e-6, 'camera stays >= need above the look point');
  const s3 = state(0, 0, 0, 1e12, -90);
  constrainGlobe(s3, 250, 1.5e8);
  assert.equal(s3.range, 1.5e8);

  // pitch limit: near-vertical far out, horizon allowed close in, never above the horizon
  assert.ok(pitchLimitForRange(16_000_000) <= -79 * DEG);
  assert.ok(pitchLimitForRange(1_500_000) >= -20 * DEG);
  assert.ok(pitchLimitForRange(5_000) > -2 * DEG && pitchLimitForRange(5_000) < 0);
  let prev = pitchLimitForRange(1);
  for (let r = 10; r < 2e8; r *= 1.3) {
    const p = pitchLimitForRange(r);
    assert.ok(p <= prev + 1e-12 && p < 0 && p >= -HALF_PI, `monotone at ${r}`);
    prev = p;
  }
  const far = state(0, 0, 0, 16_000_000, -10);
  constrainGlobe(far, 250, 1.5e8);
  assert.ok(far.pitch <= -79 * DEG);

  // track: range limits and the camera never goes below the ground
  const t = state(0, 0, 0, 5, 70);
  constrainTrack(t, 6_928_000, 6_378_387, 30, 2e7);
  assert.equal(t.range, 30);
  const t2 = state(0, 0, 0, 3_000_000, 60);
  constrainTrack(t2, 6_928_000, 6_378_387, 30, 2e7);
  const camR2 = 6_928_000 ** 2 - 2 * 3_000_000 * 6_928_000 * Math.sin(t2.pitch) + 3_000_000 ** 2;
  assert.ok(Math.sqrt(camR2) >= 6_378_387 - 1e-3);
});

test('heading helpers: north reference and sign convention', () => {
  const q = frameAtLatLon(40 * DEG, -70 * DEG, 30 * DEG, quat());
  assert.ok(Math.abs(headingFromNorth(q) - 30 * DEG) < 1e-9);
  addHeading(q, 10 * DEG);
  assert.ok(Math.abs(headingFromNorth(q) - 40 * DEG) < 1e-9);
  // forward at heading 90° points east
  const f = vec3();
  frameAxes(frameAtLatLon(0, 0, 90 * DEG, quat()), null, f, null);
  assert.ok(Math.abs(f.y - 1) < 1e-12);
});

test('geodetic conversion round-trips, including at the poles', () => {
  const n = vec3();
  const p = vec3();
  const g = geodetic();
  for (const [lat, lon, h] of [
    [0, 0, 0],
    [45, 10, 550_000],
    [89.9999, 123, 300],
    [90, 0, 20_000_000],
    [-90, 0, -1_000],
    [-33, -70, 8_000],
  ] as const) {
    normalFromLatLon(lat * DEG, lon * DEG, n);
    surfaceFromNormal(n, WGS84, p);
    p.x += n.x * h;
    p.y += n.y * h;
    p.z += n.z * h;
    toGeodetic(p, WGS84, g);
    assert.ok(Math.abs(g.lat - lat * DEG) < 1e-9, `lat ${lat}`);
    assert.ok(Math.abs(g.height - h) < 1e-3, `height ${h}: ${g.height}`);
  }
});

test('anchoring a pose as a globe state reproduces the same camera', () => {
  const s = state(-20, 140, 70, 800_000, -35);
  const p = globePose(s, WGS84, pose());
  const a = rigState();
  assert.ok(anchorFromPose(p, WGS84, 1, a));
  const p2 = globePose(a, WGS84, pose());
  assert.ok(distance(p.pos, p2.pos) < 1e-3);
  assert.ok(angleBetween(p.dir, p2.dir) < 1e-9);
  assert.ok(angleBetween(p.up, p2.up) < 1e-9);
});

// ---------------------------------------------------------------------------
// CameraRig through a fake Viewer
// ---------------------------------------------------------------------------

type Fn = (e: unknown) => void;

interface FakeEnv {
  rig: CameraRig;
  viewer: { scene: Record<string, unknown> };
  camera: { position: Vec3; direction: Vec3; up: Vec3; right: Vec3 };
  ctl: { enableInputs: boolean; enableCollisionDetection: boolean };
  canvasListeners: Map<string, Set<Fn>>;
  winListeners: Map<string, Set<Fn>>;
  preUpdateCount(): number;
  /** Advance the fake clock by `ms` and run one frame. */
  frame(ms: number): void;
  run(seconds: number, fps?: number): void;
  canvas(type: string, ev: Record<string, unknown>): void;
  now(): number;
  restore(): void;
}

const W = 1600;
const H = 900;
const FAKE_LENS: Lens = { tanY: Math.tan(30 * DEG), tanX: Math.tan(30 * DEG) * (W / H) };

function fakeEnv(start: RigState, winExtra: Record<string, unknown> = {}): FakeEnv {
  let t = 1_000;
  const realNow = performance.now;
  performance.now = () => t;

  const host = () => {
    const map = new Map<string, Set<Fn>>();
    return {
      map,
      addEventListener: (type: string, fn: Fn) => {
        if (!map.has(type)) map.set(type, new Set());
        map.get(type)!.add(fn);
      },
      removeEventListener: (type: string, fn: Fn) => {
        map.get(type)?.delete(fn);
        if (map.get(type)?.size === 0) map.delete(type);
      },
    };
  };
  const win = Object.assign(host(), winExtra);
  const can = host();
  const canvasObj = {
    style: { cursor: '', touchAction: '' },
    ownerDocument: { defaultView: win },
    addEventListener: can.addEventListener,
    removeEventListener: can.removeEventListener,
    setPointerCapture: () => {},
    releasePointerCapture: () => {},
    hasPointerCapture: () => false,
    getBoundingClientRect: () => ({ left: 0, top: 0, width: W, height: H }),
  };
  const p = globePose(start, WGS84, pose());
  const camera = {
    position: { ...p.pos },
    direction: { ...p.dir },
    up: { ...p.up },
    right: { ...p.right },
    get positionWC() {
      return this.position;
    },
    get directionWC() {
      return this.direction;
    },
    get upWC() {
      return this.up;
    },
    get rightWC() {
      return this.right;
    },
    transform: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    frustum: { fovy: 60 * DEG },
    lookAtTransform: () => {},
  };
  const pre = new Set<() => void>();
  const ctl = { enableInputs: true, enableCollisionDetection: true };
  const scene = {
    canvas: canvasObj,
    camera,
    globe: { ellipsoid: { radii: { x: WGS84.a, y: WGS84.a, z: WGS84.b } }, getHeight: () => undefined },
    preUpdate: {
      addEventListener: (fn: () => void) => {
        pre.add(fn);
        return () => pre.delete(fn);
      },
    },
    screenSpaceCameraController: ctl,
    isDestroyed: () => false,
  };
  const viewer = { scene };
  const rig = new CameraRig(viewer as unknown as Viewer);
  const env: FakeEnv = {
    rig,
    viewer,
    camera,
    ctl,
    canvasListeners: can.map,
    winListeners: win.map,
    preUpdateCount: () => pre.size,
    frame(ms) {
      t += ms;
      for (const fn of pre) fn();
    },
    run(seconds, fps = 60) {
      const n = Math.round(seconds * fps);
      for (let i = 0; i < n; i++) env.frame(1000 / fps);
    },
    canvas(type, ev) {
      const full = { timeStamp: t, preventDefault: () => {}, ctrlKey: false, metaKey: false, button: 0, ...ev };
      for (const fn of can.map.get(type) ?? []) fn(full);
    },
    now: () => t,
    restore() {
      performance.now = realNow;
    },
  };
  // first frame initialises the clock
  env.frame(0);
  return env;
}

function camPose(env: FakeEnv): Pose {
  const p = pose();
  copyVec(env.camera.position, p.pos);
  copyVec(env.camera.direction, p.dir);
  copyVec(env.camera.up, p.up);
  copyVec(env.camera.right, p.right);
  return p;
}

test('rig: takes over Cesium input, and destroy() removes every listener', () => {
  const env = fakeEnv(state(25, 5, 0, 45_000_000, -90));
  try {
    assert.equal(env.ctl.enableInputs, false);
    assert.equal(env.ctl.enableCollisionDetection, false);
    assert.ok(env.canvasListeners.size >= 6);
    assert.ok(env.winListeners.size >= 2);
    assert.equal(env.preUpdateCount(), 1);
    assert.ok(!('cameraRig' in globalThis));
    env.rig.destroy();
    assert.equal(env.canvasListeners.size, 0);
    assert.equal(env.winListeners.size, 0);
    assert.equal(env.preUpdateCount(), 0);
    assert.equal(env.ctl.enableInputs, true);
  } finally {
    env.restore();
  }
});

test('rig: flyToOverview eases to 16,000 km above 25°N 5°E and resolves', async () => {
  const env = fakeEnv(state(-10, 60, 0, 45_000_000, -90));
  try {
    let done = false;
    const pr = env.rig.flyToOverview(1).then(() => {
      done = true;
    });
    let prev = { ...env.camera.position };
    let maxStep = 0;
    for (let i = 0; i < 75; i++) {
      env.frame(1000 / 60);
      maxStep = Math.max(maxStep, distance(prev, env.camera.position));
      prev = { ...env.camera.position };
    }
    await pr;
    assert.ok(done);
    const g = toGeodetic(env.camera.position, WGS84, geodetic());
    assert.ok(Math.abs(g.height - 16_000_000) < 1, `height ${g.height}`);
    assert.ok(Math.abs(g.lat - 25 * DEG) < 1e-6 && Math.abs(g.lon - 5 * DEG) < 1e-6);
    // no frame jumps more than a few % of the distance (eased, continuous)
    assert.ok(maxStep < 2_500_000, `max per-frame step ${maxStep} m`);
  } finally {
    env.rig.destroy();
    env.restore();
  }
});

test('rig: track follows a moving target with zero lag, release flies back out', () => {
  const env = fakeEnv(state(25, 5, 0, 16_000_000, -90));
  try {
    // circular 550 km orbit, 7.6 km/s, inclined 53°
    const r = WGS84.a + 550_000;
    const w = 7_600 / r;
    const inc = 53 * DEG;
    let alive = true;
    const getPos = (out: { x: number; y: number; z: number }): boolean => {
      if (!alive) return false;
      const a = (env.now() / 1000) * w;
      out.x = r * Math.cos(a);
      out.y = r * Math.sin(a) * Math.cos(inc);
      out.z = r * Math.sin(a) * Math.sin(inc);
      return true;
    };
    env.rig.track(getPos, { rangeM: 50_000, durationS: 1.5 });
    assert.equal(env.rig.mode, 'track');
    env.run(2.5);
    const s = vec3();
    for (let i = 0; i < 30; i++) {
      env.frame(1000 / 60);
      getPos(s);
      const p = camPose(env);
      assert.ok(Math.abs(distance(p.pos, s) - 50_000) < 1e-3, 'camera at the requested range of the CURRENT position');
      const toS = normalize({ x: s.x - p.pos.x, y: s.y - p.pos.y, z: s.z - p.pos.z }, vec3());
      assert.ok(1 - dot(toS, p.dir) < 1e-12, 'looking straight at the target');
    }
    // wheel zoom in track mode changes the range
    env.canvas('wheel', { deltaY: 300, deltaMode: 0, clientX: W / 2, clientY: H / 2 });
    env.run(1.5);
    getPos(s);
    assert.ok(distance(camPose(env).pos, s) > 60_000);

    // target disappears -> automatic release to a globe view
    alive = false;
    env.frame(1000 / 60);
    assert.equal(env.rig.mode, 'globe');
    env.run(3);
    const g = toGeodetic(env.camera.position, WGS84, geodetic());
    assert.ok(Math.abs(g.height - 6_000_000) < 10, `height after release ${g.height}`);
  } finally {
    env.rig.destroy();
    env.restore();
  }
});

test('rig: mouse drag grabs the globe, wheel zooms, onUserInput fires per gesture', () => {
  const env = fakeEnv(state(40, 0, 0, 8_000_000, -90));
  try {
    const kinds: string[] = [];
    env.rig.onUserInput = (k) => kinds.push(k);
    const ndc = (x: number, y: number) => ({ x: (x / W) * 2 - 1, y: 1 - (y / H) * 2 });
    const a = ndc(800, 450);
    const g = vec3();
    assert.ok(hitEllipsoid(camPose(env), FAKE_LENS, a.x, a.y, WGS84, g));

    env.canvas('pointerdown', { pointerId: 1, pointerType: 'mouse', clientX: 800, clientY: 450 });
    for (let i = 1; i <= 10; i++) {
      env.frame(16);
      env.canvas('pointermove', { pointerId: 1, pointerType: 'mouse', clientX: 800 + i * 20, clientY: 450 - i * 10 });
    }
    env.frame(200); // pause before release -> no inertia
    env.canvas('pointerup', { pointerId: 1, pointerType: 'mouse', clientX: 1000, clientY: 350 });
    assert.equal(env.rig.lastPressDragged, true);
    env.run(1);
    const b = ndc(1000, 350);
    const hit = vec3();
    assert.ok(hitEllipsoid(camPose(env), FAKE_LENS, b.x, b.y, WGS84, hit));
    assert.ok(distance(hit, g) < 50, `grabbed point is ${distance(hit, g)} m from the cursor`);

    // wheel: zoom towards the cursor, exponential, smooth
    const before = distance(env.camera.position, hit);
    env.canvas('wheel', { deltaY: -100, deltaMode: 0, clientX: 1000, clientY: 350 });
    env.run(1.5);
    const hit2 = vec3();
    assert.ok(hitEllipsoid(camPose(env), FAKE_LENS, b.x, b.y, WGS84, hit2));
    assert.ok(distance(hit2, g) < 50, 'point under the cursor stays put while zooming');
    assert.ok(distance(env.camera.position, hit) < before * 0.85);
    assert.deepEqual(kinds, ['drag', 'zoom']);
  } finally {
    env.rig.destroy();
    env.restore();
  }
});

test('rig: the eased result does not depend on the frame rate', () => {
  const run = (fps: number): Vec3 => {
    const env = fakeEnv(state(10, 20, 0, 5_000_000, -90));
    try {
      env.canvas('wheel', { deltaY: -200, deltaMode: 0, clientX: 1200, clientY: 300 });
      env.run(0.5, fps);
      return { ...env.camera.position };
    } finally {
      env.rig.destroy();
      env.restore();
    }
  };
  const a = run(30);
  const b = run(60);
  const c = run(144);
  const range = 5_000_000;
  // (1e-5 x range is ~0.01 px on a 1600 px canvas: the cursor-anchored zoom composes
  // minimal rotations per frame, so the twist about the anchor differs by a hair between frame rates)
  assert.ok(distance(a, b) / range < 1e-5, `30 vs 60 fps differ by ${distance(a, b)} m`);
  assert.ok(distance(b, c) / range < 1e-5, `60 vs 144 fps differ by ${distance(b, c)} m`);
});

test('rig: user input during the overview flight cancels it in place (no jump) and resolves', async () => {
  const env = fakeEnv(state(-10, 60, 0, 45_000_000, -90));
  try {
    let done = false;
    const pr = env.rig.flyToOverview(3).then(() => {
      done = true;
    });
    env.run(1);
    const before = { ...env.camera.position };
    env.canvas('pointerdown', { pointerId: 7, pointerType: 'mouse', clientX: 800, clientY: 450 });
    env.canvas('pointermove', { pointerId: 7, pointerType: 'mouse', clientX: 801, clientY: 450 }); // below threshold
    env.canvas('pointermove', { pointerId: 7, pointerType: 'mouse', clientX: 806, clientY: 450 });
    await pr;
    assert.ok(done);
    assert.equal(env.rig.flying, false);
    env.frame(1000 / 60);
    // camera continues from where it was (a 5 px grab moves it only slightly)
    assert.ok(distance(before, env.camera.position) < 0.02 * len(before), `jumped ${distance(before, env.camera.position)} m`);
    env.canvas('pointerup', { pointerId: 7, pointerType: 'mouse', clientX: 806, clientY: 450 });
  } finally {
    env.rig.destroy();
    env.restore();
  }
});

test('rig: keyboard pans and zooms smoothly, ignored while typing', () => {
  const env = fakeEnv(state(0, 0, 0, 4_000_000, -90));
  try {
    const win = env.winListeners;
    const key = (type: string, k: string, code: string, target: unknown = null) => {
      for (const fn of win.get(type) ?? []) fn({ key: k, code, repeat: false, target, ctrlKey: false, metaKey: false, altKey: false, preventDefault: () => {} });
    };
    key('keydown', 'ArrowUp', 'ArrowUp', { tagName: 'INPUT' });
    env.run(0.5);
    let g = toGeodetic(env.camera.position, WGS84, geodetic());
    assert.ok(Math.abs(g.lat) < 1e-9, 'typing in an input does not move the camera');
    key('keydown', 'ArrowUp', 'ArrowUp');
    env.run(0.5);
    key('keyup', 'ArrowUp', 'ArrowUp');
    env.run(1);
    g = toGeodetic(env.camera.position, WGS84, geodetic());
    assert.ok(g.lat > 5 * DEG, `panned north (lat ${g.lat / DEG})`);
    const h0 = g.height;
    key('keydown', '+', 'Equal');
    env.run(0.5);
    key('keyup', '+', 'Equal');
    env.run(1);
    assert.ok(toGeodetic(env.camera.position, WGS84, geodetic()).height < h0 * 0.7);
  } finally {
    env.rig.destroy();
    env.restore();
  }
});

/** Circular 550 km orbit (53° inclination) at phase offset `phase` (rad), on the fake clock. */
function orbitAt(env: FakeEnv, phase: number, out: { x: number; y: number; z: number }): boolean {
  const r = WGS84.a + 550_000;
  const a = (env.now() / 1000) * (7_600 / r) + phase;
  const inc = 53 * DEG;
  out.x = r * Math.cos(a);
  out.y = r * Math.sin(a) * Math.cos(inc);
  out.z = r * Math.sin(a) * Math.sin(inc);
  return true;
}

/** Largest per-frame camera move over `frames` 60 fps frames. */
function maxStep(env: FakeEnv, frames: number): number {
  let prev = { ...env.camera.position };
  let m = 0;
  for (let i = 0; i < frames; i++) {
    env.frame(1000 / 60);
    m = Math.max(m, distance(prev, env.camera.position));
    prev = { ...env.camera.position };
  }
  return m;
}

test('rig: re-targeting a finished track flies from the old satellite (no first-frame jump)', () => {
  const env = fakeEnv(state(25, 5, 0, 16_000_000, -90));
  try {
    // A getter that follows the selection (the pre-fix SelectionController pattern).
    let sel = 0;
    const phases = [0, 20 * DEG]; // ~2,400 km apart
    const follow = (out: { x: number; y: number; z: number }): boolean => orbitAt(env, phases[sel]!, out);
    env.rig.track(follow, { durationS: 1 });
    env.run(2);
    sel = 1;
    env.rig.track(follow, { durationS: 2 });
    const step = maxStep(env, 150);
    assert.ok(step < 150_000, `re-target moved ${step} m in one frame`);
    const s = vec3();
    orbitAt(env, phases[1]!, s);
    assert.ok(Math.abs(distance(camPose(env).pos, s) - 2_000_000) < 1, 'ends on the new satellite');
  } finally {
    env.rig.destroy();
    env.restore();
  }
});

test('rig: a bound getter that turns false freezes the release flight start (no jump to the new pick)', () => {
  const env = fakeEnv(state(25, 5, 0, 16_000_000, -90));
  try {
    let sel = 0;
    const phases = [0, 25 * DEG];
    // SelectionController.targetFor: bound to one satellite, false once the selection moved on.
    const bound =
      (id: number) =>
      (out: { x: number; y: number; z: number }): boolean =>
        sel === id && orbitAt(env, phases[id]!, out);
    env.rig.track(bound(0), { durationS: 1 });
    env.run(2);
    env.rig.release(2.2);
    env.run(0.5);
    sel = 1; // the user clicks another satellite during the release flight
    const step = maxStep(env, 30);
    assert.ok(step < 150_000, `release flight lurched ${step} m in one frame`);
  } finally {
    env.rig.destroy();
    env.restore();
  }
});

test('rig: interrupting a release flight with the view above the horizon eases into the globe limits', () => {
  const env = fakeEnv(state(25, 5, 0, 16_000_000, -90));
  try {
    const p0 = vec3();
    orbitAt(env, 0, p0);
    const fixed = (out: { x: number; y: number; z: number }): boolean => {
      out.x = p0.x;
      out.y = p0.y;
      out.z = p0.z;
      return true;
    };
    env.rig.track(fixed, { durationS: 0.5 });
    env.run(1);
    // raise the track pitch well above the horizon dip seen from the satellite
    const key = (type: string, k: string) => {
      for (const fn of env.winListeners.get(type) ?? []) {
        fn({ key: k, code: 'Key' + k.toUpperCase(), repeat: false, target: null, ctrlKey: false, metaKey: false, altKey: false, preventDefault: () => {} });
      }
    };
    key('keydown', 'r');
    env.run(1);
    key('keyup', 'r');
    env.run(1);
    env.rig.release(2.2);
    env.run(0.2);
    const before = { ...env.camera.position };
    env.canvas('wheel', { deltaY: -50, deltaMode: 0, clientX: W / 2, clientY: H / 2 });
    assert.equal(env.rig.mode, 'globe');
    const step = maxStep(env, 90);
    assert.ok(distance(before, env.camera.position) > 1_000, 'the camera settles somewhere new');
    assert.ok(step < 150_000, `interrupting snapped the camera by ${step} m in one frame`);
    assert.equal(env.rig.flying, false);
    // ends on a valid globe view: the view ray hits the Earth
    const p = camPose(env);
    assert.ok(rayEllipsoid(p.pos, p.dir, WGS84.a, WGS84.b) > 0, 'looking at the ground after the settle');
  } finally {
    env.rig.destroy();
    env.restore();
  }
});

test('rig: a mouse move with no button held ends a grab whose pointerup was lost', () => {
  const env = fakeEnv(state(40, 0, 0, 8_000_000, -90));
  try {
    env.canvas('pointerdown', { pointerId: 1, pointerType: 'mouse', buttons: 1, clientX: 800, clientY: 450 });
    for (let i = 1; i <= 5; i++) {
      env.frame(16);
      env.canvas('pointermove', { pointerId: 1, pointerType: 'mouse', buttons: 1, clientX: 800 + i * 20, clientY: 450 });
    }
    assert.equal(env.rig.dragging, true);
    env.run(1);
    // button released outside the window: no pointerup, the next move has buttons === 0
    env.canvas('pointermove', { pointerId: 1, pointerType: 'mouse', buttons: 0, clientX: 900, clientY: 450 });
    assert.equal(env.rig.dragging, false);
    env.run(0.5);
    const rest = { ...env.camera.position };
    for (let i = 1; i <= 5; i++) {
      env.frame(16);
      env.canvas('pointermove', { pointerId: 1, pointerType: 'mouse', buttons: 0, clientX: 900 + i * 40, clientY: 450 });
    }
    env.run(0.5);
    assert.ok(distance(rest, env.camera.position) < 1e-6, 'a bare cursor does not grab the globe');
    // the next press is accepted again
    env.canvas('pointerdown', { pointerId: 1, pointerType: 'mouse', buttons: 1, clientX: 800, clientY: 450 });
    env.canvas('pointermove', { pointerId: 1, pointerType: 'mouse', buttons: 1, clientX: 900, clientY: 450 });
    assert.equal(env.rig.dragging, true);
    // and losing the window ends it as well
    for (const fn of env.winListeners.get('blur') ?? []) fn({});
    assert.equal(env.rig.dragging, false);
  } finally {
    env.rig.destroy();
    env.restore();
  }
});

test('rig: a dblclick already handled (defaultPrevented) is not a zoom', () => {
  const env = fakeEnv(state(25, 5, 0, 16_000_000, -90));
  try {
    const target = (out: { x: number; y: number; z: number }): boolean => orbitAt(env, 0, out);
    env.rig.track(target, { durationS: 0.5 });
    env.run(1);
    const s = vec3();
    env.canvas('dblclick', { clientX: W / 2, clientY: H / 2, defaultPrevented: true });
    env.run(1.5);
    orbitAt(env, 0, s);
    assert.ok(Math.abs(distance(camPose(env).pos, s) - 2_000_000) < 1, 'a prevented dblclick keeps the track range');
    env.canvas('dblclick', { clientX: W / 2, clientY: H / 2 });
    env.run(6);
    orbitAt(env, 0, s);
    assert.ok(Math.abs(distance(camPose(env).pos, s) - 800_000) < 10, 'a plain dblclick zooms 2.5x');
  } finally {
    env.rig.destroy();
    env.restore();
  }
});

test('rig: prefers-reduced-motion turns scripted flights into a near-instant cut', async () => {
  const queries: string[] = [];
  const matchMedia = (q: string) => {
    queries.push(q);
    return { matches: q === '(prefers-reduced-motion: reduce)' };
  };
  const env = fakeEnv(state(-10, 60, 0, 45_000_000, -90), { matchMedia });
  try {
    let done = false;
    const pr = env.rig.flyToOverview(3.5).then(() => {
      done = true;
    });
    env.run(5 / 60);
    await Promise.race([pr, new Promise((r) => setTimeout(r, 0))]);
    assert.ok(done, 'resolved within a few frames');
    assert.ok(queries.includes('(prefers-reduced-motion: reduce)'));
    const g = toGeodetic(env.camera.position, WGS84, geodetic());
    assert.ok(Math.abs(g.height - 16_000_000) < 1, `height ${g.height}`);
  } finally {
    env.rig.destroy();
    env.restore();
  }
});

test('rig: arrow keys inside a HUD scroller scroll it instead of moving the camera', () => {
  const env = fakeEnv(state(0, 0, 0, 4_000_000, -90));
  try {
    let prevented = 0;
    const key = (type: string, k: string, target: unknown) => {
      for (const fn of env.winListeners.get(type) ?? []) {
        fn({ key: k, code: k, repeat: false, target, ctrlKey: false, metaKey: false, altKey: false, preventDefault: () => prevented++ });
      }
    };
    const inScroller = { tagName: 'DIV', closest: (sel: string) => (sel.includes('.sx-panel__scroll') ? {} : null) };
    const button = { tagName: 'BUTTON', closest: () => null };
    key('keydown', 'ArrowUp', inScroller);
    env.run(0.5);
    key('keyup', 'ArrowUp', inScroller);
    env.run(0.5);
    assert.equal(prevented, 0, 'the browser keeps its default arrow scrolling');
    assert.ok(Math.abs(toGeodetic(env.camera.position, WGS84, geodetic()).lat) < 1e-9, 'camera did not pan');
    // a focused HUD button still lets the arrows drive the globe
    key('keydown', 'ArrowUp', button);
    env.run(0.5);
    key('keyup', 'ArrowUp', button);
    env.run(1);
    assert.equal(prevented, 1);
    assert.ok(toGeodetic(env.camera.position, WGS84, geodetic()).lat > 5 * DEG);
  } finally {
    env.rig.destroy();
    env.restore();
  }
});
