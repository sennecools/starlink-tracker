/**
 * Application entry: wires the modules together.
 *
 *   loadElements (HTTP)  ->  PropagatorClient (SGP4 worker)  ->  SatellitePoints (GPU)
 *                                                           \->  HUD (plain values)
 *
 * Boot: the Cesium globe and the element download start concurrently; once the
 * elements arrive they go straight to the worker (which does not need the
 * globe). When both are ready the GPU primitive is created, the first three
 * keyframes are computed, and only then is the HUD revealed and the intro
 * flight started, so the constellation is there from the first visible frame
 * (the optional laser-link web is not waited for; it fades in when ready).
 *
 * Round 2 modules (each owns its own input / frame hooks):
 *   CameraRig            owns the camera every frame (Cesium's controller is off); boot intro
 *   createSky            NASA Milky Way cube + Hipparcos stars (replaces the Tycho sky box)
 *   SelectionController  hover + click picking, info panel feed, orbit ring, track/release
 *   OrbitPath            orbit ring + ground track primitives (shaders pre-warmed at boot)
 *
 * Frame loop (all allocation-free):
 *   scene.preUpdate   keyframe scheduling, clock / live pill, camera + perf readouts (throttled)
 *                     (+ CameraRig and OrbitPath hooks)
 *   scene.postRender  hover picking (SelectionController)
 *
 * Round 3 modules (3D close-up + MODELLED laser links; SpaceX does not publish
 * the real link topology, so every UI surface says MODELED):
 *   Session links        worker 'links' request at prime and then every
 *                        LINK_SIM_INTERVAL_MS of sim time (>= LINK_MIN_WALL_MS
 *                        wall apart; sooner after a clock jump, but only
 *                        once the restart keyframe is in) ->
 *                        points.setLinks (GPU web) + CSR adjacency (selection)
 *                        + HUD LASER LINKS row (toggle -> setLinksVisible)
 *   SatelliteModels      preloads + warms every .glb (v1.5, V2 Mini, V3) at globe start; shows
 *                        models for satellites within 40 km (selected forced)
 *   LaserHighlight       animated beams from the selection to its neighbours
 *
 * Round 4 (V3 + coverage):
 *   CoverageFootprint    hologram of the selected satellite's coverage area
 *                        (25° min elevation) + cone; follows the selection
 *                        (SelectionController), shown per the panel's
 *                        COVERAGE AREA toggle (default on, remembered locally)
 *   Element sources      the server merges several CelesTrak feeds (OMM); the
 *                        HUD lists them, and every merge (even while partial,
 *                        e.g. a "recent launches" supplement with new V3s) is
 *                        hot-swapped in.
 *
 * A "session" = one element set: its own worker, catalog and SatellitePoints.
 * Whenever the server's merged element set changes (a full catalogue replacing
 * a partial snapshot, or a supplement adding new launches), a new session is
 * built in the background and swapped in without a page reload.
 */
import { Cartographic, Math as CesiumMath } from 'cesium';
import type { JulianDate, Scene } from 'cesium';
import { fetchElementsStatus, loadElements } from './data/loadElements.ts';
import { CameraRig } from './render/CameraRig.ts';
import { CoverageFootprint } from './render/CoverageFootprint.ts';
import { createGlobe } from './render/createGlobe.ts';
import type { GlobeContext } from './render/createGlobe.ts';
import { LaserHighlight } from './render/LaserHighlight.ts';
import { OrbitPath } from './render/OrbitPath.ts';
import { SatelliteModels } from './render/SatelliteModels.ts';
import { KEYFRAME_STEP_MS, SatellitePoints, julianToUnixMs } from './render/SatellitePoints.ts';
import { createSky } from './render/sky/createSky.ts';
import type { SkyHandle } from './render/sky/createSky.ts';
import { buildLinkAdjacency, countLaserEquipped } from './selection/linkAdjacency.ts';
import type { LinkAdjacency } from './selection/linkAdjacency.ts';
import { SelectionController } from './selection/SelectionController.ts';
import type { ElementRecord, ElementsCycleInfo, ElementsReady, ElementsWarming, LaserLinks, SatCatalog } from './shared/contracts.ts';
import { SHELLS } from './shared/shells.ts';
import { groupThousands } from './ui/format.ts';
import { createHud } from './ui/hud.ts'; // also imports the fonts and hud.css
import type { HudCameraInfo, HudPerfInfo, HudShellInfo } from './ui/hud.ts';
import { PropagatorClient } from './worker/PropagatorClient.ts';
import type { KeyframeResult } from './worker/PropagatorClient.ts';

// ---------------------------------------------------------------------------
// Tunables
// ---------------------------------------------------------------------------

/** HUD camera readout refresh (wall ms). */
const CAMERA_UI_INTERVAL_MS = 250;
/** HUD perf readout refresh (wall ms). */
const PERF_UI_INTERVAL_MS = 500;
/** Poll the (local) element server this often while showing a partial snapshot... */
const PARTIAL_POLL_MS = 60_000;
/** ...and this often once the full catalogue is in (supplement merges, e.g. new launches). */
const FULL_POLL_MS = 5 * 60_000;
/** localStorage key of the COVERAGE AREA preference (per viewer; optional). */
const COVERAGE_PREF_KEY = 'starlink-tracker.coverage';
/** Max keyframe requests in flight per session. */
const MAX_IN_FLIGHT = 2;
/** Recycled position buffers kept for reuse. */
const POOL_MAX = 4;
/** Sim clock within this of wall time + multiplier 1 => "LIVE". */
const LIVE_TOLERANCE_MS = 2_000;
/** Recompute the (modelled) laser-link topology every this much sim time... */
const LINK_SIM_INTERVAL_MS = 10_000;
/** ...but at most this often in wall time (time warp)... */
const LINK_MIN_WALL_MS = 1_000;
/** ...except after a clock jump of more than LINK_JUMP_MS of sim time. */
const LINK_JUMP_MS = 120_000;
const LINK_JUMP_MIN_WALL_MS = 250;
/** Back-off after a failed links request (wall ms); links are optional. */
const LINK_RETRY_WALL_MS = 30_000;

const NO_TOKEN_WARNING = 'No Cesium ion token — using the shared demo token. Add VITE_CESIUM_ION_TOKEN to .env.local';

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

function errorMessage(err: unknown): string {
  if (err instanceof Error) return err.message;
  return String(err);
}

function formatBytes(bytes: number): string {
  if (!(bytes > 0)) return '0 KB';
  if (bytes < 1_048_576) return `${groupThousands(bytes / 1024)} KB`;
  return `${(bytes / 1_048_576).toFixed(1)} MB`;
}

function isoToDate(iso: string | null | undefined): Date | null {
  if (!iso) return null;
  const ms = Date.parse(iso);
  return Number.isFinite(ms) ? new Date(ms) : null;
}

/**
 * Catalog index -> index into `records`. The worker's buildCatalog drops
 * rejected element sets but keeps the input order of the accepted ones, so a
 * single forward scan matching NORAD ids recovers the mapping. -1 = unmatched
 * (should not happen).
 */
function mapCatalogToRecords(records: readonly ElementRecord[], catalog: SatCatalog): Int32Array {
  const map = new Int32Array(catalog.count).fill(-1);
  const ids = catalog.noradIds;
  let j = 0;
  for (let i = 0; i < catalog.count; i++) {
    const id = ids[i];
    let k = j;
    while (k < records.length && records[k]!.noradId !== id) k++;
    if (k >= records.length) continue; // not found: leave -1, keep scanning from j
    map[i] = k;
    j = k + 1;
  }
  return map;
}

/** COVERAGE AREA preference (default ON); storage may be unavailable. */
function readCoveragePref(): boolean {
  try {
    return window.localStorage.getItem(COVERAGE_PREF_KEY) !== '0';
  } catch {
    return true;
  }
}

function writeCoveragePref(on: boolean): void {
  try {
    window.localStorage.setItem(COVERAGE_PREF_KEY, on ? '1' : '0');
  } catch {
    // private window / blocked storage: the preference just is not remembered
  }
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * loadElements with a couple of retries on transport errors (e.g. the dev
 * server restarting mid-boot). HTTP errors and aborts are not retried.
 */
async function loadElementsWithRetry(
  opts: { signal?: AbortSignal; onProgress?: (p: ElementsWarming) => void },
  attempts = 3,
): Promise<ElementsReady> {
  for (let i = 1; ; i++) {
    try {
      return await loadElements(opts);
    } catch (err) {
      const transport = err instanceof Error && err.message.startsWith('Cannot reach');
      if (!transport || i >= attempts || opts.signal?.aborted) throw err;
      await delay(1500 * i);
    }
  }
}

// ---------------------------------------------------------------------------
// Session: one element set -> worker + catalog + GPU primitive
// ---------------------------------------------------------------------------

class Session {
  readonly data: ElementsReady;
  readonly client: PropagatorClient;
  readonly catalog: SatCatalog;
  readonly points: SatellitePoints;
  /** Satellites per SHELLS index. */
  readonly shellCounts: number[];
  /** Catalog index -> index into data.records (rejected TLEs are not in the catalog). */
  private readonly recordIndex: Int32Array;
  /** Satellites with laser terminals (estimated from the launch, see shared/hardware.ts). */
  readonly equipped: number;
  /** Latest MODELLED laser-link topology (indices of this session's catalog) or null. */
  links: LaserLinks | null = null;
  /** Neighbour lists of `links` (rebuilt in place on every update). */
  adjacency: LinkAdjacency | null = null;
  /** Called after each new topology was applied to the points + adjacency. */
  onLinks: ((s: Session) => void) | null = null;
  /** Worker time of the latest keyframe (ms; HUD perf readout). */
  lastKeyframeMs = Number.NaN;
  disposed = false;
  failed = false;

  private readonly onFatal: (err: unknown) => void;
  private linkInFlight = false;
  /** Sim time of the last links request (NaN = none yet). */
  private linkTimeMs = Number.NaN;
  private linkLastWall = -Infinity;
  private linkRetryWall = -Infinity;
  private readonly pool: Float64Array[] = [];
  /** Sim times of the (at most two) requests in flight; NaN = free slot. */
  private req0 = Number.NaN;
  private req1 = Number.NaN;
  private inFlight = 0;

  constructor(
    data: ElementsReady,
    client: PropagatorClient,
    catalog: SatCatalog,
    shellVisible: readonly boolean[],
    onFatal: (err: unknown) => void,
    fadeIn = true,
  ) {
    this.data = data;
    this.client = client;
    this.catalog = catalog;
    this.onFatal = onFatal;
    this.points = new SatellitePoints({ catalog, fadeIn });
    this.points.onRecycle = this.recycle;
    for (let s = 0; s < shellVisible.length; s++) {
      if (!shellVisible[s]) this.points.setShellVisible(s, false);
    }
    this.shellCounts = new Array<number>(SHELLS.length).fill(0);
    const shells = catalog.shell;
    for (let i = 0; i < catalog.count; i++) {
      const s = shells[i];
      if (s < this.shellCounts.length) this.shellCounts[s]++;
    }
    this.recordIndex = mapCatalogToRecords(data.records, catalog);
    this.equipped = countLaserEquipped(catalog.lasers, catalog.count);
  }

  /** Element record of CATALOG index `i` (not the raw records index), or null. */
  recordAt(i: number): ElementRecord | null {
    if (!(i >= 0 && i < this.recordIndex.length)) return null;
    const j = this.recordIndex[i]!;
    return j >= 0 ? (this.data.records[j] ?? null) : null;
  }

  /**
   * Compute and queue the first three keyframes around `nowMs` (before the
   * primitive is shown), so the very first rendered frame already has data.
   * The first links request is queued behind them and, at boot, NOT awaited:
   * the link web is optional and fades in when it arrives (onLinksResult).
   * Three keyframes (= MAX_PENDING) cover ~3 s plus 1 s of extrapolation, so a
   * slow first links job in the serial worker cannot freeze the first visible
   * frames. A hot swap passes `awaitLinks`: the replacement then has its link
   * web and adjacency before it is shown, so the web and the selection's beams
   * carry on without a gap.
   */
  async prime(nowMs: number, awaitLinks = false): Promise<void> {
    const t0 = this.points.nextKeyframeTime(nowMs);
    if (t0 === null) return;
    const step = KEYFRAME_STEP_MS;
    const keyframes = Promise.all([
      this.client.keyframe(t0),
      this.client.keyframe(t0 + step),
      this.client.keyframe(t0 + 2 * step),
    ]);
    // Never rejects (a failure is logged, never fatal); applied when it arrives.
    const linksP = this.requestLinks(t0, performance.now());
    const kfs = await keyframes;
    if (this.disposed) return;
    for (const kf of kfs) this.points.addKeyframe(kf);
    this.lastKeyframeMs = kfs[0].computeMs;
    if (awaitLinks && !this.disposed) await linksP;
  }

  /** Per-frame: request the next keyframe (and links) when due. Allocation-free unless a request is sent. */
  schedule(nowMs: number): void {
    if (this.disposed || this.failed) return;
    this.scheduleKeyframe(nowMs);
    // Links only while the held keyframes can render `nowMs`: after a clock
    // jump (e.g. returning to a hidden tab) the topology job must not queue
    // ahead of the restart keyframes in the serial worker.
    if (this.points.coversTime(nowMs)) this.scheduleLinks(nowMs);
  }

  private scheduleKeyframe(nowMs: number): void {
    if (this.inFlight >= MAX_IN_FLIGHT) return;
    const t = this.points.nextKeyframeTime(nowMs);
    if (t === null || t === this.req0 || t === this.req1) return;
    if (Number.isNaN(this.req0)) this.req0 = t;
    else this.req1 = t;
    this.inFlight++;
    this.client.keyframe(t, this.takeBuffer()).then(this.onKeyframe, this.onKeyframeError);
  }

  private takeBuffer(): Float64Array | undefined {
    const need = this.catalog.count * 3;
    while (this.pool.length > 0) {
      const buf = this.pool.pop() as Float64Array;
      if (buf.length === need) return buf;
    }
    return undefined;
  }

  private readonly recycle = (buf: Float64Array): void => {
    if (this.disposed) return;
    if (this.pool.length < POOL_MAX && buf.length === this.catalog.count * 3) this.pool.push(buf);
  };

  private release(timeMs: number): void {
    if (timeMs === this.req0) this.req0 = Number.NaN;
    else if (timeMs === this.req1) this.req1 = Number.NaN;
    if (this.inFlight > 0) this.inFlight--;
  }

  private readonly onKeyframe = (kf: KeyframeResult): void => {
    this.release(kf.timeMs);
    if (this.disposed) return;
    this.lastKeyframeMs = kf.computeMs;
    this.points.addKeyframe(kf);
  };

  /** Per-frame: request a new link topology every LINK_SIM_INTERVAL_MS of sim time (throttled in wall time). */
  private scheduleLinks(nowMs: number): void {
    if (this.linkInFlight) return;
    const dt = Math.abs(nowMs - this.linkTimeMs);
    if (dt < LINK_SIM_INTERVAL_MS) return; // (NaN = never requested -> falls through)
    const wall = performance.now();
    if (wall < this.linkRetryWall) return;
    const minWall = dt > LINK_JUMP_MS ? LINK_JUMP_MIN_WALL_MS : LINK_MIN_WALL_MS;
    if (wall - this.linkLastWall < minWall) return;
    void this.requestLinks(nowMs, wall);
  }

  /** Resolves once the result is applied (or the request failed; never rejects). */
  private requestLinks(simMs: number, wall: number): Promise<void> {
    this.linkInFlight = true;
    this.linkTimeMs = simMs;
    this.linkLastWall = wall;
    return this.client.links(simMs).then(this.onLinksResult, this.onLinksError);
  }

  private readonly onLinksResult = (links: LaserLinks): void => {
    this.linkInFlight = false;
    if (this.disposed) return;
    this.links = links;
    this.points.setLinks(links);
    this.adjacency = buildLinkAdjacency(this.catalog.count, links.pairs, links.kinds, links.timeMs, this.adjacency);
    this.onLinks?.(this);
  };

  private readonly onLinksError = (err: unknown): void => {
    this.linkInFlight = false;
    if (this.disposed) return;
    this.linkRetryWall = performance.now() + LINK_RETRY_WALL_MS;
    console.warn('[app] laser-link topology request failed; keeping the previous links', err);
  };

  private readonly onKeyframeError = (err: unknown): void => {
    this.req0 = Number.NaN;
    this.req1 = Number.NaN;
    this.inFlight = 0;
    if (this.disposed || this.failed) return;
    this.failed = true;
    this.onFatal(err);
  };

  /** Removes the primitive from the scene (destroying it) and terminates the worker. */
  dispose(scene: Scene | null): void {
    if (this.disposed) return;
    this.disposed = true;
    this.points.onRecycle = undefined;
    this.onLinks = null;
    this.links = null;
    this.adjacency = null;
    if (scene && !scene.isDestroyed() && scene.primitives.contains(this.points)) {
      scene.primitives.remove(this.points); // destroys it (destroyPrimitives = true)
    } else if (!this.points.isDestroyed()) {
      this.points.destroy();
    }
    this.client.dispose();
    this.pool.length = 0;
  }
}

/** Spawns a worker for `data`, waits for its catalog and builds the session (not yet in the scene). */
async function buildSession(
  data: ElementsReady,
  shellVisible: readonly boolean[],
  onFatal: (err: unknown) => void,
  onCatalog?: (catalog: SatCatalog, rejected: number) => void,
  fadeIn = true,
): Promise<Session> {
  const client = new PropagatorClient();
  try {
    const ready = await client.init(data.records);
    if (ready.catalog.count === 0) {
      throw new Error(`None of the ${groupThousands(data.records.length)} element sets could be propagated.`);
    }
    onCatalog?.(ready.catalog, ready.rejected);
    const s = new Session(data, client, ready.catalog, shellVisible, onFatal, fadeIn);
    s.points.setLinksVisible(linksEnabled);
    s.onLinks = onSessionLinks;
    return s;
  } catch (err) {
    client.dispose();
    throw err;
  }
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

const hudRoot = document.getElementById('hud');
const globeRoot = document.getElementById('globe');
if (!hudRoot || !globeRoot) throw new Error('index.html must contain #globe and #hud');

const hud = createHud(hudRoot);

/** Per-shell visibility as chosen in the legend; survives session swaps. */
const shellVisible: boolean[] = SHELLS.map(() => true);

let globe: GlobeContext | null = null;
let session: Session | null = null;
let sky: SkyHandle | null = null;
let rig: CameraRig | null = null;
let orbit: OrbitPath | null = null;
let selection: SelectionController | null = null;
let models: SatelliteModels | null = null;
let laser: LaserHighlight | null = null;
let coverage: CoverageFootprint | null = null;
/** LASER LINKS overlay switch (HUD); default on, survives session swaps. */
let linksEnabled = true;
/** COVERAGE AREA switch (info panel); default on, survives selections, swaps and reloads. */
let coverageEnabled = readCoveragePref();
let fatal = false;

function fail(title: string, err: unknown): void {
  if (fatal) return;
  fatal = true;
  console.error(`[app] ${title}`, err);
  hud.showError(title, errorMessage(err));
}

const onSessionFatal = (err: unknown): void => fail('Propagation engine fault', err);

// ---- HUD feeds that depend on the session ----------------------------------

function pushShells(s: Session): void {
  const rows: HudShellInfo[] = SHELLS.map((def, index) => ({
    index,
    label: def.label,
    color: def.color,
    count: s.shellCounts[index],
    visible: shellVisible[index],
  }));
  hud.setShells(rows);
  let visible = 0;
  for (let i = 0; i < rows.length; i++) if (shellVisible[i]) visible += s.shellCounts[i];
  hud.setTracked(visible, s.catalog.count);
}

/**
 * Partial-snapshot ETA / refresh-cycle progress from the latest status poll,
 * for the session they were read for. They change without a new set (a
 * partition re-locked after a failure, a cycle running), so the HUD would
 * otherwise keep the boot / last swap values (e.g. "refresh due" for hours).
 */
let liveRefresh: { session: Session; next: string | null; cycle: { done: number; total: number } | null } | null = null;

function cycleProgress(c: ElementsCycleInfo | null | undefined): { done: number; total: number } | null {
  return c && c.running && c.partitionsTotal > 0 ? { done: c.partitionsDone, total: c.partitionsTotal } : null;
}

function pushDataInfo(s: Session): void {
  const d = s.data;
  const live = liveRefresh?.session === s ? liveRefresh : null;
  // Contributing feeds (served most records first); older servers omit them.
  const sources = d.sources?.filter((x) => x.records > 0).map((x) => x.name);
  hud.setDataInfo({
    count: s.catalog.count,
    source: d.source,
    sources: sources && sources.length > 0 ? sources : undefined,
    fetchedAt: isoToDate(d.fetchedAt) ?? new Date(),
    partial: d.partial,
    nextRefresh: isoToDate(live ? live.next : d.nextRefreshEarliest),
    cycle: live ? live.cycle : cycleProgress(d.cycle),
  });
}

/** HUD LASER LINKS row (shown from the first topology on; labelled MODELED by the HUD). */
function pushLaserHud(s: Session): void {
  if (!s.links) return;
  hud.setLaserLinks({ enabled: linksEnabled, links: s.points.linkCount, equipped: s.equipped });
}

/** A session applied a new topology: HUD count + the selection's neighbours (active session only). */
function onSessionLinks(s: Session): void {
  if (s !== session) return;
  pushLaserHud(s);
  selection?.refreshLinks();
}

hud.onLaserToggle((enabled) => {
  linksEnabled = enabled;
  if (session) {
    session.points.setLinksVisible(enabled);
    pushLaserHud(session); // confirms the HUD's optimistic state
  }
  selection?.refreshLinks(); // beams follow the switch
});

// COVERAGE AREA row (info panel): the row has already flipped optimistically.
hud.infoPanel.setCoverageVisible(coverageEnabled);
hud.infoPanel.onCoverageToggle = (on) => {
  coverageEnabled = on;
  writeCoveragePref(on);
  coverage?.setVisible(on);
};

hud.onShellToggle((index, visible) => {
  if (index < 0 || index >= shellVisible.length) return;
  shellVisible[index] = visible;
  if (session) {
    // Hidden shells are skipped by picking (the selected satellite stays
    // visible); SelectionController re-picks the hover within ~120 ms.
    session.points.setShellVisible(index, visible);
    pushShells(session);
  }
});

// ---- Frame loop --------------------------------------------------------------

const clockDate = new Date(0);
const scratchCarto = new Cartographic();
const cameraInfo: HudCameraInfo = { altitudeKm: 0, lat: 0, lon: 0 };
const perfInfo: HudPerfInfo = { fps: 0, frameMs: 0, resolutionScale: 1, msaa: 0, keyframeMs: Number.NaN, linksMs: Number.NaN };
let lastCameraUi = -Infinity;
let lastPerfUi = -Infinity;

function onPreUpdate(scene: Scene, time: JulianDate): void {
  const g = globe;
  if (!g || fatal) return; // after a fault: keep the pill off LIVE, stop scheduling
  const nowMs = julianToUnixMs(time);
  session?.schedule(nowMs);

  clockDate.setTime(nowMs);
  hud.setClock(clockDate);
  const clock = g.viewer.clock;
  hud.setLive(clock.shouldAnimate && clock.multiplier === 1 && Math.abs(nowMs - Date.now()) < LIVE_TOLERANCE_MS);

  const wall = performance.now();
  if (wall - lastCameraUi >= CAMERA_UI_INTERVAL_MS) {
    lastCameraUi = wall;
    const c = Cartographic.clone(scene.camera.positionCartographic, scratchCarto);
    cameraInfo.altitudeKm = c.height / 1000;
    cameraInfo.lat = CesiumMath.toDegrees(c.latitude);
    cameraInfo.lon = CesiumMath.toDegrees(c.longitude);
    hud.setCamera(cameraInfo);
  }
  if (wall - lastPerfUi >= PERF_UI_INTERVAL_MS) {
    lastPerfUi = wall;
    const q = g.quality;
    perfInfo.fps = q.fps;
    perfInfo.frameMs = q.frameMs;
    perfInfo.resolutionScale = q.resolutionScale;
    perfInfo.msaa = q.msaa;
    // Worker times: ~2.5 ms keyframe / ~16 ms links for 8k satellites when the
    // worker runs optimised code; ~10x both = unoptimised or throttled worker.
    perfInfo.keyframeMs = session?.lastKeyframeMs ?? Number.NaN;
    perfInfo.linksMs = session?.links?.computeMs ?? Number.NaN;
    hud.setPerf(perfInfo);
  }
}

// ---- Element-set updates -> hot swap ------------------------------------------
//
// The server merges several feeds (full catalogue, recent launches, SupGP) into
// one set; every merge changes its fetchedAt / count. Only the cheap local
// status endpoint is polled; the full set is fetched when it changed. A partial
// snapshot is polled every minute, a full one every five (supplements).

let pollTimer: number | undefined;
let pollEveryMs = 0;
let polling = false;

function schedulePolling(partial: boolean): void {
  const every = partial ? PARTIAL_POLL_MS : FULL_POLL_MS;
  if (pollTimer !== undefined && pollEveryMs === every) return;
  if (pollTimer !== undefined) clearInterval(pollTimer);
  pollEveryMs = every;
  pollTimer = window.setInterval(() => void pollForUpdates(), every);
}

/** Whether the server's set differs from the one `current` was built from. */
function isNewerSet(fetchedAt: string | null, count: number, current: ElementsReady): boolean {
  if (!fetchedAt || count === 0) return false;
  return fetchedAt !== current.fetchedAt || count !== current.records.length;
}

async function pollForUpdates(): Promise<void> {
  const current = session;
  if (polling || fatal || !current || !globe) return;
  polling = true;
  try {
    const st = await fetchElementsStatus();
    if (!isNewerSet(st.cacheFetchedAt, st.cacheCount, current.data)) {
      // Same set: still refresh the partial-snapshot ETA (a re-locked
      // partition moves it) and the cycle progress. nextAllowedFetchAt is the
      // value the set's nextRefreshEarliest came from (null when offline).
      if (session === current && current.data.partial && st.cachePartial) {
        const next = st.nextAllowedFetchAt;
        const cycle = cycleProgress(st.cycle);
        const prev = liveRefresh?.session === current ? liveRefresh : null;
        const prevNext = prev ? prev.next : current.data.nextRefreshEarliest;
        const prevCycle = prev ? prev.cycle : cycleProgress(current.data.cycle);
        if (next !== prevNext || cycle?.done !== prevCycle?.done || cycle?.total !== prevCycle?.total) {
          liveRefresh = { session: current, next, cycle };
          pushDataInfo(current);
        }
      }
      return;
    }
    // Never trade a full catalogue for a partial one (should not happen: the server only merges).
    if (st.cachePartial && !current.data.partial) return;
    const data = await loadElements();
    if (session !== current || !isNewerSet(data.fetchedAt, data.records.length, current.data)) return;
    if (data.partial && !current.data.partial) return;
    await hotSwap(data);
  } catch (err) {
    console.warn('[app] catalogue refresh check failed', err);
  } finally {
    polling = false;
  }
}

async function hotSwap(data: ElementsReady): Promise<void> {
  const g = globe;
  if (!g) return;
  let next: Session | null = null;
  try {
    // No fade-in and links awaited: the sky, link web, hologram and beams carry
    // on without blanking (the swap repeats ~once a minute during a cycle).
    next = await buildSession(data, shellVisible, onSessionFatal, undefined, false);
    await next.prime(julianToUnixMs(g.viewer.clock.currentTime), true);
  } catch (err) {
    next?.dispose(null);
    console.warn('[app] catalogue swap failed', err);
    hud.setWarning(`Updated catalogue received but could not be loaded: ${errorMessage(err)}`);
    return;
  }
  if (fatal || g.viewer.isDestroyed()) {
    next.dispose(null);
    return;
  }
  const old = session;
  g.viewer.scene.primitives.add(next.points);
  session = next;
  // Re-bind the selection (by NORAD id: highlight, orbit ring, panel, track
  // target) and reset the hover to the new primitive before the old one goes.
  selection?.resync();
  old?.dispose(g.viewer.scene);
  // SatelliteModels / LaserHighlight / CoverageFootprint notice the new
  // primitive by themselves (the selection re-bound the hologram above, keeping
  // its reveal when it is the same satellite); the new session's links were
  // applied in prime() (indices of ITS catalog), so resync re-bound the beams.
  pushShells(next);
  liveRefresh = null; // the new set carries its own ETA + cycle progress (and the old session goes)
  pushDataInfo(next);
  pushLaserHud(next);
  schedulePolling(next.data.partial);
  console.info(`[app] switched to ${next.data.source}: ${next.catalog.count} satellites`);
}

// ---- Boot ----------------------------------------------------------------------

async function boot(): Promise<void> {
  const ionToken = (import.meta.env.VITE_CESIUM_ION_TOKEN as string | undefined)?.trim() || undefined;

  // Honest progress: fraction of boot milestones completed.
  const MILESTONES = 4; // globe, elements, catalogue, first keyframes
  const stage = {
    done: 0,
    globeReady: false,
    elementsLoaded: false,
    objectCount: 0,
    warming: null as ElementsWarming | null,
    propagating: false,
  };

  const render = (): void => {
    const p = stage.done / MILESTONES;
    if (!stage.elementsLoaded) {
      const w = stage.warming;
      const detail = w
        ? `${formatBytes(w.bytesReceived)} · ${groupThousands(w.recordsReceived)} element sets received`
        : 'GET /api/elements';
      hud.setBoot(p, 'ACQUIRING ORBITAL ELEMENTS', w?.message ? `${w.message} — ${detail}` : detail);
    } else if (stage.propagating) {
      hud.setBoot(p, `PROPAGATING ${groupThousands(stage.objectCount)} OBJECTS`, 'SGP4 · building catalogue');
    } else if (!stage.globeReady) {
      hud.setBoot(p, 'INITIALISING GLOBE', 'WebGL · terrain · imagery');
    } else {
      hud.setBoot(p, 'COMPUTING EPHEMERIS', 'First keyframes');
    }
  };
  const milestone = (): void => {
    stage.done++;
    render();
  };
  render();

  const abort = new AbortController();

  const globePromise = createGlobe(globeRoot as HTMLElement, {
    ionToken,
    // The sky drops its 4k cube (and never re-upgrades) once the renderer degrades.
    onQualityChange: () => sky?.evaluate(),
  }).then((g) => {
    attachGlobeModules(g);
    stage.globeReady = true;
    milestone();
    return g;
  });

  const sessionPromise = loadElementsWithRetry({
    signal: abort.signal,
    onProgress: (w) => {
      stage.warming = w;
      render();
    },
  }).then((data) => {
    stage.elementsLoaded = true;
    stage.objectCount = data.records.length;
    stage.propagating = true;
    milestone();
    return buildSession(data, shellVisible, onSessionFatal).then(
      (built) => {
        stage.propagating = false;
        milestone(); // catalogue built
        return built;
      },
      (err: unknown) => {
        stage.propagating = false;
        throw err;
      },
    );
  });

  // Which half failed decides the error title.
  let failedPart = '';
  globePromise.catch(() => {
    failedPart ||= 'Globe initialisation failure';
  });
  sessionPromise.catch(() => {
    failedPart ||= stage.elementsLoaded ? 'Propagation engine fault' : 'Orbital data unavailable';
  });

  let g: GlobeContext;
  let s: Session;
  try {
    [g, s] = await Promise.all([globePromise, sessionPromise]);
  } catch (err) {
    abort.abort(new Error('boot failed'));
    // Clean up whichever half succeeded.
    globePromise.then(
      (x) => {
        detachGlobeModules();
        x.destroy();
      },
      () => undefined,
    );
    sessionPromise.then((x) => x.dispose(null), () => undefined);
    await Promise.resolve(); // let the .catch handlers above record which part failed
    fail(failedPart || 'Boot failure', err);
    return;
  }

  globe = g;
  if (!ionToken) hud.setWarning(NO_TOKEN_WARNING);

  // Render-loop faults: Cesium stops rendering (exception in a frame) or, after
  // a GPU reset / context reclaim, keeps drawing nothing. Show the HUD error
  // card (it covers Cesium's own raw panel) instead of a frozen globe saying LIVE.
  const faultScene = g.viewer.scene;
  faultScene.renderError.addEventListener((_s: Scene, err: unknown) => {
    hud.setLive(false);
    fail('Render engine fault', err);
  });
  faultScene.canvas.addEventListener('webglcontextlost', () => {
    hud.setLive(false);
    fail('Graphics context lost', new Error('The GPU driver reset or the browser reclaimed the WebGL context. Reload to restart the tracker.'));
  });
  // A fault before these listeners existed (while the elements were loading)
  // has already stopped Cesium's render loop.
  if (!g.viewer.useDefaultRenderLoop) {
    hud.setLive(false);
    fail('Render engine fault', new Error('Rendering stopped during start-up.'));
    return;
  }

  try {
    await s.prime(julianToUnixMs(g.viewer.clock.currentTime));
  } catch (err) {
    s.dispose(null);
    fail('Propagation engine fault', err);
    return;
  }
  g.viewer.scene.primitives.add(s.points);
  session = s;
  milestone();

  const scene = g.viewer.scene;
  scene.preUpdate.addEventListener(onPreUpdate);

  const cameraRig = rig as CameraRig; // created in attachGlobeModules (throws otherwise)
  const orbitPath = orbit as OrbitPath;
  // Click / double-click selection, hover label + cursor, info panel feed,
  // orbit ring + ground track, FOCUS (track) / RELEASE / CLOSE. It assigns the
  // panel's onFocus / onRelease / onClose itself.
  selection = new SelectionController({
    viewer: g.viewer,
    getPoints: () => session?.points ?? null,
    getRecord: (i) => session?.recordAt(i) ?? null, // catalog index, not raw records index
    getCatalog: () => session?.catalog ?? null,
    panel: hud.infoPanel,
    rig: cameraRig,
    orbit: orbitPath,
    onHoverLabel: (text, x, y) => hud.setHoverLabel(text, x, y),
    // Round 3: estimated hardware, MODELLED laser-link neighbours, close-up model.
    getLinks: () => session?.adjacency ?? null,
    linksEnabled: () => linksEnabled,
    laser: laser ?? undefined,
    models: models ?? undefined,
    // Round 4: coverage hologram follows the selection (visibility: COVERAGE AREA toggle).
    coverage: coverage ?? undefined,
  });

  pushShells(s);
  pushDataInfo(s);
  pushLaserHud(s);
  hud.setLive(true);
  schedulePolling(s.data.partial);

  hud.finishBoot();
  // The hint fades on the first real camera gesture (drag / wheel / keys / touch).
  cameraRig.onUserInput = () => {
    hud.dismissHint();
    cameraRig.onUserInput = null;
  };
  // Boot intro: the rig owns the camera every frame (never use camera.flyTo).
  void cameraRig.flyToOverview();

  if (import.meta.env.DEV) {
    // Console handles for live tuning, e.g. __starlink.points().setStyle({ brightness: 1.5 }),
    // __starlink.sky.debug.markers(true), __starlink.hud.infoPanel.show({...}).
    (window as unknown as { __starlink?: unknown }).__starlink = {
      viewer: g.viewer,
      quality: g.quality,
      hud,
      rig: cameraRig,
      sky,
      orbit: orbitPath,
      selection,
      points: () => session?.points,
      catalog: () => session?.catalog,
      session: () => session,
      // Round 3: __starlink.models.debugInfo(), __starlink.laser.setSource(...),
      // __starlink.links() (latest MODELLED topology), await __starlink.client().links(Date.now()).
      models,
      laser,
      // Round 4: __starlink.coverage.debugInfo(), __starlink.coverage.setVisible(false),
      // __starlink.pollNow() (re-check the server's merged element set now).
      coverage,
      pollNow: () => pollForUpdates(),
      links: () => session?.links ?? null,
      adjacency: () => session?.adjacency ?? null,
      client: () => session?.client,
      linksEnabled: () => linksEnabled,
    };
  }
}

/**
 * Modules that only need the viewer, created as soon as the globe exists (in
 * parallel with the element download):
 *   - the sky starts fetching its cube + stars right away (the old Tycho sky
 *     keeps drawing until the 2k cube is uploaded),
 *   - the camera rig takes over from Cesium's ScreenSpaceCameraController
 *     (inputs + collision detection off) and holds the start view,
 *   - the orbit path pre-compiles its shaders long before the first click.
 */
function attachGlobeModules(g: GlobeContext): void {
  const viewer = g.viewer;
  try {
    // Must run after createGlobe's `scene.skyBox.show = true`: createSky keeps
    // that sky box as the fallback and destroys it once its own cube shows.
    sky = createSky(viewer, { isDegraded: () => g.quality.resolutionScale < 1 });
  } catch (err) {
    sky = null;
    console.warn('[app] sky unavailable, keeping Cesium\'s default sky box', err);
  }
  rig = new CameraRig(viewer);
  orbit = new OrbitPath(viewer.scene);
  // Close-up models: every .glb in BUS_VERSIONS (incl. starlink-v3.glb) starts
  // loading (and gets its shaders warmed up sub-pixel) now, in parallel with
  // the element download. They and
  // the beams follow the current session's points via the getters (hot swaps
  // are detected by the modules themselves). Optional features: a failure here
  // only loses the close-ups / beams.
  try {
    models = new SatelliteModels({
      scene: viewer.scene,
      points: () => session?.points ?? null,
      catalog: () => session?.catalog ?? null,
    });
  } catch (err) {
    models = null;
    console.warn('[app] close-up models unavailable', err);
  }
  try {
    laser = new LaserHighlight(viewer.scene, () => session?.points ?? null);
  } catch (err) {
    laser = null;
    console.warn('[app] laser beams unavailable', err);
  }
  // Coverage hologram: shaders compile in its first frames (long before the
  // first click). Optional: a failure only loses the footprint.
  try {
    coverage = new CoverageFootprint({
      scene: viewer.scene,
      points: () => session?.points ?? null,
      catalog: () => session?.catalog ?? null,
    });
    coverage.setVisible(coverageEnabled);
  } catch (err) {
    coverage = null;
    console.warn('[app] coverage footprint unavailable', err);
  }
}

/** Reverse of attachGlobeModules (+ selection), before the viewer is destroyed. */
function detachGlobeModules(): void {
  // Selection first: its clear() still talks to the beams / models / hologram.
  selection?.destroy();
  selection = null;
  coverage?.destroy();
  coverage = null;
  laser?.destroy();
  laser = null;
  models?.destroy();
  models = null;
  orbit?.destroy();
  orbit = null;
  rig?.destroy();
  rig = null;
  sky?.destroy();
  sky = null;
}

void boot();
