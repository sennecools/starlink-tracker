/**
 * SelectionController — click a satellite to select it, show its orbit and
 * feed the info panel; double-click (or the panel's FOCUS button) to track it.
 *
 * Pointer handling lives on viewer.canvas next to the camera rig and never
 * fights it: listeners are passive, nothing is prevented or captured, and a
 * press only counts as a click when it moved < 5 px and lasted < 350 ms
 * (anything else is the rig's drag). The one exception: a mouse double-click
 * on a satellite (the browser's own dblclick, i.e. the OS threshold) is taken
 * in the capture phase and prevented, so the rig skips its dblclick zoom. Hover picking is frame-synced
 * (scene.postRender): at most one pick per rendered frame while the pointer
 * moves, plus one every ~120 ms while it rests (satellites move under it), and
 * none at all during a drag or for touch input.
 *
 * Selection lifecycle:
 *   select(i)  -> points.setSelected, panel.show(satStatic), orbit ring + ground
 *                 track, live rows ~4x per second (satLive -> panel.update),
 *                 ring/track recomputed every ~30 s of sim time.
 *   click on empty space keeps the selection; Escape / panel close clears it
 *   (releasing the camera if it was tracking).
 *   Hot swap: when getPoints()/getCatalog() return new objects (a new element
 *   set), the selection is re-resolved by NORAD id or cleared.
 *
 * Round 3 (3D close-up + MODELLED laser links):
 *   select(i)  -> panel.setHardware (estimated bus + laser terminals), the
 *                 forced close-up model (models.setForced), the selected
 *                 satellite's laser-link neighbours from the latest adjacency
 *                 (O(degree), rebuilt by main once per links update): panel
 *                 rows with live distances (~4 Hz) and LaserHighlight beams.
 *   FOCUS / double-click flies to a close-up (closeUpRangeM: CLOSE_UP_RANGE_M,
 *   farther for the ~60 m V3); re-targets (another satellite, a neighbour
 *   row) keep the current track range.
 *
 * Round 4: select(i) / clear() also drive the coverage hologram
 * (coverage.setSatellite). Its visibility is the panel's COVERAGE AREA
 * toggle, wired by main (coverage.setVisible), so it is kept across
 * selections; the hologram re-binds on hot swaps through apply().
 */
import { Cartesian2, Cartesian3, Color } from 'cesium';
import type { Viewer } from 'cesium';
import { gstime, sgp4 } from 'satellite.js';
import type { SatRec } from 'satellite.js';
import { orbitColorForShell, orbitNeedsRefresh } from '../render/OrbitPath.ts';
import type { OrbitPath } from '../render/OrbitPath.ts';
import { julianToUnixMs } from '../render/SatellitePoints.ts';
import type { SatellitePoints } from '../render/SatellitePoints.ts';
import type { ElementRecord, SatCatalog } from '../shared/contracts.ts';
import { BUS_V2MINI, BUS_V3, BUS_VERSIONS } from '../shared/hardware.ts';
import type { CameraRigApi, InfoPanelApi, PanelLink, SatStatic } from '../shared/phase2.ts';
import { SHELLS, SHELL_OTHER } from '../shared/shells.ts';
import { julianDateFromMs } from '../worker/propagate.ts';
import {
  createSatLive,
  groundTrackEcef,
  makeSatrec,
  nodeRateRadPerMs,
  orbitEci,
  orbitPeriodMs,
  satLive,
  satStatic,
} from './satDetails.ts';
import type { LinkAdjacency } from './linkAdjacency.ts';

/** A press is a click below these limits (CSS px / ms). */
const CLICK_MAX_MOVE_PX = 5;
const CLICK_MAX_MS = 350;
/** Touch: second tap within this time / distance on the same satellite = double-tap (mouse uses dblclick). */
const DOUBLE_CLICK_MS = 400;
const DOUBLE_CLICK_MAX_PX = 8;
const CLICK_RADIUS_PX = 12;
const HOVER_RADIUS_PX = 10;
/** Re-pick under a resting cursor this often (wall ms). */
const HOVER_REPICK_MS = 120;
/** Live panel rows refresh (wall ms). */
const LIVE_INTERVAL_MS = 250;
/** Orbit ring samples (2049 points: ~21 km chords, ~8 m chord sag at LEO; < 1 px down to ~10 km track range). */
const ORBIT_SAMPLES = 2048;
/** Ground track: next ~95 minutes, one point every 30 s. */
const GROUND_TRACK_MIN = 95;
const GROUND_TRACK_SAMPLES = 190;
/** Do not rebuild the ring more often than this (wall ms), e.g. under time warp. */
const ORBIT_MIN_REBUILD_WALL_MS = 1000;
/** FOCUS / double-click: camera range of the 3D close-up (a V2 Mini spans ~30 m). */
export const CLOSE_UP_RANGE_M = 120;
/** FOCUS range for the V3 (~60 m wing span, twice a V2 Mini). */
export const CLOSE_UP_RANGE_V3_M = 220;

/** Close-up camera range for a bus index (BUS_VERSIONS). */
export function closeUpRangeM(bus: number | undefined): number {
  return bus === BUS_V3 ? CLOSE_UP_RANGE_V3_M : CLOSE_UP_RANGE_M;
}
/** Laser-link rows / beams shown per satellite (Starlink carries 3-4 terminals). */
const MAX_LINK_ROWS = 8;

export interface SelectionControllerOptions {
  viewer: Viewer;
  getPoints: () => SatellitePoints | null;
  getRecord: (index: number) => ElementRecord | null;
  getCatalog: () => SatCatalog | null;
  panel: InfoPanelApi;
  /** trackRangeM (CameraRig): current track range, so re-targets keep the zoom. */
  rig: CameraRigApi & { readonly trackRangeM?: number };
  orbit: OrbitPath;
  /** Latest MODELLED laser-link adjacency of the current catalog, or null (none yet). */
  getLinks?: () => LinkAdjacency | null;
  /** Whether the laser-link overlay is switched on (the beams follow it). */
  linksEnabled?: () => boolean;
  /** Beams from the selected satellite to its link neighbours (LaserHighlight). */
  laser?: { setSource(index: number, neighbours: readonly number[], kinds: readonly number[]): void; clear(): void };
  /** Close-up 3D models (SatelliteModels): the selected satellite always gets one when near. */
  models?: { setForced(index: number): void };
  /** Coverage hologram (CoverageFootprint): follows the selection; visibility is set by main. */
  coverage?: { setSatellite(index: number, sameSatellite?: boolean): void };
  /**
   * Hover label: satellite name and the pointer position in viewport (client)
   * CSS px, or null to hide. Called when the hovered satellite changes and
   * while the pointer moves over one.
   */
  onHoverLabel?: (text: string | null, x: number, y: number) => void;
}

type XYZ = { x: number; y: number; z: number };

export class SelectionController {
  private readonly opts: SelectionControllerOptions;
  private readonly viewer: Viewer;
  private readonly canvas: HTMLCanvasElement;

  // --- selection state ---
  private selectedIndex = -1;
  private noradId = -1;
  private satrec: SatRec | null = null;
  private info: SatStatic | null = null;
  private boundPoints: SatellitePoints | null = null;
  private boundCatalog: SatCatalog | null = null;
  private tracking = false;
  private readonly live = createSatLive();
  private liveTimer: ReturnType<typeof setInterval> | null = null;
  private orbitTimeMs = Number.NaN;
  private lastOrbitWall = -Infinity;
  private ringBuf: Float64Array = new Float64Array((ORBIT_SAMPLES + 1) * 3);
  private trackBuf: Float64Array = new Float64Array((GROUND_TRACK_SAMPLES + 1) * 3);
  private readonly ringColor = new Color();
  private readonly tmpPos = new Cartesian3();
  private readonly tmpPos2 = new Cartesian3();

  // --- laser-link neighbours of the selection (MODELLED topology) ---
  /** Adjacency + version the neighbour list was taken from (null = none). */
  private linkAdj: LinkAdjacency | null = null;
  private linkVersion = -1;
  private linkBeamsOn = false;
  private readonly nbrIdx: number[] = [];
  private readonly nbrKind: number[] = [];
  /** Reused panel rows (pool) and the view handed to panel.setLinks. */
  private readonly linkRowPool: PanelLink[] = [];
  private readonly linkRows: PanelLink[] = [];

  // --- pointer state ---
  private downId = -1;
  private downX = 0;
  private downY = 0;
  private downTime = 0;
  private dragging = false;
  private lastClickTime = -Infinity;
  private lastClickIndex = -1;
  private lastClickX = 0;
  private lastClickY = 0;
  /** timeStamp of the last touch double-tap that focused (its trailing dblclick, if any, is not a zoom). */
  private lastTapFocusTime = -Infinity;
  private pointerInside = false;
  private pointerIsMouse = true;
  private clientX = 0;
  private clientY = 0;
  private rectLeft = 0;
  private rectTop = 0;
  private rectValid = false;
  private hoverDirty = false;
  private lastPickWall = -Infinity;
  private hovered = -1;
  private cursorSet = false;
  private savedCursor = '';
  private hoverText = '';
  private readonly pickPos = new Cartesian2();

  private readonly removePostRender: () => void;
  private destroyed = false;

  constructor(opts: SelectionControllerOptions) {
    this.opts = opts;
    this.viewer = opts.viewer;
    this.canvas = opts.viewer.scene.canvas;

    const c = this.canvas;
    c.addEventListener('pointerdown', this.onPointerDown, { passive: true });
    c.addEventListener('pointermove', this.onPointerMove, { passive: true });
    c.addEventListener('pointerup', this.onPointerUp, { passive: true });
    c.addEventListener('pointercancel', this.onPointerCancel, { passive: true });
    c.addEventListener('pointerenter', this.onPointerEnter, { passive: true });
    c.addEventListener('pointerleave', this.onPointerLeave, { passive: true });
    // Capture phase (and not passive): runs before the rig's dblclick zoom and can prevent it.
    c.addEventListener('dblclick', this.onDblClick, { capture: true });
    window.addEventListener('keydown', this.onKeyDown);
    window.addEventListener('resize', this.onResize, { passive: true });
    this.removePostRender = this.viewer.scene.postRender.addEventListener(this.onPostRender);

    opts.panel.onFocus = this.onPanelFocus;
    opts.panel.onRelease = this.onPanelRelease;
    opts.panel.onClose = this.onPanelClose;
    opts.panel.onSelectNeighbor = this.onPanelNeighbor;
  }

  /** Selected satellite index in the current catalog, -1 = none. */
  get selected(): number {
    return this.selectedIndex;
  }

  /** Whether the camera rig is tracking the selected satellite. */
  get isTracking(): boolean {
    return this.tracking;
  }

  /** Selects satellite `index` of the current catalog (invalid index => clear()). */
  select(index: number): void {
    if (this.destroyed) return;
    const points = this.opts.getPoints();
    const catalog = this.opts.getCatalog();
    if (!points || !catalog || !(Number.isInteger(index) && index >= 0 && index < catalog.count)) {
      this.clear();
      return;
    }
    if (!this.apply(index, points, catalog)) return;
    // Re-target a running track smoothly (from the view on the previous
    // satellite), keeping the current zoom (e.g. hopping along laser links up close).
    if (this.tracking) {
      const range = this.opts.rig.trackRangeM;
      this.opts.rig.track(this.targetFor(this.noradId), range !== undefined && range > 0 ? { rangeM: range } : {});
    }
  }

  /**
   * Re-reads the laser-link neighbours of the selection now (main calls this
   * when a new topology arrives or the overlay is toggled). Cheap: O(degree).
   */
  refreshLinks(): void {
    if (this.destroyed || this.selectedIndex < 0) return;
    this.updateNeighbours();
  }

  /** Clears the selection (and releases the camera if it was tracking it). */
  clear(): void {
    const had = this.selectedIndex >= 0 || this.info !== null;
    if (this.tracking) {
      this.tracking = false;
      if (this.opts.rig.mode === 'track') this.opts.rig.release();
    }
    this.selectedIndex = -1;
    this.noradId = -1;
    this.satrec = null;
    this.info = null;
    this.orbitTimeMs = Number.NaN;
    this.stopLive();
    this.resetNeighbours();
    this.opts.models?.setForced(-1);
    this.opts.coverage?.setSatellite(-1);
    const bound = this.boundPoints;
    if (bound && !bound.isDestroyed()) bound.setSelected(-1);
    const current = this.opts.getPoints();
    if (current && current !== bound && !current.isDestroyed()) current.setSelected(-1);
    this.boundPoints = null;
    this.boundCatalog = null;
    if (!this.opts.orbit.isDestroyed()) this.opts.orbit.clear();
    if (had) {
      this.opts.panel.setTracking(false);
      this.opts.panel.show(null);
    }
  }

  /** Starts tracking the selected satellite with the camera rig. */
  focus(): void {
    if (this.destroyed || this.selectedIndex < 0) return;
    // Already tracking it: keep the user's range (re-tracking would reset it).
    // Otherwise fly in to the 3D close-up (SatelliteModels shows the model).
    if (!(this.tracking && this.opts.rig.mode === 'track')) {
      this.tracking = true;
      this.opts.rig.track(this.targetFor(this.noradId), { rangeM: closeUpRangeM(this.boundCatalog?.bus[this.selectedIndex]) });
    }
    this.opts.panel.setTracking(true);
  }

  /**
   * Re-resolves the selection right away after getPoints()/getCatalog() started
   * returning new objects (hot swap), instead of on the next live tick: rebinds
   * by NORAD id (highlight, orbit, panel) or clears it. Hover is reset too.
   */
  resync(): void {
    if (this.destroyed) return;
    const points = this.opts.getPoints();
    if (this.selectedIndex >= 0) this.currentPoints();
    else this.setHover(points, -1);
  }

  /** Stops tracking (the selection stays). */
  release(): void {
    if (this.tracking) {
      this.tracking = false;
      if (this.opts.rig.mode === 'track') this.opts.rig.release();
    }
    this.opts.panel.setTracking(false);
  }

  destroy(): void {
    if (this.destroyed) return;
    this.clear();
    this.destroyed = true;
    const c = this.canvas;
    c.removeEventListener('pointerdown', this.onPointerDown);
    c.removeEventListener('pointermove', this.onPointerMove);
    c.removeEventListener('pointerup', this.onPointerUp);
    c.removeEventListener('pointercancel', this.onPointerCancel);
    c.removeEventListener('pointerenter', this.onPointerEnter);
    c.removeEventListener('pointerleave', this.onPointerLeave);
    c.removeEventListener('dblclick', this.onDblClick, { capture: true });
    window.removeEventListener('keydown', this.onKeyDown);
    window.removeEventListener('resize', this.onResize);
    this.removePostRender();
    this.setHover(null, -1);
    const panel = this.opts.panel;
    if (panel.onFocus === this.onPanelFocus) panel.onFocus = null;
    if (panel.onRelease === this.onPanelRelease) panel.onRelease = null;
    if (panel.onClose === this.onPanelClose) panel.onClose = null;
    if (panel.onSelectNeighbor === this.onPanelNeighbor) panel.onSelectNeighbor = null;
  }

  // -------------------------------------------------------------------------
  // Selection internals
  // -------------------------------------------------------------------------

  /** Binds the selection to `index` of `points`/`catalog`. False (and cleared) on bad data. */
  private apply(index: number, points: SatellitePoints, catalog: SatCatalog): boolean {
    const record = this.opts.getRecord(index);
    let satrec: SatRec;
    try {
      if (!record) throw new Error(`no element record for index ${index}`);
      // getRecord must use catalog indices (the worker drops rejected TLEs).
      if (record.noradId !== catalog.noradIds[index]) {
        throw new Error(`record ${record.noradId} does not match catalog index ${index} (${catalog.noradIds[index]})`);
      }
      satrec = makeSatrec(record);
    } catch (err) {
      console.warn('[SelectionController] cannot select', index, err);
      this.clear();
      return false;
    }
    const prevBound = this.boundPoints;
    if (prevBound && prevBound !== points && !prevBound.isDestroyed()) prevBound.setSelected(-1);
    // Re-binding the same NORAD id (catalogue hot swap via resync).
    const same = this.noradId === record.noradId;

    this.selectedIndex = index;
    this.noradId = record.noradId;
    this.satrec = satrec;
    this.boundPoints = points;
    this.boundCatalog = catalog;
    points.setSelected(index);

    this.info = satStatic(index, record, catalog, satrec);
    this.opts.panel.show(this.info);
    this.opts.panel.setTracking(this.tracking);
    // Estimated hardware (after show(): a new satellite resets these rows).
    const bus = BUS_VERSIONS[catalog.bus[index]] ?? BUS_VERSIONS[BUS_V2MINI]!;
    this.opts.panel.setHardware(bus.label, catalog.lasers[index] ?? 0);
    this.opts.models?.setForced(index);
    // `points` is already the live primitive here (main swaps the session
    // before resync), so the hologram binds this index to the right catalogue.
    this.opts.coverage?.setSatellite(index, same);
    this.linkVersion = -1; // force a neighbour refresh for the new satellite

    const shell = SHELLS[this.info.shellIndex] ?? SHELLS[SHELL_OTHER]!;
    orbitColorForShell(shell.color, this.ringColor);

    const now = this.simNowMs();
    this.refreshOrbit(now);
    this.updateLive(now);
    this.updateNeighbours();
    this.startLive();
    return true;
  }

  // -------------------------------------------------------------------------
  // Laser-link neighbours (MODELLED topology)
  // -------------------------------------------------------------------------

  private resetNeighbours(): void {
    this.linkAdj = null;
    this.linkVersion = -1;
    this.nbrIdx.length = 0;
    this.nbrKind.length = 0;
    this.linkRows.length = 0;
    if (this.linkBeamsOn) {
      this.linkBeamsOn = false;
      this.opts.laser?.clear();
    }
  }

  /** Neighbour list of the selection from the latest adjacency -> beams + panel rows. */
  private updateNeighbours(): void {
    const index = this.selectedIndex;
    const catalog = this.boundCatalog;
    const adj = this.opts.getLinks?.() ?? null;
    if (index < 0 || !catalog || !adj || adj.count !== catalog.count || index >= adj.count) {
      // No topology (yet) for this catalog: hide the beams. The panel's link
      // group stays hidden (new satellite) or keeps its last rows until the
      // next update; stale rows are not clickable (onPanelNeighbor checks nbrIdx).
      this.resetNeighbours();
      return;
    }
    this.linkAdj = adj;
    this.linkVersion = adj.version;
    this.nbrIdx.length = 0;
    this.nbrKind.length = 0;
    const end = adj.offsets[index + 1]!;
    for (let k = adj.offsets[index]!; k < end && this.nbrIdx.length < MAX_LINK_ROWS; k++) {
      this.nbrIdx.push(adj.neighbors[k]!);
      this.nbrKind.push(adj.kinds[k]!);
    }
    const laser = this.opts.laser;
    const enabled = this.opts.linksEnabled?.() ?? true;
    if (laser) {
      if (enabled && this.nbrIdx.length > 0) {
        laser.setSource(index, this.nbrIdx, this.nbrKind);
        this.linkBeamsOn = true;
      } else if (this.linkBeamsOn) {
        laser.clear();
        this.linkBeamsOn = false;
      }
    }
    this.updateLinkRows(this.simNowMs());
  }

  /** Panel rows with live distances (~4x per second). Allocation-free once the row pool is warm. */
  private updateLinkRows(nowMs: number): void {
    const catalog = this.boundCatalog;
    const points = this.boundPoints;
    if (!catalog || !points || points.isDestroyed() || !this.linkAdj) return;
    const n = this.nbrIdx.length;
    const rows = this.linkRows;
    rows.length = 0;
    const src = points.getPosition(this.selectedIndex, this.tmpPos, nowMs);
    for (let j = 0; j < n; j++) {
      let row = this.linkRowPool[j];
      if (!row) {
        row = { index: -1, name: '', kind: 0, distanceKm: Number.NaN };
        this.linkRowPool.push(row);
      }
      const i = this.nbrIdx[j]!;
      row.index = i;
      row.name = catalog.names[i] ?? `#${i}`;
      row.kind = this.nbrKind[j]!;
      const p = src ? points.getPosition(i, this.tmpPos2, nowMs) : undefined;
      row.distanceKm = src && p ? Cartesian3.distance(src, p) / 1000 : Number.NaN;
      rows.push(row);
    }
    this.opts.panel.setLinks(rows);
  }

  /**
   * Current points primitive; re-resolves the selection when the primitive or
   * catalog was hot-swapped (by NORAD id, or clears it).
   */
  private currentPoints(): SatellitePoints | null {
    const points = this.opts.getPoints();
    if (this.selectedIndex < 0 || !points) return points;
    const catalog = this.opts.getCatalog();
    if (points === this.boundPoints && catalog === this.boundCatalog) return points;
    if (!catalog) return null; // mid-swap: keep the selection, try again later
    const ids = catalog.noradIds;
    const wanted = this.noradId;
    let found = -1;
    for (let j = 0, n = catalog.count; j < n; j++) {
      if (ids[j] === wanted) {
        found = j;
        break;
      }
    }
    // Hover indices refer to the old catalog.
    this.setHover(points, -1);
    if (found < 0) {
      this.clear();
      return points;
    }
    this.apply(found, points, catalog);
    return points;
  }

  private resolveQueued = false;

  private scheduleResolve(): void {
    if (this.resolveQueued) return;
    this.resolveQueued = true;
    queueMicrotask(() => {
      this.resolveQueued = false;
      if (!this.destroyed) this.currentPoints();
    });
  }

  private simNowMs(): number {
    return julianToUnixMs(this.viewer.clock.currentTime);
  }

  private startLive(): void {
    if (this.liveTimer !== null) return;
    this.liveTimer = setInterval(this.onLiveTick, LIVE_INTERVAL_MS);
  }

  private stopLive(): void {
    if (this.liveTimer === null) return;
    clearInterval(this.liveTimer);
    this.liveTimer = null;
  }

  private updateLive(nowMs: number): void {
    const satrec = this.satrec;
    if (!satrec) return;
    satLive(satrec, nowMs, this.live);
    this.opts.panel.update(this.live);
  }

  private refreshOrbit(nowMs: number): void {
    const satrec = this.satrec;
    if (!satrec || !Number.isFinite(nowMs)) return;
    this.orbitTimeMs = nowMs;
    this.lastOrbitWall = performance.now();
    const orbit = this.opts.orbit;
    // OrbitPath copies the samples, so the buffers are reused.
    this.ringBuf = orbitEci(satrec, nowMs, ORBIT_SAMPLES, this.ringBuf);
    orbit.setOrbit(this.ringBuf, this.ringColor, nowMs, orbitPeriodMs(satrec), nodeRateRadPerMs(satrec));
    this.trackBuf = groundTrackEcef(satrec, nowMs, GROUND_TRACK_MIN, GROUND_TRACK_SAMPLES, this.trackBuf);
    orbit.setGroundTrack(this.trackBuf);
  }

  private readonly onLiveTick = (): void => {
    if (this.destroyed || this.selectedIndex < 0) {
      this.stopLive();
      return;
    }
    this.currentPoints(); // hot-swap check
    if (this.selectedIndex < 0) return;
    const now = this.simNowMs();
    this.updateLive(now);
    // New topology (or none any more) -> neighbours; otherwise just the distances.
    const adj = this.opts.getLinks?.() ?? null;
    if (adj !== this.linkAdj || (adj !== null && adj.version !== this.linkVersion)) this.updateNeighbours();
    else if (this.linkAdj) this.updateLinkRows(now);
    if (this.tracking && this.opts.rig.mode !== 'track') {
      // The rig left track mode on its own (user flew away, target lost).
      this.tracking = false;
      this.opts.panel.setTracking(false);
    }
    if (orbitNeedsRefresh(this.orbitTimeMs, now) && performance.now() - this.lastOrbitWall >= ORBIT_MIN_REBUILD_WALL_MS) {
      this.refreshOrbit(now);
    }
  };

  /**
   * Rig getter bound to one satellite (by NORAD id, so hot swaps keep it):
   * returns false once the selection moved on, so the rig freezes the start of
   * a re-target / release flight instead of jumping to the new satellite.
   */
  private targetFor(noradId: number): (out: XYZ) => boolean {
    return (out: XYZ): boolean => this.noradId === noradId && this.trackTarget(out);
  }

  /** Camera rig target: the selected satellite's ECEF position (metres) for this frame. */
  private readonly trackTarget = (out: XYZ): boolean => {
    const index = this.selectedIndex;
    if (this.destroyed || index < 0) return false;
    const points = this.opts.getPoints();
    // Never re-resolve a hot swap inside the rig's frame callback (it may
    // clear + release the rig re-entrantly): defer it, use SGP4 meanwhile.
    const swapped = points !== this.boundPoints || this.opts.getCatalog() !== this.boundCatalog;
    if (swapped) this.scheduleResolve();
    const nowMs = this.simNowMs();
    // Same interpolation (and time) the GPU uses for this frame -> no jitter.
    if (!swapped && points && !points.isDestroyed() && points.getPosition(index, this.tmpPos, nowMs)) {
      out.x = this.tmpPos.x;
      out.y = this.tmpPos.y;
      out.z = this.tmpPos.z;
      return true;
    }
    // Keyframes not there (yet), e.g. right after a hot swap: SGP4 directly.
    const satrec = this.satrec;
    if (!satrec) return false;
    const jd = julianDateFromMs(nowMs);
    const pv = sgp4(satrec, (jd - satrec.jdsatepoch) * 1440);
    if (pv === null || satrec.error !== 0) return false;
    const p = pv.position;
    const g = gstime(jd);
    const c = Math.cos(g);
    const s = Math.sin(g);
    out.x = (p.x * c + p.y * s) * 1000;
    out.y = (-p.x * s + p.y * c) * 1000;
    out.z = p.z * 1000;
    return Number.isFinite(out.x) && Number.isFinite(out.y) && Number.isFinite(out.z);
  };

  private readonly onPanelFocus = (): void => {
    this.focus();
  };

  private readonly onPanelRelease = (): void => {
    this.release();
  };

  private readonly onPanelClose = (): void => {
    this.clear();
  };

  /** A laser-link row was clicked: select that neighbour (only while it still is one). */
  private readonly onPanelNeighbor = (index: number): void => {
    if (this.destroyed || this.selectedIndex < 0 || !this.nbrIdx.includes(index)) return;
    this.select(index);
  };

  // -------------------------------------------------------------------------
  // Pointer / keyboard
  // -------------------------------------------------------------------------

  private updateRect(): void {
    const r = this.canvas.getBoundingClientRect();
    this.rectLeft = r.left;
    this.rectTop = r.top;
    this.rectValid = true;
  }

  private pickAt(clientX: number, clientY: number, radiusPx: number): number {
    const points = this.currentPoints();
    if (!points || points.isDestroyed()) return -1;
    if (!this.rectValid) this.updateRect();
    this.pickPos.x = clientX - this.rectLeft;
    this.pickPos.y = clientY - this.rectTop;
    return points.pick(this.viewer.scene, this.pickPos, radiusPx);
  }

  private readonly onPointerDown = (e: PointerEvent): void => {
    this.pointerIsMouse = e.pointerType === 'mouse' || e.pointerType === 'pen';
    if (e.button !== 0 || !e.isPrimary) {
      this.downId = -1;
      return;
    }
    this.updateRect();
    this.downId = e.pointerId;
    this.downX = e.clientX;
    this.downY = e.clientY;
    this.downTime = e.timeStamp;
    this.dragging = false;
  };

  private readonly onPointerMove = (e: PointerEvent): void => {
    this.clientX = e.clientX;
    this.clientY = e.clientY;
    this.pointerInside = true;
    this.pointerIsMouse = e.pointerType === 'mouse' || e.pointerType === 'pen';
    if (this.downId !== -1) {
      if (e.buttons === 0) {
        // Released outside the canvas without capture.
        this.downId = -1;
        this.dragging = false;
      } else if (e.pointerId === this.downId && !this.dragging) {
        const dx = e.clientX - this.downX;
        const dy = e.clientY - this.downY;
        if (dx * dx + dy * dy >= CLICK_MAX_MOVE_PX * CLICK_MAX_MOVE_PX) {
          this.dragging = true;
          this.setHover(this.opts.getPoints(), -1);
        }
      }
    }
    this.hoverDirty = true;
  };

  private readonly onPointerUp = (e: PointerEvent): void => {
    if (e.pointerId !== this.downId) return;
    this.downId = -1;
    const wasDrag = this.dragging;
    this.dragging = false;
    if (wasDrag || e.button !== 0) return;
    const dx = e.clientX - this.downX;
    const dy = e.clientY - this.downY;
    if (dx * dx + dy * dy >= CLICK_MAX_MOVE_PX * CLICK_MAX_MOVE_PX) return;
    if (e.timeStamp - this.downTime >= CLICK_MAX_MS) return;
    this.onClick(e.clientX, e.clientY, e.timeStamp);
  };

  private readonly onPointerCancel = (e: PointerEvent): void => {
    if (e.pointerId === this.downId) {
      this.downId = -1;
      this.dragging = false;
    }
  };

  private readonly onPointerEnter = (e: PointerEvent): void => {
    this.pointerInside = true;
    this.pointerIsMouse = e.pointerType === 'mouse' || e.pointerType === 'pen';
    this.rectValid = false;
    this.hoverDirty = true;
  };

  private readonly onPointerLeave = (): void => {
    this.pointerInside = false;
    this.setHover(this.opts.getPoints(), -1);
  };

  private readonly onResize = (): void => {
    this.rectValid = false;
  };

  private onClick(clientX: number, clientY: number, timeStamp: number): void {
    const index = this.pickAt(clientX, clientY, CLICK_RADIUS_PX);
    const dx = clientX - this.lastClickX;
    const dy = clientY - this.lastClickY;
    // Mouse/pen double-clicks come from the browser's dblclick (onDblClick).
    const isDouble =
      !this.pointerIsMouse &&
      index >= 0 &&
      index === this.lastClickIndex &&
      timeStamp - this.lastClickTime < DOUBLE_CLICK_MS &&
      dx * dx + dy * dy < DOUBLE_CLICK_MAX_PX * DOUBLE_CLICK_MAX_PX;
    this.lastClickIndex = isDouble ? -1 : index;
    this.lastClickTime = timeStamp;
    this.lastClickX = clientX;
    this.lastClickY = clientY;
    if (index < 0) return; // empty space: keep the selection
    if (index !== this.selectedIndex) this.select(index);
    if (isDouble && this.selectedIndex === index) {
      this.lastTapFocusTime = timeStamp;
      this.focus();
    }
  }

  /**
   * Double-click on a satellite = focus it. Prevents the event so the rig skips
   * its dblclick zoom (which would otherwise shrink the fresh track range).
   * On empty space, or on the satellite that is already tracked (a natural
   * "zoom in"), the rig's zoom runs as usual.
   */
  private readonly onDblClick = (e: MouseEvent): void => {
    if (this.destroyed || e.button !== 0) return;
    if (e.timeStamp - this.lastTapFocusTime < DOUBLE_CLICK_MS) {
      e.preventDefault(); // the touch double-tap that just focused
      return;
    }
    const index = this.pickAt(e.clientX, e.clientY, CLICK_RADIUS_PX);
    if (index < 0) return;
    if (index === this.selectedIndex && this.tracking && this.opts.rig.mode === 'track') return;
    e.preventDefault();
    if (index !== this.selectedIndex) this.select(index);
    if (this.selectedIndex === index) this.focus();
  };

  private readonly onKeyDown = (e: KeyboardEvent): void => {
    if (e.key !== 'Escape' || e.defaultPrevented) return;
    const t = e.target;
    if (t instanceof HTMLElement && (t.isContentEditable || t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT')) {
      return;
    }
    if (this.selectedIndex >= 0) this.clear();
  };

  /** Frame-synced hover picking (after the frame the user is looking at). */
  private readonly onPostRender = (): void => {
    if (this.destroyed || !this.pointerInside || !this.pointerIsMouse || this.dragging) return;
    if (this.downId !== -1 && !this.hoverDirty) return;
    const wall = performance.now();
    if (!this.hoverDirty && wall - this.lastPickWall < HOVER_REPICK_MS) return;
    const moved = this.hoverDirty;
    this.hoverDirty = false;
    this.lastPickWall = wall;
    const index = this.pickAt(this.clientX, this.clientY, HOVER_RADIUS_PX);
    this.setHover(this.opts.getPoints(), index, moved);
  };

  private setHover(points: SatellitePoints | null, index: number, moved = false): void {
    const changed = index !== this.hovered;
    this.hovered = index;
    if (points && !points.isDestroyed() && points.hoveredIndex !== index) points.setHovered(index);

    const pointer = index >= 0;
    if (pointer !== this.cursorSet) {
      const style = this.canvas.style;
      if (pointer) {
        this.savedCursor = style.cursor === 'pointer' ? '' : style.cursor;
        style.cursor = 'pointer';
      } else {
        style.cursor = this.savedCursor;
      }
      this.cursorSet = pointer;
    }

    const label = this.opts.onHoverLabel;
    if (!label || (!changed && !(moved && pointer))) return;
    if (pointer) {
      // "NAME\nSHELL" (the HUD draws the second line smaller); the string is
      // built only when the hovered satellite changes, not on every move.
      if (changed || this.hoverText === '') {
        const catalog = this.opts.getCatalog();
        const name = catalog && index < catalog.count ? catalog.names[index] : undefined;
        const shell = catalog && index < catalog.count ? SHELLS[catalog.shell[index]!] : undefined;
        this.hoverText = shell ? `${name ?? `#${index}`}\n${shell.label}` : (name ?? `#${index}`);
      }
      label(this.hoverText, this.clientX, this.clientY);
    } else {
      this.hoverText = '';
      label(null, this.clientX, this.clientY);
    }
  }
}
