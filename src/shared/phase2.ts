/**
 * Contracts for the second build round: camera rig, sky, selection + orbit
 * path, and the SpaceX-style info panel. Type-only (plus a few constants), so
 * every module can import it without pulling in Cesium or the DOM.
 */

// ---------------------------------------------------------------------------
// Selected-satellite details (computed on the main thread from the satrec of
// ONE satellite with satellite.js; cheap enough to refresh ~4x per second)
// ---------------------------------------------------------------------------

/** Static facts about a satellite (change only when the element set changes). */
export interface SatStatic {
  index: number;
  name: string;
  noradId: number;
  /** COSPAR id formatted as "2019-074B" (from the TLE's "19074B"). */
  cosparId: string;
  /** Launch year (4 digits) and launch number of that year, from the COSPAR id. */
  launchYear: number;
  launchNumber: number;
  shellIndex: number;
  shellLabel: string;
  inclinationDeg: number;
  eccentricity: number;
  periodMin: number;
  /** Apogee / perigee altitude above the WGS-84 equatorial radius, km. */
  apogeeKm: number;
  perigeeKm: number;
  /** Element-set epoch (ms since Unix epoch). */
  epochMs: number;
}

/** Live state at one instant (refreshed while selected). */
export interface SatLive {
  timeMs: number;
  latDeg: number;
  lonDeg: number;
  /** Geodetic height above WGS-84, km. */
  altitudeKm: number;
  /** Inertial speed, km/s. */
  speedKmS: number;
  /** Ground speed of the sub-satellite point, km/h (optional nicety). */
  groundSpeedKmH: number;
  /** True when the satellite is in Earth's shadow (cylindrical shadow model is fine). */
  inShadow: boolean;
  /** Radius of the coverage footprint on the ground for 25° min elevation, km (phase 3 preview value). */
  footprintRadiusKm: number;
}

// ---------------------------------------------------------------------------
// Camera rig (replaces Cesium's ScreenSpaceCameraController)
// ---------------------------------------------------------------------------

export type CameraMode = 'globe' | 'track';

export interface CameraRigApi {
  readonly mode: CameraMode;
  /** Smooth flight to the constellation overview (used as the boot intro too). */
  flyToOverview(durationS?: number): Promise<void>;
  /**
   * Smoothly fly to and then orbit around a moving target. `getPosition`
   * writes the target's current ECEF position (metres) into `out` and returns
   * false if unavailable (the rig then returns to globe mode).
   */
  track(getPosition: (out: { x: number; y: number; z: number }) => boolean, opts?: { rangeM?: number; durationS?: number }): void;
  /** Leave track mode, flying back out to a comfortable globe view of the same spot. */
  release(durationS?: number): void;
  /** Called when the user starts interacting (drag/wheel/touch). */
  onUserInput: ((kind: 'drag' | 'zoom' | 'tilt') => void) | null;
  destroy(): void;
}

// ---------------------------------------------------------------------------
// Sky (NASA Milky Way cube + catalogue stars as GPU points)
// ---------------------------------------------------------------------------

export type SkyQuality = '2k' | '4k';

export interface SkyApi {
  readonly quality: SkyQuality | null;
  /** Resolves when the first (2k) cube map is showing. */
  readonly ready: Promise<void>;
  /** Re-evaluate whether to upgrade/downgrade (e.g. after AdaptiveQuality changes). */
  evaluate(): void;
  destroy(): void;
}

// ---------------------------------------------------------------------------
// Info panel (right column, SpaceX metric-table style)
// ---------------------------------------------------------------------------

export interface InfoPanelApi {
  /** Show the panel for a satellite (null hides it with a transition). */
  show(info: SatStatic | null): void;
  /** Update the live rows; must be cheap (only touches changed text nodes). */
  update(live: SatLive): void;
  /** Whether the camera is currently tracking the shown satellite (drives the FOCUS/RELEASE button label). */
  setTracking(tracking: boolean): void;
  onFocus: (() => void) | null;
  onRelease: (() => void) | null;
  onClose: (() => void) | null;
  /** Hardware rows: generation label (e.g. "V2 MINI") and laser terminal count (0 = none). Always marked as estimated. */
  setHardware(busLabel: string, laserTerminals: number): void;
  /** Modelled laser links of the shown satellite (empty = none); rows are clickable. */
  setLinks(links: readonly PanelLink[]): void;
  /** A neighbour row was clicked -> select that satellite. */
  onSelectNeighbor: ((index: number) => void) | null;
  /**
   * State of the COVERAGE AREA toggle row (UI only; does not call
   * onCoverageToggle). Global preference: kept across selections. Default ON.
   */
  setCoverageVisible(on: boolean): void;
  /** The user flipped COVERAGE AREA (the row already shows the new state optimistically). */
  onCoverageToggle: ((on: boolean) => void) | null;
}

export interface PanelLink {
  index: number;
  name: string;
  /** 0 = in-plane, 1 = cross-plane. */
  kind: number;
  distanceKm: number;
}

/** HUD controls for the laser-link overlay. */
export interface LaserHudInfo {
  enabled: boolean;
  /** Number of modelled links currently drawn. */
  links: number;
  /** Satellites with laser terminals (estimated). */
  equipped: number;
}

// ---------------------------------------------------------------------------
// Coverage hologram (selected satellite)
// ---------------------------------------------------------------------------

export interface CoverageApi {
  /**
   * Show the coverage hologram for this catalogue index (-1 = none).
   * `sameSatellite`: a catalogue hot swap re-binds the satellite already shown
   * (no fade-out / reveal replay).
   */
  setSatellite(index: number, sameSatellite?: boolean): void;
  setVisible(visible: boolean): void;
  readonly visible: boolean;
  destroy(): void;
}

/** Minimum elevation angle used for the coverage footprint (FCC filings). */
export const MIN_ELEVATION_DEG = 25;
