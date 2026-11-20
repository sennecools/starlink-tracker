/**
 * Shared contracts between the modules of the tracker.
 *
 *   server/celestrakPlugin.ts  --(HTTP /api/elements)-->  src/data/loadElements.ts
 *   src/data/loadElements.ts   --(ElementRecord[])---->   src/worker/PropagatorClient.ts
 *   src/worker/* (SGP4)        --(Keyframe)------------>  src/render/SatellitePoints.ts (GPU)
 *   everything                 --(plain values)-------->  src/ui/hud.ts
 *
 * Only plain data and type declarations live here so every module can import it
 * (browser main thread, Web Worker, Node dev server, Node tests).
 */

// ---------------------------------------------------------------------------
// Orbital element data (server -> client)
// ---------------------------------------------------------------------------

/**
 * OMM mean elements of one satellite, exactly in the shape satellite.js
 * `json2satrec` takes (CelesTrak OMM JSON): numbers as numbers, angles in
 * degrees, MEAN_MOTION in rev/day, EPOCH as ISO UTC without zone
 * ("2026-11-20T05:40:30.660672"). A type alias (not an interface) so it is
 * assignable to satellite.js' index-signature `OMMJsonObject`.
 */
export type OmmElements = {
  OBJECT_NAME: string;
  /** COSPAR id, e.g. "2026-256A" (may be blank for uncatalogued objects). */
  OBJECT_ID: string;
  EPOCH: string;
  MEAN_MOTION: number;
  ECCENTRICITY: number;
  INCLINATION: number;
  RA_OF_ASC_NODE: number;
  ARG_OF_PERICENTER: number;
  MEAN_ANOMALY: number;
  EPHEMERIS_TYPE?: 0;
  CLASSIFICATION_TYPE?: 'U' | 'C';
  NORAD_CAT_ID: number;
  ELEMENT_SET_NO: number;
  REV_AT_EPOCH?: number;
  BSTAR: number;
  MEAN_MOTION_DOT: number;
  MEAN_MOTION_DDOT: number;
};

/**
 * One satellite's orbital elements. Carries EITHER `omm` (preferred; OMM CSV
 * from CelesTrak, the only format that has NORAD ids above 99,999) OR the raw
 * TLE lines (legacy cache / bundled fixture). When both are present `omm`
 * wins. Use recordToSatrec (src/worker/propagate.ts) to initialise SGP4.
 */
export interface ElementRecord {
  /** e.g. "STARLINK-1008" (trimmed). */
  name: string;
  /** NORAD catalog number (may exceed 99,999). Alpha-5 TLE ids (e.g. "A0001") are decoded. */
  noradId: number;
  /** International designator in TLE style, e.g. "19074B" (from OMM OBJECT_ID "2019-074B"); "" when unknown. */
  intlDesignator: string;
  /** Raw TLE line 1 (69 chars), TLE-sourced records only. */
  line1?: string;
  /** Raw TLE line 2 (69 chars), TLE-sourced records only. */
  line2?: string;
  /** OMM mean elements, OMM-sourced records only. */
  omm?: OmmElements;
}

/** One upstream data set that contributes records to the served (merged) element set. */
export interface ElementSourceInfo {
  /** e.g. "CelesTrak GP (18 SDS)", "CelesTrak recent launches". */
  name: string;
  /** ISO time this source last contributed (download time; newest epoch for the bundled snapshot). */
  fetchedAt: string;
  /** Records in the served set that came from this source. */
  records: number;
  /** False when the source's last contribution was a cut-off download (or a seed / snapshot). */
  complete: boolean;
}

/**
 * Progress of the partitioned full-catalogue refresh: the full Starlink
 * catalogue is downloaded as a sequence of small name-partitioned requests
 * (one at a time); a cycle is complete once every partition succeeded.
 */
export interface ElementsCycleInfo {
  /** ISO start of the current (or last) cycle; null before the first one. */
  startedAt: string | null;
  /** ISO time every partition had been downloaded completely; null while the cycle is open. */
  completedAt: string | null;
  /** The cycle completed and produced a full catalogue. */
  complete: boolean;
  partitionsTotal: number;
  /** Partitions downloaded completely within this cycle. */
  partitionsDone: number;
  /** Partitions attempted in this cycle whose last attempt failed or was cut off (retried after their own window). */
  partitionsFailed: number;
  /** Partition being downloaded (or waited for) right now, e.g. "STARLINK-38". */
  current: string | null;
  /** A cycle runner is active (downloading, pausing between partitions or waiting for a window). */
  running: boolean;
  lastError: string | null;
}

/** Body of `GET /api/elements` with HTTP 200. */
export interface ElementsReady {
  status: 'ready';
  /** Human readable source, e.g. "CelesTrak SupGP (SpaceX ephemerides)". */
  source: string;
  /** ISO time the data was downloaded from upstream. */
  fetchedAt: string;
  /** True when this is an incomplete/bundled fallback data set. */
  partial: boolean;
  /** Earliest ISO time the server is allowed to contact upstream again. */
  nextRefreshEarliest: string;
  count: number;
  /** Contributing sources, most records first. Additive (absent from older servers). */
  sources?: ElementSourceInfo[];
  /** Partitioned full-catalogue refresh progress. Additive. */
  cycle?: ElementsCycleInfo | null;
  records: ElementRecord[];
}

/** Body of `GET /api/elements` with HTTP 503 (no data available yet). */
export interface ElementsWarming {
  status: 'warming';
  message: string;
  bytesReceived: number;
  recordsReceived: number;
  /** ISO time the current upstream download started, if any. */
  startedAt: string | null;
}

/** Body of `GET /api/elements/status` (always HTTP 200). */
export interface ElementsStatus {
  hasCache: boolean;
  cacheFetchedAt: string | null;
  cachePartial: boolean;
  cacheCount: number;
  fetching: boolean;
  bytesReceived: number;
  recordsReceived: number;
  lastAttemptAt: string | null;
  lastError: string | null;
  nextAllowedFetchAt: string | null;
  /** Contributing sources of the served set, most records first. Additive. */
  sources?: ElementSourceInfo[];
  /** Downloads in flight right now (one per upstream URL at most). Additive. */
  downloads?: { name: string; bytesReceived: number; recordsReceived: number; startedAt: string }[];
  /** Partitioned full-catalogue refresh progress. Additive. */
  cycle?: ElementsCycleInfo | null;
}

// ---------------------------------------------------------------------------
// Propagation worker protocol (main thread <-> Web Worker)
// ---------------------------------------------------------------------------

/**
 * Static per-satellite metadata, computed once when the worker parses the
 * elements. Index `i` in every array is the satellite's GPU index too.
 */
export interface SatCatalog {
  count: number;
  names: string[];
  noradIds: Int32Array;
  intlDesignators: string[];
  /** Element-set epoch, ms since Unix epoch. */
  epochMs: Float64Array;
  inclinationDeg: Float32Array;
  /** Mean altitude above a spherical Earth (R = 6378.137 km) from mean motion. */
  meanAltitudeKm: Float32Array;
  periodMin: Float32Array;
  /** Index into SHELLS (src/shared/shells.ts). */
  shell: Uint8Array;
  /** Estimated hardware generation, index into BUS_VERSIONS (src/shared/hardware.ts). */
  bus: Uint8Array;
  /** Estimated number of optical inter-satellite laser terminals (0 = none). */
  lasers: Uint8Array;
}

/**
 * A MODELLED laser-link topology (SpaceX does not publish the real one).
 * `pairs` holds 2 catalogue indices per link (a < b); `kinds` one entry per
 * link: 0 = in-plane (same orbital plane, ahead/behind), 1 = cross-plane.
 */
export interface LaserLinks {
  timeMs: number;
  pairs: Uint32Array;
  kinds: Uint8Array;
  /** Worker time of the whole request: propagateMs + topologyMs. */
  computeMs: number;
  /** SGP4 propagation (positions + orbit normals) part of computeMs. */
  propagateMs?: number;
  /** Topology (computeLaserLinks) part of computeMs. */
  topologyMs?: number;
}

/**
 * Positions of every satellite at one instant.
 * `positions` holds count*3 doubles: Earth-fixed (ECEF / ITRF-ish, the frame
 * Cesium's globe is rendered in) X,Y,Z in **metres**. A satellite whose
 * propagation failed (decayed, bad elements) has NaN in all three slots.
 */
export interface Keyframe {
  timeMs: number;
  positions: Float64Array;
}

export type ToWorker =
  | { type: 'init'; records: ElementRecord[] }
  /**
   * Compute a keyframe. `recycle` optionally hands back a previously received
   * positions buffer (transferred) so the worker can reuse it instead of
   * allocating.
   */
  | { type: 'keyframe'; requestId: number; timeMs: number; recycle?: Float64Array }
  /** Compute the modelled laser-link topology at `timeMs`. */
  | { type: 'links'; requestId: number; timeMs: number };

export type FromWorker =
  | { type: 'ready'; catalog: SatCatalog; rejected: number; parseMs: number }
  | { type: 'keyframe'; requestId: number; timeMs: number; positions: Float64Array; computeMs: number }
  | ({ type: 'links'; requestId: number } & LaserLinks)
  | { type: 'error'; requestId?: number; message: string };
