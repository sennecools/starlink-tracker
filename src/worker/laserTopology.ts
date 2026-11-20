/**
 * MODELLED optical inter-satellite link (laser) topology.
 *
 * SpaceX does not publish which Starlink satellites are linked to which, so
 * this computes a plausible "+Grid" mesh, the textbook layout for a
 * Walker-delta constellation with four terminals per satellite:
 *
 *   - in-plane links (kind 0): every satellite links to the nearest satellite
 *     ahead and behind in its own orbital plane (by argument of latitude);
 *   - cross-plane links (kind 1): to the nearest satellite in the adjacent
 *     plane on each side (east/west by RAAN), same shell.
 *
 * Plane identity comes from the orbit normal of the INERTIAL state (r x v in
 * TEME, rotated into Earth-fixed axes; see `propagateInto(..., outNormals)`).
 * Two satellites are co-planar when their normals are within 0.6 degree and
 * inclination + altitude match (same shell). Within one plane the normals
 * spread by less than ~0.35 deg, but the 97.3 deg polar shell has plane PAIRS
 * only ~0.85-0.95 deg apart (launch pairs such as 2025-179 / 2025-211), so a
 * 1 deg tolerance merged them and zig-zagged "in-plane" links between two
 * planes (which then could never be cross-linked).
 *
 * Constraints:
 *   - only satellites with laser terminals (catalog.lasers > 0) on an
 *     operational shell (not SHELL_TRANSIT) with a valid position;
 *   - degree cap = catalog.lasers[i]; candidate links are accepted greedily,
 *     in-plane first, then by distance; both endpoints must have a free terminal;
 *   - cross-plane links are dropped above |lat| 70 deg (fast relative angular
 *     motion near the orbit turnaround), except on polar shells where they are
 *     kept with a shorter distance cap (planes converge near the poles);
 *   - every link needs line of sight clearing the atmosphere (min altitude of
 *     the chord >= 80 km over a spherical Earth of equatorial radius).
 *
 * Complexity: satellites are partitioned into shell/altitude clusters; two
 * uniform-grid hashes keyed by cluster (orbit normals, coplanarMaxDeg cells; positions,
 * crossPlaneMaxKm/2 cells). The cross-plane pass sweeps occupied cell pairs
 * once (each satellite pair evaluated once, both ends updated) and culls cells
 * off the cluster's spherical shell. Candidates are ordered with a stable
 * counting sort. Roughly O(n * k) + O(n log n) for the grid build; ~14 ms for
 * 10k satellites in Node with optimised (JIT) code, topology only: the worker's
 * SGP4 propagation for the normals adds ~2-3 ms (tests/laserTopology.test.ts
 * prints the timing). Without the optimising JIT (or in a throttled background
 * renderer) the same work takes ~10x longer.
 * All scratch memory is module-level and reused; only the returned arrays are
 * allocated per call (they are transferred to the main thread).
 */
import type { SatRec } from 'satellite.js';
import type { SatCatalog } from '../shared/contracts.ts';
import { SHELL_TRANSIT } from '../shared/shells.ts';
import { propagateInto } from './propagate.ts';

export const LINK_IN_PLANE = 0;
export const LINK_CROSS_PLANE = 1;

export interface LaserTopologyOptions {
  /** Max angle between orbit normals for two satellites to share a plane. */
  coplanarMaxDeg: number;
  /** Max mean-inclination difference within one shell. */
  inclinationTolDeg: number;
  /** Max mean-altitude difference within one shell (separates e.g. 540 / 550 km shells). */
  altitudeTolKm: number;
  /** Max chord length of an in-plane link. */
  inPlaneMaxKm: number;
  /** Max chord length of a cross-plane link. */
  crossPlaneMaxKm: number;
  /** Max angle between orbit normals of two linked planes (adjacent planes only). */
  crossPlaneMaxNormalDeg: number;
  /** Cross-plane links are dropped when either end is above this |latitude|... */
  highLatitudeDeg: number;
  /** ...unless the shell's inclination is at least this (polar shell)... */
  polarInclinationDeg: number;
  /** ...in which case they are kept up to this chord length. */
  polarHighLatCrossMaxKm: number;
  /** Line of sight must clear this altitude over a spherical Earth. */
  minClearanceAltKm: number;
}

export const DEFAULT_LASER_TOPOLOGY_OPTIONS: Readonly<LaserTopologyOptions> = Object.freeze({
  coplanarMaxDeg: 0.6,
  inclinationTolDeg: 0.5,
  altitudeTolKm: 8,
  inPlaneMaxKm: 6000,
  crossPlaneMaxKm: 2500,
  crossPlaneMaxNormalDeg: 20,
  highLatitudeDeg: 70,
  polarInclinationDeg: 80,
  polarHighLatCrossMaxKm: 1200,
  minClearanceAltKm: 80,
});

/** The catalog fields the topology needs (a full SatCatalog satisfies it). */
export type TopologyCatalog = Pick<SatCatalog, 'count' | 'shell' | 'lasers' | 'inclinationDeg' | 'meanAltitudeKm'>;

export interface LaserTopologyInput {
  /** Earth-fixed positions, metres, count*3; NaN = invalid satellite. */
  positions: Float64Array;
  catalog: TopologyCatalog;
  timeMs: number;
  /**
   * Orbit-plane normals (count*3, any length, normalised here) in the SAME
   * axes as `positions`, from the inertial state. Preferred source; the
   * worker gets them for free from `propagateInto(..., outNormals)`.
   */
  normals?: Float64Array;
  /**
   * Alternative: INERTIAL velocity (m/s, count*3) expressed in the same axes
   * as `positions` (normal = r x v). An Earth-fixed velocity is NOT suitable.
   */
  velocities?: Float64Array;
  /** Last resort: SGP4 records, propagated to `timeMs` to get the normals. */
  satrecs?: readonly SatRec[];
  options?: Partial<LaserTopologyOptions>;
}

export interface LaserTopologyResult {
  /** 2 catalogue indices per link, a < b, no duplicates. */
  pairs: Uint32Array;
  /** One per link: LINK_IN_PLANE or LINK_CROSS_PLANE. */
  kinds: Uint8Array;
}

const DEG = Math.PI / 180;
const EARTH_RADIUS_KM = 6378.137;
/** Max per-satellite neighbour slots (lasers are capped to this). */
const MAX_DEG = 8;
const IDX_BITS = 2 ** 21; // candidate index field of the packed sort key

// ---------------------------------------------------------------------------
// Uniform grid hash over 3-D points, partitioned by a "cluster" id (satellites
// of different clusters never share a cell), as a sorted index plus an
// open-addressing table from cell key to the run of points in that cell.
// ---------------------------------------------------------------------------

/** Cell coordinates are stored as 8-bit unsigned with this offset (+-127 cells). */
const CELL_OFFSET = 128;
/** Cluster ids 0..MAX_CLUSTER (larger ids are merged into MAX_CLUSTER). */
const MAX_CLUSTER = 255;

class CellGrid {
  /** Point indices grouped by cell. */
  order = new Int32Array(0);
  /** End (exclusive) of the run returned by the last `find`. */
  end = 0;
  /** Occupied cells after `build`: count, cell keys and [start, end) runs in `order`. */
  runCount = 0;
  runKey = new Float64Array(0);
  runStart = new Int32Array(0);
  runEnd = new Int32Array(0);
  private sortKeys = new Float64Array(0);
  private tKey = new Float64Array(0);
  private tStart = new Int32Array(0);
  private tEnd = new Int32Array(0);
  private shift = 32;
  private mask = 15;

  /** Cell coordinate of a value (unclamped). */
  static cell(v: number, scale: number): number {
    return Math.floor(v * scale);
  }

  /** Packed cell key (< 2^32), or -1 when outside the representable range. */
  static key(cluster: number, cx: number, cy: number, cz: number): number {
    const x = cx + CELL_OFFSET;
    const y = cy + CELL_OFFSET;
    const z = cz + CELL_OFFSET;
    if (x < 0 || y < 0 || z < 0 || x > 255 || y > 255 || z > 255) return -1;
    return cluster * 16_777_216 + ((x << 16) | (y << 8) | z);
  }

  /**
   * Index the points `coords[i*3..]` with `include[i] !== 0` into cells of
   * size 1/scale, keyed by `cluster[i]`. Returns the number indexed.
   */
  build(count: number, coords: Float64Array, scale: number, include: Uint8Array, cluster: Uint8Array): number {
    if (this.sortKeys.length < count) {
      this.sortKeys = new Float64Array(count);
      this.order = new Int32Array(count);
      this.runKey = new Float64Array(count);
      this.runStart = new Int32Array(count);
      this.runEnd = new Int32Array(count);
    }
    const keys = this.sortKeys;
    let m = 0;
    for (let i = 0; i < count; i++) {
      if (include[i] === 0) continue;
      const o = i * 3;
      const k = CellGrid.key(
        cluster[i]!,
        CellGrid.cell(coords[o]!, scale),
        CellGrid.cell(coords[o + 1]!, scale),
        CellGrid.cell(coords[o + 2]!, scale),
      );
      if (k < 0) continue;
      keys[m++] = k * IDX_BITS + i;
    }
    const sorted = keys.subarray(0, m);
    sorted.sort();

    let size = 16;
    while (size < m * 2) size *= 2;
    if (this.tKey.length < size) {
      this.tKey = new Float64Array(size);
      this.tStart = new Int32Array(size);
      this.tEnd = new Int32Array(size);
    }
    this.shift = 32 - Math.log2(size);
    const mask = size - 1;
    this.mask = mask;
    const tKey = this.tKey;
    tKey.fill(-1, 0, size);

    const order = this.order;
    const runKeys = this.runKey;
    const runStart = this.runStart;
    const runEnd = this.runEnd;
    const tStart = this.tStart;
    const tEnd = this.tEnd;
    const shift = this.shift;
    let runs = 0;
    let runKey = -1;
    let slot = -1;
    for (let r = 0; r < m; r++) {
      const packed = sorted[r]!;
      const cellKey = Math.floor(packed / IDX_BITS);
      order[r] = packed - cellKey * IDX_BITS;
      if (cellKey !== runKey) {
        if (slot >= 0) {
          tEnd[slot] = r;
          runEnd[runs - 1] = r;
        }
        runKey = cellKey;
        slot = Math.imul(cellKey | 0, 0x9e3779b1) >>> shift;
        while (tKey[slot] !== -1) slot = (slot + 1) & mask;
        tKey[slot] = cellKey;
        tStart[slot] = r;
        runKeys[runs] = cellKey;
        runStart[runs] = r;
        runs++;
      }
    }
    if (slot >= 0) {
      tEnd[slot] = m;
      runEnd[runs - 1] = m;
    }
    this.runCount = runs;
    return m;
  }

  /** Start of the run of point indices in cell `key` (end in `this.end`), or -1. */
  find(key: number): number {
    if (key < 0) return -1;
    const tKey = this.tKey;
    let slot = Math.imul(key | 0, 0x9e3779b1) >>> this.shift;
    for (;;) {
      const k = tKey[slot]!;
      if (k === key) {
        this.end = this.tEnd[slot]!;
        return this.tStart[slot]!;
      }
      if (k === -1) return -1;
      slot = (slot + 1) & this.mask;
    }
  }
}


// ---------------------------------------------------------------------------
// Reused scratch
// ---------------------------------------------------------------------------

let capN = 0;
// Per satellite (catalogue order).
let sPos = new Float64Array(0); // km
let sNrm = new Float64Array(0); // unit orbit normals
let sInvR = new Float64Array(0); // 1 / |position| (1/km)
let sLayer = new Float64Array(0); // shell * LAYER_STRIDE + mean altitude (km)
let sOk = new Uint8Array(0);
let sCluster = new Uint8Array(0); // shell/altitude cluster (grid partition)
/** Squared radius band (km^2) of each cluster, for culling grid cells off the shell. */
const clusterRMin2 = new Float64Array(MAX_CLUSTER + 1);
const clusterRMax2 = new Float64Array(MAX_CLUSTER + 1);
let sHigh = new Uint8Array(0); // above highLatitudeDeg
let sCap = new Uint8Array(0);
let sDeg = new Uint8Array(0);
let sNbr = new Int32Array(0);
let sNrmIn = new Float64Array(0); // normals from satrecs
let sPosIn = new Float64Array(0); // positions from satrecs (discarded)
// Gathered in grid order (contiguous memory for the neighbour scans).
let gIdx = new Int32Array(0);
let gPos = new Float64Array(0);
let gNrm = new Float64Array(0);
let gInvR = new Float64Array(0);
let gLayer = new Float64Array(0);
let gInc = new Float64Array(0);
let gHigh = new Uint8Array(0);
let gCluster = new Uint8Array(0);
// Cross-plane running best per satellite (grid order): east / west side.
let xBestE = new Int32Array(0);
let xBestEAng = new Float64Array(0);
let xBestED = new Float64Array(0);
let xBestW = new Int32Array(0);
let xBestWAng = new Float64Array(0);
let xBestWD = new Float64Array(0);
// Candidate links.
let cI = new Int32Array(0);
let cJ = new Int32Array(0);
let cDist = new Float64Array(0);
let cKind = new Uint8Array(0);
let cKeys = new Float64Array(0);
let cOrder = new Int32Array(0);
let bucketCounts = new Int32Array(0);
let outPairs = new Uint32Array(0);
let outKinds = new Uint8Array(0);
const normalGrid = new CellGrid();
const positionGrid = new CellGrid();

/** Separates shells in the combined shell/altitude "layer" value. */
const LAYER_STRIDE = 1e5;
/** Position grid: cell = crossPlaneMaxKm / POS_CELLS, search +-POS_CELLS cells. */
const POS_CELLS = 2;

function ensureScratch(n: number): void {
  if (n <= capN) return;
  const c = Math.max(n, Math.ceil(capN * 1.25), 64);
  capN = c;
  sPos = new Float64Array(c * 3);
  sNrm = new Float64Array(c * 3);
  sInvR = new Float64Array(c);
  sLayer = new Float64Array(c);
  sOk = new Uint8Array(c);
  sCluster = new Uint8Array(c);
  sHigh = new Uint8Array(c);
  sCap = new Uint8Array(c);
  sDeg = new Uint8Array(c);
  sNbr = new Int32Array(c * MAX_DEG);
  gIdx = new Int32Array(c);
  gPos = new Float64Array(c * 3);
  gNrm = new Float64Array(c * 3);
  gInvR = new Float64Array(c);
  gLayer = new Float64Array(c);
  gInc = new Float64Array(c);
  gHigh = new Uint8Array(c);
  gCluster = new Uint8Array(c);
  xBestE = new Int32Array(c);
  xBestEAng = new Float64Array(c);
  xBestED = new Float64Array(c);
  xBestW = new Int32Array(c);
  xBestWAng = new Float64Array(c);
  xBestWD = new Float64Array(c);
  cI = new Int32Array(c * 4);
  cJ = new Int32Array(c * 4);
  cDist = new Float64Array(c * 4);
  cKind = new Uint8Array(c * 4);
  cKeys = new Float64Array(c * 4);
  cOrder = new Int32Array(c * 4);
  outPairs = new Uint32Array(c * MAX_DEG);
  outKinds = new Uint8Array(c * (MAX_DEG >> 1));
}

/** Copy per-satellite data into grid order (gIdx[k] = catalogue index). */
function gather(grid: CellGrid, m: number, inclinationDeg: Float32Array): void {
  const order = grid.order;
  const idx = gIdx;
  const pos = gPos;
  const nrm = gNrm;
  const invR = gInvR;
  const lay = gLayer;
  const incs = gInc;
  const high = gHigh;
  const clu = gCluster;
  const sp = sPos;
  const sn = sNrm;
  const sr = sInvR;
  const sl = sLayer;
  const sh = sHigh;
  const sc = sCluster;
  for (let k = 0; k < m; k++) {
    const i = order[k]!;
    idx[k] = i;
    const o = i * 3;
    const g = k * 3;
    pos[g] = sp[o]!;
    pos[g + 1] = sp[o + 1]!;
    pos[g + 2] = sp[o + 2]!;
    nrm[g] = sn[o]!;
    nrm[g + 1] = sn[o + 1]!;
    nrm[g + 2] = sn[o + 2]!;
    invR[k] = sr[i]!;
    lay[k] = sl[i]!;
    incs[k] = inclinationDeg[i]!;
    high[k] = sh[i]!;
    clu[k] = sc[i]!;
  }
}

/**
 * Partition eligible satellites into clusters of compatible shell + altitude:
 * sorted by layer value, a new cluster starts wherever the gap to the previous
 * satellite exceeds the altitude tolerance. Two satellites that can link are
 * therefore always in the same cluster, so the grids never mix clusters.
 * (Clusters beyond MAX_CLUSTER are merged, which only costs speed.)
 */
function assignClusters(n: number, altTol: number): void {
  const ok = sOk;
  const lay = sLayer;
  const out = sCluster;
  const keysAll = cKeys;
  let m = 0;
  for (let i = 0; i < n; i++) {
    if (ok[i] === 0) continue;
    // Layer in metres (integer, >= 0) in the high bits, index in the low bits.
    const lm = Math.max(0, Math.min(2 ** 31, Math.round(lay[i]! * 1000)));
    keysAll[m++] = lm * IDX_BITS + i;
  }
  const keys = keysAll.subarray(0, m);
  keys.sort();
  const invR = sInvR;
  const rMin = clusterRMin2;
  const rMax = clusterRMax2;
  rMin.fill(Infinity);
  rMax.fill(0);
  let cluster = 0;
  let prev = -Infinity;
  for (let r = 0; r < m; r++) {
    const packed = keys[r]!;
    const lm = Math.floor(packed / IDX_BITS);
    const i = packed - lm * IDX_BITS;
    // +1 m: never split a pair the (unrounded) layer check would accept.
    if (r > 0 && lm - prev > altTol * 1000 + 1 && cluster < MAX_CLUSTER) cluster++;
    prev = lm;
    out[i] = cluster;
    // Radius band with a 2 km margin.
    const rad = 1 / invR[i]!;
    const lo = (rad - 2) * (rad - 2);
    const hi = (rad + 2) * (rad + 2);
    if (lo < rMin[cluster]!) rMin[cluster] = lo;
    if (hi > rMax[cluster]!) rMax[cluster] = hi;
  }
}

/** Smallest distance (km) from Earth's centre to the segment a-b (km coordinates). */
function chordMinRadius(ax: number, ay: number, az: number, bx: number, by: number, bz: number): number {
  const dx = bx - ax;
  const dy = by - ay;
  const dz = bz - az;
  const dd = dx * dx + dy * dy + dz * dz;
  let t = dd > 0 ? -(ax * dx + ay * dy + az * dz) / dd : 0;
  if (t < 0) t = 0;
  else if (t > 1) t = 1;
  const px = ax + t * dx;
  const py = ay + t * dy;
  const pz = az + t * dz;
  return Math.sqrt(px * px + py * py + pz * pz);
}

/** Smallest squared coordinate inside [lo, lo + cs]. */
function axisMin2(lo: number, cs: number): number {
  const hi = lo + cs;
  if (lo <= 0 && hi >= 0) return 0;
  return lo > 0 ? lo * lo : hi * hi;
}

/** Largest squared coordinate inside [lo, lo + cs]. */
function axisMax2(lo: number, cs: number): number {
  const hi = lo + cs;
  return Math.max(lo * lo, hi * hi);
}

/** Derived thresholds shared by the passes. */
interface Thresholds {
  cosCoplanar: number;
  cosCrossMax: number;
  /** Plane-identity tolerance as a unit-sphere chord between normals. */
  planeTolRad: number;
  incTol: number;
  altTol: number;
  minR: number;
  inPlaneMax2: number;
  crossMax: number;
  polarCap2: number;
  sinHigh: number;
  polarInc: number;
}

// NOTE on structure: every hot loop lives in its own function and reads the
// module-level scratch through local consts. Reading mutable module `let`
// bindings inside a hot loop costs a context load (and hole check) per access
// and made the cross-plane pass ~3x slower in V8.

/** Fill sPos/sNrm/... for every satellite; returns the number eligible. */
function preparePass(
  n: number,
  positions: Float64Array,
  nrmSrc: Float64Array | null,
  velSrc: Float64Array | null,
  catalog: TopologyCatalog,
  t: Thresholds,
): number {
  const pos = sPos;
  const nrm = sNrm;
  const invR = sInvR;
  const layerOut = sLayer;
  const ok = sOk;
  const deg = sDeg;
  const capOut = sCap;
  const highOut = sHigh;
  const { shell, lasers, inclinationDeg, meanAltitudeKm } = catalog;
  const sinHigh = t.sinHigh;
  const polarInc = t.polarInc;
  let eligible = 0;
  for (let i = 0, o = 0; i < n; i++, o += 3) {
    ok[i] = 0;
    deg[i] = 0;
    const cap = lasers[i]!;
    if (cap === 0 || shell[i] === SHELL_TRANSIT) continue;
    const alt = meanAltitudeKm[i]!;
    const inc = inclinationDeg[i]!;
    if (!Number.isFinite(alt) || !Number.isFinite(inc)) continue;
    const x = positions[o]! / 1000;
    const y = positions[o + 1]! / 1000;
    const z = positions[o + 2]! / 1000;
    const r = Math.sqrt(x * x + y * y + z * z);
    if (!(r > EARTH_RADIUS_KM) || !Number.isFinite(r)) continue;
    let hx: number;
    let hy: number;
    let hz: number;
    if (nrmSrc !== null) {
      hx = nrmSrc[o]!;
      hy = nrmSrc[o + 1]!;
      hz = nrmSrc[o + 2]!;
    } else {
      const vx = velSrc![o]!;
      const vy = velSrc![o + 1]!;
      const vz = velSrc![o + 2]!;
      hx = y * vz - z * vy;
      hy = z * vx - x * vz;
      hz = x * vy - y * vx;
    }
    const hl = Math.sqrt(hx * hx + hy * hy + hz * hz);
    if (!(hl > 0) || !Number.isFinite(hl)) continue;
    pos[o] = x;
    pos[o + 1] = y;
    pos[o + 2] = z;
    nrm[o] = hx / hl;
    nrm[o + 1] = hy / hl;
    nrm[o + 2] = hz / hl;
    invR[i] = 1 / r;
    layerOut[i] = shell[i]! * LAYER_STRIDE + alt;
    ok[i] = 1;
    capOut[i] = cap > MAX_DEG ? MAX_DEG : cap;
    // 0 = below the cross-link latitude cut-off; above it: 1 on a polar shell
    // (cross links kept, shorter cap), 2 otherwise (no cross links).
    const above = Math.abs(z) > sinHigh * r;
    highOut[i] = !above ? 0 : inc >= polarInc ? 1 : 2;
    eligible++;
  }
  return eligible;
}

/**
 * In-plane candidates: for each satellite the nearest co-planar satellite
 * ahead and behind. Uses the normal grid (cell = chord of coplanarMaxDeg, so
 * every co-planar normal lies in the 27 surrounding cells). Data must be
 * gathered in normal-grid order. Returns the new candidate count.
 */
function inPlanePass(m: number, grid: CellGrid, nScale: number, t: Thresholds, nc0: number): number {
  const pos = gPos;
  const nrm = gNrm;
  const invR = gInvR;
  const lay = gLayer;
  const incs = gInc;
  const idx = gIdx;
  const clu = gCluster;
  const outI = cI;
  const outJ = cJ;
  const outD = cDist;
  const outK = cKind;
  const { cosCoplanar, incTol, altTol, minR, inPlaneMax2 } = t;
  let nc = nc0;
  for (let ki = 0; ki < m; ki++) {
    const g = ki * 3;
    const ax = pos[g]!;
    const ay = pos[g + 1]!;
    const az = pos[g + 2]!;
    const hx = nrm[g]!;
    const hy = nrm[g + 1]!;
    const hz = nrm[g + 2]!;
    const layer = lay[ki]!;
    const inc = incs[ki]!;
    const invRa = invR[ki]!;
    const cx = CellGrid.cell(hx, nScale);
    const cy = CellGrid.cell(hy, nScale);
    const cz = CellGrid.cell(hz, nScale);
    const cl = clu[ki]!;
    // Nearest ahead / behind by in-plane angle, compared via its cosine.
    let ahead = -1;
    let aheadCos = -2;
    let behind = -1;
    let behindCos = -2;
    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        for (let dz = -1; dz <= 1; dz++) {
          let k = grid.find(CellGrid.key(cl, cx + dx, cy + dy, cz + dz));
          if (k < 0) continue;
          const end = grid.end;
          for (; k < end; k++) {
            if (!(Math.abs(lay[k]! - layer) <= altTol) || k === ki) continue;
            const q = k * 3;
            if (hx * nrm[q]! + hy * nrm[q + 1]! + hz * nrm[q + 2]! < cosCoplanar) continue;
            if (!(Math.abs(incs[k]! - inc) <= incTol)) continue;
            const bx = pos[q]!;
            const by = pos[q + 1]!;
            const bz = pos[q + 2]!;
            // (a x b) . h > 0  <=>  b is ahead (motion direction is h x r).
            const sn = (ay * bz - az * by) * hx + (az * bx - ax * bz) * hy + (ax * by - ay * bx) * hz;
            const cs = (ax * bx + ay * by + az * bz) * invRa * invR[k]!;
            if (sn > 0) {
              if (cs > aheadCos || (cs === aheadCos && idx[k]! < idx[ahead]!)) {
                aheadCos = cs;
                ahead = k;
              }
            } else if (sn < 0) {
              if (cs > behindCos || (cs === behindCos && idx[k]! < idx[behind]!)) {
                behindCos = cs;
                behind = k;
              }
            }
          }
        }
      }
    }
    for (let side = 0; side < 2; side++) {
      const k = side === 0 ? ahead : behind;
      if (k < 0) continue;
      const q = k * 3;
      const bx = pos[q]!;
      const by = pos[q + 1]!;
      const bz = pos[q + 2]!;
      const ddx = bx - ax;
      const ddy = by - ay;
      const ddz = bz - az;
      const d2 = ddx * ddx + ddy * ddy + ddz * ddz;
      if (d2 > inPlaneMax2) continue;
      if (chordMinRadius(ax, ay, az, bx, by, bz) < minR) continue;
      outI[nc] = idx[ki]!;
      outJ[nc] = idx[k]!;
      outD[nc] = Math.sqrt(d2);
      outK[nc] = LINK_IN_PLANE;
      nc++;
    }
  }
  return nc;
}

/**
 * Neighbour cell offsets for the cross-plane sweep: the "forward" half of the
 * (2*POS_CELLS+1)^3 cube (lexicographically > 0), so each pair of cells is
 * visited once. Pairs inside one cell are handled separately.
 */
const HALF_OFFSETS: readonly number[] = (() => {
  const out: number[] = [];
  for (let dx = -POS_CELLS; dx <= POS_CELLS; dx++) {
    for (let dy = -POS_CELLS; dy <= POS_CELLS; dy++) {
      for (let dz = -POS_CELLS; dz <= POS_CELLS; dz++) {
        if (dx > 0 || (dx === 0 && (dy > 0 || (dy === 0 && dz > 0)))) out.push(dx, dy, dz);
      }
    }
  }
  return out;
})();

/**
 * Cross-plane candidates: for each satellite, on each side (east/west by
 * RAAN), the nearest satellite of the adjacent plane within reach.
 *
 * Sweep over occupied cells of the position grid (cell = crossMax /
 * POS_CELLS): every cell is paired with itself and the forward half of its
 * neighbour cells, and every satellite pair is evaluated ONCE, updating the
 * running best of both endpoints (the pair predicate is symmetric; only the
 * side flips). Neighbour cells that cannot reach the cluster's spherical shell
 * are culled without a lookup. Data must be gathered in position-grid order.
 * Returns the new candidate count.
 */
function crossPlanePass(m: number, grid: CellGrid, cellKm: number, t: Thresholds, nc0: number): number {
  const pos = gPos;
  const nrm = gNrm;
  const lay = gLayer;
  const incs = gInc;
  const high = gHigh;
  const idx = gIdx;
  const bE = xBestE;
  const bEA = xBestEAng;
  const bED = xBestED;
  const bW = xBestW;
  const bWA = xBestWAng;
  const bWD = xBestWD;
  const outI = cI;
  const outJ = cJ;
  const outD = cDist;
  const outK = cKind;
  const rMinAll = clusterRMin2;
  const rMaxAll = clusterRMax2;
  const runKey = grid.runKey;
  const runStart = grid.runStart;
  const runEnd = grid.runEnd;
  const runCount = grid.runCount;
  const offs = HALF_OFFSETS;
  const nOffs = offs.length;
  const { cosCoplanar, cosCrossMax, planeTolRad, incTol, altTol, minR, crossMax, polarCap2 } = t;
  const crossMax2 = crossMax * crossMax;

  for (let k = 0; k < m; k++) {
    bE[k] = -1;
    bEA[k] = Infinity;
    bED[k] = Infinity;
    bW[k] = -1;
    bWA[k] = Infinity;
    bWD[k] = Infinity;
  }

  for (let run = 0; run < runCount; run++) {
    const key = runKey[run]!;
    const cl = Math.floor(key / 16_777_216);
    const xyz = key - cl * 16_777_216;
    const cx = (xyz >>> 16) - CELL_OFFSET;
    const cy = ((xyz >>> 8) & 255) - CELL_OFFSET;
    const cz = (xyz & 255) - CELL_OFFSET;
    const rMin2 = rMinAll[cl]!;
    const rMax2 = rMaxAll[cl]!;
    const a0 = runStart[run]!;
    const a1 = runEnd[run]!;
    // o === -3: the cell with itself (pairs a < b); otherwise a forward neighbour.
    for (let o = -3; o < nOffs; o += 3) {
      let b0: number;
      let b1: number;
      if (o < 0) {
        b0 = a0;
        b1 = a1;
      } else {
        const nx = cx + offs[o]!;
        const ny = cy + offs[o + 1]!;
        const nz = cz + offs[o + 2]!;
        // Cull cells that do not intersect the cluster's radius band.
        const xLo = nx * cellKm;
        const yLo = ny * cellKm;
        const zLo = nz * cellKm;
        const min2 = axisMin2(xLo, cellKm) + axisMin2(yLo, cellKm) + axisMin2(zLo, cellKm);
        if (min2 > rMax2) continue;
        const max2 = axisMax2(xLo, cellKm) + axisMax2(yLo, cellKm) + axisMax2(zLo, cellKm);
        if (max2 < rMin2) continue;
        b0 = grid.find(CellGrid.key(cl, nx, ny, nz));
        if (b0 < 0) continue;
        b1 = grid.end;
      }
      for (let ka = a0; ka < a1; ka++) {
        const ha = high[ka]!;
        if (ha === 2) continue;
        const g = ka * 3;
        const ax = pos[g]!;
        const ay = pos[g + 1]!;
        const az = pos[g + 2]!;
        const hx = nrm[g]!;
        const hy = nrm[g + 1]!;
        const hz = nrm[g + 2]!;
        const layer = lay[ka]!;
        const inc = incs[ka]!;
        const kb0 = o < 0 ? ka + 1 : b0;
        for (let kb = kb0; kb < b1; kb++) {
          const q = kb * 3;
          const ddx = pos[q]! - ax;
          const ddy = pos[q + 1]! - ay;
          const ddz = pos[q + 2]! - az;
          const d2 = ddx * ddx + ddy * ddy + ddz * ddz;
          if (d2 > crossMax2) continue;
          const hb = high[kb]!;
          if (hb === 2 || ((ha === 1 || hb === 1) && d2 > polarCap2)) continue;
          if (!(Math.abs(lay[kb]! - layer) <= altTol)) continue;
          const jx = nrm[q]!;
          const jy = nrm[q + 1]!;
          const c = hx * jx + hy * jy + hz * nrm[q + 2]!;
          if (c >= cosCoplanar || c < cosCrossMax) continue;
          if (!(Math.abs(incs[kb]! - inc) <= incTol)) continue;
          // z of h_a x h_b = sin^2(i) sin(dRAAN): b lies east of a when > 0
          // (and then a lies west of b).
          const side = hx * jy - hy * jx;
          if (side === 0) continue;
          // Unit-sphere chord between the normals: monotonic in the angle and
          // within 0.5% of it up to 20 deg, without a libm acos call.
          const ang = Math.sqrt(2 - 2 * c);
          const d = Math.sqrt(d2);
          // Prefer the adjacent plane (smallest normal angle, within the
          // co-planar tolerance), then the nearest satellite in it.
          if (side > 0) {
            let best = bEA[ka]!;
            if (ang < best - planeTolRad || (ang <= best + planeTolRad && (d < bED[ka]! || (d === bED[ka]! && idx[kb]! < idx[bE[ka]!]!)))) {
              bE[ka] = kb;
              bEA[ka] = ang;
              bED[ka] = d;
            }
            best = bWA[kb]!;
            if (ang < best - planeTolRad || (ang <= best + planeTolRad && (d < bWD[kb]! || (d === bWD[kb]! && idx[ka]! < idx[bW[kb]!]!)))) {
              bW[kb] = ka;
              bWA[kb] = ang;
              bWD[kb] = d;
            }
          } else {
            let best = bWA[ka]!;
            if (ang < best - planeTolRad || (ang <= best + planeTolRad && (d < bWD[ka]! || (d === bWD[ka]! && idx[kb]! < idx[bW[ka]!]!)))) {
              bW[ka] = kb;
              bWA[ka] = ang;
              bWD[ka] = d;
            }
            best = bEA[kb]!;
            if (ang < best - planeTolRad || (ang <= best + planeTolRad && (d < bED[kb]! || (d === bED[kb]! && idx[ka]! < idx[bE[kb]!]!)))) {
              bE[kb] = ka;
              bEA[kb] = ang;
              bED[kb] = d;
            }
          }
        }
      }
    }
  }

  let nc = nc0;
  for (let ka = 0; ka < m; ka++) {
    const g = ka * 3;
    for (let s = 0; s < 2; s++) {
      const kb = s === 0 ? bE[ka]! : bW[ka]!;
      if (kb < 0) continue;
      const q = kb * 3;
      if (chordMinRadius(pos[g]!, pos[g + 1]!, pos[g + 2]!, pos[q]!, pos[q + 1]!, pos[q + 2]!) < minR) continue;
      outI[nc] = idx[ka]!;
      outJ[nc] = idx[kb]!;
      outD[nc] = s === 0 ? bED[ka]! : bWD[ka]!;
      outK[nc] = LINK_CROSS_PLANE;
      nc++;
    }
  }
  return nc;
}

/**
 * Greedy acceptance: in-plane first, then by distance; both endpoints need a
 * free terminal; duplicates (proposed from both ends) are skipped.
 * Returns the number of links written to outPairs/outKinds.
 */
function acceptPass(nc: number, maxDistKm: number): number {
  const candI = cI;
  const candJ = cJ;
  const candD = cDist;
  const candK = cKind;
  const deg = sDeg;
  const cap = sCap;
  const nbr = sNbr;
  const pairs = outPairs;
  const kinds = outKinds;
  // Stable counting sort by (kind, distance in 1 km buckets); ties keep
  // candidate order, so the result is deterministic. O(nc + buckets).
  const perKind = Math.ceil(maxDistKm) + 1;
  const nBuckets = perKind * 2;
  if (bucketCounts.length < nBuckets + 1) bucketCounts = new Int32Array(nBuckets + 1);
  const counts = bucketCounts;
  counts.fill(0, 0, nBuckets + 1);
  const keysAll = cKeys; // bucket per candidate (as float64 scratch)
  for (let c = 0; c < nc; c++) {
    let b = Math.floor(candD[c]!);
    if (b >= perKind) b = perKind - 1;
    const key = candK[c]! * perKind + b;
    keysAll[c] = key;
    counts[key + 1]!++;
  }
  for (let b = 0; b < nBuckets; b++) counts[b + 1]! += counts[b]!;
  const order = cOrder;
  for (let c = 0; c < nc; c++) order[counts[keysAll[c]!]!++] = c;
  let links = 0;
  for (let r = 0; r < nc; r++) {
    const c = order[r]!;
    const i = candI[c]!;
    const j = candJ[c]!;
    const a = i < j ? i : j;
    const b = i < j ? j : i;
    const da = deg[a]!;
    const db = deg[b]!;
    if (da >= cap[a]! || db >= cap[b]!) continue;
    const base = a * MAX_DEG;
    let dup = false;
    for (let s = 0; s < da; s++) {
      if (nbr[base + s] === b) {
        dup = true;
        break;
      }
    }
    if (dup) continue;
    nbr[base + da] = b;
    nbr[b * MAX_DEG + db] = a;
    deg[a] = da + 1;
    deg[b] = db + 1;
    pairs[links * 2] = a;
    pairs[links * 2 + 1] = b;
    kinds[links] = candK[c]!;
    links++;
  }
  return links;
}

/**
 * Compute the modelled laser links at one instant. See the module comment for
 * the model; the result is deterministic for a given input.
 */
export function computeLaserLinks(input: LaserTopologyInput): LaserTopologyResult {
  const opt: LaserTopologyOptions = { ...DEFAULT_LASER_TOPOLOGY_OPTIONS, ...input.options };
  const { positions, catalog } = input;
  const n = catalog.count;
  if (positions.length < n * 3) {
    throw new RangeError(`computeLaserLinks: positions has ${positions.length} slots, need ${n * 3}`);
  }
  if (n * 4 >= IDX_BITS) throw new RangeError(`computeLaserLinks: too many satellites (${n})`);
  ensureScratch(n);

  // --- orbit normals source -------------------------------------------------
  let nrmSrc: Float64Array | null = null;
  let velSrc: Float64Array | null = null;
  if (input.normals !== undefined) {
    nrmSrc = input.normals;
  } else if (input.velocities !== undefined) {
    velSrc = input.velocities;
  } else if (input.satrecs !== undefined) {
    if (input.satrecs.length !== n) throw new RangeError('computeLaserLinks: satrecs.length !== catalog.count');
    if (sNrmIn.length < n * 3) {
      sNrmIn = new Float64Array(capN * 3);
      sPosIn = new Float64Array(capN * 3);
    }
    propagateInto(input.satrecs, input.timeMs, sPosIn, sNrmIn);
    nrmSrc = sNrmIn;
  } else {
    throw new Error('computeLaserLinks: need normals, velocities or satrecs');
  }
  if ((nrmSrc ?? velSrc)!.length < n * 3) {
    throw new RangeError('computeLaserLinks: normals/velocities shorter than count*3');
  }

  const crossMax = opt.crossPlaneMaxKm;
  const polarCap = Math.min(opt.polarHighLatCrossMaxKm, crossMax);
  const t: Thresholds = {
    cosCoplanar: Math.cos(opt.coplanarMaxDeg * DEG),
    cosCrossMax: Math.cos(opt.crossPlaneMaxNormalDeg * DEG),
    planeTolRad: 2 * Math.sin((opt.coplanarMaxDeg * DEG) / 2),
    incTol: opt.inclinationTolDeg,
    altTol: opt.altitudeTolKm,
    minR: EARTH_RADIUS_KM + opt.minClearanceAltKm,
    inPlaneMax2: opt.inPlaneMaxKm * opt.inPlaneMaxKm,
    crossMax,
    polarCap2: polarCap * polarCap,
    sinHigh: Math.sin(opt.highLatitudeDeg * DEG),
    polarInc: opt.polarInclinationDeg,
  };

  const eligible = preparePass(n, positions, nrmSrc, velSrc, catalog, t);
  if (eligible < 2) return { pairs: new Uint32Array(0), kinds: new Uint8Array(0) };
  assignClusters(n, opt.altitudeTolKm);

  // Cell = chord of coplanarMaxDeg, clamped to the 8-bit cell range (+-127,
  // see CellGrid.key): below ~0.46 deg, normals near +-1 would otherwise fall
  // outside the grid and be dropped. A coarser cell still holds every
  // co-planar normal within the 27-cell search; cosCoplanar stays exact.
  const nScale = Math.min(1 / (2 * Math.sin((opt.coplanarMaxDeg * DEG) / 2)), CELL_OFFSET - 2);
  const mN = normalGrid.build(n, sNrm, nScale, sOk, sCluster);
  gather(normalGrid, mN, catalog.inclinationDeg);
  let nc = inPlanePass(mN, normalGrid, nScale, t, 0);

  const cellKm = crossMax / POS_CELLS;
  const mP = positionGrid.build(n, sPos, 1 / cellKm, sOk, sCluster);
  gather(positionGrid, mP, catalog.inclinationDeg);
  nc = crossPlanePass(mP, positionGrid, cellKm, t, nc);

  const links = acceptPass(nc, Math.max(opt.inPlaneMaxKm, opt.crossPlaneMaxKm));
  return { pairs: outPairs.slice(0, links * 2), kinds: outKinds.slice(0, links) };
}
