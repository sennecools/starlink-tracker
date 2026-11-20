/**
 * Per-satellite neighbour lists of the MODELLED laser-link topology
 * (LaserLinks from the worker; SpaceX does not publish the real one).
 *
 * Built once per links update (every ~10 s of sim time) as a CSR structure,
 * so "who is linked to satellite i" is O(degree) instead of an O(links) scan:
 *
 *   neighbours of i = neighbors[offsets[i] .. offsets[i + 1])
 *   kind of each    = kinds[same range]   (0 = in-plane, 1 = cross-plane)
 *
 * Within one satellite the in-plane links come first, then the cross-plane
 * ones, each by catalogue index, so the order depends only on the neighbour
 * set and is stable between updates while the topology is unchanged.
 *
 * Pure (no Cesium / DOM) so Node tests can load it. Buffers of a previous
 * adjacency are reused when they are large enough.
 */

export interface LinkAdjacency {
  /** Satellite count the adjacency was built for (catalog.count). */
  count: number;
  /** count + 1 entries; offsets[count] = number of directed entries. */
  offsets: Int32Array;
  neighbors: Int32Array;
  kinds: Uint8Array;
  /** Number of (undirected) links accepted. */
  links: number;
  /** Sim time the topology was computed for (ms since Unix epoch). */
  timeMs: number;
  /** Incremented on every rebuild (cheap change detection). */
  version: number;
}

/**
 * Builds (or rebuilds into `reuse`) the adjacency of `count` satellites from
 * `pairs` (2 indices per link) and `kinds` (one per link). Pairs with an index
 * outside [0, count) or identical ends are skipped.
 */
export function buildLinkAdjacency(
  count: number,
  pairs: ArrayLike<number>,
  kinds: ArrayLike<number>,
  timeMs: number,
  reuse?: LinkAdjacency | null,
): LinkAdjacency {
  const n = Math.max(0, Math.floor(count));
  const links = Math.min(kinds.length, pairs.length >> 1);
  const offsets = reuse && reuse.offsets.length === n + 1 ? reuse.offsets : new Int32Array(n + 1);
  offsets.fill(0);

  // Degrees (offsets[i + 1] counts satellite i), only valid pairs.
  let valid = 0;
  for (let k = 0; k < links; k++) {
    const a = pairs[2 * k];
    const b = pairs[2 * k + 1];
    if (!(a >= 0 && a < n && b >= 0 && b < n) || a === b) continue;
    offsets[a + 1]++;
    offsets[b + 1]++;
    valid++;
  }
  for (let i = 0; i < n; i++) offsets[i + 1] += offsets[i];

  const total = offsets[n];
  const neighbors = reuse && reuse.neighbors.length >= total ? reuse.neighbors : new Int32Array(Math.ceil(total * 1.25) + 16);
  const kindOut = reuse && reuse.kinds.length >= total ? reuse.kinds : new Uint8Array(neighbors.length);

  // Fill: in-plane pass first, then cross-plane, using offsets[i] as cursor
  // (restored afterwards by shifting).
  for (let pass = 0; pass < 2; pass++) {
    for (let k = 0; k < links; k++) {
      const kind = kinds[k] === 1 ? 1 : 0;
      if (kind !== pass) continue;
      const a = pairs[2 * k];
      const b = pairs[2 * k + 1];
      if (!(a >= 0 && a < n && b >= 0 && b < n) || a === b) continue;
      const ia = offsets[a]++;
      neighbors[ia] = b;
      kindOut[ia] = kind;
      const ib = offsets[b]++;
      neighbors[ib] = a;
      kindOut[ib] = kind;
    }
  }
  // offsets[i] now holds the end of i = start of i + 1: shift right by one.
  for (let i = n; i > 0; i--) offsets[i] = offsets[i - 1];
  offsets[0] = 0;

  // Order each range by (kind, neighbour index) so it depends only on the
  // neighbour set (the worker's acceptance order flips with distances).
  // Insertion sort: the degree is at most a handful.
  for (let i = 0; i < n; i++) {
    const s = offsets[i];
    const e = offsets[i + 1];
    for (let p = s + 1; p < e; p++) {
      const nb = neighbors[p];
      const kd = kindOut[p];
      let q = p - 1;
      while (q >= s && (kindOut[q] > kd || (kindOut[q] === kd && neighbors[q] > nb))) {
        neighbors[q + 1] = neighbors[q];
        kindOut[q + 1] = kindOut[q];
        q--;
      }
      neighbors[q + 1] = nb;
      kindOut[q + 1] = kd;
    }
  }

  const out = reuse ?? ({} as LinkAdjacency);
  out.count = n;
  out.offsets = offsets;
  out.neighbors = neighbors;
  out.kinds = kindOut;
  out.links = valid;
  out.timeMs = timeMs;
  out.version = (reuse?.version ?? 0) + 1;
  return out;
}

/** Number of satellites with at least one laser terminal (estimated hardware). */
export function countLaserEquipped(lasers: ArrayLike<number>, count: number): number {
  let n = 0;
  const m = Math.min(count, lasers.length);
  for (let i = 0; i < m; i++) if (lasers[i] > 0) n++;
  return n;
}
