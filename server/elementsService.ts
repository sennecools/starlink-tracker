/**
 * Orbital-element service for the dev / preview server.
 *
 * Framework agnostic (plain Node `http` req/res + connect-style `next`), so it
 * is testable without Vite. Responsibilities:
 *
 *  - Serve `GET /api/elements` (ElementsReady | ElementsWarming) and
 *    `GET /api/elements/status` (ElementsStatus).
 *  - Download OMM CSV from CelesTrak (TLE cannot carry NORAD ids > 99,999, so
 *    every launch since late 2025, incl. all Starlink V3s, is OMM-only):
 *      primary     GP NAME=STARLINK-1 ... NAME=STARLINK-39 (18 name partitions;
 *                  "partitioned primary", see below; STARLINK rows only)
 *      supplement  GP GROUP=last-30-days    (small; STARLINK rows only; own window)
 *      fallback    SupGP FILE=starlink      (only while the primary cannot deliver
 *                                             and there is no usable full set)
 *    GROUP=starlink is no longer used: CelesTrak cuts responses off after
 *    ~1.34 MB, so the (id-sorted) full group never delivers the newest objects.
 *  - Partitioned primary: a "full refresh cycle" downloads the partitions
 *    SEQUENTIALLY in the background (one request at a time, a short pause in
 *    between), each on its own per-URL window, merging each one as it lands.
 *    A cut-off / failed partition keeps what it delivered and is retried only
 *    after its own window; the others go on. A partition without matches
 *    (CelesTrak: HTTP 404 "No GP data found") is a valid empty result. The cycle is complete (full set,
 *    partial = false) once every partition succeeded completely within it. A
 *    new cycle starts when the oldest partition success of the current one is
 *    a window old. Cycle state lives in meta.json (survives restarts); at most
 *    one cycle runner per cache dir per process (global registry).
 *  - MERGE, never replace: the served set is the union by NORAD id of the
 *    current set and every parsed download (complete or cut off), newest
 *    element-set epoch wins, records > 30 days older than the newest epoch are
 *    dropped. Persisted atomically as <rootDir>/.cache/elements.json.
 *  - Migration: the old TLE cache (.cache/starlink.tle) seeds the merged set
 *    until the first OMM download lands.
 *  - Contact each upstream URL at most ONCE PER WINDOW (default 2 h + 1 min),
 *    persisted in meta.json *before* the request is sent, so failures,
 *    timeouts, aborts and restarts can never cause an extra upstream hit.
 *  - Fall back to a bundled partial fixture so the app always has data.
 */
import * as fs from 'node:fs';
import * as fsp from 'node:fs/promises';
import * as path from 'node:path';
import * as zlib from 'node:zlib';
import type { IncomingMessage, ServerResponse } from 'node:http';
import type {
  ElementRecord,
  ElementsCycleInfo,
  ElementSourceInfo,
  ElementsReady,
  ElementsStatus,
  ElementsWarming,
} from '../src/shared/contracts.ts';
import { filterStarlink, isNoDataResponse, looksLikeOmmCsv, parseElementsText, recordEpochMs } from './omm.ts';

// ---------------------------------------------------------------------------
// Public constants / types
// ---------------------------------------------------------------------------

/**
 * primary     full catalogue. Every 'primary' upstream is one PARTITION of it
 *             (in list order); a cycle in which all of them succeeded makes
 *             the set non-partial. A single primary is a one-partition cycle.
 * supplement  small feed fetched on its own window, independent of the others.
 * fallback    tried (in order) only while the primary cannot deliver and
 *             there is no reasonably recent full download.
 */
export type UpstreamRole = 'primary' | 'supplement' | 'fallback';

export interface Upstream {
  name: string;
  url: string;
  /** Default: the first upstream is 'primary', every other one 'fallback'. */
  role?: UpstreamRole;
  /** Keep only records whose name starts with STARLINK. */
  starlinkOnly?: boolean;
  /**
   * The feed may use placeholder catalog numbers for not-yet-catalogued
   * objects (CelesTrak SupGP). Such a record never duplicates a real object
   * with the same name: it is skipped, and replaced once the real id arrives.
   */
  placeholderIds?: boolean;
  /** Name of the merged source the records are filed under (default `name`); the primary partitions share one. */
  sourceName?: string;
  /** Short label of a primary partition for progress displays (default `name`), e.g. "STARLINK-38". */
  partition?: string;
  /**
   * Minimum parsed record count for a complete download. Default: the
   * service's `minRecords` for a fallback or a lone primary, 0 for one of
   * several primary partitions (an empty partition / "No GP data found" is a
   * valid result; the cycle as a whole must reach `minRecords`).
   */
  minRecords?: number;
}

/** Merged-source name of the partitioned GP catalogue (same as the former GROUP=starlink source). */
export const PRIMARY_SOURCE = 'CelesTrak GP (18 SDS)';

/**
 * CelesTrak NAME= queries (substring match on OBJECT_NAME) that together
 * cover every STARLINK-<digits> name: 4-digit names STARLINK-1007..6xxx
 * (2019-2023; -3xxx are covered by -30..-39) and 5-digit names
 * STARLINK-30000..40xxx (2023+). Each answer is ~100 KB, far below the
 * ~1.34 MB at which CelesTrak cuts a response off. Overlaps are harmless
 * (the merge dedups by NORAD id); non-STARLINK matches are filtered out.
 */
export const STARLINK_PARTITIONS: readonly string[] = [
  ...[1, 2, 4, 5, 6, 7, 8, 9].map((d) => `STARLINK-${d}`),
  ...Array.from({ length: 10 }, (_, i) => `STARLINK-3${i}`),
];

const gpCsvUrl = (query: string): string => `https://celestrak.org/NORAD/elements/gp.php?${query}&FORMAT=csv`;

/** Upstreams (OMM CSV). Order = priority (partitions: download order; fallbacks: try order). */
export const DEFAULT_UPSTREAMS: readonly Upstream[] = [
  ...STARLINK_PARTITIONS.map(
    (p): Upstream => ({
      name: `CelesTrak GP NAME=${p}`,
      url: gpCsvUrl(`NAME=${p}`),
      role: 'primary',
      starlinkOnly: true,
      sourceName: PRIMARY_SOURCE,
      partition: p,
    }),
  ),
  {
    name: 'CelesTrak recent launches',
    url: 'https://celestrak.org/NORAD/elements/gp.php?GROUP=last-30-days&FORMAT=csv',
    role: 'supplement',
    starlinkOnly: true,
  },
  {
    name: 'CelesTrak SupGP (SpaceX ephemerides)',
    url: 'https://celestrak.org/NORAD/elements/supplemental/sup-gp.php?FILE=starlink&FORMAT=csv',
    role: 'fallback',
    placeholderIds: true,
  },
];

/**
 * TLE upstreams used before the OMM switch. They stay in meta.json as
 * historical entries and are never contacted again.
 */
export const LEGACY_TLE_URLS: readonly string[] = [
  'https://celestrak.org/NORAD/elements/supplemental/sup-gp.php?FILE=starlink&FORMAT=tle',
  'https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle',
];

/**
 * Every former upstream that stays in meta.json as a historical entry and is
 * never contacted again: the TLE URLs and the single GROUP=starlink CSV
 * (replaced by the name partitions because CelesTrak cuts it off).
 */
export const LEGACY_URLS: readonly string[] = [...LEGACY_TLE_URLS, gpCsvUrl('GROUP=starlink')];

/** CelesTrak allows one download per data set per 2 h; 1 min safety margin. */
export const DEFAULT_MIN_INTERVAL_MS = 2 * 3_600_000 + 60_000;
export const DEFAULT_REQUEST_TIMEOUT_MS = 30 * 60_000;
/** Politeness pause between two partition downloads of a cycle. */
export const DEFAULT_PARTITION_PAUSE_MS = 2_000;
/** A cycle runner waits at most this long for the next partition's window to open; beyond that it ends (resumed by a later refreshIfDue). */
export const DEFAULT_MAX_PARTITION_WAIT_MS = 15 * 60_000;
/** Minimum parsed record count for a complete primary / fallback download to count as a full catalogue. */
export const DEFAULT_MIN_RECORDS = 1000;
export const USER_AGENT = 'starlink-tracker-dev/0.1';
export const FIXTURE_SOURCE = 'Bundled partial snapshot';
/** Source name of records carried over from the pre-OMM TLE cache. */
export const SEED_SOURCE = 'TLE cache (migrated)';
/** Merged element set (JSON, see StoredDataset). */
export const CACHE_FILE = 'elements.json';
/** Pre-OMM TLE cache; read once as migration seed while CACHE_FILE does not exist. */
export const LEGACY_CACHE_FILE = 'starlink.tle';
export const META_FILE = 'meta.json';
/** Records whose epoch is this much older than the newest epoch in the set are dropped (decayed / stale). */
export const STALE_RECORD_MS = 30 * 86_400_000;

/** 2 h lock + 1 min margin after the timestamp in a CelesTrak 403 body. */
const LOCK_AFTER_TIMESTAMP_MS = 2 * 3_600_000 + 60_000;
/** Never retry a locked URL sooner than this, whatever the 403 body says. */
const MIN_LOCK_RETRY_MS = 5 * 60_000;
/** Longest legitimate lock (a 403 lock is capped at 24 h); anything further ahead was written under a wrong clock. */
const MAX_LOCK_AHEAD_MS = 24 * 3_600_000;
/** Clock skew tolerated before a persisted timestamp counts as "in the future". */
const FUTURE_TOLERANCE_MS = 5 * 60_000;
/** A full download older than this no longer blocks fallback to the next upstream. */
const STALE_FALLBACK_MS = 12 * 3_600_000;
/** Epochs further than this in the future are ignored when finding the newest epoch (bad data). */
const EPOCH_FUTURE_LIMIT_MS = 7 * 86_400_000;
const MAX_BODY_BYTES = 64 * 1024 * 1024;
const ERROR_BODY_LIMIT = 16 * 1024;

export interface ElementsServiceOptions {
  /** Project root; cache defaults to <rootDir>/.cache, fixture to <rootDir>/fixtures/... */
  rootDir: string;
  cacheDir?: string;
  /** Fallback data set (TLE or OMM CSV) when no cache exists. Default <rootDir>/fixtures/sup-gp-starlink-partial.tle */
  fixturePath?: string;
  /** Default DEFAULT_UPSTREAMS. */
  upstreams?: Upstream[];
  /** Minimum time between two requests to the same URL. Default 2 h + 1 min. */
  minIntervalMs?: number;
  /** Whole-download timeout (headers + body). Default 30 min. */
  requestTimeoutMs?: number;
  /** Disable all upstream contact. Default: env CELESTRAK_OFFLINE === '1'. */
  offline?: boolean;
  fetchImpl?: typeof fetch;
  now?: () => number;
  /** Unprefixed log lines (the Vite plugin adds "[elements] "). */
  log?: (msg: string) => void;
  /** Minimum parsed record count for a complete fallback / lone-primary download and for a complete partition cycle. Default 1000. */
  minRecords?: number;
  /** Pause between two partition downloads of a cycle. Default 2 s. */
  partitionPauseMs?: number;
  /** Longest wait of a cycle runner for the next partition's window. Default 15 min. */
  maxPartitionWaitMs?: number;
}

export type NextFunction = (err?: unknown) => void;

export interface ElementsService {
  /** Connect-style middleware: handles /api/elements and /api/elements/status, else calls next(). */
  handle(req: IncomingMessage, res: ServerResponse, next: NextFunction): void;
  status(): ElementsStatus;
  /** Start the background upstream downloads that are due and allowed. Never blocks. */
  refreshIfDue(): void;
  /**
   * Stop serving. By default in-flight downloads are NOT aborted: they keep
   * running and merge into the disk cache (an aborted download still counts
   * against the CelesTrak quota, so throwing it away on a Vite restart would
   * waste the 2 h window). A new service on the same cacheDir in the same
   * process picks the downloads' progress and results up. This service's
   * partition cycle stops after the partition in flight and is resumed by
   * the newer service (or from meta.json after a process restart).
   */
  close(opts?: { abortDownload?: boolean }): void;
  /** Resolves once no download and no partition cycle is in flight for this cache dir (tests / tooling). */
  whenIdle(): Promise<void>;
}

// ---------------------------------------------------------------------------
// Merged data set (elements.json) + pure merge
// ---------------------------------------------------------------------------

/** A source as stored with the merged set. */
export interface MergeSource {
  name: string;
  url: string | null;
  /** ISO time of this source's last contribution. */
  fetchedAt: string;
  /** Last contribution was a complete download. */
  complete: boolean;
  placeholderIds: boolean;
}

export interface MergeSet {
  records: ElementRecord[];
  /** Per record: index into `sources`. */
  origin: number[];
  sources: MergeSource[];
}

export interface MergeResult extends MergeSet {
  /** Records per source (same index as `sources`). */
  counts: number[];
  /** New NORAD ids. */
  added: number;
  /** Existing ids replaced by a newer epoch. */
  updated: number;
  /** Stale (or unusable) records removed from the set. */
  dropped: number;
  /** Placeholder-id duplicates skipped or removed. */
  duplicates: number;
  /** Incoming records without a usable epoch / id. */
  invalid: number;
}

interface Entry {
  rec: ElementRecord;
  origin: number;
  epoch: number;
}

/**
 * Union `incoming` (from `source`) into `base` by NORAD id. Pure.
 *  - Newest epoch wins; on an equal epoch the incoming record wins.
 *  - Records more than `staleMs` older than the newest epoch in the result are
 *    dropped (epochs more than 7 days after `nowMs` are ignored when finding
 *    the newest one).
 *  - Placeholder ids: an incoming record from a `placeholderIds` source whose
 *    id is unknown but whose name (and compatible COSPAR id) matches a record
 *    from a regular source is skipped; a regular record removes a
 *    placeholder-sourced record with the same name and a different id (also
 *    when its own id is already known), and so does a placeholder-source
 *    record re-listing the object under a new (e.g. real) id.
 *  - Output sorted by NORAD id; sources without records are pruned.
 */
export function mergeElementSets(
  base: MergeSet,
  incoming: readonly ElementRecord[],
  source: MergeSource,
  nowMs: number,
  staleMs: number = STALE_RECORD_MS,
): MergeResult {
  const sources = base.sources.map((s) => ({ ...s }));
  let srcIdx = sources.findIndex((s) => s.name === source.name);
  if (srcIdx === -1) {
    srcIdx = sources.length;
    sources.push({ ...source });
  } else sources[srcIdx] = { ...source };

  const byId = new Map<number, Entry>();
  const byName = new Map<string, Set<number>>();
  const nameKey = (r: ElementRecord): string => r.name.trim().toUpperCase();
  const index = (id: number, e: Entry): void => {
    const prev = byId.get(id);
    if (prev) byName.get(nameKey(prev.rec))?.delete(id);
    byId.set(id, e);
    const k = nameKey(e.rec);
    let set = byName.get(k);
    if (!set) byName.set(k, (set = new Set()));
    set.add(id);
  };
  const remove = (id: number): void => {
    const prev = byId.get(id);
    if (!prev) return;
    byName.get(nameKey(prev.rec))?.delete(id);
    byId.delete(id);
  };
  const isPlaceholder = (origin: number): boolean => sources[origin]?.placeholderIds === true;
  const intlCompatible = (a: string, b: string): boolean => a === '' || b === '' || a === b;
  /**
   * Remove placeholder-sourced records named like `rec` (compatible COSPAR id,
   * other id). With `notNewerThan` (a placeholder feed) only twins whose epoch
   * is not newer are removed, so two placeholder rows of one file never wipe
   * each other out. Returns how many were removed.
   */
  const dropPlaceholderTwins = (rec: ElementRecord, notNewerThan = Infinity): number => {
    let n = 0;
    for (const other of [...(byName.get(nameKey(rec)) ?? [])]) {
      if (other === rec.noradId) continue;
      const e = byId.get(other)!;
      if (isPlaceholder(e.origin) && e.epoch <= notNewerThan && intlCompatible(e.rec.intlDesignator, rec.intlDesignator)) {
        remove(other);
        n++;
      }
    }
    return n;
  };

  let dropped = 0;
  for (let i = 0; i < base.records.length; i++) {
    const rec = base.records[i]!;
    const epoch = recordEpochMs(rec);
    const origin = base.origin[i] ?? 0;
    if (!Number.isFinite(epoch) || !(rec.noradId > 0) || origin < 0 || origin >= sources.length) {
      dropped++;
      continue;
    }
    const prev = byId.get(rec.noradId);
    if (!prev || epoch > prev.epoch) index(rec.noradId, { rec, origin, epoch });
  }

  let added = 0;
  let updated = 0;
  let duplicates = 0;
  let invalid = 0;
  for (const rec of incoming) {
    const epoch = recordEpochMs(rec);
    const id = rec.noradId;
    if (!Number.isFinite(epoch) || !(Number.isInteger(id) && id > 0)) {
      invalid++;
      continue;
    }
    const existing = byId.get(id);
    if (!existing) {
      if (source.placeholderIds) {
        let dup = false;
        for (const other of byName.get(nameKey(rec)) ?? []) {
          const e = byId.get(other)!;
          if (!isPlaceholder(e.origin) && intlCompatible(e.rec.intlDesignator, rec.intlDesignator)) dup = true;
        }
        if (dup) {
          duplicates++;
          continue;
        }
      }
      // A regular record, or a placeholder feed re-listing the object under
      // another (e.g. its real) id, supersedes older placeholder twins.
      duplicates += dropPlaceholderTwins(rec, source.placeholderIds ? epoch : Infinity);
      index(id, { rec, origin: srcIdx, epoch });
      added++;
    } else {
      if (epoch >= existing.epoch) {
        if (epoch > existing.epoch) updated++;
        index(id, { rec, origin: srcIdx, epoch });
      }
      // A regular source confirms the id is real: placeholder twins left over
      // from an earlier placeholder -> real-id transition go away.
      if (!source.placeholderIds) duplicates += dropPlaceholderTwins(rec);
    }
  }

  // Stale filter relative to the newest plausible epoch.
  const horizon = nowMs + EPOCH_FUTURE_LIMIT_MS;
  let newest = -Infinity;
  let newestAny = -Infinity;
  for (const e of byId.values()) {
    if (e.epoch > newestAny) newestAny = e.epoch;
    if (e.epoch <= horizon && e.epoch > newest) newest = e.epoch;
  }
  if (!Number.isFinite(newest)) newest = newestAny;
  const cutoff = newest - staleMs;

  const ids = [...byId.keys()].sort((a, b) => a - b);
  const kept: Entry[] = [];
  for (const id of ids) {
    const e = byId.get(id)!;
    if (e.epoch < cutoff) dropped++;
    else kept.push(e);
  }

  // Prune sources without records, remap origins.
  const rawCounts = new Array<number>(sources.length).fill(0);
  for (const e of kept) rawCounts[e.origin]!++;
  const remap = new Array<number>(sources.length).fill(-1);
  const outSources: MergeSource[] = [];
  const counts: number[] = [];
  for (let i = 0; i < sources.length; i++) {
    if (rawCounts[i]! === 0) continue;
    remap[i] = outSources.length;
    outSources.push(sources[i]!);
    counts.push(rawCounts[i]!);
  }
  return {
    records: kept.map((e) => e.rec),
    origin: kept.map((e) => remap[e.origin]!),
    sources: outSources,
    counts,
    added,
    updated,
    dropped,
    duplicates,
    invalid,
  };
}

/** On-disk format of CACHE_FILE. */
interface StoredDataset extends MergeSet {
  format: 'starlink-tracker/elements';
  version: 1;
  /** ISO time of the last merge. */
  fetchedAt: string;
  /** ISO time of the last complete primary / fallback download; null = partial. */
  fullCompletedAt: string | null;
}

function isValidRecord(r: unknown): r is ElementRecord {
  if (!r || typeof r !== 'object') return false;
  const o = r as Partial<ElementRecord>;
  if (typeof o.name !== 'string' || typeof o.noradId !== 'number' || typeof o.intlDesignator !== 'string') return false;
  if (o.omm !== undefined) return typeof o.omm === 'object' && o.omm !== null && typeof o.omm.EPOCH === 'string';
  return typeof o.line1 === 'string' && typeof o.line2 === 'string';
}

// ---------------------------------------------------------------------------
// Persisted metadata
// ---------------------------------------------------------------------------

interface UpstreamState {
  lastAttemptAt: string | null;
  nextAllowedAt: string | null;
  lastError: string | null;
  lastStatus: number | null;
  lastSuccessAt: string | null;
}

/** Outcome of one primary partition within a cycle. */
interface PartitionOutcome {
  /** ISO time the outcome was recorded. */
  at: string;
  /** Downloaded completely (not cut off, not failed). */
  complete: boolean;
  /** STARLINK records it delivered (complete or cut off). */
  records: number;
  error: string | null;
}

/** Partitioned full-catalogue refresh cycle (persisted in meta.json). */
interface CycleState {
  /** ISO start time; also the cycle's identity. */
  startedAt: string;
  /** ISO time every partition had succeeded (or the cycle was judged implausible); null while open. */
  completedAt: string | null;
  /** Completed with a plausible full catalogue. */
  full: boolean;
  /** Cycle-level error (e.g. implausibly few records across all partitions). */
  error: string | null;
  /** Partition URL -> latest outcome within this cycle. */
  partitions: Record<string, PartitionOutcome>;
}

interface CacheMeta {
  version: 1;
  /** Summary of the served set (informational; elements.json is authoritative). */
  source: string | null;
  url: string | null;
  fetchedAt: string | null;
  count: number;
  partial: boolean;
  fullCompletedAt: string | null;
  upstreams: Record<string, UpstreamState>;
  /** Current / last partitioned primary cycle. */
  cycle: CycleState | null;
}

function emptyMeta(): CacheMeta {
  return { version: 1, source: null, url: null, fetchedAt: null, count: 0, partial: false, fullCompletedAt: null, upstreams: {}, cycle: null };
}

function parseCycle(c: unknown): CycleState | null {
  if (!c || typeof c !== 'object') return null;
  const o = c as Partial<CycleState>;
  if (typeof o.startedAt !== 'string') return null;
  const partitions: Record<string, PartitionOutcome> = {};
  if (o.partitions && typeof o.partitions === 'object') {
    for (const [url, p] of Object.entries(o.partitions)) {
      if (!p || typeof p !== 'object' || typeof p.at !== 'string') continue;
      partitions[url] = {
        at: p.at,
        complete: p.complete === true,
        records: typeof p.records === 'number' ? p.records : 0,
        error: typeof p.error === 'string' ? p.error : null,
      };
    }
  }
  return {
    startedAt: o.startedAt,
    completedAt: typeof o.completedAt === 'string' ? o.completedAt : null,
    full: o.full === true,
    error: typeof o.error === 'string' ? o.error : null,
    partitions,
  };
}

/** Metas parsed from an unreadable / corrupt meta.json (all rate-limit locks lost). */
const corruptMetas = new WeakSet<CacheMeta>();

function isCorruptMeta(m: CacheMeta): boolean {
  return corruptMetas.has(m);
}

/**
 * Missing file => empty meta. Corrupt JSON => empty meta flagged corrupt (see
 * isCorruptMeta). Any other read error (EPERM / EBUSY from AV or sync tools,
 * EISDIR, ...) is thrown: "cannot read the locks" must never mean "no locks".
 */
function readMetaSync(file: string): CacheMeta {
  let raw: string;
  try {
    raw = fs.readFileSync(file, 'utf8');
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === 'ENOENT') return emptyMeta();
    throw err;
  }
  try {
    const m = JSON.parse(raw) as Partial<CacheMeta>;
    const out = emptyMeta();
    if (typeof m.source === 'string') out.source = m.source;
    if (typeof m.url === 'string') out.url = m.url;
    if (typeof m.fetchedAt === 'string') out.fetchedAt = m.fetchedAt;
    if (typeof m.count === 'number') out.count = m.count;
    if (typeof m.partial === 'boolean') out.partial = m.partial;
    if (typeof m.fullCompletedAt === 'string') out.fullCompletedAt = m.fullCompletedAt;
    if (m.upstreams && typeof m.upstreams === 'object') {
      for (const [url, s] of Object.entries(m.upstreams)) {
        if (!s || typeof s !== 'object') continue;
        out.upstreams[url] = {
          lastAttemptAt: typeof s.lastAttemptAt === 'string' ? s.lastAttemptAt : null,
          nextAllowedAt: typeof s.nextAllowedAt === 'string' ? s.nextAllowedAt : null,
          lastError: typeof s.lastError === 'string' ? s.lastError : null,
          lastStatus: typeof s.lastStatus === 'number' ? s.lastStatus : null,
          lastSuccessAt: typeof s.lastSuccessAt === 'string' ? s.lastSuccessAt : null,
        };
      }
    }
    out.cycle = parseCycle(m.cycle);
    return out;
  } catch {
    // Corrupt meta (e.g. zero-filled after a power loss): callers that decide
    // on upstream contact re-lock every upstream conservatively.
    const out = emptyMeta();
    corruptMetas.add(out);
    return out;
  }
}

/** For display / data-labelling paths only: never throws (falls back to an empty meta). */
function readMetaQuiet(file: string): CacheMeta {
  try {
    return readMetaSync(file);
  } catch {
    return emptyMeta();
  }
}

function upstreamState(meta: CacheMeta, url: string): UpstreamState {
  let s = meta.upstreams[url];
  if (!s) {
    s = { lastAttemptAt: null, nextAllowedAt: null, lastError: null, lastStatus: null, lastSuccessAt: null };
    meta.upstreams[url] = s;
  }
  return s;
}

const iso = (ms: number): string => new Date(ms).toISOString();
const parseIso = (s: string | null | undefined): number => (s ? Date.parse(s) : NaN);

// ---------------------------------------------------------------------------
// Process-wide state (survives Vite re-evaluating this module on restart)
// ---------------------------------------------------------------------------

interface Download {
  upstream: Upstream;
  startedAtMs: number;
  bytes: number;
  newlines: number;
  /** OMM CSV (1 line per record + header) vs TLE (3 lines per record); null until the first line arrived. */
  csv: boolean | null;
  abort: (reason: string) => void;
  promise: Promise<void>;
}

interface GlobalState {
  /** `${cacheDir}\n${url}` -> in-flight download (at most one per URL). */
  inflight: Map<string, Download>;
  /** `${cacheDir}\n${url}` -> nextAllowedAt ms (in-memory guard in case disk meta is lost/unwritable). */
  overlay: Map<string, number>;
  /** cacheDir -> serialized meta write chain. */
  metaChains: Map<string, Promise<unknown>>;
  /** cacheDir -> serialized merge/commit chain. */
  commitChains: Map<string, Promise<unknown>>;
  /** cacheDir -> the partitioned-primary cycle runner (at most one). */
  cycles: Map<string, CycleRun>;
}

/** An active partitioned-primary cycle runner. */
interface CycleRun {
  /** Identity of the owning service instance. */
  owner: object;
  /** The owning service was closed: the runner stops after its current partition. */
  ownerClosed: () => boolean;
  startedAtMs: number;
  /** startedAt of the cycle being worked on (null until known). */
  cycleId: string | null;
  /** Partition being downloaded / paused before / waited for. */
  current: Upstream | null;
  phase: 'starting' | 'download' | 'pause' | 'wait';
  /** Cancels a pause / wait. */
  wake: (() => void) | null;
  /** Called once the runner ended (a newer live service that wants to resume the cycle). */
  successor: (() => void) | null;
  promise: Promise<void>;
}

const GLOBAL_KEY = Symbol.for('starlink-tracker.elementsService.v2');
/** Pre-OMM module state: its in-memory locks and meta write chain are carried over. */
const GLOBAL_KEY_V1 = Symbol.for('starlink-tracker.elementsService.v1');
const G: GlobalState = (() => {
  const g = globalThis as Record<symbol, unknown>;
  const existing = g[GLOBAL_KEY] as (Omit<GlobalState, 'cycles'> & { cycles?: Map<string, CycleRun> }) | undefined;
  if (existing) {
    // State created by an older version of this module (before partitioned cycles).
    existing.cycles ??= new Map();
    return existing as GlobalState;
  }
  const v1 = g[GLOBAL_KEY_V1] as Partial<GlobalState> | undefined;
  const fresh: GlobalState = {
    inflight: new Map(),
    overlay: v1?.overlay instanceof Map ? v1.overlay : new Map(),
    metaChains: v1?.metaChains instanceof Map ? v1.metaChains : new Map(),
    commitChains: new Map(),
    cycles: new Map(),
  };
  g[GLOBAL_KEY] = fresh;
  return fresh;
})();

let tmpCounter = 0;

async function writeFileAtomic(file: string, data: string | Uint8Array): Promise<void> {
  await fsp.mkdir(path.dirname(file), { recursive: true });
  const tmp = `${file}.${process.pid}.${++tmpCounter}.tmp`;
  // fsync before the rename: after a power loss / BSOD the renamed file must
  // not come back zero-length or zero-filled (NTFS journals metadata only).
  const fh = await fsp.open(tmp, 'w');
  try {
    await fh.writeFile(data);
    await fh.sync();
  } finally {
    await fh.close();
  }
  for (let attempt = 0; ; attempt++) {
    try {
      await fsp.rename(tmp, file);
      return;
    } catch (err) {
      const code = (err as NodeJS.ErrnoException).code;
      // Windows: a reader/AV scanner holding the target briefly causes EPERM/EBUSY.
      if (attempt >= 6 || (code !== 'EPERM' && code !== 'EACCES' && code !== 'EBUSY')) {
        await fsp.rm(tmp, { force: true }).catch(() => {});
        throw err;
      }
      await new Promise((r) => setTimeout(r, 25 * (attempt + 1)));
    }
  }
}

/** A lockfile older than this was left behind by a crashed process and is broken. */
export const FILE_LOCK_STALE_MS = 30_000;
/** How long updateMeta waits for another process's meta.json lock before giving up. */
const META_LOCK_TIMEOUT_MS = 5_000;
/** How long a merge waits for another process's elements.json lock (a commit writes a few MB). */
const COMMIT_LOCK_TIMEOUT_MS = 20_000;

export interface FileLockOptions {
  timeoutMs?: number;
  staleMs?: number;
}

/**
 * Run `fn` while holding `lockPath` (created exclusively; holds the pid).
 * Excludes other processes sharing the cache dir (dev + preview, two dev
 * servers); within one process the per-cache-dir promise chains already
 * serialize. A lockfile older than `staleMs` is broken; rejects without
 * running `fn` once `timeoutMs` passed without getting the lock.
 */
export async function withFileLock<T>(lockPath: string, fn: () => Promise<T>, opts: FileLockOptions = {}): Promise<T> {
  const timeoutMs = opts.timeoutMs ?? META_LOCK_TIMEOUT_MS;
  const staleMs = opts.staleMs ?? FILE_LOCK_STALE_MS;
  const deadline = Date.now() + timeoutMs;
  await fsp.mkdir(path.dirname(lockPath), { recursive: true });
  for (let i = 0; ; i++) {
    try {
      const fh = await fsp.open(lockPath, 'wx');
      try {
        await fh.writeFile(String(process.pid));
      } finally {
        await fh.close();
      }
      break;
    } catch (err) {
      const code = (err as NodeJS.ErrnoException).code;
      // Windows: a lockfile its holder is deleting reports EPERM/EACCES/EBUSY.
      if (code !== 'EEXIST' && code !== 'EPERM' && code !== 'EACCES' && code !== 'EBUSY') throw err;
      let stale = false;
      try {
        stale = Date.now() - (await fsp.stat(lockPath)).mtimeMs > staleMs;
      } catch {
        // Gone in the meantime: retry (bounded by the deadline below).
      }
      if (stale) {
        await fsp.rm(lockPath, { force: true }).catch(() => {});
        continue;
      }
      if (Date.now() > deadline) throw new Error(`${lockPath} is held by another process`);
      await new Promise((r) => setTimeout(r, Math.min(10 * (i + 1), 100)));
    }
  }
  try {
    return await fn();
  } finally {
    await fsp.rm(lockPath, { force: true }).catch(() => {});
  }
}

/**
 * Read-modify-write meta.json, serialized per cache dir within this process
 * and, through a lockfile next to it, across processes sharing the cache dir
 * (every lock check-and-set runs inside such a mutate). A read error or a
 * lock timeout rejects (nothing is written); a corrupt meta reaches `mutate`
 * flagged (isCorruptMeta) so the caller can restore conservative locks.
 */
function updateMeta(cacheDir: string, mutate: (m: CacheMeta) => void): Promise<CacheMeta> {
  const file = path.join(cacheDir, META_FILE);
  const prev = G.metaChains.get(cacheDir) ?? Promise.resolve();
  const next = prev.catch(() => {}).then(() =>
    withFileLock(`${file}.lock`, async () => {
      const m = readMetaSync(file);
      mutate(m);
      await writeFileAtomic(file, JSON.stringify(m, null, 2) + '\n');
      return m;
    }),
  );
  G.metaChains.set(cacheDir, next);
  return next;
}

/**
 * Run `task` after every earlier commit for this cache dir (merges never
 * interleave), holding a lockfile next to elements.json so that a merge in
 * another process sharing the cache dir cannot overwrite it from an older base
 * (commit re-reads the set when the file changed).
 */
function enqueueCommit<T>(cacheDir: string, task: () => Promise<T>): Promise<T> {
  const prev = G.commitChains.get(cacheDir) ?? Promise.resolve();
  const next = prev
    .catch(() => {})
    .then(() => withFileLock(path.join(cacheDir, `${CACHE_FILE}.lock`), task, { timeoutMs: COMMIT_LOCK_TIMEOUT_MS }));
  G.commitChains.set(cacheDir, next);
  return next;
}

// ---------------------------------------------------------------------------
// Service
// ---------------------------------------------------------------------------

interface Dataset {
  version: number;
  /** Identity of the file it was loaded from (`path|mtime|size`). */
  fileKey: string;
  /** merged = CACHE_FILE, seed = legacy TLE cache, fixture = bundled snapshot. */
  kind: 'merged' | 'seed' | 'fixture';
  source: string;
  fetchedAtMs: number;
  /** Last complete primary / fallback download (NaN = never => partial). */
  fullCompletedAtMs: number;
  partial: boolean;
  records: ElementRecord[];
  origin: number[];
  sources: MergeSource[];
  sourceInfo: ElementSourceInfo[];
  recordsJson: string | null;
}

interface Body {
  key: string;
  json: Buffer;
  gzip: Promise<Buffer> | null;
}

let datasetVersion = 0;

/** Per-source record counts -> served source list (most records first) + display name. */
function describeSources(sources: readonly MergeSource[], origin: readonly number[]): { info: ElementSourceInfo[]; label: string } {
  const counts = new Array<number>(sources.length).fill(0);
  for (const o of origin) if (o >= 0 && o < counts.length) counts[o]!++;
  const info = sources
    .map((s, i) => ({ name: s.name, fetchedAt: s.fetchedAt, records: counts[i]!, complete: s.complete }))
    .filter((s) => s.records > 0)
    .sort((a, b) => b.records - a.records);
  return { info, label: info.length > 0 ? info.map((s) => s.name).join(' + ') : 'Disk cache' };
}

export function createElementsService(opts: ElementsServiceOptions): ElementsService {
  const rootDir = path.resolve(opts.rootDir);
  const cacheDir = path.resolve(opts.cacheDir ?? path.join(rootDir, '.cache'));
  const cacheFile = path.join(cacheDir, CACHE_FILE);
  const seedFile = path.join(cacheDir, LEGACY_CACHE_FILE);
  const metaFile = path.join(cacheDir, META_FILE);
  const fixturePath = path.resolve(opts.fixturePath ?? path.join(rootDir, 'fixtures', 'sup-gp-starlink-partial.tle'));
  const upstreams: readonly Upstream[] = opts.upstreams ?? DEFAULT_UPSTREAMS;
  const minIntervalMs = opts.minIntervalMs ?? DEFAULT_MIN_INTERVAL_MS;
  const requestTimeoutMs = opts.requestTimeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS;
  const minRecords = opts.minRecords ?? DEFAULT_MIN_RECORDS;
  const offline = opts.offline ?? process.env.CELESTRAK_OFFLINE === '1';
  const fetchImpl = opts.fetchImpl ?? globalThis.fetch;
  const now = opts.now ?? Date.now;
  const log = opts.log ?? ((msg: string) => console.log(`[elements] ${msg}`));

  const partitionPauseMs = opts.partitionPauseMs ?? DEFAULT_PARTITION_PAUSE_MS;
  const maxPartitionWaitMs = opts.maxPartitionWaitMs ?? DEFAULT_MAX_PARTITION_WAIT_MS;

  const roleOf = (up: Upstream): UpstreamRole => up.role ?? (up === upstreams[0] ? 'primary' : 'fallback');
  /** The partitioned primary, in download order. */
  const partitions: readonly Upstream[] = upstreams.filter((u) => roleOf(u) === 'primary');
  const fallbacks: readonly Upstream[] = upstreams.filter((u) => roleOf(u) === 'fallback');
  const supplements: readonly Upstream[] = upstreams.filter((u) => roleOf(u) === 'supplement');
  const partitioned = partitions.length > 1;
  const minRecordsOf = (up: Upstream): number =>
    up.minRecords ?? (roleOf(up) === 'primary' && partitioned ? 0 : minRecords);
  const sourceNameOf = (up: Upstream): string => up.sourceName ?? up.name;
  const keyOf = (url: string): string => `${cacheDir}\n${url}`;
  /** Identity of this service instance (owner of the cycle runner it starts). */
  const token = {};
  const downloadsHere = (): Download[] => {
    const out: Download[] = [];
    for (const up of upstreams) {
      const dl = G.inflight.get(keyOf(up.url));
      if (dl) out.push(dl);
    }
    return out;
  };

  let closed = false;
  let dataset: Dataset | null = null;
  let body: Body | null = null;
  let serviceError: string | null = null;
  let lastFixtureWarn = '';

  // ---- data loading -------------------------------------------------------

  function statKey(file: string): string | null {
    try {
      const st = fs.statSync(file);
      return st.isFile() && st.size > 0 ? `${file}|${st.mtimeMs}|${st.size}` : null;
    } catch {
      return null;
    }
  }

  /** A persisted timestamp written while the clock was ahead counts as "due", not as fresh for months. */
  function clampFuture(ms: number): number {
    return ms > now() + FUTURE_TOLERANCE_MS ? now() - minIntervalMs : ms;
  }

  function makeDataset(
    key: string,
    kind: Dataset['kind'],
    set: MergeSet,
    fetchedAtMs: number,
    fullCompletedAtMs: number,
    sourceOverride?: string,
  ): Dataset {
    const { info, label } = describeSources(set.sources, set.origin);
    return {
      version: ++datasetVersion,
      fileKey: key,
      kind,
      source: sourceOverride ?? label,
      fetchedAtMs,
      fullCompletedAtMs,
      partial: !Number.isFinite(fullCompletedAtMs),
      records: set.records,
      origin: set.origin,
      sources: set.sources,
      sourceInfo: info,
      recordsJson: null,
    };
  }

  function loadMerged(key: string): Dataset | null {
    try {
      const t0 = performance.now();
      const raw = JSON.parse(fs.readFileSync(cacheFile, 'utf8')) as Partial<StoredDataset>;
      if (raw.format !== 'starlink-tracker/elements' || !Array.isArray(raw.records) || !Array.isArray(raw.sources)) {
        serviceError = `Cache ${cacheFile} has an unknown format; ignored`;
        log(serviceError);
        return null;
      }
      const sources: MergeSource[] = raw.sources
        .filter((s): s is MergeSource => !!s && typeof s.name === 'string')
        .map((s) => ({
          name: s.name,
          url: typeof s.url === 'string' ? s.url : null,
          fetchedAt: typeof s.fetchedAt === 'string' ? s.fetchedAt : iso(0),
          complete: s.complete === true,
          placeholderIds: s.placeholderIds === true,
        }));
      const records: ElementRecord[] = [];
      const origin: number[] = [];
      const rawOrigin = Array.isArray(raw.origin) ? raw.origin : [];
      for (let i = 0; i < raw.records.length; i++) {
        const r = raw.records[i];
        if (!isValidRecord(r)) continue;
        const o = rawOrigin[i];
        records.push(r);
        origin.push(typeof o === 'number' && o >= 0 && o < sources.length ? o : 0);
      }
      if (records.length === 0) {
        serviceError = `Cache ${cacheFile} contains no valid element sets`;
        return null;
      }
      let fetchedAtMs = parseIso(raw.fetchedAt);
      if (!Number.isFinite(fetchedAtMs)) fetchedAtMs = fs.statSync(cacheFile).mtimeMs;
      const full = parseIso(raw.fullCompletedAt);
      log(`loaded ${records.length} records from ${CACHE_FILE} in ${(performance.now() - t0).toFixed(1)} ms`);
      return makeDataset(key, 'merged', { records, origin, sources }, clampFuture(fetchedAtMs), Number.isFinite(full) ? clampFuture(full) : NaN);
    } catch (err) {
      serviceError = `Failed to read ${cacheFile}: ${(err as Error).message}`;
      log(serviceError);
      return null;
    }
  }

  /** Legacy TLE cache or bundled fixture (TLE or OMM CSV text). */
  function loadText(file: string, key: string, kind: 'seed' | 'fixture'): Dataset | null {
    try {
      const text = fs.readFileSync(file, 'utf8');
      const t0 = performance.now();
      const { records, rejected } = parseElementsText(text);
      if (records.length === 0) {
        serviceError = `${kind === 'seed' ? 'TLE cache' : 'Fixture'} ${file} contains no valid element sets`;
        return null;
      }
      let fetchedAtMs: number;
      let name: string;
      let url: string | null = null;
      if (kind === 'seed') {
        const meta = readMetaQuiet(metaFile);
        name = SEED_SOURCE;
        url = meta.url;
        fetchedAtMs = parseIso(meta.fetchedAt);
        if (!Number.isFinite(fetchedAtMs)) fetchedAtMs = fs.statSync(file).mtimeMs;
        fetchedAtMs = clampFuture(fetchedAtMs);
      } else {
        name = FIXTURE_SOURCE;
        // "As of" the newest element-set epoch in the snapshot (the file mtime
        // is meaningless after a git checkout).
        let newest = -Infinity;
        for (const r of records) {
          const e = recordEpochMs(r);
          if (e > newest) newest = e;
        }
        fetchedAtMs = Number.isFinite(newest) ? newest : fs.statSync(file).mtimeMs;
      }
      log(
        `loaded ${records.length} records from ${kind === 'seed' ? `${LEGACY_CACHE_FILE} (migration seed)` : 'bundled fixture'}` +
          `${rejected ? ` (${rejected} rejected)` : ''} in ${(performance.now() - t0).toFixed(1)} ms`,
      );
      const set: MergeSet = {
        records,
        origin: new Array<number>(records.length).fill(0),
        sources: [{ name, url, fetchedAt: iso(fetchedAtMs), complete: false, placeholderIds: false }],
      };
      return makeDataset(key, kind, set, fetchedAtMs, NaN, name);
    } catch (err) {
      serviceError = `Failed to read ${file}: ${(err as Error).message}`;
      return null;
    }
  }

  /** Current data set; re-parses only when the backing file changed on disk. */
  function currentDataset(): Dataset | null {
    const cacheKey = statKey(cacheFile);
    if (cacheKey !== null) {
      if (dataset?.fileKey === cacheKey) return dataset;
      const ds = loadMerged(cacheKey);
      if (ds) return setDataset(ds);
    }
    const seedKey = statKey(seedFile);
    if (seedKey !== null) {
      if (dataset?.fileKey === seedKey) return dataset;
      const ds = loadText(seedFile, seedKey, 'seed');
      if (ds) return setDataset(ds);
    }
    const fixtureKey = statKey(fixturePath);
    if (fixtureKey !== null) {
      if (dataset?.fileKey === fixtureKey) return dataset;
      const ds = loadText(fixturePath, fixtureKey, 'fixture');
      if (ds) return setDataset(ds);
    } else if (cacheKey === null && seedKey === null && lastFixtureWarn !== fixturePath) {
      lastFixtureWarn = fixturePath;
      log(`no cache and no fixture at ${fixturePath}; waiting for upstream`);
    }
    if (dataset !== null) {
      dataset = null;
      body = null;
    }
    return null;
  }

  function setDataset(ds: Dataset): Dataset {
    dataset = ds;
    body = null;
    return ds;
  }

  // ---- rate limiting --------------------------------------------------------

  function nextAllowedMs(meta: CacheMeta, url: string): number {
    const disk = parseIso(meta.upstreams[url]?.nextAllowedAt);
    const mem = G.overlay.get(keyOf(url)) ?? 0;
    return Math.max(Number.isFinite(disk) ? disk : 0, mem);
  }

  const doneIn = (cyc: CycleState | null, url: string): boolean => cyc?.partitions[url]?.complete === true;

  /**
   * When the current cycle is superseded by a new one: its oldest partition
   * success plus a window (CelesTrak updates GP every 2 h). A cycle without
   * any success stays open (its partitions are retried on their own windows);
   * no cycle at all => now.
   */
  function renewAtMs(cyc: CycleState | null): number {
    if (!cyc) return -Infinity;
    let oldest = Infinity;
    for (const p of partitions) {
      const o = cyc.partitions[p.url];
      if (!o?.complete) continue;
      const at = parseIso(o.at);
      oldest = Math.min(oldest, Number.isFinite(at) ? clampFuture(at) : -Infinity);
    }
    return oldest === Infinity ? Infinity : oldest + minIntervalMs;
  }

  /** Earliest time the full catalogue (some primary partition) may be fetched again (may be in the past = due). */
  function nextRefreshMs(meta: CacheMeta, ds: Dataset | null): number {
    const cyc = meta.cycle;
    const renew = renewAtMs(cyc);
    let t = Infinity;
    for (const p of partitions) {
      const na = nextAllowedMs(meta, p.url);
      t = Math.min(t, doneIn(cyc, p.url) ? Math.max(na, renew) : na);
    }
    if (!Number.isFinite(t)) t = 0;
    if (ds && Number.isFinite(ds.fullCompletedAtMs)) t = Math.max(t, ds.fullCompletedAtMs + minIntervalMs);
    if (t === 0 && ds) t = ds.fetchedAtMs;
    return t;
  }

  /** A primary partition / fallback / legacy full-catalogue download is in flight here (from any service instance). */
  function fullDownloadInFlight(): boolean {
    if (G.cycles.has(cacheDir)) return true;
    for (const up of [...partitions, ...fallbacks]) if (G.inflight.has(keyOf(up.url))) return true;
    for (const url of LEGACY_URLS) if (G.inflight.has(keyOf(url))) return true;
    return false;
  }

  /**
   * Full-catalogue scheduling: start (or resume) the partition cycle when a
   * partition is due; otherwise a fallback, but only while the partitioned
   * primary cannot deliver (a partition failed and none is due) and there is
   * no reasonably recent full download. One full-catalogue activity at a time.
   */
  function scheduleFull(meta: CacheMeta, ds: Dataset | null, t: number): void {
    const run = G.cycles.get(cacheDir);
    if (run) {
      // A runner of a closed service (Vite restart) ends after its current
      // partition; this live service resumes the cycle right after it.
      if (run.owner !== token && run.ownerClosed()) run.successor = () => refreshIfDue();
      return;
    }
    if (fullDownloadInFlight()) return;
    const fullMs = ds?.fullCompletedAtMs ?? NaN;
    const hasFull = Number.isFinite(fullMs);
    if (hasFull && t - fullMs < minIntervalMs) return; // fresh enough
    const cyc = meta.cycle;
    const startNew = cyc === null || t >= renewAtMs(cyc);
    const pending = partitions.filter((p) => startNew || !doneIn(cyc, p.url));
    if (pending.some((p) => t >= nextAllowedMs(meta, p.url))) {
      startCycle(startNew);
      return;
    }
    // The partitioned primary cannot deliver right now.
    const fullAndRecent = hasFull && t - fullMs < Math.max(STALE_FALLBACK_MS, minIntervalMs);
    if (fullAndRecent) return;
    const failing =
      partitions.length === 0 || !!cyc?.error || pending.some((p) => !!meta.upstreams[p.url]?.lastError);
    if (!failing) return; // locked / in progress, not failing
    for (const up of fallbacks) {
      if (t >= nextAllowedMs(meta, up.url)) {
        startDownload(up, null);
        return;
      }
      // Inside its window: try the next one only when this one failed too.
      if (!meta.upstreams[up.url]?.lastError) return;
    }
  }

  /** Lock every upstream for a full window from now (disk + memory). */
  function lockAllUpstreams(m: CacheMeta, reason: string): void {
    const until = now() + minIntervalMs;
    for (const up of upstreams) {
      upstreamState(m, up.url).nextAllowedAt = iso(until);
      const key = keyOf(up.url);
      G.overlay.set(key, Math.max(G.overlay.get(key) ?? 0, until));
    }
    log(`${reason}; all upstreams locked until ${iso(until)}`);
  }

  /** updateMeta for this cache dir; a corrupt meta.json heals with conservative locks, never with none. */
  function writeMeta(mutate: (m: CacheMeta) => void): Promise<CacheMeta> {
    return updateMeta(cacheDir, (m) => {
      if (isCorruptMeta(m)) lockAllUpstreams(m, `${metaFile} was corrupt`);
      mutate(m);
    });
  }

  function refreshIfDue(): void {
    if (closed || offline) return;
    const ds = currentDataset();
    let meta: CacheMeta;
    try {
      meta = readMetaSync(metaFile);
    } catch (err) {
      serviceError = `Cannot read ${metaFile}; not contacting upstream: ${(err as Error).message}`;
      return;
    }
    if (isCorruptMeta(meta)) {
      // The locks are lost: re-lock everything instead of treating it as "no locks".
      writeMeta(() => {}).catch((err) => log(`cannot repair ${metaFile}: ${(err as Error).message}`));
      return;
    }
    const t = now();
    // A lock further ahead than any legitimate one was written under a clock
    // that ran ahead. Re-anchor it to a full window from now (persisted, so it
    // expires instead of sliding) and decide again on the next call.
    const bogus = upstreams.filter((up) => nextAllowedMs(meta, up.url) > t + Math.max(MAX_LOCK_AHEAD_MS, minIntervalMs));
    if (bogus.length > 0) {
      const until = t + minIntervalMs;
      for (const up of bogus) G.overlay.set(keyOf(up.url), until);
      writeMeta((m) => {
        for (const up of bogus) upstreamState(m, up.url).nextAllowedAt = iso(until);
      }).catch((err) => log(`cannot repair ${metaFile}: ${(err as Error).message}`));
      log(`lock(s) in the future for ${bogus.map((up) => up.name).join(', ')}; clamped to ${iso(until)}`);
      return;
    }
    scheduleFull(meta, ds, t);
    // Supplements run on their own windows, independent of the full catalogue.
    for (const sup of supplements) {
      if (!G.inflight.has(keyOf(sup.url)) && t >= nextAllowedMs(meta, sup.url)) startDownload(sup, null);
    }
  }

  // ---- partitioned primary cycle ---------------------------------------------

  /** Resolves after `ms`, or earlier when the runner is woken (service closed). */
  function sleepRun(run: CycleRun, ms: number): Promise<void> {
    return new Promise<void>((resolve) => {
      const done = (): void => {
        clearTimeout(timer);
        if (run.wake === done) run.wake = null;
        resolve();
      };
      const timer = setTimeout(done, Math.max(0, ms));
      run.wake = done;
    });
  }

  /** Register and start the (single) cycle runner for this cache dir. Synchronous guard first. */
  function startCycle(startNew: boolean): void {
    const run: CycleRun = {
      owner: token,
      ownerClosed: () => closed,
      startedAtMs: now(),
      cycleId: null,
      current: null,
      phase: 'starting',
      wake: null,
      successor: null,
      promise: Promise.resolve(),
    };
    G.cycles.set(cacheDir, run);
    run.promise = (async () => {
      try {
        await runCycle(run, startNew);
      } catch (err) {
        serviceError = `refresh cycle failed: ${(err as Error).message}`;
        log(serviceError);
      } finally {
        if (G.cycles.get(cacheDir) === run) G.cycles.delete(cacheDir);
        const next = run.successor;
        run.successor = null;
        if (next) {
          try {
            next();
          } catch (err) {
            log(`cannot resume refresh cycle: ${(err as Error).message}`);
          }
        }
      }
    })();
  }

  /**
   * Download the partitions that are not yet done in the current cycle,
   * sequentially, in list order, each at most once per runner and only when
   * its own window is open. Each download merges + persists on its own.
   */
  async function runCycle(run: CycleRun, startNew: boolean): Promise<void> {
    if (startNew) {
      const id = iso(now());
      try {
        await writeMeta((m) => {
          m.cycle = { startedAt: id, completedAt: null, full: false, error: null, partitions: {} };
        });
      } catch (err) {
        serviceError = `Cannot write ${metaFile}; not contacting upstream: ${(err as Error).message}`;
        log(serviceError);
        return;
      }
      log(`full refresh cycle started (${partitions.length} partition${partitions.length === 1 ? '' : 's'})`);
    }
    const tried = new Set<string>();
    let fetched = 0;
    let waitedFor: string | null = null;
    for (;;) {
      if (closed) return;
      let meta: CacheMeta;
      try {
        meta = readMetaSync(metaFile);
      } catch (err) {
        serviceError = `Cannot read ${metaFile}; not contacting upstream: ${(err as Error).message}`;
        return;
      }
      const cyc = meta.cycle;
      if (isCorruptMeta(meta) || cyc === null) return;
      run.cycleId = cyc.startedAt;
      const t = now();
      const pending = partitions.filter(
        (p) => !tried.has(p.url) && !doneIn(cyc, p.url) && !G.inflight.has(keyOf(p.url)),
      );
      if (pending.length === 0) break;
      const next = pending.find((p) => t >= nextAllowedMs(meta, p.url));
      if (!next) {
        // Wait for the soonest window if it opens shortly; never twice for the same one.
        let soonest = pending[0]!;
        for (const p of pending) if (nextAllowedMs(meta, p.url) < nextAllowedMs(meta, soonest.url)) soonest = p;
        const waitMs = nextAllowedMs(meta, soonest.url) - t;
        if (waitMs > maxPartitionWaitMs || waitedFor === soonest.url) break;
        waitedFor = soonest.url;
        run.phase = 'wait';
        run.current = soonest;
        await sleepRun(run, waitMs);
        run.current = null;
        continue;
      }
      if (fetched > 0 && partitionPauseMs > 0) {
        run.phase = 'pause';
        run.current = next;
        await sleepRun(run, partitionPauseMs);
        if (closed) return;
      }
      run.phase = 'download';
      run.current = next;
      tried.add(next.url);
      await startDownload(next, cyc.startedAt).promise;
      fetched++;
      run.current = null;
    }
    const meta = readMetaQuiet(metaFile);
    const cyc = meta.cycle;
    const done = partitions.filter((p) => doneIn(cyc, p.url)).length;
    log(
      cyc?.completedAt
        ? `full refresh cycle ${cyc.full ? 'complete' : `finished without a full set (${cyc.error ?? 'unknown error'})`}: ${done}/${partitions.length} partitions`
        : `full refresh cycle paused: ${done}/${partitions.length} partitions done; the rest is retried after their own windows`,
    );
  }

  // ---- download -------------------------------------------------------------

  /** Record a primary partition's outcome in the cycle it was started for (if that cycle is still current). */
  function setOutcome(m: CacheMeta, cycleId: string | null, url: string, outcome: PartitionOutcome): void {
    if (cycleId === null || m.cycle?.startedAt !== cycleId) return;
    m.cycle.partitions[url] = outcome;
  }

  /**
   * Start one download. `cycleId` = the cycle a primary partition is
   * downloaded for (null for supplements / fallbacks).
   */
  function startDownload(up: Upstream, cycleId: string | null): Download {
    const startedAtMs = now();
    const ac = new AbortController();
    let abortReason = '';
    const dl: Download = {
      upstream: up,
      startedAtMs,
      bytes: 0,
      newlines: 0,
      csv: null,
      abort: (reason) => {
        if (!abortReason) abortReason = reason;
        ac.abort(new Error(reason));
      },
      promise: Promise.resolve(),
    };
    // Guards set synchronously, before anything async: one download per URL in
    // flight, and this URL is blocked in memory even if the meta write below fails.
    const overlayKey = keyOf(up.url);
    G.inflight.set(overlayKey, dl);
    const overlayBefore = G.overlay.get(overlayKey);
    G.overlay.set(overlayKey, startedAtMs + minIntervalMs);

    const recordFailure = (message: string, status: number | null, nextAllowedAtMs?: number) => {
      log(`${up.name}: ${message}`);
      if (nextAllowedAtMs !== undefined) G.overlay.set(overlayKey, nextAllowedAtMs);
      const at = iso(now());
      return writeMeta((m) => {
        const s = upstreamState(m, up.url);
        s.lastError = message;
        s.lastStatus = status;
        if (nextAllowedAtMs !== undefined) s.nextAllowedAt = iso(nextAllowedAtMs);
        setOutcome(m, cycleId, up.url, { at, complete: false, records: 0, error: message });
      });
    };

    const run = async (): Promise<void> => {
      // 1) Persist the attempt BEFORE contacting upstream, re-checking the lock
      //    against the freshly read meta (the one refreshIfDue saw may be stale).
      let lockedUntil = null as string | null;
      try {
        await writeMeta((m) => {
          const s = upstreamState(m, up.url);
          if (parseIso(s.nextAllowedAt) > startedAtMs) {
            lockedUntil = s.nextAllowedAt;
            return;
          }
          s.lastAttemptAt = iso(startedAtMs);
          s.nextAllowedAt = iso(startedAtMs + minIntervalMs);
        });
      } catch (err) {
        serviceError = `Cannot write ${metaFile}; not contacting upstream: ${(err as Error).message}`;
        log(serviceError);
        return;
      }
      if (lockedUntil !== null) {
        // No attempt was made: drop this attempt's in-memory window again.
        if (G.overlay.get(overlayKey) === startedAtMs + minIntervalMs) {
          if (overlayBefore === undefined) G.overlay.delete(overlayKey);
          else G.overlay.set(overlayKey, overlayBefore);
        }
        log(`${up.name}: locked until ${lockedUntil}; not contacting upstream`);
        return;
      }

      log(`downloading ${up.name} (next attempt allowed after ${iso(startedAtMs + minIntervalMs)})`);
      const timer = setTimeout(() => dl.abort(`timed out after ${Math.round(requestTimeoutMs / 1000)} s`), requestTimeoutMs);
      timer.unref?.();
      let text = '';
      let streamError: string | null = null;
      try {
        let res: Response;
        try {
          res = await fetchImpl(up.url, {
            // fetch (undici) negotiates gzip/deflate/br and decompresses transparently.
            headers: { 'User-Agent': USER_AGENT, Accept: 'text/csv, text/plain;q=0.9, */*;q=0.5' },
            signal: ac.signal,
            redirect: 'follow',
          });
        } catch (err) {
          await recordFailure(abortReason || `request failed: ${(err as Error).message}`, null);
          return;
        }

        if (!res.ok) {
          const errBody = await readLimited(res, ERROR_BODY_LIMIT).catch(() => '');
          if (res.status === 403 && /has not updated since/i.test(errBody)) {
            const t = now();
            const lockTs = parseLockTimestamp(errBody);
            let until = t + minIntervalMs;
            if (Number.isFinite(lockTs) && lockTs + LOCK_AFTER_TIMESTAMP_MS > t) {
              until = Math.min(Math.max(lockTs + LOCK_AFTER_TIMESTAMP_MS, t + MIN_LOCK_RETRY_MS), t + 24 * 3_600_000);
            }
            await recordFailure(`HTTP 403 locked (data not updated since last download); retry after ${iso(until)}`, 403, until);
            return;
          }
          if (res.status === 404 && isNoDataResponse(errBody) && roleOf(up) !== 'supplement' && minRecordsOf(up) === 0) {
            // CelesTrak answers a query without matches with HTTP 404 "No GP
            // data found": for a name partition that is a valid, empty result.
            text = errBody;
            dl.bytes += errBody.length;
          } else {
            const snippet = errBody.replace(/\s+/g, ' ').trim().slice(0, 200);
            await recordFailure(`HTTP ${res.status} ${res.statusText}${snippet ? `: ${snippet}` : ''}`, res.status);
            return;
          }
        } else {
          // 2) Stream the (decompressed) body, counting progress live.
          const parts: string[] = [];
          let head = '';
          const decoder = new TextDecoder();
          try {
            if (res.body) {
              const reader = res.body.getReader();
              for (;;) {
                const { done, value } = await reader.read();
                if (done) break;
                dl.bytes += value.byteLength;
                if (dl.bytes > MAX_BODY_BYTES) {
                  dl.abort(`response exceeded ${MAX_BODY_BYTES} bytes`);
                  throw new Error(abortReason);
                }
                const s = decoder.decode(value, { stream: true });
                let idx = s.indexOf('\n');
                while (idx !== -1) {
                  dl.newlines++;
                  idx = s.indexOf('\n', idx + 1);
                }
                if (dl.csv === null) {
                  head += s;
                  if (head.includes('\n') || head.length > 4096) dl.csv = looksLikeOmmCsv(head);
                }
                parts.push(s);
              }
            } else {
              const s = await res.text();
              dl.bytes += s.length;
              parts.push(s);
            }
            parts.push(decoder.decode());
          } catch (err) {
            streamError = abortReason || `download interrupted: ${(err as Error).message}`;
          }
          text = parts.join('');
        }
      } finally {
        clearTimeout(timer);
      }

      // 3) Validate. A cut-off download still contributes what it got.
      const role = roleOf(up);
      const complete = streamError === null;
      const parsed = parseElementsText(text, { truncated: !complete });
      const records = up.starlinkOnly ? filterStarlink(parsed.records) : parsed.records;
      const count = records.length;
      const detail = `${count} valid records (${parsed.rejected} rejected) in ${dl.bytes} bytes`;
      if (complete) {
        const min = minRecordsOf(up);
        // A supplement / an optional-size partition must at least be a recognisable
        // answer ("No GP data found" = a valid empty partition); the others must be plausibly large.
        const unrecognised =
          role === 'supplement' ? !parsed.recognised : min === 0 && !parsed.recognised && !isNoDataResponse(text);
        if (unrecognised || (role !== 'supplement' && count < min)) {
          await recordFailure(`${unrecognised ? 'unrecognised response' : 'implausible result'}: ${detail}`, 200);
          return;
        }
      } else if (count === 0) {
        await recordFailure(streamError!, 200);
        return;
      }

      // 4) Merge into the served set and persist (serialized per cache dir).
      try {
        await enqueueCommit(cacheDir, () => commit(up, role, records, complete, streamError, detail, startedAtMs, cycleId));
      } catch (err) {
        serviceError = `Failed to write cache: ${(err as Error).message}`;
        log(serviceError);
      }
    };

    dl.promise = run()
      .catch((err) => {
        serviceError = `download task failed: ${(err as Error).message}`;
        log(serviceError);
      })
      .finally(() => {
        if (G.inflight.get(overlayKey) === dl) G.inflight.delete(overlayKey);
      });
    return dl;
  }

  async function commit(
    up: Upstream,
    role: UpstreamRole,
    records: ElementRecord[],
    complete: boolean,
    streamError: string | null,
    detail: string,
    startedAtMs: number,
    cycleId: string | null,
  ): Promise<void> {
    const base = currentDataset();
    const fetchedAtMs = now();

    // Cycle bookkeeping for a primary partition: does this one complete the cycle?
    let cycleComplete = false;
    let cycleError: string | null = null;
    let sourceComplete = complete;
    let cycleTotal = 0;
    if (cycleId !== null) {
      await G.metaChains.get(cacheDir)?.catch(() => {});
      const cyc = readMetaQuiet(metaFile).cycle;
      if (cyc?.startedAt === cycleId) {
        let allDone = complete;
        cycleTotal = complete ? records.length : 0;
        for (const p of partitions) {
          if (p.url === up.url) continue;
          const o = cyc.partitions[p.url];
          if (o?.complete) cycleTotal += o.records;
          else allDone = false;
          // The shared source counts as complete while no partition of this cycle failed.
          if (o && !o.complete) sourceComplete = false;
        }
        if (allDone && cyc.completedAt === null) {
          if (cycleTotal >= minRecords) cycleComplete = true;
          else cycleError = `implausible result: ${cycleTotal} records across ${partitions.length} partitions`;
        }
      }
    }
    const fullCompletedAtMs =
      complete && (role === 'fallback' || cycleComplete) ? fetchedAtMs : (base?.fullCompletedAtMs ?? NaN);
    const fullChanged = fullCompletedAtMs !== (base?.fullCompletedAtMs ?? NaN) && Number.isFinite(fullCompletedAtMs);
    const merged = mergeElementSets(
      base ? { records: base.records, origin: base.origin, sources: base.sources } : { records: [], origin: [], sources: [] },
      records,
      {
        name: sourceNameOf(up),
        url: up.url,
        fetchedAt: iso(fetchedAtMs),
        complete: sourceComplete,
        placeholderIds: up.placeholderIds === true,
      },
      fetchedAtMs,
    );
    const changed = merged.added > 0 || merged.updated > 0 || merged.dropped > 0 || merged.duplicates > 0;
    // A supplement / partition that brought nothing new leaves the set (and its
    // file) as is, unless it completed the full set; a lone primary or a
    // fallback contribution is always persisted (sources, full flag).
    const always = role === 'fallback' || (role === 'primary' && !partitioned);
    const write = merged.records.length > 0 && (changed || fullChanged || always);
    const { info, label } = describeSources(merged.sources, merged.origin);
    if (write) {
      const stored: StoredDataset = {
        format: 'starlink-tracker/elements',
        version: 1,
        fetchedAt: iso(fetchedAtMs),
        fullCompletedAt: Number.isFinite(fullCompletedAtMs) ? iso(fullCompletedAtMs) : null,
        sources: merged.sources,
        origin: merged.origin,
        records: merged.records,
      };
      await writeFileAtomic(cacheFile, JSON.stringify(stored));
    }
    await writeMeta((m) => {
      if (write) {
        m.source = label;
        m.url = up.url;
        m.fetchedAt = iso(fetchedAtMs);
        m.count = merged.records.length;
        m.partial = !Number.isFinite(fullCompletedAtMs);
        m.fullCompletedAt = Number.isFinite(fullCompletedAtMs) ? iso(fullCompletedAtMs) : null;
      }
      const s = upstreamState(m, up.url);
      s.lastStatus = 200;
      s.lastError = complete ? null : streamError;
      if (complete) s.lastSuccessAt = iso(fetchedAtMs);
      setOutcome(m, cycleId, up.url, { at: iso(fetchedAtMs), complete, records: records.length, error: complete ? null : streamError });
      if (cycleId !== null && m.cycle?.startedAt === cycleId && (cycleComplete || cycleError !== null)) {
        m.cycle.completedAt = iso(fetchedAtMs);
        m.cycle.full = cycleComplete;
        m.cycle.error = cycleError;
      }
    });
    if (cycleError) log(`full refresh cycle: ${cycleError}`);
    log(
      `${complete ? 'merged' : 'merged incomplete download from'} ${up.name}: ${detail}; ` +
        `+${merged.added} new, ${merged.updated} newer, ${merged.dropped} stale dropped` +
        `${merged.duplicates ? `, ${merged.duplicates} placeholder duplicates` : ''} -> ${merged.records.length} records` +
        `${write ? '' : ' (unchanged)'} in ${((now() - startedAtMs) / 1000).toFixed(1)} s`,
    );
    // Install the merged set directly (no re-read) if this service is live.
    if (write && !closed) {
      const key = statKey(cacheFile);
      if (key) {
        setDataset({
          version: ++datasetVersion,
          fileKey: key,
          kind: 'merged',
          source: label,
          fetchedAtMs,
          fullCompletedAtMs,
          partial: !Number.isFinite(fullCompletedAtMs),
          records: merged.records,
          origin: merged.origin,
          sources: merged.sources,
          sourceInfo: info,
          recordsJson: null,
        });
      }
    }
  }

  // ---- HTTP -----------------------------------------------------------------

  const partitionLabel = (up: Upstream): string => up.partition ?? up.name;

  /** Progress of the partitioned primary (null without primary partitions or before the first cycle). */
  function cycleInfo(meta: CacheMeta): ElementsCycleInfo | null {
    if (partitions.length === 0) return null;
    const cyc = meta.cycle;
    const run = G.cycles.get(cacheDir) ?? null;
    if (cyc === null && run === null) return null;
    let done = 0;
    let failed = 0;
    let lastError: string | null = cyc?.error ?? null;
    let lastErrorAt = -Infinity;
    for (const p of partitions) {
      const o = cyc?.partitions[p.url];
      if (!o) continue;
      if (o.complete) done++;
      else {
        failed++;
        const at = parseIso(o.at);
        if (cyc?.error == null && o.error !== null && !(at <= lastErrorAt)) {
          lastError = `${partitionLabel(p)}: ${o.error}`;
          lastErrorAt = at;
        }
      }
    }
    return {
      startedAt: cyc?.startedAt ?? null,
      completedAt: cyc?.completedAt ?? null,
      complete: cyc?.completedAt != null && cyc.full && done === partitions.length,
      partitionsTotal: partitions.length,
      partitionsDone: done,
      partitionsFailed: failed,
      current: run?.current ? partitionLabel(run.current) : null,
      running: run !== null,
      lastError,
    };
  }

  function readyBody(ds: Dataset, meta: CacheMeta): Body {
    const nre = iso(nextRefreshMs(meta, ds));
    const cycle = cycleInfo(meta);
    const key = `${ds.version}|${nre}|${JSON.stringify(cycle)}`;
    if (body !== null && body.key === key) return body;
    ds.recordsJson ??= JSON.stringify(ds.records);
    const head: Omit<ElementsReady, 'records'> = {
      status: 'ready',
      source: ds.source,
      fetchedAt: iso(ds.fetchedAtMs),
      partial: ds.partial,
      nextRefreshEarliest: nre,
      count: ds.records.length,
      sources: ds.sourceInfo,
      cycle,
    };
    const headJson = JSON.stringify(head);
    const json = Buffer.from(`${headJson.slice(0, -1)},"records":${ds.recordsJson}}`, 'utf8');
    body = { key, json, gzip: null };
    return body;
  }

  function gzipOf(b: Body): Promise<Buffer> {
    b.gzip ??= new Promise<Buffer>((resolve, reject) => {
      zlib.gzip(b.json, { level: 6 }, (err, out) => (err ? reject(err) : resolve(out)));
    });
    return b.gzip;
  }

  function sendJson(req: IncomingMessage, res: ServerResponse, statusCode: number, payload: Buffer, gzipped: boolean): void {
    res.statusCode = statusCode;
    res.setHeader('Content-Type', 'application/json; charset=utf-8');
    res.setHeader('Cache-Control', 'no-store');
    res.setHeader('Vary', 'Accept-Encoding');
    if (gzipped) res.setHeader('Content-Encoding', 'gzip');
    res.setHeader('Content-Length', payload.byteLength);
    res.end(req.method === 'HEAD' ? undefined : payload);
  }

  const recordsOf = (dl: Download): number => (dl.csv ? Math.max(0, dl.newlines - 1) : Math.floor(dl.newlines / 3));

  function progress(): { bytes: number; records: number; startedAtMs: number | null; names: string[] } {
    let bytes = 0;
    let records = 0;
    let startedAtMs: number | null = null;
    const names: string[] = [];
    const dls = downloadsHere();
    for (const dl of dls) {
      bytes += dl.bytes;
      records += recordsOf(dl);
      if (startedAtMs === null || dl.startedAtMs < startedAtMs) startedAtMs = dl.startedAtMs;
      names.push(dl.upstream.name);
    }
    // A cycle runner between two partitions (starting / pausing) is still "fetching";
    // one waiting for a window is not.
    const run = G.cycles.get(cacheDir);
    if (run && run.phase !== 'wait' && !dls.some((dl) => roleOf(dl.upstream) === 'primary')) {
      names.push(run.current?.name ?? partitions[0]?.name ?? 'full catalogue');
      if (startedAtMs === null || run.startedAtMs < startedAtMs) startedAtMs = run.startedAtMs;
    }
    return { bytes, records, startedAtMs, names };
  }

  function warmingBody(): ElementsWarming {
    const p = progress();
    let message: string;
    if (p.names.length > 0) message = `Downloading orbital elements from ${p.names.join(', ')}…`;
    else if (offline) message = 'No orbital element data available (offline mode, no cache and no bundled snapshot).';
    else {
      const err = lastAttempt(readMetaQuiet(metaFile)).error ?? serviceError;
      message = err ? `Waiting to retry upstream (last error: ${err})` : 'Waiting for upstream download to start…';
    }
    return {
      status: 'warming',
      message,
      bytesReceived: p.bytes,
      recordsReceived: p.records,
      startedAt: p.startedAtMs !== null ? iso(p.startedAtMs) : null,
    };
  }

  function lastAttempt(meta: CacheMeta): { at: string | null; error: string | null } {
    let at: string | null = null;
    let atMs = -Infinity;
    let error: string | null = null;
    for (const up of upstreams) {
      const s = meta.upstreams[up.url];
      const t = parseIso(s?.lastAttemptAt);
      if (s && t > atMs) {
        atMs = t;
        at = s.lastAttemptAt;
        error = s.lastError;
      }
    }
    return { at, error };
  }

  function status(): ElementsStatus {
    const ds = currentDataset();
    const meta = readMetaQuiet(metaFile);
    const last = lastAttempt(meta);
    const p = progress();
    return {
      hasCache: ds !== null && ds.kind !== 'fixture',
      cacheFetchedAt: ds ? iso(ds.fetchedAtMs) : null,
      cachePartial: ds?.partial ?? false,
      cacheCount: ds?.records.length ?? 0,
      fetching: p.names.length > 0,
      bytesReceived: p.bytes,
      recordsReceived: p.records,
      lastAttemptAt: last.at,
      lastError: last.error ?? serviceError,
      nextAllowedFetchAt: offline
        ? null
        : (() => {
            const t = nextRefreshMs(meta, ds);
            return t > 0 ? iso(t) : null;
          })(),
      sources: ds?.sourceInfo ?? [],
      downloads: downloadsHere().map((dl) => ({
        name: dl.upstream.name,
        bytesReceived: dl.bytes,
        recordsReceived: recordsOf(dl),
        startedAt: iso(dl.startedAtMs),
      })),
      cycle: cycleInfo(meta),
    };
  }

  async function handleElements(req: IncomingMessage, res: ServerResponse): Promise<void> {
    refreshIfDue();
    const ds = currentDataset();
    if (ds === null) {
      res.setHeader('Retry-After', '2');
      sendJson(req, res, 503, Buffer.from(JSON.stringify(warmingBody()), 'utf8'), false);
      return;
    }
    const b = readyBody(ds, readMetaQuiet(metaFile));
    if (acceptsGzip(req.headers['accept-encoding'])) {
      const gz = await gzipOf(b);
      sendJson(req, res, 200, gz, true);
    } else {
      sendJson(req, res, 200, b.json, false);
    }
  }

  async function whenIdle(): Promise<void> {
    for (;;) {
      const dls = downloadsHere();
      await Promise.all([...dls.map((dl) => dl.promise), G.cycles.get(cacheDir)?.promise]);
      await G.commitChains.get(cacheDir)?.catch(() => {});
      await G.metaChains.get(cacheDir)?.catch(() => {});
      if (downloadsHere().length === 0 && !G.cycles.has(cacheDir)) return;
    }
  }

  function handle(req: IncomingMessage, res: ServerResponse, next: NextFunction): void {
    const url = req.url ?? '';
    const q = url.indexOf('?');
    let pathname = q === -1 ? url : url.slice(0, q);
    if (pathname.length > 1 && pathname.endsWith('/')) pathname = pathname.slice(0, -1);
    if (pathname !== '/api/elements' && pathname !== '/api/elements/status') return next();
    if (req.method !== 'GET' && req.method !== 'HEAD') {
      res.statusCode = 405;
      res.setHeader('Allow', 'GET, HEAD');
      res.end();
      return;
    }
    if (pathname === '/api/elements/status') {
      try {
        // A client showing a partial snapshot polls this endpoint; let that poll
        // (not only /api/elements) start the refresh once it is allowed.
        // refreshIfDue() honours every rate-limit guard and never blocks.
        if (req.method === 'GET') refreshIfDue();
        sendJson(req, res, 200, Buffer.from(JSON.stringify(status()), 'utf8'), false);
      } catch (err) {
        next(err);
      }
      return;
    }
    handleElements(req, res).catch(next);
  }

  return {
    handle,
    status,
    refreshIfDue,
    close(closeOpts) {
      if (closed) return;
      closed = true;
      // This service's cycle runner stops after its current partition (a
      // pause / wait ends now); a newer service resumes the cycle.
      const run = G.cycles.get(cacheDir);
      if (run?.owner === token) run.wake?.();
      if (closeOpts?.abortDownload) for (const dl of downloadsHere()) dl.abort('aborted: service closed');
      dataset = null;
      body = null;
    },
    whenIdle,
  };
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

function acceptsGzip(header: string | string[] | undefined): boolean {
  if (!header) return false;
  const h = Array.isArray(header) ? header.join(',') : header;
  for (const part of h.split(',')) {
    const [coding, ...params] = part.trim().toLowerCase().split(';');
    if (coding !== 'gzip' && coding !== '*') continue;
    const qp = params.map((p) => p.trim()).find((p) => p.startsWith('q='));
    if (qp && !(Number(qp.slice(2)) > 0)) continue;
    return true;
  }
  return false;
}

/** Parse "<...> has not updated since <...> YYYY-MM-DD HH:MM[:SS] UTC" => ms, or NaN. */
export function parseLockTimestamp(body: string): number {
  const m = /has not updated since[\s\S]{0,300}?(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}(?::\d{2})?)\s*UTC/i.exec(body);
  if (!m) return NaN;
  return Date.parse(`${m[1]}T${m[2]!.length === 5 ? `${m[2]}:00` : m[2]}Z`);
}

async function readLimited(res: Response, limit: number): Promise<string> {
  if (!res.body) return '';
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let out = '';
  let n = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      n += value.byteLength;
      out += decoder.decode(value, { stream: true });
      if (n >= limit) break;
    }
  } finally {
    reader.cancel().catch(() => {});
  }
  return out + decoder.decode();
}
