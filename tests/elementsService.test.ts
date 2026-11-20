/**
 * Elements service tests. No network: upstreams are fake fetch implementations
 * on made-up .invalid URLs (or the real default URLs answered by a fake fetch);
 * HTTP tests use a local server on 127.0.0.1.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import * as http from 'node:http';
import * as zlib from 'node:zlib';
import { fileURLToPath } from 'node:url';
import {
  CACHE_FILE,
  DEFAULT_MIN_INTERVAL_MS,
  DEFAULT_UPSTREAMS,
  FILE_LOCK_STALE_MS,
  FIXTURE_SOURCE,
  LEGACY_CACHE_FILE,
  DEFAULT_MAX_PARTITION_WAIT_MS,
  DEFAULT_PARTITION_PAUSE_MS,
  LEGACY_TLE_URLS,
  LEGACY_URLS,
  META_FILE,
  PRIMARY_SOURCE,
  SEED_SOURCE,
  STARLINK_PARTITIONS,
  STALE_RECORD_MS,
  USER_AGENT,
  createElementsService,
  mergeElementSets,
  parseLockTimestamp,
  withFileLock,
  type ElementsService,
  type ElementsServiceOptions,
  type MergeSource,
  type Upstream,
} from '../server/elementsService.ts';
import { parseOmmCsv } from '../server/omm.ts';
import { parseTle } from '../server/tle.ts';
import type { ElementRecord, ElementsReady, ElementsStatus, ElementsWarming } from '../src/shared/contracts.ts';

// Safety net: no test may ever reach the network. Every service gets a fake
// fetchImpl; one that is forgotten falls back to this global and fails loudly.
globalThis.fetch = (async (input: string | URL | Request) => {
  throw new Error(`network access is not allowed in tests: ${String(input)}`);
}) as typeof fetch;

const FIXTURE = fileURLToPath(new URL('../fixtures/sup-gp-starlink-partial.tle', import.meta.url));
const fixtureText = fs.readFileSync(FIXTURE, 'utf8');
const fixtureRecords = parseTle(fixtureText).records;
const NO_FIXTURE = path.join(os.tmpdir(), 'does-not-exist', 'none.tle');

const UP_A: Upstream = { name: 'Fake GP', url: 'http://upstream-a.invalid/gp' };
const UP_B: Upstream = { name: 'Fake SupGP', url: 'http://upstream-b.invalid/sup-gp' };
const UP_S: Upstream = { name: 'Fake recent launches', url: 'http://upstream-s.invalid/recent', role: 'supplement', starlinkOnly: true };
const HOUR = 3_600_000;
const T0 = Date.parse('2026-11-20T11:00:00Z');

// ---------------------------------------------------------------------------
// OMM CSV builders
// ---------------------------------------------------------------------------

const CSV_HEADER =
  'OBJECT_NAME,OBJECT_ID,EPOCH,MEAN_MOTION,ECCENTRICITY,INCLINATION,RA_OF_ASC_NODE,ARG_OF_PERICENTER,MEAN_ANOMALY,' +
  'EPHEMERIS_TYPE,CLASSIFICATION_TYPE,NORAD_CAT_ID,ELEMENT_SET_NO,REV_AT_EPOCH,BSTAR,MEAN_MOTION_DOT,MEAN_MOTION_DDOT';

interface Row {
  id: number;
  name?: string;
  objectId?: string;
  /** OMM EPOCH; default 2026-11-20T06:00:00.000000 */
  epoch?: string;
  inc?: number;
  mm?: number;
}

function csvRow(r: Row): string {
  return [
    r.name ?? `STARLINK-${r.id}`,
    r.objectId ?? '2026-200A',
    r.epoch ?? '2026-11-20T06:00:00.000000',
    (r.mm ?? 15.06).toFixed(8),
    '.00012345',
    (r.inc ?? 53.05).toFixed(4),
    ((r.id * 13) % 360).toFixed(4),
    '90.0000',
    ((r.id * 7) % 360).toFixed(4),
    '0',
    'U',
    String(r.id),
    '999',
    '100',
    '.12345000E-3',
    '.10000000E-4',
    '0',
  ].join(',');
}

function csv(rows: Row[]): string {
  return [CSV_HEADER, ...rows.map(csvRow)].join('\r\n') + '\r\n';
}

function range(from: number, n: number, extra: Partial<Row> = {}): Row[] {
  return Array.from({ length: n }, (_, i) => ({ id: from + i, ...extra }));
}

/** 26 Starship V3s (Flight 15, COSPAR 2026-256) + one non-Starlink object, like GROUP=last-30-days. */
const RECENT_ROWS: Row[] = [
  ...Array.from({ length: 26 }, (_, i) => ({
    id: 101400 + i,
    name: `STARLINK-${40096 + i}`,
    objectId: `2026-256${String.fromCharCode(65 + i)}`,
    epoch: '2026-11-20T03:12:00.000000',
    inc: 30.5,
    mm: 15.94,
  })),
  { id: 101142, name: 'XIGUANG-4 01', objectId: '2026-238A', inc: 38.9973, mm: 15.09513871 },
];

const omm = (rows: Row[]): ElementRecord[] => parseOmmCsv(csv(rows)).records;

// Default (partitioned) upstreams.
const PARTS: readonly Upstream[] = DEFAULT_UPSTREAMS.filter((u) => u.role === 'primary');
const SUP: Upstream = DEFAULT_UPSTREAMS.find((u) => u.role === 'supplement')!;
const FALL: Upstream = DEFAULT_UPSTREAMS.find((u) => u.role === 'fallback')!;
const PART_URLS: readonly string[] = PARTS.map((p) => p.url);
const PER_PART = 60;
/** Partitions that hold no objects (CelesTrak answers "No GP data found"). */
const EMPTY_PARTS = new Set(['STARLINK-7', 'STARLINK-8', 'STARLINK-9']);

/** Rows of partition i: PER_PART Starlinks whose names match it (ids 200000 + i*1000 ...). */
function partRows(i: number): Row[] {
  const p = PARTS[i]!.partition!;
  return range(200000 + i * 1000, PER_PART).map((r, k) => ({ ...r, name: `${p}${String(100 + k).padStart(3, '0')}` }));
}

/**
 * A realistic answer for partition i: its Starlinks, plus (NAME= is a
 * substring match) a non-Starlink object and one row overlapping the
 * previous partition (same NORAD id, newer epoch). Empty partitions: see partMap.
 */
function partBody(i: number): string {
  if (EMPTY_PARTS.has(PARTS[i]!.partition!)) return 'No GP data found';
  const rows = partRows(i);
  rows.push({ id: 900000 + i, name: `DEBRIS ${PARTS[i]!.partition} OBJ`, objectId: '2026-001A' });
  if (i > 0 && !EMPTY_PARTS.has(PARTS[i - 1]!.partition!)) {
    const prev = partRows(i - 1)[0]!;
    rows.push({ ...prev, epoch: '2026-11-20T07:00:00.000000' });
  }
  return csv(rows);
}

/** Starlink records the default partitions deliver in total. */
const PART_TOTAL = PARTS.filter((p) => !EMPTY_PARTS.has(p.partition!)).length * PER_PART;

function partMap(overrides: Record<number, Behaviour> = {}, extra: Record<string, Behaviour> = {}): Record<string, Behaviour> {
  const map: Record<string, Behaviour> = { ...extra };
  PARTS.forEach((p, i) => {
    // Like CelesTrak: a query without matches is answered with HTTP 404 "No GP data found".
    const empty: Behaviour = { kind: 'status', status: 404, body: 'No GP data found' };
    map[p.url] = overrides[i] ?? (EMPTY_PARTS.has(p.partition!) ? empty : { kind: 'ok', text: partBody(i) });
  });
  return map;
}

/** Partition-sourced records in a stored set. */
const partIds = (records: readonly ElementRecord[]): number[] =>
  records.map((r) => r.noradId).filter((id) => id >= 200000 && id < 300000);

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

function tmpRoot(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'starlink-elements-'));
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

interface Call {
  url: string;
  headers: Headers;
}

type Behaviour =
  | { kind: 'ok'; text: string; chunkSize?: number; delayMs?: number }
  | { kind: 'status'; status: number; body: string }
  | { kind: 'hang' } // never sends headers; honours abort
  | { kind: 'stall'; text: string } // sends headers + some body, then stalls until abort
  | { kind: 'cut'; text: string; afterBytes: number }; // CelesTrak's server-side cut-off: body ends with "terminated"

/**
 * Fake upstream: url -> behaviour; records every call, and `events`
 * ("start <url>" / "end <url>") to check that requests never overlap.
 * `onCall` runs synchronously when a request arrives.
 */
function fakeFetch(
  map: Record<string, Behaviour>,
  onCall?: (url: string) => void,
): { fetchImpl: typeof fetch; calls: Call[]; events: string[] } {
  const calls: Call[] = [];
  const events: string[] = [];
  const fetchImpl = (async (input: string | URL | Request, init?: RequestInit) => {
    const url = String(input);
    calls.push({ url, headers: new Headers(init?.headers) });
    events.push(`start ${url}`);
    onCall?.(url);
    const end = () => events.push(`end ${url}`);
    const b = map[url];
    if (!b) {
      end();
      throw new TypeError(`unexpected url ${url}`);
    }
    const signal = init?.signal ?? undefined;
    if (b.kind === 'status') {
      end();
      return new Response(b.body, { status: b.status, statusText: 'Forbidden' });
    }
    if (b.kind === 'hang') {
      return new Promise<Response>((_, reject) => {
        signal?.addEventListener(
          'abort',
          () => {
            end();
            reject(signal.reason);
          },
          { once: true },
        );
      });
    }
    const bytes = new TextEncoder().encode(b.text);
    const chunkSize = b.kind === 'ok' ? (b.chunkSize ?? 8192) : b.kind === 'cut' ? b.afterBytes : 4096;
    const delayMs = b.kind === 'ok' ? (b.delayMs ?? 1) : 0;
    let off = 0;
    const stream = new ReadableStream<Uint8Array>({
      async pull(controller) {
        if (signal?.aborted) {
          end();
          return controller.error(signal.reason);
        }
        if (b.kind === 'stall' && off >= chunkSize) {
          await new Promise<void>((resolve) => signal?.addEventListener('abort', () => resolve(), { once: true }));
          end();
          return controller.error(signal?.reason);
        }
        if (b.kind === 'cut' && off >= chunkSize) {
          end();
          return controller.error(new TypeError('terminated'));
        }
        await sleep(delayMs);
        if (off >= bytes.length) {
          end();
          return controller.close();
        }
        controller.enqueue(bytes.slice(off, off + chunkSize));
        off += chunkSize;
      },
    });
    return new Response(stream, { status: 200, headers: { 'Content-Type': 'text/plain' } });
  }) as typeof fetch;
  return { fetchImpl, calls, events };
}

/** Largest number of simultaneously open requests among `urls`. */
function maxConcurrent(events: readonly string[], urls: readonly string[]): number {
  const open = new Set<string>();
  let max = 0;
  for (const e of events) {
    const sp = e.indexOf(' ');
    const kind = e.slice(0, sp);
    const url = e.slice(sp + 1);
    if (!urls.includes(url)) continue;
    if (kind === 'start') open.add(url);
    else open.delete(url);
    max = Math.max(max, open.size);
  }
  return max;
}

function makeService(rootDir: string, clock: { t: number }, extra: Partial<ElementsServiceOptions>): ElementsService {
  return createElementsService({
    rootDir,
    fixturePath: FIXTURE,
    upstreams: [UP_A, UP_B],
    minRecords: 500,
    offline: false,
    partitionPauseMs: 0,
    maxPartitionWaitMs: 0,
    now: () => clock.t,
    log: () => {},
    ...extra,
  });
}

function readMeta(rootDir: string) {
  return JSON.parse(fs.readFileSync(path.join(rootDir, '.cache', META_FILE), 'utf8'));
}

interface Stored {
  format: string;
  version: number;
  fetchedAt: string;
  fullCompletedAt: string | null;
  sources: MergeSource[];
  origin: number[];
  records: ElementRecord[];
}

function readStored(rootDir: string): Stored {
  return JSON.parse(fs.readFileSync(path.join(rootDir, '.cache', CACHE_FILE), 'utf8')) as Stored;
}

/** A merged cache (elements.json) holding the fixture, complete ("full") as of `fetchedAtMs`. */
function seedFullCache(root: string, fetchedAtMs: number, extra: Partial<Stored> = {}): void {
  const dir = path.join(root, '.cache');
  fs.mkdirSync(dir, { recursive: true });
  const at = new Date(fetchedAtMs).toISOString();
  const stored: Stored = {
    format: 'starlink-tracker/elements',
    version: 1,
    fetchedAt: at,
    fullCompletedAt: at,
    sources: [{ name: 'Seeded', url: UP_A.url, fetchedAt: at, complete: true, placeholderIds: false }],
    origin: fixtureRecords.map(() => 0),
    records: fixtureRecords,
    ...extra,
  };
  fs.writeFileSync(path.join(dir, CACHE_FILE), JSON.stringify(stored));
  fs.writeFileSync(
    path.join(dir, META_FILE),
    JSON.stringify({ version: 1, source: 'Seeded', url: UP_A.url, fetchedAt: at, count: 853, partial: false, upstreams: {} }),
  );
}

async function withServer<T>(svc: ElementsService, fn: (port: number) => Promise<T>): Promise<T> {
  const server = http.createServer((req, res) =>
    svc.handle(req, res, (err?: unknown) => {
      res.statusCode = err ? 500 : 404;
      res.end(err ? String(err) : 'next');
    }),
  );
  await new Promise<void>((r) => server.listen(0, '127.0.0.1', r));
  try {
    return await fn((server.address() as { port: number }).port);
  } finally {
    await new Promise((r) => server.close(r));
  }
}

function get(port: number, p: string, headers: Record<string, string> = {}): Promise<{ status: number; headers: http.IncomingHttpHeaders; body: Buffer }> {
  return new Promise((resolve, reject) => {
    http
      .get({ host: '127.0.0.1', port, path: p, headers }, (res) => {
        const chunks: Buffer[] = [];
        res.on('data', (c: Buffer) => chunks.push(c));
        res.on('end', () => resolve({ status: res.statusCode ?? 0, headers: res.headers, body: Buffer.concat(chunks) }));
      })
      .on('error', reject);
  });
}

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------

test('default upstreams: 18 NAME= partitions as primary, recent-launches supplement, SupGP fallback; no TLE, no GROUP=starlink', () => {
  assert.equal(DEFAULT_UPSTREAMS.length, 20);
  assert.deepEqual(STARLINK_PARTITIONS, [
    'STARLINK-1', 'STARLINK-2', 'STARLINK-4', 'STARLINK-5', 'STARLINK-6', 'STARLINK-7', 'STARLINK-8', 'STARLINK-9',
    'STARLINK-30', 'STARLINK-31', 'STARLINK-32', 'STARLINK-33', 'STARLINK-34',
    'STARLINK-35', 'STARLINK-36', 'STARLINK-37', 'STARLINK-38', 'STARLINK-39',
  ]);
  PARTS.forEach((p, i) => {
    assert.equal(p.role, 'primary');
    assert.equal(p.url, `https://celestrak.org/NORAD/elements/gp.php?NAME=${STARLINK_PARTITIONS[i]}&FORMAT=csv`);
    assert.equal(p.partition, STARLINK_PARTITIONS[i]);
    assert.equal(p.sourceName, PRIMARY_SOURCE);
    assert.equal(p.starlinkOnly, true);
  });
  assert.equal(PRIMARY_SOURCE, 'CelesTrak GP (18 SDS)');
  // Every 4-digit (1007..6xxx, 3xxx via -30..-39) and 5-digit (30000..40xxx) Starlink name is covered.
  for (const n of [1007, 1999, 2001, 3001, 3999, 4123, 5999, 6100, 30000, 31234, 35999, 39999, 40096]) {
    const name = `STARLINK-${n}`;
    assert.ok(STARLINK_PARTITIONS.some((p) => name.includes(p)), name);
  }
  const [s, f] = DEFAULT_UPSTREAMS.slice(PARTS.length);
  assert.equal(s!.name, 'CelesTrak recent launches');
  assert.equal(s!.url, 'https://celestrak.org/NORAD/elements/gp.php?GROUP=last-30-days&FORMAT=csv');
  assert.equal(s!.role, 'supplement');
  assert.equal(s!.starlinkOnly, true);
  assert.equal(f!.name, 'CelesTrak SupGP (SpaceX ephemerides)');
  assert.equal(f!.url, 'https://celestrak.org/NORAD/elements/supplemental/sup-gp.php?FILE=starlink&FORMAT=csv');
  assert.equal(f!.role, 'fallback');
  assert.equal(f!.placeholderIds, true);
  for (const up of DEFAULT_UPSTREAMS) {
    assert.ok(!/FORMAT=tle/i.test(up.url));
    assert.ok(!/GROUP=starlink\b/i.test(up.url));
    assert.ok(!LEGACY_URLS.includes(up.url));
  }
  for (const u of LEGACY_TLE_URLS) assert.ok(LEGACY_URLS.includes(u));
  assert.ok(LEGACY_URLS.includes('https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=csv'));
  assert.equal(DEFAULT_PARTITION_PAUSE_MS, 2000);
  assert.equal(DEFAULT_MAX_PARTITION_WAIT_MS, 15 * 60_000);
  assert.equal(DEFAULT_MIN_INTERVAL_MS, 2 * HOUR + 60_000);
  assert.equal(USER_AGENT, 'starlink-tracker-dev/0.1');
  assert.equal(CACHE_FILE, 'elements.json');
  assert.equal(LEGACY_CACHE_FILE, 'starlink.tle');
  assert.equal(STALE_RECORD_MS, 30 * 24 * HOUR);
});

// ---------------------------------------------------------------------------
// Merge semantics (pure)
// ---------------------------------------------------------------------------

const SRC = (name: string, placeholderIds = false): MergeSource => ({
  name,
  url: `http://${name}.invalid`,
  fetchedAt: new Date(T0).toISOString(),
  complete: true,
  placeholderIds,
});

test('merge: union by NORAD id, newest epoch wins, equal epoch -> incoming, sorted, counts', () => {
  const base = omm([
    { id: 101003, epoch: '2026-11-20T06:00:00' },
    { id: 101001, epoch: '2026-11-20T06:00:00' },
    { id: 101002, epoch: '2026-11-20T06:00:00' },
  ]);
  const incoming = omm([
    { id: 101001, epoch: '2026-11-19T06:00:00', name: 'OLDER' }, // older: ignored
    { id: 101002, epoch: '2026-11-20T08:00:00', name: 'NEWER' }, // newer: replaces
    { id: 101003, epoch: '2026-11-20T06:00:00', name: 'SAME' }, // same epoch: incoming wins (not "updated")
    { id: 101004 }, // new
  ]);
  const m = mergeElementSets({ records: base, origin: [0, 0, 0], sources: [SRC('base')] }, incoming, SRC('new'), T0);
  assert.deepEqual(m.records.map((r) => r.noradId), [101001, 101002, 101003, 101004]);
  assert.deepEqual(m.records.map((r) => r.name), ['STARLINK-101001', 'NEWER', 'SAME', 'STARLINK-101004']);
  assert.deepEqual(m.sources.map((s) => s.name), ['base', 'new']);
  assert.deepEqual(m.origin, [0, 1, 1, 1]);
  assert.deepEqual(m.counts, [1, 3]);
  assert.equal(m.added, 1);
  assert.equal(m.updated, 1);
  assert.equal(m.dropped, 0);
  // Inputs are not mutated.
  assert.equal(base.length, 3);
});

test('merge: records > 30 days older than the newest epoch are dropped; bogus future epochs do not count', () => {
  const base = omm([
    { id: 1, epoch: '2026-10-01T00:00:00' }, // 50 days older: stale
    { id: 2, epoch: '2026-10-25T00:00:00' }, // 26 days older: kept
    { id: 3, epoch: '2027-06-01T00:00:00' }, // far future (bad data): not the reference
  ]);
  const incoming = omm([{ id: 4, epoch: '2026-11-20T06:00:00' }]);
  const m = mergeElementSets({ records: base, origin: [0, 0, 0], sources: [SRC('base')] }, incoming, SRC('new'), T0);
  assert.deepEqual(m.records.map((r) => r.noradId), [2, 3, 4]);
  assert.equal(m.dropped, 1);
  // TLE and OMM records mix; the old TLE fixture becomes stale against a much newer set.
  const later = omm([{ id: 200001, epoch: '2027-02-01T00:00:00' }]);
  const m2 = mergeElementSets(
    { records: fixtureRecords, origin: fixtureRecords.map(() => 0), sources: [SRC('fixture')] },
    later,
    SRC('new'),
    Date.parse('2027-02-01T12:00:00Z'),
  );
  assert.deepEqual(m2.records.map((r) => r.noradId), [200001]);
  assert.deepEqual(m2.sources.map((s) => s.name), ['new'], 'sources without records are pruned');
  assert.equal(m2.dropped, 853);
});

test('merge: SupGP placeholder ids never duplicate a real GP object', () => {
  const gp = omm([
    { id: 101415, name: 'STARLINK-40096', objectId: '2026-256A' },
    { id: 101416, name: 'STARLINK-40097', objectId: '2026-256B' },
  ]);
  const supgp = omm([
    { id: 270001, name: 'STARLINK-40096', objectId: '2026-256A', epoch: '2026-11-20T07:00:00' }, // placeholder dup
    { id: 270002, name: 'STARLINK-40097', objectId: '' }, // dup (no COSPAR id)
    { id: 270003, name: 'STARLINK-40098', objectId: '2026-256C' }, // not in GP yet: kept
    { id: 101416, name: 'STARLINK-40097', objectId: '2026-256B', epoch: '2026-11-20T07:00:00' }, // real id: newer epoch wins
  ]);
  const m = mergeElementSets({ records: gp, origin: [0, 0], sources: [SRC('gp')] }, supgp, SRC('supgp', true), T0);
  assert.deepEqual(m.records.map((r) => r.noradId), [101415, 101416, 270003]);
  assert.equal(m.duplicates, 2);
  // Later the GP catalogue assigns the real id: the placeholder record goes away.
  const real = omm([{ id: 101417, name: 'STARLINK-40098', objectId: '2026-256C' }]);
  const m2 = mergeElementSets({ records: m.records, origin: m.origin, sources: m.sources }, real, SRC('gp'), T0);
  assert.deepEqual(m2.records.map((r) => r.noradId), [101415, 101416, 101417]);
  assert.equal(m2.duplicates, 1);
  // Same name but a different COSPAR id is a different object.
  const other = omm([{ id: 270009, name: 'STARLINK-40096', objectId: '2026-300A' }]);
  const m3 = mergeElementSets({ records: gp, origin: [0, 0], sources: [SRC('gp')] }, other, SRC('supgp', true), T0);
  assert.equal(m3.records.length, 3);
});

test('merge: SupGP placeholder -> real id transition leaves no duplicate satellite', () => {
  const empty = { records: [], origin: [], sources: [] };
  // Run k: placeholder id. Run k+1: the same object under its real id (not in GP yet).
  const k = mergeElementSets(empty, omm([{ id: 270001, name: 'STARLINK-41234', epoch: '2026-11-20T05:00:00' }]), SRC('supgp', true), T0);
  const k1 = mergeElementSets(k, omm([{ id: 101234, name: 'STARLINK-41234', epoch: '2026-11-20T07:00:00' }]), SRC('supgp', true), T0);
  assert.deepEqual(k1.records.map((r) => r.noradId), [101234]);
  assert.equal(k1.duplicates, 1);
  // GP later publishes the real id (existing branch): still exactly one record.
  const gp = mergeElementSets(k1, omm([{ id: 101234, name: 'STARLINK-41234', epoch: '2026-11-20T08:00:00' }]), SRC('gp'), T0);
  assert.deepEqual(gp.records.map((r) => r.noradId), [101234]);
  assert.deepEqual(gp.sources.map((s) => s.name), ['gp']);

  // A placeholder twin still present when GP re-delivers an id it already has: the twin goes.
  const base = {
    records: omm([
      { id: 101234, name: 'STARLINK-41234', epoch: '2026-11-20T06:00:00' },
      { id: 270001, name: 'STARLINK-41234', epoch: '2026-11-20T07:00:00' },
    ]),
    origin: [0, 1],
    sources: [SRC('gp'), SRC('supgp', true)],
  };
  const again = mergeElementSets(base, omm([{ id: 101234, name: 'STARLINK-41234', epoch: '2026-11-20T06:00:00' }]), SRC('gp'), T0);
  assert.deepEqual(again.records.map((r) => r.noradId), [101234]);
  assert.equal(again.duplicates, 1);
  // Even an older GP epoch (record kept as is) removes the twin.
  const older = mergeElementSets(base, omm([{ id: 101234, name: 'STARLINK-41234', epoch: '2026-11-20T04:00:00' }]), SRC('gp'), T0);
  assert.deepEqual(older.records.map((r) => r.noradId), [101234]);

  // Two placeholder rows of one SupGP file never wipe out a newer one.
  const pair = mergeElementSets(
    empty,
    omm([
      { id: 270002, name: 'STARLINK-41235', epoch: '2026-11-20T07:00:00' },
      { id: 270003, name: 'STARLINK-41235', epoch: '2026-11-20T06:00:00' },
    ]),
    SRC('supgp', true),
    T0,
  );
  assert.deepEqual(pair.records.map((r) => r.noradId), [270002, 270003]);
});

// ---------------------------------------------------------------------------
// Serving
// ---------------------------------------------------------------------------

test('offline: serves bundled fixture, never contacts upstream', async () => {
  const root = tmpRoot();
  const clock = { t: T0 };
  const { fetchImpl, calls } = fakeFetch({});
  const svc = makeService(root, clock, { offline: true, fetchImpl });
  await withServer(svc, async (port) => {
    const r = await get(port, '/api/elements');
    assert.equal(r.status, 200);
    const body = JSON.parse(r.body.toString()) as ElementsReady;
    assert.equal(body.status, 'ready');
    assert.equal(body.source, FIXTURE_SOURCE);
    assert.equal(body.partial, true);
    assert.equal(body.count, 853);
    assert.equal(body.records.length, 853);
    assert.equal(body.records[0]!.name, 'STARLINK-1008');
    assert.deepEqual(body.sources?.map((s) => [s.name, s.records, s.complete]), [[FIXTURE_SOURCE, 853, false]]);
    assert.ok(Date.parse(body.fetchedAt) > Date.parse('2026-11-19T00:00:00Z'));
    const st = JSON.parse((await get(port, '/api/elements/status')).body.toString()) as ElementsStatus;
    assert.equal(st.hasCache, false);
    assert.equal(st.cachePartial, true);
    assert.equal(st.cacheCount, 853);
    assert.equal(st.nextAllowedFetchAt, null);
    assert.deepEqual(st.downloads, []);
  });
  svc.refreshIfDue();
  assert.equal(calls.length, 0);
  svc.close();
});

test('offline with neither cache nor fixture: 503 ElementsWarming', async () => {
  const root = tmpRoot();
  const svc = makeService(root, { t: T0 }, { offline: true, fixturePath: NO_FIXTURE });
  await withServer(svc, async (port) => {
    const r = await get(port, '/api/elements');
    assert.equal(r.status, 503);
    const body = JSON.parse(r.body.toString()) as ElementsWarming;
    assert.equal(body.status, 'warming');
    assert.equal(body.startedAt, null);
    assert.match(body.message, /offline/);
  });
  svc.close();
});

test('CELESTRAK_OFFLINE=1 env enables offline mode', () => {
  const prev = process.env.CELESTRAK_OFFLINE;
  process.env.CELESTRAK_OFFLINE = '1';
  try {
    const { fetchImpl, calls } = fakeFetch({});
    const svc = createElementsService({ rootDir: tmpRoot(), fixturePath: FIXTURE, upstreams: [UP_A, UP_S], fetchImpl, log: () => {} });
    svc.refreshIfDue();
    assert.equal(calls.length, 0);
    assert.equal(svc.status().fetching, false);
    svc.close();
  } finally {
    if (prev === undefined) delete process.env.CELESTRAK_OFFLINE;
    else process.env.CELESTRAK_OFFLINE = prev;
  }
});

test('background OMM CSV refresh: fixture served meanwhile, live progress, merged set committed atomically', async () => {
  const root = tmpRoot();
  const clock = { t: T0 };
  const primaryCsv = csv(range(200001, 1200));
  const { fetchImpl, calls } = fakeFetch({ [UP_A.url]: { kind: 'ok', text: primaryCsv, chunkSize: 2048, delayMs: 2 } });
  const svc = makeService(root, clock, { fetchImpl });
  await withServer(svc, async (port) => {
    // First request triggers the refresh but is answered immediately from the fixture.
    const r1 = await get(port, '/api/elements');
    assert.equal(r1.status, 200);
    assert.equal((JSON.parse(r1.body.toString()) as ElementsReady).source, FIXTURE_SOURCE);
    assert.equal(svc.status().fetching, true);

    // Only one download in flight.
    for (let i = 0; i < 5; i++) svc.refreshIfDue();
    await get(port, '/api/elements');

    // Live progress (CSV: one line per record).
    let sawProgress = false;
    for (let i = 0; i < 400 && svc.status().fetching; i++) {
      const st = svc.status();
      if (st.bytesReceived > 0 && st.recordsReceived > 0) {
        sawProgress = true;
        assert.ok(st.recordsReceived <= 1200);
        assert.equal(st.downloads?.[0]?.name, UP_A.name);
      }
      await sleep(5);
    }
    assert.ok(sawProgress, 'expected live byte/record progress');
    await svc.whenIdle();
    assert.equal(calls.length, 1);
    assert.equal(calls[0]!.url, UP_A.url);
    assert.equal(calls[0]!.headers.get('user-agent'), USER_AGENT);
    assert.match(calls[0]!.headers.get('accept') ?? '', /text\/csv/);

    // Merged cache + meta on disk, no temp files left behind.
    const files = fs.readdirSync(path.join(root, '.cache')).sort();
    assert.deepEqual(files, [CACHE_FILE, META_FILE].sort());
    const stored = readStored(root);
    assert.equal(stored.format, 'starlink-tracker/elements');
    assert.equal(stored.records.length, 853 + 1200);
    assert.equal(stored.fullCompletedAt, new Date(T0).toISOString());
    const meta = readMeta(root);
    assert.equal(meta.source, `${UP_A.name} + ${FIXTURE_SOURCE}`);
    assert.equal(meta.url, UP_A.url);
    assert.equal(meta.count, 2053);
    assert.equal(meta.partial, false);
    assert.equal(meta.fetchedAt, new Date(T0).toISOString());
    assert.equal(meta.upstreams[UP_A.url].lastAttemptAt, new Date(T0).toISOString());
    assert.equal(meta.upstreams[UP_A.url].nextAllowedAt, new Date(T0 + DEFAULT_MIN_INTERVAL_MS).toISOString());
    assert.equal(meta.upstreams[UP_A.url].lastError, null);
    assert.equal(meta.upstreams[UP_A.url].lastSuccessAt, new Date(T0).toISOString());

    const r2 = JSON.parse((await get(port, '/api/elements')).body.toString()) as ElementsReady;
    assert.equal(r2.source, `${UP_A.name} + ${FIXTURE_SOURCE}`);
    assert.equal(r2.partial, false);
    assert.equal(r2.count, 2053);
    assert.deepEqual(r2.sources, [
      { name: UP_A.name, fetchedAt: new Date(T0).toISOString(), records: 1200, complete: true },
      { name: FIXTURE_SOURCE, fetchedAt: r2.sources![1]!.fetchedAt, records: 853, complete: false },
    ]);
    assert.equal(r2.nextRefreshEarliest, new Date(T0 + DEFAULT_MIN_INTERVAL_MS).toISOString());
    const big = r2.records.find((r) => r.noradId === 200001)!;
    assert.equal(big.intlDesignator, '26200A');
    assert.equal(big.omm?.NORAD_CAT_ID, 200001);
    assert.equal(big.line1, undefined);
    assert.ok(r2.records.find((r) => r.noradId === 44714)?.line1, 'fixture TLE records stay TLE');
    const st = svc.status();
    assert.equal(st.hasCache, true);
    assert.equal(st.cachePartial, false);
    assert.equal(st.fetching, false);
  });
  svc.close();

  // Restart inside the window: no upstream hit; merged cache served.
  const { fetchImpl: f2, calls: calls2 } = fakeFetch({ [UP_A.url]: { kind: 'ok', text: primaryCsv } });
  clock.t = T0 + HOUR;
  const svc2 = makeService(root, clock, { fetchImpl: f2 });
  svc2.refreshIfDue();
  assert.equal(svc2.status().hasCache, true);
  assert.equal(svc2.status().cacheCount, 2053);
  assert.equal(calls2.length, 0);
  // After the window the primary is contacted again (once).
  clock.t = T0 + DEFAULT_MIN_INTERVAL_MS;
  svc2.refreshIfDue();
  svc2.refreshIfDue();
  await svc2.whenIdle();
  assert.equal(calls2.length, 1);
  svc2.close();
});

test('failed attempt blocks the URL for the window, persisted across restart; fallback only without full data', async () => {
  const root = tmpRoot();
  const clock = { t: T0 };
  const { fetchImpl, calls } = fakeFetch({
    [UP_A.url]: { kind: 'status', status: 500, body: 'Internal error' },
    [UP_B.url]: { kind: 'status', status: 502, body: 'Bad gateway' },
  });
  const svc = makeService(root, clock, { fetchImpl });
  svc.refreshIfDue();
  await svc.whenIdle();
  assert.deepEqual(calls.map((c) => c.url), [UP_A.url]);
  assert.match(svc.status().lastError ?? '', /HTTP 500/);

  // A failed and no full data -> falls back to B (once).
  clock.t += 1000;
  svc.refreshIfDue();
  await svc.whenIdle();
  svc.refreshIfDue();
  await svc.whenIdle();
  assert.deepEqual(calls.map((c) => c.url), [UP_A.url, UP_B.url]);
  svc.close();

  // Restart: both URLs still inside their window -> nothing.
  clock.t = T0 + HOUR;
  const f2 = fakeFetch({ [UP_A.url]: { kind: 'ok', text: csv(range(200001, 600)) }, [UP_B.url]: { kind: 'ok', text: fixtureText } });
  const svc2 = makeService(root, clock, { fetchImpl: f2.fetchImpl });
  svc2.refreshIfDue();
  await svc2.whenIdle();
  assert.equal(f2.calls.length, 0);
  const meta = readMeta(root);
  assert.equal(meta.upstreams[UP_A.url].lastAttemptAt, new Date(T0).toISOString());
  assert.match(meta.upstreams[UP_B.url].lastError, /HTTP 502/);
  assert.equal(svc2.status().nextAllowedFetchAt, new Date(T0 + DEFAULT_MIN_INTERVAL_MS).toISOString());

  // Window over -> primary again.
  clock.t = T0 + DEFAULT_MIN_INTERVAL_MS;
  svc2.refreshIfDue();
  await svc2.whenIdle();
  assert.deepEqual(f2.calls.map((c) => c.url), [UP_A.url]);
  assert.equal(svc2.status().hasCache, true);
  assert.equal(svc2.status().cachePartial, false);
  svc2.close();
});

test('403 "has not updated since" locks until timestamp + 2h + 1min; cache kept; no fallback with full data', async () => {
  assert.equal(parseLockTimestamp('GP data has not updated since 2026-11-20 10:00:00 UTC.'), Date.parse('2026-11-20T10:00:00Z'));
  assert.ok(Number.isNaN(parseLockTimestamp('Forbidden')));

  const root = tmpRoot();
  seedFullCache(root, T0 - 3 * HOUR);
  const clock = { t: T0 };
  const lockBody =
    'GP data has not updated since your last successful download of GROUP=starlink at 2026-11-20 10:00:00 UTC.\n' +
    'Data is updated once every 2 hours.';
  const { fetchImpl, calls } = fakeFetch({
    [UP_A.url]: { kind: 'status', status: 403, body: lockBody },
    [UP_B.url]: { kind: 'ok', text: fixtureText },
  });
  const svc = makeService(root, clock, { fetchImpl });
  await withServer(svc, async (port) => {
    await get(port, '/api/elements');
    await svc.whenIdle();
    assert.deepEqual(calls.map((c) => c.url), [UP_A.url]);
    const meta = readMeta(root);
    assert.equal(meta.upstreams[UP_A.url].nextAllowedAt, '2026-11-20T12:01:00.000Z');
    assert.match(meta.upstreams[UP_A.url].lastError, /403/);
    assert.equal(meta.source, 'Seeded'); // cache untouched

    const r = JSON.parse((await get(port, '/api/elements')).body.toString()) as ElementsReady;
    assert.equal(r.source, 'Seeded');
    assert.equal(r.count, 853);
    assert.equal(r.partial, false);
    assert.equal(r.nextRefreshEarliest, '2026-11-20T12:01:00.000Z');
    await svc.whenIdle();
    assert.equal(calls.length, 1, 'no retry and no fallback while recent full data exists');
  });

  // Lock lifts at the parsed time (earlier than the default window).
  clock.t = Date.parse('2026-11-20T12:01:00Z');
  svc.refreshIfDue();
  await svc.whenIdle();
  assert.equal(calls.length, 2);
  svc.close();
});

test('403 lock with a stale timestamp falls back to the full window', async () => {
  const root = tmpRoot();
  seedFullCache(root, T0 - 3 * HOUR);
  const clock = { t: T0 };
  const { fetchImpl } = fakeFetch({
    [UP_A.url]: { kind: 'status', status: 403, body: 'Data has not updated since 2026-11-19 01:00:00 UTC' },
  });
  const svc = makeService(root, clock, { fetchImpl });
  svc.refreshIfDue();
  await svc.whenIdle();
  assert.equal(readMeta(root).upstreams[UP_A.url].nextAllowedAt, new Date(T0 + DEFAULT_MIN_INTERVAL_MS).toISOString());
  svc.close();
});

test('timeouts count as attempts (headers never arrive / body stalls); a stalled body still merges what it got', async () => {
  for (const kind of ['hang', 'stall'] as const) {
    const root = tmpRoot();
    const clock = { t: T0 };
    const behaviour: Behaviour = kind === 'hang' ? { kind } : { kind, text: csv(range(300001, 500)) };
    const { fetchImpl, calls } = fakeFetch({ [UP_A.url]: behaviour });
    const svc = makeService(root, clock, { fetchImpl, upstreams: [UP_A], requestTimeoutMs: 40 });
    svc.refreshIfDue();
    await svc.whenIdle();
    const meta = readMeta(root);
    assert.match(meta.upstreams[UP_A.url].lastError, /timed out/, kind);
    assert.equal(meta.upstreams[UP_A.url].lastAttemptAt, new Date(T0).toISOString());
    const st = svc.status();
    if (kind === 'hang') {
      assert.equal(fs.existsSync(path.join(root, '.cache', CACHE_FILE)), false, 'nothing received, nothing cached');
      assert.equal(st.cacheCount, 853);
    } else {
      // ~4 KiB of CSV: a few dozen complete rows; the cut row is ignored.
      const got = st.cacheCount - 853;
      assert.ok(got > 10 && got < 40, `merged ${got} rows from the stalled download`);
      assert.equal(st.cachePartial, true);
      assert.equal(st.hasCache, true);
      assert.equal(readStored(root).fullCompletedAt, null);
      assert.equal(readMeta(root).partial, true);
    }
    clock.t += 10 * 60_000;
    svc.refreshIfDue();
    await svc.whenIdle();
    assert.equal(calls.length, 1, `${kind}: no retry inside the window`);
    svc.close();
  }
});

test('a cut-off primary download refreshes and extends a full set without making it partial', async () => {
  const root = tmpRoot();
  seedFullCache(root, T0 - 5 * HOUR);
  const clock = { t: T0 };
  // Newer epochs for fixture ids 44714.. (first rows) + new ids, cut off after ~4 KiB.
  const rows: Row[] = [
    ...fixtureRecords.slice(0, 10).map((r) => ({ id: r.noradId, name: r.name, objectId: '2019-074A', epoch: '2026-11-20T10:00:00.000000' })),
    ...range(400001, 300),
  ];
  const { fetchImpl } = fakeFetch({ [UP_A.url]: { kind: 'stall', text: csv(rows) } });
  const svc = makeService(root, clock, { fetchImpl, upstreams: [UP_A], requestTimeoutMs: 40 });
  svc.refreshIfDue();
  await svc.whenIdle();
  const stored = readStored(root);
  assert.equal(stored.fullCompletedAt, new Date(T0 - 5 * HOUR).toISOString(), 'still the earlier complete download');
  const byId = new Map(stored.records.map((r) => [r.noradId, r]));
  assert.ok(byId.get(fixtureRecords[0]!.noradId)!.omm, 'refreshed with the newer OMM element set');
  assert.ok(stored.records.length > 853);
  const st = svc.status();
  assert.equal(st.cachePartial, false);
  assert.equal(st.cacheCount, stored.records.length);
  const a = st.sources!.find((s) => s.name === UP_A.name)!;
  assert.equal(a.complete, false);
  assert.match(readMeta(root).upstreams[UP_A.url].lastError, /timed out/);
  svc.close();
});

test('implausibly small complete result is not committed', async () => {
  const root = tmpRoot();
  const clock = { t: T0 };
  const few = fixtureText.split('\n').slice(0, 30).join('\n');
  const { fetchImpl } = fakeFetch({ [UP_A.url]: { kind: 'ok', text: few } });
  const svc = makeService(root, clock, { fetchImpl, upstreams: [UP_A] });
  svc.refreshIfDue();
  await svc.whenIdle();
  assert.equal(fs.existsSync(path.join(root, '.cache', CACHE_FILE)), false);
  assert.match(svc.status().lastError ?? '', /implausible/);
  assert.equal(svc.status().cacheCount, 853); // still the fixture
  svc.close();
});

test('no cache, no fixture: 503 with live progress, then 200', async () => {
  const root = tmpRoot();
  const clock = { t: T0 };
  const { fetchImpl } = fakeFetch({ [UP_A.url]: { kind: 'ok', text: fixtureText, chunkSize: 1024, delayMs: 3 } });
  const svc = makeService(root, clock, { fetchImpl, fixturePath: NO_FIXTURE });
  await withServer(svc, async (port) => {
    const r = await get(port, '/api/elements');
    assert.equal(r.status, 503);
    assert.equal(r.headers['cache-control'], 'no-store');
    let progressed: ElementsWarming | null = null;
    for (let i = 0; i < 200; i++) {
      await sleep(10);
      const w = await get(port, '/api/elements');
      if (w.status !== 503) break;
      const body = JSON.parse(w.body.toString()) as ElementsWarming;
      assert.equal(body.startedAt, new Date(T0).toISOString());
      assert.match(body.message, /Fake GP/);
      if (body.bytesReceived > 0 && body.recordsReceived > 0) progressed = body;
    }
    assert.ok(progressed, 'expected a 503 with progress');
    await svc.whenIdle();
    const ok = await get(port, '/api/elements');
    assert.equal(ok.status, 200);
    assert.equal((JSON.parse(ok.body.toString()) as ElementsReady).source, UP_A.name);
  });
  svc.close();
});

test('gzip response, cached body reuse, headers, routing', async () => {
  const root = tmpRoot();
  const svc = makeService(root, { t: T0 }, { offline: true });
  await withServer(svc, async (port) => {
    const plain = await get(port, '/api/elements');
    assert.equal(plain.status, 200);
    assert.equal(plain.headers['content-encoding'], undefined);
    assert.equal(plain.headers['cache-control'], 'no-store');
    assert.match(plain.headers['content-type'] ?? '', /^application\/json/);

    const gz1 = await get(port, '/api/elements?x=1', { 'Accept-Encoding': 'gzip, deflate, br' });
    assert.equal(gz1.headers['content-encoding'], 'gzip');
    assert.equal(gz1.headers.vary, 'Accept-Encoding');
    assert.ok(gz1.body.length < plain.body.length / 2);
    const unzipped = zlib.gunzipSync(gz1.body);
    assert.ok(unzipped.equals(plain.body));
    const parsed = JSON.parse(unzipped.toString()) as ElementsReady;
    assert.equal(parsed.count, 853);

    const gz2 = await get(port, '/api/elements', { 'Accept-Encoding': 'gzip' });
    assert.ok(gz2.body.equals(gz1.body));
    const noGz = await get(port, '/api/elements', { 'Accept-Encoding': 'gzip;q=0' });
    assert.equal(noGz.headers['content-encoding'], undefined);

    assert.equal((await get(port, '/api/other')).body.toString(), 'next');
    assert.equal((await get(port, '/')).body.toString(), 'next');
    assert.equal((await get(port, '/api/elements/status')).status, 200);
  });
  svc.close();
});

test('external cache changes are picked up: TLE seed re-parsed on change, elements.json takes precedence', async () => {
  const root = tmpRoot();
  const dir = path.join(root, '.cache');
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, LEGACY_CACHE_FILE), fixtureText);
  const svc = makeService(root, { t: T0 }, { offline: true });
  assert.equal(svc.status().cacheCount, 853);
  assert.equal(svc.status().hasCache, true);
  const half = fixtureText.split('\n').slice(0, 300).join('\n') + '\n';
  fs.writeFileSync(path.join(dir, LEGACY_CACHE_FILE), half);
  assert.equal(svc.status().cacheCount, 100);
  seedFullCache(root, T0, { records: fixtureRecords.slice(0, 200), origin: fixtureRecords.slice(0, 200).map(() => 0) });
  assert.equal(svc.status().cacheCount, 200);
  assert.equal(svc.status().cachePartial, false);
  // A corrupt elements.json falls back to the seed instead of failing.
  fs.writeFileSync(path.join(dir, CACHE_FILE), Buffer.alloc(64));
  assert.equal(svc.status().cacheCount, 100);
  svc.close();
});

test('interrupted download is kept as partial data; fallback still allowed and completes the set', async () => {
  const root = tmpRoot();
  const clock = { t: T0 };
  const { fetchImpl, calls } = fakeFetch({
    [UP_A.url]: { kind: 'stall', text: fixtureText },
    [UP_B.url]: { kind: 'ok', text: fixtureText, chunkSize: 1 << 20, delayMs: 0 },
  });
  const svc = makeService(root, clock, { fetchImpl, fixturePath: NO_FIXTURE, requestTimeoutMs: 40 });
  svc.refreshIfDue();
  await svc.whenIdle();
  const st = svc.status();
  assert.equal(st.hasCache, true);
  assert.equal(st.cachePartial, true);
  assert.ok(st.cacheCount > 0 && st.cacheCount < 853);
  assert.equal(readMeta(root).partial, true);
  // Partial data is not a "usable full set" -> next refresh falls back to B.
  svc.refreshIfDue();
  await svc.whenIdle();
  assert.deepEqual(calls.map((c) => c.url), [UP_A.url, UP_B.url]);
  const done = svc.status();
  assert.equal(done.cachePartial, false);
  assert.equal(done.cacheCount, 853);
  svc.close();
});

test('a full download older than 12 h no longer blocks fallback when the primary keeps failing', async () => {
  const root = tmpRoot();
  seedFullCache(root, T0 - 13 * HOUR);
  const clock = { t: T0 };
  const { fetchImpl, calls } = fakeFetch({
    [UP_A.url]: { kind: 'status', status: 404, body: 'Not found' },
    [UP_B.url]: { kind: 'ok', text: fixtureText },
  });
  const svc = makeService(root, clock, { fetchImpl });
  svc.refreshIfDue();
  await svc.whenIdle();
  assert.deepEqual(calls.map((c) => c.url), [UP_A.url]);
  clock.t += 1000;
  svc.refreshIfDue();
  await svc.whenIdle();
  assert.deepEqual(calls.map((c) => c.url), [UP_A.url, UP_B.url]);
  assert.equal(readMeta(root).source, UP_B.name);
  // Fresh again: nothing more inside the window.
  clock.t += HOUR;
  svc.refreshIfDue();
  await svc.whenIdle();
  assert.equal(calls.length, 2);
  svc.close();
});

// ---------------------------------------------------------------------------
// Supplement + migration
// ---------------------------------------------------------------------------

test('supplement: own window, STARLINK rows only, lands while the primary download is still running', async () => {
  const root = tmpRoot();
  const clock = { t: T0 };
  const { fetchImpl, calls } = fakeFetch({
    [UP_A.url]: { kind: 'hang' },
    [UP_S.url]: { kind: 'ok', text: csv(RECENT_ROWS) },
  });
  const svc = makeService(root, clock, { fetchImpl, upstreams: [UP_A, UP_S], requestTimeoutMs: 400 });
  svc.refreshIfDue();
  svc.refreshIfDue();
  let st = svc.status();
  for (let i = 0; i < 300 && (!st.sources?.some((s) => s.name === UP_S.name) || st.downloads?.length !== 1); i++) {
    await sleep(5);
    st = svc.status();
  }
  // The V3s are served while the (slow) primary is still downloading.
  assert.equal(st.fetching, true);
  assert.deepEqual(st.downloads?.map((d) => d.name), [UP_A.name]);
  assert.equal(st.cacheCount, 853 + 26);
  assert.equal(st.cachePartial, true, 'a supplement never makes the set complete');
  const sup = st.sources!.find((s) => s.name === UP_S.name)!;
  assert.equal(sup.records, 26);
  assert.equal(sup.complete, true);
  const stored = readStored(root);
  const ids = new Set(stored.records.map((r) => r.noradId));
  assert.ok(ids.has(101400) && ids.has(101425));
  assert.ok(!ids.has(101142), 'non-Starlink objects are filtered out');
  const v3 = stored.records.find((r) => r.noradId === 101415)!;
  assert.equal(v3.name, 'STARLINK-40111');
  assert.equal(v3.intlDesignator, '26256P');
  assert.equal(v3.omm?.INCLINATION, 30.5);

  await svc.whenIdle();
  assert.deepEqual(calls.map((c) => c.url).sort(), [UP_A.url, UP_S.url].sort());
  const meta = readMeta(root);
  assert.match(meta.upstreams[UP_A.url].lastError, /timed out/);
  assert.equal(meta.upstreams[UP_S.url].lastError, null);
  assert.equal(meta.upstreams[UP_S.url].lastSuccessAt, new Date(T0).toISOString());
  assert.equal(meta.upstreams[UP_S.url].nextAllowedAt, new Date(T0 + DEFAULT_MIN_INTERVAL_MS).toISOString());

  // Inside both windows: nothing.
  clock.t = T0 + HOUR;
  svc.refreshIfDue();
  await svc.whenIdle();
  assert.equal(calls.length, 2);
  svc.close();

  // Both windows over: both are contacted again, once each.
  clock.t = T0 + DEFAULT_MIN_INTERVAL_MS;
  const f2 = fakeFetch({ [UP_A.url]: { kind: 'ok', text: csv(range(200001, 600)) }, [UP_S.url]: { kind: 'ok', text: csv(RECENT_ROWS) } });
  const svc2 = makeService(root, clock, { fetchImpl: f2.fetchImpl, upstreams: [UP_A, UP_S] });
  svc2.refreshIfDue();
  svc2.refreshIfDue();
  await svc2.whenIdle();
  assert.deepEqual(f2.calls.map((c) => c.url).sort(), [UP_A.url, UP_S.url].sort());
  assert.equal(svc2.status().cachePartial, false);
  assert.equal(svc2.status().cacheCount, 853 + 26 + 600);
  svc2.close();
});

test('supplement is fetched even while a full set is fresh; nothing new -> cache file untouched', async () => {
  const root = tmpRoot();
  seedFullCache(root, T0 - HOUR);
  const before = fs.readFileSync(path.join(root, '.cache', CACHE_FILE));
  const clock = { t: T0 };
  const { fetchImpl, calls } = fakeFetch({ [UP_S.url]: { kind: 'ok', text: csv([RECENT_ROWS.at(-1)!]) } });
  const svc = makeService(root, clock, { fetchImpl, upstreams: [UP_A, UP_S] });
  svc.refreshIfDue();
  await svc.whenIdle();
  assert.deepEqual(calls.map((c) => c.url), [UP_S.url], 'primary not due (fresh full set), supplement is');
  assert.ok(fs.readFileSync(path.join(root, '.cache', CACHE_FILE)).equals(before));
  assert.equal(readMeta(root).upstreams[UP_S.url].lastSuccessAt, new Date(T0).toISOString());
  svc.close();

  // An unrecognised body ("No GP data found") is a failure, recorded on the supplement only.
  const root2 = tmpRoot();
  const f2 = fakeFetch({ [UP_S.url]: { kind: 'ok', text: 'No GP data found' } });
  const svc2 = makeService(root2, { t: T0 }, { fetchImpl: f2.fetchImpl, upstreams: [UP_S] });
  svc2.refreshIfDue();
  await svc2.whenIdle();
  assert.match(readMeta(root2).upstreams[UP_S.url].lastError, /unrecognised/);
  assert.equal(fs.existsSync(path.join(root2, '.cache', CACHE_FILE)), false);
  svc2.close();
});

test('migration: the old TLE cache seeds the merged set; legacy TLE / GROUP=starlink URLs are never contacted', async () => {
  const root = tmpRoot();
  const dir = path.join(root, '.cache');
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, LEGACY_CACHE_FILE), fixtureText);
  const paused = '2026-11-21T06:00:00.000Z';
  const legacyMeta = {
    version: 1,
    source: 'CelesTrak GP (18 SDS) (incomplete download)',
    url: LEGACY_TLE_URLS[1],
    fetchedAt: '2026-11-20T10:26:35.356Z',
    count: 853,
    partial: true,
    upstreams: Object.fromEntries(
      LEGACY_URLS.map((u) => [
        u,
        { lastAttemptAt: '2026-11-20T10:22:56.098Z', nextAllowedAt: paused, lastError: 'paused', lastStatus: 200, lastSuccessAt: null },
      ]),
    ),
  };
  fs.writeFileSync(path.join(dir, META_FILE), JSON.stringify(legacyMeta));
  const P = { name: PRIMARY_SOURCE };
  const S = SUP;
  const F = FALL;
  const { fetchImpl, calls } = fakeFetch(partMap({}, { [S.url]: { kind: 'ok', text: csv(RECENT_ROWS) } }));
  const clock = { t: T0 };
  const svc = makeService(root, clock, { fetchImpl, upstreams: undefined, fixturePath: NO_FIXTURE });

  // Before any download: the seed is served.
  const st0 = svc.status();
  assert.equal(st0.hasCache, true);
  assert.equal(st0.cachePartial, true);
  assert.equal(st0.cacheCount, 853);
  assert.deepEqual(st0.sources, [{ name: SEED_SOURCE, fetchedAt: '2026-11-20T10:26:35.356Z', records: 853, complete: false }]);
  await withServer(svc, async (port) => {
    const r = JSON.parse((await get(port, '/api/elements')).body.toString()) as ElementsReady;
    assert.equal(r.source, SEED_SOURCE);
    assert.equal(r.partial, true);
    await svc.whenIdle();
  });

  assert.deepEqual(calls.map((c) => c.url).sort(), [...PART_URLS, S.url].sort());
  assert.ok(!calls.some((c) => LEGACY_URLS.includes(c.url) || c.url === F.url));
  const stored = readStored(root);
  assert.equal(stored.records.length, 853 + PART_TOTAL + 26);
  assert.ok(stored.fullCompletedAt);
  assert.equal(fs.readFileSync(path.join(dir, LEGACY_CACHE_FILE), 'utf8'), fixtureText, 'seed file left alone');
  const meta = readMeta(root);
  for (const u of LEGACY_URLS) {
    assert.equal(meta.upstreams[u].nextAllowedAt, paused, 'legacy entries stay as history');
    assert.equal(meta.upstreams[u].lastError, 'paused');
  }
  const st = svc.status();
  assert.equal(st.cachePartial, false);
  assert.deepEqual(st.sources?.map((s) => [s.name, s.records]), [
    [P.name, PART_TOTAL],
    [SEED_SOURCE, 853],
    [S.name, 26],
  ]);
  svc.close();

  // Restart: elements.json wins over the seed; nothing is due.
  clock.t = T0 + HOUR;
  const f2 = fakeFetch({});
  const svc2 = makeService(root, clock, { fetchImpl: f2.fetchImpl, upstreams: undefined, fixturePath: NO_FIXTURE });
  svc2.refreshIfDue();
  await svc2.whenIdle();
  assert.equal(svc2.status().cacheCount, 853 + PART_TOTAL + 26);
  assert.equal(f2.calls.length, 0);
  svc2.close();
});

test('rate-limit invariants on the default OMM URLs: once per URL per window, 403 lock per URL, fallback only when needed', async () => {
  const root = tmpRoot();
  const lockBody = 'GP data has not updated since your last successful download of NAME=STARLINK at 2026-11-20 10:30:00 UTC.';
  const locked: Record<number, Behaviour> = {};
  PARTS.forEach((_, i) => (locked[i] = { kind: 'status', status: 403, body: lockBody }));
  const map = partMap(locked, {
    [SUP.url]: { kind: 'ok', text: csv(RECENT_ROWS) },
    [FALL.url]: { kind: 'ok', text: csv(range(200001, 700)) },
  });
  const { fetchImpl, calls, events } = fakeFetch(map);
  const clock = { t: T0 };
  const svc = makeService(root, clock, { fetchImpl, upstreams: undefined });
  svc.refreshIfDue();
  await svc.whenIdle();
  // Every partition once, in order, one at a time; the supplement on its own.
  assert.deepEqual(calls.map((c) => c.url).filter((u) => PART_URLS.includes(u)), PART_URLS);
  assert.deepEqual(calls.map((c) => c.url).sort(), [...PART_URLS, SUP.url].sort());
  assert.equal(maxConcurrent(events, PART_URLS), 1);
  for (const c of calls) assert.equal(c.headers.get('user-agent'), USER_AGENT);
  let meta = readMeta(root);
  for (const u of PART_URLS) assert.equal(meta.upstreams[u].nextAllowedAt, '2026-11-20T12:31:00.000Z', u);
  assert.equal(meta.upstreams[SUP.url].nextAllowedAt, new Date(T0 + DEFAULT_MIN_INTERVAL_MS).toISOString());
  assert.equal(meta.upstreams[FALL.url], undefined, 'fallback not touched yet');
  assert.equal(svc.status().cycle?.partitionsFailed, PARTS.length);

  // Primary locked everywhere and no full set -> SupGP fallback, once.
  clock.t += 1000;
  for (let i = 0; i < 5; i++) {
    svc.refreshIfDue();
    await svc.whenIdle();
  }
  assert.deepEqual(calls.map((c) => c.url).slice(PARTS.length + 1), [FALL.url]);
  assert.equal(svc.status().cachePartial, false, 'a complete fallback download is a full set');

  // Hammering inside every window: no further upstream contact.
  for (const dt of [10 * 60_000, HOUR, 90 * 60_000]) {
    clock.t = T0 + dt;
    for (let i = 0; i < 3; i++) svc.refreshIfDue();
    await svc.whenIdle();
  }
  assert.equal(calls.length, PARTS.length + 2);
  for (const url of [...PART_URLS, SUP.url, FALL.url]) assert.equal(calls.filter((c) => c.url === url).length, 1, url);
  assert.ok(!calls.some((c) => LEGACY_URLS.includes(c.url)));

  // The 403 locks lift at 12:31, but the fallback completed at T0+1 s: the full set is fresh until T0+1 s+window.
  clock.t = Date.parse('2026-11-20T12:31:00Z');
  svc.refreshIfDue();
  await svc.whenIdle();
  assert.equal(calls.length, PARTS.length + 2);
  // Window over: the partitions (same, still open cycle) + supplement again; no fallback while the primary answers.
  for (const [i] of PARTS.entries()) map[PARTS[i]!.url] = { kind: 'ok', text: partBody(i) };
  clock.t = T0 + 1000 + DEFAULT_MIN_INTERVAL_MS;
  svc.refreshIfDue();
  await svc.whenIdle();
  assert.deepEqual(calls.map((c) => c.url).slice(PARTS.length + 2).sort(), [...PART_URLS, SUP.url].sort());
  meta = readMeta(root);
  assert.equal(meta.upstreams[FALL.url].lastAttemptAt, new Date(T0 + 1000).toISOString());
  assert.ok(meta.cycle.completedAt && meta.cycle.full);
  svc.refreshIfDue();
  await svc.whenIdle();
  assert.equal(calls.length, 2 * PARTS.length + 3);
  svc.close();
});

// ---------------------------------------------------------------------------
// Partitioned primary (full refresh cycles)
// ---------------------------------------------------------------------------

/** Poll until `cond` holds (real time), at most ~3 s. */
async function until(cond: () => boolean, what: string): Promise<void> {
  for (let i = 0; i < 600 && !cond(); i++) await sleep(5);
  assert.ok(cond(), `timed out waiting for ${what}`);
}

test('partitioned primary: 18 partitions sequentially in order with a pause, merged progressively, once per window; dedup; non-STARLINK filtered', async () => {
  const root = tmpRoot();
  const clock = { t: T0 };
  const seen: { url: string; stored: number; done: number | undefined; current: string | null; at: number }[] = [];
  let svc: ElementsService | null = null;
  const { fetchImpl, calls, events } = fakeFetch(partMap(), (url) => {
    let stored = 0;
    try {
      stored = partIds(readStored(root).records).length;
    } catch {
      // no elements.json yet
    }
    const st = svc!.status();
    seen.push({ url, stored, done: st.cycle?.partitionsDone, current: st.cycle?.current ?? null, at: performance.now() });
  });
  svc = makeService(root, clock, { fetchImpl, upstreams: [...PARTS], fixturePath: NO_FIXTURE, partitionPauseMs: 20 });
  svc.refreshIfDue();
  assert.equal(svc.status().fetching, true);
  assert.equal(svc.status().cycle?.running, true);
  for (let i = 0; i < 5; i++) svc.refreshIfDue();
  await svc.whenIdle();

  // Requested once each, in order, never two at a time, with the politeness pause in between.
  assert.deepEqual(calls.map((c) => c.url), PART_URLS);
  assert.equal(maxConcurrent(events, PART_URLS), 1);
  for (let k = 1; k < seen.length; k++) assert.ok(seen[k]!.at - seen[k - 1]!.at >= 15, `pause before partition ${k}`);
  // Each partition was merged and persisted before the next one was requested.
  let expected = 0;
  seen.forEach((s, k) => {
    assert.equal(s.stored, expected, `stored before partition ${k}`);
    assert.equal(s.done, k);
    assert.equal(s.current, PARTS[k]!.partition);
    if (!EMPTY_PARTS.has(PARTS[k]!.partition!)) expected += PER_PART;
  });

  const stored = readStored(root);
  assert.equal(partIds(stored.records).length, PART_TOTAL);
  assert.equal(stored.records.length, PART_TOTAL, 'overlaps deduplicated, nothing else');
  assert.ok(!stored.records.some((r) => r.noradId >= 900000), 'non-STARLINK rows filtered');
  assert.ok(stored.records.every((r) => r.name.startsWith('STARLINK-')));
  // The overlapping row (newer epoch) from the next partition won.
  assert.match(stored.records.find((r) => r.noradId === partRows(0)[0]!.id)!.omm!.EPOCH, /^2026-11-20T07:00:00/);
  assert.equal(stored.fullCompletedAt, new Date(T0).toISOString());

  const st = svc.status();
  assert.equal(st.cachePartial, false);
  assert.equal(st.fetching, false);
  assert.deepEqual(st.sources, [{ name: PRIMARY_SOURCE, fetchedAt: new Date(T0).toISOString(), records: PART_TOTAL, complete: true }]);
  assert.deepEqual(st.cycle, {
    startedAt: new Date(T0).toISOString(),
    completedAt: new Date(T0).toISOString(),
    complete: true,
    partitionsTotal: 18,
    partitionsDone: 18,
    partitionsFailed: 0,
    current: null,
    running: false,
    lastError: null,
  });
  const meta = readMeta(root);
  assert.equal(meta.partial, false);
  const empty = PARTS.find((p) => p.partition === 'STARLINK-8')!;
  assert.deepEqual(meta.cycle.partitions[empty.url], { at: new Date(T0).toISOString(), complete: true, records: 0, error: null });
  for (const u of PART_URLS) assert.equal(meta.upstreams[u].nextAllowedAt, new Date(T0 + DEFAULT_MIN_INTERVAL_MS).toISOString());

  await withServer(svc, async (port) => {
    const r = JSON.parse((await get(port, '/api/elements')).body.toString()) as ElementsReady;
    assert.equal(r.partial, false);
    assert.equal(r.count, PART_TOTAL);
    assert.equal(r.cycle?.partitionsDone, 18);
    assert.equal(r.cycle?.complete, true);
    assert.equal(r.nextRefreshEarliest, new Date(T0 + DEFAULT_MIN_INTERVAL_MS).toISOString());
  });

  // Inside the window: nothing.
  for (const t of [T0 + HOUR, T0 + DEFAULT_MIN_INTERVAL_MS - 1000]) {
    clock.t = t;
    for (let i = 0; i < 3; i++) svc.refreshIfDue();
    await svc.whenIdle();
  }
  assert.equal(calls.length, PARTS.length);
  // Oldest partition success a window old: a new cycle, every partition once more, in order.
  clock.t = T0 + DEFAULT_MIN_INTERVAL_MS;
  svc.refreshIfDue();
  svc.refreshIfDue();
  await svc.whenIdle();
  assert.deepEqual(calls.map((c) => c.url).slice(PARTS.length), PART_URLS);
  assert.equal(readMeta(root).cycle.startedAt, new Date(clock.t).toISOString());
  assert.equal(svc.status().cycle?.complete, true);
  svc.close();
});

test('partitioned primary: a cut-off partition keeps its rows and does not block the others; partial=false only after a complete cycle', async () => {
  const root = tmpRoot();
  const clock = { t: T0 };
  const CUT = 3; // STARLINK-5
  const map = partMap({ [CUT]: { kind: 'cut', text: partBody(CUT), afterBytes: 3000 } });
  const { fetchImpl, calls } = fakeFetch(map);
  const svc = makeService(root, clock, { fetchImpl, upstreams: [...PARTS], fixturePath: NO_FIXTURE });
  svc.refreshIfDue();
  await svc.whenIdle();
  assert.deepEqual(calls.map((c) => c.url), PART_URLS, 'the cut-off partition did not stop the cycle');

  const stored = readStored(root);
  const cutIds = partIds(stored.records).filter((id) => id >= 203000 && id < 203000 + PER_PART);
  assert.ok(cutIds.length > 5 && cutIds.length < PER_PART, `merged ${cutIds.length} rows of the cut-off partition`);
  assert.equal(partIds(stored.records).length, PART_TOTAL - PER_PART + cutIds.length);
  assert.equal(stored.fullCompletedAt, null);
  const cutUrl = PARTS[CUT]!.url;
  let meta = readMeta(root);
  assert.match(meta.upstreams[cutUrl].lastError, /terminated/);
  assert.equal(meta.cycle.partitions[cutUrl].complete, false);
  assert.equal(meta.cycle.partitions[cutUrl].records, cutIds.length);
  assert.equal(meta.cycle.completedAt, null);
  assert.equal(meta.partial, true);
  let st = svc.status();
  assert.equal(st.cachePartial, true);
  assert.equal(st.cycle?.partitionsDone, 17);
  assert.equal(st.cycle?.partitionsFailed, 1);
  assert.equal(st.cycle?.complete, false);
  assert.match(st.cycle?.lastError ?? '', /^STARLINK-5: .*terminated/);
  assert.equal(st.sources?.[0]?.complete, false);

  // The cut-off partition is not retried inside its own window.
  for (const t of [T0 + 60_000, T0 + HOUR, T0 + DEFAULT_MIN_INTERVAL_MS - 1000]) {
    clock.t = t;
    svc.refreshIfDue();
    await svc.whenIdle();
  }
  assert.equal(calls.length, PARTS.length);

  // Window over: a new cycle; this time every partition answers completely.
  map[cutUrl] = { kind: 'ok', text: partBody(CUT) };
  clock.t = T0 + DEFAULT_MIN_INTERVAL_MS;
  svc.refreshIfDue();
  await svc.whenIdle();
  assert.deepEqual(calls.map((c) => c.url).slice(PARTS.length), PART_URLS);
  st = svc.status();
  assert.equal(st.cachePartial, false);
  assert.equal(st.cycle?.complete, true);
  assert.equal(st.cycle?.lastError, null);
  assert.equal(partIds(readStored(root).records).length, PART_TOTAL);
  meta = readMeta(root);
  assert.equal(meta.partial, false);
  assert.equal(meta.upstreams[cutUrl].lastError, null);
  svc.close();
});

test('partitioned primary: locks respected per URL; a locked partition is retried alone after its own window and completes the cycle', async () => {
  const root = tmpRoot();
  const dir = path.join(root, '.cache');
  fs.mkdirSync(dir, { recursive: true });
  // An earlier attempt of STARLINK-2 (e.g. another process) still holds its window until T0+20 min.
  const pre = PARTS[1]!.url;
  fs.writeFileSync(
    path.join(dir, META_FILE),
    JSON.stringify({
      version: 1,
      upstreams: { [pre]: { lastAttemptAt: new Date(T0 + 20 * 60_000 - DEFAULT_MIN_INTERVAL_MS).toISOString(), nextAllowedAt: new Date(T0 + 20 * 60_000).toISOString() } },
    }),
  );
  const LOCKED = 4; // STARLINK-6: CelesTrak's 403 => locked until 09:29 + 2 h 1 min = 11:30
  const lockBody = 'GP data has not updated since your last successful download of NAME=STARLINK-6 at 2026-11-20 09:29:00 UTC.';
  const map = partMap({ [LOCKED]: { kind: 'status', status: 403, body: lockBody } });
  const { fetchImpl, calls, events } = fakeFetch(map);
  const clock = { t: T0 };
  const svc = makeService(root, clock, { fetchImpl, upstreams: [...PARTS], fixturePath: NO_FIXTURE });
  svc.refreshIfDue();
  await svc.whenIdle();
  assert.deepEqual(calls.map((c) => c.url), PART_URLS.filter((u) => u !== pre), 'the locked URL is skipped');
  const lockedUrl = PARTS[LOCKED]!.url;
  assert.equal(readMeta(root).upstreams[lockedUrl].nextAllowedAt, '2026-11-20T11:30:00.000Z');
  let st = svc.status();
  assert.equal(st.cycle?.partitionsDone, 16);
  assert.equal(st.cycle?.partitionsFailed, 1);
  assert.equal(st.cachePartial, true);
  assert.equal(st.nextAllowedFetchAt, new Date(T0 + 20 * 60_000).toISOString());

  // STARLINK-2's window opens: only it is fetched (same cycle).
  clock.t = T0 + 20 * 60_000;
  svc.refreshIfDue();
  await svc.whenIdle();
  assert.deepEqual(calls.map((c) => c.url).slice(17), [pre]);
  assert.equal(svc.status().cachePartial, true, 'still one partition missing');
  clock.t = T0 + 29 * 60_000;
  svc.refreshIfDue();
  await svc.whenIdle();
  assert.equal(calls.length, 18);

  // STARLINK-6's lock lifts: only it is fetched, completing the cycle.
  map[lockedUrl] = { kind: 'ok', text: partBody(LOCKED) };
  clock.t = T0 + 30 * 60_000;
  for (let i = 0; i < 3; i++) svc.refreshIfDue();
  await svc.whenIdle();
  assert.deepEqual(calls.map((c) => c.url).slice(18), [lockedUrl]);
  assert.equal(maxConcurrent(events, PART_URLS), 1);
  st = svc.status();
  assert.equal(st.cachePartial, false);
  assert.equal(st.cycle?.complete, true);
  assert.equal(st.cycle?.startedAt, new Date(T0).toISOString(), 'same cycle');
  assert.equal(readStored(root).fullCompletedAt, new Date(T0 + 30 * 60_000).toISOString());
  clock.t = T0 + 31 * 60_000;
  for (let i = 0; i < 3; i++) svc.refreshIfDue();
  await svc.whenIdle();
  assert.equal(calls.length, 19);
  svc.close();
});

test('partitioned primary: restart mid-cycle resumes without re-hitting completed partitions; an aborted one counts', async () => {
  const root = tmpRoot();
  const clock = { t: T0 };
  const HANG = 6; // STARLINK-8
  const f1 = fakeFetch(partMap({ [HANG]: { kind: 'hang' } }));
  const svc1 = makeService(root, clock, { fetchImpl: f1.fetchImpl, upstreams: [...PARTS], fixturePath: NO_FIXTURE });
  svc1.refreshIfDue();
  await until(() => f1.calls.length === HANG + 1, 'the hanging partition');
  svc1.close({ abortDownload: true }); // process restart: the request is cut
  await svc1.whenIdle();
  assert.deepEqual(f1.calls.map((c) => c.url), PART_URLS.slice(0, HANG + 1));
  const hangUrl = PARTS[HANG]!.url;
  let meta = readMeta(root);
  assert.equal(meta.upstreams[hangUrl].lastAttemptAt, new Date(T0).toISOString(), 'the aborted attempt counts');
  assert.match(meta.cycle.partitions[hangUrl].error, /aborted/);
  for (const u of PART_URLS.slice(0, HANG)) assert.equal(meta.cycle.partitions[u].complete, true);

  // New process 10 min later: resumes the same cycle with the partitions not yet done.
  clock.t = T0 + 10 * 60_000;
  const f2 = fakeFetch(partMap());
  const svc2 = makeService(root, clock, { fetchImpl: f2.fetchImpl, upstreams: [...PARTS], fixturePath: NO_FIXTURE });
  assert.equal(svc2.status().cycle?.partitionsDone, HANG);
  svc2.refreshIfDue();
  await svc2.whenIdle();
  assert.deepEqual(f2.calls.map((c) => c.url), PART_URLS.slice(HANG + 1));
  let st = svc2.status();
  assert.equal(st.cycle?.startedAt, new Date(T0).toISOString());
  assert.equal(st.cycle?.partitionsDone, 17);
  assert.equal(st.cachePartial, true);

  // The aborted partition waits for its window, like everything else.
  clock.t = T0 + DEFAULT_MIN_INTERVAL_MS - 1000;
  svc2.refreshIfDue();
  await svc2.whenIdle();
  assert.equal(f2.calls.length, PARTS.length - HANG - 1);
  // Window of the first successes over: a new cycle. The partitions fetched at
  // T0+10 min are still inside their own windows and follow when those open.
  clock.t = T0 + DEFAULT_MIN_INTERVAL_MS;
  svc2.refreshIfDue();
  await svc2.whenIdle();
  const resumed = PARTS.length - HANG - 1;
  assert.deepEqual(f2.calls.map((c) => c.url).slice(resumed), PART_URLS.slice(0, HANG + 1));
  assert.equal(svc2.status().cycle?.partitionsDone, HANG + 1);
  assert.equal(svc2.status().cachePartial, true);
  clock.t = T0 + 10 * 60_000 + DEFAULT_MIN_INTERVAL_MS;
  svc2.refreshIfDue();
  await svc2.whenIdle();
  assert.deepEqual(f2.calls.map((c) => c.url).slice(resumed), PART_URLS);
  st = svc2.status();
  assert.equal(st.cachePartial, false);
  assert.equal(st.cycle?.complete, true);
  meta = readMeta(root);
  assert.equal(meta.cycle.startedAt, new Date(T0 + DEFAULT_MIN_INTERVAL_MS).toISOString());
  svc2.close();
});

test('partitioned primary: a Vite restart (close without abort) hands the running cycle over; never two cycles at once', async () => {
  const root = tmpRoot();
  const clock = { t: T0 };
  const SLOW = 2;
  const { fetchImpl, calls, events } = fakeFetch(partMap({ [SLOW]: { kind: 'ok', text: partBody(SLOW), chunkSize: 512, delayMs: 10 } }));
  const svc1 = makeService(root, clock, { fetchImpl, upstreams: [...PARTS], fixturePath: NO_FIXTURE });
  svc1.refreshIfDue();
  await until(() => calls.length === SLOW + 1, 'the slow partition');
  svc1.close(); // its in-flight partition finishes; the runner stops after it
  const svc2 = makeService(root, clock, { fetchImpl, upstreams: [...PARTS], fixturePath: NO_FIXTURE });
  for (let i = 0; i < 3; i++) svc2.refreshIfDue();
  assert.equal(svc2.status().cycle?.running, true);
  assert.equal(svc2.status().fetching, true);
  await svc2.whenIdle();
  assert.deepEqual(calls.map((c) => c.url), PART_URLS);
  assert.equal(maxConcurrent(events, PART_URLS), 1);
  const st = svc2.status();
  assert.equal(st.cachePartial, false);
  assert.equal(st.cycle?.complete, true);
  assert.equal(st.cacheCount, PART_TOTAL);
  svc2.close();
});

test('partitioned primary: the runner skips a partition whose window is about to open and waits for it at the end', async () => {
  const root = tmpRoot();
  const dir = path.join(root, '.cache');
  fs.mkdirSync(dir, { recursive: true });
  const start = Date.now();
  const now = () => T0 + (Date.now() - start); // real-time clock
  const three = PARTS.slice(0, 3); // STARLINK-1, -2, -4
  const soon = three[1]!.url; // window opens 500 ms from now
  fs.writeFileSync(
    path.join(dir, META_FILE),
    JSON.stringify({ version: 1, upstreams: { [soon]: { lastAttemptAt: new Date(T0 - HOUR).toISOString(), nextAllowedAt: new Date(T0 + 500).toISOString() } } }),
  );
  const callAt: number[] = [];
  const { fetchImpl, calls, events } = fakeFetch(partMap(), () => callAt.push(Date.now() - start));
  const svc = createElementsService({
    rootDir: root,
    fixturePath: NO_FIXTURE,
    upstreams: three,
    minRecords: 100,
    offline: false,
    partitionPauseMs: 0,
    maxPartitionWaitMs: 5_000,
    fetchImpl,
    now,
    log: () => {},
  });
  svc.refreshIfDue();
  await svc.whenIdle(); // one runner: no further refreshIfDue needed
  assert.deepEqual(calls.map((c) => c.url), [three[0]!.url, three[2]!.url, soon]);
  assert.ok(callAt[2]! >= 500, `requested ${callAt[2]} ms after start, window opened at 500 ms`);
  assert.equal(maxConcurrent(events, PART_URLS), 1);
  assert.equal(svc.status().cycle?.complete, true);
  assert.equal(svc.status().cachePartial, false);
  svc.close();
});

test('partitioned primary: a complete cycle with implausibly few records is no full set; fallback allowed', async () => {
  const root = tmpRoot();
  const clock = { t: T0 };
  const empty: Record<number, Behaviour> = {};
  PARTS.forEach((_, i) => (empty[i] = { kind: 'ok', text: 'No GP data found' }));
  const { fetchImpl, calls } = fakeFetch(partMap(empty, { [FALL.url]: { kind: 'ok', text: csv(range(200001, 700)) } }));
  const svc = makeService(root, clock, { fetchImpl, upstreams: [...PARTS, FALL] });
  svc.refreshIfDue();
  await svc.whenIdle();
  assert.deepEqual(calls.map((c) => c.url), PART_URLS);
  const meta = readMeta(root);
  assert.ok(meta.cycle.completedAt);
  assert.equal(meta.cycle.full, false);
  assert.match(meta.cycle.error, /implausible result: 0 records/);
  let st = svc.status();
  assert.equal(st.cachePartial, true);
  assert.equal(st.cacheCount, 853, 'still the bundled snapshot');
  assert.equal(st.cycle?.complete, false);
  assert.match(st.cycle?.lastError ?? '', /implausible/);
  clock.t += 1000;
  svc.refreshIfDue();
  await svc.whenIdle();
  assert.deepEqual(calls.map((c) => c.url).slice(PARTS.length), [FALL.url]);
  st = svc.status();
  assert.equal(st.cachePartial, false);
  svc.close();
});

// ---------------------------------------------------------------------------
// Robustness
// ---------------------------------------------------------------------------

test('corrupt meta.json (zero-filled after a crash): no upstream hit, every upstream re-locked', async () => {
  const root = tmpRoot();
  const dir = path.join(root, '.cache');
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, META_FILE), Buffer.alloc(64));
  const clock = { t: T0 };
  const { fetchImpl, calls } = fakeFetch({
    [UP_A.url]: { kind: 'ok', text: fixtureText },
    [UP_B.url]: { kind: 'ok', text: fixtureText },
    [UP_S.url]: { kind: 'ok', text: csv(RECENT_ROWS) },
  });
  const svc = makeService(root, clock, { fetchImpl, upstreams: [UP_A, UP_S, UP_B] });
  svc.refreshIfDue();
  await svc.whenIdle();
  svc.refreshIfDue();
  await svc.whenIdle();
  assert.equal(calls.length, 0);
  const meta = readMeta(root);
  const until = new Date(T0 + DEFAULT_MIN_INTERVAL_MS).toISOString();
  assert.equal(meta.upstreams[UP_A.url].nextAllowedAt, until);
  assert.equal(meta.upstreams[UP_B.url].nextAllowedAt, until);
  assert.equal(meta.upstreams[UP_S.url].nextAllowedAt, until);
  svc.close();

  // After the window the primary (and the supplement) are due again.
  clock.t = T0 + DEFAULT_MIN_INTERVAL_MS;
  const svc2 = makeService(root, clock, { fetchImpl, upstreams: [UP_A, UP_S, UP_B] });
  svc2.refreshIfDue();
  await svc2.whenIdle();
  assert.deepEqual(calls.map((c) => c.url).sort(), [UP_A.url, UP_S.url].sort());
  svc2.close();
});

test('unreadable meta.json: no upstream hit, data still served', async () => {
  const root = tmpRoot();
  // A directory in place of meta.json: reads fail with EISDIR (not ENOENT).
  fs.mkdirSync(path.join(root, '.cache', META_FILE), { recursive: true });
  const { fetchImpl, calls } = fakeFetch({ [UP_A.url]: { kind: 'ok', text: fixtureText } });
  const svc = makeService(root, { t: T0 }, { fetchImpl });
  svc.refreshIfDue();
  await svc.whenIdle();
  assert.equal(calls.length, 0);
  const st = svc.status();
  assert.equal(st.cacheCount, 853);
  assert.match(st.lastError ?? '', /Cannot read/);
  svc.close();
});

test('lock written by another process after the refresh decision is honoured', async () => {
  const root = tmpRoot();
  const { fetchImpl, calls } = fakeFetch({ [UP_A.url]: { kind: 'ok', text: fixtureText } });
  const svc = makeService(root, { t: T0 }, { fetchImpl, upstreams: [UP_A] });
  svc.refreshIfDue(); // decides from an empty meta...
  // ...then, before the attempt is persisted, another process records its own attempt.
  const lockedUntil = new Date(T0 + HOUR).toISOString();
  fs.mkdirSync(path.join(root, '.cache'), { recursive: true });
  fs.writeFileSync(
    path.join(root, '.cache', META_FILE),
    JSON.stringify({ version: 1, upstreams: { [UP_A.url]: { lastAttemptAt: new Date(T0 - HOUR).toISOString(), nextAllowedAt: lockedUntil } } }),
  );
  await svc.whenIdle();
  assert.equal(calls.length, 0);
  assert.equal(readMeta(root).upstreams[UP_A.url].nextAllowedAt, lockedUntil);
  svc.close();
});

test('meta.json read-modify-write is exclusive across processes (lockfile): no double contact, no lost lock', async () => {
  const root = tmpRoot();
  const cache = path.join(root, '.cache');
  fs.mkdirSync(cache, { recursive: true });
  const lockFile = path.join(cache, `${META_FILE}.lock`);
  // Another process (dev + preview sharing .cache) is in the middle of its own attempt.
  fs.writeFileSync(lockFile, '99999');
  const { fetchImpl, calls } = fakeFetch({ [UP_A.url]: { kind: 'ok', text: fixtureText } });
  const svc = makeService(root, { t: T0 }, { fetchImpl, upstreams: [UP_A] });
  svc.refreshIfDue(); // decides from an empty meta, then waits for the lockfile
  await sleep(80);
  assert.equal(calls.length, 0);
  // The other process persists its attempt and releases the lock.
  const lockedUntil = new Date(T0 + HOUR).toISOString();
  fs.writeFileSync(
    path.join(cache, META_FILE),
    JSON.stringify({ version: 1, upstreams: { [UP_A.url]: { lastAttemptAt: new Date(T0).toISOString(), nextAllowedAt: lockedUntil } } }),
  );
  fs.rmSync(lockFile);
  await svc.whenIdle();
  assert.equal(calls.length, 0);
  assert.equal(readMeta(root).upstreams[UP_A.url].nextAllowedAt, lockedUntil);
  assert.ok(!fs.existsSync(lockFile));
  svc.close();
});

test('withFileLock: serializes holders, breaks a stale lockfile, times out on a held one', async () => {
  const dir = tmpRoot();
  const lock = path.join(dir, 'x.lock');
  // Concurrent holders never overlap.
  let active = 0;
  let maxActive = 0;
  const order: number[] = [];
  await Promise.all(
    [0, 1, 2, 3].map((i) =>
      withFileLock(lock, async () => {
        active++;
        maxActive = Math.max(maxActive, active);
        assert.equal(fs.readFileSync(lock, 'utf8'), String(process.pid));
        await sleep(15);
        order.push(i);
        active--;
      }),
    ),
  );
  assert.equal(maxActive, 1);
  assert.equal(order.length, 4);
  assert.ok(!fs.existsSync(lock));
  // A lockfile left behind by a crashed process is broken once stale.
  fs.writeFileSync(lock, '99999');
  const old = new Date(Date.now() - FILE_LOCK_STALE_MS - 5_000);
  fs.utimesSync(lock, old, old);
  assert.equal(await withFileLock(lock, async () => 'ran'), 'ran');
  assert.ok(!fs.existsSync(lock));
  // A fresh lockfile held by someone else: reject without running fn; the lockfile is left alone.
  fs.writeFileSync(lock, '99999');
  let ran = false;
  await assert.rejects(
    withFileLock(lock, async () => {
      ran = true;
    }, { timeoutMs: 60 }),
    /held by another process/,
  );
  assert.equal(ran, false);
  assert.equal(fs.readFileSync(lock, 'utf8'), '99999');
  // fn rejecting still releases the lock.
  fs.rmSync(lock);
  await assert.rejects(withFileLock(lock, async () => { throw new Error('boom'); }), /boom/);
  assert.ok(!fs.existsSync(lock));
});

test('timestamps written under a clock that ran ahead are clamped, not trusted for months', async () => {
  const root = tmpRoot();
  const future = T0 + 90 * 24 * HOUR;
  seedFullCache(root, future);
  const meta = readMeta(root);
  meta.upstreams = {
    [UP_A.url]: { lastAttemptAt: new Date(future).toISOString(), nextAllowedAt: new Date(future + DEFAULT_MIN_INTERVAL_MS).toISOString() },
  };
  fs.writeFileSync(path.join(root, '.cache', META_FILE), JSON.stringify(meta));
  const clock = { t: T0 };
  const { fetchImpl, calls } = fakeFetch({ [UP_A.url]: { kind: 'ok', text: fixtureText } });
  const svc = makeService(root, clock, { fetchImpl });
  svc.refreshIfDue();
  await svc.whenIdle();
  assert.equal(calls.length, 0, 'a full window from now is still respected');
  assert.equal(readMeta(root).upstreams[UP_A.url].nextAllowedAt, new Date(T0 + DEFAULT_MIN_INTERVAL_MS).toISOString());
  clock.t = T0 + DEFAULT_MIN_INTERVAL_MS - 1000;
  svc.refreshIfDue();
  await svc.whenIdle();
  assert.equal(calls.length, 0);
  clock.t = T0 + DEFAULT_MIN_INTERVAL_MS;
  svc.refreshIfDue();
  await svc.whenIdle();
  assert.deepEqual(calls.map((c) => c.url), [UP_A.url]);
  svc.close();
});
