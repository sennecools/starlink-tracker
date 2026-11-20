#!/usr/bin/env node
/**
 * snapshot-elements — freezes the orbital-element API into static files for a
 * static host (Vercel), where the Vite dev/preview middleware does not run.
 *
 *   npm run build && node scripts/snapshot-elements.ts [--out=dist]
 *
 * Runs the same service as server/celestrakPlugin.ts once: starts the refresh
 * that is due (all CelesTrak partitions on a fresh machine, nothing when the
 * local .cache is still inside its rate-limit window), waits until it settles,
 * then writes what the server would answer:
 *
 *   <out>/data/elements.json          body of GET /api/elements
 *   <out>/data/elements-status.json   body of GET /api/elements/status
 *
 * vercel.json rewrites the two API paths to these files, so the browser code
 * is unchanged. The data is as fresh as the last deploy. If CelesTrak cannot
 * be reached, the service falls back to the bundled partial fixture, so the
 * build never fails on upstream trouble.
 */
import * as fsp from 'node:fs/promises';
import * as http from 'node:http';
import * as path from 'node:path';
import type { AddressInfo } from 'node:net';
import { createElementsService } from '../server/elementsService.ts';

/** Hard cap on the upstream refresh; whatever has merged by then is used. */
const SETTLE_TIMEOUT_MS = Number(process.env.SNAPSHOT_TIMEOUT_MS ?? 10 * 60_000);

const rootDir = process.cwd();
const outArg = process.argv.find((a) => a.startsWith('--out='));
const outDir = path.resolve(rootDir, outArg ? outArg.slice('--out='.length) : 'dist');
const log = (msg: string): void => console.log(`[snapshot] ${msg}`);

const service = createElementsService({
  rootDir,
  log,
  // A build must not sit out a 2 h CelesTrak window: skip partitions that are
  // not allowed right now instead of waiting for them.
  maxPartitionWaitMs: 0,
  requestTimeoutMs: 3 * 60_000,
});

service.refreshIfDue();
const settled = await Promise.race([
  service.whenIdle().then(() => true),
  new Promise<false>((resolve) => setTimeout(resolve, SETTLE_TIMEOUT_MS, false).unref()),
]);
if (!settled) log(`refresh still running after ${SETTLE_TIMEOUT_MS / 1000} s; snapshotting what has merged so far`);

// Ask the real request handler (not internals), exactly as the browser would.
const server = http.createServer((req, res) =>
  service.handle(req, res, (err) => {
    res.statusCode = err ? 500 : 404;
    res.end();
  }),
);
await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
const base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;

try {
  const elements = await fetch(`${base}/api/elements`);
  if (elements.status !== 200) throw new Error(`/api/elements answered HTTP ${elements.status}: ${await elements.text()}`);
  const elementsBody = await elements.text();
  const statusBody = await (await fetch(`${base}/api/elements/status`)).text();

  const { count, partial, source } = JSON.parse(elementsBody) as { count: number; partial: boolean; source: string };
  const dataDir = path.join(outDir, 'data');
  await fsp.mkdir(dataDir, { recursive: true });
  await fsp.writeFile(path.join(dataDir, 'elements.json'), elementsBody);
  await fsp.writeFile(path.join(dataDir, 'elements-status.json'), statusBody);
  log(`wrote ${count} records (${partial ? 'partial' : 'full'}, ${source}) to ${path.relative(rootDir, dataDir)}`);
} finally {
  server.close();
  service.close({ abortDownload: true });
}
// An aborted download or a timer must not keep the build alive.
process.exit(0);
