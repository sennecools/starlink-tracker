/**
 * SGP4 propagation Web Worker (module worker).
 * Implements the ToWorker / FromWorker protocol from src/shared/contracts.ts.
 *
 *   init     -> parse + initialise all satrecs, reply 'ready' with the catalog
 *               (typed-array buffers transferred, zero-copy).
 *   keyframe -> propagate all satellites to timeMs into the recycled buffer
 *               (if it is the right size) or a fresh one, reply 'keyframe'
 *               with the positions buffer transferred.
 *   links    -> propagate to timeMs (positions + inertial orbit normals into
 *               reused worker-private buffers), compute the MODELLED laser
 *               topology (laserTopology.ts), reply 'links' with pairs/kinds
 *               transferred.
 *
 * Everything is wrapped in try/catch so the worker never dies on bad input;
 * failures are reported as { type: 'error' } (with requestId for keyframes).
 */
import type { SatRec } from 'satellite.js';
import type { FromWorker, SatCatalog, ToWorker } from '../shared/contracts.ts';
import { type TopologyCatalog, computeLaserLinks } from './laserTopology.ts';
import { buildCatalog, propagateInto } from './propagate.ts';

// The project compiles against the DOM lib, not 'webworker'; type the worker
// global locally instead of pulling worker typings into every file.
const ctx = self as unknown as {
  postMessage(msg: FromWorker, transfer?: Transferable[]): void;
  onmessage: ((e: MessageEvent<ToWorker>) => void) | null;
};

let satrecs: SatRec[] | null = null;
/**
 * Worker-private copy of the catalog fields the link model needs (the
 * catalog's own buffers are transferred to the main thread on init).
 */
let topoCatalog: TopologyCatalog | null = null;
/** Reused (never transferred) buffers for link computation. */
let linkPositions = new Float64Array(0);
let linkNormals = new Float64Array(0);

function post(msg: FromWorker, transfer?: Transferable[]): void {
  ctx.postMessage(msg, transfer ?? []);
}

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

function catalogTransferables(c: SatCatalog): Transferable[] {
  return [
    c.noradIds.buffer,
    c.epochMs.buffer,
    c.inclinationDeg.buffer,
    c.meanAltitudeKm.buffer,
    c.periodMin.buffer,
    c.shell.buffer,
    c.bus.buffer,
    c.lasers.buffer,
  ] as Transferable[];
}

function handleInit(msg: Extract<ToWorker, { type: 'init' }>): void {
  const t0 = performance.now();
  const built = buildCatalog(msg.records);
  const parseMs = performance.now() - t0;
  satrecs = built.satrecs;
  const c = built.catalog;
  topoCatalog = {
    count: c.count,
    shell: c.shell.slice(),
    lasers: c.lasers.slice(),
    inclinationDeg: c.inclinationDeg.slice(),
    meanAltitudeKm: c.meanAltitudeKm.slice(),
  };
  post(
    { type: 'ready', catalog: built.catalog, rejected: built.rejected, parseMs },
    catalogTransferables(built.catalog),
  );
}

function handleKeyframe(msg: Extract<ToWorker, { type: 'keyframe' }>): void {
  if (satrecs === null) {
    post({ type: 'error', requestId: msg.requestId, message: 'keyframe requested before init' });
    return;
  }
  const t0 = performance.now();
  const needed = satrecs.length * 3;
  const recycle = msg.recycle;
  // A recycled buffer is only reused if it is exactly the right size (catalog
  // may have changed via re-init) and not a view into some larger buffer.
  const positions =
    recycle !== undefined &&
    recycle.length === needed &&
    recycle.byteOffset === 0 &&
    recycle.buffer.byteLength === needed * 8
      ? recycle
      : new Float64Array(needed);
  propagateInto(satrecs, msg.timeMs, positions);
  const computeMs = performance.now() - t0;
  post(
    { type: 'keyframe', requestId: msg.requestId, timeMs: msg.timeMs, positions, computeMs },
    [positions.buffer as ArrayBuffer],
  );
}

function handleLinks(msg: Extract<ToWorker, { type: 'links' }>): void {
  if (satrecs === null || topoCatalog === null) {
    post({ type: 'error', requestId: msg.requestId, message: 'links requested before init' });
    return;
  }
  const t0 = performance.now();
  const needed = satrecs.length * 3;
  if (linkPositions.length !== needed) {
    linkPositions = new Float64Array(needed);
    linkNormals = new Float64Array(needed);
  }
  propagateInto(satrecs, msg.timeMs, linkPositions, linkNormals);
  const t1 = performance.now();
  const { pairs, kinds } = computeLaserLinks({
    positions: linkPositions,
    normals: linkNormals,
    catalog: topoCatalog,
    timeMs: msg.timeMs,
  });
  const t2 = performance.now();
  post(
    {
      type: 'links',
      requestId: msg.requestId,
      timeMs: msg.timeMs,
      pairs,
      kinds,
      computeMs: t2 - t0,
      propagateMs: t1 - t0,
      topologyMs: t2 - t1,
    },
    [pairs.buffer as ArrayBuffer, kinds.buffer as ArrayBuffer],
  );
}

ctx.onmessage = (e: MessageEvent<ToWorker>): void => {
  const msg = e.data;
  try {
    switch (msg.type) {
      case 'init':
        handleInit(msg);
        break;
      case 'keyframe':
        handleKeyframe(msg);
        break;
      case 'links':
        handleLinks(msg);
        break;
      default:
        post({ type: 'error', message: `unknown message type: ${(msg as { type?: unknown }).type}` });
    }
  } catch (err) {
    post({
      type: 'error',
      requestId: msg.type === 'keyframe' || msg.type === 'links' ? msg.requestId : undefined,
      message: errorMessage(err),
    });
  }
};
