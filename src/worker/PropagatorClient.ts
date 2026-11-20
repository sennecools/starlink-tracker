/**
 * Main-thread wrapper around the SGP4 propagation worker.
 *
 *   const client = new PropagatorClient();
 *   const { catalog } = await client.init(records);
 *   let kf = await client.keyframe(Date.now());
 *   // ...later, hand the previous buffer back so the worker reuses it:
 *   const next = await client.keyframe(Date.now() + 1000, kf.positions);
 *   // kf.positions is now DETACHED (length 0) - copy what you need first.
 *   const links = await client.links(Date.now()); // MODELLED laser links
 *
 * Requests are correlated by requestId, so several keyframes may be in flight.
 * Only one init may be in flight; a second init supersedes (rejects) the first.
 */
import type { ElementRecord, FromWorker, Keyframe, LaserLinks, SatCatalog, ToWorker } from '../shared/contracts.ts';

export interface PropagatorReady {
  catalog: SatCatalog;
  rejected: number;
  parseMs: number;
}

export type KeyframeResult = Keyframe & { computeMs: number };

interface Pending<T> {
  resolve: (value: T) => void;
  reject: (reason: Error) => void;
}

export class PropagatorClient {
  private worker: Worker | null;
  private nextRequestId = 1;
  private readonly keyframeRequests = new Map<number, Pending<KeyframeResult>>();
  private readonly linkRequests = new Map<number, Pending<LaserLinks>>();
  private initRequest: Pending<PropagatorReady> | null = null;

  constructor() {
    this.worker = new Worker(new URL('./propagator.worker.ts', import.meta.url), {
      type: 'module',
      name: 'sgp4-propagator',
    });
    this.worker.onmessage = (e: MessageEvent<FromWorker>) => this.onMessage(e.data);
    this.worker.onerror = (e: ErrorEvent) => {
      e.preventDefault();
      this.rejectAll(new Error(`propagator worker error: ${e.message || 'failed to load or crashed'}`));
    };
    this.worker.onmessageerror = () => {
      this.rejectAll(new Error('propagator worker: message could not be deserialised'));
    };
  }

  /** Number of requests (init + keyframes + links) awaiting a worker reply. */
  get pending(): number {
    return this.keyframeRequests.size + this.linkRequests.size + (this.initRequest ? 1 : 0);
  }

  /**
   * Send the element set to the worker; resolves with the static catalog.
   * May be called again later (e.g. refreshed elements) to replace the set;
   * keyframes requested afterwards use the new set.
   */
  init(records: ElementRecord[]): Promise<PropagatorReady> {
    const worker = this.worker;
    if (!worker) return Promise.reject(new Error('PropagatorClient disposed'));
    if (this.initRequest) {
      this.initRequest.reject(new Error('superseded by a newer init()'));
      this.initRequest = null;
    }
    return new Promise<PropagatorReady>((resolve, reject) => {
      this.initRequest = { resolve, reject };
      const msg: ToWorker = { type: 'init', records };
      worker.postMessage(msg);
    });
  }

  /**
   * Compute positions of every satellite at `timeMs` (ECEF metres, NaN on failure).
   * `recycle`: a positions array previously received from this client; its
   * buffer is transferred to the worker (and detached here) for reuse.
   */
  keyframe(timeMs: number, recycle?: Float64Array): Promise<KeyframeResult> {
    const worker = this.worker;
    if (!worker) return Promise.reject(new Error('PropagatorClient disposed'));
    const requestId = this.nextRequestId++;
    return new Promise<KeyframeResult>((resolve, reject) => {
      this.keyframeRequests.set(requestId, { resolve, reject });
      // Only transfer a whole, still-attached buffer; a detached one is useless.
      const canRecycle =
        recycle !== undefined &&
        recycle.buffer.byteLength > 0 &&
        recycle.byteOffset === 0 &&
        recycle.byteLength === recycle.buffer.byteLength &&
        !(typeof SharedArrayBuffer !== 'undefined' && recycle.buffer instanceof SharedArrayBuffer);
      if (canRecycle) {
        const msg: ToWorker = { type: 'keyframe', requestId, timeMs, recycle };
        worker.postMessage(msg, [recycle.buffer as ArrayBuffer]);
      } else {
        const msg: ToWorker = { type: 'keyframe', requestId, timeMs };
        worker.postMessage(msg);
      }
    });
  }

  /**
   * Compute the MODELLED laser-link topology at `timeMs` (SpaceX does not
   * publish the real one). Pairs index the catalog of the latest init();
   * `pairs` holds a < b per link, `kinds` 0 = in-plane, 1 = cross-plane.
   * Includes a full SGP4 propagation (positions + orbit normals): ~16 ms of
   * worker time in total for ~8k satellites with optimised (JIT) code, ~10x
   * that without the optimising JIT or in a throttled background renderer
   * (the result's propagateMs / topologyMs give the split). It runs in the
   * same serial worker as the keyframes: request it every few seconds, not
   * per frame.
   */
  links(timeMs: number): Promise<LaserLinks> {
    const worker = this.worker;
    if (!worker) return Promise.reject(new Error('PropagatorClient disposed'));
    const requestId = this.nextRequestId++;
    return new Promise<LaserLinks>((resolve, reject) => {
      this.linkRequests.set(requestId, { resolve, reject });
      const msg: ToWorker = { type: 'links', requestId, timeMs };
      worker.postMessage(msg);
    });
  }

  /** Terminate the worker and reject everything still pending. Idempotent. */
  dispose(): void {
    if (!this.worker) return;
    this.worker.onmessage = null;
    this.worker.onerror = null;
    this.worker.onmessageerror = null;
    this.worker.terminate();
    this.worker = null;
    this.rejectAll(new Error('PropagatorClient disposed'));
  }

  private onMessage(msg: FromWorker): void {
    switch (msg.type) {
      case 'ready': {
        const req = this.initRequest;
        this.initRequest = null;
        req?.resolve({ catalog: msg.catalog, rejected: msg.rejected, parseMs: msg.parseMs });
        break;
      }
      case 'keyframe': {
        const req = this.keyframeRequests.get(msg.requestId);
        if (!req) return;
        this.keyframeRequests.delete(msg.requestId);
        req.resolve({ timeMs: msg.timeMs, positions: msg.positions, computeMs: msg.computeMs });
        break;
      }
      case 'links': {
        const req = this.linkRequests.get(msg.requestId);
        if (!req) return;
        this.linkRequests.delete(msg.requestId);
        req.resolve({
          timeMs: msg.timeMs,
          pairs: msg.pairs,
          kinds: msg.kinds,
          computeMs: msg.computeMs,
          propagateMs: msg.propagateMs,
          topologyMs: msg.topologyMs,
        });
        break;
      }
      case 'error': {
        const err = new Error(`propagator: ${msg.message}`);
        if (msg.requestId !== undefined) {
          const req = this.keyframeRequests.get(msg.requestId) ?? this.linkRequests.get(msg.requestId);
          if (req) {
            this.keyframeRequests.delete(msg.requestId);
            this.linkRequests.delete(msg.requestId);
            req.reject(err);
          }
        } else if (this.initRequest) {
          const req = this.initRequest;
          this.initRequest = null;
          req.reject(err);
        } else {
          console.error(err);
        }
        break;
      }
    }
  }

  private rejectAll(err: Error): void {
    const init = this.initRequest;
    this.initRequest = null;
    init?.reject(err);
    const reqs = [...this.keyframeRequests.values()];
    this.keyframeRequests.clear();
    for (const r of reqs) r.reject(err);
    const links = [...this.linkRequests.values()];
    this.linkRequests.clear();
    for (const r of links) r.reject(err);
  }
}
