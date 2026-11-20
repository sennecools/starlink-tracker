/**
 * Browser-side loader for the orbital-element API served by
 * server/celestrakPlugin.ts.
 */
import type { ElementsReady, ElementsStatus, ElementsWarming } from '../shared/contracts.ts';

export interface LoadElementsOptions {
  signal?: AbortSignal;
  /** Called with live download progress while the server answers 503 (warming). */
  onProgress?: (p: ElementsWarming) => void;
  /** Initial poll interval while warming (ms). Default 2000, backs off to 5000. */
  pollMs?: number;
  /** API base, default '' (same origin). */
  baseUrl?: string;
}

const MAX_POLL_MS = 5000;
const BACKOFF = 1.5;

/**
 * Resolve with the element set. While the server is still downloading
 * (HTTP 503 ElementsWarming) it polls, reporting progress via `onProgress`.
 * Rejects with a descriptive Error on any other failure, or with the
 * signal's AbortError when aborted.
 */
export async function loadElements(opts: LoadElementsOptions = {}): Promise<ElementsReady> {
  const { signal, onProgress } = opts;
  const base = opts.baseUrl ?? '';
  let delay = Math.max(250, opts.pollMs ?? 2000);
  const maxDelay = Math.max(delay, MAX_POLL_MS);

  for (;;) {
    signal?.throwIfAborted();
    const res = await request(`${base}/api/elements`, signal);

    if (res.status === 503) {
      const warming = await readJson<ElementsWarming>(res).catch(() => null);
      if (warming && warming.status === 'warming') onProgress?.(warming);
      await sleep(delay, signal);
      delay = Math.min(maxDelay, delay * BACKOFF);
      continue;
    }
    if (!res.ok) throw new Error(`Orbital elements request failed: HTTP ${res.status} ${res.statusText}${await snippet(res)}`);

    let body: ElementsReady;
    try {
      body = await readJson<ElementsReady>(res);
    } catch (err) {
      throw new Error(`Orbital elements response is not valid JSON: ${(err as Error).message}`, { cause: err });
    }
    if (body?.status !== 'ready' || !Array.isArray(body.records)) {
      throw new Error('Orbital elements response has an unexpected shape (expected status "ready" with records).');
    }
    return body;
  }
}

/** One-shot read of the server's download / cache status. */
export async function fetchElementsStatus(opts: { signal?: AbortSignal; baseUrl?: string } = {}): Promise<ElementsStatus> {
  const res = await request(`${opts.baseUrl ?? ''}/api/elements/status`, opts.signal);
  if (!res.ok) throw new Error(`Elements status request failed: HTTP ${res.status} ${res.statusText}${await snippet(res)}`);
  return readJson<ElementsStatus>(res);
}

async function request(url: string, signal: AbortSignal | undefined): Promise<Response> {
  try {
    return await fetch(url, { signal, cache: 'no-store', headers: { Accept: 'application/json' } });
  } catch (err) {
    if (signal?.aborted) throw signal.reason ?? err;
    throw new Error(`Cannot reach the orbital elements API at ${url}: ${(err as Error).message}`, { cause: err });
  }
}

function readJson<T>(res: Response): Promise<T> {
  return res.json() as Promise<T>;
}

async function snippet(res: Response): Promise<string> {
  try {
    const t = (await res.text()).replace(/\s+/g, ' ').trim();
    return t ? ` — ${t.slice(0, 200)}` : '';
  } catch {
    return '';
  }
}

function sleep(ms: number, signal: AbortSignal | undefined): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(signal.reason);
    const onAbort = () => {
      clearTimeout(timer);
      reject(signal!.reason);
    };
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort);
      resolve();
    }, ms);
    signal?.addEventListener('abort', onAbort, { once: true });
  });
}
