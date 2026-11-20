/**
 * createSky — NASA SVS Deep Star Maps 2020 Milky Way cube + Hipparcos star
 * field, replacing Cesium's default Tycho sky box. Sun and moon are untouched.
 *
 * Boot: fetch <base>/manifest.json, then (in parallel) the six 2k faces and
 * stars.bin. Faces are decoded off the main thread (createImageBitmap via
 * Cesium's Resource.fetchImage, flipY exactly like Cesium's own loadCubeMap)
 * and uploaded one face per frame into a fresh cube map that replaces the old
 * sky only when complete (SkyCube). `ready` resolves on the frame the 2k cube
 * is first visible. Until then the previous sky box keeps drawing.
 *
 * Upgrade: a few seconds after boot, when the browser is idle, evaluate()
 * switches to the 4k cube if the device can take it (skyAssets.deviceAllows4k
 * + `isDegraded()`); if the app later reports degradation while at 4k it goes
 * back to 2k and stays there for the session (no upload ping-pong).
 *
 * Frame/epoch of the assets: see skyMath.ts and public/sky/manifest.json.
 */
import { Credit, Resource } from 'cesium';
import type { Viewer } from 'cesium';
import type { SkyApi, SkyQuality } from '../../shared/phase2.ts';
import type { FaceSource } from './cesiumInternals.ts';
import { decodeStarsBin, deviceAllows4k } from './skyAssets.ts';
import type { CubeEntry, NavigatorHints, SkyManifest } from './skyAssets.ts';
import { CUBE_FACES } from './skyMath.ts';
import { SkyEnvironment } from './SkyEnvironment.ts';
import type { SkyBoxLike } from './SkyEnvironment.ts';
import { StarField } from './StarField.ts';
import type { StarFieldStyle } from './StarField.ts';

export const SKY_CREDIT_HTML =
  'Milky Way: NASA/Goddard Space Flight Center Scientific Visualization Studio; Gaia DR2: ESA/Gaia/DPAC. ' +
  'Stars: ESA Hipparcos catalogue via d3-celestial.';

export interface CreateSkyOptions {
  /** Where public/sky is served (default `${import.meta.env.BASE_URL}sky/`). Must end with '/'. */
  baseUrl?: string;
  /** Return true while the renderer is degraded (e.g. `() => quality.level > 1`); blocks / reverts 4k. */
  isDegraded?: () => boolean;
  /** Delay after the 2k sky is showing before the 4k upgrade is considered (default 2500 ms, then idle). */
  upgradeDelayMs?: number;
  /** Never load the 4k cube (default false). */
  disable4k?: boolean;
  /** Star rendering overrides. */
  starStyle?: Partial<StarFieldStyle>;
}

/** Developer helpers for verifying the sky in the browser console (window.__sky in dev builds). */
export interface SkyDebug {
  /** Tint each cube face (+X red, -X cyan, +Y green, -Y magenta, +Z blue, -Z yellow). */
  faceColors(on: boolean): void;
  /** Rings at the galactic centre (magenta; must sit on the brightest Milky Way region) and Polaris (cyan). */
  markers(on: boolean): void;
  /** Hide/show the catalogue stars. */
  stars(on: boolean): void;
  /** Live-tune the star rendering. */
  starStyle(style: Partial<StarFieldStyle>): void;
  /** Runtime multiplier on the Milky Way cube (default 1; the baked tone map is the real control). */
  cubeBrightness(v: number): void;
  /** Force a quality (bypasses the device heuristics). */
  load(quality: SkyQuality): Promise<void>;
  /** Current state and the parsed manifest. */
  info(): { quality: SkyQuality | null; loading: SkyQuality | null; cubeSize: number; stars: number; manifest: SkyManifest | null };
}

export interface SkyHandle extends SkyApi {
  readonly debug: SkyDebug;
}

function defaultBaseUrl(): string {
  const base = import.meta.env?.BASE_URL ?? '/';
  return (base.endsWith('/') ? base : base + '/') + 'sky/';
}

function closeImages(images: FaceSource[]): void {
  for (const img of images) if (typeof ImageBitmap !== 'undefined' && img instanceof ImageBitmap) img.close();
}

function loadFaceImages(base: string, entry: CubeEntry): Promise<FaceSource[]> {
  const v = entry.hash ? '?v=' + entry.hash : '';
  const requests = CUBE_FACES.map((f) => {
    const p = new Resource({ url: base + entry.path + f + '.' + entry.ext + v }).fetchImage({
      preferImageBitmap: true,
      flipY: true,
      skipColorSpaceConversion: true,
    });
    if (!p) throw new Error('sky face request throttled');
    return p as Promise<FaceSource>;
  });
  return Promise.all(requests).then((images) => {
    for (const img of images) {
      if (img.width !== entry.size || img.height !== entry.size) {
        closeImages(images);
        throw new Error(`sky face is ${img.width}x${img.height}, expected ${entry.size}`);
      }
    }
    return images;
  });
}

async function loadStars(base: string, manifest: SkyManifest): Promise<Float32Array | null> {
  const s = manifest.stars;
  if (!s) return null;
  const res = await fetch(base + s.file + (s.hash ? '?v=' + s.hash : ''));
  if (!res.ok) throw new Error(`stars.bin: HTTP ${res.status}`);
  return decodeStarsBin(await res.arrayBuffer(), s);
}

export function createSky(viewer: Viewer, opts: CreateSkyOptions = {}): SkyHandle {
  const scene = viewer.scene;
  const base = opts.baseUrl ?? defaultBaseUrl();
  // The previous sky box keeps drawing until our first cube is uploaded, then is destroyed.
  const env = new SkyEnvironment(scene.skyBox as unknown as SkyBoxLike | undefined);
  (scene as unknown as { skyBox: unknown }).skyBox = env;

  const credit = new Credit(SKY_CREDIT_HTML, false);
  viewer.creditDisplay.addStaticCredit(credit);

  let manifest: SkyManifest | null = null;
  let quality: SkyQuality | null = null;
  let loading: SkyQuality | null = null;
  let destroyed = false;
  /** Set after a degradation-driven downgrade: no further upgrades this session. */
  let stuckAt2k = !!opts.disable4k;
  let upgradeTimer: ReturnType<typeof setTimeout> | undefined;
  let idleHandle: number | undefined;

  function loadQuality(q: SkyQuality): Promise<void> {
    const entry = manifest?.cube[q];
    if (!entry) return Promise.reject(new Error(`no ${q} cube in the sky manifest`));
    if (entry.size > scene.maximumCubeMapSize) return Promise.reject(new Error(`${q} cube exceeds MAX_CUBE_MAP_TEXTURE_SIZE`));
    loading = q;
    return loadFaceImages(base, entry)
      .then((images) => {
        // stuckAt2k during a 4k load = the app degraded meanwhile (evaluate): drop it.
        if (destroyed || (q === '4k' && stuckAt2k)) {
          closeImages(images);
          throw new Error(destroyed ? 'sky destroyed' : 'sky 4k upgrade cancelled');
        }
        return env.cube.setFaces(images, entry.size);
      })
      .then(() => {
        quality = q;
      })
      .finally(() => {
        if (loading === q) loading = null;
      });
  }

  function canUse4k(): boolean {
    if (stuckAt2k || !manifest?.cube['4k']) return false;
    if (opts.isDegraded?.()) return false;
    const coarse = typeof matchMedia === 'function' && matchMedia('(pointer: coarse)').matches;
    return deviceAllows4k(scene.maximumCubeMapSize, navigator as unknown as NavigatorHints, coarse);
  }

  function evaluate(): void {
    if (destroyed || !manifest || quality === null) return;
    if (loading === '4k' && opts.isDegraded?.()) {
      // Degraded while the 4k set is still loading: abandon it now (the 2k cube
      // stays visible) instead of uploading it only to swap back to 2k.
      stuckAt2k = true; // also makes the pending decode step bail out
      env.cube.cancel('downgraded'); // only the 4k upload can be pending here
      return;
    }
    if (loading !== null) return;
    if (quality === '4k' && opts.isDegraded?.()) {
      stuckAt2k = true;
      loadQuality('2k').catch((e: unknown) => console.warn('[sky] 2k downgrade failed', e));
      return;
    }
    if (quality === '2k' && canUse4k()) {
      loadQuality('4k')
        .then(() => evaluate()) // the app may have degraded during the upload
        .catch((e: unknown) => {
          if (!destroyed && !stuckAt2k) console.warn('[sky] 4k upgrade failed; staying at 2k', e);
        });
    }
  }

  function scheduleUpgrade(): void {
    if (destroyed || stuckAt2k) return;
    upgradeTimer = setTimeout(() => {
      upgradeTimer = undefined;
      if (destroyed) return;
      if (typeof requestIdleCallback === 'function') {
        idleHandle = requestIdleCallback(
          () => {
            idleHandle = undefined;
            evaluate();
          },
          { timeout: 5000 },
        );
      } else {
        evaluate();
      }
    }, opts.upgradeDelayMs ?? 2500);
  }

  const ready: Promise<void> = (async () => {
    const res = await fetch(base + 'manifest.json', { cache: 'no-cache' });
    if (!res.ok) throw new Error(`sky manifest: HTTP ${res.status}`);
    const m = (await res.json()) as SkyManifest;
    manifest = m;
    if (destroyed) return;
    const starsPromise = loadStars(base, m).catch((e: unknown) => {
      console.warn('[sky] star catalogue unavailable', e);
      return null;
    });
    await loadQuality('2k');
    const data = await starsPromise;
    if (data && !destroyed) {
      env.stars = new StarField({ data, epoch: m.stars?.epoch ?? m.cube['2k']?.epoch ?? 2000, style: opts.starStyle });
    }
    scheduleUpgrade();
  })();
  // Callers may ignore `ready`; never surface an unhandled rejection.
  ready.catch((e: unknown) => {
    if (!destroyed) console.warn('[sky] NASA sky unavailable, keeping the previous sky box', e);
  });

  const debug: SkyDebug = {
    faceColors(on) {
      env.cube.faceTint = on ? 0.6 : 0;
    },
    markers(on) {
      if (env.stars) env.stars.showMarkers = on;
      else console.warn('[sky] star field not loaded yet');
    },
    stars(on) {
      if (env.stars) env.stars.show = on;
    },
    starStyle(style) {
      env.stars?.setStyle(style);
    },
    cubeBrightness(v) {
      env.cube.brightness = v;
    },
    load(q) {
      if (q === '4k') stuckAt2k = false;
      return loadQuality(q);
    },
    info() {
      return { quality, loading, cubeSize: env.cube.size, stars: env.stars?.count ?? 0, manifest };
    },
  };

  const api: SkyHandle = {
    get quality() {
      return quality;
    },
    ready,
    evaluate,
    debug,
    destroy() {
      if (destroyed) return;
      destroyed = true;
      if (upgradeTimer !== undefined) clearTimeout(upgradeTimer);
      if (idleHandle !== undefined && typeof cancelIdleCallback === 'function') cancelIdleCallback(idleHandle);
      viewer.creditDisplay.removeStaticCredit(credit);
      if (!scene.isDestroyed() && (scene.skyBox as unknown) === env) scene.skyBox = undefined;
      env.destroy();
    },
  };

  if (import.meta.env?.DEV) (window as unknown as { __sky?: SkyHandle }).__sky = api;
  return api;
}
