/**
 * createGlobe — builds the Cesium Viewer: high-quality Earth (Bing aerial via
 * ion, Cesium World Terrain with water mask, Black Marble night lights),
 * physically lit atmosphere, real-time clock, no default widgets, and the
 * AdaptiveQuality controller that keeps the frame rate up.
 */
import 'cesium/Build/Cesium/Widgets/widgets.css';
import {
  Cartesian3,
  ClockRange,
  ClockStep,
  Color,
  Credit,
  DynamicAtmosphereLightingType,
  ImageryLayer,
  type ImageryProvider,
  UrlTemplateImageryProvider,
  WebMercatorTilingScheme,
  Ion,
  IonImageryProvider,
  IonWorldImageryStyle,
  JulianDate,
  ScreenSpaceEventType,
  Terrain,
  Viewer,
} from 'cesium';
import { AdaptiveQuality } from './AdaptiveQuality.ts';

// Belt and braces next to the Vite `define`: Cesium resolves its Workers /
// Assets / Widgets lazily from this global the first time it needs them.
{
  const w = window as unknown as { CESIUM_BASE_URL?: string };
  w.CESIUM_BASE_URL ??= '/cesium/';
}

export interface GlobeContext {
  viewer: Viewer;
  quality: AdaptiveQuality;
  destroy(): void;
}

export interface CreateGlobeOptions {
  /** Cesium ion access token. Empty/undefined -> Cesium's built-in default token (with a warning). */
  ionToken?: string;
  /** Called after every AdaptiveQuality level change (main.ts: lets the sky re-evaluate its cube size). */
  onQualityChange?: (quality: AdaptiveQuality) => void;
}

/**
 * Start view: far out above Europe / Atlantic. The intro flight to the
 * overview is flown by the CameraRig (src/render/CameraRig.ts), which owns the
 * camera every frame; a Cesium camera.flyTo would be overwritten by it.
 */
export const START_VIEW = { lon: 5, lat: 25, heightM: 45_000_000 } as const;

/** Cesium ion asset: "Earth at Night" (NASA Black Marble 2017). */
const NIGHT_LIGHTS_ASSET_ID = 3812;

/**
 * Public fallback for night lights: NASA GIBS VIIRS Black Marble (2016),
 * Web-Mercator WMTS, no key, CORS-enabled, max level 8.
 */
const GIBS_BLACK_MARBLE_URL =
  'https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/VIIRS_Black_Marble/default/2016-01-01/GoogleMapsCompatible_Level8/{z}/{y}/{x}.png';

const NIGHT_LAYER_OPTIONS = { dayAlpha: 0.0, nightAlpha: 1.0, brightness: 1.4 } as const;

async function addNightLights(viewer: Viewer): Promise<void> {
  let provider: ImageryProvider | undefined;
  try {
    provider = await IonImageryProvider.fromAssetId(NIGHT_LIGHTS_ASSET_ID);
  } catch {
    console.info('[globe] ion asset 3812 (Earth at Night) not in this account — using NASA GIBS Black Marble');
    provider = new UrlTemplateImageryProvider({
      url: GIBS_BLACK_MARBLE_URL,
      tilingScheme: new WebMercatorTilingScheme(),
      maximumLevel: 8,
      credit: new Credit('NASA Earth Observatory / GIBS — VIIRS Black Marble'),
    });
  }
  if (viewer.isDestroyed()) return;
  const layer = new ImageryLayer(provider, NIGHT_LAYER_OPTIONS);
  layer.errorEvent.addEventListener((err: unknown) => console.warn('[globe] night lights tile error', err));
  viewer.imageryLayers.add(layer);
}

let warnedDefaultToken = false;

export async function createGlobe(container: HTMLElement, opts: CreateGlobeOptions = {}): Promise<GlobeContext> {
  const token = opts.ionToken?.trim();
  if (token) {
    Ion.defaultAccessToken = token;
  } else if (!warnedDefaultToken) {
    warnedDefaultToken = true;
    console.warn(
      '[globe] No Cesium ion token configured; using Cesium\'s default demo token. ' +
        'Set one (e.g. VITE_CESIUM_ION_TOKEN) for production use.',
    );
  }

  // --- Earth layers (each optional: a failing asset must never break the app) ---
  let baseLayer: ImageryLayer | false = false;
  try {
    const worldImageryOptions = { style: IonWorldImageryStyle.AERIAL };
    baseLayer = ImageryLayer.fromWorldImagery(
      worldImageryOptions as unknown as ImageryLayer.WorldImageryConstructorOptions,
    );
    baseLayer.errorEvent.addEventListener((err: unknown) => console.warn('[globe] world imagery failed', err));
  } catch (err) {
    console.warn('[globe] world imagery unavailable', err);
    baseLayer = false;
  }

  let terrain: Terrain | undefined;
  try {
    // requestVertexNormals stays false on purpose: with terrain normals Cesium
    // switches the globe shader to ENABLE_VERTEX_LIGHTING, a path that ignores
    // ImageryLayer.dayAlpha/nightAlpha — the night-lights layer would then be
    // painted over the day side too (see GlobeFS.glsl, APPLY_DAY_NIGHT_ALPHA).
    terrain = Terrain.fromWorldTerrain({ requestWaterMask: true, requestVertexNormals: false });
    terrain.errorEvent.addEventListener((err: unknown) => console.warn('[globe] world terrain failed', err));
  } catch (err) {
    console.warn('[globe] world terrain unavailable', err);
    terrain = undefined;
  }

  const viewer = new Viewer(container, {
    animation: false,
    timeline: false,
    baseLayerPicker: false,
    geocoder: false,
    homeButton: false,
    infoBox: false,
    sceneModePicker: false,
    selectionIndicator: false,
    navigationHelpButton: false,
    navigationInstructionsInitiallyVisible: false,
    fullscreenButton: false,
    vrButton: false,
    projectionPicker: false,
    // Credits stay (attribution is legally required); the HUD restyles them.
    baseLayer,
    terrain,
    scene3DOnly: true,
    shouldAnimate: true,
    // Native devicePixelRatio; AdaptiveQuality scales resolution down if needed.
    useBrowserRecommendedResolution: false,
    msaaSamples: 4,
    contextOptions: {
      webgl: {
        powerPreference: 'high-performance',
        alpha: false,
        antialias: false, // Cesium does its own MSAA (WebGL2 multisample FBOs)
      },
    },
    // Continuous rendering, vsync-paced.
    requestRenderMode: false,
    targetFrameRate: undefined,
  });

  const scene = viewer.scene;
  const globe = scene.globe;

  // --- Image quality ---
  // HDR stays OFF: with Cesium's PBR-neutral tonemapper the sun-lit globe is
  // compressed into a dark blue haze and the night-lights layer bleeds onto
  // the day side (verified in the browser). LDR gives crisp, true-colour
  // imagery and saves a float render target + tonemapping pass per frame.
  scene.highDynamicRange = false;
  // 4x MSAA already antialiases geometry edges; FXAA on top would cost an
  // extra full-screen pass and blur the ~2 px satellite cores.
  scene.postProcessStages.fxaa.enabled = false;
  scene.backgroundColor = Color.BLACK;

  // --- Globe look ---
  globe.baseColor = Color.BLACK;
  globe.enableLighting = true;
  globe.dynamicAtmosphereLighting = true;
  globe.dynamicAtmosphereLightingFromSun = true;
  globe.showGroundAtmosphere = true;
  globe.showWaterEffect = true;
  globe.maximumScreenSpaceError = 1.5;
  globe.tileCacheSize = 1000;
  // Keep the day/night terminator visible from orbit down to ~30 km altitude
  // (distances are measured from the Earth's centre). Cesium's defaults
  // flatten lighting below ~3,600 km, i.e. at satellite altitudes.
  globe.lightingFadeOutDistance = 6_410_000;
  globe.lightingFadeInDistance = 6_780_000;

  // Atmosphere tuned for a crisp space look.
  scene.atmosphere.dynamicLighting = DynamicAtmosphereLightingType.SUNLIGHT;
  const sky = scene.skyAtmosphere;
  if (sky) {
    sky.show = true;
    sky.perFragmentAtmosphere = true;
    sky.atmosphereLightIntensity = 40;
    sky.saturationShift = 0.08;
    sky.brightnessShift = -0.02;
  }
  if (scene.sun) {
    scene.sun.show = true;
    scene.sun.glowFactor = 1.4;
  }
  if (scene.moon) scene.moon.show = true;
  if (scene.skyBox) scene.skyBox.show = true;

  // Night lights: only visible on the unlit hemisphere.
  // IMPORTANT: the layer is only added once its provider has *successfully*
  // loaded. An ImageryLayer whose provider failed stays "not ready" forever,
  // and Cesium then never renders the globe tiles that layer covers (seen in
  // the browser: the whole western hemisphere missing when the ion token's
  // account did not include asset 3812).
  void addNightLights(viewer);

  // --- Clock: real time (future phases add scrubbing via clock.multiplier) ---
  const clock = viewer.clock;
  clock.currentTime = JulianDate.now();
  clock.clockStep = ClockStep.SYSTEM_CLOCK_MULTIPLIER;
  clock.clockRange = ClockRange.UNBOUNDED;
  clock.multiplier = 1;
  clock.shouldAnimate = true;

  // --- Camera ---
  // Only the start view is set here. main.ts then creates the CameraRig, which
  // switches Cesium's ScreenSpaceCameraController off (inputs + collision
  // detection), writes the camera every frame and flies the boot intro.
  scene.camera.setView({
    destination: Cartesian3.fromDegrees(START_VIEW.lon, START_VIEW.lat, START_VIEW.heightM),
  });

  // No entity tracking / selection on click: SelectionController does picking.
  const handler = viewer.cesiumWidget.screenSpaceEventHandler;
  handler.removeInputAction(ScreenSpaceEventType.LEFT_DOUBLE_CLICK);
  handler.removeInputAction(ScreenSpaceEventType.LEFT_CLICK);

  const quality = new AdaptiveQuality(viewer, opts.onQualityChange ? { onChange: opts.onQualityChange } : {});

  let destroyed = false;

  const destroy = (): void => {
    if (destroyed) return;
    destroyed = true;
    quality.destroy();
    if (!viewer.isDestroyed()) viewer.destroy();
  };

  // Resolve once the first frame has rendered (or after 3 s, e.g. hidden tab).
  await new Promise<void>((resolve) => {
    let settled = false;
    const finish = (): void => {
      if (settled) return;
      settled = true;
      removeListener();
      clearTimeout(timer);
      resolve();
    };
    const removeListener = scene.postRender.addEventListener(finish);
    const timer = setTimeout(finish, 3000);
  });

  return { viewer, quality, destroy };
}
