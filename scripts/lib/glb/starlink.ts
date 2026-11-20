/**
 * Procedural Starlink models (see public/models/README.md for conventions):
 *
 *   metres, glTF Y-up, +X = flight direction (velocity), -Y = nadir,
 *   solar-array (wing) axis = ±Z, "R" wing at +Z, "L" wing at -Z
 *   (right/left when looking along +X with +Y up).
 *
 * Node layout:
 *   Starlink_<id>            root (no mesh; extras = metadata)
 *   ├─ Bus                   static bus mesh (plates, MLI, antennas, drives)
 *   ├─ ArrayPivot_L / _R     at the solar-array drive output flange; identity
 *   │  └─ Wing_L / Wing_R    rotation. Rotate the pivot about its local Z to
 *   │                        sun-track; rest pose = cells facing +Y (zenith).
 *   └─ LaserTerminal_0..n    origin = lens-dome centre (beam aperture), +Y up.
 */
import { GltfDoc, SAMPLER_CLAMP, SAMPLER_REPEAT } from './gltf.ts';
import type { MaterialDef, NodeDef } from './gltf.ts';
import { alignY, identity, mul, quatAxisAngle, rotationAxisAngle, translation } from './math.ts';
import type { Vec3 } from './math.ts';
import { MeshBuilder, annulus, box, cylinder, dish, dome, rod, solidBox } from './mesh.ts';
import type { FaceSpec } from './mesh.ts';
import { encodeRgb, rasterize } from './texture.ts';
import type { ImageFormat, TexSpec } from './texture.ts';
import { graphitePanels, mliFoil, phasedArray, solarBack, solarCells } from './starlinkTextures.ts';

export const GENERATOR = 'starlink-tracker scripts/build-models.mjs (procedural)';

export interface ModelDims {
  bus: Vec3;
  wingCount: 1 | 2;
  wingWidth: number;
  wingLength: number;
  panels: number;
  panelGap: number;
  /** Distance from the bus edge (SADA flange) to the first panel. */
  boom: number;
  laserTerminals: number;
  /**
   * Panel columns across the wing width (default 1). With 2 the wing is two
   * strips of panels either side of a central carbon spine; `wingWidth` is
   * the overall cell extent (both columns + `columnGap`).
   */
  columns?: number;
  /** Gap between panel columns (metres, default 0). */
  columnGap?: number;
  /** Scale of the drive/hub/yoke hardware (default 1). */
  hw?: number;
}

export const V2MINI: ModelDims = {
  bus: [4.1, 0.25, 2.7],
  wingCount: 2,
  wingWidth: 4.1,
  wingLength: 12.8,
  panels: 6,
  panelGap: 0.05,
  boom: 0.77,
  laserTerminals: 3,
};

export const V15: ModelDims = {
  bus: [1.4, 0.2, 2.8],
  wingCount: 1,
  wingWidth: 2.8,
  wingLength: 8.1,
  panels: 6,
  panelGap: 0.04,
  boom: 0.5,
  laserTerminals: 4,
};

/**
 * Starlink V3 (Starship-launched, 2026+): an artist's approximation. SpaceX
 * has published no drawings; public figures are ~2,000 kg, six laser
 * terminals and "about the size of a Boeing 737" (one secondhand report:
 * ~60 m deployed span). Bus = Starship-dispenser flat pack, 7.5 x 0.4 x 3.2 m.
 * Span = 2 * (1.6 + SADA 0.08 + boom 1.3 + 27) = 59.96 m.
 */
export const V3: ModelDims = {
  bus: [7.5, 0.4, 3.2],
  wingCount: 2,
  wingWidth: 4.9,
  wingLength: 27,
  panels: 22,
  panelGap: 0.05,
  boom: 1.3,
  laserTerminals: 6,
  columns: 2,
  columnGap: 0.2,
  hw: 1.6,
};

/** Depth of the solar-array drive housing protruding from the bus edge. */
const SADA_DEPTH = 0.08;
const PANEL_T = 0.03;

export interface BuiltModel {
  glb: Uint8Array;
  textures: Array<{ name: string; bytes: number; width: number; height: number; mime: string }>;
}

// ---------------------------------------------------------------------------
// Texture / material helpers
// ---------------------------------------------------------------------------

interface TexSetOpts {
  spec: TexSpec;
  sampler: number;
  base?: ImageFormat;
  orm?: ImageFormat;
  normal?: ImageFormat;
}

interface TexSet {
  base?: number;
  orm?: number;
  normal?: number;
}

class ModelContext {
  readonly doc: GltfDoc;
  readonly report: BuiltModel['textures'] = [];
  readonly repeat: number;
  readonly clamp: number;

  constructor(doc: GltfDoc) {
    this.doc = doc;
    this.repeat = doc.addSampler(SAMPLER_REPEAT);
    this.clamp = doc.addSampler(SAMPLER_CLAMP);
  }

  async texSet(o: TexSetOpts): Promise<TexSet> {
    const t0 = performance.now();
    const maps = rasterize(o.spec);
    const out: TexSet = {};
    const add = async (kind: 'base' | 'orm' | 'normal', fmt: ImageFormat | undefined): Promise<void> => {
      if (!fmt) return;
      const bytes = await encodeRgb(maps[kind], maps.width, maps.height, fmt);
      const mime = fmt.kind === 'png' ? 'image/png' : 'image/jpeg';
      const name = `${o.spec.name}_${kind}`;
      out[kind] = this.doc.addTexture(name, bytes, mime, o.sampler);
      this.report.push({ name, bytes: bytes.byteLength, width: maps.width, height: maps.height, mime });
    };
    await add('base', o.base);
    await add('orm', o.orm);
    await add('normal', o.normal);
    const ms = performance.now() - t0;
    if (process.env.MODELS_VERBOSE) console.log(`  texture ${o.spec.name} ${maps.width}x${maps.height} in ${ms.toFixed(0)} ms`);
    return out;
  }

  texturedMaterial(name: string, t: TexSet, extra: Partial<MaterialDef> = {}): number {
    return this.doc.addMaterial({
      name,
      baseColorTexture: t.base,
      ormTexture: t.orm,
      useOcclusion: t.orm !== undefined,
      occlusionStrength: 1,
      metallicFactor: t.orm !== undefined ? 1 : 0,
      roughnessFactor: 1,
      normalTexture: t.normal,
      ...extra,
    });
  }
}

const JPG = (quality: number): ImageFormat => ({ kind: 'jpeg', quality });

interface Mats {
  graphite: number;
  mli: number;
  phased: number;
  cells: number;
  back: number;
  frame: number;
  darkMetal: number;
  carbon: number;
  white: number;
  dish: number;
  lens: number;
  black: number;
}

/** Width of one panel column (= wingWidth for single-column wings). */
function columnWidth(d: ModelDims): number {
  const n = d.columns ?? 1;
  return (d.wingWidth - (n - 1) * (d.columnGap ?? 0)) / n;
}

async function makeMaterials(ctx: ModelContext, d: ModelDims, phasedSize: [number, number], seed: number): Promise<Mats> {
  const panelLen = (d.wingLength - (d.panels - 1) * d.panelGap) / d.panels;
  const panelW = columnWidth(d);
  const cellPitch = 0.155;
  const margin = 0.035;
  const cols = Math.max(1, Math.round((panelW - 2 * margin) / cellPitch));
  const rows = Math.max(1, Math.round((panelLen - 2 * margin) / cellPitch));
  const cellsTex = await ctx.texSet({
    spec: solarCells({ panelU: panelW, panelV: panelLen, cols, rows, margin, width: 2048, height: 1024, seed: seed + 1 }),
    sampler: ctx.clamp,
    base: JPG(90),
    orm: JPG(90),
    normal: JPG(90),
  });
  const backTex = await ctx.texSet({ spec: solarBack(1, 512, seed + 2), sampler: ctx.repeat, base: JPG(88), normal: JPG(88) });
  const mliTex = await ctx.texSet({ spec: mliFoil(0.5, 1024, seed + 3), sampler: ctx.repeat, base: JPG(88), orm: JPG(88), normal: JPG(90) });
  const graphiteTex = await ctx.texSet({
    spec: graphitePanels(d.bus[0], d.bus[2], 1024, seed + 4),
    sampler: ctx.clamp,
    base: JPG(90),
    orm: JPG(88),
    normal: JPG(90),
  });
  const phasedTex = await ctx.texSet({
    spec: phasedArray(phasedSize[0], phasedSize[1], 1024, seed + 5),
    sampler: ctx.clamp,
    base: JPG(90),
    orm: JPG(88),
    normal: JPG(90),
  });
  const doc = ctx.doc;
  return {
    cells: ctx.texturedMaterial('SolarCells', cellsTex),
    back: ctx.texturedMaterial('SolarArrayBack', backTex, { metallicFactor: 0, roughnessFactor: 0.8 }),
    mli: ctx.texturedMaterial('MLI_Foil', mliTex),
    graphite: ctx.texturedMaterial('BusGraphite', graphiteTex),
    phased: ctx.texturedMaterial('PhasedArray', phasedTex),
    frame: doc.addMaterial({ name: 'Aluminium', baseColorFactor: [0.72, 0.72, 0.74, 1], metallicFactor: 1, roughnessFactor: 0.38 }),
    darkMetal: doc.addMaterial({ name: 'DarkAnodised', baseColorFactor: [0.035, 0.035, 0.04, 1], metallicFactor: 0.7, roughnessFactor: 0.42 }),
    carbon: doc.addMaterial({ name: 'CarbonTube', baseColorFactor: [0.022, 0.022, 0.025, 1], metallicFactor: 0.1, roughnessFactor: 0.4 }),
    white: doc.addMaterial({ name: 'WhitePaint', baseColorFactor: [0.74, 0.74, 0.72, 1], metallicFactor: 0, roughnessFactor: 0.55 }),
    dish: doc.addMaterial({ name: 'DishReflector', baseColorFactor: [0.8, 0.8, 0.78, 1], metallicFactor: 0, roughnessFactor: 0.6, doubleSided: true }),
    lens: doc.addMaterial({ name: 'LaserLens', baseColorFactor: [0.008, 0.01, 0.02, 1], metallicFactor: 0.45, roughnessFactor: 0.06 }),
    black: doc.addMaterial({ name: 'BlackBaffle', baseColorFactor: [0.01, 0.01, 0.01, 1], metallicFactor: 0, roughnessFactor: 0.85 }),
  };
}

/** One builder per material, created lazily. */
class Parts {
  private readonly map = new Map<number, MeshBuilder>();
  private readonly prefix: string;

  constructor(prefix: string) {
    this.prefix = prefix;
  }

  get(material: number): MeshBuilder {
    let mb = this.map.get(material);
    if (!mb) {
      mb = new MeshBuilder(`${this.prefix}_${material}`);
      this.map.set(material, mb);
    }
    return mb;
  }

  list(): Array<[MeshBuilder, number]> {
    return [...this.map.entries()].map(([mat, mb]) => [mb, mat]);
  }
}

// ---------------------------------------------------------------------------
// Sub-assemblies
// ---------------------------------------------------------------------------

/** Bus body: MLI-wrapped core between two graphite plates (lip overhang 1 cm). */
function busBody(P: Parts, M: Mats, d: ModelDims): void {
  const I = identity();
  const [bx, by, bz] = d.bus;
  const plateT = 0.015;
  const lip = 0.01;
  const mli: FaceSpec = { mb: P.get(M.mli), uv: { kind: 'world', tile: 0.5 } };
  box(I, [0, 0, 0], [bx - 2 * lip, by - 2 * plateT, bz - 2 * lip], { px: mli, nx: mli, pz: mli, nz: mli });
  const plateTop: FaceSpec = { mb: P.get(M.graphite), uv: { kind: 'fit' } };
  const edge: FaceSpec = { mb: P.get(M.darkMetal), uv: { kind: 'world', tile: 1 } };
  box(I, [0, by / 2 - plateT / 2, 0], [bx, plateT, bz], { py: plateTop, ny: edge, px: edge, nx: edge, pz: edge, nz: edge });
  box(I, [0, -by / 2 + plateT / 2, 0], [bx, plateT, bz], {
    ny: { mb: P.get(M.graphite), uv: { kind: 'fit' } },
    py: edge,
    px: edge,
    nx: edge,
    pz: edge,
    nz: edge,
  });
}

/** Flat phased-array panel hanging under the bus (antenna face -Y). */
function phasedPanel(P: Parts, M: Mats, busBottom: number, cx: number, cz: number, sx: number, sz: number): void {
  const t = 0.035;
  const standoff = 0.006;
  const face: FaceSpec = { mb: P.get(M.phased), uv: { kind: 'fit' } };
  const side: FaceSpec = { mb: P.get(M.frame), uv: { kind: 'world', tile: 1 } };
  box(identity(), [cx, busBottom - standoff - t / 2, cz], [sx, t, sz], { ny: face, px: side, nx: side, pz: side, nz: side });
  // Stand-off rails.
  const rail = P.get(M.darkMetal);
  for (const s of [-1, 1]) solidBox(identity(), rail, [cx + s * (sx / 2 - 0.05), busBottom - standoff / 2, cz], [0.03, standoff, sz * 0.9]);
}

/** Gimballed parabolic gateway antenna pointing -Y, mounted at (x, busBottom, z). */
function gatewayDish(P: Parts, M: Mats, busBottom: number, x: number, z: number, radius: number): void {
  const stalk = 0.09;
  const depth = radius * 0.32;
  // Gimbal stalk + motor block.
  const down = mul(translation([x, busBottom, z]), rotationAxisAngle([1, 0, 0], Math.PI)); // +Y -> -Y
  cylinder(down, { side: P.get(M.darkMetal), top: P.get(M.darkMetal), radius: 0.03, height: stalk, segments: 16 });
  solidBox(identity(), P.get(M.white), [x, busBottom - 0.025, z], [0.1, 0.05, 0.08], 1);
  // Dish opening toward -Y: vertex at the stalk end.
  const dm = mul(translation([x, busBottom - stalk, z]), rotationAxisAngle([1, 0, 0], Math.PI));
  dish(dm, P.get(M.dish), radius, depth, 40, 10);
  // Feed at the focus with three struts.
  const f = (radius * radius) / (4 * depth);
  const focus: Vec3 = [x, busBottom - stalk - f, z];
  cylinder(mul(translation(focus), rotationAxisAngle([1, 0, 0], Math.PI)), {
    side: P.get(M.white),
    top: P.get(M.black),
    bottom: P.get(M.white),
    radius: 0.018,
    height: 0.03,
    y0: -0.015,
    segments: 12,
  });
  for (let k = 0; k < 3; k++) {
    const a = (k / 3) * Math.PI * 2 + 0.3;
    const rim: Vec3 = [x + Math.cos(a) * radius * 0.93, busBottom - stalk - depth * 0.87, z + Math.sin(a) * radius * 0.93];
    rod(identity(), P.get(M.frame), rim, focus, 0.0035, 6);
  }
}

/**
 * Two-axis gimballed gateway dish under the bus at (x, busBottom, z): an
 * azimuth turntable, a fork, an elevation axle and a tilted reflector with
 * feed and struts. `az` turns the fork about the nadir axis, `el` tilts the
 * boresight off nadir.
 */
function gimbalDish(P: Parts, M: Mats, busBottom: number, x: number, z: number, radius: number, az: number, el: number): void {
  const dm = P.get(M.darkMetal);
  const wh = P.get(M.white);
  const fr = P.get(M.frame);
  // Local frame: +Y = nadir (rotated 180° about X), then the azimuth turn.
  const base = mul(mul(translation([x, busBottom, z]), rotationAxisAngle([1, 0, 0], Math.PI)), rotationAxisAngle([0, 1, 0], az));
  cylinder(base, { side: dm, top: dm, radius: 0.12, height: 0.03, segments: 28 });
  cylinder(base, { side: fr, top: fr, radius: 0.085, height: 0.035, y0: 0.03, segments: 28 });
  for (const s of [-1, 1]) solidBox(base, wh, [s * 0.085, 0.13, 0], [0.025, 0.13, 0.08], 1);
  const pivotY = 0.17;
  const axle = mul(base, mul(translation([0, pivotY, 0]), rotationAxisAngle([0, 0, 1], Math.PI / 2)));
  cylinder(axle, { side: dm, top: dm, bottom: dm, radius: 0.026, height: 0.21, y0: -0.105, segments: 14 });
  // Reflector frame: tilted by `el` about the axle; dish opens toward +Y.
  const g = mul(base, mul(translation([0, pivotY, 0]), rotationAxisAngle([1, 0, 0], el)));
  solidBox(g, dm, [0, 0.02, 0], [0.13, 0.07, 0.13], 1);
  const v0 = 0.055;
  const depth = radius * 0.3;
  dish(mul(g, translation([0, v0, 0])), P.get(M.dish), radius, depth, 48, 12);
  const f = (radius * radius) / (4 * depth);
  const focus: Vec3 = [0, v0 + f, 0];
  cylinder(mul(g, translation(focus)), { side: wh, top: wh, bottom: P.get(M.black), radius: 0.03, height: 0.05, y0: -0.025, segments: 14 });
  for (let k = 0; k < 4; k++) {
    const a = (k / 4) * Math.PI * 2 + Math.PI / 4;
    const rim: Vec3 = [Math.cos(a) * radius * 0.93, v0 + depth * 0.86, Math.sin(a) * radius * 0.93];
    rod(g, fr, rim, focus, 0.0045, 6);
  }
}

/** Star tracker: black baffle tube tilted away from the bus, on the top plate. */
function starTracker(P: Parts, M: Mats, base: Vec3, tiltAxis: Vec3, tilt: number): void {
  const m = mul(translation(base), rotationAxisAngle(tiltAxis, tilt));
  solidBox(m, P.get(M.darkMetal), [0, 0.03, 0], [0.09, 0.06, 0.09], 1);
  cylinder(m, { side: P.get(M.black), radius: 0.04, radiusTop: 0.05, height: 0.1, y0: 0.06, segments: 20 });
  cylinder(m, { side: P.get(M.black), radius: 0.037, radiusTop: 0.047, height: 0.095, y0: 0.065, segments: 20, inward: true });
  annulus(m, P.get(M.black), 0.047, 0.05, 0.16, true, 20);
  cylinder(m, { top: P.get(M.lens), radius: 0.03, height: 0.001, y0: 0.07, segments: 20 });
}

/** Hall-effect thruster on the aft (-X) face. */
function thruster(P: Parts, M: Mats, at: Vec3): void {
  const m = mul(translation(at), rotationAxisAngle([0, 0, 1], Math.PI / 2)); // +Y -> -X
  cylinder(m, { side: P.get(M.frame), radius: 0.075, height: 0.05, segments: 28 });
  annulus(m, P.get(M.black), 0.035, 0.065, 0.05, true, 28);
  annulus(m, P.get(M.frame), 0.065, 0.075, 0.05, true, 28);
  cylinder(m, { side: P.get(M.darkMetal), top: P.get(M.darkMetal), radius: 0.035, height: 0.055, segments: 20 });
  // Hollow cathode beside the thruster.
  cylinder(mul(m, translation([0, 0, 0.11])), { side: P.get(M.frame), top: P.get(M.black), radius: 0.012, height: 0.06, segments: 10 });
}

/** Solar-array drive housing at the bus edge (static part, on the bus). */
function sadaHousing(P: Parts, M: Mats, zEdge: number, sign: number, hw = 1): void {
  const m = mul(translation([0, 0, zEdge]), alignY([0, 0, sign]));
  cylinder(m, { side: P.get(M.darkMetal), top: P.get(M.darkMetal), radius: 0.085 * hw, height: SADA_DEPTH - 0.01, segments: 28 });
  cylinder(m, { side: P.get(M.frame), radius: 0.095 * hw, height: 0.015, segments: 28 });
}

/**
 * Laser terminal (one shared mesh). Origin = lens-dome centre; flange sits
 * at y = -TERMINAL_STACK so the node goes at busTop + TERMINAL_STACK.
 */
export const TERMINAL_STACK = 0.17;

function laserTerminalMesh(ctx: ModelContext, M: Mats): number {
  const P = new Parts('LaserTerminal');
  const I = identity();
  cylinder(I, { side: P.get(M.frame), top: P.get(M.frame), bottom: P.get(M.frame), radius: 0.14, height: 0.02, y0: -TERMINAL_STACK, segments: 40 });
  cylinder(I, { side: P.get(M.white), radius: 0.125, height: 0.13, y0: -0.15, segments: 40 });
  annulus(I, P.get(M.white), 0.105, 0.125, -0.02, true, 40);
  cylinder(I, { side: P.get(M.frame), radius: 0.105, height: 0.02, y0: -0.02, segments: 40 });
  annulus(I, P.get(M.frame), 0.09, 0.105, 0, true, 40);
  dome(I, P.get(M.lens), 0.09, Math.PI / 2, 0.9, 40, 14);
  // Cooling fins / cable box on the side.
  solidBox(I, P.get(M.darkMetal), [0.13, -0.11, 0], [0.04, 0.07, 0.08], 1);
  return ctx.doc.addMesh('LaserTerminal', P.list());
}

/**
 * One wing (yoke + hinged panels), built in pivot space: pivot at the
 * origin, the wing extends toward +Z, cells face +Y.
 */
function wingMesh(ctx: ModelContext, M: Mats, d: ModelDims): number {
  const P = new Parts('Wing');
  const I = identity();
  const W = d.wingWidth;
  const L = (d.wingLength - (d.panels - 1) * d.panelGap) / d.panels;
  const root = d.boom;
  const nCols = d.columns ?? 1;
  const colGap = d.columnGap ?? 0;
  const CW = columnWidth(d);
  const hw = d.hw ?? 1;
  // Output shaft + hub.
  const zAxis = alignY([0, 0, 1]);
  cylinder(zAxis, { side: P.get(M.frame), radius: 0.05 * hw, height: 0.1 * hw, segments: 20 });
  solidBox(I, P.get(M.darkMetal), [0, 0, 0.14 * hw], [0.22 * hw, 0.12 * hw, 0.08 * hw], 1);
  // Yoke: carbon V + centre tube, root bar with end fittings.
  const yokeEnd = root - 0.045;
  const spread = W / 2 - 0.3;
  const yokeStart = 0.17 * hw;
  rod(I, P.get(M.carbon), [0, 0, yokeStart], [spread, 0, yokeEnd], 0.022 * hw, 12);
  rod(I, P.get(M.carbon), [0, 0, yokeStart], [-spread, 0, yokeEnd], 0.022 * hw, 12);
  rod(I, P.get(M.carbon), [0, 0, yokeStart], [0, 0, yokeEnd], 0.028 * hw, 12);
  solidBox(I, P.get(M.frame), [0, 0, root - 0.03], [W * 0.92, 0.045, 0.04], 1);
  for (const s of [-1, 0, 1]) solidBox(I, P.get(M.darkMetal), [s * spread, 0, yokeEnd], [0.07, 0.06, 0.05], 1);
  // Panels, in one or more columns across the wing width.
  const hingeX = CW / 2 - 0.35;
  const hinge = (cx: number, zc: number, gap: number): void => {
    for (const s of [-1, 1]) {
      const m = mul(translation([cx + s * hingeX, -PANEL_T / 2 - 0.004, zc]), rotationAxisAngle([0, 0, 1], Math.PI / 2));
      cylinder(m, { side: P.get(M.frame), top: P.get(M.frame), bottom: P.get(M.frame), radius: 0.012, height: 0.16, y0: -0.08, segments: 12 });
      solidBox(I, P.get(M.frame), [cx + s * hingeX, -PANEL_T / 2 - 0.002, zc], [0.14, 0.004, gap + 0.09], 1);
    }
  };
  for (let k = 0; k < d.panels; k++) {
    const z0 = root + k * (L + d.panelGap);
    for (let c = 0; c < nCols; c++) {
      // Single column: cx is exactly 0 (keeps V2 Mini / v1.5 output unchanged).
      const cx = -W / 2 + CW / 2 + c * (CW + colGap);
      const j = k + c;
      const cells: FaceSpec = { mb: P.get(M.cells), uv: { kind: 'fit', flipU: (j & 1) === 1, flipV: (k & 2) === 2 } };
      const back: FaceSpec = { mb: P.get(M.back), uv: { kind: 'world', tile: 1, offset: [0.37 * k + 0.53 * c, 0.61 * k] } };
      const edge: FaceSpec = { mb: P.get(M.frame), uv: { kind: 'world', tile: 1 } };
      box(I, [cx, 0, z0 + L / 2], [CW, PANEL_T, L], { py: cells, ny: back, px: edge, nx: edge, pz: edge, nz: edge });
      // Back-side stiffener frame (two longerons).
      for (const s of [-1, 1]) {
        solidBox(I, P.get(M.frame), [cx + s * (CW / 2 - 0.02), -PANEL_T / 2 - 0.006, z0 + L / 2], [0.03, 0.012, L - 0.02], 1);
      }
      hinge(cx, k === 0 ? root - 0.005 : z0 - d.panelGap / 2, k === 0 ? 0.03 : d.panelGap);
    }
  }
  if (nCols > 1) wingSpine(P, M, d, L, hw);
  return ctx.doc.addMesh('Wing', P.list());
}

/**
 * Multi-column wings: carbon spine along the wing axis between the columns,
 * cable tray under it, cross-ties and spine clamps at every panel gap, and
 * a tip fitting with an outer tie bar.
 */
function wingSpine(P: Parts, M: Mats, d: ModelDims, L: number, hw: number): void {
  const I = identity();
  const root = d.boom;
  const tip = root + d.wingLength;
  const W = d.wingWidth;
  const r = Math.min(0.045, (d.columnGap ?? 0) * 0.3);
  rod(I, P.get(M.carbon), [0, 0, root - 0.03], [0, 0, tip + 0.06], r, 16);
  solidBox(I, P.get(M.darkMetal), [0, -PANEL_T / 2 - 0.03, (root + tip) / 2], [0.06, 0.02, d.wingLength - 0.1], 1);
  for (let k = 1; k < d.panels; k++) {
    const zc = root + k * (L + d.panelGap) - d.panelGap / 2;
    solidBox(I, P.get(M.frame), [0, -PANEL_T / 2 - 0.01, zc], [W - 0.5, 0.012, 0.035], 1);
    solidBox(I, P.get(M.darkMetal), [0, 0, zc], [0.13, 0.07, 0.06], 1);
  }
  solidBox(I, P.get(M.darkMetal), [0, 0, tip + 0.08], [0.16 * hw, 0.09, 0.08], 1);
  solidBox(I, P.get(M.frame), [0, -PANEL_T / 2 - 0.01, tip - 0.02], [W * 0.92, 0.02, 0.035], 1);
}

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

interface Layout {
  id: string;
  label: string;
  dims: ModelDims;
  phased: Array<[number, number]>; // panel centres (x, z)
  phasedSize: [number, number];
  dishes: Array<[number, number, number]>; // x, z, radius
  lasers: Vec3[]; // x, z on the top plate (y ignored)
  trackers: Array<[number, number, Vec3, number]>; // x, z, tilt axis, tilt
  thrusters: Vec3[];
  seed: number;
  /** glTF asset.copyright (default: the generic procedural notice). */
  copyright?: string;
  /** Extra root-node extras, merged after the common ones. */
  extras?: Record<string, unknown>;
  /** Uniform scale of the LaserTerminal_n nodes (default: unscaled). */
  terminalScale?: number;
  /** Model-specific bus hardware, added after the common parts. */
  busExtras?: (P: Parts, M: Mats, busTop: number, busBottom: number) => void;
}

async function buildModel(L: Layout): Promise<BuiltModel> {
  const doc = new GltfDoc(GENERATOR, L.copyright ?? 'Procedurally generated for starlink-tracker. Not an official SpaceX model.');
  const ctx = new ModelContext(doc);
  const d = L.dims;
  const M = await makeMaterials(ctx, d, L.phasedSize, L.seed);
  const [, by, bz] = d.bus;
  const busTop = by / 2;
  const busBottom = -by / 2;

  // Bus.
  const P = new Parts('Bus');
  busBody(P, M, d);
  for (const [x, z] of L.phased) phasedPanel(P, M, busBottom, x, z, L.phasedSize[0], L.phasedSize[1]);
  for (const [x, z, r] of L.dishes) gatewayDish(P, M, busBottom, x, z, r);
  for (const [x, z, axis, tilt] of L.trackers) starTracker(P, M, [x, busTop, z], axis, tilt);
  for (const t of L.thrusters) thruster(P, M, t);
  L.busExtras?.(P, M, busTop, busBottom);
  sadaHousing(P, M, bz / 2, 1, d.hw);
  if (d.wingCount === 2) sadaHousing(P, M, -bz / 2, -1, d.hw);
  const busMesh = doc.addMesh('Bus', P.list());

  const wing = wingMesh(ctx, M, d);
  const terminal = laserTerminalMesh(ctx, M);

  const children: number[] = [];
  children.push(doc.addNode({ name: 'Bus', mesh: busMesh }));
  const pivotZ = bz / 2 + SADA_DEPTH;
  if (d.wingCount === 2) {
    const wl = doc.addNode({ name: 'Wing_L', mesh: wing, rotation: quatAxisAngle([0, 1, 0], Math.PI) });
    children.push(doc.addNode({ name: 'ArrayPivot_L', translation: [0, 0, -pivotZ], children: [wl] }));
  }
  const wr = doc.addNode({ name: 'Wing_R', mesh: wing });
  children.push(doc.addNode({ name: 'ArrayPivot_R', translation: [0, 0, pivotZ], children: [wr] }));
  const ts = L.terminalScale;
  L.lasers.forEach((p, i) => {
    const node: NodeDef = { name: `LaserTerminal_${i}`, mesh: terminal, translation: [p[0], busTop + TERMINAL_STACK * (ts ?? 1), p[2]] };
    if (ts !== undefined) node.scale = [ts, ts, ts];
    children.push(doc.addNode(node));
  });

  const span = d.wingCount === 2 ? 2 * (pivotZ + d.boom + d.wingLength) : bz / 2 + pivotZ + d.boom + d.wingLength;
  const root = doc.addNode({
    name: `Starlink_${L.id}`,
    children,
    extras: {
      model: L.label,
      units: 'm',
      axes: { up: '+Y', velocity: '+X', nadir: '-Y', wingAxis: 'Z', rightWing: '+Z' },
      bus: d.bus,
      wing: {
        count: d.wingCount,
        width: d.wingWidth,
        length: d.wingLength,
        panels: d.panels,
        areaM2: d.wingCount * (d.wingWidth - ((d.columns ?? 1) - 1) * (d.columnGap ?? 0)) * (d.wingLength - (d.panels - 1) * d.panelGap),
      },
      spanM: Number(span.toFixed(3)),
      laserTerminals: L.lasers.length,
      pivotRestPose: 'cells face +Y; rotate ArrayPivot_* about local Z',
      provenance: 'procedurally generated; dimensions from public reporting, details are artistic',
      ...L.extras,
    },
  });
  doc.addScene(`Starlink ${L.label}`, [root]);
  return { glb: doc.toGlb(), textures: ctx.report };
}

export function buildV2Mini(): Promise<BuiltModel> {
  const d = V2MINI;
  const [bx] = d.bus;
  const ex = bx / 2 - 0.25;
  return buildModel({
    id: 'V2Mini',
    label: 'V2 Mini',
    dims: d,
    phasedSize: [1.55, 1.1],
    phased: [
      [0.9, 0.62],
      [0.9, -0.62],
      [-0.9, 0.62],
      [-0.9, -0.62],
    ],
    dishes: [
      [1.86, 0.55, 0.16],
      [-1.86, -0.55, 0.16],
    ],
    lasers: [
      [ex + 0.03, 0, 0],
      [-ex - 0.03, 0, -0.85],
      [-ex - 0.03, 0, 0.85],
    ],
    trackers: [
      [0.25, 0.6, [1, 0, 0], 0.45],
      [0.25, -0.6, [1, 0, 0], -0.45],
    ],
    thrusters: [[-bx / 2 + 0.005, 0, 0.0]],
    seed: 2023,
  });
}

export function buildV15(): Promise<BuiltModel> {
  const d = V15;
  const [bx, , bz] = d.bus;
  return buildModel({
    id: 'V15',
    label: 'V1.0/V1.5',
    dims: d,
    phasedSize: [0.62, 1.1],
    phased: [
      [0.34, 0.62],
      [0.34, -0.62],
      [-0.34, 0.62],
      [-0.34, -0.62],
    ],
    dishes: [
      [0, bz / 2 - 0.13, 0.12],
      [0, -bz / 2 + 0.13, 0.12],
    ],
    lasers: [
      [0.52, 0, -1.2],
      [-0.52, 0, -1.2],
      [0.52, 0, 1.08],
      [-0.52, 0, 1.08],
    ],
    trackers: [
      [0.25, 0.1, [0, 0, 1], -0.45],
      [-0.25, 0.1, [0, 0, 1], 0.45],
    ],
    thrusters: [[-bx / 2 + 0.005, 0, -0.5]],
    seed: 2019,
  });
}

/**
 * Starlink V3: an ARTIST'S APPROXIMATION. SpaceX has published no drawings;
 * only the overall size (~60 m span, "about the size of a Boeing 737"), mass
 * (~2,000 kg) and six 400 Gbps laser terminals are public. Everything else
 * (flat-pack bus for Starship dispenser stacking, eight phased arrays, four
 * gimballed gateway dishes, two-column wings) is invented to read as "bigger
 * and more powerful than V2 Mini".
 */
export function buildV3(): Promise<BuiltModel> {
  const d = V3;
  const [bx, , bz] = d.bus;
  const hx = bx / 2;
  const hz = bz / 2;
  return buildModel({
    id: 'V3',
    label: 'V3',
    dims: d,
    // 4 x 2 phased arrays (~2.0 m² each) cover the nadir face between the dishes.
    phasedSize: [1.42, 1.44],
    phased: [
      [2.25, 0.78],
      [2.25, -0.78],
      [0.75, 0.78],
      [0.75, -0.78],
      [-0.75, 0.78],
      [-0.75, -0.78],
      [-2.25, 0.78],
      [-2.25, -0.78],
    ],
    dishes: [],
    // Six terminals: fore pair, aft pair and one per side (staggered).
    lasers: [
      [hx - 0.35, 0, 0.9],
      [hx - 0.35, 0, -0.9],
      [-hx + 0.35, 0, 0.9],
      [-hx + 0.35, 0, -0.9],
      [0.9, 0, hz - 0.3],
      [-0.9, 0, -hz + 0.3],
    ],
    terminalScale: 1.25,
    trackers: [
      [2.1, 1.1, [1, 0, 0], 0.45],
      [2.1, -1.1, [1, 0, 0], -0.45],
      [-2.1, 1.1, [1, 0, 0], 0.45],
    ],
    thrusters: [
      [-hx + 0.005, 0, 0.7],
      [-hx + 0.005, 0, -0.7],
    ],
    busExtras(P, M, busTop, busBottom) {
      // Four gimballed Ka/E/V/W-band gateway dishes at the fore/aft ends of the nadir face.
      gimbalDish(P, M, busBottom, hx - 0.42, 0.8, 0.27, 0.5, 0.35);
      gimbalDish(P, M, busBottom, hx - 0.42, -0.8, 0.27, -0.9, 0.3);
      gimbalDish(P, M, busBottom, -hx + 0.42, 0.8, 0.27, 2.4, 0.4);
      gimbalDish(P, M, busBottom, -hx + 0.42, -0.8, 0.27, -2.2, 0.25);
      // Dispenser stacking rails and pads on the zenith face.
      for (const z of [-0.5, 0.5]) {
        solidBox(identity(), P.get(M.darkMetal), [0, busTop + 0.015, z], [bx - 0.9, 0.03, 0.08], 1);
        for (const x of [-hx + 1.05, 0, hx - 1.05]) solidBox(identity(), P.get(M.frame), [x, busTop + 0.025, z], [0.24, 0.05, 0.24], 1);
      }
      // GNSS patch antennas.
      for (const [x, z] of [
        [-1.5, 1.15],
        [1.5, -1.15],
      ] as const) {
        cylinder(translation([x, busTop, z]), { side: P.get(M.white), top: P.get(M.white), radius: 0.07, height: 0.02, segments: 20 });
      }
      // White radiator panel on the fore (+X) face.
      solidBox(identity(), P.get(M.white), [hx + 0.006, 0, 0], [0.012, 0.3, bz - 0.5], 1);
    },
    seed: 2026,
    copyright:
      "Starlink V3 artist's approximation, procedurally generated for starlink-tracker. SpaceX has published no drawings of V3: " +
      'overall size follows public reporting (~60 m span, six laser terminals); all details are invented. Not an official SpaceX model.',
    extras: {
      provenance:
        "artist's approximation: no official SpaceX drawings of V3 exist; span (~60 m), mass (~2,000 kg) and six laser terminals " +
        'follow public reporting, bus size, antenna layout and wing construction are invented',
      wingColumns: d.columns,
      publicFigures: { massKg: 2000, laserTerminals: 6, laserGbpsEach: 400, downlinkTbps: 1, launchVehicle: 'Starship' },
    },
  });
}
