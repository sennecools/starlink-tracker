/**
 * Procedural texture samplers for the Starlink models (see texture.ts for
 * the sampler contract). All coordinates are metres on the surface.
 */
import { clamp01, fbm, hash2, lerp, newWorley, smoothstep, valueNoise, worley } from './noise.ts';
import type { Sample, TexSpec } from './texture.ts';

// ---------------------------------------------------------------------------
// Solar array front: monocrystalline cells (pseudo-square, clipped corners),
// silver busbars + interconnect ribbons, cover-glass gloss.
// ---------------------------------------------------------------------------
export interface SolarOpts {
  /** Panel size along u (image width) and v (image height), metres. */
  panelU: number;
  panelV: number;
  cols: number;
  rows: number;
  /** Border between panel edge and the cell field, metres. */
  margin: number;
  width: number;
  height: number;
  seed: number;
}

export function solarCells(o: SolarOpts): TexSpec {
  const pu = (o.panelU - 2 * o.margin) / o.cols;
  const pv = (o.panelV - 2 * o.margin) / o.rows;
  const gap = 0.0035; // cell-to-cell gap
  const corner = 0.016; // pseudo-square corner clip (along each edge)
  const busbars = 3;
  const busW = 0.0026;
  const ribbonInset = 0.012; // ribbon/edge strip position inside the margin
  const ha = (pu - gap) / 2;
  const hb = (pv - gap) / 2;
  return {
    name: 'solar_cells',
    width: o.width,
    height: o.height,
    sizeU: o.panelU,
    sizeV: o.panelV,
    tileable: false,
    ss: 3,
    normalStrength: 1,
    sample(u: number, v: number, s: Sample): void {
      const cu = u - o.margin;
      const cv = v - o.margin;
      const inField = cu >= 0 && cv >= 0 && cu < o.panelU - 2 * o.margin && cv < o.panelV - 2 * o.margin;
      if (!inField) {
        // Panel border: dark substrate with the string bus strips.
        const n = fbm(u * 40, v * 40, 3, 0, 0, o.seed + 5);
        s.r = 0.12 + n * 0.03;
        s.g = 0.125 + n * 0.03;
        s.b = 0.15 + n * 0.03;
        s.rough = 0.6;
        s.metal = 0.1;
        s.h = -0.0002;
        const dEdgeV = Math.min(v, o.panelV - v);
        if (Math.abs(dEdgeV - ribbonInset) < 0.004 && u > o.margin * 0.5 && u < o.panelU - o.margin * 0.5) {
          s.r = 0.7; s.g = 0.71; s.b = 0.74;
          s.metal = 1; s.rough = 0.32; s.h = 0.0002;
        }
        return;
      }
      const i = Math.floor(cu / pu);
      const j = Math.floor(cv / pv);
      const dx = cu - (i + 0.5) * pu;
      const dz = cv - (j + 0.5) * pv;
      const ax = Math.abs(dx);
      const az = Math.abs(dz);
      // Busbars run along v; their ribbons bridge the gap to the next cell.
      let onBus = false;
      for (let k = 0; k < busbars; k++) {
        const bx = -ha + ((k + 0.5) / busbars) * 2 * ha;
        if (Math.abs(dx - bx) < busW / 2) onBus = true;
      }
      const inCell = ax < ha && az < hb && ax - (ha - corner) + (az - (hb - corner)) < corner;
      if (!inCell) {
        if (onBus && ax < ha) {
          s.r = 0.66; s.g = 0.67; s.b = 0.7;
          s.metal = 1; s.rough = 0.35; s.h = 0.0001; s.ao = 0.95;
          return;
        }
        // Gap / clipped corner: light substrate showing through.
        const n = valueNoise(u * 300, v * 300, 0, 0, o.seed + 9);
        s.r = 0.27 + n * 0.03;
        s.g = 0.28 + n * 0.03;
        s.b = 0.31 + n * 0.03;
        s.rough = 0.65;
        s.metal = 0;
        s.h = -0.00035;
        s.ao = 0.85;
        return;
      }
      if (onBus) {
        s.r = 0.7; s.g = 0.71; s.b = 0.74;
        s.metal = 1; s.rough = 0.3; s.h = 0.00025;
        return;
      }
      // Cell body: deep blue-black with per-cell tint/brightness variation.
      const t1 = hash2(i, j, o.seed);
      const t2 = hash2(i, j, o.seed + 31);
      const t3 = hash2(i, j, o.seed + 77);
      const bright = 0.86 + 0.28 * t1;
      const hue = t2 - 0.5; // - purple-ish, + teal-ish
      let r = (0.035 - 0.01 * hue) * bright;
      let g = (0.052 + 0.016 * hue) * bright;
      let b = (0.14 + 0.01 * hue) * bright;
      // Very subtle in-cell texture (anti-reflective coating mottling).
      const m = fbm(u * 60, v * 60, 3, 0, 0, o.seed + 3) - 0.5;
      r += m * 0.012;
      g += m * 0.014;
      b += m * 0.022;
      // Unresolvable fine fingers (perpendicular to the busbars): a faint
      // average sheen plus a soft gradient toward the cell edges.
      const edge = smoothstep(0.75, 1, Math.max(ax / ha, az / hb));
      s.r = r + 0.012 + edge * 0.01;
      s.g = g + 0.013 + edge * 0.01;
      s.b = b + 0.016 + edge * 0.012;
      s.metal = 0.35;
      s.rough = 0.16 + 0.1 * t3 + m * 0.05;
      s.h = m * 0.00004;
    },
  };
}

// ---------------------------------------------------------------------------
// Solar array back side: off-white substrate with honeycomb-panel bond lines.
// Tileable, world-mapped (one tile = `tile` metres).
// ---------------------------------------------------------------------------
export function solarBack(tile: number, size: number, seed: number): TexSpec {
  const period = 8; // noise lattice cells per tile
  return {
    name: 'solar_back',
    width: size,
    height: size,
    sizeU: tile,
    sizeV: tile,
    tileable: true,
    ss: 2,
    sample(u: number, v: number, s: Sample): void {
      const x = (u / tile) * period;
      const y = (v / tile) * period;
      const mott = fbm(x, y, 4, period, period, seed) - 0.5;
      const fine = valueNoise(x * 16, y * 16, period * 16, period * 16, seed + 1) - 0.5;
      const c = 0.78 + mott * 0.06 + fine * 0.015;
      s.r = c;
      s.g = c;
      s.b = c * 0.965;
      s.rough = 0.78 + fine * 0.08;
      s.metal = 0;
      s.h = mott * 0.0004 + fine * 0.00004;
      // Bond lines every quarter tile.
      const q = tile / 4;
      const du = Math.abs(((u % q) + q) % q - q / 2);
      const dv = Math.abs(((v % q) + q) % q - q / 2);
      const line = Math.max(1 - smoothstep(0.0012, 0.0028, q / 2 - du), 1 - smoothstep(0.0012, 0.0028, q / 2 - dv));
      if (line > 0) {
        s.r = lerp(s.r, 0.6, line * 0.5);
        s.g = lerp(s.g, 0.6, line * 0.5);
        s.b = lerp(s.b, 0.58, line * 0.5);
        s.h -= line * 0.0003;
      }
    },
  };
}

// ---------------------------------------------------------------------------
// Multi-layer insulation (crinkled gold Kapton / silver aluminised foil).
// Facets: two scales of Worley cells, each with a random planar tilt, so the
// normal map has creases like crumpled foil. Tileable.
// ---------------------------------------------------------------------------
export function mliFoil(tile: number, size: number, seed: number): TexSpec {
  const wA = newWorley();
  const wB = newWorley();
  const wC = newWorley();
  // Facet lattices (cells per tile, u x v): elongated along u like real
  // wrinkled blankets, three scales.
  const aU = 9, aV = 16;
  const bU = 22, bV = 36;
  const cU = 52, cV = 80;
  return {
    name: 'mli_foil',
    width: size,
    height: size,
    sizeU: tile,
    sizeV: tile,
    tileable: true,
    ss: 2,
    normalStrength: 1,
    sample(u: number, v: number, s: Sample): void {
      const x0 = u / tile;
      const y0 = v / tile;
      // Domain warp so facet edges bend like creased foil.
      const x = x0 + (fbm(x0 * 8, y0 * 8, 3, 8, 8, seed + 21) - 0.5) * 0.03;
      const y = y0 + (fbm(x0 * 8, y0 * 8, 3, 8, 8, seed + 22) - 0.5) * 0.03;
      worley(x * aU, y * aV, aU, aV, seed, wA, 0.95);
      worley(x * bU, y * bV, bU, bV, seed + 17, wB, 0.95);
      worley(x * cU, y * cV, cU, cV, seed + 29, wC, 0.95);
      // Aluminised tape strip across the tile (smoother, silver).
      const tape = 1 - smoothstep(0.022, 0.026, Math.abs(y0 - 0.62));
      const crinkle = 1 - tape * 0.85;
      // Random planar tilt per facet at each scale (slopes, dimensionless).
      s.sx = ((wA.id - 0.5) * 0.3 + (wB.id - 0.5) * 0.18 + (wC.id - 0.5) * 0.1) * crinkle;
      s.sy = ((wA.id2 - 0.5) * 0.3 + (wB.id2 - 0.5) * 0.18 + (wC.id2 - 0.5) * 0.1) * crinkle;
      // Long soft wrinkles (ridged noise) on top.
      const ridge = 1 - Math.abs(2 * fbm(x0 * 3, y0 * 10, 3, 3, 10, seed + 3) - 1);
      s.h = ridge * ridge * 0.003 * crinkle;
      const crease = 1 - smoothstep(0, 0.05, wA.f2 - wA.f1);
      // Gold Kapton with discrete silver (aluminised) sheets.
      const patch = smoothstep(0.585, 0.6, fbm(x0 * 3, y0 * 3, 3, 3, 3, seed + 8));
      const silver = Math.max(patch, tape);
      const bright = 0.94 + 0.08 * wB.id + 0.04 * (wC.id2 - 0.5);
      s.r = clamp01(lerp(0.86, 0.8, silver) * bright);
      s.g = clamp01(lerp(0.68, 0.8, silver) * bright);
      s.b = clamp01(lerp(0.38, 0.82, silver) * bright);
      s.metal = 1;
      s.rough = clamp01(0.14 + 0.1 * wB.id2 + 0.06 * wC.id + crease * 0.1 - tape * 0.06);
      s.ao = 1 - crease * 0.12 * crinkle;
    },
  };
}

// ---------------------------------------------------------------------------
// Bus top/bottom: dark graphite composite panels with seams, fasteners and
// access covers. Fit-mapped to one face of sizeU x sizeV metres.
// ---------------------------------------------------------------------------
export function graphitePanels(sizeU: number, sizeV: number, size: number, seed: number): TexSpec {
  const nU = Math.max(1, Math.round(sizeU / 1.0));
  const nV = Math.max(1, Math.round(sizeV / 0.9));
  const pu = sizeU / nU;
  const pv = sizeV / nV;
  const seamW = 0.006;
  const border = 0.03;
  const boltPitch = 0.09;
  const boltR = 0.0055;
  return {
    name: 'bus_graphite',
    width: size,
    height: size,
    sizeU,
    sizeV,
    tileable: false,
    ss: 3,
    sample(u: number, v: number, s: Sample): void {
      const i = Math.min(nU - 1, Math.floor(u / pu));
      const j = Math.min(nV - 1, Math.floor(v / pv));
      const tone = (hash2(i, j, seed) - 0.5) * 0.025;
      const mott = fbm(u * 3, v * 3, 4, 0, 0, seed + 1) - 0.5;
      // Anisotropic streaks along u (layup direction).
      const streak = valueNoise(u * 6, v * 260, 0, 0, seed + 2) - 0.5;
      let c = 0.185 + tone + mott * 0.03 + streak * 0.007;
      s.r = c;
      s.g = c * 1.02;
      s.b = c * 1.1;
      s.rough = 0.44 + streak * 0.05 + mott * 0.1;
      s.metal = 0.25;
      s.h = streak * 0.00003;
      // Perimeter frame band.
      const dEdge = Math.min(u, v, sizeU - u, sizeV - v);
      if (dEdge < border) {
        c = 0.21 + mott * 0.02;
        s.r = c; s.g = c; s.b = c * 1.05;
        s.rough = 0.5; s.metal = 0.5;
      }
      // Seams between panels (and the frame/panel line).
      const lu = u - i * pu;
      const lv = v - j * pv;
      const panelSeam =
        dEdge > border
          ? Math.min(i > 0 ? lu : Infinity, i < nU - 1 ? pu - lu : Infinity, j > 0 ? lv : Infinity, j < nV - 1 ? pv - lv : Infinity)
          : Infinity;
      const dSeam = Math.min(panelSeam, Math.abs(dEdge - border));
      let seam = 1 - smoothstep(seamW * 0.35, seamW * 0.5, dSeam);
      // Access covers: inset rectangle with its own outline.
      if (hash2(i, j, seed + 5) < 0.55 && dEdge > border) {
        const rw = pu * (0.25 + 0.25 * hash2(i, j, seed + 6));
        const rh = pv * (0.25 + 0.25 * hash2(i, j, seed + 7));
        const rx = (pu - rw) * (0.2 + 0.6 * hash2(i, j, seed + 8));
        const ry = (pv - rh) * (0.2 + 0.6 * hash2(i, j, seed + 9));
        const ox = Math.abs(lu - (rx + rw / 2)) - rw / 2;
        const oy = Math.abs(lv - (ry + rh / 2)) - rh / 2;
        const d = Math.max(ox, oy);
        if (d < 0) {
          const k = 0.012 * (hash2(i, j, seed + 10) - 0.4);
          s.r += k; s.g += k; s.b += k;
        }
        seam = Math.max(seam, 1 - smoothstep(0.0012, 0.0024, Math.abs(d)));
        // Corner screws.
        const cx = Math.abs(lu - (rx + rw / 2)) - (rw / 2 - 0.018);
        const cy = Math.abs(lv - (ry + rh / 2)) - (rh / 2 - 0.018);
        const bd = Math.hypot(cx, cy);
        if (bd < boltR * 0.8) bolt(s, bd / (boltR * 0.8));
      }
      if (seam > 0) {
        s.r = lerp(s.r, 0.05, seam);
        s.g = lerp(s.g, 0.05, seam);
        s.b = lerp(s.b, 0.055, seam);
        s.rough = lerp(s.rough, 0.85, seam);
        s.metal = lerp(s.metal, 0, seam);
        s.h -= 0.0008 * seam;
        s.ao = 1 - 0.4 * seam;
      }
      // Fastener rows along the seams.
      const onU = Math.min(i > 0 ? lu : Infinity, i < nU - 1 ? pu - lu : Infinity);
      const onV = Math.min(j > 0 ? lv : Infinity, j < nV - 1 ? pv - lv : Infinity);
      const off = 0.014;
      const tryBolt = (across: number, along: number): void => {
        const a = Math.abs(across - off);
        const b = Math.abs(((along % boltPitch) + boltPitch) % boltPitch - boltPitch / 2);
        const d = Math.hypot(a, b);
        if (d < boltR) bolt(s, d / boltR);
      };
      if (onU < 0.03 && dEdge > border + 0.008) tryBolt(onU, v);
      if (onV < 0.03 && dEdge > border + 0.008) tryBolt(onV, u);
      if (Math.abs(dEdge - border) < 0.03 && dEdge > border) tryBolt(dEdge - border, dEdge === u || dEdge === sizeU - u ? v : u);
    },
  };
}

function bolt(s: Sample, t: number): void {
  // Domed fastener head, t in [0, 1) from the centre.
  const k = 1 - smoothstep(0.75, 1, t);
  s.r = lerp(s.r, 0.52, k);
  s.g = lerp(s.g, 0.53, k);
  s.b = lerp(s.b, 0.56, k);
  s.metal = lerp(s.metal, 1, k);
  s.rough = lerp(s.rough, 0.35, k);
  s.h += 0.0007 * Math.sqrt(Math.max(0, 1 - t * t));
}

// ---------------------------------------------------------------------------
// Phased-array antenna face: light grey radome, hexagonal lattice of patch
// elements grouped in tiles, aluminium frame. Fit-mapped.
// ---------------------------------------------------------------------------
export function phasedArray(sizeU: number, sizeV: number, size: number, seed: number): TexSpec {
  const frame = 0.028;
  const pitch = 0.02;
  const rowH = (pitch * Math.sqrt(3)) / 2;
  const tileN = 12; // elements per tile edge
  const apothem = pitch * 0.4;
  return {
    name: 'phased_array',
    width: size,
    height: size,
    sizeU,
    sizeV,
    tileable: false,
    ss: 3,
    sample(u: number, v: number, s: Sample): void {
      const dEdge = Math.min(u, v, sizeU - u, sizeV - v);
      const mott = fbm(u * 8, v * 8, 3, 0, 0, seed) - 0.5;
      if (dEdge < frame) {
        const c = 0.66 + mott * 0.03;
        s.r = c; s.g = c * 1.005; s.b = c * 1.02;
        s.metal = 0.85;
        s.rough = 0.38 + mott * 0.1;
        // Bevel toward the radome.
        s.h = 0.0006 * smoothstep(frame, frame * 0.6, dEdge);
        s.ao = 1 - 0.2 * (1 - smoothstep(frame - 0.004, frame, dEdge)) * (dEdge > frame - 0.004 ? 1 : 0);
        return;
      }
      const lu = u - frame;
      const lv = v - frame;
      // Radome base.
      const c = 0.8 + mott * 0.025;
      s.r = c; s.g = c * 1.01; s.b = c * 1.005;
      s.rough = 0.62 + mott * 0.1;
      s.metal = 0;
      // Nearest hex-lattice element.
      const r0 = Math.round(lv / rowH);
      let best = Infinity;
      let bc = 0;
      let br = 0;
      let bdx = 0;
      let bdy = 0;
      for (let r = r0 - 1; r <= r0 + 1; r++) {
        const shift = r & 1 ? pitch / 2 : 0;
        const col = Math.round((lu - shift) / pitch);
        const dx = lu - (col * pitch + shift);
        const dy = lv - r * rowH;
        const d = dx * dx + dy * dy;
        if (d < best) { best = d; bc = col; br = r; bdx = dx; bdy = dy; }
      }
      // Hexagon metric (flat sides left/right), radial distance for the feed.
      const ax = Math.abs(bdx);
      const hexD = Math.max(ax, ax * 0.5 + Math.abs(bdy) * 0.8660254);
      const radial = Math.sqrt(best);
      const tI = Math.floor(bc / tileN);
      const tJ = Math.floor(br / tileN);
      const tileTone = (hash2(tI, tJ, seed + 4) - 0.5) * 0.02;
      s.r += tileTone; s.g += tileTone; s.b += tileTone;
      const k = 1 - smoothstep(apothem - 0.0004, apothem + 0.0004, hexD);
      if (k > 0) {
        const outline = smoothstep(apothem - 0.0016, apothem - 0.0008, hexD);
        const ec = 0.7 - outline * 0.1 + (hash2(bc, br, seed + 2) - 0.5) * 0.02;
        s.r = lerp(s.r, ec, k);
        s.g = lerp(s.g, ec * 1.01, k);
        s.b = lerp(s.b, ec * 1.03, k);
        s.rough = lerp(s.rough, 0.42, k);
        s.metal = lerp(s.metal, 0.35, k);
        s.h += 0.00025 * k;
        // Feed point dot.
        if (radial < apothem * 0.2) { s.r -= 0.1; s.g -= 0.1; s.b -= 0.09; }
      }
      // Tile seams (between groups of tileN x tileN elements).
      const tw = tileN * pitch;
      const th = tileN * rowH;
      const su = Math.abs(((lu + pitch / 2) % tw) - 0);
      const sv = Math.abs(((lv + rowH / 2) % th) - 0);
      const dS = Math.min(su, tw - su, sv, th - sv);
      const seam = 1 - smoothstep(0.0008, 0.0016, dS);
      if (seam > 0) {
        s.r = lerp(s.r, 0.6, seam);
        s.g = lerp(s.g, 0.6, seam);
        s.b = lerp(s.b, 0.62, seam);
        s.h -= 0.0002 * seam;
        s.ao = 1 - 0.15 * seam;
      }
    },
  };
}
