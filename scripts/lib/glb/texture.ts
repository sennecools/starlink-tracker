/**
 * Procedural PBR texture rasteriser. A texture set is described by one
 * "material sampler" evaluated in physical units (metres across the image);
 * the rasteriser supersamples it and produces three RGB8 maps:
 *
 *   base   - baseColor, sRGB
 *   orm    - R = ambient occlusion, G = roughness, B = metallic (glTF packing)
 *   normal - tangent-space normal, glTF convention (+X = +u, +Y = image up)
 *
 * The normal comes from the supersampled height field (central differences,
 * physical slopes) plus an optional analytic slope (sx, sy) for features that
 * are discontinuous in height (crinkled foil facets).
 */
import sharp from 'sharp';

export interface Sample {
  /** sRGB base colour 0..1. */
  r: number;
  g: number;
  b: number;
  ao: number;
  rough: number;
  metal: number;
  /** Height in metres (positive = towards the viewer). */
  h: number;
  /** Extra slope dh/du and dh/dv (v downwards), dimensionless. */
  sx: number;
  sy: number;
}

export interface TexSpec {
  name: string;
  width: number;
  height: number;
  /** Physical extent of the image (metres) along u (width) and v (height). */
  sizeU: number;
  sizeV: number;
  /** True for tileable textures (wrap the height-field derivatives). */
  tileable: boolean;
  /** Supersampling factor per axis. */
  ss?: number;
  /** Multiplier on all slopes (normal-map strength). */
  normalStrength?: number;
  /** u, v in metres from the image's top-left corner. */
  sample(u: number, v: number, out: Sample): void;
}

export interface TexMaps {
  name: string;
  width: number;
  height: number;
  base: Uint8Array;
  orm: Uint8Array;
  normal: Uint8Array;
}

const to8 = (v: number): number => (v <= 0 ? 0 : v >= 1 ? 255 : Math.round(v * 255));

export function rasterize(spec: TexSpec): TexMaps {
  const { width: W, height: H } = spec;
  const ss = spec.ss ?? 2;
  const n = W * H;
  const base = new Uint8Array(n * 3);
  const orm = new Uint8Array(n * 3);
  const normal = new Uint8Array(n * 3);
  const height = new Float32Array(n);
  const slopeX = new Float32Array(n);
  const slopeY = new Float32Array(n);
  const s: Sample = { r: 0, g: 0, b: 0, ao: 1, rough: 0.5, metal: 0, h: 0, sx: 0, sy: 0 };
  const du = spec.sizeU / W;
  const dv = spec.sizeV / H;
  const inv = 1 / (ss * ss);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let r = 0, g = 0, b = 0, ao = 0, ro = 0, me = 0, h = 0, sx = 0, sy = 0;
      for (let j = 0; j < ss; j++) {
        for (let i = 0; i < ss; i++) {
          s.r = s.g = s.b = 0;
          s.ao = 1;
          s.rough = 0.5;
          s.metal = 0;
          s.h = 0;
          s.sx = 0;
          s.sy = 0;
          spec.sample((x + (i + 0.5) / ss) * du, (y + (j + 0.5) / ss) * dv, s);
          r += s.r; g += s.g; b += s.b;
          ao += s.ao; ro += s.rough; me += s.metal;
          h += s.h; sx += s.sx; sy += s.sy;
        }
      }
      const p = y * W + x;
      base[p * 3] = to8(r * inv);
      base[p * 3 + 1] = to8(g * inv);
      base[p * 3 + 2] = to8(b * inv);
      orm[p * 3] = to8(ao * inv);
      orm[p * 3 + 1] = to8(ro * inv);
      orm[p * 3 + 2] = to8(me * inv);
      height[p] = h * inv;
      slopeX[p] = sx * inv;
      slopeY[p] = sy * inv;
    }
  }
  const k = spec.normalStrength ?? 1;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let xl = x - 1, xr = x + 1, yu = y - 1, yd = y + 1;
      let sxScale = 2, syScale = 2;
      if (spec.tileable) {
        xl = (xl + W) % W; xr %= W; yu = (yu + H) % H; yd %= H;
      } else {
        if (xl < 0) { xl = 0; sxScale = 1; }
        if (xr >= W) { xr = W - 1; sxScale = 1; }
        if (yu < 0) { yu = 0; syScale = 1; }
        if (yd >= H) { yd = H - 1; syScale = 1; }
      }
      const p = y * W + x;
      const gx = ((height[y * W + xr] - height[y * W + xl]) / (sxScale * du) + slopeX[p]) * k;
      const gy = ((height[yd * W + x] - height[yu * W + x]) / (syScale * dv) + slopeY[p]) * k;
      // n = (-dh/du, +dh/dv_down, 1): +Y of the normal map points image-up.
      const l = Math.sqrt(gx * gx + gy * gy + 1);
      normal[p * 3] = to8((-gx / l) * 0.5 + 0.5);
      normal[p * 3 + 1] = to8((gy / l) * 0.5 + 0.5);
      normal[p * 3 + 2] = to8((1 / l) * 0.5 + 0.5);
    }
  }
  return { name: spec.name, width: W, height: H, base, orm, normal };
}

export type ImageFormat = { kind: 'jpeg'; quality: number } | { kind: 'png' };

/** Encodes an RGB8 image; 4:4:4 chroma for JPEG (data maps must not bleed channels). */
export async function encodeRgb(rgb: Uint8Array, width: number, height: number, fmt: ImageFormat): Promise<Uint8Array> {
  const img = sharp(Buffer.from(rgb.buffer, rgb.byteOffset, rgb.byteLength), { raw: { width, height, channels: 3 } });
  const buf =
    fmt.kind === 'jpeg'
      ? await img.jpeg({ quality: fmt.quality, chromaSubsampling: '4:4:4', mozjpeg: true }).toBuffer()
      : await img.png({ compressionLevel: 9, adaptiveFiltering: true }).toBuffer();
  return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
}

/** Linear-light helpers for colour mixing in sRGB-encoded samples. */
export function srgb(r: number, g: number, b: number): [number, number, number] {
  return [r / 255, g / 255, b / 255];
}
