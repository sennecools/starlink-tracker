/**
 * StarField — the Hipparcos catalogue (~41k stars to V = 8) as crisp GPU
 * points at infinity, in ONE GL_POINTS draw.
 *
 * - Same direction-only TEME transform as the sky box (skyGlsl.ts), and the
 *   star vectors share the cube's baked precession epoch, so stars sit
 *   exactly on the Milky Way texture.
 * - Photometry: relative flux 10^(-0.4 (m - mRef)). Faint stars use a fixed,
 *   tight Gaussian PSF (sigma0, >= 0.5 device px) with peak ∝ flux, so the
 *   V 6-8 field reads as fine dust; brighter stars saturate the core and grow
 *   slowly (sigma ∝ flux^w, capped), giving 3-5 px bright stars with a tight
 *   soft core. The peak is display-encoded (^1/2.2). No twinkle.
 * - Colour: B-V -> Ballesteros temperature -> blackbody (CIE 1931) -> linear
 *   sRGB, desaturated, encoded (256-entry LUT built once at load).
 * - Sizes in CSS px scaled by czm_pixelRatio (devicePixelRatio *
 *   resolutionScale), so they look the same on every screen.
 *
 * Draw order: Scene.renderEnvironment() executes scene.skyBox's command first,
 * then the sky atmosphere, sun, moon; the globe and everything else come in
 * later passes. Pass.ENVIRONMENT commands from scene.primitives are NOT
 * executed there (only frameState.panoramaCommandList, which runs AFTER the
 * moon and atmosphere). So this class is not added to scene.primitives: its
 * command is drawn by SkyEnvironment immediately after the cube inside the
 * sky box's own command — before atmosphere/sun/moon/globe, which therefore
 * cover the stars correctly. Additive blending, no depth test/write.
 */
import {
  BlendEquation,
  BlendFunction,
  Buffer,
  BufferUsage,
  Cartesian4,
  ComponentDatatype,
  ContextLimits,
  DrawCommand,
  Pass,
  PrimitiveType,
  RenderState,
  SceneMode,
  ShaderProgram,
  ShaderSource,
  VertexArray,
} from 'cesium';
import type { Context, FrameState } from 'cesium';
import { GALACTIC_CENTER_J2000, POLARIS_J2000, j2000ToSkyVec } from './skyMath.ts';
import { STAR_VERTEX_FLOATS, buildStarVertices } from './skyAssets.ts';
import { MARKER_ATTRIBUTES, MARKER_FS, MARKER_VS, STAR_ATTRIBUTES, STAR_FS, STAR_VS } from './skyGlsl.ts';

export interface StarFieldStyle {
  /** Magnitude whose (unsaturated) peak is exactly 1. Higher = brighter field. */
  magRef: number;
  /** PSF sigma of faint stars, CSS px. */
  sigma0Px: number;
  /** Largest PSF sigma (brightest stars), CSS px. */
  sigmaMaxPx: number;
  /** sigma = sigma0 * flux^exponent for saturated stars. */
  sigmaExponent: number;
  /** Overall intensity multiplier. */
  brightness: number;
  /** Stars fainter than this are not drawn (faded over the last 0.75 mag). */
  limitMag: number;
  /** 0 = white stars, 1 = full blackbody colour. */
  saturation: number;
}

export const DEFAULT_STAR_STYLE: Readonly<StarFieldStyle> = {
  magRef: 3.2,
  sigma0Px: 0.55,
  sigmaMaxPx: 2.0,
  sigmaExponent: 0.35,
  brightness: 1,
  limitMag: 8,
  saturation: 0.55,
};

export interface StarFieldOptions {
  /** stars.bin contents: Float32 × STAR_STRIDE per star (x, y, z, mag, bv), bright -> faint. */
  data: Float32Array;
  /** Baked epoch of the sky frame (manifest), used for the debug markers. */
  epoch: number;
  style?: Partial<StarFieldStyle>;
}

const DISPLAY_GAMMA = 1 / 2.2;

const ADDITIVE_KEEP_ALPHA = {
  depthTest: { enabled: false },
  depthMask: false,
  blending: {
    enabled: true,
    equationRgb: BlendEquation.ADD,
    equationAlpha: BlendEquation.ADD,
    functionSourceRgb: BlendFunction.ONE,
    functionDestinationRgb: BlendFunction.ONE,
    functionSourceAlpha: BlendFunction.ZERO,
    functionDestinationAlpha: BlendFunction.ONE,
  },
} as const;

export class StarField {
  /** Draw the stars (default true). */
  show = true;
  /** Debug: draw rings at the galactic centre (magenta) and Polaris (cyan). */
  showMarkers = false;
  /** 0..1 multiplier, e.g. for a fade-in. */
  fade = 1;

  readonly count: number;
  private readonly style: StarFieldStyle;
  private vertices: Float32Array | undefined;
  private readonly epoch: number;

  private vertexArray: VertexArray | undefined;
  private shaderProgram: ShaderProgram | undefined;
  private useHdr: boolean | undefined;
  private renderState: RenderState | undefined;
  private command: DrawCommand | undefined;

  private markerVertexArray: VertexArray | undefined;
  private markerProgram: ShaderProgram | undefined;
  private markerCommand: DrawCommand | undefined;

  private readonly uStarA = new Cartesian4();
  private readonly uStarB = new Cartesian4();
  private uMaxPointSize = 64;
  private markerSize = 34;
  /** RenderState.fromCache() references taken (released on destroy). */
  private renderStateRefs = 0;
  private destroyed = false;

  constructor(opts: StarFieldOptions) {
    this.style = { ...DEFAULT_STAR_STYLE, ...opts.style };
    this.epoch = opts.epoch;
    this.vertices = buildStarVertices(opts.data, this.style.saturation);
    this.count = this.vertices.length / STAR_VERTEX_FLOATS;
  }

  setStyle(style: Partial<StarFieldStyle>): void {
    Object.assign(this.style, style);
  }

  /**
   * Prepares GPU resources and returns the star command (and the marker
   * command when enabled) for this frame, or undefined when nothing draws.
   */
  update(frameState: FrameState, useHdr: boolean): DrawCommand | undefined {
    if (this.destroyed || !this.show || this.count === 0) return undefined;
    if (frameState.mode !== SceneMode.SCENE3D && frameState.mode !== SceneMode.MORPHING) return undefined;
    if (!this.command) this.createResources(frameState.context);
    if (!this.shaderProgram || this.useHdr !== useHdr) {
      this.shaderProgram?.destroy();
      this.shaderProgram = ShaderProgram.fromCache({
        context: frameState.context,
        vertexShaderSource: new ShaderSource({ sources: [STAR_VS] }),
        fragmentShaderSource: new ShaderSource({ defines: useHdr ? ['HDR'] : [], sources: [STAR_FS] }),
        attributeLocations: { ...STAR_ATTRIBUTES },
      });
      this.useHdr = useHdr;
      (this.command as DrawCommand).shaderProgram = this.shaderProgram;
    }
    const s = this.style;
    this.uStarA.x = s.magRef;
    this.uStarA.y = s.sigma0Px;
    this.uStarA.z = Math.max(s.sigma0Px, s.sigmaMaxPx);
    this.uStarA.w = s.sigmaExponent;
    this.uStarB.x = s.brightness * Math.max(0, Math.min(1, this.fade));
    this.uStarB.y = DISPLAY_GAMMA;
    this.uStarB.z = this.uMaxPointSize;
    this.uStarB.w = s.limitMag;
    return this.command;
  }

  /** The debug marker command, if enabled (call after update()). */
  markerCommandFor(frameState: FrameState): DrawCommand | undefined {
    if (this.destroyed || !this.showMarkers) return undefined;
    if (!this.markerCommand) this.createMarkerResources(frameState.context);
    this.markerSize = 34 * frameState.pixelRatio;
    return this.markerCommand;
  }

  private createResources(context: Context): void {
    const vertices = this.vertices as Float32Array;
    const buffer = Buffer.createVertexBuffer({ context, typedArray: vertices, usage: BufferUsage.STATIC_DRAW });
    const stride = STAR_VERTEX_FLOATS * 4;
    this.vertexArray = new VertexArray({
      context,
      attributes: [
        { index: STAR_ATTRIBUTES.a_dir, vertexBuffer: buffer, componentsPerAttribute: 3, componentDatatype: ComponentDatatype.FLOAT, offsetInBytes: 0, strideInBytes: stride },
        { index: STAR_ATTRIBUTES.a_colorMag, vertexBuffer: buffer, componentsPerAttribute: 4, componentDatatype: ComponentDatatype.FLOAT, offsetInBytes: 12, strideInBytes: stride },
      ],
    });
    this.vertices = undefined; // uploaded; let the CPU copy go
    this.renderState = RenderState.fromCache(ADDITIVE_KEEP_ALPHA);
    this.renderStateRefs++;
    const maxPoint = ContextLimits.maximumAliasedPointSize;
    this.uMaxPointSize = maxPoint > 1 ? Math.min(maxPoint, 63) : 63;
    this.command = new DrawCommand({
      primitiveType: PrimitiveType.POINTS,
      vertexArray: this.vertexArray,
      count: this.count,
      renderState: this.renderState,
      uniformMap: {
        u_starA: () => this.uStarA,
        u_starB: () => this.uStarB,
      },
      pass: Pass.ENVIRONMENT,
      owner: this,
      cull: false,
      occlude: false,
      castShadows: false,
      receiveShadows: false,
    });
  }

  private createMarkerResources(context: Context): void {
    const gc = j2000ToSkyVec(GALACTIC_CENTER_J2000.raDeg, GALACTIC_CENTER_J2000.decDeg, this.epoch);
    const po = j2000ToSkyVec(POLARIS_J2000.raDeg, POLARIS_J2000.decDeg, this.epoch);
    const v = new Float32Array([gc.x, gc.y, gc.z, 1.0, 0.2, 1.0, po.x, po.y, po.z, 0.2, 1.0, 1.0]);
    const buffer = Buffer.createVertexBuffer({ context, typedArray: v, usage: BufferUsage.STATIC_DRAW });
    this.markerVertexArray = new VertexArray({
      context,
      attributes: [
        { index: MARKER_ATTRIBUTES.a_dir, vertexBuffer: buffer, componentsPerAttribute: 3, componentDatatype: ComponentDatatype.FLOAT, offsetInBytes: 0, strideInBytes: 24 },
        { index: MARKER_ATTRIBUTES.a_color, vertexBuffer: buffer, componentsPerAttribute: 3, componentDatatype: ComponentDatatype.FLOAT, offsetInBytes: 12, strideInBytes: 24 },
      ],
    });
    this.markerProgram = ShaderProgram.fromCache({
      context,
      vertexShaderSource: new ShaderSource({ sources: [MARKER_VS] }),
      fragmentShaderSource: new ShaderSource({ sources: [MARKER_FS] }),
      attributeLocations: { ...MARKER_ATTRIBUTES },
    });
    this.markerCommand = new DrawCommand({
      primitiveType: PrimitiveType.POINTS,
      vertexArray: this.markerVertexArray,
      count: 2,
      shaderProgram: this.markerProgram,
      renderState: this.renderState ?? RenderState.fromCache(ADDITIVE_KEEP_ALPHA),
      uniformMap: { u_markerSize: () => Math.min(this.markerSize, this.uMaxPointSize) },
      pass: Pass.ENVIRONMENT,
      owner: this,
      cull: false,
      occlude: false,
    });
  }

  isDestroyed(): boolean {
    return this.destroyed;
  }

  destroy(): undefined {
    if (this.destroyed) return undefined;
    this.destroyed = true;
    if (this.vertexArray && !this.vertexArray.isDestroyed()) this.vertexArray.destroy();
    if (this.shaderProgram && !this.shaderProgram.isDestroyed()) this.shaderProgram.destroy();
    if (this.markerVertexArray && !this.markerVertexArray.isDestroyed()) this.markerVertexArray.destroy();
    if (this.markerProgram && !this.markerProgram.isDestroyed()) this.markerProgram.destroy();
    this.vertexArray = this.markerVertexArray = undefined;
    this.shaderProgram = this.markerProgram = undefined;
    this.command = this.markerCommand = undefined;
    this.vertices = undefined;
    for (; this.renderStateRefs > 0; this.renderStateRefs--) RenderState.removeFromCache(ADDITIVE_KEEP_ALPHA);
    this.renderState = undefined;
    return undefined;
  }
}
