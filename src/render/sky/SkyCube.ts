/**
 * SkyCube — the Milky Way cube map drawn exactly like Cesium's SkyBox
 * (same TEME direction lookup, same face orientation conventions), plus what
 * Cesium's SkyBox lacks:
 *
 * - Mipmaps + trilinear filtering: a 4096² face is minified ~2x on a 1440p
 *   screen; without mips (Cesium's SkyBox uses plain LINEAR) the sky shimmers
 *   while the camera turns.
 * - Atomic, non-blocking swaps: a new set of six decoded faces is uploaded
 *   into a NEW cube map, at most one face per frame (a 4096² texSubImage2D is
 *   ~10-30 ms), then mipmapped, and only then replaces the visible cube. The
 *   old cube keeps drawing in the meantime.
 * - Debug face tinting (orientation checks).
 *
 * Not a scene primitive on its own: SkyEnvironment (assigned to
 * scene.skyBox) calls update() and executes the returned command.
 */
import {
  BlendingState,
  BoxGeometry,
  BufferUsage,
  Cartesian3,
  Cartesian4,
  DrawCommand,
  GeometryPipeline,
  Pass,
  RenderState,
  ShaderProgram,
  ShaderSource,
  TextureMagnificationFilter,
  TextureMinificationFilter,
  VertexFormat,
} from 'cesium';
import type { Context, FrameState, VertexArray } from 'cesium';
import { CUBE_FACE_SOURCE_KEYS } from './skyMath.ts';
import { CubeMap, Sampler, contextDraw, overrideExecute, vertexArrayFromGeometry } from './cesiumInternals.ts';
import type { CubeMapFaceLike, CubeMapLike, FaceSource } from './cesiumInternals.ts';
import { CUBE_FS, CUBE_VS } from './skyGlsl.ts';

const CUBE_RENDER_STATE = { depthTest: { enabled: false }, depthMask: false, blending: BlendingState.ALPHA_BLEND };

interface PendingUpload {
  images: FaceSource[];
  size: number;
  cube: CubeMapLike | undefined;
  /** Next face index to upload; 6 = all uploaded, mipmaps pending. */
  next: number;
  resolve: () => void;
  reject: (e: unknown) => void;
  cancelled: boolean;
}

function faceOf(cube: CubeMapLike, i: number): CubeMapFaceLike {
  return cube[CUBE_FACE_SOURCE_KEYS[i]];
}

function closeImage(img: FaceSource): void {
  if (typeof ImageBitmap !== 'undefined' && img instanceof ImageBitmap) img.close();
}

export class SkyCube {
  /** Called right after the cube was drawn (same pass state); SkyEnvironment draws the stars here. */
  afterDraw: ((context: Context, passState: unknown) => void) | null = null;

  private cube: CubeMapLike | undefined;
  private pending: PendingUpload | undefined;
  private command: DrawCommand | undefined;
  private vertexArray: VertexArray | undefined;
  private shaderProgram: ShaderProgram | undefined;
  private useHdr: boolean | undefined;
  private renderState: RenderState | undefined;
  private cubeAttributeLocations: Record<string, number> = {};
  private readonly params = new Cartesian4(1, 0, 1, 0);
  private readonly uniformMap: Record<string, () => unknown>;
  private destroyed = false;

  constructor() {
    this.uniformMap = {
      u_cubeMap: () => this.cube,
      u_cubeParams: () => this.params,
    };
  }

  /** True once a cube map is showing. */
  get hasCube(): boolean {
    return this.cube !== undefined;
  }

  get size(): number {
    return this.cube ? this.cube.width : 0;
  }

  /** Overall brightness multiplier (default 1). */
  set brightness(v: number) {
    this.params.x = v;
  }

  /** Debug: tint each face with its own colour (0 = off .. 1 = solid). */
  set faceTint(v: number) {
    this.params.y = v;
  }

  /**
   * Queues six decoded faces (positiveX, negativeX, positiveY, negativeY,
   * positiveZ, negativeZ order), all `size`², decoded with flipY (as
   * Cesium's loadCubeMap does). Resolves on the frame the new cube becomes
   * visible. A newer call cancels a pending one (its promise rejects).
   */
  setFaces(images: FaceSource[], size: number): Promise<void> {
    if (this.destroyed) return Promise.reject(new Error('SkyCube destroyed'));
    this.cancelPending('superseded');
    return new Promise<void>((resolve, reject) => {
      this.pending = { images, size, cube: undefined, next: 0, resolve, reject, cancelled: false };
    });
  }

  /** Drops a queued / in-progress upload (its setFaces promise rejects); the visible cube is kept. */
  cancel(reason = 'cancelled'): void {
    this.cancelPending(reason);
  }

  private cancelPending(reason: string): void {
    const p = this.pending;
    if (!p) return;
    p.cancelled = true;
    for (let i = p.next; i < p.images.length; i++) closeImage(p.images[i]);
    if (p.cube && !p.cube.isDestroyed()) p.cube.destroy();
    this.pending = undefined;
    p.reject(new Error(`sky cube upload ${reason}`));
  }

  /** Advances a pending upload by one step. Never more than one face per frame. */
  private pump(context: Context): void {
    const p = this.pending;
    if (!p) return;
    try {
      if (!p.cube) {
        p.cube = new CubeMap({
          context,
          width: p.size,
          height: p.size,
          // Same single vertical flip as Cesium's sky box path: images are
          // decoded with imageOrientation 'flipY' (ImageBitmap; the
          // UNPACK_FLIP_Y_WEBGL flag is ignored for bitmaps) or flipped by
          // UNPACK_FLIP_Y_WEBGL (HTMLImageElement fallback).
          flipY: true,
          skipColorSpaceConversion: true,
          sampler: new Sampler({
            minificationFilter: TextureMinificationFilter.LINEAR,
            magnificationFilter: TextureMagnificationFilter.LINEAR,
          }),
        });
        return; // allocation is this frame's step
      }
      if (p.next < 6) {
        const img = p.images[p.next];
        faceOf(p.cube, p.next).copyFrom({ source: img, skipColorSpaceConversion: true });
        closeImage(img);
        p.next++;
        return;
      }
      // all faces uploaded: build the mip chain and swap
      p.cube.generateMipmap();
      p.cube.sampler = new Sampler({
        minificationFilter: TextureMinificationFilter.LINEAR_MIPMAP_LINEAR,
        magnificationFilter: TextureMagnificationFilter.LINEAR,
      });
      const old = this.cube;
      this.cube = p.cube;
      this.pending = undefined;
      if (old && !old.isDestroyed()) old.destroy();
      p.resolve();
    } catch (e) {
      this.cancelPending(`failed: ${String(e)}`);
      console.error('[sky] cube map upload failed', e);
    }
  }

  update(frameState: FrameState, useHdr: boolean): DrawCommand | undefined {
    if (this.destroyed) return undefined;
    const context = frameState.context;
    this.pump(context);
    if (!this.cube) return undefined;

    if (!this.command) {
      const geometry = BoxGeometry.createGeometry(
        BoxGeometry.fromDimensions({ dimensions: new Cartesian3(2, 2, 2), vertexFormat: VertexFormat.POSITION_ONLY }),
      );
      if (!geometry) return undefined;
      const attributeLocations = GeometryPipeline.createAttributeLocations(geometry) as Record<string, number>;
      this.vertexArray = vertexArrayFromGeometry({ context, geometry, attributeLocations, bufferUsage: BufferUsage.STATIC_DRAW });
      this.renderState = RenderState.fromCache(CUBE_RENDER_STATE);
      const command = new DrawCommand({
        vertexArray: this.vertexArray,
        renderState: this.renderState,
        uniformMap: this.uniformMap,
        pass: Pass.ENVIRONMENT,
        owner: this,
        cull: false,
        occlude: false,
        castShadows: false,
        receiveShadows: false,
      });
      overrideExecute(command, (ctx, passState) => {
        if (command.shaderProgram) contextDraw(ctx, command, passState);
        this.afterDraw?.(ctx, passState);
      });
      this.command = command;
      this.cubeAttributeLocations = attributeLocations;
    }

    if (!this.shaderProgram || this.useHdr !== useHdr) {
      this.shaderProgram?.destroy();
      this.shaderProgram = ShaderProgram.fromCache({
        context,
        vertexShaderSource: new ShaderSource({ sources: [CUBE_VS] }),
        fragmentShaderSource: new ShaderSource({ defines: useHdr ? ['HDR'] : [], sources: [CUBE_FS] }),
        attributeLocations: this.cubeAttributeLocations,
      });
      this.useHdr = useHdr;
      this.command.shaderProgram = this.shaderProgram;
    }
    return this.command;
  }

  isDestroyed(): boolean {
    return this.destroyed;
  }

  destroy(): undefined {
    if (this.destroyed) return undefined;
    this.cancelPending('destroyed');
    this.destroyed = true;
    if (this.cube && !this.cube.isDestroyed()) this.cube.destroy();
    this.cube = undefined;
    if (this.vertexArray && !this.vertexArray.isDestroyed()) this.vertexArray.destroy();
    this.vertexArray = undefined;
    if (this.shaderProgram && !this.shaderProgram.isDestroyed()) this.shaderProgram.destroy();
    this.shaderProgram = undefined;
    if (this.renderState) RenderState.removeFromCache(CUBE_RENDER_STATE);
    this.renderState = undefined;
    this.command = undefined;
    this.afterDraw = null;
    return undefined;
  }
}
