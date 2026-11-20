/**
 * Typed access to the few Cesium renderer internals the sky needs that are
 * NOT declared in src/render/cesium-private.d.ts (CubeMap, Sampler,
 * VertexArray.fromGeometry, Context#draw, DrawCommand#execute).
 *
 * They are exported from the 'cesium' entry at runtime (cesium/index.js:
 * `export { default as CubeMap } ...`, `Sampler`, `VertexArray`) but marked
 * @private, so they are reached through the namespace object with a local
 * structural type instead of a global module augmentation (which could clash
 * with declarations other modules add to cesium-private.d.ts).
 *
 * Verified against cesium 1.147.0 / @cesium/engine 26.5.0.
 */
import * as Cesium from 'cesium';
import type { Context, DrawCommand, Geometry, VertexArray } from 'cesium';

export type FaceSource = ImageBitmap | HTMLImageElement | HTMLCanvasElement | { width: number; height: number; arrayBufferView: ArrayBufferView };

export interface CubeMapFaceLike {
  copyFrom(options: { source: FaceSource; xOffset?: number; yOffset?: number; skipColorSpaceConversion?: boolean }): void;
}

export interface CubeMapLike {
  readonly width: number;
  readonly sizeInBytes: number;
  readonly positiveX: CubeMapFaceLike;
  readonly negativeX: CubeMapFaceLike;
  readonly positiveY: CubeMapFaceLike;
  readonly negativeY: CubeMapFaceLike;
  readonly positiveZ: CubeMapFaceLike;
  readonly negativeZ: CubeMapFaceLike;
  sampler: unknown;
  generateMipmap(hint?: number): void;
  isDestroyed(): boolean;
  destroy(): undefined;
}

interface CubeMapCtor {
  new (options: {
    context: Context;
    width?: number;
    height?: number;
    source?: Record<string, FaceSource>;
    flipY?: boolean;
    skipColorSpaceConversion?: boolean;
    sampler?: unknown;
    preMultiplyAlpha?: boolean;
  }): CubeMapLike;
}

interface SamplerCtor {
  new (options?: { minificationFilter?: number; magnificationFilter?: number; maximumAnisotropy?: number }): unknown;
}

interface VertexArrayStatics {
  fromGeometry(options: { context: Context; geometry: Geometry; attributeLocations: Record<string, number>; bufferUsage?: number }): VertexArray;
}

interface CesiumPrivate {
  CubeMap: CubeMapCtor;
  Sampler: SamplerCtor;
  VertexArray: VertexArrayStatics;
}

const priv = Cesium as unknown as CesiumPrivate;

export const CubeMap: CubeMapCtor = priv.CubeMap;
export const Sampler: SamplerCtor = priv.Sampler;

export function vertexArrayFromGeometry(options: Parameters<VertexArrayStatics['fromGeometry']>[0]): VertexArray {
  return priv.VertexArray.fromGeometry(options);
}

/** Context#draw(drawCommand, passState) (private API, stable since 1.0). */
export function contextDraw(context: Context, command: DrawCommand, passState: unknown): void {
  (context as unknown as { draw(c: DrawCommand, p: unknown): void }).draw(command, passState);
}

/**
 * Replaces DrawCommand#execute on one instance. Scene.executeCommand ends
 * with `command.execute(context, passState)`; overriding it lets one command
 * returned from scene.skyBox.update() draw several things in order.
 */
export function overrideExecute(command: DrawCommand, execute: (context: Context, passState: unknown) => void): void {
  (command as unknown as { execute: (context: Context, passState: unknown) => void }).execute = execute;
}
