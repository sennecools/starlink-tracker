/**
 * Minimal typings for the Cesium renderer internals used by the custom
 * satellite primitive. These classes ARE exported from the 'cesium' package
 * entry at runtime (see node_modules/cesium/Source/Cesium.js, which re-exports
 * them from @cesium/engine), but they are marked @private and therefore absent
 * from Cesium.d.ts. Only the members actually used here are declared.
 *
 * Verified against cesium 1.147.0 / @cesium/engine 26.5.0.
 */
import type {
  BoundingSphere,
  Camera,
  ComponentDatatype,
  JulianDate,
  Matrix4,
  PrimitiveType,
  SceneMode,
} from 'cesium';

declare module 'cesium' {
  /** Renderer/Context.js (private). Never constructed by app code. */
  export class Context {
    private constructor();
    readonly webgl2: boolean;
  }

  /** Scene/FrameState.js (private). Passed to primitive.update(frameState). */
  export interface FrameState {
    context: Context;
    commandList: DrawCommand[];
    passes: {
      render: boolean;
      pick: boolean;
      pickVoxel: boolean;
      snap: boolean;
      depth: boolean;
      postProcess: boolean;
      offscreen: boolean;
    };
    time: JulianDate;
    mode: SceneMode;
    camera: Camera;
    frameNumber: number;
    pixelRatio: number;
  }

  /** Renderer/BufferUsage.js (private enum object). */
  export const BufferUsage: {
    readonly STREAM_DRAW: number;
    readonly STATIC_DRAW: number;
    readonly DYNAMIC_DRAW: number;
  };

  /** Renderer/Buffer.js (private). */
  export class Buffer {
    private constructor();
    static createVertexBuffer(options: {
      context: Context;
      typedArray?: ArrayBufferView;
      sizeInBytes?: number;
      usage: number;
    }): Buffer;
    readonly sizeInBytes: number;
    vertexArrayDestroyable: boolean;
    copyFromArrayView(arrayView: ArrayBufferView, offsetInBytes?: number): void;
    isDestroyed(): boolean;
    destroy(): undefined;
  }

  export interface VertexArrayAttribute {
    index: number;
    vertexBuffer?: Buffer;
    value?: number[];
    componentsPerAttribute: number;
    componentDatatype?: ComponentDatatype;
    normalize?: boolean;
    offsetInBytes?: number;
    strideInBytes?: number;
    instanceDivisor?: number;
  }

  /** Renderer/VertexArray.js (private). destroy() also destroys its vertex buffers. */
  export class VertexArray {
    constructor(options: { context: Context; attributes: VertexArrayAttribute[]; indexBuffer?: unknown });
    readonly numberOfVertices: number;
    isDestroyed(): boolean;
    destroy(): undefined;
  }

  /** Renderer/ShaderSource.js (private). */
  export class ShaderSource {
    constructor(options?: { sources?: string[]; defines?: string[]; includeBuiltIns?: boolean });
  }

  /** Renderer/ShaderProgram.js (private). */
  export class ShaderProgram {
    private constructor();
    static fromCache(options: {
      context: Context;
      vertexShaderSource: ShaderSource | string;
      fragmentShaderSource: ShaderSource | string;
      attributeLocations?: Record<string, number>;
    }): ShaderProgram;
    isDestroyed(): boolean;
    destroy(): undefined;
  }

  /** Renderer/RenderState.js (private). */
  export class RenderState {
    private constructor();
    static fromCache(renderState?: object): RenderState;
    /** Takes the same plain descriptor object that was passed to fromCache. */
    static removeFromCache(renderState: object): void;
  }

  /** Renderer/Pass.js (private enum object). */
  export const Pass: {
    readonly ENVIRONMENT: number;
    readonly COMPUTE: number;
    readonly GLOBE: number;
    readonly OPAQUE: number;
    readonly TRANSLUCENT: number;
    readonly OVERLAY: number;
  };

  /** Renderer/ContextLimits.js (private). Valid once a context exists. */
  export const ContextLimits: {
    readonly maximumAliasedPointSize: number;
    readonly minimumAliasedPointSize: number;
  };

  /** Renderer/DrawCommand.js (private). */
  export class DrawCommand {
    constructor(options?: {
      boundingVolume?: BoundingSphere;
      modelMatrix?: Matrix4;
      primitiveType?: PrimitiveType;
      vertexArray?: VertexArray;
      count?: number;
      offset?: number;
      shaderProgram?: ShaderProgram;
      uniformMap?: Record<string, () => unknown>;
      renderState?: RenderState;
      pass?: number;
      owner?: object;
      cull?: boolean;
      occlude?: boolean;
      castShadows?: boolean;
      receiveShadows?: boolean;
      pickId?: string;
    });
    boundingVolume: BoundingSphere | undefined;
    count: number | undefined;
    pass: number;
    cull: boolean;
    occlude: boolean;
    shaderProgram: ShaderProgram | undefined;
    renderState: RenderState | undefined;
    vertexArray: VertexArray | undefined;
    uniformMap: Record<string, () => unknown> | undefined;
  }
}
