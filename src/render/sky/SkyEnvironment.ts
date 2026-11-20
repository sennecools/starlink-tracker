/**
 * SkyEnvironment — the object assigned to `scene.skyBox`.
 *
 * Cesium only needs a sky box to implement `update(frameState, useHdr)`
 * returning ONE DrawCommand (Scene.updateEnvironment), `show`, `isDestroyed()`
 * and `destroy()`. Scene.renderEnvironment() executes that command FIRST —
 * before the sky atmosphere, sun, moon and every globe/primitive pass — via
 * `command.execute(context, passState)`. SkyCube's command overrides execute
 * to draw the Milky Way cube and then (afterDraw) the StarField and the debug
 * markers with the same pass state, which is the only place where draws land
 * between the sky box and everything else without patching Cesium.
 *
 * Until our first cube map is uploaded, update() delegates to the previous
 * sky box (Cesium's default Tycho box), so the sky never flashes black; on the
 * frame our cube appears the previous sky box is destroyed.
 */
import type { DrawCommand, FrameState } from 'cesium';
import { SceneMode } from 'cesium';
import { contextDraw } from './cesiumInternals.ts';
import { SkyCube } from './SkyCube.ts';
import type { StarField } from './StarField.ts';

/** What Scene expects from scene.skyBox. */
export interface SkyBoxLike {
  show: boolean;
  update(frameState: FrameState, useHdr: boolean): DrawCommand | undefined;
  isDestroyed(): boolean;
  destroy(): unknown;
}

export class SkyEnvironment implements SkyBoxLike {
  show = true;
  readonly cube = new SkyCube();
  stars: StarField | null = null;

  private fallback: SkyBoxLike | undefined;
  private starCommand: DrawCommand | undefined;
  private markerCommand: DrawCommand | undefined;
  private destroyed = false;

  constructor(fallback?: SkyBoxLike) {
    this.fallback = fallback;
    this.cube.afterDraw = (context, passState) => {
      if (this.starCommand) contextDraw(context, this.starCommand, passState);
      if (this.markerCommand) contextDraw(context, this.markerCommand, passState);
    };
  }

  update(frameState: FrameState, useHdr: boolean): DrawCommand | undefined {
    if (this.destroyed || !this.show) return undefined;
    if (frameState.mode !== SceneMode.SCENE3D && frameState.mode !== SceneMode.MORPHING) return undefined;
    if (!frameState.passes.render) return undefined;

    const command = this.cube.update(frameState, useHdr);
    if (!command) {
      // our cube is not uploaded yet: keep the previous sky
      const fb = this.fallback;
      return fb && !fb.isDestroyed() ? fb.update(frameState, useHdr) : undefined;
    }
    if (this.fallback) {
      if (!this.fallback.isDestroyed()) this.fallback.destroy();
      this.fallback = undefined;
    }
    const stars = this.stars;
    this.starCommand = stars ? stars.update(frameState, useHdr) : undefined;
    this.markerCommand = stars ? stars.markerCommandFor(frameState) : undefined;
    return command;
  }

  isDestroyed(): boolean {
    return this.destroyed;
  }

  destroy(): undefined {
    if (this.destroyed) return undefined;
    this.destroyed = true;
    this.cube.destroy();
    this.stars?.destroy();
    this.stars = null;
    if (this.fallback && !this.fallback.isDestroyed()) this.fallback.destroy();
    this.fallback = undefined;
    this.starCommand = this.markerCommand = undefined;
    return undefined;
  }
}
