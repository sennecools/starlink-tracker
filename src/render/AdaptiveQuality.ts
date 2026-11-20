/**
 * AdaptiveQuality — dynamic resolution / MSAA scaling to keep the globe smooth.
 *
 * Frame time is measured between consecutive scene.postRender events (wall
 * clock, performance.now) and smoothed with an EMA. Quality moves along a
 * fixed ladder:
 *
 *   res 1.0 msaa 4 -> 0.9 -> 0.8 -> 0.7 -> 0.6 -> 0.5 (msaa 4) -> res 0.5 msaa 1
 *
 * - Step DOWN when the smoothed frame time stays above `slowFrameMs` (20 ms)
 *   for `downAfterMs` (1.5 s).
 * - Step UP when frames stay "good" for the current up-wait (starts at 4 s).
 *   Good = EMA below `fastFrameMs` (12 ms) OR locked to the display refresh
 *   (fastest EMA seen <= 17.2 ms, EMA within 12 % of it, and < 2 % of frames
 *   missing a vsync). The second rule is needed on 60 Hz displays, where
 *   vsync makes < 12 ms impossible to observe; it cannot tell how much GPU
 *   headroom is left, hence the probing + back-off below.
 * - Hysteresis: if a step up is followed by a step down within 10 s, the
 *   up-wait doubles (up to 64 s), so a level that cannot be held is probed at
 *   most about once a minute. After ~3 min without a step down the up-wait
 *   resets.
 * - Frame-rate caps: a step down that does not lower the frame time by at
 *   least 10 % is undone, and that frame time is remembered as a floor (a
 *   display refresh or browser throttle such as a 30 Hz cap, not GPU load).
 *   Frame times within 10 % of the floor do not count as overload; the floor
 *   is forgotten once frames get clearly faster again.
 * - Frames longer than 250 ms (tab hidden, debugger, GC hitch) are ignored and
 *   measurements settle for 1 s after every level change (framebuffers are
 *   reallocated on resize / MSAA change), and for 3 s after start.
 *
 * Nothing is allocated per frame.
 */
import type { Scene, Viewer } from 'cesium';

export interface AdaptiveQualityOptions {
  /** Frame time (ms) above which quality is reduced. Default 20. */
  slowFrameMs?: number;
  /** Frame time (ms) below which quality may increase. Default 12. */
  fastFrameMs?: number;
  /** Sustained slow time before stepping down. Default 1500 ms. */
  downAfterMs?: number;
  /** Initial sustained good time before stepping up. Default 4000 ms. */
  upAfterMs?: number;
  /** Lowest resolution scale. Default 0.5. */
  minResolutionScale?: number;
  /** Highest resolution scale. Default 1.0. */
  maxResolutionScale?: number;
  /** MSAA samples at full quality. Default 4 (ignored if MSAA unsupported). */
  maxMsaa?: number;
  /** Called after every level change. */
  onChange?: (quality: AdaptiveQuality) => void;
}

interface Level {
  readonly resolutionScale: number;
  readonly msaa: number;
}

const EMA_ALPHA = 0.08;
const MAX_VALID_FRAME_MS = 250;
const SETTLE_AFTER_CHANGE_MS = 1000;
const SETTLE_AFTER_START_MS = 3000;
const FAILED_PROBE_WINDOW_MS = 10_000;
const MAX_UP_WAIT_MS = 64_000;
/** No step down for this long resets the up-wait. Longer than any up-wait, so a waiting probe fires first. */
const STABLE_RESET_MS = 3 * MAX_UP_WAIT_MS;
/** A step down must lower the smoothed frame time by at least this fraction, or it is undone. */
const MIN_STEP_GAIN = 0.1;
/** Above floor x this, frame time counts as overload again. */
const FLOOR_SLOW_FACTOR = 1.1;
/** Below floor x this, the cap is gone. */
const FLOOR_CLEAR_FACTOR = 0.8;
const VSYNC_TOLERANCE = 1.12;
const MAX_VSYNC_INTERVAL_MS = 17.2;
const MISS_RATE_ALPHA = 0.02;
const MAX_MISS_RATE = 0.02;

export class AdaptiveQuality {
  private readonly viewer: Viewer;
  private readonly scene: Scene;
  private readonly levels: Level[];
  private readonly slowMs: number;
  private readonly fastMs: number;
  private readonly downAfterMs: number;
  private readonly baseUpWaitMs: number;
  private readonly onChange: ((q: AdaptiveQuality) => void) | undefined;
  private readonly removeListener: () => void;

  private levelIndex = 0;
  private lastFrameAt = -1;
  private ema = 16.7;
  private fastestEma = Number.POSITIVE_INFINITY;
  /** EMA of the fraction of frames that missed a vsync (dt > 1.5 x refresh). */
  private missRate = 0.1;
  private badMs = 0;
  private goodMs = 0;
  private settleUntil: number;
  private upWaitMs: number;
  private lastUpAt = Number.NEGATIVE_INFINITY;
  /** Last real step down (failed probes included); drives the up-wait reset. */
  private lastDownAt = Number.NEGATIVE_INFINITY;
  /** Last step down whose effect is still to be checked, and the EMA just before it. */
  private verifyDownAt = Number.NEGATIVE_INFINITY;
  private emaBeforeDown = 0;
  /** Frame time that lowering quality could not improve (refresh or browser cap); 0 = none. */
  private floorMs = 0;
  private destroyed = false;

  /** When false, measurements continue but the level is frozen. */
  enabled = true;

  constructor(viewer: Viewer, opts: AdaptiveQualityOptions = {}) {
    this.viewer = viewer;
    this.scene = viewer.scene;
    this.slowMs = opts.slowFrameMs ?? 20;
    this.fastMs = opts.fastFrameMs ?? 12;
    this.downAfterMs = opts.downAfterMs ?? 1500;
    this.baseUpWaitMs = opts.upAfterMs ?? 4000;
    this.upWaitMs = this.baseUpWaitMs;
    this.onChange = opts.onChange;

    const minRes = clamp(opts.minResolutionScale ?? 0.5, 0.1, 1);
    const maxRes = clamp(opts.maxResolutionScale ?? 1, minRes, 2);
    const maxMsaa = this.scene.msaaSupported ? Math.max(1, Math.round(opts.maxMsaa ?? 4)) : 1;
    const levels: Level[] = [];
    // Integer tenths avoid float drift (0.7999999...).
    for (let r = Math.round(maxRes * 10); r >= Math.round(minRes * 10); r--) {
      levels.push({ resolutionScale: r / 10, msaa: maxMsaa });
    }
    if (levels.length === 0) levels.push({ resolutionScale: maxRes, msaa: maxMsaa });
    if (maxMsaa > 1) levels.push({ resolutionScale: levels[levels.length - 1].resolutionScale, msaa: 1 });
    this.levels = levels;

    const now = performance.now();
    this.settleUntil = now + SETTLE_AFTER_START_MS;
    this.apply();

    const listener = this.onPostRender;
    this.scene.postRender.addEventListener(listener);
    this.removeListener = () => this.scene.postRender.removeEventListener(listener);
  }

  /** Smoothed frames per second. */
  get fps(): number {
    return this.ema > 0 ? 1000 / this.ema : 0;
  }

  /** Smoothed frame time in ms. */
  get frameMs(): number {
    return this.ema;
  }

  get resolutionScale(): number {
    return this.levels[this.levelIndex].resolutionScale;
  }

  get msaa(): number {
    return this.levels[this.levelIndex].msaa;
  }

  /** 0 = best quality; higher = more degraded. */
  get level(): number {
    return this.levelIndex;
  }

  get levelCount(): number {
    return this.levels.length;
  }

  /** Forces a level (e.g. a user "quality" setting); keeps adapting if enabled. */
  setLevel(index: number): void {
    const i = clamp(Math.round(index), 0, this.levels.length - 1);
    if (i === this.levelIndex) return;
    this.levelIndex = i;
    this.changed(performance.now());
  }

  // Arrow function: a single bound listener created once.
  private readonly onPostRender = (): void => {
    const now = performance.now();
    const prev = this.lastFrameAt;
    this.lastFrameAt = now;
    if (prev < 0) return;
    const dt = now - prev;
    if (dt <= 0 || dt > MAX_VALID_FRAME_MS) {
      this.badMs = 0;
      this.goodMs = 0;
      return;
    }
    this.ema += (dt - this.ema) * EMA_ALPHA;
    if (now < this.settleUntil) return;

    // Track the fastest sustained frame time = display refresh interval when
    // vsync-bound. Relax it slowly so a monitor change is picked up.
    if (this.ema < this.fastestEma) this.fastestEma = this.ema;
    else this.fastestEma += (this.ema - this.fastestEma) * 0.0005;
    const missed = dt > this.fastestEma * 1.5 ? 1 : 0;
    this.missRate += (missed - this.missRate) * MISS_RATE_ALPHA;

    if (!this.enabled) return;

    if (this.ema < this.floorMs * FLOOR_CLEAR_FACTOR) this.floorMs = 0; // cap lifted
    if (this.ema > this.slowMs && this.ema > this.floorMs * FLOOR_SLOW_FACTOR) {
      this.badMs += dt;
      this.goodMs = 0;
      if (this.badMs >= this.downAfterMs) this.stepDown(now);
      return;
    }
    this.badMs = 0;

    // "Locked to vsync" only counts for plausible display intervals (>= ~58 Hz);
    // a steady 18 ms GPU-bound frame time must not look like headroom.
    const vsyncLocked =
      this.fastestEma <= MAX_VSYNC_INTERVAL_MS &&
      this.ema < this.fastestEma * VSYNC_TOLERANCE &&
      this.missRate < MAX_MISS_RATE;
    const good = this.ema < this.fastMs || vsyncLocked;
    if (good) {
      this.goodMs += dt;
      if (this.goodMs >= this.upWaitMs) this.stepUp(now);
    } else {
      this.goodMs = 0;
    }

    if (now - this.lastDownAt > STABLE_RESET_MS && this.upWaitMs !== this.baseUpWaitMs) {
      this.upWaitMs = this.baseUpWaitMs;
      this.goodMs = 0;
    }
  };

  private stepDown(now: number): void {
    this.badMs = 0;
    this.goodMs = 0;
    // The previous step down did not lower the frame time by >= 10 %: the frame
    // interval comes from a refresh or browser cap, not from GPU load. Undo it and stop.
    if (
      now - this.verifyDownAt < SETTLE_AFTER_CHANGE_MS + this.downAfterMs + 500 &&
      this.ema > this.emaBeforeDown * (1 - MIN_STEP_GAIN)
    ) {
      this.floorMs = this.ema;
      this.verifyDownAt = Number.NEGATIVE_INFINITY;
      if (this.levelIndex > 0) {
        this.levelIndex--; // revert the useless step
        this.changed(now);
      }
      return;
    }
    if (this.levelIndex >= this.levels.length - 1) return;
    if (now - this.lastUpAt < FAILED_PROBE_WINDOW_MS) {
      // The last step up did not hold: be more patient next time.
      this.upWaitMs = Math.min(this.upWaitMs * 2, MAX_UP_WAIT_MS);
    }
    this.emaBeforeDown = this.ema;
    this.verifyDownAt = now;
    this.lastDownAt = now;
    this.levelIndex++;
    this.changed(now);
  }

  private stepUp(now: number): void {
    this.badMs = 0;
    this.goodMs = 0;
    if (this.levelIndex === 0) return;
    this.levelIndex--;
    this.lastUpAt = now;
    this.changed(now);
  }

  private changed(now: number): void {
    this.badMs = 0;
    this.goodMs = 0;
    this.settleUntil = now + SETTLE_AFTER_CHANGE_MS;
    // The refresh estimate is re-learned at the new level.
    this.fastestEma = Number.POSITIVE_INFINITY;
    this.missRate = 0.1;
    this.apply();
    if (this.onChange) {
      try {
        this.onChange(this);
      } catch (err) {
        console.warn('[AdaptiveQuality] onChange threw', err);
      }
    }
  }

  private apply(): void {
    if (this.destroyed) return;
    const lvl = this.levels[this.levelIndex];
    if (this.viewer.resolutionScale !== lvl.resolutionScale) this.viewer.resolutionScale = lvl.resolutionScale;
    if (this.scene.msaaSupported && this.scene.msaaSamples !== lvl.msaa) this.scene.msaaSamples = lvl.msaa;
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.removeListener();
  }
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}
