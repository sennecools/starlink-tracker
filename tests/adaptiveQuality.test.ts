/**
 * AdaptiveQuality controller tests: a fake viewer/scene replays synthetic frame
 * times on a fake performance.now() clock. No WebGL, no Cesium runtime.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import type { Viewer } from 'cesium';
import { AdaptiveQuality } from '../src/render/AdaptiveQuality.ts';

interface Rig {
  quality: AdaptiveQuality;
  /** Render frames for `durationMs`; `frameMs(level)` gives each frame's interval. */
  run(durationMs: number, frameMs: (level: number) => number): void;
  ups: number[];
  downs: number[];
  restore(): void;
}

function rig(): Rig {
  let t = 1_000;
  const realNow = performance.now;
  performance.now = () => t;
  const listeners: Array<() => void> = [];
  const scene = {
    msaaSupported: true,
    msaaSamples: 4,
    postRender: {
      addEventListener: (fn: () => void) => listeners.push(fn),
      removeEventListener: (fn: () => void) => listeners.splice(listeners.indexOf(fn), 1),
    },
  };
  const viewer = { scene, resolutionScale: 1 };
  const ups: number[] = [];
  const downs: number[] = [];
  let prevLevel = 0;
  const quality = new AdaptiveQuality(viewer as unknown as Viewer, {
    onChange: (q) => {
      (q.level < prevLevel ? ups : downs).push(t);
      prevLevel = q.level;
    },
  });
  return {
    quality,
    ups,
    downs,
    run(durationMs, frameMs) {
      const end = t + durationMs;
      while (t < end) {
        t += frameMs(quality.level);
        for (const fn of listeners) fn();
      }
    },
    restore() {
      quality.destroy();
      performance.now = realNow;
    },
  };
}

test('a 30 Hz refresh / browser cap does not drive quality to the bottom', () => {
  const r = rig();
  try {
    r.run(30_000, () => 1000 / 30);
    assert.ok(r.quality.level <= 1, `level ${r.quality.level} after 30 s at a 30 Hz cap`);
    // At most one probing step down, which is then undone.
    assert.ok(r.downs.length <= 1, `${r.downs.length} steps down`);
    r.run(60_000, () => 1000 / 30);
    assert.equal(r.quality.level, 0);
    assert.equal(r.downs.length, 1);
    // Cap lifted: back to normal; real GPU overload is handled again.
    r.run(10_000, () => 1000 / 60);
    assert.equal(r.quality.level, 0);
    r.run(30_000, (level) => (level === 0 ? 40 : 16.7));
    assert.equal(r.quality.level, 1);
  } finally {
    r.restore();
  }
});

test('real fill-rate overload still steps down until the frame time is acceptable', () => {
  const r = rig();
  try {
    // Frame time scales with pixel count: 40 ms at full resolution.
    r.run(60_000, () => 40 * r.quality.resolutionScale ** 2);
    assert.equal(r.quality.resolutionScale, 0.7);
    assert.ok(r.quality.frameMs < 20);
  } finally {
    r.restore();
  }
});

test('failed up-probes back off to the 64 s maximum and stay there', () => {
  const r = rig();
  try {
    // 60 Hz display: level 1 is vsync-locked, level 0 is GPU-bound at 25 ms.
    r.run(15 * 60_000, (level) => (level === 0 ? 25 : 1000 / 60));
    assert.ok(r.ups.length >= 6, `${r.ups.length} probes`);
    const gaps = r.ups.slice(1).map((t, i) => t - r.ups[i]!);
    // Waits double 4 -> 8 -> 16 -> 32 -> 64 s; from then on every probe waits >= 64 s.
    for (const gap of gaps.slice(4)) assert.ok(gap >= 64_000, `probe gap ${Math.round(gap)} ms`);
    // ~13 probes in 15 min; the old 60 s reset made it ~4 per 2 min (~30).
    assert.ok(r.ups.length <= 16, `${r.ups.length} probes in 15 min`);
  } finally {
    r.restore();
  }
});
