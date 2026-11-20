/**
 * Round-3 integration helpers: laser-link adjacency (selection panel / beams)
 * and the camera flight's look-point coupling for the 3D close-up fly-in.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { flightLookParam, smootherstep } from '../src/render/cameraMath.ts';
import { buildLinkAdjacency, countLaserEquipped } from '../src/selection/linkAdjacency.ts';

function neighbours(adj: ReturnType<typeof buildLinkAdjacency>, i: number): Array<[number, number]> {
  const out: Array<[number, number]> = [];
  for (let k = adj.offsets[i]; k < adj.offsets[i + 1]; k++) out.push([adj.neighbors[k], adj.kinds[k]]);
  return out;
}

test('link adjacency: CSR lists, in-plane first, invalid pairs skipped, buffers reused', () => {
  // links: 0-1 cross, 1-2 in, 0-2 in, 3-3 self (skip), 2-9 out of range (skip), 0-3 in
  const pairs = new Uint32Array([0, 1, 1, 2, 0, 2, 3, 3, 2, 9, 0, 3]);
  const kinds = new Uint8Array([1, 0, 0, 0, 0, 0]);
  const adj = buildLinkAdjacency(5, pairs, kinds, 1234);
  assert.equal(adj.count, 5);
  assert.equal(adj.links, 4);
  assert.equal(adj.timeMs, 1234);
  assert.equal(adj.version, 1);
  assert.deepEqual(neighbours(adj, 0), [
    [2, 0],
    [3, 0],
    [1, 1],
  ]);
  assert.deepEqual(neighbours(adj, 1), [
    [2, 0],
    [0, 1],
  ]);
  assert.deepEqual(neighbours(adj, 2), [
    [0, 0],
    [1, 0],
  ]);
  assert.deepEqual(neighbours(adj, 3), [[0, 0]]);
  assert.deepEqual(neighbours(adj, 4), []);
  assert.equal(adj.offsets[5], 8);

  // Rebuild into the same object: same buffers, new contents, version bumped.
  const { offsets, neighbors } = adj;
  const again = buildLinkAdjacency(5, new Uint32Array([3, 4]), new Uint8Array([1]), 2000, adj);
  assert.equal(again, adj);
  assert.equal(again.offsets, offsets);
  assert.equal(again.neighbors, neighbors);
  assert.equal(again.version, 2);
  assert.equal(again.links, 1);
  assert.deepEqual(neighbours(again, 0), []);
  assert.deepEqual(neighbours(again, 3), [[4, 1]]);
  assert.deepEqual(neighbours(again, 4), [[3, 1]]);

  // Empty topology.
  const empty = buildLinkAdjacency(3, new Uint32Array(0), new Uint8Array(0), 0);
  assert.deepEqual(Array.from(empty.offsets), [0, 0, 0, 0]);
});

test('link adjacency: per-satellite order depends only on the neighbour set', () => {
  // Same links of satellite 5, in two different worker acceptance orders.
  const a = buildLinkAdjacency(9, new Uint32Array([5, 8, 1, 5, 3, 5, 5, 6]), new Uint8Array([1, 0, 1, 0]), 0);
  const b = buildLinkAdjacency(9, new Uint32Array([3, 5, 5, 6, 5, 8, 1, 5]), new Uint8Array([1, 0, 1, 0]), 0);
  const expected: Array<[number, number]> = [
    [1, 0],
    [6, 0],
    [3, 1],
    [8, 1],
  ];
  assert.deepEqual(neighbours(a, 5), expected);
  assert.deepEqual(neighbours(b, 5), expected);
});

test('link adjacency: 10k satellites / 15k links builds fast', () => {
  const n = 10_000;
  const links = 15_000;
  const pairs = new Uint32Array(links * 2);
  const kinds = new Uint8Array(links);
  for (let k = 0; k < links; k++) {
    const a = (k * 7919) % n;
    const b = (a + 1 + (k % 97)) % n;
    pairs[2 * k] = Math.min(a, b);
    pairs[2 * k + 1] = Math.max(a, b);
    kinds[k] = k & 1;
  }
  let adj = buildLinkAdjacency(n, pairs, kinds, 0);
  const t0 = performance.now();
  for (let r = 0; r < 20; r++) adj = buildLinkAdjacency(n, pairs, kinds, r, adj);
  const ms = (performance.now() - t0) / 20;
  assert.equal(adj.offsets[n], links * 2);
  assert.ok(ms < 5, `rebuild ${ms.toFixed(3)} ms`);
});

test('countLaserEquipped', () => {
  assert.equal(countLaserEquipped(new Uint8Array([0, 3, 4, 0, 2]), 5), 3);
  assert.equal(countLaserEquipped(new Uint8Array([3, 3]), 1), 1);
});

test('flight look param: endpoints, monotonic, plain e for small zooms', () => {
  for (const [ra, rb] of [
    [16_000_000, 120],
    [120, 6_000_000],
    [2_000_000, 1_000_000],
    [120, 120],
  ]) {
    assert.equal(flightLookParam(ra, rb, 0), 0, `${ra}->${rb} @0`);
    assert.ok(Math.abs(flightLookParam(ra, rb, 1) - 1) < 1e-12, `${ra}->${rb} @1`);
    let prev = -1;
    for (let i = 0; i <= 200; i++) {
      const v = flightLookParam(ra, rb, i / 200);
      assert.ok(v >= prev - 1e-12, `${ra}->${rb} monotonic at ${i}`);
      assert.ok(v >= 0 && v <= 1 + 1e-12);
      prev = v;
    }
  }
  assert.equal(flightLookParam(2_000_000, 1_000_000, 0.37), 0.37);
  assert.equal(flightLookParam(120, 120, 0.5), 0.5);
});

test('flight look param: overview (16,000 km) -> 120 m close-up keeps the target near the screen centre', () => {
  const ra = 16_000_000;
  const rb = 120;
  const D = 8_000_000; // look point travels 8,000 km (target far from the overview centre)
  for (let i = 0; i <= 2000; i++) {
    const e = smootherstep(i / 2000);
    const r = Math.exp(Math.log(ra) + (Math.log(rb) - Math.log(ra)) * e);
    const err = (1 - flightLookParam(ra, rb, e)) * D; // look-point distance from the target
    // Target never further off-axis than at the start (D / ra), ...
    assert.ok(err <= r * (D / ra) + 1e-6, `e=${e.toFixed(4)} err=${err} r=${r}`);
    // ...and essentially centred once within 100 km (linear look: ~8,000x worse).
    if (r < 100_000) assert.ok(err < 0.01 * r, `e=${e.toFixed(4)} err=${err} r=${r}`);
  }
});

test('flight look param: release (120 m -> 6,000 km) keeps the satellite centred until the range opens', () => {
  const ra = 120;
  const rb = 6_000_000;
  const D = 550_000; // satellite -> ground point below
  for (let i = 0; i <= 2000; i++) {
    const e = smootherstep(i / 2000);
    const r = Math.exp(Math.log(ra) + (Math.log(rb) - Math.log(ra)) * e);
    const err = flightLookParam(ra, rb, e) * D; // look-point distance from the satellite
    assert.ok(err <= r * (D / rb) + 1e-6, `e=${e.toFixed(4)} err=${err} r=${r}`);
  }
});
