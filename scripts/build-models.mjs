#!/usr/bin/env node
/**
 * build-models — writes the procedurally generated Starlink close-up models:
 *
 *   public/models/starlink-v2mini.glb   Starlink V2 Mini (2023+), two wings
 *   public/models/starlink-v15.glb      Starlink v1.0 / v1.5 (2019-2023), one wing
 *   public/models/starlink-v3.glb       Starlink V3 (Starship, 2026+), artist's approximation
 *
 *   node scripts/build-models.mjs [--only=v2mini|v15|v3] [--out=public/models] [--no-validate]
 *
 * No Blender: geometry, PBR textures (sharp-encoded JPEG embedded in the BIN
 * chunk) and the GLB container are all written by scripts/lib/glb/*. Output is
 * deterministic (seeded noise). Each file is validated with the Khronos
 * glTF-Validator (dev dependency `gltf-validator`); errors fail the build.
 * Set MODELS_VERBOSE=1 for per-texture timings.
 *
 * Conventions (relied on by the renderer; see public/models/README.md):
 * metres, glTF Y-up, +X = velocity, -Y = nadir, wing axis ±Z; nodes "Bus",
 * "ArrayPivot_L"/"ArrayPivot_R", "LaserTerminal_<n>".
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildV15, buildV2Mini, buildV3 } from './lib/glb/starlink.ts';
import { parseGlb, sceneBounds } from './lib/glb/parse.ts';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const args = Object.fromEntries(
  process.argv.slice(2).map((a) => {
    const [k, v] = a.replace(/^--/, '').split('=');
    return [k, v ?? true];
  }),
);
const outDir = resolve(ROOT, typeof args.out === 'string' ? args.out : 'public/models');
const MAX_BYTES = 6 * 1024 * 1024;

const MODELS = [
  { id: 'v2mini', file: 'starlink-v2mini.glb', build: buildV2Mini },
  { id: 'v15', file: 'starlink-v15.glb', build: buildV15 },
  { id: 'v3', file: 'starlink-v3.glb', build: buildV3 },
];

async function validate(bytes, file) {
  const validator = (await import('gltf-validator')).default;
  const report = await validator.validateBytes(bytes, { uri: file, maxIssues: 200, format: 'glb' });
  return report;
}

const fmtKB = (n) => `${(n / 1024).toFixed(0)} KB`;

mkdirSync(outDir, { recursive: true });
if (args.only && !MODELS.some((m) => m.id === args.only)) {
  console.error(`unknown --only=${args.only} (expected ${MODELS.map((m) => m.id).join('|')})`);
  process.exit(2);
}
let failed = false;
for (const m of MODELS) {
  if (args.only && args.only !== m.id) continue;
  const t0 = performance.now();
  const { glb, textures } = await m.build();
  const path = join(outDir, m.file);
  writeFileSync(path, glb);
  const { json } = parseGlb(glb);
  const b = sceneBounds(json);
  const tris = json.meshes.reduce(
    (s, mesh) => s + mesh.primitives.reduce((t, p) => t + json.accessors[p.indices].count / 3, 0),
    0,
  );
  console.log(
    `${m.file}: ${fmtKB(glb.byteLength)}, ${json.nodes.length} nodes, ${json.meshes.length} meshes, ` +
      `${tris} triangles (unique), bbox ${b.size.map((v) => v.toFixed(2)).join(' x ')} m, ` +
      `${((performance.now() - t0) / 1000).toFixed(1)} s`,
  );
  for (const t of textures) console.log(`  ${t.name.padEnd(22)} ${t.width}x${t.height} ${t.mime.padEnd(10)} ${fmtKB(t.bytes)}`);
  if (glb.byteLength > MAX_BYTES) {
    console.error(`  ERROR: ${m.file} exceeds ${fmtKB(MAX_BYTES)}`);
    failed = true;
  }
  if (!args['no-validate']) {
    const r = await validate(glb, m.file);
    console.log(
      `  glTF-Validator ${r.validatorVersion}: ${r.issues.numErrors} errors, ${r.issues.numWarnings} warnings, ` +
        `${r.issues.numInfos} infos, ${r.issues.numHints} hints`,
    );
    for (const msg of r.issues.messages) {
      if (msg.severity <= 1 || process.env.MODELS_VERBOSE) console.log(`    [${msg.severity}] ${msg.code} ${msg.pointer ?? ''} ${msg.message}`);
    }
    if (r.issues.numErrors > 0) failed = true;
  }
}
if (failed) process.exit(1);
