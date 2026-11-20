# Starlink close-up models

| File | Satellite | Wings | Laser terminals | Size |
| --- | --- | --- | --- | --- |
| `starlink-v2mini.glb` | Starlink V2 Mini (2023+, incl. Direct-to-Cell) | 2 (`ArrayPivot_L`, `ArrayPivot_R`) | 3 | ~3.0 MB |
| `starlink-v15.glb` | Starlink v1.0 / v1.5 (2019–2023) | 1 (`ArrayPivot_R`) | 4 (v1.0 had none; harmless) | ~2.6 MB |
| `starlink-v3.glb` | Starlink V3 (Starship, 2026+), **artist's approximation** | 2 (`ArrayPivot_L`, `ArrayPivot_R`) | 6 | ~3.2 MB |

**Credits:** procedurally generated for this project by `scripts/build-models.mjs`
(geometry, PBR textures and the GLB container are all written by
`scripts/lib/glb/*`; no Blender, no third-party assets). Not official SpaceX
models: overall dimensions follow public reporting, and the details (panel
layout, antenna placement, terminal positions) are artistic approximations.

Regenerate (deterministic, ~16 s, validates with the Khronos glTF-Validator):

```sh
node scripts/build-models.mjs            # all models
node scripts/build-models.mjs --only=v2mini   # or --only=v15 / --only=v3
```

`tests/models.test.ts` rebuilds V3 and checks it is byte-identical to the
committed file, so rerun `--only=v3` after any generator change.

## Conventions (the renderer depends on these)

- Units: metres. Core glTF 2.0 only: no extensions, no Draco/meshopt. Textures
  are JPEG embedded in the BIN chunk.
- glTF **Y-up**. **+X = flight direction (velocity)**. **−Y = nadir** (the
  phased-array antennas and gateway dishes face −Y; laser terminals and star
  trackers are on the +Y zenith side). **Wing axis = ±Z**.
- "R" is at **+Z** and "L" at **−Z**: right/left when looking along +X with +Y up.
- The origin is the centre of the bus.

### Node hierarchy

```
Starlink_V2Mini | _V15 | _V3       root, no mesh; extras = metadata (dims, span, area)
├─ Bus                             static mesh: bus, MLI, phased arrays, dishes,
│                                  star trackers, thruster(s), solar-array drive housings
├─ ArrayPivot_L                    (V2 Mini, V3) translation (0, 0, -1.43) / (0, 0, -1.68), identity rotation
│  └─ Wing_L                       wing mesh, rotated 180° about Y (extends toward -Z)
├─ ArrayPivot_R                    translation (0, 0, +1.43) V2 Mini / (0, 0, +1.48) v1.5 / (0, 0, +1.68) V3
│  └─ Wing_R                       wing mesh (extends toward +Z)
└─ LaserTerminal_0..n              origin = centre of the lens dome (beam aperture), +Y up
```

- **Pivots** sit at the solar-array drive output flange on the bus edge, which
  is the wing root. The yoke, hinges and panels are children of the pivot, so
  rotating a pivot about its **local Z** turns the whole wing about the wing
  axis. Pivots have identity rotation in the file.
- **Rest pose:** the cells face **+Y** (zenith). After rotating a pivot by θ
  about +Z, the cell normal is `(−sin θ, cos θ, 0)`. To face the sun direction
  `s` (model frame), use `θ = atan2(−s.x, s.y)`. Use the same θ for both wings.
- **Laser terminals:** one shared mesh, 0.28 m flange, 0.25 m body, dark glossy
  dome of radius 0.09 m. Node translations (metres):
  - V2 Mini: `0 = (1.83, 0.295, 0)` fore, `1 = (−1.83, 0.295, −0.85)` aft-left,
    `2 = (−1.83, 0.295, 0.85)` aft-right.
  - v1.5: `(±0.52, 0.27, −1.2)` and `(±0.52, 0.27, 1.08)` (indices 0..3 =
    +x/−z, −x/−z, +x/+z, −x/+z).
  - V3 (nodes scaled ×1.25, so the dome radius is 0.1125 m):
    `0 = (3.4, 0.4125, 0.9)` fore-right, `1 = (3.4, 0.4125, −0.9)` fore-left,
    `2 = (−3.4, 0.4125, 0.9)` aft-right, `3 = (−3.4, 0.4125, −0.9)` aft-left,
    `4 = (0.9, 0.4125, 1.3)` right side, `5 = (−0.9, 0.4125, −1.3)` left side.

  Beams should start at the node's world position (`LaserTerminal_k`).
  Terminal placement is illustrative. SpaceX does not publish it, and the
  inter-satellite link topology drawn in the app is a model too.

### Dimensions (rest pose, axis-aligned bounds)

| | X (velocity) | Y (up) | Z (wing axis) |
| --- | --- | --- | --- |
| V2 Mini, overall | 4.15 | 0.73 | 30.0 (tip to tip) |
| V2 Mini bus | 4.1 | 0.25 | 2.7 |
| V2 Mini wing (each) | 4.1 | 0.03 | 12.8 (6 panels, 5 cm gaps); ~103 m² total |
| v1.5, overall | 2.8 | 0.65 | 11.48 (from −1.40 to +10.08) |
| v1.5 bus | 1.4 | 0.2 | 2.8 |
| v1.5 wing | 2.8 | 0.03 | 8.1 (6 panels, 4 cm gaps) |
| V3, overall | 7.57 | 1.19 | 60.2 (tip to tip, incl. tip fittings) |
| V3 bus | 7.5 | 0.4 | 3.2 |
| V3 wing (each) | 4.9 (2 columns of 2.35 + 0.2 spine gap) | 0.03 | 27.0 (22 panels per column, 5 cm gaps); ~244 m² total |

## Starlink V3: artist's approximation

SpaceX has published **no drawings** of Starlink V3. `starlink-v3.glb` says so
in `asset.copyright` and in the root node's `extras.provenance`. What is public,
and what the model follows: launched by Starship (Flight 14, 2026-09-28; Flight 15,
2026-11-13), ~2,000 kg, six 400 Gbps laser terminals, ~1 Tbps downlink,
"about the size of a Boeing 737", and one secondhand report of a ~60 m deployed span.
Everything else is invented so that it reads as bigger and more capable than V2 Mini:

- **Bus:** a 7.5 × 0.4 × 3.2 m flat pack (Starship dispenser stacking), with
  two stacking rails and six pads on the zenith face, a white radiator on the fore
  face and two Hall/argon thrusters on the aft face.
- **Nadir face:** eight phased-array panels (1.42 × 1.44 m each, 4 × 2 grid) and
  four two-axis **gimballed gateway dishes** (Ka/E/V/W-band; radius 0.27 m;
  turntable, fork, elevation axle, feed on four struts) at the fore and aft ends.
- **Zenith face:** six laser terminals (see above), three star trackers, two GNSS patches.
- **Wings:** larger drives (hub hardware ×1.6), a 1.3 m yoke, then **two columns**
  of 22 panels either side of a carbon spine, with cross-ties at every panel gap
  and a tip fitting. Span = 2 × (1.6 + 0.08 + 1.3 + 27) ≈ 60 m.

Same node names, axes, materials and texture pipeline as the other models, so
the renderer needs nothing V3-specific beyond the file name (`BUS_VERSIONS`).

## Loading in CesiumJS

By default, Cesium's `Model` assumes glTF's +Z is forward and rotates it onto +X. These files
already use +X as forward, so load them with:

```js
const model = await Cesium.Model.fromGltfAsync({
  url: '/models/starlink-v2mini.glb',
  forwardAxis: Cesium.Axis.X, // upAxis stays Axis.Y (default)
});
```

The model frame is then X = velocity, Z = zenith (away from Earth) and Y = left
(the R wing is at −Y). Build `modelMatrix` from an orbital frame with columns
`[v̂, ẑ × v̂, ẑ]`, where ẑ = r̂ points radially out and v̂ is the velocity made
orthogonal to r̂.

Sun tracking (node transforms are in glTF space, before the axis correction):

```js
const pivot = model.getNode('ArrayPivot_R');
pivot.matrix = Cesium.Matrix4.multiply(
  pivot.originalMatrix,
  Cesium.Matrix4.fromRotation(Cesium.Matrix3.fromRotationZ(theta)),
  new Cesium.Matrix4(),
);
```

Here `theta = atan2(−s.x, s.y)`, with `s` = the sun direction in glTF model space.
In Cesium model coordinates (X, Y, Z) that is
`s_gltf = (X, Z, −Y)`. Do the same for `ArrayPivot_L`. Cache and reuse the
matrices so nothing is allocated per frame.

## Materials

| Material | Maps | Notes |
| --- | --- | --- |
| `SolarCells` | base, ORM, normal 2048×1024 | deep blue-black pseudo-square cells, silver busbars and ribbons, per-cell tint; one texture per panel, mirrored on alternate panels |
| `SolarArrayBack` | base, normal 512² (tiling, 1 m) | off-white substrate with bond lines |
| `MLI_Foil` | base, ORM, normal 1024² (tiling, 0.5 m) | crinkled gold Kapton with silver aluminised sheets and tape (bus sides) |
| `BusGraphite` | base, ORM, normal 1024² | dark graphite panels with seams, fasteners and access covers (bus top and bottom) |
| `PhasedArray` | base, ORM, normal 1024² | light-grey radome with a hexagonal element lattice in tiles, aluminium frame |
| `Aluminium`, `DarkAnodised`, `CarbonTube`, `WhitePaint`, `LaserLens`, `BlackBaffle` | factors only | |
| `DishReflector` | factors only | double-sided, used only for the thin dish shells |

The ORM maps are packed the glTF way: R = occlusion, G = roughness, B = metallic.
Normal maps are in tangent space with +Y up (OpenGL style), as glTF requires, and every
normal-mapped primitive ships `TANGENT`s.
