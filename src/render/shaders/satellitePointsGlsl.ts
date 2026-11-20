/**
 * GLSL for SatellitePoints. Written in GLSL ES 3.00 style ('in'/'out',
 * 'out_FragColor') exactly like Cesium 1.147's own shaders; Cesium's
 * ShaderSource prepends '#version 300 es', precision, czm_ builtins and the
 * 'out_FragColor' declaration (and demodernizes for a WebGL1 fallback).
 *
 * Log depth: the VS calls czm_vertexLogDepth() and the FS calls
 * czm_writeLogDepth(); Cesium's DerivedCommand.createLogDepthCommand detects
 * these calls (regex /\s+czm_vertexLogDepth\(/ and /\s+czm_writeLogDepth\(/,
 * hence the indentation) and only adds the LOG_DEPTH define, so the same
 * source works with and without the logarithmic depth buffer.
 */

/** Attribute locations (must match the VertexArray in SatellitePoints.ts). */
export const ATTRIBUTE_LOCATIONS = {
  a_posHighA: 0,
  a_posLowA: 1,
  a_posHighB: 2,
  a_posLowB: 3,
  a_colorShell: 4,
  a_index: 5,
} as const;

export const SATELLITE_POINTS_VS = /* glsl */ `
in vec3 a_posHighA;
in vec3 a_posLowA;
in vec3 a_posHighB;
in vec3 a_posLowB;
// rgb: sRGB colour 0..255, a: shell index (unsigned bytes, not normalized)
in vec4 a_colorShell;
// satellite index (exact as float up to 2^24)
in float a_index;

// Interpolation factor between keyframe slot A (0) and slot B (1); may be
// slightly outside [0, 1] for clamped extrapolation.
uniform float u_t;
// Shell visibility flags (1 = visible) for shells 0..3 and 4..7.
uniform vec4 u_shellVis0;
uniform vec4 u_shellVis1;
// x: size constant k (CSS px * sqrt(m)), y: min core px, z: max core px,
// w: sprite diameter / core diameter (glow extent)
uniform vec4 u_size;
// x: selected index (-1 none), y: hovered index (-1 none), z: brightness,
// w: horizon ellipsoid inverse-scale (>= 1 shrinks the occluding ellipsoid)
uniform vec4 u_state;
uniform vec4 u_selectedColor;
uniform vec4 u_hoveredColor;
uniform float u_maxPointSize;

out vec3 v_color;
out float v_intensity;
out float v_glowScale;
out float v_emphasis;

void main()
{
    float wA = 1.0 - u_t;
    float wB = u_t;
    bool validA = dot(a_posHighA, a_posHighA) > 0.0;
    bool validB = dot(a_posHighB, a_posHighB) > 0.0;
    bool valid = (validA || wA == 0.0) && (validB || wB == 0.0);

    // Branch-free lookup of the shell visibility flag (no dynamic indexing,
    // so it also compiles if Cesium falls back to WebGL1 / GLSL ES 1.00).
    float shell = a_colorShell.w;
    vec4 m0 = step(abs(vec4(0.0, 1.0, 2.0, 3.0) - shell), vec4(0.5));
    vec4 m1 = step(abs(vec4(4.0, 5.0, 6.0, 7.0) - shell), vec4(0.5));
    float vis = dot(m0, u_shellVis0) + dot(m1, u_shellVis1);
    bool selected = abs(a_index - u_state.x) < 0.5;
    bool hovered = abs(a_index - u_state.y) < 0.5;
    if (selected) {
        vis = 1.0;
    }

    // Eye-relative positions (EncodedCartesian3 high/low emulated double precision).
    vec3 relA = (a_posHighA - czm_encodedCameraPositionMCHigh) + (a_posLowA - czm_encodedCameraPositionMCLow);
    vec3 relB = (a_posHighB - czm_encodedCameraPositionMCHigh) + (a_posLowB - czm_encodedCameraPositionMCLow);
    vec3 rel = relA * wA + relB * wB;

    // Analytic horizon test: is the segment eye -> point blocked by the
    // (slightly shrunk) WGS84 ellipsoid? Done in unit-sphere space.
    bool occluded = false;
    vec3 invRadii = czm_ellipsoidInverseRadii * u_state.w;
    vec3 e = czm_viewerPositionWC * invRadii;
    vec3 d = rel * invRadii;
    float c = dot(e, e) - 1.0;
    if (c > 0.0) {
        float a = dot(d, d);
        float b = dot(e, d);
        float disc = b * b - a * c;
        if (disc > 0.0 && a > 0.0) {
            float tHit = (-b - sqrt(disc)) / a;
            occluded = tHit > 0.0 && tHit < 1.0;
        }
    }

    float dist = length(rel);
    float corePx = clamp(u_size.x / sqrt(max(dist, 1.0)), u_size.y, u_size.z);
    float emphasis = 0.0;
    vec3 color = a_colorShell.rgb / 255.0;
    if (hovered) {
        corePx *= 1.7;
        emphasis = 0.6;
        color = mix(color, u_hoveredColor.rgb, 0.65);
    }
    if (selected) {
        corePx *= 2.4;
        emphasis = 1.0;
        color = u_selectedColor.rgb;
    }

    gl_Position = czm_modelViewProjectionRelativeToEye * vec4(rel, 1.0);
    czm_vertexLogDepth();

    float spritePx = min(corePx * u_size.w * czm_pixelRatio, u_maxPointSize);
    gl_PointSize = spritePx;

    v_color = color;
    v_intensity = u_state.z;
    // Sprite radius expressed in core radii (keeps the core size exact even
    // when the sprite was clamped to the max point size).
    v_glowScale = spritePx / max(corePx * czm_pixelRatio, 1e-3);
    v_emphasis = emphasis;

    if (!valid || vis < 0.5 || occluded) {
        // Outside the clip volume: points are discarded by clipping.
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        gl_PointSize = 1.0;
        v_intensity = 0.0;
    }
}
`;

export const SATELLITE_POINTS_FS = /* glsl */ `
in vec3 v_color;
in float v_intensity;
in float v_glowScale;
in float v_emphasis;

void main()
{
    // -1..1 across the sprite, r = 1 at the sprite edge.
    vec2 p = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(p, p);
    if (r2 >= 1.0 || v_intensity <= 0.0) {
        discard;
    }

    // Distance in units of the core radius.
    float rc = sqrt(r2) * v_glowScale;
    // Solid core disc with an anti-aliased edge (~1px wide at typical sizes).
    float core = 1.0 - smoothstep(0.6, 1.15, rc);
    // White-hot centre.
    float hot = exp(-rc * rc * 4.0);
    // Soft glow halo, forced to zero at the sprite edge.
    float halo = exp(-rc * rc * 0.55) * (1.0 - r2) * (0.30 + 0.25 * v_emphasis);

    vec3 base = czm_gammaCorrect(v_color);
    vec3 rgb = base * (core * 0.95 + halo) + vec3(hot * (0.45 + 0.4 * v_emphasis));
    rgb *= v_intensity;

    // Additive blending (ONE, ONE): alpha only matters for the alpha channel.
    out_FragColor = vec4(rgb, clamp(core + halo, 0.0, 1.0));
    czm_writeLogDepth();
}
`;
