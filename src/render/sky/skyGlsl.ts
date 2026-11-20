/**
 * GLSL for the sky (GLSL ES 3.00 in Cesium style: 'in'/'out', 'out_FragColor';
 * Cesium's ShaderSource adds the version, precision and czm_* built-ins).
 *
 * Every sky draw uses the transform of Cesium's SkyBoxVS:
 *   czm_viewRotation * (czm_temeToPseudoFixed * (R * direction))
 * i.e. direction-only (camera position ignored => at infinity), with the
 * direction in TEME axes. R is HALF the far-plane distance instead of SkyBoxVS'
 * full distance: a cube corner / star then never lies beyond the far plane
 * (|eye z| <= 0.87 far), so nothing can be clipped there. Depth test and
 * depth writes are off for all sky commands.
 */

/** Attribute locations of the star vertex array. */
export const STAR_ATTRIBUTES = { a_dir: 0, a_colorMag: 1 } as const;
/** Attribute locations of the debug marker vertex array. */
export const MARKER_ATTRIBUTES = { a_dir: 0, a_color: 1 } as const;

export const CUBE_VS = /* glsl */ `
in vec3 position;
out vec3 v_dir;

void main()
{
    vec3 p = czm_viewRotation * (czm_temeToPseudoFixed * ((0.5 * czm_entireFrustum.y) * position));
    gl_Position = czm_projection * vec4(p, 1.0);
    v_dir = position;
}
`;

export const CUBE_FS = /* glsl */ `
uniform samplerCube u_cubeMap;
// x: brightness, y: debug face-tint amount (0 = off), z: opacity
uniform vec4 u_cubeParams;
in vec3 v_dir;

vec3 faceTint(vec3 d)
{
    vec3 a = abs(d);
    if (a.x >= a.y && a.x >= a.z) return d.x > 0.0 ? vec3(1.0, 0.15, 0.15) : vec3(0.15, 1.0, 1.0);
    if (a.y >= a.z) return d.y > 0.0 ? vec3(0.15, 1.0, 0.15) : vec3(1.0, 0.15, 1.0);
    return d.z > 0.0 ? vec3(0.25, 0.35, 1.0) : vec3(1.0, 1.0, 0.15);
}

void main()
{
    vec3 d = normalize(v_dir);
    vec3 c = czm_textureCube(u_cubeMap, d).rgb * u_cubeParams.x;
    if (u_cubeParams.y > 0.0) c = mix(c, faceTint(d) * 0.45, u_cubeParams.y);
    out_FragColor = vec4(czm_gammaCorrect(c), czm_morphTime * u_cubeParams.z);
}
`;

export const STAR_VS = /* glsl */ `
in vec3 a_dir;
// rgb: display colour (max channel 1), a: V magnitude
in vec4 a_colorMag;

// x: reference magnitude (peak = 1), y: sigma0 (CSS px), z: sigma max (CSS px), w: sigma growth exponent
uniform vec4 u_starA;
// x: brightness (incl. fade), y: display gamma (1/2.2), z: max point size (px), w: limiting magnitude
uniform vec4 u_starB;

out vec3 v_color;
out float v_k;

void main()
{
    vec3 p = czm_viewRotation * (czm_temeToPseudoFixed * ((0.5 * czm_entireFrustum.y) * a_dir));
    gl_Position = czm_projection * vec4(p, 1.0);

    float mag = a_colorMag.w;
    // relative flux 10^(-0.4 (m - mRef)) = 2^(-1.3287712 (m - mRef))
    float flux = exp2(-1.3287712 * (mag - u_starA.x)) * u_starB.x;
    // Faint stars: fixed tight PSF, intensity ∝ flux. Bright stars: the core
    // saturates and the PSF widens slowly instead (sigma ∝ flux^w).
    float sigmaCss = clamp(u_starA.y * pow(max(flux, 1.0), u_starA.w), u_starA.y, u_starA.z);
    float peakLin = min(1.0, flux * (u_starA.y * u_starA.y) / (sigmaCss * sigmaCss));
    float peak = pow(peakLin, u_starB.y) * (1.0 - smoothstep(u_starB.w - 0.75, u_starB.w + 0.01, mag));

    // czm_pixelRatio = devicePixelRatio * resolutionScale. >= 0.5 device px
    // keeps sub-pixel stars from crawling while the camera moves.
    float sigma = max(sigmaCss * czm_pixelRatio, 0.5);
    float size = min(2.0 * ceil(3.0 * sigma) + 1.0, u_starB.z);
    if (peak < 0.4 / 255.0) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0); // outside the clip volume: culled
        size = 1.0;
    }
    gl_PointSize = size;
    v_color = a_colorMag.rgb * (peak * czm_morphTime);
    v_k = (size * size) / (2.0 * sigma * sigma);
}
`;

export const STAR_FS = /* glsl */ `
in vec3 v_color;
in float v_k;

void main()
{
    vec2 d = gl_PointCoord - 0.5;
    float r2 = dot(d, d);
    float g = exp(-r2 * v_k) * (1.0 - smoothstep(0.16, 0.25, r2));
    if (g < 0.002) discard;
    out_FragColor = vec4(czm_gammaCorrect(v_color * g), 1.0);
}
`;

export const MARKER_VS = /* glsl */ `
in vec3 a_dir;
in vec3 a_color;
uniform float u_markerSize;
out vec3 v_color;

void main()
{
    vec3 p = czm_viewRotation * (czm_temeToPseudoFixed * ((0.5 * czm_entireFrustum.y) * a_dir));
    gl_Position = czm_projection * vec4(p, 1.0);
    gl_PointSize = u_markerSize;
    v_color = a_color;
}
`;

export const MARKER_FS = /* glsl */ `
in vec3 v_color;

void main()
{
    vec2 d = (gl_PointCoord - 0.5) * 2.0;
    float r = length(d);
    float ring = smoothstep(0.70, 0.76, r) * (1.0 - smoothstep(0.88, 0.94, r));
    // crosshair ticks outside a clear centre (the star itself stays visible)
    float tick = (1.0 - smoothstep(0.03, 0.07, min(abs(d.x), abs(d.y)))) * step(0.3, r) * step(r, 0.6);
    float a = max(ring, tick);
    if (a < 0.01) discard;
    out_FragColor = vec4(v_color * a, 1.0);
}
`;
