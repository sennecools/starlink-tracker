/**
 * Pure, locale-independent formatting helpers for the HUD.
 *
 * No DOM, no Intl (its output depends on the user's locale and allocating
 * formatters is comparatively slow); every function here is deterministic so it
 * can be unit tested under plain Node (`node --test tests/`).
 */

/** Placeholder shown for missing / non-finite numbers. */
export const DASH = '—';

/** Zero-pad a non-negative integer to two digits. */
export function pad2(n: number): string {
  return n < 10 ? `0${n}` : `${n}`;
}

/**
 * Round to an integer and group thousands with commas: 9412.4 -> "9,412".
 * Non-finite input yields an em dash.
 */
export function groupThousands(n: number): string {
  if (!Number.isFinite(n)) return DASH;
  const r = Math.round(n);
  const neg = r < 0;
  const digits = `${neg ? -r : r}`;
  let out = '';
  let i = digits.length;
  while (i > 3) {
    out = `,${digits.slice(i - 3, i)}${out}`;
    i -= 3;
  }
  out = digits.slice(0, i) + out;
  return neg ? `-${out}` : out;
}

/** `toFixed` that never returns "-0.00". */
export function fixed(n: number, digits: number): string {
  if (!Number.isFinite(n)) return DASH;
  const s = n.toFixed(digits);
  return Number(s) === 0 ? (0).toFixed(digits) : s;
}

/** UTC calendar date of a Unix-ms timestamp: "2026-11-20". */
export function formatUtcDate(ms: number): string {
  const d = new Date(ms);
  return `${d.getUTCFullYear()}-${pad2(d.getUTCMonth() + 1)}-${pad2(d.getUTCDate())}`;
}

/**
 * UTC time of day of a Unix-ms timestamp: "14:03:27".
 * Pure arithmetic (no Date allocation) so it is cheap enough for a once-per-second tick.
 */
export function formatUtcTime(ms: number): string {
  const daySec = Math.floor((((ms % 86_400_000) + 86_400_000) % 86_400_000) / 1000);
  const h = Math.floor(daySec / 3600);
  const m = Math.floor((daySec % 3600) / 60);
  const s = daySec % 60;
  return `${pad2(h)}:${pad2(m)}:${pad2(s)}`;
}

/** Full top-bar clock string: "2026-11-20  14:03:27 UTC" (two spaces between date and time). */
export function formatUtcClock(date: Date): string {
  const ms = date.getTime();
  return `${formatUtcDate(ms)}  ${formatUtcTime(ms)} UTC`;
}

/** Compact UTC stamp to the minute: "2026-11-20 14:03 UTC". */
export function formatUtcStamp(date: Date): string {
  const ms = date.getTime();
  if (!Number.isFinite(ms)) return DASH;
  return `${formatUtcDate(ms)} ${formatUtcTime(ms).slice(0, 5)} UTC`;
}

/** Local wall-clock time "HH:MM" (24 h). */
export function formatLocalHHMM(date: Date): string {
  if (!Number.isFinite(date.getTime())) return DASH;
  return `${pad2(date.getHours())}:${pad2(date.getMinutes())}`;
}

/**
 * Coarse "time ago" label, uppercase for the HUD:
 * "JUST NOW", "42 S AGO", "14 MIN AGO", "3 H 05 MIN AGO", "2 D AGO".
 * Negative ages (clock skew) read as "JUST NOW".
 */
export function formatAge(ageMs: number): string {
  if (!Number.isFinite(ageMs)) return DASH;
  const s = Math.floor(ageMs / 1000);
  if (s < 5) return 'JUST NOW';
  if (s < 60) return `${s} S AGO`;
  const min = Math.floor(s / 60);
  if (min < 60) return `${min} MIN AGO`;
  const h = Math.floor(min / 60);
  if (h < 48) return `${h} H ${pad2(min % 60)} MIN AGO`;
  return `${Math.floor(h / 24)} D AGO`;
}

/** Latitude with hemisphere: 25.0123 -> "25.01°N", -3.5 -> "3.50°S". */
export function formatLat(lat: number, digits = 2): string {
  if (!Number.isFinite(lat)) return DASH;
  const s = fixed(Math.abs(lat), digits);
  return `${s}°${Number(s) === 0 || lat >= 0 ? 'N' : 'S'}`;
}

/** Normalise a longitude to [-180, 180). */
export function wrapLon(lon: number): number {
  return ((((lon + 180) % 360) + 360) % 360) - 180;
}

/** Longitude with hemisphere, wrapped to [-180, 180): 5.12 -> "5.12°E", 190 -> "170.00°W". */
export function formatLon(lon: number, digits = 2): string {
  if (!Number.isFinite(lon)) return DASH;
  const w = wrapLon(lon);
  const s = fixed(Math.abs(w), digits);
  return `${s}°${Number(s) === 0 || w >= 0 ? 'E' : 'W'}`;
}

/** "25.01°N 5.12°E". */
export function formatLatLon(lat: number, lon: number, digits = 2): string {
  return `${formatLat(lat, digits)} ${formatLon(lon, digits)}`;
}

/** Altitude: grouped integer km at/above 100 km, one decimal below. "16,042 KM", "42.5 KM". */
export function formatAltitudeKm(km: number): string {
  if (!Number.isFinite(km)) return `${DASH} KM`;
  return Math.abs(km) >= 100 ? `${groupThousands(km)} KM` : `${fixed(km, 1)} KM`;
}

export interface PerfInfo {
  fps: number;
  frameMs: number;
  resolutionScale: number;
  msaa: number;
  /** Worker time of the latest keyframe (ms); ~2.5 for 8k satellites with optimised code. */
  keyframeMs?: number;
  /** Worker time of the latest links request (ms, propagation + topology). */
  linksMs?: number;
}

/**
 * "60 FPS · 16.7 MS · RES 1.00× · MSAA 4×" (MSAA <= 1 reads "MSAA OFF"),
 * followed by " · KF 2.4 MS" / " · LINKS 16 MS" when those worker times are known.
 */
export function formatPerf(p: PerfInfo): string {
  const fps = Number.isFinite(p.fps) ? `${Math.round(p.fps)}` : DASH;
  const msaa = p.msaa > 1 ? `${Math.round(p.msaa)}×` : 'OFF';
  let s = `${fps} FPS · ${fixed(p.frameMs, 1)} MS · RES ${fixed(p.resolutionScale, 2)}× · MSAA ${msaa}`;
  if (p.keyframeMs !== undefined && Number.isFinite(p.keyframeMs)) s += ` · KF ${fixed(p.keyframeMs, 1)} MS`;
  if (p.linksMs !== undefined && Number.isFinite(p.linksMs)) s += ` · LINKS ${fixed(p.linksMs, 0)} MS`;
  return s;
}

/** Performance tier used to tint the FPS readout. */
export function perfTier(fps: number): 'good' | 'fair' | 'poor' {
  if (!(fps >= 30)) return 'poor';
  return fps >= 50 ? 'good' : 'fair';
}

/** sRGB 0..1 triple -> CSS "rgb(232 242 255)" (clamped). */
export function rgbToCss(color: readonly [number, number, number]): string {
  const c = (v: number): number => Math.round(Math.min(1, Math.max(0, v)) * 255);
  return `rgb(${c(color[0])} ${c(color[1])} ${c(color[2])})`;
}

/**
 * Grouped fixed-point number: (16042.37, 1) -> "16,042.4", (-0.04, 1) -> "0.0".
 * Non-finite input yields an em dash.
 */
export function formatGroupedFixed(n: number, digits: number): string {
  if (!Number.isFinite(n)) return DASH;
  const s = fixed(n, digits);
  const neg = s.charCodeAt(0) === 45; // '-'
  const body = neg ? s.slice(1) : s;
  const dot = body.indexOf('.');
  const intPart = dot < 0 ? body : body.slice(0, dot);
  const frac = dot < 0 ? '' : body.slice(dot);
  return `${neg ? '-' : ''}${groupThousands(Number(intPart))}${frac}`;
}

/** Plain degrees with a fixed number of decimals: 53.054 -> "53.05°". */
export function formatDeg(deg: number, digits = 2): string {
  if (!Number.isFinite(deg)) return DASH;
  return `${fixed(deg, digits)}°`;
}

/** Full UTC timestamp to the second: "2026-11-20 14:03:27 UTC". */
export function formatUtcDateTime(ms: number): string {
  if (!Number.isFinite(ms)) return DASH;
  return `${formatUtcDate(ms)} ${formatUtcTime(ms)} UTC`;
}

/** Parsed COSPAR / international designator. */
export interface CosparParts {
  year: number;
  launchNumber: number;
  piece: string;
}

const COSPAR_LONG = /^(\d{4})-(\d{3})([A-Z]{1,3})$/;
const COSPAR_TLE = /^(\d{2})(\d{3})([A-Z]{1,3})$/;

/**
 * Parse a COSPAR id, either in the long form "2019-074B" or the TLE form
 * "19074B" (two-digit years 57-99 are 19xx, 00-56 are 20xx, as in the TLE
 * convention). Whitespace and case are tolerated. Returns null when invalid.
 */
export function parseCospar(id: string): CosparParts | null {
  const s = id.trim().toUpperCase();
  let m = COSPAR_LONG.exec(s);
  if (m) return { year: Number(m[1]), launchNumber: Number(m[2]), piece: m[3] };
  m = COSPAR_TLE.exec(s);
  if (m) {
    const yy = Number(m[1]);
    return { year: yy >= 57 ? 1900 + yy : 2000 + yy, launchNumber: Number(m[2]), piece: m[3] };
  }
  return null;
}

/** "19074B" or "2019-074B" -> "2019-074B"; unparseable input -> em dash. */
export function formatCospar(id: string): string {
  const p = parseCospar(id);
  if (!p) return DASH;
  return `${p.year}-${`${p.launchNumber}`.padStart(3, '0')}${p.piece}`;
}

/** Launch cell of the info panel: (2019, 74) -> "2019 · #74". */
export function formatLaunch(year: number, launchNumber: number): string {
  if (!Number.isFinite(year) || !(year > 0)) return DASH;
  return Number.isFinite(launchNumber) && launchNumber > 0
    ? `${Math.round(year)} · #${Math.round(launchNumber)}`
    : `${Math.round(year)}`;
}

/** Apogee / perigee pair in whole km: (556.4, 547.8) -> "556 / 548". */
export function formatApsides(apogeeKm: number, perigeeKm: number): string {
  return `${groupThousands(apogeeKm)} / ${groupThousands(perigeeKm)}`;
}

/** Bottom-right camera readout: "ALT 16,000 KM · 25.00°N 5.00°E". */
export function formatCameraReadout(altitudeKm: number, lat: number, lon: number): string {
  return `ALT ${formatAltitudeKm(altitudeKm)} · ${formatLatLon(lat, lon)}`;
}

/**
 * Split a hover label "NAME\nSHELL" into its two lines (second line may be
 * empty). Only the first newline separates; the rest stays in the sub-line.
 */
export function splitLabel(text: string): [string, string] {
  const i = text.indexOf('\n');
  return i < 0 ? [text, ''] : [text.slice(0, i), text.slice(i + 1)];
}

// ---------------------------------------------------------------------------
// Hardware + modelled laser links
// ---------------------------------------------------------------------------

/** Laser terminal count cell: 0 -> "NONE", 3 -> "3", non-finite -> em dash. */
export function formatLaserTerminals(n: number): string {
  if (!Number.isFinite(n)) return DASH;
  const r = Math.round(n);
  return r <= 0 ? 'NONE' : `${r}`;
}

/** Link kind tag (LaserLinks.kinds / PanelLink.kind): 0 = in-plane, 1 = cross-plane. */
export function linkKindLabel(kind: number): string {
  if (kind === 0) return 'IN-PLANE';
  if (kind === 1) return 'CROSS-PLANE';
  return 'LINK';
}

/** Right-hand cell of the HUD "LASER LINKS" toggle: the link count, or "OFF" while disabled. */
export function formatLinkCount(enabled: boolean, links: number): string {
  return enabled ? groupThousands(links) : 'OFF';
}

/** HUD footer line: 8123 -> "8,123 LASER-EQUIPPED". */
export function formatLaserEquipped(n: number): string {
  return `${groupThousands(n)} LASER-EQUIPPED`;
}

// ---------------------------------------------------------------------------
// Data sources, shell notes, V3 hardware, toggles
// ---------------------------------------------------------------------------

/**
 * Split a label on its " · " separators ("V3 · STARSHIP" -> ["V3", "STARSHIP"]).
 * The HUD renders the parts with a round separator dot instead of U+00B7, which
 * the ASCII-only D-DIN webfont would draw in a fallback font. Empty parts are dropped.
 */
export function splitDots(text: string): string[] {
  const out: string[] = [];
  for (const part of text.split('·')) {
    const p = part.trim();
    if (p) out.push(p);
  }
  return out;
}

/** Collapse whitespace and drop a trailing parenthetical: "CelesTrak SupGP (SpaceX ephemerides)" -> "CelesTrak SupGP". */
function shortSource(name: string): string {
  return name.replace(/\s*\([^)]*\)\s*$/, '').replace(/\s+/g, ' ').trim();
}

/**
 * Hero source line, uppercase.
 *  - No / empty `sources`: the primary source as given ("CELESTRAK SUPGP (SPACEX EPHEMERIDES)").
 *  - Otherwise the contributing sources, short form (trailing parenthetical dropped),
 *    de-duplicated case-insensitively, joined with " + ":
 *    ["CelesTrak GP", "Recent launches"] -> "CELESTRAK GP + RECENT LAUNCHES".
 *  - More than `max` (default 3) sources: the first `max - 1` and "+ N MORE".
 * Nothing usable at all yields an em dash.
 */
export function formatDataSources(primary: string, sources?: readonly string[] | null, max = 3): string {
  const names: string[] = [];
  const seen = new Set<string>();
  if (sources) {
    for (const s of sources) {
      if (typeof s !== 'string') continue;
      const n = shortSource(s);
      const key = n.toUpperCase();
      if (!n || seen.has(key)) continue;
      seen.add(key);
      names.push(key);
    }
  }
  if (names.length === 0) {
    const p = (typeof primary === 'string' ? primary : '').replace(/\s+/g, ' ').trim();
    return p ? p.toUpperCase() : DASH;
  }
  const cap = Math.max(1, Math.floor(max));
  if (names.length <= cap) return names.join(' + ');
  const shown = names.slice(0, Math.max(1, cap - 1));
  return `${shown.join(' + ')} + ${names.length - shown.length} MORE`;
}

/** Secondary sub-label under a legend row, by shell id ('' = none). */
export function shellNote(shellId: string): string {
  return shellId === 'v3' ? 'NEW GENERATION · LAUNCHED ON STARSHIP' : '';
}

/** True for a V3 hardware label ("V3", "v3 (est.)", "V3 MINI"), false for "V2 MINI" / "V1.5" / "V30". */
export function isV3Label(busLabel: string): boolean {
  return /^\s*V3(?![\d.])/i.test(busLabel);
}

/** Secondary line under the info panel's HARDWARE row: the V3 model is an artist's approximation. */
export function hardwareNote(busLabel: string): string {
  return isV3Label(busLabel) ? "ARTIST'S APPROXIMATION" : '';
}

/** Text of an ON / OFF toggle cell. */
export function formatOnOff(on: boolean): 'ON' | 'OFF' {
  return on ? 'ON' : 'OFF';
}
