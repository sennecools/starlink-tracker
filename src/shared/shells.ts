/**
 * Orbital "shells" used to colour-code the constellation.
 *
 * Starlink is deployed in groups of orbital planes that share an inclination
 * (and roughly an altitude). Inclination is the most stable discriminator:
 * altitudes change as satellites raise, are lowered (SpaceX moved many shells
 * from ~550 km to ~480 km in 2026) or de-orbit, so altitude is only used to
 * flag satellites that are clearly not at an operational altitude.
 *
 * Order matters: `classifyShell` returns the first matching index.
 */
export interface ShellDef {
  id: string;
  /** Short uppercase HUD label. */
  label: string;
  /** sRGB colour, components 0..1. */
  color: readonly [number, number, number];
}

export const SHELLS: readonly ShellDef[] = [
  { id: 'inc53', label: 'INC 53°', color: [0.91, 0.95, 1.0] },
  { id: 'inc43', label: 'INC 43°', color: [0.31, 0.76, 1.0] },
  { id: 'inc70', label: 'INC 70°', color: [0.54, 0.49, 1.0] },
  { id: 'polar', label: 'POLAR 97.6°', color: [0.27, 0.95, 0.77] },
  { id: 'inc33', label: 'INC 33°', color: [1.0, 0.72, 0.3] },
  { id: 'transit', label: 'RAISING / DESCENDING', color: [1.0, 0.3, 0.43] },
  { id: 'other', label: 'OTHER', color: [0.55, 0.59, 0.66] },
  // Appended last so existing indices stay stable. Warm gold so the new
  // Starship-launched generation is easy to spot against the cool palette.
  { id: 'v3', label: 'V3 · STARSHIP', color: [1.0, 0.82, 0.52] },
] as const;

export const SHELL_TRANSIT = 5;
export const SHELL_OTHER = 6;
export const SHELL_V3 = 7;

/** Below this mean altitude a satellite is considered not (yet / anymore) operational. */
export const OPERATIONAL_MIN_ALT_KM = 420;

/**
 * `bus` is the BUS_VERSIONS index from src/shared/hardware.ts (optional for
 * backwards compatibility). V3 satellites get their own class regardless of
 * altitude: right now they are all still raising from the ~270 km Starship
 * insertion orbit, and the point is to make the new generation findable.
 */
export function classifyShell(inclinationDeg: number, meanAltitudeKm: number, bus = -1): number {
  if (bus === 3 /* BUS_V3 */) return SHELL_V3;
  if (!(meanAltitudeKm >= OPERATIONAL_MIN_ALT_KM)) return SHELL_TRANSIT;
  const i = inclinationDeg;
  if (i >= 52.5 && i <= 53.6) return 0;
  if (i >= 42.0 && i <= 44.0) return 1;
  if (i >= 69.0 && i <= 71.0) return 2;
  if (i >= 96.0 && i <= 98.8) return 3;
  if (i >= 32.0 && i <= 34.0) return 4;
  return SHELL_OTHER;
}
