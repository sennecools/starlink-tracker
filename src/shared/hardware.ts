/**
 * Estimated Starlink hardware generation per satellite, derived from the
 * COSPAR launch designator. SpaceX does not publish a per-satellite list, so
 * this is a documented heuristic based on public launch history:
 *
 *  - v1.0  2019 – mid 2021 (Groups 1/L-series). No optical links, except the
 *          10 polar test satellites launched on Transporter-1 (2021-006).
 *  - v1.5  first flown Sept 2021 (Group 2-1, COSPAR 2021-082) until early 2023.
 *          Optical inter-satellite links (4 terminals reported).
 *  - V2 Mini  from Feb 2023 (Group 6-1, COSPAR 2023-014) onwards, incl. the
 *          Direct-to-Cell variant. 3 laser terminals per satellite.
 *  - V3    launched on Starship from Starbase (first: Flight 14, 2026-09-28;
 *          Flight 15, 2026-11-13 = COSPAR 2026-256). ~2,000 kg, six 400 Gbps
 *          laser terminals. Recognised by launch year >= 2026 and an
 *          inclination < 35° (Starbase launches go to ~26–33°; Falcon 9
 *          Starlink launches from Florida/California never do), or "V3" in the name.
 */
export interface BusVersion {
  id: 'v1.0' | 'v1.5' | 'v2mini' | 'v3';
  label: string;
  /** Model file under /models/ used for the close-up 3D view. */
  model: string;
}

export const BUS_VERSIONS: readonly BusVersion[] = [
  { id: 'v1.0', label: 'V1.0', model: 'starlink-v15.glb' },
  { id: 'v1.5', label: 'V1.5', model: 'starlink-v15.glb' },
  { id: 'v2mini', label: 'V2 MINI', model: 'starlink-v2mini.glb' },
  { id: 'v3', label: 'V3', model: 'starlink-v3.glb' },
] as const;

export const BUS_V10 = 0;
export const BUS_V15 = 1;
export const BUS_V2MINI = 2;
export const BUS_V3 = 3;

/** Parse the TLE international designator "19074B" -> { year: 2019, launch: 74 }. */
export function parseIntlDesignator(intl: string): { year: number; launch: number } | null {
  const m = /^\s*(\d{2})(\d{3})/.exec(intl);
  if (!m) return null;
  const yy = Number(m[1]);
  return { year: yy >= 57 ? 1900 + yy : 2000 + yy, launch: Number(m[2]) };
}

/** Returns [busIndex, laserTerminals]. */
export function estimateHardware(intlDesignator: string, inclinationDeg: number, name = ''): [number, number] {
  if (/\bV3\b/i.test(name)) return [BUS_V3, 6];
  const d = parseIntlDesignator(intlDesignator);
  if (d && d.year >= 2026 && inclinationDeg < 35) return [BUS_V3, 6];
  if (!d) return [BUS_V2MINI, 3]; // unknown/new objects: assume current hardware
  const { year, launch } = d;
  if (year < 2021) return [BUS_V10, 0];
  if (year === 2021) {
    if (launch >= 82) return [BUS_V15, 4];
    // Transporter-1 polar test satellites carried the first laser terminals.
    if (launch === 6 && inclinationDeg > 96) return [BUS_V10, 2];
    return [BUS_V10, 0];
  }
  if (year === 2022 || (year === 2023 && launch < 14)) return [BUS_V15, 4];
  return [BUS_V2MINI, 3];
}
