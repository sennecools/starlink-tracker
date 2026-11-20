import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  DASH,
  fixed,
  formatAge,
  formatAltitudeKm,
  formatApsides,
  formatCameraReadout,
  formatCospar,
  formatDataSources,
  formatDeg,
  formatGroupedFixed,
  formatLaserEquipped,
  formatLaserTerminals,
  formatLaunch,
  formatLat,
  formatLatLon,
  formatLocalHHMM,
  formatLinkCount,
  formatLon,
  formatOnOff,
  formatPerf,
  formatUtcClock,
  formatUtcDate,
  formatUtcStamp,
  formatUtcDateTime,
  formatUtcTime,
  groupThousands,
  hardwareNote,
  isV3Label,
  linkKindLabel,
  pad2,
  parseCospar,
  perfTier,
  rgbToCss,
  shellNote,
  splitDots,
  splitLabel,
  wrapLon,
} from '../src/ui/format.ts';

test('pad2', () => {
  assert.equal(pad2(0), '00');
  assert.equal(pad2(7), '07');
  assert.equal(pad2(42), '42');
});

test('groupThousands', () => {
  assert.equal(groupThousands(0), '0');
  assert.equal(groupThousands(999), '999');
  assert.equal(groupThousands(1000), '1,000');
  assert.equal(groupThousands(9412), '9,412');
  assert.equal(groupThousands(9412.6), '9,413');
  assert.equal(groupThousands(1234567), '1,234,567');
  assert.equal(groupThousands(-16042), '-16,042');
  assert.equal(groupThousands(NaN), DASH);
  assert.equal(groupThousands(Infinity), DASH);
});

test('fixed never yields negative zero', () => {
  assert.equal(fixed(-0.001, 2), '0.00');
  assert.equal(fixed(-0, 1), '0.0');
  assert.equal(fixed(-1.256, 2), '-1.26');
  assert.equal(fixed(NaN, 2), DASH);
});

test('UTC clock formatting', () => {
  const d = new Date(Date.UTC(2026, 10, 20, 14, 3, 27, 999));
  assert.equal(formatUtcDate(d.getTime()), '2026-11-20');
  assert.equal(formatUtcTime(d.getTime()), '14:03:27');
  assert.equal(formatUtcClock(d), '2026-11-20  14:03:27 UTC');
  assert.equal(formatUtcStamp(d), '2026-11-20 14:03 UTC');
  assert.equal(formatUtcTime(Date.UTC(2026, 0, 1, 0, 0, 0)), '00:00:00');
  assert.equal(formatUtcTime(Date.UTC(2026, 0, 1, 23, 59, 59)), '23:59:59');
  // Pre-1970 timestamps (negative ms) still produce a valid time of day.
  assert.equal(formatUtcTime(Date.UTC(1969, 11, 31, 23, 59, 58)), '23:59:58');
  assert.equal(formatUtcStamp(new Date(NaN)), DASH);
});

test('formatLocalHHMM uses local time', () => {
  const d = new Date(2026, 10, 20, 9, 5, 0);
  assert.equal(formatLocalHHMM(d), '09:05');
  assert.equal(formatLocalHHMM(new Date(NaN)), DASH);
});

test('formatAge', () => {
  assert.equal(formatAge(-5000), 'JUST NOW');
  assert.equal(formatAge(0), 'JUST NOW');
  assert.equal(formatAge(42_000), '42 S AGO');
  assert.equal(formatAge(14 * 60_000 + 30_000), '14 MIN AGO');
  assert.equal(formatAge(3 * 3_600_000 + 5 * 60_000), '3 H 05 MIN AGO');
  assert.equal(formatAge(50 * 3_600_000), '2 D AGO');
  assert.equal(formatAge(NaN), DASH);
});

test('latitude / longitude', () => {
  assert.equal(formatLat(25.0123), '25.01°N');
  assert.equal(formatLat(-3.5), '3.50°S');
  assert.equal(formatLat(-0.001), '0.00°N');
  assert.equal(formatLon(5.12), '5.12°E');
  assert.equal(formatLon(-122.4194), '122.42°W');
  assert.equal(formatLon(190), '170.00°W');
  assert.equal(formatLon(-0.001), '0.00°E');
  assert.equal(formatLatLon(25.0123, 5.1234), '25.01°N 5.12°E');
  assert.equal(formatLat(NaN), DASH);
  assert.equal(wrapLon(540), -180);
  assert.equal(wrapLon(-190), 170);
});

test('formatAltitudeKm', () => {
  assert.equal(formatAltitudeKm(16042.4), '16,042 KM');
  assert.equal(formatAltitudeKm(550), '550 KM');
  assert.equal(formatAltitudeKm(42.46), '42.5 KM');
  assert.equal(formatAltitudeKm(NaN), `${DASH} KM`);
});

test('formatPerf / perfTier', () => {
  assert.equal(
    formatPerf({ fps: 59.7, frameMs: 16.74, resolutionScale: 1, msaa: 4 }),
    '60 FPS · 16.7 MS · RES 1.00× · MSAA 4×',
  );
  assert.equal(
    formatPerf({ fps: 31, frameMs: 32.2, resolutionScale: 0.75, msaa: 1 }),
    '31 FPS · 32.2 MS · RES 0.75× · MSAA OFF',
  );
  assert.equal(
    formatPerf({ fps: 60, frameMs: 16.7, resolutionScale: 1, msaa: 1, keyframeMs: 2.43, linksMs: 16.4 }),
    '60 FPS · 16.7 MS · RES 1.00× · MSAA OFF · KF 2.4 MS · LINKS 16 MS',
  );
  assert.equal(
    formatPerf({ fps: 60, frameMs: 16.7, resolutionScale: 1, msaa: 1, keyframeMs: 2.43, linksMs: NaN }),
    '60 FPS · 16.7 MS · RES 1.00× · MSAA OFF · KF 2.4 MS',
  );
  assert.equal(perfTier(60), 'good');
  assert.equal(perfTier(45), 'fair');
  assert.equal(perfTier(12), 'poor');
  assert.equal(perfTier(NaN), 'poor');
});

test('rgbToCss', () => {
  assert.equal(rgbToCss([0.91, 0.95, 1.0]), 'rgb(232 242 255)');
  assert.equal(rgbToCss([-1, 2, 0.5]), 'rgb(0 255 128)');
});

test('formatGroupedFixed', () => {
  assert.equal(formatGroupedFixed(16042.37, 1), '16,042.4');
  assert.equal(formatGroupedFixed(550.25, 1), '550.3');
  assert.equal(formatGroupedFixed(7.594, 2), '7.59');
  assert.equal(formatGroupedFixed(1234567.891, 0), '1,234,568');
  assert.equal(formatGroupedFixed(-16042.37, 1), '-16,042.4');
  assert.equal(formatGroupedFixed(-0.04, 1), '0.0');
  assert.equal(formatGroupedFixed(999.96, 1), '1,000.0');
  assert.equal(formatGroupedFixed(NaN, 1), DASH);
});

test('formatDeg / formatUtcDateTime', () => {
  assert.equal(formatDeg(53.054), '53.05°');
  assert.equal(formatDeg(97.6, 1), '97.6°');
  assert.equal(formatDeg(-0.001), '0.00°');
  assert.equal(formatDeg(NaN), DASH);
  assert.equal(formatUtcDateTime(Date.UTC(2026, 10, 20, 8, 12, 5)), '2026-11-20 08:12:05 UTC');
  assert.equal(formatUtcDateTime(NaN), DASH);
});

test('COSPAR parsing and formatting', () => {
  assert.deepEqual(parseCospar('19074B'), { year: 2019, launchNumber: 74, piece: 'B' });
  assert.deepEqual(parseCospar('2019-074B'), { year: 2019, launchNumber: 74, piece: 'B' });
  assert.deepEqual(parseCospar(' 98067a  '), { year: 1998, launchNumber: 67, piece: 'A' });
  assert.deepEqual(parseCospar('57001AB'), { year: 1957, launchNumber: 1, piece: 'AB' });
  assert.deepEqual(parseCospar('56001A'), { year: 2056, launchNumber: 1, piece: 'A' });
  assert.equal(parseCospar(''), null);
  assert.equal(parseCospar('19074'), null);
  assert.equal(parseCospar('2019-74B'), null);
  assert.equal(formatCospar('19074B'), '2019-074B');
  assert.equal(formatCospar('24001ABC'), '2024-001ABC');
  assert.equal(formatCospar('2019-074B'), '2019-074B');
  assert.equal(formatCospar('garbage'), DASH);
});

test('formatLaunch / formatApsides', () => {
  assert.equal(formatLaunch(2019, 74), '2019 · #74');
  assert.equal(formatLaunch(2024, 0), '2024');
  assert.equal(formatLaunch(NaN, 3), DASH);
  assert.equal(formatLaunch(0, 3), DASH);
  assert.equal(formatApsides(556.4, 547.8), '556 / 548');
  assert.equal(formatApsides(1234.5, NaN), `1,235 / ${DASH}`);
});

test('formatCameraReadout', () => {
  assert.equal(formatCameraReadout(16000, 25, 5), 'ALT 16,000 KM · 25.00°N 5.00°E');
  assert.equal(formatCameraReadout(42.46, -3.5, 190), 'ALT 42.5 KM · 3.50°S 170.00°W');
});

test('splitLabel', () => {
  assert.deepEqual(splitLabel('STARLINK-1008'), ['STARLINK-1008', '']);
  assert.deepEqual(splitLabel('STARLINK-1008\nINC 53°'), ['STARLINK-1008', 'INC 53°']);
  assert.deepEqual(splitLabel('A\nB\nC'), ['A', 'B\nC']);
  assert.deepEqual(splitLabel(''), ['', '']);
});

test('laser hardware + link helpers', () => {
  assert.equal(formatLaserTerminals(0), 'NONE');
  assert.equal(formatLaserTerminals(-1), 'NONE');
  assert.equal(formatLaserTerminals(3), '3');
  assert.equal(formatLaserTerminals(4.2), '4');
  assert.equal(formatLaserTerminals(NaN), DASH);

  assert.equal(linkKindLabel(0), 'IN-PLANE');
  assert.equal(linkKindLabel(1), 'CROSS-PLANE');
  assert.equal(linkKindLabel(7), 'LINK');

  assert.equal(formatLinkCount(true, 14_532), '14,532');
  assert.equal(formatLinkCount(true, 0), '0');
  assert.equal(formatLinkCount(false, 14_532), 'OFF');
  assert.equal(formatLinkCount(true, NaN), DASH);

  assert.equal(formatLaserEquipped(8123), '8,123 LASER-EQUIPPED');
  assert.equal(formatLaserEquipped(0), '0 LASER-EQUIPPED');
});

test('formatDataSources', () => {
  // primary only (backwards compatible: as given, uppercase)
  assert.equal(formatDataSources('CelesTrak SupGP (SpaceX ephemerides)'), 'CELESTRAK SUPGP (SPACEX EPHEMERIDES)');
  assert.equal(formatDataSources('CelesTrak GP', []), 'CELESTRAK GP');
  assert.equal(formatDataSources('CelesTrak GP', null), 'CELESTRAK GP');
  assert.equal(formatDataSources('  ', []), DASH);
  // contributing sources, compact
  assert.equal(formatDataSources('CelesTrak GP', ['CelesTrak GP', 'Recent launches']), 'CELESTRAK GP + RECENT LAUNCHES');
  assert.equal(
    formatDataSources('x', ['CelesTrak GP (incomplete download)', 'Recent launches (last 30 days)']),
    'CELESTRAK GP + RECENT LAUNCHES',
  );
  // de-duplicated case-insensitively, empties dropped, whitespace collapsed
  assert.equal(formatDataSources('x', ['CelesTrak  GP', 'celestrak gp', '', '  ', 'Recent launches']), 'CELESTRAK GP + RECENT LAUNCHES');
  // all empty -> primary
  assert.equal(formatDataSources('Disk cache', ['', ' ']), 'DISK CACHE');
  // cap
  assert.equal(formatDataSources('x', ['A', 'B', 'C']), 'A + B + C');
  assert.equal(formatDataSources('x', ['A', 'B', 'C', 'D']), 'A + B + 2 MORE');
  assert.equal(formatDataSources('x', ['A', 'B', 'C'], 1), 'A + 2 MORE');
});

test('splitDots + shellNote', () => {
  assert.deepEqual(splitDots('V3 · STARSHIP'), ['V3', 'STARSHIP']);
  assert.deepEqual(splitDots('INC 53°'), ['INC 53°']);
  assert.deepEqual(splitDots(' · A ·· B · '), ['A', 'B']);
  assert.deepEqual(splitDots(''), []);
  assert.equal(shellNote('v3'), 'NEW GENERATION · LAUNCHED ON STARSHIP');
  assert.deepEqual(splitDots(shellNote('v3')), ['NEW GENERATION', 'LAUNCHED ON STARSHIP']);
  assert.equal(shellNote('inc53'), '');
  assert.equal(shellNote(''), '');
});

test('V3 hardware helpers + on/off', () => {
  assert.equal(isV3Label('V3'), true);
  assert.equal(isV3Label(' v3 '), true);
  assert.equal(isV3Label('V3 (EST.)'), true);
  assert.equal(isV3Label('V2 MINI'), false);
  assert.equal(isV3Label('V1.5'), false);
  assert.equal(isV3Label('V30'), false);
  assert.equal(isV3Label('V3.5'), false);
  assert.equal(isV3Label(''), false);
  assert.equal(hardwareNote('V3'), "ARTIST'S APPROXIMATION");
  assert.equal(hardwareNote('V2 MINI'), '');
  assert.equal(hardwareNote(''), '');
  assert.equal(formatOnOff(true), 'ON');
  assert.equal(formatOnOff(false), 'OFF');
});
