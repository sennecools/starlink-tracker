/**
 * OMM CSV parser tests (server/omm.ts). No network.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { json2satrec, propagate, twoline2satrec } from 'satellite.js';
import {
  filterStarlink,
  intlDesignatorFromObjectId,
  isStarlinkName,
  looksLikeOmmCsv,
  ommEpochMs,
  parseElementsText,
  parseOmmCsv,
  recordEpochMs,
  splitCsvLine,
} from '../server/omm.ts';
import { parseTle, tleEpochMs } from '../server/tle.ts';
import { recordToSatrec, satrecEpochMs } from '../src/worker/propagate.ts';

const FIXTURE = fileURLToPath(new URL('../fixtures/sup-gp-starlink-partial.tle', import.meta.url));

const HEADER =
  'OBJECT_NAME,OBJECT_ID,EPOCH,MEAN_MOTION,ECCENTRICITY,INCLINATION,RA_OF_ASC_NODE,ARG_OF_PERICENTER,MEAN_ANOMALY,' +
  'EPHEMERIS_TYPE,CLASSIFICATION_TYPE,NORAD_CAT_ID,ELEMENT_SET_NO,REV_AT_EPOCH,BSTAR,MEAN_MOTION_DOT,MEAN_MOTION_DDOT';
/** Real row from CelesTrak GROUP=last-30-days&FORMAT=csv (NORAD id > 99,999: not available as TLE). */
const SAMPLE =
  'XIGUANG-4 01,2026-238A,2026-11-20T05:40:30.660672,15.09513871,.00127497,38.9973,266.7826,175.9060,184.1825,0,U,101142,999,455,.25205656E-3,.399E-4,0';
/** A Starship-launched V3 (Flight 15, COSPAR 2026-256) in its ~290 km insertion orbit. */
const V3_ROW =
  'STARLINK-40096,2026-256A,2026-11-20T03:12:00.000000,15.94000000,.00080000,30.5000,120.0000,90.0000,270.0000,0,U,101415,999,105,.50000000E-3,.10000000E-3,0';

test('real sample row: every field, NORAD id above 99,999', () => {
  const { records, rejected, headerOk, truncated } = parseOmmCsv(`${HEADER}\r\n${SAMPLE}\r\n`);
  assert.equal(headerOk, true);
  assert.equal(truncated, false);
  assert.equal(rejected, 0);
  assert.equal(records.length, 1);
  const r = records[0]!;
  assert.equal(r.name, 'XIGUANG-4 01');
  assert.equal(r.noradId, 101142);
  assert.equal(r.intlDesignator, '26238A');
  assert.equal(r.line1, undefined);
  assert.equal(r.line2, undefined);
  assert.deepEqual(r.omm, {
    OBJECT_NAME: 'XIGUANG-4 01',
    OBJECT_ID: '2026-238A',
    EPOCH: '2026-11-20T05:40:30.660672',
    MEAN_MOTION: 15.09513871,
    ECCENTRICITY: 0.00127497,
    INCLINATION: 38.9973,
    RA_OF_ASC_NODE: 266.7826,
    ARG_OF_PERICENTER: 175.906,
    MEAN_ANOMALY: 184.1825,
    EPHEMERIS_TYPE: 0,
    CLASSIFICATION_TYPE: 'U',
    NORAD_CAT_ID: 101142,
    ELEMENT_SET_NO: 999,
    REV_AT_EPOCH: 455,
    BSTAR: 0.00025205656,
    MEAN_MOTION_DOT: 0.0000399,
    MEAN_MOTION_DDOT: 0,
  });
  // satellite.js accepts it as-is and propagates it.
  const satrec = json2satrec(r.omm!);
  assert.equal(satrec.error, 0);
  assert.equal(satrec.satnum, '101142');
  assert.ok(Math.abs(satrecEpochMs(satrec) - ommEpochMs(r.omm!.EPOCH)) <= 1);
  const pv = propagate(satrec, new Date(Date.UTC(2026, 10, 20, 7)));
  assert.ok(pv !== null);
  const rKm = Math.hypot(pv.position.x, pv.position.y, pv.position.z);
  assert.ok(rKm > 6378 + 350 && rKm < 6378 + 650, `radius ${rKm}`);
  // The shared helper picks json2satrec for OMM records.
  assert.equal(recordToSatrec(r).satnum, '101142');
});

test('epoch and COSPAR helpers', () => {
  assert.equal(ommEpochMs('2026-11-20T05:40:30.660672'), Date.UTC(2026, 10, 20, 5, 40, 30) + 660.672);
  assert.equal(ommEpochMs('2026-11-20T05:40:30.660672Z'), ommEpochMs('2026-11-20 05:40:30.660672'));
  assert.equal(ommEpochMs('2026-11-20T05:40:30'), Date.UTC(2026, 10, 20, 5, 40, 30));
  for (const bad of ['', '2026-11-20', '2026-13-01T00:00:00', 'yesterday', '2026-11-20T25:00:00']) {
    assert.ok(Number.isNaN(ommEpochMs(bad)), bad);
  }
  assert.equal(intlDesignatorFromObjectId('2026-256A'), '26256A');
  assert.equal(intlDesignatorFromObjectId('1998-067A'), '98067A');
  assert.equal(intlDesignatorFromObjectId(' 2019-074BC '), '19074BC');
  assert.equal(intlDesignatorFromObjectId('19074B'), '19074B');
  assert.equal(intlDesignatorFromObjectId(''), '');
  assert.equal(intlDesignatorFromObjectId('UNKNOWN'), '');
});

test('CRLF, BOM, blank lines, quoted fields, header in any order / case', () => {
  const header = 'norad_cat_id,"OBJECT_NAME",EPOCH,MEAN_MOTION,ECCENTRICITY,INCLINATION,RA_OF_ASC_NODE,ARG_OF_PERICENTER,MEAN_ANOMALY,BSTAR,MEAN_MOTION_DOT,MEAN_MOTION_DDOT,OBJECT_ID';
  const row = '101415,"STARLINK-40096, ""V3""",2026-11-20T03:12:00.000000,15.94,.0008,30.5,120,90,270,.5E-3,.1E-3,0,2026-256A';
  const text = `﻿\r\n${header}\r\n\r\n${row}\r\n   \r\n`;
  assert.ok(looksLikeOmmCsv(text));
  const { records, rejected } = parseOmmCsv(text);
  assert.equal(rejected, 0);
  assert.equal(records.length, 1);
  assert.equal(records[0]!.name, 'STARLINK-40096, "V3"');
  assert.equal(records[0]!.noradId, 101415);
  assert.equal(records[0]!.intlDesignator, '26256A');
  // Optional columns absent: sane defaults.
  assert.equal(records[0]!.omm!.ELEMENT_SET_NO, 999);
  assert.equal(records[0]!.omm!.REV_AT_EPOCH, undefined);
  assert.deepEqual(splitCsvLine('a,"b,c",,"d""e"'), ['a', 'b,c', '', 'd"e']);
});

test('truncated last line is ignored; a complete unterminated last line is kept', () => {
  const full = `${HEADER}\n${SAMPLE}\n${V3_ROW}`;
  // Complete file without trailing newline: both rows.
  const a = parseOmmCsv(full);
  assert.equal(a.records.length, 2);
  assert.equal(a.truncated, false);
  // Cut inside the last row (missing columns): ignored, not rejected.
  const b = parseOmmCsv(full.slice(0, full.length - 30));
  assert.equal(b.records.length, 1);
  assert.equal(b.rejected, 0);
  assert.equal(b.truncated, true);
  // Cut inside the LAST column (".1E-3" -> ".1E"): only detectable by the caller.
  const cut = `${HEADER}\n${SAMPLE}\n${SAMPLE.slice(0, -7)}`;
  assert.equal(parseOmmCsv(cut, { truncated: true }).records.length, 1);
  // A known cut-off download never trusts its unterminated last line.
  const c = parseOmmCsv(full, { truncated: true });
  assert.equal(c.records.length, 1);
  assert.equal(c.truncated, true);
  // Terminated: nothing is truncated even with the flag.
  assert.equal(parseOmmCsv(full + '\r\n', { truncated: true }).records.length, 2);
  // Header only / cut inside the header.
  assert.equal(parseOmmCsv(`${HEADER}\n`).records.length, 0);
  assert.equal(parseOmmCsv(HEADER.slice(0, 40)).headerOk, false);
});

test('sanity checks reject bad rows', () => {
  const fields = SAMPLE.split(',');
  const col = (name: string) => HEADER.split(',').indexOf(name);
  const withField = (name: string, value: string) => {
    const f = [...fields];
    f[col(name)] = value;
    return f.join(',');
  };
  const bad = [
    withField('MEAN_MOTION', '0'),
    withField('MEAN_MOTION', '25.1'),
    withField('MEAN_MOTION', ''),
    withField('ECCENTRICITY', '1.2'),
    withField('ECCENTRICITY', '-0.1'),
    withField('INCLINATION', '181'),
    withField('INCLINATION', 'abc'),
    withField('EPOCH', '2026-11-20'),
    withField('EPOCH', 'not a date'),
    withField('NORAD_CAT_ID', ''),
    withField('NORAD_CAT_ID', '12.5'),
    withField('NORAD_CAT_ID', '-5'),
    withField('EPHEMERIS_TYPE', '2'),
    withField('BSTAR', 'x'),
    fields.slice(0, -1).join(','), // missing a column
    SAMPLE + ',extra',
  ];
  const text = [HEADER, SAMPLE, ...bad, V3_ROW, ''].join('\n');
  const { records, rejected } = parseOmmCsv(text);
  assert.deepEqual(records.map((r) => r.noradId), [101142, 101415]);
  assert.equal(rejected, bad.length);
  // Blank optional numbers default to 0.
  const blankDrag = parseOmmCsv([HEADER, withField('BSTAR', ''), ''].join('\n'));
  assert.equal(blankDrag.records[0]!.omm!.BSTAR, 0);
});

test('no / incomplete header: nothing parsed', () => {
  assert.equal(parseOmmCsv('').headerOk, false);
  assert.equal(parseOmmCsv('No GP data found\n').headerOk, false);
  const noNorad = HEADER.replace('NORAD_CAT_ID', 'CAT');
  const r = parseOmmCsv(`${noNorad}\n${SAMPLE}\n`);
  assert.equal(r.headerOk, false);
  assert.equal(r.records.length, 0);
});

test('parseElementsText detects OMM CSV vs TLE; recordEpochMs for both', () => {
  const csv = parseElementsText(`${HEADER}\n${SAMPLE}\n`);
  assert.equal(csv.format, 'omm-csv');
  assert.equal(csv.recognised, true);
  assert.equal(csv.records.length, 1);
  assert.equal(recordEpochMs(csv.records[0]!), ommEpochMs('2026-11-20T05:40:30.660672'));

  const tleText = readFileSync(FIXTURE, 'utf8');
  const tle = parseElementsText(tleText);
  assert.equal(tle.format, 'tle');
  assert.equal(tle.records.length, 853);
  assert.equal(recordEpochMs(tle.records[0]!), tleEpochMs(tle.records[0]!.line1!));

  const junk = parseElementsText('No GP data found');
  assert.equal(junk.recognised, false);
  assert.ok(Number.isNaN(recordEpochMs({ name: 'X', noradId: 1, intlDesignator: '' })));
});

test('Starlink name filter', () => {
  assert.ok(isStarlinkName('STARLINK-40096'));
  assert.ok(isStarlinkName('Starlink-1008 [DTC]'));
  assert.ok(!isStarlinkName('XIGUANG-4 01'));
  assert.ok(!isStarlinkName('STARLINKX'));
  const { records } = parseOmmCsv(`${HEADER}\n${SAMPLE}\n${V3_ROW}\n`);
  assert.deepEqual(filterStarlink(records).map((r) => r.name), ['STARLINK-40096']);
});

test('OMM and TLE of the same element set propagate to the same place', () => {
  // Re-express fixture TLEs as OMM CSV rows and compare SGP4 positions.
  const { records } = parseTle(readFileSync(FIXTURE, 'utf8'));
  const rows: string[] = [HEADER];
  for (const r of records.slice(0, 50)) {
    const l1 = r.line1!;
    const l2 = r.line2!;
    const expo = (s: string) => Number(`${s[0]}.${s.slice(1, 6)}e${s.slice(6, 8)}`);
    const epoch = new Date(tleEpochMs(l1)).toISOString().slice(0, -1);
    rows.push(
      [
        r.name,
        `20${r.intlDesignator.slice(0, 2)}-${r.intlDesignator.slice(2)}`,
        epoch,
        l2.slice(52, 63).trim(),
        `.${l2.slice(26, 33)}`,
        l2.slice(8, 16).trim(),
        l2.slice(17, 25).trim(),
        l2.slice(34, 42).trim(),
        l2.slice(43, 51).trim(),
        '0',
        'U',
        String(r.noradId),
        '999',
        '1',
        String(expo(l1.slice(53, 61))),
        l1.slice(33, 43).trim(),
        String(expo(l1.slice(44, 52))),
      ].join(','),
    );
  }
  const omm = parseOmmCsv(rows.join('\n') + '\n');
  assert.equal(omm.records.length, 50);
  for (let i = 0; i < 50; i++) {
    const a = twoline2satrec(records[i]!.line1!, records[i]!.line2!);
    const b = recordToSatrec(omm.records[i]!);
    assert.equal(b.error, 0);
    assert.equal(omm.records[i]!.intlDesignator, records[i]!.intlDesignator);
    for (const dtMin of [0, 60, 1440]) {
      const date = new Date(satrecEpochMs(a) + dtMin * 60_000);
      const pa = propagate(a, date);
      const pb = propagate(b, date);
      if (pa === null || pb === null) continue;
      const d = Math.hypot(pa.position.x - pb.position.x, pa.position.y - pb.position.y, pa.position.z - pb.position.z);
      // TLE epochs carry ~1 ms of rounding (8 decimals of a day); 7.6 km/s => a few metres.
      assert.ok(d < 0.02, `${records[i]!.name} differs by ${(d * 1000).toFixed(1)} m at +${dtMin} min`);
    }
  }
});

test('fast enough for ~12,000 rows', () => {
  const rows = [HEADER];
  for (let i = 0; i < 12_000; i++) rows.push(SAMPLE.replace(',101142,', `,${110000 + i},`));
  const text = rows.join('\r\n') + '\r\n';
  parseOmmCsv(text); // warm-up
  const t0 = performance.now();
  const { records, rejected } = parseOmmCsv(text);
  const ms = performance.now() - t0;
  assert.equal(records.length, 12_000);
  assert.equal(rejected, 0);
  console.log(`parseOmmCsv: 12,000 rows in ${ms.toFixed(1)} ms`);
  assert.ok(ms < 500, `too slow: ${ms} ms`);
});
