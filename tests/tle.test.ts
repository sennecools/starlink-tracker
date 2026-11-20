import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { decodeCatalogNumber, parseTle, tleChecksumOk, tleEpochMs } from '../server/tle.ts';

const FIXTURE = fileURLToPath(new URL('../fixtures/sup-gp-starlink-partial.tle', import.meta.url));
const fixtureText = readFileSync(FIXTURE, 'utf8');

const NAME = 'STARLINK-1008';
const L1 = '1 44714C 19074B   26324.14631944  .00089668  00000+0  70460-3 0  3248';
const L2 = '2 44714  53.1440  28.8754 0002857  13.6685 356.4349 15.70790074    19';

/** Recompute the modulo-10 checksum of a 69-char line. */
function withChecksum(line: string): string {
  let sum = 0;
  for (let i = 0; i < 68; i++) {
    const c = line[i]!;
    if (c >= '0' && c <= '9') sum += Number(c);
    else if (c === '-') sum += 1;
  }
  return line.slice(0, 68) + String(sum % 10);
}

function withCatalog(line: string, cat: string): string {
  return withChecksum(line.slice(0, 2) + cat + line.slice(7));
}

test('real fixture: 853 records, 0 rejected', () => {
  const { records, rejected } = parseTle(fixtureText);
  assert.equal(records.length, 853);
  assert.equal(rejected, 0);
  const r = records[0]!;
  assert.deepEqual(r, { name: NAME, noradId: 44714, intlDesignator: '19074B', line1: L1, line2: L2 });
  for (const rec of records) {
    assert.equal(rec.line1?.length, 69);
    assert.equal(rec.line2?.length, 69);
    assert.equal(rec.omm, undefined);
    assert.ok(rec.name.startsWith('STARLINK-'));
  }
});

test('checksum helper', () => {
  assert.ok(tleChecksumOk(L1));
  assert.ok(tleChecksumOk(L2));
  assert.ok(!tleChecksumOk(L1.slice(0, 68) + '0'));
  assert.ok(!tleChecksumOk(L1.slice(0, 60)));
});

test('CRLF, trailing spaces, blank lines and BOM', () => {
  const text = '﻿\r\n\r\n' + `${NAME}   \r\n${L1}  \r\n\r\n${L2}\t \r\n\r\n   \r\n${NAME}\r\n${L1}\r\n${L2}\r\n`;
  const { records, rejected } = parseTle(text);
  assert.equal(rejected, 0);
  assert.equal(records.length, 2);
  assert.equal(records[0]!.name, NAME);
  assert.equal(records[0]!.line1, L1);
  assert.equal(records[0]!.line2, L2);
});

test('bare 2-line sets get a NORAD fallback name, mixed with 3-line sets', () => {
  const text = `${L1}\n${L2}\n${NAME}\n${L1}\n${L2}\n${L1}\n${L2}`;
  const { records, rejected } = parseTle(text);
  assert.equal(rejected, 0);
  assert.deepEqual(records.map((r) => r.name), ['NORAD 44714', NAME, 'NORAD 44714']);
});

test('3LE "0 " name prefix is stripped', () => {
  const { records } = parseTle(`0 ${NAME}\n${L1}\n${L2}\n`);
  assert.equal(records[0]!.name, NAME);
});

test('truncated final record is dropped and counted', () => {
  const lines = fixtureText.trimEnd().split('\n');
  const truncated = lines.slice(0, -1).join('\n'); // last record lacks line 2
  const a = parseTle(truncated);
  assert.equal(a.records.length, 852);
  assert.equal(a.rejected, 1);
  // Name-only tail (cut right after a name line) is dropped silently.
  const b = parseTle(lines.slice(0, -2).join('\n'));
  assert.equal(b.records.length, 852);
  assert.equal(b.rejected, 0);
  // Half a line 1 at the very end.
  const c = parseTle(lines.slice(0, -2).join('\n') + '\n' + lines.at(-2)!.slice(0, 30));
  assert.equal(c.records.length, 852);
  assert.equal(c.rejected, 1);
});

test('bad checksum, mismatched catalog numbers and orphan lines are rejected', () => {
  const badL2 = L2.slice(0, 10) + '9' + L2.slice(11); // alters a digit, checksum now wrong
  const otherL2 = withCatalog(L2, '44715');
  const text = [NAME, L1, badL2, NAME, L1, otherL2, L2, NAME, L1, NAME, L1, L2].join('\n');
  const { records, rejected } = parseTle(text);
  assert.equal(records.length, 1);
  assert.equal(records[0]!.name, NAME);
  // bad checksum, mismatch, orphan line 2, line 1 followed by a name line
  assert.equal(rejected, 4);
});

test('Alpha-5 catalog numbers', () => {
  assert.equal(decodeCatalogNumber('A0001'), 100001);
  assert.equal(decodeCatalogNumber('H9999'), 179999);
  assert.equal(decodeCatalogNumber('J0000'), 180000); // I skipped
  assert.equal(decodeCatalogNumber('P0000'), 230000); // O skipped
  assert.equal(decodeCatalogNumber('Z9999'), 339999);
  assert.equal(decodeCatalogNumber('44714'), 44714);
  assert.equal(decodeCatalogNumber('    5'), 5);
  assert.ok(Number.isNaN(decodeCatalogNumber('I0001')));
  assert.ok(Number.isNaN(decodeCatalogNumber('O0001')));
  assert.ok(Number.isNaN(decodeCatalogNumber('A001')));
  assert.ok(Number.isNaN(decodeCatalogNumber('4A714')));

  const a1 = withCatalog(L1, 'A0001');
  const a2 = withCatalog(L2, 'A0001');
  const { records, rejected } = parseTle(`STARLINK-X\n${a1}\n${a2}\n${a1}\n${a2}\n`);
  assert.equal(rejected, 0);
  assert.equal(records[0]!.noradId, 100001);
  assert.equal(records[1]!.name, 'NORAD 100001');
});

test('epoch decoding', () => {
  // 26324.14631944 => 2026, day 324 (Nov 20) + 0.14631944 d
  const ms = tleEpochMs(L1);
  assert.equal(new Date(ms).toISOString().slice(0, 16), '2026-11-20T03:30');
});

test('fast enough for ~12,800 records', () => {
  const big = Array.from({ length: 15 }, () => fixtureText.trimEnd()).join('\n');
  parseTle(big); // warm-up
  const t0 = performance.now();
  const { records, rejected } = parseTle(big);
  const ms = performance.now() - t0;
  assert.equal(records.length, 853 * 15);
  assert.equal(rejected, 0);
  assert.ok(ms < 250, `parse took ${ms.toFixed(1)} ms`);
});
