/**
 * Selected-satellite info panel (SpaceX "spec table" style).
 *
 * Desktop: a ~420px right column below the top bar, on a right-to-left black
 * scrim (no box). Mobile (< 700px): a bottom sheet, max 60vh, scrolling inside.
 *
 * Everything is built once; `show()` rewrites the static rows (once per
 * selection) and `update()` only touches the live cells whose rounded value
 * changed, through per-glyph tabular cells (see `TabularText`), so it is cheap
 * enough to call every frame although ~4 Hz is plenty.
 *
 * This file also hosts the small DOM helpers shared with hud.ts.
 */

import type { InfoPanelApi, PanelLink, SatLive, SatStatic } from '../shared/phase2.ts';
import { MIN_ELEVATION_DEG } from '../shared/phase2.ts';
import { SHELLS } from '../shared/shells.ts';
import {
  DASH,
  fixed,
  formatAge,
  formatApsides,
  formatDeg,
  formatGroupedFixed,
  formatLat,
  formatLaserTerminals,
  formatLaunch,
  formatLon,
  formatOnOff,
  formatUtcStamp,
  groupThousands,
  hardwareNote,
  linkKindLabel,
  rgbToCss,
} from './format.ts';

// ---------------------------------------------------------------------------
// Shared DOM helpers (construction time only, except TabularText.set)
// ---------------------------------------------------------------------------

export function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className: string,
  parent?: Node,
  text?: string,
): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (className) e.className = className;
  if (text !== undefined) e.textContent = text;
  if (parent) parent.appendChild(e);
  return e;
}

/** A dedicated Text node: updating `.data` is the cheapest DOM text write. */
export function textNode(parent: Node, initial = ''): Text {
  const t = document.createTextNode(initial);
  parent.appendChild(t);
  return t;
}

export function setData(t: Text, s: string): void {
  if (t.data !== s) t.data = s;
}

const SVG_NS = 'http://www.w3.org/2000/svg';

/**
 * Thin right arrow (the D-DIN webfont only covers ASCII, so "→" would fall
 * back to a system font; an SVG keeps the stroke weight consistent).
 */
export function createArrow(className: string): SVGSVGElement {
  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('class', className);
  svg.setAttribute('viewBox', '0 0 14 10');
  svg.setAttribute('aria-hidden', 'true');
  svg.setAttribute('focusable', 'false');
  const p = document.createElementNS(SVG_NS, 'path');
  p.setAttribute('d', 'M0 5h12.5M8.5 1l4 4-4 4');
  p.setAttribute('fill', 'none');
  p.setAttribute('stroke', 'currentColor');
  p.setAttribute('stroke-width', '1.25');
  svg.appendChild(p);
  return svg;
}

/** Small round separator dot (consistent vertical centring, unlike "·" in a fallback font). */
export function sepDot(parent: Node): HTMLSpanElement {
  const s = el('span', 'sx-sep', parent);
  s.setAttribute('aria-hidden', 'true');
  return s;
}

function isDigit(code: number): boolean {
  return code >= 48 && code <= 57;
}

/**
 * Text with jitter-free digits.
 *
 * D-DIN-PRO has proportional figures and no `tnum` feature (and the webfont
 * only covers ASCII), so `font-variant-numeric: tabular-nums` does nothing.
 * Every character gets its own inline span instead; digits sit in fixed-width
 * cells (`.sx-tnum > .d`, width var(--sx-tnum-w)). `set()` only rewrites the
 * cells whose character changed and only adds/removes spans when the length
 * changes, so a ticking value costs one or two `Text.data` writes.
 */
export class TabularText {
  readonly host: HTMLElement;
  private readonly cells: Text[] = [];
  private readonly spans: HTMLSpanElement[] = [];
  private value = '';

  constructor(host: HTMLElement, initial = '') {
    this.host = host;
    host.classList.add('sx-tnum');
    if (initial) this.set(initial);
  }

  get text(): string {
    return this.value;
  }

  set(s: string): void {
    if (s === this.value) return;
    const prev = this.value;
    this.value = s;
    const n = s.length;
    // grow / shrink the cell pool (rare: only when the digit count changes)
    while (this.spans.length < n) {
      const span = document.createElement('span');
      const t = document.createTextNode('');
      span.appendChild(t);
      this.host.appendChild(span);
      this.spans.push(span);
      this.cells.push(t);
    }
    while (this.spans.length > n) {
      (this.spans.pop() as HTMLSpanElement).remove();
      this.cells.pop();
    }
    for (let i = 0; i < n; i++) {
      const code = s.charCodeAt(i);
      if (i < prev.length && prev.charCodeAt(i) === code && this.cells[i].data !== '') continue;
      const t = this.cells[i];
      t.data = s[i];
      const digit = isDigit(code);
      const span = this.spans[i];
      if ((span.className === 'd') !== digit) span.className = digit ? 'd' : '';
    }
  }
}

const REDUCED_MOTION = '(prefers-reduced-motion: reduce)';

export function prefersReducedMotion(): boolean {
  return typeof matchMedia === 'function' && matchMedia(REDUCED_MOTION).matches;
}

// ---------------------------------------------------------------------------
// Info panel
// ---------------------------------------------------------------------------

export interface InfoPanelOptions {
  /** Called when the panel opens / closes (the HUD uses it to make room). */
  onOpenChange?: (open: boolean) => void;
}

interface SpecRow {
  row: HTMLDivElement;
  value: TabularText;
  note: Text | null;
  noteEl: HTMLElement | null;
}

/** Transition length of the enter / leave animation (keep in sync with hud.css). */
const PANEL_MS = 450;

/**
 * Create the info panel and mount it into `root` (normally the HUD root, so it
 * shares its stacking context and pointer-events rules).
 *
 * Behaviour:
 *  - `show(info)` opens (or re-targets) the panel; `show(null)` closes it.
 *  - The FOCUS button calls `onFocus`, which flips to RELEASE (`onRelease`)
 *    once the integrator calls `setTracking(true)`.
 *  - CLOSE, the corner "×" and the Escape key hide the panel and call `onClose`
 *    (only `onClose`: the integrator should deselect and release the camera
 *    there; `setTracking` is reset to false by the next `show()` of another
 *    satellite).
 *  - `setHardware` / `setLinks` must be called AFTER `show()` of a new
 *    satellite: `show()` of a different satellite resets the hardware rows to
 *    a dash and hides the LASER LINKS group until the next `setLinks`.
 *  - `setLinks` is cheap to call repeatedly (~4 Hz): rows come from a reused
 *    pool; while the neighbour set (index, kind, name, in order) is unchanged
 *    only the distance cells whose whole-km value changed are rewritten.
 *  - A link row click / Enter / Space calls `onSelectNeighbor(index)`.
 *  - The COVERAGE AREA row (live telemetry group) is an aria-pressed toggle:
 *    a click flips it optimistically and calls `onCoverageToggle(on)`;
 *    `setCoverageVisible(on)` sets it silently. Default ON; kept across selections.
 *  - `setHardware` with a V3 label adds the secondary "ARTIST'S APPROXIMATION"
 *    line under the hardware value (no official V3 drawings exist).
 */
export function createInfoPanel(root: HTMLElement, opts: InfoPanelOptions = {}): InfoPanelApi {
  const reduced = prefersReducedMotion();

  const panel = el('aside', 'sx-panel', root);
  panel.setAttribute('aria-label', 'Selected satellite');
  panel.hidden = true;
  el('div', 'sx-panel__scrim', panel).setAttribute('aria-hidden', 'true');
  const body = el('div', 'sx-panel__body', panel);

  const closeX = el('button', 'sx-close', body);
  closeX.type = 'button';
  closeX.setAttribute('aria-label', 'Close satellite details');
  closeX.title = 'Close';

  const scroll = el('div', 'sx-panel__scroll', body);

  el('div', 'sx-kicker', scroll, 'Selected satellite');
  const nameEl = el('h2', 'sx-panel__name', scroll);
  const nameText = textNode(nameEl, '');
  const ids = el('div', 'sx-panel__ids', scroll);
  el('span', '', ids, 'NORAD ');
  const noradText = textNode(el('span', '', ids), DASH);
  sepDot(ids);
  el('span', '', ids, 'COSPAR ');
  const cosparText = textNode(el('span', '', ids), DASH);

  // polite announcement of the selection for screen readers (live numbers are not announced);
  // a sibling of the panel, never under its [hidden], so it is in the a11y tree before it changes
  const announce = el('div', 'sx-vh', root);
  announce.setAttribute('aria-live', 'polite');
  const announceText = textNode(announce, '');

  const addTable = (title: string): HTMLDListElement => {
    el('div', 'sx-panel__group', scroll, title);
    return el('dl', 'sx-spec', scroll);
  };

  const addRow = (table: HTMLDListElement, label: string, unit = '', withNote = false, labelNote = ''): SpecRow => {
    const row = el('div', 'sx-spec__row', table);
    const dt = el('dt', 'sx-spec__k', row, label);
    if (labelNote) el('span', 'sx-spec__knote', dt, labelNote);
    const dd = el('dd', 'sx-spec__v', row);
    const line = el('span', 'sx-spec__line', dd);
    const value = new TabularText(el('span', 'sx-spec__num', line), DASH);
    if (unit) el('span', 'sx-unit', line, unit);
    let note: Text | null = null;
    let noteEl: HTMLElement | null = null;
    if (withNote) {
      noteEl = el('span', 'sx-spec__note', dd);
      note = textNode(noteEl, '');
    }
    return { row, value, note, noteEl };
  };

  const liveTable = addTable('Live telemetry');
  const rAlt = addRow(liveTable, 'Altitude', 'KM');
  const rVel = addRow(liveTable, 'Velocity', 'KM/S');
  const rLatLon = addRow(liveTable, 'Lat / Lon');
  // second tabular value in the lat/lon row
  const lonHost = el('span', 'sx-spec__num sx-spec__num--2', rLatLon.value.host.parentElement as HTMLElement);
  const rLon = new TabularText(lonHost, '');
  const rStatus = addRow(liveTable, 'Status');
  rStatus.row.classList.add('sx-spec__row--status');
  const statusDot = el('span', 'sx-status-dot', rStatus.value.host.parentElement as HTMLElement);
  statusDot.setAttribute('aria-hidden', 'true');
  (rStatus.value.host.parentElement as HTMLElement).prepend(statusDot);
  const rCov = addRow(liveTable, 'Coverage radius', 'KM', false, `${MIN_ELEVATION_DEG}° min. elevation`);

  // COVERAGE AREA toggle: a spec row whose value cell is a small text toggle
  // (the whole row is the hit target via .sx-toggle::after). A global
  // preference, so show() does not reset it.
  const covRow = el('div', 'sx-spec__row sx-spec__row--toggle', liveTable);
  el('dt', 'sx-spec__k', covRow, 'Coverage area');
  const covBtn = el('button', 'sx-toggle', el('dd', 'sx-spec__v', covRow));
  covBtn.type = 'button';
  el('span', 'sx-vh', covBtn, 'Coverage area ');
  el('span', 'sx-toggle__mark', covBtn).setAttribute('aria-hidden', 'true');
  const covState = textNode(el('span', 'sx-toggle__text', covBtn), '');

  const orbitTable = addTable('Orbit');
  const rShell = addRow(orbitTable, 'Shell');
  const shellSwatch = el('span', 'sx-swatch', rShell.value.host.parentElement as HTMLElement);
  shellSwatch.setAttribute('aria-hidden', 'true');
  (rShell.value.host.parentElement as HTMLElement).prepend(shellSwatch);
  const rInc = addRow(orbitTable, 'Inclination');
  const rPeriod = addRow(orbitTable, 'Period', 'MIN');
  const rApsides = addRow(orbitTable, 'Apogee / Perigee', 'KM');
  const rLaunch = addRow(orbitTable, 'Launch');
  const rEpoch = addRow(orbitTable, 'Element epoch', '', true);

  // Hardware (estimated from the launch date: SpaceX publishes no per-satellite
  // list) and the modelled laser links. Group order in the panel:
  // LIVE TELEMETRY, HARDWARE, LASER LINKS · MODELED, ORBIT.
  const hwGroup = el('div', 'sx-panel__group', undefined, 'Hardware');
  const hwTable = el('dl', 'sx-spec');
  const rBus = addRow(hwTable, 'Hardware', '(EST.)', true, 'Estimated from launch date');
  // "ARTIST'S APPROXIMATION" under the value, V3 only (no official drawings exist)
  if (rBus.noteEl) {
    rBus.noteEl.classList.add('sx-spec__note--tag');
    rBus.noteEl.hidden = true;
  }
  const rLasers = addRow(hwTable, 'Laser terminals');

  const linkGroup = el('div', 'sx-panel__group');
  el('span', '', linkGroup, 'Laser links');
  sepDot(linkGroup);
  el('span', '', linkGroup, 'Modeled');
  linkGroup.title = 'Modeled topology: SpaceX does not publish which satellites are linked';
  const linkList = el('ul', 'sx-links');
  linkList.setAttribute('aria-label', 'Modeled laser links');
  const linkEmpty = el('div', 'sx-links__empty', undefined, 'No active links');
  const orbitGroupEl = orbitTable.previousElementSibling as HTMLElement;
  for (const node of [hwGroup, hwTable, linkGroup, linkList, linkEmpty]) scroll.insertBefore(node, orbitGroupEl);
  linkGroup.hidden = true;
  linkList.hidden = true;
  linkEmpty.hidden = true;

  const foot = el('div', 'sx-panel__foot', body);
  const focusBtn = el('button', 'sx-btn sx-btn--primary', foot);
  focusBtn.type = 'button';
  const focusLabel = textNode(el('span', 'sx-btn__label', focusBtn), 'View up close');
  focusBtn.appendChild(createArrow('sx-btn__arrow'));
  const closeBtn = el('button', 'sx-btn sx-btn--ghost', foot);
  closeBtn.type = 'button';
  el('span', 'sx-btn__label', closeBtn, 'Close');

  // ---- state -------------------------------------------------------------------
  let open = false;
  let current: SatStatic | null = null;
  let tracking = false;
  let hideTimer = 0;

  let kAlt = Number.NaN;
  let kVel = Number.NaN;
  let kLat = Number.NaN;
  let kLon = Number.NaN;
  let kShadow: boolean | null = null;
  let kCov = Number.NaN;
  let kAgeMin = Number.NaN;

  let hwBus = '';
  let hwLasers = Number.NaN;

  let coverageOn = true;

  interface LinkRow {
    li: HTMLLIElement;
    button: HTMLButtonElement;
    name: Text;
    kindText: Text;
    dist: TabularText;
    /** Catalogue index of the neighbour (-1 = unused pool row). */
    index: number;
    kind: number;
    nameValue: string;
    /** Last displayed distance in whole km (-1 = non-finite, NaN = forced refresh). */
    kDist: number;
  }
  /** Row pool; only the first `linkCount` rows are shown. */
  const linkRows: LinkRow[] = [];
  const rowByButton = new WeakMap<Element, LinkRow>();
  let linkCount = 0;
  /** Whether `setLinks` has been called since the current satellite was shown. */
  let linksShown = false;
  /** Set while a link row starts a neighbour hop: show() keeps the scroll position. */
  let keepScroll = false;

  const makeLinkRow = (): LinkRow => {
    const li = el('li', 'sx-links__item', linkList);
    const button = el('button', 'sx-link', li);
    button.type = 'button';
    const name = textNode(el('span', 'sx-link__name', button), '');
    const right = el('span', 'sx-link__right', button);
    const kindText = textNode(el('span', 'sx-link__kind', right), '');
    const distLine = el('span', 'sx-link__dist', right);
    const dist = new TabularText(el('span', '', distLine), DASH);
    el('span', 'sx-unit', distLine, 'KM');
    const row: LinkRow = { li, button, name, kindText, dist, index: -1, kind: -1, nameValue: '', kDist: Number.NaN };
    rowByButton.set(button, row);
    return row;
  };

  const resetLinks = (): void => {
    linksShown = false;
    linkGroup.hidden = true;
    linkList.hidden = true;
    linkEmpty.hidden = true;
  };

  const setHardwareNote = (bus: string): void => {
    const note = hardwareNote(bus);
    if (rBus.note) setData(rBus.note, note);
    if (rBus.noteEl) rBus.noteEl.hidden = note === '';
  };

  const resetHardware = (): void => {
    hwBus = '';
    hwLasers = Number.NaN;
    rBus.value.set(DASH);
    rLasers.value.set(DASH);
    setHardwareNote('');
  };

  const applyCoverage = (on: boolean): void => {
    coverageOn = on;
    covBtn.setAttribute('aria-pressed', on ? 'true' : 'false');
    covBtn.title = on ? 'Hide the coverage area on the ground' : 'Show the coverage area on the ground';
    setData(covState, formatOnOff(on));
  };
  applyCoverage(true);
  covBtn.addEventListener('click', () => {
    const next = !coverageOn;
    applyCoverage(next); // optimistic; the integrator may re-send setCoverageVisible
    api.onCoverageToggle?.(next);
  });

  const resetLive = (): void => {
    kAlt = kVel = kLat = kLon = kCov = kAgeMin = Number.NaN;
    kShadow = null;
    rAlt.value.set(DASH);
    rVel.value.set(DASH);
    rLatLon.value.set(DASH);
    rLon.set('');
    rStatus.value.set(DASH);
    statusDot.dataset.state = '';
    rCov.value.set(DASH);
  };

  const setAge = (nowMs: number): void => {
    if (!current) return;
    const age = nowMs - current.epochMs;
    const bucket = Math.floor(age / 60_000);
    if (bucket === kAgeMin) return;
    kAgeMin = bucket;
    if (rEpoch.note) setData(rEpoch.note, formatAge(age));
  };

  const setTrackingUi = (t: boolean): void => {
    tracking = t;
    setData(focusLabel, t ? 'Release' : 'View up close');
    focusBtn.classList.toggle('is-tracking', t);
    focusBtn.title = t ? 'Stop following this satellite' : 'Fly to a close-up 3D view of this satellite';
  };
  setTrackingUi(false);

  const finishHide = (): void => {
    if (!open) panel.hidden = true;
  };
  panel.addEventListener('transitionend', (e) => {
    if (e.target === panel && e.propertyName === 'opacity') finishHide();
  });

  const setOpen = (next: boolean): void => {
    if (next === open) return;
    open = next;
    clearTimeout(hideTimer);
    if (next) {
      panel.hidden = false;
      // commit the hidden state before flipping the class so the transition runs
      void panel.offsetWidth;
      panel.classList.add('is-open');
    } else {
      if (panel.contains(document.activeElement)) (document.activeElement as HTMLElement).blur();
      panel.classList.remove('is-open');
      hideTimer = window.setTimeout(finishHide, reduced ? 0 : PANEL_MS + 80);
    }
    opts.onOpenChange?.(next);
  };

  /** Hides the panel itself (works even without a listener), then tells the integrator. */
  const requestClose = (): void => {
    api.show(null);
    api.onClose?.();
  };

  focusBtn.addEventListener('click', () => {
    if (tracking) api.onRelease?.();
    else api.onFocus?.();
  });
  closeBtn.addEventListener('click', requestClose);
  // one delegated listener for all link rows (buttons: Enter / Space activate natively)
  linkList.addEventListener('click', (ev) => {
    const button = (ev.target as Element | null)?.closest('.sx-link');
    const row = button ? rowByButton.get(button) : undefined;
    if (!row || row.index < 0) return;
    // The hop is synchronous (show() hides the list, setLinks() shows it again),
    // so keeping the scroll position keeps the list and the focused row in view.
    keepScroll = true;
    try {
      api.onSelectNeighbor?.(row.index);
    } finally {
      keepScroll = false;
    }
  });
  closeX.addEventListener('click', requestClose);
  window.addEventListener('keydown', (e) => {
    if (!open || e.key !== 'Escape' || e.defaultPrevented) return;
    e.preventDefault();
    requestClose();
  });

  const api: InfoPanelApi = {
    show(info) {
      if (!info) {
        current = null;
        setData(announceText, '');
        setOpen(false);
        return;
      }
      // By NORAD id only: a catalogue hot swap that moves the satellite's index
      // is not a new selection (stale link rows stay unclickable until the next
      // links update, see SelectionController.onPanelNeighbor).
      const sameSat = current !== null && current.noradId === info.noradId;
      current = info;

      // The name never wraps ("STARLINK-" / "30193" looked broken); the font
      // size steps down with length instead (D-DIN-PRO here is ASCII-only, so
      // a U+2011 non-breaking hyphen would fall back to another font).
      const name = info.name.toUpperCase();
      setData(nameText, name);
      nameEl.dataset.len = name.length > 20 ? 'xl' : name.length > 12 ? 'l' : '';
      setData(noradText, `${info.noradId}`);
      setData(cosparText, info.cosparId || DASH);

      const shell = SHELLS[info.shellIndex];
      shellSwatch.style.setProperty('--c', shell ? rgbToCss(shell.color) : 'transparent');
      rShell.value.set(info.shellLabel || DASH);
      rInc.value.set(formatDeg(info.inclinationDeg, 2));
      rPeriod.value.set(fixed(info.periodMin, 1));
      rApsides.value.set(formatApsides(info.apogeeKm, info.perigeeKm));
      rLaunch.value.set(formatLaunch(info.launchYear, info.launchNumber));
      rEpoch.value.set(formatUtcStamp(new Date(info.epochMs)));

      if (!sameSat) {
        resetLive();
        resetHardware();
        resetLinks();
        setTrackingUi(false);
        // written after the panel is shown (and cleared first, so selecting the same name again is a change)
        setData(announceText, '');
        const msg = `Selected ${name}`;
        requestAnimationFrame(() => {
          if (current === info) setData(announceText, msg);
        });
        setAge(Date.now());
        if (!keepScroll) {
          if (open && !reduced && typeof scroll.animate === 'function') {
            scroll.animate([{ opacity: 0.2 }, { opacity: 1 }], { duration: 320, easing: 'cubic-bezier(.2,.7,.2,1)' });
          }
          scroll.scrollTop = 0;
        }
      }
      setOpen(true);
    },

    update(live: SatLive) {
      if (!current) return;
      const alt = Math.round(live.altitudeKm * 10);
      if (alt !== kAlt) {
        kAlt = alt;
        rAlt.value.set(formatGroupedFixed(live.altitudeKm, 1));
      }
      const vel = Math.round(live.speedKmS * 100);
      if (vel !== kVel) {
        kVel = vel;
        rVel.value.set(fixed(live.speedKmS, 2));
      }
      const lat = Math.round(live.latDeg * 100);
      if (lat !== kLat) {
        kLat = lat;
        rLatLon.value.set(formatLat(live.latDeg, 2));
      }
      const lon = Math.round(live.lonDeg * 100);
      if (lon !== kLon) {
        kLon = lon;
        rLon.set(formatLon(live.lonDeg, 2));
      }
      if (live.inShadow !== kShadow) {
        kShadow = live.inShadow;
        rStatus.value.set(live.inShadow ? 'IN EARTH SHADOW' : 'SUNLIT');
        statusDot.dataset.state = live.inShadow ? 'shadow' : 'sun';
      }
      const cov = Math.round(live.footprintRadiusKm);
      if (cov !== kCov) {
        kCov = cov;
        rCov.value.set(groupThousands(live.footprintRadiusKm));
      }
      setAge(live.timeMs);
    },

    setTracking(t) {
      if (t !== tracking) setTrackingUi(t);
    },

    setHardware(busLabel, laserTerminals) {
      const bus = busLabel.trim().toUpperCase();
      if (bus !== hwBus) {
        hwBus = bus;
        rBus.value.set(bus || DASH);
        setHardwareNote(bus);
      }
      if (laserTerminals !== hwLasers) {
        hwLasers = laserTerminals;
        rLasers.value.set(formatLaserTerminals(laserTerminals));
      }
    },

    setLinks(links: readonly PanelLink[]) {
      const n = links.length;
      if (!linksShown) {
        linksShown = true;
        linkGroup.hidden = false;
        linkCount = -1; // force the visibility of list / empty line below
      }
      // same neighbour set (in order)? then only the distances change
      let same = n === linkCount;
      if (same) {
        for (let i = 0; i < n; i++) {
          const l = links[i];
          const r = linkRows[i];
          if (l.index !== r.index || l.kind !== r.kind || l.name !== r.nameValue) {
            same = false;
            break;
          }
        }
      }
      if (!same) {
        while (linkRows.length < n) linkRows.push(makeLinkRow());
        for (let i = 0; i < linkRows.length; i++) {
          const r = linkRows[i];
          if (i < n) {
            const l = links[i];
            r.index = l.index;
            r.kind = l.kind;
            r.nameValue = l.name;
            r.kDist = Number.NaN;
            const upper = l.name.toUpperCase();
            setData(r.name, upper);
            setData(r.kindText, linkKindLabel(l.kind));
            r.button.title = `Select ${upper}`;
            r.li.hidden = false;
          } else {
            r.index = -1;
            r.li.hidden = true;
          }
        }
        linkCount = n;
        linkList.hidden = n === 0;
        linkEmpty.hidden = n !== 0;
        // Keyboard focus on a row that is now unused (fewer neighbours): move it
        // to the first row (or FOCUS); otherwise keep the focused row in view.
        const f = document.activeElement;
        if (f !== null && linkList.contains(f)) {
          const fr = rowByButton.get(f);
          if (fr === undefined || fr.index < 0) (n > 0 ? linkRows[0].button : focusBtn).focus();
          else if (typeof (f as HTMLElement).scrollIntoView === 'function') (f as HTMLElement).scrollIntoView({ block: 'nearest' });
        }
      }
      for (let i = 0; i < n; i++) {
        const km = links[i].distanceKm;
        const key = Number.isFinite(km) ? Math.round(km) : -1;
        const r = linkRows[i];
        if (key === r.kDist) continue;
        r.kDist = key;
        r.dist.set(groupThousands(km));
      }
    },

    setCoverageVisible(on) {
      if (on !== coverageOn) applyCoverage(on);
    },

    onFocus: null,
    onRelease: null,
    onClose: null,
    onSelectNeighbor: null,
    onCoverageToggle: null,
  };

  return api;
}
