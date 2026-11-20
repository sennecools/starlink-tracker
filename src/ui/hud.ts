/**
 * HUD / overlay UI in the spacex.com design language: D-DIN-PRO, black, one
 * text colour at two alphas, hairlines, no boxes; chrome is limited to black
 * scrims behind text.
 *
 * Pure DOM + TypeScript (no Cesium). Everything is built once in `createHud`
 * and cached; the per-frame / per-tick setters only touch `Text.data`,
 * attributes or a transform of cached nodes, and only when the displayed value
 * changed (numeric early-outs before any string formatting), so calling them
 * every frame is effectively free.
 *
 * Layout (desktop):            top-left wordmark            top-right  LIVE | clock (perf below, toggled with `)
 *                               notice line under the top bar
 *   bottom-left hero count      top-centre hint             bottom-right shell legend + LASER LINKS toggle + camera readout
 *   right column: info panel (hud.infoPanel), the legend/camera fade out while it is open.
 */

import '../styles/fonts.css';
import '../styles/hud.css';

import type { InfoPanelApi, LaserHudInfo } from '../shared/phase2.ts';
import { SHELLS } from '../shared/shells.ts';
import {
  formatAge,
  formatAltitudeKm,
  formatDataSources,
  formatLaserEquipped,
  formatLat,
  formatLinkCount,
  formatLocalHHMM,
  formatLon,
  formatPerf,
  formatUtcDate,
  formatUtcTime,
  groupThousands,
  perfTier,
  rgbToCss,
  shellNote,
  splitDots,
  splitLabel,
} from './format.ts';
import { createArrow, createInfoPanel, el, prefersReducedMotion, sepDot, setData, TabularText, textNode } from './infoPanel.ts';

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export interface HudDataInfo {
  count: number;
  /** Primary source name; shown as-is when `sources` is absent or empty. */
  source: string;
  /**
   * Contributing sources, primary first (e.g. ["CelesTrak GP", "Recent launches"]).
   * Shown compactly as "CELESTRAK GP + RECENT LAUNCHES" (trailing parentheticals
   * dropped, duplicates removed, more than three -> "A + B + 2 MORE").
   */
  sources?: readonly string[];
  fetchedAt: Date;
  /**
   * True only while the primary (full catalogue) download has never completed;
   * drives the "PARTIAL SNAPSHOT · FULL CATALOGUE AT HH:MM" segment. A complete
   * primary plus a small supplementary feed is not partial.
   */
  partial: boolean;
  nextRefresh: Date | null;
  /**
   * Progress of a running partitioned full-catalogue refresh (absent / null =
   * none). Once `nextRefresh` has passed it replaces "Full catalogue refresh
   * due" with "Full catalogue loading DONE/TOTAL".
   */
  cycle?: { done: number; total: number } | null;
}

export interface HudPerfInfo {
  fps: number;
  frameMs: number;
  resolutionScale: number;
  msaa: number;
  /** Worker time of the latest keyframe / links request (ms; NaN or absent = unknown). */
  keyframeMs?: number;
  linksMs?: number;
}

export interface HudCameraInfo {
  altitudeKm: number;
  lat: number;
  lon: number;
}

export interface HudShellInfo {
  index: number;
  label: string;
  color: readonly [number, number, number];
  count: number;
  visible: boolean;
  /**
   * Optional secondary sub-label under the row. When absent, a built-in note
   * by shell id is used (V3 · STARSHIP -> "NEW GENERATION · LAUNCHED ON STARSHIP").
   */
  note?: string;
}

export interface Hud {
  /** Full-screen boot screen. `progress` 0..1, or null for an indeterminate sweep. */
  setBoot(progress: number | null, status: string, detail?: string): void;
  /** Fade the boot screen out and reveal the HUD (idempotent). */
  finishBoot(): void;
  /** Fatal error overlay (with a working RELOAD button). */
  showError(title: string, message: string): void;
  /** UTC clock. Safe to call every frame: early-outs unless the displayed second changed. */
  setClock(date: Date): void;
  /** LIVE indicator (pulsing dot) vs "SIMULATED". */
  setLive(isRealtime: boolean): void;
  /** Hero sub-line: source(s) · catalogue age · partial-snapshot notice (see HudDataInfo). */
  setDataInfo(info: HudDataInfo): void;
  /** Hero count (visible satellites; "/ total" appears when shells are hidden). */
  setTracked(visible: number, total: number): void;
  /** Perf readout (hidden until the user presses the backquote key). Call ~2x/s. */
  setPerf(info: HudPerfInfo): void;
  /** Bottom-right camera readout. Safe to call every frame. */
  setCamera(info: HudCameraInfo): void;
  /** Shell legend; each row is a toggle button (aria-pressed). */
  setShells(shells: HudShellInfo[]): void;
  /** Register a listener for legend toggles (several listeners allowed). */
  onShellToggle(cb: (index: number, visible: boolean) => void): void;
  /**
   * LASER LINKS toggle row under the shell legend (hidden until the first
   * call). Shows the drawn link count (or OFF), the estimated number of
   * laser-equipped satellites and a "modeled topology" disclaimer. Cheap to
   * call every tick: early-outs on unchanged values.
   */
  setLaserLinks(info: LaserHudInfo): void;
  /** Register a listener for the LASER LINKS toggle (several listeners allowed). */
  onLaserToggle(cb: (enabled: boolean) => void): void;
  /** Slim dismissable notice line under the top bar; null hides it. */
  setWarning(text: string | null): void;
  /**
   * Hover label near the cursor. `text` may be "NAME" or "NAME\nSHELL" (the
   * second line renders smaller / secondary); null hides it. `x`, `y` are CSS
   * pixels relative to the viewport (= Cesium window coordinates while the
   * canvas fills the window). Positioned with a transform only; cheap to call
   * on every pointer move or frame.
   */
  setHoverLabel(text: string | null, x: number, y: number): void;
  /** Fade out the interaction hint (also happens on the first pointerdown / wheel after boot). */
  dismissHint(): void;
  /** The selected-satellite panel (mounted inside the HUD root). */
  readonly infoPanel: InfoPanelApi;
}

// ---------------------------------------------------------------------------
// constants
// ---------------------------------------------------------------------------

const COMPACT = '(max-width: 699.98px)';
const COARSE = '(pointer: coarse)';
const SHORT = '(max-height: 559.98px)';
/** Hover label offset from the cursor, CSS px. */
const HOVER_OFFSET = 14;
/** Keep the hover label this far from the viewport edges. */
const HOVER_MARGIN = 8;

function mq(query: string): boolean {
  return typeof matchMedia === 'function' && matchMedia(query).matches;
}

/**
 * Span with `text`, its " · " separators drawn as round dots (construction
 * time only; U+00B7 would fall back to a system font in the ASCII-only D-DIN).
 */
function dotted(className: string, parent: Node, text: string): HTMLSpanElement {
  const span = el('span', className, parent);
  const parts = splitDots(text);
  parts.forEach((p, i) => {
    if (i > 0) sepDot(span);
    span.appendChild(document.createTextNode(p));
  });
  return span;
}

/** Legend sub-label of a row: explicit `note`, else the built-in one for the shell id. */
function noteFor(s: HudShellInfo): string {
  return s.note ?? shellNote(SHELLS[s.index]?.id ?? '');
}

// ---------------------------------------------------------------------------
// createHud
// ---------------------------------------------------------------------------

export function createHud(root: HTMLElement): Hud {
  root.classList.add('hud');
  root.replaceChildren();
  const reduced = prefersReducedMotion();

  // ---- scrims (legibility only) ------------------------------------------------
  el('div', 'sx-scrim sx-scrim--top', root).setAttribute('aria-hidden', 'true');
  el('div', 'sx-scrim sx-scrim--bottom', root).setAttribute('aria-hidden', 'true');

  // ---- top bar -------------------------------------------------------------------
  const top = el('header', 'sx-top sx-reveal', root);
  top.style.setProperty('--d', '0');
  const brand = el('div', 'sx-brand', top);
  el('span', 'sx-wordmark', brand, 'STARLINK');
  el('span', 'sx-vrule', brand).setAttribute('aria-hidden', 'true');
  el('span', 'sx-brand__sub', brand, 'Live constellation');

  const topRight = el('div', 'sx-top__right', top);
  const live = el('div', 'sx-live', topRight);
  live.dataset.state = 'live';
  el('span', 'sx-live__dot', live).setAttribute('aria-hidden', 'true');
  const liveLabel = textNode(el('span', 'sx-live__label', live), 'LIVE');
  el('span', 'sx-vrule', topRight).setAttribute('aria-hidden', 'true');
  const clock = el('div', 'sx-clock', topRight);
  clock.setAttribute('role', 'timer');
  clock.setAttribute('aria-label', 'Coordinated Universal Time');
  const clockTime = new TabularText(el('span', 'sx-clock__time', clock), '--:--:--');
  el('span', 'sx-clock__unit', clock, 'UTC');

  const perf = el('div', 'sx-perf', root);
  perf.hidden = true;
  perf.dataset.tier = 'good';
  perf.setAttribute('aria-hidden', 'true');
  const perfText = new TabularText(el('span', '', perf), '— FPS');

  // ---- notice line -------------------------------------------------------------------
  const notice = el('div', 'sx-notice', root);
  notice.hidden = true;
  el('span', 'sx-notice__dot', notice).setAttribute('aria-hidden', 'true');
  const noticeText = textNode(el('span', 'sx-notice__text', notice));
  const noticeClose = el('button', 'sx-textbtn', notice, 'Dismiss');
  noticeClose.type = 'button';
  noticeClose.setAttribute('aria-label', 'Dismiss notice');
  // Screen-reader status for the notice: always rendered (the visible notice starts display:none,
  // and a live region that appears already filled is not announced).
  const noticeSr = el('div', 'sx-vh', root);
  noticeSr.setAttribute('role', 'status');
  const noticeSrText = textNode(noticeSr, '');

  // ---- hero (bottom-left) --------------------------------------------------------------
  const hero = el('section', 'sx-hero sx-reveal', root);
  hero.style.setProperty('--d', '1');
  hero.setAttribute('aria-label', 'Constellation');
  el('div', 'sx-kicker', hero, 'Satellites in orbit');
  const heroLine = el('div', 'sx-hero__line', hero);
  const heroCount = new TabularText(el('span', 'sx-hero__count', heroLine), '—');
  const heroOf = el('span', 'sx-hero__of', heroLine);
  heroOf.hidden = true;
  const heroOfText = textNode(heroOf, '');
  const heroSub = el('div', 'sx-hero__sub', hero);
  const subSource = textNode(el('span', '', heroSub), '');
  const subAgeWrap = el('span', 'sx-hero__seg', heroSub);
  sepDot(subAgeWrap);
  el('span', '', subAgeWrap, 'Updated ');
  const subAge = textNode(el('span', '', subAgeWrap), '');
  subAgeWrap.hidden = true;
  const subPartial = el('span', 'sx-hero__seg', heroSub);
  sepDot(subPartial);
  el('span', '', subPartial, 'Partial snapshot');
  sepDot(subPartial);
  const subPartialText = textNode(el('span', '', subPartial), '');
  subPartial.hidden = true;

  // ---- right column: legend + camera -------------------------------------------------------
  const side = el('div', 'sx-side', root);
  const legend = el('nav', 'sx-legend sx-reveal', side);
  legend.style.setProperty('--d', '2');
  legend.setAttribute('aria-label', 'Orbital shells');
  const legendHead = el('div', 'sx-legend__head', legend);
  const legendToggle = el('button', 'sx-legend__toggle', legendHead);
  legendToggle.type = 'button';
  el('span', '', legendToggle, 'Shells');
  el('span', 'sx-chev', legendToggle).setAttribute('aria-hidden', 'true');
  const legendReset = el('button', 'sx-textbtn', legendHead, 'Show all');
  legendReset.type = 'button';
  legendReset.hidden = true;
  const legendList = el('ul', 'sx-legend__list', legend);
  legendList.id = 'sx-shell-list';
  legendToggle.setAttribute('aria-controls', legendList.id);

  // LASER LINKS toggle (same table style as the shell rows) + disclaimer
  const laser = el('div', 'sx-laser sx-reveal', side);
  laser.style.setProperty('--d', '3');
  laser.setAttribute('role', 'group');
  laser.setAttribute('aria-label', 'Laser links');
  laser.hidden = true; // until the integrator calls setLaserLinks
  const laserBtn = el('button', 'sx-shell sx-laser__btn', laser);
  laserBtn.type = 'button';
  el('span', 'sx-laser__swatch', laserBtn).setAttribute('aria-hidden', 'true');
  el('span', 'sx-shell__label', laserBtn, 'Laser links');
  const laserCount = new TabularText(el('span', 'sx-shell__count', laserBtn), '');
  const laserNote = el('div', 'sx-laser__note', laser);
  const laserEquipped = textNode(el('span', 'sx-laser__seg', laserNote), '');
  const laserEst = el('span', 'sx-laser__seg', laserNote);
  sepDot(laserEst);
  el('span', '', laserEst, 'Estimated');
  const laserModeled = el('div', 'sx-laser__note', laser);
  const modeledA = el('span', 'sx-laser__seg', laserModeled, 'Modeled topology');
  sepDot(modeledA);
  el('span', 'sx-laser__seg', laserModeled, 'Not published by SpaceX');

  const cam = el('div', 'sx-cam sx-reveal', side);
  cam.style.setProperty('--d', '4');
  cam.setAttribute('aria-label', 'Camera position');
  el('span', '', cam, 'ALT ');
  const camAlt = new TabularText(el('span', '', cam), '—');
  sepDot(cam);
  const camLat = new TabularText(el('span', '', cam), '—');
  el('span', 'sx-cam__gap', cam);
  const camLon = new TabularText(el('span', '', cam), '—');

  // ---- hint (bottom-centre) -----------------------------------------------------------------
  const hint = el('div', 'sx-hint', root);
  hint.setAttribute('aria-hidden', 'true');
  const hintParts = mq(COARSE)
    ? ['Drag to rotate', 'Pinch to zoom', 'Tap a satellite', 'Double-tap to view up close']
    : ['Drag to rotate', 'Scroll to zoom', 'Click a satellite', 'Double-click to view up close'];
  // each part is unbreakable; the line wraps between parts (the dot stays at the end of a line)
  hintParts.forEach((p, i) => {
    const seg = el('span', 'sx-hint__seg', hint, p);
    if (i < hintParts.length - 1) sepDot(seg);
  });

  // ---- hover label ----------------------------------------------------------------------------
  const hover = el('div', 'sx-hover', root);
  hover.setAttribute('aria-hidden', 'true');
  const hoverName = textNode(el('div', 'sx-hover__name', hover), '');
  const hoverSubEl = el('div', 'sx-hover__sub', hover);
  const hoverSub = textNode(hoverSubEl, '');

  // ---- info panel ---------------------------------------------------------------------------------
  const infoPanel = createInfoPanel(root, {
    onOpenChange(open) {
      root.dataset.panel = open ? 'open' : '';
      if (open) hud.dismissHint();
    },
  });

  // ---- boot screen ---------------------------------------------------------------------------------
  const boot = el('section', 'sx-boot', root);
  boot.setAttribute('aria-label', 'Loading');
  const bootInner = el('div', 'sx-boot__inner', boot);
  el('div', 'sx-boot__word', bootInner, 'STARLINK');
  const bootTrack = el('div', 'sx-boot__track is-indeterminate', bootInner);
  bootTrack.setAttribute('role', 'progressbar');
  bootTrack.setAttribute('aria-valuemin', '0');
  bootTrack.setAttribute('aria-valuemax', '100');
  bootTrack.setAttribute('aria-label', 'Loading progress');
  const bootFill = el('div', 'sx-boot__fill', bootTrack);
  const bootStatusEl = el('div', 'sx-boot__status', bootInner);
  bootStatusEl.setAttribute('role', 'status');
  const bootStatus = textNode(bootStatusEl, 'Initialising');
  const bootDetail = textNode(el('div', 'sx-boot__detail', bootInner), '');

  // ---- error overlay --------------------------------------------------------------------------------
  const errorLayer = el('div', 'sx-error', root);
  errorLayer.hidden = true;
  errorLayer.setAttribute('role', 'alertdialog');
  errorLayer.setAttribute('aria-modal', 'true');
  const errorInner = el('div', 'sx-error__inner', errorLayer);
  el('div', 'sx-kicker', errorInner, 'System fault');
  const errorTitle = el('h2', 'sx-error__title', errorInner);
  errorTitle.id = 'sx-error-title';
  errorLayer.setAttribute('aria-labelledby', errorTitle.id);
  const errorMsg = el('p', 'sx-error__msg', errorInner);
  const errorReload = el('button', 'sx-btn sx-btn--primary', errorInner);
  errorReload.type = 'button';
  el('span', 'sx-btn__label', errorReload, 'Reload');
  errorReload.appendChild(createArrow('sx-btn__arrow'));
  errorReload.addEventListener('click', () => location.reload());

  // ---- state -------------------------------------------------------------------------------------------
  let lastClockSec = Number.NaN;
  let lastClockDay = Number.NaN;
  let isLive: boolean | null = true;

  let bootDone = false;
  let bootHideTimer = 0;
  let lastBootP = -1;
  let lastBootPct = -1;

  let fetchedMs = Number.NaN;
  let nextRefreshMs = Number.NaN;
  let isPartial = false;
  /** "Full catalogue loading d/t" while a refresh cycle runs, else null. */
  let cycleText: string | null = null;

  let lastVisible = Number.NaN;
  let lastTotal = Number.NaN;

  let pFps = Number.NaN;
  let pMs = Number.NaN;
  let pRes = Number.NaN;
  let pMsaa = Number.NaN;
  let pKf = Number.NaN;
  let pLinks = Number.NaN;
  /** Last perf sample (reused object), formatted only while the readout is visible. */
  const perfLast: HudPerfInfo = { fps: Number.NaN, frameMs: Number.NaN, resolutionScale: 1, msaa: 0 };

  let cAlt = Number.NaN;
  let cLat = Number.NaN;
  let cLon = Number.NaN;

  let warningValue: string | null = null;

  let hoverText: string | null = null;
  let hoverOn = false;
  let hoverW = -1;
  let hoverH = -1;
  let hoverX = Number.NaN;
  let hoverY = Number.NaN;
  let vw = window.innerWidth;
  let vh = window.innerHeight;

  let hintGone = false;

  interface ShellRow {
    index: number;
    label: string;
    note: string;
    button: HTMLButtonElement;
    count: TabularText;
    lastCount: number;
    color: string;
    visible: boolean;
  }
  let shellRows: ShellRow[] = [];
  const toggleListeners: ((index: number, visible: boolean) => void)[] = [];

  let laserOn: boolean | null = null;
  let laserLinks = Number.NaN;
  let laserEquippedN = Number.NaN;
  const laserListeners: ((enabled: boolean) => void)[] = [];

  // ---- behaviour ---------------------------------------------------------------------------------------
  window.addEventListener('resize', () => {
    vw = window.innerWidth;
    vh = window.innerHeight;
  }, { passive: true });

  const setLegendCollapsed = (collapsed: boolean): void => {
    legend.classList.toggle('is-collapsed', collapsed);
    legendList.hidden = collapsed;
    legendToggle.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
    legendToggle.title = collapsed ? 'Show orbital shells' : 'Hide orbital shells';
    updateResetVisibility();
  };
  legendToggle.addEventListener('click', () => setLegendCollapsed(!legend.classList.contains('is-collapsed')));

  function updateResetVisibility(): void {
    let anyHidden = false;
    for (const r of shellRows) if (!r.visible) anyHidden = true;
    legendReset.hidden = !anyHidden || legend.classList.contains('is-collapsed');
  }

  const emitToggle = (index: number, visible: boolean): void => {
    for (const cb of toggleListeners) cb(index, visible);
  };

  const applyRowVisible = (row: ShellRow, visible: boolean): void => {
    row.visible = visible;
    row.button.setAttribute('aria-pressed', visible ? 'true' : 'false');
    row.button.title = `${visible ? 'Hide' : 'Show'} ${row.label}`;
  };

  legendList.addEventListener('click', (ev) => {
    const target = ev.target as Element | null;
    const button = target?.closest<HTMLButtonElement>('.sx-shell');
    if (!button) return;
    const row = shellRows.find((r) => r.button === button);
    if (!row) return;
    const next = !row.visible;
    applyRowVisible(row, next); // optimistic; the integrator re-sends setShells with the truth
    updateResetVisibility();
    emitToggle(row.index, next);
  });

  const applyLaserOn = (on: boolean): void => {
    laserOn = on;
    laserBtn.setAttribute('aria-pressed', on ? 'true' : 'false');
    laserBtn.title = on ? 'Hide the modeled laser links' : 'Show the modeled laser links';
    laserCount.set(formatLinkCount(on, laserLinks));
    laserBtn.classList.toggle('is-empty', !on || laserLinks === 0);
  };

  laserBtn.addEventListener('click', () => {
    const next = !laserOn;
    applyLaserOn(next); // optimistic; the integrator re-sends setLaserLinks with the truth
    for (const cb of laserListeners) cb(next);
  });

  legendReset.addEventListener('click', () => {
    for (const row of shellRows) {
      if (!row.visible) {
        applyRowVisible(row, true);
        emitToggle(row.index, true);
      }
    }
    updateResetVisibility();
    legendToggle.focus();
  });

  noticeClose.addEventListener('click', () => {
    notice.hidden = true;
    setData(noticeSrText, '');
    root.classList.remove('has-notice');
  });

  // collapsed on phones and on very short (landscape) viewports, where 8 rows do not fit
  setLegendCollapsed(mq(COMPACT) || mq(SHORT));

  // perf readout: backquote toggles (e.code covers non-US layouts, e.key the US one)
  window.addEventListener('keydown', (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey || e.repeat) return;
    if (e.code !== 'Backquote' && e.key !== '`') return;
    const t = e.target as HTMLElement | null;
    if (t && (t.isContentEditable || t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT')) return;
    perf.hidden = !perf.hidden;
    if (!perf.hidden) renderPerf();
  });

  /** Remove the faded-out boot screen from rendering (display: none stops its animations). */
  const hideBoot = (): void => {
    if (bootDone) boot.hidden = true;
  };
  boot.addEventListener('transitionend', (e) => {
    if (e.target === boot && e.propertyName === 'opacity') hideBoot();
  });

  // hint: fades on the first real interaction after boot
  const onFirstInteraction = (): void => hud.dismissHint();
  const armHint = (): void => {
    if (hintGone) return;
    window.addEventListener('pointerdown', onFirstInteraction, { capture: true, passive: true });
    window.addEventListener('wheel', onFirstInteraction, { capture: true, passive: true });
  };
  hint.addEventListener('transitionend', (e) => {
    if (e.target === hint && hintGone) hint.hidden = true;
  });

  /** Once-per-second housekeeping driven by the clock (data age, refresh-due notice). */
  const tickSecond = (): void => {
    if (Number.isNaN(fetchedMs)) return;
    const now = Date.now();
    setData(subAge, formatAge(now - fetchedMs));
    if (isPartial && !Number.isNaN(nextRefreshMs) && now >= nextRefreshMs) {
      setData(subPartialText, cycleText ?? 'Full catalogue refresh due');
    }
  };

  function renderPerf(): void {
    perfText.set(formatPerf(perfLast));
    const tier = perfTier(perfLast.fps);
    if (perf.dataset.tier !== tier) perf.dataset.tier = tier;
  }

  const placeHover = (x: number, y: number): void => {
    if (hoverW < 0) {
      // one layout read per label change (not per move)
      hoverW = hover.offsetWidth;
      hoverH = hover.offsetHeight;
    }
    let tx = Math.round(x) + HOVER_OFFSET;
    let ty = Math.round(y) + HOVER_OFFSET;
    if (tx + hoverW > vw - HOVER_MARGIN) tx = Math.round(x) - HOVER_OFFSET - hoverW;
    if (ty + hoverH > vh - HOVER_MARGIN) ty = Math.round(y) - HOVER_OFFSET - hoverH;
    if (tx < HOVER_MARGIN) tx = HOVER_MARGIN;
    if (ty < HOVER_MARGIN) ty = HOVER_MARGIN;
    if (tx === hoverX && ty === hoverY) return;
    hoverX = tx;
    hoverY = ty;
    hover.style.transform = `translate3d(${tx}px,${ty}px,0)`;
  };

  const hud: Hud = {
    infoPanel,

    setBoot(progress, status, detail) {
      if (bootDone) {
        // Re-entering the boot screen (e.g. a full reload of the data set).
        bootDone = false;
        clearTimeout(bootHideTimer);
        boot.hidden = false;
        boot.classList.remove('is-leaving');
        root.classList.remove('is-ready');
      }
      if (progress === null) {
        if (lastBootP !== -2) {
          lastBootP = -2;
          lastBootPct = -1;
          bootTrack.classList.add('is-indeterminate');
          bootTrack.removeAttribute('aria-valuenow');
        }
      } else {
        const p = Math.min(1, Math.max(0, Number.isFinite(progress) ? progress : 0));
        if (lastBootP < 0) bootTrack.classList.remove('is-indeterminate');
        if (Math.abs(p - lastBootP) >= 0.001) {
          lastBootP = p;
          bootFill.style.transform = `scaleX(${p.toFixed(3)})`;
        }
        const pct = Math.round(p * 100);
        if (pct !== lastBootPct) {
          lastBootPct = pct;
          bootTrack.setAttribute('aria-valuenow', `${pct}`);
        }
      }
      setData(bootStatus, status);
      setData(bootDetail, detail ?? '');
    },

    finishBoot() {
      if (bootDone) return;
      bootDone = true;
      root.classList.add('is-ready');
      boot.classList.add('is-leaving');
      clearTimeout(bootHideTimer);
      bootHideTimer = window.setTimeout(hideBoot, reduced ? 50 : 1400);
      armHint();
    },

    showError(title, message) {
      errorTitle.textContent = title;
      errorMsg.textContent = message;
      errorLayer.hidden = false;
      root.classList.add('has-error');
      errorReload.focus({ preventScroll: true });
    },

    setClock(date) {
      const ms = date.getTime();
      const sec = Math.floor(ms / 1000);
      if (sec === lastClockSec) return;
      lastClockSec = sec;
      const day = Math.floor(sec / 86_400);
      if (day !== lastClockDay) {
        lastClockDay = day;
        clock.title = `${formatUtcDate(ms)} UTC`;
      }
      clockTime.set(formatUtcTime(ms));
      tickSecond();
    },

    setLive(isRealtime) {
      if (isRealtime === isLive) return;
      isLive = isRealtime;
      live.dataset.state = isRealtime ? 'live' : 'sim';
      setData(liveLabel, isRealtime ? 'LIVE' : 'SIMULATED');
    },

    setDataInfo(info) {
      const sourceLine = formatDataSources(info.source, info.sources);
      setData(subSource, sourceLine);
      // full names on hover (the line drops parentheticals and caps at three)
      const full = info.sources && info.sources.length > 0 ? info.sources.join(' + ') : info.source;
      if (full && full.toUpperCase() !== sourceLine) heroSub.title = full;
      else heroSub.removeAttribute('title');
      fetchedMs = info.fetchedAt.getTime();
      subAgeWrap.hidden = !Number.isFinite(fetchedMs);
      isPartial = info.partial;
      nextRefreshMs = info.nextRefresh ? info.nextRefresh.getTime() : Number.NaN;
      const c = info.cycle;
      cycleText = c && c.total > 0 ? `Full catalogue loading ${Math.min(c.done, c.total)}/${c.total}` : null;
      if (info.partial) {
        setData(
          subPartialText,
          info.nextRefresh ? `Full catalogue at ${formatLocalHHMM(info.nextRefresh)}` : 'Full catalogue pending',
        );
      }
      subPartial.hidden = !info.partial;
      tickSecond();
    },

    setTracked(visible, total) {
      if (visible === lastVisible && total === lastTotal) return;
      lastVisible = visible;
      lastTotal = total;
      heroCount.set(groupThousands(visible));
      heroOf.hidden = visible === total;
      setData(heroOfText, visible === total ? '' : `/ ${groupThousands(total)}`);
    },

    setPerf(info) {
      const fps = Math.round(info.fps);
      const ms = Math.round(info.frameMs * 10);
      const res = Math.round(info.resolutionScale * 100);
      const msaa = Math.round(info.msaa);
      const kf = Math.round((info.keyframeMs ?? Number.NaN) * 10);
      const lk = Math.round(info.linksMs ?? Number.NaN);
      if (
        fps === pFps &&
        ms === pMs &&
        res === pRes &&
        msaa === pMsaa &&
        (kf === pKf || (Number.isNaN(kf) && Number.isNaN(pKf))) &&
        (lk === pLinks || (Number.isNaN(lk) && Number.isNaN(pLinks)))
      ) {
        return;
      }
      pFps = fps;
      pMs = ms;
      pRes = res;
      pMsaa = msaa;
      pKf = kf;
      pLinks = lk;
      perfLast.fps = info.fps;
      perfLast.frameMs = info.frameMs;
      perfLast.resolutionScale = info.resolutionScale;
      perfLast.msaa = info.msaa;
      perfLast.keyframeMs = info.keyframeMs;
      perfLast.linksMs = info.linksMs;
      if (!perf.hidden) renderPerf(); // otherwise formatted when shown
    },

    setCamera(info) {
      const altKey = Math.abs(info.altitudeKm) >= 100 ? Math.round(info.altitudeKm) : Math.round(info.altitudeKm * 10) / 10;
      const latKey = Math.round(info.lat * 100);
      const lonKey = Math.round(info.lon * 100);
      if (altKey !== cAlt) {
        cAlt = altKey;
        camAlt.set(formatAltitudeKm(info.altitudeKm));
      }
      if (latKey !== cLat) {
        cLat = latKey;
        camLat.set(formatLat(info.lat));
      }
      if (lonKey !== cLon) {
        cLon = lonKey;
        camLon.set(formatLon(info.lon));
      }
    },

    setShells(shells) {
      let same = shells.length === shellRows.length;
      if (same) {
        for (let i = 0; i < shells.length; i++) {
          const r = shellRows[i];
          if (shells[i].index !== r.index || shells[i].label !== r.label || noteFor(shells[i]) !== r.note) {
            same = false;
            break;
          }
        }
      }
      if (!same) {
        const items: HTMLLIElement[] = [];
        shellRows = shells.map((s) => {
          const li = el('li', 'sx-legend__item');
          const button = el('button', 'sx-shell', li);
          button.type = 'button';
          el('span', 'sx-swatch', button).setAttribute('aria-hidden', 'true');
          dotted('sx-shell__label', button, s.label);
          const count = new TabularText(el('span', 'sx-shell__count', button), '');
          const note = noteFor(s);
          if (note) {
            button.classList.add('has-note');
            dotted('sx-shell__note', button, note);
          }
          items.push(li);
          return { index: s.index, label: s.label, note, button, count, lastCount: Number.NaN, color: '', visible: s.visible };
        });
        legendList.replaceChildren(...items);
      }
      for (let i = 0; i < shells.length; i++) {
        const s = shells[i];
        const row = shellRows[i];
        if (s.count !== row.lastCount) {
          row.lastCount = s.count;
          row.count.set(groupThousands(s.count));
          row.button.classList.toggle('is-empty', s.count === 0);
        }
        const color = rgbToCss(s.color);
        if (color !== row.color) {
          row.color = color;
          row.button.style.setProperty('--c', color);
        }
        if (s.visible !== row.visible || !same) applyRowVisible(row, s.visible);
      }
      updateResetVisibility();
    },

    onShellToggle(cb) {
      toggleListeners.push(cb);
    },

    setLaserLinks(info) {
      if (laser.hidden) laser.hidden = false;
      if (info.equipped !== laserEquippedN) {
        laserEquippedN = info.equipped;
        setData(laserEquipped, formatLaserEquipped(info.equipped));
      }
      if (info.enabled === laserOn && info.links === laserLinks) return;
      laserLinks = info.links;
      applyLaserOn(info.enabled);
    },

    onLaserToggle(cb) {
      laserListeners.push(cb);
    },

    setWarning(text) {
      if (text === warningValue) return;
      warningValue = text;
      if (text === null) {
        notice.hidden = true;
        setData(noticeSrText, '');
        notice.removeAttribute('title');
        root.classList.remove('has-notice');
        return;
      }
      // A new message re-opens the notice even if a previous one was dismissed.
      setData(noticeText, text);
      notice.title = text; // full text when the compact layout truncates it
      notice.hidden = false;
      root.classList.add('has-notice');
      setData(noticeSrText, '');
      requestAnimationFrame(() => {
        if (warningValue === text) setData(noticeSrText, text);
      });
    },

    setHoverLabel(text, x, y) {
      if (text === null || text === '') {
        if (hoverOn) {
          hoverOn = false;
          hover.classList.remove('is-on');
        }
        return;
      }
      if (text !== hoverText) {
        hoverText = text;
        const [name, sub] = splitLabel(text);
        setData(hoverName, name);
        setData(hoverSub, sub);
        hoverSubEl.hidden = sub === '';
        hoverW = -1;
      }
      placeHover(x, y);
      if (!hoverOn) {
        hoverOn = true;
        hover.classList.add('is-on');
      }
    },

    dismissHint() {
      if (hintGone) return;
      hintGone = true;
      window.removeEventListener('pointerdown', onFirstInteraction, { capture: true });
      window.removeEventListener('wheel', onFirstInteraction, { capture: true });
      hint.classList.add('is-gone');
      if (reduced) hint.hidden = true;
    },
  };

  return hud;
}
