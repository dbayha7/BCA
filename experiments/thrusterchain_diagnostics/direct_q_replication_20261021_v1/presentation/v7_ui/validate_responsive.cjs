#!/usr/bin/env node
'use strict';

// Runs the actual embedded reserve controller against synthetic measurements.
// No browser, headless renderer, CSS layout engine, or URL loading is used.
// Usage: node validate_responsive.cjs [HTML] [--receipt OUTPUT.json]
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const hash = value => crypto.createHash('sha256').update(value).digest('hex');
const normalize = value => value.replace(/\r\n/g, '\n').trim();
let input = path.join(__dirname, '..', 'ThrusterChain_diagnostics_results_v7_interactive.html');
let receiptPath = path.join(__dirname, 'responsive_validation.json');
const args = process.argv.slice(2);
for (let i = 0; i < args.length; i++) {
  if (args[i] === '--receipt') receiptPath = path.resolve(args[++i]);
  else input = path.resolve(args[i]);
}
const raw = fs.readFileSync(input);
const html = raw.toString('utf8');
const receipt = {
  schema: 'thrusterchain-responsive-static-controller-validation-v1',
  generatedAt: new Date().toISOString(), file: input, bytes: raw.length,
  sha256: hash(raw),
  validation: 'Static embedded-source and breakpoint checks; actual embedded controller execution with synthetic rectangle/style measurements in a Node VM.',
  limitations: [
    'No browser, headless rendering, actual CSS cascade, or layout verification.',
    'Phone, desktop, and zoom cases supply synthetic measurements; they do not measure those viewports.',
    'No visual confirmation of overflow, clipping, navigation accessibility, or mobile browser chrome behavior.'
  ],
  checks: {}, errors: []
};
function check(name, fn) {
  try { receipt.checks[name] = { passed: true, ...fn() }; }
  catch (error) {
    receipt.checks[name] = { passed: false };
    receipt.errors.push({ check: name, message: error.message });
  }
}
function attrs(source) {
  const result = {};
  for (const m of source.matchAll(/([^\s=/>]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+)))?/g)) {
    result[m[1]] = m[2] ?? m[3] ?? m[4] ?? '';
  }
  return result;
}
function elements(tag) {
  return [...html.matchAll(new RegExp('<' + tag + '\\b([^>]*)>([\\s\\S]*?)<\\/' + tag + '>', 'gi'))]
    .map(m => ({ attrs: attrs(m[1]), source: m[2] }));
}
const scripts = elements('script'), styles = elements('style');
let controller;
check('exactEmbeddedResponsiveSources', () => {
  const found = scripts.filter(s => s.attrs.id === 'v7-responsive-script');
  assert.equal(found.length, 1, 'Expected exactly one v7-responsive-script');
  controller = found[0].source;
  const localJS = fs.readFileSync(path.join(__dirname, 'responsive.js'), 'utf8');
  assert.equal(normalize(controller), normalize(localJS), 'Embedded responsive JavaScript differs from responsive.js');
  new vm.Script(controller, { filename: 'embedded-v7-responsive-script.js' });
  const localCSS = fs.readFileSync(path.join(__dirname, 'responsive.css'), 'utf8');
  const embeddedCSS = styles.filter(s => normalize(s.source) === normalize(localCSS));
  assert.equal(embeddedCSS.length, 1, 'Expected one exact responsive.css embed');
  return { normalizedScriptSha256: hash(normalize(controller)), normalizedStyleSha256: hash(normalize(localCSS)), normalization: 'CRLF to LF and outer whitespace only' };
});
check('navigationBreakpointsAgree', () => {
  const css = styles.filter(s => s.source.includes('/* Navigation. Contents (#nv)'));
  const boot = scripts.filter(s => s.source.includes('/* Before the page is drawn.'));
  const navigation = scripts.filter(s => s.source.includes('const mqWide = mm('));
  assert.equal(css.length, 1); assert.equal(boot.length, 1); assert.equal(navigation.length, 1);
  for (const [name, source] of [['CSS', css[0].source], ['boot', boot[0].source], ['navigation', navigation[0].source]]) {
    assert.ok(/min-width:\s*1200px/.test(source), name + ' lacks 1200px right-rail breakpoint');
    assert.ok(/min-width:\s*1600px/.test(source), name + ' lacks 1600px left-rail breakpoint');
    assert.ok(!/\b(?:999|1000|1239|1240)\s*px\b/.test(source), name + ' retains an old navigation breakpoint');
  }
  assert.ok(/max-width:\s*1199px/.test(css[0].source));
  assert.ok(/max-width:\s*1599px/.test(css[0].source));
  assert.ok(/mqWide\s*=\s*mm\(["']\(min-width:\s*1600px\)/.test(navigation[0].source));
  assert.ok(/mqRail\s*=\s*mm\(["']\(min-width:\s*1200px\)/.test(navigation[0].source));
  return { navStyles: 1, navScripts: 2, rightRailPx: 1200, leftRailPx: 1600 };
});
function blockFrom(source, offset) {
  const open = source.indexOf('{', offset);
  assert.ok(open >= 0, 'Missing CSS block');
  let depth = 1, end = open + 1;
  for (; end < source.length && depth; end++) {
    if (source[end] === '{') depth++;
    else if (source[end] === '}') depth--;
  }
  assert.equal(depth, 0, 'Unbalanced CSS block');
  return source.slice(open + 1, end - 1);
}
check('screenScopedBarAndContainerRules', () => {
  const css = fs.readFileSync(path.join(__dirname, 'responsive.css'), 'utf8');
  const screenStart = css.indexOf('@media screen');
  assert.ok(screenStart >= 0, 'Missing screen scope');
  const screen = blockFrom(css, screenStart);
  const barStart = screen.indexOf('html.nv-js #nv-bar {');
  assert.ok(barStart >= 0, 'Missing unconditional screen-scoped #nv-bar rule');
  const bar = blockFrom(screen, barStart);
  assert.match(bar, /display:\s*flex\s*;/);
  assert.match(bar, /height:\s*auto\s*;/);
  assert.match(screen, /padding-bottom:\s*calc\(var\(--nv-bar-space\)/);
  assert.match(screen, /container:\s*reading\s*\/\s*inline-size/);
  for (const width of [900, 620, 420]) {
    const start = screen.indexOf('@container reading (max-width: ' + width + 'px)');
    assert.ok(start >= 0, 'Missing reading-container breakpoint ' + width);
    assert.match(blockFrom(screen, start), /grid-template-columns:/);
  }
  assert.ok(screen.includes('#learn .ls-lines'), 'Missing lesson calculation layout fallback');
  return { alwaysVisibleBarRule: true, screenOnly: true, containerWidthsPx: [900, 620, 420], note: 'Syntax/placement assertions only; no cascade or layout engine' };
});

function harness(options = {}) {
  assert.ok(controller, 'Responsive script not found in assembled HTML');
  const classes = new Set(options.nvJS === false ? [] : ['nv-js']);
  const properties = new Map(options.reserve ? [['--nv-bar-space', options.reserve]] : []);
  const writes = [], removals = [], observed = [], frames = [], listeners = new Map(), visualListeners = new Map();
  const metrics = { height: 56, bottom: '0px', display: 'flex', ...options.metrics };
  let resizeObserverCallback;
  const root = {
    classList: { contains: name => classes.has(name) },
    style: {
      setProperty: (name, value) => { writes.push([name, value]); properties.set(name, value); },
      removeProperty: name => { removals.push(name); properties.delete(name); },
      getPropertyValue: name => properties.get(name) || ''
    }
  };
  const bar = { getBoundingClientRect: () => ({ height: metrics.height }) };
  const addListener = map => (name, fn) => { if (!map.has(name)) map.set(name, []); map.get(name).push(fn); };
  const window = {
    getComputedStyle: element => { assert.equal(element, bar); return { display: metrics.display, bottom: metrics.bottom }; },
    requestAnimationFrame: fn => { frames.push(fn); return frames.length; },
    addEventListener: addListener(listeners)
  };
  if (options.observer !== false) window.ResizeObserver = class {
    constructor(fn) { resizeObserverCallback = fn; }
    observe(element) { observed.push(element); }
  };
  if (options.visualViewport !== false) window.visualViewport = { addEventListener: addListener(visualListeners) };
  const document = { documentElement: root, getElementById: id => id === 'nv-bar' && options.bar !== false ? bar : null };
  vm.runInNewContext(controller, { window, document, Math, parseFloat }, { filename: 'embedded-v7-responsive-script.js', timeout: 1000 });
  return {
    classes, properties, writes, removals, metrics, frames, observed, bar, listeners, visualListeners,
    reserve: () => properties.get('--nv-bar-space'),
    emit: name => (listeners.get(name) || []).forEach(fn => fn()),
    visualResize: () => (visualListeners.get('resize') || []).forEach(fn => fn()),
    observedResize: () => { assert.ok(resizeObserverCallback, 'No ResizeObserver callback'); resizeObserverCallback([]); },
    flush: () => { const batch = frames.splice(0); batch.forEach(fn => fn()); assert.ok(frames.length === 0, 'Measurement unexpectedly scheduled itself'); }
  };
}
check('immediatePhoneDesktopAndWrappedBarReserve', () => {
  const cases = [
    ['phone supplied rectangle', { height: 76, bottom: '0px' }, '84px'],
    ['desktop supplied rectangle', { height: 58, bottom: '10px' }, '76px'],
    ['zoom/wrapped supplied rectangle', { height: 104.4, bottom: '12.2px' }, '125px'],
    ['automatic bottom', { height: 61.2, bottom: 'auto' }, '70px'],
    ['negative bottom', { height: 58, bottom: '-4px' }, '66px']
  ];
  for (const [name, metrics, expected] of cases) {
    const h = harness({ metrics });
    assert.equal(h.reserve(), expected, name);
    assert.equal(h.frames.length, 0, 'Initial measurement must be immediate');
  }
  return { cases: cases.map(([name, metrics, expected]) => ({ name, supplied: metrics, expected })) };
});
check('resizeEventsCoalesceAndUseLatestMeasurement', () => {
  const h = harness();
  assert.equal(h.observed.length, 1); assert.equal(h.observed[0], h.bar);
  for (const name of ['resize', 'load', 'afterprint']) assert.ok(h.listeners.has(name), 'Missing ' + name + ' listener');
  h.metrics.height = 80; h.emit('resize');
  h.metrics.height = 96; h.observedResize(); h.emit('load'); h.emit('afterprint');
  h.metrics.height = 108; h.metrics.bottom = '12px';
  assert.equal(h.frames.length, 1, 'Resize signals should share one animation frame');
  assert.equal(h.reserve(), '64px', 'Measurement must wait for queued frame');
  h.flush(); assert.equal(h.reserve(), '128px');
  return { signals: 4, queuedFrames: 1, expectedReserve: '128px' };
});
check('unchangedMeasurementsDoNotRewriteStyle', () => {
  const h = harness(); const writes = h.writes.length;
  h.emit('resize'); h.flush(); h.observedResize(); h.flush();
  assert.equal(h.writes.length, writes, 'Unchanged measurements should not rewrite the root style');
  return { repeatedMeasurements: 2, additionalWrites: 0 };
});
check('visualViewportResizeAndOptionalAPIs', () => {
  const full = harness();
  assert.ok(full.visualListeners.has('resize'), 'Missing visualViewport resize listener');
  full.metrics.height = 91; full.visualResize(); full.flush(); assert.equal(full.reserve(), '99px');
  const minimal = harness({ observer: false, visualViewport: false });
  assert.equal(minimal.reserve(), '64px');
  minimal.metrics.height = 72; minimal.emit('resize'); minimal.flush(); assert.equal(minimal.reserve(), '80px');
  return { visualViewportResize: true, observerAndVisualViewportAbsent: true };
});
check('hiddenZeroHeightAbsentBarAndPlainPage', () => {
  const hidden = harness({ metrics: { display: 'none' }, reserve: '90px' });
  assert.equal(hidden.reserve(), '90px'); assert.equal(hidden.writes.length, 0);
  hidden.metrics.display = 'flex'; hidden.emit('resize'); hidden.flush(); assert.equal(hidden.reserve(), '64px');
  const zero = harness({ metrics: { height: 0 }, reserve: '90px' });
  assert.equal(zero.reserve(), '90px'); assert.equal(zero.writes.length, 0);
  const absent = harness({ bar: false }); assert.equal(absent.writes.length, 0); assert.equal(absent.listeners.size, 0);
  const plain = harness({ nvJS: false, reserve: '90px' }); assert.equal(plain.reserve(), undefined);
  plain.classes.add('nv-js'); plain.emit('load'); plain.flush(); assert.equal(plain.reserve(), '64px');
  plain.classes.delete('nv-js'); plain.emit('resize'); plain.flush(); assert.equal(plain.reserve(), undefined);
  return { hiddenBarIgnored: true, zeroHeightIgnored: true, missingBarSafe: true, plainPageRemovesInlineReserve: true };
});
receipt.passed = receipt.errors.length === 0;
receipt.checkCount = Object.keys(receipt.checks).length;
fs.writeFileSync(receiptPath, JSON.stringify(receipt, null, 2) + '\n');
console.log(JSON.stringify({ passed: receipt.passed, checks: receipt.checkCount, receipt: receiptPath, errors: receipt.errors }, null, 2));
process.exitCode = receipt.passed ? 0 : 1;
