#!/usr/bin/env node
'use strict';

// Execute the actual restart source in Node VM with storage/navigation doubles.
// No browser, URL loading, headless renderer, or full-page runtime is involved.
// Usage: node validate_lesson_restart.cjs [HTML] [--receipt OUTPUT.json]
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const hash = value => crypto.createHash('sha256').update(value).digest('hex');
const normalize = value => value.replace(/\r\n/g, '\n').trim();
const args = process.argv.slice(2);
let htmlPath = path.resolve(__dirname, '..', 'ThrusterChain_diagnostics_results_v7_interactive.html');
let receiptPath = path.join(__dirname, 'lesson_restart_validation.json');
for (let i = 0; i < args.length; i++) {
  if (args[i] === '--receipt') receiptPath = path.resolve(args[++i]);
  else htmlPath = path.resolve(args[i]);
}
const sourcePath = path.join(__dirname, 'lesson_restart.js');
const source = fs.readFileSync(sourcePath, 'utf8');
const htmlRaw = fs.readFileSync(htmlPath);
const html = htmlRaw.toString('utf8');
const lessonKeys = ['tc-learn-last', 'tc-learn-seen', 'tc-learn-quiz'];
const expectedSelector = '[data-lesson-restart], #learn-recap .ls-again[href="#learn-world"]';
const receipt = {
  schema: 'thrusterchain-lesson-restart-validation-v1',
  generatedAt: new Date().toISOString(),
  html: htmlPath,
  htmlSHA256: hash(htmlRaw),
  source: sourcePath,
  sourceSHA256: hash(source),
  validation: 'Static markup/source checks and actual restart-script execution in Node VM using storage, click-event, history, and reload doubles. No browser or rendering.',
  limitations: ['Does not execute either full lesson/navigation controller', 'Does not verify rendered placement, browser reload behavior, or assistive technology'],
  checks: {}, errors: [],
};
function check(name, fn) {
  try { receipt.checks[name] = { passed: true, ...fn() }; }
  catch (error) { receipt.checks[name] = { passed: false }; receipt.errors.push({ check: name, message: error.message }); }
}
function attrs(text) {
  const out = {};
  for (const m of text.matchAll(/([^\s=/>]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+)))?/g)) out[m[1]] = m[2] ?? m[3] ?? m[4] ?? '';
  return out;
}
const markup = html.replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, '').replace(/<style\b[^>]*>[\s\S]*?<\/style>/gi, '').replace(/<!--[\s\S]*?-->/g, '');
const voidTags = new Set(['area','base','br','col','embed','hr','img','input','link','meta','param','source','track','wbr']);
const nodes = [], stack = [];
for (const token of markup.matchAll(/<(\/?)([a-zA-Z][\w:-]*)\b((?:[^>"']|"[^"]*"|'[^']*')*)>/g)) {
  const [, closing, rawTag, attributes] = token;
  const tag = rawTag.toLowerCase();
  if (closing) {
    const index = stack.map(n => n.tag).lastIndexOf(tag);
    if (index >= 0) stack.splice(index);
    continue;
  }
  const node = { tag, attrs: attrs(attributes), ancestors: stack.slice() };
  nodes.push(node);
  if (!voidTags.has(tag) && !/\/\s*$/.test(attributes)) stack.push(node);
}
const hasClass = (node, value) => (node.attrs.class || '').split(/\s+/).includes(value);
const steps = nodes.filter(node => hasClass(node, 'ls-step') && node.attrs.id).map(node => node.attrs.id);
const unrelated = {
  'tc-whole': '1',
  'tc-last': '{"id":"results-explorer","u":"Explore saved results","l":"State map"}',
  'tc-contents-wide': '0',
  'tc-side-wide': '0',
  'thrusterchain-tooltips': 'off',
  'unrelated-test-preference': 'retain-me',
};
function seededStorage() {
  return new Map(Object.entries({
    ...unrelated,
    'tc-learn-last': 'learn-recap',
    'tc-learn-seen': steps.join(' '),
    'tc-learn-quiz': JSON.stringify({ 'learn-world': 1, 'learn-recap': 0 }),
  }));
}
function runController({ denied = false, denyFirstOnly = false, historyDenied = false } = {}) {
  const storage = seededStorage(), events = [], registrations = [], removed = [], writes = [];
  let href = 'file:///C:/Users/David/dev/BCA/ThrusterChain_diagnostics_results_v7_interactive.html#learn-recap';
  let reloads = 0;
  const location = {
    get href() { return href; }, set href(value) { href = new URL(value, href).href; },
    get hash() { return new URL(href).hash; }, set hash(value) { const url = new URL(href); url.hash = value; href = url.href; events.push(['setHash', value]); },
    reload() { reloads++; events.push(['reload', href]); },
  };
  const history = {
    replaceState(state, unused, url) {
      events.push(['replaceState', state, unused, url]);
      if (historyDenied) throw new Error('History access denied');
      href = new URL(url, href).href;
    },
  };
  const localStorage = {
    getItem(key) { return storage.get(key) ?? null; },
    setItem(key, value) { writes.push([key, value]); storage.set(key, String(value)); },
    removeItem(key) {
      removed.push(key); events.push(['removeItem', key]);
      if (denied || (denyFirstOnly && removed.length === 1)) throw new Error('Storage access denied');
      storage.delete(key);
    },
    clear() { assert.fail('Restart must not clear unrelated localStorage'); },
  };
  const document = {
    addEventListener(type, callback, options) { registrations.push({ type, callback, options }); },
  };
  const window = { document, localStorage, history, location };
  vm.runInNewContext(source, { window, document, localStorage, history, location }, { filename: 'lesson_restart.js', timeout: 5000 });
  const click = registrations.find(registration => registration.type === 'click');
  assert.ok(click, 'Missing document click listener');
  const capture = click.options === true || Boolean(click.options && click.options.capture);
  assert.equal(capture, true, 'Restart must intercept before older fragment-link controllers');
  function dispatch(kind) {
    let prevented = false, stopped = false, selectorSeen = null;
    const matched = kind === 'landing-button' || kind === 'lesson-button' || kind === 'recap-link';
    const event = {
      button: 0, defaultPrevented: false,
      target: {
        closest(selector) {
          selectorSeen = selector;
          assert.equal(selector, expectedSelector);
          return matched ? { kind } : null;
        },
      },
      preventDefault() { prevented = true; this.defaultPrevented = true; events.push(['preventDefault']); },
      stopImmediatePropagation() { stopped = true; events.push(['stopImmediatePropagation']); },
    };
    click.callback(event);
    return { prevented, stopped, selectorSeen };
  }
  return { storage, removed, writes, events, registrations, dispatch, location, get reloads() { return reloads; } };
}
function preserveUnrelated(run) {
  for (const [key, value] of Object.entries(unrelated)) assert.equal(run.storage.get(key), value, 'Unrelated preference changed: ' + key);
  assert.equal(run.writes.length, 0, 'Reset must remove lesson progress rather than rewrite unrelated state');
}
function navigationReachedStart(run, expectedReloads = 1) {
  const url = new URL(run.location.href);
  assert.equal(url.protocol, 'file:');
  assert.equal(url.pathname, '/C:/Users/David/dev/BCA/ThrusterChain_diagnostics_results_v7_interactive.html');
  assert.equal(url.hash, '#learn-world'); assert.equal(run.reloads, expectedReloads);
  const replace = run.events.filter(event => event[0] === 'replaceState');
  assert.equal(replace.length, expectedReloads);
  replace.forEach(event => assert.equal(JSON.stringify(event.slice(1)), JSON.stringify([null, '', '#learn-world'])));
  const lastReload = run.events.map(event => event[0]).lastIndexOf('reload');
  const lastReplace = run.events.map(event => event[0]).lastIndexOf('replaceState');
  assert.ok(lastReplace < lastReload, 'Replace the fragment before reloading');
}

check('sourceAndEmbed', () => {
  new vm.Script(source, { filename: 'lesson_restart.js' });
  const scripts = [...html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/gi)].filter(m => !/application\/json/.test(m[1]));
  const matches = scripts.filter(m => normalize(m[2]) === normalize(source));
  assert.equal(matches.length, 1, 'Page must contain exactly one copy of the current restart source');
  return { compiled: true, matchingEmbeddedScripts: matches.length, embeddedScriptID: attrs(matches[0][1]).id || null };
});
check('persistentNativeButtons', () => {
  const buttons = nodes.filter(node => Object.hasOwn(node.attrs, 'data-lesson-restart'));
  assert.equal(buttons.length, 2, 'Expected landing and persistent lesson restart buttons');
  for (const button of buttons) {
    assert.equal(button.tag, 'button'); assert.equal(button.attrs.type, 'button');
    assert.ok(!button.ancestors.some(node => hasClass(node, 'ls-intro') || hasClass(node, 'ls-steps')), 'Restart hidden inside a lesson page');
  }
  assert.ok(buttons.some(button => button.ancestors.some(node => node.attrs.id === 'top')));
  assert.ok(buttons.some(button => button.ancestors.some(node => node.attrs.id === 'learn')));
  assert.equal(steps.length, 39);
  assert.ok(steps.includes('learn-world')); assert.ok(steps.includes('learn-recap'));
  return { nativeButtons: 2, outsideHiddenLessonPages: true, seedVisitedSteps: steps.length };
});
check('resetFromEachEntryPoint', () => {
  for (const entry of ['landing-button','lesson-button','recap-link']) {
    const run = runController(); const event = run.dispatch(entry);
    assert.equal(event.prevented, true); assert.equal(event.stopped, true);
    assert.equal(run.removed.join(','), lessonKeys.join(','));
    lessonKeys.forEach(key => assert.equal(run.storage.has(key), false));
    preserveUnrelated(run); navigationReachedStart(run);
  }
  return { entryPoints: 3, capturedAndStopped: true, removedKeys: lessonKeys, preservedPreferences: Object.keys(unrelated).length };
});
check('repeatReset', () => {
  const run = runController(); run.dispatch('lesson-button'); run.dispatch('recap-link');
  assert.equal(run.removed.join(','), [...lessonKeys, ...lessonKeys].join(','));
  lessonKeys.forEach(key => assert.equal(run.storage.has(key), false));
  preserveUnrelated(run); navigationReachedStart(run, 2);
  return { invocations: 2, idempotentStorageRemoval: true };
});
check('unrelatedClick', () => {
  const run = runController(); const before = JSON.stringify([...run.storage]); const event = run.dispatch('unrelated-link');
  assert.equal(event.prevented, false); assert.equal(event.stopped, false);
  assert.equal(run.removed.length, 0); assert.equal(run.reloads, 0); assert.equal(run.events.length, 0);
  assert.equal(JSON.stringify([...run.storage]), before); assert.equal(run.location.hash, '#learn-recap');
  return { untouched: true };
});
check('deniedStorage', () => {
  for (const options of [{ denied: true }, { denyFirstOnly: true }]) {
    const run = runController(options); const event = run.dispatch('lesson-button');
    assert.equal(event.prevented, true); assert.equal(event.stopped, true);
    assert.equal(run.removed.join(','), lessonKeys.join(','), 'Each removal must be guarded independently');
    preserveUnrelated(run); navigationReachedStart(run);
    if (options.denied) lessonKeys.forEach(key => assert.equal(run.storage.has(key), true));
    else { assert.equal(run.storage.has(lessonKeys[0]), true); lessonKeys.slice(1).forEach(key => assert.equal(run.storage.has(key), false)); }
  }
  return { scenarios: 2, eachKeyAttempted: true, reloadStillRuns: true };
});
receipt.passed = receipt.errors.length === 0;
fs.writeFileSync(receiptPath, JSON.stringify(receipt, null, 2) + '\n');
console.log(JSON.stringify({ passed: receipt.passed, receipt: receiptPath, checks: receipt.checks, errors: receipt.errors }, null, 2));
if (!receipt.passed) process.exitCode = 1;
