#!/usr/bin/env node
'use strict';

// Executes the saved explorer controller against a small in-memory DOM model.
// This is not a browser, headless renderer, layout check, or accessibility audit.
// Usage: node validate_explorer.cjs [HTML] [--receipt OUTPUT.json]
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const hash = value => crypto.createHash('sha256').update(value).digest('hex');
const args = process.argv.slice(2);
let input = path.join(__dirname, '..', 'ThrusterChain_diagnostics_results_v7_interactive.html');
let receiptPath = path.join(__dirname, 'explorer_validation.json');
for (let i = 0; i < args.length; i++) {
  if (args[i] === '--receipt') receiptPath = path.resolve(args[++i]);
  else input = path.resolve(args[i]);
}
const raw = fs.readFileSync(input);
const html = raw.toString('utf8');
const receipt = {
  schema: 'thrusterchain-explorer-mocked-dom-validation-v1',
  generatedAt: new Date().toISOString(),
  file: input,
  bytes: raw.length,
  sha256: hash(raw),
  validation: 'Static source checks and original explorer-controller execution in a minimal mocked DOM. No browser, URL loading, or headless rendering.',
  limitations: ['No CSS or layout validation', 'No full navigation or tooltip integration execution', 'No browser keyboard or assistive-technology validation'],
  checks: {},
  errors: [],
};
function check(name, fn) {
  try { receipt.checks[name] = { passed: true, ...fn() }; }
  catch (error) { receipt.checks[name] = { passed: false }; receipt.errors.push({ check: name, message: error.stack }); }
}
function attrs(text) {
  const result = {};
  for (const m of text.matchAll(/([^\s=/>]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+)))?/g)) {
    result[m[1]] = m[2] ?? m[3] ?? m[4] ?? '';
  }
  return result;
}
const scripts = [...html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/gi)].map(m => ({ attrs: attrs(m[1]), source: m[2] }));
const byId = id => {
  const found = scripts.find(s => s.attrs.id === id);
  assert.ok(found, 'Missing script ' + id);
  return found.source;
};
check('assembledInputsCurrent', () => {
  const normalize = source => source.replace(/\r\n/g, '\n').trim();
  assert.equal(hash(normalize(byId('xe-interaction'))), hash(normalize(fs.readFileSync(path.join(__dirname, 'explorer.js'), 'utf8'))), 'Embedded JavaScript differs from explorer.js');
  const style = html.match(/<style id="v7-explorer-style">([\s\S]*?)<\/style>/);
  assert.ok(style, 'Missing explorer style');
  assert.equal(hash(normalize(style[1])), hash(normalize(fs.readFileSync(path.join(__dirname, 'explorer.css'), 'utf8'))), 'Embedded CSS differs from explorer.css');
  assert.ok(normalize(html).includes(normalize(fs.readFileSync(path.join(__dirname, 'explorer.html'), 'utf8'))), 'Embedded markup differs from explorer.html');
  assert.equal(hash(JSON.stringify(JSON.parse(byId('xe-data')))), hash(JSON.stringify(JSON.parse(fs.readFileSync(path.join(__dirname, 'data.json'), 'utf8')))), 'Embedded JSON differs from data.json');
  return { filesMatched: 4 };
});
check('allJavaScriptAndJSON', () => {
  let javascript = 0, json = 0;
  for (const item of scripts) {
    if (item.attrs.type === 'application/json') { JSON.parse(item.source); json++; }
    else { new vm.Script(item.source); javascript++; }
  }
  assert.equal(javascript, 22); assert.equal(json, 12);
  return { javascriptBlocks: javascript, jsonBlocks: json };
});

// Deliberately bounded HTML-fragment model: only methods used by the explorer.
// Dynamic ID lookup is rebuilt on innerHTML changes, including removal of old IDs.
const registry = new Map(), downloads = [], timers = [], revoked = [], blobs = new Map();
let doc;
const voidTags = new Set(['area','base','br','col','embed','hr','img','input','link','meta','param','source','track','wbr']);
function unregister(node) {
  for (const child of node.children) unregister(child);
  if (node.id && registry.get(node.id) === node) registry.delete(node.id);
}
class Element {
  constructor(tag = 'div') {
    this.tagName = tag.toUpperCase(); this.children = []; this.parentElement = null;
    this.attributes = {}; this.listeners = new Map(); this._text = ''; this._html = '';
    this.value = ''; this.checked = false; this.hidden = false; this.disabled = false;
    this.scrollCalls = []; this.dataset = {};
  }
  get id() { return this.attributes.id || ''; }
  set id(value) { this.setAttribute('id', value); }
  setAttribute(key, value) {
    const oldId = this.id; this.attributes[key] = String(value);
    if (key === 'id') { if (oldId && registry.get(oldId) === this) registry.delete(oldId); registry.set(String(value), this); }
    if (key === 'value') this.value = String(value);
    if (key.startsWith('data-')) this.dataset[key.slice(5).replace(/-([a-z])/g, (_, c) => c.toUpperCase())] = String(value);
    if (key === 'hidden') this.hidden = true;
    if (key === 'checked') this.checked = true;
    if (key === 'disabled') this.disabled = true;
  }
  getAttribute(key) { return Object.hasOwn(this.attributes, key) ? this.attributes[key] : null; }
  removeAttribute(key) { delete this.attributes[key]; if (key === 'hidden') this.hidden = false; }
  appendChild(child) {
    child.parentElement = this; this.children.push(child);
    if (this.tagName === 'SELECT' && child.tagName === 'OPTION' && (this.children.length === 1 || Object.hasOwn(child.attributes, 'selected'))) this.value = child.value;
    return child;
  }
  replaceChildren(...children) {
    for (const child of this.children) unregister(child);
    this.children = []; for (const child of children) this.appendChild(child);
  }
  set innerHTML(value) { this.replaceChildren(); this._html = String(value); this._text = ''; parseFragment(this._html, this); }
  get innerHTML() { return this._html; }
  set textContent(value) { this.replaceChildren(); this._text = String(value); this._html = ''; }
  get textContent() { return this._text + this.children.map(c => c.textContent).join(''); }
  addEventListener(type, callback) { const list = this.listeners.get(type) || []; list.push(callback); this.listeners.set(type, list); }
  dispatch(type, extra = {}) {
    const event = { type, target: this, currentTarget: this, defaultPrevented: false, preventDefault() { this.defaultPrevented = true; }, ...extra };
    const callbacks = this.listeners.get(type) || [];
    assert.ok(callbacks.length, 'Missing ' + type + ' listener on ' + this.id);
    for (const callback of callbacks) callback(event);
    return event;
  }
  closest(selector) {
    for (let node = this; node; node = node.parentElement) {
      if (selector.startsWith('[') && Object.hasOwn(node.attributes, selector.slice(1, -1))) return node;
      if (selector.startsWith('#') && node.id === selector.slice(1)) return node;
    }
    return null;
  }
  querySelectorAll(selector) {
    const found = [];
    const matches = node => selector === 'input[name="xe-method"]' ? node.tagName === 'INPUT' && node.attributes.name === 'xe-method' : false;
    const walk = parent => { for (const child of parent.children) { if (matches(child)) found.push(child); walk(child); } };
    walk(this); return found;
  }
  focus(options) { doc.activeElement = this; this.focusOptions = options; }
  scrollIntoView(options) { this.scrollCalls.push(options); }
  click() {
    if (this.disabled) return;
    if (this.tagName === 'A') { downloads.push({ href: this.href, download: this.download, attached: this.parentElement === doc.body }); return; }
    return this.dispatch('click');
  }
  remove() {
    if (this.parentElement) this.parentElement.children = this.parentElement.children.filter(c => c !== this);
    unregister(this); this.parentElement = null;
  }
}
function parseFragment(source, parent) {
  const stack = [parent];
  for (const token of source.matchAll(/<!--[\s\S]*?-->|<[^>]+>|[^<]+/g)) {
    const value = token[0];
    if (value.startsWith('<!--')) continue;
    if (!value.startsWith('<')) { stack.at(-1)._text += value; continue; }
    if (value.startsWith('</')) { if (stack.length > 1) stack.pop(); continue; }
    const m = /^<([\w:-]+)\b([\s\S]*?)\/?\s*>$/.exec(value);
    if (!m) continue;
    const element = new Element(m[1]);
    for (const [key, val] of Object.entries(attrs(m[2]))) element.setAttribute(key, val);
    stack.at(-1).appendChild(element);
    if (!voidTags.has(m[1].toLowerCase()) && !value.endsWith('/>')) stack.push(element);
  }
}
doc = { body: new Element('body'), activeElement: null, getElementById: id => registry.get(id) || null, createElement: tag => new Element(tag) };
const component = html.match(/<section id="results-explorer"[\s\S]*?<\/section>/);
assert.ok(component, 'Missing static component'); parseFragment(component[0], doc.body);
const dataNode = new Element('script'); dataNode.id = 'xe-data'; dataNode.textContent = byId('xe-data'); doc.body.appendChild(dataNode);
const sourceData = JSON.parse(byId('xe-data'));
const immutableData = JSON.stringify(sourceData);
let reducedMotion = true;
const win = { matchMedia: () => ({ matches: reducedMotion, addEventListener() {}, removeEventListener() {} }) };
const sandbox = {
  document: doc, window: win,
  matchMedia: win.matchMedia,
  requestAnimationFrame: callback => { callback(); return 1; },
  Blob: class { constructor(parts, options) { this.parts = parts; this.type = options.type; } },
  URL: { createObjectURL: blob => { const url = 'blob:mock-' + blobs.size; blobs.set(url, blob); return url; }, revokeObjectURL: url => revoked.push(url) },
  setTimeout: callback => { timers.push(callback); return timers.length; },
};
check('initialization', () => {
  vm.runInNewContext(byId('xe-interaction'), sandbox, { filename: 'xe-interaction.js', timeout: 5000 });
  assert.ok(win.ThrusterExplorer);
  const state = win.ThrusterExplorer.getState();
  assert.equal(state.method, 'live_width'); assert.equal(state.context, 'all'); assert.equal(state.time, 'all');
  assert.equal(state.stateId, 24); assert.equal(state.candidateId, 7); assert.equal(state.sort, 'id');
  assert.equal(doc.getElementById('xe-app').hidden, false);
  assert.equal(doc.getElementById('xe-state-grid').children.length, 32);
  assert.equal(doc.getElementById('xe-candidate-body').children.length, 12);
  return { stateId: state.stateId, candidateId: state.candidateId, states: 32, candidates: 12 };
});
const api = win.ThrusterExplorer;
const $ = id => { const element = doc.getElementById(id); assert.ok(element, 'Missing element ' + id); return element; };
const methods = ['raw_q','live_width','matched_distance','shuffled_width','constant_offset'];
const contexts = ['all','stall_low','stall_high','monotone_low','monotone_high'];
const times = ['all','0','50','99'];
const sorts = ['id','score','truth','width'];
const setSelect = (id, value) => { $(id).value = value; $(id).dispatch('change'); };
const setMethod = method => {
  const inputs = $('results-explorer').querySelectorAll('input[name="xe-method"]');
  for (const input of inputs) input.checked = input.value === method;
  inputs.find(input => input.value === method).dispatch('change');
};
const reset = () => $('xe-reset').dispatch('click');
const chooseState = id => $('xe-state-grid').dispatch('click', { target: $('xe-state-' + id) });
const approx = (a, b, message) => assert.ok(Math.abs(a - b) < 1e-11, (message || 'Values differ') + ': ' + a + ' vs ' + b);
const mean = list => list.reduce((sum, value) => sum + value, 0) / list.length;
const nearText = (id, expected, tolerance) => assert.ok(Math.abs(parseFloat($(id).textContent) - expected) <= tolerance, id + ': ' + $(id).textContent + ' vs ' + expected);

check('sourceHashesAndSavedWinners', () => {
  const resultsDir = path.resolve(__dirname, '..', '..', 'results');
  for (const [name, expected] of Object.entries(sourceData.source_sha256)) assert.equal(hash(fs.readFileSync(path.join(resultsDir, name))), expected);
  let winners = 0;
  for (const state of sourceData.states) {
    assert.equal(state.candidates.length, 12);
    assert.equal(state.candidates.map(c => c.id).join(','), '0,1,2,3,4,5,6,7,8,9,10,11');
    for (const method of methods) {
      const best = state.candidates.reduce((winner, candidate) => candidate.scores[method] > winner.scores[method] ? candidate : winner, state.candidates[0]);
      assert.equal(state.methods[method].chosen_index, best.id, 'Saved winner differs from full precision argmax: ' + state.id + '/' + method);
      approx(state.methods[method].chosen_true_q, best.true_policy_q);
      const rawWinner = state.candidates[state.methods.raw_q.chosen_index];
      approx(state.methods[method].delta_vs_raw, best.true_policy_q - rawWinner.true_policy_q);
      winners++;
    }
  }
  return { sourceFiles: Object.keys(sourceData.source_sha256).length, savedWinners: winners, candidateRecords: 384 };
});

check('filteredSummariesAndCases', () => {
  let combinations = 0, caseActions = 0;
  for (const method of methods) for (const context of contexts) for (const time of times) {
    setMethod(method); setSelect('xe-context', context); setSelect('xe-time', time);
    const expected = sourceData.states.filter(s => s.valid && (context === 'all' || s.context === context) && (time === 'all' || s.time === Number(time)));
    const actual = api.filterStates(context, time);
    assert.equal(actual.map(s => s.id).join(','), expected.map(s => s.id).join(','));
    assert.ok(expected.length > 0);
    const deltas = expected.map(s => s.methods[method].delta_vs_raw);
    const summary = api.summarize(actual, method);
    approx(summary.gain, mean(deltas)); assert.equal(summary.n, expected.length);
    assert.equal(summary.improved, deltas.filter(x => x > 0).length);
    assert.equal(summary.harmed, deltas.filter(x => x < 0).length);
    assert.equal(summary.unchanged, deltas.filter(x => x === 0).length);
    assert.equal(summary.changedCommands, expected.filter(s => s.methods[method].changed_command).length);
    const ranking = expected.filter(s => Number.isFinite(s.methods[method].concordance) && Number.isFinite(s.methods.raw_q.concordance));
    approx(summary.concordanceChange, mean(ranking.map(s => s.methods[method].concordance - s.methods.raw_q.concordance)));
    assert.equal(summary.rankingN, ranking.length);
    for (const cardMethod of methods) nearText('xe-mean-' + cardMethod, mean(expected.map(s => s.methods[cardMethod].delta_vs_raw)), 0.000051);
    nearText('xe-gain', summary.gain, 0.00000051);
    nearText('xe-ordering', summary.concordanceChange * 100, 0.000051);
    assert.equal($('xe-outcomes').textContent, summary.improved + ' / ' + summary.harmed + ' / ' + summary.unchanged);
    assert.ok(expected.some(s => s.id === api.getState().stateId));
    for (const sourceState of sourceData.states) assert.equal($('xe-state-' + sourceState.id).hidden, !expected.some(s => s.id === sourceState.id));
    for (const kind of ['gain','loss','same']) {
      const options = expected.filter(s => kind === 'gain' ? s.methods[method].delta_vs_raw > 0 : kind === 'loss' ? s.methods[method].delta_vs_raw < 0 : s.methods[method].delta_vs_raw === 0);
      assert.equal($('xe-case-' + kind).disabled, options.length === 0);
      if (!options.length) continue;
      const target = options.reduce((a, b) => kind === 'gain' ? (b.methods[method].delta_vs_raw > a.methods[method].delta_vs_raw ? b : a) : kind === 'loss' ? (b.methods[method].delta_vs_raw < a.methods[method].delta_vs_raw ? b : a) : (b.id < a.id ? b : a));
      $('xe-case-' + kind).dispatch('click'); assert.equal(api.getState().stateId, target.id);
      assert.equal(api.getState().candidateId, target.methods[method].chosen_index); caseActions++;
    }
    combinations++;
  }
  assert.equal(combinations, 100);
  return { combinations, enabledCaseActions: caseActions };
});

check('allStatesMethodsAndSorts', () => {
  reset(); let combinations = 0;
  for (const sourceState of sourceData.states) for (const method of methods) {
    chooseState(sourceState.id); setMethod(method);
    assert.equal(api.getState().candidateId, sourceState.methods[method].chosen_index);
    for (const sort of sorts) {
      setSelect('xe-sort', sort);
      const renderedIDs = $('xe-candidate-body').children.map(row => Number(row.children[0].children[0].getAttribute('data-candidate')));
      const expected = sourceState.candidates.slice().sort((a,b) => {
        const difference = sort === 'score' ? b.scores[method] - a.scores[method] : sort === 'truth' ? b.true_policy_q - a.true_policy_q : sort === 'width' ? a.width - b.width : a.id - b.id;
        return difference || a.id - b.id;
      });
      assert.equal(renderedIDs.join(','), expected.map(c => c.id).join(','));
      assert.equal(api.getState().candidateId, sourceState.methods[method].chosen_index);
      assert.equal($('xe-rule-choice').textContent, 'Candidate ' + sourceState.methods[method].chosen_index);
      assert.equal($('xe-detail-heading').textContent, 'State ' + sourceState.id);
      for (const c of sourceState.candidates) {
        assert.equal($('xe-candidate-' + c.id).getAttribute('aria-pressed'), String(c.id === api.getState().candidateId));
        assert.ok(Number.isFinite(Number($('xe-point-' + c.id).getAttribute('cx'))));
        assert.ok(Number.isFinite(Number($('xe-point-' + c.id).getAttribute('cy'))));
      }
      combinations++;
    }
  }
  assert.equal(combinations, 640);
  return { combinations, states: 32, methods: 5, sorts: 4 };
});

check('previousNextAndCandidateViews', () => {
  reset(); let transitions = 0;
  for (const context of contexts) for (const time of times) {
    setSelect('xe-context', context); setSelect('xe-time', time);
    const expected = api.filterStates(context, time); chooseState(expected[0].id);
    assert.equal($('xe-prev').disabled, true);
    for (let i = 1; i < expected.length; i++) { $('xe-next').dispatch('click'); assert.equal(api.getState().stateId, expected[i].id); transitions++; }
    assert.equal($('xe-next').disabled, true);
    for (let i = expected.length - 2; i >= 0; i--) { $('xe-prev').dispatch('click'); assert.equal(api.getState().stateId, expected[i].id); transitions++; }
  }
  reset(); let selections = 0;
  for (const candidateId of Array.from({ length: 12 }, (_, i) => i)) {
    $('xe-candidate-body').dispatch('click', { target: $('xe-candidate-' + candidateId) });
    assert.equal(api.getState().candidateId, candidateId); assert.equal(doc.activeElement.id, 'xe-candidate-' + candidateId);
    assert.ok($('xe-candidate-detail').innerHTML.includes('Candidate ' + candidateId + ': score arithmetic'));
    $('xe-view-errors').dispatch('click'); assert.equal($('xe-error-panel').hidden, false);
    const event = $('xe-scatter').dispatch('keydown', { target: $('xe-point-' + candidateId), key: candidateId % 2 ? ' ' : 'Enter' });
    assert.equal(event.defaultPrevented, true); assert.equal(doc.activeElement.id, 'xe-point-' + candidateId);
    $('xe-scatter').dispatch('click', { target: $('xe-point-' + candidateId) });
    assert.equal(api.getState().candidateId, candidateId);
    $('xe-view-choices').dispatch('click'); assert.equal($('xe-error-panel').hidden, true);
    selections++;
  }
  const ingredients = doc.getElementById('xe-ingredients');
  if (ingredients) {
    ingredients.checked = true; ingredients.dispatch('change'); assert.equal(api.getState().ingredients, true);
    assert.equal($('results-explorer').getAttribute('data-ingredients'), 'true');
    ingredients.checked = false; ingredients.dispatch('change'); assert.equal(api.getState().ingredients, false);
  }
  reset(); assert.equal(api.getState().stateId, 24); assert.equal(api.getState().candidateId, 7);
  assert.equal(api.getState().context, 'all'); assert.equal(api.getState().view, 'choices');
  if (ingredients) assert.equal(api.getState().ingredients, false);
  for (const reduce of [true, false]) {
    reducedMotion = reduce; chooseState(24);
    assert.equal(doc.activeElement.id, 'xe-detail-heading');
    assert.equal($('xe-detail-heading').getAttribute('tabindex'), '-1');
    assert.equal($('xe-detail-heading').scrollCalls.at(-1).behavior, reduce ? 'auto' : 'smooth');
  }
  reducedMotion = true;
  return { navigationTransitions: transitions, candidatesSelectedViaTableAndPlot: selections, views: 2, ingredientsToggle: Boolean(ingredients), detailFocusMotionModes: 2 };
});

function parseCSV(csv) {
  const rows = []; let row = [], field = '', quoted = false;
  for (let i = 0; i < csv.length; i++) {
    const c = csv[i];
    if (c === '"') { if (quoted && csv[i+1] === '"') { field += '"'; i++; } else quoted = !quoted; }
    else if (c === ',' && !quoted) { row.push(field); field = ''; }
    else if (c === '\n' && !quoted) { row.push(field.replace(/\r$/, '')); rows.push(row); row = []; field = ''; }
    else field += c;
  }
  return rows;
}
check('csvAndDownload', () => {
  let combinations = 0;
  for (const state of sourceData.states) for (const method of methods) {
    const rows = parseCSV(api.csvFor(state, method));
    assert.equal(rows.length, 13); assert.equal(rows[0].length, 17);
    const column = key => rows[0].indexOf(key);
    rows.slice(1).forEach((row, i) => {
      assert.equal(row.length, 17); assert.equal(Number(row[column('state_id')]), state.id);
      assert.equal(Number(row[column('candidate_id')]), i); assert.equal(row[column('method')], method);
      approx(Number(row[column('score')]), state.candidates[i].scores[method]);
      approx(Number(row[column('measured_qpi')]), state.candidates[i].true_policy_q);
      for (let j = 0; j < 4; j++) approx(Number(row[column('action_' + j)]), state.candidates[i].action[j]);
    });
    combinations++;
  }
  reset(); $('xe-export').dispatch('click'); assert.equal(downloads.length, 1);
  const download = downloads[0]; assert.equal(download.download, 'ThrusterChain_state_24_live_width.csv'); assert.equal(download.attached, true);
  const blob = blobs.get(download.href); assert.equal(blob.type, 'text/csv;charset=utf-8');
  assert.equal(blob.parts.join(''), api.csvFor(sourceData.states.find(s => s.id === 24), 'live_width'));
  timers.splice(0).forEach(fn => fn()); assert.ok(revoked.includes(download.href));
  assert.equal(JSON.stringify(api.data), immutableData, 'Controller mutated saved evidence');
  return { combinations, columns: 17, rowsPerCSVIncludingHeader: 13, mockedDownloads: downloads.length, savedDataUnchanged: true };
});

receipt.passed = receipt.errors.length === 0;
fs.writeFileSync(receiptPath, JSON.stringify(receipt, null, 2) + '\n');
console.log(JSON.stringify({ passed: receipt.passed, receipt: receiptPath, checks: receipt.checks, errors: receipt.errors }, null, 2));
if (!receipt.passed) process.exitCode = 1;
