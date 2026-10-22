#!/usr/bin/env node
'use strict';

// Static source compilation and two focused interaction checks. This does not
// create a browser, load a URL, render HTML/CSS, or verify browser accessibility.
// With no arguments, checks sibling V5 and V6 and writes the sibling receipt.
// Usage: node validate_interactions.cjs [HTML ...] [--receipt OUTPUT.json]

const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const sha256 = value => crypto.createHash('sha256').update(value).digest('hex');
const presentation = path.dirname(__dirname);
const defaults = [
  'ThrusterChain_diagnostics_results_v5_fresh_replication.html',
  'ThrusterChain_diagnostics_results_v6_reviewed.html',
].map(name => path.join(presentation, name));

function attributes(source) {
  const attrs = {};
  for (const match of source.matchAll(/([^\s=]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+)))?/g)) {
    attrs[match[1].toLowerCase()] = match[2] ?? match[3] ?? match[4] ?? '';
  }
  return attrs;
}

function extractScripts(html) {
  return [...html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/gi)].map((match, index) => ({
    index,
    line: html.slice(0, match.index).split('\n').length,
    attrs: attributes(match[1]),
    source: match[2],
  }));
}

class Element {
  constructor(tag = 'div') {
    this.tagName = tag;
    this.children = [];
    this.value = '';
    this.textContent = '';
    this.listeners = new Map();
  }
  appendChild(child) {
    this.children.push(child);
    if (this.tagName === 'select' && (child.selected || this.children.length === 1)) {
      this.value = child.value;
    }
    return child;
  }
  replaceChildren(...children) { this.children = children; }
  addEventListener(event, callback) {
    const callbacks = this.listeners.get(event) || [];
    callbacks.push(callback);
    this.listeners.set(event, callbacks);
  }
  dispatch(event) {
    const callbacks = this.listeners.get(event) || [];
    assert.ok(callbacks.length, 'Expected a ' + event + ' listener');
    for (const callback of callbacks) callback();
  }
}

function selectFromHTML(html, id) {
  const found = [...html.matchAll(/<select\b([^>]*)>([\s\S]*?)<\/select>/gi)]
    .find(match => attributes(match[1]).id === id);
  assert.ok(found, 'Missing select: ' + id);
  const select = new Element('select');
  for (const match of found[2].matchAll(/<option\b([^>]*)>([\s\S]*?)<\/option>/gi)) {
    const attrs = attributes(match[1]);
    const option = new Element('option');
    option.value = attrs.value ?? match[2];
    option.textContent = match[2];
    option.selected = Object.hasOwn(attrs, 'selected');
    select.appendChild(option);
  }
  return select;
}

function oracleSweep(html, scripts) {
  const byId = id => {
    const item = scripts.find(script => script.attrs.id === id);
    assert.ok(item, 'Missing script: ' + id);
    return item.source;
  };
  const nodes = new Map();
  const fieldNames = ['regime', 'load', 'position', 'time', 'retention'];
  for (const field of fieldNames) {
    nodes.set('diag-' + field, selectFromHTML(html, 'diag-' + field));
  }
  for (const id of ['diag-oracle-rows', 'diag-oracle-context', 'diag-oracle-witness', 'diagnostics-oracle-data']) {
    nodes.set(id, new Element());
  }
  nodes.get('diagnostics-oracle-data').textContent = byId('diagnostics-oracle-data');
  const data = JSON.parse(byId('diagnostics-oracle-data'));
  const context = {
    document: {
      getElementById: id => nodes.get(id),
      createElement: tag => new Element(tag),
    },
  };
  vm.runInNewContext(byId('diagnostics-oracle-interaction'), context, {
    filename: 'diagnostics-oracle-interaction.js', timeout: 5000,
  });
  assert.equal(nodes.get('diag-position').children.length, data.positions.length);
  assert.equal(nodes.get('diag-time').children.length, data.horizon);
  let combinations = 0;
  for (const regime of [0, 1]) for (const load of [0, 1]) {
    for (let position = 0; position < data.positions.length; position++) {
      for (let time = 0; time < data.horizon; time++) for (const retention of ['full', 'rare', 'zero']) {
        const selection = { regime, load, position, time, retention };
        for (const [field, value] of Object.entries(selection)) nodes.get('diag-' + field).value = String(value);
        nodes.get('diag-regime').dispatch('change');
        const state = data.states.findIndex(row => row[0] === regime && row[1] === load && row[2] === data.positions[position]);
        assert.ok(state >= 0, 'State missing: ' + JSON.stringify(selection));
        const values = data.q[time][state];
        const best = Math.max(...values);
        const rows = nodes.get('diag-oracle-rows').children;
        assert.equal(rows.length, 5, 'Expected five actions');
        rows.forEach((row, action) => {
          assert.equal(row.children.length, 6);
          assert.equal(row.children[0].scope, 'row');
          assert.ok(Number.isFinite(values[action]));
          assert.equal(row.children[1].textContent, values[action].toFixed(6) + (values[action] === best ? ' · best' : ''));
          assert.equal(row.children[5].textContent, String(data.counts[retention][time][state][action]));
          assert.equal(row.className === 'diag-best', values[action] === best);
          for (const cell of row.children) assert.ok(!/NaN|undefined/.test(String(cell.textContent)));
        });
        assert.ok(nodes.get('diag-oracle-context').textContent.includes('t=' + time));
        assert.ok(nodes.get('diag-oracle-context').textContent.endsWith(retention + ' retention.'));
        assert.ok(!/NaN|undefined/.test(nodes.get('diag-oracle-witness').textContent));
        combinations++;
      }
    }
  }
  assert.equal(combinations, 864);
  return { passed: true, combinations, actionsPerCombination: 5, mode: 'original script; minimal mocked DOM' };
}

function filterSweep(html, scripts) {
  const script = scripts.find(item => item.attrs.id === 'tr-results-filter-script');
  assert.ok(script, 'Missing training filter script');
  const table = [...html.matchAll(/<table\b([^>]*)>([\s\S]*?)<\/table>/gi)]
    .find(match => attributes(match[1]).id === 'tr-policy-table');
  assert.ok(table, 'Missing training results table');
  const rows = [...table[2].matchAll(/<tr\b([^>]*)>/gi)].map(match => attributes(match[1]))
    .filter(attrs => attrs['data-tr-retention'] && attrs['data-tr-context'])
    .map(attrs => ({
      dataset: { trRetention: attrs['data-tr-retention'], trContext: attrs['data-tr-context'] },
      hidden: false,
    }));
  assert.equal(rows.length, 150);
  const nodes = new Map([
    ['tr-retention-filter', selectFromHTML(html, 'tr-retention-filter')],
    ['tr-context-filter', selectFromHTML(html, 'tr-context-filter')],
    ['tr-filter-count', new Element()],
  ]);
  vm.runInNewContext(script.source, {
    document: {
      getElementById: id => nodes.get(id),
      querySelectorAll: selector => {
        assert.equal(selector, '#tr-policy-table tbody tr');
        return rows;
      },
    },
  }, { filename: 'tr-results-filter-script.js', timeout: 5000 });
  const visibleCounts = new Set();
  let combinations = 0;
  for (const retention of ['all', 'full', 'rare', 'zero']) {
    for (const context of ['all', 'pooled', 'stall_low', 'stall_high', 'monotone_low', 'monotone_high']) {
      nodes.get('tr-retention-filter').value = retention;
      nodes.get('tr-context-filter').value = context;
      nodes.get('tr-retention-filter').dispatch('change');
      const visible = rows.filter(row => !row.hidden).length;
      const expected = (retention === 'all' ? 30 : 10) * (context === 'all' ? 5 : 1);
      assert.equal(visible, expected, 'Wrong count for ' + retention + '/' + context);
      for (const row of rows) {
        assert.equal(!row.hidden,
          (retention === 'all' || row.dataset.trRetention === retention) &&
          (context === 'all' || row.dataset.trContext === context));
      }
      assert.equal(nodes.get('tr-filter-count').textContent, visible + ' of 150 saved policy/context rows shown.');
      visibleCounts.add(visible);
      combinations++;
    }
  }
  assert.equal(combinations, 24);
  return { passed: true, combinations, savedRows: rows.length, visibleCounts: [...visibleCounts].sort((a, b) => a - b), mode: 'original script; minimal mocked DOM' };
}

function validate(filename) {
  const raw = fs.readFileSync(filename);
  const html = raw.toString('utf8');
  const scripts = extractScripts(html);
  const result = { file: path.resolve(filename), bytes: raw.length, sha256: sha256(raw), errors: [] };
  const executable = scripts.filter(script => script.attrs.type !== 'application/json');
  const json = scripts.filter(script => script.attrs.type === 'application/json');
  for (const script of executable) {
    try { new vm.Script(script.source, { filename: path.basename(filename) + ':line-' + script.line }); }
    catch (error) { result.errors.push({ check: 'javascript_compile', line: script.line, message: error.message }); }
  }
  for (const script of json) {
    try { JSON.parse(script.source); }
    catch (error) { result.errors.push({ check: 'json_parse', id: script.attrs.id, line: script.line, message: error.message }); }
  }
  result.javascriptCompile = { checked: executable.length, expected: 19 };
  result.embeddedJSON = {
    checked: json.length,
    expected: 11,
    sourceHashes: Object.fromEntries(json.map(script => [script.attrs.id, sha256(script.source)])),
  };
  if (executable.length !== 19 || json.length !== 11) result.errors.push({ check: 'script_inventory', message: 'Expected 19 executable blocks and 11 JSON blocks' });
  for (const [name, check] of [['oracle', oracleSweep], ['trainingFilter', filterSweep]]) {
    try { result[name] = check(html, scripts); }
    catch (error) { result[name] = { passed: false }; result.errors.push({ check: name, message: error.message }); }
  }
  result.passed = result.errors.length === 0;
  return result;
}

const files = [];
let receiptPath = path.join(__dirname, 'interaction_validation.json');
const args = process.argv.slice(2);
for (let i = 0; i < args.length; i++) {
  if (args[i] === '--receipt') {
    assert.ok(args[i + 1], '--receipt needs a file path');
    receiptPath = path.resolve(args[++i]);
  } else files.push(path.resolve(args[i]));
}
const receipt = {
  schema: 'thrusterchain-static-interaction-validation-v1',
  generatedAt: new Date().toISOString(),
  validation: 'Source compilation, JSON parsing, and focused original-script execution with a minimal mocked DOM. No browser or headless rendering.',
  limitations: ['No CSS/layout/contrast rendering check', 'No full navigation runtime check', 'No actual browser keyboard or assistive-technology check'],
  results: (files.length ? files : defaults).map(validate),
};
if (receipt.results.length === 2) {
  const [a, b] = receipt.results;
  const keys = new Set([...Object.keys(a.embeddedJSON.sourceHashes), ...Object.keys(b.embeddedJSON.sourceHashes)]);
  receipt.embeddedJSONUnchangedBetweenFiles = [...keys].every(key => a.embeddedJSON.sourceHashes[key] === b.embeddedJSON.sourceHashes[key]);
}
receipt.passed = receipt.results.every(result => result.passed);
fs.writeFileSync(receiptPath, JSON.stringify(receipt, null, 2) + '\n');
console.log(JSON.stringify({
  passed: receipt.passed,
  receipt: receiptPath,
  embeddedJSONUnchangedBetweenFiles: receipt.embeddedJSONUnchangedBetweenFiles,
  results: receipt.results.map(result => ({ file: path.basename(result.file), passed: result.passed, javascriptBlocks: result.javascriptCompile.checked, jsonBlocks: result.embeddedJSON.checked, oracleCombinations: result.oracle.combinations, filterCombinations: result.trainingFilter.combinations, errors: result.errors })),
}, null, 2));
if (!receipt.passed) process.exitCode = 1;
