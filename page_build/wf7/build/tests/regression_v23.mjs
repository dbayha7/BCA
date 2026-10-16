// Regression test of page version 24 against version 23 (test only; it reads the pages and writes only under
// wf7/build/tests/regression/). Three parts:
//  (a) every element id of version 23 exists in version 24, except the ones this build removes on purpose (REMOVED,
//      each with its reason);
//  (b) every line that versions 20-23 added (the lines of version 23 that version 19 does not have, counted with their
//      repeats) is in version 24 unchanged, once the attributes this build adds to an opening tag are set aside (an id
//      and a short label from the navigation's static edits, a color key from the color patch); a line that is not is
//      allowed only when it is one of the deliberate changes listed in REBASE.md (CHANGED, each with its reason and a
//      check that its replacement is there);
//  (c) in headless Edge at 390x844 and 1280x900, light and dark, the features of versions 20-23 work in version 24 as
//      in version 23: Who drives (the learned actor) for two data qualities and both test mixes, the switch bar, the
//      Observation Space range bars and their tooltips, the arm tables' tooltips beside their cells, the learning
//      curves' table and its selectors, version 23's notes and column tooltips, and the Choices rows (with "not in use
//      now"); no console errors, no sideways page scroll.
// Usage: node regression_v23.mjs   -> regression/regression_v23.json, exit 0 when everything passes
import fs from 'node:fs';
import { launch, openPage, navigate, evaluate, closePage, sleep } from '../../nav/tools/cdp.mjs';

const SCR = 'C:/Users/David/AppData/Local/Temp/claude/C--Users-David-dev-BCA/3bdab4f0-86ee-439c-a05a-34828d6b0a71/scratchpad';
const W7 = SCR + '/wf7';
const OUT = W7 + '/build/tests/regression';
fs.mkdirSync(OUT, { recursive: true });
const read = p => fs.readFileSync(p, 'utf8');
const V19 = read(SCR + '/live3/core_9369.html'), V23 = read(SCR + '/live4/core_6eea.html'), V24 = read(W7 + '/build/core_v24.html');
const STATIC = JSON.parse(read(W7 + '/nav/src/static_edits.json')).edits;
const res = { a: {}, b: {}, c: {}, ok: false };

// ------------------------------------------------------------------------------------------------ (a) ids
function staticIds(html) {        // element ids in the markup (script and style contents are not markup)
  const s = html.replace(/(<script\b[^>]*>)[\s\S]*?(<\/script>)/gi, '$1$2').replace(/(<style\b[^>]*>)[\s\S]*?(<\/style>)/gi, '$1$2');
  const ids = new Map();
  for (const m of s.matchAll(/<[a-zA-Z][\w:-]*(?:\s[^>]*?)?\sid="([^"]*)"/g)) ids.set(m[1], (ids.get(m[1]) || 0) + 1);
  return ids;
}
const REMOVED = {
  sb: 'E1: the version-18 Choices sidebar is replaced by the choices under each panel in "In this section" (REBASE.md R4)',
  'sb-tab': 'E1: the sidebar\'s tab (version 21\'s script that moved it aside goes with it, REBASE.md R5)',
  'sb-panel': 'E1: the sidebar', 'sb-title': 'E1: the sidebar', 'sb-x': 'E1: the sidebar', 'sb-body': 'E1: the sidebar',
  'wt-open': 'E2: the header\'s "Full walkthrough mode" button; the start bar\'s "Learn step by step" replaces it',
  wt: 'E5: the walkthrough dialog (the lesson and one-section display replace it)', 'wt-count': 'E5: the walkthrough dialog',
  'wt-title': 'E5: the walkthrough dialog', 'wt-close': 'E5: the walkthrough dialog', 'wt-bar': 'E5: the walkthrough dialog',
  'wt-body': 'E5: the walkthrough dialog', 'wt-stage': 'E5: the walkthrough dialog', 'wt-prev': 'E5: the walkthrough dialog',
  'wt-next-name': 'E5: the walkthrough dialog', 'wt-next': 'E5: the walkthrough dialog', 'wt-live': 'E5: the walkthrough dialog',
};
{
  const i23 = staticIds(V23), i24 = staticIds(V24);
  const gone = [...i23.keys()].filter(k => !i24.has(k));
  res.a.ids23 = i23.size; res.a.ids24 = i24.size;
  res.a.removed = gone.map(k => [k, REMOVED[k] || 'NOT EXPECTED']);
  res.a.unexpected = gone.filter(k => !REMOVED[k]);
  res.a.expectedButKept = Object.keys(REMOVED).filter(k => i24.has(k));
  res.a.dup24 = [...i24].filter(([, n]) => n > 1).map(([k]) => k);
  res.a.ok = !res.a.unexpected.length && !res.a.expectedButKept.length && !res.a.dup24.length;
}

// ------------------------------------------------------------------------------------------------ (b) lines
const count = lines => { const m = new Map(); for (const l of lines) m.set(l, (m.get(l) || 0) + 1); return m; };
const L19 = count(V19.split('\n')), L23 = V23.split('\n');
const ADDED_IDS = STATIC.filter(e => e.id).map(e => e.id);
const strip = l => {                // set aside the attributes this build adds to an opening tag
  let s = l.replace(/ data-nav-label="[^"]*"/g, '').replace(/ data-tip-key="[^"]*"/g, '');
  for (const id of ADDED_IDS) if (s.includes(' id="' + id + '"')) s = s.replace(' id="' + id + '"', '');
  return s;
};
const L24 = count(V24.split('\n').map(strip));
const CHANGED = [   // version 23 lines that version 24 changes on purpose (REBASE.md, "Deliberate changes to version 23")
  { id: 'R1', why: 'B1: the status box gets one status time and page version 24', test: l => l.includes('<h2 id="head-status-h">Status as of 15 Oct 2026 (page version 23)</h2>'),
    repl: ['<h2 id="head-status-h">Status as of 15 Oct 2026, 22:13 UTC (page version 24)</h2>'] },
  { id: 'R2', why: 'B2: Stage 2 version 7 was approved at 15:45 UTC; "waits for the owner\'s approval" was out of date', test: l => l.includes("plan version 7</i>) waits for the owner's approval"),
    repl: ["plan version 7</i>) has not started: the owner typed their approval at their own"] },
  { id: 'R3', why: 'B20: the weighting test\'s code, specification and exploratory results are in the shared repository', test: l => l === '          (<a href="#weighting-test">The weighting test</a>).</li>',
    repl: ['shared repository since 15 Oct at 17:18 UTC; its run records are not.'] },
  { id: 'R4', why: 'E1: the old sidebar\'s CSS, with version 22\'s "not in use now" rule; nav.css carries the same marking for the new Choices rows',
    test: l => /^\.sb-row\.sb-idle |^@media \(max-width: 1239px\) \{ \.sb-tab\.sb-tab-away|^@media \(pointer: coarse\) \{ \.sb-b \{ min-height: 40px; \} \.sb-x/.test(l),
    repl: ['.nv-row.nv-idle .nv-rl::after { content: " (not in use now)"; font-weight: 400; color: var(--muted); }', '.nv-row.nv-idle .nv-btns { opacity: .85; }'] },
  { id: 'R5', why: 'E1: version 21\'s script that moved the old Choices tab aside (its lines, its opening "(function () {" included) goes with the tab; nav.js keeps its intent: the phone\'s Choices button steps aside while it would cover the switch bar or the table selectors',
    test: l => l === '(function () {' || /Choices tab floats over the page's bottom left|getElementById\("sb-tab"\), seen = new Set|IntersectionObserver" in window\)\) return;$|seen\.add\(e\.target\) : seen\.delete|tab\.classList\.toggle\("sb-tab-away"|\["simtog", "lc-tctl"\]\.forEach/.test(l),
    repl: ['const OWN_SWITCHES = ["simtog", "lc-tctl"];', 'html.nv-pill-away .nv-pill:not(:focus) { visibility: hidden; }'] },
  { id: 'R6', why: 'E14: the switch bar\'s comment named the walkthrough pop-out and the old sidebar, both removed', test: l => l.includes('(the walkthrough pop-out moves the panel)') || l.includes('the choices sidebar or the task demo'),
    repl: ['   above, the panel\'s choices under "In this section" or the task demo\'s "Watch one episode" links.'] },
  { id: 'R7', why: 'N7b: the learned actors not on screen were drawn at 80% opacity (random data 2.45:1 light, 2.78:1 dark); now opaque and thin', test: l => l.includes('g.globalAlpha = on ? 1 : 0.8;'),
    repl: ['g.strokeStyle = tok(LQVAR[s.c]); g.lineWidth = on ? 2.8 : 1.5; g.globalAlpha = 1;'] },
  { id: 'R8', why: 'K13b: the learned actors\' cost readout becomes one keyed line per data quality', test: l => /const lp = LQ\.map|"At this state the cost so far is " \+ lp\.slice|lp\[lp\.length - 1\] \+ "\. All five learned from|^        "\. The cost so far is minus the return so far, so lower is better\."\];$/.test(l),
    repl: ['const keys = LQ.map(q => [typeof LQVAR === "object" && LQVAR[q] ? LQVAR[q] : "--muted", "line"]);'] },
  { id: 'R9', why: 'E8: the old sidebar\'s script (its rows of versions 21-22 included) is replaced by the navigation\'s, which carries the same rows and the "not in use now" marking',
    test: l => /a row that applies to one of the live panel's two modes only|const idle = \(r, f\) =>|^      idle\(row\(|strong_current: "Strong current \(large shift\)" \}, "data-value"\)\), simLearned\),|named\(\{ recording: "Recording", stall_aware: "Stall-aware", rising_thrust: "Rising-thrust", no_throttle: "No throttle" \}, "data-value"\)\), simLearned\),|"medium-replay": "Medium-replay", medium: "Medium", "medium-expert": "Medium-expert", expert: "Expert" \}, "data-tog-value"\)\), \(\) => !simLearned\(\)\),|named\(\{ gap: "80% hold removed", full: "All the data" \}, "data-tog-value"\)\), \(\) => !simLearned\(\)\),|named\(\{ training: "Training mix", strong: "Strong test mix" \}, "data-tog-value"\)\), \(\) => !simLearned\(\)\) \] \},|row\("Table", all\("#lc-tq button"\)|"medium-expert": "Medium-expert", expert: "Expert", all: "All five side by side" \}, "data-lc-tq"\)\),|const proxies = \[\], idleRows = \[\];|signature = sig; proxies\.length = 0; idleRows\.length = 0;|if \(r\.idle\) idleRows\.push\(\[rw, r\.idle\]\);|rw\.classList\.contains\("sb-idle"\)/.test(l),
    repl: ['row("Who drives", all(\'#simtog [data-tog-group="who"]\')', 'row("Table", all("#lc-tq button"), byAttr("data-lc-tq")', 'if (r.idle) idleRows.push([rw, r.idle]);', 'rw.classList.toggle("nv-idle", v)'] },
];
{
  const need = new Map();
  for (const l of L23) if ((L19.get(l) || 0) < 1 || false) need.set(l, (need.get(l) || 0) + 1);
  // a line version 19 has fewer times than version 23 counts too
  const c23 = count(L23);
  for (const [l, n] of c23) { const extra = n - (L19.get(l) || 0); if (extra > 0) need.set(l, n); }
  const missing = [], allowed = [];
  for (const [l, n] of need) {
    if ((L24.get(l) || 0) >= n) continue;
    const c = CHANGED.find(x => x.test(l));
    if (c) allowed.push([c.id, l.slice(0, 120)]); else missing.push(l.slice(0, 200));
  }
  const replMissing = CHANGED.flatMap(c => c.repl.filter(r => !V24.includes(r)).map(r => c.id + ': ' + r.slice(0, 120)));
  const unusedAllow = CHANGED.filter(c => !allowed.some(([id]) => id === c.id)).map(c => c.id);
  res.b = { linesAdded: need.size, missing, allowed, changedIds: CHANGED.map(c => [c.id, c.why]), replMissing, unusedAllow,
    ok: !missing.length && !replMissing.length && !unusedAllow.length };
}
console.log('(a)', JSON.stringify(res.a).slice(0, 600));
console.log('(b) added lines', res.b.linesAdded, 'missing', res.b.missing.length, 'allowed', res.b.allowed.length, 'replacement missing', res.b.replMissing.length, 'unused allowances', res.b.unusedAllow);

// ------------------------------------------------------------------------------------------------ (c) the browser
const WRAP = fs.readFileSync(SCR + '/live4/live_6eea.html').subarray(0, 531);
const page23 = OUT + '/v23_live.html';
fs.writeFileSync(page23, Buffer.concat([WRAP, Buffer.from(V23), Buffer.from('\n</body></html>')]));
const PAGES = { v23: 'file:///' + page23, v24: 'file:///' + W7 + '/build/page_v24_preview.html' };
const ev = (pg, e) => evaluate(pg, e);
const errs = pg => pg.logs.filter(l => l.level === 'error' || l.kind === 'exception' || l.kind === 'crash').map(l => (l.text || '').slice(0, 200));
// show an element: version 24 through its navigation (one section at a time), version 23 by scrolling
const show = async (pg, id) => { await ev(pg, `(() => { const e = document.getElementById(${JSON.stringify(id)}); if (!e) return false; if (window.nvGo) window.nvGo(${JSON.stringify(id)}); e.scrollIntoView({ block: 'center' }); return true; })()`); await sleep(700); };
async function hover(pg, sel, k = 0) {
  const r = await ev(pg, `(() => { const e = [...document.querySelectorAll(${JSON.stringify(sel)})][${k}]; if (!e) return null; e.scrollIntoView({ block: 'center', inline: 'center' }); const r = e.getBoundingClientRect(); return { x: r.left + r.width / 2, y: r.top + r.height / 2, l: r.left, r: r.right, t: r.top, b: r.bottom }; })()`);
  if (!r) return null;
  await pg.S('Input.dispatchMouseEvent', { type: 'mouseMoved', x: r.x - 30, y: r.y - 40 });
  await sleep(150);
  await pg.S('Input.dispatchMouseEvent', { type: 'mouseMoved', x: r.x, y: r.y });
  await sleep(1100);
  const tip = await ev(pg, `(() => { const t = document.getElementById('tt'); if (!t || t.hidden) return null; const r = t.getBoundingClientRect();
    return { text: t.textContent.replace(/\\s+/g, ' ').trim(), read: (t.querySelector('.tt-read') || {}).textContent || null, keys: [...t.querySelectorAll('.key')].map(k => k.style.getPropertyValue('--c')), l: r.left, r: r.right, t: r.top, b: r.bottom }; })()`);
  await pg.S('Input.dispatchMouseEvent', { type: 'mouseMoved', x: 2, y: 2 });
  await sleep(300);
  return { cell: r, tip };
}
const PROBE_SIM = `(() => { const $ = id => document.getElementById(id), sim = $('sim'), txt = id => ($(id) || {}).textContent || null;
  const cv = id => { const c = $(id); if (!c || !c.width) return null; const g = c.getContext('2d'), d = g.getImageData(0, 0, c.width, c.height).data; let n = 0; for (let i = 3; i < d.length; i += 40) if (d[i]) n++; return { w: c.clientWidth, h: c.clientHeight, ink: n }; };
  return { cls: sim.className, lq: sim.dataset.lq || null, lm: sim.dataset.lm || null, now: txt('simtog-now'), roT: txt('roT'), roW: txt('roW'), roJ: txt('roJ'), roP: txt('roP'), roG: txt('roG'),
    exHead: txt('exHead'), exSit: (txt('exSit') || '').slice(0, 160), exEp: (txt('exEp') || '').slice(0, 160),
    legendLearned: !$('legend-learned').hidden, legendFixed: !$('legend-fixed').hidden, noteLearned: !$('ro-note-learned').hidden, optsnote: !$('sim-optsnote').hidden,
    pressed: [...document.querySelectorAll('#simtog .simtog-b[aria-pressed="true"]')].map(b => b.dataset.togGroup + '=' + b.dataset.togValue).sort(),
    shownRows: [...document.querySelectorAll('#simtog [data-tog-mode]')].filter(n => !n.hidden).map(n => n.dataset.g),
    big: [...document.querySelectorAll('#sim .opt[aria-pressed="true"]')].map(b => b.dataset.group + '=' + b.dataset.value).sort(),
    channel: cv('channel'), cost: cv('cost') }; })()`;
const press = (g, v) => `(() => { const b = document.querySelector('#simtog [data-tog-group="${g}"][data-tog-value="${v}"]'); if (!b || b.disabled) return false; b.click(); return true; })()`;
const setT = t => `(() => { const s = document.getElementById('tSlider'); if (!s) return false; if (typeof setPlaying === 'function') setPlaying(false); s.value = ${t}; s.dispatchEvent(new Event('input', { bubbles: true })); return true; })()`;

async function features(cdp, ver, view) {
  const pg = await openPage(cdp, { width: view.w, height: view.h, mobile: view.w < 500, touch: false, dark: view.dark });
  await navigate(cdp, pg, PAGES[ver] + '#sim', { settle: 3500 });
  const r = { ver };
  // the floating Choices control steps aside while the switch bar is under it (version 21's tab; version 24's button)
  await show(pg, 'simtog');
  await ev(pg, `(() => { const s = document.getElementById('simtog'); scrollBy(0, s.getBoundingClientRect().top - (innerHeight - 200)); return 1; })()`); await sleep(700);
  r.away = await ev(pg, `(() => { const t = document.getElementById('sb-tab'), p = document.getElementById('nv-pill');
    if (t) return { kind: 'tab', away: innerWidth < 1240 && t.classList.contains('sb-tab-away') };
    return { kind: 'pill', away: !!p && document.documentElement.classList.contains('nv-pill-away') && getComputedStyle(p).visibility === 'hidden' }; })()`);
  // the switch bar, fixed mode
  await show(pg, 'simtog');
  r.simtogShown = await ev(pg, `(() => { const b = document.getElementById('simtog'); return !!b && !b.hidden && b.getClientRects().length > 0; })()`);
  for (const [g, v] of [['sit', 'strong_current'], ['world', 'no_stall'], ['ctrl', 'stall_aware']]) { await ev(pg, press(g, v)); await sleep(250); }
  await ev(pg, setT(40)); await sleep(300);
  r.fixed = await ev(pg, PROBE_SIM);
  for (const [g, v] of [['sit', 'recorded'], ['world', 'stall'], ['ctrl', 'recording']]) { await ev(pg, press(g, v)); await sleep(200); }
  // Who drives: the learned actor, two data qualities, both test mixes
  r.learned = {};
  await ev(pg, press('who', 'learned')); await sleep(500);
  for (const q of ['random', 'expert']) for (const m of ['training', 'strong']) {
    await ev(pg, press('lq', q)); await sleep(250); await ev(pg, press('lm', m)); await sleep(250);
    await ev(pg, setT(57)); await sleep(400);
    r.learned[q + '|' + m] = await ev(pg, PROBE_SIM);
  }
  // the cost readout while the learned actor drives (its keys in version 24)
  const ch = await hover(pg, '#cost');
  r.costTip = ch && ch.tip ? { lines: (ch.tip.text.match(/data: [−\d.,]+/g) || []).length, keys: ch.tip.keys } : null;   // read while the readout shows
  r.costKeys = await ev(pg, `(document.getElementById('cost').getAttribute('data-tip-keys') || '')`);
  // the Choices rows for the live panel, with the "not in use now" marking, while the learned actor drives
  r.choicesLearned = await choices(pg, ver);
  await ev(pg, press('who', 'fixed')); await sleep(500);
  r.choicesFixed = await choices(pg, ver);
  r.afterFixed = await ev(pg, PROBE_SIM);
  // Observation Space: the range bars and a cell's reading in its tooltip
  await show(pg, 'obs-gauge');
  r.obs = await ev(pg, `(() => { const t = document.querySelector('#observation-space table.obs-tbl'), k = document.getElementById('ob-key');
    return { on: !!t && t.classList.contains('ob-on'), bars: document.querySelectorAll('#observation-space .ob-bar').length, keyShown: !!k && !k.hidden && k.getClientRects().length > 0,
      reads: document.querySelectorAll('#observation-space td[data-tt-read]').length, guide: !!document.getElementById('obs-gauge') }; })()`);
  await show(pg, 'obs-table');
  const oh = await hover(pg, '#observation-space td[data-tt-read]', 3);
  r.obsTip = oh && oh.tip ? oh.tip.read : null;
  // the arm tables: a score cell's reading, and the tooltip beside the cell
  r.arms = {};
  for (const [arms, id] of [['stall', 'ats-table-stalling'], ['no_stall', 'ats-table-no-stall'], ['replication', 'ats-table-replication']]) {
    const sel = `table[data-arms="${arms}"] tr.main td.n`;
    await show(pg, ver === 'v24' ? id : 'ats-by-situation');
    const h = await hover(pg, sel, 1);
    const t = h && h.tip, c = h && h.cell;
    r.arms[arms] = t ? { read: (t.read || '').slice(0, 220), beside: t.l >= c.r - 1 || t.r <= c.l + 1, overlapsCell: !(t.l >= c.r - 1 || t.r <= c.l + 1 || t.t >= c.b - 1 || t.b <= c.t + 1) } : null;
  }
  r.howto = await ev(pg, `document.querySelectorAll('aside.tt-howto').length`);
  // the learning curves' table and its selectors
  await show(pg, ver === 'v24' ? 'lc-th' : 'lc-table');
  r.lc = await ev(pg, `(() => { const q = [...document.querySelectorAll('#lc-tq button')], t = [...document.querySelectorAll('#lc-tt button')];
    return { tq: q.map(b => b.dataset.lcTq + (b.getAttribute('aria-pressed') === 'true' ? '*' : '')), tt: t.length, rows: document.querySelectorAll('#lc-tbody tr').length, cap: (document.getElementById('lc-cap') || {}).textContent }; })()`);
  await ev(pg, `(() => { const b = document.querySelector('#lc-tq [data-lc-tq="all"]'); if (b) b.click(); return !!b; })()`); await sleep(500);
  r.lcAll = await ev(pg, `(() => { const b = document.getElementById('lc-t5box'); return { shown: !!b && !b.hidden && b.getClientRects().length > 0, rows: document.querySelectorAll('#lc-tbody5 tr').length, cols: document.querySelectorAll('#lc-h5 th').length, cap: (document.getElementById('lc-cap5') || {}).textContent, oneTable: !!document.getElementById('lc-table') && document.getElementById('lc-table').getClientRects().length > 0 }; })()`);
  await ev(pg, `(() => { const b = document.querySelectorAll('#lc-tt button')[0]; if (b) b.click(); return !!b; })()`); await sleep(400);
  r.lcAllT = await ev(pg, `(document.getElementById('lc-cap5') || {}).textContent`);
  await ev(pg, `(() => { const b = document.querySelector('#lc-tq [data-lc-tq="expert"]'); if (b) b.click(); return !!b; })()`); await sleep(400);
  r.lcBack = await ev(pg, `(() => { const b = document.getElementById('lc-t5box'); return { t5hidden: !!b && (b.hidden || !b.getClientRects().length), rows: document.querySelectorAll('#lc-tbody tr').length, pressed: [...document.querySelectorAll('#lc-t button[aria-pressed="true"]')].map(b => b.dataset.lc) }; })()`);
  // version 23's notes and the coverage table's column tooltips
  await show(pg, 'iw-plain-split');
  r.notes = await ev(pg, `['iw-plain-split', 'iw-plain-refuse'].map(id => { const e = document.getElementById(id); return e ? { shown: e.getClientRects().length > 0, text: e.textContent.replace(/\\s+/g, ' ').trim().slice(0, 120) } : null; })`);
  const hh = await hover(pg, 'table.p4-iwcov thead tr.grp th[data-sit="strong_current"]', 0);
  r.covHeadTip = hh && hh.tip ? hh.tip.text.slice(0, 240) : null;
  // the page: no sideways scroll, no console errors
  r.sideways = await ev(pg, `document.documentElement.scrollWidth - document.documentElement.clientWidth`);
  r.errors = errs(pg);
  await closePage(cdp, pg);
  return r;
}
async function choices(pg, ver) {
  if (ver === 'v24') {
    await ev(pg, `window.nvGo && window.nvGo('sim')`); await sleep(500);
    return ev(pg, `[...document.querySelectorAll('#nv-stops .nv-row')].map(r => ({ l: r.querySelector('.nv-rl').textContent, idle: r.classList.contains('nv-idle'), n: r.querySelectorAll('.nv-b').length, on: [...r.querySelectorAll('.nv-b[aria-pressed="true"]')].map(b => b.textContent.trim()) }))`);
  }
  // version 23: the sidebar's rows of "Watch one episode" (it builds them when open)
  await ev(pg, `(() => { const p = document.getElementById('sb-panel'); if (p && p.hidden) document.getElementById('sb-tab').click(); return 1; })()`); await sleep(500);
  const rows = await ev(pg, `(() => { const g = [...document.querySelectorAll('#sb-body .sb-grp')].find(s => /Watch one episode/.test(s.textContent)); return g ? [...g.querySelectorAll('.sb-row')].map(r => ({ l: r.querySelector('.sb-rl').textContent, idle: r.classList.contains('sb-idle'), n: r.querySelectorAll('.sb-b').length, on: [...r.querySelectorAll('.sb-b[aria-pressed="true"]')].map(b => b.textContent.trim()) })) : null; })()`);
  await ev(pg, `(() => { const x = document.getElementById('sb-x'); if (x && innerWidth < 1240) x.click(); return 1; })()`); await sleep(300);
  return rows;
}
async function lcRows(pg, ver) {
  if (ver === 'v24') { await ev(pg, `window.nvGo && window.nvGo('lc-th')`); await sleep(500); return ev(pg, `[...document.querySelectorAll('#nv-stops .nv-row .nv-rl')].map(x => x.textContent)`); }
  await ev(pg, `(() => { const p = document.getElementById('sb-panel'); if (p && p.hidden) document.getElementById('sb-tab').click(); return 1; })()`); await sleep(500);
  return ev(pg, `(() => { const g = [...document.querySelectorAll('#sb-body .sb-grp')].find(s => /Learning curves/.test(s.textContent)); return g ? [...g.querySelectorAll('.sb-rl')].map(x => x.textContent) : null; })()`);
}

const VIEWS = [{ name: 'm390_light', w: 390, h: 844, dark: false }, { name: 'm390_dark', w: 390, h: 844, dark: true },
  { name: 'd1280_light', w: 1280, h: 900, dark: false }, { name: 'd1280_dark', w: 1280, h: 900, dark: true }];
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const { proc, cdp } = await launch({ port: 13951, logName: 'edge_regression_v23.log' });
try {
  for (const view of VIEWS) {
    const r23 = await features(cdp, 'v23', view), r24 = await features(cdp, 'v24', view);
    // the learning curves' Choices rows (separate loads: the sidebar and the rail each show one panel's rows)
    for (const [ver, r] of [['v23', r23], ['v24', r24]]) {
      const pg = await openPage(cdp, { width: view.w, height: view.h, mobile: view.w < 500, dark: view.dark });
      await navigate(cdp, pg, PAGES[ver] + '#lc-panel', { settle: 3000 });
      r.lcChoices = await lcRows(pg, ver);
      await closePage(cdp, pg);
    }
    const strip24 = x => x && x.map(o => ({ l: o.l, idle: o.idle, n: o.n }));
    const chk = {
      noErrors: !r23.errors.length && !r24.errors.length,
      noSideways: r24.sideways <= 0,
      stepsAside: view.w >= 1000 || (r23.away.away && r24.away.away),
      simtog: r24.simtogShown && r23.simtogShown,
      fixed: same({ ...r24.fixed, channel: null, cost: null }, { ...r23.fixed, channel: null, cost: null }) && r24.fixed.channel.ink > 50 && r24.fixed.cost.ink > 50,
      learned: Object.keys(r23.learned).length === 4 && Object.keys(r23.learned).every(k => same({ ...r24.learned[k], channel: null, cost: null }, { ...r23.learned[k], channel: null, cost: null })
        && r24.learned[k].cls.includes('sim-learned') && r24.learned[k].channel.ink > 50 && r24.learned[k].cost.ink > 50 && r24.learned[k].legendLearned),
      learnedDiffers: new Set(Object.values(r24.learned).map(x => x.roJ + x.now)).size === 4,
      costTip: !!r24.costTip && r24.costTip.lines === 5 && r24.costTip.keys.join() === 'var(--lq-1),var(--lq-2),var(--lq-3),var(--lq-4),var(--lq-5)',
      afterFixed: same({ ...r24.afterFixed, channel: null, cost: null }, { ...r23.afterFixed, channel: null, cost: null }) && !r24.afterFixed.cls.includes('sim-learned'),
      choicesSim: same(strip24(r24.choicesLearned), strip24(r23.choicesLearned)) && same(strip24(r24.choicesFixed), strip24(r23.choicesFixed))
        && r24.choicesLearned.filter(x => x.idle).map(x => x.l).join() === 'Situation,World,Controller'
        && r24.choicesFixed.filter(x => x.idle).map(x => x.l).join() === "Data quality,The learner's data,Tested on",
      choicesLc: !!r23.lcChoices && !!r24.lcChoices && r23.lcChoices.every(l => r24.lcChoices.includes(l)) && r24.lcChoices.includes('Table'),
      obs: same(r24.obs, r23.obs) && r24.obs.on && r24.obs.bars > 40 && r24.obs.keyShown && !!r24.obsTip && r24.obsTip === r23.obsTip,
      arms: ['stall', 'no_stall', 'replication'].every(k => r24.arms[k] && r23.arms[k] && r24.arms[k].read === r23.arms[k].read && !!r24.arms[k].read
        && r24.arms[k].beside === r23.arms[k].beside && r24.arms[k].overlapsCell === r23.arms[k].overlapsCell
        && (view.w < 1000 || !r24.arms[k].overlapsCell)) && r24.howto === 3,   // with room, the tooltip never covers its cell (version 22: beside it when above or below would)
      lc: same(r24.lc, r23.lc) && r24.lc.tq.length === 6 && same(r24.lcAll, r23.lcAll) && r24.lcAll.shown && r24.lcAll.rows > 0 && r24.lcAllT === r23.lcAllT && same(r24.lcBack, r23.lcBack),
      notes: same(r24.notes, r23.notes) && r24.notes.every(n => n && n.shown) && r24.covHeadTip === r23.covHeadTip
        && (view.w < 600 || !!r24.covHeadTip),   // a phone shows the table stacked, without its group headers (both versions)
    };
    res.c[view.name] = { ok: Object.values(chk).every(Boolean), chk, v23: r23, v24: r24 };
    console.log(view.name, res.c[view.name].ok, JSON.stringify(chk));
  }
} finally { try { await cdp.send('Browser.close'); } catch (e) {} await sleep(400); try { proc.kill(); } catch (e) {} }

res.ok = res.a.ok && res.b.ok && Object.values(res.c).every(x => x.ok);
fs.writeFileSync(OUT + '/regression_v23.json', JSON.stringify(res, null, 1));
console.log('regression_v23 ok:', res.ok);
process.exit(res.ok ? 0 : 1);
