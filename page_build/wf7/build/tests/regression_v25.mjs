// Regression test of page version 26 against version 25 (test only; it reads the pages and writes only under
// wf7/build/tests/regression/). Made from regression_v23.mjs (the test of version 24 against version 23); three parts:
//  (a) every element id of version 25 exists in version 26, except the ones this build removes on purpose (REMOVED,
//      each with its reason);
//  (b) every line that versions 20-25 added (the lines of version 25 that version 19 does not have, counted with their
//      repeats) is in version 26 unchanged, once the attributes this build adds to an opening tag are set aside (an id
//      and a short label from the navigation's static edits, a color key from the color patch); a line that is not is
//      allowed only when it is one of the deliberate changes listed in REBASE.md (CHANGED, each with its reason and a
//      check that its replacement is there);
//  (c) in headless Edge at 390x844 and 1280x900, light and dark, the features of versions 20-25 work in version 26 as
//      in version 25: Who drives (the learned actor) for two data qualities and both test mixes; version 25's Learner
//      switch for all three learners (TD3+BC, BCA, BCA IW) on both mixes, through the panel and through the Choices,
//      with the same "Now playing", readouts, explanation (the Learner line included), legend tooltips and cost chart
//      label as version 25, the cost readout's numbers equal to version 25's with a color key per line, and with all
//      the data BCA and BCA IW disabled in both; version 24's axis titles (the canvas titles drawn on the cost chart and
//      the gauges, the SVG charts' titles inside their charts, the same number of titles as version 25, each one of the
//      page's own wordings); the switch bar, the Observation Space range bars and their tooltips, the arm tables'
//      tooltips beside their cells, the learning curves' table and its selectors, version 23's notes and column
//      tooltips, and the Choices rows (with "not in use now"); no console errors, no sideways page scroll.
// Usage: node regression_v25.mjs   -> regression/regression_v25.json, exit 0 when everything passes
import fs from 'node:fs';
import { launch, openPage, navigate, evaluate, closePage, sleep } from '../../nav/tools/cdp.mjs';

const SCR = 'C:/Users/David/AppData/Local/Temp/claude/C--Users-David-dev-BCA/3bdab4f0-86ee-439c-a05a-34828d6b0a71/scratchpad';
const W7 = SCR + '/wf7';
const OUT = W7 + '/build/tests/regression';
fs.mkdirSync(OUT, { recursive: true });
const read = p => fs.readFileSync(p, 'utf8');
const V19 = read(SCR + '/live3/core_9369.html'), V25 = read(SCR + '/live5/core_4933.html'), V26 = read(W7 + '/build/core_v26.html');
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
  const i25 = staticIds(V25), i26 = staticIds(V26);
  const gone = [...i25.keys()].filter(k => !i26.has(k));
  res.a.ids25 = i25.size; res.a.ids26 = i26.size;
  res.a.removed = gone.map(k => [k, REMOVED[k] || 'NOT EXPECTED']);
  res.a.unexpected = gone.filter(k => !REMOVED[k]);
  res.a.expectedButKept = Object.keys(REMOVED).filter(k => i26.has(k));
  res.a.dup26 = [...i26].filter(([, n]) => n > 1).map(([k]) => k);
  res.a.ok = !res.a.unexpected.length && !res.a.expectedButKept.length && !res.a.dup26.length;
}

// ------------------------------------------------------------------------------------------------ (b) lines
const count = lines => { const m = new Map(); for (const l of lines) m.set(l, (m.get(l) || 0) + 1); return m; };
const L19 = count(V19.split('\n'));
const ADDED_IDS = STATIC.filter(e => e.id).map(e => e.id);
const strip = l => {                // set aside the attributes this build adds to an opening tag
  let s = l.replace(/ data-nav-label="[^"]*"/g, '').replace(/ data-tip-key="[^"]*"/g, '');
  for (const id of ADDED_IDS) if (s.includes(' id="' + id + '"')) s = s.replace(' id="' + id + '"', '');
  return s;
};
const L26 = count(V26.split('\n').map(strip));
const CHANGED = [   // version 25 lines that version 26 changes on purpose (REBASE.md, "Deliberate changes to version 25")
  { id: 'R1', why: 'B1: the status box gets one status time and page version 26', test: l => l.includes('<h2 id="head-status-h">Status as of 16 Oct 2026 (page version 25)</h2>'),
    repl: ['<h2 id="head-status-h">Status as of 16 Oct 2026, 12:39 UTC (page version 26)</h2>'] },
  { id: 'R2', why: 'B2: Stage 2 version 7 was approved at 15:45 UTC on 15 Oct and the approval checked and recorded at 02:27 UTC on 16 Oct; "waits for the owner\'s approval" was out of date', test: l => l.includes("plan version 7</i>) waits for the owner's approval"),
    repl: ["plan version 7</i>) is cleared to start: the owner typed their approval at their own", 'and it was checked and recorded on 16 Oct at 02:27 UTC; no result of the run is in the repository yet'] },
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
  { id: 'R8', why: 'K13b: the learned actors\' cost readout (version 25\'s, which names the learner and the cloud rerun) becomes one keyed line per data quality, the learner named in its last line',
    test: l => /const lp = LQ\.map|"At this state the cost so far is " \+ lp\.slice|lp\[lp\.length - 1\] \+ "\. All five learned from|^        "\. The cost so far is minus the return so far, so lower is better\."\];$/.test(l),
    repl: ['const keys = LQ.map(q => [typeof LQVAR === "object" && LQVAR[q] ? LQVAR[q] : "--muted", "line"]);', 'lines.push("All five " + (nm ? nm + "actors " : "") + "learned from "'] },
  { id: 'R9', why: 'E8: the old sidebar\'s script (its rows of versions 21, 22 and 25 included: Who drives with "Learned actor", the learned rows, version 25\'s "Learner", Table) is replaced by the navigation\'s, which carries the same rows and the "not in use now" marking',
    test: l => /a row that applies to one of the live panel's two modes only|const idle = \(r, f\) =>|^      idle\(row\(|strong_current: "Strong current \(large shift\)" \}, "data-value"\)\), simLearned\),|named\(\{ recording: "Recording", stall_aware: "Stall-aware", rising_thrust: "Rising-thrust", no_throttle: "No throttle" \}, "data-value"\)\), simLearned\),|"medium-replay": "Medium-replay", medium: "Medium", "medium-expert": "Medium-expert", expert: "Expert" \}, "data-tog-value"\)\), \(\) => !simLearned\(\)\),|named\(\{ gap: "80% hold removed", full: "All the data" \}, "data-tog-value"\)\), \(\) => !simLearned\(\)\),|named\(\{ training: "Training mix", strong: "Strong test mix" \}, "data-tog-value"\)\), \(\) => !simLearned\(\)\) \] \},|named\(\{ td3bc: "TD3\+BC", bca: "BCA", bcaiw: "BCA IW" \}, "data-tog-value"\)\), \(\) => !simLearned\(\)\),|row\("Table", all\("#lc-tq button"\)|"medium-expert": "Medium-expert", expert: "Expert", all: "All five side by side" \}, "data-lc-tq"\)\),|const proxies = \[\], idleRows = \[\];|signature = sig; proxies\.length = 0; idleRows\.length = 0;|if \(r\.idle\) idleRows\.push\(\[rw, r\.idle\]\);|rw\.classList\.contains\("sb-idle"\)/.test(l),
    repl: ['row("Who drives", all(\'#simtog [data-tog-group="who"]\')', 'named({ fixed: "Fixed controllers", learned: "Learned actor" }, "data-tog-value")),',
      'row("Learner", all(\'#simtog [data-tog-group="lr"]\'), byAttr("data-tog-value"),', 'named({ td3bc: "TD3+BC", bca: "BCA", bcaiw: "BCA IW" }, "data-tog-value"), { idle: simFixed }),',
      'row("Table", all("#lc-tq button"), byAttr("data-lc-tq")', 'if (r.idle) idleRows.push([rw, r.idle]);', 'rw.classList.toggle("nv-idle", v)'] },
];
{
  const need = new Map();
  const c25 = count(V25.split('\n'));
  for (const [l, n] of c25) { const extra = n - (L19.get(l) || 0); if (extra > 0) need.set(l, n); }
  const missing = [], allowed = [];
  for (const [l, n] of need) {
    if ((L26.get(l) || 0) >= n) continue;
    const c = CHANGED.find(x => x.test(l));
    if (c) allowed.push([c.id, l.slice(0, 120)]); else missing.push(l.slice(0, 200));
  }
  const replMissing = CHANGED.flatMap(c => c.repl.filter(r => !V26.includes(r)).map(r => c.id + ': ' + r.slice(0, 120)));
  const unusedAllow = CHANGED.filter(c => !allowed.some(([id]) => id === c.id)).map(c => c.id);
  res.b = { linesAdded: need.size, missing, allowed, changedIds: CHANGED.map(c => [c.id, c.why]), replMissing, unusedAllow,
    ok: !missing.length && !replMissing.length && !unusedAllow.length };
}
console.log('(a)', JSON.stringify(res.a).slice(0, 600));
console.log('(b) added lines', res.b.linesAdded, 'missing', res.b.missing.length, 'allowed', res.b.allowed.length, 'replacement missing', res.b.replMissing.length, 'unused allowances', res.b.unusedAllow);
if (res.b.missing.length) console.log('(b) missing:\n  ' + res.b.missing.slice(0, 40).join('\n  '));
if (process.env.ONLY_AB) { fs.writeFileSync(OUT + '/regression_v25_ab.json', JSON.stringify(res, null, 1)); process.exit(res.a.ok && res.b.ok ? 0 : 1); }

// ------------------------------------------------------------------------------------------------ (c) the browser
const WRAP = fs.readFileSync(SCR + '/live5/live_4933.html').subarray(0, 531);
const page25 = OUT + '/v25_live.html';
fs.writeFileSync(page25, Buffer.concat([WRAP, Buffer.from(V25), Buffer.from('\n</body></html>')]));
const PAGES = { v25: 'file:///' + page25, v26: 'file:///' + W7 + '/build/page_v26_preview.html' };
// every text drawn on a canvas, by canvas id (version 24's canvas axis titles), recorded from the page's first script on
const FILLTEXT = `(() => { const o = CanvasRenderingContext2D.prototype.fillText; window.__ft = new Set();
  CanvasRenderingContext2D.prototype.fillText = function (t, ...a) { try { window.__ft.add((this.canvas.id || '?') + '|' + t); } catch (e) {} return o.call(this, t, ...a); }; })()`;
const ev = (pg, e) => evaluate(pg, e);
const errs = pg => pg.logs.filter(l => l.level === 'error' || l.kind === 'exception' || l.kind === 'crash').map(l => (l.text || '').slice(0, 200));
// show an element: version 26 through its navigation (one section at a time), version 25 by scrolling
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
// hover the cost chart at state t (its plot's geometry: left margin 46, right margin 170, or on a narrow chart 112 while
// the learned actor drives and 70 otherwise), so that both versions read the same state though their columns differ
async function hoverCost(pg, t) {
  const p = await ev(pg, `(() => { const c = document.getElementById('cost'); if (!c) return null; c.scrollIntoView({ block: 'center' });
    const b = c.getBoundingClientRect(), w = c.clientWidth, narrow = w < 560, padL = 46,
      padR = narrow ? (typeof learnedEp === 'function' && learnedEp() ? 112 : 70) : 170;
    return { x: b.left + padL + ${t} / 99 * (w - padL - padR), y: b.top + b.height / 2 }; })()`);
  if (!p) return null;
  await sleep(300);
  await pg.S('Input.dispatchMouseEvent', { type: 'mouseMoved', x: p.x - 30, y: p.y - 40 });
  await sleep(150);
  await pg.S('Input.dispatchMouseEvent', { type: 'mouseMoved', x: p.x, y: p.y });
  await sleep(1100);
  const tip = await ev(pg, `(() => { const t = document.getElementById('tt'); if (!t || t.hidden) return null;
    return { text: t.textContent.replace(/\\s+/g, ' ').trim(), keys: [...t.querySelectorAll('.key')].map(k => k.style.getPropertyValue('--c')) }; })()`);
  await pg.S('Input.dispatchMouseEvent', { type: 'mouseMoved', x: 2, y: 2 });
  await sleep(300);
  return { tip };
}
const PROBE_SIM = `(() => { const $ = id => document.getElementById(id), sim = $('sim'), txt = id => ($(id) || {}).textContent || null;
  const cv = id => { const c = $(id); if (!c || !c.width) return null; const g = c.getContext('2d'), d = g.getImageData(0, 0, c.width, c.height).data; let n = 0; for (let i = 3; i < d.length; i += 40) if (d[i]) n++; return { w: c.clientWidth, h: c.clientHeight, ink: n }; };
  return { cls: sim.className, lq: sim.dataset.lq || null, lm: sim.dataset.lm || null, now: txt('simtog-now'), roT: txt('roT'), roW: txt('roW'), roJ: txt('roJ'), roP: txt('roP'), roG: txt('roG'),
    exHead: txt('exHead'), exSit: (txt('exSit') || '').slice(0, 160), exEp: (txt('exEp') || '').slice(0, 260), exBank: (txt('exBank') || '').slice(0, 400), exLrn: txt('exLrn'),
    exLrnShown: !!$('exLrn') && !$('exLrn').hidden, lrQ: txt('simtog-lr-q'),
    lrBtns: [...document.querySelectorAll('#simtog [data-tog-group="lr"]')].map(b => b.dataset.togValue + (b.disabled ? ':off' : '') + (b.getAttribute('aria-pressed') === 'true' ? '*' : '') + '|' + b.getAttribute('data-tip-title')),
    costLabel: $('cost').getAttribute('aria-label'), costTipTitle: $('cost').getAttribute('data-tip-title'),
    legendTips: [...document.querySelectorAll('#legend-learned > span')].map(s => s.getAttribute('data-tip')),
    legendLearned: !$('legend-learned').hidden, legendFixed: !$('legend-fixed').hidden, noteLearned: !$('ro-note-learned').hidden, optsnote: !$('sim-optsnote').hidden,
    pressed: [...document.querySelectorAll('#simtog .simtog-b[aria-pressed="true"]')].map(b => b.dataset.togGroup + '=' + b.dataset.togValue).sort(),
    shownRows: [...document.querySelectorAll('#simtog [data-tog-mode]')].filter(n => !n.hidden).map(n => n.dataset.g),
    big: [...document.querySelectorAll('#sim .opt[aria-pressed="true"]')].map(b => b.dataset.group + '=' + b.dataset.value).sort(),
    channel: cv('channel'), cost: cv('cost') }; })()`;
const press = (g, v) => `(() => { const b = document.querySelector('#simtog [data-tog-group="${g}"][data-tog-value="${v}"]'); if (!b || b.disabled) return false; b.click(); return true; })()`;
const setT = t => `(() => { const s = document.getElementById('tSlider'); if (!s) return false; if (typeof setPlaying === 'function') setPlaying(false); s.value = ${t}; s.dispatchEvent(new Event('input', { bubbles: true })); return true; })()`;
const noCanvas = x => x && { ...x, channel: null, cost: null };
// the cost readout's five values: version 25 states them in one sentence ("380.30 for the BCA actor learned from random
// data, ..."), version 26 on one keyed line each ("Random data: 380.30"); both in the order random .. expert
const costNums = t => {
  if (!t) return null;
  const v26 = [...t.matchAll(/(?:Random|Medium-replay|Medium|Medium-expert|Expert) data: ([−\d.,]+)/g)].map(m => m[1]);
  if (v26.length) return v26;
  return [...t.matchAll(/([−\d.,]+) for the (?:[A-Z][A-Z +]* )?actor learned from/g)].map(m => m[1]);
};

async function features(cdp, ver, view) {
  const pg = await openPage(cdp, { width: view.w, height: view.h, mobile: view.w < 500, touch: false, dark: view.dark, initScripts: [FILLTEXT] });
  await navigate(cdp, pg, PAGES[ver] + '#sim', { settle: 3500 });
  const r = { ver };
  // the floating Choices control steps aside while the switch bar is under it (version 25's tab; version 26's button)
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
  // Who drives: the learned actor (TD3+BC), two data qualities, both test mixes
  r.learned = {};
  await ev(pg, press('who', 'learned')); await sleep(500);
  for (const q of ['random', 'expert']) for (const m of ['training', 'strong']) {
    await ev(pg, press('lq', q)); await sleep(250); await ev(pg, press('lm', m)); await sleep(250);
    await ev(pg, setT(57)); await sleep(400);
    r.learned[q + '|' + m] = await ev(pg, PROBE_SIM);
  }
  // version 25's Learner switch: all three learners, the 80% hold removed (the cloud rerun), both mixes; the cost
  // readout's numbers at one state
  r.learners = {}; r.learnerCost = {};
  for (const lr of ['td3bc', 'bca', 'bcaiw']) {
    await ev(pg, press('lr', lr)); await sleep(400);
    for (const [q, m] of [['random', 'strong'], ['medium', 'training']]) {
      await ev(pg, press('lq', q)); await sleep(250); await ev(pg, press('lm', m)); await sleep(250);
      await ev(pg, setT(63)); await sleep(400);
      r.learners[lr + '|' + q + '|' + m] = await ev(pg, PROBE_SIM);
    }
    const ch = await hoverCost(pg, 50);
    r.learnerCost[lr] = ch && ch.tip ? { nums: costNums(ch.tip.text), keys: ch.tip.keys, text: ch.tip.text.slice(0, 600) } : null;
  }
  // with all the data only TD3+BC exists: BCA and BCA IW disabled, the learner back to TD3+BC
  await ev(pg, press('lr', 'bca')); await sleep(300);
  await ev(pg, press('ld', 'full')); await sleep(500);
  r.allData = await ev(pg, PROBE_SIM);
  r.allDataChoices = await choices(pg, ver);
  await ev(pg, press('ld', 'gap')); await sleep(400);
  await ev(pg, press('lr', 'td3bc')); await sleep(300); await ev(pg, press('lq', 'expert')); await sleep(250); await ev(pg, press('lm', 'strong')); await sleep(250);
  // the Learner switch through the Choices (version 26) or the old sidebar (version 25): press BCA IW there
  r.viaChoices = await pressChoice(pg, ver, 'Learner', 'BCA IW');
  await sleep(500);
  r.afterChoice = await ev(pg, PROBE_SIM);
  await ev(pg, press('lr', 'td3bc')); await sleep(300);
  // the cost readout while the learned actor (TD3+BC, expert, strong test mix) drives (its keys in version 26)
  await ev(pg, setT(57)); await sleep(300);
  const ch = await hoverCost(pg, 50);
  r.costTip = ch && ch.tip ? { lines: (ch.tip.text.match(/data: [−\d.,]+/g) || []).length, keys: ch.tip.keys, nums: costNums(ch.tip.text) } : null;   // read while the readout shows
  // the Choices rows for the live panel, with the "not in use now" marking, while the learned actor drives
  r.choicesLearned = await choices(pg, ver);
  await ev(pg, press('who', 'fixed')); await sleep(500);
  r.choicesFixed = await choices(pg, ver);
  r.afterFixed = await ev(pg, PROBE_SIM);
  // version 24's canvas axis titles: what was drawn on the cost chart and the drawing (gauges) so far
  r.canvasText = await ev(pg, `[...(window.__ft || [])].filter(s => /^(cost|channel)\\|/.test(s)).sort()`);
  // version 24's SVG axis titles: for each chart, its titles' texts and whether each lies inside the chart
  r.svgTitles = {};
  for (const [sec, sel] of [['as-widget', '#as-widget svg.aw-strip'], ['stall-chart', '#sc-svg'], ['qv-panel', '#qv-panel svg.qv-svg'],
    ['iw-svg-dens', '#iw-svg-dens'], ['iw-svg-w', '#iw-svg-w'], ['iw-svg-thrust', '#iw-svg-thrust'], ['iw-svg-bars', '#iw-svg-bars'], ['iw-svg-cov', '#iw-svg-cov'],
    ['lc-panel', '#lc-svg1, #lc-svg2'], ['lc-grid', '#lc-grid svg'], ['beyond-chart', '#beyond-chart .dr-head']]) {
    await show(pg, sec); await sleep(500);
    await ev(pg, `(() => { const e = document.querySelector(${JSON.stringify(sel)}); if (e) e.scrollIntoView({ block: 'center' }); window.dispatchEvent(new Event('resize')); return 1; })()`); await sleep(900);
    r.svgTitles[sec] = await ev(pg, `[...document.querySelectorAll(${JSON.stringify(sel)})].slice(0, 6).map(svg => {
      const box = svg.getBoundingClientRect();
      const t = [...svg.querySelectorAll('text.axlab, text.ix-ttl, .dr-name, .dr-val')].filter(x => x.textContent.trim());
      // inside: the title's text box (its line box, a little larger than its letters) within 3 px of the chart's edges;
      // version 25's own titles reach up to 2 px past an edge (a baseline 4 px above the bottom, a rotated title at x 13)
      return { id: svg.id || null, shown: box.width > 0, n: t.length, titles: t.map(x => { const b = x.getBoundingClientRect();
        const m = [b.left - box.left, box.right - b.right, b.top - box.top, box.bottom - b.bottom].map(v => Math.round(v * 10) / 10);
        return { text: x.textContent.trim(), margins: m, inside: Math.min(...m) >= -3 }; }),
        label: (svg.getAttribute('aria-label') || '').replace(/^.*?(Horizontal axis)/, '$1').slice(0, 300) }; })`);
  }
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
    await show(pg, ver === 'v26' ? id : 'ats-by-situation');
    const h = await hover(pg, sel, 1);
    const t = h && h.tip, c = h && h.cell;
    r.arms[arms] = t ? { read: (t.read || '').slice(0, 220), beside: t.l >= c.r - 1 || t.r <= c.l + 1, overlapsCell: !(t.l >= c.r - 1 || t.r <= c.l + 1 || t.t >= c.b - 1 || t.b <= c.t + 1) } : null;
  }
  r.howto = await ev(pg, `document.querySelectorAll('aside.tt-howto').length`);
  // the learning curves' table and its selectors
  await show(pg, ver === 'v26' ? 'lc-th' : 'lc-table');
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
const rowsExpr = (ver, group) => ver === 'v26'
  ? `[...document.querySelectorAll('#nv-stops .nv-row')].map(r => ({ l: r.querySelector('.nv-rl').textContent, idle: r.classList.contains('nv-idle'), n: r.querySelectorAll('.nv-b').length, on: [...r.querySelectorAll('.nv-b[aria-pressed="true"]')].map(b => b.textContent.trim()), off: [...r.querySelectorAll('.nv-b')].filter(b => b.disabled).map(b => b.textContent.trim()) }))`
  : `(() => { const g = [...document.querySelectorAll('#sb-body .sb-grp')].find(s => ${group}.test(s.textContent)); return g ? [...g.querySelectorAll('.sb-row')].map(r => ({ l: r.querySelector('.sb-rl').textContent, idle: r.classList.contains('sb-idle'), n: r.querySelectorAll('.sb-b').length, on: [...r.querySelectorAll('.sb-b[aria-pressed="true"]')].map(b => b.textContent.trim()), off: [...r.querySelectorAll('.sb-b')].filter(b => b.disabled).map(b => b.textContent.trim()) })) : null; })()`;
const openSidebar = `(() => { const p = document.getElementById('sb-panel'); if (p && p.hidden) document.getElementById('sb-tab').click(); return 1; })()`;
const closeSidebar = `(() => { const x = document.getElementById('sb-x'); if (x && innerWidth < 1240) x.click(); return 1; })()`;
async function choices(pg, ver) {
  if (ver === 'v26') { await ev(pg, `window.nvGo && window.nvGo('sim')`); await sleep(500); return ev(pg, rowsExpr(ver)); }
  // version 25: the sidebar's rows of "Watch one episode" (it builds them when open)
  await ev(pg, openSidebar); await sleep(500);
  const rows = await ev(pg, rowsExpr(ver, '/Watch one episode/'));
  await ev(pg, closeSidebar); await sleep(300);
  return rows;
}
async function pressChoice(pg, ver, label, name) {
  const find = ver === 'v26' ? `#nv-stops .nv-row` : `#sb-body .sb-row`, rl = ver === 'v26' ? '.nv-rl' : '.sb-rl', bb = ver === 'v26' ? '.nv-b' : '.sb-b';
  if (ver === 'v26') { await ev(pg, `window.nvGo && window.nvGo('sim')`); await sleep(500); } else { await ev(pg, openSidebar); await sleep(500); }
  const out = await ev(pg, `(() => { const r = [...document.querySelectorAll(${JSON.stringify(find)})].find(r => r.querySelector(${JSON.stringify(rl)}).textContent === ${JSON.stringify(label)}); if (!r) return 'no row';
    const b = [...r.querySelectorAll(${JSON.stringify(bb)})].find(b => b.textContent.trim() === ${JSON.stringify(name)}); if (!b) return 'no button'; if (b.disabled) return 'disabled'; b.click(); return 'pressed'; })()`);
  if (ver !== 'v26') { await sleep(300); await ev(pg, closeSidebar); }
  return out;
}
async function lcRows(pg, ver) {
  if (ver === 'v26') { await ev(pg, `window.nvGo && window.nvGo('lc-th')`); await sleep(500); return ev(pg, `[...document.querySelectorAll('#nv-stops .nv-row .nv-rl')].map(x => x.textContent)`); }
  await ev(pg, openSidebar); await sleep(500);
  return ev(pg, `(() => { const g = [...document.querySelectorAll('#sb-body .sb-grp')].find(s => /Learning curves/.test(s.textContent)); return g ? [...g.querySelectorAll('.sb-rl')].map(x => x.textContent) : null; })()`);
}

const VIEWS = [{ name: 'm390_light', w: 390, h: 844, dark: false }, { name: 'm390_dark', w: 390, h: 844, dark: true },
  { name: 'd1280_light', w: 1280, h: 900, dark: false }, { name: 'd1280_dark', w: 1280, h: 900, dark: true }];
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const KEYS5 = 'var(--lq-1),var(--lq-2),var(--lq-3),var(--lq-4),var(--lq-5)';
// a title of version 26 is version 25's, or another of the page's own wordings for it (each chart takes the longest
// wording that fits its width, and the reading column is narrower beside the navigation)
const ownWording = t => t && (V26.includes(t) || V26.includes(t.split(' (')[0]) || V26.includes(t.replace(/\(up to [\d,]+\)/, '(up to ')));
const { proc, cdp } = await launch({ port: 13951, logName: 'edge_regression_v25.log' });
try {
  for (const view of VIEWS) {
    const r25 = await features(cdp, 'v25', view), r26 = await features(cdp, 'v26', view);
    // the learning curves' Choices rows (separate loads: the sidebar and the rail each show one panel's rows)
    for (const [ver, r] of [['v25', r25], ['v26', r26]]) {
      const pg = await openPage(cdp, { width: view.w, height: view.h, mobile: view.w < 500, dark: view.dark });
      await navigate(cdp, pg, PAGES[ver] + '#lc-panel', { settle: 3000 });
      r.lcChoices = await lcRows(pg, ver);
      await closePage(cdp, pg);
    }
    const stripRows = x => x && x.map(o => ({ l: o.l, idle: o.idle, n: o.n, off: o.off }));
    const titlesOk = Object.keys(r25.svgTitles).every(k => {
      const a = r25.svgTitles[k], b = r26.svgTitles[k];
      return a.length && b.length === a.length && b.every((c, i) => c.shown && c.n === a[i].n && c.n > 0 && c.label === a[i].label
        // inside, or no further outside than version 25's own title (then version 24's to answer for, and reported)
        && c.titles.every((t, j) => (t.inside || Math.min(...t.margins) >= Math.min(...a[i].titles[j].margins) - 0.5)
          && (t.text === a[i].titles[j].text || ownWording(t.text))));
    });
    const canvasTitles = ['cost|State t of the episode (0 to 99)', 'cost|State t (0 to 99)'].some(x => r26.canvasText.includes(x))
      && r26.canvasText.some(x => /^channel\|Throttle (\(bar\) )?and push/.test(x))
      && ['BCA', 'BCA IW'].every(n => r26.canvasText.some(x => x.startsWith('cost|Cost so far') && x.includes(n + ' actors')))
      && r25.canvasText.filter(x => /^cost\|State t|^channel\|Throttle/.test(x)).length > 0;
    const chk = {
      noErrors: !r25.errors.length && !r26.errors.length,
      noSideways: r26.sideways <= 0,
      stepsAside: view.w >= 1000 || (r25.away.away && r26.away.away),
      simtog: r26.simtogShown && r25.simtogShown,
      fixed: same(noCanvas(r26.fixed), noCanvas(r25.fixed)) && r26.fixed.channel.ink > 50 && r26.fixed.cost.ink > 50,
      learned: Object.keys(r25.learned).length === 4 && Object.keys(r25.learned).every(k => same(noCanvas(r26.learned[k]), noCanvas(r25.learned[k]))
        && r26.learned[k].cls.includes('sim-learned') && r26.learned[k].channel.ink > 50 && r26.learned[k].cost.ink > 50 && r26.learned[k].legendLearned),
      learnedDiffers: new Set(Object.values(r26.learned).map(x => x.roJ + x.now)).size === 4,
      learners: Object.keys(r25.learners).length === 6 && Object.keys(r25.learners).every(k => same(noCanvas(r26.learners[k]), noCanvas(r25.learners[k]))
        && r26.learners[k].exLrnShown && r26.learners[k].channel.ink > 50 && r26.learners[k].cost.ink > 50)
        && new Set(Object.values(r26.learners).map(x => x.roJ + x.now)).size === 6
        && ['bca', 'bcaiw'].every(l => /^Learned actor: BCA( IW)? trained on/.test(r26.learners[l + '|random|strong'].now)),
      learnerCost: ['td3bc', 'bca', 'bcaiw'].every(l => r26.learnerCost[l] && r25.learnerCost[l] && r26.learnerCost[l].nums.length === 5
        && same(r26.learnerCost[l].nums, r25.learnerCost[l].nums) && r26.learnerCost[l].keys.join() === KEYS5)
        && /All five BCA actors learned from/.test(r26.learnerCost.bca.text) && /All five BCA IW actors learned from/.test(r26.learnerCost.bcaiw.text),
      allData: same(noCanvas(r26.allData), noCanvas(r25.allData)) && r26.allData.lrBtns.filter(b => /:off/.test(b)).length === 2 && /^td3bc\*/.test(r26.allData.lrBtns[0])
        && same(stripRows(r26.allDataChoices).find(x => x.l === 'Learner'), stripRows(r25.allDataChoices).find(x => x.l === 'Learner'))
        && same((r26.allDataChoices.find(x => x.l === 'Learner') || {}).off, ['BCA', 'BCA IW']),
      viaChoices: r26.viaChoices === 'pressed' && r25.viaChoices === 'pressed' && same(noCanvas(r26.afterChoice), noCanvas(r25.afterChoice))
        && r26.afterChoice.pressed.includes('lr=bcaiw'),
      costTip: !!r26.costTip && r26.costTip.lines === 5 && r26.costTip.keys.join() === KEYS5 && !!r25.costTip && same(r26.costTip.nums, r25.costTip.nums),
      afterFixed: same(noCanvas(r26.afterFixed), noCanvas(r25.afterFixed)) && !r26.afterFixed.cls.includes('sim-learned'),
      choicesSim: same(stripRows(r26.choicesLearned), stripRows(r25.choicesLearned)) && same(stripRows(r26.choicesFixed), stripRows(r25.choicesFixed))
        && r26.choicesLearned.filter(x => x.idle).map(x => x.l).join() === 'Situation,World,Controller'
        && r26.choicesFixed.filter(x => x.idle).map(x => x.l).join() === "Learner,Data quality,The learner's data,Tested on"
        && r26.choicesLearned.some(x => x.l === 'Learner' && x.n === 3) && r26.choicesLearned[0].l === 'Who drives',
      choicesLc: !!r25.lcChoices && !!r26.lcChoices && r25.lcChoices.every(l => r26.lcChoices.includes(l)) && r26.lcChoices.includes('Table'),
      canvasTitles,
      svgTitles: titlesOk,
      obs: same(r26.obs, r25.obs) && r26.obs.on && r26.obs.bars > 40 && r26.obs.keyShown && !!r26.obsTip && r26.obsTip === r25.obsTip,
      arms: ['stall', 'no_stall', 'replication'].every(k => r26.arms[k] && r25.arms[k] && r26.arms[k].read === r25.arms[k].read && !!r26.arms[k].read
        && (view.w < 1000 || !r26.arms[k].overlapsCell) && (view.w >= 1000 || (r26.arms[k].beside === r25.arms[k].beside && r26.arms[k].overlapsCell === r25.arms[k].overlapsCell))) && r26.howto === 3,
      lc: same(r26.lc, r25.lc) && r26.lc.tq.length === 6 && same(r26.lcAll, r25.lcAll) && r26.lcAll.shown && r26.lcAll.rows > 0 && r26.lcAllT === r25.lcAllT && same(r26.lcBack, r25.lcBack),
      notes: same(r26.notes, r25.notes) && r26.notes.every(n => n && n.shown) && r26.covHeadTip === r25.covHeadTip
        && (view.w < 600 || !!r26.covHeadTip),   // a phone shows the table stacked, without its group headers (both versions)
    };
    res.c[view.name] = { ok: Object.values(chk).every(Boolean), chk, v25: r25, v26: r26 };
    console.log(view.name, res.c[view.name].ok, JSON.stringify(chk));
  }
} finally { try { await cdp.send('Browser.close'); } catch (e) {} await sleep(400); try { proc.kill(); } catch (e) {} }

res.ok = res.a.ok && res.b.ok && Object.values(res.c).every(x => x.ok);
fs.writeFileSync(OUT + '/regression_v25.json', JSON.stringify(res, null, 1));
console.log('regression_v25 ok:', res.ok);
process.exit(res.ok ? 0 : 1);
