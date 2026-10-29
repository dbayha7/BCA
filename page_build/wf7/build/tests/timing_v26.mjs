// Load time of page version 26 against version 25 (test only; made by make_tests_v26.py from timing_v24.mjs): 390x844,
// touch, the CPU slowed 4 times, alternating loads (version 25, then version 26, N times each, a fresh tab each time). Per load: DOMContentLoaded and the load
// event (from the navigation timing entry), the navigation's own start (window.__nvInitMs, version 26 only), the main
// thread's script and task time (Performance.getMetrics). Reports medians; the numbers are noisy on a shared machine.
// Usage: node timing_v26.mjs [N=5] -> out/timing_v26.json
import fs from 'node:fs';
import { launch, openPage, navigate, evaluate, closePage, sleep } from '../../nav/tools/cdp.mjs';
const T = 'C:/Users/David/AppData/Local/Temp/claude/C--Users-David-dev-BCA/3bdab4f0-86ee-439c-a05a-34828d6b0a71/scratchpad/wf7/build/tests';
const N = +(process.argv[2] || 5);
const PAGES = { v25: 'file:///' + T + '/regression/v25_live.html', v26: 'file:///' + T + '/../page_v26_preview.html' };
if (!fs.existsSync(T + '/regression/v25_live.html')) {
  const SCR = 'C:/Users/David/AppData/Local/Temp/claude/C--Users-David-dev-BCA/3bdab4f0-86ee-439c-a05a-34828d6b0a71/scratchpad';
  fs.mkdirSync(T + '/regression', { recursive: true });
  fs.writeFileSync(T + '/regression/v25_live.html', Buffer.concat([fs.readFileSync(SCR + '/live5/live_4933.html').subarray(0, 531),
    fs.readFileSync(SCR + '/live5/core_4933.html'), Buffer.from('\n</body></html>')]));
}
const median = a => { const s = a.filter(x => x != null).sort((x, y) => x - y); return s.length ? s[Math.floor(s.length / 2)] : null; };
const runs = { v25: [], v26: [] };
const { proc, cdp } = await launch({ port: 13961, logName: 'edge_timing_v26.log' });
try {
  for (let k = 0; k < N; k++) for (const v of ['v25', 'v26']) {
    const pg = await openPage(cdp, { width: 390, height: 844, mobile: true, touch: true });
    await pg.S('Emulation.setCPUThrottlingRate', { rate: 4 });
    const nav = await navigate(cdp, pg, PAGES[v], { settle: 2000 });
    const t = await evaluate(pg, `(() => { const n = performance.getEntriesByType('navigation')[0]; return { dcl: Math.round(n.domContentLoadedEventEnd), load: Math.round(n.loadEventEnd), init: window.__nvInitMs || null }; })()`);
    const m = await pg.S('Performance.getMetrics');
    const g = n => (m.metrics.find(x => x.name === n) || {}).value;
    t.wall = nav.wallLoadMs; t.scriptS = +g('ScriptDuration').toFixed(2); t.taskS = +g('TaskDuration').toFixed(2);
    t.errors = pg.logs.filter(l => l.level === 'error' || l.kind === 'exception').length;
    runs[v].push(t);
    console.log(v, k, JSON.stringify(t));
    await pg.S('Emulation.setCPUThrottlingRate', { rate: 1 });
    await closePage(cdp, pg);
    await sleep(500);
  }
} finally { try { await cdp.send('Browser.close'); } catch (e) {} await sleep(400); try { proc.kill(); } catch (e) {} }
const med = v => ({ dcl: median(runs[v].map(x => x.dcl)), load: median(runs[v].map(x => x.load)), scriptS: median(runs[v].map(x => x.scriptS)),
  taskS: median(runs[v].map(x => x.taskS)), init: median(runs[v].map(x => x.init)) });
const out = { n: N, note: '390x844, touch, 4x CPU throttle, alternating loads; medians (noisy: a shared machine)', medians: { v25: med('v25'), v26: med('v26') }, runs };
out.medians.dclRatio = +(out.medians.v26.dcl / out.medians.v25.dcl).toFixed(2);
out.medians.loadRatio = +(out.medians.v26.load / out.medians.v25.load).toFixed(2);
fs.mkdirSync(T + '/out', { recursive: true });
fs.writeFileSync(T + '/out/timing_v26.json', JSON.stringify(out, null, 1));
console.log(JSON.stringify(out.medians));
