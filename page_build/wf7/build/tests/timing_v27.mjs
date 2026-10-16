// Load time of page version 27 against the live page (test only; made by make_tests_v27.py from timing_v26.mjs): 390x844,
// touch, the CPU slowed 4 times, alternating loads (the live page, then version 27, N times each, a fresh tab each time). Per load: DOMContentLoaded and the load
// event (from the navigation timing entry), the navigation's own start (window.__nvInitMs, version 27 only), the main
// thread's script and task time (Performance.getMetrics). Reports medians; the numbers are noisy on a shared machine.
// Usage: node timing_v27.mjs [N=5] -> out/timing_v27.json
import fs from 'node:fs';
import { launch, openPage, navigate, evaluate, closePage, sleep } from '../../nav/tools/cdp.mjs';
const T = 'C:/Users/David/AppData/Local/Temp/claude/C--Users-David-dev-BCA/3bdab4f0-86ee-439c-a05a-34828d6b0a71/scratchpad/wf7/build/tests';
const N = +(process.argv[2] || 5);
const PAGES = { live: 'file:///' + T + '/regression/l92ba_live.html', v27: 'file:///' + T + '/../page_v27_preview.html' };
if (!fs.existsSync(T + '/regression/l92ba_live.html')) {
  const SCR = 'C:/Users/David/AppData/Local/Temp/claude/C--Users-David-dev-BCA/3bdab4f0-86ee-439c-a05a-34828d6b0a71/scratchpad';
  fs.mkdirSync(T + '/regression', { recursive: true });
  fs.writeFileSync(T + '/regression/l92ba_live.html', Buffer.concat([fs.readFileSync(SCR + '/live6/live_92ba.html').subarray(0, 531),
    fs.readFileSync(SCR + '/live6/core_92ba.html'), Buffer.from('\n</body></html>')]));
}
const median = a => { const s = a.filter(x => x != null).sort((x, y) => x - y); return s.length ? s[Math.floor(s.length / 2)] : null; };
const runs = { live: [], v27: [] };
const { proc, cdp } = await launch({ port: 13961, logName: 'edge_timing_v27.log' });
try {
  for (let k = 0; k < N; k++) for (const v of ['live', 'v27']) {
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
const out = { n: N, note: '390x844, touch, 4x CPU throttle, alternating loads; medians (noisy: a shared machine)', medians: { live: med('live'), v27: med('v27') }, runs };
out.medians.dclRatio = +(out.medians.v27.dcl / out.medians.live.dcl).toFixed(2);
out.medians.loadRatio = +(out.medians.v27.load / out.medians.live.load).toFixed(2);
fs.mkdirSync(T + '/out', { recursive: true });
fs.writeFileSync(T + '/out/timing_v27.json', JSON.stringify(out, null, 1));
console.log(JSON.stringify(out.medians));
