// Load time of page version 24 against version 23 (test only): 390x844, touch, the CPU slowed 4 times, alternating
// loads (version 23, then version 24, N times each, a fresh tab each time). Per load: DOMContentLoaded and the load
// event (from the navigation timing entry), the navigation's own start (window.__nvInitMs, version 24 only), the main
// thread's script and task time (Performance.getMetrics). Reports medians; the numbers are noisy on a shared machine.
// Usage: node timing_v24.mjs [N=5] -> out/timing_v24.json
import fs from 'node:fs';
import { launch, openPage, navigate, evaluate, closePage, sleep } from '../../nav/tools/cdp.mjs';
const T = 'C:/Users/David/AppData/Local/Temp/claude/C--Users-David-dev-BCA/3bdab4f0-86ee-439c-a05a-34828d6b0a71/scratchpad/wf7/build/tests';
const N = +(process.argv[2] || 5);
const PAGES = { v23: 'file:///' + T + '/regression/v23_live.html', v24: 'file:///' + T + '/../page_v24_preview.html' };
if (!fs.existsSync(T + '/regression/v23_live.html')) {
  const SCR = 'C:/Users/David/AppData/Local/Temp/claude/C--Users-David-dev-BCA/3bdab4f0-86ee-439c-a05a-34828d6b0a71/scratchpad';
  fs.mkdirSync(T + '/regression', { recursive: true });
  fs.writeFileSync(T + '/regression/v23_live.html', Buffer.concat([fs.readFileSync(SCR + '/live4/live_6eea.html').subarray(0, 531),
    fs.readFileSync(SCR + '/live4/core_6eea.html'), Buffer.from('\n</body></html>')]));
}
const median = a => { const s = a.filter(x => x != null).sort((x, y) => x - y); return s.length ? s[Math.floor(s.length / 2)] : null; };
const runs = { v23: [], v24: [] };
const { proc, cdp } = await launch({ port: 13961, logName: 'edge_timing_v24.log' });
try {
  for (let k = 0; k < N; k++) for (const v of ['v23', 'v24']) {
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
const out = { n: N, note: '390x844, touch, 4x CPU throttle, alternating loads; medians (noisy: a shared machine)', medians: { v23: med('v23'), v24: med('v24') }, runs };
out.medians.dclRatio = +(out.medians.v24.dcl / out.medians.v23.dcl).toFixed(2);
out.medians.loadRatio = +(out.medians.v24.load / out.medians.v23.load).toFixed(2);
fs.mkdirSync(T + '/out', { recursive: true });
fs.writeFileSync(T + '/out/timing_v24.json', JSON.stringify(out, null, 1));
console.log(JSON.stringify(out.medians));
