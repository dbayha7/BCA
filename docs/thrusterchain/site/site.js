/* ThrusterChain research interface. No network access, no model inference, no simulation beyond
   one validated plant step. Everything drawn here comes from the embedded JSON data blocks. */
(() => {
  'use strict';
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const parse = id => { const n = document.getElementById(id); return n ? JSON.parse(n.textContent) : null; };
  const ENV = parse('env-data'), RES = parse('results-data'), MAN = parse('manifest-data');
  const esc = v => String(v).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const fmt = (x, n = 3) => (x === null || x === undefined || Number.isNaN(x)) ? 'unavailable' : (typeof x === 'number' ? x.toFixed(n) : String(x));
  const sgn = (x, n = 3) => (typeof x === 'number' && Number.isFinite(x)) ? (x > 0 ? '+' : '') + x.toFixed(n) : 'unavailable';
  const store = { get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }, set(k, v) { try { localStorage.setItem(k, v); } catch (e) { } } };

  /* ---------------- Navigation ---------------- */
  const parts = $$('.part');
  const secs = $$('.sec');
  const order = secs.map(s => s.id);
  let showAll = false;
  function partOf(el) { return el.closest('.part'); }
  function slug(t) { return t.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, ''); }
  secs.forEach(sec => $$('h3', sec).forEach((h, i) => { if (!h.id) h.id = sec.id + '-' + (slug(h.textContent) || i); }));
  const tabs = $('#tabs');
  parts.forEach(p => { const a = document.createElement('a'); a.href = '#' + p.id; a.textContent = p.dataset.title; a.dataset.part = p.id; tabs.appendChild(a); });
  function currentTarget() {
    const h = decodeURIComponent(location.hash.slice(1));
    return (h && document.getElementById(h)) || parts[0];
  }
  function showPart(part) {
    parts.forEach(p => { p.hidden = !showAll && p !== part; });
    $$('a', tabs).forEach(a => a.setAttribute('aria-current', a.dataset.part === part.id ? 'page' : 'false'));
    buildSide(part);
  }
  function buildSide(part) {
    const list = $('#side-list'); list.innerHTML = '';
    $('#side-part').textContent = part.dataset.title;
    $$('.sec', part).forEach(sec => {
      const li = document.createElement('li'); li.dataset.sec = sec.id;
      const a = document.createElement('a'); a.href = '#' + sec.id; a.textContent = $('h2', sec).textContent; li.appendChild(a);
      const subs = $$('h3', sec);
      if (subs.length) { const ol = document.createElement('ol'); subs.forEach(h => { const l = document.createElement('li'); l.dataset.sub = h.id; const b = document.createElement('a'); b.href = '#' + h.id; b.textContent = h.textContent; l.appendChild(b); ol.appendChild(l); }); li.appendChild(ol); }
      list.appendChild(li);
    });
  }
  function setCurrent(secId, subId) {
    $$('#side-list li').forEach(li => li.classList.toggle('current', li.dataset.sec === secId || (subId && li.dataset.sub === subId)));
    const sec = document.getElementById(secId);
    if (sec) { $('#where').textContent = $('h2', sec).textContent; const i = order.indexOf(secId); $$('.js-prev').forEach(b => b.disabled = i <= 0); $$('.js-next').forEach(b => b.disabled = i >= order.length - 1); }
  }
  function route(scroll = true) {
    const target = currentTarget();
    const part = target.classList.contains('part') ? target : partOf(target);
    showPart(part);
    const sec = target.classList.contains('sec') ? target : (target.closest('.sec') || $('.sec', part));
    setCurrent(sec.id, target.tagName === 'H3' ? target.id : null);
    if (target.classList.contains('part')) { if (scroll) window.scrollTo({ top: 0 }); }
    else if (scroll) { requestAnimationFrame(() => target.scrollIntoView({ block: 'start' })); }
    const det = $('.side details'); if (det && window.matchMedia('(max-width: 920px)').matches) det.open = false;
  }
  function step(d) { const i = order.indexOf($('#side-list li.current')?.dataset.sec || order[0]); const j = Math.min(order.length - 1, Math.max(0, i + d)); location.hash = order[j]; }
  $$('.js-prev').forEach(b => b.addEventListener('click', () => step(-1)));
  $$('.js-next').forEach(b => b.addEventListener('click', () => step(1)));
  $('#js-all').addEventListener('click', e => { showAll = !showAll; document.body.classList.toggle('hidden-parts', showAll); e.currentTarget.setAttribute('aria-pressed', String(showAll)); e.currentTarget.textContent = showAll ? 'One part at a time' : 'Show all parts'; route(false); });
  const themeBtn = $('#js-theme');
  function applyTheme(t) { if (t === 'light' || t === 'dark') document.documentElement.dataset.theme = t; else delete document.documentElement.dataset.theme; themeBtn.textContent = 'Theme: ' + (t || 'auto'); }
  applyTheme(store.get('tc-theme'));
  themeBtn.addEventListener('click', () => { const cur = document.documentElement.dataset.theme || 'auto'; const next = cur === 'auto' ? 'light' : cur === 'light' ? 'dark' : 'auto'; store.set('tc-theme', next === 'auto' ? '' : next); applyTheme(next === 'auto' ? null : next); });
  window.addEventListener('hashchange', () => route(true));
  const spy = new IntersectionObserver(entries => {
    entries.filter(e => e.isIntersecting).forEach(e => { const el = e.target; const sec = el.classList.contains('sec') ? el : el.closest('.sec'); if (sec) setCurrent(sec.id, el.tagName === 'H3' ? el.id : null); });
  }, { rootMargin: '-80px 0px -70% 0px', threshold: 0 });
  secs.forEach(s => { spy.observe(s); $$('h3', s).forEach(h => spy.observe(h)); });

  /* ---------------- Shared SVG helpers ---------------- */
  function scale(domain, range) { const [d0, d1] = domain, [r0, r1] = range; const k = (r1 - r0) / ((d1 - d0) || 1); return v => r0 + (v - d0) * k; }
  function ticks(lo, hi, n = 5) { const span = hi - lo || 1; const raw = span / n; const p = Math.pow(10, Math.floor(Math.log10(raw))); const m = raw / p; const stepv = (m >= 5 ? 10 : m >= 2 ? 5 : m >= 1 ? 2 : 1) * p / (m >= 5 ? 2 : 1); const out = []; for (let v = Math.ceil(lo / stepv) * stepv; v <= hi + 1e-9; v += stepv) out.push(+v.toFixed(10)); return out; }
  function axisY(y, lo, hi, x, label) { return ticks(lo, hi).map(v => `<line class="gridline" x1="${x}" x2="100%" y1="${y(v)}" y2="${y(v)}"/><text class="svg-label" x="${x - 6}" y="${y(v) + 4}" text-anchor="end">${v}</text>`).join('') + (label ? `<text class="svg-label" x="${x + 4}" y="${y(hi) - 8 > 14 ? y(hi) - 8 : 14}" text-anchor="start">↑ ${esc(label)}</text>` : ''); }

  /* ---------------- Environment: plant arithmetic (validated at build time against fixtures) ---------------- */
  const KNEE = ENV.physics.knee, SLOPE = ENV.physics.stall_slope, GAMMA = ENV.config.objective_gamma;
  const f32 = v => Math.fround(v);
  function thrustOf(a, stall) { return a.map(v => { if (!stall) return v; const c = Math.max(-KNEE, Math.min(KNEE, v)); const ex = Math.max(Math.abs(v) - KNEE, 0); return c - SLOPE * Math.sign(v) * ex; }); }
  function stepPlant(x, a, stall) { const u = thrustOf(a, stall); const A = ENV.A, B = ENV.B; const nx = []; for (let i = 0; i < 13; i++) { let s = 0; for (let j = 0; j < 13; j++) s += A[i][j] * x[j]; for (let k = 0; k < 4; k++) s += B[i][k] * u[k]; nx.push(s); } return { next: nx, thrust: u }; }
  function rewardOf(x, a) { let s = 0; for (let i = 0; i < 10; i++) s += x[i] * x[i]; let e = 0; for (const v of a) e += v * v; return -(s + 0.3 * e); }
  function clipCmd(a) { return a.map(v => Math.max(-1, Math.min(1, f32(v)))); }

  /* ---------------- Environment: playback ---------------- */
  const sim = { tr: 0, t: 0, timer: null, floatIdx: 0, cmd: null };
  const trSel = $('#tr-select');
  if (trSel) {
    ENV.trajectories.forEach((tr, i) => { const o = document.createElement('option'); o.value = i; o.textContent = `${tr.name.replace(/_/g, ' ')} (${tr.world === 'stall' ? 'stalling world' : 'no-stall world'})`; trSel.appendChild(o); });
    const slider = $('#t-slider'), playBtn = $('#play');
    function tr() { return ENV.trajectories[sim.tr]; }
    function drawSim() {
      const T = tr(), x = T.states[sim.t], a = sim.t < 100 ? T.actions[sim.t] : T.actions[99], u = sim.t < 100 ? T.thrust[sim.t] : T.thrust[99];
      const stall = T.world === 'stall';
      const W = 760, H = 300, base = 150, px = 42;
      let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Five floats, four thrusters and the water at step ${sim.t}">`;
      // water band with current arrow and wave
      const c = x[10], w1 = x[11];
      s += `<rect x="0" y="${base + 50}" width="${W}" height="${H - base - 50}" fill="var(--water)" opacity=".18"/>`;
      let wave = ''; for (let i = 0; i <= 38; i++) { const xx = i * 20; const yy = base + 50 + 10 * (w1) * Math.sin(xx / 60 + sim.t * 0.27); wave += (i ? 'L' : 'M') + xx + ' ' + yy.toFixed(1) + ' '; }
      s += `<path d="${wave}" fill="none" stroke="var(--water)" stroke-width="2"/>`;
      const arrow = 60 + 90 * c; s += `<line x1="${W / 2 - arrow / 2}" y1="${H - 24}" x2="${W / 2 + arrow / 2}" y2="${H - 24}" stroke="var(--water)" stroke-width="4" marker-end="url(#arr)"/><text class="svg-label" x="${W / 2}" y="${H - 32}" text-anchor="middle">current ${c.toFixed(2)}, wave ${w1.toFixed(3)} (push on every float: ${(c + w1).toFixed(3)})</text>`;
      s += `<defs><marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="var(--water)"/></marker></defs>`;
      // floats
      const fx = i => 100 + i * 140 + x[2 * i] * px;
      for (let i = 0; i < 4; i++) { const x1 = fx(i), x2 = fx(i + 1); const mx = (x1 + x2) / 2; const cmd = a[i], th = u[i]; const stalled = stall && Math.abs(cmd) > KNEE;
        s += `<line x1="${x1}" y1="${base}" x2="${x2}" y2="${base}" stroke="var(--rule-strong)" stroke-width="2"/>`;
        s += `<rect class="thruster ${stalled ? 'stalled' : ''}" x="${mx - 14}" y="${base - 12}" width="28" height="24" rx="4"/>`;
        s += `<rect class="bar-bg" x="${mx - 30}" y="${base + 20}" width="60" height="7"/><rect class="bar ${stalled ? 'stalled' : ''}" x="${mx + (cmd < 0 ? 30 * cmd : 0)}" y="${base + 20}" width="${Math.abs(30 * cmd)}" height="7"/>`;
        s += `<rect class="bar-bg" x="${mx - 30}" y="${base + 30}" width="60" height="7"/><rect class="bar ${stalled ? 'stalled' : ''}" x="${mx + (th < 0 ? 30 * th : 0)}" y="${base + 30}" width="${Math.abs(30 * th)}" height="7" opacity=".6"/>`;
        s += `<text class="svg-label" x="${mx}" y="${base - 18}" text-anchor="middle">a${i + 1} ${cmd.toFixed(2)}${stalled ? ' stall' : ''}</text>`; }
      for (let i = 0; i < 5; i++) { const cx = fx(i); s += `<circle class="float" cx="${cx}" cy="${base}" r="16"/><text class="svg-label" x="${cx}" y="${base + 4}" text-anchor="middle">${i + 1}</text>`;
        const v = x[2 * i + 1]; s += `<line x1="${cx}" y1="${base - 28}" x2="${cx + 40 * v}" y2="${base - 28}" stroke="var(--ink-2)" stroke-width="2"/>`;
        s += `<text class="svg-label" x="${cx}" y="${base + 58}" text-anchor="middle">p ${x[2 * i].toFixed(2)}  v ${v.toFixed(2)}</text>`; }
      s += `<text class="svg-title" x="12" y="22">step ${sim.t} of 100 · remaining fraction ${((100 - sim.t) / 100).toFixed(2)} · ${stall ? 'stalling world' : 'no-stall world'}</text>`;
      s += `<text class="svg-label" x="12" y="40">top bar: command; lower bar: effective thrust (orange when the command is past the 0.8 knee)</text>`;
      s += '</svg>';
      $('#sim-stage').innerHTML = s;
      const r = sim.t < 100 ? T.rewards[sim.t] : null; let G = 0; for (let k = 0; k < sim.t; k++) G += Math.pow(GAMMA, k) * T.rewards[k];
      $('#ro-t').textContent = sim.t; $('#ro-r').textContent = r === null ? 'episode over' : r.toFixed(4); $('#ro-G').textContent = G.toFixed(4) + (sim.t === 100 ? ' (final)' : '');
      $('#ro-cost').textContent = (x.slice(0, 10).reduce((p, q) => p + q * q, 0)).toFixed(4); $('#ro-energy').textContent = sim.t < 100 ? (0.3 * a.reduce((p, q) => p + q * q, 0)).toFixed(4) : '0';
      slider.value = sim.t; $('#t-label').textContent = sim.t;
      drawWorked();
    }
    function setT(t) { sim.t = Math.max(0, Math.min(100, t)); drawSim(); }
    trSel.addEventListener('change', () => { sim.tr = +trSel.value; sim.cmd = null; setT(0); });
    slider.addEventListener('input', () => { sim.cmd = null; setT(+slider.value); });
    $('#back').addEventListener('click', () => { sim.cmd = null; setT(sim.t - 1); });
    $('#fwd').addEventListener('click', () => { sim.cmd = null; setT(sim.t + 1); });
    playBtn.addEventListener('click', () => { if (sim.timer) { clearInterval(sim.timer); sim.timer = null; playBtn.textContent = 'Play'; playBtn.setAttribute('aria-pressed', 'false'); return; } playBtn.textContent = 'Pause'; playBtn.setAttribute('aria-pressed', 'true'); sim.timer = setInterval(() => { sim.cmd = null; setT(sim.t >= 100 ? 0 : sim.t + 1); }, 120); });
    $('#tr-table').innerHTML = ENV.trajectories.map((T, i) => `<tr><td>${esc(T.name.replace(/_/g, ' '))}</td><td>${T.world === 'stall' ? 'stalling' : 'no stall'}</td><td class="num">${T.discounted_return.toFixed(3)}</td><td class="num">${T.undiscounted_return.toFixed(3)}</td><td class="num">${T.max_replay_error}</td></tr>`).join('');

    /* worked one-step arithmetic */
    const fsel = $('#w-float'); fsel.addEventListener('change', () => { sim.floatIdx = +fsel.value; drawWorked(); });
    const cmdSliders = [0, 1, 2, 3].map(i => $('#cmd' + (i + 1)));
    cmdSliders.forEach((sl, i) => sl.addEventListener('input', () => { sim.cmd = cmdSliders.map(s => +s.value); drawWorked(); }));
    $('#w-reset').addEventListener('click', () => { sim.cmd = null; drawWorked(); });
    function drawWorked() {
      const T = tr(); const t = Math.min(sim.t, 99); const x = T.states[t]; const saved = T.actions[t]; const stall = T.world === 'stall';
      const a = sim.cmd ? clipCmd(sim.cmd) : saved;
      cmdSliders.forEach((sl, i) => { if (!sim.cmd) sl.value = a[i]; $('#cmdv' + (i + 1)).textContent = (+sl.value).toFixed(2); });
      const { next, thrust } = stepPlant(x, a, stall);
      const i = sim.floatIdx; const pi = 2 * i, vi = 2 * i + 1; const A = ENV.A, B = ENV.B;
      const line = (row, name) => { const terms = []; for (let j = 0; j < 13; j++) { if (Math.abs(A[row][j]) > 1e-12 && j < 10) terms.push(`${A[row][j].toFixed(4)}×${x[j].toFixed(3)}`); } const water = A[row][10] ? `${A[row][10].toFixed(1)}×(${x[10].toFixed(3)} + ${x[11].toFixed(3)})` : null; const th = []; for (let k = 0; k < 4; k++) if (B[row][k]) th.push(`${B[row][k].toFixed(1)}×thrust${k + 1} (${thrust[k].toFixed(4)})`); return `${name}' = ${terms.slice(0, 3).join(' + ')}${terms.length > 3 ? ' + ' + (terms.length - 3) + ' smaller coupling terms' : ''}${th.length ? ' + ' + th.join(' + ') : ''}${water ? ' + ' + water : ''}\n    = <b>${next[row].toFixed(6)}</b>${sim.cmd ? '' : '   (saved next state ' + T.states[t + 1][row].toFixed(6) + ')'}`; };
      const thrustLines = a.map((v, k) => { const ex = Math.max(Math.abs(v) - KNEE, 0); return stall ? `thrust${k + 1} = clip(${v.toFixed(3)}, −0.8, 0.8) − 1.5 × sign × max(|${v.toFixed(3)}| − 0.8, 0) = ${Math.max(-KNEE, Math.min(KNEE, v)).toFixed(3)} − ${(SLOPE * ex).toFixed(3)} = <b>${thrust[k].toFixed(4)}</b>${ex > 0 ? '   ← past the knee: stalls' : ''}` : `thrust${k + 1} = ${v.toFixed(3)} (no stall in this world)`; });
      const cost = x.slice(0, 10).reduce((p, q) => p + q * q, 0), energy = a.reduce((p, q) => p + q * q, 0);
      $('#w-out').innerHTML = `<div class="formula">step t = ${t}, ${stall ? 'stalling' : 'no-stall'} world, command ${sim.cmd ? '(yours)' : '(saved)'} = [${a.map(v => v.toFixed(3)).join(', ')}]\n\n1. Thrust from command (knee 0.8 = float32(0.8), slope 1.5)\n${thrustLines.join('\n')}\n\n2. Reward on the state BEFORE the step, energy on the command even when the propeller stalls\nr = −( Σ s² + 0.3 Σ a² ) = −( ${cost.toFixed(4)} + 0.3 × ${energy.toFixed(4)} ) = <b>${rewardOf(x, a).toFixed(6)}</b>${sim.cmd ? '' : '   (saved reward ' + T.rewards[t].toFixed(6) + ')'}\n\n3. Next state of float ${i + 1} (rows of the frozen matrix A, columns of B; water pushes only the speed row)\n${line(pi, 'p' + (i + 1))}\n${line(vi, 'v' + (i + 1))}\n\n4. Water: current stays ${x[10].toFixed(3)}; wave rotates by 0.27 rad → (${next[11].toFixed(4)}, ${next[12].toFixed(4)})</div>`;
    }
    drawSim();
  }

  /* ---------------- Environment: stall curve and start laws ---------------- */
  const stallFig = $('#stall-curve');
  if (stallFig) {
    const W = 640, H = 300, x = scale([-1, 1], [50, W - 16]), y = scale([-1.05, 1.05], [H - 40, 16]);
    let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Effective thrust and energy cost against the command">`;
    s += axisY(y, -1, 1, 50, 'thrust');
    ticks(-1, 1, 4).forEach(v => s += `<text class="svg-label" x="${x(v)}" y="${H - 20}" text-anchor="middle">${v}</text>`);
    s += `<text class="svg-label" x="${W / 2}" y="${H - 4}" text-anchor="middle">command a (fraction of full throttle)</text>`;
    const pth = (fn, cls, dash) => { let d = ''; for (let i = 0; i <= 200; i++) { const a = -1 + i / 100; d += (i ? 'L' : 'M') + x(a) + ' ' + y(fn(a)) + ' '; } return `<path d="${d}" fill="none" stroke="${cls}" stroke-width="2.5" ${dash ? 'stroke-dasharray="6 4"' : ''}/>`; };
    s += pth(a => a, 'var(--rawq)', true) + pth(a => thrustOf([a], true)[0], 'var(--stall)') + pth(a => 0.3 * a * a, 'var(--accent)', true);
    s += `<line x1="${x(KNEE)}" x2="${x(KNEE)}" y1="${y(-1)}" y2="${y(1)}" stroke="var(--stall)" stroke-dasharray="3 3"/><text class="svg-label" x="${x(KNEE) + 4}" y="${y(1) + 12}">knee 0.8</text>`;
    s += `<text class="svg-label" x="${x(0.95)}" y="${y(0.95)}" text-anchor="end">no-stall world: thrust = a</text><text class="svg-label" x="${x(0.95)}" y="${y(0.45)}" text-anchor="end" fill="var(--stall)">stalling world</text><text class="svg-label" x="${x(0.3)}" y="${y(0.3 * 0.09) - 6}" text-anchor="middle" fill="var(--accent)">energy cost 0.3 a²</text>`;
    s += '</svg>'; stallFig.innerHTML = s;
  }
  const lawFig = $('#start-laws');
  if (lawFig) {
    const laws = ENV.reset.start_laws; const names = Object.keys(laws); const W = 640, H = 230;
    let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Start laws">`;
    const cols = [['current', [0, 1.2]], ['wave', [0, 0.8]], ['kick', [0, 1.6]]];
    cols.forEach(([k, dom], ci) => { const x0 = 120 + ci * 170; const xs = scale(dom, [x0, x0 + 140]);
      s += `<text class="svg-title" x="${x0}" y="18">${k === 'kick' ? 'start spread (kick)' : k === 'wave' ? 'wave amplitude' : 'current'}</text><line x1="${x0}" x2="${x0 + 140}" y1="26" y2="26" stroke="var(--rule-strong)"/>`;
      [dom[0], dom[1]].forEach(v => s += `<text class="svg-label" x="${xs(v)}" y="40" text-anchor="middle">${v}</text>`);
      names.forEach((n, ri) => { const yy = 70 + ri * 40; const v = laws[n][k]; const lo = Array.isArray(v) ? v[0] : v, hi = Array.isArray(v) ? v[1] : v; const shifted = JSON.stringify(v) !== JSON.stringify(laws.recorded[k]);
        if (Array.isArray(v)) s += `<rect x="${xs(lo)}" y="${yy - 9}" width="${Math.max(2, xs(hi) - xs(lo))}" height="18" rx="4" fill="${shifted ? 'var(--stall)' : 'var(--water)'}" opacity=".85"/>`; else s += `<circle cx="${xs(v)}" cy="${yy}" r="8" fill="${shifted ? 'var(--stall)' : 'var(--water)'}"/>`;
        if (ci === 0) s += `<text class="svg-label" x="8" y="${yy + 4}">${n.replace(/_/g, ' ')}</text>`; }); });
    s += '</svg>'; lawFig.innerHTML = s;
  }

  /* ---------------- Results: state explorer ---------------- */
  const banks = RES.development_banks.states;
  const KAPPAS = [0.03, 0.1, 0.3, 1, 3, 10];
  const TIE = 1e-6;
  function select(scores) { const m = Math.max(...scores); for (let i = 0; i < scores.length; i++) if (scores[i] >= m - (TIE + TIE * Math.abs(m))) return i; return 0; }
  const ex = { batch: 'development_1', time: 'all', idx: 0, kappa: 3 };
  const stSel = $('#st-select');
  if (stSel) {
    function filtered() { return banks.map((s, i) => ({ s, i })).filter(o => (ex.batch === 'all' || o.s.batch === ex.batch) && (ex.time === 'all' || o.s.time === +ex.time)); }
    function fillStates() { const f = filtered(); stSel.innerHTML = f.map(o => `<option value="${o.i}">${o.s.batch.replace('development_', 'batch ')} · episode ${o.s.group_id} · time ${o.s.time}</option>`).join(''); if (!f.some(o => o.i === ex.idx)) ex.idx = f[0].i; stSel.value = ex.idx; }
    $('#st-batch').addEventListener('change', e => { ex.batch = e.target.value; fillStates(); drawState(); });
    $('#st-time').addEventListener('change', e => { ex.time = e.target.value; fillStates(); drawState(); });
    stSel.addEventListener('change', () => { ex.idx = +stSel.value; drawState(); });
    $$('#kappa-seg .btn').forEach(b => b.addEventListener('click', () => { ex.kappa = +b.dataset.k; $$('#kappa-seg .btn').forEach(x => x.setAttribute('aria-pressed', String(x === b))); drawState(); drawKappaSummary(); }));
    $('#st-prev').addEventListener('click', () => { const f = filtered(); const j = f.findIndex(o => o.i === ex.idx); if (j > 0) { ex.idx = f[j - 1].i; stSel.value = ex.idx; drawState(); } });
    $('#st-next').addEventListener('click', () => { const f = filtered(); const j = f.findIndex(o => o.i === ex.idx); if (j < f.length - 1) { ex.idx = f[j + 1].i; stSel.value = ex.idx; drawState(); } });
    ['gain', 'loss', 'same'].forEach(kind => $('#case-' + kind).addEventListener('click', () => { const f = filtered(); let best = null; f.forEach(o => { const r = outcome(o.s, ex.kappa); const d = r.vBca - r.vRaw; if ((kind === 'gain' && d > TIE && (!best || d > best.d)) || (kind === 'loss' && d < -TIE && (!best || d < best.d)) || (kind === 'same' && Math.abs(d) <= TIE && !best)) best = { i: o.i, d }; }); if (best) { ex.idx = best.i; stSel.value = ex.idx; drawState(); } }));
    function outcome(s, k) { const iRaw = select(s.q1); const adj = s.q1.map((q, j) => q - k * s.u[j]); const iBca = select(adj); const iBest = s.v.indexOf(Math.max(...s.v)); return { iRaw, iBca, iBest, vRaw: s.v[iRaw], vBca: s.v[iBca], vBest: s.v[iBest], adj }; }
    function drawState() {
      const s = banks[ex.idx], k = ex.kappa, o = outcome(s, k);
      const W = 720, H = 360, L = 64, R = 16, top = 36, bot = 60;
      const vals = [...s.v, ...s.q1, ...o.adj]; const lo = Math.min(...vals), hi = Math.max(...vals); const pad = (hi - lo) * 0.08 || 1;
      const y = scale([lo - pad, hi + pad], [H - bot, top]); const x = j => L + 30 + j * ((W - L - R - 60) / 8);
      let g = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Simulator reference, raw Q1 and BCA-adjusted Q1 for nine candidate actions">`;
      g += axisY(y, lo - pad, hi + pad, L, 'Q (less negative is higher)');
      for (let j = 0; j < 9; j++) { g += `<text class="svg-label" x="${x(j)}" y="${H - bot + 18}" text-anchor="middle">${j}</text>`; g += `<line x1="${x(j)}" x2="${x(j)}" y1="${y(s.v[j])}" y2="${y(s.q1[j])}" stroke="var(--rule-strong)"/>`; }
      g += `<text class="svg-label" x="${(L + W - R) / 2}" y="${H - bot + 36}" text-anchor="middle">candidate action (0 = the frozen actor's own action, 1 = zero command, 2–8 = seeded uniform commands)</text>`;
      g += `<text class="svg-label" x="${(L + W - R) / 2}" y="${H - 8}" text-anchor="middle">signed error = predicted − reference; above the teal point = overestimated, below = underestimated</text>`;
      for (let j = 0; j < 9; j++) { g += `<circle class="pt-ref" cx="${x(j)}" cy="${y(s.v[j])}" r="6"/><rect class="pt-rawq" x="${x(j) - 5}" y="${y(s.q1[j]) - 5}" width="10" height="10"/><polygon class="pt-bca" points="${x(j)},${y(o.adj[j]) - 7} ${x(j) + 7},${y(o.adj[j]) + 5} ${x(j) - 7},${y(o.adj[j]) + 5}"/>`; }
      g += `<circle class="sel-ring" cx="${x(o.iRaw)}" cy="${y(s.q1[o.iRaw])}" r="11"/><circle class="sel-ring" cx="${x(o.iBca)}" cy="${y(o.adj[o.iBca])}" r="11" stroke-dasharray="4 3"/><circle class="sel-ring" cx="${x(o.iBest)}" cy="${y(s.v[o.iBest])}" r="11" stroke="var(--good)"/>`;
      g += `<text class="svg-title" x="${L}" y="18">${esc(s.batch.replace('development_', 'batch '))}, episode ${s.group_id}, query time ${s.time}, remaining horizon ${100 - s.time}</text>`;
      g += '</svg>';
      $('#st-chart').innerHTML = g;
      const d = o.vBca - o.vRaw;
      const verdict = o.iBca === o.iRaw ? `Q1 − ${k}·U keeps raw Q1's choice (candidate ${o.iRaw}).` : (d > TIE ? `Q1 − ${k}·U switches from candidate ${o.iRaw} to ${o.iBca}: the reference return is <b>higher by ${d.toFixed(4)}</b> (improved).` : d < -TIE ? `Q1 − ${k}·U switches from candidate ${o.iRaw} to ${o.iBca}: the reference return is <b>lower by ${(-d).toFixed(4)}</b> (harmed).` : `Q1 − ${k}·U switches candidates but the reference return is unchanged within the numerical band.`);
      $('#st-verdict').innerHTML = `${verdict} Best candidate in this bank: ${o.iBest} with reference ${o.vBest.toFixed(3)}; regret of raw Q1 = ${(o.vBest - o.vRaw).toFixed(4)}, regret of BCA = ${(o.vBest - o.vBca).toFixed(4)}.`;
      $('#st-table').innerHTML = s.q1.map((q, j) => { const e = q - s.v[j], e2 = o.adj[j] - s.v[j]; return `<tr${j === o.iBca ? ' style="font-weight:650"' : ''}><td>${j}${j === 0 ? ' (actor)' : j === 1 ? ' (zero)' : ''}${j === o.iRaw ? ' · raw pick' : ''}${j === o.iBca ? ' · BCA pick' : ''}${j === o.iBest ? ' · best' : ''}</td><td class="num">${s.v[j].toFixed(3)}</td><td class="num">${q.toFixed(3)}</td><td class="num">${s.u[j].toFixed(3)}</td><td class="num">${o.adj[j].toFixed(3)}</td><td class="num" style="color:${e > 0 ? 'var(--bad)' : 'var(--good)'}">${sgn(e, 3)}</td><td class="num" style="color:${e2 > 0 ? 'var(--bad)' : 'var(--good)'}">${sgn(e2, 3)}</td><td class="mono">[${s.actions[j].map(v => v.toFixed(2)).join(', ')}]</td></tr>`; }).join('');
      $('#st-meta').textContent = `episode seed ${s.episode_seed} · state [${s.state.slice(0, 10).map(v => v.toFixed(2)).join(', ')}], current ${s.state[10].toFixed(3)}, wave (${s.state[11].toFixed(3)}, ${s.state[12].toFixed(3)})`;
    }
    function drawKappaSummary() {
      const rows = ['development_1', 'development_2'].map(b => { const o = RES.official[b].methods['bca_kappa_' + String(ex.kappa)]; return o ? `<tr><td>${b.replace('development_', 'batch ')}</td><td class="num">64</td><td class="num">${sgn(o.gain_vs_raw_q, 4)}</td><td class="num">[${o.gain_ci95[0].toFixed(3)}, ${o.gain_ci95[1].toFixed(3)}]</td><td class="num">${o.improved}</td><td class="num">${o.harmed}</td><td class="num">${64 - o.improved - o.harmed}</td><td class="num">${o.regret.toFixed(4)}</td><td class="num">${RES.official[b].methods.raw_q.regret.toFixed(4)}</td></tr>` : ''; }).join('');
      $('#kappa-rows').innerHTML = rows;
    }
    fillStates(); drawState(); drawKappaSummary();
  }

  /* ---------------- Results: aggregate charts from the official tables ---------------- */
  const decFig = $('#decision-chart');
  if (decFig) {
    const W = 720, H = 300, L = 60, groupW = (W - L - 20) / 2;
    let g = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Improved, harmed and unchanged choices for each kappa in both development batches">`;
    const y = scale([0, 64], [H - 50, 44]);
    g += axisY(y, 0, 64, L, '');
    g += `<text class="svg-label" x="${L + 4}" y="14">↑ states, of 64 per batch</text>`;
    ['development_1', 'development_2'].forEach((b, bi) => { const x0 = L + 10 + bi * groupW; g += `<text class="svg-title" x="${x0 + groupW / 2 - 10}" y="${H - 6}" text-anchor="middle">${b.replace('development_', 'batch ')}</text>`;
      KAPPAS.forEach((k, ki) => { const o = RES.official[b].methods['bca_kappa_' + String(k)]; if (!o) return; const bw = (groupW - 20) / KAPPAS.length - 6; const xx = x0 + ki * (bw + 6); const un = 64 - o.improved - o.harmed;
        g += `<rect class="bar-same" x="${xx}" y="${y(un)}" width="${bw}" height="${y(0) - y(un)}"/><rect class="bar-good" x="${xx}" y="${y(un + o.improved)}" width="${bw}" height="${y(0) - y(o.improved)}"/><rect class="bar-bad" x="${xx}" y="${y(64)}" width="${bw}" height="${y(0) - y(o.harmed)}"/>`;
        g += `<text class="svg-label" x="${xx + bw / 2}" y="${H - 34}" text-anchor="middle">κ ${k}</text><text class="svg-label" x="${xx + bw / 2}" y="${H - 20}" text-anchor="middle">${sgn(o.gain_vs_raw_q, 2)}</text>`; }); });
    g += '</svg>'; decFig.innerHTML = g;
  }
  const errFig = $('#error-chart');
  if (errFig) {
    const r = RES.signed_error_recomputed.all; const W = 720, H = 280, L = 60;
    const maxv = Math.max(...KAPPAS.map(k => Math.max(r['kappa_' + k].positive_error_removed, r['kappa_' + k].underestimation_added)));
    const y = scale([0, maxv * 1.1], [H - 60, 24]);
    let g = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Positive error removed and underestimation added by the penalty, per kappa">`;
    g += axisY(y, 0, maxv * 1.1, L, 'mean over 1,152 candidates');
    KAPPAS.forEach((k, i) => { const d = r['kappa_' + k]; const bw = 40; const xx = L + 20 + i * 108; g += `<rect class="bar-good" x="${xx}" y="${y(d.positive_error_removed)}" width="${bw}" height="${y(0) - y(d.positive_error_removed)}"/><rect class="bar-bad" x="${xx + bw + 4}" y="${y(d.underestimation_added)}" width="${bw}" height="${y(0) - y(d.underestimation_added)}"/><text class="svg-label" x="${xx + bw + 2}" y="${H - 40}" text-anchor="middle">κ ${k}</text><text class="svg-label" x="${xx + bw + 2}" y="${H - 26}" text-anchor="middle">${(100 * d.share_overestimated).toFixed(0)}% still over</text>`; });
    g += `<text class="svg-label" x="${W / 2}" y="${H - 6}" text-anchor="middle">green: overestimation removed (mean positive error before minus after); red: underestimation added; recomputed from the saved arrays</text>`;
    g += '</svg>'; errFig.innerHTML = g;
    $('#err-table').innerHTML = ['all', 'development_1', 'development_2'].map(b => { const d = RES.signed_error_recomputed[b]; return `<tr><td>${b === 'all' ? 'both batches' : b.replace('development_', 'batch ')}</td><td class="num">${d.n}</td><td class="num">${d.mean_signed_error.toFixed(3)}</td><td class="num">${(100 * d.share_overestimated).toFixed(1)}%</td><td class="num">${d.mean_abs_error.toFixed(3)}</td><td class="num">${d.mean_u.toFixed(3)} [${d.min_u.toFixed(2)}, ${d.max_u.toFixed(2)}]</td><td class="num">${d.kappa_3.mean_signed_error.toFixed(3)}</td><td class="num">${(100 * d.kappa_3.share_overestimated).toFixed(1)}%</td><td class="num">${d.kappa_3.mean_abs_error.toFixed(3)}</td></tr>`; }).join('');
    const q = RES.official; $('#q1err-official').innerHTML = ['development_1', 'development_2'].map(b => { const e = q[b].q1_errors; const row = (lbl, o) => o && o.mean !== undefined ? `<tr><td>${b.replace('development_', 'batch ')}</td><td>${lbl}</td><td class="num">${o.mean.toFixed(3)}</td><td class="num">[${o.pointwise_ci95[0].toFixed(3)}, ${o.pointwise_ci95[1].toFixed(3)}]</td></tr>` : ''; return row('mean signed error (Q1 − V)', e.mean_signed_error) + row('mean positive optimism max(Q1 − V, 0)', e.mean_positive_optimism) + row('mean absolute error', e.mean_absolute_error); }).join('');
  }
  const selTable = $('#selector-rows');
  if (selTable) {
    const names = [['raw_q', 'Raw Q1'], ['actor', "Frozen actor's action"], ['twin_min', 'Twin-min Q'], ['bca_kappa_1', 'Q1 − 1·U'], ['bca_kappa_3', 'Q1 − 3·U (original BCA strength)'], ['bca_kappa_10', 'Q1 − 10·U'], ['matched_actor_distance_kappa_1', 'Actor-distance control, RMS-matched, κ 1'], ['contrast_actor_distance_kappa_0.1', 'Actor-distance control, contrast-matched, κ 0.1'], ['contrast_actor_distance_kappa_1', 'Actor-distance control, contrast-matched, κ 1'], ['contrast_disagreement_kappa_1', 'Head-disagreement control, contrast-matched, κ 1']];
    selTable.innerHTML = ['development_1', 'development_2'].map(b => names.map(([k, lbl]) => { const o = RES.official[b].methods[k]; return o ? `<tr><td>${b.replace('development_', 'batch ')}</td><td>${lbl}</td><td class="num">${sgn(o.gain_vs_raw_q, 4)}</td><td class="num">[${o.gain_ci95[0].toFixed(3)}, ${o.gain_ci95[1].toFixed(3)}]</td><td class="num">${o.improved} / ${o.harmed}</td><td class="num">${o.regret.toFixed(4)}</td><td class="num">${sgn(o.gain_vs_actor, 4)}</td></tr>` : ''; }).join('')).join('');
  }
  const pilotFig = $('#pilot-chart');
  if (pilotFig) {
    const P = RES.matched_policy_pilot; const pol = ['original', 'seed0_native', 'seed0_distance', 'seed0_bca', 'seed1_native', 'seed1_distance', 'seed1_bca'];
    const lbl = { original: 'original 20k checkpoint', seed0_native: 'stream 0 native', seed0_distance: 'stream 0 distance', seed0_bca: 'stream 0 BCA κ3', seed1_native: 'stream 1 native', seed1_distance: 'stream 1 distance', seed1_bca: 'stream 1 BCA κ3' };
    const cls = { original: 'pt-rawq', seed0_native: 'pt-native', seed1_native: 'pt-native', seed0_distance: 'pt-dist', seed1_distance: 'pt-dist', seed0_bca: 'pt-bca', seed1_bca: 'pt-bca' };
    const all = pol.flatMap(p => P.returns_by_policy[p]); const lo = Math.min(...all), hi = Math.max(...all); const W = 720, H = 340, L = 60;
    const y = scale([lo - 5, hi + 5], [H - 70, 24]); const x = i => L + 50 + i * 92;
    let g = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Discounted returns of seven policies on 32 shared starts">`;
    g += axisY(y, lo - 5, hi + 5, L, 'discounted return (less negative is higher)');
    pol.forEach((p, i) => { P.returns_by_policy[p].forEach((v, j) => g += `<circle class="${cls[p]}" cx="${x(i) + ((j % 8) - 3.5) * 3}" cy="${y(v)}" r="2.6" opacity=".55"/>`); const m = P.mean_discounted_returns[p]; g += `<line x1="${x(i) - 22}" x2="${x(i) + 22}" y1="${y(m)}" y2="${y(m)}" stroke="var(--ink)" stroke-width="3"/><rect x="${x(i) + 24}" y="${y(m) - 9}" width="46" height="16" rx="3" fill="var(--surface)" opacity=".85"/><text class="svg-label" x="${x(i) + 26}" y="${y(m) + 4}" text-anchor="start">${m.toFixed(2)}</text>`; const parts = lbl[p].split(' '); g += `<text class="svg-label" x="${x(i)}" y="${H - 48}" text-anchor="middle">${esc(parts.slice(0, 2).join(' '))}</text><text class="svg-label" x="${x(i)}" y="${H - 34}" text-anchor="middle">${esc(parts.slice(2).join(' '))}</text>`; });
    g += `<text class="svg-label" x="${W / 2}" y="${H - 8}" text-anchor="middle">dots: 32 episodes per policy on the same 32 reset seeds; bar: mean; both streams continue ONE trained checkpoint</text>`;
    g += '</svg>'; pilotFig.innerHTML = g;
    const C = P.comparisons; $('#pilot-rows').innerHTML = ['0', '1'].flatMap(s => ['native', 'distance', 'original'].map(c => { const o = C[s][c]; return `<tr><td>stream ${s}</td><td>BCA κ3 versus ${c === 'original' ? 'the unchanged checkpoint' : c + ' continuation'}</td><td class="num">${sgn(o.mean_gain, 4)}</td><td class="num">[${o.mean_gain_episode_paired_bootstrap_95pct[0].toFixed(3)}, ${o.mean_gain_episode_paired_bootstrap_95pct[1].toFixed(3)}]</td><td class="num">${o.improved_episodes} / ${o.harmed_episodes} / ${o.numerical_ties}</td><td>${o.positive_screen ? 'passed' : 'failed'}</td></tr>`; })).join('');
    const dsp = P.displacement; $('#disp-rows').innerHTML = Object.keys(dsp).map(k => `<tr><td>${esc(k)}</td><td class="num">${dsp[k].logged_probe_action_mse.toFixed(6)}</td><td class="num">${dsp[k].mean_standardized_physical_state_rms.toFixed(6)}</td></tr>`).join('');
  }
  const cachedRows = $('#cached-rows');
  if (cachedRows) {
    const S = RES.cached_comparisons; const sm = S.score_mechanism, lo = S.lower_only;
    const row = (lbl, d, b) => d && d[b] ? `<tr><td>${b.replace('development_', 'batch ')}</td><td>${lbl}</td><td class="num">${sgn(d[b].mean_gain_vs_q1, 4)}</td><td class="num">${d[b].mean_regret.toFixed(4)}</td><td class="num">${d[b].switches_vs_q1}</td><td class="num">${d[b].beneficial_switches} / ${d[b].harmful_switches}</td></tr>` : '';
    cachedRows.innerHTML = ['overall', 'development_1', 'development_2'].map(b => row('Raw Q1', sm.q1, b) + row('Current twin-min Q', sm.twin_min, b) + row("Frozen actor's action", sm.actor, b) + row('Four-draw mean native target (privileged)', sm.mean_native_target, b) + row('Lower-only ideal penalty min(Q1, mean target) (privileged)', lo.lower_only || lo[Object.keys(lo).find(k => /lower/.test(k))], b)).join('');
  }
  const operRows = $('#oper-rows');
  if (operRows) {
    const o = RES.operator_alignment.strata; const metrics = [['u_abs_residual', 'U versus absolute native residual |R|'], ['abs_residual_signed_error', 'absolute residual |R| versus signed Q1 error E'], ['abs_residual_positive_error', 'absolute residual |R| versus positive E'], ['u_signed_error', 'U versus signed Q1 error E'], ['u_positive_error', 'U versus positive E'], ['u_absolute_error', 'U versus absolute Q1 error']];
    operRows.innerHTML = metrics.map(([k, lbl]) => { const d = (o.overall || {})[k]; if (!d) return `<tr><td>${lbl}</td><td class="unavail" colspan="3">unavailable</td></tr>`; const vals = d.map(r => r.value).filter(v => v !== null); const mean = vals.reduce((p, q) => p + q, 0) / vals.length; return `<tr><td>${lbl}</td><td class="num">${mean.toFixed(6)}</td><td class="num">${Math.min(...vals).toFixed(6)} to ${Math.max(...vals).toFixed(6)}</td><td class="num">${d[0].defined}</td></tr>`; }).join('');
  }

  /* ---------------- Evidence: manifest ---------------- */
  const board = $('#status-board');
  if (board && MAN) {
    const rows = MAN.experiments;
    board.innerHTML = rows.map(e => `<div class="card"><div><span class="pill ${e.status}">${esc(e.status.replace(/_/g, ' '))}</span> <span class="pill era">${esc(e.era_label || e.era)}</span></div><div class="title">${esc(e.title)}</div><div class="note">${esc(e.plain || '')}</div><div class="src">${(e.sources || []).slice(0, 2).map(s => esc(s)).join('<br>')}</div></div>`).join('');
    $('#evidence-rows').innerHTML = rows.map(e => `<tr><td><b>${esc(e.title)}</b><br><span class="note">${esc(e.dates || '')} · ${esc(e.machine || '')}</span></td><td><span class="pill ${e.status}">${esc(e.status.replace(/_/g, ' '))}</span></td><td>${(e.headline || []).map(h => `<div><span class="note">${esc(h.label)}:</span> <b>${esc(h.display !== undefined ? h.display : h.value)}</b></div>`).join('') || (e.reported ? `<span class="unavail">unavailable here</span><div class="note">${esc(e.reported)}</div>` : '<span class="unavail">unavailable</span>')}</td><td>${(e.sources || []).map(s => `<div class="srcpath">${esc(s)}</div>`).join('')}</td></tr>`).join('');
    const counts = {}; rows.forEach(e => counts[e.status] = (counts[e.status] || 0) + 1);
    const countsHtml = Object.keys(counts).map(k => `<span class="pill ${k}">${esc(k.replace(/_/g, ' '))}: ${counts[k]}</span>`).join(' ');
    ['#status-counts', '#status-counts-2'].forEach(id => { const n = $(id); if (n) n.innerHTML = countsHtml; });
  }

  /* ---------------- BCA explained: worked score on a real saved state ---------------- */
  const worked = $('#bca-worked');
  if (worked && banks.length) {
    const s = banks[0]; const k = 3; const adj = s.q1.map((q, j) => q - k * s.u[j]);
    const iRaw = select(s.q1), iBca = select(adj), iBest = s.v.indexOf(Math.max(...s.v));
    worked.innerHTML = s.q1.map((q, j) => `<tr${j === iBca ? ' style="font-weight:650"' : ''}><td>${j}${j === 0 ? ' (actor)' : j === 1 ? ' (zero)' : ''}</td><td class="num">${q.toFixed(4)}</td><td class="num">${s.u[j].toFixed(4)}</td><td class="num">${(k * s.u[j]).toFixed(4)}</td><td class="num">${adj[j].toFixed(4)}</td><td class="num">${s.v[j].toFixed(4)}</td></tr>`).join('');
    $('#bca-worked-note').textContent = `Batch ${s.batch.replace('development_', '')}, episode ${s.group_id}, query time ${s.time}. Raw Q1 picks candidate ${iRaw}; Q1 − 3U picks candidate ${iBca}; the best candidate by reference is ${iBest}. Here the widths differ by at most ${(Math.max(...s.u) - Math.min(...s.u)).toFixed(3)} across the nine candidates, so a strength of 3 shifts the scores by about ${(k * Math.min(...s.u)).toFixed(2)} to ${(k * Math.max(...s.u)).toFixed(2)} but changes the order only where Q1 values are close.`;
  }

  route(false);
})();
