/* ThrusterChain: the lesson "Learn step by step" (section#learn). One IIFE with no global names: the port of the
   environment's arithmetic (TCPort) and the lesson's controls. It reads the data block, the JSON script element with the id lesson-data.
   With the page's navigation present (html.nv-on) the navigation shows one step at a time; without it the lesson
   does so itself. */
(function () {
  'use strict';
/* ThrusterChain lesson port ("Learn step by step"): the environment's arithmetic and the critic-free pieces of
   BCA's band arithmetic, mirrored from the repository's Python so that the page reproduces its numbers.

   Plain ES2017, no modules, no globals of its own: paste it inside the lesson's IIFE and it defines one local
   object, TCPort. Load the data block first:  var D = TCPort.use(JSON.parse(el.textContent));
   (use() restores the block's compact forms and rebuilds its derived tables; D is then the plain tree, keyed as the
   lesson spec: D.env.start_state.state, and TCPort.get('env:episodes.stall_aware.first_steps[2].reward')).

   Mirrors (file: function): experiments/shift/thruster_chain.py: thrust, dynamics, push, wave_amplitude,
   outside_recorded_conditions; recorder.py: _linear, LinearController (stall-aware cap 0.8, rising-thrust cap 1),
   HandOnly, Recorder.parts, full_throttle_pattern; truth.py: rollout, q_value, value, points;
   calibration/wbcp.py: crossings, calibrate (lambda_hat in exact arithmetic, lambda_hpd as an order statistic);
   the lesson's own arithmetic in judge/make_new_numbers.py (half-width, score, dose, weights) and the D2 rule and
   number formats of judge/spec_lib.py (chain, decimals, F, FP, FI).
   Exactness: every operation is done in the code's order (JavaScript has no fused multiply-add, numpy's elementwise
   operations have none either), so states, rewards and returns match numpy bit for bit; numpy's sums over arrays
   (wbcp's totals) are pairwise, and pairwiseSum() copies that algorithm. Only Math.tanh (normal driving), Math.exp
   (the weights) and Math.hypot (the wave amplitude) may differ from numpy in the last bit. */
var TCPort = (function () {
  'use strict';

  // ------------------------------------------------------------------ constants (the code's; use() checks the data's)
  var KNEE = 0.800000011920929;   // float32(0.8), thruster_chain.KNEE
  var C = {
    knee: KNEE, stallSlope: 1.5, gain: 0.3, throttleCost: 0.3, waveTurn: 0.27, gamma: 0.98, length: 100,
    trigger: 0.565, damping: 0.25, wobble: 0.3, normalCap: 0.8, fullThrottleProbability: 0.5, jitter: 0.03,
    alpha: 0.1, beta: 0.95, draws: 1000, horizon: 1000, recordedCurrentMax: 0.7, recordedWaveMax: 0.4
  };
  var DS = 10, DX = 13, DA = 4, CURRENT = 10, WAVE1 = 11, WAVE2 = 12;
  var data = null, A = null, B = null, K = null, P = null, Acols = null, Bcols = null;

  // ------------------------------------------------------------------ the data block
  var B64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
  var B64IDX = {};
  for (var bi = 0; bi < 64; bi++) B64IDX[B64.charAt(bi)] = bi;

  function b64bytes(s) {
    var clean = s.replace(/=+$/, '');
    var n = Math.floor(clean.length * 3 / 4), out = new Uint8Array(n), o = 0, i;
    for (i = 0; i + 3 < clean.length + 3 && o < n; i += 4) {
      var a = B64IDX[clean.charAt(i)], b = B64IDX[clean.charAt(i + 1)];
      var c = i + 2 < clean.length ? B64IDX[clean.charAt(i + 2)] : 0, d = i + 3 < clean.length ? B64IDX[clean.charAt(i + 3)] : 0;
      var v = (a << 18) | (b << 12) | (c << 6) | d;
      if (o < n) out[o++] = (v >> 16) & 255;
      if (o < n) out[o++] = (v >> 8) & 255;
      if (o < n) out[o++] = v & 255;
    }
    return out;
  }

  function f64array(s) {
    var bytes = b64bytes(s), view = new DataView(bytes.buffer), n = bytes.length / 8, out = new Array(n);
    for (var i = 0; i < n; i++) out[i] = view.getFloat64(8 * i, true);
    return out;
  }

  function reshape(flat, shape) {
    var off = 0;
    function build(k) {
      var m = shape[k], out = new Array(m), i;
      if (k === shape.length - 1) { for (i = 0; i < m; i++) out[i] = flat[off++]; return out; }
      for (i = 0; i < m; i++) out[i] = build(k + 1);
      return out;
    }
    return build(0);
  }

  function isPlain(o) { return o !== null && typeof o === 'object' && !Array.isArray(o); }
  function keysAre(o, ks) {
    var k = Object.keys(o);
    if (k.length !== ks.length) return false;
    for (var i = 0; i < ks.length; i++) if (!Object.prototype.hasOwnProperty.call(o, ks[i])) return false;
    return true;
  }

  /** The inverse of the data builder's compact forms: {_f64,_shape} base64 float64 arrays, {_e6} millionths,
      {_cols,_data} column tables (lists of objects). Everything else is copied as is. */
  function unpack(node) {
    if (Array.isArray(node)) return node.map(unpack);
    if (!isPlain(node)) return node;
    if (keysAre(node, ['_f64', '_shape'])) return reshape(f64array(node._f64), node._shape);
    if (keysAre(node, ['_e6'])) return node._e6.map(function (i) { return i / 1e6; });
    if (keysAre(node, ['_cols', '_data'])) {
      var cols = node._data.map(unpack), rows = [], n = cols.length ? cols[0].length : 0;
      for (var r = 0; r < n; r++) {
        var row = {};
        for (var c = 0; c < node._cols.length; c++) row[node._cols[c]] = cols[c][r];
        rows.push(row);
      }
      return rows;
    }
    var out = {};
    Object.keys(node).forEach(function (k) { out[k] = unpack(node[k]); });
    return out;
  }

  function derive(tree) {
    var A13 = tree.env.matrices.A_13x13, A10 = [], i, j;
    for (i = 0; i < DS; i++) A10.push(A13[i].slice(0, DS));
    tree.env.matrices.A_float_block_10x10 = A10;
    var s = tree.env.start_state.state, prod = [];
    for (i = 0; i < DS; i++) { var r = []; for (j = 0; j < DS; j++) r.push(A10[i][j] * s[j]); prod.push(r); }
    if (tree.env.one_transition && tree.env.one_transition.stalling_world)
      tree.env.one_transition.stalling_world.products_A10_ij_times_s_j = prod;
    return tree;
  }

  function setMatrices(A13, B13, K4) {
    A = A13; B = B13; K = K4;
    Acols = []; Bcols = [];
    var i, j;
    for (j = 0; j < DX; j++) { var col = []; for (i = 0; i < DX; i++) col.push(A[i][j]); Acols.push(col); }
    for (j = 0; j < DA; j++) { var bc = []; for (i = 0; i < DX; i++) bc.push(B[i][j]); Bcols.push(bc); }
    P = fullThrottlePattern(K);
  }

  /** Load the data block (the parsed JSON): returns the plain tree. Throws if the block disagrees with the code's
      constants (a sign that the block and the port were built from different versions). */
  function use(raw) {
    var tree = derive(unpack(raw));
    var c = tree.env.constants || {};
    var expect = { knee: KNEE, stall_slope: 1.5, gain: 0.3, throttle_cost: 0.3, wave_turn: 0.27, gamma: 0.98, length: 100,
      trigger: 0.565 };
    Object.keys(expect).forEach(function (k) {
      if (c[k] !== undefined && c[k] !== expect[k]) throw new Error('lesson data: constant ' + k + ' is ' + c[k] + ', the port expects ' + expect[k]);
    });
    if (Math.fround(0.8) !== KNEE) throw new Error('lesson port: float32(0.8) is not the knee');
    setMatrices(tree.env.matrices.A_13x13, tree.env.matrices.B_13x4, tree.env.textbook_gain.K_4x13);
    data = tree;
    return tree;
  }

  var KEYTOK = /\.?([A-Za-z_][A-Za-z_0-9\/]*)|\[(\d+)\]|\["([^"]+)"\]/y;
  /** The value at a spec key such as 'env:episodes.stall_aware.first_steps[2].reward' or
      'learn:real_band.records["stall/host/seed0"].R' (as judge/spec_lib.get; the string "inf" becomes Infinity; a
      dotted name may hold a slash, as the spec writes '...first_full_throttle_training_row.infos/state'). */
  function get(key, tree) {
    var t = tree || data, ci = key.indexOf(':');
    if (!t) throw new Error('lesson port: call use() first');
    var node = t[key.slice(0, ci)], path = key.slice(ci + 1), pos = 0, m;
    while (pos < path.length) {
      KEYTOK.lastIndex = pos;
      m = KEYTOK.exec(path);
      if (!m || m.index !== pos) throw new Error('bad key syntax at ' + path.slice(pos) + ' in ' + key);
      node = m[1] !== undefined ? node[m[1]] : (m[2] !== undefined ? node[parseInt(m[2], 10)] : node[m[3]]);
      if (node === undefined) throw new Error('no value at ' + key);
      pos = KEYTOK.lastIndex;
    }
    return (node === 'inf' || node === 'Infinity') ? Infinity : node;
  }

  function num(v) { return v === 'inf' || v === 'Infinity' ? Infinity : (v === '-inf' ? -Infinity : v); }

  // ------------------------------------------------------------------ clipping, thrust, one step
  /** np.clip for one number (keeps −0, passes NaN). */
  function clip(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }
  function clipVec(a, lo, hi) { return a.map(function (v) { return clip(v, lo, hi); }); }
  /** np.sign: +1, −1, or +0 for ±0. */
  function sign(v) { return v > 0 ? 1 : (v < 0 ? -1 : (v === 0 ? 0 : v)); }

  function isStall(world) { return !(world === false || world === 'no_stall' || world === 0); }

  /** The push of one clipped throttle: clip(a, −knee, knee) − slope sign(a) max(|a| − knee, 0); in the no-stall world
      knee 1 and slope 0 (thrust = a). world: 'stall' (default) or 'no_stall' (or true / false). */
  function thrust1(a, world) {
    var st = isStall(world), knee = st ? C.knee : 1.0, slope = st ? C.stallSlope : 0.0;
    return clip(a, -knee, knee) - slope * sign(a) * Math.max(Math.abs(a) - knee, 0.0);
  }
  function thrust(a, world) { return a.map(function (v) { return thrust1(v, world); }); }

  /** Sum of squares in the code's fixed order (v0*v0 + v1*v1 + ...). */
  function sumSquares(v, n) {
    var m = n === undefined ? v.length : n, s = v[0] * v[0];
    for (var i = 1; i < m; i++) s = s + v[i] * v[i];
    return s;
  }
  function floatCost(x) { return sumSquares(x, DS); }
  function throttleCost(a) { var c = clipVec(a, -1, 1); return C.throttleCost * sumSquares(c, DA); }

  /** thruster_chain.dynamics for one state: x' = sum_j A[:, j] x_j + sum_k B[:, k] thrust_k (in this order) and
      r = −(|s|² + 0.3 |a|²) on the clipped command. Returns {next, reward, command (clipped), thrust, floatCost,
      sumSquaresCommand}. */
  function step(x, a, world) {
    var c = clipVec(a, -1.0, 1.0), th = thrust(c, world), nxt = new Array(DX), i, j, k;
    for (i = 0; i < DX; i++) nxt[i] = x[0] * Acols[0][i];
    for (j = 1; j < DX; j++) { var xj = x[j], col = Acols[j]; for (i = 0; i < DX; i++) nxt[i] = nxt[i] + xj * col[i]; }
    for (k = 0; k < DA; k++) { var tk = th[k], bc = Bcols[k]; for (i = 0; i < DX; i++) nxt[i] = nxt[i] + tk * bc[i]; }
    var s2 = sumSquares(x, DS), a2 = sumSquares(c, DA);
    return { next: nxt, reward: -(s2 + C.throttleCost * a2), command: c, thrust: th, floatCost: s2, sumSquaresCommand: a2 };
  }

  /** The water's push c + w1 at a state. */
  function push(x) { return x[CURRENT] + x[WAVE1]; }
  /** The wave's turn by 0.27 rad with the stored cos and sin (the step's own wave rows). */
  function waveTurn(w1, w2) {
    return [w1 * A[WAVE1][WAVE1] + w2 * A[WAVE1][WAVE2], w1 * A[WAVE2][WAVE1] + w2 * A[WAVE2][WAVE2]];
  }
  function waveAmplitude(x) { return Math.hypot(x[WAVE1], x[WAVE2]); }
  /** The floats on their own: A10 s, each row summed in index order (step 1.8 (b)), and the ten products per row. */
  function floatsAlone(s) {
    var out = [];
    for (var i = 0; i < DS; i++) { var acc = s[0] * A[i][0]; for (var j = 1; j < DS; j++) acc = acc + s[j] * A[i][j]; out.push(acc); }
    return out;
  }
  function floatProducts(s, row) {
    var out = [];
    for (var j = 0; j < DS; j++) out.push(A[row][j] * s[j]);
    return out;
  }
  /** outside_recorded_conditions: current > 0.7, wave amplitude > 0.4, or the floats' standardized distance above the
      99% mark. Returns {outside, byCurrent, byWave, byFloats, floatDistance}. */
  function outsideRecorded(x, floatMean, floatSd, radiusP99) {
    var zz = [], i;
    for (i = 0; i < DS; i++) { var z = (x[i] - floatMean[i]) / floatSd[i]; zz.push(z * z); }
    var d = Math.sqrt(pairwiseSum(zz, 0, DS)), bc = x[CURRENT] > C.recordedCurrentMax, bw = waveAmplitude(x) > C.recordedWaveMax, bf = d > radiusP99;
    return { outside: bc || bw || bf, byCurrent: bc, byWave: bw, byFloats: bf, floatDistance: d };
  }
  /** Distance of a command to the deleted ball's center (√ of the four squared differences, in order). */
  function distanceTo(a, center) {
    var d = a.map(function (v, k) { return v - center[k]; });
    return { diffs: d, squares: d.map(function (v) { return v * v; }), distance: Math.sqrt(sumSquares(d, DA)) };
  }

  // ------------------------------------------------------------------ controllers
  /** recorder._linear: the float part K_s s and the water part K_e e, each a fixed-order sum over columns. */
  function linearParts(x, M) {
    var Mm = M || K, fl = [], ex = [];
    for (var k = 0; k < DA; k++) {
      var f = x[0] * Mm[k][0];
      for (var j = 1; j < DS; j++) f = f + x[j] * Mm[k][j];
      var e = x[DS] * Mm[k][DS];
      for (var jj = DS + 1; jj < DX; jj++) e = e + x[jj] * Mm[k][jj];
      fl.push(f); ex.push(e);
    }
    return { fl: fl, ex: ex };
  }
  /** LinearController: clip(−(K_s s + K_e e), −cap, cap); cap null leaves it unclipped. */
  function linear(x, cap, M) {
    var p = linearParts(x, M), u = p.fl.map(function (f, k) { return -(f + p.ex[k]); });
    return cap === null || cap === undefined ? u : clipVec(u, -cap, cap);
  }
  function stallAware(x) { return linear(x, 0.8); }
  function risingThrust(x) { return linear(x, 1.0); }
  function zeroCommand() { return [0, 0, 0, 0]; }
  /** HandOnly (normal driving without wobble): 0.8 tanh(−(0.25 K_s s + K_e e) / 0.8). */
  function handOnly(x) {
    var p = linearParts(x);
    return p.fl.map(function (f, k) { var u = -(C.damping * f + p.ex[k]); return C.normalCap * Math.tanh(u / C.normalCap); });
  }
  /** recorder.full_throttle_pattern: P = −K[:, current] / |K[0, current]|. */
  function fullThrottlePattern(M) {
    var Mm = M || K, d = Math.abs(Mm[0][CURRENT]);
    return Mm.map(function (row) { return -row[CURRENT] / d; });
  }
  /** Recorder.parts for one state with its noise (4 numbers or null) and coin (or null): normal driving
      0.8 tanh(u / 0.8) with u = −(0.25 K_s s + K_e e) + 0.3 noise; inside the region (c + w1 > 0.565) with
      coin < 0.5 the full-throttle command clip(P + 0.03 noise, −1, 1). */
  function recorder(x, noise, coin) {
    var p = linearParts(x), u = [], hand = [], full = [], k;
    for (k = 0; k < DA; k++) {
      var uk = -(C.damping * p.fl[k] + p.ex[k]);
      if (noise) uk = uk + C.wobble * noise[k];
      u.push(uk);
      hand.push(C.normalCap * Math.tanh(uk / C.normalCap));
      full.push(clip(noise ? P[k] + C.jitter * noise[k] : P[k], -1.0, 1.0));
    }
    var region = push(x) > C.trigger, press = region && coin !== null && coin !== undefined && coin < C.fullThrottleProbability;
    return { command: press ? full : hand, region: region, press: press, u: u, hand: hand, full: full, Ks_s: p.fl, Ke_e: p.ex };
  }
  /** A policy function (x, t) -> command by name: 'stall_aware', 'rising_thrust', 'zero_throttle' (or 'no_throttle'),
      'hand_only'; or replay(commands); or recorderStreams(noise[t][k], coin[t]). */
  function policy(name) {
    if (name === 'stall_aware') return function (x) { return stallAware(x); };
    if (name === 'rising_thrust') return function (x) { return risingThrust(x); };
    if (name === 'zero_throttle' || name === 'no_throttle') return function (x) { return linear(x, null, ZERO_K); };
    if (name === 'hand_only' || name === 'normal_driving_only_no_wobble') return function (x) { return handOnly(x); };
    throw new Error('unknown controller ' + name);
  }
  var ZERO_K = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]];
  function replay(commands) { return function (x, t) { return commands[t]; }; }
  function recorderStreams(noise, coin) { return function (x, t) { return recorder(x, noise[t], coin[t]).command; }; }

  // ------------------------------------------------------------------ episodes, returns, values, points
  /** truth.rollout: length steps (100) from x0; a = clip(policy(x, t), −1, 1); J = sum r, Jd = sum 0.98^t r with the
      discount made by repeated multiplication. Returns {J, Jd, states (length+1), commands, rewards, running,
      runningDiscounted}. */
  function rollout(pol, x0, world, opts) {
    var o = opts || {}, L = o.length === undefined ? C.length : o.length, g = o.gamma === undefined ? C.gamma : o.gamma;
    var x = x0.slice(), J = 0, Jd = 0, disc = 1.0, S = [x], Cm = [], R = [], run = [], runD = [];
    for (var t = 0; t < L; t++) {
      var a = clipVec(pol(x, t), -1.0, 1.0), st = step(x, a, world);
      J = J + st.reward;
      Jd = Jd + disc * st.reward;
      disc *= g;
      S.push(st.next); Cm.push(a); R.push(st.reward); run.push(J); runD.push(Jd);
      x = st.next;
    }
    return { J: J, Jd: Jd, states: S, commands: Cm, rewards: R, running: run, runningDiscounted: runD };
  }
  /** truth.q_value: the value of command a at x, then the policy for horizon − 1 more steps (1,000 in all). */
  function qValue(pol, x, a, world, horizon, gamma) {
    var H = horizon === undefined ? C.horizon : horizon, g = gamma === undefined ? C.gamma : gamma;
    var st = step(x, a, world), xx = st.next, q = st.reward, disc = 1.0;
    for (var i = 1; i < H; i++) {
      disc *= g;
      var s2 = step(xx, pol(xx, i), world);
      xx = s2.next;
      q = q + disc * s2.reward;
    }
    return q;
  }
  /** truth.value: Q(x, pi(x)). */
  function value(pol, x, world, horizon, gamma) { return qValue(pol, x, pol(x, 0), world, horizon, gamma); }
  /** Partial discounted sums r0 + γ r1 + ... + γ^(H−1) r_(H−1) for H = 1 ... H (the horizon slider of step 2.4). */
  function partialSums(pol, x0, world, H, gamma) {
    var n = H === undefined ? C.horizon : H, g = gamma === undefined ? C.gamma : gamma;
    var x = x0.slice(), acc = 0.0, disc = 1.0, out = [];
    for (var t = 0; t < n; t++) {
      var st = step(x, pol(x, t), world);
      x = st.next;
      acc += disc * st.reward;
      disc *= g;
      out.push(acc);
    }
    return out;
  }
  /** Returns of a list of rewards with discount gamma: {J, Jd, running, runningDiscounted, weights}. */
  function returns(rewards, gamma) {
    var g = gamma === undefined ? C.gamma : gamma, J = 0, Jd = 0, disc = 1.0, run = [], runD = [], w = [];
    for (var t = 0; t < rewards.length; t++) {
      w.push(disc);
      J = J + rewards[t];
      Jd = Jd + disc * rewards[t];
      disc *= g;
      run.push(J); runD.push(Jd);
    }
    return { J: J, Jd: Jd, running: run, runningDiscounted: runD, weights: w };
  }
  function discountWeights(gamma, n) {
    var g = gamma === undefined ? C.gamma : gamma, d = 1.0, out = [];
    for (var t = 0; t < n; t++) { out.push(d); d *= g; }
    return out;
  }
  /** truth.points: 100 (J − J_recording) / (J_top − J_recording). */
  function points(J, anchor) { return 100.0 * (J - anchor.recording) / (anchor.top - anchor.recording); }

  // ------------------------------------------------------------------ BCA's band arithmetic (no critic needed)
  /** numpy's pairwise summation (np.sum, ndarray.sum over a contiguous axis): blocks of 8 accumulators up to 128
      numbers, halves above that, starting from 0. */
  function pairwiseSum(a, i0, n) {
    var s = i0 === undefined ? 0 : i0, m = n === undefined ? a.length - s : n, i, res;
    if (m < 8) { res = 0.0; for (i = 0; i < m; i++) res += a[s + i]; return res; }
    if (m <= 128) {
      var r0 = a[s], r1 = a[s + 1], r2 = a[s + 2], r3 = a[s + 3], r4 = a[s + 4], r5 = a[s + 5], r6 = a[s + 6], r7 = a[s + 7];
      for (i = 8; i < m - (m % 8); i += 8) {
        r0 += a[s + i]; r1 += a[s + i + 1]; r2 += a[s + i + 2]; r3 += a[s + i + 3];
        r4 += a[s + i + 4]; r5 += a[s + i + 5]; r6 += a[s + i + 6]; r7 += a[s + i + 7];
      }
      res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
      for (; i < m; i++) res += a[s + i];
      return res;
    }
    var n2 = Math.floor(m / 2);
    n2 -= n2 % 8;
    return pairwiseSum(a, s, n2) + pairwiseSum(a, s + n2, m - n2);
  }
  function cumsum(a) { var out = new Array(a.length), c = 0; for (var i = 0; i < a.length; i++) { c = i === 0 ? a[0] : c + a[i]; out[i] = c; } return out; }

  /** A decimal number as an exact fraction [p, q] (Fraction(str(x)) in Python; the shortest decimal text is the
      same in both languages). */
  function decimalFraction(x) {
    var s = String(x), m = /^(-?)(\d+)(?:\.(\d+))?(?:e([+-]?\d+))?$/.exec(s);
    if (!m) throw new Error('not a decimal: ' + s);
    var digits = m[2] + (m[3] || ''), exp = (m[4] ? parseInt(m[4], 10) : 0) - (m[3] ? m[3].length : 0);
    var p = parseInt(digits, 10), q = 1;
    if (exp >= 0) p *= Math.pow(10, exp); else q = Math.pow(10, -exp);
    if (!(p <= 9007199254740991 && q <= 9007199254740991)) throw new Error('fraction too long: ' + s);
    var a = p, b = q;
    while (b) { var t = a % b; a = b; b = t; }
    return [(m[1] ? -1 : 1) * p / a, q / a];
  }
  /** ceil(Fraction(str(share)) × m) − 1: the order-statistic index wbcp uses for lambda_hpd (exact). */
  function ceilFractionIndex(share, m) {
    var f = decimalFraction(share), num = f[0] * m;
    if (!(Math.abs(num) <= 9007199254740991)) throw new Error('too large');
    return Math.floor((num + f[1] - 1) / f[1]) - 1;
  }

  // exact sums of doubles (Shewchuk expansions), for wbcp's exact weighted empirical threshold
  function twoSum(a, b) { var s = a + b, bb = s - a, err = (a - (s - bb)) + (b - bb); return [s, err]; }
  function grow(e, b) {
    var out = [], q = b;
    for (var i = 0; i < e.length; i++) { var r = twoSum(q, e[i]); q = r[0]; if (r[1] !== 0) out.push(r[1]); }
    out.push(q);
    return out;
  }
  var SPLIT = 134217729;   // 2^27 + 1
  function twoProduct(a, b) {     // Shewchuk's Two-Product (Dekker's splitting, no fused multiply-add needed)
    var x = a * b, c = SPLIT * a, ah = c - (c - a), al = a - ah, d = SPLIT * b, bh = d - (d - b), bl = b - bh;
    var err1 = x - (ah * bh), err2 = err1 - (al * bh), err3 = err2 - (ah * bl);
    return [x, (al * bl) - err3];
  }
  function scaleExp(e, b) {
    var out = [0];
    for (var i = 0; i < e.length; i++) { var p = twoProduct(e[i], b); out = grow(grow(out, p[1]), p[0]); }
    return out;
  }
  function expSign(e) { for (var i = e.length - 1; i >= 0; i--) if (e[i] !== 0) return e[i] > 0 ? 1 : -1; return 0; }

  /** wbcp.calibrate's Eq. (1) on sorted weights: the smallest index k whose exact weight sum w_0 + ... + w_k reaches
      (1 − alpha) × the exact total (alpha read as its decimal fraction). Uniform weights give rank ceil((1 − alpha) n). */
  function empiricalIndex(sortedWeights, alpha) {
    var f = decimalFraction(alpha === undefined ? C.alpha : alpha), p = f[0], q = f[1], total = [0], i;
    for (i = 0; i < sortedWeights.length; i++) total = grow(total, sortedWeights[i]);
    var target = scaleExp(total, -(q - p)), covered = [0];      // covered ≥ (q − p)/q · total ⟺ q·covered − (q − p)·total ≥ 0
    for (i = 0; i < sortedWeights.length; i++) {
      covered = grow(covered, sortedWeights[i]);
      var diff = scaleExp(covered, q);
      for (var j = 0; j < target.length; j++) diff = grow(diff, target[j]);
      if (expSign(diff) >= 0) return i;
    }
    return sortedWeights.length - 1;
  }
  /** The empirical threshold with equal weights: the ceil(share × n)-th smallest score (share 0.9 by default; a
      share given in whole percent, e.g. 90, also works: step 5.2's slider). Returns {rank, score}. */
  function empiricalThreshold(sortedScores, share) {
    var n = sortedScores.length, sh = share === undefined ? 0.9 : share, rank;
    if (sh > 1) rank = Math.ceil(sh * n / 100);    // whole percent: sh·n is an exact integer, so this ceiling is exact
    else rank = ceilFractionIndex(sh, n) + 1;
    rank = Math.max(1, Math.min(n, rank));
    return { rank: rank, score: sortedScores[rank - 1] };
  }

  /** wbcp.crossings for one draw: masses E_i w_i on the sorted scores, total = (pairwise) sum + E_(n+1) × testMass;
      the first rank whose running mass reaches (1 − alpha) × total, or Infinity (never crosses).
      sortedWeights null = all 1 (BCA's band). Returns {index, rank, score}. */
  function crossing(sortedScores, E, alpha, sortedWeights, testMass) {
    var n = sortedScores.length, a = alpha === undefined ? C.alpha : alpha, w = sortedWeights, tm = testMass === undefined ? 1.0 : testMass;
    var mass = new Array(n), i;
    for (i = 0; i < n; i++) mass[i] = w ? E[i] * w[i] : E[i] * 1.0;
    var total = pairwiseSum(mass, 0, n) + E[n] * tm, thr = (1.0 - a) * total, c = 0;
    for (i = 0; i < n; i++) {
      c = i === 0 ? mass[0] : c + mass[i];
      if (c >= thr) return { index: i, rank: i + 1, score: sortedScores[i], total: total };
    }
    return { index: -1, rank: null, score: Infinity, total: total };
  }
  /** One Bayesian bootstrap draw shown by hand (steps 5.3): total, masses, the unseen row's mass, running totals.
      variant 'wbcp' (default; learn_facts' worked draws): total = pairwise sum of the n row exponentials + E_(n+1),
      masses = E_i / total, running = cumulative sum of the masses. variant 'all' (lesson_new_numbers' draw 1 on the
      851 scores): total = pairwise sum of all n + 1 exponentials, running = (cumulative sum of E_i) / total. */
  function drawMasses(E, n, variant) {
    var total, masses = [], running = [], i, c = 0, cross = -1;
    if (variant === 'all') {
      total = pairwiseSum(E, 0, n + 1);
      for (i = 0; i < n; i++) { c = i === 0 ? E[0] : c + E[i]; running.push(c / total); masses.push(E[i] / total); }
      for (i = 0; i < n; i++) if (running[i] >= 0.9) { cross = i; break; }        // make_new_numbers: cum / total >= 0.9
    } else {
      total = pairwiseSum(E, 0, n) + E[n];
      var thr = (1.0 - C.alpha) * total;
      for (i = 0; i < n; i++) {
        var mi = E[i] / total; masses.push(mi);
        c = i === 0 ? mi : c + mi; running.push(c);
      }
      c = 0;
      for (i = 0; i < n; i++) { c = i === 0 ? E[0] : c + E[i]; if (c >= thr) { cross = i; break; } }   // wbcp: cum E >= 0.9 total
    }
    return { total: total, masses: masses, unseenMass: E[n] / total, running: running, crossingIndex: cross,
      crossingRank: cross < 0 ? null : cross + 1 };
  }
  /** wbcp's credible threshold: the ceil(beta × draws)-th smallest crossing (Infinity when fewer cross). */
  function credibleThreshold(crossingScores, beta) {
    var s = crossingScores.map(num).slice().sort(function (x, y) { return x - y; });
    var idx = ceilFractionIndex(beta === undefined ? C.beta : beta, s.length);
    return { value: s[idx], index: idx, sorted: s };
  }
  /** wbcp.calibrate on given exponentials (rows E[m] of n + 1 numbers, the draws BCA's generator made): weights
      rescaled by their largest, scores sorted stably, Eq. (1) in exact arithmetic, every draw's crossing, Eq. (7). */
  function calibrate(scores, Erows, weights, testMass, alpha, beta) {
    var n = scores.length, a = alpha === undefined ? C.alpha : alpha, b = beta === undefined ? C.beta : beta;
    var w = weights ? weights.slice() : null, tm = 1.0, i;
    if (w) {
      var mx = -Infinity;
      for (i = 0; i < n; i++) if (w[i] > mx) mx = w[i];
      w = w.map(function (v) { return v / mx; });
      tm = testMass / mx;
    }
    var order = scores.map(function (v, k) { return k; }).sort(function (x, y) { return (scores[x] - scores[y]) || (x - y); });
    var ss = order.map(function (k) { return scores[k]; }), sw = w ? order.map(function (k) { return w[k]; }) : null;
    var kHat = empiricalIndex(sw || ss.map(function () { return 1.0; }), a);
    var cross = Erows.map(function (E) { return crossing(ss, E, a, sw, tm).score; });
    var cred = credibleThreshold(cross, b);
    var lamHat = ss[kHat];
    var nEff = null;
    if (w) { var sum = pairwiseSum(w), dot = 0; for (i = 0; i < n; i++) dot = i === 0 ? w[0] * w[0] : dot + w[i] * w[i]; nEff = sum * sum / dot; }
    else nEff = n;
    return { threshold: Math.max(lamHat, cred.value), lambdaHat: lamHat, lambdaHatRank: kHat + 1, lambdaHpd: cred.value,
      crossings: cred.sorted, finite: cred.sorted.filter(function (v) { return v !== Infinity; }).length,
      sortedScores: ss, sortedWeights: sw, nEff: nEff };
  }
  /** Rank of a score among sorted scores: how many are at or below it (step 5.1). */
  function countAtOrBelow(sortedScores, z) { var c = 0; for (var i = 0; i < sortedScores.length; i++) if (sortedScores[i] <= z) c++; return c; }
  /** The band's half-width U = β η max(g, 1e−6) (computed as (β η) g, as the lesson's numbers were). */
  function halfWidth(beta, eta, g) { return beta * eta * Math.max(g, 1e-6); }
  /** A row's score |y − q| / (η g). */
  function score(err, eta, g) { return Math.abs(err) / (g * eta); }
  /** The dose m = 1 + 0.5 U / (U + η), and the same from the ratio r = U/η: 1 + 0.5 r / (r + 1). */
  function dose(U, eta) { return 1 + 0.5 * U / (U + eta); }
  function doseFromRatio(r) { return 1 + 0.5 * r / (r + 1); }
  /** The lower-edge penalty [U(x, π(x)) − U(x, a)]₊. */
  function lowerEdgePenalty(Upi, Ua) { return Math.max(Upi - Ua, 0.0); }
  /** The weighting law: M(ξ) = (exp(1.1 ξ) − exp(0.1 ξ)) / ξ written as exp(0.1 ξ) expm1(ξ) / ξ like the code
      (M(0) = 1), the density h(c) = exp(ξ c) / M(ξ) and the strong-test weight ω(c) = (M(−2) / M(2)) exp(4 c) =
      exp(4 (c − 0.6)). */
  function M(xi) { return xi === 0 ? 1.1 - 0.1 : Math.exp(xi * 0.1) * Math.expm1(xi * (1.1 - 0.1)) / xi; }
  function density(c, xi) { return Math.exp(xi * c) / M(xi); }
  function weightStrong(c, strongConstant) {
    var k = strongConstant === undefined ? M(-2.0) / M(2.0) : strongConstant;
    return k * Math.exp(4 * c);
  }
  function weightStrongExpForm(c) { return Math.exp(4 * (c - 0.6)); }
  /** The constant's two parts as lesson_new_numbers writes them: (exp(1.1 ξ) − exp(0.1 ξ)) / ξ. */
  function Mdiff(xi) { return (Math.exp(1.1 * xi) - Math.exp(0.1 * xi)) / xi; }

  // ------------------------------------------------------------------ the D2 rule and the number formats (spec_lib)
  var MINUS = '−';
  var P10 = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15, 1e-16,
    1e-17, 1e-18, 1e-19, 1e-20];   // Python's 10 ** -d (checked equal to these literals)
  function pow10neg(d) { return d < P10.length ? P10[d] : Math.pow(10, -d); }
  /** spec_lib.decimals: the most decimals of any number written in the text. */
  function decimals(s) {
    var re = /\d[\d,]*(?:\.(\d+))?/g, m, best = 0, any = false;
    while ((m = re.exec(s)) !== null) { any = true; var d = m[1] ? m[1].length : 0; if (d > best) best = d; }
    return any ? best : 0;
  }
  var SUPS = '⁰¹²³⁴⁵⁶⁷⁸⁹';
  /** spec_lib.ev: the value of a shown expression (numbers with U+2212 minus and thousands commas, + − × ÷ · * /,
      powers written as superscripts, √, ⌈ ⌉, exp, tanh, ln, max, min, abs, parentheses). */
  function evalShown(text) {
    var t = String(text).replace(new RegExp(MINUS, 'g'), '-').replace(/×/g, '*').replace(/÷/g, '/').replace(/·/g, '*');
    t = t.replace(/(\d),(\d{3})(?!\d)/g, '$1$2').replace(/(\d),(\d{3})(?!\d)/g, '$1$2');
    var pos = 0;
    function peek() { while (pos < t.length && t.charAt(pos) === ' ') pos++; return t.charAt(pos); }
    function expect(ch) { if (peek() !== ch) throw new Error('expected ' + ch + ' in ' + text); pos++; }
    function sup() {     // superscript digits right after a number or ')'
      var s = '';
      while (pos < t.length && SUPS.indexOf(t.charAt(pos)) >= 0) { s += SUPS.indexOf(t.charAt(pos)); pos++; }
      return s === '' ? null : parseInt(s, 10);
    }
    function number() {
      var m = /^\d+(?:\.\d+)?(?:[eE][+-]?\d+)?/.exec(t.slice(pos));
      if (!m) throw new Error('number expected at ' + t.slice(pos) + ' in ' + text);
      pos += m[0].length;
      return parseFloat(m[0]);
    }
    function args() { expect('('); var a = [expr()]; while (peek() === ',') { pos++; a.push(expr()); } expect(')'); return a; }
    function atom() {
      var ch = peek(), v, e;
      if (ch === '(') { pos++; v = expr(); expect(')'); }
      else if (ch === '√') {     // √( … ) or √number
        pos++;
        if (peek() === '(') { pos++; v = Math.sqrt(expr()); expect(')'); } else v = Math.sqrt(number());
        return v;
      } else if (ch === '⌈') { pos++; v = Math.ceil(expr()); expect('⌉'); }
      else if (/[0-9.]/.test(ch)) v = number();
      else {
        var m = /^(exp|tanh|ln|max|min|abs)/.exec(t.slice(pos));
        if (!m) throw new Error('cannot read ' + t.slice(pos) + ' in ' + text);
        pos += m[1].length;
        var a = args();
        v = m[1] === 'exp' ? Math.exp(a[0]) : m[1] === 'tanh' ? Math.tanh(a[0]) : m[1] === 'ln' ? Math.log(a[0])
          : m[1] === 'abs' ? Math.abs(a[0]) : m[1] === 'max' ? Math.max.apply(null, a) : Math.min.apply(null, a);
      }
      e = sup();
      if (e !== null) v = Math.pow(v, e);
      return v;
    }
    function unary() { var ch = peek(); if (ch === '-') { pos++; return -unary(); } if (ch === '+') { pos++; return unary(); } return atom(); }
    function term() {
      var v = unary();
      for (;;) { var ch = peek(); if (ch === '*') { pos++; v = v * unary(); } else if (ch === '/') { pos++; v = v / unary(); } else return v; }
    }
    function expr() {
      var v = term();
      for (;;) { var ch = peek(); if (ch === '+') { pos++; v = v + term(); } else if (ch === '-') { pos++; v = v - term(); } else return v; }
    }
    var out = expr();
    if (peek() !== '') throw new Error('unread text ' + t.slice(pos) + ' in ' + text);
    return out;
  }
  /** The D2 rule: '=' when the left value is the right text's value within half a unit of the right text's last
      shown decimal (times 1 + 1e−9, plus 1e−12), else '≈' (judge/spec_lib.chain). */
  function rel(leftValue, rightText) {
    var vb = evalShown(rightText), tol = 0.5 * pow10neg(decimals(rightText)) * (1 + 1e-9) + 1e-12;
    return Math.abs(leftValue - vb) <= tol ? '=' : '≈';
  }
  /** The relations of a worked line's parts (each part a shown expression): ['=', '≈', ...]. */
  function chain(parts) {
    var out = [];
    for (var i = 0; i + 1 < parts.length; i++) out.push(rel(evalShown(parts[i]), parts[i + 1]));
    return out;
  }
  /** chain() written out: 'label = p0 = p1 ≈ p2'. */
  function chainText(label, parts) {
    var r = chain(parts), s = label + ' = ' + parts[0];
    for (var i = 1; i < parts.length; i++) s += ' ' + r[i - 1] + ' ' + parts[i];
    return s;
  }
  /** Fixed-point text of a non-negative number, rounded half to even on its exact binary value (Python's format). */
  function fixedHalfEven(x, d) {
    var s = x.toFixed(d), y = x, k = 0;     // toFixed rounds an exact tie up; Python rounds it to even
    while (y !== Math.floor(y) && k < 1100) { y *= 2; k++; }
    if (k !== d + 1) return s;              // a tie needs exactly d + 1 binary fraction digits
    var digits = s.replace('.', ''), last = parseInt(digits.charAt(digits.length - 1), 10);
    if (last % 2 === 0) return s;
    var arr = digits.split(''), i = arr.length - 1;
    while (i >= 0) { if (arr[i] === '0') { arr[i] = '9'; i--; } else { arr[i] = String(parseInt(arr[i], 10) - 1); break; } }
    var str = arr.join('');
    if (d > 0) str = str.slice(0, str.length - d) + '.' + str.slice(str.length - d);
    str = str.replace(/^0+(?=\d)/, '');
    return str;
  }
  /** spec_lib.F: fixed-point text with a real minus sign; plus=true writes + on positives; |x| below half a unit of
      the last decimal is written as 0. */
  function fmt(x, d, plus) {
    var dd = d === undefined ? 4 : d, v = +x;
    if (Math.abs(v) < 0.5 * pow10neg(dd)) v = 0.0;
    var s = fixedHalfEven(Math.abs(v), dd);
    if (v < 0) return MINUS + s;
    return plus ? '+' + s : s;
  }
  /** spec_lib.FP: a share as a percentage, '91.8%'. */
  function fmtPct(x, d) {
    var v = 100 * x, neg = v < 0 || (v === 0 && 1 / v < 0);   // Python keeps the sign of −0.0
    return (neg ? '-' : '') + fixedHalfEven(Math.abs(v), d === undefined ? 1 : d) + '%';
  }
  /** spec_lib.FI: a whole number with thousands separators (rounded half to even, ASCII hyphen-minus as Python). */
  function fmtInt(x) {
    var v = +x, f = Math.floor(v), r = v - f, n = r > 0.5 ? f + 1 : (r < 0.5 ? f : (f % 2 === 0 ? f : f + 1));
    var s = String(Math.abs(n)).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
    return (n < 0 ? '-' : '') + s;
  }

  return {
    constants: C, use: use, get: get, unpack: unpack, num: num,
    clip: clip, clipVec: clipVec, sign: sign, thrust1: thrust1, thrust: thrust, step: step, push: push,
    waveTurn: waveTurn, waveAmplitude: waveAmplitude, floatsAlone: floatsAlone, floatProducts: floatProducts,
    sumSquares: sumSquares, floatCost: floatCost, throttleCost: throttleCost, outsideRecorded: outsideRecorded,
    distanceTo: distanceTo,
    linearParts: linearParts, linear: linear, stallAware: stallAware, risingThrust: risingThrust,
    zeroCommand: zeroCommand, handOnly: handOnly, fullThrottlePattern: fullThrottlePattern, recorder: recorder,
    policy: policy, replay: replay, recorderStreams: recorderStreams,
    rollout: rollout, qValue: qValue, value: value, partialSums: partialSums, returns: returns,
    discountWeights: discountWeights, points: points,
    pairwiseSum: pairwiseSum, cumsum: cumsum, decimalFraction: decimalFraction, empiricalIndex: empiricalIndex,
    empiricalThreshold: empiricalThreshold, crossing: crossing, drawMasses: drawMasses,
    credibleThreshold: credibleThreshold, calibrate: calibrate, countAtOrBelow: countAtOrBelow,
    halfWidth: halfWidth, score: score, dose: dose, doseFromRatio: doseFromRatio, lowerEdgePenalty: lowerEdgePenalty,
    M: M, Mdiff: Mdiff, density: density, weightStrong: weightStrong, weightStrongExpForm: weightStrongExpForm,
    decimals: decimals, evalShown: evalShown, rel: rel, chain: chain, chainText: chainText,
    fmt: fmt, fmtPct: fmtPct, fmtInt: fmtInt
  };
})();

  // ======================================================================= the lesson: data, helpers, controls, charts
  var doc = document, W = window, html = doc.documentElement;
  var root = doc.getElementById('learn'), dataEl = doc.getElementById('lesson-data');
  if (!root || !dataEl) return;
  var D;
  try { D = TCPort.use(JSON.parse(dataEl.textContent)); } catch (err) {
    var intro0 = root.querySelector('.ls-intro');
    if (intro0) { var p0 = doc.createElement('p'); p0.className = 'note'; p0.textContent = 'The lesson\'s numbers could not be loaded, so its controls are off; every step still reads as text.'; intro0.appendChild(p0); }
    return;
  }
  var P = TCPort, MINUS = '−', NS = 'http://www.w3.org/2000/svg';
  var store = {
    get: function (k) { try { return W.localStorage.getItem(k); } catch (e) { return null; } },
    set: function (k, v) { try { W.localStorage.setItem(k, v); } catch (e) { /* storage unavailable: this visit only */ } }
  };
  function mq(q) { try { return W.matchMedia(q); } catch (e) { return { matches: false, addEventListener: function () {} }; } }
  var mqReduce = mq('(prefers-reduced-motion: reduce)'), mqPhone = mq('(max-width: 560px)');
  var uid = 0;
  function nid(p) { uid += 1; return 'lsx-' + (p || 'c') + '-' + uid; }

  // ----------------------------------------------------------------------- numbers as shown (rule D1, U+2212 minus)
  function F(v, d) { return P.fmt(v, d === undefined ? 4 : d); }
  function Fs(v, d) { return P.fmt(v, d === undefined ? 4 : d, true); }
  function FI(n) { return P.fmtInt(n).replace('-', MINUS); }
  function FPct(x, d) { return F(100 * x, d === undefined ? 1 : d) + '%'; }
  function par(s) { return s.charAt(0) === MINUS ? '(' + s + ')' : s; }
  var SUPD0 = '⁰¹²³⁴⁵⁶⁷⁸⁹';
  /** A number as a reader sees it in a tap panel: 9 significant digits, trailing zeros dropped, a power of ten
      written 8.41483679 × 10⁻⁸ when it is very small or very large, and the minus sign U+2212. */
  function raw(v) {
    if (v === Infinity) return '+∞';
    if (v === -Infinity) return MINUS + '∞';
    if (typeof v === 'boolean') return v ? 'yes' : 'no';
    if (typeof v !== 'number') return String(v);
    if (v === 0 || Number.isInteger(v) && Math.abs(v) < 1e15) return String(v).replace(/-/g, MINUS);
    var a = Math.abs(v), s;
    if (a < 1e-4 || a >= 1e9) {
      var e = Math.floor(Math.log(a) / Math.LN10), m = v / Math.pow(10, e);
      if (Math.abs(m) >= 10) { m /= 10; e += 1; } else if (Math.abs(m) < 1) { m *= 10; e -= 1; }
      s = String(parseFloat(m.toPrecision(9))) + ' × 10' + (e < 0 ? '⁻' : '') + String(Math.abs(e)).split('').map(function (c) { return SUPD0.charAt(+c); }).join('');
    } else s = String(parseFloat(v.toPrecision(9)));
    return s.replace(/-/g, MINUS);
  }
  function vec(a, d) { return '(' + a.map(function (v) { return F(v, d); }).join(', ') + ')'; }
  function clampN(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }
  function parseShown(s) {
    var t = String(s).replace(/−/g, '-').replace(/,/g, '').replace(/%/g, '').trim();
    if (!/^[-+]?(\d+(\.\d*)?|\.\d+)(e[-+]?\d+)?$/i.test(t)) return NaN;
    return parseFloat(t);
  }

  // ----------------------------------------------------------------------- DOM helpers
  function el(tag, cls, text) {
    var e = doc.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined && text !== null) e.textContent = text;
    return e;
  }
  function sv(tag, attrs, text) {
    var e = doc.createElementNS(NS, tag);
    if (attrs) Object.keys(attrs).forEach(function (k) { e.setAttribute(k, attrs[k]); });
    if (text !== undefined && text !== null) e.textContent = text;
    return e;
  }
  function add(parent) { for (var i = 1; i < arguments.length; i++) if (arguments[i]) parent.appendChild(arguments[i]); return parent; }
  function clear(e) { while (e.firstChild) e.removeChild(e.firstChild); return e; }
  function escH(s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;'); }
  /** Text with sub- and superscripts written as in the lesson's prose: J_γ, Q^π, β_emp, a_{i,k}. */
  function richHTML(s) {
    return escH(s).replace(/([A-Za-zα-ωΑ-Ω)])_(\{[^}]+\}|[A-Za-z0-9%α-ω]+)/g, function (m, a, b) { return a + '<sub>' + b.replace(/[{}]/g, '') + '</sub>'; })
      .replace(/\^(\{[^}]+\}|[A-Za-z0-9α-ωπ]+)/g, function (m, b) { return '<sup>' + b.replace(/[{}]/g, '') + '</sup>'; });
  }
  function setRich(e, s) { e.innerHTML = richHTML(s); return e; }
  function rich(tag, cls, s) { return setRich(el(tag, cls), s); }

  // ----------------------------------------------------------------------- the live region (final results only, at most every 500 ms)
  var liveEl = doc.getElementById('ls-live'), liveT = 0, livePending = null, liveLast = 0;
  function liveFlush() {
    liveT = 0;
    if (livePending === null || !liveEl) return;
    liveEl.textContent = livePending; livePending = null; liveLast = Date.now();
  }
  /** Is this element in the step on screen (so its news may be announced)? */
  function shown(e) {
    var st = e.closest && e.closest('.ls-step');
    return !!(st && !st.hasAttribute('hidden') && !st.classList.contains('nv-off') && !st.classList.contains('ls-off') && st.getClientRects().length);
  }
  function announce(s) {
    livePending = s;
    var wait = 500 - (Date.now() - liveLast);
    if (wait <= 0 && !liveT) liveFlush();
    else if (!liveT) liveT = W.setTimeout(liveFlush, Math.max(wait, 0));
  }

  // ----------------------------------------------------------------------- worked lines built live (rule D2)
  function evalForm(p) { return String(p).replace(/clip\(([^(),]+), ([^(),]+), ([^(),]+)\)/g, 'min(max($1, $2), $3)'); }
  function relations(parts) {
    try { return { rels: P.chain(parts.map(evalForm)), ok: true }; } catch (e) {
      return { rels: parts.slice(1).map(function () { return '≈'; }), ok: false };
    }
  }
  /** A worked line that controls rebuild: label, then parts joined by = or ≈ (rule D2: = only when the shown left side
      gives the shown right side). The last part is formatted from the exact value; data-exact keeps it in full. */
  function Line(name, opts) {
    var o = opts || {};
    var e = el('div', 'ls-wl ls-live');
    e.setAttribute('data-t', name);
    var q = el('span', 'ls-wq'), c = el('span', 'ls-wc'), n = el('span', 'ls-wn');
    add(e, q, c, n);
    var self = {
      el: e,
      set: function (label, parts, exact, note, sep) {
        var r = relations(parts), lead = sep === undefined ? ' = ' : sep;
        setRich(q, label + (lead === ': ' ? ':' : ''));
        clear(c);
        parts.forEach(function (p, k) {
          var w = el('span', 'ls-wp' + (k === 0 && lead !== ' = ' ? ' ls-w0' : ''));
          if (k > 0 || lead === ' = ') { add(w, el('span', 'ls-r', k === 0 ? '=' : r.rels[k - 1])); w.appendChild(doc.createTextNode(' ')); }
          w.appendChild(doc.createTextNode(p));
          if (k > 0) c.appendChild(doc.createTextNode(' '));
          c.appendChild(w);
        });
        n.textContent = note ? ' (' + note + ')' : '';
        e.setAttribute('data-parts', JSON.stringify(parts));
        e.setAttribute('data-rels', r.rels.join(''));
        if (!r.ok) e.setAttribute('data-err', '1'); else e.removeAttribute('data-err');
        if (exact !== undefined && exact !== null) e.setAttribute('data-exact', String(exact)); else e.removeAttribute('data-exact');
        e.setAttribute('data-text', (label ? label + (lead === ': ' ? ': ' : lead === ' = ' ? ' = ' : ' ') : '') + parts.map(function (p, k) { return k === 0 ? p : r.rels[k - 1] + ' ' + p; }).join(' '));
        return self;
      },
      say: function () { if (shown(e)) announce((e.getAttribute('data-text') || '').replace(/\s+/g, ' ')); }
    };
    return self;
  }
  /** A worked line that only states values. */
  function Stmt(name) {
    var e = el('div', 'ls-wl ls-ws ls-live'), t = el('span', 'ls-wt');
    e.setAttribute('data-t', name);
    e.appendChild(t);
    return { el: e, set: function (s, exact) { setRich(t, s); e.setAttribute('data-text', t.textContent); if (exact !== undefined) e.setAttribute('data-exact', String(exact)); return this; } };
  }
  function Lines(box) { var d = el('div', 'ls-lines ls-livelines'); box.appendChild(d); return d; }

  // ----------------------------------------------------------------------- inputs
  /** A slider with its value shown (and, if typed, a typed field). The value is kept exactly (a preset may hold more
      digits than the slider's step); the slider shows the nearest position. */
  function Slider(o) {
    var wrap = el('div', 'ls-sl' + (o.cls ? ' ' + o.cls : '')), id = nid('s');
    var top = el('div', 'ls-sl-top');
    var lab = el('label', 'ls-sl-l'); lab.htmlFor = id; setRich(lab, o.label);
    var out = el('output', 'ls-sl-v'); out.setAttribute('for', id);
    add(top, lab, out);
    var r = el('input', 'ls-range'); r.type = 'range'; r.id = id;
    r.min = String(o.min); r.max = String(o.max); r.step = String(o.step);
    var num = null, v = o.value;
    var fmt = o.fmt || function (x) { return F(x, 2); };
    add(wrap, top, r);
    if (o.typed) {
      num = el('input', 'ls-num'); num.type = 'text'; num.setAttribute('inputmode', 'decimal'); num.setAttribute('autocomplete', 'off');
      num.setAttribute('aria-label', (o.typedLabel || o.label.replace(/<[^>]*>/g, '')) + ', typed');
      top.insertBefore(num, out); out.hidden = true;
    }
    function show() {
      var t = fmt(v);
      out.textContent = t + (o.unit || '');
      if (num && doc.activeElement !== num) num.value = t;
      r.value = String(o.toRange ? o.toRange(v) : v);
      r.setAttribute('aria-valuetext', (o.aria ? o.aria(v) : t + (o.unit || '')));
    }
    r.addEventListener('input', function () {
      var x = parseFloat(r.value);
      v = o.fromRange ? o.fromRange(x) : x;
      show(); if (o.onInput) o.onInput(v, true);
    });
    if (o.key) r.addEventListener('keydown', function (ev) {
      var d = { ArrowRight: 1, ArrowUp: 1, ArrowLeft: -1, ArrowDown: -1 }[ev.key];
      if (!d) return;
      ev.preventDefault();
      v = clampN(Math.round((v + d * o.key) / o.step) * o.step, o.min, o.max);
      v = parseFloat(v.toFixed(10));
      show(); if (o.onInput) o.onInput(v, true);
    });
    if (num) num.addEventListener('change', function () {
      var x = parseShown(num.value);
      if (o.typedScale) x = x * o.typedScale;
      if (isNaN(x)) { show(); return; }
      v = clampN(x, o.typedMin === undefined ? o.min : o.typedMin, o.typedMax === undefined ? o.max : o.typedMax);
      num.value = fmt(v); show(); if (o.onInput) o.onInput(v, true);
    });
    show();
    return { el: wrap, input: r, field: num, get: function () { return v; }, set: function (x) { v = x; show(); } };
  }
  /** A row of buttons, one pressed (aria-pressed). */
  function Seg(o) {
    var wrap = el('div', 'ls-seg' + (o.cls ? ' ' + o.cls : '')), lab = o.label ? rich('span', 'ls-seg-l', o.label) : null;
    var row = el('div', 'ls-seg-b'); row.setAttribute('role', 'group');
    if (o.label) row.setAttribute('aria-label', o.label.replace(/[_^{}]/g, ''));
    var cur = o.value, btns = [];
    o.options.forEach(function (op) {
      var b = el('button', 'ls-sb'); b.type = 'button'; setRich(b, op.t); b.setAttribute('data-k', String(op.k));
      b.setAttribute('aria-pressed', String(op.k) === String(cur) ? 'true' : 'false');
      b.addEventListener('click', function () { set(op.k); if (o.onChange) o.onChange(op.k, true); });
      btns.push(b); row.appendChild(b);
    });
    function set(k) { cur = k; btns.forEach(function (b) { b.setAttribute('aria-pressed', b.getAttribute('data-k') === String(k) ? 'true' : 'false'); }); }
    add(wrap, lab, row);
    return { el: wrap, get: function () { return cur; }, set: set, buttons: btns };
  }
  function Btn(text, fn, cls) {
    var b = el('button', 'btn ghost ls-b' + (cls ? ' ' + cls : '')); b.type = 'button'; setRich(b, text);
    b.addEventListener('click', function (ev) { fn(ev); });
    return b;
  }
  function Switch(label, on, fn) {
    var b = el('button', 'ls-switch'); b.type = 'button'; b.setAttribute('role', 'switch');
    var tr = el('span', 'ls-track'); tr.setAttribute('aria-hidden', 'true'); tr.appendChild(el('span', 'ls-knob'));
    add(b, tr, rich('span', 'ls-sw-t', label));
    var v = !!on;
    function show() { b.setAttribute('aria-checked', v ? 'true' : 'false'); }
    b.addEventListener('click', function () { v = !v; show(); fn(v, true); });
    show();
    return { el: b, get: function () { return v; }, set: function (x) { v = !!x; show(); } };
  }
  function Row() { var r = el('div', 'ls-row'); for (var i = 0; i < arguments.length; i++) if (arguments[i]) r.appendChild(arguments[i].el || arguments[i]); return r; }
  /** A readout: a small name and a value (data-exact keeps the full value). */
  function Ro(k, name) {
    var e = el('div', 'ro ls-ro'), kk = rich('span', 'k', k), v = el('span', 'v');
    if (name) e.setAttribute('data-t', name);
    add(e, kk, v);
    return { el: e, set: function (t, exact) { v.textContent = t; if (exact !== undefined) e.setAttribute('data-exact', String(exact)); return this; }, key: kk };
  }
  function Ros() { var d = el('div', 'readouts ls-ros'); for (var i = 0; i < arguments.length; i++) d.appendChild(arguments[i].el); return d; }
  function Note(text, cls) { return rich('p', 'note ls-cnote' + (cls ? ' ' + cls : ''), text); }
  /** A horizontal bar from 0 (HTML), for two or three quantities on one scale. */
  function Bars(items) {
    var d = el('div', 'ls-bars'), rows = {};
    items.forEach(function (it) {
      var r = el('div', 'ls-bar'), l = rich('span', 'ls-bar-l', it.label), tr = el('span', 'ls-bar-tr'), f = el('span', 'ls-bar-f'), v = el('span', 'ls-bar-v');
      f.style.background = 'var(' + it.color + ')';
      tr.appendChild(f); add(r, l, tr, v); d.appendChild(r);
      if (it.name) r.setAttribute('data-t', it.name);
      rows[it.key] = { f: f, v: v, r: r };
    });
    return {
      el: d,
      set: function (vals, scale, mark) {
        Object.keys(vals).forEach(function (k) {
          var x = vals[k], w = scale > 0 ? Math.min(Math.abs(x.v) / scale, 1) : 0;
          rows[k].f.style.width = (100 * w).toFixed(2) + '%';
          rows[k].f.classList.toggle('neg', x.v < 0);
          rows[k].v.textContent = x.t;
          rows[k].r.setAttribute('data-exact', String(x.v));
        });
        if (mark !== undefined) d.style.setProperty('--ls-mark', (100 * mark / scale).toFixed(2) + '%');
      }
    };
  }

  // ----------------------------------------------------------------------- tables made by controls (with the page's scroll cue)
  function Table(cols, opts) {
    var o = opts || {};
    var wrap = el('div', 'tblwrap'), hint = el('div', 'hint'), box = el('div', 'tbl' + (o.cls ? ' ' + o.cls : '')), fade = el('div', 'fade');
    hint.setAttribute('aria-hidden', 'true'); fade.setAttribute('aria-hidden', 'true');
    var t = el('table', 'ls-t'), th = el('thead'), tr = el('tr'), tb = el('tbody');
    cols.forEach(function (c) { tr.appendChild(rich('th', null, c)); });
    add(th, tr); add(t, th, tb); box.appendChild(t); add(wrap, hint, box, fade);
    function cue() {
      var over = box.scrollWidth > box.clientWidth + 1;
      wrap.classList.toggle('scrolls', over);
      wrap.classList.toggle('at-end', box.scrollLeft + box.clientWidth >= box.scrollWidth - 2);
      if (over) { fade.style.top = (box.offsetTop + 1) + 'px'; fade.style.height = Math.max(0, box.offsetHeight - 2) + 'px'; }
    }
    box.addEventListener('scroll', cue, { passive: true });
    W.addEventListener('resize', cue);
    return {
      el: wrap, body: tb, cue: cue,
      rows: function (rows) {
        clear(tb);
        rows.forEach(function (r) {
          var x = el('tr');
          if (r.cls) x.className = r.cls;
          r.cells.forEach(function (c, k) {
            var td = el(k === 0 && o.rowHead ? 'th' : 'td');
            if (k === 0 && o.rowHead) td.setAttribute('scope', 'row');
            if (typeof c === 'object' && c !== null) { setRich(td, c.t); if (c.cls) td.className = c.cls; if (c.exact !== undefined) td.setAttribute('data-exact', String(c.exact)); if (c.title) td.title = c.title; }
            else { setRich(td, String(c)); if (/^[−-]?\d[\d,]*(\.\d+)?%?$/.test(String(c))) td.className = 'n'; }
            x.appendChild(td);
          });
          tb.appendChild(x);
        });
        if (W.requestAnimationFrame) W.requestAnimationFrame(cue); else cue();
      }
    };
  }

  // ----------------------------------------------------------------------- charts: inline SVG sized from their box
  var charts = [];
  function swatch(color, dash) {
    var s = el('i', 'sw ls-swt');
    s.style.background = dash ? 'repeating-linear-gradient(90deg, var(' + color + ') 0 4px, transparent 4px 7px)' : 'var(' + color + ')';
    return s;
  }
  function niceTicks(lo, hi, n) {
    var span = hi - lo, step = Math.pow(10, Math.floor(Math.log(span / n) / Math.LN10)), err = n / span * step;
    if (err <= 0.15) step *= 10; else if (err <= 0.35) step *= 5; else if (err <= 0.75) step *= 2;
    var out = [], t0 = Math.ceil(lo / step - 1e-9) * step;
    for (var t = t0; t <= hi + step * 1e-9; t += step) out.push(Math.abs(t) < step * 1e-9 ? 0 : t);
    return out;
  }
  function tickText(v, d) { var s = F(v, d === undefined ? 2 : d); if (s.indexOf('.') >= 0) s = s.replace(/0+$/, '').replace(/\.$/, ''); return s; }
  /** Chart(cfg): cfg = { title (aria), height, x: {min, max, log, ticks, fmt, label}, y: {min, max, ticks, fmt, label},
      series: [{ key, name, color: '--token', dash, width, pts: [[x, y]], kind: 'line' | 'bars' | 'dots', label: true }],
      vlines: [{ x, color, dash, label }], hlines: [{ y, color, dash, label }], marks: [{ x, y, color, label, r }],
      hover: { xs: [x...], title(x), rows(x) -> [{ color, dash, name, value }] }, legend: true }. Redraws to its box's
      width; hover or tap shows every line's value with its color swatch; arrow keys move along x when focused. */
  function Chart(box, cfg) {
    var fig = el('div', 'chartbox ls-chartbox'), hold = el('div', 'ls-ch'), tip = el('div', 'ls-tip'), leg = el('div', 'legend ls-legend');
    tip.hidden = true; tip.setAttribute('aria-hidden', 'true');
    add(fig, hold, leg); hold.appendChild(tip); box.appendChild(fig);   // the readout sits over the plot (hold is positioned)
    var C = cfg, lastW = 0, hov = -1, geo = null, svgEl = null;
    function sx(x) { var a = C.x; if (a.log) return geo.l + (Math.log(x) - Math.log(a.min)) / (Math.log(a.max) - Math.log(a.min)) * geo.w; return geo.l + (x - a.min) / (a.max - a.min) * geo.w; }
    function sy(y) { return geo.t + (1 - (y - C.y.min) / (C.y.max - C.y.min)) * geo.h; }
    function legend() {
      clear(leg);
      if (C.legend === false) return;
      C.series.forEach(function (s) { if (!s.name) return; var sp = el('span'); add(sp, swatch(s.color, s.dash)); sp.appendChild(rich('span', null, s.name)); leg.appendChild(sp); });
      (C.legendExtra || []).forEach(function (s) { var sp = el('span'); add(sp, swatch(s.color, s.dash)); sp.appendChild(rich('span', null, s.name)); leg.appendChild(sp); });
    }
    function draw() {
      var Wd = Math.floor(hold.clientWidth || 0);
      if (Wd < 40) return false;
      lastW = Wd;
      var Ht = C.height || 220, ml = C.ml || 50, mr = C.mr || 14, mt = C.y.label ? 26 : 12, mb = C.x.label ? 42 : 26;
      geo = { l: ml, t: mt, w: Wd - ml - mr, h: Ht - mt - mb };
      var s = sv('svg', { 'class': 'chart ls-svg', width: Wd, height: Ht, viewBox: '0 0 ' + Wd + ' ' + Ht, role: 'img', 'aria-label': C.title || '', tabindex: C.hover ? '0' : '-1', focusable: C.hover ? 'true' : 'false' });
      var yt = C.y.ticks || niceTicks(C.y.min, C.y.max, Ht < 180 ? 3 : 5);
      yt.forEach(function (v) {
        if (v < C.y.min - 1e-12 || v > C.y.max + 1e-12) return;
        add(s, sv('line', { 'class': 'grid', x1: geo.l, x2: geo.l + geo.w, y1: sy(v), y2: sy(v) }));
        add(s, sv('text', { 'class': 'tick', x: geo.l - 6, y: sy(v) + 4, 'text-anchor': 'end' }, (C.y.fmt || tickText)(v)));
      });
      var xt = C.x.ticks || niceTicks(C.x.min, C.x.max, Math.max(2, Math.min(8, Math.floor(geo.w / 70))));
      var lastX = -Infinity;
      xt.forEach(function (v) {        // a tick label is drawn only where it does not crowd the one before it
        if (v < C.x.min - 1e-12 || v > C.x.max + 1e-12) return;
        var t = (C.x.fmt || tickText)(v), X = sx(v), half = t.length * 3.9 + 4;
        var tx = Math.max(geo.l - 2 + half, Math.min(X, Wd - 2 - half));   // moved in at the ends: not cut off, clear of the y labels
        if (tx - half < lastX) return;
        lastX = tx + half;
        add(s, sv('text', { 'class': 'tick', x: tx, y: geo.t + geo.h + 16, 'text-anchor': 'middle' }, t));
      });
      add(s, sv('line', { 'class': 'axis', x1: geo.l, x2: geo.l + geo.w, y1: geo.t + geo.h, y2: geo.t + geo.h }));
      add(s, sv('line', { 'class': 'axis', x1: geo.l, x2: geo.l, y1: geo.t, y2: geo.t + geo.h }));
      if (C.x.label) add(s, sv('text', { 'class': 'axlab', x: geo.l + geo.w / 2, y: Ht - 6, 'text-anchor': 'middle' }, C.x.label));
      if (C.y.label) add(s, sv('text', { 'class': 'axlab', x: 4, y: 14, 'text-anchor': 'start' }, C.y.label));
      (C.bands || []).forEach(function (b) {
        add(s, sv('rect', { x: Math.min(sx(b.x0), sx(b.x1)), width: Math.abs(sx(b.x1) - sx(b.x0)), y: geo.t, height: geo.h, style: 'fill: var(' + b.color + ')', opacity: b.opacity || 1 }));
      });
      (C.hlines || []).forEach(function (L) {
        if (L.y < C.y.min || L.y > C.y.max) return;
        add(s, sv('line', { x1: geo.l, x2: geo.l + geo.w, y1: sy(L.y), y2: sy(L.y), style: 'stroke: var(' + (L.color || '--muted') + ')', 'stroke-width': L.width || 1.3, 'stroke-dasharray': L.dash === false ? 'none' : '5 4' }));
        if (L.label) add(s, sv('text', { 'class': 'ann', x: L.anchor === 'start' ? geo.l + 4 : geo.l + geo.w - 2, y: sy(L.y) - 4, 'text-anchor': L.anchor || 'end' }, L.label));
      });
      (C.vlines || []).forEach(function (L) {
        if (L.x < C.x.min || L.x > C.x.max) return;
        add(s, sv('line', { x1: sx(L.x), x2: sx(L.x), y1: geo.t, y2: geo.t + geo.h, style: 'stroke: var(' + (L.color || '--muted') + ')', 'stroke-width': L.width || 1.3, 'stroke-dasharray': L.dash === false ? 'none' : '5 4' }));
        if (L.label) add(s, sv('text', { 'class': 'ann', x: sx(L.x) + (L.anchor === 'end' ? -4 : 4), y: L.bottom ? geo.t + geo.h - 5 : geo.t + 11 + (L.dy || 0), 'text-anchor': L.anchor || 'start' }, L.label));
      });
      var ends = [];
      C.series.forEach(function (S) {
        var g = sv('g', { 'data-series': S.key || S.name || '' });
        if (S.kind === 'bars') {
          var bw = S.barW ? Math.abs(sx(S.barW) - sx(0)) : Math.max(1, geo.w / Math.max(S.pts.length, 1) * 0.8);
          S.pts.forEach(function (p) {
            var y0 = sy(Math.max(C.y.min, Math.min(0, C.y.max))), y1 = sy(p[1]);
            add(g, sv('rect', { x: sx(p[0]) - bw / 2, width: Math.max(bw, 0.8), y: Math.min(y0, y1), height: Math.max(Math.abs(y1 - y0), p[1] !== 0 ? 0.8 : 0), style: 'fill: var(' + (p[2] || S.color) + ')' }));
          });
        } else if (S.kind === 'dots') {
          S.pts.forEach(function (p) { add(g, sv('circle', { cx: sx(p[0]), cy: sy(p[1]), r: S.r || 2, style: 'fill: var(' + S.color + ')' })); });
        } else {
          var d = '', pen = false;
          S.pts.forEach(function (p) {
            if (p[1] === null || !isFinite(p[1])) { pen = false; return; }
            var X = sx(p[0]), Y = sy(Math.max(C.y.min - (C.y.max - C.y.min) * 0.05, Math.min(C.y.max + (C.y.max - C.y.min) * 0.05, p[1])));
            d += (pen ? 'L' : 'M') + X.toFixed(1) + ' ' + Y.toFixed(1); pen = true;
          });
          add(g, sv('path', { d: d, style: 'stroke: var(' + S.color + ')', 'stroke-width': S.width || 2.4, 'stroke-dasharray': S.dash ? '7 5' : 'none', fill: 'none' }));
          var last = null;
          for (var i = S.pts.length - 1; i >= 0; i--) if (S.pts[i][1] !== null && isFinite(S.pts[i][1])) { last = S.pts[i]; break; }
          if (last && S.label !== false && S.name) {
            var lx = sx(last[0]) - 60, ly = null, lb = Infinity;      // the line's height about 60 px left of its end, under its label
            S.pts.forEach(function (p) { if (p[1] === null || !isFinite(p[1])) return; var dd = Math.abs(sx(p[0]) - lx); if (dd < lb) { lb = dd; ly = sy(p[1]); } });
            ends.push({ y: sy(last[1]), x: sx(last[0]), t: S.short || S.name, color: S.color, dash: !!S.dash, falls: ly !== null && ly < sy(last[1]) - 6 });
          }
        }
        g.setAttribute('data-n', String(S.pts.length));
        if (S.keep) g.setAttribute('data-pts', JSON.stringify(S.pts.map(function (p) { return [p[0], p[1]]; })));
        s.appendChild(g);
      });
      if (C.endLabels && geo.w >= 420) {      // direct labels at the right end, nudged apart (the legend names them on a phone)
        ends.sort(function (a, b) { return a.y - b.y; });
        for (var k = 1; k < ends.length; k++) if (ends[k].y - ends[k - 1].y < 14) ends[k].y = ends[k - 1].y + 14;
        ends.forEach(function (e) {
          e.text = sv('text', { 'class': 'lab ls-endlab', x: e.x - 4, y: e.y - 6, 'text-anchor': 'end' }, e.t);
          e.key = sv('line', { 'class': 'ls-endkey', x1: 0, x2: 0, y1: e.y - 10, y2: e.y - 10, style: 'stroke: var(' + e.color + ')', 'stroke-dasharray': e.dash ? '5 3' : 'none' });
          add(s, e.key, e.text);
        });
      }
      (C.marks || []).forEach(function (m) {
        if (m.x < C.x.min || m.x > C.x.max || m.y < C.y.min || m.y > C.y.max) return;
        add(s, sv('circle', { cx: sx(m.x), cy: sy(m.y), r: m.r || 5, style: 'fill: var(' + (m.color || '--ink') + '); stroke: var(--surface)', 'stroke-width': 2, 'data-mark': m.key || '' }));
        if (m.label) add(s, sv('text', { 'class': 'ann ls-marklab', x: sx(m.x) + (m.anchor === 'end' ? -8 : 8), y: sy(m.y) + (m.dy === undefined ? -8 : m.dy), 'text-anchor': m.anchor || 'start' }, m.label));
      });
      var guide = sv('line', { 'class': 'ls-guide', x1: 0, x2: 0, y1: geo.t, y2: geo.t + geo.h, visibility: 'hidden' });
      s.appendChild(guide);
      if (C.hover) {
        var hit = sv('rect', { x: geo.l, y: geo.t, width: geo.w, height: geo.h, fill: 'transparent', 'class': 'ls-hit' });
        s.appendChild(hit);
        var pick = function (ev) {
          var r = s.getBoundingClientRect(), px = ev.clientX - r.left, best = -1, bd = Infinity;
          C.hover.xs.forEach(function (x, k) { var dd = Math.abs(sx(x) - px); if (dd < bd) { bd = dd; best = k; } });
          if (best >= 0) showTip(best);
        };
        hit.addEventListener('pointermove', pick);
        hit.addEventListener('pointerdown', pick);
        s.addEventListener('pointerleave', function (ev) { if (ev.pointerType === 'mouse') hideTip(); });
        s.addEventListener('keydown', function (ev) {
          var n = C.hover.xs.length;
          if (ev.key === 'ArrowRight' || ev.key === 'ArrowLeft') { ev.preventDefault(); showTip(clampN((hov < 0 ? (ev.key === 'ArrowRight' ? -1 : n) : hov) + (ev.key === 'ArrowRight' ? 1 : -1), 0, n - 1)); }
          else if (ev.key === 'Home') { ev.preventDefault(); showTip(0); } else if (ev.key === 'End') { ev.preventDefault(); showTip(n - 1); }
          else if (ev.key === 'Escape') hideTip();
        });
        s.addEventListener('blur', hideTip);
      }
      if (svgEl) hold.replaceChild(s, svgEl); else hold.insertBefore(s, tip);
      svgEl = s;
      // each end label's key, placed just before its text once the text can be measured; a label that a dot (a mark, such
      // as the current command) or an earlier end label would cover moves to the free side of its line's end
      var dots = (C.marks || []).filter(function (m) { return !(m.x < C.x.min || m.x > C.x.max || m.y < C.y.min || m.y > C.y.max); })
        .map(function (m) { return { x: sx(m.x), y: sy(m.y), r: (m.r || 5) + 2.5 }; });     // the dot, its ring and the text's halo
      var placed = [];
      ends.forEach(function (e) {
        if (!e.key) return;
        var w = 0;
        try { w = e.text.getComputedTextLength(); } catch (err) { w = 0; }
        if (!(w > 0)) w = e.t.length * 6.6;
        var clash = function (b, x1) {         // b: the text's baseline, x1: its right end; the text and its key before it, in a box
          var x0 = x1 - w - 24, t = b - 10, u = b + 3, i, d, cx, cy;     // 13 px tall, under the 14 px kept between labels
          for (i = 0; i < dots.length; i++) {
            d = dots[i]; cx = Math.max(x0, Math.min(d.x, x1)); cy = Math.max(t, Math.min(d.y, u));
            if ((cx - d.x) * (cx - d.x) + (cy - d.y) * (cy - d.y) < d.r * d.r) return true;
          }
          for (i = 0; i < placed.length; i++) if (x0 < placed[i].x1 && placed[i].x0 < x1 && t < placed[i].u && placed[i].t < u) return true;
          return false;
        };
        var right = e.x - 4, base = e.y - 6;
        var hs = [base].concat((e.falls ? [e.y + 18, e.y - 20, e.y + 32, e.y - 34] : [e.y - 20, e.y + 18, e.y - 34, e.y + 32])
          .filter(function (b) { return b - 12 >= geo.t - 4 && b + 4 <= geo.t + geo.h; }));
        if (clash(base, right) && !hs.some(function (b) { if (clash(b, right)) return false; base = b; return true; })) {
          hs.some(function (b) {               // no height is free: the first height where moving left of the dots frees the label
            var x1 = right, moved = true;
            while (moved) {                    // left of every dot that meets the label's span at this height
              moved = false;
              dots.forEach(function (d) { if (d.y + d.r > b - 10 && d.y - d.r < b + 3 && d.x + d.r > x1 - w - 24 && d.x - d.r < x1) { x1 = d.x - d.r - 2; moved = true; } });
            }
            if (x1 - w - 24 < geo.l || clash(b, x1)) return false;
            base = b; right = x1; return true;
          });
        }
        e.text.setAttribute('x', right.toFixed(1)); e.text.setAttribute('y', base);
        e.key.setAttribute('y1', base - 4); e.key.setAttribute('y2', base - 4);
        placed.push({ x0: right - w - 24, x1: right, t: base - 10, u: base + 3 });
        var x2 = right - w - 4;
        e.key.setAttribute('x1', (x2 - 16).toFixed(1)); e.key.setAttribute('x2', x2.toFixed(1));
      });
      if (hov >= 0 && C.hover && hov < C.hover.xs.length) showTip(hov); else hideTip();
      return true;
    }
    function showTip(k) {
      if (!C.hover || !geo) return;
      hov = k;
      var x = C.hover.xs[k], X = sx(x), g = svgEl.querySelector('.ls-guide');
      g.setAttribute('x1', X); g.setAttribute('x2', X); g.setAttribute('visibility', 'visible');
      clear(tip);
      tip.appendChild(rich('p', 'ls-tip-h', C.hover.title(x, k)));
      C.hover.rows(x, k).forEach(function (r) {
        var p = el('p', 'ls-tip-r');
        add(p, r.color ? swatch(r.color, r.dash) : el('i', 'ls-swt ls-swt-none'), rich('span', 'ls-tip-n', r.name), el('b', 'ls-tip-v', r.value));
        tip.appendChild(p);
      });
      tip.hidden = false;
      var bw = hold.clientWidth, tw = Math.min(tip.offsetWidth || 180, bw);
      tip.style.left = Math.max(0, Math.min(bw - tw, X + 12 > bw - tw ? X - tw - 12 : X + 12)) + 'px';
      if (C.onHover) C.onHover(x, k);
    }
    function hideTip() { hov = -1; tip.hidden = true; if (svgEl) { var g = svgEl.querySelector('.ls-guide'); if (g) g.setAttribute('visibility', 'hidden'); } }
    legend();
    var self = {
      el: fig, svg: function () { return svgEl; },
      update: function (cfg2) { C = cfg2; legend(); draw(); },
      draw: draw, tip: showTip, hide: hideTip,
      sync: function () { if (Math.floor(hold.clientWidth || 0) !== lastW) draw(); }
    };
    if (W.ResizeObserver) { var ro = new W.ResizeObserver(function () { self.sync(); }); ro.observe(hold); }
    charts.push(self);
    return self;
  }
  W.addEventListener('resize', function () { charts.forEach(function (c) { c.sync(); }); });

  // ----------------------------------------------------------------------- a small square map: a float's point (p across, v up)
  function MapView(box, o) {
    var size = 220, pad = 22, inner = size - 2 * pad;
    var s = sv('svg', { 'class': 'ls-map', viewBox: '0 0 ' + size + ' ' + size, width: size, height: size, role: 'img', 'aria-label': o.title });
    function X(p) { return pad + (p + 1) / 2 * inner; }
    function Y(v) { return pad + (1 - (v + 1) / 2) * inner; }
    for (var t = -1; t <= 1.0001; t += 0.5) {
      add(s, sv('line', { 'class': t === 0 ? 'ls-map-ax' : 'ls-map-grid', x1: X(t), x2: X(t), y1: Y(-1), y2: Y(1) }));
      add(s, sv('line', { 'class': t === 0 ? 'ls-map-ax' : 'ls-map-grid', x1: X(-1), x2: X(1), y1: Y(t), y2: Y(t) }));
    }
    add(s, sv('text', { 'class': 'ls-map-t', x: X(1) - 2, y: Y(0) - 5, 'text-anchor': 'end' }, 'p'));
    add(s, sv('text', { 'class': 'ls-map-t', x: X(0) + 5, y: Y(1) + 10 }, 'v'));
    add(s, sv('text', { 'class': 'ls-map-t', x: X(-1), y: size - 5 }, MINUS + '1'));
    add(s, sv('text', { 'class': 'ls-map-t', x: X(1), y: size - 5, 'text-anchor': 'end' }, '1'));
    var trail = sv('path', { 'class': 'ls-map-trail', d: '' }), ring = sv('circle', { 'class': 'ls-map-ring', cx: X(0), cy: Y(0), r: 0 });
    var dot = sv('circle', { 'class': 'ls-map-dot', r: 7 });
    add(s, trail, ring, dot);
    box.appendChild(s);
    if (o.drag) {
      var dragging = false;
      var at = function (ev) {
        var r = s.getBoundingClientRect(), k = size / r.width;
        var p = ((ev.clientX - r.left) * k - pad) / inner * 2 - 1, v = 1 - ((ev.clientY - r.top) * k - pad) / inner * 2;
        o.drag(clampN(Math.round(p * 10000) / 10000, -1, 1), clampN(Math.round(v * 10000) / 10000, -1, 1));
      };
      s.addEventListener('pointerdown', function (ev) { dragging = true; try { s.setPointerCapture(ev.pointerId); } catch (e) { /* old browser */ } at(ev); ev.preventDefault(); });
      s.addEventListener('pointermove', function (ev) { if (dragging) at(ev); });
      s.addEventListener('pointerup', function () { dragging = false; });
      s.addEventListener('pointercancel', function () { dragging = false; });
      s.style.touchAction = 'none';
    }
    return {
      el: s,
      set: function (p, v, trailPts) {
        dot.setAttribute('cx', X(clampN(p, -1.2, 1.2))); dot.setAttribute('cy', Y(clampN(v, -1.2, 1.2)));
        var d = Math.sqrt(p * p + v * v);
        ring.setAttribute('r', (d / 2 * inner).toFixed(2));
        trail.setAttribute('d', (trailPts || []).map(function (q, k) { return (k ? 'L' : 'M') + X(q[0]).toFixed(1) + ' ' + Y(q[1]).toFixed(1); }).join(''));
      }
    };
  }

  // ----------------------------------------------------------------------- the channel drawn small (five floats, four thrusters, the water)
  /** ChannelView(box, o): o.parts (buttons for step 1.1), o.title. set({ ys: five float offsets (−1..1), thr: four
      throttles, stall: true/false (thrusters past 80% in the stall color), push: c + w1, arrows: five speed changes }). */
  function ChannelView(box, o) {
    var Wd = 360, Ht = o.height || 190;
    var s = sv('svg', { 'class': 'ls-chan', viewBox: '0 0 ' + Wd + ' ' + Ht, role: o.parts ? 'group' : 'img', 'aria-label': o.title });
    var FX = [36, 108, 180, 252, 324], TX = [72, 144, 216, 288], FY = o.floatY || 82, TY = FY + 44;
    add(s, sv('rect', { 'class': 'ls-ch-water', x: 0, y: 40, width: Wd, height: Ht - 64, rx: 8 }));
    var wave = sv('path', { 'class': 'ls-ch-wave', d: '' });
    s.appendChild(wave);
    var rest = sv('line', { 'class': 'ls-ch-rest', x1: 8, x2: Wd - 8, y1: FY, y2: FY });
    s.appendChild(rest);
    var ties = [], floats = [], thr = [], gauges = [], arrows = [];
    for (var k = 0; k < 4; k++) {
      ties.push(sv('path', { 'class': 'ls-ch-tie', d: '' }));
      s.appendChild(ties[k]);
    }
    for (var j = 0; j < 5; j++) {
      var f = sv('circle', { 'class': 'ls-ch-float', cx: FX[j], cy: FY, r: 13 });
      floats.push(f); s.appendChild(f);
      add(s, sv('text', { 'class': 'ls-ch-lab', x: FX[j], y: FY + 4, 'text-anchor': 'middle' }, String(j + 1)));
      var ar = sv('path', { 'class': 'ls-ch-arrow', d: '' }); arrows.push(ar); s.appendChild(ar);
    }
    for (k = 0; k < 4; k++) {
      var g = sv('g', { 'class': 'ls-ch-thr' });
      add(g, sv('rect', { 'class': 'ls-ch-gauge', x: TX[k] - 16, y: TY - 7, width: 32, height: 14, rx: 3 }));
      var fill = sv('rect', { 'class': 'ls-ch-gfill', x: TX[k], y: TY - 5, width: 0, height: 10, rx: 2 });
      gauges.push(fill); g.appendChild(fill);
      add(g, sv('line', { 'class': 'ls-ch-g80', x1: TX[k] - 12.8, x2: TX[k] - 12.8, y1: TY - 9, y2: TY + 9 }));
      add(g, sv('line', { 'class': 'ls-ch-g80', x1: TX[k] + 12.8, x2: TX[k] + 12.8, y1: TY - 9, y2: TY + 9 }));
      add(g, sv('text', { 'class': 'ls-ch-lab ls-ch-tl', x: TX[k], y: TY + 22, 'text-anchor': 'middle' }, 'T' + (k + 1)));
      thr.push(g); s.appendChild(g);
    }
    var cur = sv('path', { 'class': 'ls-ch-cur', d: '' });
    var curT = sv('text', { 'class': 'ls-ch-curt', x: 14, y: Ht - 30 }, 'current');
    add(s, cur, curT);
    var hits = [];
    if (o.parts) {
      var mk = function (x, y, w, h, label, key) {
        var hb = sv('rect', { 'class': 'ls-hitbox', x: x, y: y, width: w, height: h, rx: 6, tabindex: '0', role: 'button', 'aria-label': label, 'data-part': key, 'aria-pressed': 'false' });
        var fire = function () { o.parts(key); };
        hb.addEventListener('click', fire);
        hb.addEventListener('keydown', function (ev) { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); fire(); } });
        hits.push(hb); s.appendChild(hb);
      };
      mk(0, 0, Wd, 50, 'the wave', 'wave');
      for (j = 0; j < 5; j++) mk(FX[j] - 26, FY - 30, 52, 52, 'float ' + (j + 1), 'f' + j);
      for (k = 0; k < 4; k++) mk(TX[k] - 24, TY - 22, 48, 48, 'thruster ' + (k + 1), 't' + k);
      mk(0, Ht - 49, Wd, 49, 'the current', 'current');
    }
    box.appendChild(s);
    var self = {
      el: s,
      set: function (st) {
        var ys = st.ys || [0, 0, 0, 0, 0], amp = 24;
        var Y = ys.map(function (v) { return FY - clampN(v, -1.6, 1.6) * amp * 0.6; });
        floats.forEach(function (f, i) { f.setAttribute('cy', Y[i].toFixed(1)); });
        s.querySelectorAll('.ls-ch-lab').forEach(function (t, i) { if (i < 5) t.setAttribute('y', (Y[i] + 4).toFixed(1)); });
        ties.forEach(function (t, i) { t.setAttribute('d', 'M' + (FX[i] + 13) + ' ' + Y[i].toFixed(1) + 'L' + TX[i] + ' ' + (TY - 7) + 'L' + (FX[i + 1] - 13) + ' ' + Y[i + 1].toFixed(1)); });
        var th = st.thr || [0, 0, 0, 0];
        gauges.forEach(function (g, i) {
          var a = clampN(th[i], -1, 1), w = Math.abs(a) * 16;
          g.setAttribute('x', (a < 0 ? TX[i] - w : TX[i]).toFixed(1)); g.setAttribute('width', w.toFixed(1));
          var past = Math.abs(a) > 0.800000011920929;
          g.setAttribute('class', 'ls-ch-gfill' + (past && st.stall !== false ? ' past' : (a !== 0 ? ' on' : '')));
        });
        var push = st.push === undefined ? 0.5 : st.push, L = 40 + 200 * clampN(push / 1.6, 0, 1);
        cur.setAttribute('d', st.water === false ? '' : 'M14 ' + (Ht - 16) + 'h' + L.toFixed(1) + 'm-8 -6l8 6l-8 6');
        curT.textContent = st.water === false ? '' : (st.curLabel || ('water ' + F(push, 2)));
        var A = st.waveAmp === undefined ? 0.23 : st.waveAmp, ph = st.wavePhase || 0, d = '';
        for (var x = 0; x <= Wd; x += 12) d += (x ? 'L' : 'M') + x + ' ' + (22 - 14 * A / 0.7 * Math.sin(x / 60 + ph)).toFixed(1);
        wave.setAttribute('d', d);
        arrows.forEach(function (a, i) {
          var v = st.arrows ? st.arrows[i] : 0, len = clampN(v, -1, 1) * 32;   // at most 32 px, centered on the float
          if (!st.arrows || Math.abs(len) < 0.5) { a.setAttribute('d', ''); return; }
          var y = Y[i] - 22, x0 = FX[i] - len / 2, x1 = FX[i] + len / 2, h = len > 0 ? -5 : 5;
          a.setAttribute('d', 'M' + x0 + ' ' + y + 'L' + x1.toFixed(1) + ' ' + y + 'm' + h + ' -4l' + (-h) + ' 4l' + h + ' 4');
          a.setAttribute('class', 'ls-ch-arrow ' + (v < 0 ? 'up' : 'down'));
        });
      },
      press: function (key) { hits.forEach(function (h) { h.setAttribute('aria-pressed', h.getAttribute('data-part') === key ? 'true' : 'false'); }); }
    };
    return self;
  }

  /** The key under a channel drawing whose gauges can show the stall color (fix round 1, finding L17). */
  function StallKey() {
    var p = el('p', 'note ls-chkey');
    add(p, swatch('--k-stall'), doc.createTextNode(' a thruster gauge in this color: past 80%, so it stalls'));
    return p;
  }

  // ----------------------------------------------------------------------- the registry of controls, and visibility
  var CTL = {}, inst = {};
  function stepOf(id) { return doc.getElementById('learn-' + id); }
  function control(id) {
    if (inst[id]) return inst[id];
    var st = stepOf(id), box = st && st.querySelector('.ls-try[data-ctl="' + id + '"] .ls-ctl');
    if (!box || !CTL[id]) { inst[id] = {}; return inst[id]; }
    clear(box);
    try { inst[id] = CTL[id](box, st) || {}; } catch (e) {
      clear(box); box.appendChild(el('p', 'note', 'This control could not start in this browser.'));
      inst[id] = { failed: String(e && e.message || e) };
      box.setAttribute('data-fail', inst[id].failed);
    }
    return inst[id];
  }
  var BUILT = null;     // set by the wiring below: prepares the new worked lines of a step whose control was just built
  function enterStep(st) {
    if (!st || !st.id || st.id.indexOf('learn-') !== 0) return;
    var fresh = !inst[st.id.slice(6)], c = control(st.id.slice(6));
    if (fresh && BUILT) BUILT(st);
    if (c.enter) { try { c.enter(); } catch (e) { /* drawing failed: the numbers stay */ } }
    charts.forEach(function (ch) { if (st.contains(ch.el)) ch.sync(); });
  }
  function leaveStep(st) {
    if (!st || !st.id || st.id.indexOf('learn-') !== 0) return;
    var c = inst[st.id.slice(6)];
    if (c && c.leave) { try { c.leave(); } catch (e) { /* nothing to stop */ } }
  }
  /** A Play loop: one tick every ms; stops on leave, on any input elsewhere in the step, and with reduced motion it
      still steps (each tick is a jump, nothing slides). */
  function Player(btn, ms, tick) {
    var t = 0;
    function stop() { if (t) { W.clearInterval(t); t = 0; } btn.textContent = 'Play'; btn.setAttribute('aria-pressed', 'false'); }
    function start() { if (t) return; btn.textContent = 'Pause'; btn.setAttribute('aria-pressed', 'true'); t = W.setInterval(function () { if (tick() === false) stop(); }, ms); }
    btn.addEventListener('click', function () { if (t) stop(); else start(); });
    btn.setAttribute('aria-pressed', 'false');
    return { stop: stop, start: start, on: function () { return !!t; } };
  }

  // ======================================================================= chapter 1: the channel, one step at a time
  var SUBD = '₀₁₂₃₄₅₆₇₈₉';
  function sub(n) { return String(n).split('').map(function (c) { return SUBD.charAt(+c); }).join(''); }
  var LAB = ['p₁', 'v₁', 'p₂', 'v₂', 'p₃', 'v₃', 'p₄', 'v₄', 'p₅', 'v₅', 'c', 'w₁', 'w₂'];
  var X0 = D.env.start_state.state.slice();
  var RS = D.env.episodes.recording_controller.first_steps;
  var REC = RS[0].command.slice();
  var SA0 = D.env.episodes.stall_aware.first_steps[0].command.slice();
  var KNEE = P.constants.knee;
  var Am = D.env.matrices.A_13x13, Bm = D.env.matrices.B_13x4;
  function supDigits(n) { return String(n).split('').map(function (c) { return '⁰¹²³⁴⁵⁶⁷⁸⁹'.charAt(+c); }).join(''); }
  function upDown(a) { return a < 0 ? 'upstream' : (a > 0 ? 'downstream' : 'off'); }

  // the lesson's shared command (steps 1.3, 1.5, 1.9, 2.1): four requested throttles; the step clips them
  var CMD = {
    req: REC.slice(), subs: [],
    set: function (req, from) { this.req = req.slice(); var me = this; this.subs.forEach(function (f) { f(from, me.req); }); },
    on: function (f) { this.subs.push(f); },
    applied: function () { return this.req.map(function (v) { return P.clip(v, -1, 1); }); }
  };
  var PRESETS = [
    { k: 'rec', t: 'Recorded command', c: REC },
    { k: 'cap', t: 'Thruster 1 at 80%', c: [-0.8, REC[1], REC[2], REC[3]] },
    { k: 'sa', t: 'Stall-aware command (a fixed rule that never passes 80%; step 2.2)', c: SA0 },
    { k: 'off', t: 'All off', c: [0, 0, 0, 0] },
    { k: 'custom', t: 'Custom' }
  ];
  function presetKey(req) {
    for (var i = 0; i < PRESETS.length - 1; i++) {
      var c = PRESETS[i].c, same = true;
      for (var k = 0; k < 4; k++) if (c[k] !== req[k]) same = false;
      if (same) return PRESETS[i].k;
    }
    return 'custom';
  }
  /** Four sliders for the shared command (−1.3 to 1.3; the part beyond ±1 is shaded, clipping cuts it back). */
  function cmdSliders(box, o) {
    var me = {}, sl = [];
    var grid = el('div', 'ls-grid4');
    for (var k = 0; k < 4; k++) (function (k) {
      var s = Slider({ label: 'Thruster ' + (k + 1) + (o && o.short ? '' : ', requested throttle'), min: -1.3, max: 1.3, step: 0.01, value: CMD.req[k], typed: true,
        cls: 'ls-over', fmt: function (x) { return F(x, 4); },
        onInput: function (v) { var r = CMD.req.slice(); r[k] = v; CMD.set(r, me); } });
      sl.push(s); grid.appendChild(s.el);
    })(k);
    box.appendChild(grid);
    CMD.on(function (from, req) { if (from === me) return; sl.forEach(function (s, k) { s.set(req[k]); }); });
    me.sliders = sl;
    return me;
  }
  function presetSeg(box, label) {
    var me = {};
    var seg = Seg({ label: label || 'Command', value: presetKey(CMD.req), options: PRESETS.map(function (p) { return { k: p.k, t: p.t }; }),
      onChange: function (k) { var p = PRESETS.filter(function (x) { return x.k === k; })[0]; if (p.c) CMD.set(p.c, me); } });
    box.appendChild(seg.el);
    CMD.on(function (from, req) { seg.set(presetKey(req)); });
    return seg;
  }

  // ----------------------------------------------------------------------- 1.1 the channel and its parts
  CTL.world = function (box) {
    var out = el('p', 'ls-out', 'Tap a float, a thruster, the current or the wave.');
    out.setAttribute('data-t', 'part');
    var strip = el('div', 'ls-strip'); strip.setAttribute('aria-label', 'The 13 numbers of a state');
    var cells = LAB.map(function (l) { var c = el('span', 'ls-cell', l); strip.appendChild(c); return c; });
    var ch = ChannelView(box, { title: 'The channel: five floats on springs, four thrusters between them, the current and the wave. Each part is a button.', parts: pick, height: 204 });
    ch.set({ ys: [0, 0, 0, 0, 0], thr: [0, 0, 0, 0], push: X0[10] + X0[11], curLabel: 'current (downstream)' });
    add(box, out, strip, Note('The strip under the drawing always holds the 13 numbers of one moment; the highlighted cells belong to the part you tapped.'));
    function pick(key) {
      var on = [], t;
      if (key.charAt(0) === 'f') { var j = +key.slice(1); on = [2 * j, 2 * j + 1]; t = 'Float ' + (j + 1) + ': its position number p' + sub(j + 1) + ' and speed number v' + sub(j + 1) + ' (2 of the 13 numbers).'; }
      else if (key === 'current') { on = [10]; t = 'The current c (1 of the 13); it never changes during an episode.'; }
      else if (key === 'wave') { on = [11, 12]; t = 'The wave numbers w₁ and w₂ (2 of the 13).'; }
      else { var k = +key.slice(1); t = 'Thruster ' + (k + 1) + ': not part of the 13 numbers; it is one of the 4 throttles you set (step 1.3).'; }
      cells.forEach(function (c, i) { c.classList.toggle('on', on.indexOf(i) >= 0); });
      out.textContent = t; out.setAttribute('data-on', on.join(','));
      ch.press(key);
      announce(t);
    }
    return {};
  };

  // ----------------------------------------------------------------------- 1.2 start 0: the 13 numbers
  CTL.state = function (box) {
    var j = 0, p = X0[0], v = X0[1];
    var row = el('div', 'ls-mapwrap');
    var map = MapView(row, { title: 'A float\'s point on its map: p across, v up, rest at the center; the circle has radius d.', drag: function (pp, vv) { p = pp; v = vv; sP.set(p); sV.set(v); draw(true); } });
    var side = el('div', 'ls-mapside');
    var seg = Seg({ label: 'Load a float of start 0', value: 0, options: [0, 1, 2, 3, 4].map(function (k) { return { k: k, t: 'float ' + (k + 1) }; }),
      onChange: function (k) { j = k; p = X0[2 * k]; v = X0[2 * k + 1]; sP.set(p); sV.set(v); draw(true); } });
    var sP = Slider({ label: 'p (position number)', min: -1, max: 1, step: 0.0001, key: 0.01, value: p, fmt: function (x) { return F(x, 4); }, onInput: function (x) { p = x; draw(true); } });
    var sV = Slider({ label: 'v (speed number)', min: -1, max: 1, step: 0.0001, key: 0.01, value: v, fmt: function (x) { return F(x, 4); }, onInput: function (x) { v = x; draw(true); } });
    var back = Btn('Back to start 0', function () { p = X0[2 * j]; v = X0[2 * j + 1]; sP.set(p); sV.set(v); draw(true); });
    add(side, seg.el, sP.el, sV.el, back);
    add(row, side); box.appendChild(row);
    var lines = Lines(box), L = Line('d');
    lines.appendChild(L.el);
    function draw(user) {
      var fl = -1;
      for (var k = 0; k < 5; k++) if (X0[2 * k] === p && X0[2 * k + 1] === v) fl = k;
      var sp = F(p), svv = F(v), p2 = p * p, v2 = v * v, d = Math.sqrt(p2 + v2);
      L.set(fl >= 0 ? 'd' + sub(fl + 1) : 'd', ['√(' + par(sp) + '² + ' + par(svv) + '²)', '√(' + F(p2) + ' + ' + F(v2) + ')', '√' + F(p2 + v2), F(d)], d);
      map.set(p, v);
      if (user) L.say();
    }
    draw(false);
    return {};
  };

  // ----------------------------------------------------------------------- 1.3 four throttles, and clipping
  CTL.command = function (box) {
    var cs = cmdSliders(box);
    box.appendChild(Row(Btn('Back to the recorded command', function () { CMD.set(REC, null); })));
    var lines = Lines(box), clipL = [], sizeL = [], wordS = [];
    for (var k = 0; k < 4; k++) { clipL.push(Line('clip' + (k + 1))); sizeL.push(Line('size' + (k + 1))); add(lines, clipL[k].el, sizeL[k].el); }
    function draw(from) {
      CMD.req.forEach(function (r, k) {
        var a = P.clip(r, -1, 1), sr = F(r);
        clipL[k].set('thruster ' + (k + 1), ['clip(' + sr + ', ' + MINUS + '1, 1)', F(a)], a, null, ': ');
        sizeL[k].set('thruster ' + (k + 1) + ', size', [F(Math.abs(a)) + ' × 100', F(Math.abs(a) * 100, 2)], Math.abs(a) * 100,
          a === 0 ? 'off' : 'so ' + F(Math.abs(a) * 100, 2) + '% ' + upDown(a), ': ');
      });
      if (from) clipL[0].say();
    }
    CMD.on(function (from) { draw(from); });
    draw(null);
    return {};
  };

  // ----------------------------------------------------------------------- 1.4 the push, and the stall past 80%
  CTL.stall = function (box) {
    var a = REC[0], world = 'stall';
    var sA = Slider({ label: 'Throttle a', min: -1, max: 1, step: 0.01, value: a, typed: true, fmt: function (x) { return F(x, 4); }, onInput: function (x) { a = x; draw(true); } });
    var wSeg = Seg({ label: 'World', value: world, options: [{ k: 'stall', t: 'Stalling' }, { k: 'no_stall', t: 'No-stall' }], onChange: function (k) { world = k; draw(true); } });
    var pre = el('div', 'ls-row');
    [[0.8, '80%'], [0.9, '90%'], [1, '100%'], [REC[0], MINUS + '98.96%']].forEach(function (q) { pre.appendChild(Btn(q[1], function () { a = q[0]; sA.set(a); draw(true); })); });
    add(box, sA.el, pre, wSeg.el);
    var lines = Lines(box), caseS = Stmt('case'), exL = Line('excess'), loL = Line('loss'), puL = Line('push');
    add(lines, caseS.el, exL.el, loL.el, puL.el);
    var rS = Ro('push, stalling world', 'push-stall'), rN = Ro('push, no-stall world', 'push-nostall');
    box.appendChild(Ros(rS, rN));
    var xs = []; for (var i = -100; i <= 100; i++) xs.push(i / 100);
    var chart = Chart(box, { title: 'Push against throttle in both worlds: equal up to 80%; past 80% the stalling world\'s push falls to 0.65 at 90% and 0.5 at full throttle.',
      height: 230, x: { min: -1, max: 1, label: 'throttle a', ticks: [-1, -0.8, -0.5, 0, 0.5, 0.8, 1] }, y: { min: -1, max: 1, ticks: [-1, -0.5, 0, 0.5, 1] },
      series: [], hover: null });
    function cfg() {
      return { title: chart.cfgTitle, height: 230, x: { min: -1, max: 1, label: 'throttle a', ticks: [-1, -0.8, -0.5, 0, 0.5, 0.8, 1] }, y: { min: -1, max: 1, ticks: [-1, -0.5, 0, 0.5, 1] },
        series: [
          { key: 'no_stall', name: 'push, no-stall world', color: '--k-rise', dash: true, pts: xs.map(function (x) { return [x, x]; }) },
          { key: 'stall', name: 'push, stalling world', color: '--k-stall', pts: xs.map(function (x) { return [x, P.thrust1(x, 'stall')]; }), keep: true }],
        vlines: [{ x: -0.8, label: MINUS + '80%' }, { x: 0.8, label: '80%' }],
        marks: [{ key: 'now', x: a, y: P.thrust1(a, world), color: world === 'stall' ? '--k-stall' : '--k-rise' }],
        hover: { xs: xs, title: function (x) { return 'throttle ' + F(x, 2); },
          rows: function (x) { return [{ color: '--k-stall', name: 'stalling world', value: F(P.thrust1(x, 'stall')) }, { color: '--k-rise', dash: true, name: 'no-stall world', value: F(x) }]; } } };
    }
    chart.cfgTitle = 'Push against throttle in both worlds: equal up to 80%; past 80% the stalling world\'s push falls to 0.65 at 90% and 0.5 at full throttle.';
    function draw(user) {
      var m = Math.abs(a), fS = P.thrust1(a, 'stall'), fN = P.thrust1(a, 'no_stall'), f = world === 'stall' ? fS : fN;
      var sm = F(m);
      if (world === 'stall' && m > KNEE) {
        caseS.set('|a| = ' + sm + ' > 0.8: past 80%, so it stalls.');
        var ex = m - KNEE, lo = 1.5 * ex, sz = Math.abs(fS);
        exL.set('excess', [sm + ' ' + MINUS + ' 0.8', F(ex)], ex, null, ': ');
        loL.set('loss', ['1.5 × ' + F(ex), F(lo)], lo, null, ': ');
        var inner = '0.8 ' + MINUS + ' ' + F(lo);
        puL.set('f(a)', [a < 0 ? MINUS + '(' + inner + ')' : inner, F(fS)], fS);
        exL.el.hidden = false; loL.el.hidden = false;
      } else {
        caseS.set(world === 'stall' ? '|a| = ' + sm + ' ≤ 0.8: the push equals the throttle.' : 'No-stall world: the push always equals the throttle.');
        puL.set('f(a)', [F(a)], f);
        exL.el.hidden = true; loL.el.hidden = true;
      }
      rS.set(F(fS), fS); rN.set(F(fN), fN);
      chart.update(cfg());
      if (user) puL.say();
    }
    draw(false);
    return { enter: function () { chart.draw(); } };
  };

  // ----------------------------------------------------------------------- 1.5 what each thruster pushes
  function thrusterColumn(f) {     // the thrusters' part of each speed number: 0.3 f(a_(j−1)) + 0.3 f(a_j), the code's terms
    var out = [];
    for (var i = 0; i < 10; i++) {
      var acc = null;
      for (var k = 0; k < 4; k++) if (Bm[i][k] !== 0) acc = acc === null ? Bm[i][k] * f[k] : acc + Bm[i][k] * f[k];
      out.push(acc === null ? 0 : acc);
    }
    return out;
  }
  CTL.thrusters = function (box) {
    var ch = ChannelView(box, { title: 'Five floats and four thrusters; an arrow on each float shows what the thrusters add to its speed number this step (left: upstream).' });
    box.appendChild(StallKey());
    cmdSliders(box, { short: true });
    var lines = Lines(box), Ls = [];
    for (var j = 0; j < 5; j++) { Ls.push(Line('v' + (j + 1))); lines.appendChild(Ls[j].el); }
    function draw(from) {
      var f = P.thrust(CMD.applied(), 'stall'), col = thrusterColumn(f);
      for (var j = 0; j < 5; j++) {
        var lab = 'added to v' + sub(j + 1), x = col[2 * j + 1];
        if (j === 0) Ls[j].set(lab, ['0.3 × ' + par(F(f[0])), F(x)], x, null, ': ');
        else if (j === 4) Ls[j].set(lab, ['0.3 × ' + par(F(f[3])), F(x)], x, null, ': ');
        else {
          var s2 = f[j - 1] + f[j];
          Ls[j].set(lab, ['0.3 × (' + F(f[j - 1]) + ' + ' + par(F(f[j])) + ')', '0.3 × ' + par(F(s2)), F(x)], x, null, ': ');
        }
      }
      ch.set({ ys: [0, 0, 0, 0, 0], thr: CMD.applied(), water: false, waveAmp: 0, arrows: [1, 3, 5, 7, 9].map(function (i) { return col[i] / 0.45; }) });
      if (from) Ls[1].say();
    }
    CMD.on(draw);
    draw(null);
    return {};
  };

  // ----------------------------------------------------------------------- 1.6 the water's push
  CTL.water = function (box) {
    var c = X0[10], w1 = X0[11];
    var sC = Slider({ label: 'Current c', min: 0.1, max: 1.1, step: 0.01, value: c, fmt: function (x) { return F(x, 4); }, onInput: function (x) { c = x; draw(true); } });
    var sW = Slider({ label: 'Wave\'s push now, w₁', min: -0.7, max: 0.7, step: 0.01, value: w1, fmt: function (x) { return F(x, 4); }, onInput: function (x) { w1 = x; draw(true); } });
    add(box, Row(sC, sW), Row(Btn('Back to start 0', function () { c = X0[10]; w1 = X0[11]; sC.set(c); sW.set(w1); draw(true); })));
    var ch = ChannelView(box, { title: 'The water pushes every float the same way; the arrow on each float shows what it adds to the float\'s speed number.' });
    var lines = Lines(box), L1 = Line('push'), L2 = Line('each');
    add(lines, L1.el, L2.el);
    function draw(user) {
      var pu = c + w1, part = Am[1][10] * pu;          // 0.3 (c + w1), as the lesson's numbers were computed
      L1.set('c + w₁', [F(c) + ' + ' + par(F(w1)), F(pu)], pu);
      L2.set('each speed number', ['0.3 × ' + par(F(pu)), F(part)], part, null, ': ');
      ch.set({ ys: [0, 0, 0, 0, 0], thr: [0, 0, 0, 0], push: pu, arrows: [part, part, part, part, part].map(function (x) { return x / 0.45; }) });
      if (user) L2.say();
    }
    draw(false);
    return {};
  };

  // ----------------------------------------------------------------------- 1.7 the wave turns, the current stays
  CTL.wave = function (box) {
    var WT = D.env.wave_turn, cs = WT.cos, sn = WT.sin, period = D.env.constants.wave_period_steps;
    var w = [X0[11], X0[12]], n = 0, trail = [], last = null;
    var size = 220, R = 90;
    var s = sv('svg', { 'class': 'ls-map ls-wavec', viewBox: '0 0 ' + size + ' ' + size, width: size, height: size, role: 'img', 'aria-label': 'The wave\'s point (w₁ across, w₂ up) on its circle; each turn moves it 0.27 radians around the center.' });
    var sc = R / 0.4;
    add(s, sv('line', { 'class': 'ls-map-ax', x1: 10, x2: size - 10, y1: size / 2, y2: size / 2 }), sv('line', { 'class': 'ls-map-ax', x1: size / 2, x2: size / 2, y1: 10, y2: size - 10 }));
    add(s, sv('text', { 'class': 'ls-map-t', x: size - 12, y: size / 2 - 6, 'text-anchor': 'end' }, 'w₁'), sv('text', { 'class': 'ls-map-t', x: size / 2 + 6, y: 20 }, 'w₂'));
    var circ = sv('circle', { 'class': 'ls-map-ring', cx: size / 2, cy: size / 2, r: Math.hypot(w[0], w[1]) * sc });
    var tg = sv('g'), dot = sv('circle', { 'class': 'ls-map-dot', r: 7 });
    add(s, circ, tg, dot);
    var wrap = el('div', 'ls-mapwrap'), side = el('div', 'ls-mapside');
    var bT = Btn('Turn', function () { pl.stop(); turn(1); show(true); }), b10 = Btn('Turn ×10', function () { pl.stop(); turn(10); show(true); });
    var bP = Btn('Play', function () {}), bR = Btn('Reset', function () { pl.stop(); w = [X0[11], X0[12]]; n = 0; trail = []; last = null; show(true); });
    var pl = Player(bP, 300, function () { turn(1); show(false); });
    var rN = Ro('steps so far', 'n'), rW1 = Ro('w₁', 'w1'), rW2 = Ro('w₂', 'w2'), rPh = Ro('phase (radians)', 'phase'), rCy = Ro('cycles so far', 'cycles');
    var bars = Bars([{ key: 'p', label: 'water\'s push c + w₁', color: '--water-ink', name: 'pushbar' }]);
    add(side, Row(bT, b10, bP, bR), Ros(rN, rW1, rW2, rPh), Ros(rCy), bars.el);
    add(wrap, s, side); box.appendChild(wrap);
    var lines = Lines(box), L1 = Line('w1p'), L2 = Line('w2p');
    add(lines, L1.el, L2.el);
    box.appendChild(Note('Each turn uses the stored cos 0.27 = ' + F(cs, 6) + ' and sin 0.27 = ' + F(sn, 6) + '; the current c = ' + F(X0[10]) + ' never changes.'));
    function turn(k) {
      for (var i = 0; i < k; i++) {
        var b = w.slice(), nw = P.waveTurn(w[0], w[1]);
        last = { b: b, a: nw };
        trail.push(b); if (trail.length > 24) trail.shift();
        w = nw; n += 1;
      }
    }
    function show(user) {
      dot.setAttribute('cx', (size / 2 + w[0] * sc).toFixed(1)); dot.setAttribute('cy', (size / 2 - w[1] * sc).toFixed(1));
      clear(tg);
      trail.forEach(function (q, i) { add(tg, sv('circle', { 'class': 'ls-trail', cx: (size / 2 + q[0] * sc).toFixed(1), cy: (size / 2 - q[1] * sc).toFixed(1), r: 3, opacity: ((i + 1) / trail.length * 0.7).toFixed(2) })); });
      var ph = Math.atan2(w[1], w[0]); if (ph < 0) ph += 2 * Math.PI;
      rN.set(String(n), n); rW1.set(F(w[0]), w[0]); rW2.set(F(w[1]), w[1]); rPh.set(F(ph), ph); rCy.set(F(n / period, 2), n / period);
      bars.set({ p: { v: X0[10] + w[0], t: F(X0[10] + w[0]) } }, 1.3);
      var b = last ? last.b : [X0[11], X0[12]], a = last ? last.a : P.waveTurn(X0[11], X0[12]);
      var t1 = b[0] * cs, t2 = -(b[1] * sn), t3 = b[0] * sn, t4 = b[1] * cs;
      L1.set('w₁′', [F(cs, 6) + ' × ' + par(F(b[0], 6)) + ' ' + MINUS + ' ' + F(sn, 6) + ' × ' + par(F(b[1], 6)), F(t1) + (t2 < 0 ? ' ' + MINUS + ' ' + F(-t2) : ' + ' + F(t2)), F(a[0])], a[0]);
      L2.set('w₂′', [F(sn, 6) + ' × ' + par(F(b[0], 6)) + ' + ' + F(cs, 6) + ' × ' + par(F(b[1], 6)), F(t3) + (t4 < 0 ? ' ' + MINUS + ' ' + F(-t4) : ' + ' + F(t4)), F(a[1])], a[1]);
      if (user) L1.say();
    }
    show(false);
    return { leave: function () { pl.stop(); } };
  };

  // ----------------------------------------------------------------------- 1.8 the floats on their own
  CTL.floats = function (box) {
    var PF = D.env.matrices.per_float, j = 0, p = X0[0], v = X0[1], t = 0, trail = [], last = null;
    box.appendChild(rich('p', 'ls-sub', '(a) Let a float swing on its own'));
    var wrap = el('div', 'ls-mapwrap'), side = el('div', 'ls-mapside');
    var map = MapView(wrap, { title: 'A float\'s point spiraling in: each step turns it by θ and shrinks it by ρ.' });
    var seg = Seg({ label: 'Float', value: 0, options: [0, 1, 2, 3, 4].map(function (k) { return { k: k, t: 'float ' + (k + 1) }; }), onChange: function (k) { pl.stop(); j = k; reset(); show(true); } });
    var bS = Btn('Step', function () { pl.stop(); stepOnce(); show(true); }), bP = Btn('Play', function () {}), bR = Btn('Reset', function () { pl.stop(); reset(); show(true); });
    var pl = Player(bP, 140, function () { stepOnce(); show(false); return t < 400; });
    var rT = Ro('steps t', 't'), rP = Ro('p', 'p'), rV = Ro('v', 'v'), rD = Ro('distance', 'dist');
    add(side, seg.el, Row(bS, bP, bR), Ros(rT, rP, rV, rD));
    wrap.appendChild(side); box.appendChild(wrap);
    box.appendChild(Note('The designed swing alone; the real floats also feel the coupling.'));
    var lines = Lines(box), L1 = Line('tp'), L2 = Line('tv'), L3 = Line('ownp'), L4 = Line('ownv'), Ld = Line('dist');
    add(lines, L1.el, L2.el, L3.el, L4.el, Ld.el);
    var d0 = 0;
    function reset() { p = X0[2 * j]; v = X0[2 * j + 1]; t = 0; trail = []; last = null; d0 = Math.sqrt(p * p + v * v); }
    function stepOnce() {
      var th = PF[j].design_angle, rho = PF[j].design_radius, c = Math.cos(th), s = Math.sin(th);
      var cp = c * p, svv = s * v, sp = s * p, cv = c * v, tp = cp - svv, tv = sp + cv, op = rho * tp, ov = rho * tv;
      last = { p: p, v: v, c: c, s: s, cp: cp, sv: svv, sp: sp, cv: cv, tp: tp, tv: tv, op: op, ov: ov, rho: rho };
      trail.push([p, v]); if (trail.length > 60) trail.shift();
      p = op; v = ov; t += 1;
    }
    function show(user) {
      map.set(p, v, trail.concat([[p, v]]));
      var rho = PF[j].design_radius, dist = Math.pow(rho, t) * d0;
      rT.set(String(t), t); rP.set(F(p, 6), p); rV.set(F(v, 6), v); rD.set(F(dist, 6), dist);
      var q = last;
      [L1, L2, L3, L4].forEach(function (L) { L.el.hidden = !q; });
      if (q) {
        L1.set('turned p', [F(q.c, 6) + ' × ' + par(F(q.p, 6)) + ' ' + MINUS + ' ' + F(q.s, 6) + ' × ' + par(F(q.v, 6)), F(q.cp, 6) + (q.sv < 0 ? ' + ' + F(-q.sv, 6) : ' ' + MINUS + ' ' + F(q.sv, 6)), F(q.tp, 6)], q.tp);
        L2.set('turned v', [F(q.s, 6) + ' × ' + par(F(q.p, 6)) + ' + ' + F(q.c, 6) + ' × ' + par(F(q.v, 6)), F(q.sp, 6) + (q.cv < 0 ? ' ' + MINUS + ' ' + F(-q.cv, 6) : ' + ' + F(q.cv, 6)), F(q.tv, 6)], q.tv);
        L3.set('p' + sub(j + 1) + ' own', [String(q.rho) + ' × ' + par(F(q.tp, 6)), F(q.op, 6)], q.op);
        L4.set('v' + sub(j + 1) + ' own', [String(q.rho) + ' × ' + par(F(q.tv, 6)), F(q.ov, 6)], q.ov);
      }
      if (t === 0) Ld.set('distance from rest at the start', [F(d0, 6)], d0, null, ': ');
      else Ld.set('distance after ' + t + (t === 1 ? ' step' : ' steps'), [String(rho) + (t === 1 ? '' : supDigits(t)) + ' × ' + F(d0, 6), F(dist, 6)], dist, t === 1 ? 'a turn keeps the distance; the shrink multiplies it by ' + rho : null, ': ');
      if (user && q) L3.say();
    }
    reset(); show(false);

    // (b) multiply a row of A
    box.appendChild(rich('p', 'ls-sub', '(b) Multiply a row of A by start 0\'s ten float numbers'));
    var row = 0, off = false;
    var rSeg = Seg({ label: 'Row', value: 0, cls: 'ls-seg-dense', options: LAB.slice(0, 10).map(function (l, i) { return { k: i, t: l }; }), onChange: function (k) { row = k; rowShow(true); } });
    var sw = Switch('Coupling off (the five designed swings alone)', false, function (x) { off = x; rowShow(true); });
    add(box, rSeg.el, sw.el);
    var tab = Table(['column', 'number of A', 'start 0', 'product'], { cls: 'ls-tab10' });
    box.appendChild(tab.el);
    var lines2 = Lines(box), Ls = Line('rowsum');
    lines2.appendChild(Ls.el);
    function rowShow(user) {
      var M = off ? D.env.matrices.design_swings_O_10x10 : D.env.matrices.A_float_block_10x10, s = X0.slice(0, 10), own = [2 * Math.floor(row / 2), 2 * Math.floor(row / 2) + 1];
      var prods = s.map(function (x, k) { return M[row][k] * x; }), sum = 0;
      for (var k = 0; k < 10; k++) sum = k === 0 ? s[0] * M[row][0] : sum + s[k] * M[row][k];
      tab.rows(prods.map(function (pr, k) {
        return { cls: own.indexOf(k) >= 0 ? 'ls-own' : '', cells: [LAB[k], { t: F(M[row][k], 6), cls: 'n', exact: M[row][k] }, { t: F(s[k], 6), cls: 'n', exact: s[k] }, { t: F(pr, 6), cls: 'n', exact: pr }] };
      }));
      var ownSum = prods[own[0]] + prods[own[1]], other = sum - ownSum;
      Ls.set(LAB[row] + ' on its own' + (off ? ' (designed swing only)' : ' (A s)'), [F(ownSum, 6) + (off ? '' : (other < 0 ? ' ' + MINUS + ' ' + F(-other, 6) : ' + ' + F(other, 6))), F(sum, 6)], sum, off ? null : 'own two columns + the other eight', ': ');
      if (user) Ls.say();
    }
    rowShow(false);
    return { leave: function () { pl.stop(); } };
  };

  // ----------------------------------------------------------------------- 1.9 one whole step: before, parts, after
  function stepParts(x, a, world) {
    var st = P.step(x, a, world), s = x.slice(0, 10), own = P.floatsAlone(s), thr = thrusterColumn(st.thrust), wat = [];
    for (var i = 0; i < 10; i++) wat.push(Am[i][10] === Am[i][11] && Am[i][12] === 0 ? Am[i][10] * (x[10] + x[11]) : x[10] * Am[i][10] + x[11] * Am[i][11] + x[12] * Am[i][12]);
    return { st: st, own: own, thr: thr, wat: wat };
  }
  CTL.transition = function (box) {
    var world = 'stall', cmp = false;
    presetSeg(box, 'Command at start 0');
    var wSeg = Seg({ label: 'World', value: world, options: [{ k: 'stall', t: 'Stalling' }, { k: 'no_stall', t: 'No-stall' }], onChange: function (k) { world = k; draw(true); } });
    var sw = Switch('Compare worlds', false, function (v) { cmp = v; draw(true); });
    add(box, wSeg.el, sw.el);
    cmdSliders(box, { short: true });
    var tab = Table(['number', 'before', 'on their own (A s)', 'thrusters', 'water', 'after'], { rowHead: true });
    box.appendChild(tab.el);
    var foot = el('p', 'note ls-foot-n'); box.appendChild(foot);
    var lines = Lines(box), Lv = Line('v1');
    lines.appendChild(Lv.el);
    var base = null;
    function cellsOf(x, a, wld) {
      var q = stepParts(x, a, wld), rows = [], order = [0, 2, 4, 6, 8, 1, 3, 5, 7, 9, 10, 11, 12];
      order.forEach(function (i) {
        if (i < 10) rows.push({ i: i, c: [LAB[i], F(x[i]), F(q.own[i]), i % 2 ? F(q.thr[i]) : '0', i % 2 ? F(q.wat[i]) : '0', F(q.st.next[i])], ex: [x[i], q.own[i], q.thr[i], q.wat[i], q.st.next[i]] });
        else rows.push({ i: i, c: [LAB[i], F(x[i]), i === 10 ? 'stays' : 'turns', '', '', F(q.st.next[i])], ex: [x[i], null, null, null, q.st.next[i]] });
      });
      return { rows: rows, q: q };
    }
    function draw(user) {
      if (!base) base = cellsOf(X0, REC, 'stall');
      var a = CMD.applied(), cur = cellsOf(X0, a, world), alt = cmp ? cellsOf(X0, a, world === 'stall' ? 'no_stall' : 'stall') : null;
      var head = tab.el.querySelector('thead tr');
      while (head.children.length > 6) head.removeChild(head.lastChild);
      if (cmp) head.appendChild(el('th', null, world === 'stall' ? 'no-stall minus stalling' : 'stalling minus no-stall'));
      var notes = [];
      tab.rows(cur.rows.map(function (r, k) {
        var cells = r.c.map(function (t, ci) {
          if (ci === 0) return t;
          var o = { t: t, cls: (t !== '' && t !== 'stays' && t !== 'turns') ? 'n' : '' };
          if (r.ex[ci - 1] !== null && r.ex[ci - 1] !== undefined) o.exact = r.ex[ci - 1];
          if (t !== base.rows[k].c[ci]) o.cls += ' ls-diff';
          return o;
        });
        if (cmp) { var dlt = alt.q.st.next[r.i] - cur.q.st.next[r.i]; cells.push({ t: Math.abs(dlt) < 5e-5 ? '0' : Fs(dlt), cls: 'n', exact: dlt }); }
        if (r.i < 10 && r.i % 2 === 1) {
          var shown = parseShown(r.c[2]) + parseShown(r.c[3]) + parseShown(r.c[4]), tot = parseShown(r.c[5]);
          if (Math.abs(shown - tot) > 1e-9) notes.push(LAB[r.i] + ': the parts as shown add to ' + F(shown) + '; the exact total is ' + F(cur.q.st.next[r.i], 6) + '.');
        }
        return { cells: cells };
      }));
      foot.textContent = 'Each cell is the exact value rounded on its own; changed cells (against the recorded command) are marked.' + (notes.length ? ' ' + notes.join(' ') : '');
      var q = cur.q;
      Lv.set('v₁′', [F(q.own[1]) + ' + ' + par(F(q.thr[1])) + ' + ' + par(F(q.wat[1])), F(q.st.next[1])], q.st.next[1]);
      if (user) Lv.say();
    }
    CMD.on(function (from) { draw(!!from); });
    draw(false);
    return { enter: function () { tab.cue(); } };
  };

  // ======================================================================= chapter 2: rewards, episodes and values
  var SUPD = '⁰¹²³⁴⁵⁶⁷⁸⁹';
  function sup(n) { return String(n).split('').map(function (c) { return SUPD.charAt(+c); }).join(''); }
  var CTRLS = [{ k: 'zero_throttle', t: 'No throttle' }, { k: 'stall_aware', t: 'Stall-aware' }, { k: 'rising_thrust', t: 'Rising-thrust' }];
  var CTRL_COLOR = { zero_throttle: '--k-none', stall_aware: '--k-ref', rising_thrust: '--k-rise' };
  var WORLDS = [{ k: 'stall', t: 'Stalling' }, { k: 'no_stall', t: 'No-stall' }];
  // the episode both steps 2.2 and 2.3 follow (one controller and one world, shared)
  var EP = { ctrl: 'stall_aware', world: 'stall', subs: [], cache: {},
    set: function (k, v, from) { this[k] = v; var me = this; this.subs.forEach(function (f) { f(from); }); },
    on: function (f) { this.subs.push(f); },
    roll: function () { var key = this.ctrl + '|' + this.world; if (!this.cache[key]) this.cache[key] = P.rollout(P.policy(this.ctrl), X0, this.world); return this.cache[key]; } };

  // ----------------------------------------------------------------------- 2.1 the reward of one step
  CTL.reward = function (box) {
    presetSeg(box, 'Command at start 0');
    cmdSliders(box, { short: true });
    var lines = Lines(box), Ls = Line('s2'), La = Line('a2'), Lc = Line('cost'), Lr = Line('r');
    add(lines, Ls.el, La.el, Lc.el, Lr.el);
    var bars = Bars([{ key: 'f', label: 'float cost ‖s‖²', color: '--water-ink', name: 'bar-float' }, { key: 't', label: 'throttle cost 0.3 ‖a‖²', color: '--accent', name: 'bar-throttle' }]);
    box.appendChild(bars.el);
    box.appendChild(Note('The float cost stays the same whatever the throttles: it depends only on the state before the step.'));
    function draw(from) {
      var a = CMD.applied(), st = P.step(X0, a, 'stall'), s = X0.slice(0, 10);
      Ls.set('‖s‖²', [s.map(function (x) { return F(x * x); }).join(' + '), F(st.floatCost)], st.floatCost);
      La.set('‖a‖²', [a.map(function (x) { return F(x * x); }).join(' + '), F(st.sumSquaresCommand)], st.sumSquaresCommand);
      var tc = P.constants.throttleCost * st.sumSquaresCommand;
      Lc.set('throttle cost', ['0.3 × ' + F(st.sumSquaresCommand), F(tc)], tc, null, ': ');
      Lr.set('r', [MINUS + '(' + F(st.floatCost) + ' + ' + F(tc) + ')', F(st.reward)], st.reward);
      bars.set({ f: { v: st.floatCost, t: F(st.floatCost) }, t: { v: tc, t: F(tc) } }, 2.5);
      if (from) Lr.say();
    }
    CMD.on(draw);
    draw(null);
    return {};
  };

  // ----------------------------------------------------------------------- 2.2 an episode: 100 steps
  CTL.episode = function (box) {
    var t = 0;
    var cSeg = Seg({ label: 'Controller', value: EP.ctrl, options: CTRLS, onChange: function (k) { pl.stop(); EP.set('ctrl', k, me); } });
    var wSeg = Seg({ label: 'World', value: EP.world, options: WORLDS, onChange: function (k) { pl.stop(); EP.set('world', k, me); } });
    var me = {};
    add(box, cSeg.el, wSeg.el);
    var ch = ChannelView(box, { title: 'The channel at step t: floats at heights set by their position numbers, the four throttle gauges (the stall color past 80% in the stalling world) and the water\'s push.' });
    box.appendChild(StallKey());
    var bB = Btn('Back', function () { pl.stop(); go(t - 1, true); }), bS = Btn('Step', function () { pl.stop(); go(t + 1, true); });
    var bP = Btn('Play', function () {}), bR = Btn('Reset', function () { pl.stop(); go(0, true); });
    var pl = Player(bP, 250, function () { if (t >= 99) return false; go(t + 1, false); return t < 99; });
    var sT = Slider({ label: 'Step t', min: 0, max: 99, step: 1, value: 0, fmt: function (x) { return String(x); }, onInput: function (x) { pl.stop(); go(x, true); } });
    add(box, Row(bB, bS, bP, bR), sT.el);
    var rT = Ro('t', 't'), rW = Ro('water\'s push c + w₁', 'water'), rC = Ro('command aₜ', 'cmd'), rF = Ro('pushes', 'push');
    var rFc = Ro('float cost', 'fc'), rTc = Ro('throttle cost', 'tc'), rR = Ro('reward rₜ', 'r'), rJ = Ro('sum of rewards so far', 'J');
    rC.el.classList.add('ls-ro-wide'); rF.el.classList.add('ls-ro-wide');
    add(box, Ros(rT, rW, rFc, rTc), Ros(rC, rF), Ros(rR, rJ));
    var chart = Chart(box, { title: 'The 100 rewards of the episode; the current step is marked.', height: 170, x: { min: -1, max: 100 }, y: { min: -3, max: 0 }, series: [], hover: null });
    var det = el('details', 'ls-wblk'); det.setAttribute('data-phone', 'closed'); det.open = !mqPhone.matches;
    var sum = el('summary', null, 'Before → after: all 13 numbers of step t'); det.appendChild(sum);
    var tab = Table(['number', 'before (state t)', 'after (state t + 1)'], { rowHead: true });
    det.appendChild(tab.el); box.appendChild(det);
    function go(k, user) {
      t = clampN(Math.round(k), 0, 99); sT.set(t);
      var R = EP.roll(), x = R.states[t], a = R.commands[t], th = P.thrust(a, EP.world), st = P.step(x, a, EP.world);
      var past = a.filter(function (v) { return Math.abs(v) > KNEE; }).length;
      ch.set({ ys: [0, 2, 4, 6, 8].map(function (i) { return x[i] / 0.9; }), thr: a, stall: EP.world === 'stall', push: x[10] + x[11], waveAmp: Math.hypot(x[11], x[12]), wavePhase: Math.atan2(x[12], x[11]) });
      rT.set(String(t), t); rW.set(F(x[10] + x[11]), x[10] + x[11]); rC.set(vec(a, 4)); rF.set(vec(th, 4) + (past && EP.world === 'stall' ? '  (' + past + ' past 80%)' : ''));
      rFc.set(F(st.floatCost), st.floatCost); rTc.set(F(P.constants.throttleCost * st.sumSquaresCommand), P.constants.throttleCost * st.sumSquaresCommand);
      rR.set(F(R.rewards[t]), R.rewards[t]); rJ.set(F(R.running[t], 2), R.running[t]);
      tab.rows(LAB.map(function (l, i) { return { cells: [l, { t: F(x[i]), cls: 'n', exact: x[i] }, { t: F(R.states[t + 1][i]), cls: 'n', exact: R.states[t + 1][i] }] }; }));
      drawChart();
      if (user) announce('Step ' + t + ': reward ' + F(R.rewards[t]) + ', sum so far ' + F(R.running[t], 2) + '.');
    }
    function drawChart() {
      var R = EP.roll(), mn = Math.min.apply(null, R.rewards), lo = Math.min(-0.5, Math.floor(mn * 1.1 * 2) / 2);
      var col = CTRL_COLOR[EP.ctrl];
      chart.update({ title: 'The 100 rewards of the episode, step ' + t + ' marked.', height: 170, x: { min: -1, max: 100, label: 'step t' }, y: { min: lo, max: 0 },
        series: [{ key: 'rewards', name: 'reward rₜ', color: col, kind: 'bars', pts: R.rewards.map(function (r, k) { return [k, r, k === t ? '--ink' : col]; }), keep: true }],
        legendExtra: [{ name: 'the step shown, t = ' + t, color: '--ink' }],
        hover: { xs: R.rewards.map(function (r, k) { return k; }), title: function (x) { return 'step ' + x; },
          rows: function (x) { return [{ color: col, name: 'reward', value: F(R.rewards[x]) }, { name: 'sum so far', value: F(R.running[x], 2) }]; } } });
    }
    EP.on(function (from) { cSeg.set(EP.ctrl); wSeg.set(EP.world); go(t, from === me); });
    go(0, false);
    return { enter: function () { drawChart(); tab.cue(); }, leave: function () { pl.stop(); } };
  };

  // ----------------------------------------------------------------------- 2.3 return and discounted return
  CTL['return'] = function (box) {
    var t = 0, g = 0.98, me = {};
    var cSeg = Seg({ label: 'Controller', value: EP.ctrl, options: CTRLS, onChange: function (k) { EP.set('ctrl', k, me); } });
    var sT = Slider({ label: 'Step t', min: 0, max: 99, step: 1, value: 0, fmt: function (x) { return String(x); }, onInput: function (x) { t = x; draw(true); } });
    var sG = Slider({ label: 'Discount γ', min: 0.9, max: 0.999, step: 0.001, value: g, fmt: function (x) { return F(x, 3); }, onInput: function (x) { g = x; draw(true); } });
    add(box, cSeg.el, sT.el, sG.el);
    var rR = Ro('reward rₜ', 'r'), rW = Ro('weight γᵗ', 'w'), rJ = Ro('J so far', 'J'), rJd = Ro('Jγ so far', 'Jd');
    setRich(rJd.key, 'J_γ so far');
    box.appendChild(Ros(rR, rW, rJ, rJd));
    var lines = Lines(box), LJ = Line('J'), LJd = Line('Jd');
    add(lines, LJ.el, LJd.el);
    var cmp = Note('');
    box.appendChild(cmp);
    var chart = Chart(box, { title: 'The 100 weights γᵗ of the discounted return.', height: 170, x: { min: -1, max: 100 }, y: { min: 0, max: 1 }, series: [], hover: null });
    function draw(user) {
      var R = EP.roll(), ret = P.returns(R.rewards, g);
      var r = R.rewards[t], w = ret.weights[t], J = ret.running[t], Jd = ret.runningDiscounted[t];
      rR.set(F(r, 6), r); rW.set(F(w), w); rJ.set(F(J, 6), J); rJd.set(F(Jd, 6), Jd);
      if (t === 0) {
        LJ.set('J after 1 step', [F(r, 6)], J);
        LJd.set('J_γ after 1 step', [F(r, 6)], Jd);
      } else {
        LJ.set('J after ' + (t + 1) + ' steps', [F(ret.running[t - 1], 6) + ' + ' + par(F(r, 6)), F(J, 6)], J);
        LJd.set('J_γ after ' + (t + 1) + ' steps', [F(ret.runningDiscounted[t - 1], 6) + ' + ' + F(w, 6) + ' × ' + par(F(r, 6)), F(ret.runningDiscounted[t - 1], 6) + ' + ' + par(F(w * r, 6)), F(Jd, 6)], Jd);
      }
      var zero = D.env.episodes.zero_throttle.J_100_step_return;
      cmp.textContent = 'For comparison: no throttle from start 0 has J = ' + F(zero, 2) + ' over the whole episode' + (EP.world === 'stall' ? '.' : ' (the same in the no-stall world).');
      chart.update({ title: 'The 100 weights γᵗ of the discounted return at γ = ' + F(g, 3) + '.', height: 170, x: { min: -1, max: 100, label: 'steps ahead t' }, y: { min: 0, max: 1, ticks: [0, 0.25, 0.5, 0.75, 1] },
        series: [{ key: 'weights', name: 'weight γᵗ', color: '--accent', kind: 'bars', pts: ret.weights.map(function (x, k) { return [k, x, k === t ? '--ink' : '--accent']; }) }],
        legendExtra: [{ name: 'the step reached, t = ' + t, color: '--ink' }],
        hover: { xs: ret.weights.map(function (x, k) { return k; }), title: function (x) { return x + ' steps ahead'; }, rows: function (x) { return [{ color: '--accent', name: 'γᵗ', value: F(ret.weights[x]) }]; } } });
      if (user) LJd.say();
    }
    EP.on(function () { cSeg.set(EP.ctrl); draw(false); });
    draw(false);
    return { enter: function () { chart.draw(); } };
  };

  // ----------------------------------------------------------------------- 2.4 the value of a command
  CTL.value = function (box) {
    var HS = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000], hi = 11, sums = null;
    function ensure() {
      if (!sums) sums = { sa: P.partialSums(P.policy('stall_aware'), X0, 'stall', 1000), nt: P.partialSums(P.policy('zero_throttle'), X0, 'stall', 1000) };
      return sums;
    }
    var sH = Slider({ label: 'Horizon H (steps added up)', min: 0, max: HS.length - 1, step: 1, value: hi, fmt: function (k) { return FI(HS[k]); }, aria: function (k) { return FI(HS[k]) + ' steps'; }, onInput: function (k) { hi = k; draw(true); } });
    box.appendChild(sH.el);
    var rH = Ro('H', 'H'), rS = Ro('stall-aware controller', 'sa'), rN = Ro('no throttle', 'nt');
    box.appendChild(Ros(rH, rS, rN));
    var lines = Lines(box), Lt = Stmt('what');
    lines.appendChild(Lt.el);
    var chart = Chart(box, { title: 'Partial discounted sums against the horizon, for the stall-aware controller and for no throttle from start 0.', height: 220, x: { min: 1, max: 1000, log: true }, y: { min: -160, max: 0 }, series: [], hover: null });
    function draw(user) {
      var s = ensure(), H = HS[hi], a = s.sa[H - 1], b = s.nt[H - 1];
      rH.set(FI(H), H); rS.set(F(a), a); rN.set(F(b), b);
      Lt.set('r₀ + γ r₁ + … + γ' + (H > 1 ? sup(H - 1) : '⁰') + ' r' + sub(H - 1) + ' = ' + F(a) + ' (stall-aware) and ' + F(b) + ' (no throttle) after ' + FI(H) + (H === 1 ? ' step.' : ' steps.'));
      var xs = []; for (var h = 1; h <= 1000; h++) xs.push(h);
      chart.update({ title: 'Partial discounted sums against the horizon H (logarithmic): they settle at −12.7275 and −154.3109.', height: 220, endLabels: true,
        x: { min: 1, max: 1000, log: true, label: 'horizon H (steps)', ticks: [1, 10, 100, 1000], fmt: function (v) { return FI(v); } }, y: { min: -160, max: 0, ticks: [-150, -100, -50, 0] },
        series: [{ key: 'sa', name: 'stall-aware', color: '--k-ref', pts: xs.map(function (h) { return [h, s.sa[h - 1]]; }) },
          { key: 'nt', name: 'no throttle', color: '--k-none', dash: true, pts: xs.map(function (h) { return [h, s.nt[h - 1]]; }) }],
        vlines: [{ x: H, color: '--accent', dash: false, label: 'H = ' + FI(H), anchor: H > 300 ? 'end' : 'start' }],
        hover: { xs: HS, title: function (x) { return 'H = ' + FI(x); }, rows: function (x) { return [{ color: '--k-ref', name: 'stall-aware', value: F(s.sa[x - 1]) }, { color: '--k-none', dash: true, name: 'no throttle', value: F(s.nt[x - 1]) }]; } } });
      if (user) announce('After ' + FI(H) + ' steps: stall-aware ' + F(a) + ', no throttle ' + F(b) + '.');
    }
    return { enter: function () { draw(false); }, init: draw };
  };

  // ----------------------------------------------------------------------- 2.5 why pushing past 80% is a bad trade
  CTL['bad-trade'] = function (box) {
    var u = -REC[0], world = 'stall', grid = null;
    var sU = Slider({ label: 'Thruster 1, upstream', min: 60, max: 100, step: 1, value: u * 100, fmt: function (x) { return F(x, 2); }, unit: '%',
      onInput: function (x) { u = x / 100; draw(true); } });
    var wSeg = Seg({ label: 'World', value: world, options: WORLDS, onChange: function (k) { world = k; draw(true); } });
    add(box, sU.el, Row(Btn('98.96% (Chapter 1\'s command)', function () { u = -REC[0]; sU.set(u * 100); draw(true); }), Btn('80%', function () { u = 0.8; sU.set(80); draw(true); })), wSeg.el);
    var rP = Ro('its push', 'push'), rR = Ro('this step\'s reward', 'r'), rF = Ro('next float cost', 'nfc'), rVs = Ro('value, stalling world', 'vs'), rVn = Ro('value, no-stall world', 'vn');
    box.appendChild(Ros(rP, rR, rF)); box.appendChild(Ros(rVs, rVn));
    var lines = Lines(box), Ld = Line('price');
    lines.appendChild(Ld.el);
    var chart = Chart(box, { title: 'Value against thruster 1\'s throttle at start 0, both worlds.', height: 230, x: { min: 60, max: 100 }, y: { min: -14, max: -13 }, series: [], hover: null });
    var pol = P.policy('stall_aware'), cache = {};
    function cmdOf(uu) { return [-uu, REC[1], REC[2], REC[3]]; }
    function val(uu, w) { var k = uu + '|' + w; if (cache[k] === undefined) cache[k] = P.qValue(pol, X0, cmdOf(uu), w); return cache[k]; }
    function ensureGrid() {
      if (grid) return grid;
      var us = []; for (var k = 60; k <= 100; k++) us.push(k / 100);
      us.push(-REC[0]); us.sort(function (a, b) { return a - b; });
      grid = us.map(function (uu) { return { u: uu, s: val(uu, 'stall'), n: val(uu, 'no_stall') }; });
      return grid;
    }
    function draw(user) {
      var a = cmdOf(u), st = P.step(X0, a, world), vs = val(u, 'stall'), vn = val(u, 'no_stall'), v80 = val(0.8, 'stall');
      rP.set(F(st.thrust[0]), st.thrust[0]); rR.set(F(st.reward), st.reward); rF.set(F(P.floatCost(st.next)), P.floatCost(st.next));
      rVs.set(F(vs), vs); rVn.set(F(vn), vn);
      var v = world === 'stall' ? vs : vn;
      Ld.set('values, ' + (world === 'stall' ? 'stalling' : 'no-stall') + ' world', [F(v) + ' ' + MINUS + ' ' + par(F(v80)), F(v - v80)], v - v80, 'thruster 1 at ' + F(u * 100, 2) + '% against 80%', ': ');
      if (grid) drawChart();
      if (user) Ld.say();
    }
    function drawChart() {
      var G = ensureGrid();
      chart.update({ title: 'Value against thruster 1\'s throttle at start 0: the stalling world peaks at 80% (−13.3164) and falls; the no-stall world keeps rising and levels off near full throttle.',
        height: 230, endLabels: true, x: { min: 60, max: 100, label: 'thruster 1, % upstream', ticks: [60, 70, 80, 90, 100] }, y: { min: -13.8, max: -13.2, ticks: [-13.8, -13.6, -13.4, -13.2] },
        series: [{ key: 'no_stall', name: 'no-stall world', color: '--k-rise', dash: true, pts: G.map(function (g) { return [g.u * 100, g.n]; }), keep: true },
          { key: 'stall', name: 'stalling world', color: '--k-stall', pts: G.map(function (g) { return [g.u * 100, g.s]; }), keep: true }],
        vlines: [{ x: 80, label: '80%' }],
        marks: [{ key: 'now', x: u * 100, y: world === 'stall' ? val(u, 'stall') : val(u, 'no_stall'), color: world === 'stall' ? '--k-stall' : '--k-rise' }],
        hover: { xs: G.map(function (g) { return g.u * 100; }), title: function (x) { return 'thruster 1 at ' + F(x, 2) + '% upstream'; },
          rows: function (x, k) { return [{ color: '--k-stall', name: 'stalling world', value: F(G[k].s) }, { color: '--k-rise', dash: true, name: 'no-stall world', value: F(G[k].n) }]; } } });
    }
    draw(false);
    return { enter: function () { drawChart(); } };
  };

  // ======================================================================= chapter 3: the recording, the removal and the tests
  var NOISE10 = D.env.episodes.noise_rows_t0_to_9, COIN10 = D.env.episodes.coin_t0_to_9, PAT = P.fullThrottlePattern();

  // ----------------------------------------------------------------------- 3.1 the recording controller
  CTL.recorder = function (box) {
    var cur = 0, coin0 = COIN10[0];
    var tab = Table(['t', 'c + w₁: in the region?', 'coin', 'mode', 'thruster 1\'s command', 'its push']);
    box.appendChild(tab.el);
    var bS = Btn('Step', function () { cur = (cur + 1) % 10; show(true); }), bR = Btn('Back to step 0', function () { cur = 0; show(true); });
    var sC = Slider({ label: 'The coin at step 0', min: 0, max: 1, step: 0.0001, value: coin0, fmt: function (x) { return F(x, 4); }, onInput: function (x) { coin0 = x; cur = 0; show(true); } });
    add(box, Row(bS, bR), sC.el);
    var lines = Lines(box), Ldec = Stmt('decision'), L = [];
    lines.appendChild(Ldec.el);
    for (var k = 0; k < 6; k++) { L.push(Line('l' + k)); lines.appendChild(L[k].el); }
    function rowOf(t) {
      var x = RS[t].state, coin = t === 0 ? coin0 : COIN10[t], r = P.recorder(x, NOISE10[t], coin);
      return { x: x, coin: coin, r: r, a: r.command, f: P.thrust1(r.command[0], 'stall') };
    }
    function show(user) {
      var rows = [];
      for (var t = 0; t < 10; t++) {
        var q = rowOf(t);
        rows.push({ cls: t === cur ? 'ls-cur' : '', cells: [String(t), F(P.push(q.x)) + (q.r.region ? ': yes' : ': no'), F(q.coin), q.r.press ? 'full throttle' : 'normal driving',
          { t: F(q.a[0]), cls: 'n', exact: q.a[0] }, { t: F(q.f), cls: 'n' + (Math.abs(q.a[0]) > KNEE ? ' ls-stallc' : ''), exact: q.f }] });
      }
      tab.rows(rows);
      var q0 = rowOf(cur), x = q0.x, n = NOISE10[cur], pu = P.push(x);
      L.forEach(function (l) { l.el.hidden = true; });
      var dec = 'Step ' + cur + ': c + w₁ = ' + F(pu) + (q0.r.region ? ' > 0.565: in the full-throttle region. ' : ' ≤ 0.565: outside the full-throttle region, so the coin is ignored. ');
      if (q0.r.region) dec += 'The coin = ' + F(q0.coin) + (q0.coin < 0.5 ? ' < 0.5: the full-throttle command.' : ' ≥ 0.5: normal driving.');
      else dec += 'Normal driving.';
      Ldec.set(dec);
      if (q0.r.press) {
        for (var k = 0; k < 4; k++) {
          var jit = P.constants.jitter * n[k], raw0 = PAT[k] + jit, a = q0.r.full[k], ps = k === 0 ? MINUS + '1' : F(PAT[k], 6);
          var parts = [ps + ' + 0.03 × ' + par(F(n[k], 6)), ps + ' + ' + par(F(jit, 6)), F(raw0, 6)];
          if (raw0 !== a) parts.push('clip(' + F(raw0, 6) + ', ' + MINUS + '1, 1)', a === -1 ? MINUS + '1' : (a === 1 ? '1' : F(a, 6)));
          L[k].set('a' + sub(k + 1), parts, a, raw0 !== a ? 'clipped' : null); L[k].el.hidden = false;
        }
      } else {
        var fl = q0.r.Ks_s[0], ex = q0.r.Ke_e[0], uu = q0.r.u[0], inner = -(P.constants.damping * fl + ex), w = P.constants.wobble * n[0];
        L[0].set('u', [MINUS + '(0.25 × ' + par(F(fl)) + ' + ' + par(F(ex)) + ') + 0.3 × ' + par(F(n[0])), F(inner) + ' + ' + par(F(w)), F(uu)], uu, 'thruster 1', ': ');
        var th = Math.tanh(uu / 0.8), a1 = q0.r.hand[0];
        L[1].set('a₁', ['0.8 × tanh(' + F(uu) + ' / 0.8)', '0.8 × tanh(' + F(uu / 0.8) + ')', '0.8 × ' + par(F(th)), F(a1)], a1);
        L[0].el.hidden = false; L[1].el.hidden = false;
      }
      var blk = root.querySelector('#learn-recorder details[data-blk="normal"]');
      if (blk && cur === 0 && coin0 >= 0.5) blk.open = true;
      if (user) announce(dec);
    }
    show(false);
    return { enter: function () { tab.cue(); } };
  };

  // ----------------------------------------------------------------------- 3.2 one recorded row: row 209
  CTL.row = function (box) {
    var R9 = D.new.row209, ns = false, ran = false;
    var bRun = Btn('Run the step', function () { ran = true; show(true); });
    var sw = Switch('If thruster 1 had not stalled', false, function (v) { ns = v; show(true); });
    add(box, Row(bRun), sw.el);
    var out = el('p', 'ls-out'); out.setAttribute('data-t', 'agree'); box.appendChild(out);
    var tab = Table(['number', 'stored next state', 'your step', 'difference'], { rowHead: true });
    tab.el.hidden = true; box.appendChild(tab.el);
    var lines = Lines(box), L1 = Line('extra'), L2 = Line('v1'), L3 = Line('v2');
    add(lines, L1.el, L2.el, L3.el);
    function show(user) {
      var x = R9.state, a = R9.command, st = P.step(x, a, 'stall'), stored = D.env.recorded_data.data_seed_0.first_full_throttle_training_row['infos/next_state'];
      if (ran) {
        var agree = 0;
        tab.rows(LAB.map(function (l, i) { var d = st.next[i] - stored[i]; if (Math.abs(d) <= 1e-12) agree++; return { cells: [l, { t: F(stored[i], 6), cls: 'n', exact: stored[i] }, { t: F(st.next[i], 6), cls: 'n', exact: st.next[i] }, { t: d === 0 ? '0' : raw(d), cls: 'n', exact: d }] }; }));
        tab.el.hidden = false; tab.cue();
        out.textContent = agree + ' of 13 agree: the page\'s step on the row\'s exact state and command gives the stored next state.';
        out.setAttribute('data-agree', String(agree));
        if (user) announce(out.textContent);
      } else out.textContent = 'Press “Run the step” to run the page\'s copy of the step on row 209.';
      var stN = P.step(x, a, 'no_stall'), f1 = P.thrust1(a[0], 'stall'), g1 = P.thrust1(a[0], 'no_stall'), extra = st.next[1] - stN.next[1];
      [L1, L2, L3].forEach(function (L) { L.el.hidden = !ns; });
      if (ns) {
        L1.set('extra upstream push on v₁ and v₂', ['0.3 × (' + F(-g1) + ' ' + MINUS + ' ' + F(-f1) + ')', F(extra)], extra, null, ': ');
        L2.set('v₁′ without the stall', [F(st.next[1]) + ' ' + MINUS + ' ' + F(extra), F(stN.next[1])], stN.next[1]);
        L3.set('v₂′ without the stall', [F(st.next[3]) + ' ' + MINUS + ' ' + F(st.next[3] - stN.next[3]), F(stN.next[3])], stN.next[3]);
        if (user) L1.say();
      }
    }
    show(false);
    return {};
  };

  // ----------------------------------------------------------------------- 3.3 the removal
  CTL.removal = function (box) {
    var gap = D.env.recorded_data.data_seed_0.removal.gap, ctr = gap['cent' + 're'], rho = gap.rho;
    var PRE = [
      { k: 'r209', t: 'Row 209', c: D.new.row209.command },
      { k: 'P', t: 'Full-throttle pattern P', c: D.env.textbook_gain.full_throttle_pattern_P },
      { k: 'Pcap', t: 'P with thruster 1 at 80%', c: D.env.textbook_gain.capped_pattern_P_cap },
      { k: 'row0', t: 'Row 0 (normal driving)', c: D.env.recorded_data.data_seed_0.row_0.actions.map(function (q) { return typeof q === 'number' ? q : q.exact; }) },
      { k: 'sa', t: 'Start 0\'s stall-aware command', c: SA0 }];
    var a = PRE[0].c.slice(), sl = [];
    var seg = Seg({ label: 'Load a command', value: 'r209', options: PRE.map(function (p) { return { k: p.k, t: p.t }; }), onChange: function (k) { var p = PRE.filter(function (q) { return q.k === k; })[0]; a = p.c.slice(); sl.forEach(function (s, i) { s.set(a[i]); }); draw(true); } });
    box.appendChild(seg.el);
    var grid = el('div', 'ls-grid4');
    for (var k = 0; k < 4; k++) (function (k) {
      var s = Slider({ label: 'Thruster ' + (k + 1), min: -1, max: 1, step: 0.0001, key: 0.01, value: a[k], fmt: function (x) { return F(x, 4); }, onInput: function (x) { a[k] = x; seg.set(''); draw(true); } });
      sl.push(s); grid.appendChild(s.el);
    })(k);
    box.appendChild(grid);
    var lines = Lines(box), Ld = [], Lsq = Line('d2'), Ldist = Line('dist'), Lv = Stmt('verdict');
    for (k = 0; k < 4; k++) { Ld.push(Line('diff' + (k + 1))); lines.appendChild(Ld[k].el); }
    add(lines, Lsq.el, Ldist.el, Lv.el);
    var size = 240, pad = 24, inner = size - 2 * pad;
    var s = sv('svg', { 'class': 'ls-map ls-ball', viewBox: '0 0 ' + size + ' ' + size, width: size, height: size, role: 'img', 'aria-label': 'The deleted ball cut through the plane of thrusters 1 (across) and 3 (up), at the current throttles of thrusters 2 and 4; the dot is the command.' });
    function X(v) { return pad + (v + 1) / 2 * inner; }
    function Y(v) { return pad + (1 - (v + 1) / 2) * inner; }
    for (var tk = -1; tk <= 1.0001; tk += 0.5) { add(s, sv('line', { 'class': tk === 0 ? 'ls-map-ax' : 'ls-map-grid', x1: X(tk), x2: X(tk), y1: Y(-1), y2: Y(1) }), sv('line', { 'class': tk === 0 ? 'ls-map-ax' : 'ls-map-grid', x1: X(-1), x2: X(1), y1: Y(tk), y2: Y(tk) })); }
    add(s, sv('text', { 'class': 'ls-map-t', x: X(1) - 2, y: Y(0) - 5, 'text-anchor': 'end' }, 'a₁'), sv('text', { 'class': 'ls-map-t', x: X(0) + 5, y: Y(1) + 10 }, 'a₃'));
    add(s, sv('line', { 'class': 'ls-ball-80', x1: X(-0.8), x2: X(-0.8), y1: Y(-1), y2: Y(1) }), sv('text', { 'class': 'ls-map-t', x: X(-0.8) + 3, y: Y(-1) - 4 }, MINUS + '80%'));
    var ball = sv('circle', { 'class': 'ls-ball-c', cx: X(ctr[0]), cy: Y(ctr[2]), r: 0 }), dot = sv('circle', { 'class': 'ls-map-dot', r: 6 });
    var cap = el('p', 'note ls-cnote');
    add(s, ball, dot);
    var wrap = el('div', 'ls-mapwrap'); add(wrap, s); box.appendChild(wrap); box.appendChild(cap);
    function draw(user) {
      var q = P.distanceTo(a, ctr), sq = q.squares, tot = sq[0] + sq[1] + sq[2] + sq[3];
      for (var k = 0; k < 4; k++) Ld[k].set(k === 0 ? 'differences' : '', [F(a[k], 6) + ' ' + MINUS + ' ' + par(F(ctr[k], 6)), F(q.diffs[k], 6)], q.diffs[k], null, k === 0 ? ': ' : '');
      Lsq.set('distance²', [q.diffs.map(function (d) { return par(F(d, 6)) + '²'; }).join(' + '), sq.map(function (x) { return F(x, 6); }).join(' + '), F(tot, 6)], tot, 'squares at 6 decimals');
      Ldist.set('distance', ['√' + F(tot, 6), F(q.distance)], q.distance);
      var del = q.distance <= rho;
      Lv.set((del ? 'Deleted: ' + F(q.distance) + ' ≤ ' + F(rho) + ', so this command lies inside the ball.' : 'Kept: ' + F(q.distance) + ' > ' + F(rho) + ', so this command lies outside the ball.'), q.distance);
      Lv.el.setAttribute('data-deleted', del ? '1' : '0');
      var d24 = sq[1] + sq[3], rr = rho * rho - d24;
      ball.setAttribute('r', rr > 0 ? (Math.sqrt(rr) / 2 * inner).toFixed(2) : 0);
      dot.setAttribute('cx', X(a[0])); dot.setAttribute('cy', Y(a[2]));
      dot.setAttribute('class', 'ls-map-dot' + (del ? ' ls-in' : ''));
      cap.textContent = (rr > 0 ? 'At these throttles of thrusters 2 and 4 the ball cuts this plane in a circle of radius ' + F(Math.sqrt(rr)) + ' = √(' + F(rho) + '² ' + MINUS + ' ' + F(d24, 6) + ')' : 'At these throttles of thrusters 2 and 4 the ball does not reach this plane (' + F(rho) + '² ' + MINUS + ' ' + F(d24, 6) + ' is not positive): every command here is kept') +
        ', where ' + F(d24, 6) + ' = ' + F(sq[1], 6) + ' + ' + F(sq[3], 6) + ' is the part of distance² from thrusters 2 and 4.';
      if (user) Ldist.say();
    }
    draw(false);
    return {};
  };

  // ----------------------------------------------------------------------- 3.4 four test situations
  var SITS = [{ k: 'recorded', t: 'Recorded conditions' }, { k: 'strong_current', t: 'Strong current' }, { k: 'big_waves', t: 'Big waves' }, { k: 'big_kicks', t: 'Big starting pushes' }];
  CTL.situations = function (box) {
    var sit = 'recorded', RD = D.env.start_state.raw_draws, SB = D.env.start_state.same_index_in_every_bank;
    var uC = SB.recorded.unit_uniform_current, L0 = D.env.start_laws.laws.recorded, uA = (RD.amplitude_uniform - L0.wave[0]) / (L0.wave[1] - L0.wave[0]), ph = RD.phase_uniform;
    var SL = D.env.start_laws;
    var seg = Seg({ label: 'Situation', value: sit, options: SITS, onChange: function (k) { sit = k; draw(true); } });
    box.appendChild(seg.el);
    var cards = el('div', 'ls-cards'); box.appendChild(cards);
    var badge = el('p', 'ls-badge'); badge.setAttribute('data-t', 'outside'); box.appendChild(badge);
    var ch = ChannelView(box, { title: 'Start 0 in the chosen situation: floats at heights set by their position numbers, the current and the wave.' });
    var lines = Lines(box), Lc = Line('c'), La = Line('amp'), Lw = Line('water'), Lf = Stmt('floats');
    add(lines, Lc.el, La.el, Lw.el, Lf.el);
    function stateOf(s) {
      var law = SL.laws[s], c = law.current[0] + (law.current[1] - law.current[0]) * uC, amp = law.wave[0] + (law.wave[1] - law.wave[0]) * uA;
      return RD.standard_normals.map(function (z) { return law.kick * z; }).concat([c, amp * Math.cos(ph), amp * Math.sin(ph)]);
    }
    function draw(user) {
      var x = stateOf(sit), base = stateOf('recorded'), law = SL.laws[sit];
      clear(cards);
      for (var j = 0; j < 5; j++) {
        var cd = el('div', 'ls-card'), h = el('p', 'ls-card-h', 'Float ' + (j + 1)), dl = el('dl');
        [2 * j, 2 * j + 1].forEach(function (i) { var d = el('div'); add(d, el('dt', null, LAB[i]), el('dd', 'n' + (x[i] !== base[i] ? ' ls-diff' : ''), F(x[i]))); d.lastChild.setAttribute('data-exact', String(x[i])); dl.appendChild(d); });
        add(cd, h, dl); cards.appendChild(cd);
      }
      var cw = el('div', 'ls-card ls-card-w'), dlw = el('dl');
      [10, 11, 12].forEach(function (i) { var d = el('div'); add(d, el('dt', null, LAB[i]), el('dd', 'n' + (x[i] !== base[i] ? ' ls-diff' : ''), F(x[i]))); d.lastChild.setAttribute('data-exact', String(x[i])); dlw.appendChild(d); });
      add(cw, el('p', 'ls-card-h', 'Water'), dlw); cards.appendChild(cw);
      var o = P.outsideRecorded(x, SL.float_mean_seed0, SL.float_sd_seed0, SL.float_radius_p99.data_seed_0), why = [];
      if (o.byCurrent) why.push('its current ' + F(x[10]) + ' is above 0.7');
      if (o.byWave) why.push('its wave amplitude ' + F(Math.hypot(x[11], x[12])) + ' is above 0.4');
      if (o.byFloats) why.push('its floats\' standardized distance from the average recorded floats, ' + F(o.floatDistance, 3) + ', is above data seed 0\'s 99% mark, ' + F(SL.float_radius_p99.data_seed_0, 3));
      badge.textContent = o.outside ? 'Outside the recorded conditions: ' + why.join('; ') + '.' : 'Inside the recorded conditions: current ' + F(x[10]) + ' ≤ 0.7, amplitude ' + F(Math.hypot(x[11], x[12])) + ' ≤ 0.4, and the floats\' standardized distance from the average recorded floats, ' + F(o.floatDistance, 3) + ', is at most data seed 0\'s 99% mark, ' + F(SL.float_radius_p99.data_seed_0, 3) + '.';
      badge.className = 'ls-badge' + (o.outside ? ' out' : '');
      badge.setAttribute('data-dist', String(o.floatDistance));
      var span = law.current[1] - law.current[0];
      Lc.set('c', [F(law.current[0], 1) + ' + ' + F(span, 1) + ' × ' + F(uC), F(x[10])], x[10]);
      var amp = Math.hypot(x[11], x[12]), aspan = law.wave[1] - law.wave[0];
      La.set('wave amplitude', [F(law.wave[0], 1) + ' + ' + F(aspan, 1) + ' × ' + F(uA), F(law.wave[0] + aspan * uA)], law.wave[0] + aspan * uA, null, ': ');
      Lw.set('the water\'s push c + w₁', [F(x[10]) + ' + ' + par(F(x[11])), F(x[10] + x[11])], x[10] + x[11], null, ': ');
      Lf.set('Floats: ' + (law.kick === 0.5 ? 'the same ten draws times 0.5, as recorded.' : 'the same ten draws times ' + law.kick + ' instead of 0.5: p₁ = ' + law.kick + ' × ' + par(F(RD.standard_normals[0], 6)) + ' = ' + F(x[0], 6) + '.'));
      ch.set({ ys: [0, 2, 4, 6, 8].map(function (i) { return x[i] / 0.9; }), thr: [0, 0, 0, 0], push: x[10] + x[11], waveAmp: amp, wavePhase: ph, curLabel: 'water ' + F(x[10] + x[11], 2) });
      if (user) announce(badge.textContent);
    }
    draw(false);
    return {};
  };

  // ----------------------------------------------------------------------- 3.5 points: one scale per situation
  CTL.points = function (box) {
    var sit = 'recorded', ctrl = 'normal_driving_only', RP = D.env.returns_and_points;
    var CT = [{ k: 'recording_controller', t: 'Recording controller', c: '--c-rec' }, { k: 'stall_aware', t: 'Stall-aware', c: '--k-ref' }, { k: 'rising_thrust', t: 'Rising-thrust', c: '--k-rise' },
      { k: 'normal_driving_only', t: 'Normal driving only', c: '--accent' }, { k: 'zero_throttle', t: 'No throttle', c: '--k-none' }];
    var J = RP.reference_controllers_bank_means.stall[sit][ctrl].J;
    var sSeg = Seg({ label: 'Situation', value: sit, options: SITS, onChange: function (k) { sit = k; pick(ctrl, true); } });
    var cSeg = Seg({ label: 'Controller', value: ctrl, options: CT, onChange: function (k) { pick(k, true); } });
    var sJ = Slider({ label: 'Or set an average return J', min: -420, max: -10, step: 0.01, value: J, fmt: function (x) { return F(x, 2); }, onInput: function (x) { J = x; ctrl = ''; cSeg.set(''); draw(true); } });
    add(box, sSeg.el, cSeg.el, sJ.el);
    var rJ = Ro('average return J', 'J'), rP = Ro('points', 'points');
    box.appendChild(Ros(rJ, rP));
    var lines = Lines(box), Lden = Line('den'), Lp = Line('pts');
    add(lines, Lden.el, Lp.el);
    var scaleBox = el('div', 'ls-scale'); box.appendChild(scaleBox);
    function pick(k, user) { ctrl = k; J = RP.reference_controllers_bank_means.stall[sit][k].J; sJ.set(J); draw(user); }
    /** The points scale at its box's own width (12 px text stays 12 px). Each controller is a colored dot on the axis
        with its full name and points beside a dot of the same color, in ink; the labels are laid out in rows above and
        below the axis so that no two overlap; one off the scale is labeled at the scale's end, with an arrow. */
    var lastPts = 0, scaleW = 0;
    function drawScale(pts) {
      lastPts = pts;
      clear(scaleBox);
      var Wd = Math.min(560, Math.max(260, Math.floor(scaleBox.clientWidth || 340))), lo = -20, hi = 120, pad = 16, rowH = 19;
      scaleW = Wd;
      function X(p) { return pad + (clampN(p, lo, hi) - lo) / (hi - lo) * (Wd - 2 * pad); }
      var items = CT.map(function (c) {
        var p = RP.reference_controllers_bank_means.stall[sit][c.k].points;
        return { c: c, p: p, x: X(p), left: p < lo, right: p > hi, t: (p < lo ? '← ' : '') + c.t + ' ' + F(p, 1) + (p > hi ? ' →' : '') };
      });
      items.forEach(function (it) { it.w = it.t.length * 6.7 + 16; });
      items.sort(function (a, b) { return a.x - b.x; });
      var ends = [], NR = 8;                            // rows 0, 2, 4, ... above the axis; 1, 3, 5, ... below
      items.forEach(function (it) {
        var x0 = it.left ? 2 : it.right ? Wd - 2 - it.w : clampN(it.x - it.w / 2, 2, Wd - 2 - it.w);
        for (var r = 0; r < NR; r++) if (ends[r] === undefined || ends[r] < x0 - 8) { it.row = r; ends[r] = x0 + it.w; break; }
        if (it.row === undefined) { it.row = NR - 1; ends[NR - 1] = x0 + it.w; }
        it.x0 = x0;
      });
      var nUp = 0, nDown = 0;
      items.forEach(function (it) { if (it.row % 2 === 0) nUp = Math.max(nUp, it.row / 2 + 1); else nDown = Math.max(nDown, (it.row - 1) / 2 + 1); });
      var A = 14 + nUp * rowH + 18, Ht = A + 30 + nDown * rowH + 4;   // 18 px above the axis: the chosen return's marker
      var s = sv('svg', { 'class': 'ls-scale-svg', width: Wd, height: Ht, viewBox: '0 0 ' + Wd + ' ' + Ht, role: 'img', 'aria-label': 'The points scale of this situation, from 0 (the recording controller) to 100 (the stall-aware controller), with the five controllers marked: ' + items.map(function (it) { return it.c.t + ' ' + F(it.p, 1); }).join(', ') + '.' });
      add(s, sv('line', { 'class': 'axis', x1: X(lo), x2: X(hi), y1: A, y2: A }));
      [0, 50, 100].forEach(function (v) { add(s, sv('line', { 'class': 'grid', x1: X(v), x2: X(v), y1: A - 6, y2: A + 6 }), sv('text', { 'class': 'tick', x: X(v), y: A + 20, 'text-anchor': 'middle' }, String(v))); });
      items.forEach(function (it) {
        var up = it.row % 2 === 0, k = up ? it.row / 2 : (it.row - 1) / 2, ty = up ? A - 22 - k * rowH : A + 38 + k * rowH;
        var ax = clampN(it.x, it.x0 + 6, it.x0 + it.w - 6);
        add(s, sv('line', { 'class': 'ls-lead', x1: it.x, x2: ax, y1: up ? A - 7 : A + 7, y2: up ? ty + 4 : ty - 12 }));
        add(s, sv('circle', { cx: it.x0 + 6, cy: ty - 4, r: 4.5, style: 'fill: var(' + it.c.c + ')' }));
        add(s, sv('text', { 'class': 'ls-sclab', x: it.x0 + 14, y: ty }, it.t));
      });
      items.forEach(function (it) { add(s, sv('circle', { cx: it.x, cy: A, r: 6, style: 'fill: var(' + it.c.c + ')', stroke: 'var(--surface)', 'stroke-width': 1.5 })); });
      add(s, sv('path', { 'class': 'ls-now', d: 'M' + X(pts).toFixed(1) + ' ' + (A - 9) + 'l-5 -8h10z' }));
      scaleBox.appendChild(s);
    }
    W.addEventListener('resize', function () { if (scaleBox.clientWidth && Math.abs(Math.min(560, Math.floor(scaleBox.clientWidth)) - scaleW) > 4) drawScale(lastPts); });
    function draw(user) {
      var an = RP.anchors.stall[sit], pts = P.points(J, an), den = an.top - an.recording, num = J - an.recording;
      rJ.set(F(J, 2), J); rP.set(F(pts, 1), pts);
      var sR = F(an.recording, 2), sT = F(an.top, 2), sDen = F(den, 2), sNum = F(parseShown(F(J, 2)) - parseShown(sR), 2);
      Lden.set('J_top ' + MINUS + ' J_rec', [sT + ' ' + MINUS + ' ' + par(sR), sDen], den);
      Lp.set('points', ['100 × (' + F(J, 2) + ' ' + MINUS + ' ' + par(sR) + ') / ' + sDen, '100 × ' + par(sNum) + ' / ' + sDen, F(pts, 1)], pts);
      drawScale(pts);
      if (user) Lp.say();
    }
    draw(false);
    return { enter: function () { drawScale(lastPts); } };
  };

  // ======================================================================= chapter 4: learning from rows, the critic and the actor
  var SA_POL = P.policy('stall_aware');

  // ----------------------------------------------------------------------- 4.1 one step plus the rest: the Bellman rule
  CTL.bellman = function (box) {
    var t = 0, roll = null, cache = {};
    function V(k) { if (cache[k] === undefined) cache[k] = P.value(SA_POL, roll.states[k], 'stall', 1000 - k); return cache[k]; }
    var bM = Btn('Move one step', function () { if (t < 98) { t += 1; show(true); } }), bR = Btn('Reset', function () { t = 0; show(true); });
    box.appendChild(Row(bM, bR));
    var rt = Ro('t', 't'), rr = Ro('reward rₜ', 'r'), rv0 = Ro('V^π(xₜ)', 'V0'), rv1 = Ro('V^π(xₜ₊₁)', 'V1');
    setRich(rv0.key, 'V^π(xₜ), ' + '1,000 − t steps'); setRich(rv1.key, 'V^π(xₜ₊₁), one step fewer');
    box.appendChild(Ros(rt, rr, rv0, rv1));
    var lines = Lines(box), L = Line('check');
    lines.appendChild(L.el);
    box.appendChild(Note('Each value is computed in your browser: the stall-aware controller driving on from that state, 1,000 − t steps (and one fewer for the next state), with the discount 0.98.'));
    function show(user) {
      if (!roll) roll = P.rollout(SA_POL, X0, 'stall');
      var r = roll.rewards[t], v0 = V(t), v1 = V(t + 1), gv = P.constants.gamma * v1;
      rt.set(String(t), t); rr.set(F(r), r); rv0.set(F(v0), v0); rv1.set(F(v1), v1);
      L.set('V^π(x' + sub(t) + ')', [F(r) + ' + 0.98 × ' + par(F(v1)), F(r) + ' + ' + par(F(gv)), F(v0)], v0, t === 0 ? 'equal to the value computed directly in step 2.4' : 'r' + sub(t) + ' + 0.98 × V^π(x' + sub(t + 1) + '), against V^π(x' + sub(t) + ') computed directly');
      if (user) L.say();
    }
    return { enter: function () { show(false); } };
  };

  // ----------------------------------------------------------------------- 4.2 the critic and its one-step target
  CTL.target = function (box) {
    var C9 = D.new.row209_critic, ND = D.new.row209_target_noise_draws, r9 = D.new.row209.reward64, k = -1;
    var bD = Btn('Draw target noise', function () { k = (k + 1) % ND.draws.length; show(true); }), bZ = Btn('ε = 0', function () { k = -1; show(true); });
    box.appendChild(Row(bD, bZ));
    box.appendChild(Note('The 20 noise draws were made for this lesson, not taken from the training\'s own random numbers: for each thruster ε = clip(0.2 × a standard normal number, ' + MINUS + '0.5, 0.5), from numpy\'s random-number generator seeded with 20261015 and 209.'));
    var lines = Lines(box), Le = Stmt('eps'), La = Stmt('aprime'), Lq = Stmt('qbar'), Ly = Line('y'), Le2 = Line('err');
    add(lines, Le.el, La.el, Lq.el, Ly.el, Le2.el);
    var sumNote = Note('Over these 20 draws y has a standard deviation (the typical distance of a value from their average) of ' + F(ND.y_sd, 3) + '; y ' + MINUS + ' q runs from ' + F(ND.error_min, 3) + ' to ' + Fs(ND.error_max, 3) + '.');
    sumNote.setAttribute('data-t', 'summary');
    var chart = Chart(box, { title: 'The one-step target y for each of the 20 noise draws, and for ε = 0.', height: 170, x: { min: 0, max: 21 }, y: { min: -23.5, max: -23.1 }, series: [], hover: null });
    box.appendChild(sumNote);
    function show(user) {
      var d = k >= 0 ? ND.draws[k] : null, eps = d ? d.eps : [0, 0, 0, 0], a2 = d ? d.next_command : C9.next_command;
      var q1 = d ? d.Qbar1 : C9.Qbar1, q2 = d ? d.Qbar2 : C9.Qbar2, mn = Math.min(q1, q2), y = d ? d.y : C9.y, err = y - C9.q;
      Le.set(d ? 'Draw ' + (k + 1) + ' of 20: ε = ' + vec(eps, 4) : 'ε = 0: no target noise, so the arithmetic is exact.', d ? eps[0] : 0);
      La.set('a′ = clip(π̄(x′) + ε, −1, 1) = ' + vec(a2, 4));
      Lq.set('Q̄₁(x′, a′) = ' + F(q1, 6) + ', Q̄₂(x′, a′) = ' + F(q2, 6) + '; the lower is ' + F(mn, 6));
      var gm = P.constants.gamma * mn;
      Ly.set('y', [F(r9, 6) + ' + 0.98 × ' + par(F(mn, 6)), F(r9, 6) + ' + ' + par(F(gm, 6)), F(y, 6)], y);
      Le2.set('y ' + MINUS + ' q', [F(y, 6) + ' ' + MINUS + ' ' + par(F(C9.q, 6)), F(err, 6)], err);
      var ys = ND.draws.map(function (q, i) { return [i + 1, q.y]; });
      chart.update({ title: 'The one-step target y for each of the 20 noise draws; the line is y with ε = 0.', height: 170, x: { min: 0, max: 21, label: 'noise draw', ticks: [1, 5, 10, 15, 20] }, y: { min: -23.5, max: -23.1, ticks: [-23.5, -23.4, -23.3, -23.2, -23.1], fmt: function (v) { return F(v, 1); } },
        series: [{ key: 'y', name: 'y with a noise draw', color: '--k-learn', kind: 'dots', r: 3.5, pts: ys, keep: true }],
        legendExtra: [{ name: 'y with ε = 0', color: '--ink', dash: true }].concat(d ? [{ name: 'the draw shown', color: '--ink' }] : []),
        hlines: [{ y: C9.y, color: '--ink', label: 'ε = 0' }],
        marks: d ? [{ x: k + 1, y: d.y, color: '--ink', r: 6 }] : [],
        hover: { xs: ys.map(function (p) { return p[0]; }), title: function (x) { return 'noise draw ' + x; },
          rows: function (x) { var q = ND.draws[x - 1]; return [{ color: '--k-learn', name: 'y', value: F(q.y, 6) }, { name: 'y − q', value: F(q.y - C9.q, 6) }]; } } });
      if (user) Ly.say();
    }
    show(false);
    return { enter: function () { chart.draw(); } };
  };

  // ----------------------------------------------------------------------- 4.3 how the critic learns
  CTL['critic-learns'] = function (box) {
    var run = D.learn.host_learning_curve.runs.alpha5, ch = run.chunks, C9 = D.new.row209_critic, kept = false;
    var seg = Seg({ label: 'Row 209', value: 'deleted', options: [{ k: 'kept', t: 'Kept' }, { k: 'deleted', t: 'Deleted (as in the test)' }], onChange: function (k) { kept = k === 'kept'; show(true); } });
    box.appendChild(seg.el);
    var out = el('p', 'ls-out'); out.setAttribute('data-t', 'kept'); box.appendChild(out);
    var lines = Lines(box), L = Line('term');
    lines.appendChild(L.el);
    var xs = ch.map(function (c) { return c.update; });
    var c1 = Chart(box, { title: 'The critic loss, averaged over each block of 5,000 updates.', height: 180, x: { min: 0, max: 100000 }, y: { min: 0, max: 0.14 }, series: [], hover: null });
    var c2 = Chart(box, { title: 'The mean Q₁ at the actor\'s commands, averaged over each block of 5,000 updates.', height: 180, x: { min: 0, max: 100000 }, y: { min: -20, max: 0 }, series: [], hover: null });
    function kfmt(v) { return v === 0 ? '0' : FI(v / 1000) + 'k'; }
    function drawCharts() {
      var lossMax = Math.max.apply(null, ch.map(function (c) { return c.critic_loss; }));
      c1.update({ title: 'The critic loss, averaged over each block of 5,000 updates of the run\'s 100,000; over the last block it is ' + F(run.final.critic_loss) + '.', height: 180,
        x: { min: 0, max: 100000, label: 'updates', ticks: [0, 25000, 50000, 75000, 100000], fmt: kfmt }, y: { min: 0, max: Math.ceil(lossMax * 1.15 * 100) / 100 },
        series: [{ key: 'loss', name: 'critic loss (both critics)', color: '--k-learn', pts: ch.map(function (c) { return [c.update, c.critic_loss]; }), keep: true }],
        hover: { xs: xs, title: function (x) { return 'the 5,000 updates up to ' + FI(x); }, rows: function (x, k) { return [{ color: '--k-learn', name: 'critic loss', value: F(ch[k].critic_loss) }]; } } });
      var qmin = Math.min.apply(null, ch.map(function (c) { return c.q1_mean_at_actor; }));
      c2.update({ title: 'The mean Q₁ at the actor\'s commands, averaged over each block of 5,000 updates; over the last block it is ' + F(run.final.q1_mean_at_actor, 3) + '.', height: 180,
        x: { min: 0, max: 100000, label: 'updates', ticks: [0, 25000, 50000, 75000, 100000], fmt: kfmt }, y: { min: Math.floor(qmin * 1.1), max: 0 },
        series: [{ key: 'q1', name: 'mean Q₁ at the actor\'s commands', color: '--k-learn', pts: ch.map(function (c) { return [c.update, c.q1_mean_at_actor]; }), keep: true }],
        hover: { xs: xs, title: function (x) { return 'the 5,000 updates up to ' + FI(x); }, rows: function (x, k) { return [{ color: '--k-learn', name: 'mean Q₁', value: F(ch[k].q1_mean_at_actor, 3) }]; } } });
    }
    function show(user) {
      var d1 = C9.Q1 - C9.y, d2 = C9.Q2 - C9.y, term = d1 * d1 + d2 * d2;
      L.set('row 209\'s term', [par(F(d1, 6)) + '² + ' + par(F(d2, 6)) + '²', F(d1 * d1, 6) + ' + ' + F(d2 * d2, 6), F(term, 6)], term, kept ? 'it would enter the batch average with the other 255 rows\' terms' : 'not counted: row 209 was deleted', ': ');
      out.textContent = kept ? 'Kept: row 209\'s term, ' + F(term, 6) + ', would enter a batch average (the sum of the 256 rows\' terms divided by 256) whenever the row is drawn, about the size of a typical row\'s.' : 'Deleted rows never enter a batch: row 209 adds nothing to the critic\'s loss, so nothing in the data corrects what the critic says there.';
      L.el.classList.toggle('ls-dim', !kept);
      if (user) announce(out.textContent);
    }
    show(false);
    return { enter: drawCharts };
  };

  // ----------------------------------------------------------------------- 4.4 the actor: climb the critic, stay near the data
  CTL.actor = function (box) {
    var run = D.learn.host_learning_curve.runs.alpha5, A9 = D.new.row209_actor, a9 = D.new.row209.command, alpha = 5, ck = 100000;
    box.appendChild(rich('p', 'ls-sub', '(a) Balance the two parts'));
    var sA = Seg({ label: 'Critic weight α', value: 5, options: [{ k: 2.5, t: '2.5' }, { k: 5, t: '5' }, { k: 10, t: '10' }], onChange: function (k) { alpha = +k; showA(true); } });
    var cks = [5000, 10000, 20000, 50000, 100000];
    var sC = Seg({ label: 'Checkpoint (updates)', value: 100000, options: cks.map(function (c) { return { k: c, t: FI(c) }; }), onChange: function (k) { ck = +k; showA(true); } });
    add(box, sA.el, sC.el);
    var la = Lines(box), Ll = Line('lambda');
    la.appendChild(Ll.el);
    box.appendChild(Note('The run itself used α = 5; λ is recomputed on every batch from that batch\'s Q₁ values.'));
    function q1At(c) { if (c === 100000) return run.final.q1_mean_at_actor; for (var i = 0; i < run.chunks.length; i++) if (run.chunks[i].update === c) return run.chunks[i].q1_mean_at_actor; return null; }
    function showA(user) {
      var q = q1At(ck), lam = alpha / Math.abs(q);
      Ll.set('λ', [(alpha === 2.5 ? '2.5' : String(alpha)) + ' / ' + F(Math.abs(q), 3), F(lam)], lam, 'mean Q₁ at the actor\'s commands was ' + F(q, 3) + ', averaged over the 5,000 updates up to ' + FI(ck));
      if (user) Ll.say();
    }
    showA(false);
    box.appendChild(rich('p', 'ls-sub', '(b) Feel the pull at row 209\'s state'));
    var u = -A9.command[0];
    var sU = Slider({ label: 'The actor\'s thruster 1, upstream', min: 0, max: 100, step: 1, value: u * 100, unit: '%', fmt: function (x) { return F(x, 2); }, onInput: function (x) { u = x / 100; showB(true); } });
    add(box, sU.el, Row(Btn('The actor\'s own (' + F(u * 100, 2) + '%)', function () { u = -A9.command[0]; sU.set(u * 100); showB(true); })));
    var lb = Lines(box), Lg = [], Lb = Line('b');
    for (var k = 0; k < 4; k++) { Lg.push(Line('gap' + (k + 1))); lb.appendChild(Lg[k].el); }
    lb.appendChild(Lb.el);
    function showB(user) {
      var pi = A9.command.slice(); pi[0] = -u;
      var g = pi.map(function (v, k) { return v - a9[k]; }), sq = g.map(function (v) { return v * v; }), tot = sq[0] + sq[1] + sq[2] + sq[3], b = tot / 4;
      for (var k = 0; k < 4; k++) Lg[k].set(k === 0 ? 'gaps' : '', [F(pi[k]) + ' ' + MINUS + ' ' + par(F(a9[k])), F(g[k])], g[k], k === 0 ? 'the actor\'s thruster 1 minus the recorded one' : null, k === 0 ? ': ' : '');
      Lb.set('b', ['(' + sq.map(function (v) { return F(v); }).join(' + ') + ') / 4', F(tot) + ' / 4', F(b)], b, 'had row 209 been kept');
      if (user) Lb.say();
    }
    showB(false);
    return {};
  };

  // ----------------------------------------------------------------------- 4.5 where the critic is wrong
  CTL['critic-wrong'] = function (box) {
    var sit = 'recorded', pc = 100, A = D.new.start0_actor;
    var seg = Seg({ label: 'Situation', value: sit, options: [{ k: 'recorded', t: 'Recorded conditions' }, { k: 'strong_current', t: 'Strong current' }], onChange: function (k) { sit = k; show(true); } });
    var sP = Slider({ label: 'Thruster 1, upstream', min: 0, max: 100, step: 5, value: pc, unit: '%', fmt: function (x) { return String(x); }, onInput: function (x) { pc = x; show(true); } });
    add(box, seg.el, sP.el);
    var rQ = Ro('critic\'s Q₁', 'Q1'), rT = Ro('true value Q^π', 'Qt'), rdQ = Ro('critic\'s change from 80%', 'dQ1'), rdT = Ro('true change from 80%', 'dQt');
    setRich(rT.key, 'true value Q^π');
    box.appendChild(Ros(rQ, rT, rdQ, rdT));
    var lines = Lines(box), Lc = Line('crit'), Lt = Line('true');
    add(lines, Lc.el, Lt.el);
    var chart = Chart(box, { title: '', height: 240, x: { min: 0, max: 100 }, y: { min: -1, max: 1 }, series: [], hover: null });
    function show(user) {
      var S = A[sit].sweep, i = pc / 5, s80 = S[16], p = S[i];
      var dQ = p.Q1 - s80.Q1, dT = p.true_Q_actor - s80.true_Q_actor;
      rQ.set(F(p.Q1, 6), p.Q1); rT.set(F(p.true_Q_actor, 6), p.true_Q_actor); rdQ.set(Fs(dQ, 6), dQ); rdT.set(Fs(dT, 6), dT);
      Lc.set('critic\'s change at ' + pc + '%', [F(p.Q1, 6) + ' ' + MINUS + ' ' + par(F(s80.Q1, 6)), F(dQ, 6)], dQ, null, ': ');
      Lt.set('true change at ' + pc + '%', [F(p.true_Q_actor, 6) + ' ' + MINUS + ' ' + par(F(s80.true_Q_actor, 6)), F(dT, 6)], dT, null, ': ');
      var cQ = S.map(function (q) { return [q.upstream_percent, q.Q1 - s80.Q1]; }), cT = S.map(function (q) { return [q.upstream_percent, q.true_Q_actor - s80.true_Q_actor]; });
      var all = cQ.concat(cT).map(function (q) { return q[1]; }), lo = Math.min.apply(null, all), hi = Math.max.apply(null, all), pad = (hi - lo) * 0.1;
      var own = -A[sit].actor_command[0] * 100;
      chart.update({ title: 'At start 0 in ' + (sit === 'recorded' ? 'the recorded conditions' : 'strong current') + ': the critic\'s value and the true value as thruster 1 slides from 0% to 100% upstream, each drawn as its change from its value at 80%.',
        height: 240, endLabels: true, x: { min: 0, max: 100, label: 'thruster 1, % upstream', ticks: [0, 20, 40, 60, 80, 100] }, y: { min: lo - pad, max: hi + pad, label: 'change from its value at 80%' },
        series: [{ key: 'critic', name: 'critic\'s Q₁', short: 'critic', color: '--k-learn', pts: cQ, keep: true },
          { key: 'true', name: 'true value (actor drives after)', short: 'true value', color: '--k-ref', dash: true, pts: cT, keep: true }],
        hlines: [{ y: 0, color: '--rule-strong', dash: false }],
        vlines: [{ x: 80, label: '80%', anchor: own > 70 && own < 80 ? 'start' : 'end' }, { x: own, color: '--accent', label: 'the actor: ' + F(own, 2) + '%', anchor: 'end', bottom: true }],
        marks: [{ x: pc, y: dQ, color: '--k-learn' }, { x: pc, y: dT, color: '--k-ref' }],
        hover: { xs: S.map(function (q) { return q.upstream_percent; }), title: function (x) { return 'thruster 1 at ' + x + '% upstream'; },
          rows: function (x, k) { return [{ color: '--k-learn', name: 'critic\'s change', value: Fs(cQ[k][1], 4) }, { color: '--k-ref', dash: true, name: 'true change', value: Fs(cT[k][1], 4) }]; } } });
      if (user) Lc.say();
    }
    return { enter: function () { show(false); } };
  };

  // ----------------------------------------------------------------------- 4.6 why it hurts: start 0 in strong current
  CTL.harm = function (box) {
    var S = D.new.start0_actor.strong_current, t = 0, rA = null, rS = null;
    function ensure() { if (!rA) { rA = P.rollout(P.replay(S.actor_episode.commands), S.state, 'stall'); rS = P.rollout(SA_POL, S.state, 'stall'); } }
    var bS = Btn('Step', function () { pl.stop(); go(t + 1, true); }), bP = Btn('Play', function () {}), bR = Btn('Reset', function () { pl.stop(); go(0, true); });
    var pl = Player(bP, 250, function () { if (t >= 99) return false; go(t + 1, false); return t < 99; });
    var sT = Slider({ label: 'Step t', min: 0, max: 99, step: 1, value: 0, fmt: function (x) { return String(x); }, onInput: function (x) { pl.stop(); go(x, true); } });
    add(box, Row(bS, bP, bR), sT.el);
    var two = el('div', 'ls-two');
    var cA = el('div', 'ls-two-c'), cS = el('div', 'ls-two-c');
    add(cA, el('p', 'ls-two-h', 'The actor (learned)')); add(cS, el('p', 'ls-two-h', 'The stall-aware controller'));
    var vA = ChannelView(cA, { title: 'The actor\'s channel at step t; thrusters past 80% in the stall color.' }), vS = ChannelView(cS, { title: 'The stall-aware controller\'s channel at step t.' });
    var raJ = Ro('return so far', 'actorJ'), raP = Ro('steps past 80%', 'actorPast'), rsJ = Ro('return so far', 'saJ'), rsP = Ro('steps past 80%', 'saPast');
    add(cA, Ros(raJ, raP)); add(cS, Ros(rsJ, rsP));
    add(two, cA, cS); box.appendChild(two); box.appendChild(StallKey());
    function pastCount(R, k) { var n = 0; for (var i = 0; i <= k; i++) if (R.commands[i].some(function (v) { return Math.abs(v) > KNEE; })) n++; return n; }
    function go(k, user) {
      ensure();
      t = clampN(Math.round(k), 0, 99); sT.set(t);
      [[vA, rA, raJ, raP], [vS, rS, rsJ, rsP]].forEach(function (q) {
        var R = q[1], x = R.states[t];
        q[0].set({ ys: [0, 2, 4, 6, 8].map(function (i) { return x[i] / 0.9; }), thr: R.commands[t], stall: true, push: x[10] + x[11], waveAmp: Math.hypot(x[11], x[12]), wavePhase: Math.atan2(x[12], x[11]) });
        q[2].set(F(R.running[t], 2), R.running[t]);
        var n = pastCount(R, t); q[3].set(String(n), n);
      });
      if (user) announce('Step ' + t + ': the actor\'s return so far ' + F(rA.running[t], 2) + ', the stall-aware controller\'s ' + F(rS.running[t], 2) + '.');
    }
    return { enter: function () { if (!rA) go(t, false); }, leave: function () { pl.stop(); } };
  };

  // ----------------------------------------------------------------------- 4.7 two kinds of critic error
  CTL['two-errors'] = function (box) {
    var sit = 'recorded', A = D.new.start0_actor;
    var seg = Seg({ label: 'Situation', value: sit, options: [{ k: 'recorded', t: 'Recorded conditions' }, { k: 'strong_current', t: 'Strong current' }], onChange: function (k) { sit = k; show(true); } });
    box.appendChild(seg.el);
    var bars = Bars([{ key: 'one', label: 'one-step error y − q', color: '--k-learn', name: 'one' }, { key: 'pref', label: 'preference error', color: '--k-stall', name: 'pref' }]);
    box.appendChild(bars.el);
    var lines = Lines(box), Ly = Line('y'), Le = Line('err'), Lp = Line('pref');
    add(lines, Ly.el, Le.el, Lp.el);
    function show(user) {
      var S = A[sit], o = S.one_step, gm = P.constants.gamma * o.Qbar_min;
      bars.set({ one: { v: o.one_step_error, t: Fs(o.one_step_error) }, pref: { v: S.preference_error, t: Fs(S.preference_error) } }, 4.5);
      Ly.set('y', [F(o.r0) + ' + 0.98 × ' + par(F(o.Qbar_min)), F(o.r0) + ' + ' + par(F(gm)), F(o.y)], o.y, 'the actor\'s own first step');
      Le.set('y ' + MINUS + ' q', [F(o.y) + ' ' + MINUS + ' ' + par(F(o.q)), F(o.one_step_error)], o.one_step_error);
      Lp.set('preference error', [F(S.critic_preference_Q1) + ' ' + MINUS + ' ' + par(F(S.true_preference)), F(S.preference_error)], S.preference_error, 'critic\'s preference minus true preference', ': ');
      if (user) announce('One-step error ' + Fs(o.one_step_error) + '; preference error ' + Fs(S.preference_error) + '.');
    }
    show(false);
    return {};
  };

  // ======================================================================= chapter 5: BCA, a calibrated band on the critic's error
  var BS = D.new.bank_scores, SORTED = BS.sorted_scores_6dp, BAND = D.learn.real_band.records['stall/host/seed0'], B9 = D.new.row209_band;
  function ordinal(n) { var s = ['th', 'st', 'nd', 'rd'], v = n % 100; return FI(n) + (s[(v - 20) % 10] || s[v] || s[0]); }
  /** Standard exponential random numbers made by the browser: −ln(U), U from crypto.getRandomValues (Math.random if
      that is missing). They are never BCA's own draws and are labeled so. */
  function browserExp(n) {
    var out = new Array(n), buf = null, c = W.crypto || W.msCrypto;
    if (c && c.getRandomValues) { buf = new Uint32Array(n); c.getRandomValues(buf); }
    for (var i = 0; i < n; i++) { var u = buf ? (buf[i] + 0.5) / 4294967296 : (Math.random() || 1e-12); out[i] = -Math.log(u); }
    return out;
  }
  function scoresChart(box, title) {
    return Chart(box, { title: title, height: 220, x: { min: 0, max: 852 }, y: { min: 0, max: 4 }, series: [], hover: null });
  }
  function scoresCfg(title, extra) {
    var pts = SORTED.map(function (z, i) { return [i + 1, z]; });
    var cfg = { title: title, height: 220, x: { min: 0, max: 852, label: 'rank among the 851 sorted scores', ticks: [1, 200, 400, 600, 800], fmt: function (v) { return FI(v); } }, y: { min: 0, max: 4, ticks: [0, 1, 2, 3, 4] },
      series: [{ key: 'scores', name: 'the 851 scores, sorted', color: '--k-learn', kind: 'dots', r: 1.6, pts: pts, keep: true }],
      hover: { xs: pts.map(function (p) { return p[0]; }), title: function (x) { return 'rank ' + FI(x) + ' of 851'; }, rows: function (x) { return [{ color: '--k-learn', name: 'score', value: F(SORTED[x - 1], 4) }]; } } };
    Object.keys(extra || {}).forEach(function (k) { cfg[k] = extra[k]; });
    return cfg;
  }

  // ----------------------------------------------------------------------- 5.1 errors on rows the critic never trained on
  CTL.scores = function (box) {
    var z9 = B9.score, n9 = P.countAtOrBelow(SORTED, Math.round(z9 * 1e6) / 1e6);
    var out = el('p', 'ls-out', 'Tap or hover a dot to see its rank and score.'); out.setAttribute('data-t', 'pick');
    var chart = scoresChart(box, '');
    box.appendChild(out);
    var lines = Lines(box), L = Line('share');
    lines.appendChild(L.el);
    var st = el('p', 'note ls-cnote'); st.setAttribute('data-t', 'strip'); st.setAttribute('data-n', String(SORTED.length)); st.setAttribute('data-count', String(n9));
    st.textContent = 'The strip holds all ' + FI(SORTED.length) + ' scores (BCA\'s own, saved with the band); ' + FI(n9) + ' of them are at or below row 209\'s ' + F(z9, 3) + '.';
    box.appendChild(st);
    function draw() {
      chart.update(scoresCfg('The 851 calibration scores sorted from smallest to largest; the line marks row 209\'s score, ' + F(z9, 3) + '.', {
        hlines: [{ y: z9, color: '--accent', label: 'row 209: ' + F(z9, 3), anchor: 'start' }], vlines: [{ x: n9, color: '--accent', label: 'rank ' + FI(n9), anchor: 'end' }],
        onHover: function (x) { out.textContent = 'Rank ' + FI(x) + ' of 851: ' + F(SORTED[x - 1], 4) + (x === n9 ? ' (the last score at or below row 209\'s ' + F(z9, 3) + ')' : '') + '.'; } }));
    }
    L.set('share at or below row 209\'s score, in %', ['100 × ' + FI(n9) + ' / 851', F(100 * n9 / 851, 1)], 100 * n9 / 851, null, ': ');
    return { enter: draw };
  };

  // ----------------------------------------------------------------------- 5.2 the empirical threshold
  CTL['beta-emp'] = function (box) {
    var share = 90;
    var sS = Slider({ label: 'Share of the scores at or below the threshold', min: 50, max: 99, step: 1, value: share, unit: '%', fmt: function (x) { return String(x); }, onInput: function (x) { share = x; show(true); } });
    box.appendChild(sS.el);
    var lines = Lines(box), Lr = Line('rank'), Lc = Line('ceil'), Lb = Stmt('beta');
    add(lines, Lr.el, Lc.el, Lb.el);
    var chart = scoresChart(box, '');
    function show(user) {
      var r = P.empiricalThreshold(SORTED, share), prod = share * 851 / 100;
      var sp = (share / 100).toFixed(2).replace(/0$/, '');
      Lr.set('rank', [sp + ' × 851', F(prod, 2).replace(/\.?0+$/, '')], prod, null, ': ');
      Lc.set('⌈' + F(prod, 2).replace(/\.?0+$/, '') + '⌉', [String(r.rank)], r.rank);
      Lb.set('β_emp = z₍' + sub(r.rank) + '₎ = ' + F(r.score, 4) + (share === 90 ? ' (the band record also gives rank 766).' : '.'), r.score);
      Lb.el.setAttribute('data-rank', String(r.rank));
      chart.update(scoresCfg('The 851 sorted scores; the threshold for ' + share + '% is the score at rank ' + r.rank + ', ' + F(r.score, 4) + '.', {
        vlines: [{ x: r.rank, color: '--accent', label: 'rank ' + r.rank, anchor: 'end' }], hlines: [{ y: r.score, color: '--accent', label: 'β = ' + F(r.score, 4) }] }));
      if (user) announce('Rank ' + r.rank + ': ' + F(r.score, 4) + '.');
    }
    show(false);
    return { enter: function () { show(false); } };
  };

  // ----------------------------------------------------------------------- 5.3 one Bayesian bootstrap draw
  CTL.draw = function (box) {
    var ex0 = D.learn.calibration.examples[0], dr1 = D.new.bootstrap_real.draw1, set = '19', which = 1, own = null;
    var dSeg = Seg({ label: 'Scores', value: '19', options: [{ k: '19', t: '19 illustration scores' }, { k: '851', t: 'The 851 real scores' }], onChange: function (k) { set = k; which = 1; own = null; show(true); } });
    var wSeg = Seg({ label: 'Draw', value: 1, options: [{ k: 1, t: 'BCA\'s draw 1' }, { k: 2, t: 'BCA\'s draw 2' }, { k: 3, t: 'BCA\'s draw 3' }], onChange: function (k) { which = +k; own = null; show(true); } });
    var bB = Btn('Draw again in your browser', function () { own = browserExp(set === '19' ? 20 : 852); which = 0; wSeg.set(''); show(true); });
    add(box, dSeg.el, wSeg.el, Row(bB));
    var lab = el('p', 'note ls-cnote'); box.appendChild(lab);
    var rC = Ro('crossing', 'cross'), rT = Ro('total of the random numbers', 'total'), rU = Ro('the unseen row\'s mass', 'unseen');
    box.appendChild(Ros(rC, rT, rU));
    var lines = Lines(box), Lm = Line('mass'), Lrun = Stmt('run');
    add(lines, Lm.el, Lrun.el);
    var tabD = Table(['rank', 'score', 'random weight E', 'mass = E / total', 'running total'], { cls: 'ls-clip ls-drawtab' });
    var tabMore = Btn('Show all rows', function () {
      var bx = tabD.el.querySelector('.tbl'), open = !bx.classList.contains('ls-open');
      bx.classList.toggle('ls-open', open); tabMore.setAttribute('aria-expanded', open ? 'true' : 'false');
      tabMore.textContent = open ? 'Show fewer rows' : 'Show all rows'; W.dispatchEvent(new Event('resize'));
    }, 'ls-more-rows');
    tabMore.setAttribute('aria-expanded', 'false');
    var tabNote = Note(''); tabNote.setAttribute('data-t', 'tabnote');
    add(box, tabD.el, tabMore, tabNote);
    var cM = Chart(box, { title: '', height: 170, x: { min: 0, max: 21 }, y: { min: 0, max: 1 }, series: [], hover: null });
    var cR = Chart(box, { title: '', height: 190, x: { min: 0, max: 21 }, y: { min: 0, max: 1 }, series: [], hover: null });
    /** The draw's rows: every row of a 19-score draw; for the 851 real scores, the rows around the crossing. The row
        before the crossing, the crossing row and the unseen row are marked. */
    function drawTable(q, total) {
      var n = q.n, m = q.m, k = m.crossingRank, idx = [], d = n === 19 ? 4 : 6;
      if (n === 19) { for (var i = 0; i < n; i++) idx.push(i); }
      else { var c0 = k ? k - 1 : n - 1; for (var j = Math.max(0, c0 - 3); j <= Math.min(n - 1, c0 + 2); j++) idx.push(j); }
      var rows = idx.map(function (i) {
        return { cls: k && (i === k - 1 || i === k - 2) ? 'ls-mark' : '', cells: [FI(i + 1), { t: F(q.scores[i], 4), cls: 'n', exact: q.scores[i] },
          { t: F(q.E[i], 6), cls: 'n', exact: q.E[i] }, { t: F(m.masses[i], d), cls: 'n', exact: m.masses[i] }, { t: F(m.running[i], 4), cls: 'n', exact: m.running[i] }] };
      });
      rows.push({ cls: 'ls-mark', cells: ['unseen row', '', { t: F(q.E[n], 6), cls: 'n', exact: q.E[n] }, { t: F(m.unseenMass, d), cls: 'n', exact: m.unseenMass }, ''] });
      tabD.rows(rows);
      tabMore.hidden = n !== 19;
      tabNote.textContent = (n === 19 ? 'Every row of this draw, sorted by score; ' : 'The rows around the crossing (the other ranks are left out); ') +
        'mass = E / ' + F(total, n === 19 ? 4 : 2) + (k ? '; the marked rows are the one before the crossing, the crossing row (rank ' + FI(k) + ') and the unseen row.' : '; the unseen row (marked) holds more than 0.1 of the mass, so the running total never reaches 0.9.');
    }
    function current() {
      var n = set === '19' ? 19 : 851, scores = set === '19' ? ex0.sorted_scores : SORTED, E;
      if (own) E = own;
      else if (set === '19') { var fd = ex0.first_draws[which - 1]; E = fd.exponentials_E_by_sorted_score.concat([fd.exponential_test_atom]); }
      else E = dr1.exponentials_6dp;
      var m = P.drawMasses(E, n, set === '19' ? 'wbcp' : 'all');
      return { n: n, scores: scores, E: E, m: m };
    }
    function show(user) {
      wSeg.buttons.forEach(function (b, i) { b.disabled = set === '851' && i > 0; });
      if (!own) wSeg.set(which);
      var q = current(), m = q.m, n = q.n, bcas = !own;
      var total = (set === '851' && bcas) ? dr1.total : m.total, unseen = (set === '851' && bcas) ? dr1.unseen_mass : m.unseenMass;
      lab.textContent = own ? 'A new random draw made by your browser, not one of BCA\'s.' : (set === '19' ? 'BCA\'s own function, draw ' + which + ' of a fixed random-number generator (numpy\'s, seeded with 20261015 and 1), on 19 illustration scores (the sizes of 19 random normal numbers, not ThrusterChain data).' : 'BCA\'s own function, the first draw of the same fixed random-number generator, on the 851 real scores.');
      var k = m.crossingRank;
      rC.set(k ? 'rank ' + FI(k) + ', score ' + F(q.scores[k - 1], set === '19' ? 4 : 4) : 'never crosses', k || 0);
      rC.el.setAttribute('data-score', k ? String(q.scores[k - 1]) : 'inf');
      rT.set(set === '19' ? F(total, 4) : F(total, 2), total); rU.set(F(unseen, 4), unseen);
      var En = q.E[n];
      Lm.set('the unseen row\'s mass', [F(En, 6) + ' / ' + (set === '19' ? F(total, 4) : F(total, 2)), F(unseen, 4)], unseen, null, ': ');
      if (k) Lrun.set('Running total: ' + F(k > 1 ? m.running[k - 2] : 0, 4) + (k > 1 ? ' after the ' + ordinal(k - 1) + ' score, ' : ' before the first score, ') + F(m.running[k - 1], 4) + ' with the ' + ordinal(k) + ', so this draw crosses at the ' + ordinal(k) + ' score, ' + F(q.scores[k - 1], 4) + '.', k);
      else Lrun.set('The running total never reaches 0.9: the real rows hold only ' + F(1 - unseen, 4) + ' of the mass, so this draw never crosses.', 0);
      drawCharts(q);
      drawTable(q, total);
      if (user) announce(k ? 'Crosses at rank ' + k + '.' : 'Never crosses.');
    }
    function drawCharts(q) {
      var n = q.n, m = q.m, xs = []; for (var i = 1; i <= n + 1; i++) xs.push(i);
      var bars = m.masses.map(function (v, i) { return [i + 1, v, m.crossingRank && i + 1 === m.crossingRank ? '--ink' : '--k-learn']; });
      bars.push([n + 1, m.unseenMass, '--warn']);
      var mx = Math.max.apply(null, bars.map(function (b) { return b[1]; }));
      var tks = n === 19 ? [1, 5, 10, 15, 20] : [1, 200, 400, 600, 852];
      cM.update({ title: 'Each sorted score\'s mass in this draw, and the unseen row\'s at the right end.', height: 170, x: { min: 0, max: n + 2, ticks: tks, fmt: function (v) { return FI(v); } }, y: { min: 0, max: mx * 1.1, fmt: function (v) { return F(v, n === 19 ? 2 : 3); } },
        series: [{ key: 'mass', name: 'mass of a row', color: '--k-learn', kind: 'bars', pts: bars, keep: n === 19 }],
        legendExtra: [{ name: 'the unseen row', color: '--warn' }].concat(m.crossingRank ? [{ name: 'where this draw crosses', color: '--ink' }] : []),
        hover: { xs: xs, title: function (x) { return x === n + 1 ? 'the unseen row' : 'score ' + FI(x) + ' of ' + n; }, rows: function (x) { return [{ color: x === n + 1 ? '--warn' : '--k-learn', name: 'mass', value: F(x === n + 1 ? m.unseenMass : m.masses[x - 1], n === 19 ? 4 : 6) }]; } } });
      var run = m.running.map(function (v, i) { return [i + 1, v]; });
      cR.update({ title: 'The running total of the masses along the sorted scores; the draw crosses where it first reaches 0.9.', height: 190, x: { min: 0, max: n + 2, label: 'sorted score', ticks: tks, fmt: function (v) { return FI(v); } }, y: { min: 0, max: 1, ticks: [0, 0.5, 0.9] },
        series: [{ key: 'run', name: 'running total', color: '--ink', pts: run }],
        hlines: [{ y: 0.9, color: '--warn', label: '0.9', anchor: 'start' }],
        vlines: m.crossingRank ? [{ x: m.crossingRank, color: '--accent', label: 'crosses: ' + ordinal(m.crossingRank), anchor: 'end' }] : [],
        hover: { xs: run.map(function (p) { return p[0]; }), title: function (x) { return 'after score ' + FI(x); }, rows: function (x) { return [{ color: '--ink', name: 'running total', value: F(m.running[x - 1], 4) }]; } } });
    }
    show(false);
    return { enter: function () { drawCharts(current()); } };
  };

  // ----------------------------------------------------------------------- 5.4 the credible threshold, and BCA's beta
  CTL['beta-cred'] = function (box) {
    var CAL = D.new.bootstrap_real.calibrate, mode = 'bca', mine = null;
    var seg = Seg({ label: 'The 1,000 draws', value: 'bca', options: [{ k: 'bca', t: '1,000 draws with the lesson\'s generator' }, { k: 'browser', t: '1,000 draws in your browser' }], onChange: function (k) { mode = k; if (k === 'browser') mine = null; show(true); } });
    box.appendChild(seg.el);
    var fixedB = el('p', 'ls-out ls-fixedb'); fixedB.setAttribute('data-t', 'band-beta');
    setRich(fixedB, 'The band\'s own β: ' + F(BAND.threshold_R, 4) + ' (β_cred at rank ' + BS.rank_beta_cred.join(', ') + ' of 851), from BCA\'s own random numbers during its run; the draws below are other draws, for comparison.');
    box.appendChild(fixedB);
    var lab = el('p', 'note ls-cnote'); box.appendChild(lab);
    var r950 = Ro('the 950th smallest crossing', 'c950'), rAll = Ro('draws that cross', 'finite'), rB = Ro('β = max(β_emp, β_cred)', 'beta');
    setRich(rB.key, 'β = max(β_emp, β_cred)');
    box.appendChild(Ros(r950, rAll, rB));
    var lines = Lines(box), Lb = Line('beta');
    lines.appendChild(Lb.el);
    var chart = Chart(box, { title: '', height: 210, x: { min: 735, max: 800 }, y: { min: 0, max: 60 }, series: [], hover: null });
    function browserCounts() {
      var counts = {}, crossings = [];
      for (var d = 0; d < 1000; d++) {
        var c = P.crossing(SORTED, browserExp(852));
        crossings.push(c.rank === null ? Infinity : c.rank);
        if (c.rank !== null) counts[c.rank] = (counts[c.rank] || 0) + 1;
      }
      return counts;
    }
    function show(user) {
      var counts = mode === 'bca' ? CAL.crossings_by_rank : (mine || (mine = browserCounts()));
      var ranks = Object.keys(counts).map(Number).sort(function (a, b) { return a - b; }), cum = 0, r950v = null, finite = 0;
      ranks.forEach(function (r) { finite += counts[r]; });
      for (var i = 0; i < ranks.length; i++) { cum += counts[ranks[i]]; if (cum >= 950) { r950v = ranks[i]; break; } }
      var z950 = r950v ? SORTED[r950v - 1] : Infinity, be = BS.beta_emp, beta = Math.max(be, z950);
      setRich(lab, mode === 'bca' ? 'BCA\'s own function with the lesson\'s fixed random-number generator of step 5.3. The band\'s saved β_cred, 1.3927 (rank 782), came from BCA\'s own random numbers during its run: different draws, almost the same threshold.' : '1,000 new random draws made by your browser, not BCA\'s; each press of the button makes new ones.');
      r950.set(r950v ? 'rank ' + r950v + ', ' + F(z950, 4) : '+∞ (fewer than 950 cross)', z950); r950.el.setAttribute('data-rank', String(r950v));
      rAll.set(FI(finite) + ' of 1,000', finite); rB.set(F(beta, 4), beta);
      Lb.set('β', ['max(' + F(be, 4) + ', ' + F(z950, 4) + ')', F(beta, 4)], beta, mode === 'bca' ? 'with the lesson\'s generator' : 'with your browser\'s draws');
      var lo = Math.min(735, ranks[0] - 2), hi = Math.max(800, ranks[ranks.length - 1] + 2), mx = Math.max.apply(null, ranks.map(function (r) { return counts[r]; }));
      chart.update({ title: 'How many of the 1,000 draws cross at each rank, with the empirical threshold (rank 766), the 950th crossing and β.', height: 210,
        x: { min: lo, max: hi, label: 'rank where a draw crosses', ticks: [740, 750, 760, 770, 780, 790, 800].filter(function (v) { return v >= lo && v <= hi; }) }, y: { min: 0, max: Math.ceil(mx * 1.15) },
        series: [{ key: 'counts', name: 'draws crossing at that rank', color: '--k-learn', kind: 'bars', barW: 0.8, pts: ranks.map(function (r) { return [r, counts[r]]; }), keep: true }],
        vlines: [{ x: 766, color: '--muted', label: 'empirical: 766', anchor: 'end' }, r950v ? { x: r950v, color: '--accent', dash: false, label: '950th: ' + r950v, dy: 14 } : null].filter(Boolean),
        hover: { xs: ranks, title: function (x) { return 'rank ' + x + ' (score ' + F(SORTED[x - 1], 4) + ')'; }, rows: function (x) { return [{ color: '--k-learn', name: 'draws', value: String(counts[x]) }]; } } });
      if (user) announce('The 950th smallest crossing: rank ' + r950v + ', ' + F(z950, 4) + '.');
    }
    return { enter: function () { show(false); } };
  };

  // ----------------------------------------------------------------------- 5.5 why BCA needs at least 29 rows
  CTL['too-few'] = function (box) {
    var TB = D.learn.calibration.tables.table_n_1_to_120, EX = D.learn.calibration.examples, n = 29;
    var sN = Slider({ label: 'Number of calibration rows n', min: 1, max: 120, step: 1, value: n, fmt: function (x) { return String(x); }, onInput: function (x) { n = x; show(true); } });
    add(box, sN.el, Row(Btn('19', function () { n = 19; sN.set(n); show(true); }), Btn('28', function () { n = 28; sN.set(n); show(true); }), Btn('29', function () { n = 29; sN.set(n); show(true); }), Btn('36', function () { n = 36; sN.set(n); show(true); }), Btn('100', function () { n = 100; sN.set(n); show(true); })));
    var rP = Ro('a draw can cross: 1 − 0.9ⁿ', 'p'), rE = Ro('expected crossing draws of 1,000', 'exp'), rF = Ro('chance that fewer than 950 cross', 'short'), rV = Ro('verdict', 'verdict');
    rV.el.classList.add('ls-ro-wide');
    box.appendChild(Ros(rP, rE, rF, rV));
    var lines = Lines(box), L = Line('p');
    lines.appendChild(L.el);
    var hist = el('div', 'ls-hist'); box.appendChild(hist);
    var chart = Chart(hist, { title: '', height: 180, x: { min: 0, max: 20 }, y: { min: 0, max: 100 }, series: [], hover: null });
    var hnote = Note(''); hist.appendChild(hnote);
    function show(user) {
      var T = TB[n - 1], p = T.p_one_draw_crosses, pw = Math.pow(0.9, n), e = 1000 * p, shortP = 1 - T.p_finite_1000_draws;
      var verdict = p < 0.95 ? 'cannot certify' : (T.certifiable_production ? 'certifies with BCA\'s 1,000 draws' : 'certifies only with unlimited draws');
      var said = verdict === 'certifies only with unlimited draws' ? 'would certify only with unlimited draws; with 1,000 draws it falls short ' + FPct(shortP, 0) + ' of the time' : verdict;
      rP.set(F(p, 4), p); rE.set(F(e, 1), e); rF.set(FPct(shortP), shortP); rV.set(said); rV.el.setAttribute('data-verdict', verdict);
      L.set('1 ' + MINUS + ' 0.9' + sup(n), ['1 ' + MINUS + ' ' + F(pw, 4), F(p, 4)], p, p < 0.95 ? 'below 0.95' : 'at least 0.95');
      var exk = EX.filter(function (x) { return x.n === n; })[0];
      hist.hidden = !exk;
      if (exk) {
        var c = exk.all_draws.crossings_at_each_rank, pts = c.map(function (v, i) { return [i + 1, v]; }).filter(function (q) { return q[1] > 0; });
        var mx = Math.max.apply(null, c.concat([1]));
        chart.update({ title: 'BCA\'s 1,000 draws on ' + n + ' scores: how many cross at each rank.', height: 180, x: { min: 0, max: n + 1, label: 'rank where a draw crosses' }, y: { min: 0, max: Math.ceil(mx * 1.15) },
          series: [{ key: 'h', name: 'draws crossing at that rank', color: '--k-learn', kind: 'bars', barW: 0.7, pts: pts, keep: true }],
          hover: { xs: pts.map(function (q) { return q[0]; }), title: function (x) { return 'rank ' + x + ' of ' + n; }, rows: function (x) { return [{ color: '--k-learn', name: 'draws', value: String(c[x - 1]) }]; } } });
        hnote.textContent = FI(exk.all_draws.draws_that_cross) + ' of 1,000 draws crossed (' + FI(exk.all_draws.draws_that_never_cross) + ' never did); 950 are needed' + (exk.all_draws.draws_that_cross >= 950 ? ': certified.' : ', so β = +∞.');
      }
      if (user) announce('n = ' + n + ': ' + verdict + '.');
    }
    show(false);
    return { enter: function () { show(false); } };
  };

  // ----------------------------------------------------------------------- 5.6 BCA's band
  CTL.band = function (box) {
    var beta = BAND.threshold_R, eta = BAND.unit_u, g = B9.g, q = D.new.row209_critic.q, err = D.new.row209_critic.one_step_error;
    var sG = Slider({ label: 'Predicted size g', min: 0.5, max: 2.5, step: 0.01, value: g, fmt: function (x) { return F(x, 4); }, onInput: function (x) { g = x; show(true); } });
    add(box, sG.el, Row(Btn('g = 1', function () { g = 1; sG.set(g); show(true); }), Btn('Row 209 (' + F(B9.g, 4) + ')', function () { g = B9.g; sG.set(g); show(true); }),
      Btn('Strong current, the actor (' + F(D.new.start0_actor.strong_current.g_at_actor, 4) + ')', function () { g = D.new.start0_actor.strong_current.g_at_actor; sG.set(g); show(true); })));
    var lines = Lines(box), L = Line('U'), Li = Stmt('inside');
    add(lines, L.el, Li.el);
    var pic = el('div', 'ls-bandpic'); box.appendChild(pic);
    function show(user) {
      var U = P.halfWidth(beta, eta, g), inside = Math.abs(err) <= U;
      L.set('U', [F(beta, 4) + ' × ' + F(eta, 4) + (g === 1 ? '' : ' × ' + F(g, 4)), F(U, 4)], U);
      Li.set('|y ' + MINUS + ' q| = ' + F(Math.abs(err), 4) + (inside ? ' ≤ ' : ' > ') + F(U, 4) + ': row 209\'s error lies ' + (inside ? 'inside' : 'outside') + ' the band.');
      Li.el.setAttribute('data-inside', inside ? '1' : '0');
      drawBand(U, inside);
      if (user) L.say();
    }
    /** The band on a value axis at its box's own width (12 px text stays 12 px): ticks at round values, the critic's
        value q labeled above, and y labeled beside the band, clear of the ticks. */
    var lastU = 0, lastIn = true, bandW = 0;
    function drawBand(U, inside) {
      lastU = U; lastIn = inside;
      clear(pic);
      var Wd = Math.min(560, Math.max(260, Math.floor(pic.clientWidth || 340))), Ht = 84, lo = q - 1.1, hi = q + 1.1;
      bandW = Wd;
      function X(v) { return 16 + (v - lo) / (hi - lo) * (Wd - 32); }
      var s = sv('svg', { 'class': 'ls-scale-svg', width: Wd, height: Ht, viewBox: '0 0 ' + Wd + ' ' + Ht, role: 'img', 'aria-label': 'The band q ± U around the critic\'s value at row 209, q = ' + F(q, 4) + ', U = ' + F(U, 4) + ', with the one-step target y = q + ' + F(err, 4) + ' as a dot, ' + (inside ? 'inside' : 'outside') + ' the band.' });
      add(s, sv('rect', { x: X(q - U), y: 28, width: X(q + U) - X(q - U), height: 28, rx: 4, style: 'fill: var(--k-learn-soft); stroke: var(--k-learn)' }));
      add(s, sv('line', { 'class': 'axis', x1: 16, x2: Wd - 16, y1: 56, y2: 56 }));
      for (var v = Math.ceil((lo + 0.05) * 2) / 2; v <= hi - 0.05; v += 0.5) {
        add(s, sv('line', { 'class': 'grid', x1: X(v), x2: X(v), y1: 56, y2: 61 }), sv('text', { 'class': 'tick', x: X(v), y: 76, 'text-anchor': 'middle' }, F(v, 1)));
      }
      add(s, sv('line', { x1: X(q), x2: X(q), y1: 22, y2: 60, style: 'stroke: var(--k-learn)', 'stroke-width': 2 }));
      add(s, sv('text', { 'class': 'ann', x: clampN(X(q), 50, Wd - 50), y: 15, 'text-anchor': 'middle' }, 'q = ' + F(q, 4)));
      var xd = X(q + err), lab = 'y = q + ' + F(err, 4), lw = lab.length * 6.7, right = Math.max(X(q + U), xd) + 10;
      add(s, sv('circle', { cx: xd, cy: 42, r: 6, style: 'fill: var(' + (inside ? '--ok' : '--warn') + '); stroke: var(--surface)', 'stroke-width': 2 }));
      if (right + lw <= Wd - 2) add(s, sv('text', { 'class': 'ls-sclab', x: right, y: 46 }, lab));
      else add(s, sv('text', { 'class': 'ls-sclab', x: Math.min(X(q - U), xd) - 10, y: 46, 'text-anchor': 'end' }, lab));
      pic.appendChild(s);
    }
    W.addEventListener('resize', function () { if (pic.clientWidth && Math.abs(Math.min(560, Math.floor(pic.clientWidth)) - bandW) > 4) drawBand(lastU, lastIn); });
    show(false);
    return { enter: function () { drawBand(lastU, lastIn); } };
  };

  // ======================================================================= chapter 6: using the band, caution and weights

  // ----------------------------------------------------------------------- 6.1 caution, first use: a harder pull
  CTL.dose = function (box) {
    var r = B9.U_over_eta;
    var sR = Slider({ label: 'Half-width over the running unit, U/η', min: 0, max: 20, step: 0.01, value: r, typed: true, typedMax: 1000, fmt: function (x) { return F(x, 4); }, onInput: function (x) { r = x; show(true); } });
    add(box, sR.el, Row(Btn('Row 209 (' + F(B9.U_over_eta, 3) + ')', function () { r = B9.U_over_eta; sR.set(r); show(true); }), Btn('U = η (1)', function () { r = 1; sR.set(1); show(true); }), Btn('U = 3η', function () { r = 3; sR.set(3); show(true); })));
    var lines = Lines(box), Lq = Line('ratio'), Lm = Line('m');
    add(lines, Lq.el, Lm.el);
    var chart = Chart(box, { title: '', height: 210, x: { min: 0, max: 20 }, y: { min: 1, max: 1.55 }, series: [], hover: null });
    var xs = []; for (var i = 0; i <= 200; i++) xs.push(i / 10);
    function show(user) {
      var q = r / (r + 1), m = P.doseFromRatio(r);
      Lq.set('U / (U + η)', [F(r, 4) + ' / (' + F(r, 4) + ' + 1)', F(r, 4) + ' / ' + F(r + 1, 4), F(q, 4)], q, 'in units of η');
      Lm.set('m', ['1 + 0.5 × ' + F(q, 4), F(m, 4)], m);
      chart.update({ title: 'The dose against U/η: it rises from 1 toward 1.5 and never reaches it.', height: 210, x: { min: 0, max: 20, label: 'U/η', ticks: [0, 5, 10, 15, 20] }, y: { min: 0.95, max: 1.55, ticks: [1, 1.1, 1.2, 1.3, 1.4, 1.5] },
        series: [{ key: 'dose', name: 'dose m', color: '--k-learn', pts: xs.map(function (x) { return [x, P.doseFromRatio(x)]; }) }],
        hlines: [{ y: 1.5, color: '--warn', label: '1.5, never reached' }],
        marks: [{ x: Math.min(r, 20), y: m, color: '--k-learn', label: 'm = ' + F(m, 4), anchor: r > 12 ? 'end' : 'start', dy: 16 }, { x: B9.U_over_eta, y: P.doseFromRatio(B9.U_over_eta), color: '--ink', label: 'row 209', r: 4 }],
        hover: { xs: xs.filter(function (x, k) { return k % 5 === 0; }), title: function (x) { return 'U/η = ' + F(x, 1); }, rows: function (x) { return [{ color: '--k-learn', name: 'dose m', value: F(P.doseFromRatio(x), 4) }]; } } });
      if (user) Lm.say();
    }
    return { enter: function () { show(false); }, init: show };
  };

  // ----------------------------------------------------------------------- 6.2 caution, second use: the lower edge
  CTL['lower-edge'] = function (box) {
    var which = 'row', beta = BAND.threshold_R, eta = BAND.unit_u, SC = D.new.start0_actor.strong_current;
    var seg = Seg({ label: 'Where', value: 'row', options: [{ k: 'row', t: 'Row 209' }, { k: 'strong', t: 'Start 0 in strong current' }], onChange: function (k) { which = k; show(true); } });
    box.appendChild(seg.el);
    var bars = Bars([{ key: 'pi', label: 'at the actor\'s command', color: '--k-learn', name: 'Upi' }, { key: 'a', label: 'at the recorded command', color: '--k-ref', name: 'Ua' }]);
    box.appendChild(bars.el);
    var lines = Lines(box), L1 = Line('Upi'), L2 = Line('Ua'), Lp = Line('pen');
    add(lines, L1.el, L2.el, Lp.el);
    function show(user) {
      var gpi = which === 'row' ? B9.g_at_actor : SC.g_at_actor, ga = which === 'row' ? B9.g : SC.g_at_stall_aware;
      var Upi = P.halfWidth(beta, eta, gpi), Ua = P.halfWidth(beta, eta, ga), d = Upi - Ua, pen = P.lowerEdgePenalty(Upi, Ua);
      var other = which === 'row' ? 'recorded' : 'stall-aware';
      bars.el.querySelectorAll('.ls-bar-l')[1].textContent = 'at the ' + other + ' command';
      bars.set({ pi: { v: Upi, t: F(Upi, 4) }, a: { v: Ua, t: F(Ua, 4) } }, 0.6);
      L1.set('U at the actor\'s ' + (which === 'row' ? '' : 'stalled ') + 'command', [F(beta, 6) + ' × ' + F(eta, 6) + ' × ' + F(gpi, 6), F(Upi, 4)], Upi);
      L2.set('U at the ' + other + ' command', [F(beta, 6) + ' × ' + F(eta, 6) + ' × ' + F(ga, 6), F(Ua, 4)], Ua);
      Lp.set('penalty', ['max(' + F(Upi, 4) + ' ' + MINUS + ' ' + F(Ua, 4) + ', 0)', 'max(' + F(d, 4) + ', 0)', F(pen, 4).replace(/^0\.0000$/, '0')], pen, d < 0 ? 'the band is narrower at the actor\'s command' : 'the extra width at the actor\'s command');
      if (user) Lp.say();
    }
    show(false);
    return {};
  };

  // ----------------------------------------------------------------------- 6.3 weights under a known shift
  CTL.weights = function (box) {
    var WL = D.learn.weighting.laws, K = WL.strong_constant, c = 0.8;
    var sC = Slider({ label: 'A calibration row\'s current c', min: 0.1, max: 1.1, step: 0.01, value: c, fmt: function (x) { return F(x, 4); }, onInput: function (x) { c = x; show(true); } });
    var c1 = X0[10], c2 = D.env.start_state.same_index_in_every_bank.strong_current.current;
    add(box, sC.el, Row(Btn('0.8', function () { c = 0.8; sC.set(c); show(true); }), Btn('Start 0, recorded (' + F(c1, 4) + ')', function () { c = c1; sC.set(c); show(true); }), Btn('Start 0, strong current (' + F(c2, 4) + ')', function () { c = c2; sC.set(c); show(true); })));
    var rW = Ro('ω(c)', 'omega'), rTr = Ro('h_train(c)', 'htr'), rTe = Ro('h_test(c)', 'hte');
    setRich(rTr.key, 'h_train(c)'); setRich(rTe.key, 'h_test(c)');
    box.appendChild(Ros(rW, rTr, rTe));
    var lines = Lines(box), L = Line('omega'), Lr = Line('ratio');
    add(lines, L.el, Lr.el);
    var chart = Chart(box, { title: '', height: 220, x: { min: 0.1, max: 1.1 }, y: { min: 0, max: 3 }, series: [], hover: null });
    var xs = []; for (var i = 0; i <= 100; i++) xs.push(0.1 + i / 100);
    function show(user) {
      var w = P.weightStrong(c, K), e = 4 * (c - 0.6), htr = P.density(c, WL.training_xi), hte = P.density(c, WL.strong_xi);
      rW.set(F(w, 4), w); rTr.set(F(htr, 4), htr); rTe.set(F(hte, 4), hte);
      L.set('ω(' + F(c, 4).replace(/0+$/, '').replace(/\.$/, '') + ')', ['exp(4 × (' + F(c, 4).replace(/0+$/, '').replace(/\.$/, '') + ' ' + MINUS + ' 0.6))', 'exp(' + F(e, 4).replace(/0+$/, '').replace(/\.$/, '') + ')', F(w, 4)], w);
      Lr.set('h_test(c) / h_train(c)', [F(hte, 4) + ' / ' + F(htr, 4), F(hte / htr, 4)], hte / htr, 'the same ratio');
      chart.update({ title: 'How often each current appears in training and at the strong test; their ratio at a current is its weight.', height: 220, endLabels: true,
        x: { min: 0.1, max: 1.1, label: 'current c', ticks: [0.1, 0.3, 0.5, 0.7, 0.9, 1.1] }, y: { min: 0, max: 2.6, ticks: [0, 0.5, 1, 1.5, 2, 2.5] },
        series: [{ key: 'train', name: 'h_train: training', short: 'training', color: '--sit-rec', pts: xs.map(function (x) { return [x, P.density(x, WL.training_xi)]; }) },
          { key: 'test', name: 'h_test: strong test', short: 'strong test', color: '--sit-strong', dash: true, pts: xs.map(function (x) { return [x, P.density(x, WL.strong_xi)]; }) }],
        vlines: [{ x: c, color: '--accent', dash: false, label: 'c = ' + F(c, 2), anchor: c > 0.8 ? 'end' : 'start' }],
        hover: { xs: xs.filter(function (x, k) { return k % 5 === 0; }), title: function (x) { return 'current ' + F(x, 2); },
          rows: function (x) { return [{ color: '--sit-rec', name: 'training', value: F(P.density(x, WL.training_xi), 4) }, { color: '--sit-strong', dash: true, name: 'strong test', value: F(P.density(x, WL.strong_xi), 4) }, { name: 'weight ω', value: F(P.weightStrong(x, K), 4) }]; } } });
      if (user) L.say();
    }
    return { enter: function () { show(false); }, init: show };
  };

  // ----------------------------------------------------------------------- 6.4 the weighted threshold, and refusing a step
  CTL.refusal = function (box) {
    var SE = D.learn.weighting.small_example, stops = ['0.2', '0.4', '0.6', '0.8', '0.9', '1.0', '1.1'], cur = '0.8';
    var seg = Seg({ label: 'The test step\'s current', value: cur, options: stops.map(function (s) { return { k: s, t: s }; }), onChange: function (k) { cur = k; show(true); } });
    box.appendChild(seg.el);
    var rW = Ro('ω(c_test)', 'omega'), rS = Ro('its share of the total weight', 'share'), rD = Ro('draws that cross, of 1,000', 'finite'), rB = Ro('β', 'beta');
    setRich(rW.key, 'ω(c_test)');
    box.appendChild(Ros(rW, rS, rD, rB));
    var bars = Bars([{ key: 's', label: 'the test step\'s share', color: '--k-learn', name: 'sharebar' }]);
    box.appendChild(bars.el);
    box.appendChild(Note('The line in the bar is at 10%: a step that holds more than 10% of a draw\'s weight leaves that draw unable to cross. The 40 rows are an illustration, not ThrusterChain data: currents drawn from the training mix, each row\'s score the size of (0.05 + 0.25 c) times a standard normal number (the table above lists them).'));
    var lines = Lines(box), L = Line('share');
    lines.appendChild(L.el);
    function show(user) {
      var q = SE.per_step_thresholds[cur], w = q.test_mass, sh = q.test_mass_share_of_total, thr = P.num(q.threshold);
      rW.set(F(w, 4), w); rS.set(FPct(sh), sh); rD.set(FI(q.finite_draws), q.finite_draws);
      rB.set(thr === Infinity ? '+∞: refused' : F(thr, 4), thr);
      bars.set({ s: { v: sh, t: FPct(sh) } }, 0.2, 0.1);
      L.set('share', [F(w, 4) + ' / (' + F(SE.weights_sum, 3) + ' + ' + F(w, 4) + ')', F(sh, 4)], sh, q.finite_draws >= 950 ? q.finite_draws + ' draws cross: β = ' + F(thr, 4) : 'only ' + q.finite_draws + ' draws cross, fewer than 950: refused', ': ');
      if (user) announce('Current ' + cur + ': ' + (thr === Infinity ? 'refused, ' + q.finite_draws + ' draws cross.' : 'β = ' + F(thr, 4) + '.'));
    }
    show(false);
    return {};
  };

  // ======================================================================= chapter 7: what the tests found
  var HC = D.learn.results.harm_checks, STG = D.learn.results.stages;

  // ----------------------------------------------------------------------- 7.1 what the main test found
  CTL['main-test'] = function (box) {
    var c1 = HC.check1_critic_error, c2 = HC.check2_return_cost, c3 = HC.check3_best_case_caution.harm_placed_minus_controls.strong_current;
    var rows = [
      { h: 'Check 1: is the critic wrong where data are missing?', pass: true,
        lines: ['Measured on critics that score the fixed stall-aware controller, whose exact values are known: the stall-aware controller\'s critic learned after the removal was ' + F(c1.removed_ratio_pooled, 2) + ' times as wrong (median advantage error: the typical error in judging how much better a command is than the controller\'s own) as the same critic learned from every row, at the removed commands (seeds ' + F(c1.removed_ratio_seeds[0], 2) + ' to ' + F(c1.removed_ratio_seeds[1], 2) + '), and not more wrong far from them (' + F(c1.far_ratio_pooled, 2) + ' times).',
          'Pass mark: a ratio of at least 2 at the removed commands.'], num: c1.removed_ratio_pooled },
      { h: 'Check 2: does acting on that error cost return?', pass: true,
        lines: ['Fresh actors, all started from the same cloned actor, were trained against one frozen critic each: the fixed-controller critic learned after the removal, or the one learned from every row. At critic weight 5 the actors of the first lost ' + F(c2.alpha_5.situations.strong_current.harm_mean, 1) + ' points in strong current (at least 5 needed), and pushed past 80% ' + F(c2.alpha_5.past80_share_gapped, 4) + ' / ' + F(c2.alpha_5.past80_share_full, 4) + ' = ' + F(c2.alpha_5.past80_ratio, 2) + ' times as often (shares of visited states; at least 1.5 needed).',
          'At critic weight 2.5 the same ratio was ' + F(c2.alpha_2_5.past80_ratio, 2) + ': that weight failed, so the step-up rule (fixed in advance: if critic weight 2.5 failed, the check ran again at 5, then at 10) moved to 5.'], num: c2.alpha_5.situations.strong_current.harm_mean },
      { h: 'Check 3: could caution of BCA\'s kind help if it were placed perfectly?', pass: false,
        lines: ['Fresh actors trained against one frozen critic, a plain TD3+BC run\'s critic learned after the removal. In strong current, harm-placed caution minus strength-matched even caution (one dose on every row, tuned before any return so that it moves the actor exactly as much) = ' + Fs(c3.harm_placed_minus_matched_even.mean, 1) + '; minus amount-matched caution (every row gets the average harm-placed dose) = ' + Fs(c3.harm_placed_minus_amount_matched.mean, 1) + '; minus shuffled doses (the same doses in a random row order) = ' + Fs(c3.harm_placed_minus_shuffled.mean, 1) + '. Pass mark: +3 points against every control.',
          'Failed, so the test stopped before any BCA run: “' + HC.stop + '”.'], num: c3.harm_placed_minus_matched_even.mean }];
    var list = el('div', 'ls-path');
    rows.forEach(function (r, i) {
      var d = el('details', 'ls-wblk ls-pathrow'); d.setAttribute('data-t', 'check' + (i + 1));
      var s = el('summary'); add(s, el('span', null, r.h + ' '), el('span', 'tag ' + (r.pass ? 'ok' : 'warn'), r.pass ? 'passed' : 'failed'));
      d.appendChild(s);
      var inner = el('div', 'ls-pathin');
      r.lines.forEach(function (t) { inner.appendChild(rich('p', null, t)); });
      d.appendChild(inner); list.appendChild(d);
      d.setAttribute('data-exact', String(r.num));
    });
    box.appendChild(list);
    var hv = c2.alpha_5.situations.strong_current.harm_values;
    var tab = Table(['seed', 'points lost in strong current (check 2)'], { rowHead: true });
    tab.rows(hv.map(function (v, i) { return { cells: [String(i), { t: F(v, 1), cls: 'n', exact: v }] }; }).concat([{ cls: 'ls-sum', cells: ['all five', { t: F(c2.alpha_5.situations.strong_current.harm_mean, 1), cls: 'n', exact: c2.alpha_5.situations.strong_current.harm_mean }] }]));
    box.appendChild(tab.el);
    box.appendChild(Note('Check 1\'s ratio ran from ' + F(c1.removed_ratio_seeds[0], 2) + ' to ' + F(c1.removed_ratio_seeds[1], 2) + ' across the five seeds (0 to 4).'));
    return { enter: function () { tab.cue(); } };
  };

  // ----------------------------------------------------------------------- 7.2 what the follow-ups found
  CTL['after-stop'] = function (box) {
    var wt = D.learn.results.weighting_and_toys.weighting_test.findings[1];
    var m = /unweighted coverage ([\d.]+%)[^;]*? to ([\d.]+%)[^;]*?; exact weights restore ([\d.]+%) to ([\d.]+%)/.exec(wt);
    var R = [
      { f: 'Main test: hiding the stall harms the learned controller in strong current', n: F(HC.check2_return_cost.alpha_5.situations.strong_current.harm_mean, 1) + ' points lost', tag: 'pre', link: '#results-short', lt: 'What was found, in short' },
      { f: 'Stage 1, no BCA: a penalty on the critic\'s value at the actor\'s own command, set from the exact one-step error', n: F(STG.stage1.recovery_mean, 1) + ' of 98.9 points won back', tag: 'exp', link: '#ats-stage1', lt: 'Stage 1' },
      { f: 'Stage 2, BCA\'s lower edge with a learned extra width (seeds 0 to 4)', n: F(STG.stage2_seeds_0_4.claim_numbers.X, 1) + ' points won back', tag: 'exp', link: '#ats-stage2', lt: 'Stage 2' },
      { f: 'Stage 2, a hand-set limit to the recorded range of commands', n: F(STG.stage2_seeds_0_4.claim_numbers.Z, 1) + ' points won back', tag: 'exp', link: '#ats-stage2', lt: 'Stage 2' },
      { f: 'Stage 2\'s confirmatory run on fresh seeds 5 to 9', n: 'not run yet', tag: 'pre', tt: 'pre-registered plan, not run yet', link: '#ats-stage2', lt: 'Stage 2' },
      { f: 'Restructured BCA, step 1: caution on in 5 of 5 stalling-world seeds, off in 5 of 5 no-stall seeds', n: 'net ' + F(STG.restructured_step1.net.net, 1) + ' points (' + STG.restructured_step1.go_no_go.verdict + ')', tag: 'exp', link: '#ats-restructure', lt: 'Restructured BCA' },
      { f: 'Restructured BCA, step 2 (training)', n: 'net ' + F(STG.restructured_step2.net_primary, 1) + ' against ' + F(STG.restructured_step2.best_world_blind_net, 1) + ' for the best rival', tag: 'exp', link: '#ats-restructure', lt: 'Restructured BCA' },
      { f: 'Weighting test, strong test: coverage of one-step errors on test steps outside the removed region', n: m ? m[1] + ' to ' + m[2] + ' without weights; ' + m[3] + ' to ' + m[4] + ' with exact weights' : wt, tag: 'pre', link: '#iw-findings', lt: 'The weighting test\'s findings' }];
    var ep = doc.getElementById('ep-viewer');
    if (ep) R.push({ f: 'Watch an episode of every host and variant', n: '', tag: 'exp', link: '#ep-viewer', lt: 'Episode viewer' });
    var filt = 'all';
    var seg = Seg({ label: 'Show', value: 'all', options: [{ k: 'all', t: 'All' }, { k: 'pre', t: 'Pre-registered' }, { k: 'exp', t: 'Exploratory' }], onChange: function (k) { filt = k; show(true); } });
    box.appendChild(seg.el);
    var tab = Table(['finding', 'number', 'evidence', 'section']);
    box.appendChild(tab.el);
    box.appendChild(Note('Where Stage 2\'s confirmatory run stands is said once, under Full detail below.'));
    function show(user) {
      tab.rows(R.map(function (r) {
        var hit = filt !== 'all' && r.tag === filt;        // the chosen kind stands out; nothing is dimmed
        return { cls: hit ? 'ls-match' : '', cells: [r.f, r.n, { t: r.tt || (r.tag === 'pre' ? 'pre-registered' : 'exploratory'), cls: 'ls-tagcell' }, { t: '' }] };
      }));
      [].slice.call(tab.body.rows).forEach(function (tr, i) {
        var td = tr.cells[3]; clear(td);
        var a = el('a', null, R[i].lt); a.href = R[i].link; td.appendChild(a);
      });
      if (user) announce(filt === 'all' ? 'All findings shown.' : (filt === 'pre' ? 'Pre-registered findings are marked.' : 'Exploratory findings are marked.'));
    }
    show(false);
    return { enter: function () { tab.cue(); } };
  };

  // ----------------------------------------------------------------------- 7.3 the whole chain on one screen
  CTL.recap = function (box) {
    var a = el('a', 'btn ls-again', 'Start again from step 1.1'); a.href = '#learn-world';
    box.appendChild(Row(a));
    var out = el('p', 'note ls-cnote'); out.setAttribute('data-t', 'seen'); box.appendChild(out);
    var rt = root.querySelector('#learn-recap .ls-recap-t');      // a tap anywhere on a row of the chain opens its step
    if (rt) rt.addEventListener('click', function (ev) {
      if (ev.target.closest('a')) return;
      var tr = ev.target.closest('tbody tr'), a = tr && tr.querySelector('a[data-step-link]');
      if (a) a.click();
    });
    function show() {
      var seen = (store.get('tc-learn-seen') || '').split(' ').filter(Boolean);
      markSeen(seen);
      out.textContent = 'You have visited ' + seen.filter(function (id) { return /^learn-/.test(id) && doc.getElementById(id); }).length + ' of the 39 steps; visited steps carry a check mark in the table above and in the list of steps.';
    }
    show();
    return { enter: show };
  };
  /** Check marks beside visited steps: the recap table and the intro's list. */
  function markSeen(seen) {
    var S = {};
    (seen || (store.get('tc-learn-seen') || '').split(' ')).forEach(function (id) { if (id) S[id] = 1; });
    [].slice.call(root.querySelectorAll('a[data-step-link]')).forEach(function (a) {
      var on = !!S[a.getAttribute('data-step-link')];
      a.classList.toggle('ls-seen', on);
      if (on) a.setAttribute('aria-description', 'visited'); else a.removeAttribute('aria-description');
    });
  }

  // ======================================================================= wiring: worked lines, checks, phone blocks, and the pages
  var STEPS = [].slice.call(root.querySelectorAll('.ls-steps > .ls-step[id]'));
  var INTRO = root.querySelector('.ls-intro');
  var PAGES = (INTRO ? [INTRO] : []).concat(STEPS);
  function titleOf(p) { var h = p === INTRO ? null : p.querySelector('h3'); return h ? h.textContent.replace(/\s+/g, ' ').trim() : 'About this lesson'; }

  // ----------------------------------------------------------------------- a worked line, tapped: its numbers in full and where they come from
  var FILES = { env: 'saved with this lesson, computed from the environment\'s code', learn: 'saved with this lesson, read from the project\'s saved runs and results',
    'new': 'a new reading: computed for this lesson from the saved runs' };
  // the facts files keep two British key names (identifiers); a reader is shown them in American spelling
  var KEYWORDS = [[new RegExp('cent' + 're', 'g'), 'center'], [new RegExp('neighb' + 'ours', 'g'), 'neighbors']];
  function shownPath(k) { var t = k.slice(k.indexOf(':') + 1); KEYWORDS.forEach(function (w) { t = t.replace(w[0], w[1]); }); return t; }
  function valueText(v) {
    if (typeof v === 'number' || typeof v === 'boolean' || v === Infinity) return raw(v);
    if (Array.isArray(v)) {
      if (v.length <= 13 && v.every(function (x) { return typeof x === 'number'; })) return '(' + v.map(raw).join(', ') + ')';
      return 'a list of ' + FI(v.length) + ' values';
    }
    if (typeof v === 'string') {
      if (/^\s*-?\d/.test(v)) v = v.replace(/-/g, MINUS);     // a number kept as text gets the minus sign too
      return v.length > 90 ? v.slice(0, 88) + '…' : v;
    }
    return null;
  }
  function detailPanel(line) {
    var box = el('div', 'ls-wx'), ex = line.getAttribute('data-exact');
    if (ex !== null && ex !== '') box.appendChild(el('p', 'ls-wx-x', 'The result in full: ' + raw(Number(ex))));
    var keys = (line.getAttribute('data-keys') || '').split('|').filter(Boolean), shown = 0;
    if (keys.length) box.appendChild(el('p', 'ls-wx-k', 'The numbers on this line, in full, and where they come from:'));
    keys.forEach(function (k) {
      var v;
      try { v = P.get(k); } catch (e) { return; }
      var t = valueText(v);
      if (t === null) return;
      var p = el('p', 'ls-wx-k ls-wx-v');
      add(p, el('b', null, t), el('span', 'ls-wx-f', ' (' + FILES[k.slice(0, k.indexOf(':'))] + ')'));
      var src = el('p', 'ls-wx-src');
      add(src, doc.createTextNode('source: '), el('code', null, shownPath(k)));
      add(box, p, src); shown++;
    });
    if (line.classList.contains('ls-live')) box.appendChild(el('p', 'ls-wx-k', 'Computed in your browser by the lesson\'s copy of the environment\'s code, from the numbers you chose.'));
    else if (!shown) box.appendChild(el('p', 'ls-wx-k', 'Every number on this line comes from the lesson\'s data, rounded as shown.'));
    return box;
  }
  function toggleLine(line) {
    var open = line.getAttribute('aria-expanded') === 'true', nx = line.nextElementSibling;
    if (open) { if (nx && nx.classList.contains('ls-wx')) nx.parentNode.removeChild(nx); line.setAttribute('aria-expanded', 'false'); return; }
    line.parentNode.insertBefore(detailPanel(line), line.nextSibling);
    line.setAttribute('aria-expanded', 'true');
  }
  function prepLine(l) {
    if (l.hasAttribute('tabindex')) return;
    l.setAttribute('tabindex', '0'); l.setAttribute('role', 'button'); l.setAttribute('aria-expanded', 'false');
  }
  [].slice.call(root.querySelectorAll('.ls-wl')).forEach(prepLine);
  root.addEventListener('click', function (ev) {
    var l = ev.target.closest && ev.target.closest('.ls-wl');
    if (l && root.contains(l) && !ev.target.closest('a, button, input')) { prepLine(l); toggleLine(l); }
  });
  root.addEventListener('keydown', function (ev) {
    var l = ev.target;
    if (l && l.classList && l.classList.contains('ls-wl') && (ev.key === 'Enter' || ev.key === ' ')) { ev.preventDefault(); toggleLine(l); }
  });

  // ----------------------------------------------------------------------- check yourself (remembered with the visited marks)
  var quiz = {};
  try { quiz = JSON.parse(store.get('tc-learn-quiz') || '{}') || {}; } catch (e) { quiz = {}; }
  function mark(ck, i, user) {
    var opts = [].slice.call(ck.querySelectorAll('.ls-opt')), b = opts[i];
    if (!b) return;
    var ok = b.getAttribute('data-ok') === '1';
    opts.forEach(function (o) { o.setAttribute('aria-pressed', o === b ? 'true' : 'false'); o.classList.remove('ls-right', 'ls-wrong', 'ls-correct'); });
    b.classList.add(ok ? 'ls-right' : 'ls-wrong');
    if (!ok) opts.forEach(function (o) { if (o.getAttribute('data-ok') === '1') o.classList.add('ls-correct'); });
    var why = ck.querySelector('.ls-why'), v = ck.querySelector('.ls-verdict');
    v.textContent = ok ? 'Right.' : 'Not quite.';
    why.hidden = false; why.classList.toggle('ls-bad', !ok);
    ck.setAttribute('data-result', ok ? 'right' : 'wrong');
    if (user) { announce(v.textContent + ' ' + why.textContent.replace(/^\s*(Right\.|Not quite\.)\s*/, '')); }
  }
  [].slice.call(root.querySelectorAll('.ls-check')).forEach(function (ck) {
    var st = ck.closest('.ls-step'), id = st ? st.id : '';
    ck.addEventListener('click', function (ev) {
      var b = ev.target.closest && ev.target.closest('.ls-opt');
      if (!b) return;
      var i = [].slice.call(ck.querySelectorAll('.ls-opt')).indexOf(b);
      mark(ck, i, true);
      quiz[id] = i; store.set('tc-learn-quiz', JSON.stringify(quiz));
    });
    if (quiz[id] !== undefined) mark(ck, quiz[id], false);
  });

  // ----------------------------------------------------------------------- phone: blocks and legends start closed; long tables collapse
  // They are closed when their step is first shown (closing them in hidden steps at load would lay out all 39 steps).
  // Closing fires "toggle" on each; the page's table and tooltip scripts recompute the whole page on every toggle, so
  // these first, programmatic toggles are stopped here (capture phase) before they reach those listeners.
  var quiet = [];
  root.addEventListener('toggle', function (ev) { var k = quiet.indexOf(ev.target); if (k >= 0) { quiet.splice(k, 1); ev.stopPropagation(); } }, true);
  function phoneClose(st) {
    if (!mqPhone.matches) return;
    [].slice.call(st.querySelectorAll('details[data-phone="closed"]')).forEach(function (d) { if (d.open) { quiet.push(d); d.open = false; } });
  }
  [].slice.call(root.querySelectorAll('.ls-more-rows')).forEach(function (b) {
    b.addEventListener('click', function () {
      var p = b.previousElementSibling, box = p && p.classList.contains('tblwrap') ? p.querySelector('.tbl') : p;
      if (!box) return;
      var open = !box.classList.contains('ls-open');
      box.classList.toggle('ls-open', open); b.setAttribute('aria-expanded', open ? 'true' : 'false');
      b.textContent = open ? (b.getAttribute('data-more') === 'water' ? 'Hide the water rows' : 'Show fewer rows') : (b.getAttribute('data-more') === 'water' ? 'Show the water rows' : 'Show all ' + box.querySelectorAll('tbody tr').length + ' rows');
      W.dispatchEvent(new Event('resize'));
    });
  });

  // ----------------------------------------------------------------------- controls are built when their step is first shown (enterStep), so
  // the page loads without laying out 39 panels it does not show
  BUILT = function (st) { phoneClose(st); [].slice.call(st.querySelectorAll('.ls-wl')).forEach(prepLine); };
  markSeen();
  root.addEventListener('walkthrough:enter', function (ev) { var st = ev.target.closest && ev.target.closest('.ls-step'); if (st) enterStep(st); if (ev.target === INTRO) markSeen(); });
  root.addEventListener('walkthrough:leave', function (ev) { var st = ev.target.closest && ev.target.closest('.ls-step'); if (st) leaveStep(st); });
  if (W.IntersectionObserver) {
    var io = new W.IntersectionObserver(function (ents) {
      ents.forEach(function (e) { if (e.isIntersecting && e.target.getClientRects().length) { var st = e.target.closest('.ls-step'); if (st && !st.hasAttribute('hidden') && !st.classList.contains('ls-off') && !st.classList.contains('nv-off')) enterStep(st); } });
    }, { rootMargin: '200px 0px' });
    [].slice.call(root.querySelectorAll('.ls-try')).forEach(function (t) { io.observe(t); });
  }

  // ----------------------------------------------------------------------- one step at a time: the navigation's job when it is on the page, else ours
  var own = null;
  function navOn() { return html.classList.contains('nv-on'); }
  function ownPager() {
    if (own || navOn() || !PAGES.length) return;
    var UF = 'onbeforematch' in doc.body;
    var seen = {};
    (store.get('tc-learn-seen') || '').split(' ').forEach(function (id) { if (id) seen[id] = 1; });
    html.classList.add('ls-own');
    function off(p) { return p.getAttribute('hidden') === 'until-found' || p.classList.contains('ls-off'); }
    function hide(p) { if (UF) p.setAttribute('hidden', 'until-found'); else p.classList.add('ls-off'); }
    function showEl(p) { if (p.getAttribute('hidden') === 'until-found') p.removeAttribute('hidden'); p.classList.remove('ls-off'); }
    function link(cls, page, k, name) {
      var a = el('a', cls), b = el('b', 'ls-k'), ar = el('span', 'ls-ar', cls === 'ls-prev' ? '← ' : ' →');
      ar.setAttribute('aria-hidden', 'true');
      if (cls === 'ls-prev') add(b, ar, el('span', null, k)); else add(b, el('span', null, k), ar);
      add(a, b, el('span', 'ls-n', name));
      a.href = typeof page === 'string' ? page : '#' + (page === INTRO ? root.id : page.id);
      if (typeof page !== 'string') a.setAttribute('data-page', page === INTRO ? '' : page.id);
      return a;
    }
    var next = doc.getElementById('action-space');
    PAGES.forEach(function (p, i) {
      var f = el('nav', 'lsx-foot'); f.setAttribute('aria-label', 'Lesson steps');
      if (p === INTRO) {
        f.appendChild(el('span', 'ls-sp'));
        if (STEPS[0]) f.appendChild(link('ls-next', STEPS[0], 'Start', titleOf(STEPS[0])));
      } else {
        f.appendChild(link('ls-prev', PAGES[i - 1], 'Previous', titleOf(PAGES[i - 1])));
        if (PAGES[i + 1]) f.appendChild(link('ls-next', PAGES[i + 1], 'Next', titleOf(PAGES[i + 1])));
        else if (next) f.appendChild(link('ls-next', '#action-space', 'Next', 'Next section: Action Space'));
      }
      p.appendChild(f);
    });
    var cont = null;
    function intro() {
      var last = doc.getElementById(store.get('tc-learn-last') || '');
      var f = INTRO.querySelector('.lsx-foot'), st = f.querySelector('.ls-next .ls-k span');
      if (last && STEPS.indexOf(last) >= 0) {
        if (!cont) { cont = link('ls-cont', last, 'Continue', ''); f.insertBefore(cont, f.firstChild); }
        cont.href = '#' + last.id; cont.setAttribute('data-page', last.id); cont.querySelector('.ls-n').textContent = titleOf(last);
        if (st) st.textContent = 'Start from step 1';
      }
      var hb = doc.getElementById('ls-start');
      if (hb) { if (last && STEPS.indexOf(last) >= 0) { hb.textContent = 'Continue the lesson: ' + titleOf(last); hb.href = '#' + last.id; } else { hb.textContent = 'Learn step by step'; hb.href = '#learn'; } }
    }
    var cur = null;
    function show(p, o) {
      o = o || {};
      if (cur !== p) {
        PAGES.forEach(function (q) { if (q !== p && !off(q)) { leaveStep(q); hide(q); } });
        showEl(p); cur = p;
        if (p !== INTRO) { seen[p.id] = 1; store.set('tc-learn-last', p.id); store.set('tc-learn-seen', Object.keys(seen).join(' ')); }
        intro(); markSeen();
        W.requestAnimationFrame(function () { enterStep(p); });
      }
      if (o.hash) { try { W.history.replaceState(null, '', '#' + (p === INTRO ? root.id : p.id)); } catch (e) { W.location.hash = p === INTRO ? root.id : p.id; } }
      if (o.scroll) {
        var y = p.getBoundingClientRect().top + (W.pageYOffset || doc.documentElement.scrollTop) - 16;
        try { W.scrollTo({ top: y, behavior: 'auto' }); } catch (e) { W.scrollTo(0, y); }
      }
      if (o.focus) { var h = p.querySelector('h3') || root.querySelector('#learn-h'); if (h) { try { h.focus({ preventScroll: true }); } catch (e) { h.focus(); } } }
      if (o.say) announce(p === INTRO ? 'About this lesson' : 'Step ' + (STEPS.indexOf(p) + 1) + ' of ' + STEPS.length + ': ' + titleOf(p));
    }
    function pageFor(hash) {
      if (!hash || hash === '#') return null;
      var id = decodeURIComponent(hash.slice(1));
      if (id === root.id) return INTRO;
      var t = doc.getElementById(id);
      if (!t || !root.contains(t)) return null;
      var st = t.closest('.ls-step');
      return st || INTRO;
    }
    root.addEventListener('click', function (ev) {
      var a = ev.target.closest && ev.target.closest('a[href^="#"]');
      if (!a || ev.defaultPrevented || ev.button > 0 || ev.metaKey || ev.ctrlKey || ev.shiftKey) return;
      var p = pageFor(a.getAttribute('href'));
      if (!p) return;
      ev.preventDefault();
      show(p, { hash: true, scroll: true, focus: !!a.closest('.lsx-foot'), say: true });
    });
    root.addEventListener('keydown', function (ev) {
      var a = ev.target.closest && ev.target.closest('.lsx-foot a');
      if (!a || (ev.key !== 'ArrowLeft' && ev.key !== 'ArrowRight')) return;
      var l = a.closest('.lsx-foot').querySelector(ev.key === 'ArrowLeft' ? '.ls-prev' : '.ls-next');
      if (!l) return;
      ev.preventDefault();
      var p = pageFor(l.getAttribute('href'));
      if (p) { show(p, { hash: true, scroll: true, say: true }); var nf = p.querySelector('.lsx-foot ' + (ev.key === 'ArrowLeft' ? '.ls-prev' : '.ls-next')) || p.querySelector('.lsx-foot a'); if (nf) nf.focus({ preventScroll: true }); }
      else { W.location.hash = l.getAttribute('href'); }
    });
    W.addEventListener('hashchange', function () { var p = pageFor(W.location.hash); if (p) show(p, { scroll: true, say: true }); });
    doc.addEventListener('click', function (ev) {      // links to the lesson from outside it (the header button, other sections)
      var a = ev.target.closest && ev.target.closest('a[href^="#learn"]');
      if (!a || root.contains(a) || ev.defaultPrevented) return;
      var p = pageFor(a.getAttribute('href'));
      if (!p) return;
      ev.preventDefault();
      show(p, { hash: true, scroll: true, say: true });
    });
    if (UF) PAGES.forEach(function (p) { p.addEventListener('beforematch', function () { show(p, {}); }); });
    var first = pageFor(W.location.hash);
    PAGES.forEach(function (p) { if (p !== (first || INTRO)) hide(p); });
    cur = null;
    show(first || INTRO, { scroll: !!first });
    own = { show: show };
  }
  if (navOn()) { /* the navigation shows the pages; the controls follow its events */ }
  else if (html.classList.contains('nv-js') && doc.readyState !== 'complete') {
    W.addEventListener('load', function () { W.setTimeout(function () { if (!navOn()) ownPager(); }, 0); });
  } else ownPager();

})();
