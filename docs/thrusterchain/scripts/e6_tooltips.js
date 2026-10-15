<script>
/* Tooltips, part 2: the live panel's two drawings answer the pointer part by part, and the chart points' built-in
   titles become the page's shared tooltip (so the Tooltips switch at the top turns them off too).
   - The channel: the water's push bar, each float, each thruster and each gauge show their numbers in the state on
     screen. The cost chart shows the four controllers' costs so far at the state under the pointer.
   - It only reads the live panel's data and state (D, S, key, NAME and CTRL of the panel's script, shared by classic
     scripts) and never changes them. Without them the drawings keep their one general tooltip. */
(function () {
  "use strict";
  document.querySelectorAll("svg.chart title").forEach(t => {
    const p = t.parentElement;
    if (!p || p.localName === "svg" || p.hasAttribute("data-tip")) return;
    p.setAttribute("data-tip-focus", "no");
    p.setAttribute("data-tip", t.textContent.replace(/\s+/g, " ").trim());
    t.remove();
  });
  if (typeof D === "undefined" || typeof S === "undefined" || typeof key !== "function" ||
      typeof NAME === "undefined" || typeof CTRL === "undefined") return;
  const PERIOD = [62.8, 30.8, 17.8, 11.4, 7.8];      // steps one swing takes, float by float (Transition Dynamics)
  const MINUS = "−";
  const num = (x, d) => { const s = Math.abs(x).toFixed(d); return (x < 0 && Number(s) !== 0 ? MINUS : "") + s; };
  const pct = a => Math.round(Math.abs(a) * 100) + "%";

  function channelTip(x, y, cv) {
    const W = cv.clientWidth, narrow = W < 560;
    const chanTop = narrow ? 66 : 60, chanH = narrow ? 176 : 196, gaugeTop = chanTop + chanH + 30;
    const R = D.roll[key()], t = S.t, s = R.s[t], a = R.a[t], f = R.f[t], st = R.st[t];
    const wat = D.water[S.sit], stallWorld = S.world === "stall";
    if (y < chanTop) {
      const inReg = wat.reg[t] === 1;
      return ["The water's push, state " + t,
        "c + w₁ = " + num(s[10], 3) + " + " + num(s[11], 3) + " = " + num(wat.push[t], 3) +
        ": the current plus the push the wave adds in this state, the same on every float. " +
        (inReg ? "Above 0.565, so the state is in the full-throttle region, where the recording controller flips a coin for the full-throttle command."
               : "Not above 0.565, so the state is outside the full-throttle region.")];
    }
    const col = W / 5;
    const thrusterAt = () => {                                   // the nearest thruster, if the pointer is close to it
      let best = -1, d = 1e9;
      for (let k = 0; k < 4; k++) { const dd = Math.abs(x - col * (k + 1)); if (dd < d) { d = dd; best = k; } }
      return { k: best, d };
    };
    const thrTip = k => {
      const ak = a[k], fk = f[k], stalled = st[k] === "1", past = Math.abs(ak) > 0.8;
      const dir = Math.abs(ak) < 0.005 ? "off" : pct(ak) + (ak < 0 ? " upstream, against the water" : " downstream");
      let how;
      if (stalled) how = "Stalled: past 80% its propeller pushes less than its throttle asks.";
      else if (past && !stallWorld) how = "Past 80%, but this world has no stall: its push equals its throttle.";
      else how = "Within 80%: its push equals its throttle.";
      return ["Thruster " + (k + 1) + ", state " + t,
        "Throttle " + num(ak, 2) + " (" + dir + "); push " + num(fk, 2) + ". " + how + " It pushes floats " + (k + 1) +
        " and " + (k + 2) + ". Its throttle cost: 0.3 × (" + num(ak, 2) + ")² = " + (0.3 * ak * ak).toFixed(3) + "."];
    };
    if (y >= chanTop + chanH) {                                   // the gauges
      const n = thrusterAt();
      return n.d <= col / 2 ? thrTip(n.k) : null;
    }
    const n = thrusterAt();
    if (n.d < 15) return thrTip(n.k);
    const j = Math.max(0, Math.min(4, Math.floor(x / col)));
    const p = s[2 * j], v = s[2 * j + 1];
    const by = j === 0 ? "only by thruster 1" : j === 4 ? "only by thruster 4" : "by thrusters " + j + " and " + (j + 1);
    return ["Float " + (j + 1) + ", state " + t,
      "Position number " + num(p, 2) + ", speed number " + num(v, 2) + "; distance from rest √(p² + v²) = " +
      Math.hypot(p, v).toFixed(2) + ". Pushed " + by + ". Its own swing takes " + PERIOD[j] + " steps."];
  }

  function costTip(x, y, cv) {
    const w = cv.clientWidth, narrow = w < 560, padL = 46, padR = narrow ? 70 : 170;
    if (x < padL - 8) return null;
    const t = Math.max(0, Math.min(99, Math.round((x - padL) / (w - padL - padR) * 99)));
    const lines = CTRL.map(c => NAME.ctrl[c] + ": " + num(-D.roll[key(S.world, S.sit, c)].g[t], 2) +
      (c === S.ctrl ? " (picked)" : ""));
    return ["Cost so far at state " + t, lines.join("\n") + "\nThe cost so far is minus the return so far; lower is better."];
  }

  function wire(cv, tipAt) {
    if (!cv) return;
    const base = [cv.getAttribute("data-tip-title") || "", cv.getAttribute("data-tip") || ""];
    let last = null, inside = false, reset = 0, lastPtr = -1e9;
    const set = tp => {
      tp = tp || base;
      if (cv.getAttribute("data-tip-title") !== tp[0]) cv.setAttribute("data-tip-title", tp[0]);
      if (cv.getAttribute("data-tip") !== tp[1]) cv.setAttribute("data-tip", tp[1]);
    };
    const at = e => {
      lastPtr = performance.now();
      const r = cv.getBoundingClientRect();
      last = { x: e.clientX - r.left, y: e.clientY - r.top };
      clearTimeout(reset); inside = true;
      try { set(tipAt(last.x, last.y, cv)); } catch (err) { set(base); }
    };
    cv.addEventListener("pointermove", at);
    cv.addEventListener("pointerdown", at);
    cv.addEventListener("pointerleave", e => {         // a mouse leaving: back to the general text once the tooltip has gone
      inside = false; clearTimeout(reset);
      if (e.pointerType !== "touch") reset = setTimeout(() => { if (!inside) set(base); }, 400);
    });
    // keyboard focus shows the general text (a tap also focuses the drawing: keep the part it touched)
    cv.addEventListener("focus", () => { if (performance.now() - lastPtr > 600) set(base); });
    const refresh = () => { if (inside && last) { try { set(tipAt(last.x, last.y, cv)); } catch (err) { set(base); } } };
    const watch = ["tLabel", "exHead"].map(id => document.getElementById(id)).filter(Boolean);
    const mo = new MutationObserver(refresh);
    watch.forEach(n => mo.observe(n, { childList: true, characterData: true, subtree: true }));
  }
  wire(document.getElementById("channel"), channelTip);
  wire(document.getElementById("cost"), costTip);
})();
</script>
