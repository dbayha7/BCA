"""Tooltip text pass: the script and style changes ta_apply.py makes, as exact (old, new) snippets.

TT  - the shared tooltip script ("/* Tooltips: one shared tooltip" ... "/* e6_tooltips */"):
      * cell tuples become one label: value line per cell (never cut, the cell's own value in bold), and a note under
        them spells out every code, acronym and page nickname the row uses (GLOSS);
      * the cell reader plain() keeps a code block's lines and puts each block on its own line (no more "Box(...) 4
        numbers" or "bank., Role"), collapses source whitespace, and writes an index with no Unicode form in words
        (no more "e^(2c)" or "P_cap");
      * fit(): the box always shows all of its text. The old CSS cut a box taller than the window off mid-line with no
        sign (max-height: calc(100vh - 16px); overflow: hidden). Now a tall box first gets wider, then a smaller type,
        and only in a window too short even for that does it scroll, say so, and let the pointer in to scroll it.
DR  - the drawings script ("/* e6_tooltips */"): the live panel's computed tooltips are complete sentences, three at most.
IW  - the weighting test's script: the bars carry the shared tooltip, set on every redraw (before, each redraw made new
      SVG <title> elements, which brought back the browser's own tooltip after a lean button or the slider was used);
      the chart readouts read as label: value.
CSS - the new layout rules (appended at the end of the page's style, so they override the earlier ones).
"""

# ------------------------------------------------------------------------------------------------ the tooltip script
TT = [
    ("index maps: letters too; an index with no Unicode form in words",
     r'''  const SUP = { 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹", "+": "⁺", "−": "⁻",
    "-": "⁻", t: "ᵗ", n: "ⁿ", i: "ⁱ" };
  const SUB = { 0: "₀", 1: "₁", 2: "₂", 3: "₃", 4: "₄", 5: "₅", 6: "₆", 7: "₇", 8: "₈", 9: "₉" };
  const mapAll = (s, m) => [...s].every(ch => m[ch]) ? [...s].map(ch => m[ch]).join("") : null;
  const pow = s => mapAll(s, SUP) || (s.length > 1 ? "^(" + s + ")" : "^" + s);
''',
     r'''  const SUP = { 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹", "+": "⁺", "−": "⁻",
    "-": "⁻", "(": "⁽", ")": "⁾", a: "ᵃ", b: "ᵇ", c: "ᶜ", d: "ᵈ", e: "ᵉ", f: "ᶠ", g: "ᵍ", h: "ʰ", i: "ⁱ", j: "ʲ",
    k: "ᵏ", l: "ˡ", m: "ᵐ", n: "ⁿ", o: "ᵒ", p: "ᵖ", r: "ʳ", s: "ˢ", t: "ᵗ", u: "ᵘ", v: "ᵛ", w: "ʷ", x: "ˣ", y: "ʸ",
    z: "ᶻ" };
  const SUB = { 0: "₀", 1: "₁", 2: "₂", 3: "₃", 4: "₄", 5: "₅", 6: "₆", 7: "₇", 8: "₈", 9: "₉", "+": "₊", "−": "₋",
    "-": "₋", ",": ",", a: "ₐ", e: "ₑ", h: "ₕ", i: "ᵢ", j: "ⱼ", k: "ₖ", l: "ₗ", m: "ₘ", n: "ₙ", o: "ₒ", p: "ₚ", r: "ᵣ",
    s: "ₛ", t: "ₜ", u: "ᵤ", v: "ᵥ", x: "ₓ" };
  const mapAll = (s, m) => [...s].every(ch => m[ch]) ? [...s].map(ch => m[ch]).join("") : null;
  // an index with no Unicode form is written in words, never as code-like "^(2c)" or "_cap"
  const pow = s => mapAll(s, SUP) || " to the power " + s;
  const low = s => mapAll(s, SUB) || " (" + s + ")";
'''),
    ("tex(): subscripts through low()",
     r'''    .replace(/_\{?(\w+)\}?/g, (m, g) => mapAll(g, SUB) || "_" + g)
''',
     r'''    .replace(/_\{?(\w+)\}?/g, (m, g) => low(g))
'''),
    ("plain(): blocks on their own lines, code blocks keep their lines; cleanV; GLOSS",
     r'''      if (c.nodeType === 3) { s += c.nodeValue; continue; }
      if (c.nodeType !== 1 || c.hidden || (c.classList && c.classList.contains("tt-sr"))) continue;
      const tag = c.localName;
      if (tag === "mjx-container") s += " " + tex(c.getAttribute("aria-label") || "") + " ";
      else if (tag === "br") s += " ";
      else if (tag === "sup") s += c.getAttribute("aria-label") ? " (" + c.getAttribute("aria-label") + ")" : pow(plain(c).trim());
      else if (tag === "sub") s += mapAll(plain(c).trim(), SUB) || "_" + plain(c).trim();
      else if (c.classList.contains("rng")) s += " (" + plain(c).trim() + ")";
      else if (/^(div|p|pre|ul|ol|li|dl|dt|dd)$/.test(tag) || c.classList.contains("note")) s += " " + plain(c) + " ";
      else s += plain(c);
    }
    return s;
  }
  const clean = s => s.replace(/\s+/g, " ").trim();
''',
     r'''      if (c.nodeType === 3) { s += c.nodeValue.replace(/\s+/g, " "); continue; }   // source line breaks are spaces
      if (c.nodeType !== 1 || c.hidden || (c.classList && c.classList.contains("tt-sr"))) continue;
      const tag = c.localName;
      if (tag === "mjx-container") s += " " + tex(c.getAttribute("aria-label") || "") + " ";
      else if (tag === "br") s += "\n";
      else if (tag === "pre") s += "\n" + c.textContent.replace(/\s+$/, "") + "\n";   // a code block keeps its lines
      else if (tag === "sup") s += c.getAttribute("aria-label") ? " (" + c.getAttribute("aria-label") + ")" : pow(clean(plain(c)));
      else if (tag === "sub") s += low(clean(plain(c)));
      else if (c.classList.contains("rng")) s += " (" + clean(plain(c)) + ")";
      else if (/^(div|p|ul|ol|li|dl|dt|dd)$/.test(tag) || c.classList.contains("note")) s += "\n" + plain(c) + "\n";   // a block: its own line
      else s += plain(c);
    }
    return s;
  }
  const clean = s => s.replace(/\s+/g, " ").trim();                                     // one line: a column's name
  const cleanV = s => s.split("\n").map(l => l.replace(/\s+/g, " ").trim()).filter(Boolean).join("\n");   // a value keeps its lines

  /* ---------- notes that spell out what a row's cells use: codes, acronyms and the page's own names ---------- */
  // [what to look for, the sentence that explains it, (optional) skip it when the row already explains it]
  const GLOSS = [
    [/TD3\+BC/, "TD3+BC is twin delayed deep deterministic policy gradient (TD3) plus behaviour cloning (BC), a standard learner that learns a controller only from recorded data."],
    [/\bTD3\b(?!\+)/, "TD3 is twin delayed deep deterministic policy gradient, named for its twin critics and its delayed actor updates."],
    [/\bBCA\b/, "BCA is Bayesian Conformal Aggregation, the method this page tests: it adds caution where it judges that the critic may be wrong."],
    [/BCAShift/, "BCAShift is the prefix of the project's registered environment names; BCA in it stands for Bayesian Conformal Aggregation, the method this page tests."],
    [/\b(pos|speed) \d/, "In this table pos n and speed n are float n's position number and speed number."],
    [/\b(float32|float64)\b/, "The types float32 and float64 are decimal numbers stored with about 7 and about 16 significant digits."],
    [/\b(int32|int8)\b/, "The types int32 and int8 are whole numbers stored in 32 or 8 bits."],
    [/\bbool\b/, "A bool is a value that is either True or False (yes or no)."],
    [/\bstr\b/, "A str is text."],
    [/\bBox\(/, "Box(low, high, shape, type) is how Gymnasium describes a list of numbers with fixed bounds.", /is Gymnasium's description/],
    [/Gymnasium|\bgym\b/, "Gymnasium is the standard Python interface for environments in which controllers act and learn, and gym.make creates an environment from its registered name."],
    [/\bnumpy\b/, "The name numpy is the Python library for arrays of numbers."],
    [/SHA-256/, "SHA-256 (secure hash algorithm, 256 bits) is the standard method for computing a hash, a short fingerprint of a file that changes if any character of the file changes."],
    [/\b(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)[0-9a-f]{8,}…/, "A string of digits and letters ending in … is the start of a hash, a short fingerprint of a file, and only its first characters are shown."],
    [/\b(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)[0-9a-f]{12,}\b(?!…)/, "A long string of digits and letters is a hash, a short fingerprint of a file that changes if any character of the file changes.", /SHA-256/],
    [/\bUTC\b/, "UTC is coordinated universal time."],
    [/D4RL/, "D4RL (datasets for deep data-driven reinforcement learning) is a standard collection of recorded datasets, and the recorded files use its field names."],
    [/\b[a-z_]+\.py\b/, "A name ending in .py is a Python program file."],
    [/\.md\b/, "A name ending in .md is a text document written in Markdown, a plain-text format for documents."],
    [/Physics\(stall=\.\.\.\)/, "Physics(stall=...) is the code's physics settings, made with stall set to True for the stalling world or False for the no-stall comparison world; the three dots stand for that value."],
    [/\w\/(?=[\s,.)]|$)/, "A name ending in / is a folder of the repository."],
    [/\b(?!JSON\b)[A-Z][A-Z_]{3,}\b(?!\.md\b)|\bB1\b|H1–H4/, "Names in capital letters here are fixed names and labels in the project's code."],
    [/gapped critic/, "The gapped critic is the critic trained on the data after the removal, which never saw the removed commands."],
    [/\bgate\b|gate's/, "The gate is the four harm checks, fixed in advance, that had to pass before the test of Bayesian Conformal Aggregation could run."],
    [/room check/i, "The room check is the harm check that asked whether caution placed at the harm could help, that is, whether there is room for caution to help."],
    [/futility look/i, "The futility look was an early check, fixed in advance, that would have stopped the test if the oracle had clearly failed."],
    [/\bgain(?: check|-check| and placement checks)/, "The gain check asks whether adding Bayesian Conformal Aggregation gains at least 2 points over the same learner without it in a shifted situation where caution can help."],
    [/\bplacement checks?\b/, "The placement check asks whether Bayesian Conformal Aggregation also beats the same caution without its placement: its own doses spread evenly or shuffled, and uniform caution of the same strength."],
    [/detection checks?/, "The three detection checks ask whether the doses of Bayesian Conformal Aggregation are largest on the training rows where the critic most overrates the actor's command against the recorded one."],
    [/no-loss/, "The no-loss check asks whether adding Bayesian Conformal Aggregation costs clearly less than 2 points in the recorded conditions."],
    [/beats-copying/, "The beats-copying check asks whether, in the recorded conditions, the actor with Bayesian Conformal Aggregation scores at least 5 points above behaviour cloning, which copies the recorded commands."],
    [/no-shift-harm/, "The no-shift-harm check asks whether, on the full log with nothing removed, adding Bayesian Conformal Aggregation costs clearly less than 2 points in the recorded conditions and wherever caution can help."],
    [/harm-present/, "The harm-present check asks whether there is still harm to remove: without Bayesian Conformal Aggregation, the actor must still lose at least 3 points to pushing past 80%."],
    [/\bplain (?:run|TD3|critic|learner)\b|minus plain\b/, "Plain means without Bayesian Conformal Aggregation, so a plain run or a plain critic comes from the standard learner on its own."],
    [/\boracle/i, "The oracle is a penalty built from the critic's exact error, which no method that learns from data could have."],
    [/main arm/, "The main arm is the oracle penalty at full strength."],
    [/marks look/, "The marks look is the last step of the run, which computes the marks; it waits for the user's choice."],
    [/approval file|approval program/, "An approval file records the user's approval, given by typing the start of a hash at a terminal, and the steps that need it refuse to run without it."],
    [/choice message/, "The user's choice message is the chat message in which the user chose, for three of the penalties that look only at the recorded data, which of their two forms counts toward the pass marks; it had to contain the first 16 characters of the specification's hash."],
    [/stand-ins?\b/, "The data-only stand-ins are signals computed from the data alone that take the oracle's place.", /'bend'|'edge'/],
    [/stand-ins 'bend' and 'edge'/, "Here 'bend' and 'edge' are the code's names for the design plan's two stand-ins for the learned actor, one for a critic that curves down past 80% and one for a critic that keeps its slope."],
    [/pre-regist|\bpins?\b/i, "A pre-registration is the record of every rule and pass mark, fixed before any result existed; it pins a file by recording the file's hash, so any later change to that file is detected."],
    [/manifest/, "A code manifest is a small file that lists the code files with their hashes."],
    [/Stage 1/, "Stage 1 is the first exploratory stage after the harm checks stopped the test; it tested, without Bayesian Conformal Aggregation, which kinds of caution can reach the harm."],
    [/Stage 2/, "Stage 2 is the test of Bayesian Conformal Aggregation itself, with every rule fixed in advance; the user has approved it, but it has not run yet."],
    [/step-up rule/, "The critic-weight step-up rule, fixed in advance, reruns the return-cost check (the harm check that asks whether an actor trained against the critic loses return) at a critic weight of 5, and then 10, if it fails at the published 2.5."],
    [/\bseeds?\b/i, "A seed is the number that fixes every random draw of a run or a recording, so runs with different seeds are repeat runs that differ only by chance."],
    [/stop rule/, "The stop rule, fixed in advance, ends the test before any run of Bayesian Conformal Aggregation when the harm checks fail."],
  ];
  function glossFor(s) {
    const out = [];
    for (const [re, txt, skip] of GLOSS) if (re.test(s) && !(skip && skip.test(s)) && !out.includes(txt)) out.push(txt);
    return out;
  }
'''),
    ("tuples: one label: value line per cell, a note that spells out what the row uses",
     r'''      const tr = a.parentElement, r = info.rowIdx.get(tr);
      const box = el("div", "tt-tuple");
      const pairs = [];
      let colName = "", colDesc = "";
      if (info.kv) {
        const th = tr.querySelector("th"), h = th && info.heads.get(th);
        pairs.push([h ? h.name : "", clean(plain(a)), true]);
        if (h) { colName = h.name; colDesc = h.desc; }
      } else {
        if (info.groupOf.get(tr)) pairs.push([info.group, info.groupOf.get(tr), false]);
        const seen = new Set();
        (info.bodyGrid[r] || []).forEach((cell, c) => {
          if (!cell || seen.has(cell)) return;
          seen.add(cell);
          const v = clean(plain(cell));
          if (!v) return;
          pairs.push([info.names[c], v, cell === a]);
          if (cell === a) { colName = info.names[c]; colDesc = info.descs[c]; }
        });
      }
      box.append("(");
      pairs.forEach(([k, v, isCur], i) => {
        if (i) box.append(", ");
        const p = el("span", isCur ? "p cur" : "p");
        if (k) p.append(el("span", "k", k + ": "));
        if (isCur) p.append(el("b", null, v)); else p.append(v);          // the whole value, never cut
        box.append(p);
      });
      box.append(")");
      out.push(box);
      if (colDesc) { const d = el("div", "tt-col"); d.append(el("b", null, colName + ": "), colDesc); out.push(d); }
      return out;
''',
     r'''      const tr = a.parentElement, r = info.rowIdx.get(tr);
      const box = el("div", "tt-tuple");
      const pairs = [];
      let colName = "", colDesc = "";
      if (info.kv) {
        const th = tr.querySelector("th"), h = th && info.heads.get(th);
        pairs.push([h ? h.name : "", cleanV(plain(a)), true]);
        if (h) { colName = h.name; colDesc = h.desc; }
      } else {
        if (info.groupOf.get(tr)) pairs.push([info.group, info.groupOf.get(tr), false]);
        const seen = new Set();
        (info.bodyGrid[r] || []).forEach((cell, c) => {
          if (!cell || seen.has(cell)) return;
          seen.add(cell);
          const v = cleanV(plain(cell));
          if (!v) return;
          pairs.push([info.names[c], v, cell === a]);
          if (cell === a) { colName = info.names[c]; colDesc = info.descs[c]; }
        });
      }
      // one line per cell, "label: value", the whole value (never cut), the cell's own value in bold
      pairs.forEach(([k, v, isCur]) => {
        const p = el("div", isCur ? "p cur" : "p");
        if (k) p.append(el("span", "k", k + ":"));
        const val = el("span", k ? "v" : "v full");
        if (isCur) val.append(el("b", null, v)); else val.append(v);
        p.append(val);
        box.append(p);
      });
      out.push(box);
      if (colDesc) { const d = el("div", "tt-col"); d.append(el("b", null, colName + ": "), colDesc); out.push(d); }
      const notes = glossFor(pairs.map(([k, v]) => k + " " + v).join("\n"));
      if (notes.length) out.push(el("div", "tt-note", notes.join(" ")));
      return out;
'''),
    ("key-value tables: a row header takes its renamed label",
     r'''        info.heads.set(th, { name: names[i], desc: d ? d.desc : "" });
''',
     r'''        info.heads.set(th, { name: d ? d.name : names[i], desc: d ? d.desc : "" });
'''),
    ("show(): fit the box before placing it",
     r'''    tip.classList.toggle("tt-wide", tip.textContent.length > 360);   // a long row or legend: a wider box, fewer lines
    tip.hidden = false;
    if (a.dataset.tt) a.classList.add("tt-act");
    place();
''',
     r'''    tip.classList.toggle("tt-wide", tip.textContent.length > 360);   // a long row or legend: a wider box, fewer lines
    tip.hidden = false;
    if (a.dataset.tt) a.classList.add("tt-act");
    fit();
    place();
'''),
    ("fit(): the box always shows all of its text",
     r'''  function place() {
    if (!cur) return;
''',
     r'''  // the box shows all of its text. Taller than the window, it first gets wider, then flows a row's cells or a legend's
  // entries into columns, then takes a smaller type; only a window too short even for that makes it scroll, and then it
  // says so and lets the pointer in to scroll it
  function fit() {
    const vw = document.documentElement.clientWidth, room = window.innerHeight - 16;
    tip.classList.remove("tt-flow", "tt-small", "tt-scroll");
    tip.style.maxWidth = ""; tip.style.maxHeight = "";
    tip.style.left = "0px"; tip.style.top = "0px";
    if (tip.offsetHeight <= room) return;
    tip.style.maxWidth = Math.min(vw - 16, 720) + "px";
    if (tip.offsetHeight <= room) return;
    if (vw >= 560) { tip.classList.add("tt-flow"); if (tip.offsetHeight <= room) return; }
    tip.style.maxWidth = (vw - 16) + "px";
    if (tip.offsetHeight <= room) return;
    tip.classList.add("tt-small");
    if (tip.offsetHeight <= room) return;
    tip.classList.add("tt-scroll");
    tip.style.maxHeight = Math.max(80, room) + "px";
  }
  function place() {
    if (!cur) return;
'''),
    ("a tap inside a scrolling box does not close it",
     r'''    if (cur && !cur.contains(e.target)) hide();      // a tap or click elsewhere closes it
''',
     r'''    if (cur && !cur.contains(e.target) && !tip.contains(e.target)) hide();      // a tap or click elsewhere closes it
'''),
    ("the pointer may move into a scrolling box",
     r'''    if (cur === a && how === "hover") tHide = setTimeout(hide, 120);
  });
''',
     r'''    if (cur === a && how === "hover") tHide = setTimeout(hide, 120);
  });
  // a box that scrolls lets the pointer in: it stays while the pointer is on it and goes when the pointer leaves
  tip.addEventListener("pointerenter", () => { if (tip.classList.contains("tt-scroll")) clearTimeout(tHide); });
  tip.addEventListener("pointerleave", e => {
    if (!cur || how !== "hover" || (e.relatedTarget && cur.contains(e.relatedTarget))) return;
    clearTimeout(tHide); tHide = setTimeout(hide, 120);
  });
'''),
]

# ------------------------------------------------------------------------------------------------ the drawings script
DR = [
    ("the water's push: complete sentences",
     r'''      return ["The water's push, state " + t,
        "c + w₁ = " + num(s[10], 3) + (num(s[11], 3).startsWith(MINUS) ? " − " + num(s[11], 3).slice(1) : " + " + num(s[11], 3)) +
        " = " + num(wat.push[t], 3) +
        ": the current plus the push the wave adds in this state, the same on every float. " +
        (inReg ? "Above 0.565, so the state is in the full-throttle region, where the recording controller flips a coin for the full-throttle command."
               : "Not above 0.565, so the state is outside the full-throttle region.")];
''',
     r'''      return ["The water's push, state " + t,
        "In this state the water pushes every float with c + w₁ = " + num(s[10], 3) +
        (num(s[11], 3).startsWith(MINUS) ? " − " + num(s[11], 3).slice(1) : " + " + num(s[11], 3)) + " = " +
        num(wat.push[t], 3) + ": the current c plus the push w₁ that the wave adds now. " +
        (inReg ? "This push is above 0.565, so the state is in the full-throttle region, where the recording controller flips a coin for the full-throttle command."
               : "This push is not above 0.565, so the state is outside the full-throttle region.")];
'''),
    ("a thruster: three complete sentences",
     r'''      let how;
      if (stalled) how = "Stalled: past 80% its propeller pushes less than its throttle asks.";
      else if (past && !stallWorld) how = "Past 80%, but this world has no stall: its push equals its throttle.";
      else how = "Within 80%: its push equals its throttle.";
      return ["Thruster " + (k + 1) + ", state " + t,
        "Throttle " + num(ak, 2) + " (" + dir + "); push " + num(fk, 2) + ". " + how + " It pushes floats " + (k + 1) +
        " and " + (k + 2) + ". Its throttle cost: 0.3 × (" + num(ak, 2) + ")² = " + (0.3 * ak * ak).toFixed(3) + "."];
''',
     r'''      let how;
      if (stalled) how = "It is past 80% and stalled, so its propeller pushes less than its throttle asks";
      else if (past && !stallWorld) how = "It is past 80%, but this world has no stall, so its push equals its throttle";
      else how = "It is within 80%, so its push equals its throttle";
      return ["Thruster " + (k + 1) + ", state " + t,
        "Its throttle is " + num(ak, 2) + " (" + dir + ") and its push is " + num(fk, 2) + ". " + how +
        ", and its throttle cost is 0.3 × (" + num(ak, 2) + ")² = " + (0.3 * ak * ak).toFixed(3) + ". It pushes floats " +
        (k + 1) + " and " + (k + 2) + "."];
'''),
    ("a float: two complete sentences, p and v named",
     r'''    return ["Float " + (j + 1) + ", state " + t,
      "Position number " + num(p, 2) + ", speed number " + num(v, 2) + "; distance from rest √(p² + v²) = " +
      Math.hypot(p, v).toFixed(2) + ". Pushed " + by + ". Its own swing takes " + PERIOD[j] + " steps."];
''',
     r'''    return ["Float " + (j + 1) + ", state " + t,
      "Its position number p is " + num(p, 2) + " and its speed number v is " + num(v, 2) +
      ", so its distance from rest, √(p² + v²), is " + Math.hypot(p, v).toFixed(2) + ". It is pushed " + by +
      ", and its own swing takes " + PERIOD[j] + " steps."];
'''),
    ("the cost chart: one sentence for the four costs, the picked controller named as such",
     r'''    const lines = CTRL.map(c => NAME.ctrl[c] + ": " + num(-D.roll[key(S.world, S.sit, c)].g[t], 2) +
      (c === S.ctrl ? " (picked)" : ""));
    return ["Cost so far at state " + t, lines.join("\n") + "\nThe cost so far is minus the return so far; lower is better."];
''',
     r'''    const parts = CTRL.map(c => {
      const nm = NAME.ctrl[c], v = num(-D.roll[key(S.world, S.sit, c)].g[t], 2);
      const who = /^no throttle$/i.test(nm) ? "with no throttle" : "for the " + nm.charAt(0).toLowerCase() + nm.slice(1);
      return v + " " + who + (c === S.ctrl ? " (the controller picked in the panel)" : "");
    });
    return ["Cost so far at state " + t, "At this state the cost so far is " + parts.slice(0, -1).join(", ") + " and " +
      parts[parts.length - 1] + ". The cost so far is minus the return so far, so lower is better."];
'''),
]

# ------------------------------------------------------------------------------------------------ the weighting test
IW = [
    ("bars: a lead-in per test",
     r'''  const NAME = { none: "No shift", mild: "Mild", strong: "Strong" };
''',
     r'''  const NAME = { none: "No shift", mild: "Mild", strong: "Strong" };
  const LEAD = { none: "With no shift", mild: "Under the mild shift", strong: "Under the strong shift" };
'''),
    ("bars: the shared tooltip on every redraw, never an SVG <title>",
     r'''      el("title", {}, r).textContent = `${NAME[k]}: ${GAP[k]}% of test states use a removed command`;
''',
     r'''      // the shared tooltip, set again on every redraw (an SVG <title> here brought back the browser's own tooltip)
      r.setAttribute("data-tip-focus", "no"); r.setAttribute("data-tip-desc", "no");
      r.setAttribute("data-tip-title", NAME[k]);
      r.setAttribute("data-tip", `${LEAD[k]}, ${GAP[k]}% of test states have a recorded command in the removed cluster: the commands that hold thruster 1 at exactly 80%, which this test deletes from the training data.`);
'''),
    ("density chart readout: label: value",
     r'''c => `current ${c.toFixed(2)}<br>training ${dens(LR, c).toFixed(3)}<br>test ${dens(lam, c).toFixed(3)}`''',
     r'''c => `Current c: ${c.toFixed(2)}<br>Training density: ${dens(LR, c).toFixed(3)}<br>Test density: ${dens(lam, c).toFixed(3)}`'''),
    ("weight chart readout: label: value",
     r'''c => `current ${c.toFixed(2)}<br>weight ${wgt(lam, c).toFixed(3)}`''',
     r'''c => `Current c: ${c.toFixed(2)}<br>Weight ω(c): ${wgt(lam, c).toFixed(3)}`'''),
    ("coverage chart readout: label: value",
     r'''l => `lean ${sgn(l, 2)}<br>unweighted ${(100 * F(l, tTrain)).toFixed(1)}%<br>weighted 90.0%`''',
     r'''l => `Test lean ξ: ${sgn(l, 2)}<br>Coverage without weights: ${(100 * F(l, tTrain)).toFixed(1)}%<br>Coverage with weights: 90.0%`'''),
]

# ------------------------------------------------------------------------------------------------ style
CSS_OLD = ".tt { max-height: calc(100vh - 16px); overflow: hidden; }\n"
CSS_NEW = ".tt { max-height: none; overflow: visible; }\n"
CSS_ADD = r'''/* tooltip text pass: a row's cells as label: value lines, a note that spells out what the row uses, and a box that
   always shows all of its text (wider, then a smaller type, and only as a last resort a scroll area that says so) */
.tt-tuple { display: grid; grid-template-columns: fit-content(42%) minmax(0, 1fr); gap: 1px 0; margin: 0 -4px; }
.tt-tuple .p { display: contents; }
.tt-tuple .p > span { padding: 1px 4px; }
.tt-tuple .p > .k { color: var(--muted); }
.tt-tuple .p > .v { color: var(--ink-2); white-space: pre-line; overflow-wrap: anywhere; }
.tt-tuple .p > .v.full { grid-column: 1 / -1; }
.tt-tuple .cur { background: none; padding: 0; border-radius: 0; }
.tt-tuple .p.cur > span { background: var(--accent-soft); }
.tt-tuple .p.cur > .k { color: var(--accent-ink); border-radius: 3px 0 0 3px; }
.tt-tuple .p.cur > .v { color: var(--ink); border-radius: 0 3px 3px 0; }
.tt-tuple .p.cur > .v.full { border-radius: 3px; }
.tt-note { margin-top: 6px; padding-top: 6px; border-top: 1px solid var(--rule); color: var(--muted); font-size: 13px; }
.tt.tt-flow .tt-tuple { display: block; columns: 230px; column-gap: 18px; margin: 0; }
.tt.tt-flow .tt-tuple .p { display: block; break-inside: avoid; padding: 1px 4px; margin: 0 -4px; border-radius: 3px; }
.tt.tt-flow .tt-tuple .p > span { padding: 0; background: none; }
.tt.tt-flow .tt-tuple .p > .k { margin-right: 0.35em; }
.tt.tt-flow .tt-tuple .p.cur { background: var(--accent-soft); }
.tt.tt-flow .tt-rich { columns: 300px; column-gap: 20px; }
.tt.tt-flow .tt-rich > div, .tt.tt-flow .tt-rich > p { break-inside: avoid; }
.tt.tt-small { font-size: 12.5px; line-height: 1.35; padding: 6px 9px 7px; }
.tt.tt-small .tt-col, .tt.tt-small .tt-note { font-size: 12px; }
svg.chart .pt-hits ~ text:not([data-tip]) { pointer-events: none; }
.tt.tt-scroll { overflow-y: auto; pointer-events: auto; overscroll-behavior: contain; }
.tt.tt-scroll::after { content: "Scroll inside this box to read the rest."; position: sticky; bottom: -7px; display: block;
  margin: 6px -9px -7px; padding: 4px 9px; background: var(--surface); border-top: 1px solid var(--rule);
  color: var(--muted); font-size: 12px; }
'''

# ------------------------------------------------------------------------------------- pass 4, edit 2
# The one change page pass 4 made to the shared tooltip script (p4_e2_apply.py, section 10; recorded by
# p4_e3_sources.py; ta_apply.py does not use it): a formula's tooltip shows the whole legend beside it, every
# group of a grouped legend (one template per grouped legend, its group headings in bold).
PASS4_TT = [
    ("a formula's tooltip: every group of a grouped legend",
     r'''    const dl = k && k.querySelector("dl"), h = k && k.querySelector(".symkey-h");
    if (!dl) return;
    if (!dl.id) dl.id = "symkey-" + (keys.indexOf(k) + 1);''',
     r'''    const h = k && k.querySelector(".symkey-h");
    let dl = k && k.querySelector("dl");
    if (!dl) return;
    if (k.querySelectorAll(":scope > dl").length > 1) {   // a legend in groups: one template holds every group under its heading
      const tid = "symkey-all-" + (keys.indexOf(k) + 1);
      let tpl = document.getElementById(tid);
      if (!tpl) {
        tpl = document.createElement("template"); tpl.id = tid;
        k.querySelectorAll(":scope > .symkey-g, :scope > dl").forEach(n => {
          if (n.localName === "dl") { n.querySelectorAll(":scope > div").forEach(d => tpl.content.appendChild(d.cloneNode(true))); return; }
          const p = document.createElement("p"), b = document.createElement("b");
          b.textContent = n.textContent; p.appendChild(b); tpl.content.appendChild(p);
        });
        k.appendChild(tpl);
      }
      dl = tpl;
    } else if (!dl.id) dl.id = "symkey-" + (keys.indexOf(k) + 1);'''),
]
