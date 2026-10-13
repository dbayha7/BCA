() => {
  const GREEK = {alpha:"α",beta:"β",gamma:"γ",lambda:"λ",mu:"μ",kappa:"κ",omega:"ω",sigma:"σ",Delta:"Δ",delta:"δ",pi:"π",phi:"φ",theta:"θ",rho:"ρ",eta:"η",nu:"ν",epsilon:"ε"};
  const TEX = s => s.replace(/\\text\{([^}]*)\}/g, "$1")
    .replace(/\\(alpha|beta|gamma|lambda|mu|kappa|omega|sigma|Delta|delta|pi|phi|theta|rho|eta|nu|epsilon)/g, (m, g) => GREEK[g])
    .replace(/\\times/g, "×").replace(/\\cdot/g, "·").replace(/\\[,;: ]/g, " ").replace(/[{}]/g, "").replace(/\s+/g, " ").trim();
  const txt = n => {
    if (n.nodeType === 3) return n.nodeValue;
    if (n.nodeType !== 1) return "";
    if (n.localName === "mjx-container") return " " + TEX(n.getAttribute("aria-label") || "") + " ";
    let s = ""; for (const c of n.childNodes) s += txt(c); return s;
  };
  const clean = s => s.replace(/\s+/g, " ").trim();
  const out = [];
  document.querySelectorAll("table").forEach(t => {
    const sec = t.closest("section[id]"), sub = t.closest(".subsec[id]");
    const hrows = t.tHead ? [...t.tHead.rows] : [];
    const grid = [];
    hrows.forEach((tr, r) => { grid[r] = grid[r] || []; let c = 0;
      for (const cell of tr.cells) { while (grid[r][c]) c++; const cs = cell.colSpan || 1, rs = cell.rowSpan || 1;
        for (let i = 0; i < rs; i++) for (let j = 0; j < cs; j++) (grid[r + i] = grid[r + i] || [])[c + j] = cell; c += cs; } });
    const ncol = Math.max(0, ...grid.map(g => g.length));
    const labels = [];
    for (let c = 0; c < ncol; c++) {
      const last = grid[grid.length - 1][c];
      if (last && last.dataset.label) { labels.push(clean(last.dataset.label)); continue; }
      const parts = [];
      for (let r = 0; r < grid.length; r++) { const s = clean(txt(grid[r][c] || document.createTextNode(""))); if (s && !parts.includes(s)) parts.push(s); }
      labels.push(parts.join(": "));
    }
    const kv = !t.tHead;
    const rowLabels = kv ? [...t.querySelectorAll("tbody th")].map(th => clean(txt(th))) : [];
    const groups = [...t.querySelectorAll("tbody tr.sub")].map(tr => clean(txt(tr)));
    const fr = [...t.tBodies[0].rows].find(tr => !tr.classList.contains("sub"));
    out.push({ sec: sec ? sec.id : "", sub: sub ? sub.id : "", labels: kv ? rowLabels : labels, kv, groups,
      row: fr ? [...fr.cells].map(c => clean(txt(c)).slice(0, 50)) : [] });
  });
  return out;
}
