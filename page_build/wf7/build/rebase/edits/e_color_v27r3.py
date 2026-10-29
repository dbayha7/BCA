"""Rebase onto the live page's service version 1792181606-4140 (the other session's publish after version 26, without
a version-history entry: result F's actor-only learners as a second row of the Learner switch). K13b, the learned
actors' cost readout: the live readout now has a branch for those learners (rf: two data qualities, "Both", their
names from RFT.tipnm, their rerun's name); the anchor is the live block and the replacement keeps every branch, one
keyed line per data quality drawn. Review round 2, A3/F16: the last line names the actors ("All five actors learned",
"All five BCA actors learned", "Both result F (exact weights) actors learned")."""
import rep

src = (rep.W7 / "color" / "apply_color.py").read_text(encoding="utf-8")
i = src.index('E("K13b", ')
j = src.index('E("N7b", ')
NEW = r'''E("K13b", "tooltip", "cost chart while the learned actor drives (versions 21 and 25, and the live page's result F learners): "
  "one keyed line per data quality drawn, in its line's color (--lq-1 .. --lq-5), instead of one sentence; the learner's "
  "name, the rerun and the situation in the last line",
  '      const rf = typeof isRF === "function" && isRF(S.lr);\n'
  '      const nm = rf ? RFT.tipnm[S.lr] : typeof LRNAME !== "undefined" && S.lr && S.lr !== "td3bc" ? LRNAME[S.lr] + " " : "";\n'
  '      const rr = !!learnedEp().meta.learner;\n'
  '      const lp = (rf ? lrQs() : LQ).map(q => num(-learnedEp(q).g[t], 2) + " for the " + nm + "actor learned from " + q + " data" + (q === S.lq ? " (the one on screen)" : ""));\n'
  '      return ["Cost so far at state " + t, "At this state the cost so far is " + lp.slice(0, -1).join(", ") + " and " +\n'
  '        lp[lp.length - 1] + ". " + (rf ? "Both" : "All five") + " learned from " + (S.ld === "gap" ? "data with the 80% hold removed" : "all the data") +\n'
  '        (rf ? ", in the cloud rerun of result F\'s arms for seed 0," : rr ? ", in the cloud rerun of seed 0," : "") +\n'
  '        (typeof isSit4 === "function" && isSit4(S.lm) ? " and start from start 11 of " + lmName(S.lm)\n'
  '          : " and start from the same start of the " + (S.lm === "strong" ? "strong test mix" : "training mix")) +\n'
  '        ". The cost so far is minus the return so far, so lower is better."];',
  '      const rf = typeof isRF === "function" && isRF(S.lr);\n'
  '      const nm = rf ? RFT.tipnm[S.lr] : typeof LRNAME !== "undefined" && S.lr && S.lr !== "td3bc" ? LRNAME[S.lr] + " " : "";\n'
  '      const rr = !!learnedEp().meta.learner;\n'
  '      const QN = { random: "Random", "medium-replay": "Medium-replay", medium: "Medium", "medium-expert": "Medium-expert", expert: "Expert" };\n'
  '      const qs = rf ? lrQs() : LQ;\n'
  '      const lines = qs.map(q => (QN[q] || q) + " data: " + num(-learnedEp(q).g[t], 2) + (q === S.lq ? " (the one on screen)" : ""));\n'
  '      const keys = qs.map(q => [typeof LQVAR === "object" && LQVAR[q] ? LQVAR[q] : "--muted", "line"]);\n'
  '      lines.push((rf ? "Both " : "All five ") + nm + "actors learned from " + (S.ld === "gap" ? "data with the 80% hold removed" : "all the data") +\n'
  '        (rf ? ", in the cloud rerun of result F\'s arms for seed 0," : rr ? ", in the cloud rerun of seed 0," : "") +\n'
  '        (typeof isSit4 === "function" && isSit4(S.lm) ? " and start from start 11 of " + lmName(S.lm)\n'
  '          : " and start from the same start of the " + (S.lm === "strong" ? "strong test mix" : "training mix")) +\n'
  '        ". The cost so far is minus the return so far, so lower is better."); keys.push(null);\n'
  '      return ["Cost so far at state " + t, lines.join("\\n"), JSON.stringify(keys)];')
'''
src = src[:i] + NEW + src[j:]
(rep.W7 / "color" / "apply_color.py").write_text(src, encoding="utf-8", newline="\n")
print("wrote color/apply_color.py (K13b)")
