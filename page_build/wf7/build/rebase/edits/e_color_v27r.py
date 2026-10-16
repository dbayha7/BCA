"""Rebase onto the live page's version 26 (service version 1792172582-8406, published by the other session on 16 Oct
2026 at 17:43 UTC): the color patch's two anchors that version 26 changed.
K13b: version 26 added the main test's four situations to the learned actors' cost readout (isSit4 / lmName); the
anchor and the replacement both carry that branch, so the keyed readout names the situation as version 26 does.
B13: version 26 already says the toy notebooks were committed ("committed with their results on 15 Oct"), so the
edit is dropped."""
import rep

OLD_LINE = ('  \'        " and start from the same start of the " + (S.lm === "strong" ? "strong test mix" : "training mix") +\\n\'\n')
NEW_LINES = ('  \'        (typeof isSit4 === "function" && isSit4(S.lm) ? " and start from start 11 of " + lmName(S.lm)\\n\'\n'
             '  \'          : " and start from the same start of the " + (S.lm === "strong" ? "strong test mix" : "training mix")) +\\n\'\n')
B13 = ('E("B13", "text", "After the stop: the toy notebooks are now committed (branch forked, merged into main at 21:36 UTC)",\n'
       '  "(exploratory notebooks, not yet committed\\n      to the repository)",\n'
       '  "(exploratory notebooks, committed on 15 Oct\\n      to the repository\'s branch <code>forked</code> and merged into its main branch at 21:36 UTC)")\n')
B13_NOTE = ('# B13 (the toy notebooks committed) is dropped since the rebase onto the live page\'s version 26 (service version\n'
            '# 1792172582-8406), which already says so: "(exploratory notebooks, committed with their results on 15 Oct)".\n')
rep.apply([
    ("color/apply_color.py", OLD_LINE, NEW_LINES, 2),
    ("color/apply_color.py", B13, B13_NOTE, 1),
])
