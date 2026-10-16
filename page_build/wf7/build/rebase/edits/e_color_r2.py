"""Review round 2 (16 Oct 2026, cloud): the color patch's status statements and text fixes. Exact replacements in
color/apply_color.py's source (each old text must occur exactly once there). Findings: A2/C1/F1 (B26, B26b: the route
table's Stage 2 row), F3 (B19 and B19b: the critic viewer's sources), F4 (B27: EXPERIMENTS.md at the head named),
F5 (B22: the four learners named), F6 (B21: the near-80 test is the version 8 the GO licensed), F7 (B28: After the
stop's opening), F18 (B5: "changed", past tense)."""
import rep

P = "color/apply_color.py"
NEW_EDITS = r'''
# Review round 2 (cloud, 16 Oct 2026): status statements that still read as if version 7 would run, and the sources'
# notes that the newer head made out of date.
E("B26", "text", "After the stop, route table: Stage 2's row says version 7 was ended without running (review A2, C1, F1)",
  "<span class=\"tag\">waiting</span> Seeds 0–4 missed marks M4 and M5; seeds 5–9 are to run under version 7.",
  "<span class=\"tag\">ended</span> Seeds 0–4 missed marks M4 and M5; seeds 5–9 were to run under version 7, which was\n"
  "ended on 16 Oct 2026 without running.")
E("B26b", "text", "After the stop, route table's caption: Stage 2's row is dated apart",
  "<caption>Routes tested on the main test's data and their status, as of 15 Oct 2026, 05:41 UTC.</caption>",
  "<caption>Routes tested on the main test's data and their status, as of 15 Oct 2026, 05:41 UTC (Stage 2's row as of\n"
  "16 Oct 2026).</caption>")
E("B19b", "text", "critic viewer, sources: the code and scripts were only local until the merge; the runs still are (review F3)",
  "are so far only in the\n      project owner's local workspace (<code>bca-shift-work</code>, which is not under version "
  "control) and in a local\n      clone that has not been pushed. No branch of the shared repository holds them, so this "
  "section cannot yet be\n      rebuilt from it.",
  "were, until that merge, only in the\n      project owner's local workspace (<code>bca-shift-work</code>, which is not "
  "under version control) and in a local\n      clone that had not been pushed; the runs still are.")
E("B27", "text", "After the stop, sources: EXPERIMENTS.md at the head the paragraph names (review F4)",
  "The repository's <code>EXPERIMENTS.md</code> was last\n      updated on 13 Oct (<code>1d2ef73</code>) and still calls "
  "the policy test running; the files above are\n      newer.",
  "The repository's <code>EXPERIMENTS.md</code> at that head (its header says last updated 14 Oct; its entry 17, the\n"
  "      weighting test, came in with the weighting test's merge on 15 Oct) still calls Stage 2's policy test running; "
  "the files above are\n      newer.")
E("B28", "text", "After the stop, opening: seeds 5-9 could carry Stage 2's claim and have not run (review F7)",
  "only its fresh seeds 5–9, not yet run, can.</p>",
  "only its fresh seeds 5–9 could, and they have not run: version 7, which was to run them, was ended on 16 Oct 2026\n"
  "      without running.</p>")
'''
rep.apply([
    (P, "\n\ndef ids_of(text):", NEW_EDITS + "\n\ndef ids_of(text):", 1),
    # F3: B19's hedge goes (B19b corrects the sentences it hedged)
    (P, '"      section still cannot be rebuilt from the repository alone. The rest of this note was written before that merge.\\n"',
        '"      section still cannot be rebuilt from the repository alone.\\n"', 1),
    # F6: B21's near-80 item says it is the version 8 that the restructured BCA's GO licensed
    (P, '"        <li><b>Near-80 test</b> (approved and running; no result yet): a test that also leaves the commands near 80%\\n"',
        '"        <li><b>Near-80 test</b> (approved and running; no result yet): version 8, the version the restructured BCA\'s GO\\n"\n'
        '  "          licensed, with the restructured BCA as its main method and fresh seeds 10–14; it also leaves the commands near 80%\\n"', 1),
    # F5: B22 names the four learners
    (P, '"      learners, whose plan the project owner approved in chat and must still approve at their own terminal, and a test\\n"',
        '"      learners (TD3+BC, implicit Q-learning, conservative Q-learning and revisited behavior-regularized actor-critic,\\n"\n'
        '  "      each alone, with BCA and with BCA IW; nothing trained yet), whose plan the project owner approved in chat and must\\n"\n'
        '  "      still approve at their own terminal; and a test\\n"', 1),
])
src = (rep.W7 / P).read_text(encoding="utf-8")
# F18: B5's new text in the past tense ("changed"), its old text (the live page's) unchanged
i = src.index('E("B5", ')
j = src.index('\nE("B6", ', i)
blk = src[i:j]
old = '"version 7, which changes only where and how\\n"'
assert blk.count(old) == 2, blk.count(old)
k = blk.rindex(old)
blk = blk[:k] + '"version 7, which changed only where and how\\n"' + blk[k + len(old):]
src = src[:i] + blk + src[j:]
(rep.W7 / P).write_text(src, encoding="utf-8", newline="\n")
print("wrote", P, "(B5 tense)")
