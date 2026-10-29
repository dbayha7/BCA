"""Review round 2 (cloud, 16 Oct 2026): the toy section. C3 (the bottom line's unit), C3's missed twin (the glossary's
promise), C4 (Why toy worlds), C5 (glossary: exact binomial test, situation), C6 and C7 (chart notes), C8 (the one-case
table), C9 (0.375 and the printed 0.3755), C10 (the caps table fits at 1280), C11 ("crossed"), C13 (the role table
links its records), C16 (two experiment-2 gates introduced), B6 (short labels for the section's tables in "In this
section"), E7 (the claims table's group links are 44 px on touch screens), F11 ("No cap"), F12 (the t-test example is
the notebook's own, outside the experiment's worlds), F17 ("made the same decision as")."""
import rep

B = "toy/build_toy.py"
rep.apply([
    # C3
    (B, "'whether the action beats the safe one on average. With every gate tuned to say yes to a harmful action in '\n"
        "             'at most 5% of seeds in every world, BCA\\'s mean gate",
        "'whether the action beats the safe one on average. With every gate tuned to accept at most 5% of the harmful '\n"
        "             'proposals in every world, BCA\\'s mean gate", 1),
    # C3's twin in the glossary
    (B, "'at most about 5% of seeds.</dd>\\n'",
        "'at most about 5% of seeds (in experiment 3, it accepts at most about 5% of the harmful proposals).</dd>\\n'", 1),
    # C5: two glossary entries, after the gate entry
    (B, "'only if BCA is 95% sure that its average gain over the fallback is above 0.</dd>\\n'",
        "'only if BCA is 95% sure that its average gain over the fallback is above 0.</dd>\\n'\n"
        "             '          <dt>Exact binomial test</dt><dd>A classical test of the tail promise with equal weights: with n '\n"
        "             'tries, b of them below the fallback, it says yes only if b is so small that a world with exactly 10% of '\n"
        "             'outcomes below the fallback would show b or fewer with a chance of at most 5%.</dd>\\n'\n"
        "             '          <dt>Situation</dt><dd>In experiment 3, one decision: a safe action and the candidates the critic '\n"
        "             'proposes; its gains are per situation. (Not the main test\\'s four situations on ThrusterChain.)</dd>\\n'", 1),
    # C4
    (B, "'every true value and every rule\\'s expected return are computed exactly, in seconds, and each notebook '",
        "'every true value is known exactly, a whole experiment runs in seconds to a few minutes, and each notebook '", 1),
    # C6
    (B, '"and the other gates between these lines (table below).",',
        '"and the other gates between these lines, except the t-test harm gate at 5 survivors (0.160; table below).",', 1),
    # C7
    (B, "Dashed lines with hollow markers ignore the shift.",
        "The unweighted mean gate (dashed, hollow triangles) ignores the shift; the gray dashed oracle is a reference.", 1),
    # C8
    (B, "50,000 (5%) let it venture", "50,000 (5%) are needed to venture", 1),
    (B, "; 50 let it venture", "; 50 (5%) are needed to venture", 1),
    # C9
    (B, "is 0.375, not 0.376 as the '", "is 0.375 (0.37548, printed 0.3755), not 0.376 as the '", 1),
    # C10
    (B, '["none qualifies" if v is None else f"{v:.3f}" for v in MG[k]]', '["none" if v is None else f"{v:.3f}" for v in MG[k]]', 1),
    (B, 'note="“None qualifies”: no setting of the t-test stayed within the cap on seeds 0–99.")',
        'note="“None”: no setting of the t-test stayed within the cap on seeds 0–99.")', 1),
    # C16
    (B, "'reward is at least the critic\\'s value of safe. As controls, three gates are fed shuffled rows.</p>')",
        "'reward is at least the critic\\'s value of safe. Two variants also appear in the tables: the step-1 gate compared '\n"
        "             'with context-1 rows only, and the harm test on the average, which ventures unless a one-sided 95% t-test '\n"
        "             'bound shows the survivors\\' average below the critic\\'s value of safe. As controls, three gates are fed '\n"
        "             'shuffled rows.</p>')", 1),
    # E7
    (B, "#toy-worlds .toy-num { white-space: nowrap; }",
        "#toy-worlds .toy-num { white-space: nowrap; }\n"
        "@media (pointer: coarse) { #toy-worlds tr.sub td a { display: inline-flex; align-items: center; min-height: 44px; } }", 1),
    # F11
    (B, 'lambda i: f"Cap {MG_CAPS[i]}: gain per situation, seeds 100–199",',
        'lambda i: ("No cap" if MG_CAPS[i] == "no cap" else f"Cap {MG_CAPS[i]}") + ": gain per situation, seeds 100–199",', 1),
    # F12
    (B, '(None, "A candidate gains 0.4 on 95% of tries and loses 10 on the rest. With 29 tries:"),',
        '(None, "The notebook\'s own example, outside this experiment\'s worlds (whose disasters lose 8): a candidate gains 0.4 "\n'
        '               "on 95% of tries and loses 10 on the rest. With 29 tries:"),', 1),
    # F17
    (B, "'decided as the exact binomial test in 99.6% of cases, and it cannot say yes",
        "'made the same decision as the exact binomial test in 99.6% of cases, and it cannot say yes", 1),
    (B, 'With equal weights BCA decided as the "', 'With equal weights BCA made the same decision as the "', 1),
    ("toy/check_numbers.py", '"decided as the exact binomial test in 99.6% of cases"', '"made the same decision as the exact binomial test in 99.6% of cases"', 1),
    ("toy/check_numbers.py", '"none qualifies | 0.040 | 0.158"', '"none | 0.040 | 0.158"', 1),
    # C11
    ("toy/part_meaning.html", "947 of BCA's 1,000 random draws cleared the bar; 950 were needed",
     "947 of BCA's 1,000 random draws crossed; 950 were needed", 1),
    # B6: short labels for "In this section" (the build adds them to the tables it made)
    (B, '''def build():
    parts =''', '''# Review round 2, B6: short labels for the section's tables in "In this section" (their captions are long)
TABLE_NAV = {
    "toy-first-seed7": "Seed 7: return and regret per episode", "toy-pooled-table": "Experiment 2's six tests, pooled",
    "toy-second-case": "One case, every gate", "toy-second-regret": "Pooled regret by number of survivors, every rule",
    "toy-second-unsafe": "Unsafe-deviation rate by number of survivors", "toy-third-results": "Every gate at its stated setting",
    "toy-third-caps": "Gain at each cap on harm, every gate", "toy-third-tuned": "Claim 5: each gate's best setting",
    "toy-fourth-mix": "The five conditions and their shares", "toy-fourth-claims": "Experiment 4's six claims",
    "toy-fourth-results": "Every gate at its stated setting, 800 seeds", "toy-fourth-rungs": "Deviation rate by rung",
    "toy-fourth-evidence": "Held-out tries each tail gate needs",
}


def add_nav_labels(s):
    for tid, lab in TABLE_NAV.items():
        a = f'<table id="{tid}" '
        assert s.count(a) == 1, tid
        s = s.replace(a, f'<table id="{tid}" data-nav-label="{esc(lab)}" ', 1)
    return s


def build():
    parts =''', 1),
    (B, "    return '<section id=\"toy-worlds\">\\n    ' + body + '\\n  </section>'",
        "    return '<section id=\"toy-worlds\">\\n    ' + add_nav_labels(body) + '\\n  </section>'", 1),
])
