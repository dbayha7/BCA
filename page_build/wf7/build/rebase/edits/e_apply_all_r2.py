"""Review round 2 (cloud, 16 Oct 2026) and the rebase onto the live page's service version 1792181606-4140:
apply_all.py's version-history entry (review F8; result F's learners), I2 (F19), the checks (the route table's and the
lesson's out-of-date wording, 4140's result F learners as version-26 code), and the version-24 axis-title counts,
which leave out the lesson's own script (review A1: the lesson's charts now name both axes)."""
import rep

P = "build/apply_all.py"
rep.apply([
    # F8 and 4140: the entry
    (P, "'      <li><b>Version %s, %s.</b> Four of the project owner\\'s requests, built on the page as version 26 left it, with\\n'\n"
        "        '        everything versions 20 to 26 did kept (the status box under the title stays removed, at the owner\\'s\\n'\n"
        "        '        request): a step-by-step way",
        "'      <li><b>Version %s, %s.</b> Four of the project owner\\'s requests, built on the page as version 26 left it and as\\n'\n"
        "        '        it grew after (result F\\'s learners in the live panel), with everything versions 20 to 26 did kept, the toy\\n'\n"
        "        '        worlds section apart, which is rebuilt around version 26\\'s parts (the status box under the title stays\\n'\n"
        "        '        removed, at the owner\\'s request): a step-by-step way", 1),
    (P, "(some read for the lesson by inference and exact simulation), a small control to try\\n'",
        "(some read for the lesson by inference and exact simulation; a few marked illustrations use made-up\\n'\n"
        "        '        numbers), a small control to try\\n'", 1),
    (P, "short button, the rows of versions 21, 22 and 25 and the main test\\'s four situations included, marked \"not\\n'",
        "short button, the rows of versions 21, 22 and 25, the main test\\'s four situations and result F\\'s learners\\n'\n"
        "        '        included, marked \"not\\n'", 1),
    (P, "runs. It keeps version 26\\'s bottom line,", "runs. It carries over, reworded, version 26\\'s bottom line,", 1),
    (P, "tested and the result that would shelve a claim (set it aside),",
        "tested and (for experiments 2 to 4) the result that would shelve a claim (set it aside),", 1),
    (P, "Of the 18 claims written down before the runs, 14 are alive and 4 shelved.",
        "Of the 18 claims, written down before the runs by the notebooks\\' own record, 14 are alive and 4 shelved.", 1),
    # F19: I2 names both sources of the status
    (P, '        "Its status was last brought up to date from the repository\\n"\n        "      on %s at %s UTC; the box below says where each line of work stands." % (d, t),',
        '        "Its status was last brought up to date, from the repository and from what the project owner reported,\\n"\n        "      on %s at %s UTC; the box below says where each line of work stands." % (d, t),', 1),
    # checks: 4140's parts and code
    (P, "                    (\"<b>Version 26, 16 Oct 2026.</b>\", 1), ('<section id=\"capped-pull\">', 1), ('id=\"sim-sit4-data\"', 1),",
        "                    (\"<b>Version 26, 16 Oct 2026.</b>\", 1), ('<section id=\"capped-pull\">', 1), ('id=\"sim-sit4-data\"', 1),\n"
        "                    # the live page after version 26 (service version 1792181606-4140): result F's learners\n"
        "                    ('id=\"sim-rf-data\"', 1), ('rf_exact: \"Result F, exact weights\",\\n          rf_est: \"Result F, estimated weights\" }', 1),", 1),
    (P, "                 'id=\"capped-why\"', 'id=\"capped-toys\"', 'id=\"capped-fixes\"', \"#capped-pull td.n { white-space: normal; }\"):",
        "                 'id=\"capped-why\"', 'id=\"capped-toys\"', 'id=\"capped-fixes\"', \"#capped-pull td.n { white-space: normal; }\",\n"
        "                 # 4140's result F learners\n"
        "                 \"const isRF = lr => LR_RF.includes(lr)\", \"const rfOK = () =>\", \"const RFT\", 'data-tog-value=\"rf_exact\"',\n"
        "                 'data-tog-value=\"rf_est\"', 'data-tog-value=\"le_none\"', 'id=\"simtog-lr-rf-q\"', \"const lrQs\"):", 1),
    # A1: the lesson's charts name their axes now; its script (the page's end, after its data block) is not version 24's
    (P, "        nb, no = no_toy(base_s).count(need), no_toy(s).count(need)",
        "        nb, no = no_toy(base_s).count(need), no_toy(s[:s.index('<script type=\"application/json\" id=\"lesson-data\">')]).count(need)", 1),
    # the out-of-date wording that review round 2 found
    (P, '                  "seeds 5–9 still test the", "because the notebook is not committed"):',
        '                  "seeds 5–9 still test the", "because the notebook is not committed",\n'
        '                  # review round 2: the route table, After the stop\'s opening, the lesson\'s step 7.2\n'
        '                  "seeds 5–9 are to run under version 7", "not yet run, can", "has not run yet", "n: \'not run yet\'"):', 1),
    (P, '                    ("will not run under version 7, which was ended on 16 Oct 2026 without running.", 1),',
        '                    ("will not run under version 7, which was ended on 16 Oct 2026 without running.", 1),\n'
        '                    ("seeds 5–9 were to run under version 7, which was\\nended on 16 Oct 2026 without running.", 1),', 1),
])
