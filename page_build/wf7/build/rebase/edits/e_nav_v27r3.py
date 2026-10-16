"""Rebase onto the live page's service version 1792181606-4140: the old Choices script's "Learner" row names result
F's actor-only learners too (No extra caution, Result F with exact or estimated weights); nav.js carries the same names
and E8's block hash for 4140 is added to KNOWN."""
import rep

rep.apply([
    ("nav/src/nav.js",
     '        named({ td3bc: "TD3+BC", bca: "BCA", bcaiw: "BCA IW" }, "data-tog-value"), { idle: simFixed }),\n',
     '        named({ td3bc: "TD3+BC", bca: "BCA", bcaiw: "BCA IW", le_none: "No extra caution", rf_exact: "Result F, exact weights",\n'
     '          rf_est: "Result F, estimated weights" }, "data-tog-value"), { idle: simFixed }),\n', 1),
    ("nav/apply_nav.py",
     'KNOWN["E8"].add("b0cd8cc4deca8827")',
     'KNOWN["E8"].add("b0cd8cc4deca8827")\n'
     '# ... and the live page after version 26 (service version 1792181606-4140): its "Learner" row names result F\'s learners\n'
     'KNOWN["E8"].add("954959ae7c56fda0")', 1),
])
