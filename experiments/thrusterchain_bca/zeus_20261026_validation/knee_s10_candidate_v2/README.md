# Candidate knee_s10_v2: v1 with an in-support anchor prefix

One change from `../knee_s10_candidate_v1/`: the anchor prefix rule (`prefix_rule.py`). Everything else is v1's by
reference, pinned by hash in the result: the widened recorder, pi_ref v2, OOD definition v4 r3, the formal driver's
measurements, bounds and labels (`../knee_s10_candidate_v1/formal_g2/g2_formal.py`, imported unchanged).

Why: v1's formal G2 read NOT DATA-OOD because its stall3-run prefixes carried 12 of 96 anchors outside the no-stall
data. v2's prefix never commands past the knee, so the prefix trajectory is the same in both worlds and each anchor is
one physical state for both D3 checks. The rule, written first, is `G2_DEV_RULE_v2.md`.

| File | What it is |
|---|---|
| `G2_DEV_RULE_v2.md` | the development rule, written before the screen ran |
| `prefix_rule.py` | the v2 prefix: the recorder's draws mapped onto CUTOFF's retained support; builds each anchor in both worlds and refuses unless they agree bit for bit |
| `g2_dev_v2.py` | the development screen (v1's development entropy and stream, 48 gated anchors, paired with v1's development r3 result) |
| `test_v2.py` | tests of the prefix rule |
| `g2_development_result_v2.json` | the screen's record (written by the run) |

Tests, then the screen (Windows venv, about a minute together, no GPU), from `C:\Users\David\dev\BCA`:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -B -m unittest discover -s experiments/thrusterchain_bca/zeus_20261026_validation/knee_s10_candidate_v2 -p "test_*.py"
```

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -B experiments/thrusterchain_bca/zeus_20261026_validation/knee_s10_candidate_v2/g2_dev_v2.py
```

A development PASS admits nothing; it lets a formal G2 for v2 be sealed (new seed namespace, fresh streams, 96 gated
anchors, the user's approval) as the rule says.
