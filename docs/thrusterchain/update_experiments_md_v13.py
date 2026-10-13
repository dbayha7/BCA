"""Record ThrusterChain page version 13 in /root/bca-shift/EXPERIMENTS.md (change log, 2026-10-13). Exact, asserted."""
from pathlib import Path

P = Path("/root/bca-shift/EXPERIMENTS.md")
t = P.read_text(encoding="utf-8")
old = ("  - the ThrusterChain page went through versions 11–12: plain-language tooltips, specification sections first,\n"
       "    Results rewritten for newcomers, and the \"critic in action\" section added.")
new = ("  - the ThrusterChain page went through versions 11–13: plain-language tooltips, specification sections first,\n"
       "    Results rewritten for newcomers, and the \"critic in action\" section added. Version 13 answers the owner's 33\n"
       "    page comments: one validated colour system, a five-step offline-RL pipeline strip tying every setting to its\n"
       "    step, aligned symbol legends, two interactive charts, and the weighting test's plain TD3+BC coverage and\n"
       "    exploratory BCA results. Page and sources: `docs/thrusterchain/`.")
assert t.count(old) == 1, t.count(old)
t = t.replace(old, new)
P.write_text(t, encoding="utf-8")
print("updated", P)
