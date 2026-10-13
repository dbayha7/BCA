"""Correct entry 17's past-80% figure in /root/bca-shift/EXPERIMENTS.md to the range recomputed from the training files
(docs/thrusterchain/notes/p4_facts.md, section 8). Exact, asserted."""
from pathlib import Path

P = Path("/root/bca-shift/EXPERIMENTS.md")
t = P.read_text(encoding="utf-8")
old = ("  - **Data fact** (confirmed from the training files): the version-5 data hold no thruster-1 commands past +80% in the\n"
       "    expert data, and 3.6–9.9% in the other qualities, all random. The critic therefore guesses past 80% with or without\n"
       "    the hold removed (the page's \"critic in action\" section shows it).")
new = ("  - **Data fact** (recomputed from the training files, data seeds 0–4): in the expert data no command of any thruster\n"
       "    goes past 80%. In the other qualities, 3.25–10.11% of all training rows push thruster 1 upstream past 80%\n"
       "    (medium-expert 3.25–3.53%, random 9.98–10.11%), and every one is a uniform random command. The critic therefore\n"
       "    guesses past 80% with or without the hold removed (the page's \"critic in action\" section shows it). An earlier\n"
       "    note quoted 3.6–9.9%, which does not reproduce exactly (`docs/thrusterchain/notes/p4_facts.md`, section 8).")
assert t.count(old) == 1, t.count(old)
P.write_text(t.replace(old, new), encoding="utf-8")
print("updated", P)
