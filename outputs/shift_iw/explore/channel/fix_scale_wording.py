"""Make the 'scale alone' sentence precise in the results write-up and EXPERIMENTS.md (both copies). Exact, asserted."""
from pathlib import Path

files = {
    Path("/root/bca-shift/EXPERIMENTS.md"): [
        ("In the room checks, multiplying perfectly informed caution by 1.5\n    moves return by 0.0–0.7 points.",
         "In the room checks, scaling caution by 1.5 moves return by\n    0.0–0.1 points for the perfectly informed lower "
         "edge and 0.4–0.7 for the even pull."),
        ("and a rescale of that size moves return by at most 0.7 points even with perfect\n  information.",
         "and in the room checks a rescale of that size moves return by at most 0.7 points,\n  even for perfectly "
         "informed caution."),
    ],
}
for p in (Path("/mnt/c/Users/David/dev/bca-shift-work/channel/RESULTS_coverage_at_actor.md"),
          Path("/root/bca-shift/outputs/shift_iw/explore/channel/RESULTS_coverage_at_actor.md")):
    files[p] = [("multiplying perfectly informed caution by 1.5 changes return by 0.0–0.1 points (lower edge, oracle ×1 "
                 "against ×1.5) or 0.4–0.7 points (even caution ×1.5,\n  capped-pull family).",
                 "scaling caution by 1.5 changes return by 0.0–0.1 points for the perfectly informed lower edge (oracle "
                 "×1 against ×1.5)\n  and 0.4–0.7 points for the even pull (×1.5 against none).")]
for p, reps in files.items():
    t = p.read_text(encoding="utf-8")
    for old, new in reps:
        assert t.count(old) == 1, (p, t.count(old), old[:60])
        t = t.replace(old, new)
    p.write_text(t, encoding="utf-8")
    print("updated", p)
