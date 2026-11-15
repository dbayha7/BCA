"""Check every number on the meeting slides against the saved result files. Reads only; trains nothing.

    python -I -B check_meeting_numbers.py

The deck is "BCA on a Published Gridworld" (MEETING_SUMMARY.md). Each line prints the slide, the claim,
the number as shown on the slide and the number computed here from `results/`. A number matches when
it equals the saved value rounded to the digits the slide shows.
"""
import json
import statistics as st
import sys
from pathlib import Path

R = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "results"
rows = []


def load(path):
    return json.loads((R / path).read_text())


def check(slide, claim, shown, value, digits=3, exact=False):
    ok = (value == shown) if exact else (round(float(value), digits) == round(float(shown), digits))
    rows.append((slide, claim, shown, value if exact else round(float(value), digits + 2), "ok" if ok else "MISMATCH"))


def runs(records, recipe, method):
    return [r for r in records if r["recipe"] == recipe and r["method"] == method]


def mean(records, recipe, method, key="score"):
    return st.mean(r[key] for r in runs(records, recipe, method))


GUARD, STANDARD = "support_blind_width", "baseline"      # the deck's "BCA policy guard" and "standard learner"

# --- the environment, the problem and the standard learner ------------------------------------
clone = load("published_cloning_check.json")
check("map", "best possible player reaches the goal", 0.951, clone["optimal_goal_rate"])
for recipe, shown, digits in (("optimal", 1.00, 2), ("other_starts", 1.00, 2), ("mixed", 0.997, 3)):
    check("backup: standard learner", f"copying the recordings, {recipe}", shown,
          st.mean(d["score"] for d in clone["datasets"] if d["recipe"] == recipe), digits)

first, second = load("published_validation/records.json"), load("published_validation_2/records.json")
check("backup: standard learner", "as published (alpha 2.5), optimal", 0.75, mean(first, "optimal", STANDARD), 2)
check("backup: standard learner", "after the fix (alpha 1.0), optimal", 0.945, mean(second, "optimal", STANDARD))
check("backup: standard learner", "runs at 0.90 or more, optimal", 15,
      sum(r["score"] >= 0.90 for r in runs(second, "optimal", STANDARD)), exact=True)
pilot = {float(k): v for k, v in load("pilot_alpha.json")["by_alpha"].items()}
check("backup: standard learner", "same data, setting as published", 0.72, pilot[2.5]["recipes"]["optimal"]["mean"], 2)
check("backup: standard learner", "same data, setting after the fix", 0.95, pilot[1.0]["recipes"]["optimal"]["mean"], 2)

for recipe in ("optimal", "other_starts"):
    check("problem / remember", f"runs where the scorer rates a step into lava above the safe move, {recipe}", 15,
          sum(r["critical"]["critic_rates_a_fatal_move_above_the_way_through"] for r in runs(second, recipe, STANDARD)),
          exact=True)
at_cell = [r["critical"] for r in runs(second, "other_starts", STANDARD)]
check("problem", "a step into lava is truly worth", 0.04,
      st.median(st.mean(c["true_value"][m] for m in ("left", "right")) for c in at_cell), 2)
check("problem", "the scorer rates a step into lava at about", 0.7,
      st.median(st.mean(c["critic"][m] for m in ("left", "right")) for c in at_cell), 1)

# --- the policy guard: confirmation on nine new datasets ---------------------------------------
conf = load("confirm_version_a/confirmation_records.json")
for recipe, before, after in (("optimal", 0.953, 0.990), ("other_starts", 0.986, 0.987), ("mixed", 0.885, 0.909)):
    check("policy guard", f"standard learner, {recipe}", before, mean(conf, recipe, STANDARD))
    check("policy guard", f"policy guard, {recipe}", after, mean(conf, recipe, GUARD))
check("policy guard", "runs ending in lava, standard learner", 0.086, mean(conf, "optimal", STANDARD, "lava"))
check("policy guard", "runs ending in lava, policy guard", 0.058, mean(conf, "optimal", GUARD, "lava"))
check("policy guard", "runs ending in lava, best possible player", 0.049, 1 - clone["optimal_goal_rate"])
check("policy guard", "training runs per learner", 27, len(runs(conf, "optimal", GUARD)), exact=True)
check("policy guard (script)", "half-random runs that did badly (below 0.50)", 2,
      sum(r["score"] < 0.50 for r in runs(conf, "mixed", GUARD)), exact=True)
claims = load("confirm_version_a/confirmation.json")["claims"]
gain = claims["C2_helps_on_optimal_data"]
check("policy guard", "new datasets where BCA scored higher", 9, gain["datasets_higher"], exact=True)
rows.append(("policy guard", "chance of luck under 1 in 500", "< 0.002", gain["p"], "ok" if gain["p"] < 1 / 500 else "MISMATCH"))
for name in ("D1_more_than_decisiveness", "D2_the_true_cell_matters"):
    check("policy guard (script)", f"datasets where the guard beat the control, optimal ({name[:2]})", 9,
          claims[name]["optimal"]["datasets_higher"], exact=True)

# --- scorer correction: four builds on the three study datasets ---------------------------------
BUILDS = {"version_c": ("first build", 0.527), "version_c2": ("+ charge limit held fixed", 0.924),
          "version_c3": ("+ no charge on the look-ahead", 0.956), "version_c4": ("+ no jitter on the look-ahead", 0.865)}
recs = {m: load(f"{m}_published/version_c_records.json") for m in BUILDS}
base = recs["version_c4"]
for recipe, standard, guard in (("optimal", 0.942, 0.989), ("other_starts", 0.986, 0.987), ("mixed", 0.801, 0.875)):
    check("whole picture / attempts", f"standard learner, {recipe}", standard, mean(base, recipe, STANDARD))
    check("whole picture / attempts", f"policy guard, {recipe}", guard, mean(base, recipe, GUARD))
check("whole picture", "policy guard's worst run, mixed", 0.084, min(r["score"] for r in runs(base, "mixed", GUARD)))
for m, (name, shown) in BUILDS.items():
    check("attempts", f"scorer correction, {name}, mixed", shown, mean(recs[m], "mixed", m))
    for recipe in ("optimal", "other_starts"):
        check("scorer", f"{name}: runs still preferring lava, {recipe}", 0,
              sum(r["critical_critic_rates_fatal_above"] for r in runs(recs[m], recipe, m)), exact=True)
for recipe in ("optimal", "other_starts"):
    check("whole picture", f"scorer correction, best build, {recipe}", 0.990, mean(recs["version_c3"], recipe, "version_c3"))
    check("whole picture (script)", f"scorer correction, latest build, {recipe}", 0.990,
          mean(recs["version_c4"], recipe, "version_c4"))
    for m, name in ((STANDARD, "standard learner"), (GUARD, "policy guard")):
        check("scorer", f"{name}: runs preferring lava, {recipe}", 9,
              sum(r["critical_critic_rates_fatal_above"] for r in runs(base, recipe, m)), exact=True)
best = {(r["data_seed"], r["train_seed"]): r["score"] for r in runs(recs["version_c3"], "mixed", "version_c3")}
latest = {(r["data_seed"], r["train_seed"]): r["score"] for r in runs(recs["version_c4"], "mixed", "version_c4")}
check("whole picture", "scorer correction's worst run, mixed (best build)", 0.888, min(best.values()))
check("attempts", "latest build: runs at 0.86 or more", 8, sum(s >= 0.86 for s in latest.values()), exact=True)
check("attempts (script)", "latest build: runs beating the build before", 6, sum(latest[k] > best[k] for k in latest), exact=True)
check("attempts (script)", "latest build: middle run", 0.979, st.median(latest.values()))
check("scorer", "latest build: runs still wrong on mixed", 3,
      sum(r["critical_critic_rates_fatal_above"] for r in runs(recs["version_c4"], "mixed", "version_c4")), exact=True)
check("scorer (script)", "build before: runs wrong on mixed", 9,
      sum(r["critical_critic_rates_fatal_above"] for r in runs(recs["version_c3"], "mixed", "version_c3")), exact=True)
check("attempts", "gap to the best possible player, standard learner: 20%", 0.20, 1 - mean(base, "mixed", STANDARD), 2)
check("attempts", "gap to the best possible player, best build: 4%", 0.04, 1 - mean(recs["version_c3"], "mixed", "version_c3"), 2)

# --- the scorer's drift and the checks before training -------------------------------------------
before = load("diag_version_c3/drift_summary.json")["summary"]
after = load("version_c4_published/drift_summary.json")["summary"]
check("next", "scorer below the truth, jittered next move, optimal", 3.25, -before["optimal 20000"]["error_at_recorded_commands"], 2)
check("next", "scorer below the truth, exact next move, optimal", 0.02, -after["optimal 20000"]["error_at_recorded_commands"], 2)
check("next", "the same on mixed, before", 14.3, -before["mixed 20000"]["error_at_recorded_commands"], 1)
check("next", "the same on mixed, after", 0.11, -after["mixed 20000"]["error_at_recorded_commands"], 2)
checks = load("iw_checks_published_validation_2_refit10000.json")
check("scorer correction", "gives standard BCA back exactly: runs", 27, sum(r["check1_exact"] for r in checks["runs"]), exact=True)

# --- the setup-validation table and standard BCA (slides added November 13) ----------------------
for recipe, published, fixed in (("optimal", 0.753, 0.945), ("other_starts", 0.791, 0.986), ("mixed", 0.532, 0.843)):
    check("setup validation", f"standard learner, as published, {recipe}", published, mean(first, recipe, STANDARD))
    check("setup validation", f"standard learner, after the fix, {recipe}", fixed, mean(second, recipe, STANDARD))
check("setup validation", "as published: runs at 0.90 or more, optimal", 2,
      sum(r["score"] >= 0.90 for r in runs(first, "optimal", STANDARD)), exact=True)
plain = load("standard_bca_published/standard_bca_records.json")
for recipe, shown in (("optimal", 0.943), ("other_starts", 0.986), ("mixed", 0.817)):
    check("standard BCA", f"standard BCA, {recipe}", shown, mean(plain, recipe, "bca"))
check("standard BCA", "gain on the hardest data", 0.017, mean(plain, "mixed", "bca") - mean(plain, "mixed", STANDARD))
seeds = sorted({r["data_seed"] for r in plain})
check("standard BCA", "datasets where it scored higher, mixed", 3,
      sum(st.mean(r["score"] for r in runs(plain, "mixed", "bca") if r["data_seed"] == d)
          > st.mean(r["score"] for r in runs(plain, "mixed", STANDARD) if r["data_seed"] == d) for d in seeds), exact=True)
check("standard BCA", "margin charged to a step into lava, optimal", 0.06, mean(plain, "optimal", "bca", "critical_width_fatal"), 2)
check("standard BCA", "margin charged to the safe move, optimal", 0.06, mean(plain, "optimal", "bca", "critical_width_through"), 2)
for recipe in ("optimal", "other_starts"):
    check("standard BCA / scorer", f"standard BCA: runs where the scorer prefers lava, {recipe}", 9,
          sum(r["critical_critic_rates_fatal_above"] for r in runs(plain, recipe, "bca")), exact=True)
check("remember", "the scorer rates a step into lava at about", 0.7,
      st.median(st.mean(c["critic"][m] for m in ("left", "right")) for c in at_cell), 1)

width = max(len(r[1]) for r in rows)
bad = [r for r in rows if r[4] != "ok"]
for slide, claim, shown, value, verdict in rows:
    print(f"{verdict:8s} | {slide:24s} | {claim:{width}s} | shown {shown} | saved {value}")
print(f"\n{len(rows)} numbers checked, {len(bad)} mismatches")
sys.exit(1 if bad else 0)
