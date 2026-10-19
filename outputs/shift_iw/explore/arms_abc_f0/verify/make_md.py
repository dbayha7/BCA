#!/usr/bin/env python3
"""Writes VERIFIED_NUMBERS.md from verified.json (verify_marks.py) and le_none_check.json (le_none_check.py).
Run: PYTHONDONTWRITEBYTECODE=1 python3 -I make_md.py"""
import hashlib
import json
from pathlib import Path

SP = Path("/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad")
V = SP / "routes2" / "arms" / "verify"
RUNS = SP / "iw_rerun" / "repo" / "runs" / "shift_iw_v5_explore"
v = json.loads((V / "verified.json").read_text())
le = json.loads((V / "le_none_check.json").read_text())
rf = json.loads((SP / "resultF" / "results" / "scores_seed0.json").read_text())

QUALITIES = ("random", "medium-replay", "medium", "medium-expert", "expert")
SEEDS = (0, 1, 2, 3, 4)


def f2(x):
    return "-" if x is None else f"{x:.2f}"


def f4(x):
    return "-" if x is None else f"{x:+.4f}"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


L = []
L += ["# Verified numbers: arms A, C and F0 (EXPLORATORY)", "",
      "**EXPLORATORY.** Seeds 0-4 are seen seeds. Nothing here is confirmatory; a confirmatory claim needs fresh seeds "
      "15-19, its own pre-registration and a new go from the owner.", "",
      "An independent recomputation of every mark of arms A, C and F0 in `routes2/arms/RESULTS.md`, from the per-seed "
      "raw scores, following the frozen file's sections 3.2, 4.2-4.3, 6.2-6.3 and 7.4. It does not import or call "
      "`score.py` or any project code: `verify_marks.py` reads the raw records with plain `json`, and does the "
      "statistics in pure Python (`math.fsum`, `statistics.stdev`), not numpy.", "",
      "- Frozen file: `PASSMARKS_arms_ABC_and_F_control.md`, SHA-256 `" + v["frozen_sha256"] + "` (matches its sidecar).",
      "- `KILLS.json` SHA-256 `" + v["kills_sha256"] + "`; `results.json` SHA-256 `" + v["results_json_sha256"] +
      "` (matches its sidecar); `RESULTS.md` SHA-256 `" + v["results_md_sha256"] + "`.",
      "- Scripts (this folder): `verify_marks.py` (marks and information rows, writes `verified.json`), "
      "`le_none_check.py` (the le_none question, run with the sandbox's interpreter for flax's msgpack reader, writes "
      "`le_none_check.json`), `make_md.py` (this file).", ""]

# ---------------------------------------------------------------- verdict
mm = v["mismatches"]
devs = [t["max_abs_dev"] for t in v["table"] if t["max_abs_dev"] is not None]
L += ["## Verdict", "",
      f"- **Rows recomputed:** {len(v['table'])} (26 arm A marks, 33 arm C rows, 2 F0 marks, 34 information rows).",
      f"- **Mismatches beyond 1e-9:** {len(mm)}. The largest difference between a recomputed number (each d_k, the "
      f"mean, the lower bound) and `results.json` is {max(devs):.1e}. Every 2-decimal rendering in `RESULTS.md` "
      "(mean, seeds positive, lower bound, each d_k, Met) equals the recomputed value rendered the same way.",
      f"- **Raw-record problems:** {len(v['problems'])}. Every evaluation record verifies against its `.sha256`, has "
      "status finished, carries the frozen file's digest and label, names the current `KILLS.json`, and was scored from "
      "the arm record that is there now; its `quality` and `seed` fields equal its path (pairing by seed). Every F0 "
      "record verifies, points at A2's record of the same quality and seed with that record's current SHA-256, has "
      "KF3's two clone digests equal, and has A2's row-stream digest. Every A2 `evaluate_seed<k>.json` names its own "
      "quality and seed and is newer than A2's record. A2's seed-0 evaluation equals the result F rerun's "
      f"`scores_seed0.json` on all three banks (medium-expert: {v['a2_seed0_equals_resultF_scores']['medium-expert']}, "
      f"expert: {v['a2_seed0_equals_resultF_scores']['expert']}).",
      f"- **Arm A:** P1 {v['A']['met']['P1']}, P2 {v['A']['met']['P2']}, P3 {v['A']['met']['P3']}, P4 "
      f"{v['A']['met']['P4']}; passing qualities {', '.join(v['A']['passing'])}; GO {v['A']['GO']}. Same as "
      "`RESULTS.md`, including the outcome line and the rivals it names.",
      f"- **Arm C:** B1 {v['C']['met']['B1']} (passing {', '.join(v['C']['passing'])}), B5 {v['C']['met']['B5']}; B2-B4 "
      "not scored (the guard is void); GO False. Same as `RESULTS.md`.",
      f"- **F0:** credit on {', '.join(v['F0']['credit']) or 'neither quality'}; wording \"{v['F0']['wording']}\". Same "
      "as `RESULTS.md`.",
      "- **Arm B:** no B return entered any mark (section 'Arm B' below).",
      "- **le_none:** arm A's `none` actor is **not** result F's `le_none` actor (section 'Is arm A's none result F's "
      "no-extra-caution actor?'). A2 − none and F0 − none were therefore not computed.", ""]

# ---------------------------------------------------------------- what was checked
L += ["## What was checked, and how", "",
      "- **Per seed** k: d_k = (arm's points) − (comparator's points), same quality, same seed, same bank. Points are "
      "the records' `points` (0 = random-quality data, 100 = stall-aware controller), each over the bank's 1,000 "
      "starts; a number counts only when its `finite` flag is true, it is finite and `starts` is 1,000.",
      "- **Mean** = fsum(d_0..d_4) / 5. **Seeds positive** = count of d_k > 0; a tie (d_k = 0) would count as not "
      "positive. No d_k in any row is exactly 0 (the smallest |d_k| is 0.0007, B2 random).",
      "- **Thresholds are inclusive** and compared at full precision (mean ≥ bar; positives ≥ 4 where named).",
      "- **Lower bound** = mean − 2.132 × SD / √5. **Which SD:** the frozen file says \"SD\" with \"(t, 4 degrees of "
      "freedom)\". A t interval with n − 1 = 4 degrees of freedom is built on the sample SD (divisor n − 1), so the "
      "file implies the sample SD. The code (`score.summary`) uses `numpy.std(ddof=1)`, the sample SD. They agree. "
      "`verified.json` also holds the bound with the population SD (divisor n) for contrast; it is not what the file "
      "implies and is not used. The lower bound decides nothing.",
      "- **Best plain margin** (P4): on each passing quality, the one of `plain_1s`, `plain_2s`, `plain_4s` with the "
      "highest 5-seed mean points on that quality's strong test, chosen after the fact. **Best plain test** (B4): the "
      "one of the five plain tests with the highest 5-seed mean of rho 0.5 + test on that quality's strong test. No "
      "ties for best occurred.",
      "- **P2 and B5** on every bank (none, mild, strong) and all five qualities: 15 rows each.",
      "- **Missing numbers:** none. `KILLS.json` lists seeds 0-4 as usable for every quality of A and C and both F0 "
      "qualities as decided; every number every mark needs is present and finite.", ""]


def mark_table(section, title):
    rows = [t for t in v["table"] if t["section"] == section]
    out = [f"### {title}", "",
           "| Mark | Bar | d_0, d_1, d_2, d_3, d_4 (recomputed) | Mean: mine / RESULTS.md | Seeds positive: mine / "
           "RESULTS.md | Lower bound: mine / RESULTS.md | Met: mine / RESULTS.md | max abs dev vs results.json |",
           "|---|---|---|---|---|---|---|---|"]
    for t in rows:
        m, g = t["mine"], t["results_md"] or {}
        met_m = "yes" if m.get("meets") else "no"
        out.append(f"| {t['label']} | {t['bar']} | {', '.join(f4(x) for x in m['per_seed'])} | "
                   f"{f2(m['mean'])} / {g.get('mean', '?')} | {m['positives']} / {g.get('pos', '?')} | "
                   f"{f2(m['lower_bound'])} / {g.get('lb', '?')} | {met_m} / {g.get('met', '?')} | "
                   f"{t['max_abs_dev']:.1e} |")
    return out + [""]


def info_table(section, title):
    rows = [t for t in v["table"] if t["section"] == section]
    out = [f"### {title}", "",
           "| Row | d_0, d_1, d_2, d_3, d_4 (recomputed) | Mean: mine / RESULTS.md | Seeds positive: mine / RESULTS.md | "
           "Lower bound (mine; not shown in RESULTS.md) | max abs dev vs results.json |",
           "|---|---|---|---|---|---|"]
    for t in rows:
        m, g = t["mine"], t["results_md"] or {}
        if t["label"].startswith("rho0.5+bca-rho1"):
            q = t["label"].split(", ")[1].split(" ")[0]
            mrow = [x for x in v["table"] if x["label"] == f"B5 (rho 0.5 + BCA) - rho 1, {q}/strong"][0]
            g = dict(mean=mrow["results_md"]["mean"], pos=mrow["results_md"]["pos"])
        out.append(f"| {t['label']} | {', '.join(f4(x) for x in m['per_seed'])} | {f2(m['mean'])} / "
                   f"{g.get('mean', '?')} | {m['positives']} / {g.get('pos', '?')} | {f2(m['lower_bound'])} | "
                   f"{t['max_abs_dev']:.1e} |")
    return out + [""]


L += ["## Marks", "",
      "Each d_k is in points, arm minus comparator, strong test unless a bank is named after a slash. 'Mine' is the "
      "recomputed value; 'RESULTS.md' is the cell as printed there (2 decimals). The last column is the largest "
      "absolute difference between the recomputed d_k, mean and lower bound and the full-precision values in "
      "`results.json`.", ""]
L += mark_table("A", "Arm A (sections 4.2-4.3)")
A = v["A"]
L += ["P4's best plain margin, by 5-seed mean strong-test points of the rival:", ""]
for q, r in A["P4"].items():
    pm = r["plain_means"]
    L.append(f"- {q}: " + ", ".join(f"{p} {pm[p]:.4f}" for p in pm) + f" → best **{r['best_plain']['rival']}** "
             f"(RESULTS.md names {r['best_plain']['rival']}). A1 minus every plain margin: " +
             ", ".join(f"{p} {r['all_plain'][p]['mean']:+.2f}" for p in r["all_plain"]) + ".")
L += ["", f"P1 passing qualities: {', '.join(A['passing'])} (medium-expert's mean "
      f"{A['P1']['medium-expert']['mean']:.4f} is below +2.0 although 5 of 5 seeds are positive). P3 and P4 are "
      "judged on the passing qualities only, as section 4.3 says. P4 fails on both passing qualities whatever plain "
      "margin is taken as best: A1 − margin0 and A1 − matched_share already fail on medium-replay, and "
      "A1 − matched_share fails on medium.", "",
      "Outcome line, reconstructed from the frozen wording (P1, P2, P3 met, P4 not), equals RESULTS.md's: \"" +
      v["a_wording"] + "\"", ""]
L += mark_table("C", "Arm C (sections 6.2-6.3)")
C = v["C"]
L += [f"B1 passing qualities: {', '.join(C['passing'])}. B5 holds on all 15 quality-bank pairs.", "",
      "**B2, B3, B4 are not scored.** Section 6.4 voids the guard when B's K1 fires, or when C's own map has a "
      "thruster-1 ceiling of 0.85 or higher with at least 30 tries in rung (0.80, 0.85]. Both hold: B's K1 fired at "
      f"the seed-0 screen on {', '.join(C['guard']['B_K1_fires_on'])}, and C's own map meets the condition on "
      f"{len(C['guard']['own_map_void'])} seed-quality pairs (read here from the C records' rung counts; the list "
      "equals KILLS.json's `guard_void.by_own_map`). The rows above give their values for information; RESULTS.md "
      "prints them with Met 'no' and its summary line gives None. Had they been scored, every one of them would fail "
      "its bar by a wide margin (largest B2 mean +0.58 against +1.5; largest B4 mean −0.01 against +1.0), so "
      "nothing turns on it.", "",
      "B4's best plain test, by 5-seed mean strong-test points of rho 0.5 + test:", ""]
for q, r in C["B4"].items():
    pm = r["plain_means"]
    L.append(f"- {q}: " + ", ".join(f"{t} {pm[t]:.4f}" for t in pm) + f" → best **{r['best_plain_test']}** "
             f"(RESULTS.md names {r['best_plain_test']}).")
L += [""]
L += mark_table("F0", "F0 (section 7.4)")
L += [f"Credit needs mean ≥ +1.0 and 4 of 5 seeds positive: met on neither quality. The 'lowered return' line needs "
      "mean ≤ −1.0 with 4 of 5 negative: on neither. The outcome is the frozen 'credit on neither' wording, as "
      "RESULTS.md prints it.", ""]

L += ["## Information rows (decide nothing)", ""]
L += info_table("A-info", "Arm A: even_x1.33 − none and A1c − none (strong test)")
L += info_table("C-info", "Arm C: against rho 1 (strong test)")
L += ["`rho0.5+bca-rho1` on the strong test is B5's strong row; RESULTS.md prints it only there, so its RESULTS.md "
      "cells above are taken from that row.", ""]
L += info_table("F0-info", "F0: A2 − F0 on the mild and none banks")

# ---------------------------------------------------------------- arm B
B = v["B"]
L += ["## Arm B: no B return entered any mark", "",
      f"- `KILLS.json`: B `go_on` {B['kills_go_on']}, stopped by {B['kills_stopped_by']} (BCA's thruster-1 ceiling "
      f"0.85 or higher with at least 30 tries in rung (0.80, 0.85] at the seed-0 screen on "
      f"{', '.join(C['guard']['B_K1_fires_on'])}).",
      f"- No B evaluation exists: `{RUNS}/arms_abc_f0/eval/B` exists: {B['eval_B_dir_exists']}.",
      f"- `results.json` read no B record (inputs naming an `eval/B/` or `arms_abc_f0/B/` path: "
      f"{len(B['results_inputs_with_B'])}). Every per-seed value of G1, G2 (15 rows) and G3 (3 rows) is null "
      f"(non-null: {len(B['results_B_numbers'])}); G4's best plain test is {B['G4_best']} and its total "
      f"{B['G4_total']}; met {B['results_B_met']}; GO {B['results_B_GO']}.",
      "- No C mark reads B's map: the policies in every recomputed C mark are " +
      ", ".join(f"`{p}`" for p in B["C_mark_policies"]) + f" (B's host map used: {B['B_host_map_in_C_marks']}). The "
      "C information row 'rho 0.5 + B_host_map' (section 6.3's 'rho 0.5 with B's host-built map') has a seed-0 value "
      "only, from the screen's map; it enters no mark.", ""]

# ---------------------------------------------------------------- le_none
L += ["## Is arm A's none result F's no-extra-caution actor (le_none)?", "",
      "**No.** They share every input, but they come from different training programs and are not bit-identical.", "",
      "| | medium-expert | expert |", "|---|---|---|"]
me, ex = le["medium-expert"], le["expert"]


def both(fn):
    return f"| {fn[0]} | {fn[1](me)} | {fn[1](ex)} |"


L += [both(("Clone SHA-256 (arm A start / le_none start)",
            lambda d: f"`{d['protocol']['clone_sha256']['arm_A'][:12]}…` / `{d['protocol']['clone_sha256']['le_none'][:12]}…` "
                      f"({'same' if d['protocol']['clone_sha256']['arm_A'] == d['protocol']['clone_sha256']['le_none'] else 'differ'})")),
      both(("Row stream SHA-256 (arm A / le_none)",
            lambda d: f"`{d['protocol']['rows_sha256']['arm_A'][:12]}…` / `{d['protocol']['rows_sha256']['le_none'][:12]}…` "
                      f"({'same' if d['protocol']['rows_sha256']['arm_A'] == d['protocol']['rows_sha256']['le_none'] else 'differ'})")),
      both(("Steps", lambda d: f"arm A: {d['protocol']['steps']['arm_A']}; le_none: {d['protocol']['steps']['le_none']}")),
      both(("Alpha", lambda d: "2.5 both (settings.alpha); first-chunk lambda " +
            f"{d['protocol']['first_chunk_metrics']['arm_A'][2]:.9f} vs {d['protocol']['first_chunk_metrics']['le_none'][2]:.9f}")),
      both(("Penalty", lambda d: "arm A: none (Arm('none', 'even', 1.0)); le_none: k = 0 × the oracle width")),
      both(("Training program", lambda d: "arm A: learners.ActorOnly.program, float32 step, 12 arms vmapped; le_none: "
            "lower_edge.Runner 'oracle', float64 step under jax enable_x64 (Adam's bias correction in float64), 3 arms "
            "vmapped (le_none, le_oracle, le_oracle_x1.5)")),
      both(("First chunk's mean actor loss (arm A / le_none)",
            lambda d: f"{d['protocol']['first_chunk_metrics']['arm_A'][0]:.9f} / {d['protocol']['first_chunk_metrics']['le_none'][0]:.9f}")),
      both(("Final actors bit-identical?", lambda d: f"{d['params']['armA_none_vs_le_none']['bit_identical']}; largest "
            f"parameter difference {d['params']['armA_none_vs_le_none']['max_abs_param_diff']:.4f}")),
      both(("Arm A none vs the v5 room check's 'none' arm", lambda d: f"bit-identical: "
            f"{d['params']['armA_none_vs_room_none']['bit_identical']}")),
      both(("le_none record's own check_b (vs room 'none')", lambda d: f"{d['params']['le_record_check_b']:.4f}")),
      both(("Seed-0 strong test, points: arm A none / le_none", lambda d: f"{d['scores']['arm_A_none']['strong']:.6f} / "
            f"{d['scores']['le_none']['strong']:.6f} (difference {d['scores']['strong_points_diff_A_minus_le']:+.4f})")),
      both(("Seed-0 none / mild banks, points: arm A none / le_none",
            lambda d: f"{d['scores']['arm_A_none']['none']:.4f} / {d['scores']['le_none']['none']:.4f}; "
                      f"{d['scores']['arm_A_none']['mild']:.4f} / {d['scores']['le_none']['mild']:.4f}")), ""]
L += ["Why they differ, exactly: le_none is the k = 0 lane of `lower_edge.room_check`'s float64-step program (traced "
      "under `jax.experimental.enable_x64`, so Adam's bias correction is computed in float64 and rounded, vmapped with "
      "the two oracle lanes). Arm A's `none` is `learners.ActorOnly.program`'s float32 step (vmapped with the other 11 "
      "arms), and it is bit-identical to the version-5 room check's `none` arm. `lower_edge.py`'s own docstring says "
      "the k = 0 arm 'can differ from learners.ActorOnly's none arm by rounding', and the room_lower_edge record "
      "measures that difference (check_b: 0.0525 on medium-expert, 0.0707 on expert, the same numbers found here "
      "between the two saved actors). The rounding difference shows from the first chunk and grows into different "
      "actors whose seed-0 strong-test scores differ by 0.0065 and 0.0103 points.", "",
      "Same inputs, but not the same code path and not bit-identical, so A2 − none and F0 − none were **not** "
      "computed. (Arm A's none actors for seeds 1-4 exist; result F has no le_none for seeds 1-4 in this sandbox.)", ""]
# frozen 7.4's own seed-0 information item, which RESULTS.md omits
L += ["### Section 7.4's seed-0 information item (absent from RESULTS.md)", "",
      "The frozen file lists, as information only: 'on seed 0, F0 and A2 each minus the result F rerun's no extra "
      "caution actor (le_none)'. RESULTS.md and results.json do not report it. From the raw numbers (strong test, "
      "points; A2 and F0 from this run's records, le_none from `resultF/results/scores_seed0.json`):", "",
      "| Quality | A2 | F0 | le_none | A2 − le_none | F0 − le_none |", "|---|---|---|---|---|---|"]
for q in ("medium-expert", "expert"):
    a2 = json.loads((RUNS / "channel" / "train_f1" / q / "evaluate_seed0.json").read_text())[
        "evaluations"]["A2_cmd_strong"]["strong"]["points"]
    f0 = json.loads((RUNS / "arms_abc_f0" / "F0" / q / "F0_no_band" / "seed0" / "record.json").read_text())[
        "evaluation"]["strong"]["points"]
    ln = rf["scores"][q]["le_none"]["strong"]["points"]
    L.append(f"| {q} | {a2:.4f} | {f0:.4f} | {ln:.4f} | {a2 - ln:+.4f} | {f0 - ln:+.4f} |")
L += ["", "One seed; information only; decides nothing.", ""]

# ---------------------------------------------------------------- raw numbers and their sources
L += ["## Raw numbers and their source files", "",
      "Every raw number is a record's `points` for one policy on one bank. Sources:", "",
      f"- Arm A: `{RUNS}/arms_abc_f0/eval/A/<quality>/seed<k>/record.json` → `evaluations.<policy>.<bank>.points`.",
      f"- Arm C: `{RUNS}/arms_abc_f0/eval/C/<quality>/seed<k>/record.json` → `evaluations.<policy>.<bank>.points`.",
      f"- F0: `{RUNS}/arms_abc_f0/F0/<quality>/F0_no_band/seed<k>/record.json` → `evaluation.<bank>.points`.",
      f"- A2: `{RUNS}/channel/train_f1/<quality>/evaluate_seed<k>.json` → `evaluations.A2_cmd_strong.<bank>.points` "
      "(seeds 1-4 built in this run; seed 0 is the result F rerun's and equals `resultF/results/scores_seed0.json`).",
      "", "The per-row source path of every raw number (arm and comparator, each seed) is in `verified.json` "
      "(`table[].mine.source_arm` and `source_comparator`). Below, the raw points per policy, bank and seed, with the "
      "SHA-256 of each source file.", ""]
L += ["### Source files", "", "| Arm | Quality | Seed | File SHA-256 |", "|---|---|---|---|"]
for arm in ("A", "C"):
    for q in QUALITIES:
        for k in SEEDS:
            p = RUNS / "arms_abc_f0" / "eval" / arm / q / f"seed{k}" / "record.json"
            L.append(f"| {arm} eval | {q} | {k} | `{sha(p)}` |")
for q in ("medium-expert", "expert"):
    for k in SEEDS:
        p = RUNS / "arms_abc_f0" / "F0" / q / "F0_no_band" / f"seed{k}" / "record.json"
        L.append(f"| F0 record | {q} | {k} | `{sha(p)}` |")
        p = RUNS / "channel" / "train_f1" / q / f"evaluate_seed{k}.json"
        L.append(f"| A2 evaluate | {q} | {k} | `{sha(p)}` |")
L += [""]


def raw(arm, q, k, pol, law):
    if arm in ("A", "C"):
        r = json.loads((RUNS / "arms_abc_f0" / "eval" / arm / q / f"seed{k}" / "record.json").read_text())
        return r["evaluations"][pol][law]["points"]
    if pol == "F0":
        r = json.loads((RUNS / "arms_abc_f0" / "F0" / q / "F0_no_band" / f"seed{k}" / "record.json").read_text())
        return r["evaluation"][law]["points"]
    r = json.loads((RUNS / "channel" / "train_f1" / q / f"evaluate_seed{k}.json").read_text())
    return r["evaluations"]["A2_cmd_strong"][law]["points"]


L += ["### Raw points (6 decimals here; full precision in the source files)", "",
      "| Arm | Quality | Policy | Bank | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |", "|---|---|---|---|---|---|---|---|---|"]
pols_A = [("A1", "none mild strong"), ("none", "none mild strong"), ("untargeted", "strong"), ("margin0", "strong"),
          ("plain_1s", "strong"), ("plain_2s", "strong"), ("plain_4s", "strong"), ("matched_share", "strong"),
          ("even_x1.33", "strong"), ("A1c", "strong")]
pols_C = [("rho0.5+bca", "none mild strong"), ("rho1", "none mild strong"), ("rho0.5", "strong"),
          ("rho0.5+range_limit", "strong"), ("rho0.5+shuffled", "strong"), ("rho0.5+matched_global", "strong"),
          ("rho0.5+wmedian", "strong"), ("rho0.5+wmean", "strong"), ("rho0.5+sign", "strong"),
          ("rho0.5+wttest", "strong"), ("rho0.5+wbernstein", "strong"), ("rho0.5+knee_clip", "strong"),
          ("rho1+bca_built_for_rho1", "strong")]
for arm, pols, qs in (("A", pols_A, QUALITIES), ("C", pols_C, QUALITIES),
                      ("F", [("A2", "none mild strong"), ("F0", "none mild strong")], ("medium-expert", "expert"))):
    for q in qs:
        for pol, laws in pols:
            for law in laws.split():
                vals = [raw(arm, q, k, pol, law) for k in SEEDS]
                L.append(f"| {arm} | {q} | {pol} | {law} | " + " | ".join(f"{x:.6f}" for x in vals) + " |")
L += [""]

# ---------------------------------------------------------------- minor notes
L += ["## Notes that change no mark", "",
      "- RESULTS.md prints B2, B3 and B4 with Met 'no'. They are not scored (guard void), which its summary line "
      "shows as None. Reading 'no' as 'not met' would be harmless here, since each would fail its bar anyway.",
      "- Arm C's outcome line in RESULTS.md (\"Halving the pull, guarded by BCA's ceiling, met B1 on random, "
      "medium-replay, medium; B2-B4 are not scored ...\") is not one of section 6.5's fixed wordings; section 6.5 has "
      "none for a void guard. The numbers say the B1 gain is the release itself: rho 0.5 alone − rho 1 is +15.40, "
      "+8.70 and +7.90 on those qualities, and the BCA ceiling adds +0.08, +0.21 and +0.58 over rho 0.5 alone. "
      "Section 6.1 says the release is TD3+BC's alpha and is never credited to BCA.",
      "- Section 7.4's information items 'F0's no-credit share per chunk beside A2's' and the seed-0 'F0 and A2 each "
      "minus le_none' are not reported in RESULTS.md or results.json. The second is computed above. The F0 records "
      "hold A2's per-chunk share (`a2_no_credit_share_per_chunk`).",
      "- The C information row 'rho 0.5 + B_host_map' shows a seed-0 value only (B's maps exist only for seed 0) and "
      "carries no 'stopped by K1' tag. It is C's information item (section 6.3), not a B return, and enters no mark.",
      ""]
(V / "VERIFIED_NUMBERS.md").write_text("\n".join(L))
print("wrote", V / "VERIFIED_NUMBERS.md", len(L), "lines")
