"""Writes verify/KILLS_CHECK.md from kills_check_raw.json and the spot-check JSONs (all in this folder).
PYTHONDONTWRITEBYTECODE=1 python3 -I -B make_kills_check_md.py
"""
import json
import time
from pathlib import Path

V = Path(__file__).resolve().parent
R = json.loads((V / "kills_check_raw.json").read_text())
SC = json.loads((V / "spot_check.json").read_text())
SM = json.loads((V / "spot_maps.json").read_text())
SK0 = json.loads((V / "spot_k0_kb0_k0.json").read_text())
SKB = json.loads((V / "spot_k0_kb0_kb0.json").read_text())
KA, KB, KC, KF = R["kills"]["A"], R["kills"]["B"], R["kills"]["C"], R["kills"]["F0"]
I, O, RS = R["integrity"], R["ordering"], R["restarts"]
Q5 = ("random", "medium-replay", "medium", "medium-expert", "expert")


def pct(x, d=2):
    return "-" if x is None else f"{100 * x:.{d}f}%"


def f(x, d=3):
    return "-" if x is None else f"{x:.{d}f}"


def yn(b):
    return {True: "yes", False: "no", None: "-"}[b]


L = []
add = L.append
add("# Kill-rule and run-integrity check: arms A, B, C and F0 (EXPLORATORY)")
add("")
add("**EXPLORATORY.** Data seeds 0-4 are seen seeds. Nothing here is pre-registered or confirmatory. A confirmatory "
    "claim would need fresh seeds 15-19, its own pre-registration and a new go from the owner.")
add("")
add(f"Made {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}. Frozen rules: "
    "`routes2/arms/PASSMARKS_arms_ABC_and_F_control.md`, sections 3.3, 4.5, 5.5, 6.4 and 7.3 "
    f"(SHA-256 `{I['passmarks_sha256']}`; `sha256sum -c` against its sidecar: OK).")
add("")
add("## Verdict")
add("")
add("- **Kill decisions:** every decision I rebuilt from the raw records is the same as KILLS.md's: arm A, no rule fired "
    "(all five seeds usable on all five qualities); arm B, stopped at the seed-0 screen by K1 on medium-replay, medium "
    "and medium-expert; arm C, no KB1 or K3 stop, KB0 holds, the guard is void, so B2-B4 are not scored; F0, KF1-KF4 "
    "pass on all ten seed-quality pairs, so both qualities are decided.")
add("- **Ordering:** I found no violation. No final return of A, B or C was computed before every kill rule guarding it had been "
    "checked on all five seeds.")
add("- **Restarts:** six jobs were interrupted (three room checks at the first reclaim and three arm-A jobs at the "
    "second). All six were redone from scratch. No partial file was reused, and no finished record was overwritten.")
add("- **Integrity:** every SHA-256 sidecar matches. The run's own final untouched check passed, and my recheck now "
    "also passes: the sandbox code is unchanged, and so is the repository.")
add("- **Blockers:** none. The minor points at the end change no decision.")
add("")
add("## How this was checked")
add("")
add("- `verify/kills_check.py` (standard library only) reads the arm records, A2's records and evaluations, the package's "
    "stage records, the runner's logs (`nohup_*.out` and per-job logs) and `state/`, and recomputes every rule. KILLS.json, "
    "SCREEN_seed0.json and results.json are read only to compare against. Output: `verify/kills_check_raw.json`.")
add("- I also re-executed parts of the computation, reading only, with the jobs' arithmetic settings "
    "(one CPU, AVX2, one thread). Each time, `find -newer` confirmed that no file in the sandbox changed. "
    "`spot_check.py` recomputes the K2 value of all 25 seed-quality pairs from the saved 8,000-step actors, and both of "
    "KB1's critic gains from C's saved 8,000-step actors. `spot_k0_kb0.py` rebuilds the clone and the band read at the "
    "recorded commands, and recomputes K0's release shares for the 15 seed-quality pairs of medium-replay, medium and "
    "medium-expert. It also recomputes KB0's shares for 15 pairs from C's saved final actors. `spot_maps.py` rebuilds "
    "BCA's ceiling maps: B's 5 seed-0 maps and C's 25 guard maps, every rung's statistic and try count on all four "
    "thrusters. **Every recomputed number equals the recorded one bit for bit.**")
add("- **Not re-executed:** K1, because the step-4,000 actors are not saved. I checked it from the records' refresh "
    "history and from the per-job runtime log lines, which agree with the records on all 125 chunk shares. KF1 and KF2 "
    "were checked from their recorded values only. KF3 was checked against A2's own record.")
add("")

# ------------------------------------------------------------------ A
add("## Arm A (section 4.5)")
add("")
add("### K0, before training: A1's release shares at refresh 0 (the clone)")
add("")
add("Bars: at most 10% of recorder rows and at least 15% of random rows. Binding on seeds 1-4; seed 0 is information only. "
    "The rule is checked on medium-replay, medium and medium-expert only. *Recomputed* means that "
    "`spot_k0_kb0.py` rebuilt the clone (the SHA-256 equals the record's `start_sha256`) and the band read, and got "
    "the same shares bit for bit.")
add("")
add("| Quality | Seed | Recorder rows released | Random rows released | Binding | Passes | Guard-blocked (recorder / random) | Recomputed |")
add("|---|---|---|---|---|---|---|---|")
for q in ("medium-replay", "medium", "medium-expert"):
    for k in range(5):
        p = KA[q]["per_seed"][str(k)]
        s = SK0[f"{q}:{k}"]
        add(f"| {q} | {k} | {pct(p['k0_recorder'])} | {pct(p['k0_random'])} | {yn(p['k0_binding'])} | {yn(p['k0_pass'])} | "
            f"{p['k0_guard_blocked_recorder']} / {p['k0_guard_blocked_random']} | {yn(s['equal'] and s['clone_equal'])} |")
add("")
add("On random (no recorder rows) and expert (no random rows), K0 is not checked, as frozen. For information, A1 releases "
    "these shares of random rows on random: " + ", ".join(pct(KA['random']['per_seed'][str(k)]['k0_random'], 1) for k in range(5)) +
    ". It releases these shares of recorder rows on expert: " + ", ".join(pct(KA['expert']['per_seed'][str(k)]['k0_recorder'], 1) for k in range(5)) + ".")
add("")
add("**Verdict:** no binding K0 failure on seeds 1-4. The closest values are medium-expert seed 1's recorder share "
    "(1.02% against 10%) and medium-replay seed 2's random share (18.93% against 15%). All three qualities would also "
    "pass on seed 0. This matches KILLS.md (no K0 stop).")
add("")
add("### K1, at the step-4,000 refresh (every seed, seed 0 included)")
add("")
add("It fires if A1 releases more than 60% of the kept rows, or if mean D exceeds 3 times mean U. These values come "
    "from the records' refresh index 1 (`step` = 4000).")
add("")
add("| Quality | Seed | Rows released | Mean D | Mean U | Mean D / mean U | Fires |")
add("|---|---|---|---|---|---|---|")
for q in Q5:
    for k in range(5):
        p = KA[q]["per_seed"][str(k)]
        add(f"| {q} | {k} | {pct(p['k1_share'], 1)} | {f(p['k1_D'])} | {f(p['k1_U'])} | {f(p['k1_ratio'])} | {yn(p['k1_fires'])} |")
add("")
add("**Verdict:** K1 fires nowhere. The highest share is 40.9% (medium-replay seed 0) and the highest ratio is 1.728 "
    "(medium-expert seed 0). The runtime log line of each A job (`chunk 2/5, A1 released ...`) agrees with every record. "
    "This matches KILLS.md.")
add("")
add("### K2, at 8,000 steps: A1 - none on the strong test")
add("")
add("A quality stops if the value is -3.0 points or lower on at least 3 of 5 seeds. KILLS ran at 05:42:13, after every "
    "A job had finished (the last at 04:53:07), and before any final A actor was scored (the first `eval_A` started at "
    "05:42:15). All 25 values were recomputed from the saved 8,000-step actors and are equal bit for bit.")
add("")
add("| Quality | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Seeds at or below -3.0 | Fires | Recomputed |")
add("|---|---|---|---|---|---|---|---|---|")
for q in Q5:
    v = KA[q]["k2_values"]
    add(f"| {q} | " + " | ".join(f"{x:.2f}" for x in v) + f" | {len(KA[q]['k2_seeds_at_or_below_minus3'])} | "
        f"{yn(KA[q]['k2_fires'])} | {yn(all(SC[f'{q}:{k}']['K2_equal'] for k in range(5)))} |")
add("")
add("**Verdict:** K2 fires nowhere. The lowest value is medium-expert seed 0 at -1.80. This matches KILLS.md's K2 values "
    "exactly.")
add("")
add("### K3, at any time")
add("")
add("- All 25 A records have status `finished`, and the final actors of every arm that a mark needs are finite.")
add("- All 25 gapped band records have `band_valid` true. The band record's SHA-256 that each A record names equals "
    "the current file.")
add("- No record holds a stop for a refused gate check.")
add("")
add("**Verdict:** no K3 stop. Arm A has no stop on any quality and seeds 0-4 are usable everywhere, the same as KILLS.md.")
add("")

# ------------------------------------------------------------------ B
add("## Arm B (section 5.5)")
add("")
add("B was screened on seed 0 only. Its seed-0 maps were rebuilt by `spot_maps.py` and are equal to the records on all "
    "four thrusters, rung by rung. The walk (the last rung of the unbroken certified run from rung 1) reproduces every "
    "ceiling. On every rung, the `certified` flag equals 'statistic < 0 with at least one try'.")
add("")
add("| Quality | BCA ceilings (t1-t4) | Thruster-1 tries per rung (0.50-0.55 ... 0.95-1.00) | Thruster-1 rungs certified | t1 | Tries in (0.80, 0.85] | K1 fires |")
add("|---|---|---|---|---|---|---|")
for q in Q5:
    b = KB["per_quality"][q]
    add(f"| {q} | {', '.join(f'{x:.2f}' for x in b['t_map'])} | {', '.join(str(x) for x in b['thruster1_tries_per_rung'])} | "
        f"{''.join('1' if c else '0' for c in b['thruster1_certified'])} | {b['t1']:.2f} | {b['tries_080_085']} | {yn(b['k1_fires'])} |")
add("")
add("- **K1** (a thruster-1 ceiling of 0.85 or higher, with at least 30 tries in rung 7, (0.80, 0.85]) **fires on three "
    "qualities.** On medium-replay, t1 is 0.90 with 42 tries in rung 7. On medium, t1 is 0.90 with 54 tries. On "
    "medium-expert, t1 is 0.85 with 78 tries. It does not fire on random (t1 0.80, 12 tries) or expert (t1 0.80, 0 tries).")
add(f"- **K0, room:** on medium-expert seed 0, (host + knee clip) - host = {KB['k0_room_seed0']:.2f} points (bar +1.0), "
    "so it does not fire.")
add("- **K2, over-refusal:** no thruster-1 ceiling is 0.60 or lower (the lowest is 0.80), so it does not fire.")
add("- **K3:** no non-finite value, so it does not fire.")
add(f"- **K4:** the map clips {pct(KB['k4_seed0'], 1)} of the host's thruster-1 commands on medium-expert (bar 2%), "
    "so it does not fire.")
add("- **Which rule stopped B:** K1 at the screen (seed 0), on medium-replay, medium and medium-expert. The screen "
    "records `B_stopped_by = ['K1']`. No B job ran for seeds 1-4: `job_times.tsv` has only the five `B_*_s0` rows, "
    "and there are no B records for seeds 1-4. B's 5-seed-mean checks (K0 and K4) were never computed, which is "
    "correct for an arm stopped at the screen.")
add("- **No B return entered a mark.** There is no `eval_B` job and no `eval/B` record. Every B number in results.json "
    "is null (each `per_seed` is all null and every mean is null). G1-G4 are not met, and every RESULTS.md row for B "
    "shows '-'. The only return-like numbers in B's records are K0's host and knee-clip points, which that rule reads.")
add("")
add("**Verdict:** this matches KILLS.md: B was stopped by K1 on medium-replay, medium and medium-expert.")
add("")

# ------------------------------------------------------------------ C
add("## Arm C (section 6.4)")
add("")
add("### KB1, at 8,000 steps (every seed)")
add("")
add("G(rho 1) > 0 everywhere, so the bar is 3 G(rho 1). The 3s fallback is never used. Still, the s that C recorded "
    "equals arm A's s for the same seed and quality exactly, in all 25 pairs. Both gains were recomputed from C's "
    "saved 8,000-step actors and are equal bit for bit.")
add("")
add("| Quality | Seed | G(rho 1) | G(rho 0.5) | Bar 3 G(rho 1) | G(rho 0.5) / G(rho 1) | Fires |")
add("|---|---|---|---|---|---|---|")
for q in Q5:
    for k in range(5):
        p = KC["per_quality"][q]["per_seed"][str(k)]
        add(f"| {q} | {k} | {f(p['G_rho1'])} | {f(p['G_rho05'])} | {f(p['bar'])} | {f(p['ratio'])} | {yn(p['kb1_fires'])} |")
add("")
add("**Verdict:** KB1 fires nowhere. The highest ratio is 1.489 (random seed 1), and the bar is 3. There is no K3 "
    "(every actor is finite and no record has non-finite values). This matches KILLS.md.")
add("")
add("### KB0, the guard is idle")
add("")
add("The rule needs rho 0.5's share of thruster-1 commands past float32(0.8) to be at least 2% and at least twice "
    "rho 1's, on at least 2 of random, medium-replay and medium. All 15 seed-level shares were recomputed from C's "
    "saved final actors and are equal bit for bit.")
add("")
add("| Quality | rho 0.5 per seed | rho 1 per seed | Seed 0 passes | 5-seed mean rho 0.5 | 5-seed mean rho 1 | Twice rho 1 | Mean passes |")
add("|---|---|---|---|---|---|---|---|")
for q in ("random", "medium-replay", "medium"):
    m = KC["kb0_means"][q]
    add(f"| {q} | {', '.join(pct(x) for x in m['shares_rho05'])} | {', '.join(pct(x) for x in m['shares_rho1'])} | "
        f"{yn(KC['kb0_seed0'][q])} | {pct(m['mean_rho05'], 3)} | {pct(m['mean_rho1'], 3)} | {pct(m['twice_rho1'], 3)} | {yn(m['passes'])} |")
add("")
add("**Verdict:** KB0 holds at the screen (medium-replay and medium) and on the 5-seed means (medium-replay and "
    "medium). Random fails both times (rho 0.5's mean is 1.77%, under 2%). KB0 therefore does not stop B2-B4, which "
    "matches KILLS.md (`holds: True`).")
add("")
add("### The guard-void rule")
add("")
add("- **By B's K1 or K2:** B's K1 fired at the screen (above), so **the guard is void on that ground alone.** B's K2 "
    "did not fire.")
add("- **By C's own map** (a thruster-1 ceiling of 0.85 or higher, with at least 30 tries in rung (0.80, 0.85]): all "
    "25 guard maps were rebuilt and equal the records. The rule triggers on **18 of 25** seed-quality pairs:")
add("")
add("| Quality | Seed | Guard t1 | Tries in (0.80, 0.85] | Triggers |")
add("|---|---|---|---|---|")
for q in Q5:
    for k in range(5):
        p = KC["per_quality"][q]["per_seed"][str(k)]
        add(f"| {q} | {k} | {p['t1']:.2f} | {p['tries_080_085']} | {yn(p['own_void'])} |")
add("")
add("  The rule triggers on random seeds 0, 1, 3 and 4 (seed 3 sits exactly at 30 tries, and 'at least 30' counts it), "
    "on medium-replay seeds 0, 1, 3 and 4, on all five seeds of medium, and on all five seeds of medium-expert. It does "
    "not trigger on random seed 2 (t1 0.80), medium-replay seed 2 (t1 0.80), or expert seeds 0-4 (t1 0.80, with no "
    "tries in the rung). This is the same list as KILLS.json's `by_own_map`.")
add("- **Consequence:** B2, B3 and B4 are not scored (results.json `met` is None for each, with `B2_B4_not_scored` "
    "set). B1 and B5 are still scored, as frozen.")
add("")
add("**Verdict:** this matches KILLS.md: guard void, with B2-B4 not scored.")
add("")

# ------------------------------------------------------------------ F0
add("## F0 (section 7.3)")
add("")
add("KF3 compares F0's clone SHA-256 with `clone.actor_sha256` in A2's own record (`channel/train_f1/<Q>/A2_cmd_strong/"
    "seed<k>/record.json`). For seed 0, that record is the result F rerun's, which is in the baseline and unchanged. "
    "The A2 record each F0 read is still unchanged (its SHA-256 matches).")
add("")
add("| Quality | Seed | KF1 disagreement (rows) | KF1 at le_none (seed 0) | KF2 (a) max diff | KF2 (b) max diff | KF3 clone F0 = A2 | KF4 finite |")
add("|---|---|---|---|---|---|---|---|")
for q in ("medium-expert", "expert"):
    for k in range(5):
        p = KF[q]["per_seed"][str(k)]
        nc = p["KF1_no_caution"]
        add(f"| {q} | {k} | {p['KF1_disagreement']} ({p['KF1_rows']}) | {'%s (%s)' % (nc['disagreement'], nc['rows']) if nc else '-'} | "
            f"{p['KF2a']['max_param_diff']} | {p['KF2b']['max_param_diff']} | {yn(p['KF3_equal'])} (`{p['KF3_f0_clone'][:12]}`) | {yn(p['KF4_finite'])} |")
add("")
add("KF2 ran 300 steps. In (b), `grid`, `thresholds`, `cap`, `u_logged` and `cal_params` were replaced. "
    "**Verdict:** every check passes on all ten pairs, and both qualities are decided, as in KILLS.md.")
add("")

# ------------------------------------------------------------------ ordering
add("## Ordering (sections 3.3 and 8)")
add("")
add("All times are UTC, from `state/job_times.tsv`, the runner logs and the records' own timestamps.")
add("")
add("| Event | Time |")
add("|---|---|")
add(f"| Seed-0 screen (`screen` job) | {O['screen_window'][0]} - {O['screen_window'][1]} |")
add(f"| Last training or arm job finished | {O['last_training_job_done'][0]} ({O['last_training_job_done'][1]}) |")
add(f"| Last arm record written | {O['last_arm_record_finished'][0]} ({O['last_arm_record_finished'][1]}) |")
add(f"| Kill checks (`kills` job; KILLS.json made_at {O['KILLS_made_at']}) | {O['kills_window'][0]} - {O['kills_window'][1]} |")
add(f"| First final-return job | {O['first_eval_start'][0]} ({O['first_eval_start'][1]}) |")
add(f"| Last final-return job | {O['last_eval_done'][0]} ({O['last_eval_done'][1]}) |")
add(f"| Marks (`marks` job; results.json made_at {O['results_made_at']}) | {O['marks_window'][0]} - {O['marks_window'][1]} |")
add("")
add("- No training, arm, host, room, band or A2 job finished after the kill checks started, and no evaluation job started "
    "before they finished. There are 60 evaluation records (A 25, C 25, F 10, B 0). Each started after KILLS.json's "
    "made_at, and each carries the current KILLS.json SHA-256 and its arm record's current SHA-256.")
add("- **No final A, B or C return before its rules were checked.**")
  # noqa
add("  - A deep scan of every A, B and C training record finds no final return. The only return-like numbers are A's "
    "K2 8,000-step difference and B's K0 host and knee-clip points, and both are what those rules read.")
add("  - C's KB0 rollouts of the final actors keep commands only, as section 6.4 prescribes.")
add("  - The screen holds returns only for B's K0 and for F0, A2 and le_none on seed 0. F0's seed-0 return is "
    "allowed at the screen.")
add("  - F0's seeds 1-4 are scored inside their own jobs, after that seed's KF1-KF4. This is allowed, because no "
    "cross-seed rule guards F0.")
add("- No `eval_*`, `kills` or `marks` job appears in the earlier runner logs, and each evaluation log shows a single "
    "attempt. The smoke test's evaluations (16 October, on smoke-trained stand-ins, deleted) were not of these actors.")
add("")
add("**Verdict:** no ordering violation.")
add("")

# ------------------------------------------------------------------ restarts
add("## Restarts")
add("")
add("| Job | Interrupted run | First start | First attempt's last log line | Rerun's first progress line | Files now (mtime) |")
add("|---|---|---|---|---|---|")
for j, d in RS["interrupted_jobs"].items():
    files = "; ".join(f"{n} {m[11:]}" for n, m in d["files"].items())
    add(f"| {j} | {d['runner_log']} | {d['started']} | {d['first_attempt_last'] or '(none: stopped before its chunk 1 ended)'} | "
        f"{d['rerun_first_progress']} | {files} |")
add("")
add("- **First reclaim.** `nohup_continue.out` ends at 21:56:21 (16 Oct), but the job logs run on to 21:59:38, so the "
    "container went down at about 22:00. No job finished in those three minutes, which is why the runner wrote nothing "
    "more. The run resumed at 00:59:38 (`nohup_continue2.out`), skipping the 27 finished jobs.")
add("- **Second reclaim.** The last job-log line is at 03:39:39 (17 Oct). The run resumed at 03:44:49 "
    "(`nohup_continue3.out`), skipping 68 finished jobs. The gates were rerun at 03:45:13 and all passed, including "
    "the 45-of-45 byte-identical sandbox code copy.")
add("- **Redone from scratch.**")
add("  - Every rerun restarts at the first step (room checks: cloning step 5,000; A: chunk 1). Its numbers equal "
    "the first attempt's step by step (the same losses, and A1's shares 0.177, 0.333, 0.347 and 0.376), as a "
    "deterministic restart should.")
add("  - Every file in the six folders has an mtime inside its rerun window. The room stage writes its files only at "
    "the end.")
add("  - A_medium_s1's first attempt had written `actors_8000.msgpack` at about 03:33. The rerun rewrote it at "
    "03:52:56, and the record's `actors_8000_sha256` matches the current file. arm_a.py never reads an existing actor "
    "file.")
add("  - A_medium_s2 and A_medium_s3 left nothing behind. No A2 job was interrupted, and there is no `state/quarantine`.")
add("- **No finished record was overwritten.**")
add(f"  - `job_times.tsv` has {R['jobs']['rows']} rows for {R['jobs']['distinct']} distinct jobs, all with exit 0, "
    "so no job finished twice.")
add(f"  - Each of the {R['mtimes']['files_checked']} files this run created under `runs/` has its mtime inside the "
    "single window of the job that owns it.")
add(f"  - The inputs of SCREEN_seed0.json ({I['SCREEN_seed0.json_inputs']['n']}), KILLS.json "
    f"({I['KILLS.json_inputs']['n']}) and results.json ({I['results.json_inputs']['n']}) all still have the SHA-256 "
    "recorded in them.")
add("  - The 820 files in the EXPLORATORY label (in both `runs/` roots) match their listed SHA-256, and no file "
    "created under `runs/` is missing from the label.")
add(f"- **Sidecars and hashes.** {I['sidecars_checked']} `.sha256` sidecars checked, with 0 mismatches. They cover "
    "the arm, eval, A2/B2/C2 and package stage records, plus KILLS.json, results.json and SCREEN_seed0.json. All "
    f"{I['arm_records']} arm records carry the label and the frozen digest. Every actor file (A, C, F0, A2, B2, C2) "
    "matches the SHA-256 in its record.")
u = I["untouched_recorded"]
n = I["untouched_now"]
add("- **The final untouched check passed.**")
add(f"  - `state/untouched.json` (checked at {u['checked_at']}, logged at 05:56:21) says all_untouched = "
    f"{u['all_untouched']}. Sandbox: changed {u['sandbox']['changed']}, removed {u['sandbox']['removed']}, new outside "
    f"`runs/` {u['sandbox']['new_outside_runs']}, new under `runs/` {u['sandbox']['new_under_runs']} (820 files plus "
    f"the 2 label files). Repository: changed {u['repository']['changed']}, removed {u['repository']['removed']}, "
    f"new {u['repository']['new']}.")
add(f"  - My recheck now: all {n['sandbox_baseline_files']} sandbox and {n['repository_baseline_files']} repository "
    "baseline files are hash-identical to the baseline. No new sandbox file exists outside `runs/`, and no new file "
    "exists under the repository's read roots. "
    f"{n['repository_files_modified_after_start_marker_count']} files under `/home/user/bca` (`.git` excluded) were "
    f"modified after the run's start marker ({n['start_marker']}).")
add(f"  - The sandbox holds {len(n['sandbox_pycache_dirs'])} `__pycache__` folders. All predate the run (15-16 "
    "October), and none holds a file newer than the start marker.")
add("")
add("**Verdict:** I found no partial-file reuse, no overwritten record and no failed integrity check.")
add("")

# ------------------------------------------------------------------ minor
add("## Minor points (none changes a decision)")
add("")
add("1. **KILLS.md's B rule block is misleading.** KILLS.md (and KILLS.json `B.rules` and "
    "`C_rules.guard_void.by_B_rules`) shows B's K1 as `{\"where\": [], \"fires\": false}` and K0/K4 with `mean: null`. "
    "Those dicts cover only seeds 1-4, whose maps were never built. B's actual stop, K1 at the screen, appears only in "
    "the 'Stopped by: K1' line and in `B.stops`. A reader can be confused by 'K1 fires: false' sitting next to "
    "'Stopped by K1' and `by_B: true`.")
add("2. **B's stop line omits the seed.** It reads 'Stopped by K1 on medium-replay; medium; medium-expert', but the "
    "frozen form is '<quality, seed>'. All three are seed 0.")
add("3. **RESULTS.md, arm C (outside the kill checks, for the wording check).** The B2-B4 rows show numbers with "
    "'Met: no', but they are not scored (results.json: None). The sentence says '(KB0 failed or the guard is void)', "
    "yet KB0 held and only the guard-void rule applies. The C sentence ('Halving the pull, guarded by BCA's ceiling, "
    "met B1 ...') is not one of section 6.5's frozen sentences, and it names BCA's ceiling as the guard although the "
    "guard is void.")
add("4. **The first reclaim was later than the runner log suggests.** The runner log for the first reclaim ends three "
    "minutes before the jobs stopped, so the reclaim was at about 22:00, not 21:56. There is no 'runner was stopped' "
    "line and no untouched check at that point, as expected for a killed container. The final untouched check covers it.")
add("5. **Scope of re-execution.** K1 and KF1/KF2 were verified from recorded values, not re-executed (see 'How this "
    "was checked').")
add("")
add("Files: `verify/kills_check.py`, `verify/kills_check_raw.json`, `verify/spot_check.py` and `spot_check.json`, "
    "`verify/spot_k0_kb0.py` and `spot_k0_kb0_k0.json` / `spot_k0_kb0_kb0.json`, `verify/spot_maps.py` and "
    "`spot_maps.json`, and this file's generator `verify/make_kills_check_md.py`.")
add("")
(V / "KILLS_CHECK.md").write_text("\n".join(L))
print("written", V / "KILLS_CHECK.md", len("\n".join(L)))
