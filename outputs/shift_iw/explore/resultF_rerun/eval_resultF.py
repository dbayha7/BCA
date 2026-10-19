"""EXPLORATORY: outside the version-5 pre-registration. Evaluation of result F's seed-0 sandbox rerun (no training).

Result F ("Does the band, weighted for where it is read, raise return?", the arm "Fixed fallback, exact weights",
A2_cmd_strong) and its comparison rows, retrained by run_resultF.sh in the sandbox (iw_rerun/repo) for data seed 0,
medium-expert and expert, are read here from their finished records and scored. Nothing is trained, no band is fitted,
and nothing is written outside <resultF>/results (or <resultF>/state for the check mode).

  eval_resultF.py check    read only: imports, paths, the desktop's reference values, the start rule (start 1), the
                           four situations' anchors against sit4/situations_seed0.json, and which learners exist yet.
                           Writes state/eval_check.json.
  eval_resultF.py run      requires every core learner (A2, B2, C2, No extra caution, Perfect lower edge) on every
                           quality; writes results/scores_seed0.json, results/COMPARE.md, results/episodes_seed0.json
                           and results/situations_seed0.json ("complete": true)
  eval_resultF.py run --allow-missing
                           requires only result F's actor (A2) on every quality; scores A2, No extra caution when it
                           exists, and every other learner that exists, and lists the missing rows and why
                           ("complete": false if a core learner is missing). run_resultF.sh uses it automatically when
                           A2's record and actor are good but B2, C2 or a lower-edge job failed
  A learner other than A2 whose record is unusable (does not parse, wrong label, not finite) is listed as missing with
  the reason; A2's refuses.

Learners (seed 0; each loaded through its own script's or the package's record, its SHA-256 checked):
  f1_A2_cmd_strong   Fixed fallback, exact weights (result F's row; the live panel's candidate)
  f1_B2_cmd_train    the same with the training mix of currents (w(c) = 1)
  f1_C2_untargeted   untargeted control of the fixed fallback
  le_none            No extra caution (room-lower-edge's k = 0 arm: the same actor-only training, no penalty)
  le_oracle          Perfect lower edge (the exact oracle penalty; the scale)
  le_bca_unweighted  version 5's lower edge, unweighted (explore_arms.py --lower-edge unweighted; optional)
  est_A3_estimated   Estimated weights (optional), est_C3_untargeted its untargeted control
Every one is an actor trained actor-only against the frozen gapped TD3+BC critic of the sandbox's seed-0 host, which
is the live panel's TD3+BC learner ("host_gap" in iw_rerun/results/episodes_seed0.json).

Scores, as result F scored them: learners_iw.evaluate_actor (the exact float64 100-step return, truth.rollout, from all
1,000 fixed starts of each start bank: 'none' = the training mix, 'mild', 'strong'), points = 100 (J - J_random) /
(J_stall_aware - J_random) with the banks' anchors, the gapped normalization. Each must equal the number the script's
own evaluate mode wrote (same function, same inputs).

Episodes, in the shapes the page already reads:
  episodes_seed0.json    start 1 of the training mix ('none') and the strong mix, as iw_rerun/driver/jobs.py episodes:
                         states, commands, rewards, J, Jd, points (the mix's anchors), push, q_actor = the frozen
                         critic's min(Q1, Q2) at the actor's command (the critic the actor was trained against), and
                         qpi_actor = the exact Q^pi of the actor's own policy (horizon 600, discount 0.98)
  situations_seed0.json  the main test's four situations (thruster_chain.bank_starts, 1,000 starts each, the stalling
                         world), as sit4/situations.py: mean J, SE, Jd, points on each situation's bank (data.anchors:
                         0 = the random quality's mean return, 100 = the stall-aware controller's; sit4's stored
                         anchors, which the recomputed ones equal to float rounding), and the start-11 episode (the
                         live panel's fixed-controller start)

Comparison with the desktop's seed-0 numbers (read only from the repository): channel/results/train_f1 and train_est
evaluate_seed0.json, explore/RESULTS.json per_seed[0] (le_none, le_oracle, le_bca_unweighted), SCORE.json's per-seed
paired differences, and the desktop's own spread (the SD of J over the learning-curve checkpoints at 16,000-20,000
steps, explore/curves/results/<quality>/seed0/*.json). The sandbox's host is not the desktop's bit for bit (iw_rerun
checks/REPRO.md), so actors cannot match bit for bit; the meaningful check is the sandbox's own paired gain,
A2 minus No extra caution on the strong starts.

Run from <iw_rerun>/repo with PYTHONPATH=<iw_rerun>/repo:<iw_rerun>/explore and python -B (run_resultF.sh does).
"""
import sys

sys.dont_write_bytecode = True          # the channel scripts are imported from the repository folder: never cache

import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import experiments.shift  # noqa: E402,F401  (gymnasium path and version checks)
from experiments.shift import learners as LR  # noqa: E402
from experiments.shift import thruster_chain as TC  # noqa: E402
from experiments.shift import truth as TR  # noqa: E402
from experiments.shift_iw import data as D  # noqa: E402
from experiments.shift_iw import learners_iw as LI  # noqa: E402
from experiments.shift_iw import run as R  # noqa: E402
from experiments.shift_iw import world as W  # noqa: E402

import explore_arms as XA  # noqa: E402
import explore_eval as XE  # noqa: E402

RF = Path(__file__).resolve().parent
SRC = Path(os.environ.get("RF_SRC", "/home/user/bca"))
IW = Path(os.environ.get("RF_IW", "/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/iw_rerun"))
SIT4 = Path(os.environ.get("RF_SIT4", str(RF.parent / "sit4" / "situations_seed0.json")))
QUALITIES = tuple(os.environ.get("RF_QUALITIES", "medium-expert expert").split())
EXPL = SRC / "outputs" / "shift_iw" / "explore"
CHAN = EXPL / "channel"
DESK = CHAN / "results"
OUT = RF / "results"
SEED = 0
LAWS = ("none", "mild", "strong")
MIX_EPISODES = (("training", "none"), ("strong", "strong"))
START_MIX, START_SIT, STEPS, H = 1, 11, 100, 600
SITS = ("recorded", "big_kicks", "big_waves", "strong_current")
LABEL = ("EXPLORATORY: outside the version-5 pre-registration; result F's arms retrained for data seed 0 in the cloud "
         "sandbox (iw_rerun) with result F's own code unchanged; evaluation only")
NAMES = {
    "f1_A2_cmd_strong": "Fixed fallback, exact weights (result F)",
    "f1_B2_cmd_train": "Fixed fallback, training mix of currents",
    "f1_C2_untargeted": "Untargeted control of the fixed fallback",
    "le_none": "No extra caution",
    "le_oracle": "Perfect lower edge",
    "le_bca_unweighted": "Version 5's lower edge, unweighted",
    "est_A3_estimated": "Estimated weights",
    "est_C3_untargeted": "Untargeted control of the estimated weights",
}
PANEL = "f1_A2_cmd_strong"
# every learner of the run's core jobs (run_resultF.sh's DONE condition); `run` requires them all, `run
# --allow-missing` only the panel's
CORE = ("f1_A2_cmd_strong", "f1_B2_cmd_train", "f1_C2_untargeted", "le_none", "le_oracle")
# the desktop's curve file and arm of each learner (for the desktop's own spread)
CURVE = {"f1_A2_cmd_strong": ("f1", "A2_cmd_strong"), "f1_B2_cmd_train": ("f1", "B2_cmd_train"),
         "f1_C2_untargeted": ("f1", "C2_untargeted"), "le_none": ("oracle", "le_none"),
         "le_oracle": ("oracle", "le_oracle"), "le_bca_unweighted": ("band", "v5_unweighted"),
         "est_A3_estimated": ("est", "A3_estimated"), "est_C3_untargeted": ("est", "C3_untargeted")}
# paired differences: (name, a, b, the desktop SCORE file and key)
PAIRS = (("A2 - No extra caution", "f1_A2_cmd_strong", "le_none", "train_f1", "A2_cmd_strong - no_caution"),
         ("A2 - version 5's unweighted lower edge", "f1_A2_cmd_strong", "le_bca_unweighted", "train_f1",
          "A2_cmd_strong - v5_unweighted"),
         ("A2 - C2 (untargeted control)", "f1_A2_cmd_strong", "f1_C2_untargeted", "train_f1",
          "A2_cmd_strong - C2_untargeted"),
         ("A2 - B2 (training mix)", "f1_A2_cmd_strong", "f1_B2_cmd_train", "train_f1", "A2_cmd_strong - B2_cmd_train"),
         ("Perfect lower edge - No extra caution", "le_oracle", "le_none", "train_f1", "oracle - no_caution"),
         ("A3 - No extra caution", "est_A3_estimated", "le_none", "train_est", "A3_estimated - no_caution"))


def log(m):
    print(time.strftime("%H:%M:%S ") + str(m), flush=True)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_module(name, file):
    spec = importlib.util.spec_from_file_location(name, CHAN / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


F1 = load_module("cle_f1", "command_le_train_f1.py")
EST = load_module("cle_est", "command_le_train_est.py")
INPUTS = {}            # every file read: path -> SHA-256 (rf.py done evalF re-checks them)


def note(path):
    INPUTS[str(Path(path))] = sha(path)
    return Path(path)


def plain(x):
    return R.plain(x)


# ------------------------------------------------------------------------------------------------ the learners

def script_record(d, label, passmarks):
    """A result F script's finished arm (record.json, actor.msgpack), checked: label, pass marks, finite, SHA-256."""
    p = d / "record.json"
    if not p.is_file():
        return None, None
    try:
        rec = json.loads(note(p).read_text())
    except ValueError as error:
        raise R.Refused(f"refused: {p} does not parse ({error})") from error
    if rec.get("label") != label or rec.get("exploratory") is not True:
        raise R.Refused(f"refused: {p} is not labelled as its script labels its records")
    if rec.get("passmarks_sha256") != passmarks or rec.get("seed") != SEED or rec.get("finite") is not True:
        raise R.Refused(f"refused: {p}: pass marks, seed or finiteness differ")
    try:
        params = LR.load_tree(note(d / "actor.msgpack"), LR.actor_template(), rec["actor_sha256"])
    except (OSError, KeyError, ValueError) as error:
        raise R.Refused(f"refused: {d / 'actor.msgpack'}: {type(error).__name__}: {error}") from error
    # failed / snapshot_100_file_sha256: the keys the page builder reads in learner_facts (iw_rerun's
    # episodes_seed0.json); these actor-only learners have no snapshot_100, so the file is the actor's own file
    facts = dict(source=str(d), failed=False, snapshot_100_file=str(d / "actor.msgpack"),
                 snapshot_100_file_sha256=rec["actor_sha256"], actor_sha256=rec["actor_sha256"],
                 actor_tree_sha256=LR.tree_sha256(params),
                 rows_sha256=rec.get("rows_sha256"), clone_actor_sha256=(rec.get("clone") or {}).get("actor_sha256"),
                 this_code=rec.get("this_code"), seconds=rec.get("seconds"), finished_at=rec.get("finished_at"),
                 chunks=[{k: c.get(k) for k in ("chunk", "no_credit_share", "capped_share", "no_credit_share_past_80",
                                                 "past_80_share", "W_mean", "kish_share", "grid_points")}
                         for c in rec.get("chunks", [])])
    return params, facts


def learners_of(q, ctx, spec):
    """{name: (actor params, facts, the script's own evaluation or None)} and [(name, why missing)]. A learner other
    than A2 whose record is unusable is listed as missing with the reason; A2's refuses."""
    have, missing = {}, []
    passmarks_f1 = (CHAN / "PASSMARKS_fallback_rerun.md.sha256").read_text().split()[0]
    passmarks_est = (CHAN / "PASSMARKS_estimated_weights.md.sha256").read_text().split()[0]
    for mod, prefix, pm in ((F1, "f1", passmarks_f1), (EST, "est", passmarks_est)):
        ev_path = mod.OUT / q / f"evaluate_seed{SEED}.json"
        try:
            ev = json.loads(note(ev_path).read_text())["evaluations"] if ev_path.is_file() else {}
        except (ValueError, KeyError) as error:
            log(f"{q}: {ev_path} is unusable ({type(error).__name__}: {error}); its scores are not compared")
            ev = {}
        for arm in mod.ARMS:
            name = f"{prefix}_{arm}"
            try:
                params, facts = script_record(mod.arm_dir(q, arm, SEED), mod.LABEL, pm)
            except R.Refused as error:
                if name == PANEL:
                    raise
                missing.append((name, f"record unusable: {error}"))
                continue
            if params is None:
                missing.append((name, "not trained"))
                continue
            have[name] = (params, facts, ev.get(arm))
    rle_path = R.room_le_dir(ctx, SEED) / "record.json"
    rle = R.finished(rle_path, spec)
    if rle is None:
        missing += [("le_none", "room-lower-edge has not finished"), ("le_oracle", "room-lower-edge has not finished")]
    elif rle["room_lower_edge"].get("stopped"):
        why = f"room-lower-edge stopped the seed: {rle['room_lower_edge'].get('why')}"
        missing += [("le_none", why), ("le_oracle", why)]
    else:
        note(rle_path)
        actors = R.controllers_of(ctx, "room_lower_edge", SEED, spec)
        note(R.room_le_dir(ctx, SEED) / "actors.msgpack")
        ev_rec = R.finished(R.evaluate_path(ctx, SEED, "room_lower_edge"), spec)
        if ev_rec is not None:
            note(R.evaluate_path(ctx, SEED, "room_lower_edge"))
        for name in ("le_none", "le_oracle"):
            p = actors[name]
            have[name] = (p, dict(source=str(R.room_le_dir(ctx, SEED)), failed=False,
                                  snapshot_100_file=str(R.room_le_dir(ctx, SEED) / "actors.msgpack"),
                                  snapshot_100_file_sha256=rle["actors_sha256"],
                                  actor_tree_sha256=LR.tree_sha256(p),
                                  clone_actor_sha256=(rle["room_lower_edge"].get("clone") or {}).get("actor_sha256"),
                                  clone_reproduces_room_check=rle["room_lower_edge"].get("clone_reproduces_room_check"),
                                  check_e_passed=(rle["room_lower_edge"].get("check_e") or {}).get("passed"),
                                  check_b=rle["room_lower_edge"].get("check_b"), seconds=rle.get("seconds")),
                          (ev_rec or {}).get("evaluations", {}).get(name))
    d = XA.lower_edge_dir(XA.EXPLORE_ROOT, q, "unweighted", SEED)
    try:
        rec = XE.explore_record(d / "record.json", spec)
    except R.Refused as error:
        rec = None
        missing.append(("le_bca_unweighted", f"record unusable: {error}"))
    if rec is None:
        if not any(n == "le_bca_unweighted" for n, _ in missing):
            missing.append(("le_bca_unweighted", "not trained"))
    else:
        note(d / "record.json")
        p = LR.load_tree(note(d / "actor.msgpack"), LR.actor_template(), rec["actor_sha256"])
        ev_rec = R.finished(XE.eval_path(XA.EXPLORE_ROOT, q, SEED, "bca_lower_edge"), spec)
        have["le_bca_unweighted"] = (p, dict(source=str(d), failed=False, snapshot_100_file=str(d / "actor.msgpack"),
                                             snapshot_100_file_sha256=rec["actor_sha256"],
                                             actor_sha256=rec["actor_sha256"], actor_tree_sha256=LR.tree_sha256(p),
                                             clone_actor_sha256=((rec.get("result") or {}).get("clone") or {}).get(
                                                 "actor_sha256"),
                                             pre_registered_report=rec.get("pre_registered_report"),
                                             seconds=rec.get("seconds")),
                                     (ev_rec or {}).get("evaluations", {}).get("le_bca_unweighted"))
    return have, missing


def same_clone(have):
    """Whether result F's actor (A2), and the estimated weights' A3 when present, started from the very cloned actor
    "No extra caution" started from (their records' clone actor SHA-256). The paired difference A2 - No extra caution
    means what result F's means only if they did. The other learners' clones are shown for information."""
    clone = {n: f.get("clone_actor_sha256") for n, (_, f, _) in have.items()}
    ref = clone.get("le_none")
    needed = ["f1_A2_cmd_strong"] + (["est_A3_estimated"] if "est_A3_estimated" in have else [])
    if ref is None or any(clone.get(n) is None for n in needed):
        identical, why = False, "a clone SHA-256 is missing (A2 or No extra caution missing, or not recorded)"
    else:
        diff = [n for n in needed if clone[n] != ref]
        identical, why = not diff, ("identical" if not diff else f"{', '.join(diff)} differ from No extra caution's")
    return dict(identical=bool(identical), why=why, compared=needed + ["le_none"], clone_actor_sha256=clone)


def report_note(q, have):
    """Which report the optional v5 lower-edge record's pre_registered_report copied: the sandbox's partial report
    (data seeds 0-4 only: D1 pass, every other mark pending or not run), not the committed version-5 report."""
    if "le_bca_unweighted" not in have:
        return None
    pre = have["le_bca_unweighted"][1].get("pre_registered_report") or {}
    sand = R.Context(quality=q, log=lambda m: None).quality_root / "report" / "report.json"
    comm = SRC / "outputs" / "shift_iw" / "v5" / q / "report.json"
    got = pre.get("report_sha256")
    return dict(report_sha256=got, matches_sandbox_report=bool(sand.is_file() and got == sha(sand)),
                matches_committed_report=bool(comm.is_file() and got == sha(comm)), marks=pre.get("marks"),
                note=("the v5 lower-edge record's pre_registered_report is copied from the sandbox's own PARTIAL "
                      "report (run.py report on data seeds 0-4 only: D1 pass, every other mark pending or not run), "
                      "not from the committed version-5 report; it is not a verdict on this row"))


# ------------------------------------------------------------------------------------------------ the desktop's numbers

def desktop_reference(q):
    """The desktop's seed-0 numbers (J and points per bank), its own spread, and its per-seed paired differences."""
    ref = {}
    for fam, prefix in (("train_f1", "f1"), ("train_est", "est")):
        ev = json.loads(note(DESK / fam / q / f"evaluate_seed{SEED}.json").read_text())["evaluations"]
        for arm, v in ev.items():
            ref[f"{prefix}_{arm}"] = {law: dict(J=v[law]["J"], points=v[law]["points"]) for law in LAWS}
    res = json.loads(note(EXPL / "RESULTS.json").read_text())
    assert res["seeds"] == [0, 1, 2, 3, 4]
    arms = next(x for x in res["qualities"] if x["quality"] == q)["arms"]
    for a in ("le_none", "le_oracle", "le_bca_unweighted"):
        ref[a] = {law: dict(J=arms[a][law]["J"]["per_seed"][SEED], points=arms[a][law]["points"]["per_seed"][SEED])
                  for law in LAWS}
    spread = {}
    for name, (group, arm) in CURVE.items():
        p = EXPL / "curves" / "results" / q / f"seed{SEED}" / f"{group}.json"
        c = json.loads(note(p).read_text())["curves"][arm]
        assert c["steps"][-5:] == [16000, 17000, 18000, 19000, 20000]
        spread[name] = {law: float(np.std(c["J"][law][-5:], ddof=1)) for law in LAWS}
    paired = {}
    for fam in ("train_f1", "train_est"):
        s = json.loads(note(DESK / fam / "SCORE.json").read_text())[q]
        for label, _, _, f, key in PAIRS:
            if f == fam and f"{key} | strong" in s:
                paired[label] = {law: dict(per_seed=s[f"{key} | {law}"]["J"]["per_seed"],
                                           mean=s[f"{key} | {law}"]["J"]["mean"],
                                           lower=s[f"{key} | {law}"]["J"]["lower"]) for law in LAWS}
    d1 = json.loads(note(DESK / "train_f1" / q / "A2_cmd_strong" / f"seed{SEED}" / "record.json").read_text())
    checks = json.loads(note(DESK / "train_f1" / q / f"checks_seed{SEED}.json").read_text())
    bits = dict(rows_sha256=d1["rows_sha256"], clone_actor_sha256=d1["clone"]["actor_sha256"],
                no_credit_share=[c["no_credit_share"] for c in d1["chunks"]], seconds=d1["seconds"],
                f1_checks=dict(check1=checks["check1_max_param_diff_vs_v5_unweighted_300_steps"],
                               check2=checks["check2_refusal_types"]))
    return dict(seed0=ref, spread_sd=spread, paired=paired, bits=bits)


# ------------------------------------------------------------------------------------------------ rollouts

def choose_start(banks):
    """iw_rerun/driver/jobs.py choose_start (compute_q.py's rule): the first start whose current lies in the
    interquartile range of its own mix, in both mixes."""
    for j in range(len(banks["none"])):
        ok = True
        for _, law in MIX_EPISODES:
            c = banks[law][j, TC.CURRENT]
            lo, hi = W.inverse_cdf(np.array([0.25, 0.75]), law)
            ok &= bool(lo <= c <= hi)
        if ok:
            return j
    raise RuntimeError("no typical start")


def mix_episodes(pol, critic, prep, banks, anchors, q, name):
    out = []
    for sit, law in MIX_EPISODES:
        run = TR.rollout(pol, banks[law][START_MIX][None], W.PHYSICS, length=STEPS, record=True)
        states, cmds, rewards = run.states[0][:STEPS], run.commands[0], run.rewards[0]
        obs = (np.asarray(states, np.float32) - prep.obs_mean) / prep.obs_std
        q_act = LR.critic_rows(critic, obs, np.asarray(cmds, np.float32))[0].min(axis=1)
        qpi_act = TR.q_value_jax(pol, states, cmds, W.PHYSICS, horizon=H)
        out.append(dict(quality=q, learner=name, situation=sit, bank=law, start_index=START_MIX,
                        J=float(run.J[0]), Jd=float(run.Jd[0]), points=float(W.points(run.J[0], anchors[law]["anchor"])),
                        states=states, cmds=cmds, rewards=rewards, push=states[:, TC.CURRENT] + states[:, TC.WAVE1],
                        q_actor=q_act, qpi_actor=qpi_act))
    return out


def situations(pol, banks4, anchors4, q, name):
    avg, eps = {}, {}
    for s in SITS:
        x0 = banks4[s]
        run = TR.rollout(pol, x0, W.PHYSICS)
        Jm = float(run.J.mean())
        se = float(run.J.std(ddof=1) / math.sqrt(len(x0)))
        a = anchors4[s]["anchor"]
        avg[f"{q}|{name}|{s}"] = dict(J=Jm, se=se, Jd=float(run.Jd.mean()), points=float(W.points(Jm, a)),
                                      points_se=100.0 * se / (a["top"] - a["random"]))
        rec = TR.rollout(pol, x0[START_SIT][None], W.PHYSICS, length=STEPS, record=True)
        assert abs(float(rec.J[0]) - float(run.J[START_SIT])) <= 1e-9 * max(1.0, abs(float(run.J[START_SIT])))
        eps[f"{q}|{name}|{s}"] = dict(quality=q, learner=name, situation=s, start_index=START_SIT, J=float(rec.J[0]),
                                      Jd=float(rec.Jd[0]), points=float(W.points(rec.J[0], a)),
                                      states=rec.states[0][:STEPS].tolist(), cmds=rec.commands[0].tolist(),
                                      rewards=rec.rewards[0].tolist(), final_state=rec.states[0][STEPS].tolist())
    return avg, eps


def situation_banks():
    """The four situations' banks and anchors. sit4/situations.py (the page's other learners in the four situations)
    ran with this machine's own arithmetic, this run with the desktop's (ARITH=desktop), so the anchors recomputed
    here can differ from sit4's in the last bits (measured: at most 1.2e-13). When they agree to 1e-9 relative, sit4's
    stored anchors are used for the points, so every learner on the page shares one scale; a larger difference
    refuses. Returns (banks, anchors, how)."""
    banks4 = {s: TC.bank_starts(s) for s in SITS}
    for s, b in banks4.items():
        assert b.shape == (TC.BANK_SIZE, TC.DX), (s, b.shape)
    ours = D.anchors(banks4)
    if not SIT4.is_file():
        return banks4, ours, dict(source="recomputed here (sit4/situations_seed0.json not found)")
    stored = json.loads(note(SIT4).read_text())["anchors"]
    gaps = {s: max(abs(ours[s]["anchor"][k] - stored[s]["anchor"][k]) / abs(stored[s]["anchor"][k])
                   for k in ("random", "top")) for s in SITS}
    worst = max(gaps.values())
    how = dict(source=f"sit4/situations_seed0.json's stored anchors ({SIT4})", identical=worst == 0.0,
               max_relative_gap_to_recomputed=worst, recomputed={s: ours[s]["anchor"] for s in SITS},
               ok=bool(worst <= 1e-9))
    if not how["ok"]:
        return banks4, ours, how
    return banks4, {s: dict(anchor=stored[s]["anchor"], controllers=stored[s]["controllers"]) for s in SITS}, how


# ------------------------------------------------------------------------------------------------ comparison

def verdict(ours, stored, tol):
    gap = ours - stored
    if gap == 0.0:
        return "identical"
    if abs(gap) <= 1e-9 * abs(stored):
        return "identical to float rounding"
    if abs(gap) <= tol:
        return "within the desktop run's own spread"
    return "OUTSIDE the desktop run's own spread"


def compare(q, scores, ref):
    rows = {}
    for name, ev in scores.items():
        if name not in ref["seed0"]:
            continue
        rows[name] = {}
        for law in LAWS:
            st, tol = ref["seed0"][name][law]["J"], 3.0 * ref["spread_sd"][name][law]
            rows[name][law] = dict(desktop_J=st, sandbox_J=ev[law]["J"], gap=ev[law]["J"] - st, tolerance=tol,
                                   desktop_points=ref["seed0"][name][law]["points"], sandbox_points=ev[law]["points"],
                                   verdict=verdict(ev[law]["J"], st, tol))
    pairs = {}
    for label, a, b, _, _ in PAIRS:
        if a not in scores or b not in scores:
            pairs[label] = dict(missing=[x for x in (a, b) if x not in scores])
            continue
        pairs[label] = {}
        for law in LAWS:
            ours = scores[a][law]["J"] - scores[b][law]["J"]
            d = (ref["paired"].get(label) or {}).get(law)
            row = dict(sandbox=ours, sandbox_points=scores[a][law]["points"] - scores[b][law]["points"])
            if d:
                lo, hi = min(d["per_seed"]), max(d["per_seed"])
                row.update(desktop_seed0=d["per_seed"][SEED], desktop_per_seed=d["per_seed"], desktop_mean=d["mean"],
                           desktop_lower=d["lower"], desktop_range=[lo, hi], inside_desktop_range=bool(lo <= ours <= hi),
                           same_sign_as_desktop_mean=bool(np.sign(ours) == np.sign(d["mean"])))
            pairs[label][law] = row
    return rows, pairs


def md_report(body):
    L = ["# Result F, seed 0, rerun in the cloud sandbox: comparison with the desktop", "",
         f"**{LABEL}.**", "", f"**Overall: {body['overall']}**", ""]
    if not body["complete"]:
        L += [f"**Partial scoring** (`eval_resultF.py run --allow-missing`): {body['mode_why']}. The missing rows are "
              "listed at the end.", ""]
    for r in body["reasons"]:
        L.append(f"- {r}")
    L += ["", "Return J: the exact 100-step return averaged over the 1,000 fixed starts of a start bank (closer to 0 is "
          "better); points: 0 = the random quality, 100 = the stall-aware controller. Tolerance: 3 x the SD of the "
          "desktop's own J over its learning-curve checkpoints at 16,000-20,000 steps (the same run, seed 0).", ""]
    for q in body["qualities"]:
        c = body["comparison"][q]
        L += [f"## {q}", "", "| Learner | Starts | Desktop J (points) | Sandbox J (points) | Gap | Tolerance | Verdict |",
              "|---|---|---|---|---|---|---|"]
        for name, laws in c["returns"].items():
            for law in ("strong", "none"):
                r = laws[law]
                L.append(f"| {NAMES[name]} | {law} | {r['desktop_J']:.2f} ({r['desktop_points']:.2f}) | "
                         f"{r['sandbox_J']:.2f} ({r['sandbox_points']:.2f}) | {r['gap']:+.2f} | {r['tolerance']:.1f} | "
                         f"{r['verdict']} |")
        L += ["", "| Paired difference | Starts | Sandbox | Desktop seed 0 | Desktop seeds 0-4 | Inside the desktop's "
              "range |", "|---|---|---|---|---|---|"]
        for label, laws in c["paired"].items():
            if "missing" in laws:
                L.append(f"| {label} | - | missing: {', '.join(laws['missing'])} | | | |")
                continue
            for law in ("strong", "none"):
                r = laws[law]
                if "desktop_seed0" in r:
                    L.append(f"| {label} | {law} | {r['sandbox']:+.2f} ({r['sandbox_points']:+.2f} points) | "
                             f"{r['desktop_seed0']:+.2f} | " + ", ".join(f"{x:+.1f}" for x in r["desktop_per_seed"])
                             + f" | {r['inside_desktop_range']} |")
                else:
                    L.append(f"| {label} | {law} | {r['sandbox']:+.2f} | - | - | - |")
        b = c["bits"]
        sc = c["same_clone"]
        L += ["", f"- Same cloned actor (A2{' and A3' if 'est_A3_estimated' in sc['compared'] else ''} against No "
              f"extra caution, the clone SHA-256 in their records): **{sc['identical']}** ({sc['why']}). "
              + "; ".join(f"{NAMES.get(n, n)} {(v or 'none')[:12]}" for n, v in sc["clone_actor_sha256"].items())
              + ".",
              f"- Bit checks against the desktop: actor-only row stream identical: {b.get('rows_identical')}; data D1 "
              f"of seeds 0-4 identical: {b.get('d1_identical')}; cloned actor identical: {b.get('clone_identical')} "
              "(information only: the desktop trained the channel arms unpinned with OMP_NUM_THREADS=2, this run "
              "pins every job to one CPU with one thread, so the clone's last bits may differ).",
              f"- Result F's checks before training (sandbox): {b.get('f1_checks')}.",
              f"- 'No value credit' share per chunk, A2: sandbox {b.get('no_credit_share_sandbox')}, desktop "
              f"{b.get('no_credit_share_desktop')}.",
              f"- Scores equal the scripts' own evaluate records: {c['same_as_script_records']}."]
        if c.get("v5_lower_edge_report"):
            r = c["v5_lower_edge_report"]
            L.append(f"- Version 5's lower edge, unweighted: {r['note']} (report SHA-256 {str(r['report_sha256'])[:12]}; "
                     f"the sandbox's report: {r['matches_sandbox_report']}; the committed report: "
                     f"{r['matches_committed_report']}).")
        L.append("")
        L += ["| Learner | " + " | ".join(SITS) + " |", "|---|" + "---|" * len(SITS)]
        for name in c["situations_points"]:
            L.append(f"| {NAMES.get(name, name)} | " + " | ".join(f"{c['situations_points'][name][s]:.1f}" for s in SITS)
                     + " |")
        L.append("")
    if any(body["missing"].values()):
        L += ["## Missing learners", ""] + [f"- {q} {n}: {why}" for q, ms in body["missing"].items()
                                            for n, why in ms.items()]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------------ modes

def write_text(path, text):
    """Whole or not at all: a temporary file beside it, then os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    tmp.write_text(text, encoding="utf8")
    os.replace(tmp, path)


def write_json(path, body, **kw):
    write_text(path, json.dumps(plain(body), allow_nan=False, **kw))


def check():
    t0 = time.time()
    out = dict(kind="resultF-eval-check", label=LABEL, qualities=list(QUALITIES))
    ctx0 = R.Context(quality=QUALITIES[0], log=lambda m: None)
    spec = R.spec_digest(ctx0)
    banks, sb = D.load_start_banks(ctx0.banks_root)
    out["start_rule_picks"] = choose_start(banks)
    out["start_rule_ok"] = out["start_rule_picks"] == START_MIX
    banks4, anchors4, how = situation_banks()
    out["situation_anchors"] = how
    out["learners"], out["desktop_reference"] = {}, {}
    for q in QUALITIES:
        ctx = R.Context(quality=q, log=lambda m: None)
        have, missing = learners_of(q, ctx, spec)
        out["learners"][q] = dict(present=sorted(have), missing=dict(missing))
        ref = desktop_reference(q)
        out["desktop_reference"][q] = dict(
            A2_strong=ref["seed0"]["f1_A2_cmd_strong"]["strong"], le_none_strong=ref["seed0"]["le_none"]["strong"],
            A2_minus_no_caution_strong_per_seed=ref["paired"]["A2 - No extra caution"]["strong"]["per_seed"],
            spread_sd_strong={k: v["strong"] for k, v in ref["spread_sd"].items()}, bits=ref["bits"])
    out["paths"] = dict(f1_out=str(F1.OUT), est_out=str(EST.OUT), explore_root=str(XA.EXPLORE_ROOT),
                        v5_root=str(R.ROOT_DEFAULT), results=str(OUT), sit4=str(SIT4))
    out["seconds"] = round(time.time() - t0, 1)
    out["ok"] = bool(out["start_rule_ok"] and how.get("ok", True))
    write_text(RF / "state" / "eval_check.json", json.dumps(plain(out), indent=1))
    print(json.dumps(plain(out), indent=1)[:6000])
    return 0 if out["ok"] else 3


def run(allow_missing=False):
    t0 = time.time()
    ctx0 = R.Context(quality=QUALITIES[0], log=lambda m: None)
    spec = R.spec_digest(ctx0)
    banks, sb = D.load_start_banks(ctx0.banks_root)
    note(ctx0.banks_root / "start_banks.json")
    if choose_start(banks) != START_MIX:
        raise R.Refused(f"refused: the start rule no longer picks start {START_MIX}")
    banks4, anchors4, how = situation_banks()
    if not how.get("ok", True):
        raise R.Refused(f"refused: the four situations' anchors differ from sit4/situations_seed0.json beyond float "
                        f"rounding: {how}")
    log(f"banks and anchors ready ({time.time() - t0:.0f}s)")
    scores, facts, mix_eps, sit_avg, sit_eps, missing, comparison = {}, {}, [], {}, {}, {}, {}
    plan = {}
    for q in QUALITIES:       # every learner's record checked before any scoring, so a refusal comes at once
        ctx = R.Context(quality=q, log=lambda m: None)
        have, miss = learners_of(q, ctx, spec)
        need = (PANEL,) if allow_missing else CORE
        absent = [n for n in need if n not in have]
        if absent:
            raise R.Refused(f"refused: {q} lacks {absent} ({dict(miss)}); "
                            + ("result F's actor is needed even with --allow-missing" if allow_missing else
                               "run_resultF.sh has not finished; `run --allow-missing` scores A2 and what exists"))
        plan[q] = (ctx, have, miss)
    core_missing = {q: [n for n in CORE if n not in plan[q][1]] for q in QUALITIES}
    complete = not any(core_missing.values())
    mode_why = ("every core learner present" if complete else
                "core rows missing: " + "; ".join(f"{q} {', '.join(v)}" for q, v in core_missing.items() if v))
    for q in QUALITIES:
        ctx, have, miss = plan[q]
        missing[q] = miss
        data = R.load_data(ctx, SEED, spec)
        note(R.data_record_path(ctx, SEED))
        prep = D.prepare(data, "gap", ctx.cfg)
        host = R.load_host_state(ctx, "gap", SEED, spec)
        note(R.host_dir(ctx, "gap", SEED) / "snapshot_100.msgpack")
        critic = host.critic.params
        scores[q], same_as_script = {}, {}
        for name, (params, f, script_ev) in have.items():
            f = dict(f, critic="the frozen gapped TD3+BC host critic of seed 0 (the live panel's host_gap)",
                     critic_tree_sha256=LR.tree_sha256(critic), page_label=NAMES[name])
            facts[f"{q}/{name}"] = f
            ev = LI.evaluate_actor(params, prep, banks, sb["anchors"])
            scores[q][name] = ev
            same_as_script[name] = (None if script_ev is None else
                                    all(ev[law]["J"] == script_ev[law]["J"] for law in LAWS))
            pol = TR.MLPPolicy(params, prep.obs_mean, prep.obs_std, name=f"{q} {name} actor (float64 copy)")
            eps = mix_episodes(pol, critic, prep, banks, sb["anchors"], q, name)
            for e in eps:
                full = TR.rollout(pol, banks[e["bank"]], W.PHYSICS).J
                assert float(full.mean()) == ev[e["bank"]]["J"]
                assert abs(e["J"] - float(full[START_MIX])) <= 1e-9 * max(1.0, abs(float(full[START_MIX])))
            mix_eps += eps
            avg, se = situations(pol, banks4, anchors4, q, name)
            sit_avg.update(avg)
            sit_eps.update(se)
            log(f"{q} {name}: strong J {ev['strong']['J']:.2f} ({ev['strong']['points']:.2f} points), none J "
                f"{ev['none']['J']:.2f}; situations " + " ".join(f"{s} {avg[f'{q}|{name}|{s}']['points']:.1f}"
                                                                 for s in SITS) + f" ({time.time() - t0:.0f}s)")
        ref = desktop_reference(q)
        rows, pairs = compare(q, scores[q], ref)
        a2 = facts.get(f"{q}/f1_A2_cmd_strong", {})
        d1p = RF / "state" / f"d1_{q}.json"
        f1c = F1.OUT / q / f"checks_seed{SEED}.json"
        f1c_body = json.loads(note(f1c).read_text()) if f1c.is_file() else None
        bits = dict(rows_identical=(a2.get("rows_sha256") == ref["bits"]["rows_sha256"]) if a2 else None,
                    clone_identical=(a2.get("clone_actor_sha256") == ref["bits"]["clone_actor_sha256"]) if a2 else None,
                    d1_identical=json.loads(note(d1p).read_text())["all_identical"] if d1p.is_file() else None,
                    f1_checks=(None if f1c_body is None else dict(
                        check1_max_param_diff=f1c_body["check1_max_param_diff_vs_v5_unweighted_300_steps"],
                        disagree_no_credit=f1c_body["check2_refusal_types"]["disagree_no_credit"],
                        disagree_capped=f1c_body["check2_refusal_types"]["disagree_capped"],
                        passed=bool(f1c_body["check1_max_param_diff_vs_v5_unweighted_300_steps"] == 0.0
                                    and f1c_body["check2_refusal_types"]["disagree_no_credit"] == 0.0
                                    and f1c_body["check2_refusal_types"]["disagree_capped"] == 0.0))),
                    no_credit_share_sandbox=[round(c["no_credit_share"], 4) for c in a2.get("chunks", [])],
                    no_credit_share_desktop=[round(x, 4) for x in ref["bits"]["no_credit_share"]],
                    desktop_f1_checks=ref["bits"]["f1_checks"], seconds_sandbox=a2.get("seconds"),
                    seconds_desktop=ref["bits"]["seconds"])
        comparison[q] = dict(returns=rows, paired=pairs, bits=bits, same_as_script_records=same_as_script,
                             same_clone=same_clone(have), v5_lower_edge_report=report_note(q, have),
                             situations_points={n: {s: sit_avg[f"{q}|{n}|{s}"]["points"] for s in SITS}
                                                for n in scores[q]})
    # the overall reading (descriptive; exploratory, no pass mark)
    reasons, gain_ok, spread_ok, clone_ok = [], True, True, True
    for q in QUALITIES:
        sc = comparison[q]["same_clone"]
        if not sc["identical"]:
            clone_ok = False
            reasons.append(f"{q}: A2 and No extra caution did NOT start from the same cloned actor ({sc['why']})")
        p = comparison[q]["paired"]["A2 - No extra caution"]
        if "missing" in p:
            gain_ok = False
            reasons.append(f"{q}: A2 or No extra caution is missing")
            continue
        s = p["strong"]
        gain_ok &= s["sandbox"] > 0
        reasons.append(f"{q}: A2 - No extra caution on the strong starts {s['sandbox']:+.2f} return units "
                       f"({s['sandbox_points']:+.2f} points); desktop seed 0 {s.get('desktop_seed0', float('nan')):+.2f}"
                       f", seeds 0-4 {s.get('desktop_range')}; inside the desktop's range: "
                       f"{s.get('inside_desktop_range')}")
        for name in ("f1_A2_cmd_strong", "le_none"):
            for law in ("strong", "none"):
                v = comparison[q]["returns"][name][law]["verdict"]
                spread_ok &= not v.startswith("OUTSIDE")
        fc = comparison[q]["bits"]["f1_checks"]
        if fc is not None and not fc["passed"]:
            reasons.append(f"{q}: result F's checks before training did NOT pass in the sandbox: {fc}")
            spread_ok = False
        if any(v is False for v in comparison[q]["same_as_script_records"].values()):
            reasons.append(f"{q}: a score differs from its script's own evaluate record")
            spread_ok = False
    if not gain_ok:
        overall = "RESULT F'S GAIN NOT REPRODUCED in the sandbox on seed 0 (see the paired table)"
    elif spread_ok and clone_ok:
        overall = ("CONSISTENT WITH RESULT F: A2 beats No extra caution on the strong starts on both qualities, both "
                   "started from the same cloned actor, and A2's and No extra caution's returns lie within the desktop "
                   "run's own spread")
    elif not clone_ok:
        overall = ("GAIN SEEN, but A2 and No extra caution did not start from the same cloned actor, so the pair is "
                   "not result F's comparison (see the same-clone line)")
    else:
        overall = ("GAIN REPRODUCED, but some value lies outside the desktop run's own spread or a check differs "
                   "(see the tables)")
    if not complete:
        overall += f" [partial scoring: {mode_why}]"
    learners = sorted({n for q in QUALITIES for n in scores[q]}, key=list(NAMES).index)
    common = dict(seed=SEED, qualities=list(QUALITIES), learners=learners, learner_labels=NAMES, panel_learner=PANEL,
                  learner_facts=facts)
    scores_body = dict(kind="resultF-rerun-scores-seed0", label=LABEL, overall=overall, reasons=reasons,
                       complete=complete, allow_missing=bool(allow_missing), mode_why=mode_why,
                       core_learners=list(CORE), core_missing=core_missing,
                       how="learners_iw.evaluate_actor on the shared start banks (1,000 fixed starts each), the gapped "
                           "normalization, as result F's scripts' evaluate modes", scores=scores,
                       comparison=comparison, missing={q: dict(m) for q, m in missing.items()},
                       finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), seconds=round(time.time() - t0, 1), **common)
    eps_body = dict(kind="resultF-rerun-episodes-seed0", label=LABEL, start_index=START_MIX, steps=STEPS,
                    q_pi_horizon=H, gamma=W.GAMMA,
                    about=("Start-1 episodes of result F's rerun learners: the actor's float64 copy (truth.MLPPolicy) "
                           "drives 100 steps (truth.rollout), as iw_rerun/results/episodes_seed0.json. q_actor: the "
                           "frozen gapped host critic min(Q1, Q2) at the actor's command (the critic every one of "
                           "these actors was trained against; float32 network, gapped normalization); qpi_actor: the "
                           "exact Q^pi of its own policy (horizon 600, discount 0.98)."),
                    state_fields=["position of float 1", "speed of float 1", "position of float 2", "speed of float 2",
                                  "position of float 3", "speed of float 3", "position of float 4", "speed of float 4",
                                  "position of float 5", "speed of float 5", "current c", "wave w1", "wave w2"],
                    episodes=mix_eps, finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **common)
    sit_body = dict(kind="resultF-rerun-situations-seed0", label=LABEL, start_index=START_SIT, steps=STEPS,
                    situations=list(SITS), world="stall",
                    starts="thruster_chain.bank_starts(situation), 1,000 per situation (the main test's fixed starts)",
                    points_rule=("weighting-test scale per situation bank: 100 (J - J_random) / (J_stall_aware - "
                                 "J_random) (world.points, data.anchors)"),
                    anchors={s: dict(anchor=a["anchor"], controllers=a["controllers"]) for s, a in anchors4.items()},
                    anchors_source=how,
                    bank_start11={s: banks4[s][START_SIT].tolist() for s in SITS}, avg=sit_avg, episodes=sit_eps,
                    finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), seconds=round(time.time() - t0, 1), **common)
    write_json(OUT / "episodes_seed0.json", eps_body)
    write_json(OUT / "situations_seed0.json", sit_body)
    write_text(OUT / "COMPARE.md", md_report(scores_body))
    scores_body["inputs_sha256"] = dict(INPUTS)
    write_json(OUT / "scores_seed0.json", scores_body)      # last: rf.py done evalF reads it
    log(f"overall: {overall}")
    log(f"written {OUT} ({time.time() - t0:.0f}s)")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=LABEL)
    p.add_argument("mode", choices=("check", "run"))
    p.add_argument("--allow-missing", action="store_true",
                   help="run: require only result F's actor (A2) on every quality; score it, No extra caution when it "
                        "exists and every other learner that exists, and list the missing rows")
    a = p.parse_args(argv)
    try:
        return check() if a.mode == "check" else run(a.allow_missing)
    except (R.Refused, LI.ApprovalError) as error:
        print(str(error), file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
