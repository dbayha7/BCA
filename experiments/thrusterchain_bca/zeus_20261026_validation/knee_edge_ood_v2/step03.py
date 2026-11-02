"""Step 0.3: frozen-checkpoint checks (approval A2; user-run in the recorded WSL GPU environment).

Order of work in one attempt:
1. the user's A2_APPROVAL.json must name this package's manifest hash and the attempt number; the attempt rules of
   5.0 are enforced (attempt k needs attempts 1..k-1 to exist, the previous one ended INCOMPLETE, at most one
   code-only fix); every sealed A1 and A2 file, the reused rerun1 sources and the saved inputs are re-hashed;
2. the attempt directory step03_run/attempt_<k> is created (never reused);
3. preflight, before any value exists: the recorded worker environment (set here, before JAX is imported; a
   conflicting value refuses), rerun1's load_native and runtime_receipt, and the recorded device kind (4.8, failures);
4. inputs are authenticated by rerun1's own code (validate_protocol, assemble_exports) and by pinned hashes;
5. per cell, through rerun1's recorded build path (runtime.create, warmup-10000 restore with digest checks,
   runtime.branch, restore_snapshot, FrozenPolicy): the 4.8 known-answer reproduction, Q1 at the G0.5 points and Q1
   at the reach points; each cell's file is written as soon as the cell is done;
6. the gated label (known answer, G0.5, reach) is sealed in gated.json before the onset contrast;
7. the warning-only onset contrast on the full-data critics (rebuilt by the same path), under its own sub-caps
   ("not computed" if they bind).
Units for a rerun (5.0): "gated" and "onset". After an INCOMPLETE in the onset only, attempt k + 1 reruns the onset
alone and reuses the sealed gated results. The gated items share a cap of 0.5M transitions or 30 minutes.

Run in WSL (instructions and the approval command: A2_README.md):
  /home/david/bca-operator-audit-20261026/bin/python -B -u <this file> --execute
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np

import frozen_checks as K
import probes as P
import qpi
import region_independent as second
import step03_math as SM
import worlds
from core import PAYLOAD, RERUN1, VALIDATION, require, sha_file
from knee_region import KNEE, SLOTS, past_knee_a0

RUN = HERE / "step03_run"
MANIFEST = HERE / "A2_MANIFEST.json"
APPROVAL = HERE / "A2_APPROVAL.json"
A1_MANIFEST = HERE / "MANIFEST.json"
A1_MANIFEST_SHA256 = "b6c23e9432c91c43cda4995ceb6909f2f83022aadb9594b1db9b964a357124a6"     # as committed in c498cf33
A2_REQUIRED_FILES = {"step03.py", "step03_math.py", "test_step03.py", "A2_README.md"}
CONTEXTS = ("ordinary", "strong_current")
GATED_TRANSITIONS, GATED_SECONDS = 500_000, 30 * 60
ONSET_TRANSITIONS, ONSET_SECONDS = 200_000, 10 * 60
APPROVAL_WORDS = "I approve A2 (step 0.3, frozen-checkpoint checks) for knee_edge_ood_v1"
WORKER_ENVIRONMENT = {                     # matched_policy_pilot_v1/launch.py child_environment(), as recorded for rerun1
    "JAX_PLATFORMS": "cuda", "JAX_DEFAULT_MATMUL_PRECISION": "highest", "JAX_ENABLE_X64": "false",
    "XLA_FLAGS": "--xla_gpu_triton_gemm_any=True", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
    "BCA_TRANSFER_BUNDLE_ROOT": str(PAYLOAD / "transfer_bundle")}
RERUN1_SOURCES = {
    "runtime.py": "e17a33d00847e3f989fe1ed01c64644a194f007f295805b05a1b95198bac1c2c",
    "evaluate_worker.py": "9f8763f999741f54f47b1280b7567623deadb3c4b5e22ea1d61466633319ae73",
    "evaluation.py": "fab7228431ac49a89ac4d1518f77ed54b9cd980f55c9f409554f43dd81c30bf5",
    "gradient_audit.py": "79c9cbd8274bd172fb9eb85e39573fc2c72129e57313639986667645c0aac3f8",
    "rerun_inputs.py": "a07b3ffbc97ef95e6b836994745b4d771483b4dcffb8cdbc958815fe7e4433a3",
    "recovery_inputs.py": "965589bf822cf67bfb8f2e62549477ae1596d5255fd650c08d8bc30c7b6dec1e",
    "common.py": "efd2427dd272f017530071ac1d334b48349db2d5505a0504c0e1923258f3eef1",
    "state_exact.py": "25d59193d55423c149f8ccb0c040bfb54663b31960cb290c193afa690388eb31",
    "lock_tolerant.py": "1f801ffc3b8fb98c102f670ab13a6c6d6207ef32518df74b3d36e47a02dcc150",
    "data.py": "83fd98907134ef6ed02bda97239420dba438d298b5c1651c31c7918573e7de00",
    "protocol.json": "95a9dbe042bbe82e868cceb02810e7f1ffa71c0949925996ea17ea3f54d4cdfd",
    "DATA_PLAN.json": "eec7fe230699babcb5f22536e931d97d41e0d6105c0d25fc27d4c3a32d57122a",
    "run/evaluation/report.json": "55cae667d369abb1912101161e0468f7f88ce889295cdd21ef51531c27762be7"}
AUDIT_SHA256 = {
    "seed0_full": "d0fa699a3749a4285747c5cdd4d32cb550ebf85da50c37a8081a3ca0ea8cac6d",
    "seed0_hole": "2d173ba46f7236346263147724900ec48c66eda168ba55b76fe2f0dd686dff12",
    "seed0_random": "c39ec2680536f03207018e547a78089ee481c2c36a4f6496f678d61f17d1f15a",
    "seed1_full": "f3c3952bc36e8d90cacdd62616a9aaa7008c8218d8859ec11a2b8a83a2b83486",
    "seed1_hole": "372282b0b90d72a999eee6fe40b7c17be7a0d2123d3c1c487ed51ecf29a00143",
    "seed1_random": "69dc53e6214e752633029dd0c2b15b4ce02efcad4a7be30a27e14bc292cea51d",
    "seed2_full": "df3b19e8e5eca9e316da9ee9875bbe41d212969a14a5c5ef7da970007c452173",
    "seed2_hole": "3461821b441e73e174498b33d2b1b27efc82448e7063f8211d3eb57d63b0b40f",
    "seed2_random": "05d4df034a741ab3c788670a76902f9e7121dcc9672d808dfc39d97ebc1bb5ea"}
POOLED_PAST_KNEE = {"full": [117, 1242], "hole": [118, 1140], "random": [126, 1137]}      # protocol 4.8


class CapExhausted(RuntimeError):
    pass


class Budget:
    """Prospective transition and wall-clock caps; a call that would cross either cap raises before it runs."""

    def __init__(self, transitions, seconds):
        self.limit, self.seconds, self.used, self.start = transitions, seconds, 0, time.monotonic()

    def charge(self, times, slots):
        self.check_time()
        need = int(np.sum(100 - np.asarray(times, np.int64))) * int(slots)
        if self.used + need > self.limit:
            raise CapExhausted(f"transition cap {self.limit} would be exceeded ({self.used} + {need})")
        self.used += need

    def check_time(self):
        if time.monotonic() - self.start > self.seconds:
            raise CapExhausted(f"time cap {self.seconds} s exceeded")


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    raise TypeError(f"Unserializable {type(value).__name__}")


def _write_new(path, value):
    raw = json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=_plain) + "\n"   # serialize first
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(raw)


def _progress(directory, state):
    try:
        (directory / "progress.json").write_text(json.dumps({**state, "updated_utc": datetime.now(timezone.utc).isoformat()},
                                                            default=_plain), encoding="utf-8")
    except OSError:
        pass                                    # progress is evidence only; a locked write is skipped


# ------------------------------------------------------------------------------------------------ approval, attempts

def _attempts():
    if not RUN.is_dir():
        return []
    return sorted((p for p in RUN.iterdir() if p.is_dir() and p.name.startswith("attempt_")), key=lambda p: int(p.name[8:]))


def check_approval():
    """The user's approval must name the sealed A2 package and the attempt; the 5.0 attempt rules must allow it."""
    require(APPROVAL.is_file(), "A2_APPROVAL.json is missing: the user's own approval is required (see A2_README.md)")
    approval = json.loads(APPROVAL.read_text(encoding="utf-8"))
    require(APPROVAL_WORDS in approval.get("approval", ""), "Approval wording missing")
    manifest_sha = sha_file(MANIFEST)
    require(approval.get("package_manifest_sha256") == manifest_sha, "Approval names a different A2 package")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    require(A2_REQUIRED_FILES <= set(manifest["files"]), "A2 manifest lacks a required file")
    for name, digest in manifest["files"].items():
        require(sha_file(HERE / name) == digest, f"Sealed A2 file changed: {name}")
    require(sha_file(A1_MANIFEST) == A1_MANIFEST_SHA256, "A1 manifest differs from the committed one")
    for name, digest in json.loads(A1_MANIFEST.read_text(encoding="utf-8"))["files"].items():
        require(sha_file(HERE / name) == digest, f"Sealed A1 file changed: {name}")
    for name, digest in RERUN1_SOURCES.items():
        require(sha_file(RERUN1 / name) == digest, f"Reused rerun1 file changed: {name}")
    attempt = approval.get("attempt")
    previous = _attempts()
    require(type(attempt) is int and attempt == len(previous) + 1, f"Approval must name attempt {len(previous) + 1}")
    history, sealed_gated = [], None
    for p in previous:
        report_path = p / "report.json"
        require(report_path.is_file(), f"{p.name} has no report; resolve it before a new attempt")
        prior = json.loads(report_path.read_text(encoding="utf-8"))
        history.append({"attempt": p.name, "label": prior.get("label"), "failed_unit": prior.get("failed_unit"),
                        "counts_under_5_0": prior.get("counts_under_5_0"),
                        "a2_manifest_sha256": prior.get("approval", {}).get("a2_manifest_sha256")})
        gated_path = p / "gated.json"
        if gated_path.is_file() and json.loads(gated_path.read_text(encoding="utf-8"))["gated_label"] != "INCOMPLETE":
            sealed_gated = p.name                       # a unit is sealed only when it completed with a valid outcome
    if history:
        require(history[-1]["label"] == "INCOMPLETE", f"No re-attempt after {history[-1]['label']} (5.0)")
        manifests = {h["a2_manifest_sha256"] for h in history}
        require(len(manifests | {manifest_sha}) - 1 <= 1, "At most one code-only fix (5.0)")
    return {"approval": approval, "attempt": attempt, "units": ["onset"] if sealed_gated else ["gated", "onset"],
            "sealed_gated_attempt": sealed_gated, "a2_manifest_sha256": manifest_sha,
            "a1_manifest_sha256": A1_MANIFEST_SHA256, "history": history}


def set_worker_environment():
    """Set the recorded worker environment before JAX is imported; a conflicting preset value refuses."""
    require("jax" not in sys.modules, "JAX was imported before the worker environment was set")
    conflicts = {k: os.environ[k] for k, v in WORKER_ENVIRONMENT.items() if k in os.environ and os.environ[k] != v}
    require(not conflicts, f"Conflicting environment values: {conflicts}")
    os.environ.update(WORKER_ENVIRONMENT)


def checked_receipt(runtime, holder):
    """Preflight body: load the native code, take rerun1's receipt and assert the recorded environment."""
    def receipt_function(_):
        native = runtime.load_native()
        receipt = runtime.runtime_receipt(native)
        mismatched = {k: os.environ.get(k) for k, v in WORKER_ENVIRONMENT.items() if os.environ.get(k) != v}
        require(not mismatched, f"Worker environment differs: {mismatched}")
        holder["native"] = native
        return {**receipt, "worker_environment": dict(WORKER_ENVIRONMENT)}
    return receipt_function


def rerun1():
    """rerun1's modules, imported read-only by their own names (bytecode writing is off)."""
    if str(RERUN1) not in sys.path:
        sys.path.insert(0, str(RERUN1))
    import runtime
    import evaluate_worker
    import gradient_audit
    import rerun_inputs
    return runtime, evaluate_worker, gradient_audit, rerun_inputs


# ------------------------------------------------------------------------------------------------ saved inputs

def load_banks():
    """The saved shared banks, checked against the evaluation report's receipts. Reads the bank archives only."""
    report = json.loads((RERUN1 / "run/evaluation/report.json").read_text(encoding="utf-8"))
    banks = {}
    for context in CONTEXTS:
        receipt = report["snapshot_results"][context + "_bank"]
        meta_path = RERUN1 / "run/evaluation" / receipt["path"]
        require(sha_file(meta_path) == receipt["sha256"], f"{context} bank metadata changed")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        arrays = meta["arrays"]
        npz = RERUN1 / "run/evaluation" / arrays["path"]
        require(sha_file(npz) == arrays["sha256"], f"{context} bank arrays changed")
        with np.load(npz, allow_pickle=False) as archive:
            bank = {k: archive[k] for k in archive.files}
        require(bank["states"].dtype == np.float64 and bank["states"].shape == (32, 13), f"{context} bank states")
        require(bank["observations"].dtype == np.float32 and bank["observations"].shape == (32, 14), f"{context} bank obs")
        banks[context] = {**bank, "bank_id": meta["bank_id"], "metadata_sha256": receipt["sha256"]}
    return banks


def saved_lambdas(cell):
    record = K.saved_lambdas(cell)
    require(record["sha256"] == AUDIT_SHA256[cell], f"{cell}: saved gradient audit changed")
    return record


def saved_audit(cell):
    """The cell's saved native 20k gradient audit (pinned by hash). Reads JSON only."""
    return json.loads((VALIDATION / saved_lambdas(cell)["file"]).read_text(encoding="utf-8"))


def saved_cell(cell):
    report = json.loads((RERUN1 / "run/evaluation/report.json").read_text(encoding="utf-8"))
    out = {}
    for context in CONTEXTS:
        path = RERUN1 / f"run/evaluation/{cell}__native__20000__{context}.json"
        require(sha_file(path) == report["snapshot_results"][f"{cell}__native__20000/{context}"]["sha256"],
                f"{cell}/{context}: saved result differs from the evaluation report")
        record = json.loads(path.read_text(encoding="utf-8"))
        out[context] = {**K.read_reference(path), "high_load": np.asarray(record["q"]["per_probe"]["high_load"], bool),
                        "bank_id": record["bank_id"],
                        "visited": record.get("visited", {}).get("per_row") if context == "strong_current" else None}
    return out


def past_knee_count(actions, high):
    actions, high = np.asarray(actions, np.float32), np.asarray(high, bool)
    require(np.array_equal(past_knee_a0(actions)[0], second.agree(actions)), "Region implementations disagree")
    return [int(past_knee_a0(actions)[0][high].sum()), int(high.sum())]


# ------------------------------------------------------------------------------------------------ per cell

def q1_at(policy, observations, probes_array):
    n, slots = probes_array.shape[:2]
    q = np.asarray(policy.q(np.repeat(observations, slots, axis=0), probes_array.reshape(-1, 4)))
    require(q.shape == (n * slots, 2) and np.isfinite(q).all(), "Expected finite twin Q")
    return q[:, 0].astype(np.float64).reshape(n, slots), q.min(axis=1).astype(np.float64).reshape(n, slots)


def known_answer(cell, policy, adapter, state, saved, bank, gradient_audit, budget, ledger):
    """4.8 on one cell, both contexts. Returns (ok, record)."""
    record, ok = {}, True
    saved_record = saved_audit(cell)
    audit = gradient_audit.audit(adapter, state, saved_record["source_row_ids"])          # rerun1's own audit, live
    ledger["reproduction_values"] += 1
    live_lambda = np.asarray([x["lambda_from_unpenalized_Q1"] for x in audit["batches"]], np.float64)
    saved_rows = saved_record["per_row"]
    live_actions = np.asarray(audit["per_row"]["actor_actions"], np.float32)
    live_high = np.asarray(audit["per_row"]["high_load"], bool)
    rows_ok = bool(np.array_equal(live_high, np.asarray(saved_rows["high_load"], bool)) and K.relative_ok(
        live_actions, np.asarray(saved_rows["actor_actions"], np.float32), K.Q_RTOL)[0])
    live_count = past_knee_count(live_actions, live_high)
    saved_count = past_knee_count(saved_rows["actor_actions"], saved_rows["high_load"])
    ok &= rows_ok and live_count == saved_count
    record["past_knee"] = {"live": live_count, "saved": saved_count, "rows_match": rows_ok}
    for context in CONTEXTS:
        s, b = saved[context], bank[context]
        require(s["bank_id"] == b["bank_id"], "Saved result and bank differ")
        slot0 = np.asarray(policy.act(b["observations"]))                     # the recorded single 32-row forward pass
        actions = s["actions"]
        repeated = np.repeat(b["observations"], 6, axis=0)
        q = np.asarray(policy.q(repeated, actions.reshape(192, 4)))           # the recorded single 192-row critic call
        width = np.asarray(policy.width(repeated, actions.reshape(192, 4)))
        require(q.shape == (192, 2) and np.isfinite(q).all() and np.isfinite(width).all(), "Nonfinite critic or width")
        q1, twin = q[:, 0].reshape(32, 6), q.min(axis=1).reshape(32, 6)
        adjusted = (np.asarray(q[:, 0], np.float64) - 3.0 * np.asarray(width, np.float64)).reshape(32, 6)
        budget.charge(b["times"], 6)
        values, calls, receipt = qpi.raw_values_from_states(b["states"], b["times"], actions, policy.act,
                                                            "stall15_identity", clip_all=False)
        ok_cell, checks = K.check_cell(s, slot0=slot0, q1=q1, twin_min=twin, adjusted=adjusted, qpi=values,
                                       lam=live_lambda, saved_lambda=saved_lambdas(cell)["values"])
        ok &= ok_cell
        record[context] = {"ok": ok_cell, "checks": checks, "qpi_receipt": receipt, "actor_calls": calls}
    b = bank["ordinary"]                                                       # T15 identity and stall = no-stall
    probes_id = np.concatenate([P.build(32, ("s10",)), P.build(32, ("s10",), kind="identity_image", world="stall15_identity")], axis=1)
    budget.charge(b["times"], 2)
    values, _, _ = qpi.raw_values_from_states(b["states"], b["times"], probes_id, policy.act, "stall15_identity", clip_all=False)
    residual, identity = worlds.identity_residual(values[:, :1], values[:, 1:], np.asarray([[SLOTS["s10"]]]), "stall15_identity")
    scaled = np.abs(residual) / np.maximum(1.0, np.abs(values[:, :1]))
    identity_ok = bool(np.all(scaled <= 1e-6))
    inside = P.build(32, ("edge_700",))
    budget.charge(b["times"], 2)
    v15, _, _ = qpi.raw_values_from_states(b["states"], b["times"], inside, policy.act, "stall15_identity", clip_all=True)
    vns, _, _ = qpi.raw_values_from_states(b["states"], b["times"], inside, policy.act, "nostall", clip_all=True)
    equal_ok = qpi.cross_check(v15, vns)[0]
    policy.assert_unchanged()
    ok &= identity_ok and equal_ok
    record["identity_T15"] = {"ok": identity_ok, "max_scaled_residual": float(scaled.max()), **identity}
    record["stall_equals_nostall_within_knee"] = {"ok": equal_ok, "slot": "edge_700", "continuation": "clipped"}
    record["state_unchanged"] = True
    return bool(ok), record


def g05_and_reach(cell, policy, saved, bank):
    out = {}
    for context in CONTEXTS:
        s, b = saved[context], bank[context]
        high = s["high_load"]
        q1, _ = q1_at(policy, b["observations"][high], P.build(int(high.sum()), P.G05_POINTS, kind="g05"))
        out[f"g05_{context}"] = {"rows": np.flatnonzero(high), "q1": q1, "F": SM.fold_error(q1)}
    s = saved["ordinary"]
    out["steep_ordinary_high"] = SM.steep(s["qpi"], s["actions"], s["high_load"])[s["high_load"]]
    lam = float(np.mean(saved_lambdas(cell)["values"]))
    sc, bsc = saved["strong_current"], bank["strong_current"]
    probe_rows = sc["high_load"]
    own = sc["actions"][probe_rows, 0, 1:]                      # the actor's own a1..a3 at each probe (saved slot 0)
    obs_p = bsc["observations"][probe_rows]
    q_probe, _ = q1_at(policy, obs_p, P.build(len(obs_p), P.REACH_POINTS, a123=own, kind="reach_forecast"))
    q_probe0, _ = q1_at(policy, obs_p, P.build(len(obs_p), P.REACH_POINTS, a123=np.zeros_like(own), kind="reach_forecast"))
    visited = sc["visited"]
    v_obs, v_act = np.asarray(visited["observations"], np.float32), np.asarray(visited["actions"], np.float32)
    v_rows = np.asarray(visited["high_load"], bool) & (v_obs[:, 10].astype(np.float64) >= 0.9)
    require(np.array_equal(past_knee_a0(v_act)[0], second.agree(v_act)), "Region implementations disagree on visited actions")
    n_visit = int(v_rows.sum())
    q_visit, _ = q1_at(policy, v_obs[v_rows], P.build(n_visit, P.REACH_POINTS, a123=v_act[v_rows, 1:], kind="reach_forecast"))
    q_visit0, _ = q1_at(policy, v_obs[v_rows], P.build(n_visit, P.REACH_POINTS, a123=np.zeros((n_visit, 3), np.float32),
                                                       kind="reach_forecast"))
    probe_fit, visit_fit = SM.reach_rows(q_probe, lam), SM.reach_rows(q_visit, lam)
    out["reach"] = {"lambda_mean_saved": lam, "probe_rows": int(len(obs_p)), "visited_rows": n_visit,
                    "probe_forecast": SM.forecast(probe_fit["pi0"]), "visited_forecast": SM.forecast(visit_fit["pi0"]),
                    "probe_fit": probe_fit, "visited_fit": visit_fit,
                    "descriptive_probe_forecast_a123_zero": SM.forecast(SM.reach_rows(q_probe0, lam)["pi0"]),
                    "descriptive_visited_forecast_a123_zero": SM.forecast(SM.reach_rows(q_visit0, lam)["pi0"])}
    policy.assert_unchanged()
    return out


def onset(policy, saved, bank, g05_ordinary, budget):
    """Warning-only onset contrast on one full-data critic's steep high-load ordinary probes."""
    s, b = saved["ordinary"], bank["ordinary"]
    rows = np.flatnonzero(SM.steep(s["qpi"], s["actions"], s["high_load"]))
    index = {int(r): i for i, r in enumerate(np.asarray(g05_ordinary["rows"]))}
    q1 = np.asarray(g05_ordinary["q1"], np.float64)[[index[int(r)] for r in rows]]
    knee_col, s05_col = SM._column(P.G05_POINTS, -KNEE), SM._column(P.G05_POINTS, SLOTS["s05"])
    states, times = b["states"][rows], b["times"][rows]
    budget.charge(times, 3)
    q15, _, _ = qpi.raw_values_from_states(states, times, P.build(len(rows), ("s05", "edge_knee")), policy.act,
                                           "stall15_identity", clip_all=True)
    qns, _, _ = qpi.raw_values_from_states(states, times, P.build(len(rows), ("s05",)), policy.act, "nostall", clip_all=True)
    policy.assert_unchanged()
    return SM.onset_ratio(q1[:, s05_col], q1[:, knee_col], q15[:, 0], q15[:, 1], qns[:, 0])


def build_policy(cell, protocol, runtime, evaluate_worker, exports):
    """rerun1's recorded build path for one native 20k cell."""
    seed, condition = int(cell[4]), cell[6:]
    adapter, template = runtime.create(protocol, seed, condition)
    require(adapter.context.data.receipt["protected_transition_rows_decoded"] == 0, "Protected data decoded")
    warm_export = exports[f"{cell}__warmup__10000"]
    warm = evaluate_worker.restore_snapshot(runtime, adapter, template, warm_export)
    matching = warm_export["matching"]
    require(runtime.tree_digest(warm.aux.posterior) == matching["reference_digest"]
            and runtime.tree_digest(warm.microstep.agent_state.actor.params) == matching["anchor_actor_digest"],
            "Frozen reference/anchor differs from matching receipt")
    export = exports[f"{cell}__native__20000"]
    require(export["matching"] == matching, "Branch rematched its distance coefficient")
    aa, tt = runtime.branch(adapter, warm, protocol, seed, "native", matching["distance_coefficient"])
    state = evaluate_worker.restore_snapshot(runtime, aa, tt, export)
    return evaluate_worker.FrozenPolicy(runtime, aa, state), aa, state


# ------------------------------------------------------------------------------------------------ the attempt

def run_gated(directory, protocol, runtime, evaluate_worker, gradient_audit, exports, banks, ledger, report, input_hashes):
    """Unit "gated": every cell, then the gated label sealed in gated.json. Each cell file is written when done."""
    budget = Budget(GATED_TRANSITIONS, GATED_SECONDS)
    ka_ok, cells = True, {}
    for cell in K.CELLS:
        budget.check_time()
        _progress(directory, {"phase": "cell", "cell": cell, "transitions": budget.used, "ledger": ledger})
        policy, adapter, state = build_policy(cell, protocol, runtime, evaluate_worker, exports)
        saved = saved_cell(cell)
        ok, ka = known_answer(cell, policy, adapter, state, saved, banks, gradient_audit, budget, ledger)
        ka_ok &= ok
        cells[cell] = {"known_answer": ka, **g05_and_reach(cell, policy, saved, banks)}
        _write_new(directory / f"cell_{cell}.json", cells[cell])
        del policy, adapter, state
    budget.check_time()
    pooled = {c: [sum(cells[f"seed{s}_{c}"]["known_answer"]["past_knee"]["live"][i] for s in range(3)) for i in (0, 1)]
              for c in ("full", "hole", "random")}
    pooled_ok = pooled == POOLED_PAST_KNEE
    g05 = {s: SM.g05_seed_summary(cells[f"seed{s}_full"]["g05_ordinary"]["F"][cells[f"seed{s}_full"]["steep_ordinary_high"]])
           for s in range(3)}
    g05_strong = {s: {"median_F": float(np.median(cells[f"seed{s}_full"]["g05_strong_current"]["F"])),
                      "share_F_le_0.5": float(np.mean(cells[f"seed{s}_full"]["g05_strong_current"]["F"] <= SM.F_BAR))}
                  for s in range(3)}
    probe_fc = {c: cells[c]["reach"]["probe_forecast"] for c in K.CELLS}
    visit_fc = {c: cells[c]["reach"]["visited_forecast"] for c in K.CELLS}
    known_ok = bool(ka_ok and pooled_ok)
    label, reasons, _ = SM.label(known_answer_ok=known_ok, caps_ok=True, g05_by_seed=g05,
                                 probe_forecast_by_cell=probe_fc, visited_forecast_by_cell=visit_fc, onset_by_seed={})
    gated = {"gated_label": label, "gated_reasons": reasons, "known_answer_ok": known_ok, "pooled_past_knee": pooled,
             "pooled_past_knee_ok": pooled_ok, "g05": g05, "g05_strong_current_reported": g05_strong,
             "probe_forecast": probe_fc, "visited_forecast": visit_fc, "gated_transitions": budget.used}
    for path, digest in input_hashes.items():                    # inputs re-hashed before the gated unit is sealed
        require(sha_file(path) == digest, f"Sealed input changed during the attempt: {path}")
    _write_new(directory / "gated.json", gated)
    report.update(gated)
    gated["g05_ordinary_by_seed"] = {s: cells[f"seed{s}_full"]["g05_ordinary"] for s in range(3)}
    return gated


def _finish(directory, report):
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    if report.get("label") == "INCOMPLETE":
        unit = report.get("failed_unit") or "preflight"
        report["failed_unit"] = unit
        report["counts_under_5_0"] = unit != "preflight"         # only a pre-computation failure is exempt
        earlier = [h for h in report["approval"]["history"] if h.get("failed_unit") == unit and h.get("counts_under_5_0")]
        if (report["counts_under_5_0"] and earlier
                and earlier[-1]["a2_manifest_sha256"] == report["approval"]["a2_manifest_sha256"]):
            report["label"] = "INCONCLUSIVE"            # 5.0: a second failure of the same unit under unchanged code
            report.setdefault("reasons", []).append(f"second failure of unit {unit!r} under unchanged code (5.0)")
    _write_new(directory / "report.json", report)
    _progress(directory, {"phase": "finished", "label": report.get("label")})
    return report


def execute():
    approval = check_approval()
    directory = RUN / f"attempt_{approval['attempt']}"
    directory.mkdir(parents=True, exist_ok=False)
    _write_new(directory / "ATTEMPT.json", {"started_utc": datetime.now(timezone.utc).isoformat(), **approval})
    ledger = {"reproduction_values": 0}
    report = {"step": "0.3", "approval": approval, "units": approval["units"], "ledger": ledger, "failed_unit": "preflight"}
    try:
        set_worker_environment()
        runtime, evaluate_worker, gradient_audit, rerun_inputs = rerun1()
        status, receipt = K.preflight(checked_receipt(runtime, {}), None, ledger)
        report["preflight"] = {"status": status, "receipt": receipt}
        if status != "PASS":
            report.update(label="INCOMPLETE", reasons=["preflight: " + str(receipt)])
            return _finish(directory, report)
        report["failed_unit"] = approval["units"][0]
        protocol = json.loads((RERUN1 / "protocol.json").read_text(encoding="utf-8"))
        evaluate_worker.validate_protocol(protocol)
        exports, input_hashes = evaluate_worker.assemble_exports(protocol, rerun_inputs.continuation_root() / "run",
                                                                 sha_file(RERUN1 / "protocol.json"))
        banks = load_banks()
        if "gated" in approval["units"]:
            gated = run_gated(directory, protocol, runtime, evaluate_worker, gradient_audit, exports, banks, ledger,
                              report, input_hashes)
        else:
            prior = RUN / approval["sealed_gated_attempt"]
            gated = json.loads((prior / "gated.json").read_text(encoding="utf-8"))
            gated["g05_ordinary_by_seed"] = {s: json.loads((prior / f"cell_seed{s}_full.json").read_text(encoding="utf-8"))["g05_ordinary"]
                                             for s in range(3)}
            report["reused_gated_from"] = prior.name
        report["gated_label"], report["gated_reasons"] = gated["gated_label"], gated["gated_reasons"]
        report["failed_unit"] = "onset"
        onset_by_seed, not_computed = {}, []
        if gated["gated_label"].startswith("PASS"):
            sub = Budget(ONSET_TRANSITIONS, ONSET_SECONDS)
            for s in range(3):
                cell = f"seed{s}_full"
                try:
                    policy, _, _ = build_policy(cell, protocol, runtime, evaluate_worker, exports)
                    onset_by_seed[s] = onset(policy, saved_cell(cell), banks, gated["g05_ordinary_by_seed"][s], sub)
                    sub.check_time()
                except CapExhausted as exc:
                    onset_by_seed[s] = None
                    not_computed.append(f"seed {s}: {exc}")
                if onset_by_seed[s] is not None and onset_by_seed[s]["ratio"] is None:
                    not_computed.append(f"seed {s}: median rho15(s05) <= 0, ratio undefined")
        for path, digest in input_hashes.items():
            require(sha_file(path) == digest, f"Sealed input changed during the attempt: {path}")
        report["failed_unit"] = None
        final = SM.label(known_answer_ok=gated["known_answer_ok"], caps_ok=True,
                         g05_by_seed={int(k): v for k, v in gated["g05"].items()},
                         probe_forecast_by_cell=gated["probe_forecast"], visited_forecast_by_cell=gated["visited_forecast"],
                         onset_by_seed=onset_by_seed)
        if final[0] == "INCOMPLETE":
            report["failed_unit"] = "gated"             # a known-answer mismatch: the gated unit failed, and counts
        report.update(label=final[0], reasons=final[1] + [f"onset not computed: {x}" for x in not_computed],
                      flagged=final[2], onset=onset_by_seed,
                      written_prediction_required=("Stage 1 V0(b) or V0(d) may end INCONCLUSIVE" if final[2] else None))
    except CapExhausted as exc:
        report.update(label="INCOMPLETE", reasons=[f"{report['failed_unit']} cap: {exc}"])
    except BaseException as exc:                # any failure after the attempt starts is INCOMPLETE, with a report
        report.update(label="INCOMPLETE", reasons=[f"{type(exc).__name__}: {exc}"])
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            report["interrupted"] = type(exc).__name__
            _finish(directory, report)
            raise
    return _finish(directory, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.parse_args()
    result = execute()
    print(json.dumps({"label": result.get("label"), "reasons": result.get("reasons")}, indent=2, default=_plain))
