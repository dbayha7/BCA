"""The runner: each Unifloral learner built as its training block builds it, scanned in chunks of 2,500 outer steps,
with BCA's refreshes between chunks, the IW threshold tables, snapshots, the stop rules' hashes and checks, and one run
record per run (plan sections 3.1, 3.3-3.4, 4.1-4.2, 5.4, 6.1 and 7.4; INTERFACES.md sections 5, 10 and 13).
Training environment (/opt/bca-venv).

An experiment-local attachment (plan section 3.1): it changes none of the repository's declared methods. The learner
alone and the room references run Unifloral's own imported make_train_step (unifloral_import.make_unifloral_step); the
BCA setups run the restated steps (unifloral_<learner>_bca.py); nothing under baselines/unifloral/ is edited.

What replaces each file's training block (td3_bc.py:283-337, cql.py:329-386, iql.py:272-330, rebrac.py:313-370; plan
section 3.1, "What replaces each file's training block"):
  set-up    PRNGKey(Args.seed), the dataset as the file's own Transition, the networks and the key split, all by
            unifloral_import.init_like_training_block (td3_bc.py:285, 299-327 and the other three blocks), then
            make_train_step.
  loop      num_evals = steps // eval_interval chunks. Each chunk is
                carry, loss = jax.lax.scan(step, carry, None, args.eval_interval)
            called exactly as the block calls it (td3_bc.py:331-336), then rng, rng_eval = jax.random.split(rng)
            (td3_bc.py:340, cql.py:388, iql.py:332, rebrac.py:372) with rng_eval discarded, so the training random
            stream is the unmodified script's (test g).
  differs   the data are the prepared ThrusterChain arrays (training rows; the full log for the room references), there
            is no gym environment and no Unifloral evaluation. Args: the file's defaults with seed =
            default_rng([20261016, rep, 31]).integers(2**31 - 1), gamma 0.98 and num_updates 100,000 (plan 3.2, 3.4;
            bca_core.seeds). Stage 1 runs the first 15,000 of those 100,000 outer steps (--steps), so its runs are
            prefixes of the stage-2 runs of seed 0 (IQL's cosine schedules still run over 100,000).

Setups (INTERFACES.md section 3): alone and room = Unifloral's imported step, carry (rng, agent_state); standard, iw,
iw_ones (--weights-one: every bank weight and test mass 1, stop rule S2), iw_repeat (S8) and sham_alone (choice 19:
BCA present, never refreshed, so every dose stays exactly 1) = the restated step with BCA present, carry
(rng, agent_state, bca_state), bca_state = bca_core.initial_state(make_scale_state(...), n_train,
most_caution=iw.most_caution(setup)). CQL's BCA setups run two compiled programs, as its stage-0 report asks:
make_train_step(..., dose_active=False) for the chunks before the first refresh step and the dose-active step after
(unifloral_cql_bca.py docstring; the sham switches at the same step without refreshing).

Between chunks, in this order, after the chunk ending at outer step t:
  hashes    S4: guard.state_sha256 of Unifloral's key carried out of the chunks ending at 2,500, 50,000 and 100,000
            (before the evaluation split). S10: Unifloral's learner state and key at 10,000, just before the first
            refresh. Initial hashes of every part of the learner state (and of BCA's start) are taken before chunk 1.
  checks    S5 (plan 6.1; experiments/shift/learners.py:1209-1215): any non-finite loss or metric, any false per-step
            validity flag of the restated step, a dose outside [1, 1.5] (ReBRAC: outside its [1, 2] check), non-finite
            IQL weights, or a non-finite parameter stops the run. The run is recorded as stopped, never retried.
  split     rng, rng_eval = jax.random.split(rng); rng_eval is discarded.
  refresh   after outer steps 10,000, 15,000, ..., 95,000 (bca_core.refresh_steps; 18; none at the end): the learner's
            refresh(...) with bca_core.refresh_key(s39, t). A refresh failing BCA's own validity checks, or a NaN
            threshold, is S5; a threshold of +inf is "cannot certify": counted as refused rows, not a stop (the reading
            of S5 the scaffold report asked the runner to confirm). For the IW setups the bank scores are then
            recomputed from the stored state with the refresh's own key (the learner's own target; TD3+BC through its
            recompute_bank_scores, the others through band_parts, which is the refresh's own computation), and
            iw.iw_refresh checks stop rule S3 on them (bit for bit), tabulates R(t) on the grid and gives every training
            row its threshold. At the last refresh of an IW setup: the estimated-weight table (choice 20), the
            shuffled-weights placebo (plan 4.6) and IW's dose change (plan 5.4).
  snapshot  at 10,000 (S10's readout), 50,000 (50%), the last refresh and the last step (100%): the whole carry as a
            hand-off file (handoff.write, SHA-256 sidecars) and the actor parameter file for scoring
            (actors_f32.write_actor_file). A snapshot holds the state the next chunk starts from (after the split and
            the refresh).

The IW grid (iw.mass_grid) is built once per run on every test mass that is looked up: the training rows' and the
strong-law test steps' exact weights and the exact weight's two ends w(0.1) and w(1.1), so the last refresh's table also
serves the coverage readouts at any visited state (band_readout.py). With --weights-one every weight and mass is 1 and
the grid is [1.0] (iw.py, S2).

Permission (plan section 6.1; INTERFACES.md section 8). A run on real data calls gate.require_run_approval() before it
reads anything (so today every real run is refused: no approval exists), and every BCA entry point re-checks it
(guard.require_bca). A run on made-up arrays (--made-up-file, written by `made-up-file` from tests/madeup.py) runs with
guard.MADE_UP only after the file's arrays are shown to be the registered made-up arrays (guard.require_made_up), so a
made-up label on any other arrays is refused. Made-up runs may shrink the schedule (tests and the stage driver's dry
run); real runs use the plan's schedule only. execute() itself checks the permission against the data again (a
made-up file: guard.MADE_UP and the registered arrays; real data: a gate.Approval that still verifies), so the alone and
room setups, whose imported Unifloral step has no entry check, cannot be trained on real arrays through the library
either. s7-commands needs the approval unless both its actor file and its states file are marked made-up. Every actor
file a run writes says trained true in its meta, and evaluate.py rolls a trained actor out only with the approval.

Stop rule S9 before the first chunk (real runs): after Unifloral's files are imported, the run's environment signature
(env_signature of this process's import record) must equal prespec.json's environment.s9_signature in every key either
names; otherwise the run stops with S9 before training. A stop raised before execute (S9 here, or S11 from
read_prepared) writes no record: the runner prints a status line {"status": "stopped", "stop": {...}} and exits 3, and
the stage driver reads the rule from that line.

Resumable. A run's folder holds started.json at start (with the process id and the boot id) and record.json (with
sidecar) at the end. A run whose record exists and verifies with the same identity (data, code manifest, settings,
schedule, device) is skipped; a record with another identity, or a started run without a record, is refused (never
overwritten or retried silently; --replace-incomplete is the operator's explicit choice for the second case, and it is
refused while the started run's process is still alive on this boot).

Usage (training environment, from /root/bca-hosts, with unifloral_import.launch_env()'s variables):
  python -m experiments.shift_hosts.runner run --learner td3_bc --setup standard --quality medium-expert --seed 0 \\
      --stage stage2 [--steps 15000] [--device gpu]
  python -m experiments.shift_hosts.runner run --learner cql --setup iw --weights-one --stage stage1 --steps 15000 ...
  python -m experiments.shift_hosts.runner run ... --made-up-file runs/shift_hosts/stage0/runner/madeup0.npz \\
      --num-updates 50 --eval-interval 10 --refresh-warmup 20 --refresh-every 10
  python -m experiments.shift_hosts.runner made-up-file --seed 0 --out <path.npz>
  python -m experiments.shift_hosts.runner s7-commands --states <states.npz> --actor <actor.npz> --out <out.npz>
Exit status: 0 complete (or skipped as already complete), 3 stopped by a stop rule (the record says which), 2 refused
(no approval, unregistered arrays, a record mismatch), 1 any other error.
"""

import argparse
import dataclasses
import hashlib
import json
import math
import os
import platform
import resource
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from experiments.shift_hosts import gate, guard
from experiments.shift_hosts import handoff as H

sys.dont_write_bytecode = True

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[1]
RUNS = REPO / "runs" / "shift_hosts"
DATA = RUNS / "data"
SCHEMA = "shift-hosts-run-v1"
PREPARED_SCHEMA = "shift-hosts-prepared-v1"
SNAPSHOT_SCHEMA = "shift-hosts-snapshot-v1"
LEARNERS = ("td3_bc", "cql", "iql", "rebrac")
QUALITIES = ("medium-expert", "medium")
UNIFLORAL_SETUPS = ("alone", "room")
BCA_SETUPS = ("standard", "iw", "iw_ones", "iw_repeat", "sham_alone")
SETUPS = UNIFLORAL_SETUPS + BCA_SETUPS
IW_SETUPS = ("iw", "iw_ones", "iw_repeat")
GAMMA = 0.98                                   # ThrusterChain's discount (plan 3.2, exception 1)
NUM_UPDATES = 100_000                          # plan 3.2, exception 2
EVAL_INTERVAL = 2_500                          # Args.eval_interval (td3_bc.py:29 and the other three): the chunk length
REFRESH_WARMUP, REFRESH_EVERY = 10_000, 5_000  # plan 3.3 and 4.1: 18 refreshes, none at the end
KEY_HASH_STEPS = (2_500, 50_000, 100_000)      # stop rule S4
S10_STEP = 10_000                              # stop rule S10: just before the first refresh
STAGE1_STEPS = 15_000                          # plan 7.3, stage 1
HIGH_CURRENT_SHARE = 0.1                       # the highest-current tenth (prepare.py; shift_iw/run.py:610-613)
FLAGS = dict(                                  # each restated step's per-step validity flags (stop rule S5)
    td3_bc=("inputs_valid", "scale_inputs_valid", "dose_inputs_valid"),
    cql=("inputs_valid", "scale_inputs_valid", "dose_inputs_valid", "dose_guard_ok"),
    iql=("inputs_valid", "scale_inputs_valid", "dose_inputs_valid"),
    rebrac=("inputs_valid", "scale_inputs_valid", "dose_inputs_valid", "batch_valid", "dose_in_range"),
)
DOSE_BOUNDS = dict(td3_bc=(1.0, 1.5), cql=(1.0, 1.5), rebrac=(1.0, 2.0))   # S5; ReBRAC: its own [1, 2] check
UNIFLORAL_LOSSES = dict(                       # each imported step's loss dict (td3_bc.py:270-276 and the others)
    td3_bc=("actor_loss", "q_loss", "q_mean", "lambda", "bc_loss"),
    cql=("critic_loss", "actor_loss", "alpha_loss", "entropy", "alpha", "q_min", "q_std"),
    iql=("value_loss", "q_loss", "actor_loss"),
    rebrac=("actor_loss", "q_loss", "q_mean", "lambda", "bc_loss", "critic_bc_loss"),
)
EXIT_COMPLETE, EXIT_ERROR, EXIT_REFUSED, EXIT_STOPPED = 0, 1, 2, 3
CHUNKS_RUN = [0]       # chunks scanned in this process (the gate test shows a refused run scans none)
COVERAGE_KEY_TAG = 51  # learners_iw.TAG_Y: coverage target-noise keys (band_readout.py)
COMPILE_EVENTS = "/jax/core/compile/"   # JAX's trace, lowering and backend-compile durations (jax.monitoring)
_COMPILE = dict(seconds=0.0, listening=False)


def _on_jax_event(name, seconds, **_):
    if name.startswith(COMPILE_EVENTS):
        _COMPILE["seconds"] += float(seconds)


def compile_clock():
    """Seconds JAX has spent tracing, lowering and compiling in this process so far (jax.monitoring's
    /jax/core/compile/* durations; the listener is registered on first use). A chunk's compile time is the difference
    across it, so the steady speed excludes the start-up recompiles (CQL's step compiles over its first chunk calls:
    unifloral_cql_bca.py docstring, "compiles")."""
    if not _COMPILE["listening"]:
        from jax import monitoring

        monitoring.register_event_duration_secs_listener(_on_jax_event)
        _COMPILE["listening"] = True
    return _COMPILE["seconds"]


class StopRun(RuntimeError):
    """A stop rule fired inside a run (plan section 6.1): the run stops, its record says which rule, at which outer
    step and why, and nothing retries it."""

    def __init__(self, rule, step, reason):
        super().__init__(f"stop rule {rule} at outer step {step}: {reason}")
        self.rule, self.step, self.reason = rule, int(step), str(reason)


class RecordMismatch(RuntimeError):
    """A run folder already holds a record of another identity, or a started run without a record."""


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def processor_seconds():
    """User plus system processor time of this process and its waited-for children (plan S6b's processor-hour)."""
    s, c = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return s.ru_utime + s.ru_stime + c.ru_utime + c.ru_stime


def canonical_sha256(obj):
    return hashlib.sha256(json.dumps(H.plain(obj), sort_keys=True, separators=(",", ":")).encode("utf8")).hexdigest()


def boot_id():
    """This boot's id (the kernel's random boot_id): process ids are comparable within one boot only."""
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return None


def process_alive(pid, boot, marker=None):
    """Whether process `pid` of boot `boot` still runs on this boot (and, given a marker, its command line holds it).
    False when the boot differs or the pid is unknown."""
    if pid is None or boot is None or boot != boot_id():
        return False
    try:
        cmd = Path(f"/proc/{int(pid)}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf8", "replace")
    except (OSError, ValueError):
        return False
    return marker is None or marker in cmd


# ------------------------------------------------------------------------------------------------ stop rule S9

def env_signature(record):
    """The parts of a run's environment stop rule S9 compares (plan 6.1): Python, the executable, the package versions,
    the card's name and driver, the environment variables, XLA_FLAGS in effect, matrix precision, JAX's platforms and
    backend, the devices, and Unifloral's manifest (each learner's own file hash is stop rule S1's). record: a run
    record, or any dict with environment.import_record (unifloral_import.record())."""
    ir = record["environment"]["import_record"]
    card = ir.get("card") or {}
    return dict(python=ir.get("python"), executable=ir.get("executable"), packages=ir.get("packages"),
                card=dict(name=card.get("name"), driver_version=card.get("driver_version")) if card else None,
                env=ir.get("env"), xla_flags=(ir.get("xla_flags") or {}).get("in_effect"),
                matmul_precision=ir.get("matmul_precision"), jax_platforms=ir.get("jax_platforms"),
                backend=ir.get("backend"), devices=ir.get("devices"), device=ir.get("device"),
                unifloral_revision=ir.get("unifloral_revision"), unifloral_manifest_sha256=ir.get("manifest_sha256"))


S9_KEYS = ("python", "executable", "packages", "card", "env", "xla_flags", "matmul_precision", "jax_platforms",
           "backend", "devices", "device", "unifloral_revision", "unifloral_manifest_sha256")


def check_s9(record, reference):
    """S9: a run's environment signature equals the frozen one, in every key either of them names (a key the reference
    lacks is a difference, so a partial frozen signature cannot skip a field)."""
    sig = env_signature(record)
    keys = sorted(set(sig) | set(reference))
    diff = {k: dict(run=sig.get(k), frozen=reference.get(k, "(not in the frozen signature)")) for k in keys
            if k not in reference or sig.get(k) != reference[k]}
    return dict(passed=not diff, differs=diff)


def frozen_s9_signature(package=PACKAGE):
    """prespec.json's environment.s9_signature (decision 5), or None when prespec.json or the signature is missing.
    A signature missing any of S9_KEYS is returned as it is; check_s9 then reports the missing keys as differences."""
    pre = Path(package) / gate.PRESPEC_NAME
    try:
        env = json.loads(pre.read_text(encoding="utf8")).get("environment")
    except (OSError, ValueError, AttributeError):
        return None
    sig = env.get("s9_signature") if isinstance(env, dict) else None
    return sig if isinstance(sig, dict) and sig else None


# ------------------------------------------------------------------------------------------------ the prepared data

def prepared_path(quality, rep):
    """runs/shift_hosts/data/<quality>/seed<k>/prepared.npz (restated from prepare.path_for, which imports the scoring
    environment's modules; INTERFACES.md section 6)."""
    if quality not in QUALITIES:
        raise ValueError(f"quality must be one of {QUALITIES}, not {quality!r}")
    return DATA / quality / f"seed{int(rep)}" / "prepared.npz"


class Prepared(NamedTuple):
    path: str
    arrays: dict            # {"<set>/<field>": numpy array}, every check of handoff.read passed
    meta: dict
    npz_sha256: str
    manifest_sha256: str
    made_up: bool
    quality: str
    rep: int

    def rows(self, which):
        """{obs, action, reward, next_obs, done} of a set (float32), for unifloral_import.make_transition."""
        g = H.group(self.arrays, which)
        return {k: g[k] for k in ("obs", "action", "reward", "next_obs", "done")}

    def field(self, which, name):
        return self.arrays[f"{which}/{name}"]

    def n(self, which):
        return int(self.arrays[f"{which}/obs"].shape[0])


def read_prepared(path, *, made_up):
    """A prepared-data file after handoff.read's checks (both sidecars, the manifest, every array's dtype, shape and
    SHA-256) and stop rule S11's row counts (plan 6.1): the kept training and bank rows equal the weighting test's
    records (meta.recorded_rows, from runs/shift_iw_v5/<quality>/host/gap/seed<k>/record.json), the full log equals the
    host/full record, and the preparation's own S11 comparison passed (meta.s11.ok). made_up: whether the caller expects
    a made-up file (meta.made_up must agree). Raises StopRun("S11", 0, ...) on a count mismatch."""
    path = Path(path)
    arrays, manifest = H.read(path)
    meta = manifest.get("meta") or {}
    if meta.get("schema") != PREPARED_SCHEMA:
        raise H.HandoffError(f"{path} is not a {PREPARED_SCHEMA} file")
    if bool(meta.get("made_up")) != bool(made_up):
        raise gate.ApprovalError(f"{path}: made_up is {meta.get('made_up')!r}, the caller expects {bool(made_up)}")
    rows = {s: int(arrays[f"{s}/obs"].shape[0]) for s in ("train", "full", "bank", "test_strong")}
    rec = meta.get("recorded_rows") or {}
    want = dict(train=rec.get("gap", {}).get("train"), bank=rec.get("gap", {}).get("bank"),
                full=rec.get("full", {}).get("train"))
    bad = [f"{k}: {rows[k]} rows, recorded {v}" for k, v in want.items() if v is None or int(v) != rows[k]]
    if bad or (meta.get("s11") or {}).get("ok") is not True:
        raise StopRun("S11", 0, f"{path}: the prepared rows differ from the recorded ones ({'; '.join(bad)}) or the "
                                f"preparation's own S11 check did not pass")
    msha = H.sha256_file(H.manifest_path(path))
    return Prepared(str(path), arrays, meta, manifest["npz_sha256"], msha, bool(made_up),
                    str(meta.get("quality")), int(meta.get("seed")))


def prepared_header(path, *, made_up):
    """A Prepared without its arrays (the manifest and both sidecars checked), enough for a run's identity: lets a
    finished run be skipped before anything is imported or loaded."""
    path = Path(path)
    manifest = H.read_manifest(path)
    meta = manifest.get("meta") or {}
    if bool(meta.get("made_up")) != bool(made_up):
        raise gate.ApprovalError(f"{path}: made_up is {meta.get('made_up')!r}, the caller expects {bool(made_up)}")
    digest = H.verify_sidecar(path)
    if manifest.get("npz_sha256") != digest or meta.get("schema") != PREPARED_SCHEMA:
        raise H.HandoffError(f"{path} is not the prepared file its manifest describes")
    return Prepared(str(path), {}, meta, digest, H.sha256_file(H.manifest_path(path)), bool(made_up),
                    str(meta.get("quality")), int(meta.get("seed")))


def write_made_up_prepared(path, seed, *, rep=None, quality="made-up"):
    """A prepared-format file of the made-up arrays of tests/madeup.py (INTERFACES.md section 14), for the stage-0
    tests and the stage driver's dry run only: the same field names and dtypes as INTERFACES.md section 6 for the sets
    train, full (the made-up builder has no uncut log, so it repeats train), bank and test_strong (madeup's strong-law
    steps). meta says made_up true, the builder's seed (made_up_seed), the data seed rep the run's keys use (default:
    the builder's seed), a quality label (the dry run uses one label per quality, so their folders differ), the
    made-up ball, and the made-up label; state and next_state are the float64 of the float32 observations; the
    'estimated' weight is a fixed made-up perturbation of the exact one. Nothing here is ThrusterChain data. Returns the
    manifest."""
    from experiments.shift_hosts.tests import madeup

    if not str(quality).startswith("made-up"):
        raise ValueError("a made-up file's quality label starts with 'made-up'")
    m = madeup.build(int(seed))
    est = lambda c, w: w * np.exp(0.2 * (np.asarray(c, np.float64) - 0.6))
    sets = dict(train=("train", m.train), full=("train", m.train), bank=("bank", m.bank),
                test_strong=("strong", m.strong))
    arrays = {}
    for name, (src, rows) in sets.items():
        n = len(rows.obs)
        a = dict(obs=rows.obs, action=rows.action, reward=rows.reward, next_obs=rows.next_obs, done=rows.done,
                 next_action=m.next_action[src], next_action_true=m.next_true[src], episode=m.episode[src],
                 t=m.t[src], row=np.arange(n, dtype=np.int64), state=rows.obs.astype(np.float64),
                 next_state=rows.next_obs.astype(np.float64), reward64=rows.reward.astype(np.float64),
                 current=m.current[src], in_ball=m.in_ball[src], weight_strong=m.weight[src],
                 weight_strong_estimated=est(m.current[src], m.weight[src]))
        for k, v in a.items():
            arrays[f"{name}/{k}"] = np.asarray(v)
    n_train, n_bank = len(m.train.obs), len(m.bank.obs)
    meta = dict(schema=PREPARED_SCHEMA, made_up=True, made_up_seed=int(seed), world=m.world, quality=str(quality),
                seed=int(seed if rep is None else rep), rows={s: int(len(arrays[f"{s}/obs"])) for s in sets},
                recorded_rows=dict(gap=dict(train=n_train, bank=n_bank), full=dict(train=n_train, bank=n_bank)),
                s11=dict(ok=True, rule="made-up arrays: the builder's own counts"),
                ball=dict(gap=dict(centre=[float(v) for v in np.asarray(m.ball_center, np.float64)],
                                   rho=float(m.ball_radius))),
                about="made-up arrays (tests/madeup.py), never ThrusterChain data; full repeats train")
    return H.write(path, arrays, meta=meta)


# ------------------------------------------------------------------------------------------------ permission

def made_up_permission(prepared):
    """guard.MADE_UP for a made-up prepared file, after registering the builder's arrays in this process
    (tests/madeup.build) and checking that every set this run reads, as Unifloral's Transition leaves for TD3+BC's and
    ReBRAC's field orders, is exactly registered content (guard.require_made_up). Any other arrays are refused."""
    from experiments.shift_hosts.tests import madeup

    madeup.build(int(prepared.meta["made_up_seed"]))
    items = []
    for s in ("train", "full", "bank", "test_strong"):
        r = prepared.rows(s)
        base = (r["obs"], r["action"], r["reward"], r["next_obs"])
        items.append(base + (r["done"],))
        items.append(base + (prepared.field(s, "next_action"), r["done"]))
    guard.require_made_up(*items)
    return guard.MADE_UP


def describe_permission(permission):
    if permission is guard.MADE_UP:
        return None
    if isinstance(permission, gate.Approval):
        return permission.describe()
    raise gate.ApprovalError("no permission")


def code_identity(permission):
    """The code part of a run's identity and record: the code manifest's SHA-256 (gate.code_manifest: every package
    file and the repository code it runs), the fingerprint and prespec.json's SHA-256 when the approval names them."""
    manifest = gate.code_manifest()
    out = dict(code_manifest_sha256=canonical_sha256(manifest), files=len(manifest))
    if isinstance(permission, gate.Approval):
        out.update(fingerprint=permission.fingerprint, prespec_sha256=permission.prespec_sha256)
    else:
        out.update(fingerprint=None, prespec_sha256=None)
    return out


# ------------------------------------------------------------------------------------------------ the schedule

class Schedule(NamedTuple):
    num_updates: int        # Args.num_updates (IQL's cosine schedules run over it)
    eval_interval: int      # Args.eval_interval, the chunk length
    steps: int              # outer steps this run takes
    refresh_steps: tuple    # BCA refreshes after these outer steps (between chunks)
    switch_step: int        # CQL's dose-active program from this outer step on (the first refresh step)
    key_hash_steps: tuple   # S4
    s10_step: int           # S10
    snapshot_steps: tuple   # S10's step, 50%, the last refresh, the last step
    last_refresh: Any       # the schedule's last refresh step (the plan's 95,000), or None


def make_schedule(*, num_updates=NUM_UPDATES, eval_interval=EVAL_INTERVAL, steps=None, warmup=REFRESH_WARMUP,
                  every=REFRESH_EVERY, key_hash_steps=None):
    """The schedule of one run. The plan's (plan 3.3, 4.1, 6.1): 40 chunks of 2,500, refreshes after 10,000, ...,
    95,000, S4 at 2,500 / 50,000 / 100,000, S10 at 10,000, snapshots at 10,000, 50,000, 95,000 and 100,000; steps <
    num_updates truncates it (stage 1: 15,000, one refresh). Made-up runs pass smaller numbers."""
    from experiments.shift_hosts import bca_core as BC

    steps = int(num_updates if steps is None else steps)
    if steps <= 0 or steps % eval_interval or steps > num_updates or num_updates % eval_interval:
        raise ValueError(f"steps ({steps}) and num_updates ({num_updates}) must be positive multiples of the chunk "
                         f"length {eval_interval}, steps at most num_updates")
    full_refresh = BC.refresh_steps(num_updates, warmup, every)
    if any(t % eval_interval for t in full_refresh):
        raise ValueError("every refresh step must end a chunk")
    refresh = tuple(t for t in full_refresh if t < steps)
    switch = int(full_refresh[0]) if full_refresh else steps + 1
    chunk_end = lambda x: int(math.ceil(x / eval_interval) * eval_interval)
    if key_hash_steps is None:
        key_hash_steps = (eval_interval, chunk_end(num_updates / 2), num_updates) if num_updates != NUM_UPDATES \
            else KEY_HASH_STEPS
    key_hash = tuple(sorted({int(t) for t in key_hash_steps if t <= steps}))
    s10 = int(full_refresh[0]) if full_refresh else eval_interval
    last = int(full_refresh[-1]) if full_refresh else None
    snaps = {s10, chunk_end(num_updates / 2), steps}
    if last is not None:
        snaps.add(last)
    snaps = tuple(sorted(t for t in snaps if t <= steps))
    return Schedule(int(num_updates), int(eval_interval), steps, refresh, switch, key_hash, s10, snaps,
                    last if (last is not None and last < steps) else None)


def is_plan_schedule(s):
    return (s.num_updates == NUM_UPDATES and s.eval_interval == EVAL_INTERVAL and s.steps in (NUM_UPDATES, STAGE1_STEPS)
            and s == make_schedule(steps=s.steps))


# ------------------------------------------------------------------------------------------------ trees and snapshots

def _path_part(k):
    import jax

    tu = jax.tree_util
    if isinstance(k, tu.DictKey):
        return str(k.key)
    if isinstance(k, tu.GetAttrKey):
        return str(k.name)
    if isinstance(k, tu.SequenceKey):
        return str(k.idx)
    if isinstance(k, tu.FlattenedIndexKey):
        return str(k.key)
    return str(k)


def tree_arrays(tree, prefix):
    """{"<prefix>/<path>": numpy array} for every leaf of a pytree, named by its key path (dict keys, attribute names,
    tuple positions), each leaf through jnp.asarray first (guard.state_sha256's rule), for handoff.write."""
    import jax
    import jax.numpy as jnp

    out = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        name = "/".join([prefix] + [_path_part(k) for k in path])
        if name in out:
            raise ValueError(f"two leaves share the name {name}")
        out[name] = np.asarray(jnp.asarray(leaf))
    return out


def tree_from_arrays(template, arrays, prefix):
    """The inverse of tree_arrays: the template's structure filled with the named arrays (as JAX arrays), after
    checking every leaf's dtype and shape against the template's."""
    import jax
    import jax.numpy as jnp

    paths, treedef = jax.tree_util.tree_flatten_with_path(template)
    leaves = []
    for path, leaf in paths:
        name = "/".join([prefix] + [_path_part(k) for k in path])
        want = jnp.asarray(leaf)
        if name not in arrays:
            raise H.HandoffError(f"snapshot holds no {name}")
        a = arrays[name]
        if a.dtype != np.dtype(want.dtype) or tuple(a.shape) != tuple(want.shape):
            raise H.HandoffError(f"{name} is {a.dtype} {a.shape}, the template's is {want.dtype} {want.shape}")
        leaves.append(jnp.asarray(a))
    return jax.tree_util.tree_unflatten(treedef, leaves)


def carry_arrays(carry):
    out = dict(tree_arrays(carry[0], "rng"))
    out.update(tree_arrays(carry[1], "agent"))
    if len(carry) > 2:
        out.update(tree_arrays(carry[2], "bca"))
    return out


def carry_from_arrays(template, arrays):
    parts = [tree_from_arrays(template[0], arrays, "rng"), tree_from_arrays(template[1], arrays, "agent")]
    if len(template) > 2:
        parts.append(tree_from_arrays(template[2], arrays, "bca"))
    return tuple(parts)


def state_hashes(agent_state):
    """guard.state_sha256 of every part of a learner state (AgentTrainState fields: actor, critics, targets, value,
    temperature), and of the whole."""
    out = {f: guard.state_sha256(getattr(agent_state, f)) for f in agent_state._fields}
    out["all"] = guard.state_sha256(agent_state)
    return out


def carry_hashes(carry):
    out = dict(key=guard.state_sha256(carry[0]), agent_state=guard.state_sha256(carry[1]))
    if len(carry) > 2:
        out.update(bca_state=guard.state_sha256(carry[2]),
                   bca_state_without_refusal_rule=guard.state_sha256(carry[2]._replace(most_caution=None)))
    return out


def save_snapshot(path, carry, meta):
    """The whole carry as one hand-off file (handoff.write: .npz, manifest and both sidecars). Returns its SHA-256."""
    return H.write(path, carry_arrays(carry), meta=dict(meta, schema=SNAPSHOT_SCHEMA))["npz_sha256"]


def load_snapshot(path, template):
    """(carry, manifest) of a snapshot, after handoff.read's checks, in the template carry's structure."""
    arrays, manifest = H.read(path)
    if (manifest.get("meta") or {}).get("schema") != SNAPSHOT_SCHEMA:
        raise H.HandoffError(f"{path} is not a {SNAPSHOT_SCHEMA} file")
    return carry_from_arrays(template, arrays), manifest


# ------------------------------------------------------------------------------------------------ building a setup

class IWInputs(NamedTuple):
    bank_weights: Any       # float64 [n_bank]: the exact weights of the bank rows (all 1 for iw_ones)
    train_masses: Any       # float64 [n_train]: each training row's test mass (all 1 for iw_ones)
    test_masses: Any        # float64 [n_test]: the strong-law test steps' exact weights (readouts)
    grid: Any               # iw.mass_grid on every mass looked up
    bank_weights_est: Any   # the classifier's bank weights (the estimated-weight readout)
    train_masses_est: Any
    test_masses_est: Any
    high: Any               # bool [n_train]: the highest-current tenth (dose change readout)


class Setup(NamedTuple):
    learner: str
    setup: str
    args: Any
    built: Any              # unifloral_import.Built
    module: Any             # the restated step's module (BCA setups) or None
    steps: dict             # {"main": step} or, CQL's BCA setups, {"pre": ..., "post": ...}
    carry0: tuple
    bank: Any               # the bank's Transition (BCA setups) or None
    bca: Any                # bca_core.BCASetup or None
    iw: Any                 # IWInputs (IW setups) or None
    n_train: int
    seeds: Any
    permission: Any
    refreshes: bool         # False for the sham alone (never refreshed) and the Unifloral setups


def restated(learner):
    import importlib

    return importlib.import_module(f"experiments.shift_hosts.unifloral_{learner}_bca")


def strong_weight(c):
    """The exact strong-law weight w(c) = (M(-2) / M(2)) exp(4 c) (restated from experiments/shift_iw/world.py:84-89,
    128-135, the arithmetic of world.ratio(c, "strong"); the prepared files carry it per row as weight_strong)."""
    lo, hi = 0.1, 1.1
    norm = lambda lam: math.exp(lam * lo) * math.expm1(lam * (hi - lo)) / lam
    c = np.asarray(c, np.float64)
    return (norm(-2.0) / norm(2.0)) * np.exp((2.0 - -2.0) * c)


def high_current_rows(current, share=HIGH_CURRENT_SHARE):
    """Rows whose current is at or above the (1 - share) quantile of the set's currents (prepare.high_current, restated
    from the weighting test's run.py:610-613)."""
    c = np.asarray(current, np.float64)
    return c >= np.quantile(c, 1.0 - share)


def iw_inputs(prepared, setup):
    """The IW setup's bank weights, test masses and grid (plan 4.2; iw.py's iw_refresh docstring)."""
    from experiments.shift_hosts import iw as IW

    if setup == "iw_ones":
        bw = np.ones(prepared.n("bank"), np.float64)
        tm = np.ones(prepared.n("train"), np.float64)
        sm = np.ones(prepared.n("test_strong"), np.float64)
        grid = IW.mass_grid(tm)
    else:
        bw = np.asarray(prepared.field("bank", "weight_strong"), np.float64)
        tm = np.asarray(prepared.field("train", "weight_strong"), np.float64)
        sm = np.asarray(prepared.field("test_strong", "weight_strong"), np.float64)
        grid = IW.mass_grid(np.concatenate([tm, sm, strong_weight([0.1, 1.1])]))
    return IWInputs(bw, tm, sm, grid,
                    np.asarray(prepared.field("bank", "weight_strong_estimated"), np.float64),
                    np.asarray(prepared.field("train", "weight_strong_estimated"), np.float64),
                    np.asarray(prepared.field("test_strong", "weight_strong_estimated"), np.float64),
                    high_current_rows(prepared.field("train", "current")))


def unifloral_args(learner, rep, schedule):
    """Unifloral's Args defaults with the plan's seed rule, gamma 0.98, num_updates and the chunk length (plan 3.2)."""
    from experiments.shift_hosts import bca_core as BC
    from experiments.shift_hosts import unifloral_import as UI

    return UI.args_for(learner, seed=BC.seeds(rep).unifloral, gamma=GAMMA, num_updates=schedule.num_updates,
                       eval_interval=schedule.eval_interval)


def build_setup(learner, setup, prepared, permission, schedule):
    """One setup, built exactly as INTERFACES.md section 10 ("Setups from these pieces") says, from Unifloral's own
    training-block set-up (unifloral_import.init_like_training_block)."""
    from experiments.shift_hosts import bca_core as BC
    from experiments.shift_hosts import iw as IW
    from experiments.shift_hosts import unifloral_import as UI

    if learner not in LEARNERS or setup not in SETUPS:
        raise ValueError(f"unknown learner {learner!r} or setup {setup!r}")
    seeds = BC.seeds(prepared.rep)
    args = unifloral_args(learner, prepared.rep, schedule)
    rows = prepared.rows("full" if setup == "room" else "train")
    built = UI.init_like_training_block(learner, args, rows)
    n_train = int(built.dataset.obs.shape[0])
    if setup in UNIFLORAL_SETUPS:
        step = UI.make_unifloral_step(built, args)
        return Setup(learner, setup, args, built, None, dict(main=step), (built.rng, built.agent_state), None, None,
                     None, n_train, seeds, permission, False)
    M = restated(learner)
    base = BC.fit_key_base(seeds.s91)
    bca = BC.BCASetup(learner, base, permission)
    scale = BC.make_scale_state(learner, built.obs_mean, built.obs_std, BC.scale_init_key(base, learner))
    bca0 = BC.initial_state(scale, n_train, most_caution=IW.most_caution(setup))
    if learner == "cql":
        steps = dict(pre=M.make_train_step(args, *built.apply_fns, built.dataset, bca=bca, dose_active=False),
                     post=M.make_train_step(args, *built.apply_fns, built.dataset, bca=bca))
    else:
        steps = dict(main=M.make_train_step(args, *built.apply_fns, built.dataset, bca=bca))
    bank_next = prepared.field("bank", "next_action") if learner == "rebrac" else "roll"
    bank = UI.make_transition(learner, prepared.rows("bank"), next_action=bank_next)
    iw = iw_inputs(prepared, setup) if setup in IW_SETUPS else None
    return Setup(learner, setup, args, built, M, steps, (built.rng, built.agent_state, bca0), bank, bca, iw, n_train,
                 seeds, permission, setup != "sham_alone")


def initial_hashes(s):
    out = dict(agent_state=state_hashes(s.carry0[1]), key=guard.state_sha256(s.carry0[0]))
    if len(s.carry0) > 2:
        out["scale"] = guard.state_sha256(s.carry0[2].scale)
        out["bca_state"] = guard.state_sha256(s.carry0[2])
    return out


# ------------------------------------------------------------------------------------------------ checks (S5)

def chunk_problem(s, out, t0):
    """Stop rule S5 on one chunk's per-step outputs (outer steps t0 + 1 ... t0 + chunk): (step, reason) of the first
    problem, or None. Any non-finite loss or metric; any false validity flag (the restated step's own, FLAGS); a dose
    outside its bounds (DOSE_BOUNDS) or non-finite IQL weights."""
    found = []
    for name, v in out.items():
        a = np.asarray(v)
        if a.dtype.kind == "f" and not np.all(np.isfinite(a)):
            bad = np.flatnonzero(~np.all(np.isfinite(a.reshape(a.shape[0], -1)), axis=1))
            found.append((int(bad[0]) + 1, f"{name} is not finite"))
    if s.bca is not None:
        for name in FLAGS[s.learner]:
            a = np.asarray(out[name]).reshape(-1).astype(bool)
            if not np.all(a):
                found.append((int(np.flatnonzero(~a)[0]) + 1, f"the per-step validity flag {name} is false"))
        if s.learner in DOSE_BOUNDS:
            lo, hi = DOSE_BOUNDS[s.learner]
            dmin, dmax = np.asarray(out["dose_min"]), np.asarray(out["dose_max"])
            bad = np.flatnonzero(~((dmin >= lo) & (dmax <= hi)))
            if bad.size:
                found.append((int(bad[0]) + 1, f"a dose outside [{lo}, {hi}]"))
        else:
            wmin = np.asarray(out["weight_min"])
            bad = np.flatnonzero(~(wmin >= 0))
            if bad.size:
                found.append((int(bad[0]) + 1, "a negative IQL weight"))
    if not found:
        return None
    step = min(s for s, _ in found)
    return t0 + step, "; ".join(sorted({r for s, r in found if s == step}))


def chunk_summary(out):
    """The mean of every loss and metric over the chunk (booleans as the share true), and the dose or weight range."""
    mean = {}
    for name, v in out.items():
        a = np.asarray(v)
        if a.dtype == bool:
            mean[name] = float(a.mean())
        elif a.dtype.kind in "fiu":
            mean[name] = float(np.mean(a.astype(np.float64)))
    extra = {}
    for lo, hi in (("dose_min", "dose_max"), ("weight_min", "weight_max")):
        if lo in out:
            extra[lo] = float(np.min(np.asarray(out[lo])))
            extra[hi] = float(np.max(np.asarray(out[hi])))
    return mean, extra


# ------------------------------------------------------------------------------------------------ band at training rows

def train_readout(s, agent_state, bca_state, thresholds=None):
    """BCA's dose (TD3+BC, CQL, ReBRAC) or IQL's actor weights at every training row with the band frozen in
    bca_state (plan 5.4's band readouts: the mean dose, IQL's weight shrinkage, refused rows). thresholds: float32 per
    training row (default the state's own). Returns (per-row values float64, summary dict)."""
    import jax.numpy as jnp

    from experiments.shift_hosts import iw as IW

    ds = s.built.dataset
    ref = bca_state.reference
    R = bca_state.thresholds if thresholds is None else jnp.asarray(thresholds, jnp.float32)
    eta = bca_state.scale.apply_fn(ref.cal_params, ds.obs, ds.action)
    if s.learner != "iql":
        dose, refused = IW.per_row_dose(ref, eta, R, s.bca.settings.blend, bca_state.most_caution)
        d = np.asarray(dose.dose, np.float64)
        return d, dict(mean_dose=float(d.mean()), min_dose=float(d.min()), max_dose=float(d.max()),
                       refused_rows=int(np.sum(np.asarray(refused))),
                       supported_share=float(np.mean(np.asarray(dose.support_mask))),
                       inputs_valid=bool(np.asarray(dose.inputs_valid)))
    _, q = s.module.band_parts(s.args, s.built.apply_fns, agent_state, ds, None)
    value = s.built.apply_fns[2](agent_state.value.params, ds.obs)
    adv = jnp.asarray(q) - value
    bw, refused = IW.per_row_iql_weights(ref, eta, R, adv, s.args.beta, s.args.exp_adv_clip, bca_state.most_caution)
    w = np.asarray(bw.weights, np.float64)
    plain = np.asarray(jnp.exp(adv * s.args.beta).clip(max=s.args.exp_adv_clip), np.float64)
    a = np.asarray(adv, np.float64)
    pos = (a > 0) & np.asarray(bw.support_mask) & (plain > 1.0)
    shrink = (plain[pos] - w[pos]) / (plain[pos] - 1.0) if pos.any() else np.zeros(0)
    return w, dict(mean_weight=float(w.mean()), mean_unifloral_weight=float(plain.mean()),
                   mean_weight_shrinkage=float(shrink.mean()) if shrink.size else None,
                   positive_advantage_share=float(np.mean(a > 0)), refused_rows=int(np.sum(np.asarray(refused))),
                   supported_share=float(np.mean(np.asarray(bw.support_mask))),
                   inputs_valid=bool(np.asarray(bw.inputs_valid)))


def recompute_bank_scores(s, agent_state, new_state, refresh_key):
    """Stop rule S3's recomputation (plan 4.2): the bank scores from the stored state with the refresh's own key and
    the learner's own target, against the reference the refresh froze. TD3+BC: its recompute_bank_scores (the refresh's
    own compiled program). CQL, ReBRAC and IQL: band_parts with the refresh's noise key (the refresh's own computation:
    CQL's compiled band_parts, ReBRAC's and IQL's eager ones), eta from the frozen parameters, and the scores by
    bca_core.bank_scores (reference.freeze_reference's arithmetic). Returns float64 [n_bank]."""
    import jax.numpy as jnp

    from experiments.shift_hosts import bca_core as BC

    if s.learner == "td3_bc":
        return s.module.recompute_bank_scores(s.args, s.built.apply_fns, agent_state, new_state, s.bank, refresh_key,
                                              permission=s.permission)[0]
    guard.require_bca(s.permission, s.bank)
    ref = new_state.reference
    noise = None if s.learner == "iql" else BC.refresh_noise_key(refresh_key, s.learner)
    y, q = s.module.band_parts(s.args, s.built.apply_fns, agent_state, s.bank, noise)
    eta = new_state.scale.apply_fn(ref.cal_params, s.bank.obs, s.bank.action)
    return BC.bank_scores(eta, ref.residual_scale, jnp.asarray(y) - jnp.asarray(q))


# ------------------------------------------------------------------------------------------------ the run

class Run:
    """One run of one setup: the chunk loop with everything between chunks (module docstring). run() returns the
    record; StopRun inside it ends the run with outcome "stopped"."""

    def __init__(self, s, schedule, out_dir, prepared, *, stage, kind, device, identity, inject=None, log=print):
        self.s, self.schedule, self.out, self.prepared = s, schedule, Path(out_dir), prepared
        self.stage, self.kind, self.device, self.identity, self.inject, self.log = stage, kind, device, identity, \
            inject, log
        self.carry = s.carry0
        self.t = 0
        self.refreshes, self.losses, self.snapshots = [], [], {}
        self.hashes = dict(initial=initial_hashes(s), key_carry={}, params_10000=None, final=None)
        self.timings = dict(chunk_seconds=[], chunk_compile_seconds=[], refresh_seconds=[], table_seconds=[],
                            readout_seconds=[])
        self.readouts = None
        self.loss_sha = hashlib.sha256()
        self.refreshed_once = False

    # -------------------------------------------------------------------------------------------- pieces

    def program(self, t0):
        steps = self.s.steps
        if "main" in steps:
            return steps["main"], "main"
        name = "pre" if t0 < self.schedule.switch_step else "post"
        return steps[name], name

    def _inject(self, t0):
        """Made-up runs only: a fault for the dry run's simulated stops (unit0: BCA's unit set to 0 before the chunk,
        so every step is invalid; nan: the actor's parameters made NaN, so Unifloral's own losses are not finite)."""
        import jax
        import jax.numpy as jnp

        kind, _, at = (self.inject or "").partition(":")
        if not kind or t0 + self.schedule.eval_interval < int(at or 1) or t0 >= int(at or 1):
            return
        if kind == "unit0" and len(self.carry) > 2:
            self.carry = (self.carry[0], self.carry[1], self.carry[2]._replace(unit=jnp.asarray(0.0, jnp.float32)))
        elif kind == "nan":
            a = self.carry[1]
            actor = a.actor.replace(params=jax.tree_util.tree_map(lambda x: x * jnp.nan, a.actor.params))
            self.carry = (self.carry[0], a._replace(actor=actor)) + tuple(self.carry[2:])
        self.log(f"injected fault {kind} before outer step {t0 + 1} (made-up run)")

    def chunk(self):
        import jax

        t0 = self.t
        if self.inject:
            self._inject(t0)
        step_fn, name = self.program(t0)
        started, compiled0 = time.monotonic(), compile_clock()
        carry, out = jax.lax.scan(step_fn, self.carry, None, self.s.args.eval_interval)
        carry = jax.block_until_ready(carry)
        out = {k: np.asarray(v) for k, v in out.items()}
        CHUNKS_RUN[0] += 1
        seconds = time.monotonic() - started
        compiled = compile_clock() - compiled0
        self.carry, self.t = tuple(carry), t0 + self.s.args.eval_interval
        self.timings["chunk_seconds"].append(round(seconds, 4))
        self.timings["chunk_compile_seconds"].append(round(compiled, 4))
        sha = guard.tree_sha256(out)
        self.loss_sha.update(sha.encode())
        mean, extra = chunk_summary(out)
        self.losses.append(dict(step=self.t, program=name, seconds=round(seconds, 4),
                                compile_seconds=round(compiled, 4), sha256=sha, mean=mean, **extra))
        return out, t0

    def after_chunk(self, out, t0):
        import jax

        t = self.t
        if t in self.schedule.key_hash_steps:                    # S4: the key carried out of the chunk
            self.hashes["key_carry"][str(t)] = guard.state_sha256(self.carry[0])
        if t == self.schedule.s10_step:                          # S10: just before the first refresh
            self.hashes["params_10000"] = dict(step=t, agent_state=guard.state_sha256(self.carry[1]),
                                               key=guard.state_sha256(self.carry[0]),
                                               parts=state_hashes(self.carry[1]))
        problem = chunk_problem(self.s, out, t0)
        if problem is not None:
            raise StopRun("S5", problem[0], problem[1])
        from experiments.shift_hosts import bca_core as BC

        if not bool(BC.tree_finite(self.carry[1])):
            raise StopRun("S5", t, "a learner parameter or optimizer state is not finite")
        rng, rng_eval = jax.random.split(self.carry[0])         # the evaluation split (td3_bc.py:340), discarded
        del rng_eval
        self.carry = (rng,) + tuple(self.carry[1:])
        if self.s.refreshes and t in self.schedule.refresh_steps:
            self.refresh(t)
        if t in self.schedule.snapshot_steps:
            self.snapshot(t)

    def refresh(self, t):
        """BCA's refresh after outer step t (module docstring), with the IW step and S3 for the IW setups."""
        import jax.numpy as jnp

        from experiments.shift_hosts import bca_core as BC
        from experiments.shift_hosts import iw as IW

        s = self.s
        rk = BC.refresh_key(s.seeds.s39, t)
        agent_state, bca_state = self.carry[1], self.carry[2]
        started = time.monotonic()
        ref = s.module.refresh(s.args, s.built.apply_fns, agent_state, bca_state, s.bank, rk, permission=s.permission)
        seconds = time.monotonic() - started
        rec = dict(step=int(t), seconds=round(seconds, 4), **ref.record)
        self.timings["refresh_seconds"].append(round(seconds, 4))
        if not ref.record["valid"]:
            self.refreshes.append(rec)
            raise StopRun("S5", t, "the refresh failed BCA's own validity checks (bca_core.apply_refresh)")
        threshold = float(np.asarray(ref.state.reference.threshold))
        if math.isnan(threshold):
            self.refreshes.append(rec)
            raise StopRun("S5", t, "the refresh's threshold is NaN")
        state = ref.state
        rec.update(s3_reproduced=None, iw=None, refusal=IW.refusal_name(np.asarray(bca_state.most_caution)))
        if s.iw is not None:
            started = time.monotonic()
            scores = recompute_bank_scores(s, agent_state, ref.state, rk)
            rec["s3_recomputed_equals_refresh_scores"] = bool(
                np.asarray(scores, np.float64).tobytes() == np.asarray(ref.scores, np.float64).tobytes())
            try:
                iwr = IW.iw_refresh(ref, s.iw.bank_weights, s.iw.train_masses, s.iw.grid, permission=s.permission,
                                    bank=s.bank, scores=scores)
            except IW.ReproductionError as error:
                rec["s3_reproduced"] = False
                self.refreshes.append(rec)
                raise StopRun("S3", t, str(error)) from error
            table_seconds = time.monotonic() - started
            self.timings["table_seconds"].append(round(table_seconds, 4))
            rec.update(s3_reproduced=iwr.record["s3_reproduced"], iw=iwr.record["iw"],
                       iw_refused_rows=iwr.record["refused_rows"], table_seconds=round(table_seconds, 4))
            state = iwr.state
            if t == self.schedule.last_refresh and s.setup != "iw_ones":
                started = time.monotonic()
                self.readouts = self.last_refresh_readouts(agent_state, ref, iwr, scores)
                self.timings["readout_seconds"].append(round(time.monotonic() - started, 4))
        values, summary = train_readout(s, agent_state, state)
        rec.update(train_rows=summary, refused_rows=summary["refused_rows"],
                   mean_dose=summary.get("mean_dose", summary.get("mean_weight")))
        self.carry = (self.carry[0], self.carry[1], state)
        self.refreshes.append(rec)
        self.refreshed_once = True
        self.log(f"refresh after {t}: certified {rec['certified']}, threshold {rec['threshold']}, "
                 f"refused rows {rec['refused_rows']}, {seconds:.1f}s")

    def last_refresh_readouts(self, agent_state, ref, iwr, scores):
        """The IW setup's last-refresh readouts (plan 4.6, 5.4; choice 20): the estimated-weight table and the
        shuffled-weights placebo on the refresh's bank scores (S3's recomputation, which reproduced the stored
        threshold) and stream (iw.last_refresh_arms, iw.threshold_tables; rows and steps looked up at their estimated
        masses for the estimated arm, at their exact masses otherwise), and IW's dose change against standard BCA at
        the kept training rows (iw.dose_change: the dose under the standard threshold and standard BCA's refusal rule
        as the base; IQL: the actor weights)."""
        import jax.numpy as jnp

        from experiments.shift_hosts import iw as IW

        s, iw = self.s, self.s.iw
        R_u = float(np.asarray(ref.state.reference.threshold))
        standard_R = np.full(s.n_train, np.float32(R_u), np.float32)
        standard_state = iwr.state._replace(most_caution=jnp.asarray(False))
        base, _ = train_readout(s, agent_state, standard_state, standard_R)
        exact_vals, _ = train_readout(s, agent_state, iwr.state)
        exact = (iw.bank_weights, np.concatenate([iw.train_masses, iw.test_masses]))
        estimated = (iw.bank_weights_est, np.concatenate([iw.train_masses_est, iw.test_masses_est]))
        arms, info = IW.last_refresh_arms(self.prepared.rep, exact, estimated)
        sub = {k: arms[k] for k in ("estimated", "shuffled")}
        tables = IW.threshold_tables(scores, ref.stream, sub, permission=s.permission, bank=s.bank)
        out = dict(info=info, value=("dose" if s.learner != "iql" else "actor weight"),
                   dose_change=IW.dose_change(exact_vals, base, iw.high),
                   exact=dict(test_steps=IW.per_step_summary(IW.lookup(iwr.table, iw.test_masses)[0], R_u)))
        for name, tab in tables.items():
            m_train = iw.train_masses_est if name == "estimated" else iw.train_masses
            m_test = iw.test_masses_est if name == "estimated" else iw.test_masses
            R_tr, i_tr = IW.lookup(tab, m_train)
            R_te, i_te = IW.lookup(tab, m_test)
            vals, summ = train_readout(s, agent_state, iwr.state, IW.table_float32(tab)[i_tr])
            out[f"{name}_weights"] = dict(
                table=tab.summary(R_u), kish=IW.kish(sub[name][0]),
                train_rows=dict(IW.per_step_summary(R_tr, R_u), lookup=IW.lookup_gaps(tab, m_train, i_tr)),
                test_steps=dict(IW.per_step_summary(R_te, R_u), lookup=IW.lookup_gaps(tab, m_test, i_te)),
                dose_change=IW.dose_change(vals, base, iw.high), train_readout=summ)
        return out

    def snapshot(self, t):
        from experiments.shift_hosts import actors_f32 as A32

        s = self.s
        d = self.out / "snapshots"
        meta = dict(learner=s.learner, setup=s.setup, step=int(t), quality=self.prepared.quality,
                    seed=self.prepared.rep, made_up=self.prepared.made_up, run_id=self.identity["run_id"],
                    identity_sha256=canonical_sha256(self.identity))
        sha = save_snapshot(d / f"step{t}.npz", self.carry, meta)
        params = self.carry[1].actor.params
        # trained: a learned policy (made-up arrays or not); evaluate.require_permission rolls it out in the simulator
        # only with the user's verified approval, never with guard.MADE_UP (no learned policy's return before it)
        ameta = dict(meta, actor_tree_sha256=guard.state_sha256(params), run_dir=str(self.out), trained=True,
                     kind="run-snapshot")
        norm = {}
        if s.learner in A32.NORMALIZED:
            norm = dict(obs_mean=s.built.obs_mean, obs_std=s.built.obs_std)
        if s.learner == "rebrac":
            ameta.update(actor_ln=bool(s.args.actor_ln), norm_obs=bool(s.args.norm_obs))
        man = A32.write_actor_file(self.out / "actors" / f"step{t}.npz", s.learner, params, meta=ameta, **norm)
        self.snapshots[str(t)] = dict(path=str(d / f"step{t}.npz"), sha256=sha,
                                      actor=dict(path=str(self.out / "actors" / f"step{t}.npz"),
                                                 sha256=man["npz_sha256"]))

    # -------------------------------------------------------------------------------------------- the loop

    def run(self):
        stop = None
        try:
            while self.t < self.schedule.steps:
                out, t0 = self.chunk()
                self.after_chunk(out, t0)
        except StopRun as error:
            stop = dict(rule=error.rule, step=error.step, reason=error.reason)
            self.log(f"STOP {error}")
        self.hashes["final"] = dict(step=self.t, losses_sha256=self.loss_sha.hexdigest(),
                                    parts=state_hashes(self.carry[1]), **carry_hashes(self.carry))
        return stop


# ------------------------------------------------------------------------------------------------ records

def run_id(stage, quality, rep, learner, setup):
    return f"{stage}/{quality}/seed{int(rep)}/{learner}-{setup}"


def run_dir(root, stage, quality, rep, learner, setup):
    return Path(root) / stage / quality / f"seed{int(rep)}" / f"{learner}-{setup}"


def identity_of(stage, prepared, learner, setup, schedule, device, permission):
    return dict(schema=SCHEMA, run_id=run_id(stage, prepared.quality, prepared.rep, learner, setup), stage=stage,
                quality=prepared.quality, seed=prepared.rep, learner=learner, setup=setup, made_up=prepared.made_up,
                device=device, schedule=schedule._asdict(),
                data=dict(npz_sha256=prepared.npz_sha256, manifest_sha256=prepared.manifest_sha256),
                code=dict(code_identity(permission)))


def existing(out_dir, identity, *, replace_incomplete=False):
    """The verified record of a finished run with this identity (skip it), or None when the folder is new (or holds an
    incomplete run the operator chose to replace). Refuses a record of another identity and an incomplete run."""
    out_dir = Path(out_dir)
    rec_path, started = out_dir / "record.json", out_dir / "started.json"
    if rec_path.exists():
        rec = H.read_json(rec_path)
        if rec.get("identity_sha256") != canonical_sha256(identity):
            raise RecordMismatch(f"{rec_path} holds a run of another identity (data, code, settings or device "
                                 "differ); it is never overwritten")
        return rec
    if started.exists():
        if not replace_incomplete:
            raise RecordMismatch(f"{out_dir} holds a started run without a record (it stopped without finishing); "
                                 "nothing retries it silently: pass --replace-incomplete to start it again")
        try:
            begun = H.read_json(started)
        except (H.HandoffError, OSError, ValueError) as error:
            raise RecordMismatch(f"{started} does not verify ({error}), so whether its run still lives cannot be "
                                 "told; nothing is deleted (remove the folder by hand once its process is gone)") \
                from error
        if process_alive(begun.get("pid"), begun.get("boot_id"), "experiments.shift_hosts.runner"):
            raise RecordMismatch(f"{out_dir}: the started run's process {begun.get('pid')} is still alive on this boot; "
                                 "nothing is deleted while it runs")
        for sub in ("snapshots", "actors"):
            for f in sorted((out_dir / sub).glob("*")) if (out_dir / sub).exists() else ():
                f.unlink()
        started.unlink()
    return None


def execute(prepared, learner, setup, permission, schedule, *, stage, root=RUNS, device="cpu", kind=None,
            replace_incomplete=False, inject=None, log=print):
    """Run one setup end to end and write its record (INTERFACES.md section 13). Returns (record, status) with status
    "complete", "stopped" or "skipped" (a finished run of the same identity)."""
    from experiments.shift_hosts import bca_core as BC
    from experiments.shift_hosts import iw as IW
    from experiments.shift_hosts import unifloral_import as UI

    if inject and not prepared.made_up:
        raise gate.ApprovalError("faults are injected into made-up runs only (the dry run's simulated stops)")
    if not prepared.made_up and not is_plan_schedule(schedule):
        raise ValueError("real runs use the plan's schedule only (100,000 outer steps, or stage 1's 15,000)")
    # the permission must fit the data, here too (module docstring): the alone and room setups run Unifloral's imported
    # step, which has no entry check of its own
    if prepared.made_up:
        if permission is not guard.MADE_UP:
            raise gate.ApprovalError("a run on made-up arrays takes guard.MADE_UP, not another permission")
        made_up_permission(prepared)
    else:
        if not isinstance(permission, gate.Approval):
            raise gate.ApprovalError(f"{gate.REFUSAL} (runner.execute: a run on real data needs the user's verified "
                                     "approval, not guard.MADE_UP or anything else)")
        gate.verify(permission)
    kind = kind or ("stage0-test" if prepared.made_up else ("stage1-check" if stage == "stage1" else "run"))
    identity = identity_of(stage, prepared, learner, setup, schedule, device, permission)
    out = run_dir(root, stage, prepared.quality, prepared.rep, learner, setup)
    found = existing(out, identity, replace_incomplete=replace_incomplete)
    if found is not None:
        log(f"skip {identity['run_id']}: a finished record with the same identity exists ({found.get('outcome')})")
        return found, "skipped"
    out.mkdir(parents=True, exist_ok=True)
    H.write_json(out / "started.json", dict(identity=identity, started_utc=utc_now(), pid=os.getpid(),
                                            boot_id=boot_id()))
    started_utc, t_start, cpu0, compiled0 = utc_now(), time.monotonic(), processor_seconds(), compile_clock()
    s = build_setup(learner, setup, prepared, permission, schedule)
    run = Run(s, schedule, out, prepared, stage=stage, kind=kind, device=device, identity=identity, inject=inject,
              log=log)
    stop = run.run()
    wall = time.monotonic() - t_start
    # the steady speed: the median over the chunks after the first of each chunk's time less the compiling JAX did
    # inside it (compile_clock), so start-up recompiles (CQL's) never pass for training time (S6's speeds)
    net = [c - k for c, k in zip(run.timings["chunk_seconds"], run.timings["chunk_compile_seconds"])]
    steady = float(np.median(net[1:])) if len(net) > 1 else (net[0] if net else None)
    compile_s = compile_clock() - compiled0
    rec_ = UI.record()
    record = dict(
        schema=SCHEMA, kind=kind, identity=identity, identity_sha256=canonical_sha256(identity),
        run_id=identity["run_id"], stage=stage, quality=prepared.quality, seed=prepared.rep, learner=learner,
        setup=setup, outcome="stopped" if stop else "complete", stop=stop,
        data=dict(path=prepared.path, npz_sha256=prepared.npz_sha256, manifest_sha256=prepared.manifest_sha256,
                  made_up=prepared.made_up, rows={k: prepared.n(k) for k in ("train", "full", "bank", "test_strong")},
                  train_set="full" if setup == "room" else "train", train_sha256=guard.tree_sha256(s.built.dataset),
                  bank_sha256=guard.tree_sha256(s.bank) if s.bank is not None else None,
                  next_action_substituted=(prepared.meta.get("next_action") or {}).get("substituted")
                  if learner == "rebrac" else None),
        seeds=s.seeds._asdict(),
        settings=dict(args=dataclasses.asdict(s.args), bca=dataclasses.asdict(BC.SETTINGS) if s.bca else None,
                      num_updates=schedule.num_updates, eval_interval=schedule.eval_interval, steps=schedule.steps,
                      refresh_steps=list(schedule.refresh_steps) if s.refreshes else [],
                      snapshots=list(schedule.snapshot_steps), key_hash_steps=list(schedule.key_hash_steps),
                      s10_step=schedule.s10_step,
                      refusal=IW.refusal_name(IW.most_caution(setup)) if s.bca else None,
                      refusal_record=IW.refusal_record(learner, IW.most_caution(setup)) if s.bca else None,
                      weights_one=setup == "iw_ones",
                      programs=(dict(pre="dose_active=False before outer step %d" % schedule.switch_step,
                                     post="dose_active=True from there on") if "pre" in s.steps else None),
                      iw_grid=dict(points=int(len(s.iw.grid)), first=float(s.iw.grid[0]), last=float(s.iw.grid[-1]),
                                   sha256=hashlib.sha256(np.asarray(s.iw.grid).tobytes()).hexdigest())
                      if s.iw is not None else None),
        code=dict(identity["code"], approval=describe_permission(permission),
                  unifloral_sha256=rec_.get("sha256")),
        environment=dict(import_record=rec_, host=platform.node(),
                         taskset_cores=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None),
        timings=dict(started_utc=started_utc, finished_utc=utc_now(), wall_seconds=round(wall, 3),
                     processor_seconds=round(processor_seconds() - cpu0, 3),
                     card_seconds=round(wall, 3) if device == "gpu" else 0.0,
                     compile_seconds=round(compile_s, 3),
                     first_chunk_seconds=run.timings["chunk_seconds"][0] if run.timings["chunk_seconds"] else None,
                     steady_ms_per_outer_step=(1000.0 * steady / schedule.eval_interval) if steady else None,
                     outer_steps=run.t, **run.timings),
        hashes=run.hashes, snapshots=run.snapshots,
        validity=dict(all_steps_valid=stop is None or stop["rule"] != "S5", first_invalid_step=stop["step"]
                      if stop and stop["rule"] == "S5" else None,
                      flags=list(FLAGS[learner]) if s.bca else [], losses_checked="every floating output"),
        losses=run.losses, refreshes=run.refreshes, readouts=run.readouts)
    H.write_json(out / "record.json", record)
    log(f"{identity['run_id']}: {record['outcome']} after {run.t} outer steps, {wall:.1f}s")
    return record, record["outcome"]


# ------------------------------------------------------------------------------------------------ command line

def _log(msg):
    print(f"[{utc_now()}] {msg}", flush=True)


def _schedule_from(a, made_up):
    if not made_up:
        if any(v is not None for v in (a.num_updates, a.eval_interval, a.refresh_warmup, a.refresh_every)):
            raise ValueError("schedule overrides are for made-up runs only")
        steps = a.steps or NUM_UPDATES
        if steps not in (NUM_UPDATES, STAGE1_STEPS):
            raise ValueError(f"--steps must be {NUM_UPDATES} or stage 1's {STAGE1_STEPS}")
        return make_schedule(steps=steps)
    return make_schedule(num_updates=a.num_updates or NUM_UPDATES, eval_interval=a.eval_interval or EVAL_INTERVAL,
                         steps=a.steps, warmup=a.refresh_warmup or REFRESH_WARMUP,
                         every=a.refresh_every or REFRESH_EVERY)


def command_run(a):
    setup = "iw_ones" if a.weights_one else a.setup
    if a.weights_one and a.setup not in ("iw", "iw_ones"):
        raise ValueError("--weights-one goes with --setup iw (stop rule S2)")
    made_up = a.made_up_file is not None
    if not made_up:
        if a.quality is None or a.seed is None:
            raise ValueError("a real run names --quality and --seed")
        purpose = f"runner: {a.learner} {setup} {a.quality} seed {a.seed} ({a.stage})"
        permission = gate.require_run_approval(purpose)          # before anything is read (plan 6.1)
    device = a.device or os.environ.get("SHIFT_HOSTS_DEVICE") or "cpu"
    schedule = _schedule_from(a, made_up)
    root = Path(a.out_root) if a.out_root else RUNS
    path = a.made_up_file if made_up else prepared_path(a.quality, a.seed)
    header = prepared_header(path, made_up=made_up)
    identity = identity_of(a.stage, header, a.learner, setup, schedule, device,
                           permission if not made_up else guard.MADE_UP)
    found = existing(run_dir(root, a.stage, header.quality, header.rep, a.learner, setup), identity,
                     replace_incomplete=a.replace_incomplete)
    if found is not None:                                        # resumable: nothing imported, nothing run
        print(json.dumps(dict(status="skipped", run_id=found["run_id"], outcome=found["outcome"],
                              stop=found["stop"]), sort_keys=True), flush=True)
        return EXIT_STOPPED if found["outcome"] == "stopped" else EXIT_COMPLETE
    from experiments.shift_hosts import unifloral_import as UI

    UI.prepare(learners=(a.learner,), device=device)
    if made_up:
        prepared = read_prepared(path, made_up=True)
        permission = made_up_permission(prepared)
    else:
        check_s9_before_training(UI.record())                    # S9 before the first chunk (module docstring)
        prepared = read_prepared(path, made_up=False)
        gate.verify(permission)
    record, status = execute(prepared, a.learner, setup, permission, schedule, stage=a.stage, root=root,
                             device=device, replace_incomplete=a.replace_incomplete, inject=a.inject_fault, log=_log)
    print(json.dumps(dict(status=status, run_id=record["run_id"], outcome=record["outcome"], stop=record["stop"]),
                     sort_keys=True), flush=True)
    return EXIT_STOPPED if record["outcome"] == "stopped" else EXIT_COMPLETE


def check_s9_before_training(import_record, reference=None):
    """Stop rule S9 for a real run, before anything trains: this process's environment signature against prespec.json's
    environment.s9_signature (frozen_s9_signature; check_s9: every key either names). Raises StopRun("S9", 0, ...)."""
    reference = frozen_s9_signature() if reference is None else reference
    if not reference:
        raise StopRun("S9", 0, "prespec.json names no environment.s9_signature, so the frozen environment cannot be "
                               "checked (decision 5)")
    res = check_s9(dict(environment=dict(import_record=import_record)), reference)
    if not res["passed"]:
        raise StopRun("S9", 0, f"this run's environment differs from prespec.json's environment.s9_signature in "
                               f"{sorted(res['differs'])}: {json.dumps(H.plain(res['differs']), sort_keys=True)[:1500]}")
    return res


def command_s7(a):
    """S7's open loop, the training environment's half (plan 3.9): the learner's own float32 commands on the saved
    states, on the processor (actors_f32.open_loop_commands). It computes a learned actor's commands, never a return;
    still, unless the actor file and the states file are both marked made-up, it needs the user's verified approval
    first, as every entry point does (module docstring)."""
    from experiments.shift_hosts import actors_f32 as A32
    from experiments.shift_hosts import unifloral_import as UI

    ameta = H.read_manifest(a.actor).get("meta") or {}
    smeta = H.read_manifest(a.states).get("meta") or {}
    if not (ameta.get("made_up") is True and smeta.get("made_up") is True):
        gate.require_run_approval(f"runner s7-commands {a.actor}")
    learner = ameta.get("learner")
    UI.prepare(learners=(learner,), device="cpu")
    man = A32.open_loop_commands(a.states, a.actor, a.out)
    print(json.dumps(dict(status="complete", out=a.out, npz_sha256=man["npz_sha256"]), sort_keys=True), flush=True)
    return EXIT_COMPLETE


def command_made_up(a):
    man = write_made_up_prepared(a.out, a.seed, rep=a.rep, quality=a.quality)
    print(json.dumps(dict(status="complete", out=a.out, npz_sha256=man["npz_sha256"]), sort_keys=True), flush=True)
    return EXIT_COMPLETE


def parser():
    p = argparse.ArgumentParser(prog="python -m experiments.shift_hosts.runner", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="run one setup")
    r.add_argument("--learner", required=True, choices=LEARNERS)
    r.add_argument("--setup", required=True, choices=SETUPS)
    r.add_argument("--weights-one", action="store_true", help="IW with every weight 1 (the iw_ones setup, S2)")
    r.add_argument("--quality", choices=QUALITIES)
    r.add_argument("--seed", type=int)
    r.add_argument("--made-up-file", help="a made-up prepared file (stage-0 tests and the dry run)")
    r.add_argument("--steps", type=int, help=f"outer steps ({NUM_UPDATES}, or stage 1's {STAGE1_STEPS})")
    r.add_argument("--stage", default="stage2")
    r.add_argument("--out-root")
    r.add_argument("--device", choices=("cpu", "gpu"))
    r.add_argument("--replace-incomplete", action="store_true")
    r.add_argument("--num-updates", type=int, help="made-up runs only")
    r.add_argument("--eval-interval", type=int, help="made-up runs only")
    r.add_argument("--refresh-warmup", type=int, help="made-up runs only")
    r.add_argument("--refresh-every", type=int, help="made-up runs only")
    r.add_argument("--inject-fault", help="made-up runs only: unit0:<step> or nan:<step> (simulated stops)")
    s = sub.add_parser("s7-commands", help="S7's open loop, the training environment's half")
    s.add_argument("--states", required=True)
    s.add_argument("--actor", required=True)
    s.add_argument("--out", required=True)
    m = sub.add_parser("made-up-file", help="write a made-up prepared file (tests and the dry run)")
    m.add_argument("--seed", type=int, default=0, help="tests/madeup.build's seed")
    m.add_argument("--rep", type=int, help="the data seed the run's keys use (default: --seed)")
    m.add_argument("--quality", default="made-up", help="a label starting with 'made-up'")
    m.add_argument("--out", required=True)
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    try:
        if a.command == "run":
            return command_run(a)
        if a.command == "s7-commands":
            return command_s7(a)
        return command_made_up(a)
    except (gate.ApprovalError, RecordMismatch) as error:
        print(f"REFUSED: {error}", file=sys.stderr, flush=True)
        return EXIT_REFUSED
    except StopRun as error:
        # a stop before execute (S9 before training, S11 from read_prepared): no record exists, so the rule goes to the
        # status line the stage driver reads (stages.Driver._card_done)
        print(json.dumps(dict(status="stopped", run_id=None, outcome="stopped", record=False,
                              stop=dict(rule=error.rule, step=error.step, reason=error.reason)), sort_keys=True),
              flush=True)
        print(f"STOPPED: {error}", file=sys.stderr, flush=True)
        return EXIT_STOPPED
    except Exception:
        traceback.print_exc()
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
