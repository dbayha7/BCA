"""outputs/shift_hosts/results.json: the 12-row table, every claim with its badge, the expanded view and the reference
row "CORL-derived TD3+BC (weighting test)" (plan sections 5.4, 5.5, 6.1, 6.2, 6.3 and 9.1), built only from run records
made by the frozen code under the frozen pre-specification, each number carrying its record's SHA-256.

An experiment-local attachment (plan section 3.1): it changes none of the repository's declared methods. numpy, the
standard library and this package's handoff, gate, guard, data_copy and claims only: it imports neither the Unifloral
files, nor JAX, nor the simulator, nor runner.py or stages.py (the few rules it shares with them are restated below and
tests/test_claims.py checks them against the originals), so it runs in either environment (the scoring environment is
the natural one). The viewer's files of plan section 9.2 are not written here yet (their formats are in the claims
area's report).

Who may build what:
  build(approval)          a gate.Approval that still verifies (gate.verify): the real results. Every run record used
                           must say kind "run", made_up false (data and identity), carry the approval's fingerprint and
                           pre-specification SHA-256 and the frozen code manifest's SHA-256 (code.fingerprint,
                           code.prespec_sha256, code.code_manifest_sha256: the frozen code under the frozen rules; stop
                           rule S1) and the plan's schedule; every scores file must be scored under that approval by
                           scoring code whose SHA-256 values equal the approved code manifest. The driver's files must
                           say mode "real". The reference row is read from the verified copies of the weighting test's
                           records (data_copy). main() is this path; it writes outputs/shift_hosts/results.json with a
                           sidecar, and it refuses today: there is no frozen pre-specification and no terminal approval
                           yet.
  build(MADE_UP, frozen=)  stage-0 tests only: run records of kind "stage0-test" made from made-up prepared files
                           (the stage driver's dry run, or test fixtures in its layout), scores and copies that all say
                           made_up true, a dry-run driver, under a made-up frozen identity (Frozen(..., made_up=True); a
                           made-up run has no approval, so its records carry fingerprint and prespec_sha256 null, and
                           the identity names the code manifest SHA-256 they were made with); written only under
                           runs/shift_hosts/stage0/.
So no learned policy's return is read before the user's terminal approval: this study's records exist only after it,
and the reference row (the weighting test's learned TD3+BC) is read only under a verified approval.

What is read: the runner's, the stage driver's and band_readout's own files, where and as they write them (runner.py,
stages.py, band_readout.py; INTERFACES.md sections 13 and 16). Nothing is renamed or rewritten for this module.
  <root>                   the stage driver's --root: runs/shift_hosts for the real study (build's runs_root)
  <root>/stage2/<label>/seed<k>/<learner>-<setup>/     one run (runner.run_dir); label = the quality for real runs and
                           "made-up-<quality>" for the dry run's made-up files (the runner names the folder after the
                           prepared file's quality); setups alone, standard, iw, room, sham_alone (seed 0 only);
                           stage 3's twins, when they exist, under <root>/stage3/ alike
    record.json (+ .sha256)                          runner.execute's run record (fields below)
    lanes/scores-step<T>.npz (+ manifest, sidecars)  the driver's scoring lane, evaluate.score_actor(the record's
                           snapshots[T].actor.path, permission, out=...), at T = the final step (the schedule's steps,
                           100,000) and the 50% step (num_updates / 2 rounded up to a chunk, 50,000); the claims read
                           the final file's "scored" readout, the 50% file is shown only
    lanes/coverage-step<R>/coverage.json (+ .sha256) band_readout coverage at the record's last refresh R (95,000),
                           standard and IW setups; coverage.npz beside it, checked by its own sidecar and the SHA-256
                           coverage.json names (band_readout writes coverage.json over the npz's hand-off manifest, so
                           handoff.read cannot open the npz; an open issue for band_readout's owner)
    lanes/s7-step<T>/s7.json (+ .sha256)             S7 on seed 0's final actors; stage 1's under <root>/stage1/<label>/
  <root>/driver/status.json, state.json (+ .sha256), driver.log, STOP.json[.<time>]   the driver's mode ("real" or
                           "dry-run": the made-up mark of these files), its decisions, stop-rule checks, Claim 0
                           failures, the S9 reference, its stop reports (a STOP.json.<time> was moved aside by
                           --resume-after-stop) and its log, whose line "<run id>: S5 accepted by the operator as the
                           setup's outcome (--accept-s5)" is the only record of an accepted S5 the driver keeps (an open
                           issue for the runner's owner; state.json "accepted_s5" is read too if a later driver
                           writes it)

Where each thing comes from (the export/runner field mapping; INTERFACES.md section 17 has the same table):
  made-up mark        record.data.made_up and record.identity.made_up (both); scores: permission.kind and
                      actor_meta.made_up; s7: the states file's made_up; driver: status.json mode
  run identity        record.run_id = identity.run_id = "<stage>/<label>/seed<k>/<learner>-<setup>" = its folder;
                      record.{stage, quality, learner, setup, seed} the same; record.identity_sha256 = the canonical
                      SHA-256 of identity (runner.canonical_sha256)
  code identity (S1)  record.code and identity.code: {fingerprint, prespec_sha256, code_manifest_sha256}
  schedule            identity.schedule (runner.Schedule); real builds: exactly the plan's (PLAN_SCHEDULE)
  actor of step T     record.snapshots[T].actor.sha256 = scores meta.actor_npz_sha256; the scores' actor_meta (the actor
                      file's own meta, written by the runner) names the record's run_id, identity_sha256 and step T
  scores              lanes/scores-step<T>.npz (above); every readout and bank, recomputed from its per-start arrays
  coverage (4a, 4b)   coverage.json groups.recorded (the prepared strong-law test steps): own = the "iw" arm (each
                      step at its own threshold R(w(c))) for an IW setup, the "standard" arm for a standard setup;
                      standard = the "standard" arm; each {outside, inside, all}.coverage; the three kinds of pairs
                      (groups recorded, actor, visited; the dry run has the recorded pairs only) per arm with
                      {steps, coverage, interval, refused_share, certified_coverage} are shown as "pairs", and the
                      whole file as it is
  stop and its kind   record.outcome "stopped" with stop {rule, step, reason}: an outcome when S5 on alone or room, or
                      S5 on a BCA setup the operator accepted (driver.log, above); every other stop halts
  S4                  hashes.initial.agent_state.all (all setups, room included), data.train_sha256 (alone, standard,
                      IW, sham alone), hashes.key_carry[str(t)] for t in schedule.key_hash_steps (alone, standard, IW;
                      a stopped run only before its stop)
  S10                 hashes.params_10000 {agent_state, key} (stage 2's readout; the rule itself is stage 1's, in the
                      driver's checks)
  S3                  refreshes[*].s3_reproduced of the IW setups (and s3_recomputed_equals_refresh_scores, shown)
  S9                  record.environment.import_record through env_signature (stages.env_signature) against the
                      driver's state.json s9_reference
  IQL's Q and V       hashes.final.parts {dual_q, dual_q_target, value} (unifloral_iql_bca.critic_value_parts)
  Claim 2's pointer   IW setups: readouts.dose_change {mean, change_vs_unweighted} (iw.dose_change of the IW dose
                      against the dose at the standard threshold under standard BCA's refusal rule, kept training
                      rows, last refresh) and, for IQL, refreshes[-1].train_rows.mean_unifloral_weight:
                        change = change_vs_unweighted
                        extra  = (mean - change_vs_unweighted) - 1                (the mean standard-threshold dose - 1)
                        extra  = (mean - change_vs_unweighted) - mean_unifloral_weight          (IQL's actor weights)
                      (mean - change_vs_unweighted is the mean at the standard threshold, within float rounding)
  band readouts       refreshes[*] {step, certified, threshold, refused_rows, iw_refused_rows, mean_dose, train_rows
                      (IQL: mean_weight, mean_unifloral_weight, mean_weight_shrinkage)};
                      readouts.{estimated_weights, shuffled_weights}.dose_change (shown); the design effect of plan
                      section 5.4 is "not recorded" (runner.py does not compute it)
  compute             timings {wall_seconds, card_seconds, processor_seconds, compile_seconds,
                      steady_ms_per_outer_step}; the machine: environment.host, identity.device,
                      environment.taskset_cores
  ReBRAC's a'         record.data.next_action_substituted (the prepared file's shares; data_root's prepared manifest
                      when a record lacks them)
  stop rules          driver state.json checks (S2/<learner>, S4/..., S6, S6b/<quality>/<unit>, S8/<learner>,
                      S10/<learner>/<setup>, S9/<run> on failure), s9_reference, claim0_failures; STOP.json's rule;
                      this module's own checks (S1, S3, S4, S5, S7, S9, S10, S11)
  S7's scopes         driver state.json s7_stopped (version 3: {quality, learner, scope "learner" | "iw", setup, run,
                      stage, file, at_utc} per failed S7 check): passed to the claims (claims.judge), whose stopped
                      scopes read "not judged" with p = 1 in Holm's families; their runs enter no claim, table or
                      expanded-view number and are listed under qualities.<quality>.s7_stopped.excluded (a run that
                      trained but was never scored too, and it carries "s7_stopped" under "unscored" and "runs"); a
                      failed S7 check of a recorded scope is an outcome under integrity, any other fails S7
Every run read is listed in results.json "runs" with each of its fields "read" or "not recorded: <why>".

A stopped run is an outcome (passed to the claims as not finite) when it stopped by S5 on Unifloral's unchanged step
(alone or room: "a failure in Unifloral's own code is an outcome, not a defect", plan section 6.1) or by an S5 on a BCA
setup the operator accepted as the setup's outcome (not traced to a defect in this package); any other stop halts the
study and is listed under integrity, and its run counts as missing, so the claims stay "pending". A folder whose files
fail a check is left out and listed under "excluded" with the reason; a complete run without its final scores is listed
under "unscored".

results.json (schema "shift-hosts-results-v1"): label and caveats; built (time, permission, frozen identity, this
code's SHA-256, the driver's files read); integrity (S0-S11 from the driver's files and this module's own checks);
qualities.<quality>: reading_order, claim0_bar, complete, families (Holm), claims.<learner>.<claim>, table (the 12
phone-view rows), expanded.<learner> (setups.<setup> with records and per-seed metrics on all 7 banks, the 50%
snapshot, compute time, band readouts and coverage; pointers: IW against alone, the mild mix, room, the sham-alone
drift of choice 19, Claim 2's pointer, the 50% snapshot's comparisons, paired standard errors over the starts, CQL's
tanh-of-mean readout; readouts: ReBRAC's substituted next commands, whether IQL's Q and V are identical across its
three runs), writeups (the user's shelving format), stage3_order, totals (the card and processor seconds of every run read,
scored or not), s7_stopped (version 3's stopped scopes and the runs they leave out); study_rule; reference_row; runs; excluded; unscored;
halted_runs. Every per-seed array of a setup is in seed order and aligned with that setup's records list, which carries
each seed's record and scores SHA-256; claim deltas carry them per seed.

Run (after the user's terminal approval; from /root/bca-hosts, scoring environment):
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 taskset -c <cores> /opt/bca-shift-venv/bin/python \\
      -m experiments.shift_hosts.export [--out outputs/shift_hosts/results.json]
"""

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.shift_hosts import claims as C
from experiments.shift_hosts import data_copy as DC
from experiments.shift_hosts import gate
from experiments.shift_hosts import guard
from experiments.shift_hosts import handoff as H

SCHEMA = "shift-hosts-results-v1"
RUN_SCHEMA = "shift-hosts-run-v1"                          # runner.SCHEMA
SCORES_SCHEMA = "shift-hosts-scores-v1"                    # evaluate.SCORES_SCHEMA
COVERAGE_SCHEMA = "shift-hosts-band-readout-v1"            # band_readout.SCHEMA (kind "coverage")
STATE_SCHEMA = "shift-hosts-driver-state-v1"               # stages.STATE_SCHEMA
STATUS_SCHEMA = "shift-hosts-driver-status-v1"             # stages.STATUS_SCHEMA
STOP_SCHEMA = "shift-hosts-driver-stop-v1"                 # stages.STOP_SCHEMA
PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[1]
RUNS_ROOT = REPO / "runs" / "shift_hosts"                  # the stage driver's default --root (runner.RUNS)
DATA = RUNS_ROOT / "data"                                  # prepare.py's files (INTERFACES.md section 6)
OUT = REPO / "outputs" / "shift_hosts" / "results.json"
STAGE0 = RUNS_ROOT / "stage0"
START_BANKS_JSON = DC.start_banks_dir() / "start_banks.json"
START_BANKS_JSON_SHA256 = "cceb2f2d54c179cbfb00185c4d5dedf62c6ff7838f8357d6fe47596eeb3cb740"   # evaluate.py's pin
STAGES = ("stage2", "stage3")
STAGE_SETUPS = dict(stage2=C.CLAIM_SETUPS + ("room", "sham_alone"), stage3=C.STAGE3_SETUPS)
SETUPS = STAGE_SETUPS["stage2"] + STAGE_SETUPS["stage3"]
SETUP_SEEDS = dict({s: C.SEEDS for s in SETUPS}, sham_alone=(0,))     # choice 19: seed 0 only
UNIFLORAL_SETUPS = ("alone", "room")                                  # Unifloral's own, imported step
COVERAGE_SETUPS = ("standard", "iw")                                  # the driver's coverage lanes (stages.lanes_for)
COVERAGE_PAIRS = ("recorded", "actor", "visited")                     # band_readout.GROUPS: the three kinds of pairs
PAIR_FIELDS = ("steps", "coverage", "interval", "refused_share", "certified_coverage")   # band_readout.summarize
S7_SETUPS = C.CLAIM_SETUPS                                            # S7 lanes on seed 0's final actors
S7_NEVER_SCORED = "excluded: stopped by S7 (trained, never scored: the driver adds no lanes to a stopped scope)"
READOUTS = dict(td3_bc=("scored",), iql=("scored",), rebrac=("scored",), cql=("scored", "tanh_mean"))  # actors_f32
SCORED = "scored"
SUMMARY_FIELDS = ("J_mean", "J_se", "points_mean", "points_se", "J_discounted_mean", "regret_mean",
                  "regret_points_mean", "f1_rate", "f2_rate", "stall_share", "worst10_mean", "p10", "tracking_mean",
                  "energy_mean", "loss_from_unrecorded_throttle_points", "loss_from_unrecorded_throttle_return",
                  "J_capped_mean", "points_capped_mean", "late_peak_mean")        # evaluate.score's summary
SEED_DIR = re.compile(r"^seed(\d+)$")
STATUS_ORDER = ("not recorded", "passed", "readout", "outcome", "failed", "stopped")   # worst last
# The plan's schedule (runner.make_schedule(steps=100_000)._asdict(), as JSON): 40 chunks of 2,500, 18 refreshes after
# 10,000 ... 95,000, S4 at 2,500 / 50,000 / 100,000, S10 at 10,000, snapshots at 10,000, 50,000, 95,000 and 100,000.
PLAN_SCHEDULE = dict(num_updates=100_000, eval_interval=2_500, steps=100_000,
                     refresh_steps=list(range(10_000, 100_000, 5_000)), switch_step=10_000,
                     key_hash_steps=[2_500, 50_000, 100_000], s10_step=10_000,
                     snapshot_steps=[10_000, 50_000, 95_000, 100_000], last_refresh=95_000)
FINAL, HALF = PLAN_SCHEDULE["steps"], PLAN_SCHEDULE["steps"] // 2       # the real study's snapshot steps
# Choice 19 (decision 2 of the pre-specification; stages.CHOICE19): learners whose test (a2) matched only to rounding on
# the card, so S10 is a readout and a seed-0 sham-alone run exists. The driver's own decision is read from its stage-1
# S10 checks when they exist.
CHOICE19 = dict(td3_bc=True, cql=False, iql=True, rebrac=False)
IQL_SHARED_PARTS = ("dual_q", "dual_q_target", "value")    # unifloral_iql_bca.critic_value_parts
ACCEPTED_S5_LINE = re.compile(r"^\[[^\]]*\] (\S+): S5 accepted by the operator as the setup's outcome "
                              r"\(--accept-s5\)\s*$")    # stages.Driver._card_done's log line
POINTER_DEFINITION = ("change = readouts.dose_change.change_vs_unweighted (the mean over kept training rows of the "
                      "dose at each row's IW threshold minus the dose at the standard threshold under standard BCA's "
                      "refusal rule, last refresh; IQL: actor weights); extra = (dose_change.mean - change) - 1, the "
                      "mean standard-threshold dose minus 1 (IQL: minus "
                      "refreshes[-1].train_rows.mean_unifloral_weight, Unifloral's own mean weight)")
DESIGN_EFFECT_WHY = ("not recorded: runner.py does not compute the design effect (the correlation of the 5 bank scores "
                     "within an episode, plan section 5.4); refreshes[*].n_eff is the calibration weights' Kish size, "
                     "not it (an open issue for runner.py's owner)")
NOT_RECORDED_WHY = dict(
    S0="stage 0's suite records it (runs/shift_hosts/stage0/SUMMARY.json, frozen with the pre-specification); the "
       "stage driver does not",
    S11="the data copy and the prepared files' row counts are checked before any run (data_copy, runner.read_prepared, "
        "which stops a run with S11 before it starts); here only the scores' anchors are checked")

# the reference row (plan sections 5.1 and 6.3): the weighting test's own records, read from the verified copies
REFERENCE_NAME = "CORL-derived TD3+BC (weighting test)"
REFERENCE_ARMS = dict(alone=("hosts", "plain_gap"), room=("hosts", "plain_full"),
                      standard=("bca", "bca_unweighted"), iw=("bca", "bca_exact_strong"),
                      iw_single_threshold=("bca", "bca_exact_strong_single"))
REFERENCE_DELTAS = dict(standard_minus_alone=("standard", "alone", "capped:bca_unweighted_minus_plain_gap"),
                        iw_minus_standard=("iw", "standard", "capped:per_step_minus_unweighted"),
                        iw_minus_alone=("iw", "alone", "capped:bca_exact_strong_minus_plain_gap"),
                        room_minus_alone=("room", "alone", None))
REFERENCE_LAWS = C.WEIGHTING_BANKS
EXPLORE_LABEL = "EXPLORATORY: outside the pre-registered claims; the version-5 gates did not pass"  # explore_arms.LABEL

CAVEATS = (
    "Exploratory: the weighting test's data seeds 0-4, already seen by it; confirmation needs fresh data seeds 10-14 "
    "of its generator and a pre-registration approved at the user's terminal.",
    "Medium-expert was picked after seeing the CORL-derived TD3+BC's result there, the best of 5 qualities (one-sided "
    "p 0.0128; about 0.064 after allowing for that choice).",
    "An overlapping shift: the strong mix's currents lie inside the recorded range, so IW claims say nothing about "
    "shifts beyond the data.",
    "Unifloral's own default settings, not tuned for ThrusterChain; 100,000 outer steps with discount 0.98.",
    "JAX 0.6.2 (Unifloral pins a version below 0.6) and the card's deterministic, full float32 arithmetic.",
    "Low power: with Holm's correction a true gain of 1.8 return units passes only about 6-10% of the time; about 5-9 "
    "are needed to pass reliably (6.75-9.25 if only one learner has the gain, 5.25-7.25 if all four share it). A "
    "shelved claim means no effect large enough to see with 5 seeds, not no effect.",
    "Two point scales: the weighting banks put 0 at uniformly random commands, the four situations at the main study's "
    "recording controller; 100 is the stall-aware controller on both.",
    "Post hoc: S7's closed-loop limit is pre-specification version 2's 0.05 points per start (version 1: 0.01), set "
    "after version 1's stage 1 stopped on it (ReBRAC alone, big kicks, 0.0114 points) and before any return was read "
    "(PRESPEC.md section 19).",
    "Post hoc (pre-specification version 3, PRESPEC.md section 20): a failed S7 check no longer stops the study; it "
    "stops that learner (an alone, standard or weights-of-1 actor) or only its IW setup (an IW actor) in that quality. "
    "Such a learner's claims read 'not judged' and it stays in Holm's families with p = 1; S7's limit is unchanged at "
    "0.05 points per start.",
)


class Excluded(Exception):
    """A run folder whose files fail a check: left out of results.json and listed with the reason."""


def canonical_sha256(obj):
    """SHA-256 of the canonical JSON of obj (sorted keys, no spaces, handoff.plain first): restated from
    runner.canonical_sha256, the hash of a run's identity and of the code manifest in its record."""
    return hashlib.sha256(json.dumps(H.plain(obj), sort_keys=True, separators=(",", ":")).encode("utf8")).hexdigest()


@dataclass(frozen=True, eq=False)
class Frozen:
    """The frozen identity records must carry: the fingerprint and pre-specification SHA-256 (gate.fingerprint), the
    code manifest the scoring code's hashes must match, and the code manifest's canonical SHA-256 the run records name
    (code.code_manifest_sha256; by default the canonical SHA-256 of code_manifest). made_up marks the stage-0 tests'
    identity; a made-up run has no approval, so its records carry fingerprint and prespec_sha256 None."""

    fingerprint: str
    prespec_sha256: str
    code_manifest: dict = field(repr=False)
    made_up: bool = False
    code_manifest_sha256: str = None

    @property
    def manifest_sha256(self):
        return self.code_manifest_sha256 or canonical_sha256(self.code_manifest)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    """A path relative to the repository when it lies inside it, else as it is."""
    p = Path(path).resolve()
    try:
        return p.relative_to(REPO).as_posix()
    except ValueError:
        return p.as_posix()


def resolve(permission, frozen=None):
    """(Frozen, permission record) for a build: a verified gate.Approval (the real identity, from the approval and
    the code on disk) or guard.MADE_UP with a made-up Frozen. Anything else is refused (gate.ApprovalError)."""
    if permission is guard.MADE_UP:
        if not isinstance(frozen, Frozen) or frozen.made_up is not True:
            raise gate.ApprovalError("refused: a made-up build needs a made-up frozen identity, "
                                     "Frozen(..., made_up=True)")
        return frozen, dict(kind="made-up")
    if isinstance(permission, gate.Approval):
        gate.verify(permission)
        if frozen is not None:
            raise gate.ApprovalError("refused: a real build takes its frozen identity from the approval, not from "
                                     "the caller")
        return (Frozen(permission.fingerprint, permission.prespec_sha256, gate.code_manifest(), False),
                dict(kind="approval", **permission.describe()))
    raise gate.ApprovalError(f"refused: results.json is built with a verified gate.Approval (or guard.MADE_UP for the "
                             f"stage-0 tests), not {permission!r}")


def quality_label(quality, made_up):
    """The folder label of a quality: the quality itself, or the dry run's "made-up-<quality>" (stages.card_job)."""
    return f"made-up-{quality}" if made_up else quality


def run_id(stage, label, seed, learner, setup):
    """runner.run_id: "<stage>/<label>/seed<k>/<learner>-<setup>", also the run's folder under the root."""
    return f"{stage}/{label}/seed{int(seed)}/{learner}-{setup}"


def half_step(schedule):
    """The 50% snapshot's outer step: num_updates / 2 rounded up to a chunk's end (runner.make_schedule; 50,000)."""
    n, e = int(schedule["num_updates"]), int(schedule["eval_interval"])
    return int(math.ceil(n / 2 / e) * e)


def env_signature(record):
    """The parts of a run's environment stop rule S9 compares, restated from stages.env_signature: Python, the
    executable, the package versions, the card's name and driver, the environment variables, XLA_FLAGS in effect,
    matrix precision, JAX's platforms and backend, the devices, and Unifloral's revision and manifest."""
    ir = ((record.get("environment") or {}).get("import_record")) or {}
    card = ir.get("card") or {}
    return dict(python=ir.get("python"), executable=ir.get("executable"), packages=ir.get("packages"),
                card=dict(name=card.get("name"), driver_version=card.get("driver_version")) if card else None,
                env=ir.get("env"), xla_flags=(ir.get("xla_flags") or {}).get("in_effect"),
                matmul_precision=ir.get("matmul_precision"), jax_platforms=ir.get("jax_platforms"),
                backend=ir.get("backend"), devices=ir.get("devices"), device=ir.get("device"),
                unifloral_revision=ir.get("unifloral_revision"), unifloral_manifest_sha256=ir.get("manifest_sha256"))


# ------------------------------------------------------------------------------------------------ one run folder

def _check(conditions):
    for ok, why in conditions:
        if not ok:
            raise Excluded(why)


def stop_kind(setup, stop, accepted=False):
    """"outcome" for a stop that is the run's outcome (plan section 6.1): S5 on Unifloral's unchanged step (alone,
    room), or S5 on a BCA setup the operator accepted as the setup's outcome (stages --accept-s5, recorded in the
    driver's log); else "halt"."""
    stop = stop or {}
    if stop.get("rule") == "S5" and (setup in UNIFLORAL_SETUPS or accepted):
        return "outcome"
    return "halt"


def read_scores(folder, record, step, learner, frozen):
    """lanes/scores-step<step>.npz of a run (evaluate.write_scores through the driver's scoring lane), after every
    check: handoff's hash checks; the actor's SHA-256 against the record's snapshots[step].actor; the actor file's own
    meta (actor_meta) naming the record's run_id, identity_sha256 and this step; the permission and made-up marks; the
    scoring code's SHA-256 against the frozen code manifest; every readout and bank present; and each bank's J_mean,
    F1 and F2 rates recomputed from the per-start arrays bit for bit. None when the file does not exist."""
    path = Path(folder) / "lanes" / f"scores-step{int(step)}.npz"
    name = f"lanes/{path.name}"
    if not path.exists():
        return None
    try:
        arrays, manifest = H.read(path)
    except (H.HandoffError, OSError, ValueError, KeyError) as error:
        raise Excluded(f"{name}: {error}") from error
    meta = manifest.get("meta") or {}
    actor = (((record.get("snapshots") or {}).get(str(int(step)))) or {}).get("actor") or {}
    am = meta.get("actor_meta") or {}
    perm = meta.get("permission") or {}
    if frozen.made_up:
        perm_ok = perm.get("kind") == "made-up"
    else:
        perm_ok = (perm.get("kind") == "approval" and perm.get("fingerprint") == frozen.fingerprint
                   and perm.get("prespec_sha256") == frozen.prespec_sha256)
    code = meta.get("code_sha256") or {}
    other_code = sorted(p for p, s in code.items() if frozen.code_manifest.get(p) != s)
    _check([(meta.get("schema") == SCORES_SCHEMA, f"{name}: schema is not {SCORES_SCHEMA}"),
            (meta.get("learner") == learner and am.get("learner") == learner, f"{name}: scored another learner"),
            (actor.get("sha256") is not None and meta.get("actor_npz_sha256") == actor.get("sha256"),
             f"{name}: scored another actor file than the record's snapshots[{int(step)}].actor"),
            ((am.get("run_id"), am.get("step"), am.get("identity_sha256")) ==
             (record.get("run_id"), int(step), record.get("identity_sha256")),
             f"{name}: its actor file names another run, identity or step (actor_meta)"),
            (perm_ok and am.get("made_up") is frozen.made_up,
             f"{name}: its permission or made-up marks do not match this build"),
            (bool(code) and not other_code, f"{name}: scored by other code than the frozen code "
                                            f"(stop rule S1): {other_code[:3] or 'no code hashes'}")])
    summaries = meta.get("summaries") or {}
    _check([(set(summaries) == set(READOUTS[learner]), f"{name}: readouts {sorted(summaries)}, expected "
                                                        f"{list(READOUTS[learner])}")])
    out = dict(path=rel(path), step=int(step), npz_sha256=manifest["npz_sha256"],
               manifest_sha256=sha256_file(H.manifest_path(path)), actor_npz_sha256=meta["actor_npz_sha256"],
               anchors=meta.get("anchors") or {}, cache=meta.get("cache") or {}, readouts={})
    for readout in READOUTS[learner]:
        banks = {}
        for bank in C.BANKS:
            s = (summaries.get(readout) or {}).get(bank)
            key = f"{readout}/{bank}"
            _check([(isinstance(s, dict), f"{name}: no summary for {key}"),
                    (all(f"{key}/{m}" in arrays for m in ("J", "f1", "f2", "start")), f"{name}: {key}'s "
                                                                                      f"per-start arrays are missing")])
            J = np.asarray(arrays[f"{key}/J"], np.float64)
            f1, f2 = np.asarray(arrays[f"{key}/f1"]), np.asarray(arrays[f"{key}/f2"])
            n = int(len(J))
            same = (n == int(s.get("n", -1)) and len(f1) == n and len(f2) == n
                    and (float(J.mean()) == float(s["J_mean"]) or (math.isnan(J.mean()) and
                                                                   math.isnan(float(s["J_mean"]))))
                    and float(f1.mean()) == float(s["f1_rate"]) and float(f2.mean()) == float(s["f2_rate"])
                    and bool(np.all(np.isfinite(J))) == bool(s.get("finite")))
            _check([(same, f"{name}: {key}'s summary differs from its per-start arrays")])
            banks[bank] = dict({f: (None if s.get(f) is None else float(s[f])) for f in SUMMARY_FIELDS},
                               finite=bool(s["finite"]), n=n, f1_count=int(np.sum(f1)), f2_count=int(np.sum(f2)),
                               J_per_start=J, start=np.asarray(arrays[f"{key}/start"]))
        out["readouts"][readout] = banks
    out["finite"] = all(b["finite"] and b["J_mean"] is not None and math.isfinite(b["J_mean"])
                        for b in out["readouts"][SCORED].values())
    return out


def read_coverage(folder, record, record_sha256, frozen):
    """band_readout's coverage of a run at its last refresh, lanes/coverage-step<R>/coverage.json (module docstring),
    after its sidecar and identity checks: the record's SHA-256, run id, learner, setup and refresh step; whether an IW
    table was read equals whether the record's last refresh tabulated one; coverage.npz's own sidecar and the SHA-256
    coverage.json names for it. The made-up mark and code identity are the record's: coverage.json names the record by
    SHA-256 and carries neither (an open issue for band_readout's owner). Returns (coverage dict or None, why not)."""
    refs = [r for r in (record.get("refreshes") or []) if isinstance(r, dict)]
    if not refs:
        return None, "no band: the run never refreshed"
    last = refs[-1]
    step = int(last["step"])
    d = Path(folder) / "lanes" / f"coverage-step{step}"
    path = d / "coverage.json"
    name = f"lanes/coverage-step{step}/coverage.json"
    if not path.exists():
        lane = ("the driver's coverage lane has not written it" if record.get("setup") in COVERAGE_SETUPS else
                "the driver's coverage lanes are for the standard and IW setups only")
        return None, f"{name} does not exist ({lane})"
    try:
        cov = H.read_json(path)
        npz = H.verify_sidecar(d / "coverage.npz")
    except (H.HandoffError, OSError, ValueError) as error:
        raise Excluded(f"{name}: {error}") from error
    if not isinstance(cov, dict):
        raise Excluded(f"{name} is not a JSON object")
    iw = last.get("iw") is not None
    own_arm = "iw" if iw else "standard"
    recorded = (cov.get("groups") or {}).get("recorded") or {}

    def shares(arm):
        a = recorded.get(arm) or {}
        if not all(isinstance(a.get(g), dict) for g in ("outside", "inside", "all")):
            return None
        return {g: (None if a[g].get("coverage") is None else float(a[g]["coverage"]))
                for g in ("outside", "inside", "all")}

    _check([(cov.get("schema") == COVERAGE_SCHEMA and cov.get("kind") == "coverage",
             f"{name}: schema is not {COVERAGE_SCHEMA} (kind coverage)"),
            (cov.get("record_sha256") == record_sha256, f"{name} belongs to another run record"),
            ((cov.get("run_id"), cov.get("learner"), cov.get("setup"), cov.get("step")) ==
             (record.get("run_id"), record.get("learner"), record.get("setup"), step),
             f"{name} names another run or refresh step"),
            (bool(cov.get("iw_table")) is iw, f"{name}: whether it read an IW table differs from the record's last "
                                              f"refresh"),
            (cov.get("arrays_npz_sha256") == npz, f"{name}: coverage.npz differs from the SHA-256 it names"),
            (shares("standard") is not None and shares(own_arm) is not None,
             f"{name}: no {own_arm} and standard coverage of the recorded strong-law test steps")])
    groups = cov.get("groups") or {}
    pairs, pairs_missing = {}, {}
    for g in COVERAGE_PAIRS:
        if not isinstance(groups.get(g), dict):
            pairs_missing[g] = ("not recorded: the dry run computes the recorded pairs only (no rollout of a learned "
                                "policy)" if frozen.made_up else
                                f"not recorded: band_readout wrote no {g} group (its transitions file was not given)")
            continue
        pairs[g] = {arm: {part: {k: (groups[g][arm].get(part) or {}).get(k) for k in PAIR_FIELDS}
                          for part in ("outside", "inside", "all")}
                    for arm in ("standard", "iw") if isinstance(groups[g].get(arm), dict)}
    other = {k: v for k, v in cov.items() if k not in ("schema", "kind")}
    return dict(path=rel(path), sha256=sha256_file(path), step=step, own_arm=own_arm, own=shares(own_arm),
                standard=shares("standard"), standard_threshold=cov.get("standard_threshold"),
                arrays_npz_sha256=npz, pairs=pairs, pairs_not_recorded=pairs_missing, other=other,
                provenance="made-up mark and code identity are the run record's (coverage.json names it by "
                           "SHA-256)"), None


def dose_change_pointer(record, learner, schedule):
    """Claim 2's per-run pointer {change, extra} of an IW setup from its record (module docstring, POINTER_DEFINITION).
    Returns (pointer dict or None, why not)."""
    ro = record.get("readouts") or {}
    dc = ro.get("dose_change")
    if not isinstance(dc, dict):
        return None, "readouts.dose_change is not in the record"
    refs = [r for r in (record.get("refreshes") or []) if isinstance(r, dict)]
    last = refs[-1] if refs else {}
    want = "actor weight" if learner == "iql" else "dose"
    _check([(ro.get("value") == want, f"readouts.value is {ro.get('value')!r}, not {want!r}"),
            (last.get("step") == schedule.get("last_refresh"),
             "readouts.dose_change is not the schedule's last refresh's")])
    mean, change = dc.get("mean"), dc.get("change_vs_unweighted")
    if mean is None or change is None:
        return None, "readouts.dose_change has no mean or change_vs_unweighted"
    standard_mean = float(mean) - float(change)
    out = dict(change=float(change), iw_mean=float(mean), standard_mean=standard_mean, step=int(last["step"]),
               high_current_change=dc.get("high_current_change"), mean_abs_change=dc.get("mean_abs_change"),
               definition=POINTER_DEFINITION)
    if learner == "iql":
        uni = (last.get("train_rows") or {}).get("mean_unifloral_weight")
        if uni is None:
            return None, "refreshes[-1].train_rows.mean_unifloral_weight (IQL) is not in the record"
        out.update(unifloral_mean=float(uni), extra=standard_mean - float(uni))
    else:
        out["extra"] = standard_mean - 1.0
    return out, None


def read_s7(folder, record, frozen):
    """S7's results in a run folder (lanes/s7-step<T>/s7.json, the driver's S7 lane), each bound to the record's
    actor of that step through the states file's manifest (actor_npz_sha256, made_up). A real build needs the closed
    loop to have run (the dry run passes --no-closed-loop)."""
    out = []
    for path in sorted((Path(folder) / "lanes").glob("s7-step*/s7.json")):
        step = int(path.parent.name[len("s7-step"):])
        try:
            body = H.read_json(path)
            states = H.read_manifest(path.parent / "states.npz")
        except (H.HandoffError, OSError, ValueError) as error:
            out.append(dict(path=rel(path), step=step, passed=False, why=f"does not verify: {error}"))
            continue
        smeta = states.get("meta") or {}
        actor = (((record.get("snapshots") or {}).get(str(step))) or {}).get("actor") or {}
        bound = (smeta.get("actor_npz_sha256") == actor.get("sha256") is not None
                 and bool(smeta.get("made_up")) is frozen.made_up)
        closed = body.get("closed_loop")
        closed_ok = (closed is None and frozen.made_up) or (isinstance(closed, dict) and closed.get("passed") is True)
        passed = body.get("s7_passed") is True and bound and closed_ok
        out.append(dict(path=rel(path), sha256=sha256_file(path), step=step, passed=bool(passed),
                        s7_passed=body.get("s7_passed"), open_loop_passed=body.get("passed"),
                        closed_loop=None if closed is None else closed.get("passed"), bound_to_record=bool(bound)))
    return out


def _schedule_ok(schedule):
    try:
        steps, half = int(schedule["steps"]), half_step(schedule)
        snaps = [int(t) for t in schedule["snapshot_steps"]]
        return steps > 0 and steps in snaps and half in snaps and int(schedule["num_updates"]) >= steps
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False


def read_run(folder, quality, learner, setup, seed, frozen, *, stage="stage2", accepted_s5=frozenset()):
    """One run folder of the runner's layout -> dict(kind="outcome" | "halt" | "unscored", run=<claims input>,
    aux=<everything shown>), or Excluded. accepted_s5: the run ids whose S5 the operator accepted (read_driver)."""
    folder = Path(folder)
    path = folder / "record.json"
    try:
        rec = H.read_json(path)
    except (H.HandoffError, OSError, ValueError) as error:
        raise Excluded(f"record.json: {error}") from error
    if not isinstance(rec, dict):
        raise Excluded("record.json is not a JSON object")
    sha = sha256_file(path)
    ident = rec.get("identity") if isinstance(rec.get("identity"), dict) else {}
    data = rec.get("data") or {}
    code, icode = rec.get("code") or {}, ident.get("code") or {}
    sched = ident.get("schedule") if isinstance(ident.get("schedule"), dict) else {}
    label = quality_label(quality, frozen.made_up)
    rid = run_id(stage, label, seed, learner, setup)
    want = "stage0-test" if frozen.made_up else "run"
    names = (stage, label, learner, setup, seed)
    _check([(rec.get("schema") == RUN_SCHEMA and ident.get("schema") == RUN_SCHEMA, f"schema is not {RUN_SCHEMA}"),
            (rec.get("kind") == want, f"kind is {rec.get('kind')!r}, this build reads {want!r}"),
            (data.get("made_up") is frozen.made_up and ident.get("made_up") is frozen.made_up,
             f"made_up is {data.get('made_up')!r} (data) and {ident.get('made_up')!r} (identity), this build needs "
             f"{frozen.made_up}"),
            (rec.get("run_id") == rid and ident.get("run_id") == rid,
             f"the record's run_id {rec.get('run_id')!r} is not its folder's {rid!r}"),
            (tuple(rec.get(k) for k in ("stage", "quality", "learner", "setup", "seed")) == names
             and tuple(ident.get(k) for k in ("stage", "quality", "learner", "setup", "seed")) == names,
             "the record names another stage, quality, learner, setup or seed than its folder"),
            (rec.get("identity_sha256") == canonical_sha256(ident),
             "identity_sha256 is not the SHA-256 of the record's identity"),
            ((ident.get("data") or {}).get("npz_sha256") == data.get("npz_sha256") is not None,
             "the identity names another prepared file than data.npz_sha256"),
            (code.get("fingerprint") == frozen.fingerprint and icode.get("fingerprint") == frozen.fingerprint,
             "made by other code or under another pre-specification: code.fingerprint differs from the frozen one "
             "(stop rule S1)"),
            (code.get("prespec_sha256") == frozen.prespec_sha256
             and icode.get("prespec_sha256") == frozen.prespec_sha256,
             "names another pre-specification (stop rule S1)"),
            (code.get("code_manifest_sha256") == frozen.manifest_sha256
             and icode.get("code_manifest_sha256") == frozen.manifest_sha256,
             "made by other code: code.code_manifest_sha256 differs from the frozen code manifest's (stop rule S1)"),
            (frozen.made_up or H.plain(sched) == PLAN_SCHEDULE,
             "not the plan's schedule (100,000 outer steps in chunks of 2,500, refreshes after 10,000 ... 95,000)"),
            (_schedule_ok(sched), "its schedule names no final or 50% snapshot")])
    final, half = int(sched["steps"]), half_step(sched)
    hashes = rec.get("hashes") or {}
    refreshes = [r for r in (rec.get("refreshes") or []) if isinstance(r, dict)]
    fields = dict(record="read", timings="read" if rec.get("timings") else "not recorded: no timings",
                  environment="read" if (rec.get("environment") or {}).get("import_record") else
                  "not recorded: no import record (S9)")
    for part in ("initial", "key_carry", "params_10000", "final"):
        fields[f"hashes.{part}"] = "read" if hashes.get(part) else (
            "not recorded: the run stopped before outer step "
            f"{sched.get('s10_step')}" if part == "params_10000" else f"not recorded: hashes.{part} is empty")
    if setup in UNIFLORAL_SETUPS:
        fields["refreshes"] = fields["design_effect"] = "not recorded: Unifloral's own step has no band"
    elif setup == "sham_alone":
        fields["refreshes"] = fields["design_effect"] = "not recorded: the sham alone never refreshes (choice 19)"
    else:
        fields["refreshes"] = "read" if refreshes else "not recorded: the run stopped before its first refresh"
        fields["design_effect"] = DESIGN_EFFECT_WHY
    env = rec.get("environment") or {}
    aux = dict(record=rel(path), record_sha256=sha, run_id=rid, stage=stage, steps=dict(final=final, half=half),
               timings=rec.get("timings") or {}, hashes=hashes, refreshes=refreshes, readouts=rec.get("readouts") or {},
               validity=rec.get("validity") or {},
               machine=dict(host=env.get("host"), device=ident.get("device"), taskset_cores=env.get("taskset_cores")),
               data={k: data.get(k) for k in ("npz_sha256", "manifest_sha256", "rows", "train_set", "train_sha256",
                                              "bank_sha256", "next_action_substituted")},
               environment=env_signature(rec), schedule=sched, stop=rec.get("stop"), scores=dict(final=None, half=None),
               coverage=None, pointer=None, s7=read_s7(folder, rec, frozen), fields=fields)
    fields["s7"] = "read" if aux["s7"] else ("not recorded: S7 runs on seed 0's final actors of alone, standard and IW"
                                             if not (seed == 0 and setup in S7_SETUPS) else
                                             "not recorded: the driver's S7 lane has not written it")
    source = dict(record=sha, scores=None)
    outcome = rec.get("outcome")
    if outcome == "stopped":
        stop = dict(rec.get("stop") or {})
        accepted = rid in accepted_s5
        kind = stop_kind(setup, stop, accepted)
        stop.update(kind=kind, accepted_by_operator=bool(accepted and kind == "outcome"
                                                         and setup not in UNIFLORAL_SETUPS))
        aux["stop"] = stop
        why = f"not recorded: the run stopped ({stop.get('rule')} at outer step {stop.get('step')})"
        fields.update(scores_final=why, scores_half=why, coverage=why, dose_change=why)
        run = dict(state="stopped", finite=False, J={}, points={}, events={}, starts={}, coverage=None, pointer=None,
                   stop=stop, source=source)
        return dict(kind=kind, run=run, aux=aux)
    if outcome != "complete":
        raise Excluded(f"outcome is {outcome!r}, not 'complete' or 'stopped'")
    _check([((rec.get("timings") or {}).get("outer_steps") == final,
             "a complete run whose outer steps differ from its schedule's"),
            ((rec.get("validity") or {}).get("all_steps_valid") is True, "a complete run not valid at every step")])
    for name, step in (("final", final), ("half", half)):
        aux["scores"][name] = read_scores(folder, rec, step, learner, frozen)
        fields[f"scores_{name}"] = "read" if aux["scores"][name] else (
            f"not recorded: lanes/scores-step{step}.npz does not exist (the driver's scoring lane has not written it)")
    coverage, why = read_coverage(folder, rec, sha, frozen)
    aux["coverage"] = coverage
    fields["coverage"] = "read" if coverage else f"not recorded: {why}"
    if setup == "iw":
        pointer, why = dose_change_pointer(rec, learner, sched)
        aux["pointer"] = pointer
        fields["dose_change"] = "read" if pointer else f"not recorded: {why}"
    else:
        fields["dose_change"] = "not recorded: Claim 2's pointer is read from the IW setups only"
    final_scores = aux["scores"]["final"]
    if final_scores is None:
        return dict(kind="unscored", run=None, aux=aux)
    banks = final_scores["readouts"][SCORED]
    pointer = aux["pointer"]
    run = dict(state="complete", finite=final_scores["finite"], J={b: v["J_mean"] for b, v in banks.items()},
               points={b: v["points_mean"] for b, v in banks.items()},
               events=dict(f1={b: v["f1_count"] for b, v in banks.items()},
                           f2={b: v["f2_count"] for b, v in banks.items()}),
               starts={b: v["n"] for b, v in banks.items()},
               coverage=None if coverage is None else dict(own=coverage["own"], standard=coverage["standard"]),
               pointer=None if not pointer else dict(change=pointer["change"], extra=pointer["extra"]),
               stop=None, source=dict(source, scores=final_scores["npz_sha256"]))
    return dict(kind="outcome", run=run, aux=aux)


def collect(root, frozen, qualities=C.QUALITIES, accepted_s5=frozenset()):
    """Every run folder under root/stage2 and root/stage3 for the build's quality labels (module docstring) ->
    dict(runs={quality: {(learner, setup): {seed: run}}}, aux={quality: {(learner, setup, seed): aux}}, halts,
    excluded, unscored, unscored_aux={quality: {(learner, setup, seed): aux}} (the unscored runs' aux, for the totals
    and S7's stopped scopes), listed (every run read, with its fields), s7_stage1 (stage 1's S7 results))."""
    root = Path(root)
    runs = {q: {} for q in qualities}
    aux = {q: {} for q in qualities}
    unscored_aux = {q: {} for q in qualities}
    halts, excluded, unscored, listed, s7_stage1 = [], [], [], [], []
    for q in qualities:
        label = quality_label(q, frozen.made_up)
        for stage in STAGES:
            qdir = root / stage / label
            if not qdir.is_dir():
                continue
            for sdir in sorted(p for p in qdir.iterdir() if p.is_dir()):
                m = SEED_DIR.match(sdir.name)
                for rdir in sorted(p for p in sdir.iterdir() if p.is_dir()):
                    learner, _, setup = rdir.name.partition("-")
                    seed = int(m.group(1)) if m else None
                    if learner not in C.LEARNERS or setup not in STAGE_SETUPS[stage] or \
                            seed not in SETUP_SEEDS.get(setup, ()):
                        excluded.append(dict(path=rel(rdir), reason=f"not a learner-setup folder of this study's "
                                                                    f"{stage}"))
                        continue
                    try:
                        got = read_run(rdir, q, learner, setup, seed, frozen, stage=stage, accepted_s5=accepted_s5)
                    except Excluded as error:
                        excluded.append(dict(path=rel(rdir), reason=str(error)))
                        continue
                    a = got["aux"]
                    key = dict(quality=q, learner=learner, setup=setup, seed=seed, record=a["record"],
                               record_sha256=a["record_sha256"])
                    listed.append(dict(key, run_id=a["run_id"], kind=got["kind"], fields=dict(a["fields"])))
                    if got["kind"] == "unscored":
                        unscored.append(key)
                        unscored_aux[q][(learner, setup, seed)] = a
                        continue
                    aux[q][(learner, setup, seed)] = a
                    if got["kind"] == "halt":
                        halts.append(dict(key, run_id=a["run_id"], stop=got["run"]["stop"]))
                        continue
                    runs[q].setdefault((learner, setup), {})[seed] = got["run"]
        for rdir in sorted({p.parent.parent.parent
                            for p in (root / "stage1" / label).glob("seed*/*/lanes/s7-step*/s7.json")}):
            try:
                rec = H.read_json(rdir / "record.json")
                bad = None if (rec.get("data") or {}).get("made_up") is frozen.made_up else "another made-up mark"
            except (H.HandoffError, OSError, ValueError) as error:
                rec, bad = {}, f"record.json: {error}"
            for s in (read_s7(rdir, rec, frozen) if bad is None else
                      [dict(path=rel(rdir), passed=False, why=bad)]):
                s7_stage1.append(dict(s, run_id=rec.get("run_id")))
    return dict(runs=runs, aux=aux, halts=halts, excluded=excluded, unscored=unscored, unscored_aux=unscored_aux,
                listed=listed, s7_stage1=s7_stage1)


# ------------------------------------------------------------------------------------------------ the driver's files

def _moved_stop_verifies(path):
    """A stop report moved aside by --resume-after-stop (STOP.json.<time>) against its moved sidecar
    (STOP.json.sha256.<time>), stages.command_run's renaming."""
    stamp = path.name[len("STOP.json."):]
    side = path.parent / f"STOP.json.sha256.{stamp}"
    words = side.read_text(encoding="utf8").split() if side.is_file() else []
    return bool(words) and words[0] == sha256_file(path)


def read_driver(root, frozen):
    """The stage driver's files under <root>/driver (stages.py): status.json (its mode must be this build's: "real",
    or "dry-run" for a made-up build; otherwise nothing of the driver's is read), state.json (decisions, stop-rule
    checks, Claim 0 failures, the S9 reference), STOP.json (an active stop) and STOP.json.<time> (stops a resumed driver
    moved aside), and driver.log's accepted S5 lines (stages.Driver._card_done; state.json "accepted_s5" too if a later
    driver writes it), and state.json "s7_stopped" (version 3: the scopes a failed S7 check stopped; absent in an
    older state, read as none). Returns a dict; problems are listed, never raised."""
    d = Path(root) / "driver"
    want = "dry-run" if frozen.made_up else "real"
    out = dict(path=rel(d), read=False, mode=None, state_of_driver=None, phase=None, updated_utc=None, files={},
               problems=[], checks={}, decisions={}, claim0_failures=[], s9_reference=None, stop=None,
               resumed_stops=[], accepted_s5=[], accepted_s5_source=None, s7_stopped=[], why=None)
    if not d.is_dir():
        out["why"] = "the driver's folder does not exist"
        return out
    try:
        status = H.read_json(d / "status.json")
    except (H.HandoffError, OSError, ValueError) as error:
        out["why"] = f"status.json does not verify ({error}), so its files cannot be told from another mode's"
        return out
    mode = status.get("mode") if isinstance(status, dict) else None
    if not isinstance(status, dict) or status.get("schema") != STATUS_SCHEMA or mode != want:
        out["why"] = f"status.json is not a {want} driver's status (mode {mode!r}): the driver's files are not read"
        out["problems"].append(out["why"])
        return out
    out.update(read=True, mode=mode, state_of_driver=status.get("state"), phase=status.get("phase"),
               updated_utc=status.get("updated_utc"))
    out["files"]["status.json"] = sha256_file(d / "status.json")
    accepted, sources = set(), []
    try:
        state = H.read_json(d / "state.json")
        if not isinstance(state, dict) or state.get("schema") != STATE_SCHEMA:
            raise ValueError(f"schema is not {STATE_SCHEMA}")
        out["files"]["state.json"] = sha256_file(d / "state.json")
        out.update(checks=state.get("checks") or {}, decisions=state.get("decisions") or {},
                   claim0_failures=state.get("claim0_failures") or [], s9_reference=state.get("s9_reference"))
        if isinstance(state.get("accepted_s5"), list):
            accepted.update(str(x) for x in state["accepted_s5"])
            sources.append("state.json accepted_s5")
        s7 = state.get("s7_stopped")
        if s7 is not None:
            if not isinstance(s7, list) or not all(isinstance(e, dict) for e in s7):
                raise ValueError("s7_stopped is not a list of entries")
            out["s7_stopped"] = [dict(e) for e in s7]
    except (H.HandoffError, OSError, ValueError) as error:
        out["problems"].append(f"state.json: {error}")
    if (d / "STOP.json").exists():
        try:
            body = H.read_json(d / "STOP.json")
            if body.get("schema") != STOP_SCHEMA or body.get("mode") != want:
                raise ValueError(f"not a {want} driver's stop report")
            out["stop"] = dict(rule=body.get("rule"), reason=body.get("reason"), at_utc=body.get("at_utc"),
                               phase=body.get("phase"), run=body.get("run"), sha256=sha256_file(d / "STOP.json"))
        except (H.HandoffError, OSError, ValueError, AttributeError) as error:
            out["problems"].append(f"STOP.json: {error}")
            out["stop"] = dict(rule="unreadable", reason=str(error))
    for p in sorted(d.glob("STOP.json.*")):
        if p.name.startswith("STOP.json.sha256"):
            continue
        ok = _moved_stop_verifies(p)
        body = {}
        if ok:
            try:
                body = json.loads(p.read_text(encoding="utf8"))
            except ValueError:
                ok = False
        out["resumed_stops"].append(dict(file=p.name, sha256=sha256_file(p), verified=ok, rule=body.get("rule"),
                                         reason=body.get("reason"), at_utc=body.get("at_utc"), run=body.get("run"),
                                         note="moved aside by --resume-after-stop"))
    log = d / "driver.log"
    if log.is_file():
        out["files"]["driver.log"] = sha256_file(log)
        for line in log.read_text(encoding="utf8", errors="replace").splitlines():
            m = ACCEPTED_S5_LINE.match(line)
            if m:
                accepted.add(m.group(1))
        sources.append("driver.log (the driver writes no sidecar for it)")
    out["accepted_s5"] = sorted(accepted)
    out["accepted_s5_source"] = sources
    return out


# ------------------------------------------------------------------------------------------------ anchors

def read_start_bank_anchors(path=START_BANKS_JSON):
    """The weighting banks' anchors {law: dict(random, top)} from the verified copy of start_banks.json (its sidecar
    and evaluate.py's pinned SHA-256; fixed controllers only), and the recording controllers' no-shift points."""
    body = H.read_json(path)
    if sha256_file(path) != START_BANKS_JSON_SHA256:
        raise ValueError(f"{path} differs from its pinned SHA-256 (stop rule S11)")
    anchors = {law: dict(random=float(body["anchors"][law]["anchor"]["random"]),
                         top=float(body["anchors"][law]["anchor"]["top"])) for law in C.WEIGHTING_BANKS}
    recording = {q: float(body["anchors"]["none"]["controllers"][q]["points"]) for q in C.QUALITIES}
    return dict(anchors=anchors, recording_points=recording, path=rel(path), sha256=START_BANKS_JSON_SHA256)


def check_score_anchors(collected, anchors):
    """Every scores file's weighting-bank anchors equal start_banks.json's (the claims' bar uses them). Returns
    (mismatches, scores files checked)."""
    bad, seen = [], 0
    for q, by in collected["aux"].items():
        for key, a in by.items():
            for which, sc in a["scores"].items():
                if sc is None:
                    continue
                seen += 1
                for law in C.WEIGHTING_BANKS:
                    got = sc["anchors"].get(law) or {}
                    if (got.get("bottom"), got.get("top")) != (anchors[law]["random"], anchors[law]["top"]):
                        bad.append(dict(quality=q, run=list(key), snapshot=which, step=sc["step"], bank=law))
    return bad, seen


# ------------------------------------------------------------------------------------------------ integrity

def _worst(*statuses):
    known = [s for s in statuses if s in STATUS_ORDER]
    return max(known, key=STATUS_ORDER.index) if known else "not recorded"


def choice19_of(drv):
    """Choice 19 per learner: the driver's own decision, read from its stage-1 S10 checks (S10/<learner>/<setup>
    carries readout), else this module's restatement of stages.CHOICE19."""
    out, src = dict(CHOICE19), {}
    for k, v in (drv.get("checks") or {}).items():
        parts = k.split("/")
        if len(parts) == 3 and parts[0] == "S10" and parts[1] in out and isinstance(v, dict) and "readout" in v:
            out[parts[1]] = bool(v["readout"])
            src[parts[1]] = "driver"
    return out, {L: src.get(L, "export.CHOICE19") for L in out}


def s7_covered(quality, learner, setup, scoped):
    """Whether a failed S7 check of this learner's setup in this quality is one the driver recorded as a stopped scope
    (version 3, state.json s7_stopped): an entry stopping the learner covers every setup, one stopping the IW setup
    covers IW and its repeat (claims.S7_SCOPE)."""
    return any(e.get("quality") == quality and e.get("learner") == learner
               and (e.get("scope") == "learner" or (e.get("scope") == "iw" and C.S7_SCOPE.get(setup) == "iw"))
               for e in scoped or [])


def _s7_check_key(key):
    """(quality, learner, setup) of a driver S7 check: S7/<learner>/<setup> (stage 1, run on medium-expert) or
    S7/<quality>/seed0/<learner>/<setup>; None for any other shape."""
    parts = key.split("/")
    if len(parts) == 3:
        return "medium-expert", parts[1], parts[2]
    if len(parts) == 5 and parts[2] == "seed0":
        return parts[1], parts[3], parts[4]
    return None


def _s7_run_key(run_id):
    """(quality, learner, setup) of a run id "<stage>/<label>/seed<k>/<learner>-<setup>" (a made-up label names its
    quality after "made-up-"); None for any other shape."""
    parts = str(run_id or "").split("/")
    if len(parts) != 4:
        return None
    learner, _, setup = parts[3].partition("-")
    label = parts[1][len("made-up-"):] if parts[1].startswith("made-up-") else parts[1]
    return label, learner, setup


def driver_rule(rule, drv):
    """The driver's own status of a stop rule from its state.json and stop reports (module docstring), or None. S7
    (version 3): a failed check the driver recorded as a stopped scope (s7_stopped) is an outcome, any other failed
    check fails the rule."""
    checks = drv.get("checks") or {}
    items = {k: v for k, v in checks.items() if isinstance(v, dict) and k.split("/")[0] == rule}
    entry = None
    if rule == "S7" and (items or drv.get("s7_stopped")):
        scoped = drv.get("s7_stopped") or []
        failed = sorted(k for k, v in items.items() if not v.get("passed"))
        loose = [k for k in failed if not ((key := _s7_check_key(k)) and s7_covered(*key, scoped))]
        entry = dict(status="failed" if loose else ("outcome" if scoped or failed else "passed"), checks=sorted(items),
                     failed=failed, failed_without_a_stopped_scope=loose, s7_stopped=scoped,
                     note="version 3: a failed S7 check stops that learner or its IW setup in that quality (an "
                          "outcome); S7's limit is unchanged")
    elif rule == "S9":
        ref = drv.get("s9_reference") or {}
        if items or ref:
            entry = dict(status="failed" if items else "passed", failed=sorted(items),
                         reference_source=ref.get("source"))
    elif rule == "S5":
        c0, acc = drv.get("claim0_failures") or [], drv.get("accepted_s5") or []
        if c0 or acc:
            entry = dict(status="outcome", claim0_failures=c0, accepted_by_operator=acc,
                         accepted_source=drv.get("accepted_s5_source"))
    elif items:
        failed = sorted(k for k, v in items.items() if not v.get("passed"))
        readout = sorted(k for k, v in items.items()
                         if v.get("passed") and v.get("readout") and v.get("equal") is False)
        entry = dict(status="failed" if failed else ("readout" if readout else "passed"), checks=sorted(items),
                     failed=failed, readouts=readout)
    stop = drv.get("stop")
    if stop and stop.get("rule") == rule:
        entry = dict(entry or {}, status="stopped", stop=stop)
    resumed = [s for s in drv.get("resumed_stops") or [] if s.get("rule") == rule]
    if resumed:
        entry = dict(entry or dict(status="not recorded"), resumed_stops=resumed)
    return entry


def _init_hash(a):
    h = ((a.get("hashes") or {}).get("initial") or {}).get("agent_state")
    return h.get("all") if isinstance(h, dict) else h


def _key_carries(a):
    """A run's key carries; a stopped run's only before its stop (stages.check_s4: a step that is not valid keeps the
    old carry, key included)."""
    kc = (a.get("hashes") or {}).get("key_carry") or {}
    lim = (a.get("stop") or {}).get("step") if (a.get("stop") or {}).get("rule") else None
    return {str(t): v for t, v in kc.items() if lim is None or int(t) < int(lim)}


def own_checks(collected, frozen, drv, anchors_checked, choice19):
    """This module's own checks of the records (module docstring): S1, S3, S4, S5, S7, S9, S10 and S11."""
    anchors_bad, scores_seen = anchors_checked
    out = {}
    code_excluded = [e for e in collected["excluded"] if "stop rule S1" in e["reason"]]
    used = sum(len(by) for by in collected["aux"].values())
    out["S1"] = dict(status="failed" if code_excluded else ("passed" if used else "not recorded"),
                     details=dict(records_used=used, excluded_for_other_code=code_excluded,
                                  code_manifest_sha256=frozen.manifest_sha256))
    s3_bad, s3_seen, s4_bad, s4_seen, s10, s9_bad, s9_seen = [], 0, [], 0, [], [], 0
    ref = (drv.get("s9_reference") or {}).get("signature") if isinstance(drv.get("s9_reference"), dict) else None
    for q, by in collected["aux"].items():
        for (L, setup, seed), a in by.items():
            if setup == "iw":
                for r in a["refreshes"]:
                    s3_seen += 1
                    if r.get("s3_reproduced") is not True:
                        s3_bad.append(dict(quality=q, learner=L, seed=seed, step=r.get("step")))
            if ref:
                s9_seen += 1
                diff = sorted(k for k, v in ref.items() if a["environment"].get(k) != v)
                if diff:
                    s9_bad.append(dict(quality=q, run_id=a["run_id"], differs=diff))
        for L in C.LEARNERS:
            for seed in C.SEEDS:
                runs = {s: by.get((L, s, seed)) for s in ("alone", "standard", "iw", "room", "sham_alone")}
                runs = {s: a for s, a in runs.items() if a}
                init = {s: _init_hash(a) for s, a in runs.items() if _init_hash(a)}
                if len(init) >= 2:
                    s4_seen += 1
                    if len(set(init.values())) != 1:
                        s4_bad.append(dict(quality=q, learner=L, seed=seed, what="initial parameters", hashes=init))
                train = {s: a["data"].get("train_sha256") for s, a in runs.items() if s != "room"}
                train = {s: h for s, h in train.items() if h}
                if len(train) >= 2:
                    s4_seen += 1
                    if len(set(train.values())) != 1:
                        s4_bad.append(dict(quality=q, learner=L, seed=seed, what="training rows", hashes=train))
                keys = {s: _key_carries(runs[s]) for s in ("alone", "standard", "iw") if s in runs}
                for t in sorted({t for k in keys.values() for t in k}, key=int):
                    vals = {s: k[t] for s, k in keys.items() if t in k}
                    if len(vals) >= 2:
                        s4_seen += 1
                        if len(set(vals.values())) != 1:
                            s4_bad.append(dict(quality=q, learner=L, seed=seed, what=f"key carry {t}", hashes=vals))
                alone = (runs.get("alone") or {}).get("hashes", {}).get("params_10000")
                for s in ("standard", "iw", "sham_alone"):
                    other = (runs.get(s) or {}).get("hashes", {}).get("params_10000")
                    if alone and other:
                        equal = (alone.get("agent_state"), alone.get("key")) == (other.get("agent_state"),
                                                                                  other.get("key"))
                        s10.append(dict(quality=q, learner=L, seed=seed, setup=s, step=alone.get("step"),
                                        equal=bool(equal), choice19=bool(choice19.get(L))))
    out["S3"] = dict(status="failed" if s3_bad else ("passed" if s3_seen else "not recorded"),
                     details=dict(refreshes_checked=s3_seen, not_reproduced=s3_bad))
    out["S4"] = dict(status="failed" if s4_bad else ("passed" if s4_seen else "not recorded"),
                     details=dict(comparisons=s4_seen, differ=s4_bad))
    differ = [x for x in s10 if not x["equal"]]
    s10_status = ("not recorded" if not s10 else "failed" if any(not x["choice19"] for x in differ) else
                  "readout" if differ else "passed")
    out["S10"] = dict(status=s10_status, details=dict(
        compared=s10, note="stage 2's learner states at the S10 step against the learner alone's (the rule is stage "
                           "1's, in the driver's checks): a difference is a readout for a learner under choice 19 "
                           "(test (a2) matched only to rounding on the card) and is marked failed for the others"))
    out["S9"] = dict(status=("not recorded" if not s9_seen else "failed" if s9_bad else "passed"),
                     details=dict(records_checked=s9_seen, differ=s9_bad,
                                  reference_source=(drv.get("s9_reference") or {}).get("source")))
    s7 = [dict(s, run_id=a["run_id"]) for by in collected["aux"].values() for a in by.values() for s in a["s7"]]
    s7 += collected.get("s7_stage1") or []
    # version 3: a verified s7.json, bound to its record, that says the check failed, of a scope the driver recorded
    # as stopped, is an outcome; any other failure (one not recorded, or a file that does not verify) fails S7
    scoped = drv.get("s7_stopped") or []
    for s in s7:
        key = _s7_run_key(s.get("run_id"))
        s["stopped_scope"] = bool(not s.get("passed") and s.get("bound_to_record") is True
                                  and s.get("s7_passed") is not True and key is not None and s7_covered(*key, scoped))
    bad = [s for s in s7 if not s.get("passed") and not s["stopped_scope"]]
    out["S7"] = dict(status=("not recorded" if not s7 else "failed" if bad else
                             "outcome" if any(s["stopped_scope"] for s in s7) else "passed"),
                     details=dict(results=s7, s7_stopped=scoped))
    outcome_stops = [dict(quality=q, learner=L, setup=s, seed=sd, stop=r["stop"])
                     for q, by in collected["runs"].items() for (L, s), seeds in by.items() for sd, r in seeds.items()
                     if r["state"] == "stopped"]
    if outcome_stops:
        out["S5"] = dict(status="outcome", details=dict(
            recorded_as_outcomes=outcome_stops,
            note="S5 on Unifloral's unchanged step is that learner's Claim 0 failure; on a BCA setup accepted by the "
                 "operator (not traced to a defect) it is that setup's outcome (plan section 6.1)"))
    out["S11"] = dict(status="failed" if anchors_bad else ("passed" if scores_seen else "not recorded"),
                      details=dict(scores_checked=scores_seen, scores_with_other_anchors=anchors_bad,
                                   what="the scores files' weighting-bank anchors against start_banks.json"))
    return out


def integrity(collected, frozen, drv, anchors_checked):
    """The stop rules S0-S11 (plan section 6.1): per rule the driver's own status (driver_rule), this module's checks
    of the records (own_checks) and any recorded halt of a run read here; the status is the worst of the three
    (STATUS_ORDER), with the reason when nothing records it. unknown_stops lists halts and driver stops under a rule
    outside S0-S11 (the driver's "error", "lane", "operator") and moved stop reports whose sidecar does not verify (their
    rule is not read)."""
    choice19, choice19_source = choice19_of(drv)
    own = own_checks(collected, frozen, drv, anchors_checked, choice19)
    out = {}
    for r in C.STOP_RULES:
        d, e = driver_rule(r, drv), own.get(r)
        halts = [h for h in collected["halts"] if (h["stop"] or {}).get("rule") == r]
        status = _worst((d or {}).get("status", "not recorded"), (e or {}).get("status", "not recorded"),
                        "stopped" if halts else "not recorded")
        out[r] = dict(status=status, driver=d, export=e, halts=halts)
        if status == "not recorded":
            out[r]["why"] = NOT_RECORDED_WHY.get(r, "neither the driver's files nor a run record read here holds it "
                                                    "yet")
    known = set(C.STOP_RULES)
    out["unknown_stops"] = [h for h in collected["halts"] if (h["stop"] or {}).get("rule") not in known]
    if drv.get("stop") and drv["stop"].get("rule") not in known:
        out["unknown_stops"].append(dict(driver_stop=drv["stop"]))
    out["unknown_stops"] += [dict(resumed_stop=s, why="a moved stop report whose sidecar does not verify (its rule is "
                                                      "not read)" if not s["verified"] else "a rule outside S0-S11")
                             for s in drv.get("resumed_stops") or [] if not s["verified"] or s["rule"] not in known]
    out["driver"] = dict(read=drv["read"], mode=drv["mode"], state=drv["state_of_driver"], phase=drv["phase"],
                         why=drv["why"], problems=drv["problems"], files=drv["files"], choice19=choice19,
                         choice19_source=choice19_source, accepted_s5=drv["accepted_s5"],
                         stop=drv.get("stop"), resumed_stops=drv.get("resumed_stops") or [],
                         s7_stopped=drv.get("s7_stopped") or [])
    return out


# ------------------------------------------------------------------------------------------------ expanded view

def _view_runs(runs, aux, which="final", readout=SCORED):
    """A claims-input copy of runs with J (and finiteness) from another snapshot ("half") or readout (CQL's
    tanh-of-mean). A seed without that file is left out."""
    out = {}
    for (L, setup), seeds in runs.items():
        for sd, r in seeds.items():
            if r["state"] != "complete":
                out.setdefault((L, setup), {})[sd] = r
                continue
            sc = (aux.get((L, setup, sd)) or {}).get("scores", {}).get(which)
            if sc is None or readout not in sc["readouts"]:
                continue
            banks = sc["readouts"][readout]
            fin = all(b["finite"] for b in banks.values())
            out.setdefault((L, setup), {})[sd] = dict(r, finite=fin, J={b: v["J_mean"] for b, v in banks.items()},
                                                      points={b: v["points_mean"] for b, v in banks.items()},
                                                      source=dict(r["source"], scores=sc["npz_sha256"]))
    return out


def _paired_se(aux, L, a, b, bank, seeds=C.SEEDS):
    """Per seed, the paired difference a - b over the bank's starts: mean and standard error std / sqrt(n) (ddof 0),
    as evaluate.paired computes it (plan section 5.4, paired standard errors over the 1,000 starts)."""
    out = []
    for sd in seeds:
        sa = ((aux.get((L, a, sd)) or {}).get("scores") or {}).get("final")
        sb = ((aux.get((L, b, sd)) or {}).get("scores") or {}).get("final")
        if sa is None or sb is None:
            out.append(dict(seed=sd, mean=None, se=None, n=None))
            continue
        ba, bb = sa["readouts"][SCORED][bank], sb["readouts"][SCORED][bank]
        if not np.array_equal(ba["start"], bb["start"]):
            out.append(dict(seed=sd, mean=None, se=None, n=None, why="the setups were scored on other starts"))
            continue
        d = ba["J_per_start"] - bb["J_per_start"]
        out.append(dict(seed=sd, mean=float(d.mean()), se=float(d.std() / math.sqrt(len(d))), n=int(len(d))))
    return out


def _per_seed(by_seed, seeds, fn):
    vals = [fn(by_seed[sd]) if sd in by_seed else None for sd in seeds]
    good = [v for v in vals if v is not None and isinstance(v, (int, float)) and math.isfinite(v)]
    return dict(per_seed=vals, mean=float(np.mean(good)) if len(good) == len(seeds) and good else None)


def _setup_block(q, L, setup, runs, aux):
    seeds = SETUP_SEEDS[setup]
    by_seed = runs.get((L, setup), {})
    a = {sd: aux.get((L, setup, sd)) for sd in seeds if (L, setup, sd) in aux}
    if not by_seed and not a:
        return None
    records = []
    for sd in seeds:
        x = a.get(sd)
        r = by_seed.get(sd)
        if x is None:
            records.append(dict(seed=sd, state="missing"))
            continue
        fin, half = x["scores"].get("final"), x["scores"].get("half")
        records.append(dict(seed=sd, run_id=x["run_id"], state=(r or {}).get("state", "halted"), record=x["record"],
                            record_sha256=x["record_sha256"], scores_sha256=None if fin is None else fin["npz_sha256"],
                            scores_half_sha256=None if half is None else half["npz_sha256"],
                            coverage_sha256=None if x["coverage"] is None else x["coverage"]["sha256"],
                            machine=x["machine"], stop=x["stop"], fields=x["fields"]))
    finals = {sd: x["scores"]["final"] for sd, x in a.items() if x["scores"].get("final")}
    halves = {sd: x["scores"]["half"] for sd, x in a.items() if x["scores"].get("half")}
    half_step_ = next((x["steps"]["half"] for x in a.values()), None)
    metrics = {}
    for bank in C.BANKS:
        metrics[bank] = {f: _per_seed(finals, seeds, lambda s, f=f, b=bank: s["readouts"][SCORED][b][f])
                         for f in SUMMARY_FIELDS}
    half = {bank: dict(J=_per_seed(halves, seeds, lambda s, b=bank: s["readouts"][SCORED][b]["J_mean"]),
                       points=_per_seed(halves, seeds, lambda s, b=bank: s["readouts"][SCORED][b]["points_mean"]))
            for bank in ("none", "mild", "strong")}
    block = dict(name=C.SETUP_NAMES[setup], seeds=list(seeds), records=records, metrics=metrics,
                 half=dict(step=half_step_, **half),
                 compute={k: _per_seed(a, seeds, lambda x, k=k: (x["timings"] or {}).get(k))
                          for k in ("wall_seconds", "card_seconds", "processor_seconds", "compile_seconds",
                                    "steady_ms_per_outer_step")})
    if L == "cql":
        block["tanh_mean"] = {bank: _per_seed(finals, seeds, lambda s, b=bank: s["readouts"]["tanh_mean"][b]["J_mean"])
                              for bank in C.BANKS}
    if setup not in UNIFLORAL_SETUPS:
        def last(x, key):
            return x["refreshes"][-1].get(key) if x["refreshes"] else None

        def certified(x):
            return float(sum(1 for r in x["refreshes"] if r.get("certified")))

        def readout(x, name):
            return ((x.get("readouts") or {}).get(name) or {}).get("dose_change") if (x.get("readouts") or {}) \
                else None

        block["band"] = dict(
            value="actor weight" if L == "iql" else "dose",
            certified_refreshes=_per_seed(a, seeds, certified),
            refreshes=_per_seed(a, seeds, lambda x: float(len(x["refreshes"]))),
            refused_rows_last=_per_seed(a, seeds, lambda x: last(x, "refused_rows")),
            iw_refused_rows_last=_per_seed(a, seeds, lambda x: last(x, "iw_refused_rows")),
            mean_dose_last=_per_seed(a, seeds, lambda x: last(x, "mean_dose")),
            threshold_last=_per_seed(a, seeds, lambda x: last(x, "threshold")),
            train_rows_last=[last(a[sd], "train_rows") if sd in a else None for sd in seeds],
            s3_recomputed_equals_refresh_scores=[[r.get("s3_recomputed_equals_refresh_scores")
                                                  for r in a[sd]["refreshes"]] if sd in a else None for sd in seeds],
            dose_change=[((a.get(sd) or {}).get("readouts") or {}).get("dose_change") for sd in seeds],
            estimated_weights_dose_change=[readout(a[sd], "estimated_weights") if sd in a else None for sd in seeds],
            shuffled_weights_dose_change=[readout(a[sd], "shuffled_weights") if sd in a else None for sd in seeds],
            pointer=[(a.get(sd) or {}).get("pointer") for sd in seeds])
        block["coverage"] = [None if (a.get(sd) or {}).get("coverage") is None else
                             dict(seed=sd, step=a[sd]["coverage"]["step"], own_arm=a[sd]["coverage"]["own_arm"],
                                  own=a[sd]["coverage"]["own"], standard=a[sd]["coverage"]["standard"],
                                  standard_threshold=a[sd]["coverage"]["standard_threshold"],
                                  pairs=a[sd]["coverage"]["pairs"],
                                  pairs_not_recorded=a[sd]["coverage"]["pairs_not_recorded"],
                                  other=a[sd]["coverage"]["other"]) for sd in seeds]
    return block


def _pointer(d):
    """A paired comparison shown as a pointer: its summary and per-seed values with their sources."""
    return dict(label=C.POINTER_LABEL, comparison=d["comparison"], bank=d["bank"], summary=d["summary"],
                per_seed=d["per_seed"], missing=d["missing"])


def prepared_substituted(q, seed, data_root):
    """ReBRAC's share of training rows with a substituted next command, overall and in the highest-current tenth, and
    of bank rows, from a prepared file's manifest (plan sections 3.8 and 5.4)."""
    path = Path(data_root) / q / f"seed{seed}" / "prepared.npz"
    man = H.read_manifest(path)
    sub = man["meta"]["next_action"]["substituted"]
    return dict(seed=seed, train=sub["train"]["share"],
                train_highest_current_tenth=sub["train"]["highest_current_tenth"]["share"], bank=sub["bank"]["share"],
                source=rel(H.manifest_path(path)), manifest_sha256=sha256_file(H.manifest_path(path)))


def _rebrac_substituted(aux, q, data_root):
    """ReBRAC's substituted next commands per seed, from its run records (data.next_action_substituted, the prepared
    file's own shares, bound to the record), else from data_root's prepared manifest, else 'not recorded'."""
    out = []
    for sd in C.SEEDS:
        a = next((aux[k] for k in (("rebrac", s, sd) for s in ("alone", "standard", "iw")) if k in aux), None)
        sub = ((a or {}).get("data") or {}).get("next_action_substituted")
        try:
            out.append(dict(seed=sd, train=sub["train"]["share"],
                            train_highest_current_tenth=sub["train"]["highest_current_tenth"]["share"],
                            bank=sub["bank"]["share"], source="record.data.next_action_substituted",
                            record_sha256=a["record_sha256"]))
            continue
        except (KeyError, TypeError):
            pass
        try:
            if data_root is None:
                raise ValueError("no prepared-data folder given")
            out.append(prepared_substituted(q, sd, data_root))
        except (H.HandoffError, OSError, KeyError, ValueError, TypeError) as error:
            out.append(dict(seed=sd, why=f"not recorded: no ReBRAC record holds data.next_action_substituted and the "
                                         f"prepared manifest was not read ({error})"))
    return out


def _iql_identity(aux):
    """Whether IQL's Q and V (dual_q, dual_q_target, value) are bit-identical across its three runs (plan section
    5.4), from the records' hashes.final.parts. On the card the standard and IW runs (one compiled program) share them
    bit for bit and the alone run may drift at rounding level (unifloral_iql_bca.critic_value_parts), so the
    standard-IW pair is shown beside the three-way answer."""
    out = []
    for sd in C.SEEDS:
        recs = {s: aux.get(("iql", s, sd)) for s in C.CLAIM_SETUPS}
        if any(r is None for r in recs.values()):
            out.append(dict(seed=sd, identical=None, why="not recorded: not all three runs were read"))
            continue
        if any((r.get("stop") or {}).get("rule") for r in recs.values()):
            out.append(dict(seed=sd, identical=None, why="not recorded: a run stopped, so its final hashes are at its "
                                                         "stop"))
            continue
        parts = {s: ((r.get("hashes") or {}).get("final") or {}).get("parts") or {} for s, r in recs.items()}
        if not all(p.get(k) for p in parts.values() for k in IQL_SHARED_PARTS):
            out.append(dict(seed=sd, identical=None, why="not recorded: hashes.final.parts holds no dual_q, "
                                                         "dual_q_target and value"))
            continue
        each = {k: len({p[k] for p in parts.values()}) == 1 for k in IQL_SHARED_PARTS}
        out.append(dict(seed=sd, identical=all(each.values()), parts_identical=each,
                        standard_and_iw_identical=all(parts["standard"][k] == parts["iw"][k]
                                                      for k in IQL_SHARED_PARTS),
                        sources={s: r["record_sha256"] for s, r in recs.items()}))
    return out


def expanded_view(q, runs, aux, judged, data_root=DATA):
    """Plan section 9.1's expanded view (and section 5.4's readouts) for one quality."""
    half_runs = _view_runs(runs, aux, which="half")
    tanh_runs = _view_runs(runs, aux, readout="tanh_mean")
    out = {}
    for L in C.LEARNERS:
        setups = {s: b for s in SETUPS if (b := _setup_block(q, L, s, runs, aux)) is not None}
        pointers = dict(
            iw_minus_alone={b: _pointer(C.paired(runs, L, "iw", "alone", b)) for b in C.WEIGHTING_BANKS},
            mild={c: _pointer(C.paired(runs, L, a, b_, "mild"))
                  for c, (a, b_) in dict(standard_minus_alone=("standard", "alone"),
                                         iw_minus_standard=("iw", "standard")).items()},
            room_minus_alone={b: _pointer(C.paired(runs, L, "room", "alone", b)) for b in C.WEIGHTING_BANKS},
            half={c: {b: _pointer(C.paired(half_runs, L, a, b_, b)) for b in ("none", "strong")}
                  for c, (a, b_) in dict(standard_minus_alone=("standard", "alone"),
                                         iw_minus_standard=("iw", "standard")).items()},
            claim2_pointer=dict(judged["pointers"][L]["claim2"], definition=POINTER_DEFINITION),
            paired_se={c: {b: _paired_se(aux, L, a, b_, b) for b in ("none", "strong")}
                       for c, (a, b_) in dict(standard_minus_alone=("standard", "alone"),
                                              iw_minus_standard=("iw", "standard"),
                                              iw_minus_alone=("iw", "alone")).items()})
        sham, alone = runs.get((L, "sham_alone"), {}).get(0), runs.get((L, "alone"), {}).get(0)
        pointers["sham_minus_alone_seed0"] = dict(
            label=C.POINTER_LABEL, note="choice 19: the restated step with BCA present and the dose pinned at 1, "
                                        "against Unifloral's imported step, seed 0",
            values=None if sham is None or alone is None else
            {b: (sham["J"][b] - alone["J"][b]) if C.run_ok(sham) and C.run_ok(alone) else None
             for b in C.BANKS},
            sources=None if sham is None or alone is None else dict(sham=sham["source"], alone=alone["source"]))
        if L == "cql":
            pointers["tanh_mean"] = {c: {b: _pointer(C.paired(tanh_runs, L, a, b_, b)) for b in ("none", "strong")}
                                     for c, (a, b_) in dict(standard_minus_alone=("standard", "alone"),
                                                            iw_minus_standard=("iw", "standard")).items()}
        readouts = {}
        if L == "rebrac":
            readouts["substituted_next_commands"] = _rebrac_substituted(aux, q, data_root)
        if L == "iql":
            readouts["q_and_v_identical_across_runs"] = _iql_identity(aux)
        out[L] = dict(learner_name=C.LEARNER_NAMES[L], setups=setups, pointers=pointers, readouts=readouts)
    return out


# ------------------------------------------------------------------------------------------------ the reference row

def _require_verified(manifest, rels, dest):
    """{rel: SHA-256}: each copied file is listed as verified in the copy manifest, at the place data_copy's layout
    gives it, and still has that SHA-256 on disk. Restated from data_copy.require_verified with the copy root as a
    parameter (the real root is data_copy.DEST)."""
    by_rel = {e["rel"]: e for e in manifest.get("entries", [])}
    out = {}
    for r in rels:
        e = by_rel.get(r)
        if e is None or e.get("ok") is not True:
            raise DC.CopyError(f"S11: {r} is not a verified copy")
        if e.get("copy") != DC.copy_rel(r):
            raise DC.CopyError(f"S11: the manifest puts {r} at {e.get('copy')}, not {DC.copy_rel(r)}")
        if sha256_file(Path(dest) / DC.dest_rel(r)) != e["sha256"]:
            raise DC.CopyError(f"S11: the copy of {r} changed after it was copied and verified")
        out[r] = e["sha256"]
    return out


def _wt_record(path, made_up):
    """A weighting-test record: its sidecar checked, finished, and made-up exactly when this build is."""
    H.verify_sidecar(path)
    rec = json.loads(Path(path).read_text(encoding="utf8"))
    if rec.get("status") != "finished":
        raise ValueError(f"{path} is not finished")
    if bool(rec.get("made_up")) is not made_up:
        raise ValueError(f"{path}'s made-up mark does not match this build")
    return rec


def reference_row(permission, *, dest=None, qualities=C.QUALITIES):
    """The reference row "CORL-derived TD3+BC (weighting test)" (plan sections 5.1, 6.3 and 9.1): the weighting test's
    TD3+BC alone (plain_gap), its room run (plain_full), standard BCA (bca_unweighted), IW BCA (bca_exact_strong, per
    row) and the single-threshold arm, on its three start banks, from the verified copies of its evaluation records
    (data_copy: <quality>/evaluate/seed<k>/hosts.json and explore/<quality>/evaluate/seed<k>/bca.json). Each per-seed
    value carries its file's SHA-256. Paired differences use claims.summary; each is checked against the weighting
    test's own RESULTS.json (the same per-seed values bit for bit, the mean, lower bound and p-value within 1e-9), and
    RESULTS.json's input list must name exactly these files' SHA-256. It never enters a claim, a Holm family or a kill
    rule. permission: a verified gate.Approval (dest must be data_copy.DEST), or guard.MADE_UP with a made-up copy
    folder whose manifest and records say made_up true."""
    if permission is guard.MADE_UP:
        made_up = True
        if dest is None:
            raise gate.ApprovalError("refused: a made-up reference row reads a made-up copy folder (dest=...)")
    elif isinstance(permission, gate.Approval):
        gate.verify(permission)
        made_up = False
        if dest is not None and Path(dest).resolve() != DC.DEST.resolve():
            raise gate.ApprovalError("refused: the real reference row is read only from data_copy.DEST")
    else:
        raise gate.ApprovalError(f"refused: the reference row is read with a verified gate.Approval (or "
                                 f"guard.MADE_UP for tests), not {permission!r}")
    dest = Path(dest) if dest is not None else DC.DEST
    manifest = H.read_json(dest / "copy_manifest.json")
    if manifest.get("schema") != DC.SCHEMA or manifest.get("all_ok") is not True:
        raise DC.CopyError("S11: the copy manifest does not say every copy was verified")
    if bool(manifest.get("made_up")) is not made_up:
        raise gate.ApprovalError("refused: the copy manifest's made-up mark does not match this build")
    results_rel = DC.RESULTS
    results_sha = _require_verified(manifest, [results_rel], dest)[results_rel]
    results = json.loads((dest / DC.dest_rel(results_rel)).read_text(encoding="utf8"))
    if (results.get("kind") != "shift-iw-explore-results" or results.get("exploratory") is not True
            or bool(results.get("made_up")) is not made_up):
        raise ValueError("RESULTS.json is not the weighting test's exploratory results of this build's kind")
    inputs = results.get("inputs_sha256") or {}
    out, sources, spec = {}, {results_rel: results_sha}, set()
    for q in qualities:
        vals = {arm: {law: {w: [] for w in ("J", "points")} for law in REFERENCE_LAWS} for arm in REFERENCE_ARMS}
        for sd in C.SEEDS:
            files = dict(hosts=DC.hosts_eval_rel(q, sd), bca=DC.explore_eval_rel(q, sd))
            ver = _require_verified(manifest, [r for f in files.values() for r in (f, f + ".sha256")], dest)
            recs = {}
            for group, r in files.items():
                rec = _wt_record(dest / DC.dest_rel(r), made_up)
                want = ("shift-iw-evaluate", "hosts") if group == "hosts" else ("shift-iw-explore-evaluate", "bca")
                ok = ((rec.get("kind"), rec.get("group")) == want and rec.get("quality") == q
                      and rec.get("seed") == sd)
                if group == "bca":
                    ok = ok and (rec.get("label") == EXPLORE_LABEL and rec.get("exploratory") is True
                                 and not rec.get("smoke") and not rec.get("missing_arms"))
                if not ok:
                    raise ValueError(f"{r} is not the finished {group} evaluation of {q} seed {sd}")
                if inputs.get(str(DC.SOURCE / r)) != ver[r]:
                    raise ValueError(f"RESULTS.json does not list {r} with this SHA-256 among its inputs")
                spec.add(rec.get("spec_sha256"))
                recs[group] = rec
                sources[r] = ver[r]
            for arm, (group, name) in REFERENCE_ARMS.items():
                ev = recs[group]["evaluations"][name]
                for law in REFERENCE_LAWS:
                    for w in ("J", "points"):
                        v = float(ev[law][w]) if ev[law].get("finite", True) else math.nan
                        vals[arm][law][w].append(dict(seed=sd, v=v, sha256=ver[files[group]]))
        setups = {arm: {law: {w: dict(per_seed=vals[arm][law][w],
                                       mean=C.summary([x["v"] for x in vals[arm][law][w]])["mean"])
                              for w in ("J", "points")} for law in REFERENCE_LAWS} for arm in REFERENCE_ARMS}
        rq = next((x for x in results.get("qualities", []) if x.get("quality") == q), None)
        if rq is None:
            raise ValueError(f"RESULTS.json has no quality {q}")
        deltas, checks = {}, {}
        for name, (a, b, key) in REFERENCE_DELTAS.items():
            deltas[name] = {}
            for law in REFERENCE_LAWS:
                d = [x["v"] - y["v"] for x, y in zip(vals[a][law]["J"], vals[b][law]["J"])]
                s = C.summary(d)
                deltas[name][law] = dict(s, label=C.POINTER_LABEL)
                if key is None:
                    continue
                theirs = rq["comparisons"][key]["laws"][law]["J"]
                mine_theirs = list(zip([float(x) for x in theirs["per_seed"]], d))
                same_seeds = len(theirs["per_seed"]) == len(d) and all(
                    x == y or (math.isnan(x) and math.isnan(y)) for x, y in mine_theirs)
                close = lambda u, v: (u is None and v is None) or (u is not None and v is not None and
                                                                   abs(float(u) - float(v)) <= 1e-9)
                ok = (same_seeds and close(theirs.get("mean"), s["mean"])
                      and close(theirs.get("lower_one_sided"), s["lower"])
                      and close(theirs.get("p_one_sided_uncorrected"), s["p_one_sided"]))
                checks[f"{key}/{law}"] = ok
                if not ok:
                    raise ValueError(f"the reference row's {name} on {law} differs from RESULTS.json's {key}")
        out[q] = dict(setups=setups, deltas=deltas, results_json_cross_check=checks)
    if len(spec) != 1:
        raise ValueError("the reference records were made under different specifications")
    return dict(name=REFERENCE_NAME, label=C.POINTER_LABEL, made_up=made_up,
                note=("the repository's CORL-derived TD3+BC port, not Unifloral's: a labeled reference row and a "
                      "cross-check of direction only; it never enters a claim, a Holm family or a kill rule (plan "
                      "sections 5.1 and 6.3). Standard BCA is its unweighted arm, IW BCA its per-row exact-weighted "
                      "strong arm, room its plain TD3+BC on the full log"),
                arms={k: v[1] for k, v in REFERENCE_ARMS.items()}, spec_sha256=spec.pop(), qualities=out,
                sources=sources)


# ------------------------------------------------------------------------------------------------ build and write

def code_sha256():
    """SHA-256 of the files that build results.json."""
    return {rel(PACKAGE / n): sha256_file(PACKAGE / n) for n in ("export.py", "claims.py", "handoff.py")}


def build(permission, *, frozen=None, runs_root=RUNS_ROOT, anchors=None, reference_dest=None, data_root=DATA,
          qualities=C.QUALITIES, reference=True):
    """results.json's content (module docstring). permission and frozen as resolve(); runs_root: the stage driver's
    root (runs/shift_hosts); anchors: read_start_bank_anchors() by default; reference_dest: the copy folder of the
    reference row (data_copy.DEST by default; a made-up build must name its own); data_root: the prepared files (for
    ReBRAC's substituted shares when a record lacks them; None reads none); reference=False leaves the reference row
    out (recorded as not read)."""
    frozen, perm = resolve(permission, frozen)
    anchors = anchors or read_start_bank_anchors()
    drv = read_driver(runs_root, frozen)
    col = collect(runs_root, frozen, qualities, frozenset(drv["accepted_s5"]))
    checked = check_score_anchors(col, anchors["anchors"])
    integ = integrity(col, frozen, drv, checked)
    by_q, never_scored = {}, {}
    for q in qualities:
        runs, aux = col["runs"][q], col["aux"][q]
        judged = C.judge(q, runs, anchors["anchors"]["none"], s7_stopped=drv["s7_stopped"])
        # version 3: the runs of a scope S7 stopped enter no claim, statistic or mean (the table and the expanded view
        # included); they are listed under s7_stopped.excluded, those that trained but were never scored (the driver
        # adds no lanes to a stopped scope) too, and their compute still counts in the totals
        scopes = judged["s7"]["scopes"]
        used, _ = C.without_s7_stopped(runs, scopes)
        aux_used = {k: a for k, a in aux.items() if not C.s7_excluded(scopes, k[0], k[1])}
        for (L, setup, seed) in sorted(col["unscored_aux"][q]):
            if C.s7_excluded(scopes, L, setup):
                never_scored[(q, L, setup, seed)] = S7_NEVER_SCORED
                judged["s7"]["excluded"].append(dict(learner=L, setup=setup, seed=seed, state="complete", source=None,
                                                     why=S7_NEVER_SCORED))
        timed = list(aux.values()) + list(col["unscored_aux"][q].values())
        by_q[q] = dict(label=C.EXPLORATORY_LABEL, reading_order=C.reading_order(judged, integ),
                       claim0_bar=judged["claim0_bar"], complete=judged["complete"], families=judged["families"],
                       claims=judged["learners"], table=C.phone_table(judged, runs),
                       expanded=expanded_view(q, used, aux_used, judged, data_root),
                       writeups=C.writeups(judged), stage3_order=C.stage3_order(judged),
                       s7_stopped=dict(judged["s7"], rule=CAVEATS[-1]),
                       totals=dict(card_seconds=float(sum((a["timings"] or {}).get("card_seconds") or 0.0
                                                          for a in timed)),
                                   processor_seconds=float(sum((a["timings"] or {}).get("processor_seconds") or 0.0
                                                               for a in timed))))
        by_q[q]["_judged"] = judged
    study = C.study_rule({q: v.pop("_judged") for q, v in by_q.items()})
    if reference:
        try:
            ref = reference_row(permission, dest=reference_dest, qualities=qualities)
        except (OSError, ValueError, KeyError, TypeError, H.HandoffError, DC.CopyError) as error:
            ref = dict(name=REFERENCE_NAME, status="not read", why=f"{type(error).__name__}: {error}")
    else:
        ref = dict(name=REFERENCE_NAME, status="not read", why="left out by the caller")
    return dict(schema=SCHEMA, label=C.EXPLORATORY_LABEL, exploratory=True, made_up=frozen.made_up,
                caveats=list(CAVEATS),
                built=dict(utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), permission=perm,
                           frozen=dict(fingerprint=frozen.fingerprint, prespec_sha256=frozen.prespec_sha256,
                                       code_manifest_sha256=frozen.manifest_sha256),
                           code_sha256=code_sha256(), runs_root=rel(runs_root),
                           driver=dict(path=drv["path"], read=drv["read"], mode=drv["mode"], files=drv["files"],
                                       why=drv["why"], problems=drv["problems"], s7_stopped=drv["s7_stopped"]),
                           anchors=dict(path=anchors.get("path"), sha256=anchors.get("sha256"),
                                        weighting=anchors["anchors"],
                                        recording_points=anchors.get("recording_points")),
                           environment=dict(python=sys.version.split()[0], numpy=np.__version__)),
                badges=list(C.BADGES), reading_order_rule=list(C.STOP_RULES) + list(C.READING_ORDER),
                integrity=integ, qualities=by_q, study_rule=study, reference_row=ref,
                runs=[dict(r, s7_stopped=never_scored[k]) if k in never_scored else r
                      for r in col["listed"] for k in [(r["quality"], r["learner"], r["setup"], r["seed"])]],
                excluded=col["excluded"],
                unscored=[dict(u, s7_stopped=never_scored[k]) if k in never_scored else u
                          for u in col["unscored"] for k in [(u["quality"], u["learner"], u["setup"], u["seed"])]],
                halted_runs=[dict(h, stop=h["stop"]) for h in col["halts"]])


def write(results, path=OUT):
    """Write results.json with its sidecar (handoff.write_json). Real results go to outputs/shift_hosts/results.json or
    under runs/shift_hosts/ (git-ignored previews); made-up results only under runs/shift_hosts/stage0/."""
    path = Path(path).resolve()
    if results.get("made_up") is True:
        if STAGE0.resolve() not in path.parents:
            raise ValueError("made-up results are written only under runs/shift_hosts/stage0/")
    elif results.get("made_up") is False:
        if path != OUT.resolve() and RUNS_ROOT.resolve() not in path.parents:
            raise ValueError("real results go to outputs/shift_hosts/results.json or under runs/shift_hosts/")
        if STAGE0.resolve() in path.parents:
            raise ValueError("real results are never written among the stage-0 test files")
    else:
        raise ValueError("results must say made_up true or false")
    return H.write_json(path, results)


def main(argv=None):
    p = argparse.ArgumentParser(description="Build outputs/shift_hosts/results.json (needs the user's terminal "
                                            "approval of the frozen pre-specification).")
    p.add_argument("--out", default=str(OUT))
    a = p.parse_args(argv)
    approval = gate.require_run_approval("export results.json")
    results = build(approval)
    digest = write(results, a.out)
    print(f"wrote {a.out} ({digest})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
