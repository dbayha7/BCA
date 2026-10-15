"""EXPLORATORY trace test bed (plan section 4.6 of outputs/shift/beyond-the-data/PLAN.md). Outside the gate; changes
no pre-registered rule. NO BCA: no band, calibrator, threshold, dose, BCA update or BCA training arm is called.
Nothing from algorithms/td3_bc_bca.py or learners.bca_* is called (learners.py imports them; they are never called).
From calibration/ only calibration/bank.stratified_bank runs, because the gate's data loader
(experiments/shift/dataset.py, through acceptance.load_seed) calls it to regenerate the log's calibration-bank split;
it is a NumPy row sampler, not a BCA signal. Every signal here is a data-only stand-in.

What it is for: the gapped log hides the stall almost completely (34-59 surviving rows past 80%), so no signal from
those data can tell a harmful unsupported command from a merely unrecorded one. A trace log keeps a small seeded
random share of the rows the gate's removal took out past 80% throttle, so that the data carry a faint trace of the
stall. It asks two questions (plan 4.6):
  Q1  does the critic trained on the trace log still overrate pushing past 80%, with harm of at least 50 points in
      strong current?
  Q2  does residual evidence (minus the mean one-step error of the 10 nearest training rows) separate the stalling
      and no-stall worlds with selectivity of at least 0.2, where the recorded-range hinge cannot?

How a trace log is built (a new removal, computed here, outside the hashed package; spec v2, paired worlds):
  the gate's removal ('gap', shift.make_removal, unchanged) of the seed and world; its removed training rows whose
  recorded command is past 80% on any thruster (thruster_chain.past_knee) are that world's candidates. The two worlds'
  logs share episodes, time steps and (at almost every candidate) the recorded command, so the candidates that BOTH
  worlds have (the same log positions) are the selection pool; the few world-specific candidates are never added back
  and are reported. One fixed permutation default_rng([20261015, seed, 91]) of the shared candidates is drawn per seed
  and used in BOTH worlds; the share-p log adds back the first floor(p * N + 0.5) of them (N shared candidates). So
  the two worlds keep the same episodes and time steps, and the 0.5% rows are a subset of the 2% rows. The
  calibration bank gets the same rule over its shared removed past-80% rows with default_rng([20261015, seed, 91, 1])
  (not used by anything that runs here; kept so that a later BCA test on these logs sees the same trace in its bank).
  The evaluation split is never touched. (numpy treats a trailing 0 tag as absent, so no tag list here ends in 0.)

Spec v3 (POST HOC: chosen after seeing the 0.5% and 2% results) adds the shares 1%, 3%, 5% and 10% with the same rule,
the same permutation and the same protocol, so the rows of every smaller share are inside every larger one
(0.5% < 1% < 2% < 3% < 5% < 10%) and both worlds add back the same rows. The spec-v2 jobs (share 0, 0.5% and 2%) are
kept as they are: no phase runs on them under spec v3 (run_job refuses) and guard_out refuses any write into their
folders or to the spec-v2 pooled look, seed-0 checks and timing (FROZEN_V2_PATHS). They are read with spec v2's
SHA-256 (share_spec_sha256).

Every computing phase must run pinned to ONE CPU (require_one_cpu; launch.sh pins with taskset). The gate's results
depend on how many threads a computation uses: unpinned, the same actor-only call gives different actors (review
finding 2: on 4 CPUs the 1,000-step actors already differ; on 1 CPU they reproduce check 2's actors bit for bit).

Everything that trains calls the gate's own functions, unchanged and imported read-only, with the pre-registered
learner settings (prereg.config.learn): learners.prepare (with the gate's normalisation: the gapped log's kept rows,
as the gate uses for every critic of a seed), learners.clone, learners.fixed_critic (the stall-aware controller's
fixed-controller critic), learners.ActorOnly.run (check 2's actor-only protocol, critic swap, unweighted, alpha 5)
and learners.evaluate_actor (through acceptance._evaluate). Gate artifacts are read with the pattern of
outputs/shift/stop-report/cards/cards_outside_gate.py. Nothing is written under experiments/, runs/shift/ or
runs/shift_explore/room/: every write goes through guard_out(), which allows runs/shift_explore/trace only.
"""

import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path("/home/user/bca")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.shift.acceptance as A  # noqa: E402  (puts gymnasium on the path)
import experiments.shift.dataset as DS  # noqa: E402
import experiments.shift.learners as L  # noqa: E402
import experiments.shift.recorder as RC  # noqa: E402
import experiments.shift.shift as SH  # noqa: E402
import experiments.shift.thruster_chain as TC  # noqa: E402
import experiments.shift.truth as TR  # noqa: E402

OUT = ROOT / "runs" / "shift_explore" / "trace"
SPEC_PATH = OUT / "spec.json"
HERE = Path(__file__).resolve().parent
PLAN_PATH = ROOT / "outputs" / "shift" / "beyond-the-data" / "PLAN.md"
ROOM_SPEC_PATH = ROOT / "runs" / "shift_explore" / "room" / "spec.json"
PROTECTED = (ROOT / "experiments", ROOT / "runs" / "shift", ROOT / "runs" / "shift_explore" / "room")
LABEL = ("EXPLORATORY: trace test bed (plan section 4.6), computed outside the gate on saved gate artifacts and new "
         "trace logs; not a gated output; changes no pre-registered rule; no BCA band, calibrator, threshold, dose, "
         "BCA update or BCA training arm was used (data-only stand-ins only)")

ALPHA = 5.0
SEEDS = (0, 1, 2)
WORLDS = ("stall", "no_stall")
WORLD_INDEX = {"stall": 0, "no_stall": 1}
V2_SHARES = (0.005, 0.02)              # the trace logs of spec v2 (plan 4.6), as planned
POST_HOC_SHARES = (0.01, 0.03, 0.05, 0.1)   # spec v3, POST HOC: chosen after seeing the 0.5% and 2% results
SHARES = tuple(sorted(V2_SHARES + POST_HOC_SHARES))   # every trace log, smallest first
REFERENCE_SHARE = 0.0                  # the gate's own gapped log, read back from check 2 (no training)
FROZEN_SHARES = (REFERENCE_SHARE,) + V2_SHARES       # spec v2's jobs: complete, kept as they are under spec v3
SHARE_DIR = {0.0: "share_0pct", 0.005: "share_0.5pct", 0.01: "share_1pct", 0.02: "share_2pct", 0.03: "share_3pct",
             0.05: "share_5pct", 0.1: "share_10pct"}
POST_HOC_LABEL = "post hoc: chosen after seeing the 0.5% and 2% results"
POOLED_V2_NAME, POOLED_V3_NAME = "pooled.json", "pooled_v3.json"
FROZEN_V2_FILES = ("pooled.json", "checks_seed0.json", "timing.json")   # spec v2's outputs at the top level
TAG_TRACE, TAG_T1, TAG_T3, TAG_NOISE = 91, 92, 93, 94
CONTROLLER = "stall_aware"
ARMS = ("trace", "full", "gapped")     # actor-only arms (critic swap): the trace critic, the gate's full-log critic,
#                                        the gate's gapped critic (reported control), all on the trace log's rows
EVAL_ACTORS = ("clone", "trace", "full", "gapped")
HARM_MIN = 50.0                        # Q1: harm of at least 50 points in strong current
SELECTIVITY_MIN = 0.2                  # Q2: selectivity of at least 0.2
MIN_PAST80_PAIRS = 20                  # a world's flag rate needs at least this many past-80% pairs
H1_KEYS = ("price|region", "gap|pull|region", "full|pull|region")

MANIFEST_NAME = "code_manifest.json"
CODE_FILES = ("tracebed.py", "signals.py", "run_job.py", "checks_seed0.py", "checks_v3.py", "write_spec.py",
              "write_manifest.py", "launch.sh", "test_trace.py")


def share_origin(share):
    """Where a share comes from: the reference, spec v2 (planned) or spec v3 (post hoc)."""
    s = float(share)
    if s == REFERENCE_SHARE:
        return "reference (the gate's gapped log)"
    if s in V2_SHARES:
        return "planned (spec v2)"
    if s in POST_HOC_SHARES:
        return "post hoc (spec v3)"
    raise ValueError(f"share {share} is not a share of the test bed")


def share_label(share):
    """The label every record of a job carries: the post-hoc label for the spec-v3 shares."""
    return POST_HOC_LABEL if float(share) in POST_HOC_SHARES else share_origin(share)


def spec_shares(spec):
    """Every trace share the spec names: spec v2's 'shares' plus spec v3's post-hoc shares."""
    ext = spec.get("post_hoc_extension") or {}
    return [float(s) for s in spec["shares"]] + [float(s) for s in ext.get("shares", [])]


def share_spec_sha256(spec, spec_sha, share):
    """The spec SHA-256 a job's outputs must name: under spec v3, spec v2's for the frozen spec-v2 jobs (share 0, 0.5%,
    2%), the current spec's for the post-hoc shares."""
    ext = spec.get("post_hoc_extension")
    if ext and float(share) in [float(s) for s in ext["frozen_v2"]["shares"]]:
        return ext["frozen_v2"]["spec_sha256"]
    return spec_sha
GATE_FILES = tuple(A.NEW_FILES) + tuple(A.REUSED_FILES) + (
    "outputs/shift/stop-report/cards/cards_outside_gate.py", "runs/shift/preregistration.json",
    "runs/shift/preregistration.json.sha256")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def cpus():
    return sorted(os.sched_getaffinity(0))


def require_one_cpu(what="this phase"):
    """Refuse unless the process is pinned to exactly one CPU (review finding 2: the gate's numbers depend on the
    thread count, and one CPU reproduces them). Returns the CPU list for the records."""
    c = cpus()
    if len(c) != 1:
        raise SystemExit(f"refused: {what} must run pinned to one CPU (taskset -c <cpu> ...; launch.sh does this); "
                         f"this process may use CPUs {c}")
    return c


# ------------------------------------------------------------------------------------------------ JSON (gate pattern)

FROZEN_V2_PATHS = tuple(
    [OUT / name for name in FROZEN_V2_FILES] + [OUT / (name + ".sha256") for name in FROZEN_V2_FILES]
    + [OUT / w / f"seed{r}" / SHARE_DIR[s] for w in WORLDS for r in SEEDS for s in FROZEN_SHARES])


def guard_out(path):
    """Refuse any output outside runs/shift_explore/trace (never experiments/, runs/shift/ or the room test), and any
    write into spec v2's finished outputs (FROZEN_V2_PATHS: the share-0, 0.5% and 2% job folders, pooled.json,
    checks_seed0.json, timing.json), which spec v3 keeps as they are."""
    p = Path(path).resolve()
    if OUT.resolve() not in p.parents and p != OUT.resolve():
        raise SystemExit(f"refused: {p} is outside {OUT}")
    for protected in PROTECTED:
        if p == protected.resolve() or protected.resolve() in p.parents:
            raise SystemExit(f"refused: {p} is inside {protected}")
    for frozen in FROZEN_V2_PATHS:
        if p == frozen.resolve() or frozen.resolve() in p.parents:
            raise SystemExit(f"refused: {p} is a spec-v2 output, kept as it is under spec v3")
    return p


def write_json(path, payload):
    """acceptance.write_json's format (strict sorted JSON, '<name>.sha256' sidecar), restricted to OUT."""
    path = guard_out(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = A.canonical(payload)
    path.write_text(text, encoding="utf8")
    digest = hashlib.sha256(text.encode("utf8")).hexdigest()
    path.with_name(path.name + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf8")
    return digest


def read_json(path):
    """A JSON written by write_json; None when missing; ValueError when its sidecar does not match."""
    path = Path(path)
    if not path.exists():
        return None
    text = path.read_text(encoding="utf8")
    side = path.with_name(path.name + ".sha256")
    if not side.exists() or side.read_text(encoding="utf8").split()[0] != hashlib.sha256(text.encode("utf8")).hexdigest():
        raise ValueError(f"{path}: its SHA-256 differs from its sidecar")
    return json.loads(text)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_tree(path, tree):
    return L.save_tree(guard_out(path), tree)


def save_npz(path, **arrays):
    """np.savez under OUT; returns {name: dataset.sha256_array} of what was written."""
    path = guard_out(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return {k: DS.sha256_array(v) for k, v in arrays.items()}


def load_npz(path, sha):
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    if {k: DS.sha256_array(v) for k, v in arrays.items()} != sha:
        raise ValueError(f"{path}: arrays differ from their recorded SHA-256")
    return arrays


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if math.isfinite(x) else None
    return x


# ------------------------------------------------------------------------------------------------ code and spec pins

def gate_file_hashes():
    return {rel: file_sha256(ROOT / rel) for rel in GATE_FILES}


def code_hashes(names=CODE_FILES):
    return {n: file_sha256(HERE / n) for n in names}


def load_spec():
    spec = read_json(SPEC_PATH)
    if spec is None:
        raise SystemExit(f"refused: no spec at {SPEC_PATH}; it must be written before any run")
    return spec, file_sha256(SPEC_PATH)


def check_gate_files(spec):
    now = gate_file_hashes()
    bad = [k for k, v in spec["gate_files_sha256"].items() if now.get(k) != v]
    if bad:
        raise SystemExit("refused: gate files changed since the spec: " + ", ".join(bad))


def check_code(spec_sha):
    """Refuse unless the code manifest exists, matches its sidecar, names the current spec, and every file it lists
    (here and in its copy under OUT/code/) still has the SHA-256 it recorded. Returns (manifest, its SHA-256)."""
    path = OUT / MANIFEST_NAME
    man = read_json(path)
    if man is None:
        raise SystemExit(f"refused: no code manifest at {path}; write it (write_manifest.py) before any phase")
    if man.get("spec_sha256") != spec_sha:
        raise SystemExit("refused: the code manifest names another spec; write a new, labelled manifest")
    now = code_hashes(tuple(man["files"]))
    bad = [n for n, h in man["files"].items() if now[n] != h]
    copies = [n for n, h in man["files"].items()
              if not (OUT / "code" / n).exists() or file_sha256(OUT / "code" / n) != h]
    if bad or copies:
        raise SystemExit("refused: the code differs from the manifest (" + ", ".join(bad + [f"code/{n}" for n in copies])
                         + "); any change needs a new, labelled manifest written before the next phase")
    return man, file_sha256(path)


def begin():
    """Every phase starts here: the spec, the gate files and the code manifest. Returns (spec, spec_sha, man_sha)."""
    spec, spec_sha = load_spec()
    check_gate_files(spec)
    _, man_sha = check_code(spec_sha)
    return spec, spec_sha, man_sha


# ------------------------------------------------------------------------------------------------ gate artifacts

class GateSeed:
    """One gate seed's verified gate artifacts, read through the gate's own loaders (cards_outside_gate.py pattern):
    A.load_prereg(A.refuse_protected(A.RUNS_DEFAULT)), A.load_seed, the fixed-controller critics through
    A.get_critic(..., train=False), the saved cloned actors (get_clone's read path only), check 2's saved actors and
    values, and check 1's saved rows (the frozen region pool's exact price of pushing past 80%)."""

    def __init__(self, rep, worlds=WORLDS):
        t = time.perf_counter()
        self.prereg = A.load_prereg(A.refuse_protected(A.RUNS_DEFAULT))
        A.guard(self.prereg)
        self.cfg = self.prereg.config
        self.learn = self.cfg.learn
        outcome = A.gate_outcome(self.prereg)
        self.variant = outcome["variant"]
        last = outcome.get("alpha") or A._last_alpha_run(self.prereg, self.variant)
        if float(last) != ALPHA:
            raise SystemExit(f"refused: the gate's last alpha is {last}; the trace test bed is fixed at alpha {ALPHA}")
        self.rep = int(rep)
        self.sd = {w: A.load_seed(self.prereg.runs, self.cfg, self.variant, w, rep) for w in worlds}
        self._h1 = None
        self.load_seconds = time.perf_counter() - t

    def gapped_prepared(self, world):
        return self.sd[world].prepared("gap", self.learn)

    def critic(self, world, removal):
        rec, params = A.get_critic(self.prereg, self.sd[world], removal, CONTROLLER, train=False)
        if params is None:
            raise A.GateError(f"the {world}/{removal} critic of seed {self.rep} failed in the gate")
        return params, rec

    def clone(self, world):
        """get_clone's read path only (never its training path, which would write under runs/shift)."""
        d = A._variant_root(self.prereg, "clone", self.variant) / world / f"seed{self.rep}"
        rec = A.finished(d / "record.json", self.prereg)
        if rec is None:
            raise A.GateError(f"no saved cloned actor at {d}")
        return L.load_tree(d / "actor.msgpack", L.actor_template(), rec["result"]["file_sha256"]), rec

    def h2(self):
        d = A._variant_root(self.prereg, "harm_return", self.variant) / f"alpha{ALPHA:g}" / f"seed{self.rep}"
        rec = A.finished(d / "values.json", self.prereg)
        if rec is None:
            raise A.GateError(f"check 2 has no verified values at {d}")
        acts = L.load_tree(d / "actors.msgpack", {k: L.actor_template() for k in rec["actor_names"]},
                           rec["actors_sha256"])
        return acts, rec, d

    def h1_rows(self):
        """Check 1's saved per-row arrays of this seed (rows.npz), verified against values.json's rows_sha256."""
        if self._h1 is None:
            d = A._variant_root(self.prereg, "harm_critic", self.variant) / f"seed{self.rep}"
            rec = A.finished(d / "values.json", self.prereg)
            if rec is None:
                raise A.GateError(f"check 1 has no verified values at {d}")
            keys = [f"{w}:{CONTROLLER}:{k}" for w in WORLDS for k in H1_KEYS]
            with np.load(d / "rows.npz") as z:
                rows = {k: z[k] for k in keys}
            bad = [k for k in keys if DS.sha256_array(rows[k]) != rec["rows_sha256"].get(k)]
            if bad:
                raise A.GateError(f"check 1's rows differ from their record: {bad}")
            self._h1 = rows
        return self._h1

    def anchors(self, world):
        return A.anchors_of(self.prereg, self.variant, world)


# ------------------------------------------------------------------------------------------------ the trace removal

def trace_count(share, n):
    """floor(share * n + 0.5): the number of candidates a share-p log adds back."""
    return int(math.floor(float(share) * int(n) + 0.5))


def select_trace(candidates, share, tags):
    """The first trace_count(share, N) entries of the fixed permutation default_rng(tags).permutation(N) of the
    candidates (an ascending array), sorted. Shares of one seed use the same tags, so a smaller share's selection is a
    subset of a larger share's; both worlds use the same tags and the same (shared) candidates, so they select the same
    rows."""
    candidates = np.asarray(candidates, np.int64)
    perm = np.random.default_rng([int(t) for t in tags]).permutation(len(candidates))
    k = trace_count(share, len(candidates))
    return np.sort(candidates[perm[:k]]), k


def selection_tags(rep, bank=False):
    """Spec v2: one permutation per seed for both worlds (no world index); the bank's ends in 1 (never in 0)."""
    base = [L.LEARNER_SEED, int(rep), TAG_TRACE]
    return base + [1] if bank else base


def past80(actions, physics):
    """Per row: any thruster past 80% (thruster_chain.past_knee on the float32 command, as stored)."""
    return TC.past_knee(np.asarray(actions, np.float32), physics)


def world_candidates(sd):
    """One world's candidates: (log rows of the gate's removed training rows past 80% on any thruster, log rows of the
    gate's removed calibration-bank rows past 80%), both ascending."""
    lg, rg = sd.log, sd.removals["gap"]
    removed_rows = rg.removed_train_rows(lg)
    cand = removed_rows[past80(lg["actions"][removed_rows], sd.physics)]
    bank = lg.bank_rows
    bank_cand = bank[np.flatnonzero(rg.bank_removed & past80(lg["actions"][bank], sd.physics))]
    return np.asarray(cand, np.int64), np.sort(np.asarray(bank_cand, np.int64))


def shared_candidates(sds):
    """The candidates both worlds have (the same log positions), training and bank: the selection pools."""
    c = {w: world_candidates(sds[w]) for w in WORLDS}
    return (np.intersect1d(c["stall"][0], c["no_stall"][0]), np.intersect1d(c["stall"][1], c["no_stall"][1]), c)


@dataclass
class Trace:
    world: str
    rep: int
    share: float
    removal: object          # shift.Removal of kind 'trace'
    added_rows: np.ndarray   # log rows added back to training (ascending)
    added_bank: np.ndarray   # log rows added back to the calibration bank (ascending)
    candidates: np.ndarray   # this world's candidates: the gate's removed training rows past 80% (ascending)
    bank_candidates: np.ndarray
    shared: np.ndarray       # the candidates both worlds have: the selection pool (ascending)
    shared_bank: np.ndarray
    world_specific: np.ndarray       # this world's candidates the other world lacks (never added back)
    world_specific_bank: np.ndarray


def build_trace(sds, world, share):
    """The trace removal of one seed and world at one share (module docstring). sds: {world: the gate's loaded seed}
    of BOTH worlds (the selection pool is their shared candidates)."""
    sd = sds[world]
    lg, rg = sd.log, sd.removals["gap"]
    shared, shared_bank, c = shared_candidates(sds)
    cand, bank_cand = c[world]
    added, _ = select_trace(shared, share, selection_tags(sd.rep))
    added_bank, _ = select_trace(shared_bank, share, selection_tags(sd.rep, bank=True))
    removed = rg.removed.copy()
    removed[added] = False
    bank_removed = rg.bank_removed.copy()
    bank_removed[np.isin(lg.bank_rows, added_bank)] = False
    removal = SH.Removal("trace", removed, bank_removed, sd.gap)
    return Trace(world, int(sd.rep), float(share), removal, added, added_bank, cand, bank_cand, shared, shared_bank,
                 np.setdiff1d(cand, shared), np.setdiff1d(bank_cand, shared_bank))


def trace_prepared(sd, trace, learn):
    """learners.prepare of the trace log with the gate's normalisation (the gapped log's kept rows, sd.norm()), as the
    gate prepares every critic's data of a seed."""
    return L.prepare(sd.log, trace.removal, sd.norm(), learn)


RULE = ("candidates: the gate's removed training rows past 80% on any thruster, per world; selection pool: the "
        "candidates both worlds have (the same log positions); added: the first floor(share * N + 0.5) of "
        "default_rng([20261015, seed, 91]).permutation(N) over the N shared candidates, the same rows in both worlds; "
        "world-specific candidates are never added back; bank: the same with default_rng([20261015, seed, 91, 1]) "
        "over the shared removed bank rows past 80%")


def trace_record(sds, trace):
    """The exact rows of a trace log, their SHA-256 and the checks that the build is as intended (paired with the
    other world's build of the same seed and share)."""
    sd = sds[trace.world]
    lg, rg, tr = sd.log, sd.removals["gap"], trace.removal
    other_world = [w for w in WORLDS if w != trace.world][0]
    other = build_trace(sds, other_world, trace.share)
    olg = sds[other_world].log
    train, ev, cal = lg.split_rows("train"), lg.split_rows("evaluation"), lg.split_rows("calibration")
    kept, gate_kept = tr.kept_train_rows(lg), rg.kept_train_rows(lg)
    removed_rows = rg.removed_train_rows(lg)
    n_c, n_s, k = int(len(trace.candidates)), int(len(trace.shared)), int(len(trace.added_rows))
    nb_c, nb_s, kb = int(len(trace.bank_candidates)), int(len(trace.shared_bank)), int(len(trace.added_bank))
    p_kept = past80(lg["actions"][kept], sd.physics)
    p_gate = past80(lg["actions"][gate_kept], sd.physics)
    bank_now, bank_gate = tr.bank_rows(lg), rg.bank_rows(lg)
    gate_rec = sd.record["removals"]["gap"]
    checks = dict(
        gate_removal_matches_its_record=bool(DS.sha256_array(gate_kept) == gate_rec["kept_rows_sha256"]
                                             and DS.sha256_array(bank_gate) == gate_rec["bank_rows_sha256"]),
        count_rule=bool(k == trace_count(trace.share, n_s) and kb == trace_count(trace.share, nb_s)),
        added_unique=bool(len(np.unique(trace.added_rows)) == k and len(np.unique(trace.added_bank)) == kb),
        added_are_shared_candidates=bool(np.isin(trace.added_rows, trace.shared).all()
                                         and np.isin(trace.added_bank, trace.shared_bank).all()),
        added_are_removed_training_rows=bool(np.isin(trace.added_rows, removed_rows).all()),
        added_are_past_80=bool(past80(lg["actions"][trace.added_rows], sd.physics).all()) if k else True,
        added_bank_are_removed_bank_rows=bool(np.isin(trace.added_bank, bank_gate, invert=True).all()
                                              and np.isin(trace.added_bank, lg.bank_rows).all()),
        added_bank_are_past_80=bool(past80(lg["actions"][trace.added_bank], sd.physics).all()) if kb else True,
        kept_is_gate_kept_plus_added=bool(len(kept) == len(gate_kept) + k
                                          and np.array_equal(np.sort(np.concatenate([gate_kept, trace.added_rows])), kept)),
        bank_is_gate_bank_plus_added=bool(len(bank_now) == len(bank_gate) + kb
                                          and np.array_equal(np.sort(np.concatenate([bank_gate, trace.added_bank])),
                                                             bank_now)),
        training_rows_only=bool(not tr.removed[ev].any() and not tr.removed[cal].any()
                                and np.isin(trace.added_rows, train).all()),
        evaluation_split_untouched=bool(not tr.removed[ev].any()),
        same_rows_as_the_other_world=bool(np.array_equal(trace.added_rows, other.added_rows)
                                          and np.array_equal(trace.added_bank, other.added_bank)
                                          and np.array_equal(trace.shared, other.shared)),
    )
    same_cmd = (np.asarray(lg["actions"][trace.added_rows], np.float32)
                == np.asarray(olg["actions"][trace.added_rows], np.float32)).all(1)
    return dict(
        world=trace.world, rep=int(sd.rep), share=float(trace.share), share_dir=SHARE_DIR[trace.share],
        rule=RULE, selection_tags=selection_tags(sd.rep), bank_selection_tags=selection_tags(sd.rep, True),
        candidates=n_c, candidates_sha256=DS.sha256_array(trace.candidates),
        shared_candidates=n_s, shared_candidates_sha256=DS.sha256_array(trace.shared),
        world_specific_candidates=int(len(trace.world_specific)),
        world_specific_candidate_rows=[int(r) for r in trace.world_specific],
        added=k, realised_share_of_shared_candidates=(k / n_s) if n_s else None,
        realised_share_of_candidates=(k / n_c) if n_c else None,
        added_rows=[int(r) for r in trace.added_rows], added_rows_sha256=DS.sha256_array(trace.added_rows),
        added_rows_recorded_command_same_in_both_worlds=float(same_cmd.mean()) if k else None,
        gate_removed_training_rows=int(len(removed_rows)),
        gate_kept_training_rows=int(len(gate_kept)), gate_kept_rows_sha256=DS.sha256_array(gate_kept),
        kept_training_rows=int(len(kept)), kept_rows_sha256=DS.sha256_array(kept),
        past80_kept_rows=int(p_kept.sum()), past80_kept_share=float(p_kept.mean()),
        gate_past80_kept_rows=int(p_gate.sum()), gate_past80_kept_share=float(p_gate.mean()),
        past80_full_log_training_rows=int(past80(lg["actions"][train], sd.physics).sum()),
        bank_candidates=nb_c, shared_bank_candidates=nb_s,
        world_specific_bank_candidates=int(len(trace.world_specific_bank)),
        world_specific_bank_candidate_rows=[int(r) for r in trace.world_specific_bank],
        bank_added=kb, added_bank_rows=[int(r) for r in trace.added_bank],
        added_bank_rows_sha256=DS.sha256_array(trace.added_bank),
        gate_bank_rows=int(len(bank_gate)), bank_rows=int(len(bank_now)), bank_rows_sha256=DS.sha256_array(bank_now),
        checks=checks, passed=bool(all(checks.values())),
    )


# ------------------------------------------------------------------------------------------------ readouts

def region_pull(critic_params, sd, prepared, h1rows):
    """The pull past 80% on check 1's frozen region pool: o(s) = [Q1(s, c_hi) - Q1(s, c_cap)] + price(s), price =
    Q^c(s, c_cap) - Q^c(s, c_hi) the exact value check 1 saved for this world (c the stall-aware controller). Equal
    to acceptance.critic_errors' 'pull|region' (truth.pull_past_80) for the same critic, bit for bit."""
    inputs, _ = A.pool_inputs(sd, CONTROLLER)
    reg = inputs["region"]
    obs = prepared.normalize(reg["x"])
    q_hi = L.critic_rows(critic_params, obs, reg["acts"]["c_hi"])[0][:, 0]
    q_cap = L.critic_rows(critic_params, obs, reg["acts"]["c_cap"])[0][:, 0]
    return (q_hi - q_cap) + h1rows[f"{sd.world}:{CONTROLLER}:price|region"]


def summarise(v):
    v = np.asarray(v, np.float64)
    return dict(mean=float(v.mean()), median=float(np.median(v)), q10=float(np.quantile(v, 0.1)),
                q90=float(np.quantile(v, 0.9)), n=int(len(v)))


def job_dir(world, rep, share):
    return OUT / world / f"seed{int(rep)}" / SHARE_DIR[float(share)]
