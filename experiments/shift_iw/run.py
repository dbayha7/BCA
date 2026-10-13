"""The stage runner of the importance-weighting experiment (iw/SPEC.md, iw/spec.json).

Stages, in the order of spec.json compute.order:
  data      the start banks (once, --start-banks) and each seed's logs with exact truth (data.py)    no approval
  host      plain TD3+BC on the gapped log (main, 'gap') and on the full log (control, 'full')       no approval
  room      the room check on the frozen gapped host critic (no BCA): marks R1 and R2's slope       no approval
  room-lower-edge  the lower edge's room check (no BCA; lower_edge.room_check): check (e) on the      no approval
            exact oracle penalty, then the oracle arms (k = 0, 1, 1.5) and the even pull matched in
            strength; marks L1 and L2's slope
  band      BCA's band on each frozen host critic; the single thresholds of every arm (unweighted,   approval 'band'
            exact, estimated) and the per-step tables of R (exact, estimated); marks W0 and M3; R2's  and a report
            dose change on kept training rows (per-step, and the single threshold beside it); on the  saying D1 passed
            gapped host, L2's scale change and its proxy check (lower_edge.proxy_check: BCA's
            unweighted lower edge against the oracle's at the room check's no-caution actor, and
            check (e) on BCA's penalty; reported beside L2; it reads the room check's record). Only
            after a report on every seed, made by this code, says D1 passed (the design's stop rule)
  coverage  coverage of every arm on held-out and test steps, each test step at its own threshold   approval 'coverage'
            (per-step arms) and at the single threshold (the '_single' arms); marks M1, M2, W0-W4
  bca       the capped pull's co-trained BCA arms (unweighted, exact-weighted strong per row,         approval 'bca'
            exact-weighted strong single threshold), only after the report on all five seeds says    for the quality
            M1, R1 and R2 passed; the estimated-weighted strong arm (per row) only when the user    (+ 'bca_estimated')
            asks (its own scope)
  bca-lower-edge  the lower edge's BCA arms (lower_edge.band_arms: unweighted, exact strong, exact    approval
            mild per step; estimated strong only when the user asks), actor-only on the frozen       'bca_lower_edge'
            gapped critic with the frozen band's paired width, only after the report on all five     for the quality
            seeds says M1, L1 and L2 passed                                           (+ 'bca_lower_edge_estimated')
  evaluate  exact returns (and points) on the three start banks: plain hosts, room arms, the lower   no approval
            edge's room arms, BCA arms and the lower edge's BCA arms
  report    every pass mark of one quality from its finished stages ('pending' where an input is    no approval
            missing)
  summary   one table across the five qualities: the data's points with and without the hold,    no approval
            plain TD3+BC, every mark's status, the claim rules, and the two improvement marks with
            Holm's correction, each over its family: the qualities whose report allows its arms
            (bca_allowed for I1, bca_lower_edge_allowed for L3); judged only when every report is
            complete and every member has run or was declined by the owner (declined.json)
Helpers: freeze-spec writes iw/spec.json's sidecar (the orchestrator's step; it never overwrites a different one);
status lists what is finished; approval-template prints the approval body to fill in (it writes nothing).

Dataset qualities (spec version 5): every stage but summary, freeze-spec and approval-template runs for one quality
(--quality; world.QUALITIES), and reads and writes only that quality's folder. The start banks and the approval are
shared. Every mark is judged per quality on its own five seeds with the same numbers; nothing is pooled across
qualities.

The marks are fixed by the frozen specification. Every pass-mark number is read from iw/spec.json (pass_marks[i].
thresholds) after its SHA-256 is checked, and so are the per-step grid's ratio and the share of high-current rows
(band.per_step); none is written in this file.

The lower edge (spec version 5, part B; lower_edge.py): a second consumer of BCA's band, a penalty k [U(s, pi(s)) -
U(s, a_i)]_+ in the actor's objective, whose scale weighting moves by R(w(c_i)) / R_u per row. Its room check uses the
exact one-step oracle in the same paired form (no BCA); its BCA arms wait for the owner's second go like version 4's.

Per-step thresholds (spec version 4): each test step's threshold is R(its own weight), read from a table of R on a
log-spaced grid of test masses (coverage.threshold_table, coverage.lookup). The per-step arms are 'exact' and
'estimated'; '<arm>_single' is version 3's single threshold (one test mass for every step), kept as a reported
comparison. Names in the band record: tables '<arm>_<tilt>' (per step), thresholds '<arm>_<tilt>_single' (single). The report refuses when any record it judges
was made by other code than the code running now (LI.code_hashes(): the approval names the same hashes), when a
record no longer matches the record it was built on (band -> host (and the room check, gapped), coverage -> band,
room -> host, evaluate -> the records of the controllers it scored), or when a record names another quality or seed
than its folder, or links a record outside its quality's folder (the shared start banks excepted). The report also
records whether the code running now is the code the approval names (code_is_approved, reported).

Refusals. Every stage refuses unless iw/spec.json matches its sidecar (spec.json.sha256) and every input it needs is
finished: an input is finished when its record exists, matches its sha256sum sidecar, says 'finished' and names the
same specification SHA-256. The band, coverage, bca and bca-lower-edge stages also refuse unless
runs/shift_iw_v5/approval.json verifies (learners_iw.verify_approval: its own sidecar, the user's approval verbatim,
the specification's SHA-256, the code's SHA-256 and a scope that covers the stage; a BCA-arm scope only for the
qualities the user's go names). That file is written only by the orchestrator after the user has seen the pass
marks; nothing here creates it. Nor declined.json (the owner's decision not to run a quality's arms, which the summary
reads). The band stage also refuses until a report on every seed, made by
the code running now, says D1 passed (require_design), so an approval written before the data cannot skip the design
check. All inputs are checked before any work starts. A finished output is
never redone (delete it to redo); records are written last, so a crashed run leaves no record and is simply run again.

Layout under the root (default /root/bca-shift/runs/shift_iw_v5):
  approval.json                                   shared by every quality
  declined.json                                   (only if the owner declines a quality's arms) read by the summary
  start_banks/start_banks.hdf5, start_banks.json, stage.json            shared by every quality
  <quality>/data/seed<k>/... and <quality>/data/seed<k>/stage.json
  <quality>/host/<gap|full>/seed<k>/snapshot_050.msgpack, snapshot_100.msgpack, record.json
  <quality>/room/seed<k>/actors.msgpack, record.json   (the lock: strengths and the match, written before any return)
  <quality>/room_lower_edge/seed<k>/actors.msgpack, record.json   (its lock; no actors if check (e) stopped it)
  <quality>/band/<gap|full>/seed<k>/band.msgpack, record.json
  <quality>/coverage/<gap|full>/seed<k>/steps.npz, record.json
  <quality>/bca/<arm>/seed<k>/snapshot_*.msgpack, record.json
  <quality>/bca_lower_edge/<arm>/seed<k>/actor.msgpack, record.json
  <quality>/evaluate/seed<k>/<hosts|room|room_lower_edge|bca|bca_lower_edge>.json
  <quality>/report/report.json
  summary/summary.json
  logs/   (launch.sh)
Records name their inputs by their path under the root (inputs_sha256).
Run from the repository root: python -m experiments.shift_iw.run <stage> --quality <quality> --seeds 0 1 2 3 4 [...];
launch.sh runs one process per quality and seed (and per host kind or BCA arm) on given CPUs.
"""

import argparse
import hashlib
import json
import math
import platform
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import stats

import experiments.shift  # noqa: F401  (gymnasium path and version checks)

import jax.numpy as jnp

from calibration.network import hard_coverage
from experiments.shift import learners as LR
from experiments.shift_iw import classifier as CL
from experiments.shift_iw import coverage as CV
from experiments.shift_iw import data as D
from experiments.shift_iw import learners_iw as LI
from experiments.shift_iw import lower_edge as LE
from experiments.shift_iw import world as W

ROOT_DEFAULT = D.REPO / "runs" / "shift_iw_v5"
SEEDS = (0, 1, 2, 3, 4)
QUALITIES = W.QUALITIES
HOST_KINDS = {"gap": "gap", "full": "none"}          # host kind -> removal
BCA_ARMS = ("unweighted", "exact_strong", "exact_strong_single")   # exact_strong: per row; _single: comparison
BCA_OPTIONAL_ARMS = ("estimated_strong",)            # only when the user asks (per row)
EVAL_GROUPS = ("hosts", "room", "room_lower_edge", "bca", "bca_lower_edge")
EVAL_DEFAULT = ("hosts", "room", "room_lower_edge")    # before any BCA
LE_ARMS = LE.BAND_ARMS                               # the lower edge's BCA arms (per step)
LE_OPTIONAL_ARMS = LE.BAND_OPTIONAL_ARMS             # only when the user asks (scope 'bca_estimated')
PER_STEP_ARMS = ("exact", "estimated")
TEST_ARMS = {"gap": ("unweighted", "exact", "estimated", "exact_single", "estimated_single"),
             "full": ("unweighted", "exact", "exact_single")}
MARK_IDS = ("D1", "M1", "M2", "M3", "W0", "W1", "W2", "W3", "W4", "R1", "R2", "I1", "L1", "L2", "L3")
BAND_DEPENDENT = ("M1", "M2", "M3", "W0", "W1", "W2", "W3", "W4", "R2", "L2")   # fail when a gapped band is invalid
# Holm's families (spec.json pass_marks_rules.holm_families): each improvement mark's family is the qualities whose
# report allows its arms (this flag), and the arms' approval scope
HOLM_ELIGIBLE = {"I1": "bca_allowed", "L3": "bca_lower_edge_allowed"}
ARM_SCOPE = {"I1": "bca", "L3": LE.SCOPE}
DECLINED_KIND = "shift-iw-declined"


class Refused(RuntimeError):
    """A stage refused to run (specification, approval or inputs)."""


@dataclass
class Context:
    """A stage's place: the shared root (start banks, approval, summary) and one quality, whose records live under
    <root>/<quality> only."""

    root: Path = ROOT_DEFAULT
    spec_path: Path = LI.SPEC_JSON
    cfg: object = field(default_factory=LI.settings)
    log: object = None
    quality: str = "expert"

    def __post_init__(self):
        self.root = Path(self.root)
        self.spec_path = Path(self.spec_path)
        if self.quality not in QUALITIES:
            raise Refused(f"refused: unknown quality {self.quality!r}; one of {', '.join(QUALITIES)}")
        if self.log is None:
            self.log = lambda m: print(time.strftime("%H:%M:%S ") + str(m), flush=True)

    def at(self, quality):
        """The same context for another quality (the summary reads every quality's report)."""
        return Context(root=self.root, spec_path=self.spec_path, cfg=self.cfg, log=self.log, quality=quality)

    @property
    def quality_root(self):
        return self.root / self.quality

    @property
    def data_root(self):
        return D.data_dir(self.root, self.quality)

    @property
    def banks_root(self):
        return D.banks_dir(self.root)

    @property
    def approval_path(self):
        return self.root / "approval.json"

    @property
    def declined_path(self):
        return self.root / "declined.json"


# ------------------------------------------------------------------------------------------------ records

def plain(x):
    """Strict-JSON form: numpy and JAX scalars and arrays to Python, tuples to lists, inf and nan to strings."""
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if hasattr(x, "__array__") and not isinstance(x, (str, bytes)):
        a = np.asarray(x)
        return plain(a.tolist()) if a.ndim else plain(a.item())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        v = float(x)
        return v if math.isfinite(v) else ("inf" if v > 0 else "-inf" if v < 0 else "nan")
    return x


def num(v):
    """A number from a record ('inf' strings back to floats; None stays None)."""
    return None if v is None else float(v)


def write_record(path, rec):
    return D.write_json(path, plain(rec))


def read_record(path):
    """A record that exists and matches its sidecar, or None."""
    path = Path(path)
    side = LI.sidecar(path)
    if not path.is_file() or not side.is_file():
        return None
    words = side.read_text(encoding="utf8").split()
    if not words or words[0] != LI.sha256_file(path):
        return None
    return json.loads(path.read_text(encoding="utf8"))


def finished(path, spec):
    rec = read_record(path)
    return rec if rec is not None and rec.get("status") == "finished" and rec.get("spec_sha256") == spec else None


def require(path, spec, what):
    rec = finished(path, spec)
    if rec is None:
        raise Refused(f"refused: {what} has not finished under this specification (no verified record at {path})")
    return rec


def software():
    import flax
    import jax
    import optax
    import scipy

    return dict(python=platform.python_version(), numpy=np.__version__, jax=jax.__version__, flax=flax.__version__,
                optax=optax.__version__, scipy=scipy.__version__)


def stage_record(stage, spec, started, **fields):
    return dict(kind=f"shift-iw-{stage}", stage=stage, status="finished", spec_sha256=spec,
                machine=platform.node(), software=software(), code_sha256=LI.code_hashes(),
                seconds=time.perf_counter() - started, finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **fields)


def spec_digest(ctx):
    try:
        return LI.spec_sha256(ctx.spec_path)
    except LI.SpecError as error:
        raise Refused(str(error)) from error


def approval_for(ctx, stage):
    """The verified approval for a stage; a BCA-arm scope (learners_iw.ARM_SCOPES) also for the context's quality: the
    owner's go names its qualities, so a go for one quality (or one kind of arm) never opens another."""
    try:
        approval = LI.verify_approval(ctx.approval_path, ctx.spec_path, stage)
    except LI.ApprovalError as error:
        raise Refused(str(error)) from error
    if not approval.covers(stage, ctx.quality):
        raise Refused(f"refused: the approval's {stage!r} scope does not cover the {ctx.quality} quality (it covers "
                      f"{list(dict(approval.qualities).get(stage, ()))}); the owner's go names its qualities")
    return approval


def frozen_spec(ctx, spec):
    """iw/spec.json's body, read only when its SHA-256 equals `spec` (the frozen, verified specification)."""
    blob = Path(ctx.spec_path).read_bytes()
    if hashlib.sha256(blob).hexdigest() != spec:
        raise Refused(f"refused: {ctx.spec_path} changed while the stage ran")
    return json.loads(blob.decode("utf8"))


def spec_thresholds(body):
    """{mark id: its numbers} from spec.json's pass_marks (the only source of the pass-mark numbers)."""
    th = {m["id"]: m["thresholds"] for m in body["pass_marks"]}
    missing = [k for k in MARK_IDS if k not in th]
    if missing:
        raise Refused(f"refused: spec.json pass_marks has no thresholds for {missing}")
    return th


def spec_seeds(body):
    return [int(s) for s in body["seeds"]["data_seeds"]]


def spec_per_step(body):
    """The per-step numbers of spec.json (band.per_step): the grid's ratio between neighbouring test masses and the
    share of kept training rows, by current, that R2 reports as high-current rows."""
    ps = body["band"]["per_step"]
    ratio, share = float(ps["grid_ratio"]), float(ps["high_current_rows_top_share"])
    if not (ratio > 1.0 and 0.0 < share < 1.0):
        raise Refused("refused: spec.json band.per_step needs a grid_ratio above 1 and a share in (0, 1)")
    return dict(grid_ratio=ratio, high_current_share=share)


def spec_lower_edge(body):
    """spec.json's lower_edge numbers: the oracle arms' penalty scales and the extra even rungs (a lower_edge.RoomRule
    with check (e)'s rule), the BCA arms' penalty scale k, the arms whose slope L2 reads, and the planned and optional
    BCA arms."""
    le = body["lower_edge"]
    rc = le["room_check"]
    k = {str(n): float(v) for n, v in rc["arms"].items()}
    if sorted(k) != sorted(LE.ROOM_ARMS) or k["le_none"] != 0.0:
        raise Refused(f"refused: spec.json lower_edge.room_check.arms must give {LE.ROOM_ARMS} with le_none at 0")
    slope_arms = [str(a) for a in rc["slope_arms"]]
    if len(slope_arms) < 2 or any(a not in k for a in slope_arms) or len({k[a] for a in slope_arms}) < 2:
        raise Refused("refused: spec.json lower_edge.room_check.slope_arms needs two oracle arms of different scale")
    if tuple(le["planned_arms"]) != LE_ARMS or tuple(le["optional_arms"]) != LE_OPTIONAL_ARMS:
        raise Refused(f"refused: spec.json lower_edge's arms must be {LE_ARMS} and {LE_OPTIONAL_ARMS}")
    rule = LE.RoomRule(k=k, even_extra=tuple(float(m) for m in rc["even_pull"]["extra_rungs"]),
                       check_e=dict(le["check_e"]))
    return dict(rule=rule, k=float(le["k"]), slope_arms=slope_arms)


# ------------------------------------------------------------------------------------------------ paths and inputs

def data_record_path(ctx, seed):
    return ctx.data_root / f"seed{int(seed)}" / "stage.json"


def banks_record_path(ctx):
    return ctx.banks_root / "stage.json"


def host_dir(ctx, kind, seed):
    return ctx.quality_root / "host" / kind / f"seed{int(seed)}"


def room_dir(ctx, seed):
    return ctx.quality_root / "room" / f"seed{int(seed)}"


def room_le_dir(ctx, seed):
    return ctx.quality_root / "room_lower_edge" / f"seed{int(seed)}"


def bca_le_dir(ctx, arm, seed):
    return ctx.quality_root / "bca_lower_edge" / arm / f"seed{int(seed)}"


def band_dir(ctx, kind, seed):
    return ctx.quality_root / "band" / kind / f"seed{int(seed)}"


def coverage_dir(ctx, kind, seed):
    return ctx.quality_root / "coverage" / kind / f"seed{int(seed)}"


def bca_dir(ctx, arm, seed):
    return ctx.quality_root / "bca" / arm / f"seed{int(seed)}"


def evaluate_path(ctx, seed, group):
    return ctx.quality_root / "evaluate" / f"seed{int(seed)}" / f"{group}.json"


def report_path(ctx):
    return ctx.quality_root / "report" / "report.json"


def summary_path(ctx):
    return ctx.root / "summary" / "summary.json"


def load_data(ctx, seed, spec):
    rec = require(data_record_path(ctx, seed), spec, f"data for seed {seed}")
    data = D.load_seed(seed, ctx.data_root)
    if LI.sha256_file(ctx.data_root / f"seed{int(seed)}" / "world.json") != rec["world_json_sha256"]:
        raise Refused(f"refused: seed {seed}'s world.json differs from its data record")
    if data.world.get("quality") != ctx.quality:
        raise Refused(f"refused: seed {seed}'s data are of quality {data.world.get('quality')!r}, not {ctx.quality!r}")
    return data


def require_host(ctx, kind, seed, spec):
    rec = require(host_dir(ctx, kind, seed) / "record.json", spec, f"the {kind} host of seed {seed}")
    if rec["result"]["failed"]:
        raise Refused(f"refused: the {kind} host of seed {seed} failed (non-finite at update "
                      f"{rec['result'].get('failed_at')})")
    return rec


def load_host_state(ctx, kind, seed, spec):
    rec = require_host(ctx, kind, seed, spec)
    snap = rec["result"]["snapshots"][-1]
    if snap["update"] != rec["result"]["final_update"]:
        raise Refused(f"refused: the {kind} host of seed {seed} has no final snapshot")
    return LI.load_host(host_dir(ctx, kind, seed) / "snapshot_100.msgpack", ctx.cfg, snap["file_sha256"])


def require_band(ctx, kind, seed, spec):
    rec = require(band_dir(ctx, kind, seed) / "record.json", spec, f"the {kind} band of seed {seed}")
    if not rec["band_valid"]:
        raise Refused(f"refused: the {kind} band of seed {seed} has no valid refresh")
    return rec


# ------------------------------------------------------------------------------------------------ freeze-spec, data

def freeze_spec(ctx):
    """Write iw/spec.json's sidecar (sha256sum format) if it has none; refuse when a different one exists."""
    spec = Path(ctx.spec_path)
    digest = LI.sha256_file(spec)
    side = LI.sidecar(spec)
    if side.is_file():
        words = side.read_text(encoding="utf8").split()
        if words and words[0] == digest:
            ctx.log(f"{spec} is already frozen: {digest}")
            return digest
        raise Refused(f"refused: {side} exists and differs from {spec}; the specification changed after it was "
                      "frozen. Decide with the user before replacing it.")
    json.loads(spec.read_text(encoding="utf8"))
    side.write_text(f"{digest}  {spec.name}\n")
    ctx.log(f"froze {spec}: {digest}")
    return digest


def stage_data(ctx, seeds=(), start_banks=False, sizes=D.FULL):
    """The shared start banks (start_banks=True) and each seed's data of the context's quality (data.build_seed:
    every quality cuts the expert's ball of the same seed and computes D1 on the expert's quadrature itself) with a
    stage record."""
    spec = spec_digest(ctx)
    if start_banks and finished(banks_record_path(ctx), spec) is None:
        started = time.perf_counter()
        rec = D.build_start_banks(D.refuse_protected(ctx.banks_root), sizes.start_bank, log=ctx.log)
        write_record(banks_record_path(ctx), stage_record("data", spec, started, part="start_banks",
                                                          starts_per_law=sizes.start_bank, start_banks=rec,
                                                          start_banks_json_sha256=LI.sha256_file(
                                                              ctx.banks_root / "start_banks.json")))
    out = D.refuse_protected(ctx.data_root) if seeds else None
    for seed in seeds:
        if finished(data_record_path(ctx, seed), spec) is not None:
            ctx.log(f"data {ctx.quality} seed {seed}: already finished")
            continue
        started = time.perf_counter()
        world, times = D.build_seed(seed, out, sizes, ctx.quality, log=ctx.log)
        write_record(data_record_path(ctx, seed), stage_record(
            "data", spec, started, part="seed", quality=ctx.quality, seed=int(seed), sizes=sizes.as_dict(),
            D1=world["D1"], gap=world["gap"], ball_source=world["ball_source"], times=times,
            world_json_sha256=LI.sha256_file(ctx.data_root / f"seed{int(seed)}" / "world.json")))


# ------------------------------------------------------------------------------------------------ host

def stage_host(ctx, seeds, kinds=tuple(HOST_KINDS), updates=None):
    """Plain TD3+BC per seed on the gapped log ('gap') and the full log ('full'), both with the gapped
    normalisation rule (data.prepare)."""
    spec = spec_digest(ctx)
    jobs = [(s, k) for s in seeds for k in kinds if finished(host_dir(ctx, k, s) / "record.json", spec) is None]
    for s in sorted({s for s, _ in jobs}):
        require(data_record_path(ctx, s), spec, f"data for seed {s}")
    for seed in sorted({s for s, _ in jobs}):
        data = load_data(ctx, seed, spec)
        for kind in [k for s, k in jobs if s == seed]:
            started = time.perf_counter()
            prep = D.prepare(data, HOST_KINDS[kind], ctx.cfg)
            d = host_dir(ctx, kind, seed)
            result, _, _ = LI.train_host(prep, ctx.cfg, seed, updates=updates, out_dir=d, log=ctx.log,
                                         name=f"TD3+BC host ({kind})")
            write_record(d / "record.json", stage_record("host", spec, started, quality=ctx.quality, seed=int(seed),
                                                         host=kind, removal=HOST_KINDS[kind], result=result))
            ctx.log(f"host {ctx.quality} {kind} seed {seed}: {'FAILED' if result['failed'] else 'finished'} in "
                    f"{result['seconds']:.0f}s")


# ------------------------------------------------------------------------------------------------ room check

def stage_room(ctx, seeds):
    """The room check (no BCA) on the frozen gapped host critic; writes the lock before any return exists."""
    spec = spec_digest(ctx)
    todo = [s for s in seeds if finished(room_dir(ctx, s) / "record.json", spec) is None]
    for s in todo:
        require(data_record_path(ctx, s), spec, f"data for seed {s}")
        require_host(ctx, "gap", s, spec)
    for seed in todo:
        started = time.perf_counter()
        data = load_data(ctx, seed, spec)
        prep = D.prepare(data, "gap", ctx.cfg)
        host = load_host_state(ctx, "gap", seed, spec)
        params, rec = LI.room_check(prep, host.critic.params, ctx.cfg, seed, log=ctx.log)
        d = room_dir(ctx, seed)
        sha = LR.save_tree(d / "actors.msgpack", params)
        write_record(d / "record.json", stage_record(
            "room", spec, started, quality=ctx.quality, seed=int(seed), room=rec, actors_sha256=sha,
            inputs_sha256=links(ctx, [host_dir(ctx, "gap", seed) / "record.json"]),
            about="the room check's lock: every arm trained and even caution matched in strength to the placed arm, "
                  "written before any return is computed"))
        ctx.log(f"room seed {seed}: finished in {time.perf_counter() - started:.0f}s")


def load_room_actors(ctx, seed, spec):
    """The room check's record and actors of a seed ('none' is the no-caution actor check (e) reads and strengths are
    measured from)."""
    rec = require(room_dir(ctx, seed) / "record.json", spec, f"the room check of seed {seed}")
    return rec, LR.load_tree(room_dir(ctx, seed) / "actors.msgpack", LI.room_template(rec["room"]["names"]),
                             rec["actors_sha256"])


def stage_room_lower_edge(ctx, seeds):
    """The lower edge's room check (no BCA; spec.json lower_edge.room_check) on the frozen gapped host critic, from the
    room check's clone: check (e) on the exact oracle penalty first (a failure stops the seed), then the oracle arms
    and the even pull matched in strength; writes the lock before any return exists."""
    spec = spec_digest(ctx)
    le = spec_lower_edge(frozen_spec(ctx, spec))
    todo = [s for s in seeds if finished(room_le_dir(ctx, s) / "record.json", spec) is None]
    for s in todo:
        require(data_record_path(ctx, s), spec, f"data for seed {s}")
        require_host(ctx, "gap", s, spec)
        require(room_dir(ctx, s) / "record.json", spec, f"the room check of seed {s}")
    for seed in todo:
        started = time.perf_counter()
        data = load_data(ctx, seed, spec)
        prep = D.prepare(data, "gap", ctx.cfg)
        host = load_host_state(ctx, "gap", seed, spec)
        room_rec, room_params = load_room_actors(ctx, seed, spec)
        params, rec = LE.room_check(prep, host.critic.params, room_params, room_rec["room"], ctx.cfg, seed, le["rule"],
                                    log=ctx.log)
        d = room_le_dir(ctx, seed)
        sha = LR.save_tree(d / "actors.msgpack", params) if params else None
        write_record(d / "record.json", stage_record(
            "room_lower_edge", spec, started, quality=ctx.quality, seed=int(seed), room_lower_edge=rec,
            actors_sha256=sha, inputs_sha256=links(ctx, [host_dir(ctx, "gap", seed) / "record.json",
                                                         room_dir(ctx, seed) / "record.json"]),
            about="the lower edge's room check (no BCA): its lock, every arm trained and the even pull matched in "
                  "strength, written before any return is computed"))
        ctx.log(f"room-lower-edge {ctx.quality} seed {seed}: "
                f"{'STOPPED by check (e)' if rec['stopped'] else 'finished'} in {time.perf_counter() - started:.0f}s")


# ------------------------------------------------------------------------------------------------ band

def exact_arms(data, prep, kind):
    """The single-threshold exact arms' (bank weights, test mass): w(c) = p_tilt(c) / p_train(c) at the kept bank
    rows; the gapped test mass by quadrature for the gapped host, the closed form for the full host. 'none' is
    weights 1 and test mass 1 exactly (w = 1 everywhere under no tilt)."""
    c_bank = data.train_log.current[prep.bank_rows]
    arms = {"exact_none_single": (np.ones(len(c_bank)), 1.0)}
    for tilt in ("mild", "strong"):
        arms[f"exact_{tilt}_single"] = (W.ratio(c_bank, tilt), data.test_mass(tilt, gapped=(kind == "gap")))
    return arms


def classifiers_of(data):
    """The estimated-weight classifier of every tilt (classifier.fit_samples: observations only; 'none' fits the
    training-law sample against the second one). Deterministic: the band and coverage stages fit the same ones."""
    return {tilt: CL.fit_samples(data.samples, tilt) for tilt in W.TILT_NAMES}


def estimated_arms(data, prep, clfs=None):
    """The single-threshold estimated arms of the gapped host: each tilt's classifier's weights at the kept bank rows
    and the test mass over the test-law sample's steps outside the ball. Returns (arms, report, classifiers)."""
    c_bank = data.train_log.current[prep.bank_rows]
    bank_obs = np.asarray(data.train_log["observations"][prep.bank_rows])
    clfs = classifiers_of(data) if clfs is None else clfs
    arms, rep = {}, {}
    for tilt in W.TILT_NAMES:
        clf = clfs[tilt]
        other = "train2" if tilt == "none" else tilt
        est = CL.estimated_arm(clf, bank_obs, data.samples[other]["observations"], data.samples[other]["actions"],
                               data.gap)
        arms[f"estimated_{tilt}_single"] = (est["bank_weights"], est["test_mass"])
        rep[tilt] = dict(classifier=clf.describe(), compare=CL.compare(clf, bank_obs, c_bank, tilt),
                         test_mass=est["test_mass"], test_sample_in_ball_share=est["test_sample_in_ball_share"])
    return arms, rep, clfs


def bank_weights(data, prep, arm, tilt, clfs=None):
    """A weighted arm's weights at the kept bank rows: exact w(c), or the tilt's classifier's weight."""
    if arm == "exact":
        return W.ratio(data.train_log.current[prep.bank_rows], tilt)
    return clfs[tilt].weight(np.asarray(data.train_log["observations"][prep.bank_rows], np.float64))


def step_masses(data, prep, kind, arm, tilt, clfs=None):
    """Each step's own test mass in a per-step arm (exact w(c), or the tilt's classifier's weight at the observation):
    at the kept training rows ('train': R2 and the BCA arms read their doses there) and at the tilt's test steps the
    arm certifies ('test', at the steps where 'test_mask' is True: outside the ball on the gapped host, every step on
    the full host; a weighted arm refuses in-ball steps before any threshold)."""
    log, rows = data.test(tilt)
    mask = np.ones(len(rows), bool) if kind == "full" else ~np.asarray(log["truth/in_ball"][rows], bool)
    if arm == "exact":
        train = W.ratio(data.train_log.current[prep.rows], tilt)
        test = W.ratio(log.current[rows[mask]], tilt)
    elif arm == "estimated":
        obs = lambda lg, r: np.asarray(lg["observations"][r], np.float64)
        train = clfs[tilt].weight(obs(data.train_log, prep.rows))
        test = clfs[tilt].weight(obs(log, rows[mask]))
    else:
        raise ValueError(f"unknown per-step arm {arm!r}")
    return dict(train=np.asarray(train, np.float64), test=np.asarray(test, np.float64), test_mask=mask)


def high_current_rows(data, prep, share):
    """Kept training rows in the top `share` of current (R2's high-current rows)."""
    c = data.train_log.current[prep.rows]
    return c >= np.quantile(c, 1.0 - share)


def require_design(ctx, spec, body):
    """The stop rule of spec.json compute.order ('stop here if D1 fails'), enforced in code: refuse unless a finished
    report under this specification, made by the code running now, covers every seed of the specification and says
    D1 passed. Returns {report_sha256, D1 status} for the band record (not a link: later reports rewrite the file)."""
    rep = require(report_path(ctx), spec, "the first report (BCA's band is fitted only after it says D1 passed)")
    want = spec_seeds(body)
    if sorted(rep.get("seeds") or []) != want:
        raise Refused(f"refused: BCA's band needs a report on seeds {want} that says D1 passed; the report covers "
                      f"{rep.get('seeds')}")
    if rep.get("code_sha256") != LI.code_hashes():
        raise Refused("refused: the report was made by other code than the code running now; run the report again")
    status = _get(rep, "marks", "D1", "status")
    if status != "pass":
        raise Refused(f"refused: BCA's band is fitted only after the design check passes; the report says D1 is "
                      f"{status} (stop: spec.json compute.order)")
    return dict(report_sha256=LI.sha256_file(report_path(ctx)), D1=status)


def stage_band(ctx, seeds, kinds=tuple(HOST_KINDS)):
    """BCA's band on each frozen host critic (approval 'band', and only after a report on every seed, made by this
    code, says D1 passed: require_design): on the same scores and stream, the single threshold of every arm and the
    per-step table of R of every per-step arm (its grid covers every mass looked up: the kept training rows and the
    tilt's certified test steps); mark W0 (unit check, and the no-tilt table is the unweighted threshold); mark M3's
    statistics (the dose's largest fall as the width rises, kept training rows); the doses of every arm on the
    kept training rows (R2: per step, each row at its own threshold, with refusal at 1.5; the single threshold beside
    it); the lower edge's scale change R_i / R_u - 1 of every per-step table on the kept training rows (L2); and, on
    the gapped host, L2's proxy check (lower_edge.proxy_check with the room check's no-caution actor; reported beside
    L2, before any lower-edge BCA training), so the gapped band also needs the room check of its seed."""
    spec = spec_digest(ctx)
    approval = approval_for(ctx, "band")
    body = frozen_spec(ctx, spec)
    per_step = spec_per_step(body)
    le_rule = spec_lower_edge(body)["rule"]
    jobs = [(s, k) for s in seeds for k in kinds if finished(band_dir(ctx, k, s) / "record.json", spec) is None]
    for s, k in jobs:
        require(data_record_path(ctx, s), spec, f"data for seed {s}")
        require_host(ctx, k, s, spec)
        if k == "gap":
            require(room_dir(ctx, s) / "record.json", spec, f"the room check of seed {s} (the proxy check's actor)")
    design = require_design(ctx, spec, body) if jobs else None
    for seed in sorted({s for s, _ in jobs}):
        data = load_data(ctx, seed, spec)
        for kind in [k for s, k in jobs if s == seed]:
            started = time.perf_counter()
            prep = D.prepare(data, HOST_KINDS[kind], ctx.cfg)
            host = load_host_state(ctx, kind, seed, spec)
            band = LI.fit_band(host, prep, ctx.cfg, seed, approval, log=ctx.log)
            d = band_dir(ctx, kind, seed)
            inputs = [host_dir(ctx, kind, seed) / "record.json"] + \
                ([room_dir(ctx, seed) / "record.json"] if kind == "gap" else [])
            fields = dict(quality=ctx.quality, seed=int(seed), host=kind, band=band.record, design_check=design,
                          inputs_sha256=links(ctx, inputs))
            if not band.record["refresh"]["posterior_inputs_valid"]:
                write_record(d / "record.json", stage_record("band", spec, started, band_valid=False, **fields))
                ctx.log(f"band {kind} seed {seed}: the refresh was invalid")
                continue
            scores, stream = LI.bank_scores(band, prep.bank, ctx.cfg)
            arms = {"unweighted": None, **exact_arms(data, prep, kind)}
            classifiers, clfs = None, None
            if kind == "gap":
                est, classifiers, clfs = estimated_arms(data, prep)
                arms.update(est)
            cals = LI.thresholds(band, scores, stream, arms)
            unit_ok, _, _ = CV.unit_check(scores, stream)
            R_u = cals["unweighted"].threshold
            thresholds = {name: dict(LI.threshold_record(cal, *(arms[name] or (None, None))),
                                     ratio_to_unweighted=(cal.threshold / R_u if math.isfinite(R_u) else None))
                          for name, cal in cals.items()}
            reads = {name: LI.readout(band, LI.reference_of(band, cal), prep.train) for name, cal in cals.items()}
            u = reads["unweighted"]
            high = high_current_rows(data, prep, per_step["high_current_share"])
            doses = {name: dict(LI.dose_change(r["dose"], u["dose"], high), support_share=float(r["support"].mean()),
                                valid=r["valid"]) for name, r in reads.items()}
            # the per-step tables: one grid per arm and tilt, covering every mass that is looked up
            masses, table_arms = {}, {}
            for arm in PER_STEP_ARMS if kind == "gap" else ("exact",):
                for tilt in W.TILT_NAMES:
                    m = masses[f"{arm}_{tilt}"] = step_masses(data, prep, kind, arm, tilt, clfs)
                    grid = CV.mass_grid(np.concatenate([m["train"], m["test"]]), per_step["grid_ratio"])
                    table_arms[f"{arm}_{tilt}"] = (bank_weights(data, prep, arm, tilt, clfs), grid)
            tabs = LI.threshold_tables(band, scores, stream, table_arms)
            tables, doses_per_step, lower_edge_scale = {}, {}, {}
            for name, tab in tabs.items():
                m = masses[name]
                R_train, i_train = CV.lookup(tab, m["train"])
                R_test, i_test = CV.lookup(tab, m["test"])
                lower_edge_scale[name] = LE.scale_change(R_train, tab.thresholds, R_u, high)   # mark L2
                rd = LI.per_step_readout(band, R_train, prep.train)
                doses_per_step[name] = dict(LI.dose_change(rd["dose"], u["dose"], high),
                                            refused_rows=int(rd["refused"].sum()),
                                            support_share=float(rd["support"].mean()), valid=rd["valid"])
                tables[name] = dict(tab.summary(R_u), kish=CV.kish(table_arms[name][0]),
                                    masses=dict(train=[float(m["train"].min()), float(m["train"].max())],
                                                test=[float(m["test"].min()), float(m["test"].max())] if
                                                len(m["test"]) else None),
                                    train_rows=dict(CV.per_step_summary(R_train, R_u),
                                                    lookup=CV.lookup_gaps(tab, m["train"], i_train)),
                                    test_steps=dict(CV.per_step_summary(R_test, R_u),
                                                    lookup=CV.lookup_gaps(tab, m["test"], i_test)))
            none_tab = tabs["exact_none"]
            w0_per_step = bool(none_tab.grid.tolist() == [1.0] and none_tab.thresholds[0] == R_u
                               and none_tab.raw[0] == R_u)
            m3 = dict(CV.dose_order(u["width"], u["dose"], band.unit), supported_all=bool(u["support"].all()),
                      threshold_finite=bool(math.isfinite(R_u)))
            proxy = None
            if kind == "gap":      # reported beside L2: BCA's unweighted lower edge against the oracle's (no training)
                _, room_params = load_room_actors(ctx, seed, spec)
                proxy = LE.proxy_check(band, prep, host.critic.params, room_params["none"], R_u, seed, le_rule.check_e,
                                       stage="band")
            sha = LR.save_tree(d / "band.msgpack", band.state)
            write_record(d / "record.json", stage_record(
                "band", spec, started, band_valid=True, band_sha256=sha, thresholds=thresholds, tables=tables,
                W0_unit_check=unit_ok, W0_per_step_none=w0_per_step, M3=m3, doses=doses,
                doses_per_step=doses_per_step, lower_edge_scale=lower_edge_scale, lower_edge_proxy=proxy,
                high_current_share=per_step["high_current_share"],
                grid_ratio=per_step["grid_ratio"], classifiers=classifiers,
                bank_scores_sha256=LR.tree_sha256(scores), **fields))
            ctx.log(f"band {kind} seed {seed}: R unweighted {R_u:.4g}, exact strong single "
                    f"{cals['exact_strong_single'].threshold:.4g}, per step {tables['exact_strong']['test_steps']}, "
                    f"finished in {time.perf_counter() - started:.0f}s")


# ------------------------------------------------------------------------------------------------ coverage

def stage_coverage(ctx, seeds, kinds=tuple(HOST_KINDS)):
    """Coverage of every arm (approval 'coverage') on the held-out training-law steps (tilt 'none') and the test
    logs, outside and inside the ball (weighted arms refuse inside the ball on the gapped host): per-step arms at each
    step's own threshold from the band's tables (a step whose threshold is +inf is refused), single arms at their one
    threshold; the no-tilt per-step arm equals the unweighted arm on every certified step (mark W0); the forecast;
    and, on the gapped host, mark M2 on the 5,000 truth rows (exact critic error by learners.exact_q)."""
    spec = spec_digest(ctx)
    approval = approval_for(ctx, "coverage")
    jobs = [(s, k) for s in seeds for k in kinds if finished(coverage_dir(ctx, k, s) / "record.json", spec) is None]
    for s, k in jobs:
        require(data_record_path(ctx, s), spec, f"data for seed {s}")
        require_host(ctx, k, s, spec)
        require_band(ctx, k, s, spec)
    for seed in sorted({s for s, _ in jobs}):
        data = load_data(ctx, seed, spec)
        for kind in [k for s, k in jobs if s == seed]:
            started = time.perf_counter()
            brec = require_band(ctx, kind, seed, spec)
            prep = D.prepare(data, HOST_KINDS[kind], ctx.cfg)
            host = load_host_state(ctx, kind, seed, spec)
            band = LI.load_band(band_dir(ctx, kind, seed) / "band.msgpack", brec["band_sha256"], host, prep, ctx.cfg,
                                seed, approval, brec["band"])
            clfs = classifiers_of(data) if kind == "gap" else None
            if clfs is not None:
                for t, clf in clfs.items():
                    if plain(clf.describe()) != brec["classifiers"][t]["classifier"]:
                        raise Refused(f"refused: the {t} classifier of seed {seed} differs from the band's")
            R_u = num(brec["thresholds"]["unweighted"]["threshold"])
            tilts, arrays = {}, {}
            for ti, tilt in enumerate(W.TILT_NAMES):
                log, rows = data.test(tilt)
                trans = LI.transitions(log, rows, prep)
                y, q = LI.critic_parts(host, trans, ctx.cfg, seed, LI.coverage_key(ctx.cfg, seed, ti))
                in_ball = np.asarray(log["truth/in_ball"][rows], bool)
                episodes = np.asarray(log["infos/episode"][rows])
                arms = {}
                for arm in TEST_ARMS[kind]:
                    base = arm[:-len("_single")] if arm.endswith("_single") else arm
                    refused = CV.refusals(in_ball, base) if kind == "gap" else None
                    extra = {}
                    if arm in PER_STEP_ARMS:
                        name = f"{arm}_{tilt}"
                        tab = CV.table_from_summary(brec["tables"][name])
                        m = step_masses(data, prep, kind, arm, tilt, clfs)
                        R_test, idx = CV.lookup(tab, m["test"])
                        R_steps = np.full(len(rows), np.inf)          # in-ball steps are refused by the ball first
                        R_steps[m["test_mask"]] = R_test
                        rd = LI.per_step_readout(band, R_steps, trans, stage="coverage")
                        arrays[f"{tilt}/{arm}/R"] = R_steps
                        extra = dict(per_step=dict(CV.per_step_summary(R_test, R_u),
                                                   lookup=CV.lookup_gaps(tab, m["test"], idx)),
                                     refused_by_threshold_share_certifiable=float(np.mean(np.isinf(R_test)))
                                     if len(R_test) else None)
                        threshold = "per step"
                    else:
                        name = "unweighted" if arm == "unweighted" else f"{base}_{tilt}_single"
                        t = brec["thresholds"][name]
                        rd = LI.readout(band, LI.reference_of(band, t), trans, stage="coverage")
                        threshold = t["threshold"]
                    o = CV.outcomes(y, q, rd["width"], refused)
                    arms[arm] = dict(threshold_arm=name, threshold=threshold, valid=rd["valid"],
                                     groups=CV.groups(o, episodes, in_ball),
                                     hard_coverage_mismatches=hard_coverage_mismatches(o, y, q, rd["width"]), **extra)
                    arrays[f"{tilt}/{arm}/width"] = rd["width"]
                    arrays[f"{tilt}/{arm}/outcome"] = o
                if tilt == "none":   # mark W0: under no tilt the per-step arm is the unweighted arm on certified steps
                    keep = ~in_ball if kind == "gap" else np.ones(len(rows), bool)
                    same = (arrays["none/exact/width"][keep].tobytes() == arrays["none/unweighted/width"][keep].tobytes()
                            and np.array_equal(arrays["none/exact/outcome"][keep],
                                               arrays["none/unweighted/outcome"][keep]))
                    arms["exact"]["W0_per_step_none_identical"] = bool(same)
                arrays.update({f"{tilt}/y": y, f"{tilt}/q": q, f"{tilt}/in_ball": in_ball,
                               f"{tilt}/current": log.current[rows], f"{tilt}/episode": episodes})
                tilts[tilt] = dict(steps=int(len(rows)), episodes=int(len(np.unique(episodes))),
                                   in_ball_share=float(in_ball.mean()), in_ball_steps=int(in_ball.sum()), arms=arms)
            outside = ~arrays["none/in_ball"]
            c_none = arrays["none/current"][outside]
            covered_none = arrays["none/unweighted/outcome"][outside] == CV.COVERED
            edges, means, counts = CV.by_current(c_none, covered_none)
            forecast = dict(rule="unweighted coverage under a tilt = the w(c)-weighted mean of no-tilt held-out "
                                 "steps outside the ball (coverage.forecast)",
                            values={t: CV.forecast(c_none, covered_none, t) for t in ("mild", "strong")},
                            coverage_by_current=dict(edges=edges, coverage=means, steps=counts))
            m2 = mechanics_m2(data, prep, host, band, ctx.cfg, seed) if kind == "gap" else None
            d = coverage_dir(ctx, kind, seed)
            d.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(d / "steps.npz", **{k.replace("/", "__"): v for k, v in arrays.items()})
            write_record(d / "record.json", stage_record(
                "coverage", spec, started, quality=ctx.quality, seed=int(seed), host=kind, tilts=tilts,
                forecast=forecast, M2=m2,
                steps_npz_sha256=LI.sha256_file(d / "steps.npz"),
                inputs_sha256=links(ctx, [band_dir(ctx, kind, seed) / "record.json",
                                          host_dir(ctx, kind, seed) / "record.json"])))
            ctx.log(f"coverage {kind} seed {seed}: finished in {time.perf_counter() - started:.0f}s")


def hard_coverage_mismatches(outcome, y, q, width):
    """Certified steps where coverage.outcomes (float64 comparison of the float32 numbers) and BCA's own
    calibration.network.hard_coverage (float32) disagree; expected 0 or a handful of exact ties."""
    hc = np.asarray(hard_coverage(jnp.asarray(np.float32(y)), jnp.asarray(np.float32(q)),
                                  jnp.asarray(np.float32(width))), bool)
    certified = outcome != CV.REFUSED
    return int(np.sum((outcome[certified] == CV.COVERED) != hc[certified]))


def float_radius(states):
    """The length of a step's 10 float numbers (5 positions, 5 speeds)."""
    s = np.asarray(states, np.float64)[:, :10]
    return np.sqrt(np.sum(s * s, axis=1))


def decile_partial_spearman(width, error, control, c, deciles=10):
    """Reported beside mark M2 (spec version 5): inside each current decile (as coverage.decile_spearman), the partial
    rank correlation of width and error given `control` (the float radius), (r_UE - r_UF r_EF) /
    sqrt((1 - r_UF^2)(1 - r_EF^2)) from the three Spearman correlations, averaged over the deciles."""
    width, error, control, c = (np.asarray(v, np.float64) for v in (width, error, control, c))
    parts = np.array_split(np.argsort(c, kind="stable"), int(deciles))
    per = []
    for p in parts:
        r_ue = stats.spearmanr(width[p], error[p])[0]
        r_uf = stats.spearmanr(width[p], control[p])[0]
        r_ef = stats.spearmanr(error[p], control[p])[0]
        den = math.sqrt(max((1.0 - r_uf ** 2) * (1.0 - r_ef ** 2), 0.0))
        per.append(float((r_ue - r_uf * r_ef) / den) if den > 0 else math.nan)
    return dict(within_deciles=float(np.mean(per)), per_decile=per)


def mechanics_m2(data, prep, host, band, cfg, seed):
    """Mark M2's statistics on the truth rows (held-out steps outside the ball): Spearman of the unweighted width
    with the exact critic error |q - Q^pi| and with the noise map, inside current deciles and pooled; and, reported
    (spec version 5), the within-decile partial correlation of width and error given the float radius."""
    rows = np.asarray(data.truth["rows"], np.int64)
    log = data.train_log
    trans = LI.transitions(log, rows, prep)
    _, q = LI.critic_parts(host, trans, cfg, seed, LI.coverage_key(cfg, seed, 99))
    U = LI.readout(band, band.state.posterior, trans, stage="coverage")["width"].astype(np.float64)
    states = np.asarray(log["observations"][rows], np.float32).astype(np.float64)
    actions = np.asarray(log["actions"][rows], np.float32).astype(np.float64)
    err, true = LI.exact_error(host.actor.params, prep, states, actions, q,
                               chunk=min(int(cfg.row_chunk), max(len(rows), 1)))
    c = log.current[rows]
    radius = float_radius(log["infos/state"][rows])
    finite = np.isfinite(U)
    enough = finite.sum() > 20
    return dict(rows=int(len(rows)), finite_width_rows=int(finite.sum()),
                error=CV.decile_spearman(U[finite], err[finite], c[finite]) if enough else None,
                noise_map=CV.decile_spearman(U[finite], np.asarray(data.truth["noise_map"])[finite], c[finite])
                if enough else None,
                error_given_float_radius=decile_partial_spearman(U[finite], err[finite], radius[finite], c[finite])
                if enough else None,
                mean_float_radius=float(radius.mean()),
                mean_exact_error=float(err.mean()), mean_q=float(q.mean()), mean_true=float(true.mean()))


# ------------------------------------------------------------------------------------------------ BCA arms

def bca_weighting(data, prep, arm, grid_ratio):
    """A BCA arm's weighting: None (unweighted); per row ('exact_strong', 'estimated_strong': each training row's own
    weight is its test mass, on a grid of ratio grid_ratio covering every row's mass); or single threshold
    ('exact_strong_single': the gapped test mass by quadrature, version 3's arm, kept as a comparison)."""
    if arm == "unweighted":
        return None
    if arm == "exact_strong_single":
        return dict(name="exact strong (single threshold)", mode="single",
                    weights=bank_weights(data, prep, "exact", "strong"), test_mass=data.test_mass("strong"))
    if arm in ("exact_strong", "estimated_strong"):
        base = arm[:-len("_strong")]
        clfs = classifiers_of(data) if base == "estimated" else None
        masses = step_masses(data, prep, "gap", base, "strong", clfs)["train"]
        return dict(name=f"{base} strong (per row)", mode="per_row",
                    weights=bank_weights(data, prep, base, "strong", clfs), row_masses=masses,
                    grid=CV.mass_grid(masses, grid_ratio))
    raise ValueError(f"unknown BCA arm {arm!r}")


def stage_bca(ctx, seeds, arms=BCA_ARMS):
    """The capped pull's co-trained BCA arms (approval 'bca' for this quality, the owner's go for this kind of arm;
    the optional estimated-weighted arm also needs 'bca_estimated'), only after a report on all of the
    specification's seeds, made by the code running now, says M1, R1 and R2 passed:
    unweighted, exact-weighted strong per row (each row's dose at its own threshold; refusal 1.5) and exact-weighted
    strong single threshold (version 3's, a comparison)."""
    spec = spec_digest(ctx)
    for arm in arms:
        if arm not in BCA_ARMS + BCA_OPTIONAL_ARMS:
            raise Refused(f"refused: unknown BCA arm {arm!r}")
    approval = approval_for(ctx, "bca")
    if any(a in BCA_OPTIONAL_ARMS for a in arms):
        approval_for(ctx, "bca_estimated")
    rep = require(report_path(ctx), spec, "the report")
    body = frozen_spec(ctx, spec)
    want = spec_seeds(body)
    grid_ratio = spec_per_step(body)["grid_ratio"]
    if sorted(rep.get("seeds") or []) != want:
        raise Refused(f"refused: the BCA arms need a report on seeds {want}; the report covers {rep.get('seeds')}")
    if rep.get("code_sha256") != LI.code_hashes():
        raise Refused("refused: the report was made by other code than the code running now; run the report again")
    for mark in ("M1", "R1", "R2"):
        status = rep["marks"].get(mark, {}).get("status")
        if status != "pass":
            raise Refused(f"refused: the BCA arms run only after M1, R1 and R2 pass; the report says {mark} is "
                          f"{status}")
    jobs = [(s, a) for s in seeds for a in arms if finished(bca_dir(ctx, a, s) / "record.json", spec) is None]
    for s, _ in jobs:
        require(data_record_path(ctx, s), spec, f"data for seed {s}")
    for seed in sorted({s for s, _ in jobs}):
        data = load_data(ctx, seed, spec)
        prep = D.prepare(data, "gap", ctx.cfg)
        for arm in [a for s, a in jobs if s == seed]:
            started = time.perf_counter()
            d = bca_dir(ctx, arm, seed)
            result, _, _ = LI.bca_cotrain(prep, ctx.cfg, seed, approval,
                                          weighting=bca_weighting(data, prep, arm, grid_ratio),
                                          out_dir=d, log=ctx.log, name=f"BCA {arm}",
                                          stage="bca_estimated" if arm in BCA_OPTIONAL_ARMS else "bca")
            write_record(d / "record.json", stage_record("bca", spec, started, quality=ctx.quality, seed=int(seed),
                                                         arm=arm, result=result,
                                                         report_sha256=LI.sha256_file(report_path(ctx))))


def lower_edge_arm(data, prep, brec, arm, R_u, clfs=None):
    """A lower-edge BCA arm's thresholds on the kept training rows (lower_edge.band_arms): 'unweighted' R_u on every
    row; a per-step arm each row's own threshold from the band record's table (the smallest grid mass at least the
    row's weight), with the table's thresholds as the pool of its largest finite one."""
    if arm == "unweighted":
        return dict(name=arm, R=np.full(prep.n, R_u), pool=[R_u])
    base, tilt = arm.split("_", 1)
    tab = CV.table_from_summary(brec["tables"][f"{base}_{tilt}"])
    R = CV.lookup(tab, step_masses(data, prep, "gap", base, tilt, clfs)["train"])[0]
    return dict(name=arm, R=R, pool=tab.thresholds)


def stage_bca_lower_edge(ctx, seeds, arms=LE_ARMS):
    """The lower edge's BCA arms (approval 'bca_lower_edge' for this quality, the owner's go for this kind of arm,
    separate from the capped pull's 'bca'; the optional estimated arm also needs 'bca_lower_edge_estimated'), only after
    a report on all of the specification's seeds, made by the code running now, says M1, L1 and L2 passed: actor-only
    against the frozen gapped critic from the cloned actor, the penalty k [U_i(s, pi(s)) - U_i(s, a_i)]_+ read from
    the frozen gapped band (lower_edge.band_arms), each arm's R_i from the band record (unweighted, or each row's own
    per-step threshold). Check (e) on each arm's penalty is reported."""
    spec = spec_digest(ctx)
    for arm in arms:
        if arm not in LE_ARMS + LE_OPTIONAL_ARMS:
            raise Refused(f"refused: unknown lower-edge BCA arm {arm!r}")
    approval = approval_for(ctx, LE.SCOPE)
    optional = any(a in LE_OPTIONAL_ARMS for a in arms)
    if optional:
        approval_for(ctx, LE.SCOPE_OPTIONAL)
    rep = require(report_path(ctx), spec, "the report")
    body = frozen_spec(ctx, spec)
    le = spec_lower_edge(body)
    want = spec_seeds(body)
    if sorted(rep.get("seeds") or []) != want:
        raise Refused(f"refused: the lower edge's BCA arms need a report on seeds {want}; the report covers "
                      f"{rep.get('seeds')}")
    if rep.get("code_sha256") != LI.code_hashes():
        raise Refused("refused: the report was made by other code than the code running now; run the report again")
    for mark in ("M1", "L1", "L2"):
        status = _get(rep, "marks", mark, "status")
        if status != "pass":
            raise Refused(f"refused: the lower edge's BCA arms run only after M1, L1 and L2 pass; the report says "
                          f"{mark} is {status}")
    jobs = [(s, a) for s in seeds for a in arms if finished(bca_le_dir(ctx, a, s) / "record.json", spec) is None]
    for s in sorted({s for s, _ in jobs}):
        require(data_record_path(ctx, s), spec, f"data for seed {s}")
        require_host(ctx, "gap", s, spec)
        require_band(ctx, "gap", s, spec)
        require(room_dir(ctx, s) / "record.json", spec, f"the room check of seed {s}")
    for seed in sorted({s for s, _ in jobs}):
        started = time.perf_counter()
        todo = [a for s, a in jobs if s == seed]
        data = load_data(ctx, seed, spec)
        prep = D.prepare(data, "gap", ctx.cfg)
        host = load_host_state(ctx, "gap", seed, spec)
        brec = require_band(ctx, "gap", seed, spec)
        band = LI.load_band(band_dir(ctx, "gap", seed) / "band.msgpack", brec["band_sha256"], host, prep, ctx.cfg, seed,
                            approval, brec["band"], stage=LE.SCOPE)
        clfs = None
        if any(a.startswith("estimated") for a in todo):
            clfs = classifiers_of(data)
            for t, clf in clfs.items():
                if plain(clf.describe()) != brec["classifiers"][t]["classifier"]:
                    raise Refused(f"refused: the {t} classifier of seed {seed} differs from the band's")
        R_u = num(brec["thresholds"]["unweighted"]["threshold"])
        specs = [lower_edge_arm(data, prep, brec, a, R_u, clfs) for a in todo]
        _, room_params = load_room_actors(ctx, seed, spec)
        try:
            params, rec = LE.band_arms(prep, band, host.critic.params, specs, le["k"], ctx.cfg, seed, approval,
                                       none_params=room_params["none"], check_e_rule=le["rule"].check_e,
                                       stage=LE.SCOPE_OPTIONAL if any(a in LE_OPTIONAL_ARMS for a in todo)
                                       else LE.SCOPE, log=ctx.log)
        except ValueError as error:
            raise Refused(f"refused: the lower edge's BCA arms of seed {seed}: {error}") from error
        inputs = links(ctx, [band_dir(ctx, "gap", seed) / "record.json", host_dir(ctx, "gap", seed) / "record.json",
                             room_dir(ctx, seed) / "record.json"])
        shared = {k: v for k, v in rec.items() if k != "arms"}
        for arm in todo:
            d = bca_le_dir(ctx, arm, seed)
            sha = LR.save_tree(d / "actor.msgpack", params[arm])
            write_record(d / "record.json", stage_record(
                "bca_lower_edge", spec, started, quality=ctx.quality, seed=int(seed), arm=arm,
                result=dict(shared, arm=rec["arms"][arm], arms_in_this_job=todo), actor_sha256=sha,
                report_sha256=LI.sha256_file(report_path(ctx)), inputs_sha256=inputs))
        ctx.log(f"bca-lower-edge {ctx.quality} seed {seed}: {todo} finished in {time.perf_counter() - started:.0f}s")


# ------------------------------------------------------------------------------------------------ evaluate

def controller_records(ctx, group, seed, spec):
    """The record files of a group's finished controllers (refuses when a required one is missing)."""
    if group == "hosts":
        for kind in HOST_KINDS:
            require_host(ctx, kind, seed, spec)
        return [host_dir(ctx, kind, seed) / "record.json" for kind in HOST_KINDS]
    if group == "room":
        require(room_dir(ctx, seed) / "record.json", spec, f"the room check of seed {seed}")
        return [room_dir(ctx, seed) / "record.json"]
    if group == "room_lower_edge":
        require(room_le_dir(ctx, seed) / "record.json", spec, f"the lower edge's room check of seed {seed}")
        return [room_le_dir(ctx, seed) / "record.json"]
    if group == "bca_lower_edge":
        out = []
        for arm in LE_ARMS + LE_OPTIONAL_ARMS:
            path = bca_le_dir(ctx, arm, seed) / "record.json"
            if finished(path, spec) is not None:
                out.append(path)
            elif arm in LE_ARMS:
                raise Refused(f"refused: the lower edge's BCA arm {arm} of seed {seed} has not finished")
        return out
    if group == "bca":
        out = []
        for arm in BCA_ARMS + BCA_OPTIONAL_ARMS:
            path = bca_dir(ctx, arm, seed) / "record.json"
            if finished(path, spec) is not None:
                out.append(path)
            elif arm in BCA_ARMS:
                raise Refused(f"refused: the BCA arm {arm} of seed {seed} has not finished")
        return out
    raise Refused(f"refused: unknown evaluation group {group!r}")


def controllers_of(ctx, group, seed, spec):
    """{name: actor params} of a group's finished controllers."""
    controller_records(ctx, group, seed, spec)
    if group == "hosts":
        return {f"plain_{kind}": load_host_state(ctx, kind, seed, spec).actor.params for kind in HOST_KINDS}
    if group == "room":
        rec = require(room_dir(ctx, seed) / "record.json", spec, f"the room check of seed {seed}")
        return LR.load_tree(room_dir(ctx, seed) / "actors.msgpack", LI.room_template(rec["room"]["names"]),
                            rec["actors_sha256"])
    if group == "room_lower_edge":     # no actors when check (e) stopped the seed
        rec = require(room_le_dir(ctx, seed) / "record.json", spec, f"the lower edge's room check of seed {seed}")
        names = rec["room_lower_edge"]["names"]
        return (LR.load_tree(room_le_dir(ctx, seed) / "actors.msgpack", LE.room_template(names), rec["actors_sha256"])
                if names else {})
    if group == "bca_lower_edge":
        out = {}
        for arm in LE_ARMS + LE_OPTIONAL_ARMS:
            rec = finished(bca_le_dir(ctx, arm, seed) / "record.json", spec)
            if rec is not None:
                out[f"le_bca_{arm}"] = LR.load_tree(bca_le_dir(ctx, arm, seed) / "actor.msgpack", LR.actor_template(),
                                                    rec["actor_sha256"])
        return out
    out = {}
    for arm in BCA_ARMS + BCA_OPTIONAL_ARMS:
        rec = finished(bca_dir(ctx, arm, seed) / "record.json", spec)
        if rec is not None:
            snap = rec["result"]["snapshots"][-1]
            state = LR.load_tree(bca_dir(ctx, arm, seed) / "snapshot_100.msgpack", LI.bca_template(ctx.cfg, seed),
                                 snap["file_sha256"])
            out[f"bca_{arm}"] = state.native.actor.params
    return out


def links(ctx, paths):
    """{path under the shared root (<quality>/... or start_banks/...): SHA-256} of the records a record was built on
    (checked again by the report)."""
    return {Path(p).relative_to(ctx.root).as_posix(): LI.sha256_file(p) for p in paths}


def stage_evaluate(ctx, seeds, groups=EVAL_DEFAULT):
    """Exact 100-step returns on the three shared start banks of every controller of the groups, with points beside
    them (0 = the random quality, 100 = the stall-aware controller; world.points). Marks read the returns."""
    spec = spec_digest(ctx)
    require(banks_record_path(ctx), spec, "the start banks")
    jobs = [(s, g) for s in seeds for g in groups if finished(evaluate_path(ctx, s, g), spec) is None]
    for s, g in jobs:
        require(data_record_path(ctx, s), spec, f"data for seed {s}")
        controller_records(ctx, g, s, spec)
    if not jobs:
        return
    banks, sb = D.load_start_banks(ctx.banks_root)
    anchors = sb["anchors"]
    for seed in sorted({s for s, _ in jobs}):
        data = load_data(ctx, seed, spec)
        prep = D.prepare(data, "gap", ctx.cfg)   # every controller here uses the gapped normalisation rule
        for group in [g for s, g in jobs if s == seed]:
            started = time.perf_counter()
            inputs = links(ctx, controller_records(ctx, group, seed, spec) + [banks_record_path(ctx)])
            ev = {name: LI.evaluate_actor(p, prep, banks, anchors)
                  for name, p in controllers_of(ctx, group, seed, spec).items()}
            write_record(evaluate_path(ctx, seed, group), stage_record(
                "evaluate", spec, started, quality=ctx.quality, seed=int(seed), group=group, evaluations=ev,
                inputs_sha256=inputs, start_banks_json_sha256=LI.sha256_file(ctx.banks_root / "start_banks.json")))
            ctx.log(f"evaluate {ctx.quality} {group} seed {seed}: finished in {time.perf_counter() - started:.0f}s")


# ------------------------------------------------------------------------------------------------ report

def _get(d, *path):
    for p in path:
        if d is None:
            return None
        d = d.get(p) if isinstance(d, dict) else None
    return d


def _seed_values(recs, *path):
    """Per-seed values along a path, or None when any seed is missing one."""
    if not recs or any(r is None for r in recs):
        return None
    vals = [_get(r, *path) for r in recs]
    return None if any(v is None for v in vals) else [num(v) for v in vals]


def _mark(status, **fields):
    return dict(status=status, **fields)


def _pending(why):
    return _mark("pending", why=why)


def _summary(values):
    """Pooled mean, t-interval and one-sided lower bound of per-seed values (None when a value is not finite: a
    failed seed fails the mark, it is never dropped)."""
    if values is None:
        return None
    v = np.asarray(values, np.float64)
    if not np.all(np.isfinite(v)):
        return dict(per_seed=values, n=len(v), mean=None, sd=None, interval=None, lower_one_sided=None,
                    nonfinite=int((~np.isfinite(v)).sum()))
    return dict(per_seed=values, **CV.across_seeds(values))


def _at_least(x, bound):
    return x is not None and x >= bound


def _above(x, bound):
    return x is not None and x > bound


def _within(x, interval):
    return x is not None and interval[0] <= x <= interval[1]


def _slope(doses, points):
    x, y = np.asarray(doses, np.float64), np.asarray(points, np.float64)
    return float(np.polyfit(x, y, 1)[0]) if np.all(np.isfinite(y)) else math.nan


def p_one_sided(values):
    """One-sided p-value of the t-test that the mean of per-seed values is above 0 (n - 1 degrees of freedom: 4 for
    five seeds). p < 0.05 exactly when the one-sided 95% lower bound is above 0. None when a value is missing or not
    finite."""
    if values is None:
        return None
    v = np.asarray(values, np.float64)
    if len(v) < 2 or not np.all(np.isfinite(v)):
        return None
    sd = float(v.std(ddof=1))
    if sd == 0.0:
        return 0.0 if v.mean() > 0 else 1.0
    return float(stats.t.sf(float(v.mean()) / (sd / math.sqrt(len(v))), len(v) - 1))


def holm(pvalues, alpha):
    """Holm's correction (spec version 5): {name: p-value} -> {name: passes}. The k p-values are sorted; the i-th
    smallest (i = 1...k) passes if it and every smaller one are at most alpha / (k - i + 1). The chance of any false
    pass over the k tests stays at most alpha; with k = 1 it is the uncorrected test."""
    order = sorted(pvalues.items(), key=lambda kv: (kv[1], kv[0]))
    k, out, still = len(order), {}, True
    for i, (name, p) in enumerate(order, 1):
        still = still and p <= alpha / (k - i + 1)
        out[name] = bool(still)
    return out


def d1_judged(record, th):
    """Mark D1 on one seed's quadrature values (the data record's D1 checks) against spec.json's numbers."""
    c = (record or {}).get("D1", {}).get("checks", {})
    v = lambda k: _get(c, k, "value")
    return bool(_at_least(v("holds_removed"), th["holds_removed_min"]) and _within(v("Z"), th["Z"])
                and _within(v("in_ball_mild"), th["in_ball_mild"]) and _within(v("in_ball_strong"), th["in_ball_strong"]))


def check_records(ctx, named):
    """Refuse when a record was made by other code than the code running now; when a record names another quality or
    seed than the folder it was read from, or links (inputs_sha256) a record outside its quality's folder (the shared
    start banks excepted) or of another seed; or when a record it was built on changed since. named: [(name, record or
    None, needs links, seed)]."""
    now = LI.code_hashes()
    other = [name for name, rec, _, _ in named if rec is not None and rec.get("code_sha256") != now]
    if other:
        raise Refused(f"refused: these records were made by other code than the code running now (rerun them, or "
                      f"restore the code): {other}")
    misplaced = []
    for name, rec, _, seed in named:
        if rec is None:
            continue
        if rec.get("quality") != ctx.quality or rec.get("seed") != int(seed):
            misplaced.append(f"{name}: the record names quality {rec.get('quality')!r} and seed {rec.get('seed')!r}")
        inputs = rec.get("inputs_sha256")
        for rel in (inputs if isinstance(inputs, dict) else {}):
            parts = Path(rel).parts
            if parts and parts[0] == "start_banks":
                continue
            if not parts or parts[0] != ctx.quality or \
                    any(re.fullmatch(r"seed\d+", p) and p != f"seed{int(seed)}" for p in parts):
                misplaced.append(f"{name}: it links {rel}, outside {ctx.quality}/.../seed{int(seed)}")
    if misplaced:
        raise Refused(f"refused: these records belong to another quality or seed than the folder they were read from: "
                      f"{misplaced}")
    broken = []
    for name, rec, needs, _ in named:
        if rec is None or not needs:
            continue
        inputs = rec.get("inputs_sha256")
        if not isinstance(inputs, dict) or not inputs:
            broken.append(f"{name}: no record of its inputs")
            continue
        for rel, sha in inputs.items():
            path = ctx.root / rel
            if not path.is_file() or LI.sha256_file(path) != sha:
                broken.append(f"{name}: {rel} changed or is missing")
    if broken:
        raise Refused(f"refused: records no longer match the records they were built on: {broken}")


def build_report(ctx, seeds, spec):
    """Every pass mark of the context's quality from its finished records, judged by the numbers in the frozen
    spec.json (the same numbers for every quality); 'pending' where inputs are missing or fewer than the
    specification's seeds are given; refuses on records made by other code or a broken record chain (check_records).
    Return marks are judged in return units (spec version 5), with points beside them. The improvement mark's verdict
    is the summary's (Holm's correction across the qualities where it ran); the report records its p-value. The lower
    edge's marks (spec version 5, part B): L1 room for the lower edge (oracle, no BCA), L2 whether weighting can move
    return through it (predicted from the band, no BCA training) and L3 its improvement (its own Holm family)."""
    seeds = [int(s) for s in seeds]
    body = frozen_spec(ctx, spec)
    th, want = spec_thresholds(body), spec_seeds(body)
    expected_gain = body["reported_with_expectations_no_mark"]["exact_minus_unweighted_coverage_no_shift_strong"]
    data = [finished(data_record_path(ctx, s), spec) for s in seeds]
    host = {k: [finished(host_dir(ctx, k, s) / "record.json", spec) for s in seeds] for k in HOST_KINDS}
    band = {k: [finished(band_dir(ctx, k, s) / "record.json", spec) for s in seeds] for k in HOST_KINDS}
    cov = {k: [finished(coverage_dir(ctx, k, s) / "record.json", spec) for s in seeds] for k in HOST_KINDS}
    room = [finished(room_dir(ctx, s) / "record.json", spec) for s in seeds]
    ev = {g: [finished(evaluate_path(ctx, s, g), spec) for s in seeds] for g in EVAL_GROUPS}
    bca = {a: [finished(bca_dir(ctx, a, s) / "record.json", spec) for s in seeds] for a in BCA_ARMS + BCA_OPTIONAL_ARMS}
    room_le = [finished(room_le_dir(ctx, s) / "record.json", spec) for s in seeds]
    bca_le = {a: [finished(bca_le_dir(ctx, a, s) / "record.json", spec) for s in seeds]
              for a in LE_ARMS + LE_OPTIONAL_ARMS}
    le_spec = spec_lower_edge(body)
    named = [(f"data {s}", r, False, s) for s, r in zip(seeds, data)]
    named += [(f"bca {a} {s}", r, False, s) for a in bca for s, r in zip(seeds, bca[a])]
    for k in HOST_KINDS:
        named += [(f"host {k} {s}", r, False, s) for s, r in zip(seeds, host[k])]
        named += [(f"band {k} {s}", r, True, s) for s, r in zip(seeds, band[k])]
        named += [(f"coverage {k} {s}", r, True, s) for s, r in zip(seeds, cov[k])]
    named += [(f"room {s}", r, True, s) for s, r in zip(seeds, room)]
    named += [(f"room_lower_edge {s}", r, True, s) for s, r in zip(seeds, room_le)]
    named += [(f"bca_lower_edge {a} {s}", r, True, s) for a in bca_le for s, r in zip(seeds, bca_le[a])]
    named += [(f"evaluate {g} {s}", r, True, s) for g in EVAL_GROUPS for s, r in zip(seeds, ev[g])]
    check_records(ctx, named)
    invalid = {k: [s for s, r in zip(seeds, band[k]) if r is not None and not r.get("band_valid")] for k in HOST_KINDS}
    valid = {k: [r if r is not None and r.get("band_valid") else None for r in band[k]] for k in HOST_KINDS}
    marks, shown = {}, {}

    # D1: the design, on each seed's quadrature of the expert (the ball every quality shares)
    marks["D1"] = (_mark("pass" if all(d1_judged(r, th["D1"]) for r in data) else "fail",
                         per_seed=[r["D1"] for r in data]) if all(data) else _pending("data"))
    design = {str(s): dict(gap=r.get("gap"), D1=r.get("D1")) for s, r in zip(seeds, data) if r is not None}

    # M1: unweighted coverage on held-out training-law steps outside the ball (gapped host)
    v = _seed_values(cov["gap"], "tilts", "none", "arms", "unweighted", "groups", "outside", "coverage")
    if v is None:
        marks["M1"] = _pending("coverage (gap)")
    else:
        s = _summary(v)
        ok = _within(s["mean"], th["M1"]["pooled"]) and min(v) >= th["M1"]["seed_floor"]
        marks["M1"] = _mark("pass" if ok else "fail", coverage=s)

    # M2: width against the exact critic error inside current deciles
    v = _seed_values(cov["gap"], "M2", "error", "within_deciles")
    if v is None:
        marks["M2"] = _pending("coverage (gap)")
    else:
        s = _summary(v)
        marks["M2"] = _mark("pass" if _above(s["lower_one_sided"], th["M2"]["one_sided_lower_bound_above"]) else "fail",
                            within_deciles=s, pooled=_summary(_seed_values(cov["gap"], "M2", "error", "pooled")),
                            noise_map_within_deciles=_summary(_seed_values(cov["gap"], "M2", "noise_map",
                                                                           "within_deciles")),
                            noise_map_pooled=_summary(_seed_values(cov["gap"], "M2", "noise_map", "pooled")),
                            # reported (spec version 5): the width beyond float drift; near 0 means the width
                            # follows float drift rather than the critic's error
                            given_float_radius_within_deciles=_summary(_seed_values(
                                cov["gap"], "M2", "error_given_float_radius", "within_deciles")),
                            mean_float_radius=_seed_values(cov["gap"], "M2", "mean_float_radius"))

    # M3: the dose never falls by more than float32 rounding as the width rises (kept training rows, gapped host)
    if all(valid["gap"]):
        per = [r.get("M3") or {} for r in valid["gap"]]
        ok = all(_get(m, "max_drop") is not None and m["max_drop"] <= th["M3"]["max_dose_drop"]
                 and (m.get("supported_all") or not m.get("threshold_finite")) for m in per)
        marks["M3"] = _mark("pass" if ok else "fail", per_seed=per)
    else:
        marks["M3"] = _pending("band (gap)")

    def cov_values(kind, tilt, arm, group="outside", what="coverage"):
        return _seed_values(cov[kind], "tilts", tilt, "arms", arm, "groups", group, what)

    # W0: weights 1 and test mass 1 reproduce the unweighted threshold; under no tilt the per-step table is that
    # threshold and the per-step arm equals the unweighted arm on every certified step; every seed and host
    recs = valid["gap"] + valid["full"]
    if all(recs) and all(cov["gap"]) and all(cov["full"]):
        identical = {k: [_get(r, "tilts", "none", "arms", "exact", "W0_per_step_none_identical") for r in cov[k]]
                     for k in HOST_KINDS}
        ok = all(r.get("W0_unit_check") is True and r.get("W0_per_step_none") is True for r in recs) and \
            all(v is True for k in HOST_KINDS for v in identical[k])
        marks["W0"] = _mark("pass" if ok else "fail",
                            per_seed={k: dict(unit_check=[r.get("W0_unit_check") for r in valid[k]],
                                              per_step_table_is_unweighted=[r.get("W0_per_step_none") for r in valid[k]],
                                              per_step_coverage_identical=identical[k]) for k in HOST_KINDS})
    else:
        marks["W0"] = _pending("band and coverage")

    # W1: exact-weighted coverage outside the ball under mild and strong, each step at its own threshold (the
    # single-threshold version is reported beside it)
    w1 = {t: cov_values("gap", t, "exact") for t in ("mild", "strong")}
    if any(v is None for v in w1.values()):
        marks["W1"] = _pending("coverage (gap)")
    else:
        s = {t: _summary(v) for t, v in w1.items()}
        ok = all(_within(s[t]["mean"], th["W1"]["pooled"]) for t in s)
        marks["W1"] = _mark("pass" if ok else "fail", coverage=s,
                            single_threshold_coverage={t: _summary(cov_values("gap", t, "exact_single"))
                                                       for t in ("mild", "strong")})

    # W2: unweighted falls short under the strong tilt
    v = cov_values("gap", "strong", "unweighted")
    marks["W2"] = (_pending("coverage (gap)") if v is None else
                   _mark("pass" if float(np.mean(v)) < th["W2"]["unweighted_pooled_below"] else "fail",
                         coverage=_summary(v)))

    # W3: weighted arms refuse every in-ball test step; outside the ball the exact arm refuses none, and the estimated
    # arm refuses exactly the steps whose own threshold, as looked up (rounded up on the grid), is +inf (reported);
    # unweighted is lower inside than outside, judged at a tilt only where every seed's test log has at least
    # min_in_ball_steps steps inside the ball (spec version 5); the refusal clauses are judged everywhere
    if all(cov["gap"]):
        detail, ok = {}, True
        for t in W.TILT_NAMES:
            row = {}
            for arm in PER_STEP_ARMS:
                inside = cov_values("gap", t, arm, "inside", "refused_share")
                outside = cov_values("gap", t, arm, "outside", "refused_share")
                by_threshold = _seed_values(cov["gap"], "tilts", t, "arms", arm,
                                            "refused_by_threshold_share_certifiable")
                allowed = [th["W3"]["refused_outside"]] * len(seeds) if arm == "exact" else by_threshold
                good = None not in (inside, outside, allowed) and \
                    all(x == th["W3"]["refused_inside"] for x in inside) and \
                    all(x == a for x, a in zip(outside, allowed))
                row[arm] = dict(refused_inside=inside, refused_outside=outside,
                                refused_by_own_threshold_outside=by_threshold, ok=good)
                ok &= good
            for arm in ("exact_single", "estimated_single"):
                row[arm] = dict(refused_inside=cov_values("gap", t, arm, "inside", "refused_share"),
                                refused_outside=cov_values("gap", t, arm, "outside", "refused_share"))
            u_in, u_out = cov_values("gap", t, "unweighted", "inside"), cov_values("gap", t, "unweighted", "outside")
            steps_in = _seed_values(cov["gap"], "tilts", t, "in_ball_steps")
            judged = steps_in is not None and min(steps_in) >= th["W3"]["min_in_ball_steps"]
            lower = u_in is not None and u_out is not None and float(np.mean(u_in)) < float(np.mean(u_out))
            row["unweighted"] = dict(inside=_summary(u_in), outside=_summary(u_out), in_ball_steps=steps_in,
                                     judged=judged, lower_inside=lower if judged else None,
                                     note=None if judged else
                                     f"not judged: too few steps inside the ball (fewer than "
                                     f"{th['W3']['min_in_ball_steps']} on some seed)")
            if judged:
                ok &= lower
            detail[t] = row
        marks["W3"] = _mark("pass" if ok else "fail", tilts=detail,
                            inside_versus_outside_judged=[t for t in W.TILT_NAMES if detail[t]["unweighted"]["judged"]])
    else:
        marks["W3"] = _pending("coverage (gap)")

    # W4: estimated against exact (strong) and the empirical no-tilt control, each step at its own threshold (the
    # single-threshold gaps are reported beside them)
    est_s, ex_s = cov_values("gap", "strong", "estimated"), cov_values("gap", "strong", "exact")
    est_n, un_n = cov_values("gap", "none", "estimated"), cov_values("gap", "none", "unweighted")
    if None in (est_s, ex_s, est_n, un_n):
        marks["W4"] = _pending("coverage (gap)")
    else:
        gap_s = abs(float(np.mean(est_s)) - float(np.mean(ex_s)))
        gap_n = abs(float(np.mean(est_n)) - float(np.mean(un_n)))
        slopes = {t: _seed_values(valid["gap"], "classifiers", t, "compare", "slope_on_c") for t in W.TILT_NAMES}
        single = {k: cov_values("gap", t, a) for k, (t, a) in dict(es=("strong", "estimated_single"),
                                                                    xs=("strong", "exact_single"),
                                                                    en=("none", "estimated_single")).items()}
        marks["W4"] = _mark("pass" if gap_s <= th["W4"]["max_abs_gap"] and gap_n <= th["W4"]["max_abs_gap"]
                            else "fail", strong_estimated_minus_exact=gap_s, none_estimated_minus_unweighted=gap_n,
                            single_threshold=None if None in single.values() else dict(
                                strong_estimated_minus_exact=abs(float(np.mean(single["es"]))
                                                                 - float(np.mean(single["xs"]))),
                                none_estimated_minus_unweighted=abs(float(np.mean(single["en"]))
                                                                    - float(np.mean(un_n)))),
                            classifier_slope_on_c={t: dict(fitted=_summary(v), exact=W.ratio_slope(t))
                                                   for t, v in slopes.items()})

    # R1: room for caution on the strong start bank (gates on placed - none; placed - matched even is reported), in
    # return units (spec version 5), with points beside
    def room_points(arm, law="strong"):
        return _seed_values(ev["room"], "evaluations", arm, law, "points")

    def room_return(arm, law="strong"):
        return _seed_values(ev["room"], "evaluations", arm, law, "J")

    def minus(a, b):
        return None if a is None or b is None else [x - y for x, y in zip(a, b)]

    placed, none = room_return("placed"), room_return("none")
    if placed is None or none is None:
        marks["R1"] = _pending("evaluate (room)")
    else:
        s = _summary(minus(placed, none))
        ok = _at_least(s["mean"], th["R1"]["min_return"]) and \
            _above(s["lower_one_sided"], th["R1"]["one_sided_lower_bound_above"])
        found = [_get(r, "evaluations", LI.MATCHED) is not None for r in ev["room"]]
        matched = room_return(LI.MATCHED)
        marks["R1"] = _mark("pass" if ok else "fail", placed_minus_none=s,
                            placed_minus_none_points=_summary(minus(room_points("placed"), room_points("none"))),
                            placed_minus_matched_even=_summary(minus(placed, matched)),
                            matched_even_found_per_seed=found,
                            matched_even_note=None if all(found) else
                            "even caution of the placed arm's strength was not found on every seed (the placed arm "
                            "is stronger than the ladder's top rung, x1.5, or matching failed); placed minus matched "
                            "even is then missing",
                            match=[_get(r, "room", "match") for r in room] if all(room) else None,
                            gate_choice=next((m.get("gate_choice") for m in body["pass_marks"] if m["id"] == "R1"),
                                             None))

    # R2: weighting's reach, predicted from the frozen band and the room check's even arms: the per-step dose (each
    # kept training row at its own threshold, refusal 1.5) against the unweighted dose; the single threshold beside it
    change = _seed_values(valid["gap"], "doses_per_step", "exact_strong", "change_vs_unweighted")
    evens = {k: room_return(k) for k in LI.ROOM_EVEN_DOSES}
    evens_points = {k: room_points(k) for k in LI.ROOM_EVEN_DOSES}
    if change is None or any(v is None for v in evens.values()):
        marks["R2"] = _pending("band (gap) and evaluate (room)")
    else:
        names = list(LI.ROOM_EVEN_DOSES)
        doses = [LI.ROOM_EVEN_DOSES[k] for k in names]
        slopes = [_slope(doses, [evens[k][i] for k in names]) for i in range(len(seeds))]
        slope = float(np.mean(slopes))
        predicted = float(np.mean(change)) * slope
        slopes_points = (None if any(v is None for v in evens_points.values()) else
                         [_slope(doses, [evens_points[k][i] for k in names]) for i in range(len(seeds))])
        single = _seed_values(valid["gap"], "doses", "exact_strong_single", "change_vs_unweighted")
        marks["R2"] = _mark("pass" if math.isfinite(predicted) and
                            abs(predicted) >= th["R2"]["min_abs_predicted_return"] else "fail",
                            predicted_return=predicted, dose_change=_summary(change),
                            return_per_unit_dose=_summary(slopes),
                            predicted_points=None if slopes_points is None else
                            float(np.mean(change)) * float(np.mean(slopes_points)),
                            points_per_unit_dose=_summary(slopes_points),
                            mean_abs_dose_change=_summary(_seed_values(valid["gap"], "doses_per_step", "exact_strong",
                                                                       "mean_abs_change")),
                            high_current_dose_change=_summary(_seed_values(valid["gap"], "doses_per_step",
                                                                           "exact_strong", "high_current_change")),
                            refused_rows=_seed_values(valid["gap"], "doses_per_step", "exact_strong", "refused_rows"),
                            single_threshold=dict(dose_change=_summary(single),
                                                  predicted_return=None if single is None else
                                                  float(np.mean(single)) * slope))

    # L1: room for the lower edge (no BCA): the oracle lower edge at k = 1 minus no penalty, strong start bank, in
    # return units; a seed stopped by check (e) (the oracle penalty does not rise past 80%) fails it
    def le_return(arm, law="strong", what="J"):
        return _seed_values(ev["room_lower_edge"], "evaluations", arm, law, what)

    stopped = [s for s, r in zip(seeds, room_le) if r is not None and _get(r, "room_lower_edge", "stopped")]
    check_e = [None if r is None else {k: v for k, v in (_get(r, "room_lower_edge", "check_e") or {}).items()
                                       if k in ("passed", "failed", "e1_fall_share_ok", "e2_mean_rises",
                                                "e3_slope_past_knee_positive", "e4_passed", "e4_not_applicable",
                                                "e4_fewer_best_past80_than_q1", "fall_share_80_to_81",
                                                "best_past80_share", "q1_alone_best_past80_share", "rows", "why",
                                                "note")} for r in room_le]
    stop_why = ("check (e) on the oracle penalty failed on seeds " + str(stopped) + " (" + "; ".join(
        f"seed {s}: {', '.join(_get(r, 'room_lower_edge', 'check_e', 'failed') or ['its verdict'])}"
        for s, r in zip(seeds, room_le) if s in stopped) + "); no arm was trained there")
    not_applicable = [s for s, c in zip(seeds, check_e) if c and c.get("e4_not_applicable")]
    oracle, base = le_return("le_oracle"), le_return("le_none")
    if stopped:
        marks["L1"] = _mark("fail", why=stop_why, stopped_seeds=stopped, check_e=check_e,
                            e4_not_applicable_seeds=not_applicable)
    elif oracle is None or base is None:
        marks["L1"] = _pending("room-lower-edge and evaluate (room_lower_edge)")
    else:
        s = _summary(minus(oracle, base))
        ok = _at_least(s["mean"], th["L1"]["min_return"]) and \
            _above(s["lower_one_sided"], th["L1"]["one_sided_lower_bound_above"])
        found = [_get(r, "evaluations", LE.MATCHED) is not None for r in ev["room_lower_edge"]]
        marks["L1"] = _mark("pass" if ok else "fail", oracle_minus_none=s,
                            oracle_minus_none_points=_summary(minus(le_return("le_oracle", what="points"),
                                                                    le_return("le_none", what="points"))),
                            oracle_minus_matched_even=_summary(minus(oracle, le_return(LE.MATCHED))),
                            matched_even_found_per_seed=found,
                            matched_even_note=None if all(found) else
                            "even pull of the oracle arm's strength was not found on every seed (stronger than the top "
                            "rung, or matching failed); oracle minus matched even pull is then missing",
                            check_e=check_e, e4_not_applicable_seeds=not_applicable,
                            e4_note=None if not not_applicable else
                            f"on seeds {not_applicable} the critic alone never prefers a throttle past 80% on check "
                            f"(e)'s rows: there was no over-preference past 80% to remove; e4 was not applicable and "
                            f"the arms trained (e1-e3 checked the penalty's shape)",
                            check_b=[_get(r, "room_lower_edge", "check_b", "max_param_diff") for r in room_le],
                            le_none_strength=[_get(r, "room_lower_edge", "strengths", "le_none") for r in room_le],
                            clone_reproduces_room_check=[_get(r, "room_lower_edge", "clone_reproduces_room_check")
                                                         for r in room_le])

    # L2: can weighting move return through the lower edge? (frozen band, no BCA training): the strong-bank return
    # slope per unit of penalty scale above k = 1 (the oracle arms) times the per-step exact strong arm's scale change
    # R_i / R_u - 1 on kept training rows
    change = _seed_values(valid["gap"], "lower_edge_scale", "exact_strong", "scale_change")
    slope_arms = le_spec["slope_arms"]
    ks = [le_spec["rule"].k[a] for a in slope_arms]
    Js = [le_return(a) for a in slope_arms]
    nothing = [s for s, r in zip(seeds, valid["gap"]) if isinstance(_get(r, "lower_edge_scale"), dict)
               and _get(r, "lower_edge_scale", "exact_strong") is None]    # the band certifies nothing there
    # reported beside L2 (spec.json lower_edge.proxy_check), before any lower-edge BCA training: does BCA's unweighted
    # lower edge act like the oracle at k = 1, as the prediction assumes? (frozen gapped band, the room check's
    # no-caution actor's commands, kept training rows)
    proxy = [_get(r, "lower_edge_proxy") for r in valid["gap"]]
    band_e = [_get(p, "band_check_e", "passed") for p in proxy]
    proxy_shown = dict(
        band_over_oracle=_summary(_seed_values(valid["gap"], "lower_edge_proxy", "band_over_oracle")),
        pearson=_summary(_seed_values(valid["gap"], "lower_edge_proxy", "pearson")),
        spearman=_summary(_seed_values(valid["gap"], "lower_edge_proxy", "spearman")),
        band_mean=_seed_values(valid["gap"], "lower_edge_proxy", "band_mean"),
        oracle_mean=_seed_values(valid["gap"], "lower_edge_proxy", "oracle_mean"),
        band_check_e_passed=band_e, band_check_e_failed=[_get(p, "band_check_e", "failed") for p in proxy],
        note=None if not any(v is False for v in band_e) else
        f"check (e) on BCA's own unweighted penalty failed on seeds "
        f"{[s for s, v in zip(seeds, band_e) if v is False]}: BCA's lower edge may not act like the oracle's there",
        about="reported only: band_over_oracle is BCA's effective k against the oracle's 1 (the prediction assumes "
              "about 1), with the correlation of the two penalties row by row")
    if stopped:
        marks["L2"] = _mark("fail", why=stop_why, stopped_seeds=stopped, proxy_check=proxy_shown)
    elif nothing:
        marks["L2"] = _mark("fail", why=f"the gapped band certifies nothing (no finite threshold) on seeds {nothing}",
                            proxy_check=proxy_shown)
    elif change is None or any(v is None for v in Js):
        marks["L2"] = _pending("band (gap) and evaluate (room_lower_edge)")
    else:
        slopes = [_slope(ks, [J[i] for J in Js]) for i in range(len(seeds))]
        Jp = [le_return(a, what="points") for a in slope_arms]
        slopes_points = None if any(v is None for v in Jp) else [_slope(ks, [J[i] for J in Jp])
                                                                  for i in range(len(seeds))]
        predicted = float(np.mean(change)) * float(np.mean(slopes))
        scale = lambda name, what: _seed_values(valid["gap"], "lower_edge_scale", name, what)
        marks["L2"] = _mark("pass" if math.isfinite(predicted) and
                            abs(predicted) >= th["L2"]["min_abs_predicted_return"] else "fail",
                            predicted_return=predicted, scale_change=_summary(change),
                            return_per_unit_k=_summary(slopes), slope_arms=slope_arms, slope_k=ks,
                            predicted_points=None if slopes_points is None else
                            float(np.mean(change)) * float(np.mean(slopes_points)),
                            points_per_unit_k=_summary(slopes_points),
                            high_current_scale_change=_summary(scale("exact_strong", "high_current_scale_change")),
                            mild_scale_change=_summary(scale("exact_mild", "scale_change")),
                            refused_rows=scale("exact_strong", "refused_rows"),
                            largest_finite_threshold=scale("exact_strong", "largest_finite_threshold"),
                            proxy_check=proxy_shown)

    # a failed band refresh fails every mark that needs the band, it never leaves them pending
    for k in BAND_DEPENDENT:
        bad = invalid["gap"] + (invalid["full"] if k == "W0" else [])
        if bad:
            marks[k] = _mark("fail", why=f"BCA's band has no valid refresh on seeds {sorted(set(bad))}",
                             invalid_band_seeds={h: invalid[h] for h in HOST_KINDS})

    # cross-seed marks need every seed of the specification
    complete = sorted(seeds) == want
    if not complete:
        for k, m in marks.items():
            if k != "D1" and m["status"] in ("pass", "fail"):
                marks[k] = dict(m, status="pending", preview_status=m["status"], preview_why=m.get("why"),
                                why=f"judged only on all of seeds {want}; this report covers {seeds}")

    def family_p(d):
        """The p-value an improvement mark gives Holm's family: a seed whose difference is not finite (a diverged arm)
        fails the mark, it is never dropped, so the quality stays in the family with p = 1."""
        p = p_one_sided(d)
        return 1.0 if p is None else p

    # I1: only after M1, R1 and R2 pass. The paired difference in return on the strong start bank and its one-sided
    # p-value are recorded here; the verdict is the summary's, with Holm's correction across its family, the
    # qualities whose report allows the capped pull's arms (spec version 5). uncorrected_status is this quality's test
    # alone (one-sided 95% bound above 0)
    if any(marks[k]["status"] != "pass" for k in ("M1", "R1", "R2")):
        marks["I1"] = _mark("not run", why="M1, R1 and R2 must all pass first")
    else:
        bca_ev = lambda arm, law="strong", what="J": _seed_values(ev["bca"], "evaluations", arm, law, what)
        a, b = bca_ev("bca_exact_strong"), bca_ev("bca_unweighted")
        if a is None or b is None:
            marks["I1"] = _pending("evaluate (bca)")
        else:
            d = minus(a, b)
            s = _summary(d)
            uncorrected = _above(s["lower_one_sided"], th["I1"]["one_sided_lower_bound_above"])
            marks["I1"] = _mark("pending", why="judged in the summary, with Holm's correction across its family (the "
                                               "qualities whose report allows these arms)", ran=True,
                                p_value=family_p(d), nonfinite_seeds=s.get("nonfinite", 0),
                                uncorrected_status="pass" if uncorrected else "fail",
                                exact_strong_minus_unweighted=s,
                                exact_strong_minus_unweighted_points=_summary(
                                    minus(bca_ev("bca_exact_strong", what="points"),
                                          bca_ev("bca_unweighted", what="points"))),
                                single_threshold_minus_unweighted=_summary(minus(bca_ev("bca_exact_strong_single"),
                                                                                 b)))

    # L3: improvement through the lower edge, only after M1, L1 and L2 pass; like I1, the report records the paired
    # difference in return on the strong start bank and its one-sided p-value, and the summary judges it with Holm's
    # correction across its own family (the qualities whose report allows the lower edge's arms)
    if any(marks[k]["status"] != "pass" for k in ("M1", "L1", "L2")):
        marks["L3"] = _mark("not run", why="M1, L1 and L2 must all pass first")
    else:
        le_ev = lambda arm, law="strong", what="J": _seed_values(ev["bca_lower_edge"], "evaluations", arm, law, what)
        a, b = le_ev("le_bca_exact_strong"), le_ev("le_bca_unweighted")
        if a is None or b is None:
            marks["L3"] = _pending("bca-lower-edge and evaluate (bca_lower_edge)")
        else:
            d = minus(a, b)
            s = _summary(d)
            uncorrected = _above(s["lower_one_sided"], th["L3"]["one_sided_lower_bound_above"])
            marks["L3"] = _mark("pending", why="judged in the summary, with Holm's correction across its family (the "
                                               "qualities whose report allows these arms)", ran=True,
                                p_value=family_p(d), nonfinite_seeds=s.get("nonfinite", 0),
                                uncorrected_status="pass" if uncorrected else "fail",
                                exact_strong_minus_unweighted=s,
                                exact_strong_minus_unweighted_points=_summary(
                                    minus(le_ev("le_bca_exact_strong", what="points"),
                                          le_ev("le_bca_unweighted", what="points"))),
                                exact_mild_minus_unweighted_mild_bank=_summary(
                                    minus(le_ev("le_bca_exact_mild", "mild"), le_ev("le_bca_unweighted", "mild"))),
                                estimated_strong_minus_unweighted=_summary(minus(le_ev("le_bca_estimated_strong"), b)))

    # reported with expectations, no marks
    shown["full_data_coverage"] = {t: {arm: _summary(cov_values("full", t, arm, "all"))
                                       for arm in TEST_ARMS["full"]} for t in W.TILT_NAMES}
    shown["forecast_vs_realised_unweighted"] = {
        t: dict(forecast=_summary(_seed_values(cov["gap"], "forecast", "values", t)),
                realised=_summary(cov_values("gap", t, "unweighted"))) for t in ("mild", "strong")}
    per_step_arms = {k: PER_STEP_ARMS if k == "gap" else ("exact",) for k in HOST_KINDS}
    shown["R_weighted_over_R_unweighted"] = {
        k: dict(single={name: _summary(_seed_values(valid[k], "thresholds", name, "ratio_to_unweighted"))
                        for name in [f"{a}_{t}_single" for a in per_step_arms[k] for t in W.TILT_NAMES]},
                per_step_test_steps={f"{a}_{t}": {s: _summary(_seed_values(valid[k], "tables", f"{a}_{t}",
                                                                            "test_steps", s))
                                                  for s in ("ratio_min", "ratio_mean", "ratio_max")}
                                     for a in per_step_arms[k] for t in W.TILT_NAMES})
        for k in HOST_KINDS}
    shown["refused_by_own_threshold_outside_ball"] = {
        f"{a}_{t}": _summary(_seed_values(cov["gap"], "tilts", t, "arms", a, "refused_by_threshold_share_certifiable"))
        for a in PER_STEP_ARMS for t in W.TILT_NAMES}
    shown["possibly_rounded_into_refusal_outside_ball"] = {     # steps per seed; an upper bound (coverage.lookup_gaps)
        f"{a}_{t}": _seed_values(cov["gap"], "tilts", t, "arms", a, "per_step", "lookup",
                                 "possibly_rounded_into_refusal")
        for a in PER_STEP_ARMS for t in W.TILT_NAMES}

    def gain(arm, t):
        a, b = cov_values("gap", t, arm), cov_values("gap", t, "unweighted")
        return _summary(None if a is None or b is None else [x - y for x, y in zip(a, b)])

    shown["exact_minus_unweighted_coverage"] = dict(
        per_tilt={t: gain("exact", t) for t in W.TILT_NAMES},
        single_threshold_per_tilt={t: gain("exact_single", t) for t in W.TILT_NAMES},
        no_shift_expectation_strong=expected_gain)
    shown["single_threshold_coverage"] = {t: {a: _summary(cov_values("gap", t, a))
                                              for a in ("exact_single", "estimated_single")} for t in W.TILT_NAMES}
    shown["in_ball_share_of_each_test_log"] = {t: _seed_values(cov["gap"], "tilts", t, "in_ball_share")
                                               for t in W.TILT_NAMES}
    fg, gg = (_seed_values(ev["hosts"], "evaluations", f"plain_{k}", "strong", "points") for k in ("full", "gap"))
    shown["full_minus_gapped_plain_td3bc_strong"] = _summary(None if fg is None or gg is None else
                                                              [a - b for a, b in zip(fg, gg)])
    shown["full_minus_gapped_plain_td3bc_strong_return"] = _summary(minus(
        *(_seed_values(ev["hosts"], "evaluations", f"plain_{k}", "strong", "J") for k in ("full", "gap"))))
    shown["plain_td3bc_points"] = {k: {law: _summary(_seed_values(ev["hosts"], "evaluations", f"plain_{k}", law,
                                                                  "points")) for law in W.TILT_NAMES}
                                   for k in HOST_KINDS}
    shown["plain_td3bc_return"] = {k: {law: _summary(_seed_values(ev["hosts"], "evaluations", f"plain_{k}", law,
                                                                  "J")) for law in W.TILT_NAMES}
                                   for k in HOST_KINDS}
    shown["room_points"] = {arm: {law: _summary(room_points(arm, law)) for law in W.TILT_NAMES}
                            for arm in ("placed", *LI.ROOM_EVEN_DOSES, LI.MATCHED)}
    shown["room_return"] = {arm: {law: _summary(room_return(arm, law)) for law in W.TILT_NAMES}
                            for arm in ("placed", *LI.ROOM_EVEN_DOSES, LI.MATCHED)}
    shown["in_ball_steps_of_each_test_log"] = {t: _seed_values(cov["gap"], "tilts", t, "in_ball_steps")
                                               for t in W.TILT_NAMES}
    bu = _seed_values(ev["bca"], "evaluations", "bca_unweighted", "strong", "points")
    if bu is not None and gg is not None:
        bn_e = _seed_values(ev["bca"], "evaluations", "bca_exact_strong", "none", "points")
        bn_u = _seed_values(ev["bca"], "evaluations", "bca_unweighted", "none", "points")
        shown["bca"] = dict(unweighted_minus_plain_strong=_summary([a - b for a, b in zip(bu, gg)]),
                            exact_minus_unweighted_none_bank=_summary(None if bn_e is None or bn_u is None else
                                                                      [a - b for a, b in zip(bn_e, bn_u)]))
    shown["bca_records"] = {a: dict(uncertified_refreshes=[_get(r, "result", "uncertified_refreshes") for r in recs],
                                    refused_rows_max=[_get(r, "result", "refused_rows_max") for r in recs])
                            for a, recs in bca.items() if any(recs)}
    room_le_arms = (*LE.ROOM_ARMS, *[f"le_even_x{m:g}" for m in le_spec["rule"].even_extra], LE.MATCHED)
    shown["lower_edge"] = dict(
        room_return={arm: {law: _summary(le_return(arm, law)) for law in W.TILT_NAMES} for arm in room_le_arms},
        room_points={arm: {law: _summary(le_return(arm, law, "points")) for law in W.TILT_NAMES}
                     for arm in room_le_arms},
        scale_change={name: {what: _summary(_seed_values(valid["gap"], "lower_edge_scale", name, what))
                             for what in ("scale_change", "high_current_scale_change", "ratio_max")}
                      for name in ("exact_strong", "exact_mild", "estimated_strong")},
        bca_records={a: dict(refused_rows=[_get(r, "result", "arm", "refused_rows") for r in recs],
                             check_e_passed=[_get(r, "result", "arm", "check_e", "passed") for r in recs])
                     for a, recs in bca_le.items() if any(recs)},
        bca_return={a: {law: _summary(_seed_values(ev["bca_lower_edge"], "evaluations", f"le_bca_{a}", law, "J"))
                        for law in W.TILT_NAMES} for a, recs in bca_le.items() if any(recs)})
    return dict(quality=ctx.quality, seeds=seeds, complete=complete, marks=marks, shown=shown, thresholds=th,
                design=design, code_is_approved=code_is_approved(ctx),
                bca_allowed=complete and all(marks[k]["status"] == "pass" for k in ("M1", "R1", "R2")),
                bca_lower_edge_allowed=complete and all(marks[k]["status"] == "pass" for k in ("M1", "L1", "L2")))


def code_is_approved(ctx):
    """Reported in the report and the summary: whether approval.json verifies now, so the code running now (which
    made every record judged, check_records) is the code the owner approved. The report itself judges records against
    the code running now; the band, coverage and BCA stages refuse code other than the approved code."""
    try:
        LI.verify_approval(ctx.approval_path, ctx.spec_path)
        return True
    except LI.ApprovalError:
        return False


def stage_report(ctx, seeds=SEEDS):
    """The report of the context's quality (<quality>/report/report.json): every pass mark from its finished
    stages."""
    spec = spec_digest(ctx)
    for s in seeds:
        require(data_record_path(ctx, s), spec, f"data for seed {s}")
    started = time.perf_counter()
    body = build_report(ctx, seeds, spec)
    write_record(report_path(ctx), stage_record("report", spec, started, **body))
    for k, m in body["marks"].items():
        ctx.log(f"{ctx.quality} {k}: {m['status']}")
    return body


# ------------------------------------------------------------------------------------------------ summary

def spec_claims(body):
    """The claim rules (spec.json pass_marks_rules.claims): {claim: the marks that must all pass on a quality}."""
    claims = body["pass_marks_rules"]["claims"]
    return {name: list(c["marks"]) for name, c in claims.items()}


def holm_marks(body):
    """{mark id: family alpha} of the marks judged with Holm's correction across qualities (thresholds
    holm_across_qualities true)."""
    return {m["id"]: float(m["thresholds"]["family_alpha"]) for m in body["pass_marks"]
            if m["thresholds"].get("holm_across_qualities")}


def read_declined(ctx, spec):
    """The owner's decisions not to run a quality's arms ({mark: [quality, ...]}, I1 for the capped pull, L3 for the
    lower edge), from declined.json: written only by the orchestrator from the owner's words, with a sha256sum sidecar,
    kind 'shift-iw-declined', this specification's SHA-256 and the owner's words verbatim. {} when there is no such
    file; refuses one that does not verify."""
    path = ctx.declined_path
    if not path.exists():
        return {}
    rec = read_record(path)
    if rec is None:
        raise Refused(f"refused: {path} does not match its sidecar")
    words, declined = rec.get("user_words_verbatim"), rec.get("declined")
    if rec.get("kind") != DECLINED_KIND or rec.get("spec_sha256") != spec or not isinstance(words, str) \
            or not words.strip() or not isinstance(declined, dict) or any(m not in HOLM_ELIGIBLE for m in declined) \
            or any(not isinstance(v, list) or any(q not in QUALITIES for q in v) for v in declined.values()):
        raise Refused(f"refused: {path} must be of kind {DECLINED_KIND!r}, name this specification, hold the owner's "
                      f"words verbatim and map {sorted(HOLM_ELIGIBLE)} to lists of qualities")
    return {m: sorted(set(v)) for m, v in declined.items()}


def _family_p(value):
    """A family member's p-value: a missing or non-finite one counts as 1 (a failed seed fails the mark)."""
    try:
        p = float(value)
    except (TypeError, ValueError):
        return 1.0
    return p if math.isfinite(p) and 0.0 <= p <= 1.0 else 1.0


def build_summary(ctx, spec, qualities=QUALITIES):
    """One table across the qualities from their reports (spec version 5). Refuses when a report fails its sidecar,
    names another quality than its folder, was made under another specification or by other code than the code
    running now, or when two reports give a seed different balls or different design-check records. Judges only the
    marks with Holm's correction, each over its own family (spec.json pass_marks_rules.holm_families): the qualities
    whose report allows the mark's arms (HOLM_ELIGIBLE: bca_allowed for I1, bca_lower_edge_allowed for L3), and any
    quality where it ran. A member that ran gives its p-value (1 if not finite); a member the owner declined
    (declined.json) counts with p = 1, so declining never loosens the bar for the others. The family is judged only
    when every quality's report is complete and every member has run or was declined; until then its marks and claims
    are 'pending'. Every other mark's status is the report's. The claim rules then say, per quality, which claims its
    marks support; the table across qualities describes the results and judges nothing."""
    body = frozen_spec(ctx, spec)
    now = LI.code_hashes()
    reports, problems = {}, []
    for q in qualities:
        path = report_path(ctx.at(q))
        if not path.exists():
            continue
        rec = read_record(path)
        if rec is None:
            problems.append(f"{q}: the report does not match its sidecar")
            continue
        if rec.get("quality") != q:
            problems.append(f"{q}: the report in this quality's folder is of quality {rec.get('quality')!r}")
        if rec.get("status") != "finished" or rec.get("spec_sha256") != spec:
            problems.append(f"{q}: the report was made under another specification")
        if rec.get("code_sha256") != now:
            problems.append(f"{q}: the report was made by other code than the code running now")
        reports[q] = rec
    by_seed = {}
    for q, rec in reports.items():
        for s, d in (rec.get("design") or {}).items():
            by_seed.setdefault(s, []).append((q, json.dumps(d.get("gap"), sort_keys=True),
                                              json.dumps(d.get("D1"), sort_keys=True)))
    for s, rows in sorted(by_seed.items()):
        if len({g for _, g, _ in rows}) > 1:
            problems.append(f"seed {s}: the qualities' reports give different balls")
        if len({d for _, _, d in rows}) > 1:
            problems.append(f"seed {s}: the qualities' reports give different design-check records")
    if problems:
        raise Refused(f"refused: the reports cannot be summarised together: {problems}")
    complete = len(reports) == len(qualities) and all(r.get("complete") for r in reports.values())
    declined = read_declined(ctx, spec)
    holm_rules = holm_marks(body)
    holm_out = {}
    for mark, alpha in holm_rules.items():
        flag = HOLM_ELIGIBLE[mark]
        eligible = sorted(q for q, r in reports.items() if r.get(flag) is True)
        ran = {q: _family_p(_get(r, "marks", mark, "p_value")) for q, r in reports.items()
               if _get(r, "marks", mark, "ran")}
        dropped = sorted(q for q in declined.get(mark, []) if q in eligible and q not in ran)
        family = sorted(set(eligible) | set(ran))
        waiting = [q for q in family if q not in ran and q not in dropped]
        pvalues = dict(ran, **{q: 1.0 for q in dropped})
        judged = complete and not waiting
        verdict = holm(pvalues, alpha) if judged else {}
        holm_out[mark] = dict(
            family_alpha=alpha, family_rule=f"the qualities whose report says {flag}, and any where it ran",
            family=family, ran_in=sorted(ran), declined=dropped, waiting_for=waiting, k=len(family),
            p_values=pvalues, passes=verdict, judged=judged,
            note=None if judged else "pending: judged only when every quality's report is complete" if not complete
            else f"pending: waiting for {waiting} (their reports allow these arms; the arms have not run there, and "
                 f"the owner has not declined them in declined.json)")
    claims = spec_claims(body)
    banks = finished(banks_record_path(ctx), spec)
    rows = {}
    for q in qualities:
        rec = reports.get(q)
        status = {}
        for m in (rec or {}).get("marks", {}):
            st = rec["marks"][m]["status"]
            h = holm_out.get(m)
            if h is not None and q in h["family"]:
                st = ("pending" if not h["judged"] else "declined" if q in h["declined"] else
                      "pass" if h["passes"].get(q) else "fail")
            status[m] = st
        data_points = None
        if banks is not None:
            a = banks["start_banks"]["anchors"]
            data_points = {law: dict(with_hold=_get(a, law, "controllers", q, "points"),
                                     without_hold=_get(a, law, "controllers", f"{q} without the hold", "points"))
                           for law in W.TILT_NAMES}
        rows[q] = dict(report=rec is not None, complete=bool(rec and rec.get("complete")), marks=status,
                       claims={c: bool(rec) and all(status.get(m) == "pass" for m in ms) for c, ms in claims.items()},
                       data_points=data_points,
                       plain_td3bc_points=_get(rec, "shown", "plain_td3bc_points"),
                       plain_td3bc_return=_get(rec, "shown", "plain_td3bc_return"),
                       code_is_approved=_get(rec, "code_is_approved"),
                       label=_get(body, "qualities", "labels", q))
    return dict(qualities=list(qualities), complete=complete, rows=rows, holm=holm_out, declined=declined,
                code_is_approved=code_is_approved(ctx), claim_rules=body["pass_marks_rules"]["claims"],
                across_qualities="describes the results; judges nothing (spec.json pass_marks_rules.claims)")


def stage_summary(ctx, qualities=QUALITIES):
    """The summary across qualities (summary/summary.json)."""
    spec = spec_digest(ctx)
    started = time.perf_counter()
    body = build_summary(ctx, spec, qualities)
    write_record(summary_path(ctx), stage_record("summary", spec, started, **body))
    for q, row in body["rows"].items():
        ctx.log(f"{q}: " + (", ".join(f"{c} {'yes' if v else 'no'}" for c, v in row["claims"].items())
                            if row["report"] else "no report"))
    return body


def approval_template(ctx):
    """Print (never write) the approval body for the orchestrator to fill in with the user's words: the frozen
    specification's SHA-256, the code's SHA-256 and the first scope. A BCA-arm scope is added only after the user's
    own go for that kind of arm, with the qualities the go covers in 'qualities' ({scope: [quality, ...]}): 'bca' for
    the capped pull, 'bca_lower_edge' for the lower edge, and 'bca_estimated' / 'bca_lower_edge_estimated' only if the
    user asks for the estimated-weighted arm."""
    body = dict(kind=LI.APPROVAL_KIND, spec_sha256=spec_digest(ctx), code_sha256=LI.code_hashes(),
                scope=["band", "coverage"], qualities={}, user_approval_verbatim="<the user's words, verbatim>")
    print(json.dumps(body, indent=1, sort_keys=True), flush=True)
    return body


def status(ctx, seeds=SEEDS):
    """What has finished under the current specification for the context's quality, and the shared start banks and
    approval (no refusal: it reads records only)."""
    try:
        spec = LI.spec_sha256(ctx.spec_path)
    except LI.SpecError as error:
        ctx.log(str(error))
        return None
    rows = {"start banks": finished(banks_record_path(ctx), spec) is not None}
    for s in seeds:
        rows[f"data {s}"] = finished(data_record_path(ctx, s), spec) is not None
        for k in HOST_KINDS:
            rows[f"host {k} {s}"] = finished(host_dir(ctx, k, s) / "record.json", spec) is not None
            rows[f"band {k} {s}"] = finished(band_dir(ctx, k, s) / "record.json", spec) is not None
            rows[f"coverage {k} {s}"] = finished(coverage_dir(ctx, k, s) / "record.json", spec) is not None
        rows[f"room {s}"] = finished(room_dir(ctx, s) / "record.json", spec) is not None
        rows[f"room_lower_edge {s}"] = finished(room_le_dir(ctx, s) / "record.json", spec) is not None
        for a in BCA_ARMS + BCA_OPTIONAL_ARMS:
            rows[f"bca {a} {s}"] = finished(bca_dir(ctx, a, s) / "record.json", spec) is not None
        for a in LE_ARMS + LE_OPTIONAL_ARMS:
            rows[f"bca_lower_edge {a} {s}"] = finished(bca_le_dir(ctx, a, s) / "record.json", spec) is not None
        for g in EVAL_GROUPS:
            rows[f"evaluate {g} {s}"] = finished(evaluate_path(ctx, s, g), spec) is not None
    rows["report"] = finished(report_path(ctx), spec) is not None
    try:
        LI.verify_approval(ctx.approval_path, ctx.spec_path)
        rows["approval"] = True
    except LI.ApprovalError:
        rows["approval"] = False
    for k, v in rows.items():
        ctx.log(f"{k}: {'finished' if v else '-'}")
    return rows


# ------------------------------------------------------------------------------------------------ CLI

STAGES = ("freeze-spec", "data", "host", "room", "room-lower-edge", "band", "coverage", "bca", "bca-lower-edge",
          "evaluate", "report", "status", "approval-template", "summary")
SHARED_STAGES = ("freeze-spec", "approval-template", "summary")   # every other stage runs for one --quality


def main(argv=None):
    p = argparse.ArgumentParser(description="Importance weighting for BCA on ThrusterChain: the stage runner.")
    p.add_argument("stage", choices=STAGES)
    p.add_argument("--quality", choices=QUALITIES, default=None,
                   help="the dataset quality (every stage but freeze-spec, approval-template, summary and "
                        "'data --start-banks' alone)")
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--kinds", nargs="*", choices=tuple(HOST_KINDS), default=None, help="host, band, coverage")
    p.add_argument("--arms", nargs="*", choices=tuple(dict.fromkeys(BCA_ARMS + BCA_OPTIONAL_ARMS + LE_ARMS
                                                                    + LE_OPTIONAL_ARMS)),
                   default=None, help="bca, bca-lower-edge")
    p.add_argument("--groups", nargs="*", choices=EVAL_GROUPS, default=None, help="evaluate")
    p.add_argument("--start-banks", action="store_true", help="data: build the shared start banks")
    p.add_argument("--size", choices=sorted(D.SIZES), default="full", help="data only")
    p.add_argument("--updates", type=int, default=None, help="host only (a smoke test; the spec's is 100,000)")
    p.add_argument("--root", default=str(ROOT_DEFAULT))
    p.add_argument("--spec", default=str(LI.SPEC_JSON))
    a = p.parse_args(argv)
    seeds = list(SEEDS) if a.seeds is None else a.seeds
    kinds = tuple(HOST_KINDS) if not a.kinds else tuple(a.kinds)
    banks_only = a.stage == "data" and a.start_banks and not a.seeds
    if a.quality is None and a.stage not in SHARED_STAGES and not banks_only:
        print(f"refused: the {a.stage} stage runs for one dataset quality; give --quality "
              f"({', '.join(QUALITIES)})", file=sys.stderr, flush=True)
        return 2
    ctx = Context(root=Path(a.root), spec_path=Path(a.spec), quality=a.quality or "expert")
    try:
        if a.stage == "summary":
            stage_summary(ctx)
        elif a.stage == "freeze-spec":
            freeze_spec(ctx)
        elif a.stage == "data":
            stage_data(ctx, [] if a.seeds is None else a.seeds, start_banks=a.start_banks, sizes=D.SIZES[a.size])
        elif a.stage == "host":
            stage_host(ctx, seeds, kinds, updates=a.updates)
        elif a.stage == "room":
            stage_room(ctx, seeds)
        elif a.stage == "room-lower-edge":
            stage_room_lower_edge(ctx, seeds)
        elif a.stage == "band":
            stage_band(ctx, seeds, kinds)
        elif a.stage == "coverage":
            stage_coverage(ctx, seeds, kinds)
        elif a.stage == "bca":
            stage_bca(ctx, seeds, tuple(a.arms) if a.arms else BCA_ARMS)
        elif a.stage == "bca-lower-edge":
            stage_bca_lower_edge(ctx, seeds, tuple(a.arms) if a.arms else LE_ARMS)
        elif a.stage == "evaluate":
            stage_evaluate(ctx, seeds, tuple(a.groups) if a.groups else EVAL_DEFAULT)
        elif a.stage == "report":
            stage_report(ctx, seeds)
        elif a.stage == "status":
            status(ctx, seeds)
        elif a.stage == "approval-template":
            approval_template(ctx)
    except Refused as error:
        print(str(error), file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
