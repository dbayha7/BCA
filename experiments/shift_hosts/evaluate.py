"""Exact scores of a policy on the 7 banks, the metrics of plan section 5.4, and both halves of stop rule S7 (plan
sections 2.3, 3.9, 5.4, 6.1 and 9.2). Scoring environment (/opt/bca-shift-venv, numpy 2.2.6).

An experiment-local attachment (plan section 3.1): it changes none of the repository's declared methods. It imports
experiments/shift and experiments/shift_iw unchanged (truth.rollout, recorder.CappedController, thruster_chain,
data.load_start_banks, world.points) and never edits them.

Banks (INTERFACES.md section 3; all in the stalling world, experiments/shift_iw/world.py:69):
  0-2  none, mild, strong     the weighting test's start banks (default_rng([20261016, 1000]), data.py:603-606), read
                              from the data agent's copy runs/shift_hosts/source/start_banks/ once it exists
                              (data_copy.start_banks_dir()), else read-only from
                              /root/bca-shift/runs/shift_iw_v5/start_banks/. Either must
                              have the pinned SHA-256 below; data.load_start_banks also checks every array against its
                              manifest.
  3-6  recorded, strong_current, big_waves, big_kicks
                              the main study's banks (thruster_chain.bank_starts, default_rng([20261012, 2, 9]))
Points, two scales (plan section 2.3; risk 16), each with its own function:
  weighting banks   0 = the random quality on its fixed streams, 100 = the stall-aware controller, per bank
                    (start_banks.json "anchors"; world.points, which takes the ratio first)
  four situations   0 = the main study's recording controller, 100 = the stall-aware controller
                    (preregistration.json "anchors" "base/stall"; truth.points)

Per start (score): the 100-step return J and the discounted return Jd (truth.rollout, the float64 simulator,
truth.py:77-95; Jd is the 100-step sum with discount 0.98, as learners.evaluate_policy reports it); points; the same
policy capped at 80% (recorder.CappedController, learners.evaluate_policy's rule, learners.py:925-942); regret = the
exact best plan's return minus J (the scoring cache); F1 = J below the zero-command controller's return from the same
start; F2 = the late peak above the stall-aware controller's 99th percentile of the late peak on the same bank; the
late peak = the largest distance of the five floats from rest (the norm of the 10 position and speed numbers) over
steps 10-99 (oracle.py:140-141, 882-884); the share of steps with any thruster past 80% (thruster_chain.past_knee,
learners.evaluate_policy's unrecorded-throttle share); the tracking cost sum_t |s_t|^2 and the energy cost
0.3 sum_t |a_t|^2 over t = 0-99 (oracle.py:878-881; tracking + energy = -J).
Per bank (summary): means with standard errors std / sqrt(n) (as learners.evaluate_policy computes J_se), the loss from
unrecorded throttle (capped minus uncapped, in points and in return units), the F1 and F2 rates, the worst-10% mean (the
mean of the lowest 100 of the 1,000 returns), the 10th percentile (numpy's default linear interpolation) and the mean
regret. paired() gives two setups' paired difference over the same starts with its standard error.

Coverage hand-offs (plan sections 3.1 and 5.4; the scoring environment's half only): visited_transitions writes the
actor's own visited states on the first 200 strong-mix starts with its commands, rewards and the simulator's exact next
states; actor_transitions does the same at given states (for example recorded test states) with the actor's command.
The training environment computes the learner's y, q and U at those rows (band_readout.py) and hands them back.

S7 (plan sections 3.9 and 6.1):
  open loop    open_loop_states writes the states the float64 adapter visits from given starts (default: the viewer's
               10 fixed starts of every bank) as float32 observations, with CQL's noise there; the training environment
               answers with actors_f32.open_loop_commands; open_loop_compare checks the float32 restatement against
               those commands within 1e-7
  closed loop  closed_loop_check rolls the float32 restatement and the float64 adapter from the same starts (default:
               every start) and checks |points difference| <= 0.05 on each (CQL with the declared noise)

Permission. Scoring an actor file needs guard.MADE_UP with an actor file whose meta says made_up true and that no run
trained (stage-0 tests' made-up parameters; the runner marks every actor file it writes trained true), or a
gate.Approval that still verifies (gate.verify). So no learned policy is scored, and no learned policy's return exists,
before the user's terminal approval, not even one learned on made-up arrays (the stage driver's dry run scores
nothing). The one exception is open_loop_states with no bank (made-up states only, no simulator step), which may read a
made-up trained actor's commands under guard.MADE_UP. Fixed reference controllers (score() on a controller, the scoring
cache) need no permission: score() takes a policy object, and every path that builds a learned policy from an actor
file checks the permission first.

The scoring cache is pinned: each bank's cache file (scoring_cache.py wrote it; the zero command's and the stall-aware
controller's returns, the F2 threshold, the regret optima, the viewer's starts) must have the SHA-256 in
CACHE_NPZ_SHA256, and CQL's noise files the SHA-256 in CQL_NOISE_NPZ_SHA256 (both taken from the cache's own
cache.json), so a cache rebuilt or edited after the freeze changes no score without changing this file (stop rule S1).
"""

import hashlib
import json
import math
import sys
from collections import namedtuple
from pathlib import Path

import numpy as np

import experiments.shift  # noqa: F401  (dependency path and version checks; numpy 2.2.6 only)
from experiments.shift import recorder as RC
from experiments.shift import thruster_chain as TC
from experiments.shift import truth as TR
from experiments.shift_iw import data as D
from experiments.shift_iw import world as W
from experiments.shift_hosts import actors_f32 as A32
from experiments.shift_hosts import handoff as H
from experiments.shift_hosts import policies as P

REPO = Path(__file__).resolve().parents[2]
BANKS = P.BANKS
WEIGHTING = W.TILT_NAMES                       # ("none", "mild", "strong")
SITUATIONS = TC.SITUATIONS                     # ("recorded", "strong_current", "big_waves", "big_kicks")
PHYSICS = TC.STALL                             # the stalling world (world.PHYSICS is TC.STALL)
LENGTH = PHYSICS.length                        # 100 steps
LATE_PEAK_FROM = 10                            # oracle.py:140 (LATE_PEAK_FROM)
F2_QUANTILE = 0.99                             # plan section 5.4
WORST_SHARE = 0.10                             # the worst-10% mean
LOW_PERCENTILE = 0.10                          # the 10th percentile
S7_OPEN_TOL = 1e-7                             # plan section 3.9 / stop rule S7
S7_CLOSED_TOL_POINTS = 0.05                    # stop rule S7, pre-specification version 2 (PRESPEC.md section 19)
COPIED_BANKS = REPO / "runs" / "shift_hosts" / "source" / "start_banks"            # data_copy.start_banks_dir()
ORIGINAL_BANKS = Path("/root/bca-shift/runs/shift_iw_v5/start_banks")
START_BANKS_HDF5_SHA256 = "9ad49e478dffdbc16160c3f69637dfb15b110035d64cf5951608d2b73904d50b"   # start_banks.json
START_BANKS_JSON_SHA256 = "cceb2f2d54c179cbfb00185c4d5dedf62c6ff7838f8357d6fe47596eeb3cb740"   # its sidecar
PREREG = REPO / "outputs" / "shift" / "preregistration.json"
PREREG_SHA256 = "68d9e8779e148b7bfd2fcc6dc32e0a1091e5bfbbcb83cef2e4b58e4f6ec60fab"            # its sidecar
PREREG_ANCHORS = "base/stall"
CACHE = REPO / "runs" / "shift_hosts" / "scoring_cache"
# The scoring cache's files, pinned (module docstring; values from runs/shift_hosts/scoring_cache/cache.json "files",
# cache.json SHA-256 8519d551486e33644400ac097507780ef7c4961a04e1211c9bdb90c4727825de; each equals its sidecar)
CACHE_NPZ_SHA256 = dict(
    none="46dacea8a6c3dd7a16ff84332f6b59a0ceef01187da1ef7cb3d5ad1cc1fed69b",
    mild="95cb7ad0b45f33fcd83d95e610b6595927f2b980a620a985b6acd052819919c9",
    strong="74d7e7fe69df7dd40b5a9e50551ba757b3c68e34af2dd6a3279f9dc558293d35",
    recorded="7c55b6384a8b06784265c9b441c94399630a70f16f17df316e7485f815b9ed82",
    strong_current="68a8da582d4ea70cc075f836201261a5339aeb3620a55bb1b170d1124d44e66f",
    big_waves="482a72e871d04af5bdee88bb73f9da150bb0446becf6af54b95ad59154871ba0",
    big_kicks="9d67525487681905905779aa55b68b4b7b6d5309d45049ccfd6a0c530e5d2243",
)
CQL_NOISE_NPZ_SHA256 = dict(
    none="57fcf7f8b1d6e2a123cc20dc9aed56067ba5f9a40d6eeeda5b8e3291322c8828",
    mild="10983c027774a8fbf128e22351fbddfe2e901cb43292109b2e2785b171dead17",
    strong="4c6ff8d26e918afcad2161525b30708d9d34269ec8a09291dbe1bf458a46d144",
    recorded="d442517c7379049bf3e10b3476a7ddf27c45a6c5f9ab3db55a45c751a08dc884",
    strong_current="ec7e8a699dc08fa98df7574fc2d994c56b5e93dc9f1b2402d63e484a83a4bd89",
    big_waves="79cd4b5b85ec6f74b6b0fbee2aeb2940f54b2b9f8b18cae120310ffd80600cc3",
    big_kicks="bc2fb9a30f9be6fc9f2e1b70ed028a3929d84a68e038dca860c8a641aeb935e2",
)
SCORES_SCHEMA = "shift-hosts-scores-v1"

BankScore = namedtuple("BankScore", "bank idx per_start summary episodes")
BankScore.__doc__ = """One policy on one bank: idx the bank's start indices rolled (in order), per_start
{metric: array}, summary {metric: value}, episodes (when asked) {field: array} of the kept starts from the same
rollout."""

_MEMO = {}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bank_name(bank):
    return BANKS[P.bank_index(bank)]


# ------------------------------------------------------------------------------------------------ starts and anchors

def start_bank_root():
    """(root, kind) of the weighting test's start banks: the data agent's copy (data_copy.start_banks_dir(),
    runs/shift_hosts/source/start_banks) when it exists, else the original (read-only). Refuses (ValueError) a
    start_banks.hdf5 or start_banks.json that differs from the pinned SHA-256."""
    for root, kind in ((COPIED_BANKS, "copy"), (ORIGINAL_BANKS, "original, read-only")):
        if (root / "start_banks.hdf5").is_file():
            got = sha256_file(root / "start_banks.hdf5"), sha256_file(root / "start_banks.json")
            if got != (START_BANKS_HDF5_SHA256, START_BANKS_JSON_SHA256):
                raise ValueError(f"{root}: the start banks differ from the pinned SHA-256 (stop rule S11)")
            return root, kind
    raise FileNotFoundError(f"no start banks in {COPIED_BANKS} or {ORIGINAL_BANKS}")


def weighting_banks():
    """({law: starts float64 [1000, 13]}, start_banks.json record, source) through data.load_start_banks, which checks
    the file and every array against its manifest (and the JSON against its sidecar)."""
    if "weighting" not in _MEMO:
        root, kind = start_bank_root()
        banks, record = D.load_start_banks(root)
        source = dict(root=str(root), kind=kind, hdf5_sha256=START_BANKS_HDF5_SHA256,
                      json_sha256=START_BANKS_JSON_SHA256)
        for law, x in banks.items():
            x.setflags(write=False)
        _MEMO["weighting"] = (banks, record, source)
    return _MEMO["weighting"]


def starts(bank):
    """The bank's 1,000 starts, float64 [1000, 13] (a fresh array)."""
    name = bank_name(bank)
    if name in WEIGHTING:
        return np.array(weighting_banks()[0][name], np.float64)
    return TC.bank_starts(name)


def prereg_anchors():
    """preregistration.json's "base/stall" anchors (the main study's 0 and 100 points), after its pinned SHA-256."""
    if "prereg" not in _MEMO:
        text = PREREG.read_bytes()
        side = Path(str(PREREG) + ".sha256").read_text().split()[0]
        digest = hashlib.sha256(text).hexdigest()
        if digest != PREREG_SHA256 or side != PREREG_SHA256:
            raise ValueError(f"{PREREG} differs from its pinned SHA-256")
        _MEMO["prereg"] = json.loads(text)["anchors"][PREREG_ANCHORS]
    return _MEMO["prereg"]


def anchor(bank):
    """The bank's point scale: dict(scale, bottom, top, bottom_controller, top_controller, source)."""
    name = bank_name(bank)
    if name in WEIGHTING:
        _, record, source = weighting_banks()
        a = record["anchors"][name]["anchor"]
        return dict(scale="weighting", bottom=float(a["random"]), top=float(a["top"]),
                    bottom_controller="random quality on its fixed streams (data.quality_on_bank)",
                    top_controller="stall-aware controller", source=f"{source['root']}/start_banks.json")
    a = prereg_anchors()[name]
    return dict(scale="situation", bottom=float(a["recording"]), top=float(a["top"]),
                bottom_controller="the main study's recording controller", top_controller=a["top_controller"],
                source=f"outputs/shift/preregistration.json anchors {PREREG_ANCHORS}")


def points(J, bank):
    """Points of returns J (per start or a mean) on the bank's own scale, with the scale's own function."""
    a = anchor(bank)
    if a["scale"] == "weighting":
        return W.points(J, dict(random=a["bottom"], top=a["top"]))
    return TR.points(J, dict(recording=a["bottom"], top=a["top"]))


def points_per_return_unit(bank):
    a = anchor(bank)
    return 100.0 / (a["top"] - a["bottom"])


# ------------------------------------------------------------------------------------------------ the scoring cache

def cache_path(bank, root=CACHE):
    return Path(root) / "banks" / f"{P.bank_index(bank)}_{bank_name(bank)}.npz"


def noise_path(bank, root=CACHE):
    return Path(root) / "cql_noise" / f"{P.bank_index(bank)}_{bank_name(bank)}.npz"


def check_cache_files(root=CACHE):
    """Every pinned scoring-cache file (module docstring) against CACHE_NPZ_SHA256 and CQL_NOISE_NPZ_SHA256, through
    its sidecar: {relative name: SHA-256}. Raises ValueError on any other file."""
    out = {}
    for bank in BANKS:
        name = bank_name(bank)
        for path, want in ((cache_path(bank, root), CACHE_NPZ_SHA256[name]),
                           (noise_path(bank, root), CQL_NOISE_NPZ_SHA256[name])):
            got = H.verify_sidecar(path)
            if got != want:
                raise ValueError(f"{path} is not the pinned scoring-cache file (SHA-256 {got}, pinned {want}): the cache "
                                 "was rebuilt or changed after the freeze")
            out[f"{path.parent.name}/{path.name}"] = got
    return out


def load_cache(bank, root=CACHE):
    """The bank's scoring-cache arrays (scoring_cache.py wrote them), after handoff.read's checks, a check that the
    file is the pinned one (CACHE_NPZ_SHA256: no other cache file is ever read) and a check that its starts equal this
    bank's starts bit for bit. Returns {name: array}, with "meta" holding the manifest's meta."""
    key = ("cache", str(root), bank_name(bank))
    if key not in _MEMO:
        arrays, manifest = H.read(cache_path(bank, root))
        if manifest["npz_sha256"] != CACHE_NPZ_SHA256[bank_name(bank)]:
            raise ValueError(f"{cache_path(bank, root)} is not the pinned scoring-cache file (SHA-256 "
                             f"{manifest['npz_sha256']}, pinned {CACHE_NPZ_SHA256[bank_name(bank)]}): the cache was "
                             "rebuilt or changed after the freeze")
        if manifest["meta"].get("bank") != bank_name(bank):
            raise ValueError(f"{cache_path(bank, root)} is not the cache of {bank_name(bank)}")
        if arrays["starts"].tobytes() != starts(bank).tobytes():
            raise ValueError(f"{cache_path(bank, root)}: its starts differ from the bank's")
        arrays["meta"] = manifest["meta"]
        arrays["npz_sha256"] = manifest["npz_sha256"]
        _MEMO[key] = arrays
    return _MEMO[key]


def viewer_starts(bank, root=CACHE):
    """The viewer's fixed starts of a bank (plan section 9.2): the 10 percentile starts, easy to hard, then for the four
    situations the page's existing start (D.start = 11)."""
    c = load_cache(bank, root)
    return np.concatenate([c["viewer/starts"], c["viewer/extra"]]).astype(np.int64)


# ------------------------------------------------------------------------------------------------ per-start quantities

def late_peak(states):
    """Per start: max over steps 10-99 of the norm of the 10 float position and speed numbers (oracle.py:878-884)."""
    S = np.asarray(states)[:, :LENGTH, :TC.DS]
    radius = np.sqrt(np.sum(S * S, axis=2))
    return radius[:, LATE_PEAK_FROM:].max(1)


def costs(states, commands, physics=PHYSICS):
    """Per start: (tracking cost sum_t |s_t|^2, energy cost 0.3 sum_t |a_t|^2), t = 0-99 (oracle.py:878-881)."""
    S = np.asarray(states)[:, :LENGTH, :TC.DS]
    tracking = np.sum(S * S, axis=(1, 2))
    energy = physics.throttle_cost * np.sum(np.asarray(commands) ** 2, axis=(1, 2))
    return tracking, energy


class _AsController(RC.Controller):
    """A plain callable (for example the weighting test's ZeroCommand, which has no act method) as a recorder
    Controller, so recorder.CappedController can step it: act(x, t) = base(x)."""

    def __init__(self, base):
        self.base = base
        self.name = getattr(base, "name", type(base).__name__)

    def __call__(self, x):
        return self.base(x)


def score(policy, bank, *, idx=None, cache=None, keep=None, physics=PHYSICS):
    """A controller's per-start metrics and bank summary (module docstring). policy: anything truth.rollout steps (a
    time-indexed policy such as CQL's needs noise rows matching idx). idx: the bank's start indices to roll, in
    order (default all 1,000). keep: bank indices (each in idx) whose episodes are returned from the same rollout,
    for the viewer. No permission is checked here: score_actor checks it for actor files."""
    if not hasattr(policy, "act"):
        policy = _AsController(policy)
    name = bank_name(bank)
    X = starts(name)
    idx = np.arange(len(X)) if idx is None else np.asarray(idx, np.int64)
    X = X[idx]
    cache = load_cache(name) if cache is None else cache
    run = TR.rollout(policy, X, physics, record=True)
    capped = TR.rollout(RC.CappedController(policy), X, physics)
    pts, cpts = points(run.J, name), points(capped.J, name)
    lp = late_peak(run.states)
    tracking, energy = costs(run.states, run.commands, physics)
    past = TC.past_knee(run.commands, physics)
    J_opt, J_zero = cache["optimum/J"][idx], cache["zero/J"][idx]
    threshold = float(cache["f2/threshold"])
    per = dict(J=run.J, Jd=run.Jd, J_capped=capped.J, points=pts, points_capped=cpts, regret=J_opt - run.J,
               f1=run.J < J_zero, f2=lp > threshold, late_peak=lp, stall_share=past.mean(1), tracking=tracking,
               energy=energy)
    n = len(run.J)
    k = max(1, int(round(WORST_SHARE * n)))
    ppu = points_per_return_unit(name)
    summary = dict(
        bank=name, scale=anchor(name)["scale"], n=n, finite=bool(np.all(np.isfinite(run.J))),
        J_mean=float(run.J.mean()), J_se=float(run.J.std() / math.sqrt(n)), J_discounted_mean=float(run.Jd.mean()),
        points_mean=float(pts.mean()), points_se=float(pts.std() / math.sqrt(n)),
        J_capped_mean=float(capped.J.mean()), points_capped_mean=float(cpts.mean()),
        loss_from_unrecorded_throttle_points=float((cpts - pts).mean()),
        loss_from_unrecorded_throttle_return=float((capped.J - run.J).mean()),
        stall_share=float(past.mean()),
        regret_mean=float(per["regret"].mean()), regret_points_mean=float(per["regret"].mean() * ppu),
        regret_min=float(per["regret"].min()), optimum_J_mean=float(J_opt.mean()),
        f1_rate=float(per["f1"].mean()), f2_rate=float(per["f2"].mean()), f2_threshold=threshold,
        worst10_mean=float(np.sort(run.J)[:k].mean()), p10=float(np.quantile(run.J, LOW_PERCENTILE)),
        tracking_mean=float(tracking.mean()), energy_mean=float(energy.mean()), late_peak_mean=float(lp.mean()),
        max_cost_identity_error=float(np.max(np.abs(tracking + energy + run.J))) if n else 0.0)
    episodes = None
    if keep is not None:
        keep = np.asarray(keep, np.int64)
        where = {int(s): i for i, s in enumerate(idx)}
        if any(int(s) not in where for s in keep):
            raise ValueError("every kept start must be among the rolled starts")
        pos = np.array([where[int(s)] for s in keep], np.int64)
        commands = run.commands[pos]
        episodes = dict(start=keep, states=run.states[pos], commands=commands, thrust=TC.thrust(commands, physics),
                        stalled=TC.stalled(commands, physics), rewards=run.rewards[pos],
                        running_return=np.cumsum(run.rewards[pos], axis=1), J=run.J[pos])
    return BankScore(name, idx, per, summary, episodes)


def paired(a, b):
    """The paired difference a - b of two per-start arrays from the same starts: mean, standard error std / sqrt(n)
    and n (plan section 5.4, "paired standard errors over the 1,000 starts")."""
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    if d.ndim != 1 or len(d) == 0:
        raise ValueError("paired() takes two aligned per-start arrays")
    return dict(mean=float(d.mean()), se=float(d.std() / math.sqrt(len(d))), n=int(len(d)))


# ------------------------------------------------------------------------------------------------ actor files

def require_permission(permission, actor, *, rollout=True):
    """guard.MADE_UP with a made-up actor file that no run trained, or a gate.Approval that still verifies; else
    gate.ApprovalError. A trained actor (the runner marks every actor file it writes trained true, kind "run-snapshot")
    is a learned policy even when it learned made-up arrays, so guard.MADE_UP never rolls it out in the simulator (no
    learned policy's return before the approval); rollout=False (open_loop_states on given states with no bank: no
    simulator step) lets guard.MADE_UP read a made-up trained actor's commands. Returns a JSON-ready description."""
    from experiments.shift_hosts import gate, guard

    if permission is guard.MADE_UP:
        if actor.meta.get("made_up") is not True:
            raise gate.ApprovalError("refused: guard.MADE_UP scores only actor files whose meta says made_up true; "
                                     "a learned actor is scored only with the user's verified approval "
                                     "(gate.require_run_approval)")
        trained = actor.meta.get("trained") is True or actor.meta.get("kind") == "run-snapshot"
        if trained and rollout:
            raise gate.ApprovalError("refused: this actor file was trained by a run (meta trained true), so rolling it "
                                     "out would produce a learned policy's return; that needs the user's verified "
                                     "approval (gate.require_run_approval), even for a run on made-up arrays")
        return dict(kind="made-up")
    if isinstance(permission, gate.Approval):
        gate.verify(permission)
        return dict(kind="approval", **permission.describe())
    raise gate.ApprovalError(f"refused: scoring an actor needs guard.MADE_UP (made-up parameters) or a verified "
                             f"gate.Approval, not {permission!r}")


def score_actor(actor_path, permission, *, banks=BANKS, keep_viewer=False, out=None):
    """Score an actor parameter file on the banks, every readout of its learner (CQL: "scored", the declared-noise
    command that the claims use, and "tanh_mean", the readout). Returns {(readout, bank): BankScore}; with out, also
    writes them (write_scores)."""
    actor = A32.read_actor_file(actor_path)
    who = require_permission(permission, actor)
    result = {}
    for readout in A32.READOUTS[actor.learner]:
        for bank in banks:
            cache = load_cache(bank)
            policy = P.adapter(actor, bank=bank, readout=readout)
            keep = viewer_starts(bank) if keep_viewer else None
            result[(readout, bank_name(bank))] = score(policy, bank, cache=cache, keep=keep)
    if out is not None:
        write_scores(out, actor, result, who)
    return result


def code_sha256():
    """SHA-256 of this package's scoring files and the repository files they run (for records)."""
    here = Path(__file__).resolve().parent
    files = [here / n for n in ("evaluate.py", "policies.py", "actors_f32.py", "handoff.py", "scoring_cache.py")]
    files += [REPO / n for n in ("experiments/shift/truth.py", "experiments/shift/thruster_chain.py",
                                 "experiments/shift/recorder.py", "experiments/shift_iw/data.py",
                                 "experiments/shift_iw/world.py", "outputs/shift/oracle/oracle.py")]
    return {str(f.relative_to(REPO)): sha256_file(f) for f in files if f.is_file()}


def write_scores(out, actor, result, permission_record):
    """Write the scores of score_actor with handoff.write: per-start arrays "<readout>/<bank>/<metric>" (and the kept
    episodes under "<readout>/<bank>/episodes/<field>"), the summaries and provenance in the meta."""
    arrays, summaries = {}, {}
    for (readout, bank), s in result.items():
        arrays[f"{readout}/{bank}/start"] = s.idx
        for k, v in s.per_start.items():
            arrays[f"{readout}/{bank}/{k}"] = np.asarray(v)
        if s.episodes is not None:
            for k, v in s.episodes.items():
                arrays[f"{readout}/{bank}/episodes/{k}"] = np.asarray(v)
        summaries.setdefault(readout, {})[bank] = s.summary
    meta = dict(schema=SCORES_SCHEMA, learner=actor.learner, actor_npz_sha256=actor.npz_sha256, actor_meta=actor.meta,
                permission=permission_record, summaries=summaries,
                cache={b: load_cache(b)["npz_sha256"] for (_, b) in result},
                anchors={b: anchor(b) for (_, b) in result}, code_sha256=code_sha256(),
                environment=dict(python=sys.version.split()[0], numpy=np.__version__))
    return H.write(out, arrays, meta=meta)


# ------------------------------------------------------------------------------------------------ coverage hand-offs

def _transition_arrays(states, commands, physics=PHYSICS):
    """Exact one-step transitions from float64 states and applied commands: (reward64, next_state) by
    thruster_chain.dynamics, the function env.step and truth.rollout call."""
    nxt, r = TC.dynamics(np.asarray(states, np.float64), np.clip(np.asarray(commands, np.float64), -1.0, 1.0),
                         physics)
    return r, nxt


def visited_transitions(actor_path, out_path, permission, *, bank="strong", n_starts=200, readout="scored"):
    """Coverage at the actor's own visited states (plan section 5.4), the scoring environment's half: the scored
    rollout of the float64 adapter from the bank's first n_starts starts (default: the first 200 strong-mix starts),
    every step written to out_path (handoff.py): state float64 [n, 13] and obs float32 (the deployed observation),
    action float64 [n, 4] (the applied command) and action32 float32, reward64 float64 and reward float32, next_state
    float64 and next_obs float32 (the simulator's exact next state), start and t int32, and for CQL's scored readout
    eps float32 [n, 4]. The training environment computes the learner's y, q and U at these rows and hands them back
    (band_readout.py, plan section 3.1). Returns the manifest."""
    actor = A32.read_actor_file(actor_path)
    who = require_permission(permission, actor)
    idx = np.arange(int(n_starts))
    policy = P.adapter(actor, bank=bank, starts=idx, readout=readout)
    run = TR.rollout(policy, starts(bank)[idx], PHYSICS, record=True)
    S = run.states[:, :LENGTH].reshape(-1, TC.DX)
    A = run.commands.reshape(-1, TC.DA)
    r, nxt = _transition_arrays(S, A)
    if not (np.array_equal(nxt, run.states[:, 1:].reshape(-1, TC.DX)) and np.array_equal(r, run.rewards.reshape(-1))):
        raise AssertionError("the one-step transitions differ from the rollout's")
    arrays = dict(state=S, obs=S.astype(np.float32), action=A, action32=A.astype(np.float32), reward64=r,
                  reward=r.astype(np.float32), next_state=nxt, next_obs=nxt.astype(np.float32),
                  start=np.repeat(idx, LENGTH).astype(np.int32),
                  t=np.tile(np.arange(LENGTH, dtype=np.int32), len(idx)))
    noise = P.bank_noise(actor, bank, idx, readout)
    if noise is not None:
        arrays["eps"] = noise[:, :LENGTH].reshape(-1, TC.DA)
    meta = dict(kind="coverage-visited-transitions", learner=actor.learner, readout=readout, bank=bank_name(bank),
                starts=int(n_starts), actor_npz_sha256=actor.npz_sha256, made_up=actor.meta.get("made_up"),
                permission=who, rows=int(len(S)))
    return H.write(out_path, arrays, meta=meta)


def actor_transitions(actor_path, states, out_path, permission, *, eps=None, readout="scored"):
    """Coverage at (recorded state, the actor's command) (plan section 5.4), the scoring environment's half: at given
    float64 states (for example a prepared set's state column), the float64 adapter's command (CQL's scored readout
    needs eps float32 [n, 4], one noise row per state), the reward and the simulator's exact next state, written
    with the same fields as visited_transitions (start and t are -1). Returns the manifest."""
    actor = A32.read_actor_file(actor_path)
    who = require_permission(permission, actor)
    S = np.asarray(states, np.float64)
    if actor.learner == "cql" and readout == "scored":
        if eps is None:
            raise ValueError("CQL's scored command needs one noise row per state (eps float32 [n, 4])")
        eps = np.asarray(eps, np.float32)
        A = P.CQLPolicy(actor.params, eps[:, None, :]).act(S, 0)
    else:
        eps = None
        A = P.adapter(actor, readout=readout)(S)
    A = np.clip(np.asarray(A, np.float64), -1.0, 1.0)
    r, nxt = _transition_arrays(S, A)
    arrays = dict(state=S, obs=S.astype(np.float32), action=A, action32=A.astype(np.float32), reward64=r,
                  reward=r.astype(np.float32), next_state=nxt, next_obs=nxt.astype(np.float32),
                  start=np.full(len(S), -1, np.int32), t=np.full(len(S), -1, np.int32))
    if eps is not None:
        arrays["eps"] = eps
    meta = dict(kind="coverage-actor-transitions", learner=actor.learner, readout=readout,
                actor_npz_sha256=actor.npz_sha256, made_up=actor.meta.get("made_up"), permission=who,
                rows=int(len(S)))
    return H.write(out_path, arrays, meta=meta)


# ------------------------------------------------------------------------------------------------ S7

def open_loop_states(actor_path, out_path, permission, *, banks=BANKS, starts_by_bank=None, extra_states=None,
                     extra_eps=None):
    """S7's open loop, the scoring environment's first half: roll the float64 adapter from the given starts of each
    bank (starts_by_bank {bank: indices}; default: the viewer's fixed starts) and write every state the actor acted in
    (t = 0-99) to out_path: state float64 [n, 13], obs float32 [n, 13] (the deployed observation), bank int8, start
    int32, t int32 (-1 for extra rows), and for CQL eps float32 [n, 4] (the declared noise at that start and step).
    extra_states (float64 [m, 13]) adds made-up states, with extra_eps float32 [m, 4] for CQL. Returns the manifest."""
    actor = A32.read_actor_file(actor_path)
    who = require_permission(permission, actor, rollout=bool(len(tuple(banks))))
    cql = actor.learner == "cql"
    state, bank_col, start_col, t_col, eps = [], [], [], [], []
    for bank in banks:
        idx = viewer_starts(bank) if starts_by_bank is None else np.asarray(starts_by_bank[bank_name(bank)], np.int64)
        policy = P.adapter(actor, bank=bank, starts=idx)
        run = TR.rollout(policy, starts(bank)[idx], PHYSICS, record=True)
        S = run.states[:, :LENGTH]
        state.append(S.reshape(-1, TC.DX))
        bank_col.append(np.full(S.shape[0] * LENGTH, P.bank_index(bank), np.int8))
        start_col.append(np.repeat(idx, LENGTH).astype(np.int32))
        t_col.append(np.tile(np.arange(LENGTH, dtype=np.int32), len(idx)))
        if cql:
            eps.append(P.cql_noise(bank)[idx][:, :LENGTH].reshape(-1, TC.DA))
    if extra_states is not None:
        extra_states = np.asarray(extra_states, np.float64)
        m = len(extra_states)
        state.append(extra_states)
        bank_col.append(np.full(m, -1, np.int8))
        start_col.append(np.full(m, -1, np.int32))
        t_col.append(np.full(m, -1, np.int32))
        if cql:
            if extra_eps is None:
                raise ValueError("CQL's extra states need their noise (extra_eps float32 [m, 4])")
            eps.append(np.asarray(extra_eps, np.float32))
    state = np.concatenate(state)
    arrays = dict(state=state, obs=state.astype(np.float32), bank=np.concatenate(bank_col),
                  start=np.concatenate(start_col), t=np.concatenate(t_col))
    if cql:
        arrays["eps"] = np.concatenate(eps)
    meta = dict(kind="s7-open-loop-states", learner=actor.learner, actor_npz_sha256=actor.npz_sha256,
                made_up=actor.meta.get("made_up"), permission=who, banks=[bank_name(b) for b in banks],
                rows=int(len(state)))
    return H.write(out_path, arrays, meta=meta)


def open_loop_compare(states_path, commands_path, actor_path, tol=S7_OPEN_TOL):
    """S7's open loop, the scoring environment's second half: the float32 restatement's commands at the saved float32
    observations against the learner's own (actors_f32.open_loop_commands, training environment). Passes when every
    command differs by at most tol (1e-7) and all are finite. Checks that the three files belong together."""
    states, sman = H.read(states_path)
    theirs, cman = H.read(commands_path)
    actor = A32.read_actor_file(actor_path)
    if not (sman["meta"].get("actor_npz_sha256") == actor.npz_sha256 == cman["meta"].get("actor_npz_sha256")
            and cman["meta"].get("states_npz_sha256") == sman["npz_sha256"]):
        raise ValueError("the states, commands and actor files do not belong together")
    A32.configure()
    out = dict(learner=actor.learner, rows=int(len(states["obs"])), tol=tol, settings=A32.settings(),
               other_side=dict(device=cman["meta"].get("device"), xla_flags=cman["meta"].get("xla_flags"),
                               packages=cman["meta"].get("packages")))
    pairs = [("scored", "commands", states.get("eps"))]
    if actor.learner == "cql":
        pairs.append(("tanh_mean", "commands_tanh_mean", None))
    passed = True
    for readout, name, eps in pairs:
        mine = A32.commands(actor, states["obs"], eps, readout)
        other = theirs[name]
        d = np.abs(mine.astype(np.float64) - other.astype(np.float64))
        ok = bool(np.all(np.isfinite(mine)) and np.all(np.isfinite(other)) and d.max() <= tol)
        out[readout] = dict(max_abs=float(d.max()), mean_abs=float(d.mean()),
                            bit_identical_rows=float(np.mean(np.all(mine == other, axis=1))),
                            worst_row=int(np.argmax(d.max(1))), passed=ok)
        passed = passed and ok
    out["passed"] = passed
    return out


def closed_loop_check(actor_path, permission, *, banks=BANKS, idx=None, tol_points=S7_CLOSED_TOL_POINTS):
    """S7's closed loop (scoring environment): the float32 restatement (actors_f32.Float32Policy) and the float64
    adapter roll from the same starts (idx; default every start) of each bank; passes when every start's points
    differ by at most tol_points (0.05) and every return is finite. CQL: both readouts, the scored one with the
    declared noise. Returns per readout and bank the largest differences in points and return units."""
    actor = A32.read_actor_file(actor_path)
    who = require_permission(permission, actor)
    A32.configure()
    out, passed = dict(learner=actor.learner, tol_points=tol_points, permission=who, settings=A32.settings()), True
    for readout in A32.READOUTS[actor.learner]:
        for bank in banks:
            X = starts(bank)
            ids = np.arange(len(X)) if idx is None else np.asarray(idx, np.int64)
            p64 = P.adapter(actor, bank=bank, starts=ids, readout=readout)
            p32 = A32.Float32Policy(actor, eps=P.bank_noise(actor, bank, ids, readout), readout=readout)
            J64 = TR.rollout(p64, X[ids], PHYSICS).J
            J32 = TR.rollout(p32, X[ids], PHYSICS).J
            dp = np.abs(points(J32, bank) - points(J64, bank))
            ok = bool(np.all(np.isfinite(J32)) and np.all(np.isfinite(J64)) and dp.max() <= tol_points)
            out[f"{readout}/{bank_name(bank)}"] = dict(starts=int(len(ids)), max_points=float(dp.max()),
                                                       max_return=float(np.abs(J32 - J64).max()),
                                                       mean_points=float(dp.mean()), passed=ok)
            passed = passed and ok
    out["passed"] = passed
    return out
