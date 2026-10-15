"""Part B's learners: the TD3+BC host, BCA's band on the frozen host critic, the weighted thresholds, the room check and
the co-trained BCA arms. A thin layer over experiments/shift/learners.py, which is imported unchanged.

Terms (iw/SPEC.md):
  host            plain TD3+BC (algorithms/td3_bc.py through learners.cotrain): alpha 2.5, 100,000 updates, batch 256,
                  discount 0.98. Its critic is q = min(Q1, Q2).
  band            BCA's band q +- U(s, a) on the frozen final host critic: td3_bc_bca.fit_scale for 20,000 batches of
                  kept training rows, then td3_bc_bca.refresh, unchanged, on the kept calibration bank. This restates
                  learners.bca_band (which is locked to v1's own approval file) under this package's gate.
  score           |y - q| / (unit x max(scale, 1e-6)) on a bank row, y = reward + 0.98 x target critic's next value
                  with one draw of target noise (the refresh's own key)
  threshold R     wbcp.calibrate of the bank scores (miscoverage 0.1, credibility 0.95, 1,000 draws); +inf means
                  "cannot certify"
  width U         R x unit x max(scale, 1e-6); dose = 1 + 0.5 U / (U + unit), read by td3_bc_bca.bc_readout
  weighted arms   the same band and the same bank scores; only R differs. After BCA's unchanged refresh, the bank
                  scores are recomputed exactly as the refresh computed them, and wbcp.calibrate is called again with
                  the same random stream and the bank weights. The scale network, the unit and every other part of
                  BCA's signal stay as they are. With weights 1 and test mass 1 the recomputed threshold equals BCA's
                  own bit for bit (checked at every use). Two ways to set the test mass (spec version 4):
                    per step (the method, the owner's choice): each step's own weight is its test mass, so its
                      threshold is R(w(x)), read from a table of R on a log-spaced grid of test masses
                      (coverage.threshold_table and coverage.lookup: the smallest grid mass at least the step's own).
                      For coverage, each test step uses its own threshold (per_step_readout). In the BCA arms each
                      training row's dose uses its own threshold, width_i = R(w_i) x unit x scale(s_i, a_i)
                      (per_row_dose): a change to how BCA consumes its band, declared in SPEC.md. A row whose R is +inf
                      (cannot certify) gets the largest dose, 1 + blend = 1.5 (BCA itself would give it 1).
                    single (version 3's, kept as a reported comparison): one test mass for every step, the mean weight
                      over test-law steps outside the ball; the four threshold fields of the frozen reference
                      (threshold, lambda_hat, lambda_hpd, n_eff) are replaced and BCA reads its doses unchanged.
  room check      v1's actor-only runs (learners.ActorOnly) against the frozen gapped host critic, from an actor cloned
                  on the recorded commands: no extra caution ('none'), caution placed by the exact over-preference in
                  BCA's dose form ('placed', refreshed every 4,000 steps), even caution x1.1, x1.25 and x1.5, and even
                  caution matched in strength to 'placed' (learners.match_even between those rungs; it never
                  extrapolates, so a placed arm stronger than x1.5 leaves the matched arm missing, which the report
                  says). No BCA runs there.
  lower edge      (spec version 5, part B) a second consumer of the band, the penalty k [U(s, pi(s)) - U(s, a_i)]_+ in
                  the actor's objective, and its room check with an exact oracle: lower_edge.py.
  BCA arms        co-trained TD3+BC+BCA (td3_bc_bca's own update through learners.bca_train_step, placement 'own',
                  blend 0.5), refreshes at 10,000 updates and every 5,000 after. The single-threshold weighted arm
                  replaces R after each unchanged refresh; the per-row arm recomputes its table of R after each
                  unchanged refresh and runs bca_update_per_row (td3_bc_bca.update line for line except the dose,
                  which is per_row_dose). With every weight 1 the per-row arm equals the unweighted arm bit for bit.

The gate. Every function that fits a band, computes a threshold on band scores, reads a width or a dose, or runs a BCA
update calls require_gate first (lower_edge.py's band readers and lower-edge BCA runner too). It passes only with
  (a) a verified Approval: runs/shift_iw_v5/approval.json, written by the orchestrator after the user has seen the
      pass marks, with a sha256sum sidecar, holding the user's chat approval verbatim, the SHA-256 of iw/spec.json
      (which must also match spec.json's own sidecar), the SHA-256 of every code file (code_hashes(): this package, its
      launch.sh and every repository module the stage runner loads; the code that will fit, judge and report must be
      the code the user approved) and the stages it covers (BCA_STAGES: 'band', 'coverage'; and, each only after its
      own go from the user, 'bca' for the capped pull's arms, 'bca_lower_edge' for the lower edge's arms, and
      'bca_estimated' / 'bca_lower_edge_estimated' for their optional arms, each of these four naming the qualities
      that go covers: ARM_SCOPES, Approval.covers); it is re-read from disk at every call; or
  (b) the SYNTHETIC token, which works only on data whose content is registered as synthetic (register_synthetic:
      synthetic_prepared registers its own arrays; test fixtures register theirs). The gate recomputes the content
      hash of the train and bank arrays at every entry (fit_band, load_band, bca_cotrain), so relabelling real data
      as synthetic is refused.
Nothing in this package writes approval.json (`run.py approval-template` prints the body to fill in).

Seeds: learners' streams np.random.default_rng([20261016, seed, tag]) (settings() sets learners' master seed to the
spec's 20261016; learners' own tags 31-41 do not meet data.py's tags 1-11); fresh target-noise keys for coverage use
tag 51.
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

import experiments.shift  # noqa: F401  (gymnasium path and version checks)

import jax
import jax.numpy as jnp

import algorithms.td3_bc as BASE_TD3
import algorithms.td3_bc_bca as P
from calibration import reference as REF
from calibration.dose import level_critic_dose, tree_finite
from calibration.network import Calibrator
from experiments.shift import learners as LR
from experiments.shift import truth as TR
from experiments.shift_iw import coverage as CV
from experiments.shift_iw import world as W
from runtime.networks import Transition

DX, DA = LR.DX, LR.DA
SPEC_JSON = Path(__file__).resolve().parent.parent / "iw" / "spec.json"
ALPHA = 2.5
TAG_Y = 51                      # fresh target-noise keys for coverage (one per test group)
KEY_TARGET = 1213156420         # td3_bc_bca.refresh's fold_in for the bank's target noise
KEY_POSTERIOR = 1347375956      # td3_bc_bca.refresh's fold_in for the posterior's random stream
APPROVAL_KIND = "shift-iw-approval"
BCA_STAGES = ("band", "coverage", "bca", "bca_estimated", "bca_lower_edge", "bca_lower_edge_estimated")
# the scopes that train BCA arms: each comes only from the user's own go for that kind of arm, and names the qualities
# the go covers (approval 'qualities': {scope: [quality, ...]}); the stage runner refuses a quality not named
ARM_SCOPES = ("bca", "bca_estimated", "bca_lower_edge", "bca_lower_edge_estimated")
SYNTHETIC_WORLD = "synthetic arrays (not ThrusterChain data)"
REFUSAL = ("refused: BCA's band runs on ThrusterChain data only after the user has seen the pass marks and approved. "
           "The orchestrator records that approval in runs/shift_iw_v5/approval.json (with a sha256sum sidecar, the "
           "user's words verbatim, the SHA-256 of iw/spec.json and of the code, and the approved stages); no agent "
           "creates it")
PACKAGE = Path(__file__).resolve().parent
REPO = Path(LR.__file__).resolve().parents[2]
CODE_FILES_IW = ("__init__.py", "world.py", "data.py", "coverage.py", "classifier.py", "learners_iw.py",
                 "lower_edge.py", "run.py", "launch.sh")
# Every repository module that importing the stage runner loads (its whole import closure, package __init__ files
# included), plus experiments/shift/acceptance.py and the two modules it imports at load time (detection.py,
# stage0.py): learners.py imports acceptance lazily, in v1's own approval check, which this package never calls.
# test_learners_iw.CodeHashTest checks in a fresh process that every repository module loaded is listed here.
CODE_FILES_REUSED = (
    "algorithms/__init__.py", "algorithms/td3_bc.py", "algorithms/td3_bc_bca.py",
    "calibration/__init__.py", "calibration/bank.py", "calibration/dose.py", "calibration/network.py",
    "calibration/reference.py", "calibration/wbcp.py",
    "experiments/shift/__init__.py", "experiments/shift/acceptance.py", "experiments/shift/dataset.py",
    "experiments/shift/detection.py", "experiments/shift/learners.py", "experiments/shift/recorder.py",
    "experiments/shift/shift.py", "experiments/shift/stage0.py", "experiments/shift/thruster_chain.py",
    "experiments/shift/truth.py",
    "experiments/signal/__init__.py", "experiments/signal/gapped_lq.py", "experiments/signal/glq_match.py",
    "experiments/signal/lq_harness.py",
    "experiments/truth/__init__.py", "experiments/truth/gap.py", "experiments/truth/labeller.py",
    "experiments/truth/pools.py", "experiments/truth/state_restore.py",
    "runtime/__init__.py", "runtime/networks.py", "runtime/published.py", "runtime/rewards.py")


class ApprovalError(PermissionError):
    """Raised by every BCA path without a valid approval (or the synthetic token on synthetic arrays)."""


class SpecError(RuntimeError):
    """Raised when iw/spec.json has no sidecar or differs from it."""


# ------------------------------------------------------------------------------------------------ settings

def settings(**override):
    """learners.LearnSettings with the spec's master seed (20261016) and alpha 2.5; every other number is learners'
    own default, which is the spec's (100,000 updates, batch 256, discount 0.98, tau 0.005, Adam 3e-4, policy delay
    2, target noise 0.2 clipped at 0.5, cloning 20,000 steps, actor-only 20,000 steps refreshed every 4,000 with
    600-step rollouts, 5,000 strength states, band fit 20,000 steps, BCA warm-up 10,000 and refresh every 5,000)."""
    return replace(LR.LearnSettings(seed=W.MASTER_SEED, alpha=ALPHA), **override)


def models():
    """(actor, critic, scale network) as td3_bc_bca and learners.bca_initial_state build them."""
    return LR.ACTOR, LR.CRITIC, Calibrator(jnp.zeros(DX), jnp.ones(DX), state_dep=True)


# ------------------------------------------------------------------------------------------------ gate

def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sidecar(path):
    return Path(str(path) + ".sha256")


def code_hashes():
    """SHA-256 of every code file the experiment runs: this package (with launch.sh) and every repository module the
    stage runner loads (CODE_FILES_REUSED). Installed libraries (JAX, NumPy, ...) are not hashed; every stage record
    holds their versions. Stage records carry these hashes, the approval names them and the report refuses records
    made by other code."""
    out = {f"experiments/shift_iw/{n}": sha256_file(PACKAGE / n) for n in CODE_FILES_IW}
    out.update({n: sha256_file(REPO / n) for n in CODE_FILES_REUSED})
    return out


def spec_sha256(spec_path=SPEC_JSON):
    """The SHA-256 of iw/spec.json after checking it against its sidecar (spec.json.sha256, sha256sum format).
    Raises SpecError when the file or the sidecar is missing, or when they differ."""
    spec_path = Path(spec_path)
    if not spec_path.is_file():
        raise SpecError(f"refused: no specification at {spec_path}")
    side = sidecar(spec_path)
    if not side.is_file():
        raise SpecError(f"refused: {spec_path} has no sidecar {side.name}; the orchestrator freezes the specification "
                        "first (python -m experiments.shift_iw.run freeze-spec)")
    digest = sha256_file(spec_path)
    words = side.read_text(encoding="utf8").split()
    if not words or words[0] != digest:
        raise SpecError(f"refused: {spec_path} differs from its sidecar (the specification changed after it was "
                        "frozen)")
    try:
        json.loads(spec_path.read_text(encoding="utf8"))
    except ValueError as error:
        raise SpecError(f"refused: {spec_path} is not JSON") from error
    return digest


@dataclass(frozen=True)
class Approval:
    """A verified approval file: where it is, the specification it approves, its own SHA-256, the stages it covers,
    the SHA-256 of the user's words and, for each BCA-arm scope it holds, the qualities the user's go covers
    (((scope, (quality, ...)), ...), sorted)."""

    path: str
    spec_path: str
    spec_sha256: str
    file_sha256: str
    scope: tuple
    text_sha256: str
    qualities: tuple = ()

    def covers(self, stage, quality):
        """Whether the approval covers `stage` for `quality`: a BCA-arm scope (ARM_SCOPES) only for the qualities its
        go names; 'band' and 'coverage' for every quality."""
        if stage not in self.scope:
            return False
        return stage not in ARM_SCOPES or quality in dict(self.qualities).get(stage, ())


def verify_approval(path, spec_path=SPEC_JSON, stage=None):
    """Read and verify an approval file. It must exist and match its sidecar (path + '.sha256'); be JSON with
    kind 'shift-iw-approval', a non-empty 'user_approval_verbatim', 'spec_sha256' equal to the SHA-256 of spec.json
    (which must match spec.json's own sidecar), 'code_sha256' equal to code_hashes() now (no code file changed since
    the user approved), 'scope', a list of stages from BCA_STAGES, and, for every BCA-arm scope in it (ARM_SCOPES),
    'qualities'[scope]: a non-empty list of the qualities that go covers (world.QUALITIES); `stage`, when given, must
    be in the scope. Raises ApprovalError otherwise."""
    path = Path(path)
    if not path.is_file():
        raise ApprovalError(f"{REFUSAL} (no approval file at {path})")
    try:
        spec_digest = spec_sha256(spec_path)
    except SpecError as error:
        raise ApprovalError(f"{REFUSAL} ({error})") from error
    blob = path.read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    side = sidecar(path)
    words = side.read_text(encoding="utf8").split() if side.is_file() else []
    if not words or words[0] != digest:
        raise ApprovalError(f"{REFUSAL} (the approval file is missing its sidecar or differs from it)")
    try:
        body = json.loads(blob.decode("utf8"))
    except ValueError as error:
        raise ApprovalError(f"{REFUSAL} (the approval file is not JSON)") from error
    if not isinstance(body, dict) or body.get("kind") != APPROVAL_KIND:
        raise ApprovalError(f"{REFUSAL} (not a shift_iw approval file)")
    text = body.get("user_approval_verbatim")
    if not isinstance(text, str) or not text.strip():
        raise ApprovalError(f"{REFUSAL} (the approval file holds no verbatim approval from the user)")
    if body.get("spec_sha256") != spec_digest:
        raise ApprovalError(f"{REFUSAL} (the approval names another specification: its spec_sha256 differs from "
                            f"the SHA-256 of {spec_path})")
    code, now = body.get("code_sha256"), code_hashes()
    if code != now:
        changed = (sorted(k for k in set(now) | set(code) if code.get(k) != now.get(k)) if isinstance(code, dict)
                   else "the approval holds no code_sha256 mapping")
        raise ApprovalError(f"{REFUSAL} (the code differs from the code the user approved: {changed})")
    scope = body.get("scope")
    if not isinstance(scope, list) or not scope or any(s not in BCA_STAGES for s in scope):
        raise ApprovalError(f"{REFUSAL} (the approval's scope must be a non-empty list of stages from {BCA_STAGES})")
    named = body.get("qualities", {})
    if not isinstance(named, dict) or any(k not in scope or k not in ARM_SCOPES for k in named):
        raise ApprovalError(f"{REFUSAL} (the approval's 'qualities' must map BCA-arm scopes it holds to qualities)")
    qualities = []
    for s in scope:
        if s not in ARM_SCOPES:
            continue
        q = named.get(s)
        if not isinstance(q, list) or not q or any(x not in W.QUALITIES for x in q) or len(set(q)) != len(q):
            raise ApprovalError(f"{REFUSAL} (the approval's {s!r} scope must name the qualities the user's go covers, "
                                f"from {W.QUALITIES})")
        qualities.append((s, tuple(sorted(q))))
    if stage is not None and stage not in scope:
        raise ApprovalError(f"{REFUSAL} (the approval does not cover the {stage!r} stage; it covers {scope})")
    return Approval(str(path.resolve()), str(Path(spec_path).resolve()), spec_digest, digest, tuple(scope),
                    hashlib.sha256(text.encode("utf8")).hexdigest(), tuple(sorted(qualities)))


class _Synthetic:
    """The token that opens the BCA paths on registered synthetic arrays only (register_synthetic)."""

    def __repr__(self):
        return "SYNTHETIC"


SYNTHETIC = _Synthetic()
_SYNTHETIC_DATA = set()   # content hashes of registered synthetic train and bank arrays


def data_sha256(prepared):
    """Content hash of a Prepared's train and bank transitions (the arrays a band or a BCA arm is fitted on)."""
    return LR.tree_sha256((prepared.train, prepared.bank))


def register_synthetic(prepared):
    """Register a Prepared of synthetic arrays (world SYNTHETIC_WORLD) so the SYNTHETIC token opens BCA on exactly
    these arrays. Only synthetic_prepared and test fixtures call it. Returns the Prepared."""
    if prepared.world != SYNTHETIC_WORLD:
        raise ApprovalError(f"{REFUSAL} (only synthetic arrays can be registered, not {prepared.world!r})")
    _SYNTHETIC_DATA.add(data_sha256(prepared))
    return prepared


def require_gate(approval, data, stage):
    """Refuse unless `approval` is a verified Approval that still verifies from disk (unchanged) and covers `stage`,
    or the SYNTHETIC token on synthetic data: `data` is a Prepared (its train and bank arrays must be registered by
    content and labelled SYNTHETIC_WORLD) or a Band made by fit_band or load_band on such data. Returns the
    approval."""
    if approval is SYNTHETIC:
        world = getattr(data, "world", None)
        if world != SYNTHETIC_WORLD:
            raise ApprovalError(f"{REFUSAL} (the synthetic token opens BCA on synthetic arrays only, not on {world!r})")
        digest = data.data_sha256 if isinstance(data, Band) else data_sha256(data)
        if digest not in _SYNTHETIC_DATA:
            raise ApprovalError(f"{REFUSAL} (the synthetic token opens BCA on registered synthetic arrays only; the "
                                "content of these arrays is not registered)")
        return approval
    if not isinstance(approval, Approval):
        raise ApprovalError(REFUSAL)
    again = verify_approval(approval.path, approval.spec_path, stage)
    if again != approval:
        raise ApprovalError(f"{REFUSAL} (the approval file or the specification changed since it was verified)")
    return approval


# ------------------------------------------------------------------------------------------------ synthetic data

def synthetic_prepared(n=2048, bank=300, eval_states=256, seed=0, rep=0):
    """A learners.Prepared of small random arrays for unit tests and timing (world SYNTHETIC_WORLD). Not ThrusterChain
    data: states are floats N(0, 0.5^2), a current U[0.1, 1.1] at index 10 and waves U[-0.3, 0.3]; commands
    U[-0.9, 0.9]; rewards a fixed quadratic plus noise; next states the state plus noise. Stream:
    default_rng([20261016, 7777, seed])."""
    rng = np.random.default_rng([W.MASTER_SEED, 7777, int(seed)])

    def block(m):
        s = np.concatenate([0.5 * rng.standard_normal((m, 10)), rng.uniform(W.C_LO, W.C_HI, (m, 1)),
                            rng.uniform(-0.3, 0.3, (m, 2))], 1).astype(np.float32)
        a = rng.uniform(-0.9, 0.9, (m, DA)).astype(np.float32)
        r = (-(s[:, :10] ** 2).sum(1) - 0.1 * (a ** 2).sum(1) - s[:, 10]
             + 0.1 * rng.standard_normal(m)).astype(np.float32)
        s2 = (s + 0.05 * rng.standard_normal(s.shape)).astype(np.float32)
        return s, a, r, s2

    s, a, r, s2 = block(int(n))
    bs, ba, br, bs2 = block(int(bank))
    es = block(int(eval_states))[0]
    mean, std = s.mean(0), s.std(0) + np.float32(LR.NORM_EPS)

    def trans(o, act, rew, nxt):
        return Transition(jnp.asarray((o - mean) / std), jnp.asarray(act), jnp.asarray(rew),
                          jnp.asarray((nxt - mean) / std), jnp.zeros(len(o), jnp.float32))

    return register_synthetic(LR.Prepared(
        rep=int(rep), world=SYNTHETIC_WORLD, removal="none", physics=W.PHYSICS, spec=W.HOLD_SPEC,
        rows=np.arange(int(n)), states=s.astype(np.float64), actions=a, region=np.zeros(int(n), bool),
        train=trans(s, a, r, s2), bank_rows=np.arange(int(bank)), bank=trans(bs, ba, br, bs2), obs_mean=mean,
        obs_std=std, eval_rows=np.arange(int(eval_states)), eval_obs=jnp.asarray((es - mean) / std),
        sha256=dict(train=LR.tree_sha256((s, a, r, s2)), bank=LR.tree_sha256((bs, ba, br, bs2)))))


def synthetic_bank_current(prepared):
    """The current c (index 10) of a synthetic prepared bank's rows, un-normalised (for exact-weight tests)."""
    obs = np.asarray(prepared.bank.obs, np.float64) * prepared.obs_std + prepared.obs_mean
    return obs[:, 10]


def synthetic_train_current(prepared):
    """The current c (index 10) of a synthetic prepared's kept training rows (its float64 states)."""
    return np.asarray(prepared.states, np.float64)[:, 10]


# ------------------------------------------------------------------------------------------------ host

def train_host(prepared, cfg, rep, *, updates=None, out_dir=None, log=None, name=None):
    """One plain TD3+BC run: learners.cotrain unchanged at alpha 2.5. Saves snapshot_050 and snapshot_100 (the whole
    agent state, flax msgpack) under out_dir. Returns (record, final state, {update: state})."""
    label = name or f"TD3+BC host ({prepared.removal} removal)"
    return LR.cotrain(prepared, cfg, rep, alpha=cfg.alpha, updates=updates, out_dir=out_dir, log=log, name=label)


def load_host(path, cfg, sha256=None):
    """A saved host state (learners.save_tree of the agent state), checked against its SHA-256."""
    return LR.load_tree(path, LR.agent_template(cfg), sha256)


def state_sha256(tree):
    """SHA-256 of a state's values as JAX holds them (every leaf through jnp.asarray first, so a Python integer step
    counter and its int32 array hash alike)."""
    return LR.tree_sha256(jax.tree_util.tree_map(jnp.asarray, tree))


def transitions(log, rows, prepared):
    """Normalised float32 transitions of log rows, with the host's normalisation (learners' rule)."""
    return LR._transition(log, np.asarray(rows, np.int64), prepared.obs_mean, prepared.obs_std)


def _pad(tree, size):
    """Pad every leaf's first axis to `size` by repeating its last row."""
    def pad(x):
        x = jnp.asarray(x)
        extra = size - x.shape[0]
        return x if extra == 0 else jnp.concatenate([x, jnp.repeat(x[-1:], extra, 0)])
    return jax.tree_util.tree_map(pad, tree)


def _chunks(n, chunk):
    return [(i, min(i + chunk, n)) for i in range(0, n, chunk)]


_PROGRAMS = {}


def critic_parts(native, trans, cfg, rep, key, chunk=8192):
    """At every row of `trans`: the target y = reward + 0.98 x min over heads of the target critic at (next state,
    target actor + clipped noise), with one draw of target noise per row (td3_bc_bca.native_target; key
    fold_in(key, chunk index)), and the critic q = min(Q1, Q2) at the recorded command. float64 arrays."""
    args = LR.make_args(cfg, rep)
    mk = ("parts", LR._args_key(args), int(chunk))
    if mk not in _PROGRAMS:
        mods = models()
        _PROGRAMS[mk] = jax.jit(lambda nat, b, k: (P.native_target(args, mods, nat, b, k, 1.0),
                                                   mods[1].apply(nat.critic.params, b.obs, b.action).min(axis=-1)))
    n = len(trans.obs)
    ys, qs = [], []
    for j, (i0, i1) in enumerate(_chunks(n, chunk)):
        part = jax.tree_util.tree_map(lambda x: x[i0:i1], trans)
        y, q = _PROGRAMS[mk](native, _pad(part, chunk), jax.random.fold_in(key, j))
        ys.append(np.asarray(y, np.float64)[:i1 - i0])
        qs.append(np.asarray(q, np.float64)[:i1 - i0])
    if not ys:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(ys), np.concatenate(qs)


def coverage_key(cfg, rep, *tags):
    """A fresh target-noise key for a coverage group: PRNGKey of default_rng([20261016, rep, 51, *tags])."""
    return jax.random.PRNGKey(int(LR.rng_np(cfg, rep, TAG_Y, *tags).integers(2 ** 31 - 1)))


def exact_error(actor_params, prepared, states, actions, q, chunk=8192):
    """|q - Q^pi| at rows: Q^pi the exact float64 value of the host actor (learners.exact_q, 1,000-step rollouts from
    the float64 of the float32 observation, the state the critic sees). Returns (error, Q^pi)."""
    true = LR.exact_q(actor_params, prepared, np.asarray(states, np.float64), np.asarray(actions, np.float64),
                      x64=True, chunk=chunk)
    return np.abs(np.asarray(q, np.float64) - true), true


# ------------------------------------------------------------------------------------------------ band

@dataclass
class Band:
    """BCA's band on one frozen host critic. state: td3_bc_bca.State after the scale fit and BCA's unchanged refresh
    (its posterior is the unweighted reference); refresh_key: the refresh's key (uint32 words); approval: what opened
    the gate; data_sha256: content hash of the train and bank arrays it was fitted or loaded on (data_sha256())."""

    world: str
    rep: int
    removal: str
    approval: object
    state: object
    refresh_key: np.ndarray
    record: dict
    data_sha256: str = None

    @property
    def unit(self):
        return float(self.state.posterior.residual_scale)


def _fit_program(args, config, length):
    mk = ("fit", LR._args_key(args), int(length))
    if mk not in _PROGRAMS:
        mods = models()

        def run(st, train, rows, keys):
            def body(s, inp):
                r, k = inp
                batch = jax.tree_util.tree_map(lambda x: x[r], train)
                s, diag = P.fit_scale(args, config, mods, s, batch, k, 1.0)
                return s, diag["scale_fit_accepted"]

            return jax.lax.scan(body, st, (rows, keys))

        _PROGRAMS[mk] = jax.jit(run)
    return _PROGRAMS[mk]


def fit_band(host_state, prepared, cfg, rep, approval, *, steps=None, chunk=5000, log=None):
    """BCA's band on the frozen host (learners.bca_band restated under this package's gate): td3_bc_bca.fit_scale for
    `steps` (20,000) batches of 256 kept training rows (rows default_rng([20261016, rep, 41]); keys split from
    fold_in(rng, 41)), then one unchanged td3_bc_bca.refresh on the kept calibration bank with key
    fold_in(PRNGKey(default_rng([20261016, rep, 41, 1])), steps). The host's networks never change (checked).
    Returns a Band."""
    require_gate(approval, prepared, "band")
    steps = int(cfg.band_fit_steps if steps is None else steps)
    args = LR.make_args(cfg, rep)
    config = LR.bca_config()
    rng, state, mods = LR.bca_initial_state(args, config, native=host_state)
    rows = LR.rng_np(cfg, rep, LR.TAG_BAND).integers(prepared.n, size=(steps, cfg.batch_size))
    keys = jax.random.split(jax.random.fold_in(rng, LR.TAG_BAND), steps)
    accepted = []
    for i0, i1 in _chunks(steps, int(chunk)):
        state, acc = _fit_program(args, config, i1 - i0)(state, prepared.train, jnp.asarray(rows[i0:i1]),
                                                           keys[i0:i1])
        accepted.append(np.asarray(acc))
        if log:
            log(f"band rep {rep} ({prepared.removal}): scale fit {i1}/{steps}")
    accepted = np.concatenate(accepted) if accepted else np.zeros(0, bool)
    key = jax.random.fold_in(jax.random.PRNGKey(int(LR.rng_np(cfg, rep, LR.TAG_BAND, 1).integers(2 ** 31 - 1))),
                             steps)
    state, diag, diagnostics = P.refresh(args, config, mods, state, prepared.bank, key, 1.0,
                                         heldout_ids=np.arange(len(prepared.bank.obs)))
    host_sha = state_sha256(host_state)
    if state_sha256(state.native) != host_sha:
        raise AssertionError("the band must not change the frozen host")
    record = dict(kind="shift-iw-band", rep=int(rep), removal=prepared.removal, steps=steps,
                  accepted_share=float(accepted.mean()) if len(accepted) else None,
                  refresh={k: bool(v) for k, v in diag.items()},
                  refresh_diagnostics={k: (float(v) if not isinstance(v, (bool, np.bool_)) else bool(v))
                                       for k, v in diagnostics.items()},
                  unit=float(state.posterior.residual_scale), bank_rows=int(len(prepared.bank.obs)),
                  host_sha256=host_sha, bca_settings=LR.BCA_SETTINGS,
                  refresh_key=np.asarray(key, np.uint32).tolist(), data_sha256=data_sha256(prepared))
    return Band(prepared.world, int(rep), prepared.removal, approval, state, np.asarray(key, np.uint32), record,
                record["data_sha256"])


def band_template(host_state, cfg, rep):
    """A td3_bc_bca.State to load a saved band into (the host as its native part)."""
    return LR.bca_initial_state(LR.make_args(cfg, rep), LR.bca_config(), native=host_state)[1]


def load_band(path, sha256, host_state, prepared, cfg, rep, approval, record, stage="coverage"):
    """A saved band (learners.save_tree of Band.state), checked against its SHA-256, its host and its data."""
    require_gate(approval, prepared, stage)
    state = LR.load_tree(path, band_template(host_state, cfg, rep), sha256)
    if state_sha256(state.native) != state_sha256(host_state):
        raise AssertionError("the saved band belongs to another host")
    digest = data_sha256(prepared)
    if record.get("data_sha256") != digest:
        raise AssertionError("the saved band was fitted on other data")
    return Band(prepared.world, int(rep), prepared.removal, approval, state,
                np.asarray(record["refresh_key"], np.uint32), record, digest)


def _scores_from_state(args, state, heldout, key):
    """The bank scores exactly as td3_bc_bca.refresh and reference.freeze_reference computed them for `state`'s
    frozen reference: target with fold_in(key, 1213156420), q = min over heads, frozen scale network and unit, scale
    max(eta, 1e-6) x unit in float32 then float64, residuals float32 then float64. Also the posterior's stream (the
    uint32 words of fold_in(key, 1347375956))."""
    mods = models()
    target = P.native_target(args, mods, state.native, heldout, jax.random.fold_in(key, KEY_TARGET), 1.0)
    q = mods[1].apply(state.native.critic.params, heldout.obs, heldout.action).min(axis=-1)
    cal = mods[2].apply(state.posterior.cal_params, heldout.obs, heldout.action)
    scale = np.asarray(REF.positive_scale(cal, state.posterior.residual_scale), np.float64)
    residuals = np.asarray(target - q, np.float64)
    stream = np.asarray(jax.random.fold_in(key, KEY_POSTERIOR), np.uint32)
    return np.abs(residuals) / scale, stream


def reproduces(reference, calibration):
    """Whether a calibration stored in float32 equals a frozen reference's threshold, lambda_hat, lambda_hpd and
    n_eff bit for bit."""
    with np.errstate(over="ignore"):
        pairs = [(calibration.threshold, reference.threshold), (calibration.lambda_hat, reference.lambda_hat),
                 (calibration.lambda_hpd, reference.lambda_hpd), (calibration.n_eff, reference.n_eff)]
        return all(np.float32(a).tobytes() == np.asarray(b, np.float32).tobytes() for a, b in pairs)


def bank_scores(band, bank, cfg):
    """The band's bank scores and posterior stream (_scores_from_state); refuses unless the unweighted call on them
    reproduces BCA's own stored threshold bit for bit."""
    require_gate(band.approval, band, "band")
    args = LR.make_args(cfg, band.rep)
    scores, stream = _scores_from_state(args, band.state, bank, jnp.asarray(band.refresh_key, jnp.uint32))
    if not reproduces(band.state.posterior, CV.calibrate(scores, stream)):
        raise AssertionError("the recomputed bank scores do not reproduce BCA's own threshold")
    return scores, stream


def thresholds(band, scores, stream, arms):
    """One wbcp.calibrate per arm on the same scores and stream (coverage.arm_thresholds): arms {name: None
    (unweighted) or (bank weights, test mass)}. Returns {name: wbcp.Calibration}."""
    require_gate(band.approval, band, "band")
    return CV.arm_thresholds(scores, stream, arms)


def threshold_tables(band, scores, stream, arms):
    """The per-step arms' tables of R on the same scores and stream (coverage.threshold_table): arms {name: (bank
    weights, grid of test masses)}. Returns {name: coverage.Table}."""
    require_gate(band.approval, band, "band")
    return {name: CV.threshold_table(scores, stream, w, grid) for name, (w, grid) in arms.items()}


def table_float32(table):
    """A table's thresholds as BCA stores a threshold (float32); a finite threshold that overflows float32 is refused
    (freeze_reference's guard), so +inf in float32 always means 'cannot certify'."""
    R = np.asarray(table.thresholds, np.float64)
    with np.errstate(over="ignore"):
        stored = R.astype(np.float32)
    if np.any(np.isfinite(R) & ~np.isfinite(stored)):
        raise ValueError("a finite threshold must stay finite when stored in float32")
    return stored


def threshold_record(calibration, weights=None, test_mass=None):
    """The JSON record of one arm's threshold (inf written as the string 'inf')."""
    num = lambda v: float(v) if math.isfinite(float(v)) else ("inf" if float(v) > 0 else "-inf")
    return dict(threshold=num(calibration.threshold), lambda_hat=num(calibration.lambda_hat),
                lambda_hpd=num(calibration.lambda_hpd), n_eff=float(calibration.n_eff),
                certified=bool(calibration.certified), sigma_post=num(calibration.sigma_post),
                test_mass=None if test_mass is None else float(test_mass),
                kish=None if weights is None else CV.kish(weights))


def with_threshold(reference, threshold, lambda_hat, lambda_hpd, n_eff):
    """A frozen reference with its four threshold fields replaced (float32, as freeze_reference stores them). A
    finite threshold that overflows float32 is refused (freeze_reference's guard)."""
    threshold, lambda_hat, lambda_hpd = (float(v) for v in (threshold, lambda_hat, lambda_hpd))  # 'inf' -> inf
    with np.errstate(over="ignore"):
        stored = np.float32(threshold)
    if math.isfinite(threshold) and not np.isfinite(stored):
        raise ValueError("a finite threshold must stay finite when stored in float32")
    as32 = lambda v: jnp.asarray(v, jnp.float32)
    return reference._replace(threshold=as32(threshold), lambda_hat=as32(lambda_hat), lambda_hpd=as32(lambda_hpd),
                              n_eff=as32(float(n_eff)))


def reference_of(band, record):
    """The band's reference with an arm's threshold (a threshold_record or a wbcp.Calibration)."""
    if isinstance(record, dict):
        return with_threshold(band.state.posterior, record["threshold"], record["lambda_hat"], record["lambda_hpd"],
                              record["n_eff"])
    return with_threshold(band.state.posterior, record.threshold, record.lambda_hat, record.lambda_hpd,
                          record.n_eff)


def readout(band, reference, trans, stage="band", chunk=8192):
    """td3_bc_bca.bc_readout (blend 0.5) of the band with a given reference at every row of `trans`: width U
    (float32), dose (float64), support (bool) and whether the inputs were valid."""
    require_gate(band.approval, band, stage)
    blend = LR.BCA_SETTINGS["blend"]
    mk = ("readout", int(chunk))
    if mk not in _PROGRAMS:
        mods = models()
        _PROGRAMS[mk] = jax.jit(lambda st, b: P.bc_readout(mods, st, b, blend))
    st = band.state._replace(posterior=reference)
    n = len(trans.obs)
    W_, D_, S_, valid = [], [], [], True
    for i0, i1 in _chunks(n, chunk):
        part = jax.tree_util.tree_map(lambda x: x[i0:i1], trans)
        out = _PROGRAMS[mk](st, _pad(part, chunk))
        W_.append(np.asarray(out.width, np.float32)[:i1 - i0])
        D_.append(np.asarray(out.dose, np.float64)[:i1 - i0])
        S_.append(np.asarray(out.support_mask, bool)[:i1 - i0])
        valid &= bool(out.inputs_valid)
    cat = lambda v, dt: np.concatenate(v) if v else np.zeros(0, dt)
    return dict(width=cat(W_, np.float32), dose=cat(D_, np.float64), support=cat(S_, bool), valid=valid)


def per_row_dose(models, state, batch, R_rows, blend):
    """BCA's frozen-width dose with each row's own threshold (spec version 4, the per-step arms): width_i = R_i x
    max(eta_i, 1e-6) x unit, with frozen_level_dose's validity checks taken row by row, then dose_i = 1 + blend U_i /
    (U_i + unit) by calibration.dose.level_critic_dose (BCA's own arithmetic) for a finite R_i. A row whose R_i is
    +inf (cannot certify) gets the largest dose, 1 + blend: refusal means the most caution (BCA's own readout gives
    such a row the native multiplier 1; this is the declared change). Before the first refresh (reference not ready)
    every width is 0 and every dose 1, as in BCA. With every R_i equal to the reference's threshold this is
    td3_bc_bca.bc_readout bit for bit (finite R). Returns (calibration.dose.CQLLevelDose, refused rows)."""
    ref = state.posterior
    predictions = models[2].apply(ref.cal_params, batch.obs, batch.action)
    R = jnp.asarray(R_rows, jnp.float32)
    if predictions.ndim != 1 or R.shape != predictions.shape:
        raise ValueError("per-row doses need one threshold per row")
    unit = ref.residual_scale
    scale = jnp.maximum(predictions, 1e-06) * unit
    width = R * scale
    valid = (tree_finite(ref.cal_params)
             & jnp.all(jnp.isfinite(predictions) & (predictions > 0))
             & jnp.isfinite(unit) & (unit > 0)
             & jnp.all(jnp.isfinite(scale) & (scale > 0))
             & jnp.all(~jnp.isnan(R) & (R >= 0))
             & jnp.all(jnp.isposinf(R) | (jnp.isfinite(width) & ((R == 0) | (width > 0)))))
    width = jnp.where(ref.ready, width, jnp.zeros_like(width))
    valid = jnp.where(ref.ready, valid, True)
    unit = jnp.where(ref.ready, unit, 1.0)
    base = level_critic_dose(width, jnp.ones_like(width, dtype=bool), unit, blend, valid)
    refused = ref.ready & jnp.isposinf(R)
    dose = jnp.where(refused, jnp.asarray(1.0 + blend, base.dose.dtype), base.dose)
    return base._replace(dose=dose, inputs_valid=base.inputs_valid & REF.reference_valid(ref)), refused


def per_step_readout(band, R_rows, trans, stage="band", chunk=8192):
    """per_row_dose (blend 0.5) of the band's frozen reference with each row's own threshold R_rows at every row of
    `trans`: width U (float32), dose (float64; 1.5 where R = +inf), support (finite width), refused (R = +inf) and
    whether the inputs were valid."""
    require_gate(band.approval, band, stage)
    R_rows = np.asarray(R_rows, np.float64)
    n = len(trans.obs)
    if R_rows.shape != (n,):
        raise ValueError("one threshold per row")
    blend = LR.BCA_SETTINGS["blend"]
    mk = ("per_row_readout", int(chunk))
    if mk not in _PROGRAMS:
        mods = models()
        _PROGRAMS[mk] = jax.jit(lambda st, b, r: per_row_dose(mods, st, b, r, blend))
    with np.errstate(over="ignore"):
        R32 = R_rows.astype(np.float32)
    if np.any(np.isfinite(R_rows) & ~np.isfinite(R32)):
        raise ValueError("a finite threshold must stay finite when stored in float32")
    W_, D_, S_, F_, valid = [], [], [], [], True
    for i0, i1 in _chunks(n, chunk):
        part = jax.tree_util.tree_map(lambda x: x[i0:i1], trans)
        r = _pad(jnp.asarray(R32[i0:i1]), chunk)
        out, refused = _PROGRAMS[mk](band.state, _pad(part, chunk), r)
        W_.append(np.asarray(out.width, np.float32)[:i1 - i0])
        D_.append(np.asarray(out.dose, np.float64)[:i1 - i0])
        S_.append(np.asarray(out.support_mask, bool)[:i1 - i0])
        F_.append(np.asarray(refused, bool)[:i1 - i0])
        valid &= bool(out.inputs_valid)
    cat = lambda v, dt: np.concatenate(v) if v else np.zeros(0, dt)
    return dict(width=cat(W_, np.float32), dose=cat(D_, np.float64), support=cat(S_, bool), refused=cat(F_, bool),
                valid=valid)


def dose_change(dose, base, high=None):
    """R2's dose statistics of one arm against the unweighted doses on the same rows: the mean change, the mean
    absolute change, the largest absolute change and (given a mask of high-current rows) the mean change there."""
    d = np.asarray(dose, np.float64) - np.asarray(base, np.float64)
    out = dict(mean=float(np.mean(dose)), change_vs_unweighted=float(d.mean()), mean_abs_change=float(np.abs(d).mean()),
               max_abs_change=float(np.abs(d).max()))
    if high is not None:
        high = np.asarray(high, bool)
        out.update(high_current_rows=int(high.sum()),
                   high_current_change=float(d[high].mean()) if high.any() else None)
    return out


def scale_at(band, trans, chunk=8192):
    """The frozen scale network's prediction eta(s, a) at rows (float32), for checking U = R x unit x eta."""
    mk = ("scale", int(chunk))
    if mk not in _PROGRAMS:
        mods = models()
        _PROGRAMS[mk] = jax.jit(lambda p, b: mods[2].apply(p, b.obs, b.action))
    out = []
    for i0, i1 in _chunks(len(trans.obs), chunk):
        part = jax.tree_util.tree_map(lambda x: x[i0:i1], trans)
        out.append(np.asarray(_PROGRAMS[mk](band.state.posterior.cal_params, _pad(part, chunk)), np.float32)[:i1 - i0])
    return np.concatenate(out) if out else np.zeros(0, np.float32)


# ------------------------------------------------------------------------------------------------ room check (no BCA)

ROOM_ARMS = (LR.Arm("none", "even", 1.0), LR.Arm("placed", "harm"), LR.Arm("even_x1.1", "even", 1.1),
             LR.Arm("even_x1.25", "even", 1.25), LR.Arm("even_x1.5", "even", 1.5))
ROOM_EVEN_DOSES = {"none": 1.0, "even_x1.1": 1.1, "even_x1.25": 1.25, "even_x1.5": 1.5}   # R2's slope over these
MATCHED = "matched_even"


def room_check(prepared, critic_params, cfg, rep, *, log=None):
    """v1's room check (acceptance.h3_seed's training part) with this spec's arms, against a frozen critic: clone an
    actor on the kept rows (20,000 steps), run the arms side by side (learners.ActorOnly, 20,000 steps, one shared
    stream of rows; 'placed' uses BCA's dose form on the exact over-preference, refreshed every 4,000 steps), then
    match even caution in strength to 'placed' (learners.match_even on the ladder x1.1, x1.25, x1.5; never
    extrapolated, so a placed arm stronger than x1.5 has no matched arm). Uses strengths only: no return is computed
    here. Returns ({arm: actor params}, record)."""
    clone_params, clone_record = LR.clone(prepared, cfg, rep, log=log)
    runner = LR.ActorOnly(prepared, clone_params, cfg, rep, alpha=cfg.alpha)
    stacked, info = runner.run(list(ROOM_ARMS), [critic_params], log=log)
    names = [a.name for a in ROOM_ARMS]
    params = {a.name: runner.arm_params(stacked, i) for i, a in enumerate(ROOM_ARMS)}
    failed = dict(zip(names, (bool(f) for f in info["failed"])))
    S = {k: LR.strength(p, params["none"], prepared.eval_obs) for k, p in params.items()}
    ladder = [(a.value, S[a.name]) for a in ROOM_ARMS if a.kind == "even" and a.value > 1]
    target = S["placed"]
    if failed["placed"] or failed["none"] or not (math.isfinite(target) and target > 0):
        match = dict(multiplier=None, knob=None, ladder_multiplier=None, skipped=True,
                     reason="the placed or no-caution arm failed or has no finite positive strength")
    else:
        match, evaluated = LR.match_even(runner, [critic_params], params["none"], prepared.eval_obs, ladder, target,
                                         cfg, log=log)
        if match["multiplier"] is not None:
            if match["knob"] == 0:
                params[MATCHED] = params["none"]
            elif match["ladder_multiplier"] is not None:
                params[MATCHED] = params[f"even_x{match['ladder_multiplier']:g}"]
            else:
                params[MATCHED] = evaluated[match["multiplier"]]
            S[MATCHED] = LR.strength(params[MATCHED], params["none"], prepared.eval_obs)
    record = dict(kind="shift-iw-room", rep=int(rep), arms=[asdict(a) for a in ROOM_ARMS], names=sorted(params),
                  clone=clone_record, strengths=S, match=match, dose_history=info["history"], failed=failed,
                  rows_sha256=info["rows_sha256"], start_sha256=info["start_sha256"],
                  oracle_calls=info["oracle_calls"], chunk_metrics=info["chunk_metrics"],
                  metric_names=info["metric_names"])
    return params, record


def room_template(names):
    return {k: LR.actor_template() for k in names}


# ------------------------------------------------------------------------------------------------ BCA arms

def bca_update_per_row(args, config, models, state, batch, R_rows, it, rng, max_action=1.0):
    """learners.bca_update with placement 'own' (td3_bc_bca.update) line for line, except the dose: per_row_dose with
    each batch row's own threshold R_rows instead of td3_bc_bca.bc_readout's one threshold."""
    fitted, diag = P.fit_scale(args, config, models, state, batch, rng, max_action)
    dose, refused = per_row_dose(models, state, batch, R_rows, config.blend)
    multiplier = P.consumed_bc_multiplier(config, dose.dose, batch, it, rng)
    native, metrics = BASE_TD3.td3_bc_update(args, models[0].apply, models[1].apply, state.native, batch, it, rng,
                                             max_action, bc_multiplier=multiplier)
    valid = P.tree_finite((batch, state.native, native, metrics))
    valid = valid & diag["scale_inputs_valid"] & dose.inputs_valid
    metrics.update(bc_multiplier_mean=jnp.mean(multiplier),
                   bc_support_fraction=jnp.mean(dose.support_mask.astype(jnp.float32)),
                   bc_refused_fraction=jnp.mean(refused.astype(jnp.float32)),
                   posterior_ready=state.posterior.ready)
    result = jax.lax.cond(valid, lambda _: fitted._replace(native=native), lambda _: state, None)
    metrics.update(diag, inputs_valid=valid)
    return (result, metrics)


def bca_train_step_per_row(args, config, models, dataset, row_index, max_action=1.0):
    """learners.bca_train_step line for line (the same key splits and batch draws), calling bca_update_per_row. The
    carry gains the table of thresholds on the grid (float32, replaced at each refresh); row_index: each training
    row's grid index (coverage.lookup), so a batch row's threshold is table[row_index[row]]."""
    if not len(dataset.obs):
        raise ValueError("training pool must be nonempty")
    row_index = jnp.asarray(row_index, jnp.int32)
    if row_index.shape != (len(dataset.obs),):
        raise ValueError("one grid index per training row")

    def step(carry, _):
        rng, state, it, table = carry
        kr, kb, kn = jax.random.split(rng, 3)
        idx = jax.random.randint(kb, (args.batch_size,), 0, len(dataset.obs))
        batch = jax.tree_util.tree_map(lambda x: x[idx], dataset)
        result, metrics = bca_update_per_row(args, config, models, state, batch, table[row_index[idx]], it + 1, kn,
                                             max_action)
        proposed = (kr, result, it + 1, table)
        accepted = jax.lax.cond(metrics["inputs_valid"], lambda _: proposed, lambda _: carry, None)
        return (accepted, metrics)

    return step


def _weighting_mode(weighting, prepared):
    """Check a weighting and return its mode ('single' or 'per_row') with the weights; for 'per_row' also each kept
    training row's grid index and the largest relative mass gap of the lookup."""
    w = np.asarray(weighting["weights"], np.float64)
    if w.shape != (len(prepared.bank.obs),):
        raise ValueError("the weighting needs one weight per kept bank row")
    mode = weighting.get("mode", "single")
    if mode == "single":
        return mode, w, None, None
    if mode != "per_row":
        raise ValueError("a weighting's mode is 'single' or 'per_row'")
    grid = np.asarray(weighting["grid"], np.float64)
    masses = np.asarray(weighting["row_masses"], np.float64)
    if masses.shape != (prepared.n,):
        raise ValueError("a per-row weighting needs one test mass per kept training row")
    probe = CV.Table(grid, np.zeros(len(grid)), np.zeros(len(grid)), 0.0, 0.0, True, False)
    _, idx = CV.lookup(probe, masses)
    return mode, w, idx, float(np.max(grid[idx] / masses - 1.0))


def bca_cotrain(prepared, cfg, rep, approval, *, weighting=None, updates=None, out_dir=None, log=None, name=None,
                stage="bca"):
    """A co-trained TD3+BC+BCA arm: learners.bca_cotrain restated under this package's gate (placement 'own', so the
    update is td3_bc_bca.update line for line), with an optional weighting. weighting: None (unweighted: BCA as it
    is), dict(name, weights (one per kept bank row), test_mass) for the single-threshold arm (mode 'single'), or
    dict(name, mode='per_row', weights, grid, row_masses (one test mass per kept training row)) for the per-row arm.
    After each unchanged refresh that is valid, the bank scores are recomputed with the refresh's key (checked to
    reproduce BCA's own threshold). Single: wbcp.calibrate is called with the weights, the test mass and the same
    stream, and R is replaced; nothing else changes. Per row: the table of R on the grid is recomputed with the same
    stream (coverage.threshold_table) and carried into bca_train_step_per_row, so each training row's dose uses its own
    threshold, and a row whose threshold is +inf gets the dose 1 + blend; the count of such rows is recorded at every
    refresh. All arms of a seed share their initial state and their stream. stage: the approval scope the arm needs
    ('bca', or 'bca_estimated' for the optional estimated-weighted arm, which also needs 'bca'). Returns (record,
    final state, {update: state})."""
    require_gate(approval, prepared, "bca")
    if stage != "bca":
        require_gate(approval, prepared, stage)
    mode, w, row_index, mass_gap = (None, None, None, None) if weighting is None else \
        _weighting_mode(weighting, prepared)
    args = LR.make_args(cfg, rep)
    config = LR.bca_config()
    rng, state, mods = LR.bca_initial_state(args, config)
    total = int(cfg.updates if updates is None else updates)
    if mode == "per_row":
        grid = np.asarray(weighting["grid"], np.float64)
        step = bca_train_step_per_row(args, config, mods, prepared.train, row_index, 1.0)
        carry = (rng, state, jnp.int32(0), jnp.ones(len(grid), jnp.float32))
    else:
        step = LR.bca_train_step(args, config, mods, prepared.train, 1.0, "own")
        carry = (rng, state, jnp.int32(0))
    refreshes = LR.refresh_steps(cfg, total)
    snaps = LR.snapshot_steps(cfg, total)
    refresh_seed = int(LR.rng_np(cfg, rep, LR.TAG_REFRESH).integers(2 ** 31 - 1))
    done, chunks, states, refresh_log, programs = 0, [], {}, [], {}
    label = name or ("BCA unweighted" if weighting is None else f"BCA weighted ({weighting['name']}, {mode})")
    if weighting is None:
        wrec = None
    elif mode == "single":
        wrec = dict(name=weighting["name"], mode=mode, test_mass=float(weighting["test_mass"]),
                    weights_sha256=LR.tree_sha256(w), kish=CV.kish(w))
    else:
        wrec = dict(name=weighting["name"], mode=mode, weights_sha256=LR.tree_sha256(w), kish=CV.kish(w),
                    grid_points=int(len(grid)), grid_range=[float(grid[0]), float(grid[-1])],
                    grid_sha256=LR.tree_sha256(grid), row_masses_sha256=LR.tree_sha256(weighting["row_masses"]),
                    max_mass_gap=mass_gap, refused_dose=1.0 + float(config.blend),
                    note="each training row's dose uses its own threshold R(w_i); a row whose R is +inf gets the dose "
                         "1 + blend (BCA itself would give it 1): the declared change of spec version 4")
    record = dict(kind="shift-iw-bca", name=label, rep=int(rep), args=asdict(args), updates=total,
                  bca=LR.BCA_SETTINGS, data=prepared.describe(), weighting=wrec,
                  initial_sha256=LR.tree_sha256((state.native.actor.params, state.native.critic.params)),
                  batch_stream_sha256=LR.batch_stream_sha256(rng, prepared.n, args.batch_size),
                  refresh_seed=refresh_seed, snapshots=[], failed=False)
    for target in LR._events(total, cfg.chunk, snaps, refreshes):
        length = target - done
        if length not in programs:
            programs[length] = jax.jit(lambda c, length=length: jax.lax.scan(step, c, None, length=length))
        carry, metrics = programs[length](carry)
        done = target
        try:
            P.require_valid(metrics)
        except FloatingPointError as error:
            record.update(failed=True, failed_at=target, failure=str(error))
            break
        summary = LR._chunk_summary(metrics)
        summary["batches_with_partial_support"] = int(np.sum(np.asarray(metrics["bc_support_fraction"]) < 1))
        if "bc_refused_fraction" in metrics:
            summary["batches_with_refused_rows"] = int(np.sum(np.asarray(metrics["bc_refused_fraction"]) > 0))
        chunks.append(dict(summary, update=target))
        if target in refreshes:
            key = jax.random.fold_in(jax.random.PRNGKey(refresh_seed), target)
            new, diag, diagnostics = P.refresh(args, config, mods, carry[1], prepared.bank, key, 1.0,
                                               heldout_ids=np.arange(len(prepared.bank.obs)))
            R_u = float(diagnostics.get("threshold", math.inf))
            entry = dict(step=target, **{k: bool(v) for k, v in diag.items()}, unweighted_threshold=R_u)
            table = carry[3] if mode == "per_row" else None
            if weighting is not None and diag["posterior_inputs_valid"]:
                scores, stream = _scores_from_state(args, new, prepared.bank, key)
                if not reproduces(new.posterior, CV.calibrate(scores, stream)):
                    raise AssertionError("the recomputed bank scores do not reproduce BCA's own threshold")
                if mode == "single":
                    cal = CV.calibrate(scores, stream, w, float(weighting["test_mass"]))
                    new = new._replace(posterior=with_threshold(new.posterior, cal.threshold, cal.lambda_hat,
                                                                cal.lambda_hpd, cal.n_eff))
                    entry.update(weighted=threshold_record(cal, w, weighting["test_mass"]),
                                 weighted_certified=bool(cal.certified))
                else:
                    tab = CV.threshold_table(scores, stream, w, grid)
                    table = jnp.asarray(table_float32(tab))
                    R_rows = tab.thresholds[row_index]
                    refused_rows = int(np.sum(np.isinf(R_rows)))
                    entry.update(per_row=dict(table=tab.summary(R_u), rows=CV.per_step_summary(R_rows, R_u),
                                              refused_rows=refused_rows),
                                 weighted_certified=refused_rows == 0)
            carry = (carry[0], new, carry[2]) if mode != "per_row" else (carry[0], new, carry[2], table)
            refresh_log.append(entry)
        if target in snaps:
            states[target] = carry[1]
            snap = dict(update=target, actor_sha256=LR.tree_sha256(carry[1].native.actor.params))
            if out_dir is not None:
                snap["file_sha256"] = LR.save_tree(
                    Path(out_dir) / f"snapshot_{int(round(100 * target / total)):03d}.msgpack", carry[1])
            record["snapshots"].append(snap)
        if log:
            log(f"{label} rep {rep}: update {target}/{total}")
    per_row = [e["per_row"] for e in refresh_log if "per_row" in e]
    record.update(chunks=chunks, refreshes=refresh_log, final_update=done,
                  uncertified_refreshes=sum(1 for e in refresh_log if not e["posterior_certified"]),
                  uncertified_weighted_refreshes=sum(1 for e in refresh_log
                                                     if "weighted_certified" in e and not e["weighted_certified"]),
                  batches_with_partial_support=sum(c["batches_with_partial_support"] for c in chunks))
    if mode == "per_row":
        record.update(refused_rows_per_refresh=[p["refused_rows"] for p in per_row],
                      refused_rows_max=max((p["refused_rows"] for p in per_row), default=0),
                      tables_monotone=all(p["table"]["monotone"] for p in per_row),
                      running_max_applied=any(p["table"]["running_max_applied"] for p in per_row),
                      batches_with_refused_rows=sum(c.get("batches_with_refused_rows", 0) for c in chunks))
    return record, carry[1], states


def bca_template(cfg, rep):
    return LR.bca_initial_state(LR.make_args(cfg, rep), LR.bca_config())[1]


# ------------------------------------------------------------------------------------------------ evaluation

def evaluate_actor(actor_params, prepared, banks, anchors):
    """Exact 100-step returns (truth.rollout, float64) of a learned actor from every fixed start of each law's start
    bank, and points = 100 (J - J_random) / (J_stall_aware - J_random) with the bank's anchors (spec version 5:
    world.points, anchors from data.load_start_banks). Marks read J; points are shown beside it. Returns
    {law: dict(J, J_se, points, points_se, finite)}."""
    pol = LR.mlp_policy(actor_params, prepared)
    out = {}
    for law, x0 in banks.items():
        J = TR.rollout(pol, x0, W.PHYSICS).J
        pts = W.points(J, anchors[law]["anchor"])
        n = len(J)
        out[law] = dict(J=float(J.mean()), J_se=float(J.std(ddof=1) / math.sqrt(n)) if n > 1 else None,
                        points=float(pts.mean()), points_se=float(pts.std(ddof=1) / math.sqrt(n)) if n > 1 else None,
                        finite=bool(np.all(np.isfinite(J))), starts=int(n))
    return out
