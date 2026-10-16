"""Test (h), the scoring adapters (plan sections 3.9 and 4.5, stop rule S7), and checks of the scoring modules, on
made-up actor parameters and made-up states only: no learned policy exists or is scored here, and no BCA code runs.
An experiment-local attachment (plan section 3.1) that changes none of the repository's declared methods.

Test (h), as section 3.9 lists it:
  h1  the TD3+BC adapter equals truth.MLPPolicy (given obs_std + 1e-3) within 1e-12 on 10,000 made-up states
  h2  each float64 adapter matches the float32 restatement (actors_f32) within 1e-5 on made-up parameters, CQL with
      made-up noise and its tanh-of-mean readout too
  h3  S7, open loop, by file: the training environment (/opt/bca-venv, a subprocess of this module with --role
      unifloral) computes each Unifloral actor's own float32 commands (the imported Unifloral classes) on states saved
      here; the restatement must match within 1e-7 on the processor, CQL with the same noise. Negative controls (a
      wrong normalization, CQL's two heads swapped) must fail, so the check has teeth
  h4  S7, closed loop: the float32 restatement and the float64 adapter agree within 0.01 points on a few starts of every
      bank, CQL with the declared noise
  h5  CQL's noise stream is reproducible: the same bits twice, the declared rule, different banks differ, the training
      environment's numpy draws the same bits, and the cached files (scoring_cache.py) hold the same bits
Also: the actor file's layout and refusals; the scoring permission (made-up files only, or a verified approval);
evaluate's metrics against learners.evaluate_policy (bit for bit on the four situations) and on the fixed controllers
of the scoring cache (the stall-aware controller is 100 points on every bank and fails F2 on 1% of starts, the zero
command never fails F1, nothing beats the exact best plan); the viewer's episodes come from the scored rollout; every
learner's made-up actor goes through evaluate.score_actor and write_scores.

Run scoring_cache.py first (the cache tests and the default open-loop states need it). Then, from /root/bca-hosts:
  env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu SHIFT_HOSTS_DEVICE=cpu taskset -c 20-23 \\
      /opt/bca-shift-venv/bin/python -m experiments.shift_hosts.tests.test_scoring
Results: runs/shift_hosts/stage0/scoring/test_scoring_cpu.json (with sidecar) and the files under
runs/shift_hosts/stage0/scoring/h/.
"""

import argparse
import math
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from experiments.shift_hosts import actors_f32 as A32
from experiments.shift_hosts import handoff as H

AREA = "scoring"
REPO = Path(__file__).resolve().parents[3]
TRAINING_PYTHON = "/opt/bca-venv/bin/python"
SCORING_PYTHON = "/opt/bca-shift-venv/bin/python"
LEARNERS = A32.LEARNERS
MADE_UP_SEED = (20261016, 7779)        # the made-up tag of tests/madeup.py; words 1000+ are this module's
CLOSED_LOOP_STARTS = np.arange(0, 1000, 125)     # 8 starts per bank


def made_up_actor(learner, seed=0):
    """Deterministic made-up actor parameters in Unifloral's tree layout (flax default-shaped, float32): (params,
    norm). He-scaled hidden layers; output layers scaled so the commands spread over [-1, 1], past 80% on some steps."""
    rng = np.random.default_rng(list(MADE_UP_SEED) + [1000 + 10 * LEARNERS.index(learner) + int(seed)])

    def dense(n_in, n_out, scale, bias):
        return dict(kernel=(scale / math.sqrt(n_in) * rng.standard_normal((n_in, n_out))).astype(np.float32),
                    bias=(bias * rng.standard_normal(n_out)).astype(np.float32))

    hidden = A32.HIDDEN_LAYERS[learner]
    p, n_in = {}, A32.OBS_DIM
    for i in range(hidden):
        p[f"Dense_{i}"] = dense(n_in, A32.HIDDEN, math.sqrt(2.0), 0.1)
        n_in = A32.HIDDEN
    if learner == "cql":
        p[f"Dense_{hidden}"] = dense(A32.HIDDEN, A32.NUM_ACTIONS, 0.6, 0.5)        # log-spread head
        p[f"Dense_{hidden + 1}"] = dense(A32.HIDDEN, A32.NUM_ACTIONS, 0.5, 0.2)    # mean head
    else:
        p[f"Dense_{hidden}"] = dense(A32.HIDDEN, A32.NUM_ACTIONS, 0.5, 0.2)
    if learner == "iql":
        p["logstd"] = (0.1 * rng.standard_normal(A32.NUM_ACTIONS)).astype(np.float32)
    norm = None
    if learner in A32.NORMALIZED:
        norm = dict(obs_mean=(0.3 * rng.standard_normal(A32.OBS_DIM)).astype(np.float32),
                    obs_std=(0.2 + rng.uniform(size=A32.OBS_DIM)).astype(np.float32))
    return {"params": p}, norm


def made_up_states(n, word, extreme=False):
    """float64 [n, 13] made-up states in ThrusterChain's ranges (not its data): floats N(0, k^2) with k per row drawn
    from {0.5, 1.5, 3} (the recorded and big-kicks start spreads, and an excursion beyond both; the stall-aware
    controller's visited states on the strong and big-kicks banks reach a float radius of 8.4), current U[0, 1.5],
    wave amplitude U[0, 1] at a random phase. extreme=True multiplies a tenth of the rows' floats by 10, far outside
    anything the simulator reaches (used where both sides compute in the same precision, and reported elsewhere)."""
    rng = np.random.default_rng(list(MADE_UP_SEED) + [int(word)])
    k = rng.choice(np.array([0.5, 1.5, 3.0]), n)
    s = k[:, None] * rng.standard_normal((n, 10))
    if extreme:
        s[: n // 10] *= 10.0
    c = rng.uniform(0.0, 1.5, n)
    amp, ph = rng.uniform(0.0, 1.0, n), rng.uniform(0.0, 2 * math.pi, n)
    return np.concatenate([s, c[:, None], (amp * np.cos(ph))[:, None], (amp * np.sin(ph))[:, None]], 1)


def made_up_eps(n, word):
    return np.random.default_rng(list(MADE_UP_SEED) + [int(word)]).standard_normal((n, A32.NUM_ACTIONS)).astype(
        np.float32)


def write_made_up_actor(folder, learner, seed=0, made_up=True):
    params, norm = made_up_actor(learner, seed)
    meta = dict(made_up=made_up, note="made-up parameters for test (h), not a learned policy")
    if learner == "rebrac":
        meta.update(actor_ln=False, norm_obs=False)
    kw = {} if norm is None else dict(obs_mean=norm["obs_mean"], obs_std=norm["obs_std"])
    path = Path(folder) / f"{learner}_actor{'' if made_up else '_not_made_up'}.npz"
    A32.write_actor_file(path, learner, params, meta=meta, **kw)
    return path


def h_dir():
    from experiments.shift_hosts import tests as T

    return T.out_dir(AREA) / "h"


def setUpModule():
    A32.configure()


# ------------------------------------------------------------------------------------------------ the actor file

class ActorFileTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dir = h_dir() / "files"
        shutil.rmtree(cls.dir, ignore_errors=True)
        cls.dir.mkdir(parents=True)

    def test_round_trip_every_learner(self):
        """Each learner's made-up actor goes through write_actor_file and read_actor_file bit for bit, in the layout
        of actors_f32's docstring (Unifloral's parameter names; norm only for TD3+BC and IQL)."""
        for learner in LEARNERS:
            params, norm = made_up_actor(learner)
            path = write_made_up_actor(self.dir, learner)
            a = A32.read_actor_file(path, learner)
            self.assertEqual(a.learner, learner)
            flat = H.flatten_dict(params, "actor")
            got = H.flatten_dict(a.params, "actor")
            self.assertEqual(sorted(flat), sorted(got))
            for k in flat:
                self.assertEqual(flat[k].tobytes(), got[k].tobytes(), k)
            self.assertEqual(sorted(list(flat) + (["norm/obs_mean", "norm/obs_std"] if norm else [])),
                             sorted(A32.layout(learner)))
            self.assertEqual(a.norm is None, learner not in A32.NORMALIZED)
            self.assertTrue(a.meta["made_up"])
            self.assertEqual(a.meta["schema"], A32.SCHEMA)

    def test_layout_shapes(self):
        """The parameter counts of plan section 9.2: TD3+BC and IQL 70,404 (mean layers), CQL 137,224, ReBRAC
        136,196."""
        def count(learner):
            return sum(int(np.prod(s)) for k, s in A32.layout(learner).items()
                       if k.startswith("actor/") and not k.endswith("logstd"))
        self.assertEqual(count("td3_bc"), 70404)
        self.assertEqual(count("iql"), 70404)
        self.assertEqual(count("cql"), 137224)
        self.assertEqual(count("rebrac"), 136196)

    def test_refusals(self):
        params, norm = made_up_actor("td3_bc")
        with self.assertRaises(A32.ActorFileError):                 # no normalization constants
            A32.write_actor_file(self.dir / "x.npz", "td3_bc", params, meta=dict(made_up=True))
        bad = {"params": dict(params["params"])}
        bad["params"]["Dense_2"] = dict(kernel=np.zeros((256, 5), np.float32), bias=np.zeros(5, np.float32))
        with self.assertRaises(A32.ActorFileError):                 # wrong shape
            A32.write_actor_file(self.dir / "x.npz", "td3_bc", bad, meta=dict(made_up=True), **norm)
        f64 = {"params": {k: {kk: vv.astype(np.float64) for kk, vv in v.items()} for k, v in params["params"].items()}}
        with self.assertRaises(A32.ActorFileError):                 # wrong dtype
            A32.write_actor_file(self.dir / "x.npz", "td3_bc", f64, meta=dict(made_up=True), **norm)
        with self.assertRaises(A32.ActorFileError):                 # made_up must be stated
            A32.write_actor_file(self.dir / "x.npz", "td3_bc", params, meta={}, **norm)
        rp, _ = made_up_actor("rebrac")
        with self.assertRaises(A32.ActorFileError):                 # ReBRAC's flags must be recorded and false
            A32.write_actor_file(self.dir / "x.npz", "rebrac", rp, meta=dict(made_up=True))
        with self.assertRaises(A32.ActorFileError):
            A32.write_actor_file(self.dir / "x.npz", "rebrac", rp, meta=dict(made_up=True, actor_ln=True,
                                                                             norm_obs=False))
        path = write_made_up_actor(self.dir, "cql")
        with self.assertRaises(A32.ActorFileError):                 # read as another learner
            A32.read_actor_file(path, "rebrac")

    def test_constants_match_the_package(self):
        """policies' bank order and noise stream equal bca_core's; actors_f32's processor flags and precision equal
        unifloral_import's (so both halves of S7's open loop compile under the same settings)."""
        from experiments.shift_hosts import bca_core as BC
        from experiments.shift_hosts import policies as P
        from experiments.shift_hosts import unifloral_import as UI

        self.assertEqual(P.BANKS, BC.BANKS)
        self.assertEqual(P.CQL_NOISE_STREAM, BC.CQL_NOISE_STREAM)
        self.assertEqual(P.MASTER_SEED, BC.MASTER_SEED)
        self.assertEqual(A32.CPU_XLA_FLAGS, UI.chosen_flags("cpu"))
        self.assertEqual(A32.MATMUL_PRECISION, UI.MATMUL_PRECISION)
        s = A32.settings()
        self.assertEqual(s["xla_flags"], A32.CPU_XLA_FLAGS)
        self.assertEqual(s["matmul_precision"], "highest")


# ------------------------------------------------------------------------------------------------ h1, h2, h5

class AdapterTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dir = h_dir() / "adapters"
        shutil.rmtree(cls.dir, ignore_errors=True)
        cls.dir.mkdir(parents=True)
        cls.actors = {l: A32.read_actor_file(write_made_up_actor(cls.dir, l), l) for l in LEARNERS}
        cls.x = made_up_states(10000, 2001)
        cls.x_extreme = made_up_states(10000, 2003, extreme=True)
        cls.eps = made_up_eps(10000, 2002)

    def test_h1_td3_bc_adapter_is_mlppolicy(self):
        """h1: the TD3+BC adapter (from its actor file) equals truth.MLPPolicy given obs_std + 1e-3, within 1e-12 on
        10,000 made-up states (and on 10,000 with extreme rows: both sides are float64)."""
        from experiments.shift import truth as TR
        from experiments.shift_hosts import policies as P

        params, norm = made_up_actor("td3_bc")
        ref = TR.MLPPolicy(params, norm["obs_mean"], np.asarray(norm["obs_std"], np.float64) + 1e-3)
        d = {}
        for name, x in (("states", self.x), ("extreme", self.x_extreme)):
            got = P.adapter(self.actors["td3_bc"])(x)
            d[name] = float(np.abs(got - ref(x)).max())
            self.assertLessEqual(d[name], 1e-12)
            self.assertTrue(np.all(np.isfinite(got)))
        # IQL's adapter is the same class on its mean layers (Dense_0..2)
        iparams, inorm = made_up_actor("iql")
        mean_layers = {"params": {k: iparams["params"][k] for k in ("Dense_0", "Dense_1", "Dense_2")}}
        iref = TR.MLPPolicy(mean_layers, inorm["obs_mean"], np.asarray(inorm["obs_std"], np.float64) + 1e-3)
        d["iql"] = float(np.abs(P.adapter(self.actors["iql"])(self.x) - iref(self.x)).max())
        self.assertLessEqual(d["iql"], 1e-12)
        # for the record: the spread computed in float32 instead (obs_std + float32(1e-3)) moves commands by this much
        ref32 = TR.MLPPolicy(params, norm["obs_mean"], norm["obs_std"] + np.float32(1e-3))
        d["float32_spread_variant"] = float(np.abs(ref32(self.x) - ref(self.x)).max())
        self.__class__.h1 = dict(max_abs=d, states=len(self.x))

    def _h2(self, x):
        from experiments.shift_hosts import policies as P

        obs = x.astype(np.float32)
        out = {}
        for learner, actor in self.actors.items():
            for readout in A32.READOUTS[learner]:
                cql_scored = learner == "cql" and readout == "scored"
                c32 = A32.commands(actor, obs, self.eps if cql_scored else None, readout)
                if cql_scored:
                    c64 = P.CQLPolicy(actor.params, self.eps[:, None, :]).act(x, 0)
                else:
                    c64 = P.adapter(actor, readout=readout)(x)
                out[f"{learner}/{readout}"] = float(np.abs(c64 - c32.astype(np.float64)).max())
                self.assertTrue(np.all(np.abs(c32) <= 1.0))
                if readout == "scored":
                    out[f"{learner}/share_past_80"] = float(np.mean(np.abs(c32) > 0.8))
        return out

    def test_h2_float64_adapters_match_restatement(self):
        """h2: every float64 adapter's commands match the float32 restatement within 1e-5 on 10,000 made-up states in
        ThrusterChain's ranges (observations rounded to float32, as both see them), CQL's scored command with made-up
        noise and its tanh-of-mean readout. The same comparison on states with extreme rows is recorded, not judged
        (float32 rounding grows with the size of the activations there)."""
        out = self._h2(self.x)
        for learner in LEARNERS:
            for readout in A32.READOUTS[learner]:
                self.assertLessEqual(out[f"{learner}/{readout}"], 1e-5, f"{learner} {readout}: {out}")
        # the made-up actors exercise the whole range: some commands past 80% (stall), most not
        self.assertTrue(all(0.01 < out[f"{l}/share_past_80"] < 0.9 for l in LEARNERS), out)
        self.__class__.h2 = dict(judged=out, extreme_rows_not_judged=self._h2(self.x_extreme))

    def test_h5_cql_noise_reproducible(self):
        """h5: CQL's declared noise is the rule's draw, the same bits twice, float32 [1000, 100, 4], different per
        bank; and the scoring cache's files (if built) hold the same bits."""
        from experiments.shift_hosts import evaluate as E
        from experiments.shift_hosts import policies as P

        out = {}
        for b, bank in enumerate(P.BANKS):
            e1, e2 = P.cql_noise(bank), P.cql_noise(b)
            rule = np.random.default_rng([20261016, 1004, b]).standard_normal((1000, 100, 4)).astype(np.float32)
            self.assertEqual(e1.dtype, np.float32)
            self.assertEqual(e1.shape, (1000, 100, 4))
            self.assertEqual(e1.tobytes(), e2.tobytes())
            self.assertEqual(e1.tobytes(), rule.tobytes())
            out[bank] = H.array_sha256(e1)
            path = E.noise_path(bank)
            if path.exists():
                cached, _ = H.read(path)
                self.assertEqual(cached["eps"].tobytes(), e1.tobytes())
                out[f"{bank}/cached"] = True
        self.assertEqual(len({out[b] for b in P.BANKS}), len(P.BANKS))
        self.__class__.h5 = out


# ------------------------------------------------------------------------------------------------ h3: S7 open loop

def _unifloral_role(folder):
    """The training environment's half (python -m ... --role unifloral --dir folder): Unifloral's own commands on the
    saved states of every learner, and CQL's noise drawn with this environment's numpy."""
    from experiments.shift_hosts import unifloral_import as UI

    UI.prepare(device="cpu")
    from experiments.shift_hosts import actors_f32 as A

    folder = Path(folder)
    for learner in LEARNERS:
        A.open_loop_commands(folder / f"{learner}_states.npz", folder / f"{learner}_actor.npz",
                             folder / f"{learner}_commands.npz")
    noise = {str(b): H.array_sha256(np.random.default_rng([20261016, 1004, b]).standard_normal(
        (1000, 100, 4)).astype(np.float32)) for b in range(7)}
    H.write_json(folder / "unifloral_report.json", dict(numpy=np.__version__, executable=sys.executable,
                                                        noise_sha256=noise, record=UI.record()))
    return 0


class OpenLoopTest(unittest.TestCase):
    """h3: S7's open loop, by file, both environments on the processor."""

    @classmethod
    def setUpClass(cls):
        from experiments.shift_hosts import evaluate as E
        from experiments.shift_hosts import guard
        from experiments.shift_hosts import unifloral_import as UI

        cls.dir = h_dir() / "open_loop"
        shutil.rmtree(cls.dir, ignore_errors=True)
        cls.dir.mkdir(parents=True)
        extra = made_up_states(2000, 3001, extreme=True)
        for learner in LEARNERS:
            path = write_made_up_actor(cls.dir, learner)
            E.open_loop_states(path, cls.dir / f"{learner}_states.npz", guard.MADE_UP, extra_states=extra,
                               extra_eps=made_up_eps(2000, 3002) if learner == "cql" else None)
        if not Path(TRAINING_PYTHON).exists():
            raise unittest.SkipTest("no training environment")
        env = UI.launch_env(os.environ)
        env.pop("XLA_FLAGS", None)
        env.update(JAX_PLATFORMS="cpu", SHIFT_HOSTS_DEVICE="cpu", PYTHONDONTWRITEBYTECODE="1")
        cls.proc = subprocess.run([TRAINING_PYTHON, "-m", "experiments.shift_hosts.tests.test_scoring", "--role",
                                   "unifloral", "--dir", str(cls.dir)], cwd=str(REPO), env=env,
                                  capture_output=True, text=True, timeout=1800)

    def test_training_side_ran(self):
        self.assertEqual(self.proc.returncode, 0, self.proc.stdout[-3000:] + self.proc.stderr[-3000:])
        rep = H.read_json(self.dir / "unifloral_report.json")
        self.assertEqual(rep["executable"], TRAINING_PYTHON)
        self.assertEqual(rep["numpy"], "1.26.4")
        self.assertEqual(rep["record"]["device"], "cpu")
        self.assertEqual(rep["record"]["xla_flags"]["in_effect"], A32.CPU_XLA_FLAGS)
        self.assertNotEqual(rep["numpy"], np.__version__)

    def test_h3_restatement_matches_unifloral_within_1e7(self):
        """h3: on the saved states (the viewer's fixed starts of every bank as the float64 adapter visits them, plus
        2,000 made-up states), the restatement's float32 commands equal the learner's own within 1e-7."""
        from experiments.shift_hosts import evaluate as E

        self.assertEqual(self.proc.returncode, 0, self.proc.stderr[-3000:])
        out = {}
        for learner in LEARNERS:
            res = E.open_loop_compare(self.dir / f"{learner}_states.npz", self.dir / f"{learner}_commands.npz",
                                      self.dir / f"{learner}_actor.npz")
            out[learner] = res
            self.assertTrue(res["passed"], f"{learner}: {res}")
            self.assertEqual(res["other_side"]["device"], "cpu")
        self.__class__.h3 = {l: {k: v for k, v in r.items() if k in ("rows", "scored", "tanh_mean", "passed")}
                             for l, r in out.items()}

    def test_h3_negative_controls_fail(self):
        """The comparison has teeth: TD3+BC with its normalization spread changed by 1e-3, and CQL with its two heads
        swapped, differ from the learner's own commands by far more than 1e-7."""
        self.assertEqual(self.proc.returncode, 0, self.proc.stderr[-3000:])
        states, _ = H.read(self.dir / "td3_bc_states.npz")
        theirs, _ = H.read(self.dir / "td3_bc_commands.npz")
        a = A32.read_actor_file(self.dir / "td3_bc_actor.npz")
        wrong = a._replace(norm=dict(obs_mean=a.norm["obs_mean"], obs_std=(a.norm["obs_std"] + np.float32(1e-3))))
        d_norm = float(np.abs(A32.commands(wrong, states["obs"]) - theirs["commands"]).max())
        self.assertGreater(d_norm, 1e-5)
        states, _ = H.read(self.dir / "cql_states.npz")
        theirs, _ = H.read(self.dir / "cql_commands.npz")
        c = A32.read_actor_file(self.dir / "cql_actor.npz")
        p = dict(c.params["params"])
        p["Dense_3"], p["Dense_4"] = p["Dense_4"], p["Dense_3"]
        swapped = c._replace(params={"params": p})
        d_swap = float(np.abs(A32.commands(swapped, states["obs"], states["eps"]) - theirs["commands"]).max())
        self.assertGreater(d_swap, 1e-2)
        self.__class__.negative = dict(td3_bc_spread_changed=d_norm, cql_heads_swapped=d_swap)

    def test_h5_noise_same_bits_in_the_training_environment(self):
        """h5 across environments: numpy 1.26.4 draws CQL's declared noise with the same bits as numpy 2.2.6."""
        from experiments.shift_hosts import policies as P

        self.assertEqual(self.proc.returncode, 0, self.proc.stderr[-3000:])
        rep = H.read_json(self.dir / "unifloral_report.json")
        for b in range(7):
            self.assertEqual(rep["noise_sha256"][str(b)], H.array_sha256(P.cql_noise(b)))


# ------------------------------------------------------------------------------------------------ h4: S7 closed loop

class ClosedLoopTest(unittest.TestCase):

    def test_h4_closed_loop_within_001_points(self):
        """h4: on 8 starts of each of the 7 banks, the float32 restatement and the float64 adapter agree within 0.01
        points per start, for every learner (CQL with the declared noise, and its tanh-of-mean readout)."""
        from experiments.shift_hosts import evaluate as E
        from experiments.shift_hosts import guard

        folder = h_dir() / "closed_loop"
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)
        out = {}
        for learner in LEARNERS:
            res = E.closed_loop_check(write_made_up_actor(folder, learner), guard.MADE_UP, idx=CLOSED_LOOP_STARTS)
            out[learner] = res
            self.assertTrue(res["passed"], f"{learner}: {res}")
        self.__class__.h4 = {l: dict(passed=r["passed"], max_points=max(v["max_points"] for k, v in r.items()
                                                                        if isinstance(v, dict) and "max_points" in v))
                             for l, r in out.items()}


# ------------------------------------------------------------------------------------------------ evaluate.py

class PermissionTest(unittest.TestCase):

    def test_only_made_up_files_or_a_verified_approval(self):
        """Scoring refuses an actor file that is not made-up under guard.MADE_UP, and any other permission; the
        approval gate refuses today (no frozen pre-specification)."""
        from experiments.shift_hosts import evaluate as E
        from experiments.shift_hosts import gate, guard

        folder = h_dir() / "permission"
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)
        real_like = A32.read_actor_file(write_made_up_actor(folder, "td3_bc", made_up=False))
        made_up = A32.read_actor_file(write_made_up_actor(folder, "td3_bc"))
        with self.assertRaises(gate.ApprovalError):
            E.require_permission(guard.MADE_UP, real_like)
        with self.assertRaises(gate.ApprovalError):
            E.score_actor(real_like.path, guard.MADE_UP, banks=("strong",))
        for bad in (None, "MADE_UP", True, object()):
            with self.assertRaises(gate.ApprovalError):
                E.require_permission(bad, made_up)
        self.assertEqual(E.require_permission(guard.MADE_UP, made_up), dict(kind="made-up"))
        with self.assertRaises(gate.ApprovalError):
            gate.require_run_approval("test_scoring")
        # an actor a run trained (the runner marks it trained true, kind run-snapshot), even on made-up arrays, is a
        # learned policy: guard.MADE_UP never rolls it out (no learned policy's return before the approval); it may
        # only read its commands at given made-up states with no bank (S7's dry-run open loop)
        params, norm = made_up_actor("td3_bc", 0)
        path = folder / "td3_bc_trained.npz"
        A32.write_actor_file(path, "td3_bc", params, meta=dict(made_up=True, trained=True, kind="run-snapshot",
                                                               note="made-up parameters standing in for a run's actor"),
                             obs_mean=norm["obs_mean"], obs_std=norm["obs_std"])
        trained = A32.read_actor_file(path)
        with self.assertRaisesRegex(gate.ApprovalError, "trained"):
            E.require_permission(guard.MADE_UP, trained)
        for call in (lambda: E.score_actor(path, guard.MADE_UP, banks=("strong",)),
                     lambda: E.closed_loop_check(path, guard.MADE_UP, banks=("strong",), idx=[0]),
                     lambda: E.visited_transitions(path, folder / "v.npz", guard.MADE_UP, n_starts=2),
                     lambda: E.actor_transitions(path, np.zeros((2, 13)), folder / "a.npz", guard.MADE_UP),
                     lambda: E.open_loop_states(path, folder / "s.npz", guard.MADE_UP, banks=("strong",))):
            with self.assertRaises(gate.ApprovalError):
                call()
        self.assertEqual(E.require_permission(guard.MADE_UP, trained, rollout=False), dict(kind="made-up"))
        man = E.open_loop_states(path, folder / "s_extra.npz", guard.MADE_UP, banks=(),
                                 extra_states=np.zeros((3, 13)))
        self.assertEqual(man["meta"]["rows"], 3)
        self.assertFalse((folder / "v.npz").exists() or (folder / "a.npz").exists() or (folder / "s.npz").exists())


class EvaluateTest(unittest.TestCase):
    """evaluate.py's metrics: against learners.evaluate_policy, and on the scoring cache's fixed controllers."""

    def test_start_banks_source(self):
        """The weighting banks come from the data agent's copy when it exists (data_copy.start_banks_dir()), else
        from the original, read-only; either has the pinned SHA-256, and every cached bank's starts equal the bank's
        starts bit for bit."""
        from experiments.shift_hosts import data_copy as DC
        from experiments.shift_hosts import evaluate as E

        self.assertEqual(E.COPIED_BANKS, DC.start_banks_dir())
        root, kind = E.start_bank_root()
        if (DC.start_banks_dir() / "start_banks.hdf5").is_file():
            self.assertEqual((root, kind), (E.COPIED_BANKS, "copy"))
        else:
            self.assertEqual((root, kind), (E.ORIGINAL_BANKS, "original, read-only"))
        self.assertEqual(E.sha256_file(root / "start_banks.hdf5"), E.START_BANKS_HDF5_SHA256)
        for bank in E.BANKS:
            if E.cache_path(bank).exists():
                E.load_cache(bank)                      # raises if its starts differ from the bank's
        self.__class__.start_banks = dict(root=str(root), kind=kind)

    def test_matches_learners_evaluate_policy(self):
        """On the four situations, evaluate.score of the made-up TD3+BC adapter gives learners.evaluate_policy's
        J, J_se, discounted return, points, capped points, loss from unrecorded throttle, unrecorded-throttle share
        and finiteness, bit for bit (the same rollouts, prereg anchors)."""
        import experiments.shift.learners as L
        from experiments.shift_hosts import evaluate as E
        from experiments.shift_hosts import policies as P

        folder = h_dir() / "evaluate"
        folder.mkdir(parents=True, exist_ok=True)
        policy = P.adapter(A32.read_actor_file(write_made_up_actor(folder, "td3_bc")))
        anchors = {s: dict(recording=E.anchor(s)["bottom"], top=E.anchor(s)["top"]) for s in E.SITUATIONS}
        ref = L.evaluate_policy(policy, E.PHYSICS, anchors)
        pairs = dict(J="J_mean", J_se="J_se", J_discounted="J_discounted_mean", points="points_mean",
                     points_se="points_se", capped_points="points_capped_mean",
                     loss_from_unrecorded_throttle="loss_from_unrecorded_throttle_points",
                     unrecorded_throttle_share="stall_share", finite="finite")
        for s in E.SITUATIONS:
            mine = E.score(policy, s).summary
            for theirs, ours in pairs.items():
                self.assertEqual(ref[s][theirs], mine[ours], f"{s} {theirs}")

    def test_fixed_controllers_on_the_cache(self):
        """With the scoring cache: the stall-aware controller scores 100 points on every bank, fails F2 on exactly 1%
        of starts, never pushes past 80% (no loss from unrecorded throttle) and never beats the exact best plan; the
        zero command never fails F1; tracking + energy = -J; the viewer's episodes come from the scored rollout."""
        from experiments.shift import recorder as RC
        from experiments.shift_iw import data as D
        from experiments.shift_hosts import evaluate as E

        out = {}
        for bank in E.BANKS:
            self.assertTrue(E.cache_path(bank).exists(), "run scoring_cache.py first")
            keep = E.viewer_starts(bank)
            s = E.score(RC.stall_aware(E.PHYSICS), bank, keep=keep)
            z = E.score(D.ZeroCommand(), bank)
            m = s.summary
            self.assertAlmostEqual(m["points_mean"], 100.0, delta=1e-9)
            self.assertEqual(m["f2_rate"], 0.01)
            self.assertEqual(m["stall_share"], 0.0)
            self.assertEqual(m["loss_from_unrecorded_throttle_points"], 0.0)
            self.assertGreaterEqual(m["regret_min"], -1e-9 * abs(m["optimum_J_mean"]))
            self.assertLessEqual(m["max_cost_identity_error"], 1e-9 * abs(m["J_mean"]))
            self.assertEqual(z.summary["f1_rate"], 0.0)
            self.assertTrue(m["finite"])
            ep = s.episodes
            self.assertEqual(ep["J"].tobytes(), s.per_start["J"][keep].tobytes())
            self.assertEqual(ep["running_return"][:, -1].tobytes(), ep["J"].tobytes())
            self.assertEqual(ep["states"].shape, (len(keep), 101, 13))
            self.assertEqual(ep["commands"].shape, (len(keep), 100, 4))
            cache = E.load_cache(bank)
            self.assertEqual(cache["stall_aware/J"].tobytes(), s.per_start["J"].tobytes())
            self.assertEqual(cache["zero/J"].tobytes(), z.per_start["J"].tobytes())
            self.assertTrue(cache["meta"]["checks"]["passed"], bank)
            out[bank] = dict(points=m["points_mean"], f2=m["f2_rate"], regret_mean=m["regret_mean"],
                             zero_f1=z.summary["f1_rate"], zero_points=z.summary["points_mean"])
        p = E.paired(s.per_start["J"], s.per_start["J"])
        self.assertEqual((p["mean"], p["se"]), (0.0, 0.0))
        self.__class__.fixed = out

    def test_coverage_transitions(self):
        """The coverage hand-offs: every row's next state and reward are the simulator's (thruster_chain.dynamics at
        the row's state and applied command), visited rows follow the scored rollout start by start, and CQL's rows
        carry the declared noise of their start and step."""
        from experiments.shift import thruster_chain as TC
        from experiments.shift_hosts import evaluate as E
        from experiments.shift_hosts import guard
        from experiments.shift_hosts import policies as P

        folder = h_dir() / "coverage"
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)
        for learner in ("td3_bc", "cql"):
            path = write_made_up_actor(folder, learner)
            E.visited_transitions(path, folder / f"{learner}_visited.npz", guard.MADE_UP, n_starts=20)
            v, man = H.read(folder / f"{learner}_visited.npz")
            self.assertEqual(man["meta"]["rows"], 2000)
            nxt, r = TC.dynamics(v["state"], v["action"], E.PHYSICS)
            self.assertEqual(nxt.tobytes(), v["next_state"].tobytes())
            self.assertEqual(r.tobytes(), v["reward64"].tobytes())
            # consecutive rows of one start chain: s_{t+1} of row t is the state of row t + 1
            same = v["start"][1:] == v["start"][:-1]
            self.assertTrue(np.array_equal(v["next_state"][:-1][same], v["state"][1:][same]))
            self.assertTrue(np.array_equal(v["state"][v["t"] == 0], E.starts("strong")[:20]))
            if learner == "cql":
                eps = P.cql_noise("strong")
                self.assertEqual(v["eps"].tobytes(), eps[v["start"], v["t"]].tobytes())
            x = made_up_states(500, 4001)
            E.actor_transitions(path, x, folder / f"{learner}_actor.npz", guard.MADE_UP,
                                eps=made_up_eps(500, 4002) if learner == "cql" else None)
            w, _ = H.read(folder / f"{learner}_actor.npz")
            nxt, r = TC.dynamics(x, w["action"], E.PHYSICS)
            self.assertEqual(nxt.tobytes(), w["next_state"].tobytes())
            self.assertEqual(w["obs"].tobytes(), x.astype(np.float32).tobytes())

    def test_score_actor_every_learner(self):
        """Each learner's made-up actor through evaluate.score_actor on every bank (CQL with both readouts), written
        by write_scores and read back: finite returns, regret at least 0 on every start, rates in [0, 1], and the
        file's per-start returns equal the scores in memory."""
        from experiments.shift_hosts import evaluate as E
        from experiments.shift_hosts import guard

        folder = h_dir() / "score_actor"
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)
        out = {}
        for learner in LEARNERS:
            path = write_made_up_actor(folder, learner)
            res = E.score_actor(path, guard.MADE_UP, keep_viewer=True, out=folder / f"{learner}_scores.npz")
            arrays, man = H.read(folder / f"{learner}_scores.npz")
            self.assertEqual(sorted(man["meta"]["summaries"]), sorted(A32.READOUTS[learner]))
            for (readout, bank), s in res.items():
                self.assertTrue(s.summary["finite"])
                self.assertGreaterEqual(float(s.per_start["regret"].min()), -1e-9 * abs(s.summary["optimum_J_mean"]))
                for k in ("f1_rate", "f2_rate", "stall_share"):
                    self.assertTrue(0.0 <= s.summary[k] <= 1.0)
                self.assertEqual(arrays[f"{readout}/{bank}/J"].tobytes(), s.per_start["J"].tobytes())
                self.assertEqual(arrays[f"{readout}/{bank}/episodes/J"].tobytes(), s.episodes["J"].tobytes())
            out[learner] = {f"{r}/{b}": round(s.summary["points_mean"], 3) for (r, b), s in res.items()}
        self.__class__.made_up_points = out


def _extra():
    out = {}
    for cls, names in ((AdapterTest, ("h1", "h2", "h5")), (OpenLoopTest, ("h3", "negative")),
                       (ClosedLoopTest, ("h4",)), (EvaluateTest, ("fixed", "made_up_points", "start_banks"))):
        for n in names:
            if hasattr(cls, n):
                out[n] = H.plain(getattr(cls, n))
    out["settings"] = A32.settings()
    out["note"] = "made-up actor parameters and made-up states only; no learned policy exists or was scored"
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--role", choices=("unifloral",))
    p.add_argument("--dir")
    a = p.parse_args()
    if a.role == "unifloral":
        sys.exit(_unifloral_role(a.dir))
    A32.configure()
    from experiments.shift_hosts import tests as T

    sys.exit(T.run(sys.modules[__name__], AREA, extra=_extra))
