"""The runner's and the band readout's stage-0 tests (plan section 4.5; area "runner"), on made-up arrays only
(tests/madeup.py, registered with the guard; written in the prepared-file format by runner.write_made_up_prepared): no
ThrusterChain data is trained on, no band is fitted on real arrays, no learned policy's return is computed. BCA's code
runs here on made-up arrays only (approval item 8). An experiment-local attachment (plan section 3.1) that changes none
of the repository's declared methods.

  (g)       The runner's chunk loop equals a literal transcription of each training block with the evaluation removed
            (td3_bc.py:283-340, cql.py:329-388, iql.py:272-332, rebrac.py:313-372; the transcriptions are below), for 3
            chunks with eval_interval 50: the final key and learner state (every leaf, from the run's final snapshot)
            and every chunk's per-step losses bit for bit, and the S4 key hashes taken where the transcription's key
            leaves each chunk. Extra: a BCA setup's Unifloral part after the same 3 chunks, no refresh yet, is the
            transcription's too (the processor's (a2)).
  (b)       Weights of 1 = standard, every learner: a 300-row bank, refreshes after steps 20, 30 and 40 (every 10 after
            20), 50 steps. Every weight and mass 1, so the IW grid is [1.0]; at each refresh R(1) equals the stored
            threshold (float32) and S3 held with the bank scores recomputed from the stored state equal to the
            refresh's own bit for bit; at the end every leaf of the two setups is equal but the carried refusal rule,
            and so is every per-step output (stages.check_s2). The IW setup with exact weights also passes S3 at
            every refresh and records the last refresh's readouts.
  S4, S10   Hash plumbing: the record's key hashes (10, 20, 40) and learner-state hash at the first refresh (20) equal
            hashes computed independently from a transcription; stages.check_s4 and check_s10 pass on the runs (alone,
            standard, IW, room) and fail on a perturbed copy; S10's readout reports snapshot differences.
  resume    A finished run is skipped with its record unchanged; another identity in the same folder and a started run
            without a record are refused; --replace-incomplete starts it again; the command line skips a finished run
            without importing Unifloral's files.
  gate      Runs on real data are refused before anything is read (no prepared file opened, no chunk scanned, no folder
            made): the runner's command line in this process and as a subprocess, band_readout, the stage driver
            (run and start), a scoring lane. The made-up token on arrays that are not the registered made-up arrays is
            refused, and so is a real file read as made-up.
  stops     Made-up faults: BCA's unit set to 0 stops a BCA run with S5 at the chunk's first step; NaN parameters stop
            the learner alone with S5 (its Claim 0 outcome); the command line exits 3.
  schedule  The plan's schedule (40 chunks; 18 refreshes 10,000-95,000; S4 at 2,500, 50,000, 100,000; S10 at 10,000;
            snapshots at 10,000, 50,000, 95,000, 100,000) and stage 1's (one refresh; snapshots 10,000 and 15,000).
  weights   runner.strong_weight equals world.ratio(c, "strong") bit for bit (the scoring environment, a subprocess).
  band      band_readout: the coverage arithmetic on hand-made cases; y and q at the made-up strong-law rows equal the
            learner's own band_parts on the same rows and key (to float32 rounding: separately compiled programs; the
            bit-identical share is recorded); U = R x u x max(eta, 1e-6); the IW thresholds come from the last
            refresh's table; the viewer's arrays.

Run from /root/bca-hosts in /opt/bca-venv with the Dockerfile's variables, on the processor (cores 0-5):
  env D4RL_SUPPRESS_IMPORT_ERROR=1 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa DISPLAY= \\
    MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210 LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin \\
    PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu SHIFT_HOSTS_DEVICE=cpu \\
    taskset -c 0-5 /opt/bca-venv/bin/python -m experiments.shift_hosts.tests.test_runner
On the card (SHIFT_HOSTS_DEVICE=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false, no JAX_PLATFORMS) only (g), (b) and the
schedule case run, the rest are processor-only; there (g)'s extra is held to (a2)'s card rule (the key bit for bit,
every leaf of the learner state within 1e-6 relative; the bit-for-bit result is recorded).
Result: runs/shift_hosts/stage0/runner/test_runner_<device>.json (+ .sha256); run folders under
runs/shift_hosts/stage0/runner/test_runner/ (the card run under test_runner_gpu/).
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from experiments.shift_hosts import gate, guard
from experiments.shift_hosts import handoff as H
from experiments.shift_hosts import runner as RN
from experiments.shift_hosts import tests as T
from experiments.shift_hosts import unifloral_import as UI

AREA = "runner"
ROOT = T.STAGE0 / AREA / ("test_runner_gpu" if T.device() == "gpu" else "test_runner")
TRAINING_PYTHON = "/opt/bca-venv/bin/python"
SCORING_PYTHON = "/opt/bca-shift-venv/bin/python"
ROUNDING = 1e-5
OUTSIDE = 1e-6
CARD_RELATIVE = 1e-6        # (a2)'s card rule: every leaf within 1e-6 relative
READOUTS = {}
_SHARED = {}
ON_CARD = T.device() == "gpu"
processor_only = unittest.skipIf(ON_CARD, "processor-only case: on the card only (g) and (b) run")


def quiet(msg):
    pass


# ------------------------------------------------------------------------------------------------ helpers

def _leaves(tree):
    import jax
    import jax.numpy as jnp

    return [np.asarray(jnp.asarray(x)) for x in jax.tree_util.tree_leaves(tree)]


def _same(a, b):
    """Bit for bit after jnp.asarray on every leaf (INTERFACES.md section 3; guard.state_sha256's rule)."""
    la, lb = _leaves(a), _leaves(b)
    return len(la) == len(lb) and all(x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()
                                      for x, y in zip(la, lb))


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    d, s = float(np.max(np.abs(a - b), initial=0.0)), float(np.max(np.abs(b), initial=0.0))
    return 0.0 if d == 0.0 else (d / s if s > 0 else math.inf)


def made_up_file():
    if "file" not in _SHARED:
        path = ROOT / "data" / "madeup0.npz"
        RN.write_made_up_prepared(path, 0)
        _SHARED["file"] = path
    return _SHARED["file"]


def prepared():
    if "prepared" not in _SHARED:
        p = RN.read_prepared(made_up_file(), made_up=True)
        RN.made_up_permission(p)
        _SHARED["prepared"] = p
    return _SHARED["prepared"]


def schedule(num_updates, eval_interval, *, steps=None, warmup=None, every=None):
    return RN.make_schedule(num_updates=num_updates, eval_interval=eval_interval, steps=steps,
                            warmup=warmup if warmup is not None else 2 * eval_interval,
                            every=every if every is not None else eval_interval)


def execute(learner, setup, sched, stage, **kw):
    return RN.execute(prepared(), learner, setup, guard.MADE_UP, sched, stage=stage, root=ROOT, device=T.device(),
                      log=quiet, **kw)


def final_carry(record, learner, setup, sched):
    s = RN.build_setup(learner, setup, prepared(), guard.MADE_UP, sched)
    snap = record["snapshots"][str(record["timings"]["outer_steps"])]
    carry, _ = RN.load_snapshot(snap["path"], s.carry0)
    return carry


def subprocess_env(device="cpu"):
    env = UI.launch_env(dict(os.environ))
    env.update(SHIFT_HOSTS_DEVICE=device, JAX_PLATFORMS="cpu")
    return env


# ---------------------------------------------------------------------------------------- (g): the blocks, transcribed

def _rows_as_d4rl(rows):
    """The made-up rows under d4rl.qlearning_dataset's keys (what td3_bc.py:298 hands the block)."""
    return dict(observations=rows["obs"], actions=rows["action"], rewards=rows["reward"],
                next_observations=rows["next_obs"], terminals=rows["done"])


def block_td3_bc(args, rows):
    """td3_bc.py:283-340, literally, with the made-up rows for D4RL and the evaluation removed (its key split kept).
    Substitutions: tyro.cli(Args) -> args; args.log is False, so wandb never starts; env.single_action_space.shape[0]
    -> 4 and env.single_observation_space.shape -> (13,). Returns (rng, agent_state, losses, keys out of each chunk)."""
    import jax
    import jax.numpy as jnp

    U = UI.load("td3_bc")
    rng = jax.random.PRNGKey(args.seed)
    dataset = _rows_as_d4rl(rows)
    dataset = U.Transition(
        obs=jnp.array(dataset["observations"]),
        action=jnp.array(dataset["actions"]),
        reward=jnp.array(dataset["rewards"]),
        next_obs=jnp.array(dataset["next_observations"]),
        done=jnp.array(dataset["terminals"]),
    )
    num_actions = 4
    obs_mean = dataset.obs.mean(axis=0)
    obs_std = jnp.nan_to_num(dataset.obs.std(axis=0), nan=1.0)
    dummy_obs = jnp.zeros((13,))
    dummy_action = jnp.zeros(num_actions)
    actor_net = U.DeterministicTanhActor(num_actions, obs_mean, obs_std)
    q_net = U.DualQNetwork(obs_mean, obs_std)
    rng, rng_actor, rng_q = jax.random.split(rng, 3)
    agent_state = U.AgentTrainState(
        actor=U.create_train_state(args, rng_actor, actor_net, [dummy_obs]),
        actor_target=U.create_train_state(args, rng_actor, actor_net, [dummy_obs]),
        dual_q=U.create_train_state(args, rng_q, q_net, [dummy_obs, dummy_action]),
        dual_q_target=U.create_train_state(args, rng_q, q_net, [dummy_obs, dummy_action]),
    )
    _agent_train_step_fn = U.make_train_step(args, actor_net.apply, q_net.apply, dataset)
    return _loop(args, _agent_train_step_fn, rng, agent_state)


def block_cql(args, rows):
    """cql.py:329-388, literally, with the same substitutions as block_td3_bc."""
    import jax
    import jax.numpy as jnp

    U = UI.load("cql")
    rng = jax.random.PRNGKey(args.seed)
    dataset = _rows_as_d4rl(rows)
    dataset = U.Transition(
        obs=jnp.array(dataset["observations"]),
        action=jnp.array(dataset["actions"]),
        reward=jnp.array(dataset["rewards"]),
        next_obs=jnp.array(dataset["next_observations"]),
        done=jnp.array(dataset["terminals"]),
    )
    num_actions = 4
    dummy_obs = jnp.zeros((13,))
    dummy_action = jnp.zeros(num_actions)
    actor_net = U.TanhGaussianActor(num_actions)
    q_net = U.VectorQ(args.num_critics)
    alpha_net = U.EntropyCoef()
    rng, rng_actor, rng_q, rng_alpha = jax.random.split(rng, 4)
    actor_lr = args.actor_lr if args.actor_lr is not None else args.lr
    agent_state = U.AgentTrainState(
        actor=U.create_train_state(args, rng_actor, actor_net, [dummy_obs], actor_lr),
        vec_q=U.create_train_state(args, rng_q, q_net, [dummy_obs, dummy_action]),
        vec_q_target=U.create_train_state(args, rng_q, q_net, [dummy_obs, dummy_action]),
        alpha=U.create_train_state(args, rng_alpha, alpha_net, []),
    )
    _agent_train_step_fn = U.make_train_step(
        args, actor_net.apply, q_net.apply, alpha_net.apply, dataset
    )
    return _loop(args, _agent_train_step_fn, rng, agent_state)


def block_iql(args, rows):
    """iql.py:272-332, literally, with the same substitutions as block_td3_bc."""
    import jax
    import jax.numpy as jnp

    U = UI.load("iql")
    rng = jax.random.PRNGKey(args.seed)
    dataset = _rows_as_d4rl(rows)
    dataset = U.Transition(
        obs=jnp.array(dataset["observations"]),
        action=jnp.array(dataset["actions"]),
        reward=jnp.array(dataset["rewards"]),
        next_obs=jnp.array(dataset["next_observations"]),
        done=jnp.array(dataset["terminals"]),
    )
    num_actions = 4
    obs_mean = dataset.obs.mean(axis=0)
    obs_std = jnp.nan_to_num(dataset.obs.std(axis=0), nan=1.0)
    dummy_obs = jnp.zeros((13,))
    dummy_action = jnp.zeros(num_actions)
    actor_net = U.TanhGaussianActor(num_actions, obs_mean, obs_std)
    q_net = U.DualQNetwork(obs_mean, obs_std)
    value_net = U.StateValueFunction(obs_mean, obs_std)
    rng, rng_actor, rng_q, rng_value = jax.random.split(rng, 4)
    agent_state = U.AgentTrainState(
        actor=U.create_train_state(args, rng_actor, actor_net, [dummy_obs]),
        dual_q=U.create_train_state(args, rng_q, q_net, [dummy_obs, dummy_action]),
        dual_q_target=U.create_train_state(args, rng_q, q_net, [dummy_obs, dummy_action]),
        value=U.create_train_state(args, rng_value, value_net, [dummy_obs]),
    )
    _agent_train_step_fn = U.make_train_step(
        args, actor_net.apply, q_net.apply, value_net.apply, dataset
    )
    return _loop(args, _agent_train_step_fn, rng, agent_state)


def block_rebrac(args, rows):
    """rebrac.py:313-372, literally, with the same substitutions as block_td3_bc (next_action is the block's own
    jnp.roll of the actions, rebrac.py:335)."""
    import jax
    import jax.numpy as jnp

    U = UI.load("rebrac")
    rng = jax.random.PRNGKey(args.seed)
    dataset = _rows_as_d4rl(rows)
    dataset = U.Transition(
        obs=jnp.array(dataset["observations"]),
        action=jnp.array(dataset["actions"]),
        reward=jnp.array(dataset["rewards"]),
        next_obs=jnp.array(dataset["next_observations"]),
        next_action=jnp.roll(dataset["actions"], -1, axis=0),
        done=jnp.array(dataset["terminals"]),
    )
    num_actions = 4
    obs_mean = dataset.obs.mean(axis=0)
    obs_std = jnp.nan_to_num(dataset.obs.std(axis=0), nan=1.0)
    dummy_obs = jnp.zeros((13,))
    dummy_action = jnp.zeros(num_actions)
    actor_cls = U.DeterministicTanhActor
    actor_net = actor_cls(num_actions, obs_mean, obs_std, args.actor_ln, args.norm_obs)
    q_net = U.DualQNetwork(obs_mean, obs_std, args.critic_ln, args.norm_obs)
    rng, rng_actor, rng_q = jax.random.split(rng, 3)
    agent_state = U.AgentTrainState(
        actor=U.create_train_state(args, rng_actor, actor_net, [dummy_obs]),
        actor_target=U.create_train_state(args, rng_actor, actor_net, [dummy_obs]),
        dual_q=U.create_train_state(args, rng_q, q_net, [dummy_obs, dummy_action]),
        dual_q_target=U.create_train_state(args, rng_q, q_net, [dummy_obs, dummy_action]),
    )
    _agent_train_step_fn = U.make_train_step(args, actor_net.apply, q_net.apply, dataset)
    return _loop(args, _agent_train_step_fn, rng, agent_state)


def _loop(args, _agent_train_step_fn, rng, agent_state):
    """The blocks' common loop (td3_bc.py:329-340 and the others), literally, with the evaluation removed: the key split
    that made rng_eval is kept and rng_eval discarded. Also keeps each chunk's losses and the key carried out of it."""
    import jax

    losses, keys = [], []
    num_evals = args.num_updates // args.eval_interval
    for eval_idx in range(num_evals):
        # --- Execute train loop ---
        (rng, agent_state), loss = jax.lax.scan(
            _agent_train_step_fn,
            (rng, agent_state),
            None,
            args.eval_interval,
        )
        losses.append({k: np.asarray(v) for k, v in loss.items()})
        keys.append(rng)
        # --- Evaluate agent --- (removed; its key split stays)
        rng, rng_eval = jax.random.split(rng)
    return rng, agent_state, losses, keys


BLOCKS = dict(td3_bc=block_td3_bc, cql=block_cql, iql=block_iql, rebrac=block_rebrac)


# ------------------------------------------------------------------------------------------------ the tests

class Setup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        UI.prepare()
        if "root" not in _SHARED:
            if ROOT.exists():
                shutil.rmtree(ROOT)
            ROOT.mkdir(parents=True)
            _SHARED["root"] = ROOT
        prepared()


class TrainingBlockTest(Setup):
    def test_g_chunk_loop_equals_training_block(self):
        """(g) The runner's chunk loop equals each training block's, evaluation removed: 3 chunks of 50."""
        sched = schedule(150, 50, warmup=100, every=50)
        no_refresh = schedule(150, 50, warmup=200, every=50)
        detail = {}
        for learner in RN.LEARNERS:
            args = RN.unifloral_args(learner, prepared().rep, sched)
            rng, agent_state, losses, keys = BLOCKS[learner](args, prepared().rows("train"))
            rec, status = execute(learner, "alone", sched, "test_g")
            self.assertEqual((status, rec["outcome"]), ("complete", "complete"))
            carry = final_carry(rec, learner, "alone", sched)
            self.assertTrue(_same(carry[0], rng), f"{learner}: final key")
            self.assertTrue(_same(carry[1], agent_state), f"{learner}: final learner state")
            self.assertEqual([c["sha256"] for c in rec["losses"]], [guard.tree_sha256(l) for l in losses],
                             f"{learner}: per-step losses")
            want = {str(50 * (i + 1)): guard.state_sha256(k) for i, k in enumerate(keys)}
            self.assertEqual(rec["hashes"]["key_carry"], {t: want[t] for t in ("50", "100", "150")})
            rec_b, _ = execute(learner, "standard", no_refresh, "test_g_bca")
            carry_b = final_carry(rec_b, learner, "standard", no_refresh)
            bca_same = _same(carry_b[0], rng) and _same(carry_b[1], agent_state)
            worst = 0.0 if bca_same else max(_rel(a, b) for a, b in zip(_leaves(carry_b[1]), _leaves(agent_state)))
            if ON_CARD:          # (a2)'s card rule: bit for bit, or every leaf within 1e-6 relative (recorded)
                self.assertTrue(_same(carry_b[0], rng), f"{learner}: the key carry")
                self.assertLessEqual(worst, CARD_RELATIVE, f"{learner}: the BCA setup's Unifloral part")
            else:
                self.assertTrue(bca_same, f"{learner}: the BCA setup's Unifloral part (no refresh) is the block's")
            detail[learner] = dict(chunks=len(losses), leaves=len(_leaves(agent_state)),
                                   step_counter=int(np.asarray(agent_state.actor.step)),
                                   bca_part_bit_for_bit=bool(bca_same), bca_part_max_relative=worst)
        READOUTS["g"] = detail


class WeightsOfOneTest(Setup):
    def test_b_weights_of_one_equal_standard(self):
        """(b) IW with every weight 1 equals standard BCA in every bit (all four learners)."""
        from experiments.shift_hosts import stages as ST

        sched = schedule(50, 10, warmup=20, every=10)
        self.assertEqual(sched.refresh_steps, (20, 30, 40))
        self.assertEqual(prepared().n("bank"), 300)
        detail = {}
        for learner in RN.LEARNERS:
            std, _ = execute(learner, "standard", sched, "test_b")
            one, _ = execute(learner, "iw_ones", sched, "test_b")
            iw, _ = execute(learner, "iw", sched, "test_b")
            for r in (std, one, iw):
                self.assertEqual(r["outcome"], "complete", r["run_id"])
            res = ST.check_s2(std, one)
            self.assertTrue(res["passed"], f"{learner}: {res}")
            for ref in one["refreshes"]:
                self.assertEqual(ref["iw"]["table"]["grid"], [1.0])
                self.assertTrue(ref["s3_reproduced"] and ref["s3_recomputed_equals_refresh_scores"])
            a, _ = H.read(std["snapshots"]["50"]["path"])
            b, _ = H.read(one["snapshots"]["50"]["path"])
            self.assertEqual(sorted(a), sorted(b))
            differ = sorted(n for n in a if a[n].tobytes() != b[n].tobytes())
            self.assertEqual(differ, ["bca/most_caution"], f"{learner}: leaves that differ")
            for ref in iw["refreshes"]:
                self.assertTrue(ref["s3_reproduced"] and ref["s3_recomputed_equals_refresh_scores"], learner)
                self.assertGreater(ref["iw"]["grid_points"], 1)
            self.assertEqual(sorted(iw["readouts"]),
                             ["dose_change", "estimated_weights", "exact", "info", "shuffled_weights", "value"])
            detail[learner] = dict(thresholds=[r["threshold"] for r in std["refreshes"]], leaves=len(a),
                                   iw_refused_rows=[r["iw_refused_rows"] for r in iw["refreshes"]],
                                   iw_mean_dose=[r["mean_dose"] for r in iw["refreshes"]],
                                   dose_change=iw["readouts"]["dose_change"].get("change_vs_unweighted"))
        READOUTS["b"] = detail


class HashPlumbingTest(Setup):
    @processor_only
    def test_s4_s10_hashes(self):
        """S4 and S10: the recorded hashes equal hashes computed independently; the checks pass and catch changes."""
        import jax

        from experiments.shift_hosts import stages as ST

        learner = "td3_bc"
        sched = schedule(40, 10, warmup=20, every=10)
        self.assertEqual((sched.key_hash_steps, sched.s10_step), ((10, 20, 40), 20))
        recs = {s: execute(learner, s, sched, "test_s4")[0] for s in ("alone", "standard", "iw", "room")}
        args = RN.unifloral_args(learner, prepared().rep, sched)
        built = UI.init_like_training_block(learner, args, prepared().rows("train"))
        step = UI.make_unifloral_step(built, args)
        carry, keys, at20 = (built.rng, built.agent_state), {}, None
        for t in (10, 20, 30, 40):
            carry, _ = jax.lax.scan(step, carry, None, 10)
            keys[str(t)] = guard.state_sha256(carry[0])
            if t == 20:
                at20 = (guard.state_sha256(carry[1]), guard.state_sha256(carry[0]))
            carry = (jax.random.split(carry[0])[0], carry[1])
        for s, r in recs.items():
            self.assertEqual(r["hashes"]["key_carry"], {t: keys[t] for t in ("10", "20", "40")}, s)
            self.assertEqual((r["hashes"]["params_10000"]["agent_state"], r["hashes"]["params_10000"]["key"]), at20, s)
            self.assertEqual(r["hashes"]["initial"]["agent_state"]["all"], guard.state_sha256(built.agent_state), s)
        self.assertTrue(ST.check_s4(recs)["passed"])
        bad = {s: dict(r) for s, r in recs.items()}
        bad["iw"] = dict(recs["iw"], hashes=dict(recs["iw"]["hashes"], key_carry=dict(recs["iw"]["hashes"]["key_carry"],
                                                                                   **{"20": "0" * 64})))
        res = ST.check_s4(bad)
        self.assertFalse(res["passed"])
        self.assertEqual(res["key_differs_at"], [20])
        bad2 = dict(recs, room=dict(recs["room"], hashes=dict(recs["room"]["hashes"], initial=dict(
            recs["room"]["hashes"]["initial"], agent_state=dict(all="0" * 64)))))
        self.assertFalse(ST.check_s4(bad2)["initial_equal"])
        for s in ("standard", "iw"):
            self.assertTrue(ST.check_s10(recs["alone"], recs[s], False)["passed"])
        moved = dict(recs["standard"], hashes=dict(recs["standard"]["hashes"], params_10000=dict(
            recs["standard"]["hashes"]["params_10000"], agent_state="0" * 64)))
        self.assertFalse(ST.check_s10(recs["alone"], moved, False)["passed"])
        readout = ST.check_s10(recs["alone"], moved, True)
        self.assertTrue(readout["passed"] and readout["readout"])
        self.assertEqual(readout["difference"]["differing"], 0)     # only the hash was changed, not the snapshot
        snap = lambda s, t: recs[s]["snapshots"][t]["path"]
        same = ST.snapshot_difference(snap("alone", "20"), snap("standard", "20"))
        later = ST.snapshot_difference(snap("alone", "20"), snap("alone", "40"))
        self.assertEqual(same["differing"], 0)
        self.assertGreater(later["differing"], 0)
        self.assertGreater(later["max_relative"], 0.0)
        READOUTS["s4_s10"] = dict(key_steps=sorted(keys), snapshot_20_vs_40=later)


class ResumeTest(Setup):
    @processor_only
    def test_resume_and_refusals(self):
        """A finished run is skipped; another identity and an unfinished run are refused; replacing is explicit."""
        sched = schedule(20, 10, warmup=10, every=10)
        rec, status = execute("td3_bc", "alone", sched, "test_resume")
        self.assertEqual(status, "complete")
        path = RN.run_dir(ROOT, "test_resume", "made-up", 0, "td3_bc", "alone") / "record.json"
        before = H.sha256_file(path)
        again, status = execute("td3_bc", "alone", sched, "test_resume")
        self.assertEqual(status, "skipped")
        self.assertEqual(H.sha256_file(path), before)
        with self.assertRaises(RN.RecordMismatch):
            execute("td3_bc", "alone", schedule(20, 10, steps=10, warmup=10, every=10), "test_resume")
        d = RN.run_dir(ROOT, "test_resume", "made-up", 0, "td3_bc", "standard")
        d.mkdir(parents=True, exist_ok=True)
        H.write_json(d / "started.json", dict(note="a run that died before its record"))
        with self.assertRaises(RN.RecordMismatch):
            execute("td3_bc", "standard", sched, "test_resume")
        rec2, status = execute("td3_bc", "standard", sched, "test_resume", replace_incomplete=True)
        self.assertEqual((status, rec2["outcome"]), ("complete", "complete"))
        out = subprocess.run([TRAINING_PYTHON, "-m", "experiments.shift_hosts.runner", "run", "--learner", "td3_bc",
                              "--setup", "alone", "--made-up-file", str(made_up_file()), "--num-updates", "20",
                              "--eval-interval", "10", "--refresh-warmup", "10", "--refresh-every", "10", "--stage",
                              "test_resume", "--out-root", str(ROOT), "--device", T.device()],
                             cwd=str(RN.REPO), env=subprocess_env(), capture_output=True, text=True, timeout=600)
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        self.assertIn('"status": "skipped"', out.stdout)
        self.assertNotIn("pybullet", out.stdout + out.stderr)       # Unifloral's files (d4rl) were never imported
        self.assertEqual(H.sha256_file(path), before)


class GateTest(Setup):
    @processor_only
    def test_runner_refuses_real_data_before_reading(self):
        """The runner refuses a real run before anything is read: no prepared file opened, no chunk, no folder."""
        tmp = ROOT / "gate"
        tmp.mkdir(exist_ok=True)
        calls = []
        saved = RN.prepared_header, RN.read_prepared
        RN.prepared_header = lambda *a, **k: calls.append("header")
        RN.read_prepared = lambda *a, **k: calls.append("read")
        chunks = RN.CHUNKS_RUN[0]
        try:
            code = RN.main(["run", "--learner", "td3_bc", "--setup", "alone", "--quality", "medium-expert", "--seed",
                            "0", "--out-root", str(tmp / "real")])
        finally:
            RN.prepared_header, RN.read_prepared = saved
        self.assertEqual(code, RN.EXIT_REFUSED)
        self.assertEqual(calls, [])
        self.assertEqual(RN.CHUNKS_RUN[0], chunks)
        self.assertFalse((tmp / "real").exists())
        with self.assertRaises(gate.ApprovalError):
            gate.require_run_approval("test")
        out = subprocess.run([TRAINING_PYTHON, "-m", "experiments.shift_hosts.runner", "run", "--learner", "cql",
                              "--setup", "standard", "--quality", "medium", "--seed", "3", "--out-root",
                              str(tmp / "real2")], cwd=str(RN.REPO), env=subprocess_env(), capture_output=True,
                             text=True, timeout=600)
        self.assertEqual(out.returncode, RN.EXIT_REFUSED)
        self.assertIn("REFUSED", out.stderr)
        self.assertNotIn("pybullet", out.stdout + out.stderr)
        self.assertFalse((tmp / "real2").exists())
        READOUTS["gate_message"] = [l for l in out.stderr.splitlines() if "REFUSED" in l][0][:400]

    @processor_only
    def test_other_entry_points_refuse(self):
        """band_readout, the stage driver (run and start) and a scoring lane refuse real data without the approval."""
        from experiments.shift_hosts import band_readout as BR
        from experiments.shift_hosts import stages as ST

        tmp = ROOT / "gate2"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir()
        rec, _ = execute("td3_bc", "standard", schedule(20, 10, warmup=10, every=10), "test_gate2")
        fake = dict(rec, data=dict(rec["data"], made_up=False))
        H.write_json(tmp / "run" / "record.json", fake)
        self.assertEqual(BR.main(["coverage", "--run-dir", str(tmp / "run"), "--out", str(tmp / "out")]), 2)
        self.assertFalse((tmp / "out").exists())
        self.assertEqual(ST.main(["run", "--root", str(tmp / "driver_root")]), 2)
        self.assertFalse((tmp / "driver_root").exists())
        self.assertEqual(ST.main(["start", "--root", str(tmp / "driver_root2")]), 2)
        self.assertFalse((tmp / "driver_root2").exists())
        with self.assertRaises(gate.ApprovalError):
            ST.lane_main(["score", "--actor", str(tmp / "none.npz"), "--out", str(tmp / "none_scores.npz")])

    @processor_only
    def test_made_up_token_needs_registered_arrays(self):
        """guard.MADE_UP opens a run only on the registered made-up arrays; a real file is never read as made-up."""
        arrays, man = H.read(made_up_file())
        arrays["train/obs"] = arrays["train/obs"].copy()
        arrays["train/obs"][0, 0] += np.float32(1.0)
        H.write(ROOT / "data" / "changed.npz", arrays, meta=man["meta"])
        changed = RN.read_prepared(ROOT / "data" / "changed.npz", made_up=True)
        with self.assertRaises(gate.ApprovalError):
            RN.made_up_permission(changed)
        real = RN.prepared_path("medium-expert", 0)
        if real.exists():
            with self.assertRaises(gate.ApprovalError):
                RN.prepared_header(real, made_up=True)
        with self.assertRaises(gate.ApprovalError):
            RN.execute(prepared()._replace(made_up=False), "td3_bc", "alone", guard.MADE_UP,
                       schedule(20, 10, warmup=10, every=10), stage="test_gate3", root=ROOT, inject="nan:5",
                       log=quiet)
        with self.assertRaises(ValueError):
            RN.execute(prepared()._replace(made_up=False), "td3_bc", "alone", guard.MADE_UP,
                       schedule(20, 10, warmup=10, every=10), stage="test_gate3", root=ROOT, log=quiet)
        # execute itself checks the permission against the data (the alone and room setups run Unifloral's imported
        # step, which has no entry check): real data with the made-up token, or made-up data with anything else
        for setup in ("alone", "room"):
            with self.assertRaises(gate.ApprovalError):
                RN.execute(prepared()._replace(made_up=False), "td3_bc", setup, guard.MADE_UP, RN.make_schedule(),
                           stage="test_gate3", root=ROOT, log=quiet)
            with self.assertRaises(gate.ApprovalError):
                RN.execute(prepared(), "td3_bc", setup, object(), schedule(20, 10, warmup=10, every=10),
                           stage="test_gate3", root=ROOT, log=quiet)
        self.assertFalse((ROOT / "test_gate3").exists())


class ReviewRulesTest(Setup):
    """The review's runner fixes (made-up arrays only): a stop before training names its rule on the status line; the
    replace of an incomplete run waits for its process to be gone; s7-commands needs the approval unless both files
    are made-up; every actor file a run writes says trained true."""

    @processor_only
    def test_stop_before_training_prints_its_rule(self):
        """A prepared file whose own S11 check failed stops the runner with S11 before execute: no record, exit 3, and
        a status line {"status": "stopped", "record": false, "stop": {"rule": "S11", ...}} for the driver."""
        import contextlib
        import io

        arrays, man = H.read(made_up_file())
        meta = dict(man["meta"], s11=dict(ok=False, rule="made-up: the preparation's S11 check failed"))
        bad = ROOT / "data" / "s11_failed.npz"
        H.write(bad, arrays, meta=meta)
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = RN.main(["run", "--learner", "td3_bc", "--setup", "alone", "--made-up-file", str(bad),
                            "--num-updates", "20", "--eval-interval", "10", "--refresh-warmup", "10",
                            "--refresh-every", "10", "--stage", "test_s11", "--out-root", str(ROOT), "--device",
                            T.device()])
        self.assertEqual(code, RN.EXIT_STOPPED)
        line = json.loads([l for l in out.getvalue().splitlines() if l.startswith("{")][-1])
        self.assertEqual((line["status"], line["record"], line["stop"]["rule"]), ("stopped", False, "S11"))
        self.assertFalse(list((ROOT / "test_s11").rglob("record.json")))
        READOUTS["stop_before_training"] = line
        with self.assertRaises(RN.StopRun):                     # S9 before training: no frozen signature here
            RN.check_s9_before_training(UI.record(), reference={})
        sig = RN.env_signature(dict(environment=dict(import_record=UI.record())))
        self.assertTrue(RN.check_s9_before_training(UI.record(), reference=sig)["passed"])
        with self.assertRaises(RN.StopRun):
            RN.check_s9_before_training(UI.record(), reference=dict(sig, python="2.7"))

    @processor_only
    def test_replace_waits_for_a_live_process(self):
        """--replace-incomplete deletes nothing while the started run's process is alive on this boot."""
        d = RN.run_dir(ROOT, "test_alive", "made-up", 0, "td3_bc", "alone")
        d.mkdir(parents=True, exist_ok=True)
        (d / "snapshots").mkdir(exist_ok=True)
        (d / "snapshots" / "keep.npz").write_bytes(b"x")
        live = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)",
                                 "experiments.shift_hosts.runner"])
        try:
            H.write_json(d / "started.json", dict(pid=live.pid, boot_id=RN.boot_id(), note="made-up"))
            self.assertTrue(RN.process_alive(live.pid, RN.boot_id(), "experiments.shift_hosts.runner"))
            with self.assertRaisesRegex(RN.RecordMismatch, "still alive"):
                RN.existing(d, dict(run_id="x"), replace_incomplete=True)
            self.assertTrue((d / "snapshots" / "keep.npz").exists())
        finally:
            live.kill()
            live.wait(timeout=60)
        self.assertFalse(RN.process_alive(live.pid, RN.boot_id(), "experiments.shift_hosts.runner"))
        self.assertIsNone(RN.existing(d, dict(run_id="x"), replace_incomplete=True))
        self.assertFalse((d / "snapshots" / "keep.npz").exists())
        self.assertFalse(RN.process_alive(os.getpid(), "another-boot"))

    @processor_only
    def test_s7_commands_and_trained_actors(self):
        """An actor file a run wrote says trained true; s7-commands refuses (exit 2, nothing written) unless both the
        actor file and the states file are marked made-up."""
        sched = schedule(20, 10, warmup=10, every=10)
        rec, _ = execute("td3_bc", "alone", sched, "test_trained")
        actor = rec["snapshots"][str(rec["timings"]["outer_steps"])]["actor"]["path"]
        ameta = H.read_manifest(actor)["meta"]
        self.assertEqual((ameta["trained"], ameta["kind"], ameta["made_up"]), (True, "run-snapshot", True))
        arrays, man = H.read(actor)
        real_like = ROOT / "test_trained" / "actor_not_made_up.npz"
        H.write(real_like, arrays, meta=dict(man["meta"], made_up=False))
        states = ROOT / "test_trained" / "states.npz"
        H.write(states, dict(obs=np.zeros((2, 13), np.float32)), meta=dict(made_up=True, actor_npz_sha256="x"))
        out = ROOT / "test_trained" / "commands.npz"
        code = RN.main(["s7-commands", "--states", str(states), "--actor", str(real_like), "--out", str(out)])
        self.assertEqual(code, RN.EXIT_REFUSED)
        self.assertFalse(out.exists())
        unmarked = ROOT / "test_trained" / "states_unmarked.npz"
        H.write(unmarked, dict(obs=np.zeros((2, 13), np.float32)), meta=dict(actor_npz_sha256="x"))
        self.assertEqual(RN.main(["s7-commands", "--states", str(unmarked), "--actor", actor, "--out", str(out)]),
                         RN.EXIT_REFUSED)
        self.assertFalse(out.exists())


class StopTest(Setup):
    @processor_only
    def test_injected_faults_stop_with_s5(self):
        """A BCA step that is not valid (unit 0) and a learner alone with NaN parameters both stop with S5."""
        sched = schedule(40, 10, warmup=20, every=10)
        rec, status = execute("td3_bc", "standard", sched, "test_stop", inject="unit0:15")
        self.assertEqual((status, rec["outcome"]), ("stopped", "stopped"))
        self.assertEqual((rec["stop"]["rule"], rec["stop"]["step"]), ("S5", 11))
        self.assertIn("inputs_valid", rec["stop"]["reason"])
        self.assertFalse(rec["validity"]["all_steps_valid"])
        self.assertEqual(rec["timings"]["outer_steps"], 20)
        alone, _ = execute("td3_bc", "alone", sched, "test_stop", inject="nan:5")
        self.assertEqual((alone["stop"]["rule"], alone["stop"]["step"]), ("S5", 1))
        out = subprocess.run([TRAINING_PYTHON, "-m", "experiments.shift_hosts.runner", "run", "--learner", "iql",
                              "--setup", "alone", "--made-up-file", str(made_up_file()), "--num-updates", "40",
                              "--eval-interval", "10", "--refresh-warmup", "20", "--refresh-every", "10", "--stage",
                              "test_stop_cli", "--out-root", str(ROOT), "--inject-fault", "nan:25", "--device",
                              T.device()], cwd=str(RN.REPO), env=subprocess_env(), capture_output=True, text=True,
                             timeout=900)
        self.assertEqual(out.returncode, RN.EXIT_STOPPED, out.stderr[-2000:])
        cli = H.read_json(RN.run_dir(ROOT, "test_stop_cli", "made-up", 0, "iql", "alone") / "record.json")
        self.assertEqual((cli["stop"]["rule"], cli["stop"]["step"]), ("S5", 21))
        READOUTS["stops"] = dict(bca=rec["stop"], alone=alone["stop"], cli=cli["stop"])


class ScheduleTest(unittest.TestCase):
    def test_plan_schedules(self):
        """The plan's 100,000-step schedule and stage 1's 15,000-step prefix."""
        s = RN.make_schedule()
        self.assertEqual((s.num_updates, s.eval_interval, s.steps), (100_000, 2_500, 100_000))
        self.assertEqual(s.refresh_steps, tuple(range(10_000, 100_000, 5_000)))
        self.assertEqual(len(s.refresh_steps), 18)
        self.assertEqual((s.key_hash_steps, s.s10_step, s.switch_step), ((2_500, 50_000, 100_000), 10_000, 10_000))
        self.assertEqual(s.snapshot_steps, (10_000, 50_000, 95_000, 100_000))
        self.assertEqual(s.last_refresh, 95_000)
        self.assertTrue(RN.is_plan_schedule(s))
        s1 = RN.make_schedule(steps=15_000)
        self.assertEqual((s1.refresh_steps, s1.key_hash_steps, s1.snapshot_steps, s1.last_refresh),
                         ((10_000,), (2_500,), (10_000, 15_000), None))
        self.assertTrue(RN.is_plan_schedule(s1))
        self.assertFalse(RN.is_plan_schedule(RN.make_schedule(num_updates=40, eval_interval=10, warmup=20, every=10)))
        with self.assertRaises(ValueError):
            RN.make_schedule(steps=12_345)


class WeightTest(unittest.TestCase):
    @processor_only
    def test_strong_weight_equals_world_ratio(self):
        """runner.strong_weight restates world.ratio(c, "strong") bit for bit (scoring environment, subprocess)."""
        d = ROOT / "weights"
        d.mkdir(parents=True, exist_ok=True)
        c = np.concatenate([[0.1, 1.1, 0.6], np.random.default_rng([20261016, 7779, 3]).uniform(0.1, 1.1, 5000)])
        H.write(d / "currents.npz", dict(c=c))
        out = subprocess.run([SCORING_PYTHON, "-m", "experiments.shift_hosts.tests.test_runner", "--role", "ratio",
                              "--dir", str(d)], cwd=str(RN.REPO), capture_output=True, text=True, timeout=300,
                             env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", JAX_PLATFORMS="cpu"))
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        theirs, _ = H.read(d / "ratio.npz")
        mine = RN.strong_weight(c)
        same = float(np.mean(mine == theirs["w"]))
        READOUTS["weights"] = dict(rows=int(len(c)), w_min=float(mine.min()), w_max=float(mine.max()),
                                   bit_identical_share=same, max_relative=_rel(mine, theirs["w"]))
        self.assertEqual(mine.tobytes(), theirs["w"].tobytes())


class BandReadoutTest(Setup):
    @processor_only
    def test_coverage_arithmetic(self):
        """coverage.py's rules, restated: ties covered, +inf width and in-ball rows refused for IW, groups."""
        from experiments.shift_hosts import band_readout as BR

        y = np.array([1.0, 2.0, 3.0, 0.0, 5.0, 1.0])
        q = np.array([1.0, 1.0, 1.0, 0.0, 5.0, 3.0])
        U = np.array([0.0, 1.0, 1.0, np.inf, 0.1, 1.0], np.float32)
        ball = np.array([False, False, False, False, True, False])
        self.assertEqual(BR.outcomes(y, q, U).tolist(), [1, 1, 0, -1, 1, 0])
        self.assertEqual(BR.outcomes(y, q, U, BR.refusals(ball, "iw")).tolist(), [1, 1, 0, -1, -1, 0])
        self.assertEqual(BR.refusals(ball, "standard").tolist(), [False] * 6)
        s = BR.summarize(BR.outcomes(y, q, U), episodes=[0, 0, 1, 1, 2, 2])
        self.assertEqual((s["steps"], s["coverage"], s["refused_share"], s["certified_steps"]), (6, 0.5, 1 / 6, 5))
        g = BR.groups(BR.outcomes(y, q, U), [0, 0, 1, 1, 2, 2], ball)
        self.assertEqual((g["inside"]["steps"], g["outside"]["steps"], g["all"]["steps"]), (1, 5, 6))
        w = BR.width(np.float32(2.0), 0.5, np.array([1e-9, 1.0], np.float32))
        self.assertEqual(w.tolist(), [np.float32(2.0) * (np.float32(1e-6) * np.float32(0.5)), 1.0])
        self.assertTrue(BR.in_ball(np.array([[0.0, 0, 0, 0], [1.0, 0, 0, 0]], np.float32),
                                   dict(centre=[0.0, 0, 0, 0], rho=1.0)).tolist() == [True, True])

    @processor_only
    def test_band_values_and_viewer(self):
        """y, q and U at the made-up strong-law rows are the learner's own, for every learner's IW run."""
        import jax
        import jax.numpy as jnp

        from experiments.shift_hosts import band_readout as BR
        from experiments.shift_hosts import iw as IW

        sched = schedule(50, 10, warmup=20, every=10)
        detail = {}
        for learner in RN.LEARNERS:
            rec, _ = execute(learner, "iw", sched, "test_b")
            L = BR.load_run(RN.run_dir(ROOT, "test_b", "made-up", 0, learner, "iw"), prepared(), guard.MADE_UP)
            self.assertEqual(L.step, 40)
            self.assertIsNotNone(L.table)
            self.assertEqual(np.float32(L.R_standard).tobytes(),
                             np.float32(rec["refreshes"][-1]["threshold"]).tobytes())
            rows = prepared().rows("test_strong")
            na = prepared().field("test_strong", "next_action")
            trans = BR.transition_of(L, rows, na)
            key = BR.coverage_key(0, "recorded")
            y, q, eta = BR.band_values(L, trans, key, guard.MADE_UP)
            M = L.setup.module
            noise = jax.random.fold_in(key, 0)
            y0, q0 = M.band_parts(L.setup.args, L.setup.built.apply_fns, L.carry[1], trans, noise)
            ref = L.carry[2].reference
            e0 = L.carry[2].scale.apply_fn(ref.cal_params, trans.obs, trans.action)
            for a, b, name in ((y, y0, "y"), (q, q0, "q"), (eta, e0, "eta")):
                self.assertLessEqual(_rel(a, np.asarray(b)), OUTSIDE, f"{learner} {name}")
            if learner != "iql":
                y1, _, _ = BR.band_values(L, trans, BR.coverage_key(0, "actor"), guard.MADE_UP)
                self.assertGreater(_rel(y1, y), 1e-4, f"{learner}: another key draws another target noise")
            masses = prepared().field("test_strong", "weight_strong")
            R, beyond = BR.thresholds_at(L, masses)
            want = IW.table_float32(L.table)[IW.lookup(L.table, masses)[1]]
            self.assertEqual((R.tobytes(), beyond), (want.tobytes(), 0))
            R2, beyond2 = BR.thresholds_at(L, np.array([1e6]))
            self.assertTrue(np.isinf(R2[0]) and beyond2 == 1)
            arrays, summary = BR.readout(L, rows, key=key, permission=guard.MADE_UP, masses=masses,
                                         ball_flags=prepared().field("test_strong", "in_ball"),
                                         episodes=prepared().field("test_strong", "episode"), next_action=na)
            U = np.asarray(BC_width(R, float(np.asarray(ref.residual_scale)), arrays["eta"]))
            self.assertLessEqual(_rel(arrays["iw/U"], U), OUTSIDE)
            self.assertEqual(arrays["iw/outcome"].shape, (1000,))
            states = np.concatenate([prepared().field("test_strong", "state").reshape(10, 100, 13),
                                     prepared().field("test_strong", "next_state").reshape(10, 100, 13)[:, -1:]], 1)
            commands = rows["action"].reshape(10, 100, 4)
            obs = states[:, :100].reshape(1000, 13).astype(np.float32)
            nxt = states[:, 1:].reshape(1000, 13).astype(np.float32)
            leaves = (obs, rows["action"], np.zeros(1000, np.float32), nxt, np.zeros(1000, np.float32))
            guard.register_arrays(leaves, guard.MADE_UP_WORLD)
            if learner == "rebrac":
                guard.register_arrays(leaves[:4] + (BR.actor_next_commands(L, nxt),) + leaves[4:], guard.MADE_UP_WORLD)
            view = BR.viewer(L, guard.MADE_UP, states, commands)
            self.assertEqual(sorted(view), ["U_iw", "U_standard", "q"])
            self.assertEqual(view["q"].shape, (10, 100))
            self.assertLessEqual(_rel(view["q"].reshape(-1), q), OUTSIDE)
            self.assertTrue(np.all(np.isfinite(view["U_standard"])) and np.all(view["U_standard"] > 0))
            detail[learner] = dict(y_bit_identical=float(np.mean(y == np.asarray(y0))),
                                   q_bit_identical=float(np.mean(q == np.asarray(q0))),
                                   coverage_standard=summary["standard"]["all"]["coverage"],
                                   coverage_iw_outside=summary["iw"]["outside"]["coverage"])
        READOUTS["band"] = detail


def BC_width(R, unit, eta):
    from experiments.shift_hosts import bca_core as BC

    return BC.width(np.asarray(R, np.float32), np.float32(unit), np.asarray(eta, np.float32))


# ------------------------------------------------------------------------------------------------ the scoring side

def role_ratio(d):
    """The scoring environment's half of WeightTest: world.ratio(c, "strong") on the given currents."""
    from experiments.shift_iw import world as W

    arr, _ = H.read(Path(d) / "currents.npz")
    H.write(Path(d) / "ratio.npz", dict(w=np.asarray(W.ratio(arr["c"], "strong"), np.float64)))
    return 0


def extra():
    out = dict(readouts=READOUTS)
    try:
        out["import_record"] = UI.record()
    except RuntimeError:
        pass
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--role", choices=("ratio",))
    p.add_argument("--dir")
    a, rest = p.parse_known_args()
    if a.role == "ratio":
        sys.exit(role_ratio(a.dir))
    UI.prepare()
    sys.exit(T.run(sys.modules[__name__], AREA, extra=extra))
