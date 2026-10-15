"""Near-80 (version 8): stub mode. Every phase kind of the plan runs on a test fixture outside the repository, on
synthetic logs, with every learner replaced by a stand-in that takes ZERO gradient steps. Nothing is trained, no band
is fitted, no recorded seed is generated or loaded, and the gate's pre-registration is not read.

install(run) (called by phase_main in stub mode, before dispatch) replaces:
  settingdata.gate_seed, fresh_settings, fresh_log, fresh_gap, anchors, gate_record_sha256 -> synthetic seeds
      (synthetic_log: the pattern of test_near80.synthetic_log, tag 9000 + the logical seed, the world's physics) and
      synthetic_gap(); anchors from facts_near80.anchors_for(world, bank_n=--starts); settingdata.gate_prereg,
      acceptance.load_seed and dataset.generate -> functions that raise;
  learners.fixed_critic -> the critic's initial parameters (the same key as fixed_critic), updates 0;
  learners.clone -> the initial actor (learners.initial_actor);
  learners.ActorOnly.run -> every arm returns the start actor;
  machinery.RampRunner.run_ramp, the room's RoomRunner.run_kind and version 6's BCARunner.run_bca -> every arm returns
      the start actor (a ramp arm with a height above 0 returns the start actor plus 0.001 on every parameter, so that
      the strength-matching code path runs; this is not a gradient step); run_ramp also records each program's arm
      names (RAMP_BATCHES), so that the tests can check the batches are step 2's;
  version 6's policy.match_controls -> the first ladder rung, chosen without training; machinery.standin_matched and
      standin_locked -> raise 'stub: not run' (recorded as the stand-in's error); policy.standins_module -> None;
  v6bca.fit, v6bca.train_arm -> an untrained band (version 6's band_template, stub: true) and the start actor;
  the training programs themselves (the gate's _fixed_step, _clone_loop, _actor_only_loop, actor_step, cotrain,
      bca_cotrain, bca_band; the room's _room_loop and actor_step_w; step 2's _ramp_loop and actor_step_ramp; version
      6's _bca_loop, actor_step_bca and band.fit_band_arrays) -> guards that count the entry and raise.
Every stand-in counts its calls (CALLS) and every guard its entries (GUARD_HITS); phase_main writes both into the
attempt's usage record, and the end-to-end test asserts that no guard was entered.

Step 2's saved actors are replaced by make_step2_fixture(DIR): synthetic lock.json / actors.msgpack / values.json of
seeds 0-4 built from initial actors, and a synthetic step-2 results file equal to them; evaluate_step2_saved reads
these in stub mode.

The synthetic logs do not pass the no-training checks (they are not built to): in stub mode the rules are computed
and recorded, but they neither fail a seed nor halt the run ('stub_note' in the data records).
"""
import sys

sys.dont_write_bytecode = True

import collections  # noqa: E402
import types  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = Path("/home/user/bca")
for _p in (str(HERE), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import near80_constants as K  # noqa: E402
import ledger as LG  # noqa: E402

CALLS = collections.Counter()
GUARD_HITS = collections.Counter()
RAMP_BATCHES = []          # the arm names of every vmapped ramp program the phase ran (its batch), in call order
TAG_BASE = 9000
N_SURVIVORS = 60          # survivors near full throttle outside the ball: enough for BCA's posterior to certify
SURVIVOR_REWARD = -5.0    # the survivors' recorded reward: the second critic over-predicts them, the switch is on
STUB_STARTS = 5
PATTERN = np.array([-1.0, -0.137, -0.585, -0.406])
STUB_NOTE = ("stub mode: synthetic logs are not built to pass the no-training checks; the rules are recorded but fail "
             "no seed and halt nothing")
_INSTALLED = {}


class StubViolation(RuntimeError):
    """A training function was entered in stub mode."""


# --------------------------------------------------------------------------------------------- synthetic data

def synthetic_log(tag, world="stall", seed=7, sizes=(500, 50, 50), length=20):
    """A small synthetic log with the gate's Log interface (test_near80.synthetic_log's pattern, in the given world):
    normal driving capped below 80%, a full-throttle cluster (inside the ball), 60 survivors near full throttle
    outside the ball, and a few rows on and around the band's edges."""
    from experiments.shift import dataset as DS
    from experiments.shift import recorder as RC
    from experiments.shift import thruster_chain as TC
    rng = np.random.default_rng([int(seed), int(tag)])
    E = int(sum(sizes))
    n = E * length
    mu = np.array([-0.55, -0.08, -0.35, -0.25])
    a = 0.8 * np.tanh((mu + 0.45 * rng.standard_normal((n, 4))) / 0.8)
    ft = rng.uniform(size=n) < 0.15
    a[ft] = np.clip(PATTERN + 0.03 * rng.standard_normal((int(ft.sum()), 4)), -1, 1)
    train_rows = np.arange(sizes[0] * length)
    surv = rng.choice(train_rows[~ft[train_rows]], N_SURVIVORS, replace=False)
    a[surv] = np.array([-0.985, 0.4, 0.3, 0.3]) + 0.004 * rng.standard_normal((N_SURVIVORS, 4))
    a32 = a.astype(np.float32)
    rewards = np.zeros(n, np.float32)
    rewards[surv] = SURVIVOR_REWARD      # so that BCA's switch is on in the stub run and the matching path runs
    x = 0.3 * rng.standard_normal((n, TC.DX))
    x[:, TC.CURRENT] = rng.uniform(0.1, 0.7, n)
    x[:, TC.WAVE1] = rng.uniform(-0.2, 0.3, n)
    x2 = x + 0.01 * rng.standard_normal((n, TC.DX))
    split = np.repeat(np.arange(3, dtype=np.int8), np.asarray(sizes) * length)
    t = np.tile(np.arange(length, dtype=np.int32), E)
    arrays = {
        "observations": x.astype(np.float32), "actions": a32, "rewards": rewards,
        "next_observations": x2.astype(np.float32), "terminals": np.zeros(n, bool), "timeouts": t == length - 1,
        "infos/state": x.astype(np.float32).astype(np.float64), "infos/next_state": x2,
        "infos/episode": np.repeat(np.arange(E, dtype=np.int32), length), "infos/t": t, "infos/split": split,
        "infos/full_throttle_command": ft,
    }
    first = sizes[0]
    offs = np.random.default_rng([int(seed) + 1, int(tag)]).integers(0, length, (sizes[1], 5))
    bank = np.sort(np.unique(np.concatenate([(first + e) * length + np.unique(o) for e, o in enumerate(offs)])))
    meta = dict(rep=int(tag), world=world, bank_rows=bank.tolist(), sizes=list(sizes), length=length,
                physics=TC.world_physics(world).as_dict(), recorder=RC.DEFAULT_SPEC.as_dict(), stub=True)
    return DS.Log(arrays, meta)


def synthetic_gap():
    from experiments.shift import shift as SH
    return SH.Gap(centre=tuple(float(v) for v in PATTERN), rho=0.3, mass=0.15, rule="base", q=0.15,
                  competing_ratio=2.0, record_sha256="synthetic")


def synthetic_seed(seed, world):
    """A SeedData-like synthetic seed with a gate-like data record (logical seed -> tag 9000 + seed)."""
    import settingdata as SD
    from experiments.shift import acceptance as A
    from experiments.shift import shift as SH
    K.group_of(seed)
    lg = synthetic_log(TAG_BASE + int(seed), world)
    gap = synthetic_gap()
    rg, rr = SH.make_removal(lg, "gap", gap), SH.make_removal(lg, "random", gap)
    checks, ok = SD._ball_checks(lg, gap, rg, rr)
    record = dict(log_sha256=lg.array_hashes(), gap=gap.as_dict(), removals={k: r.summary(lg) for k, r in
                                                                           (("gap", rg), ("random", rr))},
                  checks=checks, passed=True, stub=True)
    if world == "stall":
        _, marks = A.world_marks(lg, gap, lg.spec, lg.physics)
        record["t1_t5"] = {k: True for k in marks}
        record["t1_t5_on_synthetic_log"] = {k: bool(v) for k, v in marks.items()}
    return SD.SeedView(seed, world, record, lg, gap, lg.spec, lg.physics)


def stub_anchors(run, world):
    import facts_near80 as F
    return {s: dict(recording=r, top=t) for s, (r, t) in
            F.anchors_for(world, bank_n=run.starts or STUB_STARTS).items()}


# --------------------------------------------------------------------------------------------- stand-ins

def _counted(name, fn):
    def wrapper(*a, **kw):
        CALLS[name] += 1
        return fn(*a, **kw)
    wrapper.__name__ = f"stub_{name}"
    wrapper.__stub__ = name
    return wrapper


def _guard(name):
    def guard(*a, **kw):
        GUARD_HITS[name] += 1
        raise StubViolation(f"stub mode: {name} was entered; it would take a gradient step")
    guard.__stub_guard__ = name
    return guard


def _raiser(what):
    def f(*a, **kw):
        raise StubViolation(f"stub mode: {what} is not used")
    return f


def _initial_critic(rep, settings=None):
    """The critic's initial parameters of learners.fixed_critic for data seed rep (no update)."""
    import jax
    import jax.numpy as jnp
    from experiments.shift import learners as L
    args = L.make_args(settings or L.LearnSettings(), rep)
    rng = jax.random.fold_in(jax.random.PRNGKey(args.seed), L.KEY_TD3_INIT)
    rng, _, kc = jax.random.split(rng, 3)
    return L.CRITIC.init(kc, jnp.zeros(L.DX), jnp.zeros(L.DA))


def stub_fixed_critic(prepared, controller, settings, rep, *, placebo=False, updates=None, out_dir=None, log=None):
    from experiments.shift import learners as L
    params = _initial_critic(rep, settings)
    record = dict(kind="shift-fixed-critic", controller=controller, rep=int(rep), updates=0, final_update=0,
                  initial_sha256=L.tree_sha256(params), critic_sha256=L.tree_sha256(params), failed=False, stub=True,
                  data=prepared.describe())
    if out_dir is not None:
        record["file_sha256"] = L.save_tree(Path(out_dir) / "critic.msgpack", params)
    return record, types.SimpleNamespace(critic=types.SimpleNamespace(params=params))


def stub_clone(prepared, settings, rep, steps=None, *, chunk=5_000, log=None):
    from experiments.shift import learners as L
    params = L.initial_actor(settings, rep)
    return params, dict(kind="shift-clone", rep=int(rep), steps=0, losses=[], failed=False, stub=True,
                        actor_sha256=L.tree_sha256(params))


def _stacked(params, n):
    import jax
    import jax.numpy as jnp
    return jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (n,) + jnp.shape(x)), params)


def stub_actor_only_run(self, arms, critics, *, weighted=True, log=None):
    from experiments.shift import learners as L
    n = len(arms)
    return _stacked(self.start_params, n), dict(history=[], chunk_metrics=[], metric_names=[], oracle_calls=0,
                                                rows_sha256=self.rows_sha256,
                                                start_sha256=L.tree_sha256(self.start_params), failed=[False] * n,
                                                stub=True)


def _offset(params, eps):
    import jax
    return jax.tree_util.tree_map(lambda x: x + np.asarray(eps, np.asarray(x).dtype), params)


def _arm_out(params, arm):
    return dict(params=params, finite=True, chunk_metrics=[], arm=arm.record(), seconds=0.0)


def stub_run_kind(self, kind, arms, log_fn=None):
    self.seconds[kind] = self.seconds.get(kind, 0.0)
    return {a.name: _arm_out(self.start_params, a) for a in arms}


def stub_run_ramp(self, arms, log_fn=None):
    self.seconds["ramp"] = self.seconds.get("ramp", 0.0)
    RAMP_BATCHES.append([a.name for a in arms])
    out = {}
    for a in arms:
        h = float(a.coef[0]) * float(a.coef[1]) if len(a.coef) == 2 else 0.0
        out[a.name] = _arm_out(_offset(self.start_params, 1e-3) if h > 0 else self.start_params, a)
    return out


def stub_run_bca(self, kind, arms, log_fn=None):
    self.seconds[kind] = self.seconds.get(kind, 0.0)
    return {a.name: dict(_arm_out(self.start_params, a), refits=None) for a in arms}


def stub_match_controls(runner, trained, none, prepared, target, scale, families, log_fn=None):
    out = {}
    for f in families:
        rungs = sorted(((v["arm"].get("knob"), k) for k, v in trained.items() if v["arm"].get("family") == f),
                       key=lambda x: (x[0] is None, x[0]))
        name = rungs[0][1] if rungs else None
        out[f] = dict(record=dict(stub=True, knob=None if not rungs else rungs[0][0],
                                  how="stub: the first ladder rung, chosen without training"),
                      arm=name, matched=name is not None)
    return out


def stub_standin(*a, **kw):
    raise RuntimeError("stub: not run")


def stub_v6_fit(prep, critic, seed, world, learn, steps=None, log=None):
    import v6bca
    band = v6bca.band_template()
    rec = dict(kind="near80-v6-band", seed=int(seed), world=world, critic="fixed", weighted=False, steps=0,
               stub=True, failed=False, uniform=dict(R=1.0, R_c=1.0, U_max=1.0, unit=1.0, certified=True,
                                                     bank_rows=int(len(prep.bank_rows))))
    return band, rec, dict(scores=np.zeros(len(prep.bank_rows)), eta=np.ones(len(prep.bank_rows)))


def stub_v6_train_arm(pg, clone, critic, learn, seed, band, cfg, shared, steps, chunk, log=None):
    import v6bca
    arm = types.SimpleNamespace(record=lambda: dict(name=v6bca.V6_ARM, kind="bca_u", coef=list(v6bca.V6_COEF)))
    return _arm_out(clone, arm), None


# --------------------------------------------------------------------------------------------- install

LEARNER_GUARDS = ("_fixed_step", "_clone_loop", "_actor_only_loop", "actor_step", "cotrain", "bca_cotrain",
                  "bca_band", "bca_train_step", "bca_update")
NEEDS_STEP2 = ("policy", "v6_band", "readouts", "evaluate", "lock")


def install(run):
    """Replace every learner entry point and data loader for one stub phase process (module docstring)."""
    if _INSTALLED.get("done"):
        return
    import settingdata as SD
    from experiments.shift import acceptance as A
    from experiments.shift import dataset as DS
    from experiments.shift import learners as L
    SD.gate_seed = _counted("gate_seed", synthetic_seed)
    SD.fresh_settings = _counted("fresh_settings", lambda: (_recorder_spec(), (500, 50, 50), 1.5))
    SD.fresh_log = _counted("fresh_log", lambda seed, world, spec, sizes, slope:
                            synthetic_log(TAG_BASE + int(seed), world))
    SD.fresh_gap = _counted("fresh_gap", lambda lg: (synthetic_gap(), dict(reason="stub", stub=True)))
    SD.anchors = _counted("anchors", stub_anchors)
    SD.gate_record_sha256 = lambda *a, **kw: None
    SD.gate_prereg = _raiser("the gate's pre-registration")
    SD.gate_critic = _counted("gate_critic", lambda sd, removal: _initial_critic(sd.rep))
    SD.gate_clone = _counted("gate_clone", lambda seed, world: L.initial_actor(L.LearnSettings(), seed))
    SD.gate_check2_actor = _counted("gate_check2_actor", lambda seed, name: L.initial_actor(L.LearnSettings(), seed))
    A.load_seed = _raiser("acceptance.load_seed")
    DS.generate = _raiser("dataset.generate")
    L.fixed_critic = _counted("fixed_critic", stub_fixed_critic)
    L.clone = _counted("clone", stub_clone)
    L.ActorOnly.run = _counted("actor_only_run", stub_actor_only_run)
    for name in LEARNER_GUARDS:
        if hasattr(L, name):
            setattr(L, name, _guard(f"learners.{name}"))
    if run.key.kind in NEEDS_STEP2 or run.key.kind == "lock":
        import policy_phase as PP
        S, MC, _ = PP.load_step2(run)
        MC.RampRunner.run_ramp = _counted("run_ramp", stub_run_ramp)
        MC.H.RoomRunner.run_kind = _counted("run_kind", stub_run_kind)
        MC.P.BCARunner.run_bca = _counted("run_bca", stub_run_bca)
        MC.P.match_controls = _counted("match_controls", stub_match_controls)
        MC.standin_matched = _counted("standin_matched", stub_standin)
        MC.standin_locked = _counted("standin_locked", stub_standin)
        MC.P.standins_module = _counted("standins_module", lambda prereg: None)
        MC.best_standin = _counted("best_standin", lambda: "stub")
        for mod, names in ((MC, ("_ramp_loop", "actor_step_ramp")), (MC.H, ("_room_loop", "actor_step_w")),
                           (MC.P, ("_bca_loop", "actor_step_bca")), (MC.B, ("fit_band_arrays",))):
            for name in names:
                setattr(mod, name, _guard(f"{getattr(mod, '__name__', 'module')}.{name}"))
        import v6bca
        v6bca.fit = _counted("v6bca_fit", stub_v6_fit)
        v6bca.train_arm = _counted("v6bca_train_arm", stub_v6_train_arm)
    _INSTALLED["done"] = True


def _recorder_spec():
    from experiments.shift import recorder as RC
    return RC.DEFAULT_SPEC


def report():
    """What phase_main writes into the attempt's usage record in stub mode."""
    return dict(stub_calls=dict(CALLS), stub_guard_hits=dict(GUARD_HITS), stub_ramp_batches=list(RAMP_BATCHES))


# --------------------------------------------------------------------------------------------- step 2's fixture

def step2_root(fixture):
    return Path(fixture) / "step2"


def step2_results(fixture):
    return step2_root(fixture) / "results.json"


STEP2_STALL_IDS = ("none", "clone", "full_none", "perfect_band", "range_hinge") + K.STEP2_RAMP_ARMS + \
    ("even_pull", "push_less", "random_1", "random_2", "standin")


def step2_names(world):
    """Step 2's logical ids -> actor names of a synthetic lock (the stand-in unmatched, as a failed rival)."""
    ids = {}
    for logical in STEP2_STALL_IDS:
        if logical == "full_none" and world != "stall":
            continue
        if logical == "standin":
            ids[logical] = None
        elif logical in K.MATCHED:
            ids[logical] = f"{logical}@0.1" if world == "stall" else f"{logical}_locked"
        else:
            ids[logical] = logical
    return ids


def make_step2_fixture(fixture, starts=STUB_STARTS):
    """Synthetic step-2 saved files for seeds 0-4 (both worlds) under DIR/step2/ and the matching results file:
    every actor is the initial actor of its seed; the values are its evaluation on the synthetic main-setting data,
    computed exactly as evaluate_phase computes them, so the step-2 checks pass on the fixture."""
    import settingdata as SD
    import policy_phase as PP
    import phases as PH
    from experiments.shift import acceptance as A
    from experiments.shift import learners as L
    from experiments.shift import thruster_chain as TC
    from flax import serialization
    fixture = Path(fixture)
    root = step2_root(fixture)
    if step2_results(fixture).exists():
        return LG.read_json_verified(step2_results(fixture))
    run = PH.Run(PH.make_key("evaluate", "main", "stall", 0), fixture / "out", mode="stub", fixture=fixture,
                 starts=starts)
    S, MC, s2 = PP.load_step2(run)
    learn = L.LearnSettings()
    cfg = types.SimpleNamespace(bank_size=int(starts))
    points = {w: {} for w in K.WORLDS}
    for s in K.EXPLORATORY:
        for w in K.WORLDS:
            sd = synthetic_seed(s, w)
            prep = SD.prepared("main", sd, learn)
            actor = L.initial_actor(learn, s)
            ev = A._evaluate(actor, prep, stub_anchors(run, w), cfg)
            ids = step2_names(w)
            names = sorted({v for v in ids.values() if v} | {K.W0})
            d = root / "policy" / f"seed{s}" / w
            blob = serialization.to_bytes({n: actor for n in names})
            file_sha = LG.write_bytes_once(d / "actors.msgpack", blob)
            lock = dict(kind="step2-policy-lock", mode="official", prespec_sha256=s2["prespec_sha256"],
                        code_manifest_sha256=s2["code_manifest_sha256"], seed=s, world=w, steps=K.ACTOR_STEPS,
                        chunk=K.CHUNK, full_length=True, checks=dict(b_w0=dict(passed=True)), cpus=[0],
                        names=names, actor_sha256={n: L.tree_sha256(actor) for n in names},
                        actors_file="actors.msgpack", actors_file_sha256=file_sha, arm_ids=ids,
                        scored_names=sorted({v for v in ids.values() if v}),
                        finite={n: True for n in names}, stub=True)
            lock_sha = LG.write_json_once(d / "lock.json", lock)
            sits = {sit: dict(points={n: ev[sit]["points"] for n in lock["scored_names"]}) for sit in TC.SITUATIONS}
            LG.write_json_once(d / "values.json", dict(kind="step2-policy-values", mode="official", seed=s, world=w,
                                                       bank_size=int(starts),
                                                       lock_sha256=LG.file_sha256(d / "lock.json"),
                                                       arm_ids=ids, situations=sits, stub=True,
                                                       lock_text_sha256=lock_sha))
            for logical, name in ids.items():
                row = points[w].setdefault(logical, {sit: dict(per_seed=[]) for sit in TC.SITUATIONS})
                for sit in TC.SITUATIONS:
                    row[sit]["per_seed"].append(None if name is None else sits[sit]["points"][name])
    res = dict(kind="step2-results-stub", mode="official", seeds=list(K.EXPLORATORY), points=points, stub=True)
    LG.write_json_once(step2_results(fixture), res)
    return res


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="stub mode helpers (test fixtures only)")
    ap.add_argument("cmd", choices=("step2-fixture",))
    ap.add_argument("fixture")
    ap.add_argument("--starts", type=int, default=STUB_STARTS)
    a = ap.parse_args(argv)
    fx = Path(a.fixture).resolve()
    if LG.inside(fx, ROOT):
        print("refused: a test fixture must lie outside /home/user/bca", file=sys.stderr)
        return K.EXIT_REFUSED
    if not (fx / K.TEST_MARKER).is_file():
        print(f"refused: {fx} is not a near-80 test fixture (no {K.TEST_MARKER})", file=sys.stderr)
        return K.EXIT_REFUSED
    make_step2_fixture(fx, a.starts)
    print(f"step 2 fixture written under {step2_root(fx)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
