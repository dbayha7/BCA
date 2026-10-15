"""Stage 2: one seed's artifacts, behind one interface, for gate seeds 0-4 (read from the gate) and fresh seeds 5-9
(made here, under the Stage 2 pre-registration, with no BCA).

GateContext(rep), seeds 0-4: the room harness's SeedContext (the gate's verified loaders: acceptance.load_prereg,
load_seed, get_critic(train=False), the saved cloned actor, check 2's and check 3's saved actors), plus none(world)
(check 2's saved gapped-critic actor of the world), td3bc() (check 2's plain TD3+BC actor), full_none() (check 2's
full-log critic's actor) and host() (check 3's frozen co-trained plain TD3+BC run).

FreshContext(rep), seeds 5-9: the same interface on new logs, critics and clones made by the 'fresh_data' and
'fresh_learners' phases (run_stage2.py) with the gate's own unchanged code (experiments/shift: dataset.generate,
shift.define_gap / make_removal, learners.fixed_critic, clone, ActorOnly, cotrain), base world (stall slope 1.5), the
gate's recorder settings (runs/shift/pre/stage0_base.json), the gate's sizes. Outputs go to
runs/shift_explore/stage2/fresh/. These seeds were never generated under the base world (the gate's retune variant
generated seeds 5-9 at stall slope 3.0, a different world, and trained nothing on them). Nothing here is BCA.

A fresh seed that fails is recorded, never replaced: data.json with passed=False (its data checks), learners.json
with failed=True (a learner). recorded_failure(rep) reads both; FreshSeed and FreshContext refuse such a seed; the
detection marks count it as a failed seed, the policy marks as a stopped seed (policy.load_values), and the pooled
look accepts it in place of its deployment values.
"""

import math
import time
from pathlib import Path

import stage2_common as C

ALPHA = 5.0
FRESH_DIR = "fresh"


def _gate():
    import experiments.shift.acceptance as A
    import experiments.shift.learners as L
    import experiments.shift.dataset as DS
    import experiments.shift.shift as SH
    import experiments.shift.thruster_chain as TC
    import experiments.shift.recorder as RC
    return A, L, DS, SH, TC, RC


def gate_prereg():
    A = _gate()[0]
    prereg = A.load_prereg(A.refuse_protected(A.RUNS_DEFAULT))
    A.guard(prereg)
    return prereg


# --------------------------------------------------------------------------------------------- seeds 0-4

def gate_context(rep, worlds=("stall",)):
    H = C.load_room_harness()
    A, L = _gate()[:2]

    class GateContext(H.SeedContext):
        """The room harness's SeedContext (gate seeds 0-4) with the Stage 2 accessors."""

        role = "exploratory"

        def _h2_actor(self, key):
            if "_h2" not in self.__dict__:
                self._h2 = self.h2()
            acts, rec, _ = self._h2
            if rec["failed"].get(key) or key not in acts:
                raise SystemExit(f"refused: check 2's {key} actor of seed {self.rep} failed in the gate")
            return acts[key]

        def none(self, world="stall"):
            return self._h2_actor("gapped" if world == "stall" else "no_stall_gapped")

        def td3bc(self):
            return self._h2_actor("td3bc_gapped")

        def full_none(self):
            return self._h2_actor("full")

        def host(self, world="stall"):
            if world != "stall":
                raise A.GateError("the gate trained no TD3+BC host in the no-stall world")
            return H.SeedContext.host(self)

    if int(rep) not in C.EXPLORATORY_SEEDS:
        raise SystemExit(f"refused: seed {rep} is not a gate seed (0-4)")
    return GateContext(rep, worlds=worlds)


# --------------------------------------------------------------------------------------------- seeds 5-9

def fresh_dir(rep, world, *parts, paths=None):
    paths = paths or C.PATHS
    return paths.out / FRESH_DIR / world / f"seed{int(rep)}" / Path(*parts) if parts else \
        paths.out / FRESH_DIR / world / f"seed{int(rep)}"


def make_fresh_data(rep, world, prereg_stage2, log_fn=C.log):
    """acceptance.data_seed restated for a fresh seed (no pools, no maps: Stage 2 reads neither): the log, the gap,
    the removals, the leak checks and, in the stalling world, the world marks T1-T5 (required, as for gate seeds).
    Writes fresh/<world>/seed<k>/data.json. Not BCA."""
    A, L, DS, SH, TC, RC = _gate()
    if int(rep) not in C.CONFIRMATORY_SEEDS:
        raise SystemExit(f"refused: fresh data are made for seeds 5-9 only, not {rep}")
    out = fresh_dir(rep, world, "data.json")
    if out.exists():
        raise SystemExit(f"refused: {out} exists")
    prereg = gate_prereg()
    cfg = prereg.config
    t0 = time.perf_counter()
    spec = A.stage0_spec(prereg.runs, "base")
    slope = cfg.slope("base")
    physics = TC.world_physics(world, slope)
    lg = DS.generate(rep, world, spec, cfg.sizes, slope)
    gap, grec = SH.define_gap(lg)
    rec = dict(label=C.LABEL_LONG, kind="stage2-fresh-data", rep=int(rep), world=world, variant="base",
               sizes=list(cfg.sizes), stall_slope=slope, recorder=spec.as_dict(), physics=physics.as_dict(),
               log_sha256=lg.array_hashes(), bank_rows=int(len(lg.bank_rows)),
               gap_record_sha256=SH.GAP.canonical_sha256(grec), prereg_stage2_sha256=prereg_stage2.sha256,
               gate_prereg_sha256=prereg.digest)
    if gap is None:
        rec.update(defined=False, reason=grec.get("reason"), passed=False, seconds=time.perf_counter() - t0)
        C.write_json(out, rec)
        return rec
    rg, rr = SH.make_removal(lg, "gap", gap), SH.make_removal(lg, "random", gap)
    train, bank = lg.split_rows("train"), lg.bank_rows
    checks = dict(gap_rows_in_gapped_training=SH.leak_count(rg.kept_train_rows(lg), lg, gap),
                  gap_rows_in_gapped_bank=SH.leak_count(rg.bank_rows(lg), lg, gap),
                  random_same_count=bool(rr.count == rg.count and rr.bank_removed.sum() == rg.bank_removed.sum()),
                  leak_counter_finds_every_training_gap_row=bool(SH.leak_count(train, lg, gap) == rg.count),
                  leak_counter_finds_every_bank_gap_row=bool(SH.leak_count(bank, lg, gap) == int(rg.bank_removed.sum())),
                  evaluation_split_untouched=bool(not rg.removed[lg.split_rows("evaluation")].any()))
    passed = (checks["gap_rows_in_gapped_training"] == 0 and checks["gap_rows_in_gapped_bank"] == 0
              and all(v for k, v in checks.items() if isinstance(v, bool)))
    rec.update(defined=True, gap=gap.as_dict(), removals={k: r.summary(lg) for k, r in (("gap", rg), ("random", rr))},
               checks=checks)
    if world == "stall":
        card, marks = A.world_marks(lg, gap, spec, physics)
        rec.update(world_card=card, t1_t5=marks)
        passed = passed and all(marks.values())
    rec.update(passed=bool(passed), seconds=time.perf_counter() - t0)
    C.write_json(out, rec)
    log_fn(f"fresh data seed {rep} {world}: {'pass' if passed else 'FAIL'} ({rec['seconds']:.0f}s)")
    return rec


class FreshSeed:
    """SeedData-like view of a fresh seed: the regenerated log checked against its data.json, its gap and removals."""

    def __init__(self, rep, world, paths=None):
        A, L, DS, SH, TC, RC = _gate()
        rec = C.read_json(fresh_dir(rep, world, "data.json", paths=paths))
        if not rec.get("passed"):
            raise SystemExit(f"refused: fresh seed {rep} ({world}) failed its data checks; it is never replaced")
        spec = RC.RecorderSpec(**rec["recorder"])
        lg = DS.generate(rep, world, spec, tuple(rec["sizes"]), rec["stall_slope"])
        if lg.array_hashes() != rec["log_sha256"]:
            raise SystemExit(f"refused: fresh seed {rep} ({world}): the regenerated log differs from its record")
        g = rec["gap"]
        gap = SH.Gap(centre=tuple(g["centre"]), rho=g["rho"], mass=g["mass"], rule=g["rule"], q=g["q"],
                     competing_ratio=math.inf if g["competing_ratio"] is None else g["competing_ratio"],
                     record_sha256=g["record_sha256"])
        self.rep, self.world, self.variant, self.record = int(rep), world, "base", rec
        self.log, self.gap, self.spec, self.physics = lg, gap, spec, lg.physics
        self.removals = {k: SH.make_removal(lg, k, gap) for k in SH.REMOVALS}
        s = self.removals["gap"].summary(lg)
        if s["kept_rows_sha256"] != rec["removals"]["gap"]["kept_rows_sha256"]:
            raise SystemExit(f"refused: fresh seed {rep} ({world}): the gap removal differs from its record")
        self.arrays = {}
        self._prepared = {}

    def norm(self):
        L = _gate()[1]
        return L.normalization(self.log, self.removals["gap"])

    def prepared(self, kind, learn):
        L = _gate()[1]
        if kind not in self._prepared:
            self._prepared[kind] = L.prepare(self.log, self.removals[kind], self.norm(), learn)
        return self._prepared[kind]


FRESH_ARTIFACTS = {
    # name: (world, what, how)
    "critic_gap": ("stall", "critic", "learners.fixed_critic on the gapped log, stall-aware controller"),
    "critic_full": ("stall", "critic", "learners.fixed_critic on the full log, stall-aware controller (harm reference)"),
    "critic_gap_no_stall": ("no_stall", "critic", "learners.fixed_critic on the no-stall gapped log, stall-aware"),
    "clone": ("stall", "actor", "learners.clone on the gapped log"),
    "clone_no_stall": ("no_stall", "actor", "learners.clone on the no-stall gapped log"),
    "none": ("stall", "actor", "learners.ActorOnly (alpha 5, unweighted) from the clone against critic_gap"),
    "full_none": ("stall", "actor", "the same against critic_full, side by side (vmapped) with 'none'"),
    "none_no_stall": ("no_stall", "actor", "learners.ActorOnly (alpha 5) from clone_no_stall against critic_gap_no_stall"),
    "host": ("stall", "agent", "learners.cotrain, plain TD3+BC, alpha 5, gapped log (100,000 updates)"),
}


def make_fresh_learners(rep, prereg_stage2, log_fn=C.log, paths=None):
    """Every learner a fresh seed needs, with the gate's own code (no BCA). Writes fresh/<world>/seed<k>/learners.json
    and one msgpack per artifact (SHA-256 recorded).

    A FAILED LEARNER IS RECORDED, NOT RAISED (pre-registration: 'a fresh seed that fails its data checks or a learner
    fails every mark; never replaced'): when a critic, a cloned actor, an actor-only arm or plain TD3+BC fails
    (learners' own 'failed' flag), learners.json is written with failed=True and the reason, and nothing after the
    failure is trained. recorded_failure() then reports the seed as failed, every later phase refuses to use it, and
    the pooled looks count it as a stopped seed (it fails every mark)."""
    A, L, DS, SH, TC, RC = _gate()
    out = fresh_dir(rep, "stall", "learners.json", paths=paths)
    if out.exists():
        raise SystemExit(f"refused: {out} exists")
    prereg = gate_prereg()
    learn = prereg.config.learn
    t0 = time.perf_counter()
    st, tw = FreshSeed(rep, "stall", paths), FreshSeed(rep, "no_stall", paths)
    pg, pf, pt = st.prepared("gap", learn), st.prepared("none", learn), tw.prepared("gap", learn)
    rec = dict(label=C.LABEL_LONG, kind="stage2-fresh-learners", rep=int(rep), alpha=ALPHA,
               prereg_stage2_sha256=prereg_stage2.sha256, gate_prereg_sha256=prereg.digest, files={}, records={},
               failed=False)

    def save(name, tree, record):
        world = FRESH_ARTIFACTS[name][0]
        rec["files"][name] = dict(path=str(fresh_dir(rep, world, f"{name}.msgpack", paths=paths)),
                                  sha256=C.save_tree(fresh_dir(rep, world, f"{name}.msgpack", paths=paths), tree, paths))
        rec["records"][name] = {k: v for k, v in record.items() if k not in ("chunks",)}

    def fail(name, record):
        rec["records"][name] = {k: v for k, v in record.items() if k not in ("chunks",)}
        rec.update(failed=True, failed_learner=name,
                   reason=f"fresh learner {name} of seed {rep} failed; the seed fails every mark and is never replaced",
                   seconds=time.perf_counter() - t0)
        C.write_json(out, rec, paths)
        log_fn(f"fresh learners seed {rep}: FAILED at {name}")
        return rec

    crit = {}
    for name, prep in (("critic_gap", pg), ("critic_full", pf), ("critic_gap_no_stall", pt)):
        r, state = L.fixed_critic(prep, "stall_aware", learn, rep, log=log_fn)
        if r["failed"]:
            return fail(name, r)
        crit[name] = state.critic.params
        save(name, crit[name], r)
    clones = {}
    for name, prep in (("clone", pg), ("clone_no_stall", pt)):
        params, r = L.clone(prep, learn, rep, log=log_fn)
        if r.get("failed"):
            return fail(name, r)
        clones[name] = params
        save(name, params, r)
    ao = L.ActorOnly(pg, clones["clone"], learn, rep, alpha=ALPHA)
    stacked, info = ao.run([L.Arm("none", critic=0), L.Arm("full_none", critic=1)],
                           [crit["critic_gap"], crit["critic_full"]], weighted=False, log=log_fn)
    for i, name in enumerate(("none", "full_none")):
        if info["failed"][i]:
            return fail(name, dict(info, failed=True))
        save(name, ao.arm_params(stacked, i), dict(info, failed=info["failed"][i]))
    ao = L.ActorOnly(pt, clones["clone_no_stall"], learn, rep, alpha=ALPHA)
    stacked, info = ao.run([L.Arm("none_no_stall", critic=0)], [crit["critic_gap_no_stall"]], weighted=False,
                           log=log_fn)
    if info["failed"][0]:
        return fail("none_no_stall", dict(info, failed=True))
    save("none_no_stall", ao.arm_params(stacked, 0), dict(info, failed=info["failed"][0]))
    r, state, _ = L.cotrain(pg, learn, rep, alpha=ALPHA, log=log_fn, name=f"fresh TD3+BC seed {rep}")
    if r["failed"]:
        return fail("host", r)
    save("host", state, r)
    rec["seconds"] = time.perf_counter() - t0
    C.write_json(out, rec, paths)
    log_fn(f"fresh learners seed {rep}: done ({rec['seconds']:.0f}s)")
    return rec


def recorded_failure(rep, paths=None):
    """The reason a fresh seed (5-9) is failed on record, or None: its data checks failed in either world
    (data.json passed=False), or a learner failed (learners.json failed=True). Missing records are not failures (the
    phase has not run). Always None for gate seeds 0-4. Reads only Stage 2's own records (sidecar-verified)."""
    if int(rep) not in C.CONFIRMATORY_SEEDS:
        return None
    for world in ("stall", "no_stall"):
        d = C.read_json(fresh_dir(rep, world, "data.json", paths=paths), required=False)
        if d is not None and not d.get("passed"):
            return f"fresh seed {rep} failed its data checks in the {world} world (never replaced)"
    lr = C.read_json(fresh_dir(rep, "stall", "learners.json", paths=paths), required=False)
    if lr is not None and lr.get("failed"):
        return lr.get("reason") or f"a fresh learner of seed {rep} failed (never replaced)"
    return None


class FreshContext:
    """The GateContext interface for a fresh seed (5-9)."""

    role = "confirmatory"

    def __init__(self, rep, worlds=("stall",), paths=None):
        A, L = _gate()[:2]
        if int(rep) not in C.CONFIRMATORY_SEEDS:
            raise SystemExit(f"refused: seed {rep} is not a fresh seed (5-9)")
        self.prereg = gate_prereg()
        self.cfg, self.learn, self.variant, self.rep = self.prereg.config, self.prereg.config.learn, "base", int(rep)
        self.paths = paths
        self.sd = {w: FreshSeed(rep, w, paths) for w in worlds}
        self.rec = C.read_json(fresh_dir(rep, "stall", "learners.json", paths=paths))
        if self.rec.get("failed"):
            raise SystemExit(f"refused: {self.rec.get('reason')}")
        self._prepared = {}

    def _load(self, name, template):
        f = self.rec["files"][name]
        return C.load_tree(f["path"], template, f["sha256"])

    def prepared(self, world="stall"):
        if world not in self._prepared:
            if world not in self.sd:
                self.sd[world] = FreshSeed(self.rep, world, self.paths)
            self._prepared[world] = self.sd[world].prepared("gap", self.learn)
        return self._prepared[world]

    def critic(self, world="stall", removal="gap"):
        L = _gate()[1]
        name = {("stall", "gap"): "critic_gap", ("stall", "none"): "critic_full",
                ("no_stall", "gap"): "critic_gap_no_stall"}[(world, removal)]
        return self._load(name, L.critic_template()), self.rec["records"][name]

    def clone(self, world="stall"):
        L = _gate()[1]
        name = "clone" if world == "stall" else "clone_no_stall"
        return self._load(name, L.actor_template()), self.rec["records"][name]

    def none(self, world="stall"):
        L = _gate()[1]
        return self._load("none" if world == "stall" else "none_no_stall", L.actor_template())

    def full_none(self):
        L = _gate()[1]
        return self._load("full_none", L.actor_template())

    def host(self, world="stall"):
        L = _gate()[1]
        if world != "stall":
            raise SystemExit("refused: no fresh TD3+BC host in the no-stall world (host.py trains its own)")
        return self._load("host", L.agent_template(self.learn)), self.rec["records"]["host"]

    def td3bc(self):
        return self.host()[0].actor.params

    def anchors(self, world="stall"):
        A = _gate()[0]
        return A.anchors_of(self.prereg, self.variant, world)


def context(rep, worlds=("stall",), paths=None):
    """GateContext for seeds 0-4, FreshContext for seeds 5-9."""
    if int(rep) in C.EXPLORATORY_SEEDS:
        return gate_context(rep, worlds)
    if int(rep) in C.CONFIRMATORY_SEEDS:
        return FreshContext(rep, worlds, paths)
    raise SystemExit(f"refused: seed {rep} is neither a gate seed (0-4) nor a fresh seed (5-9)")
