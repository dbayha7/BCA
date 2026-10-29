"""Read-only frozen-model action-only score engine. No fitting or experiment runner.

Call configure_runtime() before importing NumPy/JAX in a fresh worker. Callers own
all calibration/evaluation random streams and must supply independent label noise.
"""
from dataclasses import dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import types

RUN_NAME = "core_20261020T2250Z"
JOB = "jobs/replicate_01/full/bca"
FROZEN_HEAD = "b007052a2494dc651ac00469fa9e922b287cd1e5"
CONTEXT_ORDER = ("stall_low", "stall_high", "monotone_low", "monotone_high")


def configure_runtime(cpu_indices=(8, 9, 10, 11)):
    """Bound this process only; do not modify or stop other processes."""
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "JAX_NUM_THREADS", "TF_NUM_INTRAOP_THREADS",
                "TF_NUM_INTEROP_THREADS"):
        os.environ[key] = "1"
    os.environ.update(JAX_PLATFORMS="cpu", JAX_PLATFORM_NAME="cpu", CUDA_VISIBLE_DEVICES="")
    sys.dont_write_bytecode = True
    mask = sum(1 << int(i) for i in cpu_indices)
    if os.name == "nt":
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        handle = kernel.GetCurrentProcess()
        kernel.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_uint)
        kernel.SetProcessAffinityMask.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
        if not kernel.SetPriorityClass(handle, 0x40) or not kernel.SetProcessAffinityMask(handle, mask):
            raise OSError(ctypes.get_last_error(), "Could not set bounded IDLE execution")
    return dict(cpu_indices=list(cpu_indices), affinity_mask=mask, priority="IDLE",
                threads=1, platform="cpu")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def create_snapshot(*, bundle_root, canonical_run_dir, destination):
    """Copy verified evaluation inputs into this experiment; never repair canonical files.

    The copied MANIFEST is the original release inventory, not a claim that its
    unrelated report members are all copied. SNAPSHOT.json lists this subset.
    """
    bundle, run, dest = Path(bundle_root).resolve(), Path(canonical_run_dir).resolve(), Path(destination).resolve()
    if not dest.is_relative_to(Path(__file__).resolve().parent):
        raise ValueError("snapshot must be within the action-only experiment")
    if dest.exists():
        raise FileExistsError("snapshot already exists; reuse it after verifying SNAPSHOT.json")
    inventory=_json(bundle/"MANIFEST.json")["files"]
    evidence=bundle/"evidence"/RUN_NAME
    rels=[r for r in inventory if r.startswith("source/") and r.endswith(".py")]
    rels += [f"evidence/{RUN_NAME}/{p}" for p in
             (JOB+"/manifest.json","data/seed_1101/full/dataset.json","provenance.json","config.json")]
    copies=[]
    for relative in rels:
        src=bundle/relative
        if sha256(src)!=inventory[relative]["sha256"]:
            raise ValueError("snapshot source verification failed: "+relative)
        copies.append((src,dest/relative))
    record=_json(evidence/JOB/"manifest.json")
    checkpoint=run/JOB/record["checkpoint"]["file"]
    if sha256(checkpoint)!=record["checkpoint"]["sha256"]:
        raise ValueError("snapshot checkpoint verification failed")
    copies += [(bundle/"MANIFEST.json",dest/"MANIFEST.json")]
    for relative in (JOB+"/"+record["checkpoint"]["file"],JOB+"/normalization.npz","audit/seed_1101/source_audit.npz"):
        copies.append((run/relative,dest/"run"/relative))
    # All destination operations are constrained to the new task-local subtree.
    dest.mkdir(parents=True)
    files={}
    for src,target in copies:
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src,target)
        digest=sha256(target)
        if digest!=sha256(src):
            raise IOError("snapshot copy differs: "+str(src))
        files[target.relative_to(dest).as_posix()]=dict(source=str(src),sha256=digest,bytes=target.stat().st_size)
    snapshot=dict(schema="frozen-action-only-input-subset-v1",files=files,
                  scope="verified release source and metadata subset plus original checkpoint/normalization/source audit bytes",
                  original_bundle=str(bundle),original_run=str(run),canonical_mutations=0)
    (dest/"SNAPSHOT.json").write_text(json.dumps(snapshot,indent=2)+"\n",encoding="utf-8")
    return dict(bundle_root=str(dest),canonical_run_dir=str(dest/"run"),snapshot_sha256=sha256(dest/"SNAPSHOT.json"))


def action_grid(atoms):
    """Lexicographic Cartesian grid: coordinate 3 varies fastest."""
    import numpy as np
    atoms = np.asarray(atoms, np.float64)
    if atoms.shape != (5,) or not np.all(np.diff(atoms) > 0):
        raise ValueError("five strictly increasing action atoms required")
    indices = np.indices((5, 5, 5, 5)).reshape(4, -1).T
    return atoms[indices], indices


def project_actions(actions, atoms):
    """Nearest Euclidean grid point, ties to lower coordinate/lexicographic index."""
    import numpy as np
    actions = np.asarray(actions, np.float64)
    if actions.shape[-1:] != (4,) or not np.isfinite(actions).all():
        raise ValueError("finite [...,4] actions required")
    idx = np.argmin(np.abs(actions[..., None] - np.asarray(atoms)), axis=-1)
    flat = np.sum(idx * np.array([125, 25, 5, 1]), axis=-1)
    return flat.astype(np.int64)


def select_states(rows, seed, *, per_context=16, times=(0, 50, 99)):
    """Outcome-blind episode sampling, then fixed time cycle within each context."""
    import numpy as np
    if per_context != 16 or tuple(times) != (0, 50, 99):
        raise ValueError("this engine requires 16 episodes/context and times(0,50,99)")
    first = np.flatnonzero(np.asarray(rows["time"]) == 0)
    chosen = []
    for context in range(4):
        episodes = np.sort(np.asarray(rows["episode_id"])[first[np.asarray(rows["context"])[first] == context]])
        if len(episodes) < per_context or len(np.unique(episodes)) != len(episodes):
            raise ValueError("insufficient unique source episodes")
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 2026102101, context]))
        for j, episode in enumerate(rng.permutation(episodes)[:per_context]):
            matches = np.flatnonzero((rows["episode_id"] == episode) & (rows["time"] == times[j % 3]))
            if len(matches) != 1:
                raise ValueError("selected source transition is absent or duplicated")
            chosen.append(int(matches[0]))
    selected = np.asarray(chosen, np.int64)
    if len(np.unique(rows["episode_id"][selected])) != 64:
        raise ValueError("state pool must use64 distinct episodes")
    return {key: np.asarray(rows[key])[selected].copy() for key in
            ("row_id", "episode_id", "time", "observation", "state", "context", "regime", "action",
             "reward", "next_observation", "action_probability", "action_atom_index")}


def _load_source_closure(bundle):
    """Use the preserved source closure under canonical module names, never current checkout code."""
    bundle = Path(bundle).resolve()
    source = bundle / "source"
    inventory = _json(bundle / "MANIFEST.json")["files"]
    verified = {}
    for relative, item in inventory.items():
        if relative.startswith("source/") and relative.endswith(".py"):
            actual = sha256(bundle / relative)
            if actual != item["sha256"]:
                raise ValueError("frozen source differs from release manifest: " + relative)
            verified[relative.removeprefix("source/")] = actual
    for package in ("algorithms", "calibration", "runtime", "experiments"):
        existing = sys.modules.get(package)
        if existing is not None:
            locations = list(getattr(existing, "__path__", []))
            if not locations or any(not Path(p).resolve().is_relative_to(source) for p in locations):
                raise RuntimeError("fresh worker required; non-frozen package already imported: " + package)
    sys.path.insert(0, str(source))
    # The release has an experiments namespace without a root __init__.py.
    if "experiments" not in sys.modules:
        namespace = types.ModuleType("experiments")
        namespace.__path__ = [str(source / "experiments")]
        namespace.__package__ = "experiments"
        sys.modules["experiments"] = namespace
    training = importlib.import_module("experiments.thrusterchain_diagnostics.training")
    data = importlib.import_module("experiments.thrusterchain_diagnostics.training_data")
    plant = importlib.import_module("experiments.thrusterchain_diagnostics.continuous")
    for module in (training, data, plant):
        if not Path(module.__file__).resolve().is_relative_to(source):
            raise RuntimeError("module escaped frozen source closure")
    return training, data, plant, verified


@dataclass(frozen=True)
class ScoreBatch:
    scores: object
    residual: object
    absolute_error: object
    scale: object
    target: object
    q: object
    eta: object


class FrozenScoreEngine:
    """Final100k native critic/target networks, frozen95k calibrator/unit.

    ``score`` consumes supplied standard-normal target label noise, never draws it.
    ``scale`` is max(eta,1e-6)*unit, so width=threshold*scale has residual units.
    No calibrator, actor, critic, optimizer, reference or counter is updated.
    """

    @classmethod
    def load(cls, *, bundle_root, canonical_run_dir, state_seed, repository_root=None):
        import numpy as np
        self = cls()
        bundle, run = Path(bundle_root).resolve(), Path(canonical_run_dir).resolve()
        evidence = bundle / "evidence" / RUN_NAME
        inventory = _json(bundle / "MANIFEST.json")["files"]
        paths = [evidence / JOB / "manifest.json", evidence / "data/seed_1101/full/dataset.json",
                 evidence / "provenance.json", evidence / "config.json"]
        for p in paths:
            relative = p.relative_to(bundle).as_posix()
            if relative not in inventory or sha256(p) != inventory[relative]["sha256"]:
                raise ValueError("preserved metadata fingerprint differs: " + relative)
        record, data_record, provenance, config = map(_json, paths)
        if (record["status"] != "completed" or record["final_update"] != 100000
                or record["checkpoint"]["update"] != 100000 or record["identity"]["arm"] != "bca"
                or record["identity"]["seed"] != 2101):
            raise ValueError("requires replicate01/full/BCA completed100k checkpoint")
        if data_record["seed"] != 1101 or data_record["retention"] != "full":
            raise ValueError("requires full source seed1101")
        good_refreshes = [r for r in record["refreshes"] if r["posterior_inputs_valid"]]
        if not good_refreshes or good_refreshes[-1]["update"] != 95000:
            raise ValueError("expected final frozen reference refreshed at95000")
        T, D, C, closure_hashes = _load_source_closure(bundle)
        if T._source_hashes() != record["identity"]["source_sha256"]:
            raise ValueError("loaded native source does not match trained identity")
        for p in (T, D, C):
            rel = Path(p.__file__).relative_to(bundle / "source").as_posix()
            expected = provenance["source_sha256"].get(rel)
            if expected and closure_hashes[rel] != expected:
                raise ValueError("suite source differs from original training provenance")
        settings = T.TrainingSettings(**record["identity"]["settings"])
        if settings.horizon != 100 or settings.gamma != 1:
            raise ValueError("requires finite H100 gamma1 native objective")
        template, models, args, native_config = T.initialize(settings, "bca", 2101)
        job_dir = run / JOB
        carry = T._restore(job_dir, template, record)
        normpath = job_dir / "normalization.npz"
        with np.load(normpath, allow_pickle=False) as archive:
            mean, std = archive["mean"].copy(), archive["std"].copy()
        if T._hash_arrays(dict(mean=mean, std=std)) != record["identity"]["data"]["normalization_sha256"]:
            raise ValueError("normalizer fingerprint differs")
        result = T.TrainingResult(carry, models, mean, std, record)
        if T.tree_sha256(result.state.native.actor.params) != record["checkpoint"]["actor_sha256"]:
            raise ValueError("restored actor fingerprint differs")
        if not bool(result.state.posterior.ready):
            raise ValueError("frozen posterior is not ready")
        auditpath = run / "audit/seed_1101/source_audit.npz"
        with np.load(auditpath, allow_pickle=False) as archive:
            rows = {key: archive[key] for key in archive.files}
        if D.array_hash(rows) != data_record["source_rows_sha256"]:
            raise ValueError("source audit array fingerprint differs")
        self.T, self.D, self.C, self.result = T, D, C, result
        self.args, self.settings, self.native_config = args, settings, native_config
        self.selected = select_states(rows, int(state_seed))
        self.atoms = D.ACTION_ATOMS.copy()
        self.grid, self.grid_indices = action_grid(self.atoms)
        _, coordinate_mass = D.behavior_parameters(self.selected["observation"], C.Objective())
        self.beta = np.prod(coordinate_mass[:, np.arange(4)[None, :], self.grid_indices], axis=-1)
        if self.beta.shape != (64, 625) or not np.allclose(self.beta.sum(1), 1, atol=2e-14, rtol=0):
            raise ValueError("categorical law is not normalized")
        if not np.all(self.beta > 0):
            raise ValueError("full data behavior must have positive625-atom support")
        self.actor_actions = np.clip(result.policy(self.selected["observation"]), -1, 1).astype(np.float32)
        self.actor_action_index = project_actions(self.actor_actions, self.atoms)
        self.unit = float(np.asarray(result.state.posterior.residual_scale))
        self.reference_update = 95000
        self._catalog = None
        self._make_programs()
        head = None
        if repository_root is not None:
            head = subprocess.check_output(["git", "-C", str(repository_root), "rev-parse", "HEAD"], text=True).strip()
        self.metadata = dict(schema="frozen-action-only-engine-v1", run=RUN_NAME, job=JOB,
            frozen_head=FROZEN_HEAD, current_checkout_head=head,
            source_closure=str(bundle / "source"), source_sha256=closure_hashes,
            bundle_manifest_sha256=sha256(bundle / "MANIFEST.json"),
            metadata_files={str(p): sha256(p) for p in paths},
            canonical_metadata_missing=not (job_dir / "manifest.json").exists(),
            metadata_recovery="preserved release metadata only; canonical files never restored or overwritten",
            checkpoint=dict(path=str(job_dir / record["checkpoint"]["file"]), **record["checkpoint"]),
            normalization=dict(path=str(normpath), file_sha256=sha256(normpath), arrays_sha256=record["identity"]["data"]["normalization_sha256"]),
            source_audit=dict(path=str(auditpath), file_sha256=sha256(auditpath), arrays_sha256=data_record["source_rows_sha256"]),
            frozen_reference=dict(update=95000, unit=self.unit, threshold=float(result.state.posterior.threshold),
                                  cal_params_sha256=T.tree_sha256(result.state.posterior.cal_params)),
            model_hashes={name:T.tree_sha256(getattr(result.state.native,name).params) for name in ("actor","actor_target","critic","critic_target")},
            selection=dict(seed=int(state_seed), states=64, distinct_episodes=64, per_context=16,
                           times=[0,50,99], counts_per_context=[6,5,5], outcome_selection=False,
                           rule="uniform seeded episode permutation within context; fixed time cycle by selection order",
                           selected_arrays_sha256=D.array_hash(self.selected),
                           row_ids=self.selected["row_id"].tolist(), episode_ids=self.selected["episode_id"].tolist(),
                           times_by_state=self.selected["time"].tolist(),contexts=self.selected["context"].tolist(),
                           regimes=self.selected["regime"].tolist(),loads=self.selected["state"][:,10].tolist(),
                           context_order=list(CONTEXT_ORDER)),
            action_law=dict(atoms=self.atoms.tolist(), grid_size=625, grid_order="lexicographic; final coordinate fastest",
                            beta="exact product of original .84/.04 categorical masses at selected observed states",
                            projection="deployed float32 actor, nearest grid, coordinate ties lower/lexicographic first",
                            deployed_actor_actions=self.actor_actions.tolist(),
                            projected_actor_indices=self.actor_action_index.tolist()),
            score="abs(native final100k smoothed one-step TD target - final100k twin minimum) / (max(frozen95k eta,1e-6)*frozen95k unit)",
            label_noise="caller-supplied independent standard normal[batch,4]; native .2 scale,.5 noise clip,[-1,1] target action clip",
            objective=config["settings"], new_training_updates=0,
            limitation="conditional action-only law on64 fixed source states; no occupancy or long-horizonQ coverage claim")
        return self

    def _make_programs(self):
        import jax
        import jax.numpy as jnp
        models, state = self.result.models, self.result.state
        def fixed(obs, action, next_obs):
            q = models[1].apply(state.native.critic.params, obs, action).min(axis=-1)
            eta = models[2].apply(state.posterior.cal_params, obs, action)
            scale = jnp.maximum(eta, 1e-6) * state.posterior.residual_scale
            next_action = models[0].apply(state.native.actor_target.params, next_obs)
            return q, eta, scale, next_action
        def target(next_obs, next_action, reward, done, label_noise):
            noise = jnp.clip(label_noise * self.args.policy_noise, -self.args.noise_clip, self.args.noise_clip)
            action = jnp.clip(next_action + noise, -1., 1.)
            q = models[1].apply(state.native.critic_target.params, next_obs, action).min(axis=-1)
            return reward + (1. - done) * self.args.discount * q
        self._fixed, self._target = jax.jit(fixed), jax.jit(target)

    def target_probabilities(self, lambda_):
        import numpy as np
        if not np.isfinite(lambda_) or not 0 <= lambda_ <= 1:
            raise ValueError("mixture lambda must lie in[0,1]")
        probabilities = (1. - float(lambda_)) * self.beta.copy()
        probabilities[np.arange(64), self.actor_action_index] += float(lambda_)
        return probabilities

    def _indices(self, state_index, action_index):
        import numpy as np
        z, a = np.asarray(state_index), np.asarray(action_index)
        if z.shape != a.shape or z.ndim != 1 or not np.issubdtype(z.dtype,np.integer) or not np.issubdtype(a.dtype,np.integer):
            raise ValueError("aligned integer index vectors required")
        if np.any((z<0)|(z>=64)) or np.any((a<0)|(a>=625)):
            raise ValueError("state/action index out of range")
        return z, a

    def transitions(self, state_index, action_index):
        """Counterfactual actions at actual saved states; never reset or resample state."""
        import numpy as np
        z,a=self._indices(state_index,action_index)
        states=self.selected["state"][z]
        action=self.grid[a].astype(np.float32)
        nxt=np.empty_like(states); reward=np.empty(len(z),np.float64)
        for regime in (0,1):
            choose=self.selected["regime"][z]==regime
            nxt[choose],reward[choose]=self.C.dynamics(states[choose],action[choose],regime)
        obs=self.selected["observation"][z].astype(np.float32)
        next_raw=np.column_stack((nxt,self.selected["regime"][z],(99-self.selected["time"][z])/100)).astype(np.float32)
        return dict(obs=(obs-self.result.mean)/self.result.std, action=action,
                    next_obs=(next_raw-self.result.mean)/self.result.std,
                    reward=reward.astype(np.float32), done=(self.selected["time"][z]==99).astype(np.float32))

    def prepare_catalog(self, batch_size=4096):
        """Optional read-only40k pair cache; never draws labels or calibrates."""
        import numpy as np
        pieces={k:[] for k in ("q","eta","scale","next_action","next_obs","reward","done")}
        pairs=np.indices((64,625)).reshape(2,-1)
        for lo in range(0,pairs.shape[1],batch_size):
            t=self.transitions(*pairs[:,lo:lo+batch_size])
            fixed=self._fixed(t["obs"],t["action"],t["next_obs"])
            for k,v in zip(("q","eta","scale","next_action"),fixed): pieces[k].append(np.asarray(v))
            for k in ("next_obs","reward","done"):pieces[k].append(t[k])
        self._catalog={k:np.concatenate(v,axis=0) for k,v in pieces.items()}
        return {k:v.shape for k,v in self._catalog.items()}

    def score(self, state_index, action_index, label_noise):
        import numpy as np
        z,a=self._indices(state_index,action_index)
        noise=np.asarray(label_noise,np.float32)
        if noise.shape!=(len(z),4) or not np.isfinite(noise).all():
            raise ValueError("independent standard-normal label_noise must be finite[N,4]")
        if self._catalog is None:
            t=self.transitions(z,a)
            q,eta,scale,next_action=map(np.asarray,self._fixed(t["obs"],t["action"],t["next_obs"]))
        else:
            ix=z*625+a
            t={k:self._catalog[k][ix] for k in ("next_obs","reward","done")}
            q,eta,scale,next_action=(self._catalog[k][ix] for k in ("q","eta","scale","next_action"))
        target=np.asarray(self._target(t["next_obs"],next_action,t["reward"],t["done"],noise))
        residual=target-q
        if not np.isfinite(target).all() or not np.isfinite(scale).all() or not (scale>0).all():
            raise FloatingPointError("invalid frozen score")
        # Native freeze_reference first forms float32 residual, then converts
        # residual and positive_scale to float64 before absolute value/division.
        scores=np.abs(residual.astype(np.float64))/scale.astype(np.float64)
        return ScoreBatch(scores,residual,np.abs(residual),scale,target,q,eta)
