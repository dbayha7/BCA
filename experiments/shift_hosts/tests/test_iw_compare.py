"""Test (c3), IW arithmetic = the weighting test's (plan section 4.5, stop rule S0): iw.py against the weighting test's
own experiments/shift_iw/coverage.py (mass_grid, threshold_table, lookup, lookup_gaps, per_step_summary, calibrate, kish,
Table.summary, table_from_summary) and experiments/shift_iw/learners_iw.py (table_float32, per_row_dose, reproduces,
dose_change), on made-up inputs. An experiment-local attachment (plan section 3.1) that changes none of the repository's
declared methods.

How. The driver runs in /opt/bca-venv (numpy 1.26.4). It builds made-up inputs (tests/madeup.py, and a made-up standard
refresh by bca_core's calibrate wrapper, which is reference.freeze_reference) and writes them by file (handoff.py). The
original side runs in /opt/bca-shift-venv (numpy 2.2.6), where experiments/shift_iw imports: it runs the originals on
the inputs it reads from that file and writes every output by file. The driver reads the same file, runs iw.py through
the same function (run_side; only the bindings differ, original_api against restated_api), and compares every output
bit for bit: arrays by dtype, shape and bytes, numbers through their float64 bytes, records as sorted JSON text.

per_row_dose runs op by op (jax.disable_jit) and compiled (jax.jit) on each side, with the original's refusal rule
(most_caution True); op by op is compared with op by op, compiled with compiled. Its scale network is a made-up
stand-in whose prediction is a given vector (eta), so both sides read the same eta bits whatever the compiler does with
a network; one more case runs the weighting test's own scale network (learners_iw.models()[2]) on made-up parameters,
op by op, and iw.py reads the prediction the original side computed. Processor only. This runs the weighting test's BCA
arithmetic on made-up arrays only (approval item 8).

The one environmental exception: Kish's effective size n_eff = (sum w)^2 / np.dot(w, w) (in wbcp.calibrate, so in every
table and calibration, and in coverage.kish). numpy's sum gives the same bits in both environments, but np.dot is
OpenBLAS's, and the two numpy builds carry different OpenBLAS versions (0.3.23.dev with numpy 1.26.4, scipy-openblas
0.3.29 with numpy 2.2.6), whose dot products can differ in the last bits. So n_eff and kish are checked apart: each side
records its own sum and dot of every weight vector, each side's n_eff (and kish) must be exactly (sum)^2 / dot of its own
parts, the sums must agree bit for bit, and the values must agree bit for bit wherever the two dots agree (and within
1e-14 relative everywhere). Everything else is compared bit for bit. n_eff is a reported number only: no threshold,
lookup or dose reads it, and stop rule S3 compares it inside one process.
  env $(launch env) SHIFT_HOSTS_DEVICE=cpu JAX_PLATFORMS=cpu taskset -c 22-23 /opt/bca-venv/bin/python -m \
      experiments.shift_hosts.tests.test_iw_compare
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import unittest
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments.shift_hosts import handoff as H

AREA = "iw"
REPO = Path(__file__).resolve().parents[3]
SHIFT_PYTHON = "/opt/bca-shift-venv/bin/python"
WEIGHTING_TEST = Path("/root/bca-shift/experiments/shift_iw")       # the weighting test's own copy (read only)
BLEND = 0.5

# name: (bank scores, stream, bank weights, the masses the grid is built on and looked up, grid ratio)
TABLES = {
    "exact": ("300", "a", "exact", ("train", "strong"), 1.01),
    "ones": ("300", "a", "ones", ("ones",), 1.01),
    "estimated": ("300", "a", "estimated", ("train_est", "strong_est"), 1.01),
    "shuffled": ("300", "a", "shuffled", ("train", "strong"), 1.01),
    "bank36": ("36", "a", "exact36", ("train",), 1.01),
    "heavy": ("300", "a", "exact", ("heavy",), 1.05),
    "stream_b": ("300", "b", "exact", ("train", "strong"), 1.01),
}
# (table, masses): "@grid" every 7th grid point of that table, "@below" half its first point and its first point
LOOKUPS = (("exact", "train"), ("exact", "strong"), ("exact", "@grid"), ("exact", "@below"), ("ones", "ones"),
           ("estimated", "train_est"), ("estimated", "strong_est"), ("shuffled", "train"), ("shuffled", "strong"),
           ("bank36", "train"), ("heavy", "heavy"), ("heavy", "@grid"), ("stream_b", "train"))
# name: (bank scores, stream, bank weights or None, test mass or None)
CALIBRATIONS = {
    "unweighted": ("300", "a", None, None), "exact_1.7": ("300", "a", "exact", 1.7), "ones_1": ("300", "a", "ones", 1.0),
    "estimated_2.2": ("300", "a", "estimated", 2.2), "unweighted_36": ("36", "a", None, None),
    "unweighted_b": ("300", "b", None, None), "exact36_7.39": ("36", "a", "exact36", 7.39),
}
KISH = ("exact", "estimated", "shuffled", "ones", "exact36")
WEIGHTS = KISH                                      # every bank-weight vector used (TABLES, CALIBRATIONS, KISH)
# name: (stored reference, bank scores, stream)
REPRODUCES = {"own": ("300", "300", "a"), "other_stream": ("300", "300", "b"), "nudged_score": ("300", "nudged", "a"),
              "bank36": ("36", "36", "a"), "wrong_bank": ("300", "36", "a")}
# name: (reference, eta, per-row thresholds)
DOSES = {
    "table": ("ready", "net", "table"), "levels": ("ready", "net", "levels"), "refused": ("ready", "net", "refused"),
    "equal": ("ready", "net", "equal"), "zeros": ("ready", "net", "zeros"), "all_inf": ("ready", "net", "inf"),
    "bank36": ("ready", "net", "bank36"), "tiny_eta": ("ready", "tiny", "table"), "nan_eta": ("ready", "nan", "table"),
    "not_ready_inf": ("not_ready", "net", "inf"), "not_ready_table": ("not_ready", "net", "table"),
    "bad_unit": ("bad_unit", "net", "table"),
}
DOSE_FIELDS = ("width", "dose", "support_mask", "inputs_valid", "has_support")
MODES = ("eager", "jit")
State = namedtuple("State", "posterior")           # what learners_iw.per_row_dose reads of a td3_bc_bca state
Batch = namedtuple("Batch", "obs action")


class FixedEta:
    """A made-up stand-in for the scale network: its prediction at any rows is params["eta"] (learners_iw.per_row_dose
    calls models[2].apply(cal_params, obs, action), line 664)."""

    @staticmethod
    def apply(params, obs, action):
        return params["eta"]


# ------------------------------------------------------------------------------------------------ inputs (driver)

def build_inputs():
    """The made-up inputs, as {name: array} for handoff.write (training environment, after unifloral_import.prepare)."""
    import jax
    import jax.numpy as jnp

    from calibration.network import Calibrator
    from experiments.shift_hosts import bca_core as BC
    from experiments.shift_hosts import iw as IW
    from experiments.shift_hosts.tests import madeup

    m = madeup.build(0)
    n = m.n
    rng = np.random.default_rng([20261016, 7779, 70])
    base = BC.fit_key_base(BC.seeds(0).s91)
    scale = BC.make_scale_state("td3_bc", jnp.asarray(m.train.obs.mean(0)), jnp.asarray(m.train.obs.std(0)),
                                BC.scale_init_key(base, "td3_bc"))
    unit = jnp.asarray(0.3, jnp.float32)
    resid = (0.2 * (1.0 + m.current["bank"]) * rng.standard_normal(len(m.bank.reward))).astype(np.float32)
    eta_bank = scale.apply_fn(scale.params, m.bank.obs, m.bank.action)
    key_a = BC.posterior_key(BC.refresh_key(BC.seeds(0).s39, 10_000), "td3_bc")
    key_b = BC.posterior_key(BC.refresh_key(BC.seeds(0).s39, 15_000), "td3_bc")
    cal = BC.calibrate_bank(scale.params, unit, eta_bank, resid, key_a)          # reference.freeze_reference
    cal36 = BC.calibrate_bank(scale.params, unit, eta_bank[:36], resid[:36], key_a)
    if not (cal.valid and cal36.valid and cal.diagnostics["certified"] and cal36.diagnostics["certified"]):
        raise AssertionError("the made-up refreshes must certify")
    scores = cal.scores
    k = int(np.argmin(np.abs(scores - IW.calibrate(scores, cal.stream).threshold)))
    nudged = scores.copy()
    nudged[k] = nudged[k] * (1 + 1e-6)                    # the score the stored threshold sits on
    w = m.weight["bank"]
    train, strong = m.weight["train"], m.weight["strong"][~m.in_ball["strong"]]
    noise = np.exp(0.1 * rng.standard_normal(len(w) + len(train) + len(strong)))  # made-up "classifier" weights
    arrays = {
        "scores/300": scores, "scores/36": cal36.scores, "scores/nudged": nudged,
        "stream/a": np.asarray(cal.stream, np.uint32), "stream/b": BC.posterior_stream(key_b),
        "w/exact": w, "w/estimated": w * noise[:len(w)], "w/shuffled": w[IW.placebo_permutation(0, len(w))],
        "w/ones": np.ones(len(w)), "w/exact36": w[:36].copy(),
        "m/train": train, "m/strong": strong, "m/train_est": train * noise[len(w):len(w) + len(train)],
        "m/strong_est": strong * noise[len(w) + len(train):], "m/ones": np.ones(n),
        "m/heavy": np.geomspace(0.05, 5000.0, 157),
        "train/obs": m.train.obs, "train/action": m.train.action,
    }
    for name, c in (("300", cal), ("36", cal36)):
        for f in ("residual_scale", "threshold", "lambda_hat", "lambda_hpd", "n_eff"):
            arrays[f"ref/{name}/{f}"] = np.asarray(getattr(c.reference, f), np.float32)
        arrays[f"ref/{name}/ready"] = np.asarray(c.reference.ready, bool)
    # per-row doses: the weighting test's scale network (learners_iw.models()[2]) with made-up parameters
    net = Calibrator(jnp.zeros(13), jnp.ones(13), state_dep=True)
    params = net.init(jax.random.fold_in(base, 7), jnp.zeros((1, 13)), jnp.zeros((1, 4)))
    eta = np.asarray(net.apply(params, m.train.obs, m.train.action), np.float32)
    tiny = eta.copy()
    tiny[::7] = np.float32(1e-8)                          # below the 1e-6 floor
    tiny[3] = np.float32(1e-6)
    nan = eta.copy()
    nan[5] = np.float32(np.nan)
    R_u = np.float32(cal.reference.threshold)
    table = IW.threshold_table(scores, cal.stream, w, IW.mass_grid(np.concatenate([train, strong])))
    table36 = IW.threshold_table(cal36.scores, cal.stream, w[:36], IW.mass_grid(train))
    dose = (1.0 + 0.5 * rng.uniform(size=n)).astype(np.float64)
    arrays.update({
        "eta/net": eta, "eta/tiny": tiny, "eta/nan": nan,
        "R/table": IW.row_thresholds(table, train), "R/levels": (m.R_levels * R_u).astype(np.float32),
        "R/refused": np.where(np.isinf(m.R_refused), np.inf, R_u).astype(np.float32),
        "R/equal": np.full(n, R_u, np.float32), "R/zeros": np.where(np.arange(n) % 5 == 0, 0.0, R_u).astype(np.float32),
        "R/inf": np.full(n, np.inf, np.float32), "R/bank36": IW.row_thresholds(table36, train),
        "dc/dose": dose, "dc/base": dose - 0.05 * rng.standard_normal(n),
        "dc/high": m.current["train"] >= np.quantile(m.current["train"], 0.9),
    })
    arrays.update(H.flatten_dict(jax.tree_util.tree_map(np.asarray, params), "calp"))
    return arrays


# ------------------------------------------------------------------------------------------------ both sides

def _reference(arrays, name, variant="ready", cal_params=None):
    """A calibration.reference.FrozenReference from the stored fields: 'ready' (the made-up refresh's), 'not_ready'
    (initial_reference's values) or 'bad_unit' (unit 0, an invalid reference)."""
    import jax.numpy as jnp

    from calibration.reference import FrozenReference

    f32 = lambda v: jnp.asarray(v, jnp.float32)
    params = {} if cal_params is None else cal_params
    if variant == "not_ready":
        return FrozenReference(params, f32(1.0), f32(np.inf), f32(np.inf), f32(np.inf), f32(0.0), jnp.asarray(False))
    g = lambda f: arrays[f"ref/{name}/{f}"]
    unit = f32(0.0) if variant == "bad_unit" else f32(g("residual_scale"))
    return FrozenReference(params, unit, f32(g("threshold")), f32(g("lambda_hat")), f32(g("lambda_hpd")),
                           f32(g("n_eff")), jnp.asarray(bool(g("ready"))))


def run_side(api, arrays):
    """Every comparison's outputs from one side's functions (api) on the inputs: ({name: array}, {name: record})."""
    import jax.numpy as jnp

    out, rec, tabs, unweighted = {}, {}, {}, {}
    for name, (s, st, w, masses, ratio) in TABLES.items():
        scores, stream = arrays[f"scores/{s}"], arrays[f"stream/{st}"]
        grid = api.mass_grid(np.concatenate([arrays[f"m/{k}"] for k in masses]), ratio)
        tab = api.threshold_table(scores, stream, arrays[f"w/{w}"], grid)
        R_u = api.calibrate(scores, stream).threshold
        tabs[name], unweighted[name] = tab, R_u
        out[f"grid/{name}"] = grid
        for f in ("raw", "thresholds"):
            out[f"table/{name}/{f}"] = getattr(tab, f)
        for f in ("lambda_hat", "n_eff"):
            out[f"table/{name}/{f}"] = np.float64(getattr(tab, f))
        for f in ("monotone", "running_max_applied"):
            out[f"table/{name}/{f}"] = np.bool_(getattr(tab, f))
        summary = tab.summary(R_u)
        rec[f"summary/{name}"] = summary
        back = api.table_from_summary(json.loads(json.dumps(H.plain(summary))))
        for f in ("grid", "raw", "thresholds"):
            out[f"back/{name}/{f}"] = getattr(back, f)
    for tname, mname in LOOKUPS:
        tab = tabs[tname]
        if mname == "@grid":
            m = tab.grid[::7].copy()
        elif mname == "@below":
            m = np.array([tab.grid[0] / 2, tab.grid[0]])
        else:
            m = arrays[f"m/{mname}"]
        R, idx = api.lookup(tab, m)
        key = f"lookup/{tname}/{mname.lstrip('@')}"
        out[key + "/R"], out[key + "/idx"] = R, np.asarray(idx).astype(np.int64)
        out[key + "/float32"] = api.row_float32(tab, m)
        rec[key + "/gaps"] = api.lookup_gaps(tab, m, idx)
        rec[key + "/summary"] = api.per_step_summary(R, unweighted[tname])
    for name, (s, st, w, t) in CALIBRATIONS.items():
        c = api.calibrate(arrays[f"scores/{s}"], arrays[f"stream/{st}"], None if w is None else arrays[f"w/{w}"], t)
        for f in ("threshold", "lambda_hat", "lambda_hpd", "n_eff"):
            out[f"cal/{name}/{f}"] = np.float64(getattr(c, f))
        out[f"cal/{name}/crossings"] = np.asarray(c.crossings, np.float64)
    for name in KISH:
        out[f"kish/{name}"] = np.float64(api.kish(arrays[f"w/{name}"]))
    for name in WEIGHTS:            # n_eff's parts in this environment: as coverage.kish and as wbcp.calibrate (scaled)
        w = np.asarray(arrays[f"w/{name}"], np.float64)
        for label, v in (("raw", w), ("scaled", w / w.max())):
            out[f"parts/{name}/{label}/sum"] = np.float64(v.sum())
            out[f"parts/{name}/{label}/dot"] = np.float64(np.dot(v, v))
    for name, (r, s, st) in REPRODUCES.items():
        rec[f"reproduces/{name}"] = bool(api.reproduces(_reference(arrays, r), arrays[f"scores/{s}"],
                                                        arrays[f"stream/{st}"]))
    rec["dose_change/high"] = api.dose_change(arrays["dc/dose"], arrays["dc/base"], arrays["dc/high"])
    rec["dose_change/plain"] = api.dose_change(arrays["dc/dose"], arrays["dc/base"])
    for name, (variant, e, r) in DOSES.items():
        eta = arrays[f"eta/{e}"]
        ref = _reference(arrays, "300", variant, {"eta": jnp.asarray(eta)})
        for mode in MODES:
            d, refused = api.per_row_dose(ref, eta, arrays["train/obs"], arrays["train/action"], arrays[f"R/{r}"], mode)
            for f in DOSE_FIELDS:
                out[f"dose/{name}/{mode}/{f}"] = np.asarray(getattr(d, f))
            out[f"dose/{name}/{mode}/refused"] = np.asarray(refused)
    return out, rec


def _jit_and_eager(fn):
    import jax

    jitted = jax.jit(fn)

    def run(mode, *args):
        if mode == "eager":
            with jax.disable_jit():
                return fn(*args)
        return jitted(*args)

    return run


def original_api():
    """The weighting test's own functions (experiments/shift_iw/coverage.py, learners_iw.py), /opt/bca-shift-venv."""
    import jax.numpy as jnp

    from experiments.shift_iw import coverage as CV
    from experiments.shift_iw import learners_iw as LI

    models = (None, None, FixedEta)
    dose = _jit_and_eager(lambda st, b, r: LI.per_row_dose(models, st, b, r, BLEND))
    return SimpleNamespace(
        mass_grid=CV.mass_grid, threshold_table=CV.threshold_table, lookup=CV.lookup, lookup_gaps=CV.lookup_gaps,
        per_step_summary=CV.per_step_summary, kish=CV.kish, table_from_summary=CV.table_from_summary,
        calibrate=lambda s, st, w=None, t=None: CV.calibrate(s, st, w, t),
        row_float32=lambda tab, m: LI.table_float32(tab)[CV.lookup(tab, m)[1]],
        reproduces=lambda ref, s, st: LI.reproduces(ref, CV.calibrate(s, st)),
        dose_change=LI.dose_change,
        per_row_dose=lambda ref, eta, obs, act, R, mode: dose(mode, State(ref), Batch(jnp.asarray(obs), jnp.asarray(act)),
                                                             jnp.asarray(R)))


def restated_api():
    """iw.py's functions (training environment)."""
    import jax.numpy as jnp

    from experiments.shift_hosts import iw as IW

    dose = _jit_and_eager(lambda ref, e, r: IW.per_row_dose(ref, e, r, BLEND, True))
    return SimpleNamespace(
        mass_grid=IW.mass_grid, threshold_table=IW.threshold_table, lookup=IW.lookup, lookup_gaps=IW.lookup_gaps,
        per_step_summary=IW.per_step_summary, kish=IW.kish, table_from_summary=IW.table_from_summary,
        calibrate=lambda s, st, w=None, t=None: IW.calibrate(s, st, w, t),
        row_float32=IW.row_thresholds, reproduces=IW.reproduces, dose_change=IW.dose_change,
        per_row_dose=lambda ref, eta, obs, act, R, mode: dose(mode, ref, jnp.asarray(eta), jnp.asarray(R)))


def _blas():
    """The BLAS library numpy was built with (np.dot's), as numpy reports it."""
    try:
        cfg = np.show_config(mode="dicts")
        blas = cfg.get("Build Dependencies", {}).get("blas", {})
        return dict(name=blas.get("name"), version=blas.get("version"))
    except Exception as error:          # an older numpy without mode="dicts"
        return dict(error=repr(error))


def _net_params(arrays):
    import jax
    import jax.numpy as jnp

    return jax.tree_util.tree_map(jnp.asarray, H.unflatten_dict({k: v for k, v in arrays.items()
                                                                if k.startswith("calp/")}, "calp"))


def original_side(folder):
    """In /opt/bca-shift-venv: the originals on the inputs; writes original.npz (its manifest is original.json) and
    the records original_records.json."""
    import experiments.shift  # noqa: F401  (numpy 2.2.6 and gymnasium checks, as the weighting test imports it)
    import jax
    import jax.numpy as jnp

    from experiments.shift_iw import coverage as CV
    from experiments.shift_iw import learners_iw as LI

    folder = Path(folder)
    arrays, _ = H.read(folder / "inputs.npz")
    started = time.perf_counter()
    out, rec = run_side(original_api(), arrays)
    # the weighting test's own scale network (learners_iw.models()[2]) on the made-up parameters, op by op
    models = LI.models()
    params = _net_params(arrays)
    ref = _reference(arrays, "300", "ready", params)
    obs, act = jnp.asarray(arrays["train/obs"]), jnp.asarray(arrays["train/action"])
    with jax.disable_jit():
        out["net/eta"] = np.asarray(models[2].apply(params, obs, act))
        d, refused = LI.per_row_dose(models, State(ref), Batch(obs, act), jnp.asarray(arrays["R/table"]), BLEND)
    for f in DOSE_FIELDS:
        out[f"net/{f}"] = np.asarray(getattr(d, f))
    out["net/refused"] = np.asarray(refused)
    rec["meta"] = dict(numpy=np.__version__, blas=_blas(), jax=jax.__version__, executable=sys.executable,
                       coverage=CV.__file__, learners_iw=LI.__file__, coverage_sha256=H.sha256_file(CV.__file__),
                       learners_iw_sha256=H.sha256_file(LI.__file__), xla_flags=os.environ.get("XLA_FLAGS"),
                       jax_platforms=os.environ.get("JAX_PLATFORMS"), seconds=round(time.perf_counter() - started, 3))
    H.write(folder / "original.npz", out, meta=dict(side="original (experiments/shift_iw, /opt/bca-shift-venv)"))
    H.write_json(folder / "original_records.json", rec)
    return 0


# ------------------------------------------------------------------------------------------------ the comparison

def _bits(a):
    a = np.asarray(a)
    return a.dtype.str, a.shape, a.tobytes()


def _text(obj):
    return json.dumps(H.plain(obj), sort_keys=True)


class IWAgainstOriginalTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from experiments.shift_hosts import tests as T
        from experiments.shift_hosts import unifloral_import as UI

        UI.prepare(device="cpu")
        import jax
        import jax.numpy as jnp

        from experiments.shift_hosts import iw as IW

        cls.dir = T.out_dir(AREA) / "iw_compare"
        shutil.rmtree(cls.dir, ignore_errors=True)
        cls.dir.mkdir(parents=True)
        H.write(cls.dir / "inputs.npz", build_inputs(),
                meta=dict(tables=TABLES, lookups=LOOKUPS, calibrations=CALIBRATIONS, kish=KISH, reproduces=REPRODUCES,
                          doses=DOSES, blend=BLEND))
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", JAX_PLATFORMS="cpu")
        started = time.perf_counter()
        cls.run_out = subprocess.run([SHIFT_PYTHON, "-m", "experiments.shift_hosts.tests.test_iw_compare", "--role",
                                      "original", "--dir", str(cls.dir)], cwd=str(REPO), env=env, capture_output=True,
                                     text=True, timeout=1800)
        cls.original_seconds = time.perf_counter() - started
        inputs, _ = H.read(cls.dir / "inputs.npz")
        cls.inputs = inputs
        started = time.perf_counter()
        cls.mine, cls.mine_rec = run_side(restated_api(), inputs)
        cls.mine_seconds = time.perf_counter() - started
        cls.theirs, cls.theirs_rec = {}, {}
        if cls.run_out.returncode == 0:
            cls.theirs, _ = H.read(cls.dir / "original.npz")
            cls.theirs_rec = H.read_json(cls.dir / "original_records.json")
            # iw.py on the prediction the weighting test's own scale network gave on the original side, op by op
            ref = _reference(inputs, "300", "ready", _net_params(inputs))
            with jax.disable_jit():
                d, refused = IW.per_row_dose(ref, jnp.asarray(cls.theirs["net/eta"]), jnp.asarray(inputs["R/table"]),
                                             BLEND, True)
            for f in DOSE_FIELDS:
                cls.mine[f"net/{f}"] = np.asarray(getattr(d, f))
            cls.mine["net/refused"] = np.asarray(refused)
            # informational: the same network in this environment, op by op
            from calibration.network import Calibrator

            net = Calibrator(jnp.zeros(13), jnp.ones(13), state_dep=True)
            with jax.disable_jit():
                here = np.asarray(net.apply(_net_params(inputs), jnp.asarray(inputs["train/obs"]),
                                            jnp.asarray(inputs["train/action"])))
            cls.eta_same_here = here.tobytes() == cls.theirs["net/eta"].tobytes()
        H.write(cls.dir / "restated.npz", cls.mine, meta=dict(side="restated (iw.py, /opt/bca-venv)"))
        H.write_json(cls.dir / "restated_records.json", cls.mine_rec)

    def _compare(self, prefix, skip=()):
        """Every array named prefix... (but those ending in a `skip` suffix) bit for bit; returns how many."""
        keep = lambda k: k.startswith(prefix) and not k.endswith(tuple(skip))
        names = sorted(k for k in self.theirs if keep(k))
        self.assertTrue(names, prefix)
        self.assertEqual(names, sorted(k for k in self.mine if keep(k)))
        bad = [k for k in names if _bits(self.theirs[k]) != _bits(self.mine[k])]
        self.assertEqual(bad, [], f"{len(bad)} of {len(names)} differ")
        return len(names)

    def _compare_records(self, prefix, drop=()):
        """Every record named prefix... as sorted JSON text, with the fields in `drop` left out of dict records."""
        cut = lambda r: {k: v for k, v in r.items() if k not in drop} if isinstance(r, dict) else r
        names = sorted(k for k in self.theirs_rec if k.startswith(prefix))
        self.assertTrue(names, prefix)
        self.assertEqual(names, sorted(k for k in self.mine_rec if k.startswith(prefix)))
        bad = [k for k in names if _text(cut(self.theirs_rec[k])) != _text(cut(self.mine_rec[k]))]
        self.assertEqual(bad, [], f"{len(bad)} of {len(names)} differ")

    def test_original_side_ran(self):
        """The originals ran in /opt/bca-shift-venv (numpy 2.2.6) from the merged copy of the weighting test's files,
        which are byte for byte the weighting test's own (/root/bca-shift, read only)."""
        self.assertEqual(self.run_out.returncode, 0, (self.run_out.stdout + self.run_out.stderr)[-4000:])
        meta = self.theirs_rec["meta"]
        self.assertEqual(meta["executable"], SHIFT_PYTHON)
        self.assertEqual(meta["numpy"], "2.2.6")
        self.assertEqual(Path(meta["coverage"]).resolve(), (REPO / "experiments/shift_iw/coverage.py").resolve())
        self.assertEqual(Path(meta["learners_iw"]).resolve(), (REPO / "experiments/shift_iw/learners_iw.py").resolve())
        self.assertEqual(meta["coverage_sha256"], H.sha256_file(WEIGHTING_TEST / "coverage.py"))
        self.assertEqual(meta["learners_iw_sha256"], H.sha256_file(WEIGHTING_TEST / "learners_iw.py"))
        print(f"\n  original side {self.original_seconds:.1f} s (its own work {meta['seconds']} s), "
              f"restated side {self.mine_seconds:.1f} s")

    def test_mass_grids(self):
        self.assertEqual(self._compare("grid/"), len(TABLES))
        self.assertEqual(self.theirs["grid/ones"].tolist(), [1.0])

    def test_threshold_tables(self):
        """Grid, raw and used thresholds, lambda_hat, the monotone flags: bit for bit (n_eff: test_n_eff_and_kish)."""
        self._compare("table/", skip=("/n_eff",))

    def test_table_summaries_and_round_trip(self):
        self._compare_records("summary/", drop=("n_eff",))
        self._compare("back/")

    def test_lookups(self):
        self._compare("lookup/")

    def test_lookup_gaps_and_per_step_summaries(self):
        self._compare_records("lookup/")

    def test_calibrations(self):
        """threshold, lambda_hat, lambda_hpd and all 1,000 posterior crossings: bit for bit (n_eff below)."""
        self._compare("cal/", skip=("/n_eff",))

    def test_n_eff_and_kish(self):
        """n_eff and kish are (sum w)^2 / np.dot(w, w). The sums agree bit for bit across the environments; each side's
        value is exactly (sum)^2 / dot of its own parts; the two sides agree bit for bit wherever their np.dot agrees,
        and within 1e-14 relative everywhere. Their only difference is np.dot (each numpy's own OpenBLAS)."""
        t, m = self.theirs, self.mine
        for name in WEIGHTS:
            for label in ("raw", "scaled"):
                p = f"parts/{name}/{label}"
                self.assertEqual(_bits(t[p + "/sum"]), _bits(m[p + "/sum"]), p)
        uses = [(f"kish/{k}", k, "raw") for k in KISH]
        uses += [(f"table/{n}/n_eff", spec[2], "scaled") for n, spec in TABLES.items()]
        uses += [(f"cal/{n}/n_eff", spec[2], "scaled") for n, spec in CALIBRATIONS.items() if spec[2] is not None]
        for n, spec in CALIBRATIONS.items():          # unweighted: every weight 1, so n_eff is the bank's size exactly
            if spec[2] is None:
                self.assertEqual(_bits(t[f"cal/{n}/n_eff"]), _bits(m[f"cal/{n}/n_eff"]), n)
                self.assertEqual(float(t[f"cal/{n}/n_eff"]), float(len(self.inputs[f"scores/{spec[0]}"])))
        differ = []
        for key, w, label in uses:
            p = f"parts/{w}/{label}"
            for side in (t, m):
                again = np.float64(side[p + "/sum"]) ** 2 / np.float64(side[p + "/dot"])
                self.assertEqual(_bits(np.float64(again)), _bits(side[key]), key)
            same_dot = _bits(t[p + "/dot"]) == _bits(m[p + "/dot"])
            if same_dot:
                self.assertEqual(_bits(t[key]), _bits(m[key]), key)
            self.assertLessEqual(abs(float(m[key]) / float(t[key]) - 1.0), 1e-14, key)
            if _bits(t[key]) != _bits(m[key]):
                differ.append(key)
        dots = sum(_bits(t[f"parts/{w}/{lb}/dot"]) != _bits(m[f"parts/{w}/{lb}/dot"]) for w in WEIGHTS
                   for lb in ("raw", "scaled"))
        print(f"\n  np.dot differs in {dots} of {2 * len(WEIGHTS)} weight vectors (OpenBLAS "
              f"{_blas().get('version')} here, {self.theirs_rec['meta']['blas'].get('version')} there); "
              f"n_eff or kish differs in the last bits in {len(differ)} of {len(uses)}: {differ}")

    def test_reproduces(self):
        self._compare_records("reproduces/")
        r = self.theirs_rec
        self.assertTrue(r["reproduces/own"] and r["reproduces/bank36"])
        self.assertFalse(r["reproduces/nudged_score"] or r["reproduces/wrong_bank"])

    def test_dose_change(self):
        self._compare_records("dose_change/")

    def test_per_row_dose(self):
        """Every per-row dose case, op by op and compiled: width, dose, support, validity and refused rows."""
        self.assertEqual(self._compare("dose/"), len(DOSES) * len(MODES) * (len(DOSE_FIELDS) + 1))

    def test_per_row_dose_compiled_against_op_by_op(self):
        """Within each side, XLA's compiled dose may differ from the op-by-op one in the last float32 bit (the compiler
        may fuse the arithmetic); the comparison above matches compiled with compiled and op by op with op by op, so
        the restatement equals the original in each. Here: masks, validity and refused rows are equal, widths and doses
        agree to float32 rounding (relative 1e-6), and both sides differ on exactly the same rows."""
        rows = {}
        for label, side in (("original", self.theirs), ("restated", self.mine)):
            for name in DOSES:
                g = lambda mode, f: side[f"dose/{name}/{mode}/{f}"]
                for f in ("support_mask", "inputs_valid", "has_support", "refused"):
                    self.assertEqual(_bits(g("eager", f)), _bits(g("jit", f)), (label, name, f))
                for f in ("width", "dose"):
                    a, b = g("eager", f).astype(np.float64), g("jit", f).astype(np.float64)
                    both = np.isfinite(a) & np.isfinite(b)
                    self.assertEqual(np.isnan(a).tolist(), np.isnan(b).tolist())
                    self.assertTrue(np.array_equal(a[~both & ~np.isnan(a)], b[~both & ~np.isnan(b)]))
                    np.testing.assert_allclose(a[both], b[both], rtol=1e-6, atol=0)
                    rows[(label, name, f)] = np.flatnonzero(g("eager", f).view(np.uint32) != g("jit", f).view(np.uint32))
        differ = 0
        for name in DOSES:
            for f in ("width", "dose"):
                self.assertEqual(rows[("original", name, f)].tolist(), rows[("restated", name, f)].tolist(), (name, f))
                differ += len(rows[("original", name, f)])
        print(f"\n  compiled against op by op: {differ} float32 values differ in the last bit, on both sides alike")

    def test_per_row_dose_with_the_weighting_tests_scale_network(self):
        """The weighting test's own scale network on made-up parameters (op by op): iw.py, given the prediction the
        original computed, gives the original's width, dose, support, validity and refused rows bit for bit."""
        self._compare("net/", skip=("/eta",))
        print(f"\n  the scale network's prediction is the same bits in both environments: {self.eta_same_here}")

    def test_the_inputs_reach_every_branch(self):
        """The cases exercise what they should: refused rows (dose 1.5), infinite thresholds in the tables, a grid of
        [1.0], rows below the 1e-6 floor, invalid inputs, a reference not ready (dose exactly 1)."""
        t = self.theirs
        self.assertTrue(np.isinf(t["table/heavy/thresholds"][-1]) and np.isinf(t["table/bank36/thresholds"][-1]))
        self.assertTrue(np.isfinite(t["table/exact/thresholds"]).all())
        self.assertTrue(np.isinf(t["lookup/bank36/train/float32"]).any())
        for mode in MODES:
            d = lambda name, f: t[f"dose/{name}/{mode}/{f}"]
            self.assertTrue(np.all(d("refused", "dose")[d("refused", "refused")] == 1.5))
            self.assertTrue(d("refused", "refused").any() and not d("refused", "refused").all())
            self.assertTrue(np.all(d("all_inf", "dose") == 1.5))
            self.assertTrue(np.all(d("bank36", "dose")[d("bank36", "refused")] == 1.5) and d("bank36", "refused").any())
            self.assertTrue(np.all(d("not_ready_inf", "dose") == 1.0) and not d("not_ready_inf", "refused").any())
            self.assertTrue(np.all(d("not_ready_table", "dose") == 1.0))
            self.assertFalse(bool(d("nan_eta", "inputs_valid")) or bool(d("bad_unit", "inputs_valid")))
            self.assertTrue(bool(d("table", "inputs_valid")) and bool(d("tiny_eta", "inputs_valid")))
            self.assertTrue(np.all(d("zeros", "dose")[::5] == 1.0))
            dose = d("table", "dose")
            self.assertTrue(np.all((dose > 1.0) & (dose < 1.5)))
        self.assertGreater(len(np.unique(t["dose/levels/eager/dose"])), 4)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--role", choices=("original",))
    p.add_argument("--dir")
    a = p.parse_args()
    if a.role:
        sys.exit(original_side(a.dir))
    from experiments.shift_hosts import tests as T
    from experiments.shift_hosts import unifloral_import as UI

    UI.prepare(device="cpu")
    sys.exit(T.run(sys.modules[__name__], AREA, device_name="cpu", extra=UI.record))
