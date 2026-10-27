"""The guard's comparison (plan section 4.5, stop rule S0): guard.py's restatement against the weighting test's own
gate (experiments/shift_iw/learners_iw.py:54-57, 273-322), plus the approval gate's refusals (gate.py). An
experiment-local attachment (plan section 3.1) that changes none of the repository's declared methods.

The driver runs in /opt/bca-venv. It writes made-up arrays and a list of cases by file (handoff.py); the original side
runs in /opt/bca-shift-venv, where learners_iw imports (numpy 2.2.6), applies learners_iw's own register_synthetic and
require_gate with its SYNTHETIC token to every case, and saves each decision and content hash by file; the driver
applies guard.py to the same cases and compares decisions and hashes. The original side runs the weighting test's BCA
gate on made-up arrays only (approval item 8). No JAX device is used and no learning happens.
  PYTHONDONTWRITEBYTECODE=1 SHIFT_HOSTS_DEVICE=none taskset -c 0-3 /opt/bca-venv/bin/python -m \
      experiments.shift_hosts.tests.test_guard_compare
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import unittest
from collections import namedtuple
from pathlib import Path

import numpy as np

from experiments.shift_hosts import gate
from experiments.shift_hosts import guard
from experiments.shift_hosts import handoff as H

AREA = "scaffold"
REPO = Path(__file__).resolve().parents[3]
SHIFT_PYTHON = "/opt/bca-shift-venv/bin/python"
FIELDS = ("obs", "action", "reward", "next_obs", "done")
REAL_LABEL = "ThrusterChain stalling world (real data)"
# (case, data prefix, world label, action, token): actions run in this order on both sides
CASES = (
    ("A_registered", "A", "made_up", "register_then_check", "token"),
    ("B_unregistered", "B", "made_up", "check", "token"),
    ("A_relabelled_real", "A", "real", "check", "token"),
    ("A_register_relabelled", "A", "real", "register", "token"),
    ("A_one_value_changed", "A1", "made_up", "check", "token"),
    ("A_bank_value_changed", "A2", "made_up", "check", "token"),
    ("A_obs_float64", "A64", "made_up", "check", "token"),
    ("A_token_none", "A", "made_up", "check", "none"),
    ("A_token_string", "A", "made_up", "check", "string"),
    ("A_token_object", "A", "made_up", "check", "object"),
    ("A_again", "A", "made_up", "check", "token"),
    ("B_registered_after", "B", "made_up", "register_then_check", "token"),
)
Container = namedtuple("Container", "world train bank")


def case_arrays():
    """Made-up arrays of seeds 0 (A) and 1 (B) and three altered copies of A, under prefixes A, B, A1, A2, A64."""
    from experiments.shift_hosts.tests import madeup

    a, b = madeup.build(0, register=False), madeup.build(1, register=False)
    out = {}
    for prefix, m in (("A", a), ("B", b)):
        for s in ("train", "bank"):
            for f in FIELDS:
                out[f"{prefix}/{s}/{f}"] = getattr(getattr(m, s), f)
    for prefix in ("A1", "A2", "A64"):
        for k in [k for k in out if k.startswith("A/")]:
            out[prefix + k[1:]] = out[k].copy()
    out["A1/train/obs"][0, 0] = np.nextafter(out["A1/train/obs"][0, 0], np.float32(np.inf))
    out["A2/bank/reward"][-1] = np.nextafter(out["A2/bank/reward"][-1], np.float32(-np.inf))
    out["A64/train/obs"] = out["A/train/obs"].astype(np.float64)
    return out


def original_side(folder):
    """In /opt/bca-shift-venv: the weighting test's own gate on every case. Writes original_decisions.json."""
    import experiments.shift  # noqa: F401  (numpy 2.2.6 and gymnasium checks)
    from experiments.shift import learners as LR
    from experiments.shift_iw import learners_iw as LI
    from runtime.networks import Transition

    folder = Path(folder)
    arrays, _ = H.read(folder / "cases.npz")
    world = {"made_up": LI.SYNTHETIC_WORLD, "real": REAL_LABEL}
    token = {"token": LI.SYNTHETIC, "none": None, "string": "SYNTHETIC", "object": object()}

    def prepared(prefix, label):
        tr = Transition(*(arrays[f"{prefix}/train/{f}"] for f in FIELDS))
        bk = Transition(*(arrays[f"{prefix}/bank/{f}"] for f in FIELDS))
        n, nb = len(tr.obs), len(bk.obs)
        return LR.Prepared(rep=0, world=world[label], removal="none", physics=None, spec=None, rows=np.arange(n),
                           states=np.asarray(tr.obs, np.float64), actions=tr.action, region=np.zeros(n, bool),
                           train=tr, bank_rows=np.arange(nb), bank=bk, obs_mean=np.zeros(13, np.float32),
                           obs_std=np.ones(13, np.float32), eval_rows=np.arange(0),
                           eval_obs=np.zeros((0, 13), np.float32))

    out = {}
    for name, prefix, label, action, tok in CASES:
        data = prepared(prefix, label)
        try:
            if action in ("register", "register_then_check"):
                LI.register_synthetic(data)
            if action in ("check", "register_then_check"):
                LI.require_gate(token[tok], data, "band")
            decision = "accept"
        except LI.ApprovalError:
            decision = "refuse"
        out[name] = dict(decision=decision, digest=LI.data_sha256(data))
    H.write_json(folder / "original_decisions.json", dict(
        decisions=out, numpy=np.__version__, executable=sys.executable, learners_iw=LI.__file__,
        synthetic_world=LI.SYNTHETIC_WORLD))
    return 0


def restated_decisions(arrays):
    """guard.py on every case, in the same order (in a fresh registry)."""
    guard._MADE_UP_DATA.clear()
    world = {"made_up": guard.MADE_UP_WORLD, "real": REAL_LABEL}
    token = {"token": guard.MADE_UP, "none": None, "string": "MADE_UP", "object": object()}
    out = {}
    for name, prefix, label, action, tok in CASES:
        from experiments.shift_hosts.tests.madeup import Rows

        data = Container(world[label], Rows(*(arrays[f"{prefix}/train/{f}"] for f in FIELDS)),
                         Rows(*(arrays[f"{prefix}/bank/{f}"] for f in FIELDS)))
        try:
            if action in ("register", "register_then_check"):
                guard.register_made_up(data)
            if action in ("check", "register_then_check"):
                guard.require_bca(token[tok], data)
            decision = "accept"
        except gate.ApprovalError:
            decision = "refuse"
        out[name] = dict(decision=decision, digest=guard.data_sha256(data))
    return out


class GuardAgainstOriginalTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from experiments.shift_hosts import tests as T

        cls.dir = T.out_dir(AREA) / "guard_compare"
        shutil.rmtree(cls.dir, ignore_errors=True)
        cls.dir.mkdir(parents=True)
        H.write(cls.dir / "cases.npz", case_arrays(), meta=dict(cases=[list(c) for c in CASES]))
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", JAX_PLATFORMS="cpu")
        cls.run_out = subprocess.run([SHIFT_PYTHON, "-m", "experiments.shift_hosts.tests.test_guard_compare",
                                      "--role", "original", "--dir", str(cls.dir)], cwd=str(REPO), env=env,
                                     capture_output=True, text=True, timeout=900)
        arrays, _ = H.read(cls.dir / "cases.npz")
        cls.mine = restated_decisions(arrays)

    def test_original_side_ran(self):
        self.assertEqual(self.run_out.returncode, 0, (self.run_out.stdout + self.run_out.stderr)[-4000:])
        rec = H.read_json(self.dir / "original_decisions.json")
        self.assertEqual(rec["executable"], SHIFT_PYTHON)
        self.assertEqual(rec["synthetic_world"], guard.MADE_UP_WORLD)

    def test_same_decisions_and_hashes(self):
        """Every case gets the same decision and the same content hash (data_sha256 = tree_sha256 of (train, bank))
        from the restatement as from the original."""
        theirs = H.read_json(self.dir / "original_decisions.json")["decisions"]
        H.write_json(self.dir / "restated_decisions.json", dict(decisions=self.mine, numpy=np.__version__))
        self.assertEqual(sorted(theirs), sorted(self.mine))
        for name in theirs:
            self.assertEqual(theirs[name], self.mine[name], name)

    def test_expected_decisions(self):
        """The decisions themselves: only registered made-up content with the made-up token is accepted."""
        accepted = sorted(k for k, v in self.mine.items() if v["decision"] == "accept")
        self.assertEqual(accepted, ["A_again", "A_registered", "B_registered_after"])
        digests = {k: v["digest"] for k, v in self.mine.items()}
        self.assertEqual(len({digests[k] for k in ("A_registered", "A_one_value_changed", "A_bank_value_changed",
                                                    "A_obs_float64", "B_unregistered")}), 5)


class BareTreeTest(unittest.TestCase):
    """The declared extension: the bare pytrees a restated step receives (Unifloral's Transition), registered by
    tests/madeup.py."""

    @classmethod
    def setUpClass(cls):
        from experiments.shift_hosts.tests import madeup

        cls.m = madeup.build(0)

    def test_registered_trees_open_bca(self):
        Transition5 = namedtuple("Transition", "obs action reward next_obs done")
        Transition6 = namedtuple("Transition", "obs action reward next_obs next_action done")
        for which in ("train", "bank", "strong"):
            self.assertIs(guard.require_bca(guard.MADE_UP, Transition5(*self.m.leaves(which, "td3_bc"))),
                          guard.MADE_UP)
            self.assertIs(guard.require_bca(guard.MADE_UP, Transition6(*self.m.leaves(which, "rebrac"))),
                          guard.MADE_UP)
        self.assertIs(guard.require_made_up(self.m), guard.MADE_UP)

    def test_other_content_is_refused(self):
        leaves = self.m.leaves("train", "td3_bc")
        for bad in ((leaves[0][:100],) + leaves[1:],                       # a subset of rows
                    (leaves[0].astype(np.float64),) + leaves[1:],          # another dtype
                    leaves[:4],                                            # a field missing
                    (leaves[0] + np.float32(1e-3),) + leaves[1:]):         # changed values
            with self.assertRaises(gate.ApprovalError):
                guard.require_bca(guard.MADE_UP, bad)
        with self.assertRaises(gate.ApprovalError):
            guard.register_arrays(leaves, "ThrusterChain")
        with self.assertRaises(gate.ApprovalError):
            guard.require_bca(guard.MADE_UP)

    def test_only_the_token_or_a_verified_approval(self):
        tree = self.m.leaves("train", "td3_bc")
        for permission in (None, "MADE_UP", object(), True):
            with self.assertRaises(gate.ApprovalError):
                guard.require_bca(permission, tree)
        forged = gate.Approval("approval_0000000000000000.json", "0" * 64, "0" * 64, "0" * 64, "0" * 40,
                               gate.APPROVAL_EMAIL, "David Bayha")
        with self.assertRaises(gate.ApprovalError):
            guard.require_bca(forged, tree)                   # gate.verify refuses: no approval on disk

    def test_tree_hash_is_learners_rule(self):
        """tree_sha256: dtype string, shape string, then bytes, for each leaf in pytree order (learners.py:184-190)."""
        import hashlib

        x, y = np.arange(6, dtype=np.float32).reshape(2, 3), np.array([True, False])
        h = hashlib.sha256()
        for a in (x, y):
            h.update(str(a.dtype).encode() + str(a.shape).encode())
            h.update(a.tobytes())
        self.assertEqual(guard.tree_sha256((x, dict(b=y))), h.hexdigest())


class GateRefusalTest(unittest.TestCase):
    """gate.require_run_approval refuses every run on real data without the user's committed approval."""

    def setUp(self):
        from experiments.shift_hosts import tests as T

        self.pkg = T.out_dir(AREA) / "gate_tmp" / "shift_hosts"
        shutil.rmtree(self.pkg.parent, ignore_errors=True)
        self.pkg.mkdir(parents=True)
        (self.pkg / "module.py").write_text("x = 1\n")

    def tearDown(self):
        shutil.rmtree(self.pkg.parent, ignore_errors=True)

    def test_refuses_now_in_the_real_package(self):
        """Without an approval file in experiments/shift_hosts every run on real data is refused; while prespec.json
        does not exist (stage 0), the refusal says so."""
        if list(gate.PACKAGE.glob(gate.APPROVAL_PREFIX + "*.json")):
            self.skipTest("an approval file exists; this stage-0 check no longer applies")
        with self.assertRaises(gate.ApprovalError) as ctx:
            gate.require_run_approval("test")
        if not (gate.PACKAGE / gate.PRESPEC_NAME).exists():
            self.assertIn("no frozen pre-specification", str(ctx.exception))

    def _approve(self, **override):
        fp, prespec_sha, manifest = gate.fingerprint(self.pkg, REPO)
        body = dict(kind=gate.APPROVAL_KIND, fingerprint=fp, typed=fp[:16], prespec_sha256=prespec_sha,
                    code_manifest=manifest, approved_by="test fixture (never the user's approval)")
        body.update(override)
        path = gate.approval_path(fp, self.pkg)
        path.write_text(json.dumps(body))
        H.write_sidecar(path)
        return path, fp

    def test_every_check_before_git(self):
        (self.pkg / gate.PRESPEC_NAME).write_text(json.dumps(dict(version=1)))
        with self.assertRaises(gate.ApprovalError) as ctx:                 # no approval file
            gate.require_run_approval("test", package=self.pkg, repo=REPO)
        self.assertIn("no approval file", str(ctx.exception))
        path, fp = self._approve()
        with self.assertRaises(gate.ApprovalError) as ctx:                 # everything right but not committed
            gate.require_run_approval("test", package=self.pkg, repo=REPO)
        self.assertIn("not committed in git HEAD", str(ctx.exception))
        manifest = gate.fingerprint(self.pkg, REPO)[2]
        self.assertTrue(any(k.endswith("shift_hosts/module.py") for k in manifest))
        self.assertIn("baselines/unifloral/algorithms/td3_bc.py", manifest)
        self.assertFalse(any("approval_" in k or k.endswith(gate.PRESPEC_NAME) for k in manifest))
        for override, words in ((dict(typed="0" * 16), "another fingerprint"),
                                (dict(prespec_sha256="0" * 64), "another prespec.json"),
                                (dict(code_manifest={}), "code manifest"),
                                (dict(kind="other"), "not a shift-hosts approval")):
            self._approve(**override)
            with self.assertRaises(gate.ApprovalError) as ctx:
                gate.require_run_approval("test", package=self.pkg, repo=REPO)
            self.assertIn(words, str(ctx.exception))
        path, fp = self._approve()
        path.write_text(path.read_text() + " ")                            # sidecar now differs
        with self.assertRaises(gate.ApprovalError) as ctx:
            gate.require_run_approval("test", package=self.pkg, repo=REPO)
        self.assertIn("sidecar", str(ctx.exception))
        self._approve()
        (self.pkg / "module.py").write_text("x = 2\n")                     # a code change: a new fingerprint
        self.assertNotEqual(gate.fingerprint(self.pkg, REPO)[0], fp)
        with self.assertRaises(gate.ApprovalError) as ctx:
            gate.require_run_approval("test", package=self.pkg, repo=REPO)
        self.assertIn("no approval file", str(ctx.exception))

    def test_reads_git_without_writing(self):
        """last_commit reads the author of the latest commit touching a file (read-only git)."""
        last = gate.last_commit(REPO / "experiments" / "shift_iw" / "world.py", REPO)
        self.assertIsNotNone(last)
        self.assertEqual(len(last[0]), 40)
        self.assertNotEqual(last[1], gate.APPROVAL_EMAIL)
        self.assertTrue(gate.committed_in_head(REPO / "experiments" / "shift_iw" / "world.py", REPO))
        self.assertFalse(gate.committed_in_head(self.pkg / "module.py", REPO))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--role", choices=("original",))
    p.add_argument("--dir")
    a = p.parse_args()
    if a.role:
        sys.exit(original_side(a.dir))
    from experiments.shift_hosts import tests as T

    sys.exit(T.run(sys.modules[__name__], AREA, device_name="none"))
