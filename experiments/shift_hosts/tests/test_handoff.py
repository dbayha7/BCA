"""Test (i), hand-offs between the two environments (plan section 4.5): every handoff.py file written on made-up arrays
in one environment is read in the other; the SHA-256 sidecars verify, dtypes and shapes match, and the arrays are bit
for bit equal, in both directions. No JAX device and no BCA code is used. An experiment-local attachment (plan section
3.1) that changes none of the repository's declared methods.

The driver runs in /opt/bca-venv (numpy 1.26.4) and starts the other side in /opt/bca-shift-venv (numpy 2.2.6):
  1. the driver writes A (the made-up arrays plus edge-case arrays: NaN, -0.0, infinities, subnormals, integer limits,
     booleans, a 0-d array, an empty array, a JAX-style uint32 key);
  2. the other side reads A, reports every array's dtype, shape and content hash as it read them, re-writes A's arrays
     as B and writes its own arrays (made there with numpy 2.2.6) as C, with the hashes it computed before writing;
  3. the driver reads B and C and compares them with A's arrays in memory and with the other side's hashes;
  4. the driver re-writes C as D; the other side reads D and compares it with its own arrays, regenerated in memory.
Tampering (a changed byte, a changed manifest, a missing sidecar) and refused names and dtypes are checked too.
  PYTHONDONTWRITEBYTECODE=1 SHIFT_HOSTS_DEVICE=none taskset -c 0-3 /opt/bca-venv/bin/python -m \
      experiments.shift_hosts.tests.test_handoff
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from experiments.shift_hosts import handoff as H

AREA = "scaffold"
REPO = Path(__file__).resolve().parents[3]
OTHER_PYTHON = {"/opt/bca-venv/bin/python": "/opt/bca-shift-venv/bin/python",
                "/opt/bca-shift-venv/bin/python": "/opt/bca-venv/bin/python"}


def edge_arrays():
    """Arrays whose bits are easy to lose: special floats, integer limits, odd shapes."""
    f32 = np.array([0.0, -0.0, np.nan, np.inf, -np.inf, 1e-45, -1e-45, np.finfo(np.float32).max,
                    np.finfo(np.float32).tiny, 1.0 / 3.0], np.float32)
    nan_payload = np.array([0x7FC00001, 0xFFC12345], np.uint32).view(np.float32)
    f64 = np.array([0.0, -0.0, np.nan, np.inf, -np.inf, 5e-324, np.finfo(np.float64).max, 0.1], np.float64)
    out = {"edge/f32": f32, "edge/f32_nan_payload": nan_payload, "edge/f64": f64,
           "edge/bool": np.array([[True, False], [False, True]]), "edge/scalar_f64": np.asarray(np.pi),
           "edge/empty_f32": np.zeros((0, 4), np.float32), "edge/key_uint32": np.array([0, 2 ** 32 - 1], np.uint32),
           "edge/f32_3d": np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 7.0,
           "edge/fortran_f32": np.asfortranarray(np.arange(12, dtype=np.float32).reshape(3, 4))}
    for t in (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64):
        info = np.iinfo(t)
        out[f"edge/{np.dtype(t).name}"] = np.array([info.min, info.max, 0, 1], t)
    return out


def other_arrays():
    """Arrays made by the other side (deterministic within one numpy version)."""
    rng = np.random.default_rng([20261016, 7779, 99])
    return {"other/normal_f32": rng.standard_normal((500, 13)).astype(np.float32),
            "other/uniform_f64": rng.uniform(-1, 1, 1000), "other/ints": rng.integers(0, 2 ** 40, 64),
            "other/flags": rng.uniform(size=77) < 0.5}


def report(arrays):
    return {k: dict(dtype=v.dtype.str, shape=list(v.shape), sha256=H.array_sha256(v)) for k, v in arrays.items()}


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype.str == b.dtype.str and a.shape == b.shape and \
        np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes()


def other_side(role, folder):
    """The other environment's half (python -m ... --role echo|verify --dir folder)."""
    folder = Path(folder)
    if role == "echo":
        a, manifest = H.read(folder / "a.npz")
        H.write(folder / "b.npz", a, meta=dict(role="echo of a", reader_numpy=np.__version__))
        mine = other_arrays()
        H.write(folder / "c.npz", mine, meta=dict(role="made by the other side"))
        H.write_json(folder / "echo_report.json", dict(
            numpy=np.__version__, python=sys.version.split()[0], executable=sys.executable,
            read_a=report(a), a_meta=manifest["meta"], wrote_c=report(mine)))
        return 0
    if role == "verify":
        d, _ = H.read(folder / "d.npz")
        mine = other_arrays()
        ok = sorted(d) == sorted(mine) and all(same(d[k], mine[k]) for k in mine)
        H.write_json(folder / "verify_report.json", dict(numpy=np.__version__, executable=sys.executable,
                                                         d_equals_own_arrays=bool(ok)))
        return 0 if ok else 3
    raise ValueError(role)


class HandoffAcrossEnvironmentsTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from experiments.shift_hosts import tests as T
        from experiments.shift_hosts.tests import madeup

        cls.dir = T.out_dir(AREA) / "handoff"
        shutil.rmtree(cls.dir, ignore_errors=True)
        cls.dir.mkdir(parents=True)
        m = madeup.build(0, register=False)
        cls.a = {**m.arrays(), **edge_arrays()}
        cls.meta = dict(kind="test (i)", seed=0, note="made-up arrays, not ThrusterChain data", inf=float("inf"))
        cls.manifest_a = H.write(cls.dir / "a.npz", cls.a, meta=cls.meta)
        cls.other = OTHER_PYTHON.get(sys.executable)
        if cls.other is None or not Path(cls.other).exists():
            raise unittest.SkipTest(f"no other environment for {sys.executable}")
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        cls.echo = subprocess.run([cls.other, "-m", "experiments.shift_hosts.tests.test_handoff", "--role", "echo",
                                   "--dir", str(cls.dir)], cwd=str(REPO), env=env, capture_output=True, text=True,
                                  timeout=600)

    def test_other_side_ran_with_the_other_numpy(self):
        self.assertEqual(self.echo.returncode, 0, self.echo.stdout + self.echo.stderr)
        rep = H.read_json(self.dir / "echo_report.json")
        self.assertNotEqual(rep["numpy"], np.__version__)
        self.assertEqual(rep["executable"], self.other)

    def test_a_read_there_bit_for_bit(self):
        """Direction 1: written here, read there: every dtype, shape and content hash as written."""
        rep = H.read_json(self.dir / "echo_report.json")
        self.assertEqual(rep["read_a"], report(self.a))
        self.assertEqual(rep["a_meta"], H.plain(self.meta))

    def test_b_and_c_read_here_bit_for_bit(self):
        """Direction 2: written there, read here. B (A's arrays re-written there) equals A in memory; C (made there)
        matches the hashes the other side computed before writing."""
        b, mb = H.read(self.dir / "b.npz")
        self.assertEqual(sorted(b), sorted(self.a))
        for k in self.a:
            self.assertTrue(same(b[k], self.a[k]), k)
        self.assertEqual(b["edge/scalar_f64"].shape, ())                  # a 0-d array stays 0-d
        self.assertEqual(b["edge/empty_f32"].shape, (0, 4))
        self.assertEqual(mb["writer"]["executable"], self.other)
        c, _ = H.read(self.dir / "c.npz")
        self.assertEqual(report(c), H.read_json(self.dir / "echo_report.json")["wrote_c"])

    def test_d_round_trip_back_there(self):
        """C re-written here as D, read there and compared with the other side's own arrays, regenerated there."""
        c, _ = H.read(self.dir / "c.npz")
        H.write(self.dir / "d.npz", c, meta=dict(role="c re-written by the driver"))
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        out = subprocess.run([self.other, "-m", "experiments.shift_hosts.tests.test_handoff", "--role", "verify",
                              "--dir", str(self.dir)], cwd=str(REPO), env=env, capture_output=True, text=True,
                             timeout=600)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertTrue(H.read_json(self.dir / "verify_report.json")["d_equals_own_arrays"])

    def test_same_arrays_same_bytes(self):
        """Writing the same arrays twice gives the same file (deterministic members and dates). Whether the other
        numpy writes the same bytes is recorded, not required."""
        again = H.write(self.dir / "a_again.npz", self.a, meta=self.meta)
        self.assertEqual(again["npz_sha256"], self.manifest_a["npz_sha256"])
        mb = H.read_manifest(self.dir / "b.npz")
        self.__class__.cross_numpy_same_bytes = mb["npz_sha256"] == self.manifest_a["npz_sha256"]


class HandoffRefusalTest(unittest.TestCase):

    def setUp(self):
        from experiments.shift_hosts import tests as T

        self.dir = T.out_dir(AREA) / "handoff_refusals"
        shutil.rmtree(self.dir, ignore_errors=True)
        self.dir.mkdir(parents=True)
        self.arrays = {"x/obs": np.arange(26, dtype=np.float32).reshape(2, 13), "x/done": np.zeros(2, np.float32)}
        H.write(self.dir / "f.npz", self.arrays)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_round_trip_and_subset(self):
        got, man = H.read(self.dir / "f.npz")
        self.assertTrue(all(same(got[k], self.arrays[k]) for k in self.arrays))
        sub, _ = H.read(self.dir / "f.npz", names=["x/done"])
        self.assertEqual(list(sub), ["x/done"])
        self.assertEqual(H.group(got, "x").keys(), {"obs", "done"})
        with self.assertRaises(H.HandoffError):
            H.read(self.dir / "f.npz", names=["x/missing"])

    def test_changed_byte_is_refused(self):
        p = self.dir / "f.npz"
        blob = bytearray(p.read_bytes())
        blob[len(blob) // 2] ^= 1
        p.write_bytes(bytes(blob))
        with self.assertRaises(H.HandoffError):
            H.read(p)

    def test_changed_byte_with_a_rewritten_sidecar_is_refused(self):
        """Even with a fresh sidecar, the manifest's hash of the .npz (and of every array) still refuses it."""
        p = self.dir / "f.npz"
        blob = bytearray(p.read_bytes())
        blob[len(blob) // 2] ^= 1
        p.write_bytes(bytes(blob))
        H.write_sidecar(p)
        with self.assertRaises(H.HandoffError):
            H.read(p)

    def test_changed_manifest_is_refused(self):
        m = self.dir / "f.json"
        body = json.loads(m.read_text())
        body["arrays"]["x/obs"]["shape"] = [13, 2]
        m.write_text(json.dumps(body))
        with self.assertRaises(H.HandoffError):
            H.read(self.dir / "f.npz")

    def test_missing_sidecar_is_refused(self):
        (self.dir / "f.npz.sha256").unlink()
        with self.assertRaises(H.HandoffError):
            H.read(self.dir / "f.npz")

    def test_refused_names_and_dtypes(self):
        for bad in ("../x", "/abs", "a//", "", "sp ace"):
            with self.assertRaises(H.HandoffError):
                H.write(self.dir / "g.npz", {bad: np.zeros(1)})
        for arr in (np.array(["s"]), np.array([object()], dtype=object), np.zeros(2, np.complex64),
                    np.zeros(2, dtype=[("a", np.float32)])):
            with self.assertRaises(H.HandoffError):
                H.write(self.dir / "g.npz", {"a": arr})
        with self.assertRaises(H.HandoffError):
            H.write(self.dir / "g.txt", {"a": np.zeros(1)})

    def test_parameter_trees_round_trip(self):
        """A flax-style parameter tree goes through flatten_dict, a file and unflatten_dict unchanged."""
        tree = {"params": {"Dense_0": {"kernel": np.ones((13, 4), np.float32), "bias": np.zeros(4, np.float32)},
                           "Dense_1": {"kernel": np.full((4, 1), 0.5, np.float32)}}, "logstd": np.zeros(4, np.float32)}
        flat = H.flatten_dict(tree, "actor")
        self.assertIn("actor/params/Dense_0/kernel", flat)
        H.write(self.dir / "p.npz", flat)
        back = H.unflatten_dict(H.read(self.dir / "p.npz")[0], "actor")
        self.assertEqual(back.keys(), tree.keys())
        self.assertTrue(same(back["params"]["Dense_1"]["kernel"], tree["params"]["Dense_1"]["kernel"]))
        self.assertTrue(same(back["logstd"], tree["logstd"]))

    def test_plain_json(self):
        rec = H.plain(dict(a=np.float32(1.5), b=[np.inf, -np.inf, np.nan], c=np.arange(3), d=(np.bool_(True),)))
        self.assertEqual(rec, dict(a=1.5, b=["inf", "-inf", "nan"], c=[0, 1, 2], d=[True]))
        H.write_json(self.dir / "r.json", rec)
        self.assertEqual(H.read_json(self.dir / "r.json"), rec)
        (self.dir / "r.json").write_text("{}")
        with self.assertRaises(H.HandoffError):
            H.read_json(self.dir / "r.json")


def _extra():
    return dict(cross_numpy_same_bytes=getattr(HandoffAcrossEnvironmentsTest, "cross_numpy_same_bytes", None),
                driver_numpy=np.__version__, other_python=OTHER_PYTHON.get(sys.executable))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--role", choices=("echo", "verify"))
    p.add_argument("--dir")
    a = p.parse_args()
    if a.role:
        sys.exit(other_side(a.role, a.dir))
    from experiments.shift_hosts import tests as T

    sys.exit(T.run(sys.modules[__name__], AREA, device_name="none", extra=_extra))
