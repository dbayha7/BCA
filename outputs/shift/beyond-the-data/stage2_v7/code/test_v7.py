"""Stage 2, version 7: tests of the v7 core (V7_SPEC.md, every 'Tested by' item that concerns these files).

SYNTHETIC FIXTURES ONLY: every fixture lives in a temporary folder under the scratchpad (outside /home/user/bca);
nothing is written under runs/ (neither v6's output folder nor R7/), under the gate's folders or under any pinned
folder. Gate actors in the fixtures are randomly perturbed network templates; no ThrusterChain log is generated. The
real files read (never written): V7/world_constants.json, the gate's thruster_chain/recorder (to recompute A, B, K and to
run the gate's load_prereg + guard with the constants), v6's committed pre-registration files and stage2_common.py.

What is tested (spec section in brackets):
  [C1]  the constants: refusal for a changed file, a tolerance exceeded, B differing; read-only, same-object returns;
        the column cache cleared; an unknown argument tuple exits 97; a tripwire call exits 97 even inside an
        'except Exception' block; the unchanged gate accepts with the constants substituted.
  [C2]  through the real launch_v7.sh: a 17th start refuses (16 slot locks held); a double start refuses (phase lock).
  [C3]  launch_v7.sh refuses a list, a range or two processors; the wrapper refuses two visible processors.
  [C4]  through the real launch_v7.sh: stray variables in the caller's environment never reach the phase, which sees
        exactly the eight; the wrapper started directly with a stray variable refuses (exit 97).
  [C5]  the phase table (done, failed on record, partial records that never raise), the classification of
        interruptions (exit 97 before the memory patterns; each attempt of a shared per-phase log classified on its own
        lines only: a memory error, then an exit-97 refusal, then an unrelated failure), a torn attempt record moved
        aside and written again (with an estimated end), a SIGKILLed phase through launch_v7.sh (phase-specific; its
        writes listed), moving a row aside; the test seam refuses a fixture config missing any path, a path outside
        the fixture (a symbolic link to the real folder), v6's pinned run_stage2.py, and a fixture around production.
  [C6]  a missing or changed HANDOFF.json, WITHDRAWN.json and an inherited phase refuse.
  [C7]  the redirect: no entry imports nothing; a policy entry replaces exactly its keys; a changed gate file still
        refuses through the original loader; real multi-actor files; a host entry calls the original get_host path;
        world keys; every exit 97 case; gate files unchanged; through launch_v7.sh a stub phase's lock records the
        re-made actors; the probes' calls carry the gate's arguments and never reach a training or writing branch;
        step 2's rule and the forecast; --remake.
  [C10] the ledger's audit hook; the approval route (approve_local_v7 refuses a non-terminal; import; the agent-identity
        cases copied verbatim from S2C/test_stage2.py:864-870; the new approver check; the import record on origin's
        branch); the writers (draft, final, refusal once approved; Q7 recorded only with the user's words and time;
        the final refuses a HANDOFF.json of another draft or one without a READY full self-test with P3 passed); the
        approval screen shows the self-test, P3, Q7 and the forecast's verdict with a warning above 1e-5.

Run (two processors, low priority; the v6 run keeps processors 2 and 3):
  cd /home/user/bca && env -i PATH=/usr/bin:/bin HOME=/root PYTHONPATH=/home/user/bca PYTHONDONTWRITEBYTECODE=1 \\
      JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 taskset -c 0,1 nice -n 10 /tmp/claude-0/bcavenv/bin/python \\
      outputs/shift/beyond-the-data/stage2_v7/code/test_v7.py
"""

import ast
import copy
import fcntl
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
if "/home/user/bca" not in sys.path:
    sys.path.insert(1, "/home/user/bca")

import v7_common as K  # noqa: E402
import v7_ledger as LG  # noqa: E402
import v7_verify as VF  # noqa: E402
import v7_refs as RF  # noqa: E402
import v7_probes as PR  # noqa: E402
import v7_constants as VC  # noqa: E402
import approve_v7 as AP7  # noqa: E402
import approve_local_v7 as AL7  # noqa: E402
import write_prereg_v7 as WP  # noqa: E402
import write_manifest_v7 as WM  # noqa: E402
import run_stage2_v7 as W  # noqa: E402

TEST_ROOT = K.SCRATCH / "v7_tests"
ENV = dict(K.PHASE_ENV)
LAUNCH = CODE / "launch_v7.sh"
APPROVER = f"David Bayha <{K.APPROVER_EMAIL}>"
PRELUDE = f"import sys\nsys.dont_write_bytecode = True\nsys.path.insert(0, {str(CODE)!r})\n"


def run(cmd, env=None, timeout=900, cwd=K.ROOT):
    return subprocess.run([str(c) for c in cmd], env=ENV if env is None else env, cwd=str(cwd), capture_output=True,
                          text=True, timeout=timeout)


def run_py(code, cpus="0", env=None, timeout=900):
    return run(["taskset", "-c", cpus, "nice", "-n", "10", K.PYTHON, "-c", PRELUDE + textwrap.dedent(code)], env=env,
               timeout=timeout)


def last_json(proc):
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
    if not lines:
        raise AssertionError(f"no JSON output (exit {proc.returncode}):\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}")
    return json.loads(lines[-1])


def write_json_sidecar(path, payload, canonical=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (K.canonical(payload) if canonical else json.dumps(payload)).encode("utf8")
    path.write_bytes(text)
    Path(str(path) + ".sha256").write_text(f"{hashlib.sha256(text).hexdigest()}  {path.name}\n")
    return hashlib.sha256(text).hexdigest()


def learners():
    import experiments.shift.learners as L
    return L


def perturbed(tree, k):
    import jax
    import jax.numpy as jnp
    return jax.tree_util.tree_map(lambda x: x + 0.01 * (k + 1) if jnp.issubdtype(jnp.asarray(x).dtype, jnp.floating)
                                  else x, tree)


class Tmp(unittest.TestCase):
    def setUp(self):
        TEST_ROOT.mkdir(parents=True, exist_ok=True)
        self.tmp = Path(tempfile.mkdtemp(dir=TEST_ROOT))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


# ============================================================================================ C1: constants

def synthetic_constants(dst, dA=0.0, dK=0.0, flipB=False, extra_space=False):
    import numpy as np
    body = json.loads((K.ROOT / K.CONSTANTS_REL).read_text(encoding="utf8"))

    def mod(block, delta=0.0, flip=False):
        M = np.array([[float.fromhex(s) for s in row] for row in block["hex"]], np.float64)
        i, j = np.unravel_index(np.argmax(np.abs(M)), M.shape)
        if delta:
            M[i, j] += delta
        if flip:
            v = M.view(np.uint64)
            v[0, 0] ^= np.uint64(1)
        block["hex"] = [[float(x).hex() for x in row] for row in M]
        block["sha256_le_c_order"] = K.sha256_bytes(np.ascontiguousarray(M, "<f8").tobytes())

    for e in body["functions"][VC.MAT_FN]["tuples"].values():
        mod(e["A"], dA)
        mod(e["B"], flip=flipB)
    for e in body["functions"][VC.GAIN_FN]["tuples"].values():
        mod(e["K"], dK)
    text = json.dumps(body, indent=1, sort_keys=True) + (" " if extra_space else "")
    Path(dst).write_text(text, encoding="utf8")
    digest = K.sha256_bytes(text.encode("utf8"))
    Path(str(dst) + ".sha256").write_text(f"{digest}  {Path(dst).name}\n")
    return digest


class ConstantsTests(Tmp):
    def test_real_file_matches_pin_and_is_canonical(self):
        c = VC.load_constants(K.ROOT / K.CONSTANTS_REL, K.CONSTANTS_SHA256)
        self.assertEqual(c.sha256, K.CONSTANTS_SHA256)
        self.assertEqual(len(c.ab), 1)
        self.assertEqual(len(c.k), 1)
        A, B = next(iter(c.ab.values()))
        self.assertEqual((A.shape, B.shape, next(iter(c.k.values())).shape), ((13, 13), (13, 4), (4, 13)))
        self.assertFalse(A.flags.writeable)

    def test_changed_file_refuses(self):
        p = self.tmp / "world_constants.json"
        sha = synthetic_constants(p, extra_space=True)
        with self.assertRaises(K.V7Refusal) as cm:
            VC.load_constants(p, K.CONSTANTS_SHA256)            # the pin is the real file's
        self.assertIn("is not the pinned", str(cm.exception))
        Path(str(p) + ".sha256").write_text("0" * 64 + "  x\n")   # pinned digest, wrong sidecar
        with self.assertRaises(K.V7Refusal):
            VC.load_constants(p, sha)

    def _diffs(self, **kw):
        p = self.tmp / "c.json"
        sha = synthetic_constants(p, **kw)
        return VC.local_differences(VC.load_constants(p, sha))

    def test_tolerances(self):
        VC.check_tolerances(self._diffs())                        # this machine: bit-identical
        VC.check_tolerances(self._diffs(dA=5e-15, dK=5e-13))     # within 1e-14 and 1e-12
        with self.assertRaises(K.V7Refusal) as cm:
            VC.check_tolerances(self._diffs(dA=3e-14))
        self.assertIn("A computed here differs", str(cm.exception))
        with self.assertRaises(K.V7Refusal) as cm:
            VC.check_tolerances(self._diffs(dK=3e-12))
        self.assertIn("K computed here differs", str(cm.exception))
        with self.assertRaises(K.V7Refusal) as cm:
            VC.check_tolerances(self._diffs(flipB=True))
        self.assertIn("B computed here differs", str(cm.exception))

    def test_installed_returns_read_only_cached_and_columns_cleared(self):
        out = last_json(run_py("""
            import json, v7_common as K, v7_constants as C
            f = C.apply(K.ROOT / K.CONSTANTS_REL, K.CONSTANTS_SHA256)
            import experiments.shift.thruster_chain as TC, experiments.shift.recorder as RC
            cols_after_install = TC._columns.cache_info().currsize
            A1, B1 = TC.matrices(TC.STALL); A2, B2 = TC.matrices(TC.NO_STALL)
            K1 = RC.textbook_gain(TC.STALL); K2 = RC.textbook_gain(TC.NO_STALL)
            try:
                A1[0, 0] = 1.0
                writable = True
            except ValueError:
                writable = False
            import numpy as np
            x = np.ones(13); a = np.zeros(4)
            TC.dynamics(x, a)
            print(json.dumps(dict(same=A1 is A2 and B1 is B2 and K1 is K2, writable=writable,
                                  dtypes=[str(A1.dtype), str(K1.dtype)], cols_after_install=cols_after_install,
                                  cols_used=TC._columns.cache_info().currsize, pinned=TC._matrices.v7_pinned,
                                  originals_cached=f["caches_at_import"], diffs=f["differences"])))
            """))
        self.assertTrue(out["same"])
        self.assertFalse(out["writable"])
        self.assertEqual(out["dtypes"], ["float64", "float64"])
        self.assertEqual(out["cols_after_install"], 0)
        self.assertEqual(out["cols_used"], 1)
        self.assertTrue(out["pinned"])
        self.assertTrue(out["diffs"]["A_bit_identical"] and out["diffs"]["K_bit_identical"])

    def test_unknown_tuple_and_tripwires_exit_97(self):
        ledger = self.tmp / "1.writes"
        ledger.write_text("")
        cases = {
            "unknown tuple": "TC.matrices(TC.Physics(gain=0.31, wave_turn=0.27))",
            "unknown gain tuple": "RC.textbook_gain(TC.Physics(gain=0.3, wave_turn=0.27, throttle_cost=0.31))",
            "transition_matrix": "GLQ.transition_matrix(GLQ.STAGE0_SPEC)",
            "solve_discrete_are": "import scipy.linalg as SL; SL.solve_discrete_are(1, 1, 1, 1)",
            "original _matrices": "ORIG(0.3, 0.27)",
        }
        for name, call in cases.items():
            with self.subTest(case=name):
                proc = run_py(f"""
                    import v7_common as K, v7_constants as C, v7_ledger as LG
                    led = LG.Ledger({str(ledger)!r})
                    import experiments.shift.thruster_chain as TC, experiments.shift.recorder as RC
                    import experiments.signal.gapped_lq as GLQ
                    ORIG = TC._matrices
                    C.apply(K.ROOT / K.CONSTANTS_REL, K.CONSTANTS_SHA256)
                    try:
                        {call}
                    except Exception as error:      # pinned v6 code catches any Exception (S2C/policy.py:419, :558)
                        print("CAUGHT", type(error).__name__)
                    print("CONTINUED")
                    """)
                self.assertEqual(proc.returncode, 97, proc.stderr[-2000:])
                self.assertNotIn("CAUGHT", proc.stdout)
                self.assertNotIn("CONTINUED", proc.stdout)
                self.assertIn("REFUSED (v7, exit 97)", proc.stderr)
        events, _ = LG.read_events(ledger)
        self.assertEqual(sum(e["ev"] == "v7-exit" for e in events), len(cases))

    def test_gate_accepts_with_constants(self):
        out = last_json(run_py("""
            import json, v7_common as K, v7_constants as C
            C.apply(K.ROOT / K.CONSTANTS_REL, K.CONSTANTS_SHA256)
            import experiments.shift.acceptance as A
            p = A.load_prereg(A.refuse_protected(A.RUNS_DEFAULT)); A.guard(p)
            import experiments.shift.thruster_chain as TC
            print(json.dumps(dict(ok=True, calls=TC._matrices.v7_calls)))
            """))
        self.assertTrue(out["ok"])
        self.assertGreater(out["calls"]["matrices"], 0)


# ============================================================================================ C5, C10: ledger

def run_stage2_parser_actions():
    """The add_argument calls of S2C/run_stage2.py's main(), executed against a fresh parser."""
    import argparse
    tree = ast.parse((K.S2C / "run_stage2.py").read_text(encoding="utf8"))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    ap = argparse.ArgumentParser()
    for node in ast.walk(main):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            src = ast.unparse(node).replace("ap.add_argument", "_ap.add_argument", 1)
            eval(src, {"_ap": ap, "GROUPS": {"exploratory": 0, "confirmatory": 0}, "int": int, "tuple": tuple})
    return ap


def actions(ap):
    return sorted((tuple(a.option_strings), a.dest, tuple(a.choices) if a.choices else None, a.default, a.type,
                   a.required) for a in ap._actions if a.dest != "help")


class LedgerTests(Tmp):
    def test_parser_is_run_stage2s(self):
        self.assertEqual(actions(LG.phase_parser()), actions(run_stage2_parser_actions()))

    def test_phase_keys_and_world_rules(self):
        k = lambda *a: LG.phase_key(LG.parse_phase_args(a))   # noqa: E731
        self.assertEqual(k("--phase", "policy_no_stall_train", "--seed", "3", "--world", "stall")["world"], "no_stall")
        self.assertEqual(k("--phase", "policy_train", "--seed", "3", "--world", "no_stall")["world"], "stall")
        self.assertEqual(k("--phase", "host_train", "--seed", "2")["world"], "stall")
        self.assertEqual(LG.phase_id(k("--phase", "host_train", "--seed", "2", "--world", "no_stall")),
                         "host_train_seed2_no_stall")
        self.assertEqual(LG.phase_id(k("--phase", "policy_train", "--se", "2")), "policy_train_seed2")
        for bad in (("--phase", "policy_train"), ("--phase", "nope", "--seed", "2"), ("--phase", "pooled"),
                    ("--phase", "deploy", "--seed", "12")):
            with self.subTest(args=bad), self.assertRaises(K.V7Refusal):
                k(*bad)
        for refused in (("--phase", "select_standin"), ("--phase", "detect_fit", "--seed", "2"),
                        ("--phase", "fresh_data", "--seed", "3"), ("--phase", "detect_marks", "--group", "exploratory")):
            with self.subTest(args=refused), self.assertRaises(K.V7Refusal):
                LG.check_phase_allowed(k(*refused))
        for ok in (("--phase", "policy_train", "--seed", "2"), ("--phase", "fresh_data", "--seed", "5"),
                   ("--phase", "detect_marks", "--group", "confirmatory", "--version", "weighted"),
                   ("--phase", "pooled", "--group", "exploratory")):
            LG.check_phase_allowed(k(*ok))

    def test_audit_hook_logs_writes_as_they_happen(self):
        n, path = LG.allocate_attempt(self.tmp / "r7", "host_train_seed6_stall")
        self.assertEqual((n, path.name), (1, "1.writes"))
        d = self.tmp / "w"
        proc = run_py(f"""
            import os, signal, numpy as np, v7_ledger as LG
            from pathlib import Path
            led = LG.Ledger({str(path)!r}); led.install_audit_hook()
            d = Path({str(d)!r}); d.mkdir()
            (d / "a.json").write_text("x"); (d / "b.msgpack").write_bytes(b"y"); np.savez(d / "c.npz", z=np.zeros(2))
            open(d / "e.log", "a").write("z"); os.close(os.open(d / "f.bin", os.O_WRONLY | os.O_CREAT))
            os.rename(d / "a.json", d / "a2.json"); os.replace(d / "a2.json", d / "a3.json")
            (d / "a3.json").read_text(); open(d / "b.msgpack", "rb").read()
            os.kill(os.getpid(), signal.SIGKILL)
            """)
        self.assertEqual(proc.returncode, -9)
        events, torn = LG.read_events(path)
        writes = {(e["ev"], Path(e.get("dst") or e["path"]).name) for e in events}
        for want in (("open-w", "a.json"), ("open-w", "b.msgpack"), ("open-w", "c.npz"), ("open-w", "e.log"),
                     ("open-w", "f.bin"), ("rename", "a2.json"), ("rename", "a3.json")):
            self.assertIn(want, writes)
        self.assertEqual(sum(1 for e in events if e["ev"] == "open-w" and e["path"].endswith("b.msgpack")), 1)
        self.assertEqual(torn, 0)
        self.assertEqual(LG.allocate_attempt(self.tmp / "r7", "host_train_seed6_stall")[0], 2)

    def test_classification(self):
        c = LG.classify_attempt
        self.assertEqual(c(True, 2, "b", "b")[0], "done")                      # a recorded stop exits 2
        self.assertEqual(c(True, 137, "b", "c")[0], "done")
        self.assertEqual(c(False, 137, "b", "c")[0], "machine-wide")           # the boot id changed
        self.assertEqual(c(False, None, "b", "c")[0], "machine-wide")
        self.assertEqual(c(False, 137, "b", "b", runner_signals=[9])[0], "machine-wide")
        for status in (129, 130, 143, -1, -2, -15):
            self.assertEqual(c(False, status, "b", "b")[0], "machine-wide", status)
        self.assertEqual(c(False, None, "b", "b")[0], "machine-wide")          # no exit status recorded
        for status in (137, 139, 135, 134, -9, -11):
            self.assertEqual(c(False, status, "b", "b")[0], "phase-specific", status)
        self.assertEqual(c(False, 1, "b", "b", log_tail="Traceback\n...\nMemoryError\nFAILED")[0], "phase-specific")
        self.assertEqual(c(False, 1, "b", "b", log_tail="jaxlib.xla_extension.XlaRuntimeError: RESOURCE_EXHAUSTED: x")[0],
                         "phase-specific")
        self.assertEqual(c(False, 97, "b", "b")[0], "v7-refusal")
        self.assertEqual(c(False, 97, "b", "b", log_tail="MemoryError\nREFUSED (v7, exit 97): tripwire")[0],
                         "v7-refusal")                                         # exit 97 before the memory patterns
        self.assertEqual(c(False, 1, "b", "b", log_tail="ValueError: x")[0], "failed")
        self.assertEqual(c(False, 0, "b", "b")[0], "failed")

    def test_phase_table_done_failed_partial(self):
        out = self.tmp / "out"
        key = dict(phase="policy_train", seed=6, world="stall")
        self.assertFalse(LG.phase_done(key, out)[0])
        (out / "policy" / "seed6").mkdir(parents=True)
        (out / "policy" / "seed6" / "actors.msgpack").write_bytes(b"m")
        (out / "policy" / "seed6" / "lock.json").write_text("{}")             # no sidecar: partial, never raises
        done, how, detail = LG.phase_done(key, out)
        self.assertFalse(done)
        self.assertEqual(detail["partial"], ["policy/seed6/lock.json"])
        Path(str(out / "policy" / "seed6" / "lock.json") + ".sha256").write_text("torn")
        self.assertFalse(LG.phase_done(key, out)[0])
        write_json_sidecar(out / "policy" / "seed6" / "stop.json", dict(stopped=True))
        self.assertEqual(LG.phase_done(key, out)[:2], (True, "failed-on-record"))
        fd = dict(phase="fresh_data", seed=7)
        write_json_sidecar(out / "fresh" / "stall" / "seed7" / "data.json", dict(passed=False))
        self.assertEqual(LG.phase_done(fd, out)[:2], (True, "failed-on-record"))
        fit = dict(phase="detect_fit", seed=7, world="stall", critic="fixed")
        write_json_sidecar(out / "detect/uniform/stall/fixed/seed7/band.json", dict(failed=False))
        self.assertFalse(LG.phase_done(fit, out)[0])                            # band.json written, tau.json not
        write_json_sidecar(out / "detect/uniform/stall/fixed/seed7/tau.json", dict(tau=1))
        self.assertEqual(LG.phase_done(fit, out)[:2], (True, "final"))
        st = dict(phase="detect_standins", seed=8, world="no_stall")
        for n in LG.STANDIN_NAMES[:4]:
            write_json_sidecar(out / f"detect_fresh/seed8/no_stall/scores/{n}.json", {})
        self.assertFalse(LG.phase_done(st, out)[0])                             # all five are needed
        write_json_sidecar(out / f"detect_fresh/seed8/no_stall/scores/{LG.STANDIN_NAMES[4]}.json", {})
        self.assertTrue(LG.phase_done(st, out)[0])
        self.assertEqual(K.read_record(out / "nothing.json")[0], "missing")

    def test_move_interrupted_row_only(self):
        out, r7 = self.tmp / "out", self.tmp / "r7"
        key = dict(phase="deploy", seed=6, world="stall")
        (out / "cache").mkdir(parents=True)
        (out / "cache" / "scores_stall_seed6_abc.npz").write_bytes(b"c")
        (out / "cache" / "scores_no_stall_seed6_abc.npz").write_bytes(b"other world")
        (out / "deploy" / "stall" / "seed6").mkdir(parents=True)
        (out / "deploy" / "stall" / "seed6" / "values.json").write_text("{")   # torn, no sidecar
        rec = LG.move_interrupted(key, out, r7, "phase-specific", "killed by SIGKILL")
        self.assertEqual(sorted(rec["files"]), ["cache/scores_stall_seed6_abc.npz", "deploy/stall/seed6/values.json"])
        self.assertFalse((out / "cache" / "scores_stall_seed6_abc.npz").exists())
        self.assertTrue((out / "cache" / "scores_no_stall_seed6_abc.npz").exists())
        moved = list((r7 / "interrupted" / "deploy_seed6_stall").rglob("*.npz"))
        self.assertEqual(len(moved), 1)
        self.assertEqual(moved[0].read_bytes(), b"c")

    def test_kill_boundaries(self):
        """C5: a phase killed at each boundary is not done and exactly its row's existing files move; a recorded
        failure is done (the runner never moves it)."""
        cases = [
            ("model file before its record", dict(phase="policy_train", seed=6, world="stall"),
             {"policy/seed6/actors.msgpack": b"m"}, False),
            ("a JSON before its sidecar", dict(phase="policy_train", seed=7, world="stall"),
             {"policy/seed7/actors.msgpack": b"m", "policy/seed7/lock.json": b"{}"}, False),
            ("the stalling data.json before the no-stall one", dict(phase="fresh_data", seed=6),
             {"fresh/stall/seed6/data.json": dict(passed=True)}, False),
            ("a failed stalling data.json", dict(phase="fresh_data", seed=7),
             {"fresh/stall/seed7/data.json": dict(passed=False)}, True),
            ("a non-failed band.json before tau.json", dict(phase="detect_fit", seed=6, world="stall", critic="fixed"),
             {"detect/uniform/stall/fixed/seed6/bank_scores.npz": b"s", "detect/uniform/stall/fixed/seed6/band.msgpack":
              b"b", "detect/uniform/stall/fixed/seed6/band.json": dict(failed=False)}, False),
            ("a failed band.json", dict(phase="detect_fit", seed=7, world="stall", critic="fixed"),
             {"detect/uniform/stall/fixed/seed7/band.json": dict(failed=True)}, True),
            ("a deploy cache file", dict(phase="deploy", seed=6, world="no_stall"),
             {"cache/scores_no_stall_seed6_k.npz": b"c"}, False),
            ("a recorded stop", dict(phase="policy_no_stall_train", seed=6, world="no_stall"),
             {"policy/seed6/no_stall/stop.json": dict(stopped=True)}, True),
        ]
        for name, key, files, done in cases:
            with self.subTest(case=name):
                out, r7 = self.tmp / name.replace(" ", "_") / "out", self.tmp / name.replace(" ", "_") / "r7"
                expect = []
                for rel, content in files.items():
                    p = out / rel
                    p.parent.mkdir(parents=True, exist_ok=True)
                    if isinstance(content, dict):
                        write_json_sidecar(p, content)
                        expect += [rel, rel + ".sha256"]
                    else:
                        p.write_bytes(content)
                        expect.append(rel)
                (out / "unrelated.json").write_text("{}")
                self.assertEqual(LG.phase_done(key, out)[0], done)
                self.assertEqual(LG.row_existing_files(key, out), sorted(expect))
                if not done:
                    rec = LG.move_interrupted(key, out, r7, "phase-specific", name)
                    self.assertEqual(sorted(rec["files"]), sorted(expect))
                    self.assertEqual(LG.row_existing_files(key, out), [])
                    self.assertTrue((out / "unrelated.json").exists())

    def test_finish_attempt_record(self):
        out, r7 = self.tmp / "out", self.tmp / "r7"
        key = dict(phase="host_train", seed=6, world="stall")
        n, path = LG.allocate_attempt(r7, "host_train_seed6_stall")
        LG.append_event(path, "launch-start", key=key, argv=["--phase", "host_train", "--seed", "6"], cpu=3, slot=1,
                        boot_id="boot", r7=str(r7), out_root=str(out))
        d = out / "host" / "stall" / "seed6"
        d.mkdir(parents=True)
        (d / "actors.msgpack").write_bytes(b"model")
        write_json_sidecar(d / "lock.json", dict(actor_sha256={}))
        stray = self.tmp / "elsewhere.txt"
        stray.write_text("s")
        for p in (d / "actors.msgpack", d / "lock.json", Path(str(d / "lock.json") + ".sha256"), stray):
            LG.append_event(path, "open-w", path=str(p), mode="wb")
        LG.append_event(path, "py-exit", code=0, vmhwm_kb=123, cpu_seconds=4.5)
        LG.append_event(path, "launch-end", status=0, boot_id="boot")
        rec = LG.finish_attempt(path, out_root=out, r7=r7)
        self.assertEqual(rec["classification"]["kind"], "done")
        self.assertEqual(rec["flagged_writes"], [str(stray)])
        self.assertEqual({Path(w["path"]).name: w["sha256"] for w in rec["writes"]}["actors.msgpack"],
                         hashlib.sha256(b"model").hexdigest())
        self.assertEqual((rec["vmhwm_kb"], rec["cpu_seconds"]), (123, 4.5))
        self.assertTrue(path.with_name("1.json").exists())
        self.assertEqual(LG.finish_attempt(path, out_root=out, r7=r7)["attempt"], 1)     # written once

    def test_shared_log_classifies_each_attempt_on_its_own_lines(self):
        """C5: every attempt of a phase appends to one log; each attempt is classified on its own lines only."""
        out, r7 = self.tmp / "out", self.tmp / "r7"
        key = dict(phase="host_train", seed=6, world="stall")
        log = r7 / "logs" / "phase_host_train_seed_6.log"
        log.parent.mkdir(parents=True)
        recs = []
        for status, lines in ((1, "Traceback (most recent call last):\nMemoryError: out of memory"),
                              (97, "REFUSED (v7, exit 97): tripwire: transition_matrix was called"),
                              (1, "ValueError: something unrelated")):
            n, path = LG.allocate_attempt(r7, "host_train_seed6_stall")
            LG.append_event(path, "launch-start", key=key, argv=["--phase", "host_train", "--seed", "6"], boot_id="b",
                            r7=str(r7), out_root=str(out))
            with open(log, "a") as h:
                h.write(f"2026-10-14T10:0{n}:00Z start: --phase host_train --seed 6 (cpu 3, slot 1, ledger {path})\n"
                        f"{lines}\n2026-10-14T10:0{n}:30Z FAILED (exit status {status}): --phase host_train --seed 6\n")
            LG.append_event(path, "launch-end", status=status, boot_id="b")
            recs.append(LG.finish_attempt(path, out_root=out, log_path=log, r7=r7))
            with open(log, "a") as h:
                h.write(f"host_train_seed6_stall attempt {n}: {recs[-1]['classification']['kind']}\n")
        self.assertEqual([r["classification"]["kind"] for r in recs], ["phase-specific", "v7-refusal", "failed"])
        self.assertTrue(all(r["log_segment_found"] for r in recs))
        # a log with no start line naming the attempt: no memory error is read from another attempt's lines
        n, path = LG.allocate_attempt(r7, "host_train_seed6_stall")
        LG.append_event(path, "launch-start", key=key, boot_id="b", r7=str(r7), out_root=str(out))
        LG.append_event(path, "launch-end", status=1, boot_id="b")
        rec = LG.finish_attempt(path, out_root=out, log_path=log, r7=r7)
        self.assertEqual((rec["classification"]["kind"], rec["log_segment_found"]), ("failed", False))

    def test_torn_record_is_moved_aside_and_written_again(self):
        out, r7 = self.tmp / "out", self.tmp / "r7"
        _, path = LG.allocate_attempt(r7, "host_train_seed6_stall")
        LG.append_event(path, "launch-start", key=dict(phase="host_train", seed=6, world="stall"), boot_id="b",
                        r7=str(r7), out_root=str(out))
        LG.append_event(path, "open-w", path=str(out / "x"), mode="wb")
        path.with_name("1.json").write_text('{"torn": tr')                        # killed between JSON and sidecar
        rec = LG.finish_attempt(path, out_root=out, r7=r7, now_boot="b")
        self.assertEqual(K.read_record(path.with_name("1.json"))[0], "ok")
        self.assertEqual(rec["kind"], "stage2-v7-ledger-attempt")
        moved = list((r7 / "interrupted" / "ledger_records" / "host_train_seed6_stall").rglob("1.json"))
        self.assertEqual([m.read_text() for m in moved], ['{"torn": tr'])
        self.assertIsNone(rec["end_utc"])
        self.assertIsNotNone(rec["end_utc_estimated"])                           # the write log's last line
        self.assertFalse(rec["handover"])

    def test_finish_without_end_is_machine_wide(self):
        out, r7 = self.tmp / "out", self.tmp / "r7"
        _, path = LG.allocate_attempt(r7, "host_train_seed6_stall")
        LG.append_event(path, "launch-start", key=dict(phase="host_train", seed=6, world="stall"), boot_id="old",
                        r7=str(r7), out_root=str(out))
        rec = LG.finish_attempt(path, out_root=out, r7=r7, now_boot="new")
        self.assertEqual(rec["classification"]["kind"], "machine-wide")


# ============================================================================================ C3, C4, C6, C10

class VerifyTests(Tmp):
    def test_environment_exact(self):
        VF.check_environment(ENV, K.ROOT, K.PYTHON)
        for name, env in (("stray", dict(ENV, OPENBLAS_CORETYPE="Haswell")), ("PWD", dict(ENV, PWD="/home/user/bca")),
                          ("XLA", dict(ENV, XLA_FLAGS="--x")), ("missing", {k: v for k, v in ENV.items()
                                                                            if k != "OMP_NUM_THREADS"}),
                          ("wrong", dict(ENV, JAX_PLATFORMS="gpu"))):
            with self.subTest(case=name), self.assertRaises(K.V7Refusal):
                VF.check_environment(env, K.ROOT, K.PYTHON)
        with self.assertRaises(K.V7Refusal):
            VF.check_environment(ENV, "/tmp", K.PYTHON)
        with self.assertRaises(K.V7Refusal):
            VF.check_environment(ENV, K.ROOT, "/usr/bin/python3")

    def test_one_processor(self):
        self.assertEqual(VF.check_one_processor(affinity={3}), 3)
        self.assertEqual(VF.check_one_processor(3, affinity={3}), 3)
        for aff, cpu in (({0, 1}, None), (set(range(4)), 0), ({2}, 3)):
            with self.subTest(aff=aff), self.assertRaises(K.V7Refusal):
                VF.check_one_processor(cpu, affinity=aff)

    def test_handoff_rules(self):
        v7 = self.tmp / "v7"
        prereg = VF.PreregV7(v7 / "prereg_v7.json", "0" * 64, dict(handoff=dict(handoff_sha256=None)))
        with self.assertRaises(K.V7Refusal) as cm:
            VF.check_handoff(prereg, v7)
        self.assertIn("handoff missing", str(cm.exception))
        sha = write_json_sidecar(v7 / "handoff" / "HANDOFF.json", dict(kind="stage2-v7-handoff",
                                                                       phases={"policy_train_seed2": "inherited"}))
        with self.assertRaises(K.V7Refusal):
            VF.check_handoff(prereg, v7)                                         # not the pinned one
        prereg.payload["handoff"]["handoff_sha256"] = sha
        VF.check_handoff(prereg, v7, "policy_train_seed3")
        with self.assertRaises(K.V7Refusal) as cm:
            VF.check_handoff(prereg, v7, "policy_train_seed2")
        self.assertIn("inherited", str(cm.exception))
        (v7 / "handoff" / "WITHDRAWN.json").write_text("{}")
        with self.assertRaises(K.V7Refusal) as cm:
            VF.check_handoff(prereg, v7, "policy_train_seed3")
        self.assertIn("withdrawn", str(cm.exception))

    def test_v7_code_and_v6_pins(self):
        v7 = self.tmp / "v7"
        (v7 / "code").mkdir(parents=True)
        (v7 / "code" / "a.py").write_text("x = 1\n")
        man = dict(files={"code/a.py": K.file_sha256(v7 / "code" / "a.py")})
        msha = write_json_sidecar(v7 / "code_manifest_v7.json", man)
        prereg = VF.PreregV7(v7 / "prereg_v7.json", "0" * 64, dict(code=dict(code_manifest_v7_sha256=msha)))
        VF.check_v7_code(prereg, v7)
        (v7 / "code" / "a.py").write_text("x = 2\n")
        with self.assertRaises(K.V7Refusal):
            VF.check_v7_code(prereg, v7)
        VF.check_v6_pins()
        self.assertEqual(VF.check_s_folder(), len(K.v6_manifest()["files"]))


# ============================================================================================ C10: approval route

class FakeGit:
    """A stand-in for v6's stage2_common.git (copied from S2C/test_stage2.py:627-699, unchanged in behaviour): no test
    runs git. It holds commits and origin's branch and answers exactly the read-only queries v6's module makes."""

    def __init__(self, author_re, branch="synthetic-branch", email="noreply@anthropic.com"):
        self.commits, self.refs, self.branch, self.email, self.calls, self.n = {}, {}, branch, email, [], 0
        self.author_re = author_re

    @property
    def ref(self):
        return "refs/remotes/origin/" + self.branch

    def commit(self, changes=None, deleted=(), author="Synthetic User <user@example.org>", committer=None,
               parent=None, push=True):
        parent = self.refs.get(self.ref) if parent is None else parent
        files = dict(self.commits[parent]["files"]) if parent else {}
        files.update(changes or {})
        for rel in deleted:
            files.pop(rel)
        self.n += 1
        h = hashlib.sha1(f"synthetic commit {self.n}".encode()).hexdigest()
        self.commits[h] = dict(parent=parent, files=files, author=author, committer=committer or author)
        if push:
            self.refs[self.ref] = h
        return h

    def history(self, h):
        out = []
        while h:
            out.append(h)
            h = self.commits[h]["parent"]
        return out

    def _rev(self, rev):
        return self.refs.get(rev, rev if rev in self.commits else None)

    def __call__(self, *args):
        self.calls.append(args)
        a = list(args)
        if a[0] == "rev-parse" and a[1:3] == ["--verify", "--quiet"] and a[3].endswith("^{commit}"):
            h = self._rev(a[3][:-len("^{commit}")])
            return (0, (h + "\n").encode()) if h else (1, b"")
        if a[0] == "cat-file" and a[1] == "-e" and a[2].endswith("^{commit}"):
            return (0, b"") if a[2][:-len("^{commit}")] in self.commits else (128, b"")
        if a[0] == "cat-file" and a[1] == "blob":
            rev, rel = a[2].split(":", 1)
            h = self._rev(rev)
            if h is None or rel not in self.commits[h]["files"]:
                return 128, b""
            return 0, self.commits[h]["files"][rel]
        if a[0] == "log" and a[1:3] == ["--no-renames", "--diff-filter=A"] and a[3] == "--format=%H":
            ref, sep, rel = a[4:7]
            h = self._rev(ref)
            if h is None:
                return 128, b""
            added = [c for c in self.history(h) if rel in self.commits[c]["files"] and not (
                self.commits[c]["parent"] and rel in self.commits[self.commits[c]["parent"]]["files"])]
            return 0, "".join(c + "\n" for c in added).encode()
        if a[0] == "log" and a[1] == "-1":
            c = self.commits.get(a[3])
            if c is None:
                return 128, b""
            if a[2] == "--format=%G?":
                return 0, b"N\n"
            (an, ae), (cn, ce) = (self.author_re.fullmatch(c[k]).groups() for k in ("author", "committer"))
            return 0, f"{an}\0{ae}\0{cn}\0{ce}\0{c['parent'] or ''}\n".encode()
        if a[0] == "merge-base" and a[1] == "--is-ancestor":
            h = self._rev(a[3])
            return (0, b"") if h and a[2] in self.history(h) else (1, b"")
        if a[0] == "config" and a[1:] == ["--get", "user.email"]:
            return (0, (self.email + "\n").encode()) if self.email else (1, b"")
        raise AssertionError(f"unexpected git query {args}")


class Tty(io.StringIO):
    def isatty(self):
        return True

    def fileno(self):
        raise OSError("no file descriptor (test)")


class NotTty(io.StringIO):
    def isatty(self):
        return False


def synthetic_v7_root(v7, final=True, extra_payload=None):
    """A committed v7 folder: approve_local_v7.py (the real one) at the root and in code/, one more file, the manifest
    and prereg_v7.json (final unless asked)."""
    (v7 / "code").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CODE / "approve_local_v7.py", v7 / "approve_local_v7.py")
    shutil.copyfile(CODE / "approve_local_v7.py", v7 / "code" / "approve_local_v7.py")
    (v7 / "code" / "x.py").write_text("x = 1\n")
    files = {rel: K.file_sha256(v7 / rel) for rel in ("approve_local_v7.py", "code/approve_local_v7.py", "code/x.py")}
    msha = write_json_sidecar(v7 / "code_manifest_v7.json", dict(kind=K.MANIFEST_KIND, files=files))
    payload = dict(kind=K.PREREG_KIND, label=K.LABEL, version=7, approvable=final, missing=[] if final else ["handoff"],
                   predecessor=dict(K.V6), code=dict(code_manifest_v7_sha256=msha, files=files),
                   approval=dict(local_terminal_allowed=True, branch="synthetic-branch", approver_email=K.APPROVER_EMAIL),
                   remake=dict(forecast_tolerance=K.W0_TOL),
                   handoff=dict(summary=["synthetic"], probe_records={}, remake_table=[], labels={}) if final else None,
                   changes=["c"], failures_v7_can_cause=["f"])
    payload.update(extra_payload or {})
    write_json_sidecar(v7 / "prereg_v7.json", payload)
    return VF.load_prereg_v7(v7 / "prereg_v7.json")


class ApprovalTests(Tmp):
    def setUp(self):
        super().setUp()
        self.v6c = K.load_v6_common()
        self.saved = self.v6c.git
        self.git = FakeGit(self.v6c._AUTHOR_RE)
        self.v6c.git = self.git
        self.v7 = self.tmp / "v7"
        self.prereg = synthetic_v7_root(self.v7)
        self.push_folder()

    def tearDown(self):
        self.v6c.git = self.saved
        super().tearDown()

    def rel(self, name):
        return K.V7_REL + "/" + name

    def push_folder(self):
        files = {self.rel(p.relative_to(self.v7).as_posix()): p.read_bytes() for p in self.v7.rglob("*") if p.is_file()}
        return self.git.commit(files, author="Claude <noreply@anthropic.com>")

    def local_approve(self, stdin=None, typed=None):
        out = io.StringIO()
        saved_out, saved_err, saved_git = sys.stdout, sys.stderr, AL7.git
        sys.stdout, sys.stderr, AL7.git = out, out, (lambda *a: None)
        try:
            code = AL7.main([], folder=str(self.v7), script_path=str(self.v7 / "approve_local_v7.py"),
                            stdin=Tty() if stdin is None else stdin, input_fn=lambda _: typed or self.prereg.prefix)
        finally:
            sys.stdout, sys.stderr, AL7.git = saved_out, saved_err, saved_git
        return code, out.getvalue()

    def user_push(self, author=APPROVER, committer=None, push=True):
        code, _ = self.local_approve()
        self.assertEqual(code, 0)
        f = self.v7 / K.APPROVAL_NAME.format(prefix=self.prereg.prefix)
        side = Path(str(f) + ".sha256")
        return self.git.commit({self.rel(f.name): f.read_bytes(), self.rel(side.name): side.read_bytes()},
                               author=author, committer=committer, push=push)

    def imp(self):
        return AP7.import_v7(self.v7, log=lambda *a: None, v6c=self.v6c)

    def push_record(self):
        rec = self.v7 / K.IMPORT_NAME.format(prefix=self.prereg.prefix)
        side = Path(str(rec) + ".sha256")
        return self.git.commit({self.rel(rec.name): rec.read_bytes(), self.rel(side.name): side.read_bytes()},
                               author="Claude <noreply@anthropic.com>")

    def refuses(self, pattern=None, wrote_nothing=True):
        before = sorted(p.name for p in self.v7.iterdir())
        with self.assertRaises((AP7.ApproveError, K.V7Refusal)) as cm:
            self.imp()
        if pattern:
            self.assertIn(pattern, str(cm.exception))
        self.assertEqual(sorted(p.name for p in self.v7.iterdir()), before)
        if wrote_nothing:
            self.assertFalse((self.v7 / K.IMPORT_NAME.format(prefix=self.prereg.prefix)).exists())

    def test_local_refuses_non_tty_and_draft(self):
        code, text = self.local_approve(stdin=NotTty())
        self.assertEqual(code, 2)
        self.assertIn("not a terminal", text)
        self.assertFalse(any(self.v7.glob("approval_v7_*")))
        code, text = self.local_approve(typed="0" * 16)
        self.assertEqual(code, 2)
        self.assertFalse(any(self.v7.glob("approval_v7_*")))
        draft = synthetic_v7_root(self.tmp / "draft", final=False)
        saved, sys.stderr = sys.stderr, io.StringIO()
        try:
            code = AL7.main([], folder=str(draft.path.parent), script_path=str(draft.path.parent / "approve_local_v7.py"),
                            stdin=Tty(), input_fn=lambda _: draft.prefix)
        finally:
            sys.stderr = saved
        self.assertEqual(code, 2)
        self.assertFalse(any(draft.path.parent.glob("approval_v7_*")))

    def test_local_shows_the_summary_and_writes_once(self):
        code, text = self.local_approve()
        self.assertEqual(code, 0)
        self.assertIn(self.prereg.sha256, text)
        self.assertIn(K.V6["prereg_sha256"], text)
        body = json.loads((self.v7 / K.APPROVAL_NAME.format(prefix=self.prereg.prefix)).read_text())
        VF.check_local_approval_v7(json.dumps(body).encode(), self.prereg)
        code, text = self.local_approve()
        self.assertEqual(code, 2)
        self.assertIn("already exists", text)

    def test_import_and_verify(self):
        commit = self.user_push()
        body = self.imp()
        self.assertEqual((body["commit"], body["author"]), (commit, APPROVER))
        with self.assertRaises(K.V7Refusal) as cm:                               # not on origin's branch yet
            VF.verify_v7_chain(self.prereg, self.v7, self.v6c)
        self.assertIn("not on origin's branch", str(cm.exception))
        self.push_record()
        got = VF.verify_v7_chain(self.prereg, self.v7, self.v6c)
        self.assertEqual(got["commit"], commit)
        self.assertTrue(all(q[0] in ("rev-parse", "cat-file", "log", "merge-base", "config") for q in self.git.calls))
        self.refuses("exists already", wrote_nothing=False)
        with self.assertRaises(K.V7Refusal):                                     # a later edit voids it
            write_json_sidecar(self.v7 / "prereg_v7.json", dict(self.prereg.payload, extra=1))
            VF.verify_v7_chain(VF.load_prereg_v7(self.v7 / "prereg_v7.json"), self.v7, self.v6c)

    def test_agent_identities_refused(self):
        # copied verbatim from S2C/test_stage2.py:864-870
        git_cases = {
            "not pushed": dict(push=False),
            "the author is the session": dict(author="Claude <noreply@anthropic.com>"),
            "the author is the session, other case": dict(author="Someone <NoReply@Anthropic.com>"),
            "the committer is the session": dict(committer="Claude <noreply@anthropic.com>"),
            "the session's configured email": dict(author="Agent <agent@session.example>"),
        }
        for why, kw in git_cases.items():
            with self.subTest(case=why):
                self.tearDown()
                self.setUp()
                self.git.email = "agent@session.example"
                self.user_push(**kw)
                self.refuses()

    def test_approver_positive_check(self):
        for why, kw in (("another author", dict(author="Synthetic User <user@example.org>")),
                        ("another committer", dict(committer="Synthetic User <user@example.org>"))):
            with self.subTest(case=why):
                self.tearDown()
                self.setUp()
                self.user_push(**kw)
                self.refuses("not v6's approver")
        self.assertRaises(VF.ChainProblem, VF.check_approver, dict(author=APPROVER, committer="X <x@y.z>"))

    def test_no_file_and_two_adding_commits(self):
        self.refuses("no approval of this fingerprint")
        self.tearDown()
        self.setUp()
        self.local_approve()
        self.refuses("exactly one commit")


# ============================================================================================ C7: the redirect

STUB_COMMON = '''"""Test stub of v6's stage2_common (synthetic fixture)."""
import hashlib, json, sys, types
from pathlib import Path
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "out"
EXPLORATORY_SEEDS = (0, 1, 2, 3, 4)
CONFIRMATORY_SEEDS = (5, 6, 7, 8, 9)
_MODULES = {}


def log(*a):
    print(*a, flush=True)


def load_room_harness():
    if "room_harness" in _MODULES:
        return _MODULES["room_harness"]
    path = HERE / "stub_harness.py"
    mod = types.ModuleType("room_harness")
    mod.__file__ = str(path)
    sys.modules["room_harness"] = mod
    exec(compile(path.read_bytes(), str(path), "exec"), mod.__dict__)
    _MODULES["room_harness"] = mod
    sys.modules["harness"] = mod
    return mod


def write_json(path, payload):
    path = Path(path)
    if path.exists():
        raise SystemExit(f"refused: {path} exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=1, sort_keys=True)
    path.write_text(text)
    path.with_name(path.name + ".sha256").write_text(hashlib.sha256(text.encode()).hexdigest() + "  " + path.name + "\\n")
'''

STUB_HARNESS = '''"""Test stub of the room harness's SeedContext loaders (the code of ROOM/harness.py:248-255 and :267-271 with
GATE/acceptance.py:1078-1079), reading a synthetic gate folder."""
import json
from pathlib import Path
import experiments.shift.learners as L
GATE = Path(__file__).resolve().parent.parent / "gate"
CALLS = []


class SeedContext:
    def __init__(self, rep, worlds=("stall",)):
        self.rep = int(rep)
        self.learn = L.LearnSettings()
        self.worlds = worlds

    def h2(self):
        CALLS.append(["h2", self.rep])
        d = GATE / f"seed{self.rep}"
        rec = json.loads((d / "values.json").read_text())
        acts = L.load_tree(d / "actors.msgpack", {k: L.actor_template() for k in rec["actor_names"]},
                           rec["actors_sha256"])
        return acts, rec, d

    def host(self):
        CALLS.append(["get_host", self.rep, "train=False"])
        d = GATE / f"seed{self.rep}" / "host"
        rec = json.loads((d / "record.json").read_text())
        state = L.load_tree(d / "snapshot_100.msgpack", L.agent_template(self.learn), rec["file_sha256"])
        return state, rec
'''

STUB_RUN = '''"""Test stub of run_stage2.py: what a phase reads through fresh.gate_context, written as a lock."""
import argparse, json, os, signal, sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage2_common as C
ap = argparse.ArgumentParser()
ap.add_argument("--phase")
ap.add_argument("--seed", type=int)
ap.add_argument("--world", default="stall")
a = ap.parse_args()
if a.phase in ("policy_train", "policy_no_stall_train", "host_train"):
    import fresh
    import experiments.shift.learners as L
    if a.phase == "policy_train":
        ctx = fresh.gate_context(a.seed)
        acts = dict(none=ctx.none(), full_none=ctx.full_none(), td3bc=ctx.td3bc())
        lock = C.OUT / "policy" / f"seed{a.seed}" / "lock.json"
    elif a.phase == "policy_no_stall_train":
        ctx = fresh.gate_context(a.seed, worlds=("stall", "no_stall"))
        acts = dict(none=ctx.none("no_stall"))
        lock = C.OUT / "policy" / f"seed{a.seed}" / "no_stall" / "lock.json"
    else:
        ctx = fresh.gate_context(a.seed)
        acts = dict(plain=ctx.host(a.world)[0].actor.params)
        lock = C.OUT / "host" / a.world / f"seed{a.seed}" / "lock.json"
    C.write_json(lock, dict(names=sorted(acts), actor_sha256={k: L.tree_sha256(v) for k, v in acts.items()}))
elif a.phase == "host_eval":
    import experiments.shift.thruster_chain as TC
    C.write_json(C.OUT / "host" / a.world / f"seed{a.seed}" / "values.json", dict(
        env=sorted(os.environ), argv=sys.argv[1:], cwd=os.getcwd(), affinity=sorted(os.sched_getaffinity(0)),
        path0=sys.path[0], pinned=bool(getattr(TC._matrices, "v7_pinned", False)),
        common=C.__file__, v7_folder_on_path=any("stage2_v7" in p for p in sys.path)))
elif a.phase == "deploy":
    (C.OUT / "cache").mkdir(parents=True, exist_ok=True)
    (C.OUT / "cache" / f"scores_{a.world}_seed{a.seed}_abc.npz").write_bytes(b"partial cache")
    os.kill(os.getpid(), signal.SIGKILL)
'''


class RedirectFixture:
    """A synthetic gate folder for seeds 2 and 3 (check 2's actors and the plain TD3+BC run), the laptop's reference
    files in P1a's and P2's formats (real L.save_tree files holding three, two and one actors), a stub scratchpad
    folder (stub stage2_common, stub harness, the REAL S2C/fresh.py, a stub run_stage2.py), a synthetic final
    prereg_v7.json with a re-make table, HANDOFF.json, and the test-mode config."""

    NAMES = ["full", "gapped", "no_stall_full", "no_stall_gapped", "random", "td3bc_gapped"]

    @classmethod
    def build(cls, root):
        L = learners()
        root = Path(root)
        root.mkdir(parents=True)
        (root / W.TEST_MARKER).write_text(W.TEST_MARKER_TEXT)
        s = root / "s"
        s.mkdir(parents=True)
        (s / "stage2_common.py").write_text(STUB_COMMON)
        (s / "stub_harness.py").write_text(STUB_HARNESS)
        (s / "run_stage2.py").write_text(STUB_RUN)
        shutil.copyfile(K.S2C / "fresh.py", s / "fresh.py")
        facts = dict(gate={}, refs={})
        for seed in (2, 3):
            g = root / "gate" / f"seed{seed}"
            acts = {n: perturbed(L.actor_template(), 10 * seed + i) for i, n in enumerate(cls.NAMES)}
            sha = L.save_tree(g / "actors.msgpack", acts)
            (g / "values.json").write_text(json.dumps(dict(actor_names=cls.NAMES, actors_sha256=sha,
                                                           failed={n: False for n in cls.NAMES})))
            host = perturbed(L.agent_template(L.LearnSettings()), 50 + seed)
            hsha = L.save_tree(g / "host" / "snapshot_100.msgpack", host)
            (g / "host" / "record.json").write_text(json.dumps(dict(file_sha256=hsha)))
            facts["gate"][str(seed)] = dict({n: L.tree_sha256(a) for n, a in acts.items()},
                                       host_actor=L.tree_sha256(host.actor.params))
        r7 = root / "r7"
        refs = r7 / "refs" / "seed2"
        stall = {n: perturbed(L.actor_template(), 100 + i) for i, n in enumerate(("full", "gapped", "random"))}
        nost = {n: perturbed(L.actor_template(), 200 + i) for i, n in enumerate(("no_stall_full", "no_stall_gapped"))}
        host2 = perturbed(L.agent_template(L.LearnSettings()), 300)
        f_stall = L.save_tree(refs / "check2_stall.msgpack", stall)
        f_nost = L.save_tree(refs / "check2_no_stall.msgpack", nost)
        f_host = L.save_tree(refs / "host_plain" / "snapshot_100.msgpack", host2)
        for pid in ("p1_seed2_stall", "p1_seed2_no_stall", "p2_seed2"):
            write_json_sidecar(r7 / "selftest" / f"{pid}.json", dict(kind="synthetic", probe=pid))
        rec_sha = {pid: K.file_sha256(r7 / "selftest" / f"{pid}.json")
                   for pid in ("p1_seed2_stall", "p1_seed2_no_stall", "p2_seed2")}
        facts["refs"] = dict({n: L.tree_sha256(a) for n, a in {**stall, **nost}.items()},
                             host_actor=L.tree_sha256(host2.actor.params))
        entries = [
            dict(phase="policy_train", seed=2, world="stall", reference_file="refs/seed2/check2_stall.msgpack",
                 file_sha256=f_stall, names_in_file=["full", "gapped", "random"],
                 replace={"gapped": dict(actor_sha256=facts["refs"]["gapped"], lock_name="none"),
                          "full": dict(actor_sha256=facts["refs"]["full"], lock_name="full_none")},
                 probe_record="p1_seed2_stall", probe_record_sha256=rec_sha["p1_seed2_stall"]),
            dict(phase="policy_no_stall_train", seed=2, world="no_stall",
                 reference_file="refs/seed2/check2_no_stall.msgpack", file_sha256=f_nost,
                 names_in_file=["no_stall_full", "no_stall_gapped"],
                 replace={"no_stall_gapped": dict(actor_sha256=facts["refs"]["no_stall_gapped"], lock_name="none")},
                 probe_record="p1_seed2_no_stall", probe_record_sha256=rec_sha["p1_seed2_no_stall"]),
            dict(phase="host_train", seed=2, world="stall", reference_file="refs/seed2/host_plain/snapshot_100.msgpack",
                 file_sha256=f_host, replace={"actor": dict(actor_sha256=facts["refs"]["host_actor"], lock_name="plain")},
                 probe_record="p2_seed2", probe_record_sha256=rec_sha["p2_seed2"]),
        ]
        v7 = root / "v7"
        (v7 / "code").mkdir(parents=True)
        (v7 / "code" / "dummy.txt").write_text("synthetic\n")
        msha = write_json_sidecar(v7 / "code_manifest_v7.json",
                                  dict(files={"code/dummy.txt": K.file_sha256(v7 / "code" / "dummy.txt")}))
        hsha = write_json_sidecar(v7 / "handoff" / "HANDOFF.json",
                                  dict(kind="stage2-v7-handoff", phases={"host_eval_seed3_stall": "inherited"}))
        payload = dict(kind=K.PREREG_KIND, label=K.LABEL, version=7, approvable=True, missing=[],
                       predecessor=dict(K.V6), constants=dict(file=K.CONSTANTS_REL, sha256=K.CONSTANTS_SHA256),
                       code=dict(code_manifest_v7_sha256=msha, files={}),
                       handoff=dict(handoff_sha256=hsha, remake_table=entries))
        write_json_sidecar(v7 / "prereg_v7.json", payload)
        (root / "config.json").write_text(json.dumps(dict(r7=str(r7), s_folder=str(s), v7_root=str(v7),
                                                          prereg_v7=str(v7 / "prereg_v7.json"), check_chain=False,
                                                          check_v6=False)))
        (root / "out").mkdir()
        return dict(root=str(root), s=str(s), r7=str(r7), v7=str(v7), entries=entries, payload=payload, facts=facts)


def gate_digest(root):
    return {p.relative_to(root).as_posix(): K.file_sha256(p) for p in sorted(Path(root, "gate").rglob("*"))
            if p.is_file()}


class RedirectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        TEST_ROOT.mkdir(parents=True, exist_ok=True)
        cls.tmp = Path(tempfile.mkdtemp(dir=TEST_ROOT))
        cls.fx = RedirectFixture.build(cls.tmp / "fx")
        cls.gate_before = gate_digest(cls.fx["root"])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def sub(self, body, fx=None):
        fx = fx or self.fx
        head = (f"import json, sys\nimport v7_refs as RF, v7_ledger as LG\n"
                f"FX = json.loads({json.dumps(json.dumps(fx))})\nS, R7 = FX['s'], FX['r7']\n"
                "def key(*a):\n    return LG.phase_key(LG.parse_phase_args(a))\n")
        return run_py(head + textwrap.dedent(body))

    def test_no_entry_imports_nothing(self):
        out = last_json(self.sub("""
            got = RF.install_for(FX["payload"], key("--phase", "policy_train", "--seed", "3"), S, R7)
            imported = "stage2_common" in sys.modules
            sys.path.insert(0, S)
            import stage2_common as C
            H = C.load_room_harness()
            print(json.dumps(dict(label=got["label"], imported=imported,
                                  untouched=not hasattr(H.SeedContext.h2, "v7_redirect")
                                  and not hasattr(H.SeedContext.host, "v7_redirect")
                                  and not hasattr(C.load_room_harness, "v7_redirect"))))
            """))
        self.assertEqual(out, dict(label=RF.LABEL_GATE, imported=False, untouched=True))

    def test_policy_entry_replaces_exactly_its_keys(self):
        out = last_json(self.sub("""
            got = RF.install_for(FX["payload"], key("--phase", "policy_train", "--seed", "2"), S, R7)
            sys.path.insert(0, S)
            import fresh
            import experiments.shift.learners as L
            ctx = fresh.gate_context(2)
            acts, rec, d = ctx.h2()
            H = sys.modules["room_harness"]
            print(json.dumps(dict(label=got["label"], shas={k: L.tree_sha256(v) for k, v in acts.items()},
                                  none=L.tree_sha256(ctx.none()), full_none=L.tree_sha256(ctx.full_none()),
                                  td3bc=L.tree_sha256(ctx.td3bc()), calls=H.CALLS,
                                  host_untouched=not hasattr(H.SeedContext.host, "v7_redirect"))))
            """))
        g, r = self.fx["facts"]["gate"]["2"], self.fx["facts"]["refs"]
        self.assertEqual(out["label"], RF.LABEL_REMADE)
        self.assertEqual(out["shas"]["gapped"], r["gapped"])
        self.assertEqual(out["shas"]["full"], r["full"])
        for k in ("random", "td3bc_gapped", "no_stall_gapped", "no_stall_full"):
            self.assertEqual(out["shas"][k], g[k], k)                           # every other key is the gate's
        self.assertEqual((out["none"], out["full_none"], out["td3bc"]), (r["gapped"], r["full"], g["td3bc_gapped"]))
        self.assertIn(["h2", 2], out["calls"])                                   # the original ran first
        self.assertTrue(out["host_untouched"])
        self.assertEqual(gate_digest(self.fx["root"]), self.gate_before)        # gate files byte-identical

    def test_no_stall_entry_matches_whatever_world_says(self):
        out = last_json(self.sub("""
            got = RF.install_for(FX["payload"], key("--phase", "policy_no_stall_train", "--seed", "2", "--world",
                                                    "stall"), S, R7)
            sys.path.insert(0, S)
            import fresh
            import experiments.shift.learners as L
            ctx = fresh.gate_context(2, worlds=("stall", "no_stall"))
            print(json.dumps(dict(replace=sorted(got["entry"]["replace"]), none=L.tree_sha256(ctx.none("no_stall")),
                                  gapped=L.tree_sha256(ctx.none("stall")))))
            """))
        self.assertEqual(out["replace"], ["no_stall_gapped"])
        self.assertEqual(out["none"], self.fx["facts"]["refs"]["no_stall_gapped"])
        self.assertEqual(out["gapped"], self.fx["facts"]["gate"]["2"]["gapped"])

    def test_host_entry_and_world_keys(self):
        out = last_json(self.sub("""
            own = RF.install_for(FX["payload"], key("--phase", "host_train", "--seed", "2", "--world", "no_stall"), S, R7)
            imported = "stage2_common" in sys.modules
            got = RF.install_for(FX["payload"], key("--phase", "host_train", "--seed", "2"), S, R7)
            sys.path.insert(0, S)
            import fresh
            import experiments.shift.learners as L
            ctx = fresh.gate_context(2)
            st, rec = ctx.host("stall")
            H = sys.modules["room_harness"]
            print(json.dumps(dict(own=own["label"], imported_for_no_stall=imported, label=got["label"],
                                  actor=L.tree_sha256(st.actor.params), calls=H.CALLS,
                                  h2_untouched=not hasattr(H.SeedContext.h2, "v7_redirect"))))
            """))
        self.assertEqual(out["own"], RF.LABEL_OWN)
        self.assertFalse(out["imported_for_no_stall"])
        self.assertEqual(out["label"], RF.LABEL_REMADE)
        self.assertEqual(out["actor"], self.fx["facts"]["refs"]["host_actor"])
        self.assertIn(["get_host", 2, "train=False"], out["calls"])             # the original get_host path ran
        self.assertTrue(out["h2_untouched"])

    def test_exit_97_inside_replaced_methods(self):
        cases = {
            "another seed": """
                RF.install_for(FX["payload"], key("--phase", "policy_train", "--seed", "2"), S, R7)
                sys.path.insert(0, S); import fresh
                fresh.gate_context(3).none()
                """,
            "reference changed after the check": """
                RF.install_for(FX["payload"], key("--phase", "host_train", "--seed", "2"), S, R7)
                from pathlib import Path
                p = Path(R7) / "refs/seed2/host_plain/snapshot_100.msgpack"
                p.write_bytes(p.read_bytes() + b"x")
                sys.path.insert(0, S); import fresh
                fresh.gate_context(2).host("stall")
                """,
            "per-actor pin": """
                e = json.loads(json.dumps(FX["payload"]))
                e["handoff"]["remake_table"][0]["replace"]["gapped"]["actor_sha256"] = "0" * 64
                RF.install_for(e, key("--phase", "policy_train", "--seed", "2"), S, R7)
                sys.path.insert(0, S); import fresh
                try:
                    fresh.gate_context(2).none()
                except Exception:
                    print("CAUGHT")
                """,
        }
        for name, body in cases.items():
            with self.subTest(case=name):
                fx = self.fx
                if name == "reference changed after the check":
                    root = self.tmp / "copy_for_change"
                    shutil.copytree(self.fx["root"], root)
                    fx = json.loads(json.dumps(self.fx).replace(self.fx["root"], str(root)))
                proc = self.sub(body, fx)
                self.assertEqual(proc.returncode, 97, proc.stdout[-1500:] + proc.stderr[-1500:])
                self.assertNotIn("CAUGHT", proc.stdout)

    def test_changed_gate_file_refuses_through_original_loader(self):
        root = self.tmp / "copy_gate"
        shutil.copytree(self.fx["root"], root)
        fx = json.loads(json.dumps(self.fx).replace(self.fx["root"], str(root)))
        p = root / "gate" / "seed2" / "actors.msgpack"
        p.write_bytes(p.read_bytes()[:-1] + b"\x00")
        proc = self.sub("""
            RF.install_for(FX["payload"], key("--phase", "policy_train", "--seed", "2"), S, R7)
            sys.path.insert(0, S); import fresh
            fresh.gate_context(2).none()
            """, fx)
        self.assertNotEqual(proc.returncode, 0)
        self.assertNotEqual(proc.returncode, 97)
        self.assertIn("SHA-256 differs from the recorded one", proc.stderr)      # v6's own refusal, unchanged

    def test_refused_entries_and_draft_tables(self):
        good = self.fx["entries"][0]
        bad = {
            "seed 0": dict(good, seed=0), "seed 1": dict(good, seed=1), "seed 5": dict(good, seed=5),
            "seed 9": dict(good, seed=9), "deploy": dict(good, phase="deploy"),
            "policy_eval": dict(good, phase="policy_eval"), "host_eval": dict(self.fx["entries"][2], phase="host_eval"),
            "no-stall host": dict(self.fx["entries"][2], world="no_stall"), "detection": dict(good, phase="detect_fit"),
            "pooled": dict(good, phase="pooled"), "a world mismatch": dict(good, world="no_stall"),
            "a wrong lock name": dict(good, replace={"gapped": dict(good["replace"]["gapped"], lock_name="plain"),
                                                     "full": good["replace"]["full"]}),
            "one key of the pair": dict(good, replace={"gapped": good["replace"]["gapped"]}),
            "another file": dict(good, reference_file="refs/seed3/check2_stall.msgpack"),
        }
        for name, e in bad.items():
            with self.subTest(case=name), self.assertRaises(K.V7Refusal):
                RF.table(dict(self.fx["payload"], handoff=dict(remake_table=[e])))
        with self.assertRaises(K.V7Refusal) as cm:
            RF.table(dict(self.fx["payload"], approvable=False, missing=["handoff"]))
        self.assertIn("draft", str(cm.exception))
        self.assertEqual(RF.table(dict(self.fx["payload"], approvable=False, missing=["handoff"], handoff=None)), [])
        self.assertEqual(len(RF.table(self.fx["payload"])), 3)

    def test_reference_pin_checked_before_hand_over(self):
        e = dict(self.fx["entries"][0], file_sha256="0" * 64)
        with self.assertRaises(K.V7Refusal) as cm:
            RF.install(e, self.fx["s"], self.fx["r7"])
        self.assertIn("differs from its pin", str(cm.exception))
        e = dict(self.fx["entries"][0], probe_record_sha256="0" * 64)
        with self.assertRaises(K.V7Refusal) as cm:
            RF.install(e, self.fx["s"], self.fx["r7"])
        self.assertIn("probe record", str(cm.exception))
        self.assertNotIn("stage2_common", sys.modules)


# ============================================================================================ launch_v7.sh end to end

class LaunchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        TEST_ROOT.mkdir(parents=True, exist_ok=True)
        cls.tmp = Path(tempfile.mkdtemp(dir=TEST_ROOT))
        cls.fx = RedirectFixture.build(cls.tmp / "fx")
        cls.root = Path(cls.fx["root"])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def launch(self, cpu, *args, env=None):
        caller = dict(ENV, **(env or {}))
        return run(["sh", LAUNCH, "--v7-test", self.root, cpu] + list(args), env=caller)

    def attempt_record(self, pid):
        recs = sorted((self.root / "r7" / "ledger" / pid).glob("*.json"), key=lambda p: int(p.name.split(".")[0]))
        return json.loads(recs[-1].read_text())

    def test_stub_phase_lock_records_remade_actors(self):
        proc = self.launch("0", "--phase", "policy_train", "--seed", "2")
        logs = "".join(p.read_text() for p in (self.root / "r7" / "logs").glob("*.log"))
        self.assertEqual(proc.returncode, 0, proc.stderr + logs[-3000:])
        lock = json.loads((self.root / "out" / "policy" / "seed2" / "lock.json").read_text())
        r, g = self.fx["facts"]["refs"], self.fx["facts"]["gate"]["2"]
        self.assertEqual(lock["actor_sha256"]["none"], r["gapped"])
        self.assertEqual(lock["actor_sha256"]["full_none"], r["full"])
        self.assertEqual(lock["actor_sha256"]["td3bc"], g["td3bc_gapped"])
        rec = self.attempt_record("policy_train_seed2")
        self.assertEqual(rec["classification"]["kind"], "done")
        self.assertTrue(rec["test_mode"])
        self.assertTrue(rec["reference_lock_check"]["ok"])
        self.assertEqual(rec["reference"]["label"], RF.LABEL_REMADE)
        self.assertTrue(rec["constants"]["differences"]["A_bit_identical"])
        self.assertEqual(rec["flagged_writes"], [])
        self.assertIn(str(self.root / "out" / "policy" / "seed2" / "lock.json"), [w["path"] for w in rec["writes"]])
        proc = self.launch("0", "--phase", "host_train", "--seed", "2")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        lock = json.loads((self.root / "out" / "host" / "stall" / "seed2" / "lock.json").read_text())
        self.assertEqual(lock["actor_sha256"]["plain"], r["host_actor"])
        self.assertTrue(self.attempt_record("host_train_seed2_stall")["reference_lock_check"]["ok"])

    def test_environment_exact_through_launch(self):
        proc = self.launch("1", "--phase", "host_eval", "--seed", "2",
                           env=dict(OPENBLAS_CORETYPE="Haswell", XLA_FLAGS="--xla_dump_to=/tmp/x", NPY_FOO="1",
                                    LD_PRELOAD="", BCA_SHIFT_DEPS="/x"))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        vals = json.loads((self.root / "out" / "host" / "stall" / "seed2" / "values.json").read_text())
        rec = self.attempt_record("host_eval_seed2_stall")
        self.assertEqual(rec["checks"]["environment"]["variables"], sorted(K.PHASE_ENV))   # exactly the eight
        self.assertFalse(set(vals["env"]) & {"OPENBLAS_CORETYPE", "XLA_FLAGS", "NPY_FOO", "LD_PRELOAD",
                                             "BCA_SHIFT_DEPS", "PWD", "OLDPWD"})
        self.assertEqual(vals["affinity"], [1])
        self.assertEqual(vals["cwd"], str(K.ROOT))
        self.assertEqual(vals["argv"], ["--phase", "host_eval", "--seed", "2"])
        self.assertEqual(vals["path0"], self.fx["s"])
        self.assertTrue(vals["pinned"])
        self.assertFalse(vals["v7_folder_on_path"])
        self.assertEqual(Path(vals["common"]).parent, Path(self.fx["s"]))

    def test_wrapper_refuses_stray_variable_and_two_processors(self):
        for name, env, cpus in (("stray", dict(ENV, OPENBLAS_CORETYPE="Haswell"), "0"),
                                ("PWD", dict(ENV, PWD="/home/user/bca"), "0"), ("two processors", ENV, "0,1")):
            with self.subTest(case=name):
                _, path = LG.allocate_attempt(self.root / "r7", "host_eval_seed3_no_stall")
                proc = run(["taskset", "-c", cpus, K.PYTHON, CODE / "run_stage2_v7.py", "--v7-ledger", path,
                            "--v7-cpu", "0", "--v7-test-config", self.root / "config.json", "--",
                            "--phase", "host_eval", "--seed", "3", "--world", "no_stall"], env=env)
                self.assertEqual(proc.returncode, 97, proc.stderr)
                events, _ = LG.read_events(path)
                self.assertEqual([e["ev"] for e in events if e["ev"] in ("v7-exit", "handover")], ["v7-exit"])
        self.assertFalse((self.root / "out" / "host" / "no_stall" / "seed3").exists())

    def test_test_mode_needs_every_path_inside_the_fixture(self):
        """The test seam never falls back to a production path and never runs v6's pinned code (C1, C6)."""
        fx = Path(tempfile.mkdtemp(dir=TEST_ROOT))
        try:
            (fx / W.TEST_MARKER).write_text(W.TEST_MARKER_TEXT)
            s = fx / "s"
            s.mkdir()
            (s / "run_stage2.py").write_text("# stub\n")
            full = dict(r7=str(fx / "r7"), s_folder=str(s), v7_root=str(fx / "v7"), prereg_v7=str(fx / "v7/p.json"),
                        check_chain=False, check_v6=False)
            cfgp = fx / "config.json"
            for k in ("r7", "s_folder", "v7_root", "prereg_v7"):
                with self.subTest(missing=k):
                    cfgp.write_text(json.dumps({x: v for x, v in full.items() if x != k}))
                    with self.assertRaisesRegex(K.V7Refusal, "explicitly"):
                        W.Config.from_test_file(cfgp)
            cfgp.write_text(json.dumps(full))
            cfg = W.Config.from_test_file(cfgp)
            self.assertEqual((cfg.r7, cfg.s_folder), (fx / "r7", s))
            (fx / "real").symlink_to(K.S_FOLDER)
            cfgp.write_text(json.dumps(dict(full, s_folder=str(fx / "real"))))
            with self.assertRaisesRegex(K.V7Refusal, "inside the fixture"):
                W.Config.from_test_file(cfgp)
            shutil.copyfile(K.S2C / "run_stage2.py", s / "run_stage2.py")      # v6's pinned entry point
            cfgp.write_text(json.dumps(full))
            with self.assertRaisesRegex(K.V7Refusal, "pinned run_stage2.py"):
                W.Config.from_test_file(cfgp)
            for place in (K.SCRATCH, K.ROOT / "outputs"):
                with self.subTest(place=place), self.assertRaisesRegex(K.V7Refusal, "inside or around"):
                    W.check_fixture_location(place)
        finally:
            shutil.rmtree(fx, ignore_errors=True)

    def test_inherited_phase_refused(self):
        proc = self.launch("0", "--phase", "host_eval", "--seed", "3")
        self.assertEqual(proc.returncode, 97)
        self.assertEqual(self.attempt_record("host_eval_seed3_stall")["classification"]["kind"], "v7-refusal")

    def test_processor_argument_one_integer(self):
        for cpu in ("0,1", "0-3", "", "0 1", "x"):
            with self.subTest(cpu=cpu):
                proc = self.launch(cpu, "--phase", "host_eval", "--seed", "4")
                self.assertEqual(proc.returncode, 97)
                self.assertIn("one integer", proc.stderr)

    def test_slot_cap_and_double_start(self):
        slots = self.root / "r7" / "slots"
        slots.mkdir(parents=True, exist_ok=True)
        held = []
        try:
            for i in range(1, 17):
                f = open(slots / f"slot_{i}.lock", "w")
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                held.append(f)
            proc = self.launch("0", "--phase", "host_eval", "--seed", "4")
            self.assertEqual(proc.returncode, 97)
            self.assertIn("all 16 slots are busy", proc.stderr)
        finally:
            for f in held:
                f.close()
        locks = self.root / "r7" / "locks"
        locks.mkdir(parents=True, exist_ok=True)
        with open(locks / "host_eval_seed4_stall.lock", "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            proc = self.launch("0", "--phase", "host_eval", "--seed", "4")
        self.assertEqual(proc.returncode, 97)
        self.assertIn("already running", proc.stderr)
        self.assertFalse((self.root / "out" / "host" / "stall" / "seed4").exists())

    def test_killed_phase_is_phase_specific_and_its_writes_listed(self):
        proc = self.launch("0", "--phase", "deploy", "--seed", "2", "--world", "no_stall")
        self.assertEqual(proc.returncode, 137)
        rec = self.attempt_record("deploy_seed2_no_stall")
        self.assertEqual(rec["classification"]["kind"], "phase-specific")
        self.assertEqual(rec["exit_status"], 137)
        cache = str(self.root / "out" / "cache" / "scores_no_stall_seed2_abc.npz")
        w = {x["path"]: x for x in rec["writes"]}
        self.assertTrue(w[cache]["in_row"])
        moved = LG.move_interrupted(rec["key"], self.root / "out", self.root / "r7", "phase-specific", "test")
        self.assertEqual(list(moved["files"]), ["cache/scores_no_stall_seed2_abc.npz"])


# ============================================================================================ C7: probes

class StubTree(dict):
    pass


def tree(w):
    return StubTree(w=float(w))


class StubL:
    def __init__(self, values, cotrain_failed=False, gate_bytes=b""):
        self.values, self.calls, self.cotrain_failed = values, [], cotrain_failed

    class Arm:
        def __init__(self, name, kind="even", value=1.0, source=None, index=0, critic=0):
            self.name, self.kind, self.value, self.critic = name, kind, value, critic

    def ActorOnly(self, prepared, start, settings, rep, *, alpha=None, steps=None, chunk=None):
        self.calls.append(("ActorOnly", prepared, start, settings, rep, dict(alpha=alpha, steps=steps, chunk=chunk)))
        L = self

        class AO:
            def run(self, arms, critics, *, weighted=True, log=None):
                L.calls.append(("run", [(a.name, a.critic) for a in arms], list(critics), dict(weighted=weighted)))
                return [tree(L.values[a.name]) for a in arms], dict(failed=[False] * len(arms), rows_sha256="r",
                                                                     start_sha256="s")

            def arm_params(self, stacked, i):
                return stacked[i]
        return AO()

    def tree_sha256(self, t):
        return hashlib.sha256(json.dumps(t, sort_keys=True).encode()).hexdigest()

    def save_tree(self, path, t):
        blob = json.dumps(t, sort_keys=True).encode()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(blob)
        return hashlib.sha256(blob).hexdigest()

    def cotrain(self, prepared, settings, rep, *, alpha=None, table=None, stream=0, updates=None, out_dir=None, log=None,
                name=None):
        self.calls.append(("cotrain", prepared, settings, rep, dict(alpha=alpha, table=table, stream=stream,
                                                                    updates=updates, out_dir=str(out_dir), name=name)))
        st = State(self.values["host_actor"], self.values["host_critic"])
        for n in ("snapshot_050.msgpack", "snapshot_100.msgpack"):
            self.save_tree(Path(out_dir) / n, dict(actor=st.actor.params, critic=st.critic.params, n=n))
        return dict(failed=self.cotrain_failed, snapshots=[]), st, {}


class State:
    def __init__(self, a, c):
        self.actor = type("T", (), dict(params=tree(a)))()
        self.critic = type("T", (), dict(params=tree(c)))()


class StubA:
    def __init__(self):
        self.calls = []

    def get_critic(self, prereg, sd, removal, controller, train=True, **kw):
        self.calls.append(("get_critic", sd, removal, controller, train))
        if train is not False:
            raise AssertionError("get_critic with train=True must never be called")
        return {}, f"critic:{sd}:{removal}"

    def get_host(self, prereg, sd, removal, alpha, arm="plain", table=None, train=True, **kw):
        self.calls.append(("get_host", sd, removal, alpha, arm, train))
        if train is not False:
            raise AssertionError("get_host with train=True must never be called")
        return dict(result=dict(snapshots=[dict(file_sha256="gate-snap")])), State(1.0, 2.0)

    def __getattr__(self, name):          # get_clone, h2_seed, stage_*, write_*: any other gate function
        raise AssertionError(f"the probes must never call acceptance.{name}")


def stub_harness(values, gate):
    A, L = StubA(), StubL(values)
    calls = []

    class Prep:
        def __init__(self, world):
            self.world, self.physics = world, f"physics:{world}"

        def __repr__(self):
            return f"Prep({self.world})"

        def __eq__(self, other):
            return isinstance(other, Prep) and other.world == self.world

    class SeedContext:
        def __init__(self, rep, worlds=("stall",)):
            calls.append(("SeedContext", rep, worlds))
            self.rep, self.prereg, self.learn = rep, "PREREG", "LEARN"
            self.sd = {w: f"sd:{w}" for w in ("stall", "no_stall")}

        def prepared(self, world="stall"):
            return Prep(world)

        def clone(self, world="stall"):
            calls.append(("clone read path", world))
            return tree(values[f"clone_{world}"]), {}

        def critic(self, world="stall", removal="gap"):
            return A.get_critic(self.prereg, self.sd[world], removal, "stall_aware", train=False)[1], {}

        def h2(self):
            return {k: tree(v) for k, v in gate.items()}, dict(failed={}, actors_sha256="g"), "gate-dir"

        def host(self):
            rec, state = A.get_host(self.prereg, self.sd["stall"], "gap", 5.0, "plain", train=False)
            return state, rec

    def ProgramConfig(physics="STALL", next_command="stall_aware"):
        calls.append(("ProgramConfig", physics, next_command))
        return ("cfg", physics, next_command)

    def build_shared(prep, critic, rep, *, cfg=None, need_harm=True, log_fn=None):
        calls.append(("build_shared", prep, critic, rep, cfg, need_harm))
        return {}, {}

    class RoomRunner:
        def __init__(self, prep, start, critic, settings, rep, *, cfg=None, shared=None, **kw):
            calls.append(("RoomRunner", prep, start, critic, settings, rep, cfg))

        def run(self, arms, log_fn=None):
            calls.append(("RoomRunner.run", [tuple(a) for a in arms]))
            return {a[0]: dict(params=tree(values["p1b"] if a[0] == "w0_check" else 9.0), finite=True) for a in arms}

    def RoomArm(name, kind, coef=(0.0,), pull=1.0):
        return (name, kind, tuple(coef), pull)

    def max_param_diff(a, b):
        return abs(a["w"] - b["w"])

    H = type("H", (), {})()
    for k, v in dict(A=A, L=L, SeedContext=SeedContext, ProgramConfig=ProgramConfig, build_shared=build_shared,
                     RoomRunner=RoomRunner, RoomArm=RoomArm, max_param_diff=max_param_diff, calls=calls).items():
        setattr(H, k, v)
    return H


BASE_VALUES = dict(gapped=1.0, full=2.0, random=3.0, no_stall_gapped=4.0, no_stall_full=5.0, none=1.0, full_none=2.0,
                   none_no_stall=4.0, p1b=1.0, clone_stall=0.5, clone_no_stall=0.6, host_actor=1.0, host_critic=2.0)
GATE_SAME = dict(gapped=1.0, full=2.0, random=3.0, no_stall_gapped=4.0, no_stall_full=5.0, td3bc_gapped=1.0)


class ProbeTests(Tmp):
    INPUTS = dict(remake_block_sha256="b")

    def test_p1_calls_carry_the_gates_arguments(self):
        H = stub_harness(BASE_VALUES, GATE_SAME)
        rec = PR.run_p1(H, 2, "stall", self.tmp, self.INPUTS)
        L, A = H.L, H.A
        self.assertEqual(L.calls[0][1:], (H.SeedContext(2).prepared("stall"), tree(0.5), "LEARN", 2,
                                          dict(alpha=5.0, steps=None, chunk=None)))
        self.assertEqual(L.calls[1][1], [("gapped", 0), ("full", 1), ("random", 2)])        # side by side, as the gate
        self.assertEqual(L.calls[1][2], ["critic:sd:stall:gap", "critic:sd:stall:none", "critic:sd:stall:random"])
        self.assertEqual(L.calls[1][3], dict(weighted=False))
        self.assertEqual(L.calls[3][1], [("none", 0), ("full_none", 1)])                    # P1c: S2C/fresh.py:245-246
        self.assertEqual(L.calls[3][2], ["critic:sd:stall:gap", "critic:sd:stall:none"])
        self.assertTrue(all(c[4] is False for c in A.calls if c[0] == "get_critic"))
        runner = [c for c in H.calls if c[0] == "RoomRunner"][0]
        self.assertEqual(runner[1:6], (H.SeedContext(2).prepared("stall"), tree(0.5), "critic:sd:stall:gap", "LEARN", 2))
        self.assertEqual([c for c in H.calls if c[0] == "RoomRunner.run"][0][1], [("w0_check", "plain", (0.0,), 1.0)])
        self.assertIn(("ProgramConfig", "STALL", "stall_aware"), H.calls)
        self.assertIn(("build_shared", H.SeedContext(2).prepared("stall"), "critic:sd:stall:gap", 2,
                       ("cfg", "STALL", "stall_aware"), False), H.calls)
        self.assertEqual(rec["p1a"]["names_written"], ["full", "gapped", "random"])
        ref = self.tmp / rec["p1a"]["reference_file"]["path"]
        self.assertEqual(json.loads(ref.read_bytes()), {"full": {"w": 2.0}, "gapped": {"w": 1.0}, "random": {"w": 3.0}})
        self.assertEqual(rec["forecast"]["F_ref"], 0.0)
        self.assertFalse(rec["p1a"]["actors"]["gapped"]["differs"])
        self.assertEqual(PR.result_line(rec), "bit-identical")
        with self.assertRaises(K.V7Refusal):                                     # a record is never replaced
            PR.run_p1(stub_harness(BASE_VALUES, GATE_SAME), 2, "stall", self.tmp, self.INPUTS)

    def test_p1_no_stall_group_and_p1c(self):
        H = stub_harness(BASE_VALUES, GATE_SAME)
        PR.run_p1(H, 3, "no_stall", self.tmp, self.INPUTS)
        self.assertEqual(H.L.calls[1][1], [("no_stall_gapped", 0), ("no_stall_full", 1)])
        self.assertEqual(H.L.calls[1][2], ["critic:sd:no_stall:gap", "critic:sd:no_stall:none"])
        self.assertEqual(H.L.calls[3][1], [("none_no_stall", 0)])                          # S2C/fresh.py:252-253
        self.assertIn(("ProgramConfig", "physics:no_stall", "stall_aware"), H.calls)
        self.assertEqual([c for c in H.calls if c[0] == "RoomRunner.run"][0][1],
                         [("w0_check", "plain", (0.0,), 1.0), ("pull_2", "plain", (0.0,), 2.0)])

    def test_p2_call_and_record(self):
        H = stub_harness(dict(BASE_VALUES, host_actor=1.5), GATE_SAME)
        rec = PR.run_p2(H, 4, self.tmp, self.INPUTS)
        call = [c for c in H.L.calls if c[0] == "cotrain"][0]
        self.assertEqual(call[1:4], (H.SeedContext(4).prepared("stall"), "LEARN", 4))
        self.assertEqual(call[4], dict(alpha=5.0, table=None, stream=0, updates=None,
                                       out_dir=str(self.tmp / "refs" / "seed4" / "host_plain"),
                                       name="TD3+BC plain (stall, gap log)"))
        self.assertIn(("get_host", "sd:stall", "gap", 5.0, "plain", False), H.A.calls)
        self.assertTrue(rec["actor"]["differs"])
        self.assertEqual(rec["actor"]["max_abs_diff"], 0.5)
        self.assertFalse(rec["td3bc_gapped"]["equals_gate_host_actor"] is None)
        self.assertTrue((self.tmp / "refs/seed4/host_plain/record.json").exists())
        self.assertEqual(PR.result_line(rec), "reference will be re-made if not inherited")

    def test_interrupted_probe_files_moved_then_rerun(self):
        ref = PR.p1_reference(self.tmp, 2, "stall")
        ref.parent.mkdir(parents=True)
        ref.write_bytes(b"partial from a killed probe")
        rec = PR.run_p1(stub_harness(BASE_VALUES, GATE_SAME), 2, "stall", self.tmp, self.INPUTS)
        self.assertEqual(list(rec["moved_aside"]["files"]), ["refs/seed2/check2_stall.msgpack"])
        moved = list((self.tmp / "interrupted" / "probe_p1_seed2_stall").rglob("check2_stall.msgpack"))
        self.assertEqual(moved[0].read_bytes(), b"partial from a killed probe")

    def test_never_reaches_a_training_or_writing_branch(self):
        H = stub_harness(BASE_VALUES, GATE_SAME)
        for name in ("get_clone", "h2_seed", "stage_harm_return", "write_prereg"):
            with self.assertRaises(AssertionError):
                getattr(H.A, name)
        PR.run_p1(H, 4, "stall", self.tmp, self.INPUTS)
        PR.run_p2(H, 4, self.tmp, self.INPUTS)
        self.assertTrue(all(c[-1] is False for c in H.A.calls))
        self.assertIn(("clone read path", "stall"), H.calls)


def p1_record(seed, world, differs=False, reproduces=False, f_ref=0.0, f_fresh=0.0, failed=False, block="b",
              ref_sha="0" * 64):
    keys = [n for n, _ in PR.P1A_ARMS[world]]
    hx = lambda t: hashlib.sha256(t.encode()).hexdigest()   # noqa: E731
    actors = {k: dict(laptop_sha256=hx(("L" if differs else "G") + k), gate_sha256=hx("G" + k), max_abs_diff=1e-3 if differs
                      else 0.0, differs=differs, laptop_failed=failed, gate_failed=False) for k in keys}
    p1b = dict(reproduces_gate=reproduces, max_abs_diff_vs_gate=f_ref, max_abs_diff_vs_p1a=f_ref,
               max_abs_diff_vs_p1c=f_fresh)
    return dict(kind="stage2-v7-probe-p1", probe=PR.probe_id("p1", seed, world), seed=seed, world=world,
                inputs=dict(remake_block_sha256=block),
                p1a=dict(names_written=sorted(keys), actors=actors,
                         reference_file=dict(path=f"refs/seed{seed}/check2_{world}.msgpack", sha256=ref_sha)),
                p1b=p1b, forecast=PR.forecast_of(actors, p1b, None, world))


def p2_record(seed, differs=False, failed=False, block="b", ref_sha="1" * 64):
    hx = lambda t: hashlib.sha256(t.encode()).hexdigest()   # noqa: E731
    actor = dict(laptop_sha256=hx("LA" if differs else "GA"), gate_sha256=hx("GA"), max_abs_diff=0.1 if differs else 0.0,
                 differs=differs, laptop_failed=failed, gate_failed=False)
    snaps = {n: dict(path=f"refs/seed{seed}/host_plain/{n}", sha256=ref_sha) for n in ("snapshot_050.msgpack",
                                                                                        "snapshot_100.msgpack")}
    return dict(kind="stage2-v7-probe-p2", probe=PR.probe_id("p2", seed), seed=seed, inputs=dict(remake_block_sha256=block),
                actor=actor, reference_file=snaps["snapshot_100.msgpack"], snapshots=snaps)


class RuleTests(unittest.TestCase):
    def records(self, **over):
        recs = {PR.probe_id("p1", s, w): p1_record(s, w) for s in PR.PROBE_SEEDS for w in PR.WORLDS}
        recs.update({PR.probe_id("p2", s): p2_record(s) for s in PR.PROBE_SEEDS})
        recs.update(over)
        return recs

    def table(self, recs, inherited=()):
        return PR.remake_table(recs, {pid: "f" * 64 for pid in recs}, inherited)

    def test_bit_identical_makes_no_entry(self):
        entries, labels = self.table(self.records())
        self.assertEqual(entries, [])
        self.assertEqual(labels["seed2 policy stall"], "laptop, gate's reference (v6 as written)")
        self.assertEqual(labels["seed3 host no_stall"], "laptop, own run")

    def test_p1b_reproducing_keeps_the_gates_file(self):
        recs = self.records(p1_seed2_stall=p1_record(2, "stall", differs=True, reproduces=True))
        entries, labels = self.table(recs)
        self.assertEqual(entries, [])
        self.assertEqual(labels["seed2 policy stall"], "laptop, gate's reference kept (P1b reproduced it)")
        self.assertEqual(PR.result_line(recs["p1_seed2_stall"]), "gate's file kept (P1b reproduced it)")

    def test_entries_kinds_and_inheritance(self):
        recs = self.records(p1_seed2_stall=p1_record(2, "stall", differs=True),
                            p1_seed3_no_stall=p1_record(3, "no_stall", differs=True),
                            p2_seed4=p2_record(4, differs=True), p1_seed4_stall=p1_record(4, "stall", differs=True))
        entries, labels = self.table(recs, inherited={"policy_train_seed4", "deploy_seed4_stall"})
        got = {(e["phase"], e["seed"], e["world"]): e for e in entries}
        self.assertEqual(sorted(got), [("host_train", 4, "stall"), ("policy_no_stall_train", 3, "no_stall"),
                                       ("policy_train", 2, "stall")])
        self.assertEqual({k: v["lock_name"] for k, v in got[("policy_train", 2, "stall")]["replace"].items()},
                         {"gapped": "none", "full": "full_none"})
        self.assertEqual(got[("host_train", 4, "stall")]["replace"]["actor"]["lock_name"], "plain")
        for e in entries:
            RF.validate_entry(e)
        self.assertEqual(labels["seed4 policy stall"], "inherited (cloud, v6)")
        self.assertEqual(labels["seed4 deploy stall"], "inherited (cloud, v6)")
        self.assertEqual(labels["seed4 deploy no_stall"], "laptop, gate's reference (v6 as written)")
        self.assertEqual(labels["seed2 policy stall"], "laptop, reference re-made on the laptop")

    def test_remake_failed(self):
        recs = self.records(p1_seed2_stall=p1_record(2, "stall", differs=True, failed=True),
                            p2_seed3=p2_record(3, differs=True, failed=True))
        entries, labels = self.table(recs)
        self.assertEqual(entries, [])
        self.assertEqual(labels["seed2 policy stall"], "laptop, re-make failed")
        self.assertEqual(labels["seed3 host stall"], "laptop, re-make failed")

    def test_forecast(self):
        recs = self.records(p1_seed2_stall=p1_record(2, "stall", f_ref=3e-6),
                            p1_seed3_no_stall=p1_record(3, "no_stall", f_fresh=1e-5))
        fc = PR.forecast_all(recs)
        self.assertFalse(fc["not_ready"])                                        # reported, stops nothing
        self.assertEqual(fc["rows"]["p1_seed2_stall"]["F_ref"], 3e-6)
        recs["p1_seed4_stall"] = p1_record(4, "stall", f_fresh=2e-5)
        fc = PR.forecast_all(recs)
        self.assertTrue(fc["not_ready"])
        self.assertIn("NOT READY (P1b: check (b) forecast", fc["line"])
        self.assertFalse(PR.forecast_all(recs, q7="proceed")["not_ready"])
        recs["p1_seed4_stall"] = p1_record(4, "stall", f_ref=None)
        self.assertTrue(PR.forecast_all(recs)["not_ready"])                     # no comparison counts as above

    def test_p3_compare(self):
        self.assertTrue(PR.p3_compare([dict(tree_sha256="a", bytes_sha256="b")] * 6)[0])
        self.assertFalse(PR.p3_compare([dict(tree_sha256="a", bytes_sha256="b"),
                                        dict(tree_sha256="a", bytes_sha256="c")])[0])


class RemakeTests(Tmp):
    def test_remake_lands_at_the_original_path(self):
        H = stub_harness(BASE_VALUES, GATE_SAME)
        rec = PR.run_p1(H, 2, "stall", self.tmp, ProbeTests.INPUTS)
        rel = rec["p1a"]["reference_file"]["path"]
        pin = rec["p1a"]["reference_file"]["sha256"]
        (self.tmp / rel).write_bytes(b"damaged")
        out = PR.remake(stub_harness(BASE_VALUES, GATE_SAME), "p1_seed2_stall", self.tmp, {rel: pin})
        self.assertEqual(out["renamed"], [rel])
        self.assertEqual(K.file_sha256(self.tmp / rel), pin)
        moved = list((self.tmp / "interrupted" / "refs").rglob("check2_stall.msgpack"))
        self.assertEqual([p.read_bytes() for p in moved], [b"damaged"])
        self.assertEqual(PR.remake(H, "p1_seed2_stall", self.tmp, {rel: pin})["damaged"], [])

    def test_remake_with_another_sha_stays_halted(self):
        H = stub_harness(BASE_VALUES, GATE_SAME)
        rec = PR.run_p1(H, 3, "stall", self.tmp, ProbeTests.INPUTS)
        rel, pin = rec["p1a"]["reference_file"]["path"], rec["p1a"]["reference_file"]["sha256"]
        (self.tmp / rel).write_bytes(b"damaged")
        with self.assertRaises(K.V7Refusal) as cm:
            PR.remake(stub_harness(dict(BASE_VALUES, gapped=1.25), GATE_SAME), "p1_seed3_stall", self.tmp, {rel: pin})
        self.assertIn("stays halted", str(cm.exception))
        self.assertFalse((self.tmp / rel).exists())                             # never a second, different reference


# ============================================================================================ C10: the writers

def complete_v7_tree(v7):
    """A copy of the real V7 code and constants, with placeholder files for everything another step writes."""
    shutil.copytree(CODE, v7 / "code", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for n in ("world_constants.json", "world_constants.json.sha256"):
        shutil.copyfile(K.V7 / n, v7 / n)
    for rel in WM.REQUIRED:
        p = v7 / rel
        if p.exists() or rel in ("approve_local_v7.py", WM.DIFF_REL):
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        if rel in WM.BYTE_COPIES:
            shutil.copyfile(K.ROOT / WM.BYTE_COPIES[rel], p)
        else:
            p.write_text(f"placeholder for {rel}\n")


def synthetic_records(r7, block, probe_sha, differs=()):
    for s in PR.PROBE_SEEDS:
        for w in PR.WORLDS:
            pid = PR.probe_id("p1", s, w)
            ref = Path(r7) / f"refs/seed{s}/check2_{w}.msgpack"
            ref.parent.mkdir(parents=True, exist_ok=True)
            ref.write_bytes(pid.encode())
            rec = p1_record(s, w, differs=pid in differs, block=block, ref_sha=K.file_sha256(ref))
            rec["inputs"].update(constants_sha256=K.CONSTANTS_SHA256, gate_prereg_sha256=K.GATE_PREREG_SHA256,
                                 v6_code_manifest_sha256=K.V6["code_manifest_sha256"], probe_script_sha256=probe_sha)
            write_json_sidecar(PR.record_path(r7, pid), rec)
        pid = PR.probe_id("p2", s)
        d = Path(r7) / f"refs/seed{s}/host_plain"
        d.mkdir(parents=True, exist_ok=True)
        for n in ("snapshot_050.msgpack", "snapshot_100.msgpack"):
            (d / n).write_bytes(b"snap")
        rec = p2_record(s, differs=pid in differs, block=block, ref_sha=K.file_sha256(d / "snapshot_100.msgpack"))
        rec["inputs"].update(constants_sha256=K.CONSTANTS_SHA256, gate_prereg_sha256=K.GATE_PREREG_SHA256,
                             v6_code_manifest_sha256=K.V6["code_manifest_sha256"], probe_script_sha256=probe_sha)
        write_json_sidecar(PR.record_path(r7, pid), rec)


class WriterTests(Tmp):
    def setUp(self):
        super().setUp()
        self.v7, self.r7 = self.tmp / "v7", self.tmp / "r7"
        complete_v7_tree(self.v7)
        self.man = WM.main([], v7_root=self.v7, log=lambda *a: None)

    def test_manifest(self):
        self.assertEqual(self.man["missing_required"], [])
        self.assertEqual((self.v7 / "approve_local_v7.py").read_bytes(), (CODE / "approve_local_v7.py").read_bytes())
        diff = (self.v7 / WM.DIFF_REL).read_text()
        self.assertIn("+exec 8>\"$R7/locks/$PHASE_ID.lock\"", diff)
        self.assertIn("-nice -n 5 taskset", diff)
        self.assertNotIn("prereg_v7.json", self.man["files"])
        again = WM.main([], v7_root=self.v7, log=lambda *a: None)
        self.assertEqual(K.canonical(again), K.canonical(self.man))              # no time: the same bytes
        (self.v7 / "laptop/lib/envfacts.py").write_text("changed\n")
        with self.assertRaises(SystemExit):
            WM.main([], v7_root=self.v7, log=lambda *a: None)

    def test_draft_then_final(self):
        body, digest = WP.write_draft(self.v7, self.r7, log=lambda *a: None)
        self.assertFalse(body["approvable"])
        self.assertEqual(body["missing"], ["handoff"])
        self.assertEqual(body["predecessor"], K.V6)
        self.assertEqual(body["constants"]["sha256"], K.CONSTANTS_SHA256)
        self.assertEqual(body["remake"]["q7_if_forecast_above_tolerance"], "stop")
        prereg = VF.load_prereg_v7(self.v7 / "prereg_v7.json")
        with self.assertRaises(K.V7Refusal):
            VF.require_final(prereg)
        ok, why = AP7.approvable_now(prereg, self.v7)
        self.assertFalse(ok)
        block = K.block_sha256(body["remake"])
        synthetic_records(self.r7, block, self.man["files"]["code/v7_probes.py"],
                          differs={"p1_seed2_stall", "p2_seed3"})
        with self.assertRaisesRegex(K.V7Refusal, "can no longer change"):        # Q7 cannot change any more
            WP.write_draft(self.v7, self.r7, q7="proceed", q7_answered=True, q7_quote="proceed",
                           q7_utc="2026-10-14T09:00:00Z", log=lambda *a: None)
        _, digest = WP.write_draft(self.v7, self.r7, log=lambda *a: None)       # the same block: allowed
        hand = self.v7 / "handoff" / "HANDOFF.json"
        write_json_sidecar(hand, dict(kind="stage2-v7-handoff", snapshot=dict(commit="c" * 40),
                                      phases={"policy_train_seed4": "inherited"}, stop_rule="stop at once",
                                      draft_prereg_v7_sha256=digest, selftest=READY_SELFTEST))
        final, fdigest = WP.write_final(hand, self.v7, self.r7, log=lambda *a: None)
        self.assertEqual(final["handoff"]["selftest"], READY_SELFTEST)
        self.assertTrue(final["approvable"])
        self.assertEqual(WP.non_handoff(final), WP.non_handoff(body))
        kinds = sorted((e["phase"], e["seed"]) for e in final["handoff"]["remake_table"])
        self.assertEqual(kinds, [("host_train", 3), ("policy_train", 2)])
        self.assertEqual(final["handoff"]["labels"]["seed4 policy stall"], "inherited (cloud, v6)")
        self.assertEqual(len(final["handoff"]["probe_records"]), 9)
        prereg = VF.load_prereg_v7(self.v7 / "prereg_v7.json")
        self.assertEqual(len(RF.table(prereg.payload)), 2)
        self.assertTrue(AP7.approvable_now(prereg, self.v7)[0])
        WP.write_final(hand, self.v7, self.r7, log=lambda *a: None)              # rewriting a final before approval
        prereg = VF.load_prereg_v7(self.v7 / "prereg_v7.json")
        (self.v7 / K.APPROVAL_NAME.format(prefix=prereg.prefix)).write_text("{}")
        for fn in (lambda: WP.write_final(hand, self.v7, self.r7, log=lambda *a: None),
                   lambda: WP.write_draft(self.v7, self.r7, log=lambda *a: None)):
            with self.assertRaises(K.V7Refusal):
                fn()
        with self.assertRaises(SystemExit):
            WM.main([], v7_root=self.v7, log=lambda *a: None)

    def test_q7_answer_needs_the_users_words(self):
        with self.assertRaisesRegex(K.V7Refusal, "recorded answer"):              # 'proceed' is never inferred
            WP.write_draft(self.v7, self.r7, q7="proceed", log=lambda *a: None)
        for quote, utc in ((None, "2026-10-14T09:00:00Z"), ("proceed", None), ("proceed", "2026-10-14 09:00")):
            with self.subTest(quote=quote, utc=utc), self.assertRaises(K.V7Refusal):
                WP.write_draft(self.v7, self.r7, q7="proceed", q7_answered=True, q7_quote=quote, q7_utc=utc,
                               log=lambda *a: None)
        body, _ = WP.write_draft(self.v7, self.r7, log=lambda *a: None)
        self.assertTrue(body["remake"]["q7_status"].startswith("open"))
        self.assertIsNone(body["remake"]["q7_user_message"])
        body, _ = WP.write_draft(self.v7, self.r7, q7="stop", q7_answered=True, q7_quote="keep the default: stop",
                                 q7_utc="2026-10-14T09:00:00Z", log=lambda *a: None)
        self.assertEqual(body["remake"]["q7_status"], WP.Q7_ANSWERED)
        self.assertEqual(body["remake"]["q7_user_message"], dict(quote="keep the default: stop",
                                                                 utc="2026-10-14T09:00:00Z"))

    def test_final_refusals(self):
        body, dsha = WP.write_draft(self.v7, self.r7, log=lambda *a: None)
        hand = self.v7 / "handoff" / "HANDOFF.json"
        write_json_sidecar(hand, dict(kind="stage2-v7-handoff", phases={}, draft_prereg_v7_sha256=dsha,
                                      selftest=READY_SELFTEST))
        with self.assertRaises(K.V7Refusal):                                     # no probe records
            WP.write_final(hand, self.v7, self.r7, log=lambda *a: None)
        synthetic_records(self.r7, "another block", self.man["files"]["code/v7_probes.py"])
        with self.assertRaises(K.V7Refusal) as cm:
            WP.write_final(hand, self.v7, self.r7, log=lambda *a: None)
        self.assertIn("does not bind", str(cm.exception))
        shutil.rmtree(self.r7)
        synthetic_records(self.r7, K.block_sha256(body["remake"]), self.man["files"]["code/v7_probes.py"])
        rec = json.loads(PR.record_path(self.r7, "p1_seed3_no_stall").read_text())
        rec["forecast"]["above_tolerance"] = ["F_fresh"]
        rec["forecast"]["F_fresh"] = 5e-5
        write_json_sidecar(PR.record_path(self.r7, "p1_seed3_no_stall"), rec)
        with self.assertRaises(K.V7Refusal) as cm:
            WP.write_final(hand, self.v7, self.r7, log=lambda *a: None)
        self.assertIn("NOT READY (P1b", str(cm.exception))
        rec["forecast"]["above_tolerance"], rec["forecast"]["F_fresh"] = [], 0.0
        write_json_sidecar(PR.record_path(self.r7, "p1_seed3_no_stall"), rec)
        (self.r7 / "refs/seed2/check2_stall.msgpack").write_bytes(b"damaged")
        with self.assertRaises(K.V7Refusal) as cm:
            WP.write_final(hand, self.v7, self.r7, log=lambda *a: None)
        self.assertIn("HALT: reference", str(cm.exception))
        (self.r7 / "refs/seed2/check2_stall.msgpack").write_bytes(b"p1_seed2_stall")
        for name, over, msg in (("another draft", dict(draft_prereg_v7_sha256="e" * 64), "new handoff"),
                                ("no self-test", dict(selftest=None), "P3 passed"),
                                ("P3 failed", dict(selftest=dict(READY_SELFTEST, p3=dict(ok=False))), "P3 passed"),
                                ("NOT READY", dict(selftest=dict(READY_SELFTEST, verdict="NOT READY (gate)")),
                                 "P3 passed")):
            with self.subTest(case=name):
                for q in (hand, Path(str(hand) + ".sha256")):
                    if q.exists():
                        q.unlink()
                write_json_sidecar(hand, dict(dict(kind="stage2-v7-handoff", phases={}, draft_prereg_v7_sha256=dsha,
                                                   selftest=READY_SELFTEST), **over))
                with self.assertRaisesRegex(K.V7Refusal, msg):
                    WP.write_final(hand, self.v7, self.r7, log=lambda *a: None)


READY_SELFTEST = dict(form="full", verdict="READY (waiting for handoff/approval)", p3=dict(ok=True, distinct=1),
                      import_record="handoff/probes/selftest_import_x.json", import_record_sha256="a" * 64)


class ApprovalScreenTests(unittest.TestCase):
    def test_summary_shows_selftest_q7_and_forecast(self):
        remake = dict(forecast_tolerance=1e-5, q7_if_forecast_above_tolerance="proceed", q7_status=WP.Q7_ANSWERED,
                      q7_user_message=dict(quote="go on anyway", utc="2026-10-14T09:00:00Z"))
        handoff = dict(summary=[], selftest=dict(READY_SELFTEST, selftest_utc="t1", imported_utc="t2"),
                       forecast=dict(above=["p1_seed2_stall"], line="READY as to the check (b) forecast (gaps above "
                                     "1e-5 accepted under Q7: p1_seed2_stall)"),
                       probe_records={}, remake_table=[], labels={})
        payload = dict(label="l", changes=[], failures_v7_can_cause=[], remake=remake, handoff=handoff)
        text = AL7.summary(dict(prereg=payload, n_files=1, sha256="a" * 64, v6_sha256="b" * 64))
        for want in ("repeatability test P3", "passed", "READY (waiting", "'proceed' (answered by the user)",
                     '"go on anyway" (2026-10-14T09:00:00Z)', "accepted under Q7", "WARNING", "ABOVE 1e-05"):
            self.assertIn(want, text)
        handoff.update(forecast=dict(above=[], line="READY as to the check (b) forecast"),
                       selftest=dict(READY_SELFTEST, p3=dict(ok=False)))
        remake.update(q7_if_forecast_above_tolerance="stop", q7_status=WP.Q7_OPEN, q7_user_message=None)
        text = AL7.summary(dict(prereg=payload, n_files=1, sha256="a" * 64, v6_sha256="b" * 64))
        self.assertNotIn("WARNING", text)
        self.assertIn("NOT PASSED", text)
        self.assertIn("Every forecast gap is at most 1e-05", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
