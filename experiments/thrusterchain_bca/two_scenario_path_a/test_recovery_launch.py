import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import recovery_launch as R

AMENDMENT = "a" * 64


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value), encoding="utf-8")


def progress_line(step, critic, actor, aux, cal):
    counts = {"critic": critic, "actor": actor, "auxiliary": aux, "calibrations": cal}
    return json.dumps({"local_step": step, "physical_counts": {"attempted": counts, "completed": counts,
                       "execution_counts_known": True}, "observed_est": "2026-11-02T05:%02d:00-05:00" % (step % 60)})


class ZeusFixture:
    """Byte-level stand-in for Zeus's smoke and interrupted production attempt."""

    def __init__(self, root):
        self.root = Path(root)
        self.smoke, self.run = self.root / "smoke_attempt_1", self.root / "run_amendment_1"
        for name in ("RESULT.json", "WORKER_EXIT.json", "report.json", "ARTIFACTS.json"):
            write(self.smoke / name, {"name": name, "elapsed_seconds": 400})
        self.proof = {n: R.L.sha(self.smoke / n) for n in ("RESULT.json", "WORKER_EXIT.json", "report.json", "ARTIFACTS.json")}
        self.smoke_report = {"learning": {"counts": {"completed": dict(R.S.LEARNING_CAPS)}},
                             "physical": {"counts": {"completed": {"prefix_parity": 2688, "bank_edges": 288, "forced_qpi": 100, "returns": 150}}},
                             "runtime": {"device_kinds": ["Zeus GPU"]}}
        write(self.run / "ATTEMPT.json", {"mode": "production", "smoke_proof": self.proof, "amendment_manifest_sha256": AMENDMENT})
        warm = self.run / R.REUSED
        self.job(warm, "warmup")
        (warm / "full_state.msgpack").write_bytes(b"zeus-warmup-state")
        write(warm / "checkpoint.json", {"arm": "warmup", "condition": "honeypot", "seed_index": 0, "state_digest": "d" * 64,
              "checkpoint_sha256": R.L.sha(warm / "full_state.msgpack"), "counts": {"critic": 10000, "refresh_accepted": 1}})
        write(warm / "report.json", {"status": "completed_training_job", "execution_counts_known": True, "simulator_steps": 0,
              "condition": "honeypot", "seed_index": 0, "arm": "warmup", "elapsed_seconds": 512.5,
              "counts": {"attempted": dict(R.JOB_CAPS), "completed": dict(R.JOB_CAPS), "execution_counts_known": True},
              "checkpoint_sha256": R.L.sha(warm / "full_state.msgpack"), "checkpoint_receipt_sha256": R.L.sha(warm / "checkpoint.json")})
        write(warm / "WORKER_EXIT.json", {"status": "completed", "identity_unchanged": True, "report_sha256": R.L.sha(warm / "report.json")})
        write(warm / "SUPERVISOR.json", {"status": "accepted", "exit_code": 0})
        (warm / "worker.log").write_text("warmup log\n", encoding="utf-8")
        native = self.run / R.INTERRUPTED
        self.job(native, "native")
        write(native / "TRAINING_START.json", {"status": "failed_training_job"})
        (native / "progress.jsonl").write_text("\n".join((progress_line(1, 1, 0, 1, 0),
            progress_line(3000, 3000, 1500, 3000, 0), progress_line(3100, 3100, 1550, 3100, 0))) + "\n", encoding="utf-8")
        write(self.run / "RESULT.json", {"status": "incomplete", "error": "InterruptedError: Supervisor signal 2", "elapsed_seconds": 1890.0})

    def job(self, directory, arm):
        write(directory / "JOB.json", {"id": "honeypot_s0_" + arm, "condition": "honeypot", "seed_index": 0, "arm": arm,
              "remaining_seconds": 11000, "amendment_manifest_sha256": AMENDMENT, "smoke_proof": self.proof})

    def lineage(self):
        return R.prior_lineage(self.smoke, self.run, smoke_check=lambda d: self.smoke_report, amendment_manifest_sha256=AMENDMENT)


class PlanTests(unittest.TestCase):
    def test_remaining_jobs_keep_sealed_order_without_reused_warmup(self):
        jobs = R.remaining_jobs()
        self.assertEqual(len(jobs), 18)
        self.assertEqual(jobs[0]["id"], R.INTERRUPTED)
        self.assertEqual(jobs[-1], {"id": "evaluation", "arm": "evaluation"})
        self.assertEqual([j for j in R.L.jobs() if j["id"] != R.REUSED], jobs)
        self.assertNotIn(R.REUSED, [j["id"] for j in jobs])

    def test_new_work_caps_and_approval_text_agree(self):
        self.assertEqual(R.production_caps(), {"critic": 170000, "actor": 85000, "auxiliary": 170000, "calibrations": 17})
        a = R.allocation()
        self.assertEqual(a["production"]["training_jobs"], 17)
        self.assertEqual(a["titan_smoke"]["transition_cap"], 3320)
        self.assertEqual(R.BANK_FILES, 398)
        for fragment in ("170000 critic", "85000 actor", "17 calibrations", "237600 evaluation transitions",
                         "%d seconds" % R.PRODUCTION_SECONDS, "OPENBLAS_CORETYPE=Haswell", "no automatic retry"):
            self.assertIn(fragment, R.approval_text())

    def test_real_checkout_matches_both_seals(self):
        self.assertEqual(R.L.sha(HERE / "MANIFEST.json"), R.S.ORIGINAL_MANIFEST)
        self.assertEqual(R.L.sha(HERE / "AMENDMENT_MANIFEST.json"), R.AMENDMENT_MANIFEST_SHA256)

    def test_real_saved_controller_trace_has_the_pinned_counts(self):
        path = R.L.REPAIR / "danger_attempt_1/transitions.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        self.assertEqual(len(rows), 613)
        self.assertEqual(sum(r["variant"] == "safe" for r in rows), 598)

    def test_next_job_fits_protects_evaluation_reserve(self):
        reserve, margin = R.EVALUATION_RESERVE_SECONDS, R.JOB_MARGIN
        self.assertTrue(R.next_job_fits("native", reserve, []))
        self.assertFalse(R.next_job_fits("native", reserve - 1, []))
        self.assertTrue(R.next_job_fits("bca", reserve + margin * 1000, [900, 1000]))
        self.assertFalse(R.next_job_fits("bca", reserve + margin * 1000 - 1, [900, 1000]))
        self.assertTrue(R.next_job_fits("evaluation", reserve, [5000]))
        self.assertFalse(R.next_job_fits("evaluation", reserve - 1, []))

    def test_titan_environment_is_required(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "OPENBLAS_CORETYPE"):
                R.require_titan_environment()
        with patch.dict(os.environ, {"OPENBLAS_CORETYPE": "SkylakeX"}):
            with self.assertRaises(ValueError):
                R.require_titan_environment()
        with patch.dict(os.environ, {"OPENBLAS_CORETYPE": "Haswell"}):
            R.require_titan_environment()


class LineageTests(unittest.TestCase):
    def test_exact_prior_attempt_is_authenticated_and_bounded(self):
        with TemporaryDirectory() as d:
            z = ZeusFixture(d)
            lineage = z.lineage()
            self.assertEqual(lineage["zeus_smoke_proof"], z.proof)
            self.assertEqual(lineage["reused_warmup"]["counts"], R.JOB_CAPS)
            self.assertEqual(set(lineage["reused_warmup"]["files"]), {"JOB.json", "full_state.msgpack", "checkpoint.json",
                             "report.json", "WORKER_EXIT.json", "SUPERVISOR.json", "worker.log"})
            interrupted = lineage["interrupted"]
            # train.PhysicalLedger counts are per job: the lower bound is the last persisted record, the cap the upper.
            self.assertEqual(interrupted["lower"], {"critic": 3100, "actor": 1550, "auxiliary": 3100, "calibrations": 0})
            self.assertEqual(interrupted["upper"], R.JOB_CAPS)
            self.assertEqual(interrupted["source"], "progress_lower_bound_with_planned_job_cap")
            self.assertEqual(lineage["attempt"]["terminal_status"], "incomplete")
            account = R.accounting(lineage)
            self.assertEqual(account["prior_optimizer_upper"], {"critic": 14 + 10000 + 10000, "actor": 8 + 5000 + 5000,
                             "auxiliary": 14 + 10000 + 10000, "calibrations": 3 + 1 + 1})
            self.assertEqual(account["cumulative_optimizer_upper"]["critic"], 20014 + 14 + 170000)
            self.assertEqual(account["cumulative_simulator_maximum"], 189655 + 3226 + 3320 + 237600)
            self.assertLessEqual(account["cumulative_simulator_maximum"], 498400)

    def test_changed_or_differently_shaped_prior_attempt_is_refused(self):
        cases = {
            "changed_checkpoint_byte": lambda z: (z.run / R.REUSED / "full_state.msgpack").write_bytes(b"zeus-warmup-statf"),
            "extra_started_job": lambda z: (z.run / "honeypot_s0_bca").mkdir(),
            "attempt_bound_to_other_smoke": lambda z: write(z.run / "ATTEMPT.json", {"mode": "production",
                "smoke_proof": {**z.proof, "report.json": "0" * 64}, "amendment_manifest_sha256": AMENDMENT}),
            "attempt_bound_to_other_amendment": lambda z: write(z.run / "ATTEMPT.json", {"mode": "production",
                "smoke_proof": z.proof, "amendment_manifest_sha256": "b" * 64}),
            "interrupted_job_was_accepted": lambda z: write(z.run / R.INTERRUPTED / "SUPERVISOR.json", {"status": "accepted"}),
            "interrupted_job_has_final_checkpoint": lambda z: (z.run / R.INTERRUPTED / "full_state.msgpack").write_bytes(b"x"),
            "warmup_not_supervisor_accepted": lambda z: write(z.run / R.REUSED / "SUPERVISOR.json", {"status": "failed", "exit_code": 1}),
            "warmup_job_reservation_differs": lambda z: write(z.run / R.REUSED / "JOB.json", {"id": R.REUSED, "condition": "honeypot",
                "seed_index": 1, "arm": "warmup", "amendment_manifest_sha256": AMENDMENT, "smoke_proof": z.proof}),
            "prior_study_completed": lambda z: write(z.run / "RESULT.json", {"status": "completed_path_a_study"}),
        }
        for name, damage in cases.items():
            with self.subTest(case=name), TemporaryDirectory() as d:
                z = ZeusFixture(d)
                damage(z)
                with self.assertRaises(ValueError):
                    z.lineage()

    def test_interrupted_bounds_from_progress_report_or_nothing(self):
        with TemporaryDirectory() as d:
            job = Path(d)
            self.assertEqual(R.interrupted_bounds(job)["lower"], dict.fromkeys(R.KEYS, 0))
            (job / "progress.jsonl").write_text(progress_line(100, 100, 50, 100, 0) + "\n" + progress_line(5000, 5000, 2500, 5000, 1)
                                                + "\n" + '{"local_step": 51', encoding="utf-8")
            bounds = R.interrupted_bounds(job)
            self.assertEqual(bounds["lower"], {"critic": 5000, "actor": 2500, "auxiliary": 5000, "calibrations": 1})
            self.assertEqual(bounds["upper"], R.JOB_CAPS)
            self.assertTrue(bounds["truncated_final_progress_line"])
            (job / "progress.jsonl").write_text('{"cut\n' + progress_line(100, 100, 50, 100, 0) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Corrupted"):
                R.interrupted_bounds(job)
            (job / "progress.jsonl").unlink()
            exact = {"critic": 4321, "actor": 2160, "auxiliary": 4321, "calibrations": 0}
            write(job / "report.json", {"status": "failed_training_job", "execution_counts_known": True,
                  "counts": {"attempted": exact, "completed": exact}})
            bounds = R.interrupted_bounds(job)
            self.assertEqual((bounds["lower"], bounds["upper"], bounds["source"]), (exact, exact, "failed_job_report_exact"))
            write(job / "report.json", {"status": "failed_training_job", "execution_counts_known": False,
                  "counts": {"attempted": {**exact, "critic": 4322}, "completed": exact}})
            bounds = R.interrupted_bounds(job)
            self.assertEqual((bounds["lower"], bounds["upper"]), (exact, R.JOB_CAPS))
            write(job / "report.json", {"status": "completed_training_job", "execution_counts_known": True, "counts": {}})
            with self.assertRaisesRegex(ValueError, "completion"):
                R.interrupted_bounds(job)

    def test_ceiling_is_enforced(self):
        with TemporaryDirectory() as d:
            lineage = ZeusFixture(d).lineage()
            with patch.object(R, "SIMULATOR_CEILING", 189655 + 3226 + 3320 + 237600 - 1):
                with self.assertRaisesRegex(ValueError, "ceiling"):
                    R.accounting(lineage)


class EquivalenceTests(unittest.TestCase):
    def banks(self, root):
        root.mkdir()
        write(root / "smoke_evaluation_plan.json", {"plan": 1})
        write(root / "smoke_own_policy_anchor.json", {"anchor": 29})
        for seed in range(3):
            for condition in ("honeypot", "bottleneck"):
                name = "smoke_%s_seed%d" % (condition, seed)
                write(root / (name + "_starts.json"), {"seed": seed})
                write(root / (name + "_coverage.json"), {"condition": condition})
                for i in range(16):
                    write(root / (name + "_anchor_%02d.json" % i), {"anchor": i})
                    for slot in range(3):
                        write(root / (name + "_edge_%02d_%d.json" % (i, slot)), {"edge": [i, slot]})
        return root

    def test_bank_files_must_match_byte_for_byte_and_learner_files_are_ignored(self):
        with TemporaryDirectory() as d:
            zeus, titan = self.banks(Path(d) / "zeus"), self.banks(Path(d) / "titan")
            write(zeus / "smoke_native_evaluation.json", {"q": 1.0})
            write(titan / "smoke_native_evaluation.json", {"q": 2.0})
            write(titan / "smoke_evaluation_report.json", {"counts": 9})
            result = R.bank_equivalence(zeus, titan)
            self.assertEqual((result["files_compared"], result["identical"]), (398, True))
            write(titan / "smoke_bottleneck_seed2_edge_15_2.json", {"edge": [15, 2.0000000000000004]})
            result = R.bank_equivalence(zeus, titan)
            self.assertEqual(result["differing"], ["smoke_bottleneck_seed2_edge_15_2.json"])
            (titan / "smoke_honeypot_seed0_anchor_03.json").unlink()
            with self.assertRaisesRegex(ValueError, "file sets"):
                R.bank_equivalence(zeus, titan)

    def rows(self):
        rng = np.random.default_rng(7)
        rows = []
        for i in range(613):
            state = rng.normal(size=13)
            action = np.clip(state[:4], -.8, .8).astype(np.float32)
            rows.append({"variant": "bad" if i < 15 else "safe", "time": 29 + i % 40,
                         "state": state.tolist(), "action": action.tolist()})
        return rows

    def test_controller_actions_must_match_saved_float32_bytes(self):
        rows = self.rows()
        def control(state, t):
            return np.clip(state[:4], -.8, .8).astype(np.float32)
        result = R.controller_equivalence(control, rows)
        self.assertEqual((result["compared_controller_actions"], result["identical"], result["simulator_transitions"]), (598, True, 0))
        def one_ulp_off(state, t):
            a = control(state, t)
            if np.array_equal(state, np.asarray(rows[20]["state"])):
                a[0] = np.nextafter(a[0], np.float32(2))
            return a
        result = R.controller_equivalence(one_ulp_off, rows)
        self.assertEqual((result["mismatched"], result["first_mismatched_rows"], result["identical"]), (1, [5], False))
        with self.assertRaisesRegex(ValueError, "count"):
            R.controller_equivalence(control, rows[:-1])
        with self.assertRaisesRegex(ValueError, "float32"):
            R.controller_equivalence(lambda s, t: np.clip(s[:4], -.8, .8), rows)

    def test_reused_warmup_copy_is_byte_identical_and_exclusive(self):
        with TemporaryDirectory() as d:
            z = ZeusFixture(d)
            source = z.run / R.REUSED
            expected = R.tree_hashes(source)
            receipt = R.install_reused_warmup(source, Path(d) / "copy", expected)
            self.assertTrue(receipt["byte_identical"])
            self.assertEqual(R.tree_hashes(Path(d) / "copy"), expected)
            with self.assertRaises(FileExistsError):
                R.install_reused_warmup(source, Path(d) / "copy", expected)
            (source / "worker.log").write_text("edited\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed before reuse"):
                R.install_reused_warmup(source, Path(d) / "copy2", expected)
            self.assertFalse((Path(d) / "copy2").exists())


class TitanSmokeAcceptanceTests(unittest.TestCase):
    def fixture(self, root):
        write(root / "AMENDMENT_MANIFEST.json", {"fixture": True})
        write(root / "RECOVERY_MANIFEST.json", {"recovery": True})
        out = root / "titan_smoke_1"
        out.mkdir()
        banks = [{"seed_index": s, "condition": c, "anchors": 16, "common_edges": 48, "omitted_count": 16,
                  "retained_count": 32, "passed": True} for s in range(3) for c in ("honeypot", "bottleneck")]
        actual = {"prefix_parity": 2688, "bank_edges": 288, "forced_qpi": 40, "returns": 100}
        report = {"status": "passed_runtime_smoke", "passed": True,
            "amendment_manifest_sha256": R.L.sha(root / "AMENDMENT_MANIFEST.json"),
            "recovery_manifest_sha256": R.L.sha(root / "RECOVERY_MANIFEST.json"),
            "learning": {"status": "passed_learning_smoke", "execution_counts_known": True,
                "counts": {"attempted": R.S.LEARNING_CAPS, "completed": R.S.LEARNING_CAPS},
                "restored_actor_next_step_exact": {"native": True, "bca": True}},
            "physical": {"status": "completed_technical_evaluation_smoke", "passed": True, "completed_disposable_arms": 2,
                "bank_coverage": banks, "counts": {"execution_counts_known": True, "attempted": actual, "completed": actual}},
            "titan_checks": {"passed": True, "environment": dict(R.TITAN_ENV), "controller": {"identical": True},
                "banks": {"identical": True, "files_compared": 398}, "warmup_restore": {"exact": True}}}
        write(out / "RESULT.json", {"status": "passed_runtime_smoke", "mode": "smoke", "execution_counts_known": True, "elapsed_seconds": 500})
        self.seal(out, report)
        return out, report

    def seal(self, out, report):
        write(out / "report.json", report)
        write(out / "ARTIFACTS.json", {"report.json": R.L.sha(out / "report.json")})
        write(out / "WORKER_EXIT.json", {"status": "completed", "identity_unchanged": True,
              "report_sha256": R.L.sha(out / "report.json"), "artifacts_sha256": R.L.sha(out / "ARTIFACTS.json")})

    def test_sealed_smoke_rule_and_titan_checks_are_both_required(self):
        with TemporaryDirectory() as d, patch.object(R, "HERE", Path(d)), patch.object(R.S, "HERE", Path(d)):
            out, original = self.fixture(Path(d))
            self.assertEqual(R.titan_smoke_passed(), original)
            self.assertEqual(set(R.titan_smoke_identity()), {"RESULT.json", "WORKER_EXIT.json", "report.json", "ARTIFACTS.json"})
            damage = {
                "bank_bytes_differ": lambda r: r["titan_checks"]["banks"].update(identical=False),
                "too_few_bank_files": lambda r: r["titan_checks"]["banks"].update(files_compared=397),
                "controller_differs": lambda r: r["titan_checks"]["controller"].update(identical=False),
                "warmup_not_exact": lambda r: r["titan_checks"]["warmup_restore"].update(exact=False),
                "default_openblas": lambda r: r["titan_checks"].update(environment={"OPENBLAS_CORETYPE": "Zen"}),
                "other_recovery_seal": lambda r: r.update(recovery_manifest_sha256="0" * 64),
                "sealed_smoke_failed": lambda r: r["learning"]["restored_actor_next_step_exact"].update(bca=False),
            }
            for name, change in damage.items():
                report = copy.deepcopy(original)
                change(report)
                self.seal(out, report)
                with self.subTest(case=name), self.assertRaises((ValueError, KeyError)):
                    R.titan_smoke_passed()
            self.seal(out, original)
            write(out / "RESULT.json", {"status": "passed_runtime_smoke", "mode": "smoke", "execution_counts_known": True, "elapsed_seconds": 901})
            with self.assertRaises(ValueError):
                R.titan_smoke_passed()
            self.assertTrue(R.titan_smoke_passed(require_supervisor=False)["passed"])


class SupervisorTests(unittest.TestCase):
    def environment(self, d):
        return [patch.object(R, "HERE", Path(d)), patch.object(R, "require_wsl"),
                patch.dict(os.environ, dict(R.TITAN_ENV)), patch.object(R, "check", return_value={}),
                patch.object(R.signal, "signal"), patch.object(R.signal, "SIGHUP", signal.SIGTERM, create=True)]

    def run_with(self, patches, function, *args):
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            return function(*args)

    def test_no_reservation_or_copy_before_titan_smoke_passes(self):
        with TemporaryDirectory() as d, patch.object(R, "titan_smoke_identity", side_effect=ValueError("No passed Titan smoke")), \
             patch.object(R.subprocess, "Popen") as spawn:
            with self.assertRaisesRegex(ValueError, "No passed Titan smoke"):
                self.run_with(self.environment(d), R.supervise, "production")
            spawn.assert_not_called()
            self.assertFalse((Path(d) / "run_recovery_1").exists())

    def test_production_without_haswell_is_refused_before_anything(self):
        with TemporaryDirectory() as d, patch.object(R, "HERE", Path(d)), patch.object(R.os, "name", "posix"), \
             patch.dict(os.environ, {"OPENBLAS_CORETYPE": "Zen"}), patch.object(R, "check") as checked:
            for mode in ("smoke", "production"):
                with self.assertRaisesRegex(ValueError, "OPENBLAS_CORETYPE"):
                    R.supervise(mode)
            checked.assert_not_called()
            self.assertEqual(list(Path(d).iterdir()), [])

    def prepared(self, d):
        z = ZeusFixture(d)
        write(Path(d) / "RECOVERY_MANIFEST.json", {"prior": {"reused_warmup": {"files": R.tree_hashes(z.run / R.REUSED)}}})
        return z

    def test_stop_at_job_boundary_spawns_nothing_and_keeps_exact_copy(self):
        with TemporaryDirectory() as d, patch.object(R, "titan_smoke_identity", return_value={"proof": 1}), \
             patch.object(R, "next_job_fits", return_value=False), patch.object(R, "run_child") as child:
            self.prepared(d)
            result = self.run_with(self.environment(d), R.supervise, "production")
            child.assert_not_called()
            self.assertEqual(result["status"], "stopped_at_job_boundary")
            self.assertTrue(result["execution_counts_known"])
            out = Path(d) / "run_recovery_1"
            self.assertEqual(R.tree_hashes(out / R.REUSED), R.tree_hashes(Path(d) / "run_amendment_1" / R.REUSED))
            self.assertFalse((out / R.INTERRUPTED).exists())
            self.assertEqual(json.loads((out / "RESULT.json").read_text())["status"], "stopped_at_job_boundary")

    def test_failed_job_stops_without_retry_and_unknown_counts(self):
        with TemporaryDirectory() as d, patch.object(R, "titan_smoke_identity", return_value={"proof": 1}), \
             patch.object(R, "run_child", return_value=1) as child:
            self.prepared(d)
            result = self.run_with(self.environment(d), R.supervise, "production")
            child.assert_called_once()
            self.assertEqual(child.call_args[0][0]["id"], R.INTERRUPTED)
            self.assertEqual(result["status"], "incomplete")
            self.assertFalse(result["execution_counts_known"])
            self.assertEqual(result["jobs"][0]["exit_code"], 1)
            self.assertIn("Recovery worker failed", result["error"])

    def test_changed_zeus_warmup_blocks_copy_and_dispatch(self):
        with TemporaryDirectory() as d, patch.object(R, "titan_smoke_identity", return_value={"proof": 1}), \
             patch.object(R, "run_child") as child:
            z = self.prepared(d)
            (z.run / R.REUSED / "full_state.msgpack").write_bytes(b"changed")
            result = self.run_with(self.environment(d), R.supervise, "production")
            child.assert_not_called()
            self.assertEqual(result["status"], "incomplete")
            self.assertIn("changed before reuse", result["error"])
            self.assertFalse((Path(d) / "run_recovery_1" / R.REUSED).exists())

    def test_process_receipt_failure_terminates_owned_child(self):
        child = SimpleNamespace(pid=12345, poll=Mock(return_value=None), wait=Mock(return_value=0))
        with patch.object(R.subprocess, "Popen", return_value=child) as spawn, \
             patch.object(R.L, "save", side_effect=OSError("receipt write failed")), \
             patch.object(R.os, "killpg", create=True) as kill:
            with self.assertRaises(OSError):
                R.run_child({"id": "smoke"}, HERE, R.time.monotonic() + 5, None)
            self.assertIn(str(HERE / "recovery_launch.py"), spawn.call_args[0][0])
            kill.assert_called_once_with(12345, signal.SIGTERM)
            child.wait.assert_called_once_with(timeout=5)

    def test_wsl_is_required_for_execution(self):
        with patch.object(R.os, "name", "nt"):
            for function, argument in ((R.supervise, "smoke"), (R.supervise, "production"), (R.worker_main, "smoke")):
                with self.assertRaisesRegex(ValueError, "WSL"):
                    function(argument)

    def test_cli_help(self):
        done = subprocess.run([sys.executable, "-B", str(HERE / "recovery_launch.py"), "--help"], capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        for flag in ("--check", "--prepare", "--approve", "--smoke", "--execute"):
            self.assertIn(flag, done.stdout)


if __name__ == "__main__":
    unittest.main()
