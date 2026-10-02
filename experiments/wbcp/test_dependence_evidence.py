"""Checks for experiments/wbcp/dependence_evidence.py on synthetic benchmark results; NumPy, SciPy
and PyYAML only, no runs directory.

python -m unittest experiments.wbcp.test_dependence_evidence
"""
import json
import os
import tempfile
import unittest

from scipy import stats

from calibration import bank
from experiments.wbcp import dependence_evidence as E

TD3_Q = "min over the two critic heads at (obs, action)"


def benchmark_result(pool="runs/wbcp_frozen/maze2d-s202609171-u100000", k=5, n=1024, failures=258, trials=4000,
                     spacing="reservation", gammas=(0.0,), algorithm="td3_bc", dataset="maze2d-large-v1", q=TD3_Q,
                     rows=(1025, 1025), bank_trim=None):
    """A d4rl_benchmark.py result; bank_trim=None is one drawn before the remainder trim (no marker)."""
    low, high = stats.binomtest(failures, trials).proportion_ci(method="exact")
    arm = dict(fail=failures / trials, ci=[low, high], failures=failures, trials=trials)
    size = dict(mean=(rows[0] + rows[1]) / 2, min=rows[0], max=rows[1])
    blocks = [dict(tilt=tilt, gamma=gamma, n=n, score=score, calibration_size=size, arms={"BQ-CP": arm})
              for tilt in ("policy", "state") for gamma in gammas for score in ("normalized", "raw")]
    settings = dict(frozen=pool, per_episode=k, spacing=spacing, n=[n], trials=trials, seed=2026093001,
                    gamma=list(gammas), population="heldout", blocks=False, shuffle_tilt=None,
                    alpha=0.1, beta=0.95, draws=1000)
    frozen = dict(algorithm=algorithm, dataset=dataset, npz_sha256="ab" * 32, score_definition=dict(q=q))
    marker = {} if bank_trim is None else dict(bank_trim=bank_trim)
    return dict(settings=settings, **marker, frozen=frozen, blocks=blocks)


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = self.directory.name
        os.makedirs(os.path.join(self.root, "configs"))
        os.makedirs(os.path.join(self.root, "runs"))
        datasets = "datasets:\n  maze2d:\n    environment: maze2d-large-v1\n  hopper:\n    environment: hopper-medium-v2\n"
        for host in ("td3_bc", "cql"):
            with open(os.path.join(self.root, "configs", host + ".yaml"), "w") as handle:
                handle.write(datasets)

    def tearDown(self):
        self.directory.cleanup()

    def put(self, name, record):
        with open(os.path.join(self.root, "runs", name), "w") as handle:
            json.dump(record, handle)
        return "runs/" + name

    def test_an_entry_carries_the_run_its_interval_and_its_provenance(self):
        path = self.put("maze2d_resv5.json", benchmark_result())
        seen = []
        (entry,) = E.entries_from_run(path, self.root, health=lambda pool, key: seen.append((pool, key)) or {"status": "ok"})
        self.assertEqual(seen, [("runs/wbcp_frozen/maze2d-s202609171-u100000", "maze2d")])
        self.assertEqual((entry["host"], entry["dataset"], entry["config_dataset"], entry["rows_per_episode"],
                          entry["size"], entry["sampler"], entry["score"]),
                         ("td3_bc", "maze2d-large-v1", "maze2d", 5, 1024, "reservation", bank.MIN_Q))
        self.assertEqual((entry["uniform_bca_failure"], entry["failures"], entry["trials"]), (258 / 4000, 258, 4000))
        low, high = stats.binomtest(258, 4000).proportion_ci(method="exact")
        self.assertEqual(entry["ci95"], [low, high])
        self.assertEqual((entry["source"], entry["source_sha256"]), (path, E.sha256(os.path.join(self.root, path))))
        self.assertEqual(entry["critic_health"], {"status": "ok"})
        self.assertIsNone(entry["bank_trim"])  # no marker: drawn before the remainder trim
        self.assertIn("before the remainder trim (2026-10-01): these banks held 1,025 rows against n = 1,024",
                      entry["note"])
        self.assertEqual(bank.evidence_status(entry["ci95"]), bank.EXCEEDS)  # 6.5% [5.7, 7.3]

    def test_host_and_dataset_come_from_the_pool_name_and_must_match_its_record(self):
        self.put("a.json", benchmark_result(pool="runs/wbcp_frozen/hopper-medium-v2-s202609171-u100000",
                                            dataset="hopper-medium-v2", k=5, n=1103))
        self.put("b.json", benchmark_result(pool="runs/wbcp_frozen/cql-hopper-s202609171-u100000", k=6,
                                            algorithm="cql", dataset="hopper-medium-v2", q="min(Q1, Q2) at (obs, action)"))
        entries, skipped = E.build(("runs/*.json",), self.root)
        self.assertEqual([(e["host"], e["dataset"], e["rows_per_episode"]) for e in entries],
                         [("cql", "hopper-medium-v2", 6), ("td3_bc", "hopper-medium-v2", 5)])
        self.assertEqual(skipped, [])
        self.put("c.json", benchmark_result(pool="runs/wbcp_frozen/cql-maze2d-s202609171-u100000"))  # records td3_bc
        with self.assertRaisesRegex(ValueError, "records td3_bc"):
            E.build(("runs/*.json",), self.root)

    def test_files_that_are_not_configured_bank_evidence_are_skipped_with_a_reason(self):
        self.put("rows_per_episode.json", {"hosts": {}})
        self.put("strat.json", benchmark_result(spacing="stratified"))
        self.put("shift.json", benchmark_result(gammas=(0.5, 1.0)))
        self.put("draws.json", dict(benchmark_result(), settings=dict(benchmark_result()["settings"], draws=400)))
        entries, skipped = E.build(("runs/*.json",), self.root)
        self.assertEqual(entries, [])
        reasons = dict(skipped)
        self.assertEqual(set(reasons), {"runs/rows_per_episode.json", "runs/strat.json", "runs/shift.json",
                                        "runs/draws.json"})
        self.assertIn("not a d4rl_benchmark.py result", reasons["runs/rows_per_episode.json"])
        self.assertIn("not BCA's sampler", reasons["runs/strat.json"])
        self.assertIn("shift-only", reasons["runs/shift.json"])
        self.assertIn("WBCP settings", reasons["runs/draws.json"])

    def test_two_runs_of_one_design_and_unknown_scores_are_errors(self):
        self.put("one.json", benchmark_result())
        self.put("two.json", benchmark_result(failures=250))
        with self.assertRaisesRegex(ValueError, "same design"):
            E.build(("runs/*.json",), self.root)
        os.remove(os.path.join(self.root, "runs", "two.json"))
        self.put("q1.json", benchmark_result(k=6, q="Q1 at (obs, action)"))
        with self.assertRaisesRegex(ValueError, "no score label"):
            E.build(("runs/*.json",), self.root)

    def test_a_recorded_interval_that_does_not_match_its_counts_is_an_error(self):
        record = benchmark_result()
        for block in record["blocks"]:
            block["arms"]["BQ-CP"] = dict(block["arms"]["BQ-CP"], ci=[0.04, 0.07])
        path = self.put("bad.json", record)
        with self.assertRaisesRegex(ValueError, "does not match its counts"):
            E.entries_from_run(path, self.root)

    def test_registry_round_trips_through_the_lookup(self):
        self.put("maze2d_resv5.json", benchmark_result())
        self.put("hopper_resv6.json", benchmark_result(pool="runs/wbcp_frozen/cql-hopper-s202609171-u100000", k=6,
                                                       failures=205, algorithm="cql", dataset="hopper-medium-v2",
                                                       q="min(Q1, Q2) at (obs, action)", rows=(1026, 1026)))
        entries, _ = E.build(("runs/*.json",), self.root)
        path = os.path.join(self.root, "evidence.json")
        with open(path, "w") as handle:
            handle.write(E.dumps(E.registry(entries, ("runs/*.json",))))
        loaded = bank.load_evidence(path)
        self.assertEqual(loaded, entries)
        self.assertEqual(bank.dependence_evidence("cql", "hopper-medium-v2", 6, 1024, score=bank.MIN_Q, entries=loaded),
                         dict(entries[0], status=bank.CONSISTENT))  # 5.1% [4.5, 5.9]
        self.assertEqual(bank.dependence_evidence("td3_bc", "maze2d-large-v1", 5, 1024, score=bank.MIN_Q,
                                                  entries=loaded)["status"], bank.EXCEEDS)
        self.assertEqual(bank.dependence_evidence("iql", "maze2d-large-v1", 5, 1024, score=bank.TARGET_MIN_Q,
                                                  entries=loaded)["status"], bank.NOT_VALIDATED)
        self.assertIn("n, sampler and score exactly", E.registry(entries)["status_rule"])

    def test_every_deployed_score_has_a_freeze_definition_and_iql_scores_its_target_heads(self):
        self.assertLessEqual(set(bank.DEPLOYED_SCORE.values()), set(E.SCORES.values()))
        iql_q = ("min over the twin Polyak target Q heads at (obs, action), the copy IQL's actor advantage reads "
                 "(calibration.iql_reference.refresh)")  # freeze_iql.py's score_definition["q"]
        self.assertEqual(E.score_label(dict(score_definition=dict(q=iql_q))), bank.DEPLOYED_SCORE["iql"])
        self.assertEqual(E.score_label(dict(score_definition=dict(q=TD3_Q))), bank.DEPLOYED_SCORE["td3_bc"])

    def test_the_note_says_whether_the_remainder_trim_changes_the_design(self):
        self.assertIn("bit for bit", E.trim_note(dict(mean=248.0, min=248, max=248), 248, False))
        self.assertIn("bit for bit", E.trim_note(dict(mean=8176.2, min=7888, max=8192), 8192, False))
        self.assertIn("1,002-1,035 rows (mean 1,033.3)", E.trim_note(dict(mean=1033.26, min=1002, max=1035), 1024, False))

    def test_the_bank_trim_marker_decides_the_note_not_the_bank_sizes(self):
        # walker2d K = 23, n = 1,024: 1,035 rows drawn; after the trim every bank holds at most n rows
        walker = dict(pool="runs/wbcp_frozen/walker2d-s202609171-u100000", k=23, dataset="walker2d-medium-replay-v2")
        with open(os.path.join(self.root, "configs", "td3_bc.yaml"), "a") as handle:
            handle.write("  walker2d:\n    environment: walker2d-medium-replay-v2\n")
        before = self.put("before.json", benchmark_result(**walker, rows=(1002, 1035)))
        (entry,) = E.entries_from_run(before, self.root)
        self.assertIsNone(entry["bank_trim"])
        self.assertTrue(entry["note"].startswith("Measured before the remainder trim (2026-10-01): these banks held "
                                                 "1,002-1,035 rows"))
        for rows, tail in (((1024, 1024), "held 1,024 rows against n = 1,024."),
                           ((1002, 1024), "shorter than K leave fewer than n.")):
            after = self.put("after.json", benchmark_result(**walker, rows=rows, bank_trim=bank.REMAINDER_TRIM))
            (entry,) = E.entries_from_run(after, self.root)
            self.assertEqual(entry["bank_trim"], bank.REMAINDER_TRIM)
            self.assertTrue(entry["note"].startswith("Measured with the remainder trim (2026-10-01)"), entry["note"])
            self.assertTrue(entry["note"].endswith(tail), entry["note"])
            self.assertNotIn("bit for bit", entry["note"])
            self.assertNotIn("before", entry["note"])
        odd = self.put("odd.json", benchmark_result(**walker, rows=(1024, 1024), bank_trim="truncated (2026-09-01)"))
        with self.assertRaisesRegex(ValueError, "unknown sampler revision"):
            E.entries_from_run(odd, self.root)
        over = self.put("over.json", benchmark_result(**walker, rows=(1002, 1035), bank_trim=bank.REMAINDER_TRIM))
        with self.assertRaisesRegex(ValueError, "over n = 1,024"):
            E.entries_from_run(over, self.root)


if __name__ == "__main__":
    unittest.main()
