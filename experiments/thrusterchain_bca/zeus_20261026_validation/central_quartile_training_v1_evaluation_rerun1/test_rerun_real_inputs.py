"""Read-only checks of the real sealed inputs: pins, inventory, routing, time cap, authorization.

Hashes and receipts only; no recorded result file is parsed and no model is loaded.
"""
import unittest

import authorization
import evaluate_worker as worker
import evaluation
import rerun_inputs as rerun
from common import HERE, read, sha


@unittest.skipUnless(rerun.continuation_root().is_dir(), 'continuation1 package not present')
class RealInputTests(unittest.TestCase):
    def test_every_continuation_file_matches_its_pin_and_receipts_agree(self):
        lineage = rerun.verify_continuation(all_artifacts=True)
        self.assertEqual(lineage['training_physical_accounting']['cumulative_recorded_lower_bound'],
                         {'critic': 360600, 'actor': 180300, 'auxiliary': 360600, 'calibrations': 36})
        self.assertFalse(lineage['training_physical_accounting']['cumulative_counts_exact'])
        self.assertEqual(len(rerun.plan()['continuation_files_sha256']), 368)

    def test_recorded_inventory_is_exactly_the_plan_minus_the_missing_context(self):
        files = rerun.recorded_evaluation()['snapshot_results']
        planned = {spec['id']+'/'+context for spec in worker.snapshot_plan(read(HERE/'protocol.json'))
                   for context in evaluation.CONTEXTS}
        self.assertEqual(len(planned), 144)
        self.assertEqual({k for k in files if '/' in k}, planned - {rerun.NEW_CONTEXT})
        self.assertEqual(sorted(k for k in files if '/' not in k), rerun.BANK_KEYS)
        gate = worker.ReproductionGate()
        self.assertEqual(len(gate.expected), 143)
        self.assertEqual(gate.first_attempt['completed_snapshot_contexts'], 143)

    def test_real_routing_and_time_cap(self):
        protocol = read(HERE/'protocol.json')
        jobs = {(s['cell'], s['stage']) for s in worker.snapshot_plan(protocol)}
        roots = {job: rerun.job_directory(*job) for job in jobs}
        self.assertEqual(len(roots), 36)
        self.assertTrue(all(path.is_dir() for path in roots.values()))
        continuation = {job for job, path in roots.items() if path.is_relative_to(rerun.continuation_root())}
        self.assertEqual({c+'/'+s for c, s in continuation}, set(rerun.CONTINUATION_JOBS))
        self.assertEqual(rerun.rerun_seconds(), 7200)

    def test_authorization_chain_and_unchanged_sources(self):
        approval = authorization.verify(HERE, read(HERE/'protocol.json'))
        self.assertFalse(approval['training_authorized'])
        sealed = read(rerun.continuation_root()/'PREPARED.json')['files']
        for name in authorization.UNCHANGED:
            self.assertEqual(sha(HERE/name), sealed[name], name)


if __name__ == '__main__': unittest.main()
