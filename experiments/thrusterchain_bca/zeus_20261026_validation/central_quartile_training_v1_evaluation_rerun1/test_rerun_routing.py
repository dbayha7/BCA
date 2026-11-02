"""Mixed original/continuation routing with the real pin-authentication functions; synthetic files only."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import evaluate_worker as worker

PENDING = {('seed2_hole', 'bca'), *(('seed2_random', a) for a in ('native', 'bca', 'distance'))}
REAL_RECOVERY = worker.recovery_inputs.recovery()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, sort_keys=True)+'\n').encode(); path.write_bytes(raw)
    return dict(path=path.name, sha256=worker.digest(raw), bytes=len(raw))


def counts(step):
    return dict(critic=step, actor=step//2, actor_target=step//2, critic_target=step//2,
                outer_steps=step//2, auxiliary=step, aux_fit_attempts=step,
                refresh_attempts=0 if step == 0 else 1 if step == 10000 else 2,
                refresh_accepted=0 if step == 0 else 1 if step == 10000 else 2, phase=0)


def delta(final, initial):
    return {k: final[k]-initial[k] for k in final if k != 'phase'}


def receipt(directory, cell, stage, complete=True):
    directory.mkdir(parents=True, exist_ok=True)
    desc = write_json(directory/'matching_receipt.json',
                      dict(cell=cell, protocol_sha256='synthetic', distance_coefficient=2.5))
    report = dict(status='completed_central_quartile_'+('warmup' if stage == 'warmup' else 'branch'),
                  protocol_sha256='synthetic', cell=cell, condition=cell[6:], seed_index=int(cell[4]),
                  arm='native' if stage == 'warmup' else stage, execution_counts_known=True,
                  matching_receipt_sha256=desc['sha256'])
    seal_receipt(directory, report, complete)
    return report


def seal_receipt(directory, report, complete=True):
    desc = write_json(directory/'report.json', report)
    stage = 'warmup' if report['status'].endswith('warmup') else report['arm']
    write_json(directory/'SUPERVISOR_RESULT.json', dict(job=report['cell']+'/'+stage, completed=complete,
               exit_code=0 if complete else -9, timed_out=False, interrupted=not complete,
               report_sha256=desc['sha256']))


def checkpoint(directory, step, matching_sha):
    folder = directory/'checkpoints'/f'step_{step:05d}'; folder.mkdir(parents=True, exist_ok=True)
    def binary(name, raw):
        (folder/name).write_bytes(raw)
        return dict(path=name, sha256=worker.digest(raw), bytes=len(raw))
    manifest = dict(schema='unifloral.central_quartile.checkpoint.v1', protocol_sha256='synthetic',
        cell='seed2_hole', condition='hole', seed_index=2, kind='branch', arm='bca',
        absolute_critic_step=step, complete_optimizer_target_rng_state=True, resume_authorized=False,
        counts=counts(step), matching_receipt_sha256=matching_sha, normalization={'fixture': 'common'},
        artifact=binary('full_state.msgpack', f'full state {step}'.encode()),
        actor_params=binary('actor_params.msgpack', f'actor {step}'.encode()) if step != 16000 else None)
    desc = write_json(folder/'manifest.json', manifest)
    return {**desc, 'path': f'checkpoints/step_{step:05d}/manifest.json'}


class RerunRoutingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name).resolve()
        self.current, self.prior, self.elsewhere = base/'continuation1', base/'authorized1', base/'elsewhere'
        for folder in (self.current, self.prior, self.elsewhere): folder.mkdir()
        self.prior_pins, self.continuation_pins = {}, {}
        recovery = {**copy.deepcopy(REAL_RECOVERY), 'prior_files_sha256': self.prior_pins}
        for obj, name, value in [(worker.recovery_inputs, 'prior_root', lambda: self.prior),
                                 (worker.recovery_inputs, 'recovery', lambda: recovery),
                                 (worker.rerun_inputs, 'continuation_root', lambda: self.current),
                                 (worker.rerun_inputs, 'plan',
                                  lambda: {'continuation_files_sha256': self.continuation_pins})]:
            patcher = patch.object(obj, name, value); patcher.start(); self.addCleanup(patcher.stop)

    def pin(self):
        for root, pins in ((self.prior, self.prior_pins), (self.current, self.continuation_pins)):
            pins.update({p.relative_to(root).as_posix(): worker.sha(p) for p in root.rglob('*') if p.is_file()})

    def resumed(self):
        old, new = self.prior/'run/seed2_hole/bca', self.current/'run/seed2_hole/bca'
        old_report = receipt(old, 'seed2_hole', 'bca', False)
        descriptor15 = checkpoint(old, 15000, old_report['matching_receipt_sha256'])
        descriptor16 = checkpoint(old, 16000, old_report['matching_receipt_sha256'])
        report = receipt(new, 'seed2_hole', 'bca')
        for path in (old/'checkpoints/step_15000').iterdir():
            target = new/'checkpoints/step_15000'/path.name; target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
        descriptor20 = checkpoint(new, 20000, report['matching_receipt_sha256'])
        report.update(initial_counts=counts(10000), segment_initial_counts=counts(16000), final_counts=counts(20000),
            logical_branch_counts_known=True, prior_physical_execution_counts_known=False,
            additional_updates=delta(counts(20000), counts(10000)),
            segment_additional_updates=delta(counts(20000), counts(16000)),
            ledger={'committed_additional_updates': delta(counts(20000), counts(16000))},
            continuation_lineage={'source_job': 'seed2_hole/bca', 'resume_from_step': 16000,
                'source_checkpoint': {'path': 'run/seed2_hole/bca/'+descriptor16['path'],
                                      'sha256': descriptor16['sha256']},
                'prefix_unique_replay_source_rows': None},
            checkpoint_manifests={'15000': descriptor15, '20000': descriptor20})
        source16 = json.loads((old/descriptor16['path']).read_bytes())
        report['continuation_restore_receipt'] = dict(status='passed_exact_continuation_restore',
            source_checkpoint=report['continuation_lineage']['source_checkpoint'], counts=counts(16000),
            optimizer_updates=0, calibration_updates=0, simulator_transitions=0,
            exact_state={'comparison': 'exact_decoded_state', 'numeric_tolerances_used': False,
                         'checkpoint_sha256': source16['artifact']['sha256']})
        seal_receipt(new, report); self.pin()
        return old, new, report

    def test_real_routing_sends_32_jobs_to_original_and_4_to_continuation(self):
        seen = {}
        def training_seal(directory, cell, stage, protocol_sha):
            seen[(cell, stage)] = Path(directory)
            return {'initial_parameter_hashes': {'actor': cell[4]}}, {}, {}
        def checkpoint_export(directory, report, spec, protocol_sha):
            return {'manifest': {'normalization': {'shared': True}}}, {}
        protocol = worker.read(Path(__file__).parent/'protocol.json')
        with patch.object(worker, 'training_seal', side_effect=training_seal),\
                patch.object(worker, 'checkpoint_export', side_effect=checkpoint_export):
            exports, _ = worker.assemble_exports(protocol, self.current/'run', 'synthetic')
            self.assertEqual(len(exports), 72)
            self.assertEqual(sum(path.is_relative_to(self.prior) for path in seen.values()), 32)
            self.assertEqual({k for k, path in seen.items() if path.is_relative_to(self.current)}, PENDING)
            for wrong in (self.prior/'run', worker.HERE/'run'):
                with self.assertRaisesRegex(ValueError, 'Undeclared training root'):
                    worker.assemble_exports(protocol, wrong, 'synthetic')
        with self.assertRaisesRegex(ValueError, 'Unknown evaluation job'):
            worker.rerun_inputs.job_directory('seed9_full', 'warmup')

    def test_both_roots_are_pin_authenticated_against_self_consistent_reseal(self):
        prior = self.prior/'run/seed0_full/warmup'; report = receipt(prior, 'seed0_full', 'warmup')
        current = self.current/'run/seed2_random/native'; branch = receipt(current, 'seed2_random', 'native')
        self.pin()
        worker.training_seal(prior, 'seed0_full', 'warmup', 'synthetic')
        worker.training_seal(current, 'seed2_random', 'native', 'synthetic')
        report['foreign_replacement'] = True; seal_receipt(prior, report)
        with self.assertRaisesRegex(ValueError, 'Prior artifact changed'):
            worker.training_seal(prior, 'seed0_full', 'warmup', 'synthetic')
        branch['foreign_replacement'] = True; seal_receipt(current, branch)
        with self.assertRaisesRegex(ValueError, 'Continuation artifact changed'):
            worker.training_seal(current, 'seed2_random', 'native', 'synthetic')

    def test_unpinned_escaped_or_outside_artifacts_are_rejected(self):
        self.pin()
        late = self.current/'run/seed2_random/bca'; receipt(late, 'seed2_random', 'bca')
        with self.assertRaisesRegex(ValueError, 'Unpinned continuation artifact'):
            worker.training_seal(late, 'seed2_random', 'bca', 'synthetic')
        outside = self.elsewhere/'run/seed2_random/bca'; receipt(outside, 'seed2_random', 'bca')
        with self.assertRaisesRegex(ValueError, 'outside the pinned original and continuation roots'):
            worker.training_seal(outside, 'seed2_random', 'bca', 'synthetic')
        with self.assertRaisesRegex(ValueError, 'Unsafe or missing'):
            worker.checked(self.current, '../authorized1/x.json')
        with self.assertRaisesRegex(ValueError, 'escaped'):
            worker.rerun_inputs.authenticate_continuation_file('../authorized1/x.json')

    def test_resumed_branch_reads_its_continuation_receipt_not_the_interrupted_original(self):
        old, new, report = self.resumed()
        actual, _, hashes = worker.training_seal(new, 'seed2_hole', 'bca', 'synthetic')
        self.assertEqual(actual['segment_additional_updates']['critic'], 4000)
        self.assertTrue(any(Path(k).is_relative_to(self.prior) for k in hashes))
        with self.assertRaisesRegex(ValueError, 'Incomplete supervisor'):
            worker.training_seal(old, 'seed2_hole', 'bca', 'synthetic')
        copy_root = self.elsewhere/'run/seed2_hole/bca'
        for path in new.rglob('*'):
            if path.is_file():
                target = copy_root/path.relative_to(new); target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())
        with self.assertRaisesRegex(ValueError, 'outside the pinned'):
            worker.training_seal(copy_root, 'seed2_hole', 'bca', 'synthetic')

    def test_imported_15k_export_and_final_counts_stay_bound_to_original_source(self):
        _, new, report = self.resumed()
        spec = dict(cell='seed2_hole', condition='hole', seed=2, stage='bca', arm='bca', checkpoint=15000)
        worker.checkpoint_export(new, report, spec, 'synthetic')
        spec20 = {**spec, 'checkpoint': 20000}
        worker.checkpoint_export(new, report, spec20, 'synthetic')
        bad = copy.deepcopy(report); bad['final_counts']['actor_target'] -= 1
        with self.assertRaisesRegex(ValueError, 'report/final checkpoint counters'):
            worker.checkpoint_export(new, bad, spec20, 'synthetic')
        actor = new/'checkpoints/step_15000/actor_params.msgpack'; actor.write_bytes(b'foreign actor')
        with self.assertRaisesRegex(ValueError, 'Continuation artifact changed'):
            worker.checkpoint_export(new, report, spec, 'synthetic')


if __name__ == '__main__': unittest.main()
