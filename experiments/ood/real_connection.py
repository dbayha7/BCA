"""Bounded real-simulator engineering connection for the accepted first pair.

Reuses the closed CPU checkpoint query gate. This is a NEW full-state integration
gate, not scientific state collection or an action-harm result. All simulator
calls, including the audited constructor's native step, have prior reservations.
No retries, GPU allocation, training changes, coverage claim or permissive factory.
"""
from copy import deepcopy
import gzip
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

from experiments.ood.collect import file_hash, read_artifact, write_artifact, sha, require
from experiments.ood.resources import ResourceLedger, scope_caps, prior_adapter_usage


def linux_path(name):
    name = str(name).replace('\\', '/')
    return Path('/mnt/'+name[0].lower()+name[2:] if len(name)>2 and name[1:3]==':/' else name)


class AccountedSimulator:
    """One bounded engineering instance; any failed physical call poisons it."""
    def __init__(self, ledger, scope, token, dataset, path, max_transitions):
        from experiments.ood.simulator import LocomotionAdapter
        from gym.envs.mujoco.mujoco_env import MujocoEnv
        require(scope == 'engineering/td3_bc/hopper' and dataset == 'hopper-medium-v2'
                and type(max_transitions) is int and 0 < max_transitions <= 8,
                'Only the declared bounded Hopper engineering connection is executable.')
        self.ledger, self.scope, self.token = ledger, scope, token
        self.path = Path(path); self.path.mkdir(exist_ok=False)
        self.maximum, self.calls, self.failed = max_transitions, 0, False
        observed = []
        original = MujocoEnv.do_simulation

        def constructor_physics(env, control, frames):
            require(not observed and frames == 4, 'Unexpected constructor integration count.')
            record = dict(control=np.asarray(control).copy(), frames=int(frames))
            write_artifact(self.path/'constructor-control-before-physics.json', record)
            observed.append(record)
            return original(env, control, frames)

        def construct():
            with patch.object(MujocoEnv, 'do_simulation', constructor_physics):
                return LocomotionAdapter(dataset, max_transitions=max_transitions,
                                         evidence_dir=self.path/'transitions')
        self.sim = self.ledger.call(token+'/constructor', [scope], 1, 4, construct,
                                   evidence=dict(kind='constructor', maximum_physics=4))
        try:
            require(len(observed) == 1, 'Constructor transition not observed.')
            write_artifact(self.path/'constructor-completed.json', dict(environment=1, physics=4,
                identity=self.sim.identity, observed=observed))
        except BaseException:
            self.sim.close()
            raise

    def step(self, action):
        require(not self.failed and self.calls < self.maximum, 'Failed/exhausted engineering instance.')
        self.calls += 1
        number = self.calls
        before = dict(state=self.sim.capture(), action=np.asarray(action).copy())
        input_hash = write_artifact(self.path/f'call-{number:02d}-input.json', before)
        try:
            record = self.ledger.call(self.token+f'/step/{number}', [self.scope], 1, 4,
                lambda: self.sim.step(action), evidence=dict(input_sha256=input_hash))
            write_artifact(self.path/f'call-{number:02d}-completed.json',
                           dict(record=record, after=self.sim.capture()))
            return record
        except BaseException as exc:
            self.failed = True
            write_artifact(self.path/f'call-{number:02d}-failed.json',
                           dict(error=repr(exc), reservation_not_refunded=True))
            raise

    def close(self):
        self.sim.close()


def _load_pair(repo, lane, binding):
    """Prepare the same bound arrays, without constructing extra environments."""
    import h5py
    from runtime.config import typed
    from experiments.ood.adapters import CheckpointAdapter
    manifest = json.loads(gzip.decompress((repo/'experiments/standard_bca/manifest.json.gz').read_bytes()))
    models = {}
    with patch('gym.make', side_effect=RuntimeError('Only the accounted constructor is allowed')):
        for method, cp in binding['checkpoints'].items():
            run = lane/'runs'/cp['run_id']
            row, = [r for r in manifest['runs'] if r['row']['run_id'] == cp['run_id']]
            runner, args, spec, protocol = typed(deepcopy(row['row']))
            audit = json.loads(linux_path(binding['inputs'][method+'/accepted_audit']['path']).read_text())
            require(file_hash(run/'preparation.json') == audit['evidence_sha256']['preparation.json'], 'Changed preparation.')
            metadata = json.loads((run/'preparation.json').read_text())['metadata']
            with h5py.File(metadata['raw_identity']['path'], 'r') as source:
                raw = {k: source[k][()] for k in metadata['raw_array_hashes']}
            prepared = runner.prepare(raw, args, spec, protocol, max_action=1., max_episode_steps=1000,
                                      raw_identity=metadata['raw_identity'])
            require(prepared.metadata['run_input_hashes'] == cp['preparation_input_hashes'], 'Changed prepared arrays.')
            checkpoint = linux_path(cp['checkpoint_path'])
            require(checkpoint == run/'checkpoint_1000000.msgpack', 'Wrong checkpoint path.')
            models[method] = CheckpointAdapter(checkpoint, cp['checkpoint_sha256'], 'td3_bc', args, spec.config(), prepared)
            del prepared, raw
    return models


def connect(repo, accepted_attempt, lane, attempt, ledger_path):
    require(os.environ.get('JAX_PLATFORMS') == 'cpu' and os.environ.get('CUDA_VISIBLE_DEVICES') == '',
            'CPU/hide-GPU settings must be present at startup.')
    require(not str(ledger_path).startswith('/mnt/'), 'Persistent resource ledger must use local Linux storage.')
    validation_path = repo/'docs/validation/ood-trained-td3-hopper-connection.json'
    validation = json.loads(validation_path.read_text())
    require(validation['trained_cpu_query_gate_accepted'] is True
            and validation['pair_id'] == 'td3_bc/hopper/s202609171', 'Closed trained gate required.')
    for relative, expected in validation['artifacts_sha256'].items():
        # Reuse receipts; do not rerun original queries or inspect closed journals.
        require(file_hash(repo/relative) == expected, 'Changed closed query evidence.')
    accepted = accepted_attempt/'trained-connection-v1'
    precommit = read_artifact(accepted/'precommit.json', file_hash(accepted/'precommit.json'))
    require(file_hash(accepted/'precommit.json') == validation['artifacts_sha256'][
        'outputs/ood/td3_bc/hopper/s202609171/trained-connection-v2/precommit.json'], 'Wrong closed gate attempt.')
    for relative, expected in precommit['source_sha256'].items():
        require(file_hash(repo/relative) == expected, 'Closed query implementation changed.')
    binding = read_artifact(accepted_attempt/'training-binding.json', precommit['binding_sha256'])
    declaration = read_artifact(accepted_attempt/'full-declaration.json', precommit['declaration_sha256'])
    for relative, expected in binding['frozen_source_sha256'].items():
        require(file_hash(repo/relative) == expected, 'Frozen scientific source changed.')
    for item in binding['inputs'].values():
        require(file_hash(linux_path(item['path'])) == item['sha256'], 'Changed accepted training evidence.')
    prior_path = repo/'experiments/ood/validation.json'
    prior = json.loads(prior_path.read_text())
    history = prior_adapter_usage(prior)
    # Check the existing history declarations/exit receipts by their exact leads;
    # this is cumulative accounting, not reopening old model/result audits.
    for old in prior['attempts']:
        for kind in ('declaration', 'actual_exit'):
            item = old[kind]
            require(file_hash(repo/item['path']) == item['sha256'], 'Changed prior engineering '+kind)
    attempt.mkdir(parents=True, exist_ok=False)
    sources = ['experiments/ood/real_connection.py', 'experiments/ood/resources.py',
               'experiments/ood/simulator.py', 'experiments/ood/adapters.py', 'experiments/ood/collect.py',
               'runtime/environment.py']
    cell = 'td3_bc/hopper'
    seeds = declaration['engineering_streams'][cell][2:4]
    settings = dict(schema='ood-real-connection-engineering-v1', pair_id=binding['pair_id'],
        trained_query_validation_sha256=file_hash(validation_path), prior_receipt_sha256=file_hash(prior_path),
        full_declaration_sha256=precommit['declaration_sha256'], training_binding_sha256=precommit['binding_sha256'],
        source_sha256={p:file_hash(repo/p) for p in sources}, checkpoint_sha256=precommit['checkpoint_sha256'],
        reset_seed=seeds[0], action_space_seed=seeds[1], engineering_stream_indices=[2,3],
        methods=['host','bca'], checkpoints_frozen=True, maximum_environment_transitions=9,
        maximum_physics_steps=36, maximum_actor_query_batches=4, state_absolute_tolerance=0.,
        repeated_observation_absolute_tolerance=0., reward_absolute_tolerance=1e-7,
        device='cpu', gpu_allocation=0, scientific_collection=False, outcomes=False, coverage=False,
        action_space_seeding='Both wrapper/native action RNGs seeded identically before each paired reset; no action sampling after reset.',
        ledger_path=str(ledger_path), ledger_physical_counts_are_reservations=True)
    settings_sha = write_artifact(attempt/'declaration.json', settings)
    caps = scope_caps([p['pair_id'] for p in declaration['pairs']], list(declaration['engineering_streams']))
    with ResourceLedger(ledger_path, caps, precommit['declaration_sha256']) as ledger:
        original_usage = ledger.audit()
        if original_usage['entries'] == 0:
            for old in history:
                ledger.reserve(old['token'], old['scopes'], old['environment'], old['physics'],
                               evidence=dict(prior_receipt_sha256=file_hash(prior_path), history=old))
        # An interrupted history import cannot silently become a new baseline.
        for old in history:
            row = ledger.db.execute('SELECT payload FROM entries WHERE token=?', (old['token'],)).fetchone()
            require(row is not None and json.loads(row[0])['evidence']['history'] == old,
                    'Incomplete/changed cumulative history; stop for separate inspection.')
        write_artifact(attempt/'resources-before.json', ledger.audit())
        real = None
        try:
            from runtime.environment import setup
            setup()
            import jax
            require(jax.default_backend() == 'cpu', 'Wrong backend.')
            prior_runtime = read_artifact(accepted/'runtime.json', file_hash(accepted/'runtime.json'))
            packages = {p:importlib.metadata.version(p) for p in prior_runtime['packages']}
            require(packages == prior_runtime['packages'], 'Changed trained query runtime.')
            write_artifact(attempt/'runtime.json', dict(python=sys.version, packages=packages,
                devices=[str(d) for d in jax.devices()], gpu_allocation=0))
            models = _load_pair(repo, lane, binding)
            real = AccountedSimulator(ledger, 'engineering/'+cell, str(attempt),
                                      'hopper-medium-v2', attempt/'simulator', 8)
            initial_hash = None
            comparisons = []
            for method in ('host','bca'):
                sim = real.sim
                sim.wrapper.action_space.seed(int(seeds[1])); sim.base.action_space.seed(int(seeds[1]))
                observation = sim.reset(int(seeds[0]))
                initial = sim.capture()
                saved_hash = write_artifact(attempt/(method+'-reset-state.json'), initial)
                if initial_hash is None:
                    initial_hash = sha(initial)
                require(sha(initial) == initial_hash, 'Paired full reset states differ.')
                for capture in ('reset','after-first-action'):
                    state = sim.capture()
                    action = models[method].actions(observation[None,:])[0]
                    write_artifact(attempt/f'{method}-{capture}-action.json', dict(observation=observation, action=action,
                        saved_state_sha256=sha(state), checkpoint_sha256=precommit['checkpoint_sha256'][method]))
                    first = real.step(action)
                    after_first = sim.capture()
                    sim.restore(state, expected_sha256=sim.snapshot_hash(state))
                    second = real.step(action)
                    after_second = sim.capture()
                    comparison = dict(method=method, capture=capture, first=first, second=second,
                        first_state=after_first, second_state=after_second,
                        records_exact=sha(first)==sha(second), states_exact=sha(after_first)==sha(after_second))
                    write_artifact(attempt/f'{method}-{capture}-repeat.json', comparison)
                    require(comparison['records_exact'] and comparison['states_exact'], 'Repeated real transition differs.')
                    comparisons.append(dict(method=method, capture=capture, records_exact=True, states_exact=True,
                        reward_error=first['reward_error'], incoming_dtype=action.dtype.str,
                        applied_dtype=first['applied_action'].dtype.str))
                    observation = second['observation']
            for adapter in models.values():
                adapter.assert_unchanged()
            require(real.calls == 8 and all(file_hash(repo/p)==v for p,v in settings['source_sha256'].items()),
                    'Unexpected calls or changed execution source.')
            final_usage = ledger.audit()
            write_artifact(attempt/'resources-after.json', final_usage)
            result = dict(schema='ood-real-connection-result-v1', declaration_sha256=settings_sha,
                accepted_engineering=True, full_state_roundtrips_exact=True, paired_reset_states_exact=True,
                comparisons=comparisons, environment_transitions=9, constructor_transitions=1,
                explicit_transitions=8, physics_steps=36, actor_query_batches=4,
                cumulative_global_reserved=final_usage['reserved']['global'],
                cumulative_cell_reserved=final_usage['reserved']['engineering/'+cell],
                scientific_state_collection_accepted=False, scientific_outcomes_accepted=False,
                coverage_accepted=False, actual_exit_must_be_checked_separately=True)
            write_artifact(attempt/'acceptance.json', result)
            return result
        except BaseException as exc:
            write_artifact(attempt/'failure.json', dict(error=repr(exc), ledger=ledger.audit(),
                           actual_calls_started=None if real is None else real.calls,
                           no_automatic_retry=True, scientific_outcomes_accepted=False))
            raise
        finally:
            if real is not None:
                real.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    for name in ('repo','accepted-attempt','lane','attempt','ledger'):
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(connect(args.repo, args.accepted_attempt, args.lane, args.attempt, args.ledger)))
