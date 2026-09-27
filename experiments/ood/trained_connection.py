"""One CPU-only connection check for the first accepted standard TD3 Hopper pair.

Reads the explicit accepted checkpoints once for a new adapter gate. Does not run
the learner, create an environment, collect outcomes or accept a simulator gate.
All input banks and reference arrays precede comparison, including failed gates.
"""
import argparse
from copy import deepcopy
import gzip
import importlib.metadata
import json
import os
from pathlib import Path
import sys

from experiments.ood.collect import file_hash, read_artifact, write_artifact, require


def connect(repo, attempt, lane):
    # Set before JAX/Flax or any host import. A failed CPU attempt is not retried
    # on another backend to obtain a passing answer.
    for key, value in dict(JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='',
                           XLA_PYTHON_CLIENT_PREALLOCATE='false', MUJOCO_PY_FORCE_CPU='1',
                           OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                           D4RL_SUPPRESS_IMPORT_ERROR='1').items():
        os.environ[key] = value
    binding_path = attempt/'training-binding.json'
    declaration_path = attempt/'full-declaration.json'
    binding = read_artifact(binding_path, file_hash(binding_path))
    declaration = read_artifact(declaration_path, file_hash(declaration_path))
    require(binding['pair_id'] == 'td3_bc/hopper/s202609171' and binding['training_accepted'] is True
            and binding['ready_for_collection'] is False, 'Wrong or unaccepted first pair.')
    out = attempt/'trained-connection-v1'
    out.mkdir(exist_ok=False)
    source_files = ['experiments/ood/trained_connection.py', 'experiments/ood/adapters.py',
                    'experiments/ood/collect.py', 'runtime/environment.py']
    engineering = declaration['engineering_streams']['td3_bc/hopper']
    precommit = dict(kind='trained_adapter_engineering_only', device='cpu',
        binding_sha256=file_hash(binding_path), declaration_sha256=file_hash(declaration_path),
        checkpoint_sha256={m: c['checkpoint_sha256'] for m, c in binding['checkpoints'].items()},
        seed_selection=int(engineering[0]), target_seed=int(engineering[1]), selected_rows=64,
        environment_transitions=0, physics_steps=0, training_updates=0,
        maximum_direct_and_restored_query_batches=12,
        action_absolute_tolerance=1e-6, target_and_width_comparison='exact arrays',
        source_sha256={p:file_hash(repo/p) for p in source_files})
    write_artifact(out/'precommit.json', precommit)
    for name, expected in binding['frozen_source_sha256'].items():
        require(file_hash(repo/name) == expected, 'Changed frozen scientific source: '+name)
    for item in binding['inputs'].values():
        # The binding may contain Windows paths when built on the shared drive.
        name = item['path'].replace('\\', '/')
        if len(name) > 2 and name[1:3] == ':/':
            name = '/mnt/'+name[0].lower()+name[2:]
        require(file_hash(name) == item['sha256'], 'Changed training acceptance input.')

    from runtime.environment import setup
    setup()
    import jax
    import jax.numpy as jnp
    import h5py
    import numpy as np
    from flax import serialization as S
    from runtime.config import typed
    from experiments.ood.adapters import CheckpointAdapter, parity_gate
    from unittest.mock import patch

    require(jax.default_backend() == 'cpu', 'Declared CPU backend not active.')
    runtime = {p:importlib.metadata.version(p) for p in ('jax', 'jaxlib', 'flax', 'numpy', 'h5py')}
    write_artifact(out/'runtime.json', dict(python=sys.version, executable=sys.executable,
        packages=runtime, devices=[str(d) for d in jax.devices()], simulator_calls_forbidden=True))
    manifest = json.loads(gzip.decompress((repo/'experiments/standard_bca/manifest.json.gz').read_bytes()))
    query_count = 0
    results = {}
    # prepare(raw,...) never constructs a simulator; forbid any accidental path.
    with patch('gym.make', side_effect=RuntimeError('Simulator construction forbidden in zero-step adapter gate')):
        for method, cp in binding['checkpoints'].items():
            run = lane/'runs'/cp['run_id']
            record, = [r for r in manifest['runs'] if r['row']['run_id'] == cp['run_id']]
            runner, args, spec, protocol = typed(deepcopy(record['row']))
            cfg = spec.config()
            audit_input = binding['inputs'][method+'/accepted_audit']['path'].replace('\\','/')
            if len(audit_input)>2 and audit_input[1:3]==':/':
                audit_input='/mnt/'+audit_input[0].lower()+audit_input[2:]
            audit = json.loads(Path(audit_input).read_text(encoding='utf-8'))
            preparation_path = run/'preparation.json'
            require(file_hash(preparation_path) == audit['evidence_sha256']['preparation.json'], 'Changed preparation evidence.')
            original = json.loads(preparation_path.read_text(encoding='utf-8'))['metadata']
            with h5py.File(original['raw_identity']['path'], 'r') as stream:
                raw = {k:stream[k][()] for k in original['raw_array_hashes']}
            prepared = runner.prepare(raw, args, spec, protocol, max_action=1., max_episode_steps=1000,
                                      raw_identity=original['raw_identity'])
            require(prepared.metadata['run_input_hashes'] == cp['preparation_input_hashes'],
                    'Actual prepared arrays/norm/IDs differ from accepted training.')
            selection = np.sort(np.random.default_rng(precommit['seed_selection']).choice(
                len(prepared.heldout.obs), precommit['selected_rows'], replace=False))
            batch = jax.tree.map(lambda x:x[selection], prepared.heldout)
            raw_obs = np.asarray(batch.obs)*np.asarray(prepared.obs_std)+np.asarray(prepared.obs_mean)
            normalized = (jnp.asarray(raw_obs)-prepared.obs_mean)/prepared.obs_std
            key = jax.random.PRNGKey(precommit['target_seed'])
            write_artifact(out/(method+'-inputs.json'), dict(prepared_hashes=prepared.metadata['run_input_hashes'],
                heldout_positions=selection, converted_ids=np.asarray(prepared.heldout_ids)[selection],
                raw_observations=raw_obs, normalized_observations=np.asarray(normalized),
                transition=[np.asarray(x) for x in batch], target_key=np.asarray(key)))
            checkpoint = Path(cp['checkpoint_path'])
            require(checkpoint == run/'checkpoint_1000000.msgpack' and file_hash(checkpoint) == cp['checkpoint_sha256'],
                    'Wrong final checkpoint file.')
            rng, template, models = runner.P.initialize(args, cfg, prepared.training.obs.shape[1],
                                                        prepared.training.action.shape[1], prepared.max_action)
            direct = S.from_bytes(dict(state=template, training_rng=rng, step=jnp.int32(0)), checkpoint.read_bytes())
            state = direct['state']
            references = dict(actions=np.asarray(models[0].apply(state.native.actor.params, normalized)),
                              target=np.asarray(runner.P.native_target(args, models, state.native, batch, key)))
            query_count += 2
            if method == 'bca':
                readout = runner.P.bc_readout(models, state, batch, cfg.blend)
                references.update(width=np.asarray(readout.width), dose=np.asarray(readout.dose))
                query_count += 1
            write_artifact(out/(method+'-direct-references.json'), references)
            adapter = CheckpointAdapter(checkpoint, cp['checkpoint_sha256'], 'td3_bc', args, cfg, prepared)
            actual = dict(actions=adapter.actions(raw_obs), target=adapter.targets(batch, key))
            noise_action, target_q = adapter.target_components(batch, key)
            query_count += 3
            actual['independent_target'] = np.asarray(batch.reward+(1-batch.done)*args.discount*target_q)
            if method == 'bca':
                score = adapter.score(batch.obs, batch.action, normalized=True)
                actual.update(width=score['width'], dose=score['dose'], score=score)
                query_count += 1
            errors = {k:np.abs(np.asarray(actual[k], np.float64)-np.asarray(ref, np.float64))
                      for k, ref in references.items()}
            write_artifact(out/(method+'-restored-and-errors.json'), dict(actual=actual, error=errors,
                target_noise_actions=np.asarray(noise_action), devices=[str(d) for d in jax.devices()]))
            parity_gate(out/(method+'-action-gate.npz'), references['actions'], actual['actions'], device='cpu')
            for key_name in ('target', 'width', 'dose'):
                if key_name in references:
                    require(np.array_equal(references[key_name], actual[key_name]), 'Exact restored '+key_name+' gate failed.')
            require(np.array_equal(references['target'], actual['independent_target']), 'Independent native target arithmetic failed.')
            adapter.assert_unchanged()
            results[method] = dict(accepted=True, checkpoint_sha256=cp['checkpoint_sha256'],
                checkpoint_counters=adapter.counts, prepared_hashes=prepared.metadata['run_input_hashes'],
                selected_rows=len(selection), maximum_action_error=float(errors['actions'].max()),
                target_exact=True, width_and_dose_exact=True if method=='bca' else None,
                native_width='N/A' if method=='host' else None)
    require(query_count == 12, 'Unexpected query accounting.')
    result = dict(schema='ood-trained-connection-v1', accepted=True, pair_id=binding['pair_id'],
        training_binding_sha256=file_hash(binding_path), precommit_sha256=file_hash(out/'precommit.json'),
        methods=results, device='cpu', direct_and_restored_query_batches=query_count,
        environment_transitions=0, physics_steps=0, training_updates=0,
        trained_action_parity_accepted=True, simulator_execution_accepted=False,
        pre_outcome_banks_accepted=False, ready_for_collection=False, residual_coverage_accepted=False,
        note='This accepts frozen trained-checkpoint queries only, not real simulator outcomes or OOD ranking.')
    write_artifact(out/'acceptance.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True, type=Path)
    parser.add_argument('--attempt', required=True, type=Path)
    parser.add_argument('--lane', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(connect(args.repo, args.attempt, args.lane), indent=2))
