"""Paired auxiliary-only refits on frozen checkpoints and an already used bank.

The experiment is exploratory and never updates an actor, critic, target, or
their optimizer. Each paired native/BCA fit uses identical initialization,
minibatch indices, fitting keys and calibration key; residual targets differ
because the six restored checkpoints differ. No simulator call is made here.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
V4 = HERE.parent / 'common_action_bca_dev_v4'
spec = importlib.util.spec_from_file_location('_paired_v4_runner', V4 / 'runner.py')
v4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v4)
v3, old, runtime = v4.v3, v4.old, v4.runtime

SEEDS = (0, 1, 2)
ARMS = ('native', 'bca')
CONTEXTS = ('ordinary', 'strong_current')
REPLICATES = 4
STEPS = 2000
BATCH = 256
MAX_SECONDS = 7200
INIT_SEED_BASE = 2027011000
CAL_SEED_BASE = 2027015000
SEAL_FILE = HERE / 'PAIRED_REFITS_SEAL.json'
SEALED_SOURCES = ('PROTOCOL.md', 'paired_refits.py', 'test_paired_refits.py')


def require(value, message):
    if not value:
        raise ValueError(message)


def model_seed(seed, replicate):
    require(seed in SEEDS and 0 <= replicate < REPLICATES, 'bad seed or replicate')
    return INIT_SEED_BASE + seed * 10 + replicate


def calibration_seed(seed, replicate):
    require(seed in SEEDS and 0 <= replicate < REPLICATES, 'bad seed or replicate')
    return CAL_SEED_BASE + seed * 10 + replicate


def choices(scores):
    a = np.asarray(scores, dtype=np.float64)
    require(a.ndim == 2 and a.shape[1] == 15 and np.isfinite(a).all(), 'invalid scores')
    # Match the frozen v4 tie rule, including the tolerance.
    return np.argmax(a.max(axis=1, keepdims=True) - a <= 1e-6, axis=1)


def paired_step(pairs, batch, key, indices, ledger, traces):
    """Apply the same immutable input tuple to each frozen-critic fit."""
    for arm in ARMS:
        current = pairs[arm]
        traces[arm]['minibatches'].update(np.asarray(indices, dtype='<i8').tobytes())
        traces[arm]['fit_keys'].update(np.asarray(key, dtype='<u4').tobytes())
        current['aux'], metrics = current['fit'](current['aux'], batch, key)
        require(bool(np.asarray(metrics['scale_fit_accepted'])), 'auxiliary fit rejected')
        ledger['auxiliary_updates'] += 1


def contrast_summary(widths, q_native, q_bca, sources):
    """Describe paired width action contrasts and original-bank choices.

    ``widths`` is [replicate, arm, state, action]. Anchor every contrast at
    the same native-actor action in bank slot 0. This was a development bank,
    not fresh validation, and the state/action entries are dependent.
    """
    w = np.asarray(widths, np.float64)
    qn, qb = np.asarray(q_native, np.float64), np.asarray(q_bca, np.float64)
    source = np.asarray(sources)
    require(w.shape == (REPLICATES, 2, 32, 15) and qn.shape == qb.shape == (32, 15)
            and source.shape == (32,) and np.isfinite(w).all()
            and np.isfinite(qn).all() and np.isfinite(qb).all(), 'bad paired-bank arrays')
    out = {}
    for name, mask in (('all', np.ones(32, bool)), ('actor_visited', source != 'random_prefix'),
                       ('random_prefix', source == 'random_prefix')):
        z = 3.0 * ((w[:, 1, mask, :] - w[:, 1, mask, 0, None])
                   - (w[:, 0, mask, :] - w[:, 0, mask, 0, None]))
        disagreements = []
        for r in range(REPLICATES):
            a = choices(qn[mask] - 3.0 * w[r, 0, mask, :])
            b = choices(qb[mask] - 3.0 * w[r, 1, mask, :])
            disagreements.append(int(np.sum(a != b)))
        out[name] = {'states': int(mask.sum()), 'anchor_slot': 0,
                     'rms_paired_mean_contrast': float(np.sqrt(np.mean(np.mean(z, axis=0) ** 2))),
                     'rms_across_refit_sd': float(np.sqrt(np.mean(np.var(z, axis=0, ddof=1)))),
                     'mean_abs_contrast': float(np.mean(np.abs(z))),
                     'winner_disagreements_by_refit': disagreements}
    return out


def write_json(path, value):
    with Path(path).open('x', encoding='utf8', newline='\n') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def input_pins():
    v4_seal, _, archive = v4.check_seal()
    result_path = V4 / 'diagnostic_run_01' / 'RESULT.json'
    result = json.loads(result_path.read_bytes())
    require(result.get('status') == 'completed_common_action_continuation_diagnostic'
            and result.get('source_seal_sha256') == v4_seal, 'v4 result identity differs')
    require(v4.digest(archive) == result['prior_calibration_archive_sha256'],
            'v4 calibration archive differs')
    return {'v4_source_seal_sha256': v4_seal, 'v4_result_sha256': v4.digest(result_path),
            'v3_calibration_archive_sha256': v4.digest(archive)}


def seal():
    require(not SEAL_FILE.exists(), 'paired refit source seal already exists')
    write_json(SEAL_FILE, {'schema': 'bca.paired_auxiliary_refits.v5', 'inputs': input_pins(),
                           'sources': {name: v4.digest(HERE / name) for name in SEALED_SOURCES}})
    return v4.digest(SEAL_FILE)


def check_seal():
    data = json.loads(SEAL_FILE.read_bytes())
    require(data.get('schema') == 'bca.paired_auxiliary_refits.v5'
            and set(data.get('sources', {})) == set(SEALED_SOURCES), 'invalid paired refit source seal')
    require(all(v4.digest(HERE / name) == expected for name, expected in data['sources'].items()),
            'paired refit protocol or source differs from seal')
    require(data.get('inputs') == input_pins(), 'paired refit input pins differ')
    return v4.digest(SEAL_FILE)


def run(output):
    start = time.monotonic()
    source_seal = check_seal()
    v4_seal, v3prior, archive = v4.check_seal()
    prior_path = V4 / 'diagnostic_run_01' / 'RESULT.json'
    prior = json.loads(prior_path.read_bytes())
    require(prior.get('status') == 'completed_common_action_continuation_diagnostic'
            and prior.get('source_seal_sha256') == v4_seal, 'v4 result identity differs')
    require(v4.digest(archive) == prior['prior_calibration_archive_sha256'], 'calibration archive differs')
    terminal, _, _ = old.completed()
    proto_raw = (v3.RERUN / 'protocol.json').read_bytes()
    protocol = json.loads(proto_raw)
    require(old.sha(proto_raw) == terminal['protocol_sha256'], 'training protocol differs')
    native = runtime.load_native()
    receipt = old.cpu_receipt(native)
    jax, jnp = native.jax, native.jnp
    cal, row_ids, episode_ids = v4.archived_calibration(native, archive)
    ledger = {'auxiliary_updates': 0, 'calibration_refreshes': 0, 'actor_updates': 0,
              'critic_updates': 0, 'actor_target_updates': 0, 'critic_target_updates': 0,
              'simulator_transitions': 0, 'candidate_width_rows': 0}

    def progress(event, **extra):
        require(time.monotonic() - start < MAX_SECONDS, 'paired refit deadline exceeded')
        item = {'event': event, 'elapsed_seconds': time.monotonic() - start,
                'committed_counts': dict(ledger), **extra}
        with (output / 'progress.jsonl').open('a', encoding='utf8') as f:
            f.write(json.dumps(item, sort_keys=True, allow_nan=False) + '\n')
            f.flush()
        print(json.dumps(item, sort_keys=True), flush=True)

    source_hashes = {}
    cases = []
    progress('started')
    for seed in SEEDS:
        data = runtime.module('_paired_refits_data', v3.RERUN / 'data.py').load_condition(native, 'hole')
        adapter, template = runtime._initialize_data(protocol, seed, 'hole', native, data, receipt)
        ctx = adapter.context
        train_ids = np.asarray(data.source_row_ids, np.int64)
        require(len(train_ids) == 67108 and len(np.unique(train_ids // 100)) == 768,
                'training episode identity differs')
        require(old.sha(train_ids.astype('<i8').tobytes()) == v3prior['training_row_ids_sha256'],
                'training source rows differ from v3')
        checkpoints = {}
        for arm in ARMS:
            state, hashes, norm, branch = old.checkpoint(seed, arm, protocol, old.sha(proto_raw), adapter, template)
            source_hashes.update(hashes)
            before = runtime.tree_digest(state)
            v3case = next(c for c in v3prior['cases'] if c['seed'] == seed and c['arm'] == arm)
            require(before == v3case['before'], 'frozen state digest differs from v3')
            checkpoints[arm] = {'state': state, 'before': before, 'norm': norm, 'branch': branch}
        oldcase = next(c for c in prior['cases'] if c['seed'] == seed)
        bank = {}
        for context in CONTEXTS:
            p = oldcase['contexts'][context]['per_probe']
            obs = np.asarray(p['observations'], np.float32)
            action = np.asarray(p['actions'], np.float32)
            require(obs.shape == (32, 14) and action.shape == (32, 15, 4), 'v4 bank shape differs')
            bank[context] = {'obs': obs, 'action': action,
                             'q_native': np.asarray(p['scores']['native']['raw_q1'], np.float64),
                             'q_bca': np.asarray(p['scores']['bca']['raw_q1'], np.float64),
                             'sources': np.asarray(p['sources'])}
        bank_widths = {c: np.empty((REPLICATES, 2, 32, 15), np.float32) for c in CONTEXTS}
        replicas = []
        # Static model structure and authenticated normalization are identical
        # across refits of the same checkpoint. Native and auxiliary parameter
        # values remain explicit dynamic inputs through ``state``. Reuse the
        # compiled function instead of recompiling it for each initialization.
        fit_cache = {}
        for rep in range(REPLICATES):
            init = model_seed(seed, rep)
            calkey = jax.random.PRNGKey(calibration_seed(seed, rep))
            pairs = {}
            pipeline = v3.q1_bca.make_q1_pipeline(ctx.pipeline)
            for arm in ARMS:
                saved = checkpoints[arm]
                models, aux, config, args = v3.q1_bca.fresh_auxiliary_exact_normalization(
                    ctx, saved['state'].microstep.agent_state, init,
                    expected_mean=saved['norm']['saved_mean'], expected_std=saved['norm']['saved_std'])
                if arm not in fit_cache:
                    fit_cache[arm] = jax.jit(lambda st, b, k, _args=args, _config=config, _models=models:
                                             pipeline.fit_scale(_args, _config, _models, st, b, k))
                fit = fit_cache[arm]
                pairs[arm] = {'models': models, 'aux': aux, 'config': config, 'args': args, 'fit': fit,
                              'initial_digest': runtime.tree_digest((aux.calibrator, aux.posterior,
                                                                     aux.residual_scale))}
            require(pairs['native']['initial_digest'] == pairs['bca']['initial_digest'],
                    'paired auxiliary initializations differ')
            rng = np.random.Generator(np.random.PCG64(init))
            key = jax.random.PRNGKey(init)
            traces = {arm: {'minibatches': hashlib.sha256(), 'fit_keys': hashlib.sha256()}
                      for arm in ARMS}
            for step in range(STEPS):
                ix = rng.integers(0, len(train_ids), BATCH)
                batch = old.sliced(jax, ctx.train_transition, ix)
                key, sub = jax.random.split(key)
                paired_step(pairs, batch, sub, ix, ledger, traces)
                if (step + 1) % 250 == 0:
                    progress('paired_steps', seed=seed, replicate=rep, steps_per_arm=step + 1)
            hashes = {arm: {name: value.hexdigest() for name, value in traces[arm].items()} for arm in ARMS}
            require(hashes['native'] == hashes['bca'], 'paired minibatches or fit keys differ')
            replica = {'replicate': rep, 'init_and_sampler_seed': init,
                       'calibration_seed': calibration_seed(seed, rep),
                       'paired_trace_sha256': hashes, 'arms': {}}
            for arm_i, arm in enumerate(ARMS):
                current = pairs[arm]
                aux, flags, details = pipeline.refresh(current['args'], current['config'], current['models'],
                                                       current['aux'], cal, calkey, heldout_ids=row_ids)
                ledger['calibration_refreshes'] += 1
                require(int(np.asarray(aux.calibrator.step)) == STEPS and bool(flags['posterior_certified'])
                        and bool(np.asarray(ctx.pipeline.reference_valid(aux.posterior))), 'calibration rejected')
                current['aux'] = aux
                for context in CONTEXTS:
                    x = bank[context]
                    repeated = jnp.asarray(np.repeat(x['obs'], 15, axis=0))
                    actions = jnp.asarray(x['action'].reshape(-1, 4))
                    width, usable, valid = pipeline.live_width(current['models'], aux.posterior, repeated, actions)
                    require(bool(np.asarray(usable).all()) and bool(np.asarray(valid).all()), 'invalid width')
                    value = np.asarray(width, np.float32).reshape(32, 15)
                    require(np.isfinite(value).all() and np.all(value >= 0), 'invalid width values')
                    bank_widths[context][rep, arm_i] = value
                    ledger['candidate_width_rows'] += 32 * 15
                from flax import serialization
                raw = serialization.to_bytes({'calibrator': aux.calibrator, 'posterior': aux.posterior,
                                              'residual_scale': aux.residual_scale, 'aux_rng': key})
                filename = f'seed{seed}_rep{rep}_{arm}_q1_aux.msgpack'
                with (output / filename).open('xb') as stream:
                    stream.write(raw)
                replica['arms'][arm] = {'initial_digest': current['initial_digest'],
                    'threshold': float(np.asarray(aux.posterior.threshold)),
                    'residual_unit': float(np.asarray(aux.residual_scale)),
                    'artifact': {'path': filename, 'sha256': old.sha(raw), 'bytes': len(raw)},
                    'calibration_flags': runtime.host(flags), 'calibration_details': runtime.host(details)}
            replicas.append(replica)
            for arm in ARMS:
                require(runtime.tree_digest(checkpoints[arm]['state']) == checkpoints[arm]['before'],
                        'frozen learner changed during auxiliary fitting')
            progress('paired_refit_complete', seed=seed, replicate=rep)
        contexts = {}
        for context in CONTEXTS:
            path = output / f'seed{seed}_{context}_widths.npz'
            with path.open('xb') as stream:
                np.savez_compressed(stream, widths=bank_widths[context])
            x = bank[context]
            contexts[context] = {'summary': contrast_summary(bank_widths[context], x['q_native'], x['q_bca'],
                                                               x['sources']),
                                 'width_artifact': {'path': path.name, 'sha256': v4.digest(path)}}
        cases.append({'seed': seed, 'checkpoints': {a: checkpoints[a]['before'] for a in ARMS},
                      'replicates': replicas, 'contexts': contexts})
        write_json(output / f'seed{seed}_case.json', cases[-1])
        progress('seed_complete', seed=seed)
    require(ledger['auxiliary_updates'] == 48000 and ledger['calibration_refreshes'] == 24
            and ledger['candidate_width_rows'] == 23040 and all(ledger[k] == 0 for k in
            ('actor_updates', 'critic_updates', 'actor_target_updates', 'critic_target_updates',
             'simulator_transitions')), 'final accounting differs')
    for path, digest in source_hashes.items():
        require(v4.digest(path) == digest, 'checkpoint source changed')
    require(v4.check_seal()[0] == v4_seal and check_seal() == source_seal,
            'source or prior seal changed')
    result = {'status': 'completed_paired_auxiliary_refits', 'runtime': receipt,
              'source_seal_sha256': source_seal, 'v4_source_seal_sha256': v4_seal,
              'v4_result_sha256': v4.digest(prior_path),
              'calibration_archive_sha256': v4.digest(archive), 'calibration_rows': row_ids.tolist(),
              'calibration_episode_ids': episode_ids.tolist(), 'source_hashes': source_hashes,
              'ledger': ledger, 'cases': cases, 'elapsed_seconds': time.monotonic() - start,
              'limitations': ['v3 calibration episodes and v4 action banks were already used in development',
                  'four matched refits do not establish the causal effect of BCA training',
                  'width is a symmetric TD residual quantity, not signed true-Q error']}
    write_json(output / 'RESULT.json', result)
    progress('complete')
    return result


if __name__ == '__main__':
    if sys.argv[1:] == ['--seal']:
        print(seal())
        sys.exit(0)
    if sys.argv[1:] == ['--check-seal']:
        print(check_seal())
        sys.exit(0)
    require(sys.argv[1:] == ['--run'], 'Use --seal, --check-seal, or --run')
    require(os.environ.get('JAX_PLATFORMS') == 'cpu', 'explicit CPU required')
    check_seal()
    output = HERE / 'paired_refits_run_01'
    output.mkdir(exist_ok=False)
    try:
        print(json.dumps({'status': run(output)['status']}), flush=True)
    except BaseException as exc:
        write_json(output / 'FAILURE.json', {'status': 'failed_no_automatic_retry', 'error': str(exc),
                   'traceback': traceback.format_exc(),
                   'counts_note': 'See progress.jsonl; work after last saved event may be unknown'})
        raise
