"""Collect a disjoint, bounded behavior bank for the frozen diagnostic.

Only new behavior trajectories are collected. No learner is called, and the
collector's retained-row choice never reads Q, scores, or evaluation outcomes.
"""
from pathlib import Path
import hashlib
import importlib.util
import json

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
RERUN = BASE / 'central_quartile_training_v1_evaluation_rerun1'
RECORDER = BASE.parent / 'titan_20261026/payload/transfer_bundle/core_v4/thrusterchain/datasets.py'
ROOT_SEED = 2026122300
ROW_SEED = 2026122301
EPISODES = 128
HORIZON = 100
NEW_BANK_SEEDS = list(range(2026122000, 2026122064)) + list(range(2026122100, 2026122164))
NEW_PREFIX_SEEDS = [2026122200, 2026122201]
NEW_CALIBRATION_KEYS = list(range(2026122400, 2026122406))
PREVIOUS = BASE / 'signed_action_bca_dev_v2'


def require(value, message):
    if not value:
        raise ValueError(message)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    instance = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(instance)
    return instance


def seed_audit(recorder, original_seeds, old_protocol, consumed_seeds=()):
    """Audit all source seed vectors, including metadata-only protected roles."""
    resets = np.asarray([recorder._episode_reset_seed(ROOT_SEED, i) for i in range(EPISODES)], np.int64)
    old = set(int(x) for x in np.asarray(original_seeds).ravel())
    old.update(int(x) for x in consumed_seeds)
    old.update((2026121401, 2026121402, 2026121500, 2026121501, 2026121510))
    for context in ('ordinary', 'strong_current'):
        spec = old_protocol['evaluation'][context]
        old.update(int(x) for x in spec['q_seeds'])
        old.update(int(x) for x in spec['free_seeds'])
        old.add(int(spec['prefix_seed']))
    old.add(int(old_protocol['evaluation']['parity_seed']))
    # Older development/confirmation blocks explicitly reserved by the source.
    old.update(range(2026102400, 2026102464))
    old.update(range(2026102500, 2026102564))
    old.update(range(2026102700, 2026102764))
    proposed = list(map(int, resets)) + NEW_BANK_SEEDS + NEW_PREFIX_SEEDS + NEW_CALIBRATION_KEYS + [ROOT_SEED, ROW_SEED]
    require(len(set(proposed)) == len(proposed), 'New reset and stream seeds overlap')
    require(not old.intersection(proposed), 'New seeds overlap original or earlier evaluation seeds')
    return resets


def preflight():
    """Validate recorder identity and all source/evaluation seed roles without simulation."""
    recorder = module('_fresh_frozen_diagnostic_recorder', RECORDER)
    logged = module('_fresh_frozen_diagnostic_logged', BASE / 'bca_one_sided_v1/logged_data.py')
    source = logged.provenance()  # Reads seed/role metadata, not protected tuples.
    prior = json.loads((RERUN / 'protocol.json').read_text(encoding='utf-8'))
    v2 = json.loads((PREVIOUS / 'diagnostic_run_01/RESULT.json').read_bytes())
    require(v2.get('status') == 'completed_frozen_checkpoint_diagnostic', 'Previous diagnostic incomplete')
    root = PREVIOUS / 'diagnostic_run_01'
    prior_cal = root / v2['fresh_data']['artifact']
    require(hashlib.sha256(prior_cal.read_bytes()).hexdigest() == v2['fresh_data']['artifact_sha256'],
            'Previous calibration archive changed')
    with np.load(prior_cal, allow_pickle=False) as archive:
        consumed = np.asarray(archive['episode_seeds'], np.int64).tolist()
    for context in ('ordinary', 'strong_current'):
        bank = root / (context + '_bank.npz')
        require(hashlib.sha256(bank.read_bytes()).hexdigest() == v2['banks'][context]['npz_sha256'],
                'Previous evaluation bank changed')
        with np.load(bank, allow_pickle=False) as archive:
            consumed.extend(np.asarray(archive['q_seeds'], np.int64).tolist())
            consumed.extend(np.asarray(archive['free_seeds'], np.int64).tolist())
    seeds = seed_audit(recorder, source['vectors']['episode_seeds'], prior, consumed)
    require(source['metadata']['behavior']['epsilon'] == .25
            and source['metadata']['behavior']['name'] == 'continuous_region_mixture_v1',
            'Original behavior law differs')
    return recorder, logged, seeds


def collect(native, core, reference, ledger, deadline, output):
    """Collect 128 complete 100-step episodes; return 64+64 intact tuples."""
    recorder, logged, seeds = preflight()
    fields = {
        'observations': np.empty((EPISODES, HORIZON + 1, 14), np.float32),
        'actions': np.empty((EPISODES, HORIZON, 4), np.float32),
        'rewards': np.empty((EPISODES, HORIZON), np.float64),
        'terminated': np.zeros((EPISODES, HORIZON), bool),
    }
    before = ledger['transition_steps_completed']
    for i, seed in enumerate(seeds):
        reference._check_time(deadline)
        require(ledger['resets_attempted'] < ledger['max_resets'], 'Collection reset cap exceeded')
        env = reference._env(core)
        try:
            ledger['resets_attempted'] += 1
            obs, info = env.reset(seed=int(seed))
            ledger['resets_completed'] += 1
            reference._check_env(env, obs, False, False, info, 0)
            fields['observations'][i, 0] = obs
            rng = np.random.Generator(np.random.PCG64(recorder._stream_seed_sequence(ROOT_SEED, 'actions', i)))
            for t in range(HORIZON):
                requested, _, _ = recorder.sample_requested_action(rng, .25)
                action = requested.astype(np.float32)
                next_obs, reward = reference._step(env, action, ledger, 'calibration', deadline, None)
                fields['observations'][i, t + 1] = next_obs
                fields['actions'][i, t] = action
                fields['rewards'][i, t] = reward
                fields['terminated'][i, t] = (t == HORIZON - 1)
        finally:
            env.close()
    require(ledger['transition_steps_completed'] - before == EPISODES * HORIZON,
            'Collection transition count differs')
    obs, action = fields['observations'][:, :-1], fields['actions']
    hole = (obs[..., 10].astype(np.float64) + obs[..., 11].astype(np.float64) >= .4) & (np.abs(action[..., 0]) <= .25)
    rng = np.random.Generator(np.random.PCG64(ROW_SEED))
    times = []
    for i in range(EPISODES):
        eligible = np.flatnonzero(~hole[i])
        require(len(eligible) > 0, 'Fresh episode has no eligible calibration row; no backfill')
        times.append(int(eligible[rng.integers(len(eligible))]))
    times = np.asarray(times, np.int64)
    ix = np.arange(EPISODES)
    ids = np.arange(400000000000, 400000000000 + EPISODES * 100, 100, dtype=np.int64) + times
    rows = {
        'obs': obs[ix, times], 'action': action[ix, times],
        'reward': fields['rewards'][ix, times].astype(np.float32),
        'next_obs': fields['observations'][ix, times + 1],
        'done': fields['terminated'][ix, times].astype(np.float32),
    }
    require(not np.any(hole[ix, times]) and all(np.isfinite(v).all() for v in rows.values()),
            'Invalid or excluded selected transition')
    archive = Path(output) / 'fresh_behavior_episodes.npz'
    with archive.open('xb') as f:
        np.savez_compressed(f, **fields, episode_seeds=seeds, selected_times=times,
                            selected_row_ids=ids)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    def batch(sl):
        return native.Transition(**{k: np.asarray(v[sl], np.float32) for k, v in rows.items()})
    receipt = {
        'episodes': EPISODES, 'steps': EPISODES * HORIZON, 'episode_seeds': seeds.tolist(),
        'root_seed': ROOT_SEED, 'row_seed': ROW_SEED, 'row_ids': ids.tolist(),
        'selected_times': times.tolist(), 'calibration_indices': list(range(64)),
        'check_indices': list(range(64, 128)), 'calibration_row_ids': ids[:64].tolist(),
        'check_row_ids': ids[64:].tolist(), 'source_dataset_sha256': logged.EXPECTED_DATA_SHA,
        'recorder_sha256': hashlib.sha256(RECORDER.read_bytes()).hexdigest(),
        'artifact': archive.name, 'artifact_sha256': digest,
        'law': 'recorded start; state-independent continuous_region_mixture_v1 epsilon 0.25; float32 execution',
        'selection': 'one uniform retained non-hole tuple per episode; no backfill',
        'independence': 'new disjoint reset/action streams checked against original, v1 and v2; original protected tuples not read',
    }
    return batch(slice(0, 64)), batch(slice(64, 128)), receipt
