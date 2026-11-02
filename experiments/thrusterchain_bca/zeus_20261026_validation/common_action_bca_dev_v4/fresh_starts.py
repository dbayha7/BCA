"""Fresh random-prefix and frozen-actor-prefix states for the v4 diagnostic.

The first stream is shared across seeds. The actor-visited streams are generated
with frozen actors and independent resets. Nothing here reads simulator Qpi or
changes model parameters. Prefix steps are charged to the caller's ledger.
"""
from pathlib import Path
import hashlib
import importlib.util
import json

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
V2 = BASE / 'signed_action_bca_dev_v2'
V3 = BASE / 'signed_action_bca_dev_v3'
RERUN = BASE / 'central_quartile_training_v1_evaluation_rerun1'
TIMES = (0, 25, 50, 75)
RANDOM_SEEDS = {'ordinary': 2026123000, 'strong_current': 2026123100}
VISITED_SEEDS = {'ordinary': 2026123200, 'strong_current': 2026124200}
PREFIX_SEEDS = {'ordinary': 2026125000, 'strong_current': 2026125001}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    out = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(out)
    return out


def random_seeds(context):
    require(context in RANDOM_SEEDS, 'unknown context')
    return np.arange(RANDOM_SEEDS[context], RANDOM_SEEDS[context]+16, dtype=np.int64)


def visited_seeds(context, seed, arm):
    require(context in VISITED_SEEDS and seed in (0,1,2) and arm in ('native','bca'),
            'unknown visited stream')
    base = VISITED_SEEDS[context] + 20*seed + (8 if arm == 'bca' else 0)
    return np.arange(base, base+8, dtype=np.int64)


def seed_audit():
    """Authenticate prior seed archives and refuse any reset/stream collision.

    Original dataset access is metadata-only; protected transition arrays are
    never read. V2/V3 values were already consumed development diagnostics.
    """
    v3fresh = module('_v4_prior_fresh', V3/'fresh_data.py')
    logged = v3fresh.module('_v4_logged_metadata', BASE/'bca_one_sided_v1/logged_data.py')
    src = logged.provenance()
    old = set(map(int, np.asarray(src['vectors']['episode_seeds']).ravel()))
    old.add(int(src['metadata']['seed']))
    protocol = json.loads((RERUN/'protocol.json').read_bytes())
    old.add(int(protocol['evaluation']['parity_seed']))
    for context in ('ordinary','strong_current'):
        p = protocol['evaluation'][context]
        old.update(map(int,p['q_seeds'])); old.update(map(int,p['free_seeds']))
        old.add(int(p['prefix_seed']))
    for start in (2026102400,2026102500,2026102700):
        old.update(range(start,start+64))
    old.update((2026121401,2026121402,2026121500,2026121501,2026121510,
                2026122200,2026122201,2026122300,2026122301))
    old.update(range(2026122400,2026122406))
    for previous in (V2,V3):
        root = previous/'diagnostic_run_01'
        result = json.loads((root/'RESULT.json').read_bytes())
        expected = 'completed_frozen_checkpoint_diagnostic' if previous == V2 else 'completed_expanded_frozen_diagnostic'
        require(result['status'] == expected, 'previous diagnostic incomplete')
        behavior = root/result['fresh_data']['artifact']
        require(digest(behavior) == result['fresh_data']['artifact_sha256'], 'old behavior archive changed')
        with np.load(behavior,allow_pickle=False) as archive:
            old.update(map(int,archive['episode_seeds']))
        for context in ('ordinary','strong_current'):
            if previous == V2:
                archive_path = root/(context+'_bank.npz'); entry = result['banks'][context]
                keys = ('q_seeds','free_seeds')
            else:
                archive_path = root/(context+'_pool_and_selection.npz'); entry = result['pools'][context]
                keys = ('seeds',)
            require(digest(archive_path) == entry['npz_sha256'], 'old state archive changed')
            with np.load(archive_path,allow_pickle=False) as archive:
                for key in keys: old.update(map(int,archive[key]))
    proposed = []
    for context in ('ordinary','strong_current'):
        proposed.extend(map(int,random_seeds(context)))
        proposed.append(PREFIX_SEEDS[context])
        for seed in (0,1,2):
            for arm in ('native','bca'):
                proposed.extend(map(int,visited_seeds(context,seed,arm)))
    require(len(proposed)==130 and len(set(proposed))==130, 'new streams are not distinct')
    require(not old.intersection(proposed), 'new streams overlap original or consumed prior streams')
    return {'new_reset_seeds': len(proposed)-2, 'new_prefix_streams': 2,
            'original_dataset_sha256':logged.EXPECTED_DATA_SHA,
            'v2_result_sha256':digest(V2/'diagnostic_run_01/RESULT.json'),
            'v3_result_sha256':digest(V3/'diagnostic_run_01/RESULT.json'),
            'protected_transition_rows_read':0, 'checked_prior_seed_count':len(old)}


def _pool(context, kind, states, times, seeds, reference, **extra):
    states = np.asarray(states,np.float64); times = np.asarray(times,np.int64); seeds=np.asarray(seeds,np.int64)
    require(states.shape==(len(times),13) and seeds.shape==times.shape and np.isfinite(states).all(),
            'invalid completed state pool')
    observations=reference.observation(states,times)
    require(observations.shape==(len(times),14) and observations.dtype==np.float32
            and np.isfinite(observations).all(), 'invalid observations')
    h=hashlib.sha256()
    for a in (states,times,seeds,observations): h.update(np.ascontiguousarray(a).tobytes())
    return {'context':context,'source':kind,'states':states,'times':times,'seeds':seeds,
            'observations':observations,'state_pool_sha256':h.hexdigest(),**extra}


def random_pool(core, reference, evaluation, context, ledger, deadline=None, progress=None):
    """16 common random-prefix states/context, exactly 600 charged steps."""
    seeds=random_seeds(context); times=np.repeat(np.asarray(TIMES,np.int64),4)
    law=core.canonical.START_LAWS['recorded' if context=='ordinary' else 'strong_current']
    requests=np.random.Generator(np.random.PCG64(PREFIX_SEEDS[context])).uniform(-1,1,(16,100,4)).astype(np.float32)
    initial=ledger['transition_steps_completed']; resets=ledger['resets_completed']; states=[]
    for i,seed in enumerate(seeds):
        env=evaluation._reset(core,reference,seed,law,ledger,deadline,progress)
        try:
            for t in range(int(times[i])):
                reference._step(env,requests[i,t],ledger,'random_prefix',deadline,progress)
            states.append(np.asarray(env.state,np.float64).copy())
        finally:
            env.close()
    require(ledger['transition_steps_completed']-initial==600 and ledger['resets_completed']-resets==16,
            'random-prefix accounting differs')
    return _pool(context,'random_prefix',states,times,seeds,reference,prefix_seed=PREFIX_SEEDS[context])


def visited_pool(core, reference, evaluation, policy, context, seed, arm, ledger, deadline=None, progress=None):
    """8 fresh frozen-actor-visited states/cell, exactly 300 charged steps."""
    seeds=visited_seeds(context,seed,arm);times=np.repeat(np.asarray(TIMES,np.int64),2)
    law=core.canonical.START_LAWS['recorded' if context=='ordinary' else 'strong_current']
    initial=ledger['transition_steps_completed'];resets=ledger['resets_completed'];states=[]
    policy.assert_unchanged()
    for i,reset_seed in enumerate(seeds):
        env=evaluation._reset(core,reference,reset_seed,law,ledger,deadline,progress)
        try:
            for t in range(int(times[i])):
                observation=reference.observation(np.asarray(env.state,np.float64)[None,:],np.asarray([t],np.int64))
                action=reference.actor_actions(policy,observation,ledger,deadline=deadline,progress=progress)[0]
                reference._step(env,action,ledger,'actor_prefix',deadline,progress)
            states.append(np.asarray(env.state,np.float64).copy())
        finally:
            env.close()
    policy.assert_unchanged()
    require(ledger['transition_steps_completed']-initial==300 and ledger['resets_completed']-resets==8,
            'actor-prefix accounting differs')
    return _pool(context,'actor_visited',states,times,seeds,reference,seed=int(seed),arm=arm)
