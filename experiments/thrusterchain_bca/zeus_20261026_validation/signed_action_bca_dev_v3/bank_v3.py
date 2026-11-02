"""Predeclared 15-action and Q-pi-blind state selection for frozen diagnostics.

Nothing in the pool, bank or selection accepts a Q-pi, reward, or method score.
One common selected set of physical states is used by all six checkpoints.
"""
import hashlib

import numpy as np


TIMES = (0, 25, 50, 75)
DELTA = np.float32(.2)
FIXED_A0 = (-.5, -.25, 0., .25, .5)
SLOT_NAMES = ('actor', 'a0_minus', 'a0_plus', 'a1_minus', 'a1_plus',
              'a2_minus', 'a2_plus', 'a3_minus', 'a3_plus', 'nearest_logged',
              'fixed_minus_half', 'fixed_minus_quarter', 'fixed_zero',
              'fixed_plus_quarter', 'fixed_plus_half')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def build_pool(core, reference, evaluation, *, context, seeds, prefix_seed,
               start_law, ledger, deadline=None, progress=None):
    """Create 64 new states/context: 16 at each t=0,25,50,75.

    Within every contiguous time block, slots 0..3 are natural, selected without
    critic access. Slots 4..15 are the later Q1-ambiguity selection pool. This
    charges exactly 64 resets and 2,400 prefix transitions, no learned updates.
    The caller collision-audits all random streams and saves pool identities.
    """
    require(context in ('ordinary', 'strong_current'), 'invalid context')
    seeds = np.asarray(seeds)
    require(seeds.shape == (64,) and seeds.dtype.kind in 'iu'
            and len(np.unique(seeds)) == 64 and np.all(seeds >= 0), 'invalid reset seeds')
    require(set(start_law) == {'current', 'wave', 'kick'}, 'explicit reset law required')
    times = np.repeat(np.asarray(TIMES, np.int64), 16)
    requests = np.random.Generator(np.random.PCG64(int(prefix_seed))).uniform(
        -1., 1., (64, 100, 4)).astype(np.float32)
    states = []
    before = ledger['transition_steps_completed']
    reset_before = ledger['resets_completed']
    for i, seed in enumerate(seeds):
        env = evaluation._reset(core, reference, seed, start_law, ledger, deadline, progress)
        try:
            for t in range(int(times[i])):
                reference._step(env, requests[i, t], ledger, 'prefix', deadline, progress)
            states.append(np.asarray(env.state, np.float64).copy())
        finally:
            env.close()
    states = np.asarray(states, np.float64)
    require(states.shape == (64, 13) and np.isfinite(states).all(), 'invalid pool states')
    require(ledger['transition_steps_completed'] - before == 2400
            and ledger['resets_completed'] - reset_before == 64, 'pool budget differs')
    obs = reference.observation(states, times)
    require(obs.shape == (64, 14) and obs.dtype == np.float32 and np.isfinite(obs).all(),
            'invalid pool observations')
    digest = hashlib.sha256()
    for item in (states, times, np.asarray(seeds, dtype='<i8'), obs):
        digest.update(np.ascontiguousarray(item).tobytes())
    digest.update(str(int(prefix_seed)).encode())
    return {'context': context, 'states': states, 'observations': obs, 'times': times,
            'seeds': seeds.astype(np.int64), 'prefix_seed': int(prefix_seed),
            'start_law': dict(start_law), 'pool_id': digest.hexdigest(),
            'natural_indices': np.asarray([16 * t + i for t in range(4) for i in range(4)], np.int64)}


def candidate_bank(actor_actions, neighbor_actions):
    """Return float32 [N,15,4] exact candidate bank; preserve clipped duplicates.

    The neighbor must be the first, deterministically tie-broken, time-stratified
    retained-training row. It is a comparator, not a claim of support or value.
    """
    actor, logged = np.asarray(actor_actions), np.asarray(neighbor_actions)
    require(actor.ndim == 2 and actor.shape[1] == 4 and actor.dtype == np.float32
            and logged.shape == actor.shape and logged.dtype == np.float32
            and np.isfinite(actor).all() and np.isfinite(logged).all()
            and np.all(np.abs(actor) <= 1) and np.all(np.abs(logged) <= 1),
            'actor and logged comparators must be bounded finite float32 [N,4]')
    result = np.repeat(actor[:, None, :], len(SLOT_NAMES), axis=1)
    for coordinate in range(4):
        result[:, 1 + 2*coordinate, coordinate] = np.clip(actor[:, coordinate] - DELTA, -1., 1.)
        result[:, 2 + 2*coordinate, coordinate] = np.clip(actor[:, coordinate] + DELTA, -1., 1.)
    result[:, 9] = logged
    result[:, 10:] = 0.
    result[:, 10:, 0] = np.asarray(FIXED_A0, np.float32)
    require(result.shape == (len(actor), 15, 4) and result.dtype == np.float32
            and np.isfinite(result).all() and np.all(np.abs(result) <= 1), 'invalid candidate bank')
    return result


def _unique_gap(actions, values):
    """Unique exact float32 vectors only; ties use the earliest bank slot."""
    indices = []
    for i, action in enumerate(actions):
        if not any(np.array_equal(action, actions[j]) for j in indices):
            indices.append(i)
    require(len(indices) >= 2, 'fewer than two distinct candidates')
    order = sorted(indices, key=lambda i: (-float(values[i]), i))
    best, second = float(values[order[0]]), float(values[order[1]])
    return (best - second) / (1. + abs(best)), len(actions) - len(indices)


def select_indices(times, raw_q1, banks):
    """Pick 16 natural + 16 ambiguous states once using *only* six frozen Q1s.

    Inputs: integer times [64], raw_q1 [6,64,15], float32 banks [6,64,15,4].
    For each checkpoint/state, rank *distinct complete vectors* by raw Q1 and
    compute (top-second)/(1+abs(top)). Score each state by the median of its six
    checkpoint gaps. Independently in each time block, select its first four as
    natural, then four of the other twelve with lowest score. Equal scores break
    by original pool index. Outcomes measure that conditional selected stream.
    """
    t = np.asarray(times)
    q = np.asarray(raw_q1)
    a = np.asarray(banks)
    require(t.shape == (64,) and t.dtype.kind in 'iu'
            and np.array_equal(t, np.repeat(np.asarray(TIMES), 16)), 'invalid pool times')
    require(q.shape == (6, 64, 15) and q.dtype.kind == 'f' and np.isfinite(q).all(), 'invalid Q1 scores')
    require(a.shape == (6, 64, 15, 4) and a.dtype == np.float32
            and np.isfinite(a).all() and np.all(np.abs(a) <= 1), 'invalid candidate actions')
    gaps, duplicates = np.empty((6,64), np.float64), np.empty((6,64), np.int64)
    for policy in range(6):
        for state in range(64):
            gaps[policy, state], duplicates[policy, state] = _unique_gap(a[policy, state], q[policy, state])
    median = np.median(gaps, axis=0)
    selected, strata = [], []
    for block in range(4):
        start = block*16
        natural = list(range(start, start + 4))
        # np.lexsort uses last key as primary: gap, then stable pool index.
        possible = np.arange(start+4, start+16)
        enriched = possible[np.lexsort((possible, median[possible]))[:4]].tolist()
        selected.extend(natural + enriched)
        strata.extend(['natural'] * 4 + ['ambiguous_q1'] * 4)
    indices = np.asarray(selected, np.int64)
    require(len(np.unique(indices)) == 32, 'selection overlaps or duplicates')
    return {'indices': indices, 'strata': strata, 'median_normalized_gap': median,
            'per_checkpoint_normalized_gap': gaps, 'duplicate_action_slots': duplicates,
            'rule': 'median of six distinct-action top-two raw-Q1 gaps / (1+abs(top Q1)); lowest four of the twelve candidates at each time; pool index breaks ties',
            'scope': 'Q1-only, before Q-pi reference; conditional enriched stratum'}
