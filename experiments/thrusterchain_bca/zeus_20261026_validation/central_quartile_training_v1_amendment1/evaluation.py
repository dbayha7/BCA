"""Fixed, inference-only learning-curve evaluation. No work occurs on import.

The orchestrator supplies the authenticated simulator, the pinned passed-gate
reference adapter, an immutable policy, and a prospective resource ledger.
No fitting, calibration, model loading, source-data decoding or CLI lives here.
All values are finite-horizon Q^pi for the evaluated policy, never Q-star.
"""
import hashlib
import time

import numpy as np

GAMMA = .98
HORIZON = 100
Q_ANCHORS = 32
FREE_EPISODES = 32
FIXED_A0 = (-.5, -.25, 0., .25, .5)
VISITED_TIMES = (0, 25, 50, 75)
ERROR_TOLERANCE = .1
NUMERICAL_TIE = 1e-6
CONTEXTS = ('ordinary', 'strong_current')
CONDITIONS = ('full', 'hole', 'random')
ARMS = ('native', 'bca', 'distance')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def transition_budget():
    """Three seeds x three datasets; common 0/10k, then three 15/20k arms."""
    untrained, warmup, branch_each_checkpoint = 9, 9, 27
    trained = warmup + 2 * branch_each_checkpoint
    snapshots = untrained + trained
    prefixes = 2 * int(np.tile([0, 25, 50, 75], 8).sum())
    q_values = trained * 2 * Q_ANCHORS * 6
    continuations = trained * 2 * 6 * int(np.tile([100, 75, 50, 25], 8).sum())
    free_episodes = snapshots * 2 * FREE_EPISODES
    return {'untrained_snapshots': untrained, 'trained_snapshots': trained,
            'total_snapshots': snapshots, 'contexts': 2,
            'q_values': q_values, 'q_continuation_steps': continuations,
            'free_episodes': free_episodes, 'free_steps': free_episodes * HORIZON,
            'prefix_steps': prefixes, 'parity_steps': 104,
            'total_steps': continuations + free_episodes * HORIZON + prefixes + 104,
            'bank_resets': 128, 'parity_resets': 1,
            'actor_rows': snapshots * 2 * FREE_EPISODES * HORIZON + trained * 2 *
                          (Q_ANCHORS + 12000 - Q_ANCHORS * 6),
            'distance_anchor_actor_rows': 3 * 3 * 2 * 2 * Q_ANCHORS,
            'actor_rows_total_including_distance': 1953792,
            'visited_rows': trained * 2 * FREE_EPISODES * len(VISITED_TIMES),
            'critic_rows': q_values + trained * 2 * FREE_EPISODES * len(VISITED_TIMES),
            'width_rows_max': q_values + trained * 2 * FREE_EPISODES * len(VISITED_TIMES),
            'updates': 0}


def fixed_candidate_bank(actor_actions):
    actor = np.asarray(actor_actions)
    require(actor.shape == (Q_ANCHORS, 4) and actor.dtype == np.float32
            and np.all(np.isfinite(actor)) and np.all(np.abs(actor) <= 1),
            'Expected 32 deployed actor commands')
    bank = np.zeros((Q_ANCHORS, 6, 4), np.float32)
    bank[:, 0] = actor
    bank[:, 1:, 0] = np.asarray(FIXED_A0, np.float32)
    return bank


def hole_mask(observations, actions):
    obs, act = np.asarray(observations, np.float32), np.asarray(actions, np.float32)
    require(obs.shape[:-1] == act.shape[:-1] and obs.shape[-1] == 14 and act.shape[-1] == 4,
            'Hole mask shape mismatch')
    high = obs[..., 10].astype(np.float64) + obs[..., 11].astype(np.float64) >= .4
    return high, high & (np.abs(act[..., 0].astype(np.float64)) <= .25)


def error_metrics(scores, values, tolerance=ERROR_TOLERANCE):
    score, value = np.asarray(scores, np.float64), np.asarray(values, np.float64)
    require(score.shape == value.shape and np.all(np.isfinite(score)) and np.all(np.isfinite(value)),
            'Error metrics require aligned finite arrays')
    error = (score - value).ravel()
    over, under = error > tolerance, error < -tolerance
    mean = lambda x: float(np.mean(x)) if len(x) else None
    return {'rows': len(error), 'over': int(over.sum()), 'under': int(under.sum()),
            'near_zero': int((~(over | under)).sum()), 'tolerance': float(tolerance),
            'mae': mean(np.abs(error)), 'rmse': float(np.sqrt(np.mean(error**2))) if len(error) else None,
            'mean_signed_error': mean(error),
            'mean_over_magnitude': mean(error[over]), 'mean_under_magnitude': mean(-error[under]),
            'max_abs_error': float(np.max(np.abs(error))) if len(error) else None}


def paired_error_change(before, after, values, tolerance=ERROR_TOLERANCE):
    """Same critic before/after score adjustment against the SAME Q^pi labels."""
    before, after, values = map(lambda v: np.asarray(v, np.float64), (before, after, values))
    require(before.shape == after.shape == values.shape and
            all(np.all(np.isfinite(x)) for x in (before, after, values)), 'Paired error shape/finite mismatch')
    eb, ea = before - values, after - values
    improvement = np.abs(eb) - np.abs(ea)
    cb = np.where(eb > tolerance, 1, np.where(eb < -tolerance, -1, 0))
    ca = np.where(ea > tolerance, 1, np.where(ea < -tolerance, -1, 0))
    labels = {-1: 'under', 0: 'near_zero', 1: 'over'}
    return {'rows': int(improvement.size),
            'mean_absolute_error_improvement': float(improvement.mean()) if improvement.size else None,
            'improved': int((improvement > tolerance).sum()),
            'worsened': int((improvement < -tolerance).sum()),
            'near_tied': int((np.abs(improvement) <= tolerance).sum()),
            'change_tolerance': tolerance,
            'class_transitions': {labels[i] + '_to_' + labels[j]: int(((cb == i) & (ca == j)).sum())
                                  for i in (-1, 0, 1) for j in (-1, 0, 1)},
            'per_row_absolute_error_improvement': improvement.tolist()}


def decision_metrics(scores, values, tolerance=ERROR_TOLERANCE):
    """One forced first-action choice; this is not a trained policy rollout."""
    score, value = np.asarray(scores, np.float64), np.asarray(values, np.float64)
    require(score.shape == value.shape == (Q_ANCHORS, 6)
            and np.all(np.isfinite(score)) and np.all(np.isfinite(value)), 'Decision bank mismatch')
    selected = np.argmax(score.max(axis=1, keepdims=True) - score <= NUMERICAL_TIE, axis=1)
    chosen = value[np.arange(Q_ANCHORS), selected]
    gain = chosen - value[:, 0]
    regret = value.max(axis=1) - chosen
    return {'states': Q_ANCHORS, 'selected_slot': selected.tolist(),
            'mean_gain_over_own_actor': float(gain.mean()), 'mean_regret': float(regret.mean()),
            'improved_over_actor': int((gain > tolerance).sum()),
            'harmed_vs_actor': int((gain < -tolerance).sum()),
            'near_tied_vs_actor': int((np.abs(gain) <= tolerance).sum()),
            'regret_above_tolerance': int((regret > tolerance).sum()),
            'gain_per_state': gain.tolist(), 'regret_per_state': regret.tolist(),
            'scope': 'Forced single action followed by own actor; candidates include own actor slot0'}


def signed_candidate_margins(raw, values, adjusted=None, tolerance=ERROR_TOLERANCE):
    """Signed center/flank effects; the hole is allowed to contain good actions.

    All five fixed vectors are common. Slot0 (own actor) is deliberately excluded.
    The center is slot3, retained comparison flanks are slots1 and5. Numerical
    best-action ties are retained as ties rather than forced into a good/bad claim.
    """
    raw, values = np.asarray(raw, np.float64), np.asarray(values, np.float64)
    require(raw.ndim==2 and raw.shape[-1]==6 and raw.shape == values.shape and np.isfinite(raw).all()
            and np.isfinite(values).all(), 'Signed candidate margin shape/finite mismatch')
    n=len(raw)
    def margins(x): return x[:,3] - np.maximum(x[:,1], x[:,5])
    true = margins(values)
    inside_best = values[:,2:5].max(axis=1)
    outside_best = np.maximum(values[:,1], values[:,5])
    gap = inside_best-outside_best
    result = {'states':n, 'center_minus_best_retained_qpi':true.tolist(),
        'mean_center_minus_best_retained_qpi':float(true.mean()) if n else None,
        'center_better_than_retained':int((true>tolerance).sum()),
        'center_worse_than_retained':int((true < -tolerance).sum()),
        'center_near_tied':int((np.abs(true)<=tolerance).sum()),
        'best_fixed_action_strictly_in_band':int((gap>NUMERICAL_TIE).sum()),
        'best_fixed_action_strictly_outside_band':int((gap < -NUMERICAL_TIE).sum()),
        'best_fixed_action_tied_across_band':int((np.abs(gap)<=NUMERICAL_TIE).sum()),
        'fraction_best_fixed_action_strictly_in_band':float(np.mean(gap>NUMERICAL_TIE)) if n else None,
        'raw_center_minus_best_retained':margins(raw).tolist(),
        'raw_margin_error':error_metrics(margins(raw),true),
        'scope':'Signed advantage under own continuation policy; no assumption that a withheld action is inferior'}
    if adjusted is not None:
        adjusted=np.asarray(adjusted,np.float64)
        require(adjusted.shape==raw.shape and np.isfinite(adjusted).all(), 'Adjusted margin invalid')
        result.update(adjusted_center_minus_best_retained=margins(adjusted).tolist(),
            adjusted_margin_error=error_metrics(margins(adjusted),true),
            same_critic_margin_change=paired_error_change(margins(raw),margins(adjusted),true))
        b=np.argmax(raw.max(axis=1,keepdims=True)-raw<=NUMERICAL_TIE,axis=1)
        a=np.argmax(adjusted.max(axis=1,keepdims=True)-adjusted<=NUMERICAL_TIE,axis=1); indices=np.arange(n)
        gain=values[indices,a]-values[indices,b]
        result['adjustment_choice_change']={'changed_states':int((a!=b).sum()),
            'improved_states':int((gain>tolerance).sum()), 'harmed_states':int((gain < -tolerance).sum()),
            'near_tied_states':int((np.abs(gain)<=tolerance).sum()),
            'mean_qpi_gain_over_raw_choice':float(gain.mean()) if n else None, 'gain_per_state':gain.tolist()}
    return result


def q_panels(observations, bank, raw, values, adjusted=None, twin_min=None):
    """Fixed actions support same-(s,a) cross-model comparisons; slot0 is separate."""
    obs, bank, raw, values = map(np.asarray, (observations, bank, raw, values))
    require(obs.shape == (32, 14) and bank.shape == (32, 6, 4)
            and raw.shape == values.shape == (32, 6), 'Panel shape mismatch')
    high, hole = hole_mask(np.repeat(obs[:, None], 6, axis=1), bank)
    fixed = np.zeros((32, 6), bool); fixed[:, 1:] = True
    fixed_band = np.zeros((32, 6), bool); fixed_band[:, 2:5] = True
    own = np.zeros((32, 6), bool); own[:, 0] = True
    masks = {'all_fixed': fixed, 'high_band_fixed_primary': fixed_band & high,
             'low_band_fixed_control': fixed_band & ~high,
             'high_flanks_fixed': fixed & ~fixed_band & high,
             'low_flanks_fixed': fixed & ~fixed_band & ~high,
             'own_actor': own, 'own_actor_high_hole': own & hole}
    panels = {name: {'raw': error_metrics(raw[m], values[m]), 'state_count': int(np.any(m, axis=1).sum())}
              for name, m in masks.items()}
    if twin_min is not None:
        twin_min = np.asarray(twin_min, np.float64)
        require(twin_min.shape == raw.shape and np.all(np.isfinite(twin_min)), 'Invalid twin-min scores')
        for name, m in masks.items():
            panels[name]['twin_min'] = error_metrics(twin_min[m], values[m])
    if adjusted is not None:
        adjusted = np.asarray(adjusted, np.float64)
        require(adjusted.shape == raw.shape and np.all(np.isfinite(adjusted)), 'Invalid adjusted scores')
        for name, m in masks.items():
            panels[name]['adjusted'] = error_metrics(adjusted[m], values[m])
            panels[name]['same_critic_change'] = paired_error_change(raw[m], adjusted[m], values[m])
    return {'panels': panels, 'raw_decisions': decision_metrics(raw, values),
            'adjusted_decisions': None if adjusted is None else decision_metrics(adjusted, values),
            'signed_candidate_margins':{
                name:signed_candidate_margins(raw[m],values[m],None if adjusted is None else adjusted[m])
                for name,m in {'all':np.ones(32,bool),'high_load_hole':high[:,0],
                               'low_load_retained_band':~high[:,0]}.items()},
            'independent_unit': 'Training seed; states/episodes nested, six candidates not independent',
            'reference': 'Each model own deterministic actor continuation; Qpi, no Qstar or common-policy claim',
            'band_boundary': 'abs(a0)<=.25 inclusive; fixed slots2,3,4, not strictly interior',
            'per_probe': {'q1': raw.tolist(), 'qpi': values.tolist(),
                          'twin_min': None if twin_min is None else twin_min.tolist(),
                          'adjusted': None if adjusted is None else adjusted.tolist(),
                          'signed_error': (raw-values).tolist(),
                          'high_load': high[:, 0].tolist(), 'actions': bank.tolist()}}


def paired_seed_interaction(rows, *, seeds, context, checkpoint, metric):
    """Positive means BCA reduces the specified error more under hole than random.

    Rows must be per-policy/per-seed aggregates on the same bank. Full-data
    effects are reported separately; three seeds are three independent units.
    No test or confidence interval is manufactured from the 54 branch policies.
    """
    require(len(seeds) == len(set(seeds)) == 3, 'Exactly three independent training seeds required')
    selected = [x for x in rows if x['context'] == context and x['checkpoint'] == checkpoint]
    table = {}
    bank_id = set()
    for row in selected:
        key = (row['seed'], row['condition'], row['arm'])
        require(key not in table, 'Duplicate policy aggregate')
        table[key] = row
        bank_id.add(row['bank_id'])
    require(len(bank_id) == 1, 'Cross-policy comparison requires identical fixed state/action bank')
    results = []
    for seed in seeds:
        effects = {}
        for condition in CONDITIONS:
            native = table.get((seed, condition, 'native'))
            bca = table.get((seed, condition, 'bca'))
            require(native is not None and bca is not None, 'Incomplete paired seed/condition coverage')
            n, b = float(native[metric]), float(bca[metric])
            require(np.isfinite(n) and np.isfinite(b), 'Nonfinite paired metric')
            effects[condition] = n - b
        results.append({'seed': seed, 'native_minus_bca_error': effects,
                        'hole_minus_random_interaction': effects['hole'] - effects['random']})
    interaction = [x['hole_minus_random_interaction'] for x in results]
    return {'seed_count': 3, 'per_seed': results, 'mean_interaction': float(np.mean(interaction)),
            'min_interaction': float(min(interaction)), 'max_interaction': float(max(interaction)),
            'all_three_strictly_positive': all(v > 0. for v in interaction),
            'accuracy_screen_is_separate': 'BCA adjusted versus native Q1 MAE improvement >=1 in each seed; this helper tests only the >0 hole-minus-random interaction',
            'uncertainty': 'Only three independent training seeds; range is descriptive, not a confidence interval',
            'estimand': 'Separate trained-model own-policy Q-error contrast; not common-policy Q difference'}


def _deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError('Evaluation deadline reached before dispatch')


def _reset(core, reference, seed, law, ledger, deadline, progress):
    _deadline(deadline)
    require(ledger.get('resets_attempted', 0) < ledger['max_resets'], 'Evaluation reset cap exceeded')
    env = reference._env(core)
    try:
        ledger['resets_attempted'] += 1
        reference._notify(progress, ledger)
        obs, info = env.reset(seed=int(seed), options={'start': dict(law)})
        ledger['resets_completed'] += 1
        reference._check_env(env, obs, False, False, info, 0)
        return env
    except BaseException:
        env.close()
        raise


def _seeds(values, n, label):
    values = np.asarray(values)
    require(values.shape == (n,) and values.dtype.kind in 'iu' and len(np.unique(values)) == n,
            'Invalid ' + label + ' seeds')
    return values


def build_context_bank(core, reference, *, context, q_seeds, free_seeds, prefix_seed,
                       start_law, ledger, deadline=None, progress=None):
    """Generate once from frozen seeds/laws, before any policy or outcome query.

    The caller must collision-audit seeds and persist arrays/digests. Current is
    an observed reset shift; the simulator dynamics and observation stay fixed.
    This performs 64 resets and exactly1200 random-prefix transitions/context.
    """
    require(context in CONTEXTS, 'Unknown evaluation context')
    q_seeds, free_seeds = _seeds(q_seeds, 32, 'Q'), _seeds(free_seeds, 32, 'free')
    require(not set(q_seeds.tolist()) & set(free_seeds.tolist()), 'Q/free reset seeds must be disjoint')
    require(set(start_law) == {'current', 'wave', 'kick'}, 'Explicit complete reset law required')
    times = np.tile(np.asarray([0, 25, 50, 75], np.int64), 8)
    requests = np.random.Generator(np.random.PCG64(int(prefix_seed))).uniform(-1, 1, (32, 100, 4)).astype(np.float32)
    states, starts = [], []
    for i, seed in enumerate(q_seeds):
        env = _reset(core, reference, seed, start_law, ledger, deadline, progress)
        try:
            for t in range(int(times[i])):
                reference._step(env, requests[i, t], ledger, 'prefix', deadline, progress)
            states.append(env.state)
        finally:
            env.close()
    for seed in free_seeds:
        env = _reset(core, reference, seed, start_law, ledger, deadline, progress)
        try:
            starts.append(env.state)
        finally:
            env.close()
    states, starts = np.asarray(states, np.float64), np.asarray(starts, np.float64)
    digest = hashlib.sha256()
    for v in (states, times, starts, q_seeds.astype('<i8'), free_seeds.astype('<i8')):
        digest.update(np.ascontiguousarray(v).tobytes())
    return {'context': context, 'bank_id': digest.hexdigest(), 'states': states, 'times': times,
            'observations': reference.observation(states, times), 'free_starts': starts,
            'q_seeds': q_seeds, 'free_seeds': free_seeds, 'prefix_seed': int(prefix_seed),
            'start_law': dict(start_law), 'fixed_a0': FIXED_A0,
            'freshness': 'New collision-audited evaluation seeds required; no evaluation-driven tuning'}


def _free_rollouts(core, reference, policy, starts, ledger, deadline, progress):
    current = np.asarray(starts, np.float64).copy()
    require(current.shape == (32, 13) and np.all(np.isfinite(current)), 'Invalid free-policy starts')
    returns = np.zeros(32, np.float64)
    undiscounted = np.zeros(32, np.float64)
    high_counts, hole_counts = np.zeros(32, np.int64), np.zeros(32, np.int64)
    rewards=np.empty((32,100),np.float64)
    observations=np.empty((32,4,14),np.float32)
    actions=np.empty((32,4,4),np.float32)
    states=np.empty((32,4,13),np.float64)
    for t in range(100):
        obs = reference.observation(current, np.full(32, t, np.int64))
        action = reference.actor_actions(policy, obs, ledger, deadline=deadline, progress=progress)
        high, hole = hole_mask(obs, action)
        high_counts += high; hole_counts += hole
        if t in VISITED_TIMES:
            slot=VISITED_TIMES.index(t)
            observations[:,slot]=obs; actions[:,slot]=action; states[:,slot]=current
        current, reward = reference._dynamics(core, current, action, ledger, 'free_policy', deadline, progress)
        rewards[:,t]=reward
        returns += GAMMA ** t * reward
        undiscounted += reward
    return {'episodes': 32, 'returns': returns.tolist(), 'mean_return': float(returns.mean()),
            'undiscounted_returns': undiscounted.tolist(),
            'mean_undiscounted_return': float(undiscounted.mean()),
            'episode_return_std': float(returns.std(ddof=1)), 'high_steps': int(high_counts.sum()),
            'hole_steps': int(hole_counts.sum()), 'hole_steps_per_episode': hole_counts.tolist(),
            'high_steps_per_episode': high_counts.tolist(),
            'episodes_entering_hole': int((hole_counts > 0).sum()),
            'high_step_hole_fraction': float(hole_counts.sum()/high_counts.sum()) if high_counts.sum() else None,
            'cached_visited':{'times':list(VISITED_TIMES),'observations':observations.tolist(),
                'states_float64':states.tolist(),'actions':actions.tolist(),'rewards':rewards.tolist()},
            'reference': 'Free deployed actor, discounted100-step return, no forced action or terminal bootstrap'}


def queried_q(policy, observations, actions, ledger, reference, deadline, progress):
    _deadline(deadline)
    n=len(observations)
    attempted=ledger.get('critic_rows_attempted',0)+n
    require(attempted<=ledger['max_critic_rows'],'Evaluation critic row cap exceeded')
    ledger['critic_rows_attempted']=attempted
    reference._notify(progress,ledger)
    result=np.asarray(policy.q(observations,actions))
    ledger['critic_rows_completed']=ledger.get('critic_rows_completed',0)+n
    require(result.shape==(n,2) and np.isfinite(result).all(),'Expected finite native twin-Q')
    return result


def visited_policy_panel(policy, free, ledger, reference, adjusted_score, deadline, progress):
    """Join support-mask exposure and prediction errors on the SAME visited rows.

    Discounted suffixes use cached trajectory rewards; no simulator or actor call.
    The known construction mask is not a joint-density or nearest-support estimate.
    """
    cache=free['cached_visited']
    require(cache['times']==list(VISITED_TIMES),'Visited time contract differs')
    obs=np.asarray(cache['observations'],np.float32).reshape(128,14)
    action=np.asarray(cache['actions'],np.float32).reshape(128,4)
    rewards=np.asarray(cache['rewards'],np.float64)
    require(rewards.shape==(32,100) and np.isfinite(rewards).all(),'Visited rewards invalid')
    suffix=np.zeros(32,np.float64); values=np.empty((32,4),np.float64)
    for t in range(99,-1,-1):
        suffix=rewards[:,t]+GAMMA*suffix
        if t in VISITED_TIMES: values[:,VISITED_TIMES.index(t)]=suffix
    values=values.ravel()
    q=queried_q(policy,obs,action,ledger,reference,deadline,progress)
    adjusted=None if adjusted_score is None else np.asarray(adjusted_score(obs,action,q[:,0]),np.float64)
    if adjusted is not None: require(adjusted.shape==(128,) and np.isfinite(adjusted).all(),'Visited adjusted scores invalid')
    high,hole=hole_mask(obs,action)
    masks={'all':np.ones(128,bool),'high_hole':hole,'high_nonhole':high & ~hole,'low_band':~high & (np.abs(action[:,0])<=.25),
           'low_nonband':~high & (np.abs(action[:,0])>.25)}
    panels={}
    for name,m in masks.items():
        panels[name]={'rows':int(m.sum()),'episode_families':int(np.unique(np.repeat(np.arange(32),4)[m]).size),
            'raw':error_metrics(q[m,0],values[m]),'twin_min':error_metrics(q[m].min(axis=1),values[m])}
        if adjusted is not None: panels[name].update(adjusted=error_metrics(adjusted[m],values[m]),
            same_critic_change=paired_error_change(q[m,0],adjusted[m],values[m]))
    return {'panels':panels,'per_row':{'episode_index':np.repeat(np.arange(32),4).tolist(),
        'time':np.tile(VISITED_TIMES,32).tolist(),'observations':obs.tolist(),'actions':action.tolist(),
        'high_load':high.tolist(),'construction_hole':hole.tolist(),'qpi_suffix':values.tolist(),
        'q1':q[:,0].tolist(),'twin_min':q.min(axis=1).tolist(),
        'adjusted':None if adjusted is None else adjusted.tolist()},
        'reference':'Exact discounted suffix of the same deterministic free trajectory, finite horizon, zero terminal bootstrap',
        'support_scope':'Known withholding mask only. State typicalness and joint four-action support at these visited rows are unmeasured.',
        'mechanism_scope':'Hole entry does not establish critic exploitation; symmetric-tail squared BC itself favors zero',
        'additional_simulator_transitions':0,'additional_actor_rows':0}


def evaluate_snapshot(core, reference, policy, bank, ledger, *, trained=True,
                      adjusted_score=None, deadline=None, progress=None):
    """Evaluate immutable snapshot. Caller authenticates sources and owns deadline.

    policy provides act(obs), q(obs,action)->(N,2), assert_unchanged(). Optional
    adjusted_score(obs,action,q1) returns frozen Q1-3U; caller verifies calibration
    readiness and charges width-row counters. Untrained snapshots run free only.
    """
    policy.assert_unchanged()
    out = {'context': bank['context'], 'bank_id': bank['bank_id'], 'trained': bool(trained),
           'free': _free_rollouts(core, reference, policy, bank['free_starts'], ledger, deadline, progress),
           'q': None}
    if trained:
        obs = bank['observations']
        own = reference.actor_actions(policy, obs, ledger, deadline=deadline, progress=progress)
        actions = fixed_candidate_bank(own)
        repeated = np.repeat(obs, 6, axis=0)
        q=queried_q(policy,repeated,actions.reshape(192,4),ledger,reference,deadline,progress)
        raw = q[:, 0].reshape(32, 6)
        adjusted = None
        if adjusted_score is not None:
            _deadline(deadline)
            adjusted = np.asarray(adjusted_score(repeated, actions.reshape(192, 4), q[:, 0])).reshape(32, 6)
            require(np.all(np.isfinite(adjusted)), 'Invalid calibrated adjusted score; no fallback')
            policy.assert_unchanged()
        values = reference.continuation_values(core, policy, bank['states'], bank['times'], actions,
                                               ledger, deadline=deadline, progress=progress)
        out['q'] = q_panels(obs, actions, raw, values, adjusted, q.min(axis=1).reshape(32, 6))
        out['visited']=visited_policy_panel(policy,out['free'],ledger,reference,adjusted_score,deadline,progress)
    policy.assert_unchanged()
    return out
