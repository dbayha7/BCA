"""Label-blind fixed-selector arithmetic; standard library only."""
import math

KAPPAS = (0, .03, .1, .3, 1, 3, 10)
FAMILIES = ('bca', 'contrast_bca', 'contrast_disagreement', 'contrast_actor_distance')


def finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Expected a finite numeric score')
    return float(value)


def rms(values):
    values = [finite(x) for x in values]
    if not values:
        raise ValueError('Empty score vector')
    maximum = max(abs(x) for x in values)
    return 0.0 if maximum == 0 else maximum * math.sqrt(math.fsum((x / maximum) ** 2 for x in values) / len(values))


def selection(scores, atol, rtol):
    scores = [finite(x) for x in scores]
    atol, rtol = finite(atol), finite(rtol)
    if not scores or min(atol, rtol) < 0:
        raise ValueError('Nonempty scores and nonnegative tie tolerance required')
    best = max(scores)
    close = [index for index, value in enumerate(scores) if abs(value - best) <= atol + rtol * abs(best)]
    ordered = sorted(scores)
    chosen = close[0]
    return {'index': chosen, 'exact_argmax_index': scores.index(best), 'near_tie': len(close) > 1,
            'inconclusive_selection': len(close) > 1, 'near_best_indices': close,
            'maximum_score': best, 'selected_score': scores[chosen],
            'top_two_gap': None if len(scores) < 2 else ordered[-1] - ordered[-2]}


def make_methods(queries, protocol):
    if tuple(protocol['kappa_grid']) != KAPPAS or len(queries) != protocol['query_count'] or not queries:
        raise ValueError('Fixed grid and declared query count required')
    factors = {key: finite(protocol['contrast_factors'][key]) for key in ('bca', 'disagreement', 'actor_distance')}
    if min(factors.values()) <= 0:
        raise ValueError('Unavailable contrast factors must not receive a fallback')
    validated = []
    for query in queries:
        if query.get('width_usable') is not True or query.get('width_valid') is not True:
            raise ValueError('Invalid calibration blocks selector comparison')
        q = [[finite(x) for x in row] for row in query['q']]
        n = len(q)
        if not 1 <= n <= 9 or any(len(row) != 2 for row in q) or len(query['actions']) != n:
            raise ValueError('Aligned candidate bank and twin Q values required')
        signals = {key: [finite(x) for x in query[key]] for key in ('u', 'disagreement', 'actor_distance')}
        if any(len(values) != n or min(values) < 0 for values in signals.values()):
            raise ValueError('Aligned nonnegative penalty signals required')
        if signals['actor_distance'][0] != 0:
            raise ValueError('Candidate zero must be the frozen actor')
        validated.append((q, signals))

    methods = {}
    def add(name, family, kappa=None):
        signal = {'bca': 'u', 'contrast_bca': 'u', 'contrast_disagreement': 'disagreement',
                  'contrast_actor_distance': 'actor_distance'}.get(family)
        factor = {'contrast_bca': factors['bca'], 'contrast_disagreement': factors['disagreement'],
                  'contrast_actor_distance': factors['actor_distance']}.get(family, 1.0)
        degenerate = signal is not None and all(all(value == 0 for value in signals[signal]) for _, signals in validated)
        entry = {'family': family, 'kappa_penalty': kappa, 'available': True, 'unavailable_reason': None,
                 'rms_match_factor': None, 'penalty_factor': factor,
                 'factor_kind': 'frozen_fitting_candidate_contrast' if family.startswith('contrast_') else 'unscaled',
                 'penalty_signal_degenerate': degenerate,
                 'actual_u_scale_zero': all(all(value == 0 for value in signals['u']) for _, signals in validated),
                 'usefulness_claim_allowed': False, 'selected_indices': [], 'selection_details': [],
                 'scores': [], 'penalty_rms': [], 'penalty_rms_to_q1_rms': []}
        for q, signals in validated:
            penalty = [0.0] * len(q)
            if family == 'twin_min':
                scores = [min(row) for row in q]
            elif family == 'actor':
                scores = [-value for value in signals['actor_distance']]
            else:
                if signal is not None:
                    penalty = [finite(kappa * factor * value) for value in signals[signal]]
                scores = [finite(row[0] - value) for row, value in zip(q, penalty)]
            choice = selection(scores, protocol['tie_atol'], protocol['tie_rtol'])
            if family == 'actor':
                choice = {'index': 0, 'exact_argmax_index': 0, 'near_tie': False,
                          'inconclusive_selection': False, 'near_best_indices': [0], 'maximum_score': 0.0,
                          'selected_score': 0.0, 'top_two_gap': None,
                          'rule': 'fixed actor index; no score-based selection'}
            p, qrms = rms(penalty), rms([row[0] for row in q])
            entry['selected_indices'].append(choice['index'])
            entry['selection_details'].append(choice)
            entry['scores'].append(scores)
            entry['penalty_rms'].append(p)
            entry['penalty_rms_to_q1_rms'].append(None if qrms == 0 else p / qrms)
        methods[name] = entry
    for name in ('raw_q', 'actor', 'twin_min'):
        add(name, name)
    for family in FAMILIES:
        for kappa in KAPPAS:
            add(family + '_kappa_' + format(kappa, 'g'), family, kappa)
    return methods
