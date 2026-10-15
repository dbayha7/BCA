"""Read-only: print the exploratory BCA arms' absolute points and paired differences (strong starts) from
iw_explore/RESULTS.json, for the weighting-test draft (p4_draftB)."""
import json, os

R = os.path.join(os.path.dirname(__file__), '..', '..', 'iw_explore', 'RESULTS.json')
d = json.load(open(R, encoding='utf-8'))
ARMS = ['plain_gap', 'bca_unweighted', 'bca_exact_strong', 'bca_exact_strong_single', 'none', 'le_none',
        'le_bca_unweighted', 'le_bca_exact_strong', 'le_bca_exact_mild', 'le_oracle']
for law in ('strong', 'none', 'mild'):
    print('== absolute, law', law, ': points mean (lo..hi seed) | return mean')
    for q in d['qualities']:
        row = [q['quality']]
        for a in ARMS:
            e = q['arms'][a][law]
            p = e['points']
            row.append('%s=%.1f(%.1f..%.1f)|%.1f' % (a, p['mean'], min(p['per_seed']), max(p['per_seed']), e['J']['mean']))
        print('  '.join(row))
print()
for law in ('strong', 'none'):
    print('== comparisons, law', law, ': return mean (lower) [points mean (lower)]')
    for q in d['qualities']:
        print(' ', q['quality'])
        for k, c in q['comparisons'].items():
            e = c['laws'][law]
            print('    %-48s %+7.1f (%+7.1f) [%+6.2f (%+6.2f)]' % (k, e['J']['mean'], e['J']['lower_one_sided'],
                                                               e['points']['mean'], e['points']['lower_one_sided']))
print()
for q in d['qualities']:
    print(q['quality'], json.dumps(q['pre_registered'])[:900])
    print('  checks:', {k: v.get('band_penalty_shape_check_passed', v.get('failed')) for k, v in q['checks'].items()})
