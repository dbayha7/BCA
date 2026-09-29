"""Independently reconstruct saved synthetic arithmetic; never rerun science."""
import datetime as dt
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import beta

ROOT = Path('C:/Users/David Bayha/Documents/GitHub/BCA')
WORK = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/weighted_conformal/2026-09-29'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    if path.exists():
        raise FileExistsError('Refusing to overwrite: ' + str(path))
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf8')


def floor(scores, ratios, query, alpha):
    # Independent direct cumulative fraction oracle. It does not import the
    # implementation under test, its cache, or its binary-search algorithm.
    masses = [Fraction.from_float(float(w)) for w in ratios]
    threshold = (1 - alpha) * (sum(masses) + Fraction.from_float(float(query)))
    running = Fraction(0)
    for i in sorted(range(len(scores)), key=lambda i: scores[i]):
        running += masses[i]
        if running >= threshold:
            return float(scores[i])
    return float('inf')


before = json.loads((WORK / 'before.json').read_text(encoding='utf8'))
changed = [name for name, sha in before['protected_files'].items() if digest(Path(name)) != sha]
if changed:
    raise AssertionError({'pre_existing_files_changed': changed})
manifest = json.loads((OUT / 'manifest.json').read_text())
result = json.loads((OUT / 'results.json').read_text())
for name, sha in manifest['sources_sha256'].items():
    assert digest(ROOT / name) == sha, name
assert digest(OUT / 'manifest.json') == result['manifest_sha256']
assert digest(OUT / 'synthetic_panels.json.gz') == result['panels_sha256']
payload = gzip.decompress((OUT / 'synthetic_panels.json.gz').read_bytes())
assert hashlib.sha256(payload).hexdigest() == result['uncompressed_panels_sha256']
rows = json.loads(payload)
cfg = manifest['protocol']
names = list(result['arms'])
counts = {name: 0 for name in names}
sum_radius = {name: 0. for name in names}
infinite = {name: 0 for name in names}
for row in rows:
    i = row['trial']
    scores = np.array(row['calibration_scores'])
    ratios = np.where(row['calibration_hard'], 4., .25)
    query = 4. if row['query_hard_source_target'][1] else .25
    cal_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([cfg['seed'], 0, i])))
    original_hard = cal_rng.random(cfg['calibration_size']) < cfg['source_hard_probability']
    original_scores = cal_rng.random(cfg['calibration_size']) * np.where(original_hard, 6., 1.)
    np.testing.assert_array_equal(original_hard, row['calibration_hard'])
    np.testing.assert_array_equal(original_scores, scores)
    for stream, probability in ((1, .2), (2, .8)):
        r = np.random.Generator(np.random.PCG64(np.random.SeedSequence([cfg['seed'], stream, i])))
        hard = bool(r.random() < probability)
        y = float(r.random() * (6 if hard else 1))
        assert hard == row['query_hard_source_target'][stream-1]
        assert y == row['query_response_source_target'][stream-1]
    ordinary = floor(scores, np.ones(len(scores)), 1, Fraction(1, 10))
    weighted = floor(scores, ratios, query, Fraction(1, 10))
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([cfg['seed'], 3, i])))
    draws = rng.exponential(size=(128, len(scores)))
    # Direct products in the independent oracle; all masses are moderate here.
    order = np.argsort(scores, kind='stable')
    mass = (draws * ratios)[..., order]
    quantiles = []
    for draw in mass:
        cumulative = np.cumsum(draw)
        index = np.searchsorted(cumulative, .9 * cumulative[-1], side='left')
        quantiles.append(scores[order[index]])
    bayes = float(sorted(quantiles)[121])
    expected = [ordinary, ordinary, weighted, max(weighted, bayes)]
    saved = [float('inf') if x is None else x for x in row['radii']]
    np.testing.assert_array_equal(expected, saved)
    assert row['weighted_query_mass'] == float(Fraction.from_float(query) /
           (sum(Fraction.from_float(float(x)) for x in ratios) + Fraction.from_float(query)))
    ys = row['query_response_source_target']
    for name, radius, y in zip(names, expected, [ys[0], ys[1], ys[1], ys[1]]):
        counts[name] += y <= radius
        if np.isfinite(radius):
            sum_radius[name] += radius
        else:
            infinite[name] += 1

for name in names:
    r = result['arms'][name]
    assert int(counts[name]) == r['covered']
    assert len(rows) == r['trials']
    assert infinite[name] == r['infinite_count']
    assert abs(sum_radius[name] / (len(rows) - infinite[name]) - r['mean_finite_radius']) < 1e-12
    k, n = int(counts[name]), len(rows)
    bounds = [float(beta.ppf(.025, k, n-k+1)), float(beta.ppf(.975, k+1, n-k))]
    np.testing.assert_allclose(bounds, r['coverage_binomial_interval'], atol=1e-14, rtol=0)
assert result['real_rl_coverage_verified'] is False
assert result['host_training_integration_complete'] is False
assert result['training_updates'] == result['simulator_steps'] == result['models_loaded'] == 0
assert json.loads((OUT / 'test_exit.json').read_text())['actual_exit'] == 0

write(OUT / 'independent_verification.json', {
    'verified_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
    'independent_trial_reconstructions': len(rows), 'radius_comparisons': 4*len(rows),
    'coverage_counts_and_intervals_match': True, 'source_and_input_hashes_match': True,
    'pre_existing_protected_files_unchanged': len(before['protected_files']),
    'real_rl_coverage_verified': False, 'host_training_integration_complete': False,
    'verifier_sha256': digest(Path(__file__)),
})

plt.rcParams.update({'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(figsize=(10.8, 5.8))
values = np.array([result['arms'][n]['coverage'] for n in names]) * 100
ci = np.array([result['arms'][n]['coverage_binomial_interval'] for n in names]) * 100
colors = ['#6B7C93', '#C46555', '#147D92', '#6B54A3']
ax.bar(range(4), values, color=colors, width=.62)
ax.errorbar(range(4), values, yerr=np.array([values-ci[:, 0], ci[:, 1]-values]),
            fmt='none', color='#162A43', capsize=5, linewidth=1.6)
ax.axhline(90, color='#172D4C', linestyle='--', linewidth=1.4, label='Nominal coverage: 90%')
for x, val, upper in zip(range(4), values, ci[:, 1]):
    ax.text(x, upper+1.1, f'{val:.2f}%', ha='center', va='bottom', fontweight='bold')
ax.set_xticks(range(4), ['Ordinary\nno shift', 'Ordinary\nshifted', 'Weighted conformal\nshifted', 'Weighted + Bayesian\nshifted'])
ax.set_ylabel('Observed response coverage (%)')
ax.set_ylim(0, 105)
ax.set_yticks(range(0, 101, 20))
ax.grid(axis='y', color='#DCE2EA', alpha=.6)
ax.set_axisbelow(True)
ax.legend(loc='lower right', frameon=False)
fig.suptitle('Weighted threshold check with known density ratios', x=.09, ha='left',
             fontsize=18, fontweight='bold', y=.98)
fig.text(.09, .915, 'Synthetic only | 2,000 independent trials | 95% binomial intervals', color='#42546B')
fig.text(.09, .025, 'Fixed score, unchanged conditional response law. No RL models, training or simulator outcomes.',
         fontsize=10.5, color='#42546B')
fig.subplots_adjust(left=.09, right=.97, top=.86, bottom=.19)
fig.savefig(OUT / 'synthetic_coverage.png', dpi=170)
plt.close(fig)

table = '\n'.join(f"| {label} | {result['arms'][name]['coverage']*100:.2f}% | "
                   f"{result['arms'][name]['coverage_binomial_interval'][0]*100:.2f}–"
                   f"{result['arms'][name]['coverage_binomial_interval'][1]*100:.2f}% | "
                   f"{result['arms'][name]['mean_finite_radius']:.3f} |"
                   for name, label in zip(names, ['Ordinary, source population', 'Ordinary, shifted population',
                                                  'Weighted conformal, shifted', 'Weighted + Bayesian maximum, shifted']))
report = f'''# Weighted conformal correction: mathematical validation

This is a controlled scalar experiment, not an RL or OOD-policy result. The intended IW method now has a separately tested weighted-threshold reference. Four-host training integration remains incomplete.

![Known-ratio synthetic coverage](synthetic_coverage.png)

| Calculation | Coverage | 95% binomial interval | Mean finite radius |
|---|---:|---:|---:|
{table}

All four arms had zero infinite radii in these 2,000 trials. Separate boundary tests deliberately require infinity and verify that it is preserved. The Bayesian maximum is wider; its extra coverage is not evidence of better interval efficiency or a 95%-confidence population-risk guarantee.

## What this checks

The source contains 20% high-error examples; the target contains 80%. Conditional responses are unchanged: Uniform(0,1) for easy examples and Uniform(0,6) for hard examples. The frozen predictor is zero and scale is one. Exact target/source ratios are 0.25 and 4, used at BOTH calibration examples and the query. Each trial has 199 fresh calibration examples and independent source/target queries. Alpha stays 0.1. Both radius components remain available.

Ordinary conformal loses coverage under this constructed shift. The corrected weighted threshold recovers approximately the nominal coverage here. This tests the weighted threshold mechanism under its stated assumptions; it does not show that the old standard-BCA result was caused by this mismatch, nor that existing policy/affinity/AWR fitting weights are correct ratios.

## Verification and reproducibility

- 15 unit checks passed, including exact finite-population enumeration, finite-sample query mass, unequal weights, support refusal, ties, rescaling and interval arithmetic.
- An independent implementation reconstructed all 2,000 saved trials and all 8,000 reported radii, plus coverage counts, binomial intervals and source/input hashes.
- All {len(before['protected_files'])} pre-existing tracked repository/protected inventory files stayed byte-identical during this work.
- No model loaded, learner update or simulator step. Actual exits are recorded separately.
- `manifest.json` binds the code and predeclared YAML before outcomes. `synthetic_panels.json.gz` retains the scalar inputs and outputs; `null` radius means positive infinity. Bootstrap draws reproduce from the pinned PCG64 streams.

See [the corrected design](../../../docs/WEIGHTED_CONFORMAL_DESIGN.md) and [reproduction instructions](../../../experiments/weighted_conformal/README.md). Remaining real-data gates are the target population and valid ratio/support, independent calibration sampling, query-specific host bindings, infinity behavior, and the frozen training comparison.
'''
(OUT / 'READOUT.md').write_text(report, encoding='utf8')
print(json.dumps({'independently_reconstructed': len(rows), 'protected_unchanged': len(before['protected_files']),
                  'figure': str(OUT / 'synthetic_coverage.png')}))
