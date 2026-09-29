"""Run the predeclared known-ratio synthetic check, with no RL imports."""
import argparse
import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import scipy
from scipy.stats import beta
import yaml

from reference import build_reference, density_ratios


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf8')


def interval(k, n, confidence):
    tail = (1 - confidence) / 2
    return [float(beta.ppf(tail, k, n - k + 1)) if k else 0.,
            float(beta.ppf(1 - tail, k + 1, n - k)) if k < n else 1.]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    cfg = yaml.safe_load((here / 'protocol.yaml').read_text(encoding='utf8'))
    if cfg['schema'] != 'weighted-conformal-math-v1' or cfg['ratios'] != 'known':
        raise ValueError('This runner implements only the frozen known-ratio scalar protocol')
    if cfg['training_updates'] or cfg['simulator_steps'] or cfg['positive_scale'] != 1:
        raise ValueError('This is a zero-training, zero-simulator, unit-scale reference')
    if cfg['predictor'] != 'zero':
        raise ValueError('No score fitting is implemented by this protocol')
    n, trials = cfg['calibration_size'], cfg['trials']
    p, q = cfg['source_hard_probability'], cfg['target_hard_probability']
    if not (0 < p < 1 and 0 < q < 1 and n > 0 and trials > 0):
        raise ValueError('Invalid populations or sample counts')
    known_ratios = density_ratios([1-p, p], [1-q, q])
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'status': 'frozen_before_outcomes',
        'protocol': cfg,
        'sources_sha256': {str(path.relative_to(root)): sha(path) for path in sorted(here.glob('*'))
                           if path.is_file()},
        'numpy': np.__version__, 'scipy': scipy.__version__, 'python': platform.python_version(),
        'rng': 'NumPy PCG64(SeedSequence([seed, stream, trial]))',
        'streams': {'0': 'calibration X and Y', '1': 'source query X and Y',
                    '2': 'target query X and Y', '3': 'exponential bootstrap draws'},
        'known_ratios_easy_hard': known_ratios.tolist(),
        'training_updates': 0, 'simulator_steps': 0, 'models_loaded': 0,
        'real_rl_coverage_verified': False,
    }
    write(args.output / 'manifest.json', manifest)
    test = subprocess.run([sys.executable, str(here / 'test_reference.py')],
                          text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          timeout=120)
    (args.output / 'tests.txt').write_text(test.stdout, encoding='utf8')
    write(args.output / 'test_exit.json', {'actual_exit': test.returncode})
    if test.returncode:
        raise RuntimeError('Reference checks failed; no synthetic experiment executed')
    names = ('ordinary_source', 'ordinary_shifted', 'weighted_conformal_shifted', 'weighted_bca_shifted')
    observations = {name: {'covered': [], 'radius': []} for name in names}
    banks = []
    for trial in range(trials):
        streams = [np.random.Generator(np.random.PCG64(np.random.SeedSequence([cfg['seed'], s, trial])))
                   for s in range(4)]
        hard = streams[0].random(n) < p
        scores = streams[0].random(n) * np.where(hard, cfg['hard_error_max'], cfg['easy_error_max'])
        query_hard = [bool(streams[s].random() < prob) for s, prob in ((1, p), (2, q))]
        responses = [float(streams[s].random() * (cfg['hard_error_max'] if h else cfg['easy_error_max']))
                     for s, h in zip((1, 2), query_hard)]
        draws = streams[3].exponential(size=(cfg['bootstrap_draws'], n))
        common = dict(alpha=cfg['alpha'], credibility=cfg['credibility'])
        plain = build_reference(scores, np.ones(n), draws, **common).radii([1])
        weighted = build_reference(scores, known_ratios[hard.astype(int)], draws,
                                   **common).radii([known_ratios[int(query_hard[1])]])
        radii = (plain.conformal[0], plain.conformal[0], weighted.conformal[0], weighted.full[0])
        ys = (responses[0], responses[1], responses[1], responses[1])
        for name, radius, y in zip(names, radii, ys):
            observations[name]['covered'].append(bool(y <= radius))
            observations[name]['radius'].append(None if np.isinf(radius) else float(radius))
        banks.append({'trial': trial, 'calibration_hard': hard.astype(int).tolist(),
                      'calibration_scores': scores.tolist(), 'query_hard_source_target': query_hard,
                      'query_response_source_target': responses,
                      'radii': [None if np.isinf(r) else float(r) for r in radii],
                      'weighted_query_mass': float(weighted.query_mass[0])})
    summaries = {}
    for name, values in observations.items():
        k = sum(values['covered'])
        finite = [v for v in values['radius'] if v is not None]
        summaries[name] = {'covered': k, 'trials': trials, 'coverage': k/trials,
                           'coverage_binomial_interval': interval(k, trials, cfg['interval_confidence']),
                           'infinite_count': trials-len(finite),
                           'mean_finite_radius': float(np.mean(finite)) if finite else None}
    result = {'status': 'completed_synthetic_only', 'nominal_coverage': 1-cfg['alpha'],
              'arms': summaries, 'test_exit': test.returncode,
              'training_updates': 0, 'simulator_steps': 0, 'models_loaded': 0,
              'real_rl_coverage_verified': False, 'host_training_integration_complete': False,
              'manifest_sha256': sha(args.output / 'manifest.json'),
              'scope': 'Known ratios, independent scalar examples, fixed score, unchanged Y|X; not RL evidence.'}
    payload = json.dumps(banks, separators=(',', ':'), allow_nan=False).encode('utf8')
    (args.output / 'synthetic_panels.json.gz').write_bytes(gzip.compress(payload, mtime=0))
    result['panels_sha256'] = sha(args.output / 'synthetic_panels.json.gz')
    result['uncompressed_panels_sha256'] = hashlib.sha256(payload).hexdigest()
    write(args.output / 'results.json', result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
