"""Independent saved-array audit; no production analysis/model/simulator imports.

Run from any directory. Writes only an exclusive independent receipt in this folder.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parent
METHODS = ('raw_q', 'live_width', 'matched_distance', 'shuffled_width', 'constant_offset')
TOL = 1e-6


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def arrays(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k].copy() for k in f.files}


def close(actual, expected):
    if expected is None:
        assert actual is None, (actual, expected)
    else:
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def main(output):
    assert not output.exists(), 'Receipt already exists; preserve it and select a new filename.'
    prepared, result = ROOT/'prepared', ROOT/'results'
    manifest = read(prepared/'preparation_manifest.json')
    status, summary = read(result/'status.json'), read(result/'paired_results.json')
    freeze, execution = read(ROOT/'freeze.json'), read(result/'execution_receipt.json')
    assert status['status'] == 'complete'
    assert sha(result/'raw_results.npz') == 'f31718e6c1261cdd784d6d009cc7494499467f5881f4ddf4931257c7667920bb'
    assert sha(result/'paired_results.json') == '6e815ac9fe1e86b08619a45812635fe9ea677abbd4c6cb77508617c398bd157b'
    for name, h in status['output_sha256'].items():
        assert sha(result/name) == h, name
    for name, h in manifest['artifact_sha256'].items():
        assert sha(prepared/name) == h, name
    for name, h in freeze['source_sha256'].items():
        assert sha(ROOT/name) == h, name
    assert freeze['source_sha256'] == manifest['source_sha256'] == execution['source_sha256'] == summary['provenance']['source_sha256']
    assert sha(prepared/'preparation_manifest.json') == freeze['preparation_manifest_sha256'] == execution['preparation_manifest_sha256'] == summary['provenance']['preparation_manifest_sha256']
    assert sha(ROOT/'freeze.json') == execution['freeze_sha256'] == summary['provenance']['freeze_sha256']
    assert sha(ROOT/'PREFLIGHT_REVIEW.md') == freeze['independent_preflight_review_sha256']
    assert sha(ROOT/'TEST_RECEIPT.json') == freeze['pure_test_receipt_sha256']
    assert sha(ROOT/'tests/test_replication.py') == freeze['test_source_sha256']
    assert status['parameters_before'] == status['parameters_after'] == manifest['parameters_before'] == manifest['parameters_after'] == execution['parameters_before'] == manifest['prior_validation']['parameters']
    for name, h in manifest['prior_validation']['file_sha256'].items():
        assert sha(ROOT.parent/'direct_q_diagnostic_20261021_v2'/name) == h, name
    assert manifest['truth_labels_used'] == manifest['plant_transitions'] == 0
    assert status['new_validation_transitions'] == status['new_training_updates'] == 0

    a, b = arrays(result/'raw_results.npz'), arrays(prepared/'inputs.npz')
    frozen_scores = arrays(prepared/'scores.npz')
    draw = arrays(prepared/'bootstrap_plan.npz')['state_indices']
    saved_boot = arrays(result/'bootstrap_statistics.npz')
    for key, value in b.items():
        np.testing.assert_array_equal(a[key], value)
    for method in METHODS:
        np.testing.assert_array_equal(a['score_'+method], frozen_scores[method])
    q, truth, width = a['q1'].astype(float), a['true_policy_q'], a['width'].astype(float)
    assert q.shape == truth.shape == width.shape == (32, 12)
    assert np.isfinite(truth).all() and np.isfinite(q).all() and np.isfinite(width).all()
    assert (width >= 0).all() and a['width_usable'].all() and not a['width_refusal'].any()
    assert a['rollout_completed'].all() and a['rollout_finite'].all() and a['rollout_evaluated'].all()
    assert not a['rollout_failure'].any()
    np.testing.assert_array_equal(a['rollout_steps'], np.broadcast_to((100-a['time'])[:, None], (32, 12)))
    assert int(a['rollout_steps'].sum()) == status['main_transitions'] == 21696
    assert status['completed_main_returns'] == status['evaluated_main_returns'] == 384
    assert status['nonfinite_evaluated_returns'] == status['failure_events_evaluated'] == 0
    assert summary['actual_transition_counts'] == {'main': 21696, 'new_validation': 0}
    assert status['valid_states'] == summary['valid_states'] == 32
    assert status['invalid_states'] == summary['invalid_states'] == 0
    assert summary['invalid_reasons'] == {}

    strata = a['context']*100 + a['time']
    assert draw.shape == (10000, 32)
    np.testing.assert_array_equal(strata[draw], np.broadcast_to(strata, draw.shape))
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([2026102123, 7])))
    expected_draw = np.empty_like(draw)
    for label in np.unique(strata):
        ids = np.flatnonzero(strata == label)
        expected_draw[:, ids] = rng.choice(ids, size=(10000, len(ids)), replace=True)
    np.testing.assert_array_equal(draw, expected_draw)

    records = summary['state_records']
    assert len(records) == 32
    error = q-truth
    predictors = {'width': width, 'distance': a['distance'], 'shuffled': a['shuffled_width']}
    rho = {key: np.full(32, np.nan) for key in predictors}
    con = {key: np.full(32, np.nan) for key in METHODS}
    selected = {key: np.argmax(frozen_scores[key], axis=1) for key in METHODS}
    selected_q = {key: truth[np.arange(32), idx] for key, idx in selected.items()}
    counts = {key: {name: 0 for name in ('changed_index', 'changed_command', 'selected_actor_command', 'actor_restoration')} for key in METHODS}
    pair_count, true_ties = [], []
    pairs = list(itertools.combinations(range(12), 2))
    for i, record in enumerate(records):
        assert record['state_id'] == i and record['valid'] and record['invalid_reasons'] == []
        for key, predictor in predictors.items():
            x, y = predictor[i], error[i]
            if np.ptp(x) and np.ptp(y):
                rho[key][i] = pearsonr(x, y).statistic
                close(record['rho_'+key], rho[key][i])
                close(record['pairwise_rho_'+key], rho[key][i])
                assert record['rho_undefined_reasons'][key] == []
            else:
                assert record['rho_'+key] is None
        comparable = [(j, k) for j, k in pairs if abs(truth[i,j]-truth[i,k]) > TOL]
        pair_count.append(len(comparable)); true_ties.append(66-len(comparable))
        base = selected['raw_q'][i]
        close(record['actor_true_q'], truth[i,6]); assert record['raw_chosen_index'] == base
        for method in METHODS:
            scores = frozen_scores[method][i]
            credits = [.5 if abs(scores[j]-scores[k]) <= TOL else float((scores[j]-scores[k])*(truth[i,j]-truth[i,k]) > 0) for j,k in comparable]
            con[method][i] = np.mean(credits) if credits else np.nan
            mr = record['methods'][method]; j = selected[method][i]
            close(mr['concordance'], con[method][i])
            assert mr['comparable_pairs'] == len(comparable) and mr['true_near_ties'] == 66-len(comparable)
            assert mr['predicted_near_ties'] == sum(abs(scores[jj]-scores[kk]) <= TOL for jj,kk in comparable)
            assert mr['chosen_index'] == j and mr['concordance_undefined_reason'] is None
            close(mr['chosen_true_q'], truth[i,j]); close(mr['delta_vs_raw'], truth[i,j]-truth[i,base]); close(mr['advantage_vs_actor'], truth[i,j]-truth[i,6])
            flags = dict(changed_index=bool(j != base), changed_command=not np.array_equal(a['actions'][i,j], a['actions'][i,base]), selected_actor_command=np.array_equal(a['actions'][i,j], a['actions'][i,6]), actor_restoration=not np.array_equal(a['actions'][i,base],a['actions'][i,6]) and np.array_equal(a['actions'][i,j],a['actions'][i,6]))
            for key, flag in flags.items():
                assert mr[key] == bool(flag); counts[method][key] += int(flag)
    assert counts == summary['counts']

    expected = {}
    def add(name, values, mask):
        expected[name] = (np.asarray(values), np.asarray(mask))
    for key, values in rho.items():
        add('primary_rho_'+key, values, np.isfinite(values))
    for control in ('distance', 'shuffled'):
        mask = np.isfinite(rho['width']) & np.isfinite(rho[control])
        assert summary['pair_common_defined_rho_states'][control] == int(mask.sum())
        add('primary_rho_width_minus_'+control, rho['width']-rho[control], mask)
        add('paired_rho_width_for_'+control, rho['width'], mask)
        add('paired_rho_'+control+'_for_width', rho[control], mask)
    for key in rho:
        assert summary['individually_defined_rho_states'][key] == int(np.isfinite(rho[key]).sum())
    conmask = np.all(np.isfinite(np.array(list(con.values()))), axis=0)
    assert summary['common_defined_concordance_states'] == int(conmask.sum())
    for method in METHODS:
        add('concordance_'+method, con[method], conmask)
        add('primary_concordance_change_'+method, con[method]-con['raw_q'], conmask)
        add('secondary_chosen_true_q_'+method, selected_q[method], np.ones(32, bool))
        add('secondary_delta_vs_raw_'+method, selected_q[method]-selected_q['raw_q'], np.ones(32, bool))
        add('secondary_advantage_vs_actor_'+method, selected_q[method]-truth[:,6], np.ones(32, bool))
    add('secondary_original_actor_true_q', truth[:,6], np.ones(32, bool))
    for control in ('matched_distance', 'shuffled_width'):
        add('secondary_selected_q_live_width_minus_'+control, selected_q['live_width']-selected_q[control], np.ones(32, bool))
    assert set(expected) == set(summary['metrics'])
    assert set(saved_boot) == {name+suffix for name in expected for suffix in ('__mean','__denominator')}
    for name, (values, mask) in expected.items():
        record = summary['metrics'][name]
        valid = mask & np.isfinite(values)
        sampled_valid = valid[draw]; denominator = np.sum(sampled_valid, axis=1)
        total = np.sum(np.where(sampled_valid, values[draw], 0), axis=1)
        bs = np.divide(total, denominator, out=np.full(10000, np.nan), where=denominator > 0)
        np.testing.assert_array_equal(saved_boot[name+'__denominator'], denominator)
        close(saved_boot[name+'__mean'], bs)
        close(record['mean'], values[valid].mean() if valid.any() else None)
        assert record['defined_states'] == int(valid.sum()) and record['planned_states'] == 32
        undefined = int((~np.isfinite(bs)).sum())
        assert record['undefined_bootstrap_draws'] == undefined
        assert record['bootstrap_defined_denominator_min'] == int(denominator.min())
        assert record['bootstrap_defined_denominator_max'] == int(denominator.max())
        close(record['ci95'], None if undefined else np.percentile(bs, [2.5,97.5], method='linear'))
        assert record['interval_status'] == ('unavailable_undefined_draws' if undefined else 'available')

    candidate_rows = list(csv.DictReader((result/'candidate_records.csv').open(newline='', encoding='utf-8')))
    assert len(candidate_rows) == 384
    for r in candidate_rows:
        i,j = int(r['state_id']),int(r['candidate_id'])
        assert r['state_valid'] == r['rollout_evaluated'] == r['rollout_completed'] == 'True'
        assert r['failure_event'] == 'False' and int(r['steps']) == 100-a['time'][i]
        for key, value in [('q1',q[i,j]),('true_policy_q',truth[i,j]),('signed_q1_error',error[i,j]),('width',width[i,j]),('distance',a['distance'][i,j]),('beta',a['beta'][i,j])]: close(float(r[key]),value)
        assert r['support_class'] == a['support_class'][i,j]
        for method in METHODS: close(float(r['score_'+method]), frozen_scores[method][i,j])
        for k in range(4): close(float(r['action_'+str(k)]),a['actions'][i,j,k])
    effects = {}
    for method in METHODS:
        d = selected_q[method]-selected_q['raw_q']
        effects[method] = dict(improved=int((d>TOL).sum()), harmed=int((d < -TOL).sum()), unchanged=int((abs(d)<=TOL).sum()), changed_indices=np.flatnonzero(selected[method] != selected['raw_q']).tolist(), delta_vs_raw=d.tolist())
    receipt = dict(status='PASS', created_utc=datetime.now(timezone.utc).isoformat(), scope='independent saved-array arithmetic only; no production analyze import or model/simulator calls',
        checks=dict(result_hashes=7, prepared_artifact_hashes=6, production_source_hashes=3, prior_file_hashes=8, unchanged_parameter_hashes=8, state_correlations=96, state_concordances=160, selected_choices=160, candidate_rows=384, aggregate_metrics=len(expected), bootstrap_draws_per_metric=10000, bootstrap_statistic_entries=10000*len(expected)),
        true_pair_counts=pair_count, true_near_tie_pairs=int(sum(true_ties)), effects=effects, support_counts={str(v):int(np.sum(a['support_class']==v)) for v in np.unique(a['support_class'])},
        primary_metrics={k:summary['metrics'][k] for k in ('primary_rho_width','primary_rho_width_minus_distance','primary_rho_width_minus_shuffled','primary_concordance_change_live_width','secondary_delta_vs_raw_live_width','secondary_selected_q_live_width_minus_matched_distance','secondary_selected_q_live_width_minus_shuffled_width')},
        file_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__), ROOT/'freeze.json', prepared/'preparation_manifest.json', result/'status.json', result/'raw_results.npz',result/'paired_results.json',result/'bootstrap_statistics.npz']}, new_model_calls=0,new_simulator_transitions=0)
    with output.open('x', encoding='utf-8') as f:
        json.dump(receipt,f,indent=2,allow_nan=False); f.write('\n')
    print(json.dumps({'status':'PASS','metrics':len(expected),'bootstrap_entries':len(expected)*10000,'true_near_ties':sum(true_ties),'effects':{k:{n:v for n,v in e.items() if n!='delta_vs_raw'} for k,e in effects.items()},'receipt_sha256':sha(output)},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt',type=Path,default=ROOT/'INDEPENDENT_POSTRUN_RECEIPT.json')
    main(parser.parse_args().receipt)
