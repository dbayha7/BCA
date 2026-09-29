"""Read only the completed local collection/precommit; never dispatch science.

This is a bounded milestone review, not the full saved-transition/outcome audit.
The archive remains open to the worker; each indexed SELECT finishes immediately.
"""
from collections import Counter
from pathlib import Path
import datetime
import hashlib
import json
import math
import sqlite3
import struct
import zlib

W = Path(__file__).resolve().parent
ROOT = Path('/home/dbayha/bca-work/ood-live-v2')
A = ROOT / 'td3-hopper-s202609171-v1'
KEY = W.parent / 'monitor_20260928T230801Z/key_tables/v2_00.u32'
PLAN = 'f11ebe8f8e3ccc4af10511ed1a241e8f7c5f0aa9f84770103dd6928e59b60756'
sha = lambda b: hashlib.sha256(b).hexdigest()
canonical = lambda v: json.dumps(v, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path, limit=20_000_000):
    before = path.stat()
    assert before.st_size <= limit, str(path)
    raw = path.read_bytes()
    after = path.stat()
    assert len(raw) == before.st_size == after.st_size
    assert (before.st_dev, before.st_ino, before.st_mtime_ns) == (after.st_dev, after.st_ino, after.st_mtime_ns)
    return raw


def array(v, dtype, shape, key='hex'):
    assert set(v) == {key, 'dtype', 'shape'} and v['dtype'] == dtype and v['shape'] == shape
    raw = bytes.fromhex(v[key])
    assert len(raw) == math.prod(shape) * {'<f4': 4, '<u4': 4, '<f8': 8}[dtype]
    return raw


def main():
    started = now()
    declaration_raw = read(A / 'declaration.json')
    declaration = json.loads(declaration_raw)
    assert declaration['pair'] == ['td3_bc', 'hopper', 202609171]
    assert declaration['scientific_plan_sha256'] == PLAN
    assert declaration['horizon'] == 250 and declaration['reset_blocks'] == 64
    assert declaration['captures'] == [100, 300]
    for path, digest in declaration['sources'].items():
        assert sha(read(Path(path))) == digest, path
    whole_raw = read(A / 'whole-pair-precommit.json')
    whole = json.loads(whole_raw)
    assert whole['before_outcomes'] is True and whole['outcome_calls'] == 0
    assert whole['declaration_sha256'] == sha(declaration_raw)
    states_raw = read(A / 'states.json')
    states = json.loads(states_raw)
    assert states['reset_blocks'] == 64 and states['collectors'] == 2
    assert states['captures'] == [100, 300] and states['replacement_states'] == 0
    assert len(states['rows']) == len(whole['rows']) == len(declaration['schedule']) == 256
    keyraw = read(KEY, 512000)
    assert len(keyraw) == 512000 and sha(keyraw) == whole['keys_sha256'] == declaration['key_file_sha256']
    db = sqlite3.connect((A / 'evidence.sqlite').as_uri() + '?mode=ro', uri=True, timeout=2)
    db.execute('PRAGMA query_only=ON')
    archive_pins = {}

    def artifact(name):
        row = db.execute('SELECT id,metadata,payload,digest FROM artifacts WHERE name=?', (name,)).fetchone()
        assert row is not None, name
        index, metadata, payload, digest = row
        meta = json.loads(metadata)
        assert sha(canonical(meta)) == digest and meta['index'] == index and meta['name'] == name
        assert meta['complete'] is True and 0 < meta['bytes'] <= 2_000_000
        decoder = zlib.decompressobj()
        raw = decoder.decompress(payload, 2_000_001)
        assert decoder.eof and not decoder.unused_data and len(raw) == meta['bytes']
        assert sha(raw) == meta['raw_sha256']
        archive_pins[name] = dict(index=index, sha256=sha(raw), metadata_sha256=digest)
        return json.loads(raw), sha(raw), index

    strata = {}
    coverage = Counter()
    max_precommit_id = -1
    first_captured = None
    for i, (row, manifest, nominal) in enumerate(zip(states['rows'], whole['rows'], declaration['schedule'])):
        assert i == row['state_index'] == nominal['state_index'] == manifest['index']
        collector = 'host' if i < 128 else 'bca'
        assert nominal['collector'] == collector
        assert nominal['reset_block'] == (i % 128) // 2 and nominal['capture_step'] == (100 if i % 2 == 0 else 300)
        assert all(row[k] == v for k, v in nominal.items())
        assert nominal['reset_seed'] == declaration['schedule'][i % 128]['reset_seed']
        saved, digest, _ = artifact(f'states/state{i:03d}')
        assert digest == row['artifact_sha256'] == manifest['state_artifact_sha256']
        assert saved == {k: v for k, v in row.items() if k != 'artifact_sha256'}
        assert row['status'] == manifest['status']
        label = f'{collector}/capture{nominal["capture_step"]}'
        counts = strata.setdefault(label, dict(captured=0, missing=0))
        counts[row['status']] += 1
        if row['status'] == 'missing':
            terminal = row['terminal']
            assert row['reason'] == 'native_episode_ended_before_or_at_capture'
            assert terminal['terminated'] or terminal['truncated']
            assert 1 <= terminal['elapsed_steps'] <= nominal['capture_step']
            if i % 2 == 0:
                assert states['rows'][i + 1]['status'] == 'missing'
                assert states['rows'][i + 1]['terminal'] == terminal
            continue
        assert row['status'] == 'captured'
        if first_captured is None:
            first_captured = i
        snapshot = row['snapshot']
        assert sha(canonical(snapshot)) == row['snapshot_content_sha256']
        assert snapshot['elapsed_steps'] == nominal['capture_step']
        assert snapshot['terminated'] is False and snapshot['truncated'] is False
        bank, bankhash, bank_id = artifact(f'precommit/state{i:03d}/bank')
        scores, scorehash, score_id = artifact(f'precommit/state{i:03d}/scores')
        assert bank_id < score_id
        max_precommit_id = max(max_precommit_id, score_id)
        assert bankhash == manifest['bank_sha256'] and scorehash == manifest['scores_sha256']
        assert bank['pair'] == declaration['pair'] and bank['capture'] == nominal
        assert bank['protocol_sha256'] == PLAN and bank['pool_size'] == 8192
        assert bank['anchor_slot'] == 0 and bank['host_reference_slot'] == 12 and bank['bca_policy_slot'] == 13
        assert bank['pins']['execution_declaration'] == sha(declaration_raw)
        assert bank['pins']['state_snapshot'] == row['snapshot_content_sha256']
        assert bank['pins']['support_bank'] == declaration['support_bank_sha256']
        assert array(bank['keys'], '<u4', [250, 2]) == keyraw[i * 2000:(i + 1) * 2000]
        assert bank['state_support'] == ('near' if bank['state_distance'] <= bank['state_q95'] else 'distant')
        coverage['state_' + bank['state_support']] += 1
        present = []
        owners = []
        applied_seen = []
        missing = Counter()
        assert len(bank['slots']) == 14
        for j, slot in enumerate(bank['slots']):
            role = 'near' if j < 4 else 'moderate' if j < 8 else 'strong' if j < 12 else 'host_policy' if j == 12 else 'bca_policy'
            assert slot['slot'] == j and slot['role'] == role
            if slot['status'] == 'missing':
                assert j not in (0, 12, 13) and slot['alias_of'] is None
                assert slot['reason'] == 'fixed_pool_band_quota_unfilled'
                missing[role] += 1
                coverage['missing_' + role] += 1
                continue
            assert slot['status'] == 'present'
            present.append(j)
            sent = struct.unpack('<3f', array(slot['sent'], '<f4', [3]))
            applied = struct.unpack('<3f', array(slot['applied'], '<f4', [3]))
            assert all(math.isfinite(x) and abs(x) <= 1 for x in sent + applied)
            # Numeric equality matches the declared alias rule, including signed zero.
            owner = next((k for k, a in applied_seen if a == applied), j)
            assert slot['alias_of'] == owner
            applied_seen.append((j, applied))
            if owner == j:
                owners.append(j)
            else:
                coverage['aliased_nominal_slots'] += 1
            distance = slot['distance']
            assert math.isfinite(distance) and distance >= 0
            band = 'near' if distance <= bank['action_q95'] else 'moderate' if distance <= bank['action_q99'] else 'strong'
            assert slot['band'] == band
            if j < 12:
                assert band == role
            coverage['present_' + role] += 1
        assert bank['unique_trajectory_slots'] == owners
        assert bank['missing_support_slots'] == {role: missing[role] for role in ('near', 'moderate', 'strong')}
        assert bank['support_quotas_complete'] == (sum(missing.values()) == 0)
        coverage['complete_quota_states'] += int(bank['support_quotas_complete'])
        coverage['unique_applied_actions'] += len(owners)
        assert scores['slots'] == present and scores['host_width'] is None
        assert array(scores['constant'], '<f8', [14], '__array__') == bytes(14 * 8)
        array(scores['random'], '<f8', [14], '__array__')
    captured = sum(v['captured'] for v in strata.values())
    missing = sum(v['missing'] for v in strata.values())
    assert captured == states['captured'] and missing == states['missing'] and captured + missing == 256
    first_outcome = None
    if first_captured is not None:
        name = f'panels/host/state{first_captured:03d}/slot0/started'
        found = db.execute('SELECT id FROM artifacts WHERE name=?', (name,)).fetchone()
        if found is not None:
            first_outcome = dict(name=name, index=found[0])
            assert first_outcome['index'] > max_precommit_id
    db.close()
    assert read(A / 'states.json') == states_raw and read(A / 'whole-pair-precommit.json') == whole_raw
    result = dict(utc=now(), started=started, scope='saved collection rows and precommit integrity only',
        pair=declaration['pair'], nominal_captures=256, captured=captured, missing=missing, strata=strata,
        coverage=dict(coverage), source_hashes_checked=len(declaration['sources']),
        declaration_sha256=sha(declaration_raw), states_sha256=sha(states_raw), whole_pair_precommit_sha256=sha(whole_raw),
        continuation_key_sha256=sha(keyraw), archive_pins=archive_pins, first_outcome_started=first_outcome,
        all_reviewed_bank_score_records_precede_first_outcome=first_outcome is not None,
        execution_finished=False, scientific_outcome_acceptance=False,
        limits=['No full collection transition/native payload or ledger audit.',
                'No support-neighbor or candidate-generation reconstruction, model query, or outcome review.',
                'Source/file hashes and archive ordering do not prove power-loss durability.'],
        target_imports=False, model_queries=0, simulator_calls=0, ledger_or_lease_open=False)
    with (W / 'collection_milestone_review.json').open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != 'archive_pins'}))


if __name__ == '__main__':
    main()
