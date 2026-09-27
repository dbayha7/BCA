"""Exclusive OOD stages and exact-oracle execution, with no training entrypoint.

Scientific pair preparation is supported; real simulator execution stays closed
until a separately bound trained-adapter/runtime acceptance is implemented. Toy
execution cannot promote that gate. This module never searches for checkpoints.
"""
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from functools import partial
import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from experiments.ood.adapters import require
from experiments.ood.oracle import Oracle

STANDARD_MANIFEST = '13ae3e6693d1cc71228e7b676243232c52ac6766c6a2c9b0623aaac1145f97fe'


def encode(value):
    """Preserve dtype, signed zero, inactive fields and legacy RNG tuples."""
    if isinstance(value, np.ndarray):
        require(value.dtype.kind in 'biuf', 'Unsupported array dtype.')
        return {'__array__': value.tobytes().hex(), 'dtype': value.dtype.str,
                'shape': list(value.shape)}
    if isinstance(value, np.generic):
        return encode(np.asarray(value))
    if isinstance(value, tuple):
        return {'__tuple__': [encode(x) for x in value]}
    if isinstance(value, list):
        return [encode(x) for x in value]
    if isinstance(value, dict):
        require(all(isinstance(k, str) for k in value), 'Non-string artifact key.')
        return {k: encode(v) for k, v in value.items()}
    return value


def decode(value):
    if isinstance(value, dict):
        if set(value) == {'__array__', 'dtype', 'shape'}:
            dtype = np.dtype(value['dtype'])
            require(dtype.kind in 'biuf', 'Unsupported saved array dtype.')
            return np.frombuffer(bytes.fromhex(value['__array__']), dtype=dtype).reshape(value['shape']).copy()
        if set(value) == {'__tuple__'}:
            return tuple(decode(x) for x in value['__tuple__'])
        return {k: decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode(x) for x in value]
    return value


def packed(value):
    return json.dumps(encode(value), sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def sha(value):
    return hashlib.sha256(packed(value)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_artifact(path, value):
    raw = packed(value)+b'\n'
    with Path(path).open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(raw).hexdigest()


def read_artifact(path, expected):
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, 'Changed committed artifact: '+str(path))
    return decode(json.loads(raw))


def bind_pair(readiness, audits, acceptances, processes):
    """Join already accepted audits to separate process receipts, without weights.

    Use bind_pair_files for file-hash validation. This function validates values;
    neither it nor a process receipt accepts the real OOD execution connection.
    """
    try:
        require(readiness['schema'] == 'ood-actions-first-readiness-v1', 'Unknown readiness schema.')
        require(readiness['scientific_manifest_sha256'] == STANDARD_MANIFEST, 'Stale training source manifest.')
        require(set(readiness['checkpoints']) == set(audits) == set(acceptances) == set(processes)
                == {'host', 'bca'}, 'Require both accepted methods.')
        for method, binding in readiness['checkpoints'].items():
            audit, accepted, process = audits[method], acceptances[method], processes[method]
            run = binding['run_id']
            require(audit['accepted'] is True and accepted['accepted'] is True
                    and audit['run_id'] == accepted['run_id'] == process['run_id'] == run,
                    'Unaccepted or mismatched training pair.')
            audit_exit = accepted['audit_actual_exit']
            if isinstance(audit_exit, dict):
                require(audit_exit.get('closed') is True, 'Audit process is not closed.')
                audit_exit = audit_exit.get('actual_exit')
            require(type(audit_exit) is int and audit_exit == 0,
                    'Audit did not actually exit successfully.')
            actual = audit['actual_worker_exit']
            require(type(actual['actual_returncode']) is int and actual['actual_returncode'] == 0
                    and actual['timeout'] is False and actual['interruption_signal'] is None,
                    'Failed or interrupted training.')
            require(process['schema'] == 'ood-process-exit-v1' and type(process['exit_code']) is int
                    and process['exit_code'] == 0 and process['manifest_sha256'] == STANDARD_MANIFEST
                    and audit['manifest_sha256'] == STANDARD_MANIFEST
                    and process['started'] == actual['started'] and process['ended'] == actual['ended']
                    and process['result_sha256'] == audit['evidence_sha256']['result.json'],
                    'Stale actual process receipt.')
            for name, expected in [('host_updates', 1000000), ('critic_updates', 1000000), ('actor_updates', 500000)]:
                require(type(audit[name]) is int and audit[name] == expected, 'Wrong accepted update count.')
            final = [c for c in audit['checkpoints'] if c['step'] == 1000000]
            require(len(final) == 1 and final[0]['sha256'] == binding['checkpoint_sha256']
                    == process['reported_final_checkpoint_sha256'], 'Stale final checkpoint binding.')
            require(final[0]['decoded_counters'] == binding['counters'], 'Checkpoint counters disagree.')
            require(audit['run_input_hashes'] == binding['preparation_input_hashes']
                    and audit['data_cache_sha256'] == binding['data_cache_sha256'], 'Stale data acceptance.')
        host, bca = (readiness['checkpoints'][m] for m in ('host', 'bca'))
        require(host['data_cache_sha256'] == bca['data_cache_sha256'], 'Unpaired data cache.')
        for key in ('training', 'heldout', 'training_ids', 'heldout_ids', 'obs_mean', 'obs_std',
                    'max_action', 'max_episode_steps'):
            require(host['preparation_input_hashes'][key] == bca['preparation_input_hashes'][key],
                    'Unpaired preparation: '+key)
    except (KeyError, TypeError) as exc:
        raise ValueError('Incomplete pair evidence: '+str(exc)) from exc
    return dict(schema='ood-accepted-training-pair-v1', pair_id=readiness['pair_id'],
                scientific_manifest_sha256=STANDARD_MANIFEST,
                checkpoints=deepcopy(readiness['checkpoints']), training_accepted=True,
                process_receipts_sha256={m: sha(p) for m, p in processes.items()},
                trained_adapter_accepted=False, simulator_execution_accepted=False,
                ready_for_collection=False, fresh_coverage_accepted=False,
                checkpoint_files_reread=False, model_queries=0, simulator_steps=0)


def bind_pair_files(root, readiness_path, process_paths):
    root = Path(root)
    readiness = json.loads(Path(readiness_path).read_text(encoding='utf-8'))
    raw = gzip.decompress((root/'experiments/standard_bca/manifest.json.gz').read_bytes())
    require(hashlib.sha256(raw).hexdigest() == STANDARD_MANIFEST, 'Changed frozen manifest.')
    manifest = json.loads(raw)
    for name, expected in manifest['source_sha256'].items():
        require(file_hash(root/name) == expected, 'Changed frozen training source: '+name)
    audits, acceptances, processes, inputs = {}, {}, {}, {}
    for method, cp in readiness['checkpoints'].items():
        for field, target in [('accepted_audit', audits), ('acceptance_receipt', acceptances)]:
            item = cp[field]
            path = root/item['path']
            require(file_hash(path) == item['sha256'], 'Changed accepted evidence: '+field)
            target[method] = json.loads(path.read_text(encoding='utf-8'))
            inputs[method+'/'+field] = dict(path=str(path), sha256=item['sha256'])
        path = Path(process_paths[method])
        processes[method] = json.loads(path.read_text(encoding='utf-8'))
        inputs[method+'/process'] = dict(path=str(path), sha256=file_hash(path))
    result = bind_pair(readiness, audits, acceptances, processes)
    result.update(inputs=inputs, frozen_source_files_checked=len(manifest['source_sha256']),
                  frozen_source_sha256=manifest['source_sha256'], readiness_sha256=file_hash(readiness_path))
    return result


class Ledger:
    """Append-only, hash-chained reservations; never refund an uncertain call.

    One exclusive OS lock covers validation and append, so concurrent attempts
    share their ceiling. A torn final line fails closed. The ledger belongs to
    the campaign, not an individual attempt. Constructor calls consume a declared
    engineering reservation before constructing any real simulator.
    """
    def __init__(self, path, caps):
        self.path, self.caps = Path(path), dict(caps)
        require(caps and all(type(v) is int and v >= 0 for v in caps.values()), 'Invalid resource caps.')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            write_artifact(self.path, dict(schema='ood-resource-ledger-v1', caps=caps))
        except FileExistsError:
            pass
        with self.locked() as stream:
            self._read(stream)

    @contextmanager
    def locked(self):
        # Lock a stable companion inode; never replace or truncate the ledger.
        with self.path.with_suffix(self.path.suffix+'.lock').open('a+b') as lock:
            lock.seek(0, 2)
            if lock.tell() == 0:
                lock.write(b'0'); lock.flush()
            lock.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                with self.path.open('r+b') as stream:
                    yield stream
            finally:
                lock.seek(0)
                if os.name == 'nt':
                    msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _read(self, stream):
        stream.seek(0)
        lines = stream.readlines()
        require(lines and all(line.endswith(b'\n') for line in lines), 'Incomplete ledger; no retry.')
        header = json.loads(lines[0])
        require(header == dict(schema='ood-resource-ledger-v1', caps=self.caps), 'Changed ledger ceilings.')
        totals = {k: dict(environment=0, physics=0) for k in self.caps}
        previous = hashlib.sha256(lines[0]).hexdigest()
        for index, raw in enumerate(lines[1:]):
            record = json.loads(raw)
            require(record['previous'] == previous and record['index'] == index, 'Broken ledger chain.')
            require(record['sha256'] == sha({k:v for k,v in record.items() if k!='sha256'}), 'Changed ledger entry.')
            category = record['category']
            require(category in totals and record['environment'] == 1
                    and type(record['physics']) is int and record['physics'] >= 0, 'Malformed reservation.')
            totals[category]['environment'] += 1
            totals[category]['physics'] += record['physics']
            require(totals[category]['environment'] <= self.caps[category], 'Exceeded saved ledger cap.')
            previous = record['sha256']
        return totals, previous, len(lines)-1

    def totals(self):
        with self.locked() as stream:
            return self._read(stream)[0]

    def call(self, category, owner, frame_skip, callback):
        require(type(frame_skip) is int and frame_skip >= 0, 'Invalid physics accounting.')
        with self.locked() as stream:
            totals, previous, index = self._read(stream)
            require(category in totals and totals[category]['environment'] < self.caps[category], 'Resource ceiling reached.')
            record = dict(index=index, previous=previous, category=category, owner=owner,
                          environment=1, physics=frame_skip, utc=datetime.now(timezone.utc).isoformat())
            record['sha256'] = sha(record)
            stream.seek(0, 2); stream.write(packed(record)+b'\n'); stream.flush(); os.fsync(stream.fileno())
        return callback()


class Attempt:
    """No resume/retry API. A started or failed stage cannot run again."""
    @classmethod
    def oracle(cls, path, ledger_path, *, episodes=2, captures=(0, 2), horizon=3, simulator_limit=3):
        require(type(episodes) is int and episodes > 0 and tuple(captures) == (0, 2)
                and 1 <= horizon <= 5 and simulator_limit >= 2, 'Invalid oracle fixture declaration.')
        self = cls()
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        self.settings = dict(kind='exact_oracle_fixture_only', episodes=episodes,
                             captures=list(captures), horizon=horizon, simulator_limit=simulator_limit)
        self.streams = dict(collection=list(range(1900928000, 1900928000+episodes)),
            candidates=list(range(1900930000, 1900930000+episodes*4)),
            random_score=list(range(1900940000, 1900940000+episodes*4)),
            continuation=list(range(1900950000, 1900950000+episodes*4)))
        self.predecessors = {}
        self.declaration_sha = write_artifact(self.path/'declaration.json',
            dict(settings=self.settings, streams=self.streams, ready_for_science=False,
                 source_sha256={str(p.name): file_hash(p) for p in [Path(__file__), Path(__file__).with_name('oracle.py')]}))
        self.ledger = Ledger(ledger_path, dict(engineering=10000, collection=10000,
                                              repeat_checks=10000, outcomes=10000))
        return self

    @classmethod
    def prepare_pair(cls, path, declaration, binding):
        """Freeze the full declaration and accepted training pair; launches nothing."""
        from experiments.ood.protocol import validate_manifest
        validate_manifest(declaration)
        require(binding.get('training_accepted') is True and binding.get('ready_for_collection') is False,
                'Expected independently accepted training with OOD gates still pending.')
        pair = next((p for p in declaration['pairs'] if p['pair_id'] == binding['pair_id']), None)
        require(pair is not None and set(pair['checkpoint_ids']) ==
                {c['run_id'] for c in binding['checkpoints'].values()}, 'Pair differs from full declaration.')
        path = Path(path); path.mkdir(parents=True, exist_ok=False)
        write_artifact(path/'full-declaration.json', declaration)
        write_artifact(path/'training-binding.json', binding)
        return dict(path=str(path), training_accepted=True, behavioral_execution_accepted=False,
                    coverage_execution_accepted=False, simulator_steps=0)

    def require_scientific_execution(self):
        raise ValueError('Real execution is pending trained-action/runtime/state/resource acceptance; fixtures cannot promote it.')

    def read_stage(self, name):
        require(name in self.predecessors, 'Missing completed predecessor: '+name)
        return read_artifact(self.path/(name+'.json'), self.predecessors[name])

    @contextmanager
    def stage(self, name, predecessors=()):
        declaration = read_artifact(self.path/'declaration.json', self.declaration_sha)
        for name_in_source, expected in declaration['source_sha256'].items():
            require(file_hash(Path(__file__).with_name(name_in_source)) == expected,
                    'Execution source changed after declaration.')
        require(sha(declaration['settings']) == sha(self.settings) and sha(declaration['streams']) == sha(self.streams),
                'Changed settings or random streams.')
        # Check predecessors before constructor calls or opening new outcome files.
        for pred in predecessors:
            self.read_stage(pred)
        write_artifact(self.path/(name+'.started.json'), dict(declaration=self.declaration_sha,
                        predecessors={p:self.predecessors[p] for p in predecessors}))
        try:
            yield
        except BaseException as exc:
            write_artifact(self.path/(name+'.failed.json'), dict(error_type=type(exc).__name__,
                           message=str(exc), reservations=self.ledger.totals(), retry_allowed=False))
            raise

    def finish(self, name, payload):
        self.predecessors[name] = write_artifact(self.path/(name+'.json'), payload)
        write_artifact(self.path/(name+'.completed.json'), dict(sha256=self.predecessors[name],
                                                               reservations=self.ledger.totals()))


class OracleSimulator:
    """Only the already declared toy dynamics; no Gym, checkpoint or GPU access."""
    frame_skip = 0
    constructor_transitions = 0
    action_dim = 1

    def __init__(self, limit=3):
        self.limit = limit
        self.env = Oracle(limit=limit)

    def reset(self, seed):
        self.env = Oracle(limit=self.limit)
        return np.array([self.env.x], np.float64)

    def capture(self):
        return self.env.capture()

    def restore(self, snapshot, expected_sha256=None):
        require(expected_sha256 is None or sha(snapshot) == expected_sha256, 'Changed oracle state.')
        self.env.restore(deepcopy(snapshot))
        require(sha(snapshot) == sha(self.capture()), 'Incomplete oracle roundtrip.')

    def step(self, action):
        action = np.asarray(action)
        require(action.shape == (1,) and action.dtype.kind == 'f', 'Invalid oracle action.')
        old = self.env.x
        obs, reward, terminated, truncated = self.env.step(float(action[0]))
        reconstructed = -(old+float(action[0])-1)**2-.1*float(action[0])**2
        return dict(observation=np.array([obs]), reward=reward, terminated=terminated, truncated=truncated,
                    proposed_action=action.copy(), applied_action=action.copy(), sim_ctrl=action.copy(),
                    reconstructed_reward=reconstructed, reward_error=abs(reward-reconstructed), physics_steps=0)

    def close(self):
        pass


def _simulator(attempt, factory):
    # Do not call an arbitrary factory, which might construct a real simulator
    # before its provenance and constructor budget are accepted.
    require(attempt.settings['kind'] == 'exact_oracle_fixture_only', 'Unsupported execution kind.')
    require(isinstance(factory, partial) and factory.func is OracleSimulator
            and not factory.args and factory.keywords == {'limit': attempt.settings['simulator_limit']},
            'Only an explicit exact-oracle constructor is permitted; no arbitrary factory calls.')
    sim = factory()
    require(type(sim) is OracleSimulator, 'Only exact-oracle execution is accepted in this version.')
    return sim


def _action(policy, obs, key=None):
    action = np.asarray(policy(np.asarray(obs)[None], key=key))
    require(action.ndim == 2 and action.shape[0] == 1 and action.dtype.kind == 'f'
            and np.isfinite(action).all() and (np.abs(action) <= 1).all(), 'Invalid frozen policy action.')
    return action[0]


def _step(attempt, sim, category, owner, action):
    # Save even the request that subsequently fails. Each owner is a unique step.
    number = attempt.ledger.totals()[category]['environment']
    p = attempt.path/'transitions'; p.mkdir(exist_ok=True)
    write_artifact(p/f'{category}-{number:07d}-input.json', dict(owner=owner, action=action, state=sim.capture()))
    rec = attempt.ledger.call(category, owner, sim.frame_skip, lambda: sim.step(action))
    write_artifact(p/f'{category}-{number:07d}-output.json', rec)
    require(np.isfinite(rec['reward']) and np.isfinite(rec['reward_error']) and rec['reward_error'] <= 1e-7,
            'Independent reward gate failed.')
    require(np.array_equal(rec['applied_action'], rec['sim_ctrl']), 'Applied controls disagree.')
    return rec


def collect_states(attempt, factory, policies):
    with attempt.stage('states'):
        require(set(policies) == {'host', 'bca'}, 'Require both fixed collectors.')
        sim = _simulator(attempt, factory)
        rows, reset_states = [], {}
        try:
            for collector in ('host', 'bca'):
                for episode, seed in enumerate(attempt.streams['collection']):
                    obs = sim.reset(seed)
                    snapshot = sim.capture()
                    if episode in reset_states:
                        require(sha(snapshot) == reset_states[episode], 'Paired reset states disagree.')
                    else:
                        reset_states[episode] = sha(snapshot)
                    ended = False
                    for step in range(max(attempt.settings['captures'])+1):
                        if step in attempt.settings['captures']:
                            row = dict(state_id=f'{collector}/{episode}/{step}', collector=collector,
                                       episode=episode, reset_seed=seed, capture_step=step,
                                       status='missing' if ended else 'captured',
                                       missing_reason='episode_ended_before_capture' if ended else None)
                            if not ended:
                                snapshot = sim.capture()
                                row.update(observation=obs.copy(), snapshot=snapshot, snapshot_sha256=sha(snapshot),
                                           reset_shared_block=episode if step == 0 else None)
                            rows.append(row)
                        if ended or step == max(attempt.settings['captures']):
                            continue
                        record = _step(attempt, sim, 'collection', f'{collector}/{episode}/{step}',
                                       _action(policies[collector], obs))
                        obs = record['observation']
                        ended = record['terminated'] or record['truncated']
            attempt.finish('states', dict(kind=attempt.settings['kind'], rows=rows))
        finally:
            sim.close()


def support_query(bank, observation, actions):
    x, a = np.asarray(observation), np.asarray(actions)
    require(x.shape == bank['observations'].shape[1:] and a.ndim == 2
            and a.shape[1] == bank['actions'].shape[1] and np.isfinite(x).all()
            and np.isfinite(a).all(), 'Support query shape/values disagree.')
    distances = np.square(bank['observations'].astype(np.float64)-x).mean(axis=1)
    order = np.argsort(distances, kind='stable')[:bank['neighbors']]
    nearby = bank['actions'][order]
    rms = np.sqrt(np.square(a[:, None, :].astype(np.float64)-nearby[None]).mean(axis=-1)).min(axis=1)
    return rms, nearby[0].copy()


def support_bank(observations, actions, training_ids, episode_ids, *, seed, max_rows=32768, neighbors=32):
    """Training-complement arrays only; IDs must be accepted before calling."""
    obs, actions, ids = map(np.asarray, (observations, actions, training_ids))
    n = len(obs)
    require(obs.ndim == actions.ndim == 2 and n == len(actions) == len(ids) and n > 0
            and ids.shape == (n,) and ids.dtype.kind in 'iu' and len(set(ids.tolist())) == n
            and np.isfinite(obs).all() and np.isfinite(actions).all()
            and (abs(actions) <= 1).all(), 'Invalid training-complement inputs.')
    require(type(seed) is int and 0 <= seed < 2**32 and type(max_rows) is int and max_rows > 0
            and type(neighbors) is int and neighbors > 0, 'Invalid support preparation settings.')
    rng = np.random.default_rng(seed)
    reference_episodes, validation_episodes, validation_indices = [], [], []
    if episode_ids is None:
        pool = np.arange(n)
    else:
        episodes = np.asarray(episode_ids)
        require(episodes.shape == (n,) and episodes.dtype.kind in 'iu', 'Invalid episode IDs.')
        permutation = rng.permutation(np.unique(episodes))
        cut = int(.8*len(permutation))
        require(0 < cut < len(permutation), 'Need nonempty episode-disjoint support partitions.')
        reference_episodes, validation_episodes = permutation[:cut].tolist(), permutation[cut:].tolist()
        pool = np.flatnonzero(np.isin(episodes, reference_episodes))
        validation_indices = [int(rng.choice(np.flatnonzero(episodes == episode))) for episode in validation_episodes]
    selected = np.sort(rng.choice(pool, min(max_rows, len(pool)), replace=False))
    bank = dict(observations=obs[selected].copy(), actions=actions[selected].copy(),
                training_ids=ids[selected].copy(), reference_indices=selected,
                neighbors=min(neighbors, len(selected)), seed=seed,
                reference_episodes=reference_episodes, validation_episodes=validation_episodes,
                validation_indices=validation_indices, threshold=None,
                thresholded_labels_available=episode_ids is not None,
                input_sha256=sha(dict(observations=obs, actions=actions, training_ids=ids, episode_ids=episode_ids)))
    validation_distances = [float(support_query(bank, obs[i], actions[i:i+1])[0][0]) for i in validation_indices]
    bank['validation_distances'] = validation_distances
    if validation_distances:
        bank['threshold'] = float(np.quantile(validation_distances, .95, method='linear'))
    return bank


def freeze_candidates(attempt, policies, scorer, support):
    with attempt.stage('candidates', ('states',)):
        states = attempt.read_stage('states')['rows']
        rows = []
        for index, state in enumerate(states):
            if state['status'] != 'captured':
                rows.append(deepcopy(state)); continue
            obs = state['observation']
            host, bca = (_action(policies[m], obs) for m in ('host', 'bca'))
            _, nearest = support_query(support, obs, host[None])
            rng = np.random.default_rng(attempt.streams['candidates'][index])
            uniform = rng.uniform(-1, 1, host.shape).astype(host.dtype)
            direction = rng.normal(size=host.shape)
            require(np.isfinite(direction).all() and np.square(direction).mean() > 0, 'Invalid saved direction.')
            direction /= np.sqrt(np.square(direction).mean())
            proposed = np.stack([host, bca, nearest, uniform] +
                                [host+sign*rho*direction for rho in (.05, .15, .30) for sign in (-1, 1)])
            actions = np.clip(proposed, -1, 1)
            # The oracle's applied transform is identity. Real D4RL transform and
            # incoming dtype need separate trained/runtime acceptance, not inference.
            aliases = [next((j for j in range(i) if np.array_equal(actions[i], actions[j])), i)
                       for i in range(len(actions))]
            distances, _ = support_query(support, obs, actions)
            scores = scorer(np.repeat(obs[None], len(actions), axis=0), actions)
            required = {'width', 'scale', 'dose', 'usable', 'radius', 'bayesian_radius', 'conformal_radius', 'unit'}
            require(set(scores) == required, 'Missing frozen BCA score components.')
            for k in ('width', 'scale', 'dose', 'usable'):
                require(np.asarray(scores[k]).shape == (10,), 'Wrong score bank shape.')
            require(np.isfinite(scores['width']).all() and (scores['width'] >= 0).all()
                    and np.asarray(scores['usable']).all(), 'Unusable score must stop, not be repaired.')
            random = np.random.default_rng(attempt.streams['random_score'][index]).random(10)
            for i, j in enumerate(aliases):
                if i != j:
                    require(distances[i] == distances[j]
                            and all(scores[k][i] == scores[k][j] for k in ('width', 'scale', 'dose', 'usable')),
                            'Identical applied actions have conflicting scores.')
                    random[i] = random[j]
            seed = attempt.streams['continuation'][index]
            # Explicit toy-only keys; real execution must use adapters.continuation_keys.
            keys = np.column_stack([np.full(attempt.settings['horizon'], seed, np.uint32),
                                    np.arange(attempt.settings['horizon'], dtype=np.uint32)])
            rows.append(dict(state_id=state['state_id'], status='captured', snapshot_sha256=state['snapshot_sha256'],
                proposed=proposed, actions=actions, alias=aliases, direction=direction,
                preclip_rms=np.sqrt(np.square(proposed-host).mean(axis=1)),
                postclip_rms=np.sqrt(np.square(actions-host).mean(axis=1)),
                clipping_fraction=(proposed != actions).mean(axis=1), support=distances,
                support_threshold=support['threshold'], random=random, constant=np.ones(10),
                step_keys=keys, step_keys_sha256=sha(keys), key_contract='oracle_fixture_only', **scores))
        attempt.finish('candidates', dict(kind=attempt.settings['kind'], support_bank=support, rows=rows,
            slot_names=['host_reference', 'bca', 'nearest_recorded', 'uniform'] +
                       [f'perturb_{rho}_{sign:+d}' for rho in (.05, .15, .30) for sign in (-1, 1)]))


def run_outcomes(attempt, factory, policies):
    with attempt.stage('outcomes', ('states', 'candidates')):
        states, bank = attempt.read_stage('states')['rows'], attempt.read_stage('candidates')['rows']
        require(len(states) == len(bank) and set(policies) == {'host', 'bca'}, 'Unpaired outcome inputs.')
        sim = _simulator(attempt, factory)
        sim.reset(attempt.streams['collection'][0])
        results = []
        try:
            for state, candidate in zip(states, bank):
                require(state['state_id'] == candidate['state_id'], 'Changed state order.')
                if state['status'] != 'captured':
                    results.extend([dict(state_id=state['state_id'], continuation=m, status='missing',
                                         reason=state['missing_reason']) for m in ('host', 'bca')])
                    continue
                require(state['snapshot_sha256'] == candidate['snapshot_sha256'] == sha(state['snapshot'])
                        and sha(candidate['step_keys']) == candidate['step_keys_sha256'], 'Changed state or step keys.')
                for continuation in ('host', 'bca'):
                    returns, lengths, endings, references = [], [], [], []
                    for slot, first_action in enumerate(candidate['actions']):
                        alias = candidate['alias'][slot]
                        if alias != slot:
                            returns.append(returns[alias]); lengths.append(lengths[alias]); endings.append(endings[alias])
                            references.append(references[alias]); continue
                        owner = f'{state["state_id"]}/{continuation}/{slot}'
                        sim.restore(state['snapshot'], expected_sha256=state['snapshot_sha256'])
                        repeat = _step(attempt, sim, 'repeat_checks', owner+'/repeat', first_action)
                        repeat_state = sim.capture()
                        sim.restore(state['snapshot'], expected_sha256=state['snapshot_sha256'])
                        total, first_record = 0., None
                        for step in range(attempt.settings['horizon']):
                            action = (first_action if step == 0 else _action(policies[continuation], record['observation'],
                                                                           candidate['step_keys'][step]))
                            record = _step(attempt, sim, 'outcomes', owner+f'/{step}', action)
                            if step == 0:
                                first_record = record
                                require(sha(record) == sha(repeat) and sha(sim.capture()) == sha(repeat_state),
                                        'Repeated first transition/state differs.')
                                require(np.array_equal(record['applied_action'], first_action), 'Unexpected applied transform.')
                            total += record['reward']
                            if record['terminated'] or record['truncated']:
                                break
                        returns.append(total); lengths.append(step+1)
                        endings.append(dict(terminated=record['terminated'], truncated=record['truncated']))
                        references.append(sha(first_record))
                    result = dict(state_id=state['state_id'], continuation=continuation, status='complete',
                                  returns=returns, harm=(returns[0]-np.asarray(returns)).tolist(), lengths=lengths,
                                  endings=endings, first_transition_sha256=references, aliases=candidate['alias'])
                    # Append each panel immediately; a later interruption leaves all completed panels.
                    write_artifact(attempt.path/f'panel-{len(results):05d}.json', result)
                    results.append(result)
            attempt.finish('outcomes', dict(kind=attempt.settings['kind'], scientific_results=False, rows=results))
        finally:
            sim.close()
