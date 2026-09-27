"""Durable, cumulative OOD reservations, separate from measured transition counts.

SQLite FULL-synchronous transactions serialize competing reservations. Each call
has an exclusive token and a hash-chained entry committed BEFORE the callback.
A crash/failure never refunds a reservation. Shared engineering history is charged
once globally and to every host cell it supported. Full-chain audits occur on
open and stage completion; indexed aggregates make each reservation bounded work.
Use a local filesystem (WSL ext4), not an NFS database. No training entrypoint.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

from experiments.ood.collect import packed, sha, require


def scope_caps(pairs, cells):
    require(len(pairs) == len(set(pairs)) and len(cells) == len(set(cells)), 'Duplicate scope.')
    caps = {'global': [39998400, 159993600]}
    for cell in cells:
        caps['engineering/'+cell] = [10000, 40000]
    for pair in pairs:
        for category, count in [('collection', 12800), ('outcomes', 1280000), ('repeat_checks', 5120)]:
            caps[category+'/'+pair] = [count, count*4]
    # Fresh coverage requires a separately accepted execution contract.
    return caps


def prior_adapter_usage(receipt):
    require(receipt['schema'] == 'bca-ood-adapter-engineering-v1', 'Unknown prior receipt.')
    require(receipt['resource_ledger']['all_attempts_environment_transitions'] == 72
            and receipt['resource_ledger']['all_attempts_physics_steps'] == 288, 'Changed prior totals.')
    entries = []
    for index, attempt in enumerate(receipt['attempts']):
        require(attempt['attempt'] == f'adapters-runtime-v{index+1}'
                and attempt['explicit_transitions'] == (0 if index < 3 else 20)
                and attempt['constructor_transitions'] == 2
                and attempt['actual_exit_code'] == (1 if index < 3 else 0), 'Changed adapter history.')
        # Each declaration/test ran one environment for each of Hopper/Walker.
        # These simulator checks were shared by both initial-tranche hosts.
        for dataset in ('hopper', 'walker2d'):
            environment = 1 + attempt['explicit_transitions']//2
            entries.append(dict(token='prior/'+attempt['attempt']+'/'+dataset,
                scopes=['engineering/'+h+'/'+dataset for h in ('td3_bc', 'rebrac')],
                environment=environment, physics=environment*4,
                actual_exit=attempt['actual_exit_code'], declaration=attempt['declaration'],
                note='Shared adapter evidence: one physical charge, allowance charged to both supported host cells.'))
    require(len(entries) == 12 and sum(x['environment'] for x in entries) == 72, 'Incomplete history.')
    return entries


class ResourceLedger:
    def __init__(self, path, caps, protocol_sha256):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        require('global' in caps and all(isinstance(k, str) and isinstance(v, list) and len(v) == 2
                and all(type(n) is int and n >= 0 for n in v) for k,v in caps.items()), 'Invalid ceilings.')
        self.header = dict(schema='ood-resource-ledger-v2', caps=caps, protocol_sha256=protocol_sha256,
                           counts='reservations, not proof of completed physical transitions')
        self.db = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        try:
            self.db.execute('PRAGMA synchronous=FULL')
            self.db.execute('PRAGMA journal_mode=DELETE')
            self.db.execute('BEGIN IMMEDIATE')
            exists = self.db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='header'").fetchone()
            if not exists:
                self.db.execute('CREATE TABLE header (id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT NOT NULL)')
                self.db.execute('CREATE TABLE entries (id INTEGER PRIMARY KEY, token TEXT UNIQUE NOT NULL, payload TEXT NOT NULL, digest TEXT NOT NULL)')
                self.db.execute('CREATE TABLE totals (scope TEXT PRIMARY KEY, environment INTEGER NOT NULL, physics INTEGER NOT NULL)')
                self.db.execute('INSERT INTO header VALUES (1,?)', (packed(self.header).decode(),))
                self.db.executemany('INSERT INTO totals VALUES (?,0,0)', [(k,) for k in caps])
                for table in ('header', 'entries'):
                    for action in ('UPDATE', 'DELETE'):
                        self.db.execute(f"CREATE TRIGGER {table}_{action} BEFORE {action} ON {table} BEGIN SELECT RAISE(ABORT,'immutable journal'); END")
            self.db.execute('COMMIT')
            self.audit()
        except BaseException:
            if self.db.in_transaction:
                self.db.execute('ROLLBACK')
            self.db.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.db.close()

    def _header(self):
        actual, = self.db.execute('SELECT payload FROM header WHERE id=1').fetchone()
        require(json.loads(actual) == self.header, 'Changed resource protocol or ceilings.')

    def audit(self):
        self.db.execute('BEGIN')
        try:
            self._header()
            require(self.db.execute('PRAGMA quick_check').fetchall() == [('ok',)], 'Damaged resource database.')
            totals = {k: [0,0] for k in self.header['caps']}
            previous = sha(self.header)
            count = 0
            for index, token, payload, digest in self.db.execute('SELECT id,token,payload,digest FROM entries ORDER BY id'):
                entry = json.loads(payload)
                require(index == count and entry['index'] == index and token == entry['token']
                        and entry['previous'] == previous and sha(entry) == digest, 'Broken reservation chain.')
                self._validate(entry['scopes'], entry['environment'], entry['physics'])
                for scope in ['global', *entry['scopes']]:
                    totals[scope][0] += entry['environment']; totals[scope][1] += entry['physics']
                    require(all(a <= b for a,b in zip(totals[scope], self.header['caps'][scope])), 'Exceeded ledger ceiling.')
                previous = digest; count += 1
            saved = {s:[e,p] for s,e,p in self.db.execute('SELECT scope,environment,physics FROM totals')}
            require(saved == totals, 'Reservation aggregate corruption.')
            return dict(schema='ood-resource-audit-v2', header_sha256=sha(self.header),
                        entries=count, last_sha256=previous, reserved=totals,
                        physical_completion_not_inferred=True)
        finally:
            self.db.execute('ROLLBACK')

    def _validate(self, scopes, environment, physics):
        require(type(environment) is int and environment > 0 and type(physics) is int and physics >= 0,
                'Invalid transition/physics reservation.')
        require(isinstance(scopes, list) and scopes and len(scopes) == len(set(scopes))
                and all(s in self.header['caps'] and s != 'global' for s in scopes), 'Invalid accounting scopes.')

    def reserve(self, token, scopes, environment, physics, evidence=None):
        self._validate(scopes, environment, physics)
        require(isinstance(token, str) and token, 'Exclusive call token required.')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            self._header()
            require(self.db.execute('SELECT id FROM entries WHERE token=?', (token,)).fetchone() is None,
                    'Already reserved call; retry is forbidden.')
            tail = self.db.execute('SELECT id,payload,digest FROM entries ORDER BY id DESC LIMIT 1').fetchone()
            if tail:
                require(sha(json.loads(tail[1])) == tail[2], 'Changed reservation head.')
            index, previous = (tail[0]+1, tail[2]) if tail else (0, sha(self.header))
            entry = dict(index=index, previous=previous, token=token, scopes=scopes,
                         environment=environment, physics=physics, evidence=evidence,
                         utc=datetime.now(timezone.utc).isoformat())
            for scope in ['global', *scopes]:
                old = self.db.execute('SELECT environment,physics FROM totals WHERE scope=?', (scope,)).fetchone()
                require(old is not None and all(a+b <= c for a,b,c in zip(old, [environment,physics], self.header['caps'][scope])),
                        'Resource ceiling reached: '+scope)
                self.db.execute('UPDATE totals SET environment=environment+?,physics=physics+? WHERE scope=?',
                                (environment, physics, scope))
            digest = sha(entry)
            self.db.execute('INSERT INTO entries VALUES (?,?,?,?)', (index, token, packed(entry).decode(), digest))
            self.db.execute('COMMIT')
            return digest
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def call(self, token, scopes, environment, physics, callback, evidence=None):
        self.reserve(token, scopes, environment, physics, evidence)
        return callback()
