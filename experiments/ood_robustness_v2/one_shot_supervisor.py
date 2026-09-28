"""One cooperating engineering child, exact environment, durable observed exits.

Linux only. Semantic reviewer provenance and the entire actual entrypoint/runtime
must be independently accepted. This is not a sandbox, descendant containment or
permission to create a real resource ledger or perform physics.
"""
import copy
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

from execution_capsule import (EngineeringCapsule, CPU_ENV, identity, path_value,
                               directory_identity, require, digest)
from precommit_file import canonical, write_once, _object, LIMIT
from ancestor_guard import no_sidecars


def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()


def parse(raw):
    require(type(raw) is bytes and 0 < len(raw) <= LIMIT, 'Bounded protocol bytes required.')
    try:
        value = json.loads(raw, object_pairs_hook=_object,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite protocol.')))
        canonical(value)
    except (UnicodeError, RecursionError) as e:
        raise ValueError('Invalid protocol JSON.') from e
    return value


def process_identity(pid):
    require(type(pid) is int and pid > 0, 'Positive process identity required.')
    raw = Path('/proc/%d/stat' % pid).read_text()
    tail = raw[raw.rfind(')') + 2:].split()
    require(len(tail) >= 20, 'Incomplete process observation.')
    return dict(pid=pid, ppid=int(tail[1]), pgrp=int(tail[2]), sid=int(tail[3]), start_ticks=int(tail[19]))


def process_argv(pid):
    raw = Path('/proc/%d/cmdline' % pid).read_bytes()
    require(raw.endswith(b'\0'), 'Incomplete argv observation.')
    return [v.decode() for v in raw[:-1].split(b'\0')]


def process_environment(pid):
    raw = Path('/proc/%d/environ' % pid).read_bytes()
    require(raw.endswith(b'\0'), 'Incomplete environment observation.')
    result = {}
    for item in raw[:-1].split(b'\0'):
        k, v = item.decode().split('=', 1)
        require(k not in result, 'Duplicate environment field.')
        result[k] = v
    return result


def evidence_path(spec):
    root=path_value(spec['paths']['run_root'] + '.supervision-v1')
    for name in ('lock','ancestor'):
        resource=path_value(spec['paths'][name])
        require(not root.is_relative_to(resource) and not resource.is_relative_to(root),
                'Supervisor evidence aliases or contains a resource path.')
    return root


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def pinned_inputs_unchanged(capsule):
    require(not capsule.failed and digest([capsule.spec,capsule.declaration_pin,capsule.trusted_review,
            capsule.pins,capsule.parent_identity,capsule.ancestor])==capsule._contract,
            'Consumed capsule contract mutated.')
    for pin in capsule.pins.values():
        require(identity(pin['path']) == pin['identity'], 'Pinned input changed during dispatch.')
    require(directory_identity(capsule.spec['paths']['parent']) == capsule.parent_identity, 'Parent directory replaced.')
    require(identity(capsule.ancestor['path']) == capsule.ancestor['identity'], 'Original ancestor changed.')
    no_sidecars(capsule.ancestor['path'])


def protocol_file(path):
    before = identity(path)
    require(0 < before[2] <= LIMIT, 'Unbounded protocol file.')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as f:
        s = os.fstat(f.fileno())
        require([s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns] == before, 'Protocol file replaced.')
        raw = f.read(LIMIT + 1)
    require(identity(path) == before and len(raw) == before[2], 'Protocol changed during read.')
    return parse(raw)


class OneShotSupervisor:
    """Caller must independently trust supervisor, reviewer and worker route.

    A fixed exclusive evidence directory makes repeated attempts refuse even
    after this observer dies. A missing actual exit is unknown, never success.
    The worker owns any future lease; the parent acquires no resource lock.
    """
    def __init__(self, *, declaration_pin, trusted_review, timeout_seconds=300, max_log_bytes=8_000_000):
        require(type(timeout_seconds) in (int,float) and math.isfinite(timeout_seconds)
                and 0 < timeout_seconds <= 86400, 'Bounded positive timeout required.')
        require(type(max_log_bytes) is int and 0 < max_log_bytes <= 64_000_000, 'Bounded log ceiling required.')
        self.capsule = EngineeringCapsule(declaration_pin=declaration_pin, trusted_review=trusted_review)
        self.pid = os.getpid(); self.owner = process_identity(self.pid)
        self.used = False; self.failed = False
        self.timeout_seconds = timeout_seconds; self.max_log_bytes = max_log_bytes
        self._contract = digest([self.timeout_seconds,self.max_log_bytes,self.owner])

    def run(self):
        require(not self.used and not self.failed and os.getpid() == self.pid, 'Supervisor reused, failed or forked.')
        self.used = True
        p = None; observed = None; root = None; root_id = None
        started = utc(); failure = None; timed_out = False; released = False
        signals_sent = []; result = None; logs = []
        try:
            require(digest([self.timeout_seconds,self.max_log_bytes,self.owner]) == self._contract,
                    'Supervisor settings mutated.')
            require(process_identity(self.pid) == self.owner, 'Supervisor identity changed.')
            binding = self.capsule.take_bootstrap()
            spec = self.capsule.spec; root = evidence_path(spec)
            # Explicit claim survives all later failure. No cleanup/reopen path.
            root.mkdir(mode=0o700, exist_ok=False); root_id = directory_identity(str(root))
            fsync_directory(root.parent)
            envelope = dict(schema='ood-v2-worker-envelope-v1', declaration_pin=self.capsule.declaration_pin,
                            trusted_review=self.capsule.trusted_review, parent=self.owner,
                            evidence_root=str(root), evidence_identity=root_id, binding_sha256=digest(binding))
            write_once(root/'intent.json', dict(schema='ood-v2-supervisor-intent-v1', started=started,
                envelope=envelope, command=binding['command'], cwd=binding['cwd'], environment=binding['environment'],
                timeout_seconds=self.timeout_seconds, max_log_bytes=self.max_log_bytes, automatic_retry=False))
            pinned_inputs_unchanged(self.capsule)
            for name in ('stdout','stderr'):
                f=(root/name).open('xb'); f.flush(); os.fsync(f.fileno()); logs.append(f)
            fsync_directory(root)
            p = subprocess.Popen(binding['command'], cwd=binding['cwd'], env=copy.deepcopy(binding['environment']),
                                 stdin=subprocess.PIPE, stdout=logs[0], stderr=logs[1], start_new_session=True,
                                 close_fds=True)
            deadline=time.monotonic()+self.timeout_seconds
            p.stdin.write(canonical(envelope)+b'\n'); p.stdin.flush()
            while not (root/'ready_ack').exists():
                require(p.poll() is None, 'Worker exited before ready handshake.')
                require(time.monotonic()<deadline, 'Worker timeout before ready handshake.')
                self._check_logs(root, logs, root_id)
                time.sleep(.02)
            ready=protocol_file(root/'ready.json')
            observed=process_identity(p.pid)
            require(observed['ppid']==self.pid and observed['pgrp']==p.pid and observed['sid']==p.pid,
                    'Worker process relationship changed.')
            require(process_argv(p.pid)==binding['command'] and process_environment(p.pid)==CPU_ENV,
                    'Worker actual command/environment differs.')
            require(ready==dict(schema='ood-v2-worker-ready-v1', worker=observed, parent=self.owner,
                               binding_sha256=envelope['binding_sha256']), 'Invalid worker readiness binding.')
            pinned_inputs_unchanged(self.capsule)
            self._check_logs(root, logs, root_id)
            write_once(root/'observed_ready.json',dict(utc=utc(),ready=ready,command=process_argv(p.pid),
                                                     environment=process_environment(p.pid)))
            # Intent and actual ready identity must be durable BEFORE release.
            release=dict(schema='ood-v2-worker-release-v1',worker=observed,binding_sha256=envelope['binding_sha256'])
            write_once(root/'release_intent.json',release)
            p.stdin.write(canonical(release)+b'\n');p.stdin.flush();p.stdin.close();released=True
            while p.poll() is None:
                if time.monotonic()>=deadline:
                    timed_out=True;raise TimeoutError('Worker runtime timeout.')
                self._check_logs(root,logs,root_id)
                require(process_identity(self.pid)==self.owner, 'Supervisor identity changed.')
                time.sleep(.02)
            require(p.returncode==0,'Worker actual exit is nonzero.')
            self._check_logs(root,logs,root_id)
            pinned_inputs_unchanged(self.capsule)
            result=protocol_file(root/'route_result.json')
            require(result==dict(schema='ood-v2-worker-route-result-v1',worker=observed,
                binding_sha256=envelope['binding_sha256'],callback_completed=True,science_accepted=False),
                'Worker completion binding missing or invalid.')
        except BaseException as exc:
            self.failed=True;failure=type(exc).__name__+': '+str(exc)
            timed_out=timed_out or 'timeout' in str(exc).lower()
        finally:
            if p is not None and p.poll() is None:
                # Signal only this still-identical child session. Never a shared worker.
                try:
                    current=process_identity(p.pid)
                    require(current['ppid']==self.pid and current['pgrp']==p.pid and current['sid']==p.pid
                            and (observed is None or current==observed), 'Refusing to signal changed process identity.')
                    os.killpg(p.pid,signal.SIGTERM);signals_sent.append(int(signal.SIGTERM))
                    try:p.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        require(process_identity(p.pid)==current,'Worker identity changed before forced stop.')
                        os.killpg(p.pid,signal.SIGKILL);signals_sent.append(int(signal.SIGKILL));p.wait(timeout=2)
                except ProcessLookupError:p.wait(timeout=2)
                except FileNotFoundError:p.wait(timeout=2)
            if p is not None and p.stdin is not None and not p.stdin.closed:p.stdin.close()
            for f in logs:f.flush();os.fsync(f.fileno());f.close()
            if root_id is not None:
                require(directory_identity(str(root))==root_id,'Evidence directory changed; retain unresolved intent.')
                receipt=dict(schema='ood-v2-supervisor-actual-exit-v1',started=started,ended=utc(),
                    command=copy.deepcopy(self.capsule.spec['command']),declaration_sha256=self.capsule.declaration_pin['sha256'],
                    review_sha256=self.capsule.trusted_review['report']['sha256'],parent=self.owner,worker=observed,
                    launched_pid=None if p is None else p.pid,actual_returncode=None if p is None else p.returncode,
                    timeout=timed_out,signals_sent=signals_sent,release_sent=released,route_completed=failure is None,
                    failure=failure,science_accepted=False,automatic_retry=False)
                write_once(root/'actual_exit.json',receipt)
        if failure is not None:raise RuntimeError(failure)
        return receipt

    def _check_logs(self,root,logs,root_id):
        require(directory_identity(str(root))==root_id,'Evidence directory changed.')
        for name,f in zip(('stdout','stderr'),logs):
            s=os.fstat(f.fileno());t=(root/name).lstat()
            require((s.st_dev,s.st_ino)==(t.st_dev,t.st_ino) and t.st_nlink==1,
                    'Log path replaced or aliased.')
            require(s.st_size<=self.max_log_bytes,'Worker log ceiling exceeded.')
