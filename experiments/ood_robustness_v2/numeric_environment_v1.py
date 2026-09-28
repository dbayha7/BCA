"""Exact numeric import profile, not a scientific or native-simulator environment.

Never mutates os.environ. The caller must independently bind this source, the
policy and actual import route. Phase names alone do not prove which imports ran.
"""
import copy
import hashlib
import json
import os
from pathlib import Path


def contract():
    return dict(schema='ood-v2-numeric-environment-v1',
        environment={'CUDA_VISIBLE_DEVICES':'','JAX_PLATFORMS':'cpu',
            'XLA_PYTHON_CLIENT_PREALLOCATE':'false','OMP_NUM_THREADS':'1',
            'OPENBLAS_NUM_THREADS':'1','LC_CTYPE':'C.UTF-8',
            'TF_CPP_MIN_LOG_LEVEL':'1','TPU_SKIP_MDS_QUERY':'1'},
        phases=['python','numpy','jax','cpu_backend'],
        scope='Python3.10/NumPy2.2.6/JAX0.6.2 CPU imports only',
        native_environment_accepted=False,scientific_execution_accepted=False)


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def process_environment(pid):
    raw=Path('/proc/%d/environ'%pid).read_bytes()
    if not raw.endswith(b'\0'):raise ValueError('Incomplete kernel environment.')
    pairs=[x.decode().split('=',1) for x in raw[:-1].split(b'\0')]
    if any(len(x)!=2 for x in pairs) or len({x[0] for x in pairs})!=len(pairs):
        raise ValueError('Malformed or duplicated kernel environment.')
    return dict(pairs)


class NumericEnvironmentGuard:
    def __init__(self,policy,*,kernel_reader=process_environment,effective_reader=None,pid_reader=os.getpid):
        self.failed=False;self.completed=False;self.index=-1
        self.kernel_reader=kernel_reader;self.effective_reader=effective_reader or (lambda:dict(os.environ));self.pid_reader=pid_reader
        self.pid=pid_reader();self.policy=copy.deepcopy(policy)
        try:
            if canonical(policy)!=canonical(contract()):raise ValueError('Exact versioned numeric profile required.')
            self.policy_sha256=hashlib.sha256(canonical(policy)).hexdigest()
            self.check()
        except BaseException:self.failed=True;raise

    def check(self):
        try:
            if self.failed or self.pid_reader()!=self.pid:raise ValueError('Environment guard failed or forked.')
            if hashlib.sha256(canonical(self.policy)).hexdigest()!=self.policy_sha256:raise ValueError('Environment policy mutated.')
            effective=self.effective_reader();kernel=self.kernel_reader(self.pid)
            if type(effective) is not dict or type(kernel) is not dict or effective!=self.policy['environment'] or kernel!=self.policy['environment']:
                raise ValueError('Exact launch and effective numeric environments required.')
            return dict(pid=self.pid,kernel_environment=kernel,effective_environment=effective)
        except BaseException:self.failed=True;raise

    def advance(self,phase):
        try:
            state=self.check()
            if self.completed or self.index+1>=len(self.policy['phases']) or phase!=self.policy['phases'][self.index+1]:
                raise ValueError('Skipped, repeated or unknown import phase.')
            self.index+=1
            return dict(phase=phase,**state)
        except BaseException:self.failed=True;raise

    def complete(self):
        try:
            self.check()
            if self.completed or self.index!=3:raise ValueError('Incomplete or reused environment route.')
            self.completed=True
        except BaseException:self.failed=True;raise
