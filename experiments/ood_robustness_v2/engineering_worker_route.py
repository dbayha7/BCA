"""Cooperating one-use callback route; no scientific entrypoint or default resources.

Full native/runtime/lease/storage acceptance must precede real use. No Python
sandbox: independently reviewed entrypoint must make this its only callback route.
"""
import copy
import os
import sys
from pathlib import Path
from execution_capsule import EngineeringCapsule, CPU_ENV, require, digest, directory_identity
from one_shot_supervisor import (parse, process_identity, process_argv, process_environment,
                                 pinned_inputs_unchanged, evidence_path, protocol_file,fsync_directory)
from precommit_file import LIMIT,canonical,write_once


def read_line(stream):
    raw=stream.readline(LIMIT+1)
    require(raw.endswith(b'\n') and len(raw)<=LIMIT,'Truncated or oversized worker envelope.')
    return parse(raw)


class WorkerRoute:
    def __init__(self,*,stream=None):
        self.failed=False;self.used=False;self.pid=os.getpid()
        self.stream=sys.stdin.buffer if stream is None else stream
        try:
            self.envelope=read_line(self.stream);e=self.envelope
            require(type(e) is dict and set(e)=={'schema','declaration_pin','trusted_review','parent',
                'evidence_root','evidence_identity','binding_sha256'} and e['schema']=='ood-v2-worker-envelope-v1',
                'Exact worker envelope required.')
            self.capsule=EngineeringCapsule(declaration_pin=e['declaration_pin'],trusted_review=e['trusted_review'])
            self.worker=process_identity(self.pid);self.parent=copy.deepcopy(e['parent'])
            require(self.worker['ppid']==self.parent['pid'] and self.worker['pgrp']==self.pid and self.worker['sid']==self.pid,
                    'Worker must own its child session under the bound observer.')
            require(process_identity(self.parent['pid'])==self.parent,'Parent identity changed.')
            spec=self.capsule.spec
            require(process_argv(self.pid)==spec['command'] and process_environment(self.pid)==CPU_ENV,
                    'Actual isolated argv/environment mismatch.')
            # Python may add locale compatibility fields to os.environ after exec.
            # This exact contract refuses those too until separately accepted.
            require(dict(os.environ)==CPU_ENV,'Effective Python environment changed.')
            self.root=evidence_path(spec)
            require(str(self.root)==e['evidence_root'] and directory_identity(str(self.root))==e['evidence_identity'],
                    'Observer output path changed.')
            # Compare with the exact output of this independently checked capsule.
            self.binding=self.capsule.take_bootstrap()
            require(digest(self.binding)==e['binding_sha256'],'Bootstrap binding mismatch.')
            intent=protocol_file(self.root/'intent.json')
            require(intent['envelope']==e and intent['command']==spec['command']
                    and intent['environment']==CPU_ENV,'Durable observer intent differs.')
            self._contract=digest([self.envelope,self.worker,self.parent,self.binding])
            self.assert_unchanged()
            write_once(self.root/'ready.json',dict(schema='ood-v2-worker-ready-v1',worker=self.worker,
                parent=self.parent,binding_sha256=e['binding_sha256']))
            # Marker is created only after ready.json has completed file+directory
            # fsync. The observer never reads a still-being-written ready payload.
            fd=os.open(self.root/'ready_ack',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
            try:os.fsync(fd)
            finally:os.close(fd)
            fsync_directory(self.root)
            release=read_line(self.stream)
            expected=dict(schema='ood-v2-worker-release-v1',worker=self.worker,binding_sha256=e['binding_sha256'])
            require(release==expected and protocol_file(self.root/'release_intent.json')==expected,
                    'Release is missing, altered or not durably recorded.')
            observed=protocol_file(self.root/'observed_ready.json')
            require(observed['ready']==dict(schema='ood-v2-worker-ready-v1',worker=self.worker,
                parent=self.parent,binding_sha256=e['binding_sha256']) and observed['command']==spec['command']
                and observed['environment']==CPU_ENV,'Observed process receipt differs.')
            self.assert_unchanged()
        except BaseException:self.failed=True;raise

    def assert_unchanged(self):
        try:
            require(not self.failed and self.pid==os.getpid(),'Worker route failed or forked.')
            require(digest([self.envelope,self.worker,self.parent,self.binding])==self._contract,'Worker contract mutated.')
            require(process_identity(self.pid)==self.worker and process_identity(self.parent['pid'])==self.parent,
                    'Worker/observer process identity lost.')
            require(directory_identity(str(self.root))==self.envelope['evidence_identity'],'Observer output directory changed.')
            require(dict(os.environ)==CPU_ENV and process_environment(self.pid)==CPU_ENV,'Worker CPU environment changed.')
            require(process_argv(self.pid)==self.capsule.spec['command'],'Worker command changed.')
            pinned_inputs_unchanged(self.capsule)
        except BaseException:self.failed=True;raise

    def run(self,callback):
        try:
            require(not self.used and callable(callback),'Worker callback already consumed or invalid.')
            self.used=True;self.assert_unchanged()
            value=callback(copy.deepcopy(self.binding),self.assert_unchanged)
            self.assert_unchanged()
            write_once(self.root/'route_result.json',dict(schema='ood-v2-worker-route-result-v1',worker=self.worker,
                binding_sha256=self.envelope['binding_sha256'],callback_completed=True,science_accepted=False))
            return value
        except BaseException:self.failed=True;raise
