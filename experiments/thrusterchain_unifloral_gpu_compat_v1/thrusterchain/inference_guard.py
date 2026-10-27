"""Cooperative Python instrumentation for the benchmark's built-in evaluator.

Counters cover supported patched lookups and the active benchmark context hooks.
Reads, previously saved file-I/O callables, native calls, arbitrary adaptation and
thread isolation are not comprehensively monitored. Zero counters are not proof
of arbitrary-code inference purity or of no I/O. Artifact writes normally happen
outside this context, after rollouts finish; this is not an OS security sandbox.
"""
import builtins
import io
import os
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch

class InferenceGuard:
    """Refuse supported calls, without claiming exhaustive I/O monitoring.

    Context checks inside built-in training functions also cover aliases of those
    functions. Saved aliases of Python file-opening functions have no such hook.
    With activate_context=False, a nested evaluator owns its context ledger; that
    evaluator's forbidden-call counts are not merged into this guard's ledger.
    """
    def __init__(self, activate_context=True):
        self.events={"training_calls":0,"dataset_collection_calls":0,"write_attempts":0}
        self.stack=ExitStack()
        self.activate_context=activate_context
        self.ledger={"forbidden_api_calls":0}

    def __enter__(self):
        from . import datasets, regular
        if self.activate_context:
            token=datasets._ACTIVE_EVALUATION.set(self.ledger)
            self.stack.callback(datasets._ACTIVE_EVALUATION.reset,token)
        def deny(kind):
            def rejected(*args,**kwargs):
                self.events[kind]+=1
                raise RuntimeError("inference-only guard refused "+kind)
            return rejected
        self.stack.enter_context(patch.object(regular,"train_behavior_cloning",deny("training_calls")))
        self.stack.enter_context(patch.object(datasets,"generate_dataset",deny("dataset_collection_calls")))
        original_open=builtins.open; original_io_open=io.open; original_os_open=os.open
        original_path_open=Path.open
        def check_open(original):
            def guarded(file,mode="r",*args,**kwargs):
                if isinstance(mode,str) and any(ch in mode for ch in "wax+"):
                    return deny("write_attempts")(file,mode)
                return original(file,mode,*args,**kwargs)
            return guarded
        def guarded_os(file,flags,*args,**kwargs):
            if flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):
                return deny("write_attempts")(file,flags)
            return original_os_open(file,flags,*args,**kwargs)
        self.stack.enter_context(patch.object(builtins,"open",check_open(original_open)))
        self.stack.enter_context(patch.object(io,"open",check_open(original_io_open)))
        self.stack.enter_context(patch.object(os,"open",guarded_os))
        self.stack.enter_context(patch.object(Path,"open",check_open(original_path_open)))
        return self

    def __exit__(self,*exc):
        return self.stack.__exit__(*exc)

    def receipt(self):
        return {**self.events,**self.ledger,
                "scope":"Cooperative instrumentation of supported benchmark entry points and Python file-write lookups",
                "guarantee":"Supported instrumented calls are refused; no OS sandbox or arbitrary-code purity guarantee",
                "context_guard_activated":bool(self.activate_context),
                "supported_write_lookups":["builtins.open", "io.open", "os.open", "pathlib.Path.open"],
                "counter_semantics":{
                    "training_calls":"Calls through the patched regular.train_behavior_cloning attribute",
                    "dataset_collection_calls":"Calls through the patched datasets.generate_dataset attribute",
                    "write_attempts":"Write-mode calls through supported patched lookups while this guard is active",
                    "forbidden_api_calls":"Benchmark hooks using this guard's context-local ledger; separately owned evaluator ledgers are not merged"},
                "zero_counter_interpretation":"Zero means no attempt observed through this guard's instrumented paths; it does not establish absence of I/O, data access, or adaptation",
                "all_file_writes_monitored":False,
                "read_io_monitored":False,
                "arbitrary_code_inference_purity_verified":False,
                "unsupported_monitoring":["read_io", "preimported_file_io_aliases",
                    "already_open_file_handles", "native_calls", "threads", "subprocesses",
                    "transient_state_adaptation", "arbitrary_external_training"]}

