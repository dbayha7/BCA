"""Validator safety fixtures only; no real Gym/JAX/model calls."""
from pathlib import Path
from contextlib import ExitStack
import json
import os
import signal
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch

import validate_runtime as v
import design
import launch


class ValidationTests(unittest.TestCase):
    def test_validation_reservation_is_separate_and_immutable(self):
        with tempfile.TemporaryDirectory(prefix='.fixture_validate_', dir=Path(__file__).resolve().parent) as directory:
            output=v.reserve_validation(directory,'abc',17)
            self.assertEqual(output.name,'runtime_validation_v1')
            self.assertFalse((Path(directory)/'ATTEMPT.json').exists())
            with self.assertRaises(FileExistsError): v.reserve_validation(directory,'def',17)

    def test_checker_cap_is_prospective(self):
        class Base:
            def reset(self): return ('observation','info')
            def step(self): return ('observation','reward')
        ledger=dict(resets_attempted=0,resets_completed=0,transitions_attempted=0,transitions_completed=0)
        env=v.counted_environment(Base,ledger,transition_cap=1,reset_cap=1)()
        self.assertEqual(env.reset(),('observation','info'))
        env.step()
        with self.assertRaisesRegex(ValueError,'transition cap'): env.step()
        with self.assertRaisesRegex(ValueError,'reset cap'): env.reset()
        self.assertEqual(ledger,dict(resets_attempted=1,resets_completed=1,transitions_attempted=1,transitions_completed=1))

    def test_checker_failure_retains_attempted_count(self):
        class Base:
            def step(self): raise RuntimeError('fixture failure')
        ledger=dict(resets_attempted=0,resets_completed=0,transitions_attempted=0,transitions_completed=0)
        env=v.counted_environment(Base,ledger)()
        with self.assertRaises(RuntimeError): env.step()
        self.assertEqual((ledger['transitions_attempted'],ledger['transitions_completed']),(1,0))

    def test_checker_progress_is_saved_before_and_after_dispatch(self):
        class Base:
            def step(self): return None
        ledger=dict(resets_attempted=0,resets_completed=0,transitions_attempted=0,transitions_completed=0)
        snapshots=[]
        env=v.counted_environment(Base,ledger,progress=lambda:snapshots.append(dict(ledger)))()
        env.step()
        self.assertEqual([(x['transitions_attempted'],x['transitions_completed']) for x in snapshots],[(1,0),(1,1)])

    def test_wrong_runtime_rejects_before_validation_reservation(self):
        with patch.object(v.sys,'platform','win32'), patch.object(v,'reserve_validation') as reserve:
            with self.assertRaisesRegex(ValueError,'existing isolated Ubuntu'): v.main({}, {})
            reserve.assert_not_called()

    def test_declared_training_and_simulator_arithmetic(self):
        c=design.maximum_counts()
        self.assertEqual(c['critic_updates'],360000)
        self.assertEqual(c['actor_updates'],180000)
        self.assertEqual(c['calibrations'],36)
        self.assertEqual(c['simulator_transition_cap'],c['evaluation_free_steps']+c['q_continuation_steps']+c['evaluation_prefix_steps']+c['parity_steps'])

    def _main_fixture(self, stack, directory, supervise):
        root=Path(directory)
        (root/'protocol.json').write_text('{}')
        (root/'pyvenv.cfg').write_text('include-system-site-packages = false\n')
        stack.enter_context(patch.object(v,'HERE',root))
        stack.enter_context(patch.object(v.sys,'platform','linux'))
        stack.enter_context(patch.object(v.sys,'prefix',str(root)))
        stack.enter_context(patch.object(launch,'EXPECTED_ENV',str(root)))
        stack.enter_context(patch.object(launch,'verify_prepared',return_value=({'p':1},{'m':2})))
        stack.enter_context(patch.object(launch,'verify_inputs',return_value={}))
        stack.enter_context(patch.object(launch,'scoped_process_check'))
        stack.enter_context(patch.object(launch,'supervise',side_effect=supervise))
        return root

    def test_changed_protocol_rejected_before_reservation(self):
        with tempfile.TemporaryDirectory(prefix='.fixture_validate_', dir=Path(__file__).resolve().parent) as directory, ExitStack() as stack:
            root=self._main_fixture(stack,directory,lambda *a,**k:None)
            with self.assertRaisesRegex(ValueError,'fresh prepared inputs'): v.main({'p':9},{'m':2})
            self.assertFalse((root/'runtime_validation_v1').exists())

    def test_spawn_failure_keeps_terminal_receipt_and_restores_signal_handlers(self):
        previous=signal.getsignal(signal.SIGTERM)
        observed=[]
        def fail(*args,**kwargs):
            observed.append(signal.getsignal(signal.SIGTERM))
            raise OSError('synthetic spawn failure')
        with tempfile.TemporaryDirectory(prefix='.fixture_validate_', dir=Path(__file__).resolve().parent) as directory, ExitStack() as stack:
            root=self._main_fixture(stack,directory,fail)
            self.assertEqual(v.main({'p':1},{'m':2}),1)
            result=json.loads((root/'runtime_validation_v1/RESULT.json').read_text())
            self.assertIn('spawn failure',result['error'])
            self.assertFalse(result['passed'])
            self.assertFalse(result['execution_counts_known'])
            self.assertFalse((root/'ATTEMPT.json').exists())
        self.assertTrue(callable(observed[0]))
        self.assertEqual(signal.getsignal(signal.SIGTERM),previous)

    def test_killed_worker_retains_last_durable_attempt_counts(self):
        def stopped(command,output,*args,**kwargs):
            v.atomic_json(output/'progress.json',dict(model_progress={'counts':{'critic_attempted':3,'critic_completed':2}},
                gymnasium_ledger={'resets_attempted':0,'resets_completed':0,'transitions_attempted':0,'transitions_completed':0}))
            return dict(exit_code=-9,timed_out=True,interrupted=False,error=None,elapsed_seconds=900.)
        with tempfile.TemporaryDirectory(prefix='.fixture_validate_', dir=Path(__file__).resolve().parent) as directory, ExitStack() as stack:
            root=self._main_fixture(stack,directory,stopped)
            self.assertEqual(v.main({'p':1},{'m':2}),1)
            result=json.loads((root/'runtime_validation_v1/RESULT.json').read_text())
            self.assertEqual(result['accounting_scope'],'last_progress_lower_bound')
            self.assertEqual(result['last_progress']['model_progress']['counts']['critic_attempted'],3)
            self.assertFalse(result['execution_counts_known'])

    def _worker_fixture(self, stack, directory, runtime):
        root=Path(directory); (root/'protocol.json').write_text('{}')
        output=v.reserve_validation(root,v.sha(root/'protocol.json'),os.getppid())
        stack.enter_context(patch.object(v,'HERE',root))
        stack.enter_context(patch.object(launch,'verify_prepared',return_value=({},{})))
        stack.enter_context(patch.object(launch,'verify_inputs',return_value={}))
        stack.enter_context(patch.object(v.sys,'version_info',(3,12,14)))
        stack.enter_context(patch.object(v.importlib.metadata,'version',return_value='1.0.0'))
        return root,output

    def test_worker_entry_marker_is_exclusive_before_model_calls(self):
        with tempfile.TemporaryDirectory(prefix='.fixture_validate_', dir=Path(__file__).resolve().parent) as directory, ExitStack() as stack:
            _,output=self._worker_fixture(stack,directory,None)
            (output/'WORKER_ENTRY.json').write_text('{}')
            load=stack.enter_context(patch.object(v,'module'))
            with self.assertRaises(FileExistsError):v.worker(output)
            load.assert_not_called()

    def test_model_exception_preserves_callback_snapshot(self):
        def fail(protocol,progress):
            progress({'stage':'dispatch','counts':{'critic_attempted':1,'critic_completed':0},'execution_counts_known':False})
            raise KeyboardInterrupt('fixture interrupted')
        with tempfile.TemporaryDirectory(prefix='.fixture_validate_', dir=Path(__file__).resolve().parent) as directory, ExitStack() as stack:
            _,output=self._worker_fixture(stack,directory,None)
            stack.enter_context(patch.object(v,'module',return_value=SimpleNamespace(synthetic_model_check=fail)))
            self.assertEqual(v.worker(output),1)
            result=json.loads((output/'report.json').read_text())
            self.assertEqual(result['model_progress']['counts']['critic_attempted'],1)
            self.assertFalse(result['execution_counts_known'])

    def test_reset_failure_cannot_claim_exact_counts(self):
        def model(protocol,progress):
            result={'status':'passed_synthetic_runtime_check','counts':{},'execution_counts_known':True}
            progress(result);return result
        class Env:
            def __init__(self,**kwargs):pass
            def reset(self):raise RuntimeError('fixture reset failure')
            def close(self):pass
        core=SimpleNamespace(ThrusterChainEnv=Env)
        runtime=SimpleNamespace(synthetic_model_check=model,
            real_source_initialization_check=lambda *a,**k:dict(status='passed_real_source_initialization_check',
                counts={'source_tuples_completed':0},execution_counts_known=True))
        loader=SimpleNamespace(_core=lambda:(core,{}))
        checker=ModuleType('gymnasium.utils.env_checker');checker.check_env=lambda env:env.reset()
        with tempfile.TemporaryDirectory(prefix='.fixture_validate_', dir=Path(__file__).resolve().parent) as directory, ExitStack() as stack:
            _,output=self._worker_fixture(stack,directory,None)
            stack.enter_context(patch.object(v,'module',side_effect=lambda name,path:runtime if 'runtime' in name else loader))
            stack.enter_context(patch.dict(sys.modules,{'gymnasium.utils.env_checker':checker}))
            self.assertEqual(v.worker(output),1)
            result=json.loads((output/'report.json').read_text())
            self.assertEqual(result['gymnasium_ledger']['resets_attempted'],1)
            self.assertEqual(result['gymnasium_ledger']['resets_completed'],0)
            self.assertFalse(result['execution_counts_known'])


if __name__=='__main__': unittest.main()
