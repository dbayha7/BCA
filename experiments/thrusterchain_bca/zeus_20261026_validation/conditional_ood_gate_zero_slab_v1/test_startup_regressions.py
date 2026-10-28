"""No dynamics dispatch, model imports, episode decoding or WSL launch."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import launch
import reference
import worker

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / 'conditional_ood_gate_v1'
CORE = HERE.parents[1] / 'titan_20261026/payload/transfer_bundle/core_v4/thrusterchain'


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def synthetic_dynamic_core():
    path = CORE / 'canonical/thrusterchain.py'
    definition = next(n for n in ast.parse(path.read_bytes()).body if isinstance(n, ast.FunctionDef) and n.name == 'dynamics')
    dynamic = ModuleType('synthetic_dynamic_canonical')
    dynamic.__dict__['STALL'] = None
    # Define the authentic function, but never call it. Its code filename and
    # globals match the actual AST-constructed module's import behavior.
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path.resolve()), 'exec'), dynamic.__dict__)
    physics = {'stall': True, 'stall_slope': 1.5, 'knee': reference.KNEE,
               'gain': .3, 'throttle_cost': .3, 'wave_turn': .27, 'gamma': .98, 'length': 100}
    config = {'physics': physics, 'state_dim':13, 'action_dim':4, 'observation_dim':14,
              'observation_mode':'time_aware', 'task_semantics':'finite_horizon', 'objective_gamma':.98,
              'horizon':100, 'terminal_reward':0., 'terminal_bootstrap':False,
              'matrix_sha256':reference.MATRIX_SHA256}
    return SimpleNamespace(__file__=str(CORE / 'environment.py'), canonical=dynamic,
                           dynamics=dynamic.dynamics, canonical_config=lambda: copy.deepcopy(config),
                           Physics=lambda **kw: SimpleNamespace(as_dict=lambda: physics.copy()))


class StartupRegressions(unittest.TestCase):
    def test_reproduces_original_authentication_failure(self):
        prior = module_at('original_gate_reference', PRIOR / 'reference.py')
        with self.assertRaisesRegex(AttributeError, '__file__'):
            prior.authenticate_core(synthetic_dynamic_core())

    def test_dynamic_module_without_file_authenticates_fixed_source(self):
        core = synthetic_dynamic_core()
        self.assertFalse(hasattr(core.canonical, '__file__'))
        receipt = reference.authenticate_core(core)
        self.assertEqual(receipt['canonical_sha256'], reference.CANONICAL_SHA256)
        self.assertEqual(Path(receipt['canonical_source']).resolve(), (CORE / 'canonical/thrusterchain.py').resolve())

    def test_wrong_compilation_path_or_globals_rejected(self):
        core = synthetic_dynamic_core()
        core.__file__ = str(CORE / 'wrong/environment.py')
        with self.assertRaisesRegex(ValueError, 'compilation path'):
            reference.authenticate_core(core)
        core = synthetic_dynamic_core()
        other = ModuleType('wrong_globals')
        other.dynamics = core.dynamics
        core.canonical = other
        with self.assertRaisesRegex(ValueError, 'globals binding'):
            reference.authenticate_core(core)

    def test_source_hash_is_still_enforced(self):
        with patch.object(reference, 'CANONICAL_SHA256', 'incorrect fixture hash'):
            with self.assertRaisesRegex(ValueError, 'source hash'):
                reference.authenticate_core(synthetic_dynamic_core())

    def test_worker_persists_original_exception_and_phase(self):
        captured = {}
        def capture(path, value, **kwargs):
            captured[path.name] = copy.deepcopy(value)
            return 'synthetic-receipt-hash'
        with tempfile.TemporaryDirectory() as d, patch.object(launch, 'verify_prepared', side_effect=AttributeError('original startup fixture')), patch.object(worker, '_json', side_effect=capture):
            result = worker.run(Path(d), {}, None, float('inf'))
        saved = captured['report.json']
        self.assertEqual(result['status'], 'failed_or_interrupted_design_gate')
        self.assertEqual(saved['error']['type'], 'AttributeError')
        self.assertEqual(saved['error']['message'], 'original startup fixture')
        self.assertIn('original startup fixture', saved['error']['traceback'])
        self.assertEqual(saved['failed_phase'], 'verify_prepared')
        self.assertEqual(saved['ledger']['transition_steps_attempted'], 0)

    def test_geometry_contract_mismatch_is_refused_before_any_work(self):
        captured = {}
        def capture(path, value, **kwargs):
            captured[path.name] = copy.deepcopy(value)
            return 'synthetic-receipt-hash'
        contract = json.loads((HERE / 'contract.json').read_text(encoding='utf-8'))
        contract['region']['half_width'] = 0.05
        with tempfile.TemporaryDirectory() as d, \
                patch.object(launch, 'verify_prepared', return_value={}), \
                patch.object(worker, '_runtime', return_value={}), \
                patch.object(worker, 'HERE', Path(d)), \
                patch.object(worker, '_json', side_effect=capture):
            (Path(d) / 'contract.json').write_text(json.dumps(contract), encoding='utf-8')
            result = worker.run(Path(d), contract, None, float('inf'))
        self.assertEqual(result['status'], 'failed_or_interrupted_design_gate')
        self.assertIn('Geometry constants differ', captured['report.json']['error']['message'])
        self.assertEqual(captured['report.json']['ledger']['transition_steps_attempted'], 0)


if __name__ == '__main__':
    unittest.main()
