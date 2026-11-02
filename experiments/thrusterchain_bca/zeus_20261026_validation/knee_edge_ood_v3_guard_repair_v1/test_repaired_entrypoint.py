"""CPU startup/adapter fixtures; no production workflow or simulator calls."""
from pathlib import Path
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import types

import numpy as np
import launch_repaired as L


class EntryPoint(unittest.TestCase):
    def cli(self, *args, cwd=None):
        return subprocess.run([sys.executable, "-B", str(L.HERE / "launch_repaired.py"), *map(str,args)],
                              cwd=cwd, capture_output=True, text=True, timeout=30)

    def test_check_from_unrelated_working_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=self.cli("check",cwd=tmp)
        self.assertEqual(p.returncode,0,p.stderr)
        data=json.loads(p.stdout)
        self.assertFalse(data["scientific_execution_enabled"])
        self.assertEqual(Path(data["loaded_modules"]["stage1_worker"]).parent,L.HERE)
        self.assertEqual(Path(data["loaded_modules"]["retained_support"]).parent,L.CANDIDATE)

    def test_changed_source_refused(self):
        real=L.sha
        def changed(path):
            return real(path) if Path(path).name=='SOURCE_SNAPSHOT.json' else '0'*64
        with patch.object(L,"sha",side_effect=changed):
            with self.assertRaisesRegex(ValueError,"Changed or missing"):
                L.verify_package()

    def test_changed_snapshot_lineage_refused(self):
        real=L.sha
        def changed(path):
            return '0'*64 if Path(path).name=='SOURCE_SNAPSHOT.json' else real(path)
        with patch.object(L,"sha",side_effect=changed):
            with self.assertRaisesRegex(ValueError,"snapshot lineage"):
                L.verify_package()

    def test_added_importable_package_is_detected_and_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'step04_math';p.mkdir();(p/'__init__.py').write_text('raise RuntimeError("must never import")')
            detected=L.importable_child_packages((Path(tmp),))
            self.assertEqual(detected,[str(p)])
            with patch.object(L,'importable_child_packages',return_value=detected):
                with self.assertRaisesRegex(ValueError,'Unlisted importable'):
                    L.verify_package()

    def test_wrong_module_binding_refused(self):
        with patch.dict(sys.modules,{"units":types.SimpleNamespace(__file__="wrong.py")}):
            with self.assertRaisesRegex(ValueError,"Wrong imported"):
                L.local_module("units")

    def test_no_execute_command(self):
        p=self.cli("execute")
        self.assertEqual(p.returncode,2)
        self.assertIn("invalid choice",p.stderr)

    def test_known_saved_analysis_startup(self):
        p=self.cli("analyse-known-v2")
        self.assertEqual(p.returncode,0,p.stderr)
        data=json.loads(p.stdout)
        self.assertEqual(data["accounting"]["corrected_total"],10994000)
        self.assertEqual(data["amended_decision"],"INCONCLUSIVE")

    def test_spend_command_preserves_unknown_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            unit=Path(tmp)/"missing_receipt";unit.mkdir()
            (unit/'job.json').write_text(json.dumps({'timeout_seconds':300}))
            p=self.cli("spend",tmp)
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertIsNone(json.loads(p.stdout)["unknown_seconds_upper_bound"])

    def bundle(self,tmp,roles=None):
        n=70; times=np.repeat([0,25,50],n)
        obs=np.random.default_rng(777).normal(size=(3*n,14)).astype(np.float32)
        obs[:,13]=(100-times)/100
        actions=np.zeros((3*n,4),np.float32); actions[:,0]=-.7;actions[-10:,0]=-.9
        f=Path(tmp)/'arrays.npz'
        np.savez(f,obs=obs,actions=actions,episode_ids=np.tile(np.arange(n),3),times=times,
                 deep_episode=np.ones(3*n,bool),row_roles=np.zeros(3*n,int) if roles is None else roles,
                 start_obs=np.repeat(obs[:1],128,axis=0),
                 anchor_obs=np.repeat(obs[[0,n,2*n]],16,axis=0),anchor_times=np.repeat([0,25,50],16))
        metadata={'schema':'retained_support_fitting_export_v1','data_kind':'synthetic_fixture','world':'stall3',
                  'arrays_file':f.name,'arrays_sha256':L.sha(f)}
        p=Path(tmp)/'bundle.json';p.write_text(json.dumps(metadata))
        return p

    def test_support_cli_runs_retained_only_without_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=self.cli('support',self.bundle(tmp))
        self.assertEqual(p.returncode,0,p.stderr)
        data=json.loads(p.stdout)
        self.assertEqual(data['data_kind'],'synthetic_fixture')
        self.assertFalse(data['formal_admission'])
        self.assertEqual(data['result']['status'],'INCONCLUSIVE')
        self.assertEqual(data['result']['retained_deep_reference_groups']['50']['rows'],60)

    def test_support_rejects_changed_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle=self.bundle(tmp)
            (Path(tmp)/'arrays.npz').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                L.support_bundle(bundle)

    def test_support_rejects_nonfitting_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle=self.bundle(tmp,np.ones(210,int))
            with self.assertRaisesRegex(ValueError,'Only fitting'):
                L.support_bundle(bundle)

    def test_test_driver_uses_separate_processes_and_no_live_execute(self):
        calls=[]
        def fake(command,**kw):
            calls.append((command,kw));return types.SimpleNamespace(stdout='',stderr='',returncode=0)
        with patch.object(L.subprocess,'run',side_effect=fake):
            result=L.tests()
        self.assertEqual(len(calls),2)
        self.assertEqual({c[1]['cwd'] for c in calls},{L.HERE,L.CANDIDATE})
        self.assertEqual(result['simulator_steps'],0)
        self.assertTrue(all('execute' not in c[0] for c in calls))


if __name__ == '__main__':
    unittest.main()
