"""CPU-only worker boundary fixtures. No model, dataset or learner execution."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import launch
import train_worker as worker


class WorkerTests(unittest.TestCase):
    def protocol(self):
        return worker.read(worker.HERE/'protocol.json')

    def job(self, kind='warmup'):
        return {'kind':kind,'cell':'seed0_full','condition':'full','seed_index':0,
                **({'arm':'bca'} if kind == 'branch' else {})}

    def test_exact_job_budgets_and_rejected_expansion(self):
        p = self.protocol()
        for job in launch.job_specs(p)[:-1]:
            self.assertEqual(worker.validate_job(p,job),10000)
        for key,value in [('branch_critic_updates',10001),('checkpoint_every',500),
                          ('calibration_boundaries',[10000,16000]),('bca_kappa',2),('resume',True)]:
            q=copy.deepcopy(p);q[key]=value
            with self.assertRaises(ValueError):worker.validate_job(q,self.job())
        for key,value in [('condition','restricted'),('seed_index',3),('seed_index',True),('cell','seed1_full')]:
            j=self.job();j[key]=value
            with self.assertRaises(ValueError):worker.validate_job(p,j)

    def test_direct_worker_cannot_bypass_review_hold(self):
        with patch.object(worker,'module') as loader:
            with self.assertRaisesRegex(ValueError,'disabled'):
                worker.run({'execution_authorized':True,'execution_status':'approved_for_one_attempt'},
                           Path('not_created'),self.job())
            loader.assert_not_called()

    def test_checkpoint_is_exclusive_and_hashes_full_state(self):
        with tempfile.TemporaryDirectory(dir=worker.HERE,prefix='fixture_') as d:
            root=Path(d)
            kwargs=dict(counts={'critic':15000,'actor':7500},job=self.job('branch'),
                        protocol_sha='synthetic',normalization={'same':True},matching_receipt_sha256='match')
            seal=worker.save_checkpoint(root,b'full optimizer target rng state',b'actor',**kwargs)
            path=root/seal['path'];manifest=json.loads(path.read_text())
            self.assertEqual(worker.sha(path),seal['sha256'])
            self.assertEqual(worker.sha(path.parent/'full_state.msgpack'),manifest['artifact']['sha256'])
            self.assertEqual(manifest['absolute_critic_step'],15000)
            self.assertTrue(manifest['complete_optimizer_target_rng_state'])
            self.assertFalse(manifest['resume_authorized'])
            self.assertEqual(manifest['matching_receipt_sha256'],'match')
            original=path.read_bytes()
            with self.assertRaises(FileExistsError):
                worker.save_checkpoint(root,b'different',b'actor',**kwargs)
            self.assertEqual(path.read_bytes(),original)

    def test_checkpoint_rejects_invalid_boundary(self):
        with tempfile.TemporaryDirectory(dir=worker.HERE,prefix='fixture_') as d:
            for count in (-1000,1,999,20001,True):
                with self.assertRaises(ValueError):
                    worker.save_checkpoint(Path(d),b'none',None,counts={'critic':count},
                        job=self.job(),protocol_sha='synthetic',normalization={})
            self.assertFalse((Path(d)/'checkpoints').exists())

    def test_partial_checkpoint_is_preserved_not_resumed(self):
        with tempfile.TemporaryDirectory(dir=worker.HERE,prefix='fixture_') as d:
            root=Path(d)
            folder=root/'checkpoints/step_01000';folder.mkdir(parents=True)
            (folder/'full_state.msgpack').write_bytes(b'partial evidence')
            with self.assertRaises(FileExistsError):
                worker.save_checkpoint(root,b'replacement',None,counts={'critic':1000},
                    job=self.job(),protocol_sha='synthetic',normalization={})
            self.assertEqual((folder/'full_state.msgpack').read_bytes(),b'partial evidence')
            self.assertFalse((folder/'manifest.json').exists())

    def test_wrong_cell_warmup_path_rejected_before_reads(self):
        with patch.object(worker,'read') as read:
            with self.assertRaisesRegex(ValueError,'outside declared cell'):
                worker.verify_warmup(worker.HERE/'run/seed1_full/warmup','synthetic',self.job('branch'))
            read.assert_not_called()


if __name__=='__main__':unittest.main()
