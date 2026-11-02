"""Pure input, authorization and deadline fixtures; no model or production work."""
import copy
from datetime import datetime
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import authorization
import recovery_inputs as r
from common import digest, read

class InputsTests(unittest.TestCase):
    def test_deadline_fixed_est_and_four_hours(self):
        end=datetime.fromisoformat(r.recovery()['original_deadline_est']).timestamp()
        self.assertEqual(r.recovery_seconds(end-20000),14400)
        self.assertEqual(r.recovery_seconds(end-100),100)
        with self.assertRaisesRegex(ValueError,'expired'):r.recovery_seconds(end)
        with self.assertRaisesRegex(ValueError,'expired'):r.recovery_seconds(end+1)

    def test_original_consumed_allowance_cannot_be_reset(self):
        original=r.recovery()
        original['remaining_consumed_time_allowance_seconds']=30
        end=datetime.fromisoformat(original['original_deadline_est']).timestamp()
        with patch.object(r,'recovery',return_value=original):
            self.assertEqual(r.recovery_seconds(end-20000),30)

    def test_changed_criteria_rejected(self):
        original=r.recovery()
        for key,value in [('maximum_recovery_seconds',14401),('resume_checkpoint_step',15000),
                          ('prior_execution_counts_known',True),('new_training_jobs',[])]:
            modified=copy.deepcopy(original);modified[key]=value
            with patch.object(r,'read',return_value=modified),self.assertRaises(ValueError):r.recovery()

    def test_authenticated_bytes_reject_corruption_and_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);(folder/'saved').write_bytes(b'known')
            config={'prior_files_sha256':{'saved':digest(b'known')}}
            with patch.object(r,'prior_root',return_value=folder),patch.object(r,'recovery',return_value=config):
                self.assertEqual(r.authenticate_prior_file('saved'),b'known')
                with self.assertRaisesRegex(ValueError,'Unpinned'):r.authenticate_prior_file('unknown')
                with self.assertRaisesRegex(ValueError,'escaped'):r.authenticate_prior_file('../escape')
                (folder/'saved').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'changed'):r.authenticate_prior_file('saved')

    def test_completed_and_pending_job_mapping(self):
        self.assertEqual(r.job_directory('seed0_full','bca'),r.prior_root()/'run/seed0_full/bca')
        self.assertEqual(r.job_directory('seed2_hole','bca'),r.HERE/'run/seed2_hole/bca')
        self.assertEqual(r.job_directory('seed2_random','warmup'),r.prior_root()/'run/seed2_random/warmup')
        with self.assertRaises(ValueError):r.job_directory('../invalid','bca')

    def test_real_readonly_approval_and_prior_terminal_identity(self):
        value=authorization.verify(r.HERE,read(r.HERE/'protocol.json'))
        self.assertTrue(value['execution_authorized'])
        self.assertFalse(value['prior_counts_known'])
        self.assertEqual(value['new_work_caps']['critic'],34000)

if __name__=='__main__':unittest.main()
