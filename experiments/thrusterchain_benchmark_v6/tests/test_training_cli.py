"""Actual staged CLI checks with small budgets and durable output boundaries."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from thrusterchain.datasets import generate_dataset

ROOT = Path(__file__).resolve().parents[1]


class TrainingCLITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scratch=ROOT/".verification"/"training_cli"
        scratch.mkdir(parents=True,exist_ok=True)
        cls.tmp=tempfile.TemporaryDirectory(dir=scratch)
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.path=Path(cls.tmp.name)
        cls.dataset=cls.path/"fixed.npz"
        generate_dataset(cls.dataset,episodes=10,seed=512)

    def command(self,*args,ok=True):
        result=subprocess.run([sys.executable,"-B","-m","thrusterchain",*map(str,args)],
                              cwd=ROOT,text=True,capture_output=True)
        if ok:
            self.assertEqual(result.returncode,0,result.stderr)
            return json.loads(result.stdout)
        self.assertNotEqual(result.returncode,0)
        return result.stderr

    def test_closed_form_fit_does_not_execute_default_million_updates(self):
        target=self.path/"closed_form"
        result=self.command("train",self.dataset,target)
        self.assertEqual(result["default_offline_optimizer_budget"],1000000)
        self.assertEqual(result["budget_applicability"],"not_applicable")
        self.assertEqual(result["executed_optimizer_updates"],0)
        self.assertEqual(result["candidate_fits"],4)
        out=self.path/"closed_strict.json"
        result=self.command("evaluate",target,out,"--episodes",1)
        self.assertTrue(result["frozen_policy"])
        self.assertEqual(result["mode"],"strict_offline")
        self.assertEqual(result["online_optimizer_updates"],0)
        self.assertIn("does not support",self.command("evaluate",target,self.path/"closed_adapt.json",
            "--no-frozen-policy","--online-interactions",1,"--online-updates",1,
            "--offline-fraction",0.5,"--learning-settings","{}",ok=False))
        self.assertFalse((self.path/"closed_adapt.json").exists())

    def test_closed_form_rejects_explicit_optimizer_budget(self):
        target=self.path/"unsupported"
        self.assertIn("closed-form BC has no optimizer",self.command("train",self.dataset,target,
                      "--offline-updates",5,ok=False))
        self.assertFalse(target.exists())

    def test_cli_integer_budget_validation(self):
        for value in ("true","1.0","3000001","-1"):
            target=self.path/("invalid_"+value)
            self.command("train",self.dataset,target,"--adapter","affine-sgd",
                         "--offline-updates",value,ok=False)
            self.assertFalse(target.exists())

    def test_invalid_output_parent_fails_before_training_or_evaluation(self):
        from unittest.mock import patch
        from argparse import Namespace
        from thrusterchain.__main__ import train_stage, evaluation_stage
        blocked=self.path/"regular_file_parent"
        blocked.write_text("a file, not a directory",encoding="utf8")
        training=Namespace(frozen_directory=blocked/"child",offline_updates=None,batch_size=256,
            adapter="affine-sgd",resume_checkpoint=None,learning_settings=None,
            dataset=self.dataset,seed=0)
        evaluating=Namespace(output=blocked/"result.json",frozen_policy=True,
            online_interactions=None,online_updates=None,offline_fraction=None,learning_settings=None,
            learning=None,batch_size=256,episodes=1,seed=0,frozen_directory=self.path/"missing",
            dataset=self.dataset)
        with patch("thrusterchain.training_runner.train_offline",side_effect=AssertionError("training called")), \
             patch("thrusterchain.training_runner.run_mode",side_effect=AssertionError("evaluation called")):
            with self.assertRaises(NotADirectoryError): train_stage(training)
            with self.assertRaises(NotADirectoryError): evaluation_stage(evaluating)

    def test_iterative_training_strict_and_adaptive_stages(self):
        from thrusterchain.training_protocol import LearnerCheckpoint
        target=self.path/"iterative"
        self.command("train",self.dataset,target,"--adapter","affine-sgd",
                     "--offline-updates",3,"--batch-size",8)
        cp=LearnerCheckpoint.load(target/"checkpoint.json")
        self.assertEqual(cp.optimizer_steps,3)
        strict=self.command("evaluate",target,self.path/"sgd_strict.json","--dataset",self.dataset,
                            "--episodes",1)
        self.assertIn("frozen_evaluation",strict)
        self.assertEqual(LearnerCheckpoint.load(target/"checkpoint.json").digest,cp.digest)
        adapt=self.command("evaluate",target,self.path/"sgd_adapt.json","--dataset",self.dataset,
            "--no-frozen-policy","--online-interactions",3,"--online-updates",2,
            "--offline-fraction",0.5,"--batch-size",8,"--learning-settings","{}","--episodes",1)
        self.assertEqual(set(adapt["raw_adaptation"]),set(adapt["frozen_evaluation"]))
        for entry in adapt["checkpoints"].values():
            self.assertEqual(LearnerCheckpoint.load(entry["path"]).optimizer_steps,5)
        self.assertEqual(LearnerCheckpoint.load(target/"checkpoint.json").digest,cp.digest)

    def test_toggle_off_requires_declared_budgets_and_strict_refuses_them(self):
        for options in (("--no-frozen-policy",),("--online-updates","1"),
            ("--no-frozen-policy","--online-interactions","0","--online-updates","0",
             "--offline-fraction","0.5","--learning-settings","{}","--learning","disabled")):
            self.command("evaluate",self.path/"nonexistent",self.path/"no_output.json",*options,ok=False)
            self.assertFalse((self.path/"no_output.json").exists())


if __name__=="__main__":
    unittest.main()
