import unittest
from pathlib import Path
from thrusterchain.inference_guard import InferenceGuard

class GuardTests(unittest.TestCase):
    def test_stale_aliases_and_model_refitting_refused(self):
        from thrusterchain.regular import train_behavior_cloning
        from thrusterchain.datasets import generate_dataset, DatasetError
        from thrusterchain.planning import identify_from_training
        with InferenceGuard() as guard:
            for operation in (lambda:train_behavior_cloning(None),generate_dataset,
                              lambda:identify_from_training(None)):
                with self.assertRaises(DatasetError): operation()
        self.assertEqual(guard.ledger["forbidden_api_calls"],3)

    def test_training_collection_and_writes_refused(self):
        from thrusterchain import regular,datasets
        with InferenceGuard() as guard:
            with self.assertRaises(RuntimeError): regular.train_behavior_cloning(None)
            with self.assertRaises(RuntimeError): datasets.generate_dataset()
            with self.assertRaises(RuntimeError): Path("must_not_be_created.txt").write_text("no")
        self.assertEqual(guard.events,dict(training_calls=1,dataset_collection_calls=1,write_attempts=1))
        self.assertFalse(Path("must_not_be_created.txt").exists())

if __name__=="__main__": unittest.main()

