"""Unknown recorded next actions are placeholders, not usable action labels."""
import unittest

import numpy as np

from thrusterchain.training_protocol import TransitionBatch


class PlaceholderTests(unittest.TestCase):
    def fields(self):
        return dict(observations=np.zeros((3, 2)), actions=np.zeros((3, 2)),
                    rewards=np.zeros(3), next_observations=np.ones((3, 2)),
                    terminated=np.array([False, False, True]),
                    truncated=np.array([False, True, False]),
                    bootstrap_mask=np.array([1., 1., 0.]),
                    origin=np.zeros(3, np.uint8), group_ids=np.array([0, 0, 1]),
                    next_actions=np.array([[.2, -.1], [0., 0.], [0., 0.]]),
                    next_action_valid=np.array([True, False, False]))

    def test_invalid_nonzero_placeholder_is_rejected_independent_of_bootstrap(self):
        fields = self.fields()
        batch = TransitionBatch(**fields)
        np.testing.assert_array_equal(batch.bootstrap_mask, [1., 1., 0.])
        np.testing.assert_array_equal(batch.next_action_valid, [True, False, False])
        for row in (1, 2):
            with self.subTest(row=row):
                altered = fields["next_actions"].copy()
                altered[row, 0] = .3
                with self.assertRaisesRegex(ValueError, "zero placeholders"):
                    TransitionBatch(**{**fields, "next_actions": altered})

    def test_default_missing_and_selection_preserve_validity(self):
        fields = self.fields()
        del fields["next_actions"], fields["next_action_valid"]
        batch = TransitionBatch(**fields)
        self.assertFalse(batch.next_action_valid.any())
        self.assertFalse(batch.next_actions.any())
        selected = batch.select([1, 0, 1])
        np.testing.assert_array_equal(selected.bootstrap_mask, [1., 1., 1.])
        self.assertFalse(selected.next_action_valid.any())


if __name__ == "__main__":
    unittest.main()
