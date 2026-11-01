"""Bounded checks for paired randomness, scores and optimizer scope."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import paired_refits as p


class PairedRefitTests(unittest.TestCase):
    def test_same_inputs_and_only_auxiliary_counter_changes(self):
        calls = []
        actor, critic, target = object(), object(), object()
        frozen = (actor, critic, target)
        def fake_fit(state, batch, key):
            calls.append((batch, key.copy(), frozen))
            return state + 1, {'scale_fit_accepted': True}
        pair = {arm: {'fit': fake_fit, 'aux': 0} for arm in p.ARMS}
        ledger = {'auxiliary_updates': 0, 'actor_updates': 0, 'critic_updates': 0,
                  'actor_target_updates': 0, 'critic_target_updates': 0}
        traces = {arm: {'minibatches': hashlib.sha256(), 'fit_keys': hashlib.sha256()}
                  for arm in p.ARMS}
        generator = np.random.Generator(np.random.PCG64(p.model_seed(0, 0)))
        for step in range(3):
            ix = generator.integers(0, 50, p.BATCH)
            key = np.asarray([step, step + 17], np.uint32)
            batch = object()
            p.paired_step(pair, batch, key, ix, ledger, traces)
            self.assertIs(calls[-1][0], calls[-2][0])
            np.testing.assert_array_equal(calls[-1][1], calls[-2][1])
            self.assertEqual(calls[-1][2], frozen)
        for k in ('minibatches', 'fit_keys'):
            self.assertEqual(traces['native'][k].hexdigest(), traces['bca'][k].hexdigest())
        self.assertEqual(ledger, {'auxiliary_updates': 6, 'actor_updates': 0, 'critic_updates': 0,
                                  'actor_target_updates': 0, 'critic_target_updates': 0})

    def test_bank_summary_tracks_anchored_contrasts_and_choice(self):
        widths = np.zeros((4, 2, 32, 15), np.float32)
        widths[:, 1, :, 1] = 2.
        qn = np.zeros((32, 15)); qb = np.zeros((32, 15))
        qn[:, 1] = qb[:, 1] = 1.
        sources = np.array(['random_prefix'] * 16 + ['native_visited'] * 8 + ['bca_visited'] * 8)
        out = p.contrast_summary(widths, qn, qb, sources)
        self.assertEqual(out['actor_visited']['states'], 16)
        self.assertEqual(out['actor_visited']['winner_disagreements_by_refit'], [16] * 4)
        self.assertAlmostEqual(out['actor_visited']['rms_paired_mean_contrast'], np.sqrt(36 / 15))
        self.assertEqual(out['actor_visited']['rms_across_refit_sd'], 0.)

    def test_seed_ranges_are_pair_unique_and_calibration_is_separate(self):
        fit = [p.model_seed(s, r) for s in p.SEEDS for r in range(p.REPLICATES)]
        cal = [p.calibration_seed(s, r) for s in p.SEEDS for r in range(p.REPLICATES)]
        self.assertEqual(len(set(fit)), 12)
        self.assertEqual(len(set(cal)), 12)
        self.assertFalse(set(fit) & set(cal))

    def test_seal_rejects_source_and_prior_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in p.SEALED_SOURCES:
                (root / name).write_text('fixed ' + name, encoding='utf8')
            with mock.patch.object(p, 'HERE', root), \
                    mock.patch.object(p, 'SEAL_FILE', root / 'seal.json'), \
                    mock.patch.object(p, 'input_pins', return_value={'v4': 'original'}) as pins:
                sealed = p.seal()
                self.assertEqual(p.check_seal(), sealed)
                (root / 'PROTOCOL.md').write_text('changed', encoding='utf8')
                with self.assertRaisesRegex(ValueError, 'source differs'):
                    p.check_seal()
                (root / 'PROTOCOL.md').write_text('fixed PROTOCOL.md', encoding='utf8')
                pins.return_value = {'v4': 'different'}
                with self.assertRaisesRegex(ValueError, 'input pins differ'):
                    p.check_seal()


if __name__ == '__main__':
    unittest.main()
