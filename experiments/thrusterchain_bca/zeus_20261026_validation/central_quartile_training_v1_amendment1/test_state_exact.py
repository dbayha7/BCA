"""Serialization-only CPU fixtures; no learners, real data or simulator."""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import copy
import struct
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from state_exact import assert_exact_decoded, assert_exact_state


def state_fixture():
    return {
        'weights': {'kernel': np.array([[1., -0.], [2., 3.]], np.float32),
                    'bias': np.array([0., 1.], np.float32)},
        'optimizer': {'momentum': np.array([.25, -.5], np.float32), 'count': np.int32(7)},
        'targets': {'kernel': np.array([1., 2.], np.float32)},
        'rng': np.array([123, 456], np.uint32),
        'metadata': {'phase': 1, 'ready': True, 'tag': 'fixture', 'empty': {}, 'none': None},
    }


def reverse_maps(node):
    if type(node) is dict:
        return {key: reverse_maps(node[key]) for key in reversed(node)}
    return node


class ExactStateTests(unittest.TestCase):
    def test_nested_key_order_ignored_but_every_leaf_counted(self):
        expected = state_fixture()
        restored = reverse_maps(expected)
        raw = serialization.to_bytes(expected)
        self.assertNotEqual(raw, serialization.to_bytes(restored))
        result = assert_exact_state(raw, restored)
        self.assertTrue(result['mapping_order_ignored'])
        self.assertFalse(result['numeric_tolerances_used'])
        self.assertEqual(result['leaves'], 10)
        self.assertEqual(result['arrays'], 5)
        self.assertEqual(result['numpy_scalars'], 1)
        self.assertEqual(result['checkpoint_bytes'], len(raw))

    def test_actual_jit_reconstruction_order_is_not_corruption(self):
        template = {'kernel': np.array([1., 2.], np.float32), 'bias': np.array([0.], np.float32)}
        transformed = jax.jit(lambda x: x)(template)
        raw = serialization.to_bytes(transformed)
        restored = serialization.from_bytes(template, raw)
        self.assertEqual(list(template), ['kernel', 'bias'])
        self.assertEqual(list(transformed), ['bias', 'kernel'])
        self.assertNotEqual(raw, serialization.to_bytes(restored))
        self.assertEqual(assert_exact_state(raw, restored)['arrays'], 2)

    def test_jax_and_numpy_arrays_share_wire_kind(self):
        value = {'array': np.array([1., 2.], np.float32)}
        self.assertEqual(assert_exact_state(serialization.to_bytes(value),
            {'array': jnp.asarray(value['array'])})['arrays'], 1)

    def test_weights_optimizer_target_and_rng_mutations_rejected(self):
        original = state_fixture()
        raw = serialization.to_bytes(original)
        for path in [('weights', 'kernel'), ('optimizer', 'momentum'), ('targets', 'kernel'), ('rng',)]:
            with self.subTest(path=path):
                changed = copy.deepcopy(original)
                array = changed
                for key in path:
                    array = array[key]
                array.flat[0] += 1
                with self.assertRaisesRegex(ValueError, 'leaf bytes differ'):
                    assert_exact_state(raw, changed)

    def test_exact_float_bit_change_is_rejected(self):
        left = np.array([1.], np.float32)
        right = np.nextafter(left, np.float32(2.))
        with self.assertRaisesRegex(ValueError, 'leaf bytes differ'):
            assert_exact_state(serialization.to_bytes({'x': left}), {'x': right})

    def test_shape_and_dtype_changes_rejected(self):
        raw = serialization.to_bytes({'x': np.array([1., 2.], np.float32)})
        for value, reason in [(np.array([[1., 2.]], np.float32), 'shape'),
                              (np.array([1., 2.], np.float64), 'dtype')]:
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                assert_exact_state(raw, {'x': value})

    def test_numpy_scalar_array_and_python_scalar_are_distinct(self):
        values = [np.float32(1.), np.array(1., np.float32), 1., 1, True]
        for i, left in enumerate(values):
            for j, right in enumerate(values):
                with self.subTest(left=i, right=j):
                    raw = serialization.to_bytes({'x': left})
                    if i == j:
                        assert_exact_state(raw, {'x': right})
                    else:
                        with self.assertRaisesRegex(ValueError, 'kind'):
                            assert_exact_state(raw, {'x': right})

    def test_missing_extra_and_empty_mapping_changes_rejected(self):
        raw = serialization.to_bytes({'a': np.int32(2), 'empty': {}})
        for changed in ({'a': np.int32(2)}, {'a': np.int32(2), 'empty': {}, 'extra': 3},
                        {'a': np.int32(2), 'empty': {'unexpected': {}}}):
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, 'mapping keys'):
                assert_exact_state(raw, changed)

    def test_flax_ignored_extra_checkpoint_field_is_detected(self):
        raw = serialization.to_bytes({'known': np.int32(1), 'extra': np.int32(2)})
        restored = serialization.from_bytes({'known': np.int32(0)}, raw)
        self.assertEqual(set(restored), {'known'})
        with self.assertRaisesRegex(ValueError, 'mapping keys'):
            assert_exact_state(raw, restored)

    def test_decoded_container_kinds_and_lengths_are_strict(self):
        for left, right in [([], {}), ((), []), ([1], []), ([1, 2], [2, 1])]:
            with self.subTest(left=left, right=right), self.assertRaises(ValueError):
                assert_exact_decoded(left, right)
        assert_exact_decoded([], [])
        assert_exact_decoded((), ())

    def test_typed_keys_are_not_coerced(self):
        with self.assertRaisesRegex(ValueError, 'mapping keys'):
            assert_exact_decoded({1: 'value'}, {True: 'value'})
        with self.assertRaisesRegex(ValueError, 'mapping keys'):
            assert_exact_decoded({'name': 1}, {b'name': 1})

    def test_array_signed_zero_and_nan_payload_bits(self):
        values = np.array([0x00000000, 0x80000000, 0x7f800000, 0xff800000, 0x7fc00011],
                          dtype=np.uint32).view(np.float32)
        raw = serialization.to_bytes({'x': values})
        assert_exact_state(raw, {'x': values.copy()})
        for index, bits in [(0, 0x80000000), (4, 0x7fc00012)]:
            changed = values.copy()
            changed.view(np.uint32)[index] = bits
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, 'leaf bytes differ'):
                assert_exact_state(raw, {'x': changed})

    def test_python_float_signed_zero_and_nan_payload_bits(self):
        nan1 = struct.unpack('!d', bytes.fromhex('7ff8000000000011'))[0]
        nan2 = struct.unpack('!d', bytes.fromhex('7ff8000000000012'))[0]
        for value in [float('inf'), float('-inf'), nan1, -0.]:
            assert_exact_state(serialization.to_bytes({'x': value}), {'x': value})
        for left, right in [(0., -0.), (nan1, nan2)]:
            with self.subTest(left=left), self.assertRaisesRegex(ValueError, 'scalar value/bits'):
                assert_exact_state(serialization.to_bytes({'x': left}), {'x': right})

    def test_metadata_values_and_kinds_are_checked(self):
        for left, right in [('tag', b'tag'), (True, 1), (None, 0), (7, 8), ('a', 'b')]:
            with self.subTest(left=left), self.assertRaises(ValueError):
                assert_exact_state(serialization.to_bytes({'metadata': left}), {'metadata': right})

    def test_no_state_mutation_and_path_diagnostic(self):
        original = state_fixture()
        raw = serialization.to_bytes(original)
        before = serialization.to_bytes(original)
        assert_exact_state(raw, original)
        self.assertEqual(before, serialization.to_bytes(original))
        changed = copy.deepcopy(original)
        changed['optimizer']['count'] = np.int32(8)
        with self.assertRaisesRegex(ValueError, r"\$\['optimizer'\]\['count'\]"):
            assert_exact_state(raw, changed, label='Full-state restore')

    def test_unsupported_or_malformed_state_refused(self):
        with self.assertRaisesRegex(ValueError, 'checkpoint input must be bytes'):
            assert_exact_state('not bytes', {})
        with self.assertRaisesRegex(ValueError, 'decoding failed'):
            assert_exact_state(b'\xc1', {})
        with self.assertRaisesRegex(ValueError, 'unsupported'):
            assert_exact_decoded(object(), object())
        with self.assertRaisesRegex(ValueError, 'object/structured'):
            assert_exact_decoded(np.array([object()], object), np.array([object()], object))


if __name__ == '__main__':
    unittest.main()
