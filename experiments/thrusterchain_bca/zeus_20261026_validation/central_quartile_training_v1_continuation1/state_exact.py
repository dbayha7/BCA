"""Exact comparison of Flax's decoded checkpoint schema, not map wire order.

Flax restores mappings in template order; JAX may reconstruct them in sorted
order. Map ordering is not a state value. All other decoded distinctions remain
strict: key sets, container kinds, array/scalar kinds, shape, dtype and bits.

JAX arrays and NumPy ndarrays share Flax's ndarray wire representation. NumPy
scalars have a different wire extension from zero-dimensional arrays and Python
scalars; they are not interchangeable here. Flax converts registered objects
and sequences into its state-dict schema. Static apply functions, class types
and optimizer definitions are authenticated through the caller's template and
source manifest, not encoded in the checkpoint. File hashes remain separate.
"""
import hashlib
import struct

import jax
import numpy as np
from flax import serialization


def _kind(value):
    if type(value) is dict:
        return 'dict'
    if type(value) is list:
        return 'list'
    if type(value) is tuple:
        return 'tuple'
    if isinstance(value, (np.ndarray, jax.Array)):
        return 'array'
    if isinstance(value, np.generic):
        return 'numpy_scalar'
    if type(value) in (type(None), bool, int, float, complex, str, bytes):
        return type(value).__name__
    return 'unsupported:' + type(value).__module__ + '.' + type(value).__qualname__


def _key_identity(key):
    # Flax normally makes state-dict keys strings. Typed identities also keep
    # bool/int and bytes/str differences strict in direct decoded fixtures.
    if type(key) not in (str, bytes, int, bool):
        raise ValueError('Unsupported checkpoint mapping key type: ' + type(key).__name__)
    return (type(key).__name__, key)


def assert_exact_decoded(expected, actual, *, label='Checkpoint state'):
    """Compare decoded trees without tolerances; return only small diagnostics."""
    counts = dict(containers=0, leaves=0, arrays=0, numpy_scalars=0, python_scalars=0)

    def fail(path, reason):
        raise ValueError(label + ' differs at ' + path + ': ' + reason)

    def visit(left, right, path):
        left_kind, right_kind = _kind(left), _kind(right)
        if left_kind != right_kind:
            fail(path, 'kind ' + left_kind + ' != ' + right_kind)
        if left_kind.startswith('unsupported:'):
            fail(path, left_kind)
        if left_kind == 'dict':
            counts['containers'] += 1
            left_keys = {_key_identity(key): key for key in left}
            right_keys = {_key_identity(key): key for key in right}
            if left_keys.keys() != right_keys.keys():
                missing = sorted((repr(k) for k in left_keys.keys() - right_keys.keys()))
                extra = sorted((repr(k) for k in right_keys.keys() - left_keys.keys()))
                fail(path, 'mapping keys differ; missing=' + repr(missing) + '; extra=' + repr(extra))
            for identity in sorted(left_keys, key=repr):
                key = left_keys[identity]
                visit(left[key], right[right_keys[identity]], path + '[' + repr(key) + ']')
            return
        if left_kind in ('list', 'tuple'):
            counts['containers'] += 1
            if len(left) != len(right):
                fail(path, 'sequence length ' + str(len(left)) + ' != ' + str(len(right)))
            for index, (x, y) in enumerate(zip(left, right)):
                visit(x, y, path + '[' + str(index) + ']')
            return
        counts['leaves'] += 1
        if left_kind in ('array', 'numpy_scalar'):
            counts['arrays' if left_kind == 'array' else 'numpy_scalars'] += 1
            x, y = np.asarray(left), np.asarray(right)
            if x.dtype.hasobject or y.dtype.hasobject or x.dtype.fields or y.dtype.fields:
                fail(path, 'object/structured dtype is not supported checkpoint state')
            if x.shape != y.shape:
                fail(path, 'shape ' + repr(x.shape) + ' != ' + repr(y.shape))
            # dtype.name distinguishes custom numeric dtypes (e.g. bfloat16)
            # that can share a void dtype.str. dtype.str preserves byte order.
            if x.dtype != y.dtype or x.dtype.str != y.dtype.str or x.dtype.name != y.dtype.name:
                fail(path, 'dtype ' + repr(x.dtype) + ' != ' + repr(y.dtype))
            if x.dtype.metadata != y.dtype.metadata:
                fail(path, 'dtype metadata differs')
            if x.tobytes(order='C') != y.tobytes(order='C'):
                fail(path, 'leaf bytes differ')
            return
        counts['python_scalars'] += 1
        if left_kind == 'float':
            equal = struct.pack('!d', left) == struct.pack('!d', right)
        elif left_kind == 'complex':
            equal = struct.pack('!dd', left.real, left.imag) == struct.pack('!dd', right.real, right.imag)
        else:
            equal = left == right
        if not equal:
            fail(path, 'scalar value/bits differ')

    visit(expected, actual, '$')
    return dict(comparison='exact_decoded_state', mapping_order_ignored=True,
                numeric_tolerances_used=False, **counts)


def assert_exact_state(raw, restored, *, label='Checkpoint state'):
    """Match every decoded saved field to a restored Flax state, bit exactly.

    Authenticate ``raw`` against its file manifest and the template's static
    definitions before this call. This function does not replace those checks.
    """
    if type(raw) is not bytes:
        raise ValueError(label + ': checkpoint input must be bytes')
    try:
        expected = serialization.msgpack_restore(raw)
    except Exception as exc:
        raise ValueError(label + ': checkpoint decoding failed: ' + type(exc).__name__) from exc
    actual = serialization.to_state_dict(restored)
    receipt = assert_exact_decoded(expected, actual, label=label)
    receipt.update(checkpoint_bytes=len(raw), checkpoint_sha256=hashlib.sha256(raw).hexdigest())
    return receipt
