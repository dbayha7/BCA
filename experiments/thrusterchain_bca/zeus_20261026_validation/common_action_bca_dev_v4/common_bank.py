"""Q-pi-blind common action candidates for paired frozen critics.

For each state, this creates ONE complete bank shared by both critics and both
frozen-policy continuations. No reference value, critic score, calibration, or
reward enters this construction. Clipped duplicate vectors are retained.
"""
import numpy as np


DELTA = np.float32(.2)
FIXED_A0 = (-.5, -.25, 0., .25, .5)
SLOT_NAMES = ('native_actor', 'bca_actor',
              'midpoint_a0_minus', 'midpoint_a0_plus',
              'midpoint_a1_minus', 'midpoint_a1_plus',
              'midpoint_a2_minus', 'midpoint_a2_plus',
              'midpoint_a3_minus', 'midpoint_a3_plus',
              'fixed_minus_half', 'fixed_minus_quarter', 'fixed_zero',
              'fixed_plus_quarter', 'fixed_plus_half')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def candidate_bank(native_actions, bca_actions):
    """Construct immutable float32 [N,15,4] candidates in documented order.

    Each input must be the deployed actor output for the SAME N observations.
    Slot 0 is the native actor and slot 1 is the BCA actor. Slots 2..9 perturb
    their arithmetic midpoint by -/+0.2 on one coordinate at a time. Slots
    10..14 are complete historical fixed vectors, with other coordinates zero.
    Inputs are already clipped by their actual actor; invalid inputs fail closed.
    The midpoint and clipping are float32 because actions execute as float32.
    """
    native = np.asarray(native_actions)
    bca = np.asarray(bca_actions)
    _require(native.ndim == 2 and native.shape[0] > 0 and native.shape[1] == 4
             and bca.shape == native.shape and native.dtype == np.float32
             and bca.dtype == np.float32 and np.isfinite(native).all()
             and np.isfinite(bca).all() and np.all(np.abs(native) <= 1.)
             and np.all(np.abs(bca) <= 1.),
             'aligned bounded finite float32 [N,4] actor actions required')
    midpoint = (native + bca) * np.float32(.5)
    actions = np.repeat(midpoint[:, None, :], len(SLOT_NAMES), axis=1)
    actions[:, 0] = native
    actions[:, 1] = bca
    for coordinate in range(4):
        for offset, direction in ((0, -DELTA), (1, DELTA)):
            slot = 2 + 2 * coordinate + offset
            actions[:, slot, coordinate] = np.clip(midpoint[:, coordinate] + direction, -1., 1.)
    actions[:, 10:] = np.float32(0.)
    actions[:, 10:, 0] = np.asarray(FIXED_A0, np.float32)
    _require(actions.shape == (len(native), 15, 4) and actions.dtype == np.float32
             and np.isfinite(actions).all() and np.all(np.abs(actions) <= 1.),
             'invalid common action bank')
    # Bytes-backed immutability ensures neither scoring path can alter the bank.
    return np.frombuffer(actions.tobytes(order='C'), dtype=np.float32).reshape(actions.shape)


def duplicate_slot_counts(actions):
    """Number of duplicate slots per state, counting after each first occurrence.

    Duplicates remain in the bank and are charged separately; this receipt is
    descriptive and can also support pairwise diagnostics without inflating N.
    """
    a = np.asarray(actions)
    _require(a.ndim == 3 and a.shape[0] > 0 and a.shape[1:] == (15, 4)
             and a.dtype == np.float32 and np.isfinite(a).all()
             and np.all(np.abs(a) <= 1.), 'valid float32 [N,15,4] bank required')
    counts = np.empty(len(a), np.int64)
    for i, row in enumerate(a):
        seen = []
        duplicate = 0
        for action in row:
            if any(np.array_equal(action, previous) for previous in seen):
                duplicate += 1
            else:
                seen.append(action)
        counts[i] = duplicate
    return counts
