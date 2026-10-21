"""The made-up-data guard: BCA's code runs on arrays only when their content hash is registered as made-up, or with the
user's terminal approval on real data (gate.py). Restated from the weighting test's gate,
experiments/shift_iw/learners_iw.py:54-57 (the SYNTHETIC token) and 273-322 (_Synthetic, register_synthetic,
data_sha256, require_gate), and experiments/shift/learners.py:184-190 (tree_sha256).

An experiment-local attachment (plan section 3.1): it changes none of the repository's declared methods. It is
restated rather than imported because learners_iw.py imports experiments.shift (line 73), which refuses numpy other
than 2.2.6 (experiments/shift/__init__.py:101-103), and the training environment has numpy 1.26.4 (plan section 3.1,
module table). tests/test_guard_compare.py runs the original in /opt/bca-shift-venv on the same made-up arrays and
compares the decisions and content hashes here.

Restated unchanged in substance:
  MADE_UP            the token (learners_iw.SYNTHETIC, lines 273-280); it opens BCA only on registered made-up arrays
  MADE_UP_WORLD      the label made-up data carry (learners_iw.SYNTHETIC_WORLD, line 100), the same string
  tree_sha256        SHA-256 over every leaf's dtype, shape and bytes, in pytree order (learners.py:184-190)
  data_sha256        tree_sha256((data.train, data.bank)) of a container (learners_iw.py:284-286)
  register_made_up   only data labelled MADE_UP_WORLD can be registered (learners_iw.py:289-295)
  require_made_up    the token's branch of require_gate (learners_iw.py:303-311): the label, then the content hash
Extension (declared): the restated Unifloral steps receive Unifloral's own Transition namedtuples, which carry no
label. A bare pytree of arrays (no `world` attribute) is accepted when tree_sha256 of exactly that pytree was
registered; register_arrays registers such a pytree, and only with the made-up label. Since tree_sha256 reads only the
leaves, a Transition built from registered arrays (same order, dtypes, shapes and bytes) has the registered hash.

require_bca(permission, *data) is the one check every BCA entry point calls (outside jit): permission MADE_UP ->
every item must be registered made-up content; a gate.Approval -> gate.verify re-reads the approval from disk; anything
else is refused (gate.ApprovalError).
"""

import hashlib

import numpy as np

from experiments.shift_hosts import gate
from experiments.shift_hosts.gate import ApprovalError

MADE_UP_WORLD = "synthetic arrays (not ThrusterChain data)"          # learners_iw.py:100, SYNTHETIC_WORLD
REFUSAL = ("refused: BCA's code runs on made-up arrays only when their content is registered as made-up "
           "(tests/madeup.py), and on ThrusterChain data only with the user's terminal approval "
           "(gate.require_run_approval)")


class _MadeUp:
    """The token that opens BCA's code on registered made-up arrays only (learners_iw.py:273-277)."""

    def __repr__(self):
        return "MADE_UP"


MADE_UP = _MadeUp()
_MADE_UP_DATA = set()      # content hashes of registered made-up arrays (learners_iw.py:281)


def tree_sha256(tree):
    """SHA-256 over every leaf of a pytree, in pytree order: dtype string, shape string, then the bytes (restated from
    experiments/shift/learners.py:184-190; leaves through np.asarray, so JAX and numpy arrays hash alike)."""
    import jax

    h = hashlib.sha256()
    for x in jax.tree_util.tree_leaves(tree):
        a = np.ascontiguousarray(np.asarray(x))
        h.update(str(a.dtype).encode() + str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def state_sha256(tree):
    """tree_sha256 with every leaf through jnp.asarray first, so a Python integer step counter and its int32 array hash
    alike (restated from learners_iw.py:380-383). The hash of every parameter, state and key hash in run records (S4,
    S10). Creates JAX arrays, so call it after unifloral_import.prepare()."""
    import jax
    import jax.numpy as jnp

    return tree_sha256(jax.tree_util.tree_map(jnp.asarray, tree))


def _is_container(data):
    return hasattr(data, "train") and hasattr(data, "bank")


def data_sha256(data):
    """Content hash of a container's train and bank arrays (learners_iw.py:284-286), or of a bare pytree."""
    return tree_sha256((data.train, data.bank)) if _is_container(data) else tree_sha256(data)


def register_made_up(data):
    """Register a container of made-up arrays (attribute world == MADE_UP_WORLD, with train and bank), so MADE_UP
    opens BCA on exactly these arrays (learners_iw.py:289-295). Only tests/madeup.py and test fixtures call it.
    Returns `data`."""
    if getattr(data, "world", None) != MADE_UP_WORLD:
        raise ApprovalError(f"{REFUSAL} (only made-up arrays can be registered, not {getattr(data, 'world', None)!r})")
    _MADE_UP_DATA.add(data_sha256(data))
    return data


def register_arrays(tree, world):
    """Register a bare pytree of made-up arrays (the declared extension); `world` must be MADE_UP_WORLD. Returns the
    content hash."""
    if world != MADE_UP_WORLD:
        raise ApprovalError(f"{REFUSAL} (only made-up arrays can be registered, not {world!r})")
    digest = tree_sha256(tree)
    _MADE_UP_DATA.add(digest)
    return digest


def is_registered(data):
    return data_sha256(data) in _MADE_UP_DATA


def require_made_up(*items):
    """Refuse (ApprovalError) unless every item is registered made-up content: a container must carry the made-up label
    and have its (train, bank) hash registered (learners_iw.py:303-311); a bare pytree must have its own hash
    registered. Returns MADE_UP."""
    if not items:
        raise ApprovalError(f"{REFUSAL} (nothing to check)")
    for data in items:
        if hasattr(data, "world") or _is_container(data):
            world = getattr(data, "world", None)
            if world != MADE_UP_WORLD:
                raise ApprovalError(f"{REFUSAL} (the made-up token opens BCA on made-up arrays only, not on "
                                    f"{world!r})")
        if data_sha256(data) not in _MADE_UP_DATA:
            raise ApprovalError(f"{REFUSAL} (the made-up token opens BCA on registered made-up arrays only; the "
                                "content of these arrays is not registered)")
    return MADE_UP


def require_bca(permission, *data):
    """The check every BCA entry point calls before running BCA's code (a band fit, a refresh, a dose or weight read,
    a restated step with BCA present): MADE_UP on registered made-up arrays (require_made_up), or a gate.Approval that
    still verifies from disk (real data). Returns the permission."""
    if permission is MADE_UP:
        require_made_up(*data)
        return permission
    if isinstance(permission, gate.Approval):
        return gate.verify(permission)
    raise ApprovalError(REFUSAL)
