"""The single probe-building function (protocol 4.4, forced-probe arrays).

It serves every forced-action evaluation: KC, F, the onset contrast, the reach forecast, D1/D2 and slots 1-5 of the
4.8 reproduction of saved probes. Each kind has a sealed point set, checked by exact float32 bits:
- "gated": Part 0 slots, given by NAME (S9 and the past-knee slots), a1..a3 exactly 0.0. A rounded number such as
  float32(-0.9) is refused, and a caller's expected name list detects a swapped slot;
- "actor_own": the same named Part 0 slots with the actor's own a1..a3 (reported sets only);
- "g05": step 0.3's G0.5 / F points {-0.5, -0.60000002, -0.69999999, -KNEE, s05, -0.89999998, -1}, a1..a3 = 0;
- "reach_forecast": the G0.5 points plus {-0.25, 0, 0.25, 0.5}, with each row's own a1..a3 (exempt from zero);
- "identity_image": the float32 T-image of a named past-knee slot in a stalling world, a1..a3 = 0;
- "known_answer": the 4.8 reproduction's FIXED_A0 slots 1-5, a1..a3 = 0 (not gated probes).
Protocol 4.4 says gated arrays hold "the Part 0 slot value"; step 0.3 names point sets outside the Part 0 table, so
those are sealed here as their own sets (disclosed in README.md). Every array is checked by both region
implementations.
"""
import numpy as np

import region_independent as second
import worlds
from core import require
from knee_region import GATED_PAST_KNEE, KNEE, S9, SLOTS, past_knee_a0

FIXED_A0 = (-0.5, -0.25, 0.0, 0.25, 0.5)   # the prior study's slots 1-5 (evaluation.py fixed_candidate_bank)
NAMED = S9 + GATED_PAST_KNEE
G05_POINTS = tuple(float(np.float32(x)) for x in (-0.5, -0.6, -0.7, -KNEE, SLOTS["s05"], -0.9, -1.0))
REACH_POINTS = G05_POINTS + tuple(float(np.float32(x)) for x in (-0.25, 0.0, 0.25, 0.5))
KINDS = ("gated", "actor_own", "g05", "reach_forecast", "identity_image", "known_answer")
OWN_A123 = ("actor_own", "reach_forecast")


def _bits(values):
    return [int(b) for b in np.asarray(values, np.float32).view(np.uint32)]


def _allowed_bits(kind, world):
    if kind in ("gated", "actor_own"):
        return set(_bits([SLOTS[n] for n in NAMED]))
    if kind == "g05":
        return set(_bits(G05_POINTS))
    if kind == "reach_forecast":
        return set(_bits(REACH_POINTS))
    if kind == "identity_image":
        image, _ = worlds.identity_shift(np.asarray([SLOTS[n] for n in GATED_PAST_KNEE], np.float64), world)
        return set(_bits(image))
    return set(_bits(FIXED_A0))


def resolve(slots, kind, world=None):
    """Float32 slot values for a kind: names for the named kinds, values otherwise."""
    if kind in ("gated", "actor_own"):
        require(all(isinstance(n, str) and n in NAMED for n in slots), "Named kinds take Part 0 slot names")
        return np.asarray([SLOTS[n] for n in slots], np.float32)
    if kind == "identity_image":
        require(all(isinstance(n, str) and n in GATED_PAST_KNEE for n in slots), "Identity images take past-knee slot names")
        return worlds.identity_shift(np.asarray([SLOTS[n] for n in slots], np.float64), world)[0]
    return np.asarray(slots, np.float32)


def build(n_rows, slots, *, a123=None, kind="gated", world=None):
    """(n_rows, len(slots), 4) float32 probes with the slot value at index 0 (`slots` are names for named kinds)."""
    require(kind in KINDS, "Unknown probe kind")
    values = resolve(slots, kind, world)
    require(values.ndim == 1 and len(values) > 0, "Slots must be a nonempty list")
    out = np.zeros((int(n_rows), len(values), 4), np.float32)
    out[..., 0] = values[None, :]
    if kind in OWN_A123:
        a123 = np.asarray(a123, np.float32)
        require(a123.shape == (int(n_rows), 3), "Own a1..a3 must be (n_rows, 3) float32")
        out[..., 1:] = a123[:, None, :]
    else:
        require(a123 is None, "This kind carries a1..a3 = 0")
    assert_probes(out, slots, kind=kind, a123=a123, world=world)
    return out


def assert_probes(probes, slots, *, kind="gated", a123=None, world=None):
    """Index 0 equals the expected slots in order (names for named kinds, so a swap is caught) and lies in the
    kind's sealed set; the a1..a3 rule holds; both region implementations agree."""
    require(kind in KINDS, "Unknown probe kind")
    probes = np.asarray(probes)
    values = resolve(slots, kind, world)
    require(probes.dtype == np.float32 and probes.shape[-1] == 4, "Probes must be float32 (..., 4)")
    require(np.array_equal(probes[..., 0].view(np.uint32), np.broadcast_to(values, probes.shape[:-1]).view(np.uint32)),
            "Slot value at index 0 differs from the expected slot (by bits)")
    allowed = _allowed_bits(kind, world)
    bad = [float(v) for v, b in zip(values, _bits(values)) if b not in allowed]
    require(not bad, f"Not in the sealed point set of kind {kind!r}: {bad}")
    if kind in OWN_A123:
        require(np.array_equal(probes[..., 1:], np.broadcast_to(np.asarray(a123, np.float32)[:, None, :], probes[..., 1:].shape)),
                "Exempt probe must carry the row's own a1..a3 exactly")
    else:
        require(bool(np.all(probes[..., 1:] == 0.0)), "a1..a3 must be exactly 0.0")
    primary = past_knee_a0(probes)[0]
    require(np.array_equal(primary, second.agree(probes)), "Region implementations disagree on a probe")
    return primary


def receipt(slots, kind, world=None):
    values = resolve(slots, kind, world)
    return {"kind": kind, "slots": list(slots) if kind in ("gated", "actor_own", "identity_image") else None,
            "values": [float(v) for v in values], "bits": [hex(b) for b in _bits(values)], "world": world}


def known_answer_slots_1_to_5(n_rows):
    """Slots 1-5 of the 4.8 reproduction, built by the same function under the known-answer kind."""
    return build(n_rows, FIXED_A0, kind="known_answer")
