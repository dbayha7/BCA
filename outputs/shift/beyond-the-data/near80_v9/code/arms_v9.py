"""Near-80 (version 9): the arm table (PREREGISTRATION.md sections 3.3 and 5; DECISION.md sections 2.2 and 2.4).

Step 2's arm rule (restructure/step2/step2lib.py arm_table, lines 340-364, never edited) restated for one seed in both
worlds, from the switch records' entries (BCA's two pooled groups), with version 9's changes:
  the refusal rule (section 3.3), which never raises:
    obca_k1, obca_k2   refused in a world where BCA's switch is on and its R_tail is +infinity;
    obca_forced        refused in BOTH worlds when the stalling world's R_tail is +infinity, the switch on or off (it
                       takes the stalling world's height in both worlds);
    fixed_gated        never refused (its height is 1.0 where BCA's switch is on);
    a negative height still refuses its arm, as step 2's rule does;
  splitconf_k1 in conformal_k1's slot: the textbook split-conformal switch, the ceil((n + 1) * 0.9)-th smallest score
    of each group; on when the full-throttle group's value minus the rest's is above 0.10; height that gap. Step 2's
    uncorrected gap (conformal_k1's) is kept as a reported number with no actor;
  the dose arms plain_h25 and plain_h50: the plain switch (mean(full-throttle) - mean(rest) > 0.10) with the ramp at
    the fixed heights 2.5 and 5.0, trained only in a world where that switch is on; where it is off each is the
    no-caution actor by definition (never trained, never in the height-0 check);
  plain_limit: the plain switch turning on the hand-set range limit; computed, never trained: where the switch is on it
    is the hand-set range limit's actor (range_hinge), where it is off the no-caution actor (none).
Every other arm of the main setting reads no switch. When step 2's own arm_table does not refuse, the restated rule
must equal it on the five arms it shares (obca_k1, obca_k2, obca_forced, fixed_gated, meangap_k1), t_lo and t_up, and
its conformal_k1 must equal the reported step-2 gap (a built-in check, policy_phase).

step2_table gives step 2's own six-arm row (conformal_k1 at step 2's rank, no refusal rule, no dose arm): the
plumbing's reproduction check trains it (section 8.2), so its bit-for-bit comparison with step 2's saved actors is not
exposed to the changed slot.
Standard library only (with near80_constants).
"""
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import near80_constants as K  # noqa: E402

STEP2_SHARED = ("obca_k1", "obca_k2", "obca_forced", "fixed_gated", "meangap_k1")


class TableError(Exception):
    """A switch entry cannot be read (a recorded stop in the policy phase)."""


def num(x):
    """A recorded number (the strings 'inf', '-inf' and 'nan' included); nan when it cannot be read."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def r_tail(cert):
    """BCA's R_tail (+inf when not certified); a record without it reads the height (R_tail - R_rest)."""
    return num(cert["R_tail"]) if "R_tail" in cert else num(cert.get("height"))


def splitconf_gap(e):
    """The textbook split-conformal gap of an entry (learner_phases.certificate_entry's 'split_conformal')."""
    sc = e.get("split_conformal")
    if not isinstance(sc, dict):
        raise TableError("the entry has no textbook split-conformal quantiles")
    return num(sc.get("gap"))


def _ramp(switch, h, k=1.0):
    return dict(switch=bool(switch), h=float(h), k=float(k), refused=False, refused_reason=None)


def arm_table(per, seed, rule=K.ARM_RULE):
    """{world: row} for one seed in both worlds (module docstring). per: {'<world>/seed<k>': entry}. row: t_lo, t_up,
    arms {name: {switch, h, k, refused, refused_reason} | plain_limit {switch, actor, source}}, sources, refused (the
    refused arms), bca_refused (obca_k1 refused here), dose_on (the plain switch is on here: the dose arms train),
    the_six (the six arm names of step 2's program, splitconf_k1 in conformal_k1's slot). TableError when an entry
    cannot be read."""
    rule = {**dict(fixed_height=1.0, conformal_margin=0.10, rank_p=0.05), **(rule or {})}
    out = {}
    try:
        stall = per[f"stall/seed{int(seed)}"]
        stall_h = num(stall["certificate"]["height"])
        stall_inf = not math.isfinite(r_tail(stall["certificate"]))
        for w in K.WORLDS:
            key = f"{w}/seed{int(seed)}"
            e = per[key]
            c, g, rt = e["certificate"], e["groups"], e["rank_test"]
            on_bca, h_bca = bool(c["on"]), num(c["height"])
            tail_inf = not math.isfinite(r_tail(c))
            gap90 = num(g["q90_full_throttle"]) - num(g["q90_rest"])
            gap_sc = splitconf_gap(e)
            on_sc = bool(gap_sc > float(rule["conformal_margin"]))
            p_rank = num(rt["p"])
            on_rank = bool(p_rank < float(rule["rank_p"]))
            meangap = num(g["mean_full_throttle"]) - num(g["mean_rest"])
            on_plain = bool(math.isfinite(meangap) and meangap > K.MEANDIFF_MARGIN)
            arms = dict(
                obca_k1=_ramp(on_bca, h_bca if on_bca else 0.0),
                obca_k2=_ramp(on_bca, h_bca if on_bca else 0.0, 2.0),
                obca_forced=_ramp(True, stall_h),
                fixed_gated=_ramp(on_bca, float(rule["fixed_height"]) if on_bca else 0.0),
                splitconf_k1=_ramp(on_sc, gap_sc if on_sc else 0.0),
                meangap_k1=_ramp(on_rank, max(meangap, 0.0) if on_rank else 0.0))
            for name in K.DOSE_ARMS:
                arms[name] = dict(_ramp(on_plain, K.DOSE_HEIGHTS[name] if on_plain else 0.0),
                                  trained=on_plain, where_off="the no-caution actor (a switched-off ramp is no caution)")
            if on_bca and tail_inf:
                for name in ("obca_k1", "obca_k2"):
                    arms[name].update(refused=True, refused_reason=(
                        f"refused: {key}: BCA's switch is on but its R_tail is not finite (BCA could not certify the "
                        f"full-throttle group of {g.get('n_full_throttle')} rows)"))
            if stall_inf:
                arms["obca_forced"].update(refused=True, refused_reason=(
                    f"refused: stall/seed{int(seed)}: the stalling world's R_tail is not finite, and the always-on ramp "
                    f"takes the stalling world's height in both worlds"))
            for name, a in arms.items():
                if a["refused"]:
                    continue
                if a["switch"] and not math.isfinite(a["h"]):
                    if name == "fixed_gated":
                        raise TableError(f"{key}: fixed_gated has a height {a['h']} that is not finite")
                    a.update(refused=True, refused_reason=f"refused: {key}: arm {name} is switched on but its height "
                                                          f"{a['h']} is not finite")
                elif a["h"] < 0:
                    a.update(refused=True, refused_reason=f"refused: {key}: arm {name} has a negative height {a['h']}")
            arms[K.PLAIN_LIMIT] = dict(switch=on_plain, actor=K.HAND_SET_LIMIT if on_plain else "none",
                                       source=("the hand-set range limit's actor (the plain switch is on)" if on_plain
                                               else "the no-caution actor (the plain switch is off)"),
                                       computed=True, refused=False, refused_reason=None)
            out[w] = dict(t_lo=num(e["t_lo"]), t_up=num(e["t_up"]), arms=arms, the_six=K.STEP2_RAMP_ARMS,
                          sources=dict(bca_on=on_bca, bca_height=h_bca, bca_R_tail=r_tail(c), stall_bca_height=stall_h,
                                       stall_R_tail_infinite=stall_inf, splitconf_gap=gap_sc,
                                       splitconf_q_tail=num((e.get("split_conformal") or {}).get("q_tail")),
                                       splitconf_q_rest=num((e.get("split_conformal") or {}).get("q_rest")),
                                       step2_conformal_gap_reported=gap90,
                                       plain_gap_information_only=num(c.get("plain_gap", "nan")), rank_p=p_rank,
                                       mean_gap=meangap, plain_switch_on=on_plain),
                          refused=sorted(n for n, a in arms.items() if a.get("refused")),
                          bca_refused=bool(arms[K.PRIMARY]["refused"]), dose_on=on_plain)
    except (KeyError, TypeError) as error:
        raise TableError(f"the switch records cannot be read ({error!r})") from None
    return out


def step2_table(per, seed, rule=K.ARM_RULE):
    """Step 2's own six-arm row for one seed in both worlds (step2lib.arm_table's rule exactly, conformal_k1 at step
    2's rank; no refusal rule, no dose arm, no plain_limit): what the plumbing's reproduction trains. Raises TableError
    where step 2's rule would refuse."""
    rule = {**dict(fixed_height=1.0, conformal_margin=0.10, rank_p=0.05), **(rule or {})}
    out = {}
    try:
        stall_h = num(per[f"stall/seed{int(seed)}"]["certificate"]["height"])
        for w in K.WORLDS:
            key = f"{w}/seed{int(seed)}"
            e = per[key]
            c, g, rt = e["certificate"], e["groups"], e["rank_test"]
            on_bca, h_bca = bool(c["on"]), num(c["height"])
            gap90 = num(g["q90_full_throttle"]) - num(g["q90_rest"])
            on_conf = bool(gap90 > float(rule["conformal_margin"]))
            p_rank = num(rt["p"])
            on_rank = bool(p_rank < float(rule["rank_p"]))
            meangap = num(g["mean_full_throttle"]) - num(g["mean_rest"])
            arms = dict(obca_k1=_ramp(on_bca, h_bca if on_bca else 0.0),
                        obca_k2=_ramp(on_bca, h_bca if on_bca else 0.0, 2.0),
                        obca_forced=_ramp(True, stall_h),
                        fixed_gated=_ramp(on_bca, float(rule["fixed_height"]) if on_bca else 0.0),
                        conformal_k1=_ramp(on_conf, gap90 if on_conf else 0.0),
                        meangap_k1=_ramp(on_rank, max(meangap, 0.0) if on_rank else 0.0))
            for name, a in arms.items():
                if (a["switch"] and not math.isfinite(a["h"])) or a["h"] < 0:
                    raise TableError(f"refused: {key}: step 2's arm {name} has the height {a['h']}")
            out[w] = dict(t_lo=num(e["t_lo"]), t_up=num(e["t_up"]), arms=arms, the_six=K.STEP2_OWN_RAMP_ARMS,
                          sources=dict(bca_on=on_bca, bca_height=h_bca, conformal_gap=gap90, rank_p=p_rank,
                                       mean_gap=meangap),
                          refused=[], bca_refused=False, dose_on=False, step2_own_table=True)
    except (KeyError, TypeError) as error:
        raise TableError(f"the switch records cannot be read ({error!r})") from None
    return out


def same_as_step2(row, s2row):
    """True when the restated row's five shared arms, t_lo and t_up equal step2lib.arm_table's exactly, and step 2's
    conformal_k1 equals the reported step-2 gap's rule."""
    if (row["t_lo"], row["t_up"]) != (s2row["t_lo"], s2row["t_up"]):
        return False
    for name in STEP2_SHARED:
        a, b = row["arms"][name], s2row["arms"][name]
        if (bool(a["switch"]), float(a["h"]), float(a["k"])) != (bool(b["switch"]), float(b["h"]), float(b["k"])):
            return False
    gap = row["sources"]["step2_conformal_gap_reported"]
    c2 = s2row["arms"]["conformal_k1"]
    on = bool(gap > float(K.ARM_RULE["conformal_margin"]))
    return (bool(c2["switch"]), float(c2["h"])) == (on, float(gap) if on else 0.0)


def refusal_events(table):
    """The refusal events of a seed's table (section 3.3 / the stop table): which world refused, and how."""
    out = []
    for w, row in table.items():
        s = row["sources"]
        if s["bca_on"] and not math.isfinite(s["bca_R_tail"]):
            out.append(dict(world=w, event="refusal_on"))
        elif w == "stall" and s["stall_R_tail_infinite"]:
            out.append(dict(world=w, event="stall_inf_off"))
        elif w == "no_stall" and not s["bca_on"] and not math.isfinite(s["bca_R_tail"]):
            out.append(dict(world=w, event="no_stall_inf_off"))
    return out
