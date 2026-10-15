"""Shift cards for the ThrusterChain stop report, computed OUTSIDE the gate after a reporting bug.

acceptance.report_cards passes the whole per-situation anchors dict to shift.shift_card, which expects one situation's
anchors, so every report-card stage raised KeyError: 'top'. The code is hashed into the pre-registration, so it is not
edited. This script repeats report_cards line for line with the single difference an -> an[situation], reads every
gated output through the gate's own verified loaders, and writes only outside runs/shift. No pass rule depends on the
cards; they are a reported description of where each policy acts.

Usage (from /home/user/bca): python cards_outside_gate.py <seed> <out_dir>
"""
import json
import sys
import time
from pathlib import Path

import experiments.shift.acceptance as A
import experiments.shift.learners as L
import experiments.shift.recorder as RC
import experiments.shift.shift as SH
import experiments.shift.thruster_chain as TC
import experiments.shift.truth as TR


def main(rep, out_dir):
    prereg = A.load_prereg(A.refuse_protected(A.RUNS_DEFAULT))
    A.guard(prereg)
    cfg = prereg.config
    outcome = A.gate_outcome(prereg)
    variant = outcome["variant"]
    started = time.perf_counter()
    st = A.load_seed(prereg.runs, cfg, variant, "stall", rep)
    pg = st.prepared("gap", cfg.learn)
    recorder = RC.Recorder(st.spec, st.physics)
    ref = SH.card_reference(st.log, st.removals["gap"], recorder)
    policies = {"recording controller": TR.recorder_on_bank(st.physics, st.spec, TC.BANK_SIZE),
                "stall-aware controller": RC.stall_aware(st.physics),
                "rising-thrust controller": RC.rising_thrust(st.physics)}
    alpha = outcome.get("alpha") or A._last_alpha_run(prereg, variant)
    d = A._variant_root(prereg, "harm_return", variant) / f"alpha{alpha:g}" / f"seed{rep}"
    rec = A.finished(d / "values.json", prereg)
    acts = L.load_tree(d / "actors.msgpack", {k: L.actor_template() for k in rec["actor_names"]}, rec["actors_sha256"])
    for k, label in (("clone", "cloned actor"), ("gapped", "actor of the gapped critic (H2)"),
                     ("full", "actor of the full-log critic (H2)"), ("td3bc_gapped", "plain TD3+BC on the gapped log")):
        if k in acts and not rec["failed"].get(k):
            policies[label] = L.mlp_policy(acts[k], pg, label)
    d = A._variant_root(prereg, "room", variant) / f"alpha{alpha:g}" / f"seed{rep}"
    blind = A.finished(d / "room_blind.json", prereg)
    acts = L.load_tree(d / "actors.msgpack", {k: L.actor_template() for k in blind["names"]}, blind["actors_sha256"])
    for k, label in (("none", "room check, no extra caution"), ("harm_placed", "room check, harm-placed")):
        if k in acts:
            policies[label] = L.mlp_policy(acts[k], pg, label)
    an = A.anchors_of(prereg, variant, "stall")
    cards = {}
    for name, pol in policies.items():
        cards[name] = {s: SH.shift_card(pol, s, st.physics, ref, an[s], ms_states=cfg.card_ms_states,
                                        knn_states=cfg.card_knn_states) for s in A.SITUATIONS}
        print(f"shift cards, seed {rep}: {name}", flush=True)
    out = dict(kind="shift-cards-outside-gate", prereg_sha256=prereg.digest, variant=variant, rep=int(rep),
               alpha=alpha, seconds=time.perf_counter() - started,
               note="computed outside the gate: acceptance.report_cards passed the whole anchors dict instead of "
                    "anchors[situation]; this repeats it with that one difference. Not a gated output.",
               cards=cards)
    path = Path(out_dir) / f"cards_seed{rep}.json"
    A.write_json(path, out)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2])
