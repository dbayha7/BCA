"""Write controllers_data.json: the points of the learned actors and the room check's arms, for the page's controller
guide and its controller tooltips. Read only, from the gate's committed result files:

  runs/shift/68d9e8779e148b7b/harm_return/base/alpha5/seed{0..4}/values.json
      the return check's actor-only runs at alpha = 5 (cloned actor, the actors of the full-log, gapped and
      random-removal critics, the critic swap's actors, the no-stall comparison world's actors) and plain TD3+BC
      (critic and actor trained together on the gapped log): points and the share of steps past 80% per situation
  outputs/shift/stop-report/check3_room/points_by_arm.json
      the room check's arms (drawn from the gated per-seed room values.json files): points and the share of steps
      past 80% per situation
  runs/shift/68d9e8779e148b7b/room/base/alpha5/seed{0..4}/room_blind.json
      the room check's lock, written before any return: how closely strength-matched even caution was matched in
      strength to harm-placed caution, and its multiplier

Every value is a mean over the gate seeds 0-4 with the lowest and highest seed. The script checks that each source
file was written under the pre-registration whose fingerprint the page reports, and that the numbers the page's
Results section already prints agree with these files.
"""

import hashlib
import json
import os
import sys

BCA = "/home/user/bca"
PREREG = "68d9e8779e148b7bfd2fcc6dc32e0a1091e5bfbbcb83cef2e4b58e4f6ec60fab"
RET = f"{BCA}/runs/shift/{PREREG[:16]}/harm_return/base/alpha5"
ROOM = f"{BCA}/outputs/shift/stop-report/check3_room/points_by_arm.json"
SEEDS = [0, 1, 2, 3, 4]
SIT = ["recorded", "strong_current", "big_waves", "big_kicks"]


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def stats(vals):
    vals = [float(v) for v in vals]
    return dict(mean=sum(vals) / len(vals), lo=min(vals), hi=max(vals), values=vals)


sources = []

# the return check's actors, alpha = 5
ret = {}
for s in SEEDS:
    p = f"{RET}/seed{s}/values.json"
    d = json.load(open(p))
    assert d["prereg_sha256"] == PREREG and d["alpha"] == 5.0 and d["kind"] == "shift-harm_return_seed", p
    assert not any(d["failed"].values()), p
    ret[s] = d
    sources.append(dict(path=os.path.relpath(p, BCA), sha256=sha(p)))
actors = ret[0]["actor_names"]
assert all(ret[s]["actor_names"] == actors for s in SEEDS)
gate_points = {a: {sit: stats(ret[s]["situations"][sit]["points"][a] for s in SEEDS) for sit in SIT} for a in actors}
gate_past80 = {a: {sit: stats(ret[s]["situations"][sit]["unrecorded_throttle_share"][a] for s in SEEDS) for sit in SIT}
               for a in actors}

# the room check's arms
room = json.load(open(ROOM))
assert room["prereg_sha256"] == PREREG and room["kind"] == "shift-room-points-by-arm" and room["alpha"] == 5.0
sources.append(dict(path=os.path.relpath(ROOM, BCA), sha256=sha(ROOM)))
seeds = room["seeds"]
assert sorted(int(k) for k in seeds) == SEEDS
arms = sorted(seeds["0"]["points"])
assert all(sorted(seeds[str(s)]["points"]) == arms for s in SEEDS)


def arm_vals(field, arm, sit, s):
    v = seeds[str(s)][field]
    if arm == "shuffled":  # the page reports the mean of the two shuffles
        return (v["shuffled_0"][sit] + v["shuffled_1"][sit]) / 2
    return v[arm][sit]


room_arms = [a for a in arms if not a.startswith("shuffled_")] + ["shuffled"]
room_points = {a: {sit: stats(arm_vals("points", a, sit, s) for s in SEEDS) for sit in SIT} for a in room_arms}
room_past80 = {a: {sit: stats(arm_vals("share_past_80", a, sit, s) for s in SEEDS) for sit in SIT} for a in room_arms}

# the numbers the page's Results section prints (literal there) must agree with these files
printed_room = {"none": (87.6, -10.1, 72.7, 53.7), "harm_placed": (88.2, -7.6, 73.7, 54.1),
                "strength_matched_even": (87.7, -5.8, 72.8, 53.3), "amount_matched": (87.6, -7.0, 72.6, 53.3),
                "shuffled": (87.6, -7.1, 72.6, 53.4), "reversed": (87.4, -6.4, 72.4, 53.1),
                "cap2": (88.1, -5.6, 74.2, 53.6), "even_x1.5": (87.2, -4.5, 72.3, 51.9),
                "even_x2": (85.7, -0.7, 70.7, 48.3), "even_x3": (82.4, 7.1, 67.3, 41.0)}
# The Results table now reads these from controllers_data.json; this pins them to the stop report's values. The
# report prints 53.5 for shuffled caution in big starting pushes because it rounded each shuffle before averaging; the
# files give 53.42 (53.4), which the page prints with that note. A printed value that differs by one in its last digit
# is reported below; a larger difference stops the build.
bad = [(a, sit, round(room_points[a][sit]["mean"], 2), v) for a, row in printed_room.items()
       for sit, v in zip(SIT, row) if abs(room_points[a][sit]["mean"] - v) > 0.1]
for a, row in printed_room.items():
    for sit, v in zip(SIT, row):
        if round(room_points[a][sit]["mean"], 1) != v and abs(room_points[a][sit]["mean"] - v) <= 0.1:
            print(f"note: Results prints {v} for {a} in {sit}; the files give {room_points[a][sit]['mean']:.3f}")
if round(gate_points["td3bc_gapped"]["strong_current"]["mean"], 1) != -16.8:
    bad.append(("td3bc_gapped", "strong_current", gate_points["td3bc_gapped"]["strong_current"]["mean"], -16.8))
if bad:
    sys.exit(f"the page's printed numbers disagree with the result files: {bad}")

# the room check's blind strength match (its lock, before any return): the largest relative difference in strength
# between strength-matched even caution and harm-placed caution over the seeds, and the multipliers it chose
match_err, match_mult = [], []
for s in SEEDS:
    p = f"{BCA}/runs/shift/{PREREG[:16]}/room/base/alpha5/seed{s}/room_blind.json"
    d = json.load(open(p))
    assert d["prereg_sha256"] == PREREG and d["alpha"] == 5.0, p
    assert open(p + ".sha256").read().split()[0] == sha(p), p
    st = d["strengths"]
    err = abs(st["strength_matched_even"] - st["harm_placed"]) / st["harm_placed"]
    assert abs(err - d["match"]["relative_error"]) < 1e-9, (p, err, d["match"]["relative_error"])
    match_err.append(err)
    match_mult.append(d["match"]["multiplier"])
    sources.append(dict(path=os.path.relpath(p, BCA), sha256=sha(p)))
strength_match = dict(max_relative_error=max(match_err), multiplier_lo=min(match_mult), multiplier_hi=max(match_mult))

out = dict(
    note="Read only from the gate's committed result files (alpha = 5, gate seeds 0-4): points 0 = the recording "
         "controller, 100 = the stall-aware controller in the stalling world; in the no-stall comparison world 100 is "
         "the rising-thrust controller. past80: the share of visited steps at which some thruster is past 80%.",
    prereg_sha256=PREREG, alpha=5.0, seeds=SEEDS, sources=sources,
    gate=dict(points=gate_points, past80=gate_past80),
    room=dict(points=room_points, past80=room_past80, strength_match=strength_match),
)
json.dump(out, open("controllers_data.json", "w"), indent=1)
print(f"wrote controllers_data.json: {len(gate_points)} gate actors, {len(room_points)} room arms, "
      f"{len(sources)} source files")
