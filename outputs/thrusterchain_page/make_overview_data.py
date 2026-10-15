"""The facts the ThrusterChain page's task overview quotes, computed from the environment's own code.

Run from the repository root (read-only; nothing under /home/user/bca is written, nothing trains):

  cd /home/user/bca && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 nice -n 10 \
      /tmp/claude-0/bcavenv/bin/python <this file>

Writes overview_data.json beside this file; build.py loads it as D["overview"], so its placeholders are
{{overview/<key>#fmt}}. Follows make_data.py's conventions: only experiments.shift.{thruster_chain, recorder, truth,
stage0} are imported (stage0 only to rerun the trigger bisection), the SHA-256 of every file in experiments/shift is
taken at the start and at the end, and the script refuses to write if any changed. Values that page_data.json already
holds are not repeated here (the page keeps using their page_data paths); a few are re-derived and checked against
page_data.json so that both files agree.

Methods, by section (each section of the JSON also carries a 'how' string):
  signs        one exact step of dynamics() from rest; K's current column; the full-throttle pattern; the live
               panel's drawing rules read from template.html.
  who_pushes   linear algebra on A, B and v from thruster_chain.matrices(): the alternating-sum identity, the
               least-squares throttles, the closed-loop steady state of the textbook rule K (no cap, no wave), the
               steady-state cost-optimal constant throttles; checked by a 4,000-step simulation.
  water_alone  (I - A)^-1 0.3 v (steady offset per unit current) and (e^{0.27 i} I - A)^-1 0.3 v (the steady response
               to a wave of amplitude 1), checked by simulation; exact rollouts of every reference controller on the
               four evaluation banks (1,000 starts x 100 steps, float64, truth.bank_rollout) split into float cost
               and throttle cost per visited state.
  waves        the water's push c + w1 on every visited state of the banks (it does not depend on the controller);
               start-law endpoints; stage0.bisect_trigger rerun.
  swings       eigenvalues and eigenvectors of A (the page's Swing table uses the same eigenvalues).
  controllers  exact bank rollouts of the stall-aware and rising-thrust controllers; the closed-loop wave response
               of K and of normal driving (complex steady state); steady states in a steady current (4,000 steps).
  starts       the bank starts themselves; closed-loop eigenvalues; paired return differences between banks.
  recording    the recording controller on the banks with its fixed streams (default_rng([7, 1])), with each of its
               two departures from normal driving switched off in turn.
  interface    inspect.signature, a real reset() and step() through gymnasium.make, gymnasium's registry.
"""

import hashlib
import inspect
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

REPO = Path("/home/user/bca")
HERE = Path(__file__).resolve().parent
OUT = HERE / "overview_data.json"
PAGE_DATA = HERE / "page_data.json"
TEMPLATE = HERE / "template.html"
sys.path.insert(0, str(REPO))
if Path.cwd().resolve() != REPO:
    raise SystemExit("run from the repository root, /home/user/bca")

import numpy as np  # noqa: E402

import experiments.shift as PKG  # noqa: E402,F401  (puts gymnasium on the path)
import experiments.shift.thruster_chain as TC  # noqa: E402
import experiments.shift.recorder as RC  # noqa: E402
import experiments.shift.truth as TR  # noqa: E402
import experiments.shift.stage0 as S0  # noqa: E402
import gymnasium as gym  # noqa: E402

T0 = time.time()
SHIFT_DIR = REPO / "experiments" / "shift"
SITS = TC.SITUATIONS
PLAIN_SITUATION = {"recorded": "recorded conditions", "strong_current": "strong current", "big_waves": "big waves",
                   "big_kicks": "big starting pushes"}
TURN = TC.STALL.wave_turn
GAIN = TC.STALL.gain
CAP = 0.8


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def file_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(SHIFT_DIR.iterdir()) if p.is_file()}


def rnd(v, d=6):
    """Round floats (nested lists / arrays / dicts) to d decimals and convert numpy types; NaN/inf -> None."""
    if isinstance(v, dict):
        return {str(k): rnd(x, d) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [rnd(x, d) for x in v]
    if isinstance(v, np.ndarray):
        return rnd(v.tolist(), d)
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        v = float(v)
        if not math.isfinite(v):
            return None
        r = round(v, d)
        return 0.0 if r == 0 else r
    return v


def check(cond, msg):
    if not cond:
        raise SystemExit(f"check failed: {msg}")


HASH_START = file_hashes()
PD = json.loads(PAGE_DATA.read_text())
data = {}

A, B = TC.matrices(TC.STALL)
A10, B10 = A[:TC.DS, :TC.DS], B[:TC.DS]          # B10 = 0.3 x (thruster k -> speeds of floats k, k + 1)
v = TC.push_pattern()                              # 1 on every float's speed number
B5 = (B10 / GAIN)[1::2]                            # 5 x 4: which floats' speed numbers each thruster pushes
K = RC.textbook_gain()
Ks, Kc, Kw = K[:, :TC.DS], K[:, TC.CURRENT], K[:, TC.WAVE1:]
P = RC.full_throttle_pattern(K)
I10 = np.eye(TC.DS)


def per_float(vec):
    """10 float numbers -> per float dict(position, speed, distance)."""
    vec = np.asarray(vec, np.float64)
    return [dict(float=j + 1, position=vec[2 * j], speed=vec[2 * j + 1], distance=math.hypot(vec[2 * j], vec[2 * j + 1]))
            for j in range(5)]


def swing_peak(S):
    """Complex steady response S (10,) to e^{i 0.27 t}: per float, the largest distance from rest over a cycle
    (largest singular value of the real 2 x 2 map) and the cycle-average of the squared distance."""
    out = []
    for j in range(5):
        gp, gv = S[2 * j], S[2 * j + 1]
        M = np.array([[gp.real, -gp.imag], [gv.real, -gv.imag]])
        out.append(dict(float=j + 1, peak_distance=float(np.linalg.svd(M, compute_uv=False)[0]),
                        mean_squared_distance=float((abs(gp) ** 2 + abs(gv) ** 2) / 2)))
    return out


class NoThrottle(RC.Controller):
    name = "no throttle"

    def __call__(self, x):
        return np.zeros(np.asarray(x).shape[:-1] + (TC.DA,))


class StreamedParts(RC.Controller):
    """The recording controller on a bank with one or both of its departures from normal driving switched on:
    wobble (noise on normal driving, jitter on the full-throttle command) and the full-throttle command (coin)."""

    def __init__(self, rec, noise, coin, wobble, full_throttle):
        self.rec, self.noise, self.coin = rec, noise, coin
        self.wobble, self.full_throttle = wobble, full_throttle
        self.name = f"recording controller, wobble={wobble}, full_throttle={full_throttle}"

    def act(self, x, t):
        return self.rec(x, self.noise[:, t] if self.wobble else None, self.coin[:, t] if self.full_throttle else None)


# ================================================================================================ (1) signs
log("signs")
x_rest = np.zeros(TC.DX)
x_rest[TC.CURRENT] = 1.0                            # current 1, wave 0, floats at rest
x1, _ = TC.dynamics(x_rest, np.zeros(TC.DA))
x2, _ = TC.dynamics(x1, np.zeros(TC.DA))
x_still = np.zeros(TC.DX)                           # no water at all, floats at rest
th1 = {}
for label, cmd, phys in (("stall_full", -1.0, TC.STALL), ("no_stall_full", -1.0, TC.NO_STALL), ("at_80", -0.8, TC.STALL),
                         ("stall_90", -0.9, TC.STALL)):
    a = np.zeros(TC.DA)
    a[0] = cmd
    nx, _ = TC.dynamics(x_still, a, phys)
    th1[label] = dict(command=cmd, speed_change_float1=nx[1], speed_change_float2=nx[3],
                      other_floats_unchanged=bool(np.all(nx[[0, 2, 4, 5, 6, 7, 8, 9]] == 0)))
check(np.all(x1[1:TC.DS:2] > 0) and np.all(x1[0:TC.DS:2] == 0), "one step of water from rest raises every speed number")
check(np.all(x2[0:TC.DS:2] < 0), "two steps of water from rest lower every position number")
check(np.all(Kc > 0), "K's current column is positive, so the textbook hold is negative")
check(P[0] == -1.0, "the full-throttle pattern puts thruster 1 at -1")
tmpl = TEMPLATE.read_text()
panel_rules = dict(
    gauges="each thruster's throttle, -1 (full throttle against the water) at the top; labels '↑ against the water' and "
           "'↓ with the water'",
    floats="a float is drawn below the rest line when its position number is negative (downstream, where the water "
           "carries it); the small arrow beside it shows its speed number, pointing down for a positive speed number",
    thruster_arrows="the arrow at each thruster shows the push it really gives; a negative push points up (upstream)",
    channel="water flows down the channel ('downstream ↓'); the push meter shows c + w1 with the 0.565 mark",
    found_in_template=all(s in tmpl for s in ("−1 (full throttle against the water) at the top",
                                              "draw negative positions below rest (downstream)",
                                              "A negative throttle pushes upstream, against the water")))
check(panel_rules["found_in_template"], "the live panel's sign wording is still in template.html")
data["signs"] = dict(
    how="one exact step of thruster_chain.dynamics() from rest; K and P from recorder.py; panel rules read from "
        "template.html (drawing code of the channel and gauges)",
    water_one_step_from_rest_current_1=dict(speeds=x1[1:TC.DS:2], positions=x1[0:TC.DS:2]),
    water_two_steps_from_rest_current_1=dict(positions=x2[0:TC.DS:2], speeds=x2[1:TC.DS:2]),
    water_push_per_step_on_each_speed_number_per_unit_push=GAIN,
    thruster1_one_step_from_rest=th1,
    textbook_hold_per_unit_current=-Kc,
    full_throttle_pattern_thruster1=P[0],
    summary=("the water's push c + w1, when positive, raises every float's speed number (downstream) and the swing then "
             "turns it into a negative position number; a negative throttle lowers the speed numbers of the two floats "
             "a thruster sits between (upstream, against the water); the textbook hold -K_e e is negative for a "
             "positive current; the full-throttle command puts thruster 1 at -1"),
    panel=panel_rules,
)

# ================================================================================================ (2) who pushes whom
log("who pushes whom")
alt = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
alt_B = alt @ B5
check(np.all(alt_B == 0), "every thruster cancels out of the alternating sum of the floats' pushes")
f_ls, *_ = np.linalg.lstsq(B5, -np.ones(5), rcond=None)
left_ls = B5 @ f_ls + 1.0
# closed-loop steady state of the uncapped textbook rule, steady current 1, no wave (linear, exact)
A_cl = A10 - B10 @ Ks
s_K = np.linalg.solve(I10 - A_cl, GAIN * v - B10 @ Kc)
a_K = -(Ks @ s_K + Kc)
left_K = (v + B10 @ a_K / GAIN)[1::2]
# steady-state cost-optimal constant throttle (floats settled, cost |s|^2 + 0.3 |a|^2 per step)
Minv = np.linalg.inv(I10 - A10)
Mb, nv = Minv @ B10, Minv @ (GAIN * v)
a_opt = -np.linalg.solve(Mb.T @ Mb + TC.STALL.throttle_cost * np.eye(TC.DA), Mb.T @ nv)
s_opt = Mb @ a_opt + nv
left_opt = (v + B10 @ a_opt / GAIN)[1::2]
# check the closed-form steady state by simulation of the no-stall world (thrust = command), current 1, wave 0
xs = np.zeros(TC.DX)
xs[TC.CURRENT] = 1.0
rt_ns = RC.rising_thrust(TC.NO_STALL)
lin = RC.LinearController(K, None, "K uncapped")
for _ in range(4000):
    xs, _ = TC.dynamics(xs, lin(xs), TC.NO_STALL)
check(np.allclose(xs[:TC.DS], s_K, atol=1e-10) and np.allclose(lin(xs), a_K, atol=1e-10),
      "simulated steady state equals the closed form")
c80_closed = CAP / abs(a_K[0])
c100_closed = 1.0 / abs(a_K[0])
c80_hold = PD["recording_controller"]["current_where_textbook_hold_reaches_80pct_thruster1"]
check(abs(c80_hold - CAP / Kc[0]) < 1e-12, "page_data's 0.879 is 0.8 / K's current entry for thruster 1")
load_order = [int(k) + 1 for k in np.argsort(-np.abs(a_K))]
data["who_pushes"] = dict(
    how="linear algebra on thruster_chain.matrices(): B5 = which floats' speed numbers each thruster pushes; "
        "least squares min |B5 f + 1|; the textbook rule's closed-loop steady state s = (I - A + 0.3 B K_s)^-1 "
        "(0.3 v - 0.3 B K_c) c with the floats settled (no cap, no wave), checked by a 4,000-step simulation; the "
        "steady-state cost-optimal constant throttle minimises |s|^2 + 0.3 |a|^2 over settled floats",
    thruster_reaches_floats={f"thruster_{k + 1}": [k + 1, k + 2] for k in range(4)},
    alternating_sum=dict(
        signs=alt, thruster_contribution=alt_B,
        statement="push on float 1 - push on float 2 + float 3 - float 4 + float 5: each thruster adds the same push "
                  "to two neighbouring floats, so it adds once with + and once with - and cancels; the water adds "
                  "1 - 1 + 1 - 1 + 1 = 1 times its push. So no throttles can cancel the water's push on all five "
                  "floats: this alternating sum of the pushes left over is always the water's push itself"),
    least_squares=dict(throttles_per_unit_push=f_ls, leftover_push_per_float=left_ls,
                       leftover_sum_of_squares=float(left_ls @ left_ls), no_throttle_sum_of_squares=5.0,
                       leftover_share_of_sum_of_squares=float(left_ls @ left_ls) / 5.0,
                       note="balancing the pushes alone (ignoring how far each push moves its float): thrusters 1 "
                            "and 4 carry the most, 0.8 per unit push; every float keeps 0.2 of the push, floats 1, 3 "
                            "and 5 downstream and floats 2 and 4 upstream"),
    textbook_steady=dict(throttles_per_unit_current=a_K, leftover_push_per_float=left_K,
                         float_offsets_per_unit_current=per_float(s_K),
                         float_cost_per_unit_current_squared=float(s_K @ s_K),
                         throttle_cost_per_unit_current_squared=float(TC.STALL.throttle_cost * a_K @ a_K),
                         thrusters_by_load=load_order,
                         current_where_thruster1_reaches_80pct=c80_closed,
                         current_where_thruster1_reaches_full_throttle=c100_closed,
                         note="what K holds once the floats have settled in a steady current with no wave and no cap "
                              "(the no-stall world, or the stalling world while every throttle stays within 80%). "
                              "page_data's 0.879 and 1.099 are K's current column alone, with the floats at rest; "
                              "with the floats settled the feedback adds to thruster 1, so 80% is reached already at "
                              "this current"),
    steady_optimal=dict(throttles_per_unit_current=a_opt, leftover_push_per_float=left_opt,
                        float_cost_per_unit_current_squared=float(s_opt @ s_opt),
                        throttle_cost_per_unit_current_squared=float(TC.STALL.throttle_cost * a_opt @ a_opt),
                        largest_difference_from_textbook_steady=float(np.abs(a_opt - a_K).max())),
)

# ================================================================================================ (3) the water alone
log("the water alone: linear responses")
n_c = np.linalg.solve(I10 - A10, GAIN * v)                     # steady offset per unit current, no throttle
G_w = np.linalg.solve(np.exp(1j * TURN) * I10 - A10, GAIN * v)  # steady response to w1 = cos(0.27 t)
# check both by simulation (no throttle): current 1 for 4,000 steps; wave amplitude 1 after 4,000 steps
xc = np.zeros(TC.DX)
xc[TC.CURRENT] = 1.0
xw = np.zeros(TC.DX)
xw[TC.WAVE1] = 1.0
zero = np.zeros(TC.DA)
peak_sim = np.zeros(5)
for t in range(4000 + 100):
    if t >= 4000:
        peak_sim = np.maximum(peak_sim, np.hypot(xw[0:TC.DS:2], xw[1:TC.DS:2]))
    xc, _ = TC.dynamics(xc, zero)
    xw, _ = TC.dynamics(xw, zero)
check(np.allclose(xc[:TC.DS], n_c, atol=1e-10), "simulated steady offset equals (I - A)^-1 0.3 v")
wave_peaks = swing_peak(G_w)
check(all(abs(peak_sim[j] - wave_peaks[j]["peak_distance"]) < 2e-3 for j in range(5)),
      "simulated wave swing equals the complex steady response")
mean_c = {s: float(np.mean(TC.START_LAWS[s]["current"])) for s in SITS}

log("the water alone and every reference controller on the banks")
CONTROLLERS = ("no_throttle", "recording", "normal_driving_only", "stall_aware", "rising_thrust")
runs = {}
for s in SITS:
    for name in CONTROLLERS:
        pol = dict(no_throttle=NoThrottle(), recording=TR.recorder_on_bank(TC.STALL), normal_driving_only=RC.hand_only(),
                   stall_aware=RC.stall_aware(), rising_thrust=RC.rising_thrust())[name]
        runs[(s, name)] = TR.bank_rollout(pol, s, TC.STALL, record=True)
        ref = PD["reference_returns"]["returns"]["stall"][s][name]["J"]
        check(abs(runs[(s, name)].J.mean() - ref) < 5e-5, f"bank return of {name} in {s} matches page_data")
cost_split = {}
for s in SITS:
    cost_split[s] = {}
    for name in CONTROLLERS:
        run = runs[(s, name)]
        X, C = run.states[:, :-1], run.commands
        fc = np.sum(X[..., :TC.DS] ** 2, -1)
        tc = TC.STALL.throttle_cost * np.sum(C ** 2, -1)
        check(abs((fc + tc).sum(1).mean() + run.J.mean()) < 1e-8, "float cost + throttle cost = minus the return")
        cost_split[s][name] = dict(J=float(run.J.mean()), cost_per_state=float((fc + tc).mean()),
                                   float_cost_per_state=float(fc.mean()), throttle_cost_per_state=float(tc.mean()),
                                   float_share_of_cost=float(fc.mean() / (fc + tc).mean()))
for s in SITS:
    nt, sa = cost_split[s]["no_throttle"], cost_split[s]["stall_aware"]
    cost_split[s]["no_throttle_over_stall_aware"] = nt["cost_per_state"] / sa["cost_per_state"]
    cost_split[s]["stall_aware_float_cost_removed_share"] = 1 - sa["float_cost_per_state"] / nt["float_cost_per_state"]
data["water_alone"] = dict(
    how="no throttle: (I - A)^-1 0.3 v is where the floats settle per unit of steady current; (e^{0.27 i} I - A)^-1 "
        "0.3 v is their steady swing under a wave of amplitude 1 (largest distance from rest over a cycle); both "
        "checked by 4,000-step simulations. Costs: exact float64 rollouts on the four evaluation banks (1,000 "
        "starts x 100 steps each), averaged over every visited state",
    steady_offset_per_unit_current=per_float(n_c),
    steady_float_cost_per_unit_current_squared=float(n_c @ n_c),
    steady_offset_at_mean_current={s: dict(current=mean_c[s], float1_position=n_c[0] * mean_c[s],
                                           float1_distance=math.hypot(n_c[0], n_c[1]) * mean_c[s],
                                           float_cost=float(n_c @ n_c) * mean_c[s] ** 2) for s in SITS},
    all_positions_negative_all_speeds_positive=bool(np.all(n_c[0::2] < 0) and np.all(n_c[1::2] > 0)),
    wave_swing_per_unit_amplitude=wave_peaks,
    wave_float_cost_per_unit_amplitude_squared=float(np.sum(np.abs(G_w) ** 2) / 2),
    wave_swing_float2_at_amplitude={f"{a:g}": wave_peaks[1]["peak_distance"] * a for a in (0.1, 0.25, 0.4, 0.55, 0.7)},
    cost_split=cost_split,
)

# ================================================================================================ (4) the waves
log("the waves")
waves = dict(how="the water's push c + w1 at every visited state of the four banks (the same for every controller, "
                 "since no thruster changes the water); start-law endpoints; stage0.bisect_trigger rerun",
             quarter_cycle_steps=PD["environment"]["dynamics"]["wave_period_steps"] / 4,
             cycles_per_episode=TC.STALL.length / PD["environment"]["dynamics"]["wave_period_steps"],
             situations={})
for s in SITS:
    law = TC.START_LAWS[s]
    x0 = TC.bank_starts(s)
    c, amp = x0[:, TC.CURRENT], TC.wave_amplitude(x0)
    push = TC.push(runs[(s, "stall_aware")].states[:, :-1])
    push_nt = TC.push(runs[(s, "no_throttle")].states[:, :-1])
    check(np.array_equal(push, push_nt), "the water's push does not depend on the controller")
    reg = PD["reference_returns"]["returns"]["stall"][s]["stall_aware"]["full_throttle_region_steps"]
    check(abs((push > RC.TRIGGER).mean() - reg) < 5e-5, "region share matches page_data")
    waves["situations"][s] = dict(
        plain=PLAIN_SITUATION[s],
        push_lowest_possible=law["current"][0] - law["wave"][1], push_highest_possible=law["current"][1] + law["wave"][1],
        push_lowest_seen=float(push.min()), push_highest_seen=float(push.max()), push_mean=float(push.mean()),
        negative_push_share_of_states=float((push < 0).mean()),
        starts_whose_crest_passes_the_mark=float(np.mean(c + amp > RC.TRIGGER)),
        starts_always_past_the_mark=float(np.mean(c - amp > RC.TRIGGER)),
        starts_never_past_the_mark=float(np.mean(c + amp <= RC.TRIGGER)),
        starts_wave_bigger_than_current=float(np.mean(amp > c)),
        mean_current=float(c.mean()), mean_amplitude=float(amp.mean()))
trigger, history = S0.bisect_trigger()
check(trigger == RC.TRIGGER, "the bisection still returns 0.565")
waves["trigger"] = dict(
    value=RC.TRIGGER, share_target=RC.DEFAULT_SPEC.full_throttle_share_target, bisection_result=trigger,
    bisection_last_share=history[-1]["full_throttle_share"], bisection_last_midpoint=history[-1]["trigger"],
    bisection_settings=dict(S0.BISECTION, seeds=list(S0.BISECTION_SEEDS)),
    textbook_hold_thruster1_when_current_alone_is_at_mark=Kc[0] * RC.TRIGGER,
    textbook_steady_thruster1_when_current_alone_is_at_mark=abs(a_K[0]) * RC.TRIGGER,
    why="set by bisection (14 halvings of [0, 1.2] on design seeds 900-902, 300 episodes each) so that the recording "
        "controller gives the full-throttle command on 13% of steps; it is a calibration of how common that command is "
        "in the log, not the point where holding needs 80%: a push of 0.565 from the current alone asks thruster 1 "
        "for about half throttle")
data["waves"] = waves

# ================================================================================================ (5) the swings
log("the swings")
ev, V = np.linalg.eig(A10)
keep = [i for i in np.argsort(np.angle(ev)) if np.angle(ev[i]) > 0]
modes = PD["environment"]["dynamics"]["oscillator_modes"]
swings = []
for j, i in enumerate(keep):
    w = np.abs(V[:, i]) ** 2
    w = w / w.sum()
    own = [w[2 * f] + w[2 * f + 1] for f in range(5)]
    rho, th = abs(ev[i]), float(np.angle(ev[i]))
    check(abs(rho - modes[j]["radius"]) < 1e-6 and abs(th - modes[j]["angle_rad"]) < 1e-6,
          "swing order matches page_data's oscillator_modes")
    swings.append(dict(swing=j + 1, float_with_most_weight=int(np.argmax(own)) + 1, weight_on_that_float=max(own),
                       fade_per_step=rho, turn_per_step=th, period_steps=2 * math.pi / th,
                       half_life_steps=math.log(0.5) / math.log(rho), left_after_100_steps=rho ** 100,
                       turn_gap_to_wave=abs(th - TURN), period_gap_to_wave=abs(2 * math.pi / th - 2 * math.pi / TURN),
                       steady_offset_per_unit_current=per_float(n_c)[j]["distance"],
                       wave_swing_per_unit_amplitude=wave_peaks[j]["peak_distance"]))
check([s_["float_with_most_weight"] for s_ in swings] == [1, 2, 3, 4, 5], "swing j lives on float j")
closest = min(swings, key=lambda s_: s_["turn_gap_to_wave"])["swing"]
most_wave = max(swings, key=lambda s_: s_["wave_swing_per_unit_amplitude"])["swing"]
most_current = max(swings, key=lambda s_: s_["steady_offset_per_unit_current"])["swing"]
data["swings"] = dict(
    how="eigenvalues of A (the 10 x 10 float dynamics) as on the page's Swing table, eigenvector weight per float, "
        "half-life ln 0.5 / ln(fade), steady responses from section water_alone",
    list=swings, closest_turn_to_wave=closest, strongest_wave_response=most_wave,
    strongest_current_response=most_current,
    lowest_weight_on_own_float=min(s_["weight_on_that_float"] for s_ in swings),
    wave_turn=TURN, wave_period_steps=PD["environment"]["dynamics"]["wave_period_steps"])

# ================================================================================================ (6) the controllers
log("the controllers: closed-loop wave response")


def wave_closed_loop(Ks_, Kw_):
    rhs = GAIN * v - B10 @ (Kw_[:, 0] - 1j * Kw_[:, 1])
    S = np.linalg.solve(np.exp(1j * TURN) * I10 - A10 + B10 @ Ks_, rhs)
    a = -(Ks_ @ S + Kw_[:, 0] - 1j * Kw_[:, 1])
    return dict(float_swing_per_unit_amplitude=[p["peak_distance"] for p in swing_peak(S)],
                float_cost_per_unit_amplitude_squared=float(np.sum(np.abs(S) ** 2) / 2),
                throttle_swing_per_unit_amplitude=np.abs(a),
                throttle_lead_steps_against_push=np.angle(-a) / TURN,
                throttle_cost_per_unit_amplitude_squared=float(TC.STALL.throttle_cost * np.sum(np.abs(a) ** 2) / 2))


wave_K = wave_closed_loop(Ks, Kw)
wave_nd = wave_closed_loop(RC.DEFAULT_SPEC.damping * Ks, Kw)
ff = [dict(thruster=k + 1, current_hold=Kc[k], wave_hold_size=math.hypot(Kw[k, 0], Kw[k, 1]),
           wave_hold_lead_steps=math.atan2(-Kw[k, 1], Kw[k, 0]) / TURN) for k in range(4)]
# check the closed-loop wave response by simulation (uncapped K, no-stall world, wave amplitude 1, no current)
xw = np.zeros(TC.DX)
xw[TC.WAVE1] = 1.0
peak = np.zeros(5)
amax = np.zeros(4)
for t in range(3000 + 100):
    a = lin(xw)
    if t >= 3000:
        peak = np.maximum(peak, np.hypot(xw[0:TC.DS:2], xw[1:TC.DS:2]))
        amax = np.maximum(amax, np.abs(a))
    xw, _ = TC.dynamics(xw, a, TC.NO_STALL)
check(np.allclose(peak, wave_K["float_swing_per_unit_amplitude"], atol=2e-3)
      and np.allclose(amax, wave_K["throttle_swing_per_unit_amplitude"], atol=2e-3),
      "simulated closed-loop wave response equals the complex steady state")

log("the controllers: per situation on the banks")
CREST = 0.9  # a crest (trough) is a state where the wave adds (takes away) at least 90% of its amplitude
sa_stats, rt_stats = {}, {}
for s in SITS:
    run = runs[(s, "stall_aware")]
    X, C = run.states[:, :-1], run.commands
    push = TC.push(X)
    a1 = C[..., 0]
    atcap = np.abs(C) == CAP
    check(not np.any(np.abs(C) > CAP), "the stall-aware controller never passes 80%")
    amp = TC.wave_amplitude(X)
    crest = X[..., TC.WAVE1] >= CREST * amp
    trough = X[..., TC.WAVE1] <= -CREST * amp
    region = push > RC.TRIGGER
    slope, intercept = np.polyfit(push.ravel(), a1.ravel(), 1)
    sa_stats[s] = dict(
        at_cap_any_thruster=float(atcap.any(-1).mean()), at_cap_thruster1=float(atcap[..., 0].mean()),
        at_cap_by_thruster=atcap.mean((0, 1)),
        thruster1_vs_push_correlation=float(np.corrcoef(a1.ravel(), push.ravel())[0, 1]),
        thruster1_vs_push_slope=float(slope), thruster1_vs_push_intercept=float(intercept),
        crest=dict(states=int(crest.sum()), push=float(push[crest].mean()), thruster1=float(a1[crest].mean()),
                   at_cap_thruster1=float(atcap[..., 0][crest].mean())),
        trough=dict(states=int(trough.sum()), push=float(push[trough].mean()), thruster1=float(a1[trough].mean()),
                    at_cap_thruster1=float(atcap[..., 0][trough].mean())),
        at_cap_share_inside_region=float(atcap.any(-1)[region].mean()) if region.any() else None,
        region_share_of_capped_states=float(region[atcap.any(-1)].mean()) if atcap.any() else None,
        mean_throttle_by_thruster=C.mean((0, 1)))
    run = runs[(s, "rising_thrust")]
    C = run.commands
    past = np.abs(C) > TC.KNEE
    th = TC.thrust(C, TC.STALL)
    lost = np.abs(C) - np.abs(th)
    check(abs(past.any(-1).mean() - PD["reference_returns"]["returns"]["stall"][s]["rising_thrust"]["steps_past_80"])
          < 5e-5, "rising-thrust past-80% share matches page_data")
    rt_stats[s] = dict(
        past_80_thruster1=float(past[..., 0].mean()), past_80_by_thruster=past.mean((0, 1)),
        push_lost_per_stalled_thruster_state=float(lost[past].mean()) if past.any() else 0.0,
        push_lost_per_state=float(lost.sum(-1).mean()),
        thruster1_throttle_when_past=float(np.abs(C[..., 0][past[..., 0]]).mean()) if past[..., 0].any() else None,
        thruster1_push_when_past=float(np.abs(th[..., 0][past[..., 0]]).mean()) if past[..., 0].any() else None,
        extra_cost_against_stall_aware=cost_split[s]["stall_aware"]["J"] - cost_split[s]["rising_thrust"]["J"])

log("the controllers: thruster 1 by the water's push (pooled over the four banks)")
edges = np.round(np.arange(-0.6, 1.6001, 0.2), 10)
pooled = {name: [] for name in ("push", "sa", "rt", "rt_thrust")}
for s in SITS:
    pooled["push"].append(TC.push(runs[(s, "stall_aware")].states[:, :-1]).ravel())
    pooled["sa"].append(runs[(s, "stall_aware")].commands[..., 0].ravel())
    c_rt = runs[(s, "rising_thrust")].commands[..., 0].ravel()
    pooled["rt"].append(c_rt)
    pooled["rt_thrust"].append(TC.thrust(c_rt, TC.STALL))
pooled = {k: np.concatenate(v_) for k, v_ in pooled.items()}
by_push = []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (pooled["push"] >= lo) & (pooled["push"] < hi)
    if not m.any():
        continue
    by_push.append(dict(push_from=lo, push_to=hi, states=int(m.sum()),
                        stall_aware_thruster1=float(pooled["sa"][m].mean()),
                        stall_aware_thruster1_at_cap=float((np.abs(pooled["sa"][m]) == CAP).mean()),
                        rising_thrust_thruster1=float(pooled["rt"][m].mean()),
                        rising_thrust_thruster1_past_80=float((np.abs(pooled["rt"][m]) > TC.KNEE).mean()),
                        rising_thrust_thruster1_push=float(pooled["rt_thrust"][m].mean())))
check(sum(b["states"] for b in by_push) == len(pooled["push"]), "every pooled state falls in a bin")

log("the controllers: steady states in a steady current")
currents = np.array([0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4])
steady = {}
for name, pol, phys in (("no_throttle", NoThrottle(), TC.STALL), ("stall_aware", RC.stall_aware(), TC.STALL),
                        ("rising_thrust", RC.rising_thrust(), TC.STALL),
                        ("rising_thrust_no_stall_world", rt_ns, TC.NO_STALL)):
    x = np.zeros((len(currents), TC.DX))
    x[:, TC.CURRENT] = currents
    tail_fc, tail_a, tail_th1, tail_tc, tail_x = [], [], [], [], []
    for t in range(4000):
        a = np.clip(pol(x), -1.0, 1.0)
        if t >= 3800:
            tail_fc.append(np.sum(x[:, :TC.DS] ** 2, -1))
            tail_a.append(a)
            tail_th1.append(TC.thrust(a, phys)[:, 0])
            tail_tc.append(TC.STALL.throttle_cost * np.sum(a ** 2, -1))
            tail_x.append(x[:, :TC.DS])
        x, _ = TC.dynamics(x, a, phys)
    tail_x, tail_a, tail_fc = np.stack(tail_x), np.stack(tail_a), np.stack(tail_fc)
    settled = np.abs(tail_x - tail_x[-1]).max((0, 2)) < 1e-9
    steady[name] = [dict(current=float(c), float_cost=float(tail_fc[:, i].mean()),
                         float_cost_lowest=float(tail_fc[:, i].min()), float_cost_highest=float(tail_fc[:, i].max()),
                         thruster1=float(tail_a[:, i, 0].mean()), throttle_lowest=tail_a[:, i].min(0),
                         throttle_highest=tail_a[:, i].max(0), thruster1_push=float(np.mean(t_)),
                         throttle_cost=float(np.mean(tc_)), settled=bool(st))
                    for i, (c, t_, tc_, st) in enumerate(zip(currents, np.stack(tail_th1).T, np.stack(tail_tc).T,
                                                             settled))]
check(all(r["settled"] for r in steady["stall_aware"]), "the stall-aware controller settles in every steady current")
data["controllers"] = dict(
    how="the textbook rule's hold terms from K; its closed-loop steady response to a wave of amplitude 1 (complex "
        "steady state of the linear closed loop, no cap, checked by simulation); exact bank rollouts of the "
        "stall-aware and rising-thrust controllers (stalling world); crest = states with w1 >= 0.9 x amplitude, "
        "trough = w1 <= -0.9 x amplitude; steady states: 4,000 steps from rest in a steady current with no wave, "
        "averaged over the last 200 (settled = the floats no longer move; where they do not, a thruster keeps "
        "crossing into the stall and back, and the lowest and highest values of that repeating cycle are given)",
    textbook_rule=("K is the discounted (0.98) linear-quadratic regulator of the model in which push equals throttle: "
                   "command = -(K_s s + K_e e), a fixed table that pushes against the water's current and wave "
                   "(K_e e, using both wave numbers, so it times its push to the wave) and against the floats' "
                   "distance from rest (K_s s)"),
    crest_share_of_amplitude=CREST,
    hold_terms=ff,
    wave_response=dict(textbook_rule=wave_K, normal_driving=wave_nd,
                       no_throttle_float_swing_per_unit_amplitude=[p["peak_distance"] for p in wave_peaks],
                       no_throttle_float_cost_per_unit_amplitude_squared=data["water_alone"][
                           "wave_float_cost_per_unit_amplitude_squared"]),
    stall_aware=sa_stats, rising_thrust=rt_stats, thruster1_by_push=by_push,
    steady_current=dict(currents=currents, table=steady),
)

# ================================================================================================ (7) starting pushes
log("starting pushes")
xr = TC.bank_starts("recorded")
starts = dict(how="the bank starts themselves (shared seed: start i of every bank uses the same random draws); "
                  "half-lives ln 0.5 / ln |eigenvalue| of A (no throttle), of A - 0.3 B K_s (the textbook rule's "
                  "float feedback, no cap) and of A - 0.3 B 0.25 K_s (normal driving); paired bank return "
                  "differences",
              situations={})
for s in SITS:
    x0 = TC.bank_starts(s)
    d0 = TC.float_radius(x0)
    starts["situations"][s] = dict(float_start_sd=TC.START_LAWS[s]["kick"], mean_start_distance=float(d0.mean()),
                                   mean_start_float_cost=float(np.mean(d0 ** 2)),
                                   largest_start_distance=float(d0.max()))
xk, xsc, xbw = TC.bank_starts("big_kicks"), TC.bank_starts("strong_current"), TC.bank_starts("big_waves")
u = (xr[:, TC.CURRENT] - 0.1) / 0.6
starts["bank_relations"] = dict(
    big_kicks_floats_are_three_times_recorded=bool(np.array_equal(xk[:, :TC.DS], 3 * xr[:, :TC.DS])),
    big_kicks_water_equals_recorded=bool(np.array_equal(xk[:, TC.DS:], xr[:, TC.DS:])),
    strong_current_floats_and_wave_equal_recorded=bool(np.array_equal(xsc[:, :TC.DS], xr[:, :TC.DS])
                                                       and np.array_equal(xsc[:, TC.WAVE1:], xr[:, TC.WAVE1:])),
    strong_current_current_map_max_error=float(np.abs(xsc[:, TC.CURRENT] - (0.7 + 0.4 * u)).max()),
    big_waves_amplitude_minus_recorded=float(np.mean(TC.wave_amplitude(xbw) - TC.wave_amplitude(xr))),
    big_waves_amplitude_map_max_error=float(np.abs(TC.wave_amplitude(xbw) - TC.wave_amplitude(xr) - 0.3).max()),
    big_waves_floats_and_current_equal_recorded=bool(np.array_equal(xbw[:, :TC.CURRENT + 1], xr[:, :TC.CURRENT + 1])))
check(starts["bank_relations"]["big_kicks_floats_are_three_times_recorded"], "big-kick starts are 3x recorded")


def half_lives(M):
    e = np.linalg.eigvals(M)
    e = e[np.argsort(-np.abs(e))]
    out, seen = [], set()
    for z in e:
        key = (round(abs(z), 9), round(abs(np.angle(z)), 9))
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(fade_per_step=abs(z), turn_per_step=abs(np.angle(z)),
                        half_life_steps=math.log(0.5) / math.log(abs(z))))
    return out


hl_K = half_lives(A10 - B10 @ Ks)
hl_nd = half_lives(A10 - RC.DEFAULT_SPEC.damping * B10 @ Ks)
starts["half_life"] = dict(no_throttle_by_swing=[s_["half_life_steps"] for s_ in swings],
                           no_throttle_slowest=swings[0]["half_life_steps"],
                           textbook_feedback_slowest=hl_K[0]["half_life_steps"],
                           textbook_feedback_modes=hl_K,
                           normal_driving_slowest=hl_nd[0]["half_life_steps"], normal_driving_modes=hl_nd)
starts["extra_cost_of_big_starts"] = {name: cost_split["recorded"][name]["J"] - cost_split["big_kicks"][name]["J"]
                                      for name in CONTROLLERS}   # positive: how much more the 100 steps cost
starts["stall_aware_at_cap"] = dict(recorded=sa_stats["recorded"]["at_cap_any_thruster"],
                                    big_kicks=sa_stats["big_kicks"]["at_cap_any_thruster"])
data["starts"] = starts

# ================================================================================================ (8) recording controller
log("recording controller on the banks")
rec = RC.Recorder()
streams = RC.StreamedRecorder.on_bank(rec)
recording = dict(how="the recording controller with its fixed per-bank streams (default_rng([7, 1]): noise, then "
                     "coin), exact rollouts on the four banks; the full-throttle command is given where c + w1 > "
                     "0.565 and the coin is below 0.5; 'wobble only' and 'full-throttle command only' switch one "
                     "departure from normal driving off; 'neither' is normal driving only",
                 normal_driving_share_train=1 - PD["dataset"]["full_throttle_share_train"],
                 situations={})
for s in SITS:
    x = TC.bank_starts(s).copy()
    press_all, region_all, J = [], [], np.zeros(len(x))
    for t in range(TC.STALL.length):
        cmd, region, press = rec.parts(x, streams.noise[:, t], streams.coin[:, t])
        x, r = TC.dynamics(x, np.clip(cmd, -1.0, 1.0))
        J += r
        press_all.append(press)
        region_all.append(region)
    press_all, region_all = np.stack(press_all, 1), np.stack(region_all, 1)
    check(np.array_equal(J, runs[(s, "recording")].J), "the recording controller replays its bank rollout exactly")
    parts_J = {}
    for label, wob, ft in (("wobble_only", True, False), ("full_throttle_only", False, True)):
        parts_J[label] = float(TR.bank_rollout(StreamedParts(rec, streams.noise, streams.coin, wob, ft), s).J.mean())
    both, neither = cost_split[s]["recording"]["J"], cost_split[s]["normal_driving_only"]["J"]
    recording["situations"][s] = dict(
        full_throttle_share=float(press_all.mean()), region_share=float(region_all.mean()),
        full_throttle_share_inside_region=float(press_all[region_all].mean()) if region_all.any() else None,
        J_both=both, J_wobble_only=parts_J["wobble_only"], J_full_throttle_only=parts_J["full_throttle_only"],
        J_neither=neither, cost_of_wobble_alone=neither - parts_J["wobble_only"],
        cost_of_full_throttle_alone=neither - parts_J["full_throttle_only"], cost_of_both=neither - both)
recording["wobble_sd"] = RC.DEFAULT_SPEC.wobble
recording["jitter_sd"] = RC.DEFAULT_SPEC.jitter
recording["full_throttle_probability"] = RC.DEFAULT_SPEC.full_throttle_probability
data["recording"] = recording

# ================================================================================================ (9) interface
log("interface")
sig = inspect.signature(TC.ThrusterChainEnv.__init__)
ctor = [dict(name=p.name, default=p.default) for p in sig.parameters.values() if p.name != "self"]
env = gym.make(TC.env_id("stall", "recorded"))
obs, info_reset = env.reset(seed=0)
obs1, rew1, term1, trunc1, info_step = env.step(np.zeros(TC.DA, np.float32))
unknown_refused = False
try:
    env.reset(options={"bogus": 1})
except ValueError:
    unknown_refused = True
env.close()
spec = gym.spec(TC.env_id("stall", "recorded"))
data["interface"] = dict(
    how="inspect.signature(ThrusterChainEnv.__init__); one reset(seed=0) and one step through gymnasium.make; the "
        "gymnasium registry",
    constructor=ctor,
    constructor_line="ThrusterChainEnv(" + ", ".join(f"{c['name']}={c['default']!r}" for c in ctor) + ")",
    reset_info_keys=list(info_reset.keys()), step_info_keys=list(info_step.keys()),
    step_info_types={k: (f"{type(v_).__name__} {getattr(v_, 'dtype', '')} {list(getattr(v_, 'shape', []))}").strip()
                     for k, v_ in info_step.items()},
    reset_options=["state", "start"], unknown_option_refused=unknown_refused,
    observation_dtype=str(obs.dtype), observation_shape=list(obs.shape), action_dtype=str(env.action_space.dtype),
    step_returns=["observation", "reward", "terminated", "truncated", "info"],
    terminated_after_one_step=bool(term1), truncated_after_one_step=bool(trunc1),
    max_episode_steps=spec.max_episode_steps, entry_point=spec.entry_point,
    registered_ids=list(TC.ENV_IDS), registered_count=len(TC.ENV_IDS),
    metadata=dict(TC.ThrusterChainEnv.metadata))

# ================================================================================================ meta
HASH_END = file_hashes()
if HASH_END != HASH_START:
    raise SystemExit("a file in experiments/shift changed while this script ran; refusing to write")
data["meta"] = dict(
    script=str(Path(__file__).resolve()), created=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=round(time.time() - T0),
    python=platform.python_version(), numpy=np.__version__, gymnasium=gym.__version__,
    flags={k: os.environ.get(k) for k in ("JAX_PLATFORMS", "OMP_NUM_THREADS", "PYTHONDONTWRITEBYTECODE")},
    nothing_trains="no learner is imported or trained; every number is exact simulation of fixed controllers, "
                   "linear algebra on the environment's own matrices, or the bank starts themselves",
    banks=dict(size=TC.BANK_SIZE, seed=list(TC.BANK_SEED), recording_streams=list(RC.RECORDING_BANK_STREAM)),
    discount=TC.STALL.gamma,
    experiments_shift_sha256=HASH_START,
    page_data_sha256=hashlib.sha256(PAGE_DATA.read_bytes()).hexdigest(),
)
OUT.write_text(json.dumps(rnd(data, 6), indent=1, allow_nan=False, ensure_ascii=False))
log(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.1f} kB)")
