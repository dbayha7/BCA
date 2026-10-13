"""Every number and rollout the ThrusterChain environment page quotes, computed from the environment's own code.

Run from the repository root (read-only; nothing under /home/user/bca is written, nothing trains):

  cd /home/user/bca && PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 nice -n 10 \
      /tmp/claude-0/bcavenv/bin/python <this file>

Writes page_data.json beside this file. Only experiments.shift.{thruster_chain, recorder, truth, dataset, shift,
stage0} are imported (stage0 only for its outcome-free world card and arithmetic helpers). learners.py and
acceptance.py are never imported: their fixed settings and pass marks are read as literals from the source text with
the ast module. The SHA-256 of every file in experiments/shift is taken at the start and at the end, and the script
fails if any changed.
"""

import ast
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

REPO = Path("/home/user/bca")
HERE = Path(__file__).resolve().parent
OUT = HERE / "page_data.json"
sys.path.insert(0, str(REPO))
if Path.cwd().resolve() != REPO:
    raise SystemExit("run from the repository root, /home/user/bca")

import numpy as np  # noqa: E402

import experiments.shift as PKG  # noqa: E402  (puts gymnasium on the path)
import experiments.shift.thruster_chain as TC  # noqa: E402
import experiments.shift.recorder as RC  # noqa: E402
import experiments.shift.truth as TR  # noqa: E402
import experiments.shift.dataset as DS  # noqa: E402
import experiments.shift.shift as SH  # noqa: E402
import experiments.shift.stage0 as S0  # noqa: E402
import gymnasium as gym  # noqa: E402

T0 = time.time()
SHIFT_DIR = REPO / "experiments" / "shift"
PLAIN_SITUATION = {"recorded": "recorded conditions", "strong_current": "strong current", "big_waves": "big waves",
                   "big_kicks": "big starting pushes"}
PLAIN_WORLD = {"stall": "stalling world", "no_stall": "no-stall comparison world"}
CALIBRATION_SEED = 900


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def file_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(SHIFT_DIR.iterdir()) if p.is_file()}


def rnd(v, d=4):
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


def f32s(v):
    """The shortest decimal string that is exactly this float32 (what the HDF5 file stores)."""
    return np.format_float_positional(np.float32(v), unique=True, trim="-")


HASH_START = file_hashes()
data = {}

# ================================================================================================ (1) environment
log("environment")
env = gym.make(TC.env_id("stall", "recorded"))
obs_space, act_space = env.observation_space, env.action_space
float_names = []
for j in range(5):
    float_names += [dict(index=2 * j, name=f"position of float {j + 1}", low=-TC.FLOAT_BOUND, high=TC.FLOAT_BOUND),
                    dict(index=2 * j + 1, name=f"speed of float {j + 1}", low=-TC.FLOAT_BOUND, high=TC.FLOAT_BOUND)]
float_names += [dict(index=10, name="strength of the current", low=0.0, high=1.5),
                dict(index=11, name="wave, first number (the push it adds now: amplitude x cos(phase))", low=-1.0,
                     high=1.0),
                dict(index=12, name="wave, second number (amplitude x sin(phase), the push the wave added a quarter "
                                    "cycle earlier)", low=-1.0, high=1.0)]
ids = []
for eid, (world, situation) in TC.ENV_IDS.items():
    spec = gym.spec(eid)
    law = TC.START_LAWS[situation]
    ids.append(dict(id=eid, world=world, world_plain=PLAIN_WORLD[world], situation=situation,
                    situation_plain=PLAIN_SITUATION[situation], max_episode_steps=spec.max_episode_steps,
                    entry_point=spec.entry_point, kwargs=dict(spec.kwargs),
                    start_law=dict(current=list(law["current"]), wave_amplitude=list(law["wave"]),
                                   float_start_sd=law["kick"], wave_phase=[0.0, "2 pi"])))

A, B = TC.matrices(TC.STALL)
A10 = A[:TC.DS, :TC.DS]
eig = np.linalg.eigvals(A10)
pairs = sorted({(round(abs(e), 6), round(abs(np.angle(e)), 6)) for e in eig}, key=lambda p: p[1])
knee = TC.KNEE
above = float(np.nextafter(np.float32(0.8), np.float32(1.0)))
grid = [round(0.01 * i, 2) for i in range(101)]
special = [0.0, 0.2, 0.4, 0.6, 0.8, knee, above, 0.85, 0.9, 0.95, 1.0]
curve = lambda pts: [dict(throttle=p, thrust_stall=float(TC.thrust(np.array([p]), TC.STALL)[0]),
                          thrust_no_stall=float(TC.thrust(np.array([p]), TC.NO_STALL)[0]),
                          stalled=bool(TC.stalled(np.array([p]), TC.STALL)[0]),
                          energy_cost=TC.STALL.throttle_cost * p * p) for p in pts]
env_r = gym.make(TC.env_id("stall", "recorded"), render_mode="ansi")
o0, i0 = env_r.reset(seed=0)
render_reset = env_r.render()
aw = RC.stall_aware()
cmd0 = aw(i0["state"]).astype(np.float32)
o1, r1, te1, tr1, i1 = env_r.step(cmd0)
render_step = env_r.render()
bound = TC.observation_bound()
data["environment"] = dict(
    name="ThrusterChain", gymnasium_version=gym.__version__, numpy_version=np.__version__,
    import_line="import experiments.shift.thruster_chain  # registers the ids (from the repository root, before "
                "import gymnasium)",
    make_line=f"gymnasium.make('{TC.env_id('stall', 'recorded')}')",
    observation_space=dict(repr=str(obs_space), shape=list(obs_space.shape), dtype=str(obs_space.dtype),
                           low=rnd(obs_space.low, 6), high=rnd(obs_space.high, 6), entries=float_names),
    action_space=dict(repr=str(act_space), shape=list(act_space.shape), dtype=str(act_space.dtype),
                      low=rnd(act_space.low, 6), high=rnd(act_space.high, 6),
                      entries=[dict(index=k, name=f"throttle of thruster {k + 1}",
                                    pushes=f"floats {k + 1} and {k + 2}", low=-1.0, high=1.0) for k in range(4)]),
    registered_ids=ids,
    episode=dict(length=TC.STALL.length, terminated="always False", truncated="True after step 100",
                 discount=TC.STALL.gamma),
    physics=dict(stall=TC.STALL.as_dict(), no_stall=TC.NO_STALL.as_dict(), knee_exact=repr(knee),
                 next_float32_above_knee=repr(above),
                 thrust_at_next_float32_above_knee=repr(float(TC.thrust(np.array([above]))[0])),
                 thrust_at_0_9=repr(float(TC.thrust(np.array([0.9]))[0])),
                 thrust_at_float32_0_9=repr(float(TC.thrust(np.array([np.float32(0.9)]))[0])),
                 thrust_at_1=repr(float(TC.thrust(np.array([1.0]))[0])),
                 command_exactly_0_8_stalls=dict(float64=bool(TC.stalled(np.array([0.8]))[0]),
                                                 float32=bool(TC.stalled(np.array([np.float32(0.8)]))[0]))),
    thrust_formula=dict(below_or_at_knee="thrust(a) = a  for |a| <= 0.8",
                        past_knee="thrust(a) = sign(a) * (0.8 - 1.5 * (|a| - 0.8))  for |a| > 0.8 (stalling world)",
                        no_stall="thrust(a) = a everywhere (no-stall comparison world)"),
    thrust_curve=dict(grid=curve(grid), key_points=curve(special)),
    dynamics=dict(
        formula="s' = A s + 0.3 B thrust(a) + 0.3 v (c + w1);  c' = c;  (w1, w2)' = (w1, w2) turned by 0.27 rad",
        A_floats_10x10=rnd(A10, 6), A_full_13x13=rnd(A, 6), B_13x4=rnd(B, 6),
        v=rnd(TC.push_pattern(), 1),
        oscillator_modes=[dict(radius=r, angle_rad=a) for r, a in pairs],
        wave_turn_rad=TC.STALL.wave_turn, wave_period_steps=2 * math.pi / TC.STALL.wave_turn,
        input_norm_B=bound["input_norm"], push_norm_v=bound["push_norm"]),
    reward=dict(formula="r = -(|s|^2 + 0.3 |a|^2)", throttle_cost=TC.STALL.throttle_cost,
                note="s: the 10 float numbers (float64 state); a: the clipped command; energy is charged on the "
                     "command even while a thruster stalls"),
    start_laws={s: dict(plain=PLAIN_SITUATION[s], current=list(v["current"]), wave_amplitude=list(v["wave"]),
                        float_start_sd=v["kick"]) for s, v in TC.START_LAWS.items()},
    start_set=dict(float_radius_max=TC.START_FLOAT_RADIUS, current=list(TC.START_CURRENT),
                   wave_amplitude_max=TC.START_WAVE_AMPLITUDE, wave_rounding_allowance=TC.WAVE_ROUNDING),
    observation_bound=rnd(bound, 4),
    banks=dict(size=TC.BANK_SIZE, seed=list(TC.BANK_SEED), note="the four banks share the seed, so start i of "
               "every bank uses the same random draws mapped through different start laws"),
    reset_seed0=dict(observation=rnd(o0, 4), info_state=rnd(i0["state"], 6), info_t=i0["t"]),
    step_example=dict(command=[f32s(v) for v in cmd0], reward=rnd(r1, 6), terminated=te1, truncated=tr1,
                      info=dict(t=i1["t"], push=rnd(i1["push"], 6), thrust=rnd(i1["thrust"], 6),
                                stalled=[bool(b) for b in i1["stalled"]], command=rnd(i1["command"], 6))),
    render_example=dict(after_reset=render_reset, after_one_step=render_step),
)
env.close()
env_r.close()

# ================================================================================================ (2) recording controller
log("recording controller")
K = RC.textbook_gain()
P = RC.full_throttle_pattern(K)
spec = RC.DEFAULT_SPEC
c_cap = np.clip(P, -0.8, 0.8)
data["recording_controller"] = dict(
    textbook_gain_K_4x13=rnd(K, 4),
    K_current_column=rnd(K[:, TC.CURRENT], 4), K_wave_columns=rnd(K[:, TC.WAVE1:], 4),
    K_note="command = -(K_floats s + K_water (c, w1, w2)); the textbook optimal linear controller (discounted "
           "linear-quadratic regulator, discount 0.98) of the model in which thrust keeps rising with throttle",
    current_where_textbook_hold_reaches_80pct_thruster1=0.8 / K[0, TC.CURRENT],
    current_where_textbook_hold_reaches_full_throttle_thruster1=1.0 / K[0, TC.CURRENT],
    full_throttle_pattern=rnd(P, 4), full_throttle_pattern_exact=[repr(float(v)) for v in P],
    full_throttle_pattern_capped_at_80=rnd(c_cap, 4),
    thrust_of_pattern=dict(stall=rnd(TC.thrust(P, TC.STALL), 4), no_stall=rnd(TC.thrust(P, TC.NO_STALL), 4),
                           capped_pattern_either_world=rnd(TC.thrust(c_cap, TC.STALL), 4)),
    energy_cost=dict(pattern=TC.STALL.throttle_cost * float(np.sum(P * P)),
                     capped_pattern=TC.STALL.throttle_cost * float(np.sum(c_cap * c_cap))),
    settings=spec.as_dict(), trigger=RC.TRIGGER, bank_stream_seed=list(RC.RECORDING_BANK_STREAM),
    normal_driving_formula="u = -(0.25 K_floats s + K_water e) + 0.3 n, n ~ N(0, I4); command = 0.8 tanh(u / 0.8)",
    full_throttle_formula="in the full-throttle region (c + w1 > 0.565), with probability 0.5 per step: "
                          "command = clip(P + 0.03 n, -1, 1)",
)

# ================================================================================================ (3) reference returns
log("reference returns on the banks")


class NoThrottle(RC.Controller):
    name = "no throttle"

    def __call__(self, x):
        return np.zeros(np.asarray(x).shape[:-1] + (TC.DA,))


CONTROLLERS = ("recording", "stall_aware", "rising_thrust", "no_throttle", "normal_driving_only")
CONTROLLER_PLAIN = dict(recording="recording controller", stall_aware="stall-aware controller",
                        rising_thrust="rising-thrust controller", no_throttle="no throttle",
                        normal_driving_only="normal driving only")


def make_controller(name, physics):
    if name == "recording":
        return TR.recorder_on_bank(physics)
    if name == "stall_aware":
        return RC.stall_aware(physics)
    if name == "rising_thrust":
        return RC.rising_thrust(physics)
    if name == "no_throttle":
        return NoThrottle()
    if name == "normal_driving_only":
        return RC.hand_only(RC.DEFAULT_SPEC, physics)
    raise ValueError(name)


anchors = {}
returns = {}
bank_J = {}
for world, physics in TC.WORLDS.items():
    an = TR.anchors(physics)
    anchors[world] = an
    returns[world] = {}
    for s in TC.SITUATIONS:
        row = {}
        rec = RC.Recorder(RC.DEFAULT_SPEC, physics)
        for name in CONTROLLERS:
            run = TR.bank_rollout(make_controller(name, physics), s, physics, record=True)
            bank_J[(world, s, name)] = run.J
            past = TC.past_knee(run.commands, physics)
            region = rec.in_region(run.states[:, :-1])
            row[name] = dict(
                J=float(run.J.mean()), J_se=float(run.J.std() / math.sqrt(len(run.J))), J_discounted=float(run.Jd.mean()),
                points=float(TR.points(run.J.mean(), an[s])),
                steps_past_80=float(past.mean()), thruster_steps_past_80=float(TC.past_knee(
                    run.commands[..., None], physics).mean()),
                full_throttle_region_steps=float(region.mean()),
                final_float_distance_mean=float(TC.float_radius(run.states[:, -1]).mean()))
        returns[world][s] = row
    log(f"  {world} done")

# the documented env path (float32 commands through env.step) for the stall-aware controller in strong current
log("env path check")
env_sc = gym.make(TC.env_id("stall", "strong_current"))
ctrl = RC.stall_aware()
env_J = []
for start in TC.bank_starts("strong_current"):
    obs, info = env_sc.reset(options={"state": start})
    total, done = 0.0, False
    while not done:
        obs, reward, terminated, truncated, info = env_sc.step(ctrl(info["state"]).astype(np.float32))
        total, done = total + reward, terminated or truncated
    env_J.append(total)
env_sc.close()
env_J = np.asarray(env_J)
exact_J = bank_J[("stall", "strong_current", "stall_aware")]

DESIGN_STALL = dict(
    recorded=dict(recording=-79.8, stall_aware=-26.4, normal_driving_only=-37.9, normal_driving_only_points=78.4,
                  rising_thrust=-35.1, rising_thrust_points=83.6, stall_aware_se=0.37),
    strong_current=dict(recording=-253.9, stall_aware=-103.8, normal_driving_only=-146.9,
                        normal_driving_only_points=71.3, rising_thrust=-335.2, rising_thrust_points=-54.1,
                        stall_aware_se=1.63),
    big_waves=dict(recording=-106.8, stall_aware=-36.2, normal_driving_only=-55.6, normal_driving_only_points=72.5,
                   rising_thrust=-61.8, rising_thrust_points=63.8, stall_aware_se=0.49),
    big_kicks=dict(recording=-263.2, stall_aware=-163.5, normal_driving_only=-205.4, normal_driving_only_points=58.0,
                   rising_thrust=-219.2, rising_thrust_points=44.2, stall_aware_se=2.80))
DESIGN_NO_STALL = dict(recorded=dict(recording=-74.2, rising_thrust=-25.7),
                       strong_current=dict(recording=-146.8, rising_thrust=-68.6),
                       big_waves=dict(recording=-84.1, rising_thrust=-33.1),
                       big_kicks=dict(recording=-256.0, rising_thrust=-149.2))
DESIGN_T7 = dict(recorded=53.4, strong_current=150.1, big_waves=70.6, big_kicks=99.7)
comparison, differs = [], []
for world, table in (("stall", DESIGN_STALL), ("no_stall", DESIGN_NO_STALL)):
    for s, vals in table.items():
        for key, dv in vals.items():
            if key.endswith("_points"):
                cv = returns[world][s][key[:-len("_points")]]["points"]
            elif key.endswith("_se"):
                cv = returns[world][s][key[:-len("_se")]]["J_se"]
            else:
                cv = returns[world][s][key]["J"]
            dec = 2 if key.endswith("_se") else 1
            same = round(cv, dec) == round(dv, dec)
            comparison.append(dict(world=world, situation=s, quantity=key, design=dv, computed=round(cv, 3),
                                   same_at_shown_precision=same))
            if not same:
                differs.append(f"{world}/{s}/{key}: design {dv}, computed {cv:.3f}")
for s, dv in DESIGN_T7.items():
    cv = returns["stall"][s]["stall_aware"]["J"] - returns["stall"][s]["recording"]["J"]
    same = round(cv, 1) == dv
    comparison.append(dict(world="stall", situation=s, quantity="stall_aware_minus_recording", design=dv,
                           computed=round(cv, 3), same_at_shown_precision=same))
    if not same:
        differs.append(f"stall/{s}/stall_aware_minus_recording: design {dv}, computed {cv:.3f}")

data["reference_returns"] = dict(
    note="exact undiscounted 100-step returns, mean over the 1,000 fixed starts of each evaluation bank "
         "(truth.bank_rollout: float64 commands through dynamics()); points use each world's own anchors: "
         "0 = recording controller, 100 = stall-aware controller (stalling world) or rising-thrust controller "
         "(no-stall comparison world)",
    controllers={k: CONTROLLER_PLAIN[k] for k in CONTROLLERS},
    anchors=rnd(anchors, 4), returns=rnd(returns, 4),
    env_path_check=dict(controller="stall-aware controller", situation="strong current",
                        how="gym.make('BCAShift/ThrusterChain-StrongCurrent-v0'), reset(options={'state': start}) "
                            "for each of the 1,000 bank starts, commands rounded to float32 and sent through "
                            "env.step (the snippet in SHIFT_ENV.md)",
                        mean_return_env_float32=float(env_J.mean()), mean_return_exact_float64=float(exact_J.mean()),
                        largest_per_start_difference=float(np.abs(env_J - exact_J).max())),
    design_comparison=dict(rows=comparison, differences=differs,
                           summary=("every design number matches at its shown precision" if not differs else
                                    f"{len(differs)} design numbers differ at their shown precision")),
)
log(f"  design differences: {differs}")

# ================================================================================================ (4) live panel
log("live panel rollouts")
rec_x = TC.bank_starts("recorded")
steps = np.arange(TC.STALL.length)
amp = TC.wave_amplitude(rec_x)
phase = np.arctan2(rec_x[:, TC.WAVE2], rec_x[:, TC.WAVE1])
push_path = rec_x[:, TC.CURRENT][:, None] + amp[:, None] * np.cos(phase[:, None] + TC.STALL.wave_turn * steps[None])
region_frac = (push_path > RC.TRIGGER).mean(1)
START_RULE = ("the lowest bank index whose recorded-conditions start spends between 30% and 45% of its 100 steps in "
              "the full-throttle region (water's push c + w1 above 0.565); the same index is used in every situation "
              "(the banks share their random draws)")
idx = int(np.flatnonzero((region_frac >= 0.30) & (region_frac <= 0.45))[0])
streams = RC.StreamedRecorder.on_bank(RC.Recorder())
noise_i, coin_i = streams.noise[idx], streams.coin[idx]
PANEL_CONTROLLERS = ("no_throttle", "recording", "stall_aware", "rising_thrust")
panel = dict(start_index=idx, start_rule=START_RULE, controllers={k: CONTROLLER_PLAIN[k] for k in PANEL_CONTROLLERS},
             worlds=PLAIN_WORLD, situations=PLAIN_SITUATION,
             fields=dict(states="101 x 13: the state before each step (row t) and after the last step (row 100), "
                                "float64 rounded to 4 decimals; the observation is float32 of it",
                         commands="100 x 4: the applied command (rounded to float32, then clipped), 4 decimals",
                         thrust="100 x 4: the push each thruster applied (info['thrust'])",
                         stalled="100 x 4: 1 when the thruster stalled (info['stalled'])",
                         rewards="100: reward of each step", running_return="100: return so far after each step",
                         full_throttle_command="recording controller only, 100: 1 when it gave the full-throttle "
                                               "command"),
             start_states={}, water={}, rollouts={})
for s in TC.SITUATIONS:
    start = TC.bank_starts(s)[idx]
    panel["start_states"][s] = rnd(start, 6)
    panel["water"][s] = dict(current=rnd(start[TC.CURRENT], 4), wave_amplitude=rnd(TC.wave_amplitude(start), 4),
                             float_start_distance=rnd(TC.float_radius(start), 4))
    for world, physics in TC.WORLDS.items():
        rec = RC.Recorder(RC.DEFAULT_SPEC, physics)
        for name in PANEL_CONTROLLERS:
            e = gym.make(TC.env_id(world, s))
            o, info = e.reset(options={"state": start})
            x = info["state"]
            S, C, TH, ST, R, RR, PU, FT = [x], [], [], [], [], [], [], []
            total, disc_total, disc = 0.0, 0.0, 1.0
            policy = None if name == "recording" else make_controller(name, physics)
            for t in range(TC.STALL.length):
                if name == "recording":
                    cmd, region, press = rec.parts(x, noise_i[t], coin_i[t])
                    FT.append(int(bool(press)))
                else:
                    cmd = policy(x)
                o, r, term, trunc, info = e.step(np.asarray(cmd, np.float64).astype(np.float32))
                total += r
                disc_total += disc * r
                disc *= physics.gamma
                C.append(info["command"]), TH.append(info["thrust"]), ST.append(info["stalled"].astype(int))
                R.append(r), RR.append(total), PU.append(info["push"])
                x = info["state"]
                S.append(x)
                if term:
                    raise AssertionError("terminated must stay False")
            assert trunc, "the episode must be truncated after 100 steps"
            e.close()
            if name == "recording":
                exact_policy = RC.StreamedRecorder(rec, noise_i[None], coin_i[None])
            else:
                exact_policy = policy
            exact = TR.rollout(exact_policy, start[None], physics)
            C = np.asarray(C)
            ST = np.asarray(ST)
            item = dict(states=rnd(np.asarray(S), 4), commands=rnd(C, 4), thrust=rnd(np.asarray(TH), 4),
                        stalled=ST.tolist(), rewards=rnd(R, 4), running_return=rnd(RR, 3),
                        summary=dict(return_env=rnd(total, 4), discounted_return=rnd(disc_total, 4),
                                     return_exact_float64_commands=rnd(exact.J[0], 4),
                                     steps_past_80=int(TC.past_knee(C, physics).sum()),
                                     stalled_thruster_steps=int(ST.sum()),
                                     steps_in_full_throttle_region=int((np.asarray(PU) > RC.TRIGGER).sum()),
                                     final_float_distance=rnd(TC.float_radius(S[-1]), 4),
                                     bank_mean_return=rnd(returns[world][s][name]["J"], 4)))
            if name == "recording":
                item["full_throttle_command"] = FT
                item["summary"]["full_throttle_commands"] = int(sum(FT))
            panel["rollouts"][f"{world}|{s}|{name}"] = item
            if world == "stall" and name == "no_throttle":
                panel["water"][s]["push"] = rnd(PU, 4)
                panel["water"][s]["in_full_throttle_region"] = [int(p > RC.TRIGGER) for p in PU]
data["live_panel"] = panel
log(f"  start index {idx}")

# ================================================================================================ (5, 6) recorded log
log(f"world card on calibration seed {CALIBRATION_SEED} (generates the stalling and no-stall logs)")
card, lg, gap = S0.world_card(CALIBRATION_SEED, RC.DEFAULT_SPEC, 1.5, with_no_stall=True)
log("  world card done")
L = TC.STALL.length
press_all = lg.diagnostic("infos/full_throttle_command").reshape(-1, L)
n_train_eps = DS.SIZES[0]
episode = None
EPISODE_RULE = ("the lowest-numbered training episode whose first 8 rows hold at least one full-throttle command "
                "and at least one normal-driving command")
for e in range(n_train_eps):
    first8 = press_all[e, :8]
    if first8.any() and (~first8).any():
        episode = e
        break
rows = episode * L + np.arange(L)
r8 = rows[:8]
obs, act, rew = lg["observations"], lg["actions"], lg["rewards"]
nobs, state = lg["next_observations"], lg["infos/state"]
press_ep = press_all[episode]
rec = RC.Recorder()
region_ep = rec.in_region(state[rows])
in_ball_ep = gap.contains(act[rows])
past_ep = TC.past_knee(act[rows])
first_rows = []
for i in r8:
    first_rows.append(dict(
        row=int(i), episode=int(lg["infos/episode"][i]), t=int(lg["infos/t"][i]), split=int(lg["infos/split"][i]),
        observations=[f32s(v) for v in obs[i]], actions=[f32s(v) for v in act[i]], rewards=f32s(rew[i]),
        next_observations=[f32s(v) for v in nobs[i]], terminals=bool(lg["terminals"][i]),
        timeouts=bool(lg["timeouts"][i]), full_throttle_command=bool(press_ep[i - rows[0]]),
        in_full_throttle_region=bool(region_ep[i - rows[0]]), removed_by_densest_15pct=bool(in_ball_ep[i - rows[0]]),
        past_80=bool(past_ep[i - rows[0]]), water_push=rnd(TC.push(state[i]), 6)))
# worked reward arithmetic for step 0
i = int(r8[0])
s10 = state[i][:TC.DS]
a4 = np.clip(np.asarray(act[i], np.float32).astype(np.float64), -1.0, 1.0)
sq_s = s10 * s10
sq_a = a4 * a4
s2, a2 = float(np.sum(sq_s)), float(np.sum(sq_a))
r_exact = -(s2 + TC.STALL.throttle_cost * a2)
_, r_dyn = TC.dynamics(state[i], a4)
worked = dict(
    state_float64=[repr(float(v)) for v in state[i]], floats=[repr(float(v)) for v in s10],
    squares_of_floats=[rnd(v, 8) for v in sq_s], sum_squares_floats=rnd(s2, 8),
    command=[repr(float(v)) for v in a4], squares_of_command=[rnd(v, 8) for v in sq_a],
    sum_squares_command=rnd(a2, 8), throttle_cost_term=rnd(TC.STALL.throttle_cost * a2, 8),
    reward_float64=repr(r_exact), reward_from_dynamics=repr(float(r_dyn)), reward_stored_float32=f32s(rew[i]),
    stored_equals_float32_of_exact=bool(np.float32(r_exact) == rew[i]),
    thrust=rnd(TC.thrust(a4), 6), stalled=[bool(b) for b in TC.stalled(a4)],
    water_push=rnd(TC.push(state[i]), 6), trigger=RC.TRIGGER,
    formula="r = -(sum of the 10 squared float numbers + 0.3 x sum of the 4 squared throttles)")
replay8 = DS.replay(lg, rows=rows)
data["recorded_episode"] = dict(
    seed=CALIBRATION_SEED, seed_role="calibration seed (never used by the gate's training runs)",
    episode=int(episode), episode_rule=EPISODE_RULE, rows=[int(rows[0]), int(rows[-1])],
    first_8_rows=first_rows, worked_reward_step0=worked,
    full_throttle_steps=[int(t) for t in np.flatnonzero(press_ep)],
    full_throttle_region_steps=[int(t) for t in np.flatnonzero(region_ep)],
    removed_rows_steps=[int(t) for t in np.flatnonzero(in_ball_ep)],
    past_80_steps=[int(t) for t in np.flatnonzero(past_ep)],
    episode_return=rnd(float(np.sum(rew[rows].astype(np.float64))), 4),
    whole_episode=dict(states=rnd(state[rows], 4), actions=rnd(np.asarray(act[rows], np.float64), 4),
                       rewards=rnd(np.asarray(rew[rows], np.float64), 4),
                       final_state=rnd(lg["infos/next_state"][rows[-1]], 4)),
    replay_check=dict(rows=replay8["rows"], mismatches=replay8["mismatches"], rule=replay8["rule"]),
)
log(f"  episode {episode}")

# dataset facts
log("dataset facts")
train = lg.split_rows("train")
gap_rm = SH.make_removal(lg, "gap", gap)
rnd_rm = SH.make_removal(lg, "random", gap)
a_train = np.asarray(lg["actions"][train], np.float64)
past_train = TC.past_knee(a_train)
press_train = lg.diagnostic("infos/full_throttle_command")[train]
region_train = rec.in_region(lg["infos/state"][train])
rnd_kept = ~rnd_rm.removed[train]
gap_kept = ~gap_rm.removed[train]
kept_rows = gap_rm.kept_train_rows(lg)
xk = lg["infos/state"][kept_rows]
sel = np.random.default_rng(3).choice(len(xk), 2000, replace=False)
m, m_se = SH.removed_share(xk[sel], gap_rm, rec)
reg_sel = rec.in_region(xk[sel])
ref = SH.card_reference(lg, gap_rm, rec)
ep_returns = lg["rewards"][train].astype(np.float64).reshape(-1, L).sum(1)
data["dataset"] = dict(
    seed=CALIBRATION_SEED, world="stall",
    rows=dict(total=lg.n_rows, train=int(len(train)), calibration=int(len(lg.split_rows("calibration"))),
              evaluation=int(len(lg.split_rows("evaluation"))), calibration_bank=int(len(lg.bank_rows))),
    episodes=dict(train=DS.SIZES[0], calibration=DS.SIZES[1], evaluation=DS.SIZES[2], length=L),
    arrays={k: dict(dtype=str(lg[k].dtype), shape=list(lg[k].shape)) for k in DS.D4RL_KEYS + DS.INFO_KEYS},
    full_throttle_share_train=float(press_train.mean()), full_throttle_share_all_rows=float(press_all.mean()),
    full_throttle_region_share_train=float(region_train.mean()),
    full_throttle_share_within_region=float(press_train[region_train].mean()),
    past_80_share_train=float(past_train.mean()),
    every_full_throttle_command_past_80=card["every_full_throttle_command_past80"],
    nothing_else_past_80=card["nothing_else_past80"],
    recording_controller_mean_episode_return_train=float(ep_returns.mean()),
    gap=dict(centre=rnd(gap.centre, 4), radius=rnd(gap.rho, 4), mass=rnd(gap.mass, 4), rule=gap.rule,
             competing_ball_ratio=rnd(gap.competing_ratio, 4),
             distance_centre_to_full_throttle_pattern=rnd(float(np.linalg.norm(np.asarray(gap.centre) - P)), 4)),
    gap_removal=dict(
        removed_training_rows=gap_rm.count, removed_share=gap_rm.rate(lg), kept_training_rows=int(gap_kept.sum()),
        bank_rows_kept=int((~gap_rm.bank_removed).sum()), bank_rows_removed=int(gap_rm.bank_removed.sum()),
        full_throttle_commands_removed_share=card["full_throttle_commands_removed"],
        removed_rows_that_are_full_throttle_commands=card["removed_rows_full_throttle_share"],
        full_throttle_commands_kept=int((press_train & gap_kept).sum()),
        kept_rows_past_80=card["kept_past80_rows"], kept_rows_past_80_share=card["kept_past80_share"],
        full_throttle_region_rows_kept_share=card["region_rows_kept"],
        identical_rule_share_kept=card["identical_rule_share"]),
    random_removal=dict(removed_training_rows=rnd_rm.count, kept_training_rows=int(rnd_kept.sum()),
                        past_80_rows_kept_share_of_all_past_80=card["random_keeps_past80"],
                        kept_rows_past_80_share=float(past_train[rnd_kept].mean()),
                        full_throttle_commands_removed_share=float(np.mean(~rnd_kept[press_train])),
                        bank_rows_removed=int(rnd_rm.bank_removed.sum())),
    no_stall_world=card.get("no_stall_world"),
    removed_share_m=dict(rows=int(len(sel)), draws=SH.MS_DRAWS, sample="2,000 kept training rows, "
                         "np.random.default_rng(3)", quantiles=dict(zip(("p10", "p50", "p90", "p99"),
                                                                       np.quantile(m, [0.1, 0.5, 0.9, 0.99]))),
                         mean=float(m.mean()), mean_in_region=float(m[reg_sel].mean()),
                         mean_outside_region=float(m[~reg_sel].mean()), max_se=float(m_se.max())),
    card_reference=dict(float_radius_p99=ref.float_radius_p99,
                        heldout_distance_quantiles=dict(p50=float(np.quantile(ref.heldout_distance, 0.5)),
                                                        p90=float(np.quantile(ref.heldout_distance, 0.9)))),
    design_ranges=dict(
        source="SHIFT_ENV.md and stage0.COMPUTED (calibration seeds 900-904) and the design record (10 seeds)",
        full_throttle_commands_removed="99.1-99.8%", gap_centre_thruster1="-0.74 to -0.78",
        kept_rows_past_80="0.04-0.14% (SHIFT_ENV.md); 0.03-0.14% (stage0.COMPUTED)",
        past_80_before="12-14%", random_keeps_past_80="84-85%", competing_ratio="1.24-1.44",
        identical_rule="99.86-99.97% (SHIFT_ENV.md); 99.87-99.97% (stage0.COMPUTED)",
        region_rows_kept="46-48% (SHIFT_ENV.md); 44-47% (stage0.COMPUTED and the design record)",
        radius="0.252-0.298", m_quantiles="0.000 / 0.005 / 0.521 / 0.576; mean 0.537 in the region, 0.008 outside"),
)
data["dataset"] = rnd(data["dataset"], 6)

# ================================================================================================ (7) shift cards
log("shift cards (stalling world, gap of the calibration seed)")
cards = {}
for s in TC.SITUATIONS:
    cards[s] = {}
    for name in CONTROLLERS:
        c = SH.shift_card(make_controller(name, TC.STALL), s, TC.STALL, ref, anchors=anchors["stall"][s])
        c.pop("policy", None)
        cards[s][name] = rnd(c, 4)
    log(f"  {s} done")
DESIGN_CARD_SEED0 = dict(
    recorded=dict(region=0.279, recording=dict(unrecorded=0.139, covered=0.99, ball=0.156),
                  rising_thrust=dict(unrecorded=0.097, covered=0.94, ball=0.149, ball_past_80=0.018),
                  stall_aware=dict(unrecorded=0.0, ball=0.171)),
    strong_current=dict(region=0.937, recording=dict(unrecorded=0.469, covered=0.0),
                        rising_thrust=dict(unrecorded=0.840, covered=0.0, ball=0.155, floats_beyond_p99=0.226),
                        stall_aware=dict(unrecorded=0.0, ball=0.597, floats_beyond_p99=0.012)),
    big_waves=dict(region=0.396, recording=dict(unrecorded=0.198), rising_thrust=dict(unrecorded=0.262)),
    big_kicks=dict(recording=dict(unrecorded=0.139), rising_thrust=dict(unrecorded=0.208, covered=0.34)))

# the true price of the full-throttle pattern against the same pattern capped at 80%, on the frozen region pool
log("true price of pushing past 80% on the full-throttle-region pool")
pools = SH.make_pools(lg, gap, rec)
xr = lg["infos/state"][pools["region"]]
price = {}
for cname in ("stall_aware", "rising_thrust"):
    for world, physics in TC.WORLDS.items():
        pol = make_controller(cname, physics)
        q_hi = TR.q_value(pol, xr, np.broadcast_to(pools["c_hi"], (len(xr), TC.DA)), physics)
        q_cap = TR.q_value(pol, xr, np.broadcast_to(pools["c_cap"], (len(xr), TC.DA)), physics)
        d = q_cap - q_hi
        price[f"{cname}|{world}"] = dict(median_true_price=float(np.median(d)), mean=float(d.mean()),
                                         p10=float(np.quantile(d, 0.1)), p90=float(np.quantile(d, 0.9)),
                                         share_positive=float(np.mean(d > 0)))
CARD_KEYS = dict(region=("full_throttle_region_share",), unrecorded=("unrecorded_throttle_share",),
                 covered=("covered_share",), ball=("inside_removed_ball", "share"),
                 ball_past_80=("inside_removed_ball", "past_80"),
                 floats_beyond_p99=("outside_recorded_conditions", "by_floats"))
card_comparison = []
for s, entry in DESIGN_CARD_SEED0.items():
    for who, vals in entry.items():
        if who == "region":
            vals, who = {"region": vals}, "recording"
        for key, dv in vals.items():
            cv = cards[s][who]
            for part in CARD_KEYS[key]:
                cv = cv[part]
            card_comparison.append(dict(situation=s, controller=who, measure=key, design_seed0=dv,
                                        computed_seed900=cv, difference=None if cv is None else round(cv - dv, 4)))
data["shift_cards"] = dict(
    design_comparison=dict(note="the design's card used the gap of data seed 0; this one uses calibration seed "
                                f"{CALIBRATION_SEED}, so small differences are expected", rows=card_comparison),
    note="shift.shift_card for each reference controller on each bank, stalling world, with the gap and kept rows "
         f"of calibration seed {CALIBRATION_SEED}; measures 1-8 of SHIFT_ENV.md",
    cards=cards, design_card_seed0=DESIGN_CARD_SEED0,
    region_pool=dict(rows=int(len(pools["region"])), c_hi=rnd(pools["c_hi"], 4), c_cap=rnd(pools["c_cap"], 4),
                     true_price=rnd(price, 4),
                     note="true price = Q(s, capped pattern) - Q(s, full-throttle pattern), exact 1,000-step "
                          "discounted value with the named controller continuing; positive means pushing past "
                          "80% is worse. Design: 0.355 median (stall-aware, stalling world), -0.009 no-stall"),
)

# ================================================================================================ (8) methods
log("methods (literals from the source text)")


def lit(node):
    """A literal value of an AST node: constants, tuples, lists, dicts and dict(...) calls with keyword literals;
    anything else is returned as its source text in brackets."""
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        pass
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict" and not node.args:
        return {kw.arg: lit(kw.value) for kw in node.keywords if kw.arg is not None}
    if isinstance(node, ast.Dict):
        return {lit(k): lit(v) for k, v in zip(node.keys, node.values)}
    if isinstance(node, (ast.Tuple, ast.List)):
        return [lit(e) for e in node.elts]
    return "(" + ast.unparse(node)[:200] + ")"


def literals(path, names):
    tree = ast.parse(Path(path).read_text())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in names:
                out[node.targets[0].id] = lit(node.value)
        if isinstance(node, ast.ClassDef) and node.name in names:
            out[node.name] = {b.target.id: lit(b.value) for b in node.body
                              if isinstance(b, ast.AnnAssign) and isinstance(b.target, ast.Name) and b.value is not None}
    return out


lit_learners = literals(SHIFT_DIR / "learners.py", {"ROOM_EVEN_LADDER", "DOSE_GAIN", "PLACEMENTS", "BCA_SETTINGS",
                                                    "ORACLE_KINDS", "DERIVED_KINDS", "CONTROLLERS", "LearnSettings"})
lit_accept = literals(SHIFT_DIR / "acceptance.py", {"PASS_MARKS", "MARK_TEXT", "MARK_LABELS", "HOLM_FAMILIES",
                                                    "VERDICTS", "VERDICT_ORDER", "STOP_RULE", "ALPHA_RULE",
                                                    "STOP_MESSAGES", "BUDGET", "BCA_ARMS", "ARM_TEXT", "GateConfig",
                                                    "HOST_ARMS"})
ratios = [0.0, 0.05, 0.1, 0.2, 1 / 3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 100.0]
dose_table = []
for q in ratios:
    x = 3.0 * q
    dose_table.append(dict(D_over_mean_D=q, x=x, harm_placed_dose=1 + 0.5 * x / (1 + x), cap2_dose=1 + x / (1 + x)))
k_ref = S0.CRITIC_K_REFERENCE
pm = lit_accept["PASS_MARKS"]["H4"]
power_table = []
for sd in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0):
    n = max(pm["n_min"], int(math.ceil(((pm["z_alpha"] + pm["z_power"]) * pm["inflation"] * sd / pm["effect"]) ** 2)))
    power_table.append(dict(sd_points=sd, N=n))
data["methods"] = dict(
    td3_bc=dict(
        actor_loss="-lambda * mean Q1(s, pi(s)) + mean over rows of [multiplier_i * mean over the 4 throttles of "
                   "(pi(s_i) - a_i)^2],  lambda = alpha / mean |Q1| (algorithms/td3_bc.py, _actor_loss_fn)",
        settings=lit_learners.get("LearnSettings"),
        networks="actor 13-256-256-4 ReLU with tanh output; two critics 256-256 ReLU (algorithms/td3_bc.py)"),
    actor_fixed_point=dict(
        formula="pi = (m a + k c) / (m + k), k = d lambda |H| / 2 (d = 4 throttles); share of the way from the "
                "recorded command a toward the critic's best command c is k / (m + k)",
        critic_weight_k_reference=k_ref,
        share_of_way={f"dose_{m:g}": k_ref / (m + k_ref) for m in (1.0, 1.25, 1.5, 2.0, 3.0)},
        dose_cap_leverage=S0.dose_cap_leverage(k_ref)),
    bca_dose=dict(
        formula="dose = 1 + blend * w / (w + u): w = the band width at the row (calibrated threshold x predicted "
                "error scale x unit), u = the unit (residual scale); blend 0.5, so doses lie in [1, 1.5) "
                "(calibration/dose.py, level_critic_dose)",
        settings=lit_learners.get("BCA_SETTINGS")),
    caution_placements=dict(
        harm_placed="1 + 0.5 x / (1 + x), x = 3 D_i / mean(D), D_i the exact over-preference at row i of the "
                    "arm's current actor, refreshed every 4,000 steps",
        cap2="1 + x / (1 + x) (doses up to 2: is the dose cap the limit?)",
        amount_matched="every row gets the mean harm-placed dose of the same refresh",
        shuffled="the harm-placed doses permuted by a fixed seeded permutation (two versions)",
        reversed="the harm-placed doses with their ranks flipped (largest dose on the smallest D_i)",
        even="every row the same multiplier; ladder " + ", ".join(f"x{m:g}" for m in lit_learners["ROOM_EVEN_LADDER"]),
        strength_matched_even="even caution whose strength S (root-mean-square change of the command from the "
                              "no-caution arm on 5,000 evaluation-split states) equals the harm-placed arm's; the "
                              "multiplier is found from strengths alone and locked before any return is computed",
        dose_table=dose_table, dose_gain=lit_learners.get("DOSE_GAIN"),
        room_even_ladder=lit_learners.get("ROOM_EVEN_LADDER"),
        bca_placements=lit_learners.get("PLACEMENTS")),
    power=dict(formula="N = max(10, ceil(((2.128 + 0.842) x 1.25 x s / 3)^2)), s = the larger standard deviation of "
                       "the two paired differences across the gate seeds (points)", table=power_table),
    gate=dict(pass_marks=lit_accept.get("PASS_MARKS"), mark_text=lit_accept.get("MARK_TEXT"),
              mark_labels=lit_accept.get("MARK_LABELS"), holm_families=lit_accept.get("HOLM_FAMILIES"),
              verdicts=lit_accept.get("VERDICTS"), verdict_order=lit_accept.get("VERDICT_ORDER"),
              stop_rule=lit_accept.get("STOP_RULE"), alpha_rule=lit_accept.get("ALPHA_RULE"),
              stop_messages=lit_accept.get("STOP_MESSAGES"), budget=lit_accept.get("BUDGET"),
              gate_config=lit_accept.get("GateConfig"), bca_arms=lit_accept.get("BCA_ARMS"),
              arm_text=lit_accept.get("ARM_TEXT"), host_arms=lit_accept.get("HOST_ARMS")),
    stage0_marks=dict(rule=S0.MARKS, computed_in_design=S0.COMPUTED),
)
def sig(v, n=6):
    """Round floats to n significant figures (so small pass-mark thresholds such as 1e-12 survive)."""
    if isinstance(v, dict):
        return {str(k): sig(x, n) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [sig(x, n) for x in v]
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (float, np.floating)) and math.isfinite(float(v)):
        return float(f"{float(v):.{n}g}")
    return v


data["methods"] = sig(data["methods"], 6)

# ================================================================================================ (9) code names
data["code_names"] = [
    dict(plain="the environment", code="experiments.shift.thruster_chain.ThrusterChainEnv"),
    dict(plain="the one dynamics function", code="thruster_chain.dynamics (numpy), thruster_chain.dynamics_jax"),
    dict(plain="thrust per thruster", code="thruster_chain.thrust"),
    dict(plain="stalling world / no-stall comparison world", code="world 'stall' / 'no_stall'; Physics(stall=...)"),
    dict(plain="recorded conditions / strong current / big waves / big starting pushes",
         code="situations 'recorded' / 'strong_current' / 'big_waves' / 'big_kicks'"),
    dict(plain="the evaluation banks", code="thruster_chain.bank_starts(situation)"),
    dict(plain="the textbook gain", code="recorder.textbook_gain"),
    dict(plain="recording controller", code="recorder.Recorder; on a bank: recorder.StreamedRecorder"),
    dict(plain="full-throttle command", code="recorder.full_throttle_pattern; log diagnostic "
                                            "infos/full_throttle_command"),
    dict(plain="full-throttle region", code="recorder.Recorder.in_region (trigger recorder.TRIGGER = 0.565)"),
    dict(plain="stall-aware controller", code="recorder.stall_aware()"),
    dict(plain="rising-thrust controller", code="recorder.rising_thrust()"),
    dict(plain="normal driving only", code="recorder.hand_only() (class HandOnly)"),
    dict(plain="the recorded log", code="dataset.generate, dataset.write_hdf5, dataset.write_d4rl_training"),
    dict(plain="removal of the densest 15% (the gap)", code="shift.define_gap, shift.make_removal(log, 'gap', gap)"),
    dict(plain="random removal", code="shift.make_removal(log, 'random', gap)"),
    dict(plain="removed share m(s)", code="shift.removed_share"),
    dict(plain="the shift card", code="shift.shift_card"),
    dict(plain="exact action value, returns, points", code="truth.q_value, truth.rollout, truth.points, truth.anchors"),
    dict(plain="over-preference", code="truth.over_preference"),
    dict(plain="pull past 80%", code="truth.pull_past_80"),
    dict(plain="step-by-step loss accounting", code="truth.attribution"),
    dict(plain="noise map", code="truth.noise_map"),
    dict(plain="plain TD3+BC", code="learners.cotrain (algorithms/td3_bc.py unchanged)"),
    dict(plain="fixed-controller critics", code="learners.fixed_critic"),
    dict(plain="behaviour cloning", code="learners.clone"),
    dict(plain="critic swap and caution placements (actor-only runs)", code="learners.ActorOnly, learners.room_arms, "
                                                                           "learners.harm_doses"),
    dict(plain="strength matching", code="learners.match_even (experiments/signal/glq_match.py)"),
    dict(plain="BCA", code="algorithms/td3_bc_bca.py (unchanged), learners.bca_update, calibration/dose.py"),
    dict(plain="the gate and its stages", code="experiments.shift.acceptance"),
]

# ================================================================================================ meta
HASH_END = file_hashes()
if HASH_END != HASH_START:
    raise SystemExit("a file in experiments/shift changed while this script ran; refusing to write")
data["meta"] = dict(
    script=str(Path(__file__).resolve()), created=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=round(time.time() - T0),
    python=platform.python_version(), numpy=np.__version__, gymnasium=gym.__version__,
    flags={k: os.environ.get(k) for k in ("JAX_PLATFORMS", "OMP_NUM_THREADS", "PYTHONDONTWRITEBYTECODE")},
    nothing_trains="no critic, actor, cloning or BCA component is trained or imported (learners.py and acceptance.py "
                   "are read as text only); every number is exact simulation of fixed controllers, the generated "
                   "log, or arithmetic",
    experiments_shift_sha256=HASH_START,
    plain_names=dict(situations=PLAIN_SITUATION, worlds=PLAIN_WORLD, controllers=CONTROLLER_PLAIN),
)
def clean(v):
    """numpy types to Python, non-finite floats to None (no rounding)."""
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, np.ndarray):
        return clean(v.tolist())
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v) if math.isfinite(float(v)) else None
    return v


OUT.write_text(json.dumps(clean(data), indent=None, separators=(",", ":"), allow_nan=False))
log(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")
