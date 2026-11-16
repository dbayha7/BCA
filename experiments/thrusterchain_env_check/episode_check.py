"""ThrusterChain episode check: do episodes played through the environment match the scoring code?

Runs marks T1 to T6 of EPISODE_CHECK.md, which was written before this script was run and fixes what is
played, the tolerances and the commands. Trains nothing and plays no learned policy: only fixed command
sequences and the two fixed reference controllers (the zero command and the stall-aware controller).

    --setup wsl      the scoring interpreter and the frozen code copy; reads the pinned scoring cache and the
                     saved precision-check results, writes pinned_reference.npz and the WSL result file
    --setup windows  the repository's interpreter and code copy; reads pinned_reference.npz, also runs T5
                     (same numbers on both setups) and T6 (the JAX twin), writes the Windows result file
    --smoke          wiring and timing only: a few starts, nothing judged, nothing written

Each result file is written once and never overwritten.
"""
import argparse
import datetime
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SEED = 2026111511
BANKS = ("none", "mild", "strong", "recorded", "strong_current", "big_waves", "big_kicks")
SITUATIONS = BANKS[3:]
N_FIXED, N_REFERENCE = 250, 1000
DISTURBANCE = 1e-6
EXACT, POINTS, PLATFORM_POINTS, STARTS, JAX, IDENTITY = 1e-9, 1e-3, 1e-6, 1e-12, 1e-12, 1e-9
GAIN_LIMIT = 100.0
REFERENCE = HERE / "pinned_reference.npz"
IMPORTED = ("experiments/shift/thruster_chain.py", "experiments/shift/truth.py", "experiments/shift/recorder.py",
            "experiments/shift/__init__.py", "experiments/signal/gapped_lq.py", "experiments/signal/lq_harness.py")


class Zero:
    """The zero command on every thruster."""
    name = "the zero command"

    def __call__(self, x):
        return np.zeros(np.shape(x)[:-1] + (4,))


class Replay:
    """Fixed commands [starts, steps, 4], replayed by the scoring code."""

    def __init__(self, commands):
        self.commands = np.asarray(commands, np.float64)

    def act(self, x, t):
        return self.commands[:, t]


class Disturbed:
    """A controller with one of the stress test's disturbances."""

    def __init__(self, base, scale=None, round_state=False):
        self.base, self.scale, self.round_state = base, scale, round_state

    def act(self, x, t):
        if self.round_state:                                  # the state as a deployed network sees it
            x = np.asarray(x, np.float64).astype(np.float32).astype(np.float64)
        command = self.base.act(x, t)
        return command if self.scale is None else command * self.scale[:, t]


def act_of(controller):
    return controller.act if hasattr(controller, "act") else (lambda x, t: controller(x))


def play(env, start, length, act=None, commands=None, cost=0.3):
    """One episode through the public environment, with the bookkeeping of mark T2.

    Returns (states [length + 1, 13], rewards [length], return, bookkeeping violations)."""
    obs, info = env.reset(options={"state": start})
    state = info["state"]
    bad = int(obs.dtype != np.float32 or not np.array_equal(obs, state.astype(np.float32)) or info["t"] != 0)
    states, rewards, total, tracking, energy = [state], [], 0.0, 0.0, 0.0
    for t in range(length):
        action = commands[t] if commands is not None else np.asarray(act(state[None], t)[0], np.float32)
        tracking += float(np.sum(state[:10] ** 2))
        obs, reward, terminated, truncated, info = env.step(action)
        state = info["state"]
        energy += cost * float(np.sum(info["command"] ** 2))
        bad += int(terminated or truncated != (t == length - 1) or info["t"] != t + 1 or obs.dtype != np.float32
                   or not np.array_equal(obs, state.astype(np.float32)))
        total += reward
        states.append(state)
        rewards.append(reward)
    try:                                                      # a 101st step must be refused
        env.step(action)
        bad += 1
    except Exception:
        pass
    size = max(1.0, abs(total))
    bad += int(abs(total - sum(rewards)) > IDENTITY * size) + int(abs(tracking + energy + total) > IDENTITY * size)
    return np.array(states), np.array(rewards), total, bad


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recorded_precision(runs_root):
    """The host study's saved closed-loop precision results (its check S7), by quality and setup. Read only."""
    latest = {}
    for tag, root in (("v3", runs_root / "stage2"), ("v4", runs_root / "supplement_v4" / "stage2")):
        for path in sorted(root.glob("*/seed*/*/lanes/s7-step100000/s7.json")):
            quality, seed, setup = path.parts[-6], path.parts[-5], path.parts[-4]
            closed = json.loads(path.read_text())["closed_loop"]
            banks = {k.split("/", 1)[1]: v for k, v in closed.items() if k.startswith("scored/")}
            worst = max(banks, key=lambda b: banks[b]["max_points"])
            latest[(quality, setup, seed)] = dict(
                version=tag, largest_per_start_points=banks[worst]["max_points"], worst_bank=worst,
                largest_bank_mean_points=max(abs(v["mean_points"]) for v in banks.values()),
                within_limit=all(v["passed"] for v in banks.values()))
    table = {}
    for (quality, setup, seed), row in sorted(latest.items()):
        entry = table.setdefault(f"{quality} / {setup}", dict(policies=0, within_limit=0, largest_per_start_points=0.0,
                                                               worst_bank="", largest_bank_mean_points=0.0))
        entry["policies"] += 1
        entry["within_limit"] += int(row["within_limit"])
        if row["largest_per_start_points"] > entry["largest_per_start_points"]:
            entry["largest_per_start_points"], entry["worst_bank"] = row["largest_per_start_points"], row["worst_bank"]
        entry["largest_bank_mean_points"] = max(entry["largest_bank_mean_points"], row["largest_bank_mean_points"])
    return table


def run(args):
    code_root = Path(args.code_root).resolve()
    sys.path.insert(0, str(code_root))
    import experiments.shift  # noqa: F401  (dependency path and version checks; must come before gymnasium)
    import gymnasium as gym
    from experiments.shift import recorder as RC
    from experiments.shift import thruster_chain as TC
    from experiments.shift import truth as TR

    smoke, wsl = args.smoke, args.setup == "wsl"
    result_path = HERE / f"episode_check_result_{args.setup}.json"
    if not smoke and result_path.exists():
        sys.exit(f"{result_path.name} already exists. The check is run once; it is not overwritten.")
    if not smoke and wsl and REFERENCE.exists():
        sys.exit(f"{REFERENCE.name} already exists. The WSL run writes it once.")
    n_fixed, n_ref = (5, 20) if smoke else (N_FIXED, N_REFERENCE)
    length, started = TC.STALL.length, time.time()

    # ---- the banks: starts, pinned reference returns, point scales
    pinned = {}
    if wsl:
        from experiments.shift_hosts import evaluate as EV
        for bank in BANKS:
            cache = EV.load_cache(bank)        # checks the pinned SHA-256 and that the starts are the bank's
            pinned[bank] = SimpleNamespace(starts=np.array(cache["starts"]), zero=np.array(cache["zero/J"]),
                                           stall_aware=np.array(cache["stall_aware/J"]),
                                           ppu=float(EV.points_per_return_unit(bank)), wsl_zero=None, wsl_stall_aware=None)
    elif REFERENCE.exists():
        with np.load(REFERENCE, allow_pickle=False) as saved:
            for bank in BANKS:
                pinned[bank] = SimpleNamespace(**{key: (float(saved[f"{bank}/{key}"]) if key == "ppu"
                                                        else np.array(saved[f"{bank}/{key}"]))
                                                  for key in ("starts", "zero", "stall_aware", "ppu", "wsl_zero",
                                                              "wsl_stall_aware")})
    elif smoke:                                # a Windows wiring test before the WSL run: situation banks only
        for bank in SITUATIONS:
            pinned[bank] = SimpleNamespace(starts=TC.bank_starts(bank), zero=None, stall_aware=None, ppu=1.0,
                                           wsl_zero=None, wsl_stall_aware=None)
    else:
        sys.exit(f"{REFERENCE.name} is missing: run the WSL setup first.")
    banks = tuple(pinned)

    envs = {"stall": gym.make(TC.env_id("stall", "recorded"), disable_env_checker=True),
            "no_stall": gym.make(TC.env_id("no_stall", "recorded"), disable_env_checker=True)}
    stall_aware, zero = RC.stall_aware(TC.STALL), Zero()
    out = {"T1": {}, "T2": {}, "T3": {}, "T4": {}, "T4_gain": {}, "T5": {}, "T6": {}}

    # ---- T1 and T2 on fixed command sequences
    signs = np.array([1, -1, 1, -1], np.float32)
    differing, violations, episodes = 0, 0, 0
    for world, physics in TC.WORLDS.items():
        for bank in banks if world == "stall" else [b for b in banks if b in SITUATIONS]:
            starts = pinned[bank].starts[:n_fixed]
            rng = np.random.default_rng([SEED, 1, BANKS.index(bank), int(world == "stall")])
            sequences = {name: np.broadcast_to(np.float32(level) * signs, (len(starts), length, 4)).copy()
                         for name, level in (("zero", 0.0), ("at the 80% point", 0.8), ("90%", 0.9),
                                             ("full throttle", 1.0), ("out of range", 1.5))}
            sequences["random"] = rng.uniform(-1, 1, (len(starts), length, 4)).astype(np.float32)
            for name, commands in sequences.items():
                scored = TR.rollout(Replay(commands), starts, physics, record=True)
                for i, start in enumerate(starts):
                    states, rewards, total, bad = play(envs[world], start, length, commands=commands[i],
                                                       cost=physics.throttle_cost)
                    same = (np.array_equal(states, scored.states[i]) and np.array_equal(rewards, scored.rewards[i])
                            and total == scored.J[i])
                    differing += int(not same)
                    violations += bad
                    episodes += 1
        print(f"  fixed commands, {TC.WORLD_LABELS[world]}: {time.time() - started:6.0f} s", flush=True)
    out["T1"] = dict(episodes=episodes, episodes_that_differ=differing)

    # ---- T3: the reference controllers through the public environment, against the pinned numbers
    env = envs["stall"]
    for bank in banks:
        p = pinned[bank]
        starts = p.starts[:n_ref]
        played = {}
        for label, controller in (("zero", zero), ("stall_aware", stall_aware)):
            totals = np.zeros(len(starts))
            for i, start in enumerate(starts):
                _, _, totals[i], bad = play(env, start, length, act=act_of(controller))
                violations += bad
                episodes += 1
            played[label] = totals
        record = TR.rollout(stall_aware, starts, TC.STALL, record=True)
        entry = dict(points_per_return_unit=p.ppu, starts=len(starts))
        if p.zero is not None:
            entry.update(
                a_zero_command_return_gap=float(np.abs(played["zero"] - p.zero[:n_ref]).max()),
                b_scoring_code_return_gap=float(np.abs(record.J - p.stall_aware[:n_ref]).max()),
                c_public_environment_points_gap=float(np.abs(played["stall_aware"] - p.stall_aware[:n_ref]).max() * p.ppu),
                c_public_environment_return_gap=float(np.abs(played["stall_aware"] - p.stall_aware[:n_ref]).max()),
                c_public_environment_mean_points_gap=float(abs((played["stall_aware"] - p.stall_aware[:n_ref]).mean()) * p.ppu))
        out["T3"][bank] = entry

        # ---- T4: the three disturbances, with the scoring code
        rng = np.random.default_rng([SEED, 4, BANKS.index(bank)])
        scale = 1.0 + DISTURBANCE * rng.standard_normal((len(starts), length, 4))
        moved = starts * (1.0 + DISTURBANCE * rng.standard_normal(starts.shape))
        assert bool(np.all(TC.in_start_set(moved)))
        gaps = {"commands": TR.rollout(Disturbed(stall_aware, scale=scale), starts, TC.STALL).J - record.J,
                "state read as float32": TR.rollout(Disturbed(stall_aware, round_state=True), starts, TC.STALL).J - record.J,
                "start": TR.rollout(stall_aware, moved, TC.STALL).J - record.J}
        out["T4"][bank] = {name: float(np.abs(gap).max() * p.ppu) for name, gap in gaps.items()}
        # Reported, not judged (EPISODE_CHECK.md, note of 2026-11-15): the gain of each disturbance, the largest
        # relative change in the return divided by the relative size of the disturbance.
        sizes = {"commands": DISTURBANCE, "state read as float32": 2.0 ** -24, "start": DISTURBANCE}
        out["T4_gain"][bank] = {name: float((np.abs(gap) / np.abs(record.J)).max() / sizes[name])
                                for name, gap in gaps.items()}

        # ---- T5: the other setup's numbers, and the regenerated starts
        five = {}
        if bank in SITUATIONS:
            five["starts_gap"] = float(np.abs(TC.bank_starts(bank)[:n_ref] - starts).max())
            five["starts_bit_for_bit"] = bool(np.array_equal(TC.bank_starts(bank)[:n_ref], starts))
        if p.wsl_zero is not None:
            for label in ("zero", "stall_aware"):
                other = getattr(p, f"wsl_{label}")[:n_ref]
                five[f"{label}_points_gap"] = float(np.abs(played[label] - other).max() * p.ppu)
                five[f"{label}_starts_bit_for_bit"] = int(np.sum(played[label] == other))
        out["T5"][bank] = five
        pinned[bank].played = played

        # ---- T6: the JAX twin on the visited states (where JAX is installed)
        if args.setup == "windows" and bank in SITUATIONS:
            from jax.experimental import enable_x64
            with enable_x64():
                import jax.numpy as jnp
                visited, commands = record.states[:, :-1].reshape(-1, TC.DX), record.commands.reshape(-1, TC.DA)
                nxt, reward = TC.dynamics(visited, commands, TC.STALL)
                nxt_j, reward_j = TC.dynamics_jax(jnp.asarray(visited, jnp.float64), jnp.asarray(commands, jnp.float64), TC.STALL)
                out["T6"][bank] = dict(
                    states=int(len(visited)),
                    next_state_gap=float(np.max(np.abs(np.asarray(nxt_j) - nxt) / np.maximum(1.0, np.abs(nxt)))),
                    reward_gap=float(np.max(np.abs(np.asarray(reward_j) - reward) / np.maximum(1.0, np.abs(reward)))))
        print(f"  reference controllers, bank {bank}: {time.time() - started:6.0f} s", flush=True)
    out["T2"] = dict(episodes=episodes, violations=violations)

    # ---- the marks
    have_pinned = all(pinned[b].zero is not None for b in banks)
    marks = {"T1 same commands, same episode": out["T1"]["episodes_that_differ"] == 0,
             "T2 episode bookkeeping": out["T2"]["violations"] == 0}
    if have_pinned:
        marks["T3 pinned references reproduce"] = all(
            e["a_zero_command_return_gap"] <= EXACT and e["b_scoring_code_return_gap"] <= EXACT
            and e["c_public_environment_points_gap"] <= POINTS for e in out["T3"].values())
    # Amendment 1 (EPISODE_CHECK.md section 9, accepted by the owner on 2026-11-15): T4 is judged on the gain.
    # T4 as first written (an absolute 0.001 points) is still reported, but no longer counts toward the verdict.
    marks["T4 the environment does not magnify small errors (gain at most 100)"] = all(
        g <= GAIN_LIMIT for e in out["T4_gain"].values() for g in e.values())
    superseded = {"T4 as first written (within 0.001 points)": all(g <= POINTS for e in out["T4"].values() for g in e.values())}
    five_ok = all(e.get("starts_gap", 0.0) <= STARTS for e in out["T5"].values())
    if args.setup == "windows" and have_pinned:
        five_ok = five_ok and all(e["zero_points_gap"] <= PLATFORM_POINTS and e["stall_aware_points_gap"] <= PLATFORM_POINTS
                                  for e in out["T5"].values())
    marks["T5 same numbers on both setups" if args.setup == "windows" else "T5 regenerated starts equal the pinned ones"] = five_ok
    if out["T6"]:
        marks["T6 the JAX twin agrees"] = all(e["next_state_gap"] <= JAX and e["reward_gap"] <= JAX for e in out["T6"].values())
    verdict = "PASS" if all(marks.values()) else "FAIL"

    print(f"\n{'bank':15s} {'T3a zero':>10s} {'T3b scoring':>11s} {'T3c env pts':>11s} {'T4 commands':>11s} {'T4 float32':>10s} "
          f"{'T4 start':>9s} {'T5 pts':>9s}")
    for bank in banks:
        three, four, five = out["T3"][bank], out["T4"][bank], out["T5"][bank]
        fmt = lambda v: "      -" if v is None else f"{v:.2e}"   # noqa: E731
        print(f"{bank:15s} {fmt(three.get('a_zero_command_return_gap')):>10s} {fmt(three.get('b_scoring_code_return_gap')):>11s} "
              f"{fmt(three.get('c_public_environment_points_gap')):>11s} {fmt(four['commands']):>11s} "
              f"{fmt(four['state read as float32']):>10s} {fmt(four['start']):>9s} {fmt(five.get('stall_aware_points_gap')):>9s}")
    print(f"\n{'gain (reported, not judged)':27s} {'commands':>10s} {'float32':>10s} {'start':>10s}")
    for bank in banks:
        gain = out["T4_gain"][bank]
        print(f"{bank:27s} {gain['commands']:10.2f} {gain['state read as float32']:10.2f} {gain['start']:10.2f}")
    print(f"\nT1: {out['T1']['episodes']} episodes on fixed commands, {out['T1']['episodes_that_differ']} differ")
    print(f"T2: {out['T2']['episodes']} episodes in all, {out['T2']['violations']} bookkeeping violations")
    for bank, e in out["T6"].items():
        print(f"T6 {bank}: {e['states']} states, next state gap {e['next_state_gap']:.2e}, reward gap {e['reward_gap']:.2e}")
    for name, ok in marks.items():
        print(f"  {name:72s} {'pass' if ok else 'FAIL'}")
    for name, ok in superseded.items():
        print(f"  {name:72s} {'pass' if ok else 'fail'}   (reported, not counted)")
    print(f"{'SMOKE, not judged' if smoke else 'VERDICT'} ({args.setup}): {verdict}   ({time.time() - started:.0f} s)")

    precision = None
    if wsl:
        precision = recorded_precision(code_root / "runs" / "shift_hosts")
        print("\nRecorded by the host study for trained policies (read, not played): final actors, closed loop")
        print(f"{'quality / setup':34s} {'policies':>8s} {'within 0.05':>11s} {'largest per start':>17s} {'bank':>15s} {'largest bank mean':>17s}")
        for name, e in precision.items():
            print(f"{name:34s} {e['policies']:8d} {e['within_limit']:11d} {e['largest_per_start_points']:17.4f} "
                  f"{e['worst_bank']:>15s} {e['largest_bank_mean_points']:17.5f}")

    if smoke:
        return
    if wsl:
        arrays = {}
        for bank in BANKS:
            p = pinned[bank]
            arrays.update({f"{bank}/starts": p.starts, f"{bank}/zero": p.zero, f"{bank}/stall_aware": p.stall_aware,
                           f"{bank}/ppu": np.asarray(p.ppu), f"{bank}/wsl_zero": p.played["zero"],
                           f"{bank}/wsl_stall_aware": p.played["stall_aware"]})
        np.savez(REFERENCE, **arrays)
        print("wrote", REFERENCE.name)
    result_path.write_text(json.dumps({
        "written": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "plan": "EPISODE_CHECK.md",
        "setup": args.setup, "seed": SEED, "code_root": str(code_root),
        "interpreter": sys.version.split()[0], "numpy": np.__version__, "gymnasium": gym.__version__,
        "imported_files_sha256": {rel: sha256(code_root / rel) for rel in IMPORTED},
        "tolerances": dict(exact_return=EXACT, points=POINTS, platform_points=PLATFORM_POINTS, starts=STARTS, jax=JAX,
                           identity=IDENTITY, disturbance=DISTURBANCE, gain_limit=GAIN_LIMIT),
        "marks": marks, "superseded_marks": superseded, "verdict": verdict, "measurements": out,
        "recorded_precision_of_trained_policies": precision}, indent=1), newline="\n")
    print("wrote", result_path.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", choices=("wsl", "windows"), required=True)
    parser.add_argument("--code-root", required=True, help="the repository copy whose experiments/ package is imported")
    parser.add_argument("--smoke", action="store_true", help="wiring and timing only; nothing is judged or written")
    run(parser.parse_args())
