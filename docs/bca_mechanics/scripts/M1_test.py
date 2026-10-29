r"""M1 test: the residual, the critic's one-step error y - q at a recorded row.

What this script does
  Part S  Synthetic checks with a known truth, through the repository's own code (algorithms/td3_bc_bca.py
          native_target, lines 146-156, and algorithms/td3_bc.py td3_bc_update, lines 106-121) on random networks and
          random rows. Nothing is trained; the networks are the host's own classes at their random initialization.
          S1  the target y equals the written formula r + 0.98 (1 - d) min(Qbar1, Qbar2)(s', clip(pibar(s') + eps, -1, 1)),
              restated in NumPy from the same noise draw, at 512 rows; rows with d = 1 give y = r exactly; noise 0 gives
              the noise-free formula; y reads the TARGET copies, q = min(Q1, Q2)(s, a) reads the online critic.
          S2  the host's own critic loss (td3_bc_update) equals the loss built from BCA's y under the same key, and
              differs under another key: BCA's y is the host's own target, draw for draw.
          S3  the noise law: eps = clip(Normal(0, 0.2^2), -0.5, 0.5) per row and thruster; SD 0.197, 1.2% of draws
              clipped, a different draw at another key.
          S4  a world whose exact value is known (self-loop states, reward r(s) = s, true value s / (1 - 0.98) for every
              command), pushed through the real native_target: the exact critic gives residual 0; an error b at every
              (state, command) gives residual -(1 - 0.98) b = -0.02 b; an error b only at the recorded rows gives -b; an
              error b only at the next step's rows gives +0.98 b. The residual reports an error only where it changes
              between a row and its next step.
  Part R  The real-data results on the near-80 run's ten version-6 bands (stalling and no-stall worlds, seeds 0-4),
          read from docs/bca_mechanics/data/M1/M1_real_results.json and M1_real_rows.npz, which M1_real.py wrote in the
          pinned environment (see its header; nothing trained, only forward passes of the saved critic): the identity
          checks (rows, saved critic, host loss, scores byte for byte, threshold) and the residual's statistics per
          band, by world, and by thirds of the water current and of the throttle.
  Figure  docs/bca_mechanics/figures/M1.svg (three panels: y against q; the residual's histogram with the unit u;
          the synthetic bias sensitivity).

How to run
  From the repository root with the repository on the import path (Part S needs algorithms/, calibration/, runtime/
  and JAX, flax, optax):
    Windows:  set PYTHONPATH=M:\  && set JAX_PLATFORMS=cpu && python docs\bca_mechanics\scripts\M1_test.py
    POSIX:    PYTHONPATH=. JAX_PLATFORMS=cpu python docs/bca_mechanics/scripts/M1_test.py
  Options: --data <dir> (default docs/bca_mechanics/data/M1, relative to the repository root), --figure <path>
  (default docs/bca_mechanics/figures/M1.svg), --png <path> (an extra raster copy), --no-figure.
  Without the repository importable, Part S falls back to a NumPy restatement of the formula (printed as such) and S2
  is skipped; the numbers quoted in modules/M1.md came from the real code.
  Environment used for modules/M1.md: Python 3.14.2, NumPy 2.5.4, JAX 0.12.0, flax 0.12.10, Matplotlib 3.11.2 on
  Windows, JAX_PLATFORMS=cpu; about 20 seconds. Part R's inputs were produced in the PINNED environment (Python
  3.10.20, NumPy 2.2.6, JAX 0.6.2 on CPU) by M1_real.py through the bridge job M1_real.sh; this script only reads them.
Lines marked [M1.x] carry the module map's labels (docs/bca_mechanics/MODULES.md, section M1). Exit code 0 when every
check passes, 1 otherwise.
"""
import argparse
import collections
import json
import math
import os
import sys
import time
from dataclasses import replace

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

GAMMA, NOISE, CLIP = 0.98, 0.2, 0.5          # ThrusterChain's settings (M1_real_results.json "settings")
FAILS = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)
    return ok


# ------------------------------------------------------------------------------------------ the repository's code

try:
    import jax
    import jax.numpy as jnp
    import algorithms.td3_bc as BASE
    import algorithms.td3_bc_bca as P
    from runtime.networks import Transition
    REAL = True
except Exception as exc:                                   # noqa: BLE001
    REAL = False
    REAL_WHY = repr(exc)
    Transition = collections.namedtuple("Transition", "obs action reward next_obs done")


def synthetic_rows(rng, n, dx=13, da=4):
    """Random rows in ThrusterChain's shapes: 13 state numbers, 4 throttles in [-1, 1], a reward, a next state, and
    d = 1 for about one row in ten (ThrusterChain itself never ends an episode: d is always 0 there)."""
    obs = rng.normal(size=(n, dx)).astype(np.float32)
    action = rng.uniform(-1, 1, size=(n, da)).astype(np.float32)
    reward = rng.normal(size=n).astype(np.float32)
    next_obs = (obs + 0.1 * rng.normal(size=(n, dx))).astype(np.float32)
    done = (rng.uniform(size=n) < 0.1).astype(np.float32)
    return Transition(obs, action, reward, next_obs, done)


def restated_target(reward, done, pi_next, eps, qbar_fn, gamma=GAMMA, clip=CLIP):
    """The formula of the module map, in NumPy float64, from the same noise draw eps (shape (n, da)):
    [M1.1] eps clipped at +-0.5; [M1.2] a' = clip(pibar(s') + eps, -1, 1); [M1.3] y = r + gamma (1 - d) min heads."""
    eps = np.clip(np.asarray(eps, np.float64), -clip, clip)                        # [M1.1 target noise]
    a_next = np.clip(np.asarray(pi_next, np.float64) + eps, -1.0, 1.0)             # [M1.2 next command]
    qbar = np.asarray(qbar_fn(a_next.astype(np.float32)), np.float64)              # (n, 2): the two target heads
    return np.asarray(reward, np.float64) + gamma * (1.0 - np.asarray(done, np.float64)) * qbar.min(axis=1)  # [M1.3]


def part_s():
    print("\nPart S. Synthetic checks with a known truth")
    print(f"  code path: {'the repository code (algorithms/td3_bc_bca.py native_target, algorithms/td3_bc.py td3_bc_update)' if REAL else 'NumPy restatement only (repository not importable: ' + REAL_WHY + ')'}")
    out = {}
    rng = np.random.default_rng(20261020)
    batch = synthetic_rows(rng, 512)

    if REAL:
        args = replace(BASE.Args(), seed=0, dataset="synthetic", discount=GAMMA, policy_noise=NOISE, noise_clip=CLIP,
                       lr=3e-4, alpha=2.5)
        # two independent random initializations: the online networks from seed 0, the target copies from seed 1, so
        # that reading the wrong copy would show
        _, online, (actor, critic) = BASE.initialize(args, 13, 4, 1.0)
        _, other, _ = BASE.initialize(replace(args, seed=1), 13, 4, 1.0)
        native = BASE.AgentTrainState(online.actor, other.actor, online.critic, other.critic)
        models = (actor, critic)
        jb = Transition(*[jnp.asarray(x) for x in batch])
        key = jax.random.PRNGKey(7)

        # S1: the formula
        y = np.asarray(P.native_target(args, models, native, jb, key, 1.0), np.float64)
        eps = np.asarray(jax.random.normal(key, batch.action.shape)) * NOISE
        pi_next = np.asarray(actor.apply(native.actor_target.params, jb.next_obs))
        y_np = restated_target(batch.reward, batch.done, pi_next, eps,
                               lambda a: critic.apply(native.critic_target.params, jb.next_obs, jnp.asarray(a)))
        d1 = batch.done == 1
        y_quiet = np.asarray(P.native_target(replace(args, policy_noise=0.0), models, native, jb, key, 1.0), np.float64)
        y_quiet_np = restated_target(batch.reward, batch.done, pi_next, np.zeros_like(eps),
                                     lambda a: critic.apply(native.critic_target.params, jb.next_obs, jnp.asarray(a)))
        # the wrong copies: y from the online critic / online actor instead of the target copies
        y_wrong_q = restated_target(batch.reward, batch.done, pi_next, eps,
                                    lambda a: critic.apply(native.critic.params, jb.next_obs, jnp.asarray(a)))
        heads = np.asarray(critic.apply(native.critic.params, jb.obs, jb.action), np.float64)     # [M1.4] online critic
        q = heads.min(axis=1)
        heads_t = np.asarray(critic.apply(native.critic_target.params, jb.obs, jb.action), np.float64)
        print("\n  S1. The target y against the formula, 512 random rows (float32 code, float64 restatement)")
        print(f"      |y| up to {np.abs(y).max():.3f}; largest |y - formula| = {np.abs(y - y_np).max():.2e}; "
              f"rows with d = 1: {int(d1.sum())}, largest |y - r| there = {np.abs(y[d1] - batch.reward[d1]).max():.2e}")
        print(f"      noise 0: largest |y - (r + 0.98 (1 - d) min Qbar(s', pibar(s')))| = {np.abs(y_quiet - y_quiet_np).max():.2e}")
        print(f"      had y read the online critic instead of the target copy, the gap would be {np.abs(y - y_wrong_q).max():.3f}")
        print(f"      q = min(Q1, Q2) at the recorded command: heads differ by up to {np.abs(heads[:, 0] - heads[:, 1]).max():.3f}; "
              f"q equals the smaller head in {int(np.all(q == heads.min(axis=1)))*512} of 512 rows; "
              f"online and target critics differ at (s, a) by up to {np.abs(heads - heads_t).max():.3f}")
        check("S1 y equals the formula to float32 rounding", np.abs(y - y_np).max() < 1e-5)
        check("S1 d = 1 gives y = r exactly", d1.any() and np.abs(y[d1] - batch.reward[d1]).max() == 0.0)
        check("S1 noise 0 gives the noise-free formula", np.abs(y_quiet - y_quiet_np).max() < 1e-5)
        check("S1 y reads the target copies (the online copy would differ)", np.abs(y - y_wrong_q).max() > 1e-3)
        out["S1_max_gap"] = float(np.abs(y - y_np).max())

        # S2: the host's own target
        _, metrics = BASE.td3_bc_update(args, actor.apply, critic.apply, native, jb, 1, key, 1.0)
        host_loss = float(metrics["critic_loss"])
        bca_loss = float(np.square(heads - y[:, None]).mean(axis=0).sum())
        _, metrics_other = BASE.td3_bc_update(args, actor.apply, critic.apply, native, jb, 1, jax.random.fold_in(key, 1), 1.0)
        host_other = float(metrics_other["critic_loss"])
        rel = abs(host_loss - bca_loss) / abs(host_loss)
        print("\n  S2. The host's own critic loss against the loss built from BCA's y, same key")
        print(f"      host (td3_bc_update) {host_loss:.7g}; from BCA's y {bca_loss:.7g}; relative gap {rel:.1e}; "
              f"host under another noise key {host_other:.7g} (relative gap {abs(host_other - host_loss) / abs(host_loss):.1e})")
        check("S2 host loss equals the loss from BCA's y (same key)", rel < 1e-5)
        check("S2 another key gives another loss (the equality is not trivial)", abs(host_other - host_loss) / abs(host_loss) > 1e-5)
        out.update(S2_host_loss=host_loss, S2_bca_loss=bca_loss, S2_rel=rel)

        # S3: the noise law
        big = np.asarray(jax.random.normal(jax.random.PRNGKey(11), (25000, 4))) * NOISE
        clipped = np.clip(big, -CLIP, CLIP)
        share_clipped = float(np.mean(np.abs(big) > CLIP))
        expected_share = 2 * (1 - 0.5 * (1 + math.erf(2.5 / math.sqrt(2))))
        y_other = np.asarray(P.native_target(args, models, native, jb, jax.random.fold_in(key, 2), 1.0), np.float64)
        print("\n  S3. The noise law (100,000 draws) and a second draw")
        print(f"      SD of the clipped noise {clipped.std():.4f} (0.2 before clipping, 0.197 expected after); largest |eps| "
              f"{np.abs(clipped).max():.3f} (0.5 cap); share clipped {share_clipped:.4f} (expected {expected_share:.4f})")
        print(f"      SD(y - y with noise 0) {np.std(y - y_quiet):.4f}; SD(y at key A - y at key B) {np.std(y - y_other):.4f} "
              f"(about sqrt 2 times the first for independent draws); SD(y) {np.std(y):.4f}")
        check("S3 noise is clipped at 0.5 with SD near 0.197", np.abs(clipped).max() <= CLIP and abs(clipped.std() - 0.197) < 0.004)
        check("S3 another key gives another draw of the same law", np.std(y - y_other) > 0 and 1.2 < np.std(y - y_other) / max(np.std(y - y_quiet), 1e-12) < 1.65)
        out.update(S3_sd=float(clipped.std()), S3_share_clipped=share_clipped)
    else:
        print("  S1-S3 skipped (they call the repository's code); S4 runs on the NumPy restatement.")
        args = None

    # S4: a world with a known exact value
    print("\n  S4. A world with a known value: self-loop states, reward r(s) = s, true value s / (1 - 0.98) at every command")
    n = 4096
    s = rng.uniform(-1, 1, size=n).astype(np.float32)
    a_rec = rng.uniform(0.8, 1.0, size=(n, 1)).astype(np.float32)        # the recorded command, far from pibar = 0
    world = Transition(s[:, None], a_rec, s, s[:, None], np.zeros(n, np.float32))

    def known_critic(kind, b):
        def apply(params, obs, action):
            base = obs[:, 0] / (1.0 - GAMMA)
            if kind == "exact":
                bias = 0.0
            elif kind == "everywhere":
                bias = b
            elif kind == "recorded rows only":
                bias = b * (action[:, 0] > 0.75)       # the recorded commands are in [0.8, 1]; next commands in [-0.5, 0.5]
            elif kind == "next step only":
                bias = b * (action[:, 0] <= 0.75)
            else:
                raise ValueError(kind)
            q = base + bias
            return np.stack([q, q], axis=-1) if isinstance(q, np.ndarray) else jnp.stack([q, q], axis=-1)
        return apply

    class Actor0:                                            # pibar(s) = 0 for every state
        @staticmethod
        def apply(params, obs):
            return jnp.zeros((obs.shape[0], 1), jnp.float32) if REAL else np.zeros((obs.shape[0], 1), np.float32)

    class Net:
        def __init__(self, fn):
            self.apply = fn

    class Holder:
        def __init__(self):
            self.params = None

    def residual(kind, b):
        fn = known_critic(kind, b)
        if REAL:
            nat = BASE.AgentTrainState(Holder(), Holder(), Holder(), Holder())
            jw = Transition(*[jnp.asarray(x) for x in world])
            y = np.asarray(P.native_target(args, (Actor0, Net(fn)), nat, jw, jax.random.PRNGKey(3), 1.0), np.float64)
            qv = np.asarray(fn(None, jw.obs, jw.action), np.float64).min(axis=1)                 # [M1.4]
        else:
            eps = rng.normal(size=(n, 1)) * NOISE
            y = restated_target(world.reward, world.done, np.zeros((n, 1)), eps, lambda a: fn(None, world.obs, a))
            qv = fn(None, world.obs, world.action).min(axis=1)
        return y - qv                                                                              # [M1.5 residual]

    rows = []
    e0 = residual("exact", 0.0)
    print(f"      exact critic: largest |residual| {np.abs(e0).max():.2e} (float32 rounding of values up to 50)")
    check("S4 the exact critic has residual 0", np.abs(e0).max() < 1e-3)
    biases = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)
    slopes = {"everywhere": -(1 - GAMMA), "recorded rows only": -1.0, "next step only": GAMMA}
    print("      bias b in the critic ->  mean residual (expected): " + "; ".join(f"b = {b:+.1f}" for b in biases))
    for kind, slope in slopes.items():
        means = [float(residual(kind, b).mean()) for b in biases]
        sds = [float(residual(kind, b).std()) for b in biases]
        rows.append((kind, biases, means))
        print(f"      {kind:19s} " + "  ".join(f"{m:+.4f} ({slope * b:+.3f})" for m, b in zip(means, biases))
              + f"   [SD within a case at most {max(sds):.1e}]")
        check(f"S4 bias {kind}: residual = {slope:+.2f} x bias", all(abs(m - slope * b) < 2e-3 for m, b in zip(means, biases)))
    out["S4"] = rows
    print("      reading: a critic that is too high by 1.0 everywhere shows a residual of only -0.02; the residual sees an "
          "error only where it changes between the row and its next step.")
    return out


# ------------------------------------------------------------------------------------------ Part R: real data

def part_r(data_dir):
    print("\nPart R. Real data: the near-80 run's ten version-6 bands (recomputed in the pinned environment by M1_real.py)")
    jpath, npath = os.path.join(data_dir, "M1_real_results.json"), os.path.join(data_dir, "M1_real_rows.npz")
    if not (os.path.exists(jpath) and os.path.exists(npath)):
        print(f"  skipped: {jpath} or {npath} missing")
        return None
    with open(jpath) as f:
        rec = json.load(f)
    arrays = dict(np.load(npath))
    env = rec["environment"]
    print(f"  written {rec['written']} from {rec['run']} with Python {env['python']}, NumPy {env['numpy']}, JAX {env['jax']}, "
          f"JAX_PLATFORMS={env['JAX_PLATFORMS']}, OMP_NUM_THREADS={env['OMP_NUM_THREADS']}; settings {rec['settings']}")
    bands = [b for b in rec["bands"] if not b.get("failed")]
    print(f"  bands on record: {len(rec['bands'])}, failed: {len(rec['bands']) - len(bands)}")

    print("\n  Table R1. Identity checks per band (True = equal)")
    hdr = f"  {'world':9s} {'seed':>4s} {'n':>4s} {'rows':>5s} {'critic':>6s} {'native':>6s} {'host loss':>11s} {'from BCA y':>11s} {'rel gap':>8s} {'scores':>6s} {'eta':>5s} {'sha':>5s} {'R recorded':>11s} {'R recomputed':>12s} {'equal':>5s}"
    print(hdr)
    all_ok = True
    for b in bands:
        ok = (b["rows_hashes_equal"] and b["critic_sha_equal"] and b["native_sha_equal"] and b["loss_rel_diff"] < 1e-5
              and b["scores_equal"] and b["eta_equal"] and b["scores_sha_equal"] and b["R_equal_float32"] and b["all_finite"])
        all_ok &= ok
        print(f"  {b['world']:9s} {b['seed']:4d} {b['n']:4d} {str(b['rows_hashes_equal']):>5s} {str(b['critic_sha_equal']):>6s} "
              f"{str(b['native_sha_equal']):>6s} {b['host_loss']:11.7g} {b['bca_loss']:11.7g} {b['loss_rel_diff']:8.1e} "
              f"{str(b['scores_equal']):>6s} {str(b['eta_equal']):>5s} {str(b['scores_sha_equal']):>5s} {b['R_recorded']:11.6f} "
              f"{b['R_recomputed']:12.6f} {str(b['R_equal_float32']):>5s}")
    check("R1 every band: rows, critic, host loss, scores (byte for byte), eta, threshold all reproduce", all_ok)
    check("R1 d is 0 at every held-out row (ThrusterChain never ends an episode)", all(b["done_all_zero"] for b in bands))
    check("R1 every y and e is finite", all(b["all_finite"] for b in bands))

    print("\n  Table R2. The residual e = y - q at the held-out rows, per band (return units unless noted)")
    print(f"  {'world':9s} {'seed':>4s} {'n':>4s} {'unit u':>7s} {'SD(e)':>6s} {'mean e':>7s} {'e>0':>5s} {'med|e|':>7s} {'90%|e|':>7s} {'max|e|':>7s} "
          f"{'|e|<=u eta':>10s} {'|e|<=U':>7s} {'SD noise':>8s} {'SD(e) fresh':>11s} {'q range':>16s} {'r range':>14s}")
    stats = []
    for b in bands:
        print(f"  {b['world']:9s} {b['seed']:4d} {b['n']:4d} {b['unit']:7.4f} {b['sd_e']:6.4f} {b['mean_e']:+7.4f} {b['share_e_positive']:5.3f} "
              f"{b['median_abs_e']:7.4f} {b['q90_abs_e']:7.4f} {b['max_abs_e']:7.4f} {b['share_within_unit_eta']:10.3f} {b['share_within_band']:7.3f} "
              f"{b['sd_noise_part']:8.4f} {b['sd_e_fresh']:11.4f} {b['q_min']:7.3f}..{b['q_max']:6.3f} {b['r_min']:6.3f}..{b['r_max']:6.3f}")
        stats.append(b)
    for world in ("stall", "no_stall"):
        ws = [b for b in bands if b["world"] == world]
        if ws:
            print(f"  {world:9s} mean  {'':4s} {np.mean([b['n'] for b in ws]):4.0f} {np.mean([b['unit'] for b in ws]):7.4f} {np.mean([b['sd_e'] for b in ws]):6.4f} "
                  f"{np.mean([b['mean_e'] for b in ws]):+7.4f} {np.mean([b['share_e_positive'] for b in ws]):5.3f} {np.mean([b['median_abs_e'] for b in ws]):7.4f} "
                  f"{np.mean([b['q90_abs_e'] for b in ws]):7.4f} {np.mean([b['max_abs_e'] for b in ws]):7.4f} {np.mean([b['share_within_unit_eta'] for b in ws]):10.3f} "
                  f"{np.mean([b['share_within_band'] for b in ws]):7.3f} {np.mean([b['sd_noise_part'] for b in ws]):8.4f} {np.mean([b['sd_e_fresh'] for b in ws]):11.4f}")
    check("R2 the band at the stored R covers 90% or more of the held-out residuals in every band",
          all(b["share_within_band"] >= 0.90 for b in bands))
    ratios = [b["unit"] / b["sd_e"] for b in bands]
    check("R2 the unit u is within a factor 2 of SD(e) on the held-out rows in every band (a sanity bound; the unit's fit is M2's)",
          all(0.5 < r < 2.0 for r in ratios), f"u / SD(e) from {min(ratios):.2f} to {max(ratios):.2f}")
    print("      columns: unit u = the frozen running scale u_f of the band (M2); |e|<=u eta = share of rows inside the band at "
          "R = 1; |e|<=U = share inside the band U = R u eta (the held-out coverage, 0.90 expected or a little above); SD noise = "
          "SD of (y - y with noise 0); SD(e) fresh = SD(e) with another noise draw.")

    print("\n  Table R3. The residual by thirds of the water current and of the throttle (each third holds a third of the band's rows)")
    print("      share>q90 = share of the third's rows whose |e| exceeds the band's overall 90% quantile of |e| (0.100 for an "
          "ordinary third); med|e| = the third's median |e|; mean e > 0 means the critic's value sits below its own target.")
    print(f"  {'world':9s} {'seed':>4s} {'n':>4s} {'full>=0.9':>9s} | {'current thirds: cuts':>20s} {'share>q90 low/mid/high':>24s} {'med|e| low/mid/high':>22s} "
          f"| {'throttle thirds: cuts':>21s} {'share>q90 low/mid/high':>24s} {'med|e| low/mid/high':>22s}")
    groups = []
    for b in bands:
        tag = f"{b['world']}_seed{b['seed']}"
        e, thr, cur = arrays[tag + "_e"].astype(np.float64), arrays[tag + "_throttle"], arrays[tag + "_current"]
        q90 = np.quantile(np.abs(e), 0.9)
        g = dict(world=b["world"], seed=b["seed"], n=len(e), n_full=int(np.sum(thr >= 0.9)))
        for name, x in (("current", cur), ("throttle", thr)):
            cuts = np.quantile(x, [1 / 3, 2 / 3])
            third = np.digitize(x, cuts)
            g[name] = dict(cuts=[float(c) for c in cuts],
                           above=[float(np.mean(np.abs(e[third == k]) > q90)) for k in range(3)],
                           med=[float(np.median(np.abs(e[third == k]))) for k in range(3)],
                           mean=[float(e[third == k].mean()) for k in range(3)])
        groups.append(g)
        c, t = g["current"], g["throttle"]
        print(f"  {g['world']:9s} {g['seed']:4d} {g['n']:4d} {g['n_full']:9d} | {c['cuts'][0]:9.3f} {c['cuts'][1]:10.3f} "
              f"{c['above'][0]:8.3f}{c['above'][1]:8.3f}{c['above'][2]:8.3f} {c['med'][0]:7.4f}{c['med'][1]:8.4f}{c['med'][2]:7.4f} "
              f"| {t['cuts'][0]:10.3f} {t['cuts'][1]:10.3f} {t['above'][0]:8.3f}{t['above'][1]:8.3f}{t['above'][2]:8.3f} "
              f"{t['med'][0]:7.4f}{t['med'][1]:8.4f}{t['med'][2]:7.4f}")
    for world in ("stall", "no_stall"):
        ws = [g for g in groups if g["world"] == world]
        if ws:
            ca = np.mean([g["current"]["above"] for g in ws], axis=0)
            cm = np.mean([g["current"]["med"] for g in ws], axis=0)
            ta = np.mean([g["throttle"]["above"] for g in ws], axis=0)
            tm = np.mean([g["throttle"]["med"] for g in ws], axis=0)
            print(f"  {world:9s} mean {'':4s} {'':9s} | {'':20s} {ca[0]:8.3f}{ca[1]:8.3f}{ca[2]:8.3f} {cm[0]:7.4f}{cm[1]:8.4f}{cm[2]:7.4f} "
                  f"| {'':21s} {ta[0]:8.3f}{ta[1]:8.3f}{ta[2]:8.3f} {tm[0]:7.4f}{tm[1]:8.4f}{tm[2]:7.4f}")
    rising = sum(g["current"]["above"][2] > g["current"]["above"][0] for g in groups)
    print(f"      full>=0.9 = rows with a throttle of 90% or more among the held-out rows: {min(g['n_full'] for g in groups)} to "
          f"{max(g['n_full'] for g in groups)} per band, too few for a group of their own (the harm switch, M10, reads them in the "
          f"training rows instead). Large residuals are more common in the high-current third than in the low-current third in "
          f"{rising} of {len(groups)} bands.")
    return dict(record=rec, bands=bands, arrays=arrays, groups=groups)


# ------------------------------------------------------------------------------------------ the figure

def figure(path, s_out, r_out, png=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"svg.fonttype": "none", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8a86",
                         "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.axisbelow": True,
                         "legend.frameon": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
    BLUE, ORANGE, AQUA, VIOLET, YELLOW, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#eda100", "#0b0b0b", "#52514e"
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)
    fig.suptitle("Module M1, the residual: the critic's one-step error y - q at a recorded row", fontsize=11, color=INK)

    ax = axes[0]
    if r_out is not None and "stall_seed0_q" in r_out["arrays"]:
        a = r_out["arrays"]
        q, y = a["stall_seed0_q"], a["stall_seed0_y"]
        lo, hi = float(min(q.min(), y.min())), float(max(q.max(), y.max()))
        ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1, linestyle="--", label="y = q (no error)")
        ax.scatter(q, y, s=7, color=BLUE, alpha=0.55, linewidths=0, label=f"held-out rows, stalling world, seed 0 (n = {len(q)})")
        b0 = next(b for b in r_out["bands"] if b["world"] == "stall" and b["seed"] == 0)
        ax.text(0.03, 0.97, f"SD of y - q: {b0['sd_e']:.3f}\nunit u: {b0['unit']:.3f}", transform=ax.transAxes, va="top", color=INK)
        ax.set_title("A. Target against value, one band")
        ax.set_xlabel("critic's value q = min(Q1, Q2)(s, a)  (return units)")
        ax.set_ylabel("host's target y = r + 0.98 min Qbar(s', a')  (return units)")
        ax.legend(loc="lower right")
    else:
        ax.set_title("A. (no real-data arrays found)")

    ax = axes[1]
    if r_out is not None:
        a = r_out["arrays"]
        for world, color, name in (("stall", BLUE, "stalling world"), ("no_stall", ORANGE, "no-stall world")):
            tag = f"{world}_seed0_e"
            if tag not in a:
                continue
            e = a[tag]
            b0 = next(b for b in r_out["bands"] if b["world"] == world and b["seed"] == 0)
            ax.hist(e, bins=np.linspace(-1.0, 1.0, 61), histtype="step", linewidth=1.6, color=color,
                    label=f"{name}, seed 0 (n = {len(e)}, u = {b0['unit']:.3f})")
            for sgn in (-1, 1):
                ax.axvline(sgn * b0["unit"], color=color, linewidth=0.9, linestyle=":")
        ax.plot([], [], color=MUTED, linewidth=0.9, linestyle=":", label="dotted: +- the unit u of each band")
        ax.set_title("B. The residual at the held-out rows")
        ax.set_xlabel("residual y - q  (return units)")
        ax.set_ylabel("held-out rows")
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.3)          # headroom for the legend above the peak
        ax.legend(loc="upper right", fontsize=8)

    ax = axes[2]
    if s_out is not None and "S4" in s_out:
        styles = {"everywhere": (AQUA, "-", "error b at every (state, command): slope -0.02"),
                  "recorded rows only": (VIOLET, "--", "error b only at the recorded rows: slope -1"),
                  "next step only": (YELLOW, "-.", "error b only at the next step: slope +0.98")}
        for kind, biases, means in s_out["S4"]:
            color, ls, label = styles[kind]
            ax.plot(biases, means, color=color, linestyle=ls, linewidth=2, marker="o", markersize=5, label=label)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.set_title("C. Known-value world: what the residual sees")
        ax.set_xlabel("error b added to the critic's value  (return units)")
        ax.set_ylabel("mean residual y - q  (return units)")
        ax.legend(loc="upper right", fontsize=8)
    fig.savefig(path, format="svg")
    if png:
        fig.savefig(png, dpi=130)
    print(f"\n  figure written: {path}" + (f" and {png}" if png else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "docs", "bca_mechanics", "data", "M1"))
    ap.add_argument("--figure", default=os.path.join(ROOT, "docs", "bca_mechanics", "figures", "M1.svg"))
    ap.add_argument("--png", default=None)
    ap.add_argument("--no-figure", action="store_true")
    opt = ap.parse_args()
    t0 = time.perf_counter()
    print("M1 test: the residual y - q")
    print(f"  Python {sys.version.split()[0]}, NumPy {np.__version__}" + (f", JAX {jax.__version__}" if REAL else ""))
    s_out = part_s()
    r_out = part_r(opt.data)
    if not opt.no_figure:
        os.makedirs(os.path.dirname(opt.figure), exist_ok=True)
        figure(opt.figure, s_out, r_out, opt.png)
    print(f"\n{'ALL CHECKS PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}  ({time.perf_counter() - t0:.0f} s)")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
