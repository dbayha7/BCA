"""M8 test: the width penalty on critic values (toy worlds only), a = argmax_a [q(t, c, a) - 1.0 * U(t, c, a)].

What this script does
  Part A  Reproduces the toy world of bca_channel_toy.ipynb (branch main, cells 2, 4, 6, 8, 12, 13, 15, 17, 19) with
          NumPy only: 200 seeds, 7 evidence sizes, 2 harm shapes, 10 worlds. It prints the pooled regret of the
          width-penalty rule A1, of A2 (the same width inside the behavior-cloning multiplier), of the world-blind
          rules and of the oracle, the "room" and each rule's gain / room, and applies kill rule test 1
          (gain at least a quarter of the room) exactly as cell 17 does. Every number must equal the notebook's.
  Part S  Synthetic checks with a known truth, on the same toy: S1 the penalty equals the band's half-width and the
          decision follows the closed-form boundary in 100% of cells; S2 the width at the untried action does not
          follow that action's true outcome (one shared threshold); S3 the exact decomposition of A1's regret into
          the cost at cells the data cover well and the cost at the gap; S4 the widths are calibrated where they
          are measured (coverage of fresh logged rows, the notebook's check 6).
  Part B  The real-data test on ThrusterChain's Stage 2 records (seeds 0-4, stalling world, fixed critic): for every
          (state, actor's command, its 80% cap) pair, does Q1 - U prefer the cap when the command is harmful? Uses the
          recorded Q1 values and the recorded band widths; nothing is trained. Skipped when no path is given.
  Figure  docs/bca_mechanics/figures/M8.svg (three panels).

How to run (any Python 3.10+ with NumPy and Matplotlib; no repository import; about 8 s, 3 to 5 s of them for the toy)
  Toy parts only:
    python docs/bca_mechanics/scripts/M8_test.py
  With the real-data part, from the recorded files (read only):
    python docs/bca_mechanics/scripts/M8_test.py --runs <root>
        root holds runs/shift_explore/detect/seed<K>/stall/{pairs,labels}.npz and
        runs/shift_explore/stage2/detect/{uniform,weighted}/stall/fixed/seed<K>/scores/bca_width_<ver>_fixed.npz
        (the owner's WSL clone /home/user/bca holds them). The command used for docs/bca_mechanics/modules/M8.md,
        from the worktree root in Git Bash on the owner's Windows desktop, with PYTHONDONTWRITEBYTECODE=1 set:
          python docs/bca_mechanics/scripts/M8_test.py --runs //wsl.localhost/Ubuntu-24.04/home/user/bca
    python docs/bca_mechanics/scripts/M8_test.py --copy <dir>
        dir holds pairs/seed<K>/{pairs,labels}.npz and {uniform,weighted}/stall/seed<K>/scores/bca_width_<ver>_fixed.npz
  Environment used for the numbers in docs/bca_mechanics/modules/M8.md: Python 3.14, NumPy 2.5.4, Matplotlib 3.11.2
  (Windows). The toy's random streams are NumPy's default_rng with SeedSequence, stable across these versions; the
  notebook itself ran under the repository's .venv (Python 3.10, NumPy 2.2.6) and printed the same numbers.

Verbatim copies: the functions marked "copied" are the notebook's own code (cell numbers given), unchanged, so that
the toy here is the toy there. Lines marked [M8.x] carry the module map's labels (docs/bca_mechanics/MODULES.md).
"""
import argparse, math, os, time
from fractions import Fraction
from functools import lru_cache
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(os.path.dirname(HERE), "figures", "M8.svg")

# ----------------------------------------------------------------------------------------------------------------
# Constants: copied from bca_channel_toy.ipynb cell 2 (the gate-only constants left out).
# ----------------------------------------------------------------------------------------------------------------
ALPHA, BETA, DRAWS = 0.1, 0.95, 1000        # BCA settings as in production
SIGMA = 0.2                                 # sd of the Gaussian noise on every reward
P_LOG = np.array([0.85, 0.15])              # context mix in the log
P_TEST = np.array([0.35, 0.65])             # context mix at deployment
N_LOG, TRAIN_SHARE, N_MIN = 1800, 2 / 3, 20
EVIDENCE = (5, 10, 20, 30, 45, 60, 100)     # survivors available to calibration
ADVANTAGE = (1.0, 0.5, 0.25, 0.1, -0.1, -0.25, -0.5, -1.0, -2.0, -4.6)
HARMS = ("shortfall", "tail")
N_SEEDS, ROOT_SEED = 200, 20261014
GAIN = 1.0                                  # A1: critic minus GAIN x width
BC_WEIGHT, BLEND = 0.4, 0.5                 # host BC weight; A2 multiplier 1 + BLEND x U / (U + unit)
SAFE, VENTURE = 0, 1
CATASTROPHE = -5.0
A_FRAC, B_FRAC = Fraction(str(ALPHA)), Fraction(str(BETA))

# ----------------------------------------------------------------------------------------------------------------
# The toy world: copied from cell 4.
# ----------------------------------------------------------------------------------------------------------------
def gap_law(harm, adv):
    if harm == "shortfall":
        return [(1.0, 1.0 + adv)]
    p = (1.0 - adv) / 7.0
    return [(p, CATASTROPHE), (1.0 - p, 2.0)]

def law(t, c, a, harm="tail", adv=-4.6):
    if t == 0:
        if a == SAFE: return [(1.0, 0.1)]
        return [(1.0, 0.3)] if c == 0 else [(0.5, -0.5), (0.5, 1.1)]
    if a == SAFE: return [(1.0, 1.0)]
    if c == 0: return [(0.5, 0.0), (0.5, 3.0)]
    return gap_law(harm, adv)

def mean_of(t, c, a, harm="tail", adv=-4.6):
    return sum(p * m for p, m in law(t, c, a, harm, adv))

def phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def below_safe(harm, adv):
    return sum(p * phi((1.0 - m) / SIGMA) for p, m in gap_law(harm, adv))

def draw(lawlist, u, z):
    cum = np.cumsum([p for p, _ in lawlist])
    k = np.minimum(np.searchsorted(cum, u, side="right"), len(lawlist) - 1)
    return np.array([m for _, m in lawlist])[k] + SIGMA * z

def collect(seed):
    rng = np.random.default_rng([ROOT_SEED, seed])
    c = rng.choice(2, N_LOG, p=P_LOG)
    a = np.zeros((N_LOG, 2), dtype=int)
    a[:, 0] = rng.random(N_LOG) < 0.5
    a[:, 1] = (c == 0) & (rng.random(N_LOG) < 0.25)
    u, z = rng.random((N_LOG, 2)), rng.standard_normal((N_LOG, 2))
    r = np.zeros((N_LOG, 2))
    for t in range(2):
        for cc in range(2):
            for aa in range(2):
                m = (c == cc) & (a[:, t] == aa)
                if m.any(): r[m, t] = draw(law(t, cc, aa), u[m, t], z[m, t])
    train = rng.random(N_LOG) < TRAIN_SHARE
    split = lambda m: {"c": c[m], "a": a[m], "r": r[m]}
    survivors = {"u": rng.random(100), "z": rng.standard_normal(100)}
    return split(train), split(~train), survivors

def survivor_rewards(surv, harm, adv, n):
    return draw(gap_law(harm, adv), surv["u"][:n], surv["z"][:n])

# ----------------------------------------------------------------------------------------------------------------
# The critic and the width's scale: copied from cell 6.
# ----------------------------------------------------------------------------------------------------------------
def targets(data, t, q):
    y = data["r"][:, t].copy()
    if t == 0: y += q[1, data["c"], SAFE]
    return y

def fit_critic(train):
    q, n = np.zeros((2, 2, 2)), np.zeros((2, 2, 2), dtype=int)
    for t in (1, 0):
        y = targets(train, t, q)
        for c in range(2):
            for a in range(2):
                m = (train["c"] == c) & (train["a"][:, t] == a)
                n[t, c, a] = m.sum()
                if n[t, c, a] >= N_MIN: q[t, c, a] = y[m].mean()
        for c in range(2):
            for a in range(2):
                if n[t, c, a] < N_MIN:
                    o = 1 - c
                    assert t == 1 and min(n[1, c, SAFE], n[1, o, a], n[1, o, SAFE]) >= N_MIN
                    q[1, c, a] = q[1, c, SAFE] + q[1, o, a] - q[1, o, SAFE]
    return q, n

def fit_scale(train, q):
    scale, unit = np.zeros((2, 2, 2)), np.zeros(2)
    for t in range(2):
        res = targets(train, t, q) - q[t, train["c"], train["a"][:, t]]
        pooled, unit[t] = res.var(), res.std()
        for c in range(2):
            for a in range(2):
                m = (train["c"] == c) & (train["a"][:, t] == a)
                k = int(m.sum()); v = res[m].var() if k > 1 else 0.0
                scale[t, c, a] = max(0.05, math.sqrt((k * v + 5 * pooled) / (k + 5)))
    return scale, unit

# ----------------------------------------------------------------------------------------------------------------
# BCA's exact threshold (equal weights, infinitely many draws): copied from cell 8.
# ----------------------------------------------------------------------------------------------------------------
def log_pmf(x, n, p):
    return (math.lgamma(n + 1) - math.lgamma(x + 1) - math.lgamma(n - x + 1)
            + x * math.log(p) + (n - x) * math.log1p(-p))

@lru_cache(maxsize=None)
def certified_above(n):
    cdf, best = 0.0, -1
    for x in range(n + 1):
        cdf += math.exp(log_pmf(x, n, ALPHA))
        if cdf > 1 - BETA: break
        best = x
    return best

def bca_exact(scores):
    """[M8.1 threshold] max(plain 90% quantile, the score with x scores above it, x the largest with
    P(Binomial(n, 0.1) <= x) <= 0.05); +infinity when n < 29."""
    s = np.sort(np.asarray(scores, dtype=float)); n = s.size
    x = certified_above(n)
    hpd = s[n - x - 1] if x >= 0 else math.inf
    return max(s[math.ceil((1 - A_FRAC) * n) - 1], hpd)

# ----------------------------------------------------------------------------------------------------------------
# The rules: copied from cell 12 (the gates left out; A1, A2 and the world-blind rules kept line for line).
# ----------------------------------------------------------------------------------------------------------------
METHODS = ["oracle switch", "greedy", "range limit", "count penalty", "host BC actor",
           "A1 width penalty", "A2 width BC multiplier"]
WORLD_BLIND = ["greedy", "range limit", "count penalty", "host BC actor"]
M = {m: i for i, m in enumerate(METHODS)}
ADV = np.array(ADVANTAGE)
MU = np.array([[[mean_of(t, c, a) for a in range(2)] for c in range(2)] for t in range(2)])

def regret(pi, adv):
    mu = MU.copy(); mu[1, 1, VENTURE] = 1.0 + adv
    got = np.take_along_axis(mu, pi[..., None], axis=-1)[..., 0]
    return float(((mu.max(axis=-1) - got) * P_TEST[None, :]).sum())

def bc_policy(q, n, dose):
    beta_hat = n / n.sum(axis=-1, keepdims=True)
    leave = (beta_hat * dose).sum(axis=-1, keepdims=True) - beta_hat * dose
    return (q - BC_WEIGHT * leave).argmax(axis=-1)

def run_seed(seed):
    train, cal, surv = collect(seed)
    q, n = fit_critic(train)
    scale, unit = fit_scale(train, q)
    greedy = q.argmax(axis=-1)
    assert n[1, 1, VENTURE] == 0 and greedy[1, 1] == VENTURE
    claimed = q[1, 1, VENTURE] - q[1, 1, SAFE]
    s_rest = q[1, cal["c"], cal["a"][:, 1]] - cal["r"][:, 1]
    a0, a1 = cal["a"][:, 0], cal["a"][:, 1]
    # [M8.1 threshold, step 0] scores |y - q| / scale on the held-out rows of step 0 (cell 12 line 51)
    r0 = bca_exact(np.abs(targets(cal, 0, q) - q[0, cal["c"], a0]) / scale[0, cal["c"], a0])
    bank1 = np.abs(s_rest) / scale[1, cal["c"], a1]                       # cell 12 line 52
    fixed = {"greedy": greedy,
             "range limit": np.where(n >= N_MIN, q, -np.inf).argmax(axis=-1),
             "count penalty": (q - GAIN / np.sqrt(n + 1)).argmax(axis=-1),
             "host BC actor": bc_policy(q, n, np.ones_like(q))}
    reg = np.zeros((len(EVIDENCE), len(HARMS), len(ADVANTAGE), len(METHODS)))
    dev = np.zeros(reg.shape, dtype=bool)
    widths = np.zeros((len(EVIDENCE), len(HARMS), len(ADVANTAGE), 2, 2, 2))
    r1_seen = []
    for i, ne in enumerate(EVIDENCE):
        for h, harm in enumerate(HARMS):
            for w, adv in enumerate(ADVANTAGE):
                s_gap = q[1, 1, VENTURE] - survivor_rewards(surv, harm, adv, ne)
                pol = dict(fixed)
                mu = MU.copy(); mu[1, 1, VENTURE] = 1.0 + adv
                pol["oracle switch"] = mu.argmax(axis=-1)
                # [M8.1 threshold, step 1] ONE threshold on all held-out scores of step 1, survivors included (line 88)
                r1 = bca_exact(np.concatenate([bank1, np.abs(s_gap) / scale[1, 1, VENTURE]]))
                r1_seen.append(r1)
                # [M8.2 width] U(t, c, a) = r_t * scale(t, c, a) (line 90)
                width = np.stack([r0 * scale[0], r1 * scale[1]])
                assert np.all(np.isfinite(width))
                # [M8.3 penalty] a = argmax_a [q - 1.0 * U] at every (t, c) (line 92); A2 = the dose of M7 (line 93)
                pol["A1 width penalty"] = (q - GAIN * width).argmax(axis=-1)
                pol["A2 width BC multiplier"] = bc_policy(q, n, 1 + BLEND * width / (width + unit[:, None, None]))
                for name, pi in pol.items():
                    reg[i, h, w, M[name]] = regret(pi, adv)
                    dev[i, h, w, M[name]] = pi[1, 1] == VENTURE
                widths[i, h, w] = width
    info = dict(claimed=claimed, q_gap=q[1, 1, VENTURE], q_safe=q[1, 1, SAFE], r0=r0,
                r1_min=min(r1_seen), r1_max=max(r1_seen), n_rest=s_rest.size)
    return reg, dev, info, (q, n, scale, unit, widths)

# ----------------------------------------------------------------------------------------------------------------
# Part A: the notebook's own numbers.
# ----------------------------------------------------------------------------------------------------------------
def part_a():
    t0 = time.perf_counter()
    REG = np.zeros((N_SEEDS, len(EVIDENCE), len(HARMS), len(ADVANTAGE), len(METHODS)))
    DEV = np.zeros(REG.shape, dtype=bool)
    INFO, fits = [], []
    for seed in range(N_SEEDS):
        REG[seed], DEV[seed], info, fit = run_seed(seed)
        INFO.append(info); fits.append(fit)
    print(f"Part A  the toy of bca_channel_toy.ipynb, {N_SEEDS} seeds in {time.perf_counter() - t0:.1f} s "
          f"(the notebook, with its gates, took 20.9 s)")
    print("  Sanity values over seeds (notebook cell 13):")
    for key in ("claimed", "q_gap", "q_safe", "r0", "r1_min", "r1_max", "n_rest"):
        v = np.array([d[key] for d in INFO], dtype=float)
        print(f"    {key:8} mean {v.mean():8.3f}   min {v.min():8.3f}   max {v.max():8.3f}")
    pooled = REG.mean(axis=(0, 1, 2, 3))
    best_wb = min(WORLD_BLIND, key=lambda m: pooled[M[m]])
    room = pooled[M[best_wb]] - pooled[M["oracle switch"]]
    gain = {m: pooled[M[best_wb]] - pooled[M[m]] for m in METHODS}
    print(f"\n  Table A1. Pooled regret (200 seeds x 7 evidence sizes x 2 harm shapes x 10 worlds); lower is better."
          f"\n  Best world-blind rule: {best_wb}.  Room = {room:.4f}.  Kill rule test 1: gain / room must be at least 0.25.")
    UNSAFE = np.array([[below_safe(h, a) > ALPHA for a in ADVANTAGE] for h in HARMS])
    unsafe_mask = np.broadcast_to(UNSAFE[None, None], DEV.shape[:4])
    print(f"  {'rule':26} {'pooled regret':>13} {'gain / room':>11} {'ventures at the gap in unsafe worlds':>37}")
    for m in METHODS:
        print(f"  {m:26} {pooled[M[m]]:13.4f} {gain[m] / room:11.2f} {DEV[..., M[m]][unsafe_mask].mean():37.3f}")
    for m in ("A1 width penalty", "A2 width BC multiplier"):
        shelved = gain[m] < 0.25 * room
        print(f"  Test 1, {m}: gain / room = {gain[m] / room:+.2f} -> {'SHELVED' if shelved else 'ALIVE'}")
    print("\n  Table A2. Pooled regret by number of held-out tries of the untried action (notebook cell 19):")
    print(f"  {'survivors':26}" + "".join(f"{ne:>8d}" for ne in EVIDENCE))
    for m in ("range limit", "A1 width penalty", "A2 width BC multiplier"):
        print(f"  {m:26}" + "".join(f"{REG[:, i, ..., M[m]].mean():8.4f}" for i in range(len(EVIDENCE))))
    print("  Share of (seed, harm shape) where the rule's decision at the gap changes somewhere along the 10 worlds:")
    for m in ("range limit", "A1 width penalty", "A2 width BC multiplier"):
        d = DEV[..., M[m]]; varies = d.any(axis=-1) & ~d.all(axis=-1)
        print(f"  {m:26}" + "".join(f"{varies[:, i].mean():8.2f}" for i in range(len(EVIDENCE))))
    return REG, DEV, INFO, fits, pooled, room, gain

# ----------------------------------------------------------------------------------------------------------------
# Part S: synthetic checks with a known truth.
# ----------------------------------------------------------------------------------------------------------------
def part_s(REG, DEV, fits, pooled):
    print("\nPart S  synthetic checks with a known truth (the same toy)")
    i30, hS, wP = EVIDENCE.index(30), HARMS.index("shortfall"), ADVANTAGE.index(0.5)
    # S1: the penalty is the half-width itself (gain 1), and the decision is the closed-form boundary.
    agree = total = 0
    for seed in range(N_SEEDS):
        q, n, scale, unit, widths = fits[seed]
        W = widths.reshape(-1, 2, 2, 2)
        pen = GAIN * W
        assert np.array_equal(pen, W)
        choice = (q[None] - pen).argmax(axis=-1)
        closed = ((q[None, :, :, VENTURE] - q[None, :, :, SAFE]) > (W[..., VENTURE] - W[..., SAFE])).astype(int)
        agree += int((choice == closed).sum()); total += choice.size
    print(f"  S1  penalty = 1.0 x half-width in every cell (checked); A1 takes the venture action iff "
          f"q_venture - q_safe > U_venture - U_safe: {agree}/{total} cells agree ({agree / total:.4f})")
    q, n, scale, unit, widths = fits[0]
    W = widths[i30, hS, wP]
    print("  Table S1. Seed 0, 30 held-out tries, shortfall world with true advantage +0.50 at the gap (venturing there is right):")
    print(f"  {'cell (step, context)':24} {'rows s/v':>9} {'q_safe':>7} {'q_vent':>7} {'U_safe':>7} {'U_vent':>7} "
          f"{'q-U safe':>9} {'q-U vent':>9} {'A1 picks':>9} {'truth':>8}")
    mu = MU.copy(); mu[1, 1, VENTURE] = 1.5
    for t in range(2):
        for c in range(2):
            pick = "venture" if (q[t, c, VENTURE] - W[t, c, VENTURE]) > (q[t, c, SAFE] - W[t, c, SAFE]) else "safe"
            truth = "venture" if mu[t, c, VENTURE] > mu[t, c, SAFE] else "safe"
            tag = f"({t}, {c})" + (" the gap" if (t, c) == (1, 1) else "")
            print(f"  {tag:24} {n[t, c, SAFE]:>4d}/{n[t, c, VENTURE]:<4d} {q[t, c, SAFE]:7.3f} {q[t, c, VENTURE]:7.3f} "
                  f"{W[t, c, SAFE]:7.3f} {W[t, c, VENTURE]:7.3f} {q[t, c, SAFE] - W[t, c, SAFE]:9.3f} "
                  f"{q[t, c, VENTURE] - W[t, c, VENTURE]:9.3f} {pick:>9} {truth:>8}")
    print(f"  scale at the gap (0 training rows) = the pooled residual SD of step 1: {scale[1, 1, VENTURE]:.3f}; "
          f"at safe (1, 1): {scale[1, 1, SAFE]:.3f}; unit u: step 0 {unit[0]:.3f}, step 1 {unit[1]:.3f}")
    # S2: the width at the gap by world against the decisions.
    print("  Table S2. Width at the gap and the decisions by world (30 held-out tries, mean over 200 seeds):")
    print(f"  {'true advantage':>14} {'shape':>9} {'U at gap':>8} {'U at safe (1,1)':>15} {'U_v - U_s':>9} "
          f"{'claimed q_v - q_s':>17} {'A1 ventures':>11} {'A2 ventures':>11} {'oracle':>8}")
    claimed = np.array([f[0][1, 1, VENTURE] - f[0][1, 1, SAFE] for f in fits])
    s2 = []
    for h, harm in enumerate(HARMS):
        for w, adv in enumerate(ADVANTAGE):
            Ug = np.array([f[4][i30, h, w, 1, 1, VENTURE] for f in fits])
            Us = np.array([f[4][i30, h, w, 1, 1, SAFE] for f in fits])
            a1 = DEV[:, i30, h, w, M["A1 width penalty"]].mean(); a2 = DEV[:, i30, h, w, M["A2 width BC multiplier"]].mean()
            s2.append((adv, harm, Ug.mean(), Us.mean(), a1, a2))
            print(f"  {adv:>+14.2f} {harm:>9} {Ug.mean():8.3f} {Us.mean():15.3f} {(Ug - Us).mean():9.3f} "
                  f"{claimed.mean():17.3f} {a1:11.2f} {a2:11.2f} {'venture' if adv > 0 else 'safe':>8}")
    # S3: exact decomposition of A1's pooled regret.
    dA1 = DEV[..., M["A1 width penalty"]]
    gap_only = P_TEST[1] * np.where(dA1, np.maximum(0, -ADV), np.maximum(0, ADV))
    away = REG[..., M["A1 width penalty"]] - gap_only
    print(f"  S3  A1's pooled regret {pooled[M['A1 width penalty']]:.4f} = {away.mean():.4f} lost at the cells the data cover "
          f"(A1 refuses venture at (1, 0), outcomes 0 or 3, worth 0.5 x 0.35 = 0.175, and at (0, 1), outcomes -0.5 or 1.1, "
          f"worth 0.2 x 0.65 = 0.130) + {gap_only.mean():.4f} lost at the gap (the refuse-thin-data rule loses "
          f"{pooled[M['range limit']]:.4f} there, and A1 also almost never ventures there)")
    # S4: coverage where the width is measured (notebook check 6).
    tr, ca, _ = collect(0)
    q0, n0 = fit_critic(tr); sc0, un0 = fit_scale(tr, q0)
    r0_ = bca_exact(np.abs(targets(ca, 0, q0) - q0[0, ca["c"], ca["a"][:, 0]]) / sc0[0, ca["c"], ca["a"][:, 0]])
    r1_ = bca_exact(np.abs(targets(ca, 1, q0) - q0[1, ca["c"], ca["a"][:, 1]]) / sc0[1, ca["c"], ca["a"][:, 1]])
    fresh = collect(10_000)[1]
    covs = []
    for t, r_ in ((0, r0_), (1, r1_)):
        covs.append((np.abs(targets(fresh, t, q0) - q0[t, fresh["c"], fresh["a"][:, t]])
                     <= r_ * sc0[t, fresh["c"], fresh["a"][:, t]]).mean())
    print(f"  S4  coverage of A1's widths on fresh logged rows (seed 0 thresholds, notebook check 6): step 0 {covs[0]:.3f}, "
          f"step 1 {covs[1]:.3f} (target at least 0.90): the width is calibrated where it is measured")
    return s2

# ----------------------------------------------------------------------------------------------------------------
# Part B: ThrusterChain Stage 2 records (seeds 0-4, stalling world, fixed critic).
# ----------------------------------------------------------------------------------------------------------------
def stage2_paths(args, seed, ver):
    if args.copy:
        return (os.path.join(args.copy, "pairs", f"seed{seed}", "pairs.npz"),
                os.path.join(args.copy, "pairs", f"seed{seed}", "labels.npz"),
                os.path.join(args.copy, ver, "stall", f"seed{seed}", "scores", f"bca_width_{ver}_fixed.npz"))
    r = args.runs
    return (os.path.join(r, "runs", "shift_explore", "detect", f"seed{seed}", "stall", "pairs.npz"),
            os.path.join(r, "runs", "shift_explore", "detect", f"seed{seed}", "stall", "labels.npz"),
            os.path.join(r, "runs", "shift_explore", "stage2", "detect", ver, "stall", "fixed", f"seed{seed}", "scores",
                         f"bca_width_{ver}_fixed.npz"))

def part_b(args):
    print("\nPart B  ThrusterChain, Stage 2 records: does a = argmax [Q1 - 1.0 U] drop a harmful full-throttle command for its 80% cap?")
    if not (args.copy or args.runs):
        print("  skipped: give --copy <dir> or --runs <root> (see the header)")
        return None
    rows = []
    print(f"  {'band':9} {'seed':>4} {'set':>3} {'harmful':>7} {'harmless off-data':>17} {'Q1(cmd)-Q1(cap)':>15} {'U(cmd)-U(cap)':>13} "
          f"{'flipped: harmful':>16} {'flipped: harmless':>17} {'gain needed':>11}")
    for ver in ("uniform", "weighted"):
        for seed in range(5):
            pp, lp, sp = stage2_paths(args, seed, ver)
            pz, lz, U = np.load(pp), np.load(lp), np.load(sp)["scores"]
            st, role, pair = pz["set"], pz["role"], pz["pair"]
            for S in ("T1", "T3"):
                mc, mk = (st == S) & (role == "command"), (st == S) & (role == "cap")
                oc, ok = np.argsort(pair[mc]), np.argsort(pair[mk])
                assert np.array_equal(pair[mc][oc], pair[mk][ok])
                dU = U[mc][oc] - U[mk][ok]
                cp = lz["critic_pref"][mc][oc]
                assert np.array_equal(cp, lz["q1_command"][mc][oc] - lz["q1_cap"][mc][oc])   # critic_pref = Q1(cmd) - Q1(cap)
                h, uns = lz["harmful"][mc][oc], lz["unsupported"][mc][oc]
                hm = uns & ~h
                # the rule prefers the cap iff Q1(cmd) - U(cmd) < Q1(cap) - U(cap), that is iff cp < dU
                flip_h, flip_nh = float(np.mean(cp[h] < dU[h])), float(np.mean(cp[hm] < dU[hm]))
                need = cp[h] / np.where(dU[h] > 0, dU[h], np.nan)
                rows.append(dict(band=ver, seed=seed, set=S, n_h=int(h.sum()), n_hm=int(hm.sum()), cp=float(cp[h].mean()),
                                 dU=float(dU[h].mean()), flip_h=flip_h, flip_nh=flip_nh, need=float(np.nanmedian(need))))
                print(f"  {ver:9} {seed:4d} {S:>3} {h.sum():7d} {hm.sum():17d} {cp[h].mean():15.3f} {dU[h].mean():13.4f} "
                      f"{flip_h:16.3f} {flip_nh:17.3f} {np.nanmedian(need):11.1f}")
    print("  T1: 1,500 training states in the full-throttle region x (the actor's command, the same command capped at 80%);"
          "\n  T3: 3,000 strong-current rollout states. 'harmful': the critic prefers the command over its cap by at least 0.1"
          "\n  more than the truth and the cap is truly better by at least 0.05. 'harmless off-data': commands past 80% that are"
          "\n  not harmful. Q1(cmd)-Q1(cap) and U(cmd)-U(cap): means over the harmful pairs, critic value units. 'flipped': the"
          "\n  share of pairs where Q1 - U prefers the cap. 'gain needed': the median over harmful pairs of the gain k at which"
          "\n  Q1 - k U would prefer the cap. 'uniform' = the standard band; 'weighted' = Stage 2's classifier-weighted band.")
    return rows

# Recorded values of Part B (this script, 20 October 2026, from the files named in the header): (band, seed, set,
# flipped share on harmful pairs, flipped share on harmless off-data pairs). Used for the figure when Part B is skipped.
RECORDED_B = [
    ("uniform", 0, "T1", 0.000, 0.014), ("uniform", 0, "T3", 0.001, 0.045), ("uniform", 1, "T1", 0.000, 0.012),
    ("uniform", 1, "T3", 0.000, 0.067), ("uniform", 2, "T1", 0.000, 0.033), ("uniform", 2, "T3", 0.000, 0.027),
    ("uniform", 3, "T1", 0.000, 0.000), ("uniform", 3, "T3", 0.001, 0.057), ("uniform", 4, "T1", 0.000, 0.037),
    ("uniform", 4, "T3", 0.001, 0.042),
    ("weighted", 0, "T1", 0.956, 0.986), ("weighted", 0, "T3", 0.752, 0.970), ("weighted", 1, "T1", 0.928, 0.914),
    ("weighted", 1, "T3", 0.360, 0.824), ("weighted", 2, "T1", 0.937, 0.967), ("weighted", 2, "T3", 0.420, 0.824),
    ("weighted", 3, "T1", 0.949, 0.990), ("weighted", 3, "T3", 0.538, 0.780), ("weighted", 4, "T1", 0.806, 0.790),
    ("weighted", 4, "T3", 0.271, 0.445),
]

# ----------------------------------------------------------------------------------------------------------------
# Figure.
# ----------------------------------------------------------------------------------------------------------------
def figure(pooled, room, rows_b, DEV):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, GREEN, GRAY, INK = "#0072B2", "#E69F00", "#009E73", "#7F7F7F", "#222222"
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    # Panel 1: pooled regret by rule, with the kill-rule bar.
    ax = axes[0]
    names = ["oracle (knows each world)", "refuse thin data", "count penalty", "A2: width in the cloning multiplier",
             "A1: width penalty on values", "greedy (no safeguard)"]
    keys = ["oracle switch", "range limit", "count penalty", "A2 width BC multiplier", "A1 width penalty", "greedy"]
    vals = [pooled[M[k]] for k in keys]
    cols = [GREEN, GRAY, GRAY, ORANGE, BLUE, GRAY]
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, color=cols, height=0.62)
    for yi, v in zip(y, vals):
        ax.text(v + 0.008, yi, f"{v:.3f}", va="center", fontsize=9, color=INK)
    bar = pooled[M["range limit"]] - 0.25 * room
    ax.axvline(bar, color=INK, lw=1, ls="--")
    ax.text(bar + 0.012, y[0] + 0.05, f"kill rule: regret at most {bar:.3f}\n(a quarter of the room closed)", fontsize=8,
            color=INK, va="center")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("pooled regret (return lost against perfect choices; lower is better)")
    ax.set_ylabel("decision rule")
    ax.set_xlim(0, 0.7)
    ax.set_title("Toy world: regret pooled over 200 seeds, 10 worlds,\n7 evidence sizes and 2 harm shapes", fontsize=10)
    # Panel 2: venture share at the gap by world (30 survivors).
    ax = axes[1]
    i30 = EVIDENCE.index(30)
    x = np.arange(len(ADVANTAGE))
    for harm, ls in (("shortfall", "-"), ("tail", "--")):
        h = HARMS.index(harm)
        a1 = [DEV[:, i30, h, w, M["A1 width penalty"]].mean() for w in range(len(ADVANTAGE))]
        a2 = [DEV[:, i30, h, w, M["A2 width BC multiplier"]].mean() for w in range(len(ADVANTAGE))]
        ax.plot(x, a1, color=BLUE, ls=ls, marker="o", ms=5, lw=2, label=f"A1 width penalty, {harm} harm")
        ax.plot(x, a2, color=ORANGE, ls=ls, marker="s", ms=5, lw=2, label=f"A2 width in multiplier, {harm} harm")
    oracle = [1.0 if a > 0 else 0.0 for a in ADVANTAGE]
    ax.step(x, oracle, where="mid", color=GREEN, lw=2, label="oracle: venture iff truly better")
    ax.set_xticks(x); ax.set_xticklabels([f"{a:+.2f}" for a in ADVANTAGE], rotation=45, fontsize=8)
    ax.set_xlabel("true advantage of the untried action at the gap (the 10 worlds)")
    ax.set_ylabel("share of 200 seeds that take the untried action")
    ax.set_ylim(-0.03, 1.08)
    ax.set_title("Toy world, 30 held-out tries of the untried action:\nthe width rules do not follow the truth", fontsize=10)
    ax.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(0.01, 0.36))
    # Panel 3: real data, flip shares.
    ax = axes[2]
    if rows_b:
        data = [(r["band"], r["seed"], r["set"], r["flip_h"], r["flip_nh"]) for r in rows_b]
        src = "recomputed from the Stage 2 files"
    else:
        data = RECORDED_B; src = "recorded values (run with --copy or --runs to recompute)"
    labels = [f"seed {s} {S}" for b, s, S, _, _ in data if b == "uniform"]
    xg = np.arange(len(labels))
    for band, col, dx, nm in (("uniform", BLUE, -0.17, "standard band"), ("weighted", ORANGE, 0.17, "classifier-weighted band")):
        fh = [fh for b, s, S, fh, fn in data if b == band]; fn = [fn for b, s, S, fh, fn in data if b == band]
        ax.scatter(xg + dx, fh, color=col, marker="o", s=44, zorder=3, label=f"{nm}: harmful commands")
        ax.scatter(xg + dx, fn, facecolors="none", edgecolors=col, marker="D", s=42, zorder=3, lw=1.6,
                   label=f"{nm}: harmless off-data commands")
    ax.axhline(1.0, color=GREEN, lw=1.2, ls="--")
    ax.text(len(labels) - 0.6, 1.04, "as intended: every harmful command flipped, no harmless one", ha="right", fontsize=7.5, color=INK)
    ax.set_xticks(xg); ax.set_xticklabels(labels, rotation=45, fontsize=8)
    ax.set_xlabel("seed and pair set (T1: training states; T3: strong current)")
    ax.set_ylabel("share of pairs where Q1 - U prefers the cap")
    ax.set_ylim(-0.03, 1.62)
    ax.set_title(f"ThrusterChain, Stage 2 records (stalling world, gain 1):\n{src}", fontsize=10)
    ax.legend(fontsize=7, loc="upper left", ncol=1)
    for a in axes:
        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
        a.grid(axis="x" if a is axes[0] else "y", color="#DDDDDD", lw=0.6); a.set_axisbelow(True)
    fig.suptitle("M8  The width penalty on critic values: a = argmax [q - 1.0 U]", fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, format="svg")
    print(f"\nFigure written: {FIG}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--copy", default=None, help="folder with the copied Stage 2 files (layout in the header)")
    ap.add_argument("--runs", default=None, help="root that holds runs/shift_explore/... (the recorded layout)")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()
    REG, DEV, INFO, fits, pooled, room, gain = part_a()
    part_s(REG, DEV, fits, pooled)
    rows_b = part_b(args)
    if not args.no_figure:
        figure(pooled, room, rows_b, DEV)

if __name__ == "__main__":
    main()
