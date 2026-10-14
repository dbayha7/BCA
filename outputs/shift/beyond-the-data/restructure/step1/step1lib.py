"""Step 1 of the restructured BCA (the decisive test): shared definitions.

EXPLORATORY, designed after seeing seeds 0-4; written before any step-1 critic is trained. See PRESPEC.md and
prespec.json beside this file. Everything here is either a pure NumPy/SciPy function (tested in test_step1.py on
synthetic data) or a small path / hash helper. The heavy loaders (the gate's acceptance module, JAX) are imported by
train_critics.py and evaluate.py only.

Nothing here writes outside the step-1 output folder (runs/shift_explore/restructure/step1) or, in scratch mode, a
folder outside the repository. Bytecode writing is switched off before any import so that no __pycache__ is written
into a pinned folder.
"""
import sys

sys.dont_write_bytecode = True

import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path("/home/user/bca")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calibration import wbcp  # noqa: E402  (BCA's own Bayesian weighted conformal posterior, unchanged)

HERE = Path(__file__).resolve().parent
PRESPEC_PATH = HERE / "prespec.json"
PRESPEC_SHA_PATH = HERE / "prespec.json.sha256"
MANIFEST_PATH = HERE / "code_manifest.json"
CODE_FILES = ("step1lib.py", "train_critics.py", "evaluate.py", "run_step1.sh", "test_step1.py", "freeze.py",
              "PRESPEC.md", "prespec.json")
OFFICIAL_OUT = ROOT / "runs" / "shift_explore" / "restructure" / "step1"
OFFICIAL_CRITICS = OFFICIAL_OUT / "critics"
# Pinned folders: never written (the official output folder above is the only repository path this step writes).
PINNED = tuple(ROOT / p for p in (
    "experiments", "runs/shift", "runs/shift_explore/stage2", "runs/shift_explore/room",
    "runs/shift_explore/standins", "runs/shift_explore/detect", "runs/shift_explore/deploy",
    "runs/shift_explore/fallback", "runs/shift_explore/trace", "runs/shift_explore/anatomy",
    "runs/shift_explore/empirical", "runs/shift_explore/principled", "outputs/shift/beyond-the-data/stage2",
    "outputs/shift/beyond-the-data/stage2_v7", "outputs/shift/beyond-the-data/stage2_gpu",
    "outputs/shift/beyond-the-data/stage2_v7_design", "calibration", "algorithms", "runtime"))
WORLDS = ("stall", "no_stall")
WORLD_TAG = {"stall": 1, "no_stall": 2}      # rng tags never end in 0 (numpy drops a trailing 0 tag)
LABEL = ("EXPLORATORY: restructured BCA, step 1 (the decisive test). Designed after seeing seeds 0-4; the "
         "specification was written before any step-1 critic was trained. Outside the gate; changes no "
         "pre-registered rule and no v6/v7 file.")


# ------------------------------------------------------------------------------------------------ paths and hashes

def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def file_sha256(path):
    return sha256_bytes(Path(path).read_bytes())


def array_sha256(a):
    a = np.ascontiguousarray(np.asarray(a))
    return sha256_bytes(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())


def inside(path, root):
    p, r = Path(path).resolve(), Path(root).resolve()
    return p == r or r in p.parents


def guard_out(path, official):
    """Refuse any output path except: official mode, inside runs/shift_explore/restructure/step1; scratch mode,
    outside the repository altogether. A pinned folder is refused in both modes."""
    p = Path(path).resolve()
    for pin in PINNED:
        if inside(p, pin):
            raise SystemExit(f"refused: {p} is inside the pinned folder {pin}")
    if official:
        if not inside(p, OFFICIAL_OUT):
            raise SystemExit(f"refused: official outputs go under {OFFICIAL_OUT}, not {p}")
    elif inside(p, ROOT):
        raise SystemExit(f"refused: scratch outputs must lie outside {ROOT} (got {p})")
    return p


def write_json_once(path, payload, official):
    """Strict JSON with a .sha256 sidecar; refuses to overwrite (write-once)."""
    p = guard_out(path, official)
    if p.exists():
        raise SystemExit(f"refused: {p} exists (write-once)")
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(jsonable(payload), indent=1, sort_keys=True, allow_nan=False)
    p.write_text(text, encoding="utf8")
    digest = sha256_bytes(text.encode("utf8"))
    p.with_name(p.name + ".sha256").write_text(f"{digest}  {p.name}\n", encoding="utf8")
    return digest


def read_json_checked(path):
    p = Path(path)
    text = p.read_text(encoding="utf8")
    side = p.with_name(p.name + ".sha256")
    if side.exists() and side.read_text().split()[0] != sha256_bytes(text.encode("utf8")):
        raise SystemExit(f"refused: {p} differs from its .sha256 sidecar")
    return json.loads(text)


def jsonable(x):
    """Plain JSON types; non-finite floats become the strings 'inf', '-inf', 'nan' (strict JSON has none)."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return [jsonable(v) for v in x.tolist()]
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        if math.isfinite(v):
            return v
        return "nan" if math.isnan(v) else ("inf" if v > 0 else "-inf")
    if isinstance(x, Path):
        return str(x)
    return x


def load_prespec():
    """prespec.json, verified against prespec.json.sha256 (written by freeze.py)."""
    text = PRESPEC_PATH.read_bytes()
    if not PRESPEC_SHA_PATH.exists():
        raise SystemExit("refused: prespec.json.sha256 is missing; run freeze.py first")
    want = PRESPEC_SHA_PATH.read_text().split()[0]
    if sha256_bytes(text) != want:
        raise SystemExit("refused: prespec.json changed after it was frozen")
    return json.loads(text), want


def verify_manifest():
    """code_manifest.json (written by freeze.py): every listed file must still hash to its frozen value."""
    if not MANIFEST_PATH.exists():
        raise SystemExit("refused: code_manifest.json is missing; run freeze.py first")
    man = json.loads(MANIFEST_PATH.read_text())
    bad = [n for n, h in man["files"].items() if file_sha256(HERE / n) != h]
    if bad:
        raise SystemExit(f"refused: files changed after freeze.py: {bad}")
    return man, file_sha256(MANIFEST_PATH)


def cpus():
    return sorted(os.sched_getaffinity(0))


def require_one_cpu(what):
    c = cpus()
    if len(c) != 1:
        raise SystemExit(f"refused: {what} must run pinned to one CPU (taskset -c <cpu> ...); this process may use {c}")
    return c


def rng(tags):
    tags = [int(t) for t in tags]
    if tags and tags[-1] == 0:
        raise ValueError("an rng tag list must not end in 0 (numpy would drop it)")
    return np.random.default_rng(tags)


# ------------------------------------------------------------------------------------------------ the gap rule

def throttle(actions):
    """Throttle index t(a) = max_k |a_k| over the four thrusters, float64 of the stored float32 command."""
    return np.abs(np.asarray(actions, np.float32)).max(axis=1).astype(np.float64)


def gap_rule(t, upper_quantile=0.99, min_gap=0.05, min_tail=9):
    """The outcome-free gap rule (calibration design e4_tailband.strata, restated). Reads commands only.

    Sort the throttle index of the rows strictly above its 99th percentile (numpy's default 'linear' quantile); the
    largest difference between consecutive sorted values (the first one if several are equal) is the gap, from t_lo
    (its lower end) to t_up (its upper end). The rule is defined when the gap is at least min_gap wide and at least
    min_tail rows have t >= t_up. Full-throttle group: t > t_lo; rest: t <= t_lo. On the rows the rule was applied to
    this is the same split as t >= t_up (no row lies strictly inside the gap; ties at t_lo go to the rest, ties at t_up
    to the full-throttle group).

    Returns dict(defined, t_lo, t_up, tail (bool per row), n_tail, gap, n_above_quantile)."""
    t = np.asarray(t, np.float64)
    hi = np.sort(t[t > np.quantile(t, upper_quantile)])
    if len(hi) < 2:
        return dict(defined=False, t_lo=float(t.max()), t_up=None, tail=np.zeros(len(t), bool), n_tail=0, gap=0.0,
                    n_above_quantile=int(len(hi)), reason="fewer than two rows above the 99th percentile")
    g = np.diff(hi)
    i = int(np.argmax(g))
    t_lo, t_up = float(hi[i]), float(hi[i + 1])
    tail = t > t_lo
    if not np.array_equal(tail, t >= t_up):
        raise AssertionError("a row lies strictly inside the gap: the rule is inconsistent")
    defined = bool((t_up - t_lo) >= min_gap and int(tail.sum()) >= min_tail)
    out = dict(defined=defined, t_lo=t_lo, t_up=t_up, tail=tail if defined else np.zeros(len(t), bool),
               n_tail=int(tail.sum()) if defined else 0, gap=float(t_up - t_lo), n_above_quantile=int(len(hi)))
    if not defined:
        out["reason"] = f"gap {t_up - t_lo:.4f} < {min_gap} or tail rows {int(tail.sum())} < {min_tail}"
    return out


def full_throttle(t, t_lo):
    """Group of any row (held out or not) once t_lo is fixed: True = full-throttle group (t > t_lo)."""
    return np.asarray(t, np.float64) > float(t_lo)


# ------------------------------------------------------------------------------------------------ scores

def overestimate(q_sa, reward, done, q_next, gamma):
    """One-sided overestimate (value units): s = Q(s, a) - [r + gamma (1 - d) Q(s', c(s'))], with Q = min of the two
    heads at both ends. Positive when the critic predicted more than the one-step outcome delivered. s = -delta for the
    TD residual delta = r + gamma (1 - d) Q(s', c(s')) - Q(s, a)."""
    q_sa, reward, done, q_next = (np.asarray(v, np.float64) for v in (q_sa, reward, done, q_next))
    return q_sa - (reward + float(gamma) * (1.0 - done) * q_next)


# ------------------------------------------------------------------------------------------------ the certificate

def pairwise_quantile(x, y, q):
    """Exact q-quantile (numpy 'inverted_cdf': an order statistic) of the n_x * n_y differences x_i - y_j: the
    distribution of the difference of independent draws from two posteriors given their draws. inf - finite = +inf,
    finite - inf = -inf, inf - inf (undefined) counts as -inf (conservative: no evidence of a gap)."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    with np.errstate(invalid="ignore"):
        d = (x[:, None] - y[None, :]).ravel()
    d[np.isnan(d)] = -np.inf
    return float(np.quantile(d, q, method="inverted_cdf"))


def posterior(scores, tags, cfg):
    """BCA's own WBCP posterior (calibration/wbcp.calibrate), uniform weights, at the frozen alpha, credibility and
    number of draws; rng = default_rng(tags)."""
    return wbcp.calibrate(np.asarray(scores, np.float64), rng(tags), alpha=float(cfg["alpha"]),
                          beta=float(cfg["credibility"]), draws=int(cfg["draws"]))


def certificate(tail, rest, tags, cfg):
    """The world-level certificate (PRESPEC.md, CERTIFICATE).

    tail, rest: one-sided overestimates of the full-throttle group and the rest (both held out from the critic).
    Posteriors: wbcp.calibrate(tail, default_rng(tags + [1])) and wbcp.calibrate(rest, default_rng(tags + [2])).
    Statistic: the 5% quantile of the posterior gap lambda_tail - lambda_rest (every pair of draws; pairwise_quantile).
    On iff statistic > margin. Also returns, for information: the posterior median gap, P(gap <= margin) over pairs,
    the plain gap (empirical 90% quantiles), the design's e11 form (tail posterior minus the rest's plain 90%
    quantile), and the height R_tail - R_rest (BCA's deployed thresholds, max(lambda_hat, lambda_hpd))."""
    tail, rest = np.asarray(tail, np.float64), np.asarray(rest, np.float64)
    margin = float(cfg["margin"])
    if len(tail) == 0 or len(rest) == 0:
        return dict(on=False, reason="an empty group", n_tail=int(len(tail)), n_rest=int(len(rest)), q05=None)
    ct = posterior(tail, list(tags) + [1], cfg)
    cr = posterior(rest, list(tags) + [2], cfg)
    q = float(cfg["statistic_quantile"])
    q05 = pairwise_quantile(ct.crossings, cr.crossings, q)
    q50 = pairwise_quantile(ct.crossings, cr.crossings, 0.5)
    with np.errstate(invalid="ignore"):
        d = (ct.crossings[:, None] - cr.crossings[None, :]).ravel()
    d[np.isnan(d)] = -np.inf
    rest_plain = float(np.quantile(rest, 1.0 - float(cfg["alpha"]), method="inverted_cdf"))
    e11 = ct.crossings - rest_plain
    return dict(on=bool(q05 > margin), q05=q05, q50=q50, margin=margin, P_gap_le_margin=float(np.mean(d <= margin)),
                n_tail=int(len(tail)), n_rest=int(len(rest)),
                lambda_hat_tail=float(ct.lambda_hat), lambda_hat_rest=float(cr.lambda_hat),
                plain_gap=float(ct.lambda_hat - cr.lambda_hat),
                R_tail=float(ct.threshold), R_rest=float(cr.threshold), height=float(ct.threshold - cr.threshold),
                tail_certified=bool(ct.certified), finite_share_tail=float(np.mean(np.isfinite(ct.crossings))),
                finite_share_rest=float(np.mean(np.isfinite(cr.crossings))),
                e11_form_q05=float(np.quantile(e11, q, method="inverted_cdf")), e11_form_on=bool(
                    float(np.quantile(e11, q, method="inverted_cdf")) > margin),
                mean_tail=float(tail.mean()), mean_rest=float(rest.mean()))


def rank_test(tail, rest):
    """Plain one-sided Mann-Whitney U test: are the full-throttle group's overestimates stochastically larger than
    the rest's? scipy.stats.mannwhitneyu(tail, rest, alternative='greater') (default method)."""
    from scipy import stats
    tail, rest = np.asarray(tail, np.float64), np.asarray(rest, np.float64)
    if len(tail) == 0 or len(rest) == 0:
        return dict(U=None, p=None)
    r = stats.mannwhitneyu(tail, rest, alternative="greater")
    return dict(U=float(r.statistic), p=float(r.pvalue), auc=float(r.statistic / (len(tail) * len(rest))))


def shuffled_certificates(tail, rest, cert_tags, perm_tags, n_perm, cfg):
    """Outcome-shuffle control for one seed-world: the pooled scores (tail then rest) are permuted with
    default_rng(perm_tags + [j + 1]) for j = 0..n_perm-1; the first n_tail permuted scores form the full-throttle
    group and the others the rest; the certificate is recomputed with the SAME posterior tags as the real one (common
    random numbers). Returns lists of on/off and of the statistic."""
    tail, rest = np.asarray(tail, np.float64), np.asarray(rest, np.float64)
    pooled = np.concatenate([tail, rest])
    k = len(tail)
    on, stat, height = [], [], []
    for j in range(int(n_perm)):
        p = pooled[rng(list(perm_tags) + [j + 1]).permutation(len(pooled))]
        c = certificate(p[:k], p[k:], cert_tags, cfg)
        on.append(bool(c["on"]))
        stat.append(c["q05"])
        height.append(c.get("height"))
    return dict(on=on, q05=stat, height=height)


# ------------------------------------------------------------------------------------------------ the net

def deployment_net(points, certs, seeds):
    """NET ACROSS WORLDS of the gated deployment check, strong current, pooled over seeds.

    points[world][seed] = dict(never=points without any check, on=points with the check switched on) (strong
    current); certs[world][seed] = bool. With the certificate on the check runs; off, it never fires (points = never).
      gain_stall    = mean over seeds of (check - never) in the stalling world
      loss_no_stall = mean over seeds of (never - check) in the no-stall world
      net           = gain_stall - loss_no_stall"""
    def check(w, s):
        return points[w][s]["on"] if certs[w][s] else points[w][s]["never"]
    gain = float(np.mean([check("stall", s) - points["stall"][s]["never"] for s in seeds]))
    loss = float(np.mean([points["no_stall"][s]["never"] - check("no_stall", s) for s in seeds]))
    return dict(net=gain - loss, gain_stall=gain, loss_no_stall=loss)


def shuffle_summary(points, shuffled_on, seeds, net_real):
    """shuffled_on[world][seed] = list of on/off over permutations j (same j across seed-worlds). The shuffled net of
    permutation j uses certs[w][s] = shuffled_on[w][s][j]. The go/no-go statistic is the MEAN shuffled net."""
    n = len(shuffled_on["stall"][seeds[0]])
    nets = []
    for j in range(n):
        certs = {w: {s: bool(shuffled_on[w][s][j]) for s in seeds} for w in WORLDS}
        nets.append(deployment_net(points, certs, seeds)["net"])
    nets = np.asarray(nets)
    return dict(n_perm=int(n), mean=float(nets.mean()), q95=float(np.quantile(nets, 0.95)),
                max=float(nets.max()), share_at_least_real=float(np.mean(nets >= net_real)),
                on_rate={w: {str(s): float(np.mean(shuffled_on[w][s])) for s in seeds} for w in WORLDS})


def go_no_go(certs, net, shuffled_mean, seeds, rule):
    """The four approved conditions (PRESPEC.md, GO/NO-GO). Conditions 1-3 decide; 4 is reported beside them."""
    c1_on = all(certs["stall"][s] for s in seeds)
    c1_off = not any(certs["no_stall"][s] for s in seeds)
    c2 = net >= float(rule["net_min"])
    c3 = shuffled_mean <= float(rule["shuffle_max_fraction"]) * net
    go = bool(c1_on and c1_off and c2 and c3)
    # Information only (review): the other reading of "half of that" (half of the 90-point bar). It is stricter than
    # C3 whenever C2 holds; it is reported so that the verdict can be read under both readings. It does not decide.
    c3_alt = shuffled_mean <= float(rule["shuffle_max_fraction"]) * float(rule["net_min"])
    return dict(C1_certificate_on_5_of_5_stall=bool(c1_on), C1_certificate_off_5_of_5_no_stall=bool(c1_off),
                C1=bool(c1_on and c1_off), C2_net_at_least_90=bool(c2), C3_shuffled_at_most_half=bool(c3),
                C4_rank_test_reported=True, verdict="GO" if go else "NO-GO",
                info_C3_also_at_most_half_of_90=bool(c3_alt))


def record_problems(rec, spec, spec_sha, man_sha, world, seed):
    """Why an OFFICIAL step-1 critic record cannot be used (empty list: usable). Checks, beyond the record's own
    SHA-256 sidecar: official mode, the frozen prespec.json and code manifest it was trained under, the seed and world,
    the full 100,000 updates, the gap-minus-survivors log, every training-data check, one CPU, and that the run's
    learner arguments and initial critic parameters equal the gate critic's (the 'exact gate protocol')."""
    problems = []
    if rec.get("mode") != "official":
        problems.append("not an official run")
    if rec.get("prespec_sha256") != spec_sha:
        problems.append("trained under another prespec.json")
    if rec.get("code_manifest_sha256") != man_sha:
        problems.append("trained under another code manifest")
    if rec.get("world") != world or rec.get("seed") != seed:
        problems.append(f"record is for {rec.get('world')}/seed {rec.get('seed')}")
    if (rec.get("result") or {}).get("updates") != spec["training"]["updates"]:
        problems.append("not 100,000 updates")
    if rec.get("variant_of_run") != "gap minus survivors":
        problems.append("not the gap-minus-survivors log")
    checks = rec.get("checks") or {}
    if not checks or not all(bool(v) for v in checks.values()):
        problems.append(f"training-data checks not all true: {checks}")
    if len(rec.get("cpus") or []) != 1:
        problems.append(f"not run on one CPU: {rec.get('cpus')}")
    cmp = rec.get("compare_with_gate") or {}
    if cmp.get("args_equal_gate") is not True:
        problems.append("learner arguments differ from the gate critic's")
    if cmp.get("initial_params_equal_gate") is not True:
        problems.append("initial critic parameters differ from the gate critic's")
    return problems
