"""Near-80 (version 9): the held-out top-up of BCA's full-throttle group (PREREGISTRATION.md section 3.2; DECISION.md
section 2.2, item 2) and the check that its random streams are unused.

The top-up. When a seed-world's outcome-free count (near80lib.outcome_free_count: the size of BCA's pooled
full-throttle group, from commands, the ball and the gap rule only) is below 36, held-out blocks of 100 new recorded
episodes are added, block k = 1, 2, ... until that world's count reaches 36 or k = 10. Block k is simulated exactly as
experiments/shift/dataset.py's generate does (lines 190-222: the same recorder, physics, episode length and float32
commands, each start reset into a registered environment and stepped in lockstep; dataset.py is not changed), from
  starts and wobble noise   np.random.default_rng([20261012, seed, 61, k]): draw_starts(100, ...) then N(0, I) noise
  coin                      np.random.default_rng([20261012, seed, 62, k]).uniform((100, length))
(the same streams in both worlds). Its rows are held-out rows: no critic or actor trains on them, and the groups
read only those whose command lies outside the ball. The data phase writes each block's array SHA-256 into the data
record before any critic exists; the switch phase regenerates the blocks and checks them.

The tag check (tag_check): no random stream under master seed 20261012 on any branch has third element 61 or 62
(tag 12 is shift.make_pools'; 61 and 62 were unused under 20261012 on 20 October 2026). freeze.py runs it on every
branch and pins its record in pre_approval/.

No learner is trained here. Standard library and NumPy (and, for a block, the gate's environment).
"""
import sys

sys.dont_write_bytecode = True

import ast  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import near80_constants as K  # noqa: E402

MASTER = int(K.TOPUP["master"])
TAG_STARTS, TAG_COIN = int(K.TOPUP["tag_starts"]), int(K.TOPUP["tag_coin"])
BLOCK_EPISODES = int(K.TOPUP["block_episodes"])
CAP = int(K.TOPUP["cap"])
TARGET = int(K.TOPUP["target"])
ARRAY_KEYS = ("observations", "actions", "rewards", "next_observations", "terminals", "timeouts", "infos/state",
              "infos/next_state", "infos/t")
THIS_FILE = "outputs/shift/beyond-the-data/near80_v9/code/topup.py"


class TopupError(Exception):
    """A top-up block could not be made or does not match its record."""


def stream_tags(seed, k):
    """([20261012, seed, 61, k], [20261012, seed, 62, k]): block k's two streams (k = 1-10)."""
    if not 1 <= int(k) <= CAP:
        raise TopupError(f"top-up block {k} is outside 1-{CAP}")
    return [MASTER, int(seed), TAG_STARTS, int(k)], [MASTER, int(seed), TAG_COIN, int(k)]


def draw_block_streams(seed, k, length):
    """(starts, noise, coin) of block k, drawn as dataset.draw_streams draws a log's (module docstring)."""
    from experiments.shift import thruster_chain as TC
    t_starts, t_coin = stream_tags(seed, k)
    rng = np.random.default_rng(t_starts)
    starts = TC.draw_starts(BLOCK_EPISODES, rng, TC.START_LAWS["recorded"])
    noise = rng.standard_normal((BLOCK_EPISODES, length, TC.DA))
    coin = np.random.default_rng(t_coin).uniform(size=(BLOCK_EPISODES, length))
    return starts, noise, coin


def simulate_episodes(world, spec, stall_slope, starts, noise, coin):
    """The recorded episodes of these streams, stepped through env.step exactly as dataset.generate does (its loop,
    restated): every start reset into the world's registered 'recorded' environment, the recording controller's
    command rounded to float32 and passed to env.step, the environments stepped in lockstep. Returns the arrays of
    ARRAY_KEYS (flattened episode-major, as generate's) and the full-throttle flags (a diagnostic, never read)."""
    import gymnasium as gym
    from experiments.shift import recorder as RC
    from experiments.shift import thruster_chain as TC
    physics = TC.world_physics(world, stall_slope)
    E, L = int(len(starts)), physics.length
    recorder = RC.Recorder(spec, physics)
    eid = TC.env_id(world, "recorded")
    envs = [gym.make(eid, stall_slope=physics.stall_slope) for _ in range(E)]
    X = np.empty((E, L + 1, TC.DX))
    obs = np.empty((E, L + 1, TC.DX), np.float32)
    act = np.empty((E, L, TC.DA), np.float32)
    rew = np.empty((E, L), np.float32)
    press = np.empty((E, L), bool)
    term = np.empty((E, L), bool)
    trunc = np.empty((E, L), bool)
    for e, env in enumerate(envs):
        obs[e, 0], info = env.reset(options={"state": starts[e]})
        X[e, 0] = info["state"]
    for t in range(L):
        cmd, _, pr = recorder.parts(X[:, t], noise[:, t], coin[:, t])
        act[:, t] = cmd.astype(np.float32)
        press[:, t] = pr
        for e, env in enumerate(envs):
            o, r, te, tr, info = env.step(act[e, t])
            obs[e, t + 1], X[e, t + 1], rew[e, t], term[e, t], trunc[e, t] = o, info["state"], r, te, tr
    for env in envs:
        env.close()
    if term.any() or not trunc[:, -1].all() or trunc[:, :-1].any():
        raise TopupError("episodes must end by truncation at t = 100 only")
    flat = lambda v: v.reshape(E * L, *v.shape[2:])
    arrays = {"observations": flat(obs[:, :-1]), "actions": flat(act), "rewards": flat(rew),
              "next_observations": flat(obs[:, 1:]), "terminals": flat(term), "timeouts": flat(trunc),
              "infos/state": flat(X[:, :-1]), "infos/next_state": flat(X[:, 1:]),
              "infos/t": np.tile(np.arange(L, dtype=np.int32), E)}
    return arrays, flat(press)


def make_block(seed, world, spec, stall_slope, k, length=None):
    """Block k of a seed-world: {k, tags, arrays (ARRAY_KEYS), episodes, rows, sha256 {key: array SHA-256}}."""
    from experiments.shift import thruster_chain as TC
    L = TC.world_physics(world, stall_slope).length if length is None else int(length)
    starts, noise, coin = draw_block_streams(seed, k, L)
    arrays, _ = simulate_episodes(world, spec, stall_slope, starts, noise, coin)
    return block_record(seed, k, arrays, BLOCK_EPISODES)


def block_record(seed, k, arrays, episodes):
    from experiments.shift import dataset as DS
    t_starts, t_coin = stream_tags(seed, k)
    for key in ARRAY_KEYS:
        if key not in arrays:
            raise TopupError(f"top-up block {k} has no {key}")
    return dict(k=int(k), tags=dict(starts_and_noise=t_starts, coin=t_coin), arrays=arrays, episodes=int(episodes),
                rows=int(len(arrays["actions"])), sha256={key: DS.sha256_array(arrays[key]) for key in ARRAY_KEYS})


def blocks_sha256(blocks):
    """One SHA-256 of every block's array SHA-256 (in block order): the data record's top-up fingerprint."""
    body = json.dumps([[b["k"], b["sha256"]] for b in blocks], sort_keys=True).encode("utf8")
    return hashlib.sha256(body).hexdigest()


def summary(blocks):
    """The data record's view of the blocks (no arrays)."""
    return [dict(k=b["k"], tags=b["tags"], episodes=b["episodes"], rows=b["rows"], sha256=b["sha256"],
                 added_to_count=b.get("added_to_count")) for b in blocks]


def top_up(count_of, make, n_tail, target=TARGET, cap=CAP):
    """The top-up rule of one world. count_of(blocks) -> the full-throttle count with these blocks added;
    make(k) -> block k. Adds blocks 1, 2, ... while the count is below target and fewer than cap blocks were made.
    Returns (blocks, count after, cap_reached)."""
    blocks, count = [], int(n_tail)
    while count < int(target) and len(blocks) < int(cap):
        b = make(len(blocks) + 1)
        before = count
        blocks.append(b)
        count = int(count_of(blocks))
        b["added_to_count"] = count - before
    return blocks, count, bool(count < int(target))


def check_blocks(regenerated, recorded):
    """Problems (empty: none) when regenerated blocks differ from the data record's summary."""
    out = []
    if len(regenerated) != len(recorded or []):
        return [f"{len(regenerated)} top-up blocks regenerated, the data record has {len(recorded or [])}"]
    for b, r in zip(regenerated, recorded):
        if b["k"] != r.get("k") or b["sha256"] != r.get("sha256") or b["tags"] != r.get("tags"):
            out.append(f"top-up block {b['k']} differs from the data record's")
    return out


# --------------------------------------------------------------------------------------------- the tag check

RNG_CALL = re.compile(r"default_rng\s*\(")


def _const_names(tree):
    """Module-level NAME = <int> assignments of a module (the tags' names)."""
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, int):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out[t.id] = node.value.value
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Tuple):
            for t in node.targets:
                if isinstance(t, ast.Tuple) and len(t.elts) == len(node.value.elts):
                    for n, v in zip(t.elts, node.value.elts):
                        if isinstance(n, ast.Name) and isinstance(v, ast.Constant) and isinstance(v.value, int):
                            out[n.id] = v.value
    return out


def _value(node, names):
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.Name):
        return names.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        return node.attr if node.attr != "DATA_SEED" else MASTER
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "int" and node.args:
        return _value(node.args[0], names)
    return ast.dump(node)[:40]


def streams_in_source(text, name=""):
    """Every default_rng([master, x, third, ...]) of a Python source whose first element is 20261012 (a literal,
    DATA_SEED or a module constant equal to it): [(line, third element as written or resolved)]. A file that does not
    parse is listed with line 0 and its error."""
    try:
        tree = ast.parse(text)
    except SyntaxError as error:
        return [(0, f"unparsed: {error}")]
    names = _const_names(tree)
    out = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "attr", getattr(node.func, "id", None)) ==
                "default_rng" and node.args):
            continue
        arg = node.args[0]
        if not isinstance(arg, (ast.List, ast.Tuple)) or len(arg.elts) < 3:
            continue
        first = _value(arg.elts[0], names)
        if first == MASTER or first in ("DATA_SEED", "MASTER") and names.get(str(first), MASTER) == MASTER:
            out.append((node.lineno, _value(arg.elts[2], names)))
    return out


def tag_hits(text, name=""):
    """The streams of a source whose third element is 61 or 62 (top-up's own module excluded by the caller)."""
    return [(line, v) for line, v in streams_in_source(text, name) if v in (TAG_STARTS, TAG_COIN)]


def _git(repo, *args):
    done = subprocess.run(["git", "-C", str(repo), *args], stdin=subprocess.DEVNULL, capture_output=True, timeout=300)
    if done.returncode not in (0, 1):
        raise TopupError(f"git {' '.join(args[:2])} failed: {done.stderr.decode('utf8', 'replace')[-200:]}")
    return done.stdout.decode("utf8", "replace")


def tag_check(repo, refs=None):
    """The tag check (module docstring) on every branch of a git repository (refs: default every local and remote
    branch): for each ref, every Python file that calls default_rng is parsed and its streams under master 20261012
    listed; a stream whose third element is 61 or 62 is a hit (this module excepted). Returns the record."""
    repo = Path(repo)
    if refs is None:
        refs = [r for r in _git(repo, "for-each-ref", "--format=%(refname)", "refs/heads", "refs/remotes").split()
                if not r.endswith("/HEAD")]
    hits, scanned, unresolved = [], {}, []
    for ref in refs:
        files = sorted({line.split(":", 2)[1] for line in _git(repo, "grep", "-l", "default_rng", ref, "--",
                                                                "*.py").splitlines() if ":" in line})
        sha = _git(repo, "rev-parse", ref).strip()
        scanned[ref] = dict(commit=sha, files=len(files))
        for f in files:
            if f == THIS_FILE:
                continue
            text = _git(repo, "show", f"{sha}:{f}")
            for line, v in streams_in_source(text, f):
                if v in (TAG_STARTS, TAG_COIN):
                    hits.append(dict(ref=ref, file=f, line=line, third=v))
                elif not isinstance(v, int):
                    unresolved.append(dict(ref=ref, file=f, line=line, third=str(v)))
    return dict(kind="near80v9-tag-check", master=MASTER, tags=[TAG_STARTS, TAG_COIN], refs=scanned, hits=hits,
                unresolved=unresolved, passed=not hits,
                rule="no stream under master seed 20261012 on any branch has third element 61 or 62 (this module "
                     "excepted); a third element that is not a resolvable integer is listed, never a hit")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="the top-up's tag check (no stream uses tags 61 or 62)")
    ap.add_argument("what", choices=("tag-check",))
    ap.add_argument("--repo", default=str(K.ROOT))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    rec = tag_check(a.repo)
    text = json.dumps(rec, indent=1, sort_keys=True)
    if a.out:
        p = Path(a.out)
        if p.exists():
            print(f"refused: {p} exists", file=sys.stderr)
            return K.EXIT_REFUSED
        p.write_text(text + "\n", encoding="utf8")
    print(f"tag check: {'passed' if rec['passed'] else 'FAILED'} ({len(rec['refs'])} refs, {len(rec['hits'])} hits, "
          f"{len(rec['unresolved'])} unresolved third elements)")
    return 0 if rec["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
