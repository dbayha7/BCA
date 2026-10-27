#!/usr/bin/env python3
"""S7 probe: a read-only, outcome-blind look at stop rule S7's closed loop (four-learner study, experiments/shift_hosts).

RUN IT ON THE LAPTOP, TWICE, BEFORE ANY VERSION-2 BYTE REACHES /root/bca-hosts (WSL, as root):
  git -C /mnt/c/Users/David/dev/BCA fetch origin claude/shift-hosts-v2      # the Windows clone; /root/bca-hosts stays
  mkdir -p /root/s7probe/out && P=outputs/shift_hosts_v2/probe_s7.py                # a copy outside the repository
  git -C /mnt/c/Users/David/dev/BCA show origin/claude/shift-hosts-v2:$P > /root/s7probe/probe_s7.py
  cd /root/bca-hosts && env -u LD_LIBRARY_PATH -u XLA_FLAGS PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
      JAX_PLATFORMS=cpu SHIFT_HOSTS_DEVICE=cpu taskset -c 2 /opt/bca-shift-venv/bin/python -B \
      /root/s7probe/probe_s7.py --list-s7 --compare --out /root/s7probe/out/rebrac_alone.txt
  (the same command with --step 10000 --compare --out /root/s7probe/out/rebrac_alone_10000.txt in place of the last
  line's options: the trend over training, shown and not judged)
  sha256sum /root/s7probe/probe_s7.py /root/s7probe/out/*.txt                 # all go into PRESPEC.md 19.6
It refuses to run unless bytecode writing is off and PYTHONDONTWRITEBYTECODE=1 (the command gives both -B and the
variable), from inside the repository, or with --out inside the repository or on an existing file. It stops unless
an approval verifies and that approval is version 1's (fingerprint 8c5e8421922dcceb...), so version 1's bytes are on
disk and the arithmetic is the lane's. It has no option to skip that check.

Written in the cloud on 17-18 Oct 2026 for PRESPEC.md (version 2 draft) section 19.6. The cloud ran it end to end on
made-up actor files only (calibration/smoke_probe.py and calibration/calibrate.py beside this file: numpy 2.2.6,
JAX 0.6.2 and flax 0.10.7 in a scratch environment, the four situation banks real, the three weighting banks made
up); never on a trained actor.

Question it answers: is ReBRAC alone's big-kicks difference (largest |points difference| 0.01139 against version 1's
0.01 limit) consistent with float32 rounding amplified by the closed loop, or is there a defect in the float64 adapter
(policies.py) or the float32 restatement (actors_f32.py)?

What it does, per bank (and per readout; CQL has two), on the actor the S7 lane checked:
  1. Rolls again exactly what evaluate.closed_loop_check rolls (policies.adapter and actors_f32.Float32Policy from all
     1,000 starts, truth.rollout, the stalling world; evaluate.py:597-605) and checks that the lane's largest and mean
     differences reproduce s7.json bit for bit.
  2. Prints the distribution of |J32 - J64| in points over the starts (quantiles, counts above 0.001, 0.005, 0.01,
     0.02 and 0.05, the top 10 starts by index).
  3. Measures what S7 never measured on a trained actor: the gap between the two networks at the SAME state, at every
     state of both trajectories (the gap injected per step), and the propagated part (the float64 network at the two
     trajectories' different states). Beside it, a float32 reference spread: the same actor's formula evaluated a
     second way in float32, in plain numpy (Numpy32), against the restatement at the same states. That is the size of
     float32 rounding at this actor's own activations, which grows with the network's scale.
  4. Controls that leave the restatement out. Judged: (e) the float64 adapter plus offsets of the measured injected
     size at each start, step and command coordinate, five sign draws: two with signs fixed per start and coordinate
     (same_fixed0-1), three with signs drawn per step (same_random0-2). Shown, not judged (their perturbations are
     smaller than the injected gap): (a) the float64 adapter with its command rounded to float32 (what env.step does,
     thruster_chain.py:443), (b) every float coordinate of the start moved by +1 ulp, (c) by -1 ulp, and (d) the
     float32 restatement with a +1 ulp start. Each is compared with J32 - J64 start by start (rank correlation, the
     worst start's rank).
  5. For the judged cell's worst start (and the run's overall worst start, if that is another cell; --detail-all:
     every cell's): per step, the state and command divergence, the injected and propagated gaps, the reward
     difference, and any branch taken differently by the two runs (observation float32 rounding, the 80% knee, a
     command at exactly +-1, flipped ReLU units for the raw-observation networks): counts of disagreements only.

Outcome-blind: it prints differences, ratios, ranks, indices, counts of disagreements and SHA-256 values only. It never
prints or returns a return, a points level, a mean return, a command or a state.

It writes nothing but its own output: no file under the repository, no cache, no bytecode. It calls no evaluate entry
point; it builds the same objects from their parts, so the gate's permission check (evaluate.require_permission) is
not on its path. It therefore checks the gate itself (gate.require_run_approval) and stops unless version 1's approval
verifies. It is not itself approved code: it runs because the user chose "Investigate" (PRESPEC.md 19.3) and makes no
record of the study.

Location: OUTSIDE /root/bca-hosts. gate.code_manifest hashes every file under experiments/shift_hosts/
(gate.package_files, rglob) and the EXTERNAL_CODE globs (gate.py:56-65, 108-151); gate.bytecode_files looks under the
package and in every folder holding an EXTERNAL_CODE file (experiments/ itself included, because of
experiments/__init__.py). A file outside the repository is in neither, so it changes no hash and cannot trip S1.

Options: --run-dir (default runs/shift_hosts/stage1/medium-expert/seed0/rebrac-alone), --compare (also td3_bc, cql
and iql alone, at the same --step), --step (an earlier snapshot of the same runs, e.g. 10000: is the tail growing with
training?), --list-s7 (every stage-1 s7.json's per-bank largest and mean differences), --detail-all, --out FILE (also
write the output to FILE, a new file outside the repository).

How to read it (PRESPEC.md section 19.6, fixed before it runs on a trained actor). Judged only on ReBRAC alone's
stage-1 actor at its final step (15,000), with its s7.json present; every other actor, step and cell is shown, not
judged, and the output marks the judged cells "JUDGED". The reading is "consistent with float32 rounding amplified by
the closed loop" when the lane's numbers reproduce on all seven banks and all three hold:
  (i)   on each of the seven banks, the largest injected gap (every state of both rollouts) is at most
        max(1e-5, 10 x the largest float32 reference spread at the same states): 1e-5 is stage 0's bound for the
        adapter against the restatement (test (h2)); the reference spread scales it to this actor's activations;
  (ii)  on big kicks' worst start, the propagated gap reaches the injected one at some step t >= 1
        (prop_t >= max(inj64_t, inj32_t)), and on every such propagation-dominated step with |dx_t| > 0 the state
        divergence grows by at most 10 times (one decade) in one step; steps where a new injection dominates are shown,
        not judged;
  (iii) on big kicks, the same-size control (e), taken as the largest of its five draws, moves the bank's largest
        difference by 0.1 to 10 times what the restatement does (the bank's largest |J32 - J64|); the same ratio at
        the worst start is shown, not judged, because that start is selected as the restatement's most extreme of
        1,000 and the control's value there is not, which biases that ratio low.
Anything else reads as a possible defect. Readings (ii) and (iii) test amplification only: control (e) takes its size
from the measured gap, so they hold for any small injected gap, defect or rounding alike. Reading (i) alone bounds the
injected gap, so a defect in the float64 adapter smaller than that bound per command coordinate is not excluded.
These rules were set after the cloud calibration on made-up actors and before any trained actor was probed: an
earlier absolute bound of about 1e-6 flagged defect-free made-up actors; the drafted rules (an absolute 1e-5 in (i),
every step in (ii)) flagged defect-free made-up actors whose kernels were scaled by 1.25 to 3, and the drafted
worst-start ratio of (iii) came within 1.6 times of its limit. The calibration's outputs are in calibration/outputs/
beside this file (outputs/shift_hosts_v2/ on the branch claude/shift-hosts-v2).
"""

import argparse
import hashlib
import math
import os
import sys
from pathlib import Path

import numpy as np

REPO_DEFAULT = "/root/bca-hosts"
RUN_DEFAULT = "runs/shift_hosts/stage1/medium-expert/seed0/rebrac-alone"
COMPARE = ("td3_bc-alone", "cql-alone", "iql-alone")
V1_FINGERPRINT = "8c5e8421922dccebe5b86f47b44c8d86def2dbdec7da84ba78ba2ba807728179"   # PRESPEC.md 19.1
JUDGED_RUN = "rebrac-alone"                        # the actor whose lane stopped (PRESPEC.md 19.2)
JUDGED_STEP = 15000                                # its final stage-1 step, the one S7 checked
JUDGED_BANK = "big_kicks"                          # the bank that failed: readings (ii) and (iii)
LIMITS = (0.001, 0.005, 0.01, 0.02, 0.05)          # points; v1's limit is 0.01, the proposed v2 limit 0.05
QUANTILES = (0.5, 0.9, 0.99, 0.999)
STAGE0_BOUND = 1e-5                                # reading (i): stage 0's test (h2) bound ...
REF_FACTOR = 10.0                                  # ... or 10 x the float32 reference spread, whichever is larger
GROWTH_DECADES = 1.0                               # reading (ii): at most 10 times per propagation-dominated step
WINDOW = (0.1, 10.0)                               # reading (iii)
FIXED_DRAWS, RANDOM_DRAWS = 2, 3                   # control (e)'s sign draws


def die(msg):
    print(f"probe refused: {msg}", flush=True)
    raise SystemExit(2)


class Tee:
    """stdout and a new output file (--out), line for line."""

    def __init__(self, stream, handle):
        self.stream, self.handle = stream, handle

    def write(self, text):
        self.stream.write(text)
        self.handle.write(text)
        return len(text)

    def flush(self):
        self.stream.flush()
        self.handle.flush()


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lg(v):
    """log10 for display; -inf for 0."""
    v = float(v)
    return float("-inf") if v <= 0 else math.log10(v)


def ranks(v):
    r = np.empty(len(v))
    r[np.argsort(v, kind="stable")] = np.arange(len(v))
    return r


def spearman(a, b):
    ra, rb = ranks(np.asarray(a)), ranks(np.asarray(b))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = math.sqrt(float((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


class Rounded:
    """Control (a): the float64 adapter's command rounded to float32, as env.step does (thruster_chain.py:443)."""

    def __init__(self, base):
        self.base = base

    def act(self, x, t):
        return np.asarray(self.base.act(x, t), np.float32).astype(np.float64)


class Offset:
    """Control (e): the float64 adapter plus an offset of a given size per start, step and command coordinate (the
    measured injected gap), with signs that do not come from the restatement (fixed per start and coordinate, or
    drawn per step). Its rounding size is the restatement's; its direction is not."""

    def __init__(self, base, size, signs):
        self.base, self.size, self.signs = base, size, signs

    def act(self, x, t):
        return np.asarray(self.base.act(x, t), np.float64) + self.size[:, t] * self.signs[:, t]


class Numpy32:
    """Reading (i)'s reference: the actor's formula evaluated a second way in float32, in plain numpy (its own
    summation order, tanh and exp), on the observation rounded to float32 as both networks see it, with nan_to_num as
    both have. |Numpy32 - restatement| is the size of float32 rounding at this actor's own activations. Layers as
    actors_f32 restates them: TD3+BC and IQL (o - obs_mean) / (obs_std + 1e-3), Dense_0-1 relu, tanh(Dense_2); ReBRAC
    Dense_0-2 relu, tanh(Dense_3); CQL Dense_0-2 relu, log_std Dense_3 clipped to [-5, 2], mean Dense_4,
    tanh(std eps + mean) or tanh(mean)."""

    def __init__(self, actor, A32, eps=None, readout="scored"):
        p = actor.params.get("params", actor.params)
        self.W = {k: np.asarray(v["kernel"], np.float32) for k, v in p.items() if isinstance(v, dict) and "kernel" in v}
        self.b = {k: np.asarray(v["bias"], np.float32) for k, v in p.items() if isinstance(v, dict) and "bias" in v}
        self.learner, self.readout = actor.learner, readout
        self.eps = None if eps is None else np.asarray(eps, np.float32)
        self.norm = None
        if actor.learner in A32.NORMALIZED:
            self.norm = (np.asarray(actor.norm["obs_mean"], np.float32),
                         np.asarray(actor.norm["obs_std"], np.float32) + np.float32(A32.NORM_EPS))
        self.log_std = tuple(np.float32(v) for v in A32.CQL_LOG_STD)
        if actor.learner == "cql" and readout == "scored" and self.eps is None:
            raise ValueError("CQL's scored command needs its declared noise")

    def _dense(self, h, k):
        return h @ self.W[k] + self.b[k]

    def act(self, x, t):
        h = np.asarray(x, np.float64).astype(np.float32)
        if self.norm is not None:
            h = (h - self.norm[0]) / self.norm[1]
        hidden = ("Dense_0", "Dense_1") if self.norm is not None else ("Dense_0", "Dense_1", "Dense_2")
        for k in hidden:
            h = np.maximum(self._dense(h, k), np.float32(0.0))
        if self.learner == "cql":
            log_std, mean = self._dense(h, "Dense_3"), self._dense(h, "Dense_4")
            if self.readout == "tanh_mean":
                out = np.tanh(mean)
            else:
                out = np.tanh(np.exp(np.clip(log_std, *self.log_std)) * self.eps[:, int(t)] + mean)
        else:
            out = np.tanh(self._dense(h, "Dense_2" if self.norm is not None else "Dense_3"))
        if out.dtype != np.float32:
            raise TypeError(f"the float32 reference computed in {out.dtype}")
        return np.nan_to_num(out).astype(np.float64)


def nudged(X, ds, direction):
    """The starts with every float coordinate (the first ds) moved by one ulp toward +inf or -inf."""
    Y = np.array(X, np.float64, copy=True)
    Y[:, :ds] = np.nextafter(Y[:, :ds], direction * np.inf)
    return Y


def relu_patterns(Ws, bs, obs):
    """ReLU on/off pattern of each hidden layer (float64) at float32-rounded observations, raw-observation nets."""
    h = np.asarray(obs, np.float32).astype(np.float64)
    out = []
    for W, b in zip(Ws, bs):
        z = h @ W + b
        out.append(z > 0)
        h = np.maximum(z, 0.0)
    return out


def hidden_layers(p64):
    """(Ws, bs) of the three hidden layers for ReBRAC (ReBRACPolicy) and CQL (CQLPolicy, CQLMeanPolicy); else None."""
    if hasattr(p64, "Ws"):
        return p64.Ws[:3], p64.bs[:3]
    body = getattr(p64, "body", None)
    if body is not None and hasattr(body, "Ws"):
        return body.Ws[:3], body.bs[:3]
    return None


class Probe:
    def __init__(self, repo, mods, detail_all=False, manifest_sha256=None):
        self.repo, self.detail_all = repo, detail_all
        self.manifest_sha256 = manifest_sha256
        self.A32, self.P, self.E, self.TR, self.TC, self.H = mods
        self.verdicts = []

    # ---------------------------------------------------------------------------------------------- one bank
    def bank(self, actor, readout, bank, s7_entry):
        A32, P, E, TR, TC = self.A32, self.P, self.E, self.TR, self.TC
        X = E.starts(bank)
        ids = np.arange(len(X))
        noise = P.bank_noise(actor, bank, ids, readout)
        p64 = P.adapter(actor, bank=bank, starts=ids, readout=readout)          # as closed_loop_check builds them
        p32 = A32.Float32Policy(actor, eps=noise, readout=readout)
        n32 = Numpy32(actor, A32, eps=noise, readout=readout)
        r64 = TR.rollout(p64, X[ids], E.PHYSICS, record=True)
        r32 = TR.rollout(p32, X[ids], E.PHYSICS, record=True)
        dp = np.abs(E.points(r32.J, bank) - E.points(r64.J, bank))
        dJ = np.abs(r32.J - r64.J)
        finite = bool(np.all(np.isfinite(r32.J)) and np.all(np.isfinite(r64.J)))
        L = r64.commands.shape[1]
        inj64, inj32, prop, ref = (np.empty((len(ids), L)) for _ in range(4))
        gap64 = np.empty((len(ids), L, r64.commands.shape[2]))
        for t in range(L):
            x64, x32 = r64.states[:, t], r32.states[:, t]
            a64_at64 = np.asarray(p64.act(x64, t), np.float64)                # = r64.commands[:, t] before the clip
            a32_at32 = np.asarray(p32.act(x32, t), np.float64)                # = r32.commands[:, t] before the clip
            a32_at64 = np.asarray(p32.act(x64, t), np.float64)
            a64_at32 = np.asarray(p64.act(x32, t), np.float64)
            gap64[:, t] = np.abs(a32_at64 - a64_at64)
            inj64[:, t] = gap64[:, t].max(1)                                     # injected on the float64 path
            inj32[:, t] = np.abs(a32_at32 - a64_at32).max(1)                   # injected on the float32 path
            prop[:, t] = np.abs(a64_at32 - a64_at64).max(1)                    # propagated (same network)
            ref[:, t] = np.maximum(np.abs(n32.act(x64, t) - a32_at64).max(1),  # float32 against float32
                                   np.abs(n32.act(x32, t) - a32_at32).max(1))
        ctrl = {}
        ctrl["round"] = TR.rollout(Rounded(p64), X[ids], E.PHYSICS).J
        ctrl["up"] = TR.rollout(p64, nudged(X[ids], TC.DS, +1), E.PHYSICS).J
        ctrl["down"] = TR.rollout(p64, nudged(X[ids], TC.DS, -1), E.PHYSICS).J
        ctrl32 = TR.rollout(p32, nudged(X[ids], TC.DS, +1), E.PHYSICS).J
        rng = np.random.default_rng([20261018, 7, P.BANKS.index(E.bank_name(bank))])    # control (e)'s signs only
        for d in range(FIXED_DRAWS):
            fixed = np.repeat(rng.choice([-1.0, 1.0], size=(len(ids), 1, gap64.shape[2])), L, axis=1)
            ctrl[f"same_fixed{d}"] = TR.rollout(Offset(p64, gap64, fixed), X[ids], E.PHYSICS).J
        for d in range(RANDOM_DRAWS):
            ctrl[f"same_random{d}"] = TR.rollout(Offset(p64, gap64, rng.choice([-1.0, 1.0], size=gap64.shape)), X[ids],
                                                 E.PHYSICS).J
        cdp = {k: np.abs(E.points(v, bank) - E.points(r64.J, bank)) for k, v in ctrl.items()}
        cdp["f32_up"] = np.abs(E.points(ctrl32, bank) - E.points(r32.J, bank))
        inj = max(float(inj64.max()), float(inj32.max()))
        refmax = float(ref.max())
        bound = max(STAGE0_BOUND, REF_FACTOR * refmax)
        out = dict(bank=E.bank_name(bank), readout=readout, starts=len(ids), finite=finite, dp=dp,
                   max_return_diff=float(dJ.max()), inj=inj, cdp=cdp,
                   i=dict(inj=inj, ref=refmax, bound=bound, passed=bool(inj <= bound),
                          within_stage0=bool(inj <= STAGE0_BOUND)))
        if s7_entry is not None:
            out["reproduced"] = (float(dp.max()) == s7_entry.get("max_points")
                                 and float(dJ.max()) == s7_entry.get("max_return")
                                 and float(dp.mean()) == s7_entry.get("mean_points"))
        j = int(np.argmax(dp))
        out["worst"] = j
        same = np.max([v for k, v in cdp.items() if k.startswith("same_")], axis=0)
        ratio_w = float(same[j] / dp[j]) if dp[j] > 0 else float("nan")
        ratio_b = float(same.max() / dp.max()) if dp.max() > 0 else float("nan")
        out["iii"] = dict(worst=ratio_w, bank=ratio_b, passed=bool(WINDOW[0] <= ratio_b <= WINDOW[1]))
        out["trace"] = self.trace(j, r64, r32, inj64, inj32, prop, p64)
        return out

    # ---------------------------------------------------------------------------------------------- one start
    def trace(self, j, r64, r32, inj64, inj32, prop, p64):
        """Differences only, per step, for start j (row j of the bank's batch, so the arithmetic is the lane's)."""
        TC = self.TC
        S64, S32 = r64.states[j], r32.states[j]                                # [L + 1, 13]
        C64, C32 = r64.commands[j], r32.commands[j]                            # [L, 4], clipped (applied)
        dx = np.sqrt(((S32[:, :TC.DS] - S64[:, :TC.DS]) ** 2).sum(1))           # L + 1
        da = np.abs(C32 - C64).max(1)
        dr = r32.rewards[j] - r64.rewards[j]
        obs_diff = (S32.astype(np.float32) != S64.astype(np.float32)).sum(1)    # coordinates the networks saw differ
        knee_xor = ((np.abs(C32) > TC.KNEE) != (np.abs(C64) > TC.KNEE)).sum(1)
        sat_xor = ((np.abs(C32) == 1.0) != (np.abs(C64) == 1.0)).sum(1)
        layers = hidden_layers(p64)
        relu = None
        if layers is not None:
            pa, pb = relu_patterns(*layers, S32[:-1]), relu_patterns(*layers, S64[:-1])
            relu = np.stack([(u != v).sum(1) for u, v in zip(pa, pb)], 1)     # [L, 3] units flipped per layer
        rows = []
        for t in range(len(da)):
            rows.append(dict(t=t, dx=dx[t], dx_next=dx[t + 1], da=da[t], inj64=inj64[j, t], inj32=inj32[j, t],
                             prop=prop[j, t], dr=dr[t], cum=float(dr[:t + 1].sum()), obs=int(obs_diff[t]),
                             knee=int(knee_xor[t]), sat=int(sat_xor[t]),
                             relu=None if relu is None else [int(v) for v in relu[t]]))
        steps = [t for t in range(1, len(da)) if dx[t] > 0 and dx[t + 1] > 0]
        jumps = [math.log10(dx[t + 1] / dx[t]) for t in steps]
        # reading (ii): only the steps where the propagated gap dominates the injected one
        dom = [t for t in range(1, len(da)) if prop[j, t] > 0 and prop[j, t] >= max(inj64[j, t], inj32[j, t])]
        dsteps = [t for t in dom if dx[t] > 0 and dx[t + 1] > 0]
        djumps = [math.log10(dx[t + 1] / dx[t]) for t in dsteps]
        dmax = max(djumps) if djumps else None
        growth = dict(max_one_step_decades=max(jumps) if jumps else None,
                      at_step=steps[int(np.argmax(jumps))] if jumps else None,
                      steps_growing=int(sum(1 for g in jumps if g > 0)),
                      total_decades=(math.log10(dx[-1] / dx[1]) if dx[1] > 0 and dx[-1] > 0 else None),
                      dominated_steps=len(dom), first_dominated=dom[0] if dom else None,
                      dominated_max_decades=dmax,
                      dominated_at_step=dsteps[int(np.argmax(djumps))] if djumps else None,
                      passed=bool(dom) and (dmax is None or dmax <= GROWTH_DECADES))
        return dict(rows=rows, growth=growth)

    # ---------------------------------------------------------------------------------------------- one actor
    def run(self, run_dir, step=None):
        A32, P, H = self.A32, self.P, self.H
        rec = H.read_json(run_dir / "record.json")
        final = int(rec["timings"]["outer_steps"])
        step = final if step is None else int(step)
        snap = (rec.get("snapshots") or {}).get(str(step))
        if not snap:
            die(f"{run_dir}: no snapshot at step {step} (have {sorted((rec.get('snapshots') or {}), key=int)})")
        actor = A32.read_actor_file(snap["actor"]["path"])
        if actor.npz_sha256 != snap["actor"]["sha256"]:
            die(f"{snap['actor']['path']} is not the record's actor")
        s7, s7_path = None, run_dir / "lanes" / f"s7-step{step}" / "s7.json"
        if s7_path.is_file():
            s7 = H.read_json(s7_path)
            print(f"probe: {s7_path} sha256 {sha256_of(s7_path)}", flush=True)
            smeta = (H.read_manifest(s7_path.parent / "states.npz").get("meta") or {})
            if smeta.get("actor_npz_sha256") != actor.npz_sha256:
                die(f"{s7_path} was made for another actor")
        made = (rec.get("code") or {}).get("code_manifest_sha256")
        print(f"probe: {run_dir.name}'s record was made on code manifest {str(made)[:16]}; this process runs "
              f"{str(self.manifest_sha256)[:16]} ({'the same' if made == self.manifest_sha256 else 'DIFFERENT'})",
              flush=True)
        judged = run_dir.name == JUDGED_RUN and step == final == JUDGED_STEP and s7 is not None
        print(f"\n=== {rec.get('run_id')} step {step}: learner {actor.learner}, actor {actor.npz_sha256[:16]}, "
              f"s7.json {'present (tol_points ' + str((s7.get('closed_loop') or {}).get('tol_points')) + ')' if s7 else 'absent'}"
              f"; {'JUDGED (PRESPEC.md 19.6)' if judged else 'shown, not judged'}", flush=True)
        results = []
        for readout in A32.READOUTS[actor.learner]:
            for bank in P.BANKS:
                entry = None if s7 is None else (s7.get("closed_loop") or {}).get(f"{readout}/{bank}")
                res = self.bank(actor, readout, bank, entry)
                res["judged_i"] = judged
                res["judged_cell"] = judged and readout == "scored" and res["bank"] == JUDGED_BANK
                results.append(res)
                self.print_bank(res)
        self.print_controls(results)
        worst = max(results, key=lambda r: float(r["dp"].max()))
        cell = [r for r in results if r["judged_cell"]]
        shown = results if self.detail_all else cell + ([worst] if worst not in cell else [])
        for r in shown:
            self.print_trace(r)
        if judged:
            self.print_verdict(rec.get("run_id"), step, results, cell[0])
        return results

    # ---------------------------------------------------------------------------------------------- printing
    @staticmethod
    def tag(judged):
        return "JUDGED" if judged else "shown"

    def print_bank(self, r):
        dp = r["dp"]
        q = np.quantile(dp, QUANTILES)
        counts = " ".join(f">{x:g}:{int((dp > x).sum())}" for x in LIMITS)
        top = np.argsort(-dp, kind="stable")[:10]
        rep = r.get("reproduced")
        i = r["i"]
        ratio = i["inj"] / i["ref"] if i["ref"] > 0 else float("inf")
        print(f"[{r['readout']}/{r['bank']}] starts {r['starts']} finite {r['finite']} reproduced "
              f"{'n/a' if rep is None else rep} | |dpoints| p50 {q[0]:.2e} p90 {q[1]:.2e} p99 {q[2]:.2e} "
              f"p99.9 {q[3]:.2e} max {dp.max():.6g} mean {dp.mean():.3e} | {counts} | max |dJ| "
              f"{r['max_return_diff']:.3e}", flush=True)
        print(f"    (i) injected gap max {i['inj']:.2e}; float32 reference spread max {i['ref']:.2e} (ratio "
              f"{ratio:.3g}); bound max(1e-5, 10 x spread) {i['bound']:.2e}: passed {i['passed']} "
              f"[{self.tag(r['judged_i'])}] (stage 0's 1e-5 alone, shown: {i['within_stage0']})", flush=True)
        print("    top 10 starts (index:|dpoints|): " + " ".join(f"{int(k)}:{dp[k]:.2e}" for k in top), flush=True)

    def print_controls(self, results):
        print("\n--- controls: |dpoints| of rounding-size perturbations that do not involve the restatement ---")
        print("    round = float64 adapter with float32-rounded commands; up/down = start floats +-1 ulp (float64 "
              "adapter); f32_up = restatement, +1 ulp; same_fixed0-1 / same_random0-2 = float64 adapter plus offsets "
              "of the measured injected size, signs fixed per start or drawn per step (control (e), judged in (iii) "
              "as the largest of its five draws); the others are shown, not judged", flush=True)
        for r in results:
            dp, j = r["dp"], r["worst"]
            parts = []
            for k, v in r["cdp"].items():
                rank = int((v > v[j]).sum()) + 1
                parts.append(f"{k}: max {v.max():.2e}, rho {spearman(dp, v):+.2f}, worst start {v[j]:.2e} "
                             f"(rank {rank})")
            print(f"[{r['readout']}/{r['bank']}] J32-J64 worst start {j}: {dp[j]:.2e} | " + " | ".join(parts),
                  flush=True)
            x = r["iii"]
            print(f"    (iii) same-size control (largest of 5 draws) / J32-J64: bank's largest {x['bank']:.3g} (within "
                  f"0.1-10: {x['passed']}) [{self.tag(r['judged_cell'])}]; at the worst start {x['worst']:.3g} (shown)",
                  flush=True)

    def print_trace(self, r):
        tr = r["trace"]
        print(f"\n--- worst start of {r['readout']}/{r['bank']}: start {r['worst']}, |dpoints| {r['dp'][r['worst']]:.6g}"
              f" (differences only) [{self.tag(r['judged_cell'])}] ---", flush=True)
        print("  t  log|dx_t| log|dx_t+1| log|da_t| log inj64 log inj32 log prop    d r_t       cum dJ  obs knee "
              "sat relu(l0,l1,l2)", flush=True)
        for w in tr["rows"]:
            relu = "-" if w["relu"] is None else ",".join(str(v) for v in w["relu"])
            print(f"{w['t']:3d} {lg(w['dx']):9.2f} {lg(w['dx_next']):10.2f} {lg(w['da']):9.2f} {lg(w['inj64']):9.2f} "
                  f"{lg(w['inj32']):9.2f} {lg(w['prop']):9.2f} {w['dr']:+.3e} {w['cum']:+.3e} {w['obs']:4d} "
                  f"{w['knee']:4d} {w['sat']:3d} {relu}", flush=True)
        g = tr["growth"]
        print(f"  all steps (shown): largest one-step growth {g['max_one_step_decades']} decades at step "
              f"{g['at_step']}; {g['steps_growing']} steps growing; {g['total_decades']} decades from step 1 to the end",
              flush=True)
        print(f"  (ii) propagation-dominated steps (prop >= max(inj64, inj32), t >= 1): {g['dominated_steps']}, the "
              f"first at step {g['first_dominated']}; largest one-step growth on them {g['dominated_max_decades']} "
              f"decades at step {g['dominated_at_step']} (at most 1): passed {g['passed']} [{self.tag(r['judged_cell'])}]",
              flush=True)

    def print_verdict(self, run_id, step, results, cell):
        rep = [r.get("reproduced") is True for r in results]
        i_ok = all(r["i"]["passed"] for r in results)
        worst_i = max(results, key=lambda r: r["i"]["inj"] / r["i"]["bound"])   # the bank nearest its bound (or over)
        ii, iii = cell["trace"]["growth"], cell["iii"]
        dp = cell["dp"]
        ok = all(rep) and i_ok and ii["passed"] and iii["passed"]
        self.verdicts.append(ok)
        print(f"\n=== READINGS (PRESPEC.md 19.6), judged on {run_id} step {step} ===", flush=True)
        print(f"  the lane's numbers reproduced bit for bit on {sum(rep)} of {len(rep)} banks: {all(rep)}", flush=True)
        print(f"  (i) injected gap within max(1e-5, 10 x float32 reference spread) on every bank: {i_ok} (highest "
              f"share of its bound: {worst_i['bank']}, {worst_i['i']['inj']:.2e} against {worst_i['i']['bound']:.2e})",
              flush=True)
        print(f"  (ii) {JUDGED_BANK}'s worst start {cell['worst']}: propagation reaches injection at step "
              f"{ii['first_dominated']}; largest growth on propagation-dominated steps {ii['dominated_max_decades']} "
              f"decades: {ii['passed']}", flush=True)
        print(f"  (iii) {JUDGED_BANK}: same-size control / J32-J64, bank's largest {iii['bank']:.3g}: {iii['passed']} "
              f"(at the worst start {iii['worst']:.3g}, shown)", flush=True)
        print(f"  {JUDGED_BANK} starts above 0.01 points: {int((dp > 0.01).sum())}; above 0.05: {int((dp > 0.05).sum())}",
              flush=True)
        print(f"  VERDICT: consistent with float32 rounding amplified by the closed loop: {ok}"
              + ("" if ok else "  -> reads as a possible defect, or the lane did not reproduce: do not freeze"),
              flush=True)


def list_s7(repo, H):
    """Every stage-1 s7.json's per-bank closed-loop maxima and means (differences only)."""
    root = repo / "runs" / "shift_hosts" / "stage1"
    for path in sorted(root.glob("*/seed*/*/lanes/s7-step*/s7.json")):
        body = H.read_json(path)
        cl = body.get("closed_loop") or {}
        cells = [f"{k}: max {v['max_points']:.2e} mean {v['mean_points']:.1e}" for k, v in sorted(cl.items())
                 if isinstance(v, dict) and "max_points" in v]
        print(f"{path.parent.parent.parent.name}: s7_passed {body.get('s7_passed')} open max_abs "
              f"{(body.get('scored') or {}).get('max_abs')} | " + " | ".join(cells), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", default=REPO_DEFAULT)
    ap.add_argument("--run-dir", default=RUN_DEFAULT, help="relative to --repo, or absolute")
    ap.add_argument("--step", type=int, default=None, help="snapshot step (default: the run's final step)")
    ap.add_argument("--compare", action="store_true", help="also the stage-1 alone actors of TD3+BC, CQL and IQL")
    ap.add_argument("--list-s7", action="store_true", help="print every stage-1 s7.json's per-bank maxima")
    ap.add_argument("--detail-all", action="store_true", help="a step table for every cell's worst start")
    ap.add_argument("--out", help="also write the output to this new file (outside the repository)")
    a = ap.parse_args(argv)

    if not sys.flags.dont_write_bytecode or os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        die("run with PYTHONDONTWRITEBYTECODE=1 and python -B: bytecode beside the hashed code trips stop rule S1")
    repo = Path(a.repo).resolve()
    here = Path(__file__).resolve()
    if repo in here.parents:
        die(f"keep this probe outside {repo} (it must not be in the code manifest or beside hashed code)")
    if a.out:
        out = Path(a.out).expanduser().resolve()
        if repo == out or repo in out.parents:
            die(f"--out {out} lies inside {repo}; write it outside the repository (for example /root/s7probe/out/)")
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            handle = open(out, "x", encoding="utf8")
        except FileExistsError:
            die(f"--out {out} exists; name a new file")
        sys.stdout = Tee(sys.__stdout__, handle)
    print(f"probe: {here} sha256 {sha256_of(here)}; argv {sys.argv[1:] if argv is None else argv}", flush=True)
    sys.path.insert(0, str(repo))

    from experiments.shift_hosts import actors_f32 as A32
    settings = A32.configure()                      # before JAX's backend starts: the S7 lane's flags and precision
    from experiments.shift import thruster_chain as TC
    from experiments.shift import truth as TR
    from experiments.shift_hosts import evaluate as E
    from experiments.shift_hosts import gate
    from experiments.shift_hosts import handoff as H
    from experiments.shift_hosts import policies as P

    manifest = gate.code_manifest()
    if any(here == (repo / k).resolve() for k in manifest):
        die("this file is in the code manifest")
    print(f"probe: python {sys.version.split()[0]}, numpy {np.__version__}, jax settings {settings}", flush=True)
    print(f"probe: code manifest {gate.manifest_sha256(manifest)[:16]} ({len(manifest)} files)", flush=True)
    try:
        ap_ok = gate.require_run_approval("S7 probe (read-only, outcome-blind)")
    except gate.ApprovalError as error:
        die(f"version 1's approval does not verify ({error}): stop; do not run the probe")
    if ap_ok.fingerprint != V1_FINGERPRINT:
        die(f"the approval that verifies is {ap_ok.fingerprint[:16]}, not version 1's {V1_FINGERPRINT[:16]}: this "
            "probe runs only on version 1's bytes")
    print(f"probe: approval {ap_ok.fingerprint[:16]} verifies (version 1; commit {ap_ok.commit[:7]})", flush=True)

    mods = (A32, P, E, TR, TC, H)
    probe = Probe(repo, mods, detail_all=a.detail_all, manifest_sha256=gate.manifest_sha256(manifest))
    if a.list_s7:
        list_s7(repo, H)
    run_dir = Path(a.run_dir)
    run_dir = run_dir if run_dir.is_absolute() else repo / run_dir
    probe.run(run_dir, a.step)
    if a.compare:
        for name in COMPARE:
            probe.run(run_dir.parent / name, a.step)

    left = gate.bytecode_files()
    print(f"\nprobe: bytecode beside hashed code after the run: {[str(p) for p in left] or 'none'}; wrote nothing "
          f"but {'--out ' + a.out if a.out else 'standard output'}", flush=True)
    return 0 if not left else 1


if __name__ == "__main__":
    sys.exit(main())
