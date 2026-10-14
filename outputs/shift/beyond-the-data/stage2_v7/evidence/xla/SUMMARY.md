# XLA evidence for Stage 2 version 7 (information only)

**This is an emulation forecast made on the cloud, not a v7 result and not a laptop result.** Every run below was made
on 14 Oct 2026 on the cloud's Xeon processor, outside the v7 wrapper, before `prereg_v7.json` exists or pins anything.
The laptop's own self-test (`V7_SPEC.md:944-970`, probes P1 and P2) replaces every number here.

It supplies two items the spec still lacked: the rerun of `xla_repro.py` (`V7_SPEC.md:252-254`) and the one-processor
repeat of the feasibility review's run (`V7_SPEC.md:839-847`). It also forecasts check (b) under emulation
(`V7_SPEC.md:643-661`).

## Words used here

- **Actor**: the small neural network that turns the system's state into thruster commands. Training changes its
  numbers (its *parameters*).
- **Gate's saved actor**: an actor that the earlier pre-registered gate test trained and saved under `runs/shift/`.
- **Same**: bit-identical. The SHA-256 fingerprint of all parameters equals the gate's, so the largest parameter gap is 0.
- **XLA**: the compiler that turns the JAX training code into processor instructions.
- **AVX-512, AVX2**: sets of wide arithmetic instructions. The cloud's Xeon has AVX-512. The laptop's Core Ultra 9
  275HX has only AVX2.
- **Emulated AVX2-only**: three settings on the cloud's Xeon. `XLA_FLAGS=--xla_cpu_max_isa=AVX2` stops XLA from using
  AVX-512. `OPENBLAS_CORETYPE=Haswell` makes numpy and scipy use their AVX2 matrix routines.
  `NPY_DISABLE_CPU_FEATURES=<AVX-512 list>` turns off numpy's own AVX-512 code. Each run checked, after loading numpy,
  that the emulation was active.
- **Cloud as it is**: the control. None of the three settings is used.
- **P1a, P1b, P1c, P2**: the spec's probes (`V7_SPEC.md:592-615`).
  - P1a is the gate's own check-2 training call.
  - P1b is the room test's program with the penalty weight at 0. It is the closest program to check (b) that does not
    run BCA.
  - P1c is the actor-only training that the fresh seeds 5-9 use.
  - P2 is the gate's plain TD3+BC host training (100,000 updates).
- **Check (b)**: the policy test's stop rule. BCA's program with every coefficient at 0 must match check 2's actor
  within 1e-5 on every parameter (`outputs/shift/beyond-the-data/stage2/code/policy.py:387-392`, `:545-548`).
- **F_ref, F_fresh**: the two forecast gaps (`V7_SPEC.md:643-649`). F_ref is P1b's actor against the actor check (b)
  would read. F_fresh is P1b's actor against P1c's.
- **Re-made reference**: the laptop's own copy of a gate actor. It is used only where the laptop does not reproduce
  the gate's actor (`V7_SPEC.md:625-641`).

## Every run

All 23 runs finished. None was refused, and none failed. The table cells give the result and the time in seconds.

**1. The gate's check-2 call, stalling world (`xla_repro.py`, the spec's P1a).** Each run retrains three actors side
by side (gapped, full and random; 20,000 steps).

| Gate seed | Emulated AVX2-only | Cloud as it is |
|---|---|---|
| 0 | 3 of 3 same, 213 s | 3 of 3 same, 209 s |
| 2 | 3 of 3 same, 211 s | 3 of 3 same, 207 s |
| 3 | 3 of 3 same, 215 s | 3 of 3 same, 209 s |
| 4 | 3 of 3 same, 266 s | 3 of 3 same, 211 s |

**2. The feasibility review's run, seed 2 (cloud as it is).**

| Call | Review run (processors not recorded) | Repeat on processor 0 |
|---|---|---|
| P1a stalling world (3 actors) | 3 same, 185 s | 3 same, 213 s |
| P1a no-stall world (2 actors) | 2 same, 128 s | 2 same, 142 s |
| P2 host (actor and critic) | both same, 489 s | both same, 578 s |

The repeat's reference files have the same SHA-256 as the review's: `check2_stall` `ac4348ea...` and `check2_no_stall`
`4ed93edb...`. The host snapshots are `cbc2b541...` and `fef4290d...`, which are also the gate's own snapshot
fingerprints.

The review did not record a processor count. Its file times show that the stalling-world run and the host run started
timing within the same second (02:55:24 UTC). So they ran side by side, and the no-stall run followed the stalling one.

**3. The check (b) forecast, seed 2.** These runs call `V7/code/v7_probes.py`'s `run_p1` (`:193`) and `run_p2`
(`:258`) directly.

| Probe | Emulated AVX2-only | Cloud as it is |
|---|---|---|
| P1 stalling world: P1a (3 actors), P1b, P1c | all same as the gate's, 402 s | all same, 420 s |
| F_ref and F_fresh, stalling world | 0 and 0 | 0 and 0 |
| P1 no-stall world: P1a (2 actors), P1b, P1c | all same as the gate's, 332 s | all same, 342 s |
| F_ref and F_fresh, no-stall world | 0 and 0 | 0 and 0 |
| P2 host: actor and critic | both same, 562 s | both same (table 2) |
| Re-made reference needed for seed 2 | none | none |

In every P1 run, F_ref was measured against the gate's saved actor, because P1a reproduced that actor exactly. In P2,
check 2's `td3bc_gapped` equals the gate's host actor.

**4. Does the emulation really change the machine code?** XLA's compiled code for one 4,000-step chunk of the seed-2
check-2 program (`experiments/shift/learners.py:139`) was counted with `objdump`. The count is of instructions that
use 512-bit registers, which only AVX-512 has.

| | Emulated AVX2-only | Cloud as it is |
|---|---|---|
| Whole program | 0 of 27,219 | 1,593 of 27,015 |
| The training loop alone | 0 of 7,263 | 916 of 7,502 |
| The three actors after the chunk | same bits in both modes | |

So the cap does change the code, and the result still came out bit for bit the same.

## What this predicts for the laptop (emulation forecast, which the laptop's self-test replaces)

On an emulated AVX2-only processor, every run gave exactly the gate's bits. That held for:

- the gate's check-2 call on gate seeds 0, 2, 3 and 4;
- both worlds of P1 for seed 2 (P1a, P1b and P1c);
- the 100,000-update host run P2 for seed 2.

It held although the emulated training loop used none of the 916 AVX-512 instructions that the cloud's version
contains.

If the laptop behaves like this emulation, two things follow:

- **No re-made reference looks needed.** P1a and P2 would be bit-identical, so C7's table would have no entry
  (`V7_SPEC.md:625-641`).
- **Q7's stop does not look likely.** Both forecast gaps, F_ref and F_fresh, were exactly 0, far below 1e-5, in
  both worlds. So the self-test's "NOT READY (P1b: check (b) forecast)" stop (`V7_SPEC.md:651-657`, `:1148`) does not
  look likely.

This is only a forecast, for these reasons:

- The emulation caps the instruction set on the Xeon. XLA still tunes its code for the Xeon's model.
- Code compiled ahead of time inside jaxlib is unaffected by the flag.
- The laptop's real Arrow Lake processor, WSL and system libraries may choose differently.
- Seeds 3 and 4 were tested only with the stalling-world P1a. P1b, P1c and P2 ran for seed 2 only.
- P1b is the room test's program, not BCA's own program. So even a zero gap makes a check (b) stop on the laptop
  unlikely, not impossible (`V7_SPEC.md:658-661`).

## How the runs were made

- **Driver.** `run_xla.sh` ran one job at a time. Each job ran on processor 0 at low priority (`taskset -c 0 nice -n
  10`). The v6 run kept processors 2 and 3, and another agent used processor 1. So the times are indicative only.
- **Environment.** Each process got exactly `launch_v7.sh`'s eight variables (`env -i`), plus the three emulation
  variables in emulated mode. `xla_emul_common.check_mode` refused anything else, and every process saw exactly one
  processor ([0]). JAX itself adds `TF_CPP_MIN_LOG_LEVEL` and `TPU_SKIP_MDS_QUERY` when it is imported. They appear in
  the recorded variables, after the check.
- **Constants.** In every run except the feasibility repeat, the constants of `V7/world_constants.json` (SHA-256
  `6602f9d1...`) were substituted by `V7/code/v7_constants.py:210` (`apply`), as the wrapper does it. The local
  values differed from the file as follows:
  - emulated runs: A by at most 1.73e-18 and K by at most 6.99e-15; B was identical;
  - cloud-as-it-is runs: no difference at all.

  The tripwire counted 0 calls in every run. The feasibility repeat substituted nothing, as the review's run did.
- **The gate's software check.** It refuses any `XLA_FLAGS` (`experiments/shift/acceptance.py:115`, `:491-502`). In
  emulated runs it ran with only the `XLA_FLAGS` item excused. That item was compared as if unset, which is how the
  gate's pre-registration recorded it, and every other item was still checked. In cloud-as-it-is runs the check ran
  unchanged. The machine-code check appends a dump flag to `XLA_FLAGS`, so it excused that item in both modes.
- **Why not the probes' command line.** `v7_probes.py`'s command line was not used for table 3, because its
  environment check refuses these variables (`V7/code/v7_verify.py:41`). Its record files went to a scratchpad folder,
  never to `runs/`, and were copied here. Their `inputs` block names the modules used:
  - `v7_probes.py` `c6521973...` and `v7_constants.py` `a055d067...`, both as committed at `317a001`;
  - no draft re-make block, because none exists yet.
- **Machine.** Intel Xeon Processor @ 2.10GHz (family 6, model 207, with AVX-512), Linux 6.18.44, glibc 2.39, Python
  3.10.20, numpy 2.2.6, scipy 1.15.3, jax and jaxlib 0.6.2. OpenBLAS used the SkylakeX kernel, or Haswell when
  emulated.

## Files

- `SHA256SUMS`: every file in this folder.
- `xla_repro_original.py`: the earlier script, byte for byte (`25d2d84a...`).
- `xla_repro.py`: the adapted copy, with its changes listed in its docstring.
- `xla_emul_common.py`: the shared set-up.
- `run_xla.sh`: the driver.
- `driver.txt`: the driver's start and end times.
- `logs/`: each run's output.
- `repro/`: section 1.
- `feas_review_seed2/`: section 2.
  - `cloud_review_run/`: the review's script and `out/`, copied unchanged.
  - `cpu0_repeat/`: the repeat.
  - `SHA256SUMS`: the folder's own list.
- `forecast_seed2.py` and `forecast_seed2/{emulated,default}/`: section 3, with the probe records and their sidecars.
- `xla_dump_check.py` and `dump_check/`: section 4.
- `summarize_xla.py` and `summary.json`: everything above, gathered into one file.

Model files (`*.msgpack`) are ignored by git (`.gitignore:11`), so they will not be committed. Their SHA-256 are in
`SHA256SUMS` and in the JSON records. The models from these runs stay in the scratchpad folder `v7_build2/`.
