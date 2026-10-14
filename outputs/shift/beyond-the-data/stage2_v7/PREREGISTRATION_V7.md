# Stage 2, version 7: the declaration (pre-registration)

**Version 6 governs.** Version 6 of the Stage 2 pre-registration, `outputs/shift/beyond-the-data/stage2/prereg.json`
with SHA-256 fingerprint `9b38c7c48a303cfd67e5db6f137e3a162997bb533d86b3f7647e0467b91ca3b1` (code manifest
`4b34af991a56066c41fbb55cb02db4074ae0b7e85b1e50802a53983edda2a753`, approved in commit `7dab7d5`, imported in commit
`2eab3c4`, runs-side approval `b78b87726a2e7030dfb45a0ad45d86cbae19cc46fb9f4b8d9724975cfb6c2e72`), stays the governing
document for everything this declaration does not list as changed. Version 7 ("v7") changes operations only: where
and how the remaining Stage 2 phases run. Each v7 phase needs both version 6's approval and version 7's.

**What you approve.** The machine-readable form of this declaration is `prereg_v7.json` in this folder. You approve
its fingerprint with `approve_local_v7.py` (README.md, step 10). It is written twice: a draft (it cannot be approved),
then, after the handoff, the final, which may differ from the committed draft only in the handoff fields; the code
(`code/write_prereg_v7.py`) enforces that. Every v7 file's fingerprint is in `code_manifest_v7.json`, which
`prereg_v7.json` pins.

**Where this text comes from.** Sections 1, 3, 5 and 7 below are those sections of the v7 specification (version 3,
14 Oct 2026), unchanged except for cross-references to parts not reproduced here; section 2 of the specification
(what was known when v7 was written) is the appendix. The steps you take are in README.md, which is section 6 of the
specification. The last part, "Details the code fixes", lists where the code settles a detail the specification
left open. File references give `file:line` in the files as they are now; `S2C/`, `GATE/` and the other shorthands
are defined just below.

A *test* named `test_v7.py` below is in `code/test_v7.py` or, for the laptop package and the cloud scripts, in
`code/test_v7_laptop.py`.

**Path shorthands used below**

| Shorthand | Path |
|---|---|
| `S2/` | `outputs/shift/beyond-the-data/stage2/` (v6, pinned) |
| `S2C/` | `outputs/shift/beyond-the-data/stage2/code/` (v6 code, pinned; byte-identical to the copy the cloud runs from) |
| `GPU/` | `outputs/shift/beyond-the-data/stage2_gpu/` (the v6 package for the user's machine, unchanged) |
| `GATE/` | `experiments/shift/` (the gate's code, hashed into the gate's pre-registration) |
| `ROOM/` | `runs/shift_explore/room/code/` (the room test's pinned code) |
| `OUT/` | `runs/shift_explore/stage2/` (v6's output folder) |
| `V7/` | `outputs/shift/beyond-the-data/stage2_v7/` (NEW, committed) |
| `R7/` | `runs/shift_explore/stage2_v7/` (NEW, run-side, not committed) |
| `PROBE/` | `/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/v7_hw_probe/` (cloud scratchpad; not in git) |

**Terms**

- *Phase*: one step of the pre-registered work, run as one Python process (for example "train the policy test on seed 6").
- *Lane*: a slot in which one phase runs. "Processes at once" means lanes in use.
- *Processor*: one logical processor as Linux sees it. Inside WSL (Windows Subsystem for Linux) each is a virtual
  processor that Windows places on a real core of its choosing.
- *AVX-512*: a set of wide arithmetic instructions. The cloud's Xeon has it. The laptop's Core Ultra 9 275HX does not.
- *OpenBLAS kernel*: the family of matrix routines that numpy and scipy choose for a processor. Different kernels can
  give different last bits.
- *ULP* ("unit in the last place"): the gap between a float64 number and its nearest neighbour.
- *Gate*: the earlier pre-registered ThrusterChain test. Stage 2 builds on its saved files (`runs/shift/`) and code (`GATE/`).
- *Anchors*: 64 numbers recorded in the gate's pre-registration (the returns of two reference controllers, with their
  standard errors) that define 0 and 100 points. Every phase recomputes them and requires exact equality
  (`GATE/acceptance.py:862`, `:959-966`).
- *A, B, K*: ThrusterChain's plant matrices (A is 13x13, B is 13x4) and the textbook controller gain (K is 4x13).
- *Check 2*: the gate's retraining of the actor alone, from the cloned actor, against each saved critic. It made the
  "none" actor (no caution) that Stage 2 compares with (`GATE/acceptance.py:1459-1484`).
- *Check (b)*: the policy test's stop rule. The BCA program with all coefficients 0 must match check 2's "none" actor
  within 1e-5 on every parameter, or the seed stops (`S2/prereg.json:389`; `S2C/policy.py:387-392`, `:545-548`).
- *The room test*: the earlier pre-registered test of the training program with an oracle penalty (`ROOM/`). Its pass
  is a condition for the policy test.
- *Strength matching*: a control arm's coefficient is tuned until its actor's commands move as far from a reference
  actor as BCA's do (root-mean-square change on 5,000 evaluation states). *Hinge hosts* are full TD3+BC runs with a
  range-hinge penalty, matched in this way to the BCA host (`S2C/host.py:23-27`, `:368-401`).
- *Variant B*: the deployment check. At run time the actor's command is replaced by a clipped one when BCA's width U
  exceeds *tau*, a threshold set from calibration rows (`S2/prereg.json:220-224`; `S2C/deploy_check.py:95-104`).
- *Fingerprint*: the SHA-256 of a file's bytes. A *sidecar* is the `<file>.sha256` file that holds it.
- *Run files*: the gitignored outputs under `runs/`. The cloud pushed them to the orphan branch `claude/stage2-run-files`.
- *Final record*: the file whose presence, with a matching sidecar, means a phase finished (table in C5).
- *Reference*: one of the gate's saved actors that a laptop-trained actor is measured against (C7 lists three).
- *Re-made reference*: the laptop's own copy of a reference, made with the gate's exact call (C7). Used only for a
  seed 2-4 phase whose probes show the laptop reproduces the gate's copy bit for bit neither with the gate's call (P1a,
  P2) nor, for a policy reference, with the room's program (P1b).
- *Phase-specific interruption*: a phase killed while the machine stayed up, by a signal the runner did not send, or
  ending in a memory error (C5). Restarts of the machine or of WSL are not phase-specific.
- *Unfinished*: a phase with three phase-specific interruptions and a fourth when run alone (C5). It is retried until
  done; the user's `--retry` sets only when. Never a failure.
- *Probes*: P1 per seed and world: P1a (the gate's check-2 call, which also makes the re-made policy references), P1b
  (the room's program at W = 0, the check (b) forecast) and P1c (the fresh seeds' actor-only structure, the forecast for
  seeds 5-9); P2 (host reproduction, which makes the re-made host reference); P3 (repeatability); all without BCA.
  R1 (band refit comparison, information only, after approval).
- *Check (b) forecast*: the largest parameter gap, on the laptop, between P1b's actor and the actor check (b) will be
  compared with (C7). Above 1e-5 the self-test says NOT READY.
- *Ledger*: v7's per-phase record of what ran, where, and which files it wrote (C10).

---

## 1. What v7 is for, what it changes, and what it does not change

**Purpose.** v7 lets the remaining Stage 2 work run on the user's laptop. Under v6 that is impossible: the laptop's
processor has no AVX-512, so the gate's exact checks refuse every phase. v7 supplies the gate's three world constants
(A, B, K) from a file made on the cloud's processor, so every unchanged gate and v6 check still runs in full. It also
allows more processes at once on the laptop, and fixes in advance how seeds 0-4 continue across the two machines.
Once v7 has started, nothing is held back: every remaining phase runs. Before it starts, the laptop's self-test can
say NOT READY, for example when the check (b) forecast fails (C7, C11); the cloud then keeps running v6 (C6). The v6
study code runs byte for byte as approved. Each phase needs
both v6's approval and v7's.

**What v7 changes** (every operational departure from v6; details in section 3 below):

1. *World constants*: A, B and K come from a file made in the cloud (C1).
2. *Processes at once*: up to 16 on the laptop, instead of v6's "at most 2" (`S2/prereg.json:54`); none in the cloud
   after the handoff (C2, C6).
3. *Interrupted phases*: a killed phase's files are moved out of v6's output folder (never deleted) and the phase
   reruns: without limit after a machine-wide interruption; after phase-specific ones, twice, then once alone, then on
   the user's `--retry`, until done (C5). v6 has no written rule for this. Every other output stays write-once.
4. *Re-made references*: on seeds 2-4 only, when the laptop does not reproduce one of the gate's saved reference
   actors bit for bit (C7's rule), the laptop makes its own copy with the gate's exact call, and only that seed's and
   world's training phase reads it. That replaces the actors behind these locked arms, per seed and world with an
   entry: in the stalling-world policy test, "none" and "full_none" (and so every strength measured from "none"); in the
   no-stall policy test, "none"; in the stalling-world full host, "plain" (and so its target strength and its
   matching). v6 names the gate's saved files (`S2/prereg.json:383`, `:272`).
5. *Check (b) forecast*: before anything is approved, the self-test measures whether the laptop's own programs agree
   as check (b) requires; a gap above 1e-5 makes it NOT READY (C7).
6. *Environment*: Ubuntu 24.04 in WSL, OpenBLAS kernel left to its default, an exact environment per phase (C4).
7. *Compute budget*: v6's "about 72-92 process-hours" stays an estimate that stops nothing, as in v6 (no v6 code
   enforces it). v7 counts the process-hours actually used and reports them (C2).
8. *Records*: a ledger, provenance files and the probes (C7, C10, C11).

Running on another machine can itself cause recorded failures that the cloud would not have had. Section 5 lists them.

A v6 output written under v7 names v6's approval (`approval_sha256`), whose text allows at most 2 processes. That
field alone does not show which rules applied. The ledger and provenance files do (C10).

**v7 does not change any of these** (all stay exactly as in `S2/PREREGISTRATION.md` and `S2/prereg.json`, version 6,
fingerprint `9b38c7c48a303cfd...`):

- the method: BCA's band, the uniform and weighted versions, k, every ladder, every coefficient, and the frozen choice
  of the weighted version and of the rival stand-in (`range_hinge`);
- the detection test's pair sets, labels, widths, marks and claim rule;
- the policy test's arms, matching rules, marks M1-M5, and check (b) with its 1e-5 tolerance and its stop rule (on a
  seed 2-4 world with a C7 entry, the actors behind "none" and "full_none" are laptop-made copies; the arms and rules
  are unchanged);
- the deployment check (Variant B), tau's rule and its actors (for seeds 0-4 it always reads the gate's saved "none"
  and plain TD3+BC, never a re-made copy); it runs in every case, as v6 says (`S2/prereg.json:223`);
- the full host, including its coefficient-0 check (1e-5, reported) (on a seed 2-4 with a C7 host entry, the actor
  behind "plain" is a laptop-made copy);
- the confirmation rules: fresh seeds 5-9 alone carry the claim; a failed seed is recorded and never replaced;
- the analysis, the pooled looks and the claim sentence;
- the processor-only rule (no GPU), the recorded Python and library versions, the numeric flags, one processor per phase;
- the approval route and the refusal of agent identities;
- the gate's pre-registration, its files and all its checks, which stay exact to the last bit.

---

## 3. The changes

Each change gives the v6 rule and where it lives, the v7 rule, why, how code enforces it, and how it is tested.

### C1. Running on a processor without AVX-512: "pinned world constants"

**v6 rule.** Every phase that builds a seed context runs the gate's `load_prereg` and `guard` before any work
(`S2C/fresh.py:42-46`; `ROOM/harness.py:215-216`). `load_prereg` rebuilds the gate's whole pre-registration body,
including `anchors = anchor_table(...)`, and refuses on any difference outside the volatile keys; "anchors" is not
volatile (`GATE/acceptance.py:753`, `:862`, `:959-966`). For gate seeds 0-4, each context also regenerates every
world's log and compares each array's SHA-256 with the gate's record (`GATE/acceptance.py:705-716`, called at
`ROOM/harness.py:225`).

**What goes wrong on the laptop.** Two constants come from LAPACK, whose last bits depend on the OpenBLAS kernel:
A, through an SVD norm, `np.linalg.norm(X, 2)` (`experiments/signal/gapped_lq.py:239`, used by
`GATE/thruster_chain.py:161-176`); and K, through `scipy.linalg.solve_discrete_are` and `np.linalg.solve`
(`GATE/recorder.py:75-84`). Under an emulated AVX2-only processor (Haswell kernels, numpy's AVX-512 paths off),
71 of A's 169 entries differ (largest 1.73e-18, at most 2 ULP), all 52 entries of K differ (largest 7.0e-15), 44 of
the 64 anchors differ (largest 3.4e-13, at most 20 ULP) and all 10 base gate logs fail their hash check
(`V7/evidence/probe_Haswell_unpinned.json`; earlier `PROBE/haswell_npyoff.json`). Every kernel family the laptop can
run (Haswell, Sandybridge, Nehalem, Katmai) leaves 42-58 anchors differing (up to 27 ULP, under Nehalem), changes 71 of
A's entries and 49-52 of K's (49 under Katmai), and K differs by at most 1.0436e-14 (Katmai) (`V7/evidence/probe_summary.json`).
Only SkylakeX kernels reproduce the cloud's bits, and they need AVX-512. Pinning the kernel therefore cannot fix this.

**v7 rule.** Every v7 phase process does the following, in order, before any Stage 2 or gate computation:

1. Load `V7/world_constants.json`: the cloud's exact A, B and K as float64 hexadecimal strings, keyed by the exact
   argument tuples the gate uses, `(gain, wave_turn)` for A and B and `(gain, wave_turn, throttle_cost, gamma)` for K.
   Refuse unless its SHA-256 equals the value in `V7/prereg_v7.json`.
2. Recompute A, B and K locally with the gate's unchanged functions and record the differences. Refuse unless the
   largest absolute difference is at most **1e-14 for A** and **1e-12 for K**, and B is bit-identical. Measured
   worst cases: 1.73e-18 (A) and 1.0436e-14 (K). This tolerance only confirms that the file describes this world. It is
   never used to accept an anchor or a log.
3. Replace `experiments.shift.thruster_chain._matrices` and `experiments.shift.recorder._gain` with functions that
   return the file's arrays (read-only, float64, the same array object on every call, as `lru_cache` gives now), and
   clear `thruster_chain._columns`' cache. An argument tuple not in the file stops the process (step 4's exit).
   There is no fallback to computing it.
4. Install a tripwire on `experiments.signal.gapped_lq.transition_matrix` and `scipy.linalg.solve_discrete_are`. Every
   caller found goes through the two replaced functions (`GATE/recorder.py:77`, `:89`, `:202`, `:258`, `:263`, `:267`;
   `GATE/learners.py:591`; `GATE/truth.py:391`, `:413`, `:433`; `GATE/acceptance.py:2720`; `GATE/stage0.py:399`, `:438`;
   `GATE/thruster_chain.py:181`, `:186`, `:209`, `:238`, `:352`; the full list is `V7/evidence/argument_tuples.json`),
   so no call is expected. **The first call stops the process at once**: the tripwire writes
   one line to the phase's ledger file and calls `os._exit(97)`. It never raises an exception, because pinned v6 code
   catches any `Exception` around the stand-in rival and records it as the rival failing (`S2C/policy.py:419`, `:558`).
   `launch_v7.sh` classifies exit 97 as a v7 refusal (C5).
5. For a phase with an entry in C7's re-make table only: install the reference redirect (C7, step 3).
6. Run the pinned `run_stage2.py` unchanged, as a script (`runpy.run_path(..., run_name="__main__")`, `sys.argv` as
   `launch.sh` sets it), from the folder `launch.sh` fixes
   (`/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/ood/stage2/`, `S2C/launch.sh:10`, `:19`).

**Acceptance rule (declared now).** A processor is accepted if and only if steps 1-2 pass and the **unchanged gate
checks then pass exactly**: `load_prereg` and `guard` accept all 64 anchors to the last bit, with the software flags
equal; and, for seeds 0-4, `load_seed` matches every log array's SHA-256 for each world the phase loads. v7 adds no
tolerance to any gate check. If any refuses, the phase refuses and writes nothing.

**Why this and not a tolerance on the anchors.** The gate's exact checks stay exact, and become end-to-end proof that
the laptop computes the gate's world bit for bit. A tolerance on the anchors would not fix the logs: under emulation,
seed 3's regenerated training actions differ.

**Evidence status (14 Oct, saved).** The constants file exists: `V7/world_constants.json`, SHA-256
`6602f9d17e977a3e1f6eb017eb1f0b9ec80b018041d62146f507983f36a5e379` (its sidecar matches), written by
`V7/code/make_world_constants.py` with SkylakeX kernels (numpy 2.2.6 with OpenBLAS 0.3.29, scipy 1.15.3 with OpenBLAS
0.3.28, Python 3.10.20, glibc 2.39) on the cloud's Xeon (family 6, model 207, with AVX-512); it records the full
processor flag list and no time, so a rerun gives identical bytes (checked on a second processor), and it refuses to
overwrite a file with different content. Before writing, the maker ran the gate's `load_prereg` and `guard` with
nothing substituted, and they accepted. Exactly one argument tuple exists for each function: `_matrices(0.3, 0.27)`
and `_gain(0.3, 0.27, 0.3, 0.98)`, keyed by their float64 hexadecimal forms; all 65 physics sources found (55 gate
data records and the gate pre-registration's world block among them) give these tuples, and no code outside tests
sets gain, wave_turn, throttle_cost or gamma (`V7/evidence/argument_tuples.json`, `crosscheck_constants.json`). The
file is bit-identical to this machine's own values and to the earlier `plant_default.npz`, whose sums are in
`V7/evidence/npz_sha256.json`.

The reruns of `V7/evidence/pinned_constants.py` (driver `run_probes.sh`, one processor, nice 10, the gate's software
check on, `XLA_FLAGS` unset, numpy's AVX-512 paths confirmed off in the four emulated families) were saved as
`V7/evidence/probe_<family>_{pinned,unpinned}.json` and summarised in `probe_summary.json`. They agree with the earlier
report: with the constants substituted, the unchanged gate accepted all 64 anchors exactly and `load_seed` matched all
10 base gate logs under every family (default SkylakeX, Haswell, Sandybridge, Nehalem, Katmai; about 90-95 s each);
the replacements were asked only for the two tuples, the originals were never reached, and the tripwire counted 0
calls. Without the constants, the four emulated families leave 42-58 anchors differing and 0 of 10 logs matching.
Every family is within the tolerances of step 2. The rerun of `xla_repro.py` (C7) is saved in `V7/evidence/xla/`
(`summary.json` and its sidecar, `SHA256SUMS`, the scripts, logs and records; information only, run on the cloud's
Xeon outside the v7 wrapper, one processor at a time): with the constants substituted, check 2's actors of seeds 0, 2,
3 and 4 were reproduced bit for bit both on the cloud's processor as it is and with an AVX2-only processor emulated
(`XLA_FLAGS=--xla_cpu_max_isa=AVX2`, Haswell OpenBLAS kernels, numpy's AVX-512 paths off); for seed 2, P1a, P1b, P1c
and P2 run the probes' calls under the emulation were bit-identical to the gate's actors, so both forecast gaps F_ref
and F_fresh were exactly 0; and the training program's compiled code, although it holds no AVX-512 instruction under
the emulation, gave the same bits after one chunk. An emulation cannot show what the laptop's own processor does (XLA
still tunes for the Xeon's model); the laptop's self-test decides. The earlier glibc 2.40 run was not repeated; on the
laptop the self-test decides (ST4, ST6).

**Why it is a declared departure.** At run time it changes what two gate functions return, although they return the
gate's own recorded-machine values. No pinned file is edited. The user accepted it (Q1).

**Enforced by:** `V7/code/v7_constants.py` (steps 1-4), called first by `V7/code/run_stage2_v7.py`. Importing
`thruster_chain` also imports JAX and Flax (through `experiments/signal/gapped_lq.py`); the wrapper runs **no JAX
computation** before the hand-over, and JAX picks its backend only at first use, under the one-processor affinity.
All v7 module names start with `v7_` or `run_stage2_v7`, so none can shadow a v6 or gate module, and the wrapper
removes its own folder from `sys.path` before the hand-over. `OPENBLAS_CORETYPE` stays unset (C4).

**Tested by:**
- (a) `V7/code/test_v7.py`: refusal when the file is changed, for an unknown argument tuple (exit 97), when a
  tolerance is exceeded or B differs; read-only returns; cache cleared; a tripwire call exits 97 even inside an
  `except Exception` block.
- (b) in the cloud, before the file is pinned: the reruns above (done, saved), including the default SkylakeX
  kernels, where the file makes no difference, the gate accepts, and all 10 base logs regenerate.
- (c) on the laptop, the self-test (C11).
- (d) `V7/evidence/` holds the probe scripts, every JSON output, and the SHA-256 and size of each `.npz` file. The
  `.npz` files themselves (33-88 MB each, ignored by `.gitignore:12`) stay out of git.

**Fallback (declared now).** If the laptop fails the gate check even with the file (for example because of the C
library, C4), nothing is loosened. The run does not start and the user is told. Any other route is a new version.

### C2. Processes at once, memory and the compute budget

**v6 rule.** "A compute budget of about 72-92 process-hours, at most 2 processes at once" (`S2/prereg.json:54`,
`:200-201`). No v6 code enforces either part. `S2C/launch.sh:1-21` starts one process. Only the `GPU/` runner limits
lanes to 1 or 2 (`GPU/lib/runner.py:322-323`).

**v7 rule (the user's decision, Q3).** At most **16** Stage 2 phase processes at once, **all on the laptop**; default
**12**. **None** run in the cloud from the handoff on (C6). The lane count may be lowered or raised (1-16) on any
restart; it changes no number. WSL is given 48 GB.

**Why 12, with 16 as the cap.**
- Work left, at the cloud's per-core speed: about 48-58 process-hours, less what the cloud finishes before the
  handoff. Seeds 5-9: about 38-48 (`GPU/README.md:22-23`). Seeds 2-4: policy tests about 3.4 (3 x about 68 min, cloud
  log); deployment about 0.4; stalling-world hosts about 3.8 (3 x about 75 min); no-stall hosts about 1.3 (3 x 21-29
  min); evaluations about 0.1. Probes about 1.5.
- One fresh seed's longest chain, about 4.3 hours: data about 8 min and learners about 40 min (`S2/prereg.json`
  compute, `fresh_seeds_build`), no-BCA detection about 15 min per world (`fresh_seeds_bca`: about 2.5 h for 5 seeds),
  band fits and scores about 20 min, policy test about 68 min, stalling host about 75 min, no-stall host about 30 min.
- So 12 lanes give at least max(48-58 / 12, about 4.3) = **about 4.5-5 hours at cloud per-core speed. This is a lower
  bound, not a forecast**: lanes wait on dependencies, and the laptop's per-core speed under load is unknown.
  16 lanes would save little.
- Cores: 24 logical processors; 12 lanes leave half for Windows, WSL and thermal headroom.
- The count changes no number: every random stream comes from fixed seeds; time and process ids appear only in logs.
  Two cloud lanes in parallel gave check (b) exactly 0.0 (`S2C/policy.py:6`, `:23`; `S2C/band.py:244-248`).

**Memory (declared now).** The runner keeps a reserved peak for each phase kind (for example `policy_train`). A kind
not yet measured on the laptop reserves **7 GiB**. After each finished run of a kind, it reserves 1.25 times the
largest VmHWM measured for that kind on the laptop, except `policy_train` and `policy_no_stall_train`, which reserve
the larger of **7 GiB and twice** that peak: they are the only kinds in which pinned code turns a memory error into a
written result (below). A phase starts only if (i) the reserved peaks of the running phases
plus its own are at most WSL's `MemTotal` minus **8 GiB**, and (ii) `MemAvailable` is at least its own reservation plus
8 GiB. At most **2** `policy_train` phases run at once until one has been measured (the cloud ran 2 in 16 GB, so their
sum was under about 16 GB; the size of each is unknown). Why: memory exhaustion can appear as a Python exception rather
than a kill, and inside the stand-in rival's path pinned v6 code records any exception as the rival failing, which is
written once and counts against BCA (mark M4) (`S2C/policy.py:408-420`, `:550-559`).

**Stand-in errors (declared now).** When a new policy lock records `standin.error` or `standin.no_stall_error`, the
runner starts no new `policy_train` or `policy_no_stall_train` phase (the only phases that run the stand-in rival,
`S2C/policy.py:410-420`, `:549-559`) and prints the error; every other phase continues. The lock stays (write-once).
If the error names memory (`MemoryError`, `RESOURCE_EXHAUSTED` or "out of memory"), the provenance file labels that
seed's stand-in comparison "machine-caused: memory". New policy training starts again only when the user's review is
recorded with `run_v7.sh --reviewed <phase>`, typed at a terminal: from a second window while the runner runs (the
runner picks it up within its retry poll, as it does a `--retry` request), or as a restart. Typing it again only says
the review exists.

**Compute budget (declared now; Q3).** v6's 72-92 process-hours stays an estimate that stops nothing. The provenance
file reports the process-hours used: the cloud's from its logs, the laptop's from the ledger, killed attempts, reruns
and probes included, each listed separately (laptop hours are at the laptop's speed).

**Enforced by:**
- `V7/laptop/lib/runner_v7.py`: `--lanes` 1-16, default 12, the lanes being the first N of the 16 processors the
  self-test checks with the gate (ST4, ST7); a machine-wide `flock` (as `GPU/lib/runner.py:333-337`);
  the memory admission rule; the stand-in error block on new policy training.
- `V7/code/launch_v7.sh`: one of 16 slot locks (`flock -n R7/slots/slot_<i>.lock`), refusing when none is free, so
  the cap holds even outside the runner; and a per-phase lock (`flock -n R7/locks/<phase>.lock`), because the v6
  writers check first and write second and cannot stop a double start (`S2C/stage2_common.py:247-252`).
- The ledger records each phase's VmHWM and CPU time.

**Tested by:** `test_v7.py`: a 17th start refuses; a double start refuses; admission refuses when reservations exceed
the limit; the policy_train cap; the policy kinds reserve max(7 GiB, 2 x peak); a lock with `standin.error` blocks
new `policy_train` and `policy_no_stall_train` starts while other phases still start. Self-test P3 (C3).

### C3. One processor per phase: kept, and checked in code

**v6 rule.** `taskset -c "$CPU"` in `S2C/launch.sh:19`. It accepts a list such as `0-3`; only the detection script
checks for one processor (`runs/shift_explore/detect/code/common.py:370-382`). The reason: XLA's results depend on how
many processors a process sees. Between 1 and 4 processors, commands differed by up to 4.5e-7 and critic values by up
to 2.3e-5 (`runs/shift_explore/detect/code/write_spec.py:261-275`).

**v7 rule.** Kept. `launch_v7.sh` refuses unless the processor argument is one integer, and the wrapper refuses unless
`os.sched_getaffinity(0)` has exactly one element. The runner gives distinct processors to running phases.

**Why.** It is the same rule, now enforced for every phase. Inside WSL, Windows decides which physical core runs each
virtual processor and can move it during a run, so placement on performance or efficiency cores can be neither
observed nor controlled. Both core types have the same instruction set; whether the core type can change results is
unknown.

**Tested by:** `test_v7.py` (a list, a range or two processors refuses), and **P3, a repeatability test under load**:
a fixed 1,000-step `learners.ActorOnly` run, three times with the machine idle and three times with the other lanes
busy, on two different lane processors. All runs must give identical bytes. Otherwise NOT READY, which ends v7 as
specified (README.md, 'If something stops'). The cloud enforces it: `cloud/make_handoff.py import-selftest` refuses a
self-test that is not a READY full one with P3 passed, `HANDOFF.json` pins the self-test it rests on (its import
record's SHA-256, verdict and P3 result), and `write_prereg_v7.py final` refuses without it. P3 shows only that
results do not depend on lane or load.

### C4. Paths, operating system and environment

**v6 rule.** Every pinned component hard-codes `/home/user/bca` (`S2C/stage2_common.py:41`; `S2C/launch.sh:11`,
`:14-15`). The v6 code runs from the scratchpad folder above (`S2C/launch.sh:10`), and its fallback pin derives from
that folder (`S2C/stage2_common.py:42-43`, `:112`; `S2/prereg.json:346-349`). Python is `/tmp/claude-0/bcavenv`
(`S2C/launch.sh:19`). The gate records the dependency folder `/root/.cache/bca-shift/py`, which comes from HOME
(`GATE/__init__.py:39`, `:64-78`). The gate checks Python 3.10.20, eight library versions, and the flags
`JAX_PLATFORMS=cpu` and `OMP_NUM_THREADS=1` with the others unset (`GATE/acceptance.py:114-115`, `:454-470`, `:491-502`).

**v7 rule.** The layout is unchanged; no path becomes configurable. The repository sits at `/home/user/bca` on ext4
inside WSL. The v6 code and `fallback.py` are placed byte for byte at the scratchpad paths on every start, because WSL
empties `/tmp`. `/tmp/claude-0/bcavenv` links to `/root/bcavenv`. Everything runs as root with `HOME=/root`, with uv
0.11.32, the same lock file (`GPU/requirements-lock.txt`; v7's copy is the same bytes) and the dependency folder
installed with uv.

**Exact environment.** `launch_v7.sh` changes to `/home/user/bca` and then starts the wrapper through `env -i` with
exactly eight variables: `PATH`, `HOME=/root`, `LANG=C.UTF-8`, `LC_ALL=C.UTF-8` (as `GPU/lib/common.sh:11`) and
`launch.sh`'s four (`PYTHONPATH=/home/user/bca`, `PYTHONDONTWRITEBYTECODE=1`, `JAX_PLATFORMS=cpu`, `OMP_NUM_THREADS=1`).
`env -i` is needed because `/bin/sh` is dash, which exports `PWD` and `OLDPWD` after a `cd`. The wrapper refuses any
other variable (including `OPENBLAS_*`, `NPY_*`, `XLA_*`, `CUDA_*`, `LD_*`, `BCA_SHIFT_DEPS`) and requires the working
folder `/home/user/bca`.

Two changes from the `GPU/` README:
- **WSL Ubuntu 24.04** instead of 22.04 (the user's decision, Q2). numpy's float64 `sin` and `cos`, which the start
  states use (`GATE/thruster_chain.py:308-315`), come from glibc's math library. The cloud runs Ubuntu 24.04.5 with
  glibc 2.39 (`GPU/env_record.json:920`). glibc 2.39 and 2.40 gave identical results on 1,000,000 values
  (`PROBE/trig_239.npz` against `PROBE/trig_240.npz`). The gate's exact anchor and log checks decide in the self-test.
- **`OPENBLAS_CORETYPE` unset.** No kernel the laptop can run reproduces the cloud's bits, and C1 makes the kernel
  irrelevant. The kernel chosen is recorded (as `GPU/lib/envfacts.py` does).

**Why.** The pinned code, the v6 pins and the gate's pre-registration hold these path strings and compare them.

**Enforced by:** `V7/laptop/restore_v7.sh` (placement, links, refuse-to-overwrite) and the wrapper's environment
check. The gate's own check covers the versions, the flags and the dependency folder.

**Tested by:** the self-test, and `test_v7.py` through the **real `launch_v7.sh`** (a stray variable refuses; the
eight pass), not only with synthetic environments.

### C5. Interrupted phases, unfinished phases and partial files

**v6 rule.** Outputs are written once (`S2C/stage2_common.py:244-255`, `:273-282`). Several phases write files before
their final record, and some write several records in sequence. A killed phase leaves files that block its own rerun.
The `GPU/` runner calls this a failure and stops (`GPU/lib/runner.py:276-280`), and its `resolve()` calls `read_json`,
whose sidecar refusal (`SystemExit`) would stop the runner itself (`GPU/lib/runner.py:94-101`, `:243-251`;
`S2C/stage2_common.py:258-270`). v6 has no written rule for interruptions, although the cloud relaunched killed phases.

**The phase table (pinned in `prereg_v7.json`).** Each phase's files, in the order written, under `OUT/`. Every JSON
has its `.sha256` sidecar and counts only if the sidecar matches.

| Phase | Files, in order | Final record | Failed on record |
|---|---|---|---|
| fresh_data K | `fresh/stall/seedK/data.json`, `fresh/no_stall/seedK/data.json` (`S2C/run_stage2.py:175-179`) | the no-stall `data.json` | a stalling `data.json` with passed=false |
| fresh_learners K | `fresh/{stall,no_stall}/seedK/<name>.msgpack` for the 9 names of `S2C/fresh.py:180-190`, then `fresh/stall/seedK/learners.json` | `learners.json` | `learners.json` with failed=true |
| detect_pairs K W | `detect_fresh/seedK/W/pairs.npz`, `pairs.json` | `pairs.json` | none |
| detect_labels K W | `detect_fresh/seedK/W/labels.npz`, `labels.json` | `labels.json` | none |
| detect_standins K W | `detect_fresh/seedK/W/scores/<name>.npz`, `<name>.json`, for the five stand-ins | all five `<name>.json` | none |
| detect_fit K W C | `detect/uniform/W/C/seedK/bank_scores.npz`, `band.msgpack`, `band.json`, `tau.json` (`S2C/band.py:438-448`; `S2C/run_stage2.py:322-329`) | `tau.json` | `band.json` with failed=true (no `tau.json` follows) |
| detect_fit_weighted K W | `detect/weighted/W/fixed/seedK/band.msgpack`, `band.json`, `tau.json` | `tau.json` | none |
| detect_score K W C V | `detect/V/W/C/seedK/scores/*`, `scores_record.json` | `scores_record.json` | none |
| detect_marks G V | `detect/marks_G_V.json` | itself | none |
| policy_train K | `policy/seedK/actors.msgpack`, `lock.json`; or `stop.json` | `lock.json` or `stop.json` | `stop.json` |
| policy_eval K | `policy/seedK/values.json` | itself | none |
| policy_no_stall_train K | `policy/seedK/no_stall/actors.msgpack`, `lock.json`; or `stop.json` | `lock.json` or `stop.json` | `stop.json` |
| policy_no_stall_eval K | `policy/seedK/no_stall/values.json` | itself | none |
| deploy K W | `cache/scores_W_seedK_*.npz` (`scratchpad/ood/fallback/fallback.py:132-145`), `deploy/W/seedK/values.json` (`S2C/deploy_check.py:171`, `:201`) | `values.json` | none |
| host_train K W | `host/W/seedK/actors.msgpack`, `lock.json` (`S2C/host.py:418-419`) | `lock.json` | none |
| host_eval K W | `host/W/seedK/values.json` | itself | none |
| pooled G | `pooled_G.json` | itself | none |

A phase is **done** when its final record or a failed-on-record file is present and verified, whatever its exit code
(a recorded stop exits 2, `S2C/policy.py:297`). A stalling `data.json` with passed=false marks the seed failed on
record: the seed's later phases are skipped, as v6's `fresh.recorded_failure` does (`S2C/fresh.py:268-281`).

**v7 rule (declared before any laptop result).** A phase attempt is **interrupted** if it is not done, no process of
the phase is running, and it ended in one of two ways:
- *machine-wide*: the machine or WSL restarted while it ran (the WSL boot id, `/proc/sys/kernel/random/boot_id`, which
  the ledger records at each attempt's start, has changed: a Windows Update restart, `wsl --shutdown`, power loss); or
  the runner stopped it (the runner records every signal it sends, for example after a second Ctrl-C); or it ended by
  SIGHUP, SIGINT or SIGTERM, or with no exit status recorded, on the same boot (a closed terminal or tmux session);
- *phase-specific*: on the same boot, it was killed by SIGKILL, SIGSEGV, SIGBUS or SIGABRT that the runner did not
  send (the out-of-memory killer sends SIGKILL; `launch_v7.sh` records the real exit status, which v6's `launch.sh`
  turns into 1, `S2C/launch.sh:19-20`), or its own lines of the log end in a memory error (`MemoryError`,
  `RESOURCE_EXHAUSTED` or "out of memory"). Every attempt of a phase appends to one log, so an attempt is classified
  only on the lines from the start line that names its write log to the next start line; an exit 97 is a v7
  refusal whatever those lines say.

For an interrupted phase, the runner moves **every existing file in that phase's row of the table** (including a JSON
whose sidecar is missing or wrong), never deleting it, to `R7/interrupted/<phase>/<UTC time>/` at the same relative
paths, records their SHA-256 and the kind of interruption in `R7/interrupted/<phase>/<UTC time>.json`, and reruns the
phase from the start.

- A machine-wide interruption is not counted. The phase reruns automatically, with no limit.
- After the first and second phase-specific interruption, the phase reruns automatically as usual.
- After the third, the runner reruns it once more **alone**: when no other runnable phase remains, it starts this
  phase on one lane with all the memory, and starts nothing else until it ends.
- If that attempt is also interrupted phase-specifically, the phase is **unfinished**. The runner prints
  `UNFINISHED: <phase>`, keeps running every phase that does not need it, and then prints
  `WAITING FOR RETRY: <phases>`.

**An unfinished phase is retried until it is done (declared now).** Whether it is retried is not a choice: the user's
command `sh .../run_v7.sh --retry <phase>` sets only *when* the next attempt starts (for example after closing other
programs or restarting WSL). Between attempts the user may free memory or restart WSL, but changes no file, setting
or code. Each command allows exactly one attempt, run alone as above; it refuses unless its standard input is
a terminal and the phase is unfinished, and writes, once, `R7/retry/<phase>/<n>.json` (+ sidecar: UTC time, attempt
number, the command line). If the runner is running, it picks the request up within a minute; otherwise the command
starts the runner. A retry interrupted machine-wide reruns automatically; one interrupted phase-specifically leaves
the phase unfinished again (its files moved as above). There is no limit on the number of retries. Every retry is
listed in the ledger and the provenance file. Results gathered while a phase is unfinished (`make_results_v7.py`, C10)
name it in `BUNDLE.json`, in the provenance file and in the runner's last line, and no pooled look that needs it exists.

Exit 97 (a v7 refusal, C1, C7) halts the run. Any other non-zero exit, or exit 0 without the phase being done, halts
new phases and names the log, as `GPU/` does. Recorded failures and inherited files are never moved. The runner never
exits on a damaged record: it reads records through a wrapper that turns a sidecar refusal into "partial" and reports
it. Files are found only through the table, never by folder differences or times. A phase that is done on file and
not inherited must have been made by a ledgered laptop attempt classified done that wrote each file that makes it
done, with its current SHA-256 (and, with a re-make entry, its lock must record the re-made actors); otherwise the run
halts, at start-up and whenever such a phase is found or ends. Every attempt left without a record (its launch shell,
or the self-test, did not survive) has its record written once no process of it runs: by the runner at start-up,
done phases included, by the self-test (under its lock) for probe attempts, and before results or run files are
gathered.

**Why.** Sleep, Windows Update, `wsl --shutdown`, power loss and the out-of-memory killer are likely over a run of
several hours. Every random stream is seeded, so a rerun on the same machine with one processor gives the same numbers.
A restart of the machine says nothing about the phase, so it is never counted; one restart can interrupt 12-16 phases
at once, and counting it would freeze them all. Repeated phase-specific interruptions suggest a cause in the phase
(usually memory), so the phase is given the whole machine before anyone is asked. Declaring that every unfinished
phase is retried until done means no result depends on a person's choice made after reading a phase's log.

**Tested by:** `test_v7.py`, with dummy phases killed by SIGKILL at each boundary: between a model file and its
record; between a JSON and its sidecar; between the stalling and the no-stall `data.json` (passed and failed); between
a non-failed `band.json` and `tau.json`; after a deploy cache file. Each case is moved and rerun, or skipped as failed
on record. A recorded `stop.json` and a failed `band.json` are never moved. A torn sidecar never stops the runner.
Classification: a changed boot id, a runner-sent signal, SIGHUP/SIGINT/SIGTERM and a missing exit status are
machine-wide and rerun any number of times without counting; SIGKILL on the same boot and a log ending in a memory
error count. After a third counted interruption the next attempt waits until nothing else is runnable and then runs
alone; after a fourth, independent phases still start and dependent ones do not; no further attempt starts without a
request; `--retry` from a non-terminal, or for a phase that is not unfinished, refuses; a request written while the
runner runs is picked up, recorded, and allows exactly one attempt, run alone; `BUNDLE.json` names an unfinished phase.

### C6. Handoff from the cloud, which v6 outputs v7 accepts, and the way back

**v6 situation.** See the appendix (what was known when v7 was written): the driver `explore_policy.sh` and the watchdog Routine `trig_01YCP5oPNVkUPRgta25ag1y4`.
The run-files snapshot is commit `901c877` on `claude/stage2-run-files`: 2,470 files plus `RUNFILES_SHA256SUMS`,
779.6 MB, with no host folder and no pooled file.

**Until the handoff (the user's decision, Q5, option c).** The cloud keeps running v6 (at most 2 processes, with its
driver and watchdog) until the handoff. Every phase it finishes before the handoff is inherited and needs no probe. It runs only
while its session is active.

**v7 rule.**
1. **Stop the cloud (cloud task, at the user's go-ahead, step 9).** In this order: delete the watchdog Routine
   `trig_01YCP5oPNVkUPRgta25ag1y4` and any other Routine that could start a Stage 2 phase, and confirm with the
   Routines list (the PR check-in `trig_0196VjNUjbaeZy2u8XXU7MBW` and any re-armed successor start no phase; they are
   recorded, and deleted if the user asks); stop the driver's `sh` processes; stop running phases at once (they rerun
   on the laptop from the start), unless the user chose "let them finish"; confirm that no `run_stage2.py` remains.
   The user can check the Routines in claude.ai's Routines list.

   **When the stop rule is chosen (declared now).** Which seed 2-4 phases are inherited, and so which can use a
   re-made reference, depends on what the cloud finishes before it stops. So the choice between "stop at once" (the
   default) and "let them finish" is made **before the full self-test (step 8) runs**, in a message to the cloud
   session; with no message, the default holds. It cannot be changed after any P1 or P2 record exists. The go-ahead
   itself (step 9) is given at the first convenient time after the self-test, not timed to the P1/P2 results.
2. **Final snapshot.** Push a "part 4" commit to `claude/stage2-run-files` with the new files and a new
   `RUNFILES_SHA256SUMS`, logs included.
3. **`V7/handoff/HANDOFF.json`** (+ sidecar), written once by the cloud. It holds: the snapshot commit and the SHA-256
   of its sums file; the stop time; the stop rule used ("stop at once" or "let them finish"), the UTC time it was
   chosen (or "default") and the write times of the nine P1/P2 records, so anyone can see the choice came first; the
   go-ahead's time; each Routine's id and final state; the processes stopped, the check that no `run_stage2.py` and
   no `explore_policy.sh` process is left, and the last log lines; for every seed and phase, whether it is inherited,
   partial or not started; the list of partial files; the cloud's own time of the first self-test paste (the stop
   rule must be chosen before it too, and its time is given as `YYYY-MM-DDTHH:MM:SSZ` or "default"; against the
   records' write times, which are the laptop's clock, the comparison allows the 2 minutes to which the full self-test
   checks that clock before any probe runs); every self-test pasted to the cloud, accepted or refused; the full
   self-test it rests on, which must be the most recent full self-test pasted, READY, with P3 passed, run under the
   committed draft (no pasted full self-test may have found different results in P3: that ends v7; a P3 run that did
   not finish is only run again); and the draft `prereg_v7.json` it
   goes with (the final refuses another draft, and once `HANDOFF.json` exists the draft writer refuses to write any
   other draft, because `HANDOFF.json` is written once).
4. **Accepted v6 outputs (a rule, not a choice).** A phase is **inherited** when its final record or failed-on-record
   file is in the final snapshot with a matching sidecar; all its table files are then inherited as is. Files under
   `OUT/` that belong to no phase (`prereg.json`, `code_manifest.json`, `approval.json`, `code/`, the history folders,
   `best_standin.json`) are inherited as is. `OUT/logs/` is not inherited. Files of a phase that is not inherited are
   **partial**: they stay in git, in the snapshot, and are not extracted. The gate's files (`runs/shift/`) and the
   room, stand-ins, detection, fallback and deployment folders are read-only inputs, verified against the same sums.
5. **One machine per phase.** On the laptop, by code: the runner refuses any inherited phase (the v6 code also refuses
   to overwrite), and refuses every phase until `HANDOFF.json` is present, matches the final `prereg_v7.json`, and the
   restored files match it. In the cloud, only by procedure: step 1, recorded in `HANDOFF.json`, and the return check
   (C10), which compares the cloud's own copy of `OUT/` with the snapshot if that copy still exists.
6. **The way back (declared now).** If v7 is not approved, or the laptop refuses before its first phase starts, the
   cloud writes `V7/handoff/WITHDRAWN.json` (a new record; `HANDOFF.json` is never edited) and may resume v6 with
   `explore_policy.sh`, at most 2 processes, with a new watchdog only if the user asks. The v7 runner refuses once
   `WITHDRAWN.json` exists. Once any v7 phase has started on the laptop (its first ledger file exists), the cloud never
   resumes Stage 2; any further route is the user's decision, as a new version. `WITHDRAWN.json` records the user's
   own confirmation that no v7 phase has started, quoted with the time of their message, and, when the user pastes
   it, the listing of `R7/ledger/`; a listing that names any folder other than a probe's is refused.

**Restore.** `restore_v7.sh` fetches only the run-files branch (in the shell, where git can ask for the user's
sign-in), checks that the pinned commit is on it (its tip, or an ancestor of the tip once the branch has moved on), and
extracts with `git archive <commit> runs | tar -x --skip-old-files`, leaving out `OUT/logs/` and the partial files.
It never touches the main branch's index or tracked files (`runs/` is ignored, `.gitignore:6`). It then checks every
extracted file against `RUNFILES_SHA256SUMS`, minus `OUT/logs/`. Before the handoff it restores `901c877` (which holds
the gate's files the probes need); after it, the final snapshot, adding only new files.

**Why.** The two machines must never both run a write-once phase. Running the v6 code under the v6 approval means
inherited files are read natively: they carry v6's fingerprint, which the v6 code expects.

**Tested by:** `test_v7.py` (missing handoff, changed inherited file, phase with an inherited output, partial file not
extracted, `WITHDRAWN.json` refuses, extraction leaves `git status --porcelain --untracked-files=no` empty), and the
self-test's restore check.

### C7. Seeds 0-4: continuing across two machines

Remaining phases: the policy test and deployment for seeds 2-4; `host_train` and `host_eval` in both worlds for seeds
2-4; then the exploratory pooled look. Phases the cloud finishes before the handoff are inherited instead (Q5). No
detection phase and no `select_standin` runs again; the runner refuses them. **Nothing is held** (the user's decision,
Q4): once v7 has started, every remaining phase runs, and the exploratory pooled look is written as v6 says.

Where a laptop computation meets a cloud-made file or number:

| Laptop computation and cloud-made input | Where | Stops a seed? | v7 handling |
|---|---|---|---|
| Gate anchors (64) and gate log hashes | `GATE/acceptance.py:959-966`, `:715-716` | refuses the phase | C1: exact |
| **Check (b)**: BCA program at 0 against check 2's "none" actor, 1e-5 | `S2C/policy.py:366`, `:387-392`, `:522`, `:545-548` | **yes** (`stop.json`, never replaced) | re-make rule and check (b) forecast below |
| Policy matching and the stand-in: strengths of laptop arms measured from "none"; the stand-in's surfaces and classifier fitted with it | `S2C/policy.py:245-250`, `:399-402`, `:411`; `standins.py:840` (in `runs/shift_explore/standins/code/`) | no | re-make rule (same actor as check (b)) |
| Weighted band re-fitted on the laptop for the policy arms, while detection and deployment use the cloud's weighted band | `S2C/policy.py:374-377`, `:529-532`; `S2C/band.py:506-511`, `:537-546`; `S2C/run_stage2.py:541-546` | no | declared; R1 compares (information) |
| Stalling-world host: strength measured from the gate's plain TD3+BC; the matcher assumes coefficient 0 has strength 0 | `S2C/host.py:354`, `:368`, `:399` | no | re-make rule below |
| No-stall host: hinge at the coefficient locked by the stalling host (cloud-made when that host is inherited) | `S2C/run_stage2.py:567-572` | no | declared; provenance names the source |
| Full host's coefficient-0 check | `S2C/host.py:363-367` | no; both sides run in one process | none needed |
| Deployment: laptop width U against the cloud's tau and band, on the gate's saved "none" and plain TD3+BC | `S2C/run_stage2.py:541-557` | no | v6 as written; never redirected |
| Evaluations: points use the recorded anchors | `GATE/acceptance.py:995-999` | no | nothing recomputed |
| numpy's own dispatch: float64 `exp`, `log`, `expm1`, `log1p`, `arctan`, `power` differ in last bits between AVX2 and AVX-512 (e.g. `exp` on 45,972 of 1,000,000 values, largest 2.9e-11); `tanh`, `sin`, `cos`, `sqrt` identical | `PROBE/uf_avx2.npz` against `PROBE/uf_default.npz` | no | information. The world uses only the identical four (`GATE/recorder.py:164`, `:216`; `GATE/thruster_chain.py:283`, `:308-315`). BCA's numpy arithmetic on seeds 0-4 can differ in last bits |

#### The re-make rule and the check (b) forecast for seeds 2-4 (declared now, before any laptop outcome)

**The references.** Three of the gate's saved files are references that laptop-trained actors are measured against
(paths under `runs/shift/68d9e8779e148b7b/`):

| Name | What | File | Loaded through | Read by | Lock name it becomes |
|---|---|---|---|---|---|
| R-stall | check 2's stalling-world actors "gapped" and "full" | `harm_return/base/alpha5/seed<s>/actors.msgpack` | `ROOM/harness.py:248-255`, then `S2C/fresh.py:60-69`, `:74-75` | policy_train: check (b), every strength match, the stand-in, the refit for the reported shape, the scored arms (`S2C/policy.py:366-424`) | "gapped" becomes `none` and "full" becomes `full_none` in `OUT/policy/seed<s>/lock.json` (`S2C/policy.py:366`, `:424`) |
| R-no_stall | check 2's no-stall actor "no_stall_gapped" | the same file | the same, `S2C/fresh.py:68-69` | policy_no_stall_train: check (b), the stand-in, the scored "none" (`S2C/policy.py:522-562`) | "no_stall_gapped" becomes `none` in `OUT/policy/seed<s>/no_stall/lock.json` (`:522`, `:562`) |
| R-host | the gate's plain TD3+BC run, final snapshot | `hosts/base/stall/gap/alpha5/plain/seed<s>/snapshot_100.msgpack` | `ROOM/harness.py:267-271`, which calls `GATE/acceptance.py:1057-1080`; then `S2C/fresh.py:77-80` | host_train, stalling world only: the target strength, every hinge strength and the match, the scored arm "plain" (`S2C/host.py:354-399`) | the state's actor becomes `plain` in `OUT/host/stall/seed<s>/lock.json` (`S2C/host.py:354`, `:359`) |

R-host's actor is the same actor as check 2's "td3bc_gapped" in R-stall's file (`GATE/acceptance.py:1465`, `:1487`; a
review's cloud run found the same SHA-256 for seed 2). The deployment check reads that key as "td3bc"
(`S2C/fresh.py:71-72`, `S2C/run_stage2.py:556`). It is never replaced.

**Step 1: the probes (in the step-8 self-test, no BCA).** For each seed s in {2, 3, 4}: one P1 process per seed and
world (P1a, then P1b, then P1c) and one P2 process per seed, each on one processor, with the constants file, the phase
environment (C3, C4) and the ledger hook (C10). Every gate input is read through a read-only path: `load_seed`;
`prepared("gap", learn)`; the critics through `get_critic(..., "stall_aware", train=False)`; the clone through the
read-only path of `ROOM/harness.py:240-246` (`A.finished` on the clone's `record.json`, then `L.load_tree`; the probe
refuses when the record does not verify), never through `get_clone` (`GATE/acceptance.py:1033-1045`), whose other
branch trains a clone and overwrites files in the gate's folder; the gate's saved actors through the readers of
`ROOM/harness.py:248-255` and `get_host(..., train=False)`.
- **P1a stall (the gate's call).** `ao = L.ActorOnly(pg, clone_s, learn, s, alpha=5.0)`,
  `stacked, info = ao.run([L.Arm("gapped", critic=0), L.Arm("full", critic=1), L.Arm("random", critic=2)],
  [critic_gap, critic_full, critic_random], weighted=False)` and `ao.arm_params(stacked, i)`
  (`GATE/acceptance.py:1459-1464`, `:1475-1480`, `:1482-1483`). The three arms run side by side, as in the gate. Writes
  `R7/refs/seed<s>/check2_stall.msgpack`, a dictionary of gapped, full and random, with the gate's `L.save_tree`
  (`GATE/learners.py:277-282`).
- **P1a no-stall.** The same for `:1484` (no_stall_gapped and no_stall_full side by side, from the no-stall clone).
  Writes `R7/refs/seed<s>/check2_no_stall.msgpack`, a dictionary of the two.
- **P1b (the check (b) forecast).** The room harness's own no-BCA W = 0 arm (`ROOM/harness.py:783`,
  `RoomArm("w0_check", "plain", (0.0,), 1.0)`) through `RoomRunner`, with check (b)'s start (the clone), critic,
  batch-row stream and program settings (`S2C/policy.py:371-372`, `:385` stalling; `:527-528`, `:537` no-stall), and
  check (b)'s group size: alone in the stalling world, as check (b) runs w0_check alone (`S2C/policy.py:387`); in the
  no-stall world inside a two-arm group of one kind, with the second arm `RoomArm("pull_2", "plain", (0.0,), 2.0)`, as
  check (b) runs w0_check side by side with bca_k1 (`S2C/policy.py:540`, `:170-173`). This structure (one shared
  critic, arms side by side) is the one BCARunner's program has (`ROOM/harness.py:473-477`; `S2C/policy.py:120-125`).
  Writes no reference.
- **P1c (the forecast for seeds 5-9).** The fresh seeds' actor-only structure on seed s's gate inputs: in the stalling
  world `L.ActorOnly(pg, clone_s, learn, s, alpha=5.0).run([L.Arm("none", critic=0), L.Arm("full_none", critic=1)],
  [critic_gap, critic_full], weighted=False)` (`S2C/fresh.py:245-246`); in the no-stall world one arm,
  `[L.Arm("none_no_stall", critic=0)]` with the no-stall gap critic (`:252-253`). Writes no reference.
- **P2.** `L.cotrain(pg, learn, s, alpha=5.0, table=None, stream=0, out_dir=R7/refs/seed<s>/host_plain,
  name="TD3+BC plain (stall, gap log)")`: the call of `GATE/acceptance.py:1071-1072` for the arm "plain"
  (`HOST_ARMS`, `:1048`), 100,000 updates. The gate's own code writes `snapshot_050.msgpack` and `snapshot_100.msgpack`
  (`GATE/learners.py:568-570`); v7 writes the run's record as `R7/refs/seed<s>/host_plain/record.json`.

Each P1 process writes one record, `R7/selftest/p1_seed<s>_<w>.json`, and each P2 process `p2_seed<s>.json`. Per actor
they hold the parameter SHA-256 (`L.tree_sha256`) of the laptop's copy and of the gate's saved actor, the largest
absolute parameter difference, `differs` (true when the two SHA-256 differ), whether the laptop's run reported a
failure, and each reference file's SHA-256. P1 also holds P1b's and P1c's actor SHA-256 and their largest differences
from the gate's actor, from P1a's actor and from each other. P2 compares the final actor (the only part
`S2C/host.py:359` uses), reports the critic, and records the SHA-256 of check 2's "td3bc_gapped" and whether it equals
the gate's host actor, so the gap between the host test's "plain" and the deployment check's "td3bc" is on record.

**Step 2: which references are re-made (fixed at the handoff, from pinned inputs only).** Entries are keyed by
`(phase, seed, world)`. An entry is made for:
- `(policy_train, s, stall)` when P1a stall `differs` for "gapped" or "full", P1b stall does **not** reproduce the
  gate's "gapped" bit for bit, and policy_train s is not inherited: both "gapped" and "full" then come from
  `check2_stall.msgpack` (one call made them side by side, so they stay a pair);
- `(policy_no_stall_train, s, no_stall)` when P1a no-stall `differs` for "no_stall_gapped", P1b no-stall does not
  reproduce the gate's "no_stall_gapped" bit for bit, and that phase is not inherited;
- `(host_train, s, stall)` when P2 `differs` for the final actor, and that phase is not inherited.

When P1b reproduces the gate's actor bit for bit, the gate's file is kept even if P1a differs: P1b is the closer model
of check (b), so check (b) is expected to be exact with the gate's file. No other entry is allowed: seeds 0-1 and 5-9,
deployment, every evaluation, the no-stall host, detection and the pooled look never read a re-made reference. Where
no entry is made, the phase runs on the laptop unchanged and reads the gate's file. If the laptop's P1a or P2 run
reported a failure (non-finite parameters) for an actor the gate's did not, no entry is made; the phase reads the
gate's actor (v6 as written) and is labelled "re-make failed". The cloud's `write_prereg_v7.py` writes the table, with
each probe record's SHA-256, its `differs` flags, the lock name each replaced key becomes (the references table) and
the forecast gaps (step 2b), into the final `prereg_v7.json`, so the user sees and approves it (C10).

**Step 2b: the check (b) forecast (declared now).** For each seed s in {2, 3, 4} and each world, the self-test
computes two largest absolute parameter differences:
- *F_ref*: P1b's W = 0 actor against the actor check (b) would read on the laptop: the gate's actor when step 2 makes
  no entry, P1a's copy when it makes one. It is computed for every seed and world, since which phases are inherited is
  known only at the handoff.
- *F_fresh*: P1b's W = 0 actor against P1c's "none" actor of the same world: the same comparison for the program
  structures that seeds 5-9 will use.

If any F_ref or F_fresh is above **1e-5** (check (b)'s tolerance `W0_TOL`, `ROOM/harness.py:89`, used at
`S2C/policy.py:59`, `:389`, `:546`), the self-test prints **NOT READY (P1b: check (b) forecast)**. Nothing is approved
or started, the cloud keeps running v6 (Q5), and the user decides with the cloud on a new version. Gaps above 0 and at
most 1e-5 are reported and stop nothing. Why: such a gap is between two laptop programs, so no re-made reference can
close it; a check (b) stop is permanent, and on seeds 5-9 it would fail the whole confirmatory group
(`S2C/policy.py:576-577`, `:661`, `:668-670`) for a reason that has nothing to do with BCA. The user may instead choose
to proceed whatever P1b shows; that choice must be in the draft `prereg_v7.json` before any P1 record exists (section
7, Q7). *What the forecast does not prove:* check (b) runs the policy test's own BCA program (`S2C/policy.py:96`,
`:120-125`; BCA's penalty at coefficient 0 and, for the weighted version, a band refit before every chunk, `:136-144`),
which v7 does not run before approval. A forecast within 1e-5 makes a check (b) stop on the laptop unlikely, not
impossible.

**Step 3: the redirect (at run time; no pinned file is edited).** `V7/code/v7_refs.py`, step 5 of C1:
1. It reads this process's `--phase`, `--seed` and `--world` (the arguments `launch_v7.sh` passes on unchanged),
   parsing `--world` as `run_stage2.py` does, with its default `"stall"` (`S2C/run_stage2.py:655`). A policy phase's
   world is fixed by its name (`policy_train`: stall; `policy_no_stall_train`: no_stall), because `run_stage2.py`
   ignores `--world` for them (`S2C/run_stage2.py:685-686`); `host_train`'s world is its `--world`. It looks up
   `(phase, seed, world)` in the verified final `prereg_v7.json`. No entry (including `host_train --world no_stall`
   for a seed whose stalling host has one): it does nothing and imports nothing, and the process is the plain v7 one,
   labelled "gate's reference", or "own run" for the no-stall host. An entry outside the three kinds or outside seeds
   2-4 refuses (exit 97).
2. With an entry, before the hand-over and without JAX: it checks each reference file's SHA-256 against the pin, and
   imports v6's `stage2_common` from the folder `launch.sh` runs from. That module is import-light (no JAX, no gate
   import, `S2C/stage2_common.py:5-6`); `run_stage2.py`'s own `import stage2_common` (`S2C/run_stage2.py:49-50`) then
   finds the same module object (the folder v7_refs added to `sys.path` is removed again; `run_stage2.py` adds it
   itself, `S2C/run_stage2.py:49`). It replaces the module attribute `stage2_common.load_room_harness` by a wrapper that
   calls the original unchanged (which verifies and executes `ROOM/harness.py`'s pinned bytes and caches the module,
   `S2C/stage2_common.py:747-770`, `:738-744`) and then, once, replaces methods of that module's class `SeedContext`,
   the base class of every `GateContext` (`S2C/fresh.py:52-55`). Every v6 module reaches the harness through this
   function (`S2C/policy.py:50`, `S2C/fresh.py:52`, `S2C/host.py:295`, `S2C/deploy_check.py:51`).
3. For a policy entry, `SeedContext.h2` (`ROOM/harness.py:248-255`) is replaced by a function that calls the original
   first (so the gate's `values.json` and `actors.msgpack` are read and checked exactly as in v6), requires
   `self.rep` to equal the entry's seed, and then replaces in the returned dictionary exactly the entry's keys
   ("gapped" and "full", or "no_stall_gapped"). The reference file holds several actors, so it is loaded with a
   dictionary template, as at `ROOM/harness.py:253-254`: `L.load_tree(<reference file>, {k: L.actor_template() for k
   in <the names P1a wrote, pinned>}, <pinned file SHA-256>)`. (A single-actor template raises ValueError on such a
   file; checked on a real file in P1a's format.) Each of the entry's keys is then picked, and its `L.tree_sha256` must
   equal the pinned per-actor value. The record and every other key (including "td3bc_gapped", which deployment reads)
   stay the gate's. `GateContext` reaches it through `self.h2()` (`S2C/fresh.py:62`).
4. For a host entry, `SeedContext.host` (`ROOM/harness.py:267-271`) is replaced by a function that calls the original
   first (so `snapshot_100.msgpack` is read and checked as in v6, `GATE/acceptance.py:1078-1079`), requires the seed,
   and returns `(state, record)` with the state loaded by the gate's call at `GATE/acceptance.py:1079` on the
   reference file: `L.load_tree(R7/refs/seed<s>/host_plain/snapshot_100.msgpack, L.agent_template(self.learn),
   <pinned SHA-256>)`; its actor's `L.tree_sha256` must equal the pinned value. `GateContext` reaches it through
   `H.SeedContext.host(self)` (`S2C/fresh.py:80`), which only the stalling-world host calls (`S2C/host.py:353-357`).
5. Only the method the entry needs is replaced. Any failure inside a replacement (wrong seed, wrong SHA-256, a missing
   key) writes one ledger line and calls `os._exit(97)`; it never raises, for the reason in C1 step 4.

**Labels.** Each ledger attempt records `reference`: "gate's saved file", "re-made on the laptop" or "own run" (the
no-stall host), with the entry, the reference files and their SHA-256, and, after exit, whether the lock (when one is
written) records, under the lock name of the references table, the re-made actor's SHA-256 for each replaced actor (it
must; otherwise the run halts). Each probe process is ledgered like a phase, so every reference file it wrote is listed
with its SHA-256. The provenance file labels each seed and component (policy stalling world, policy no-stall world,
stalling host, deployment per world) "inherited (cloud, v6)", "laptop, gate's reference (v6 as written)", "laptop,
gate's reference kept (P1b reproduced it)", "laptop, reference re-made on the laptop" or "laptop, re-make failed",
with the probe record's largest parameter difference and the forecast gaps. A check (b) stop in a "re-make failed"
seed and world is labelled "machine-caused (re-make failed)". For each seed and world with an entry, it sets side by
side the gate's recorded check-2 points for the replaced actor (the gate's `values.json` in R-stall's folder, which also
holds "td3bc_gapped") and the points `policy_eval` or `host_eval` gives the re-made actor, for each situation both
report; both values already exist, and nothing is recomputed. For a seed with a host entry, it states that the host
test's "plain" (laptop) and the deployment check's "td3bc" (the gate's) are different actors, with P2's difference, and
that the lock's fixed text `records["plain"]["source"]` ("the gate's saved run", `S2C/host.py:355`) does not describe
that seed.

**What the re-make changes in the comparisons** (only in phases with an entry):
- *Check (b)* compares the laptop's BCA program at 0 with the laptop's own check-2 actor: two laptop computations, as
  for seeds 5-9. The tolerance and the stop rule are unchanged; a stop stays on record (v6), with the P1 results and
  the forecast beside it in the provenance file.
- *Matching*: bca_k1's target strength, every control's and the stand-in's strengths, and the stand-in's surfaces and
  classifier are measured from the laptop's "none"; the matched coefficients follow from them. So M1, M2 and M4 compare
  bca_k1 with controls matched against the laptop's "none".
- *Scores read directly from a replaced actor*: `policy_eval` evaluates the lock's "none" and "full_none", now the
  laptop's copies. In the stalling world they feed M3 (bca_k1 minus none, in recorded conditions, big waves and big
  kicks, `S2C/policy.py:650-658`) and the claim numbers harm (full_none minus none, `:677`), X (`:678`), Y (through
  Yd, none minus clone, `:680`, `:684`) and Z (`:681`). In the no-stall world they feed W (`:682`), V (`:683`) and every
  reported no-stall cost (`:688`). These numbers then compare actors from one machine. In the stalling host, "plain" is
  scored by `host_eval` and reported in the pooled look's host block (`S2C/run_stage2.py:641`).
- *The weighted-band refit*: `refit(none)` (`S2C/policy.py:375-377`) fits the classifier with the laptop's "none". It
  feeds only the reported penalty shape (`:382`). The bca_w arms refit with their own actors before every chunk
  (`:136-144`) and do not use "none". The cloud's weighted band, fitted from the gate's "none" (`S2C/band.py:538`),
  still serves detection and deployment.
- *Stalling host*: the target, the hinge strengths, the match and the scored "plain" arm use the laptop's plain
  TD3+BC. The matcher's "coefficient 0 has strength 0" (`S2C/host.py:399`) becomes a same-machine statement, which the
  coefficient-0 check (`:363-367`, laptop against laptop) supports.
- *Deployment*: unchanged. It reads the gate's saved "none" and plain TD3+BC ("td3bc_gapped"; `S2C/run_stage2.py:546`,
  `:556`), with the cloud-made bands and tau made from them, and compares rules on one actor at a time. So for a seed
  and world with a policy entry, the policy test's "none" and the deployment check's "none" are two different actors;
  and for a seed with a host entry, the host test's "plain" and the deployment check's "td3bc" are two different
  actors. The probe records say how far apart. M5 is read on the gate's actor, as v6 says.
- *No-stall host*: unaffected; it trains its own plain TD3+BC in the same process (`S2C/host.py:356-357`) and takes
  the coefficient from the stalling lock.

**Method or operation.** v6 names, for seeds 0-4, "check 2's saved gapped-critic actor" and "the gate's saved run"
(`S2/prereg.json:383`, `:272`). A re-made copy departs from that wording. It is declared an operational departure: the
computation is the gate's exact call on the gate's inputs, with every setting unchanged; its result is a different
numerical realization of that call, whose size the probe records. (The actor comes out of 20,000 actor-only steps or
100,000 TD3+BC updates, so last-bit differences in the arithmetic can grow into larger parameter differences; P1 and
P2 record the largest.) The rule picks no seed or world by any outcome of BCA: in both branches the phase reads an
actor-only result of the gate's call made before approval (when P1a is bit-identical, the gate's file holds the
laptop's bits), and no v6 or gate check compares a lock's actors with the gate's (below). v6 itself makes these
references on the running machine for seeds 5-9 (`S2C/fresh.py:180-190`, `:245-258`; `S2/prereg.json:260`). (Version 1
of the specification called a laptop-made reference a method change. The user chose to use one rather than hold seeds, Q4;
this paragraph gives the reason it is classed as operational, and the user approves it as such.)

**Q4 feasibility (checked against the code).**
- *A redirect needs no pinned edit.* The loaders are methods of `SeedContext`, a class in the module that v6 builds in
  memory from `ROOM/harness.py`'s verified bytes and caches (`S2C/stage2_common.py:738-744`, `:747-770`).
  `GateContext` subclasses it at run time and reaches the loaders by attribute lookup at call time
  (`S2C/fresh.py:55`, `:62`, `:80`). Replacing two class attributes in memory changes what those calls return; the
  byte checks read files, not objects. The stand-ins module builds a second copy of the harness for itself
  (`standins.py:81-112`), but Stage 2 calls none of its loaders: `build_shared` reads only `ctx.sd`
  (`standins.py:833`). A review run demonstrated the wrapper in memory on seed 2 in the cloud
  (`scratchpad/feas_review/redirect_demo.py`): `ctx.none()` and `ctx.full_none()` returned the replacements, and
  `ctx.td3bc()` still returned the gate's host actor.
- *Nothing hashes the redirected references in a way the redirect trips.* The gate's pre-registration hashes code,
  not run outputs (`GATE/acceptance.py:936-950`, `:972-979`). The loaders' own checks of the gate's files
  (`ROOM/harness.py:250-254`; `GATE/acceptance.py:1078-1079`) still run, on the untouched files, because each
  replacement calls the original first. Stage 2 never compares a lock's actor with the gate's: `tree_sha256` serves only
  each lock's self-consistency (`S2C/policy.py:303`, `:317`; `S2C/host.py:424`, `:443`), and the one cross check,
  `room_lock_target` (`standins.py:368-383`), is not among the stand-in functions Stage 2 calls (`S2C/policy.py:243-250`,
  `:553-555`). `R7/refs/` lies outside `OUT/` and `runs/shift/`, so the run-files sums are unaffected.
- *Two details, fixed in version 3:* the reference files hold several actors, so step 3 loads them with a dictionary
  template; and a host entry belongs to the stalling world only, so entries are keyed by world.
- *One fixed text* becomes inaccurate for a re-made host (`S2C/host.py:355`); the provenance file corrects it.
- *Conclusion:* feasible as designed. No blocker, and no alternative is needed.

**Why bit-identity decides, not 1e-5.** The cloud measured check (b) at exactly 0.0. A nonzero machine gap would use
part of check (b)'s budget, and a stop it caused would be permanent. Re-making only where needed leaves every
bit-identical seed and world exactly as v6.

An inherited lock may be evaluated on the laptop (`policy_eval`, `policy_no_stall_eval`, `host_eval`): that only
evaluates the cloud's actors.

**Probe records (declared now).** P1 and P2 run once, in the step-8 self-test, for all of seeds 2-4 (which phases the
cloud will finish is known only at the handoff), before the handoff and the approval. Every record and reference file
is written once. An interrupted probe (no record yet) has its files moved as in C5 and reruns; a written record is
never replaced, and no later self-test reruns it. Each record holds the fingerprints of its inputs, none of which the
handoff can change: the constants file, the probe script, the gate's pre-registration (`68d9e877...`), the pinned v6
code manifest, and the re-make and forecast block of the committed draft `prereg_v7.json` (this rule and the answer to
Q7). So once any record exists, on the laptop or imported by the cloud, that block (Q7's answer included) and the probe
script can no longer change: the draft writer refuses, and the self-test reports a record bound to another draft as
NOT READY. The self-test prints each record's SHA-256, its `differs` flags and the forecast gaps; the cloud writes them, with
the table of step 2, into the final `prereg_v7.json`, which cannot be written without all nine records. The short
self-test and the runner (before anything starts) check every record and reference file against those pins, and the
wrapper checks the reference file of a re-made entry it reads; a mismatch halts the run, because it means `R7/` was
damaged.

**A damaged reference (declared now).** It can be replaced only by the same call reproducing the pinned SHA-256.
`selftest_v7.sh --remake <probe>` moves the damaged file (never deleting it) to `R7/interrupted/refs/<UTC time>/` at its
relative path, recording its SHA-256; reruns the probe's call into the temporary folder `R7/refs_remake/<UTC time>/`;
and moves the new file to the original path (one rename) only if its SHA-256 equals the pin. Otherwise the run stays
halted. The redirect only ever reads the original path, after checking the pin. There is never a second, different
reference, and the probe record is not replaced (the remake is ledgered). A damaged probe record is put back by the
same command from the cloud's byte copy (`V7/handoff/probes/selftest/<probe>.json`), only when that copy's SHA-256
equals the final pin; the damaged file is moved to `R7/interrupted/records/<UTC time>/`. Before the cloud holds a copy
and the final pins it, a damaged probe record cannot be repaired under v7.

**R1 (information only, after the v7 import).** For seeds 2-4 and both worlds, recompute the weighted band with v6's
pinned band code from the gate's "none" actor and, where re-made, from the laptop's copy, and compare each with the
cloud's `OUT/detect/weighted/<w>/fixed/seed<s>/band.msgpack`. Written to `R7/info/`; routes nothing. (The policy lock
does not keep this refit's diagnostics: `S2C/policy.py:377` discards them.)

**Order (declared).** A world's deployment check runs only when that world's policy phases are done or skipped (the
no-stall arm is skipped after a stalling stop, as `GPU/lib/runner.py:162-176`). The host phases follow v6's order:
stalling world, then the no-stall world, which reads the stalling host's lock (`S2C/run_stage2.py:567-572`;
`S2/prereg.json` `run_order` step 3). The exploratory pooled look runs once every seed 0-4 phase it needs is done (v6
refuses it otherwise: `S2C/policy.py:623`, `S2C/run_stage2.py:614-624`). An unfinished phase delays it until that phase
is done (C5). No seed is dropped or replaced. Section 5 lists the recorded failures that running on the laptop can cause.

**Tested by** (`test_v7.py`, with stub gate files, stub reference files and one real reference file):
- no entry: `stage2_common` is not imported before the hand-over and `SeedContext` is untouched;
- a policy entry: `h2` returns the gate's dictionary with exactly the entry's keys replaced; every other key is
  byte-identical (same `tree_sha256`); a changed gate `actors.msgpack` still refuses through the original loader;
- the loader works on a real `L.save_tree` file holding several actors (three for the stalling world, two for the
  no-stall world), not only on a stub holding one;
- a host entry: `host` returns the re-made state, and the original `get_host` path was called;
- world keys: for a seed with a `(host_train, s, stall)` entry, `host_train --world no_stall` installs no redirect and
  is labelled "own run"; `host_train` without `--world` counts as stall; `policy_no_stall_train` matches only a
  no_stall entry, whatever `--world` says;
- exit 97 for: another seed in the replaced method; an entry for seeds 0-1 or 5-9, deploy, any evaluation, the
  no-stall host, detection or the pooled look; a reference file or per-actor SHA-256 that differs from the pin; a
  table in a draft `prereg_v7.json`;
- the gate's files are byte-identical before and after;
- through the real `launch_v7.sh`, a stub phase with an entry writes a lock whose `actor_sha256`, under the lock name
  of the references table, equals the re-made one for each replaced actor;
- the probe calls pass the gate's arguments (`GATE/acceptance.py:1475-1477`, `:1071-1072`; `S2C/fresh.py:245-253` for
  P1c), checked with recording stubs; and the probes never call a gate function that has a training or writing branch
  (`get_clone`; `get_critic` or `get_host` with `train=True`; any stage function), checked with stubs that fail if
  called;
- step 2's rule: P1b bit-identical to the gate keeps the gate's file although P1a differs; the forecast prints
  NOT READY for a gap above 1e-5 (stub actors) and only reports a smaller one;
- `--remake`: a remade file with the pinned SHA-256 lands at the original path and is read; the damaged file is moved
  and never read; a remake with another SHA-256 leaves the run halted.

**Cloud evidence.** Before the draft is pushed, P1a and P2 for one seed must reproduce the gate's saved actors bit for
bit on the cloud's processor, each run on one processor (0 or 1, so the v6 run on processors 2 and 3 is undisturbed). A review run already did so for seed 2 on 14 Oct
(`scratchpad/feas_review/p_probe.py`, outputs in `scratchpad/feas_review/out/`): P1a stall (gapped, full, random),
P1a no-stall (both actors) and P2's final actor and critic were all bit-identical to the gate's, taking about 185 s,
128 s and 488 s, with the read-only clone path. The implementer copies the script and outputs into `V7/evidence/`,
records the processor count that run used (it was not recorded), and repeats it on one processor if that count was
not 1. If a later run does not reproduce, the implementer finds out why before anything is pinned: a wrong argument is
fixed; a cause outside the call (for example the gate having run with another processor count) is recorded in
`V7/evidence/` and stated in the provenance file. Either way the rule above still applies; nothing waits on it.

### C8. Seeds 5-9: entirely on the laptop

**v6 rule.** Fresh seeds 5-9, with every setting frozen from seeds 0-4 (`S2/prereg.json:205`), in the order of
`S2/prereg.json` `run_order` step 4.

**v7 rule.** Every phase for seeds 5-9 runs on the laptop through the wrapper, from `fresh_data` to the confirmatory
detection marks and the confirmatory pooled look, with the plan of `GPU/lib/runner.py:89-197`. The confirmatory
results come from one machine. Their only inputs from v6 are the frozen choices (the version decision files and
`best_standin.json`) and the gate's files. Check (b) compares two laptop programs, each run on one processor (C3): the
BCA program at 0 and the actor-only run of `fresh_learners` (`S2C/fresh.py:245-253`). Whether the laptop makes these
two agree within 1e-5 is forecast before approval by P1c against P1b (C7, step 2b); no re-made reference is ever used
here. Seeds 0-4 and 5-9 are scheduled together, longest chains first. An unfinished seed 5-9 phase delays the
confirmatory pooled look until it is done (C5).

**Tested by:** the runner's `--plan-only` listing and `test_v7.py`'s plan tests: dependencies, inherited phases
skipped, failed-on-record seeds skipped, re-make entries only on seeds 2-4.

### C9. The GPU: out of v7, except an information-only timing

**v6 rule.** Processor only. The gate refuses any backend but `cpu` and any `XLA_FLAGS` (`GATE/acceptance.py:115`,
`:468-470`; `GPU/README.md:11-26`).

**v7 rule.** Unchanged. The self-test may time one training step on the RTX 5070 Ti, for information only, in the
separate environment `/root/bcagpu` that no phase uses (`GPU/README.md:141-146`), with a byte copy of
`GPU/lib/gpu_probe.py`.

**Why.** Repeatable GPU training needs `XLA_FLAGS`, which the gate refuses; check (b) on seeds 0-4 compares with a
processor-made actor; at 12 lanes the wall-clock time is already bound by one seed's chain. Nothing in the evidence
favours the GPU. Whether `jax[cuda12]==0.6.2` supports this Blackwell card is unknown.

### C10. Approval route, provenance and returning results

**Approval: the same route as v6.**
1. The user runs `python3 V7/approve_local_v7.py` at their own terminal. It refuses unless stdin is a terminal. It
   shows v7's fingerprint, v6's fingerprint, the list of changes (section 1), the failures v7 can cause (section 5),
   the handoff summary, the probe results, the forecast gaps and the table of re-made references with their lock names
   (C7), requires the user to type the first 16 characters, and writes
   `V7/approval_v7_<prefix>.json` (+ sidecar) once. Modelled on `S2C/approve_local.py:297-342`.
2. The user commits and pushes it under their own name.
3. The cloud runs `V7/code/approve_v7.py import`. It requires exactly one commit on
   `origin/claude/main-bca-connection-x4o46s` that added the file, and refuses if the author or committer email is
   `noreply@anthropic.com` or the session's configured `user.email` (the rule of `S2C/stage2_common.py:83`,
   `:483-521`, `:524-549`, copied verbatim). New, positive check: both emails must equal v6's approver,
   `35092758+dbayha7@users.noreply.github.com`. This is still not authentication: git does not authenticate authors
   (`S2/prereg.json:47`). The import writes `V7/import_v7_<prefix>.json`, which the agent commits and pushes.
4. Every v7 phase re-checks this chain with read-only git, as v6 does (`S2C/stage2_common.py:600-684`). The v6 code
   inside the phase separately re-checks v6's own approval (`7dab7d5`, `2eab3c4`), unchanged.

The agent-identity tests are copied verbatim from `S2C/test_stage2.py:864-870`.

**Why not a full v7 copy of the study code.** A copy with new fingerprints would need changed loaders in `band.py`,
`policy.py` and `host.py` to accept v6 files (`S2C/band.py:459`; `S2C/policy.py:313`, `:595-600`; `S2C/host.py:439`):
edited study code to review for accidental method changes. Running the v6 code as is avoids that.

**The cost of this choice.** Outputs made under v7 carry v6's fingerprint (`9b38c7c4...`) and approval (`b78b8772...`).
Three things say which outputs v7 made:
- **The ledger.** The wrapper installs `sys.addaudithook` before the hand-over. Every `open` in a write mode (which
  covers `write_text`, `write_bytes` and `np.savez`), `os.rename` and `os.replace` is appended, as it happens, to
  `R7/ledger/<phase>/<attempt>.writes`, so a killed attempt still leaves its list. After each attempt,
  `R7/ledger/<phase>/<attempt>.json` (+ sidecar) records: the arguments and processor; start and end times, exit
  status and its classification (done, machine-wide or phase-specific interruption, C5), and the boot id at start
  and end; every file written, with SHA-256; writes outside the phase's table row and outside
  `R7/` (flagged); VmHWM and CPU time; the processor model, its AVX-512 flag, OpenBLAS kernels, numpy's dispatch,
  glibc and the WSL kernel; the constants file's fingerprint, the A and K differences and the tripwire state; the
  reference used (C7); any retry request (C5); the v7 and v6 fingerprints. Probe processes are ledgered the same way.
  C5 uses the table, not the ledger, to find files; the ledger is a cross-check.
- **The provenance files.** `R7/provenance_{exploratory,confirmatory}.json`, written by a no-BCA bookkeeping step after
  each pooled look, and again whenever `make_results_v7.py` gathers results (naming every phase still unfinished).
  Per seed and component (detection, policy per world, deployment per world, host per world) they give the source
  (v6-cloud inherited or v7-laptop), the reference used and its label (C7), the SHA-256, the machine facts, retries and
  unfinished phases, the probe, forecast and R1 results, the labels of C2 and C7 (including "machine-caused: memory"
  and "machine-caused (re-make failed)"), the gate's and the re-made actor's points side by side (C7, Labels), and the
  process-hours. They recompute no number.
- **The status note.** The pooled look's own status sentence is v6's and cannot change, so the provenance file states
  that the exploratory group mixes machines and, where it applies, references re-made on the laptop.

**Returning results.** `V7/laptop/lib/make_results_v7.py` copies into `V7/results/`: every JSON record made under v7
with its sidecar; the ledger, the provenance files, the probe records and the self-test reports; the logs, renamed to
`.txt` because `.gitignore:13` ignores `*.log` (as `GPU/lib/make_results.py:132` does); and `BUNDLE.json`, an index of
fingerprints, machine facts, the reference files' SHA-256, the SHA-256 of every inherited file as found on the laptop
at the end, and the list of unfinished phases (empty when the run finished). No model or `.npz` file is copied (`.gitignore:11-12`). The user commits and pushes `results/`
under their own name (step 13). Then, as the user decided (Q6), the laptop's new run files, model files and
`R7/refs/` included, go to a new orphan branch (step 14).

**The return check.** In the cloud, `V7/cloud/check_return_v7.py` checks: every fingerprint; the ledger's
completeness; the constants facts in every phase and probe attempt that computed anything (one that reached the
hand-over or is done; an attempt interrupted before the constants step needs none) (in place of `gate_accepts True`,
`GPU/cloud/check_return.py:150-151`);
that inherited files are unchanged (BUNDLE.json against the final snapshot's sums); that no phase has both an inherited
and a new output; that every new file under `OUT/` is named, with its SHA-256, by a ledger attempt whose wrapper
verified this v7 import record before the phase ran (the attempt's checks name the import record's SHA-256: an order
that needs no clock, so the laptop's clock is never compared with the cloud's here); that the
cloud's own copy of `OUT/` (outside `logs/`), if it still exists, still matches the final snapshot; that every commit
touching `results/` or the laptop's run-files branch is the approver's; that the run-files branch's sums match the
ledger, every model file matches the SHA-256 its lock records, and every reference file matches its pin; and that each
seed 2-4 lock made on the laptop records, in its `actor_sha256` field only, the re-made actor where the table has an
entry and the gate's actor where it has none. It writes `V7/return_check.json` and the cloud reports the result in the
session. It refuses on any failure, before any result is read.

### C11. The laptop self-test (no BCA, no outcome)

It writes only to `R7/selftest/`, `R7/refs/` and `R7/ledger/` (with `--remake`, also `R7/interrupted/refs/` and
`R7/refs_remake/`), runs no BCA code (P1b runs the room's program with W = 0) and starts no phase, so it needs no
approval. The
full form runs once, at step 8. The **short form** runs before every run and after every restore: ST1, ST2 (versions),
ST3, ST4, ST6, ST7, ST8, and a check that the P1 and P2 records and the reference files exist and (after the handoff)
match the final `prereg_v7.json`.

| Check | Required or information |
|---|---|
| ST1. Restore and fingerprints: run files against the pinned sums, v6 code placement, v7 files against the v7 manifest, v6 chain read-only | required |
| ST2. Versions, installed files, dependency folder, glibc, processor model and flags, OpenBLAS kernel, numpy dispatch, XLA's processor name | required (versions); information (the rest) |
| ST3. A and K, local against pinned (C1 step 2) | required |
| ST4. Gate `load_prereg` + `guard` with constants, on **every** lane processor (about 3.5 s each). No processor is dropped: any failure is NOT READY | required |
| ST5. The same without constants: how many anchors differ on this laptop | information |
| ST6. `load_seed` for seeds 0-4 in both worlds, with constants (about 80 s at cloud speed) | required |
| P1. Policy reproduction (P1a), check (b) forecast (P1b) and the seeds 5-9 forecast (P1c), seeds 2-4 x both worlds (6 processes, about 6-10 min each at cloud speed); P1a makes the R-stall and R-no_stall copies | required to finish; decides C7's table; a forecast gap above 1e-5 is NOT READY (P1b), unless the draft records the user's choice to proceed (Q7); full form only |
| P2. Host reproduction, seeds 2-4 (3 processes, about 8-15 min; 488 s at cloud speed for seed 2); makes R-host copies | required to finish; its result decides C7's table; full form only |
| P3. Repeatability under load (C3) | required; full form only |
| ST7. One processor visible per lane (as `GPU/lib/selftest_checks.py:158-159`) | required |
| ST8. Git: full clone, `7dab7d5` present, origin branch present, no git identity equal to the approver's or Anthropic's, no changed tracked file outside `V7/results/` (which the results step rewrites; as `GPU/lib/selftest_checks.py:203-231`) | required |
| ST9. WSL memory (a warning below 40 GiB), free disk (at least 20 GB), P3's training run time (no lane-hour estimate), GPU timing; the laptop's clock against the `Date` of an HTTPS reply from github.com | information, except the clock in the full form: more than 2 minutes off is NOT READY before any probe runs (the records' times enter the handoff, C6) |

The last line reads `READY`, `READY (waiting for handoff/approval)` (after a `WAIT` line naming what is still missing:
the handoff, or the approval chain with its reason) or `NOT READY (<what failed>)`, followed by the
probe results (for each seed and world: "bit-identical", "gate's file kept (P1b reproduced it)" or "reference will be
re-made if not inherited", and the two forecast gaps F_ref and F_fresh) and a block to paste into the cloud session.

---

## 5. Risks, and what stops a seed, compared with v6

**What stops a seed: the same kinds as v6.** Data checks fail (`data.json` passed=false); a learner fails
(`learners.json` failed=true); a band fit fails; check (b) fails (`stop.json`). All are recorded and never replaced
(`S2/prereg.json`, `confirmation.failed_seed`). v7 adds no new kind of stop. **But running on the laptop can cause
recorded failures that the cloud would not have had, and v7 cannot rule them out:**
- *Check (b) stops from the laptop's arithmetic*, on seeds 2-4 (with or without a re-made reference) and on every
  confirmatory seed 5-9: on the laptop, check (b) always compares two different programs (the BCA program at 0 and an
  actor-only run). The self-test forecasts this (P1b against P1a or the gate's file, and against P1c) and says NOT
  READY above 1e-5 (C7, step 2b), but a forecast is not a proof. A stalling-world stop leaves that seed with no values,
  so every mark M1-M4 of its group fails (`S2C/policy.py:576-577`, `:661`, `:668-670`).
- *"Re-make failed"* (C7, step 2): check (b) then compares the laptop's BCA program with a cloud-made actor. A stop
  there is labelled "machine-caused (re-make failed)".
- *Memory exhaustion inside the stand-in rival*: pinned code records any exception there as the rival failing
  (`S2C/policy.py:419`, `:558`), which fails M4 for the whole group, the confirmatory group included. Guarded by C2's
  reservations (at least 7 GiB, or twice the measured peak, for the two policy kinds); labelled "machine-caused: memory".
- *Other numerical failures on seeds 5-9* (a learner or a band fit with non-finite values, a fresh data check) are
  recorded as v6 says. Whether another machine would have failed the same way cannot be told; this is true of any
  machine v6 runs on.
A check (b) stop on a laptop seed 2-4 stays a recorded stop, whether or not its reference was re-made, with the probe
results and forecast beside it in the provenance file.

**What v7 adds:**
- **Re-made references** (C7): on seeds 2-4, where the probes are not bit-identical. Every phase still runs; the
  provenance file labels them.
- **NOT READY before anything starts**: the self-test's required checks, including the check (b) forecast (C7, C11).
  The cloud keeps running v6 meanwhile.
- **Unfinished** (C5): a phase with four phase-specific interruptions, the last when run alone, waits for the user's
  `--retry`, and is retried until done; every other phase continues. Never a failure. Machine-wide restarts never count.
- **Run halts** (no seed is marked failed): the constants file's fingerprint is wrong; A or K is beyond tolerance; the
  gate refuses even with the constants; the environment is not exact; more than one processor is visible; the v7 or v6
  chain fails; the handoff is missing, differs or is withdrawn; the tripwire fires, an unknown tuple appears, or a
  reference redirect fails (exit 97, before any output is written); a probe record or reference file differs from its
  pin; a phase exits non-zero for an unclassified reason.
- **Waits:** a slot is busy; the memory rule (C2) says no; new policy training after a policy lock with a stand-in
  error (until reviewed; other phases continue).

| Risk | Likelihood | Guard |
|---|---|---|
| Arrow Lake's real XLA code does not reproduce the cloud's training | unknown (emulation reproduced seed 0; output not kept) | P1 and P2 detect it; the reference is re-made on the laptop for that seed and world (C7) |
| The laptop's BCA program at 0 and its actor-only program disagree beyond 1e-5 (check (b) stops on seeds 2-9) | unknown (exactly 0.0 in the cloud) | forecast by P1b against P1a or the gate's file and against P1c; NOT READY above 1e-5 (C7, step 2b); residual risk stated above |
| The policy test's "none" and the deployment check's "none" differ for a re-made world; the host test's "plain" and the deployment check's "td3bc" differ for a re-made host | certain where a re-make happens | declared (C7); the probe records give the size; provenance labels both pairs |
| Core type or load changes results | unknown (same instruction set) | P3 (repeatability only) |
| Memory exhaustion records a rival failure, or kills phases | unknown (policy_train's peak unmeasured) | C2 reservations (max(7 GiB, 2 x peak) for the policy kinds) and policy_train cap; memory-error exits rerun, alone after the third (C5); stand-in error block and label |
| Sleep, updates or a WSL shutdown kill phases | likely at least once | C5: machine-wide interruptions rerun without limit and are never counted; README power and update settings; tmux |
| The cloud finishes little before the handoff (its container restarts cut seeds 2-3's policy_train twice; see the appendix) | possible | the laptop runs every phase not inherited; nothing waits on the cloud |
| Both machines write the same phase | prevented on the laptop by code; in the cloud only by procedure | C6 steps 1 and 5; HANDOFF.json records it; the return check compares the cloud's copy |
| Laptop speed per core, and throttling | unknown | the self-test's speed line; results unaffected |
| A code path recomputes A or K outside the two replaced functions | none found (grep; the saved probes counted 0 calls, C1) | tripwire exits at the first call |
| Exploratory results mix machines | certain, by design | ledger and provenance files; seeds 0-4 carry no claim |

---

## 7. Decisions

The user answered all six questions of version 1 in the session on 13-14 Oct 2026. The designer (an agent) applied them as
follows; the user approves the result with v7 (C10). The second review raised one new question (Q7), which has a
default.

| Q | Question | Decision (the user) | Where applied |
|---|---|---|---|
| 1 | Accept the constants file as a declared operational departure? | Yes | C1 |
| 2 | Ubuntu 24.04 in WSL instead of 22.04? | Yes; an installed 22.04 can stay | C4, steps 1-2 |
| 3 | 12 lanes by default, at most 16, laptop only, WSL given 48 GB, the compute budget an estimate that stops nothing (actual hours reported)? | Yes | C2, step 0 |
| 4 | What if P1 or P2 shows the laptop does not reproduce the gate's training bit for bit for seed 2, 3 or 4? | The user rejected holding those phases: "figure out how to fix them and then proceed. dont hold up progress" | C7's re-make rule (designed by the agent, declared before any laptop outcome); C5 follows the same aim (machine-wide restarts never counted; an unfinished phase is retried until done) |
| 5 | When should the cloud stop? | Option (c), "Keep going until handoff": the cloud keeps running v6 until the handoff; phases it finishes are inherited | C6 (until the handoff; step 1); the appendix |
| 6 | Also push the laptop's new run files, model files included, to a new orphan branch? | Yes | C10, step 14 |
| 7 | (New, from the second review.) If the self-test's check (b) forecast shows a gap above 1e-5 between two of the laptop's own programs, should v7 stop before anything starts (the cloud keeps running v6, and the user decides with the cloud on a new version), or proceed anyway, accepting that check (b) may then stop seeds 2-9 for a machine reason that no re-made reference can fix? | **Open. Default: stop (NOT READY).** The designer recommends the default: proceeding would likely spend the one-shot confirmatory seeds 5-9 on a machine artefact. The answer must be in the draft `prereg_v7.json` before any P1 record exists, with the user's words quoted and the time of their message (an answer is never inferred); the draft writer refuses a later change, the probe records the cloud imported included, so the cloud asks whether step 8 has started before it records an answer. The answer that governs is the one recorded in `prereg_v7.json` (`remake`: `q7_if_forecast_above_tolerance`, `q7_status`, `q7_user_message`), which the approval screen shows | C7 step 2b, C11, section 5 |

One choice is left open by Q5 and has a default: at the handoff, running cloud phases stop at once; the user may
instead say "let them finish". The choice must be made **before the full self-test (step 8)**, because which phases
are inherited decides which can use a re-made reference (C6).

---

## Appendix: what was known when v7 was written (section 2 of the specification)


v7 was designed **after** the exploratory detection outcome was known and **after** seeds 0-1's policy, deployment and
host outputs existed. That is why v7 changes only operations.

**State at 02:36 UTC on 14 Oct** (from the cloud driver's log `OUT/logs/explore_policy.log`, file listings and `ps`;
no result value was read):

- Detection on seeds 0-4 is finished: 25 band fits on record, none failed. The uniform version failed direction or
  size; the weighted version passed both; the claim rule is not met for either. So the version is "weighted"
  (`OUT/detect/marks_exploratory_{uniform,weighted}.json`, read by earlier passes).
- The rival stand-in is `range_hinge` (`OUT/best_standin.json`). The room test's condition allowed training.
- Seeds 0 and 1 are finished in the cloud: the policy test in both worlds (check (b) exactly 0.0, read by earlier
  passes from the locks), the deployment check in both worlds, and the full host in both worlds (the no-stall
  `host_train` ended at 01:22:43 and 01:30:36 UTC, `host_eval` at 01:23:15 and 01:31:10). These files are not in the
  run-files snapshot `901c877`, so a "part 4" snapshot commit is certain (C6).
- Seeds 2 and 3: `policy_train` started at 01:23 and 01:31 UTC, was cut by driver relaunches at 01:57 and 02:29, and
  runs again since 02:29:13 on processors 2 and 3 (pids 275 and 276). Seed 4 has not started.
- The driver (`scratchpad/ood/stage2_run/explore_policy.sh:38-43`) continues, per lane: seeds 2 and 3 (policy test,
  deployment, hosts); then seed 4; then the exploratory pooled look. It has no notion of a handoff.
- What restarts it: the Routine `trig_01YCP5oPNVkUPRgta25ag1y4`, "Stage 2 run watchdog (hourly)", cron `56 * * * *`.
  Its prompt relaunches `explore_policy.sh` when no `run_stage2.py` process is alive and the run is not finished. The
  log shows starts at 16:56, 17:05, 19:57, 20:56, 21:57 and 23:46 UTC on 13 Oct and 01:57 and 02:29 on 14 Oct. A
  second Routine, `trig_0196VjNUjbaeZy2u8XXU7MBW` ("Check BCA PR #1 (safety net 12)"), reads the same log and refreshes
  the PR description; it starts no phase. The cloud runs only while its session is active; a reclaimed container
  kills running phases (policy_train needed 3 starts for seeds 0-1, host_train 5).
- Cloud timings, two lanes in parallel (from the driver's log): policy_train 50-53 min; policy_eval about 3 min;
  policy_no_stall_train about 10 min; policy_no_stall_eval about 3 min; deployment about 4 min (stalling world) and
  3.5 min (no-stall world); host_train about 75 min in the stalling world and 21-29 min in the no-stall world;
  host_eval under 1 min.
- Measured memory peaks (VmHWM): host_train about 0.91 GB in the stalling world and about 0.94 GB in the no-stall
  world. policy_train's peak is unknown.

**Not read by anyone in this workflow, as far as the record shows:** any policy `values.json` (the points behind
M1-M5), any deployment or host result value, any pooled look (none exists), and any policy mark.

**Not read by this writer:** no policy, deployment, detection-marks or host output file. This writer read code, the
v6 pre-registration, the `GPU/` package, the probe files in `PROBE/` (anchors, A, K, ufunc outputs: no study outcome),
the driver's log (start and end lines), the Routines list, the run-files file list and folder listings.

---

## Details the code fixes

The specification leaves these details open, or the code departs from its letter in these ways; the code settles them
as follows. None of them changes a pass mark, a claim, the analysis or a method setting.

1. **How the probe records reach the cloud.** After its verdict, the full self-test prints a block of lines holding
   the nine probe records byte for byte (compressed, each with its SHA-256). You paste it into the cloud session
   (README.md, step 8, which shows how to copy the saved report with `clip.exe`). `cloud/make_handoff.py
   import-selftest --from-transcript` reads your message byte for byte from the cloud session's own transcript (one
   wrong character would fail the block's SHA-256 check, so nothing is retyped). It refuses unless the block comes
   from a READY full self-test whose repeatability test P3 passed (a refused block is recorded, with why, and none of
   its records is stored), checks each record's fingerprint and that it binds to the committed draft (its re-make and
   forecast block, the constants file, the gate's pre-registration, version 6's code manifest and the probe script),
   all before storing any, stores each, once, in `handoff/probes/selftest/`, and writes an import record with the
   cloud's own time and where the block was read. The final `prereg_v7.json` is
   written from these records (`code/write_prereg_v7.py final --records handoff/probes`). The reference files
   themselves (model files, a few MB each) stay on the laptop: the final pins their SHA-256 from the records; the
   runner (before anything starts) and the short self-test check every record and reference file against those pins,
   and the wrapper checks the reference file of a re-made entry it reads (C7 'Probe records').
2. **HANDOFF.json** also records the draft `prereg_v7.json` it goes with (`write_prereg_v7.py final` refuses another
   draft, and, because HANDOFF.json is written once, `write_prereg_v7.py draft` then refuses to write any other
   draft; if a v7 file had to change after the handoff, the only route is the way back, `WITHDRAWN.json`), every
   self-test pasted to the cloud, the full self-test it rests on (the most recent full self-test pasted, which must be
   READY with P3 passed and have run under that draft; a pasted full self-test whose P3 found different results ends
   v7, while one whose P3 run did not finish is only run again; its import
   record's SHA-256, verdict and P3 result are also copied into the final `prereg_v7.json` and shown on the approval
   screen, with whether it ran under the draft), the cloud's time of the first self-test paste, the laptop's measured
   clock offsets, the check that no `run_stage2.py` and no `explore_policy.sh` process was left, and the cloud's
   process-hours read from the snapshot's own logs (C2's report of hours used). The stop rule's choice time must be
   written exactly as `YYYY-MM-DDTHH:MM:SSZ` (or "default") and must come before both the first probe record (the
   laptop's clock, which the full self-test checks to within 2 minutes; that much is allowed) and the first paste
   (the cloud's clock). `make_handoff.py ready` checks the self-test side before the cloud stops anything.
3. **The restore before the handoff** leaves out the files of every phase that is not finished in the snapshot
   `901c877`, found with the phase table on the snapshot's own bytes, as C6 step 4 does after the handoff. After the
   handoff, the list it computes must equal HANDOFF.json's. The restore accepts the pinned commit when it is the
   run-files branch's tip **or an ancestor of it** (the specification says "equals"): before the handoff the cloud's
   "part 4" snapshot moves the branch on, and the pinned commit and the pinned list of fingerprints still fix every
   byte restored. The download runs in the shell (`restore_v7.sh`), where git can ask for your sign-in and shows its
   progress; the Python part then works on the downloaded branch only.
4. **The order on seeds 0-4** follows the structure used for seeds 5-9 (`GPU/lib/runner.py:89-197`): a seed's
   deployment check and its full host start after that seed's policy test (both worlds) is done or skipped. This keeps
   C7's order.
5. **A retry request's attempt.** The runner writes `R7/retry/<phase>/<n>.attempt_<m>.json` naming the attempt that a
   request allowed, and that attempt's ledger record names the request.
6. **--reviewed** must be typed at a terminal (like `--retry`) and writes `R7/reviewed/<phase>.json` once (the time,
   the errors reviewed, the command line). Typed in a second window while the runner runs, it is picked up within the
   runner's retry poll (no restart); typed again, it only says the review exists. The error a no-stall lock copies from its stalling lock
   (`S2C/policy.py:549`) is lifted by the review of the stalling phase; an error of the no-stall arm's own needs the
   no-stall phase's review.
7. **When only blocked policy training is left**, the runner ends with `WAITING FOR REVIEW` instead of waiting forever;
   restart it inside tmux with `--reviewed <phase>`.
8. **A phase whose last attempt failed** (an exit that is not an interruption) is started again on the next start, as
   `GPU/` does; its files are not moved (only an interrupted phase's files are). Files under `OUT/` of a phase that is
   not inherited and that no laptop attempt wrote halt the run and are never moved: for a phase not yet done, any of
   its table files; for a phase done on file, its final record (or failed-on-record file) unless an attempt classified
   done names it with its current SHA-256, and, for a phase with a re-make entry, a lock that does not record the
   re-made actors. The runner checks this at start-up, whenever it finds a phase done and whenever a phase ends. The
   provenance files take each phase's source ("v7-laptop") from this ledger evidence, not from HANDOFF.json alone.
9. **ST4 and ST7** run on the 16 processors that lanes can use, so every `--lanes` from 1 to 16 is covered: the runner
   always uses the first N of those 16 processors and refuses any other. The probes' logs are in `R7/selftest/logs/`.
10. **Q7's status** is recorded in the draft: open, with the default "stop" applying, until your answer is recorded.
    It is recorded only with your words quoted and the UTC time of your message (`write_prereg_v7.py draft
    --q7-answered --q7-quote ... --q7-utc ...`); "proceed" is refused without them, and nothing infers an answer.
    Both are in the re-make block, so the probe records bind to them: once any record exists, on the laptop or
    imported by the cloud, the draft writer refuses to change them (or the probe script), and the self-test reports a
    record bound to another draft as NOT READY, naming the draft to put back. The approval screen shows the answer,
    its status and the forecast's verdict line, with a plain warning when a gap above 1e-5 was accepted under Q7.
11. **The return check** writes `return_check.json` whether it passes or not, with each check's result. It requires
    the constants facts of every attempt that computed anything (a phase attempt that reached the hand-over or is
    done, a probe attempt that wrote its record); an attempt interrupted before the constants step needs none. A new
    file under `OUT/` must be named by an attempt whose wrapper verified this v7 import record (an order, not a time:
    the laptop's clock is never compared with the cloud's there).
12. **Interrupted attempts are classified on their own output.** Every attempt of a phase appends to one log, so an
    attempt's memory error is read only from its own lines (from the start line that names its write log to the next
    one), and an exit 97 is a v7 refusal whatever an earlier attempt printed.
13. **Attempts left without a record** (a launch shell or a self-test that did not survive) get their record once no
    process of them runs: the runner writes them at start-up (done phases included), the self-test for probe attempts
    (it holds `R7/selftest/selftest.lock`, so only one self-test runs at a time), and the results and run-file steps
    before they gather anything. A record torn between its JSON and its sidecar is moved aside to
    `R7/interrupted/ledger_records/` (never deleted) and written again. An attempt whose launch shell did not survive
    counts toward the process-hours up to the last line of its write log (marked as an estimate).
14. **R1** (information only) runs automatically once every phase is done (`run_v7.sh` starts it before gathering the
    results); it routes nothing, and a failure of it is reported without stopping the results.
15. **ST9** reports P3's measured training time (and WSL's memory, with a warning below 40 GiB), for information; no
    lane-hour estimate is computed from it. It also compares the laptop's clock with the `Date` of an HTTPS reply
    from github.com: in the full self-test, more than 2 minutes off is NOT READY before any probe runs (the records'
    times enter the handoff); elsewhere it is a warning, as is a clock that could not be checked.
16. **Results before the end** (for example while a phase waits for `--retry`) are gathered only on request, with
    `sh .../laptop/run_v7.sh results`; they are marked PARTIAL and name every unfinished phase.
17. **The way back** (`WITHDRAWN.json`) records your own confirmation, quoted with its time, that no v7 phase has
    started on the laptop, and, when you paste it, the listing of `R7/ledger/`; it is refused when that listing names
    any folder other than a probe's.
18. **The test seams** (`--v7-test`, used only by `code/test_v7.py` and `code/test_v7_laptop.py`) never fall back to a
    production path: every path must be given and lie inside a marked fixture folder outside the repository, and a
    fixture holding version 6's pinned `run_stage2.py` or `stage2_common.py` is refused.
19. **The short self-test's waiting verdict** `READY (waiting for handoff/approval)` comes with a `WAIT` line naming
    what is missing: the handoff (the draft is still here) or the approval chain, with its own reason (for example
    no import record yet) and the step to take (`git pull`, or steps 10-11). `run_v7.sh` shows the self-test's `WAIT`,
    `HALT` and `refused:` lines, and the last lines when it ends without a verdict.
20. **ST8** does not count changes under `results/`, which the results step rewrites (a committed interim copy no
    longer stops a restart), and names the tracked files that changed.
21. **A damaged probe record** is put back by `selftest_v7.sh --remake <probe>` from the cloud's byte copy in
    `handoff/probes/selftest/`, only when its SHA-256 equals the final pin; the damaged file is moved to
    `R7/interrupted/records/` (C7, 'A damaged reference').
22. **Lanes and --cpus**: the processors the self-test checks are pick_cpus(16) on the machine (a test fixture's own
    list); lanes are the first N of them, and a `--cpus` outside them is refused, in test mode too.
