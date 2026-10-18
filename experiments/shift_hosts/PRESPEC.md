# Unifloral hosts on ThrusterChain: pre-specification version 2 (post hoc) [DRAFT]

> **DRAFT of 17-18 October 2026: not frozen, not approved.** Drafted in the cloud from version 1's frozen files and
> revised after review. Before the freeze, on the laptop: the investigation of section 19.6 runs while version 1's
> approval still verifies (the probe checks it, so the code it imports is version 1's), and its result is written
> there; you decide what happens if 0.05 is exceeded later (section 19.7); this paragraph is deleted, " [DRAFT]" leaves
> this title and `prespec.json`'s, `prespec.json`'s `status` becomes `"frozen"` and its three `[TO FILL ...]` slots
> are filled (section 19.8, step 2). Only then does the stage-0 suite run on these bytes, and only then does
> version 2 have its fingerprint.

**Version 2 (post hoc).** Drafted 17 October 2026, after version 1's stage 1 stopped on stop rule S7 on 16 October.
It changes one rule after the start, with a value chosen after the failing value was seen, so the change is post hoc:
S7's closed-loop limit is 0.05 points per start, not 0.01. Nothing else in the rules changes; the change's
consequences (stage 1 reruns in full, one stage-0 test is adapted to the committed package, and `results.json` carries
a post-hoc caveat) are listed in section 19.7. Version 1 was frozen at commit `2e59db0` and approved at `608eefa`
(fingerprint `8c5e8421922dcceb`). Section 19 is version 2's write-up: what version 1 ran, the stop, your decision,
what had been seen, what the change can and cannot affect, the investigation, the restart and the budget. The rest is
version 1's text, changed only in the S7 row of section 9.2, the list of differences below and one note each in
sections 13 and 18. Where it speaks of the present ("Stage 0 is built and tested", "Nothing has
trained on ThrusterChain data"), it describes 16 October, before stage 1 started.

**Version 1.** Drafted 16 October 2026, after stage 0, and checked against the package as it stood that morning. It
is frozen, with `prespec.json` beside it, before any run on real ThrusterChain data. You approve both at your own
terminal with `approve_hosts.py` (section 18).

**What has and has not happened.** Stage 0 is built and tested. Nothing has trained on ThrusterChain data, no band has
been fitted on real arrays, and no learned policy's return has been produced or read. BCA's code has run only on
made-up arrays registered with the made-up-data guard (approval item 8). The data were copied and prepared without
learning, and fixed, non-learned controllers were scored for the scoring cache.

**How this document binds.**
- The approved plan is `PLAN_UNIFLORAL.md`, SHA-256 `c2cd916f39a9aeef506ee78bc58e21e9c5df6ad7bf3556d469ab70187e6149cb`.
  How its three reviews changed it is in `FIX_LOG.md` beside it, SHA-256
  `3d95c01db8942be3fb91c86bfe0f17a05a82f292ac44d26812d034e5b84041d1`. "Plan 6.1" below means section 6.1 of that plan.
- `prespec.json` holds the machine-readable part: settings, choices, decisions, stop-rule thresholds, claims, caps,
  seeds, data hashes and the frozen environment of stop rule S9. The code reads three of its keys: the stage driver
  reads `environment.s9_signature` (section 4.6), and `approve_hosts.py` shows `title` and `approval_summary` before
  you type. Everything else in it states the rules; the code holds the same numbers as constants, and they match.
- The fingerprint you approve is the SHA-256 of `prespec.json`'s SHA-256 together with the code manifest. The code
  manifest holds the SHA-256 of every file of `experiments/shift_hosts/` (this document included; `prespec.json` and
  the approval files excluded) and of the repository code the package runs (section 9.2, S1). Every run on real data
  re-checks the fingerprint, so any later change to code, settings or rules is refused until a new version is approved.
- Where this document and the plan differ, this document holds. Every difference is listed with its reason: the
  decisions in section 7, the amendments in section 16, the readings in section 17 and version 2's post-hoc change in
  section 19. Everything else is as the plan says.
- All of it is exploratory (section 15). No claim from data seeds 0-4 is confirmatory.

---

## 1. The question

For each of Unifloral's four offline learners (TD3+BC, CQL, IQL and ReBRAC), on the weighting test's ThrusterChain
logs, judged on the strong-mix start bank (plan 1, 6.2):
- **Claim 1.** Does standard BCA beat the learner alone?
- **Claim 2.** Does importance-weighted (IW) BCA beat standard BCA?

Around them: a gate that the learner learned at all (Claim 0), whether IW's band keeps its promised coverage under the
shift (Claims 4a and 4b), flags for lost return (1c, 2c, 3, 3b), a harm flag for physical failures (Claim 5), and, only
where Claim 1 or 2 passes, whether a gain needs the band's placement (Claim 6, stage 3). Section 10 has every rule.

## 2. The 12 setups, and the other runs

Per quality (plan 5.1):

| Learner | Alone | + standard BCA | + IW BCA | Room reference (alone, full log) |
|---|---|---|---|---|
| TD3+BC | Unifloral's imported step | restated step, one threshold for every row | restated step, a threshold per row | imported step |
| CQL | imported | restated, one threshold | restated, per row | imported |
| IQL | imported | restated, one threshold | restated, per row | imported |
| ReBRAC | imported | restated, one threshold | restated, per row | imported |

- **"Imported"** means Unifloral's file is loaded unchanged, after its SHA-256 is checked, and its own
  `make_train_step` runs (plan 3.1).
- **"Restated"** means each `make_train_step` copied line for line into `unifloral_<learner>_bca.py`, with BCA's
  additions: the dose (IQL: its actor weights), the band fit, per-row thresholds and each learner's safety checks.
  Standard and IW share one restated step; only their thresholds and refusal rule differ (CQL: section 5.5).
- **Every run is new**, on the graphics card, 100,000 outer steps: 60 setup runs (4 learners × 3 setups × 5 seeds) and
  20 room references per quality.
- **Sham-alone runs (choice 19; decision 2).** One per quality for TD3+BC and one for IQL, data seed 0, 100,000 steps:
  the restated step with BCA present and never refreshed, so every dose (IQL: weight) stays exactly Unifloral's. They
  are reported beside the claims and change no verdict. So each quality has 82 card runs of 100,000 steps.
- **Stage 1** adds 20 runs of 15,000 outer steps on medium-expert, data seed 0: per learner alone, standard, IW, IW with
  every weight 1 (stop rule S2) and a repeat of IW (stop rule S8). No return is read from them.
- **Reference row, not one of the 12**: "CORL-derived TD3+BC (weighting test)", read from the weighting test's own
  records, never in a claim, a Holm family or a kill rule (plan 6.3). It is read only after your approval.

## 3. Data

**Source** (plan 2.1). The weighting test's data, spec version 5, stalling world, master seed 20261016, built by
`experiments/shift_iw/data.py`. Two qualities, each with data seeds 0-4:
- **medium-expert**: 65% of steps replaced by uniform random commands in half the episodes, none in the other half;
- **medium** (approved as a separate test bed, with its own Holm families and cap): 65% random in every episode.

Each seed has 1,000 episodes of 100 steps: episodes 0-499 train (50,000 rows), 500-699 calibrate (the bank takes 5 rows
from each), 700-999 are held out; plus 300-episode test logs under the mild and strong current laws.

**The removal (the gap)** (plan 2.2). The densest-15% ball of commands of the same seed's expert log
(`experiments/truth/gap.find_gap`) is cut from the training rows and bank rows. Held-out rows and test steps are never
cut. The full log, uncut, serves only the room references.

**The copy** (plan 3.1, stop rule S11; amendment A8). 297 files were copied from read-only `/root/bca-shift` into
`/root/bca-hosts/runs/shift_hosts/source/`, each checked against the weighting test's own SHA-256 manifests or
sidecars, and byte for byte against its source. `copy_manifest.json` SHA-256:
`2b8c825bfb7d0529e253dcb2de4d5c7849a1907ec86ffe5d9c68e39186c62977` (all checks passed; source unchanged). The
weighting test's `RESULTS.json`, which has no sidecar, was checked by its git blob instead.

**The prepared files** (`prepare.py`, no learning). One per quality and seed at
`runs/shift_hosts/data/<quality>/seed<k>/prepared.npz` with its manifest `prepared.json`, each with a SHA-256 sidecar.
They hold the raw observations, commands, rewards, next observations, done (0 on every row), ReBRAC's next commands,
exact and estimated IW weights, the bank, the full log, and the strong- and mild-law test steps. Every count equals the
weighting test's recorded count and lies in plan 2.2's ranges (S11 passed on all ten).

| Quality | Seed | Training rows | Bank rows | `prepared.npz` SHA-256 | `prepared.json` SHA-256 |
|---|---|---|---|---|---|
| medium-expert | 0 | 45,301 | 901 | `58c4f7571f5bca3ccf34cca7f210ccd80d0cc953349f1ae672dd96d8635f4176` | `6c5f95c0b426ca72ff5f349e2add6faf3e1a25bdf5ce54c17db6a710a751d05b` |
| medium-expert | 1 | 45,372 | 891 | `2447d191a5a135bb0508d7f3870dacb62c0696de35d4c1b74a4c9e68f707e9e2` | `c9c48b881747f8a5837b3e84cdf2a8feaa6a1e623d37a310cc24389954890583` |
| medium-expert | 2 | 45,005 | 869 | `df6120f00fcf1d7155c7cf75bd091febf49f8e14c24c8c6a20b0d90df9de8b9c` | `a90a615704883d4d72dfb35adee730182159875e390f04d4e9b00a4e03157f02` |
| medium-expert | 3 | 44,923 | 924 | `4746e8a0566eabdefd07c98aa757541e54acfc9275bc98b9df148c539644807c` | `9372c7acdb1134ccfc59ace280ad9dd53f35146d9911985833e9280ac5d50610` |
| medium-expert | 4 | 44,902 | 892 | `7dc21939e9d3d2089540c9c0ebde4e601785923e6424565c53938d143ba1c0c0` | `34f45fa8ffa3e21adb612e2f9a85e20734e8f7b7818955ab95b73a60288d7269` |
| medium | 0 | 47,439 | 942 | `5c433e8ac1b57f78f97835457172e481dc941b0a113af965da2f76cc5ed4da25` | `c4be01d37ff3d57c0c32ef538342a35a97f3002ee63a8ec43e7df5d5af939ca0` |
| medium | 1 | 47,479 | 933 | `c25704833fc7a0ff1f2d93f13f2ce4360fd56725a52cf982e3dff1c0b7b57116` | `84fddc455d4528867d59a07edee97fa32e0aa03c9276c7bf28a93845f2263db9` |
| medium | 2 | 47,548 | 934 | `607bbd1922060c612a23ece36afd29709be59e79a4cddcee67bfb49d3f14260c` | `f9e617f9f925f1008f7b86ab0f853ea16cef5c787ea950b39b71851af14b81d0` |
| medium | 3 | 47,502 | 964 | `46cd251426d523f7601956cbdef50c8e384b985ae754474c74fd03335ea33f11` | `0954d0b9b30b5a2078bb275d4846a078156ff8fbbb8cf5130d70ab3c45eb1fe2` |
| medium | 4 | 47,458 | 954 | `03368678ce6c71fdee6c3ae1f5fa8888dd77857ed8cd684e075c8541f585cb13` | `49784c1811e070b1385d2f5e342d3d3fce3a4933be8e96e507c46c1de899f9db` |

Every seed: full log 50,000 training rows and 1,000 bank rows; 30,000 strong-law and 30,000 mild-law test steps. The
summary of all ten is `runs/shift_hosts/data/summary.json`, SHA-256
`a6ad49e2a6d1cd1c8571ac756235729a57cbd2bb1d4d0ddf104ab4838ee48d3b`.

**Readouts of the data, counted during preparation** (no learning):
- **Exact IW weights** w(c) = p_strong(c) / p_train(c) = 0.090718 e^(4c), from the observed current only, in
  [e^−2, e^2] = [0.135, 7.389] on every row of every set and seed (plan 4.2).
- **Effective bank size** (Kish, exact weights): medium-expert 299-347 rows, medium 326-369 rows, far above the 36 rows
  a threshold needs.
- **ReBRAC's substituted next commands** (rows whose next recorded command does not exist; plan 3.8, choice 4):
  medium-expert 7.8-8.4% of training rows (23.8-28.8% of the highest-current tenth); medium 5.3-5.6% (16.0-17.8%),
  a little above the plan's estimate of about 4-5%. Reported beside every ReBRAC number.

**Evaluation starts** (plan 2.3). The weighting test's three start banks of 1,000 fixed starts each (no shift, mild
mix, strong mix; drawn from `default_rng([20261016, 1000])`) and the main study's four situations in the stalling world
(recorded conditions, strong current, big waves, big kicks). Their copies are pinned in `evaluate.py`:
`start_banks.hdf5` SHA-256 `9ad49e478dffdbc16160c3f69637dfb15b110035d64cf5951608d2b73904d50b` and `start_banks.json`
SHA-256 `cceb2f2d54c179cbfb00185c4d5dedf62c6ff7838f8357d6fe47596eeb3cb740`. On the strong bank the points anchors are
−1,351.41 (random commands) and −80.61 (the stall-aware controller). The scoring cache (each bank's zero-command and
stall-aware returns, the F2 threshold, the regret optima and the viewer's fixed starts, and CQL's noise files; fixed
controllers only) is pinned in `evaluate.py`: the 7 bank files' and the 7 noise files' SHA-256 are constants there
(taken from the cache's `cache.json`, SHA-256 `8519d551486e33644400ac097507780ef7c4961a04e1211c9bdb90c4727825de`), and
`evaluate.load_cache` reads no other file, so a cache rebuilt after this freeze would change the code manifest (S1).
The cache was built on 15 October with earlier versions of `evaluate.py`, `actors_f32.py` and `scoring_cache.py` (its
`cache.json` records their hashes). On 16 October the fixed controllers were rolled again with the final scoring code
on all 7 banks, and the zero-command and stall-aware returns, the late peaks, the F2 thresholds and CQL's noise
reproduced the cached arrays bit for bit; the regret optima (the exact solver, 0.55 processor-hours) were not solved
again.

## 4. Settings

### 4.1 Unifloral's defaults, pinned

Each learner uses its file's own Python `Args` defaults at Unifloral commit `f2dc1278eae18ed3e22c92255119c54885414d17`
(plan 3.2), read from the pinned files for this document:

| | TD3+BC (`td3_bc.py:23-46`) | CQL (`cql.py:25-49`) | IQL (`iql.py:24-46`) | ReBRAC (`rebrac.py:24-52`) |
|---|---|---|---|---|
| Learning rate | 3e-4 | 3e-4 critics and temperature; actor 3e-5 | 3e-4, cosine to 0 over num_updates for every network | 1e-3 |
| Batch | 256 | 256 | 256 | 1,024 |
| Target rate (Polyak) | 0.005 | 0.005 | 0.005 | 0.005 |
| Main weights | alpha 2.5; target noise 0.2 clipped at 0.5; 2 critic updates per step | 10 critics; conservative weight 10; CQL temperature 1 | beta 3; expectile 0.7; weight cap 100 | actor cloning 0.001; critic cloning 0.01; target noise 0.2 clipped at 0.5; 2 critic updates per step |
| Networks | normalize observations inside | raw observations | normalize inside | raw (norm_obs off); critic layer normalization on, actor off |

**ThrusterChain's exceptions, the same for all four** (plan 3.2):
1. discount (`gamma`) 0.98, not 0.99;
2. `num_updates` 100,000 outer steps (40 chunks of `eval_interval` = 2,500), not 1,000,000;
3. the data handed in as each file's own `Transition` from the prepared arrays, in place of D4RL; raw observations;
   `done` 0 on every row; the time-limit rows (t = 99) kept with their true next states (choice 2);
4. scoring by the project's exact float64 rollouts (section 5.6), not Unifloral's evaluation;
5. `seed` by the rule of section 4.5.

Nothing is tuned, and no other value moves.

**The pin.** `configs/unifloral.json` (SHA-256 `7d0c0eccda6730c0b40bf9085cdbc31f565acbd8010db337b01ef5e64cf26a58`)
names the revision and each file's SHA-256, checked before every import (stop rule S1):

| File | SHA-256 |
|---|---|
| `baselines/unifloral/algorithms/td3_bc.py` | `6f5ab6f98d706d902a18f11e1c81e99e2691dc4c8a53843bf6521c9f95d00eee` |
| `baselines/unifloral/algorithms/cql.py` | `41edbf1c11ad7b408ee6eafd94420e8fd1932afc8bbf72032bc1b9ff47213061` |
| `baselines/unifloral/algorithms/iql.py` | `088fca23120fc76fc9e33d01dfe6511726de9790dd3063c806d6a3e2a96992d7` |
| `baselines/unifloral/algorithms/rebrac.py` | `d7ea2b2bdf59b35ea6fb2370ca926b8d291bcad566bd0f5a9adbeafe74178d4f` |

Each file is imported by path under a module name other than `__main__`, so its training block never runs, with
bytecode writing off (plan 3.1).

### 4.2 The JAX 0.6.2 deviation

Unifloral pins `jax[cuda] < 0.6` (`baselines/unifloral/requirements.txt:2`). Training uses WSL `/opt/bca-venv`, with
JAX 0.6.2 and the CUDA 12 plugin (plan 3.1, choice 9). Equality with an older JAX is not verified. `/opt/bca-venv` also
carries a local numpy-alias fix (`zzz_numpy_aliases.pth`) that Unifloral's Dockerfile lacks.

### 4.3 The card's arithmetic (choice 8)

- Every Unifloral file sets `XLA_FLAGS="--xla_gpu_triton_gemm_any=True"` when imported. After the imports, and only if
  JAX's backend has not started, `unifloral_import.prepare()` replaces it with `--xla_gpu_deterministic_ops=true` and
  sets matrix precision "highest" (full float32). The record keeps the flags before, after the imports and in effect.
- Card runs set `XLA_PYTHON_CLIENT_PREALLOCATE=false` and Unifloral's Dockerfile variables
  (`baselines/unifloral/Dockerfile:42-45, 57-58`).
- Training-environment processes on the processor (stop rule S7's training-side commands, the band readouts) also set
  `--xla_cpu_multi_thread_eigen=false` (amendment A5).
- This changes rounding, not the learning rule; runs are not upstream's bits (plan risk 4).

### 4.4 The two environments

- **Training:** `/opt/bca-venv` (Python 3.10.12, numpy 1.26.4, JAX 0.6.2, flax 0.10.7, optax 0.2.8, distrax 0.1.5), the
  only environment where the Unifloral files import.
- **Data and scoring:** `/opt/bca-shift-venv` (numpy 2.2.6), the only one where `experiments/shift` imports.
- No process holds both a Unifloral network and the simulator; whatever needs both goes by file with SHA-256 sidecars
  (`handoff.py`; plan 3.1, test (i)).

### 4.5 Seeds and random streams (plan 3.4)

All streams use the weighting test's master seed 20261016. For data seed `rep`:

| Stream | Rule | rep 0 | rep 1 | rep 2 | rep 3 | rep 4 |
|---|---|---|---|---|---|---|
| Unifloral's `Args.seed` | `default_rng([20261016, rep, 31]).integers(2**31 - 1)` | 1903938727 | 1254516358 | 233067169 | 887362575 | 1751479981 |
| BCA's fit stream s91 | tag 91, same rule | 1122691669 | 1435880987 | 513185736 | 284360605 | 1463282055 |
| BCA's refresh stream s39 | tag 39, same rule | 512079666 | 1501176378 | 1809802209 | 783476312 | 382535294 |

(Computed for this document in both environments, with the same values.)

- Unifloral's key: `PRNGKey(Args.seed)`, the file's own network split, and one split after every 2,500-step chunk, where
  the training block splits off its evaluation key and discards it, so the training stream is the unmodified script's
  (test (g)). The port's fold-in constant 1414676809 is not used.
- A learner's three setups start from the same key, so they share initial parameters and batch indices (S4). The room
  reference shares the initial parameters.
- BCA's randomness never touches Unifloral's key. Per-step fit key: `fold_in(PRNGKey(s91), outer step)`, with each
  port's fold constants for the fit's target noise and bootstrap masses, and the scale network's start from
  `fold_in(fold_in(PRNGKey(s91), 0), the port's init constant)` (amendment A4). Refresh key:
  `fold_in(PRNGKey(s39), step)`, with the ports' constants for target noise and the posterior draws (TD3+BC and CQL
  1213156420 and 1347375956; ReBRAC 1380470604 and 1380994899; IQL 1347375956 only).
- CQL's scoring noise: `default_rng([20261016, 1004, bank index])`, standard normals 1,000 × 100 × 4 rounded to float32,
  bank index in the order none, mild, strong, recorded, strong current, big waves, big kicks (choice 7).
- New streams, declared here: the shuffled-weights placebo's permutation `default_rng([20261016, rep, 95])` (tag 95 is
  unused elsewhere in the repository); the coverage readouts' target-noise keys reuse the weighting test's tag 51,
  `default_rng([20261016, rep, 51, 2])` plus a group tag (1 actor commands, 2 visited states); the coverage episode
  bootstrap seed (20261016, 77) as in `shift_iw/coverage.py`.

### 4.6 The frozen environment (stop rule S9)

`prespec.json` → `environment.s9_signature` is in `stages.env_signature`'s format and was taken from a card process of
the final stage-0 suite (suite run 20261015T230917Z, its `unifloral_import.record()`). All 58 card-process import
records of stage 0 carry this same signature, and all 20 stage-0 card run records pass S9 against `prespec.json` as
the driver reads it. Every card run's signature must equal it in every key (decision 5):
- Python 3.10.12; executable `/opt/bca-venv/bin/python`;
- packages: jax, jaxlib, jax-cuda12-plugin and jax-cuda12-pjrt 0.6.2; numpy 1.26.4; scipy 1.15.3; flax 0.10.7;
  optax 0.2.8; distrax 0.1.5; chex 0.1.90; gym 0.23.1; d4rl 1.1; tyro 1.0.13; wandb 0.27.0; mujoco-py 2.1.2.14;
- card: NVIDIA GeForce RTX 5070 Ti Laptop GPU, driver 610.62; backend and device "gpu"; devices ["cuda:0"];
- `XLA_FLAGS` in effect `--xla_gpu_deterministic_ops=true`; matrix precision "highest"; `jax_platforms` unset;
- the recorded variables: the Dockerfile's (`D4RL_SUPPRESS_IMPORT_ERROR=1`, `MUJOCO_GL=osmesa`,
  `PYOPENGL_PLATFORM=osmesa`, `DISPLAY` empty, `MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210`,
  `LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin`), `OMP_NUM_THREADS=1`, `PYTHONDONTWRITEBYTECODE=1`,
  `SHIFT_HOSTS_DEVICE=gpu`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `JAX_PLATFORMS` and `CUDA_VISIBLE_DEVICES` unset;
- Unifloral's revision and `configs/unifloral.json`'s SHA-256 (above).

**Launch condition.** The stage driver gives each card run its parent's environment plus the Dockerfile's variables,
keeping any other `LD_LIBRARY_PATH` entries after MuJoCo's folder. Root's login shell in WSL sets
`LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:/usr/lib/wsl/lib:` (`/etc/profile.d/mujoco.sh`), which would be recorded
as `/root/.mujoco/mujoco210/bin:/usr/lib/wsl/lib` and stop the first card run with S9. So the driver is started with
`LD_LIBRARY_PATH` unset (section 18), exactly as the stage-0 card tests ran. A newer graphics driver, or any other
change in this list, stops the driver with S9 and needs a new version.

## 5. BCA and IW as built

### 5.1 What all four share (plan 4.1)

- **The band** q ± U around the critic's value q, meant to hold the critic's one-step error |y − q| 90% of the time,
  where y is the learner's own learning target.
- **The width** U = R × u × max(η, 1e-6): η is the scale network (state and command in, 2 hidden layers of 64), u a
  running average of the error size (averaging 0.99), R the threshold.
- **The scale networks**, built exactly as each port builds them, given the training rows' mean and standard deviation
  as Unifloral's training block computes them (choice 3): TD3+BC, CQL and ReBRAC `Calibrator(mean, std,
  state_dep=True)` with Adam at learning rate 1e-3 and optax's default epsilon 1e-8; IQL `StateDependentCalibrator`
  (hidden 64) through `create_cal_train_state`, epsilon 1e-5.
- **The fit**, once per outer step on that step's batch, with Bayesian-bootstrap weights and the soft coverage loss
  (sharpness 20, width penalty 0.005); IQL's fitter adds its balance term (weight 0.5).
- **Refreshes** after outer steps 10,000, 15,000, …, 95,000 (18; none at the end), between scan chunks: η and u are
  frozen; each bank row is scored |y − q| / (u × max(η, 1e-6)); `calibration/wbcp.calibrate` sets R with miscoverage
  0.1, credibility 0.95 and 1,000 draws. R = +∞ means "cannot certify"; certifying needs at least 36 rows.
- **The dose** = 1 + 0.5 × U / (U + u), between 1 and 1.5, read at the recorded (state, command) with no gradient. Before
  the first refresh the width is 0, so every BCA setup starts as its learner alone (S10).
- **Calibration rows**: the weighting test's bank, 5 rows per calibration episode cut by the same ball, never trained
  on; the same rows for every learner (section 3).
- **Standard BCA** weights every bank row equally and passes one threshold for every row.

### 5.2 Where the band enters each learner (plan 4.3)

| Learner | Dose or weight multiplies | Band target y | Band value q | Where the fit sits |
|---|---|---|---|---|
| TD3+BC | each row's cloning error in the actor loss (`td3_bc.py:239, 245`) | r + 0.98 × min of the 2 target critics at the noisy target command, BCA's own noise draw | min(Q1, Q2) at the start of the step | once per outer step, before both critic updates (choice 13) |
| ReBRAC | `actor_bc_coef * (sg(m) * bc_loss).mean()` (`rebrac.py:268, 274`); doses must lie in [1, 2] | r + 0.98 × [min target critics − 0.01 × Σ (ã′ − a′)²], a′ by Unifloral's roll | min(Q1, Q2), start of the step | as TD3+BC |
| CQL | each row's conservative gap for all ten critics (`cql.py:304-305`); a negative or non-finite dose stops the step | r + 0.98 × [min of the 10 target critics − α log π(a′ \| s′)], a′ from the just-updated actor, BCA's own draw | min of the 10 online critics | once per outer step, just before the critic loss (choice 14) |
| IQL | positive-advantage actor weights w = 1 + A / (A + U) × (min(e^(3A), 100) − 1); other rows keep Unifloral's weight | r + 0.98 × V(s′), V before its update | min of the 2 target critics this step's advantage reads (choice 15) | once per outer step, after the target averaging |

With a dose of exactly 1 (IQL: Unifloral's own weights) each restated step reproduces Unifloral's expressions.

### 5.3 Validity and safety checks

- Each restated step returns per-step validity flags: TD3+BC and IQL `inputs_valid`, `scale_inputs_valid`,
  `dose_inputs_valid`; CQL adds `dose_guard_ok`; ReBRAC adds `batch_valid` and `dose_in_range`. The runner stops a run
  on any false flag (S5).
- TD3+BC, CQL and ReBRAC discard an invalid step whole, random key included, as the ports do.
- **IQL** keeps its learner part (Unifloral's key and agent state) on learner-side validity and its BCA part on the
  whole step's validity (decision 4, amendment A2). The flag still reports every failure, so S5 stops the run.
- IQL's standard rule: unsupported rows get weight 0, and the actor step is skipped when no row of the batch is
  supported (computed, then not kept: amendment A6).
- Two port checks are not carried over, with their reasons in amendment A6.

### 5.4 What IW changes (plan 4.2)

- **Exact weights** w(c) from the observed current (section 3). Each kept bank row j gets w(c_j); each training row i is
  treated as a test point with its own mass w(c_i) (weighted conformal prediction, after Tibshirani, Barber, Candès and
  Ramdas, 2019).
- **After each standard refresh** of an IW setup: the bank scores are recomputed from the stored state with the
  refresh's own key and the learner's own target, and an unweighted `calibrate` must reproduce the stored threshold bit
  for bit (S3). Then R(t) is tabulated with the bank weights on the grid t = 1.01^k, with the same random stream. The grid
  covers every training row's and strong-law test step's exact mass plus the exact weight's two ends w(0.1) and w(1.1)
  (403 points in the tests). Each row rounds its mass up to the next grid point.
- **Width** U_i = R(t_i) × u × max(η_i, 1e-6); each batch row's threshold is read through Unifloral's own
  `batch_indices`. Only R changes; η and u are the standard band's.
- **Refused rows** (R(t_i) = +∞), choice 16: TD3+BC dose 1.5 (the weighting test's own code); CQL and ReBRAC dose 1.5
  (new rules); IQL weight 1 where the advantage is positive, Unifloral's weight elsewhere (a new rule). Standard BCA's
  own rule: dose 1; IQL weight 0, with the actor step skipped when no row is supported. The refusal rule is carried in
  the BCA state. Under IW's rule a refused IQL row counts as supported, so a batch whose rows are all refused still
  updates the actor.
- **Weights of 1** give a grid of [1.0], and IW must equal standard bit for bit (S2).
- **At an IW setup's last refresh (95,000)**, also recorded and reported, never judged: the estimated-weight table and
  its coverage (choice 20; a logistic classifier on the 13 observed numbers), the shuffled-weights placebo (bank weights
  permuted with the tag-95 stream) and IW's dose change against the standard threshold over kept training rows (IQL:
  its actor weights).

### 5.5 CQL's two compiled programs (decision 1)

CQL's standard and IW runs use Unifloral's pre-refresh program (`dose_active=False`, Unifloral's literal gap line) for
the chunks up to outer step 10,000, then the dose-active program from the first refresh on. BCA's reads of the step's
state, batch and keys sit behind one optimization barrier, which changes no value. With this, CQL's test (a2) on the
card is bit for bit; a single program drifted 1.55e-4 relative after 60 steps at a dose of exactly 1.

### 5.6 Scoring (plan 2.3, 3.9, 5.4)

- Every actor runs in float64, its parameters cast and its observation rounded to float32 first, through the exact
  float64 simulator, for 100 steps on each of the 7 banks; `nan_to_num` on the command.
- Scoring actions: TD3+BC and ReBRAC the tanh output; IQL the mean (already through tanh); CQL a sampled tanh-Gaussian
  command with the declared noise (choice 7), with the tanh of its mean scored and shown, never judged.
- Snapshots are scored at 50,000 (shown) and 100,000 (judged).

## 6. The 20 choices (plan 4.4), as frozen

| # | Question | Frozen value |
|---|---|---|
| 1 | Budget unit | 100,000 outer steps for all four learners |
| 2 | Time-limit rows (t = 99) | kept, done = 0 |
| 3 | Normalization statistics | Unifloral's own: mean and standard deviation of the training rows, NaN replaced by 1 (TD3+BC, IQL and every scale network) |
| 4 | ReBRAC's next command | Unifloral's roll over kept rows (`rebrac.py:335`); bank rows by the same roll within the cut calibration episodes |
| 5 | ReBRAC's actor cloning weight | 0.001 |
| 6 | CQL's strength | weight 10 over 10 pooled critics |
| 7 | CQL scoring | sampled, ε = `default_rng([20261016, 1004, bank])` standard normals (1,000 × 100 × 4) rounded to float32, the same for every CQL setup and seed; the tanh of the mean scored and shown |
| 8 | Card arithmetic | deterministic operations and full float32 (`--xla_gpu_deterministic_ops=true`, matrix precision "highest"); Unifloral's import-time Triton flag replaced after import, and recorded |
| 9 | Environment | `/opt/bca-venv`, JAX 0.6.2, a recorded deviation |
| 10 | Device | all four learners on the card |
| 11 | IQL | three separate runs; the alone run is Unifloral's unchanged step (test (d) not needed) |
| 12 | Room references | included: each learner alone on the full log, 20 runs per quality |
| 13 | TD3+BC and ReBRAC band fits | one per outer step, before both critic updates, critics at the start of the step, BCA's own noise draw |
| 14 | CQL band | q = min of the 10 online critics; fit just before the critic loss with the updated actor, α and target critics and BCA's own draw; one dose per row for all 10 critics; the refresh uses the stored state |
| 15 | IQL band | q = the target copy this step's advantage reads; refresh q = Polyak(stored target, stored critic, 0.005); fitter keys from BCA's stream |
| 16 | IW refused rows | most caution: dose 1.5; IQL weight 1 on positive advantage |
| 17 | Medium quality | included, approved separately, with its own Holm families and cap; it runs only if every stop rule passed through stage 2a |
| 18 | Holm family | the four learners per claim per quality; add-ons never join; every learner stays in |
| 19 | If test (a2) matches only to rounding on the card | keep the pairing with the imported alone run; one seed-0 sham-alone run per affected learner and quality, reported beside the claims: TD3+BC and IQL yes, CQL and ReBRAC no (decision 2) |
| 20 | Estimated-weight readout | at the last refresh only: one extra threshold table and its coverage |

## 7. Decisions made after the build

These five were decided before this pre-specification, on the stage-0 results, with no ThrusterChain outcome seen.

1. **CQL's two compiled programs** (section 5.5). Plan 4.3 has one restated step for both BCA setups and test (a2)
   allows at most 1e-6 relative on the card. The single program drifted 1.55e-4 relative on the card at a dose of
   exactly 1. Before the first refresh every dose is exactly 1, so the pre-refresh program changes no value; the plan's
   source report names this fallback. Effect: CQL's (a2) and S10 are bit for bit; choice 19 does not apply to CQL.
2. **Choice 19 follows the plan's letter.** TD3+BC and IQL recorded card rounding differences in test (a2): TD3+BC's
   state and key were bit for bit and only its logged actor and cloning losses differed (1-2 rounding units, largest
   relative 8.0e-8); IQL differed in every learner part within 3.8e-7 relative. Both are inside (a2)'s card rule, so
   for them S10 becomes a readout, and each quality gets one seed-0 sham-alone run for each. CQL and ReBRAC were bit for
   bit: no sham runs, and S10 stops for them.
3. **S5's thresholds.** A NaN threshold stops the run. R = +∞ is a counted refusal ("cannot certify"), not a stop
   (plan 4.1 defines R = +∞ that way; plan S5 lists "non-finite threshold").
4. **IQL keeps the learner part of its carry on learner-side validity** (its decision 1; section 5.3; amendment A2).
5. **The S9 signature is the final environment's** (section 4.6).

## 8. Stages and order (plan 5.1, 7.3, 7.4)

- **Stage 0, done.** Code transfer to branch `claude/shift-hosts`; the package; the tests (a1)-(i) and the guard's
  comparison on made-up arrays (S0); the data copy and preparation (S11) and the scoring cache, with no learning
  (amendment A9).
- **Stage 1** (medium-expert, data seed 0, the first 15,000 of the 100,000 outer steps, so one refresh at 10,000). Per
  learner: alone, standard and IW one at a time, then IW with every weight 1 and the repeat of IW two at once (to
  measure the two-at-once speed). Then S7 on all 20 stage-1 actors, and the checks S2, S4, S8, S10 (a readout for TD3+BC
  and IQL), S9 on every run, and S6 (the projection, the alone speeds and the two-at-once decision). No return is read.
- **Stage 2a** (medium-expert), starting by itself once every stage-1 stop rule passes, in this fixed order:
  1. the sham-alone runs (TD3+BC and IQL, seed 0, 100,000 steps);
  2. seeds 0, 1, 2, 3 and 4, each seed all 12 setups (every learner) before the next seed;
  3. the 20 room references, last, since they change no verdict.

  S6b runs before each of these units; S4 after every seed and after the room block. Beside the card queue, scoring
  lanes score every finished actor at 50% and 100%, compute every BCA run's coverage at its last refresh, keep the
  viewer's episodes for seed 0's final actors, and run S7 on every seed-0 final actor.
- **Stage 2b** (medium): the same, only after every stage-2a lane has finished and every stop rule has passed.
- **Stage 3** (controls: flattened and shuffled twins, optional estimated-weight IW), only where Claim 1 or 2 passes. Its
  code is not in this package, so it runs only under a later pre-specification version that you approve at your
  terminal, within its cap (section 12), in the fixed order: Claim 2's flattened and shuffled twins learner by learner
  in Holm order (smallest p first), then Claim 1's flattened twins in Holm order, then the estimated-weight arm; a
  learner's controls run for all 5 seeds or not at all.

**Where.** This laptop, WSL `/root/bca-hosts`, as root. The card queue in `/opt/bca-venv`, one run at a time, or two
when stage 1 shows the gain of the two-at-once rule (section 9.2, S6). Up to 8 processor lanes, each pinned with
`taskset`, with `OMP_NUM_THREADS=1` and `PYTHONDONTWRITEBYTECODE=1`. `/root/bca-shift` is only read. Outputs go to `runs/shift_hosts/`
(git ignores them); results to `outputs/shift_hosts/results.json`, built only from run records whose fingerprint is the
approved one.

**The driver's command line, fixed now.** The real study is started only as section 18 shows: `stages start` with
`--stage0-card-hours 0 --stage0-processor-hours 0` (stage 0 is booked separately, section 13), card runs on cores 0-1 and lanes on
cores 2-9 (the defaults). The operator's only other options answer a stop or an interruption: `--accept-s5 RUN_ID`
(an S5 on a BCA setup traced to the setup, not to a package defect; section 9.1), `--resume-after-stop` and
`--replace-incomplete RUN_ID ...` (the named runs that died without a record, each only once its process is gone). The
options `--choice19`, `--learners`, `--qualities`, `--seeds`, `--card-device`, `--card-cores`, `--lane-cores`, `--root`,
`--caps`, `--inject` and `--dry-run` exist for the dry run and the tests and are not used on real data. On real data
the driver reads the learners, qualities, data seeds, choice 19 and the S9 signature from `prespec.json` and refuses
any option that would change them or the limits (a root other than `runs/shift_hosts`, `--caps`, `--inject`, more than
2 card cores or 8 lanes); before it writes anything it also requires the approval, stop rule S0 on exactly this code
and stage 0's compute entry. Its state records its mode and configuration, and a resume with another one, or a driver
of the other mode, is refused; one driver runs per root (a lock), and a dry run never uses the real root. The
driver's log records the learners, qualities, seeds, device, cores and choice 19 it ran with.

## 9. Stop rules

### 9.1 What a stop means (plan 6.1)

- Any stop rule halts the stage driver: it launches nothing new, lets running children finish, writes
  `driver/STOP.json` and `driver/STOP.txt`, and exits. Nothing is retried silently; a restart over a stop report is
  refused unless the operator says so, and a recorded stop is never rerun.
- **A failure in Unifloral's own code is an outcome, not a defect.** S5 on a learner-alone or room-reference run is
  recorded as that learner's Claim 0 failure, enters every Holm family with p = 1, is never "fixed", and the driver goes
  on.
- **S5 on a BCA setup** halts until it is traced. It stops the study only when traced to a defect in the new package's
  code. Otherwise it is that setup's outcome (recorded with the driver's `--accept-s5`): its return counts as not
  finite, and that learner's claim enters with p = 1.
- **Any change to frozen code, settings or choices after your approval** makes a new, labeled pre-specification
  version. It runs only after you approve it with `approve_hosts.py` at your own terminal, never through a chat reply,
  never by an agent. Its write-up says whether any return had been read before the change. S1 enforces this.

### 9.2 The rules, with their exact thresholds

"Bit for bit" means equal `tobytes()` of every leaf of the learner state, the optimizer states, the key carry and the
per-step outputs. "Relative difference" of a leaf is max|a − b| / max|b|.

| Rule | Stops if |
|---|---|
| S0. Faithful copies | Any of tests (a1)-(i) or the guard's comparison fails (plan 4.5), with (a2)'s card exception: on the card, every leaf within 1e-6 relative after 60 steps is a recorded rounding difference (then S10 is a readout and choice 19 applies); anything larger stops. "To float32 rounding" is a relative difference of at most 1e-5. Test (d) is not needed (choice 11). Checked before stage 1 on the frozen bytes: the stage driver refuses to start on real data unless `runs/shift_hosts/stage0/SUMMARY.json` verifies, says the suite passed, and was made on exactly the code on disk (`gate.require_s0`) |
| S1. Pinned code | At the start of any run: an imported Unifloral file's SHA-256 differs from `configs/unifloral.json` (the bytes hashed are the bytes executed); or compiled bytecode (`__pycache__`, `*.pyc`) lies beside the hashed code; or the fingerprint (`prespec.json`'s SHA-256 plus the code manifest: every file of `experiments/shift_hosts/`, whatever its name or suffix, but `prespec.json` and the approval files at the package root, and the repository code it runs, `gate.EXTERNAL_CODE`: Unifloral's four files, its Dockerfile and requirements, `configs/unifloral.json`, `calibration/*.py`, `algorithms/*.py`, `runtime/*.py`, `experiments/__init__.py`, `experiments/shift/*.py`, `experiments/shift_iw/*.py` and `launch.sh`, `experiments/truth/*.py`, `experiments/signal/*.py` and `outputs/shift/oracle/oracle.py`) differs from the approved one; or the approval file, its sidecar and `prespec.json` are not committed in git HEAD as on disk, under your GitHub address (amendment A12). The driver re-verifies the approval before every unit and halts on a run record of another fingerprint |
| S2. Weights of 1 change nothing | For any learner, in stage 1 (seed 0, real data, 15,000 outer steps), IW with every weight 1 differs from standard in any bit after the refresh: the final key, learner state, BCA state (all but the carried refusal rule), any per-step output; or at the refresh R(1) (grid [1.0]) differs from the stored standard threshold in float32 |
| S3. Threshold reproduction | At any IW refresh, the bank scores recomputed from the stored state with the refresh key fail to reproduce the stored standard threshold bit for bit (threshold, λ̂, λ_hpd and n_eff, with the refresh's own posterior stream) |
| S4. Shared start | A learner's setups differ in initial-parameter hashes (its room reference included), or its alone, standard and IW setups differ in the hash of Unifloral's key carried out of the chunks ending at outer steps 2,500, 50,000 and 100,000 (in stage 1, also the weights-of-1 and repeat runs, at 2,500). A stopped run is compared only before its first bad step. Checked after stage 1, after every seed and after the room block |
| S5. Numbers | Any non-finite loss, metric, parameter, dose, weight or return; a NaN threshold (R = +∞ is a counted refusal, decision 3); a dose outside [1, 1.5] (ReBRAC: outside its [1, 2] check); any false per-step validity flag (section 5.3); a refresh failing BCA's own validity checks. A failed run is recorded, never silently retried; what follows is section 9.1. A non-finite scored return (any bank of a scores file) is that learner's Claim 0 outcome for an alone or room run, and halts the driver for a BCA setup unless accepted with `--accept-s5` |
| S6. Compute | After stage 1: (a) stage 1's measured speeds, at the chosen number of runs sharing the card, project stages 1-2 of a quality, with its totals so far, above its approved cap; or (b) a learner alone's stage-1 time per outer step is more than twice its random-array reference (TD3+BC 0.6 ms, CQL 2.1, IQL 0.33, ReBRAC 1.6; both one run at a time, since stage 1 runs each learner alone by itself, so the limit is never scaled by a two-at-once slowdown). During stage 2: an alone run takes more than twice its stage-1 time per outer step at the chosen concurrency (the stage-1 time times the measured slowdown of a run sharing the card, when two run at once). A resumed driver reuses stage 1's S6 decision while stage 1's records are unchanged. Stop and re-cost |
| S6b. Budget, as it runs | Before each unit of a quality (the sham-alone block, each seed, the room-reference block): the running totals plus the scoring work already queued plus the unit's projected cost would pass the quality's approved cap. The driver stops between whole units, never inside one; a resumed driver projects only the runs of a unit that have no record yet, and skips the check for a unit with nothing left to run. A card-hour is one hour of wall-clock time with at least one card run active (the union of card-run intervals on the monotonic clock, so two runs at once never count twice); a processor-hour is one hour of measured processor time (user plus system) summed over all processes, the driver's own included, which counts a card run's busy waiting as it really is. A child still running when its driver ended is counted to the driver's last heartbeat, and a run that finished unwatched is counted from its own record |
| S7. Float32 check | Open loop, by file, on the processor: the float32 restatement in `actors_f32.py` and the learner's own float32 network disagree by more than 1e-7 on any saved state (CQL with the same ε). Closed loop, in the scoring environment: that restatement and the float64 scoring adapter disagree by more than 0.05 points on any start (CQL with the declared noise; version 2's post-hoc limit, section 19; version 1's was 0.01). Run on every stage-1 actor and on every seed-0 final actor of the alone, standard and IW setups; stage 1's checks and the end of each quality read every result again, and a resumed driver reads a finished lane's result before skipping it |
| S8. Card repeatability | The two identical 15,000-step runs of a learner's IW setup (stage 1, seed 0) differ in any bit: final key, learner state, BCA state, any per-step output or any refresh's threshold |
| S9. Recorded environment | A card run's environment signature (`stages.env_signature`: Python, executable, package versions, card name and driver, the recorded environment variables, `XLA_FLAGS` in effect, matrix precision, JAX's platforms, backend and devices, Unifloral's revision and manifest SHA-256) differs from `prespec.json`'s `environment.s9_signature` in any key either names. Checked by each real run before its first chunk and again on its record; the driver refuses to start without a complete signature |
| S10. Equal before the first refresh | At outer step 10,000, just before the first refresh, a BCA setup's learner state (Unifloral's parameters and optimizer states) or key differs in any bit from its learner alone's (stage 1, seed 0, real data). For TD3+BC and IQL this is a readout, with the snapshots' leaf-by-leaf differences, not a stop (decision 2); for CQL and ReBRAC it stops |
| S11. Data copy | Any copied weighting-test file's SHA-256 differs from its manifest, or a prepared array's row counts or recorded hashes differ from the weighting test's recorded ones (section 3); checked at the copy, at preparation and when each run reads its prepared file |

**The two-at-once rule** (part of S6's decision): stage 2 runs two card runs at once only if, in stage 1, the aggregate
throughput of two runs at once (the IW setup's weights-of-1 and repeat runs together, outer steps per wall-clock second)
is at least 1.15 times one run's (the IW run alone), for the four learners together, and stop rule S8 held for every
learner; otherwise one run at a time.

## 10. Claims, kill rules and bars (written before any result)

Everything is judged per quality, on data seeds 0-4, on the strong-mix bank unless stated, with Δ, L, the upper bound
and Holm's correction as in section 11. 1.8 return units is version 4's 3 points (`iw/SPEC.md:168`), a size bar chosen
now for both claims. No mark moves after any outcome is seen.

| # | Claim | Shelved (kill) or flagged if |
|---|---|---|
| 0 | The learner learned (a gate for reading its BCA rows) | The learner alone's mean over the 5 seeds is below the quality's bar on the no-shift bank, or any seed's return of the learner alone is not finite, or a run of Unifloral's unchanged step (alone or room reference) stopped (S5). Bars: the recording controller's no-shift points minus 10, rounded: medium-expert 60 points (from 69.97; about −341.2 return units), medium 39 points (from 48.66; about −501.9 return units), converted to return units through the no-shift bank's anchors in `start_banks.json` (−800.29 and −35.19, rounded; the code uses the file's exact values). Exactly at the bar passes. The rows then read "did not learn", Claims 1-6 are not judged for it, and it stays in both Holm families with p = 1. No seed is ever dropped |
| 1 | Standard BCA beats the learner alone on the strong mix | Shelved if Δ(standard − alone) < +1.8, or L ≤ 0, or it fails Holm across the four learners. A TD3+BC pass on medium-expert is labeled "expected partly by selection; needs seeds 10-14" |
| 1c | Standard BCA costs return on the strong mix (a flag) | Flagged if Δ(standard − alone) ≤ −1.8 with upper bound < 0 |
| 2 | IW BCA beats standard BCA on the strong mix | Shelved if Δ(IW − standard) < +1.8, or L ≤ 0, or it fails Holm, or Claim 4a fails for that learner. Expected to be shelved for all four learners |
| 2c | IW costs return against standard on the strong mix (a flag) | Flagged if Δ(IW − standard) ≤ −1.8 with upper bound < 0 |
| 3 | IW costs nothing without the shift (a flag) | Flagged if, on the no-shift bank, Δ(IW − standard) ≤ −1.8 with upper bound < 0 |
| 3b | Standard BCA costs nothing without the shift (a flag) | Flagged if, on the no-shift bank, Δ(standard − alone) ≤ −1.8 with upper bound < 0 |
| 4a | IW gives the band its promised coverage under the shift (read before Claim 2) | At the IW setup's last refresh, on strong-law test steps outside the ball, each step at its own threshold, the pooled coverage (the mean of the 5 seeds' shares) lies outside [89%, 96%] (inclusive bounds) |
| 4b | The shift breaks the standard band (a label, not a gate) | The IW setup's band at its standard threshold covers 88% or more of the same steps. Claim 2 is then still judged, but a pass is labeled "gain from extra width, not repaired calibration" |
| 5 | BCA adds no physical failures | Flagged as harm if a BCA setup's F1 or F2 rate exceeds its learner alone's by 2 percentage points or more on any of the 7 banks (compared exactly on start counts), with L > 0 |
| 6 | The gain needs the band's placement, not just more caution (stage 3, only where Claim 1 or 2 passes) | Shelved if Δ(setup − its flattened twin) < +1.8 or L ≤ 0; for Claim 2 also if Δ(IW − shuffled IW) < +1.8 or L ≤ 0. The gain is then credited to the amount of caution |

**Failure events** (plan 5.4; fixed now, using no learned policy). F1, worse than doing nothing: the episode's return is
strictly below the zero-command controller's return from the same start. F2, late peak: from step 10 on, the largest
distance of the five floats from rest (the norm of their 10 position and speed numbers, as `oracle.py` defines the late
peak) is strictly above the 99th percentile of the same quantity for the stall-aware controller on the same bank.
Only these events may be called "harm".

**Reading order, fixed now.** First S0-S11 (with S6b). Then, per quality: Claim 0, then 4a and 4b, then 1, 1c, 2, 2c,
3, 3b, 5, and 6 only if stage 3 runs.

**Holm family membership, fixed now** (plan 6.2):
1. Every learner stays in each family of its quality. A learner that fails Claim 0, has a non-finite return on any
   seed in either setup the claim compares (section 17: an IW failure does not touch Claim 1), or (Claim 2 only) fails
   Claim 4a enters with p = 1. It fails and stays in, so it never loosens the bar for the others.
2. Claims 1 and 2 are judged only when all 5 seeds of all four learners of that quality (alone, standard and IW) are
   complete; until then they read "pending". A stop at the cap leaves them pending, never judged on fewer seeds. Every
   other claim waits for the 5 seeds of its own inputs.
3. Runs go seed by seed across learners (section 8), so a stop leaves complete seeds.

**Badges**: passed, shelved, flagged, pending, not judged, did not learn. A flag whose condition does not hold reads
"passed" with the verdict "not flagged"; Claim 4b reads "passed" when the shift breaks the standard band (below 88%).
Flags and Claim 6 are not corrected for multiple comparisons: they are pointers, not evidence.

**Claim 2's pointer, never a gate.** Beside Claim 2: the mean dose change from standard to IW thresholds over kept
training rows at the last refresh, times Δ(standard − alone) / (mean standard dose − 1); for IQL the same with its actor
weights (the change in BCA's weight, over BCA's weight minus Unifloral's). It uses outcomes, so it never gates.

**Study rule.** If Claims 1 and 2 are shelved for all four learners on every quality run, the study ends after seeds
0-4, with no stage 3 and no confirmation.

**Write-ups.** Every shelved claim and every "did not learn" uses your format: The idea / What we tested (and the kill
rule) / What happened / Why.

**Predictions, stated now so that surprises show** (plan 6.3): TD3+BC in the reference row's direction, a gain on
medium-expert and a loss on medium; ReBRAC |Δ| < 1.8 on both qualities; IQL small on medium-expert, a small cost
possible on medium; CQL uncertain, with Claim 0 a real risk; IW minus standard within ±1.8 for every learner and
quality, and IW's coverage in 89-96%. IQL's Q and V are expected to be identical among its standard, IW and sham-alone
runs on the card, and to drift at rounding level against its imported alone run (amendment A7).

## 11. Statistics (plan 5.5) and metrics (plan 5.4)

**Metrics, per setup, bank and seed** (final snapshot, with the 50% snapshot shown): return (the claims' measure),
points, the discounted return, regret against the exact best plan (start by start, and its mean), paired standard
errors over the starts, the stall share, the loss from unrecorded throttle (the same policy capped at 80% minus
uncapped), the failure events F1 and F2 (section 10), the worst-10% mean, the 10th percentile, tracking and energy
costs; for BCA setups the band readouts (certified refreshes, refused rows, mean dose or IQL's weight shrinkage, the
design effect, IW's dose change) and coverage at the last refresh (strong-law test steps inside and outside the ball,
standard against IW, and at the three kinds of pairs); CQL's tanh-of-the-mean score; ReBRAC's substituted next
commands; IQL's Q and V agreement; the learner states at 10,000 against alone (S10); compute and code hashes in every
run record. Only return enters a claim, except coverage (Claims 4a and 4b) and F1 and F2 (Claim 5).

- **Per seed**: a setup's mean return over a bank's 1,000 starts (return units: the plain 100-step return, closer to 0
  is better).
- **Δ**: the paired per-seed difference between two setups of one learner, averaged over the 5 seeds; sd over the seeds
  with n − 1.
- **L and the upper bound**: one-sided 95% bounds, mean ∓ T95 × sd / √5 with Student's t on 4 degrees of freedom,
  T95 = 2.13184678632665 (the plan's 2.132, computed exactly). L > 0 exactly when the one-sided p < 0.05.
- **One-sided p**: P(T4 ≥ mean / (sd / √5)); when sd = 0, p = 0 if the mean is above 0, else 1.
- **Holm's correction**: within each quality, Claims 1 and 2 each form one family of the four learners, on one-sided
  p-values. The p-values are sorted (ties by learner name); the i-th smallest passes if it and every smaller one are at
  most 0.05 / (4 − i + 1): 0.0125, 0.0167, 0.025, 0.05. The page shows the adjusted p, the running maximum of
  min(1, (4 − i + 1) p). The medium families are separate, so medium changes nothing about medium-expert. Each family
  keeps the chance of any false pass at 5%; across two families per quality the chance of at least one false pass is up
  to about 10% with medium-expert alone and about 19% with medium too.
- **No partial statistics**: no mean, bound or p-value is shown until exactly 5 finite seeds exist; per-seed values are
  shown as they arrive.
- **Power** (plan 5.5): with Holm's correction a true gain of 1.8 return units passes about 6-10% of the time on
  medium-expert, and about 5-9 return units are needed to pass reliably (6.75-9.25 if only one learner has the gain,
  5.25-7.25 if all four share it). So a shelved claim means "no effect large enough to see with 5 seeds", not "no
  effect". The result that motivated this plan (+6.93, p 0.0128) would pass Holm's
  first step only beside another learner's pass. If card rounding enlarges the seed spreads (choice 19), power is lower.

## 12. Budget caps (as approved)

| Quality | Covers | Card-hours | Processor-hours |
|---|---|---|---|
| medium-expert | stages 1 and 2a (stage 0 is booked separately, section 13) | 14 | 21 |
| medium | stage 2b (approved separately) | 12 | 16 |
| stage 3, per quality | the controls, only where a claim passes, in the fixed order of section 8 | 4 | 6 |

The caps are checked by S6 after stage 1 and by S6b before each unit (section 9). The first two rows are the
constants of `budget.CAPS`, which the driver checks; stage 3's code is not in this package, so its cap binds the later
version that runs it. The plan's estimates were about 7.8 card-hours (high 12.2) and 12.1 processor-hours (high 18.4)
for stages 0-2a, and 7.1 (high 10.6) and 9.8 (high 14.7) for stage 2b (plan 7.3). Changing a cap needs your yes. You
approved the caps with stage 0 inside the medium-expert one; on 16 October you decided to book stage 0 separately
instead (section 13). The numbers of the caps are unchanged.

## 13. Budget re-cost

- **Your decision (16 October 2026, about 13:45 UTC, in chat): stage 0 is booked separately.** Its compute is recorded
  here as spent and is not counted against the caps of section 12. The driver's first start therefore enters stage 0
  as zero: `--stage0-card-hours 0 --stage0-processor-hours 0` (section 18). The caps themselves are unchanged: 14
  card-hours and 21 processor-hours for medium-expert's stages 1 and 2a, 12 and 16 for medium's stage 2b, and 4 and 6
  per quality for stage 3.
- **Why.** Building and testing the package (stage 0) used far more than the plan's line of 0.05 card-hours and 0.5
  processor-hours. Counted inside the medium-expert cap, it would leave stages 1-2a over the processor cap even at the
  central estimate, and S6 would stop the study right after stage 1. The real runs themselves cost about what the plan
  said.
- **Stage 0, spent (booked separately):** about 4.2 card-hours and 17 processor-hours through 16 October 14:40 UTC
  (range 3.8-4.5 and 15-19): the budget re-cost's 3.3 and 11.2 through 08:17 UTC (measured items and estimates per
  build area), the review-and-fix pass's 0.29 and 2.3, and the export check's dry run of about 1 processor-hour. The
  final suite pass on the frozen bytes comes on top; its own measurement is in `runs/shift_hosts/stage0/SUMMARY.json`.
- **Re-costed projection** (the plan's cost model, `wf6/understand/env/cost_model.py` as corrected in
  `wf6/plan/checks/recost.py`, with the measured speeds below; one card run at a time):

  | Part | Card-hours, central / high | Processor-hours, central / high | Cap |
  |---|---|---|---|
  | Stage 1 (medium-expert, seed 0) | 0.62 / 1.13 | 1.28 / 2.48 | |
  | Stage 2a (medium-expert: sham-alone runs, seeds 0-4, room references, exports) | 6.62 / 10.07 | 8.21 / 13.97 | |
  | **Stages 1-2a** | **7.25 / 11.2** | **9.5 / 16.46** | 14 / 21 |
  | **Stage 2b** (medium, the same runs) | **6.62 / 10.07** | **8.21 / 13.97** | 12 / 16 |
  | Stage 3, one learner's Claim 2 twins | 1.61 / 2.48 | 1.78 / 2.98 | 4 / 6 per quality |

  Both qualities fit their caps at the high estimate. If the card throttles hard (clocks of 1-2 GHz, seen while heavy
  processor work ran beside it), stages 1-2a come to about 13.8 card-hours and 19.1 processor-hours, still inside 14
  and 21, and stage 2b to about 12.6 and 16.5, just over 12 and 16: S6b would then stop medium between whole seeds.
- **Measured speeds** (16 October, clean card benchmark on made-up arrays, one core per card run as the driver pins
  it; ms per outer step, alone and with BCA after its first refresh): TD3+BC 0.58 and 1.26, CQL 2.06 and 3.20, IQL 0.31
  and 1.13, ReBRAC 1.56 and 2.89. BCA's extra cost per step is 0.7-1.3 ms, against the plan's 0.4. An IW threshold
  table on a 900-row bank takes 3.5-5.2 s. Keep heavy processor work off the laptop while card runs are going.
- **Lane costs before measurement.** `budget.DEFAULT_LANE_SECONDS`, which S6 uses for a lane kind that has not yet run,
  was lowered before the freeze to 45, 90, 120 and 70 processor-seconds for score, coverage, S7 and viewer lanes (the
  score figure from test_scoring's measured 59.6 s for 5 actor readouts on all 7 full banks; the others are
  allowances). The first defaults, 120, 180, 300 and 90 s, added about 6 processor-hours of projection that no
  measurement supports. Once a lane of a kind has run, its measured time replaces the default.
- **Version 2 (section 19.9).** Version 1's stopped stage 1 used 0.2785 card-hours and 0.4090 processor-hours. They
  stay counted against medium-expert's cap in the kept ledger, and stage 1 runs again on top of them. The version-2
  stage-0 suite is stage-0 work, booked separately. The caps are unchanged.

## 14. Seeds: exploratory now, reserved for a later confirmation (plan 5.3)

- **Two seed namespaces.** This study's data come from the weighting test's generator, master seed 20261016
  (`shift_iw/world.py:62`); the main study's from master seed 20261012. The same seed number gives different data in
  the two.
- **Now**: data seeds 0-4 of the weighting test's generator only. The weighting test has already seen them.
- **Reserved**: data seeds 10, 11, 12, 13 and 14 of the weighting test's generator (master seed 20261016), for a later
  confirmation of any claim that passes here. They are separate from version 8's main-study seeds 10-14. Nothing in
  this study creates, reads or trains on them. New data for them come from the same code (about a minute per seed),
  under a confirmatory pre-registration that keeps these pass marks and that you approve at your own terminal with a
  commit.

## 15. What is exploratory

All of it. The data seeds 0-4 were seen by the weighting test, and medium-expert was chosen after seeing that test's
TD3+BC result, the best of 5 qualities (one-sided p 0.0128; about 0.064 after allowing for the choice among 5; plan
2.1). No claim from seeds 0-4 is confirmatory: a pass is a pointer toward a confirmation on seeds 10-14, never evidence
by itself. `results.json` and the page carry this label. Holm's correction covers only Claims 1 and 2 per quality;
everything else (7 banks, many metrics, flags) is a pointer, not evidence. Nothing here speaks to the main study's data
(plan risk 11).

## 16. Amendments made while building (declared before any run)

Each changes something the plan fixed. All were made before any run on ThrusterChain data, and every one that touches
BCA's arithmetic was found and tested on made-up arrays. Implementation details that change nothing the plan fixed are
left out; the build reports list them.

- **A1. CQL's two compiled programs** (plan 4.3, 4.5 test (a2)). Decision 1, section 5.5. Reason: the single program's
  1.55e-4 card drift at a dose of exactly 1 would have failed S0; the pre-refresh program is exact while every dose is
  1.
- **A2. IQL's split validity** (plan 4.1, "a step whose BCA inputs are not valid is discarded whole, random key
  included"). Built: Unifloral's key and agent state are kept unless the learner side is valid (valid weight inputs,
  finite weights, finite new state and losses); BCA's part is kept unless the whole step is valid. Reason: selecting the
  whole carry on a flag that depends on the band fit made XLA compile Unifloral's own arithmetic differently on the
  processor (1 rounding unit at step 1, 1.7e-5 relative after 60 steps), failing (a2); the split restores bit for bit.
  Effect: if only the band fit fails, the learner's whole update of that step stands, the actor's included. The IQL
  port also keeps its Q and V update then (`algorithms/iql_bca.py:220-226`) but skips the actor's
  (`iql_bca.py:265-280`), so the two differ only on such a step; it is still flagged and S5 stops the run there, so no
  reported result depends on it. Decision 4.
- **A3. S5's threshold condition** (plan 6.1 S5). A NaN threshold stops; R = +∞ is a counted refusal. Decision 3.
- **A4. BCA's fit keys keep the ports' fold constants** (plan 3.4, which names only `fold_in(PRNGKey(s91), outer
  step)` for the scale network's start, the target noise and the bootstrap masses). Built: within each step's fit key
  the port's own fold constants separate the target noise and the bootstrap masses, and the scale network starts from
  `fold_in(fold_in(PRNGKey(s91), 0), the port's init constant)` (step 0 is never a fit step). Reason: the same
  treatment the plan gives the refresh constants, and it lets test (c1) compare the fit cores with the ports'. The
  stream stays separate from Unifloral's key and shared by the standard and IW setups.
- **A5. A processor flag** (plan 4.4 choice 8 names the card's flags only). Training-environment processes on the
  processor also set `--xla_cpu_multi_thread_eigen=false`, one thread per pinned process, as the environment report's
  timings did. Card runs are unchanged.
- **A6. Safety checks carried over from the ports** (plan 3.1, "each learner's safety checks, carried over from its
  port"). Not carried over: ReBRAC's check that stored statistics equal the scale network's (`rebrac_bca.py:262-263`),
  because here the statistics live only inside the scale network and cannot disagree; and the IQL port's zeroing of
  unsupported rows' inputs (`calibration/iql_state.py:47-48`), because those rows carry weight exactly 0 and a
  non-finite row already makes Unifloral's own losses non-finite (an invalid step). IQL's skip rule is computed, then
  not kept (the old actor, its Adam state and step count stay), instead of the port's conditional computation; the
  effect is the same and Unifloral's actor arithmetic stays untouched. Added: ReBRAC's flag also checks the batch, the
  old state's finiteness and the reference's validity; TD3+BC's and CQL's include the reference's validity.
- **A7. The IQL readout's expected answer** (plan 3.7, 5.4 and risk 7: "Q and V bit-identical across its three runs,
  expected yes"). Now: identical among standard, IW and sham alone (one compiled program); on the card, a rounding-level
  drift against the imported alone run from step 3 on (3.8e-7 relative after 60 steps), which may grow over 100,000
  steps; on the processor all three agree bit for bit. Reason: IQL's stage-0 card diagnosis. The comparison stays paired.
- **A8. Where the copies are** (plan 3.1, "copied into `/root/bca-hosts/runs/`"). They are under
  `runs/shift_hosts/source/` (`runs/shift_iw_v5/<rest>` → `source/<rest>`; `runs/shift_iw_v5_explore/<rest>` →
  `source/explore/<rest>`; `RESULTS.json` → `source/explore/RESULTS.json`). Reason: the build allowed writes only under
  `runs/shift_hosts/`. The bytes are identical and checked (S11). Also copied, beyond the plan's list: the band records
  the estimated-weight classifier must match, and the reference row's BCA evaluations. The weighting test's
  `RESULTS.json` has no manifest or sidecar, so S11 checks it by its git blob in HEAD instead (evaluation files are
  hashed as bytes and never parsed before your approval).
- **A9. Work moved to stage 0** (plan 7.3 put the data copy and preparation in stage 1, and the regret solves in stage
  2a). They ran in stage 0, with no learning: the data copy (S11), the preparation of both qualities, and the scoring
  cache (regret plans of the fixed best-plan solver, the viewer's fixed starts from the stall-aware controller, CQL's
  noise). Their compute counts in stage 0's ledger entry (section 13). The regret solves cost 0.55 processor-hours,
  about twice the plan's estimate.
- **A10. S6b also before the sham-alone block and the room-reference block** (plan 6.1, "before each new seed").
  Reason: both are whole units too, so a stop still falls between whole units.
- **A11. The terminal approval block** (plan 10, item 15). The push from WSL is
  `GIT_CONFIG_GLOBAL=/root/.bca-gitconfig git push origin claude/shift-hosts`, because a plain `git push` as root fails
  on this working copy (git's ownership check; tried with `--dry-run`). `approve_hosts.py` prints the full block,
  with the fingerprint's first 16 characters in the commit message (section 18).
- **A12. S1's reach** (plan 6.1 S1: "a restated or package file's hash differs from the frozen pre-specification").
  Built stricter: the fingerprint also covers the repository code the package runs or its stage-0 tests import
  (section 9.2, S1), `experiments/signal/*.py` included after an import audit showed that scoring loads it through
  `experiments/shift`; and every run checks that the approval file, its sidecar and `prespec.json` are committed in git
  HEAD as on disk, under your GitHub address. Reason: a change in any code a run executes must change the
  fingerprint, and the approval must name a commit anyone can check out and re-hash.
- **A13. The page's files** (plan 3.1 module table: `export.py` writes "`results.json`, plus the episode and actor files
  for the page"; plan 9.2). Built: `export.py` writes `results.json` only. The inputs of the viewer are recorded by the
  runs themselves: the scoring lanes keep the viewer's episodes for seed 0's final actors, `band_readout`'s viewer lane
  writes the band arrays, and every actor file has a SHA-256 sidecar. The page's viewer files are formatted from these
  after the runs; nothing a claim reads depends on them. Reason: the viewer's file formats were defined in the claims
  report, but their writer was not built in stage 0.

## 17. Readings where the plan was silent or open (frozen here)

**Stop rules and the driver.**
- S2's caveat: the weights-of-1 run carries IW's refusal rule. If a stage-1 refresh ever refused (R = +∞; it never did in
  the weighting test), its rows would get dose 1.5 where standard gives 1, and S2 would differ by design; the driver
  still stops, and the stop report says so (the S2 check lists the uncertified refreshes and carries that note).
- The stage driver, beyond the plan's words: a lane's own verdict (S7's result, a non-finite scored return) is read
  again when a resumed driver finds the lane done, so a failed check never passes on resume; a halt also stops a lane
  from starting its next step; an unexpected error or SIGTERM writes the stop report too; a run that died without a
  record restarts only when named (`--replace-incomplete RUN_ID`) and only once its process is gone.
- S3 recomputes the bank scores through the refresh's own computation (TD3+BC's `recompute_bank_scores` reuses the
  refresh's compiled program; the others use `band_parts`, the refresh's own computation), because a separately compiled
  recomputation differs by float32 rounding (7.8e-7) and would fail falsely. The run record says whether the recomputed
  scores equal the refresh's own bit for bit.
- S4's and S10's hashes are taken on the key carried out of a chunk, before the evaluation split.
- S6's speeds: the steady time per outer step is the median over chunks 2 onward of each chunk's time less the compile
  time JAX reports inside it, so CQL's start-up recompiles do not count as training. Two at once is decided once, for
  the four learners together; per-learner gains are recorded.
- S6's stage-2 alone check (more than twice its stage-1 time per outer step) reads the plan's "stop and re-cost" as
  applying during stage 2. Heavy outside use of the shared card can trigger it.
- S7's saved states are the viewer's fixed starts on all 7 banks as the float64 adapter visits them (about 7,400 rows).
  The adapter-to-restatement check (within 1e-5) is judged on states in ThrusterChain's ranges.
- The gate covers every run on real data, the learner alone included, every scoring of a learned policy, and the reading
  of the reference row's learned results; actor files are scored only with the made-up token on made-up files or with
  your verified approval.
- Default cores: card runs on cores 0-1 (one per run at once), scoring lanes on 2-9.
- Stage 2 of each quality starts with its sham-alone block (plan 5.1 fixes seed by seed with the room references last;
  it does not place choice 19's runs).
- The driver takes no external lock for the card. Plan 7.4 asks for one only if one is in use on this laptop. The
  repository's `train.py` takes an exclusive lock on `BCA_GPU_LOCK`, which the README sets to
  `/home/dbayha/bca-work/resource-locks/local-rtx5070ti.lock`; in this laptop's WSL that folder does not exist and no
  process held a file lock on 16 October, so none is in use. The main session checks again before stage 1 that no other process is using the card.

**S0's tests: what "equal" meant** (all passed on the final stage-0 suite).
- Within one compiled program, comparisons are bit for bit. Where the two sides are compiled separately (a step's dose
  against an outside formula in (b2) and (b3), the eager refresh against the compiled step), "equal" is within one
  float32 rounding step (at most 1e-6 relative), because the compiler fuses multiply-adds differently; the threshold
  lookup itself is bit for bit.
- Test (c3): the effective sample sizes (n_eff, Kish) are compared through each side's own sum and dot parts, since
  `np.dot` comes from different OpenBLAS builds in the two environments; everything else is bit for bit.
- Test (a3) added cross-checks the plan did not name (TD3+BC against `td3_alpha` = 2.5 / 1.5; ReBRAC at cloning weight
  100, where a misplaced dose would be visible).
- Devices: (a1) and (a2) ran on the processor and on the card, as the plan says; TD3+BC's (a4), (b2), (b3) and refresh
  tests ran on the processor only (the plan names the processor for (a4) and no device for the rest).
- Test (e)'s bank rule (ReBRAC's rolled next commands within the cut calibration episodes) is checked on the real
  prepared arrays in data preparation, which compares arrays and does no learning, as the plan allows; on made-up
  arrays only the training and strong-law rolls are checked.

**Snapshots, coverage and readouts.**
- Snapshots at 10,000 (S10's readout of differences), 50,000, 95,000 (the last refresh; coverage needs that state and
  band) and 100,000. Each is the state the next chunk starts from.
- Coverage at the last refresh, at three kinds of pairs (recorded state and command; recorded state and the actor's
  command; the actor's visited states, first 200 strong starts, and its commands), with exact next states from the
  simulator. CQL's command at recorded strong-law states uses the strong bank's declared noise at the test step's own
  episode and step; ReBRAC's a′ at actor and visited rows is the stored actor's command at the next state; the IW arm
  refuses in-ball rows. Ties count as covered.
- Pooled coverage for Claims 4a and 4b is the mean of the 5 seeds' shares. Claim 4b's "standard threshold" is the IW
  setup's own band at its standard threshold, on the same steps as 4a; the standard setup's band is shown beside it.
- The highest-current tenth is the rows with current at or above the set's 90% quantile. Estimated weights are taken at
  the float64 of the stored float32 observation.

**Scoring and metrics.**
- Paired standard errors over the starts use the population formula (std / √n), as `learners.evaluate_policy` does; the
  discounted return is the simulator's 100-step discounted sum.
- F1 and F2 (plan 5.4): F1 is a return strictly below the zero command's from the same start; F2's "after step 10" is
  read as from state 10 on (the oracle's `LATE_PEAK_FROM = 10`), the 99th percentile by numpy's linear interpolation,
  and a policy fails F2 when its late peak is strictly above it.
- The TD3+BC and IQL adapters' spread is float64(std) + 1e-3 (at most 2.0e-7 from the float32 sum).
- The viewer's fixed starts: the starts whose stall-aware return is nearest the 5th, 15th, …, 95th percentile (numpy
  linear percentiles; ties to the lowest index), easy to hard, plus the page's start 11 for the four situations.

**Claims.**
- A run is "not finite" when it stopped as an outcome or any of its 7 banks' scored returns is not finite. For Holm,
  p = 1 applies only when one of the two setups compared in that claim is not finite on a seed (an IW failure does not
  touch Claim 1).
- A room reference's S5 fails its learner's Claim 0 (plan 6.1), although plan 5.1 says room references change no
  verdict. They run last, so Claims 0-2 can change until all 20 exist, and `results.json` says so.
- Claim 5's 2-percentage-point bar is compared exactly on start counts; L comes from the per-seed percentage points.
- Claim 6 reads "pending" for a passing claim until its stage-3 twins exist, and "not judged" where Claims 1 and 2 did
  not pass. Stage 3's setup names: standard flattened, IW flattened, IW shuffled, IW with estimated weights.
- The sham-alone drift (choice 19) is reported as the sham's strong-mix return minus the imported alone run's, seed 0,
  beside the claims; it changes no verdict.

## 18. What happens after approval

For version 2, section 19.8 replaces steps 1-3 below; steps 4-7 stand.

1. **You approve at your terminal** (plan 10, item 15, with amendment A11). In a Windows terminal, one line at a time:

   ```bash
   wsl -u root
   cd /root/bca-hosts
   PYTHONDONTWRITEBYTECODE=1 /opt/bca-venv/bin/python experiments/shift_hosts/approve_hosts.py
   ```

   It refuses unless `prespec.json` and every file of the code manifest are committed exactly as on disk. It shows the
   title, what you approve, the fingerprint and its first 16 characters, and asks you to type those 16 characters. It
   then writes `experiments/shift_hosts/approval_<first 16>.json` and its `.sha256` sidecar, nothing else, and prints the
   commands that follow:

   ```bash
   git add experiments/shift_hosts/approval_<first 16>.json experiments/shift_hosts/approval_<first 16>.json.sha256
   git -c user.name="David Bayha" -c user.email=35092758+dbayha7@users.noreply.github.com commit -m "Unifloral hosts: approve the pre-specification <first 16>"
   GIT_CONFIG_GLOBAL=/root/.bca-gitconfig git push origin claude/shift-hosts
   exit
   ```

   Then in PowerShell: `cd C:\Users\David\dev\BCA` and `git push origin claude/shift-hosts`. No agent runs any of this.
   Until that commit exists, every run on real data refuses.
2. **Before stage 1, the main session checks** (reads only, apart from the suite's own outputs under
   `runs/shift_hosts/stage0/`): the gate passes (`gate.require_run_approval()`); a full stage-0 suite
   (`run_stage0.sh`, about 65 minutes, made-up arrays only) passes on the frozen bytes (S0), since the last full pass
   ran before the review's fixes and before this document, `prespec.json` and `approve_hosts.py` joined the
   package (the driver itself refuses to start until `runs/shift_hosts/stage0/SUMMARY.json` was made on exactly the
   code on disk, `gate.require_s0`); the graphics driver is still 610.62 (S9); no other process uses the card
   (section 17); the card is idle and cool, since a card as hot as during the stage-0 suite would stop S6's stage-1
   alone check.
3. **The main session starts the stage driver**, as root in `/root/bca-hosts`, with `LD_LIBRARY_PATH` unset (section
   4.6) and stage 0 entered as zero (it is booked separately, section 13):

   ```bash
   env -u LD_LIBRARY_PATH PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 /opt/bca-venv/bin/python \
       -m experiments.shift_hosts.stages start --stage0-card-hours 0 --stage0-processor-hours 0
   ```

   It runs detached and writes `runs/shift_hosts/driver/status.json`, which the main session polls; `stages
   request-stop` asks it to halt gracefully. No other option is given (section 8, "The driver's command line").
4. **Stage 1** (about an hour, estimate), its checks, then **stage 2a** by itself, then **stage 2b** if every stop rule
   passed and every stage-2a lane finished (section 8).
5. **On a stop**: the driver writes `driver/STOP.json` and `STOP.txt`, and nothing is retried. The main session reads
   the report and tells you. A stop that needs a code or settings fix waits for a new pre-specification version and your
   terminal approval (section 9.1). An S5 on a BCA setup is traced; if no package defect is found, it is recorded as that
   setup's outcome.
6. **When a quality's runs are complete**: `export.py` builds `results.json` in the scoring environment, only from run
   records made by the approved code under this pre-specification; the claims are read in the fixed order; every shelved
   claim gets its write-up; the page gets the "Four learners" table and the episode viewer (plan 9), the viewer's files
   formatted from the recorded outputs (amendment A13).
7. **Afterward**: if Claims 1 and 2 are shelved everywhere, the study ends after seeds 0-4. Where a claim passes, stage 3
   needs a new pre-specification version (its code is not built yet) within its cap, and a confirmation needs fresh data
   seeds 10-14 of the weighting test's generator under a pre-registration you approve at your terminal.

## 19. Version 2: the change made after the start (post hoc)

This section is the write-up that section 9.1 asks of a new version. Everything in it was decided after version 1's
stage 1 had started and stopped, so none of it is pre-specified in version 1's sense, and the change it makes is
labeled post hoc wherever it is reported (section 19.10).

### 19.1 What version 1 ran

- **Version 1** was frozen on 16 October 2026 at 16:21 UTC (commit `2e59db0`, fingerprint
  `8c5e8421922dccebe5b86f47b44c8d86def2dbdec7da84ba78ba2ba807728179`) and approved by you at your terminal (approval
  file written 16:29:12 UTC, committed as `608eefa`).
- At 16:53 UTC the main session's pre-start checks of section 18 passed (the gate verified, S0 passed on the frozen
  bytes). The auto-mode safety check refused to let the main session start the driver, so you started it yourself, at
  about 16:58 UTC, in WSL `/root/bca-hosts`, with section 18's command (driver process 410).
- **Stage 1** (medium-expert, data seed 0, 15,000 outer steps) ran in its fixed order. TD3+BC, CQL and IQL completed
  all five setups (alone, standard, IW, IW with every weight 1, the IW repeat), and all 15 of their S7 lanes passed
  both halves. ReBRAC alone completed (exit 0, 49.2 s wall); its S7 lane failed. ReBRAC standard, started at 17:12:00,
  was running at the halt and was left to finish, as section 9.1 says (no process was left by 17:15; its record is
  checked on the laptop). ReBRAC's IW, weights-of-1 and repeat runs never started.
- In all, 17 card runs ended and 16 S7 lanes ran (15 passed, 1 failed). Stage 1's checks S2, S4, S6, S8 and S10 were
  never computed (the driver's `status.json` lists no checks).

### 19.2 The stop and its numbers

At 17:12:14 UTC the driver stopped: `STOP (S7): stage1/medium-expert/seed0/rebrac-alone/s7: the float32 check failed
(see its s7.json)`. Nothing was retried.

ReBRAC alone's `lanes/s7-step15000/s7.json`:
- **Open loop**: passed bit for bit. On all 7,400 saved states the restatement's commands equal Unifloral's own float32
  network's (`bit_identical_rows` 1.0, `max_abs` 0.0; limit 1e-7).
- **Closed loop**: failed on one bank, big kicks, and passed on the other six. Differences between the float32
  restatement and the float64 scoring adapter, over each bank's 1,000 starts (`tol_points` 0.01):

  | Bank | Largest \|points difference\| | Largest \|return difference\| | Mean \|points difference\| |
  |---|---|---|---|
  | none | 0.00023 | 0.0018 | 4.6e-7 |
  | mild | 0.00016 | 0.0016 | 1.0e-6 |
  | strong | 0.00084 | 0.0107 | 2.4e-6 |
  | recorded | 0.00035 | 0.00019 | 1.7e-6 |
  | strong current | 0.0024 | 0.0036 | 1.3e-5 |
  | big waves | 0.00046 | 0.00032 | 2.3e-6 |
  | **big kicks** | **0.01139** | **0.0114** | **6.5e-5** |

- `s7.json` keeps each bank's largest and mean difference, not how many starts passed the limit. The big-kicks mean
  (0.0645 points summed over 1,000 starts) allows between 1 and 6 starts above 0.01. "One start of 1,000", as the
  question to you put it, was an inference, not a record.
- The 15 passing lanes of TD3+BC, CQL and IQL: their largest big-kicks differences were 2.5e-5 to 2.5e-4 points and
  their largest big-waves differences 9.3e-6 to 7.9e-5 points (the display, cut at 900 characters per file, showed
  those two entries; the other five banks were not displayed).
- Compute at the stop (`STOP.txt`): medium-expert 0.2785 card-hours and 0.4090 processor-hours; medium none.

### 19.3 Your decision

On 16 October 2026 at 17:16:09 UTC, in chat, you chose the option the main session (Claude) had proposed and marked
recommended, "Investigate, then version 2 (Recommended)", described as: "I confirm that one episode is rounding
amplification, not a bug. Then I write pre-spec version 2, labeled post-hoc: S7's closed-loop limit becomes 0.05 points
per start, everything else unchanged. The test suite reruns and you approve at the terminal. Needs the laptop
charged." The question (17:16:03 UTC) summarized the stop as "ReBRAC: one start of 1,000 off by 0.0114 points vs the
0.01 limit"; the stop report and the per-bank numbers of section 19.2 were in the session's tool output (17:15:43 UTC),
not in a message to you. The question gave the limit in points only; section 19.5 converts it to return units (on the
strong mix, 0.05 points is 0.635 return units per start). On 17 October you said to go ahead with drafting version 2.

The value 0.05 was proposed by the main session after the failing value was known (it is 4.4 times 0.01139) and was
fixed by nothing before it, so the change is post hoc. It was chosen before the investigation of section 19.6 ran, and
that investigation's result does not move it (section 19.6 says what happens if it finds a defect).

### 19.4 What had and had not been seen before the change

- **Seen**: S7's difference statistics for 16 stage-1 actors (per bank, the largest and mean absolute points
  difference and the largest absolute return difference between the float32 and float64 versions of one policy; the
  open loop's command differences), the driver's log lines (wall-clock and processor seconds per run; each displayed
  card run ended "exit 0 (complete)"), its status and the budget totals. Two weak facts follow from these: every
  stage-1 run shown completed without a stop on its training numbers (no S5), and the S7-checked actors' returns were
  finite (S7's pass flag requires it, `evaluate.py:606`, and the failing bank's largest difference is finite), at
  15,000 steps.
- **Not seen**: no return, points level or score of any learned policy, and no training metric. Stage 1 writes no
  scores file: its only lanes are S7 lanes (`stages.py:877-882`). `s7.json` holds differences only
  (`evaluate.py:568-585, 605-609`); the returns behind them exist only in the lane's memory.
- **On the laptop, never read**: the 17 actors; their run records (per-chunk loss means, including the critics' value
  estimates); and S7's saved states and commands, from which 74 episode returns per actor could be computed.
- So **no return had been read before the change**, and no stage-1 return has been displayed in the session since;
  that none was read on the laptop is yours to confirm before you approve. The investigation of section 19.6 prints
  differences only and keeps it that way. The decision used only a float32-against-float64 difference of one policy
  with itself (its cause is the question of section 19.6); it carries no return level and nothing that orders the
  setups on any claim.

### 19.5 What the change can and cannot affect

- **What the claims read.** Every claim reads only the scored returns of the float64 adapter (`evaluate.score_actor`,
  through `policies.adapter`; sections 5.6 and 10). The float32 restatement appears only inside S7 (`evaluate.py:576,
  602`), and S7's closed-loop limit only in S7's own pass flag (`evaluate.py:606`). No claim's number (return, Δ,
  bound, p-value, Holm step, Claim 0 bar or badge) depends on the limit: each is computed the same way under either
  limit.
- **What it decides.** Whether the driver halts, as soon as any S7 lane fails (`stages.py:345-346, 946-952,
  989-990`): on the 20 stage-1 actors, and on each quality's 12 seed-0 final actors of the alone, standard and IW
  setups (`stages.py:876-879`); stage 1's checks and each quality's end read every result again (`stages.py:1043,
  1231-1250`).
  Seeds 1-4 get no S7 lane. And S7's status in `results.json`.
- **The new limit in return units.** S7 is judged in points on each bank's own scale; the claims are judged in return
  units, with a bar of 1.8 on the mean of 5 seeds (section 10). Per start, 0.05 points is (version 1's 0.01 in
  brackets):

  | Bank | Points per return unit | 0.05 points in return units | Version 1 |
  |---|---|---|---|
  | none (Claims 0, 3, 3b) | 0.1307 | 0.383 | 0.077 |
  | mild | 0.0993 | 0.503 | 0.101 |
  | strong (Claims 1, 1c, 2, 2c) | 0.0787 | 0.635 | 0.127 |
  | recorded | 1.871 | 0.027 | 0.005 |
  | strong current | 0.666 | 0.075 | 0.015 |
  | big waves | 1.416 | 0.035 | 0.007 |
  | big kicks | 1.003 | 0.050 | 0.010 |

  (The ratios are each bank's 100 / (top anchor − bottom anchor), and equal `s7.json`'s largest points difference
  over its largest return difference.)
- **What it bounds.** S7 checks that the policy scored in float64 acts as the learner's own float32 network does, and
  only for seed 0's final actors (seeds 1-4 are never checked). At 0.05 points, a strong-bank start may differ by up to
  0.635 return units between the two; if every start sat at the limit in one direction, one setup's seed-0 strong-mix
  mean could differ by 0.635 return units, 35% of the 1.8 bar (version 1: 0.127, 7%). Each setup of a Δ may differ in
  its own direction, so a seed-0 Δ on the strong mix may stand up to 2 × 0.635 = 1.27 return units (71% of the bar;
  version 1: 0.25, 14%) from the same Δ computed with the float32 networks, and up to 0.77 (43%) on the no-shift bank
  (Claims 3 and 3b). Observed differences are far smaller: ReBRAC alone's largest strong-bank difference was 0.0107
  return units on one start, its mean 2.4e-6 points.

### 19.6 The investigation (your chosen first step)

- **Question**: is ReBRAC alone's big-kicks difference consistent with float32 rounding that the closed loop
  amplified, or is there a defect in the float64 adapter or the float32 restatement?
- **Tool**: `probe_s7.py`, carried on the branch `claude/shift-hosts-v2` at `outputs/shift_hosts_v2/probe_s7.py`
  (outside the code manifest) and run from a copy outside the repository, on the laptop at `/root/s7probe/`, so it is
  in neither the code manifest nor the bytecode check (`gate.py:108-151`); the copy is taken with `git show` from the
  Windows clone, so nothing in `/root/bca-hosts` changes (section 19.8, step 1). It runs on the laptop, in the scoring environment, with
  `python -B` and `PYTHONDONTWRITEBYTECODE=1`, before any version-2 byte reaches `/root/bca-hosts`, and only while
  version 1's approval still verifies: it checks that the approval that verifies is version 1's (fingerprint
  `8c5e8421922dcceb`), so the code it imports is version 1's, and it has no option to skip that check. The probe itself
  is not approved code: it runs because you chose "Investigate" (section 19.3) and makes no record of the study. It
  writes nothing under the repository and has no option to print anything but differences, ratios, ranks, counts,
  start indices and SHA-256 values: never a return, a points level, a command or a state. It runs twice: on the
  stage-1 actors at their final step (15,000), and on the same runs' 10,000-step actors (the trend, shown and not
  judged).
- **What it measures**, per bank of ReBRAC alone's stage-1 actor (and, with `--compare`, of the other three learners'
  alone actors): the lane's own rollouts again, which must reproduce `s7.json` bit for bit; the distribution of
  |points difference| over the 1,000 starts (quantiles, and counts above 0.001, 0.005, 0.01, 0.02 and 0.05); the gap
  between the two networks at the same state, at every step of both rollouts (the gap injected per step, which S7
  never measured on a trained actor); a float32 reference spread at the same states (the same actor's formula
  evaluated a second way in float32, in plain numpy, against the restatement: the size of float32 rounding at this
  actor's own activations); a control of the same size that leaves the restatement out (the float64 adapter plus
  offsets of the measured injected size at each start, step and command coordinate, five sign draws, signs fixed per
  start or drawn per step); smaller controls, shown and not judged (the float64 adapter with its command rounded to
  float32, or with every float of the start moved by one unit in the last place, and the restatement with its start so
  moved); and, for the worst start of big kicks, the divergence step by step with any branch the two runs took
  differently (the observation's float32 rounding, the 80% knee, a command at exactly ±1, flipped ReLU units).
- **How it is read, fixed now, before it runs on a trained actor.** Only ReBRAC alone's 15,000-step actor is judged;
  every other actor, step and cell is shown and not judged, and the output marks the judged cells. The reading is
  "consistent with float32 rounding amplified by the closed loop" when the lane's numbers reproduce bit for bit on all
  seven banks and all three hold:
  1. on each of the seven banks, the largest injected gap, over every state of both rollouts, is at most max(1e-5, 10 ×
     the largest float32 reference spread at the same states); 1e-5 is stage 0's bound for the adapter against the
     restatement (test (h2)), and the reference spread scales it to this actor's activations, since float32 rounding
     grows with them (test (h2) records its extreme rows unjudged for that reason, `tests/test_scoring.py:262-265`);
  2. on big kicks' worst start, the propagated gap (the float64 network at the two runs' different states) reaches the
     injected one at some step t ≥ 1, and on every such propagation-dominated step the state divergence grows by at
     most 10 times in one step; steps where a new injection dominates are shown, not judged;
  3. on big kicks, the same-size control, taken as the largest of its five draws, moves the bank's largest difference
     by 0.1 to 10 times the restatement's largest difference. The same ratio at the worst start is shown, not judged:
     that start is selected as the restatement's most extreme of 1,000 and the control's value there is not, which
     biases that ratio low.

  Any other outcome reads as a possible defect. Readings 2 and 3 test amplification only: the control takes its size
  from the measured gap, so they hold for any small injected gap, defect or rounding alike. Reading 1 alone bounds the
  injected gap, so a defect in the float64 adapter smaller than its bound per command coordinate is not excluded.
- **How these rules were set.** After a cloud calibration on made-up actors and before any trained actor was probed. An
  earlier bound of about 1e-6 flagged defect-free made-up actors. The rules first drafted here (an absolute 1e-5 in
  reading 1, every step in reading 2, the worst start in reading 3) flagged defect-free made-up ReBRAC actors whose
  kernels were scaled by 1.25 to 3 (reading 1 at scales 2 and 3, reading 2 at 1.25 and 3), and the worst-start ratio
  fell to 0.16 at scale 2, within 1.6 times of its limit. With the rules above, on made-up ReBRAC actors scaled by 1,
  1.25, 1.5, 2 and 3 and no defect (the four situation banks real, the three weighting banks made up), every bank
  reproduced the lane's computation bit for bit and all three readings held: the injected gap was 1.1 to 2.5 times the
  reference spread and at most 25% of its bound, the largest growth on propagation-dominated steps 0.07 to 0.86 of a
  decade, and the control's ratio 0.73 to 1.69. Made-up defects in the float64 adapter, run through the same judged
  path, an output bias off by 1e-4 (kernels scaled by 1 and by 2) and a wrong branch on rare states, failed reading 1;
  readings 2 and 3 held for them, as they must. A 1e-4 bias at scale 3, where the reference spread is 3e-5, was not
  detected: that is the size limit stated above. The made-up actors of all four learners at scale 1 gave the same
  picture. The calibration's harness and outputs are in `outputs/shift_hosts_v2/calibration/`, outside the package.
- **If the lane's numbers do not reproduce, or the reading is a possible defect**, this draft is not frozen. A defect
  is a code fix and needs its own version (section 9.1); the 0.05 limit is not used to cover it.
- **Result.** [TO FILL ON THE LAPTOP BEFORE THE FREEZE: the date and time; the SHA-256 of the probe file and of both
  saved outputs; whether the lane's numbers reproduced bit for bit on every bank; the number of big-kicks starts above
  0.01 and above 0.05; readings 1-3 with their numbers (the largest injected gap against its bound, the first
  propagation-dominated step and the largest growth on such steps, the control's ratio) and the verdict line; the
  largest |points difference| of each bank for the four alone actors at 15,000 and at 10,000 steps.]

### 19.7 The change, exactly, and what stays

- **The change.** `evaluate.S7_CLOSED_TOL_POINTS` is 0.05, not 0.01 (`evaluate.py:95`, the one place it is enforced):
  per start, on all 7 banks and all 1,000 starts of each, on both of CQL's readouts, for every stage-1 actor and every
  seed-0 final actor of the alone, standard and IW setups. The open loop's 1e-7 and stage 0's adapter-against-
  restatement 1e-5 are unchanged.
- **Edits that follow from it** (all in the code manifest, so the fingerprint changes): the limit's statements in
  `evaluate.py`'s and `actors_f32.py`'s docstrings; test (h4) in `tests/test_scoring.py`, renamed and its docstrings
  updated (its body reads the constant, so it now checks 0.05); one caveat added to `export.CAVEATS`, so
  `results.json` states the post-hoc limit (section 19.10; no number `export.py` computes changes); this document's
  title, header, list of differences, S7 row (section 9.2), section 13 and section 18 notes and this section; and
  `prespec.json` (version, title, approval summary, S7, the history of versions). One stage-0 test changes because of
  the restart, not the limit: since 16 October the package is committed in `/root/bca-hosts`, so `git status` can no
  longer show it as new, and the hygiene test (`tests/test_all_stage0.py`) now requires that git answers (exit status
  0 and a HEAD) and shows no change outside the package. It runs in the suite whose summary S0 reads
  (`gate.py:293-303`), so it changes what passes S0 in exactly that way.
- **Unchanged**: everything else: every other threshold, claim, mark and bar, the seeds, the caps, the settings, the
  Unifloral pins, the data and the scoring cache, the fingerprint's format (`gate.py`), `approve_hosts.py`, and the
  repository code outside the package. Version 1's approval files stay committed and unchanged; they are outside the
  code manifest, and the gate opens only the current fingerprint's file.
- **Stage 1 runs again, in full** (20 runs) under version 2's fingerprint. Version 1's records name the old
  fingerprint and can never count: the runner refuses a record of another identity, and the driver halts on a record of
  another fingerprint (`runner.py:1043-1053`, `stages.py:779-782`). This reruns a stage that stopped. Section 9.1's "a
  recorded stop is never rerun" still holds for version 1's records, which stay unchanged in an archive (section
  19.8), but the stage itself runs again under this version. Repeated card runs are bit for bit (S8), so ReBRAC alone is
  expected to reproduce its 0.0114 points, which passes at 0.05.
- **Risk, stated now.** 0.05 is 4.4 times the one value seen, which is the largest of a heavy-tailed distribution (on
  big kicks, 177 times the mean), measured on 15,000-step actors only; nothing shows how that tail grows by 100,000
  steps, where the claims are judged. On made-up ReBRAC actors with no defect, rounding-only differences grew much
  faster than the network's scale: with every kernel scaled by 1, 1.25, 1.5, 2 and 3, big kicks' largest difference was
  0.00065, 0.0072, 0.13, 0.14 and 22.6 points, and 0, 0, 1, 3 and 32 of 1,000 starts exceeded 0.05 (the cloud
  calibration of section 19.6). So a later actor may exceed 0.05: one of ReBRAC's stage-1 BCA actors, or any seed-0
  final actor. A failing S7 lane halts the driver as soon as it finishes (`stages.py:345-346, 989-990`), and S7 has no
  accept option (section 8). Stage 2a runs the sham-alone block first and then seed 0 learner by learner
  (`stages.py:1123-1145, 1173`; TD3+BC, CQL, IQL, ReBRAC, `runner.py:125`), so a stop on a ReBRAC seed-0 actor would
  come after the sham block's 2 runs and 9 seed-0 runs have trained, with their scoring lanes run or queued (scores
  files on disk, unread). The investigation's 10,000-step run shows whether the tail grew from 10,000 to 15,000 steps.
- **If 0.05 is exceeded later.** [TO FILL BEFORE THE FREEZE, YOUR DECISION: what happens then. The review proposed
  "the stop stands; that quality is reported as stopped by S7; no later version relaxes S7 again". With nothing
  written here, section 9.1 applies: the study waits for another version, written after scores files exist, whose
  write-up says whether any return had been read.]

### 19.8 The restart, in order

Version 2 replaces steps 1-3 of section 18 with these; steps 4-7 stand.

1. **The investigation** (section 19.6), on the laptop, with version 1's bytes on disk and its approval verifying:
   two runs, the final step (`--list-s7 --compare`) and the 10,000-step actors (`--step 10000 --compare`). Nothing
   under `runs/shift_hosts/` is moved or changed before it. The probe is copied out of the Windows clone, not out of
   `/root/bca-hosts`, which stays at `608eefa` (the commands are at the top of `probe_s7.py`).
2. **The freeze edits**, in this document and `prespec.json` only: section 19.6's result and section 19.7's decision are
   filled in; `prespec.json`'s three `[TO FILL ...]` slots are filled in (two lines of `approval_summary`, the
   investigation's result and your decision, so the approval screen shows both, and `post_hoc_changes` P1's
   `if_exceeded_later`); the draft paragraph under the title is deleted; " [DRAFT]" leaves both titles; `prespec.json`'s
   `status` reads `"frozen"`. Then `grep -n "TO FILL" PRESPEC.md prespec.json` prints nothing, and `prespec.json` still
   loads and `stages.frozen_config()` still works.
3. **The version-2 bytes in both places**: `/root/bca-hosts`, with the branch `claude/shift-hosts` fast-forwarded to
   the version-2 draft commit (the reviewed draft, committed without the freeze edits) and step 2's freeze edits
   applied on top, not yet committed; and the Windows staging folder with the same bytes. `run_stage0.sh` copies the
   staging folder into WSL first, so the staging folder must hold the final bytes. The two must be equal byte for byte
   (LF line endings), with no `__pycache__` anywhere. Uncommitted package edits pass the hygiene test (section 19.7).
4. **Version 1's stage 1 moved aside, never deleted** (git ignores `runs/`): `runs/shift_hosts/stage1/` to
   `runs/shift_hosts_v1/stage1/`; `runs/shift_hosts/driver/logs/` to `runs/shift_hosts_v1/driver_logs/` (the driver
   reads a job's last status line from the end of its log, `stages.py:719-729`, so version 1's lines must not sit
   there); and `driver/state.json`, `state.json.sha256`, `state.json.prev` and `state.json.prev.sha256` to
   `runs/shift_hosts_v1/driver_state/` (version 1's state holds wall times under the run labels version 2 reuses, and a
   resumed driver would keep them for a run that finished unwatched, `stages.py:819-824`, and S6 would read them,
   `stages.py:1070-1076`; without the file and its `.prev` copy the driver starts a fresh state, `stages.py:445,
   454-456`). Kept where they are: `source/`, `data/` and `scoring_cache/` (S11; their hashes are pinned), `stage0/`
   (the suite rewrites `SUMMARY.json`; version 1's copy stays in `stage0/suite/20261016T144401Z/`), and the rest of
   `driver/`: the ledger (version 1's spend and stage 0's entry, section 19.9; the start reads only the ledger,
   `stages.py:1512-1523`), version 1's `STOP.json`, its sidecar and `STOP.txt`, `driver.log` and `nohup.out`.
5. **The full stage-0 suite** (`run_stage0.sh`, made-up arrays only) on exactly these bytes, before any approval of
   version 2 exists: several stage-0 tests require that the gate refuses, and one launches a real runner command
   expecting the refusal (`tests/test_runner.py:494-521`), which holds only while no approval matches the code on disk.
   Because version 1's approval file is in the package, `test_guard_compare`'s
   `test_refuses_now_in_the_real_package` skips (`tests/test_guard_compare.py:248-249`), as `test_all_stage0`'s gate
   test already does once `prespec.json` exists; `test_runner`'s check still tests the refusal on real data. Version
   1's suite took about 97 minutes, card timing first: a charged laptop and an idle, cool card. The driver refuses to
   start until this passes (S0, `gate.require_s0`).
6. **The pre-commit check**, as for version 1: the staging folder equals WSL; no `__pycache__`; `gate.fingerprint()`
   gives the fingerprint expected for the frozen bytes; `gate.s0_status()` passed. Then the freeze commit of exactly
   these bytes ("Unifloral hosts: freeze pre-specification version 2 (fingerprint <first 16>)"), pushed to `origin`
   and GitHub; `approve_hosts.py` refuses until every file is committed.
7. **You approve at your terminal** with section 18 step 1's commands, unchanged but for the new first 16 characters.
   `approve_hosts.py` writes `approval_<new first 16>.json` and its sidecar; you commit them under your address and
   push. No agent runs this.
8. **The pre-start checks** of section 18 step 2, without the suite (it ran in step 5): the gate passes, S0 passes,
   the graphics driver is still 610.62 (S9), no other process uses the card, and the card is idle and cool.
9. **The start**, by you, as root in `/root/bca-hosts`: section 18's command with `--resume-after-stop`, which the kept
   `driver/STOP.json` requires (`stages.py:1544-1551`; it moves version 1's stop report aside with a time stamp). The
   stage-0 options change nothing now, since the ledger already holds stage 0's entry.

   ```bash
   env -u LD_LIBRARY_PATH PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 /opt/bca-venv/bin/python \
       -m experiments.shift_hosts.stages start --stage0-card-hours 0 --stage0-processor-hours 0 --resume-after-stop
   ```

   Keep the Ubuntu window open for the whole run (minimized is fine): WSL shuts Ubuntu down when no window is
   attached, and the driver dies with it. `start` returns once the driver runs in the background; it does not itself
   look for `STOP.json`, so a refusal of the background run appears only in `driver/nohup.out`
   (`stages.py:1544-1551, 1569-1589`). After `start`, check `driver/nohup.out` and then `driver/driver.log` (`tail -f`
   once it exists) for "driver starts", and that `driver/STOP.json` became `STOP.json.<time>`. Stage 1 then runs all 20
   runs from the start, and stage 2a follows by itself once every stage-1 stop rule passes.

### 19.9 Budget

- **Version 1's stage 1 is spent**: 0.2785 card-hours and 0.4090 processor-hours, counted against medium-expert's cap.
  The kept ledger holds it; S6 and S6b add the ledger's totals to their projections (`budget.py:240-254, 327-337`).
  Its version-1 entries carry the labels that version 2's stage-1 runs reuse. On a clean run this changes nothing. If a
  version-2 stage-1 run finishes while no driver watches it (a crash), the resumed driver finds version 1's closed
  interval under the same label and adds nothing for it (`budget.py:218-238`), so that run's compute would be
  under-counted; the write-up then adds that run's own recorded time.
- **Stage 1 again**: 0.62 card-hours (high 1.13) and 1.28 processor-hours (high 2.48), as section 13 projects.
- **Stages 1-2a with version 1's spend**: 7.53 card-hours (high 11.48) and 9.91 processor-hours (high 16.87), inside
  14 and 21. In section 13's hard-throttle case (13.8 card-hours) the total comes to about 14.08 card-hours, just over
  the cap of 14. If stage 1 itself measures the throttled speeds, S6 at the end of stage 1 stops the study, since it
  projects both qualities from stage 1's speeds and medium's hard-throttle 12.6 card-hours is over its cap of 12 as
  well (`stages.py:1102-1116`); if the throttling starts later, S6b stops medium-expert between whole units. Either is
  S6 or S6b as approved, not a new rule.
- **The version-2 stage-0 suite and the investigation** are stage-0 work: booked separately, like stage 0 (section
  13), and not entered in the ledger. The suite's own measurement is in `runs/shift_hosts/stage0/SUMMARY.json`.
- The caps are unchanged. Medium (stage 2b) is unaffected.

### 19.10 Where the post-hoc label appears

- In this document (its title, its header and this section) and in `prespec.json` (`version`, `title`, `status`,
  `supersedes`, `post_hoc_changes`, S7's entry, and the version-2 notes under `driver_command` and `budget_recost`);
  before you type, `approve_hosts.py` shows the title and `approval_summary`, whose first lines state this change, what
  it can affect, the investigation's result and your decision on a later exceedance.
- In `results.json`: its `caveats` list (`export.CAVEATS`, written at `export.py:1429`) states that S7's closed-loop
  limit is version 2's post-hoc 0.05 points. It also records the approved fingerprint, which names this version, and
  with the kept driver folder it lists version 1's S7 stop under S7 as a stop resumed by the operator.
- Every write-up that reports S7 calls it "S7, with version 2's post-hoc limit of 0.05 points (section 19)".
- The study stays exploratory (section 15).

## Words used here

- **Learner (host)**: the offline reinforcement-learning method BCA is added to. **Outer step**: one call of a Unifloral
  file's training step, with one batch.
- **BCA**: puts a calibrated band around the critic's one-step error and turns its width into extra caution, the
  **dose**, in one term of the learner's loss. **Refresh**: recomputing the band's threshold R on the bank. **IW**:
  importance weighting of the bank rows toward the strong-mix test law.
- **Imported / restated**: run unchanged from the pinned file / copied line for line with BCA's additions.
- **Sham alone**: the restated step with BCA present but never refreshed, so it learns as the learner alone, in the BCA
  setups' compiled program.
- **Room reference**: the learner alone on the full, uncut log.
- **Return units**: the plain 100-step return, closer to 0 is better. **Points**: 0 is random commands, 100 the
  stall-aware controller, per bank.
- **Δ, L, upper bound**: the mean paired difference over 5 seeds and its one-sided 95% bounds. **Holm's correction**:
  keeps the chance of any false pass among the four learners at 5%.
- **Card-hour / processor-hour**: section 9.2, S6b. **Fingerprint**: the SHA-256 that ties your approval to this
  pre-specification and the code.
