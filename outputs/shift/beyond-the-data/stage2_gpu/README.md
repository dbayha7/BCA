# Stage 2 on your own machine: the confirmatory seeds 5-9

This folder lets you run the confirmatory part of Stage 2 (the flags claim) on your own computer instead of in the
cloud. Stage 2 is the test of BCA described in `outputs/shift/beyond-the-data/stage2/PREREGISTRATION.md`, version 6,
which you approved (commit `7dab7d5`). The confirmatory part uses fresh seeds 5-9, with every setting frozen on
seeds 0-4. Seeds 0-4 are the exploratory part and run in the cloud.

A "phase" below is one step of the pre-registered work, run as one process (for example "train the policy test on
seed 6"). The code that runs each phase is the pinned Stage 2 code, unchanged.

## First: this package uses your processor, not your graphics card

The pre-registration you approved requires the processor for every phase, and its code enforces it. Before each phase
the code checks that JAX (the numerical library the code uses) runs on the processor (`JAX_PLATFORMS=cpu`), that the
flag `XLA_FLAGS` is not set, that `OMP_NUM_THREADS=1`, and that Python and eight libraries have exactly the recorded
versions. A run on the graphics card (GPU), or with the setting that makes the graphics card repeatable
(`XLA_FLAGS=--xla_gpu_deterministic_ops=true`), is refused at the first step of every phase. Using the graphics card
would need a code change, so a new pre-registration version and a new approval from you. This package changes no code,
so it runs on your processor, with the same settings as the cloud.

What you gain: the cloud is freed for seeds 0-4, and your processor may be faster per core. What does not change: the
pre-registration allows at most 2 processes at once, in total (the cloud included), and about 38-48 process-hours for
seeds 5-9. That is about 19-24 hours with 2 processes, or twice that with 1.

The self-test can also measure, for information only, how fast one training step is on your graphics card. That
number helps you decide whether a later version that allows the graphics card is worth a new approval.

## Will your processor work? The self-test answers this first

Every phase begins with the gate's own check of its pre-registration. Part of that check recomputes, with numpy, the
64 recorded "anchor" numbers (the returns of two reference controllers and their standard errors) and compares them
with the recorded ones exactly, to the last bit. The last bits depend on which set of matrix routines numpy's built-in
matrix library (OpenBLAS) chooses for your processor:

- processors with AVX-512 instructions (a set of wide arithmetic instructions) are expected to get the same routines
  as the cloud's processor, and then the check passes. Examples: AMD Ryzen 7000 and 9000 desktop processors; most
  Intel Xeon processors.
- processors without AVX-512 get other routines; then some anchor numbers differ in their last bits (44 of them in
  the cloud's own test) and every phase refuses. Examples: Intel Core 12th, 13th and 14th generation and Core Ultra;
  AMD Ryzen 1000 to 5000 desktop processors.

The self-test runs this check first and prints your processor's model. If it fails, nothing in this package can run on
that machine without a code change; tell the cloud, and the work stays in the cloud (or you decide on a new version).

## What runs here, and what stays in the cloud

Here, for each fresh seed 5, 6, 7, 8 and 9, in the pre-registered order:

1. new data and learners (no BCA): `fresh_data`, `fresh_learners`;
2. the detection test's no-BCA steps, stalling world then no-stall world: `detect_pairs`, `detect_labels`,
   `detect_standins`;
3. BCA's band and the widths: `detect_fit` (fixed critic in both worlds, host critic in the stalling world),
   `detect_fit_weighted` (only if seeds 0-4 chose the weighted version), `detect_score`;
4. the policy test: `policy_train`, `policy_eval`, `policy_no_stall_train`, `policy_no_stall_eval`;
5. the deployment check: `deploy`, both worlds (after the policy test, so that BCA's trained actor is checked too);
6. the full host on that seed: `host_train`, `host_eval`, stalling world then no-stall world;

then, once every seed has finished or failed on record, the confirmatory detection marks (`detect_marks`) and the
pooled look (`pooled`). Steps 4 and 6 run only if a version passed on seeds 0-4 (the room test already passed).

In the cloud: everything on seeds 0-4, including the full host on seeds 0-4. That full host needs the gate's saved
model files (about 270 MB), which git cannot carry in this package. The cloud also writes the two late files below and
checks your results when they come back.

## Setup, step by step

Type the commands exactly. Steps 1 and 2 are in Windows PowerShell; everything from step 3 on is inside Ubuntu.

**Step 1. Install WSL 2 with Ubuntu 22.04** (WSL is the Windows Subsystem for Linux). Open PowerShell as
administrator:

    wsl --install -d Ubuntu-22.04
    wsl --update
    wsl -l -v

Restart Windows if asked. When Ubuntu opens the first time it asks for a Linux user name and password; choose any.
`wsl -l -v` must show `Ubuntu-22.04` with VERSION `2`; if it shows 1: `wsl --set-version Ubuntu-22.04 2`.

**Step 2. Open Ubuntu as the administrator account (root).** Every later step runs in this window:

    wsl -d Ubuntu-22.04 -u root
    cd /root

**Step 3. Check the graphics card and the processor:**

    nvidia-smi
    grep -m1 "model name" /proc/cpuinfo
    grep -m1 -o avx512f /proc/cpuinfo

`nvidia-smi` should list your NVIDIA card (if not, update the NVIDIA driver in Windows). If the last command prints
nothing, your processor has no AVX-512: stop here and tell the cloud the processor model.

**Step 4. Install the tools:**

    apt-get update && apt-get install -y git curl tmux ca-certificates

**Step 5. Get the repository, with its whole history, inside Linux** (not under `/mnt/c`, which is much slower):

    mkdir -p /home/user
    git clone --branch claude/main-bca-connection-x4o46s https://github.com/dbayha7/bca.git /home/user/bca

The repository is private, so git asks you to sign in: give your GitHub user name and, as the password, a personal
access token (GitHub, Settings, Developer settings, Personal access tokens), or first let Git for Windows sign you in:
`git config --global credential.helper "/mnt/c/Program Files/Git/mingw64/bin/git-credential-manager.exe"`.
Do not add `--depth`: the code needs your approval commit. If a clone is already shallow:
`git -C /home/user/bca fetch --unshallow`. Do not set a git name or email in this root account (the code refuses to
run under the identity that made your approval commit; step 11 shows how to commit anyway).

**Step 6. Install the exact Python and libraries the cloud uses.** uv is the installer the cloud used; its version
matters because it decides the exact files installed:

    curl -LsSf https://astral.sh/uv/0.11.32/install.sh | sh
    . /root/.local/bin/env
    uv --version
    uv python install 3.10.20
    uv venv --python 3.10.20 /root/bcavenv
    uv pip install --python /root/bcavenv/bin/python --no-deps --require-hashes --only-binary :all: -r /home/user/bca/outputs/shift/beyond-the-data/stage2_gpu/requirements-lock.txt

`uv --version` must print `uv 0.11.32`. The lock file pins 67 packages (jax 0.6.2, jaxlib 0.6.2, flax 0.10.7,
optax 0.2.8, numpy 2.2.6, scipy 1.15.3, h5py 3.16.0 and the rest), each to the one file the cloud installed. It leaves
out ten packages the cloud has but Stage 2 never uses (they need a separate install of the MuJoCo physics simulator).

**Step 7. Install the environment's own small dependency folder** (gymnasium 1.0.0, kept apart from the rest):

    BCA_SHIFT_PYTHON=/root/bcavenv/bin/python sh /home/user/bca/experiments/shift/install_deps.sh

It must end with `ok: gymnasium 1.0.0 ...`. It installs into `/root/.cache/bca-shift/py`. Run it in the same window as
step 6, so that uv is found; otherwise the folder's fingerprint differs from the recorded one and every phase refuses.

**Step 8. Run the self-test and paste everything it prints back to the cloud:**

    sh /home/user/bca/outputs/shift/beyond-the-data/stage2_gpu/selftest.sh

It takes about a minute and runs no phase. Its last line is one of:
- `RESULT: READY, WAITING FOR THE CLOUD'S LATE FILES` (expected now),
- `RESULT: READY` (after step 9),
- `RESULT: NOT READY (...)`, naming what failed.

The report is also saved as `/home/user/bca/runs/shift_explore/stage2_offcloud/selftest_<time>.txt`.

*Optional: measure the graphics card.* In a separate environment that the phases never use (about 3 GB to download):

    uv venv --python 3.10.20 /root/bcagpu
    uv pip install --python /root/bcagpu/bin/python "jax[cuda12]==0.6.2" "flax==0.10.7" "optax==0.2.8"

Run the self-test again; it then also prints one training step's speed on the graphics card and on one processor.

**Step 9. Wait for the cloud's two late files.** The cloud first finishes the detection test on seeds 0-4, which
decides the version (and, with the stand-ins test, the rival). It then commits two files into `payload/late/` and
tells you. Then:

    git -C /home/user/bca pull --ff-only
    sh /home/user/bca/outputs/shift/beyond-the-data/stage2_gpu/selftest.sh

**Step 10. Run, inside tmux** (tmux keeps the run going if the window closes):

    tmux new -s stage2
    sh /home/user/bca/outputs/shift/beyond-the-data/stage2_gpu/run_confirmatory.sh --lanes 1

`--lanes` is how many phases run at once: 1 or 2. Use the number the cloud gives you, because the pre-registration
allows at most 2 processes at once in total, the cloud included. To see the plan and what has finished, without
running anything: add `--plan-only`. To leave tmux running: press Ctrl-b, then d; to come back: `tmux attach -t stage2`.
Keep Windows awake while it runs (Settings, System, Power: never sleep when plugged in).

To stop safely: press Ctrl-C once and wait; the phases already running finish first. If Windows or WSL restarted,
open Ubuntu as root again (step 2), `tmux new -s stage2`, and run the same command: it puts back what a restart
removed, skips every finished phase and continues.

**Step 11. Send the results back.** When the run ends it prints `Finished.` and has copied the results into
`outputs/shift/beyond-the-data/stage2_gpu/results/`. Commit them under your own name and push:

    cd /home/user/bca
    git add outputs/shift/beyond-the-data/stage2_gpu/results
    git -c user.name="David Bayha" -c user.email=35092758+dbayha7@users.noreply.github.com commit -m "Stage 2: confirmatory results, fresh seeds 5-9, run on my machine"
    git push origin claude/main-bca-connection-x4o46s

(Or copy that `results` folder from `\\wsl$\Ubuntu-22.04\home\user\bca\outputs\shift\beyond-the-data\stage2_gpu\`
into the same place in your Windows clone, after `git pull` there, and commit and push it as usual.) Then tell the
cloud. It checks every file before reading any result (see "The check in the cloud" below).

## If something stops

- **A phase failed:** the run stops starting new phases, lets the running ones finish, deletes nothing and names the
  log to look at. Paste the last 30 lines of the output to the cloud. Do not delete or edit anything under
  `/home/user/bca/runs/`: every phase writes once, and a seed that fails is recorded, never replaced.
- **A refusal before any phase:** the message says why (the late files are not there yet, a file differs from its
  fingerprint, git's identity, a shallow clone, a changed file in the repository, the gate's check on a processor).
- **"another run_confirmatory.sh is running"** or **"a phase from an earlier run is still running":** wait for it.

## What the package holds

| File | What it is |
|---|---|
| `README.md` | this page |
| `selftest.sh` | the self-test (step 8) |
| `run_confirmatory.sh` | the run (step 10) |
| `restore.sh` | puts the files below where the pinned code expects them and checks every fingerprint; the other two scripts run it first, every time |
| `requirements-lock.txt` | the exact libraries (step 6) |
| `env_record.json` (+ `.sha256`) | the cloud's environment, which the self-test compares yours with |
| `payload/MANIFEST.json` (+ `.sha256`) | every payload file, where it goes, and its SHA-256 fingerprint (a checksum that changes if a single byte changes) |
| `payload/files/` | 154 files, 1.06 MB, that git does not otherwise carry (the repository ignores `runs/`) |
| `payload/late/` | written later by the cloud: the two late files and their index `LATE.json` |
| `lib/` | the scripts' helpers (no study code) |
| `cloud/` | scripts the cloud runs: building the payload and the environment record, publishing the late files, checking your results |
| `results/` | written by the run: the results to commit (step 11) |

The 154 payload files are exactly what the confirmatory phases read outside git, traced from the pinned code:

- the gate's pre-registration and the records its check re-reads: `runs/shift/preregistration.json`,
  `runs/shift/pre/stage0_base.json`, `stage0_retune.json`, `build_checks.json` and the 55 data records it lists
  (`runs/shift/pre/data/.../record.json`), each with its `.sha256` file;
- the room test's code manifest, spec, pinned `harness.py` and pooled look (the run condition of steps 4 and 6);
- the detection script's spec, code manifest and its 12 pinned files;
- the stand-ins test's code manifest and its 10 pinned files;
- `runs/shift_explore/stage2/approval.json`, the import of your approval.

`restore.sh` also copies, from files git already carries, the Stage 2 code (17 files) and the fallback diagnostic to
the folder the pinned code's paths are fixed to (`/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/ood/`),
the Stage 2 pre-registration and code manifest to `runs/shift_explore/stage2/`, and links `/tmp/claude-0/bcavenv` to
`/root/bcavenv`. WSL empties `/tmp` at every start, which is why this happens on every run. It never overwrites a file
that differs; it refuses. Then it checks the pre-registration's own chain of fingerprints: your approval names this
pre-registration; the pre-registration names the code manifest, which names every Stage 2 file and every repository
file Stage 2 imports; it names the room test, the fallback diagnostic, the stand-ins and the detection script by their
manifests; the gate's pre-registration names its records.

**The two late files** (the run refuses until both are there; the cloud commits them into `payload/late/` with
`LATE.json`):
- `best_standin.json`: the rival stand-in, chosen before any BCA run;
- `detect/marks_exploratory_uniform.json`: the detection marks on seeds 0-4, which decide the version; and, only if
  the uniform version failed direction or size there, `detect/marks_exploratory_weighted.json` too.

No model file and nothing large is in the package or in the results: model files (`.msgpack`) and arrays (`.npz`)
stay on your machine.

## What the self-test checks

In this order: the payload and every fingerprint (`restore.sh`); the gate's own check on your processor (the anchors);
the versions (Python 3.10.20, the eight recorded libraries, all 67 locked packages and their installed files, the
dependency folder); ThrusterChain's environment checker on all 8 environment versions; your approval, the import
record and every pin, through the pinned code itself, read only; git (a full clone, your approval commit present,
the branch from GitHub present, no git identity, no changed file); one processor per phase; the late files; and, for
information, the graphics card and the memory and disk. Every check that touches the pinned code runs as its own
process pinned to one processor with exactly the settings a phase gets.

## The rules the run keeps

- **The code is unchanged.** Every run verifies every fingerprint first, and the pinned code checks them again at
  every phase. Each phase runs through the pinned `launch.sh`, unchanged.
- **Processor only, as recorded.** `launch.sh` sets `JAX_PLATFORMS=cpu` and `OMP_NUM_THREADS=1`; nothing else from
  your shell reaches a phase (the scripts start every process with a clean environment). The backend is recorded for
  every result in `results/BUNDLE.json`; the gate's check inside each phase refuses any other backend.
- **One processor per phase.** `launch.sh` pins each phase with `taskset -c <processor>`. The detection script
  refuses to run otherwise, and the policy test needs it too: its check (b) compares, to within 0.00001, an actor
  trained by `policy_train` with one saved by the earlier `fresh_learners` process, which holds only when both see the
  same number of processors. By default the two lanes use processors on different physical cores.
- **Write once, never replace.** A phase counts as finished when its output files exist and match their fingerprints.
  A rerun starts only what is missing. A seed that fails its data checks, a learner, a band fit or the policy test's
  check is recorded and never replaced; the phases the code would refuse for it are skipped, by the code's own rules.
- **At most 2 processes at once**, in total with the cloud.
- **Results go back whole.** The bundle holds every JSON record of seeds 5-9 with its fingerprint file, the phase
  logs, the run's own log and status, and the self-test report, plus `BUNDLE.json`, which names each file's
  fingerprint and records the software, the processor, the code and pre-registration fingerprints and git.

## The check in the cloud

`cloud/check_return.py` reads what you pushed straight from the branch as fetched from GitHub. It checks every file's
fingerprint and sidecar; that every result names this pre-registration, the gate's, this approval and the pinned
detection script; that the code, `launch.sh`, the software, the dependency folder and the backend are the recorded
ones; that the late files are the cloud's own; that every commit touching `results/` is yours (the identity of your
approval commit) and none is the cloud's; and that every phase finished or failed on record, and each width score,
threshold and values file names its own band or lock. Nothing is read before it passes.

## Notes on the numbers

- The anchors (above) are the only check that compares your machine's arithmetic with the cloud's, and they depend
  only on numpy's matrix library, not on JAX.
- Fresh seeds 5-9 compare nothing with model files made in the cloud: each seed's data, learners, pair sets, bands
  and actors are made and used on your machine. Your numbers will not equal what the cloud would have produced for
  these seeds, which is expected and allowed: the seeds are new.
- The policy test's check (b) and the full host's check with coefficient 0 compare two computations on your machine;
  the first across two processes, which the one-processor rule keeps exact.
- For a later version that allows the graphics card, two things would need settling: that version's code would have
  to stop recomputing the anchors on processors without AVX-512 (declared as a change), and the training between
  `fresh_learners` and `policy_train` would have to repeat exactly on the graphics card, or check (b) would stop
  seeds. The NVIDIA libraries would also need exact pins: `jax[cuda12]==0.6.2` sets only lower limits on them.
