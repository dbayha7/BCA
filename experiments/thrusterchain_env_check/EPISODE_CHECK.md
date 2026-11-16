# ThrusterChain episode check: do episodes played through the environment match the scoring code?

Written 2026-11-15, before any episode was played. Sections 1 to 7 are the plan and the pass marks.
Section 8 is filled in after the run and changes nothing above it.

Jump to: [1 Why](#1-why) · [2 The two routes](#2-the-two-routes) · [3 What is played](#3-what-is-played) ·
[4 Pass marks](#4-pass-marks) · [5 Trained policies: read, not played](#5-trained-policies-read-not-played) ·
[6 What the check can and cannot catch](#6-what-the-check-can-and-cannot-catch) ·
[7 Files and commands](#7-files-and-commands) · [8 Result](#8-result) ·
[9 Amendment 1](#9-amendment-1-t4-restated)

## 1 Why

Every ThrusterChain score in the Unifloral host study (the 12 setups: TD3+BC, CQL, IQL and ReBRAC, each
alone, with standard BCA and with importance-weighted BCA) comes from the scoring code: a float64
simulation, `truth.rollout`, that calls the plant's one dynamics function directly. No score comes from
playing the registered Gymnasium environment step by step, which is what an outside reader would do.
The existing tests compare the two only on a few cases and to a tolerance of 0.001.

ThrusterChain has no randomness, so this is not a statistical check like the gridworld one
(`experiments/ood_evidence/EPISODE_CHECK.md`). Here the two routes must agree exactly, or to a stated
rounding tolerance. The check also asks the stress-test question for a deterministic system: does the
environment itself magnify tiny numerical errors?

It trains nothing, fits nothing and plays no learned policy. Only fixed reference controllers are
played, which the host study's own rules allow without an approval (`evaluate.py`, "Permission").

## 2 The two routes

| Route | What it is |
|---|---|
| **Public environment** | `gymnasium.make("BCAShift/ThrusterChain-v0")`, `reset(options={"state": start})`, then `step(action)` 100 times with float32 actions, adding up the rewards. This goes through the registered id and its wrappers. |
| **Scoring code** | `truth.rollout(controller, starts)`: the float64 loop every reported score uses. |

**Pinned reference numbers.** The host study's scoring cache
(`/root/bca-hosts/runs/shift_hosts/scoring_cache`, written 2026-10-15 and pinned by SHA-256 in
`evaluate.py`) holds, for each of the 7 evaluation banks, the 1,000 starts and the per-start return of
two fixed controllers: the zero command, and the stall-aware controller that defines 100 points.
Every score's point scale and failure flag F1 rest on these numbers.

**Points.** A return difference is turned into points with the study's own scale for the bank
(`evaluate.points_per_return_unit`). The smallest difference between setups in the study's summary is
about 0.04 points, and the study's own precision limit (its check S7) is 0.05 points per start.

## 3 What is played

Fixed here, before the run. Seed of the check: 2026111511.

| Part | Controllers or commands | Starts |
|---|---|---|
| Fixed command sequences | zero; every thruster at the 80% point (float32 0.8, alternating sign); 90%; full throttle; out of range (1.5 and -1.5, which the plant clips); random float32 commands, a fresh draw per start and step | the first 250 starts of each of the 7 banks in the stalling world, and of the 4 situation banks in the no-stall comparison world |
| Reference controllers | the zero command; the stall-aware controller (`recorder.stall_aware`) | all 1,000 starts of each of the 7 banks |
| Stress test | the stall-aware controller with (i) every command multiplied by 1 + 0.000001 z, z a standard normal draw per start, step and thruster; (ii) the state it reads rounded to float32 first, as a deployed network sees it; (iii) every number of the start multiplied by 1 + 0.000001 z | all 1,000 starts of each of the 7 banks |

The 7 banks: none, mild, strong (the three current laws of the weighting test), and recorded, strong
current, big waves, big kicks (the four situations).

The check is run twice, with the same script: in WSL with the scoring interpreter
(`/opt/bca-shift-venv`, the setup the scores were made with) and the frozen code copy
`/root/bca-hosts`; then on Windows with the repository's interpreter and code copy. The two code copies
were compared file by file beforehand and are identical for every file the check imports.

## 4 Pass marks

The check passes only if every mark holds on both machines' setups. One failure fails the check. It is
run once; a failure is investigated and reported, not rerun with other settings.

| Mark | What must hold | Allowed |
|---|---|---|
| **T1 Same commands, same episode** | For every fixed command sequence and start, the states and rewards of the public environment are bit for bit those of the scoring code replaying the same commands. | 0 differences |
| **T2 Episode bookkeeping** | Every episode lasts exactly 100 steps; `truncated` is true at step 100 and never before; `terminated` is never true; a 101st step is refused; the observation is the float32 of the state; the return is the sum of the rewards; tracking cost plus energy cost equals minus the return. | 0 violations; the two sums within 1e-9 of the return's size |
| **T3 Pinned references reproduce** | (a) The zero command played through the public environment gives the pinned per-start return. (b) The stall-aware controller recomputed with the scoring code gives the pinned per-start return. (c) The stall-aware controller played through the public environment, where its commands are rounded to float32, gives the pinned per-start return. | (a), (b): within 1e-9 return units. (c): within 0.001 points on every start |
| **T4 The environment does not magnify small errors** | Under each of the three disturbances of section 3, the stall-aware controller's return changes by a negligible amount on every start of every bank. | within 0.001 points on every start |
| **T5 Same numbers on both setups** | The Windows run's per-start returns for both reference controllers through the public environment equal the WSL run's. The four situation banks regenerated by `thruster_chain.bank_starts` equal the pinned starts. | returns within 0.000001 points on every start; starts within 1e-12 |
| **T6 The JAX twin agrees** | `dynamics_jax` in float64 gives the same next state and reward as `dynamics` on every state the stall-aware controller visits on the four situation banks. Windows run only (JAX is installed there). | within 1e-12 of each number's size (at least 1) |

Notes on the marks.

- 0.001 points is 50 times tighter than the host study's own precision limit and about 40 times
  smaller than the smallest difference between setups it reports.
- The disturbance in T4, one part in a million, is about 17 times larger than the rounding of a
  command to float32 (at most 6 parts in 100 million).
- For T5, bit-for-bit equality is reported but not required. A difference of one rounding step in a
  wave number between Windows and WSL was recorded on 2026-11-01.
- Reported but not judged: the largest difference seen for each mark, in return units and in points.

**Note added 2026-11-15 after the wiring test and before the judged run. No mark is changed.**
The wiring test (20 starts per bank, not judged) showed that T4 as written is badly posed. Its tolerance
is absolute (0.001 points), but a disturbance of one part in a million moves a large return by about one
part in a million even when nothing is magnified. On the strong-current bank, where returns are largest,
the start disturbance moved one of the 20 returns by 0.0011 points, and the command disturbance by 0.0004.
So T4 as written may fail in the judged run without the environment magnifying anything.

T4 stays exactly as written and is judged as written. If it fails, it is reported as failed. The judged
run also reports, without judging it, the **gain** of each disturbance on each bank: the largest relative
change in the return (the change divided by the return's size) divided by the relative size of the
disturbance (0.000001 for commands and starts; 2 to the power -24, about 0.00000006, for rounding the state
to float32). A gain near 1 means the return moves in proportion to the disturbance. A magnifying system
has a gain in the thousands or more.

Proposed for the owner's decision, not applied here, and written before any gain was computed: restate T4
as "the gain is at most 100 for every disturbance and bank". For this plant a gain of up to about 10 is
expected from the arithmetic alone (the cost is roughly quadratic in the state, and the largest of
thousands of random draws is several standard deviations out).

## 5 Trained policies: read, not played

The host study already compared two arithmetic routes for every trained policy it scored (its check
S7): the learner's own float32 network against the float64 scoring adapter, closed loop from every
start of every bank. Those results are saved with each run. This check reads them and tabulates them
by setup. It does not play, score or re-judge any trained policy.

Written expectation, before reading them: most setups agree within 0.05 points on every start; some
ReBRAC, IQL and TD3+BC runs exceed it on a few starts (the host study stopped or flagged those), while
bank averages differ by far less. If T4 passes, such gaps come from how steeply those policies react,
not from the environment.

## 6 What the check can and cannot catch

It can catch:

- any difference between the registered environment and the scoring code in dynamics, clipping, the
  80% point, reward timing, episode length or the return's bookkeeping;
- a scoring cache that no longer matches the code;
- an environment whose scores move when numbers are rounded (T4) or when the machine setup changes (T5).

It cannot catch:

- an error in the one dynamics function itself, which both routes call. That function has its own
  tests (a second matrix form, a proven bound, physics properties in
  `zeus_20261026_validation/environment_audit_v1`), which this check does not repeat;
- anything about a learner, a critic or BCA;
- agreement on another machine (Titan), which was not available.

## 7 Files and commands

| File | Role |
|---|---|
| `episode_check.py` | runs T1 to T6, prints a table, writes one result file per setup |
| `pinned_reference.npz` | written by the WSL run: the pinned starts and reference returns of the 7 banks, the point scales, and the WSL run's own returns, so the Windows run can be compared |
| `episode_check_result_wsl.json`, `episode_check_result_windows.json` | written once each |

```
wsl -u root -e bash -lc 'PYTHONDONTWRITEBYTECODE=1 /opt/bca-shift-venv/bin/python -B /mnt/c/Users/David/dev/BCA/experiments/thrusterchain_env_check/episode_check.py --setup wsl --code-root /root/bca-hosts'
PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -B experiments/thrusterchain_env_check/episode_check.py --setup windows --code-root .
```

The WSL run reads `/root/bca-hosts` only; it writes nothing there. A wiring test with a few starts
(`--smoke`) may be run first; it is not judged and writes nothing.

## 8 Result

Written after the two judged runs. Nothing in sections 1 to 7 was changed after them.

**Verdict by the marks as written: FAIL, on T4 only.** Five of the six marks pass on both setups.
T4 fails as written, for the reason the note in section 4 gave before the judged run: its tolerance
was absolute. The reported gains show no magnification. Whether to accept the restated T4 is the
owner's decision; it is not applied here.

Both runs were made once on 2026-11-15 (WSL 133 seconds, Windows 157 seconds), each after a wiring
test that was not judged. Both used Python 3.10, NumPy 2.2.6 and Gymnasium 1.0.0, and the same six
imported files (identical SHA-256 on both setups).

| Mark | WSL | Windows | Largest value seen |
|---|---|---|---|
| T1 Same commands, same episode | pass | pass | 0 of 16,500 episodes differ |
| T2 Episode bookkeeping | pass | pass | 0 violations in 30,500 episodes |
| T3 Pinned references reproduce | pass | pass | (a) 0; (b) 0.0000000000009 return units; (c) 0.0000085 points |
| T4 The environment does not magnify small errors, as written | **fail** | **fail** | 0.0021 points (start disturbance, big kicks) against 0.001 |
| T5 Same numbers on both setups | pass | pass | returns: 0.0000000000000009 points; starts: 0.0000000000000001 |
| T6 The JAX twin agrees | not run | pass | 0 on 400,000 states |

**T4 in detail** (largest change on any of the 1,000 starts, in points; identical on both setups).

| Bank | Commands disturbed | State read as float32 | Start disturbed |
|---|---|---|---|
| none | 0.00010 | 0.0000002 | 0.00025 |
| mild | 0.00009 | 0.0000001 | 0.00023 |
| strong | 0.00007 | 0.0000001 | 0.00025 |
| recorded | 0.00026 | 0.0000010 | 0.00033 |
| strong current | 0.00060 | 0.0000014 | **0.00147** |
| big waves | 0.00033 | 0.0000008 | 0.00064 |
| big kicks | 0.00067 | 0.0000027 | **0.00212** |

Two of the 21 entries exceed 0.001 points, both for the start disturbance. Rounding the state to
float32, the disturbance a deployed network actually sees, moves no return by more than 0.000003 points.

**The gain** (reported, not judged): the largest relative change in the return divided by the
relative size of the disturbance.

| Bank | Commands disturbed | State read as float32 | Start disturbed |
|---|---|---|---|
| none | 4.7 | 0.16 | 10.4 |
| mild | 4.7 | 0.15 | 14.7 |
| strong | 4.6 | 0.16 | 18.4 |
| recorded | 2.5 | 0.22 | 4.9 |
| strong current | 5.0 | 0.25 | 13.7 |
| big waves | 3.3 | 0.13 | 7.1 |
| big kicks | 2.5 | 0.19 | 4.8 |

The largest gain is 18. A disturbance of one part in a million changes the return by at most 18 parts
in a million. That is above the "about 10" the note expected from the arithmetic and below the
proposed limit of 100. Under the reference controller the environment responds in proportion to a
disturbance; it does not magnify it.

**T5 in detail.** Through the public environment, Windows and WSL give the same return bit for bit
on all 7,000 starts for the zero command and on 6,999 of 7,000 for the stall-aware controller; the
one difference is one rounding step. The four situation banks regenerated on Windows differ from the
pinned starts by one rounding step in a wave number, as recorded on 2026-11-01; this does not change
a return by more than that one rounding step.

**Trained policies: what the host study recorded (read, not played).** The study saved a closed-loop
precision result for 84 final policies: its float32 network against the float64 scoring adapter, on
every start of every bank. 25 of the 84 are within the study's 0.05-point limit on every start.

| Data quality | Learner (all three variants) | Policies | Within 0.05 on every start | Largest gap on one start | Largest gap in a bank average |
|---|---|---|---|---|---|
| medium-expert | TD3+BC | 15 | 10 | 0.28 | 0.001 |
| medium-expert | IQL | 15 | 1 | 1.59 | 0.007 |
| medium-expert | ReBRAC | 15 | 5 | 2.50 | 0.063 |
| medium-expert | CQL | 3 | 3 | 0.00002 | 0.00001 |
| medium | TD3+BC | 15 | 0 | 35.6 | 0.272 |
| medium | IQL | 15 | 0 | 14.3 | 0.111 |
| medium | ReBRAC | 3 | 3 | 0.00001 | 0.000004 |
| medium | CQL | 3 | 3 | 0.0002 | 0.00001 |

The written expectation in section 5 was too optimistic: most policies do not agree within 0.05
points on every start, and on medium data single starts move by up to 36 points. Bank averages move
far less, as expected, but on medium data by up to 0.27 points on the worst bank. That is the same
size as some of the differences between setups the study reports on medium data (for IQL, +0.47 and
-0.05), so those differences should be read with this in mind. This table gives each learner's worst
bank; it does not break the gap down for the strong-current-law bank.

**What this establishes.**

- The registered Gymnasium environment and the scoring code are the same system: same commands give
  bit-identical episodes, and the episode bookkeeping is right.
- The pinned reference returns that set the 100-point scale and the failure flag F1 reproduce, through
  the scoring code and through the public environment.
- Scores of the reference controllers do not depend on which of the two setups computes them.
- The environment does not magnify small numerical errors. So the precision gaps recorded for trained
  policies come from how sharply those policies react to tiny changes in what they read, not from the
  environment.

**What it does not establish.** Anything in the "cannot catch" list of section 6. The precision gaps of
the trained policies are a property of those policies that remains; this check does not reduce them.

Files: `episode_check_result_wsl.json`, `episode_check_result_windows.json`,
`episode_check_output_wsl.txt`, `episode_check_output_windows.txt`, `pinned_reference.npz`.

## 9 Amendment 1: T4 restated

**Accepted by the owner on 2026-11-15** ("Yes, accept it"), after reading the result in section 8,
including the largest gain of 18. The limit of 100 was written in the note of section 4 before any gain
was computed.

**T4 as restated.** For each of the three disturbances of section 3 and each of the 7 banks, the gain
is at most 100. The gain is the largest relative change in the return divided by the relative size of
the disturbance, as defined in the note of section 4.

**Judged on the gains the two judged runs saved; nothing was rerun.** Each result file holds 21 gains.
The largest is 18.4 on both setups (the start disturbance on the strong bank). Every gain is at most
100, so the restated T4 passes on both setups.

**Verdict under the amended marks: PASS on all six marks.** The record in section 8 stands: T4 as
first written failed, because its tolerance was absolute.

The script now judges the restated T4 and still reports T4 as first written, without counting it. The
two saved result files were written before this amendment and carry the original marks.
