# The best possible controller for ThrusterChain, and how far each controller falls short of it

This folder holds an exact answer to one question: from each start in the evaluation banks, what is the best
score any controller could possibly get, and how far below that is each controller we have?

It is evaluator work only. Nothing was trained and no BCA band was fitted. It reads the saved run files and the
repository's own world code and never changes them. It changes nothing in version 6 (the frozen Stage 2
pre-registration) or version 7.

## Words used here

- **Return**: the score of one 100-step episode, the sum of the rewards. Each step's reward is minus (the squared
  distance of the five floats from rest, plus 0.3 times the squared throttles). Closer to 0 is better.
- **Points**: the gate's scale. 0 points = the recording controller (who made the log). 100 points = the gate's top
  reference controller: the stall-aware controller in the stalling world (the textbook controller capped at 80%
  throttle), the rising-thrust controller in the no-stall world (the same controller allowed full throttle).
- **The optimum**: the best possible list of 100 commands from one start, computed exactly.
- **Regret**: the optimum's score minus a controller's score. 0 would be perfect; bigger is worse.
- **The knee**: 80% throttle. In the stalling world a thruster pushes less, not more, past 80%.
- **Tracking cost**: the part of the cost from the floats being away from rest. **Energy cost**: the part from the
  throttles. They add up to minus the return.
- **Discounted value**: what the learners and the truth code value: the sum of rewards over 1,000 steps, each step
  weighted by 0.98 to the power of its step number.
- **Seeds**: gate actors come from data seeds 0-4; Stage 2 (version 6) arms from seeds 0-1.
- The controllers: **clone** (copies the recorded commands), **none** (trained against the critic of the log with
  the gap, no extra caution), **full** (the same, against a critic trained on the complete log), **bca_k1** (BCA's
  caution at strength 1), **perfect band** (caution from the true error, a reference), **range hinge** (a penalty on
  commands outside the recorded range), **even pull** (more copying of the log everywhere, matched in strength to
  bca_k1), **stand-in** (the best data-only rival, matched).

## The short answer

**1. The best possible score is only a little above 100 points.** The 100-point controllers are already close to
the best possible.

| world | recorded | strong current | big waves | big kicks |
|---|---|---|---|---|
| stalling | 100.8 | 109.0 | 102.6 | 106.7 |
| no-stall | 100.6 | 104.2 | 101.0 | 103.9 |

So the room above 100 is 1-9 points. A controller at 85 points in strong current is 24 points short of the best
possible, not 15.

**2. Regret in points, stalling world** (mean over seeds; smaller is better):

| controller | recorded | strong | waves | kicks |
|---|---|---|---|---|
| recording controller | 100.8 | 109.0 | 102.6 | 106.7 |
| stall-aware (100 pts) | 0.8 | 9.0 | 2.6 | 6.7 |
| clone | 44.5 | 54.6 | 59.0 | 104.7 |
| none | 15.3 | 123.3 | 32.6 | 54.2 |
| full | 9.5 | 24.3 | 16.0 | 42.4 |
| bca_k1 (seeds 0-1) | 9.8 | 86.9 | 20.4 | 46.3 |
| none (seeds 0-1) | 15.7 | 126.2 | 33.4 | 54.9 |
| perfect band | 7.3 | 29.0 | 12.6 | 36.9 |
| range hinge | 7.1 | 15.7 | 10.3 | 31.7 |
| even pull (matched) | 18.2 | 124.8 | 38.2 | 62.0 |
| stand-in (matched) | 7.9 | 35.5 | 13.1 | 36.6 |

In strong current, BCA (k = 1) closes about 31% of the gap between "none" and the best possible (126 to 87 points
of regret; seed 0: 122 to 95, seed 1: 131 to 79). The range hinge closes about 88% (to 16).

**3. No-stall world** (where the commands missing from the log help): none's regret is 7-13 points, full's is the
same, bca_k1's is 11-27 (caution costs 4-18 points here), the clone's is 53-129.

**4. Where the loss comes from.** Almost all of it is tracking (floats left away from rest), not energy. Stalling
world, strong current, per episode: tracking 58 for the optimum, 71 stall-aware, 96 full, 190 bca_k1, 236 none;
energy 32, 33, 31, 31, 39. The worst 5% of starts show the same order (mean return -186, -226, -283, -524, -538).

**5. The discounted objective agrees with the 100-step score.** A perfect discounted optimiser, scored on the
100-step score, loses only 0.3-1.2 points. Across all controllers the two objectives order 97.6-99.6% of pairs the
same way (Kendall tau 0.95-0.99). Every key comparison (full minus none, BCA minus none, range hinge minus none,
perfect band minus none, none minus clone) has the same sign under both, in every seed and situation, except one
near-tie (no-stall world, seed 4, big waves: -0.65 points vs +0.01 discounted).

**6. Float32 deployment does not matter.** Running a trained actor the deployed way (the environment's float32
observations and commands, the float32 network) changes a start's score by at most 0.0006 points and a bank's mean
by at most 0.00001 points (two actors checked: seed 0's none and bca_k1, all four banks).

## Surprises

- The ceiling is low (point 1). The stall-aware reference is within 1-9 points of perfect.
- In strong current, the best plan holds thruster 1 at exactly the 80% limit in 74% of all steps (holding against
  the current). In the no-stall world it uses throttle past 80% in 17% of thruster-steps there: the commands the
  gapped log lacks are worth having in that world.
- A standard local optimiser given the true stalling thrust and started from all-zero commands sometimes ends far
  from the best plan (up to 16 times the best cost). The exact answer needs the reformulation below.
- The best plan's last command is always 0 (to rounding): the state after step 100 is never scored.

## How the best plan is found

1. In the stalling world, any command past 80% is beaten by the command equal to its own push: same push (so the
   floats move the same way), lower throttle (so less energy). So the best plan never goes past 80%, and below 80%
   the push equals the command.
2. Then the floats' positions are a straight-line (linear) function of the 400 commands, the same function for
   every start. The cost is a sum of squares with each command boxed in (+-0.8, or +-1 in the no-stall world). Such
   a problem has one best answer and exact solvers for it. Each of the 8,000 starts is solved exactly
   (scipy lsq_linear, the bounded-variable method).
3. Nothing is random, so a controller that reacts to what it sees cannot beat the best fixed plan from the same
   start.

## How it was checked (all passed)

- Every best plan was replayed through the world's own step function: the score matches to 4 parts in 10^15.
- A guaranteed bound on how far each plan could be from the best (the Frank-Wolfe gap): at most 2 parts in 10^12.
- A second solver agrees to 6 parts in 10^16. 880,000 random and single-command nudges: none lowered the cost.
- Stalling world, with the true stalling thrust over the full range: pushing any of the 140,651 commands sitting at
  80% further raises the cost (smallest slope 1.2); 800,000 sampled plans (770,000 of them past 80%) and 2,000 local
  searches: none cheaper. (37 of the 400 searches started at the best plan nudged a command past 80% by a rounding
  amount, with no change in cost.)
- Every one of the 130 controllers scores at or below the best plan on every one of its 4,000 starts (smallest
  regret 0.004).
- Every controller's points equal its recorded points in values.json exactly (568 numbers); the anchors recomputed
  equal the pre-registered ones exactly; every actor file matched its recorded SHA-256 (Stage 2: the file and each
  actor against lock.json).
- The discounted best plans (200 starts per situation, 1,000 steps): guaranteed within 6 parts in 10^13.

## Files

- `oracle.py`: all the code, with comments.
- `results.md`: all the tables (sections 1-9).
- `results.json`: every number (per controller, per seed, per situation; the checks; the files read).
- `timings.json`: seconds per phase (the machine was shared with other jobs).
- `cache/`: the best plans themselves (`optimum100.npz`, 8,000 x 100 x 4 commands), the discounted plans, per-start
  scores of every controller, and the logs. Everything here can be rebuilt by rerunning.

## How to rerun

From /tmp, with `PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu PYTHONPATH=/home/user/bca`:

`/root/bcavenv/bin/python /home/user/bca/outputs/shift/oracle/oracle.py all`

It takes about 17 minutes with 8 processes. Single phases: solve, validate, nonconvex, discounted, score, float32,
report.

## Limits

- The best plan knows the step number, so it can relax near the end of the episode; a controller that sees only the
  state cannot. Point 5 suggests this is worth about 1 point or less.
- Gate actors: check 2 (both alpha rungs). The room check's actors were not scored.
