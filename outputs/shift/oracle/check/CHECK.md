# Independent check of the exact optimal reference (outputs/shift/oracle/)

**Verdict: everything passes. I found no bug in the computation, and no key number changes.** There are three small
reporting caveats at the end.

This check uses none of `oracle.py`. It touches the world only through the repository's own
`experiments/shift/thruster_chain.py`. Nothing was trained and no BCA band was fitted. All output is in this folder:
`check_oracle.py` (the code), `results.json` (every number) and `cache/`.

Words used here:
- **Return**: the score of one 100-step episode.
- **Points**: the gate's 0-to-100 scale.
- **The knee**: 80% throttle. In the stalling world, pushing a thruster past it gives less push.
- **Plan**: a fixed list of 100 commands from one start.
- **Relative difference**: a difference divided by the size of the number.

## 1. The best possible return, re-derived another way: PASS

- **Method.** I used a different solver: a general-purpose local optimiser (L-BFGS-B) working on the 400 raw commands.
  - The cost is minus the return of a 100-step rollout of the repository's `dynamics_jax`, in 64-bit floats.
  - The slopes come from JAX automatic differentiation of that rollout.
  - There is no matrix reformulation.
  - Every reported return comes from replaying the plan through the numpy `dynamics()`.
- **Starts checked.** 1,000 starts: 125 for each world and situation. Of each 125, 100 are spread over the bank and 25
  are the starts where the oracle's plan sits on the throttle limit most often (the hardest cases).
- **Result.** The returns agree to at most **3.4e-15 relative** (worst case: no-stall world, big kicks, start 725).
  - My plan is never better than theirs by more than rounding (1.5e-15).
  - The largest difference in any single command is 4.6e-7.
- **A second, independent guarantee for all 8,000 of their plans.** I recomputed the Frank-Wolfe gap (a guaranteed
  upper bound on how far a plan's cost is above the best possible cost) using my own JAX slopes rather than their
  matrix.
  - The gap is at most 2.5e-10 relative. That limit comes from my slope method, not from their plans.
  - Every command is within its bounds: ±0.800000012 in the stalling world, ±1 in the no-stall world.

## 2. The stalling world over the full range, with the true stalling thrust: PASS

The oracle's argument is that replacing each command by its own push gives the same float positions with less energy,
so the best plan never goes past the knee.

- **The thrust rule itself.** I checked every 32-bit command between 0.5 and 1 (16.8 million, both signs) and 10
  million random 64-bit commands. In every case:
  - the push is at most the knee;
  - the push is at most the command, and strictly less past the knee;
  - applying the rule twice gives exactly the same number;
  - the push equals the command at or below the knee.
- **Random plans.** I tried 40,000 random full-range plans on 80 starts (39,254 of them use commands past the knee).
  - Replacing each command by its own push gives exactly the same float positions in every case.
  - The cost never rises, and it strictly falls whenever a command is past the knee.
  - No sampled plan costs less than the best plan.
- **Local searches.** I ran 640 local searches with the true stalling thrust and bounds ±1: 80 starts × 8 starting
  plans (all zero, three random, random past the knee, the recording controller's full-throttle pattern, all at full
  throttle, and the best plan pushed past the knee).
  - None beat the best plan. The closest was −4.2e-16 relative, which is rounding.
  - After the replace-by-push step and a re-solve inside ±0.8, every run reached the best cost to 3e-15.
  - Their surprise is confirmed: started from all-zero commands, local search ended up to 10.5 times the best cost
    on my starts (they saw up to 16 times on theirs). Many runs stop with throttle past the knee.

## 3. Anchors, the points formula and the recorded scores: PASS

- **Anchors** (the 0-point and 100-point returns). I wrote my own versions of:
  - the textbook controller gain (from scipy's equation solver; identical to the repository's);
  - the recording controller (its own noise and coin streams, `default_rng([7, 1])`);
  - the stall-aware and rising-thrust controllers;
  - the rollout loop.

  They match the pre-registered anchors to 5.7e-14. The oracle's anchors equal the pre-registered ones exactly.
- **Points formula.** The oracle uses the same formula as `acceptance._evaluate` → `learners.evaluate_policy` →
  `truth.points`: points = 100 (J − J_recording) / (J_top − J_recording) for each start, then the mean over the bank.
- **Recorded scores.** I re-scored all 142 saved actors (gate α 2.5 and 5, seeds 0-4; Stage 2 seeds 0-1, both
  worlds) with my own numpy network and the normalisation from the cloned actors' records.
  - All **568 recorded points match `values.json` exactly** (largest difference 0.0).
  - Every file's SHA-256 matches its record (Stage 2: against `lock.json`).
  - Stage 2 actors reproduce with the gate's normalisation for the same seed, so the float32 check's choice
    (`oracle.py:983`) is right.
- **The oracle's tables.** Recomputed from their plans in my own loop, the optimum's points equal theirs exactly: 100.8,
  109.0, 102.6, 106.7 in the stalling world and 100.6, 104.2, 101.0, 103.9 in the no-stall world.
  - The regret table equals theirs exactly, for every family listed.
  - Every controller is at or below the best plan on every start (smallest regret 0.0042).

## 4. Reading oracle.py for bugs: PASS (none found)

- **Horizon.** Rewards are charged at steps t = 0..99, and the state after step 100 is never charged
  (`oracle.py:249-271`). Their reported mean return equals this sum exactly. Shifted windows would give different
  numbers (stalling world, strong current: −90.32 reported, against −88.96 or −91.49 for the shifted windows). The last
  command is 0 to within 4e-15, as this horizon implies.
- **Reward sign and coordinates.** The cost is |s|² over coordinates 0-9 (the floats' positions and speeds) plus
  0.3 |a|². That matches `dynamics()` to 4e-16. The current and wave coordinates are not charged.
- **Knee.** The knee is `TC.KNEE` = float32(0.8) = 0.800000011920929. A command at the knee does not stall. The largest
  stalling-world command is exactly the knee, and none goes past it. 74.1% of thruster-1 steps sit at the limit in
  strong current, so their "74%" is right.
- **Clipping and bounds** are correct in both worlds.
- **Bank seeds.** I redrew the banks myself from `default_rng([20261012, 2, 9])` in the documented draw order. All four
  are identical to `bank_starts`.
- **The rest of the code** is right:
  - the reduced least-squares form (`oracle.py:327-335`);
  - the slope of the cost just past the knee (`oracle.py:500-503`). The 1.200 minimum is the theoretical floor,
    2.5 × 0.6 × 0.8.
  - the discounted problem's scaled variables and backward pass (`oracle.py:621-637`).

## Caveats (reporting, not computation)

1. **Mixed seeds in the summary message.**
   - Its "none" row (strong current 123.3) is the gate's mean over seeds 0-4. Its "bca_k1" row (86.9) uses seeds 0-1
     only. The like-for-like "none" for seeds 0-1 is **126.2**.
   - The tracking figures mix seeds the same way: 96 for full and 236 for none are seeds 0-4; 190 for bca_k1 is seeds
     0-1.
   - The README gets this right (it has a "none (seeds 0-1)" row).
2. **The one sign disagreement is labelled loosely.** The "−0.65 points against +0.01" exception (no-stall world,
   seed 4, big waves) quotes the whole-bank points difference. The sign test itself compared −0.18 (100-step return
   over the first 200 starts) with +0.01 (discounted value) (`oracle.py:1206-1211`). The conclusion, a near-tie, is
   unchanged.
3. **Which version of the code made the saved results.**
   - `oracle.py` was last edited at 21:03:57. Every saved intermediate result was made before that: the solve at
     20:24, the score phase at 21:03:19. Only the final report ran on the final file.
   - The numbers are confirmed independently above, so nothing changes. But the "17-minute full run" has not been
     shown end to end on the final file.

Run time of this check: about 22 minutes with 6 processes on the shared machine. No `__pycache__` folder was created
(the only two predate this session), and nothing was installed.
