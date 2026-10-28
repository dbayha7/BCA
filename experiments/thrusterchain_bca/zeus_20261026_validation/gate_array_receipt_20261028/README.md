# Gate array receipt, 28 October 2026

This folder pins the six saved run arrays of the two completed conditional-hole gates. Git has tracked the arrays
since commits `e0a0777c` and `fe751257`. But no gate record named them or pinned their SHA-256 hashes, and nothing
checked them against the gate reports. This folder adds both. Neither gate folder was changed to make it.

- **The gates:** the stall-knee retry (`../conditional_ood_gate_v1_retry_20261028`, failed) and the central-quartile
  gate (`../conditional_ood_gate_central_quartile_v1`, passed).
- **The arrays:** each gate's worker wrote three of them, in this order:
  - `prelabel_probes.npz`: the critic's two scores (Q1 and Q2) for six candidate commands at 64 anchor states;
  - `candidate_values.npz`: each candidate's true return under the frozen actor;
  - `free_policy.npz`: the actor's own 16 episodes of 100 steps.

  The worker writes the first file before it computes the true returns. That order comes from `worker.py`; this
  receipt cannot prove it.
- **The record:** `RECEIPT.json` holds every result below, written by `check_arrays.py`.

## What is pinned

| Gate | Array | Bytes | SHA-256 |
|---|---|---|---|
| Stall-knee retry | `prelabel_probes.npz` | 16,337 | `193ae58f492cb40a401a28aadc89e306c26c772ef3f97ea7a178a4fc64745406` |
| Stall-knee retry | `candidate_values.npz` | 13,888 | `fe5602dfafe44b43618f2ceaabf1cdab178f615804a33fa4e86ffe87b8c2f146` |
| Stall-knee retry | `free_policy.npz` | 102,255 | `d1bb867cc63c4117602f0e1662a817468ccece198c6a2697e8c2e2a5e1f76107` |
| Central quartile | `prelabel_probes.npz` | 16,279 | `8d8dcccbacad4ec5b892065653a296eaf56160a9f191ad5b1138177d55933ba3` |
| Central quartile | `candidate_values.npz` | 13,844 | `1b80f960dd66332c040b83f3c7871dec76822cffaa290acee9fffd6db9869933` |
| Central quartile | `free_policy.npz` | 102,414 | `f7e3b7df58c3b45b29440bcce2f5fdb7c63302cc73be80abab50b27a66762c1f` |

**Where the hashes live.** The same six hashes are constants in `check_arrays.py`, so a rerun verifies them rather
than only recomputing them.

For each array, `RECEIPT.json` also records:
- its git blob ID;
- its member names, shapes and data types;
- its modified time, in UTC and in local time with the offset.

## What the check does

`check_arrays.py` runs 42 checks for each gate, in six groups.

1. **Provenance, 7 checks.** These tie the arrays to the run record.
   - `report.json` matches the hash the supervisor recorded in `RESULT.json`.
   - `PREPARED.json` matches the manifest hash the worker verified before and after the run.
   - `contract.json` matches the contract hash the run recorded.
   - Every file of the gate folder pinned in `PREPARED.json` still has its pinned hash: 14 files for the retry and
     15 for the central gate.
   - The `geometry.py`, `reference.py` and `analysis.py` that this check runs are those pinned copies.
2. **The arrays, 15 checks.** For each of the three files, it confirms that:
   - its SHA-256 equals the pinned hash above;
   - the committed copy equals the file on disk;
   - it is unchanged since the commit that first recorded it;
   - its member names, shapes and data types are as `worker.py` writes them;
   - every floating-point member is finite.
3. **Identities, 11 checks.**
   - **Agreement between the first two arrays:** Q1, the commands, the states, the times and the episode IDs agree
     bit for bit between the first two arrays.
   - **Agreement with the report:** the anchor times, episode IDs and episode seeds equal those recorded in
     `report.json` (`data_identity`).
   - **Time cycle:** the anchor times follow the contract's cycle of 0, 25, 50 and 75.
   - **Free-play seeds:** the free-play start seeds are the first 16 anchor seeds, as the worker draws them.
   - **Free-play count:** the count in `RESULT.json` agrees with the arrays' fixed 16 × 100 shape. This checks
     `RESULT.json`, not the arrays' content.
4. **Code invariants, 6 checks.** These are rules the writing code always obeys.
   - **Observations:** the anchor observations equal `reference.observation(states, times)` bit for bit.
   - **Time column:** each free-play episode's time column runs 1.00, 0.99 and so on down to 0.01, in order.
   - **Start states:** the first free-play observation equals the anchor observation recorded at time 0 in the same
     episode. Four anchors allow this comparison.
   - **Candidate commands:** slots 1 to 5 use the gate's fixed first commands, and every slot keeps the actor's
     other three commands.
   - **Fixed sentence:** the report's `reset_scope` sentence is the exact string that
     `reference.free_policy_rollouts` returns, read from the code itself. The reanalysis is fed the code's own copy,
     so the comparison checks the report's copy as well.
5. **Reanalysis, 1 check.** This reruns the gate's own `analysis.analyze_values`, called exactly as `worker.py`
   calls it. It then compares every value in the output with the `values` section of `report.json`.
   - **Where the free-play counts come from:** they are rebuilt from `free_policy.npz`.
   - **Central gate:** the rebuild uses `geometry.region_masks`, the same function its `reference.py` uses.
   - **Retry:** its `reference.py` writes the same formula inline, so for that gate the rebuild is an equivalent
     re-implementation.
6. **Controls, 2 checks.** These show that the comparison can fail.
   - **Small nudge:** changing one true return by 1e-9 must produce mismatches.
   - **Random inputs:** 10 draws each keep the real arrays but replace every input the analysis reads with random
     values: the two load terms, the actor's first command, the true returns, Q1, and free play's commands and
     returns. Every draw must fail to reproduce the report.

**The pass rule.**
- **Float tolerance, fixed before the first run:** a float may differ from the report by at most
  1e-12 × max(1, |reported value|). This allows for the gates having run numpy 2.5.4 on Linux, while this check ran
  numpy 2.2.6 on Windows.
- **Made stricter after review:**
  - integers, booleans, strings, empty values, dictionary keys and list lengths must match exactly;
  - an integer paired with a float is a mismatch;
  - NaN never matches;
  - an infinity matches only itself.

**Read-only, as measured.**
- **No simulator or learner modules** were loaded.
- **Both gate folders were left exactly as they were:**
  - their files and hashes were the same before and after;
  - the files on disk are exactly the files git tracks;
  - there are no untracked directories and no `__pycache__` folder;
  - git status was clean, ignored files included.
- **The frozen code ran from its hashed bytes.** Each gate module runs from the exact bytes the script hashed, so a
  stale compiled file could never stand in for the frozen code.
- **Git touched nothing.** Git runs with optional locks off, so even `git status` cannot rewrite the index.

**Read-only by construction.** The script opens no dataset file or protected split, and it trains and fits nothing.
It writes only `RECEIPT.json`.

## Results

**Every check passed.** Both gates passed all 42 checks. In each, rerunning the analysis reproduced all 2,041 exact
values and all 2,226 floats **bit for bit**, with no difference at all.

**Both controls behaved as they should.**
- **Small nudge:** changing one true return by 1e-9 produced 5 mismatches in each gate.
- **Random inputs:** every one of the 10 random draws failed. Per draw, between 2,648 and 2,719 of the 4,267 values
  changed for the retry, and between 2,641 and 2,714 for the central gate.
- **Values that depend on the data:**
  - **How many changed:** 3,120 values for the retry and 3,146 for the central gate changed in at least one draw.
    That is a lower bound on the values that depend on the data.
  - **The rest:** they are either constants, such as loop indices, fixed counts and fixed sentences, or values that
    agreed by chance in every draw.

| Reproduced number | Stall-knee retry | Central quartile |
|---|---|---|
| Value decision | Failed | Passed |
| High-load anchors with a gap of at least 1 unit | 8 of 29 | 20 of 23 |
| Free-play high-load steps that entered the hole | 0 of 927 | 576 of 773 |
| Free-play episodes with an entry | 0 of 16 | 14 of 16 |
| Anchors where the actor's own command was in the hole | 2 | 8 |
| Raw-Q1 choice against the actor's command: improved, worsened, tied | 34, 6, 24 | 29, 16, 19 |
| Mean regret within the bank | 0.76 | 0.40 |

The four critic-error cells (rows, mean absolute error and mean signed error) are in `RECEIPT.json`.

## Rerun

From the repository root:

```bash
.venv/Scripts/python.exe -I -B experiments/thrusterchain_bca/zeus_20261026_validation/gate_array_receipt_20261028/check_arrays.py
```

- **Exit code:** 0 when every check passes, including the six pinned hashes.
- **Writes:** it rewrites only `RECEIPT.json`.

## Limits

- **Bytes, not timing.** This fixes the arrays' bytes as they are now.
  - **Write order:** it cannot prove that `prelabel_probes.npz` was written before the true returns existed.
  - **The gap before the commits:** it cannot prove that nothing changed between the runs and the commits that first
    recorded the arrays. That gap was about five hours for the retry and about three for the central gate.
  - **Modified times:** they agree with the write order, but they are descriptive only and easy to change.
- **Some content is pinned by hash only.** No check ties these parts to the report or to a code invariant, so a
  change to them could leave every reproduced number unchanged:
  - the second critic head, Q2;
  - the float64 states beyond their float32 rounding, which the observation rule ties;
  - the 11 state and observation coordinates other than the two load terms, beyond the observation and start-state
    rules;
  - the two load terms, at the anchors and in free play, beyond which side of the 0.4 cutoff each falls on;
  - every first command (the actor's at the anchors, and every free-play step), beyond which side of the band edges
    it falls on;
  - the actor's other three commands: at the anchors, beyond their being the same in every slot; in free play, not
    checked at all;
  - which free-play trajectory belongs to which seed and return, beyond two ties: the four episodes that start at an
    anchor recorded at time 0 (episodes 0, 4, 8 and 12), and the per-episode hole counts;
  - the order of steps within free play, beyond the time column.

  The pinned hashes still catch any change to any of these.
- **The support test is not rechecked.** It needs training rows, which are not in these arrays; its record is each
  gate's `geometry.json`. So the overall gate decision is not recomputed.
- **Development data only.** These are consumed development anchors. They can serve as a fixture for checking code,
  never as evaluation data.

## Script history

**Version 1, 21 checks.**
- **The first run** failed one check because of a bug in the script: it looked for the supervisor's free-play count
  under `ledger` in `RESULT.json`, but the count is under `counts`.
- **After that fix,** all 21 checks passed.

**First independent review.**
- **Who:** three reviewers, each followed by a skeptic who tried to refute their findings. One rebuilt every number
  from scratch, one audited the script, and one checked the claims.
- **What held:** every hash and every reproduced number.
- **What they found:**
  - checks that could not fail;
  - missing links to the run record;
  - unchecked identities and code invariants;
  - a risk that a stale compiled file could stand in for the frozen code;
  - loose integer-and-float and NaN matching;
  - a folder test blind to ignored files;
  - times recorded only in local time;
  - some overstated wording.

**Version 2, 36 checks.**
- **New checks:** 15 more checks, two of them the controls.
- **Other changes:**
  - the frozen code runs from its hashed bytes;
  - the stricter integer-and-float and NaN rules;
  - a folder test that sees ignored files and `__pycache__`;
  - times in UTC.
- **Changed check:** the old check of `contract.json` against `PREPARED.json` now checks it against the run record.

**Second independent review.**
- **Who:** two reviewers, each followed by a skeptic. One checked every claim, and one changed copies of the arrays
  in memory to test whether each check can fail.
- **What held:** every check could fail, and every number held.
- **What they found:**
  - the list of content pinned by hash only was incomplete;
  - the random control's count had been mislabeled as the exact number of data-dependent values;
  - the `reset_scope` check matched any string in the code;
  - an infinity could match a finite float;
  - `git status` could rewrite the index;
  - a rerun did not verify the pins;
  - the version 2 counts and pass-rule history were misstated.

  One further finding was refuted: floats inside the tolerance are the documented rule, not a hidden gap.

**This version, 42 checks.**
- **New checks:** the six pinned hashes as constants, with a check of each, and a check that each array is unchanged
  since its recording commit.
- **Fixed checks:**
  - the exact `reset_scope` string, read from the code;
  - 10 random draws instead of one, reported as a lower bound;
  - the infinity rule;
  - git with optional locks off.
- **Rewritten:** the full list of content pinned by hash only.

All 42 checks pass.
