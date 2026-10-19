# Arms A, B, C and result F's no-band control (exploratory, 16-17 Oct 2026)

**Exploratory.** Data seeds 0-4 were already seen. Nothing here is confirmatory; a confirmatory claim needs fresh
seeds 15-19, its own pre-registration and the owner's go.

This folder keeps what a cloud session ran in a sandbox copy of this repository (byte-identical code, at
`.../scratchpad/iw_rerun/repo`). The absolute paths inside the scripts and records are that sandbox's.

## The outcome

No arm earned BCA credit under the marks frozen before any code was written.

| Arm | Strong-test result (points, 5-seed mean) | BCA's credit |
|---|---|---|
| A, a pull that copies only good rows | +11.3 medium-replay, +10.1 medium (5 of 5 seeds each); no harm anywhere | None: a plain margin or a matched share did as well or better ("Filtering helps; the band adds nothing over a plain margin.") |
| B, a certified throttle ceiling | Stopped by its kill rule K1 at the seed-0 screen (thruster-1 ceiling 0.85-0.90 with 42-78 tries in the 80-85% rung) | None; no return scored |
| C, less pull, guarded | +15.5 random, +8.9 medium-replay, +8.5 medium (5 of 5); no harm anywhere | None: halving the pull alone gave +15.4, +8.7, +7.9, and the guard was void |
| F0, result F without the band | Result F beat it by +0.14 (medium-expert) and +0.08 (expert), against a +1.0 bar | None: "Result F's gain comes from refusing credit past the data, not from BCA's band." |

## Files

- `PASSMARKS_arms_ABC_and_F_control.md` and its `.sha256`: the pass marks and kill rules, frozen first.
- `code/`, `run_arms.sh`: the program (screen, kill checks, arms, scoring).
- `SCREEN_seed0.*`, `KILLS.*`, `RESULTS.md`, `results.json`: the seed-0 screen, the kill checks on all seeds, the
  scored marks. Each JSON has a `.sha256` sidecar.
- `verify/`: two independent re-computations (marks from raw scores; kill rules from raw records and logs) and the
  wording check against the frozen file. `VERIFIED_NUMBERS.md`, `KILLS_CHECK.md` and `WORDING.md` are the summaries.
- `state/`: the runner's job states.
- `records/`: the run records (JSON and their `.sha256`), copied from the sandbox's `runs/` tree:
  - `arms_abc_f0/`, `channel/` (result F's A2 seeds 1-4), and the five quality folders: what this run made under
    `runs/shift_iw_v5_explore/`;
  - `shift_iw_v5_sandbox/`: the seed 1-4 data reports, hosts, room checks and bands the run built as inputs;
  - `EXPLORATORY_*.txt`: every file the runs made, with its SHA-256.

  Actor, band and data binaries (`*.msgpack`, `*.hdf5`) are not committed; their SHA-256 values are in the
  `EXPLORATORY_*.txt` lists and the records.

Related folders beside this one: `../routes2/` (the proposal and the capped pull's diagnosis), `../resultF_rerun/`
(result F's seed-0 cloud rerun) and `../situations_seed0/` (the seed-0 learned actors in the main test's four
situations).
