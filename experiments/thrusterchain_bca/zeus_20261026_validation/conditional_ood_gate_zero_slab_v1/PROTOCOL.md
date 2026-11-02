# Conditional OOD design gate: zero-thrust slab (candidate A)

This is a no-training development screen of a replacement conditional hole. It runs on the pinned Titan Unifloral
TD3+BC checkpoint, with frozen actor, critic and normalization. The gate does not train any model, fit calibration,
compute a BCA-adjusted score or establish a missing-label effect. Every number and the rule that set it is in
`DESIGN.md`. The machine-readable specification is `contract.json`, authenticated by `DECISION_FROZEN.json`.
`PREPARED.json` authenticates the executable sources and inputs. There is one attempt, with no automatic retry.

## Region

Withhold logged training rows whose executed first-thruster command `a0` lies within 0.1 of zero, at states whose
observed load `g` is at least 0.4.
- **Load:** float32 observation coordinates 10 and 11, each promoted to float64 and then added.
- **Command:** the executed float32 command, promoted to float64.
- **Alternative bands:** `[−0.3, −0.1)` and `(0.1, 0.3]`. Both stay below the stall knee.
- **Low load:** the same commands keep their labels.

## Fixed screen

- **Geometry and support.**
  - **Neighbors.** For each anchor, take the 64 nearest original training rows in the same ten-step time bin and load
    side, excluding the anchor's episode. Ties break by source row ID. Distance is RMS over 13 coordinates,
    standardized by original training statistics. There is no absolute radius.
  - **Locality.** An anchor passes if its distance to the 64th neighbor is at most the 0.95 quantile of the same
    statistic over training rows at the same time and side. Each training row's own episode is excluded from its pool.
  - **Counts.** A passing anchor also needs 32 rows from 8 episodes before and after removal. It needs 2 original band
    rows from 2 episodes, and 2 retained rows from 2 episodes in each adjacent band.
  - **Hurdle.** At least 80% of anchors on each load side must pass, with at least 16 anchors per side.
  - **Also checked:** global coverage, per-bin and per-depth coverage, exact random-removal counts, state balance and
    calibration eligibility, as in the contract.
- **Within-model cells.** Errors are reported for high and low load crossed with inside and outside the band.
- **Calibration.** The original calibration rows (seed 2026102362) are kept where they fall outside the hole. An
  excluded row is replaced once, uniformly among eligible rows in its episode, using seed 2026102902. No calibration
  is fitted.
- **Anchors.** 64 anchors are drawn from the last 64 of the 128 development tuning-validation groups, sorted by group
  ID, at cyclic times 0, 25, 50 and 75. Each state and time is restored exactly by recorded-prefix replay, checking
  every observation.
- **Candidates.** Six commands share the actor's other three coordinates: the actor's own, −0.2, `nextafter(−0.1, 0)`,
  0.0, `nextafter(0.1, 0)` and +0.2. Each is continued with the same actor to step 100, at γ = 0.98, with no terminal
  bootstrap.
- **Value gate.** At least 25% of high-load anchors must have a center-to-flank return gap of at least 1 unit.
- **Exposure gate.** The actor runs freely for 16 full episodes, from the first 16 anchor start seeds. It needs at least
  100 high-load steps, entry into the hole on at least 1% of them, and entry in at least 2 episodes.
- **Raw critic errors.** These are reported with `tau(v) = 1e-4 + 1e-5·|v|`. No BCA-adjusted score is examined.

The three components each finish once, even if geometry fails. An integrity, runtime or resource violation stops the
run immediately. A failed criterion remains a failed design gate. A pass only permits proposing a separate,
reviewed training study.

## Budget

At most 900 wall-clock seconds, including imports and compilation. At most 28,104 simulator transitions: 2,400 prefix,
24,000 candidate continuation, 1,600 free-policy and 104 parity. Zero critic, actor, auxiliary or calibration updates.
At most 960 permitted source episodes are decoded, and protected roles stay opaque. The support reference adds only
CPU distance arithmetic on training rows.

## Failure

Any failure shelves this exact region. There is no change to the band, threshold, quantile or neighbor count, and no
automatic rerun. It is written up in the standing format: The idea / What we tested (and the kill rule) / What happened
/ Why.
