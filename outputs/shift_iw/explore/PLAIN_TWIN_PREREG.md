# Plain TD3+BC coverage with its two critics' disagreement: written before running

Date: 2026-10-13. EXPLORATORY: this measure is outside the frozen version-5 specification, and the version-5 report and
its marks do not change.

## Why

The owner asked: "we need coverage results for plain td3 as well to compare against both bca versions". Plain TD3+BC
gives one number per command and no band. The owner chose its built-in uncertainty signal as its error bar. Question:
"Plain TD3+BC gives only a single number per command, with no band. What should count as its error bar ...?" Answer:
"Its two critics' disagreement (Recommended)".

## What is measured

- TD3+BC trains two critics, Q1 and Q2, and uses the lower one, q = min(Q1, Q2).
- Plain band: q ± |Q1 − Q2|. It is not tuned or scaled.
- "Covered" uses the same rule as BCA's coverage: the one-step target y lies within the band.
- Same everything as BCA's version-5 coverage stage: the host critic trained with the 80% hold removed ("gap" host),
  and its full-data twin ("full" host) for comparison; the same test steps (held-out training-law steps for no shift,
  and the mild and strong test logs); the same targets y. The script reads y, q and the in-ball flags from each seed's
  steps.npz and recomputes Q1 and Q2 at the same rows. It checks that min(Q1, Q2) reproduces the saved q.
- Reported per quality and shift, outside and inside the removed 80% hold region:
  - the coverage of the plain band;
  - next to it, unweighted BCA and per-step exact-weighted BCA (the weighted arm refuses steps inside the removed
    region);
  - the median plain-band width as a share of BCA's unweighted median width.

## Expectations (written before running)

1. Plain coverage is far below the 90% target with no shift, on every quality: below 50%. Both critics learn from the
   same data and the same targets, so they agree much more closely than the size of the one-step error.
2. Plain coverage drops, or at best stays flat, from no shift to the strong shift. Nothing corrects it.
3. Inside the removed 80% hold region, the plain band is not clearly wider than outside it: the median width inside is
   less than twice the median outside, so it gives no usable warning about the missing action. Weighted BCA refuses
   all of those steps.
4. The plain band's median width is well under half of unweighted BCA's median width.

If an expectation fails, the write-up says so and gives the likely reason.
