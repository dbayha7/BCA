# Frozen outcomes and page wording: arms A, B, C and F0 (EXPLORATORY)

**EXPLORATORY.** Data seeds 0-4 are seen seeds. Nothing here is pre-registered or confirmatory. A confirmatory claim needs fresh data seeds 15-19, its own pre-registration and a new go from the owner.

This file decides, from the frozen pass-mark file and RESULTS.md's numbers alone, the frozen outcome and the frozen sentence of each arm, and what the owner's page may and may not say.

- Frozen file: `routes2/arms/PASSMARKS_arms_ABC_and_F_control.md`, SHA-256 `a073be97003938c112e9ba77581a311e812fe5851f7e64211a11fb4bf47d9ecf`. It matches its sidecar (`sha256sum -c` OK).
- Numbers: `routes2/arms/RESULTS.md` (SHA-256 `ade7a08e…`), `results.json` (`7c9a1d30…`, matches its sidecar), `KILLS.json` (`96db647e…`, matches its sidecar), `SCREEN_seed0.json` (`9c940a8a…`, matches its sidecar). `verify/VERIFIED_NUMBERS.md` recomputed every mark from the raw records and found no mismatch.
- Two post hoc rows (marked as such below) come from `verify/wording_check.py`, which reads arm C's raw evaluation records with plain `json` and writes `verify/wording_check.json`. They decide nothing.
- Units: "points" are 0 for the random-quality data and 100 for the stall-aware controller. "The strong test" is the strong start bank (lam +2), the primary bank. Each difference is arm minus comparator, as a 5-seed mean over seeds 0-4. "4 of 5" means the number of seeds with a positive difference.
- Sentences in quotation marks are copied verbatim from the frozen file. Where a sentence has no number slots, the numbers are given beside it.

---

## Labels every page line about these arms must carry

- "Exploratory; seeds 0-4 already seen; not pre-registered." Each block (arm A, arm C, F0) carries this label, and so does any table taken from it.
- Arms A and C were trained **actor-only**: only the policy was trained, against TD3+BC's frozen critic. Their comparator is "TD3+BC trained the same way" (same start, same rows, same alpha), not TD3+BC trained end to end. Co-training was not part of the owner's go.
- The test bed is the weighting test with the 80% hold removed from training.
- Information-only rows carry "information only; decides nothing". Rows computed after the results were known also carry "post hoc".
- The lower bounds in RESULTS.md decide nothing. Do not present them as significance tests.

---

## Arm A: credible-target pull ("a pull that copies only good rows")

### Outcome

P1 met, P2 met, P3 met, **P4 not met**. GO not reached. No kill rule fired. Passing qualities: medium-replay and medium. BCA gets no credit (P4 is the mark that decides that, section 4.2).

| Mark | Value (strong test unless a bank is named) | Bar | Met |
|---|---|---|---|
| P1, medium-replay | +11.27, 5 of 5 | at least +2.0, 4 of 5 | yes |
| P1, medium | +10.06, 5 of 5 | at least +2.0, 4 of 5 | yes |
| P1, medium-expert | +0.93, 5 of 5 | at least +2.0, 4 of 5 | no (below +2.0) |
| P2, all 15 quality-bank pairs | lowest +0.43 (expert, none bank) | at least −1.0 | yes |
| P3, medium-replay / medium | +13.25 / +9.68 | at least +1.5 | yes |
| P4 vs margin 0, medium-replay | −2.47, 1 of 5 | at least +1.0 | no |
| P4 vs best plain margin (1 spread), medium-replay | −3.11, 0 of 5 | at least +1.0 | no |
| P4 vs matched share, medium-replay | −0.18, 1 of 5 | at least +1.0 | no |
| P4 vs margin 0, medium | +1.20, 4 of 5 | at least +1.0 | yes |
| P4 vs best plain margin (2 spreads), medium | −0.42, 1 of 5 | at least +1.0 | no |
| P4 vs matched share, medium | −0.26, 1 of 5 | at least +1.0 | no |

### Frozen sentence (4.4, "P1, P2 and P3 met, P4 not"), filled

> "Filtering helps; the band adds nothing over a plain margin."

The frozen file adds: "It names the rival that tied or won." Filled:

> Rivals that tied or won. On medium-replay: margin 0 won (A1 − margin 0 = −2.47 points, 1 of 5 seeds positive); the best plain margin, 1 residual spread, won (−3.11, 0 of 5); the matched-share margin tied (−0.18, 1 of 5). On medium: the best plain margin, 2 residual spreads, tied (−0.42, 1 of 5); the matched-share margin tied (−0.26, 1 of 5). Margin 0 on medium did not tie or win: A1 beat it by +1.20, with 4 of 5 seeds positive.

No harm line is added, because P2 was met.

### How "tied or won" is read

- A rival "tied or won" on a passing quality when A1 failed to beat it by P4's bar, that is, when A1 − rival is below +1.0 on the 5-seed mean. This is the bar the frozen file uses to decide whether A1 beat a rival. RESULTS.md's list of five uses the same reading.
- Within that set, a rival **won** when it was ahead by at least the same 1.0 points (A1 − rival of −1.0 or lower). It **tied** when the gap was under 1.0 either way. This split is this file's reading. The frozen file only requires the rivals to be named.
- All five named rivals have a 5-seed mean at or above A1's, so "did as well or better" is true of every one of them.
- The best plain margin is chosen after the fact on each quality (section 3.2), which favors the rival: 1 spread on medium-replay, 2 spreads on medium.

### Plain language for the page

> **A pull that copies only good rows.** TD3+BC pulls its policy toward every recorded command. This version skips the recorded commands that the critic rates clearly worse than the policy's own choice. BCA's band sets what "clearly" means. Only the policy was trained, against TD3+BC's frozen critic.
>
> On medium-replay and medium data, it scored 11.3 and 10.1 points above TD3+BC trained the same way, on the strong-current test. All 5 seeds agreed. It lowered no score on any data quality or test. On medium-expert data it gained only 0.9 points, below the 2-point bar.
>
> BCA's band was not the reason for the gain. On medium-replay, skipping every row the critic rated worse at all (no margin) beat it by 2.5 points. A plain fixed margin with no BCA beat it by 3.1 points. On medium, a plain fixed margin, and skipping the same number of rows without the band's shape, each came within half a point of it. So skipping bad rows helps, and BCA's band adds nothing over a plain margin.

Plain names for the rivals: "margin 0" is "no margin: skip every row the critic rates worse at all". "1 (or 2) residual spreads" is "a plain fixed margin, 1 (or 2) times the typical size of the critic's error, measured without BCA". "Matched share" is "skip the same number of rows, the worst-rated ones, without the band's shape". "Untargeted" is "skip as many rows, picked at random".

### Forbidden for arm A

- That BCA's band picks the right rows, targets bad rows, or made the gain. P4 failed on both passing qualities.
- "Arm A passed", "a GO", or calling the pull a BCA success.
- That it gained on medium-expert or expert. Those rows only pass the no-harm mark.
- "TD3+BC" with no mention that the policy was trained alone against a frozen critic.
- Using code names (A1, margin0, plain_1s, matched_share) on the page.

---

## Arm B: certified throttle ceiling

### Outcome

**Stopped by kill rule K1** at the seed-0 screen. That is "the certificate missed the stall" (section 5.5). No return of arm B was computed, so G1-G4 are all not met, GO is not reached and G4 is not met. Arm B gets no credit. The room check K0 held (knee clip − host, medium-expert, seed 0: +6.99 points), and so did K4 (BCA's map clipped 28.8% of the host's thruster-1 commands). So the stop is not for lack of room.

### Frozen wording, filled

Section 5.4 has no line for a fired kill rule. The frozen stop line (4.4's pattern, and 3.3's "stopped by <rule>") applies:

> "Stopped by K1 on medium-replay, medium and medium-expert, seed 0: BCA's thruster-1 ceiling was 0.85 or higher with at least 30 tries in rung (0.80, 0.85]."

The values were: medium-replay, ceiling 0.90 with 42 tries in that rung; medium, 0.90 with 54 tries; medium-expert, 0.85 with 78 tries. Random (0.80, 12 tries) and expert (0.80, 0 tries) did not fire.

Section 5.4's standing rule also applies: never "BCA improves TD3+BC". Section 5.4's "G1 not met" line ("The certified ceiling did not raise return over TD3+BC on medium-expert.") is **not** used. It would read as a measured result, and no return of arm B exists.

### Plain language for the page

> **A certified throttle ceiling.** This arm was stopped by its own safety rule before any score was computed. In the first check (data seed 0), BCA's certificate let thruster 1 run up to 85% or 90% on three data qualities, past the 80% point where the stall begins. It did so even though 42 to 78 held-out tries sat in the 80-85% step. The rule frozen before the run calls this "the certificate missed the stall" and stops the arm. So this arm has no result, good or bad.

### Forbidden for arm B

- Any return, gain or loss attributed to arm B. None was computed.
- "The certified ceiling did not raise return." It was never scored.
- Presenting arm C's information rows as arm B's result: "TD3+BC with half the pull plus BCA's ceiling", "TD3+BC (actor-only) plus a BCA ceiling built for it" (+4.82 on medium-expert, +8.98 on expert), or "half the pull plus B's map" (one seed). They were trained and tested differently, and the certificate failed its sanity check.
- "BCA's certificate found the stall", or calling the ceiling a BCA success.

---

## Arm C: guarded release ("less pull, guarded")

### Outcome

- **B1 met** on random, medium-replay and medium (the passing qualities). **B5 met** on all 15 quality-bank pairs.
- **The guard is void (section 6.4), so B2, B3 and B4 are not scored.** Both void conditions hold:
  - B's K1 fired at the seed-0 screen on medium-replay, medium and medium-expert.
  - C's own map had a thruster-1 ceiling of 0.85 or higher, with at least 30 tries in rung (0.80, 0.85], in 18 seed-quality pairs: random seeds 0, 1, 3 and 4; medium-replay seeds 0, 1, 3 and 4; medium seeds 0-4; medium-expert seeds 0-4.
- **KB0 held** at seed 0 and on the 5-seed means. Medium-replay passed (3.8% of thruster-1 commands past 80%, against 1.3% for TD3+BC), and so did medium (9.7% against 2.8%). Random failed (1.8%), but KB0 needs only 2 of the 3 qualities. So 6.5's KB0 line ("The guard had nothing to guard…") does **not** apply.
- GO (B1 + B2 + B3 + B5) is not reached. BCA gets no credit: B4 is the mark that decides that (reading 13) and it is not scored, and a void guard is never credited (reading 15).

| Mark | Value (strong test unless a bank is named) | Bar | Met |
|---|---|---|---|
| B1, random | +15.49, 5 of 5 | at least +3.0, 4 of 5 | yes |
| B1, medium-replay | +8.90, 5 of 5 | at least +3.0, 4 of 5 | yes |
| B1, medium | +8.48, 5 of 5 | at least +3.0, 4 of 5 | yes |
| B5, all 15 quality-bank pairs | lowest +2.32 (medium-expert, none bank) | at least −1.0 | yes |
| B2, B3, B4 | not scored (guard void) | | not scored |

### Frozen sentence, filled: the most conservative wording the file supports

Section 6.5 has no sentence for a void guard. The applicable line is the one for "B1 met, B2 or B3 not":

> "Less pull helps; BCA adds nothing."

It is reported with these lines:

> B1 is reported alone: +15.49 points on random, +8.90 on medium-replay and +8.48 on medium, 5 of 5 seeds each; no harm on any quality or bank (lowest +2.32). B2-B4 are not scored: the guard is void, because B's K1 fired on medium-replay, medium and medium-expert, and C's own map had a thruster-1 ceiling of 0.85 or higher with at least 30 tries in rung (0.80, 0.85] in 18 seed-quality pairs. The release is TD3+BC's alpha and is never credited to BCA.

Why this line, and not another:
- A mark that cannot be scored cannot be met. Section 3.2 says a mark that needs a missing number is not met, and the void rule works like a refused check. So "B2 or B3 not" holds, and this line applies.
- The two higher lines need B2 and B3 met, so they cannot be used. One is "A held-out-outcome guard makes a bolder TD3+BC safe; …". The other is "Exploratory: with BCA's certificate as its guard, …".
- "BCA adds nothing" is the frozen file's no-credit statement. It is backed by readings 13 and 15, and by sections 6.1 and 6.5 ("The release is TD3+BC's alpha").
- "B1 is reported alone" borrows the KB0 line's own rule for a guard that cannot be judged.
- Information only, decides nothing: halving the pull alone, with no ceiling, scored +15.40, +8.70 and +7.90 points over TD3+BC on the same three qualities (5 of 5 each). That is nearly all of B1.

### Plain language for the page

> **Less pull, guarded.** This version halves TD3+BC's pull toward the recorded commands. That is the same as turning up TD3+BC's own alpha setting, so it is not BCA. When acting, each thruster was also capped by a ceiling from BCA's certificate. Only the policy was trained, against TD3+BC's frozen critic.
>
> On random, medium-replay and medium data, it scored 15.5, 8.9 and 8.5 points above TD3+BC trained the same way, on the strong-current test. All 5 seeds agreed. It lowered no score on any data quality or test.
>
> BCA gets no credit for this. Its ceiling failed the same safety check that stopped the certified ceiling: it let thruster 1 run to 85% or more, past where the stall begins. So the guard is void, and the checks of what the guard added were not scored. Halving the pull alone, with no ceiling, scored about the same: 15.4, 8.7 and 7.9 points (information only; decides nothing). Less pull helps; BCA adds nothing.

### Forbidden for arm C

- RESULTS.md's own sentence, "Halving the pull, guarded by BCA's ceiling, met B1 …", as the outcome line. It is not a frozen sentence, and "guarded by BCA's ceiling" presents a void guard as working.
- "Guarded by BCA", "BCA made a bolder TD3+BC safe", or "BCA's guard was needed".
- "A held-out-outcome guard makes a bolder TD3+BC safe", or the all-met sentence.
- Quoting B2, B3 or B4 values as marks or as BCA's contribution (for example "BCA added +0.58").
- "KB0 failed" or "the guard had nothing to guard". KB0 held.
- Crediting the release, which is TD3+BC's alpha, to BCA.
- Crediting the ceiling gains on medium-expert and expert (+5.40 and +9.38, which are B5 values) to BCA. See the information rows below.

---

## F0: result F's no-band control ("result F without the band")

### Outcome

Credit on **neither** quality. No harm row. Both qualities were decided (KF1-KF4 held on every seed).

| Mark | Value (strong test) | Bar for credit | Credit |
|---|---|---|---|
| A2 − F0, medium-expert | +0.14, 5 of 5 | at least +1.0, 4 of 5 | no |
| A2 − F0, expert | +0.08, 3 of 5 | at least +1.0, 4 of 5 | no |

The harm row ("On <quality>, the band's finite part lowered return…") needs −1.0 or lower with 4 of 5 seeds negative. Neither quality comes close.

### Frozen sentence (7.4, "Credit on neither"), filled

> "Result F's gain comes from refusing credit past the data, not from BCA's band."

The sentence has no number slots. Beside it: the band's finite part added +0.14 points on medium-expert (5 of 5 positive) and +0.08 on expert (3 of 5), both below the +1.0 bar.

### Plain language for the page

> **Result F without the band.** Result F's policy gets no value credit for commands the recording could never have made, or inside the removed 80% hold. Everywhere else it adds BCA's band. We reran it with the band removed, keeping only that no-credit rule. The band added 0.14 points on medium-expert data and 0.08 on expert data, on the strong-current test. The bar for credit was 1 point, with 4 of 5 seeds agreeing. So result F's gain comes from refusing credit past the data, not from BCA's band.

Caveats to keep beside it:
- The no-credit rule uses the recording's exact probabilities, which are known only in this toy world.
- On medium-expert, the rule covers only part of the commands past 80%. On seed 0 it covered 14-44% per chunk. On expert it amounts to "no credit past 80%".
- A literal "no thruster past 80%" control and the box control were not run.
- A2 and F0 were compared in the same sandbox only, not against the owner's desktop run.

### Forbidden for F0

- "BCA's band earned part of result F's gain", or anything implying result F's gain is BCA's band.
- "The band lowered return". The harm row did not fire, and A2 − F0 is slightly positive.
- "Result F shows BCA works".
- Calling F0 "no credit past 80%" on medium-expert.
- Any comparison of F0 with the desktop result F.

---

## Information-only rows the page may quote

Each row must carry its label. "Pre-listed" means that section 4.3, 5.3, 6.3, 7.4 or 8 lists the row as information before the run. "Post hoc" means it was computed after the results were known. All values are strong test, points, 5-seed means, with seeds positive, unless noted.

### Arm A (pre-listed, section 4.3; label "information only; decides nothing")

| Row | random | medium-replay | medium | medium-expert | expert | Caveats |
|---|---|---|---|---|---|---|
| Constant 1.33× pull − TD3+BC (`even_x1.33 − none`) | −7.59 (0 of 5) | −4.64 (0) | −4.44 (0) | +0.50 (5) | +0.24 (5) | This is the actor-only twin of the page's "optional check": the same constant extra pull, with no band. It is not that check. The page's check trains the critic too, has not run, and was not part of the go. Setups differ: actor-only, on the weighting test's strong bank with the 80% hold removed, while the page's capped-pull table is co-trained and uses the main test's four situations. So do not set the two tables side by side as "matching" (post hoc if you do). Allowed reading: "a constant extra pull alone, with no band, also scored below TD3+BC on the weak data". |
| A1 − band with rows shuffled (`A1 − shuffled_band`) | +0.00 (3) | −0.86 (1) | −0.44 (1) | +0.06 (2) | −0.07 (3) | It supports "the band's shape adds nothing". |
| Capped pull retargeted − TD3+BC (`A1c − none`) | +19.91 (5) | +11.74 (5) | +10.30 (5) | +1.07 (5) | +0.41 (4) | It uses A1's skip rule plus BCA's dose. No plain-dose control was run, and its gain tracks A1's own skip rule, which plain margins match. Never "the capped pull now works" or "BCA's dose raised return". |
| Exact-weight band − A1 (`A1_IW − A1`) | −2.04 (0) | −1.55 (0) | −2.97 (0) | −0.80 (0) | −0.29 (2) | Exact weights are an oracle input. Exact weighting made the rule worse. |
| Co-trained TD3+BC − actor-only TD3+BC (`host − none`) | +0.07 (3) | −1.15 (1) | +0.29 (3) | −1.26 (0) | −0.74 (0) | It shows the actor-only comparator is close to, but not the same as, end-to-end TD3+BC. |
| The main test's four situations, A1 − TD3+BC (section 8) | for example random: +14.00 / +16.21 / +15.61 / +19.97 (recorded / big kicks / big waves / strong current) | | | | | Pre-listed, after the marks. See RESULTS.md's last table. |

### Arm C (pre-listed, section 6.3, unless marked post hoc)

| Row | random | medium-replay | medium | medium-expert | expert | Caveats |
|---|---|---|---|---|---|---|
| Half the pull alone − TD3+BC (`rho0.5 − rho1`) | +15.40 (5) | +8.70 (5) | +7.90 (5) | **−0.81 (0)** | **−0.45 (0)** | It is the release on its own, which is TD3+BC's alpha, not BCA. Quote the strong-data losses too: halving the pull alone slightly lowered the score on medium-expert and expert, with every seed behind. |
| Half the pull + cap at 80% − TD3+BC (`rho0.5+knee_clip − rho1`) | +15.63 (5) | +9.08 (5) | +9.31 (5) | +7.23 (5) | +9.39 (5) | The cap at 80% knows where the stall begins, which a real user would not. It is a yardstick, not a method. |
| Half the pull + BCA ceiling − TD3+BC (B5's strong row) | +15.49 | +8.90 | +8.48 | +5.40 (5) | +9.38 (5) | On medium-expert the BCA ceiling is void in all 5 seeds (it certified 0.85 or higher). On expert there are no tries past 0.80, so the ceiling stops there by the support rule ("no tries, no step"), not by reading outcomes. Section 5.3 says a gain from that is never credited to BCA. |
| **Post hoc:** BCA ceiling − cap at 80%, on the half-pull policy | −0.14 (0) | −0.18 (0) | −0.83 (0) | **−1.84 (0)** | **−0.01 (0)** | "Post hoc; information only; decides nothing." The BCA ceiling did worse than the 80% cap on medium-expert in every seed, and tied it on expert. The cap knows where the stall begins. |
| TD3+BC + BCA ceiling built for it − TD3+BC (`rho1+bca_built_for_rho1 − rho1`) | −0.10 (1) | −0.23 (1) | +0.11 (4) | +4.82 (5) | +8.98 (5) | No cap at 80%, no plain test and no other rival was run for this policy. So nothing here says BCA beats plain held-out tests, or beats anything at all. Same void and support-rule caveats as above. It is not arm B's result. |
| **Post hoc:** BCA ceiling − best plain test, on the half-pull policy, medium-expert and expert | | | | −0.02 (3 of 5; best: weighted t-test) | −0.005 (0 of 5; best: weighted median) | "Post hoc; information only; decides nothing." These five plain-test ceilings were scored in the raw records but are in no mark and not in RESULTS.md. BCA's certificate did no better than the best plain test, so nothing on medium-expert or expert says BCA beats plain held-out tests. Arm B, the other place such a test was planned, ran no plain test at all. |
| Against the co-trained TD3+BC (`… − host` rows in RESULTS.md) | | | | | | Same caveats as the rows above. |

### F0 (pre-listed, section 7.4)

| Row | Value | Caveats |
|---|---|---|
| A2 − F0, mild and none banks | medium-expert +0.12 / +0.09 (5 of 5 each); expert +0.06 / +0.04 (4 of 5 each) | Information only. |
| Seed 0: A2 − "no extra caution" and F0 − "no extra caution", strong test | medium-expert +1.46 / +1.41; expert +6.29 / +6.01 | One seed; from SCREEN_seed0.md (not in RESULTS.md). It shows that the no-credit rule alone keeps result F's seed-0 gain over "no extra caution". |
| Main test's four situations, A2 − F0 | +0.02 to +0.22 | Pre-listed (section 8), after the marks. |

### Arm B (kill-check quantities only)

- Room: capping the co-trained TD3+BC at 80% raised its score by 6.99 points on medium-expert, seed 0 (B's K0). Label: one seed; kill-check quantity; the cap at 80% knows where the stall begins.
- No other number of arm B exists.

---

## Forbidden claims, whole page

- "BCA improves TD3+BC", in any form: "BCA raised return", "BCA's band (or certificate, or ceiling) helped", "BCA made TD3+BC bolder and safe".
- "Confirmed", "pre-registered", "significant", "proven", or any line that drops "exploratory" and "seeds 0-4 already seen".
- Any claim about fresh seeds, other tasks or real systems.
- Calling arm B, or arm C's guard, a BCA success. Both failed the certificate's sanity check (K1, or the void rule).
- Implying result F's gain is BCA's band.
- Crediting the release (half the pull) to BCA. It is TD3+BC's alpha.
- "Arm A passed" or "a GO" for any arm. No arm reached GO.
- Saying a plain held-out test lost to BCA anywhere. No scored mark shows it, and the post hoc rows show the opposite or a tie.
- Saying "TD3+BC" for arms A and C without noting that only the policy was trained, against a frozen critic.

## Page text (version 29) these results now make out of date

- "Two things are open: the control without the band, the same no-value-credit rule alone with no band, was not run …" and "So the toys predict that the untested control would recover most of result F's gain." The control has now run (exploratory, seeds 0-4), and it did recover the gain.
- "Potential fixes (untested ideas). None of these has run." All three have now run (exploratory). The certified ceiling was stopped by its own safety rule. The "less pull, guarded" gain is TD3+BC's alpha, with a void guard. The "copies only good rows" gain does not need BCA's band.
- "An optional check, also waiting for the project owner's go …" is still true. Only its actor-only twin ran, as information only.

## Where RESULTS.md's wording departs from the frozen file

- **Blocker (a frozen wording):** arm C's outcome line in RESULTS.md and in `results.json` (`C.wording`): "Halving the pull, guarded by BCA's ceiling, met B1 on random, medium-replay, medium; B2-B4 are not scored (KB0 failed or the guard is void)."
  - It is not one of 6.5's sentences. The frozen file supports "Less pull helps; BCA adds nothing."
  - "Guarded by BCA's ceiling" presents a void guard as guarding, against reading 15.
  - It omits "The release is TD3+BC's alpha".
  - Its stated reason is half wrong: KB0 held, both at seed 0 and on the 5-seed means. B2-B4 are unscored only because the guard is void.
  - The numbers and marks are right. Only the sentence must be replaced.
- **Minor:** RESULTS.md prints B2-B4 with "Met: no", though they are not scored (`results.json` has null). The outcome is unchanged.
- **Minor:** arm B's stop line omits "seed 0" (4.4's pattern is "<quality, seed>") and the ceiling and tries values. The outcome is unchanged.
- **Minor:** arm A's line names the rivals by code name (margin0, plain_1s, plain_2s, matched_share). The page needs the plain names given above.

All other outcomes agree with the frozen file: arm A's sentence, arm B's stop, F0's "credit on neither", and every number.
