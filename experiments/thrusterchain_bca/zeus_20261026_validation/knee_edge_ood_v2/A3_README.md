# A3: step 0.4, collection and exact probes, knee_edge_ood_v2 (user-run)

**Requires** v1's single step 0.3 report, carried by `PROTOCOL_V2.md` change 1: PASS, or INCONCLUSIVE for G0.5's
share rule only, with the known-answer reproduction and recount passing (it is; its sha256 is pinned in
`step04.py`; its reach forecasts are recorded, not gated). Your A3 approval must include "I acknowledge that knee_edge_ood_v1 step 0.3 ended INCONCLUSIVE on G0.5 and the written prediction: Stage 1 V0(b) or V0(d) is likely to end INCONCLUSIVE". Cap: 3,000,000 exact simulator steps or 1 hour.

**Unit "collection"** (simulator, no models):
- collects `knee_edge_v1` (stall slope 3.0) and its no-stall twin, 1,280 episodes each, from the sealed dataset
  entropy; each dataset is validated by its schema (bit-for-bit lineage and replay) and saved once;
- reconciliation 4.7 in both worlds (plumbing: INCOMPLETE; expected values incl. L1.4: INCONCLUSIVE), the twin check
  and the data-level thrust check (stall3 thrust past the knee equals T3, not slope 1.5);
- the Stage 1 data plan: stream receipts for CUTOFF and RANDOM in both worlds;
- the 96 step 0.4 anchors, rebuilt from their sealed draw rules by recorder-prefix replay, and L1.5 as amended by
  `PROTOCOL_AMENDMENT_1.md`: each of the 128 Stage 1 E2 bank starts and 48 gated anchors is compared with the 0.99
  quantile of its own group (deep-load fitting rows at the same time step), bars 95% and 90%; the unamended general
  threshold is reported alongside.

**Unit "probes"** (GPU, frozen actors):
- exact Q^pi at all 96 anchors under each of the three frozen full-data native 20k actors, clipped, in `stall3`
  (S9, s05, s10, s15, s20, p10 and the T3 images), `nostall` (S9 and the past-knee slots) and `stall15_identity`
  (-KNEE, the past-knee slots and the T15 images);
- the 4.5 cross-check (the environment evaluator on the same layout, compared at -KNEE, s10 and s20), the literal
  identities (1e-6) and stall = no-stall on the supported slots (1e-9);
- L2 per continuation seed on the 48 gated E2 anchors (item 5 first, then items 1-4 and 6), and the reported
  Stage 1 reachability forecast (medians against 0.60 and 0.65).

| Outcome | Label |
|---|---|
| Identity, equality, cross-check, twin, count, split or a1..a3 failure | INCOMPLETE |
| L2 item 5; L1.4; L1.5; a 4.7(ii) expected value | INCONCLUSIVE |
| L2 item 1, 2, 3, 4 or 6 for any continuation seed | FAIL ("not shown model-relevant"): FAIL-consistent if the same item fails for all three seeds, otherwise FAIL-mixed |
| Everything holds | PASS; the A4 request then carries the standing prediction "Stage 1 V0(d) may end INCONCLUSIVE" |

**Amendment 1 (your decision, 2026-10-30):** L1.5 as written failed on in-distribution states in a smoke test
(training deep-load starts passed only 80.5%); it is calibrated within group, and your A3 approval names it.

**Settled conventions (the protocol leaves them open):** the descriptive general threshold uses the 640 fitting rows
whose sha256("knee-edge-l15-v1:" + row id) is smallest, one per episode first (no new seed); L1.5's training rows are
all of a world's fitting rows; the "E2 start states" are the 128 starts of the Stage 1 E2 bank; the stall15_identity layout adds
-KNEE so the cross-check has its three slots in every world; nostall uses the same 19-slot array as stall3 (its
T3-image values are reported only), so the 1e-9 stall = no-stall check on S9 compares identical actor batches.
- The gated values are the raw dynamics loop's; the environment evaluator cross-checks them at -KNEE, s10 and s20 in
  every world, as 4.5 allows.
- "Valid anchor" is "all values finite". Both evaluators raise on any bad action or state, so a failed anchor
  computation ends the probes unit INCOMPLETE by design; an anchor is never silently excluded.
- L1.5 normalizes the 14 inputs with the mean and standard deviation of all of a world's fitting rows, not with the
  host's CUTOFF n RANDOM statistics (Part 3.8).
- The transition cap charges collection, replay and continuation transitions (about 2.10M). Prefix replays are not
  charged (about 0.51M more, about 2.61M in all); the 1-hour cap binds first.

**Known limitation of amendment 1 (stated with the request):** the start-state bar (95% of the 128 E2 bank starts
within the t = 0 deep group's 0.99 quantile) has little power: the t = 0 group is about 118 rows, its quantile is
inflated by the spread of the starting kick, and in smoke tests start sets drawn at currents wholly outside the
data's support still passed at about 97%. Within stall3 the starts pass by construction. The gated anchors at t = 25
and 50 carry the check's real content.

**Integrity at run time:** before anything runs, the A1 and A2 manifests must equal their committed hashes, every file
they list must be unchanged, and rerun1's reused sources must match their recorded hashes. v1's step 0.3 report must
equal its pinned sha256 and come from the committed A2 package. The prior study's exports are re-hashed before `probes.json` is sealed. A sealed
collection unit is reused only under unchanged producing code (`step04.py`, `step04_math.py`); otherwise it is
re-collected and must reproduce the sealed record exactly (5.0).

**Attempts (5.0).** Each attempt writes `step04_run/attempt_<k>/`. Attempt k needs your approval naming k. A re-attempt
is allowed only after an INCOMPLETE and repeats only the unsealed unit; a preflight failure computes nothing and does
not count; a second failure of the same unit under unchanged code is INCONCLUSIVE; at most one code-only fix.

## How to run it (PowerShell, from `C:\Users\David\dev\BCA`, one command at a time)

0. Commit v1's step 0.3 record if it is not committed yet (v2 reads it from there):

```powershell
git add experiments/thrusterchain_bca/zeus_20261026_validation/knee_edge_ood_v1/step03_run
```

```powershell
git commit -m "Record knee_edge_ood_v1 step 0.3 (A2) attempt 1: INCONCLUSIVE"
```

1. **Approve, in your own words, at your terminal** (the text carries the required acknowledgement):

```powershell
.\.venv\Scripts\python.exe -c "import json,hashlib;p='experiments/thrusterchain_bca/zeus_20261026_validation/knee_edge_ood_v2/';h=hashlib.sha256(open(p+'A3_MANIFEST.json','rb').read()).hexdigest();open(p+'A3_APPROVAL.json','x').write(json.dumps({'approval':'I approve A3 (step 0.4, collection and exact probes) for knee_edge_ood_v2, including protocol amendment 1 (L1.5 calibrated within group) and PROTOCOL_V2.md, attempt 1. I acknowledge that knee_edge_ood_v1 step 0.3 ended INCONCLUSIVE on G0.5 and the written prediction: Stage 1 V0(b) or V0(d) is likely to end INCONCLUSIVE','package_manifest_sha256':h,'attempt':1},indent=2)+chr(10))"
```

2. **Commit the approval** under your identity:

```powershell
git add experiments/thrusterchain_bca/zeus_20261026_validation/knee_edge_ood_v2/A3_APPROVAL.json
```

```powershell
git commit -m "Approve A3 for knee_edge_ood_v2 (attempt 1)"
```

3. **Run it once in WSL** (about 10-30 minutes):

```powershell
wsl -d Ubuntu -- /home/david/bca-operator-audit-20261026/bin/python -B -u /mnt/c/Users/David/dev/BCA/experiments/thrusterchain_bca/zeus_20261026_validation/knee_edge_ood_v2/step04.py --execute
```

It prints the label at the end. The full record is `step04_run/attempt_1/report.json`, with `collection.json`,
`probes.json`, one `probes_seed*.json` per continuation seed and the two datasets (`knee_edge_v1_*.npz`).

Files: `step04.py` (driver), `step04_math.py` (L1.5, L2 and the label), `attempts.py` (5.0 attempt rules and upstream
checks, shared with later steps), `units.py` (the exact simulator-step counter, shared with the stages), `test_step04.py`,
`PROTOCOL_AMENDMENT_1.md`, `PROTOCOL_V2.md`, this README, and `A3_MANIFEST.json` (their hashes).

**Caps after the 2026-10-30 audit (`PROTOCOL_V2.md` change 6):** every simulator step is counted exactly, whichever
module takes it (collection, reconciliation replay, anchor rebuilds, continuations, the cross-check), and the step that
would cross the transition cap raises before it runs; the prospective formula charges are recorded alongside. The hour
is a hard cap: a watchdog writes the attempt INCOMPLETE and ends the process at 3,600 s whatever it is doing.
