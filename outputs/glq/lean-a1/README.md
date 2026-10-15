# Gapped LQ gate under Amendment A1: the world is rejected again

Run 2026-10-12 on a 4-CPU cloud container. Pre-registration `experiments/signal/glq_preregistration.json`, SHA-256
`8528a61385157a76…` (Amendment A1). Fresh seeds 5–9 (the lean path). No BCA arm and no BCA band fit were run.

## Verdict

**The Gapped LQ world fails its re-registered gate in both allowed versions, so it is rejected and BCA is not run in
it.** The original gate rejected it on seeds 0–4 (`outputs/glq/lean-v1/README.md`). Amendment A1 changed four rules
after seeing those seeds (post hoc, labelled as such) and was tested only on seeds that had never been trained on. The
world fails those rules too.

- Steps 1–4 pass in both versions: build checks, Stage 0, state richness and thin data.
- Step 5 (R2, "the critic's error sits where data are thin") fails in both versions.
- Step 6 (R3, "the return lost sits where data are thin") fails at every alpha of the ladder in both versions.
- The second pre-registered retune (`wrong_gain`) still fails Stage 0 (the transferred policy is unstable,
  √γ·ρ = 1.0023), so no version is left.

Files: `verdict/base.md`, `verdict/thin_share.md` (the gate's own tables), `r2/`, `r3/`, each with its SHA-256
sidecar.

## Step 6 (R3): where the return is lost

A seed passes when all four R3 marks hold. The mark that fails is the share of the host's shortfall against the EXACT
arm that sits in the two thin contexts. Under A1 it must be at least 1.5 times their share of episodes: 0.45 in the
base world (thin contexts 30% of episodes) and 0.75 in thin_share (50%).

| Version | Alpha | Seeds passing (need 4) | Thin share of the shortfall, seeds 5–9 |
|---|---|---|---|
| base | 2.5 | 0 | 0.55, 0.42, 0.49, 0.50, 0.43 (the cost mark also fails: 0.012–0.042 against 0.05) |
| base | 5 | 3 | 0.55, 0.52, 0.39, 0.49, 0.43 |
| base | 10 | 2 | 0.51, 0.33, 0.49, 0.44, 0.43 |
| thin_share | 2.5 | 0 | 0.56, 0.65, 0.67, 0.59, 0.55 |
| thin_share | 5 | 0 | 0.59, 0.61, 0.58, 0.51, 0.57 |
| thin_share | 10 | 0 | 0.71, 0.59, 0.59, 0.53, 0.53 |

The other three marks mostly pass: the critic's error costs return in the reversed context, the host trails the EXACT
arm by 0.36–0.66 normalised score, and the host still helps where data are plentiful (normalised score 0.46–0.58 in
the well-recorded contexts at alpha 10).

## Step 5 (R2): where the critic's error sits

Gated on the fixed-policy (FQE) critic; a seed passes when both concentration marks hold.

| Version | Error in the reversed context vs the well-recorded ones (mark ≥ 3×) | Thin contexts' share of the squared error (mark ≥ 1.5 × row share) | Seeds passing (need 4) |
|---|---|---|---|
| base | 3.06–8.06×, 5 of 5 seeds | 0.38–0.70 against 0.45, 3 of 5 | 3 |
| thin_share | 2.75–6.08×, 4 of 5 seeds | 0.52–0.78 against 0.75, 2 of 5 | 2 |

The density ladder behaves as designed (the error in the thin contexts falls at each of the 4 data-density steps and
evens out at the top step), so the critic does react to how much data there is.

Reported, not gated:

- The co-trained (HOST) critic's error is not concentrated at all: 1.02–1.48× in the reversed context, and the thin
  contexts hold 0.16–0.28 of the squared error.
- Accuracy where data are plentiful is poor for both critics. The critic's error moves the action it recommends by
  2.7–4.8 units (FQE) and 1.8–2.4 units (HOST), against recording noise of 0.3.

## What it means

The same as the first run, now on fresh seeds and under the user's amended rules. In this world the learned critic is
wrong almost everywhere, not mainly where data are thin, so the return lost is spread over every context. A test of
whether caution placed where the critic is wrong beats even caution needs a world where the harm is concentrated
where data are missing. This world does not provide that.

The replacement is the ThrusterChain environment (`experiments/shift/`, EXPERIMENTS.md entry 16): a real Gymnasium
environment with a deliberately created, exactly measured distribution shift, and harm checks that must pass before
BCA is judged.

## What ran

- Seeds 5–9. Per version: 15 HOST/FQE/EXACT critics at alpha 2.5, 10 HOST/EXACT critics at each of alpha 5 and 10,
  and 12 FQE density-ladder critics (rungs 0, 0.03, 0.1, 0.3; seeds 5–7). 94 critic runs of 100k updates in all,
  about 5–7 minutes each on 2 CPUs.
- Not run, because steps 5 and 6 failed: R6, R5 world gates, the Level-1 screen, Tier A, the null world and the
  global-linear class.
