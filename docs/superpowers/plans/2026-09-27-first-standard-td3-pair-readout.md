# First standard TD3+BC Hopper paired readout

Audit the newly closed no-IW BCA run from saved files on CPU, retaining the
already accepted host audit without rerunning or repulling its training files.
No subagents, model queries, simulator steps, learner updates or new GPU workers.

1. Verify actual exit, frozen source/settings/data, three checkpoint identities
   and counters, every scan/evaluation bank, accepted/abstained scale fits and
   all 198 posterior refreshes with both radius components.
2. Check paired data, evaluation banks and saved RNG identities against the
   closed host audit. Preserve every final episode and all failed audit attempts.
3. Publish learning/Q/loss/scale/radius and episode comparison plots. Report one
   paired seed as descriptive evidence, with five-seed uncertainty still pending.
4. Preserve zero importance tilt and Bayesian bootstrap masses. Keep OOD gates,
   unavailable targets and later development-only IW selection separate.
5. Verify frozen integrity and exact owned artifacts; publish excluding weights
   and retain actual push exit and matching remote-head receipts.
