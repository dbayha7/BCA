# Restructuring BCA: the read-only analyses (exploratory, seeds 0-4)

Copied from the cloud session's scratch folder on 14 Oct 2026 so they survive the container. Scripts, JSON
results and notes only; the `.npz` data dumps, logs and caches were left out (the scripts regenerate them from the
saved run files). Everything here was designed and run after seeing seeds 0-4, read-only on the saved models, and
supports no claim.

- `impact/`: the diagnosis of where BCA and importance-weighted BCA stand after Stage 2 seeds 0-4 (why M4 and M5
  fail, what the weighting contributes). Summary in the 14 Oct conversation and `../../results-2026-10-14/README.md`.
- `restructure/`: the redesign study. `ANSWER.md` is the plan; `map.md` maps BCA as implemented; the four design
  folders (`action_aware/`, `consumer/`, `calibration/`, `testbed/`) hold each designer's scripts and evidence;
  `judge_builder/` the judges' checks; `designs_dump.txt` every design record.
- Shelved routes, with the reasons: critic disagreement as BCA's width (`restructure/action_aware/`), the run-time
  classifier guard (`restructure/consumer/`), the critic-scaled range penalty (`restructure/action_aware/`, BCA-X).
- The route that went forward is outcome-reading BCA (`restructure/calibration/`, `restructure/testbed/`); its
  decisive test is `../step1/` (GO on 14 Oct).

Paths inside the scripts point at the cloud scratch folder
(`/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/`); adjust them to rerun.
