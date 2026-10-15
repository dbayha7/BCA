# Route search for BCA (exploratory, seeds 0-4)

The user asked (14 Oct 2026) to keep testing new routes for BCA to contribute, shelving dead ends with a plain
explanation. Every route is tested on gate seeds 0-4 only, after seeing their results, with its kill rule written
(and hash-checked) before the test; nothing here supports a claim or moves a pre-registered pass mark.

- `round1/ROUND1.md`: the round-1 log, one entry per route (the idea / what we tested and the kill rule / what
  happened / why), plus the routes not tested and why.
- `round1/scratch/<route>/`: each route's `KILL_RULE.md` (+ `.sha256`), test script and the verifier's rechecks, as
  they were in the cloud session's scratch folder
  (`/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/routes/`).
- `round1/run_outputs/<route>/`: the JSON results written under `runs/shift_explore/routes/<route>/` (gitignored),
  so the sources cited in ROUND1.md as `results.json` or `summary.json` are here.

Data dumps (`.npz`, `.npy`) and logs are left out. Paths inside the scripts point at the cloud scratch folder and
`runs/`; adjust them to rerun.

Earlier shelved routes (critic disagreement as BCA's width, the run-time classifier guard, the critic-scaled range
penalty, a larger k, freezing the classifier) are explained in the 14 Oct conversation
(`docs/session-history/`) and in `../analysis/`.
