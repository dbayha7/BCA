# Stage 1: room test of the lower-edge channel (EXPLORATORY, no BCA)

Committed before any return of a trained arm was computed. It records the design of the test in advance.

- `spec.json` (SHA-256 `56bbda24…`) gives every arm, formula, ladder, matching rule, mark and seed. Its
  `changes_before_any_run` list records each change from `../PLAN.md` section 4.1 and why it was made.
- `code_manifest.json` pins the eight scripts, and `code/` holds copies of them. Every phase refuses to run if a
  file differs from the manifest.
- `checks_seed0*.json` and `checks_e_seed*.json` are the seed-0 checks (a)–(e), run before any training, together
  with the per-seed check (e) on the penalty's shape.
- `timing.json` gives the cost per 1,000 steps of each kind of arm.

**The main change from the plan (A1).** The plan's oracle was |delta*|, the size of the critic's exact one-step
error at the actor's command. That oracle *rewards* the first few points past 80%. The critic was trained toward next
commands with smoothing noise added, so delta* is already about +0.15 at the recorded commands, and its size first
shrinks as the critic's false preference grows. The main oracle is therefore re-referenced to the row's recorded
command: [delta*(s, a_i) − delta*(s, a)]_+. This is how much more the critic overrates the actor's command than the
recorded one. The plan's literal form, and a two-sided form, are kept as reported arms. Both fail check (e), which
tests whether a penalty rises past 80%.

**What the result will mean.** BCA's band is symmetric, and its width at a single (state, command) pair is not
enough by itself. To carry this harm, the width must be read *relative to the recorded command at the same state*.
That is what the plan's paired detection marks already require (width at the command against width at its 80% cap).

Seeds 0 and 1 run first; the futility look then decides whether seeds 2–4 run. Outputs go to
`runs/shift_explore/room/` (not committed).
