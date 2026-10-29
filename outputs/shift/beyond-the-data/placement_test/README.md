# Placement full test (design, draft 2, not frozen)

Status, 19 Oct 2026: **design only, for the owner's decisions.** Nothing was generated, trained, scored or run on
ThrusterChain data for it. The only computations were micro-benchmarks on made-up arrays (`bench/`,
`reviews/review_feas/`) and arithmetic on made-up numbers (`measure/`, `revise/`, `reviews/`).

The test covers three places to use BCA. Each is tested for return and for validity or coverage, with online
ThrusterChain episodes that simulate deployment:

- A, harm switch on a declared group or world;
- B, safety certificate per deviation, offline and in a staged deployment;
- C, value lower edge at the actor's own command.

Read `DESIGN.md`. Its section 0 is a one-page summary, and section 14 lists the decisions the owner makes before
anything is frozen. Once those are settled, the text becomes `PRESPEC.md` of a new package `experiments/placement_test/`,
with its own gate, and is frozen before the pilot (DESIGN.md section 9.4). No run starts without the owner's approval at
their own terminal.

Files:

- `DESIGN.md`: the test specification. Section 13 is the review round (70 findings, each applied, applied in part or
  superseded, with reasons).
- `infra_cost_notes.md`: the reusable code, with file:line, the gaps that need new code, and the compute measurements.
- `seed_registry.json`: the seed namespaces and their status.
- `measure/`: how each option's return and coverage are measured, plus the evidence-floor arithmetic.
- `bench/`: timing scripts and outputs (cloud container, 4 CPUs, no GPU).
- `revise/`: draft 1 and the revision's arithmetic.
- `reviews/`: the reviewers' arithmetic checks.
- `analysis/`: the two analyses the design builds on. `root_cause.txt` explains why BCA, used as a dose on TD3+BC's
  behavior-cloning weight, did not raise the host's return. `placement_options.txt` lays out the options for where
  BCA could act instead.

Sources named in DESIGN.md as `scratchpad/...` were working files of the cloud session that wrote this. Two are not
copied here: `scratchpad/placement/result.json` (its readable form is `analysis/placement_options.txt`) and
`scratchpad/status/live_c264.html` (a copy of the published ThrusterChain page, version 28).
