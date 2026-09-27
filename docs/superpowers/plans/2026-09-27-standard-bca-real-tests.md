# Standard BCA real-tests implementation plan

> Execute in this session without subagents, as requested. User authorized real tests on September 27 and confirmed no importance weighting first. Do not ask again to authorize this stage.

**Goal:** complete all four host/BCA comparisons at 1M host updates, five declared seeds and seven configured datasets, starting with no importance tilt, followed by independently chosen IW comparisons.

**Architecture:** retain four host files and four BCA extensions. One experiment-level setting selects uniform importance factors versus the archived host-specific fitting heuristic. Preserve Bayesian bootstrap masses, both Bayesian/conformal radius components, host losses, data reservations, normalization, update counts and evaluation banks. No-IW does not mean removing Bayesian bootstrap randomness or IQL's native actor AWR weights.

**Tools:** existing JAX/Flax/MuJoCo runtime, Slurm and the shared local GPU lock. At most two GPU workers total; one local and one cluster initially. A frozen source copy trains while editable source work continues.

## Scope and evidence

All four hosts (IQL, CQL, TD3+BC, ReBRAC), seven existing datasets, five predeclared seeds. Preserve the 280 physical declarations / 315 actor trajectories of the minimal repository, with shared-Q/V IQL host/BCA actors as the primary pair; standalone IQL host is context, not another independent seed. No old run is relabeled. Exact bundled Unifloral baselines stay references; no BCA is newly grafted onto them.

The old 810-group campaign is halted and remains separate. Its eighteen results and original controller failure are immutable. Neither that queue nor old IQL Maze/CQL HalfCheetah behavioral recoveries are resumed by this plan.

## Tasks

- [ ] Add known-answer no-IW tests before changing production code: all four typed configurations; uniform importance factors times unchanged Bayesian masses; IQL unchanged native AWR/shared Q/V/gain1; original IW recipes remain representable; unknown mode refused.
- [ ] Add one commented `calibration_weighting: none` setting in experiment.yaml; version BCA run identity. Allow existing off paths for TD3/ReBRAC/CQL and implement the IQL off fitting path. Update pseudocode and all affected integration snippets. Preserve prior source pins/validation separately; new acceptance requires new checks.
- [ ] Run CPU numerical and typed-protocol checks. Verify host equations and original Unifloral bytes unchanged. Preserve every failed engineering attempt and actual exit.
- [ ] Inspect cluster reachability, queue, allocated resources and available runtimes. Validate source/data preparation and installed simulator identity on the actual execution host. No heavy training on login nodes.
- [ ] Freeze complete training manifests with exact rows, source/config/data hashes, seed banks, resource ceilings and distinct output roots. Pilot actual launcher success/failure/timeout/lock behavior with harmless workers. Never silently retry or select by score.
- [ ] Dispatch validated first-stage training in fixed order across hosts with the existing one-local/two-total GPU cap. Record real scheduler/process identities and actual child exits; run from immutable source snapshots. A successful submission is not completion.
- [ ] Bind accepted 1M checkpoint/event/data/counter identities into OOD manifests. Run bounded, disjoint engineering checks before scientific collection; retain 1e-6 action and 1e-7 reward gates and all missing states. Extend adapters for IQL/CQL and later environments before claiming coverage of the full grid.
- [ ] Collect the real standard-BCA OOD comparisons once; publish complete/missing/failed status, paired five-seed uncertainty and all figure types. Preserve unresolved ReBRAC fresh-target contract as unavailable instead of inventing recorded next actions.

## Importance-weighting stage

No IW winner is selected from these final test banks. Before launching an IW stage, define its intended reweighting target, small justified candidate set, development-only selection data and fixed selection criterion. Freeze the chosen recipe before held-out comparison. Distinguish affinity/AWR/policy-density heuristics from an actual policy-to-behavior density ratio, and fitting-weight permutation from consumed-dose permutation. Retain standard BCA as a control. No new gain sweep.

## Current preflight

Cluster SSH succeeds through the existing WSL control connection; scheduler reported no active dbayha jobs. Local 5070 Ti was visible with no reported compute application or holder of the existing lock. These observations must be rechecked at dispatch. The default sandbox helper failed before execution; read-only commands succeeded through approved escalation. No scientific jobs launched at plan creation.
