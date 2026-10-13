# Launchers for the exploratory Stage 2 seeds 0-4

The three shell scripts that ran Stage 2 on seeds 0-4 in the cloud container, copied here so they survive the
container. They are not part of the pinned Stage 2 code (they only call the pinned `launch.sh` once per phase) and
are not hashed into the pre-registration.

| Script | What it ran |
|---|---|
| `explore_detect.sh` | the detection test on seeds 0-4, uniform version, then its exploratory marks |
| `explore_weighted.sh` | the declared weighted version on seeds 0-4, then its exploratory marks |
| `explore_policy.sh` | the policy test, deployment check and full host on seeds 0-4, then the pooled look; resumable (skips a phase whose outputs exist), one lock so only one copy runs, 2 lanes (processors 2 and 3) |

They expect the container layout: the repository at `/home/user/bca`, the Stage 2 code at
`/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/ood/stage2/` and the run folders under
`runs/shift_explore/` (gitignored).
