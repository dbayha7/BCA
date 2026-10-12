"""D4RL-RT truth instrument: the true Q^pi of a frozen policy at logged and visited Hopper/Walker2d states.

Modules (see experiments/truth/README.md for stages, commands and costs):
  state_restore  converted row -> raw row -> qpos/qvel (infos, or rebuilt from the observation), the exact
                 env step on bare MjSim instances, and the one-step reproduction gate
  labeller       batched rollout truth Q^pi(s, a): many MjSims in lockstep, one jitted actor call per step;
                 deterministic, jitter and noisy-target modes; central-difference action gradients;
                 resumable shards with a strict JSON manifest
  synthetic      D4RL-format HDF5 files made in the real simulator, for tests only (never findings)
  pools, extraction, acceptance   option C's pools, room-to-win machinery and R1-R7 acceptance gate
  gap, gap_train, gap_acceptance  the action-gap track: the frozen densest-cluster gap and the gapped copy, the
                 gapped-host training wrapper around train.py, and the G0-G4 protocol with its verdict table
"""
