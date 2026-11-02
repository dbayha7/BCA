"""Amendment A2 (SCREEN_RULE_r5.md): the GPU device kinds a research job of this screen may run on, by machine.

knee_edge_ood_v3 pins one recorded device kind (frozen_checks.RECORDED_DEVICE_KIND, Zeus's laptop GPU) and checks it
in the preflight and in every training bind. A2 lets the same sealed package run on Titan: the worker (binding 8)
sets the recorded kind to the one allowed kind the machine has, every report carries it, and the supervisor reuses a
completed unit only if it was produced on the same kind as the current run.
"""
ALLOWED_DEVICE_KINDS = {
    "NVIDIA GeForce RTX 5070 Ti Laptop GPU": "Zeus: the knee-edge recorded environment (rerun1 receipt)",
    "NVIDIA GeForce RTX 5070 Ti": "Titan: the payload's own machine (unifloral_checkpoint_v1 checks, 2026-10-23)",
}
