"""Amendment A2 (CONFIRMATION_RULE.md): the GPU device kinds a research job of this confirmation may run on, by machine.

knee_edge_ood_v3 pins one recorded device kind (frozen_checks.RECORDED_DEVICE_KIND, Zeus's laptop GPU) and checks it
in every research training bind. The confirmation's worker (change 4) sets the recorded kind to the one allowed kind
the machine has, right after the worker environment is set, so the same sealed package runs on Zeus or Titan. The
same list as the r5 screen's amendment A2 (`../native_screen_r5/device_kinds.py`).
"""
ALLOWED_DEVICE_KINDS = {
    "NVIDIA GeForce RTX 5070 Ti Laptop GPU": "Zeus: the knee-edge recorded environment (rerun1 receipt)",
    "NVIDIA GeForce RTX 5070 Ti": "Titan: the payload's own machine (unifloral_checkpoint_v1 checks, 2026-10-23)",
}
