"""ThrusterChain with Unifloral's four learners (TD3+BC, CQL, IQL, ReBRAC), alone, with standard BCA and with
importance-weighted (IW) BCA: the 12-setup experiment of PLAN_UNIFLORAL.md (15 October 2026).

This package is an experiment-local attachment (plan section 3.1). It changes none of the repository's declared
methods: the four host+BCA algorithms of ALGORITHMS.md stay the CORL-derived design in algorithms/, and the vendored
Unifloral files in baselines/unifloral/ (upstream commit f2dc127, pinned in configs/unifloral.json) are imported
unchanged, never edited (INTEGRATION.md:1402 says they "have no BCA entry points"; this package adds them here only).

The contracts every module follows are in INTERFACES.md beside this file. This file imports nothing, so that
`import experiments.shift_hosts.<module>` works in both Python environments:
  /opt/bca-venv        training (JAX 0.6.2 with the card, numpy 1.26.4, distrax, d4rl, gym): the only environment
                       where the Unifloral files import
  /opt/bca-shift-venv  data preparation and scoring (numpy 2.2.6): experiments/shift and experiments/shift_iw import
                       there, the Unifloral files do not
Arrays pass between the two only through handoff.py files, each with a SHA-256 sidecar.

Nothing here trains on ThrusterChain data without the user's terminal approval (gate.require_run_approval); BCA's code
runs on made-up arrays only when their content is registered as made-up (guard.require_made_up).
"""
