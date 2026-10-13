"""Importance weighting for BCA on ThrusterChain: a controlled, fully known distribution shift.

Goal (the user's): check that BCA's parts work on data where every value is known, then show that weighting BCA's
calibration by the known test distribution restores its coverage (and raises return only if weighting moves BCA's
caution enough to matter). The specification and its pass marks are bca-shift-work/iw/SPEC.md and spec.json.

The world is v1's ThrusterChain (experiments/shift), imported read-only and never edited. Only the law of the current
strength c changes between training and test; given c everything else is identical, so the exact likelihood ratio of
any recorded step is w(c) = p_test(c) / p_train(c). The recording controller's most common hard-water command, thruster
1 held at exactly the 80% limit, is removed from training and calibration by v1's densest-15% rule (the action gap).

Part A (this package so far; no learning and no BCA on ThrusterChain data):
  world       current laws and exact densities, the recording controller, the removal, exact weights, quadrature over c
  data        the training-law log, calibration bank, test logs per tilt, classifier samples and start banks, with
              exact truth for every row; HDF5 files with SHA-256 manifests; deterministic seeds
  coverage    coverage of any band on any rows, with and without weights, outside and inside the gap (thresholds by
              calibration/wbcp.calibrate); pure array functions
  classifier  the estimated-weight classifier (logistic regression on the observation, Newton's method)
Tests: JAX_PLATFORMS=cpu python -m unittest experiments.shift_iw.test_world experiments.shift_iw.test_data

Import order: experiments.shift first (it puts gymnasium on the path and checks the versions).
"""

import experiments.shift  # noqa: F401
