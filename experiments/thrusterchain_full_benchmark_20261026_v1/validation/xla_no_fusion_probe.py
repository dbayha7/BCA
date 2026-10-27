"""Bounded retained parity cases, with compiler settings supplied by caller.

This file is validation-only and is never imported by the study runner.
"""
import os
import unittest

import jax
import compiled.test_engine as tests

print("XLA_FLAGS:", os.environ.get("XLA_FLAGS"), flush=True)
print("JAX:", jax.__version__, "devices:", jax.devices(), flush=True)
success = True
for method, test_name in (("td3bc", "test_real_runner_sampling_and_compiled_resume"),
                          ("cql", "test_cache_cross_dataset_and_seed")):
    tests.METHODS = (method,)
    print("\nCASE", method, test_name, flush=True)
    suite = unittest.TestSuite([tests.CompiledEngineTests(test_name)])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    success = success and result.wasSuccessful()
raise SystemExit(0 if success else 1)
