"""Run bounded mechanical tests and save all test outcomes and analytic evidence."""
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import sys
import time
import unittest

os.environ.setdefault("JAX_PLATFORMS","cpu")
os.environ.setdefault("OMP_NUM_THREADS","1")
import jax
jax.config.update("jax_enable_x64",True)
import numpy as np
from test_objective import scalar_loss

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"evidence"
if OUT.exists():
    raise SystemExit("Evidence directory already exists; preserve it and use a new version.")
OUT.mkdir()
started=time.time()
buffer=io.StringIO()
suite=unittest.defaultTestLoader.discover(str(ROOT),pattern="test_objective.py")
result=unittest.TextTestRunner(stream=buffer,verbosity=2).run(suite)
log=buffer.getvalue()
(OUT/"test_log.txt").write_text(log,encoding="utf-8")
theta=.4
c=2.5/2.4
f=lambda t:scalar_loss(t,fixed_scale=c)
h=1e-5
report={
 "candidate":"simple-direct-width-mechanics-v1",
 "purpose":"proposed actor-score objective; mechanical verification only",
 "executed_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime(started)),
 "tests_run":result.testsRun,
 "failures":[{"test":str(t),"traceback":s} for t,s in result.failures],
 "errors":[{"test":str(t),"traceback":s} for t,s in result.errors],
 "skipped":[{"test":str(t),"reason":s} for t,s in result.skipped],
 "success":result.wasSuccessful(),
 "scalar_fixture":{
  "theta":theta,"Q":2.4,"width":2.8,"raw_q_host_scale":c,
  "expected_host_gradient":-c+.8,
  "expected_live_gradient":c+.8,
  "observed_host_gradient":float(jax.grad(lambda t:scalar_loss(t,"none"))(theta)),
  "observed_detached_gradient":float(jax.grad(lambda t:scalar_loss(t,"detached"))(theta)),
  "observed_live_gradient":float(jax.grad(scalar_loss)(theta)),
  "finite_difference_frozen_scale":float((f(theta+h)-f(theta-h))/(2*h)),
  "finite_difference_recomputed_scale_is_different_derivative":float((scalar_loss(theta+h)-scalar_loss(theta-h))/(2*h))
 },
 "source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.glob("*.py"))},
 "environment":{"python":sys.version,"platform":platform.platform(),"jax":jax.__version__,
                "jaxlib":importlib.metadata.version("jaxlib"),"numpy":np.__version__,
                "backend":jax.default_backend(),"x64":bool(jax.config.jax_enable_x64)},
 "runtime_seconds":time.time()-started,
 "limits":["No trained checkpoint loaded","No actor/critic training or simulator rollout",
           "No empirical calibration or Q-coverage guarantee","Not Option C",
           "Finite-input primitive requires external statistical support/refusal decision",
           "Synthetic rankings are illustrative, not evidence about learned uncertainty"]
}
(OUT/"results.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
print(log,end="")
print(json.dumps({"success":report["success"],"tests_run":report["tests_run"],
                  "runtime_seconds":report["runtime_seconds"],"evidence":str(OUT)}))
raise SystemExit(0 if result.wasSuccessful() else 1)
