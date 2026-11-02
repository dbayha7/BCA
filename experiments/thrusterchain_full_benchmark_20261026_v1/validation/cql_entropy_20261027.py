"""Supplemental prospective entropy/branch check; see protocol."""
import contextlib
import io
import json
from pathlib import Path
import runpy
import jax
import numpy as np

with contextlib.redirect_stdout(io.StringIO()):
    x=runpy.run_path(str(Path(__file__).with_name('cql_reference_20261027.py')))
args=x['args']; state=x['state']; new=x['new']; metrics=x['metrics']
assert args.use_automatic_entropy_tuning and not args.backup_entropy
assert not args.cql_max_target_backup and not args.cql_lagrange
assert args.cql_importance_sample and args.bc_steps==0
assert float(np.asarray(state.log_alpha.params['constant'])) == 0.
assert float(metrics['alpha']) == 1.
_,numpy_logpi=x['draw'](x['pa'],x['obs'],x['epspi'])
expected = -float(np.mean(numpy_logpi-2.))
observed = -float(np.asarray(new.log_alpha.params['constant']))/.01
np.testing.assert_allclose(observed,expected,atol=2e-5,rtol=2e-5)
assert float(np.asarray(new.log_alpha_prime.params['constant'])) == 1.
assert int(new.log_alpha.step)==1 and int(new.log_alpha_prime.step)==1
print(json.dumps({'status':'pass','device':list(map(str,jax.devices())),
    'alpha_gradient_numpy':expected,'alpha_gradient_observed':observed,
    'absolute_error':abs(expected-observed),'log_alpha_prime_unchanged':True},indent=2))
