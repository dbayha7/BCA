"""Read-only gradient audit of the actual frozen native calibrator and candidate.

No training, calibration refresh, transition sampling, policy rollout or optimizer
step. Uses the previously verified 64-state bank and authenticated source closure.
"""
from pathlib import Path
import sys
import json
import hashlib
import time
import argparse
import re
ROOT=Path(__file__).resolve().parent
TASK=ROOT.parents[1]
sys.path.insert(0,str(TASK/"action_only_20261021"))
import engine
engine.configure_runtime()
import jax
import jax.numpy as jnp
import numpy as np
from objective import actor_objective, freeze_parameters, validate_inputs

parser=argparse.ArgumentParser()
parser.add_argument("--evidence-name",default="checkpoint_evidence")
cli=parser.parse_args()
if not re.fullmatch(r"[A-Za-z0-9_-]+",cli.evidence_name):
    raise SystemExit("Evidence name must be a single directory name.")
OUT=ROOT/cli.evidence_name
if OUT.exists():
    raise SystemExit("Existing checkpoint evidence must be preserved.")
OUT.mkdir()
start=time.time()
record={"status":"started","source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in sorted(ROOT.glob("*.py"))},
        "scope":"actual frozen-model gradient audit, zero training/rollouts"}
(OUT/"started.json").write_text(json.dumps(record,indent=2)+"\n")
BUNDLE=TASK/"results_20261021/release/ThrusterChain_results_20261021"
RUN=Path(r"C:\Users\David\dev\BCA\experiments\thrusterchain_diagnostics\training_runs\core_20261020T2250Z")
e=engine.FrozenScoreEngine.load(bundle_root=BUNDLE,canonical_run_dir=RUN,state_seed=2026102101)
from calibration.dose import frozen_level_dose
state=e.result.state
before=e.T.tree_sha256(state)
obs=jnp.asarray((e.selected["observation"].astype(np.float32)-e.result.mean)/e.result.std)
logged=jnp.asarray(e.selected["action"],dtype=jnp.float32)
cp=freeze_parameters(state.native.critic.params)
up=freeze_parameters(state.posterior.cal_params)
R=float(state.posterior.threshold)
unit=float(state.posterior.residual_scale)
def eta(a):
    return e.result.models[2].apply(up,obs,a)
def raw_width(a):
    return jax.lax.stop_gradient(state.posterior.threshold*state.posterior.residual_scale)*jnp.maximum(eta(a),1e-6)
def native_width(a):
    return frozen_level_dose(state.posterior,eta(a),.5).width
def q1(a):
    return e.result.models[1].apply(cp,obs,a)[...,0]
def actor(params):
    return e.result.models[0].apply(params,obs)
action=actor(state.native.actor.params)
u=raw_width(action)
validate_inputs(action,logged,q1(action),u)
native=native_width(action)
np.testing.assert_allclose(np.asarray(u),np.asarray(native),rtol=2e-6,atol=1e-5)
gw=jax.grad(lambda a:raw_width(a).sum())(action)
gn=jax.grad(lambda a:native_width(a).sum())(action)
np.testing.assert_array_equal(np.asarray(gn),np.zeros_like(np.asarray(gn)))
def actor_loss(params,mode):
    a=actor(params)
    return actor_objective(a,logged,q1(a),raw_width(a),mode=mode).loss
host=jax.grad(lambda p:actor_loss(p,"none"))(state.native.actor.params)
live=jax.grad(lambda p:actor_loss(p,"live"))(state.native.actor.params)
detached=jax.grad(lambda p:actor_loss(p,"detached"))(state.native.actor.params)
def flat(tree):
    return np.concatenate([np.asarray(x,dtype=np.float64).ravel() for x in jax.tree_util.tree_leaves(tree)])
hostv,livev,detv=map(flat,(host,live,detached))
np.testing.assert_array_equal(hostv,detv)
# Directional finite differences use raw candidate width. No normalizer enters this check.
g=np.asarray(gw,dtype=np.float64)
direction=g/np.linalg.norm(g) if np.linalg.norm(g)>0 else np.ones_like(g)/np.sqrt(g.size)
direction=direction.astype(np.float32)
expected=float(np.sum(g*direction))
finite=[]
for h in (.003,.001,.0003):
    plus_actions=action+jnp.asarray(h*direction)
    minus_actions=action-jnp.asarray(h*direction)
    in_box=bool(np.all(np.abs(np.asarray(plus_actions))<=1) and
                np.all(np.abs(np.asarray(minus_actions))<=1))
    plus=np.asarray(raw_width(plus_actions),dtype=np.float64).sum()
    minus=np.asarray(raw_width(minus_actions),dtype=np.float64).sum()
    fd=float((plus-minus)/(2*h))
    finite.append(dict(step=h,analytic=expected,finite_difference=fd,
                       relative_error=abs(fd-expected)/max(1.,abs(expected)),
                       both_perturbations_in_action_box=in_box))
# Require at least the two larger predeclared steps to agree within 3%;
# save all three, including any floating-point/kink discrepancy.
fd_ok=all(x["relative_error"]<=.03 and x["both_perturbations_in_action_box"] for x in finite[:2])
after=e.T.tree_sha256(state)
if before!=after: raise AssertionError("Frozen state changed")
actual_off_grid=np.any(~np.isin(np.asarray(action),e.atoms),axis=1)
report=dict(status="completed" if fd_ok else "finite_difference_check_failed",
 candidate="simple-direct-width-mechanics-v1",
 checkpoint=e.metadata["checkpoint"],reference=e.metadata["frozen_reference"],
 selection=e.metadata["selection"],
 engine_sha256=hashlib.sha256(Path(engine.__file__).read_bytes()).hexdigest(),
 audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 model_hashes=e.metadata["model_hashes"],source_sha256=e.metadata["source_sha256"],
 state_hash_before=before,state_hash_after=after,unchanged=(before==after),
 observations=64,action_dimensions=4,
 raw_width=dict(min=float(np.min(u)),max=float(np.max(u)),
                nonzero_action_gradient_rows=int(np.sum(np.linalg.norm(g,axis=1)>0)),
                gradient_l2=float(np.linalg.norm(g)),gradient_max_abs=float(np.max(np.abs(g)))),
 native_detached_width_gradient_l2=float(np.linalg.norm(np.asarray(gn))),
 actor_gradient=dict(host_l2=float(np.linalg.norm(hostv)),live_l2=float(np.linalg.norm(livev)),
                     live_minus_host_l2=float(np.linalg.norm(livev-hostv)),
                     detached_minus_host_l2=float(np.linalg.norm(detv-hostv))),
 finite_difference=finite,finite_difference_pass=fd_ok,
 finite_difference_target="directional derivative with respect to action of sum of raw widths; not full actor-parameter loss",
 off_grid_actor_rows=int(actual_off_grid.sum()),
 backend=jax.default_backend(),jax=jax.__version__,x64=bool(jax.config.jax_enable_x64),
 runtime_seconds=time.time()-start,
 limits=["This changes differentiation, not learned calibrator parameters",
         "Native frozen_level_dose returns a fully detached width; candidate recomputes raw R*unit*eta",
         "Scalar R*unit means action geometry comes from eta, not per-action IW thresholding",
         "All queried continuous actions must be audited for categorical support separately",
         "Finite residual width is not a Q-error certificate",
         "No calibration refresh, optimizer update, training or rollout"])
(OUT/"results.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
np.savez_compressed(OUT/"per_state_gradients.npz",observation=np.asarray(obs),
 action=np.asarray(action),width=np.asarray(u),candidate_width_action_gradient=g,
 native_width_action_gradient=np.asarray(gn),row_id=e.selected["row_id"])
print(json.dumps({k:report[k] for k in ("status","raw_width","native_detached_width_gradient_l2",
 "actor_gradient","finite_difference","off_grid_actor_rows","runtime_seconds")},indent=2))
raise SystemExit(0 if fd_ok else 1)
