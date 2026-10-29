"""Bounded fixed-fixture checks of authenticated, unchanged pure physics.

No Gymnasium environment, policy, dataset, trajectory or benchmark rollout is
constructed. Test-only Gym interface names permit the real canonical AST loader
to define its environment class; attempting to instantiate it fails explicitly.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
CORE = HERE.parents[1] / 'titan_20261026/payload/transfer_bundle/core_v4'
CANONICAL = CORE / 'thrusterchain/canonical'
FIXTURES = CORE / 'contracts/dynamics_fixtures.json'
MAX_NUMPY_ROWS = 128
MAX_JAX_ROWS = 32


def require(condition, message):
    if not condition:
        raise ValueError(message)


@contextmanager
def pure_canonical():
    gym, spaces = ModuleType('gymnasium'), ModuleType('gymnasium.spaces')
    class NoEnvironment:
        def __init__(self, *args, **kwargs):
            raise AssertionError('Pure physics audit forbids environment construction')
    def no_box(*args, **kwargs):
        raise AssertionError('Pure physics audit forbids environment spaces')
    gym.Env, gym.spaces, spaces.Box = NoEnvironment, spaces, no_box
    name = '_environment_audit_canonical'
    with patch.dict(sys.modules, {'gymnasium':gym, 'gymnasium.spaces':spaces}):
        spec = importlib.util.spec_from_file_location(name, CANONICAL / '__init__.py',
                                                      submodule_search_locations=[str(CANONICAL)])
        loader = importlib.util.module_from_spec(spec)
        sys.modules[name] = loader
        spec.loader.exec_module(loader)
        source = loader.load_canonical()
        require(Path(source.dynamics.__code__.co_filename).resolve() == (CANONICAL/'thrusterchain.py').resolve(),
                'Wrong compilation path')
        require(source.dynamics.__globals__ is source.__dict__, 'Wrong source binding')
        yield source, loader


def run_checks():
    result = {'schema':'bca.environment_pure_physics_audit.v1',
              'started_utc':datetime.now(timezone.utc).isoformat(), 'status':'running',
              'counts':{'numpy_dynamics_rows':0, 'jax_dynamics_rows':0,
                        'environments_constructed':0,'resets':0,'rollout_episodes':0,
                        'dataset_rows_decoded':0,'model_calls':0,'optimizer_updates':0},
              'caps':{'numpy_dynamics_rows':MAX_NUMPY_ROWS,'jax_dynamics_rows':MAX_JAX_ROWS},
              'versions':{'python':sys.version,'numpy':np.__version__},'checks':{},
              'scope':'Fixed synthetic/archival one-step fixtures and analytic matrix arithmetic; no empirical sweep or trained-policy evaluation',
              'gymnasium':'Test-only class/space names; environment instantiation explicitly prohibited'}
    def check(name, passed, **evidence):
        result['checks'][name] = {'passed':bool(passed), **evidence}
        require(passed, 'Failed physics check: ' + name)
    def maxdiff(a,b):
        return float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
    try:
        with pure_canonical() as (src, loader):
            source_hash = hashlib.sha256((CANONICAL/'thrusterchain.py').read_bytes()).hexdigest()
            result['source_sha256'] = source_hash
            result['matrix_sha256'] = loader.MATRIX_SHA256
            result['fixture_sha256'] = hashlib.sha256(FIXTURES.read_bytes()).hexdigest()
            result['physics'] = src.STALL.as_dict()
            def dynamics(x,a,p=None):
                rows = 1 if np.asarray(x).ndim == 1 else len(x)
                require(result['counts']['numpy_dynamics_rows']+rows <= MAX_NUMPY_ROWS,'NumPy row cap')
                result['counts']['numpy_dynamics_rows'] += rows
                return src.dynamics(x,a,src.STALL if p is None else p)

            fixtures = json.loads(FIXTURES.read_text())
            state_residual = reward_residual = 0.
            for case in fixtures['cases']:
                p = src.STALL if case['world']=='stall' else src.NO_STALL
                nxt,r = dynamics(case['state'],case['command'],p)
                state_residual=max(state_residual,maxdiff(nxt,case['next_state']))
                reward_residual=max(reward_residual,abs(float(r)-case['reward']))
            check('archival_numpy_parity',state_residual==reward_residual==0.,
                  cases=len(fixtures['cases']),max_state_residual=state_residual,max_reward_residual=reward_residual,tolerance=0.)

            x=np.array([v['state'] for v in fixtures['cases'][:8]],np.float64)
            a=np.array([v['command'] for v in fixtures['cases'][:8]],np.float64)
            batch,rewards=dynamics(x,a)
            scalar=[dynamics(xx,aa) for xx,aa in zip(x,a)]
            check('scalar_batch_bitwise',np.array_equal(batch,np.stack([z[0] for z in scalar]))
                  and np.array_equal(rewards,np.asarray([z[1] for z in scalar])),rows=8,tolerance=0.)

            invariant_states=np.zeros((4,13),np.float64)
            invariant_states[:,10]=[0.,.4,1.1,1.5]
            invariant_states[:,11:]=[[0.,0.],[.3,.4],[1.,0.],[0.,-1.]]
            nxt,_=dynamics(invariant_states,np.zeros((4,4)))
            wave=maxdiff(np.sum(nxt[:,11:]**2,axis=1),np.sum(invariant_states[:,11:]**2,axis=1))
            check('current_and_wave_invariants',np.array_equal(nxt[:,10],invariant_states[:,10]) and wave<=5e-16,
                  max_wave_squared_amplitude_residual=wave,tolerance=5e-16,
                  current_exact=True,note='Wave norm invariant, not energy conservation of driven/damped float coordinates')

            aa,bb=src.matrices(); a10=aa[:10,:10]
            norm=float(np.linalg.norm(a10,2)); rho=float(np.max(np.abs(np.linalg.eigvals(a10))))
            # A finite power sum plus a rigorous geometric tail bound in the
            # induced norm. No dynamics calls or simulation are involved.
            m=np.eye(10); partial=0.
            for _ in range(1000):
                partial+=float(np.linalg.norm(m,2)); m=m@a10
            tail=float(np.linalg.norm(m,2)/(1.-norm))
            input_bound=float(np.linalg.norm(bb[:10],2)*2. + np.linalg.norm(aa[:10,10])*(2.5+src.WAVE_ROUNDING))
            state_bound=100.+(partial+tail)*input_bound
            check('matrix_stability_and_bound',rho<1 and norm<1 and state_bound<300,
                  physical_matrix_spectral_radius=rho,physical_matrix_norm2=norm,
                  power_sum_1000=partial,tail_upper_bound=tail,per_step_input_bound=input_bound,
                  conservative_float_radius_bound=state_bound,declared_observation_limit=300.,
                  note='Physical10D subsystem is contractive; current/wave augmented modes are neutral, not contractive')

            k=src.KNEE
            points=np.array([0.,np.nextafter(k,0.),k,np.nextafter(k,np.inf),1.,-k,-1.])
            expected=np.clip(points,-k,k)-1.5*np.sign(points)*np.maximum(abs(points)-k,0)
            actual=src.thrust(points)
            check('thrust_boundaries',np.array_equal(actual,expected) and actual[0]==0 and actual[2]==k
                  and not src.stalled(np.array([k]))[0] and src.stalled(np.array([np.nextafter(k,np.inf)]))[0],
                  inputs=points.tolist(),thrust=actual.tolist(),tolerance=0.)

            reward_state=np.zeros(13);reward_state[:2]=[3.,4.]
            _,rr=dynamics(reward_state,np.ones(4))
            check('reward_prestate_and_command',float(rr)==-(25.+1.2),actual=float(rr),expected=-26.2,
                  note='Reward uses current state and clipped command, not next state or diminished thrust')

            clipped_actions=np.array([[2.,-2.,0.,0.],[1.,-1.,.2,-.3],[-1.2,.9,.8,-.8]])
            cx=np.repeat(reward_state[None,:],3,axis=0)
            cn,cr=dynamics(cx,clipped_actions);en,er=dynamics(cx,np.clip(clipped_actions,-1,1))
            check('action_clipping_consistency',np.array_equal(cn,en) and np.array_equal(cr,er),rows=3,tolerance=0.)

            # Every post-knee throttle used below has an unstalled command with
            # identical thrust but smaller command cost. Future state is equal.
            dominance=[]
            for high in (float(np.nextafter(np.float32(1),np.float32(0))),
                         -float(np.nextafter(np.float32(.9),np.float32(1)))):
                action=np.array([high,0.,0.,0.]); lower=action.copy();lower[0]=src.thrust(action)[0]
                require(float(np.float32(lower[0]))==lower[0], 'Dominance twin must be exactly deployable')
                hn,hr=dynamics(reward_state,action);ln,lr=dynamics(reward_state,lower)
                difference=float(lr-hr); exact=.3*(high*high-lower[0]*lower[0])
                dominance.append({'post_knee':high,'equivalent_unstalled':float(lower[0]),
                                  'successor_max_residual':maxdiff(hn,ln),'reward_improvement':difference,
                                  'formula_residual':abs(difference-exact)})
            check('post_knee_dominance',all(x['successor_max_residual']==0. and x['reward_improvement']>0.
                  and x['formula_residual']<5e-15 for x in dominance),cases=dominance,
                  note='These float32-representable twins have equal successors and future return. Arbitrary continuous-thrust twins may round at deployment; no universal exact quantized-action dominance claim.')

            # Terminal Q has no future term, so energy alone chooses zero action.
            _,zero_r=dynamics(reward_state,np.zeros(4));_,nonzero_r=dynamics(reward_state,np.full(4,.5))
            check('last_action_incentive',abs(float(zero_r-nonzero_r)-.3)<=1e-14,
                  terminal_q_improvement=float(zero_r-nonzero_r),
                  note='At t99, Qpi=r; no terminal state cost/bootstrap. Earlier control can improve future state costs.')

            import jax
            import jax.numpy as jnp
            result['versions']['jax']=jax.__version__
            result['backend']=jax.default_backend()
            require(result['backend']=='cpu','This preparation fixture is CPU only')
            def jd(x,a):
                rows=1 if x.ndim==1 else len(x)
                require(result['counts']['jax_dynamics_rows']+rows <= MAX_JAX_ROWS,'JAX row cap')
                result['counts']['jax_dynamics_rows'] += rows
                return src.dynamics_jax(x,a,src.STALL)
            with jax.experimental.enable_x64():
                sx=np.array([[.1,-.2,.3,-.4,.5,-.6,.7,-.8,.9,-1.,.4,.2,-.1],
                             [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.5,1.,0.],
                             [-.4,.2,-.3,.1,0.,.1,.2,-.2,.3,-.1,.1,-.3,.2]],np.float64)
                sa=np.array([[.2,-.3,.1,.4],[.9,-.9,.2,-.2],[0.,.7,-.7,.4]],np.float64)
                nn,nr=dynamics(sx,sa);jn,jr=jd(jnp.asarray(sx),jnp.asarray(sa))
                jp=max(maxdiff(nn,np.asarray(jn)),maxdiff(nr,np.asarray(jr)))
                check('jax64_numpy_parity',jp<=2e-13,max_residual=jp,tolerance=2e-13,rows=3,dtype=str(jn.dtype))
                z=np.r_[sx[0],sa[0]]
                direction=np.array([.2,-.3,.1,.4,-.2,.1,.3,-.1,.2,-.4,.1,.2,-.3,.3,-.2,.1,.2])
                def function(v):
                    ns,reward=jd(v[:13],v[13:]);return jnp.concatenate((ns,jnp.asarray(reward)[None]))
                _,tangent=jax.jvp(function,(jnp.asarray(z),),(jnp.asarray(direction),))
                residuals=[]
                for eps in (1e-4,1e-5,1e-6):
                    plus,pr=dynamics((z+eps*direction)[:13],(z+eps*direction)[13:])
                    minus,mr=dynamics((z-eps*direction)[:13],(z-eps*direction)[13:])
                    finite=(np.r_[plus,pr]-np.r_[minus,mr])/(2*eps)
                    residuals.append({'epsilon':eps,'max_residual':maxdiff(finite,np.asarray(tangent))})
                slope=float(jax.grad(lambda a:src.thrust_jax(a))(jnp.float64(0.)))
                check('jax_jvp_finite_difference_smooth',max(v['max_residual'] for v in residuals)<=2e-8
                      and slope==1.,directions=1,epsilon_checks=residuals,tolerance=2e-8,
                      zero_throttle_slope=slope,note='Smooth interior fixture; derivatives at clipping/stall cusps are not uniquely defined')
            result['status']='passed_fixed_physics_checks'
    except BaseException as exc:
        result['status']='failed_fixed_physics_checks'
        result['error']={'type':type(exc).__name__,'message':str(exc)}
    result['finished_utc']=datetime.now(timezone.utc).isoformat()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result=run_checks()
    with args.output.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    print(json.dumps({'status':result['status'],'checks':len(result['checks']),
                      'counts':result['counts'],'error':result.get('error')}))
    return 0 if result['status']=='passed_fixed_physics_checks' else 1


if __name__=='__main__':
    raise SystemExit(main())
