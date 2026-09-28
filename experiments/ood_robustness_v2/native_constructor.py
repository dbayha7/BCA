"""Observe one native constructor call without constructing/importing science.

Runtime integration must bind the exact native classes/factory/source and reserve
an independently accepted engineering call. This component never seeds/samples,
replaces native controls, invents wrapper state, or accepts scientific outcomes.
"""
import copy
import hashlib
import re
from unittest.mock import patch
import numpy as np
from precommit_bank import capture_schedule
from recorded_step import require,content_hash,encode

DATA_FIELDS=('qpos','qvel','act','qacc_warmstart','ctrl','qfrc_applied','xfrc_applied','mocap_pos','mocap_quat','userdata')
MODEL_FIELDS=('body_pos','body_quat','site_pos')


def _rng_existing(obj):
    # Never use obj.np_random: its lazy property can initialize an absent RNG.
    rng=vars(obj).get('_np_random')
    if rng is None: return dict(status='not_initialized',state=None)
    require(hasattr(rng,'bit_generator') or hasattr(rng,'get_state'),'Unknown existing RNG interface.')
    state=rng.bit_generator.state if hasattr(rng,'bit_generator') else rng.get_state()
    return dict(status='present',state=copy.deepcopy(state))


def capture_native(env):
    try:
        data={k:copy.deepcopy(getattr(env.sim.data,k)) for k in DATA_FIELDS}
        model={k:copy.deepcopy(getattr(env.sim.model,k)) for k in MODEL_FIELDS}
        result=dict(schema='ood-v2-constructor-native-state-v1',data=data,model=model,
            model_sha256=hashlib.sha256(env.sim.model.get_mjb()).hexdigest(),time=float(env.sim.data.time),
            udd_state=copy.deepcopy(env.sim.get_state().udd_state),dt=float(env.dt),frame_skip=int(env.frame_skip),
            native_low=env.action_space.low.copy(),native_high=env.action_space.high.copy(),
            actuator_ctrlrange=env.sim.model.actuator_ctrlrange.copy(),environment_rng=_rng_existing(env),
            action_rng=_rng_existing(env.action_space),wrapper_state_available=False,
            unavailable_wrapper_fields=['TimeLimit._elapsed_steps','OrderEnforcing._has_reset',
                                        'NormalizedBoxEnv.action_rng','adapter.terminated','adapter.truncated'],
            presampling_rng_state_recorded=False)
    except (AttributeError,TypeError) as e: raise ValueError('Incomplete constructor native evidence interface.') from e
    encode(result)
    return result


def validate_constructor(before,action,after,returned,environment):
    require(environment in ('hopper','walker2d'),'Unknown native constructor environment.')
    dim,n=(3,6) if environment=='hopper' else (6,9)
    a=np.asarray(action)
    require(a.dtype==np.float32 and a.shape==(dim,) and np.isfinite(a).all() and (abs(a)<=1).all(),'Invalid sampled native action.')
    for state in (before,after):
        require(state['schema']=='ood-v2-constructor-native-state-v1' and state['wrapper_state_available'] is False,
                'Wrong constructor native schema; no invented wrapper state.')
        require(set(state['data'])==set(DATA_FIELDS) and set(state['model'])==set(MODEL_FIELDS),'Incomplete native capture.')
        for k in ('qpos','qvel'):
            v=np.asarray(state['data'][k]);require(v.dtype==np.float64 and v.shape==(n,) and np.isfinite(v).all(),'Wrong native state geometry.')
        require(np.array_equal(state['native_low'],-np.ones(dim,np.float32)) and np.array_equal(state['native_high'],np.ones(dim,np.float32)),'Nonunit native bounds.')
        require(np.array_equal(state['actuator_ctrlrange'],np.tile([-1.,1.],(dim,1))),'Actuator range differs from native bounds.')
        require(state['frame_skip']==4 and np.isfinite(state['dt']) and state['dt']>0,'Wrong constructor physics interval.')
    for key in ('model','model_sha256','dt','frame_skip','native_low','native_high','actuator_ctrlrange','environment_rng','action_rng'):
        require(content_hash(before[key])==content_hash(after[key]),'Constructor changed static model/bounds/RNG: '+key)
    require(after['time']==before['time']+before['dt'],'Constructor native time interval differs.')
    ctrl=np.asarray(after['data']['ctrl'])
    require(ctrl.dtype==np.float64 and ctrl.shape==a.shape and np.array_equal(ctrl,a),'Actual constructor control mismatch.')
    require(type(returned) is tuple and len(returned)==4,'Expected native four-value return.')
    observation,reward,done,info=returned
    require(isinstance(reward,(float,np.floating)) and np.isfinite(reward) and type(done) is bool and type(info) is dict,'Invalid native returned scalar/flags.')
    qpos,qvel=after['data']['qpos'],after['data']['qvel']
    expected_obs=np.concatenate([qpos[1:],np.clip(qvel,-10,10)])
    require(isinstance(observation,np.ndarray) and observation.dtype==np.float64 and np.array_equal(observation,expected_obs),'Constructor observation arithmetic differs.')
    if environment=='hopper':
        vector=np.concatenate([qpos,qvel]);healthy=np.isfinite(vector).all() and (abs(vector[2:])<100).all() and qpos[1]>.7 and abs(qpos[2])<.2
    else: healthy=.8<qpos[1]<2 and -1<qpos[2]<1
    require(done==(not bool(healthy)) and not done,'Constructor health/done mismatch or terminated construction.')
    forward=(float(qpos[0])-float(before['data']['qpos'][0]))/before['dt']
    cost=-.001*float(np.square(a.astype(np.float64)).sum());reconstructed=forward+1.+cost
    error=abs(float(reward)-reconstructed)
    require(np.isfinite(error) and error<=1e-7,'Unchanged constructor reward1e-7 gate failed.')
    return dict(forward=forward,action_cost=cost,reconstructed_reward=reconstructed,reward_absolute_error=error,
                exact_control=True,exact_observation=True,native_health_checked=True,observation_size=len(expected_obs))


def construct_recorded(factory,native_class,physics_class,ledger,store,pair,token):
    capture_schedule(*pair)
    require(type(token) is str and re.fullmatch(r'[A-Za-z0-9_-]{1,100}',token) is not None,'Invalid constructor token.')
    require('step' in native_class.__dict__ and 'do_simulation' in physics_class.__dict__,'Bind exact defining native classes.')
    native_step=native_class.__dict__['step'];physical_step=physics_class.__dict__['do_simulation']
    require(callable(factory) and callable(native_step) and callable(physical_step),'Invalid native constructor interfaces.')
    host,environment,_=pair;scope=f'engineering/{host}/{environment}';seen={};adapter=None

    def observe_physics(env,control,frames):
        require(seen.get('env') is env and not seen.get('physics_entered',False),'Unexpected/extra constructor physics; stop before call.')
        require(type(frames) is int and frames==4 and np.asarray(control).dtype==np.float32 and np.array_equal(control,seen['action']),
                'Constructor intercepted control/frame mismatch; stop before physics.')
        seen['physics_entered']=True
        seen['applied_sha256']=store.put('constructor/applied-before-physics',dict(control=np.asarray(control).copy(),frames=frames,
            native_before=capture_native(env),token=token,scope=scope))
        return physical_step(env,control,frames)

    def observe_native(env,action):
        require('env' not in seen,'Unexpected second constructor native step; stop before physics.')
        seen['env']=env;seen['action']=np.asarray(action).copy();seen['before']=capture_native(env)
        dim=3 if environment=='hopper' else 6
        require(seen['action'].dtype==np.float32 and seen['action'].shape==(dim,) and np.isfinite(seen['action']).all()
                and (abs(seen['action'])<=1).all(),'Invalid actual sampled constructor action; no substitution.')
        seen['input_sha256']=store.put('constructor/input',dict(token=token,scope=scope,action=seen['action'],native_before=seen['before'],
            action_origin='native action_space.sample before native step; unmodified',presampling_entropy_reconstruction_claimed=False))
        try:
            returned=native_step(env,action)
        except BaseException as error:
            # A native exception can follow some/all physics. Preserve any
            # available poststate without inventing a return or acknowledging it.
            try:
                store.put('constructor/failed-native-after',dict(native_after=capture_native(env),
                    return_available=False,exception_type=type(error).__name__,exception_message=str(error),
                    physics_method_entered=seen.get('physics_entered',False)))
            except BaseException:
                pass  # Original exception remains authoritative; reservation stays pending.
            raise
        seen['after']=capture_native(env);seen['returned']=copy.deepcopy(returned)
        seen['output_sha256']=store.put('constructor/output',dict(returned=seen['returned']))
        seen['after_sha256']=store.put('constructor/after',seen['after'])
        return returned

    def callback():
        nonlocal adapter
        try:
            with patch.object(native_class,'step',observe_native),patch.object(physics_class,'do_simulation',observe_physics):
                adapter=factory()
            require('returned' in seen and seen.get('physics_entered',False),'Constructor did not record exactly one complete native call.')
            require(adapter.base is seen['env'] and adapter.transitions==0 and adapter.frame_skip==4,'Factory returned a different native adapter.')
            require(content_hash(capture_native(adapter.base))==content_hash(seen['after']),'Native state changed after observed constructor step.')
            arithmetic=validate_constructor(seen['before'],seen['action'],seen['after'],seen['returned'],environment)
            result=dict(schema='ood-v2-constructor-evidence-v1',token=token,scope=scope,pair=list(pair),
                environment_reserved=1,physics_reserved=4,full_native_states_saved=True,wrapper_restore_state_claimed=False,
                native_input_sha256=seen['input_sha256'],intercepted_applied_sha256=seen['applied_sha256'],
                native_output_sha256=seen['output_sha256'],native_after_sha256=seen['after_sha256'],
                presampling_rng_state_reconstructed=False,independent_scientific_acceptance=False,**arithmetic)
            receipt=store.put('constructor/completed',result)
            return (adapter,result),receipt
        except BaseException:
            obj=adapter if adapter is not None else seen.get('env')
            if obj is not None:
                try: obj.close()
                except BaseException as close_error:
                    # Keep original failure authoritative; attempt to retain cleanup diagnosis.
                    try: store.put('constructor/cleanup-error',dict(type=type(close_error).__name__,message=str(close_error)))
                    except BaseException: pass
            raise
    return ledger.call(token,scope,callback)
