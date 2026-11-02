"""Bounded frozen GPU operator audit wiring; imported without scientific work.

Only run() loads models or calls the evaluator. No learner, fit, refresh, new
dataset generation, or continuation rollout is called by this module.
"""
from copy import deepcopy
import hashlib
import importlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
READER = BASE / 'unifloral_development_mechanism_audit_v1p1'
READER_SHA = '099f928807618cc4dccd30265f70dc1175e8a6059ae9924d1ff1e15447c88278'
READER_MANIFEST_SHA = '310781618e4eecade8a2c3b1143a2eacb1f9e40c559aa62339d953eb570f941c'
SIMULATION = BASE / 'unifloral_frozen_usefulness_v1_seedfix'
SIMULATION_SHA = '0c78e20eb16884a1868c372e45dcfc2659723058b01d04655f689614ade7d824'
SIMULATION_PROTOCOL_SHA = '7572712fdf65f834f2aeF7c977cc7e138837ed7a9c40f8121cc04109734f2cb1'.lower()
BUNDLE_SHA = '3443280fd4340ba64aef4d9601136257014bf2a6c61ca8c890eba06afa76690a'
_DEFAULT_BUNDLE = (r'C:\Users\David Bayha\Downloads\titan_transfer_20261022_v1\titan_transfer_20261022_v1'
                   if os.name == 'nt' else '/mnt/c/Users/David Bayha/Downloads/titan_transfer_20261022_v1/titan_transfer_20261022_v1')
BUNDLE = Path(os.environ.get('BCA_TRANSFER_BUNDLE_ROOT', _DEFAULT_BUNDLE))
DEV_SEEDS = [2026103041, 2026103042, 2026103043, 2026103044]
FIT_SEEDS = [2026103051, 2026103052, 2026103053, 2026103054]


class OperatorExecutionError(RuntimeError):
    def __init__(self, message, ledger):
        super().__init__(message)
        self.ledger = deepcopy(ledger)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _module(name, path, digest):
    raw = Path(path).read_bytes()
    if _sha(raw) != digest:
        raise ValueError('Pinned execution source changed: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


def _finite(value, shape, name, dtype=None):
    import numpy as np
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in 'fiu' or not np.all(np.isfinite(array)):
        raise ValueError('Invalid finite shape for ' + name)
    if dtype is not None and array.dtype != np.dtype(dtype):
        raise ValueError('Storage dtype changed: ' + name)
    return array


def assert_replay(actual, expected, name, rtol=2e-5, atol=2e-6):
    import numpy as np
    actual = np.asarray(actual); expected = np.asarray(expected)
    if actual.shape != expected.shape or not np.all(np.isfinite(actual)) or not np.all(np.isfinite(expected)):
        raise ValueError('Invalid replay arrays: ' + name)
    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
        raise ValueError('Saved forward replay failed: ' + name)
    return {'name': name, 'rows': len(actual), 'max_abs_difference': float(np.max(np.abs(actual-expected))),
            'rtol': rtol, 'atol': atol, 'passed': True}


def new_ledger():
    return {'schema': 'unifloral.operator_alignment.execution.v1', 'status': 'pending',
            'transition_rows_attempted': 0, 'transition_rows_completed': 0,
            'max_transition_rows': 1152, 'full_continuations': 0,
            'current_actor_rows': 0, 'current_critic_rows': 0, 'width_rows': 0,
            'target_actor_rows': 0, 'target_critic_rows': 0, 'max_target_critic_rows': 5632,
            'native_smoothing_draws': 0, 'max_native_smoothing_draws': 8,
            'fitting_transition_rows_read': 0, 'new_diagnostic_fits': 0,
            'policy_optimizer_updates': 0, 'calibrator_updates': 0,
            'new_candidate_actions': 0, 'confirmation_files_read': 0,
            'checkpoint_rng_advanced': False, 'conditional_continuation_mc_error': 0,
            'native_smoothing_draws_are_new_not_original_refresh_replay': True}


def observation(states, remaining):
    import numpy as np
    return np.concatenate((np.asarray(states, np.float64), np.asarray(remaining, np.float64)[:, None]/100.), axis=1).astype(np.float32)


def one_step(states, actions, times, dynamics, physics, ledger):
    """One charged canonical transition per saved state/action; pure fakeable API."""
    import numpy as np
    n = len(states)
    states = _finite(states, (n,13), 'full saved physical states', np.float64)
    actions = _finite(actions, (n,4), 'saved deployed actions', np.float32)
    times = _finite(times, (n,), 'saved times')
    if not np.all(times == times.astype(np.int64)) or np.any(times < 0) or np.any(times >= 100):
        raise ValueError('Invalid finite-horizon times')
    if np.any(np.abs(actions) > 1):
        raise ValueError('Refusing hidden action clipping')
    if ledger['transition_rows_attempted'] + n > ledger['max_transition_rows']:
        raise ValueError('One-step transition budget would be exceeded')
    ledger['transition_rows_attempted'] += n
    next_states, rewards = dynamics(states, actions.astype(np.float64), physics)
    ledger['transition_rows_completed'] += n
    next_states = _finite(next_states, (n,13), 'next physical states', np.float64)
    rewards = _finite(rewards, (n,), 'pre-transition physical rewards', np.float64)
    if np.any(np.sqrt(np.sum(next_states[:,:10]**2,axis=1)) > 300.):
        raise ValueError('Canonical physical state bound exceeded')
    state_cost = states[:,0]*states[:,0]
    for j in range(1,10): state_cost = state_cost + states[:,j]*states[:,j]
    a64 = actions.astype(np.float64)
    action_cost = a64[:,0]*a64[:,0]
    for j in range(1,4): action_cost = action_cost + a64[:,j]*a64[:,j]
    expected_reward = -(state_cost + physics.throttle_cost*action_cost)
    if not np.array_equal(rewards, expected_reward):
        raise ValueError('Canonical pre-transition reward accounting failed')
    done = (times.astype(np.int64)+1 == 100).astype(np.float32)
    return {'next_obs': observation(next_states,99-times), 'next_states': next_states,
            'physics_rewards': rewards, 'rewards': rewards.astype(np.float32), 'done': done}


def assemble_smoothed_actions(actor_actions, raw_noise):
    """Float32 native clipping arithmetic, usable without model imports in tests."""
    import numpy as np
    n = len(actor_actions)
    actor_actions = _finite(actor_actions,(n,4),'target actor actions',np.float32)
    raw_noise = _finite(raw_noise,(n,4),'scaled Gaussian smoothing noise',np.float32)
    if np.any(np.abs(actor_actions)>1): raise ValueError('Target actor emitted out-of-bounds actions')
    clipped = np.clip(raw_noise,np.float32(-.5),np.float32(.5))
    return clipped,np.clip(actor_actions+clipped,np.float32(-1),np.float32(1))


def _validate_protocol(protocol):
    expected={'gamma':.98,'horizon':100,'development_rows':1152,'fitting_rows':256,
              'max_transition_steps':1152,'max_target_critic_rows':5632,'smoothing_draws':4,
              'development_noise_seeds':DEV_SEEDS,'fitting_noise_seeds':FIT_SEEDS,
              'noise_std':.2,'noise_clip':.5,'action_limit':1.,'replay_rtol':2e-5,'replay_atol':2e-6}
    if any(protocol.get(key)!=value for key,value in expected.items()):
        raise ValueError('Frozen operator execution task or budget changed')


def verify(protocol):
    """Stdlib hashes/metadata only: no NumPy, JAX, model or physics execution."""
    _validate_protocol(protocol)
    if _sha((READER/'prepared_files.json').read_bytes()) != READER_MANIFEST_SHA:
        raise ValueError('Prior development reader manifest changed')
    reader = _module('_operator_saved_development_reader',READER/'analyze_saved.py',READER_SHA)
    identity,_ = reader.verify()
    raw = (BUNDLE/'BUNDLE_MANIFEST.json').read_bytes()
    if _sha(raw) != BUNDLE_SHA: raise ValueError('Core bundle manifest changed')
    files = json.loads(raw)['files']
    core_files = {name:digest for name,digest in files.items() if name.startswith('core_v4/')}
    if not core_files: raise ValueError('Missing pinned core source map')
    for name,digest in core_files.items():
        path = (BUNDLE/name).resolve()
        if not path.is_relative_to(BUNDLE.resolve()) or _sha(path.read_bytes()) != digest:
            raise ValueError('Core source changed: '+name)
    if _sha((SIMULATION/'simulation.py').read_bytes()) != SIMULATION_SHA:
        raise ValueError('Pinned simulator helper changed')
    prior_protocol_raw = (SIMULATION/'protocol.json').read_bytes()
    if _sha(prior_protocol_raw) != SIMULATION_PROTOCOL_SHA: raise ValueError('Pinned evaluator protocol changed')
    return {'development_identity':identity, 'bundle_manifest_sha256':BUNDLE_SHA,
            'core_files_verified':len(core_files),'simulation_sha256':SIMULATION_SHA,
            'simulation_protocol_sha256':SIMULATION_PROTOCOL_SHA,
            'scope':'Hashes and metadata only; no model, simulator or numeric outcome execution'}


def _source_reader_and_core(protocol):
    sources=verify(protocol)
    reader=_module('_operator_saved_development_reader',READER/'analyze_saved.py',READER_SHA)
    simulation = _module('_operator_pinned_simulation',SIMULATION/'simulation.py',SIMULATION_SHA)
    prior_protocol_raw = (SIMULATION/'protocol.json').read_bytes()
    if _sha(prior_protocol_raw) != SIMULATION_PROTOCOL_SHA: raise ValueError('Pinned evaluator protocol changed')
    sys.path.insert(0,str(BUNDLE/'core_v4'))
    core = importlib.import_module('thrusterchain.environment')
    if not Path(core.__file__).resolve().is_relative_to((BUNDLE/'core_v4').resolve()):
        raise ValueError('A different ThrusterChain core was already imported')
    core_config = simulation._core_contract(core,json.loads(prior_protocol_raw))
    return reader,simulation,core,{**sources,'core_config':core_config}


def _development_inputs(reader):
    import numpy as np
    batches = reader.load_batches()
    states=[]; observations=[]; actions=[]; qs=[]; us=[]; values=[]; metadata=[]; actor_obs=[]; actor_actions=[]
    for spec,batch in zip(reader.BATCHES,batches):
        raw = reader.input_bytes(spec['directory']+'/queries_and_actions.npz')
        with np.load(io.BytesIO(raw),allow_pickle=False) as archive:
            for i,row in enumerate(batch['queries']):
                physical = _finite(archive['state_'+str(i)],(13,),'saved full state',np.float64)
                if not np.array_equal(row['observation'],observation(physical[None,:],np.array([100-row['time']]))[0]):
                    raise ValueError('Saved full state/time/observation mismatch')
                actor_obs.append(row['observation']); actor_actions.append(row['actions'][0])
                for candidate in range(9):
                    states.append(physical.copy()); observations.append(row['observation']); actions.append(row['actions'][candidate])
                    qs.append([row['q1'][candidate],row['q2'][candidate]]); us.append(row['u'][candidate]); values.append(row['values'][candidate])
                    metadata.append({key:row[key] for key in ('group_id','time','batch','episode_seed')})
                    metadata[-1]['candidate_id']=candidate
    if len(states)!=1152 or len(actor_obs)!=128: raise ValueError('Development candidate bank size changed')
    return {'states':np.asarray(states,np.float64),'observations':np.asarray(observations,np.float32),
        'actions':np.asarray(actions,np.float32),'saved_q':np.asarray(qs,np.float64),'saved_u':np.asarray(us,np.float64),
        'saved_values':np.asarray(values,np.float64),'metadata':metadata,
        'actor_observations':np.asarray(actor_obs,np.float32),'saved_actor_actions':np.asarray(actor_actions,np.float32)}


def _current(bindings,bank,ledger):
    import numpy as np
    n=len(bank['observations']); bindings.assert_unchanged()
    ledger['current_critic_rows']+=n
    q=_finite(bindings.policy.q(bank['observations'],bank['actions']),(n,2),'current twin Q',np.float32)
    ledger['width_rows']+=n
    u,usable,valid=bindings.policy.width(bank['observations'],bank['actions'])
    u=_finite(u,(n,),'frozen radius',np.float32)
    if not np.all(np.asarray(usable,dtype=bool)) or not np.all(np.asarray(valid,dtype=bool)) or np.any(u<0):
        raise ValueError('Frozen width is invalid or unusable; no zero fallback')
    bank.update(q_current=q,u=u,u_usable=usable,u_valid=valid)
    bindings.assert_unchanged()


def _targets(bindings,bank,seeds,ledger):
    import numpy as np
    import jax
    import jax.numpy as jnp
    n=len(bank['next_obs'])
    ledger['target_actor_rows']+=n
    target_actor=_finite(bindings.target_act(bank['next_obs']),(n,4),'target actor',np.float32)
    arrays={name:[] for name in ('raw_noise','clipped_noise','next_actions','target_q','native_y')}
    for seed in seeds:
        if ledger['native_smoothing_draws']>=ledger['max_native_smoothing_draws'] or ledger['target_critic_rows']+n>ledger['max_target_critic_rows']:
            raise ValueError('Native target draw budget would be exceeded')
        ledger['native_smoothing_draws']+=1
        raw=jax.random.normal(jax.random.PRNGKey(seed),shape=(n,4),dtype=jnp.float32)*jnp.float32(.2)
        clipped=jnp.clip(raw,jnp.float32(-.5),jnp.float32(.5))
        next_action=jnp.clip(jnp.asarray(target_actor)+clipped,jnp.float32(-1),jnp.float32(1))
        ledger['target_critic_rows']+=n
        target_q=_finite(bindings.target_q(bank['next_obs'],np.asarray(next_action)),(n,2),'target twin Q',np.float32)
        y=jnp.asarray(bank['rewards'],jnp.float32)+jnp.float32(.98)*(jnp.float32(1)-jnp.asarray(bank['done'],jnp.float32))*jnp.min(jnp.asarray(target_q),axis=-1)
        for name,value in zip(arrays,(raw,clipped,next_action,target_q,y)):
            arrays[name].append(np.asarray(value))
        bindings.assert_unchanged()
    bank.update({name:np.stack(values) for name,values in arrays.items()})
    bank['target_actor_next']=target_actor
    bank['smoothing_seeds']=list(seeds)


def run(protocol):
    """Called only by the manual GPU worker after runtime/provenance gates."""
    import numpy as np
    ledger=new_ledger()
    try:
        _validate_protocol(protocol)
        own_manifest=json.loads((ROOT/'prepared_files.json').read_bytes())
        checkpoint_io=_module('_operator_alignment_checkpoint_io',ROOT/'checkpoint_io.py',own_manifest['files']['checkpoint_io.py'])
        bindings=checkpoint_io.load(protocol)
        before=bindings.assert_unchanged()
        snapshot=bindings.receipt.get('snapshot_receipt',{})
        if snapshot.get('final_refresh_verified') is not True or snapshot.get('current_equals_reference') is not True:
            raise ValueError('Reference/current terminal snapshot equality has not been verified')
        reader,simulation,core,sources=_source_reader_and_core(protocol)
        development=_development_inputs(reader)
        ledger['current_actor_rows']+=128
        actor_replay=assert_replay(bindings.policy.act(development['actor_observations']),development['saved_actor_actions'],'current actor')
        _current(bindings,development,ledger)
        replays=[actor_replay,assert_replay(development['q_current'],development['saved_q'],'current twin Q'),
                 assert_replay(development['u'],development['saved_u'],'frozen width radius')]
        times=np.asarray([row['time'] for row in development['metadata']],np.int64)
        physics=core.Physics(stall=True,stall_slope=1.5)
        development.update(one_step(development['states'],development['actions'],times,core.dynamics,physics,ledger))
        if not np.array_equal(development['next_obs'],simulation._observation(development['next_states'],99-times)):
            raise ValueError('Next observation differs from pinned evaluator encoding')
        development['snapshot_receipt']=deepcopy(snapshot)
        transition=bindings.fitting_transition
        fitting={'observations':_finite(transition.obs,(256,14),'fitting observations',np.float32),
                 'actions':_finite(transition.action,(256,4),'fitting actions',np.float32),
                 'rewards':_finite(transition.reward,(256,),'fitting rewards',np.float32),
                 'next_obs':_finite(transition.next_obs,(256,14),'fitting next observations',np.float32),
                 'done':_finite(transition.done,(256,),'fitting terminal masks',np.float32),
                 'metadata':[{'group_id':int(group),'row_index':int(row)} for group,row in zip(bindings.fitting_group_ids,bindings.fitting_row_indices)],
                 'snapshot_receipt':deepcopy(snapshot)}
        if len(fitting['metadata'])!=256 or np.any((fitting['done']!=0)&(fitting['done']!=1)) or np.any(np.abs(fitting['actions'])>1):
            raise ValueError('Invalid authenticated fitting transition bank')
        ledger['fitting_transition_rows_read']=256
        _current(bindings,fitting,ledger)
        fitting['scale_bank_q_role']='training_rows_not_heldout_calibration'
        _targets(bindings,development,DEV_SEEDS,ledger)
        _targets(bindings,fitting,FIT_SEEDS,ledger)
        after=bindings.assert_unchanged()
        if before!=after: raise ValueError('Frozen full checkpoint state changed')
        if ledger['transition_rows_attempted']!=1152 or ledger['transition_rows_completed']!=1152 or ledger['target_critic_rows']!=5632:
            raise ValueError('Operator audit budget accounting mismatch')
        sources_after,_=reader.verify()
        if sources_after!=sources['development_identity']: raise ValueError('Saved development inputs changed')
        ledger['status']='completed_bounded_operator_execution'
        return {'development':development,'fitting':fitting,'ledger':ledger,'checkpoint_receipt':bindings.receipt,
                'snapshot_before':before,'snapshot_after':after,'forward_replays':replays,'execution_sources':sources}
    except BaseException as error:
        ledger['status']='interrupted' if isinstance(error,KeyboardInterrupt) else 'blocked_or_failed'
        raise OperatorExecutionError(type(error).__name__+': '+str(error),ledger) from error
