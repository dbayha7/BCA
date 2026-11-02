"""Experimental CQL adapter hoisting native Polyak cond branch closures.

The native CQL update, Adam optimizer, random draws, host-side minibatch
preprocessing and commit-on-finite behavior remain unchanged. This separate
candidate is not wired into the study launcher. GPU validation is pending.
"""
import copy
import hashlib
import importlib.metadata as metadata
import inspect
import os
from pathlib import Path
import subprocess
import sys
import jax
import jax.numpy as jnp
import numpy as np
import optax
from algorithms import cql as host
from runtime.networks import Transition
from thrusterchain.datasets import assert_training_allowed
from host_integration import td3bc_adapter as helpers
from host_integration.cql_adapter import CQLAdapter

VARIANT = 'eager-cql-cached-polyak-v1'

def source_identity():
    return {'variant': VARIANT,
            'candidate_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'native_update_sha256': hashlib.sha256(inspect.getsource(host.cql_update).encode()).hexdigest()}

def runtime_identity():
    names=('XLA_FLAGS','JAX_PLATFORMS','JAX_PLATFORM_NAME','JAX_DEFAULT_MATMUL_PRECISION',
           'JAX_DISABLE_JIT','JAX_ENABLE_X64','JAX_DEFAULT_PRNG_IMPL','JAX_THREEFRY_PARTITIONABLE',
           'CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES','NVIDIA_TF32_OVERRIDE',
           'CUBLAS_WORKSPACE_CONFIG','XLA_PYTHON_CLIENT_PREALLOCATE','XLA_PYTHON_CLIENT_MEM_FRACTION',
           'JAX_COMPILATION_CACHE_DIR')
    packages=('jax','jaxlib','jax-cuda12-plugin','jax-cuda12-pjrt','numpy','scipy','flax','optax',
              'chex','distrax','tensorflow-probability','gymnasium','ml-dtypes',
              'nvidia-cublas-cu12','nvidia-cuda-runtime-cu12','nvidia-cuda-nvcc-cu12',
              'nvidia-cudnn-cu12','nvidia-cufft-cu12','nvidia-cusolver-cu12','nvidia-cusparse-cu12',
              'nvidia-nvjitlink-cu12')
    versions={}
    for name in packages:
        try: versions[name]=metadata.version(name)
        except metadata.PackageNotFoundError: versions[name]=None
    backend=jax.default_backend()
    gpu_inventory=None
    if backend=='gpu':
        gpu_inventory=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,compute_cap',
                                      '--format=csv,noheader'],check=True,capture_output=True,text=True).stdout.strip().splitlines()
        if not gpu_inventory: raise RuntimeError('nvidia-smi did not report a GPU')
    return {'environment':{name:os.environ.get(name) for name in names},
            'jax_config':{'disable_jit':bool(jax.config.jax_disable_jit),
              'enable_x64':bool(jax.config.jax_enable_x64),
              'default_matmul_precision':str(jax.config.jax_default_matmul_precision),
              'default_prng_impl':str(jax.config.jax_default_prng_impl),
              'threefry_partitionable':bool(jax.config.jax_threefry_partitionable)},
            'backend':backend,'devices':[str(d) for d in jax.devices()], 'gpu_inventory':gpu_inventory,
            'device_kinds':[d.device_kind for d in jax.devices()],
            'platform_versions':[d.client.platform_version for d in jax.devices()],
            'versions':versions,'python':sys.version}

def execution_identity():
    return {'source':source_identity(),'runtime':runtime_identity()}

def _make_update(model):
    rate=model._args.soft_target_update_rate
    def soft(src,tgt):
        return tgt.replace(step=tgt.step+1,params=optax.incremental_update(src.params,tgt.params,rate))
    def cached_do(state):
        return state._replace(critic1_target=soft(state.critic1,state.critic1_target),
                              critic2_target=soft(state.critic2,state.critic2_target))
    def cached_skip(state): return state
    source=inspect.getsource(host.cql_update)
    if source.count('    def _soft(src, tgt):')!=1: raise RuntimeError('unexpected native CQL source')
    i=source.index('    def _soft(src, tgt):'); j=source.index('    agent_state = jax.lax.cond(',i)
    expected='''    def _soft(src, tgt):
        return tgt.replace(
            step=tgt.step + 1,
            params=optax.incremental_update(
                src.params, tgt.params, args.soft_target_update_rate
            ),
        )

    def _do_polyak(state):
        return state._replace(
            critic1_target=_soft(state.critic1, state.critic1_target),
            critic2_target=_soft(state.critic2, state.critic2_target),
        )

'''
    if source[i:j]!=expected: raise RuntimeError('native Polyak body changed')
    source=source[:i]+source[j:]
    old='it % args.target_update_period == 0, _do_polyak, lambda s: s, agent_state'
    new='it % args.target_update_period == 0, _cached_do, _cached_skip, agent_state'
    if source.count(old)!=1: raise RuntimeError('unexpected native CQL target update')
    source=source.replace(old,new)
    namespace=dict(host.__dict__); namespace.update(_cached_do=cached_do,_cached_skip=cached_skip)
    exec(compile(source,'<cql-cached-polyak-v1>','exec'),namespace)
    return namespace['cql_update']

class CachedCQLAdapter(CQLAdapter):
    def initialize(self,train_batch,seed,learning_settings=None):
        super().initialize(train_batch,seed,learning_settings)
        self._metadata=copy.deepcopy(self._metadata)
        self._metadata['adapter']['execution_identity']=execution_identity()
        self._candidate_update=_make_update(self)

    def restore(self,checkpoint):
        identity=checkpoint.metadata.get('adapter',{}).get('execution_identity')
        if identity!=execution_identity():
            raise ValueError('CQL candidate source or runtime identity differs from checkpoint')
        super().restore(checkpoint)
        self._candidate_update=_make_update(self)

    def update(self,batch):
        assert_training_allowed('update CQL candidate')
        self._require()
        obs,nxt,action,reward=helpers._arrays(batch)
        if obs.shape[-1]!=self._dim or np.any(np.asarray(batch.origin)!=0):
            raise ValueError('strict offline CQL requires matching offline transition batches')
        native=Transition(jnp.asarray((obs-self._mean)/self._scale),jnp.asarray(action),
            jnp.asarray(reward),jnp.asarray((nxt-self._mean)/self._scale),
            jnp.asarray(batch.terminated,dtype=jnp.float32))
        next_rng,update_rng=jax.random.split(self._rng)
        state,metrics=self._candidate_update(self._args,self._actor.apply,self._critic1.apply,
            self._critic2.apply,self._state,native,int(self._state.critic1.step)+1,update_rng,1.,-4.)
        if not helpers._finite_tree((state,metrics,next_rng)):
            raise FloatingPointError('nonfinite update; previous state and RNG retained')
        self._state,self._rng=state,next_rng
        return {key:float(value) for key,value in metrics.items()}
