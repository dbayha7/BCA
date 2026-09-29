"""Synthetic CPU parity for the selected host/BCA pair, including a frozen refresh."""
from pathlib import Path
import sys, json, hashlib, importlib
from dataclasses import replace
HERE=Path(__file__).resolve().parent
ROOT=Path('/users/dbayha/bca-standard-noiw-v1/cql-validation-recovery-v1/source')
host,version,method=sys.argv[1:]
sys.path.insert(0,str(ROOT))
from runtime.environment import setup
setup()
if version=='old':
    sys.path.insert(0,'/mnt/c/Users/David Bayha/Documents/Codex/2026-09-09/this-is-my-phd-research-im/work/minimal_host_bca_v1/original/algorithms')
    P=importlib.import_module(host)
else:P=importlib.import_module('algorithms.'+host+'_bca')
import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization
assert jax.default_backend()=='gpu'
g=np.random.default_rng(137)
arrays=[jnp.asarray(g.normal(size=s).astype(np.float32)) for s in ((48,3),(48,2),(48,),(48,3))]
arrays[1]=jnp.tanh(arrays[1]);arrays.append(jnp.zeros(48))
fingerprint=lambda t:hashlib.sha256(serialization.to_bytes(t)).hexdigest()
if host=='iql':
    H=P.H
    args=H.Args(seed=137,batch_size=8,num_updates=100,allow_off_config=True,posterior=H.P.PosteriorArgs(mode='full',reserve_size=16,draws=8,fit_size=16,cal_hidden_dim=8))
    data=H.C.Transition(*arrays)
    train=jax.tree.map(lambda x:x[:32],data);hold=jax.tree.map(lambda x:x[32:],data)
    native,rng,actor=H.initialize_agent(args,3,2,1.)
    if method=='host':
        carry=(rng,native,jnp.int32(0));step=H.BASE.make_train_step(args,native.actor.apply_fn,native.qf.apply_fn,native.vf.apply_fn,train)
    else:
        if version=='old':
            variants=(P.ScaleVariant('noiw',P.W.ScaleIWConfig(mode='off',beta=args.beta),True),)
            arms=(P.IWArm('host',-1,'off',args.beta),P.IWArm('bca',0,'full',args.beta))
        else:variants,arms=P.default_design(args.beta, fitting_mode="off")
        fitters=P.make_fitters(args,native,3,2,variants)
        carry=P.initialize_shared(args,native,rng,fitters,arms)
        step=P.make_shared_train_step(args,train,fitters,variants,arms)
else:
    N=P.BASE;kwargs=dict(seed=137,batch_size=8,allow_off_config=True)
    if host=='rebrac':kwargs.update(hidden_dim=8,actor_n_hiddens=2,critic_n_hiddens=2)
    args=N.Args(**kwargs)
    data=N.C.TransitionNA(*arrays,arrays[1]) if host=='rebrac' else N.C.Transition(*arrays)
    train=jax.tree.map(lambda x:x[:32],data);hold=jax.tree.map(lambda x:x[32:],data)
    if method=='host':cfg=P.Config('native_pool' if version=='old' else 'host')
    else:
        common=dict(posterior=P.PosteriorConfig(alpha=.2,credibility=.8,draws=8),blend=.5)
        if host=='cql':
            common['iw']=P.ScaleIWConfig('off',.25,.05,32)
            if version=='old':common['width_objective']='matched'
        else:
            common['iw']=P.AffinityIWConfig()
            if version=='old' and host=='td3_bc':common['width_objective']='matched'
        cfg=P.Config(('posterior_matched' if host=='rebrac' else 'posterior') if version=='old' else 'bca',**common)
    rng,state,models=P.initialize(args,cfg,train) if host=='rebrac' else P.initialize(args,cfg,3,2)
    carry=(rng,state,jnp.int32(0));step=P.make_train_step(args,cfg,models,train)
initial=fingerprint(carry)
scan=jax.jit(lambda c:jax.lax.scan(step,c,None,length=2))
carry,metrics=scan(carry);jax.block_until_ready(carry)
before=dict(state=fingerprint(carry),metrics=fingerprint(metrics))
if method=='bca':
    if host=='iql':
        e=carry.extras[0]
        ref,record=H.P.refresh(args.posterior,fitters[0],e.calibration,carry.nuisance,train,hold,carry.rng,2,discount=args.discount)
        carry=carry._replace(extras=(e._replace(posterior=ref),))
    else:
        kw=dict(training_ids=np.arange(32),heldout_ids=np.arange(32,48)) if host in ('td3_bc','rebrac') else {}
        refreshed,metrics_refresh=P.refresh(args,cfg,models,carry[1],train,hold,jax.random.PRNGKey(910),**kw)
        P.require_valid(metrics_refresh);carry=(carry[0],refreshed,carry[2])
    refresh=fingerprint(carry)
    carry,metrics=scan(carry);jax.block_until_ready(carry)
else:refresh=None
result=dict(host=host,version=version,method=method,initial=initial,before=before,refresh=refresh,final=fingerprint(carry),metrics=fingerprint(metrics),synthetic_updates=4 if method=='bca' else 2,simulator_steps=0)
if version=='new':
    if host=='iql':
        from runtime.iql_checkpoint import checkpoint_counts
        counts=checkpoint_counts(serialization.msgpack_restore(serialization.to_bytes(carry)),'shared' if method=='bca' else 'native',4 if method=='bca' else 2,np.asarray(carry.actors.step).tolist() if method=='bca' else [2],[int(e.calibration.calibrator.step) for e in carry.extras] if method=='bca' else [])
    else:
        from runtime.validation import checkpoint_counts
        counts=checkpoint_counts(serialization.msgpack_restore(serialization.to_bytes(dict(state=carry[1],training_rng=carry[0],step=carry[2]))),'td3' if host=='td3_bc' else host,int(carry[2]),getattr(args,'policy_freq',2))
    result['decoded_checkpoint_checked']=True
(HERE/f'{host}_{version}_{method}.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
