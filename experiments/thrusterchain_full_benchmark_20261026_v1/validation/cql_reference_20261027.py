"""Independent NumPy loss and finite-difference check; see frozen protocol."""
from dataclasses import replace
import json
import math
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from algorithms import cql
from runtime.networks import Transition

def actor(p, obs):
    return obs @ p['w'] + p['b'], jnp.broadcast_to(p['log_std'], obs.shape[:-1] + (2,))

def critic(p, obs, act):
    if act.ndim == obs.ndim + 1:
        obs = jnp.broadcast_to(obs[..., None, :], act.shape[:-1] + (2,))
    return (obs @ p['s']) + (act @ p['a']) + p['b']

def ts(params):
    return TrainState.create(apply_fn=None, params=jax.tree.map(jnp.asarray, params), tx=optax.sgd(.01))

obs = np.array([[.3,-.8],[-.2,.7]], np.float32)
nxt = np.array([[.5,.4],[-.6,-.1]], np.float32)
act = np.array([[.1,-.4],[.7,.2]], np.float32)
reward = np.array([.4,-.7],np.float32)
done = np.array([0,1],np.float32)
pa = {'w':np.array([[.13,-.17],[.21,.08]],np.float32), 'b':np.array([.02,-.03],np.float32), 'log_std':np.array([-.6,-.3],np.float32)}
p1 = {'s':np.array([.41,-.23],np.float32), 'a':np.array([.31,-.37],np.float32), 'b':np.array(.12,np.float32)}
p2 = {'s':np.array([-.16,.22],np.float32), 'a':np.array([-.19,.26],np.float32), 'b':np.array(-.08,np.float32)}
t1 = {'s':np.array([.27,-.13],np.float32), 'a':np.array([.21,-.18],np.float32), 'b':np.array(.04,np.float32)}
t2 = {'s':np.array([-.06,.12],np.float32), 'a':np.array([-.29,.16],np.float32), 'b':np.array(-.04,np.float32)}
state = cql.AgentTrainState(ts(pa), ts(p1), ts(p2), ts(t1), ts(t2), ts({'constant':np.array(0.,np.float32)}), ts({'constant':np.array(1.,np.float32)}))
args = replace(cql.Args(), cql_n_actions=3, discount=.98, bc_steps=0, cql_reward_scale=1., cql_reward_bias=0.)
batch = Transition(*map(jnp.asarray, (obs,act,reward,nxt,done)))
rng = jax.random.PRNGKey(319)
keys = jax.random.split(rng, 6)
epspi = np.asarray(jax.random.normal(keys[0], (2,2)),dtype=np.float64)
epsnext = np.asarray(jax.random.normal(keys[2], (2,2)),dtype=np.float64)
u = np.asarray(jax.random.uniform(keys[3],(2,3,2),minval=-1.,maxval=1.),dtype=np.float64)
epscur = np.asarray(jax.random.normal(keys[4],(2,3,2)),dtype=np.float64)
epsnxt = np.asarray(jax.random.normal(keys[5],(2,3,2)),dtype=np.float64)
obs,nxt,act,reward,done = [np.asarray(v,dtype=np.float64) for v in (obs,nxt,act,reward,done)]
pa,p1,p2,t1,t2 = [{k:np.asarray(v,dtype=np.float64) for k,v in p.items()} for p in (pa,p1,p2,t1,t2)]

def draw(p, x, eps):
    mean = x @ p['w'] + p['b']
    ls = np.clip(p['log_std'], -20, 2)
    std = np.exp(ls)
    raw = mean + eps*std
    logbase = -np.square(raw-mean)/(2*np.square(std))-np.log(std)-.5*np.log(2*np.pi)
    logdet = 2*(np.log(2)-raw-np.logaddexp(0.,-2*raw))
    return np.tanh(raw),np.sum(logbase-logdet,axis=-1)

def q(p,x,a):
    if a.ndim == x.ndim+1: x=x[:,None,:]
    return x @ p['s'] + a @ p['a'] + p['b']

def stable_lse(x):
    m=x.max(axis=-1)
    return m+np.log(np.exp(x-m[:,None]).sum(axis=-1))

next_a,_ = draw(pa,nxt,epsnext)
target = reward+(1.-done)*.98*np.minimum(q(t1,nxt,next_a),q(t2,nxt,next_a))
cur,cur_lp=draw(pa,obs[:,None,:],epscur)
next_candidates,next_lp=draw(pa,nxt[:,None,:],epsnxt)

def critic_loss(p):
    data=q(p,obs,act)
    td=np.square(data-target).mean()
    cand=np.concatenate([q(p,obs,u)-np.log(.5**2),q(p,obs,next_candidates)-next_lp,q(p,obs,cur)-cur_lp],axis=1)
    gap=(stable_lse(cand)-data).mean()
    return td+10*gap, td, 10*gap

def actor_loss(p):
    a,lp=draw(p,obs,epspi)
    return (lp-np.minimum(q(p1,obs,a),q(p2,obs,a))).mean()

def grad(fun,p):
    out={}
    for k,v in p.items():
        g=np.zeros_like(v)
        for idx in np.ndindex(v.shape):
            pplus={kk:x.copy() for kk,x in p.items()}; pminus={kk:x.copy() for kk,x in p.items()}
            pplus[k][idx]+=1e-5; pminus[k][idx]-=1e-5
            g[idx]=(fun(pplus)-fun(pminus))/(2e-5)
        out[k]=g
    return out

ga=grad(actor_loss,pa)
g1=grad(lambda p:critic_loss(p)[0],p1)
g2=grad(lambda p:critic_loss(p)[0],p2)

def step(carry,_):
    s,m=cql.cql_update(args,actor,critic,critic,carry,batch,1,rng,1.,-2.)
    return s,m

new,metrics=jax.jit(lambda s:jax.lax.scan(step,s,jnp.arange(1)))(state)
metrics=jax.tree.map(lambda x:np.asarray(x)[0],metrics)
checks={}
def check(name,actual,expected,atol,rtol):
    a,e=np.asarray(actual,dtype=np.float64),np.asarray(expected,dtype=np.float64)
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(e)), name
    checks[name]={'max_abs':float(np.max(np.abs(a-e))),'atol':atol,'rtol':rtol}
    np.testing.assert_allclose(a,e,atol=atol,rtol=rtol,err_msg=name)

l1,td1,gap1=critic_loss(p1); l2,td2,gap2=critic_loss(p2)
for name,expected in {'policy_loss':actor_loss(pa),'qf1_loss':td1,'qf2_loss':td2,
                      'cql_min_qf1_loss':gap1,'cql_min_qf2_loss':gap2,
                      'qf_loss':l1+l2,'average_target_q':np.minimum(q(t1,nxt,next_a),q(t2,nxt,next_a)).mean()}.items():
    check(name,metrics[name],expected,2e-5,2e-5)
for name,old,gradients in [('actor',pa,ga),('critic1',p1,g1),('critic2',p2,g2)]:
    params=getattr(new,name).params
    for k in old:
        check(name+'.gradient.'+k,(old[k]-np.asarray(params[k]))/.01,gradients[k],2e-4,5e-4)
for name,old,target_old in [('critic1',p1,t1),('critic2',p2,t2)]:
    for k in old:
        expected=.995*target_old[k]+.005*np.asarray(getattr(new,name).params[k])
        check(name+'.target.'+k,getattr(new,name+'_target').params[k],expected,2e-4,5e-4)
for name in ('actor','critic1','critic2','critic1_target','critic2_target','log_alpha','log_alpha_prime'):
    assert int(getattr(new,name).step)==1,name
print(json.dumps({'jax':jax.__version__,'devices':list(map(str,jax.devices())), 'checks':checks, 'target':target.tolist(), 'status':'pass'},indent=2))
