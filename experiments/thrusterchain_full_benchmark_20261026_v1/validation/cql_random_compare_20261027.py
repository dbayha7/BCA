"""Check CQL random primitives under the existing eager and jit boundaries."""
import json
import jax
import numpy as np
def draw(r):
    a,b,c,d,e,f=jax.random.split(r,6)
    return (jax.random.normal(a,(7,4)),jax.random.normal(b,(7,4)),jax.random.normal(c,(7,4)),
            jax.random.uniform(d,(7,10,4),minval=-1.,maxval=1.),
            jax.random.normal(e,(7,10,4)),jax.random.normal(f,(7,10,4)))
key=jax.random.PRNGKey(20)
eager=draw(key); compiled=jax.jit(draw)(key)
out=[]
for i,(a,b) in enumerate(zip(eager,compiled)):
    x,y=np.asarray(a),np.asarray(b)
    out.append({'index':i,'equal':bool(np.array_equal(x,y)),
                'max_abs':float(np.max(np.abs(x-y))),'mismatched':int(np.count_nonzero(x!=y))})
print(json.dumps({'device':str(jax.devices()[0]),'arrays':out},indent=2))
