import json, numpy as np, math, sys
from math import erf, sqrt
d=json.load(open(sys.argv[1]))
A=np.array(d['A']); B=np.array(d['B']); A10=A[:10,:10]; B10=B[:10,:]
k=float(np.float32(0.8))
Q=np.zeros((13,13)); Q[:10,:10]=np.eye(10); R=0.3*np.eye(4); gam=0.98
P=np.zeros((13,13)); Ks=[]
for t in range(99,-1,-1):
    S=R+gam*B.T@P@B
    K=np.linalg.solve(S, gam*B.T@P@A)
    P=Q+gam*A.T@P@A - gam*A.T@P@B@K
    Ks.append(K)
Ks=Ks[::-1]
def Phi(z): return 0.5*(1+erf(z/sqrt(2)))
# float-state contribution to the unconstrained LQ a0: a0* = -K[0,:10] s - K[0,10:] (c,w1,w2)
# (a) t=0: s ~ 0.5 N(0,I)
# (b) late t: s under the uniform recorder: stationary covariance of s_t = A10 s + 0.3 B1 a + 0.3 v g_t, a uniform on cube (var 1/3), ignore deterministic push mean part
Cov_a=np.eye(4)/3
Sig=np.zeros((10,10))
for _ in range(5000):
    Sig=A10@Sig@A10.T + B10@Cov_a@B10.T
print('stationary float covariance under the uniform recorder: trace', Sig.trace().round(3), 'sd per coord', np.sqrt(np.diag(Sig)).round(3))
# also transient: start 0.25 I, propagate t steps
def cov_at(t):
    S=0.25*np.eye(10)
    for _ in range(t): S=A10@S@A10.T + B10@Cov_a@B10.T
    return S
for t in (0,10,25,50,75,90):
    K=Ks[t]; ks=K[0,:10]; gc,gw,gw2=-K[0,10],-K[0,11],-K[0,12]
    sd=math.sqrt(ks@cov_at(t)@ks)
    row=[]
    for g in (0.4,0.6,0.8,1.0):
        mu=gc*g  # take c=g, w1=0 (w2 unknown; ignore)
        p_at_or_beyond=Phi((-k-mu)/sd)  # unconstrained a0* <= -k -> constrained optimum pinned at -k
        p_band=Phi((-k+0.05-mu)/sd)-Phi((-k-0.05-mu)/sd)
        row.append((g, round(mu,3), round(p_at_or_beyond,3), round(p_band,3)))
    print(f't={t}: sd of float-driven a0* = {sd:.3f}; per g: (g, mean a0*, P[a0*<=-k], P[a0* in band h=0.05]) =', row)
# the LQ gains show how much a0* depends on w2 as well (wave phase), so g alone does not fix the counter-command
K0=Ks[0]
print('t=0: a0* = %.3f c %+.3f w1 %+.3f w2 + float terms'%(-K0[0,10],-K0[0,11],-K0[0,12]))
# how much of the throttle is needed per unit g if ALL four thrusters are LQ-optimal (floats at rest)?
print('t=0 all-thruster LQ command per unit c (floats at rest):', (-K0[:,10]).round(3), ' per unit w1:', (-K0[:,11]).round(3))
# two-step: value of being at a0=-k vs -0.7375-equivalent? skip. Instead: LQ value loss from forcing a0 away from optimum is quadratic with curvature (R+gam B'PB)[0,0]
S0=R+gam*B.T@P@B
print('actor-side curvature of the Q in a0 (R + gam B^T P B)[0,0] at t=0: %.4f ; cross terms row0: %s'%(S0[0,0], S0[0].round(3)))
print('=> changing a0 by 0.05 away from the optimum costs about %.4f return units; by 0.1 about %.4f'%(S0[0,0]*0.05**2, S0[0,0]*0.1**2))
