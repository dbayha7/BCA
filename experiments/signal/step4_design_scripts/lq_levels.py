# Closed-form J of linear gains in the step-3 LQ system (copies lq_harness formulas; no data, no training).
import math, numpy as np
from scipy import linalg
A=np.array([[0.8,0.2,0.0],[0.0,0.8,0.2],[0.1,0.0,0.7]]); B=np.array([[0.5,0.0],[0.2,0.4],[0.0,0.5]])
Qc,Rc=np.eye(3),np.eye(2); noise=0.01*np.eye(3); init=0.25*np.eye(3); g=0.95
D=np.array([[1.0,-1.0,0.5],[0.5,1.0,-1.0]])/math.sqrt(4.5)
def J(K):
    M=A-B@K
    if not math.sqrt(g)*np.max(np.abs(np.linalg.eigvals(M)))<1: return -math.inf
    Pi=np.vstack([np.eye(3),-K]); G=-linalg.block_diag(Qc,Rc)
    P=linalg.solve_discrete_lyapunov(math.sqrt(g)*M.T,Pi.T@G@Pi); P=(P+P.T)/2
    h=g*float(np.trace(P@noise))/(1-g); return float(np.trace(P@init))+h
P=linalg.solve_discrete_are(math.sqrt(g)*A,math.sqrt(g)*B,Qc,Rc)
Ks=np.linalg.solve(Rc+g*B.T@P@B,g*B.T@P@A)
J1,J0=J(Ks),J(0*Ks)
print('K*',Ks.round(3),'J*',J1,'J0',J0)
for gg in [0,0.1,0.2,0.25,0.3,0.4,0.5,0.6,0.75,1.0]:
    q=(J(gg*Ks)-J0)/(J1-J0); print('gain %.2f J %.4f q %.3f'%(gg,J(gg*Ks),q), ' offsets J:', [round(J(gg*Ks+o*D),3) for o in (0.5,1,2)])
# bisection for q targets
for qt in (0.25,0.5,0.75,0.9):
    lo,hi=0,1
    for _ in range(60):
        m=(lo+hi)/2
        if (J(m*Ks)-J0)/(J1-J0)<qt: lo=m
        else: hi=m
    print('q=%.2f -> gain %.4f'%(qt,hi))
