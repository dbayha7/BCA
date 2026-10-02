# Shared closed-form helpers copied from experiments/signal/lq_harness.py formulas (no data, no training).
import math, numpy as np
from scipy import linalg
A=np.array([[0.8,0.2,0.0],[0.0,0.8,0.2],[0.1,0.0,0.7]]); B=np.array([[0.5,0.0],[0.2,0.4],[0.0,0.5]])
Qc,Rc=np.eye(3),np.eye(2); noise=0.01*np.eye(3); init=0.25*np.eye(3); g=0.95; sb=0.2; L=50
D=np.array([[1.0,-1.0,0.5],[0.5,1.0,-1.0]])/math.sqrt(4.5)
Gr=-linalg.block_diag(Qc,Rc)
def qq(K,G):
    M=A-B@K; Pi=np.vstack([np.eye(3),-K])
    P=linalg.solve_discrete_lyapunov(math.sqrt(g)*M.T,Pi.T@G@Pi); P=(P+P.T)/2
    F=np.hstack([A,B]); H=G+g*F.T@P@F; return (H+H.T)/2
def J(K):
    M=A-B@K
    if not math.sqrt(g)*np.max(np.abs(np.linalg.eigvals(M)))<1: return -math.inf
    Pi=np.vstack([np.eye(3),-K]); P=linalg.solve_discrete_lyapunov(math.sqrt(g)*M.T,Pi.T@Gr@Pi); P=(P+P.T)/2
    return float(np.trace(P@init))+g*float(np.trace(P@noise))/(1-g)
def Sigma(K):
    S=init.copy(); acc=np.zeros((3,3)); M=A-B@K
    for t in range(L):
        acc+=S; S=M@S@M.T+sb**2*B@B.T+noise
    return acc/L
P_=linalg.solve_discrete_are(math.sqrt(g)*A,math.sqrt(g)*B,Qc,Rc); Ks=np.linalg.solve(Rc+g*B.T@P_@B,g*B.T@P_@A)
def kgrad(H,K,S): return -2*(H[3:,:3]-H[3:,3:]@K)@S
def cos(x,y): return float(np.sum(x*y)/np.linalg.norm(x)/np.linalg.norm(y))
J1,J0=J(Ks),J(0*Ks)
