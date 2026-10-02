# Closed-form alignment_K (cosine of Q1's and Q^pi's K-gradients) vs error size, per behaviour level; no data, no training.
import math, numpy as np
from scipy import linalg
A=np.array([[0.8,0.2,0.0],[0.0,0.8,0.2],[0.1,0.0,0.7]]); B=np.array([[0.5,0.0],[0.2,0.4],[0.0,0.5]])
Qc,Rc=np.eye(3),np.eye(2); noise=0.01*np.eye(3); init=0.25*np.eye(3); g=0.95; sb=0.2; L=50
D=np.array([[1.0,-1.0,0.5],[0.5,1.0,-1.0]])/math.sqrt(4.5)
def qq(K,G):
    M=A-B@K; Pi=np.vstack([np.eye(3),-K])
    P=linalg.solve_discrete_lyapunov(math.sqrt(g)*M.T,Pi.T@G@Pi); P=(P+P.T)/2
    F=np.hstack([A,B]); H=G+g*F.T@P@F; return (H+H.T)/2
def Sigma(K):
    S=init.copy(); acc=np.zeros((3,3)); M=A-B@K
    for t in range(L):
        acc+=S; S=M@S@M.T+sb**2*B@B.T+noise
    return acc/L
P=linalg.solve_discrete_are(math.sqrt(g)*A,math.sqrt(g)*B,Qc,Rc); Ks=np.linalg.solve(Rc+g*B.T@P@B,g*B.T@P@A)
Gr=-linalg.block_diag(Qc,Rc)
def devform(Kb):
    C=np.hstack([Kb,np.eye(2)]); return C.T@C
def kgrad(H,K,S):  # E[grad_a (z^T H z)|a=-Ks (-s)^T]
    return -2*(H[3:,:3]-H[3:,3:]@K)@S
def cos(x,y): return float(np.sum(x*y)/np.linalg.norm(x)/np.linalg.norm(y))
levels={'expert':(Ks,[Ks]),'medium':(0.1338*Ks,[0.1338*Ks]),'poor':(0*Ks,[0*Ks]),'mixed':(0.5*Ks,[Ks,0*Ks])}
for name,(Kb,comps) in levels.items():
    S=sum(Sigma(K) for K in comps)/len(comps)
    for off in (0.5,1.0,2.0):
        Kp=Kb+off*D; H=qq(Kp,Gr); Gt=kgrad(H,Kp,S)
        Haa=np.linalg.eigvalsh(H[3:,3:])
        out=[]
        for kap in (0.5,1,2,4,8,16):
            Ge=2*kap*off*D@S
            Hb=qq(Kp,kap*devform(Kb)); Gs=kgrad(Hb,Kp,S)
            out.append('k%g q1 %.2f sb %.2f'%(kap,cos(Gt+Ge,Gt),cos(Gt+Gs,Gt)))
        print(name,off,'Haa eig',Haa.round(2),'|',' ; '.join(out))
