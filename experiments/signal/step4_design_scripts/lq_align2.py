# Common-start variant: pi = 0.5 K* + offset D for every behaviour level. Closed form; no data, no training.
exec(open('lq_align.py').read().split("levels=")[0])
def J(K):
    M=A-B@K
    if not math.sqrt(g)*np.max(np.abs(np.linalg.eigvals(M)))<1: return -math.inf
    Pi=np.vstack([np.eye(3),-K])
    P=linalg.solve_discrete_lyapunov(math.sqrt(g)*M.T,Pi.T@Gr@Pi); P=(P+P.T)/2
    return float(np.trace(P@init))+g*float(np.trace(P@noise))/(1-g)
J1,J0=J(Ks),J(0*Ks)
levels={'expert':(Ks,[Ks]),'medium':(0.1338*Ks,[0.1338*Ks]),'poor':(0*Ks,[0*Ks]),'mixed':(0.5*Ks,[Ks,0*Ks])}
for off in (0.5,1.0):
    Kp=0.5*Ks+off*D; H=qq(Kp,Gr)
    print('start offset',off,'J_pi %.3f q_pi %.3f'%(J(Kp),(J(Kp)-J0)/(J1-J0)),'Haa',np.linalg.eigvalsh(H[3:,3:]).round(2))
    for name,(Kb,comps) in levels.items():
        S=sum(Sigma(K) for K in comps)/len(comps); Gt=kgrad(H,Kp,S)
        out=[]
        for kap in (0.25,0.5,1,1.5,2,3):
            Ge=-2*kap*((Kb-Kp))@S  # grad_a kappa||a+Kb s||^2 at a=-Kp s is 2kappa(Kb-Kp)s; times (-s)^T
            Hb=qq(Kp,kap*devform(Kb)); Gs=kgrad(Hb,Kp,S)
            out.append('k%g q1 %.2f sb %.2f'%(kap,cos(Gt+Ge,Gt),cos(Gt+Gs,Gt)))
        print('  ',name,' ; '.join(out), '| cos(Kb-Kp, grad J dir)=%.2f'%cos(Kb-Kp, -Gt))
