from lq_common import *
Se,Sp=Sigma(Ks),Sigma(0*Ks)
Kbc=Ks@Se@np.linalg.inv(Se+Sp)
print('K_BC mixed',Kbc.round(3).tolist(),'proj on K*', round(float(np.sum(Kbc*Ks)/np.sum(Ks*Ks)),3))
print('J(K_BC) %.4f q %.3f'%(J(Kbc),(J(Kbc)-J0)/(J1-J0)), '| J(0.5K*) %.4f q %.3f'%(J(0.5*Ks),(J(0.5*Ks)-J0)/(J1-J0)), '| medium J %.4f'%J(0.1338*Ks))
Kp=0.5*Ks+0.5*D; H=qq(Kp,Gr)
for name,Kb,S in (('expert',Ks,Se),('mixed(K_BC)',Kbc,(Se+Sp)/2),('medium',0.1338*Ks,Sigma(0.1338*Ks)),('poor',0*Ks,Sp)):
    Gt=kgrad(H,Kp,S); print(name,'cos(Kb-Kp, Q^pi ascent)=%.2f'%cos(Kb-Kp,Gt))
