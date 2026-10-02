exec(open('lq_align.py').read().split("levels=")[0])
for name,K in (('expert',Ks),('medium',0.1338*Ks),('poor',0*Ks)):
    S=Sigma(K); print(name,'tr E[ss^T] over 50-step episodes %.4f'%np.trace(S), 'eig',np.linalg.eigvalsh(S).round(4))
import inspect
print('torch_adam eps check skipped')
