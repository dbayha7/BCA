import math
# P(c + A cos U >= 0.4), c~U[0.1,0.7], A~U[0.1,0.4], U~U[0,2pi): midpoint quadrature
N=4000; M=2000; tot=0.0
for i in range(M):
    A=0.1+0.3*(i+0.5)/M
    for j in range(N):
        u=2*math.pi*(j+0.5)/N
        p=(0.7-(0.4-A*math.cos(u)))/0.6
        tot+=min(1.0,max(0.0,p))
P=tot/(M*N)
print('P_high',P)
print('band prob requested',0.1/2)
print('expected hole rows train',76800*0.05*P)
print('expected retained rows hole cond',76800-76800*0.05*P)
print('expected hole rows among 128 old calib rows',128*0.05*P)
# episodes that never reach high load: c + A < 0.4
q=0.0
for i in range(M):
    A=0.1+0.3*(i+0.5)/M
    q+=min(1,max(0,(0.4-A-0.1)/0.6))
q/=M
print('P(episode never high load)',q, 'expected episodes of 768', 768*q)
# episodes always high load: c - A >= 0.4
r=0.0
for i in range(M):
    A=0.1+0.3*(i+0.5)/M
    r+=min(1,max(0,(0.7-(0.4+A))/0.6))
r/=M
print('P(episode always high)',r, 768*r)
for n in (32,64,128):
    print(n,'expected band rows',n*0.05,'P(zero in band)',0.95**n,'P(<=1)',0.95**n+n*0.05*0.95**(n-1))
print('4D ball',(0.1/2)**4, 64*(0.1/2)**4)
