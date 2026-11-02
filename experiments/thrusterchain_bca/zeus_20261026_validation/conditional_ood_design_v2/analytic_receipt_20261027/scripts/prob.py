import numpy as np, math
# start law: c~U[0.1,0.7], A~U[0.1,0.4], phase~U[0,2pi); w1=A cos(phase). g=c+w1
# P(g>=thr) by quadrature over (c,A,phase)
def P_g_ge(thr, clo=0.1, chi=0.7, alo=0.1, ahi=0.4, n=2001):
    cs=np.linspace(clo,chi,n); As=np.linspace(alo,ahi,n)
    # for fixed c,A: P(cos phase >= (thr-c)/A) = arccos(clip((thr-c)/A,-1,1))/pi
    C,Aa=np.meshgrid(cs,As,indexing='ij')
    z=np.clip((thr-C)/Aa,-1,1)
    p=np.arccos(z)/math.pi
    # trapezoid in both
    return np.trapz(np.trapz(p,As,axis=1),cs)/((chi-clo)*(ahi-alo))
for thr in (0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.9375,1.0,1.0625):
    print('recorded P(g>=%.4f) = %.4f'%(thr,P_g_ge(thr)))
print('recorded P(0.9375<=g<=1.0625) = %.5f'%(P_g_ge(0.9375)-P_g_ge(1.0625)))
print('recorded P(0.85<=g<=1.1) = %.5f'%(P_g_ge(0.85)))
print('strong current P(g>=0.4) = %.4f'%P_g_ge(0.4,0.7,1.1))
print('strong current P(g>=0.9375) = %.4f'%P_g_ge(0.9375,0.7,1.1))
# episodes always high / always low
# P(c - A >= 0.4) and P(c + A <= 0.4)
cs=np.linspace(0.1,0.7,4001); As=np.linspace(0.1,0.4,4001); C,Aa=np.meshgrid(cs,As,indexing='ij')
dA=(0.6*0.3)
print('P(episode always g>=0.4) = %.4f'%(np.trapz(np.trapz((C-Aa>=0.4).astype(float),As,axis=1),cs)/dA))
print('P(episode always g<0.4) = %.4f'%(np.trapz(np.trapz((C+Aa<=0.4).astype(float),As,axis=1),cs)/dA))
# action band probability under uniform a0 on [-1,1]
k=float(np.float32(0.8))
for h in (0.025,0.05,0.10):
    lo,hi=-k-h,-k+h
    p=(min(hi,1)-max(lo,-1))/2
    print('h=%g band [%.12f, %.12f] P=%.4f; expected rows/76800: %.1f ; per 100-step episode: %.2f ; joint with g>=0.4 (x0.5): rows %.1f, per episode %.2f'%(h,lo,hi,p,p*76800,p*100,p*0.5*76800,p*0.5*100))
    # float32 boundary: nearest float32 to the edges and whether |float64(a0)+k| == h exactly
    for e in (lo,hi):
        f=np.float32(e)
        for cand in (np.nextafter(f,np.float32(-2)), f, np.nextafter(f,np.float32(2))):
            val=abs(float(cand)+k)
            print('   edge %.12f: float32 %.12f -> |a0+k|=%.17g  <=h? %s  (diff from h %.3g)'%(e,float(cand),val,val<=h,val-h))
print('ulp of float32 near 0.8:', np.spacing(np.float32(0.8)))
print('expected rows within one float32 ulp of either band edge: %.2e'%(76800*0.5*2*float(np.spacing(np.float32(0.8)))))
# expected rows within float32 rounding of the g threshold: density of g at 0.4 x 2*ulp(0.4) x 76800
dens=(P_g_ge(0.3999)-P_g_ge(0.4001))/0.0002
print('density of g at 0.4 ~ %.3f ; ulp float32(0.4)=%.3g ; expected ambiguous rows ~ %.2e'%(dens,np.spacing(np.float32(0.4)),dens*2*float(np.spacing(np.float32(0.4)))*76800))
print('threshold 0.4 as float32:', repr(float(np.float32(0.4))))
# global cutoff for comparison
print('global cutoff a0<=0.5 expected removed = 0.25*76800 =', 0.25*76800, ' observed 19048')
# per-episode hole-row count distribution: Binomial-ish? conditional on episode's share of high-load steps q_e, rows~Bin(100, q_e*h)
# P(zero hole rows in an episode) for h=0.05: E[(1-0.05 q)^100] approx; q in {0 w.p. 1/9, 1 w.p. 1/9, else ~ between}
import itertools
# simulate the high-load share analytically: q(c,A) = fraction of t in 0..99 with c + A cos(phase+0.27 t) >= 0.4 averaged over phase ~ arccos(...)/pi (same marginal), so use q = P(g>=0.4 | c,A)
Cq=np.arccos(np.clip((0.4-C)/Aa,-1,1))/math.pi
for h in (0.025,0.05,0.10):
    pz=np.trapz(np.trapz((1-h*Cq)**100,As,axis=1),cs)/dA
    print('h=%g: approx P(episode has zero hole rows) = %.3f (treating steps as independent given (c,A))'%(h,pz))
