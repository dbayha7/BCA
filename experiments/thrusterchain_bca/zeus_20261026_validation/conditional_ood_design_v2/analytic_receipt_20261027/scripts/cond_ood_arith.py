import json, math, sys
import numpy as np

env = json.load(open(r"C:/Users/David/dev/BCA/docs/thrusterchain/site/data/environment.json"))
A = np.array(env["A"]); B = np.array(env["B"])
k = env["physics"]["knee"]; slope = env["physics"]["stall_slope"]
rec = json.load(open(r"C:/Users/David/dev/BCA/experiments/thrusterchain_bca/zeus_20261026_validation/bca_one_sided_v1/run/data_receipt.json"))
std = np.array(rec["normalization"]["std"]); mean = np.array(rec["normalization"]["mean"])
print("observation bounds:", env["observation_bounds"])

# ---------- A. load probabilities (closed form by fine quadrature) ----------
def p_high(c_lo, c_hi, thr=0.4, A_lo=0.1, A_hi=0.4, n=2000):
    # P(c + A cos(phi) >= thr): phase uniform, so P(cos phi >= x) = acos(x)/pi for x in [-1,1]
    cs = np.linspace(c_lo, c_hi, n, endpoint=False) + (c_hi-c_lo)/(2*n)
    As = np.linspace(A_lo, A_hi, n, endpoint=False) + (A_hi-A_lo)/(2*n)
    C, Am = np.meshgrid(cs, As, indexing="ij")
    x = (thr - C)/Am
    p = np.where(x <= -1, 1.0, np.where(x >= 1, 0.0, np.arccos(np.clip(x,-1,1))/np.pi))
    return p.mean()
pH_rec = p_high(0.1, 0.7); pH_strong = p_high(0.7, 1.1)
print(f"P(g>=0.4) recorded law = {pH_rec:.4f}; strong current = {pH_strong:.4f}; low-load fraction under strong = {1-pH_strong:.4f}")
print("note: phase uniform + rotation-invariant waves => same marginal at every t; current constant")

# ---------- B. expected hole / ring rows under the 1/16 recorder ----------
N_train, N_cal, N_bin = 76800, 12800, 7680
for h in (0.025, 0.05, 0.10):
    p_band = 2*h/2.0   # a0 density 1/2 on [-1,1]; band length 2h
    lo, hi = -k-2*h, -k+2*h
    ring_len = (min(-k-h, 1) - max(lo,-1)) + (min(hi,1) - (-k+h))
    p_ring = ring_len/2.0
    print(f"h={h}: band a0 in [{-k-h:.3f},{-k+h:.3f}] P(band)={p_band:.3f}; ring len={ring_len:.3f} P(ring)={p_ring:.3f}; "
          f"E[hole rows] train={N_train*pH_rec*p_band:.0f} cal={N_cal*pH_rec*p_band:.0f} per10stepbin={N_bin*pH_rec*p_band:.0f}; "
          f"E[ring rows at high load] train={N_train*pH_rec*p_ring:.0f}; E[band rows at LOW load, retained] train={N_train*(1-pH_rec)*p_band:.0f}")
# episodes with >=1 hole row (c, A fixed per episode; wave phase rotates through all angles over 100 steps
# so per-episode high-load row count ~ 100 * P_phase(c + A cos >= .4)); use quadrature
n=600
cs = np.linspace(0.1,0.7,n,endpoint=False)+0.3/n; As=np.linspace(0.1,0.4,n,endpoint=False)+0.15/n
C,Am=np.meshgrid(cs,As,indexing="ij"); x=(0.4-C)/Am
frac_high=np.where(x<=-1,1.0,np.where(x>=1,0.0,np.arccos(np.clip(x,-1,1))/np.pi))
for h in (0.025,0.05,0.10):
    p=2*h/2.0
    p_none=(1-p)**(100*frac_high)
    print(f"h={h}: expected share of episodes with >=1 hole row ~ {1-p_none.mean():.3f}; share of episodes with zero high-load rows = {(frac_high==0).mean():.3f}")
print(f"episode share with zero high-load rows (c+A<0.4): {(frac_high==0).mean():.3f}")

# ---------- C. covariance propagation under the recorder, checked against receipt std ----------
def thrust(a):
    a=np.asarray(a,float); return np.clip(a,-k,k) - slope*np.sign(a)*np.maximum(np.abs(a)-k,0)
# var of thrust for a uniform on [-1,1] (recorder is uniform density 1/16 overall)
aa=np.linspace(-1,1,200001); var_u=np.mean(thrust(aa)**2); mean_u=np.mean(thrust(aa))
print(f"thrust(a) for uniform a: mean={mean_u:.4f} var={var_u:.4f}")
mu0=np.zeros(13); mu0[10]=0.4
S0=np.zeros((13,13)); S0[:10,:10]=0.25*np.eye(10); S0[10,10]=0.6**2/12
EA2=(0.4**3-0.1**3)/(3*0.3); S0[11,11]=S0[12,12]=EA2/2
Su=var_u*np.eye(4)
mus=[mu0]; Ss=[S0]
for t in range(100):
    mu=A@mus[-1]; S=A@Ss[-1]@A.T + B@Su@B.T
    mus.append(mu); Ss.append(S)
mus=np.array(mus[:100]); Ss=np.array(Ss[:100])
tot_mean=mus.mean(0); tot_var=Ss[:,range(13),range(13)].mean(0)+((mus-tot_mean)**2).mean(0)
pred_std=np.sqrt(tot_var)
names=env["state_order"]
print("coord  predicted_std(linear plant, recorder)  receipt_std")
for i in range(13): print(f"{names[i]:8s} {pred_std[i]:8.3f} {std[i]:8.3f}")
# per-time-bin neighbour probability: two independent same-time states, standardized by receipt std,
# squared distance ~ sum lambda_j chi2_1 with lambda = eig(2 D^-1 S_t D^-1); Gaussian approximation
rng=np.random.default_rng(0)
D=np.diag(1/std[:13])
def p_within(St, r_rms, m=200000):
    Sz=2*D@St@D; w,V=np.linalg.eigh(Sz); w=np.clip(w,0,None)
    z=rng.standard_normal((m,13))*np.sqrt(w)
    d2=(z**2).sum(1); return (d2<=13*r_rms**2).mean(), np.sqrt(np.quantile(d2/13,[8/767, 64/7680]))
print("time  P(RMS<=0.5 | same t)  E[other-episode points within]  RMS radius for 8 episodes  RMS radius for p=64/7680")
for t in (0,5,25,50,75,95):
    p,(r8,r64)=p_within(Ss[t],0.5)
    print(f"t={t:3d}  {p:.2e}  {767*p:.3f}  {r8:.3f}  {r64:.3f}")
# ---------- D. L-infinity action box ----------
pbox=(0.1**4)/16
print(f"L-inf 0.05 box in 4 actions: P per row={pbox:.2e}; E rows in 76,800 = {N_train*pbox:.2f}; at high load & one 10-step bin = {N_bin*pH_rec*pbox:.4f}")
# ---------- E. band vs ring: thrust, one-step cost and value gap under neutral continuations ----------
print("a0:      thrust   0.3*a0^2")
for a0 in (-1.0,-0.9,-0.85,-0.8,-0.75,-0.7,-0.6,0.0): print(f"{a0:6.2f}  {thrust(a0):7.4f}  {0.3*a0*a0:.4f}")
v_load=np.array([0,1,0,1,0,1,0,1,0,1],float)
def step(x,a):
    s=x[:10]; c=x[10]; w=x[11:13]
    r=-(np.sum(s**2)+0.3*np.sum(np.asarray(a)**2))
    xn=A@x + B@thrust(a)
    return xn,r
def value(x0,t0,first,cont):
    x=x0.copy(); G=0; a=np.array(first,float)
    for j in range(100-t0):
        x,r=step(x,a); G+=0.98**j*r; a=cont(x)
    return G
zero=lambda x: np.zeros(4)
# neutral continuation 2: fixed recorder-like mean action = 0 as well; add a simple proportional damping controller as second neutral reference
def pd(x):  # method-neutral reference: oppose the load on each thruster, clipped
    g=x[10]+x[11]; return np.clip(-np.ones(4)*g,-1,1)
rng2=np.random.default_rng(1)
gaps=[]
for trial in range(200):
    x=np.zeros(13); x[:10]=0.5*rng2.standard_normal(10); c=rng2.uniform(0.1,0.7); Aamp=rng2.uniform(0.1,0.4); ph=rng2.uniform(0,2*np.pi)
    x[10]=c; x[11]=Aamp*np.cos(ph); x[12]=Aamp*np.sin(ph)
    if x[10]+x[11] < 0.4: continue
    for cont,name in ((zero,"zero"),(pd,"pd")):
        vals={a0: value(x,0,[a0,0,0,0],cont) for a0 in (-0.9,-0.85,-0.8,-0.75,-0.7,-0.6)}
        gaps.append((name, vals[-0.8]-vals[-0.7], vals[-0.8]-vals[-0.9], vals[-0.8]-vals[-0.6], max(vals.values())-min(vals.values())))
gaps_np={n:np.array([g[1:] for g in gaps if g[0]==n]) for n in ("zero","pd")}
for n,g in gaps_np.items():
    print(f"continuation={n}: n_states={len(g)}  v(-0.8)-v(-0.7): mean {g[:,0].mean():+.3f} [{g[:,0].min():+.3f},{g[:,0].max():+.3f}]; "
          f"v(-0.8)-v(-0.9): mean {g[:,1].mean():+.3f}; v(-0.8)-v(-0.6): mean {g[:,2].mean():+.3f}; bank spread over 6 a0 values: mean {g[:,3].mean():.3f} max {g[:,3].max():.3f}")
print("reference: development-bank mean |Q1 - v| = 36.0 (development_1) / 29.4 (development_2); within-state v spread median 5.97 across 9 mixed candidates")
