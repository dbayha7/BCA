import json, numpy as np
D1="/home/user/bca/runs/shift_explore/deploy/seed{k}_stall_gapped.json"
S2="/home/user/bca/runs/shift_explore/stage2/deploy/stall/seed{k}/values.json"
for k in range(5):
    d=json.load(open(D1.format(k=k)))['situations']['strong_current']['rules']
    s=json.load(open(S2.format(k=k)))['results']['none']['strong_current']
    P=lambda r: np.array(r['per_start_points'])
    never2,never1=P(s['none']),P(d['none:None'])
    sa2,sa1=P(s['sa95']),P(d['sa95:clip99'])
    bca,rnd,perf,ratio,rratio,allc=P(s['bca']),P(s['random:bca']),P(d['unrec:clip99']),P(d['ratio:clip99']),P(d['random:ratio:clip99']),P(d['always:clip99'])
    sh=perf-bca; dr=bca-rnd
    print(f"seed{k} never maxdiff {np.abs(never2-never1).max():.1e} sa95 maxdiff {np.abs(sa2-sa1).max():.1e} n={len(bca)}")
    print(f"  shortfall vs perfect: median {np.median(sh):.1f} p90 {np.quantile(sh,.9):.1f} mean {sh.mean():.1f}; >10 below perfect bca {(sh>10).sum()} random {((perf-rnd)>10).sum()}")
    print(f"  bca>rnd {(dr>1e-6).sum()} bca<rnd {(dr<-1e-6).sum()} mean when worse {dr[dr<-1e-6].mean() if (dr<-1e-6).any() else 0:.1f}; mean dr {dr.mean():.2f}")
    # share of total dr from losing starts
    print(f"  sum of losses {dr[dr<0].sum()/200:.1f} sum of wins {dr[dr>0].sum()/200:.1f}")
    # what if BCA's worst k starts replaced by perfect
    print(f"  perfect - always clip per start mean {(perf-allc).mean():.2f}; ratio-rratio {(ratio-rratio).mean():.1f}")
