import json, numpy as np
R='/home/user/bca/runs/shift_explore/stage2/policy'
P=json.load(open('/home/user/bca/runs/shift_explore/stage2/pooled_exploratory.json'))
m4=np.array(P['policy']['M4_best_standin']['values'])
S=[];H=[];LS=[];LK=[];B=[];K2=[];S2=[]
for s in range(5):
    l=json.load(open(f'{R}/seed{s}/lock.json')); v=json.load(open(f'{R}/seed{s}/values.json'))
    st=l['standin']; S.append(l['target_strength'])
    H.append(v['situations']['strong_current']['points'][st['arm']]); B.append(v['situations']['strong_current']['points']['bca_k1'])
    K2.append(v['situations']['strong_current']['points']['bca_k2']); S2.append(l['strengths']['bca_k2'])
    LS.append(st['record']['ladder_S']); LK.append(st['record']['ladder_knobs'])
    print('seed',s,'BCA S',round(l['target_strength'],4),'hinge matched S',round(st['record']['S'],4),'knob',round(st['record']['knob'],3),'coef',round(st['coefficient'][0],3),
          'ladder S',np.round(st['record']['ladder_S'],4).tolist(),'| hinge pts',round(H[-1],1),'BCA pts',round(B[-1],1),'M4',round(m4[s],1),
          '| hand-set hinge S',round(l['strengths']['range_hinge'],4))
S=np.array(S);H=np.array(H);B=np.array(B)
print('corr(M4, BCA S)',np.corrcoef(m4,S)[0,1].round(3),' corr(hinge pts, S)',np.corrcoef(H,S)[0,1].round(3),' corr(BCA pts,S)',np.corrcoef(B,S)[0,1].round(3))
print('k2-k1 pts per seed',np.round(np.array(K2)-B,1).tolist(),'dS',np.round(np.array(S2)-S,4).tolist())
# standins classifier ratio vs BCA gain per seed
M=json.load(open('/home/user/bca/runs/shift_explore/standins/marks.json'))
cr=np.array(M['marks']['classifier_ratio']['gain_strong_current']['values'])
none=np.array([json.load(open(f'{R}/seed{s}/values.json'))['situations']['strong_current']['points']['none'] for s in range(5)])
g=B-none
print('BCA gain',np.round(g,1).tolist(),'mean',g.mean().round(1),' classifier-ratio gain',np.round(cr,1).tolist(),'mean',cr.mean().round(1),' diff',np.round(g-cr,1).tolist(),'mean',(g-cr).mean().round(1),'corr',np.corrcoef(g,cr)[0,1].round(2))
print('none_points_own_minus_room',json.dumps(M.get('none_points_own_minus_room'))[:300])
