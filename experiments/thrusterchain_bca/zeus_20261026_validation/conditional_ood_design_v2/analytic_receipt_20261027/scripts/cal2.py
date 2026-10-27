import json,sys
d=json.load(open(sys.argv[1],encoding='utf-8'))
ce=d['calibration_episode_indices']; ve=d['validation_episode_indices']; fe=d['fit_episode_indices']
print('cal&val',len(set(ce)&set(ve)),'cal&fit',len(set(ce)&set(fe)),'val&fit',len(set(ve)&set(fe)))
print('gid cal == ep', d['group_ids']['calibration']==ce)
print('rowid == ep*100+t', all(r==e*100+t for r,e,t in zip(d['row_ids']['calibration'],ce,d['calibration_times'])))
print('max(ce) < min(ve)', max(ce)<min(ve))
print('pairwise', d['pairwise_episode_overlap'])
