import json,sys
d=json.load(open(sys.argv[1],encoding='utf-8'))
for k in ('dataset_path','calibration_role','validation_role','coverage_limit','bindings_sha256'):
    print(k,':',d[k])
ce=d['calibration_episode_indices']; ve=d['validation_episode_indices']; fe=d['fit_episode_indices']
print('cal ep min/max/len/unique',min(ce),max(ce),len(ce),len(set(ce)),'sorted',ce==sorted(ce))
print('cal ep first10',ce[:10],'last5',ce[-5:])
print('cal times min/max',min(d['calibration_times']),max(d['calibration_times']))
print('cal group ids first5',d['group_ids']['calibration'][:5],'min/max',min(d['group_ids']['calibration']),max(d['group_ids']['calibration']))
print('cal row ids first5',d['row_ids']['calibration'][:5])
print('val ep min/max',min(ve),max(ve))
print('fit ep min/max',min(fe),max(fe))
print('cal∩val',len(set(ce)&set(ve)),'cal∩fit',len(set(ce)&set(fe)))
print('gid cal == ep?', d['group_ids']['calibration']==ce)
cg=d['group_ids']['calibration']; vg=d['group_ids']['validation']
print('cal group contiguous', sorted(cg)==list(range(min(cg),min(cg)+128)))
print('val group min/max',min(vg),max(vg), sorted(vg)==list(range(min(vg),min(vg)+128)))
