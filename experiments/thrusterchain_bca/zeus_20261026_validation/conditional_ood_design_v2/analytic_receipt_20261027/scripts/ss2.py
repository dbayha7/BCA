import json,sys
a=json.load(open(sys.argv[1],encoding='utf-8')); b=json.load(open(sys.argv[2],encoding='utf-8'))
print('same episode ids', a['calibration_episode_indices']==b['calibration_episode_ids'])
for k,v in b.items():
    if any(w in k.lower() for w in ('seed','rule','select','time')) and not any(w in k.lower() for w in ('final',)) and 'selection_and' not in k:
        print(k,':',json.dumps(v)[:240])
