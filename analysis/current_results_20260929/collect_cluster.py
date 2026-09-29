"""Read-only extraction of CLOSED saved evaluations. Never opens SQLite or weights."""
from pathlib import Path
import io, json, hashlib, gzip, sys
from datetime import datetime, timezone
import numpy as np

reuse = set(json.loads(sys.argv[1])) if len(sys.argv) > 1 else set()
queues = [Path('/users/dbayha/bca-standard-noiw-v1/campaign/queue-cluster-v1'),
          Path('/users/dbayha/bca-standard-noiw-v1/cql-validation-recovery-v2/queue-v1')]
sha = lambda b: hashlib.sha256(b).hexdigest()
out = {'captured_utc': datetime.now(timezone.utc).isoformat(), 'queues': [], 'runs': [], 'errors': []}
for q in queues:
    qb = (q/'queue_status.json').read_bytes(); qs = json.loads(qb)
    out['queues'].append({'path': str(q), 'sha256':sha(qb), 'status':qs['status'],
                          'current':qs.get('current'), 'completed_count':len(qs['completed']), 'total':qs['total']})
    for item in qs['completed']:
        rid = item['run_id']; d = q/'runs'/rid
        if len(sys.argv)>2 and sys.argv[2]=='iql-only' and not rid.startswith('iql-'): continue
        try:
            rb = (d/'result.json').read_bytes(); r = json.loads(rb)
            assert sha(rb) == item['result_sha256']
            ex = json.loads((q/'attempts'/rid/'actual_exit.json').read_text())
            assert ex['actual_returncode'] == 0
            assert r['steps_completed'] == 1000000
            assert r.get('completed') is True or r.get('status') == 'complete'
            cfgb = (d/'resolved.json').read_bytes(); cfg = json.loads(cfgb)
            sb = (d/'source.json').read_bytes()
            run = {'run_id':rid,'root':str(d),'result_sha256':sha(rb),'source_manifest_sha256':sha(sb),
                   'source_pins':json.loads(sb),'resolved_sha256':sha(cfgb),'data':cfg.get('cache'),
                   'environment':cfg['environment'],'host':cfg['host'],'method':cfg['method'],
                   'actual_exit':ex,'steps_completed':r['steps_completed'],
                   'counters':r.get('checkpoints',r.get('state_counters')),
                   'reused_local':rid in reuse,'trajectories':[]}
            if rid not in reuse:
                if cfg['host'] == 'iql':
                    def bank(v, kind, step):
                        b = (d/v['path']).read_bytes(); assert sha(b) == v['sha256']
                        with np.load(io.BytesIO(b),allow_pickle=False) as z:
                            scores=np.asarray(z['final_scores' if kind=='final' else 'scores']).reshape(-1).tolist()
                            seeds=np.asarray(z['environment_seeds']).reshape(-1).tolist()
                            returns=np.asarray(z['final_returns' if kind=='final' else 'returns']).reshape(-1).tolist()
                        assert len(scores)==(20 if kind=='final' else cfg['options']['curve_episodes'])
                        return {'kind':kind,'step':int(step),'scores':scores,'seeds':seeds,'returns':returns,
                                'source':v['path'],'sha256':sha(b)}
                    arms = list(r['artifacts']['final'])
                    for arm in arms:
                        banks=[bank(v[arm],'periodic',step) for step,v in sorted(r['artifacts']['curves'].items(),key=lambda x:int(x[0]))]
                        banks.append(bank(r['artifacts']['final'][arm],'final',1000000))
                        assert len(banks)==201
                        run['trajectories'].append({'arm':arm if cfg['method']=='bca' else 'standalone_host','banks':banks})
                else:
                    sk,rk=('score','return') if cfg['host']=='cql' else ('normalized_score','raw_return')
                    banks=[{'kind':b['kind'],'step':b['step'],'scores':[e[sk] for e in b['episodes']],
                            'returns':[e[rk] for e in b['episodes']], 'seeds':[e['seed'] for e in b['episodes']]} for b in r['evaluations']]
                    assert len(banks)==201
                    run['trajectories']=[{'arm':'bca' if cfg['method']=='bca' else 'host','banks':banks}]
                for t in run['trajectories']:
                    assert [b['step'] for b in t['banks'] if b['kind']=='periodic']==list(range(5000,1000001,5000))
                    assert len([b for b in t['banks'] if b['kind']=='final'])==1
                    assert all(np.isfinite(b['scores']).all() for b in t['banks'])
            # CQL sparse scalar snapshots from closed journals only. Hash before and after.
            if cfg['host']=='cql' and rid not in reuse:
                ep = d/'events.jsonl.gz'; eb = ep.read_bytes(); assert sha(eb)==r['events_sha256']
                scalar=[]; refresh=[]
                with gzip.GzipFile(fileobj=io.BytesIO(eb)) as f:
                    for line in f:
                        e=json.loads(line)
                        if e.get('kind')=='accepted_scan': scalar.append({'step':e['step'],'metrics_last':e['metrics_last']})
                        elif e.get('kind')=='refresh': refresh.append(e)
                assert sha(ep.read_bytes()) == sha(eb)
                run['scalar_snapshots']=scalar;run['refreshes']=refresh;run['events_sha256']=sha(eb)
            assert sha((d/'result.json').read_bytes())==sha(rb)
            out['runs'].append(run)
        except Exception as e:
            out['errors'].append({'run_id':rid,'type':type(e).__name__,'error':str(e)})
out['finished_utc']=datetime.now(timezone.utc).isoformat()
print(json.dumps(out,separators=(',',':'),allow_nan=False))
