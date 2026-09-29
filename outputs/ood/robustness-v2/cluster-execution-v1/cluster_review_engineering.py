"""Independent saved engineering arithmetic; no model/simulator/physical calls."""
from pathlib import Path
import datetime, hashlib, json, sqlite3, zlib
import numpy as np

ROOT=Path(__file__).resolve().parent
ATTEMPT=ROOT/'results'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def decode(v):
    if isinstance(v,list):return [decode(x) for x in v]
    if isinstance(v,dict):
        if set(v)=={'__array__','dtype','shape'}:
            return np.frombuffer(bytes.fromhex(v['__array__']),dtype=v['dtype']).reshape(v['shape']).copy()
        if set(v)=={'__tuple__'}:return tuple(decode(x) for x in v['__tuple__'])
        return {k:decode(x) for k,x in v.items()}
    return v

def main():
    engineering=json.loads((ATTEMPT/'engineering.json').read_bytes())
    declaration=json.loads((ATTEMPT/'declaration.json').read_bytes())
    assert engineering['live_gate_passed'] is True and engineering['environment_transitions']==5
    assert all(sha(Path(p).read_bytes())==h for p,h in declaration['sources'].items())
    db=sqlite3.connect((ATTEMPT/'evidence.sqlite').as_uri()+'?mode=ro',uri=True)
    db.execute('PRAGMA query_only=ON')
    pins={}
    def read(name,decoded=True):
        meta,compressed,digest=db.execute('SELECT metadata,payload,digest FROM artifacts WHERE name=?',(name,)).fetchone()
        meta=json.loads(meta);assert meta['complete'] is True and sha(canonical(meta))==digest
        raw=zlib.decompress(compressed);assert len(raw)==meta['bytes'] and sha(raw)==meta['raw_sha256']
        pins[name]=sha(raw);v=json.loads(raw);return decode(v) if decoded else v
    comparisons=[]
    for method in ('host','bca'):
        parity=read('engineering/'+method+'/parity')
        error=np.abs(parity['direct'].astype(np.float64)-parity['fast'].astype(np.float64))
        assert np.array_equal(error,parity['error']) and error.max()<=1e-6
        outputs=[];afters=[]
        for name in ('first','repeat'):
            prefix='calls/engineering/'+method+'/'+name
            rec=read(prefix+'/output');before=read(prefix+'/before')['snapshot'];after=read(prefix+'/after')
            sent=rec['proposed_action'];applied=np.clip(-np.ones_like(sent)+(sent+np.float32(1))*np.float32(.5)*np.float32(2),-1,1)
            assert sent.dtype==np.float32 and rec['applied_action'].dtype==np.float32
            assert np.array_equal(applied,rec['applied_action']) and np.array_equal(applied,rec['sim_ctrl'])
            # Hopper's dt is four .002-second physics frames.
            forward=(float(after['qpos'][0])-float(before['qpos'][0]))/.008
            cost=-.001*float(np.sum(applied.astype(np.float64)**2))
            reconstructed=forward+1.+cost
            assert forward==rec['forward'] and cost==rec['action_cost']
            assert abs(rec['reward']-reconstructed)<=1e-7
            assert after['elapsed_steps']==before['elapsed_steps']+1==rec['elapsed_steps']
            outputs.append(read(prefix+'/output',False));afters.append(read(prefix+'/after',False))
        assert outputs[0]==outputs[1] and afters[0]==afters[1]
        comparisons.append(dict(method=method,action_error=float(error.max()),full_record_repeat_exact=True,full_state_repeat_exact=True))
    before=read('constructor/input')['native_before'];after=read('constructor/after')
    control=read('constructor/applied-before-physics')['control'];returned=read('constructor/output')['returned']
    forward=(float(after['data']['qpos'][0])-float(before['data']['qpos'][0]))/before['dt']
    expected=forward+1.-.001*float(np.sum(control.astype(np.float64)**2))
    assert np.array_equal(after['data']['ctrl'],control) and abs(float(returned[1])-expected)<=1e-7
    db.close()
    result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),accepted_live_engineering=True,
        scope='saved live engineering only; ongoing collection and future outcomes not independently accepted',
        comparisons=comparisons,constructor_reward_error=abs(float(returned[1])-expected),
        evidence_sha256=pins,declaration_sha256=sha((ATTEMPT/'declaration.json').read_bytes()),
        new_simulator_calls=0,model_queries=0,scientific_result_complete=False)
    with (ATTEMPT/'engineering-independent-review.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='evidence_sha256'}))

if __name__=='__main__':main()
