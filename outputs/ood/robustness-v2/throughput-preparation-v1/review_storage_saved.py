"""Independent saved-frame/read-only SQL review; never imports the benchmark."""
from pathlib import Path
import datetime,hashlib,json,sqlite3,struct,sys,zlib
root=Path(sys.argv[1]);report=json.loads((root/'report.json').read_bytes())
sha=lambda b:hashlib.sha256(b).hexdigest()
expected=['reserve','before','native_input','native_applied','native_output','output','after','completed','ack']
reviewed=[]

def records(paths):
    items=[];tails=0
    for path in paths:
        if path.suffix=='.sqlite':
            db=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)
            db.execute('PRAGMA query_only=ON')
            table='evidence' if path.name=='archive.sqlite' else 'accounting'
            rawrows=[x[0] for x in db.execute('SELECT frame FROM '+table+' ORDER BY id')];db.close()
        else:
            data=path.read_bytes();rawrows=[];at=0
            while at<len(data):
                if len(data)-at<36:tails+=len(data)-at;break
                size=int.from_bytes(data[at:at+4],'big')
                assert size<100000
                if len(data)-at<36+size:tails+=len(data)-at;break
                rawrows.append(data[at:at+36+size]);at+=36+size
        for frame in rawrows:
            n=int.from_bytes(frame[:4],'big');body=frame[36:]
            assert n==len(body) and hashlib.sha256(body).digest()==frame[4:36]
            header,compressed=body.split(b'\n',1);meta=json.loads(header)
            raw=zlib.decompress(compressed)
            assert len(raw)==meta['bytes'] and sha(raw)==meta['sha256']
            items.append((meta,raw,frame[4:36].hex()))
    items.sort(key=lambda r:r[0]['index']);previous='0'*64;calls={}
    for index,(meta,raw,digest) in enumerate(items):
        assert meta['index']==index and meta['previous']==previous
        kinds=calls.setdefault(meta['call'],[]);assert meta['kind']==expected[len(kinds)];kinds.append(meta['kind'])
        if meta['kind']=='reserve':assert json.loads(raw)==dict(token=meta['call'],environment=1,physics=4)
        if meta['kind']=='ack':assert json.loads(raw)==dict(token=meta['call'],evidence_head=previous)
        previous=digest
    return items,tails,calls

baseline=None
for sample in report['samples']:
    folder=root/sample['name'];mode=sample['mode']
    paths=[folder/'archive.sqlite',folder/'ledger.sqlite'] if mode=='sqlite_shared' else [folder/'journal.bin'] if mode=='append_shared' else [folder/'retained-node-journal.bin']
    if mode=='append_node_replicated':assert paths[0].read_bytes()==(folder/'replica.bin').read_bytes()
    for pin in sample['files']:
        path=Path(pin['path'])
        if str(path).startswith(report['node_root']+'/'):path=folder/('retained-node-'+path.name)
        assert path.stat().st_size==pin['bytes'] and sha(path.read_bytes())==pin['sha256']
    items,tail,calls=records(paths)
    assert len(items)==288 and len(calls)==32 and all(k==expected for k in calls.values()) and tail==0
    same=[(m['call'],m['kind'],sha(raw),digest) for m,raw,digest in items]
    if baseline is None:baseline=same
    assert same==baseline,'Logical bytes/chain differ between modes'
    assert sample['seconds']>0 and sample['calls_per_second']==32/sample['seconds']
    reviewed.append(dict(name=sample['name'],records=len(items),same_logical_bytes=True))
for receipt in report['crash_receipts']:
    mode=receipt['mode'];cut=receipt['cut']
    folder=root/('death-partial' if cut=='partial_frame' else f'death-{mode}-{cut}')
    paths=[folder/'archive.sqlite',folder/'ledger.sqlite'] if mode=='sqlite_shared' else [folder/'journal.bin'] if mode=='append_shared' else [folder/'retained-node-journal.bin']
    if mode=='append_node_replicated':assert paths[0].read_bytes()==(folder/'replica.bin').read_bytes()
    items,tail,calls=records(paths)
    assert receipt['actual_exit']==23 and len(calls)==1 and list(calls)==[0]
    assert len(items)==dict(reserve=1,pre_action=4,post_output=8,complete=9,partial_frame=1)[cut]
    assert tail==(3 if cut=='partial_frame' else 0)
    assert (calls[0][-1]=='ack')==(cut=='complete')
assert len(reviewed)==12 and len(report['crash_receipts'])==17
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),report_sha256=sha((root/'report.json').read_bytes()),
    sample_count=12,frames_checked=12*288,death_cases=17,identical_logical_bytes=True,
    reservation_and_completion_dispositions_match=True,partial_frame_retained=True,
    source_sha256=sha(Path(__file__).read_bytes()),scientific_storage_acceptance=False,component_import=False)))
