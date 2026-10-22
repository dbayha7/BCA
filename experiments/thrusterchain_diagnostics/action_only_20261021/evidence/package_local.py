"""Create a private analysis-only delivery archive; no network or upload action."""
from pathlib import Path
import hashlib
import json
import zipfile
import datetime as dt

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT.parent/'results_20261021/release/ThrusterChain_results_20261021'
DEST=ROOT/'release/ThrusterChain_action_only_evidence_20261021.zip'
PREFIX='ThrusterChain_action_only_20261021/'


def digest(content):return hashlib.sha256(content).hexdigest()


def main():
    if DEST.exists():raise RuntimeError('Delivery already exists; preserve it and use a new version')
    files={}
    for path in ROOT.iterdir():
        if path.is_file() and path.suffix in {'.py','.json','.md','.txt','.log'}:
            if path.name.startswith(('upload','library','delivery')):continue
            files[path.name]=path
    files['state_bank.npz']=ROOT/'state_bank.npz'
    for directory in ['tests','analysis','evidence']:
        for path in (ROOT/directory).rglob('*'):
            if path.is_file() and path.suffix in {'.py','.json','.md','.txt','.csv','.svg'} and '__pycache__' not in path.parts:
                files[path.relative_to(ROOT).as_posix()]=path
    for path in (ROOT/'results').glob('replicate_*.json'):
        files[path.relative_to(ROOT).as_posix()]=path
    for path in (PARENT/'source').rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts:
            files['parent_run/'+path.relative_to(PARENT).as_posix()]=path
    for name in ['MANIFEST.json','PORTABLE_VERIFICATION.json']:
        if (PARENT/name).exists():files['parent_run/original_'+name]=PARENT/name
    meta=PARENT/'evidence/core_20261020T2250Z'
    for path in meta.glob('*.json'):
        files['parent_run/evidence/'+path.name]=path
    for path in (meta/'jobs/replicate_01/full/bca').glob('*.json'):
        files['parent_run/evidence/replicate_01_full_bca/'+path.name]=path
    for path in (ROOT/'presentation').glob('*.py'):
        files[path.relative_to(ROOT).as_posix()]=path
    for path in (ROOT/'presentation').glob('*.json'):
        files[path.relative_to(ROOT).as_posix()]=path
    inventory={}
    DEST.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(DEST,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for name,path in sorted(files.items()):
            data=path.read_bytes()
            inventory[name]=dict(bytes=len(data),sha256=digest(data))
            archive.writestr(PREFIX+name,data)
        manifest=dict(schema='action-only-analysis-delivery-v1',created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            files=inventory,counts=dict(files=len(inventory),completed_replications=200,paired_records=600),
            exclusions=['73,304,210 bytes of raw replication NPZ arrays retained locally with hashes',
                        'original trained model checkpoint binaries', '41,387,360-byte original source audit NPZ',
                        '11MB standalone HTML guide delivered separately', 'private Library/authentication helpers and receipts'],
            verification='Raw arrays were locally verified before packaging. Records-only reaggregation does not reverify omitted raw bytes.',
            command='python analysis/aggregate_results.py --records-only --output-dir analysis/reproduced',
            rerun_limit='Scoring replay additionally requires the authenticated checkpoint, normalizer, source audit and root identities from engine_identity.json.',
            privacy='Prepared for the existing private Library delivery scope; this archive does not grant publication rights or independently establish institutional approval.')
        archive.writestr(PREFIX+'DELIVERY_MANIFEST.json',json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    with zipfile.ZipFile(DEST) as archive:
        for name,item in inventory.items():
            data=archive.read(PREFIX+name)
            assert len(data)==item['bytes'] and digest(data)==item['sha256']
    receipt=dict(file=str(DEST),bytes=DEST.stat().st_size,sha256=digest(DEST.read_bytes()),verified_members=len(inventory),
                 archive_integrity='all member byte lengths and SHA256 digests verified',created_utc=manifest['created_utc'])
    (ROOT/'release/bundle_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt))


if __name__=='__main__':main()
