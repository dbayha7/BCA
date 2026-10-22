"""Copy this completed evidence into a new, nonexisting project subfolder only."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT=Path(__file__).resolve().parents[1]
PARENT=Path(r'C:\Users\David\dev\BCA\experiments\thrusterchain_diagnostics')
DEST=PARENT/'action_only_20261021'

if __name__=='__main__':
    if DEST.exists():raise RuntimeError('Refusing to overwrite an existing project snapshot')
    if DEST.resolve().parent!=PARENT.resolve() or not PARENT.is_dir():raise RuntimeError('Unexpected destination')
    shutil.copytree(ROOT,DEST,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.pending','execution.lock'))
    checked=0
    for path in DEST.rglob('*'):
        if path.is_file():
            original=ROOT/path.relative_to(DEST)
            if hashlib.sha256(path.read_bytes()).digest()!=hashlib.sha256(original.read_bytes()).digest():
                raise RuntimeError('Copied bytes differ: '+str(path))
            checked+=1
    print(json.dumps(dict(destination=str(DEST),verified_files=checked,original=str(ROOT),
        overwrite=False,existing_files_changed=False,kind='completed evidence snapshot; not a duplicate execution')))
