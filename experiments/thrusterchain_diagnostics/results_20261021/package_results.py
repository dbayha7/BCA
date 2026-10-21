"""Build analysis evidence without modifying or including training checkpoints."""
from pathlib import Path
import hashlib, json, shutil, zipfile

base=Path(__file__).resolve().parent
repo=Path(r'C:\Users\David\dev\BCA')
suite=repo/'experiments/thrusterchain_diagnostics'
run=suite/'training_runs/core_20261020T2250Z'
release=base/'release/ThrusterChain_results_20261021'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def cp(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():
        if sha(src)!=sha(dst): raise FileExistsError(dst)
        return
    shutil.copy2(src,dst)
def tree(src,dst):
    for p in sorted(src.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.msgpack') and not p.name.startswith(('layout_preview','stage_preview','audit_preview')):
            cp(p,dst/p.relative_to(src))
def main():
    if release.exists(): raise FileExistsError(release)
    for folder in ('analysis','audit','method_audit','module_fixtures','penalty_alignment'):
        tree(base/folder,release/folder)
    for name in ('review.md','DELIVERY_README.md'):
        if (base/name).exists(): cp(base/name,release/name)
    tree(base/'presentation',release/'presentation')
    cp(base/'inputs/titan_results.json',release/'inputs/titan_results.json')
    cp(Path(__file__),release/'package_results.py')
    cp(base/'analysis_penalty_alignment.py',release/'analysis_penalty_alignment.py')
    reserved=json.loads((run/'.worker_1.lock').read_text())['job_ids']
    matrix=json.loads((run/'job_matrix.json').read_text())
    for name in ('config.json','job_matrix.json','provenance.json','prepared_manifest.json','execution_plan.json'):
        cp(run/name,release/'evidence/core_20261020T2250Z'/name)
    for folder in ('banks','oracle','evaluation/controls'):
        tree(run/folder,release/'evidence/core_20261020T2250Z'/folder)
    for p in (run/'data').rglob('*.json'):
        cp(p,release/'evidence/core_20261020T2250Z'/p.relative_to(run))
    for job in matrix:
        if job['job_id'] in reserved: continue
        jd=run/'jobs'/job['job_id']
        for name in ('manifest.json','driver_status.json','normalization.npz','reservation.npz'):
            cp(jd/name,release/'evidence/core_20261020T2250Z/jobs'/job['job_id']/name)
        for folder in ('evaluation/jobs','evaluation_status/jobs'):
            tree(run/folder/job['job_id'],release/'evidence/core_20261020T2250Z'/folder/job['job_id'])
    for name in ('release_proof.json','release_proof_before_initialization.json','previous_worker_lock.json'):
        p=run/'migration_worker1'/name
        if p.exists(): cp(p,release/'evidence/core_20261020T2250Z/migration_worker1'/name)
    modules={'algorithms':['__init__','td3_bc','td3_bc_bca'], 'calibration':['__init__','bank','dose','network','reference','wbcp'], 'runtime':['__init__','networks','published','rewards']}
    for folder,names in modules.items():
        for name in names: cp(repo/folder/(name+'.py'),release/'source'/folder/(name+'.py'))
    cp(repo/'configs/corl_reference_scores.csv',release/'source/configs/corl_reference_scores.csv')
    for p in suite.iterdir():
        if p.is_file() and p.suffix in ('.py','.md','.toml','.txt'):
            cp(p,release/'source/experiments/thrusterchain_diagnostics'/p.name)
    for folder in ('tests','configs'):
        tree(suite/folder,release/'source/experiments/thrusterchain_diagnostics'/folder)
    method=json.loads((base/'method_audit/evidence.json').read_text(encoding='utf-8'))
    for name,record in method['source_files'].items():
        src=(repo/name).resolve()
        assert src.is_relative_to(repo.resolve())
        assert sha(src).lower()==record['sha256'].lower(),name
        cp(src,release/'source'/name)
    for name,record in method['evidence_files'].items():
        src=(repo/name).resolve()
        assert src.is_relative_to(repo.resolve())
        assert sha(src).lower()==record['sha256'].lower(),name
        cp(src,release/'historical_evidence'/name)
    closure=json.loads((base/'module_fixtures/verification_v2/source_closure.json').read_text(encoding='utf-8'))
    for name,digest in closure['files'].items():
        src=(repo/name).resolve()
        assert src.is_relative_to(repo.resolve())
        assert sha(src).lower()==digest.lower(),name
        cp(src,release/'source'/name)
    files={p.relative_to(release).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(release.rglob('*')) if p.is_file()}
    manifest=dict(schema='thrusterchain-analysis-evidence-v1',files=files,
        titan_report_library_id='libfile_579a1a6d882c8191b8caeb161bcd51d9',
        titan_report_sha256='1f65b23ab78d06a16390941a9b146b6ec80e7cde6d44e0e790e92f64b15ca2ee',
        purpose='analysis and code evidence; no training checkpoints or full offline datasets',
        limitations=['Titan summaries imported and checked; no independent Titan rollouts or checkpoint bytes',
                     'frozen prepared_manifest contains hashes of omitted large inputs; training-run verification requires original full run'])
    (release/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    archive=base/'release/ThrusterChain_results_evidence_20261021.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(release.rglob('*')):
            if p.is_file(): z.write(p,Path(release.name)/p.relative_to(release))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert not any(n.endswith('.msgpack') or '/.library_helpers/' in n for n in z.namelist())
    print(json.dumps(dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),files=len(files)+1)))
if __name__=='__main__': main()
