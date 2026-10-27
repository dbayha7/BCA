"""Create a compact immutable source/tests/results bundle and verify every entry."""
import argparse,hashlib,json,zipfile
from pathlib import Path
from run_benchmark import source_manifest

ROOT=Path(__file__).resolve().parent

def package(output):
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise FileExistsError(output)
    roots=["thrusterchain","tests","contracts","docs","audits","inputs","third_party"]
    selected=[]
    for name in roots:
        for p in (ROOT/name).rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".pyc",".pyo"):
                if name == "audits" and (
                    "run_copy" in p.parts or "artifact_fixture" in p.parts or
                    (any(part.startswith(("revision_v3", "revision_v4")) for part in p.parts) and p.suffix == ".npz")
                ):
                    continue
                selected.append(p)
    selected.extend(p for p in ROOT.glob("*") if p.is_file() and p.suffix in (".py",".toml",".txt",".md"))
    selected.append(ROOT/"CURRENT_SOURCE_MANIFEST.json")
    for result_name in ("smoke_v2", "revision_v2_smoke", "revision_v3_smoke", "revision_v4_smoke", "revision_v4_training_example", "revision_v4_smoke_final", "revision_v4_training_example_final"):
        selected.extend(p for p in (ROOT/"results"/result_name).rglob("*") if p.is_file())
    selected=sorted(set(selected),key=lambda p:p.relative_to(ROOT).as_posix())
    files=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in selected]
    manifest=dict(schema="thrusterchain.release-manifest.v1",source=source_manifest(),
        files=files,entry_count=len(files),purpose="private independent review and research meeting",
        profile="smoke",research_nine_cell_training_complete=False,
        design_archive_verified=False,
        historical_root_preserved=True,
        revision="20261022_v4", basis_archive_sha256="92bdeae507396894066a8374889af67a1e00de9a8d06f8a8f95c1177993b6498",
        omitted_generated_review_fixtures="Large audit run_copy/artifact_fixture directories and v3/v4 audit NPZ mutations remain local. Reproduction scripts, receipts, hashes and all preserved/fresh result records are included.",
        archive_scope="Revision v4 budget and adaptation protocols, source/tests/contracts/docs; preserved earlier results, fresh strict smoke and small real-update protocol example. No full-budget neural study. Earlier v1/v2/v3 archives unchanged.")
    with zipfile.ZipFile(output,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p,entry in zip(selected,files):
            z.write(p,entry["path"])
        z.writestr("MANIFEST.json",json.dumps(manifest,indent=2,sort_keys=True,allow_nan=False))
    with zipfile.ZipFile(output) as z:
        loaded=json.loads(z.read("MANIFEST.json"))
        assert len(z.namelist())==len(files)+1
        for entry in loaded["files"]:
            content=z.read(entry["path"])
            assert len(content)==entry["bytes"]
            assert hashlib.sha256(content).hexdigest()==entry["sha256"]
    digest=hashlib.sha256(output.read_bytes()).hexdigest()
    receipt=dict(file=str(output.resolve()),bytes=output.stat().st_size,sha256=digest,
        entries=len(files)+1,embedded_manifest_verified=True,
        source_sha256=manifest["source"]["source_sha256"])
    with output.with_suffix(".receipt.json").open("x",encoding="utf8") as f:
        json.dump(receipt,f,indent=2,sort_keys=True)
    print(json.dumps(receipt))
    return receipt

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();package(args.output)

