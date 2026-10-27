"""Create final analysis bundle after review, preserving intermediate packaging."""
from pathlib import Path
import hashlib
import json
import zipfile

root = Path(__file__).resolve().parent
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
paths = []
for folder, prefix in ((root, "direct_q_diagnostic_20261021_v2"),
        (root.parent/"direct_q_diagnostic_20261021_v1", "preserved_v1_failure"),
        (root.parent/"direct_q_plan_20261021_v1", "mechanical_stage")):
    for p in sorted(folder.rglob("*")):
        if not p.is_file(): continue
        rel = p.relative_to(folder)
        if any(x in {"__pycache__", "release", "presentation", "release_reviewed", "presentation_reviewed"} for x in rel.parts): continue
        if rel.as_posix() == "report_assets/direct_q_section.html": continue # superseded draft, retained on disk
        paths.append((p, prefix+"/"+rel.as_posix()))
manifest = {n:dict(bytes=p.stat().st_size,sha256=sha(p)) for p,n in paths}
archive = root/"release/ThrusterChain_direct_Q_diagnostic_evidence_20261021_final.zip"
with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as z:
    for p,n in paths: z.write(p,n)
    z.writestr("EVIDENCE_MANIFEST.json", json.dumps(manifest,indent=2)+"\n")
    z.writestr("README.txt", "Analysis-only evidence; no checkpoint binaries or complete historical source closure.\nStart with direct_q_diagnostic_20261021_v2/report_assets/RESULTS.md.\nReproduce saved-record reporting with python report_assets/build_report.py from that directory.\nNumPy is needed. No model inference or extra rollouts occur.\nSee PROVENANCE.md for preserved failure history, limits and external model/source requirements.\nThe final section is report_assets/section.html; an earlier draft is excluded.\n")
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for n,meta in manifest.items(): assert hashlib.sha256(z.read(n)).hexdigest()==meta["sha256"]
guide = root/"presentation_reviewed/ThrusterChain_diagnostics_results_v4_direct_Q.html"
receipt = dict(status="local_final_deliverables_verified", library_delivery="incomplete; no new upload attempted",
    html=dict(path=str(guide),bytes=guide.stat().st_size,sha256=sha(guide)),
    evidence_zip=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),members=len(manifest)+2),
    supersedes_packaging_receipt="delivery_receipt.json; scientific results unchanged",
    omitted="checkpoint binaries, complete historical source closure, earlier HTML draft, helper credentials and upload request")
path = root/"release/final_delivery_receipt.json"
with path.open("x",encoding="utf-8") as f: json.dump(receipt,f,indent=2);f.write("\n")
print(json.dumps(receipt,indent=2))
