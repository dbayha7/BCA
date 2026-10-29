"""Records-only packaging: preserve original HTML bytes and all frozen evidence."""
from pathlib import Path
import hashlib
import json
import re
import zipfile
from html.parser import HTMLParser

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "action_only_20261021/presentation/ThrusterChain_diagnostics_results_v3_delivery.html"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class Inventory(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.links, self.units, self.scripts, self.images = [], [], [], [], []
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a: self.ids.append(a["id"])
        if tag == "a" and a.get("href", "").startswith("#"): self.links.append(a["href"][1:])
        if "data-unit" in a: self.units.append(a["data-unit"])
        if tag == "script": self.scripts.append(a)
        if tag == "img": self.images.append(a)

def main():
    # This file is ancillary packaging code; it never loads models or changes outcomes.
    target = ROOT / "presentation_reviewed"
    target.mkdir(exist_ok=True)
    assert not any(target.iterdir()), "Never overwrite an existing delivery"
    old_bytes = OLD.read_bytes()
    assert hashlib.sha256(old_bytes).hexdigest() == "9e3ddabd4bfd6a1d3f3bf7f801c6b6f3310ab9843e722e2726212f563f86fe38"
    old = old_bytes.decode("utf-8")
    fragment = (ROOT / "report_assets/section.html").read_text(encoding="utf-8")
    assert not any(ord(c) < 32 and c not in "\n\r\t" for c in fragment)
    assert "<script" not in fragment.lower(), "New section must not add executable scripts"
    marker = "<style>\r\n#action-only-results" if "<style>\r\n#action-only-results" in old else "<style>\n#action-only-results"
    assert old.count(marker) == 1
    nav_marker = '<li class="nv-sec"><a href="#training-results" data-unit="training-results">'
    assert old.count(nav_marker) == 1
    nav = '<li class="nv-sec"><a href="#direct-q-ranking" data-unit="direct-q-ranking">Direct value penalty: frozen action test</a></li>\n<li class="nv-sec"><a href="#action-only-results" data-unit="action-only-results">Controlled action-only shift</a></li>\n'
    combined = old.replace(marker, fragment + "\n" + marker).replace(nav_marker, nav + nav_marker)
    assert combined.replace(fragment + "\n", "", 1).replace(nav, "", 1).encode("utf-8") == old_bytes
    before, after = Inventory(), Inventory()
    before.feed(old); after.feed(combined)
    new_ids = [x for x in after.ids if x not in before.ids]
    assert len(new_ids) == len(set(new_ids))
    assert all(after.ids.count(x) == 1 for x in new_ids)
    assert set(before.ids) <= set(after.ids)
    assert after.scripts == before.scripts and after.images == before.images
    assert re.findall(r'<script\b[^>]*>.*?</script>', combined, re.S | re.I) == re.findall(r'<script\b[^>]*>.*?</script>', old, re.S | re.I)
    missing_before, missing_after = set(before.links)-set(before.ids), set(after.links)-set(after.ids)
    assert not (missing_after-missing_before), missing_after-missing_before
    assert "direct-q-ranking" in after.ids and "direct-q-ranking" in after.units
    html_path = target / "ThrusterChain_diagnostics_results_v4_direct_Q.html"
    html_path.write_bytes(combined.encode("utf-8"))
    qa = dict(original_html_sha256=sha(OLD), revised_html_sha256=sha(html_path),
        original_bytes_preserved_exactly_after_removing_additions=True,
        original_script_blocks_preserved=True, added_script_blocks=0,
        image_attributes_preserved=True, new_ids=len(new_ids), added_broken_fragment_links=0,
        existing_unresolved_fragment_links=sorted(missing_before),
        browser_rendering="not tested; prior access restriction was not bypassed",
        new_navigation_targets=["direct-q-ranking", "action-only-results"])
    (target / "static_qa.json").write_text(json.dumps(qa, indent=2)+"\n", encoding="utf-8")
    release = ROOT / "release_reviewed"
    release.mkdir(exist_ok=False)
    # Compact evidence supports reanalysis; no checkpoint binaries or raw upload helpers.
    sources = []
    for folder, prefix in ((ROOT, "direct_q_diagnostic_20261021_v2"),
                           (ROOT.parent / "direct_q_diagnostic_20261021_v1", "preserved_v1_failure"),
                           (ROOT.parent / "direct_q_plan_20261021_v1", "mechanical_stage")):
        for p in sorted(folder.rglob("*")):
            if not p.is_file(): continue
            rel = p.relative_to(folder)
            if any(part in {"__pycache__", "release", "presentation", "release_reviewed", "presentation_reviewed"} for part in rel.parts): continue
            sources.append((p, prefix+"/"+rel.as_posix()))
    manifest = {name: dict(sha256=sha(p), bytes=p.stat().st_size) for p,name in sources}
    archive = release / "ThrusterChain_direct_Q_diagnostic_evidence_20261021.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for p,name in sources: z.write(p,name)
        z.writestr("EVIDENCE_MANIFEST.json", json.dumps(manifest, indent=2)+"\n")
        z.writestr("SCOPE.txt", "Analysis-only evidence. No checkpoint binaries or complete historical source closure.\nSee direct_q_diagnostic_20261021_v2/PROVENANCE.md and report_assets/RESULTS.md.\n")
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,meta in manifest.items():
            assert hashlib.sha256(z.read(name)).hexdigest() == meta["sha256"]
    receipt = dict(html=dict(path=str(html_path), bytes=html_path.stat().st_size, sha256=sha(html_path)),
        evidence_zip=dict(path=str(archive), bytes=archive.stat().st_size, sha256=sha(archive), members=len(manifest)+2),
        static_qa=qa, library_delivery="not attempted for these new artifacts; prior action-only transfer remains blocked")
    (release / "delivery_receipt.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))

if __name__ == "__main__":
    main()
