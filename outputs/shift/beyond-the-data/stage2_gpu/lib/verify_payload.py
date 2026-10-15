"""Place the payload and the committed Stage 2 files where the pinned code expects them, and verify every hash.
Standard library only; reads and copies files, never deletes or overwrites a file that differs. Called by restore.sh.

  1. payload/MANIFEST.json matches its .sha256 sidecar; every payload file and every committed source has the SHA-256
     the manifest records; payload/files/ holds nothing the manifest does not list.
  2. Each file is copied to its place (repository paths under /home/user/bca, and the two folders under /tmp that the
     pinned code's paths are fixed to). A place that already holds the same bytes is left alone; a place that holds
     different bytes is refused.
  3. The late files (payload/late/), when present: their index, sidecars, kinds and fingerprint; placed the same way.
  4. The pre-registration's own chain of pins, read from the placed files, independent of the manifest: the import
     record names this prereg.json; prereg.json pins the code manifest, which pins every Stage 2 file and every
     repository file Stage 2 imports; prereg.json pins the room test, the fallback diagnostic, the stand-ins and the
     detection script by their manifests, which pin their files; the gate's pre-registration pins its records.

Exit status 0 when everything verifies (late files may be absent unless --require-late), 1 otherwise.
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True

ROOT = Path("/home/user/bca")
STAGE2_REL = "outputs/shift/beyond-the-data/stage2"
FINGERPRINT_LEN = 16


class Refused(Exception):
    pass


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(p):
    return sha_bytes(Path(p).read_bytes())


def sidecar_ok(path):
    """True when '<path>.sha256' exists and names the SHA-256 of path's bytes."""
    path = Path(path)
    side = path.with_name(path.name + ".sha256")
    if not path.exists() or not side.exists():
        return False
    return side.read_text(encoding="utf8").split()[:1] == [sha_file(path)]


def read_verified_json(path, what):
    path = Path(path)
    if not path.exists():
        raise Refused(f"{what} is missing: {path}")
    if not sidecar_ok(path):
        raise Refused(f"{what} differs from its .sha256 sidecar: {path}")
    return json.loads(path.read_text(encoding="utf8"))


def target_path(target, repo):
    return Path(target) if str(target).startswith("/") else Path(repo) / target


def place(data, dest, placed):
    """Copy bytes to dest unless dest already holds exactly them; refuse when it holds anything else."""
    dest = Path(dest)
    if dest.is_symlink():
        raise Refused(f"{dest} is a symbolic link; refusing to write through it")
    if dest.exists():
        if dest.read_bytes() != data:
            raise Refused(f"{dest} exists with different contents; it is never overwritten (if it is a stale copy "
                          f"under /tmp, delete /tmp/claude-0 and run restore.sh again; under runs/, ask the cloud)")
        placed["unchanged"] += 1
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(dest.parent), prefix=".restore-")
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
    os.chmod(tmp, 0o644)
    os.replace(tmp, dest)
    placed["copied"] += 1


# ------------------------------------------------------------------------------------------------ steps 1-3

def check_and_place(pkg, repo, do_place):
    manifest_path = pkg / "payload" / "MANIFEST.json"
    man = read_verified_json(manifest_path, "the payload manifest")
    if man.get("kind") != "stage2-offcloud-payload-manifest":
        raise Refused("payload/MANIFEST.json is not a Stage 2 payload manifest")
    if Path(man["repository_root"]) != Path(repo):
        raise Refused(f"the payload is for a repository at {man['repository_root']}, not {repo}")
    files_dir = pkg / "payload" / "files"
    listed = {e["name"] for e in man["payload_files"]}
    present = {p.name for p in files_dir.iterdir()} if files_dir.exists() else set()
    if present - listed:
        raise Refused(f"payload/files holds files the manifest does not list: {sorted(present - listed)[:5]}")
    placed = dict(copied=0, unchanged=0)
    todo = []
    for e in man["payload_files"]:
        p = files_dir / e["name"]
        if not p.exists():
            raise Refused(f"payload file missing: {p}")
        data = p.read_bytes()
        if sha_bytes(data) != e["sha256"]:
            raise Refused(f"payload file differs from the manifest: {p}")
        todo.append((data, target_path(e["target"], repo)))
    for e in man["committed_sources"]:
        p = Path(repo) / e["source"]
        if not p.exists():
            raise Refused(f"committed file missing (pull the branch): {p}")
        data = p.read_bytes()
        if sha_bytes(data) != e["sha256"]:
            raise Refused(f"committed file differs from the manifest (pull the branch, or it was changed): {p}")
        for t in e["targets"]:
            todo.append((data, target_path(t, repo)))
    if do_place:
        for data, dest in todo:
            place(data, dest, placed)
    else:
        for data, dest in todo:
            if not dest.exists() or dest.read_bytes() != data:
                raise Refused(f"{dest} is not in place (run restore.sh)")
            placed["unchanged"] += 1
    return man, placed


def late_files(pkg, repo, man, do_place):
    """('absent', reason) or ('present', facts) for the cloud's late files; refuses anything inconsistent."""
    spec = man["late_files"]
    folder = pkg / spec["folder"]
    index = folder / spec["index"]
    if not index.exists():
        return "absent", "payload/late/LATE.json is not there yet (the cloud commits it after seeds 0-4 decide)"
    lt = read_verified_json(index, "the late-file index")
    fp = man["prereg_sha256"]
    if lt.get("kind") != "stage2-offcloud-late-files" or lt.get("prereg_sha256") != fp:
        raise Refused("payload/late/LATE.json is not this pre-registration's late-file index")
    names = dict(lt.get("files") or {})
    if set(names) - set(spec["files"]):
        raise Refused(f"LATE.json lists unexpected files: {sorted(set(names) - set(spec['files']))}")
    present = {p.name for p in folder.iterdir()}
    allowed = {spec["index"], spec["index"] + ".sha256"} | set(names) | {n + ".sha256" for n in names}
    if present - allowed:
        raise Refused(f"payload/late holds files LATE.json does not list: {sorted(present - allowed)}")
    bodies = {}
    for name, sha in names.items():
        p = folder / name
        body = read_verified_json(p, f"late file {name}")
        if sha_file(p) != sha:
            raise Refused(f"late file {name} differs from LATE.json")
        bodies[name] = body
    bs = bodies.get("best_standin.json")
    if bs is None or bs.get("kind") != "stage2-best-standin" or bs.get("prereg_sha256") != fp:
        raise Refused("best_standin.json is missing or is not this pre-registration's")
    uni = bodies.get("marks_exploratory_uniform.json")

    def marks_ok(b, version):
        return (b is not None and b.get("kind") == "stage2-detection-marks" and b.get("group") == "exploratory"
                and b.get("version") == version and b.get("seeds") == [0, 1, 2, 3, 4] and b.get("prereg_sha256") == fp)

    if not marks_ok(uni, "uniform"):
        raise Refused("marks_exploratory_uniform.json is missing or is not this pre-registration's exploratory "
                      "uniform detection marks")
    failed = bool(uni["marks"].get("direction_or_size_failed"))
    wei = bodies.get("marks_exploratory_weighted.json")
    if failed and not marks_ok(wei, "weighted"):
        raise Refused("the uniform version failed direction or size on seeds 0-4, so the weighted version's "
                      "exploratory marks must come too, and they are missing or not this pre-registration's")
    if not failed and wei is not None:
        raise Refused("the uniform version passed on seeds 0-4, yet weighted marks were sent (never declared)")
    if failed:
        chosen = None if wei["marks"].get("direction_or_size_failed") else "weighted"
    else:
        chosen = "uniform"
    placed = dict(copied=0, unchanged=0)
    for name in names:
        for suffix in ("", ".sha256"):
            data = (folder / (name + suffix)).read_bytes()
            dest = target_path(spec["files"][name], repo)
            dest = dest.with_name(dest.name + suffix)
            if do_place:
                place(data, dest, placed)
            elif not dest.exists() or dest.read_bytes() != data:
                raise Refused(f"{dest} is not in place (run restore.sh)")
    return "present", dict(best_standin=bs.get("best"), uniform_failed_direction_or_size=failed,
                           chosen_version=chosen, files=sorted(names), placed=placed)


# ------------------------------------------------------------------------------------------------ step 4

def chain_checks(repo, man):
    """The pre-registration's own pins, from the placed files. Returns a list of plain-language lines; raises
    Refused at the first problem."""
    R = Path(repo)
    s2 = R / "runs" / "shift_explore" / "stage2"
    lines = []
    fp = man["prereg_sha256"]
    prefix = fp[:FINGERPRINT_LEN]
    prereg = read_verified_json(s2 / "prereg.json", "Stage 2's prereg.json")
    if sha_file(s2 / "prereg.json") != fp:
        raise Refused("runs/shift_explore/stage2/prereg.json is not the approved pre-registration")
    imp_path = R / STAGE2_REL / f"import_{prefix}.json"
    imp = read_verified_json(imp_path, "the committed import record")
    approval_path = s2 / "approval.json"
    approval = read_verified_json(approval_path, "runs/shift_explore/stage2/approval.json")
    if imp.get("kind") != "stage2-import-record" or imp.get("prereg_sha256") != fp:
        raise Refused("the import record does not name this pre-registration")
    if imp.get("runs_side_approval_sha256") != sha_file(approval_path):
        raise Refused("approval.json is not the file the import record names")
    local = R / imp.get("approval_file", "")
    if not local.exists() or sha_file(local) != imp.get("approval_file_sha256"):
        raise Refused("the user's committed approval file is missing or differs from the import record")
    if approval.get("prereg_sha256") != fp or approval.get("mode") != "local-terminal" or \
            (approval.get("git") or {}).get("commit") != imp.get("commit"):
        raise Refused("approval.json does not approve this pre-registration through the user's commit")
    lines.append(f"approval: prereg.json {prefix}..., approved by commit {imp['commit'][:12]} "
                 f"({imp.get('author')}), import record committed")
    code = prereg["code"]
    manifest_path = s2 / "code_manifest.json"
    cman = read_verified_json(manifest_path, "Stage 2's code manifest")
    if sha_file(manifest_path) != code["code_manifest_sha256"]:
        raise Refused("Stage 2's code manifest is not the one prereg.json pins")
    if cman["files"] != code["files"]:
        raise Refused("Stage 2's code manifest lists other file hashes than prereg.json")
    code_dir = Path(cman["source_dir"])
    if str(code_dir) != man["stage2_code_dir"]:
        raise Refused("the manifest's Stage 2 code folder is not the pinned one")
    for name, h in cman["files"].items():
        for p in (s2 / "code" / name, code_dir / name):
            if not p.exists() or sha_file(p) != h:
                raise Refused(f"{p} differs from the pinned SHA-256")
    lines.append(f"Stage 2 code: {len(cman['files'])} files match the pinned manifest in {code_dir} and runs/")
    if cman.get("repository_files") != code.get("repository_files"):
        raise Refused("the code manifest's repository files differ from prereg.json's")
    bad = [rel for rel, h in cman["repository_files"].items() if not (R / rel).exists() or sha_file(R / rel) != h]
    if bad:
        raise Refused(f"repository files differ from the pinned SHA-256 (pull the branch; never edit them): {bad}")
    lines.append(f"repository files Stage 2 imports: {len(cman['repository_files'])} match")
    pins = prereg["pins"]
    room = R / "runs" / "shift_explore" / "room"
    rman = read_verified_json(room / "code_manifest.json", "the room test's code manifest")
    if sha_file(room / "code_manifest.json") != pins["room"]["manifest_sha256"]:
        raise Refused("the room test's code manifest is not the pinned one")
    if sha_file(room / "spec.json") != pins["room"]["spec_sha256"] or rman["spec_sha256"] != pins["room"]["spec_sha256"]:
        raise Refused("the room test's spec is not the pinned one")
    harness = Path(pins["room"]["harness_copy"])
    if harness != room / "code" / "harness.py" or sha_file(harness) != pins["room"]["harness_sha256"] or \
            rman["files"]["harness.py"] != pins["room"]["harness_sha256"]:
        raise Refused("the room test's harness copy is not the pinned one")
    pooled = read_verified_json(room / "pooled.json", "the room test's pooled look")
    if pooled.get("kind") != "room-pooled" or pooled.get("spec_sha256") != pins["room"]["spec_sha256"]:
        raise Refused("the room test's pooled look is not the pinned room test's")
    fb = Path(pins["fallback_py"]["path"])
    if str(fb) != man["fallback_path"] or not fb.exists() or sha_file(fb) != pins["fallback_py"]["sha256"]:
        raise Refused(f"the fallback diagnostic at {fb} is not the pinned one")
    lines.append("room test (manifest, spec, harness, pooled look) and fallback diagnostic: match the pins")
    gate_files = {}
    for comp, mkey, extra in (("detect", "detect_manifest_sha256", "detect_module"),
                              ("standins", "standins_manifest_sha256", "standins_py")):
        d = R / "runs" / "shift_explore" / comp
        m = read_verified_json(d / "code_manifest.json", f"the {comp} code manifest")
        if sha_file(d / "code_manifest.json") != pins[mkey]:
            raise Refused(f"the {comp} code manifest is not the pinned one")
        bad = [n for n, h in m["files"].items() if not (d / "code" / n).exists() or sha_file(d / "code" / n) != h]
        if bad:
            raise Refused(f"the {comp} code differs from its pinned manifest: {bad}")
        pin = pins[extra]
        if Path(pin["path"]).parent != d / "code" or sha_file(pin["path"]) != pin["sha256"]:
            raise Refused(f"the pinned {comp} module is not in place")
        if comp == "detect":
            spec = read_verified_json(d / "spec.json", "the detection script's spec")
            if sha_file(d / "spec.json") != m["spec_sha256"]:
                raise Refused("the detection script's spec is not the one its manifest names")
            gate_files = spec.get("gate_files_sha256") or {}
        else:
            if m.get("spec_sha256") != pins["standins_marks_spec_sha256"]:
                raise Refused("the stand-ins manifest names another spec than prereg.json pins")
        lines.append(f"{comp}: manifest and {len(m['files'])} files match the pins")
    gate = R / "runs" / "shift"
    gp = read_verified_json(gate / "preregistration.json", "the gate's pre-registration")
    gsha = sha_file(gate / "preregistration.json")
    rspec = json.loads((room / "spec.json").read_text(encoding="utf8"))
    if gsha != man["gate_prereg_sha256"] or gsha != rspec.get("gate_prereg_sha256") or \
            gsha != cman["repository_files"].get("runs/shift/preregistration.json"):
        raise Refused("the gate's pre-registration is not the one the room test and Stage 2 pin")
    for rel, h in gate_files.items():
        if rel.startswith("runs/") and sha_file(R / rel) != h:
            raise Refused(f"{rel} differs from the detection spec's record")
    for v, rec in (gp.get("stage0") or {}).items():
        p = gate / "pre" / f"stage0_{v}.json"
        if rec is not None and (not sidecar_ok(p) or sha_file(p) != rec["sha256"]):
            raise Refused(f"{p} is not the record the gate's pre-registration names")
    bc = gate / "pre" / "build_checks.json"
    if not sidecar_ok(bc) or sha_file(bc) != gp["build_checks"]["sha256"]:
        raise Refused("the gate's build_checks.json is not the recorded one")
    for key, rec in gp["data"].items():
        p = gate / "pre" / "data" / key / "record.json"
        if not sidecar_ok(p) or sha_file(p) != rec["record_sha256"]:
            raise Refused(f"the gate's data record {key} is not the recorded one")
    lines.append(f"gate pre-registration {gsha[:16]}...: its stage records and {len(gp['data'])} data records match")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--place", action="store_true", help="copy files into place (otherwise only check them)")
    ap.add_argument("--require-late", action="store_true", help="refuse unless the cloud's late files are present")
    ap.add_argument("--json", help="also write the result as JSON to this file")
    a = ap.parse_args(argv)
    pkg = Path(__file__).resolve().parent.parent
    out = dict(ok=False)
    try:
        man, placed = check_and_place(pkg, ROOT, a.place)
        print(f"PASS payload: {len(man['payload_files'])} files ({man['payload_bytes']} bytes) and "
              f"{len(man['committed_sources'])} committed files verified; {placed['copied']} copied, "
              f"{placed['unchanged']} already in place")
        status, late = late_files(pkg, ROOT, man, a.place)
        lines = chain_checks(ROOT, man)
        for line in lines:
            print(f"PASS {line}")
        if status == "absent":
            print(f"WAIT late files: {late}")
            if a.require_late:
                raise Refused("the cloud's late files (best_standin.json and the exploratory detection marks) are not "
                              "here yet; run 'git pull' once the cloud says they are committed")
        else:
            print(f"PASS late files: best stand-in {late['best_standin']}; version for seeds 5-9: "
                  f"{late['chosen_version'] or 'none passed (detection and deployment check only, no training arm)'}")
        out = dict(ok=True, payload_files=len(man["payload_files"]), placed=placed, late_status=status,
                   late=late if status == "present" else None, prereg_sha256=man["prereg_sha256"], chain=lines)
        code = 0
    except Refused as error:
        print(f"FAIL {error}")
        out = dict(ok=False, error=str(error))
        code = 1
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(out, indent=1, sort_keys=True), encoding="utf8")
    return code


if __name__ == "__main__":
    sys.exit(main())
