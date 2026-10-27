"""Copy the weighting test's data and the run records this study reads into the working copy, checking every copied
file against the weighting test's own SHA-256 records (plan section 3.1, "Data copy"; stop rule S11, plan section 6.1).
No learning, no BCA code, and no learned policy's return: the evaluation files are copied and hashed as bytes and never
parsed here.

An experiment-local attachment (plan section 3.1): it changes none of the repository's declared methods. The source,
WSL /root/bca-shift, is only read: every file is copied with `cp -p` (never moved), nothing runs with the source as
working directory, and every source file's size and modification time are checked again after the copy.

Where the copies go (LAYOUT; paths of the source relative to /root/bca-shift, of the copy relative to
runs/shift_hosts/source/ in this working copy):
  runs/shift_iw_v5/<rest>              -> <rest>             the weighting test's data root becomes source/ itself, so
                                                             experiments.shift_iw.data reads source/ as its root
                                                             (data_dir, banks_dir; data.py:217-226) and the start banks
                                                             are at source/start_banks/ (evaluate.py reads them there)
  runs/shift_iw_v5_explore/<rest>      -> explore/<rest>     the exploratory BCA arms of the reference row
  outputs/shift_iw/explore/RESULTS.json -> explore/RESULTS.json
What is copied:
  runs/shift_iw_v5/<quality>/data/seed<k>/        every file of the seed's data (experiments/shift_iw/data.py:44-49),
                                                  quality medium-expert and medium, seeds 0-4
  runs/shift_iw_v5/start_banks/                   the shared start banks and their anchors (data.py:46, 662-678)
  runs/shift_iw_v5/<quality>/host/{gap,full}/seed<k>/record.json
                                                  the weighting test's TD3+BC host runs: the kept-row counts and hashes
                                                  stop rule S11 compares with (plan section 2.2), and the reference row
  runs/shift_iw_v5/<quality>/evaluate/seed<k>/hosts.json
                                                  the reference row's "alone" evaluations (plan section 5.1)
  runs/shift_iw_v5/<quality>/band/gap/seed<k>/record.json
                                                  the band record, which holds the estimated-weight classifier that
                                                  prepare.py must reproduce, as the weighting test's own runner demands
                                                  (experiments/shift_iw/run.py:762-766)
  runs/shift_iw_v5_explore/<quality>/bca/{unweighted,exact_strong}/seed<k>/record.json
                                                  the reference row's standard and IW BCA runs
  runs/shift_iw_v5_explore/<quality>/evaluate/seed<k>/bca.json
                                                  the reference row's BCA evaluations (RESULTS.json lists them
                                                  as inputs)
  outputs/shift_iw/explore/RESULTS.json           the weighting test's exploratory results
each record with its .sha256 sidecar. Snapshots and band networks (.msgpack) are not copied: nothing here reads them.

How each file is checked; the copy and the source must both match, and the copy must equal the source byte for byte:
  <name>.hdf5            the file_sha256 of <name>.hdf5.manifest.json (data.py:158-176); after the copy, every array
                         is checked too, by shift_iw.data.load_seed and load_start_banks (data.py:179-196, 719-737)
  <name> with a sidecar  the digest in <name>.sha256 (data.py:199-206; the run records alike)
  RESULTS.json           its git blob in this working copy's HEAD (it came with the weighting test's commits); it has
                         no sidecar
  manifests, sidecars    nothing records their hashes; byte equality with the source is the check, and build.json's
  and build.json         world_json_sha256 must equal world.json's SHA-256 (build.json holds timings only, data.py:49)
Cross-checks: world.json's own list of file hashes (data.py:728-730), the world_json_sha256 of each data stage.json and
build.json, the start banks' stage.json against start_banks.json, and the ball of medium-expert and medium being the
same for each seed (the expert's ball, data.py:422-425). The hdf5 manifests' code hashes are compared with the code in
this working copy as information only. The destination must hold exactly the listed copies (and the copy manifest):
any other file there fails the copy (`--prune` removes such files, inside runs/shift_hosts/source/ only).

The copy manifest runs/shift_hosts/source/copy_manifest.json (with a sidecar) lists every file: source, copy, size,
SHA-256, the recorded hash and where it is recorded, and the outcome. prepare.py reads only files this manifest lists
as verified, and checks their SHA-256 again before reading (require_verified).

Run (data environment, from the repository root):
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu taskset -c 20-21 \\
      /opt/bca-shift-venv/bin/python -m experiments.shift_hosts.data_copy [--prune]
The exit status is 0 only when every check passed; otherwise the manifest says which failed (stop rule S11).
"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from experiments.shift_hosts import handoff as H

SCHEMA = "shift-hosts-copy-v2"
PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[1]                                  # /root/bca-hosts
SOURCE = Path("/root/bca-shift")                           # read only, never written
DEST = REPO / "runs" / "shift_hosts" / "source"
DEST_REL = "runs/shift_hosts/source"                       # DEST relative to the repository
MANIFEST = DEST / "copy_manifest.json"
V5 = "runs/shift_iw_v5"
EXPLORE = "runs/shift_iw_v5_explore"
RESULTS = "outputs/shift_iw/explore/RESULTS.json"
# (source prefix or file, where it lands under DEST); the first match wins (module docstring)
LAYOUT = ((V5 + "/", ""), (EXPLORE + "/", "explore/"), (RESULTS, "explore/RESULTS.json"))
QUALITIES = ("medium-expert", "medium")
SEEDS = (0, 1, 2, 3, 4)
BCA_ARMS = ("unweighted", "exact_strong")
HOST_KINDS = ("gap", "full")
# every file of a seed's data folder (data.py:44-49, 546-594) and of the start banks (data.py:662-678); the folders
# must hold exactly these, so nothing is silently left out
DATA_FILES = ("build.json", "samples.hdf5", "samples.hdf5.manifest.json", "stage.json", "stage.json.sha256",
              "test_mild.hdf5", "test_mild.hdf5.manifest.json", "test_strong.hdf5", "test_strong.hdf5.manifest.json",
              "train_law.hdf5", "train_law.hdf5.manifest.json", "truth.hdf5", "truth.hdf5.manifest.json",
              "world.json", "world.json.sha256")
START_BANK_FILES = ("stage.json", "stage.json.sha256", "start_banks.hdf5", "start_banks.hdf5.manifest.json",
                    "start_banks.json", "start_banks.json.sha256")
CHUNK = 1 << 22


class CopyError(RuntimeError):
    """Stop rule S11: a copied file differs from its recorded SHA-256, or the copy is missing or not verified."""


# ------------------------------------------------------------------------------------------------ layout and paths

def dest_rel(rel):
    """Where a source file (rel: its path relative to /root/bca-shift) lands, relative to DEST (LAYOUT)."""
    for prefix, target in LAYOUT:
        if prefix.endswith("/") and rel.startswith(prefix):
            return target + rel[len(prefix):]
        if rel == prefix:
            return target
    raise CopyError(f"S11: no place in the copy layout for {rel}")


def copied(rel):
    """The copy of a source file (rel: its path relative to /root/bca-shift)."""
    return DEST / dest_rel(rel)


def copy_rel(rel):
    """The copy's path relative to the repository (runs/shift_hosts/source/...)."""
    return f"{DEST_REL}/{dest_rel(rel)}"


def v5_root():
    """The copied root that experiments.shift_iw.data reads as its root (data_dir, banks_dir; data.py:217-226)."""
    return DEST


def seed_dir(quality, seed):
    return DEST / quality / "data" / f"seed{int(seed)}"


def start_banks_dir():
    """runs/shift_hosts/source/start_banks (data.banks_dir(v5_root()); evaluate.py's COPIED_BANKS)."""
    return DEST / "start_banks"


def seed_rel(quality, seed, name):
    return f"{V5}/{quality}/data/seed{int(seed)}/{name}"


def host_record_rel(quality, kind, seed):
    return f"{V5}/{quality}/host/{kind}/seed{int(seed)}/record.json"


def band_record_rel(quality, seed):
    return f"{V5}/{quality}/band/gap/seed{int(seed)}/record.json"


def hosts_eval_rel(quality, seed):
    return f"{V5}/{quality}/evaluate/seed{int(seed)}/hosts.json"


def explore_record_rel(quality, arm, seed):
    return f"{EXPLORE}/{quality}/bca/{arm}/seed{int(seed)}/record.json"


def explore_eval_rel(quality, seed):
    return f"{EXPLORE}/{quality}/evaluate/seed{int(seed)}/bca.json"


def seed_rels(quality, seed):
    """The copied files prepare.py reads for one seed: its data folder, both host records and the band record."""
    rels = [seed_rel(quality, seed, n) for n in DATA_FILES]
    rels += [host_record_rel(quality, k, seed) for k in HOST_KINDS] + [band_record_rel(quality, seed)]
    return rels + [r + ".sha256" for r in rels[len(DATA_FILES):]]


# ------------------------------------------------------------------------------------------------ hashing

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def git_blob_id(path):
    """git's blob id of a file (SHA-1 of "blob <size>\\0" and the bytes), computed here; nothing is written."""
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _git(*args):
    """A read-only git command in this working copy (never in the source)."""
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True,
                          env=dict(os.environ, GIT_OPTIONAL_LOCKS="0")).stdout


def recorded_blob(rel):
    """The blob id git records for rel in this working copy's HEAD, HEAD itself, and the last commit that changed rel
    (read-only `git ls-tree` and `git log`)."""
    out = _git("ls-tree", "HEAD", "--", rel).split()
    return dict(blob=out[2] if len(out) >= 4 and out[1] == "blob" else None, head=_git("rev-parse", "HEAD").strip(),
                last_commit=_git("log", "-1", "--format=%H %s", "--", rel).strip())


def sidecar_digest(path):
    """(digest, name) of a sha256sum-format sidecar."""
    words = Path(path).read_text(encoding="utf8").split()
    return (words[0] if words else None), (words[1].lstrip("*") if len(words) > 1 else None)


def stat_of(path):
    s = os.stat(path)
    return dict(bytes=int(s.st_size), mtime_ns=int(s.st_mtime_ns))


# ------------------------------------------------------------------------------------------------ the list

def listing(rel_dir, expected):
    """The files of a source folder, which must be exactly `expected` (CopyError otherwise)."""
    folder = SOURCE / rel_dir
    if not folder.is_dir():
        raise CopyError(f"S11: the source folder {folder} does not exist")
    found = sorted(p.name for p in folder.iterdir())
    if found != sorted(expected) or any(not (folder / n).is_file() for n in found):
        raise CopyError(f"S11: {folder} holds {found}, expected exactly {sorted(expected)}")
    return [f"{rel_dir}/{n}" for n in sorted(expected)]


def file_list(qualities=QUALITIES, seeds=SEEDS):
    """Every file to copy, as paths relative to /root/bca-shift (module docstring)."""
    rels = listing(f"{V5}/start_banks", START_BANK_FILES)
    for q in qualities:
        for k in seeds:
            rels += listing(f"{V5}/{q}/data/seed{int(k)}", DATA_FILES)
            records = [host_record_rel(q, kind, k) for kind in HOST_KINDS]
            records += [hosts_eval_rel(q, k), band_record_rel(q, k)]
            records += [explore_record_rel(q, arm, k) for arm in BCA_ARMS] + [explore_eval_rel(q, k)]
            for r in records:
                rels += [r, r + ".sha256"]
    rels += [RESULTS]
    places = [dest_rel(r) for r in rels]
    if len(set(places)) != len(places):
        raise CopyError("S11: two source files would land on the same copy")
    return rels


# ------------------------------------------------------------------------------------------------ checks

def expected_hash(rel, rels):
    """(how, expected digest or None, where it is recorded) of one file, read from the COPIED record files (which are
    themselves checked byte for byte against the source)."""
    name = rel.rsplit("/", 1)[-1]
    if rel == RESULTS:
        rec = recorded_blob(rel)
        return "git blob", rec["blob"], (f"git ls-tree HEAD ({rec['head']}) of {REPO}; last changed by "
                                         f"{rec['last_commit']}")
    if name.endswith(".hdf5"):
        man = rel + ".manifest.json"
        if man not in rels:
            raise CopyError(f"S11: {rel} has no manifest in the list")
        body = json.loads(copied(man).read_text())
        if body.get("file") != name:
            raise CopyError(f"S11: {man} describes {body.get('file')!r}, not {name}")
        return "sha256", body.get("file_sha256"), man
    if name.endswith(".sha256") or name.endswith(".manifest.json"):
        return "bytes", None, "no recorded hash (a record file itself): equal to the source byte for byte"
    if rel + ".sha256" in rels:
        digest, named = sidecar_digest(copied(rel + ".sha256"))
        if named is not None and named != name:
            raise CopyError(f"S11: {rel}.sha256 names {named!r}, not {name}")
        return "sha256", digest, rel + ".sha256"
    if name == "build.json":
        return "bytes", None, "not hashed by the weighting test (timings only, data.py:49): equal to the source"
    raise CopyError(f"S11: no rule checks {rel}")


def cross_checks(qualities, seeds):
    """Hashes the weighting test recorded inside its own files (module docstring), on the copies."""
    out, ok = [], True

    def add(what, expected, got):
        nonlocal ok
        good = expected is not None and expected == got
        ok &= good
        out.append(dict(check=what, expected=expected, found=got, ok=bool(good)))

    sb = start_banks_dir()
    add("start_banks/stage.json start_banks_json_sha256 = SHA-256 of start_banks.json",
        json.loads((sb / "stage.json").read_text()).get("start_banks_json_sha256"),
        sha256_file(sb / "start_banks.json"))
    balls = {}
    for q in qualities:
        for k in seeds:
            d = seed_dir(q, k)
            world = json.loads((d / "world.json").read_text())
            world_sha = sha256_file(d / "world.json")
            for name, digest in sorted(world["files"].items()):
                add(f"{q} seed{k} world.json files[{name}] = SHA-256 of {name}", digest, sha256_file(d / name))
            for rec in ("stage.json", "build.json"):
                add(f"{q} seed{k} {rec} world_json_sha256 = SHA-256 of world.json",
                    json.loads((d / rec).read_text()).get("world_json_sha256"), world_sha)
            balls.setdefault(int(k), {})[q] = world["gap"]
    for k, by_q in sorted(balls.items()):
        names = list(by_q)
        for other in names[1:]:
            add(f"seed{k}: {other}'s ball (world.json gap) = {names[0]}'s (both are the expert's ball)",
                json.dumps(by_q[names[0]], sort_keys=True), json.dumps(by_q[other], sort_keys=True))
    return out, ok


def deep_checks(qualities, seeds):
    """Every array of every copied hdf5 file against its manifest, by the weighting test's own readers
    (data.load_seed: files, arrays, meta, world.json and its file list; data.load_start_banks). Needs the data
    environment (experiments.shift refuses numpy other than 2.2.6)."""
    try:
        from experiments.shift_iw import data as D
    except Exception as error:         # not the data environment: recorded, and the copy is not verified
        return [dict(check="import experiments.shift_iw.data", ok=False, error=repr(error))], False
    out, ok = [], True
    try:
        if D.banks_dir(v5_root()) != start_banks_dir():
            raise CopyError(f"data.banks_dir of the copied root is {D.banks_dir(v5_root())}, not {start_banks_dir()}")
        banks, record = D.load_start_banks(D.banks_dir(v5_root()))
        out.append(dict(check="data.load_start_banks", ok=True, banks={k: list(v.shape) for k, v in banks.items()}))
    except Exception as error:
        ok = False
        out.append(dict(check="data.load_start_banks", ok=False, error=repr(error)))
    for q in qualities:
        for k in seeds:
            try:
                if D.data_dir(v5_root(), q) / f"seed{int(k)}" != seed_dir(q, k):
                    raise CopyError(f"data.data_dir of the copied root does not give {seed_dir(q, k)}")
                sd = D.load_seed(k, D.data_dir(v5_root(), q))
                out.append(dict(check=f"data.load_seed {q} seed{k}", ok=True, train_log_rows=sd.train_log.n_rows,
                                test_rows={t: v.n_rows for t, v in sd.tests.items()}))
            except Exception as error:
                ok = False
                out.append(dict(check=f"data.load_seed {q} seed{k}", ok=False, error=repr(error)))
    return out, ok


def code_provenance(qualities, seeds):
    """The code hashes each hdf5 manifest recorded, against the files in this working copy (information only: the
    data are checked by their own hashes)."""
    recorded = {}
    paths = [start_banks_dir() / "start_banks.hdf5.manifest.json"]
    paths += [seed_dir(q, k) / f"{n}.hdf5.manifest.json" for q in qualities for k in seeds
              for n in ("train_law", "test_mild", "test_strong", "samples", "truth")]
    for p in paths:
        for rel, digest in json.loads(p.read_text()).get("code_sha256", {}).items():
            recorded.setdefault(rel, set()).add(digest)
    out = {}
    for rel, digests in sorted(recorded.items()):
        here = REPO / rel
        now = sha256_file(here) if here.is_file() else None
        out[rel] = dict(recorded=sorted(digests), in_working_copy=now, same=(len(digests) == 1 and now in digests))
    return out


# ------------------------------------------------------------------------------------------------ copy

def guard_paths():
    """Refuse a destination inside the source, or one that is not runs/shift_hosts/source of this working copy."""
    dest, src = DEST.resolve(), SOURCE.resolve()
    if dest == src or src in dest.parents:
        raise CopyError(f"refused: the destination {dest} lies inside the read-only source {src}")
    if REPO.resolve() == src or src in REPO.resolve().parents:
        raise CopyError(f"refused: this working copy {REPO} lies inside the read-only source {src}")
    if dest != (REPO / DEST_REL).resolve() or (REPO / "runs" / "shift_hosts").resolve() not in dest.parents:
        raise CopyError(f"refused: {dest} is not {REPO / DEST_REL}")


def unlisted(rels):
    """Files under DEST that are neither a listed copy nor the copy manifest (and its sidecar), as paths relative to
    DEST; symbolic links count as unlisted and are never followed."""
    allowed = {dest_rel(r) for r in rels} | {MANIFEST.name, H.sidecar(MANIFEST).name}
    out = []
    if DEST.exists():
        for root, dirs, files in os.walk(DEST, followlinks=False):
            for name in files + [d for d in dirs if (Path(root) / d).is_symlink()]:
                rel = (Path(root) / name).relative_to(DEST).as_posix()
                if rel not in allowed or (Path(root) / name).is_symlink():
                    out.append(rel)
    return sorted(out)


def prune(rels, log=print):
    """Remove every unlisted file under DEST (an earlier layout's copies, for example), then empty folders. Only
    inside runs/shift_hosts/source of this working copy (guard_paths); the source is never touched."""
    guard_paths()
    removed = []
    for rel in unlisted(rels):
        p = DEST / rel
        if DEST.resolve() not in p.parent.resolve().parents and p.parent.resolve() != DEST.resolve():
            raise CopyError(f"refused to remove {p}: not inside {DEST}")
        p.unlink()
        removed.append(rel)
    for root, dirs, files in sorted(os.walk(DEST, topdown=False), key=lambda x: -len(x[0])):
        if Path(root) != DEST and not os.listdir(root):
            os.rmdir(root)
    if removed:
        log(f"pruned {len(removed)} unlisted files from {DEST}")
    return removed


def copy_one(src, dst):
    """cp -p (mode, ownership and times kept); the source is never moved or changed."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cp", "-p", "--", str(src), str(dst)], check=True)


def run(qualities=QUALITIES, seeds=SEEDS, *, prune_unlisted=False, log=print):
    """Copy, check and write the copy manifest. Returns the manifest (all_ok True only if every check passed)."""
    guard_paths()
    started, started_utc = time.perf_counter(), datetime.now(timezone.utc).isoformat(timespec="seconds")
    rels = file_list(qualities, seeds)
    pruned = prune(rels, log) if prune_unlisted else []
    before = {rel: stat_of(SOURCE / rel) for rel in rels}
    entries, ok = [], True
    for rel in rels:
        src, dst = SOURCE / rel, copied(rel)
        if dst.is_symlink():
            raise CopyError(f"S11: {dst} is a symbolic link; a copy must be a plain file (run with --prune)")
        src_sha = sha256_file(src)
        if dst.is_file() and sha256_file(dst) == src_sha:
            action = "already present, equal to the source"
        else:
            action = "copied" if not dst.exists() else "copied again (the earlier copy differed)"
            copy_one(src, dst)
        entries.append(dict(rel=rel, source=str(src), copy=copy_rel(rel), destination=str(dst), action=action,
                            source_sha256=src_sha, **before[rel]))
    rel_set = set(rels)
    for e in entries:          # after every copy, so each check reads copied record files
        dst = Path(e["destination"])
        how, expected, where = expected_hash(e["rel"], rel_set)
        copy_sha = sha256_file(dst)
        e.update(sha256=copy_sha, copy_bytes=dst.stat().st_size, check=how, recorded_in=where)
        good = copy_sha == e["source_sha256"] and e["copy_bytes"] == e["bytes"]
        if how == "sha256":
            e["expected"] = expected
            good &= expected is not None and copy_sha == expected and e["source_sha256"] == expected
        elif how == "git blob":
            e["expected"] = expected
            e["git_blob"] = git_blob_id(dst)
            good &= expected is not None and e["git_blob"] == expected and git_blob_id(e["source"]) == expected
        e["ok"] = bool(good)
        ok &= good
        if not good:
            log(f"S11 FAILED: {e['rel']}")
    after = {rel: stat_of(SOURCE / rel) for rel in rels}
    changed = sorted(rel for rel in rels if after[rel] != before[rel])
    ok &= not changed
    extra = unlisted(rels)
    ok &= not extra
    crosses, cross_ok = cross_checks(qualities, seeds)
    deeps, deep_ok = deep_checks(qualities, seeds)
    ok = bool(ok and cross_ok and deep_ok)
    manifest = dict(
        schema=SCHEMA, kind="data copy (plan section 3.1; stop rule S11)", all_ok=ok,
        s11="passed" if ok else ("FAILED: see entries, cross_checks, deep_checks, source_changed and "
                                 "unlisted_in_destination"),
        source_root=str(SOURCE), destination_root=str(DEST),
        layout=[dict(source=s, copy=f"{DEST_REL}/{t}") for s, t in LAYOUT],
        qualities=list(qualities), seeds=[int(k) for k in seeds],
        files=len(entries), bytes=int(sum(e["bytes"] for e in entries)),
        by_check={h: sum(1 for e in entries if e["check"] == h) for h in ("sha256", "git blob", "bytes")},
        entries=entries, cross_checks=crosses, deep_checks=deeps, source_changed=changed,
        source_unchanged=not changed, unlisted_in_destination=extra, pruned=pruned,
        code_provenance=code_provenance(qualities, seeds),
        code=dict(data_copy_sha256=sha256_file(__file__), handoff_sha256=sha256_file(H.__file__)),
        environment=dict(python=sys.version.split()[0], executable=sys.executable, host=platform.node(),
                         affinity=sorted(os.sched_getaffinity(0))),
        started_utc=started_utc, seconds=round(time.perf_counter() - started, 2),
        note="evaluation files (hosts.json, bca.json, RESULTS.json) were hashed as bytes and never parsed")
    H.write_json(MANIFEST, manifest)
    log(f"copy: {len(entries)} files, {manifest['bytes'] / 1e6:.1f} MB, "
        f"{manifest['by_check']} -> {'all verified' if ok else 'S11 FAILED'} ({manifest['seconds']:.0f} s); "
        f"manifest {MANIFEST}")
    if extra:
        log(f"S11 FAILED: {len(extra)} unlisted files in {DEST} (run with --prune to remove them): {extra[:5]}")
    return manifest


# ------------------------------------------------------------------------------------------------ readers

def read_manifest():
    """The copy manifest after its sidecar check; CopyError (S11) unless every copy was verified."""
    try:
        manifest = H.read_json(MANIFEST)
    except (H.HandoffError, FileNotFoundError) as error:
        raise CopyError(f"S11: no verified copy manifest ({error}); run experiments.shift_hosts.data_copy") from error
    if manifest.get("schema") != SCHEMA or manifest.get("all_ok") is not True:
        raise CopyError(f"S11: the copy manifest {MANIFEST} does not say every copy was verified")
    return manifest


def require_verified(manifest, rels):
    """{rel: SHA-256} of copied files after checking that the copy manifest lists each as verified, at the place the
    layout gives it, and that the file on disk still has that SHA-256 (CopyError, stop rule S11, otherwise)."""
    by_rel = {e["rel"]: e for e in manifest["entries"]}
    out = {}
    for rel in rels:
        e = by_rel.get(rel)
        if e is None or e.get("ok") is not True:
            raise CopyError(f"S11: {rel} is not a verified copy")
        if e.get("copy") != copy_rel(rel):
            raise CopyError(f"S11: the manifest puts {rel} at {e.get('copy')}, not {copy_rel(rel)}")
        if sha256_file(copied(rel)) != e["sha256"]:
            raise CopyError(f"S11: {copied(rel)} changed after it was copied and verified")
        out[rel] = e["sha256"]
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Copy the weighting test's data and run records with SHA-256 checks (S11).")
    p.add_argument("--qualities", nargs="+", default=list(QUALITIES), choices=QUALITIES)
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--prune", action="store_true",
                   help="first remove files under runs/shift_hosts/source that are not in this run's copy list "
                        "(with --qualities or --seeds, the other qualities' and seeds' copies too)")
    a = p.parse_args(argv)
    manifest = run(tuple(a.qualities), tuple(a.seeds), prune_unlisted=a.prune)
    return 0 if manifest["all_ok"] else 3


if __name__ == "__main__":
    sys.exit(main())
