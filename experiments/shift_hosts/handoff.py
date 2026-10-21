"""Files passed between the two Python environments, each with SHA-256 sidecars (plan section 3.1, "Two Python
environments, handed off by file"; test (i)).

An experiment-local attachment (plan section 3.1): it changes none of the repository's declared methods. numpy and the
standard library only, so it imports in /opt/bca-venv (numpy 1.26.4) and /opt/bca-shift-venv (numpy 2.2.6) alike. No
process holds both a Unifloral network and the simulator; whatever needs both goes through these files.

One hand-off <stem>.npz is four files:
  <stem>.npz          an uncompressed zip of .npy members (numpy.lib.format version 1.0, no pickles), one per array,
                      named "<array name>.npy", in sorted name order, every member dated 1980-01-01 00:00:00, so the
                      same arrays give the same bytes
  <stem>.npz.sha256   sha256sum line of the .npz ("<64 hex>  <stem>.npz")
  <stem>.json         the manifest: schema, the .npz's name and SHA-256, for every array its dtype (numpy's dtype.str,
                      e.g. "<f4"), shape and the SHA-256 of its C-order bytes, the caller's meta (JSON; inf and NaN as
                      the strings "inf", "-inf", "nan") and the writer (Python, numpy, executable, host)
  <stem>.json.sha256  sha256sum line of the manifest
read() verifies both sidecars, the manifest's .npz hash, the member list and every array's dtype, shape and content
hash, and returns fresh arrays bit for bit as written. Array names: letters, digits, "_", ".", "-" and "/" (for
groups such as "train/obs"), no "..". Dtypes: bool, integers and floats (no objects, strings or records).

write_json / read_json give any JSON record (run records, test results) the same sidecar; plain() makes numpy values
and non-finite floats JSON-ready.
"""

import hashlib
import io
import json
import math
import os
import platform
import re
import sys
import zipfile
from pathlib import Path

import numpy as np

SCHEMA = "shift-hosts-handoff-v1"
FIXED_DATE = (1980, 1, 1, 0, 0, 0)
NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_./-]*$")
KINDS = "biuf"          # bool, signed and unsigned integers, floats


class HandoffError(ValueError):
    """A hand-off file is missing, differs from its sidecar or manifest, or holds something not allowed."""


def sha256_bytes(blob):
    return hashlib.sha256(blob).hexdigest()


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


def array_sha256(a):
    """SHA-256 of an array's C-order bytes (its dtype and shape are recorded beside it)."""
    return sha256_bytes(np.asarray(a).tobytes(order="C"))


def c_order(x):
    """A C-contiguous numpy array of x, keeping its shape (np.ascontiguousarray would turn a 0-d array into shape
    (1,))."""
    a = np.asarray(x)
    return a if a.flags.c_contiguous else np.array(a, order="C")


def sidecar(path):
    return Path(str(path) + ".sha256")


def write_sidecar(path):
    """Write <path>.sha256 in sha256sum format; returns the digest."""
    path = Path(path)
    digest = sha256_file(path)
    sidecar(path).write_text(f"{digest}  {path.name}\n", encoding="utf8")
    return digest


def verify_sidecar(path):
    """The file's SHA-256 after checking it against <path>.sha256 (HandoffError when either is missing or they
    differ)."""
    path = Path(path)
    side = sidecar(path)
    if not path.is_file():
        raise HandoffError(f"{path} does not exist")
    if not side.is_file():
        raise HandoffError(f"{path} has no sidecar {side.name}")
    words = side.read_text(encoding="utf8").split()
    digest = sha256_file(path)
    if not words or words[0] != digest:
        raise HandoffError(f"{path} differs from its sidecar {side.name}")
    return digest


def plain(obj):
    """A JSON-ready copy: numpy scalars and arrays to Python values and lists, tuples to lists, Paths to strings,
    non-finite floats to "inf", "-inf" or "nan"."""
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return plain(obj.tolist())
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return v
    if isinstance(obj, Path):
        return str(obj)
    if obj is None or isinstance(obj, str):
        return obj
    if hasattr(obj, "__array__"):           # JAX arrays and other array-likes
        return plain(np.asarray(obj))
    raise TypeError(f"not JSON-ready: {type(obj).__name__}")


def write_json(path, obj):
    """Write plain(obj) as JSON (sorted keys) and its sidecar; returns the file's SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(plain(obj), indent=1, sort_keys=True, allow_nan=False) + "\n"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf8")
    os.replace(tmp, path)
    return write_sidecar(path)


def read_json(path):
    """A JSON file after checking its sidecar."""
    verify_sidecar(path)
    return json.loads(Path(path).read_text(encoding="utf8"))


def writer():
    return dict(python=sys.version.split()[0], numpy=np.__version__, executable=sys.executable,
                host=platform.node())


def manifest_path(path):
    return Path(path).with_suffix(".json")


def _check_name(name):
    if not isinstance(name, str) or not NAME.match(name) or ".." in name or name.endswith("/"):
        raise HandoffError(f"not an allowed array name: {name!r}")


def _check_array(name, a):
    if a.dtype.kind not in KINDS or a.dtype.fields is not None or a.dtype.hasobject:
        raise HandoffError(f"{name}: dtype {a.dtype} is not allowed (bool, integers and floats only)")


def write(path, arrays, meta=None):
    """Write `arrays` ({name: array-like}, numpy or JAX arrays) to <stem>.npz with its manifest <stem>.json and both
    sidecars. Returns the manifest."""
    path = Path(path)
    if path.suffix != ".npz":
        raise HandoffError(f"a hand-off file ends in .npz, not {path.name}")
    if not arrays:
        raise HandoffError("nothing to hand off")
    entries, specs = [], {}
    for name in sorted(arrays):
        _check_name(name)
        a = c_order(arrays[name])
        _check_array(name, a)
        buf = io.BytesIO()
        np.lib.format.write_array(buf, a, version=(1, 0), allow_pickle=False)
        entries.append((name + ".npy", buf.getvalue()))
        specs[name] = dict(dtype=a.dtype.str, shape=[int(s) for s in a.shape], sha256=array_sha256(a))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for member, blob in entries:
            info = zipfile.ZipInfo(member, date_time=FIXED_DATE)
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16
            zf.writestr(info, blob)
    os.replace(tmp, path)
    npz_sha = write_sidecar(path)
    manifest = dict(schema=SCHEMA, npz=path.name, npz_sha256=npz_sha, arrays=specs, meta=plain(meta or {}),
                    writer=writer())
    write_json(manifest_path(path), manifest)
    return manifest


def read_manifest(path):
    """The manifest of <stem>.npz, after checking its sidecar and schema."""
    manifest = read_json(manifest_path(path))
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise HandoffError(f"{manifest_path(path)} is not a {SCHEMA} manifest")
    return manifest


def read(path, names=None):
    """({name: array}, manifest) of a hand-off file, after every check of the module docstring. `names` (optional)
    reads only those arrays (every member's presence is still checked)."""
    path = Path(path)
    manifest = read_manifest(path)
    digest = verify_sidecar(path)
    if manifest.get("npz") != path.name or manifest.get("npz_sha256") != digest:
        raise HandoffError(f"{path} is not the file its manifest describes")
    specs = manifest["arrays"]
    wanted = list(specs) if names is None else list(names)
    missing = [n for n in wanted if n not in specs]
    if missing:
        raise HandoffError(f"{path} holds no arrays {missing}")
    out = {}
    with zipfile.ZipFile(path) as zf:
        members = sorted(zf.namelist())
        if members != sorted(n + ".npy" for n in specs):
            raise HandoffError(f"{path}: its members differ from the manifest")
        for name in wanted:
            spec = specs[name]
            a = np.lib.format.read_array(io.BytesIO(zf.read(name + ".npy")), allow_pickle=False)
            if a.dtype.str != spec["dtype"] or [int(s) for s in a.shape] != spec["shape"]:
                raise HandoffError(f"{path}: {name} is {a.dtype.str} {list(a.shape)}, the manifest says "
                                   f"{spec['dtype']} {spec['shape']}")
            if array_sha256(a) != spec["sha256"]:
                raise HandoffError(f"{path}: {name}'s content differs from the manifest")
            out[name] = a
    return out, manifest


def group(arrays, prefix):
    """{field: array} of the arrays named "<prefix>/<field>" (e.g. group(arrays, "train") -> obs, action, ...)."""
    p = prefix.rstrip("/") + "/"
    return {k[len(p):]: v for k, v in arrays.items() if k.startswith(p) and "/" not in k[len(p):]}


def flatten_dict(tree, prefix):
    """{"<prefix>/<key>/<key>/...": array} of a nested dict of arrays (a flax parameter tree such as
    {"params": {"Dense_0": {"kernel": ..., "bias": ...}}}), for write(). Keys must be allowed names without "/"."""
    out = {}

    def walk(node, path):
        if isinstance(node, dict) or hasattr(node, "items") and not hasattr(node, "shape"):
            for k, v in node.items():
                if "/" in str(k):
                    raise HandoffError(f"a key holds '/': {k!r}")
                walk(v, f"{path}/{k}")
        else:
            out[path] = np.asarray(node)

    walk(tree, prefix.rstrip("/"))
    return out


def unflatten_dict(arrays, prefix):
    """The nested dict of the arrays named "<prefix>/..." (the inverse of flatten_dict)."""
    p = prefix.rstrip("/") + "/"
    tree = {}
    for name, value in arrays.items():
        if not name.startswith(p):
            continue
        node, keys = tree, name[len(p):].split("/")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
    return tree
