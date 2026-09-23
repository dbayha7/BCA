"""Linux/WSL runtime setup, including reuse of an installed MuJoCo CPU extension."""

from pathlib import Path
import ast
import collections
import collections.abc
import importlib
import importlib.util
import os
import sys


def setup():
    for name in ("Mapping", "MutableMapping"):
        if not hasattr(collections, name):
            setattr(collections, name, getattr(collections.abc, name))
    mujoco = Path(
        os.environ.get("MUJOCO_PY_MUJOCO_PATH", str(Path.home() / ".mujoco/mujoco210"))
    )
    libraries = os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
    if str(mujoco / "bin") not in libraries:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
            [str(mujoco / "bin"), *filter(None, libraries)]
        )
    if "mujoco_py" in sys.modules:
        return
    spec = importlib.util.find_spec("mujoco_py")
    if spec is None:
        raise ImportError(
            "Install requirements.txt and MuJoCo 2.1 before validating a numerical run."
        )
    package = Path(spec.origin).parent
    tag = str(sys.version_info.major) + str(sys.version_info.minor)
    cached = sorted(
        (package / "generated").glob(
            "*_" + tag + "_linuxcpuextensionbuilder_" + tag + ".so"
        )
    )
    if not cached or sys.platform != "linux":
        importlib.import_module("mujoco_py")
        return
    if len(cached) != 1:
        raise RuntimeError("Ambiguous installed MuJoCo CPU extensions.")
    if "MUJOCO_PY_FORCE_REBUILD" in os.environ:
        raise RuntimeError(
            "Unset MUJOCO_PY_FORCE_REBUILD to reuse the installed extension."
        )
    from importlib import abc, machinery, util

    builder = package / "builder.py"

    class Loader(abc.Loader):

        def create_module(self, spec):
            return None

        def exec_module(self, module):
            tree = ast.parse(builder.read_bytes(), str(builder))
            entry = [
                n
                for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "load_cython_ext"
            ]
            if len(entry) != 1 or [a.arg for a in entry[0].args.args] != [
                "mujoco_path"
            ]:
                raise RuntimeError("Unsupported mujoco-py builder interface.")
            entry[0].body = ast.parse(
                'return load_dynamic_ext("cymj", _bca_cached_extension)'
            ).body
            module.__dict__["_bca_cached_extension"] = str(cached[0])
            exec(
                compile(ast.fix_missing_locations(tree), str(builder), "exec"),
                module.__dict__,
            )

    class Finder(abc.MetaPathFinder):

        def find_spec(self, fullname, path=None, target=None):
            if fullname != "mujoco_py.builder":
                return None
            found = machinery.PathFinder.find_spec(fullname, path)
            if found is None or Path(found.origin).resolve() != builder.resolve():
                raise RuntimeError("Unexpected MuJoCo builder location.")
            return util.spec_from_file_location(fullname, builder, loader=Loader())

    finder = Finder()
    sys.meta_path.insert(0, finder)
    try:
        importlib.import_module("mujoco_py")
    finally:
        sys.meta_path.remove(finder)
