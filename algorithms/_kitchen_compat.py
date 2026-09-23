"""BCA kitchen compat.

Scientific definitions extracted from the recorded BCA source; see configs/sources.json.
"""

import os
import sys
import types as _types

STUB_ATTR = "_bca_kitchen_stub"


class _MujocoPyStub(_types.ModuleType):

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return type(name, (object,), {})


def _install_mujoco_py() -> bool:
    try:
        import mujoco_py

        return True
    except Exception:
        sys.modules["mujoco_py"] = _MujocoPyStub("mujoco_py")
        return False


def _install_kitchen_stub() -> None:
    _k = _types.ModuleType("d4rl.kitchen")
    _k.__path__ = []
    setattr(_k, STUB_ATTR, True)
    sys.modules.setdefault("d4rl.kitchen", _k)
    sys.modules.setdefault(
        "d4rl.kitchen.kitchen_envs", _types.ModuleType("d4rl.kitchen.kitchen_envs")
    )


def _try_real_kitchen() -> bool:
    if os.environ.get("BCA_KITCHEN_STUB", "") == "1":
        return False
    try:
        import d4rl.kitchen

        return True
    except BaseException:
        return False


MUJOCO_PY_REAL = _install_mujoco_py()
KITCHEN_REAL = _try_real_kitchen()
if not KITCHEN_REAL:
    _install_kitchen_stub()


def kitchen_is_real() -> bool:
    mod = sys.modules.get("d4rl.kitchen")
    return mod is not None and (not getattr(mod, STUB_ATTR, False))


def kitchen_status() -> str:
    return "kitchen=%s mujoco_py=%s forced_stub=%s" % (
        "real" if kitchen_is_real() else "stub",
        "real" if MUJOCO_PY_REAL else "stub",
        os.environ.get("BCA_KITCHEN_STUB", "") == "1",
    )


def prepare_kitchen_backend(dataset, verbose: bool = True):
    if "kitchen" not in str(dataset):
        return None
    if not kitchen_is_real():
        raise RuntimeError(
            "kitchen dataset %r requested but d4rl.kitchen is the empty fallback stub in this process, so the kitchen ids were never registered. Either the real import failed on this machine (check dm_control and mujoco_py) or BCA_KITCHEN_STUB=1 is set. Refusing rather than failing later inside gym.make. Status: %s"
            % (dataset, kitchen_status())
        )
    if not MUJOCO_PY_REAL:
        raise RuntimeError(
            "kitchen dataset %r requested but mujoco_py is the permissive stub in this process. Kitchen construction needs the real mujoco_py backend, because the dm_control backend rejects the kitchen assets on mujoco 3.x. Refusing. Status: %s"
            % (dataset, kitchen_status())
        )
    import d4rl.kitchen.adept_envs.mujoco_env as _menv

    previous = _menv.USE_DM_CONTROL
    _menv.USE_DM_CONTROL = False
    if verbose:
        print(
            "[kitchen] USE_DM_CONTROL %s -> False (mujoco_py backend; dm_control's MuJoCo 3.x rejects the kitchen XMLs)"
            % previous,
            flush=True,
        )
    return previous
