#!/bin/sh
# Install the shift environment's dependencies (requirements-shift.txt: gymnasium 1.0.0 and three small helpers)
# into their own directory, never into the shared virtual environment.
#
#   BCA_SHIFT_DEPS    target directory (default: ~/.cache/bca-shift/py); experiments/shift/__init__.py appends it
#                     to the end of sys.path (it can add these packages but never shadow the environment's modules)
#   BCA_SHIFT_PYTHON  the Python that will import the packages: the repository's environment (requirements.txt,
#                     numpy 2.2.6) (default: python3); only its version is used
#
# The script refuses a target inside that Python's own environment (sys.prefix), installs with --no-deps and
# hash checking, and then checks that gymnasium 1.0.0 imports from the target while numpy still comes from the
# environment. Usage:  BCA_SHIFT_PYTHON=/path/to/venv/bin/python sh experiments/shift/install_deps.sh
set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
REQ="$HERE/requirements-shift.txt"
TARGET=${BCA_SHIFT_DEPS:-$HOME/.cache/bca-shift/py}
PYTHON=${BCA_SHIFT_PYTHON:-python3}

PYVER=$("$PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
"$PYTHON" - "$TARGET" <<'EOF'
import os, sys
target = os.path.realpath(sys.argv[1])
prefix = os.path.realpath(sys.prefix)
if target == prefix or target.startswith(prefix + os.sep):
    sys.exit(f"refused: the target {target} is inside the Python environment {prefix}; "
             "the shift dependencies must live in their own directory")
EOF

mkdir -p "$TARGET"
echo "installing into $TARGET for Python $PYVER"
if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$PYTHON" --target "$TARGET" --no-deps --require-hashes -r "$REQ"
elif "$PYTHON" -m pip --version >/dev/null 2>&1; then
    "$PYTHON" -m pip install --target "$TARGET" --no-deps --require-hashes --only-binary=:all: -r "$REQ"
else
    pip3 install --target "$TARGET" --python-version "$PYVER" --only-binary=:all: --no-deps --require-hashes \
        -r "$REQ"
fi

"$PYTHON" - "$TARGET" <<'EOF'
import os, sys
target = os.path.realpath(sys.argv[1])
sys.path.append(target)
import gymnasium, numpy
where = os.path.realpath(gymnasium.__file__)
if gymnasium.__version__ != "1.0.0" or not where.startswith(target + os.sep):
    sys.exit(f"gymnasium {gymnasium.__version__} imported from {where}, expected 1.0.0 from {target}")
if os.path.realpath(numpy.__file__).startswith(target + os.sep):
    sys.exit("numpy was imported from the target directory; remove it from there")
print(f"ok: gymnasium {gymnasium.__version__} from {target}; numpy {numpy.__version__} from the environment")
EOF
