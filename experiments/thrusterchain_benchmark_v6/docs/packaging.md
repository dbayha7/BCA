# Wheel attribution artifacts

Version 0.1.2 retains the packaging correction introduced in 0.1.1 and explicitly lists the existing `third_party/BCA_LICENSE.txt` and
`third_party/BCA_NOTICE.txt` in `project.license-files`. The build requirement is
`setuptools>=77`, which supports that field. The wheel retains the original bytes
at these paths:

```text
thrusterchain_benchmark-0.1.2.dist-info/licenses/third_party/BCA_LICENSE.txt
thrusterchain_benchmark-0.1.2.dist-info/licenses/third_party/BCA_NOTICE.txt
```

Its `METADATA` advertises both original relative paths through `License-File`
headers. The files remain in the corresponding `.dist-info/licenses/third_party/`
directory after installation. The existing canonical JSON resources also remain
package data.

`BCA_LICENSE.txt` contains the retained Apache License 2.0 text.
`BCA_NOTICE.txt` retains the existing source attributions and the MIT notice for
the portions identified in that artifact. This packaging change does not add a
project-wide `License` or `License-Expression`, determine ownership, or authorize
a public release. It verifies retention of the supplied artifacts.

## Verification

Run the integration tests from the source root with an existing Python environment
that has pip and setuptools 77 or later:

```text
python -m unittest discover -s tests -p test_wheel_notices.py -v
```

The tests copy the build inputs into `.verification/packaging/`, build a wheel
offline with `pip wheel --no-index --no-build-isolation --no-deps --no-cache-dir`,
and install it into a fresh local target with
`pip install --no-index --no-deps --no-cache-dir --no-compile --target`.
They check both `License-File` headers, pinned SHA-256 values for the source,
wheel, and installed files, wheel `RECORD` digests, installed `RECORD` entries,
and the packaged Python and JSON sources. They do not install dependencies or
change the active environment. Test staging is cleaned on completion. Pip uses a
fresh, short task-specific temporary directory, which is also cleaned on
completion; on Windows it is under the user's local `AppData/Local/Temp` to
avoid the nested cache exceeding Windows path-length limits.

For a retained wheel, local installation, logs, and a JSON receipt, use a fresh
run name:

```text
python audits/revision_v3_wheel/verify_wheel.py --run-name final_wheel
```

The retained build and install directories are under `.verification/packaging/`;
the receipt is under `audits/revision_v3_wheel/`. Receipts identify every staged
input by SHA-256, record the commands and tool versions, and verify each packaged
source against that snapshot. Run this after source changes are complete.
An optional `--source PATH --expect-missing` reproduces the earlier omission
using a copied source tree, leaving the selected source unchanged.
