# Installed native import prerequisites: inspected, execution held

The existing CPU mujoco_py extension and an explicit installed GLFW library now
have saved ELF dependency listings and independently checked file identities.
The inspection exited0 on September28 at23:58:55UTC. A separate stdlib review
exited0 on September29 at00:01:13UTC. Neither process imported mujoco_py or GLFW,
loaded a checkpoint, constructed a model/simulator, acquired either resource
lease or package build lock, or performed physics.

The inspection recorded54 installed source files across mujoco_py, GLFW and
fasteners, with AST import inventories for37 Python files. Ten relevant source
files are preserved as exact bytes. Two readelf operations and two system loader
`--list` operations exited0. The loader listings resolve24 distinct library paths
across the two targets; the independent review rehashed/restatted all54 sources,
24 library paths, both targets, inspection tools and the existing package lock.
The loader was invoked only to list dependencies, without dispatching the target's
ELF initializers. The exact commands, raw outputs and exit codes are retained.

The cached extension is the existing2,318,681-byte
`cymj_2.1.2.14_310_linuxcpuextensionbuilder_310.so`, SHA256
`32b62a6873ffa7e91b019e876754d4bf452fb7314758eb545bdb4517fdaebb61`.
Its four direct dependencies are libglewosmesa, libmujoco210, libOSMesa8 and libc.
The proposed GLFW target is the existing378,696-byte `glfw/x11/libglfw.so`, SHA256
`9f283bef3d8c70a2e3a85fcb6ab5dfc23509d1983d0b14ad5f90d51cbd6d74c5`.
These target identities were independently rechecked. The previous native cache,
builder and discovery source pins are unchanged.

The concrete import routes require more than the accepted numeric environment:

- MuJoCo discovery needs an explicit directory; otherwise it consults the home
  directory. The Linux builder requires that directory's bin in LD_LIBRARY_PATH.
- MUJOCO_PY_FORCE_CPU selects the CPU builder and avoids its nvidia-smi subprocess
  discovery. MUJOCO_PY_FORCE_REBUILD must remain absent.
- The package takes its existing `mujocopy-buildlock`, opens it in append mode and
  checks/creates its parent directory. This package lock is distinct from the
  original scientific shared resource lease. It was only inspected here.
- A cached extension ImportError can fall through to compilation, moving files
  and patchelf calls. An explicit, independently tested refusal of that fallback
  is still required before native import. File existence alone does not prevent it.
- GLFW's ordinary search calls subprocesses to query candidate library versions.
  Its existing PYGLFW_LIBRARY branch selects one named installed library directly.
  This extra route matters for the eventual single-worker and subprocess policy.

`proposed_native_environment.json` records exactly the accepted eight numeric
fields plus MUJOCO_PY_MUJOCO_PATH, MUJOCO_PY_FORCE_CPU, LD_LIBRARY_PATH and
PYGLFW_LIBRARY. It is explicitly a **12-field proposal, never launched**. The
original numeric guard and scientific capsule/worker/supervisor remain unchanged.
No native environment compatibility or unrestricted environment override is
accepted. Selecting this GLFW import target still requires acceptance of the
actual route; it is not permission to change physics libraries or model behavior.

The declared ELF dependency listing excludes later dlopen/plugin/driver discovery
and does not establish ABI compatibility. The Python AST inventory includes
inactive and function-local imports; it is not a complete transitive execution
graph. Matching installed source and binary hashes does not prove that the binary
was built from those exact sources. No native factory, full restore schema,
contextmanager binding, storage path or physical execution gate is closed here.

The first saved-review process exited1 because its helper searched all classes
for `_do_open` and found more than one definition. Its source/stdout/stderr/exit
are preserved. The separately versioned review selects InterProcessLock._do_open
explicitly and exited0. The inspection and loader commands were not rerun. No
test suite or preimplementation red was run or claimed. This was a review-selector
error, not native import failure or scientific retry.

The next concrete task is a source-bound cached-import route that refuses
rebuilding and unauthorized subprocess/writes, binds the package-lock behavior,
and independently reviews the exact proposed launch/import environment before a
single import-only observation. The full transitive dependency graph and actual
native field/contextmanager/storage/worker-owned shared lease integration still
need acceptance. No real extension or physical engineering may start before the
independent engineering gates. Every physical call must be durably reserved, then
action1e-6/reward1e-7/full-state repeat and pairwide precommit/science gates apply.

The prior key producer remains exit1; its40 saved key tables remain held from
execution despite their closed arithmetic review. No keys were regenerated and
no closed review/test/audit reran. Training acceptance stays10 runs/five pairs;
the last own cluster snapshot remains dated22:51:51UTC. No cluster probe occurred.
All prior closed v2 code/receipts, approved amendment, original plan and accepted
training artifacts remain unchanged. The local training mirror matches107/108
files, with only the previously reviewed advisor .gitattributes metadata append.

The publication includes only small source/evidence/documents, with exact raw
source bytes encoded inside the evidence package. Native libraries, weights,
binary keys, SQLite and NPZ are excluded. Native execution, saved-key execution
and scientific execution acceptance remain false.
