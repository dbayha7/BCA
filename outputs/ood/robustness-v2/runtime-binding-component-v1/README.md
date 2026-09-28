# V2 live callable/source binding component

Status:32 synthetic tests passed, actual exit0/no skips. No real model query,
simulator call, extension ledger, resource lock or scientific worker occurred.
The stream amendment remains unapproved and unapplied. Existing closed sources
and results are unchanged; no previous test suite or accepted audit was rerun.

`runtime_bindings.py` addresses a specific gap in the earlier archive wrapper:
a function can advertise the right source filename while running different code.
The new guard reads an externally hash/stat-pinned, bounded source file, rejects
symlinks and hardlinks, compiles without executing that source and compares the
complete named live function's code tree against its defining source. Bytecode,
nested code/constants, names, flags, arguments, line data and location are bound.
It checks the exact defining module/class attribute and externally supplied
positional/keyword defaults. Literal types and signed zero remain distinct.

The guard requires the exact referenced-global and direct static attribute-chain
inventories. It retains their literal values or live identities/code/defaults,
including intermediate module objects. Changed owners, code, provenance metadata,
defaults, globals, declared module attributes, source identity, process or binding
contract poison further use. Shadowed builtins and helper-code changes are caught.
Properties are not evaluated. Closures and direct dynamic namespace access are
refused. Runtime imports inside a function are also refused because a local import
would otherwise escape the referenced-global inventory. There is no dispatch or
scientific import in the guard, and it never calls the function it checks.

This is deliberately not a complete dependency-graph proof or a sandbox against
arbitrary Python. The caller must independently establish expected object
references, every transitive dependency, imported/native binary identities, JAX
callable internals, decorator wrappers, actual instance fields and runtime paths.
Direct inventory equality cannot establish those facts. Source contents are
hashed/compiled on binding; subsequent checks enforce exact file stat identity
and current live bindings. The receipt explicitly keeps dependency-graph,
native-runtime, execution and scientific acceptance false.

Tests use temporary synthetic modules, never native environment construction.
They cover forged code with the same filename, live/default/global/attribute/owner
substitution, mutable literals and signed zero, builtin shadowing, source replacement,
symlink/hardlink refusal, closure/import refusal, poisoning and compile-without-
execution. An integration test compiles the exact frozen ArtifactArchive class
AST in a synthetic namespace and checks its constructor/append/read methods against
the full pinned source. It constructs no archive and imports no production module.
The supplied serializers/global objects in that test are synthetic bindings, not
acceptance of the production namespace.

Preserved development evidence:27-test first pass0; expanded31-test run1 correctly
rejected locally defined fixture serializers with closures. The fixture helpers
were moved to module scope without weakening the guard, giving31-test pass0.
Source review then identified function-local imports as an uncovered route and
added an explicit refusal and test; the final32-test run exited0. No missing-module
red test was captured and no real scientific retry occurred.

The first exporter exited1 before creating any package or repository change:
its generated prior-component list accidentally renamed the prior global-barrier
validation to the new, not-yet-written binding validation. A separate v2 exporter
corrected that single prior-file lookup and retained the initial script/exit.

A separate read-only inspection of two exact frozen source files compiled13
entrypoints without importing/constructing them, actual exit0. Nine passed the
generic requirement scan; four need specialized bindings: FrozenPolicy.actions
and LocomotionAdapter.__init__ perform local imports, while capture/restore use
dynamic access to declared native fields. ArchiveFile.open additionally has a
contextmanager wrapper requiring separate binding. These are known integration
requirements, not changed sources or failed scientific execution. The source-level
DATA_FIELDS/MODEL_FIELDS inventory is saved without claiming live schema coverage.

Next: build the actual execution declaration/capsule and supervisor, with a
separately reviewed binding for these exact imports/declared-field operations and
decorator wrapper. Preserve frozen numerics and reject additional dynamic access;
do not simply relax the generic guard. Bind complete runtime/checkpoint/data/
support/full-state and actual path identities, final-device storage and shared
ownership before independent engineering execution acceptance. Reserve all real
engineering calls afterward, then require native action1e-6/reward1e-7/full-first-
repeat and independently saved science acceptance before any outcome. The stream
amendment still requires explicit user direction; no real v2 keys are generated.
