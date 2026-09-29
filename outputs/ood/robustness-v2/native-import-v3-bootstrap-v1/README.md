# Native import v3: worker bootstrap and diagnostics only

The first concrete dispatcher now implements worker-owned root creation and
bounded diagnostic writes with one audit/profile owner and a shared first
refusal. **The full native import route remains unimplemented and dispatch is
held.** This is a phase implementation using explicit persistent fixture roots,
not execution of the declared production profile.

`native_import_dispatch_v3.py` SHA256
`5f3bd9a4f408ffadbb0ff707fa2581c5da2c3f4c903fe4bbc9fa877e96db3d29`;
`test_native_import_dispatch_v3.py` SHA256
`a86faf02b312eb7eb7adb04155f9afd81ab71fa4706d9ef793358270d33d7562`.
These new sources are closed at the final fixture pass; later integration must
use a separately versioned derivative. Previous closed sources remain unchanged.

## Implemented behavior

The dispatcher binds the unchanged declaration hash
`5f922e3d0d27116139fd99293f537427f6d369f702b7fd685dbbe9de67567763`,
its own compiled method/helper code and direct references, process identity,
exact fifteen fixture environment fields, parent directory and native filesystem
API objects. The fixture binding changes only TMPDIR to its own scratch path;
it explicitly rejects the declared production roots and any production purpose.

After the parent independently reads the child's Linux PID, parent, start ticks,
session, process group, isolated command and environment and releases it, the
child creates diagnostics then scratch through a retained parent descriptor.
Both must be absent before either creation. Directories use mode0700 and
O_DIRECTORY/O_NOFOLLOW; names and descriptors are checked together before each
admitted operation. Four files, in declared order, use exclusive nofollow0600
creation: events.jsonl, final.json, modules.json and maps.txt. Parent directory
and diagnostic directory changes are fsynced before acknowledgment.

Each admitted native operation has an actual request with positional metadata
and ordered keywords, followed by observed native entry and return. Original
native APIs are called once. Diagnostic writes use only declared descriptors,
an8MiB per-file/32MiB total ceiling and fsync before acknowledgment. Bytes in
argument metadata are represented by exact length and SHA256; arbitrary object
repr/properties are not used. Direct native aliases have unavailable arguments
recorded as null, not invented empty tuples. Originating caller filename/line and
the first refusal survive poisoned checks.

After an ordinary refusal, only the pinned diagnostic method can request these
bounded writes, with identity checks still required. Changed code, globals,
binding, ownership, directory identity, diagnostic links or file length prevent
that diagnostic route too. This does not grant a blanket write exception. Hooks
remain installed and fixture children use os._exit; no cleanup or callback replay
is used to manufacture evidence.

## Observed fixture evidence

Final35 isolated tests exited0 with no skips at03:58:36.210224UTC:
34 children exited0 (one successful bootstrap and33 expected refusals), and one
deliberate child exited23 after acknowledged fsync. The death fixture retains its
37-byte events marker and empty other diagnostic files; its final result is
unavailable. Persistent paths are under
`/home/dbayha/bca-work/ood-native-v3-bootstrap-fixtures/monitor_20260929T034335Z/bootstrap_tests_v4`.

Negative cases cover premature/wrong/repeated release, repeated installation or
creation, pre-existing roots, outside writes, direct tickets/native aliases,
environment and observer changes, diagnostic paths/limits, stale code/defaults/
helpers/globals/binding/identity, replaced/mode-changed roots, hardlinked/appended
diagnostic files and attempted mutation after a caught refusal. Pre-existing
root and later parent tampering cases intentionally create those synthetic
conditions; they do not authorize production parent-owned roots or shared aliases.
The byte-ceiling case retains exactly8MiB of x bytes locally; this payload is not
published. No outside/extra file was created.

Two cases emit synthetic import and rename audit events. They do not import
filelock or execute a rename. Saved target-module inventories are empty snapshots.
The own-hook canary tests installation/suppression of this observer only; native
registration return does not establish effectiveness of other audit hooks.

Separate stdlib saved-evidence review exited0 at04:08:07.051917UTC, source
`44070218e609e10db4dac806029faa52ed45602468387eaf264201f340061586`.
It read all35 saved child receipts and retained source/configuration/file bytes
and metadata, checked release identities and exact creation tuples,446 complete
native operation triples in final result rows, plus creation evidence for the
deliberate death, and reconstructed diagnostic writes against retained bytes.
It did not import the component or replay children. The death fixture has an
fsync acknowledgment but no final write-event journal; that limit is preserved.
Both actual v3 roots were still absent at the review's04:08:03.686297UTC snapshot.
This is dated evidence, not future freshness or storage power-loss acceptance.

## Development history and limits

Preserve29-test v1 pass0 at03:53:43.188412 and33-test v2 pass0 at03:55:57.093091.
The first35-test v3 attempt exited1 at03:57:36.152741; its first complete child
exited2 with RecursionError and first_refusal null. Moving frame inspection into
the audit handler before event classification recursively triggered the
sys._getframe audit event. A separately saved final source classifies relevant
events before frame inspection; the unchanged35-test source then passed as v4.
All four code versions, actual exits and raw logs remain visible. No
preimplementation red or scientific failure/retry is claimed. A later read-only
guessed helper path was absent (shell1, target exit unavailable); no helper ran.

This is cooperative Python instrumentation, not a security sandbox, continuous
native-C syscall monitor or complete transitive global/native graph binding.
The fixture parent supplies one identity observation before release; it does not
implement the production parent observer, declared180-second/log bounds or second
identity observation before completion acknowledgment. Shared-source provenance,
bootstrap registrations, real callback owners/defaults/closures/state, internal
threading partials, finalizer detach, once-observed filelock capability and exact
cached native/package-lock routes are still pending. No actual target callbacks,
filelock capability, JAX/native import or production v3 entry point ran.

The saved-evidence review accepts bootstrap fixtures only. It is **not** the
independent actual-route execution prereview. The production declaration remains
unchanged and unlaunched, with actual entry/parent observer hashes null. No real
extension, scientific lease, model, simulator, new key or physics occurred.
The failed producer's40 tables remain held pending explicit independent execution
disposition; no regeneration or closed key review rerun. Training remains14runs/
seven pairs, both ReBRAC cells3/5. The02:24:54UTC cluster snapshot is historical;
no cluster probe or queue change occurred here.

Immutable review exited0 at03:56:15.570935UTC, checking previous closed code and
receipts, original scientific plan, approved one-stream amendment and training
artifacts. Local training matches107/108 only because of the previously reviewed
publication metadata append. Publication requires the separate actual push0,
matching remote and exact blob receipts; this document itself grants no launch.

Next work must implement the remaining **actual** registration/capability/native
phases in one dispatcher with these lifecycle constraints, then bind the complete
relevant live code/global/native/state graph and exact production supervisor
before independent prereview of one new import-only observation. Do not stack
the closed observers, preimport filelock, fake capability results, edit installed
dependencies, retry closed native probes or silently widen the old contracts.
