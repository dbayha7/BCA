# Native import release and completion protocol: fixture acceptance only

The new parent/worker protocol now observes actual Linux child identity and the
exact fixture environment before release and again before completion acknowledgment.
It uses the unchanged closed v4 dispatcher for worker-owned diagnostics, scratch,
ordinary capability entry/return and natural finalizer detach. It installs no
additional audit hook or profile. **Full filelock/native integration and production
dispatch remain held.** The production declaration is unchanged and unlaunched;
its planned actual entrypoint and observer bindings are still unavailable.

## Closed implementation and checks

- `native_import_protocol_v1.py` SHA256
  `b0df0b3aaeff4abc914dca37b264ce6d4a4a3d8f4d5bca34dbb473ff0b07c9db`.
- `test_native_import_protocol_v1.py` SHA256
  `24220f0557b8cbfdd664c1cdaf03efc8ed8587ae56e7b7d53f5fe9eeda73bfe3`.
- Final23 new isolated fixtures: suite actual0, no skips, ended
  2026-09-29T04:55:29.227725UTC. Sources are now closed; later integration requires
  a separately versioned derivative. No old closed suite, probe or review ran.
- Separate stdlib saved-byte review actual0 at05:05:43.150912UTC, source
  `01a5ad1ef5aa899fd1794b136eedcdc37ae9e504cf422dfdecdb80645f436346`.
  It checked469 retained files and exports279 small exact-byte items without
  component import, target-definition execution or fixture replay.

The17 launched children have actual exits: three0, eight1, one23 and five-15.
Two0 exits completed the handshake; a third0 without completion was refused.
The eight1 exits are expected synthetic worker refusals. The23 is deliberate
death after the31-byte bootstrap event was fsynced. Five identified private child
groups received SIGTERM for timeout, stdout/stderr overflow, wrong readiness or
fake completion. Six other cases were refused before launch; their child exit is
null because no child existed. These are engineering fixtures, not scientific
failures. No preimplementation red is claimed.

## Exact process, environment and ownership boundary

Each fixture lives under
`/home/dbayha/bca-work/ood-native-v4-capability-fixtures/monitor_20260929T044506Z/protocol_tests_v2/<case>`.
The protocol rejects production purpose before claim or spawn. It binds the
unchanged15-field production declaration
`5f922e3d0d27116139fd99293f537427f6d369f702b7fd685dbbe9de67567763`, with only
fixture TMPDIR different. There is no inherited environment merge or stripping.
The actual isolated command is pinned Python3.10 `-I worker.py <fixture-parent>`.
Source bytes, full compiled class methods, direct helper code/defaults/globals,
parent identity and input hashes are checked repeatedly.

The parent exclusively creates its separate supervision claim, fsyncs launch
intent, streams stdout/stderr, and independently reads `/proc` PID, parent PID,
start ticks, session/group, argv and environment. Release intent is durable before
the release packet. Only the released worker creates diagnostics then scratch,
both0700, through the closed dispatcher's retained parent descriptor. The only
parent-precreated scratch is the explicit preexisting-root negative fixture.
No failed claim or worker root is cleaned, repaired, reused or retried.

The declared timeout is180s; the timeout fixture explicitly uses0.5s to exercise
termination. Both logs have8MiB ceilings. Overflow retains exactly8MiB, records
the first excess chunk's length/hash and leaves remaining unread pipe bytes
unavailable. It does not claim all emitted overflow bytes were saved. Those two
8MiB payloads stay local and are not published. Signaling requires the current
private child identity; arbitrary descendants and parent-death containment are
not proved. A failed `/proc` identity check cannot authorize signaling.

## Completion and failure evidence

After ordinary capability completion, the worker writes the four declared0600
exclusive/no-follow diagnostic files, bounded8MiB each/32MiB total, using the
unchanged dispatcher's pinned diagnostic method and fsync route. The completion
packet binds actual worker identity, exact environment, root identities and
file identities/lengths/SHA256. The parent opens roots/files read-only using
retained directory descriptors, checks names against descriptors before/after
readback, confirms the second process observation matches the first, then fsyncs
acknowledgment intent before sending it. The worker rechecks the acknowledgment,
pending-packet identity and diagnostic bytes before its acknowledgment receipt
and captured native `os._exit(0)`. The parent also checks diagnostics after exit.
Native return0 alone does not satisfy this protocol.

Failure keeps observers installed and attempts only the closed bounded diagnostic
write route; identity tampering can refuse that write too. The worker reports
available first-refusal metadata and uses captured native `os._exit(1)`, avoiding
shutdown/finalizer callback execution. Two cases receive a real parent completion
ack but deliberately reject it or mutate pending state; they exit1 without
worker acknowledgment. Their failure event is appended after the pending
diagnostic snapshot. The review preserves this changed event file and does not
mistake the earlier manifest for final bytes. Profile-removal refusal retains
profile-present true in its failure packet; no profile-loss claim is made.

Four complete capability lifecycle journals are saved: two success cases and two
later acknowledgment refusals. Each has four boundaries, twelve scratch audit
events, thirteen native entry/return pairs with unavailable C args/kwargs null,
and actual TemporaryDirectory/weakref registry entry and natural detach. All four
observed baseline registries were empty/registration flag false, yielding one
original classmethod atexit registration each. This is an observation of these
fixtures, not a general startup assumption. The wrong-result case reaches the
ordinary capability before its deliberate result alteration according to the
pinned worker route, but saves no final lifecycle journal; no fifth journal is
invented. No observer invokes/replays the capability or target callbacks.

Source review verifies fsync-before-ack ordering; this protocol's event file does
not retain a full dynamic diagnostic native-operation/fsync journal. Saved readback
and actual exits do not prove power-loss durability or complete native syscall
containment. Target-module inventories are empty snapshots. Maps are saved, not
accepted as a complete native dependency/runtime graph. Bootstrap stdlib imports
and registrations before the observer are not retrospectively counted.

## Preserved development history and remaining work

The first17-test development suite passed0 at04:54:07.797281UTC. Two original
child-exit receipts were null after a `/proc` PermissionError while attempting to
stop a worker which had reported failure; the precise access-error cause is not
established. Original null receipts and raw bytes remain unchanged. The separate
final version waits for the reported worker's natural exit before signaling and
performs a bounded final reap even when `/proc` is unavailable. Six new cases
cover pending-state change, premature completion, repeated release, altered
capability result, profile-removal refusal and a changed parent input hash.
Final23 pass0 includes observed1 exits for both formerly unavailable cases in
new children; this does not retrospectively assign exits to the old children.

The new suite covers source/spec/environment/production-purpose refusals,
preexisting scratch, method override, logs, timeout, readiness and completion.
It does not independently exercise every supervisor root/diagnostic replacement,
parent-death or native-method mutation case. Direct helper binding is not full
transitive stdlib/native/state-graph acceptance. Cooperative Python checks are
not a security sandbox. The synthetic registered capability module is not the
full filelock package or real filelock namespace.

Both actual production roots were absent at the dated05:05:39.620862UTC review
snapshot. Neither was created or claimed; this is not future freshness acceptance.
The immutable-input check passed0 at05:05:37.908797UTC, source
`260a7d2a9237f3ee872548a111058742b9abe80cc484ba76b33c147198eb3353`, including
closed v4, all prior sources/receipts, training, original plan and approved
amendment. The known local107/108 training-source match remains only the reviewed
publication metadata delta, not a scientific exception.

Next implement actual filelock audit/fork/shutdown callbacks and captured state,
native bound lock methods, Random.seed owner/closure, logging captured defaults,
internal threading partial/list/state, full relevant live source/global/native
graph and pre-boundary provenance, plus cached builder/package-lock refusal.
Bind this concrete parent/completion route in a separate actual entrypoint and
independently prereview the full route before one new versioned import-only
observation. The closed v4 dispatcher still explicitly refuses that full route.

The40 numerically reviewed failed-producer key tables remain held pending explicit
independent execution disposition. No new keys, model, simulator, extension,
scientific lease, ledger, physics, cluster probe or training change occurred.
Training stays14 accepted runs/seven pairs, both ReBRAC cells3/5. The62 closures/
50 pending audits at02:24:54UTC remain historical. All queues and holds persist.
