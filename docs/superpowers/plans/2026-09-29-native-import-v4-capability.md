# Compose worker bootstrap with observed capability and finalizer lifecycle

Implement a separately versioned concrete dispatcher derivative. Reuse the
closed bootstrap logic by exact source derivation; do not stack/install old
observers. Add ordinary module/capability entry and exit, the exact saved filelock
capability's scratch lifecycle, and actual TemporaryDirectory/weakref.finalize
registry construction and detach. Keep shutdown callbacks unexecuted and capture
conditional original atexit classmethod registration without replay.

The production fifteen-field declaration stays unchanged and unlaunched.
New persistent fixture bindings use different named roots. They execute the
exact saved capability definition through a registered synthetic module with real
stdlib objects, not the full filelock package. Complete registration/state/native
graph acceptance and the production parent observer remain separate and held.
No subagents, closed suite reruns or old probe retries.

1. Implement one dispatcher for bootstrap, observed scratch operations, bounded
   diagnostics and the exact weakref classmethod registration route.
2. Test new compositions with actual stdlib finalizer entries, both initial
   registration-flag states, existing unrelated finalizers, expected identity/
   order refusals and failure retention. Preserve all versions and actual exits.
3. Review saved evidence independently, without replay, and check prior inputs.
4. Publish only owned new sources and small evidence, accurately retaining the
   full-route hold and next actual callback/native integration requirements.
