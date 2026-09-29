# Actual import route: stdlib registration and finalizer obligations

Follow concrete imports from saved filelock._async_read_write (ThreadPoolExecutor)
and the saved capability callback (TemporaryDirectory -> weakref.finalize).
Inspect exact installed concurrent.futures sources and the corresponding stdlib
threading/weakref/tempfile sources without importing target packages or executing
their definitions. Save source bytes, hashes, stat identities, callsite conditions,
callback forms and shared-state initializers. Reuse the previous saved filelock
source graph as the external lead; do not claim a complete external import graph.

Independently review saved source and current identities, reconstruct the precise
lazy ThreadPoolExecutor import route and registration helper bodies, and identify
whether existing plain-function-only registration fixtures cover each callback
form. Keep import timing, live registration count, current registry contents and
installation effectiveness explicitly unobserved. No blanket exception or runtime
dispatch follows from static findings. Record concrete additional bindings for the
actual composed import guard and leave the 15-field/native/scientific gates held.
Verify closed artifacts unchanged and publish owned small source/evidence/docs
with actual push/remote/blob receipts. No old suite/probe/audit is rerun.

Follow-up from inspected _base.py imports and saved filelock._api: inspect logging, secrets and random sources and their three known aliases. Bind logging shutdown/atfork and random singleton callback source/state obligations through separate supplemental review; retain the first seven-source inspection/review unchanged. Still no target execution or complete external-closure claim.
