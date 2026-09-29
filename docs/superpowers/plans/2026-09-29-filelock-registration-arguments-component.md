# Exact filelock registration-argument observer fixtures

Close the concrete diagnostic gap where the old profile hook recorded a C-call
location but could not see keyword arguments. Add a separately versioned observer
that installs explicit Python wrappers for sys.addaudithook, os.register_at_fork
and atexit.register. Each wrapper must validate the full pinned caller code,
registered namespace, exact declared sequence and actual callback argument names,
objects, compiled code, defaults and captured keyword-default references before
forwarding exactly once to the original native registration API. Never call or
replace a callback. Preserve first refusal, including actual callback metadata.

Freeze direct callback global references and require a separately supplied state
verifier before/after each registration and at completion. Its identity/code and
defaults are checked. This does not establish transitive dependency/state graph
completeness: the real filelock state verifier and full stdlib graph remain pending.
Profile observation must distinguish wrapper requests, native registration entry
and return, and refuse bypass, repeated calls and swapped native APIs. Ordinary
native registration return alone does not prove an audit hook was installed:
existing hooks can suppress sys.addaudithook. Do not invent that proof.

Use isolated synthetic module fixtures containing exact saved installed callback
and registration-helper source text. Real native registration APIs receive those
synthetic-namespace callbacks, but no full filelock/JAX/MuJoCo import, fork, SQLite
connection or shutdown callback is executed. Keep hooks installed and use os._exit
after saving results; shutdown lifecycle acceptance remains pending. New refusal
fixtures cover actual argument substitution, keywords/order, code/default/global
and captured-state changes, callsite/namespace/source/API changes, swallowed
refusal, repeat/missing completion and direct native bypass. Save original sources,
all child output and actual exits. Do not rerun closed suites or probes.

A separate stdlib saved-evidence reviewer checks source text, all saved results,
forwarding entry/return evidence, no callback replay and limitations without
importing the component or replaying fixtures. Verify prior immutable artifacts,
then publish only owned small evidence/code with exact push/remote/blob receipts.
No real15-field import profile or scientific execution is authorized here. Full
actual hook/state/stdlib graph, worker-owned scratch, composed native route and
independent prereview remain required before one new versioned import observation.
