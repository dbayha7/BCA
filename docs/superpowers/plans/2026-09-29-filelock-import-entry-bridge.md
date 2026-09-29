# Filelock import-entry bridge and concrete stdlib provenance

Resolve the four known hard-linked stdlib sources using bounded Conda metadata
and candidate environment/package paths. Record every discovered same-inode alias,
bytes, ownership, mode and link count without changing installed files or treating
package metadata as proof of runtime safety. Independently review saved inventory.

Implement a new bridge that observes the named capability function at Python
call/return boundaries inside its module namespace. It must never invoke or replay
the callback or replace its result. Bind module execution code, entry callsite,
function code/defaults/globals, fresh owned root and explicit TMPDIR; reuse the
closed scratch lifecycle through composition without changing its source. Refuse
extra entries, changed globals/code/module/callsite, premature return, swallowed
refusals and outside writes; retain first event arguments and failed scratch.

Test the new bridge in isolated synthetic modules using the exact installed
callback source and real stdlib filesystem helpers. These fixtures are not the
full imported filelock namespace and do not accept a native launch profile.
Review saved outputs in a separate process without replay, preserve any failure,
and publish only owned small evidence. No closed suite or real failed probe rerun.

Record remaining actual stdlib live graph and filelock hook/state bindings
explicitly. No new real filelock/JAX/MuJoCo observation before independent acceptance
of those prerequisites and a separately versioned launch contract. No keys,
scientific resource lease/ledger/archive, models, simulators or physics.
