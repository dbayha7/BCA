# Runtime callable and path binding component

The previous archive component explicitly did not prove that a live factory's
code or globals matched its pinned source file. Close that specific integration
gap with a new component; preserve every closed source and the pending stream hold.

1. Implement bounded, externally hash/stat-pinned source reads rejecting redirected
   paths and hardlinks. Compile source without executing or importing it and match
   a named live Python function's complete code tree to its defining source.
2. Bind the static owner/attribute, defaults and exact referenced globals, including
   explicit static module attribute chains. Reject closures and dynamic namespace
   access that this component cannot prove. Recheck source and live binding changes
   before use. Do not treat source paths or filenames alone as runtime acceptance.
3. Exercise substitution attacks with the same filename/source bytes, changed
   code/defaults/globals/module attributes/owners/files and unsupported closures,
   using temporary synthetic source modules only. No native scientific import,
   model, simulation, resource writer or prior test suite is needed.
4. Document the limits: expected runtime object references and dependency inventory
   require independent external acceptance; native compiled/JAX callable internals,
   checkpoint/state contents, dispatch paths and supervisor remain separate gates.
   Publish only owned code and small exact-byte validation evidence.

This is a prerequisite of the actual execution capsule, not permission to create
a real extension ledger or execute a worker. No seed correction is applied.
