# Bind the installed filelock scratch probe before another native import

Create a narrow, separately versioned scratch lifecycle component for the installed
filelock._strict._probe_link_follow_symlinks function. Bind its exact source/code,
PID and one fresh private directory; allow only its default-tempdir probe, one
temporary directory, empty probe-source, one hard link and observed cleanup.
Reject extra operations, paths, altered ownership, repeated calls and partial
completion. Save first refused operation arguments and preserve failed fixtures.

Inspect the concrete tempfile/pathlib/shutil dependencies and filelock module-level
calls without importing target packages. Tests may execute the exact extracted
filelock function with actual stdlib filesystem operations inside fresh fixtures;
they do not import filelock/JAX/MuJoCo or accept a complete native import route.
Do not rerun old tests/probes or alter installed/closed sources.

Independently review saved fixture exits, event order and before-cleanup contents.
Publish only owned new source and small evidence; preserve prior failures and
unrelated work. Keep monitoring active and quiet for routine component preparation.
The future launch contract, real imported namespace, hardlinked stdlib source
provenance and full runtime integration still need separate acceptance before a
new import-only observation; no models, physics, key or resource lease here.
