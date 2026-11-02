# Recovery v2: exact state content and separate byte encodings

The original checkpoint and failed recovery v1 are preserved. Recovery v1 stopped after the original checkpoint loader had successfully verified all numeric leaves, because it additionally required new MessagePack bytes to match the original file bytes. That extra condition confuses encoding identity with state identity.

The immutable terminal checkpoint contains 215 numeric leaves. A read-only standard-library MessagePack audit found all 215 exact matches against metadata by path, shape, dtype and raw-value hash. Saved Dense parameter maps have `bias, kernel` order. Flax 0.10.7 restores dictionaries using fresh-template order and serializes dictionaries in their current order. A map-order change is therefore a supported explanation; v1 did not save restored bytes, so the exact changed maps remain to be measured by v2.

Reference: https://raw.githubusercontent.com/google/flax/v0.10.7/flax/serialization.py (`_dict_state_dict`, `_restore_dict`, `to_bytes`).

V2 retains the original loader and its complete integrity checks, plus:

- Original payload SHA must match its commit marker and metadata.
- Decoded original payload, restored state and decoded reencoded state must each match the complete saved numeric descriptor tree exactly. No numerical tolerance applies to these checks.
- Canonical content hash and counters must be identical before and after export/inference.
- Reencoded runtime bytes must remain identical before and after export/inference. This runtime encoding hash is recorded separately from the immutable original file hash.
- Exported parameters and frozen reference must match exactly. The existing 32-row actor/Q/width parity checks retain rtol 2e-5 and atol 2e-6.

The exporter is byte-identical to recovery v1's seed-fixed exporter. The evaluator is unchanged from the separately prepared seed-fixed version. Their legacy `full_training_state_sha256` field identifies the runtime state encoding; `RECOVERY_LINK.json` explicitly links it to the immutable original checkpoint, exact canonical content and exported manifest.

Run in PowerShell:

```powershell
wsl -d Ubuntu -- /home/dbayha/bca-gpu/bin/python -B -u "/mnt/c/Users/David Bayha/Documents/Codex/2026-10-22/task/unifloral_export_recovery_v2/recover.py" --allow-outside-upstream-constraints
```

Zero optimizer updates, zero calibration refreshes, zero simulator steps; 300-second child limit. Static template construction is followed by full restoration of the prespecified terminal checkpoint. Highest precision, the original runtime package checks and scientific settings remain required. No installations or training retries occur.

Outputs go to a new timestamped folder. `encoding_identity.json` records observed map-order differences without printing tensor values. A passing report records 11 unchanged inference parity checks and prints/saves the exact subsequent frozen-diagnostic command. No diagnostic runs automatically. A failed or partial recovery must not be used for evaluation.

Preparation performed stdlib integrity regressions and AST/hash checks only. Numerical export recovery still requires user execution because the earlier WSL denial remains binding. This stage establishes checkpoint/export consistency, not uncertainty calibration coverage or BCA efficacy.
