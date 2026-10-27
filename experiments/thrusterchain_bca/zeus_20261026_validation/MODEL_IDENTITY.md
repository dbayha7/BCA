# Exact Titan UniFloral model identity

The required model is the transferred **original UniFloral standalone TD3+BC** at revision `f2dc1278eae18ed3e22c92255119c54885414d17`. The terminal Titan checkpoint, its target networks, frozen BCA auxiliary state, native configuration and normalization remain unchanged. CORL code, models and configurations are not substitutes.

`corlverify` is only the name of an existing environment providing CPython 3.12.14. The proposed new `/home/david/bca-operator-audit-20261026` venv has no system-site-package inheritance and installs only the reviewed dependency lock. Its base interpreter's directory name does not choose an RL implementation. No new environment or packages have been installed.

## Reverification completed

The repeated stdlib-only preflight passed, verifying all 360 package files with `scientific_execution: false`. [Raw receipt](identity_preflight.stdout.jsonl). An independent source/metadata inspection verified the exact 40-file native snapshot and equality of the normalization arrays to the checkpoint metadata. [Identity receipt](unifloral_identity.json).

| Artifact | SHA-256 |
| --- | --- |
| UniFloral `upstream/algorithms/td3_bc.py` | `6f5ab6f98d706d902a18f11e1c81e99e2691dc4c8a53843bf6521c9f95d00eee` |
| Full terminal `state.msgpack` | `874d65b7f4dcb5ec285a0d4360439c73a787b5e8c567ff7c0138e0c1e22948e8` |
| Checkpoint metadata | `b2a4b47570974319efbc7548083bda1f3362ce0c325eb8c5443bc9a71c58329b` |
| Inference parameters | `b8f2d15f54aab85e64583bc253c069aab9eed91988aec4ac064f87522736d90f` |
| Normalization receipt | `4f6be8817906c2ed99a9e2ef1355c43d9ce5640ec3984e104625b85d25ba5dc3` |
| Native checkpoint config | `c68fc06feac7dfca921413f370f40be8870f35ec59fba158ba503dd0f0fc0067` |

The same terminal counts are preserved: 20,000 critic, 10,000 actor, 20,000 auxiliary updates, last reference refresh at critic 20,000. The original config retains `gamma=0.98`, batch size 256 and `kappa_penalty=0.0`. Normalization remains `raw_native_mean_std_plus_0.001`, float32, reward scale 1.0, and 14-dimensional time-aware observations. No re-normalization or method change is authorized to repair a runtime issue.

## Module origin and remaining live check

The frozen checkpoint loader explicitly loads `payload/unifloral_td3bc_validation_v1/native_loader.py`, which verifies and compiles selected unchanged computational definitions from `payload/unifloral_td3bc_validation_v1/upstream/algorithms/td3_bc.py`. It sets the resulting module name to `unifloral_pinned_td3bc_computational_slice` and `__file__` to that exact source path. This reviewed harness excludes upstream CLI/data-loading imports; it does not replace native network or update definitions.

Checkpoint helpers are loaded from the explicit transferred `unifloral_checkpoint_v1` files, with conflicting already-loaded helper paths rejected. The evaluator similarly checks that `thrusterchain.environment.__file__` lies within the transferred `payload/transfer_bundle/core_v4`. The BCA bridge uses its hash-verified uncertainty-only source boundary. No repository-root CORL host is selected by these paths.

Actual model imports, checkpoint restoration and forward replay remain **unexecuted** in the new Zeus runtime. After an isolated environment is approved and prepared, verify and record the actual native module `__file__` and computational-definition origins before the scientific audit proceeds. The bounded worker must then restore the exact checkpoint, compare reconstructed bindings/normalization, prove current actor/critic/reference equality to the exported state and pass frozen replay tolerances before accepting its result. Any mismatch stops execution; it must not be fixed by substituting CORL sources, changing the model or loosening tolerances.

Only preparation and stdlib identity verification occurred here: zero evaluator transitions, model fitting, model/optimizer/calibration updates or confirmation reads. The user subsequently completed package resolution manually; the reviewed isolated installation remains pending approval. No WSL retry occurred.
