# Logged-data prerequisite audit

**The required logged fitting and calibration data are available.** A separate logged validation population is also available under the existing tuning-validation role. This preparation read no completed confirmation file, privileged candidate transition/value file, or numeric transition row in the dataset's reserved selection/final-calibration roles. No model, simulator, fit or calibration refresh was executed.

The transferred dataset is `titan_20261026/payload/transfer_bundle/research_runs/pilot_20261023T115408Z/regular_1280_2026102301.npz`: **25,253,032 bytes**, SHA-256 `4c857aaa0373bee4c74db33036303b4cf75c703b289a459525f75090741f3226`, content digest `6b58db68263dff0d7e9b8d6dfa7c1ca5d4127ed9ee6a3b9069add1c565e88331`. Its metadata and the committed checkpoint bindings agree. [data_provenance.json](data_provenance.json) records the exact row/episode IDs, sampling times, hashes and checks.

| Role | Available source | Selected rows | Unique episodes | Prior use |
|---|---|---:|---:|---|
| Auxiliary fitting | Exact existing scale-bank indices within 76,800 training rows | 256 | 221 | Used in original fitting diagnostics; training distribution |
| Calibration | First 128 tuning-validation groups sorted by group ID; one recorded time each, PCG64 seed 2026102362 | 128 | 128 | Original BCA reference, reused at earlier refreshes |
| Local validation | Remaining 128 tuning-validation groups; one fixed time each, PCG64 seed 2026102671 | 128 | 128 | Not consumed by the original checkpoint train/reference loader; global prior inspection history is not certified |
| Selection / final calibration | Reserved dataset roles, 128 episodes each | 0 | 0 | Numeric transition rows remain uninspected by this preparation |

All three selected banks are pairwise episode-disjoint. The 256 fitting rows span 221 groups, so they are not 256 independent episodes. All original training rows belong to 768 episodes. Neither fitting nor calibration episode seeds overlap the known development/confirmation query-seed ranges. Original calibration row IDs and times match the committed bindings exactly; fitting observations, actions and group IDs match the original exported scale bank exactly, including dtypes.

**Calibration reuse is explicit.** These 128 reference episodes already calibrated original BCA. A new candidate may only describe its use as exploratory reuse of an episode-disjoint reference, not as fresh confirmation or an untouched coverage validation. The remaining tuning-validation episodes are disjoint from the original training/reference loader; that is not a claim that no earlier research analysis ever inspected them. The prototype must not repurpose the reserved dataset roles or completed confirmation outputs to resolve this limitation.

The logged transition contract is the original finite-horizon ThrusterChain task: gamma 0.98, horizon 100, 14 observation coordinates with remaining time `(100-t)/100`, four float32 deployed actions in [-1,1], no truncation, termination only at step 100, and no terminal bootstrap. Each successor observation is from the same logged episode at the next time index. Reward/done fields are `[N]`; observation fields are `[N,14]` and actions `[N,4]`. The native TD target requires no logged next-action pairing.

The frozen native training mean/std and epsilon 0.001 remain authoritative. They must not be refitted on the small fitting/calibration/validation banks. A CPU reduction diagnostic differed from the saved GPU statistics by at most **5.960464477539063e-08** for mean and std; no normalization values were changed. Template dataset identity matches the committed binding exactly.

## Safe loader and fixtures

[logged_data.py](logged_data.py) verifies the dataset/binding hashes, metadata, exact split selection, finite values, action bounds, time coordinate and masks. Its ZIP/NPY reader decodes only explicitly allowed train/tuning episode records; selection/final-calibration records are skipped as opaque archive bytes. Whole-file hashing covers opaque bytes for identity without interpreting protected transition rows.

`load_logged_banks()` returns `fit`, `calibration`, `validation`, and `receipt`. Each bank contains NumPy `obs`, `action`, `reward`, `next_obs`, `done`, `group_ids`, and `row_ids`. Arrays are read-only. `load_training_template(native, path)` returns the original `(train, heldout, heldout_ids, data_info)` contract for exact checkpoint template restoration, decoding only the 768 train episodes and original 128 calibration episodes. It performs array construction, not model loading or optimization.

Eight [fixtures](test_logged_data.py) passed in 0.166 seconds. They cover protected-role rejection before archive access, duplicate/out-of-range rejection, member allowlisting, order-preserving subset reads, exact original export parity, calibration-binding parity, group separation, scalar shapes and read-only banks. The narrow template adapter was separately exercised with a simple Transition tuple; its train/calibration shapes and dataset-identity binding matched. No original eager full-array loader was executed.

From the repository root:

```powershell
.\.venv\Scripts\python.exe -B experiments/thrusterchain_bca/zeus_20261026_validation/bca_one_sided_v1/test_logged_data.py
```

The real candidate must fail closed on a changed/missing file or split, protected-role access, episode leakage, changed normalization contract, or an attempted fresh-confirmation claim using this reused calibration bank. Its fitting/calibration execution remains a separate step. There is no structural data-availability blocker, but fresh calibration/generalization guarantees are not established by this audit.
