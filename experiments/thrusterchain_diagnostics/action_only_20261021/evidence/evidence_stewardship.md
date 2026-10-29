# Evidence stewardship audit: controlled action-only experiment

This audit checked the completed local experiment without changing its frozen code, protocol, results, canonical inputs, permissions, or services. All 200 registered replications are present, all 200 raw-file hashes match their JSON records, and their identities match the frozen experiment. The installed environment was captured **after execution**, at **2026-10-21 14:45:44.705105 UTC**. It is not a pre-run dependency lock.

The machine-readable receipt is [environment_versions.json](environment_versions.json), SHA-256 `376106c8a14084d2e262a3d20682700a54e881ed8949e4ac67c3789530aa0a66`. It includes every installed distribution name/version, all 200 JSON/raw checksums, array schemas, input sizes, source fingerprints, backend observations, and the checks below. Its analysis-file hashes describe the files observed during this audit; a later delivery manifest should identify the final analysis/presentation revision.

## Provenance chain and preservation

| Layer | Evidence and audit result |
| --- | --- |
| Parent trained model | `core_20261020T2250Z/jobs/replicate_01/full/bca`, completed update100000, optimization seed2101, source-data seed1101. Original checkpoint SHA-256 `4088278230fc413c509e27e189daa342468cfc4fb6d2479e3e8c29e149dac3d8`, 5,210,088 bytes. The model and scale were read only. |
| Original implementation | Training revision `b007052a2494dc651ac00469fa9e922b287cd1e5`. The observed checkout had changed to `30c5c68cbcfab7339ed4b69c109513e7acbb121b`. The engine imported the preserved release closure, not the changed checkout. This audit independently matched all **77 Python source files** in `engine_identity.json` against their bytes and the release `MANIFEST.json`. |
| Metadata preservation | Original canonical Python/JSON metadata were unavailable during preparation. Preserved job manifest, full dataset metadata, provenance, and config came from `../results_20261021/release/ThrusterChain_results_20261021/evidence/core_20261020T2250Z`. Their hashes match the engine identity. Missing canonical files were not reconstructed or overwritten. |
| Normalization and physical source | The 620-byte normalizer and 41,387,360-byte `source_audit.npz` remain local. This audit verified their byte hashes against the engine identity; engine loading additionally verified their array-content hashes against the original metadata. The source audit describes project-generated ThrusterChain trajectories. |
| Selected states and action laws | `state_bank.npz` hash `2001f5444bb004063dce61e8f05d65defac70ecd2a05ac43385cb0cacc0bb63e` is bound into the freeze and every replication. It retains the selected observations, physical states, episode/row identities, context/time/regime fields, actor commands, grid, and exact behavior masses. Selection uses 64 distinct source episodes, 16/context, times0/50/99, without outcome selection. |
| Frozen experiment | `freeze.json` hash `1fc66520f2715c1e5c2396c1fcc7db9279bfecb20dedf61d58227fd575d7a18f`, timestamp **14:38:46.234545 UTC**, binds the protocol, four implementation/protocol hashes, five preparatory evidence hashes, and replication IDs0–199. All these current hashes match. The experiment is a post-hoc control specified before its own results, not a retrospective preregistration of the parent training study. |
| Execution | First saved replication completed **14:38:54.280383 UTC**; last completed **14:39:12.949644 UTC**. Terminal status was recorded at **14:39:12.954650 UTC**, with200 completed and no failed IDs. Individual records use offset-aware timestamps; this audit converted them to UTC before comparison. |
| Derived evidence | Analysis retains600 paired records,1200 method rows,8400 context/time subgroup rows, aggregate counts/intervals, and a coverage-partition figure. Analysis and presentation are derived artifacts, separately versioned from the frozen experiment. |

The inputs are **logically fixed by authenticated hashes and the run's overwrite/resume rules**. This is not filesystem immutability: the inspected checkpoint, normalizer, source audit, and state-bank files do not carry the Windows read-only attribute. No write-once storage, ACL protection, independent archival copy, retention schedule, or disaster-recovery restore was verified. This audit changed no access controls.

## Record completeness and unusable cases

The receipt independently checked all200 JSON files and all200 NPZ files, not a favorable prefix or sample. There were no missing or terminal-failed records. The local NPZ files total **73,304,210 bytes** and JSON files total **10,683,173 bytes**.

- Every JSON record has the expected replication number, status, freeze hash, raw basename/hash, all three mixture levels, original checkpoint hash, state-bank hash, support-audit hash, and sizes128/4096.
- The200 calibration-bank identities are distinct, as are the600 mixture query identities. There are25,600 calibration draws and2,457,600 evaluation rows across the three mixtures. The two methods evaluate the same rows, producing4,915,200 method assessments; these are not independent experimental replications.
- All raw schemas agree. Scores and label arrays are finite, scales are positive, state/action indices are in range, and thresholds contain no NaN. Infinite thresholds are retained as outcomes. All200 lambda-zero uniform/IW threshold and refusal arrays are identical.
- No actual bank in this run had zero positive weight. That does not remove the implemented and tested all-zero-bank case.
- Weighted posterior-infinite outcomes are explicit: **136,470/819,200** queries at mixture0.5 and **268,864/819,200** at mixture1.0. Uniform calibration had no refusals. The refusal reason is preserved beside accepted outcomes; these cases contribute vacuous coverage and neutral multiplier1, not evidence of useful finite uncertainty.
- Coverage numerators/denominators, finite denominators, fallback counts, and undefined finite-only summaries remain visible. The deployed continuous actor is off-grid at64/64 states; its unsupported commands are not assigned a fabricated finite ratio or a coverage result.

`TEST_EVIDENCE.md` records21 passing focused pre-execution tests:15 calibration tests and6 genuine-checkpoint/grid tests. Negative fixtures include zero-weight preservation, all-zero versus structural-support refusal, finite-mass posterior-infinite refusal, invalid inputs, and undefined metric handling. A first restricted-shell model-test setup could not read the optional Git HEAD; the approved scoped read-only rerun passed. That permission failure was retained in the evidence rather than represented as a scientific failure.

Earlier mixed-measure defects in the exploratory historical command branch remain separately recorded at `../results_20261021/module_fixtures/verification_v2/fixture_results.json` relative to the experiment root: SHA-256 `3e94afc2bf04d6d61643feb51172054a3e2ee8ef6513f2769759e387e0265751`,20 harness passes,18 scientific-contract passes,2 contract failures. Those negative cases are prior-method evidence, not failures relabeled as successes in this exact categorical experiment.

## Environment and randomization

Observed after completion: **Python3.10.11; NumPy2.2.6; SciPy1.15.3; JAX0.6.2; jaxlib0.6.2; Flax0.10.7; Optax0.2.8; Chex0.1.90; msgpack1.1.2; ml-dtypes0.5.4; TensorStore0.1.78**. Full installed distribution metadata are in the JSON. `orbax` and `orbax-checkpoint` distribution metadata were not found and are recorded as null; no version was guessed. The first environment probe stopped before writing output when it assumed that optional distribution existed. The completed probe records absence explicitly; no package was installed or altered.

The registry identifies **Intel Core Ultra9 275HX**,24 logical processors; the OS reports build26200. The post-run probe observed affinity mask`0xF00` (logical CPUs8–11), priority class`0x40` (IDLE), JAX CPU backend, and the requested one-thread numerical-library environment. `preflight.json` records the same intended run restrictions. The later probe is corroboration, not proof of every backend setting throughout execution. It did not use GPUs, stop other jobs, or contact another machine.

The environment capture is installed-version evidence, not a complete installer lock. Package wheel hashes, compiler/XLA build provenance, drivers, and an independently reproduced clean environment were not captured before freezing. Exact numerical equivalence across platforms is not promised; native arithmetic parity tests use documented float32/JIT tolerances.

The state selection seed is`2026102101`. Fresh experiment streams use NumPy PCG64 with `SeedSequence([2026102120,replication_id,stream_id])`: calibration states/actions/noise are streams1–3; posterior exponentials4; evaluation states/base actions/mixture uniforms/noise5–8. IDs0–199 are all retained. Calibration and evaluation are independent within each replication; methods share banks and posterior draws; mixture levels use declared common random streams. Raw noise, IDs, scores, scales, targets, Q values, weights, query masses, thresholds, and refusal masks remain in local NPZ files. Posterior draws are reproducible from the documented stream recipe rather than separately stored as another large array.

## Reproduction and plot commands

These are commands for a separately chosen reproduction or verification context; this audit did not rerun the experiment or regenerate its original outputs. From the experiment directory with the original local inputs available:

```powershell
# Focused component checks; engine fixtures bind themselves to CPUs8–11/IDLE.
& 'C:\Users\David\dev\BCA\.venv\Scripts\python.exe' -B -m unittest discover -s tests -v

# Existing completed runs are reused only after frozen-identity/raw-hash checks.
& 'C:\Users\David\dev\BCA\.venv\Scripts\python.exe' -B run_experiment.py run

# Full local reaggregation: verify raw NPZ checksums, preserve original analysis.
& 'C:\Users\David\dev\BCA\.venv\Scripts\python.exe' -B analysis/aggregate_results.py --output-dir analysis_reproduction

# Saved-record-only reaggregation from a compact bundle, into a distinct folder.
& 'C:\Users\David\dev\BCA\.venv\Scripts\python.exe' -B analysis/aggregate_results.py --records-only --output-dir records_only_reproduction

# Analysis contracts: missing/undefined metrics and paired inference.
& 'C:\Users\David\dev\BCA\.venv\Scripts\python.exe' -B -m unittest discover -s analysis -p test_analysis.py -v
```

`analysis/aggregate_results.py` produces `aggregate.json`, the three CSV tables, `paired_records.json`, `RESULTS.md`, `analysis_validation.json`, and `coverage_partition.svg`. The SVG is generated directly by its `stack_svg` function; there is no hidden plotting service. The records-only mode checks JSON hashes against the original full-analysis record manifest and explicitly records that raw NPZ checksums were not rechecked. It requires a separate output directory. Neither reaggregation mode recomputes model scores, calibration, or training.

## Portable delivery, remaining gaps, and administrative boundary

The compact delivery should retain implementation, protocol/freeze, source maps, environment/audit receipts, selected state/law bank, all200 JSON records, analysis code/tables/figures, failure/refusal evidence, and final file hashes. It intentionally excludes model checkpoint bytes, the41.4 MB original source audit, and the73.3 MB raw evaluation NPZ files. Those remain local. A recipient can verify and reaggregate recorded statistics from the compact bundle; they cannot independently reproduce model scores or inspect raw draw-level arrays without the separately identified local inputs. Hashes identify missing assets but do not substitute for their preservation. This audit did not assemble or transfer a bundle.

Concrete remaining stewardship gaps are: no pre-run dependency lock; no verified independent raw/model backup or retention/restore plan; changed checkout and missing canonical metadata make continued preservation of the authenticated release closure essential; compact delivery is analysis-reproducible rather than self-contained model replay; and the final delivery inventory must bind whichever later analysis/presentation revision is shipped. Scientific limits remain one frozen model, one fixed observed state bank, and one-step residuals; no across-trained-seed, occupancy-shift, true-Q, or policy-benefit conclusion follows.

The parent relayed at **14:43 UTC** that the user confirmed approval for the device/AI-cloud workflow. This is **user-confirmed workflow approval, not an independently audited institutional determination**. Exact institutional data classification, ownership, publication authority, and licensing were not inferred. This experiment uses project-generated ThrusterChain material; a broader statement about BCA/Farama does not identify exact external datasets or their licenses. No legal or compliance conclusion is made. This audit remained local and made no uploads, public releases, installs, access changes, new service connections, or transfers; any already-authorized Library delivery remains the parent's separate action.
