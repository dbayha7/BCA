# Historical actor audit: completed Q1-minus-width study

Audit date: 2026-10-22. The completed study is `C:/Users/David/Documents/Codex/2026-10-20/task/direct_bca_usefulness_20261022_v1` (called **H** below). The comparison source checkout is `C:/Users/David/dev/BCA` (**B**). This audit writes only the new benchmark's `docs/historical_actor_audit.md` and `audits/historical_actor/`. No historical source, model, result, environment, security setting, or Git state was changed. No Git or network operation was performed. Applicable ancestors and project roots contained no AGENTS.md or project `.agents` instruction files; the user-level `.agents` contains skills. The offline-RL audit and data-audit skills supplied the objective and normalization review procedure.

**Finding:** the completed experiment implemented the explicitly approved absolute `Q1 - U` actor-training candidate. Its executable equation, normalization, finite/infinite handling, checkpoint identity, and inference path are recoverable. It was not the native BCA cloning intervention, a paired lower-edge penalty, an importance-weighted construction, or a verified original-thesis reconstruction. Fresh local replay restored all 45 terminal checkpoints and exactly reproduced 540/540 sampled saved action rows. The two historical evaluation stages have distinct floating-point normalization programs; their existing correction remains necessary.

## Exact actor and update contract

Let `z=N(s)` be the stored observation transformation, `a_pi=pi_theta(z)` the current unsmoothed actor command, and `q1=Q1_phi_new(z,a_pi)` the first head of the online critic **after the current critic optimizer step**. For batch size B=256 and action dimension A=4, the direct arm minimizes

```text
D = stop_gradient(mean_B(abs(q1)))
c = 2.5 / D
U_i = R_frozen * (max(eta_frozen(z_i, a_pi_i), 1e-6) * u_frozen)
L_actor = -c * mean_B(q1 - U) + mean_B,A((a_pi - a_logged)^2)
```

The width coefficient is exactly 1; BC has coefficient 1 and averages across both batch and action coordinates. The denominator is raw `mean(abs(Q1))`, not `abs(mean(Q1))`, penalized Q, twin-min Q, or a denominator with an added epsilon. Zero/nonfinite D is rejected. Actor parameters are the only differentiated tree; the critic parameters are fixed during that derivative, while `dQ1/da` remains live. Frozen scale parameters, residual unit, and threshold are explicitly stopped, while `d eta/da` remains live. Thus width can change the actor gradient; a fully detached direct penalty would not do so under this raw-Q normalizer.

Authoritative code: **H/direct_extension.py:47-70,73-85,108-143**. Exact source excerpts and hashes are preserved in [source_excerpts.txt](../audits/historical_actor/source_excerpts.txt) and [source_hashes.json](../audits/historical_actor/source_hashes.json).

All arms use the same architecture: actor `15->256 ReLU->256 ReLU->4 tanh` with max action 1; two independent critic networks on concatenated normalized state and action, each `19->256 ReLU->256 ReLU->1` (**H/source_snapshot/algorithms/td3_bc.py:47-92**). Adam learning rate is 0.0003. Critic regression is

```text
a_next = clip(pi_target(z_next) + clip(Normal(mean=0,std=0.2), -0.5,0.5), -1,1)
y = stop_gradient(r + (1-done) * 1.0 * min(Q1_target,Q2_target)(z_next,a_next))
L_critic = sum_heads(mean_B((Q_head(z,a_logged)-y)^2))
```

The width enters neither critic MSE nor the bootstrap target (**direct_extension.py:94-106**). The actor and both targets update at even accepted update numbers; target interpolation uses the newly updated online models with tau=0.005 (**:126-139**). Scale fitting occurs every critic step, before that step's critic update; the actor consumes the preceding frozen reference (**:88-91,115**). Terminal `done=1` at time 99 removes continuation for the intrinsic horizon 100. Rewards are untransformed. PRNG, model state, and accepted update count are preserved on invalid updates (**:141-163**).

Controls consume `P=0` (host) or `P=0.5*abs(Q1-Q2)` (two-head array standard deviation), replacing U in the same loss. Disagreement starts only when `reference.ready` becomes true; it remains active if a later valid threshold is infinite. All three arms still fit the same auxiliary scale network and refresh schedule. The unused native `blend=0.5` configuration does not secretly multiply this direct arm's BC term.

There are five paired data/reservation/optimizer seed triples, three retentions, and three methods: 45 jobs. Every job has 100,000 critic and scale steps, 50,000 actor steps and 18 refreshes. The first refresh follows update 10,000; the first delayed actor step that consumes it is 10,002. Refreshes are at 10,000,15,000,...,95,000; the terminal 100,000 checkpoint retains the 95,000 reference. All 45 manifests and restored counters confirm these counts. Across the 15 direct jobs, 75,000 of 750,000 actor steps used the recorded numerical fallback, exactly the 5,000 warmup actor steps per job. There were zero whole-run calibration refusals. See [training_counters_summary.json](../audits/historical_actor/training_counters_summary.json).

## What eta, u, and R actually represent

The native auxiliary network has two 64-unit ReLU layers and output `eta=softplus(log_eta)+0.01`. It receives normalized observation z and the queried action. A subtle inherited internal preprocessing is present: the calibrator is initialized with observation mean 0 and std 1, and its own forward applies `(z-0)/(1+0.001)` before concatenation with the action. This extra factor applies to all 15 calibrator observation coordinates; it is not an additional actor normalization fit (**source_snapshot/algorithms/td3_bc_bca.py:129-142; calibration/network.py:8-30**).

On logged training actions, the scale fit uses detached current twin-min q and detached smoothed target y. With one exponential-simplex draw p per minibatch, its loss is

```text
coverage_i = sigmoid(20*(y_i/u - (q_i/u-eta_i)))
             * sigmoid(20*((q_i/u+eta_i)-y_i/u))
L_eta = (sum_i p_i*coverage_i - 0.9)^2 + 0.005*sum_i p_i*eta_i^2
u_new = 0.99*u + 0.01*max(std_B(y-q),1e-6)
```

The scale Adam LR is 0.001. The unit starts at 1. Frozen calibration uses 128 retained rows, one per reserved episode, and scores `abs(y-min(Q1,Q2))/(max(eta,1e-6)*u)`. Numerator and positive scale are formed in native float32 and then converted to float64 for the normalized-score division and NumPy threshold calculation. Width deployment preserves `R*(max(eta,1e-6)*u)` in that float32 order. Sources: **td3_bc_bca.py:159-198,234-279; reference.py:148-194**.

`R=max(lambda_hat,lambda_hpd)`. `lambda_hat` is the smallest score whose empirical covered mass is at least 0.9. For each of 1,000 exponential draws, the posterior crossing includes one worst-case test atom; a draw with excess test-atom mass has an infinite crossing. `lambda_hpd` is the 950th sorted crossing. Native `freeze_reference` omits explicit weights and test mass, so **all calibration weights and the test mass are 1** (**reference.py:175-179; wbcp.py:45-130**). The code supports a weighted algorithm, but this experiment did not provide likelihood ratios or run IWBCA.

These are one-step smoothed twin-min residual bands, subsequently queried at current actor actions and used to penalize Q1. They are not certified long-horizon Q1 error bars. The bank is reused during co-training. Finite widths do not establish support for the deterministic continuous actor under the categorical logger, nor safety or policy-value coverage.

## Exact observation and action normalization

The raw observation has 15 coordinates: the 13-component state, observed episode-fixed regime, and `(100-t)/100`. Only retained **training-complement** current observations fit normalization. Cast those observations to float32, compute mean and population standard deviation on coordinates `0:13`, then add float32 0.001 to the standard deviation. Set mean of coordinates 13-14 to 0 and std to 1. Apply `(raw_float32-mean)/std` to both current and next observations. The regime and time retain their declared units in actor/critic inputs. Dataset actions are already in [-1,1]; no action or reward normalization is applied (**training_runner.py:340-348,383-391**).

Fresh verification reloaded each original learner NPZ using its frozen SHA-256 and each saved training/withheld index array. All 15 distinct dataset/retention normalizers recomputed **exactly**, all 45 checkpoints used their appropriate matched normalizer, and training versus withheld episode overlap was zero. Full mean/std values, data hashes, normalization file hashes, and array digests are in [local_inference_verification.json](../audits/historical_actor/local_inference_verification.json).

The restored actor is deployed directly. It does not evaluate a width, run a Q-minus-width optimizer, or do action selection during final-policy inference. Frozen Experiments 1/2 separately evaluate candidate scores using prospectively selected replica-01 host critics. Those analyses must not be confused with the final trained actor's inference computation.

## Refusal and fallback distinctions

| Case | Completed direct-Q study behavior | Evidence |
|---|---|---|
| Reference not ready | Branch before multiplication; U=0 and zero width action gradient; continue host actor objective and count fallback | direct_extension.py:62-69,145-148 |
| Valid ready reference with R=+infinity | Same U=0 numerical fallback; no finite threshold substitution, no certificate | :58-69 |
| NaN/negative/inconsistent stored threshold, invalid/failing-positive scale or unit, overflow of finite width inputs | Mark invalid; reject complete proposed state/carry; driver preserves last atomic checkpoint and records failed attempt | :54-69,141-163; reference.py:201-218; training_runner.py:680-683,710-730 |
| Insufficient reserved calibration bank | Direct whole run gets status `refused`, no trained checkpoint; controls can still run with unavailable calibration | training_runner.py:288-337,624-630 |
| Off-grid actor command under categorical logger | No importance ratio and no support-triggered rejection in actor loss; numerical width can remain finite. Evaluation records support separately | direct_extension.py:50-51; final_policy_evaluation.py:209-217 |

The categorical logger atoms are exact float32 `[-1,-0.8,0,0.8,1]` per coordinate. Zero retention removes high-load commands above the float32 0.8 knee; finite heuristic widths do not repair this missing support. The final evaluation counts grid support and retained conditional action support, not state-visitation or occupancy support.

The following older/native methods have different placements and refusals; they must retain distinct names:

| Method | Actual intervention and refusal |
|---|---|
| Native TD3+BC BCA | Detached logged-action BC multiplier `m=1+blend*U/(U+u)`; valid +inf or not-ready gives m=1. Actor is `-c*mean(Q1)+mean(m*mean_action(BC^2))`. B/algorithms/td3_bc_bca.py:283-310; td3_bc.py:128-143; calibration/dose.py:23-95. |
| Native ReBRAC BCA | Detached logged-action multiplier on actor BC, which sums action errors; actor value uses twin-min. Existing critic-target BC coefficient stays unchanged. B/algorithms/rebrac_bca.py:269-280,382-409; rebrac.py:194-199,220-237. |
| Native CQL BCA | Detached row multiplier on the clipped conservative **critic** gap; dual uses original unweighted gap. No direct actor Q-U. B/algorithms/cql_bca.py:284-298; cql.py:295-320,342-348. |
| Native IQL BCA | Before reference readiness, retain native weights. After readiness, detached post-cap AWR weights: for positive supported advantage A, `1+A/(A+U)*(w_host-1)` where `w_host=min(exp(beta*A),cap)`; nonpositive supported rows retain host weights; unusable width rows get zero. Shared Q/V updates remain native. B/algorithms/iql_bca.py:174-193,219-282; calibration/iql_reference.py:102-105; advantage.py:32-70. |
| Historical IW cloning | Ready +inf rows get maximum dose `1+blend` (1.5), unlike native m=1 and unlike direct U=0. B/experiments/shift_iw/learners_iw.py:655-683. |
| Historical paired lower-edge | Uses `W=[U(s,pi)-U(s,a_logged)]_+` inside raw-Q-normalized actor loss. Replaces +inf with the largest finite table threshold; all-infinite table raises. B/experiments/shift_iw/lower_edge.py:196-224,293-312. |

## Fresh local verification and prior precision issues

The fresh replay used the installed historical interpreter, Python 3.10.11, NumPy 2.2.6, JAX/JAXlib 0.6.2, CPU cores 8-11, IDLE process priority, one numerical thread, `-B`, and `PYTHONDONTWRITEBYTECODE=1`. It used only checkpoint restoration and saved observations: starts 0,16,32,48 and times 0,50,99 for each of 45 jobs. This is a fixed deterministic sample spanning all four context strata, not a new evaluation return bank. Each checkpoint loader authenticated source fingerprints, serialized checkpoint identity, restored state, and normalizer. All full-state digests remained unchanged.

| Fresh check | Result |
|---|---|
| Exp3 compiled actor vs saved requested actions | 540/540 complete action rows exactly equal; max absolute error 0 |
| Exp3 vs NumPy-normalized compiled apply | Max absolute difference 1.519918441772461e-6 |
| Exp3 vs native eager training helper | Max absolute difference 1.6391277313232422e-6; all comparisons pass the historical 2e-5 absolute/relative tolerance |
| Normalizers | 15/15 exact recomputations; zero reserved-episode overlap; matching transforms across each method triple |
| Mathematical fixtures | 6/6 passed: native width equality, live-action gradient on smooth stencils, manufactured smooth derivative, stopped scale-parameter derivative, not-ready/+inf fallback, finite overflow rejection |
| Historical writes, trained updates, plant transitions, QP solves | 0 |

There were 1,620 actor forward calls (three paths for each of 540 rows), 45 checkpoint-template initializations, and one separate fixture initialization. Those initializations create in-memory restoration/test structures; they do not advance any trained checkpoint. The six fixtures used initialized networks and no optimizer updates. Verification took 14.153 seconds excluding inventory; fixtures took 4.464 seconds. The off-config message in the captured logs is expected: this synthetic task has no published CORL configuration and the frozen run explicitly allowed that declared variant. It is not a failed restore or permission denial.

**The 45-run normalization issue is already documented and distinct from the older exporter issue.** Exp3 compiles `(raw-mean)/std` together with actor apply (**final_policy_evaluation.py:334-343**); Exp1/2 performs NumPy float32 normalization before compiled apply (**evaluation/frozen_adapter.py:80,129-138**); `TrainingResult.policy` uses NumPy normalization followed by eager apply (**training_runner.py:463-471**). H/audit_normalization/NORMALIZATION_REPLAY_REPORT.md corrects the earlier inaccurate "same compiled native actor form" statement. Its saved audit tested all 57,600 rows of nine replica-01 policies: all Exp3 actions exactly matched saved commands; cross-compiled-path max difference was 1.7285346984863281e-6, with zero rows outside 2e-5 tolerance but only 923 bitwise-equal rows. That evidence and this new 45-policy sample do not prove equality on all unseen inputs or equality of alternate closed-loop returns.

**Older Unifloral/Four-host exporter caveat remains unresolved in this local evidence.** `C:/Users/David/Documents/Codex/2026-10-20/task/results_20261021/method_audit/REPORT.md:55` reports remaining Four-host v4 S7 float32-versus-float64 scoring discrepancies. Its hash is `33436f1b28062d3d19b3fde8e754cd4f265076b292b1f18425c1d8058ed686a1`. The cited `B/outputs/shift_hosts_v4/RESULTS_SUMMARY.md` and `B/experiments/shift_hosts` are absent locally. The action was read-only source recovery; its outcome is unavailable source, not denied, pending, or uncertain execution. No original exporter/S7 rerun was possible and no claim of repair is made. This missing older source does not block the separately authenticated 45-run native checkpoint replay.

## Preservation and evidence identities

[historical_inventory_before.json](../audits/historical_actor/historical_inventory_before.json) records path, size, mtime and SHA-256 for **all 22,683 files / 10,381,157,668 bytes** in H, including every historical intermediate/final checkpoint and result. A second complete rehash found **zero added, removed, changed, or timestamp-modified files**: [historical_preservation_check.json](../audits/historical_actor/historical_preservation_check.json). The inventory is a content record, not a filesystem write lock or independent backup. Its SHA-256 is `69cc77a4fd9ee406d2ebc819dd45ea0cb28d5f700a47f881a5be2513f34e788b`.

| Exact artifact | SHA-256 |
|---|---|
| H/direct_extension.py | `6af068817b3e3ca8b3c183d8a203ad88c44d3f18ab19046ba03e0f8894f1f462` |
| H/training_runner.py | `5b2abe01c12df06d088b8565fb7a77b0b84826fd9a47b3e604f174e567811052` |
| H/final_policy_evaluation.py | `934f86f54e71ef75ae1968d88c0a0173169b420c9d13c171d0118052fedfae33` |
| H/evaluation/frozen_adapter.py | `20b128a046e35e1c290c262d32783b8eb60f7e49df7a657840f29de1ec946c17` |
| H/TRAINING_FREEZE.json | `27a8102103e6c1616b082e71e8b794339ddfa9f820348ff6ddb2faa177311a27` |
| H/COMPLETION_RECEIPT.json | `d0fe170773590d8242bab859d9cc57be451742285e2e88e264a0a1d5b672a444` |
| New local_inference_verification.json | `32d39a2992acd921fc4239766f4912ddb263b698fb289a7fa59bb55f01ecf236` |
| New verify_historical.py | `2553036c35eccbefd912cceed9ac9c8ccb639878c6861ab6ebd00500edc59953` |
| New contract_fixtures.json | `653ad89dcb811ebe75114f150125d52120aff6ea07446a7bc597d4ba0ebd4751` |

The fresh replay JSON records every one of the 45 terminal checkpoint hashes, native actor/full-state hashes, normalization hashes, and sampled trajectory-array hashes. For example, replica_01/full/bca_direct terminal checkpoint has serialized SHA-256 `4c001b5584d74815e43c1df68b82a592337a30a9a24cf128cbf0129616efa6ac`, actor-array digest `481ccd9941d38ac6fca13d02f5b4f21865a36f93267871e82b904500ef52b433`, and full-state digest `536363d4f3da866cc72b1ad10dc3b7fa0e10a80ca831ced6ad385067814d2b67`. These are distinct identities, not interchangeable hash types.

## Commands and interpretation

Executed from `C:/Users/David/Documents/Codex/2026-10-22/task/thrusterchain_benchmark_20261022_v1`:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:\Users\David\dev\BCA\.venv\Scripts\python.exe' -B '.\audits\historical_actor\verify_historical.py' inventory
& 'C:\Users\David\dev\BCA\.venv\Scripts\python.exe' -B '.\audits\historical_actor\verify_historical.py' verify 2>&1 | Tee-Object -FilePath '.\audits\historical_actor\local_inference_stdout.log'
& 'C:\Users\David\dev\BCA\.venv\Scripts\python.exe' -B '.\audits\historical_actor\capture_sources.py'
& 'C:\Users\David\dev\BCA\.venv\Scripts\python.exe' -B '.\audits\historical_actor\run_contract_fixtures.py' 2>&1 | Tee-Object -FilePath '.\audits\historical_actor\contract_fixtures_stdout.log'
& 'C:\Users\David\dev\BCA\.venv\Scripts\python.exe' -B '.\audits\historical_actor\verify_historical.py' preservation
```

All five commands completed with exit code 0. Core JSON outputs use exclusive creation; a repeat must use a separately named new audit directory to preserve these receipts. No historical evaluator CLI, training driver, rollout, or QP command was invoked. The inference verification checks reproducibility, not policy usefulness: the completed study's final report still finds no established BCA-versus-host trained-return benefit, with all three reported intervals crossing zero. Its negative/inconclusive scientific results remain intact.
