# Approved Path A recovery: preserve the completed warmup

This additive recovery preserves the original Path A protocol, smoke amendment, passed smoke, approvals and interrupted attempt. It does not alter training, data, calibration, seeds, evaluation, physics or rewards. Do not use an older launch command or reset an existing directory.

## Exact lineage and remaining work

The interrupted `run_amendment_1` supervisor recorded signal 2 at November 2, 2026, 4:57:31 a.m. EST after 769.481604245 seconds. Its result SHA256 is `559d1e9ed3a0255904a9b2a1459a5629cac6f45b08b2be6b722570a332aa69a4`.

`honeypot_s0_warmup` completed 10,000 critic, 5,000 actor and 10,000 auxiliary updates and one calibration. Its 10,350,811-byte full-state checkpoint SHA256 is `5422f85ac77d2a51341bb1154d7ba057a623854fd5ec8232b762bf0443bc218f`. The report, checkpoint metadata and full-state bytes are authenticated together. That checkpoint contains model and target parameters, optimizers, phase/held batch, native and auxiliary RNG, uncertainty model and calibration reference.

`honeypot_s0_native` has no saved checkpoint or final receipt. Its last durable progress was 4,200 critic/auxiliary and 2,100 actor updates, with zero branch calibrations. Its unknown tail remains bounded by the whole job's declared 10,000 critic/auxiliary, 5,000 actor and one calibration. The recovery does not claim to resume at update 14,200.

The new `run_recovery_1` reservation copies only the completed warmup files byte-for-byte into the expected sibling directory, records `REUSED_FROM.json`, and verifies the copy against the sealed originals. This copy performs no training and is not counted as a second warmup. The unchanged production functions then restart the unfinished native branch from its original 10k fork, execute the remaining jobs, and run the unchanged evaluator. There are **17 new training jobs plus one evaluation**, producing the originally planned 12 final policies. The smoke and completed warmup are never rerun.

The user reported a header-only `ps -p 435,1693` result at approximately 5:15 a.m. EST on November 2. This is evidence that those two recorded processes were absent at that observation, not a global liveness guarantee. The recovery cannot stop unrelated processes. Any still-observed recorded process requires review rather than automatic termination or bypass.

## Approved accounting

| Item | Critic | Actor | Auxiliary | Calibration |
|---|---:|---:|---:|---:|
| Completed disposable smoke | 14 | 8 | 14 | 3 |
| Completed warmup, reused | 10,000 | 5,000 | 10,000 | 1 |
| Interrupted lost segment, lower to upper | 4,200-10,000 | 2,100-5,000 | 4,200-10,000 | 0-1 |
| New recovery work, maximum | 170,000 | 85,000 | 170,000 | 17 |
| Cumulative physical upper bound | **190,014** | **95,008** | **190,014** | **22** |

New recovery work can have exact counts if completed, while cumulative physical counts remain intervals because the old interruption tail is unknown. The final cumulative lower bounds would be 184,214 critic/auxiliary, 92,108 actor and 21 calibrations. These are physical counts; no extra updates are added to the intended final learned models.

The remaining production process allowance was `11,490 - 769.481604245 = 10,720.518395755` seconds. Reserve another 60 seconds for recovery preparation and round the new process ceiling down to **10,660 seconds (2h 57m 40s)**. The internal deadline is 10,645 seconds; the manual external timeout is 10,655 seconds plus five-second kill grace. All new jobs share this allowance. The combined charges are `1,950 + 60 + 900 + 769.481604245 + 60 + 10,660 = 14,399.481604245` seconds, within the original four-hour ceiling. The earlier smoke's unused reservation is not reclaimed. A timeout does not authorize extension or another retry.

No collection, setup diagnostic or smoke is repeated. Simulator counts remain **192,975 already consumed**, plus at most **237,600** for the unchanged scientific evaluation: **430,575 maximum**, within 498,400. Evaluation remains 1,536 return episodes and the frozen own-policy/common-controller probes.

## Scientific question remains unchanged

The two conditions share the fixed ThrusterChain goal/danger task but have separate offline retention masks. Honeypot removes source, next-state and crossing transitions touching the absorbing danger region. Bottleneck removes transitions crossing the fixed B-to-C projection slab, retaining genuine upstream/downstream fragments but no intact successful fitting route. This is unsupported-connector generalization, not guaranteed stitching. A change in retained state/time mixture remains possible.

For each condition and each of three independent initialization seeds, a 10k native warmup forks into matched 10k native and BCA continuations. BC coefficient stays 1, native TD3 alpha stays 2.5, and BCA uses the original actor score `Q1 - 3U` with live action gradients. Native critic targets and the original uncertainty/calibration machinery are unchanged. Auxiliary work is matched, so this is not an efficiency comparison.

Primary results are per-seed paired policy-return differences (128 shared starts per final policy). Secondary results include goal/danger/horizon outcomes, actual unsupported-edge exposure, raw and adjusted-score signed error/MAE, over/under/near-zero counts with tolerance 1, and beneficial/harmful ranking changes. Own-policy Q references force a first action then follow that exact deployed actor with actual full state/time, discount .98, and true terminal handling. A separate controller panel provides common-reference action values.

The downward penalty cannot directly raise a fixed underestimated Q. This study can expose helpful avoidance, excessive pessimism, no benefit or mixed effects. It does not guarantee error signs, prove Bayesian uniqueness, isolate withholding harm without an additional data control, or turn three-seed development evidence into broad confirmation. Path B remains separate and is not launched by this approval.

## Manual launch after source-only verification

The separately sealed recovery manifest and approval authenticate the original source/data identity, completed smoke, original approval, stopped run and reused warmup. No WSL, CUDA, training, calibration or simulator execution is performed during preparation. CPU tests exercise fixtures and saved evidence only; the already completed GPU smoke supplies the existing runtime evidence.

Run this command once on Zeus after the recovery seal and approval are verified:

```powershell
wsl -d Ubuntu -- bash -lc 'cd /mnt/c/Users/David/dev/BCA && timeout --signal=TERM --kill-after=5s 10655s /home/david/bca-operator-audit-20261026/bin/python -B -u experiments/thrusterchain_bca/two_scenario_path_a/recovery_launch.py --execute'
```

The source-only command is `recovery_launch.py --check`. The new terminal receipt will be `run_recovery_1/RESULT.json`; the final scientific report will be `run_recovery_1/evaluation/report.json`. Per-job progress remains in each job's `progress.jsonl`. Keep the foreground terminal open. No automatic retry or mid-job resume is added; another interruption preserves evidence and stops the attempt. Assistant-side WSL access remains denied.
