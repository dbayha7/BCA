# Recovery interrupted by a second desktop reboot

At the September 28, 13:21 UTC check, the recovery worker was absent. Its last durable progress was **419/512 closed panels at 06:07:40 UTC**. Windows reports boot at 06:12:44 UTC and a Kernel-Power event 41 recording an unclean prior shutdown. The cause and exact worker termination time remain unknown.

Neither the recovery worker nor Windows supervisor actual-exit receipt exists. There is no completion.json or failure.json. **Actual exits remain unknown.** Both attempts, all partial traces, the original cumulative resource ledger and the accepted 237-record replay gate remain preserved. This monitor launched no retry or new controller, and performed no training update, simulator call or model query. Progress counts cover closed panels only; later partial work has not been independently audited. No accepted harm ranking, regret or coverage follows.

The first process/descriptor scan had permission-limited system entries; its original output is preserved. A separate root read-only scan found no matching recovery process or open shared-lock descriptor and no inspection errors. No lock was acquired. The startup CUDA-discovery warning in worker.log is historical and does not establish a new scientific failure.

The cluster SSH status probe timed out (actual exit 255); current cluster state is unverified. The last successful monitor snapshot was at 05:41 UTC, with 17 queue closures and ReBRAC Pen-human BCA at 920k updates. Those closure receipts still require their respective independent audits. Local CQL remains stopped on its earlier schema failure; neither it nor the old 810/native campaigns was restarted. All 108 frozen training files and the manifest hash checked unchanged.

`evidence.json.gz` contains ten exact original small probe/log/receipt byte streams as base64 with per-file SHA256 and lengths. Every stream was round-trip checked. No checkpoints, model weights, outcome archive or ledger are included. Pack and member hashes are in [the validation receipt](../../../../../../docs/validation/ood-recovery-second-interruption.json). Publication is established only by the separate push-exit and publication receipts at the workspace location recorded there.

Keep monitoring metadata quietly after this interruption is reported. Further recovery requires applicable user direction and separately checked handling that preserves both interrupted attempts and all prior accounting; no automatic replay or resume. An inaccessible status probe is not grounds for resubmitting the cluster job.
