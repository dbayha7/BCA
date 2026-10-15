# Version 7 self-test reports from the Framework Desktop, for the cloud's handoff

These are byte-for-byte copies of the full self-test reports written on the user's Framework Desktop under
`/home/user/bca/runs/shift_explore/stage2_v7/selftest/`. They are here so the cloud can import the self-test without
the user pasting the block (README step 8; `make_handoff.py import-selftest --paste FILE`).

| File | What it is |
|---|---|
| `report_full_20261015T002933Z.txt` | **The full self-test to import.** Commit 66a8cd3. `READY (waiting for handoff/approval)`: all nine probe records bit-identical (F_ref and F_fresh 0.0, P2 largest difference 0.0), check (b) forecast READY (Q7 'stop'), P3 repeatability passed (6 runs identical), ST3-ST7 pass (0 of 64 anchors differ even without the constants). It holds the `V7 SELF-TEST BLOCK`. |
| `report_full_20261015T001200Z.txt` | The first attempt (commit 7173a31), for the record. It crashed at ST3 on a bug in `run_check_probe` (fixed in 66a8cd3) before any probe ran: no probe record was written and it holds no block. |
| `SHA256SUMS` | SHA-256 of both reports. |

For the cloud, after `git pull`:

    python3 outputs/shift/beyond-the-data/stage2_v7/cloud/make_handoff.py import-selftest --paste outputs/shift/beyond-the-data/stage2_v7_transfer/report_full_20261015T002933Z.txt
