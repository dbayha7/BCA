Means over seeds 0, 1, 2, 3, 4. Columns: points; share of steps past 80% on any thruster; share of steps outside the recorded conditions; mean removed share m(s) at the visited states (how often the recording controller's command there fell in the removed region); points lost at the past-80% steps; median distance from the visited (state, command) pairs to the 10th nearest training row, in standard units (held-out recorded rows: 2.49).

**Recorded conditions**

| Controller | points | past 80% | outside recorded | removed share | loss from past-80% steps | distance to data |
|---|---|---|---|---|---|---|
| recording controller | 0.0 | 0.14 | 0.01 | 0.17 | 15.2 | 2.48 |
| stall-aware controller | 100.0 | 0.00 | 0.01 | 0.17 | 0.0 | 1.85 |
| rising-thrust controller | 83.6 | 0.10 | 0.01 | 0.17 | 16.4 | 1.85 |
| cloned actor | 56.3 | 0.00 | 0.01 | 0.17 | 0.1 | 1.87 |
| plain TD3+BC on the gapped log | 86.1 | 0.04 | 0.01 | 0.17 | 8.4 | 1.82 |
| actor of the full-log critic (H2) | 91.3 | 0.01 | 0.01 | 0.17 | 0.9 | 1.86 |
| actor of the gapped critic (H2) | 85.5 | 0.05 | 0.01 | 0.17 | 8.4 | 1.84 |
| room check, no extra caution | 87.6 | 0.04 | 0.01 | 0.17 | 6.9 | 1.83 |
| room check, harm-placed | 88.2 | 0.04 | 0.01 | 0.17 | 5.5 | 1.83 |

**Strong current**

| Controller | points | past 80% | outside recorded | removed share | loss from past-80% steps | distance to data |
|---|---|---|---|---|---|---|
| recording controller | 0.0 | 0.47 | 1.00 | 0.57 | 56.6 | 3.94 |
| stall-aware controller | 100.0 | 0.00 | 1.00 | 0.57 | 0.0 | 2.82 |
| rising-thrust controller | -54.1 | 0.84 | 1.00 | 0.58 | 154.1 | 4.33 |
| cloned actor | 54.4 | 0.09 | 1.00 | 0.57 | 7.0 | 3.18 |
| plain TD3+BC on the gapped log | -16.8 | 0.72 | 1.00 | 0.57 | 117.2 | 3.98 |
| actor of the full-log critic (H2) | 84.6 | 0.41 | 1.00 | 0.56 | 13.9 | 3.03 |
| actor of the gapped critic (H2) | -14.3 | 0.73 | 1.00 | 0.57 | 116.1 | 3.93 |
| room check, no extra caution | -10.1 | 0.68 | 1.00 | 0.57 | 111.1 | 3.81 |
| room check, harm-placed | -7.6 | 0.65 | 1.00 | 0.57 | 107.6 | 3.74 |

**Big waves**

| Controller | points | past 80% | outside recorded | removed share | loss from past-80% steps | distance to data |
|---|---|---|---|---|---|---|
| recording controller | 0.0 | 0.20 | 1.00 | 0.24 | 29.1 | 3.09 |
| stall-aware controller | 100.0 | 0.00 | 1.00 | 0.24 | 0.0 | 2.41 |
| rising-thrust controller | 63.8 | 0.26 | 1.00 | 0.24 | 36.2 | 2.49 |
| cloned actor | 43.6 | 0.01 | 1.00 | 0.24 | 0.7 | 2.62 |
| plain TD3+BC on the gapped log | 70.9 | 0.17 | 1.00 | 0.24 | 23.0 | 2.46 |
| actor of the full-log critic (H2) | 86.6 | 0.04 | 1.00 | 0.24 | 1.5 | 2.48 |
| actor of the gapped critic (H2) | 70.0 | 0.16 | 1.00 | 0.24 | 22.7 | 2.47 |
| room check, no extra caution | 72.7 | 0.16 | 1.00 | 0.24 | 20.3 | 2.46 |
| room check, harm-placed | 73.7 | 0.15 | 1.00 | 0.24 | 18.4 | 2.45 |

**Big kicks**

| Controller | points | past 80% | outside recorded | removed share | loss from past-80% steps | distance to data |
|---|---|---|---|---|---|---|
| recording controller | 0.0 | 0.14 | 0.17 | 0.17 | 9.2 | 2.66 |
| stall-aware controller | 100.0 | 0.00 | 0.11 | 0.17 | 0.0 | 1.88 |
| rising-thrust controller | 44.2 | 0.21 | 0.15 | 0.17 | 55.8 | 1.90 |
| cloned actor | 2.0 | 0.07 | 0.18 | 0.17 | 11.4 | 1.99 |
| plain TD3+BC on the gapped log | 48.7 | 0.16 | 0.14 | 0.17 | 44.6 | 1.87 |
| actor of the full-log critic (H2) | 64.3 | 0.11 | 0.13 | 0.16 | 24.3 | 1.90 |
| actor of the gapped critic (H2) | 52.5 | 0.16 | 0.14 | 0.17 | 40.8 | 1.89 |
| room check, no extra caution | 53.7 | 0.15 | 0.14 | 0.17 | 39.7 | 1.88 |
| room check, harm-placed | 54.1 | 0.14 | 0.14 | 0.17 | 37.0 | 1.88 |
