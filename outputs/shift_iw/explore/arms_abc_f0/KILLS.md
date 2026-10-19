# Kill checks across seeds 0-4 (EXPLORATORY)

**EXPLORATORY.** EXPLORATORY: arms A/B/C and result F's no-band control; seen seeds 0-4; pass marks frozen in routes2/arms/PASSMARKS_arms_ABC_and_F_control.md (SHA-256 a073be97003938c112e9ba77581a311e812fe5851f7e64211a11fb4bf47d9ecf) before this code was written. Seeds 0-4 are seen seeds.

Made 2026-10-17T05:42:13+0000. Complete: yes.

## Arm A

| Quality | Stopped by | K2 values (A1 - none at 8,000 steps) | Seeds usable |
|---|---|---|---|
| random | - | 0: 20.01, 1: 17.55, 2: 19.68, 3: 18.89, 4: 13.27 | [0, 1, 2, 3, 4] |
| medium-replay | - | 0: 14.00, 1: 12.76, 2: 10.73, 3: 15.02, 4: 10.47 | [0, 1, 2, 3, 4] |
| medium | - | 0: 7.34, 1: 10.33, 2: 6.10, 3: 7.51, 4: 7.12 | [0, 1, 2, 3, 4] |
| medium-expert | - | 0: -1.80, 1: -1.28, 2: -1.59, 3: -0.71, 4: -1.32 | [0, 1, 2, 3, 4] |
| expert | - | 0: 0.65, 1: 0.62, 2: 0.26, 3: 0.87, 4: 0.43 | [0, 1, 2, 3, 4] |

## Arm B

Stopped by: K1.

- Stopped by K1 on medium-replay; medium; medium-expert: BCA's thruster-1 ceiling 0.85 or higher with at least 30 tries in rung (0.80, 0.85].

- **K0:** {"mean": null, "seeds": 0, "fires": false}
- **K1:** {"where": [], "fires": false}
- **K2:** {"pairs": [], "fires": false}
- **K3:** {"where": [], "fires": false}
- **K4:** {"mean": null, "seeds": 0, "fires": false}

## Arm C

| Quality | Stopped by | Seeds usable |
|---|---|---|
| random | - | [0, 1, 2, 3, 4] |
| medium-replay | - | [0, 1, 2, 3, 4] |
| medium | - | [0, 1, 2, 3, 4] |
| medium-expert | - | [0, 1, 2, 3, 4] |
| expert | - | [0, 1, 2, 3, 4] |

- **K3 (two seeds stop C):** {"where": [], "fires": false, "rule": "two seeds with a non-finite gain, weight, actor or return stop C (as in B)"}.
- **KB0 on the 5-seed means:** {"random": {"mean_share_rho05": 0.017679999999999998, "mean_share_rho1": 0.005236, "passes": false}, "medium-replay": {"mean_share_rho05": 0.038186000000000005, "mean_share_rho1": 0.012509999999999999, "passes": true}, "medium": {"mean_share_rho05": 0.09686399999999999, "mean_share_rho1": 0.027598, "passes": true}}; holds: True.
- **Guard void:** {"by_B": true, "by_B_rules": {"K1": {"where": [], "fires": false}, "K2": {"pairs": [], "fires": false}}, "B_maps_built_for": [], "by_own_map": ["random seed 0", "random seed 1", "random seed 3", "random seed 4", "medium-replay seed 0", "medium-replay seed 1", "medium-replay seed 3", "medium-replay seed 4", "medium seed 0", "medium seed 1", "medium seed 2", "medium seed 3", "medium seed 4", "medium-expert seed 0", "medium-expert seed 1", "medium-expert seed 2", "medium-expert seed 3", "medium-expert seed 4"], "void": true}. B2-B4 scored: False.

## F0

| Quality | Decided | Note |
|---|---|---|
| medium-expert | True | - |
| expert | True | - |

## Arm C: every rule that fired

- none
