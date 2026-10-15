# ThrusterChain: the exact best controller, and how far each controller falls short of it

All numbers are exact float64 simulations. Nothing was trained; no BCA band was fitted. Points: 0 = the recording controller, 100 = the gate's top reference controller (stall-aware in the stalling world, rising-thrust in the no-stall world). 'Regret' = the optimum's score minus the controller's score, so a smaller regret is better and 0 would be perfect.

## 1. The optimum's score (1,000 starts per situation)

**stalling world**

| situation | optimum points | optimum return | top ref. return |
|---|---|---|---|
| recorded | 100.8 | -25.92 | -26.36 |
| strong cur. | 109.0 | -90.32 | -103.79 |
| big waves | 102.6 | -34.34 | -36.19 |
| big kicks | 106.7 | -156.83 | -163.50 |

**no-stall comparison world**

| situation | optimum points | optimum return | top ref. return |
|---|---|---|---|
| recorded | 100.6 | -25.40 | -25.68 |
| strong cur. | 104.2 | -65.36 | -68.61 |
| big waves | 101.0 | -32.58 | -33.11 |
| big kicks | 103.9 | -145.10 | -149.25 |

## 2. Regret in points (optimum minus controller), mean over seeds

Gate actors: seeds 0-4. Stage 2 (v6) arms: seeds 0-1. The controller's own points are in results.json.

**stalling world**

| controller | rec. | strong | waves | kicks |
|---|---|---|---|---|
| optimum (its points) | 100.8 | 109.0 | 102.6 | 106.7 |
| recording controller (0 pts) (n=1) | 100.8 | 109.0 | 102.6 | 106.7 |
| normal driving only (n=1) | 22.4 | 37.7 | 30.1 | 48.7 |
| stall-aware (100 pts) (n=1) | 0.8 | 9.0 | 2.6 | 6.7 |
| rising-thrust (n=1) | 17.2 | 163.1 | 38.8 | 62.5 |
| clone (n=5) | 44.5 | 54.6 | 59.0 | 104.7 |
| none (gapped critic) (n=5) | 15.3 | 123.3 | 32.6 | 54.2 |
| full (full-log critic) (n=5) | 9.5 | 24.3 | 16.0 | 42.4 |
| random thinning (n=5) | 9.6 | 27.4 | 16.0 | 42.6 |
| plain TD3+BC (n=5) | 14.7 | 125.7 | 31.7 | 58.0 |
| none, alpha 2.5 (n=5) | 17.8 | 121.2 | 36.1 | 60.3 |
| full, alpha 2.5 (n=5) | 14.1 | 34.7 | 23.1 | 51.3 |
| none, seeds 0-1 only (n=2) | 15.7 | 126.2 | 33.4 | 54.9 |
| full, seeds 0-1 only (n=2) | 9.7 | 28.4 | 15.6 | 42.4 |
| bca_k1 (n=2) | 9.8 | 86.9 | 20.4 | 46.3 |
| bca_k2 (n=2) | 10.5 | 77.7 | 19.9 | 47.0 |
| bca_unpaired (n=2) | 10.5 | 88.1 | 20.3 | 47.5 |
| bca_shuffled (n=2) | 9.8 | 86.3 | 18.3 | 47.0 |
| perfect band (n=2) | 7.3 | 29.0 | 12.6 | 36.9 |
| range hinge (n=2) | 7.1 | 15.7 | 10.3 | 31.7 |
| even pull (matched) (n=2) | 18.2 | 124.8 | 38.2 | 62.0 |
| push-less (matched) (n=2) | 18.6 | 121.3 | 32.6 | 52.4 |
| random 1 (matched) (n=2) | 18.5 | 126.4 | 32.4 | 57.2 |
| random 2 (matched) (n=2) | 18.1 | 126.2 | 37.0 | 56.1 |
| stand-in (matched) (n=2) | 7.9 | 35.5 | 13.1 | 36.6 |

**no-stall comparison world**

| controller | rec. | strong | waves | kicks |
|---|---|---|---|---|
| optimum (its points) | 100.6 | 104.2 | 101.0 | 103.9 |
| recording controller (0 pts) (n=1) | 100.6 | 104.2 | 101.0 | 103.9 |
| normal driving only (n=1) | 25.7 | 104.3 | 45.1 | 56.5 |
| stall-aware (n=1) | 2.0 | 49.2 | 7.1 | 17.2 |
| rising-thrust (100 pts) (n=1) | 0.6 | 4.2 | 1.0 | 3.9 |
| clone (n=5) | 52.5 | 128.6 | 89.8 | 94.7 |
| none (gapped critic) (n=5) | 7.1 | 9.2 | 12.9 | 12.6 |
| full (full-log critic) (n=5) | 7.1 | 8.9 | 13.3 | 12.5 |
| none, seeds 0-1 only (n=2) | 7.2 | 9.0 | 13.2 | 12.8 |
| bca_k1 (n=2) | 10.8 | 26.6 | 20.1 | 19.5 |
| perfect band (n=2) | 7.7 | 9.9 | 14.1 | 13.6 |
| range hinge (n=2) | 8.3 | 42.8 | 17.0 | 19.1 |
| even pull (locked) (n=2) | 13.6 | 18.3 | 25.5 | 23.4 |
| stand-in (locked) (n=2) | 8.0 | 31.9 | 16.0 | 18.0 |

**The same regret in return units** (optimum's mean return minus the controller's; the bank-mean return of the optimum is in section 1)

**stalling world**

| controller | rec. | strong | waves | kicks |
|---|---|---|---|---|
| recording controller (0 pts) | 53.88 | 163.62 | 72.47 | 106.41 |
| normal driving only | 11.97 | 56.56 | 21.26 | 48.56 |
| stall-aware (100 pts) | 0.44 | 13.47 | 1.85 | 6.68 |
| rising-thrust | 9.20 | 244.90 | 27.42 | 62.37 |
| clone | 23.77 | 82.01 | 41.69 | 104.47 |
| none (gapped critic) | 8.19 | 185.07 | 23.01 | 54.02 |
| full (full-log critic) | 5.09 | 36.52 | 11.30 | 42.26 |
| random thinning | 5.14 | 41.18 | 11.33 | 42.53 |
| plain TD3+BC | 7.84 | 188.79 | 22.41 | 57.85 |
| none, alpha 2.5 | 9.51 | 182.01 | 25.46 | 60.16 |
| full, alpha 2.5 | 7.52 | 52.15 | 16.30 | 51.14 |
| none, seeds 0-1 only | 8.39 | 189.51 | 23.60 | 54.73 |
| full, seeds 0-1 only | 5.20 | 42.57 | 11.04 | 42.33 |
| bca_k1 | 5.24 | 130.51 | 14.43 | 46.21 |
| bca_k2 | 5.62 | 116.68 | 14.07 | 46.84 |
| bca_unpaired | 5.62 | 132.32 | 14.33 | 47.36 |
| bca_shuffled | 5.24 | 129.63 | 12.89 | 46.92 |
| perfect band | 3.87 | 43.54 | 8.87 | 36.81 |
| range hinge | 3.81 | 23.58 | 7.27 | 31.59 |
| even pull (matched) | 9.74 | 187.43 | 26.97 | 61.80 |
| push-less (matched) | 9.95 | 182.19 | 23.00 | 52.27 |
| random 1 (matched) | 9.88 | 189.75 | 22.90 | 57.04 |
| random 2 (matched) | 9.66 | 189.51 | 26.15 | 55.91 |
| stand-in (matched) | 4.22 | 53.27 | 9.23 | 36.54 |

**no-stall comparison world**

| controller | rec. | strong | waves | kicks |
|---|---|---|---|---|
| recording controller (0 pts) | 48.80 | 81.44 | 51.52 | 110.88 |
| normal driving only | 12.48 | 81.52 | 23.01 | 60.29 |
| stall-aware | 0.96 | 38.44 | 3.61 | 18.40 |
| rising-thrust (100 pts) | 0.28 | 3.25 | 0.53 | 4.15 |
| clone | 25.48 | 100.55 | 45.77 | 101.03 |
| none (gapped critic) | 3.45 | 7.21 | 6.58 | 13.49 |
| full (full-log critic) | 3.46 | 6.93 | 6.78 | 13.38 |
| none, seeds 0-1 only | 3.50 | 7.07 | 6.72 | 13.63 |
| bca_k1 | 5.26 | 20.80 | 10.27 | 20.78 |
| perfect band | 3.74 | 7.75 | 7.20 | 14.47 |
| range hinge | 4.03 | 33.46 | 8.67 | 20.41 |
| even pull (locked) | 6.58 | 14.32 | 13.01 | 24.95 |
| stand-in (locked) | 3.88 | 24.97 | 8.15 | 19.20 |

## 3. Points of the same controllers (mean over seeds)

**stalling world**

| controller | rec. | strong | waves | kicks |
|---|---|---|---|---|
| recording controller (0 pts) | 0.0 | 0.0 | -0.0 | -0.0 |
| normal driving only | 78.4 | 71.3 | 72.5 | 58.0 |
| stall-aware (100 pts) | 100.0 | 100.0 | 100.0 | 100.0 |
| rising-thrust | 83.6 | -54.1 | 63.8 | 44.2 |
| clone | 56.3 | 54.4 | 43.6 | 2.0 |
| none (gapped critic) | 85.5 | -14.3 | 70.0 | 52.5 |
| full (full-log critic) | 91.3 | 84.6 | 86.6 | 64.3 |
| random thinning | 91.2 | 81.5 | 86.6 | 64.1 |
| plain TD3+BC | 86.1 | -16.8 | 70.9 | 48.7 |
| none, alpha 2.5 | 83.0 | -12.2 | 66.6 | 46.4 |
| full, alpha 2.5 | 86.7 | 74.2 | 79.5 | 55.4 |
| none, seeds 0-1 only | 85.1 | -17.2 | 69.2 | 51.8 |
| full, seeds 0-1 only | 91.1 | 80.6 | 87.0 | 64.3 |
| bca_k1 | 91.0 | 22.1 | 82.2 | 60.4 |
| bca_k2 | 90.3 | 31.3 | 82.7 | 59.7 |
| bca_unpaired | 90.3 | 20.8 | 82.3 | 59.2 |
| bca_shuffled | 91.0 | 22.6 | 84.4 | 59.6 |
| perfect band | 93.6 | 80.0 | 90.1 | 69.8 |
| range hinge | 93.7 | 93.3 | 92.3 | 75.0 |
| even pull (matched) | 82.6 | -15.9 | 64.4 | 44.7 |
| push-less (matched) | 82.2 | -12.4 | 70.1 | 54.3 |
| random 1 (matched) | 82.3 | -17.4 | 70.2 | 49.5 |
| random 2 (matched) | 82.8 | -17.2 | 65.6 | 50.6 |
| stand-in (matched) | 92.9 | 73.5 | 89.6 | 70.1 |

**no-stall comparison world**

| controller | rec. | strong | waves | kicks |
|---|---|---|---|---|
| recording controller (0 pts) | 0.0 | 0.0 | -0.0 | 0.0 |
| normal driving only | 74.9 | -0.1 | 55.9 | 47.4 |
| stall-aware | 98.6 | 55.0 | 94.0 | 86.6 |
| rising-thrust (100 pts) | 100.0 | 100.0 | 100.0 | 100.0 |
| clone | 48.1 | -24.4 | 11.3 | 9.2 |
| none (gapped critic) | 93.5 | 94.9 | 88.1 | 91.2 |
| full (full-log critic) | 93.4 | 95.3 | 87.7 | 91.3 |
| none, seeds 0-1 only | 93.4 | 95.1 | 87.9 | 91.1 |
| bca_k1 | 89.7 | 77.6 | 80.9 | 84.4 |
| perfect band | 92.9 | 94.3 | 86.9 | 90.3 |
| range hinge | 92.3 | 61.4 | 84.0 | 84.8 |
| even pull (locked) | 87.0 | 85.9 | 75.5 | 80.5 |
| stand-in (locked) | 92.6 | 72.2 | 85.0 | 85.9 |

## 4. Checks

| check | result |
|---|---|
| replay through the world vs the least-squares objective (largest relative gap, 8,000 plans) | 3.7e-15 |
| largest optimality gap guaranteed by the Frank-Wolfe bound (relative) | 1.8e-12 |
| largest projected gradient | 4.3e-13 |
| second solver (TRF) vs BVLS, largest relative cost gap (200 plans) | 5.7e-16 |
| random and single-command nudges tried / that lowered the cost | 880,000 / 0 |
| largest stalling-world optimal command (knee = 0.800000012) | 0.800000012 |
| stalling world, every at-knee optimal command (140,651): smallest slope of the cost for pushing it past the knee (must be > 0) | 1.200 |
| stalling world, 800,000 sampled full-range plans with the true thrust (769,841 use past-knee commands): plans cheaper than the optimum | 0 (closest relative difference -4.0e-16: rounding) |
| stalling world, local search (L-BFGS-B) with the true thrust: runs (400 starts x 5 starting plans) that beat the optimum | 0 (closest relative difference -4.5e-16: rounding) |
| every controller's return <= the optimum's, start by start (130 controllers x 4,000 starts; smallest regret seen, must be >= 0) | 4.2e-03 |
| the gate's saved cloned actors equal the 'clone' actors scored here | 20 checked, all equal: True |
| controllers' points vs their recorded values.json points (568 numbers) | largest difference 0.0e+00 |
| anchors recomputed vs pre-registered (largest difference) | 0.0e+00 |
| discounted optimum: largest guaranteed relative gap / replay gap | 6.3e-13 / 1.4e-14 |

## 5. How the optimum uses the thrusters

Share of thruster-steps at the limit (0.8 in the stalling world, 1.0 in the no-stall world), and mean |command|.

| world, situation | at limit | steps w/ any at limit | mean abs cmd | past 0.8 |
|---|---|---|---|---|
| stall, recorded | 2.0% | 6.5% | 0.228 | 0.0% |
| stall, strong cur. | 20.0% | 74.9% | 0.455 | 0.0% |
| stall, big waves | 6.0% | 21.9% | 0.265 | 0.0% |
| stall, big kicks | 7.2% | 15.9% | 0.266 | 0.0% |
| no-stall, recorded | 0.5% | 1.7% | 0.229 | 1.6% |
| no-stall, strong cur. | 9.4% | 35.7% | 0.465 | 16.9% |
| no-stall, big waves | 1.7% | 5.9% | 0.264 | 4.8% |
| no-stall, big kicks | 4.5% | 9.8% | 0.272 | 6.1% |

## 6. Where the loss comes from: tracking cost, energy cost, peak, worst starts

Tracking = sum of squared float deviations; energy = 0.3 x sum of squared commands; both per episode (tracking + energy = minus the return). Late peak = the largest float deviation |s| over steps 10-99 of an episode, mean over starts (over all steps the peak is usually the start itself, which no controller changes: 99%, 81%, 97%, 100% of starts in the stalling world's four situations). Worst 5% = mean return of the worst 50 of 1,000 starts.

**stalling world, recorded conditions**

| controller | tracking | energy | late peak | worst 5% |
|---|---|---|---|---|
| optimum | 15.0 | 10.9 | 0.48 | -51.9 |
| stall-aware (100 pts) | 15.3 | 11.0 | 0.48 | -53.9 |
| clone | 41.7 | 8.0 | 0.96 | -100.8 |
| none (gapped critic) | 23.7 | 10.4 | 0.67 | -96.3 |
| full (full-log critic) | 20.7 | 10.3 | 0.61 | -63.6 |
| bca_k1 | 21.6 | 9.6 | 0.64 | -70.7 |
| range hinge | 20.0 | 9.8 | 0.61 | -59.6 |

**stalling world, strong current**

| controller | tracking | energy | late peak | worst 5% |
|---|---|---|---|---|
| optimum | 58.3 | 32.1 | 1.03 | -185.5 |
| stall-aware (100 pts) | 70.8 | 33.0 | 1.18 | -225.6 |
| clone | 149.1 | 23.2 | 1.69 | -354.5 |
| none (gapped critic) | 236.0 | 39.4 | 2.15 | -537.6 |
| full (full-log critic) | 96.0 | 30.9 | 1.32 | -283.4 |
| bca_k1 | 189.9 | 31.0 | 1.97 | -524.0 |
| range hinge | 85.2 | 28.7 | 1.24 | -245.0 |

**stalling world, big waves**

| controller | tracking | energy | late peak | worst 5% |
|---|---|---|---|---|
| optimum | 19.4 | 14.9 | 0.58 | -67.2 |
| stall-aware (100 pts) | 21.2 | 15.0 | 0.61 | -76.2 |
| clone | 64.0 | 12.0 | 1.14 | -150.2 |
| none (gapped critic) | 41.3 | 16.0 | 0.94 | -154.3 |
| full (full-log critic) | 31.4 | 14.3 | 0.76 | -91.9 |
| bca_k1 | 34.8 | 13.9 | 0.85 | -128.3 |
| range hinge | 28.4 | 13.2 | 0.74 | -83.2 |

**stalling world, big kicks**

| controller | tracking | energy | late peak | worst 5% |
|---|---|---|---|---|
| optimum | 141.9 | 14.9 | 1.80 | -388.3 |
| stall-aware (100 pts) | 148.4 | 15.1 | 1.86 | -413.6 |
| clone | 248.6 | 12.7 | 2.77 | -627.4 |
| none (gapped critic) | 194.0 | 16.8 | 2.31 | -545.2 |
| full (full-log critic) | 183.4 | 15.7 | 2.28 | -509.8 |
| bca_k1 | 188.5 | 14.5 | 2.30 | -539.8 |
| range hinge | 174.6 | 13.8 | 2.18 | -485.1 |

**no-stall comparison world, recorded conditions**

| controller | tracking | energy | late peak | worst 5% |
|---|---|---|---|---|
| optimum | 14.2 | 11.2 | 0.46 | -49.4 |
| rising-thrust (100 pts) | 14.4 | 11.3 | 0.45 | -50.1 |
| clone | 43.0 | 7.9 | 0.97 | -103.4 |
| none (gapped critic) | 18.7 | 10.1 | 0.56 | -54.8 |
| full (full-log critic) | 18.7 | 10.2 | 0.56 | -55.0 |
| bca_k1 | 21.2 | 9.5 | 0.63 | -59.9 |

**no-stall comparison world, strong current**

| controller | tracking | energy | late peak | worst 5% |
|---|---|---|---|---|
| optimum | 28.7 | 36.7 | 0.71 | -109.8 |
| rising-thrust (100 pts) | 31.5 | 37.1 | 0.73 | -123.1 |
| clone | 141.5 | 24.4 | 1.61 | -280.1 |
| none (gapped critic) | 37.4 | 35.1 | 0.79 | -121.6 |
| full (full-log critic) | 36.6 | 35.7 | 0.79 | -120.7 |
| bca_k1 | 56.6 | 29.5 | 0.97 | -145.1 |

**no-stall comparison world, big waves**

| controller | tracking | energy | late peak | worst 5% |
|---|---|---|---|---|
| optimum | 17.0 | 15.6 | 0.52 | -58.6 |
| rising-thrust (100 pts) | 17.3 | 15.8 | 0.51 | -60.5 |
| clone | 66.7 | 11.6 | 1.16 | -145.8 |
| none (gapped critic) | 24.3 | 14.9 | 0.64 | -68.5 |
| full (full-log critic) | 24.8 | 14.6 | 0.64 | -68.6 |
| bca_k1 | 29.6 | 13.3 | 0.73 | -78.1 |

**no-stall comparison world, big kicks**

| controller | tracking | energy | late peak | worst 5% |
|---|---|---|---|---|
| optimum | 128.4 | 16.7 | 1.60 | -351.2 |
| rising-thrust (100 pts) | 132.4 | 16.8 | 1.64 | -367.4 |
| clone | 233.9 | 12.2 | 2.66 | -562.1 |
| none (gapped critic) | 143.6 | 15.0 | 1.81 | -379.5 |
| full (full-log critic) | 143.4 | 15.0 | 1.81 | -378.9 |
| bca_k1 | 152.5 | 13.4 | 1.93 | -395.1 |

## 7. The discounted objective (what the learners optimise) vs the 100-step score

Discounted value = sum over 1,000 steps of 0.98^t x reward (truth.py), first 200 starts of each bank.

**7a. A perfect discounted optimiser, scored on the 100-step objective** (points on the same 200 starts)

| world, situation | 100-step optimum | discount-optimal plan | regret |
|---|---|---|---|
| stall, recorded | 100.9 | 100.5 | 0.46 |
| stall, strong cur. | 109.1 | 108.5 | 0.56 |
| stall, big waves | 102.6 | 102.0 | 0.57 |
| stall, big kicks | 109.9 | 109.6 | 0.29 |
| no-stall, recorded | 100.6 | 100.1 | 0.50 |
| no-stall, strong cur. | 104.3 | 103.2 | 1.15 |
| no-stall, big waves | 101.0 | 100.2 | 0.74 |
| no-stall, big kicks | 106.2 | 105.9 | 0.28 |

**7b. Do the two objectives rank the controllers the same way?** (all deterministic controllers of the world)

| world, situation | controllers | Kendall tau | pairs ordered differently |
|---|---|---|---|
| stall, recorded | 90 | 0.951 | 98 of 4005 (2.4%) |
| stall, strong cur. | 90 | 0.968 | 64 of 4005 (1.6%) |
| stall, big waves | 90 | 0.955 | 91 of 4005 (2.3%) |
| stall, big kicks | 90 | 0.956 | 88 of 4005 (2.2%) |
| no-stall, recorded | 38 | 0.963 | 13 of 703 (1.8%) |
| no-stall, strong cur. | 38 | 0.991 | 3 of 703 (0.4%) |
| no-stall, big waves | 38 | 0.972 | 10 of 703 (1.4%) |
| no-stall, big kicks | 38 | 0.980 | 7 of 703 (1.0%) |

**7c. Key comparisons: same sign under both objectives?** (count of seed x situation cases)

| comparison | same sign | cases |
|---|---|---|
| harm (full log minus gapped log, alpha 5) | 20 | 20 |
| harm (full log minus gapped log, alpha 2.5) | 20 | 20 |
| no-stall harm (alpha 5) | 19 | 20 |
| critic-guided minus cloning (alpha 5) | 20 | 20 |
| BCA (bca_k1) minus none | 8 | 8 |
| range hinge minus none | 8 | 8 |
| perfect band minus none | 8 | 8 |
| no-stall BCA (bca_k1) minus none | 8 | 8 |

**7d. Discounted regret (discounted optimum minus controller), mean over seeds**

**stalling world** (discounted optimum value: recorded -18.5, strong cur. -51.9, big waves -23.0, big kicks -134.8)

| controller | rec. | strong | waves | kicks |
|---|---|---|---|---|
| normal driving only | 7.74 | 29.47 | 12.24 | 35.38 |
| stall-aware (100 pts) | 0.16 | 6.77 | 0.87 | 5.26 |
| rising-thrust | 6.09 | 124.27 | 15.31 | 46.45 |
| clone | 14.37 | 43.61 | 22.92 | 75.17 |
| none (gapped critic) | 5.09 | 93.87 | 12.66 | 39.33 |
| full (full-log critic) | 3.08 | 19.75 | 6.28 | 32.29 |
| random thinning | 3.12 | 22.18 | 6.40 | 33.02 |
| plain TD3+BC | 5.10 | 96.31 | 12.69 | 42.70 |
| none, alpha 2.5 | 5.79 | 92.63 | 14.09 | 43.93 |
| full, alpha 2.5 | 4.61 | 28.05 | 9.01 | 38.46 |
| none, seeds 0-1 only | 5.30 | 95.95 | 13.00 | 39.56 |
| full, seeds 0-1 only | 3.16 | 22.83 | 6.27 | 32.36 |
| bca_k1 | 3.16 | 68.12 | 8.22 | 33.18 |
| bca_k2 | 3.42 | 61.51 | 8.01 | 33.45 |
| bca_unpaired | 3.37 | 69.08 | 8.20 | 34.00 |
| bca_shuffled | 3.09 | 68.67 | 7.36 | 34.17 |
| perfect band | 2.36 | 22.76 | 4.91 | 27.67 |
| range hinge | 2.33 | 12.15 | 4.09 | 23.44 |
| even pull (matched) | 6.07 | 95.41 | 14.81 | 44.87 |
| push-less (matched) | 5.83 | 92.80 | 12.61 | 37.00 |
| random 1 (matched) | 6.09 | 96.43 | 12.92 | 41.13 |
| random 2 (matched) | 5.99 | 95.91 | 14.29 | 40.15 |
| stand-in (matched) | 2.59 | 28.42 | 5.24 | 26.89 |

**no-stall comparison world** (discounted optimum value: recorded -18.1, strong cur. -38.9, big waves -21.9, big kicks -125.9)

| controller | rec. | strong | waves | kicks |
|---|---|---|---|---|
| normal driving only | 8.13 | 42.47 | 13.39 | 44.28 |
| stall-aware | 0.56 | 19.77 | 2.02 | 14.17 |
| rising-thrust (100 pts) | 0.04 | 1.38 | 0.13 | 3.35 |
| clone | 15.40 | 51.53 | 25.00 | 72.78 |
| none (gapped critic) | 2.01 | 3.43 | 3.58 | 10.11 |
| full (full-log critic) | 2.00 | 3.31 | 3.60 | 10.04 |
| none, seeds 0-1 only | 2.04 | 3.38 | 3.70 | 10.18 |
| bca_k1 | 3.23 | 10.43 | 5.74 | 15.48 |
| perfect band | 2.17 | 3.73 | 3.91 | 10.82 |
| range hinge | 2.54 | 16.71 | 4.91 | 15.24 |
| even pull (locked) | 3.98 | 7.20 | 7.23 | 18.77 |
| stand-in (locked) | 2.45 | 12.46 | 4.62 | 14.39 |

## 8. Float32 deployment vs the float64 evaluator (full banks)

| actor, situation | max diff (return) | 99.9% (return) | max diff (points) | bank mean (points) |
|---|---|---|---|---|
| gapped, recorded | 2.2e-04 | 1.0e-04 | 4.0e-04 | -3.0e-07 |
| gapped, strong cur. | 7.9e-04 | 4.3e-04 | 5.3e-04 | -7.8e-06 |
| gapped, big waves | 6.7e-05 | 5.5e-05 | 9.4e-05 | -2.4e-06 |
| gapped, big kicks | 2.3e-04 | 1.7e-04 | 2.3e-04 | -2.7e-06 |
| bca_k1, recorded | 3.3e-05 | 3.0e-05 | 6.1e-05 | 1.1e-06 |
| bca_k1, strong cur. | 1.2e-04 | 1.1e-04 | 8.2e-05 | 6.9e-06 |
| bca_k1, big waves | 3.0e-05 | 2.7e-05 | 4.3e-05 | 2.1e-06 |
| bca_k1, big kicks | 8.9e-05 | 7.5e-05 | 8.9e-05 | 9.1e-07 |

## 9. Timings (seconds, wall clock)

| phase | seconds |
|---|---|
| discounted | 36 |
| float32 | 51 |
| nonconvex | 532 |
| score | 210 |
| solve | 138 |
| validate | 12 |

