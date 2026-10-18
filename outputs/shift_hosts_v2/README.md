# Pre-specification version 2 of the four-learner study: the S7 probe and its cloud calibration

These files support the draft of pre-specification version 2 (`experiments/shift_hosts/PRESPEC.md`, section 19). They
lie outside the study's code manifest and outside `gate.EXTERNAL_CODE` (`experiments/shift_hosts/gate.py:56-65,
108-151`), so adding or changing them changes no fingerprint, and they are never imported by the package.

- `probe_s7.py`: the read-only, outcome-blind probe of section 19.6. It runs on the laptop from a copy at
  `/root/s7probe/`, with version 1's bytes still in `/root/bca-hosts` (the commands are at the top of the file). It
  prints differences, ratios, counts, start indices and SHA-256 values only, never a return, a points level, a command
  or a state.
- `calibration/`: the cloud harnesses that set and checked the probe's reading rules on made-up actors only (no trained
  actor, nothing trained): `smoke_probe.py` (all four learners, steps 15,000 and 10,000, the refusals),
  `smoke_negative.py` (made-up defects in the float64 adapter), `calibrate.py` with `run_cal.sh` and `run_final.sh`
  (made-up ReBRAC actors with kernels scaled by 1 to 3, with and without a defect). They ran in the cloud session's
  scratch folder (numpy 2.2.6, JAX 0.6.2 and flax 0.10.7 in a scratch environment; the four situation banks real, the
  three weighting banks made up), and their absolute paths are that folder's.
- `calibration/outputs/`: their outputs. `out_<scale>_<defect>.txt` are the calibration cases of section 19.6 and
  19.7 ("Risk"); `smoke_out_15000.txt`, `smoke_out_10000.txt` and `negative.log` are the final smoke test and the
  negative controls.
