"""The stage-0 suite's own module (plan section 4.5 and stop rule S0, section 6.1; area "suite"), on made-up arrays
only: nothing trains on ThrusterChain data, no band is fitted on real arrays, and no learned policy's return is
produced or read. An experiment-local attachment (plan section 3.1) that changes none of the repository's declared
methods.

Two roles, chosen by the device (tests.device(), SHIFT_HOSTS_DEVICE):

  none (the default; training environment, no JAX device; the "summary" phase of run_stage0.sh, which passes
        --suite-dir <the suite run's folder> and --staging <the staging folder's package>)
    plan     Every test plan section 4.5 names, (a1)-(a4) for each learner, (b), (b2), (b3), (c1), (c2) = (a4), (c3),
             (e), (f), (g), (h), (i) and the guard's comparison ((d) is not needed: choice 11 keeps three IQL runs), is
             mapped to the test cases that run it (PLAN below). Each case must have passed in this suite run on every
             device the plan names for it, in the result file the suite copied (with its SHA-256) right after the
             module ran; and every module the suite ran must have passed whole.
    a2       The (a2) card readout per learner (plan 4.5 (a2), choice 19): bit for bit, or a recorded rounding-level
             difference with its largest relative difference, from each learner's own card result.
    gate     runner.py and stages.py (also band_readout.py and a scoring lane) refuse a real prepared file without the
             approval file, before anything trains. Each refused command must exit 2 with "REFUSED", import no
             Unifloral file (python -X importtime), and leave every file it could write as it was (the real prepared
             file's SHA-256 included). Two safety nets make sure that even a gate defect could not train: JAX_PLATFORMS
             names a platform that does not exist (JAX imports but cannot compute), and for the commands the gate
             refuses first, stand-in jax, jaxlib, flax, optax, distrax and chex packages first on PYTHONPATH raise on
             import, which also proves those commands import no JAX at all. The real file passed as --made-up-file
             needs no approval by design (made-up runs) and must be refused by the made-up check instead; it imports
             JAX first (runner.make_schedule imports bca_core), so it runs without the stand-ins. The check runs only
             while neither prespec.json nor any approval file exists in the package: after the freeze the same commands
             would be real runs, so it is skipped then.
    timing   The card timing readout (the gpu role's result) against the plan's figures (section 7.1: one outer step
             alone TD3+BC 0.6 ms, CQL 2.1, IQL 0.33, ReBRAC 1.6; BCA's extra 0.2-0.8 ms per step) and per-run costs
             (section 7.2), restating the plan's cost model (scratchpad wf6/understand/env/cost_model.py, card_run, with
             the review's corrections of wf6/plan/checks/recost.py) with the measured numbers.
    hygiene  The WSL package equals the staging folder (sync.sh's rule: carriage returns dropped from *.py), no
             __pycache__ under /root/bca-hosts, git status shows only the package as new, and the code manifest is
             the one the suite started with.
    After the tests the main block writes runs/shift_hosts/stage0/SUMMARY.json and
    runs/shift_hosts/stage0/manifest_draft.json (each with a sidecar): the draft holds the SHA-256 of every package
    file and of the four Unifloral files (gate.code_manifest). It is a draft: the freeze (prespec.json, the
    fingerprint, the user's terminal approval) comes later.

  gpu (the "timing" phase of run_stage0.sh, first in the suite while the card is cool; training environment, the
       card, one core as the environment report measured the plan's figures, nothing else running)
    timing   One outer step per learner and setup, deterministic full float32 (choice 8), on made-up arrays shaped like
             ThrusterChain's (tests/madeup.build with 500 training episodes, about 45,000 kept rows, as the
             environment report's 45,000 random rows; 180 calibration episodes, a 900-row bank): Unifloral's imported
             step (alone); the restated step with BCA before the first refresh (standard and IW share one program;
             CQL's pre-refresh program, dose_active=False); and after one refresh (standard with one threshold, IW with
             a threshold per row from a real IW table; CQL's dose-active program). Each is jax.jit of lax.scan over one
             chunk of 2,500 outer steps, as the runner scans and the environment report timed; the first call
             (compiling) is not counted, the compile time JAX reports inside a timed call is subtracted (runner's
             compile_clock), and the median and the minimum over four timed calls are kept. Also, per learner, two
             refreshes (bank scores and the calibration; the first includes any compiling), one IW table on the 900-row
             bank (iw.iw_refresh: stop rule S3's check, the table, the lookups), and the card's state (nvidia-smi:
             clocks, power, temperature, throttle reasons) before and after the learner. The losses must be finite and
             BCA's validity flags true (runner.chunk_problem). The card is shared with Windows and slows under sustained
             load (its software power cap), so these are the speeds of that moment.

Run: run_stage0.sh (beside the staging folder's package) runs both roles. By hand, from /root/bca-hosts with the
Dockerfile's variables (INTERFACES.md section 2), PYTHONDONTWRITEBYTECODE=1 and OMP_NUM_THREADS=1:
  card:    SHIFT_HOSTS_DEVICE=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false taskset -c 2 \\
               /opt/bca-venv/bin/python -m experiments.shift_hosts.tests.test_all_stage0
  summary: SHIFT_HOSTS_DEVICE=none JAX_PLATFORMS=cpu taskset -c 6-11 /opt/bca-venv/bin/python \\
               -m experiments.shift_hosts.tests.test_all_stage0 --suite-dir <run folder> [--staging <package folder>]
Results: runs/shift_hosts/stage0/suite/test_all_stage0_{gpu,none}.json (+ sidecars).
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
import types
import unittest
from datetime import datetime, timezone
from pathlib import Path

from experiments.shift_hosts import gate
from experiments.shift_hosts import handoff as H
from experiments.shift_hosts import tests as T
from experiments.shift_hosts import unifloral_import as UI

AREA = "suite"
REPO = T.REPO
PACKAGE = REPO / "experiments" / "shift_hosts"
STAGE0 = T.STAGE0
SUITE = STAGE0 / AREA
TRAINING_PYTHON = "/opt/bca-venv/bin/python"
LEARNERS = ("td3_bc", "cql", "iql", "rebrac")
P, G, N = "cpu", "gpu", "none"
READOUTS = {}
OPTIONS = dict(suite_dir=None, staging=None)

# ------------------------------------------------------------------------------------------------ the plan's tests
# Each entry: plan test, learner (or None), what plan section 4.5 says it proves, and the cases that run it:
# (area, module, "Class.test_method" or "*" for every case of the module, devices on which it must have passed).
# A device left out of a case's tuple is not required there (the plan names no device, or the module skips it there).


def _c(area, module, case, *devices):
    return (area, module, case, tuple(devices))


PLAN = [
    dict(test="a1", learner="td3_bc", proves="the restated step is Unifloral's step (BCA absent, 60 outer steps, "
                                             "processor and card, bit for bit)",
         cases=[_c("td3bc", "test_td3_bc", "A1CopyIsUnifloralTest.test_a1_sixty_steps_bit_for_bit", P, G),
                _c("td3bc", "test_td3_bc", "SourceTest.test_make_train_step_is_unifloral_line_for_line", P, G),
                _c("td3bc", "test_td3_bc", "SourceTest.test_own_target_is_unifloral_compute_target", P, G),
                _c("td3bc", "test_td3_bc", "SourceTest.test_uses_the_imported_modules", P, G)]),
    dict(test="a1", learner="rebrac", proves="as (a1) for TD3+BC",
         cases=[_c("rebrac", "test_rebrac", "A1CopyEqualsUnifloralTest.test_a1_sixty_steps_bit_for_bit", P, G),
                _c("rebrac", "test_rebrac", "A1CopyEqualsUnifloralTest.test_a1_same_traced_program", P, G),
                _c("rebrac", "test_rebrac", "SourceTest.test_make_train_step_lines_are_unifloral_lines", P, G),
                _c("rebrac", "test_rebrac", "SourceTest.test_own_target_is_unifloral_compute_target", P, G),
                _c("rebrac", "test_rebrac", "SourceTest.test_unifloral_pieces_are_the_imported_ones", P, G)]),
    dict(test="a1", learner="cql", proves="as (a1) for TD3+BC",
         cases=[_c("cql", "test_cql", "CopyTest.test_a1_bca_absent_equals_imported", P, G),
                _c("cql", "test_cql", "CopyTest.test_a0_restated_lines_are_unifloral", P, G)]),
    dict(test="a1", learner="iql", proves="as (a1) for TD3+BC",
         cases=[_c("iql", "test_iql", "A1A2Test.test_a1_bca_absent_equals_imported_step", P, G),
                _c("iql", "test_iql", "RestatedTextTest.test_restated_lines_are_unifloral_verbatim", P)]),
    dict(test="a2", learner="td3_bc", proves="BCA present before the first refresh (dose exactly 1) changes nothing: "
                                             "bit for bit on the processor; on the card bit for bit or every leaf "
                                             "within 1e-6 relative after 60 steps",
         cases=[_c("td3bc", "test_td3_bc", "A2BCAPresentDoseOneTest.test_a2_standard_and_iw_before_any_refresh", P, G)]),
    dict(test="a2", learner="rebrac", proves="as (a2) for TD3+BC",
         cases=[_c("rebrac", "test_rebrac", "A2DoseOneEqualsUnifloralTest.test_a2_standard_rule", P, G),
                _c("rebrac", "test_rebrac", "A2DoseOneEqualsUnifloralTest.test_a2_iw_rule", P, G)]),
    dict(test="a2", learner="cql", proves="as (a2) for TD3+BC (CQL's pre-refresh program, dose_active=False)",
         cases=[_c("cql", "test_cql", "CopyTest.test_a2_bca_present_before_refresh_equals_imported", P, G)]),
    dict(test="a2", learner="iql", proves="as (a2) for TD3+BC (IQL's weights exactly Unifloral's)",
         cases=[_c("iql", "test_iql", "A1A2Test.test_a2_bca_present_before_refresh_equals_imported_step", P, G),
                _c("iql", "test_iql", "A1A2Test.test_a2_weights_are_unifloral_exp_adv_in_test_mode", P)]),
    dict(test="a3", learner="td3_bc", proves="a real dose multiplies the intended per-row term at the right "
                                             "strength (loss and gradient = -lambda mean Q1 + mean(m bc))",
         cases=[_c("td3bc", "test_td3_bc", "A3RealDoseTest.test_a3_loss_and_gradient_equal_the_formula", P, G),
                _c("td3bc", "test_td3_bc", "A3RealDoseTest.test_a3_the_dose_changes_nothing_else", P, G),
                _c("td3bc", "test_td3_bc", "A3RealDoseTest.test_a3_cross_check_with_the_imported_file", P, G)]),
    dict(test="a3", learner="rebrac", proves="-lambda mean q + 0.001 mean(m bc); every dose 1.5 = Unifloral at "
                                             "actor_bc_coef 0.0015",
         cases=[_c("rebrac", "test_rebrac", "A3RealDoseTest.test_a3_loss_and_gradient_equal_the_external_formula", P, G),
                _c("rebrac", "test_rebrac", "A3RealDoseTest.test_a3_nothing_else_changes", P, G),
                _c("rebrac", "test_rebrac", "A3RealDoseTest.test_a3_cross_check_dose_15_against_coef_0015", P, G),
                _c("rebrac", "test_rebrac", "A3RealDoseTest.test_a3_per_row_placement_at_a_large_coefficient", P, G)]),
    dict(test="a3", learner="cql", proves="squared error + 10 mean_ik m_i (q_ood_i - Q_k); every dose 1.5 = "
                                          "Unifloral at cql_min_q_weight 15",
         cases=[_c("cql", "test_cql", "DoseAndBandTest.test_a3_dose_scales_each_rows_gap_as_the_external_formula", P),
                _c("cql", "test_cql", "DoseAndBandTest.test_a3_every_dose_one_and_a_half_equals_weight_fifteen", P)]),
    dict(test="a3", learner="iql", proves="mean(w nll) with w from calibration/advantage.py",
         cases=[_c("iql", "test_iql", "A3A4Test.test_a3_fixed_width_loss_and_gradient_equal_the_formula", P)]),
    dict(test="a4", learner="td3_bc", proves="the band reads the learner's own target y and critic q (bit for bit on "
                                             "the processor with Unifloral's key)",
         cases=[_c("td3bc", "test_td3_bc", "A4OwnTargetAndCriticTest.test_a4_with_unifloral_keys_y_and_q_are_unifloral_s_own", P),
                _c("td3bc", "test_td3_bc", "A4OwnTargetAndCriticTest.test_a4_with_bca_keys_y_is_the_same_function_at_bca_s_draw", P)]),
    dict(test="a4", learner="rebrac", proves="as (a4) for TD3+BC",
         cases=[_c("rebrac", "test_rebrac", "A4OwnTargetAndCriticTest.test_a4_with_unifloral_keys_y_and_q_are_unifloral_internal_values", P, G),
                _c("rebrac", "test_rebrac", "A4OwnTargetAndCriticTest.test_a4_control_bca_keys_change_y_only", P, G)]),
    dict(test="a4", learner="cql", proves="as (a4) for TD3+BC (CQL's target and _q_loss_fn's q_pred)",
         cases=[_c("cql", "test_cql", "DoseAndBandTest.test_a4_with_unifloral_key_y_and_q_are_unifloral_own", P),
                _c("cql", "test_cql", "DoseAndBandTest.test_a4_with_bca_key_y_is_the_learners_target_and_unifloral_key_untouched", P)]),
    dict(test="a4", learner="iql", proves="as (a4) for TD3+BC (q_targets and v_target)",
         cases=[_c("iql", "test_iql", "A3A4Test.test_a4_band_reads_the_learners_own_target_and_critic", P)]),
    dict(test="b", learner=None, proves="weights of 1 = standard: IW changes nothing but R (300-row bank, refreshes "
                                        "at 20 and every 10, 50 steps, all four learners)",
         cases=[_c("runner", "test_runner", "WeightsOfOneTest.test_b_weights_of_one_equal_standard", P, G),
                _c("iw", "test_iw", "WeightsOfOneTest.test_grid_is_one_and_R1_is_the_standard_threshold", P, G),
                _c("iw", "test_iw", "WeightsOfOneTest.test_iw_refresh_with_weights_of_one_is_the_standard_state", P, G),
                _c("iw", "test_iw", "WeightsOfOneTest.test_doses_at_the_standard_threshold_are_frozen_level_dose", P, G),
                _c("iw", "test_iw", "WeightsOfOneTest.test_iql_weights_at_the_standard_threshold_are_the_ports", P, G)]),
    dict(test="b2", learner="td3_bc", proves="each batch row reads its own threshold through Unifloral's "
                                             "batch_indices (4 levels)",
         cases=[_c("td3bc", "test_td3_bc", "B2RowThresholdTest.test_b2_each_row_reads_its_own_threshold", P)]),
    dict(test="b2", learner="rebrac", proves="as (b2) for TD3+BC",
         cases=[_c("rebrac", "test_rebrac", "B2PerRowThresholdTest.test_b2_each_row_reads_its_own_threshold", P, G)]),
    dict(test="b2", learner="cql", proves="as (b2) for TD3+BC",
         cases=[_c("cql", "test_cql", "ThresholdAndRefreshTest.test_b2_each_batch_rows_dose_reads_its_own_threshold", P)]),
    dict(test="b2", learner="iql", proves="as (b2) for TD3+BC (IQL's weights)",
         cases=[_c("iql", "test_iql", "B2B3Test.test_b2_each_rows_own_threshold", P)]),
    dict(test="b2", learner=None, proves="the per-row arithmetic itself (iw.py)",
         cases=[_c("iw", "test_iw", "EachRowsOwnThresholdTest.test_doses_and_weights_at_four_levels", P, G),
                _c("iw", "test_iw", "EachRowsOwnThresholdTest.test_table_thresholds_row_by_row", P, G)]),
    dict(test="b3", learner="td3_bc", proves="refused rows (R = +inf on a third of rows): standard dose 1, IW dose "
                                             "1.5, other rows unchanged",
         cases=[_c("td3bc", "test_td3_bc", "B3RefusedRowsTest.test_b3_refused_rows", P)]),
    dict(test="b3", learner="rebrac", proves="as (b3) for TD3+BC",
         cases=[_c("rebrac", "test_rebrac", "B3RefusedRowsTest.test_b3_standard_and_iw_refusal_rules", P, G)]),
    dict(test="b3", learner="cql", proves="as (b3) for TD3+BC",
         cases=[_c("cql", "test_cql", "ThresholdAndRefreshTest.test_b3_refused_rows_standard_and_iw_rules", P)]),
    dict(test="b3", learner="iql", proves="standard weight 0 and the actor step skipped when no row is supported; "
                                          "IW weight 1 where A > 0, Unifloral's elsewhere",
         cases=[_c("iql", "test_iql", "B2B3Test.test_b3_refused_rows_standard_and_iw_rules", P),
                _c("iql", "test_iql", "B2B3Test.test_b3_no_supported_row_skips_the_actor_step", P)]),
    dict(test="b3", learner=None, proves="the refusal arithmetic itself (iw.py)",
         cases=[_c("iw", "test_iw", "RefusedRowsTest.test_doses_under_both_rules", P, G),
                _c("iw", "test_iw", "RefusedRowsTest.test_every_row_refused", P, G),
                _c("iw", "test_iw", "RefusedRowsTest.test_iql_standard_rule_gives_weight_zero_and_skips", P, G),
                _c("iw", "test_iw", "RefusedRowsTest.test_iql_most_caution_gives_plain_cloning_on_positive_rows", P, G),
                _c("iw", "test_iw", "RefusedRowsTest.test_before_the_first_refresh_nothing_changes", P, G),
                _c("iw", "test_iw", "RefusedRowsTest.test_the_refusal_rule_is_a_traced_flag", P, G)]),
    dict(test="c1", learner=None, proves="the shared fit cores and scale-network constructors equal the ports' bit "
                                         "for bit (td3_bc_bca, cql_bca, rebrac_bca fit_scale; BootstrapScaleFitter)",
         cases=[_c("scaffold", "test_bca_core", "*", P)]),
    dict(test="c2", learner=None, proves="covered by (a4): each learner's own y and q",
         cases=[c for e in () for c in e]),          # filled below from the (a4) entries
    dict(test="c3", learner=None, proves="iw.py equals the weighting test's threshold_table, lookup and "
                                         "per_row_dose (originals run in /opt/bca-shift-venv)",
         cases=[_c("iw", "test_iw_compare", "*", P)]),
    dict(test="e", learner="rebrac", proves="ReBRAC's next commands are Unifloral's roll (made-up arrays here; the "
                                            "real arrays and the bank rule in data preparation)",
         cases=[_c("rebrac", "test_rebrac", "ENextCommandTest.test_e_next_commands_are_unifloral_roll", P, G),
                _c("scaffold", "test_import", "TrainingBlockBuilderTest.test_rebrac_roll_is_unifloral_and_madeups", P, G),
                _c("data", "test_prepare", "PreparedTest.test_roll_rule", P),
                _c("data", "test_prepare", "TrainingEnvironmentTest.test_training_environment_reads_and_rolls", P)]),
    dict(test="f", learner=None, proves="import integrity: SHA-256 before every import; XLA_FLAGS repaired before "
                                        "JAX's backend started",
         cases=[_c("scaffold", "test_import", "*", P, G)]),
    dict(test="g", learner=None, proves="the runner's chunk loop equals a literal transcription of each training "
                                        "block for 3 chunks of 50",
         cases=[_c("runner", "test_runner", "TrainingBlockTest.test_g_chunk_loop_equals_training_block", P, G)]),
    dict(test="h", learner=None, proves="the scoring adapters score the trained network (section 3.9, both halves "
                                        "of S7)",
         cases=[_c("scoring", "test_scoring", "*", P)]),
    dict(test="i", learner=None, proves="hand-off files arrive unchanged across the two environments",
         cases=[_c("scaffold", "test_handoff", "*", N)]),
    dict(test="guard", learner=None, proves="the made-up-data guard makes the weighting test's decisions with the "
                                            "same hashes (original run in /opt/bca-shift-venv)",
         cases=[_c("scaffold", "test_guard_compare", "GuardAgainstOriginalTest.test_original_side_ran", N),
                _c("scaffold", "test_guard_compare", "GuardAgainstOriginalTest.test_same_decisions_and_hashes", N),
                _c("scaffold", "test_guard_compare", "GuardAgainstOriginalTest.test_expected_decisions", N),
                _c("scaffold", "test_guard_compare", "BareTreeTest.test_registered_trees_open_bca", N),
                _c("scaffold", "test_guard_compare", "BareTreeTest.test_other_content_is_refused", N),
                _c("scaffold", "test_guard_compare", "BareTreeTest.test_only_the_token_or_a_verified_approval", N),
                _c("scaffold", "test_guard_compare", "BareTreeTest.test_tree_hash_is_learners_rule", N)]),
]
for _e in PLAN:
    if _e["test"] == "c2":
        _e["cases"] = [c for a in PLAN if a["test"] == "a4" for c in a["cases"]]
NOT_NEEDED = {"d": "choice 11 keeps three separate IQL runs, so (d) (a three-actor IQL step) is not needed"}

# every module the suite runs, (area, result stem), and the job label run_stage0.sh gives it
MODULES = [
    ("scaffold", "test_import_cpu", "import_cpu"), ("scaffold", "test_import_gpu", "import_gpu"),
    ("scaffold", "test_handoff_none", "handoff_none"), ("scaffold", "test_guard_compare_none", "guard_none"),
    ("scaffold", "test_bca_core_cpu", "bca_core_cpu"),
    ("td3bc", "test_td3_bc_cpu", "td3_bc_cpu"), ("td3bc", "test_td3_bc_gpu", "td3_bc_gpu"),
    ("rebrac", "test_rebrac_cpu", "rebrac_cpu"), ("rebrac", "test_rebrac_gpu", "rebrac_gpu"),
    ("cql", "test_cql_cpu", "cql_cpu"), ("cql", "test_cql_gpu", "cql_gpu"),
    ("iql", "test_iql_cpu", "iql_cpu"), ("iql", "test_iql_gpu", "iql_gpu"),
    ("iw", "test_iw_cpu", "iw_cpu"), ("iw", "test_iw_gpu", "iw_gpu"), ("iw", "test_iw_compare_cpu", "iw_compare_cpu"),
    ("data", "test_prepare_cpu", "prepare_cpu"), ("scoring", "test_scoring_cpu", "scoring_cpu"),
    ("claims", "test_claims_none", "claims_score"), ("claims", "test_claims_none-training-env", "claims_train"),
    ("runner", "test_runner_cpu", "runner_cpu"), ("runner", "test_runner_gpu", "runner_gpu"),
    ("runner", "test_stages_cpu", "stages_cpu"), ("suite", "test_all_stage0_gpu", "timing_gpu"),
]

# ------------------------------------------------------------------------------------------------ the plan's costs
# Section 7.1-7.3 and the cost model it was computed with (wf6/understand/env/cost_model.py, as corrected by
# wf6/plan/checks/recost.py and recost_runs.py): (low, central, high).
PLAN_ALONE_MS = dict(td3_bc=(0.586, 0.602, 0.607), cql=(2.081, 2.106, 2.111), iql=(0.318, 0.320, 0.339),
                     rebrac=(1.606, 1.611, 1.644))                 # card, deterministic full float32, measured
S6_REFERENCE_MS = dict(td3_bc=0.6, cql=2.1, iql=0.33, rebrac=1.6)  # stop rule S6's references
PLAN_BCA_EXTRA_MS = (0.2, 0.4, 0.8)                                # one band fit per outer step (choice 13)
CARD_LOAD = (1.0, 1.2, 1.5)                                        # Windows shares the card
CARD_COMPILE_S = dict(alone=(20.0, 45.0, 90.0), bca=(40.0, 75.0, 120.0))
CARD_REFRESH_S = (18.0, 36.0, 90.0)                                # per 18 refreshes
IW_EXTRA_S = (110.0, 300.0, 415.0)                                 # IW tables per 18 refreshes
CARD_CPU_SHARE = (0.6, 0.9, 0.97)                                  # processor-s per wall-s of a card run
PLAN_PER_RUN_MIN = dict(td3_bc=(2.0, 3.9, 9.1), cql=(5.0, 6.9, 12.1), iql=(1.4, 3.3, 8.6), rebrac=(4.0, 5.9, 11.1))
PLAN_STAGE_CARD_H = dict(stage1=(0.35, 0.65, 1.04), stage2a=(4.28, 7.12, 11.02), total_0_2a=(4.69, 7.82, 12.15))
PLAN_STAGE_PROC_H = dict(stage2a=(5.47, 10.41, 15.94), total_0_2a=(6.68, 12.13, 18.44))
PLAN_OTHER_PROC_H = dict(stage1=(0.5, 0.64, 0.8), stage2a=(2.9, 4.0, 5.25))   # besides card-run waiting (recost.py)
CAPS = dict(card_hours=14.0, processor_hours=21.0)                 # medium-expert, stages 0-2a
N_OUTER, WARMUP, EVERY, STAGE1_STEPS = 100_000, 10_000, 5_000, 15_000


def n_refreshes(n_outer):
    return max(0, (n_outer - WARMUP) // EVERY)


# ------------------------------------------------------------------------------------------------ small helpers

def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha256(obj):
    """runner.canonical_sha256's rule (the run records' code_manifest_sha256)."""
    return hashlib.sha256(json.dumps(H.plain(obj), sort_keys=True, separators=(",", ":")).encode("utf8")).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def suite_dir():
    if OPTIONS["suite_dir"]:
        return Path(OPTIONS["suite_dir"])
    latest = SUITE / "LATEST"
    return SUITE / latest.read_text(encoding="utf8").strip() if latest.is_file() else None


def load_jobs(run):
    jobs = {}
    for p in sorted((run / "jobs").glob("*.json")) if run else ():
        j = json.loads(p.read_text(encoding="utf8"))
        jobs[j["label"]] = j
    return jobs


def job_label(area, stem):
    """run_stage0.sh's job label for a result file (ReBRAC's details file comes from its module's job)."""
    stem = stem.replace("_details_", "_")
    return next((lab for a, s, lab in MODULES if (a, s) == (area, stem)), None)


def load_result(run, area, stem):
    """(result body, problems) of a result file this suite run copied, after its sidecar and the job's recorded
    SHA-256 (the file the module wrote at the end of its job)."""
    jobs = load_jobs(run)
    label = job_label(area, stem)
    path = run / "results" / area / f"{stem}.json" if run else None
    problems = []
    if path is None or not path.is_file():
        return None, [f"{area}/{stem}.json: not produced in this suite run"]
    digest = sha256_file(path)
    side = Path(str(path) + ".sha256")
    if not side.is_file() or side.read_text(encoding="utf8").split()[0] != digest:
        problems.append(f"{area}/{stem}.json: its sidecar is missing or differs")
    job = jobs.get(label)
    if job is None:
        problems.append(f"{area}/{stem}.json: no job record {label}")
    elif job["results"].get(f"{area}/{stem}.json") != digest:
        problems.append(f"{area}/{stem}.json: not the file the job recorded")
    return json.loads(path.read_text(encoding="utf8")), problems


def case_outcomes(body, suffix):
    """{case name: outcome} of a result's cases ending in '.<suffix>' (or every case for '*')."""
    out = {}
    for c in (body or {}).get("cases", []):
        name = c["name"]
        short = name.split(".", 1)[1] if "." in name else name
        if suffix == "*" or name == suffix or name.endswith("." + suffix):
            out[short] = c["outcome"]
    return out


def plan_table(run):
    """Every PLAN entry with each case's outcome per required device, and the problems found."""
    rows, problems = [], []
    cache = {}
    for e in PLAN:
        cases = []
        for area, module, case, devices in e["cases"]:
            outcome = {}
            for dev in devices:
                stem = f"{module}_{dev}"
                if (area, stem) not in cache:
                    cache[(area, stem)] = load_result(run, area, stem)
                body, probs = cache[(area, stem)]
                problems += [p for p in probs if p not in problems]
                got = case_outcomes(body, case)
                if not got:
                    outcome[dev] = "missing"
                    problems.append(f"({e['test']}) {area}/{stem}: no case {case}")
                elif all(v == "passed" for v in got.values()):
                    outcome[dev] = "passed" if case != "*" else f"passed ({len(got)} cases)"
                else:
                    bad = {k: v for k, v in got.items() if v != "passed"}
                    outcome[dev] = f"NOT passed: {bad}"
                    problems.append(f"({e['test']}) {area}/{stem}: {bad}")
            cases.append(dict(area=area, module=module, case=case, devices=list(devices), outcome=outcome))
        rows.append(dict(test=e["test"], learner=e["learner"], proves=e["proves"], cases=cases,
                         passed=all(all(str(v).startswith("passed") for v in c["outcome"].values()) for c in cases)
                         and bool(cases)))
    return rows, problems


# ------------------------------------------------------------------------------------------------ (a2) on the card

def _a2(run, learner, dev):
    """The (a2) record of a learner on a device: dict(match, max_relative, ...) from its own result file."""
    area = dict(td3_bc="td3bc", rebrac="rebrac", cql="cql", iql="iql")[learner]
    if learner == "td3_bc":
        body, _ = load_result(run, area, f"test_td3_bc_{dev}")
        a2 = (((body or {}).get("extra") or {}).get("details") or {}).get("a2") or {}
        recs = {k: a2[k] for k in ("standard", "iw") if isinstance(a2.get(k), dict)}
        if not recs:
            return None
        bit = all(r["bit_for_bit"] for r in recs.values())
        out = dict(match="bit for bit" if bit else "rounding", max_relative=max(r["max_rel"] for r in recs.values()),
                   state_and_key_bit_for_bit=all(r.get("state_and_key_bit_for_bit") for r in recs.values()),
                   losses_bit_for_bit=all(r.get("losses_bit_for_bit") for r in recs.values()),
                   differing=sorted({u["path"] for r in recs.values() for u in r.get("unequal", [])}),
                   setups={k: r.get("verdict") for k, r in recs.items()})
    elif learner == "rebrac":
        body, _ = load_result(run, area, f"test_rebrac_details_{dev}")
        d = (body or {}).get("details") or {}
        recs = {k: d[f"a2_{k}"] for k in ("standard", "iw") if isinstance(d.get(f"a2_{k}"), dict)}
        if not recs:
            return None
        bit = all(r["carry"]["bit_for_bit"] and r["losses"]["bit_for_bit"] for r in recs.values())
        out = dict(match="bit for bit" if bit else "rounding",
                   max_relative=max(max(r["carry"]["max_leaf_relative_difference"],
                                        r["losses"]["max_leaf_relative_difference"]) for r in recs.values()),
                   state_and_key_bit_for_bit=all(r["carry"]["bit_for_bit"] for r in recs.values()),
                   losses_bit_for_bit=all(r["losses"]["bit_for_bit"] for r in recs.values()),
                   setups={k: "bit for bit" if r["carry"]["bit_for_bit"] and r["losses"]["bit_for_bit"] else "rounding"
                           for k, r in recs.items()})
    elif learner == "cql":
        body, _ = load_result(run, area, f"test_cql_{dev}")
        a2 = (((body or {}).get("extra") or {}).get("details") or {}).get("a2") or {}
        pre = {k: a2[f"pre_refresh_{k}"] for k in ("standard", "iw") if isinstance(a2.get(f"pre_refresh_{k}"), dict)}
        if not pre:
            return None
        bit = all(r["bit_for_bit"] for r in pre.values())
        act = {k: a2[f"dose_active_{k}"] for k in ("standard", "iw") if isinstance(a2.get(f"dose_active_{k}"), dict)}
        out = dict(match="bit for bit" if bit else "rounding", max_relative=max(r["max_rel"] for r in pre.values()),
                   program="the pre-refresh program (dose_active=False), which CQL's BCA runs use until the first "
                           "refresh",
                   state_and_key_bit_for_bit=all((r.get("state") or {}).get("bit_for_bit") for r in pre.values()),
                   losses_bit_for_bit=all((r.get("losses") or {}).get("bit_for_bit") for r in pre.values()),
                   single_program_readout=dict(
                       what="the dose-active program at dose exactly 1 (one program per run, as INTERFACES.md first "
                            "described; what a never-refreshed sham alone would run): recorded, not asserted",
                       bit_for_bit=all(r["bit_for_bit"] for r in act.values()) if act else None,
                       max_relative=max(r["max_rel"] for r in act.values()) if act else None,
                       within_1e6=all(r["max_rel"] <= 1e-6 for r in act.values()) if act else None))
    else:
        body, _ = load_result(run, area, f"test_iql_{dev}")
        a2 = (((body or {}).get("extra") or {}).get("readouts") or {}).get("a2") or {}
        if not a2:
            return None
        bit = bool(a2["bit_for_bit_state"] and a2["bit_for_bit_losses"] and a2["bit_for_bit_key"])
        out = dict(match="bit for bit" if bit else "rounding",
                   max_relative=max(a2["max_relative_state"], a2["max_relative_losses"]),
                   state_and_key_bit_for_bit=bool(a2["bit_for_bit_state"] and a2["bit_for_bit_key"]),
                   key_bit_for_bit=bool(a2["bit_for_bit_key"]), losses_bit_for_bit=bool(a2["bit_for_bit_losses"]),
                   per_group_relative=a2.get("per_group_relative"))
    out["within_card_rule"] = out["match"] == "bit for bit" or out["max_relative"] <= 1e-6
    return out


def a2_readout(run):
    out = {}
    for L in LEARNERS:
        cpu, gpu = _a2(run, L, P), _a2(run, L, G)
        out[L] = dict(processor=cpu, card=gpu)
        if gpu is not None:
            out[L]["verdict"] = (
                "bit for bit on the card" if gpu["match"] == "bit for bit" else
                f"a recorded rounding-level difference on the card (largest relative {gpu['max_relative']:.3g}, "
                f"within 1e-6: {gpu['within_card_rule']}); by the plan's letter S10 becomes a readout and choice 19 "
                f"applies")
    body, _ = load_result(run, "runner", "test_runner_gpu")
    g = (((body or {}).get("extra") or {}).get("readouts") or {}).get("g") or {}
    out["runner_g_bca_part_after_150_steps"] = {L: dict(bit_for_bit=v.get("bca_part_bit_for_bit"),
                                                        max_relative=v.get("bca_part_max_relative"))
                                                for L, v in g.items()}
    return out


# ------------------------------------------------------------------------------------------------ the timing readout

def _costs(ms_alone, ms_pre, ms_post, refresh_first_s, refresh_s, table_s, k=1):
    """Card wall seconds of one run of each setup, by the plan's model (cost_model.card_run with recost.py's
    corrections) with the measured per-step times, refresh and table seconds (k: 0 low, 1 central, 2 high load and
    start-up). ms_pre applies to the first 10,000 outer steps, ms_post to the rest. The first refresh may compile
    (refresh_first_s), the others reuse it (refresh_s). IW adds 20 tables: one per refresh, and at the last refresh the
    estimated-weight table (choice 20) and the shuffled-weights placebo (plan 4.6) that
    runner.last_refresh_readouts builds."""
    load = CARD_LOAD[k]
    nref = n_refreshes(N_OUTER)

    def bca(n, pre, post):
        warm = min(n, WARMUP)
        return (warm * pre + (n - warm) * post) / 1000.0 * load

    def refreshes(n):
        return 0.0 if n == 0 else refresh_first_s + (n - 1) * refresh_s

    alone = N_OUTER * ms_alone / 1000.0 * load + CARD_COMPILE_S["alone"][k]
    standard = bca(N_OUTER, ms_pre, ms_post) + CARD_COMPILE_S["bca"][k] + refreshes(nref)
    iw = standard + table_s * (nref + 2)
    stage1 = dict(alone=STAGE1_STEPS * ms_alone / 1000.0 * load + CARD_COMPILE_S["alone"][k],
                  standard=bca(STAGE1_STEPS, ms_pre, ms_post) + CARD_COMPILE_S["bca"][k]
                  + refreshes(n_refreshes(STAGE1_STEPS)))
    stage1["iw"] = stage1["standard"] + table_s * n_refreshes(STAGE1_STEPS)
    return dict(alone=alone, standard=standard, iw=iw, stage1=stage1)


def timing_readout(run):
    """The gpu role's measurements against the plan's figures and the budget (module docstring)."""
    body, problems = load_result(run, AREA, "test_all_stage0_gpu")
    t = (((body or {}).get("extra") or {}).get("timing")) if body else None
    if not t:
        return dict(recorded=False, problems=problems)
    out = dict(recorded=True, problems=problems, data=t.get("data"), device=t.get("device"),
               chunk=t.get("chunk"), learners={})
    per_run, stage1, sham = {}, {}, {}
    for L in LEARNERS:
        r = t["learners"][L]
        progs = ("alone", "standard_before_refresh", "iw_before_refresh", "standard_after_refresh", "iw_after_refresh")
        ms = {k: r[k]["median_ms"] for k in progs}
        ms_min = {k: r[k].get("min_ms") for k in progs}
        refresh_first_s, refresh_s = r["refresh_seconds"][0], r["refresh_seconds"][-1]   # the first may compile
        table_s = r["iw_table_seconds"]
        pre = max(ms["standard_before_refresh"], ms["iw_before_refresh"])
        post_std, post_iw = ms["standard_after_refresh"], ms["iw_after_refresh"]
        costs = {k: _costs(ms["alone"], pre, max(post_std, post_iw), refresh_first_s, refresh_s, table_s, k)
                 for k in range(3)}
        per_run[L] = costs
        out["learners"][L] = dict(
            ms_per_outer_step=ms, ms_per_outer_step_min=ms_min,
            card_before=r.get("card_before"), card_after=r.get("card_after"),
            plan_alone_ms=PLAN_ALONE_MS[L][1], s6_reference_ms=S6_REFERENCE_MS[L],
            alone_vs_plan=ms["alone"] / PLAN_ALONE_MS[L][1],
            alone_within_twice_s6_reference=ms["alone"] <= 2 * S6_REFERENCE_MS[L],
            bca_extra_ms=dict(before_refresh=pre - ms["alone"], after_refresh=max(post_std, post_iw) - ms["alone"]),
            bca_extra_within_plan_range=PLAN_BCA_EXTRA_MS[0] <= max(pre, post_std, post_iw) - ms["alone"]
            <= PLAN_BCA_EXTRA_MS[2],
            refresh_seconds=r["refresh_seconds"], iw_table_seconds=table_s, bank_rows=r.get("bank_rows"),
            grid_points=r.get("grid_points"),
            first_call_seconds={k: r[k]["first_call_seconds"] for k in ms},
            per_run_card_minutes={s: dict(measured_central=round(costs[1][s] / 60.0, 2),
                                          measured_low=round(costs[0][s] / 60.0, 2),
                                          measured_high=round(costs[2][s] / 60.0, 2),
                                          plan_central=PLAN_PER_RUN_MIN[L][1],
                                          plan_range=[PLAN_PER_RUN_MIN[L][0], PLAN_PER_RUN_MIN[L][2]])
                                  for s in ("alone", "standard", "iw")})
        stage1[L] = costs
        # choice 19's sham alone: the BCA program never refreshed, 100,000 steps at the pre-refresh speed
        sham[L] = [N_OUTER * pre / 1000.0 * CARD_LOAD[k] + CARD_COMPILE_S["bca"][k] for k in range(3)]
    totals = {}
    for k, name in enumerate(("low", "central", "high")):
        s2 = sum(5 * (per_run[L][k]["alone"] + per_run[L][k]["standard"] + per_run[L][k]["iw"]
                      + per_run[L][k]["alone"]) for L in LEARNERS) / 3600.0
        s1 = sum(stage1[L][k]["stage1"]["alone"] + stage1[L][k]["stage1"]["standard"]
                 + 3 * stage1[L][k]["stage1"]["iw"] for L in LEARNERS) / 3600.0
        sh = dict(td3_bc_iql=(sham["td3_bc"][k] + sham["iql"][k]) / 3600.0,
                  all_four=sum(sham[L][k] for L in LEARNERS) / 3600.0)
        proc = dict(stage1=s1 * CARD_CPU_SHARE[k] + PLAN_OTHER_PROC_H["stage1"][k],
                    stage2a=s2 * CARD_CPU_SHARE[k] + PLAN_OTHER_PROC_H["stage2a"][k])
        totals[name] = dict(stage1_card_hours=round(s1, 2), stage2a_card_hours=round(s2, 2),
                            sham_alone_card_hours=dict((a, round(b, 2)) for a, b in sh.items()),
                            stage1_processor_hours=round(proc["stage1"], 2),
                            stage2a_processor_hours=round(proc["stage2a"], 2),
                            plan_stage1_card_hours=PLAN_STAGE_CARD_H["stage1"][k],
                            plan_stage2a_card_hours=PLAN_STAGE_CARD_H["stage2a"][k],
                            plan_stage2a_processor_hours=PLAN_STAGE_PROC_H["stage2a"][k])
    out["totals"] = totals
    out["model"] = ("the plan's card_run (cost_model.py with recost.py's corrections): load factor (1.0, 1.2, 1.5), "
                    "compile seconds alone (20, 45, 90) and with BCA (40, 75, 120), per-step times, refresh and IW "
                    "table seconds measured here; IW adds 20 tables per run; stage 1 = per learner alone, standard and "
                    "three IW runs (IW, weights of 1, repeat) to 15,000 outer steps, one at a time; stage 2a = per "
                    "learner 5 seeds x (alone, standard, IW, room); processor-hours = card time x the card run's "
                    "processor share (0.6, 0.9, 0.97) + the plan's other processor work")
    return out


# ------------------------------------------------------------------------------------------------ the real-data gate

BLOCKED = ("jax", "jaxlib", "flax", "optax", "distrax", "chex")
REAL_PREPARED = REPO / "runs" / "shift_hosts" / "data" / "medium-expert" / "seed0" / "prepared.npz"


def _approval_present():
    return [p.name for p in PACKAGE.iterdir() if p.name == gate.PRESPEC_NAME or p.name.startswith(gate.APPROVAL_PREFIX)]


def _watch():
    """The files a refused command could have written or changed: the run roots (every file under runs/shift_hosts/
    outside stage0/ and the copied source data), the real prepared files, and the package folder."""
    root = REPO / "runs" / "shift_hosts"
    out = {}
    for p in sorted(root.iterdir()) if root.is_dir() else ():
        if p.name in ("stage0", "source", "scoring_cache"):
            out[p.name + "/"] = "skipped (not a run root)"
            continue
        if p.is_file():
            st = p.stat()
            out[p.name] = (st.st_size, st.st_mtime_ns)
            continue
        for f in sorted(p.rglob("*")):
            if f.is_file():
                st = f.stat()
                out[f.relative_to(root).as_posix()] = (st.st_size, st.st_mtime_ns)
    for f in sorted(PACKAGE.rglob("*")):
        if f.is_file():
            st = f.stat()
            out["package/" + f.relative_to(PACKAGE).as_posix()] = (st.st_size, st.st_mtime_ns)
    return out


def _blockers(folder):
    """Stand-in packages that raise on import, first on PYTHONPATH (module docstring)."""
    folder.mkdir(parents=True, exist_ok=True)
    for name in BLOCKED:
        d = folder / name
        d.mkdir(exist_ok=True)
        (d / "__init__.py").write_text(
            f"raise ImportError('blocked by the stage-0 suite gate check: {name} was imported (nothing may train)')\n",
            encoding="utf8")
    return folder


NO_BACKEND = "suite_gate_blocked"     # JAX_PLATFORMS: JAX imports, but no backend can start, so nothing computes


def _refused_command(argv, blockers=None, timeout=180):
    """Run one command of the training environment as a subprocess (Dockerfile variables), with -X importtime and
    JAX_PLATFORMS set to a platform that does not exist (no JAX computation can run); with `blockers`, the stand-in
    packages first on PYTHONPATH (no JAX import can succeed). Returns its record."""
    env = UI.launch_env(dict(os.environ))
    env.update(JAX_PLATFORMS=NO_BACKEND, SHIFT_HOSTS_DEVICE="cpu", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1")
    env.pop("PYTHONPATH", None)
    if blockers is not None:
        env["PYTHONPATH"] = str(blockers)
    env.pop("XLA_FLAGS", None)
    t0 = time.monotonic()
    out = subprocess.run([TRAINING_PYTHON, "-X", "importtime", "-m"] + argv, cwd=str(REPO), env=env,
                         capture_output=True, text=True, timeout=timeout)
    seconds = time.monotonic() - t0
    imported = []
    other = []
    for line in out.stderr.splitlines():
        if line.startswith("import time:"):
            name = line.rsplit("|", 1)[-1].strip()
            if name and name != "imported package":
                imported.append(name)
        else:
            other.append(line)
    top = {n.split(".")[0] for n in imported}
    return dict(argv=argv, exit=out.returncode, seconds=round(seconds, 2), refused="REFUSED" in out.stderr,
                stand_in_jax=blockers is not None, jax_platforms=NO_BACKEND,
                message=next((l for l in other if "REFUSED" in l), "")[:600],
                imported_modules=len(imported), jax_imported=bool(top & set(BLOCKED)),
                unifloral_imported=any(n.startswith("shift_hosts_unifloral") for n in imported),
                blocked_import_hit="blocked by the stage-0 suite gate check" in out.stderr,
                stdout_tail=out.stdout[-300:], stderr_tail="\n".join(other[-6:])[-800:])


def gate_check(work):
    """The refusals of module docstring (gate). Returns the record."""
    present = _approval_present()
    if present:
        return dict(ran=False, why=f"skipped: {present} exist in the package, so these commands would be real runs")
    work = Path(work)
    if work.exists():
        shutil.rmtree(work)
    blockers = _blockers(work / "blockers")
    man = H.read_manifest(REAL_PREPARED)              # the manifest only (data preparation's output, no learning)
    meta = man.get("meta") or {}
    real = dict(path=str(REAL_PREPARED.relative_to(REPO)), schema=meta.get("schema"), quality=meta.get("quality"),
                seed=meta.get("seed"), made_up=meta.get("made_up"), rows=meta.get("rows"),
                npz_sha256=sha256_file(REAL_PREPARED))
    fake_run = work / "fake_real_run"                 # a run folder whose record says real (band_readout's entry)
    fake_run.mkdir(parents=True)
    H.write_json(fake_run / "record.json", dict(run_id="suite-gate-check/fake", learner="td3_bc", setup="standard",
                                                data=dict(made_up=False), note="not a run: the suite's gate check"))
    before = _watch()
    s = str(REAL_PREPARED.relative_to(REPO))
    # (argv, refused before any JAX import?): every gate refusal comes first; the real file passed as a made-up
    # file needs no approval by design and is refused by the made-up check (runner.prepared_header) after the
    # schedule is made, which imports bca_core and so JAX, but no backend can start (NO_BACKEND)
    commands = [
        (["experiments.shift_hosts.runner", "run", "--learner", "td3_bc", "--setup", "alone", "--quality",
          "medium-expert", "--seed", "0"], True),
        (["experiments.shift_hosts.runner", "run", "--learner", "cql", "--setup", "standard", "--quality",
          "medium-expert", "--seed", "0"], True),
        (["experiments.shift_hosts.runner", "run", "--learner", "iql", "--setup", "iw", "--quality", "medium",
          "--seed", "4", "--stage", "stage1", "--steps", "15000"], True),
        (["experiments.shift_hosts.runner", "run", "--learner", "rebrac", "--setup", "alone", "--made-up-file", s],
         False),
        (["experiments.shift_hosts.stages", "run", "--stage0-card-hours", "0.3", "--stage0-processor-hours", "1.0"],
         True),
        (["experiments.shift_hosts.stages", "start", "--stage0-card-hours", "0.3", "--stage0-processor-hours",
          "1.0"], True),
        (["experiments.shift_hosts.stages", "lane", "score", "--actor", str(work / "no_actor.npz"), "--out",
          str(work / "no_scores.npz")], True),
        (["experiments.shift_hosts.band_readout", "coverage", "--run-dir", str(fake_run), "--out",
          str(work / "no_coverage")], True),
    ]
    results = []
    for argv, before_jax in commands:
        r = _refused_command(argv, blockers if before_jax else None)
        r["refused_before_any_jax_import"] = before_jax
        results.append(r)
    after = _watch()
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    real["npz_sha256_after"] = sha256_file(REAL_PREPARED)
    shutil.rmtree(blockers)
    return dict(ran=True, real_prepared=real, commands=results, changed_files=changed,
                watched_files=len(before), approval_files_present=present)


# ------------------------------------------------------------------------------------------------ hygiene

def hygiene(staging=None):
    out = {}
    pyc = [str(p) for p in REPO.rglob("__pycache__") if ".git" not in p.parts]
    out["pycache"] = pyc
    st = subprocess.run(["git", "--no-optional-locks", "-C", str(REPO), "status", "--porcelain"], capture_output=True,
                        text=True)
    out["git_status"] = st.stdout.splitlines()
    out["git_head"] = subprocess.run(["git", "--no-optional-locks", "-C", str(REPO), "rev-parse", "HEAD"],
                                     capture_output=True, text=True).stdout.strip()
    if staging:
        staging = Path(staging)
        mism = []
        names = set()
        for root in (staging, PACKAGE):
            for p in root.rglob("*"):
                rel = p.relative_to(root)
                if p.is_file() and "__pycache__" not in rel.parts and not p.name.startswith(gate.APPROVAL_PREFIX):
                    names.add(rel.as_posix())
        for n in sorted(names):
            a, b = staging / n, PACKAGE / n
            if not a.is_file() or not b.is_file():
                mism.append(f"{n}: only in {'WSL' if b.is_file() else 'staging'}")
                continue
            ab = a.read_bytes()
            if n.endswith(".py"):
                ab = ab.replace(b"\r\n", b"\n")      # sync.sh drops carriage returns from *.py
            if ab != b.read_bytes():
                mism.append(f"{n}: differs")
        out["staging_vs_wsl"] = dict(checked=True, files=len(names), mismatches=mism)
    else:
        out["staging_vs_wsl"] = dict(checked=False)
    return out


def hygiene_manifest(run, tag):
    """(canonical SHA-256, file count) of the code manifest run_stage0.sh recorded in hygiene/<tag>.txt, or None."""
    path = run / "hygiene" / f"{tag}.txt" if run else None
    if path is None or not path.is_file():
        return None
    lines = path.read_text(encoding="utf8").splitlines()
    for i, line in enumerate(lines):
        if line.startswith("== code manifest") and i + 1 < len(lines):
            words = lines[i + 1].split()
            if len(words) == 2:
                return words[0], int(words[1])
    return None


def s9_draft(run):
    """A draft of prespec.json's environment.s9_signature (runner report section 8, item 5): stages.env_signature
    of the suite's card timing process (its unifloral_import.record()). Informational."""
    body, _ = load_result(run, AREA, "test_all_stage0_gpu")
    ir = (((body or {}).get("extra") or {}).get("unifloral_import")) if body else None
    if not ir:
        return None
    from experiments.shift_hosts import stages as ST

    return dict(signature=ST.env_signature(dict(environment=dict(import_record=ir))),
                source="the suite's card timing process (tests/test_all_stage0.py, gpu role, taskset core 2)",
                caveat="a draft only: a card run the stage driver starts gets the same recorded variables (the "
                       "Dockerfile's, SHIFT_HOSTS_DEVICE=gpu, XLA_PYTHON_CLIENT_PREALLOCATE=false) when the driver is "
                       "started without JAX_PLATFORMS or CUDA_VISIBLE_DEVICES. On real data the driver has no fallback: "
                       "it refuses to start without prespec.json's complete environment.s9_signature, and every real "
                       "run checks it before its first chunk")


# ------------------------------------------------------------------------------------------------ the manifest draft

def manifest_draft(run):
    man = gate.code_manifest()
    pins = json.loads((REPO / "configs" / "unifloral.json").read_text(encoding="utf8"))["source_sha256"]
    unifloral = {}
    for L in LEARNERS:
        rel = f"baselines/unifloral/algorithms/{L}.py"
        unifloral[rel] = dict(sha256=man.get(rel), pinned=pins.get(f"algorithms/{L}.py"),
                              equal=man.get(rel) == pins.get(f"algorithms/{L}.py"))
    package = {k: v for k, v in man.items() if k.startswith("experiments/shift_hosts/")}
    external = {k: v for k, v in man.items() if not k.startswith("experiments/shift_hosts/")}
    return dict(schema="shift-hosts-manifest-draft-v1", draft=True, made_utc=utc_now(),
                about=("A draft of the code manifest for the freeze (plan section 10, item 15): the SHA-256 of every "
                       "package file and of the repository files the package runs (gate.code_manifest, the same "
                       "function the gate fingerprints). Not the freeze: prespec.json, PRESPEC.md and approve_hosts.py "
                       "are not in the package yet, so no fingerprint exists."),
                suite_run=None if run is None else run.name,
                code_manifest_sha256=canonical_sha256(man), files=len(man), package_files=len(package),
                unifloral_four=unifloral, package=package, external=external,
                not_yet_in_the_package=[n for n in ("PRESPEC.md", "prespec.json", "approve_hosts.py")
                                        if not (PACKAGE / n).exists()],
                fingerprint=None)


# ------------------------------------------------------------------------------------------------ tests (none role)

class _Cache:
    run = None
    plan = None
    problems = None


def _plan():
    if _Cache.plan is None:
        _Cache.run = suite_dir()
        _Cache.plan, _Cache.problems = plan_table(_Cache.run)
    return _Cache.run, _Cache.plan, _Cache.problems


class PlanTest(unittest.TestCase):
    def test_every_plan_test_passed_on_its_devices(self):
        """(a1)-(a4), (b), (b2), (b3), (c1), (c2), (c3), (e)-(i), the guard: each case passed in this suite run."""
        run, rows, problems = _plan()
        self.assertIsNotNone(run, "no suite run folder")
        READOUTS["plan"] = rows
        READOUTS["plan_problems"] = problems
        ids = {r["test"] for r in rows}
        for t in ("a1", "a2", "a3", "a4", "b", "b2", "b3", "c1", "c2", "c3", "e", "f", "g", "h", "i", "guard"):
            self.assertIn(t, ids)
        for L in LEARNERS:
            for t in ("a1", "a2", "a3", "a4", "b2", "b3"):
                self.assertTrue(any(r["test"] == t and r["learner"] == L for r in rows), (t, L))
        self.assertEqual(problems, [])
        self.assertTrue(all(r["passed"] for r in rows))

    def test_every_module_of_the_suite_passed(self):
        """Every module the suite ran exited 0 and its result says passed (whole, not only the plan's cases)."""
        run = suite_dir()
        jobs = load_jobs(run)
        mods, bad = [], []
        for area, stem, label in MODULES:
            job = jobs.get(label)
            body, probs = load_result(run, area, stem)
            rec = dict(module=f"{area}/{stem}", label=label, exit=None if job is None else job["exit"],
                       seconds=None if job is None else job["seconds"],
                       passed=None if body is None else body.get("passed"),
                       tests_run=None if body is None else body.get("tests_run"),
                       skipped=None if body is None else body.get("skipped"),
                       failures=None if body is None else body.get("failures"),
                       errors=None if body is None else body.get("errors"),
                       processor_seconds=None if body is None else body.get("processor_seconds"),
                       device=None if job is None else job["device"], problems=probs)
            mods.append(rec)
            if job is None or job["exit"] != 0 or body is None or not body.get("passed") or probs:
                bad.append(rec)
        READOUTS["modules"] = mods
        self.assertEqual(bad, [])


class A2CardTest(unittest.TestCase):
    def test_a2_card_readout_per_learner(self):
        """(a2): bit for bit on the processor; on the card bit for bit or within 1e-6 relative, per learner."""
        run = suite_dir()
        r = a2_readout(run)
        READOUTS["a2"] = r
        for L in LEARNERS:
            self.assertIsNotNone(r[L]["processor"], L)
            self.assertEqual(r[L]["processor"]["match"], "bit for bit", L)
            self.assertIsNotNone(r[L]["card"], L)
            self.assertTrue(r[L]["card"]["within_card_rule"], (L, r[L]["card"]))


class GateTest(unittest.TestCase):
    def test_real_data_is_refused_and_nothing_trains(self):
        """runner.py and stages.py (and band_readout, a scoring lane) refuse a real prepared file without the
        approval file: exit 2, REFUSED, no JAX and no Unifloral import, nothing written or changed."""
        run = suite_dir()
        work = (run if run else SUITE) / "gate"
        rec = gate_check(work)
        READOUTS["gate"] = rec
        if not rec["ran"]:
            self.skipTest(rec["why"])
        self.assertEqual(rec["real_prepared"]["schema"], "shift-hosts-prepared-v1")
        self.assertFalse(bool(rec["real_prepared"]["made_up"]))
        self.assertEqual(rec["real_prepared"]["quality"], "medium-expert")
        self.assertEqual(rec["real_prepared"]["npz_sha256"], rec["real_prepared"]["npz_sha256_after"])
        for c in rec["commands"]:
            self.assertEqual(c["exit"], 2, c)
            self.assertTrue(c["refused"], c)
            self.assertFalse(c["unifloral_imported"], c)
            self.assertFalse(c["blocked_import_hit"], c)
            if c["refused_before_any_jax_import"]:
                self.assertFalse(c["jax_imported"], c)
        self.assertEqual(rec["changed_files"], [])
        for name in ("stage1", "stage2", "driver"):
            self.assertFalse((REPO / "runs" / "shift_hosts" / name).exists(), name)


class TimingReadoutTest(unittest.TestCase):
    def test_timing_recorded_and_valid(self):
        """The card timing (gpu role) ran on this suite run: every program valid and finite; its readout against the
        plan's figures and budget is recorded (not a pass mark: S6 judges speeds in stage 1)."""
        run = suite_dir()
        r = timing_readout(run)
        READOUTS["timing"] = r
        if not r.get("recorded"):
            self.skipTest(f"no timing result in this suite run: {r.get('problems')}")
        body, _ = load_result(run, AREA, "test_all_stage0_gpu")
        self.assertTrue(body["passed"])
        self.assertEqual(r["problems"], [])


class HygieneTest(unittest.TestCase):
    def test_hygiene(self):
        """The WSL package equals the staging folder; no __pycache__; git shows only the package as new; the code
        manifest is the same before the suite, after it, and now (no code changed while the tests ran)."""
        h = hygiene(OPTIONS["staging"])
        run = suite_dir()
        now = gate.code_manifest()
        h["code_manifest"] = dict(before=hygiene_manifest(run, "before"), after=hygiene_manifest(run, "after"),
                                  now=[canonical_sha256(now), len(now)])
        READOUTS["hygiene"] = h
        self.assertEqual(h["pycache"], [])
        self.assertEqual(h["git_status"], ["?? experiments/shift_hosts/"])
        if h["staging_vs_wsl"]["checked"]:
            self.assertEqual(h["staging_vs_wsl"]["mismatches"], [])
        for tag in ("before", "after"):
            got = h["code_manifest"][tag]
            self.assertIsNotNone(got, f"no {tag} manifest recorded by run_stage0.sh")
            self.assertEqual(list(got), h["code_manifest"]["now"], tag)


class ManifestDraftTest(unittest.TestCase):
    def test_unifloral_files_equal_their_pins_and_the_package_is_listed(self):
        """The draft's four Unifloral files equal configs/unifloral.json (S1); every package file is listed."""
        d = manifest_draft(suite_dir())
        READOUTS["manifest"] = {k: d[k] for k in ("code_manifest_sha256", "files", "package_files",
                                                   "not_yet_in_the_package")}
        for rel, v in d["unifloral_four"].items():
            self.assertTrue(v["equal"], rel)
        # every regular file of the package but prespec.json and the root approval files (gate.package_files' rule,
        # restated here so the draft is checked against the folder itself)
        on_disk = set()
        for p in PACKAGE.rglob("*"):
            rel = p.relative_to(PACKAGE)
            if (p.is_file() and "__pycache__" not in rel.parts and p.suffix != ".pyc"
                    and rel.as_posix() != gate.PRESPEC_NAME and not gate.is_approval_file(rel)):
                on_disk.add(f"experiments/shift_hosts/{rel.as_posix()}")
        self.assertEqual(set(d["package"]), on_disk)
        self.assertEqual(on_disk, {f"experiments/shift_hosts/{p.relative_to(PACKAGE).as_posix()}"
                                   for p in gate.package_files(PACKAGE)})


# ------------------------------------------------------------------------------------------------ the card timing

CHUNK = 2_500
TIMED_CALLS = dict(alone=4, bca=4)
TIMING_SEED = 0
CARD_QUERY = ("pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used,"
              "clocks_throttle_reasons.active")


def card_state():
    """The card's state from nvidia-smi (read only, 20 s at most), between timed programs, never during one."""
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={CARD_QUERY}", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20)
        return dict(zip(CARD_QUERY.split(","), [v.strip() for v in out.stdout.strip().split(",")]))
    except (OSError, subprocess.TimeoutExpired) as error:
        return dict(error=repr(error))


def _program(step):
    import jax

    return jax.jit(lambda c: jax.lax.scan(step, c, None, length=CHUNK))


def _time(prog, carry, calls, check):
    """First call (compiling, not counted), then `calls` timed calls; JAX's compile time inside a timed call is
    subtracted (runner.compile_clock). check(outs) -> problem or None."""
    import jax
    import numpy as np

    from experiments.shift_hosts import runner as RN

    c0, t0 = RN.compile_clock(), time.perf_counter()
    carry, outs = prog(carry)
    jax.block_until_ready((carry, outs))
    first, first_compile = time.perf_counter() - t0, RN.compile_clock() - c0
    problems = [check(outs)]
    walls, comps = [], []
    for _ in range(calls):
        c0, t0 = RN.compile_clock(), time.perf_counter()
        carry, outs = prog(carry)
        jax.block_until_ready((carry, outs))
        walls.append(time.perf_counter() - t0)
        comps.append(RN.compile_clock() - c0)
        problems.append(check(outs))
    ms = [1000.0 * (w - k) / CHUNK for w, k in zip(walls, comps)]
    rec = dict(first_call_seconds=round(first, 3), first_call_compile_seconds=round(first_compile, 3),
               timed_wall_seconds=[round(w, 4) for w in walls], timed_compile_seconds=[round(k, 4) for k in comps],
               ms_per_outer_step=[round(x, 5) for x in ms], median_ms=round(statistics.median(ms), 5),
               min_ms=round(min(ms), 5), problems=[p for p in problems if p])
    return rec, carry


@unittest.skipUnless(T.device() == "gpu", "the card timing runs in the gpu role only")
class TimingTest(unittest.TestCase):
    """The gpu role (module docstring): one outer step per learner and setup on the card."""

    def test_one_outer_step_per_learner_and_setup(self):
        """Alone, standard and IW before and after one refresh, one refresh and one IW table, per learner."""
        import importlib

        import jax
        import numpy as np

        from experiments.shift_hosts import bca_core as BC
        from experiments.shift_hosts import guard
        from experiments.shift_hosts import iw as IW
        from experiments.shift_hosts import runner as RN
        from experiments.shift_hosts.tests import madeup

        rec = UI.record()
        m = madeup.build(TIMING_SEED, train_episodes=500, calibration_episodes=180)
        seeds = BC.seeds(TIMING_SEED)
        base = BC.fit_key_base(seeds.s91)
        masses = np.concatenate([m.weight["train"], m.weight["strong"], madeup.ratio_strong(np.array([0.1, 1.1]))])
        grid = IW.mass_grid(masses)
        out = dict(device=dict(backend=rec.get("backend"), devices=rec.get("devices"), card=rec.get("card"),
                               xla_flags=(rec.get("xla_flags") or {}), matmul_precision=rec.get("matmul_precision")),
                   data=dict(made_up=True, train_rows=m.n, bank_rows=int(len(m.bank.obs)),
                             strong_rows=int(len(m.strong.obs)), seed=TIMING_SEED,
                             about="tests/madeup.build(0, train_episodes=500, calibration_episodes=180)"),
                   chunk=CHUNK, timed_calls=TIMED_CALLS, grid_points=int(len(grid)), learners={})
        problems = []
        for L in LEARNERS:
            t_learner = time.perf_counter()
            mod = importlib.import_module(f"experiments.shift_hosts.unifloral_{L}_bca")
            args = UI.args_for(L, seed=seeds.unifloral, gamma=0.98, num_updates=N_OUTER)
            built = UI.init_like_training_block(L, args, m.rows("train"))
            bank = UI.make_transition(L, m.rows("bank"), next_action=m.next_action["bank"] if L == "rebrac" else "roll")
            s_alone = types.SimpleNamespace(bca=None, learner=L)
            s_bca = types.SimpleNamespace(bca=True, learner=L)

            def check(s):
                return lambda outs: (lambda p: None if p is None else f"step {p[0]}: {p[1]}")(
                    RN.chunk_problem(s, {k: np.asarray(v) for k, v in outs.items()}, 0))

            r = dict(train_rows=m.n, bank_rows=int(len(m.bank.obs)), grid_points=int(len(grid)),
                     card_before=card_state())
            r["alone"], _ = _time(_program(UI.make_unifloral_step(built, args)), (built.rng, built.agent_state),
                                  TIMED_CALLS["alone"], check(s_alone))
            bca = BC.BCASetup(L, base, guard.MADE_UP)
            scale = BC.make_scale_state(L, built.obs_mean, built.obs_std, BC.scale_init_key(base, L))
            pre_kw = dict(dose_active=False) if L == "cql" else {}
            prog_pre = _program(mod.make_train_step(args, *built.apply_fns, built.dataset, bca=bca, **pre_kw))
            r["standard_before_refresh"], c_std = _time(
                prog_pre, (built.rng, built.agent_state, BC.initial_state(scale, m.n, most_caution=False)),
                TIMED_CALLS["bca"], check(s_bca))
            r["iw_before_refresh"], c_iw = _time(
                prog_pre, (built.rng, built.agent_state, BC.initial_state(scale, m.n, most_caution=True)),
                TIMED_CALLS["bca"], check(s_bca))
            refresh_s = []
            t0 = time.perf_counter()
            r_std = mod.refresh(args, built.apply_fns, c_std[1], c_std[2], bank, BC.refresh_key(seeds.s39, 10_000),
                                permission=guard.MADE_UP)
            jax.block_until_ready(r_std.state)
            refresh_s.append(round(time.perf_counter() - t0, 3))
            t0 = time.perf_counter()
            r_iw = mod.refresh(args, built.apply_fns, c_iw[1], c_iw[2], bank, BC.refresh_key(seeds.s39, 10_000),
                               permission=guard.MADE_UP)
            jax.block_until_ready(r_iw.state)
            refresh_s.append(round(time.perf_counter() - t0, 3))
            t0 = time.perf_counter()
            iwr = IW.iw_refresh(r_iw, m.weight["bank"], m.weight["train"], grid, permission=guard.MADE_UP, bank=bank)
            jax.block_until_ready(iwr.state)
            r["iw_table_seconds"] = round(time.perf_counter() - t0, 3)
            r["refresh_seconds"] = refresh_s
            r["refresh"] = dict(certified=bool(r_std.record.get("certified")), threshold=r_std.record.get("threshold"),
                                iw_s3_reproduced=iwr.record.get("s3_reproduced"),
                                iw_refused_rows=iwr.record.get("refused_rows"))
            prog_post = _program(mod.make_train_step(args, *built.apply_fns, built.dataset, bca=bca))
            r["standard_after_refresh"], _ = _time(prog_post, (c_std[0], c_std[1], r_std.state), TIMED_CALLS["bca"],
                                                   check(s_bca))
            r["iw_after_refresh"], _ = _time(prog_post, (c_iw[0], c_iw[1], iwr.state), TIMED_CALLS["bca"],
                                             check(s_bca))
            r["seconds"] = round(time.perf_counter() - t_learner, 1)
            r["card_after"] = card_state()
            for k in ("alone", "standard_before_refresh", "iw_before_refresh", "standard_after_refresh",
                      "iw_after_refresh"):
                problems += [f"{L} {k}: {p}" for p in r[k]["problems"]]
            if not r["refresh"]["certified"]:
                problems.append(f"{L}: the made-up refresh did not certify")
            if r["refresh"]["iw_s3_reproduced"] is not True:
                problems.append(f"{L}: the IW refresh did not reproduce the threshold (S3)")
            out["learners"][L] = r
            print(f"timing {L}: alone {r['alone']['median_ms']:.3f} ms, BCA before refresh "
                  f"{r['standard_before_refresh']['median_ms']:.3f} / {r['iw_before_refresh']['median_ms']:.3f} ms, "
                  f"after {r['standard_after_refresh']['median_ms']:.3f} / {r['iw_after_refresh']['median_ms']:.3f} ms;"
                  f" refresh {refresh_s} s; IW table {r['iw_table_seconds']} s ({r['seconds']} s)", flush=True)
        out["problems"] = problems
        READOUTS["timing"] = out
        self.assertEqual(problems, [])


def _card_module():
    """A module holding only TimingTest, under this module's name (for tests.run on the card)."""
    mod = types.ModuleType(__spec__.name if __spec__ is not None else __name__)
    mod.TimingTest = TimingTest
    return mod


def _extra_gpu():
    return dict(timing=READOUTS.get("timing"), unifloral_import=UI.record())


def _extra_none():
    return dict(readouts=READOUTS)


# ------------------------------------------------------------------------------------------------ the summary

def write_summary(run, status):
    """runs/shift_hosts/stage0/SUMMARY.json and manifest_draft.json (module docstring)."""
    jobs = load_jobs(run)
    mods = READOUTS.get("modules") or []
    own_path = STAGE0 / AREA / "test_all_stage0_none.json"
    own = json.loads(own_path.read_text(encoding="utf8")) if own_path.is_file() else None
    rows = READOUTS.get("plan") or []
    plan_view = {}
    for r in rows:
        key = r["test"] if r["learner"] is None else f"{r['test']}/{r['learner']}"
        if key in plan_view:
            key = f"{key}/{len(plan_view)}"
        plan_view[key] = dict(passed=r["passed"], proves=r["proves"],
                              cases=[dict(case=f"{c['area']}/{c['module']}::{c['case']}", outcome=c["outcome"])
                                     for c in r["cases"]])
    card_s = sum(j["seconds"] for j in jobs.values() if j["device"] == "gpu")
    wall_s = sum(j["seconds"] for j in jobs.values())
    proc_s = sum((m.get("processor_seconds") or 0.0) for m in mods)
    gate_rec = READOUTS.get("gate") or {}
    s0_problems = list(READOUTS.get("plan_problems") or [])
    s0_problems += [f"module {m['module']}: exit {m['exit']}, passed {m['passed']}, {m['problems']}"
                    for m in mods if m["exit"] != 0 or not m["passed"] or m["problems"]]
    summary = dict(
        schema="shift-hosts-stage0-summary-v1", made_utc=utc_now(), suite_run=None if run is None else run.name,
        suite_folder=None if run is None else str(run), made_up_arrays_only=True,
        s0=dict(rule="S0: any of tests (a1)-(i) or the guard's comparison fails (plan 6.1), with (a2)'s card "
                     "exception", passed=bool(rows) and not s0_problems and all(r["passed"] for r in rows),
                problems=s0_problems, not_needed=NOT_NEEDED),
        plan_tests=plan_view, a2_card=READOUTS.get("a2"), modules=mods,
        gate=dict(passed=None if not gate_rec else (gate_rec.get("ran") and all(
            c["exit"] == 2 and c["refused"] and not c["unifloral_imported"] and not c["blocked_import_hit"]
            and (not c["refused_before_any_jax_import"] or not c["jax_imported"])
            for c in gate_rec.get("commands", [])) and not gate_rec.get("changed_files")), record=gate_rec),
        timing=READOUTS.get("timing"), hygiene=READOUTS.get("hygiene"), manifest=READOUTS.get("manifest"),
        s9_signature_draft=s9_draft(run),
        compute=dict(suite_wall_seconds_summed_over_jobs=round(wall_s, 1),
                     card_job_seconds=round(card_s, 1), card_hours=round(card_s / 3600.0, 3),
                     processor_seconds_from_results=round(proc_s, 1),
                     processor_hours=round(proc_s / 3600.0, 3),
                     note="processor seconds are each module's own count (self and waited-for children); card hours "
                          "are the summed wall time of the card jobs, which ran one at a time"),
        this_module=None if own is None else dict(passed=own.get("passed"), tests_run=own.get("tests_run"),
                                                  skipped=own.get("skipped"), failures=own.get("failures"),
                                                  errors=own.get("errors"),
                                                  cases={c["name"].split(".", 1)[-1]: c["outcome"]
                                                         for c in own.get("cases", [])}),
        status=status)
    H.write_json(STAGE0 / "SUMMARY.json", summary)
    H.write_json(STAGE0 / "manifest_draft.json", manifest_draft(run))
    if run is not None:
        for name in ("SUMMARY.json", "manifest_draft.json"):
            for suffix in ("", ".sha256"):
                shutil.copy2(STAGE0 / (name + suffix), run / (name + suffix))
    print(f"summary: S0 {'passed' if summary['s0']['passed'] else 'NOT passed'} -> {STAGE0 / 'SUMMARY.json'}")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--suite-dir")
    p.add_argument("--staging")
    a = p.parse_args()
    OPTIONS.update(suite_dir=a.suite_dir, staging=a.staging)
    if T.device() == "gpu":
        UI.prepare(device="gpu")
        sys.exit(T.run(_card_module(), AREA, extra=_extra_gpu))
    module = types.ModuleType(__spec__.name if __spec__ is not None else __name__)
    for name in ("PlanTest", "A2CardTest", "GateTest", "TimingReadoutTest", "HygieneTest", "ManifestDraftTest"):
        setattr(module, name, globals()[name])
    status = T.run(module, AREA, device_name="none", extra=_extra_none)
    write_summary(suite_dir(), status)
    sys.exit(status)
