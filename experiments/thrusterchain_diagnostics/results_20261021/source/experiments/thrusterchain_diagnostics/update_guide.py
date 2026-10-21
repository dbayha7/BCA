"""Add the diagnostics contract without rebuilding the standalone historical guide.

Run from the repository root; --source can override the preserved reference copy.
The original file is read only. All original script blocks and embedded assets are
preserved verbatim; assertions reject a different source instead of guessing.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from hashlib import sha256
from html.parser import HTMLParser
from pathlib import Path
import re
import sys


SECTION = r'''
  <style>
    #diag-oracle-panel { max-width: 100%; }
    .diag-selectors { display: flex; flex-wrap: wrap; gap: 12px; }
    .diag-selectors label { display: flex; flex-direction: column; gap: 3px; font-weight: 600; font-size: 15px; }
    .diag-selectors select { font: inherit; color: var(--ink); background: var(--surface); border: 1px solid var(--rule-strong); border-radius: 5px; padding: 7px; min-height: 40px; }
    #diag-oracle-rows .diag-best { background: var(--ok-soft); }
  </style>
  <section id="diagnostics-suite" aria-labelledby="diagnostics-heading">
    <h2 id="diagnostics-heading">ThrusterChain diagnostics suite</h2>
    <p class="status"><span class="tag acc">implementation and smoke checks</span> Added 20 October 2026.
      This suite makes model assumptions, support and uncertainty targets inspectable. It does not report a newly
      trained method, a confirmatory comparison, or a demonstrated improvement in BCA.</p>
    <p>The question is where an offline decision fails: the data may not identify an action's consequences,
      a learned value may be wrong, an uncertainty interval may cover a different target, or a useful signal may
      act through an ineffective intervention. The suite separates these questions with an observed-regime
      continuous task, a finite model with enumerated expectations, and a deliberately hidden paired-world test.</p>
    <p class="note">The <a href="#learn">lesson</a>, <a href="#results">main-test results</a>,
      <a href="#weighting-test">weighting test</a>, <a href="#toy-worlds">toy studies</a>, saved actors and
      interactive panels below are the historical study. Their 13-number observation, continuing discounted
      critic and reported scores remain historical specifications. The new suite is an additive experiment
      namespace: <code>experiments/thrusterchain_diagnostics/</code>.</p>

    <h3 id="diag-models">Three models, with different claims</h3>
    <div class="tbl"><table>
      <caption>Choose the model before interpreting a diagnostic.</caption>
      <thead><tr><th scope="col">Model</th><th scope="col">Decision and objective</th><th scope="col">What it can establish</th></tr></thead>
      <tbody>
        <tr><th scope="row">Observed continuous ThrusterChain</th><td>The existing 13 physical coordinates,
          plus the observed regime and time remaining. Four bounded continuous throttles; a regime is fixed
          within each episode. Primary objective: 100 steps, discount 1.</td><td>Deterministic rollouts for
          specified starts and controllers; constrained finite-horizon optimal-control references when the
          solver certifies the applicable quadratic program. It retains the physical control problem.</td></tr>
        <tr><th scope="row">Finite diagnostic model</th><td>36 states: 2 regimes × 2 load levels × 9 position
          bins. Five discrete actions <code>[-1, -float32(0.8), 0, knee, 1]</code>; the positive knee uses the
          implemented float32 boundary. Exported observations include position, regime, load and remaining-time
          fraction. Time-indexed decisions over 8 steps, discount 0.98.</td><td>Enumerated
          transition and reward laws support dynamic programming, policy values, optimal values and occupancy
          measures, up to floating-point arithmetic. These are results for this specified finite model.</td></tr>
        <tr><th scope="row">Hidden paired worlds</th><td>A stall world and a monotone world share the same
          observed start distribution and behavior actions in their common region. The regime identity is
          hidden from learner data and metadata.</td><td>When the full learner-visible logging law is identical,
          behavior logs cannot identify which continuation applies beyond support. Separate probes can supply
          new information; ordinary behavior logs cannot.</td></tr>
      </tbody>
    </table></div>
    <p>The monotone regime makes extrapolation beyond the knee potentially useful, while the stall regime makes
      it potentially harmful. The point of observing the regime in the main track is to let a method distinguish
      those cases. The hidden track asks a different question: which distinctions are impossible from the specified
      logs alone? A finite discretization is a separate diagnostic model, not a proof about the continuous task.</p>
    <p><b>The finite action grid has its own limitation.</b> It omits intermediate commands such as the roughly
      half-throttle command that matches a fully stalled continuous action's thrust. Continuous stall dominance
      therefore does not hold universally on this five-action grid. The declared contrast is specific: at time 0,
      position 1.5 and high load, action −1 is worse than the negative knee in the stall world and better in the
      monotone world. Near other states, a high action can be preferable even in the finite stall model.</p>

    <h3 id="diag-explorer">Inspect the stored finite oracle</h3>
    <div class="box" id="diag-oracle-panel">
      <p>Choose a regime, load, position, time and retention condition. The five values below are stored
        <b>finite-model Q*</b>: take that action once, then follow an optimal policy until step 8, discount 0.98.
        They are not learned predictions or continuous ThrusterChain values.</p>
      <div class="diag-selectors">
        <label>Regime <select id="diag-regime"><option value="0">Stall</option><option value="1">Monotone</option></select></label>
        <label>Load <select id="diag-load"><option value="0">Low · 0.1</option><option value="1" selected>High · 0.35</option></select></label>
        <label>Position <select id="diag-position"></select></label>
        <label>Time <select id="diag-time"></select></label>
        <label>Retention <select id="diag-retention"><option value="full">Full</option><option value="rare">Rare · 5%</option><option value="zero" selected>Zero</option></select></label>
      </div>
      <p id="diag-oracle-context" class="note" aria-live="polite">Enable JavaScript to inspect the embedded finite-oracle values.</p>
      <div class="tbl"><table>
        <thead><tr><th scope="col">Action</th><th scope="col">Stored Q*</th><th scope="col">Critical region</th>
          <th scope="col">Retention probability</th><th scope="col">Trajectory source support</th>
          <th scope="col">Retained trajectory rows</th></tr></thead>
        <tbody id="diag-oracle-rows"></tbody>
      </table></div>
      <p id="diag-oracle-witness" aria-live="polite"></p>
      <p class="note">Critical means position ≥ 0.5, high load and action −1, fixed before outcomes. Source support
        means positive time-state-action mass under the smoke logging law after retention; it differs from the
        realized count in 64 trajectories. A zero count can occur despite positive support. The fixed-state audit
        supplies retention propensities for every query. Changing retention does not change the underlying Q*.</p>
      <p class="note">Source: <code>artifacts/smoke_verified_20261020/evaluator/finite_oracle.npz</code> and the matching
        full/rare/zero fixed-state and trajectory audit files. Arrays are embedded in this HTML for offline use.</p>
    </div>

    <h3 id="diag-oracles">What “exact” means here</h3>
    <div class="tbl"><table>
      <thead><tr><th scope="col">Quantity</th><th scope="col">Contract and limitation</th></tr></thead>
      <tbody>
        <tr><th scope="row">Finite-model value</th><td>Backward recursion computes
          <code>Q[t,s,a] = r[s,a] + gamma × sum(P[s,a,s′] × V[t+1,s′])</code>, with
          <code>V[H] = 0</code>. A specified policy averages over its actions; the optimal recursion maximizes.
          All relevant states, actions and next states are enumerated. “Exact” refers to this model and objective,
          with floating-point numerical error reported by checks.</td></tr>
        <tr><th scope="row">Continuous policy return</th><td>A known deterministic model and a fixed start,
          controller, horizon and discount determine a rollout. An average over a fixed start bank is that bank's
          average. It is not the exact expectation under a continuous population of starts. An arbitrary
          continuing-policy rollout also has truncation and numerical error.</td></tr>
        <tr><th scope="row">Continuous optimum</th><td>A valid reduction and a numerical optimality certificate
          can certify the constrained quadratic program for a stated start, horizon and discount. In the stall
          regime, a command beyond the knee is dominated by a command giving the same thrust with less throttle
          cost; the relevant optimum can therefore be sought on the nonstalling branch. Bounds and solver residuals
          must accompany the result. A solver's success flag alone is not a certificate.</td></tr>
        <tr><th scope="row">Historical references</th><td>The historical continuous QP references remain useful
          numerically certified optimal-control benchmarks for their stated starts. Their finite-bank averages
          are not population expectations. The old continuing discount-0.98 critic target and the old
          undiscounted 100-step score are different objectives; their difference does not apply to the new
          primary finite-horizon contract.</td></tr>
      </tbody>
    </table></div>
    <p>Always report the model, observed state, policy, start distribution or bank, horizon, discount, terminal
      convention and numerical tolerance with an oracle result. A small Bellman residual, a small rollout error
      and a small optimality gap answer different questions.</p>

    <h3 id="diag-data">Data, retention and the paired-world check</h3>
    <p>Full, rare and zero retention describe how much of the designated action region remains in the training
      data. They are separate conditions. Removing rows after a trajectory has been generated changes which
      transitions are observed; it does not undo the effect of removed actions on later states. Keep the original
      trajectory identity and time so that retention and dependence remain auditable.</p>
    <ul>
      <li><b>Trajectory data</b> preserve episode paths and expose changes in the states a policy visits.
        Evaluation uses the stated start law and policy; a retained-row distribution is not automatically a
        policy occupancy measure.</li>
      <li><b>Fixed-state queries</b> hold the state distribution fixed while changing action support or a prediction
        target. They isolate action uncertainty, but do not reproduce the full trajectory problem.</li>
      <li><b>Hidden-world logs</b> use a common behavior action sequence and matched randomness at every step,
        inside the region where both transition laws agree. Equal one-step transitions or equal action marginals
        alone are insufficient: the full learner-visible paths, rewards, termination fields and metadata must
        agree. World identity remains only in evaluator-side information. Hidden finite observations contain
        position, load and remaining-time fraction, with no regime coordinate.</li>
      <li><b>Rare probes</b> are held separately from the identical common-region logs. A probe beyond the shared
        action region may distinguish the worlds, so it changes the information available to the learner.
        Report its budget and split explicitly.</li>
    </ul>
    <p>Full/rare/zero retention and hidden/observed regime are different axes. A zero-support experiment measures
      nonidentification as well as estimation; it cannot be repaired by pretending that an undefined ratio is
      merely large.</p>
    <p>The finite-horizon boundary ends continuation even though the exported environment convention marks it
      as a time-limit truncation rather than an absorbing physical termination: the final transition has
      <code>bootstrap_mask = 0</code>. Remaining-time observations and this explicit mask keep the finite-horizon
      learning target aligned with the oracle.</p>

    <h3 id="diag-ratios">Ratios and support have a named target</h3>
    <p>A context ratio compares two distributions over a stated context, such as a reset condition or a fixed-state
      query. An occupancy ratio compares the states or state-action pairs visited by specified policies over time.
      Both numerator and denominator must use the same variables, time convention and normalization. Discounted
      occupancy includes the declared discount; a context ratio generally does not correct policy-induced state shift.</p>
    <div class="tbl"><table>
      <thead><tr><th scope="col">Case</th><th scope="col">Required interpretation</th></tr></thead>
      <tbody>
        <tr><th scope="row">Target mass with positive reference mass</th><td>The ratio is defined. Report its
          target, effective sample size and concentration, not just a mean weight.</td></tr>
        <tr><th scope="row">Positive target mass, zero reference mass</th><td>Unsupported target mass is explicit.
          It is not covered by importance weighting. Do not add an epsilon and claim identification.</td></tr>
        <tr><th scope="row">Both masses zero</th><td>The cell contributes no target mass. Any storage convention
          must remain distinguishable from a supported ratio of zero.</td></tr>
        <tr><th scope="row">Deterministic continuous action versus behavior density</th><td>A point-mass policy
          and a continuous action density do not admit an ordinary pointwise density ratio. A discrete diagnostic
          model, a specified stochastic policy or an explicitly defined smoothed estimand is required; each changes
          the claim.</td></tr>
      </tbody>
    </table></div>

    <h3 id="diag-targets">What the diagnostics measure</h3>
    <p>A one-step residual is a training target discrepancy. A policy-value error is
      <code>Q_hat[t,s,a] − Q_pi[t,s,a]</code>. Coverage of residuals does not establish coverage of policy-value
      error: propagation through the transition law, the evaluation policy and the horizon matters. Neither
      coverage statement alone establishes better decisions or returns.</p>
    <ul>
      <li><b>Prediction:</b> evaluate signed value error and uncertainty width against a named finite-model
        oracle or a continuous rollout with its stated approximation error.</li>
      <li><b>Coverage:</b> distinguish one-step residual coverage from policy-value coverage; condition summaries
        on regime, support and time where data permit. Marginal coverage can conceal failure in a small but
        consequential action region.</li>
      <li><b>Decision:</b> compare the selected action's true advantage or regret with its estimated advantage.
        A band can be calibrated and still fail to rank actions usefully.</li>
      <li><b>Return:</b> evaluate the resulting policy on the matched finite-horizon objective. Attribute a change
        only after checking whether the uncertainty signal, placement or intervention itself caused it.</li>
    </ul>
    <p class="note">The implementation's smoke examples use specified policies and diagnostic arrays. They test
      arithmetic and interfaces; they are not trained BCA predictions or a claim that BCA attains coverage.</p>

    <h3 id="diag-controls">Controls that distinguish signal from intervention</h3>
    <p>Keep both harmful and beneficial extrapolation in a comparison. A fixed cap can prevent the stall while
      discarding useful monotone-regime actions. A regime-aware cap uses the observed regime and supplies a
      stronger informed control. In the hidden track it is evaluator privileged and must not be presented as
      a feasible learner baseline.</p>
    <div class="tbl"><table>
      <thead><tr><th scope="col">Control</th><th scope="col">Question it answers</th></tr></thead>
      <tbody>
        <tr><th scope="row">No added intervention</th><td>What does the same policy or learning setup do without
          the extra mechanism?</td></tr>
        <tr><th scope="row">Fixed cap</th><td>Does simply excluding beyond-knee actions explain the gain, and
          what does it cost when extrapolation is beneficial?</td></tr>
        <tr><th scope="row">Regime-aware cap</th><td>How much is explained by knowing which regime should be
          restricted? This control reads an observation in the observed track and privileged information in the
          hidden track.</td></tr>
        <tr><th scope="row">Uniform or shuffled placement, with matched strength</th><td>Does where the
          intervention acts matter, beyond its total strength? Matching must name a strength measure and preserve
          the same host and evaluation objective.</td></tr>
        <tr><th scope="row">Oracle-error placement</th><td>What happens if estimation error in the placement
          signal is removed while the chosen dose mapping and intervention are held fixed? This is an informed
          diagnostic, not the return-optimal intervention.</td></tr>
      </tbody>
    </table></div>
    <p><b>Historical “perfect aim” has a limited meaning.</b> The old harm-placed test supplied oracle
      over-preference to one chosen capped dose map. Its failure justified that experiment's pre-registered stop.
      It does not prove that every possible dose map, placement, optimizer, policy class or use of uncertainty
      must fail. Error magnitude is not itself the causal value of intervening. A failure of an oracle-error
      placement is therefore neither an impossibility theorem nor a certificate that the best intervention
      has been tested.</p>

    <h3 id="diag-usage">Run the implementation</h3>
    <!-- DIAGNOSTICS_USAGE_BEGIN -->
    <p>From the BCA repository root, with NumPy and SciPy available, run the smoke command into a new directory:</p>
    <pre class="code">python -B -m experiments.thrusterchain_diagnostics smoke --output experiments/thrusterchain_diagnostics/artifacts/your_new_run</pre>
    <p>To select the delivered configuration explicitly, append
      <code>--config experiments/thrusterchain_diagnostics/configs/smoke.json</code>. Existing output directories
      are refused. Run the focused checks with:</p>
    <pre class="code">python -B -m unittest discover -s experiments/thrusterchain_diagnostics/tests -v</pre>
    <p>The smoke configuration fixes data seed 20261020 and independent holdout seed 20261021; optimization seed
      20261022 is reserved, with zero optimizer updates, zero online selection steps and zero budget top-ups.
      Train and holdout randomness use distinct named streams.</p>
    <div class="tbl"><table>
      <thead><tr><th scope="col">Saved output</th><th scope="col">Purpose</th></tr></thead>
      <tbody>
        <tr><th scope="row"><code>report.json</code>, <code>config.json</code></th><td>Machine-readable smoke diagnostics,
          objective contracts, runtime provenance and budgets.</td></tr>
        <tr><th scope="row"><code>learner/*.npz</code> and JSON</th><td>Retained learner rows and provenance. Oracle
          arrays and complete unretained rows belong to the evaluator output.</td></tr>
        <tr><th scope="row"><code>evaluator/*_audit.npz</code></th><td>Complete logs, critical masks and retention
          indicators for auditing what the learner received.</td></tr>
        <tr><th scope="row"><code>evaluator/finite_holdout.npz</code>, <code>finite_oracle.npz</code></th><td>Independent
          holdout rows and enumerated finite-model kernels, Q*/V* and occupancies.</td></tr>
        <tr><th scope="row"><code>evaluator/continuous_qp_stall.npz</code>, <code>continuous_qp_monotone.npz</code></th><td>
          Continuous control sequences and numerical oracle checks, with separate hidden-world probe outputs.</td></tr>
      </tbody>
    </table></div>
    <!-- DIAGNOSTICS_USAGE_END -->
    <p>The implementation is additive. Import its modules or run its command-line entry point from the repository
      root; the historical environment and the lesson's browser-side simulator retain their existing behavior.
      The guide itself opens as one standalone HTML file and requires no running Python process.</p>

    <h3 id="diag-evidence">Evidence and the next scientific claim</h3>
    <!-- DIAGNOSTICS_EVIDENCE_BEGIN -->
    <p>The delivered run <code>artifacts/smoke_verified_20261020/report.json</code> has status
      <code>smoke_only_no_learned_methods</code>. Its continuous witness starts at the specified deterministic
      high-load state with current 1.5, using 100 steps and discount 1. These are numerical implementation checks:</p>
    <div class="tbl"><table>
      <thead><tr><th scope="col">Witness</th><th scope="col">Recorded result</th><th scope="col">Interpretation</th></tr></thead>
      <tbody>
        <tr><th scope="row">Continuous stall optimum</th><td>Return −596.322853; Frank–Wolfe gap
          3.01 × 10<sup>−12</sup>; replay error 5.68 × 10<sup>−13</sup>.</td><td>A numerically certified
          reference for this start and objective.</td></tr>
        <tr><th scope="row">Continuous monotone optimum</th><td>Return −349.968029; gap
          4.01 × 10<sup>−12</sup>; replay error 9.09 × 10<sup>−13</sup>.</td><td>The monotone regime permits
          useful above-knee controls.</td></tr>
        <tr><th scope="row">Monotone cap controls</th><td>Fixed cap: −762.806836; regime-aware cap: −432.070051.</td>
          <td>Removing a needless cap helps this fixed controller at this start. Neither control is the optimal
            QP policy.</td></tr>
        <tr><th scope="row">Finite high-action contrast</th><td>At time 0, position 1.5 and high load,
          Q*(−1) − Q*(−knee) is −1.274147 in stall and +0.593793 in monotone.</td><td>A specific harmful/useful
          extrapolation witness in the finite model.</td></tr>
        <tr><th scope="row">Retained critical trajectory rows</th><td>Full / rare / zero: finite 19 / 1 / 0;
          continuous 438 / 20 / 0.</td><td>Realized smoke counts, not population probabilities.</td></tr>
        <tr><th scope="row">Hidden paired worlds</th><td>Common-support kernel, reward, observation and initial-law
          certificates pass; complete learner logs and metadata match.</td><td>The behavior law carries no world
          information under this construction. Separate rare probes are allowed to reveal differences.</td></tr>
      </tbody>
    </table></div>
    <p id="diag-test-evidence" class="note">The focused test run passed 50 tests with zero skips, including
      comparison to the original physical dynamics. Evidence: <code>artifacts/verification/tests.txt</code>.
      The guide preserves all 27 original script/data blocks. Node syntax checks passed for all 18 executable
      JavaScript blocks, and a DOM test checked all 864 new-selector combinations (4,320 action rows).
      No browser surface was available, so visual rendering and live navigation were not verified.</p>
    <!-- DIAGNOSTICS_EVIDENCE_END -->
    <p>Before claiming a method improvement, freeze the comparison: objective, start laws, support and retention
      conditions, learner-visible inputs, oracle target, controls, seeds, compute budget and decision rule. Train
      the methods under that protocol and report harmful and beneficial extrapolation separately. The present
      release establishes inspectable diagnostic machinery, not that empirical claim.</p>

    <h3 id="diag-positioning">Related work and the attribution hypothesis</h3>
    <p>Controlled diagnostic benchmarks already exist. <a href="https://proceedings.mlr.press/v97/fu19a.html">Fu
      et al., Diagnosing Bottlenecks in Deep Q-learning (2019)</a> use controlled experiments to isolate sources
      of error. <a href="https://arxiv.org/abs/1908.03568">bsuite</a> organizes small tasks around behavioral
      capabilities, and <a href="https://arxiv.org/abs/2006.13888">RL Unplugged</a> supplies offline-RL datasets
      and evaluation protocols. <a href="https://proceedings.mlr.press/v199/schweighofer22a.html">Schweighofer
      et al. (2022)</a> study how offline dataset characteristics affect performance.</p>
    <p>The proposed contribution is a diagnostic attribution hypothesis: within a matched task family, separate
      lack of information, value-estimation error, uncertainty calibration and the effect of the intervention,
      while retaining both useful and harmful extrapolation. An observed track, a finite oracle model and an
      indistinguishable-log control make that hypothesis testable. Their combination is not, by itself, evidence
      of novelty or demonstrated scientific value.</p>
  </section>
'''


REPLACEMENTS = [
    ("ThrusterChain is a small simulated task, built so that every true value in it can be computed\n      exactly:",
     "ThrusterChain is a small simulated control task with deterministic dynamics and inspectable value references:"),
    ("The physics has no randomness, so a value can be computed exactly by simulation; after 1,000 steps what is left is at most 8.4 × 10⁻⁸ times the largest single-step cost.",
     "For a fixed start and controller, deterministic simulation computes a truncated discounted value. After 1,000 steps the remaining tail is at most 8.4 × 10⁻⁸ times a valid bound on the future single-step cost; floating-point error is separate."),
    ("and the true value of a command is that sum computed exactly, by simulation, with a\n      fixed controller driving afterwards (<a href=\"#episode-end\">Episode End</a> explains how). Because every true value\n      can be computed, every error the critic makes can be measured exactly.",
     "and a command's reference value is estimated by deterministic simulation with a\n      fixed controller driving afterwards (<a href=\"#episode-end\">Episode End</a> explains how). This permits\n      direct checks of critic error at specified queries, subject to the simulation's truncation and numerical error."),
    ("Every true value on this page is that same sum, simulated far\n          enough that the rest no longer matters.",
     "The historical continuing-policy reference values in this part use that sum, simulated with a\n          finite truncation whose error is checked. The new diagnostics suite has a separate finite-horizon contract."),
    ("stall, has no randomness, and every true value in it is computed by exact simulation.",
     "stall and has no process randomness. Specified policy values can be checked by deterministic rollouts,\n      with a stated horizon, discount and numerical or truncation error; this does not make every population expectation exact."),
    ("The <a href=\"#rw-return\">true value</a> is the sum of all the rewards that follow, later ones\n          counting a little less, computed exactly.",
     "The <a href=\"#rw-return\">reference value</a> is the discounted sum of rewards for a stated continuation\n          controller, evaluated by deterministic rollout with a checked truncation."),
    ("First, in a world with linear dynamics and a\n        quadratic cost every true value is itself a quadratic formula of the state and the command, so data from\n        any one region of states and commands pin down the true values everywhere, and missing data hide nothing.",
     "First, under a fixed linear controller with unclipped linear dynamics and quadratic cost, a value can\n        have a quadratic form. The design treated this as a reason to introduce the stall. Identifying that form\n        from data still requires suitable model assumptions and excitation; local data alone do not identify\n        arbitrary policies or all out-of-support values."),
    ("The same quantity, computed exactly. The physics has no randomness, so the simulator gives command",
     "The same quantity, checked by rollout. For the specified state and actor, the simulator gives command"),
    ("In this world every true value can be\n      computed exactly, so the critic's guesses can be checked.",
     "For specified states, actions and continuation policies, deterministic rollouts provide value references\n      with checked truncation error, so the critic's guesses can be checked."),
    ("asks whether perfect aim could help at all. If it could not, no BCA result could show a gain.",
     "asks whether an oracle-error placement helps under the chosen dose map. Failure stops this protocol;\n      it does not bound every possible learned placement or intervention."),
    ("placed perfectly and without BCA: if even that cannot help, no result of BCA could show that it helps.",
     "placed using oracle error and without BCA. This is the protocol's stop check for one intervention,\n          not a proof that every BCA intervention must fail."),
    ("That caution is placed by the exact error, which BCA can only estimate, so BCA is not expected to do\n              better. If even perfect placement cannot beat the same caution placed without that knowledge by 3 points, no BCA result\n              could show that BCA's placement helps, and running BCA would spend the test on a question it cannot\n              answer.",
     "The test uses oracle error in its chosen dose map and requires a 3-point advantage over the controls\n              before proceeding. Failing that rule stops this pre-registered experiment. Oracle error knowledge\n              does not make the dose map return-optimal or rule out every other use of BCA."),
    ("What it means for BCA.</b> By the test's own logic, no BCA result could have shown a gain here.",
     "What it means for BCA.</b> The pre-registered protocol stopped after this oracle-error placement failed\n        its pass mark. This establishes a failure of the tested placement and intervention, not a general impossibility."),
    ("The fair order is to ask first whether perfect aim\n      could help at all. If it cannot, no BCA result could show a gain.",
     "The protocol first asks whether oracle-error placement under this dose map\n      meets its pass mark. Failure justifies its planned stop, without proving that no alternative could help."),
    ("No signal learned from data can know the critic's error better.",
     "This removes estimation error in the placement signal; it does not establish the return-optimal dose or placement."),
    ("No signal learned from data can know the\n      critic's error better.",
     "This removes estimation error in the signal, but does not establish the\n      return-optimal intervention."),
    ("Why it never ran, and why it could not have helped here.</b> By the test's own logic no BCA result could\n      have shown a gain: if perfect aim in this lever cannot win the return back, a learned signal cannot rescue it.",
     "Why it never ran, and what the stop establishes.</b> Oracle-error placement failed this experiment's\n      required gain, so its pre-registered rule stopped the BCA run. That failure concerns this dose map and lever;\n      it does not prove that every learned signal or alternative intervention must fail."),
    ("It is perfect aim inside BCA's range.",
     "It supplies oracle error to the chosen dose map inside BCA's range; it is not an optimal-intervention certificate."),
    ("These optimum values are exploratory: they were computed exactly on 15 October 2026 (UTC), after the main",
     "These optimum references are exploratory: they were computed by numerically checked quadratic programs\n      on 15 October 2026 (UTC), after the main"),
    ("(a reference computed exactly for every start; nothing learned; 15 Oct at 02:09)",
     "(a numerically certified reference for each stated start; nothing learned; 15 Oct at 02:09)"),
    ("That does not change the logic. If the pull cannot win\n      the return back when it knows exactly where the critic is wrong, a learned signal cannot rescue it, at least not\n      within BCA's dose range.",
     "That comparison explains the protocol's stop. Oracle error knowledge did not make this dose map\n      return-optimal, so its failure does not rule out another placement, optimization procedure or intervention\n      within the same dose range."),
    ("states the first two as one cause in two halves: BCA as built cannot improve its host because of its lever,\n    and on ThrusterChain also because of its signal.",
     "interprets the first two as a limitation of this lever and, on ThrusterChain, its signal. This remains\n    a historical attribution hypothesis, not an impossibility theorem for every host or intervention."),
]


ORACLE_JS = r'''
<script id="diagnostics-oracle-interaction">
(function () {
  'use strict';
  const data = JSON.parse(document.getElementById('diagnostics-oracle-data').textContent);
  const el = id => document.getElementById(id);
  const fields = ['regime', 'load', 'position', 'time', 'retention'];
  function option(select, value, label, selected) {
    const o = document.createElement('option'); o.value = String(value); o.textContent = label;
    o.selected = !!selected; select.appendChild(o);
  }
  data.positions.forEach((x, i) => option(el('diag-position'), i, String(x), x === 1.5));
  for (let t = 0; t < data.horizon; t++) option(el('diag-time'), t, String(t) + ' · ' + (data.horizon - t) + ' steps left', t === 0);
  const signed = x => (x > 0 ? '+' : '') + x.toFixed(6);
  function render() {
    const regime = Number(el('diag-regime').value), load = Number(el('diag-load').value);
    const pos = Number(el('diag-position').value), t = Number(el('diag-time').value);
    const retention = el('diag-retention').value;
    const s = data.states.findIndex(x => x[0] === regime && x[1] === load && x[2] === data.positions[pos]);
    const q = data.q[t][s], best = Math.max.apply(null, q), tbody = el('diag-oracle-rows');
    tbody.replaceChildren();
    q.forEach((value, a) => {
      const tr = document.createElement('tr');
      const propensity = data.retention[retention][t][s][a];
      const loggingMass = data.logging_occupancy[t][s][a];
      const support = loggingMass <= 0 ? 'No · unvisited by logging law' : propensity <= 0 ? 'No · removed' : 'Yes';
      const label = a === 1 ? '−knee' : a === 3 ? '+knee' : String(data.actions[a]);
      const cells = [label, value.toFixed(6) + (value === best ? ' · best' : ''), data.critical[t][s][a] ? 'Yes' : 'No',
        (propensity * 100).toFixed(propensity === 0 || propensity === 1 ? 0 : 1) + '%', support,
        String(data.counts[retention][t][s][a])];
      cells.forEach((text, i) => { const cell = document.createElement(i === 0 ? 'th' : 'td');
        if (i === 0) { cell.scope = 'row'; cell.title = String(data.actions[a]); }
        cell.textContent = text; tr.appendChild(cell); });
      if (value === best) tr.className = 'diag-best';
      tbody.appendChild(tr);
    });
    el('diag-oracle-context').textContent = (regime ? 'Monotone' : 'Stall') + ' · load ' + data.loads[load] +
      ' · position ' + data.positions[pos] + ' · t=' + t + ' · ' + (data.horizon - t) + ' steps remaining · ' + retention + ' retention.';
    el('diag-oracle-witness').textContent = 'Q*(−1) − Q*(−knee) = ' + signed(q[0] - q[1]) +
      '. Positive means the high negative action is better for this state and optimal continuation; negative means it is worse.';
  }
  fields.forEach(name => el('diag-' + name).addEventListener('change', render));
  render();
})();
</script>
'''


def oracle_payload(artifacts):
    """Read saved oracle values and audit rows; never rerun or fit a learner."""
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.thrusterchain_diagnostics.finite import FiniteConfig, FiniteMDP
    report = json.loads((artifacts / 'report.json').read_text(encoding='utf-8'))
    config = report['finite']['contract']['config']
    mdp = FiniteMDP(FiniteConfig(**config))
    assert (mdp.horizon, mdp.n_states, mdp.n_actions, mdp.gamma) == (8, 36, 5, .98)
    sources = [artifacts / 'report.json', artifacts / 'evaluator' / 'finite_oracle.npz']
    with np.load(sources[-1], allow_pickle=False) as z:
        assert np.array_equal(z['P'], mdp.P) and np.array_equal(z['R'], mdp.R), 'Artifact/model mismatch'
        q = z['optimal_Q'].copy()
        occupancy = z['logging_occupancy'].copy()
    shape = q.shape
    retention, counts = {}, {}
    critical = None
    for mode in ('full', 'rare', 'zero'):
        fixed = artifacts / 'evaluator' / f'finite_fixed_{mode}_audit.npz'
        trajectory = artifacts / 'evaluator' / f'finite_trajectory_{mode}_audit.npz'
        sources += [fixed, trajectory]
        propensity_sa = np.full(shape[1:], np.nan)
        with np.load(fixed, allow_pickle=False) as z:
            idx = (z['state'], z['action'])
            propensity_sa[idx] = z['retention_propensity']
            flags_sa = np.zeros(shape[1:], dtype=bool)
            flags_sa[idx] = z['critical']
            assert np.array_equal(propensity_sa[idx], z['retention_propensity']), 'Retention must be time independent'
            assert np.array_equal(flags_sa[idx], z['critical']), 'Critical mask must be time independent'
            propensity = np.broadcast_to(propensity_sa, shape)
            flags = np.broadcast_to(flags_sa, shape)
            if critical is None:
                critical = flags
            else:
                assert np.array_equal(critical, flags)
        assert np.all(np.isfinite(propensity)), 'Fixed-state audit must cover every query'
        empirical = np.zeros(shape, dtype=int)
        with np.load(trajectory, allow_pickle=False) as z:
            keep = z['retained']
            np.add.at(empirical, (z['time'][keep], z['state'][keep], z['action'][keep]), 1)
        retention[mode] = propensity.tolist()
        counts[mode] = empirical.tolist()
    return {
        'schema': 'thrusterchain-guide-finite-oracle-v1',
        'artifact_dir': artifacts.name,
        'horizon': mdp.horizon, 'gamma': mdp.gamma,
        'positions': mdp.positions.tolist(), 'loads': list(mdp.config.loads),
        'actions': mdp.actions.tolist(),
        'states': np.column_stack([mdp.state_regime, mdp.state_load, mdp.state_position]).tolist(),
        'q': q.tolist(), 'logging_occupancy': occupancy.tolist(),
        'critical': critical.tolist(), 'retention': retention, 'counts': counts,
        'source_sha256': {str(f.relative_to(artifacts)).replace('\\', '/'): sha256(f.read_bytes()).hexdigest() for f in sources},
    }


class Inventory(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ids = []
        self.links = []
        self.assets = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if a.get('id'):
            self.ids.append(a['id'])
        if tag == 'a' and a.get('href', '').startswith('#'):
            self.links.append(a['href'][1:])
        for k in ('src', 'href', 'xlink:href'):
            if a.get(k, '').startswith('data:'):
                self.assets.append(a[k])


def scripts(s):
    return re.findall(r'<script\b[^>]*>.*?</script>', s, re.S | re.I)


def replace_once(text, old, new):
    n = text.count(old)
    if n != 1:
        raise ValueError(f'Expected one source match, got {n}: {old[:100]!r}')
    return text.replace(old, new, 1)


def build(source, payload):
    # read_text normalizes line endings; preserve the source's exact line ending
    # convention on write and verify against its original normalized scripts.
    s = source
    for old, new in REPLACEMENTS:
        s = replace_once(s, old, new)
    s = replace_once(s, '<h1>ThrusterChain</h1>', '''<h1>ThrusterChain</h1>
    <p class="note"><b>20 October addition:</b> <a href="#diagnostics-suite">the diagnostics suite</a> separates
      continuous control, finite-model oracles and hidden-world identifiability. The earlier lesson and results
      are preserved as the historical study; no new learner has been trained for this addition.</p>''')
    s = replace_once(s, '<h2 id="learn-h">Learn step by step</h2>', '''<h2 id="learn-h">Learn step by step</h2>
    <p class="note"><span class="tag">historical study</span> This preserved lesson teaches the original
      13-number environment and its historical experiments. The <a href="#diagnostics-suite">diagnostics suite</a>
      adds separate observed-regime and finite-horizon models.</p>''')
    s = replace_once(s, '<li class="nv-sec"><a href="#learn" data-unit="learn">Learn step by step</a></li>',
        '<li class="nv-sec"><a href="#diagnostics-suite" data-unit="diagnostics-suite">Diagnostics suite · new</a></li>\n      '
        '<li class="nv-sec"><a href="#learn" data-unit="learn">Learn step by step</a></li>')
    s = replace_once(s, '<section id="learn" class="lesson" aria-labelledby="learn-h">',
        SECTION.replace('smoke_verified_20261020', payload['artifact_dir']) + '\n  <section id="learn" class="lesson" aria-labelledby="learn-h">')
    # Historical scope notices at the old specification/results entries, without
    # changing heading IDs, chart data, interaction code or stored numerical runs.
    historical = ('action-space', 'observation-space', 'starting-state', 'dynamics', 'rewards',
                  'episode-end', 'task', 'description', 'recording-controller', 'reference-scores',
                  'dataset', 'shift', 'critic-in-action', 'methods', 'harm-checks', 'bca-test',
                  'results', 'weighting-test', 'capped-pull', 'toy-worlds', 'bca-uses', 'arguments', 'code-names')
    for sid in historical:
        pattern = r'(<section id="' + re.escape(sid) + r'"[^>]*>\s*<h2\b[^>]*>.*?</h2>)'
        s, n = re.subn(pattern, lambda m: m.group(1) + '''
    <p class="note"><span class="tag">historical study</span> This section retains the earlier experiment and its
      reported results. See the <a href="#diagnostics-suite">new suite</a> for its distinct models, objectives and
      <a href="#diag-oracles">oracle contracts</a>.</p>''', s, count=1, flags=re.S)
        # toy-worlds has an introductory element before its h2; it gets the same
        # notice immediately after its own heading in a second bounded operation.
        if n != 1:
            start = s.index(f'<section id="{sid}"')
            end = s.index('</section>', start)
            h2end = s.index('</h2>', start, end) + len('</h2>')
            s = s[:h2end] + '''
    <p class="note"><span class="tag">historical study</span> Earlier results are preserved here. The
      <a href="#diagnostics-suite">new suite</a> has separate models and oracle contracts.</p>''' + s[h2end:]
    at = s.index('<h2>', s.index('<section id="bca-uses">'))
    at = s.index('</h2>', at) + len('</h2>')
    s = s[:at] + '''
    <p class="note">The preserved interactive history uses labels such as “perfect aim” and “could not have
      helped.” Read these as historical labels for the particular test: oracle-error placement is not an
      optimal intervention, and its failure does not establish impossibility. See <a href="#diag-controls">the
      corrected interpretation</a>.</p>''' + s[at:]
    s = replace_once(s,
        "The environment itself has not changed since the main test's pre-registration (12 Oct 2026, 15:07 UTC), and its\n      registered ids all end in <code>-v0</code>. The versions below are versions of this page, newest first, with\n      times in UTC.",
        "The historical environment described below has not changed since the main test's pre-registration\n      (12 Oct 2026, 15:07 UTC), and its registered ids end in <code>-v0</code>. The diagnostics suite is an\n      additive namespace with its own contracts. The versions below are versions of this page, newest first,\n      with times in UTC.")
    vh = s.index('<section id="version-history">')
    at = s.index('<ul class="leads">', vh) + len('<ul class="leads">')
    s = s[:at] + '''
      <li><b>Diagnostics addition, 20 Oct 2026.</b> Added the observed continuous task, finite-model oracle and
        hidden paired-world contracts, data and support definitions, controls, reproducible usage and implementation
        evidence. Clarified deterministic rollouts versus population expectations and oracle-error placement versus
        optimal intervention. Historical results and all embedded scripts and assets are retained. No new learner
        training or confirmatory BCA result is reported.</li>''' + s[at:]
    assert scripts(s) == scripts(source), 'An original script/data block changed'
    data_block = '<script type="application/json" id="diagnostics-oracle-data">' + json.dumps(payload, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c') + '</script>'
    s = replace_once(s, '</body></html>', data_block + '\n' + ORACLE_JS + '\n</body></html>')
    assert scripts(s)[:len(scripts(source))] == scripts(source), 'Original script order changed'
    before, after = Inventory(), Inventory()
    before.feed(source)
    after.feed(s)
    assert Counter(before.assets) == Counter(after.assets), 'Embedded assets changed'
    assert set(before.ids) <= set(after.ids), 'Historical anchors removed'
    missing = set(after.links) - set(after.ids) - {''}
    old_missing = set(before.links) - set(before.ids) - {''}
    assert missing <= old_missing, f'New broken anchors: {missing - old_missing}'
    duplicate_new = {key for key, count in Counter(after.ids).items() if count > 1 and key not in Counter(before.ids)}
    assert not duplicate_new, f'Duplicate new anchors: {duplicate_new}'
    return s


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path(__file__).parent / 'reference' / 'ThrusterChain_10_20_1.html')
    p.add_argument('--output', type=Path, default=Path(__file__).with_name('ThrusterChain_diagnostics.html'))
    p.add_argument('--artifacts', type=Path, default=Path(__file__).parent / 'artifacts' / 'smoke_release_20261020')
    args = p.parse_args()
    if args.source.resolve() == args.output.resolve():
        p.error('--output must differ from the preserved source')
    raw = args.source.read_bytes()
    newline = '\r\n' if b'\r\n' in raw else '\n'
    source = raw.decode('utf-8').replace('\r\n', '\n')
    result = build(source, oracle_payload(args.artifacts))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(result.replace('\n', newline).encode('utf-8'))
    print(f'Wrote {args.output} ({args.output.stat().st_size:,} bytes)')
    print(f'Original script/data blocks preserved: {len(scripts(source))}')
    print(f'Source SHA-256: {sha256(raw).hexdigest()}')


if __name__ == '__main__':
    main()
