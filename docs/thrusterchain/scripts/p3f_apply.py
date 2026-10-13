"""Pass 3, final fixes: apply the verified findings to thrusterchain.html with exact, asserted replacements.

  wsl.exe -d Ubuntu-22.04 -u root -e bash -lc 'python3 /mnt/c/Users/David/dev/bca-shift-work/page/scripts/p3f_apply.py'
writes the trial copy tmp/p3f/applied.html; add --write to write thrusterchain.html as well.

Every replacement names how often its old text must occur. The v11 backup is only read (md5 checked before and
after), and the page must be the one this pass started from (md5 checked). No formula is changed, so no MathJax
re-typesetting is needed; the math containers are checked to be byte-identical before and after.
"""
import hashlib
import json
import os
import re
import shutil
import sys

PAGE_DIR = "/mnt/c/Users/David/dev/bca-shift-work/page"
PAGE = os.path.join(PAGE_DIR, "thrusterchain.html")
BACKUP = os.path.join(PAGE_DIR, "thrusterchain.v11.html")
TMP = os.path.join(PAGE_DIR, "tmp", "p3f")
V11_MD5 = "79f00557ba8803625b9f36bc497f7cc6"
PRE_MD5 = "3bebbe761bb9eee2e4546f034bcd0eda"   # the page as this pass found it
LOG = []


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def once(s, old, new, what, count=1):
    n = s.count(old)
    assert n == count, f"{what}: expected {count} match(es), found {n}: {old[:160]!r}"
    LOG.append(f"{what} ({count})")
    return s.replace(old, new)


def section(s, sid):
    a = s.index('<section id="%s">' % sid)
    b = s.index("</section>", a) + len("</section>")
    return a, b


def in_section(s, sid, old, new, what, count=1):
    a, b = section(s, sid)
    return s[:a] + once(s[a:b], old, new, f"{sid}: {what}", count) + s[b:]


MATH = re.compile(r"<mjx-container.*?</mjx-container>", re.S)

# ====================================================================================================== Results
RESULTS = [
    # --- the question, in plain words: BCQ's name, the band built on the one-step error, the two kinds of error
    ("question: BCQ and the band",
     """        keep the actor close to the recorded commands. Batch-constrained Q-learning (BCQ; Fujimoto, Meger and Precup,
        2019) and TD3+BC (Fujimoto and Gu, 2021) work this way, and so does BCA. BCA adds a <i>band</i> to TD3+BC: a
        range around the critic's estimate that should contain the critic's error. On the training rows where the band
        is wide, so that the critic may be wrong, BCA makes TD3+BC's pull toward the recorded command stronger.</li>""",
     """        keep the actor close to the recorded commands. Batch-constrained deep Q-learning (BCQ; Fujimoto, Meger and
        Precup, 2019) and TD3+BC (Fujimoto and Gu, 2021) work this way, and so does BCA. BCA adds a <i>band</i> to
        TD3+BC, built on the critic's <i>one-step error</i>: the reward plus 0.98 times the critic's value of the next
        step, minus its value of this step. The band is a range that should contain this one-step error 90% of the
        time, so its width says how wrong the critic may be. On the training rows where the band is wide, BCA makes
        TD3+BC's pull toward the recorded command stronger.</li>"""),
    ("question: two kinds of critic error",
     """        choose them. Conservative Q-learning (CQL; Kumar, Zhou, Tucker and Levine, 2020) is the best known.</li>
    </ul>
""",
     """        choose them. Conservative Q-learning (CQL; Kumar, Zhou, Tucker and Levine, 2020) is the best known.</li>
    </ul>
    <p><b>Two kinds of critic error.</b> This section uses two, and always says which one it means. The critic's
      <i>error against the exact value</i> is how far its estimate is from the true value, which in this environment
      can be computed exactly by simulation; the harm checks below measure this error. The critic's <i>one-step
      error</i>, defined above, needs no exact value, so BCA can compute it from recorded data alone; BCA's band covers
      this error.</p>
"""),
    ("question: why the command was removed",
     """    <p>In this environment the recording lost its most common command on purpose. The rule that removed it, the
      densest-15% removal, deletes the 15% of recorded commands that sit closest together. This project used the same
      rule in its earlier runs on the Hopper datasets of D4RL (datasets for deep data-driven reinforcement learning),
      the standard collection of recorded datasets for offline reinforcement learning (Fu, Kumar, Nachum, Tucker and
      Levine, 2020).""",
     """    <p>In this environment the recording lost its most common command on purpose. This copies a common problem in
      offline reinforcement learning, a recording that lacks commands a better controller would need, in a world where
      every true value can be computed, so that the critic's guesses at the missing commands can be checked exactly.
      The rule that removed the command, the densest-15% removal, deletes the 15% of recorded commands that sit closest
      together. The removal is this project's own rule, which it also applied in its earlier runs to the Hopper
      datasets (recordings of a simulated one-legged hopping robot) of D4RL (datasets for deep data-driven
      reinforcement learning), the standard collection of recorded datasets for offline reinforcement learning (Fu,
      Kumar, Nachum, Tucker and Levine, 2020); D4RL itself contains no such removal."""),
    # --- what was found, in short
    ("in short: which error",
     """removal was 2.15 to 2.40 times as wrong as the critic trained on the full recording, and no more wrong
        elsewhere.""",
     """removal was 2.15 to 2.40 times as wrong as the critic trained on the full recording (in its error against
        the exact value), and no more wrong elsewhere."""),
    ("in short: the third question",
     """      <li><b>Could a harder pull toward the recorded commands win the return back?</b> With doses inside BCA's range,
        placed by the critic's exact error, the pull scored 1.9 points below the same pull spread evenly in strong
        current, and only 0.9 and 0.8 points above it with big waves and big starting pushes, against a required 3. So
        no, and the pre-registered rule stopped the test before any BCA run.</li>""",
     """      <li><b>Could a harder pull toward the recorded commands win the return back?</b> An actor was given extra pull,
        multiplied on each training row by a dose from 1 to 1.5 (BCA's range of doses) and placed by the critic's exact
        error. In strong current it scored 1.9 points below an actor given a uniform extra pull tuned to change the
        actor by the same amount (strength-matched uniform caution), and with big waves and big starting pushes only
        0.9 and 0.8 points above it, against a required 3. So no: at BCA's strengths, a harder pull toward the recorded
        commands could not win back the lost return in strong current, even when it was placed by the exact error, and
        the pre-registered rule stopped the test before any BCA run.</li>"""),
    # --- finding 1
    ("critic-error: which error the ratio uses",
     """<i>error ratio</i>: the critic's typical error on those rows (the median size of its error in judging how much""",
     """<i>error ratio</i>: the critic's typical error against the exact value on those rows (the median size of its
      error in judging how much"""),
    ("critic-error: pooled defined",
     """<td>at least 2× pooled over the seeds, and at least 1.5× in every seed</td>""",
     """<td>at least 2× pooled (combined over the five seeds), and at least 1.5× in every seed</td>"""),
    ("critic-error: no row spans, plain wording",
     """<td rowspan="2">the critic trained after the removal exceeds the full-log critic by at least a quarter of the median true price of pushing past 80%, in at least 4 of the 5 seeds</td><td rowspan="2"><span class="tag ok">pass</span></td></tr>
        <tr><td>Extra preference for full throttle: the critic trained on the full log</td><td class="n">−0.06 to −0.01</td></tr>""",
     """<td>the critic trained after the removal exceeds the full-log critic by at least a quarter of the median true loss of value from pushing past 80% (see <a href="#shift">The distribution shift</a>), in at least 4 of the 5 seeds</td><td><span class="tag ok">pass</span></td></tr>
        <tr><td>Extra preference for full throttle: the critic trained on the full log</td><td class="n">−0.06 to −0.01</td><td>one rule covers this row and the row above: see the row above</td><td><span class="tag ok">pass</span></td></tr>"""),
    # --- finding 2
    ("return-cost: why it matters",
     """      <dt>Why it matters in offline reinforcement learning</dt><dd>TD3+BC's pull toward the recorded commands exists to
        stop exactly this kind of damage. But the pull acts only at the states in the recording. Every state of strong
        current lies outside the recorded conditions (its current is above 0.7), so there nothing holds the actor near
        a recorded command, and it acts on what it learned from the critic. Offline reinforcement learning expects the
        damage where the learned controller meets states the recording does not cover, and strong current is that case
        here.</dd>""",
     """      <dt>Why it matters in offline reinforcement learning</dt><dd>TD3+BC's pull toward the recorded commands exists to
        stop exactly this kind of damage, but it acts only during training, at the states in the recording. What the
        actor learns there from the critic, it then applies everywhere, including states the recording does not cover.
        Every state of strong current lies outside the recorded conditions (its current is above 0.7). So a wrong
        belief that the actor learned from the critic at recorded states, for example that more throttle past 80% is
        better, is carried into strong current, where holding against the current asks thruster 1 for 80% or more.
        Offline reinforcement learning expects the damage where the learned controller meets states the recording does
        not cover, and strong current is that case here.</dd>"""),
    ("return-cost: why a fixed controller is scored",
     """        return is caused by the critic. They were scored on the 1,000 strong-current starts.""",
     """        return is caused by the critic. This differs from normal TD3+BC, where the critic scores the actor itself:
        fixing the controller that the critic scores makes the critic's exact values computable, and it makes the
        critic the only difference between the two actors. Plain TD3+BC, reported below, shows the same pattern. The
        actors were scored on the 1,000 fixed strong-current starts."""),
    ("return-cost: the five rules, and why weight 2.5 failed",
     """    <p><b>At critic weight 2.5</b> the loss was already 75 to 93 points, and four of the five rules passed. The rule
      that failed checks that the loss comes from the stall: the actor of the critic trained after the removal must
      push past 80% at least 1.5 times as often as the actor of the full-log critic. It did so 1.46 times as often,
      because at this weight the pull toward the recorded commands is strong enough that both actors push past 80%
      often in strong current. The step-up rule then ran the check again at 5.</p>""",
     """    <p><b>The five pass rules.</b> The check passes only if all five hold (they are set out in full in <a
      href="#harm-checks">The harm checks</a>): the actor of the critic trained after the removal scores at least 5
      points below the actor of the full-log critic; that loss exceeds what removing the same number of rows at random
      explains by at least 3 points; it exceeds the same comparison in the no-stall comparison world by at least 3
      points; that actor pushes past 80% at least 1.5 times as often as the actor of the full-log critic; and the lost
      return sits at the states where it pushes past 80%.</p>
    <p><b>At critic weight 2.5</b> the loss was already 75 to 93 points, and four of the five rules passed. The rule
      that failed is the fourth, which checks that the loss comes from the stall: the actor of the critic trained after
      the removal pushed past 80% only 1.46 times as often as the actor of the full-log critic. The reason is that the
      full log still contains the recording controller's full-throttle commands, so at this weight the pull toward
      the recorded commands drags the actor of the full-log critic past 80% as well, and the two actors differ less.
      The step-up rule then ran the check again at 5.</p>"""),
    # --- finding 3
    ("caution: which critic, which baseline",
     """        from the same cloned start, against the same frozen critic (the critic of a plain TD3+BC run on the log after
        the removal), at critic weight 5.</dd>""",
     """        from the same cloned start, against the same frozen critic (the critic of a plain TD3+BC run on the log after
        the removal), at critic weight 5. This critic differs from the return-cost check's: in a plain TD3+BC run the
        critic and the actor are trained together, as in normal use, so it is the kind of critic BCA sits on. Against
        it, the actor with no extra caution scores −10.1 in strong current, near the −14.3 of the return-cost check's
        actor, and every gain below is measured from −10.1.</dd>"""),
    ("caution: amount-matched defined",
     """<tr><td>Amount-matched</td><td>the same total extra caution as harm-placed, spread evenly</td><td>the same total, with no aim</td></tr>""",
     """<tr><td>Amount-matched</td><td>every row gets the mean of the harm-placed doses, so the sum of extra dose (dose minus 1) over the rows is the same as harm-placed caution's; strength-matched uniform caution instead matches how much the actor's commands change</td><td>the same total amount of extra pull, with no aim</td></tr>"""),
    ("caution: the reasoning, step by step",
     """      <li><b>In strong current, where 99 points are lost, the pull barely moves the actor.</b> The share of
        strong-current states at which the actor pushes past 80% stays between 0.61 and 0.68 for every variant with
        doses of 1.5 or less. Widening the range to 2 beats strength-matched uniform caution by only 0.2 points, so
        BCA's cap is not what limits it.</li>
      <li><b>Only strong, uniform caution reaches the harm, and only in part.</b> Uniform ×3 lowers the share past 80%
        to 0.50 and recovers 17 of the 99 points. It still leaves the actor 47 points below the cloned actor, and it
        costs 5 to 13 points in the other situations.</li>
      <li><b>Why the pull cannot reach the harm.</b> The pull acts at the recorded training states. Every strong-current
        state lies outside the recorded conditions, so there the actor is on its own, and what it carries there from
        training is a critic that sees no cost in pushing past 80%: the log after the removal never shows the stall,
        and it cannot tell the stalling world from the no-stall comparison world. Strong current is also where the
        removed commands are needed most: the stall-aware controller's own command lies inside the removed region at
        57.0% of the states it visits in strong current, against 17.2% in the recorded conditions (the <a
        href="#shift-card">shift table</a>). For comparison, the cloned actor, which has no critic, pushes past 80% on
        9% of the strong-current states it visits and scores 54.</li>""",
     """      <li><b>In strong current the pull barely moves the actor.</b> With no extra caution the actor scores −10.1
        there, and for every variant with doses of 1.5 or less the share of strong-current states at which it pushes
        past 80% stays between 61% and 68%. Widening the range to 2 beats strength-matched uniform caution by only 0.2
        points, so BCA's cap of 1.5 is not what limits it.</li>
      <li><b>Only strong, uniform caution reaches the harm, and only in part.</b> Uniform ×3 lowers the share past 80%
        to 50% and raises the strong-current score from −10.1 to +7.1. That still leaves the actor 47 points below the
        cloned actor (54.4), and it costs 5 to 13 points in the other situations.</li>
      <li><b>Why the pull cannot reach the harm.</b> The pull draws the actor toward the recorded command. At the
        training states in the full-throttle region, the recorded thruster-1 throttle averages 0.51, while that of the
        actor trained against the critic of the log after the removal averages 0.71 (<a href="#beyond-the-data">After
        the stop</a>). So a harder pull says "drive more like the recording"; it never tells the actor that pushing past
        80% is bad. The critic's wrong belief, that more throttle past 80% is better, therefore stays: the log after the
        removal never shows the stall, so nothing in the data contradicts it. The actor learns that belief at the
        training states and carries it into strong current, where every state lies outside the recorded conditions.
        Strong current is also where the removed commands are needed most: the stall-aware controller's own command
        lies inside the removed region at 57.0% of the states it visits in strong current, against 17.2% in the recorded
        conditions (<a href="#shift-card">the shift table</a>, in the section The distribution shift). For comparison,
        the cloned actor, which has no critic, pushes past 80% on 9% of the strong-current states it visits and scores
        54.4. Stage 1, in <a href="#beyond-the-data">After the stop</a>, shows that a penalty on the critic's value of
        the actor's own command, applied at the same training states, does reach the harm.</li>"""),
    # --- what the stop means
    ("stop: signal defined",
     """<dt>The problem</dt><dd>A stop before BCA could have two causes: BCA's signal could be poor, or caution of BCA's""",
     """<dt>The problem</dt><dd>A stop before BCA could have two causes: BCA's signal (its band width, meant to be large
        where the critic is wrong) could be poor, or caution of BCA's"""),
    ("stop: no path in running text",
     """The full stop report, with every table, the
        controls and the files, is in the repository at <code>outputs/shift/stop-report/README.md</code>.</dd>""",
     """The full stop report, with every table, the
        controls and the files, is in the repository (the sources note after the timeline gives its place).</dd>"""),
    # --- the forecast
    ("forecast: fixed imitations",
     """The design plan forecast the harm from fixed stand-ins for the learned
      actor, simulated exactly""",
     """The design plan forecast the harm from fixed imitations of the learned
      actor (fixed rules that move part of the way toward the command the critic rates highest), simulated exactly"""),
    ("forecast: imitation crosses 80%",
     """caution placed exactly where the stand-in
      crosses 80%""",
     """caution placed exactly where the fixed imitation
      crosses 80%"""),
    ("forecast: two imitations",
     """the way one of the two
      stand-ins does.""",
     """the way one of the two
      fixed imitations does."""),
    ("forecast: steeper-stall rerun written out",
     """counting the steeper-stall rerun: about 25–30%;""",
     """counting the one allowed rerun with a steeper stall (past 80%, push falls by 3.0 per unit of throttle
        instead of 1.5): about 25–30%;"""),
    ("forecast: either imitation",
     """larger than either stand-in predicted (9.5 and 40.4).""",
     """larger than either fixed imitation predicted (9.5 and 40.4)."""),
    ("forecast: why caution could not help",
     """The learned actor instead carries the critic's wrong belief into states the
      data never covered, where the pull toward the recorded commands does not reach.</p>""",
     """The learned actor instead learns the critic's wrong belief at the training states,
      where a harder pull draws it toward the recorded command rather than away from the stall, and carries that belief
      into states the data never covered.</p>"""),
    # --- the world checks: labelled blocks and the table's terms
    ("world: labelled blocks",
     """    <p><b>The problem.</b> Before any learner runs, the world itself must have the properties the test relies on: the
      removal must take the full-throttle commands and hide the stall while keeping the states; the two worlds must
      look the same in the data; the stall must matter; and the forecast harm must be large enough to find. <b>The
      reasoning.</b> None of this needs learning, so nine <i>world checks</i> test it directly, with rules taken from
      the pre-registration. They ran before the pre-registration, on design seeds 900–904: five recordings made only to
      design and check the world.</p>
    <p><b>How to read the table.</b> Each row is one check, and its first cell says what the check guards against.
      "Rule" is the pass rule from the pre-registration. "Seed 900" is the value on the design seed whose recorded log
      this page shows. The last column gives the lowest and the highest value over the five design seeds.</p>""",
     """    <dl class="kv kv1 pro">
      <dt>The problem</dt><dd>Before any learner runs, the world itself must have the properties the test relies on:
        the removal must take the full-throttle commands and hide the stall while keeping the states; the two worlds
        must look the same in the data; the stall must matter; and the forecast harm must be large enough to
        find.</dd>
      <dt>The reasoning</dt><dd>None of this needs learning, so nine <i>world checks</i> test it directly, on design
        seeds 900–904: five recordings made only to design and check the world. The checks were run while the world
        was being designed, before the pre-registration was written. The pre-registration then fixed their pass rules
        and recorded that the run had passed, with the run's fingerprint, so the rules in the table are the
        pre-registration's.</dd>
      <dt>Terms used in the table</dt><dd>The densest-15% removal treats each recorded command as a point in four
        dimensions, one per thruster, and deletes every command inside the densest <i>ball</i> (all points within a
        fixed distance, its <i>radius</i>, of a centre) that holds 15% of them. Full throttle on thruster 1 is −1,
        because it pushes upstream, against the water. A row <i>follows the identical rule in both worlds</i> when none
        of its throttles is past 80%, so the same physics applies to it in both worlds. <i>Normal driving only</i> is
        the recording controller's normal driving without its random noise and never the full-throttle command (<a
        href="#recording-controller">The recording controller</a>). <i>TD3's target noise</i> is the random noise that
        TD3 adds to the next command when it trains its critic (<a href="#methods">Methods</a>).</dd>
      <dt>How to read the table</dt><dd>Each row is one check, and its first cell says what the check guards against.
        "Rule" is the pass rule from the pre-registration. "Seed 900" is the value on the design seed whose recorded
        log this page shows. The last column gives the lowest and the highest value over the five design seeds.</dd>
    </dl>"""),
    ("world: the outcome as a block (start)",
     """    <p><b>The outcome.</b> Every check passed on every seed.""",
     """    <dl class="kv kv1 pro">
      <dt>The outcome</dt><dd>Every check passed on every seed."""),
    ("world: the outcome as a block (end)",
     """.</span></p>

    <h3 id="results-timeline">Timeline</h3>""",
     """.</span></dd>
    </dl>

    <h3 id="results-timeline">Timeline</h3>"""),
    # --- timeline: codes move to a sources note
    ("timeline: no hash in running text",
     """      <li>15:07: the pre-registration was written. Its SHA-256 fingerprint is <code>68d9e8779e148b7b…</code>, and a
        copy was saved in the repository as a fixed version (saved version <code>bdb8c8a</code> in the repository's history) before any learner ran.</li>""",
     """      <li>15:07: the pre-registration was written, and a fingerprinted copy was saved in the repository before any
        learner ran.</li>"""),
    ("timeline: sources note",
     """The harm checks stopped the test.</li>
    </ul>
    <div class="subsec" id="beyond-the-data">""",
     """The harm checks stopped the test.</li>
    </ul>
    <p class="note">Sources in the repository: the pre-registration (SHA-256 fingerprint
      <code>68d9e8779e148b7b…</code>, saved in commit <code>bdb8c8a</code>) and the full stop report,
      <code>outputs/shift/stop-report/README.md</code>.</p>
    <div class="subsec" id="beyond-the-data">"""),
    # --- after the stop
    ("after: where this fits",
     """    <p><b>Where this fits.</b> This subsection is the record of an exploratory follow-up made after the stop, kept so
      that its findings and their numbers stay available and can be checked. It is no longer where the work on this
      environment is led: the main line is now <a href="#weighting-test">the weighting test</a>, the next section, and
      the work is led from there. The environment itself did not change.</p>""",
     """    <p><b>Where this fits.</b> This subsection records the exploratory follow-up to the stop, so that its findings
      and their numbers stay available and can be checked, and its parts still under way (Stage 2 and the data-only
      penalties) are reported here. The project's main experiment on this environment is now <a
      href="#weighting-test">the weighting test</a>, in the next section. The environment itself did not change.</p>"""),
    ("after: in short, as a linked summary",
     """    <h4>In short</h4>
    <ul>
      <li><b>Where the loss comes from.</b> Past 80%, the critic trained after the removal rates more throttle as
        better, where the truth is that it is worse. The actor learns that belief at training states and carries it
        into strong current.</li>
      <li><b>Why BCA as built cannot touch it.</b> BCA reads its band only at recorded commands, where this critic is
        accurate, and its dose only strengthens the pull toward the recorded command.</li>
      <li><b>What can reach it (Stage 1).</b> A penalty on the critic's value of the actor's own command, built from
        the critic's exact error, recovered 93.9 of the 98.9 points lost in strong current. Untargeted caution of the
        same strength recovered at most 4.6.</li>
      <li><b>At deployment.</b> A distance check that clips commands the data do not show recovers about half the harm
        (57%), and what limits it is detection.</li>
      <li><b>Next.</b> BCA's own test of this route (Stage 2) is written and approved by the project owner; no BCA run
        has started.</li>
    </ul>""",
     """    <h4>In short: a summary of the findings argued below</h4>
    <ul>
      <li><b>Where the loss comes from</b> (<a href="#ats-loss">argued below</a>). Past 80%, the critic trained after
        the removal rates more throttle as better, where the truth is that it is worse. The actor learns that belief at
        training states and carries it into strong current.</li>
      <li><b>Why BCA as built cannot correct it</b> (<a href="#ats-bca">argued below</a>). BCA reads its band only at
        recorded commands, where this critic is accurate, and its dose only strengthens the pull toward the recorded
        command.</li>
      <li><b>What can reach it</b> (<a href="#ats-stage1">Stage 1</a>). A penalty on the critic's value of the actor's
        own command, built from the critic's exact error, recovered 93.9 of the 98.9 points lost in strong current.
        Untargeted caution of the same strength recovered at most 4.6.</li>
      <li><b>At deployment</b> (<a href="#ats-deploy">argued below</a>). A distance check that <i>clips</i> commands
        lying far from the recorded data (it replaces any throttle outside the range of the recorded commands with the
        nearest value inside it) recovers about half the harm (57%). What limits it is detection: it misses most of the
        commands it should catch.</li>
      <li><b>Next</b> (<a href="#ats-claims">the two claims</a>). BCA's own test of this route (Stage 2) is written and
        approved by the project owner. No Stage 2 run of BCA has started; the BCA runs now under way belong to the
        weighting test.</li>
    </ul>"""),
    ("after: plan file named in sources only",
     """answered with a written plan
      (<code>PLAN.md</code>, listed under Sources at the end of this subsection).""",
     """answered with a written plan
      (listed under Sources at the end of this subsection)."""),
    ("after: options list, which BCA run",
     """project owner; no BCA run has started yet;</li>""",
     """project owner; no Stage 2 run of BCA has started yet;</li>"""),
    ("after: anchor for the loss finding",
     """    <div class="finding">
      <p class="f-title">Where the loss comes from</p>""",
     """    <div class="finding" id="ats-loss">
      <p class="f-title">Where the loss comes from</p>"""),
    ("after: anchor and wording of the BCA finding",
     """    <div class="finding">
      <p class="f-title">Why BCA as built cannot touch that belief</p>""",
     """    <div class="finding" id="ats-bca">
      <p class="f-title">Why BCA as built cannot correct that belief</p>"""),
    ("after: anchor for Stage 1",
     """    <h4>Stage 1: can caution reach""",
     """    <h4 id="ats-stage1">Stage 1: can caution reach"""),
    ("stage 1: no commit codes, bank explained",
     """    <p><b>Locked before any return.</b> Every arm, formula, control and mark was written into a file and committed to
      the repository (commit <code>a1ea28e</code>) before any return of a trained arm was
      computed. The controls were matched from strengths alone. A lock file holding every actor's fingerprint was then
      written, and only after it was checked were exact returns computed, on the full 1,000-start bank of each
      situation. Seeds 0 and 1 ran first, with a <i>futility look</i>: an early check, fixed
      in advance, that would have stopped the test if the oracle had clearly failed (the first row of the marks below).
      It said continue. The results were committed in commit <code>fbad4da</code>.</p>""",
     """    <p><b>Locked before any return.</b> Every arm, formula, control and mark was written into a file and committed to
      the repository before any return of a trained arm was computed. The controls were matched from strengths alone. A
      lock file holding every actor's fingerprint was then written, and only after it was checked were exact returns
      computed, on all 1,000 fixed starting states on which each situation is tested (its <i>evaluation bank</i>).
      Seeds 0 and 1 ran first, with a <i>futility look</i>: an early check, fixed in advance, that would have stopped
      the test if the oracle had clearly failed (the first row of the marks below). It said continue. The results were
      then committed to the repository as well.</p>"""),
    ("stage 1 found: rounding note",
     """which shows the damage the critic does when
      nothing stops it.</p>""",
     """which shows the damage the critic does when
      nothing stops it. Differences are computed from the unrounded means, so they can differ by 0.1 from the
      difference of the rounded scores shown.</p>"""),
    ("stage 1 found: what to look for in the table",
     """untargeted penalties of matched strength, and the clip, which uses no signal.</p>""",
     """untargeted penalties of matched strength, and the clip, which uses no signal. What to look for: in every
      situation the oracle and its variants score far above the untargeted penalties; and in strong current the oracle
      and four of its five variants lower the share of states past 80% to 50% to 59% (the plan's literal form only to
      68%), and the clip to 26%, against 73% with no extra caution, while the untargeted penalties stay at 64% to
      73%.</p>"""),
    ("no-stall: the rising-thrust variant explained",
     """The oracle costs
      at most 1.3 points (in strong current), and 4.6 when it is scored with the next command of the rising-thrust
      controller.""",
     """The oracle costs
      at most 1.3 points (in strong current). Its exact error assumes a controller that acts after the step: normally
      the stall-aware controller, the one the critic scores. Assuming instead the <i>rising-thrust controller</i> (the
      no-stall world's counterpart of the stall-aware controller: the same textbook command, allowed up to 100%
      throttle because push keeps rising), the oracle costs 4.6 points in strong current."""),
    ("no-stall: row label",
     """<td>Oracle, scored with the next command of the rising-thrust controller</td>""",
     """<td>Oracle, assuming the rising-thrust controller acts after the step</td>"""),
    ("clip finding: the trace test bed explained",
     """          is why the claim that BCA identifies the harm, not just novelty, needs the trace test bed below.</dd>""",
     """          is why the claim that BCA identifies the harm, not just novelty, needs a test bed in which a trace of the
          removed commands is kept (0.5% or 2% of them, described under the second claim below).</dd>"""),
    ("deployment: anchor",
     """    <h4>At deployment, with no training:""",
     """    <h4 id="ats-deploy">At deployment, with no training:"""),
    ("deployment: bank explained",
     """It was scored
      on the first 200 starts of each bank, where the harm""",
     """It was scored
      on the first 200 of the 1,000 fixed starting states of each situation, where the harm"""),
    ("deployment: caption",
     """stalling world; mean over seeds 0–4, on the first 200 starts of each bank.</span></caption>""",
     """stalling world; mean over seeds 0–4, on the first 200 of the 1,000 fixed starting states of each situation.</span></caption>"""),
    ("claims: anchor",
     """    <h4>The two claims, and where they stand</h4>""",
     """    <h4 id="ats-claims">The two claims, and where they stand</h4>"""),
    ("first claim: plain status, no codes",
     """        <p><b>Status: approved by the project owner; not run yet.</b> Stage 2, BCA itself on the route above, is written as
          a pre-registration (version 6): first as an exploration on <span class="nw">seeds
          0–4</span>, then as the deciding test, with every rule fixed in advance,
          on fresh <span class="nw">seeds 5–9</span>. It is committed to the
          repository and pushed (commit <code>0af5e0a</code>), in
          <code>outputs/shift/beyond-the-data/stage2/</code>, the folder the project owner approves from on the project owner's own
          machine. Its fingerprint, the SHA-256 of <code>prereg.json</code>, is
          <code>9b38c7c48a303cfd67e5db6f137e3a162997bb533d86b3f7647e0467b91ca3b1</code>. The fingerprints it records of the other pieces' code all
          match the current files. The project owner approved this exact file on 13 October 2026 at 08:32 UTC, typing
          the start of its fingerprint at a terminal on the project owner's own machine and committing the record (commit
          <code>7dab7d5</code>). The approval was then checked against git and
          recorded here on 13 October 2026 at 08:33 UTC (commit
          <code>2eab3c4</code>). No BCA run has started yet.</p>
        <p>Ahead of it come three exploratory pieces with no BCA. The <a href="#standins-ready">data-only stand-ins</a>,
          signals computed from the data alone and put in the oracle's slot, set the bar BCA must clear. They are
          running: their training started on 13 October 2026 at 08:19 UTC. Three of them can be measured in
          two forms, the plan's formulas or forms measured relative to the row's recorded command. Before any of their
          marks was computed, the project owner chose the forms measured from the recorded command to count toward the marks;
          the plan's formulas are reported beside them. The further deployment checks are
          running, and the locked rehearsal of how detection will be scored has finished. None of their results is
          reported on this page.</p>""",
     """        <p><b>Status: approved by the project owner; not run yet.</b> Stage 2, BCA itself on the route above, is written
          as Stage 2's pre-registration (sixth version): first as an exploration on <span class="nw">seeds 0–4</span>,
          then as the deciding test, with every rule fixed in advance, on fresh <span class="nw">seeds 5–9</span>. It is
          committed to the repository with its fingerprint, and the fingerprints it records of the other pieces' code
          all match the current files. The project owner approved this exact file on 13 October 2026 at 08:32 UTC by
          typing the start of its fingerprint at a terminal on the project owner's own machine; the approval was checked
          against the repository's history and recorded at 08:33 UTC. No Stage 2 run of BCA has started; the BCA runs
          now under way belong to the weighting test.</p>
        <p id="standins-ready">Ahead of it come three exploratory pieces with no BCA. First, eight <i>data-only
          penalties</i> (signals computed from the recorded data alone, put where the oracle was) are being trained,
          each matched in strength to the oracle, to set the bar BCA must clear; their training started on 13 October
          2026 at 08:19 UTC. A penalty counts as recovering enough of the harm if, pooled over seeds 0–4, its gain over
          no extra caution in strong current is at least half the oracle's, it scores at least no extra caution minus 3
          points in the recorded conditions, and it beats a matched even pull by at least 15 points in strong current.
          Three of the penalties can be measured in two forms, the plan's formulas or forms measured relative to the
          row's recorded command; before any result was computed, the project owner chose the forms measured from the
          recorded command to count, with the plan's formulas reported beside them. Second, further deployment checks
          are running. Third, the locked rehearsal of how detection will be scored has finished. None of their results
          is reported on this page.</p>"""),
    ("sources: codes moved here",
     """      and <code>README.md</code> with the approval steps).""",
     """      and <code>README.md</code> with the approval steps). Stage 1's specification was committed in
      <code>a1ea28e</code> before any return was computed, and its results in <code>fbad4da</code>. Stage 2's
      pre-registration has SHA-256 <code>9b38c7c48a303cfd67e5db6f137e3a162997bb533d86b3f7647e0467b91ca3b1</code>
      (committed in <code>0af5e0a</code>); the project owner's approval was committed in <code>7dab7d5</code> and
      checked in <code>2eab3c4</code>. The data-only penalties' specification (third version) has SHA-256
      <code>62ad77b7ce39d904…</code> and its code manifest <code>88e5fd388911a991…</code>; their launch record is
      <code>runs/shift_explore/standins/logs/launch_all.log</code>."""),
]

# ====================================================================================================== weighting test
WEIGHTING = [
    ("status line",
     """    <p class="status"><span class="tag ok">pre-registered results in</span> <span class="tag acc">BCA training:
      exploratory, running</span> Version 5 of the specification was approved and frozen before any run, and its report
      finished on 13 October 2026 at 14:34 UTC. BCA's band and its coverage were measured on all five data qualities.
      Under the pre-registered rules no quality qualified for training BCA itself; at the project owner's request,
      BCA's training runs are now running as an exploratory run, outside the pre-registered claims.</p>""",
     """    <p class="status"><span class="tag ok">pre-registered results: complete</span> <span class="tag acc">BCA training:
      exploratory, running</span> The fifth version of the weighting test's specification was approved and frozen
      before any run, and its report finished on 13 October 2026 at 14:34 UTC. BCA's band, and its <i>coverage</i> (how
      often the band contains the critic's one-step error), were measured on all five data qualities. Under the
      pre-registered rules no quality was allowed a training run of BCA itself, because on every quality one of the two
      conditions that a return gain needs was missing (finding 4 below); at the project owner's request, BCA's
      training runs are now running as an exploratory run, outside the pre-registered claims.</p>"""),
    ("claim 1 as a question",
     """<div class="box"><h3>1. BCA's mechanics work</h3>
        <p>With no shift, BCA's band should cover the critic's error about 90% of the time, be wider where the critic
          is truly more wrong, and give caution that follows the width.</p>""",
     """<div class="box"><h3>1. Do BCA's mechanics work?</h3>
        <p>With no shift, BCA's band should contain the critic's one-step error about 90% of the time, be wider where
          the critic's error against the exact value is larger, and give caution that follows the width.</p>"""),
    ("claim 2 as a question", """<h3>2. Weighting helps under a known shift</h3>""",
     """<h3>2. Does weighting help under a known shift?</h3>"""),
    ("claim 2: no undefined distribution", """<p>When the test distribution is known, <span class="term\"""",
     """<p>When it is known exactly how the test episodes differ from training, <span class="term\""""),
    ("claim 2: calibration rows defined",
     """          the calibration rows by the exact ratio of test to training probability should restore the 90% <span""",
     """          the calibration rows (1,000 recorded steps kept out of training, on which BCA sets how wide its band must
          be) by the exact ratio of each row's chance at test to its chance in training should restore the 90% <span"""),
    ("claim 2: what could improve",
     """          that unweighted BCA loses, and could improve the result on the test distribution.</p>""",
     """          that unweighted BCA loses, and could improve the learned controller's return in the test episodes.</p>"""),
    ("strip: specification", """<span class="t">Specification, version 5</span>""",
     """<span class="t">Specification, fifth version</span>"""),
    ("strip: band and coverage", """<span class="s">done: passed on all 5 qualities</span>""",
     """<span class="s">done: both claims hold on all 5 qualities; one side check failed on medium-expert</span>"""),
    ("strip: checks before training", """<span class="s">done: no quality qualified</span>""",
     """<span class="s">done: no quality allowed a BCA training run</span>"""),
    ("how to read: test settings defined first",
     """    <p><b>How to read the numbers.</b> <i>Coverage</i> is the share of steps whose critic error falls inside the band;""",
     """    <p><b>How to read the numbers.</b> The test episodes come in three <i>test settings</i>, which differ only in how
      often each strength of current appears: the <i>strong test</i>, in which strong currents are the most common;
      the <i>mild test</i>, which makes every current from 0.1 to 1.1 equally likely; and <i>no shift</i>, which draws
      currents as training did (the section <a href="#iw-shift">The known shift</a>, below, gives the exact
      distributions). <i>Coverage</i> is the share of steps whose critic one-step error falls inside the band;"""),
    ("qualities: the 80% hold explained first",
     """The removed region (the 80% hold) is the expert's, and it is cut from every
      quality, so all five share it.""",
     """In this test the recording controller often holds thruster 1 at exactly 80%
      when the water pushes hard (the <i>80% hold</i>). Because the hold repeats exactly, it is the densest cluster of
      commands, and the densest-15% removal deletes it (the <i>removed region</i>; the section <a href="#iw-hold">The
      removed command</a>, below, explains it). The removed region is the expert's, and it is cut from every quality,
      so all five share it."""),
    ("qualities: what the data column is",
     """"The data" is the recording itself, replayed.""",
     """"The data" is the controller that made the recording, with the quality's random
      share, scored on the strong test."""),
    ("qualities: which columns are seed means",
     """All three are in this test's points on the strong test.</p>""",
     """All three are in this test's points on the strong test; the two TD3+BC columns are means over seeds 0–4.</p>"""),
    ("qualities: caption",
     """100 = stall-aware controller); mean over seeds 0–4.</span></caption>""",
     """100 = stall-aware controller); the two TD3+BC columns are means over seeds 0–4.</span></caption>"""),
    ("qualities: random's removal cost is noise",
     """    <p>TD3+BC's actor scores above its own data on every quality except the expert,""",
     """    <p>On the random quality the full log and the log after the removal are almost identical (they differ in only
      0.12–0.17% of rows, because random commands almost never fall in the removed region), so its −1.7 is run-to-run
      noise, not a cost. TD3+BC's actor scores above its own data on every quality except the expert,"""),
    ("finding 1: which error the band holds",
     """<dt>The problem</dt><dd>A band that claims to hold the critic's error 90% of the time is useful only if it does,""",
     """<dt>The problem</dt><dd>A band that claims to hold the critic's one-step error 90% of the time is useful only if
          it does,"""),
    ("finding 1: the guarantee, not the only warning",
     """acting, so a calibrated statement of the critic's error is the only warning it gets. Conformal calibration,""",
     """acting, so a calibrated statement of the critic's error is valuable: unlike other warnings, such as
          disagreement between several critics or how rare a command is in the data, it comes with a guarantee of how
          often it holds. Conformal calibration,"""),
    ("finding 1: error against the exact value",
     """between the band's width and the critic's exact error, within each tenth of the steps sorted by current, so""",
     """between the band's width and the critic's error against the exact value (how far its estimate is from the
          true value, computed by simulation), within each tenth of the steps sorted by current, so"""),
    ("finding 2: one-step errors",
     """and the critic's errors are larger
          there""",
     """and the critic's one-step errors are larger
          there"""),
    ("finding 2: threshold defined",
     """<dt>What was measured</dt><dd>Coverage on test steps outside the removed region for three bands: unweighted,
          exactly weighted (each step at its own threshold), and weighted by an estimate from a classifier that sees
          only observations.</dd>""",
     """<dt>What was measured</dt><dd>Coverage on test steps outside the removed region for three bands: unweighted,
          exactly weighted, and weighted by an estimate from a classifier that sees only observations. BCA fixes the
          band's width with a <i>threshold</i>: the band multiplier β that it sets on the calibration rows (<a
          href="#methods">Methods</a>). With weights, that multiplier depends on the weight of the step being judged,
          so each step gets its own threshold; "one threshold for every row", a comparison reported below, uses a
          single multiplier for all.</dd>"""),
    ("finding 2: 11.0, percentage points",
     """Under the strong test, weighting raises coverage by 6.0 to 11.1 percentage points. The estimated weights land
          within 0.1 to 0.8 points of the exact ones under the strong test, and with no shift within 0.0 to 0.2 points
          of the unweighted band.""",
     """Under the strong test, weighting raises coverage by 6.0 to 11.0 percentage points. The estimated weights land
          within 0.1 to 0.8 percentage points of the exact ones under the strong test, and with no shift within 0.0 to
          0.2 percentage points of the unweighted band."""),
    ("finding 3: the 0.2-point difference",
     """          against 83.3%).""",
     """          against 83.3%). On medium-replay the difference is only 0.2 percentage points; the rule asks only for
          lower coverage inside."""),
    ("finding 3: why only the weighted band refuses",
     """<dd>The refusal works as designed everywhere.""",
     """<dd>The refusal works as designed everywhere. The weighted band refuses because these steps have zero chance
          in training, so their weight would be a number divided by zero; the unweighted band uses no weights and so
          cannot notice them."""),
    ("finding 4: the rule of the second check",
     """<li><b>Does weighting move the pull enough?</b> On medium-expert and expert,""",
     """<li><b>Does weighting move the pull enough?</b> This check asks only whether weighting moves return by a
          measurable amount, in either direction; whether caution helps at all is the previous check's job. On
          medium-expert and expert,"""),
    ("finding 4: fifth version",
     """Added in version 5 of the specification, it acts where value-pessimism""",
     """Added in the fifth version of the specification, it acts where value-pessimism"""),
    ("finding 4: BCA's own lower edge, and the link to After the stop",
     """barely move together row by row (correlation −0.17 to 0.29). So the prediction may overstate what BCA's own
          lower edge would do.</li>""",
     """barely move together row by row (correlation −0.17 to 0.29). On 5 of the 25 seeds BCA's own penalty also
          failed the check before training that the perfect one passed on every seed (on four of them it did not pull
          the best throttle back below 80%, and on one it did not rise past 80%). This is the point that <a
          href="#beyond-the-data">After the stop</a> made: BCA's band is calibrated and tracks the error at recorded
          commands, where finding 1 measured it, but not at the actor's own commands, where this penalty is read. So the
          prediction may overstate what BCA's own lower edge would do.</li>"""),
    ("finding 4: what it means, with its condition",
     """the actor's own command, given perfect information, can raise it a lot.</dd>""",
     """the actor's own command, given perfect information, can raise it a lot where the learned actor is already good
          (medium-expert +5.8 points, expert +8.5), though it lowered return slightly on the three lower
          qualities.</dd>"""),
    ("explore: column headers",
     """<th class="n">Capped pull: exact weights, each row at its own threshold, minus unweighted</th><th class="n">Lower edge: exact weights, step by step, minus unweighted</th>""",
     """<th class="n">Capped pull: exact weights (each training row at its own threshold) minus unweighted</th><th class="n">Lower edge: exact weights (each step at its own threshold) minus unweighted</th>"""),
    ("explore: note",
     """runs (unweighted; exact weights step by step for the strong test; the same for the mild test, shown only).</p>""",
     """runs (unweighted; exact weights, each step at its own threshold, for the strong test; the same for the mild
      test, shown only).</p>"""),
    ("known shift: anchor", """    <h3>The known shift: how often each current appears</h3>""",
     """    <h3 id="iw-shift">The known shift: how often each current appears</h3>"""),
    ("known shift: the lean's own letter",
     """leans toward weak or strong currents by a set amount λ over the range 0.1 to 1.1:""",
     """leans toward weak or strong currents by a set amount, the <i>lean</i> ξ (the Greek letter xi; λ
      on this page is the critic's scale), over the range 0.1 to 1.1:"""),
    ("removed command: anchor", """    <h3>The removed command: a common, good move</h3>""",
     """    <h3 id="iw-hold">The removed command: a common, good move</h3>"""),
    ("removed command: this test's data", """In the version-5 data the densest-15% removal""",
     """In this test's data the densest-15% removal"""),
    ("static readout: weight range", """<span class="v" id="iw-t-range">0.14 – 7.36</span>""",
     """<span class="v" id="iw-t-range">0.14 – 7.39</span>"""),
    ("static readout: hold demand", """<span class="v" id="iw-ro-d">0.912</span>""",
     """<span class="v" id="iw-ro-d">0.911</span>"""),
    ("static readout: training density", """<span class="v" id="iw-ro-pt">0.509</span>""",
     """<span class="v" id="iw-ro-pt">0.516</span>"""),
    ("static readout: weight", """<span class="v" id="iw-ro-wc">2.71</span>""",
     """<span class="v" id="iw-ro-wc">2.72</span>"""),
    ("why weighting: one-step errors",
     """so that 90% of errors fall inside. If the critic's errors are""",
     """so that 90% of the critic's one-step errors fall inside. If those errors are"""),
    ("why weighting: legend, one-step error", """<dt>Critic's error</dt><dd>How far the critic's one-step prediction misses:""",
     """<dt>Critic's one-step error</dt><dd>How far the critic's one-step prediction misses:"""),
    ("why weighting: legend, band",
     """Its width is set from the calibration set so that 90% of errors fall inside.</dd>""",
     """Its width is set from the calibration set so that 90% of one-step errors fall inside.</dd>"""),
    ("why weighting: legend, coverage",
     """<dt>Coverage</dt><dd>The share of rows whose critic error actually falls inside the band.""",
     """<dt>Coverage</dt><dd>The share of rows whose critic one-step error actually falls inside the band."""),
    ("what ran: thresholds",
     """with every threshold: unweighted, exact weights step by step, and
        estimated weights step by step.</li>""",
     """with every kind of threshold: unweighted, exact weights with each
        step at its own threshold, and estimated weights with each step at its own threshold.</li>"""),
    ("pass marks: fifth version",
     """<p>Fixed in version 5 of the specification before any run,""",
     """<p>Fixed in the fifth version of the weighting test's specification before any run,"""),
    ("pass marks: percentage points in the rule",
     """gives coverage within 2 points of unweighted</td>""",
     """gives coverage within 2 percentage points of unweighted</td>"""),
    ("pass marks: no signs on W4",
     """Strong: +0.1 to +0.8 points from exact. No shift: +0.0 to +0.2 points from unweighted</td>""",
     """Strong: within 0.1 to 0.8 percentage points of exact. No shift: within 0.0 to 0.2 percentage points of unweighted</td>"""),
    ("pass marks: BCA's own penalty's check",
     """with row-by-row correlation −0.17 to 0.29</td>""",
     """with row-by-row correlation −0.17 to 0.29; BCA's own penalty failed the same check before training on 5 of the 25 seeds</td>"""),
    ("note: fifth version", """version 5 uses uniform random commands, as D4RL does.</p>""",
     """the fifth version uses uniform random commands, as D4RL does.</p>"""),
]

# ====================================================================================================== dynamics
DYNAMICS = [
    ("draw condition",
     """which some motion would grow from step to step would have been replaced by the next one.""",
     """which some motion would keep growing over time would have been replaced by the next one."""),
    ("push, not kick", """Give one float a single push, a kick of 1 to
      its speed number,""", """Give one float a single push of 1 to
      its speed number,"""),
    ("thrusters: time window",
     """and every other float by at most 0.018, about 6% as much.""",
     """and, within 20 steps (and still within 100), every other float by at most 0.018, about 6% as much."""),
    ("wave response defined",
     """per unit of wave
      amplitude, float 2 swings out to 2.60 from rest with the coupling and 2.37 without it.""",
     """once the wave has
      set it swinging steadily, float 2's largest distance from rest (the length of its position and speed numbers) is
      2.60 times the wave's amplitude with the coupling and 2.37 times without it."""),
    ("steering over several steps",
     """the four thrusters together can move the floats' ten numbers in any combination of
      directions.""",
     """the four thrusters together can, over several steps, steer the ten float numbers to
      any combination (in a single step they push only the five speed numbers)."""),
    ("why weak: whose default, which properties",
     """where
      0.02 was a default setting. The search that tuned that world's other settings, among them the swings' shrink
      factors, never changed the coupling's size; it only tried four other random draws of the coupling, to check that
      the world's planned properties did not depend on one particular draw.""",
     """where
      0.02 was the default value of the coupling's size in that world's code. The search that tuned that world's other
      settings, among them the swings' shrink factors, never changed the coupling's size. It only tried four other
      random draws of the coupling, to check that the earlier world's design targets did not depend on one particular
      draw. Those targets were computed exactly before any data were made: for example, that its recording controller
      gets at most halfway from doing nothing to the best controller, and that every controller it used stays
      stable."""),
    ("matrix: the one entry that rounds to 0",
     """<td class="n">−0.0044</td><td class="n">0.0026</td><td class="n">−0.0011</td><td class="n">0</td>""",
     """<td class="n">−0.0044</td><td class="n">0.0026</td><td class="n">−0.0011</td><td class="n">0.00001</td>"""),
    ("matrix: rounding note",
     """<td class="n">0.6891</td><td class="n">0.6656</td></tr>
</tbody></table></div>""",
     """<td class="n">0.6891</td><td class="n">0.6656</td></tr>
</tbody></table></div>
        <p class="note">Every entry is rounded to 4 decimal places, except the speed 3 row's speed 2 entry, 0.00001:
          the smallest number of the coupling, which would round to 0.</p>"""),
]

# ====================================================================================================== other sections
OTHER = [
    ("description", "the earlier world's thin data",
     """like the float cost. Its harm checks
        rejected it twice:""",
     """like the float cost. It had five observed
        versions of its physics, one per episode, and in two of them the recording barely moved two of its four
        controls (the counterparts of the four thrusters here), so the data there were thin. Its harm checks
        rejected it twice:"""),
    ("description", "the toy world before it",
     """Full throttle, the end of the throttle range, is such an edge.</p>
    </div>""",
     """Full throttle, the end of the throttle range, is such an edge.</p>
      <p><strong>Before that, a smaller world.</strong> A three-number linear-quadratic toy world (3 state numbers, 2
        throttles, discount 0.95, 50 steps) was retired earlier, because it could not test BCA: its critic errors were
        written by hand instead of learned, it had no thinly recorded data, and even perfectly placed caution changed
        the score by only −1.6 to +0.04 points.</p>
    </div>"""),
    ("harm-checks", "stop rule: the narrow claim",
     """<li>If caution cannot help in any shifted situation, the test stops and shows the wider-range result (doses up
          to 2), so the reader can see whether BCA's dose range is the limit.</li>""",
     """<li>If caution placed by the exact error does not beat the three controls by 3 points in any shifted
          situation (the best-case caution check fails), the test stops and shows the wider-range result (doses up to
          2), so the reader can see whether BCA's dose range is the limit.</li>"""),
    ("harm-checks", "no 'price'",
     """by at least 25% of the median true price of pushing past 80%,""",
     """by at least 25% of the median true loss of value from pushing past 80%,"""),
    ("methods", "forecast: fixed imitation (1)",
     """the design plan built a stand-in
      for the
      learned actor:""",
     """the design plan built a fixed imitation
      of the
      learned actor (a fixed rule, not a learned network):"""),
    ("methods", "forecast: fixed imitation (2)", """It simulated that stand-in exactly on the""",
     """It simulated that fixed imitation exactly on the"""),
    ("methods", "forecast: fixed imitation (3)", """together with the same stand-in with every throttle capped at 80%""",
     """together with the same imitation with every throttle capped at 80%"""),
    ("methods", "forecast: fixed imitation (4)",
     """The forecast harm is the capped stand-in's points minus the
      stand-in's points:""",
     """The forecast harm is the capped imitation's points minus the
      imitation's points:"""),
    ("methods", "forecast: fixed imitation (5)", """On the recorded training states the stand-in crosses 80%""",
     """On the recorded training states the fixed imitation crosses 80%"""),
    ("methods", "forecast: fixed imitation (6)", """so the
      stand-in pushes into the stall again and again.""", """so the
      fixed imitation pushes into the stall again and again."""),
    ("code-names", "fixed imitations 'bend' and 'edge'", """stand-ins <code>'bend'</code> and <code>'edge'</code>""",
     """fixed imitations <code>'bend'</code> and <code>'edge'</code>"""),
]

CSS = """/* pass 3, final fixes */
/* the weighting test's progress strip: all six steps in one row from about 700 px up */
.iw-pipe { max-width: none; grid-template-columns: repeat(auto-fit, minmax(108px, 1fr)); }
/* a status chip that wraps keeps its dot on its first line */
.iw-chip { align-items: flex-start; }
.iw-chip::before { flex: none; margin-top: .45em; margin-top: calc(0.5lh - 4px); }
/* the two claims of After the stop, one above the other at every width */
.claims { grid-template-columns: minmax(0, 1fr); max-width: none; }
/* a group heading of a sideways-scrolling table keeps its words in view */
.subsec table.rows8 tr.sub td[colspan] .grp { position: sticky; left: 10px; }
/* the walkthrough changes sections with a short vertical move, so nothing is ever wider than the pop-out */
.wt-stage.wt-leave-next { transform: translateY(-10px); }
.wt-stage.wt-leave-back { transform: translateY(10px); }
.wt-stage.wt-enter-next { transform: translateY(10px); }
.wt-stage.wt-enter-back { transform: translateY(-10px); }
"""

MU_OLD = "on the same scale as the caution multiplier m. It is μ"
MU_NEW = "on the same scale as the dose m (the caution multiplier on the pull toward the recorded command). It is μ"
BRACKETS_OLD = "with the lowest to highest seed in brackets"
BRACKETS_NEW = "with the range from the lowest to the highest seed after it (in small type below the mean in the table)"


def edit_cols(s):
    head = "  const COLS = "
    i = s.index(head)
    z = s.index(";\n", i)
    raw = s[i + len(head):z]
    cols = json.loads(raw.replace("<\\/", "</"))
    dump = lambda o: json.dumps(o, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    assert dump(cols) == raw, "COLS does not round-trip"

    # Results: the stand-ins' item table is gone; seed ranges shown below; the evaluation bank explained
    res = cols["results"]
    assert len(res) == 12 and set(res[11]["cols"]) == {"Item", "Value"}
    del res[11]
    n = 0
    for e in res:
        for k, v in e["cols"].items():
            if isinstance(v, str) and BRACKETS_OLD in v:
                e["cols"][k] = v.replace(BRACKETS_OLD, BRACKETS_NEW)
                n += v.count(BRACKETS_OLD)
    assert n == 13, f"in brackets: {n}"
    sit = res[4]["cols"]["Situation"]
    assert sit.startswith("Each row is one evaluation bank of 1,000 starts.")
    res[4]["cols"]["Situation"] = sit.replace("Each row is one evaluation bank of 1,000 starts.",
                                              "Each row is one situation, tested on its 1,000 fixed starting states (its "
                                              "evaluation bank).")
    n = 0
    for k, v in res[10]["cols"].items():
        if "on the first 200 starts of the bank" in v:
            res[10]["cols"][k] = v.replace("on the first 200 starts of the bank",
                                           "on the first 200 of the situation's 1,000 fixed starting states")
            n += 1
    assert n == 4, f"first 200 starts: {n}"
    LOG.append("COLS results: item table entry removed; 13 seed-range notes; evaluation bank; first 200 starts (4)")

    # Weighting test
    w = cols["weighting-test"]
    assert len(w) == 4
    d = w[0]["cols"]
    assert d["The data"].startswith("This gives the points of the recording itself, replayed on the strong test.")
    d["The data"] = d["The data"].replace(
        "This gives the points of the recording itself, replayed on the strong test.",
        "This gives the points on the strong test of the controller that made the recording, with this quality's "
        "random share.")
    assert d["What the removal costs"].endswith("how much the missing hold costs the learner.")
    d["What the removal costs"] += (" On the random quality the two logs are almost identical, so its value there is "
                                    "run-to-run noise.")
    old1 = "Capped pull: exact weights, each row at its own threshold, minus unweighted"
    old2 = "Lower edge: exact weights, step by step, minus unweighted"
    c1 = w[1]["cols"]
    assert list(c1) == ["Quality", old1, old2]
    w[1]["cols"] = {
        "Quality": c1["Quality"],
        "Capped pull: exact weights (each training row at its own threshold) minus unweighted":
            "This gives the return on the strong test of BCA with its capped pull and exact weights, each training row "
            "at its own threshold (its own band multiplier), minus that of unweighted BCA, in units of return. Until "
            "the run finishes, the cell says running.",
        "Lower edge: exact weights (each step at its own threshold) minus unweighted":
            "This gives the return on the strong test of the lower edge with exact weights, each step at its own "
            "threshold (its own band multiplier), minus that of the unweighted lower edge, in units of return. Until "
            "the run finishes, the cell says running.",
    }
    c2 = w[2]["cols"]
    assert list(c2)[1] == "λ"
    new2 = {}
    for k, v in c2.items():
        if k == "λ":
            new2["ξ"] = ("The lean ξ (xi) says how far the current's distribution leans toward strong currents. Its "
                         "density is proportional to e to the power ξc, over 0.1 to 1.1. Training uses ξ = −2.")
        else:
            new2[k] = v
    w[2]["cols"] = new2
    assert w[3]["cols"]["Passes when"] == "This is the rule, fixed in version 5 of the specification before any run."
    w[3]["cols"]["Passes when"] = ("This is the rule, fixed in the fifth version of the weighting test's specification "
                                   "before any run.")
    LOG.append("COLS weighting-test: the data column, the removal cost, explore headers, the lean ξ, pass-mark rule")

    # Dynamics: a push, not a kick
    dy = cols["dynamics"][0]["cols"]
    assert "gets a single kick of 1" in dy["Push given to"]
    dy["Push given to"] = dy["Push given to"].replace("gets a single kick of 1", "gets a single push of 1")

    # Methods: the forecast's fixed imitations
    n = 0
    for e in cols["methods"]:
        for k, v in e["cols"].items():
            if isinstance(v, str) and "stand-in" in v:
                v2 = v.replace("that stand-in", "that fixed imitation").replace("the stand-in itself",
                                                                                "the fixed imitation itself")
                assert "stand-in" not in v2, v2
                e["cols"][k] = v2
                n += 1
    assert n == 1, f"methods stand-in entries: {n}"
    LOG.append("COLS dynamics (push) and methods (fixed imitation)")

    out = dump(cols)
    assert "stand-in" not in out and "kick of 1" not in out and BRACKETS_OLD not in out
    return s[:i + len(head)] + out + s[z:]


def main():
    write = "--write" in sys.argv
    assert md5(BACKUP) == V11_MD5, "the v11 backup changed"
    got = md5(PAGE)
    assert got == PRE_MD5, f"the page is not the one this pass started from (md5 {got})"
    os.makedirs(TMP, exist_ok=True)
    shutil.copyfile(PAGE, os.path.join(TMP, "thrusterchain.pre-apply.html"))
    s0 = open(PAGE, encoding="utf-8").read()
    s = s0
    math0 = MATH.findall(s0)

    for what, old, new in RESULTS:
        s = in_section(s, "results", old, new, what)
    # the stand-ins' operator log goes; its plain content now sits in the first claim's box
    a, b = section(s, "results")
    R = s[a:b]
    h = R.index('\n\n    <h3 id="standins-ready">')
    e = R.index("\n  </section>", h)
    cut = R[h:e]
    assert "Do\n      not launch them again" in cut and "<section" not in cut and cut.count("<h3") == 1
    s = s[:a] + R[:h] + R[e:] + s[b:]
    LOG.append(f"results: the stand-ins' operator log removed ({len(cut):,} characters)")

    for what, old, new in WEIGHTING:
        s = in_section(s, "weighting-test", old, new, what)
    # the lean gets its own letter everywhere in the weighting test
    a, b = section(s, "weighting-test")
    W = s[a:b]
    keep = "λ\n      on this page is the critic's scale"
    nl = W.count("λ") - W.count(keep)
    W = W.replace(keep, "\0KEEP\0").replace("λ", "ξ").replace("\0KEEP\0", keep)
    assert nl == 26, f"weighting-test λ: {nl}"   # 27 on the page before, one of them reworded in the edit above
    s = s[:a] + W + s[b:]
    LOG.append(f"weighting-test: the lean λ -> ξ ({nl})")

    for what, old, new in DYNAMICS:
        s = in_section(s, "dynamics", old, new, what)
    for sid, what, old, new in OTHER:
        s = in_section(s, sid, old, new, what)

    # page-wide: chip case, the μ tooltip's name for the dose
    s = once(s, '<span class="tag acc">EXPLORATORY</span>', '<span class="tag acc">exploratory</span>',
             "chips in lower case", 2)
    s = once(s, MU_OLD, MU_NEW, "μ tooltip: the dose", 9)

    # the weighting test's script: the lean's letter in the chart labels
    s = once(s, '"test lean λ (training is −2)"', '"test lean ξ (training is −2)"', "script: axis label")
    s = once(s, "`Test lean λ: ${sgn(l, 2)}", "`Test lean ξ: ${sgn(l, 2)}", "script: hover")
    s = once(s, ", λ = ${l > 0 ?", ", ξ = ${l > 0 ?", "script: legend (named setting)")
    s = once(s, "`Test (λ = ${sgn(l, 2)})`", "`Test (ξ = ${sgn(l, 2)})`", "script: legend (slider)")

    # tooltip data: column descriptions and the glossary of table cells
    s = edit_cols(s)
    s = once(s, """[/stand-ins?\\b/, "The data-only stand-ins are signals computed from the data alone that take the oracle's place.", /'bend'|'edge'/],""",
             """[/data-only penalt(?:y|ies)/, "The data-only penalties are signals computed from the recorded data alone, put where the oracle was."],""",
             "GLOSS: data-only penalties")
    s = once(s, """[/stand-ins 'bend' and 'edge'/, "Here 'bend' and 'edge' are the code's names for the design plan's two stand-ins for the learned actor,""",
             """[/fixed imitations 'bend' and 'edge'/, "Here 'bend' and 'edge' are the code's names for the design plan's two fixed imitations of the learned actor,""",
             "GLOSS: fixed imitations")

    s = once(s, "</style>", CSS + "</style>", "style")

    # ------------------------------------------------------------------ checks
    assert MATH.findall(s) == math0, "a math container changed"
    a, b = section(s, "weighting-test")
    assert s[a:b].count("λ") == 1, "λ left in the weighting test"
    for bad in ("stand-in", "EXPLORATORY", "kick of 1", "pre-registered results in<", "Do\n      not launch",
                "caution cannot help", "11.1 percentage", "+0.1 to +0.8 points", "favourite", "version-history",
                "touch that belief", "touch it"):
        assert bad not in s, f"left on the page: {bad!r}"
    for anchor in ("ats-loss", "ats-bca", "ats-stage1", "ats-deploy", "ats-claims", "iw-shift", "iw-hold"):
        assert s.count(f'id="{anchor}"') == 1 and s.count(f'href="#{anchor}"') >= 1, anchor
    for href in set(re.findall(r'href="#([^"]+)"', s)):
        assert f'id="{href}"' in s, f"link to a missing id: #{href}"

    out = os.path.join(TMP, "applied.html")
    open(out, "w", encoding="utf-8", newline="").write(s)
    if write:
        open(PAGE, "w", encoding="utf-8", newline="").write(s)
    assert md5(BACKUP) == V11_MD5, "the v11 backup changed"
    print(f"{len(LOG)} edits; {len(math0)} math containers unchanged; {len(s):,} bytes; trial copy {out} md5 {md5(out)}")
    print("page written, md5 " + md5(PAGE) if write else "page not written (add --write)")
    for x in LOG:
        print("  -", x)


if __name__ == "__main__":
    main()
