"""Version 14 of the ThrusterChain page: the weighting test's return work of 2026-10-14 (exploratory, each step with its
expectations or pass marks written first; sources in /root/bca-shift/outputs/shift_iw/explore/channel/).

Adds findings D (where the band is read), E (weighting for where it is read) and F (return), updates "What the results
mean", the header summary, the run timeline's text (still six steps: the checks count them), result B's closing line,
the navigation, and adds one column-description entry (COLS) per new table. Every edit is an exact, counted
replacement. No new formulas (nothing to pre-render).

    cd C:/Users/David/dev/bca-shift-work/page && python scripts/p5_return_apply.py
"""
import json
import re
import shutil
from pathlib import Path

PAGE = Path(__file__).resolve().parent.parent / "thrusterchain.html"
s = PAGE.read_text(encoding="utf-8")
assert 'id="iw-where"' not in s, "version 14 is already applied"
shutil.copy(PAGE, PAGE.with_name("thrusterchain.pre-v14.html"))


def rep(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, (n, old[:120])
    s = s.replace(old, new)


def tip(title, text):
    t = text.replace("&", "&amp;").replace('"', "&quot;").replace("'", "&#x27;")
    return f'data-tip-title="{title}" data-tip="{t}"'


# ------------------------------------------------------------------------------------------------ findings D, E, F

D = f"""
    <div class="finding" id="iw-where">
      <p class="f-title">D. Where is BCA's band read, and is it honest there?</p>
      <p class="stepnote"><span class="stepchip" data-step="band">Step 4</span><span class="stepchip" data-step="test">Step 5</span>
        The band at the commands the trained actor gives, not at the recorded ones.</p>
      <p class="status"><span class="tag acc">exploratory</span> Asked by the project owner on 14 October 2026, after
        result B: return should rise too, so which use of the band can change the controller? Expectations were written
        down before this check ran; no training was involved.</p>
      <dl class="kv kv1 pro">
        <dt>Why it was asked</dt><dd>The checks with perfect information already rank the two uses of the band. The
          capped pull gains at most 0.2 points even when its extra pull is placed by the critic's exact error. The lower
          edge, a penalty on the critic's value of the actor's own command, gains 5.8 points on medium-expert and 8.4 on
          expert under the <span class="km" data-sit="strong_current">strong test</span>, and nothing on the three lower
          qualities. So the lower edge is the use with room. Yet BCA's lower edge hardly moved the actor (result B). The
          lower edge reads the band at the actor's own command, while every coverage figure in findings 1 to 3 was
          measured at recorded commands.</dd>
        <dt>What was measured</dt><dd>At every training row kept after the removal: the band at the command of the actor
          trained with no extra caution, against that command's one-step error. The error comes from one exact step of
          the simulator and BCA's own target rule. The strong test's figures reweight the rows by the exact weight of
          their current. A command is "past 80%" when it stalls a thruster.</dd>
        <dt>What was expected (written before running)</dt><dd>(1) At recorded commands the band still covers at least
          85%. (2) At the actor's commands it covers under 90%, and under 50% past 80%. (3) Weighting by the current adds
          under 5 points there, while the band past 80% would have to be more than 3 times wider on the top two
          qualities.</dd>
      </dl>
      <div class="tbl"><table class="grouped stack p5-where">
        <caption>The band at recorded commands and at the actor's own commands, mean over seeds 0–4. Coverage is the
          share of commands whose one-step error lies inside the band (target 90%); at the actor's commands it is
          weighted to the <span class="km" data-sit="strong_current">strong test</span>. "Weighted" is BCA's band with
          exact weights for the strong test's currents.</caption>
        <thead>
          <tr class="grp"><th class="g0"></th><th class="g0"></th><th class="g0"></th>
            <th colspan="2">Actor's commands within 80%</th><th colspan="2">Actor's commands past 80%</th>
            <th colspan="2">What the band would need past 80%</th></tr>
          <tr><th>Data quality</th>
            <th class="n" data-label="Actor past 80% in hard water">Actor past 80% in hard water</th>
            <th class="n" data-label="Coverage at recorded commands">At recorded commands</th>
            <th class="n" data-label="Within 80%: unweighted">Unweighted</th>
            <th class="n" data-label="Within 80%: weighted">Weighted</th>
            <th class="n" data-label="Past 80%: unweighted">Unweighted</th>
            <th class="n" data-label="Past 80%: weighted">Weighted</th>
            <th class="n" data-label="Widening needed for 90%">Widening needed</th>
            <th class="n" data-label="Widening the weighting gives">Weighting gives</th></tr>
        </thead>
        <tbody>
          <tr><td>random</td><td class="n">3.5%</td><td class="n">97.2</td><td class="n">93.2</td><td class="n">97.2</td>
            <td class="n">85.2</td><td class="n">92.5</td><td class="n">1.16×</td><td class="n">1.22×</td></tr>
          <tr><td>medium-replay</td><td class="n">9.8%</td><td class="n">97.0</td><td class="n">93.1</td><td class="n">96.7</td>
            <td class="n">85.9</td><td class="n">92.0</td><td class="n">1.13×</td><td class="n">1.19×</td></tr>
          <tr><td>medium</td><td class="n">17.9%</td><td class="n">97.4</td><td class="n">93.2</td><td class="n">98.0</td>
            <td class="n">81.2</td><td class="n">90.4</td><td class="n">1.34×</td><td class="n">1.34×</td></tr>
          <tr><td>medium-expert</td><td class="n">52.6%</td><td class="n">97.5</td><td class="n">95.0</td><td class="n">99.2</td>
            <td class="n">48.4</td><td class="n">67.8</td><td class="n">3.42×</td><td class="n">1.75×</td></tr>
          <tr><td>expert</td><td class="n">61.8%</td><td class="n">94.7</td><td class="n">95.7</td><td class="n">98.7</td>
            <td class="n">24.7</td><td class="n">34.9</td><td class="n">8.13×</td><td class="n">1.45×</td></tr>
        </tbody>
      </table></div>
      <p class="note">At recorded commands the band is measured on the rows its scale was fitted on, so it covers more
        than 90% there. "Widening needed" is how many times wider the band past 80% would have to be to cover 90% of
        the actor's commands there. "Weighting gives" is the mean factor by which the weighting widens it, nearly the
        same on every row.</p>
      <dl class="kv kv1 pro">
        <dt>What happened</dt><dd><b>On the two qualities where the lower edge has room, the band is honest where it
          was calibrated and not where the lower edge reads it.</b> Within 80% it covers 95% to 96% of the actor's
          commands. But the actor pushes past 80% on more than half of the hard-water rows, and there the critic rates
          the actor's command too high, by 3.4 units of return per step on medium-expert and 1.7 on expert. The band
          past 80% is only 15% to 25% wider than within 80%, and it covers 48% and 25%. On the three lower qualities the
          band is 10 to 18 times wider than on expert data, and those actors rarely go past 80%, so the band is roughly
          honest at their commands.</dd>
        <dt>Expected against what happened</dt><dd>(1) <b>Met</b>: 93% to 98% at recorded commands. (2) <b>Met on
          the top two qualities only.</b> (3) <b>Partly met</b>: the widening needed is above 3 on the top two, but
          the weighting added 6 to 10 points of coverage there, more than the 5 expected, though still far short of
          90% past 80%.</dd>
        <dt>What it means</dt><dd>The lower edge is the right use of the band, but it reads the band where the band was
          never calibrated. Weighting by the current widens the band by nearly the same factor on every row, and the
          checks with perfect information show that scaling caution by 1.5 moves return by at most 0.7 points. To change
          return, the weights must change <i>where</i> the band is wide: they must describe the commands at which it is
          read.</dd>
      </dl>
    </div>

    <div class="finding" id="iw-cmdband">
      <p class="f-title">E. Weighting the band for where it is read</p>
      <p class="stepnote"><span class="stepchip" data-step="band">Step 4</span>
        The same band, with weights for the actor's own commands under the strong test. No training yet.</p>
      <p class="status"><span class="tag acc">exploratory</span> Pass marks written down before the band was built,
        with the project owner's go.</p>
      <dl class="kv kv1 pro">
        <dt>What was built</dt><dd>BCA's band with one change, its weights. Each training row and command gets the exact
          <i {tip("Importance weight", "An importance weight says how much more often a case occurs where the band is used than in the data the band was calibrated on. A weight of 2 means twice as often; an infinite weight means the data never contain such a case, so nothing in them can vouch for it.")}>importance
          weight</i> of the place where the lower edge reads the band: the strong test's currents, and the actor's own
          commands. It is the current's weight from findings 2 and 3, times how likely the actor's command is, divided by
          how likely the recording controller was to give it. The actor's command is spread by BCA's own target noise
          (0.2), so that it has a likelihood. The recording controller's command likelihood is known exactly in this
          world: its normal driving, each episode's random share and the removed hold. Where the recording never gives a
          command, past 80% in an episode with no random commands or inside the removed hold, the weight is infinite and
          the band refuses, by the same rule that refused the removed hold in finding 3. The scale network, the
          calibration rows and their scores are BCA's own, unchanged.</dd>
        <dt>Pass marks (written before running)</dt><dd>(C1) The weights are right: commands drawn from the actor's
          spread, at strong-test rows, are covered or refused at least 88% of the time. (C2) It is honest where the
          lower edge reads: at the actor's own commands, within 80% and past 80%, at least 88%. (C3) It is targeted, not
          blanket: within 80% it refuses at most 10% of the actor's commands. (C4) Its penalty rises past 80% and pulls
          the actor's best throttle back below it, on at least 4 of 5 seeds. C2 to C4 are judged on medium-expert and
          expert.</dd>
      </dl>
      <div class="tbl"><table class="grouped stack p5-cmd">
        <caption>The band weighted for the actor's own commands under the <span class="km" data-sit="strong_current">strong
          test</span>, before any training, mean over seeds 0–4 (the actor trained with no extra caution).</caption>
        <thead>
          <tr class="grp"><th class="g0"></th><th class="g0"></th><th colspan="3">At the actor's own commands</th>
            <th class="g0"></th></tr>
          <tr><th>Quality</th>
            <th class="n" data-label="C1: actor's spread, covered or refused">C1: actor's spread, covered or refused</th>
            <th class="n" data-label="C2: within 80%, covered or refused">C2: within 80%, covered or refused</th>
            <th class="n" data-label="C2: past 80%, refused">C2: past 80%, refused</th>
            <th class="n" data-label="C3: within 80%, refused">C3: within 80%, refused</th>
            <th class="n" data-label="C4: penalty shape passes">C4: penalty shape passes</th></tr>
        </thead>
        <tbody>
          <tr><td>random</td><td class="n">98.8%</td><td class="n">100.0%</td><td class="n">99.9%</td><td class="n">99.2%</td><td class="n">2 of 5</td></tr>
          <tr><td>medium-replay</td><td class="n">99.1%</td><td class="n">99.9%</td><td class="n">100.0%</td><td class="n">89.2%</td><td class="n">1 of 5</td></tr>
          <tr><td>medium</td><td class="n">98.2%</td><td class="n">99.3%</td><td class="n">100.0%</td><td class="n">71.5%</td><td class="n">1 of 5</td></tr>
          <tr><td>medium-expert</td><td class="n">98.5%</td><td class="n">99.5%</td><td class="n">100.0%</td><td class="n">47.6%</td><td class="n">1 of 5</td></tr>
          <tr><td>expert</td><td class="n">96.7%</td><td class="n">98.0%</td><td class="n">100.0%</td><td class="n">25.0%</td><td class="n">0 of 5</td></tr>
        </tbody>
      </table></div>
      <dl class="kv kv1 pro">
        <dt>What happened</dt><dd><b>C1 and C2 met; C3 and C4 missed.</b> The weighted band is honest at the actor's
          commands and refuses every one past 80%, where the unweighted band covered 25% to 48%. But it also refuses 25%
          (expert) and 48% (medium-expert) of the actor's commands within 80%. Some of these are the actor reaching into
          the removed hold (15 and 9 points), as the action gap intends. The rest come from the weights: the actor's
          spread is far narrower than the recorded commands, so the weights fall on few calibration rows. On the three
          lower qualities this refuses 72% to 99%. The penalty failed because version 5's lower edge gives a refused
          command the largest finite threshold, only 1.7 to 3.0 times the usual one, and the same value from 80% up, so
          it does not rise past 80%.</dd>
        <dt>Checked after the fact, not a pass mark</dt><dd>If a refused command earns no more value than the row's
          recorded command, the penalty's shape check passes on 5 of 5 seeds on both qualities. The share of hard-water
          rows whose best throttle lies past 80% falls from 80% to 1.8% on expert, and from 85% to 16.5% on
          medium-expert. The perfect lower edge reaches 34% and 52%.</dd>
      </dl>
    </div>

    <div class="finding" id="iw-return">
      <p class="f-title">F. Does the band, weighted for where it is read, raise return?</p>
      <p class="stepnote"><span class="stepchip" data-step="test">Step 5</span>
        Actors trained with the new lower edge, tested on the <span class="km" data-sit="strong_current">strong test</span>
        and with no shift.</p>
      <p class="status"><span class="tag acc">exploratory</span> Three training runs, each approved by the project owner
        and each with its pass marks frozen, with a fingerprint of the file, before its code was written.</p>
      <dl class="kv kv1 pro">
        <dt>What was run</dt><dd>The lower edge of result B, actor only against the frozen critic, with three changes
          tried in turn. <b>First test</b>: the band of finding E, and a refused command earns no more value than the
          recorded command. <b>Fixed fallback</b>: that "no value credit" only where the weight is truly infinite (the
          data never contain such a command); where the calibration rows are merely too few, version 5's capped
          threshold. <b>Estimated weights</b>: the fixed fallback with nothing known: the current's weight from version
          5's fitted classifier, and the command's weight from a second classifier that tells recorded commands from the
          actor's. Each is compared with version 5's unweighted lower edge, with an untargeted control (the same amount
          of "no value credit" at randomly chosen rows), and with the same weights for the training mix of currents,
          which shows what knowing the test's currents adds. Medium-expert and expert, seeds 0–4.</dd>
        <dt>Pass marks</dt><dd>Against version 5's unweighted lower edge, on the strong test: a gain of at least 1.8
          units of return with a lower bound above 0. The same against the untargeted control. With no shift: no loss
          beyond 1.8 units. Estimated weights must also keep half of the exact weights' gain.</dd>
      </dl>
      <div class="tbl"><table class="grouped stack p5-ret">
        <caption>Points (0 = uniform random commands, 100 = the stall-aware controller), mean over seeds 0–4, on the
          <span class="km" data-sit="strong_current">strong test</span> and with no shift. Every row trains the actor
          only, from the same cloned actor, against the same frozen critic.</caption>
        <thead>
          <tr class="grp"><th class="g0"></th><th colspan="2">medium-expert</th><th colspan="2">expert</th></tr>
          <tr><th>Actor</th>
            <th class="n" data-label="medium-expert, strong test">Strong test</th>
            <th class="n" data-label="medium-expert, no shift">No shift</th>
            <th class="n" data-label="expert, strong test">Strong test</th>
            <th class="n" data-label="expert, no shift">No shift</th></tr>
        </thead>
        <tbody>
          <tr><td>No extra caution</td><td class="n">92.1</td><td class="n">96.1</td><td class="n">90.7</td><td class="n">95.9</td></tr>
          <tr><td>Version 5's lower edge, unweighted</td><td class="n">92.1</td><td class="n">96.1</td><td class="n">90.6</td><td class="n">95.9</td></tr>
          <tr><td>Version 5's lower edge, weighted for the currents</td><td class="n">92.2</td><td class="n">96.1</td><td class="n">90.6</td><td class="n">95.9</td></tr>
          <tr><td>First test: no value credit for every refusal</td><td class="n">82.3</td><td class="n">91.4</td><td class="n">96.8</td><td class="n">98.0</td></tr>
          <tr><td>Fixed fallback, exact weights</td><td class="n">94.1</td><td class="n">96.7</td><td class="n">96.5</td><td class="n">98.0</td></tr>
          <tr><td>Fixed fallback, training mix of currents</td><td class="n">94.1</td><td class="n">96.7</td><td class="n">96.4</td><td class="n">97.9</td></tr>
          <tr><td>Untargeted control of the fixed fallback</td><td class="n">92.2</td><td class="n">96.0</td><td class="n">90.7</td><td class="n">95.9</td></tr>
          <tr><td>Estimated weights</td><td class="n">92.3</td><td class="n">96.0</td><td class="n">93.0</td><td class="n">96.8</td></tr>
          <tr><td>Untargeted control of the estimated weights</td><td class="n">92.2</td><td class="n">96.0</td><td class="n">90.7</td><td class="n">95.9</td></tr>
          <tr><td>Perfect lower edge, for scale</td><td class="n">97.9</td><td class="n">98.0</td><td class="n">99.1</td><td class="n">99.0</td></tr>
        </tbody>
      </table></div>
      <dl class="kv kv1 pro">
        <dt>What happened</dt><dd><b>First test: a large gain on expert, a large loss on medium-expert.</b> On expert
          it gained 78.6 units of return on the strong test (6.2 points, lower bound +72.8), and the untargeted control
          gained nothing. On medium-expert it lost 124.8 units: refusals piled up in the strongest currents and on
          episodes with random commands, and with no value credit a refused row only copies its recorded command, which
          there is random most of the time.</dd>
        <dd><b>Fixed fallback: every pass mark met on both qualities.</b> On the strong test it gains 25.0 units on
          medium-expert (2.0 points, lower bound +22.6) and 74.2 on expert (5.8 points, lower bound +68.3), every seed
          positive: 34% and 69% of the perfect lower edge's gain. The untargeted control gains about 1 unit. With no
          shift, return rises too (+4.5 and +15.6). The same weights for the training mix of currents do as well
          (−0.2 and +0.5 units).</dd>
        <dd><b>Estimated weights: a gain on expert, none on medium-expert.</b> On expert, +29.4 units (2.3 points,
          every seed +25 to +35), 40% of the exact weights' gain, short of the half the pass marks asked. On
          medium-expert +1.5, under the 1.8 bar: the classifier finds only 5% of the past-80% commands the exact rule
          refuses, because which episodes hold random commands cannot be seen from a state and a command.</dd>
        <dt>What it means</dt><dd><b>Weighting with the known test distribution raises return through the lower edge,
          once the weights describe where the band is read.</b> The gain comes from the actor's-command part of the
          weight; the current part, which fixed coverage in findings 2 and 3, adds nothing to return. Here the weighting
          acts as a calibrated check that the data contain a command: it refuses where the recording never goes and
          certifies elsewhere, with no throttle limit given to it. Estimating that support from data works on clean
          expert data and not on mixed data. Each finding rests on one environment and on actors trained against a
          frozen critic.</dd>
      </dl>
    </div>
"""

anchor = """    </div>


    <h3 id="iw-results">What the results mean</h3>"""
rep(anchor, "    </div>\n" + D + "\n\n    <h3 id=\"iw-results\">What the results mean</h3>")

# ------------------------------------------------------------------------------------------------ result B's closing line

rep("""or aimed at the wrong thing, the recorded commands, when those are poor (the capped pull).</dd>""",
    """or aimed at the wrong thing, the recorded commands, when those are poor (the capped pull). Findings D to F
          below show what changes when the band is read and weighted where the actor acts.</dd>""")

# ------------------------------------------------------------------------------------------------ What the results mean

old3_start = """      <li><b>Weighting did not raise return</b> (finding 4 and <a href="#iw-arms">exploratory result B</a>)."""
i = s.find(old3_start)
assert i > 0 and s.count(old3_start) == 1
j = s.find("</li>", i) + len("</li>")
old3 = s[i:j]
new3 = """      <li><b>Weighting raises return once the band is weighted for where it is read</b> (exploratory results B,
        <a href="#iw-where">D</a>, <a href="#iw-cmdband">E</a> and <a href="#iw-return">F</a>). As BCA uses its band, weighting
        did not raise return (finding 4 and result B): the band is calibrated at recorded commands, and its penalty at
        the actor's own commands was 1/30 to 1/160 of the exact one. Weighted for the actor's commands under the strong
        test, the band becomes honest there and refuses every command past 80% that the data lack. With a refused
        command earning no more value than the recorded one, return on the strong test rises by 2.0 points on
        medium-expert and 5.8 on expert, on every seed, with no cost without the shift; an untargeted control gains
        nothing. The gain comes from the command part of the weight, not the current part. With weights estimated from
        data it holds on expert (2.3 points) and vanishes on medium-expert.</li>"""
s = s[:i] + new3 + s[j:]
rep("""    <p>This is the main test's lesson again, on different data: BCA's band is calibrated where the data are, and too
      small to matter where the actor goes beyond them.</p>""",
    """    <p>This is the main test's lesson again, on different data, with its answer: BCA's band is calibrated where the
      data are and too small to matter where the actor goes beyond them, until it is weighted for the commands where it
      is read.</p>""")

# ------------------------------------------------------------------------------------------------ header summary

hi = s.find('<li><span class="tag">Weighting test: exploratory</span> Weighting did not raise the')
assert hi > 0
hj = s.find("</li>", hi) + len("</li>")
old_h = s[hi:hj]
ret_tip = re.search(r'<i data-tip-title="Return" data-tip="[^"]*">return</i>', old_h).group(0)
new_h = ("""<li><span class="tag">Weighting test: exploratory</span> As BCA uses its band, weighting did not raise the """
         + ret_tip + """: the band's caution at the actor's own commands is far too small. Weighted instead for the
        actor's own commands, where its penalty is read, the band refuses the commands the data lack, and return on the
        strong test rose by 2.0 points (medium-expert) and 5.8 points (expert) on every seed, against nothing for an
        untargeted control; with estimated weights, 2.3 points on expert and none on medium-expert
        (<a href="#iw-return">Exploratory result F</a>).</li>""")
s = s[:hi] + new_h + s[hj:]

# ------------------------------------------------------------------------------------------------ timeline (six steps kept)

rep("""finished at 18:10 UTC the same day and showed no return gain from weighting
      (<a href="#iw-explore">Exploratory results</a>).</p>""",
    """finished at 18:10 UTC the same day and showed no return gain from weighting
      (<a href="#iw-explore">Exploratory results</a>). On 14 October, at the project owner's request and with pass
      marks written first each time, the band was read and weighted at the actor's own commands; trained that way, the
      actor's return rose on medium-expert and expert (<a href="#iw-return">result F</a>).</p>""")
rep("""<span class="t">Exploratory results</span><span class="s">shown below, outside the pre-registered claims</span>""",
    """<span class="t">Band read where the actor acts</span><span class="s">done: return rises on medium-expert and expert (results D–F)</span>""")

# ------------------------------------------------------------------------------------------------ navigation

rep("""  <a href="#weighting-test">Weighting test</a>
  <a href="#iw-results" class="sub">What the results mean</a>
""",
    """  <a href="#weighting-test">Weighting test</a>
  <a href="#iw-return" class="sub">Return from weighting</a>
  <a href="#iw-results" class="sub">What the results mean</a>
""")

# ------------------------------------------------------------------------------------------------ column descriptions

m = re.search(r"const COLS = (\{.*?\});\n", s, flags=re.S)
COLS_SRC = m.group(1)
COLS = json.loads(COLS_SRC)
assert json.dumps(COLS, ensure_ascii=False, separators=(",", ":")) == COLS_SRC, "COLS does not round-trip"
WHERE = {"cols": {
    "Data quality": "Each row is one data quality of the weighting test, from random to expert.",
    "Actor's commands within 80%": "These two columns are measured at the actor's commands that keep every thruster at or below 80%.",
    "Actor's commands past 80%": "These two columns are measured at the actor's commands that push a thruster past 80%, where it stalls.",
    "What the band would need past 80%": "These two columns compare how much wider the band past 80% would have to be with how much the weighting widens it.",
    "Actor past 80% in hard water": "This is the share of hard-water training rows at which the actor trained with no extra caution pushes a thruster past 80%.",
    "Coverage at recorded commands": "This is the band's coverage at the training rows' recorded commands, with no shift. These are the rows its scale was fitted on, so it covers more than 90%.",
    "Within 80%: unweighted": "Within 80%: the share of the actor's commands whose one-step error lies inside BCA's unweighted band, weighted to the strong test.",
    "Within 80%: weighted": "Within 80%: the same share for BCA's band with exact weights for the strong test's currents.",
    "Past 80%: unweighted": "Past 80%: the share of the actor's commands whose one-step error lies inside BCA's unweighted band, weighted to the strong test.",
    "Past 80%: weighted": "Past 80%: the same share for BCA's band with exact weights for the strong test's currents.",
    "Widening needed for 90%": "This is how many times wider the band past 80% would have to be to cover 90% of the actor's commands there.",
    "Widening the weighting gives": "This is the mean factor by which weighting for the strong test's currents widens the band, nearly the same on every row."}}
CMD = {"cols": {
    "Quality": "Each row is one data quality of the weighting test, from random to expert.",
    "At the actor's own commands": "These three columns are measured at the commands of the actor trained with no extra caution.",
    "C1: actor's spread, covered or refused": "Commands drawn from the actor's command plus BCA's target noise, at strong-test rows: the share the weighted band covers or refuses. The pass mark is 88%.",
    "C2: within 80%, covered or refused": "At the actor's own commands within 80%: the share covered or refused. The pass mark is 88%.",
    "C2: past 80%, refused": "At the actor's own commands past 80%: the share the weighted band refuses, because the recording never gives such a command there.",
    "C3: within 80%, refused": "At the actor's own commands within 80%: the share refused. The pass mark asks for at most 10%, so that refusal stays targeted.",
    "C4: penalty shape passes": "The number of seeds on which the lower-edge penalty from this band rises past 80% and pulls the actor's best throttle back below it. The pass mark asks for 4 of 5."}}
RET = {"cols": {
    "Actor": "Each row is one way of training the actor, all from the same cloned actor against the same frozen critic.",
    "medium-expert": "These two columns are the medium-expert data quality.",
    "expert": "These two columns are the expert data quality.",
    "medium-expert, strong test": "Points of medium-expert actors on the strong test, whose currents lean strong. 0 is uniform random commands and 100 the stall-aware controller.",
    "medium-expert, no shift": "Points of medium-expert actors with no shift, on starts drawn like the training data.",
    "expert, strong test": "Points of expert actors on the strong test, whose currents lean strong. 0 is uniform random commands and 100 the stall-aware controller.",
    "expert, no shift": "Points of expert actors with no shift, on starts drawn like the training data.",
    "Strong test": "Points on the strong test, whose currents lean strong.",
    "No shift": "Points with no shift, on starts drawn like the training data."}}
COLS["weighting-test"] = COLS["weighting-test"] + [WHERE, CMD, RET]
NEW_SRC = json.dumps(COLS, ensure_ascii=False, separators=(",", ":"))
assert s.count(COLS_SRC) == 1
s = s.replace(COLS_SRC, NEW_SRC, 1)

PAGE.write_text(s, encoding="utf-8")
print("version 14 written:", len(s), "characters")
