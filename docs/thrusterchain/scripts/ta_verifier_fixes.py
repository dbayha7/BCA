"""Tooltip text pass, round 2: the verifier's findings, each checked against the page first.

The texts live in ta_texts.py (hand-written tooltips, legends, column descriptions, renamed columns) and ta_code.py (the
notes under cell tuples, GLOSS; the script snippets). This script edits those two sources with exact, asserted
replacements (each old snippet must occur exactly once; a snippet already replaced is skipped), keeps a copy of both
sources as they were, then rebuilds the page from version 10 with ta_apply.py --redo, which applies every text with its
own exact, asserted replacements and checks that the version 10 backup is untouched.
  wsl.exe -d Ubuntu-22.04 -u root -e bash -lc 'python3 /mnt/c/Users/David/dev/bca-shift-work/page/scripts/ta_verifier_fixes.py'
Add --sources-only to edit the sources without rebuilding the page.
"""
import hashlib
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE_DIR = os.path.dirname(HERE)
SAVE = os.path.join(PAGE_DIR, "tmp", "ta", "src-before-verifier")
V10 = os.path.join(PAGE_DIR, "thrusterchain.v10.html")
V10_MD5 = "adf3837e6d032f77f6c8bf56bf0221a8"

BCA = "Bayesian Conformal Aggregation"
FULLLOG = "(the critic trained on the full log, with nothing removed)"
NORMAL = "(the recording controller's ordinary mode, which never goes past 80%)"
HILL = "(the critic's value drawn as a hill over the possible commands)"

# ------------------------------------------------------------------------------------------------ ta_texts.py
TEXTS = [
    # the comment that lists the records kept as they were: tip-132 now has a new text
    ("comment: tip-132 is no longer kept",
     "tip-092, tip-106, tip-109, tip-110, tip-132, tip-203, tip-222.",
     "tip-092, tip-106, tip-109, tip-110, tip-203, tip-222."),

    # finding 14: the Recorded conditions option button
    ("tip-027: what the actor must not lose, and against what",
     '''"of 0.5 around rest. The learned actor must not lose here."),''',
     '''"of 0.5 around rest. Here the learned actor must not lose points: adding the caution of "
                      f"{BCA} must cost it clearly less than 2 points (the no-loss check)."),'''),

    # finding 7: the discount, finished comparisons and the value of γ
    ("tip-132: γ is 0.98, and the comparison is finished",
     '''    "tip-135": (None, "The return J is the sum''',
     '''    "tip-132": (None, "The discount γ is 0.98: a reward t steps ahead counts 0.98ᵗ times as much as a reward now."),
    "tip-135": (None, "The return J is the sum'''),
    ("tip-138: the comparison is finished",
     '''"tip-138": (None, "This sum adds every future reward, each counted 0.98ᵗ as much, and it is what a learner's critic "
                      "estimates."),''',
     '''"tip-138": (None, "This sum adds every future reward, a reward t steps ahead counted 0.98ᵗ times as much as a "
                      "reward now, and it is what a learner's critic estimates."),'''),
    ("legend tip-133, J_γ: the comparison is finished",
     '''"It is the discounted return, in which each reward counts 0.98<sup><i>t</i></sup> as much.",''',
     '''"It is the discounted return, in which a reward <i>t</i> steps ahead counts 0.98<sup><i>t</i></sup> times as "
        "much as a reward now.",'''),
    ("legend tip-133, γ: the comparison is finished",
     '''"The discount γ = 0.98, multiplied by itself <i>t</i> times, gives how much a reward <i>t</i> steps ahead "
        "counts.",''',
     '''"The discount γ = 0.98, multiplied by itself <i>t</i> times, gives how much a reward <i>t</i> steps ahead "
        "counts compared with a reward now.",'''),
    ("column: recording controller's discounted return, the comparison is finished",
     '''"in which a reward t steps ahead counts 0.98 to the power t as much."),''',
     '''"in which a reward t steps ahead counts 0.98 to the power t times as much as a reward now."),'''),

    # finding 12: what the squashing is
    ("tip-151: the squashing, with its formula",
     '''"tip-151": (None, "The squashing keeps every normal-driving throttle inside ±80%."),''',
     '''"tip-151": (None, "The squashing, 0.8 tanh(u / 0.8), keeps every throttle of normal driving (the recording "
                      "controller's ordinary mode) inside ±80% by bending each command u smoothly; tanh turns any "
                      "number into one between −1 and 1."),'''),

    # finding 9: the hill, normal driving and the full-log critic explained where they are used
    ("tip-176: the hill explained",
     '''"tip-176": (None, "This is the command the critic rates best at this state, the top of its hill."),''',
     f'''"tip-176": (None, "This is the command the critic rates best at this state, the top of its hill {HILL}."),'''),
    ("tip-177: normal driving explained",
     '''"tip-177": (None, "The curvature κ says how steeply the critic's value falls away from its favourite command along "
                      "each throttle. It is 2.36, the median measured on the exact value of normal driving (standing in "
                      "for the critic) at normal driving's own command."),''',
     f'''"tip-177": (None, "The curvature κ says how steeply the critic's value falls away from its favourite command (the "
                      "command it rates best) along each throttle. It is 2.36, the median measured on the exact value "
                      "of normal driving {NORMAL}, standing in for the critic, at normal driving's own command."),'''),
    ("tip-181: no hill left unexplained",
     '''"tip-181": (None, "The critic's pull μ is how hard the critic's hill draws the actor toward its favourite command, "
                      "on the same scale as the caution multiplier m. It is μ = dλκ / 2, where d = 4 is the number of "
                      "throttles, λ the critic's scale and κ the curvature of its hill, and the forecast fixes it at "
                      "0.58."),''',
     '''"tip-181": (None, "The critic's pull μ is how hard the critic draws the actor toward its favourite command (the "
                      "command it rates best), on the same scale as the caution multiplier m. It is μ = dλκ / 2, where "
                      "d = 4 is the number of throttles, λ the critic's scale and κ the curvature of the critic's value "
                      "around that command, and the forecast fixes it at 0.58."),'''),
    ("legend tip-179, a*: the hill explained",
     '''"It is the critic's favourite command, the top of its hill.",''',
     f'''"It is the critic's favourite command, the top of its hill {HILL}.",'''),
    ("legend tip-179, κ: normal driving explained",
     '''"It is the curvature of the critic's hill: 2.36, the median measured on the exact value of normal driving at "
        "its own command.",''',
     f'''"It is the curvature of the critic's hill: 2.36, the median measured on the exact value of normal driving "
        "{NORMAL} at its own command.",'''),
    ("column: normal driving only, explained",
     '''("The average discounted return of normal", "This is the average discounted return of normal driving only."),''',
     '''("The average discounted return of normal", "This is the average discounted return of normal driving only, the "
     "recording controller's normal driving without its wobble and without the full-throttle command."),'''),
    ("column: the design plan's forecast, normal driving explained",
     '''"critic that curves down past 80%. The plan stood in for the learned actor with normal driving moved part of the "
     "way toward''',
     f'''"critic that curves down past 80%. The plan stood in for the learned actor with normal driving {NORMAL} "
     "moved part of the way toward'''),
    ("column: Comparison, the full-log critic explained",
     '''"critic's error divided by the full-log critic's, and the fifth row''',
     f'''"critic's error divided by the error of the full-log critic {FULLLOG}, and the fifth row'''),
    ("column: α 2.5 actor of the full-log critic, explained",
     '''"the actor trained against the full-log critic, at the published critic weight α = 2.5,''',
     f'''"the actor trained against the full-log critic {FULLLOG}, at the published critic weight α = 2.5,'''),
    ("column: α 5 actor of the full-log critic, explained",
     '''"the full-log critic, at critic weight α = 5."),''',
     f'''"the full-log critic {FULLLOG}, at critic weight α = 5."),'''),
    ("column: α 5 loss, the full-log critic explained",
     '''("Points lost by trusting", "This is the number of points lost by trusting the critic trained after the removal: "
     "the full-log critic's actor minus this actor, at α = 5."),''',
     f'''("Points lost by trusting", "This is the number of points lost by trusting the critic trained after the removal: "
     "the points of the actor of the full-log critic {FULLLOG} minus those of this actor, at α = 5."),'''),

    # finding 4: "seed" explained at its first use in each text
    ("tip-209: repeat runs (seeds) at the first use",
     '''"tip-209": (None, "This is the larger of two standard deviations over the harm-check seeds, one for each of two "
                      "paired differences, in points. It sets N, the number of repeat runs (seeds) that the test of "
                      f"{BCA} needs."),''',
     '''"tip-209": (None, "This is the larger of two standard deviations across the harm checks' repeat runs (seeds), one "
                      "for each of two paired differences (each between two runs of the same seed that differ in one "
                      f"thing), in points. It sets N, the number of repeat runs that the test of {BCA} needs."),'''),
    ("tip-212: repeat runs (seeds)",
     '''"tip-212": (None, "This is the number of seeds needed, rounded up to a whole number."),''',
     '''"tip-212": (None, "This is the number of repeat runs (seeds) needed, rounded up to a whole number."),'''),
    ("tip-213: repeat runs (seeds)",
     '''"tip-213": (None, "This is the number of seeds needed to detect a 3-point effect, before rounding."),''',
     '''"tip-213": (None, "This is the number of repeat runs (seeds) needed to detect a 3-point effect, before rounding."),'''),
    ("column: Shape (one data seed)",
     '''"This says how many numbers the field holds in one data seed's log: one per "
     "recorded row (1,000 episodes of 100 steps), times the numbers in each row."),''',
     '''"This says how many numbers the field holds in the log of one recording (one "
     "data seed, which fixes its random draws): one per recorded row (1,000 episodes of 100 steps), times the numbers "
     "in each row."),'''),
    # finding 5 (with finding 4): the σ column, N and the seed-count check explained
    ("column: Spread σ",
     '''("A possible value of σ", "Each row is a possible value of σ (sigma), the larger of the standard deviations, over "
     "the harm-check seeds, of the seed-count check's two paired differences, in points. The seed-count check never "
     "ran, so these rows only show what N each σ would require."),''',
     '''("A possible value of σ", "Each row is a possible value of σ (sigma), in points: the larger standard deviation, "
     "across repeat runs (seeds), of the two differences that the seed-count check would have measured, each between "
     "two runs of the same seed that differ in one thing. The seed-count check, the fourth harm check, sets how many "
     "repeat runs (N) a fair test needs; it never ran, so these rows only show what N each σ would require."),'''),
    ("column: BCA-test seeds N",
     '''f"This is the number of seeds N that a fair test of {BCA} (BCA) "''',
     '''f"This is the number of repeat runs (seeds) N that a fair test of {BCA} (BCA) "'''),
    ("column: Seeds 0–4",
     '''"This gives the lowest to the highest value over harm-check seeds "
     "0–4: an error ratio''',
     '''"This gives the lowest to the highest value over the harm checks' "
     "repeat runs (seeds 0–4): an error ratio'''),
    ("column: Seed",
     '''("The harm-check seed", "This is the harm-check seed, and a seed fixes every random draw of a run."),''',
     '''("The harm-check seed", "This is the harm-check seed, the number of one repeat run; a seed fixes every random "
     "draw of a run."),'''),
    ("columns: points, mean over repeat runs (results[2])",
     '''f"This gives the points, as a mean over seeds 0–4, {p} {d}. {SCALE}")''',
     '''f"This gives the points, as a mean over repeat runs (seeds) 0–4, {p} {d}. {SCALE}")'''),
    ("column: Over strength-matched uniform",
     '''"uniform caution of the same strength, as a mean over seeds 0–4. The rule needs at least +3."),''',
     '''"uniform caution of the same strength, as a mean over repeat runs (seeds) 0–4. The rule needs at least +3."),'''),
    ("column: Seed 900",
     '''("The value on design seed 900", "This is the value on design seed 900, the seed whose recorded log this page "
     "uses; a seed fixes every random draw of a run."),''',
     '''("The value on design seed 900", "This is the value on design seed 900, the seed whose recorded log this page "
     "uses. A seed fixes every random draw of a run, and the design seeds are recordings made only to design and check "
     "the world."),'''),
    ("column: Seeds 900–904",
     '''"This gives the lowest and the highest value over design seeds "
     "900–904, and every check passed on every seed."),''',
     '''"This gives the lowest and the highest value over design seeds "
     "900–904, five recordings made only to design and check the world (a seed fixes every random draw), and every "
     "check passed on every seed."),'''),
    ("column: share of states past 80%, repeat runs",
     '''"thruster is past 80%, as a mean over seeds 0–4 with the lowest to highest seed in brackets."),''',
     '''"thruster is past 80%, as a mean over repeat runs (seeds) 0–4 with the lowest to highest seed in brackets."),'''),
    ("columns: points, mean over repeat runs (results[5])",
     '''f"This gives the points, as a mean over seeds 0–4 with the lowest to highest seed in brackets, {p} {d}. {SCALE}")''',
     '''f"This gives the points, as a mean over repeat runs (seeds) 0–4 with the lowest to highest seed in brackets, "
     f"{p} {d}. {SCALE}")'''),
    ("columns: cost of caution, repeat runs",
     '''f"costs more, as a mean over seeds 0–4 with the lowest to highest seed in brackets, {p} {d}. In this world 100 "''',
     '''f"costs more, as a mean over repeat runs (seeds) 0–4 with the lowest to highest seed in brackets, {p} {d}. In "
     "this world 100 "'''),
    # finding 15 (and 4): the Stage 1 marks table
    ("column: Mark, a pass mark explained",
     '''("One mark of Stage 1", "Each row is one mark of Stage 1, the first exploratory stage''',
     '''("One mark of Stage 1", "Each row is one pass mark (a pass-or-fail rule) of Stage 1, the first exploratory stage'''),
    ("column: the mark's rule, repeat runs",
     '''"This is the mark's rule. Pooled means combined over seeds 0–4, unless the rule names seeds."),''',
     '''"This is the mark's rule. Pooled means combined over repeat runs (seeds) 0–4, unless the "
     "rule names particular seeds."),'''),
    ("columns: change in points, repeat runs",
     '''f"This is the change in points from that actor alone, in the stalling world, as a mean over seeds 0–4 on the "''',
     '''f"This is the change in points from that actor alone, in the stalling world, as a mean over repeat runs (seeds) "
     f"0–4 on the "'''),

    # finding 16: o(x)
    ("legend tip-191, o(x): the observation of a row, not the row",
     '''"It is the critic's extra preference for full throttle at observation <i>x</i>, a kept row in the full-throttle "
        "region.",''',
     '''"It is the critic's extra preference for full throttle at observation <i>x</i>, taken from a kept training row "
        "in the full-throttle region.",'''),

    # finding 10: lower-case column labels renamed, as pos n and speed n were
    ("RENAME: thrusters, current, waves and the import row",
     '''        RENAME[(_sec, _i, f"speed {_n}")] = f"Float {_n} speed"''',
     '''        RENAME[(_sec, _i, f"speed {_n}")] = f"Float {_n} speed"
for _k in range(1, 5):
    RENAME[("dynamics", 2, f"thruster {_k}")] = f"Thruster {_k}"
RENAME[("recording-controller", 1, "current c")] = "Current c"
RENAME[("recording-controller", 1, "wave w₁")] = "Wave w₁"
RENAME[("recording-controller", 1, "wave w₂")] = "Wave w₂"
RENAME[("description", 0, "import")] = "Import lines"'''),
]

# ------------------------------------------------------------------------------------------------ ta_code.py
CODE = [
    # finding 2: BCAShift
    ("GLOSS: BCAShift",
     r'''it adds caution where it judges that the critic may be wrong."],
''',
     r'''it adds caution where it judges that the critic may be wrong."],
    [/BCAShift/, "BCAShift is the prefix of the project's registered environment names; BCA in it stands for Bayesian Conformal Aggregation, the method this page tests."],
'''),
    # finding 17: .md, and capital-letter names that are documents
    ("GLOSS: capital letters skip a .md document's name",
     r'''    [/\b(?!JSON\b)[A-Z][A-Z_]{3,}\b|\bB1\b|H1–H4/,''',
     r'''    [/\b(?!JSON\b)[A-Z][A-Z_]{3,}\b(?!\.md\b)|\bB1\b|H1–H4/,'''),
    ("GLOSS: .md and Physics(stall=...)",
     r'''    [/\b[a-z_]+\.py\b/, "A name ending in .py is a Python program file."],
''',
     r'''    [/\b[a-z_]+\.py\b/, "A name ending in .py is a Python program file."],
    [/\.md\b/, "A name ending in .md is a text document written in Markdown, a plain-text format for documents."],
    [/Physics\(stall=\.\.\.\)/, "Physics(stall=...) is the code's physics settings, made with stall set to True for the stalling world or False for the no-stall comparison world; the three dots stand for that value."],
'''),
    # finding 13: the BCA test's checks, from the page's own definitions
    ("GLOSS: the BCA test's checks and plain",
     r'''    [/\boracle/i,''',
     r'''    [/\bgain(?: check|-check| and placement checks)/, "The gain check asks whether adding Bayesian Conformal Aggregation gains at least 2 points over the same learner without it in a shifted situation where caution can help."],
    [/\bplacement checks?\b/, "The placement check asks whether Bayesian Conformal Aggregation also beats the same caution placed blindly: its own doses spread evenly or shuffled, and uniform caution of the same strength."],
    [/detection checks?/, "The three detection checks ask whether the doses of Bayesian Conformal Aggregation are largest on the training rows where the critic most overrates the actor's command against the recorded one."],
    [/no-loss/, "The no-loss check asks whether adding Bayesian Conformal Aggregation costs clearly less than 2 points in the recorded conditions."],
    [/beats-copying/, "The beats-copying check asks whether, in the recorded conditions, the actor with Bayesian Conformal Aggregation scores at least 5 points above behaviour cloning, which copies the recorded commands."],
    [/no-shift-harm/, "The no-shift-harm check asks whether, on the full log with nothing removed, adding Bayesian Conformal Aggregation costs clearly less than 2 points in the recorded conditions and wherever caution can help."],
    [/harm-present/, "The harm-present check asks whether there is still harm to remove: without Bayesian Conformal Aggregation, the actor must still lose at least 3 points to pushing past 80%."],
    [/\bplain (?:run|TD3|critic|learner)\b|minus plain\b/, "Plain means without Bayesian Conformal Aggregation, so a plain run or a plain critic comes from the standard learner on its own."],
    [/\boracle/i,'''),
    # finding 3: the choice message
    ("GLOSS: the choice message",
     r'''"The user's choice message is the chat message in which the user chose which form of three stand-ins counts toward the marks; it had to hold the specification's first 16 characters."''',
     r'''"The user's choice message is the chat message in which the user chose, for three of the penalties that look only at the recorded data, which of their two forms counts toward the pass marks; it had to contain the first 16 characters of the specification's hash."'''),
    # finding 1: the design plan's stand-ins 'bend' and 'edge'
    ("GLOSS: stand-ins, and the design plan's 'bend' and 'edge'",
     r'''    [/stand-ins?\b/, "The data-only stand-ins are signals computed from the data alone that take the oracle's place."],
''',
     r'''    [/stand-ins?\b/, "The data-only stand-ins are signals computed from the data alone that take the oracle's place.", /'bend'|'edge'/],
    [/stand-ins 'bend' and 'edge'/, "Here 'bend' and 'edge' are the code's names for the design plan's two stand-ins for the learned actor, one for a critic that curves down past 80% and one for a critic that keeps its slope."],
'''),
    # finding 11: one note for the pre-registration and its pins, defined before it is used
    ("GLOSS: pre-registration and pins in one note",
     r'''    [/\bpins?\b/, "A pre-registration pins a file by recording its hash, so that any later change to the file is detected."],
''',
     r'''    [/pre-regist|\bpins?\b/i, "A pre-registration is the record of every rule and pass mark, fixed before any result existed; it pins a file by recording the file's hash, so any later change to that file is detected."],
'''),
    ("GLOSS: the old pre-registration entry goes (merged above)",
     r'''    [/pre-regist/i, "A pre-registration is the record of every rule and pass mark, written down and fixed before any result existed."],
''',
     ""),
    # finding 6: the return-cost check explained
    ("GLOSS: step-up rule, the return-cost check explained",
     r'''reruns the return-cost check at a critic weight of 5''',
     r'''reruns the return-cost check (the harm check that asks whether an actor that trusts the critic loses return) at a critic weight of 5'''),
    # finding 4: seed explained under every tuple that uses it
    ("GLOSS: seed",
     r'''    [/stop rule/,''',
     r'''    [/\bseeds?\b/i, "A seed is the number that fixes every random draw of a run or a recording, so runs with different seeds are repeat runs that differ only by chance."],
    [/stop rule/,'''),
    # finding 8: the weighting bars name the removed cluster
    ("weighting bars: the removed cluster named",
     r'''of test states have a recorded command in the cluster that the removal deleted from training.`);''',
     r'''of test states have a recorded command in the removed cluster: the commands that hold thruster 1 at exactly 80%, which this test deletes from the training data.`);'''),
    # finding 10: a key-value table's row header takes its renamed label too ("import" -> "Import lines")
    ("tuples: a renamed row header of a key-value table",
     '''    ("show(): fit the box before placing it",''',
     '''    ("key-value tables: a row header takes its renamed label",\n'''
     '''     r[Q]        info.heads.set(th, { name: names[i], desc: d ? d.desc : "" });\n[Q],\n'''
     '''     r[Q]        info.heads.set(th, { name: d ? d.name : names[i], desc: d ? d.desc : "" });\n[Q]),\n'''
     '''    ("show(): fit the box before placing it",'''.replace("[Q]", "'" * 3)),
]


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def patch(path, edits):
    s = open(path, encoding="utf-8").read()
    done = skipped = 0
    for what, old, new in edits:
        n_old = s.count(old)
        if (new and s.count(new) == 1) or (not new and n_old == 0):
            skipped += 1                                  # applied by an earlier run (a deletion: its merge is checked above)
            continue
        assert n_old == 1, f"{os.path.basename(path)}: {what}: expected the old text once, found {n_old}: {old[:100]!r}"
        s = s.replace(old, new, 1)
        done += 1
    open(path, "w", encoding="utf-8").write(s)
    print(f"{os.path.basename(path)}: {done} replacements, {skipped} already applied")


def main():
    assert md5(V10) == V10_MD5, "the version 10 backup is not the published version 10"
    os.makedirs(SAVE, exist_ok=True)
    for name in ("ta_texts.py", "ta_code.py"):
        dst = os.path.join(SAVE, name)
        if not os.path.exists(dst):
            shutil.copy2(os.path.join(HERE, name), dst)
    patch(os.path.join(HERE, "ta_texts.py"), TEXTS)
    patch(os.path.join(HERE, "ta_code.py"), CODE)
    if "--sources-only" in sys.argv:
        return
    r = subprocess.run([sys.executable, os.path.join(HERE, "ta_apply.py"), "--redo"], capture_output=True, text=True)
    print(r.stdout, r.stderr, sep="")
    assert r.returncode == 0, "ta_apply.py --redo failed"
    assert md5(V10) == V10_MD5, "the version 10 backup changed"


if __name__ == "__main__":
    main()
