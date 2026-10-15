"""Owner comment 34a4973f: "fix the layout of this table, the top left section is too bunched up".

The Stage 1 symbol legend's first row put four symbols (L_actor, lambda, Q_1, mean_i) in one narrow term cell.
Each gets its own row and its own definition. One exact, asserted replacement; a second run finds it applied.
"""
from pathlib import Path

P = Path(__file__).resolve().parent.parent / "thrusterchain.html"
OLD = ('<div><dt><i>L</i><sub>actor</sub>, λ, <i>Q</i><sub>1</sub>, mean<sub><i>i</i></sub></dt><dd>They are as in the '
       'actor loss of the standard learner, twin delayed deep deterministic policy gradient plus behaviour cloning (see '
       'Methods): what the actor lowers, the critic\'s scale, the critic\'s first estimate and the batch average.</dd>'
       '</div>')
NEW = ('<div><dt><i>L</i><sub>actor</sub></dt><dd>It is what the actor lowers during training, its loss. It is built '
       'as in the standard learner, twin delayed deep deterministic policy gradient plus behaviour cloning (see '
       'Methods).</dd></div>'
       '<div><dt>λ</dt><dd>It is the critic\'s scale: a fixed weight divided by the critic\'s average size in the batch, '
       'so the critic\'s value and the pull toward the recorded command are weighed on the same footing.</dd></div>'
       '<div><dt><i>Q</i><sub>1</sub></dt><dd>It is the critic\'s first estimate. The learner trains two critics, and '
       'the actor climbs the first one.</dd></div>'
       '<div><dt>mean<sub><i>i</i></sub></dt><dd>It is the average over the training rows <i>i</i> in the batch.</dd>'
       '</div>')

h = P.read_text(encoding="utf-8")
if NEW in h and OLD not in h:
    print("already applied")
else:
    assert h.count(OLD) == 1, f"expected the old legend row once, found {h.count(OLD)}"
    P.write_text(h.replace(OLD, NEW), encoding="utf-8")
    print("applied")
