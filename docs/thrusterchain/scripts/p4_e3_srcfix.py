import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
E = [
("hold instead; it keeps TD3+BC's published critic weight (part 2, the settings) and adds a weighted version of\n      BCA's calibration.",
 "hold instead; it keeps TD3+BC's published critic weight (<a href=\"#m-settings\">the settings</a>) and adds a\n      weighted version of BCA's calibration."),
("frozen critic and the seed-count check its runs, and they are the weighting test's learner.</td></tr>",
 "frozen critic and the seed-count check its runs, and the weighting test builds BCA's band on their\n            critics.</td></tr>"),
("The design plan measured the critic's typical size as 20.3 (part 2, how far\n      the actor moves). So at",
 "The design plan measured the critic's typical size as 20.3\n      (<a href=\"#m-how-far\">how far the actor moves</a>, below). So at"),
(r"""    <p class="wex"><b>Worked example.</b> With the design plan's measurements (the critic's typical size 20.3, so
      <span class="nw">⟦s::λ, the critic's scale::\lambda = 2.5 / 20.3⟧;</span> the curvature 2.36), the critic's pull is
      ⟦s::μ, the critic's pull::\mu = 4 \times (2.5 / 20.3) \times 2.36 / 2 = 0.58⟧. With no extra caution""",
 r"""    <p class="wex"><b>Worked example.</b> The design plan measured the critic's pull as
      <span class="nw">⟦s::μ, the critic's pull::\mu = 0.58⟧</span> (how, in the paragraph after the legend). With no extra caution"""),
("""    <p>The band's ⟦T::The band's lower edge::lower edge⟧ is the critic's value minus the band's width.""",
 """    <p>The band's <i>lower edge</i> is the critic's value minus the band's width."""),
("""<dd>It is the critic weight: 2.5 as published, and 5 as chosen by the harm checks.</dd>""",
 """<dd>It is the critic weight: 2.5 as published, and 5 in the main test, where the harm checks' step-up rule chose it; the weighting test keeps 2.5.</dd>"""),
("""<dd>It is the caution multiplier, or dose: 1 in plain TD3+BC, and from 1 up to 1.5 with BCA.</dd>""",
 """<dd>It is the caution multiplier, or dose: 1 in plain twin delayed deep deterministic policy gradient plus behaviour cloning (TD3+BC), and from 1 up to 1.5 with Bayesian Conformal Aggregation (BCA).</dd>"""),
("""<dd>It is the frozen critic of a plain TD3+BC run on the log after the removal.</dd>""",
 """<dd>It is the frozen critic of a run of plain twin delayed deep deterministic policy gradient plus behaviour cloning (TD3+BC) on the log after the removal.</dd>"""),
("""<dd>It is the harm-placed dose in BCA's form: 1 with no over-preference, 1.375 at the mean, and always below BCA's cap of 1.5.</dd>""",
 """<dd>It is the harm-placed dose in the form of Bayesian Conformal Aggregation (BCA): 1 with no over-preference, 1.375 at the mean, and always below the cap of 1.5.</dd>"""),
("""<dd>It is BCA's dose at training row <i>i</i>, from 1 up to 1.5.</dd>""",
 """<dd>It is the dose of Bayesian Conformal Aggregation (BCA) at training row <i>i</i>, from 1 up to 1.5.</dd>"""),
("""<dd>It is the number of repeat runs (seeds) that the test of BCA would use: at least 10, and above 20 the test stops.</dd>""",
 """<dd>It is the number of repeat runs (seeds) that the test of Bayesian Conformal Aggregation (BCA) would use: at least 10, and above 20 the test stops.</dd>"""),
("""A behaviour constraint keeps the learned actor close to the commands in the recording by pulling it toward them. TD3+BC's own weight on its pull toward the recorded commands works this way, and so does the dose of Bayesian Conformal Aggregation (BCA).""",
 """A behaviour constraint keeps the learned actor close to the commands in the recording by pulling it toward them. The weight on that pull in twin delayed deep deterministic policy gradient plus behaviour cloning (TD3+BC) works this way, and so does the dose of Bayesian Conformal Aggregation (BCA)."""),
]
for old, new in E:
    assert s.count(old) == 1, (s.count(old), old[:90])
    s = s.replace(old, new)
open(p, "w", encoding="utf-8", newline="").write(s)
print(len(E), "edits")
