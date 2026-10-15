"""Manual judgements for tt_audit.py, made after reading every authored text, every table's tuples and the
generators. Keys: "id:<record id>" (several ids may be joined with commas) or a substring of the record's title or
text. Values: (type, detail) pairs; type "drop" removes the automatic findings whose detail contains the given text.
"""

NP = "no main verb (a noun phrase or a clause only): "
LEAD = "a noun phrase before ':' with no verb of its own (an implied 'X is'): "

MANUAL = {
    # ---------------------------------------------------------------- sentences the automatic check missed
    # (a noun phrase whose verb sits in a relative clause with no 'that' or 'which': "the push the wave adds")
    "id:tip-014,tip-027": [("fragment", LEAD + "'The rule the training data were drawn from'")],
    "id:tip-015": [("fragment", LEAD + "'Strong current (…), big waves (…) and big starting pushes (…)'")],
    "id:tip-043": [("fragment", NP + "'The current c plus the push w₁ the wave adds in this state, the same on every float.'")],
    "id:tip-060": [("fragment", LEAD + "'The largest push the wave adds to the current or takes away'")],
    "id:tip-071": [("fragment", NP + "'The push it really gives, shown by the dark tick.'")],
    "id:tip-073": [("fragment", LEAD + "'The number of steps one swing of a float takes'")],
    "id:tip-100": [("fragment", NP + "'The largest stretch t steps of the floats' own motion can apply to any list of numbers.'"),
                   ("unclear", "'stretch' is the size (norm) of a matrix power; the tooltip does not say what is stretched or why it matters")],
    "id:tip-101": [("fragment", NP + "'The most a matrix can stretch any list of numbers.'"),
                   ("unclear", "'matrix' and 'stretch' are not explained")],
    "id:tip-104": [("fragment", "entry '∼': no subject: 'Is drawn at random from.'")],
    "id:tip-140": [("fragment", NP + "'The most the rest of the never-ending sum can add, as a multiple of …'")],
    "id:tip-149": [("fragment", NP + "'The part of the command that damps the floats' swinging, a quarter as hard as K would.'")],
    "id:tip-176": [("fragment", LEAD + "'The command the critic rates best at this state'")],
    "id:tip-192": [("fragment", NP + "'The 100 states it visits from the start.'")],
    "id:tip-212": [("fragment", NP + "'The seeds needed, as a whole number.'")],
    "id:tip-213": [("fragment", NP + "'The number of seeds needed to detect a 3-point effect, before rounding.'")],
    "id:tip-224": [("fragment", "no subject (starts with the participle 'Found'): 'Found with one exact simulator step from the training state.'")],
    "The push thruster 1 really gives under the rising-thrust controller": [("fragment", NP + "the first sentence")],
    "The push thruster 1 really gives at that throttle": [("fragment", LEAD + "'The push thruster 1 really gives at that throttle in the stalling world'")],
    "The push the thruster really gives in the stalling world": [("fragment", LEAD + "'The push the thruster really gives in the stalling world'")],
    "The push each thruster really gives with that command": [("fragment", LEAD + "'The push each thruster really gives with that command in the stalling world'")],
    "The lowest value the observation space allows": [("fragment", NP + "'The lowest value the observation space allows for this number (its low bound).'")],
    "The highest value the observation space allows": [("fragment", NP + "'The highest value the observation space allows (its high bound).'")],
    "The range the current's strength c is drawn from": [("fragment", NP + "the first sentence")],
    "The range the wave's amplitude is drawn from": [("fragment", NP + "'The range the wave's amplitude is drawn from, uniformly.'")],
    "The float number the thrusters' pushes are added to": [("fragment", LEAD + "'The float number the thrusters' pushes are added to'")],
    "The weight the textbook rule K gives float": [("fragment", NP + "'The weight the textbook rule K gives float n's … number for the row's thruster.'")],
    "The weight K gives the": [("fragment", NP + "the first sentence ('The weight K gives …')")],
    "The evaluation bank the averages come from": [("fragment", NP + "'The evaluation bank the averages come from, in the stalling world.'")],
    "One of the three parts the recorded episodes are divided into": [("fragment", NP + "the whole text")],
    "of training episode 2 of the recorded log this page uses": [("fragment", NP + "the first sentence ('State n of training episode 2 of the recorded log this page uses …')"),
                                                                ("code", "'design seed 900' is a run label, not explained")],
    "The middle value, over 3,000 kept rows": [("fragment", NP + "the first sentence ('The middle value … of the true price of pushing past 80%: …')")],
    "The other explanation of any harm the test finds": [("fragment", NP + "the whole text")],
    "Of the states past 80%, the share inside the recorded conditions": [
        ("fragment", NP + "'Of the states past 80%, the share inside the recorded conditions.'"),
        ("unclear", "the second sentence starts with the symbol '—' ('— means the controller never pushes past 80%')")],
    "Points lost by going past 80%": [("fragment", NP + "the whole text ('Points lost by going past 80%: …')")],
    "The evaluation bank the forecast is scored on": [("fragment", NP + "the whole text")],
    "The design plan's forecast of the points lost": [("fragment", NP + "the whole text"),
                                                     ("unclear", "'the capped stand-in' and 'the stand-in' are page nicknames that this tooltip does not explain")],
    "The number of seeds a fair BCA test needs": [("fragment", LEAD + "'The number of seeds a fair BCA test needs'")],
    "Points lost by trusting the critic": [("fragment", NP + "the whole text")],
    "One of the nine world checks, which test the world": [("fragment", NP + "the whole text")],
    "The rule the id uses for drawing each start": [("fragment", NP + "the whole text")],
    "The name this page uses.": [("fragment", NP + "the whole text")],
    "A range around the critic's estimate that should contain": [("fragment", NP + "'A range around the critic's estimate that should contain its one-step error.'")],
    "In the weighting test: states where": [("fragment", NP + "'In the weighting test: states where thruster 1's hold demand, …, is above 0.65.'")],
    "The float cost in each settled state under the rising-thrust controller in the stalling world": [("fragment", NP + "the first sentence")],
    "The standard deviation of the bell-shaped (normal) draw": [("fragment", NP + "the first sentence")],
    "The names gym.make accepts": [("fragment", LEAD + "'The names gym.make accepts'")],
    "The file that holds the environment's code": [("fragment", NP + "the whole text")],
    "One setting of TD3+BC, the learner BCA sits on": [("fragment", NP + "the whole text")],
    "The Python type it takes:": [("fragment", LEAD + "'The Python type it takes'")],
    "The ThrusterChainEnv arguments the id sets.": [("fragment", NP + "the whole text")],

    # ---------------------------------------------------------------- unclear, nicknames, symbols left unexplained
    "fingerprinted": [("unclear", "'fingerprinted' (recorded as a hash) is jargon; the four checks are listed as questions inside one long sentence")],
    "Each calibration row counts in proportion": [("unclear", "three definitions chained by two colons in one sentence; 'calibration row' is not explained")],
    "Stall-aware thruster 1": [("unclear", "each line sets a controller, a thruster number and a value side by side with no label "
                                           "('Stall-aware thruster 1 0.34'); the stall-aware line drops the word 'throttle', and "
                                           "'at its 80% limit on 0%' does not say 'of the band's states'")],
    "the step-up rule": [("unclear", "'the step-up rule' is not explained here")],
    "μ = dλκ / 2": [("unclear", "the formula μ = dλκ / 2 uses d, λ and κ without saying what they are")],
    "id:tip-184,tip-185": [("unclear", "two different tooltips share the heading 'The share for a straight critic' and say different things")],
    "1 with none, 1.375 at the mean": [("unclear", "'1 with none' leaves 'none' unexplained (a row with no over-preference)")],
    "as much as the oracle does": [("unclear", "'the oracle' (the penalty built from the critic's exact error) is not explained here")],
    "the new recording controller flips a coin": [("unclear", "'the new recording controller', 'hold demand' and the symbols c, w₁ and w₂ are not explained in this tooltip")],
    "√(p² + v²)": [("unclear", "p and v in √(p² + v²) are not tied to the position and speed numbers in this tooltip")],
    "(picked)": [("fragment", "the four controller lines ('Recording controller: 2.04 (picked)') are label: value pairs with no full stop"),
                 ("unclear", "'(picked)' does not say what picked means (the controller chosen in the panel)")],
    "This is the floats on their own": [("unclear", "grammar: 'This is the floats on their own' (the subject and the verb do not agree)")],
    "data-only stand-ins": [("unclear", "'the data-only stand-ins run' is an internal name, not explained")],
    "One mark of Stage 1": [("unclear", "'Stage 1' and 'the stop' are internal milestones, not explained here")],
    "the early futility look": [("unclear", "'the early futility look' is jargon, not explained")],
    "Meaning: True: the stalling world": [("unclear", "'Meaning: True: the stalling world. False: …' puts two colons in a row, "
                                                      "and 'True' and 'False' are Python values the tuple does not explain")],
    "The names gym.make accepts": [("unclear","'gym.make' is a Python call; the tooltip does not say it creates the environment")],

    # ---------------------------------------------------------------- automatic findings that are wrong
    "the Python library for arrays of numbers": [("drop", "library name: numpy")],
    "float32 and float64 are decimal numbers": [("drop", "programming type name")],
    "bool is True or False, str is text": [("drop", "programming type name")],
    "0.8 tanh(u/0.8)": [("drop", "u/0.8")],
}


def TUPLE_RULES(rec):
    """Findings that hold for every tuple of a table, from reading the table and the generator."""
    out = []
    rid, text = rec["id"], rec.get("text") or ""
    if not rid.startswith("row-"):
        return out
    t = rid[5:7]
    if t in ("07", "08", "11"):
        out.append(("unclear", "every entry carries an abbreviated column name ('pos 1', 'speed 1', …): the row reads as "
                               "11 to 14 'pos n: number' pairs; 'pos' is an abbreviation"))
    if t == "09":
        if "import:" in text:
            out.append(("unclear", "the generator (plain()) collapses the code block's line breaks, so three Python lines "
                                   "and a comment run together: 'import experiments.shift.thruster_chain # registers the 8 "
                                   "ids import gymnasium as gym env = …'"))
        if "float32)" in text:
            out.append(("unclear", "the generator joins the code line and its explanation with only a space "
                                   "('Box(…, float32) 4 numbers, …'), which reads as a broken sentence"))
    if t == "41":
        out.append(("unclear", "internal bookkeeping (hashes, file and folder names, 'the marks look', 'approval file', "
                               "'the user's choice message', 'pins') with no explanation"))
    if t == "36" and "gapped critic" in text:
        out.append(("unclear", "'the gate's actor on the gapped critic' uses two internal nicknames"))
    if "room check" in text.lower() or "the room" in text.lower():
        out.append(("unclear", "'the room check' is a page nickname, not explained in the tuple"))
    if t == "39" and "Futility look" in text:
        out.append(("unclear", "'Futility look' is jargon, not explained in the tuple"))
    if t in ("44", "45", "46"):
        out.append(("code", "inherent: this table lists code names; the tuple repeats them without explanation"))
    if "P_cap" in text:
        out.append(("unclear", "the generator (plain()) writes the subscript of P_cap as '_cap', which reads like a code name"))
    return out


MECHANISMS = [
    {"name": "cut() helper in the cell-tuple builder (removed)",
     "where": "e5_tooltips.js line 51 'const cut = (s, n) => s.length > n ? s.slice(0, n - 1).trimEnd() + \"…\" : s;' and "
              "line 263 'if (isCur) p.append(el(\"b\", null, cut(v, 240))); else p.append(cut(v, 64));'",
     "effect": "every other cell of the row was cut to 63 characters plus '…', the hovered cell to 239 plus '…'",
     "status": "removed by e7_texts.py JS1 ('tuples: the whole value, never cut', 'tuples: no cut helper'); not in the "
               "working copy or in published version 10 (version 9 had no tooltips). No other generator shortens text: "
               "every authored tooltip traces to its source (e5_cols.py, e6_content.py, e7_texts.py or the weighting-test script) and none is a shortened copy of it."},
    {"name": "the tooltip box's height cap (live)",
     "where": "CSS added by e7_texts.py CSS: '.tt { max-height: calc(100vh - 16px); overflow: hidden; }', with place() "
              "pinning a tall box to the top (y = 8 px)",
     "effect": "a tooltip taller than the window minus 16 px is cut off mid-line at the bottom, with no scroll bar (the "
               "box lets every pointer event through) and no sign that text is missing; on phones 100vh is the height "
               "with the browser bar hidden, so with the bar showing the last lines can also sit under it",
     "status": "live in version 10. Measured: the tallest tooltip is the dynamics formula legend, 396 px at 1200 px wide "
               "and 498 px at 390 px wide; tt_probe.py shows it cut after '… for the thrusters and the water alike.' in "
               "a 260-px-tall window (shots/tt-inv-1200x260-dynamics-legend.png). The Action Space 'Stalls' cell tooltip "
               "is 172 px tall, so it loses its last line ('throttle cost keeps rising.') only in a window under about "
               "188 px tall, such as a small embedded preview: the one way version 10 can show a tooltip that stops at "
               "'… while its throttle cost' or '… while its'."},
    {"name": "native SVG titles that come back (live)",
     "where": "the weighting test's drawBars() creates an SVG <title> per bar on every redraw; the e7 drawings script "
              "converts <title> to data-tip only once, at load",
     "effect": "after a lean button or the slider is used, the three bars show the browser's own tooltip (no full stop, "
               "not affected by the Tooltips switch) and lose their data-tip; not a cut, but a tooltip outside the system",
     "status": "live in version 10 (tt_probe.py: before = 3 data-tip, 0 <title>; after clicking 'Strong' = 0 data-tip, 3 <title>)"},
    {"name": "information lost by the cell-text reader plain() (live, not a cut)",
     "where": "shared tooltip script, plain() and tex()",
     "effect": "a <pre> code block loses its line breaks (spec table, 'import' row); a superscript it cannot map is "
               "written as '^(…)' (weighting test table: 'e^(2c)'); a block's text is joined to the next with only a "
               "space ('Box(…, float32) 4 numbers'); a value that ends with a full stop is followed by ', ' "
               "('… every bank., Role: …')",
     "status": "live in version 10; the text is complete but reads as broken"},
]
