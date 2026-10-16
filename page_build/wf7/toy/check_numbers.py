"""Check every number the toy-worlds section quotes against this page's re-runs of the notebooks.

For each entry: the shown text must occur in the section (toy_fragment.html, tags removed), and the source text
must occur in the re-run output of the named notebook (run/out/*.json); the script reports the cell it was found in
and whether the notebook's own stored output (src/*.ipynb, the committed version) has it in the same cell. Numbers
the section computes by hand are recomputed here. Writes run/out/number_checks.md (a table for TOY.md).
Usage: python -X utf8 check_numbers.py"""
import html
import json
import math
import os
import re
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "run", "out")
SRC = os.path.join(HERE, "src")

RUNS = {  # key: (re-run json, stored notebook or None, notebook as named on the page)
    "D": ("discrete.json", "bca_discrete_toy.ipynb", "bca_discrete_toy.ipynb"),
    "DS": ("discrete_sweep.json", None, "bca_discrete_toy.ipynb, sweep on (RUN_SWEEP = True)"),
    "DX": ("extra_discrete.json", None, "bca_discrete_toy.ipynb + one information-only cell"),
    "C": ("channel.json", "bca_channel_toy.ipynb", "bca_channel_toy.ipynb"),
    "M": ("mean_gate.json", "bca_mean_gate_toy.ipynb", "bca_mean_gate_toy.ipynb"),
    "W": ("weights.json", "bca_weights_toy.ipynb", "bca_weights_toy.ipynb"),
    "WX": ("extra_weights.json", None, "bca_weights_toy.ipynb + one information-only cell"),
    "F": ("findings.json", "bca_findings.ipynb", "bca_findings.ipynb (6320224)"),
    "L": ("walkthrough.json", "bca_findings_walkthrough.ipynb", "bca_findings.ipynb, lesson version (2467e29)"),
}


def run_cells(key):
    d = json.load(open(os.path.join(OUT, RUNS[key][0]), encoding="utf-8"))
    out = {}
    for c in d["cells"]:
        out[c["cell"]] = "\n".join((o.get("text") or "") + (o.get("plain") or "") + (o.get("markdown") or "")
                                   for o in c["outputs"])
    return out


def stored_cells(key):
    nb = RUNS[key][1]
    if not nb:
        return {}
    d = json.load(open(os.path.join(SRC, nb), encoding="utf-8"))
    out = {}
    for i, c in enumerate(d["cells"]):
        if c["cell_type"] != "code":
            continue
        parts = []
        for o in c.get("outputs", []) or []:
            if o.get("output_type") == "stream":
                parts.append("".join(o.get("text", [])))
            else:
                dd = o.get("data", {})
                for k in ("text/plain", "text/markdown"):
                    if k in dd:
                        parts.append("".join(dd[k]))
        out[i] = "\n".join(parts)
    return out


def md_cells(key):
    """Markdown cells of the stored notebook (for facts the notebooks state in words, such as kill rules)."""
    nb = RUNS[key][1]
    d = json.load(open(os.path.join(SRC, nb), encoding="utf-8"))
    return {i: "".join(c["source"]) for i, c in enumerate(d["cells"]) if c["cell_type"] == "markdown"}


def frag_text():
    s = open(os.path.join(HERE, "toy_fragment.html"), encoding="utf-8").read()
    s = re.sub(r"<style>.*?</style>", " ", s, flags=re.S)
    s = re.sub(r'<span class="toy-sr">[^<]*</span>', " / ", s)
    s = re.sub(r"<t[dh]\b[^>]*>", " | ", s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s)
    return re.sub(r"\s+", " ", s)


def pat(p):
    return re.compile(p[3:]) if p.startswith("re:") else re.compile(re.escape(p))


# (section, what, shown in the section, source key, text in the source's output)
E = [
    # ---- overview and the shared world
    ("World", "w0 = 35/85", "35 / 85 = 7 / 17 = 0.412", "D", "Known context ratios: [0.41176471 4.33333333]"),
    ("World", "w1 = 65/15", "65 / 15 = 13 / 3 = 4.333", "D", "Known context ratios: [0.41176471 4.33333333]"),
    ("World", "w-bar", "= 0.144 + 2.817 = 2.961", "D", "E_test[w] = 2.960784313725491"),
    ("World", "seed 7 calibration contexts", "264 rows of context 0 and 36 of context 1", "DX", "EXTRA cal contexts seed 7: 264 36"),
    ("World", "weight sums", "264.71² / 720.76", "DX", "re:264\\.705\\d* 720\\.76"),
    ("World", "effective number 97.2", "= 97.2", "D", "Calibration effective sample size (ESS): 97.22"),
    # ---- BCA's tools
    ("Tools", "29 rows", "0.9 28 = 0.0523", "L", "0.0523"),
    ("Tools", "46 tries allow one", "46 tries allow one", "L", "46 tries allow 1"),
    ("Tools", "30 tries allow none", r"re:x\s*\(30\) = 0", "L", "and 30 tries allow 0"),
    ("Tools", "smallest n", "fewer than 29 tries", "C", "Smallest n with any finite threshold: 29"),
    ("Tools", "mean gate g at 29 tries", "g = 0.11", "M", "re:29\\s+0\\.11\\s+0\\.45\\s+0\\.61"),
    # ---- experiment 1
    ("Exp 1", "training counts", "396 training rows", "D", "[[396 107]"),
    ("Exp 1", "critic venture (1,0)", "venture at 1.626", "D", "1.62616822"),
    ("Exp 1", "count scales", "count scale 0.05025", "D", "0.05025189 0.09667365"),
    ("Exp 1", "thresholds", "16.8212", "D", "(1, 16.8212, 16.8212, 16.8212, 14.211, 14.211, 14.211)"),
    ("Exp 1", "weighted first-stage", "6.2714", "D", "(0, 0.0, 0.0, 0.0, 4.3418, 6.2714, 6.2714)"),
    ("Exp 1", "stage-0 scores", "16 of the 300 held-out scores", "DX", "[(0.0, 284), (4.3418, 12), (6.2714, 4)]"),
    ("Exp 1", "break-even", "= 13.49", "DX", "EXTRA break-even at (1,0): 13.4886"),
    ("Exp 1", "BCA widths at (1,0)", "1.000 − 0.845 = 0.155", "DX", "0.8452978"),
    ("Exp 1", "coverage 0.675", "67.5%", "D", "BCA                  0  0.674  0.6750"),
    ("Exp 1", "weighted conformal 0.8375", "83.75%", "D", "weighted conformal   0  0.842  0.8375"),
    ("Exp 1", "weighted BCA 1", "| 100%", "D", "weighted BCA         0  1.000  1.0000"),
    ("Exp 1", "last stage floor 0.9125", "91.25%", "D-md", "0.9125, 0.95625 or 1 at stage 1"),
    ("Exp 1", "no-shift 0.925", "92.5%", "D", "BCA                  0.9250  1.0000"),
    ("Exp 1", "seed 7 harmful no penalty", "−1.515", "D", "no penalty            -1.515   2.990"),
    ("Exp 1", "seed 7 support only", "1.475", "D", "support only           1.475   0.000"),
    ("Exp 1", "seed 7 BCA", "0.175", "D", "BCA                    1.300   0.175"),
    ("Exp 1", "seed 7 harmless support only", "0.650", "D", "support only           1.475   0.650"),
    ("Exp 1", "seed 7 harmless BCA", "0.825", "D", "BCA                    1.300   0.825"),
    ("Exp 1", "seed 7 harmless no penalty", "2.125", "D", "no penalty             2.125   0.000"),
    ("Exp 1", "sweep empirical", "68.8%", "DS", "empirical            0.6880  0.9836  0.005"),
    ("Exp 1", "sweep conformal", "69.0%", "DS", "conformal            0.6904  0.9858  0.005"),
    ("Exp 1", "sweep BCA", "78.1%", "DS", "BCA                  0.7814  0.9996  0.045"),
    ("Exp 1", "sweep weighted empirical", "99.1%", "DS", "weighted empirical   0.9911  0.9219  0.945"),
    ("Exp 1", "sweep weighted conformal", "99.3%", "DS", "weighted conformal   0.9927  0.9228  0.955"),
    ("Exp 1", "sweep weighted BCA", "99.9%", "DS", "weighted BCA         0.9992  0.9578  0.995"),
    ("Exp 1", "sweep no shift", "92.8%", "DS", "empirical            0.9280  0.9602  1.000"),
    ("Exp 1", "sweep no shift BCA", "95.0%", "DS", "BCA                  0.9496  0.9989  1.000"),
    ("Exp 1", "sweep returns", "1.308", "DS", "BCA                    1.308   1.308"),
    ("Exp 1", "sweep weighted conformal return", "1.436", "DS", "weighted conformal     1.436   1.436"),
    ("Exp 1", "sweep weighted BCA return", "1.341", "DS", "weighted BCA           1.341   1.341"),
    ("Exp 1", "sweep no penalty", "−1.520 harmful, 2.120 harmless", "DS", "no penalty            -1.520   2.120"),
    ("Exp 1", "BCA minus support only", "182 of the 200 seeds (mean −0.162", "DS", "BCA minus support only            -0.162 | 1 / 17 / 182"),
    ("Exp 1", "weighted BCA minus support only", "weighted BCA in 144 (−0.129)", "DS", "-0.129 | 2 / 54 / 144"),
    ("Exp 1", "tie seed", "One sweep seed had a tie", "DS", "Seeds with a tie in some method's choice (taken as safe): 1 [(7, ['no penalty', 'support only'])]"),
    ("Exp 1", "refusal 0", "no method refused a single test target", "DS", "Largest exact refusal rate in any seed, method and stage: 0.0"),
    # ---- experiment 2
    ("Exp 2", "fingerprint", "bb9e0325…dbc3", "C", "Kill rules SHA-256: bb9e03252e9c6be5fc30e72d1172a134498198559d30f51d9552fd86a6afdbc3"),
    ("Exp 2", "unsafe shares", "89.4% of outcomes fall below 1.0, in the tail shape 17.9%", "C", "-0.25: 0.894"),
    ("Exp 2", "tail −0.25", "17.9%", "C", "-0.25: 0.179"),
    ("Exp 2", "room", "printed as 0.1215", "C", "Room = 0.1215"),
    ("Exp 2", "room split", "0.00130 (in 2 of the 200 seeds", "L", "plus 0.00130 elsewhere: in 2 of the 200 seeds"),
    ("Exp 2", "test 1", "−2.50 and −2.09 of the room", "C", "A1 width penalty: gain / room = -2.50"),
    ("Exp 2", "test 2", "+0.58 of the room", "C", "gain / room = +0.58; shuffled keeps -4.44 of it"),
    ("Exp 2", "test 3", "0.028 against 0.051", "C", "G2R 0.0282 against G1 0.0508"),
    ("Exp 2", "test 4a", "99.6% agreement", "C", "agreement 0.9961"),
    ("Exp 2", "test 4b", "5.7% against 0.2%", "C", "split conformal 0.057 against BCA 0.002"),
    ("Exp 2", "test 4c", "29.0% against 0.2%", "C", "plug-in mean 0.290 against BCA 0.002"),
    ("Exp 2", "G2 cost", "at 0.060 more regret", "C", "G2's pooled regret minus G5's: +0.0601"),
    ("Exp 2", "seed 0 critic", "the critic values the gap at 1.447 and safe at 1.004", "L", "1.447 - 1.004 = 0.443"),
    ("Exp 2", "seed 0 tail case", "5 of its 30 survivors", "L", "#i : r_i < Q_safe = 5  of  30"),
    ("Exp 2", "case step-1", "75,000 of the 1,000,000 pairs", "L", "75,000 pairs within 0.10; 50,000 needed | venture"),
    ("Exp 2", "case harm test", "75 of 1,000 draws", "L", "75 draws at or below the claimed advantage; 50 needed | venture"),
    ("Exp 2", "case t-test", "upper bound on the average outcome 1.659", "L", "upper bound 1.659 against 1.004 | venture"),
    ("Exp 2", "case plain average", "average outcome 0.827", "L", "r = 0.827 against 1.004 | fall back"),
    ("Exp 2", "case tail", "threshold 6.883", "L", "λ_BCA = 6.883 against 0.443 | fall back"),
    ("Exp 2", "case split", "6.576, above 0.443", "L", "s_(28) = 6.576 against 0.443 | fall back"),
    ("Exp 2", "case binomial", "5 of 30 below 1.004; 30 tries allow 0", "L", "b = 5 against x(n) = 0 | fall back"),
    ("Exp 2", "regret table: tail", "0.1215 | 0.1215 | 0.1215 | 0.0485 | 0.0411 | 0.0390 | 0.0373", "C", "G2 BCA tail certificate          0.1215  0.1215  0.1215  0.0485  0.0411  0.0390  0.0373"),
    ("Exp 2", "regret table: split", "0.1215 | 0.0318 | 0.0265 | 0.0251 | 0.0266 | 0.0224 | 0.0207", "C", "G3 split-conformal certificate   0.1215  0.0318  0.0265  0.0251  0.0266  0.0224  0.0207"),
    ("Exp 2", "regret table: plain", "0.0297 | 0.0275 | 0.0149 | 0.0125 | 0.0102 | 0.0084 | 0.0065", "C", "G5 plug-in mean                  0.0297  0.0275  0.0149  0.0125  0.0102  0.0084  0.0065"),
    ("Exp 2", "regret table: G1", "0.0854 | 0.0693 | 0.0482 | 0.0441 | 0.0397 | 0.0362 | 0.0329", "C", "G1 step-1 certificate            0.0854  0.0693  0.0482  0.0441  0.0397  0.0362  0.0329"),
    ("Exp 2", "regret table: G1 local", "0.0661 | 0.0505 | 0.0296 | 0.0264 | 0.0224 | 0.0191 | 0.0161", "C", "G1 local reference               0.0661  0.0505  0.0296  0.0264  0.0224  0.0191  0.0161"),
    ("Exp 2", "regret table: G2R", "0.0619 | 0.0457 | 0.0256 | 0.0219 | 0.0176 | 0.0139 | 0.0108", "C", "G2R BCA harm test                0.0619  0.0457  0.0256  0.0219  0.0176  0.0139  0.0108"),
    ("Exp 2", "regret table: G6", "0.1600 | 0.0944 | 0.0612 | 0.0443 | 0.0341 | 0.0289 | 0.0208", "C", "G6 mean harm test                0.1600  0.0944  0.0612  0.0443  0.0341  0.0289  0.0208"),
    ("Exp 2", "regret table: A1", "0.4274 | 0.4253 | 0.4250 | 0.4250 | 0.4247 | 0.4247 | 0.4247", "C", "A1 width penalty                 0.4274  0.4253  0.4250  0.4250  0.4247  0.4247  0.4247"),
    ("Exp 2", "regret table: A2", "0.4075 | 0.4075 | 0.4018 | 0.3985 | 0.3872 | 0.3368 | 0.2891", "C", "A2 width BC multiplier           0.4075  0.4075  0.4018  0.3985  0.3872  0.3368  0.2891"),
    ("Exp 2", "unsafe table: G2", "0.0% | 0.0% | 0.0% | 0.3% | 0.3% | 0.2% | 0.2%", "C", "G2 BCA tail certificate           0.000   0.000   0.000   0.003   0.003   0.002   0.002"),
    ("Exp 2", "unsafe table: G4", "0.0% | 0.0% | 0.0% | 0.4% | 0.0% | 0.1% | 0.2%", "C", "G4 exact binomial certificate     0.000   0.000   0.000   0.004   0.000   0.001   0.002"),
    ("Exp 2", "unsafe table: G3", "0.0% | 8.9% | 7.6% | 6.6% | 5.0% | 5.8% | 5.7%", "C", "G3 split-conformal certificate    0.000   0.089   0.076   0.066   0.050   0.058   0.057"),
    ("Exp 2", "unsafe table: G5", "27.2% | 30.6% | 26.1% | 26.8% | 26.5% | 25.7% | 26.0%", "C", "G5 plug-in mean                   0.272   0.306   0.261   0.268   0.265   0.257   0.260"),
    ("Exp 2", "unsafe table: G2R", "39.2% | 34.2% | 27.2% | 26.5% | 23.9% | 22.6% | 19.6%", "C", "G2R BCA harm test                 0.392   0.342   0.272   0.265   0.239   0.226   0.196"),
    ("Exp 2", "unsafe table: G1", "60.2% | 56.8% | 52.0% | 51.5% | 49.1% | 47.7% | 44.8%", "C", "G1 step-1 certificate             0.602   0.568   0.520   0.515   0.491   0.477   0.448"),
    ("Exp 2", "unsafe table: G6", "62.5% | 55.5% | 50.0% | 46.3% | 43.8% | 42.3% | 39.2%", "C", "G6 mean harm test                 0.625   0.555   0.500   0.463   0.438   0.423   0.392"),
    ("Exp 2", "worst world split conformal", "split conformal ventured in 32% to 38% of seeds", "C", "G3 split-conformal certificate  n=5: 0.00  n=10: 0.37  n=20: 0.38  n=30: 0.35  n=45: 0.32  n=60: 0.35  n=100: 0.38"),
    ("Exp 2", "worst world tail", "the BCA tail certificate in at most 3%", "C", "G2 BCA tail certificate         n=5: 0.00  n=10: 0.00  n=20: 0.00  n=30: 0.03"),
    ("Exp 2", "all rules: greedy", "greedy | 0.5505 | −3.53 | 100% | 100%", "C", "greedy                                 0.5505       -3.53             1.000          1.000"),
    ("Exp 2", "all rules: count", "count penalty | 0.1222 | −0.01", "C", "count penalty                          0.1222       -0.01"),
    ("Exp 2", "all rules: host", "host BC actor | 0.4939 | −3.06 | 86.5% | 86.5%", "C", "host BC actor                          0.4939       -3.06             0.865          0.865"),
    ("Exp 2", "all rules: A1", "A1 | 0.4253 | −2.50 | 0.3% | 0.3%", "C", "A1 width penalty                       0.4253       -2.50             0.003          0.003"),
    ("Exp 2", "all rules: A2", "A2 | 0.3755 | −2.09 | 64.9% | 65.3%", "C", "A2 width BC multiplier                 0.3755       -2.09             0.649          0.653"),
    ("Exp 2", "all rules: G1", "G1 | 0.0508 | 0.58 | 51.7% | 42.4%", "C", "G1 step-1 certificate                  0.0508        0.58             0.517          0.424"),
    ("Exp 2", "all rules: G1 local", "G1 local reference | 0.0329 | 0.73 | 24.1% | 42.4%", "C", "G1 local reference                     0.0329        0.73             0.241          0.424"),
    ("Exp 2", "all rules: G2R", "G2R | 0.0282 | 0.77 | 27.6% | 42.4%", "C", "G2R BCA harm test                      0.0282        0.77             0.276          0.424"),
    ("Exp 2", "all rules: G2", "G2 | 0.0758 | 0.38 | 0.2% | 0.2%", "C", "G2 BCA tail certificate                0.0758        0.38             0.002          0.002"),
    ("Exp 2", "all rules: G3", "G3 | 0.0392 | 0.68 | 5.7% | 7.8%", "C", "G3 split-conformal certificate         0.0392        0.68             0.057          0.078"),
    ("Exp 2", "all rules: G4", "G4 | 0.0753 | 0.38 | 0.1% | 0.1%", "C", "G4 exact binomial certificate          0.0753        0.38             0.001          0.001"),
    ("Exp 2", "all rules: G5", "G5 | 0.0157 | 0.87 | 27.0% | 29.0%", "C", "G5 plug-in mean                        0.0157        0.87             0.270          0.290"),
    ("Exp 2", "all rules: G6", "G6 | 0.0634 | 0.48 | 48.5% | 68.1%", "C", "G6 mean harm test                      0.0634        0.48             0.485          0.681"),
    ("Exp 2", "all rules: G1 shuffled", "G1 shuffled | 0.4356 | −2.58 | 89.9% | 92.0%", "C", "G1 shuffled                            0.4356       -2.58             0.899          0.920"),
    ("Exp 2", "all rules: G2R shuffled", "G2R shuffled | 0.3543 | −1.91 | 71.0% | 79.1%", "C", "G2R shuffled                           0.3543       -1.91             0.710          0.791"),
    ("Exp 2", "all rules: G2 shuffled", "G2 shuffled | 0.1211 | 0.00 | 0.4% | 0.5%", "C", "G2 shuffled                            0.1211        0.00             0.004          0.005"),
    ("Exp 2", "oracle 25%", "oracle switch | 0 | 1 | 25.0% | 25.0%", "C", "oracle switch                          0.0000        1.00             0.250          0.250"),
    ("Exp 2", "width coverage", "90.2% and 93.5%", "C", "stage 0: 0.902"),
    ("Exp 2", "step-1 shortfall", "94% to 98% of seeds", "C", "re:G1 step-1 certificate\\s+1\\.00\\s+1\\.00\\s+1\\.00\\s+1\\.00\\s+0\\.98\\s+0\\.98\\s+0\\.94"),
    ("Exp 2", "reference group", "10.7% of them a noisy cell's low outcomes", "C", "over-prediction near 1.5): 0.107"),
    ("Exp 2", "reference swing", "swung from 0.35 to 1.50", "C", "mean 1.04, from 0.35 to 1.50"),
    ("Exp 2", "sampler at 28 and 29", "only 4 of 8 score sets matched", "C", "re: 28\\s+4/8\\s+8/8"),
    # ---- experiment 3
    ("Exp 3", "fingerprint", "50a0f49d…b1d3", "M", "Kill rules fingerprint (SHA-256): 50a0f49d678b2205e2dba3af5ea741d8cbc887679127ccdd8612f2c43f93b1d3"),
    ("Exp 3", "t-test example", "0.95 29 = 0.23", "M", "With 29 tries, 23% of samples contain no loss"),
    ("Exp 3", "mean gate on clean samples", r"re:\(\s*10 / 10\.4\s*\)\s*29 = 0\.32", "M", "puts 32% of its weighted averages below 0"),
    ("Exp 3", "claim 1", "worst harm rate 0.3%", "M", "worst harm rate 0.003; worlds above their allowance: 0 -> ALIVE"),
    ("Exp 3", "claim 2", "17.4% against 0.0% and 0.0%", "M", "good proposals accepted 17.4% against Hoeffding 0.0% and empirical Bernstein 0.0% -> ALIVE"),
    ("Exp 3", "claim 3", "up to 31.3% of harmful proposals", "M", "worst rare-disaster harm rate 0.313 -> ALIVE"),
    ("Exp 3", "claim 4", "0.0567 against 0.1829 and 0.1625", "M", "gain 0.0567 against plain average 0.1829 and shrunk average 0.1625 -> SHELVED"),
    ("Exp 3", "claim 5", "0.0845 against 0.0400", "M", "tuned BCA mean gate gains 0.0845; best tuned rival BCA average gains 0.0400 -> ALIVE"),
    ("Exp 3", "claim 6", "0.0270 against 0.0453; harm 0.000", "M", "gain 0.0270 against 0.0453 deciding alone; worst average harmful share of acceptances 0.000 -> SHELVED"),
    ("Exp 3", "proposals", "Picking the truly best candidate in every situation would gain 0.294", "M", "would gain 0.294"),
    ("Exp 3", "results: no gate", "No gate (the critic alone) | 0.138 | 66% | 100% | 100.0% | 100.0%", "M", "no gate                           0.138        66%           100%      100.0%               100.0%"),
    ("Exp 3", "results: count", "0.065 | 31% | 32% | 43.2% | 12.4%", "M", "count rule                        0.065        31%            32%       43.2%                12.4%"),
    ("Exp 3", "results: plain", "Plain average | 0.158 | 76% | 83% | 41.4% | 36.2%", "M", "plain average                     0.158        76%            83%       41.4%                36.2%"),
    ("Exp 3", "results: shrunk", "Shrunk average | 0.151 | 73% | 69% | 90.0% | 16.4%", "M", "shrunk average                    0.151        73%            69%       90.0%                16.4%"),
    ("Exp 3", "results: cautious", "BCA's cautious average | 0.136 | 65% | 60% | 11.8% | 1.7%", "M", "BCA average                       0.136        65%            60%       11.8%                 1.7%"),
    ("Exp 3", "results: t-test", "t-test | 0.080 | 39% | 44% | 31.3% | 31.3%", "M", "t-test                            0.080        39%            44%       31.3%                31.3%"),
    ("Exp 3", "results: mean gate", "BCA mean gate | 0.045 | 22% | 17% | 0.3% | 0.0%", "M", "BCA mean gate                     0.045        22%            17%        0.3%                 0.0%"),
    ("Exp 3", "results: tail", "BCA tail gate | 0.003 | 2% | 1% | 0.0% | 0.0%", "M", "BCA tail gate                     0.003         2%             1%        0.0%                 0.0%"),
    ("Exp 3", "results: split", "Split-conformal tail | 0.019 | 9% | 10% | 7.6% | 7.6%", "M", "split-conformal tail              0.019         9%            10%        7.6%                 7.6%"),
    ("Exp 3", "results: shared", "BCA mean gate, shared budget | 0.027 | 13% | 10% | 0.0% | 0.0%", "M", "BCA mean gate, shared budget      0.027        13%            10%        0.0%                 0.0%"),
    ("Exp 3", "results: oracle", "0.208 | 100% | 100% | 0.0% | 0.0%", "M", "oracle gate                       0.208       100%           100%        0.0%                 0.0%"),
    ("Exp 3", "caps: mean", "0.030 | 0.061 | 0.084 | 0.103 | 0.137 | 0.137", "M", "BCA mean gate              0.030     0.061     0.084     0.103     0.137     0.137"),
    ("Exp 3", "caps: cautious", "0.000 | 0.021 | 0.040 | 0.098 | 0.136 | 0.136", "M", "BCA average                0.000     0.021     0.040     0.098     0.136     0.136"),
    ("Exp 3", "caps: shrunk", "0.001 | 0.019 | 0.019 | 0.056 | 0.087 | 0.151", "M", "shrunk average             0.001     0.019     0.019     0.056     0.087     0.151"),
    ("Exp 3", "caps: plain", "0.001 | 0.007 | 0.007 | 0.036 | 0.078 | 0.158", "M", "plain average              0.001     0.007     0.007     0.036     0.078     0.158"),
    ("Exp 3", "caps: t-test", "none qualifies | 0.040 | 0.158", "M", "t-test                      none      none      none      none     0.040     0.158"),
    ("Exp 3", "caps: Hoeffding", "0.005 | 0.017 | 0.017 | 0.017 | 0.017 | 0.017", "M", "Hoeffding                  0.005     0.017     0.017     0.017     0.017     0.017"),
    ("Exp 3", "caps: Bernstein", "0.006 | 0.006 | 0.006 | 0.006 | 0.006 | 0.006", "M", "empirical Bernstein        0.006     0.006     0.006     0.006     0.006     0.006"),
    ("Exp 3", "caps: tail", "0.003 | 0.003 | 0.017 | 0.017 | 0.027 | 0.046", "M", "BCA tail gate              0.003     0.003     0.017     0.017     0.027     0.046"),
    ("Exp 3", "caps: split", "0.006 | 0.006 | 0.006 | 0.019 | 0.019 | 0.062", "M", "split-conformal tail       0.006     0.006     0.006     0.019     0.019     0.062"),
    ("Exp 3", "caps: count", "0.000 | 0.000 | 0.000 | 0.011 | 0.011 | 0.138", "M", "count rule                 0.000     0.000     0.000     0.011     0.011     0.138"),
    ("Exp 3", "within cap", "every tuned gate stayed within its cap", "M", "every tuned gate stayed within its cap: True"),
    ("Exp 3", "tuned: mean", "level 0.2 | 0.0864 | 0.0845 | 2.3%", "M", "BCA mean gate                     0.2            0.0864               0.0845                 2.3%"),
    ("Exp 3", "tuned: cautious", "margin 0.5 | 0.0410 | 0.0400 | 3.8%", "M", "BCA average                       0.5            0.0410               0.0400                 3.8%"),
    ("Exp 3", "tuned: shrunk", "margin 0.5 | 0.0165 | 0.0187 | 1.6%", "M", "shrunk average                    0.5            0.0165               0.0187                 1.6%"),
    ("Exp 3", "tuned: Hoeffding", "level 0.5 | 0.0167 | 0.0174 | 0.1%", "M", "Hoeffding                         0.5            0.0167               0.0174                 0.1%"),
    ("Exp 3", "tuned: tail", "share 0.2 | 0.0187 | 0.0172 | 2.9%", "M", "BCA tail gate                     0.2            0.0187               0.0172                 2.9%"),
    ("Exp 3", "tuned: plain", "margin 1.5 | 0.0061 | 0.0070 | 0.8%", "M", "plain average                     1.5            0.0061               0.0070                 0.8%"),
    ("Exp 3", "tuned: Bernstein", "level 0.5 | 0.0061 | 0.0061 | 0.0%", "M", "empirical Bernstein               0.5            0.0061               0.0061                 0.0%"),
    ("Exp 3", "tuned: split", "share 0.05 | 0.0064 | 0.0056 | 0.3%", "M", "split-conformal tail             0.05            0.0064               0.0056                 0.3%"),
    ("Exp 3", "tuned: count", "100 tries | 0.0001 | 0.0003 | 0.2%", "M", "count rule                        100            0.0001               0.0003                 0.2%"),
    ("Exp 3", "good accepted by tries", "1%, 5% and 17% of good proposals, the plain average 71%, 80% and 89%", "M", "BCA mean gate                         1%       5%      17%"),
    ("Exp 3", "good accepted, plain", "71%, 80% and 89%", "M", "plain average                        71%      80%      89%"),
    ("Exp 3", "cautious in hardest worlds", "0.008 to 0.048 per situation", "M", "BCA average                        0.0078    0.0219    0.0437    0.0482"),
    ("Exp 3", "worlds losing return", "each lost return in 4 worlds", "M", "plain average                                 4            -0.0109"),
    # ---- experiment 4
    ("Exp 4", "fingerprint", "e32e04d7…f44e", "W", "Kill rules fingerprint (SHA-256): e32e04d79b52e3a234e2576ef7cddf9c7b667436315b58330fe827b2dafff44e"),
    ("Exp 4", "evidence: none", "None | 1.00 | 1.00 | 29 | 29 | 216", "W", "none             1.00             1.00                    29                  29                    216"),
    ("Exp 4", "evidence: moderate", "Moderate | 0.73 | 0.40 | 39 | 73 | 287", "W", "moderate         0.73             0.40                    39                  73                    287"),
    ("Exp 4", "evidence: strong", "Strong | 0.34 | 0.20 | 83 | 145 | 382", "W", "strong           0.34             0.20                    83                 145                    382"),
    ("Exp 4", "claim 1", "up to 89.9%; 17 unsafe worlds above 7.3%", "W", "worst unsafe-world rate 0.899; unsafe worlds above the allowance (0.073): 17 -> ALIVE"),
    ("Exp 4", "claim 2", "at most 4.3%", "W", "worst unsafe-world rate 0.043; above the allowance: 0 -> ALIVE"),
    ("Exp 4", "claim 3", "33.6% against 19.0% and 0.0%", "W", "deviation rate in safe shifted worlds 0.336, against rejection-sampled 0.190 and stratified 0.000 -> ALIVE"),
    ("Exp 4", "claim 4", "38.8% against 4.3%", "W", "worst unsafe-world rate 0.388 for weighted split conformal, against 0.043 for weighted BCA -> ALIVE"),
    ("Exp 4", "claim 5", "at most 4.4%", "W", "worst harmful-world rate 0.044; above the allowance: 0 -> ALIVE"),
    ("Exp 4", "claim 6", "0.0975 against 0.0923", "W", "tuned weighted BCA mean gate gains 0.0975; best tuned rival weighted split conformal gains 0.0923 -> ALIVE"),
    ("Exp 4", "results: always", "Always deviate | 1.000 | 1.000 | 1.000 | 0.0950", "W", "always deviate                      1.000          1.000          1.000  0.0950"),
    ("Exp 4", "results: tail unweighted", "0.899 | 0.550 | 0.736 | 0.1106", "W", "BCA tail, unweighted                0.899          0.550          0.736  0.1106"),
    ("Exp 4", "results: tail weighted", "0.043 | 0.336 | 0.025 | 0.0838", "W", "BCA tail, weighted                  0.043          0.336          0.025  0.0838"),
    ("Exp 4", "results: split", "0.388 | 0.750 | 0.266 | 0.1423", "W", "weighted split conformal            0.388          0.750          0.266  0.1423"),
    ("Exp 4", "results: rejection", "0.036 | 0.190 | 0.021 | 0.0644", "W", "rejection-sampled exact             0.036          0.190          0.021  0.0644"),
    ("Exp 4", "results: stratified", "0.000 | 0.000 | 0.000 | 0.0023", "W", "stratified exact                    0.000          0.000          0.000  0.0023"),
    ("Exp 4", "results: mean unweighted", "0.986 | 0.647 | 0.936 | 0.1253", "W", "BCA mean, unweighted                0.986          0.647          0.936  0.1253"),
    ("Exp 4", "results: mean weighted", "0.119 | 0.415 | 0.044 | 0.0982", "W", "BCA mean, weighted                  0.119          0.415          0.044  0.0982"),
    ("Exp 4", "results: rejection mean", "0.119 | 0.261 | 0.026 | 0.0789", "W", "rejection-sampled BCA mean          0.119          0.261          0.026  0.0789"),
    ("Exp 4", "results: t-test", "0.440 | 0.741 | 0.381 | 0.1395", "W", "weighted t-test                     0.440          0.741          0.381  0.1395"),
    ("Exp 4", "results: weighted average", "0.730 | 0.927 | 0.549 | 0.1536", "W", "weighted average                    0.730          0.927          0.549  0.1536"),
    ("Exp 4", "results: plain", "1.000 | 0.888 | 1.000 | 0.1461", "W", "plain average                       1.000          0.888          1.000  0.1461"),
    ("Exp 4", "results: oracle", "1.000 | 1.000 | 0.000 | 0.1719", "W", "oracle                              1.000          1.000          0.000  0.1719"),
    ("Exp 4", "world counts", "144 unsafe worlds", "WX", "EXTRA_COUNTS unsafe worlds 144 harmful 108 safe shifted 96 allowance 0.0731"),
    ("Exp 4", "worst unweighted mean", "the mean gate in up to 93.6%", "WX", "EXTRA worst unweighted mean world: 0.93625"),
    ("Exp 4", "effective tries", "averaged 41.8 effective tries of 120", "W", "41.8 /  24.1"),
    ("Exp 4", "w-bar strong", "= 2.9096", "WX", "2.909642857142857"),
    ("Exp 4", "usual below 0", "0.0062", "WX", "\"usual_below\": 0.006209665"),
    ("Exp 4", "rung truth", "0.1056", "WX", "0.105589"),
    ("Exp 4", "rungs: tail unweighted", "0.999 | 0.998 | 0.945 | 0.769 | 0.588 | 0.426 | 0.219 | 0.045", "WX", "\"BCA tail, unweighted\": [0.99875, 0.9975, 0.945, 0.76875, 0.5875, 0.42625, 0.21875, 0.045]"),
    ("Exp 4", "rungs: tail weighted", "0.895 | 0.541 | 0.211 | 0.072 | 0.037 | 0.014 | 0.006 | 0.003", "WX", "\"BCA tail, weighted\": [0.895, 0.54125, 0.21125, 0.0725, 0.0375, 0.01375, 0.00625, 0.0025]"),
    ("Exp 4", "seed 1 tries", "38, 33, 27, 14 and 8", "L", "| 0 (mild) | 38 | 0 | 0.143 | 4.8% |"),
    ("Exp 4", "seed 1 total weight", "= 113.879", "L", "Σ_i w_i = 113.879"),
    ("Exp 4", "seed 1 share", "= 0.849", "L", "1 - (17.250)/(113.879) = 0.849"),
    ("Exp 4", "seed 1 draws", "124 of 1,000", "L", "draws that cross = 124  of  1,000"),
    ("Exp 4", "seed 1 unweighted", "6 ≤ 6: yes, deviate", "L", "b = 6 ≤ x(120) = 6  ⇒  yes: deviate"),
    ("Exp 4", "seed 1 world rates", "deviated in 42.6% and the weighted gate in 1.4%", "L", "the unweighted gate deviated in 0.426 of them and the weighted gate in 0.014"),
    ("Exp 4", "proof-covered version", "worst harmful-world rate 2.6% against 4.4%", "W", "BCA mean, largest weight           0.026          0.273  0.0804          0.1      0.0954                0.055"),
    ("Exp 4", "wbcp check", "in 300 of 300 random cases", "W", "All checks passed"),
    # ---- together
    ("Together", "findings re-run", "508 printed result lines", "F", "All checks passed"),
    ("Together", "findings: 14 of 18", "14 alive and 4 shelved", "F-md", "Fourteen of the eighteen claims survived."),
    ("Exp 4", "wbcp check: 300 cases", "in 300 of 300 random cases", "F-md", "Check 2 compares the weighted tail gate with production `calibration/wbcp.py` on 300 random cases"),
    ("Exp 1", "weighted conformal 191 of 200 (the notebook's text)", "191 of the 200 sweep seeds", "D-md", "it reaches 0.90 in 191 of 200 seeds"),
]

ARITH = [  # (what, shown, value computed here, tolerance)
    ("0.9^28", "0.0523", 0.9 ** 28, 5e-5),
    ("0.9^29", "0.0471", 0.9 ** 29, 5e-5),
    ("0.9^30", "0.0424", 0.9 ** 30, 5e-5),
    ("P(Bin(30,0.1) <= 1)", "0.1837", 0.9 ** 30 + 30 * 0.1 * 0.9 ** 29, 5e-5),
    ("(1/1.11)^29", "0.0485", (1 / 1.11) ** 29, 5e-5),
    ("(1/1.10)^29", "0.0630", (1 / 1.10) ** 29, 5e-5),
    ("tail p at -0.25", "0.179", 1.25 / 7, 5e-4),
    ("room at the gap", "0.12025", 0.65 * (1 + 0.5 + 0.25 + 0.1) / 10, 1e-9),
    ("98.9% of the room", "98.9%", 0.12025 / 0.12155, 5e-4),
    ("ESS seed 7", "97.2", (264 * 7 / 17 + 36 * 13 / 3) ** 2 / (264 * (7 / 17) ** 2 + 36 * (13 / 3) ** 2), 0.05),
    ("context-1 weight share", "59%", 36 * 13 / 3 / (264 * 7 / 17 + 36 * 13 / 3), 0.005),
    ("zero-score weight share", "73.8%", (264 * 7 / 17 + 20 * 13 / 3) / (264 * 7 / 17 + 36 * 13 / 3), 5e-4),
    ("after 12 scores", "93.5%", (264 * 7 / 17 + 32 * 13 / 3) / (264 * 7 / 17 + 36 * 13 / 3), 5e-4),
    ("break-even", "13.49", (1.62616822 - 1.0) / (0.09667365 - 0.05025189), 0.005),
    ("w-bar strong", "2.9096", .05 ** 2 / .35 + .1 ** 2 / .25 + .15 ** 2 / .2 + .3 ** 2 / .12 + .4 ** 2 / .08, 5e-5),
    ("120 / w-bar", "41.2", 120 / 2.909642857142857, 0.05),
    ("allowance", "0.073", 0.05 + 3 * math.sqrt(0.05 * 0.95 / 800), 5e-4),
    ("total weight seed 1", "113.879", 38 / 7 + 33 * 0.4 + 27 * 0.75 + 14 * 2.5 + 8 * 5, 5e-4),
    ("weight share below", "15.1%", 17.25 / (38 / 7 + 33 * 0.4 + 27 * 0.75 + 14 * 2.5 + 8 * 5), 5e-4),
    ("cautious / plain", "86%", 0.136 / 0.158, 0.005),
    ("t-test clean share", "0.23", 0.95 ** 29, 0.005),
    ("mean gate clean share", "0.32", (10 / 10.4) ** 29, 0.005),
    ("rung 0.10 share below", "0.1056", 0.10 + 0.90 * 0.006209665325776159, 5e-5),
    ("rung 0.12 mean", "−0.04", 0.88 * 0.5 - 0.12 * 4, 1e-9),
]


def main():
    frag = frag_text()
    cache_r, cache_s = {}, {}
    rows, bad = [], 0
    for sec, what, shown, key, src in E:
        if key.endswith("-md"):                      # stated in the notebook's own words (a markdown cell)
            base = key[:-3]
            cells = [c for c, t in md_cells(base).items() if pat(src).search(t)]
            ok_frag = bool(pat(shown).search(frag))
            good = ok_frag and bool(cells)
            if not good:
                bad += 1
                print(f"PROBLEM {sec} / {what}: shown {'found' if ok_frag else 'NOT FOUND'}; markdown cell {cells}")
            rows.append((sec, what, shown, RUNS[base][2], f"{cells[0] if cells else None} (text)",
                         "the notebook's text", "ok" if good else "PROBLEM"))
            continue
        if key not in cache_r:
            cache_r[key], cache_s[key] = run_cells(key), stored_cells(key)
        ok_frag = bool(pat(shown).search(frag))
        p = pat(src)
        cells = [c for c, t in cache_r[key].items() if p.search(t)]
        cell = cells[0] if cells else None
        stored = cache_s[key]
        if RUNS[key][1] is None:
            st = "not stored (re-run only)"
        else:
            st = "same" if cell is not None and p.search(stored.get(cell, "")) else "DIFFERS"
        good = ok_frag and cell is not None and st != "DIFFERS"
        if not good:
            bad += 1
            print(f"PROBLEM {sec} / {what}: shown {'found' if ok_frag else 'NOT FOUND'}; source cell {cell}; stored {st}")
        rows.append((sec, what, shown, RUNS[key][2], cell, st, "ok" if good else "PROBLEM"))
    arith = []
    for what, shown, val, tol in ARITH:
        x = float(shown.replace("%", "").replace("−", "-")) / (100 if shown.endswith("%") else 1)
        ok = abs(x - val) <= tol and shown in frag
        if not ok:
            bad += 1
            print(f"PROBLEM arithmetic {what}: shown {shown}, computed {val:.6f}, in section: {shown in frag}")
        arith.append((what, shown, f"{val:.6g}", "ok" if ok else "PROBLEM"))
    md = ["| Part | Number | Shown as | Notebook | Cell (re-run) | Stored output | Check |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append("| " + " | ".join(str(x).replace("|", "\\|") for x in r) + " |")
    md += ["", "| Worked arithmetic | Shown as | Computed | Check |", "|---|---|---|---|"]
    for r in arith:
        md.append("| " + " | ".join(r) + " |")
    open(os.path.join(OUT, "number_checks.md"), "w", encoding="utf-8", newline="\n").write("\n".join(md) + "\n")
    print(f"{len(rows)} quoted numbers and {len(arith)} worked results checked; problems: {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
