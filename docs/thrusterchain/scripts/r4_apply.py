"""Edit 4 of 4 (comment D and the version-5 results): replace Results with the newcomer rewrite and update the weighting
test with the version-5 results. Exact, asserted replacements only.

  wsl.exe -d Ubuntu-22.04 -u root -e bash -lc 'python3 /mnt/c/Users/David/dev/bca-shift-work/page/scripts/r4_apply.py'
writes the trial copy tmp/r4/applied.html; add --write to write thrusterchain.html as well.

Sources:
  scripts/r4_results_src.html    the new <section id="results"> with placeholders:
      {{KEEP:name}}               a block cut from the current Results (named below), with its listed label edits
      {{SW|title|tex}}            a symbol: the page's own math for tex, inside the tooltip span the page already gives
                                  the symbol titled `title` ({{SWn|...}}: the same, kept out of the Tab order)
  scripts/r4_weighting_src.html  fragments for <section id="weighting-test"> between <!-- FRAG:name --> and <!-- /FRAG -->
  scripts/r4_cols.py             the column descriptions (COLS) of both sections, also recorded in ta_texts.py
The v11 backup is only read (md5 checked before and after). The page must be the one edit 3 left (md5 checked).
"""
import hashlib
import html as H
import json
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mjx  # noqa: E402
import r4_cols  # noqa: E402

PAGE_DIR = "/mnt/c/Users/David/dev/bca-shift-work/page"
PAGE = os.path.join(PAGE_DIR, "thrusterchain.html")
BACKUP = os.path.join(PAGE_DIR, "thrusterchain.v11.html")
TMP = os.path.join(PAGE_DIR, "tmp", "r4")
V11_MD5 = "79f00557ba8803625b9f36bc497f7cc6"
PRE_MD5 = "d892f25b755a31f17b3d230e76cb35e6"   # the page after edit 3
LOG = []


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def once(s, old, new, what, count=1):
    n = s.count(old)
    assert n == count, f"{what}: expected {count} match(es), found {n}: {old[:140]!r}"
    LOG.append(what)
    return s.replace(old, new)


def cut(s, start, end, what, include_end=True):
    """The block from the one occurrence of `start` to the first `end` after it."""
    assert s.count(start) == 1, f"{what}: start marker found {s.count(start)} times"
    a = s.index(start)
    b = s.index(end, a)
    return s[a:b + (len(end) if include_end else 0)]


def esc_tips(block):
    """Apostrophes inside tooltip attributes written as the page writes them (&#x27;)."""
    return re.sub(r'(data-tip(?:-title)?=")([^"]*)(")', lambda m: m.group(1) + m.group(2).replace("'", "&#x27;") + m.group(3),
                  block)


def main():
    write = "--write" in sys.argv
    assert md5(BACKUP) == V11_MD5, "the v11 backup changed"
    got = md5(PAGE)
    assert got == PRE_MD5, f"the page is not the one edit 3 left (md5 {got})"
    os.makedirs(TMP, exist_ok=True)
    shutil.copyfile(PAGE, os.path.join(TMP, "thrusterchain.pre-edit4.html"))
    s0 = open(PAGE, encoding="utf-8").read()
    s = s0

    # ------------------------------------------------------------------ Results: the kept blocks
    a = s.index('<section id="results">')
    b = s.index("</section>", a) + len("</section>")
    assert b < s.index('<section id="weighting-test">')
    R = s[a:b]
    K = {}

    t = cut(R, "    <ul>\n      <li>15:07: the code's automatic tests passed.", "    </ul>\n", "timeline")
    t = once(t, "<b>the best-case caution check found no situation where caution can help</b>, so the seed-count check",
             "<b>the best-case caution check failed</b>: in no shifted situation did caution placed by the exact error beat\n"
             "        the same caution placed without that knowledge by the required 3 points, so the seed-count check",
             "timeline: 20:39")
    K["timeline"] = t

    t = cut(R, "    <h4>Stage 1: can caution reach", "    <h4>What Stage 1 found</h4>", "stage1", include_end=False)
    t = once(t, "<p>The actor's loss (see <a href=\"#methods\">Methods</a>) has two parts:",
             "<p>The actor's training (see <a href=\"#methods\">Methods</a>) has two parts:", "stage1: training")
    t = once(t, "The room check tested the pull. Stage 1\n      tests the value: every penalty arm",
             "The best-case caution check tested the pull. Stage 1\n      tests the value, the place where value-pessimism"
             " methods such as conservative Q-learning (CQL) act: every penalty arm", "stage1: tests the value")
    t = once(t, "Everything else is the gate's actor-only run on the gapped critic, unchanged: the frozen gapped critic,\n"
                "      the same cloned start",
             "Everything else is the return-cost check's actor-only run on the critic trained after the removal, unchanged:\n"
             "      the same frozen critic, the same cloned start", "stage1: everything else")
    t = once(t, "the same rows in the\n      same order, gate seeds 0–4.</p>",
             "the same rows in the\n      same order, seeds 0–4. Each variant of this training is called an <i>arm</i> of the"
             " comparison.</p>", "stage1: seeds")
    t = once(t, "</span> the room\n      check used:", "</span> the best-case\n      caution check used:", "stage1: strength")
    t = once(t, "the caution the room check tried;", "the caution the best-case caution check tried;", "stage1: tried")
    t = once(t, '<p class="f-title">Why the error is measured from the recorded command (a change from the plan, made before any\n        training)</p>',
             '<p class="f-title">Why the error is measured from the recorded command (a change from the plan, made before any\n        training)</p>',
             "stage1: finding title kept")
    K["stage1"] = t

    old_row = "No extra caution (the gate's actor on the gapped critic)"
    new_row = "No extra caution (the return-cost check's actor on this critic)"
    t = cut(R, '    <figure class="chartbox drbox" id="beyond-chart">', "    </figure>\n", "chart")
    t = once(t, old_row, new_row, "chart: row name")
    t = once(t, H.escape(old_row, quote=True), H.escape(new_row, quote=True), "chart: row label and tip", count=2)
    K["chart"] = t

    t = cut(R, '<div class="tbl"><table class="rows8" style="min-width:760px">', "</tbody></table></div>\n", "stall table")
    t = once(t, "Stalling world, against the gate's frozen gapped critic. Points: mean over seeds 0–4, lowest to highest seed below.",
             "Stalling world, against the frozen critic trained after the removal. Points (0 = recording controller, 100 ="
             " stall-aware controller): mean over seeds 0–4, lowest to highest seed below.", "stall table: caption")
    t = once(t, old_row, new_row, "stall table: row")
    K["stalltable"] = t

    t = cut(R, '<div class="tbl"><table class="rows8" style="min-width:640px">\n<caption><span class="cap">No-stall',
            "</tbody></table></div>\n", "no-stall table")
    K["nostalltable"] = t

    t = cut(R, '<div class="tbl"><table class="rows8" style="min-width:640px">\n<caption><span class="cap">Replication on the room check',
            "</tbody></table></div>\n", "replication table")
    t = once(t, "Replication on the room check's critic (the frozen critic of a plain TD3+BC run on the log after the removal). Points: mean",
             "Replication on the best-case caution check's critic (the frozen critic of a plain TD3+BC run on the log after"
             " the removal). Points (0 = recording controller, 100 = stall-aware controller): mean", "replication: caption")
    t = once(t, "No extra caution (the room check's own actor)", "No extra caution (the best-case caution check's own actor)",
             "replication: row 1")
    t = once(t, "Room check: caution placed at the harm (reused)", "Best-case caution check: harm-placed (reused)", "replication: row 4")
    t = once(t, "Room check: even caution matched in strength (reused)",
             "Best-case caution check: strength-matched uniform (reused)", "replication: row 5")
    t = once(t, "Room check: caution placed at the harm with doses up to 2 (reused)",
             "Best-case caution check: wider range, doses up to 2 (reused)", "replication: row 6")
    K["reptable"] = t

    t = cut(R, '<div class="tbl"><table class="stack" style="min-width:720px">\n<thead><tr><th scope="col">Mark</th>',
            "</tbody></table></div>\n", "marks table")
    t = once(t, "<td>Room</td>", "<td>Beats untargeted caution</td>", "marks: row name")
    t = once(t, "(half of the 98.9-point harm the gate measured)", "(half of the 98.9-point harm the harm checks measured)",
             "marks: recovery")
    K["markstable"] = t

    t0 = R.index("    <h4>At deployment, with no training")
    t1 = R.rindex("  </section>")
    t = R[t0:t1]
    t = once(t, "This diagnostic used the gate's saved actors,", "This diagnostic used the harm checks' saved actors,", "tail: saved actors")
    t = once(t, "the gapped critic's actor, in strong current)", "the actor of the critic trained after the removal, in strong current)",
             "tail: harm")
    t = once(t, "Change in points from the gapped critic's actor alone,",
             "Change in points from the actor of the critic trained after the removal, alone,", "tail: caption")
    n = len(re.findall(r"The gapped data are", t))
    assert n == 1, n
    t = t.replace("The gapped data are", "The data after the removal are")
    t = once(t, "Outside the gate.", "Outside the pre-registered test.", "tail: stand-ins status")
    t = once(t, "<i>captures the room</i>", "<i>recovers enough of the harm</i>", "tail: stand-ins mark name")
    t, n = re.subn(r"\b([Tt])he(\s+)user\b", r"\1he\2project owner", t)
    assert n == 12, f"tail: {n} mentions of the user"
    LOG.append(f"tail: the user -> the project owner ({n})")
    for w in (r"\buser\b", r"\bgate\b", r"gapped", r"room check", r"power check", r"recorder\b"):
        assert not re.search(w, re.sub(r"<mjx-container.*?</mjx-container>", "", t, flags=re.S)), f"tail still has {w}"
    K["tail"] = t

    # ------------------------------------------------------------------ Results: assemble the new section
    src = open(os.path.join(HERE, "r4_results_src.html"), encoding="utf-8").read()
    names = re.findall(r"\{\{KEEP:(\w+)\}\}", src)
    assert sorted(names) == sorted(K), (names, sorted(K))
    for k in names:
        src = once(src, "{{KEEP:%s}}\n" % k, K[k], "keep " + k)
    new_results = esc_tips(src.rstrip("\n"))
    s = s[:a] + new_results + s[b:]
    LOG.append("Results replaced")

    # ------------------------------------------------------------------ the weighting test
    W0 = s.index('<section id="weighting-test">')
    W1 = s.index("</section>", W0)
    frag = dict(re.findall(r"<!-- FRAG:(\w+) -->\n(.*?)<!-- /FRAG -->", open(os.path.join(HERE, "r4_weighting_src.html"),
                                                                             encoding="utf-8").read(), flags=re.S))
    assert sorted(frag) == ["claims", "marks", "pipe", "ran", "results", "status"], sorted(frag)
    frag = {k: esc_tips(v) for k, v in frag.items()}
    sec = s[W0:W1]

    def span(sec, start, end, what):
        assert sec.count(start) == 1, f"{what}: start {sec.count(start)}"
        i = sec.index(start)
        j = sec.index(end, i)
        return i, j

    i, j = span(sec, '    <p class="status"><span class="tag acc">being built</span>', '    <div class="iw-claims">', "status")
    assert "The stop above did not end the work" in sec[i:j]
    sec = sec[:i] + frag["status"] + sec[j:]
    i, j = span(sec, '    <div class="iw-claims">', '    <div class="iw-pipe"', "claims")
    sec = sec[:i] + frag["claims"] + sec[j:]
    i, j = span(sec, '    <div class="iw-pipe"', '    <h3>The known shift: how often each current appears</h3>', "pipe")
    sec = sec[:i] + frag["pipe"] + "\n" + frag["results"] + "\n" + sec[j:]
    sec = once(sec, "The densest-15% removal takes out 99.7–99.8% of the holds.",
               "In the version-5 data the densest-15% removal took out 99.57–99.79% of the holds (lowest to highest seed).",
               "weighting: holds removed")
    i, j = span(sec, '      <p class="note">The real test uses a finite calibration set', "</p>\n", "coverage note")
    sec = (sec[:i] + '      <p class="note">The test itself used a finite calibration set, so weighted coverage scattered from'
           ' seed to seed (pooled, it was 91.0% to 94.6% under the strong test), and it used estimated weights as well as exact'
           ' ones. The weight ω(<i>c</i>) assumes that training and test differ only in the current. The removal breaks that'
           ' for one command: the 80% hold occurs in test episodes but has zero chance in the training data after the removal,'
           ' so its ratio of test to training probability is a number divided by zero, which has no limit. No band can be'
           ' calibrated for a test state with that command, so the weighted band must decline to give it one, and it did: it'
           ' refused 100% of the test steps that use the removed hold and 0% of the others (<a href="#iw-results">What the'
           ' results mean</a>). That refusal is BCA correctly flagging the missing common command.' + sec[j:])
    LOG.append("weighting: coverage note")
    sec = once(sec, "The real test measures the critic's actual errors.", "The test itself measured the critic's actual errors.",
               "weighting: chart legend, past tense")
    i, j = span(sec, "    <h3>What runs, step by step</h3>", "    <h3>Pass marks</h3>", "what runs")
    sec = sec[:i] + frag["ran"] + "\n" + sec[j:]
    i = sec.index("    <h3>Pass marks</h3>")
    assert sec.count("    <h3>Pass marks</h3>") == 1
    tail = sec[i:]
    assert tail.rstrip().endswith("</p>") and "Reference\n      returns for 0 / 100 points" in tail
    sec = sec[:i] + frag["marks"] + "  "
    s = s[:W0] + sec + s[W1:]
    LOG.append("weighting test updated")

    # ------------------------------------------------------------------ header, rail
    s = once(s, "The third did not: caution of the kind BCA applies cannot win back that lost return, in any situation, even when"
                " placed exactly where the critic is wrong.",
             "The third did not: a harder pull toward the recorded commands, at BCA's strengths, could not win back the return"
             " lost in strong current, even when it was placed by the critic's exact error.", "header: outcome")
    s = once(s, '<p class="status"><span class="tag acc">next</span> The project owner chose not to end the work at this stop.'
                ' The next experiment on this environment is the weighting test,',
             '<p class="status"><span class="tag acc">main line</span> The project owner chose not to end the work at this'
             ' stop. The main line of work on this environment is now the weighting test,', "header: main line")
    s = once(s, "The question is whether that improves on unweighted BCA.</p>",
             "The question is whether that improves on unweighted BCA. Its pre-registered part has run: BCA's mechanics work"
             " on all five data qualities tested, and weighting restores the band's coverage under the shift. Whether"
             " weighting also raises return could not be tested under its rules, and BCA's training runs are now running as"
             " an exploratory run.</p>", "header: results in")
    s = once(s, '  <a href="#weighting-test">Weighting test</a>\n',
             '  <a href="#weighting-test">Weighting test</a>\n  <a href="#iw-results" class="sub">What the results mean</a>\n',
             "rail: what the results mean")

    # ------------------------------------------------------------------ tooltip script: column descriptions and notes
    head = "  const COLS = "
    i = s.index(head)
    z = s.index(";\n", i)
    raw = s[i + len(head):z]
    cols = json.loads(raw.replace("<\\/", "</"))
    dump = lambda o: json.dumps(o, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    assert dump(cols) == raw, "COLS does not round-trip"
    old_res = cols["results"]
    assert len(old_res) == 11 and "Comparison" in old_res[0]["cols"] and "Item" in old_res[10]["cols"]
    cols["results"] = r4_cols.results(old_res)
    cols["weighting-test"] = r4_cols.weighting(cols["weighting-test"])
    s = once(s, head + raw + ";\n", head + dump(cols) + ";\n", "COLS")
    g0 = s.index("const GLOSS = [")
    g1 = s.index("];", g0)
    gl = s[g0:g1]
    gl2, n = re.subn(r"\b([Tt])he user\b", r"\1he project owner", gl)
    assert n == 5, f"GLOSS: {n} mentions of the user"
    s = s[:g0] + gl2 + s[g1:]
    LOG.append(f"GLOSS: the user -> the project owner ({n})")

    # ------------------------------------------------------------------ style
    css = ("/* edit 4: problem, measurement and meaning lists in one column; headings inside a section */\n"
           ".kv.kv1 { grid-template-columns: minmax(0, 1fr); gap: 2px; }\n"
           ".kv.kv1 dd { margin-bottom: 10px; }\n"
           "section > h4 { font-family: var(--display); font-size: 17.5px; line-height: 1.3; font-weight: 600; margin: 8px 0 0;"
           " color: var(--ink); text-wrap: balance; letter-spacing: -0.005em; }\n"
           "/* a long status chip of the weighting test wraps instead of widening the page */\n"
           ".iw-claims .iw-chip, #weighting-test td .iw-chip { white-space: normal; }\n"
           "@media (max-width: 560px) {\n"
           "  /* a card table keeps its caption across the full width */\n"
           "  table.stack caption { display: block; }\n"
           "}\n")
    s = once(s, "</style>", css + "</style>", "style")

    # ------------------------------------------------------------------ math: symbols copied from the page, fresh ids
    def sym(m):
        nonlocal s
        focus_no, title, tex = m.group(1) == "SWn", m.group(2), m.group(3)
        et = H.escape(title, quote=True)
        mt = re.search(r'<span class="sym" data-tip-title="%s" data-tip="([^"]*)"' % re.escape(et), s0)
        assert mt, f"no symbol span titled {title!r}"
        spans = mjx.find(s0, tex, display=False)
        assert spans, f"no inline container for {tex!r}"
        cont = s0[spans[0][0]:spans[0][1]]
        return ('<span class="sym" data-tip-title="%s" data-tip="%s"%s>' % (et, mt.group(1), ' data-tip-focus="no"' if focus_no else "")
                + "\0" + cont + "\0</span>")

    s = re.sub(r"\{\{(SWn?)\|([^|}]+)\|([^}]+)\}\}", sym, s)
    # retag every copied container (marked by NULs) so that its glyph ids are new on the page
    while "\0" in s:
        p = s.index("\0")
        q = s.index("\0", p + 1)
        s = s[:p] + mjx.retag(s[p + 1:q], s, prefix="MJXD") + s[q + 1:]
        LOG.append("symbol copied")
    assert "{{" not in s, s[s.index("{{") - 100:s.index("{{") + 100]
    nglyph = mjx.check(s)

    out = os.path.join(TMP, "applied.html")
    open(out, "w", encoding="utf-8", newline="").write(s)
    if write:
        open(PAGE, "w", encoding="utf-8", newline="").write(s)
    assert md5(BACKUP) == V11_MD5, "the v11 backup changed"
    print(f"{len(LOG)} edits; {nglyph} glyph ids, all unique; {len(s):,} bytes; trial copy {out} md5 {md5(out)}")
    print("page written, md5 " + md5(PAGE) if write else "page not written (add --write)")
    for x in LOG:
        print("  -", x)


if __name__ == "__main__":
    main()
