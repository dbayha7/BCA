"""Edit 1, part A: the new reading order (comments 2, 3 and 4), with no change to any moved block's text.

usage: python3 e1_moves.py IN.html OUT.html     (IN and OUT may be the same file; OUT may never be the backup)

1. header: the walkthrough bar after the h1; the glance box and both status lines wrapped in div#glance
2. the live panel (figure#sim) moves up, right after the header
3. the quick facts (div#spec) move into Description, after its parts list
4. Description moves down, after Episode End, and is renamed "Summary of the environment" (id kept)
5. Starting State moves up, after Observation Space
6. Reference Scores moves up, after The recording controller
7. the contents rail follows the new order
8. id="task-situations" on The task's "How each situation changes the challenge"
9. the sentences that described the old positions in words
Checks: every moved block keeps its text; ids are unique; every href="#x" resolves; the data line is unchanged.
"""
import re
import sys

from e1_common import (BACKUP, cut, data_line_hash, insert_after_section, once, replace_once, section_span,
                       strip_text, sub_ws)

ORDER = ["sim", "task", "action-space", "observation-space", "starting-state", "dynamics", "rewards", "episode-end",
         "description", "recording-controller", "reference-scores", "dataset", "shift", "methods", "harm-checks",
         "bca-test", "results", "weighting-test", "arguments", "version-history", "code-names"]


def main():
    src, dst = sys.argv[1], sys.argv[2]
    assert not dst.endswith("thrusterchain.v5.html"), "never modify the backup"
    s = open(src, encoding="utf-8").read()
    assert 'id="glance"' not in s and 'id="wt-open"' not in s, "the structure edit is already applied"
    data0 = data_line_hash(s)
    ids0 = re.findall(r'\bid="([^"]+)"', s)

    # texts of the blocks that move, before
    def block_text(s_, sid):
        a, b = section_span(s_, sid)
        return strip_text(s_[a:b])
    before = {sid: block_text(s, sid) for sid in ["description", "starting-state", "reference-scores"]}
    sim_a = once(s, '    <figure class="panel" id="sim"', "live panel")
    sim_b = s.index("\n    </figure>\n", sim_a) + len("\n    </figure>\n")
    assert s[sim_a:sim_b].count("<figure") == 1
    before["sim"] = strip_text(s[sim_a:sim_b])
    spec_a = once(s, '  <div class="specblock" id="spec">', "quick facts")
    spec_b = once(s, '  <section id="description">', "description start")
    before["spec"] = strip_text(s[spec_a:spec_b])

    # 1. header
    s = replace_once(s, "    <h1>ThrusterChain</h1>\n",
                     "    <h1>ThrusterChain</h1>\n"
                     '    <div class="wt-bar" hidden>\n'
                     '      <button type="button" class="btn wt-open" id="wt-open" aria-haspopup="dialog" '
                     'aria-controls="wt">Full walkthrough mode</button>\n'
                     '      <span class="note">Read the page one section at a time, in order, with Back and Next '
                     'buttons. The normal page stays as it is.</span>\n'
                     "    </div>\n", "h1")
    g0 = once(s, '    <div class="glance">', "glance box")
    g1 = once(s, "  </header>\n", "header end")
    assert g0 < g1
    inner = s[g0:g1]
    s = s[:g0] + '    <div class="head-body" id="glance" data-wt-title="At a glance">\n' + inner + "    </div>\n" + s[g1:]

    # 2. the live panel, right after the header
    sim_a = once(s, '    <figure class="panel" id="sim"', "live panel")
    sim_b = s.index("\n    </figure>\n", sim_a) + len("\n    </figure>\n")
    s, sim = cut(s, sim_a, sim_b)
    s = replace_once(s, "  </header>\n\n  <section id=\"task\">", "  </header>\n\n" + sim + "\n  <section id=\"task\">",
                     "header to task")

    # 3. the quick facts into Description, after its parts list
    spec_a = once(s, '  <div class="specblock" id="spec">', "quick facts")
    spec_b = once(s, '  <section id="description">', "description start")
    spec = s[spec_a:spec_b].rstrip("\n") + "\n"
    s = s[:spec_a] + s[spec_b:]
    da, db = section_span(s, "description")
    s = replace_once(s, "    </ul>\n    <div class=\"callout\">", "    </ul>\n" + spec + "    <div class=\"callout\">",
                     "description: parts list then callout", da, db)

    # 4. Description down, after Episode End; its heading becomes a summary
    s, desc = cut(s, *section_span(s, "description"))
    desc = replace_once(desc, "    <h2>Description</h2>\n", "    <h2>Summary of the environment</h2>\n",
                        "description heading")
    s = insert_after_section(s, "episode-end", desc)

    # 5. Starting State up, after Observation Space
    s, st = cut(s, *section_span(s, "starting-state"))
    s = insert_after_section(s, "observation-space", st)

    # 6. Reference Scores up, after The recording controller
    s, rs = cut(s, *section_span(s, "reference-scores"))
    s = insert_after_section(s, "recording-controller", rs)

    # 7. the contents rail
    n0 = once(s, '<nav class="toc" aria-label="On this page">', "rail")
    n1 = s.index("</nav>", n0) + len("</nav>")
    rail = ['<nav class="toc" aria-label="On this page">', '  <span class="label">On this page</span>',
            '  <a href="#glance">At a glance</a>', '  <a href="#sim">Watch one episode</a>',
            '  <a href="#task">The task</a>', '  <a href="#action-space">Action Space</a>',
            '  <a href="#observation-space">Observation Space</a>', '  <a href="#starting-state">Starting State</a>',
            '  <a href="#dynamics">Transition Dynamics</a>', '  <a href="#rewards">Rewards</a>',
            '  <a href="#episode-end">Episode End</a>', '  <a href="#description">Summary</a>',
            '  <a href="#recording-controller">Recording controller</a>',
            '  <a href="#reference-scores">Reference Scores</a>', '  <a href="#dataset">Recorded data</a>',
            '  <a href="#shift">Distribution shift</a>', '  <a href="#methods">Methods</a>',
            '  <a href="#harm-checks">Harm checks</a>', '  <a href="#bca-test">BCA test</a>',
            '  <a href="#results">Results</a>', '  <a href="#weighting-test">Weighting test</a>',
            '  <a href="#arguments">Arguments</a>', '  <a href="#version-history">Version History</a>',
            '  <a href="#code-names">Names in the code</a>', '</nav>']
    s = s[:n0] + "\n".join(rail) + s[n1:]

    # 8. a target for the Starting State rewrite
    ta, tb = section_span(s, "task")
    s = replace_once(s, "    <h3>How each situation changes the challenge</h3>",
                     '    <h3 id="task-situations">How each situation changes the challenge</h3>', "task h3", ta, tb)

    # 9. words that described the old positions
    s = sub_ws(s, 'The whole task is explained in <a href="#task">The task</a>, just below.',
               'The whole task is explained in <a href="#task">The task</a>, right after\n        '
               '<a href="#sim">Watch one episode</a>, which comes next.', "glance: just below")
    ta, tb = section_span(s, "task")
    s = sub_ws(s, "To see the water act alone, pick <b>No throttle</b> in the panel below.",
               "To see the water act alone, pick <b>No throttle</b> as the controller in\n      "
               '<a href="#sim">Watch one episode</a> above.', "task: panel below", ta, tb)
    da, db = section_span(s, "description")
    s = sub_ws(s, "The springs drawn in the panel are only a picture.",
               'The springs drawn in\n        <a href="#sim">Watch one episode</a> are only a picture.',
               "description: the panel", da, db)

    # ---------- checks ----------
    assert data_line_hash(s) == data0, "the data line changed"
    ids = re.findall(r'\bid="([^"]+)"', s)
    dup = {i for i in ids if ids.count(i) > 1}
    assert not dup, f"duplicate ids: {sorted(dup)[:10]}"
    assert set(ids) == set(ids0) | {"glance", "wt-open", "task-situations"}, set(ids) ^ (set(ids0) | {"glance", "wt-open", "task-situations"})
    for h in set(re.findall(r'href="#([^"]+)"', s)):
        if h.startswith("MJX"):
            continue
        assert f'id="{h}"' in s, f"href #{h} does not resolve"
    # moved blocks keep their text (apart from the three positional sentences and the renamed heading)
    after_desc = block_text(s, "description")
    expect_desc = before["description"].replace("Description", "Summary of the environment", 1)
    assert strip_text(spec) == before["spec"]
    got = after_desc.replace(before["spec"] + " ", "").replace("Watch one episode are only", "the panel are only")
    assert got == expect_desc, "description text changed"
    assert block_text(s, "starting-state") == before["starting-state"]
    assert block_text(s, "reference-scores") == before["reference-scores"]
    sim_a = once(s, '    <figure class="panel" id="sim"', "live panel after")
    sim_b = s.index("\n    </figure>\n", sim_a) + len("\n    </figure>\n")
    assert strip_text(s[sim_a:sim_b]) == before["sim"]
    # main's children, in order
    m0 = once(s, "<main>", "main")
    m1 = once(s, "</main>", "main end")
    kids = re.findall(r'\n {2,4}<(?:section|figure class="panel"|header class="head") id="([^"]+)"', s[m0:m1])
    assert kids == ["top"] + ORDER, kids
    assert s[m0:m1].count("<section") == len(ORDER) - 1
    rail_ids = re.findall(r'href="#([^"]+)"', "\n".join(rail))
    assert rail_ids == ["glance"] + ORDER, rail_ids

    open(dst, "w", encoding="utf-8").write(s)
    print(f"moves: wrote {dst} ({len(s)} chars); main order: {' > '.join(kids)}")


if __name__ == "__main__":
    main()
