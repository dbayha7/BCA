# ------------------------------------------------------------------ at most four items per slide
def gloss_split(src, groups, first, total):
    out = []
    for gi, ks in enumerate(groups):
        g = clone(src)
        keep = set()
        for k in ks:
            keep |= {f"Term {k}", f"Text {k + 1}", f"Text {k + 2}"}
        for sh in list(g.shapes):
            nm = sh.name
            is_term = nm.startswith("Term ") or (nm.startswith("Text ") and nm not in ("Text 0", "Text 1", "Text 3"))
            if is_term and nm not in keep:
                sh._element.getparent().remove(sh._element)
        for j, k in enumerate(ks):
            col, row = j % 2, j // 2
            x, y = 0.89 + col * 6.05, 1.6 + row * 2.4
            card, ttl, desc = shp(g, f"Term {k}"), shp(g, f"Text {k + 1}"), shp(g, f"Text {k + 2}")
            card.left, card.top, card.width, card.height = Inches(x), Inches(y), Inches(5.75), Inches(2.15)
            ttl.left, ttl.top, ttl.width, ttl.height = Inches(x + 0.22), Inches(y + 0.18), Inches(5.35), Inches(0.42)
            desc.left, desc.top, desc.width, desc.height = Inches(x + 0.22), Inches(y + 0.66), Inches(5.35), Inches(1.4)
            for p in ttl.text_frame.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(17)
            for p in desc.text_frame.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(14.5)
        e = shp(g, "Text 0")
        for p in e.text_frame.paragraphs:
            for r in p.runs:
                r.text = r.text.replace("(1 OF 2)", f"({first + gi} OF {total})").replace("(2 OF 2)", f"({first + gi} OF {total})")
        out.append(g)
    return out


gl_a = gloss_split(S[3], [[4, 7, 10], [13, 16, 19], [22, 25, 28]], 1, 6)        # the toy-problem term (31) is dropped
gl_b = gloss_split(S[4], [[4, 7, 10, 13], [16, 19, 22, 25], [28, 34, 37]], 4, 6)  # "comparison doses" (31) is dropped


def region_split(src, keep_left, card_top=(1.7, 4.9)):
    """Roadmap: keep the four cards on one side (by position) and spread them across the slide."""
    g = clone(src)
    lo, hi = card_top
    step_old, x_old = 1.457, 0.89
    for sh in list(g.shapes):
        t = Emu(sh.top).inches
        if not (lo <= t < hi):
            continue
        k = int(round((Emu(sh.left).inches - x_old) / step_old))
        if (k < 4) != keep_left:
            sh._element.getparent().remove(sh._element)
            continue
        j = k if keep_left else k - 4
        off = Emu(sh.left).inches - (x_old + k * step_old)
        sh.left = Inches(0.89 + j * 2.92 + off)
        sh.width = Inches(2.8 if off < 0.04 else 2.62)
    return g


road_a = region_split(rd, True)
road_b = region_split(rd, False)
for g, which in ((road_a, 1), (road_b, 2)):
    set_text(g, "Text 0", f"ROADMAP ({which} OF 2)")
delete(road_a, "Shape 40", "Text 41", "Text 42")
shp(road_a, "Shape 37").width = Inches(11.56)
for n in ("Text 38", "Text 39"):
    shp(road_a, n).width = Inches(11.56)
set_text(road_a, "Text 1", "Studies 1–4: is BCA's guarantee real?")
delete(road_b, "Shape 37", "Text 38", "Text 39")
for n in ("Shape 40", "Text 41", "Text 42"):
    shp(road_b, n).left = Inches(0.89); shp(road_b, n).width = Inches(11.56)
set_text(road_b, "Text 1", "Studies 5–8: does it help the actor?")


def rows_split(src, first_half):
    g = clone(src)
    cut = 1.55 + 0.48 * 4
    for sh in list(g.shapes):
        t = Emu(sh.top).inches
        if t < 1.5:
            continue
        if (t < cut) != first_half:
            sh._element.getparent().remove(sh._element)
        elif not first_half:
            sh.top = sh.top - Inches(0.48 * 4)
    return g


sum_a = rows_split(sm, True); sum_b = rows_split(sm, False)
set_text(sum_a, "Text 1", "Eight questions, eight answers (1 of 2)")
set_text(sum_b, "Text 1", "Eight questions, eight answers (2 of 2)")

REPLACE = {4: gl_a, 5: gl_b, 19: [road_a, road_b], 60: [sum_a, sum_b]}
