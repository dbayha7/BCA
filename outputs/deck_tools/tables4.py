# ------------------------------------------------------------------ tables of at most four rows
def cell_set(cell, text):
    tf = cell.text_frame
    p0 = tf.paragraphs[0]
    runs = p0.runs
    if runs:
        runs[0].text = text
        for r in runs[1:]:
            r._r.getparent().remove(r._r)
    else:
        p0.text = text
    for p in tf.paragraphs[1:]:
        p._p.getparent().remove(p._p)


def table_of(slide):
    for sh in slide.shapes:
        if sh.has_table:
            return sh
    raise KeyError("no table")


# "Types of Sampling": merge the two K-rows-per-episode samplers into one row
ts = table_of(S[5]).table
r3, r4 = list(ts.rows)[3], list(ts.rows)[4]
cell_set(r3.cells[0], "K rows per episode (random or evenly spaced)")
cell_set(r3.cells[1], "About n/K episodes, with K rows from each: at random positions, or one from each of K equal "
                      "stretches of the episode")
r4._tr.getparent().remove(r4._tr)

# appendix "Where do the write-ups disagree?": two slides of four rows
def rows_keep(src, keep, label):
    g = clone(src)
    gf = table_of(g)
    rows = list(gf.table.rows)
    for i, r in enumerate(rows):
        if i > 0 and i not in keep:
            r._tr.getparent().remove(r._tr)
    t = shp(g, "Text 1")
    set_runs(t, [t.text_frame.paragraphs[0].runs[0].text + label])
    return g


dis_a = rows_keep(S[73], {1, 2, 3, 4}, " (1 of 2)")
dis_b = rows_keep(S[73], {5, 6, 7, 8}, " (2 of 2)")
REPLACE[74] = [dis_a, dis_b]
