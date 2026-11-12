"""Transcribe the three gridworld maps of Kumar et al. (ICLR 2022), Figure 7, from a rendered page.

Reads a binary PPM of the page, finds each map from its blue start cell (top left) and green goal
cell (bottom right), and classifies the 10 x 10 cells by the color at their centers.
"""
import sys

import numpy as np

path = sys.argv[1]
with open(path, "rb") as f:
    raw = f.read()
parts = raw.split(maxsplit=4)
assert parts[0] == b"P6", parts[0]
width, height, maxval = int(parts[1]), int(parts[2]), int(parts[3])
img = np.frombuffer(parts[4][-width * height * 3:], np.uint8).reshape(height, width, 3).astype(int)
r, g, b = img[..., 0], img[..., 1], img[..., 2]
# The figure sits in the top third of the page.
top = slice(0, height // 3)
blue = (r < 90) & (g < 90) & (b > 170)
green = (r < 90) & (g > 90) & (b < 90)


def boxes(mask):
    """Bounding boxes of the separate blobs of a mask, left to right."""
    cols = np.flatnonzero(mask[top].any(axis=0))
    groups, start = [], cols[0]
    for a, c in zip(cols[:-1], cols[1:]):
        if c - a > 5:
            groups.append((start, a))
            start = c
    groups.append((start, cols[-1]))
    out = []
    for x0, x1 in groups:
        rows = np.flatnonzero(mask[top][:, x0:x1 + 1].any(axis=1))
        out.append((x0, rows[0], x1, rows[-1]))
    return out


def kind(pixel):
    pr, pg, pb = pixel
    if pr < 90 and pg < 90 and pb > 170:
        return "S"
    if pr < 90 and pg > 90 and pb < 90:
        return "G"
    if pr > 180 and pg < 90 and pb < 90:
        return "L"
    if abs(pr - pg) < 25 and abs(pg - pb) < 25 and pr < 200:
        return "#"
    if pr > 225 and pg > 225 and pb > 225:
        return "."
    return "?"


starts, goals = boxes(blue), boxes(green)
print("page", width, "x", height, "| start cells", starts, "| goal cells", goals)
for name, s, gbox in zip(("single_critical", "multiple_critical", "cliffwalk"), starts, goals):
    x0, y0, x1, y1 = s[0], s[1], gbox[2], gbox[3]
    cw, ch = (x1 - x0 + 1) / 10.0, (y1 - y0 + 1) / 10.0
    print(f"\n{name}: grid x {x0}..{x1}, y {y0}..{y1}, cell {cw:.1f} x {ch:.1f} px; start cell {s[2] - s[0] + 1} x {s[3] - s[1] + 1} px")
    rows = []
    for row in range(10):
        line = ""
        for col in range(10):
            cx, cy = int(x0 + (col + 0.5) * cw), int(y0 + (row + 0.5) * ch)
            patch = img[cy - 3:cy + 4, cx - 3:cx + 4].reshape(-1, 3)
            line += kind(np.median(patch, axis=0))
        rows.append(line)
        print("   ", " ".join(line))
    counts = {k: sum(line.count(k) for line in rows) for k in "SGL#.?"}
    print("    counts:", counts)
