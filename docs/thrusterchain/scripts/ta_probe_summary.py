"""Print a one-line summary per probe of tmp/ta/probe.json (ta_probe.py's output)."""
import json

d = json.load(open("/mnt/c/Users/David/dev/bca-shift-work/page/tmp/ta/probe.json", encoding="utf-8"))
for k, v in d.items():
    if "hidden" in v:
        print(f"{k:42s} hides text={v['clipped']} scroll mode={v['scrollMode']} class={v['cls']!r} "
              f"content {v['sh']} px in a {v['ch']} px box, top {v['top']:.0f} bottom {v['bottom']:.0f} of {v['vh']}; "
              f"last visible words: {v['visibleEnd'][-70:]!r}")
w = d["weighting bars"]
print("weighting bars: before clicking Strong", len(w["before"]["titles"]), "<title>,", len(w["before"]["tips"]),
      "data-tip; after", len(w["after clicking Strong"]["titles"]), "<title>,", len(w["after clicking Strong"]["tips"]), "data-tip")
