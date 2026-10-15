#!/usr/bin/env python3
"""Pass 4 design mock-up. READ-ONLY on the page: writes tmp/p4/design/mock.html, a copy of thrusterchain.html with
notes/p4_design.css appended after the main <style> and a few blocks of mock markup (the pipeline strip, the live
panel's compact strip and step notes, a regrouped "Reading the drawing" legend, situation marks in the shift table and
in one paragraph). Every replacement is counted. Run from the page directory:
    python3 scripts/p4_design_mock.py
"""
import hashlib
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE_DIR = os.path.dirname(HERE)
PAGE = os.path.join(PAGE_DIR, "thrusterchain.html")
CSS = os.path.join(PAGE_DIR, "notes", "p4_design.css")
OUT_DIR = os.path.join(PAGE_DIR, "tmp", "p4", "design")
OUT = os.path.join(OUT_DIR, "mock.html")

page_md5 = hashlib.md5(open(PAGE, "rb").read()).hexdigest()
s = open(PAGE, encoding="utf-8").read()
css = open(CSS, encoding="utf-8").read()


def rep(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, f"expected {count} of {old[:80]!r}, found {n}"
    s = s.replace(old, new)


# 1. lang for hyphenation (before the CSS goes in: its comments quote these tags)
rep('<div class="wrap">', '<div class="wrap" lang="en">')
rep('<dialog class="wt" id="wt"', '<dialog class="wt" lang="en" id="wt"')

# 2. the design CSS after the main style block (the first </style>)
i = s.index("</style>")
s = s[:i + len("</style>")] + "\n<style>\n/* p4 design mock */\n" + css + "\n</style>" + s[i + len("</style>"):]

# 3. the pipeline strip at the top of the at-a-glance block
STRIP = '''<nav class="pipe" id="mock-strip" aria-label="The offline reinforcement-learning pipeline in ThrusterChain, step by step">
      <ol>
        <li class="pipe-step" data-step="collect"><a href="#dataset">
          <span class="pipe-head"><span class="pipe-n">1</span><span class="pipe-t">Collect the data</span></span>
          <span class="pipe-d">The recording controller drives in the <span class="km" data-sit="recorded">recorded conditions</span>, in the stalling world. Every step becomes one recorded row.</span>
          <span class="pipe-go">The recorded data</span></a></li>
        <li class="pipe-step" data-step="remove"><a href="#shift">
          <span class="pipe-head"><span class="pipe-n">2</span><span class="pipe-t">Remove commands</span></span>
          <span class="pipe-d"><i class="key removed" data-k="region"></i>Main test: the densest 15% of commands, which takes the recording controller's full-throttle commands. <i class="key removed" data-k="hold"></i>Weighting test: the 80% hold.</span>
          <span class="pipe-go">The distribution shift</span></a></li>
        <li class="pipe-step" data-step="train"><a href="#methods">
          <span class="pipe-head"><span class="pipe-n">3</span><span class="pipe-t">Train TD3+BC</span></span>
          <span class="pipe-d">On what is left: first the critic, which rates each command, then the actor, which gives the commands the critic rates highest while a pull keeps it near the recorded ones.</span>
          <span class="pipe-go">Methods</span></a></li>
        <li class="pipe-step" data-step="band"><a href="#bca-test">
          <span class="pipe-head"><span class="pipe-n">4</span><span class="pipe-t">BCA's band</span></span>
          <span class="pipe-d">A calibrated band on the critic's one-step error. Where the band is wide, BCA adds caution.</span>
          <span class="pipe-go">The BCA test</span></a></li>
        <li class="pipe-step" data-step="test"><a href="#results">
          <span class="pipe-head"><span class="pipe-n">5</span><span class="pipe-t">Test the actor</span></span>
          <span class="pipe-d">In the <span class="km" data-sit="recorded">recorded conditions</span> (in distribution) and in three shifted situations (out of distribution): <span class="km" data-sit="strong_current">strong current</span>, <span class="km" data-sit="big_waves">big waves</span> and <span class="km" data-sit="big_kicks">big starting pushes</span>.</span>
          <span class="pipe-go">Results</span></a></li>
      </ol>
    </nav>
    '''
rep('<div class="glance">', STRIP + '<div class="glance">')

# 4. the live panel: compact strip after the opening sentence, a step note under each option row
SMALL = '''
        <nav class="pipe pipe-sm" id="mock-strip-sm" aria-label="Where this panel sits in the pipeline">
          <ol>
            <li class="pipe-step is-here" data-step="collect"><a href="#dataset"><span class="pipe-n">1</span>Collect</a></li>
            <li class="pipe-step" data-step="remove"><a href="#shift"><span class="pipe-n">2</span>Remove</a></li>
            <li class="pipe-step" data-step="train"><a href="#methods"><span class="pipe-n">3</span>Train</a></li>
            <li class="pipe-step" data-step="band"><a href="#bca-test"><span class="pipe-n">4</span>Band</a></li>
            <li class="pipe-step is-here" data-step="test"><a href="#results"><span class="pipe-n">5</span>Test</a></li>
          </ol>
        </nav>'''
rep("through the episode's 100 states one at a time.</p>", "through the episode's 100 states one at a time.</p>" + SMALL)
rep('How does the episode start?</span></h3>',
    'How does the episode start?</span></h3>\n          <p class="stepnote"><span class="stepchip" data-step="collect">Step 1</span><span class="stepchip" data-step="test">Step 5</span> The recorded conditions are where the data were collected; the other three situations are the out-of-distribution tests.</p>')
rep('What happens past 80% throttle?</span></h3>',
    'What happens past 80% throttle?</span></h3>\n          <p class="stepnote"><span class="stepchip" data-step="remove">Step 2</span> The stall is what the removal hides: after it, the data show almost no throttle past 80%, so the critic cannot learn the stall.</p>')
rep('Who sets the throttles?</span></h3>',
    'Who sets the throttles?</span></h3>\n          <p class="stepnote"><span class="stepchip" data-step="collect">Step 1</span><span class="stepchip" data-step="train">Step 3</span> The recording controller made the data; the rising-thrust controller acts the way a learned actor would if its critic never saw the stall.</p>')

# 5. "Reading the drawing", regrouped by part of the drawing, with marks that match it
a = s.index('<div class="symkey"><p class="symkey-h">Reading the drawing</p>')
b = s.index('</dl></div>', a) + len('</dl></div>')
DRAWKEY = '''<div class="symkey" id="mock-drawkey"><p class="symkey-h">Reading the drawing</p>
      <p class="symkey-g">The water bar at the top</p>
      <dl><div><dt>Top bar</dt><dd>It shows the water's push on every float, <i>c</i> + <i>w</i><sub>1</sub>: the current <i>c</i> plus the push <i>w</i><sub>1</sub> that the wave adds in this state.</dd></div>
      <div><dt><i class="key tile" data-k="region"></i>Full-throttle region</dt><dd>It is the part of the bar past 0.565, where the recording controller gives the full-throttle command in about half of the states.</dd></div></dl>
      <p class="symkey-g">The channel and the floats</p>
      <dl><div><dt><i class="key sitall"></i>Frame</dt><dd>The frame around the drawing has the colour of the situation you picked: <span class="km" data-sit="recorded">recorded conditions</span>, <span class="km" data-sit="strong_current">strong current</span>, <span class="km" data-sit="big_waves">big waves</span> or <span class="km" data-sit="big_kicks">big starting pushes</span>.</dd></div>
      <div><dt>Dashed line</dt><dd>It marks rest. A float is drawn below it when its position number is negative, where the water carries it.</dd></div>
      <div><dt>Number above a float</dt><dd>It is the float's position number <i>p</i>, one of its two numbers; both are 0 at rest.</dd></div>
      <div><dt>Small arrow by a float</dt><dd>It shows the float's speed number <i>v</i>, the other coordinate of its swing.</dd></div></dl>
      <p class="symkey-g">The thrusters and their gauges</p>
      <dl><div><dt>Arrow at a thruster</dt><dd>It shows the push the thruster really gives, <i>f</i>(<i>a</i>); pointing up is upstream, against the water.</dd></div>
      <div><dt>Gauge bar</dt><dd>It shows the thruster's throttle <i>a</i>, from −1 (full throttle against the water, at the top) to 1.</dd></div>
      <div><dt><i class="key tile" data-k="stall"></i>Red gauge, STALLED</dt><dd>In the stalling world a throttle past the dashed 80% marks stalls: the gauge turns red and its dark tick, the push, falls back.</dd></div></dl>
      <p class="symkey-g">The chart under the drawing</p>
      <dl><div><dt><i class="key line" data-k="recording"></i>Recording controller</dt><dd>Its cost so far; it wears the colour of the recorded data, because it made them.</dd></div>
      <div><dt><i class="key line" data-k="stall_aware"></i>Stall-aware controller</dt><dd>Its cost so far; it is the 100-point reference, so it is drawn in ink.</dd></div>
      <div><dt><i class="key line" data-k="rising_thrust"></i>Rising-thrust controller</dt><dd>Its cost so far; violet means "push keeps rising", which is also the colour of the no-stall comparison world.</dd></div>
      <div><dt><i class="key dash" data-k="no_throttle"></i>No throttle</dt><dd>Its cost so far, with all thrusters off.</dd></div></dl></div>'''
s = s[:a] + DRAWKEY + s[b:]

# 6. the shift table: each situation's group row carries its colour; the paragraph under it marks its mentions
for key, label in [("recorded", "Recorded conditions"), ("strong_current", "Strong current"), ("big_waves", "Big waves"),
                   ("big_kicks", "Big starting pushes")]:
    rep(f'<tr class="sub"><td>{label}</td>',
        f'<tr class="sub" data-sit="{key}"><td><span class="km" data-sit="{key}">{label}</span></td>')
rep('<p>Read the strong-current rows. Every visited state is outside the recorded conditions, and\n      93.7% are in the full-throttle region.',
    '<p id="mock-para">Read the <span class="km" data-sit="strong_current">strong-current</span> rows. Every visited state is outside the <span class="km" data-sit="recorded">recorded conditions</span>, and\n      93.7% are in the <span class="km" data-k="region">full-throttle region</span>.')

# 7. an ordinary legend to look at: the one in Transition Dynamics
rep('<div class="symkey"><p class="symkey-h">What the symbols mean</p><dl><div><dt><i>s</i><sub><i>t</i></sub>, <i>s</i><sub><i>t</i>+1</sub></dt>',
    '<div class="symkey" id="mock-legend"><p class="symkey-h">What the symbols mean</p><dl><div><dt><i>s</i><sub><i>t</i></sub>, <i>s</i><sub><i>t</i>+1</sub></dt>')

os.makedirs(OUT_DIR, exist_ok=True)
open(OUT, "w", encoding="utf-8", newline="\n").write(s)
assert hashlib.md5(open(PAGE, "rb").read()).hexdigest() == page_md5, "the page changed while the mock was built"
print("wrote", OUT, len(s), "chars; page md5 unchanged", page_md5)
