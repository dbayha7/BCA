"""Review round 2, re-check (cloud): a deep link to a lesson step at load could leave the step's Try it unbuilt. The
navigation shows the step and sends walkthrough:enter on the next animation frame, but the lesson's script comes
after the navigation's in the page and can attach its listener after that frame. The lesson now also enters, once,
the step the navigation already shows (one section at a time) when its listeners are attached (enterStep builds a
step's controls once, so a later event changes nothing)."""
import rep

rep.apply([
    ("lesson/lesson.js",
     "  root.addEventListener('walkthrough:leave', function (ev) { var st = ev.target.closest && ev.target.closest('.ls-step'); if (st) leaveStep(st); });\n",
     "  root.addEventListener('walkthrough:leave', function (ev) { var st = ev.target.closest && ev.target.closest('.ls-step'); if (st) leaveStep(st); });\n"
     "  // a step the navigation showed before these listeners existed (a deep link at load): its walkthrough:enter is gone\n"
     "  if (html.classList.contains('nv-one')) [].slice.call(root.querySelectorAll('.ls-step')).forEach(function (st) {\n"
     "    if (!st.hasAttribute('hidden') && !st.classList.contains('nv-off') && !st.classList.contains('ls-off') && st.getClientRects().length) enterStep(st);\n"
     "  });\n", 1),
])
