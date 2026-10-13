"""Write the page's demo script (the task diagram's player) to scripts/e4_demo.js as a Node module, for e4_test_node.js.
Read-only on the page. usage (WSL): python3 e4_demo_extract.py"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(HERE, "..", "thrusterchain.html")
s = open(PAGE, encoding="utf-8").read()
a = s.index("<script>\n/* ThrusterChain: the task diagram (#task-diagram) plays one demo episode.")
b = s.index("</script>", a)
js = s[a + len("<script>\n"):b]
assert s.count("/* ThrusterChain: the task diagram (#task-diagram) plays one demo episode.") == 1
assert "module.exports = { init, bind, arm, draw" in js
open(os.path.join(HERE, "e4_demo.js"), "w", encoding="utf-8").write(js)
print("wrote e4_demo.js,", len(js), "chars")
