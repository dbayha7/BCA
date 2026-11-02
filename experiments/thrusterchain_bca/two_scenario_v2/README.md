# Two-scenario task, v2: how to run the bounded setup

v1's setup attempt failed on one frozen number (PROTOCOL.md, "Revision v2"). v2 changes that number only.

1. Tests (Windows):
```bash
PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -B -m unittest discover -s experiments/thrusterchain_bca/two_scenario_v2 -p "test_*.py"
```
2. Seal (Windows; writes MANIFEST.json; refuses if one exists):
```bash
PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -B experiments/thrusterchain_bca/two_scenario_v2/setup.py --prepare
```
3. Approve at your terminal (writes SETUP_APPROVAL.json; the words are matched exactly against setup.py's APPROVAL):
```bash
.venv/Scripts/python.exe -c "import json,hashlib;p='experiments/thrusterchain_bca/two_scenario_v2/';h=hashlib.sha256(open(p+'MANIFEST.json','rb').read()).hexdigest();open(p+'SETUP_APPROVAL.json','x').write(json.dumps({'approval':'I approve the two-scenario v2 bounded setup on the ThrusterChain stall3 plant: frozen geometry with connector radius fraction 0.15, one 1,280-parent collection and one exact replay, at most 260,800 plant transitions and 840 seconds; no policy training and no scientific evaluation are authorized','manifest_sha256':h,'setup_only':True,'transition_cap':260800,'process_cap_seconds':840,'training_authorized':False},indent=2,sort_keys=True)+chr(10))"
```
4. Run the setup (WSL, CPU only, under 14 minutes, no training):
```powershell
wsl -d Ubuntu -- bash -lc 'cd /mnt/c/Users/David/dev/BCA && timeout --signal=TERM --kill-after=5s 835s /home/david/bca-operator-audit-20261026/bin/python -B -u experiments/thrusterchain_bca/two_scenario_v2/setup.py --execute-setup'
```
Outputs go to `two_scenario_v2/setup_attempt_1/`. Policy training stays blocked until the geometry and protocol are reviewed, as in v1.
