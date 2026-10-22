"""Build self-contained V7 from preserved V6 and saved records only."""
from pathlib import Path
import csv,hashlib,json
from responsive import patch_responsive

HERE=Path(__file__).resolve().parent
P=HERE.parent
EXPERIMENT=P.parent
SOURCE=P/'ThrusterChain_diagnostics_results_v6_reviewed.html'
OUTPUT=P/'ThrusterChain_diagnostics_results_v7_interactive.html'
METHODS=['raw_q','live_width','matched_distance','shuffled_width','constant_offset']
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()=='99c8364283c0391d1448cfc79b8743380d150291ca82ad814232120f3a481208'

def value(text):
    if text=='True': return True
    if text=='False': return False
    if text=='': return None
    try:return float(text)
    except ValueError:return text

def rows(name):
    return list(csv.DictReader((EXPERIMENT/'results'/name).open(newline='',encoding='utf8')))

states=[]
candidates=rows('candidate_records.csv')
for r in rows('state_records.csv'):
    state={k:value(r[k]) for k in ['context','time','valid','rho_width','rho_distance','rho_shuffled','actor_true_q']}
    state['id']=int(r['state_id']);state['time']=int(r['time'])
    state['methods']={method:{k.split('__',1)[1]:value(v) for k,v in r.items() if k.startswith(method+'__')} for method in METHODS}
    state['candidates']=[]
    for c in candidates:
        if int(c['state_id'])!=state['id']:continue
        item={k:value(c[k]) for k in ['candidate_slot','support_class','beta','q1','true_policy_q','signed_q1_error','width','width_usable','native_width_refusal','distance','shuffled_width','shuffled_source_index']}
        item['id']=int(c['candidate_id']);item['action']=[float(c['action_'+str(i)]) for i in range(4)]
        item['scores']={method:float(c['score_'+method]) for method in METHODS}
        state['candidates'].append(item)
    assert [c['id'] for c in state['candidates']]==list(range(12))
    states.append(state)
assert len(states)==32 and all(s['valid'] for s in states)
data={'source':'Fresh-state replication, 21 October 2026','states':states,'source_sha256':{name:hashlib.sha256((EXPERIMENT/'results'/name).read_bytes()).hexdigest() for name in ['state_records.csv','candidate_records.csv']}}
s=patch_responsive(SOURCE.read_text('utf8'))
def replace(old,new):
    global s
    assert s.count(old)==1,(old[:90],s.count(old))
    s=s.replace(old,new,1)
replace('ThrusterChain diagnostics — results and fresh-state replication | V6','ThrusterChain — interactive diagnostics and evidence | V7')
replace('21 October 2026 · V6 reading revision of V5 evidence.','21 October 2026 · V7 interactive edition of the V5 evidence.')
replace('<section id="direct-q-replication"',HERE.joinpath('explorer.html').read_text('utf8')+'\n<section id="direct-q-replication"')
replace('<a class="btn" href="#direct-q-replication">Read the fresh-state result</a>','<a class="btn xe-primary" href="#results-explorer">Explore the saved results</a>\n      <a class="btn ghost" href="#direct-q-replication">Read the evidence</a>')
replace('<li class="nv-sec"><a href="#direct-q-replication"', '<li class="nv-sec"><a href="#results-explorer" data-unit="results-explorer">Explore the saved results</a></li>\n      <li class="nv-sec"><a href="#direct-q-replication"')
replace('<h2 id="fr-heading">','<p class="note"><a href="#results-explorer">Open the interactive explorer</a> to inspect each state and compare the five scoring rules.</p>\n<h2 id="fr-heading">')
replace('<a class="btn start-learn" id="start-learn" href="#learn">Learn step by step</a>', '<a class="btn start-learn" id="start-learn" href="#learn">Learn step by step</a>\n      <button type="button" class="btn ghost" id="restart-lesson-home" data-lesson-restart title="Clear lesson progress and quiz answers; return to step 1.1">Restart lesson</button>')
replace('<h2 id="learn-h">Learn step by step</h2>', '<h2 id="learn-h">Learn step by step</h2>\n<div class="lesson-restart-toolbar">\n  <button type="button" class="btn ghost" id="restart-lesson" data-lesson-restart aria-describedby="lesson-restart-note">Restart lesson</button>\n  <p id="lesson-restart-note">Start from step 1.1. Clears visited marks and quiz answers, and resets the Try-it controls.</p>\n</div>')
replace('</body>', '\n<style id="lesson-restart-style">\n'+HERE.joinpath('lesson_restart.css').read_text('utf8')+'\n</style>\n<script id="lesson-restart-script">\n'+HERE.joinpath('lesson_restart.js').read_text('utf8')+'\n</script>\n</body>')
replace('</body>','\n<style id="v7-explorer-style">\n'+HERE.joinpath('explorer.css').read_text('utf8')+'\n</style>\n<script type="application/json" id="xe-data">'+json.dumps(data,separators=(',',':'),allow_nan=False).replace('</','<\/')+'</script>\n<script id="xe-interaction">\n'+HERE.joinpath('explorer.js').read_text('utf8')+'\n</script>\n</body>')
replace('</body>', '\n<style id="v7-responsive-style">\n'+HERE.joinpath('responsive.css').read_text('utf8')+'\n</style>\n<script id="v7-responsive-script">\n'+HERE.joinpath('responsive.js').read_text('utf8')+'\n</script>\n</body>')
OUTPUT.write_text(s,encoding='utf8',newline='\n')
(HERE/'data.json').write_text(json.dumps(data,separators=(',',':'),allow_nan=False),'utf8')
print(json.dumps({'output':str(OUTPUT),'bytes':OUTPUT.stat().st_size,'sha256':hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),'states':len(states),'candidates':len(candidates)},indent=2))
