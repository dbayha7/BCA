"""Static structure and preservation checks; this does not render a browser."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import re
from lxml import html
from responsive import navigation_breakpoints

HERE = Path(__file__).resolve().parent
P = HERE.parent
BASE = P / 'ThrusterChain_diagnostics_results_v6_reviewed.html'
FINAL = P / 'ThrusterChain_diagnostics_results_v7_interactive.html'

def parse(path):
    return html.document_fromstring(path.read_text('utf8'), parser=html.HTMLParser(huge_tree=True))

def table_rows(doc):
    return Counter(tuple(' '.join(cell.text_content().split()) for cell in row.xpath('./td|./th'))
                   for row in doc.xpath('//table/tbody/tr'))

old, new = parse(BASE), parse(FINAL)
ids = [e.get('id') for e in new.xpath('//*[@id]')]
duplicates = [key for key, n in Counter(ids).items() if n > 1]
id_set = set(ids)
refs = []
for e in new.iter():
    if not isinstance(e.tag, str):
        continue
    for a in ['aria-labelledby', 'aria-describedby', 'aria-controls', 'for']:
        refs.extend((e.get('id'), a, v) for v in e.get(a, '').split() if v not in id_set)
    href = e.get('href', '')
    if href.startswith('#') and len(href) > 1 and href[1:] not in id_set:
        refs.append((e.get('id'), 'href', href))
lost_rows = table_rows(old) - table_rows(new)
old_scripts = [(e.get('id'), e.get('type'), e.text) for e in old.xpath('//script')]
new_scripts = [(e.get('id'), e.get('type'), e.text) for e in new.xpath('//script')]
old_source = [e.text_content() for e in old.xpath('//pre[@class="fr-source"]')]
new_source = [e.text_content() for e in new.xpath('//pre[@class="fr-source"]')]
original_hash = hashlib.sha256(BASE.read_bytes()).hexdigest()
addenda = [new.get_element_by_id(k) for k in ['direct-q-replication','direct-q-ranking','action-only-results']]
tables = [t for s in addenda for t in s.xpath('.//table')]
uncaptioned = [i for i,t in enumerate(tables) if not t.xpath('./caption')]
assert not duplicates, duplicates
assert not refs, refs
assert not lost_rows, list(lost_rows)[:3]
expected_scripts = [(id_, type_, navigation_breakpoints(text) if text and ('/* Before the page is drawn.' in text or 'const mqWide = mm(' in text) else text) for id_, type_, text in old_scripts]
assert sum(a != b for a, b in zip(old_scripts, expected_scripts)) == 2
assert expected_scripts == [s for s in new_scripts if s[0] not in ['xe-data','xe-interaction','lesson-restart-script','v7-responsive-script']]
assert old_source == new_source
assert original_hash == '99c8364283c0391d1448cfc79b8743380d150291ca82ad814232120f3a481208'
assert new.get('lang') == 'en'
assert len(new.xpath('/html/head/title')) == 1
assert not uncaptioned
assert new.get_element_by_id('results-explorer').getparent().tag == 'main'
assert len(new.xpath('//*[@data-unit="results-explorer"]')) == 1
restart_buttons = new.xpath('//button[@data-lesson-restart]')
assert len(restart_buttons) == 2
assert all(b.get('type') == 'button' for b in restart_buttons)
assert new.get_element_by_id('restart-lesson').getparent().getparent().get('id') == 'learn'
assert not new.get_element_by_id('restart-lesson').xpath('ancestor::*[contains(concat(" ",normalize-space(@class)," ")," ls-intro ") or contains(concat(" ",normalize-space(@class)," ")," ls-steps ")]')
receipt = dict(status='PASS', browser_rendered=False, final_sha256=hashlib.sha256(FINAL.read_bytes()).hexdigest(),
    original_v6_sha256=original_hash, original_v6_unchanged=True,
    unique_ids=len(ids), duplicate_ids=duplicates, broken_references=refs,
    original_table_rows_preserved=sum(table_rows(old).values()),
    original_script_blocks_identical=len(old_scripts)-2, navigation_scripts_changed_only_for_breakpoints=2, numbered_source_panels_preserved=len(old_source),
    captioned_addendum_tables=len(tables), lost_original_rows=0, lesson_restart_buttons=len(restart_buttons),
    limitations=['Static parsing and source inspection only.', 'No live navigation, viewport, print or screen-reader validation.'])
(HERE/'structure_check.json').write_text(json.dumps(receipt,indent=2)+'\n','utf8')
print(json.dumps(receipt,indent=2))
