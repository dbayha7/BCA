"""Static structure and preservation checks; this does not render a browser."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import re
from lxml import html

HERE = Path(__file__).resolve().parent
P = HERE.parent
BASE = P / 'ThrusterChain_diagnostics_results_v5_fresh_replication.html'
FINAL = P / 'ThrusterChain_diagnostics_results_v6_reviewed.html'

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
passes = json.loads((HERE / 'passes.json').read_text())
snapshots_valid = all(hashlib.sha256((HERE / f'pass_{p["pass_number"]:02d}.html').read_bytes()).hexdigest() == p['sha256'] for p in passes)
addenda = [new.get_element_by_id(k) for k in ['direct-q-replication','direct-q-ranking','action-only-results']]
tables = [t for s in addenda for t in s.xpath('.//table')]
uncaptioned = [i for i,t in enumerate(tables) if not t.xpath('./caption')]
assert not duplicates, duplicates
assert not refs, refs
assert not lost_rows, list(lost_rows)[:3]
assert old_scripts == new_scripts
assert old_source == new_source
assert original_hash == '4d5d72b3d2b057f40012b17af4c792a640911c458483bab8379ad8226fe3cb5f'
assert len(passes) >= 5 and len({p['sha256'] for p in passes}) == len(passes) and snapshots_valid
assert FINAL.read_bytes() == (HERE / f'pass_{len(passes):02d}.html').read_bytes()
assert new.get('lang') == 'en'
assert len(new.xpath('/html/head/title')) == 1
assert not uncaptioned
receipt = dict(status='PASS', browser_rendered=False, final_sha256=hashlib.sha256(FINAL.read_bytes()).hexdigest(),
    original_v5_sha256=original_hash, original_v5_unchanged=True, passes=len(passes), pass_hashes_verified=snapshots_valid,
    unique_ids=len(ids), duplicate_ids=duplicates, broken_references=refs,
    original_table_rows_preserved=sum(table_rows(old).values()),
    all_script_blocks_identical=len(old_scripts), numbered_source_panels_preserved=len(old_source),
    captioned_addendum_tables=len(tables), lost_original_rows=0,
    limitations=['Static parsing and source inspection only.', 'No live navigation, viewport, print or screen-reader validation.'])
(HERE/'structure_validation.json').write_text(json.dumps(receipt,indent=2)+'\n','utf8')
print(json.dumps(receipt,indent=2))
