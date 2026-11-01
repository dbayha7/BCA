# Erratum: the writing time stated in G2_RULE.md

G2_RULE.md says "Written 2026-10-31 07:20 EDT, before the screen code ran". That time was an estimate and is wrong.
The file system records:
- G2_RULE.md last modified 07:07:43 EDT;
- g2_feasibility.py last modified 07:12:32 EDT;
- g2_development_result.json written 07:13:51 EDT.

The result records the rule's sha256 as 2a868bae..., which matches the file. So the order (rule, then code, then
result) holds on file times and the hash, not on the stated time. G2_RULE.md is not edited, because its hash is in the
result. Found by the review workflow wf_e1964e57-92d.

The first development run used pi_ref_v1_dev.py and g2_feasibility_v1_dev.py. Those are byte copies of the files that
produced g2_development_result.json, kept so that run can be reproduced. Its label (INCONCLUSIVE, H1 at 43 of 48)
stands.
