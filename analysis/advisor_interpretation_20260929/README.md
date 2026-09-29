# Advisor interpretation artifacts

The existing native Google Slides deck was edited in place. JSON files retain the content of the added analysis and synthetic-result slides. The exported PPTX/PDF and page renders are under outputs/advisor_interpretation/2026-09-29.

Rebuild the newly rendered figures from accepted, hash-recorded JSON:

    python analysis/advisor_interpretation_20260929/render_synthetic.py --output <new-output-directory>
    python analysis/advisor_interpretation_20260929/render_cql.py --output <new-output-directory>

These plotting commands produce no new samples or model queries. Dependencies: NumPy and matplotlib. Synthetic figures use the accepted v2 results. CQL plots retain all 1,000 original sparse last-row values and change only chart readability. Original plots and evidence remain in their prior publication.

The deck retains the original twelve result pages, adds result-specific interpretation and an explicit weighted-CP bridge, and retains actual source qualifications in speaker notes. Private provider response snapshots and transient signed image URLs are excluded from publication.
