# Rebuild the PowerPoint advisor update

Run `python build_powerpoint.py` with python-pptx installed. The script edits the archived local 32-slide PowerPoint in place in memory and saves a separate 26-slide output. No Google Slides access is used. `raw-template.json` is an ID/order map only. All retained plot image bytes must match the source.

PowerPoint desktop exported the delivered PDF and rendered all 26 slides for visual inspection. No scientific trial was repeated. The first rendering exposed UTF-8 decoding and a title-fit problem; the delivered file fixes both.
