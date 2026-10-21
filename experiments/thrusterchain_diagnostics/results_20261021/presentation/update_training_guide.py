"""Build a new, offline results guide from a verified saved aggregate; stdlib only.

Never imports or executes scientific code. The canonical guide is read-only.
An existing output is refused; use a new filename for a revised presentation.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import html
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
import statistics
from string import Template
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
BASE = Path(r"C:\Users\David\dev\BCA\experiments\thrusterchain_diagnostics\ThrusterChain_diagnostics.html")
CONTEXTS = ["pooled", "stall_low", "stall_high", "monotone_low", "monotone_high"]
RETENTIONS = ["full", "rare", "zero"]
LABEL = {"pooled": "Pooled", "stall_low": "Stall / low", "stall_high": "Stall / high", "monotone_low": "Monotone / low", "monotone_high": "Monotone / high", "td3_bc": "TD3+BC", "bca": "TD3+BC+BCA"}
SCRIPT = r"""(() => {
  'use strict';
  const retention = document.getElementById('tr-retention-filter');
  const context = document.getElementById('tr-context-filter');
  const count = document.getElementById('tr-filter-count');
  const rows = Array.from(document.querySelectorAll('#tr-policy-table tbody tr'));
  if (!retention || !context || !count) return;
  function update() {
    let visible = 0;
    rows.forEach(row => {
      const show = (retention.value === 'all' || row.dataset.trRetention === retention.value) &&
        (context.value === 'all' || row.dataset.trContext === context.value);
      row.hidden = !show;
      if (show) visible += 1;
    });
    count.textContent = visible + ' of ' + rows.length + ' saved policy/context rows shown.';
  }
  retention.addEventListener('change', update);
  context.addEventListener('change', update);
  update();
})();"""

def esc(value):
    return html.escape(str(value), quote=True)

def fmt(value, digits=3):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            return esc(str(value))
        return f"{value:,.{digits}f}" if digits else f"{value:,.0f}"
    return esc(value)

def pct(value):
    return fmt(100 * value, 1) + "%" if isinstance(value, (float, int)) else fmt(value)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def table(headers, rows, ident=None, attrs=None):
    id_part = f' id="{esc(ident)}"' if ident else ""
    body = []
    for i, row in enumerate(rows):
        extra = attrs[i] if attrs else ""
        body.append(f"<tr{extra}>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>")
    return '<div class="table-wrap tr-wide"><table' + id_part + '><thead><tr>' + "".join(f"<th scope=\"col\">{esc(h)}</th>" for h in headers) + '</tr></thead><tbody>' + "".join(body) + '</tbody></table></div>'

def details(title, content):
    return f"<details><summary>{esc(title)}</summary>{content}</details>"

def raw_details(title, data):
    return details(title, '<pre>' + esc(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)) + '</pre>')

def prose(value):
    if value is None:
        return "Not supplied in the saved record."
    if isinstance(value, (list, dict)):
        return '<pre>' + esc(json.dumps(value, indent=2, ensure_ascii=False)) + '</pre>'
    return esc(value)

def fixture_records(data):
    if isinstance(data, list): return data
    for key in ("fixtures", "results", "tests", "cases"):
        if isinstance(data.get(key), list): return data[key]
    return []

def render_audit(method, stage_map, fixtures):
    if not method:
        return '<h3 id="tr-stage-map">Stage verification — internal preview</h3><p>Source audit and isolated fixture records are still pending. No stage test is claimed by this preview.</p>'
    fs = fixture_records(fixtures or {})
    stages = [dict(s) for s in method.get("stages", [])]
    extra = stage_map.get("stages", []) if stage_map else []
    existing = {s["id"] for s in stages}
    for s in stages:
        matched_map = next((m for m in extra if m["id"] == s["id"]), {})
        s["fixture_ids"] = sorted(set(s.get("fixture_ids", []) + matched_map.get("fixture_ids", [])))
    stages = stages + [dict(s, branch=s.get("branch", "IW fixture branch" if str(s["id"]).startswith("W") else "native BCA fixture"), status="isolated fixtures: " + s.get("status", "not recorded")) for s in extra if s["id"] not in existing and s["id"] != "E1"]
    out = ['<h3 id="tr-stage-map">Follow the computation, one stage at a time</h3>', '<p><b>Branch boundary:</b> the 30 trained jobs below use native TD3+BC+BCA. Importance-weighted BCA is a separate version branch with separate ratio, query-mass and refusal rules. Its isolated fixtures are not additional trained ThrusterChain results. A larger IID penalty is a debugging question, not automatically a bug: IID can imply ratio one while residual noise and calibrated widths remain large.</p>']
    out.append(table(["Verified finding / scope", "Evidence status", "What to take away"], [[esc(f.get("id")) + "<br>" + esc(", ".join(f.get("sources", []))), esc(f.get("status")), esc(f.get("finding"))] for f in method.get("new_findings", [])]))
    out.append('<p>The exploratory command-weighted branch has a confirmed mixed-measure contract defect: clipped target sampling creates atom masses that its smooth-density ratio helper omits. Its claim to an exact command ratio and associated calibration interpretation is invalid until the measure is corrected and retested. This finding does not invalidate the native 30-job experiment, the separately specified current-law IW arithmetic, or the historical returns as descriptive measurements.</p>')
    rows = []
    for s in stages:
        sid = str(s["id"])
        rows.append([f'<a href="#tr-stage-{esc(sid)}">{esc(sid)}</a>', esc(s.get("branch", "see source")), esc(s.get("title", sid)), esc(s.get("status", "source trace; see measured fixtures"))])
    out.append(table(["Stage", "Version branch", "Operation and purpose", "Evidence status"], rows))
    out.append('<p>Source-audited means the code path was traced; it is not a passed runtime test. PASS below means a stated isolated invariant met its stated fixture expectation. An illustrative counterexample can pass its reproduction test while showing an undesirable penalty pattern. Untested links to actual error or harm remain untested.</p>')
    summary = fixtures.get("summary", {})
    out.append('<div class="callout"><b>Measured fixture outcome:</b> ' + fmt(summary.get("passed"),0) + ' narrow scientific checks pass; ' + fmt(summary.get("failed"),0) + ' fail. ' + fmt(summary.get("harness_passed", summary.get("passed")),0) + ' fixture executions reproduced their expected behavior. Reproducing the two clipping counterexamples confirms the command-law defect; it does not make that implementation pass.</div>')
    out.append(raw_details("Isolated fixture summary and explicitly untested links", {k: fixtures.get(k) for k in ("summary", "untested", "coverage_claim", "environment") if k in fixtures}))
    for s in stages:
        sid = str(s["id"])
        out.append(f'<article class="tr-stage" id="tr-stage-{esc(sid)}"><h4>{esc(sid)} · {esc(s.get("title", sid))}</h4>')
        out.append(table(["Stage contract", "Meaning"], [["Version / premise", prose(s.get("branch", "see source")) + "<br>" + prose(s.get("premise"))], ["Input → output", prose(s.get("inputs", s.get("input"))) + "<br>→<br>" + prose(s.get("outputs", s.get("output")))], ["Purpose", prose(s.get("purpose"))], ["Invariant", prose(s.get("invariant"))]]))
        pseudo = s.get("pseudocode", [])
        if pseudo:
            out.append('<p><b>Code-derived pseudocode</b> — labels identify this teaching stage; references identify actual source lines.</p><pre class="tr-code"><code>')
            for index, line in enumerate(pseudo, 1):
                if isinstance(line, str): line = {"line": index, "text": line}
                ref = line.get("source", line.get("source_ref", ""))
                if isinstance(ref, dict):
                    first = ref.get("start", ref.get("first_line", "?"))
                    last = ref.get("end", ref.get("last_line", first))
                    ref = f'{ref.get("path", "")}:L{first}–L{last}'
                out.append(f'<span><b>{esc(line.get("line", index))}</b>{esc(line.get("text", line))}' + (f'  [{esc(ref)}]' if ref else '') + '</span>')
            out.append('</code></pre>')
        matched = []
        for f in fs:
            stage = f.get("stage", f.get("stages", []))
            stageids = stage if isinstance(stage, list) else re.findall(r'[NWI]\d+', str(stage))
            if stage == "E1": stageids = ["I11"]
            if sid in stageids or f.get("id") in s.get("fixture_ids", []): matched.append(f)
        out.append('<p><b>Source audit:</b> ' + prose(s.get("status", "source traced")) + '. ' + prose(s.get("evidence", "See the preserved source references.")) + '</p>')
        out.append('<p><b>Isolated test:</b> ' + prose(s.get("isolated_test", "See matching measured fixtures below.")) + '</p>')
        if s.get("fixture_scope_note"):
            out.append('<p><b>Scope of the measured check:</b> ' + prose(s["fixture_scope_note"]) + '</p>')
        if matched:
            out.append(table(["Fixture / premise", "Expected", "Observed", "Measured status / takeaway"], [[esc(f.get("id", "")) + "<br>" + prose(f.get("premise")), prose(f.get("expected")), prose(f.get("observed")), '<b>' + esc(f.get("status", "unknown")) + '</b><br>' + prose(f.get("interpretation", f.get("takeaway")))] for f in matched]))
        else:
            out.append('<p><b>Measured isolated-fixture status:</b> no matching new runtime fixture is recorded for this stage. Source inspection and prior integration checks do not replace that missing isolated test.</p>')
        out.append('<p><b>Takeaway:</b> ' + prose(s.get("takeaway")) + '</p></article>')
    out.append('<h3 id="tr-fixture-cases">Consistent and exceptional cases to reproduce</h3>')
    out.append('<p>The fixture seeds and inputs below are explicit debugging cases, not a sampled estimate of how often a pattern occurs in deployed RL. “Always” is restricted to a tested invariant; an exceptional case is a concrete reproducible counterexample, not an estimated population frequency. <b>A reproduced counterexample confirms a defect or limitation; it is not an implementation pass.</b> The green arithmetic checks therefore cannot validate the separate command-ratio branch.</p>')
    out.append(table(["Case", "Stage", "Premise", "Expected", "Observed", "Status", "Takeaway"], [[esc(f.get("id")), prose(f.get("stage", f.get("stages"))), prose(f.get("premise")), prose(f.get("expected")), prose(f.get("observed")), esc(f.get("status")), prose(f.get("interpretation", f.get("takeaway")))] for f in fs]))
    out.append(raw_details("All fixture inputs, seeds, source references and recorded results", fixtures))
    out.append('<h3 id="tr-method-decisions">Which version is ready for which claim?</h3>')
    out.append(table(["Method / branch", "Decision", "Reason"], [[esc(v.get("method")), esc(v.get("decision")), esc(v.get("reason"))] for v in method.get("decisions", [])]))
    out.append(raw_details("Source-audit decisions and proposed follow-up ablations", {k: method.get(k) for k in ("repository", "source_files", "studies", "decisions", "proposed_ablation")}))
    out.append('<p>To reproduce these debugging cases, run the saved <code>module_fixtures/run_fixtures.py</code> with the arguments in <code>module_fixtures/commands.ps1</code>. The panels are suitable for a presentation screenshot, but no browser screenshot is represented as captured or verified in this session.</p>')
    return ''.join(out)

def render_penalty(p):
    start = '<h3 id="tr-penalty-alignment">Does the penalty track actual error or worse actions?</h3>'
    if not p: return start + '<p>Penalty alignment records are pending; no alignment result is asserted.</p>'
    out = [start, '<p>The native dose is a bounded multiplier on behavior-cloning regularization, not an estimate of physical harm. These diagnostics measure the <b>multiplier only, not the full BC loss or its gradient</b>. A larger coefficient does not establish a larger actual penalty because squared distance from the logged action also varies. In training, native BCA computes the multiplier at the <b>logged action</b> and weights that row\'s cloning loss. The six-candidate analysis below evaluates the frozen multiplier function at diagnostic commands; it does not reconstruct training weights or show direct penalties imposed on those candidate commands. It uses saved terminal query arrays from 11 Zeus BCA jobs: one full-retention replicate and all five rare and zero replicates. The four Titan full BCA raw query arrays are unavailable here; no values are imputed.</p>']
    rows = []
    for r, v in p.get("summary", {}).items():
        rows.append([esc(r), fmt(v.get("available_seeds"), 0) + "/" + fmt(v.get("planned_seeds"), 0), ", ".join(fmt(x) for x in v.get("error_correlations", [])), ", ".join(fmt(x) for x in v.get("opportunity_loss_correlations", [])), ", ".join(fmt(x) for x in v.get("within_state_loss_concordance", []))])
    out.append(table(["Retention", "Available BCA replicates", "Dose vs absolute Q error: Spearman per replicate", "Dose vs six-action opportunity loss: Spearman per replicate", "Within-state opportunity-loss concordance"], rows))
    out.append(f'<div class="callout"><b>Positive pooled associations coexist with reproducible ordering exceptions.</b><p>Across {fmt(p.get("total_states"),0)} designed query states, {fmt(p.get("inversion_count"),0)} assign a larger BC multiplier to the best-Q candidate than to the worst-Q candidate. These are counterexamples to universal multiplier ordering by candidate opportunity loss, not proof of harmful regularization. They do not show that uncertainty should always be smaller for good actions, or that the dose formula is incorrect.</p></div>')
    out.append('<p>Opportunity loss is the best of six same-state policy-Q values minus candidate Q, not physical damage or regret over the full action space. The correlations are descriptive on this designed query law, with no independent-sample p-value. Within-state concordance excludes true target ties and reduces confounding by state difficulty. Each policy defines its own continuation target.</p>')
    contexts = p.get("per_seed_context", [])
    out.append(details("All available per-replicate/context alignment measurements", table(["Job", "Context", "States", "Dose / Q-error rank correlation", "Dose / opportunity-loss rank correlation", "Width / native-residual rank correlation", "Within-state error concordance", "Within-state loss concordance"], [[esc(v.get("job_id")), esc(v.get("context")), fmt(v.get("states"),0), fmt(v.get("dose_vs_abs_Q_error_spearman")), fmt(v.get("dose_vs_candidate_opportunity_loss_spearman")), fmt(v.get("width_vs_native_residual_spearman")), fmt(v.get("within_state_error_ordering",{}).get("concordant")), fmt(v.get("within_state_opportunity_loss_ordering",{}).get("concordant"))] for v in contexts])))
    cases = p.get("reproduction_cases", [])
    if cases:
        first = cases[0]
        out.append('<p><b>First reproducible exception in deterministic job/state order:</b> ' + esc(first.get("job_id")) + ', query ' + fmt(first.get("query_state_index"),0) + ', time ' + fmt(first.get("time"),0) + '. The best candidate (' + esc(first.get("best_candidate")) + ') has Q ' + fmt(first.get("best_true_Q")) + ' and dose ' + fmt(first.get("best_dose"),6) + '; the worst (' + esc(first.get("worst_candidate")) + ') has Q ' + fmt(first.get("worst_true_Q")) + ' and dose ' + fmt(first.get("worst_dose"),6) + '. This is a saved example, not an outcome-selected representative claim.</p>')
    out.append(raw_details("Every qualifying counterexample, including commands, Q values, doses and query identities", cases))
    out.append('<p>These values describe the diagnostic dose at the 100k checkpoint, not the full history of interventions during learning. They concern native BCA only; no IW-BCA outcome is inferred. Reproduce using <code>analysis_penalty_alignment.py --run-dir &lt;run-or-evidence-directory&gt; --output &lt;new-directory&gt;</code>. No additional rollout or training is required.</p>')
    return ''.join(out)

def get_summary(job, split="final"):
    return job.get("evaluation", {}).get(split) or {}

def context_data(summary, context):
    return summary.get("pooled", {}) if context == "pooled" else summary.get("strata", {}).get(context, {})

def manifest(job):
    p = job.get("preparation", {}) or {}
    return p.get("manifest", p)

def jkey(job):
    j = job["job"]
    return [esc(j["retention"]), fmt(j["replicate"], 0), LABEL.get(j["arm"], esc(j["arm"]))]

def validate(a, partial):
    if a.get("complete") is not True and not partial:
        raise ValueError("Verified aggregate is not complete; --allow-partial is only for internal preview")
    jobs = a.get("jobs", [])
    identities = [(j["job"]["replicate"], j["job"]["retention"], j["job"]["arm"]) for j in jobs]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate job identities")
    if not partial:
        expected = {(r, t, arm) for r in range(1, 6) for t in RETENTIONS for arm in ("td3_bc", "bca")}
        if set(identities) != expected:
            raise ValueError("Final aggregate must account for all 30 planned jobs")
        for j in jobs:
            if j.get("status") in ("completed", "complete"):
                if j.get("updates") != 100000 or not get_summary(j) or not get_summary(j, "validation"):
                    raise ValueError("Completed job missing terminal budget or required saved evaluations")
    for c in a.get("paired_comparisons", []):
        reps = c.get("replicates", [])
        if c.get("n") == 5:
            if len(reps) != 5 or len({r["replicate"] for r in reps}) != 5 or c.get("df") != 4:
                raise ValueError("Invalid five-replicate comparison")
            for r in reps:
                if not math.isclose(r["delta"], r["bca"] - r["td3_bc"], abs_tol=1e-8):
                    raise ValueError("Paired difference inconsistent with returns")
            if not math.isclose(statistics.mean(r["delta"] for r in reps), c["mean_difference"], abs_tol=1e-8):
                raise ValueError("Paired aggregate mean inconsistent")

def comparison_table(comparisons):
    rows = []
    for c in comparisons:
        reps = c.get("replicates", [])
        complete = c.get("n") == 5 and len(reps) == 5
        means = [fmt(statistics.mean(r[k] for r in reps)) + " ± " + fmt(statistics.stdev(r[k] for r in reps)) if complete else "withheld" for k in ("td3_bc", "bca")]
        interval = "[" + ", ".join(fmt(v) for v in c.get("ci95", [])) + "]" if complete else "withheld"
        rows.append([esc(c["retention"]), LABEL.get(c["context"], esc(c["context"]))] + means + [fmt(c.get("mean_difference")) if complete else "withheld", fmt(c.get("sample_sd_difference")) if complete else "withheld", interval, fmt(c.get("n"), 0), fmt(c.get("host_confounded_pairs"), 0)])
    return table(["Retention", "Context", "TD3+BC mean ± SD", "+BCA mean ± SD", "BCA − TD3+BC", "Paired SD", "Exploratory t 95% interval", "Trained replicates", "Cross-host pairs"], rows)

def delta_plot(comparisons):
    cs = [c for c in comparisons if c.get("context") == "pooled" and c.get("n") == 5]
    if not cs:
        return "<p>Plot withheld: no complete five-replicate pooled comparison.</p>"
    values = [0.0] + [r["delta"] for c in cs for r in c["replicates"]] + [v for c in cs for v in c["ci95"]]
    low, high = min(values), max(values)
    pad = max((high - low) * .12, 1)
    low, high = low - pad, high + pad
    x = lambda v: 150 + (v - low) / (high - low) * 640
    parts = ['<figure><svg viewBox="0 0 900 300" role="img" aria-labelledby="tr-delta-title tr-delta-desc" style="width:100%;max-width:980px;background:var(--surface);border:1px solid var(--rule);border-radius:10px">', '<title id="tr-delta-title">Terminal pooled return differences by retention</title>', '<desc id="tr-delta-desc">Each circle is one paired trained-replicate difference, BCA minus TD3 plus BC. Diamonds show five-replicate means and lines exploratory paired t 95 percent intervals. Negative differences favor TD3 plus BC. Four of five full and zero pairs are cross-host.</desc>']
    parts += [f'<line x1="{x(0):.2f}" x2="{x(0):.2f}" y1="25" y2="235" stroke="currentColor" stroke-dasharray="4 4"/>']
    for idx, c in enumerate(cs):
        y = 65 + idx * 65
        parts += [f'<text x="20" y="{y+5}" fill="currentColor" font-size="16">{esc(c["retention"])}</text>', f'<line x1="{x(c["ci95"][0]):.2f}" x2="{x(c["ci95"][1]):.2f}" y1="{y}" y2="{y}" stroke="#486e92" stroke-width="3"/>']
        for k, r in enumerate(c["replicates"]):
            cy = y - 14 + k * 7
            parts.append(f'<circle cx="{x(r["delta"]):.2f}" cy="{cy}" r="4.5" fill="#b35a35"><title>replicate {r["replicate"]}: {r["delta"]:.6f}</title></circle>')
        m = x(c["mean_difference"])
        parts.append(f'<path d="M {m:.2f},{y-7} L {m+7:.2f},{y} L {m:.2f},{y+7} L {m-7:.2f},{y} Z" fill="#214d72"/>')
    for k in range(6):
        value = low + (high - low) * k / 5
        parts.append(f'<text x="{x(value):.2f}" y="258" text-anchor="middle" fill="currentColor" font-size="12">{value:.1f}</text>')
    parts.append('<text x="470" y="286" text-anchor="middle" fill="currentColor" font-size="13">BCA − TD3+BC return; circles = five trained replicates</text></svg><figcaption>Saved terminal results. Lines are exploratory t intervals, not rollout-level confidence intervals. Host confounding limits the full and zero contrasts.</figcaption></figure>')
    return "".join(parts)

def sections(a, partial, method=None, stage_map=None, fixtures=None, penalty=None, execution=None, titan=None):
    jobs = sorted(a["jobs"], key=lambda j: (RETENTIONS.index(j["job"]["retention"]), j["job"]["replicate"], j["job"]["arm"]))
    final = [c for c in a.get("paired_comparisons", []) if c["split"] == "final"]
    terminal = [c for c in final if c["context"] == "pooled"]
    d = {}
    d["stage_audit"] = render_audit(method, stage_map, fixtures)
    d["penalty_alignment"] = render_penalty(penalty)
    d["status"] = '<div class="callout"><b>' + ("INTERNAL PARTIAL PREVIEW — not a final results release" if partial else "Completed training matrix · terminal checkpoint primary · 21 October 2026") + '</b><p>All values below are read from saved records. The evidence distinguishes Zeus verification from imported Titan report evidence.</p></div>'
    if len(terminal) == 3:
        text = "; ".join(f'{esc(c["retention"])} {fmt(c["mean_difference"])}' for c in terminal)
        d["headline"] = f'<div class="callout"><b>The pooled comparison does not show a BCA return benefit at this budget.</b><p>The mean BCA-minus-TD3+BC differences are {text}. Context results vary. These five-replicate, fixed-bank results evaluate this intervention and training budget; they do not establish a general failure of calibration or BCA. Full and zero also confound method with host.</p></div>'
    else:
        d["headline"] = '<p>Terminal comparison withheld until all three five-replicate conditions are available.</p>'
    d["budget"] = table(["Item", "Frozen core setting"], [["Per job", "100,000 accepted critic updates; 50,000 delayed actor/target updates; batch 256 (25.6 million sampled training rows)"], ["Learner", "Two 256-unit hidden layers; Adam learning rate 0.0003; TD3+BC alpha 2.5; tau 0.005; policy delay 2; target noise 0.2 clipped at 0.5"], ["BCA intervention", "Warm-up 10,000; refresh every 5,000; blend 0.5; scale LR 0.001; beta 20; width penalty 0.005; scale EMA 0.99; 1,000 posterior draws"], ["Seeds", "Data 1101–1105; native optimization 2101–2105; reservation 3101–3105; separate named evaluation streams"], ["Observation and reward", "15 coordinates; H = 100; gamma = 1; raw reward = −(sum of first 10 state coordinates squared + 0.3 × command norm squared)"]])
    d["comparisons"] = comparison_table(final)
    d["plot"] = delta_plot(final)
    rows = []
    for j in jobs:
        source = j.get("source", {})
        rows.append(jkey(j) + [esc(j.get("host", "unknown")), esc(j.get("backend", "unknown")), esc(j.get("status", "missing")), fmt(j.get("updates"), 0), fmt(j.get("actor_updates"), 0), fmt(j.get("training_seconds"), 1), "arrays + checkpoint verified" if source.get("independently_verified_arrays") and source.get("independently_verified_checkpoint") else "imported report; no independent rollout rerun"])
    d["jobs"] = table(["Retention", "Replicate", "Arm", "Host", "Backend", "Status", "Critic updates", "Actor updates", "Training seconds", "Evidence"], rows, "tr-job-table")
    rows, attrs = [], []
    for j in jobs:
        s = get_summary(j)
        for ctx in CONTEXTS:
            v = context_data(s, ctx)
            rows.append(jkey(j) + [LABEL[ctx], fmt(v.get("mean_return")), fmt(v.get("regret_lower"), 6) + " – " + fmt(v.get("regret_upper"), 6), pct(v.get("failure_rate")), pct(v.get("high_action_fraction")), pct(v.get("critical_fraction")), fmt(v.get("completed"), 0) + "/" + fmt(v.get("starts"), 0)])
            attrs.append(f' data-tr-retention="{esc(j["job"]["retention"])}" data-tr-context="{ctx}"')
    d["policy"] = table(["Retention", "Replicate", "Arm", "Context", "Mean return", "QP regret lower – upper", "Failure", "Above-knee steps", "Critical steps", "Completed / starts"], rows, "tr-policy-table", attrs)
    d["oracle"] = raw_details("Continuous QP certificate and bank record", a.get("oracle", {}))
    rows = []
    for c in a.get("controls", []):
        for ctx in CONTEXTS:
            v = context_data(c.get("evaluation", {}), ctx)
            bounds = (c.get("derived_regret_bounds") or {}).get(ctx)
            rows.append([esc(c.get("name")), esc(c.get("split")), LABEL[ctx], fmt(v.get("mean_return")), " – ".join(fmt(x,6) for x in bounds) if bounds else "—", pct(v.get("failure_rate")), pct(v.get("high_action_fraction"))])
    d["controls"] = table(["Untrained model-based control", "Bank", "Context", "Mean return", "QP regret lower – upper", "Failure", "Above-knee steps"], rows)
    seen, rows, ctxrows, bankcounts = set(), [], [], {r: [] for r in RETENTIONS}
    for j in jobs:
        jj = j["job"]; key = (jj["replicate"], jj["retention"])
        if key in seen:
            continue
        seen.add(key); m = manifest(j)
        bankcounts[jj["retention"]].append(m.get("calibration_critical_rows"))
        tr = m.get("training_rows")
        rows.append([esc(jj["retention"]), fmt(jj["replicate"], 0), fmt(jj.get("data_seed"), 0), fmt(tr, 0), fmt(m.get("training_critical_rows"), 0), fmt(m.get("withheld_rows"), 0), fmt(m.get("calibration_rows"), 0), fmt(m.get("calibration_critical_rows"), 0), fmt(25600000/tr, 1) if tr else "—", esc(str(m.get("bca_refusal") or "none"))])
        for ctx in (m.get("calibration_context_counts", {}) or {}):
            ctxrows.append([esc(jj["retention"]), fmt(jj["replicate"], 0), esc(ctx), fmt(m.get("training_context_counts", {}).get(ctx), 0), fmt(m.get("training_critical_context_counts", {}).get(ctx), 0), fmt(m.get("calibration_context_counts", {}).get(ctx), 0), fmt(m.get("calibration_critical_context_counts", {}).get(ctx), 0)])
    d["data"] = table(["Retention", "Replicate", "Data seed", "Training rows", "Critical training rows", "Withheld rows", "Calibration rows", "Critical calibration rows", "Sample draws / train row", "Initial BCA refusal"], rows)
    d["critical_note"] = '<div class="callout"><b>128 pooled rows are not 128 critical rows.</b><p>Critical bank counts by replicate: ' + "; ".join(f'{r} [{", ".join(fmt(v,0) for v in vs)}]' for r, vs in bankcounts.items()) + '. Rare replicates 3 and 4 have zero critical rows; the other rare banks have only 1, 3 and 1. No critical-conditional coverage claim is justified. No top-up or retuning was made after seeing these counts.</p></div>'
    d["data_contexts"] = details("Retained and reserved context counts (all 15 data conditions)", table(["Retention", "Replicate", "Context code", "Training", "Critical training", "Bank", "Critical bank"], ctxrows))
    rows, calrows, calctx = [], [], []
    for j in jobs:
        s = get_summary(j); c = s.get("critic_diagnostics", {}) or {}; q = c.get("q_error", {}); rank = c.get("action_ranking", {}); br = c.get("bellman_residual", {})
        rows.append(jkey(j) + [esc(c.get("status", "absent")), fmt(q.get("critic_1", {}).get("rmse")), fmt(q.get("critic_2", {}).get("rmse")), fmt(q.get("twin_min", {}).get("rmse")), fmt(q.get("twin_min", {}).get("mae")), fmt(q.get("twin_min", {}).get("mean_error")), fmt(rank.get("mean_true_regret_of_predicted_best")), pct(rank.get("pairwise_accuracy_excluding_true_ties")), fmt(br.get("rmse"))])
        c = s.get("calibration_diagnostics", {}) or {}; p = c.get("pooled", {})
        calrows.append(jkey(j) + [esc(c.get("status", "absent")), fmt(c.get("posterior_ready")), fmt(c.get("last_refresh_numerically_certified")), fmt(c.get("frozen_threshold")), pct(c.get("coverage")), fmt(p.get("mean_finite_half_width")), pct(p.get("infinite_half_width_fraction")), pct(p.get("native_dose_usable_fraction")), fmt(p.get("mean_bc_multiplier")), pct(c.get("band_refusal_rate"))])
        for ctx, v in c.get("strata", {}).items():
            calctx.append(jkey(j) + [LABEL.get(ctx, esc(ctx)), pct(v.get("coverage")), fmt(v.get("mean_finite_half_width")), fmt(v.get("mean_absolute_residual")), pct(v.get("native_dose_usable_fraction")), fmt(v.get("mean_bc_multiplier"))])
    means = []
    for s in a.get("method_summary", []):
        if s.get("split") != "final" or s.get("context") != "pooled": continue
        cells = [esc(s["retention"]), LABEL.get(s["arm"], esc(s["arm"]))]
        for key in ("twin_min_rmse", "bellman_residual_rmse", "action_ranking_regret"):
            v = s.get("metrics", {}).get(key, {})
            cells.append(fmt(v.get("mean")) + " ± " + fmt(v.get("sample_sd")))
        means.append(cells)
    d["critic"] = '<p>These means and sample standard deviations summarize five per-replicate diagnostics; they do not pool errors before taking an RMSE.</p>' + table(["Retention", "Arm", "Policy-Q RMSE mean ± SD", "Bellman residual RMSE mean ± SD", "Six-action ranking regret mean ± SD"], means)
    d["critic"] += details("All 30 per-replicate critic and ranking diagnostics", table(["Retention", "Replicate", "Arm", "Status", "Q1 RMSE", "Q2 RMSE", "Min-Q RMSE", "Min-Q MAE", "Min-Q bias", "Six-action ranking regret", "Pairwise accuracy", "Bellman residual RMSE"], rows))
    d["calibration"] = table(["Retention", "Replicate", "Arm", "Status", "Ready", "Last numerical certificate", "Frozen threshold", "Query residual coverage", "Finite half-width", "Infinite width", "Dose usable", "Mean BC multiplier", "Band refusal"], calrows)
    d["calibration_contexts"] = details("Native residual coverage and width in each regime/load context", table(["Retention", "Replicate", "Arm", "Context", "Coverage", "Finite half-width", "Absolute residual", "Dose usable", "BC multiplier"], calctx))
    d["refreshes"] = '<p>The native finite-width multiplier is <code>1 + 0.5 × width / (width + residual_scale)</code>. An invalid refresh keeps the previous valid reference. The tables read the saved final reference; they do not refit calibration on diagnostic queries.</p>'
    d["refreshes"] += details("Saved refresh audit: local evidence and imported-record limits", table(["Retention", "Replicate", "Arm", "Host", "Refresh records", "Numerically certified", "Inputs valid"], [jkey(j) + [esc(j.get("host")), fmt((j.get("refresh_audit") or {}).get("count"), 0), fmt((j.get("refresh_audit") or {}).get("numerically_certified"), 0), fmt((j.get("refresh_audit") or {}).get("inputs_valid"))] for j in jobs if j["job"]["arm"] == "bca"]))
    d["midpoint"] = details("All midpoint paired context contrasts", comparison_table([c for c in a.get("paired_comparisons", []) if c["split"] == "validation"]))
    midrows = []
    for j in jobs:
        v = get_summary(j, "validation").get("pooled", {})
        midrows.append(jkey(j) + [fmt(v.get("mean_return")), pct(v.get("failure_rate")), pct(v.get("high_action_fraction"))])
    d["midpoint"] += table(["Retention", "Replicate", "Arm", "50k pooled return", "Failure", "Above-knee steps"], midrows)
    d["interpretation"] = '<p><b>Intervention utility:</b> the pooled terminal point estimates favor TD3+BC in all three conditions. This does not identify the causal source of a gap: the native objective, optimization budget, retained data, intervention placement and host differences can all matter. The fixed controls use known model information and reveal room for improvement without providing an equal-information learning comparison.</p><p><b>Calibration behavior:</b> finite, numerically accepted bands and usable native doses do not imply accurate long-horizon critics or critical-group coverage. Below-nominal empirical residual coverage on the designed query bank describes this shifted diagnostic law and frozen reference; it is not a standalone theorem counterexample. Large policy-Q error and a small one-step residual can coexist.</p><p><b>Diagnostic utility:</b> the suite exposes these distinctions and localizes context effects. That is useful attribution evidence, not proof of methodological novelty, optimal intervention placement, impossibility, or generalization to other environments.</p><p>A next comparison should be declared before new outcomes: remove the host/arm confound with a common execution environment, use independent post-training calibration diagnostics, and separate intervention-placement and dose ablations from training-budget changes. A critical-specific calibration design would need a separately specified sampling law and adequate independent critical observations; these frozen banks cannot supply that certificate retrospectively.</p>'
    prov = {k: a.get(k) for k in ("source_run", "generated_utc", "config", "provenance", "evidence", "limitations") if k in a}
    env = a.get("provenance", {}).get("environment", {})
    execution = execution or {}; titan = titan or {}
    d["provenance"] = table(["Execution record", "Observed / recorded setting"], [["Zeus hardware", esc(execution.get("processor", "unavailable")) + "; " + fmt(execution.get("logical_processors"),0) + " logical processors"], ["Zeus launch plan", fmt(execution.get("workers"),0) + " CPU workers; " + esc(execution.get("priority", "unknown")) + " priority; cores " + esc(str(execution.get("worker_cores"))) + "; numeric threads " + fmt(execution.get("numeric_threads"),0)], ["Zeus runtime", esc(env.get("python")) + "; " + esc(env.get("platform"))], ["Zeus package versions", esc(", ".join(f"{k} {v}" for k,v in env.get("packages",{}).items()))], ["Titan imported execution", esc(titan.get("actual_training_backend", "cpu")) + "; " + esc(titan.get("backend_decision",{}).get("environment", "full environment not supplied"))], ["Titan fixture decision", esc(titan.get("backend_decision",{}).get("reason", "See imported source record."))], ["Freeze provenance", esc(a.get("provenance",{}).get("created")) + "; commit " + esc(env.get("git_revision"))]])
    d["provenance"] += raw_details("Frozen configuration, execution versions, hashes and evidence limits", prov)
    d["provenance"] += raw_details("Initial execution plan and imported Titan backend decision", {"initial_zeus_execution_plan":execution,"titan_backend_decision":titan.get("backend_decision")})
    d["provenance"] += details("Per-job source and checkpoint records", '<pre>' + esc(json.dumps([{k: j.get(k) for k in ("job_id", "job", "host", "backend", "source", "checkpoint", "initial_native_sha256", "initial_prng_sha256", "final_prng_sha256")} for j in jobs], indent=2, ensure_ascii=False)) + '</pre>')
    commands = '''# From BCA root; use a NEW path for a new scientific run.
python -B -m experiments.thrusterchain_diagnostics.full_run config --output NEW_CONFIG.json
python -B -m experiments.thrusterchain_diagnostics.full_run prepare --output NEW_RUN --config NEW_CONFIG.json
# Three separately launched, bounded CPU workers (indices 0, 1 and 2):
python -B -m experiments.thrusterchain_diagnostics.full_run run --output NEW_RUN --worker-index 0 --workers 3
python -B -m experiments.thrusterchain_diagnostics.full_run oracle --output NEW_RUN
python -B -m experiments.thrusterchain_diagnostics.full_run evaluate --output NEW_RUN
python -B -m experiments.thrusterchain_diagnostics.full_run status --output NEW_RUN

# Presentation only: saved verified aggregate, no training or rollout.
python -B update_training_guide.py --aggregate ../analysis/aggregate.json --base PATH_TO_EXISTING_DIAGNOSTICS.html --output NEW_RESULTS.html'''
    d["commands"] = '<pre>' + esc(commands) + '</pre>'
    return d

class Scan(HTMLParser):
    def __init__(self):
        super().__init__(); self.ids = []; self.links = []; self.external = []
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a: self.ids.append(a["id"])
        if tag == "a" and a.get("href", "").startswith("#"): self.links.append(a["href"][1:])
        for key in ("src", "href"):
            if tag != "a" and a.get(key, "").startswith(("http:", "https:", "//")):
                self.external.append([tag, key, a[key]])

def verify(base, output, node):
    s = Scan(); s.feed(output)
    missing = sorted(set(s.links) - set(s.ids))
    duplicates = [k for k, n in Counter(s.ids).items() if n > 1]
    scripts = lambda t: re.findall(r'<script\b[^>]*>.*?</script\s*>', t, re.S | re.I)
    svgs = lambda t: re.findall(r'<svg\b.*?</svg\s*>', t, re.S | re.I)
    oldscripts, newscripts = scripts(base), scripts(output)
    oldsvgs, newsvgs = Counter(svgs(base)), Counter(svgs(output))
    preserved_scripts = all(newscripts.count(block) >= oldscripts.count(block) for block in set(oldscripts))
    preserved_svg = all(newsvgs[k] >= n for k, n in oldsvgs.items())
    oldscan = Scan(); oldscan.feed(base)
    oldids = set(oldscan.ids).issubset(set(s.ids))
    if missing or duplicates or s.external or not preserved_scripts or not preserved_svg or not oldids:
        raise ValueError(f"HTML structural check failed: missing={missing}, duplicates={duplicates}, external={s.external}, oldscripts={preserved_scripts}, svg={preserved_svg}, ids={oldids}")
    syntax_count = 0
    if node:
        with tempfile.TemporaryDirectory(prefix="thruster-guide-", dir=HERE) as folder:
            for i, block in enumerate(newscripts):
                opening, content = block.split(">", 1)
                if 'type="application/json"' in opening or "type='application/json'" in opening: continue
                content = re.sub(r'</script\s*>$', '', content, flags=re.I)
                p = Path(folder) / f"script-{i}.js"; p.write_text(content, encoding="utf-8")
                subprocess.run([str(node), "--check", str(p)], check=True, capture_output=True)
                syntax_count += 1
            fixture = Path(folder) / "filters.js"
            fixture.write_text("""const assert = require('assert');
const elements = {'tr-retention-filter': {value:'all',addEventListener(){}},'tr-context-filter': {value:'pooled',addEventListener(){}},'tr-filter-count': {textContent:''}};
const rows = []; for(const r of ['full','rare','zero']) for(const c of ['pooled','stall_low','stall_high','monotone_low','monotone_high']) for(let i=0;i<10;i++) rows.push({dataset:{trRetention:r,trContext:c},hidden:false});
global.document = {getElementById: id=>elements[id], querySelectorAll: ()=>rows};
const script = """ + json.dumps(SCRIPT) + ";\n" + """
for(const r of ['all','full','rare','zero']) for(const c of ['all','pooled','stall_low','stall_high','monotone_low','monotone_high']) {
elements['tr-retention-filter'].value=r; elements['tr-context-filter'].value=c; eval(script);
assert.equal(rows.filter(x=>!x.hidden).length, (r==='all'?3:1)*(c==='all'?5:1)*10);
}
console.log('24 filter combinations passed');
""", encoding="utf-8")
            subprocess.run([str(node), str(fixture)], check=True, capture_output=True)
    return {"ids": len(s.ids), "internal_links": len(s.links), "missing_internal_links": missing, "duplicate_ids": duplicates, "existing_ids_preserved": oldids, "existing_script_blocks": len(oldscripts), "existing_script_blocks_preserved": preserved_scripts, "existing_svg_blocks": sum(oldsvgs.values()), "existing_svg_blocks_preserved": preserved_svg, "output_svg_blocks": sum(newsvgs.values()), "external_asset_references": s.external, "javascript_syntax_checked": syntax_count, "new_filter_mock_dom_combinations": 24 if node else 0, "browser_render_navigation_console_verified": False}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--aggregate", type=Path, default=HERE.parent / "analysis" / "aggregate.json")
    p.add_argument("--base", type=Path, default=BASE)
    p.add_argument("--template", type=Path, default=HERE / "training_results_section.html")
    p.add_argument("--output", type=Path, default=HERE / "ThrusterChain_diagnostics_results.html")
    p.add_argument("--allow-partial", action="store_true")
    p.add_argument("--method-audit", type=Path, default=HERE.parent / "method_audit" / "evidence.json")
    p.add_argument("--stage-map", type=Path, default=HERE.parent / "module_fixtures" / "verification_v2" / "stage_map.json")
    p.add_argument("--fixtures", type=Path, default=HERE.parent / "module_fixtures" / "verification_v2" / "fixture_results.json")
    p.add_argument("--penalty-alignment", type=Path, default=HERE.parent / "penalty_alignment" / "penalty_alignment.json")
    p.add_argument("--execution-plan", type=Path, default=HERE / "execution_plan.json")
    p.add_argument("--titan-report", type=Path, default=HERE.parent / "inputs" / "titan_results.json")
    p.add_argument("--node", type=Path, default=Path(r"C:\Users\David\AppData\Local\hermes\node\node.exe"))
    args = p.parse_args()
    if args.output.exists() or args.output.resolve() == args.base.resolve():
        raise FileExistsError("Output must be a new file; canonical guide and existing outputs are preserved")
    if args.output.with_suffix(".build.json").exists():
        raise FileExistsError("Build manifest exists; choose a new output filename")
    a = json.loads(args.aggregate.read_text(encoding="utf-8")); validate(a, args.allow_partial)
    base = args.base.read_text(encoding="utf-8")
    if 'id="training-results"' in base:
        raise ValueError("Base already contains training results; use the preserved diagnostics guide")
    audit_paths = [args.method_audit, args.stage_map, args.fixtures, args.penalty_alignment, args.execution_plan, args.titan_report]
    audit_data = []
    for path in audit_paths:
        if not path.exists() and not args.allow_partial:
            raise FileNotFoundError(f"Final advisor-stage report requires {path}")
        audit_data.append(json.loads(path.read_text(encoding="utf-8")) if path.exists() else None)
    section = Template(args.template.read_text(encoding="utf-8")).substitute(sections(a, args.allow_partial, *audit_data))
    anchor = '<section id="diagnostics-suite" aria-labelledby="diagnostics-heading">'
    if base.count(anchor) != 1: raise ValueError("Expected diagnostics section anchor not found exactly once")
    output = base.replace(anchor, section + '\n' + anchor, 1)
    nav_anchor = '<li class="nv-sec"><a href="#diagnostics-suite"'
    if output.count(nav_anchor) != 1: raise ValueError("Expected diagnostics contents link not found")
    output = output.replace(nav_anchor, '<li class="nv-sec"><a href="#training-results" data-unit="training-results">BCA stage verification &amp; training results</a></li>\n' + nav_anchor, 1)
    # Keep historical prose and measurements verbatim, adding conspicuous adjacent corrections.
    for ident in ("lg-ctrl", "iw-explore", "iw-results", "toy-tc-resultf"):
        heading = re.compile(r'(<h([1-6])\b[^>]*\bid="' + re.escape(ident) + r'"[^>]*>.*?</h\2>)', re.S)
        note = '<p class="note" id="tr-correction-' + ident + '"><b>21 October 2026 contract correction:</b> historical command-weighted/A2 returns below remain descriptive, but the claimed exact command ratio and its calibration interpretation omit clipping-created mass. That claim requires correction and retesting; it does not affect the separate current-law context weighting or the new native BCA run. <a href="#tr-stage-I11">See the source trace and reproduced failures.</a></p>'
        output, count = heading.subn(lambda m: m.group(1) + note, output, count=1)
        if count != 1: raise ValueError("Missing historical correction location: " + ident)
    output = output.replace('<section id="version-history">', '<section id="version-history"><p class="note" id="tr-correction-history"><b>Current audit:</b> the preserved historical command-likelihood claims are qualified by the <a href="#tr-stage-I11">21 October mixed-measure correction</a>. Saved numerical returns are retained; the associated exact-ratio calibration claim requires repair and retesting.</p>', 1)
    # Remove only optional remote font resources; offline system-font fallbacks already exist.
    output = re.sub(r'<link\b[^>]*href="https://fonts\.(?:googleapis|gstatic)\.com[^>]*>\s*', '', output)
    output = output.replace('</body>', '<script id="tr-results-filter-script">' + SCRIPT + '</script>\n</body>', 1)
    checks = verify(base, output, args.node if args.node.exists() else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as f: f.write(output)
    evidence = {"schema": "thrusterchain-results-guide-build-v1", "aggregate_complete": a.get("complete"), "internal_partial_preview": args.allow_partial, "job_count": len(a["jobs"]), "sources": {str(path): sha(path) for path in [args.base, args.aggregate, args.template, Path(__file__)] + [p for p in audit_paths if p.exists()]}, "output": str(args.output.resolve()), "output_sha256": sha(args.output), "output_bytes": args.output.stat().st_size, "checks": checks}
    args.output.with_suffix(".build.json").write_text(json.dumps(evidence, indent=2) + '\n', encoding="utf-8")
    print(json.dumps(evidence, indent=2))

if __name__ == "__main__":
    main()
