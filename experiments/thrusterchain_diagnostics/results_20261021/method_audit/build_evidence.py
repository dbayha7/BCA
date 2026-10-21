"""Read-only source/result extraction; writes only this task's audit JSON."""
from pathlib import Path
import hashlib
import json
import statistics
from datetime import datetime, timezone

ROOT = Path(r"C:\Users\David\dev\BCA")
OUT = Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def source(path, start, end):
    return {"path": path, "absolute_path": str(ROOT / path), "start": start, "end": end,
            "sha256": sha(ROOT / path)}

def stage(ident, title, premise, inputs, outputs, purpose, lines, invariant, test, takeaway, depends):
    return {"id": ident, "branch": "native_td3_bc_bca", "title": title, "premise": premise,
            "inputs": inputs, "outputs": outputs, "purpose": purpose,
            "pseudocode": [{"line": f"{ident}.{i+1}", "text": text, "source": source(p, a, b)}
                           for i, (text, p, a, b) in enumerate(lines)],
            "invariant": invariant, "isolated_test": test,
            "status": "source_verified; independent_fixture_status_separate",
            "evidence": "Actual working-tree source inspected; no new training in this audit.",
            "takeaway": takeaway, "depends_on": depends}

P = "algorithms/td3_bc_bca.py"
stages = [
stage("N1", "Reserve calibration episodes", "Trajectory membership is known; reservation precedes fitting.",
      "offline transitions and episode IDs", "matched training complement and held-out bank",
      "Prevent direct training/calibration row overlap.",
      [("reserve whole episodes; choose thinned rows; exclude reserved episode rows from training", "calibration/reference.py",49,122),
       ("validate training indices and choose training pool",P,85,111)],
      "No train/calibration overlap; no unreported top-ups.",
      "Check IDs and perturb held-out values without changing training normalization; existing new-suite tests cover episode separation.",
      "Procedural hold-out does not itself prove independence after repeated adaptive bank reuse.", []),
stage("N2", "Compute one-step TD residual", "Host discount, terminal masks and action scales are correct.",
      "recorded transition, online critic, target actor/critic, independent target-noise key", "y and detached q",
      "Define exactly which error is calibrated.",
      [("a_next = clip(pi_target(s_next) + clip(N(0,sigma^2), -c, c), -1, 1)",P,146,150),
       ("y = stopgrad(r + (1-done)*gamma*min Q_target(s_next,a_next))",P,151,156),
       ("q = stopgrad(min Q_online(s,a_recorded)); residual=y-q",P,162,174)],
      "Terminal target equals reward; no gradients through target/critic in calibrator fit.",
      "Fixed critic/transition analytic target, terminal mask, and gradient checks.",
      "This residual is not error against true long-horizon Q.",["N1"]),
stage("N3", "Fit positive residual scale", "Training data alone fit scale; one eta per row.",
      "training residual, scale network, residual unit, exponential bootstrap weights", "updated calibrator and unit",
      "Learn relative local width before held-out threshold selection.",
      [("eta=softplus(network(s,a))+0.01", "calibration/network.py",16,31),
       ("loss=(sum(normalized_exp_weights*soft_coverage(y/u,q/u,eta))-(1-alpha))^2 + width_penalty*sum(weights*eta^2)",P,180,195),
       ("u_new=ema*u+(1-ema)*max(std(y-q),1e-6)",P,196,197)],
      "Positive finite scale; detached y,q; gradients update calibrator only.",
      "Fixed low/high residual fixtures, positivity and gradient isolation.",
      "Scale loss success is not proof that widths rank harmful actions.",["N2"]),
stage("N4", "Freeze and score held-out rows", "The scale is frozen for this threshold computation.",
      "held-out residuals, scale params, residual unit", "frozen standardized scores/reference",
      "Calibrate the fitted scale against held-out residuals.",
      [("scores=abs(y-q)/(max(eta_frozen,1e-6)*u_frozen)","calibration/reference.py",152,179),
       ("refresh valid -> replace reference; invalid -> preserve old state",P,257,278)],
      "Aligned finite scores; no evaluation labels; invalid refresh cannot silently replace reference.",
      "Analytic standardized residual and invalid/empty/NaN refresh cases.",
      "Bank reuse and adaptive actor feedback must be analyzed separately.",["N1","N2","N3"]),
stage("N5", "Select posterior threshold", "Scores are representative under the stated calibration/test law.",
      "scores, alpha=.1,beta=.95,1000 exponential draws; native weights/testmass=1", "R, empirical/posterior components, ESS, refusal",
      "Choose a conservative score cutoff under the implemented WBCP model.",
      [("sort scores; empirical=weighted (1-alpha) bank quantile", "calibration/wbcp.py",83,102),
       ("each draw: bankmass=w_i*E_i; testmass=w_query*E_query at worst case; find first crossing", "calibration/wbcp.py",43,54),
       ("posterior=beta order statistic of crossings; R=max(empirical,posterior)", "calibration/wbcp.py",104,117)],
      "Common weight/testmass scaling invariant; impossible crossings return infinity.",
      "Independent tiny-bank quantile/query-atom fixture; common scaling; refusal cases.",
      "Correct arithmetic does not establish true-Q or adaptive-policy coverage.",["N4"]),
stage("N6", "Convert threshold to width", "Frozen scale and unit remain valid.",
      "R, frozen eta at recorded action, frozen unit", "width U and support/readiness",
      "Restore original residual units.",
      [("U=R*max(eta_frozen(s,a_recorded),1e-6)*u_frozen", "calibration/dose.py",65,87),
       ("not ready -> U=0 and unit=1", "calibration/dose.py",90,95)],
      "One width per row; finite positive scale; explicit infinity and readiness.",
      "Known threshold/eta/unit product and not-ready fixture.",
      "Recorded-action width does not automatically transfer to actor actions.",["N5"]),
stage("N7", "Convert width to detached multiplier", "blend in[0,1], positive unit; support semantics fixed.",
      "U, unit, blend, validity", "per-row cloning multiplier",
      "Bound intervention strength relative to the residual unit.",
      [("finite supported: m=1+blend*U/(U+u); unsupported: m=1", "calibration/dose.py",44,61),
       ("stopgrad all returned dose values", "calibration/dose.py",62,62)],
      "m in[1,1+blend]; m=1 for native infinite/unsupported width; no dose gradient.",
      "Zero/finite/infinite/invalid width, monotonicity, scale invariance, detached gradients.",
      "IID does not imply low residual or low m; high m does not prove useful intervention.",["N6"]),
stage("N8", "Apply multiplier to actor cloning loss", "Same host initialization, batch/noise/update schedule in paired comparison.",
      "recorded action, actor action, online Q1, m", "native critic/actor updates",
      "Connect uncertainty estimate to an actual learning intervention.",
      [("call native TD3+BC update with m, keeping native critic objective",P,295,310),
       ("actor_loss=-alpha/stopgrad(mean(abs(Q1)))*mean(Q1)+mean(stopgrad(m)*mean_action((pi-a)^2))", "algorithms/td3_bc.py",128,139),
       ("update actor and target networks only on policy_freq steps", "algorithms/td3_bc.py",141,164)],
      "Per-row multiplier before row mean; gradients through actor only, not m; native target timing.",
      "All-one equality; nonuniform hand-computed actor loss; exact host equivalence.",
      "Cloning poor recorded actions can harm return; compare to equal-strength constant intervention.",["N7"]),
stage("N9", "Validate and accept update", "Inputs and proposed state are finite/valid.",
      "proposed carry and validity metrics", "accepted state/PRNG/counter or unchanged carry",
      "Prevent invalid computations from appearing as successful training progress.",
      [("if inputs_valid: accept(next_rng,new_state,it+1); else: preserve carry",P,327,343)],
      "Invalid update advances neither state, RNG nor accepted update count.",
      "Inject invalid input and compare every carry leaf exactly.",
      "Checkpoint/recovery correctness is separately audited and cannot substitute for scientific success.",["N8"])
]

iw=json.loads((OUT/"iw_stages.json").read_text(encoding="utf-8"))
for i,s in enumerate(iw["stages"]):
    refs=s.get("sources",[])
    parsed=[]
    for ref in refs:
        p,ln=ref.rsplit(":",1)
        try: a=int(ln.split("-")[0])
        except ValueError: continue
        parsed.append(source(p,a,a))
    stages.append({"id":f"I{i+1}","original_id":s["id"],"branch":"importance_weighted_variants",
        "title":s["id"].replace("_"," "),"premise":" ".join(s.get("premises",[])),
        "inputs":s.get("inputs","See code-derived pseudocode and source references."),
        "outputs":s.get("outputs","See code-derived pseudocode."),
        "purpose":s.get("purpose",s["id"].replace("_"," ")),
        "pseudocode":[{"line":f"I{i+1}.1","text":s["pseudocode"],"source":parsed[0] if parsed else None}],
        "sources":parsed,"invariant":" ".join(s.get("invariants",[])),
        "isolated_test":s.get("tests",s.get("test_references","See independent module_fixtures report.")),
        "status":"source_verified; inspected_existing_tests_not_executed_by_this_audit",
        "evidence":s.get("evidence","Actual source inspection; fixture outcomes separately reported."),
        "takeaway":" ".join(s.get("limitations",[])),"depends_on":s.get("dependencies",[])})

command_path="outputs/shift_iw/explore/channel/command_band_check.py"
stages.append({"id":"I11","branch":"exploratory_command_weighted_lower_edge",
 "title":"Check the action probability measure before calling the ratio exact",
 "premise":"Density ratios must include every continuous and singular component of the actual sampled source and target laws.",
 "inputs":"clipped Gaussian actor spread, clipped action, episode-conditioned behavior law",
 "outputs":"ratio and refusal classification for each target measure component",
 "purpose":"Prevent target-only atoms from being treated as low-weight supported actions.",
 "pseudocode":[
  {"line":"I11.1","text":"sample z=clip(.2*N(0,1),-.5,.5); action=clip(actor+z,-1,1)","source":source(command_path,261,264)},
  {"line":"I11.2","text":"implemented spread density: finite Gaussian density only if all |action-actor|<.5; otherwise -infinity","source":source(command_path,80,85)},
  {"line":"I11.3","text":"implemented logratio=log(current_ratio)+logspread+logZ-logbehavior when behavior density finite","source":source(command_path,130,136)},
  {"line":"I11.4","text":"negative-infinite logmass maps to floor without refusal; positive-infinite logmass refuses","source":source(command_path,160,168)}],
 "invariant":"Target mass on a source-null atom cannot receive a finite supported density ratio.",
 "isolated_test":"Actor centre0 with first action+.5; actor first-coordinate centre.8 with first action1; inspect clipped-noise hyperplane and action-boundary mass.",
 "status":"scientific_contract_FAIL_independently_reviewed_and_reproduced",
 "contract_status":"FAIL",
 "evidence":"Source independently confirmed by final_independent_review and reproduced by two scientific_review verification_v2 negative fixtures. Harness PASS, scientific contract FAIL; no source edits or policy runs.",
 "takeaway":"Historical policy gains remain descriptive. Globally exact command-ratio calibration claim is blocked until mixed-measure law is repaired and rechecked; native/current-law IW is unaffected by this specific defect.",
 "depends_on":["I1","I2","I3","I10"]})

fixture_path=OUT.parent/"module_fixtures/verification_v2/fixture_results.json"
if not fixture_path.exists():
    fixture_path=OUT.parent/"module_fixtures/fixture_results.json"
fixtures=json.loads(fixture_path.read_text()) if fixture_path.exists() else None
fixture_map={
 "N4":["iid_same_target_low_high_residual"],
 "N5":["hand_query_atom_crossings","independent_tail_risk_posterior","controlled_iw_increase_and_decrease"],
 "N7":["native_dose_bound_direction_and_infinite_fallback","dose_is_detached_from_width_gradient"],
 "N8":["known_oracle_alignment_and_counterexample"],
 "N9":["invalid_reference_does_not_become_ready"],
 "I1":["known_population_ratio"],"I2":["iid_classifier_class_size_correction"],
 "I3":["hand_query_atom_crossings"],
 "I5":["common_rescaling_invariance","bank_only_normalization_exception","concentration_ess_and_no_native_ratio_clipping"],
 "I6":["all_one_weights_reduce_to_native","independent_tail_risk_posterior","controlled_iw_increase_and_decrease"],
 "I7":["zero_support_is_refused_not_epsilon_repaired","query_mass_table_and_conservative_lookup"],
 "I8":["identical_thresholds_different_refusal_policy","iql_refusal_is_host_specific"],
 "I11":["exploratory_noise_clip_atom","exploratory_action_clip_atom"]}
for s in stages:
    s["fixture_ids"]=fixture_map.get(s["id"],[])
    s["fixture_scope_note"]="Mapped fixtures establish their own narrow stated premise, not every invariant of this stage. An empty mapping is untested by this isolated harness."
    if fixtures:
        ids=set(s["fixture_ids"])
        s["fixture_outcomes"]=[{"id":c["id"],"status":c["status"],"harness_status":c.get("harness_status",c["status"]),"contract_status":c.get("contract_status",c["status"]),"interpretation":c["interpretation"]} for c in fixtures["cases"] if c["id"] in ids]

v5path=ROOT/"outputs/shift_iw/v5/summary.json"
v5=json.loads(v5path.read_text())
source_files={f:{"sha256":sha(ROOT/f),"matches_v5_record":sha(ROOT/f)==h} for f,h in v5["code_sha256"].items()}
for f,h in iw["hashes"].items():
    if (ROOT/f).is_file(): source_files[f]={"sha256":sha(ROOT/f),"iw_audit_hash":h}

study_paths=["outputs/weighted_conformal/2026-09-29-v2/results.json",
 "outputs/weighted_conformal/2026-09-29-v2/independent_verification.json",
 "outputs/shift_iw/v5/summary.json","outputs/shift_iw/explore/RESULTS.json",
 "outputs/shift_iw/explore/channel/results/train/SCORE.json",
 "outputs/shift_iw/explore/channel/results/train_f1/SCORE.json",
 "outputs/shift_iw/explore/channel/results/train_est/SCORE.json",
 "outputs/shift_hosts/results.json"]
evidence_files={f:{"sha256":sha(ROOT/f),"bytes":(ROOT/f).stat().st_size} for f in study_paths}
paired={}
for name in ["train_f1","train_est"]:
    p=ROOT/f"outputs/shift_iw/explore/channel/results/{name}/SCORE.json"
    data=json.loads(p.read_text())
    paired[name]={}
    for quality,rows in data.items():
        kept={k:v for k,v in rows.items() if ("| strong" in k and k.startswith(("A2","A3"))) or k=="marks" or k.startswith("V")}
        for k,v in kept.items():
            if isinstance(v,dict) and "J" in v:
                assert abs(statistics.mean(v["J"]["per_seed"])-v["J"]["mean"])<1e-10
        paired[name][quality]=kept

passmarks={}
for name in ["fallback_rerun","estimated_weights"]:
    f=f"outputs/shift_iw/explore/channel/PASSMARKS_{name}.md"
    p=ROOT/f; digest=sha(p)
    passmarks[f]={"sha256":digest,"sidecar_matches":digest==Path(str(p)+".sha256").read_text().split()[0]}
for name in ["command_band_check.py","command_le_train_f1.py","command_le_train_est.py"]:
    f=f"outputs/shift_iw/explore/channel/{name}"
    source_files[f]={"sha256":sha(ROOT/f)}

record={
 "schema":"bca-method-audit-v1","created_utc":datetime.now(timezone.utc).isoformat(),
 "repository":{"path":str(ROOT),"head":"b007052a2494dc651ac00469fa9e922b287cd1e5",
  "working_tree_untracked":[".claude/","bca/","experiments/thrusterchain_diagnostics/","outputs/shift_hosts/","outputs/shift_hosts_v4/RESULTS_SUMMARY.md","outputs/shift_hosts_v4/operations/"],
  "source_identity":"Current working-tree bytes hashed; do not substitute old cloud checkout.",
  "v5_recorded_source_count":len(v5["code_sha256"]),"v5_all_recorded_hashes_match":all(sha(ROOT/f)==h for f,h in v5["code_sha256"].items())},
 "source_files":source_files,"stages":stages,"iw_versions":iw["versions"],
 "evidence_files":evidence_files,"passmarks":passmarks,
 "studies":{"historical_synthetic":json.loads((ROOT/study_paths[0]).read_text()),
  "historical_synthetic_verification":json.loads((ROOT/study_paths[1]).read_text()),
  "v5_claims_and_marks":{q:{"claims":v["claims"],"marks":v["marks"]} for q,v in v5["rows"].items()},
  "exploratory_paired":paired},
 "new_findings":[
  {"id":"F1","status":"source_verified","finding":"Native infinite threshold gives multiplier1; historical per-query IW gives maximum1.5 atblend.5. Refusal mapping confounds an unqualified all-one weighting comparison.","sources":["calibration/dose.py:65-69","experiments/shift_iw/learners_iw.py:655-683"]},
  {"id":"F2","status":"scientific_contract_FAIL_independently_reviewed_and_reproduced","finding":"Exploratory command target samples clipped Gaussian/action atoms but spread_logdensity evaluates only interior continuous density. At centre0,a=.5 logratio becomes-inf and threshold lookup floors it rather than refusing; at action boundary finite density ignores target-only atom. Do not call this globally exact mixed-measure ratio.","fixture_ids":["exploratory_noise_clip_atom","exploratory_action_clip_atom"],"sources":["outputs/shift_iw/explore/channel/command_band_check.py:80-85","outputs/shift_iw/explore/channel/command_band_check.py:130-136","outputs/shift_iw/explore/channel/command_band_check.py:160-168","outputs/shift_iw/explore/channel/command_band_check.py:261-264"]},
  {"id":"F3","status":"contract_gap","finding":"A2 conditions behavior density on privileged episode random share, A3 estimates observable-state/action odds. They need not agree pointwise; a common marginal target is possible after explicit derivation. Overall support agreement is not identification of conditional support.","sources":["outputs/shift_iw/explore/channel/command_band_check.py:120-136","outputs/shift_iw/explore/channel/command_le_train_est.py:200-215"]},
  {"id":"F4","status":"source_verified","finding":"Exploratory command branch adds exponent clamps and query floor; core current-law IW does not clip or temper weights. Track clipping/ESS/refusals separately and do not transfer coverage claims.","sources":["calibration/wbcp.py:83-85","outputs/shift_iw/explore/channel/command_band_check.py:147-165","outputs/shift_iw/explore/channel/command_le_train_f1.py:74","outputs/shift_iw/explore/channel/command_le_train_est.py:205"]}
 ],
 "decisions":[
  {"method":"native_td3_bc_bca","decision":"inconclusive_general_benefit","reason":"One-step calibration arithmetic and full-run evidence must be separated from trueQ/harm ranking; current30 suite has noIW arm."},
  {"method":"historical_weighted_conformal_reference","decision":"advance_reference_only","reason":"Known-ratio scalar synthetic evidence; explicitly not RL integration and superseded formulation."},
  {"method":"current_law_capped_IW","decision":"redesign_intervention_before_more_training","reason":"Coverage gate passes do not deliver consistent policy gain; poor recorded actions amplify cloning harm."},
  {"method":"A2_support_aware_lower_edge","decision":"conditional_candidate_blocked_on_ratio_contract_repair","reason":"Existing paired gains survive count-matched untargeted control, but privileged law and clipped atom defect preclude global exact-ratio calibration claim."},
  {"method":"A3_estimated_support_lower_edge","decision":"redesign_estimator_and_contract","reason":"Criticalsupport recall~5%mediumexpert and50%gainretentioncriterion failsbothqualities; expert signal remains exploratory."}],
 "proposed_ablation":{"status":"proposal_not_launched_freeze_required","first_gate":"Complete deterministic stages and repair/derive mixedmeasure and observable target ratio contracts before policy jobs.",
  "initial_quality":"one predeclared quality after observable-law repair; not selected using new test returns","optional_extension_quality":"the other of medium-expert/expert, separately preregistered","fresh_paired_seeds":5,"actor_updates":20000,"budget_source":"experiments/shift/learners.py:138-139; experiments/shift_iw/lower_edge.py:25","critic":"same frozen critic within pair",
  "arms":["matched unweighted lower edge","oracle ratio under same observable law","estimated ratio under that law","constant/magnitude and refusal matched oracle intervention","constant/magnitude and refusal matched estimated intervention"],
  "proposed_initial_jobs":25,"two_quality_upper_matrix_jobs":50,"primary":"predeclare target-specific return difference versus both matched baseline and strength-matched control; no test checkpoint selection",
  "diagnostics":["finite coverage","refusal rates by reason","critical support recall","false refusal","pertrajectory and row ESS","clippedmass","longhorizon Qerror and opportunity loss ranking"]},
 "limitations":["No new learning launched; source inspected and saved result arithmetic checked only.","Isolated fixture outcomes owned by sibling module_fixtures report; source verification is not test pass.","Sequential exploratory development and multiple comparisons limit causal/confirmatory claims.","Current A2/A3 source and passmarks were hashed and saved result arithmetic checked; every historical checkpoint/input was not reauthenticated against every per-run source manifest. Current source hashes do not alone certify the full historical provenance chain.","Clipped-noise atom contract defect does not erase recorded policy returns, but blocks claiming globally exact ratio calibration.","Current 30-run native aggregate and checkpoint recovery are separately audited; no IWBCA inference from BCA-only results."],
 "tests_this_audit":{"new_training":0,"new_simulation":0,"per_seed_means_verified":True,"passmark_sidecars_verified":True,"source_hashes_verified":True},
 "independent_fixtures":{"path":str(fixture_path),"sha256":sha(fixture_path),"summary":fixtures["summary"],"coverage_claim":fixtures["coverage_claim"]} if fixtures else None
}
alignment_path=OUT.parent/"penalty_alignment/penalty_alignment.json"
if alignment_path.exists():
    alignment=json.loads(alignment_path.read_text())
    record["native_penalty_utility"]={"path":str(alignment_path),"sha256":sha(alignment_path),
        **{k:alignment[k] for k in ["source","missing_raw_jobs","summary","inversion_count","total_states","definitions","limitations"]}}
    record["native_penalty_utility"]["audit_clarification"]="Six-candidate frozen readout diagnostic, not actual training weights on those candidate commands. Native BCA reads its multiplier at recorded actions before weighting logged-row cloning loss. Multiplier ordering is not actual loss/gradient ordering."
(OUT/"evidence.json").write_text(json.dumps(record,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
print(json.dumps({"path":str(OUT/"evidence.json"),"stages":len(stages),"source_files":len(source_files),"v5_source_count":len(v5["code_sha256"]),"all_v5_source_hashes_match":record["repository"]["v5_all_recorded_hashes_match"],"sha256":sha(OUT/"evidence.json")}))
