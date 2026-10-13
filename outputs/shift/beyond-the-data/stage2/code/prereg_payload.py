"""Stage 2: the machine-readable pre-registration (runs/shift_explore/stage2/prereg.json), built by write_prereg.py.
Its human-readable twin is PREREGISTRATION.md (pinned by the code manifest, so the approved hash covers both).

Pins are read here from the other components' code manifests. When one is missing (the detection script or the
stand-ins test not yet locked), the payload says approvable: false and lists missing_pins; approve.py refuses a draft.
"""

import json
from pathlib import Path

import stage2_common as C

VERSION_NOTE = "version numbers count rewrites of prereg.json before approval; earlier versions are kept in prereg_history/"


def _manifest(out_dir):
    p = Path(out_dir) / "code_manifest.json"
    try:
        man = C.read_json(p, required=False)
    except SystemExit:
        return None, None
    return (None, None) if man is None else (man, C.file_sha256(p))


def pins():
    out, missing = {}, []
    out["room"] = dict(C.ROOM_PINNED, harness_copy=str(C.ROOM_OUT / "code" / "harness.py"))
    if C.file_sha256(C.ROOM_OUT / "code_manifest.json") != C.ROOM_PINNED["manifest_sha256"]:
        missing.append("room (the room's code manifest changed)")
    out["fallback_py"] = dict(path=str(C.FALLBACK_PY), sha256=C.FALLBACK_PY_SHA256)
    if C.file_sha256(C.FALLBACK_PY) != C.FALLBACK_PY_SHA256:
        missing.append("fallback_py (changed)")
    man, msha = _manifest(C.STANDINS_OUT)
    if man is None or "standins.py" not in man.get("files", {}):
        out["standins_py"], out["standins_marks_spec_sha256"], out["standins_manifest_sha256"] = None, None, None
        missing.append("standins_py (runs/shift_explore/standins/code_manifest.json not written yet)")
    else:
        path = C.STANDINS_OUT / "code" / "standins.py"
        out["standins_py"] = dict(path=str(path), sha256=man["files"]["standins.py"])
        out["standins_marks_spec_sha256"] = man.get("spec_sha256")
        out["standins_manifest_sha256"] = msha
        if not path.exists() or C.file_sha256(path) != man["files"]["standins.py"]:
            missing.append("standins_py (its pinned copy differs from its manifest)")
    man, msha = _manifest(C.DETECT_OUT)
    if man is None:
        out["detect_module"], out["detect_manifest_sha256"] = None, None
        missing.append("detect_module (runs/shift_explore/detect/code_manifest.json not written yet: the detection "
                       "script is not locked)")
    else:
        module = man.get("module", "detect.py")
        path = C.DETECT_OUT / "code" / module
        out["detect_module"] = dict(path=str(path), sha256=man["files"].get(module), module=module)
        out["detect_manifest_sha256"] = msha
        if not path.exists() or C.file_sha256(path) != man["files"].get(module):
            missing.append("detect_module (its pinned copy differs from its manifest)")
    return out, missing


def check_pins(prereg, pins_fn=None):
    """The pins recorded in prereg.json must equal the pins computed NOW from the other components' code manifests
    (room test, fallback diagnostic, stand-ins, detection script), and none may be missing. Called by approve.py
    before writing an approval (a pin that changed between writing and approving refuses the approval) and by every
    Stage 2 BCA phase (run_stage2.bca_gate). Raises SystemExit naming the pins that differ. pins_fn: a stand-in for
    pins() in the synthetic tests."""
    now, missing = (pins_fn or pins)()
    if missing:
        raise SystemExit(f"refused: pinned components are missing or changed now: {missing}")
    recorded = prereg.payload.get("pins")
    if not isinstance(recorded, dict):
        raise SystemExit("refused: prereg.json records no pins")
    now = C.plain(now)
    differ = sorted(k for k in set(now) | set(recorded) if C.canonical(now.get(k)) != C.canonical(recorded.get(k)))
    if differ:
        raise SystemExit(f"refused: the pinned components changed since prereg.json was written ({differ}); rerun "
                         f"write_prereg.py and approve the new version")
    return now


def payload(version, paths=None):
    paths = paths or C.PATHS
    p, missing = pins()
    man = C.read_json(paths.manifest, required=False)
    if man is None:
        missing.append("code manifest (run write_manifest.py)")
        code = dict(code_manifest_sha256=None, files=None)
    else:
        code = dict(code_manifest_sha256=C.file_sha256(paths.manifest), files=man["files"],
                    repository_files=man.get("repository_files"))
    return dict(
        kind="stage2-preregistration", label=C.LABEL, label_long=C.LABEL_LONG, version=int(version),
        version_note=VERSION_NOTE, method_id=C.METHOD_ID,
        approvable=not missing, missing_pins=missing,
        status=dict(
            seeds_0_4="EXPLORATORY. Post hoc with respect to seeds 0 and 1 of the room test (its futility look was "
                      "seen by the orchestrating agent before this was written) and, more broadly, designed after the "
                      "gate's own results on seeds 0-4 (the stop report, the anatomy and the mechanism readings). "
                      "Nothing on seeds 0-4 supports a claim.",
            seeds_5_9="CONFIRMATORY. Fresh seeds 5-9 (base world, stall slope 1.5): new logs, critics and cloned actors "
                      "made under this pre-registration, every setting frozen from seeds 0-4. Only seeds 5-9 carry the "
                      "confirmatory claim.",
            written_after=dict(
                room_test="the room test (runs/shift_explore/room/spec.json, SHA-256 56bbda24...) was written, "
                          "reviewed, amended (change A1 and the others in its changes_before_any_run) and launched "
                          "before any of its outcomes existed",
                futility_look=("the room test's futility look on seeds 0 and 1 (runs/shift_explore/room/futility.json, "
                               "file time 2026-10-13 04:19 UTC) was seen by the orchestrating agent, not by the writer "
                               "of this pre-registration; the writer read no futility.json and no seed's values.json "
                               "of the room test, and no Stage 1 stand-in, deployment or detection outcome (only "
                               "folder listings, file names and times were seen)"),
                gate=("the gate's stop report (outputs/shift/stop-report/README.md) and the exploratory anatomy and "
                      "mechanism readings on seeds 0-4 were read"),
                known_change="change A1 of the room test (a penalty is read relative to the recorded command at the "
                             "same state) was known and is applied here")),
        method=method(),
        detection_test=detection_test(),
        policy_test=policy_test(),
        deployment_check=deployment_check(),
        full_host=full_host(),
        confirmation=confirmation(),
        comparison_set=comparison_set(),
        run_order=run_order(),
        compute=compute(),
        claims=claims(),
        approval=approval(),
        standing_rules=[
            "No BCA band fit, calibrator fit, threshold, dose or width readout, deployment check or BCA training arm runs "
            "on ThrusterChain data before the user approves this exact prereg.json.",
            "Nothing is written under experiments/, runs/shift/ or runs/shift_explore/room/ (stage2_common.guard_out refuses).",
            "Every output is written once, with a SHA-256 sidecar; every lock is written before any return is computed.",
            "Unit tests run BCA functions only on small synthetic arrays.",
        ],
        deviations=deviations(),
        open_choices=open_choices(),
        revision=revision(),
        pins=p, code=code,
        outputs=str(paths.out),
        plan=dict(path=str(C.PLAN_PATH), sha256=C.file_sha256(C.PLAN_PATH) if C.PLAN_PATH.exists() else None,
                  sections=["2", "3", "4.6", "5"],
                  scratch_copy=str(C.SCRATCH_OOD / "PLAN.md"),
                  scratch_copy_sha256=C.file_sha256(C.SCRATCH_OOD / "PLAN.md")),
        room_spec=dict(path=str(C.ROOM_OUT / "spec.json"), sha256=C.ROOM_PINNED["spec_sha256"],
                       change_A1="changes_before_any_run[0]"),
    )


LOCAL_BRANCH = "claude/main-bca-connection-x4o46s"


def approval():
    """Section 9. Version 6: the local-terminal route is the ONLY mode accepted (the user's choice); the cloud-terminal
    and chat modes stay in the code but refuse (cloud_terminal_allowed and chat_mode_allowed are false)."""
    folder = C.REPO_STAGE2_REL
    return dict(
        primary_mode=C.LOCAL_MODE,
        primary_mode_chosen_by=("the user, who chose to approve by typing at a terminal on the user's own machine, not in "
                                "the cloud session (which gives the user no terminal), and to send the approval "
                                "through the user's own git push"),
        local_terminal_allowed=True,
        cloud_terminal_allowed=False,
        chat_mode_allowed=False,
        accepted_modes=[C.LOCAL_MODE],
        local_terminal=dict(
            folder=folder, script=folder + "/approve_local.py", approval_file=C.LOCAL_APPROVAL_PATTERN,
            sidecar=C.LOCAL_APPROVAL_PATTERN + ".sha256", import_record=C.IMPORT_RECORD_PATTERN,
            one_file_per_fingerprint=("each version's approval has its own file, named after the first 16 characters "
                                      "of its fingerprint, so an approval of an earlier version never blocks a later "
                                      "one, and exactly one commit adds each file"),
            branch=LOCAL_BRANCH,
            committed_files=("prereg.json and its .sha256 (byte-identical to runs/shift_explore/stage2/prereg.json), "
                             "PREREGISTRATION.md, approve_local.py, code_manifest.json and its .sha256, and code/ "
                             "(a copy of every pinned script), plus README.md with the user's steps; later the user's "
                             "approval file and the import record"),
            user_steps=[
                "git fetch origin && git checkout " + LOCAL_BRANCH + " && git pull",
                "read " + folder + "/PREREGISTRATION.md (section 9 for this approval, section 14 for the defaults "
                "approving accepts)",
                "python3 " + folder + "/approve_local.py   (from the repository root; any Python 3.8 or later; "
                "standard library only)",
                "check the fingerprint it prints, then type its first 16 characters",
                "git add " + C.LOCAL_APPROVAL_PATTERN + " " + C.LOCAL_APPROVAL_PATTERN + ".sha256",
                "git commit -m \"Stage 2: approve pre-registration <first 16 characters> (local terminal)\"",
                "git push origin " + LOCAL_BRANCH,
                "once the import record is pushed, check on GitHub that the commit it names is yours",
            ],
            approve_local_checks=("prereg.json matches its .sha256; it is final (approvable) and accepts this mode; "
                                  "the code manifest is the one prereg.json pins and matches its .sha256; every file in "
                                  "code/, PREREGISTRATION.md, approve_local.py next to it and the running script have "
                                  "the SHA-256 the manifest records; no approval of this fingerprint exists; standard "
                                  "input is a terminal; the typed characters equal the first 16 characters of the "
                                  "SHA-256 (spaces at either end ignored, otherwise exact). Any failure: nothing is "
                                  "written."),
            approval_file_holds=("kind 'stage2-local-approval', the label, mode 'local-terminal', the committed "
                                 "prereg.json path, its SHA-256 and version, the typed characters, the UTC time, the "
                                 "host name, the local user name, the git HEAD commit and branch (read-only git "
                                 "queries; empty when git is unavailable), the terminal's name, the Python version, "
                                 "the SHA-256 of the code manifest and of approve_local.py, and a statement that the "
                                 "approval covers that exact pre-registration only and accepts every default"),
            import_step=("after the user's push, the orchestrating agent pulls and runs 'approve.py import' (no "
                         "arguments). Import reads the commit facts itself, with read-only git queries: the one commit "
                         "on origin's branch (as fetched) that added approval_<first 16>.json, its author and "
                         "committer, and the file as stored in it. It refuses unless exactly one commit added the file; "
                         "that commit is on origin's branch and the branch's tip still holds the file unchanged; the "
                         "file matches its sidecar and the file in that commit; it is a 'local-terminal' approval of the "
                         "CURRENT prereg.json (its SHA-256 and the typed first 16 characters); neither the author nor "
                         "the committer is the cloud session's own git identity (" + ", ".join(C.AGENT_GIT_EMAILS) +
                         ", or its configured user.email); the prereg.json, manifest, PREREGISTRATION.md, "
                         "approve_local.py and code copies stored in that commit and in the committed folder are the "
                         "current ones; no runs-side approval exists; and every check at approval passes (final "
                         "prereg.json, pinned code, every other job's pin). It writes runs/shift_explore/stage2/"
                         "approval.json (mode 'local-terminal', the verbatim local file, the git facts) and the import "
                         "record " + C.IMPORT_RECORD_PATTERN + " (the commit, author, committer, and the SHA-256 of the "
                         "runs-side approval), which the orchestrating agent commits and pushes. Every BCA entry point "
                         "re-checks all of it with git and refuses until the import record is on origin's branch."),
            strength=("nothing in the cloud session types or writes the user's approval file: it is typed at the "
                      "user's own terminal and arrives through the user's own git push. Import reads the commit facts "
                      "from git itself, and the import record that names the commit must be on GitHub before any BCA "
                      "run, so the user can check that commit without relying on a message from the agent. This added "
                      "strength holds only if the user does check, on GitHub, that the commit the import record names "
                      "is the user's own (a signed commit shows as verified)."),
            weakness=("it cannot prove who typed or committed it: approve_local.py checks only that standard input is a "
                      "terminal, and git's author and committer fields are set by whoever makes the commit and are not "
                      "authenticated; anyone able to push to the repository, an agent with push access included, could "
                      "make such a commit under the user's name. The refusal of the cloud session's own git identity "
                      "guards against a mistake, not against forgery. runs/shift_explore/stage2/approval.json is not "
                      "committed (runs/ is in .gitignore) and could be written by hand; the git checks make a forgery "
                      "visible on GitHub but cannot prevent it.")),
        what_approval_allows=[
            "The detection test: fit BCA's band (uniform; the weighted version only if the uniform one fails direction "
            "or size on seeds 0-4) on the stored critics and read its width at the locked detection pair sets, seeds "
            "0-4, then fresh seeds 5-9.",
            "The deployment check (Variant B) on seeds 0-9, in both worlds.",
            "The policy test's BCA arms and controls, and the full host (TD3+BC co-trained with BCA's lower edge for "
            "100,000 updates), only if a version passed detection direction and size and the room test passed.",
            "Fresh seeds 5-9: new logs, critics, cloned actors and plain TD3+BC (no BCA), then the same BCA stages with "
            "every setting frozen from seeds 0-4.",
            "The detection marks and the pooled looks: exploratory on seeds 0-4 (no claim), confirmatory on seeds 5-9.",
            "A compute budget of about 72-92 process-hours, at most 2 processes at once.",
        ],
        what_approval_does_not_allow=[
            "Any other version of prereg.json: any edit to it, or to a pinned script, voids the approval.",
            "Any setting, mark or rule other than those written in the pre-registration, or any other stage.",
            "BCA on any data other than the runs the pre-registration names.",
        ],
        modes=dict(
            local_terminal=("ACCEPTED, the only mode. approve_local.py on the user's own machine (see local_terminal), "
                            "committed and pushed by the user, imported with approve.py import, and the import record "
                            "committed and pushed"),
            terminal=("NOT ACCEPTED by this version (cloud_terminal_allowed: false). approve.py terminal in the cloud "
                      "session (the gate's mechanism: standard input is a terminal and the characters typed equal the "
                      "first 16 characters of prereg.json's SHA-256) refuses, and so does every BCA entry point for an "
                      "approval.json in this mode. Any agent with a shell can satisfy both checks through a "
                      "pseudo-terminal, and the user has no terminal in the cloud session"),
            chat=("NOT ACCEPTED by this version (chat_mode_allowed: false). approve.py chat (the exact sentence "
                  "'approve <first 16 characters>, accept chat mode', written to a file by the orchestrating agent) "
                  "refuses, and so does every BCA entry point for an approval.json in this mode. The program cannot "
                  "tell a genuine user turn from a file an agent wrote")),
        modes_not_accepted_note=("Earlier versions declared the cloud terminal and chat as alternatives. Version 6 "
                                 "records the user's choice of the local terminal only; using another mode needs a new "
                                 "version of prereg.json and a new fingerprint."),
        chat_sentence=C.CHAT_SENTENCE,
        trust=("The code accepts only an approval that 'approve.py import' wrote from the user's pushed file, but it "
               "cannot stop the orchestrating agent from going around it: the agent could write runs/shift_explore/"
               "stage2/approval.json by hand (it is not committed), or make and push a commit in the user's name, since "
               "git does not authenticate authors or committers. Import and every BCA entry point read git themselves "
               "and need the import record on GitHub, so such a forgery would be visible there, not prevented. So the "
               "route rests on the orchestrating agent following the procedure, and on the user checking on GitHub "
               "that the commit the import record names is the user's own. The modes this version does not accept "
               "(cloud terminal, chat) would rest entirely on trusting the agent."),
        defaults_accepted=("Approving this fingerprint accepts every default listed in open_choices; any other choice "
                           "needs a new version of prereg.json and a new fingerprint."),
        enforcement=("approve.py (import included) refuses unless prereg.json is final, it accepts the mode, the code "
                     "matches the code manifest and every other component's pin (room test, fallback diagnostic, "
                     "stand-ins, detection script) still matches what prereg.json records. Every Stage 2 entry point "
                     "that fits, reads or trains with BCA calls stage2_common.require_approval: runs/shift_explore/"
                     "stage2/approval.json must match its sidecar, name this prereg.json and its current SHA-256, be in "
                     "an accepted mode (version 6: local-terminal only), hold the verbatim local approval of this "
                     "SHA-256 with its git facts, the committed file still present and unchanged, git confirming the "
                     "commit, its author and committer and the file in it on origin's branch, and the import record "
                     "on origin's branch; every pinned script must be unchanged. Every BCA phase (run_stage2.bca_gate) "
                     "also re-checks the other components' pins. Any edit of prereg.json voids the approval."),
        after_a_new_version=("If a new version replaces this one before the import (for example because another job's "
                             "pin moved), the user approves the new fingerprint the same way: a new file named after "
                             "it; the earlier file stays and approves only the earlier version. After the import, "
                             "write_prereg.py and write_manifest.py refuse: the approved files are frozen, and a pin "
                             "that moves then stops every BCA phase until the user decides."))


def method():
    return dict(
        name="BCA lower-edge consumer (new method version)", id=C.METHOD_ID,
        signal_unchanged=("BCA's band, unchanged: algorithms/td3_bc_bca.fit_scale (scale network eta(s, a), two hidden "
                          "layers of 64 ReLU units, softplus + 0.01; unit u by averaging 0.99; Bayesian-bootstrap masses; "
                          "coverage sharpness 20; width penalty 0.005; learning rate 0.001) and td3_bc_bca.refresh "
                          "(frozen eta_f and u_f; threshold R = max(plain 90% quantile, 95%-credible quantile) of "
                          "weighted Bayesian conformal prediction on the calibration bank, miscoverage 0.1, credibility "
                          "0.95, 1,000 draws), the gate's settings experiments/shift/learners.BCA_SETTINGS"),
        width="U(s, a) = R * u_f * max(eta_f(s, a), 1e-6)   (calibration/dose.py)",
        what_changes=[
            "the width is read at any (s, a), in particular at the actor's own command pi(s)",
            "the gradient flows through the frozen scale network's command input; its parameters get no gradient "
            "(this breaks the design sentence 'host optimisation does not backpropagate through the calibrator', "
            "ALGORITHMS.md:127-130, so it is a new method version)",
            "weighted version only: the actor's gradient ALSO flows through the frozen classifier's command input "
            "(log w(s, a)), both in the tabulated threshold R_c(w(s, a)) and in the softplus refusal term, so host "
            "optimisation backpropagates through the threshold calculation as well (the second half of the same "
            "sentence, ALGORITHMS.md:127-130); the classifier's parameters get no gradient",
            "the width enters the actor's objective in critic value units with no 1.5 cap (no dose)",
            "'cannot certify' (R = +infinity) means maximum caution: the width is U_max, never 0. This acts per "
            "command only in the weighted version. With uniform weights it is all or nothing (one threshold), and it "
            "is inert: the paired penalty is 0 with zero gradient, the unpaired penalty is a constant with zero "
            "gradient, and in the deployment check tau = U_max, so the check never fires. With about 850 bank rows "
            "the uniform case cannot happen (certifiable for n >= 36)",
        ],
        primary_penalty="P(s_i, a) = [ U(s_i, a) - U(s_i, a_i) ]_+   (paired, one-sided; change A1: zero at the "
                        "recorded command a_i, never rewards a command)",
        actor_objective=("L(phi) = -lambda * (1/B) sum_i [ Q1(s_i, pi_phi(s_i)) - k * P(s_i, pi_phi(s_i)) ] "
                         "+ (1/B) sum_i (1/4) sum_j (pi_phi(s_i)_j - a_ij)^2"),
        normaliser="lambda = alpha / stopgrad( (1/B) sum_i |Q1(s_i, pi_phi(s_i))| ), computed on Q1 alone; alpha = 5",
        k=dict(primary=1.0, declared_secondary=2.0, rule="fixed in advance by meaning (the lower edge of the band); "
                                                          "never tuned on returns"),
        unpaired_secondary_arm="Q1(s_i, pi(s_i)) - k * U(s_i, pi(s_i)), k = 1 (the plan's literal consumer)",
        cloning_pull="weight 1 on every row (no dose); the critic loss unchanged",
        cannot_certify=dict(
            uniform=("one threshold for all inputs: if R = +infinity the width is U_max at every pair, so the paired "
                     "penalty is zero with zero gradient, the unpaired one a constant with zero gradient, and the "
                     "deployment check's tau equals U_max, so it never fires (inert everywhere); with about 850 bank "
                     "rows this cannot happen (BCA's own reference.certifiable(n) holds for n >= 36)"),
            weighted="per command: the refusal term U_max softplus(log w - l*) keeps growing beyond l*",
            R_c="R if finite, else the largest finite of (plain 90% quantile, credible quantile)",
            U_max="R_c * u_f * max over the bank rows of max(eta_f(s_j, a_j), 1e-6)"),
        weights=dict(primary="uniform (BCA as written)",
                     fallback=weighted_version(),
                     rule=("the weighted version is used only if the detection test's direction or size mark fails with "
                           "uniform weights on seeds 0-4; never chosen on returns")),
    )


def weighted_version():
    return dict(
        width="U~(s, a) = R_c(w(s, a)) * u_f * max(eta_f(s, a), 1e-6) + U_max * softplus(log w(s, a) - l*)",
        ratio=("w(s, a) = g / (1 - g), log w the logit of a classifier g (stage2/weights.py, the stand-in classifier "
               "ratio's estimator restated: 64-64 ReLU, inputs obs/1.001 and the command; target pairs "
               "(s_i, clip(pi(s_i) + a_i - c(s_i), -1, 1)) against recorded pairs (s_i, a_i) at the same training rows, c "
               "the cloned actor; balanced logistic loss; 2,000 Adam steps, learning rate 0.001, 512 rows from "
               "default_rng([20261015, seed, 82]); initialisation default_rng([20261015, seed, 84]))"),
        refit=("detection test: once, with the gapped critic's actor; policy test: before every 4,000-step chunk with "
               "the arm's own actor; full host: at every BCA refresh with the current actor"),
        threshold=("BCA's threshold rule (calibration/wbcp.calibrate) with bank weights w_j = w(s_j, a_j) and test mass "
                   "t, on the threshold half of the bank, tabulated on 48 test masses evenly spaced in log t from "
                   "min_j log w_j - 1 to log(0.1/0.9 * sum_j w_j) + 3, the same posterior draws for every t "
                   "(default_rng([20261015, seed, 72])); R_c(t) = R(t) with +infinity replaced by its largest finite "
                   "value, read by linear interpolation in log t; l* = the log of the smallest tabulated t with "
                   "R(t) = +infinity"),
        U_max="the largest finite width over the threshold half: max_j R_c(w_j) u_f max(eta_f(s_j, a_j), 1e-6)",
        bank_split="a fixed permutation default_rng([20261015, seed, 71]): first half sets the threshold, second half "
                   "only audits coverage (reported), because the actor trains against the threshold",
        eta_and_unit="the same eta_f and u_f as the uniform fit of the same seed (bank scores saved at that fit)",
        gradient_path=("the actor's gradient flows through the frozen scale network's command input AND the frozen "
                       "classifier's command input log w(s, a), in both terms: the tabulated threshold R_c(w(s, a)) "
                       "(piecewise linear in log w) and the refusal term U_max softplus(log w - l*). So host "
                       "optimisation backpropagates through the threshold calculation as well as the calibrator "
                       "(ALGORITHMS.md:127-130). Only the parameters of the scale network and the classifier are "
                       "frozen. Chosen because the threshold's rise with w is BCA's own route to distance awareness; "
                       "holding R_c(w) fixed in the gradient would leave only the scale network and the refusal term "
                       "to steer"))


def detection_test():
    return dict(
        what="the band is fitted and read; no actor is trained with BCA",
        fit=dict(
            primary=("each seed's frozen fixed-controller gapped critic of each world (check 2's; the critic the "
                     "policy test trains against): learners.bca_band's procedure restated (20,000 scale-fit steps on "
                     "batch rows default_rng([20261015, seed, 41]), one refresh on the gapped calibration bank) with "
                     "the residual target r + 0.98 min_h Q_target,h(s', clip(c(s') + noise)) where c is the "
                     "stall-aware controller (the critic's scored policy) and the noise TD3's smoothing noise (0.2, "
                     "clipped at 0.5), as the critic was trained; the gate saved only the critic's online parameters, "
                     "so the target critic is the same frozen critic"),
            reported=("exactly as written on check 3's frozen co-trained plain TD3+BC critic (stalling world): the "
                      "host's own actor target and critic target in the residual (learners.bca_band's native)"),
            outputs="band.msgpack, band.json, bank_scores.npz (BCA's own scores, reproduced bit for bit), tau.json"),
        readout=("the width only at the locked detection script's pair sets T1-T4 (detect_contract.py), through "
                 "band.WidthReader and the script's declared hook score_bca_width; widths and labels are hashed "
                 "separately and joined only afterwards by the script's measures(); for seeds 0-4 the rehearsal's own "
                 "locked pair sets, labels and stand-in scores are used (made before any BCA run); for seeds 5-9 they "
                 "are made in Stage 2's folder by the same locked code, the labels in a process holding no signal "
                 "code"),
        marks=dict(
            pooling=("each mark on the mean over seeds of the per-seed value AND in at least 4 of 5 seeds; every seed "
                     "needed: a seed failed on record (a band fit that failed, a fresh seed whose data or learners "
                     "failed) fails every mark; the marks are written once and are refused while any seed's inputs "
                     "(width scores, tau, labels, the stand-ins' scores) are merely missing"),
            direction=("on harmful past-80% pairs the width at the command exceeds the width at its 80% cap on at "
                       "least 75% of T1 pairs and at least 65% of T3 pairs (chance 50%)"),
            size=("the median over harmful past-80% T1 pairs of ([U(cmd) - U(rec)]_+ - [U(cap) - U(rec)]_+) / (the "
                  "critic's over-preference of the command over its cap) is at least 1, so k = 1 would cancel the "
                  "false preference (change A1: the consumer reads the width relative to the recorded command; the "
                  "raw (U(cmd) - U(cap)) / over-preference is reported)"),
            ranking="ranking score at least 0.75 on T1 and T2 together and at least 0.70 on T3",
            calibration_sanity="5% +/- 1.5% of T4 rows lie above tau (reported as a mark; not part of the claim rule)",
            selectivity="flag rate at past-80% pairs, stalling minus no-stall world: reported only (gated only in a "
                        "trace test bed)"),
        claim_rule=("'BCA flags the actor's reliance on commands the critic cannot vouch for' may be claimed only if "
                    "direction, size and ranking hold and BCA's pooled ranking score is no more than 0.02 below the "
                    "best data-only stand-in's, on T1 and T2 together and on T3 (stand-ins scored by the same locked "
                    "script on the same pairs: the five detection stand-ins distance to data, behaviour density, "
                    "classifier ratio, two-head disagreement and residual evidence; not the range hinge and not the "
                    "perfect band). All five must have finite ranking scores on every seed of the group; if any is "
                    "missing or not finite, this part of the rule fails (a rival that cannot be scored counts "
                    "against BCA)"),
        claim_fields=("only the confirmatory group (fresh seeds 5-9) can carry the claim: on seeds 0-4 the output's "
                      "'claim' is null and its text reads 'exploratory, not a claim'"),
        version_decision=("if direction or size fails with uniform weights on seeds 0-4, the weighted version is "
                          "declared and the detection test repeated with it; no uniform-weight policy test runs. If "
                          "the weighted version also fails direction or size, no BCA training arm runs (the "
                          "deployment check and the detection report remain)"),
        harm_claim="'Identifies the harm' is not tested here: it needs selectivity >= 0.2 in a trace test bed")


def policy_test():
    return dict(
        protocol=("the room test's (runs/shift_explore/room/code/harness.py, executed from its pinned bytes): check "
                  "2's actor-only protocol, the frozen fixed-controller gapped critic of the world, start from the "
                  "saved cloned actor, alpha 5, 20,000 actor steps in 4,000-step chunks, check 2's batch-row stream "
                  "default_rng([20261015, seed, 35]), Adam 3e-4, arms of one kind vmapped side by side"),
        penalty_slot=("kW_i = k_paired [U(s_i, pi) - U(s_i, a_i)]_+ + k_unpaired U(s_i, pi) + k_shuffled "
                      "[U(s_sigma(i), pi) - U(s_sigma(i), a_sigma(i))]_+, sigma the room's permutation "
                      "default_rng([20261015, seed, 61])"),
        arms=dict(
            none="check 2's saved gapped-critic actor (seeds 0-4) or the fresh one (5-9)",
            w0_check="the BCA program with all coefficients 0: must match 'none' within 1e-5 on every parameter, else "
                     "the seed stops (stop.json)",
            bca_k1="k_paired = 1: PRIMARY", bca_k2="k_paired = 2: declared secondary",
            bca_unpaired="k_unpaired = 1: declared secondary (the plan's literal consumer)",
            bca_shuffled="k_shuffled = 1: BCA's width from another training state (reported)",
            perfect_band="the room's main oracle [delta*(s_i, a_i) - delta*(s_i, a)]_+, k = 1 (reference)",
            range_hinge="the room's 100 sum_j ([a_j - hi_j]_+^2 + [lo_j - a_j]_+^2), k = 1 (reported)",
            even_pull="(1 + nu) times the cloning pull, matched (gating control)",
            random_1="random network of the scale network's shape, matched (gating control)",
            random_2="a second random network, matched (gating control)",
            push_less="c mean_j pi_j^2, matched (reported)",
            best_standin="the best data-only stand-in (best_standin.json), matched with the stand-ins' own ladder and "
                         "matcher (rival)"),
        matching=("blind to bca_k1's strength S* (root-mean-square change of the command from 'none' on the 5,000 "
                  "evaluation-split states): ladders (even pull nu 0.1, 0.25, 0.5, 1, 2; push less and random surfaces "
                  "knobs 0.1, 0.3, 1, 3, 10 with the room's own slope scale), the room's blind extension, then the "
                  "gate's matcher glq_match.match_parallel (tolerance 0.01, at most 5 rounds); strengths only, "
                  "before any return; region strengths reported"),
        lock="lock.json with every actor's parameter SHA-256 is written before any return; eval verifies it first",
        evaluation=("acceptance._evaluate: exact 100-step returns on the full 1,000-start banks of recorded "
                    "conditions, strong current, big waves and big kicks; points (0 = recording controller, 100 = "
                    "stall-aware controller); share past 80% overall and per thruster; shift cards of none and "
                    "bca_k1 (reported, report-card bug fixed)"),
        no_stall_world=("none, bca_k1 (the no-stall world's own band), the even pull at the stalling world's locked "
                        "nu, the range hinge, the perfect band and the best stand-in at its locked coefficient; not "
                        "re-matched; w0_check against the no-stall none"),
        marks=dict(
            pooling="pooled mean over the group's seeds of the paired per-seed difference; every seed needed",
            M1="strong current: bca_k1 - matched even pull >= 15 points, and positive in at least 4 of 5 seeds",
            M2="strong current: bca_k1 - each matched random surface >= 15 points",
            M3=("keeps the improvement: recorded conditions bca_k1 >= none - 3 and >= cloned actor + 25; big waves and "
                "big kicks bca_k1 >= none - 3"),
            M4="strong current: bca_k1 >= the matched best stand-in - 5",
            M5=("the deployment check on the no-caution actor beats random switching at its own rate by >= 10 points "
                "in strong current, with a cost of at most 2 in recorded conditions (check - never switching >= -2)")),
        reported=["the no-stall cost of every arm", "the share of steps past 80% on every thruster",
                  "the distance of visited pairs from the data (shift cards)", "region strengths",
                  "the BCA penalty's shape past the knee (the room's check (e) measurements, not a stop)"],
        run_condition=("only with a version that passed detection direction and size, and only if the room test's "
                       "pooled look passed its room, recovery and keeps-improvement marks (plan 4.8)"))


def deployment_check():
    return dict(
        rule="a_exec(s) = pi(s) if U(s, pi(s)) <= tau, else clip_j(pi(s)_j, lo_j, hi_j) (also when U is not finite)",
        tau=("the split-conformal 95% level (the ceil((n + 1) 0.95)-th smallest value) of U at every held-out recorded "
             "row: the calibration split's rows outside the removed ball, outside the calibration bank and outside "
             "the detection script's T4 rows (so the calibration-sanity mark reads T4 on rows tau never saw); its "
             "rate on the fallback diagnostic's 5,000 evaluation-split rows reported"),
        recorded_range="lo_j, hi_j: the 0.5% and 99.5% quantiles of the gapped training commands per thruster",
        band=("the chosen version's band on the frozen critic of the actor being checked: the fixed-controller "
              "critic's for 'none' and bca_k1, the as-written (uniform) host band for plain TD3+BC; the phase "
              "refuses until the version decision exists and uses the uniform band only if neither version passed"),
        uncertified_uniform=("a uniform band that cannot certify gives U = U_max everywhere, so tau = U_max and the "
                             "check never fires: maximum caution does not reach Variant B with uniform weights "
                             "(unreachable with about 850 bank rows)"),
        actors="none (mark M5), bca_k1 and plain TD3+BC (reported); both worlds (no-stall reported)",
        comparisons=("never switching; random switching at the check's own firing rate (a coin per start and step); "
                     "the fallback diagnostic's distance check (sa95) and random switching at its rate; all with the "
                     "clip fallback, the first 200 starts of each 1,000-start bank, exact simulation"),
        runs="always (plan 4.8: the deployment check remains even when no training arm runs)")


def full_host():
    return dict(
        what=("TD3+BC co-trained with BCA's lower edge (k = 1, the detection test's version) for 100,000 updates, "
              "critic and actor together; BCA's scale fit every update; the frozen reference first refreshed after "
              "update 10,000, then every 5,000 updates; width 0 before the first refresh; U_max (and the weighted "
              "extras) recomputed at every refresh; the same initial parameters and batch stream as plain TD3+BC"),
        arms=dict(stall=("BCA host; plain TD3+BC (the gate's saved run / the fresh run); TD3+BC plus the range hinge "
                         "matched blind to the BCA host's strength (against plain TD3+BC's final actor; ladder c = 0.1, "
                         "0.3, 1, 3, 10, extension x sqrt(10), glq_match)"),
                  no_stall="BCA host; plain TD3+BC trained here (learners.cotrain); the hinge host at the locked c"),
        checks="the hinge program with c = 0 reproduces learners.cotrain for 2,000 updates within 1e-5 (reported)",
        reported="points in every situation and both worlds; the claim sentence's numbers for the host",
        run_condition="the same as the policy test")


def confirmation():
    return dict(
        seeds=list(C.CONFIRMATORY_SEEDS),
        made_here=("fresh_data: experiments/shift/dataset.generate(seed, world, the gate's recorder settings, the "
                   "gate's sizes, stall slope 1.5) for both worlds, shift.define_gap and the removals, the gate's leak "
                   "checks and (stalling world) world marks T1-T5, required; fresh_learners: fixed-controller critics "
                   "(gapped, full log, no-stall gapped), cloned actors (both worlds), the no-caution actors and the "
                   "full-log critic's actor (learners.ActorOnly, alpha 5, side by side), plain TD3+BC (alpha 5); all "
                   "with the gate's unchanged code; outputs runs/shift_explore/stage2/fresh/"),
        frozen=("the version (uniform or weighted) chosen on seeds 0-4, k, every ladder, tolerance and rule, the best "
                "stand-in's identity, tau's rule"),
        marks="the detection marks and claim rule plus the policy marks M1-M5, on seeds 5-9 pooled (4 of 5 seeds)",
        failed_seed=("a fresh seed that fails its data checks or a learner fails every mark; never replaced. "
                     "Mechanism: fresh_data writes data.json with passed=false, fresh_learners writes learners.json "
                     "with failed=true (fresh.recorded_failure reads both); every later phase refuses to use the seed, "
                     "the detection marks count it as a failed seed, the policy marks as a stopped seed, and the "
                     "pooled look accepts it in place of its deployment values, so the confirmatory pooled look can "
                     "still be written"),
        harm=("the harm on fresh seeds is the full-log critic's actor minus the no-caution actor in strong current "
              "(the '99 points' of seeds 0-4)"))


def comparison_set():
    return dict(
        standins=("POLICY TEST rivals: the data-only stand-ins built in scratchpad/ood/standins/ (Stage 1, plan 4.2): "
                  "range hinge, behaviour density, classifier ratio, distance to data, two-head disagreement, in the "
                  "five forms that carry the stand-ins test's mark. The user chooses those forms in that test's own "
                  "approval: 'approve F1-F3' gives density, distance and two-head read from the recorded command "
                  "(change A1); 'decline F1-F3' gives the plan's literal forms. Stage 2 follows that choice, as "
                  "recorded in the stand-ins' marks.json (mark_families)"),
        policy_standins=dict(if_F1_F3_approved=["range_hinge", "density", "classifier_ratio", "distance", "two_head"],
                             if_F1_F3_declined=["range_hinge", "density_plan", "classifier_ratio", "distance_plan",
                                                "two_head_plan"]),
        detection_standins=["distance", "density", "classifier_ratio", "two_head", "residual_evidence"],
        detection_standins_note=("DETECTION CLAIM rivals (a different set, scored by the locked detection script on "
                                 "the same pairs): distance to data, behaviour density, classifier ratio, two-head "
                                 "disagreement, residual evidence (minus the mean one-step error of the 10 nearest "
                                 "training rows). The range hinge and the perfect band are not detection stand-ins."),
        best_standin_rule=("chosen by select_standin before any BCA run, from the stand-ins' marks.json (seeds 0-4): "
                           "among the five mark forms (its mark_families), those that capture the room; the best is the one with the largest pooled "
                           "matched gain over no caution in strong current; if none captures the room, the largest "
                           "pooled gain among all five; ties to the earlier name; written to best_standin.json"),
        detection_best=("among the five detection stand-ins, the one with the largest pooled ranking score on the same "
                        "locked pair sets (on T1 with T2, and on T3, separately); all five needed on every seed"),
        deployment_rivals="random switching at the check's rate; the distance check")


def run_order():
    return [
        "0. the pinned components (room test, stand-ins, detection script) are locked and prereg.json is approvable "
        "now (approve.py show); the user approves by the only accepted route (approve_local.py on the user's own "
        "machine, committed and pushed by the user, then approve.py import, then the import record committed and "
        "pushed), possibly before Stage 1's outcomes exist: each phase reads what it needs when it runs (the room's "
        "pooled look, the stand-ins' marks); any rewrite of a pinned component before the import makes Stage 2 refuse "
        "until write_manifest.py and write_prereg.py are rerun and the new version approved (a new approval file, "
        "named after the new fingerprint); after the import the approved files are frozen",
        "1. select_standin (no BCA)",
        "2. seeds 0-4, both worlds: detect_pairs, detect_labels, detect_standins (no BCA: verify the rehearsal's, "
        "read-only; each refuses when the rehearsal's file is absent); detect_fit (fixed; host in the stalling world); "
        "detect_score; detect_marks --group exploratory --version uniform (refused while any seed's inputs are "
        "missing); if direction or size failed: detect_fit_weighted and detect_score --version weighted for the "
        "fixed critic in both worlds, then detect_marks --version weighted (the host critic is always read as "
        "written, uniform)",
        "3. seeds 0-4: deploy (both worlds; refused until the version decision exists; the uniform band only if "
        "neither version passed); if a version passed and the room passed: policy_train, policy_eval, "
        "policy_no_stall_train, policy_no_stall_eval, host_train / host_eval (stall, then no_stall); pooled --group "
        "exploratory (refused until the detection marks and every seed's deployment values exist)",
        "4. seeds 5-9: fresh_data, fresh_learners (no BCA; may run as soon as prereg.json is final), then the same "
        "detection (frozen version), policy, deployment and host phases; detect_marks and pooled --group confirmatory",
    ]


def compute():
    return dict(
        unit="process-hours (one CPU busy for one hour); at most 2 processes at once",
        detection_seeds_0_4=dict(fits="about 0.8 (15 band fits: 5 seeds x stall fixed, stall host, no-stall fixed)",
                                 labels_and_standin_scores="0 (the rehearsal's are reused)",
                                 weighted_if_declared="about 2", total="about 1 (+2 if weighted)"),
        policy_seeds_0_4="about 15 (about 2.5-3 per seed: about 25 arm-runs with matching, the stand-in's data side "
                         "and ladder, 12-14 exact evaluations, the no-stall arms)",
        deployment_seeds_0_4="about 3 (3 actors x 4 situations x 5 rules x 200 starts, both worlds)",
        full_host_seeds_0_4="about 13-22 (per seed: the BCA host about 12 min, 5-10 hinge hosts of about 8 min, the "
                            "no-stall hosts, evaluations)",
        fresh_seeds_build="about 3 (per seed: data about 4 min, 3 critics about 18 min, clones, 3 actor-only runs, "
                          "one plain TD3+BC about 8 min)",
        fresh_seeds_bca="about 35-45 (the same stages as seeds 0-4, plus about 2.5 for their pair sets, labels and "
                        "stand-in scores)",
        total="about 72-92 process-hours: about 36-46 hours on 2 processes; the decisive first look (detection on "
              "seeds 0-4) takes about 1 hour on 2 processes",
        basis="the gate's timings (fixed-controller critic about 6 min, plain host about 8 min, BCA run about 12 min) "
              "and the room test's (plain arm 3 s and band arm 6 s per 1,000 steps)")


def claims():
    return dict(
        flags_claim=("BCA flags when the actor relies on commands the data do not support, and acting on that flag "
                     "repairs the harm while keeping TD3+BC's gains."),
        flags_claim_rule=("supported (confirmatory) only if, on fresh seeds 5-9 with every setting frozen, the "
                          "detection claim rule holds AND the policy marks M1-M5 all pass"),
        sentence=("In strong current, BCA's lower edge recovers X of H points and keeps Y of TD3+BC's gain over cloning "
                  "in the recorded conditions. A hand-set limit to the recorded range recovers Z. In the no-stall "
                  "world, where the missing commands help, BCA costs W points and the range limit V."),
        sentence_numbers=dict(
            H="the harm: full-log critic's actor minus the no-caution actor, strong current (98.9 on seeds 0-4)",
            X="bca_k1 minus none, strong current", Y="(bca_k1 - clone) / (none - clone), recorded conditions",
            Z="range hinge (k = 1) minus none, strong current", W="no-stall world: none minus bca_k1, strong current",
            V="no-stall world: none minus range hinge, strong current"),
        not_claimed=("'BCA identifies the harm itself, not just novelty' (the harm claim) is not tested by this "
                     "pre-registration; it needs the trace test bed (plan 4.6) under its own pre-registration"),
        new_version_caveat=("results speak for the lower-edge version of BCA only, not for BCA as published (dose on "
                            "the cloning pull), and the reverse"))


def deviations():
    return [
        "Change A1 (room spec) applied to BCA: the primary consumer is the paired form [U(s, pi) - U(s, a_i)]_+; the "
        "plan's Q1 - k U(s, pi) runs as the declared secondary arm 'bca_unpaired'.",
        "The size mark is read on the paired penalty's difference between the command and its cap (what the actor "
        "feels); the plan's raw width difference is reported.",
        "Fixed-controller critic: BCA's residual target uses the stall-aware controller's next command with TD3's "
        "smoothing noise (td3_bc_bca.native_target as written); the target critic is the frozen critic itself because "
        "the gate saved only online parameters.",
        "Uniform 'cannot certify' is all-or-nothing (one threshold), so under uniform weights it carries no command "
        "information; per-command refusal exists only in the weighted version.",
        "The plan's Variant A-weighted gives R_c 'R with +infinity replaced by its largest finite value'; the grid "
        "range of the 48 test masses and the common posterior draws are fixed here (weighted_version.threshold).",
        "Both random surfaces are matched and M2 must hold against each (the plan names 'the random surface').",
        "Push less everywhere is matched and reported (not in plan 5.3's list).",
        "The controls' ladders use the room's own slope scale (the perfect band's slope at the cloned actor's "
        "commands), so they do not depend on BCA.",
        "The weighted version's classifier restates the stand-in classifier ratio's estimator (plan 4.4 'the "
        "classifier fixed in the rehearsal'); the locked detection script's signals.fit_classifier is the same "
        "estimator (same network, rows, initialisation, optimiser and steps).",
        "Plan 4.2 names no rule for 'the best stand-in'; it is fixed here (comparison_set.best_standin_rule).",
        "M5 is read on the no-caution actor (where the harm is); bca_k1 and plain TD3+BC are reported.",
        "Fresh seeds: no random-removal critic is trained (no mark needs it); the no-caution actor and the full-log "
        "critic's actor are trained side by side (two arms) rather than check 2's three.",
        "The full host's no-stall plain TD3+BC is trained here (the gate trained none there).",
        "Shift cards are produced with the report-card bug fixed (anchors of one situation), reported only.",
        "Deployment check: tau from every held-out recorded row except the detection script's T4 rows (not a "
        "5,000-row sample); T4 is left out so that the calibration-sanity mark is informative.",
        "Detection on seeds 0-4 reuses the rehearsal's locked pair sets, labels and stand-in scores (all made before any "
        "BCA run, by the pinned script); seeds 5-9 get their own, made by the same code in Stage 2's folder.",
        "The policy test's rival is chosen among the five stand-in forms the user chose for the stand-ins test's mark "
        "(that test's approval, recorded in its marks.json), not always the re-referenced forms.",
        "The host critic's detection readout and plain TD3+BC's deployment check always use the as-written (uniform) "
        "band, whichever version is chosen (the plan says 'exactly as written').",
        "The weighted version's gradient also flows through the classifier's command input, into the tabulated "
        "threshold and the refusal term (the plan's formula does not say; declared in method.what_changes).",
    ]


def revision():
    return dict(
        version_6=("rewritten after a review of version 5 (never approved). The method, rules, marks and settings are "
                   "those of version 5 (and so of version 4). One pin changed, outside Stage 2: the stand-ins job "
                   "re-locked its code (its spec version 3, change C6: the user's choice of stand-in forms now gates "
                   "only its marks look; code manifest written 2026-10-13 07:18:30 UTC) after version 5 was written, "
                   "so version 5 could no longer be approved; version 6 pins the new stand-ins manifest, spec and "
                   "standins.py (pins.standins_manifest_sha256, standins_marks_spec_sha256, standins_py). The "
                   "functions Stage 2 calls in standins.py (build_shared, classifier_fit_for, slope_scales, "
                   "StandinRunner, family_arm, locked_arm, LADDER, match_standins) and the marks.json fields "
                   "select_standin reads are unchanged (checked against the code). The detection script's pin "
                   "(07:00 UTC) and the room test's and fallback diagnostic's pins did not move. Approval: only the "
                   "user's chosen mode, local-terminal, is accepted (cloud_terminal_allowed and chat_mode_allowed "
                   "are false; approve.py terminal and chat and every BCA entry point refuse them); 'approve.py "
                   "import' reads the commit facts itself with read-only git queries (exactly one commit on origin's "
                   "branch added the file; author and committer are read, and neither may be the cloud session's "
                   "identity), checks the files stored in that commit, and writes an import record in the committed "
                   "folder that must be committed and pushed before any BCA run; every BCA entry point re-checks the "
                   "commit with git and needs the import record on origin's branch; the approval file is named after "
                   "the fingerprint (approval_<first 16>.json), so a later version can be approved without deleting "
                   "an earlier file; 'approve.py show' and the staging build check the code and every pin now and say "
                   "whether the fingerprint can be approved; the documents state plainly that the route still rests "
                   "on the orchestrating agent following the procedure and on the user's check of the commit."),
        version_5=("rewritten at the user's choice of approval route (version 4 was never approved). The method, "
                   "rules, marks and settings are those of version 4. One pin changed, outside Stage 2: the "
                   "detection rehearsal's job re-locked its detection script (its spec version 5, code manifest "
                   "written 2026-10-13 07:00 UTC) after version 4 was written, so version 4 could no longer be "
                   "approved; version 5 pins the re-locked script (pins.detect_module, pins.detect_manifest_sha256), "
                   "which still passes detect_contract.check_detect_module. Approval: the primary route is now "
                   "local-terminal: "
                   "approve_local.py (standard library only, pinned by the code manifest) run by the user at a "
                   "terminal on the user's own machine from the committed folder outputs/shift/beyond-the-data/"
                   "stage2/, which the user commits and pushes; approve.py import verifies it and records the commit "
                   "and its author; stage2_common.verify_approval accepts mode 'local-terminal' and re-checks it at "
                   "every BCA entry point. The cloud-terminal and chat modes stay as declared alternatives with their "
                   "weaknesses stated. The approval section lists what approval allows and does not allow."),
        version_4=("rewritten after a review of version 3 (never approved). Method: the weighted version's gradient "
                   "path through the classifier and the threshold is declared; uniform 'cannot certify' is declared "
                   "inert (paired penalty, unpaired gradient and the deployment check). Rules: the detection claim "
                   "needs all five detection stand-ins on every seed; seeds 0-4 carry no claim in any output; a fresh "
                   "seed's recorded data or learner failure counts as a failed or stopped seed; the detection marks, "
                   "the deployment check and the pooled look refuse while their inputs are incomplete; Stage 2 never "
                   "builds pair sets in the rehearsal's folder. Approval: chat mode takes exactly one sentence; "
                   "approval re-checks every pin; every BCA phase re-checks every pin; both modes' reliance on the "
                   "orchestrating agent is stated. Documents: the two stand-in sets are named separately."),
        previous_version_3_sha256="6b1da63f7e5f8625343c52aac26218890770a7d14735159e1f6a3c6994bc5441",
        previous_version_4_sha256="d94ac6b835a6d7ee5fd8a18024af278d909ebb21687e2e884029886c931bf98f",
        previous_version_sha256="0f314318b5a5daa1f4537e4bad006c1af347dd382936d7856bbcd3a5ac59c259")


def open_choices():
    return [
        "How to approve: at a terminal on your own machine with approve_local.py, then commit and push the approval "
        "file (your choice, and the only mode this version accepts; a terminal in the cloud session or chat mode "
        "would need a new version). The route cannot authenticate the commit's author or committer, and it rests on "
        "the orchestrating agent following the procedure and on your check, on GitHub, of the commit the import "
        "record names.",
        "Accept change A1 for BCA (paired primary, the plan's unpaired form secondary).",
        "Run the full host whenever detection and the room passed (default), or only if the policy marks pass first.",
        "The classifier: the locked detection script restates weights.py's estimator exactly (one estimator); nothing "
        "to choose unless another estimator is wanted.",
        "M5 on the no-caution actor (default) or on the bca_k1 actor.",
        "M2 against both random surfaces (default) or one.",
        "Detection labels for seeds 0-4: reuse the rehearsal's locked labels (default) or recompute them.",
        "The compute budget (about 72-92 process-hours).",
        "Weighted version only (used only if the uniform version fails direction or size): the actor's gradient "
        "flows through the tabulated threshold R_c(w) as well as the refusal term (default), or the threshold is held "
        "fixed in the gradient and only the refusal term and the scale network steer.",
    ]


if __name__ == "__main__":
    print(json.dumps(pins(), indent=1))
