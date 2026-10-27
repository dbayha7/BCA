"""SMOKE TEST of eval_resultF.py's run mode (not results): it feeds the sandbox's existing TD3+BC host actor in place
of result F's actor and of "No extra caution", so the scoring, episodes, four situations, comparison and report code
run end to end before any training, by the partial path (`run --allow-missing`: B2, C2, the Perfect lower edge and
the optional rows are missing and listed). Trains nothing; writes only <resultF>/smoke/eval/. EXPLORATORY."""
import sys

sys.dont_write_bytecode = True
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import eval_resultF as E  # noqa: E402

E.OUT = E.RF / "smoke" / "eval"
E.LABEL = "SMOKE TEST (the TD3+BC host actor stands in for every learner; not results). " + E.LABEL


def fake_learners(q, ctx, spec):
    host = E.R.load_host_state(ctx, "gap", E.SEED, spec)
    snap = E.R.host_dir(ctx, "gap", E.SEED) / "snapshot_100.msgpack"
    f = dict(source="SMOKE: the gapped TD3+BC host's actor", failed=False, snapshot_100_file=str(snap),
             snapshot_100_file_sha256=E.sha(snap), actor_tree_sha256=E.LR.tree_sha256(host.actor.params),
             clone_actor_sha256="SMOKE: one stand-in clone for both")
    return ({"f1_A2_cmd_strong": (host.actor.params, dict(f), None), "le_none": (host.actor.params, dict(f), None)},
            [(n, "smoke test") for n in E.NAMES if n not in ("f1_A2_cmd_strong", "le_none")])


E.learners_of = fake_learners
try:   # without --allow-missing the run mode needs every core learner, and refuses before scoring anything
    E.run(allow_missing=False)
    raise AssertionError("run without --allow-missing did not refuse while B2, C2 and le_oracle are missing")
except E.R.Refused as error:
    print(f"smoke: run without --allow-missing refused as expected: {error}", flush=True)
code = E.run(allow_missing=True)
body = E.json.loads((E.OUT / "scores_seed0.json").read_text())
sit = E.json.loads((E.OUT / "situations_seed0.json").read_text())
assert body["complete"] is False and body["allow_missing"] is True, body["mode_why"]
assert all(body["comparison"][q]["same_clone"]["identical"] for q in E.QUALITIES)
assert "seconds" in sit and all(not f["failed"] and "snapshot_100_file_sha256" in f
                                for f in sit["learner_facts"].values())
print(f"smoke: partial scoring ran end to end; missing rows listed: {body['core_missing']}")
sys.exit(code)
