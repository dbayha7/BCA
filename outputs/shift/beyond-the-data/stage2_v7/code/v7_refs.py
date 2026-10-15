"""Stage 2, version 7, change C7 step 3: the reference redirect (V7_SPEC.md C7, the user's Q4 decision).

On seeds 2-4 only, where the self-test's probes show the laptop does not reproduce one of the gate's saved reference
actors bit for bit (C7 step 2's rule, fixed at the handoff in the final prereg_v7.json), the phase that reads that
reference reads the laptop's own copy instead, made before approval with the gate's exact call (v7_probes.py: P1a for
check 2's actors, P2 for the plain TD3+BC run). Entries are keyed (phase, seed, world). Exactly three kinds exist:

  (policy_train, s, stall)          check 2's 'gapped' and 'full' (one call made them side by side, so they stay a
                                    pair) -> lock names 'none' and 'full_none' in OUT/policy/seed<s>/lock.json
  (policy_no_stall_train, s, no_stall)   check 2's 'no_stall_gapped' -> 'none' in OUT/policy/seed<s>/no_stall/lock.json
  (host_train, s, stall)            the plain TD3+BC run's final state -> 'plain' in OUT/host/stall/seed<s>/lock.json

Any other entry (seeds 0-1 or 5-9, deployment, an evaluation, the no-stall host, detection, the pooled look), or any
table in a draft prereg_v7.json, refuses with exit 97. Deployment always reads the gate's saved actors.

How (no pinned file is edited; the gate's files stay as they are and the gate's own checks read them):
  1. The phase's --phase, --seed and --world are read exactly as run_stage2.py reads them (v7_ledger.phase_key); a
     policy phase's world is fixed by its name, host_train's is its --world (default 'stall'). No entry: nothing is
     installed and nothing is imported (the process is the plain v7 one).
  2. With an entry, before the hand-over and without JAX: the reference file's SHA-256 is checked against the pin;
     v6's stage2_common is imported from the folder launch.sh runs from (import-light; run_stage2.py's own
     'import stage2_common' then finds the same module object); its attribute load_room_harness is replaced by a
     wrapper that calls the original unchanged (which verifies and executes ROOM/harness.py's pinned bytes and caches
     the module) and then, once, replaces ONE method of that module's class SeedContext, the base class of every
     GateContext (S2C/fresh.py:52-55):
  3. a policy entry replaces SeedContext.h2 (ROOM/harness.py:248-255) by a function that calls the original first (so
     the gate's values.json and actors.msgpack are read and checked exactly as in v6), requires self.rep to be the
     entry's seed, loads the reference file with a dictionary template of the names P1a wrote (pinned), as
     ROOM/harness.py:253-254 does, and replaces in the returned dictionary exactly the entry's keys, each checked
     against its pinned per-actor SHA-256 (L.tree_sha256). Every other key (td3bc_gapped included) and the record stay
     the gate's.
  4. a host entry replaces SeedContext.host (ROOM/harness.py:267-271) by a function that calls the original first (so
     snapshot_100.msgpack is read and checked as in v6), requires the seed and returns (state, record) with the state
     loaded by the gate's own call (GATE/acceptance.py:1079) from the reference file, its actor checked against the pin.
  5. Any failure inside a replacement writes one ledger line and calls os._exit(97); it never raises.
"""

import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import v7_common as K  # noqa: E402

KINDS = {
    ("policy_train", "stall"): dict(method="h2", lock="policy/seed{s}/lock.json",
                                    lock_names={"gapped": "none", "full": "full_none"}, file="check2_stall.msgpack"),
    ("policy_no_stall_train", "no_stall"): dict(method="h2", lock="policy/seed{s}/no_stall/lock.json",
                                                lock_names={"no_stall_gapped": "none"}, file="check2_no_stall.msgpack"),
    ("host_train", "stall"): dict(method="host", lock="host/stall/seed{s}/lock.json", lock_names={"actor": "plain"},
                                  file="host_plain/snapshot_100.msgpack"),
}
LABEL_GATE = "gate's saved file"
LABEL_REMADE = "re-made on the laptop"
LABEL_OWN = "own run"
LABEL_NONE = "no reference read"
_HEX64 = re.compile(r"[0-9a-f]{64}")


def _refuse(reason):
    import v7_ledger
    v7_ledger.hard_exit(reason)


def validate_entry(e):
    """One table entry, against the three kinds. V7Refusal for anything else."""
    if not isinstance(e, dict):
        raise K.V7Refusal(f"a re-make table entry is not an object: {e!r}")
    kind = KINDS.get((e.get("phase"), e.get("world")))
    if kind is None:
        raise K.V7Refusal(f"a re-make entry for ({e.get('phase')}, {e.get('world')}) is not allowed: only policy_train "
                          f"(stall), policy_no_stall_train (no_stall) and host_train (stall) can read a re-made reference")
    seed = e.get("seed")
    if not isinstance(seed, int) or seed not in K.REMAKE_SEEDS:
        raise K.V7Refusal(f"a re-make entry for seed {seed!r} is not allowed (seeds 2-4 only)")
    replace = e.get("replace")
    if not isinstance(replace, dict) or set(replace) != set(kind["lock_names"]):
        raise K.V7Refusal(f"the re-make entry ({e['phase']}, {seed}, {e['world']}) must replace exactly "
                          f"{sorted(kind['lock_names'])}")
    for k, spec in replace.items():
        if not isinstance(spec, dict) or not _HEX64.fullmatch(str(spec.get("actor_sha256"))) or \
                spec.get("lock_name") != kind["lock_names"][k]:
            raise K.V7Refusal(f"the re-make entry's actor {k!r} has no valid pin or lock name")
    if not _HEX64.fullmatch(str(e.get("file_sha256"))):
        raise K.V7Refusal("the re-make entry has no valid file SHA-256")
    if e.get("reference_file") != f"refs/seed{seed}/{kind['file']}":
        raise K.V7Refusal(f"the re-make entry's reference file must be refs/seed{seed}/{kind['file']} under R7/")
    if kind["method"] == "h2":
        names = e.get("names_in_file")
        if not isinstance(names, list) or not set(replace) <= set(names):
            raise K.V7Refusal("the re-make entry does not list the names its reference file holds")
    return e


def table(prereg_payload):
    """The re-make table of a prereg_v7.json payload, every entry validated. A draft holds no table."""
    handoff = prereg_payload.get("handoff")
    entries = (handoff or {}).get("remake_table") if isinstance(handoff, dict) else None
    final = prereg_payload.get("approvable") is True and not prereg_payload.get("missing")
    if not final:
        if entries:
            raise K.V7Refusal("a re-make table in a draft prereg_v7.json is refused (it is fixed at the handoff)")
        return []
    entries = entries or []
    seen = set()
    for e in entries:
        validate_entry(e)
        k = (e["phase"], e["seed"], e["world"])
        if k in seen:
            raise K.V7Refusal(f"two re-make entries for {k}")
        seen.add(k)
    return entries


def lookup(entries, key):
    """The entry for (phase, seed, effective world), or None. key is v7_ledger.phase_key(...)."""
    for e in entries:
        if (e["phase"], e["seed"], e["world"]) == (key.get("phase"), key.get("seed"), key.get("world")):
            return e
    return None


def label_for(key, entry):
    p, s, w = key.get("phase"), key.get("seed"), key.get("world")
    if entry is not None:
        return LABEL_REMADE
    if p == "host_train" and w == "no_stall":
        return LABEL_OWN
    if s in (0, 1, 2, 3, 4) and p in ("policy_train", "policy_no_stall_train", "host_train", "deploy"):
        return LABEL_GATE
    return LABEL_NONE


def ledger_entry(entry):
    """The entry as the ledger records it (with the lock that must record the re-made actors)."""
    kind = KINDS[(entry["phase"], entry["world"])]
    return dict(entry, lock=kind["lock"].format(s=entry["seed"]))


def import_stage2_common(s_folder):
    """v6's stage2_common, imported under its own name from the folder launch.sh runs from; the folder is removed from
    sys.path again (run_stage2.py adds it itself, S2C/run_stage2.py:49)."""
    s_folder = str(Path(s_folder))
    existing = sys.modules.get("stage2_common")
    if existing is not None:
        mod = existing
    else:
        sys.path.insert(0, s_folder)
        try:
            import stage2_common as mod  # noqa: F401
        finally:
            while s_folder in sys.path:
                sys.path.remove(s_folder)
    if Path(mod.__file__).resolve().parent != Path(s_folder).resolve():
        raise K.V7Refusal(f"stage2_common came from {mod.__file__}, not from {s_folder}")
    return mod


def _patch_h2(H, entry, ref_path, on_fail):
    L = H.L
    orig = H.SeedContext.h2
    seed, fsha = int(entry["seed"]), entry["file_sha256"]
    names = list(entry["names_in_file"])
    replace = {k: v["actor_sha256"] for k, v in entry["replace"].items()}

    def h2(self):
        acts, rec, d = orig(self)
        out = None
        try:
            if int(self.rep) != seed:
                on_fail(f"the redirect for seed {seed} was asked for seed {self.rep}")
            if K.file_sha256(ref_path) != fsha:
                on_fail(f"HALT: reference {ref_path} differs from its pin")
            ref = L.load_tree(ref_path, {k: L.actor_template() for k in names}, fsha)
            out = dict(acts)
            for k, want in replace.items():
                if k not in ref:
                    on_fail(f"the reference file {ref_path} holds no actor {k!r}")
                if L.tree_sha256(ref[k]) != want:
                    on_fail(f"HALT: reference actor {k!r} in {ref_path} differs from its pin")
                out[k] = ref[k]
        except BaseException as error:  # noqa: BLE001 - never raise into the phase (C7 step 3.5)
            on_fail(f"the reference redirect failed: {type(error).__name__}: {error}")
        return out, rec, d

    h2.v7_redirect = dict(entry)
    H.SeedContext.h2 = h2
    return orig


def _patch_host(H, entry, ref_path, on_fail):
    L = H.L
    orig = H.SeedContext.host
    seed, fsha = int(entry["seed"]), entry["file_sha256"]
    want = entry["replace"]["actor"]["actor_sha256"]

    def host(self):
        state, rec = orig(self)
        new = None
        try:
            if int(self.rep) != seed:
                on_fail(f"the redirect for seed {seed} was asked for seed {self.rep}")
            if K.file_sha256(ref_path) != fsha:
                on_fail(f"HALT: reference {ref_path} differs from its pin")
            new = L.load_tree(ref_path, L.agent_template(self.learn), fsha)
            if L.tree_sha256(new.actor.params) != want:
                on_fail(f"HALT: the re-made host actor in {ref_path} differs from its pin")
        except BaseException as error:  # noqa: BLE001
            on_fail(f"the reference redirect failed: {type(error).__name__}: {error}")
        return new, rec

    host.v7_redirect = dict(entry)
    H.SeedContext.host = host
    return orig


def install(entry, s_folder=None, r7=None, on_fail=None):
    """Steps 2-4 for one entry. Returns a description for the ledger. V7Refusal before the hand-over (missing or changed
    reference file); later failures stop the process (exit 97)."""
    validate_entry(entry)
    on_fail = on_fail or _refuse
    r7 = Path(r7 or K.R7)
    ref_path = r7 / entry["reference_file"]
    if not ref_path.is_file():
        raise K.V7Refusal(f"HALT: reference {ref_path} is missing")
    if K.file_sha256(ref_path) != entry["file_sha256"]:
        raise K.V7Refusal(f"HALT: reference {ref_path} differs from its pin")
    if entry.get("probe_record"):
        rec = r7 / "selftest" / f"{entry['probe_record']}.json"
        if not rec.is_file() or K.file_sha256(rec) != entry.get("probe_record_sha256"):
            raise K.V7Refusal(f"HALT: probe record {entry['probe_record']} differs from its pin")
    C = import_stage2_common(s_folder or K.S_FOLDER)
    if getattr(C.load_room_harness, "v7_redirect", None) is not None:
        raise K.V7Refusal("a reference redirect is installed already in this process")
    original = C.load_room_harness
    kind = KINDS[(entry["phase"], entry["world"])]
    state = dict(patched=False)

    def load_room_harness():
        H = original()
        if not state["patched"]:
            if kind["method"] == "h2":
                _patch_h2(H, entry, ref_path, on_fail)
            else:
                _patch_host(H, entry, ref_path, on_fail)
            state["patched"] = True
        return H

    load_room_harness.v7_redirect = dict(entry)
    load_room_harness.__wrapped__ = original
    C.load_room_harness = load_room_harness
    return dict(label=LABEL_REMADE, entry=ledger_entry(entry), reference_file=str(ref_path),
                file_sha256=entry["file_sha256"], method=kind["method"])


def install_for(prereg_payload, key, s_folder=None, r7=None, on_fail=None):
    """The wrapper's step 5: look up (phase, seed, world); install the redirect only when there is an entry."""
    entries = table(prereg_payload)
    entry = lookup(entries, key)
    if entry is None:
        return dict(label=label_for(key, None), entry=None)
    return install(entry, s_folder, r7, on_fail)
