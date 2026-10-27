"""Stage-0 tests of the data area (plan sections 2.2, 3.1 "Data copy", 3.8, 4.2, 4.5 test (e), 4.6 and stop rule S11 of
section 6.1): the copied weighting-test files verify against its SHA-256 records, sit where the readers look for them
(runs/shift_hosts/source/ is the weighting test's data root; the start banks are at source/start_banks, where
evaluate.py reads them) and are the only files there; the prepared files verify and follow INTERFACES.md section 6;
their row counts and row hashes equal the weighting test's records; ReBRAC's roll rule holds (test (e) on the real
arrays, also through unifloral_import.make_transition in the training environment); the exact weights are finite and
in [0.135, 7.389] and equal the logs' truth/weight_strong; the estimated weights are the weighting test's classifier's.
Real ThrusterChain arrays are only read and compared: no learner step, no band fit, no BCA code, no return.

An experiment-local attachment (plan section 3.1) that changes none of the repository's declared methods.

Driver (data environment: it reads the weighting test's logs with experiments.shift_iw, and JAX on the processor for
jnp.roll):
  cd /root/bca-hosts && PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu SHIFT_HOSTS_DEVICE=cpu \\
      taskset -c 20-21 /opt/bca-shift-venv/bin/python -m experiments.shift_hosts.tests.test_prepare
It starts the training environment's half in /opt/bca-venv (with unifloral_import.launch_env()):
  python -m experiments.shift_hosts.tests.test_prepare --role training --dir <folder>
which reads every prepared file with handoff.read under numpy 1.26.4 and builds ReBRAC's Transition with
unifloral_import.make_transition (Unifloral's own roll), comparing arrays only. The result goes to
runs/shift_hosts/stage0/data/test_prepare_cpu.json (tests/__init__.py).
"""

import argparse
import copy
import json
import math
import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from experiments.shift_hosts import data_copy as DC
from experiments.shift_hosts import handoff as H

AREA = "data"
REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "runs" / "shift_hosts" / "data"
QUALITIES = DC.QUALITIES
SEEDS = DC.SEEDS
TRAINING_PYTHON = "/opt/bca-venv/bin/python"
ALL = ("train", "full", "bank", "test_strong", "test_mild")
WEIGHTED = ("train", "bank", "test_strong", "test_mild")
LENGTH = 100
# INTERFACES.md section 6, restated here so the files are checked against the contract, not against prepare.py
CONTRACT = {"obs": ("<f4", (13,), ALL), "action": ("<f4", (4,), ALL), "reward": ("<f4", (), ALL),
            "next_obs": ("<f4", (13,), ALL), "done": ("<f4", (), ALL), "next_action": ("<f4", (4,), ALL),
            "next_action_true": ("|b1", (), ALL), "episode": ("<i4", (), ALL), "t": ("<i4", (), ALL),
            "row": ("<i8", (), ALL), "state": ("<f8", (13,), ALL), "next_state": ("<f8", (13,), ALL),
            "reward64": ("<f8", (), ALL), "current": ("<f8", (), ALL), "weight_strong": ("<f8", (), WEIGHTED),
            "weight_strong_estimated": ("<f8", (), WEIGHTED), "in_ball": ("|b1", (), ALL)}
# plan section 2.2 (read from runs/shift_iw_v5/<quality>/host/gap/seed*/record.json)
PLAN_ROWS = {"medium-expert": dict(train=(44_902, 45_372), bank=(869, 924)),
             "medium": dict(train=(47_439, 47_548), bank=(933, 964))}
TASK_RANGE = (0.135, 7.389)                    # the exact weight's range as the task states it (rounded)
EXACT_RANGE = (math.exp(-2.0), math.exp(2.0))  # w(c) = e^(4c - 2.4) on [0.1, 1.1]


def path_for(quality, seed):
    """runs/shift_hosts/data/<quality>/seed<k>/prepared.npz (INTERFACES.md section 6); restated, so this module
    imports in both environments (prepare.py imports experiments.shift, which refuses numpy 1.26.4)."""
    return OUT / quality / f"seed{int(seed)}" / "prepared.npz"


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and \
        np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes()


def keys():
    return [(q, k) for q in QUALITIES for k in SEEDS]


# ------------------------------------------------------------------------------------------------ the training side

def training_side(folder):
    """The training environment's half: every prepared file read with handoff.read under numpy 1.26.4, then ReBRAC's
    Transition built by unifloral_import.make_transition (Unifloral's jnp.roll for training rows, rebrac.py:335; the
    prepared array for bank rows), compared with the prepared arrays bit for bit. Arrays only: no step is taken."""
    from experiments.shift_hosts import unifloral_import as UI

    UI.prepare(learners=("rebrac",), device="cpu")
    import jax

    fields = ("obs", "action", "reward", "next_obs", "done")
    files, ok = {}, True
    for q, k in keys():
        arrays, manifest = H.read(path_for(q, k))
        g = {s: H.group(arrays, s) for s in ("train", "full", "bank")}
        tr = UI.make_transition("rebrac", g["train"], next_action="roll")
        fu = UI.make_transition("rebrac", g["full"], next_action="roll")
        bk = UI.make_transition("rebrac", g["bank"], next_action=g["bank"]["next_action"])
        rec = dict(npz_sha256=manifest["npz_sha256"], arrays=len(arrays),
                   hashes_as_read=all(H.array_sha256(v) == manifest["arrays"][n]["sha256"] for n, v in arrays.items()),
                   train_roll_equals_prepared=same(np.asarray(tr.next_action), g["train"]["next_action"]),
                   full_roll_equals_prepared=same(np.asarray(fu.next_action), g["full"]["next_action"]),
                   bank_next_action_passed_through=same(np.asarray(bk.next_action), g["bank"]["next_action"]),
                   transition_fields_equal=all(same(np.asarray(getattr(t, f)), g[s][f]) for t, s in
                                               ((tr, "train"), (fu, "full"), (bk, "bank")) for f in fields))
        rec["ok"] = all(v for n, v in rec.items() if isinstance(v, bool))
        ok &= rec["ok"]
        files[f"{q}/seed{k}"] = rec
    record = UI.record()
    H.write_json(Path(folder) / "training_env_report.json", dict(
        numpy=np.__version__, jax=jax.__version__, executable=sys.executable,
        devices=[str(d) for d in jax.devices()], unifloral_sha256=record.get("sha256"), files=files, ok=bool(ok)))
    return 0 if ok else 3


# ------------------------------------------------------------------------------------------------ the copy

class CopyTest(unittest.TestCase):
    """The copy of the weighting test's files (plan section 3.1, stop rule S11)."""

    @classmethod
    def setUpClass(cls):
        cls.manifest = DC.read_manifest()
        cls.entries = {e["rel"]: e for e in cls.manifest["entries"]}

    def test_copy_manifest_says_every_file_verified(self):
        """The copy manifest's sidecar verifies; every one of the listed files was checked and passed; hdf5 files by
        their .manifest.json, files with a sidecar by it, RESULTS.json by its git blob; the source was unchanged."""
        m = self.manifest
        self.assertTrue(m["all_ok"])
        self.assertEqual(m["s11"], "passed")
        self.assertTrue(m["source_unchanged"])
        self.assertEqual(m["source_changed"], [])
        self.assertEqual(m["unlisted_in_destination"], [])
        self.assertEqual(sorted(self.entries), sorted(DC.file_list(QUALITIES, SEEDS)))
        self.assertTrue(all(e["ok"] for e in m["entries"]))
        self.assertTrue(all(c["ok"] for c in m["cross_checks"]) and all(d["ok"] for d in m["deep_checks"]))
        for rel, e in self.entries.items():
            self.assertEqual(e["sha256"], e["source_sha256"], rel)
            self.assertEqual((e["copy"], e["destination"], e["source"]),
                             (DC.copy_rel(rel), str(DC.copied(rel)), str(DC.SOURCE / rel)), rel)
            if rel.endswith(".hdf5"):
                self.assertEqual((e["check"], e["recorded_in"]), ("sha256", rel + ".manifest.json"), rel)
            elif rel + ".sha256" in self.entries:
                self.assertEqual((e["check"], e["recorded_in"]), ("sha256", rel + ".sha256"), rel)
            elif rel == DC.RESULTS:
                self.assertEqual(e["check"], "git blob")
                self.assertEqual(e["git_blob"], e["expected"])
            else:
                self.assertEqual(e["check"], "bytes", rel)
                self.assertTrue(rel.endswith((".sha256", ".manifest.json", "/build.json")), rel)
        self.assertEqual(m["by_check"]["git blob"], 1)
        self.assertEqual(sum(m["by_check"].values()), len(self.entries))

    def test_copies_match_the_records_now(self):
        """Recomputed from the files now, not from the copy manifest: every copied hdf5 equals its copied manifest's
        file_sha256, every file with a sidecar equals its sidecar, and every file still has the hash the manifest
        lists (require_verified)."""
        rels = list(self.entries)
        DC.require_verified(self.manifest, rels)
        for rel in rels:
            path = DC.copied(rel)
            if rel.endswith(".hdf5"):
                body = json.loads(DC.copied(rel + ".manifest.json").read_text())
                self.assertEqual(DC.sha256_file(path), body["file_sha256"], rel)
            elif rel + ".sha256" in self.entries:
                self.assertEqual(DC.sha256_file(path), DC.sidecar_digest(DC.copied(rel + ".sha256"))[0], rel)

    def test_source_unchanged_now(self):
        """Every source file still has the size and modification time it had when it was copied (reads only)."""
        for rel, e in self.entries.items():
            s = os.stat(DC.SOURCE / rel)
            self.assertEqual((s.st_size, s.st_mtime_ns), (e["bytes"], e["mtime_ns"]), rel)

    def test_layout_matches_the_readers(self):
        """The copy layout is the task's (runs/shift_iw_v5/<rest> -> runs/shift_hosts/source/<rest>; the exploratory
        records under source/explore/), so the weighting test's readers take source/ as their root and the scoring
        side finds the start banks where evaluate.py reads them (runs/shift_hosts/source/start_banks, restated here
        as a literal path)."""
        from experiments.shift_iw import data as D

        root = REPO / "runs" / "shift_hosts" / "source"
        self.assertEqual(DC.v5_root(), root)
        self.assertEqual(D.banks_dir(DC.v5_root()), root / "start_banks")
        self.assertEqual(DC.start_banks_dir(), root / "start_banks")
        for name in DC.START_BANK_FILES:
            self.assertTrue((root / "start_banks" / name).is_file(), name)
        for q, k in keys():
            self.assertEqual(D.data_dir(DC.v5_root(), q) / f"seed{k}", root / q / "data" / f"seed{k}")
            self.assertEqual(DC.seed_dir(q, k), root / q / "data" / f"seed{k}")
            for name in DC.DATA_FILES:
                self.assertTrue((root / q / "data" / f"seed{k}" / name).is_file(), (q, k, name))
            for arm in DC.BCA_ARMS:
                self.assertEqual(DC.copied(DC.explore_record_rel(q, arm, k)),
                                 root / "explore" / q / "bca" / arm / f"seed{k}" / "record.json")
            self.assertEqual(DC.copied(DC.host_record_rel(q, "gap", k)),
                             root / q / "host" / "gap" / f"seed{k}" / "record.json")
        self.assertEqual(DC.copied(DC.RESULTS), root / "explore" / "RESULTS.json")
        with self.assertRaises(DC.CopyError):
            DC.dest_rel("runs/shift_iw_v4/expert/data/seed0/world.json")

    def test_destination_holds_only_the_copy(self):
        """Every file under runs/shift_hosts/source is a listed, verified copy or the copy manifest (with its
        sidecar); there are no symbolic links."""
        listed = {DC.dest_rel(rel) for rel in self.entries}
        found = set()
        for dirpath, dirs, files in os.walk(DC.DEST, followlinks=False):
            for name in files + dirs:
                p = Path(dirpath) / name
                self.assertFalse(p.is_symlink(), p)
                if p.is_file():
                    found.add(p.relative_to(DC.DEST).as_posix())
        self.assertEqual(found, listed | {"copy_manifest.json", "copy_manifest.json.sha256"})
        self.assertEqual(DC.unlisted(list(self.entries)), [])

    def test_every_array_verifies(self):
        """The weighting test's own readers verify every array of every copied hdf5 against its manifest."""
        from experiments.shift_iw import data as D

        banks, _ = D.load_start_banks(D.banks_dir(DC.v5_root()))
        self.assertEqual({k: v.shape for k, v in banks.items()}, {t: (1000, 13) for t in ("none", "mild", "strong")})
        for q, k in keys():
            sd = D.load_seed(k, D.data_dir(DC.v5_root(), q))
            self.assertEqual((sd.train_log.n_rows, sd.world["quality"], sd.world["seed"]), (100_000, q, k))

    def test_require_verified_refuses(self):
        """A copy whose hash differs from the manifest's, an entry not verified, or a file not listed is refused."""
        rel = DC.seed_rel("medium", 0, "world.json")
        bad = copy.deepcopy(self.manifest)
        next(e for e in bad["entries"] if e["rel"] == rel)["sha256"] = "0" * 64
        with self.assertRaises(DC.CopyError):
            DC.require_verified(bad, [rel])
        bad = copy.deepcopy(self.manifest)
        next(e for e in bad["entries"] if e["rel"] == rel)["ok"] = False
        with self.assertRaises(DC.CopyError):
            DC.require_verified(bad, [rel])
        bad = copy.deepcopy(self.manifest)
        next(e for e in bad["entries"] if e["rel"] == rel)["copy"] = "runs/shift_hosts/source/runs/" + rel
        with self.assertRaises(DC.CopyError):
            DC.require_verified(bad, [rel])
        with self.assertRaises(DC.CopyError):
            DC.require_verified(self.manifest, ["runs/shift_iw_v5/expert/data/seed0/world.json"])
        self.assertEqual(DC.require_verified(self.manifest, [rel]), {rel: self.entries[rel]["sha256"]})


# ------------------------------------------------------------------------------------------------ the prepared files

class PreparedTest(unittest.TestCase):
    """The prepared files (INTERFACES.md section 6) against the copied logs and records."""

    @classmethod
    def setUpClass(cls):
        from experiments.shift_iw import data as D

        cls.files = {key: H.read(path_for(*key)) for key in keys()}
        cls.data = {(q, k): D.load_seed(k, D.data_dir(DC.v5_root(), q)) for q, k in keys()}

    def record(self, quality, kind, seed):
        return H.read_json(DC.copied(DC.host_record_rel(quality, kind, seed)))["result"]["data"]

    def test_files_verify_and_follow_the_contract(self):
        """Every file reads with all its sidecar and content checks; it holds exactly the contract's arrays, with the
        contract's dtypes and shapes; the sets have 50,000 (full) and 30,000 (test) rows."""
        for (q, k), (arrays, manifest) in self.files.items():
            meta = manifest["meta"]
            self.assertEqual((meta["schema"], meta["quality"], meta["seed"]), ("shift-hosts-prepared-v1", q, k))
            want = {f"{s}/{f}" for f, (_, _, sets) in CONTRACT.items() for s in sets}
            self.assertEqual(set(arrays), want, (q, k))
            for name, x in arrays.items():
                s, f = name.split("/")
                dtype, tail, _ = CONTRACT[f]
                self.assertEqual((x.dtype.str, x.shape), (dtype, (meta["rows"][s],) + tail), (q, k, name))
            self.assertEqual((meta["rows"]["full"], meta["rows"]["test_strong"], meta["rows"]["test_mild"]),
                             (50_000, 30_000, 30_000))
            self.assertTrue(meta["s11"]["ok"])

    def test_sources_are_the_verified_copies(self):
        """The manifest's source files are this seed's verified copies (its data folder, both host records, the band
        record and their sidecars), each still with the listed SHA-256, and each copied from the path it names."""
        copy_manifest = DC.read_manifest()
        for (q, k), (_, manifest) in self.files.items():
            src = manifest["meta"]["source"]
            rels = DC.seed_rels(q, k)
            verified = DC.require_verified(copy_manifest, rels)
            self.assertEqual(src["files"], {DC.copy_rel(r): verified[r] for r in rels}, (q, k))
            self.assertEqual(src["originals"], {DC.copy_rel(r): f"{DC.SOURCE}/{r}" for r in rels}, (q, k))
            for path, digest in src["files"].items():
                self.assertEqual(DC.sha256_file(REPO / path), digest, path)
            self.assertEqual(src["world_json_sha256"], DC.sha256_file(DC.seed_dir(q, k) / "world.json"))

    def test_counts_equal_the_records(self):
        """Stop rule S11: the kept training and bank rows equal the weighting test's host/gap records, the full log
        its host/full records, and both lie in plan section 2.2's ranges."""
        for (q, k), (arrays, manifest) in self.files.items():
            rows = manifest["meta"]["rows"]
            gap, full = self.record(q, "gap", k), self.record(q, "full", k)
            self.assertEqual((rows["train"], rows["bank"]), (gap["rows"], gap["bank_rows"]), (q, k))
            self.assertEqual(rows["full"], full["rows"])
            lo, hi = PLAN_ROWS[q]["train"]
            self.assertTrue(lo <= rows["train"] <= hi, (q, k))
            lo, hi = PLAN_ROWS[q]["bank"]
            self.assertTrue(lo <= rows["bank"] <= hi, (q, k))

    def test_rows_reproduce_the_record_hashes(self):
        """The row indices hash (guard.tree_sha256, the weighting test's learners.tree_sha256) to the recorded
        rows / bank_rows hashes, and equal the kept rows recomputed here: split 0 (and the bank) minus the ball."""
        from experiments.shift_hosts import guard

        for (q, k), (arrays, _) in self.files.items():
            gap, full = self.record(q, "gap", k), self.record(q, "full", k)
            self.assertEqual(guard.tree_sha256(arrays["train/row"]), gap["sha256"]["rows"], (q, k))
            self.assertEqual(guard.tree_sha256(arrays["bank/row"]), gap["sha256"]["bank_rows"], (q, k))
            self.assertEqual(guard.tree_sha256(arrays["full/row"]), full["sha256"]["rows"], (q, k))
            log = self.data[(q, k)].train_log
            inb = np.asarray(log["truth/in_ball"], bool)
            split = np.asarray(log["infos/split"])
            self.assertTrue(same(arrays["train/row"], np.flatnonzero((split == 0) & ~inb).astype(np.int64)))
            self.assertTrue(same(arrays["full/row"], np.flatnonzero(split == 0).astype(np.int64)))
            bank = log.bank_rows
            self.assertTrue(same(arrays["bank/row"], bank[~inb[bank]]))

    def test_fields_equal_the_logs(self):
        """Every field equals the copied log at its row, bit for bit; done is 0; the observations are the float32 of
        the exact states (data.py:322-325); current is state[:, 10]."""
        pairs = dict(obs="observations", action="actions", reward="rewards", next_obs="next_observations",
                     episode="infos/episode", t="infos/t", state="infos/state", next_state="infos/next_state",
                     reward64="infos/reward64", in_ball="truth/in_ball")
        for (q, k), (arrays, _) in self.files.items():
            sd = self.data[(q, k)]
            logs = dict(train=sd.train_log, full=sd.train_log, bank=sd.train_log, test_strong=sd.tests["strong"],
                        test_mild=sd.tests["mild"])
            for s, log in logs.items():
                g = H.group(arrays, s)
                r = g["row"]
                for f, key in pairs.items():
                    self.assertTrue(same(g[f], np.asarray(log[key])[r]), (q, k, s, f))
                self.assertTrue(same(g["current"], np.asarray(log["infos/state"])[r, 10]))
                self.assertTrue(same(g["done"], np.zeros(len(r), np.float32)))
                self.assertTrue(same(g["obs"], g["state"].astype(np.float32)), (q, k, s))
                self.assertTrue(same(g["next_obs"], g["next_state"].astype(np.float32)), (q, k, s))
                self.assertFalse(np.asarray(log["terminals"])[r].any())

    def test_roll_rule(self):
        """Test (e) on the real arrays and ReBRAC's a' rule (plan section 3.8): next_action equals np.roll and JAX's
        jnp.roll of the set's commands (train, full, test sets), and the bank's equals the roll over the calibration
        episodes' kept rows at the bank rows. Restated independently: a' is the command of the next row of the roll's
        domain after the row (wrapping), and next_action_true holds exactly when that is the next log row; where it
        holds, a' is the log's command at the next row. No removed command reaches train or bank. The substituted
        shares equal the manifest's."""
        import jax.numpy as jnp

        for (q, k), (arrays, manifest) in self.files.items():
            sd = self.data[(q, k)]
            log = sd.train_log
            inb = np.asarray(log["truth/in_ball"], bool)
            split = np.asarray(log["infos/split"])
            domains = dict(train=np.flatnonzero((split == 0) & ~inb), full=np.flatnonzero(split == 0),
                           bank=np.flatnonzero((split == 1) & ~inb),
                           test_strong=np.arange(sd.tests["strong"].n_rows),
                           test_mild=np.arange(sd.tests["mild"].n_rows))
            logs = dict(train=log, full=log, bank=log, test_strong=sd.tests["strong"], test_mild=sd.tests["mild"])
            sub = manifest["meta"]["next_action"]["substituted"]
            for s in ALL:
                g = H.group(arrays, s)
                acts = np.asarray(logs[s]["actions"])
                dom = domains[s]
                if s != "bank":
                    self.assertTrue(same(g["next_action"], np.roll(g["action"], -1, axis=0)), (q, k, s))
                    self.assertTrue(same(g["next_action"], np.asarray(jnp.roll(jnp.asarray(g["action"]), -1, axis=0))))
                else:
                    pos = np.searchsorted(dom, g["row"])
                    rolled = np.asarray(jnp.roll(jnp.asarray(acts[dom]), -1, axis=0))[pos]
                    self.assertTrue(same(g["next_action"], rolled), (q, k))
                nxt_row = dom[np.searchsorted(dom, g["row"], side="right") % len(dom)]
                self.assertTrue(same(g["next_action"], acts[nxt_row]), (q, k, s))
                in_dom = np.zeros(len(acts) + 1, bool)
                in_dom[dom] = True
                true = (g["t"] < LENGTH - 1) & in_dom[g["row"] + 1]
                self.assertTrue(same(g["next_action_true"], true), (q, k, s))
                self.assertTrue(same(g["next_action"][true], acts[g["row"][true] + 1]))
                if s in ("train", "bank"):
                    self.assertFalse(inb[nxt_row].any())
                    self.assertFalse(sd.gap.contains(g["next_action"]).any())
                self.assertEqual(sub[s]["substituted"], int((~true).sum()), (q, k, s))
                c = g["current"]
                hi = c >= np.quantile(c, 0.9)
                self.assertEqual(sub[s]["highest_current_tenth"]["substituted"], int((~true[hi]).sum()))
                self.assertEqual(sub[s]["highest_current_tenth"]["rows"], int(hi.sum()))
            self.assertTrue(all(v["equal"] for v in manifest["meta"]["next_action"]["test_e"].values()))

    def test_exact_weights(self):
        """The exact weights are finite, lie in [0.135, 7.389] (and in [e^-2, e^2], the exact range), equal the logs'
        truth/weight_strong bit for bit and world.ratio(c, 'strong') = 0.0907 e^(4c); the bank's Kish effective size
        equals the manifest's."""
        from experiments.shift_iw import world as W

        for (q, k), (arrays, manifest) in self.files.items():
            sd = self.data[(q, k)]
            logs = dict(train=sd.train_log, bank=sd.train_log, test_strong=sd.tests["strong"],
                        test_mild=sd.tests["mild"])
            for s in WEIGHTED:
                g = H.group(arrays, s)
                w = g["weight_strong"]
                self.assertTrue(np.isfinite(w).all())
                self.assertTrue(((w >= TASK_RANGE[0]) & (w <= TASK_RANGE[1])).all(), (q, k, s, w.min(), w.max()))
                self.assertTrue(((w >= EXACT_RANGE[0] * (1 - 1e-12)) & (w <= EXACT_RANGE[1] * (1 + 1e-12))).all())
                self.assertTrue(same(w, np.asarray(logs[s]["truth/weight_strong"])[g["row"]]), (q, k, s))
                np.testing.assert_allclose(w, W.ratio(g["current"], "strong"), rtol=1e-15, atol=0)
                np.testing.assert_allclose(w, math.exp(-2.4) * np.exp(4.0 * g["current"]), rtol=1e-13)
            bw = arrays["bank/weight_strong"]
            self.assertEqual(manifest["meta"]["weights"]["bank_kish"]["exact"], float(bw.sum() ** 2 / (bw ** 2).sum()))

    def test_estimated_weights(self):
        """The estimated weights are finite and positive, and equal, bit for bit, the weights of a classifier fitted
        here as the weighting test fitted it (classifier.fit_samples(samples, 'strong'), run.py:560-563), whose record
        equals the weighting test's band record (run.py:762-766)."""
        from experiments.shift_iw import classifier as CL

        for (q, k), (arrays, manifest) in self.files.items():
            clf = CL.fit_samples(self.data[(q, k)].samples, "strong")
            band = H.read_json(DC.copied(DC.band_record_rel(q, k)))["classifiers"]["strong"]["classifier"]
            self.assertEqual(json.loads(json.dumps(H.plain(clf.describe()))), band, (q, k))
            for s in WEIGHTED:
                g = H.group(arrays, s)
                we = g["weight_strong_estimated"]
                self.assertTrue(np.isfinite(we).all() and (we > 0).all())
                self.assertTrue(same(we, clf.weight(g["obs"].astype(np.float64))), (q, k, s))
            bw = arrays["bank/weight_strong_estimated"]
            self.assertEqual(manifest["meta"]["weights"]["bank_kish"]["estimated"],
                             float(bw.sum() ** 2 / (bw ** 2).sum()))

    def test_ball_and_in_ball(self):
        """The ball is world.json's, the same for both qualities of a seed, and was recomputed from the expert's log;
        no kept training or bank row lies in it; the full log's in-ball rows are exactly the removed ones; the test
        steps' flags equal the ball and world.json's counts."""
        for (q, k), (arrays, manifest) in self.files.items():
            sd = self.data[(q, k)]
            meta = manifest["meta"]
            self.assertEqual(meta["ball"]["gap"], sd.world["gap"])
            other = [o for o in QUALITIES if o != q][0]
            self.assertEqual(meta["ball"]["gap"], self.files[(other, k)][1]["meta"]["ball"]["gap"])
            self.assertTrue(meta["ball"]["recomputed"]["equal"])
            self.assertEqual(meta["ball"]["recomputed"]["gap"], sd.world["gap"])
            self.assertFalse(arrays["train/in_ball"].any() or arrays["bank/in_ball"].any())
            self.assertEqual(int(arrays["full/in_ball"].sum()), 50_000 - meta["rows"]["train"])
            for s, tilt in (("test_strong", "strong"), ("test_mild", "mild")):
                g = H.group(arrays, s)
                self.assertTrue(same(g["in_ball"], sd.gap.contains(g["action"])))
                self.assertEqual(int(g["in_ball"].sum()), sd.world["test_in_ball_steps"][tilt])

    def test_reproducible(self):
        """Preparing one seed again from the copies gives the same arrays bit for bit and the same manifest meta
        (apart from the ball's recomputation, skipped here)."""
        from experiments.shift_hosts import prepare as P

        q, k = "medium", 2
        arrays, meta, _ = P.prepare_seed(q, k, copy_manifest=DC.read_manifest(), ball=None)
        stored, manifest = self.files[(q, k)]
        self.assertEqual(sorted(arrays), sorted(stored))
        self.assertTrue(all(same(arrays[n], stored[n]) for n in arrays))
        a, b = H.plain(meta), copy.deepcopy(manifest["meta"])
        a["ball"].pop("recomputed")
        b["ball"].pop("recomputed")
        self.assertEqual(a, b)


# ------------------------------------------------------------------------------------------------ the other environment

class TrainingEnvironmentTest(unittest.TestCase):
    """Test (e) through the runner's own path: in /opt/bca-venv (numpy 1.26.4), unifloral_import.make_transition's
    jnp.roll equals every prepared train/next_action and full/next_action, the bank's array passes through unchanged,
    and the Transition's fields equal the prepared arrays (arrays only, on the processor)."""

    @classmethod
    def setUpClass(cls):
        from experiments.shift_hosts import tests as T
        from experiments.shift_hosts import unifloral_import as UI

        if not Path(TRAINING_PYTHON).exists():
            raise unittest.SkipTest(f"{TRAINING_PYTHON} does not exist")
        cls.dir = T.out_dir(AREA) / "training_env"
        cls.dir.mkdir(parents=True, exist_ok=True)
        env = UI.launch_env(dict(os.environ, JAX_PLATFORMS="cpu", SHIFT_HOSTS_DEVICE="cpu"))
        cls.proc = subprocess.run([TRAINING_PYTHON, "-m", "experiments.shift_hosts.tests.test_prepare", "--role",
                                   "training", "--dir", str(cls.dir)], cwd=str(REPO), env=env, capture_output=True,
                                  text=True, timeout=1800)

    def test_training_environment_reads_and_rolls(self):
        self.assertEqual(self.proc.returncode, 0, (self.proc.stdout + self.proc.stderr)[-3000:])
        rep = H.read_json(self.dir / "training_env_report.json")
        self.assertEqual(rep["numpy"], "1.26.4")
        self.assertEqual(rep["executable"], TRAINING_PYTHON)
        self.assertEqual(sorted(rep["files"]), sorted(f"{q}/seed{k}" for q, k in keys()))
        for name, rec in rep["files"].items():
            self.assertEqual(rec["npz_sha256"], H.read_manifest(path_for(*name.split("/seed")))["npz_sha256"])
            self.assertTrue(rec["ok"], (name, rec))
        self.assertTrue(rep["ok"])


def _extra():
    """The table of counts and shares (from the verified manifests) for the result file."""
    rows = []
    for q, k in keys():
        try:
            meta = H.read_manifest(path_for(q, k))["meta"]
        except Exception as error:          # the result file must still be written
            rows.append(dict(quality=q, seed=k, error=repr(error)))
            continue
        sub = meta["next_action"]["substituted"]
        rows.append(dict(quality=q, seed=k, rows=meta["rows"], recorded=meta["recorded_rows"]["gap"],
                         substituted_train=sub["train"]["share"],
                         substituted_train_high=sub["train"]["highest_current_tenth"]["share"],
                         substituted_bank=sub["bank"]["share"], bank_kish=meta["weights"]["bank_kish"]))
    return dict(table=rows, copy_manifest_sha256=DC.sha256_file(DC.MANIFEST) if DC.MANIFEST.exists() else None)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--role", choices=("training",))
    p.add_argument("--dir")
    a = p.parse_args()
    if a.role == "training":
        sys.exit(training_side(a.dir))
    from experiments.shift_hosts import tests as T

    sys.exit(T.run(sys.modules[__name__], AREA, device_name="cpu", extra=_extra))
