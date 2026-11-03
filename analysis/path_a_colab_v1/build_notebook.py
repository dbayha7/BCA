"""Build the reviewable notebook from exact handoff bytes; no scientific imports."""
import base64
import hashlib
import json
from pathlib import Path
import zipfile
import zlib

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BUNDLE = Path(r"C:\Users\David\AppData\Local\Temp\BCA_PathA_Titan_20261102")
ARCHIVE_SHA = "c7f492c7e3d46a8508f8f7df30a975e216230e9e7e6289e2edb53f4793ef64fc"
LOCK = REPO / "experiments/thrusterchain_bca/zeus_20261026_validation/resolution_20261026T215529_236287Z/resolved-requirements.txt"
LOCK_SHA = "017b189f4786193db03c32fd7b3d691ae320c1274eff11e1e0c3766764655d14"


def digest(raw): return hashlib.sha256(raw).hexdigest()


def build():
    archive = BUNDLE / "path_a_titan_recovery.zip"
    assert digest(archive.read_bytes()) == ARCHIVE_SHA
    assert digest(LOCK.read_bytes()) == LOCK_SHA
    text_files, binary_files = {}, {}
    with zipfile.ZipFile(archive) as z:
        manifest_raw = z.read("TRANSFER_MANIFEST.json")
        manifest = json.loads(manifest_raw)
        for name, row in manifest["files"].items():
            raw = z.read(name)
            assert digest(raw) == row["sha256"] and len(raw) == row["bytes"]
            if name.endswith((".npz", ".msgpack")):
                binary_files[name] = row
            else:
                raw.decode("utf-8")
                text_files[name] = base64.b64encode(raw).decode()
    assert len(text_files) == 657 and len(binary_files) == 14
    extra = {
        "colab_profile.py": (HERE / "colab_profile.py").read_bytes(),
        "unpack_path_a.py": (BUNDLE / "unpack_path_a.py").read_bytes(),
        "TRANSFER_MANIFEST.json": manifest_raw,
        "resolved-requirements.txt": LOCK.read_bytes(),
    }
    packed = json.dumps({"text_files": text_files, "extras": {k: base64.b64encode(v).decode() for k, v in extra.items()},
                         "binary_files": binary_files}, sort_keys=True, separators=(",", ":")).encode()
    payload = base64.b85encode(zlib.compress(packed, 9)).decode()
    cells = []

    def md(text, name):
        for old, new in {
            "17new": "17 new", "12policies": "12 policies", "6paired": "6 paired",
            "1,536return": "1,536 return", "58hash": "58 hash", "58-package": "58-package",
            "all671": "all 671", "all657": "all 657", "The657": "The 657",
            "three seeds": "three seeds", "17calibrations": "17 calibrations",
            "least": "least", "1,536 return": "1,536 return", "17 calibrations": "17 calibrations",
            "170,000 new": "170,000 new", "170,000": "170,000", "85,000 actor": "85,000 actor",
            "of498,400": "of 498,400", "bounds remain190,014": "bounds remain 190,014",
            "updates,22": "updates, 22", "and430,575": "and 430,575",
            "remaining10,660": "remaining 10,660", "seconds,60": "seconds, 60",
            "reserved,120": "reserved, 120", "below10,480": "below 10,480",
            "libraries can": "libraries can", "downloads58": "downloads 58",
            "Python3.12": "Python 3.12", "JAX0.6.2": "JAX 0.6.2",
            "and102controller": "and 102 controller", "/294exact": "/294 exact",
            "The120-second": "The 120-second", "all17new": "all 17 new",
            "exit0": "exit 0", "requiresone": "requires one",
            "tolerance1 and": "tolerance 1 and", "tolerance1e": "tolerance 1e",
            "horizon100": "horizon 100", "seeds;12": "seeds; 12",
            "updates and17": "updates and 17", "updates;1,536": "updates; 1,536",
            "episodes and237,600": "episodes and 237,600", "update/early": "update/early",
        }.items(): text = text.replace(old, new)
        cells.append({"cell_type": "markdown", "id": name, "metadata": {}, "source": text.splitlines(True)})
    def code(text, name, hidden=False):
        cells.append({"cell_type": "code", "id": name, "metadata": {"jupyter": {"source_hidden": hidden}},
                      "execution_count": None, "outputs": [], "source": text.splitlines(True)})

    md("""# Path A: frozen two-scenario BCA study — Colab transport

**Prepared, not executed on Colab.** Last accessible Zeus observation (November 2, 2026, 12:30 p.m. EST): the 10k Honeypot seed-0 warmup is complete; its native continuation was interrupted. No completed Path A BCA policy or final evaluation is observed. Titan's execution state is **unknown**. This notebook must not run until the operator checks both machines and selects exactly one runner.

This single notebook contains all 657 audited source/config/evidence files, the additive Colab adapter, exact extraction code and the full 58-package wheel lock. It requires **one external binary input**, the existing `path_a_titan_recovery.zip` (112,349,548 bytes). Its 14 binary files are the immutable dataset, saved warmup and smoke checkpoints. They are not embedded here. No private files are downloaded from a guessed URL.

The code preserves the original experiment: Honeypot and Bottleneck datasets; native UniFloral versus original BCA actor score **Q1 − 3U**, with unchanged BC strength; three paired independent seeds; 12 final policies at 20k cumulative critic updates. It reuses the saved 10k warmup and runs only the remaining **17 training jobs plus one evaluation**. It does not restart the historical run or Path B.

**Manual sequence:** select a GPU runtime → read/fill source-status evidence → start persistent accounting → supply the ZIP → install the exact isolated runtime → prepare explicit Colab migration → run zero-update compatibility → only on its pass, launch the continuation. Run cells individually. Default switches block installation, validation and training. Do not use Run All to authorize work.

If Titan validation or training has already started, this notebook refuses migration: preserve that attempt and reconcile its work first. Operator evidence is not a distributed lock; do not run Titan/Zeus and Colab concurrently.
""", "intro")
    md("""## Frozen question, evidence and limits

- **Honeypot:** genuine supported approaches to an omitted dangerous branch. The saved 15-state controller diagnostic found safe-minus-fatal advantages 118.66–120.04. This establishes the chosen controller contrast, not learned-policy optimism.
- **Bottleneck:** genuine A→B and C→D fragments; the separating slab removes the observed connection. The corrected fitting dataset has zero intact successful routes and 716 genuine fragment pairs. This is unsupported-connector generalization, not guaranteed stitching or proof of an isolated action-only shift.
- **Primary outcome:** paired policy returns per seed on identical evaluation starts. **Secondary:** omitted-edge exposure, raw Q1 versus each policy's own finite-horizon Qπ, signed errors/MAE, over-/underestimation, helpful/harmful downward score adjustments and ranking. Common-controller probes are reported separately. A downward penalty cannot directly raise an underestimated fixed Q.
- Data roles, normalization, next-action provenance, terminal flags, exact state/time restoration, γ=.98 and horizon100 remain frozen. Error tolerance1 and numerical tie tolerance1e−10 are unchanged. No tuning from test outcomes and no Bayesian-uniqueness claim.

**Budget:** at most170,000 new critic /85,000 actor /170,000 auxiliary updates and17 calibrations;1,536 return episodes and237,600 evaluation transitions. Conservative cumulative upper bounds remain190,014 critic /95,008 actor /190,014 auxiliary updates,22 calibrations and430,575 transitions of498,400. Unknown work after the prior interruption remains unknown.

The original four-hour allowance is not reset. Of the remaining10,660 seconds,60 seconds are already reserved,120 are reserved for target compatibility, and Colab installation/upload/extraction/waits further reduce science time below10,480 seconds. A slow setup can leave too little time; the code stops instead of granting more. Runtime loss has no automatic resume. Save the persistent receipts; a new continuation requires explicit reconciliation.
""", "protocol")
    code('''import base64, hashlib, importlib.util, json, math, os, platform, shutil, signal, subprocess, sys, time, zipfile, zlib
from pathlib import Path, PurePosixPath
from datetime import datetime, timedelta, timezone
EST = timezone(timedelta(hours=-5))
ARCHIVE_SHA256 = "''' + ARCHIVE_SHA + '''"
LOCK_SHA256 = "''' + LOCK_SHA + '''"
EXPECTED_PROFILE_SHA256 = "''' + digest(extra["colab_profile.py"]) + '''"
PAYLOAD_SHA256 = "''' + digest(packed) + '''"
PAYLOAD_B85 = "''' + payload + '''"
raw_payload = zlib.decompress(base64.b85decode(PAYLOAD_B85))
if hashlib.sha256(raw_payload).hexdigest() != PAYLOAD_SHA256:
    raise ValueError("Embedded source payload differs")
PAYLOAD = json.loads(raw_payload)
print("Embedded sources:", len(PAYLOAD["text_files"]), "; external binaries:", len(PAYLOAD["binary_files"]))
''', "embedded-source", True)
    md("""## 1. Check ownership before setup

Read the latest receipts on **Zeus and Titan**, not just this notebook's snapshot. Capture their actual status in text, including observation time and which machine/paths you inspected. Keep Zeus's migration hold. Set `SOURCE_STATUS_CONFIRMED` only if Titan validation AND recovery never started and there is no additional execution. If uncertain, stop here. Do not substitute an old screenshot or simply write the desired answer.

This is the only migration case implemented. The notebook will retain your evidence and its hash alongside a separately bound Colab approval. It does not repurpose the Titan approval as proof of target compatibility.
""", "ownership-guide")
    code('''SOURCE_STATUS_CONFIRMED = False  # Set only after checking both machines.
OPERATOR = ""  # Your name.
CHECKED_AT_EST = ""  # Actual observation, e.g. YYYY-MM-DDTHH:MM:SS-05:00.
SOURCE_STATUS_CAPTURE = """"""  # Paste actual read-only receipt/process observations here.
PERSISTENT_ROOT = Path("/content/drive/MyDrive/BCA_PathA_Colab_v1")
VENV = Path("/content/path_a_colab_venv_v1")
BEGIN_SETUP = False

if not (SOURCE_STATUS_CONFIRMED and OPERATOR.strip() and CHECKED_AT_EST and SOURCE_STATUS_CAPTURE.strip()):
    raise RuntimeError("STOP: source/Titan attempt status remains unknown. No installation or experiment started.")
observed = datetime.fromisoformat(CHECKED_AT_EST)
if observed.utcoffset() != timedelta(hours=-5):
    raise ValueError("Use the actual EST observation time, including -05:00")
if not BEGIN_SETUP:
    raise RuntimeError("Review setup and accounting, then explicitly set BEGIN_SETUP=True.")
''', "ownership")
    md("""## 2. Start one persistent setup ledger

Select **Runtime → Change runtime type → GPU** first. This profile requires an existing Linux x86-64 **Python3.12** runtime. It will not replace Colab's system Python or silently substitute package versions. The isolated environment leaves notebook packages unchanged.

The following cell mounts your Google Drive only when you run it and grant access, then creates a new persistent directory. It never clears a previous attempt. Upload/extraction/install/wait time is counted from before the mount. Do not rename/delete the directory to bypass an earlier attempt. If the kernel dies before the ledger is saved, its cost is unknown and must be reconciled, not treated as zero.
""", "ledger-guide")
    code('''if "_path_a_setup_started" in globals():
    raise RuntimeError("Setup was already started in this kernel. Do not reset its clock or retry automatically.")
if sys.version_info[:2] != (3, 12) or sys.platform != "linux" or platform.machine() != "x86_64":
    raise RuntimeError("This exact wheel lock needs Linux x86-64 Python3.12; no install attempted.")
_path_a_setup_started = {"schema": "path_a.colab_setup.v1",
    "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    "monotonic_started": time.monotonic(), "wall_started": time.time(),
    "timing_complete": True, "unknown_runtime_seconds": 0,
    "includes_install_upload_extract_and_waits": True}
from google.colab import drive
drive.mount("/content/drive", force_remount=False)
if not str(PERSISTENT_ROOT).startswith("/content/drive/MyDrive/"):
    raise ValueError("Choose an explicit persistent MyDrive subdirectory")
PERSISTENT_ROOT.mkdir(parents=True, exist_ok=False)
SESSION = PERSISTENT_ROOT
REPO = SESSION / "BCA-PathA"
LEDGER = SESSION / "SETUP_LEDGER.json"
def exclusive_json(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\\n"); f.flush(); os.fsync(f.fileno())
exclusive_json(LEDGER, _path_a_setup_started)
with (SESSION / "SOURCE_STATUS_CAPTURE.txt").open("x", encoding="utf-8") as f:
    f.write(SOURCE_STATUS_CAPTURE)
def setup_elapsed():
    ledger = json.loads(LEDGER.read_text())
    if ledger["boot_id"] != Path("/proc/sys/kernel/random/boot_id").read_text().strip():
        raise RuntimeError("Different runtime boot; no automatic continuation")
    values = [time.monotonic()-ledger["monotonic_started"], time.time()-ledger["wall_started"]]
    if not all(math.isfinite(x) and x >= 0 for x in values):
        raise ValueError("Invalid setup clock")
    return max(values)
def setup_time_left():
    remaining = math.floor(10660 - 60 - 120 - setup_elapsed())
    if remaining <= 15:
        raise RuntimeError("Existing allowance exhausted; preserve evidence and stop")
    return remaining
print("Persistent ledger:", LEDGER, "Remaining science upper bound:", setup_time_left(), "seconds")
''', "ledger")
    md("""## 3. Supply the exact binary archive and reconstruct the embedded code

Choose either an existing local/Drive ZIP path or the browser file picker. Required ZIP SHA256:
`c7f492c7e3d46a8508f8f7df30a975e216230e9e7e6289e2edb53f4793ef64fc`.

The archive is verified before extraction. All671 file hashes are checked; conflicting files, traversal paths, links and duplicate names are rejected. The657 embedded text files must match the ZIP byte for byte. The historical CRLF approval bytes are deliberately preserved; no approval is reinterpreted or resealed. No model is loaded here.
""", "input-guide")
    code('''INPUT_ZIP = ""  # Optional existing path. Leave empty for browser upload.
setup_time_left()
if INPUT_ZIP:
    source_zip = Path(INPUT_ZIP).resolve()
else:
    from google.colab import files
    print("Choose path_a_titan_recovery.zip (112.35 MB). Do not upload unrelated files.")
    uploaded = files.upload()
    if set(uploaded) != {"path_a_titan_recovery.zip"}:
        raise ValueError("Exactly the expected archive is required")
    source_zip = Path("path_a_titan_recovery.zip").resolve()
    del uploaded
def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""): h.update(block)
    return h.hexdigest()
if sha_file(source_zip) != ARCHIVE_SHA256:
    raise ValueError("Wrong archive; no extraction")
setup_time_left()
unpacker_raw = base64.b64decode(PAYLOAD["extras"]["unpack_path_a.py"])
unpack_path = SESSION / "unpack_path_a.py"
with unpack_path.open("xb") as f: f.write(unpacker_raw)
namespace = {"__name__": "notebook_embedded_unpacker"}
exec(compile(unpacker_raw, str(unpack_path), "exec"), namespace)
extraction = namespace["unpack"](source_zip, ARCHIVE_SHA256, REPO)
for name, encoded in PAYLOAD["text_files"].items():
    if (REPO / name).read_bytes() != base64.b64decode(encoded):
        raise ValueError("Embedded code/evidence differs from archive: " + name)
extra_dir = REPO / "analysis/path_a_colab_v1"
extra_dir.mkdir(parents=True, exist_ok=False)
for name in ("colab_profile.py", "resolved-requirements.txt"):
    with (extra_dir / name).open("xb") as f: f.write(base64.b64decode(PAYLOAD["extras"][name]))
if sha_file(extra_dir / "colab_profile.py") != EXPECTED_PROFILE_SHA256 or sha_file(extra_dir / "resolved-requirements.txt") != LOCK_SHA256:
    raise ValueError("Embedded additive profile/lock changed")
exclusive_json(SESSION / "EXTRACTION.json", extraction)
print(extraction)
print("Exact source files available for inspection at", REPO)
''', "extract")
    md("""## 4. Install the exact isolated runtime — manual, no scientific work

This cell downloads58 hash-pinned binary wheels from official PyPI; GPU libraries can require several GB of disk/network transfer. It creates a new venv without system-site packages. No packages are installed until you set the switch. No training, calibration or simulator call is made. Missing wheels, incompatible Python or dependency conflicts stop the process. Do not loosen the lock to force a pass.

Installation and all waiting consume the same remaining allowance. Colab GPU types and session limits vary; a GPU allocation is not a guarantee this study finishes. See [Colab FAQ](https://research.google.com/colaboratory/faq.html). This uses the historical locked JAX0.6.2 stack, not the latest install recipe from [JAX documentation](https://docs.jax.dev/en/latest/installation.html).
""", "install-guide")
    code('''INSTALL_EXACT_RUNTIME = False
if not INSTALL_EXACT_RUNTIME:
    raise RuntimeError("Installation is disabled. Review the lock and available budget, then enable this cell explicitly.")
def run_owned(args, log_path, timeout_seconds, env=None):
    process = None
    started = time.monotonic()
    timeout_tool = shutil.which("timeout")
    if timeout_tool is None:
        raise RuntimeError("GNU timeout is required for an independent process cap; nothing dispatched")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("No remaining process allowance")
    command = [timeout_tool, "--signal=TERM", "--kill-after=5s", str(timeout_seconds) + "s"] + [str(x) for x in args]
    try:
        with Path(log_path).open("x", encoding="utf-8") as log:
            # Independent OS timeout remains active even if notebook polling or Drive I/O stalls.
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       env=env, start_new_session=True)
            with Path(log_path).open(encoding="utf-8", errors="replace") as reader:
                while process.poll() is None:
                    chunk = reader.read()
                    if chunk: print(chunk, end="", flush=True)
                    if time.monotonic()-started >= timeout_seconds:
                        raise TimeoutError("Existing process allowance reached")
                    time.sleep(.5)
                print(reader.read(), end="", flush=True)
            if process.returncode:
                raise RuntimeError("Process failed; preserve log and do not retry: " + str(process.returncode))
    finally:
        if process is not None and process.poll() is None:
            try: os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try: os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                process.wait()
if VENV.exists():
    raise FileExistsError("Isolated environment already exists; do not reset a partial attempt")
exclusive_json(SESSION / "INSTALL_ATTEMPT.json", {"started_est": datetime.now(EST).isoformat(),
    "lock_sha256": LOCK_SHA256, "executable": str(VENV / "bin/python"), "no_automatic_retry": True})
run_owned([sys.executable, "-m", "venv", str(VENV)], SESSION / "venv.log", setup_time_left()-5)
PYTHON = VENV / "bin/python"
run_owned([PYTHON, "-m", "pip", "--isolated", "install", "--disable-pip-version-check",
    "--require-hashes", "--only-binary=:all:", "--no-deps", "--index-url", "https://pypi.org/simple",
    "-r", extra_dir / "resolved-requirements.txt"], SESSION / "install.log", setup_time_left()-5)
run_owned([PYTHON, "-m", "pip", "--isolated", "check"], SESSION / "pip_check.log", setup_time_left()-5)
exclusive_json(SESSION / "INSTALL_RESULT.json", {"status": "installed_pending_gpu_compatibility",
    "elapsed_setup_seconds": setup_elapsed(), "training_started": False})
print("Installation completed. GPU compatibility has NOT been established.")
''', "install")
    md("""## 5. Bind this Colab instance and explicitly select it as sole runner

This creates a separate Colab manifest/approval, preserving all original manifests and attempts. Read the exact statement printed below. Type `APPROVE COLAB MIGRATION` only after confirming the supplied evidence remains current and both other machines remain stopped for Path A. A previous Titan validation/recovery cannot be relabeled as a new Colab attempt.
""", "approve-guide")
    code('''profile_path = extra_dir / "colab_profile.py"
spec = importlib.util.spec_from_file_location("path_a_colab_profile", profile_path)
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
evidence_path = SESSION / "SOURCE_STATUS_CAPTURE.txt"
status_path = SESSION / "MIGRATION_STATUS.json"
exclusive_json(status_path, {"schema": "path_a.colab_migration.v1", "selected_owner": "Colab",
    "titan_validation": "not_started", "titan_recovery": "not_started", "zeus_hold_preserved": True,
    "other_execution": "none", "additional_scientific_work": 0, "additional_runtime_seconds_known": True,
    "additional_runtime_seconds": 0, "checked_at_est": CHECKED_AT_EST, "operator": OPERATOR,
    "attestation": C.MIGRATION_ATTESTATION, "evidence": [{"path": str(evidence_path), "sha256": sha_file(evidence_path)}]})
print(C.prepare(REPO, status_path, LEDGER, str(PYTHON)))
print(C.APPROVAL_STATEMENT)
if input("Type APPROVE COLAB MIGRATION to record this exact approval: ").strip() != "APPROVE COLAB MIGRATION":
    raise RuntimeError("No approval recorded; no validation or training started")
C.authorize(REPO, OPERATOR, C.APPROVAL_STATEMENT)
print("Approval bound to", sha_file(extra_dir / "COLAB_MANIFEST.json"))
''', "approve")
    md("""## 6. Zero-update Colab compatibility — once

This checks strict source/data/dependencies, actual CUDA runtime, complete warmup restore, eight-row actor/Q1/uncertainty inference and102controller calls/294exact action comparisons. **Zero optimizer updates, calibrations, resets or simulator steps.** The120-second cap includes cleanup. The real-update/early-termination smoke already passed on Zeus; this check does not establish bitwise equivalence of all future updates on a different GPU.

Failure is a technical outcome. Preserve its receipt; do not retry, loosen comparisons or train anyway.
""", "validate-guide")
    code('''RUN_ZERO_UPDATE_VALIDATION = False
if not RUN_ZERO_UPDATE_VALIDATION:
    raise RuntimeError("Target validation is disabled until explicitly selected.")
runtime_env = os.environ.copy()
runtime_env.update(JAX_PLATFORMS="cuda", JAX_DEFAULT_MATMUL_PRECISION="highest",
                   JAX_ENABLE_X64="false", JAX_DISABLE_JIT="false", XLA_FLAGS="--xla_gpu_triton_gemm_any=True")
runtime_env.pop("OPENBLAS_CORETYPE", None)
runtime_env["PYTHONDONTWRITEBYTECODE"] = "1"
run_owned([PYTHON, "-B", "-u", profile_path, "--repo", REPO, "--validate"],
          SESSION / "validation_console.log", 115, env=runtime_env)
validation = json.loads((REPO / C.RELATIVE / "colab_validation_1/RESULT.json").read_text())
if validation["status"] != "passed_colab_validation":
    raise RuntimeError("Compatibility did not pass; training remains blocked")
print(validation)
''', "validate")
    md("""## 7. Conditional production continuation — once

This is the only training cell. It requires the authenticated compatibility receipt and a new exclusive output reservation. It restores the existing warmup, restarts only its unfinished native branch, executes the unchanged remaining jobs and frozen evaluation. The supervisor records known/unknown work and stops all children it owns on interruption. It cannot guarantee cleanup after a Colab machine loss; persistent receipts must be reconciled before any future execution.

Do not open a second notebook or run another machine. Keep this session active. A per-job exit0 is not study completion. The hard ceiling is recomputed after setup and all waits; the notebook does not estimate completion from GPU availability.
""", "run-guide")
    code('''RUN_PATH_A = False
if not RUN_PATH_A:
    raise RuntimeError("Scientific execution is disabled. Enable only after the validated migration review.")
p = C.Profile(REPO)
p.check(approval=True)  # Source/evidence check; exact interpreter proof runs inside the isolated worker.
validation_seconds = json.loads((p.validation / "RESULT.json").read_text())["elapsed_seconds"]
cap = C.budget(json.loads(LEDGER.read_text()), validation_seconds)["scientific_cap_seconds"]
print("Remaining recovery ceiling including cleanup:", cap, "seconds")
run_owned([PYTHON, "-B", "-u", profile_path, "--repo", REPO, "--execute"],
          SESSION / "production_console.log", cap-5, env=runtime_env)
print(json.loads((p.output / "RESULT.json").read_text()))
''', "run")
    md("""## 8. Read-only progress and complete results

This cell may be run again to inspect saved files. It never resumes work. Results remain under your persistent Drive directory. For a completed experiment, require all17new training jobs plus evaluation accepted,12policies,6paired cells and1,536return episodes. Compare per-seed paired returns first; episodes are nested within three seeds, not independent training replications. Inspect signed/absolute Q errors and harmful corrections separately. Runtime success alone is not BCA efficacy.

Raw arrays/traces and per-job receipts remain in the extracted tree for review. Do not publish private data or checkpoints automatically.
""", "results-guide")
    code('''out = REPO / "experiments/thrusterchain_bca/two_scenario_path_a/run_colab_recovery_1"
if not out.exists():
    print("No Colab recovery reservation observed.")
else:
    accepted = sorted(out.glob("*/SUPERVISOR.json"))
    print("Accepted new jobs:", len(accepted), "/18 including evaluation")
    progress = sorted(out.glob("*/progress.jsonl"), key=lambda x: x.stat().st_mtime)
    if progress:
        print("Latest saved file:", progress[-1])
        print("Saved timestamp:", datetime.fromtimestamp(progress[-1].stat().st_mtime, EST).isoformat())
        print(progress[-1].read_text().splitlines()[-1])
    terminal_path = out / "RESULT.json"
    if terminal_path.exists():
        final = json.loads(terminal_path.read_text()); print(json.dumps(final, indent=2))
        if final.get("status") == "completed_colab_path_a_recovery":
            report = json.loads((out / "evaluation/report.json").read_text())
            expected = {"status": "completed_frozen_evaluation", "completed_policy_evaluations": 12,
                        "completed_paired_cells": 6, "completed_return_episodes": 1536}
            if not all(report.get(k) == v for k, v in expected.items()):
                raise ValueError("Evaluation incomplete; do not summarize partial cells as final")
            print("condition | seed | native return | BCA return | paired BCA-native")
            for cell in report["cells"]:
                print(cell["condition"], cell["seed_index"], cell["arms"]["native"]["return_mean"],
                      cell["arms"]["bca"]["return_mean"], cell["paired_returns"]["paired_mean_difference"])
            print(json.dumps(report["per_condition"], indent=2))
            print("Full signed error, MAE, exposure and ranking metrics:", out / "evaluation/report.json")
    else:
        print("No terminal receipt yet. This alone does not prove the worker is alive.")
''', "results")
    md("""## Provenance and remaining limitations

This notebook is a transport of an existing approved study, not a new environment or BCA method. Original source/approvals/failed attempts stay byte-exact. Colab's profile and fresh operator migration evidence are separate. The notebook cannot inspect Titan remotely or supply its missing status.

CPU preparation verifies notebook syntax, all657 embedded source bytes, exact671-file extraction, the58-package lock, blocking/default behavior and profile accounting fixtures. **No Colab GPU, install, model, training or simulator execution occurred while preparing it.** Actual target compatibility remains mandatory. Google Drive durability and Colab session availability are runtime dependencies, not guaranteed by local tests. A runtime/kernel restart is not an automatic resume; retain attempts and reconcile costs.

Repository skills applied: `offline-rl-experiment-design`, `bca-code-and-calibration-audit`, `thrusterchain-benchmark-validation`, plus installed `offline-rl-audit`, `offline-rl-data-audit`, `offline-rl-experiments`. No notebook-specific skill was available. Notebook structure follows the [Jupyter nbformat specification](https://nbformat.readthedocs.io/en/latest/format_description.html).
""", "limitations")
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"}, "accelerator": "GPU",
        "colab": {"name": "Path_A_Colab.ipynb", "provenance": []},
        "path_a": {"archive_sha256": ARCHIVE_SHA, "payload_sha256": digest(packed),
            "profile_sha256": digest(extra["colab_profile.py"]), "executed": False}}, "cells": cells}
    path = HERE / "Path_A_Colab.ipynb"
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path.read_bytes()),
                      "code_cells": sum(c["cell_type"] == "code" for c in cells), "embedded_sources": len(text_files)}))


if __name__ == "__main__": build()
