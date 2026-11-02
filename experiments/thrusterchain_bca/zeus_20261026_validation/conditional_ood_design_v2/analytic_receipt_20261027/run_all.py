"""Rerun the analytic-check scripts behind REVIEW_20261027.md and record a receipt.

Read-only by construction: every script reads only the JSON files listed in RUNS (and coverage_check.py also
hashes, without decoding, the dataset file and the v1 source-receipt files). No simulator package, model, training
code or dataset array is loaded. Outputs go to ./outputs and ./RECEIPT.json beside this file.

Usage (from the repository root):
    .venv/Scripts/python.exe -I -B <this folder>/run_all.py
"""
import hashlib, json, os, platform, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
SCRIPTS, OUT = HERE / "scripts", HERE / "outputs"
ENV = "docs/thrusterchain/site/data/environment.json"
RES = "docs/thrusterchain/site/data/results.json"
DREC = "experiments/thrusterchain_bca/zeus_20261026_validation/bca_one_sided_v1/run/data_receipt.json"
BIND = "experiments/thrusterchain_bca/titan_20261026/payload/unifloral_checkpoint_v1/train_20261023T183248_552877Z/bindings.json"
RETRY = "experiments/thrusterchain_bca/zeus_20261026_validation/support_shift_pilot_v1_retry_20261027/run/seed0_restricted/warmup/data_receipt.json"
V1REC = "experiments/thrusterchain_bca/zeus_20261026_validation/conditional_ood_design_v1/SOURCE_RECEIPT.json"
NPZ = "experiments/thrusterchain_bca/titan_20261026/payload/transfer_bundle/research_runs/pilot_20261023T115408Z/regular_1280_2026102301.npz"

# script, command-line arguments (repo-relative), files read through hard-coded paths, what it supports
RUNS = [
    ("coverage_check.py", [], [ENV, V1REC, NPZ], "receipt hashes; band shares; P(g>=0.4); hole rows; episodes without hole rows"),
    ("prob.py", [], [], "P(g>=thr) by quadrature; strong-current share; float32 band edges; episodes always high/low"),
    ("arith.py", [], [], "P(g>=0.4); hole rows; calibration rows in hole; band binomials; 4-D ball"),
    ("a2.py", [], [], "P(all three a0 bands nonempty); training-time extrapolation"),
    ("verify_geometry.py", [], [ENV, RES], "LQ curvature; P(a0*<=-k|g); least-squares counter; development-bank band errors and actor range"),
    ("verify_findings.py", [], [ENV, RES], "covariance vs receipt std; neighbour radius; value gaps under neutral continuations"),
    ("indep_check.py", [ENV, DREC], [], "depth shares; strong-current load; per-bin radius quantiles; LQ pinned-optimum shares"),
    ("geom.py", [ENV, DREC], [], "per-bin coverage; binomials; float32 edges; pair-distance probabilities"),
    ("geom2.py", [ENV, DREC], [], "LQ a0* among high-load states; same-side radius quantiles"),
    ("lq2.py", [ENV], [], "LQ curvature cross-check; float-driven a0* spread"),
    ("cond_ood_arith.py", [], [ENV, DREC], "load shares; ring/band rows; covariance; radius; value gaps (second implementation)"),
    ("bank.py", [RES], [], "development-bank actor a0 range; in-band candidates"),
    ("cal.py", [DREC], [], "calibration episode identity"),
    ("cal2.py", [DREC], [], "calibration overlap checks"),
    ("cmp.py", [DREC, BIND], [], "calibration rows equal bindings split receipt"),
    ("ss2.py", [DREC, RETRY], [], "support-shift retry used the same calibration episodes"),
]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    py = sys.executable
    numpy_version = subprocess.run([py, "-I", "-c", "import numpy; print(numpy.__version__)"],
                                   capture_output=True, text=True).stdout.strip()
    inputs = sorted({p for _, a, r, _ in RUNS for p in a + r})
    receipt = {
        "schema": "bca.review_analytic_receipt.v1",
        "purpose": "Reproduce the analytic figures in ../REVIEW_20261027.md from saved scripts (closed-form and Monte Carlo arithmetic only).",
        "utc_started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "interpreter": py, "python": platform.python_version(), "numpy": numpy_version,
        "flags": ["-I", "-B", "-X", "utf8"], "cwd": str(REPO),
        "read_only": {"simulator_package_imported": False, "dataset_arrays_decoded": False,
                      "dataset_file_hashed_only": True, "protected_splits_read": False, "training_or_fitting": False},
        "inputs": [{"path": p, "bytes": (REPO / p).stat().st_size, "sha256": sha(REPO / p)} for p in inputs],
        "runs": [],
    }
    for name, args, reads, supports in RUNS:
        script = SCRIPTS / name
        # -X utf8: printing only; the Windows console code page cannot encode symbols some scripts print.
        cmd = [py, "-I", "-B", "-X", "utf8", str(script)] + [str(REPO / a) for a in args]
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace")
        dt = time.perf_counter() - t0
        out_path, err_path = OUT / (name[:-3] + ".stdout.txt"), OUT / (name[:-3] + ".stderr.txt")
        out_path.write_text(proc.stdout, encoding="utf-8", newline="\n")
        err_path.write_text(proc.stderr, encoding="utf-8", newline="\n")
        receipt["runs"].append({
            "script": f"scripts/{name}", "script_sha256": sha(script), "args": args, "hard_coded_reads": reads,
            "supports": supports, "exit_code": proc.returncode, "seconds": round(dt, 2),
            "stdout": f"outputs/{out_path.name}", "stdout_sha256": sha(out_path),
            "stderr": f"outputs/{err_path.name}", "stderr_bytes": err_path.stat().st_size,
        })
        print(f"{name:22s} exit {proc.returncode}  {dt:7.1f} s  stderr {err_path.stat().st_size} B", flush=True)
    receipt["utc_finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (HERE / "RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
