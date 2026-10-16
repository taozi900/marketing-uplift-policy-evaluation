"""Run stages01-09 in a fresh directory, preserving the archived reports."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import csv
import os
import shutil
import subprocess
import sys
import time


STAGES = [
    "01_read_and_audit.py", "02_lock_split.py", "03_train_effect.py",
    "04_train_balance.py", "05_train_subgroups.py", "06_train_bayes.py",
    "07_train_ranker.py", "08_validate_policies.py", "09_final_test.py",
]


def table_equal(actual, expected):
    with actual.open(encoding="utf-8-sig", newline="") as handle:
        a = list(csv.DictReader(handle))
    with expected.open(encoding="utf-8-sig", newline="") as handle:
        b = list(csv.DictReader(handle))
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x.keys() != y.keys():
            return False
        for key in x:
            if x[key] == y[key]:
                continue
            try:
                if math.isclose(float(x[key]), float(y[key]), rel_tol=1e-10, abs_tol=1e-10):
                    continue
            except ValueError:
                pass
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="New empty directory. Default: runs/reproduction")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve() if args.output_dir else root / "runs" / "reproduction"
    if output == root or (output.exists() and any(output.iterdir())):
        raise ValueError("Choose a new empty output directory; archived or previous results are retained.")
    raw = root / "data/raw/hillstrom.csv"
    source_manifest = root / "data/raw/source_manifest.json"
    if not raw.is_file():
        raise FileNotFoundError("First run: python tools/fetch_data.py")
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    if hashlib.sha256(raw.read_bytes()).hexdigest() != manifest["csv_sha256"]:
        raise ValueError("Source CSV hash check failed.")
    output.mkdir(parents=True, exist_ok=True)
    (output / "data/raw").mkdir(parents=True, exist_ok=True)
    log_dir = output / "logs"
    log_dir.mkdir()
    for name in STAGES:
        shutil.copyfile(root / name, output / name)
    shutil.copyfile(raw, output / "data/raw/hillstrom.csv")
    shutil.copyfile(source_manifest, output / "data/raw/source_manifest.json")
    started = time.perf_counter()
    status = []
    child_env = os.environ.copy()
    child_env["PYTHONIOENCODING"] = "utf-8"
    for name in STAGES:
        print(f"Running {name} ...", flush=True)
        result = subprocess.run([sys.executable, "-B", str(output / name)], cwd=output,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=child_env)
        log_path = log_dir / (Path(name).stem + ".log")
        log_path.write_bytes(result.stdout)
        status.append({"stage": name, "exit_code": result.returncode})
        if result.returncode:
            print(result.stdout.decode("utf-8", errors="replace"))
            raise RuntimeError(f"Stage failed; see {log_path}")
    comparisons = {}
    for name in ["09_test_policy_results.csv", "09_test_policy_differences.csv"]:
        comparisons[name] = table_equal(output / "reports" / name, root / "reports" / name)
    overall = json.loads((output / "reports/09_test_overall_effect.json").read_text(encoding="utf-8"))
    expected = json.loads((root / "reports/09_test_overall_effect.json").read_text(encoding="utf-8"))
    comparisons["09_test_overall_effect.json"] = all(
        math.isclose(overall[key], expected[key], abs_tol=1e-10, rel_tol=1e-10)
        for key in ["risk_difference_pp", "difference_ci_low_pp", "difference_ci_high_pp"]
    ) and overall["groups"] == expected["groups"]
    report = {"stages": status, "seconds": round(time.perf_counter() - started, 3),
              "archive_numeric_comparisons": comparisons,
              "scope": "Fresh run in current environment; not a new independent experiment."}
    (output / "reproduction_check.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Completed. Outputs: {output}")
    print(f"Matches archived numeric tables: {all(comparisons.values())}")
    if not all(comparisons.values()):
        raise ValueError("Numeric results differ; inspect environment and logs. Frozen guards were not weakened.")


if __name__ == "__main__":
    main()
