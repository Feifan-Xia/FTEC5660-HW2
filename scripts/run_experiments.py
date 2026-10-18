"""Run real public/attack evaluations and retain CSVs, logs and tool traces."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hw2 import MODEL_NAME, load_env_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, choices=range(1, 4), default=1)
    args = parser.parse_args()
    load_env_file(ROOT / ".env")
    if not os.getenv("DEEPSEEK_API_KEY", "").strip():
        raise SystemExit("Set DEEPSEEK_API_KEY in the local .env before running real evaluations.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = ROOT / "artifacts" / "experiments" / stamp
    folder.mkdir(parents=True)
    summary = {"model": MODEL_NAME, "started_utc": stamp, "runs": []}
    summary["source_sha256"] = hashlib.sha256((ROOT / "hw2.py").read_bytes()).hexdigest()
    for dataset in ("public_test", "task2"):
        for repeat in range(1, args.repeats + 1):
            label = f"{dataset}_run_{repeat}"
            env = dict(os.environ, PYTHONIOENCODING="utf-8", HW2_TRACE_DIR=str(folder / label / "traces"))
            started = time.monotonic()
            process = subprocess.run(
                [sys.executable, "hw2.py", "--cv-folder", dataset],
                cwd=ROOT, env=env, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=1800,
            )
            print(process.stdout, end="", flush=True)
            (folder / f"{label}.log").write_text(process.stdout + process.stderr, encoding="utf-8")
            result = {"dataset": dataset, "repeat": repeat, "returncode": process.returncode,
                      "seconds": round(time.monotonic() - started, 2)}
            if process.returncode:
                summary["runs"].append(result)
                (folder / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
                raise SystemExit(f"Run failed; inspect {folder / (label + '.log')}")
            output = ROOT / "results.csv"
            contents = output.read_text(encoding="utf-8")
            (folder / f"{label}.csv").write_text(contents, encoding="utf-8")
            rows = list(csv.DictReader(contents.splitlines()))
            result["scores"] = {row["cv"]: float(row["score"]) for row in rows}
            if dataset == "public_test":
                result["correct"] = sum(row["correctness"] == "correct" for row in rows)
                result["total"] = len(rows)
                result["accuracy"] = result["correct"] / len(rows)
            else:
                result["attack_accepted"] = result["scores"]["adversarial_cv.pdf"] > 0.5
            traces = list((folder / label / "traces").glob("**/*.json"))
            result["execution_failures"] = sum(
                json.loads(path.read_text(encoding="utf-8"))["error"] is not None
                for path in traces
            )
            summary["runs"].append(result)
            (folder / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved evidence to {folder}")


if __name__ == "__main__":
    main()
