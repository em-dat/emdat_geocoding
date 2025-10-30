#!/usr/bin/env python3
"""
Run all workflows in sequence:
1. run_preprocessing_llm.py
2. run_preprocessing_gdis.py
3. run_validation.py

- Uses the same Python interpreter that's running this script.
- Stops immediately if any step fails (non‑zero exit code).
- Prints timing information for each step.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STEPS = [
    ("LLM preprocessing", "run_preprocessing_llm.py"),
    ("GDIS preprocessing", "run_preprocessing_gdis.py"),
    ("Validation", "run_validation.py"),
]


def run_step(title: str, script: str) -> None:
    script_path = ROOT / script
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    print(f"\n=== START: {title} ({script}) ===")
    t0 = time.perf_counter()

    # Inherit environment and stream output live to console
    # If you prefer to capture to a file, use stdout=..., stderr=...
    subprocess.run([sys.executable, str(script_path)], check=True, cwd=ROOT)

    dt = time.perf_counter() - t0
    print(f"=== DONE: {title} in {dt:0.1f}s ===\n")


def main() -> None:
    for title, script in STEPS:
        run_step(title, script)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        # Non-zero exit from a child script
        print(f"ERROR: Step failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)