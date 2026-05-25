#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run the full neuromorphic-device ML workflow.

This script is path-independent:
- works from terminal
- works from Spyder
- works regardless of current working directory
- uses the same Python executable that launched this script
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, repo_root):
    print("\nRunning:")
    print(" ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True, cwd=repo_root)


def main():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    data_dir = repo_root / "data" / "raw"
    output_root = repo_root / "outputs"

    cycle_script = script_dir / "run_cycle_analysis.py"
    mnist_script = script_dir / "run_device_fit_mnist.py"

    required_files = [
        data_dir / "S_13_P_D_C_final_50_cycle.csv",
        data_dir / "LTP_LTD_CsPbI3_perovskite.xlsx",
    ]

    missing = [p for p in required_files if not p.exists()]
    if missing:
        print("\nMissing required input files:")
        for p in missing:
            print(f"  - {p}")
        print("\nPut the required CSV/XLSX files in data/raw/ and run again.")
        raise FileNotFoundError("Missing required input files.")

    commands = [
        [
            sys.executable,
            str(cycle_script),
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_root / "cycle_analysis"),
        ],
        [
            sys.executable,
            str(mnist_script),
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_root / "device_fit_mnist"),
            "--seed",
            "0",
        ],
    ]

    for cmd in commands:
        run_command(cmd, repo_root)

    print("\nAll done.")
    print(f"Outputs saved to: {output_root}")


if __name__ == "__main__":
    main()
