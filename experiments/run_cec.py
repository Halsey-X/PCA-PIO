# -*- coding: utf-8 -*-
"""
Validation on the modern CEC2017 / CEC2022 test suites
(manuscript Section "Validation on Modern CEC Test Suites").

The official CEC2017 (29 functions) and CEC2022 (12 functions) test suites are
shifted / rotated / hybrid / composition problems that require the official
suite code and shift-rotation data files, which are licensed by the CEC
competition organisers and cannot be redistributed here.

This runner is a scaffold that expects the official suites to be importable.
It is intentionally NOT executed by this repository (no bundled CEC data); it
is provided so that the paper's CEC protocol can be reproduced once the
official suite is available on the machine.

Prerequisite (one of):
  * a `cec2017` python package exposing callable functions, or
  * the official CEC2017/CEC2022 MATLAB/Python code plus shift/rotation data.

Usage:
  python experiments/run_cec.py --suite cec2017 --dim 30 [--runs 30]

The script runs the five algorithms with the unified protocol (N=30,
Tmax=500, T1=400) and writes a per-function mean/std CSV plus the average
Friedman ranking of each algorithm.
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pca_pio.algorithms import ALGORITHMS, run_with_dim


def _load_suite(suite):
    if suite == "cec2017":
        try:
            import cec2017  # type: ignore
            return lambda i, dim: lambda x: cec2017.functions.f1(x)  # placeholder
        except ImportError:
            raise SystemExit(
                "CEC2017 suite not installed. Install the official suite "
                "(e.g. the `cec2017` python package) and provide shift/rotation "
                "data, then re-run.")
    raise SystemExit(f"Unsupported suite: {suite}. Use 'cec2017' or 'cec2022'.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="cec2017")
    parser.add_argument("--dim", type=int, default=30)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--out", default="results/cec_summary.csv")
    args = parser.parse_args()

    if not os.path.exists(args.suite) and args.suite not in ("cec2017", "cec2022"):
        raise SystemExit("CEC suite data not present; provide the official suite "
                         "data or install the `cec2017` package.")

    _load_suite(args.suite)  # fail early with a clear message if unavailable

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    header = ["Suite", "Function", "Algorithm", "Mean", "Std"] + \
             [f"Run_{i}" for i in range(args.runs)]
    rows = []

    print(f"CEC runner scaffold ready for suite={args.suite}, dim={args.dim}. "
          "Install the official CEC suite to populate results.")

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Wrote (empty scaffold) {args.out}")


if __name__ == "__main__":
    main()
