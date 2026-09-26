# -*- coding: utf-8 -*-
"""
Multi-scenario UAV 3D path planning (manuscript Section "Multi-Scenario UAV 3D
Path Planning").

Plans a B-spline (Catmull-Rom) 3D path with 6 control points (dimension 18)
across the mountain / urban / maritime terrains for PSO, GWO, PIO, CPIO and
PCA-PIO. Reports average path length, average turning angle, collisions and
success rate over 30 independent runs (N=30, Tmax=200).

Usage:  python experiments/run_uav.py   (writes results/uav_pathplanning.csv)
"""
import csv
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pca_pio.algorithms import ALGORITHMS
from pca_pio.uav import SCENARIOS, run_uav_planning

N = 30
TMAX = 200
RUNS = 30


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "uav_pathplanning.csv")

    header = ["Scenario", "Algorithm", "MeanPathLength", "MeanTurningAngle",
              "Collisions", "SuccessRate", "MeanCost"]
    rows = []
    t0 = time.time()

    for si, scenario in enumerate(SCENARIOS):
        for alg in ALGORITHMS:
            lengths, angles, collisions, costs = [], [], 0, []
            for r in range(RUNS):
                seed = 300000 + si * 100000 + r
                metrics, best_cost = run_uav_planning(
                    scenario, alg, seed, N=N, Tmax=TMAX)
                lengths.append(metrics["path_length"])
                angles.append(metrics["turning_angle"])
                collisions += metrics["collisions"]
                costs.append(best_cost)
            mean_len = float(sum(lengths) / len(lengths))
            mean_ang = float(sum(angles) / len(angles))
            success = 1.0 - collisions / RUNS
            mean_cost = float(sum(costs) / len(costs))
            rows.append([scenario, alg, f"{mean_len:.4f}", f"{mean_ang:.4f}",
                         str(collisions), f"{success * 100:.1f}%", f"{mean_cost:.4f}"])
            print(f"  {scenario:<9} {alg:<9} len={mean_len:.1f} ang={mean_ang:.1f} "
                  f"coll={collisions} success={success * 100:.1f}%")

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\nWrote {out_path}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
