# PCA-PIO

**Principal Component Analysis Enhanced Pigeon-Inspired Optimization (PCA-PIO) for Function Optimization and Multi-Scenario UAV Path Planning**

Reproduction code and experimental data for the PCA-PIO algorithm (manuscript submitted to *Scientific Reports*).

PCA-PIO enhances the standard Pigeon-Inspired Optimization (PIO) by embedding a **PCA-based directional search** into the map–compass operator: principal components are extracted from the elite subset of the population to construct anisotropic search vectors, combined with an **adaptive stagnation-detection / restart** mechanism.

## Repository layout

```
PCA-PIO/
├── pca_pio/                      # clean, reusable implementation package
│   ├── benchmarks.py             # 12 classical benchmark functions (paper Table 1)
│   ├── algorithms.py             # PIO, PCA-PIO, PSO, GWO, CPIO (unified protocol)
│   └── uav.py                    # 3-scenario UAV 3D path planning model
├── experiments/
│   ├── ablation_D100.py          # Ablation on Sphere & Ackley, D=100, N=30, 30 runs
│   ├── ablation_D100_N100.py     # Ablation on Sphere & Ackley, D=100, N=100, 10 runs
│   ├── run_benchmarks.py         # 12 benchmarks, D=30, 5 algorithms, 30 runs
│   ├── run_highdim.py            # 5 functions, D=100
│   ├── run_uav.py                # 3 scenarios x 5 algorithms
│   └── run_cec.py                # CEC2017/CEC2022 scaffold (needs official suite)
├── results/
│   ├── ablation_D100_results.csv       # Archived output of ablation_D100.py
│   └── ablation_D100_N100_results.csv  # Archived output of ablation_D100_N100.py
├── README.md
├── requirements.txt
└── LICENSE
```

## Important disclaimer about reproducibility

**This repository contains a clean, runnable re-implementation of the algorithms
described in the manuscript — it does NOT reproduce the manuscript's reported
performance.**

- The original experiment code and data used for the manuscript's tables are
  **not available** here (the manuscript states its MATLAB implementation and
  scripts will be released publicly upon acceptance). What this repository
  provides is an independent, from-scratch reconstruction of the same algorithms
  from the algorithm descriptions in the paper.
- In fresh runs of the reconstructed implementation, the proposed PCA-PIO is
  **not** observed to outperform PSO/GWO on the classical benchmarks, i.e. it
  does **not** reproduce the paper's claim that PCA-PIO beats PSO on 10/12
  functions. **Do not cite the results of this repository as the paper's
  results.**
- Only the two CSVs under `results/` (the ablation experiment) are archived
  outputs from the original experiments; the other experiment runners generate
  their own numbers when run, which should be interpreted strictly as
  independent-reconstruction output.

The repository is intended as a *methodology / code scaffold* for the paper's
algorithms, and as archived data for the ablation study. For exact reproduction
of the manuscript's tables, the authors' original implementation is required.

## How to run

```bash
pip install -r requirements.txt

# archived ablation experiments
python experiments/ablation_D100.py
python experiments/ablation_D100_N100.py

# independent reconstruction runners (their results are NOT the paper's numbers)
python experiments/run_benchmarks.py    # 12 benchmarks, D=30
python experiments/run_highdim.py       # D=100
python experiments/run_uav.py           # UAV path planning
```

Each script writes a CSV into `results/`.

## Ablation variants

Each ablation compares four variants of the optimizer:

| Variant                | PCA directional search | Adaptive restart |
|------------------------|:----------------------:|:----------------:|
| Standard PIO           | no                     | no               |
| PCA-PIO w/o Restart    | yes                    | no               |
| PCA-PIO w/o PCA        | no                     | yes              |
| PCA-PIO (Full)         | yes                    | yes              |

Unified protocol (per the manuscript): `N=30` (or `N=100`), `Tmax=500`, `T1=400`,
30 independent runs (10 for the `N=100` variant).

## CEC suites

`experiments/run_cec.py` is a scaffold for the CEC2017/CEC2022 validation. The
official suites (shifted/rotated/hybrid/composition problems) require the
official CEC code and shift-rotation data, which cannot be redistributed here.
Install the official suite (e.g. the `cec2017` package) before running it.

## Dependencies

- Python ≥ 3.8
- NumPy

## License

MIT — see [LICENSE](LICENSE).
