# PCA-PIO

**Principal Component Analysis Enhanced Pigeon-Inspired Optimization (PCA-PIO) for Function Optimization and Multi-Scenario UAV Path Planning**

Reproduction code and experimental data for the ablation study of the proposed PCA-PIO algorithm (manuscript submitted to *Scientific Reports*).

PCA-PIO enhances the standard Pigeon-Inspired Optimization (PIO) by embedding a **PCA-based directional search** into the map–compass operator: principal components are extracted from the elite subset of the population to construct anisotropic search vectors, combined with an **adaptive stagnation-detection / restart** mechanism. This improves convergence accuracy and robustness on ill-conditioned and multimodal problems.

## Repository layout

```
PCA-PIO/
├── experiments/
│   ├── ablation_D100.py        # Ablation on Sphere & Ackley, D=100, N=30, 30 runs
│   └── ablation_D100_N100.py   # Ablation on Sphere & Ackley, D=100, N=100, 10 runs
├── results/
│   ├── ablation_D100_results.csv       # Archived output of ablation_D100.py
│   └── ablation_D100_N100_results.csv  # Archived output of ablation_D100_N100.py
├── README.md
├── requirements.txt
└── LICENSE
```

## Ablation variants

Each experiment compares four variants of the optimizer:

| Variant                | PCA directional search | Adaptive restart |
|------------------------|:----------------------:|:----------------:|
| Standard PIO           | no                     | no               |
| PCA-PIO w/o Restart    | yes                    | no               |
| PCA-PIO w/o PCA        | no                     | yes              |
| PCA-PIO (Full)         | yes                    | yes              |

**Unified protocol (per the manuscript):** population size `N=30` (or `N=100`), maximum iterations `Tmax=500`, map–compass stage iterations `T1=400`, and independent runs per function (30 runs for `N=30`; 10 runs for the `N=100` variant). Benchmark functions: Sphere (unimodal) and Ackley (multimodal).

## How to reproduce

```bash
pip install -r requirements.txt

# D=100, N=30, 30 independent runs (takes a while)
python experiments/ablation_D100.py

# D=100, N=100, 10 independent runs
python experiments/ablation_D100_N100.py
```

Each script writes a CSV into `results/` (mean, std, and per-run fitness values).

## Notes on reproducibility

- The two scripts implement the PIO and PCA-PIO algorithms from scratch (NumPy only, no third-party optimizer). The implementations mirror the algorithm descriptions in the manuscript.
- Seeding in the original scripts uses Python's built-in `hash()` on a per-run label. Python randomizes string hashing per process (`PYTHONHASHSEED`), so per-run values can vary slightly between runs. For bit-identical reproduction, pin the hash seed, e.g.:
  ```bash
  PYTHONHASHSEED=0 python experiments/ablation_D100.py
  ```
- The CSVs under `results/` are the archived outputs from the paper's experiments; re-running a script regenerates a statistically comparable, not necessarily bit-identical, table.
- This repository currently contains the **ablation subset** of the manuscript's experiments. The broader studies reported in the paper (12 classical benchmarks, CEC2017/CEC2022 suites, PSO/GWO/CPIO comparisons, statistical tests, and UAV 3D path planning) are described in the manuscript but are not backed by code in this repository yet.

## Dependencies

- Python ≥ 3.8
- NumPy

## License

MIT — see [LICENSE](LICENSE).
