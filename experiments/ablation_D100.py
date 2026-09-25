#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PCA-PIO D=100 Ablation Experiment
Reproduce the algorithm from the paper and run ablation on Sphere and Ackley at D=100.
4 variants: Standard PIO, PCA-PIO w/o Restart, PCA-PIO w/o PCA, PCA-PIO (Full)
Each: N=30, Tmax=500, T1=400, 30 independent runs.
"""
import numpy as np
import time
import os

# ============================================================
# Benchmark functions
# ============================================================
def sphere(x):
    return np.sum(x ** 2)

def ackley(x):
    d = len(x)
    sum1 = np.sum(x ** 2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + 20 + np.e

# ============================================================
# PIO and PCA-PIO implementation
# ============================================================
def run_pio(func, D, N=30, Tmax=500, T1=400, R=0.2, seed=None):
    """Standard PIO"""
    rng = np.random.default_rng(seed)
    # Initialize
    X = rng.uniform(-100, 100, (N, D)) if func == sphere else rng.uniform(-32, 32, (N, D))
    V = rng.uniform(-1, 1, (N, D))
    fitness = np.array([func(x) for x in X])
    best_idx = np.argmin(fitness)
    x_gbest = X[best_idx].copy()
    f_gbest = fitness[best_idx]

    for t in range(1, T1 + 1):
        # Map-compass operator
        V = V * np.exp(-R * t) + rng.uniform(0, 1, (N, D)) * (x_gbest - X)
        X = X + V
        # Boundary handling
        if func == sphere:
            X = np.clip(X, -100, 100)
        else:
            X = np.clip(X, -32, 32)
        fitness = np.array([func(x) for x in X])
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < f_gbest:
            x_gbest = X[best_idx].copy()
            f_gbest = fitness[best_idx]

    # Landmark operator
    for t in range(T1 + 1, Tmax + 1):
        sorted_idx = np.argsort(fitness)
        top_half = sorted_idx[:N // 2]
        x_center = np.mean(X[top_half], axis=0)
        X = X + rng.uniform(0, 1, (N, D)) * (x_center - X)
        if func == sphere:
            X = np.clip(X, -100, 100)
        else:
            X = np.clip(X, -32, 32)
        fitness = np.array([func(x) for x in X])
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < f_gbest:
            x_gbest = X[best_idx].copy()
            f_gbest = fitness[best_idx]

    return f_gbest


def run_pca_pio(func, D, N=30, Tmax=500, T1=400, R=0.2, CR=0.95, alpha=0.1,
                p_start=0.3, p_end=0.1, use_pca=True, use_restart=True, seed=None):
    """PCA-PIO with configurable components"""
    rng = np.random.default_rng(seed)
    bound = 100 if func == sphere else 32

    # Initialize
    X = rng.uniform(-bound, bound, (N, D))
    V = rng.uniform(-1, 1, (N, D))
    fitness = np.array([func(x) for x in X])
    best_idx = np.argmin(fitness)
    x_gbest = X[best_idx].copy()
    f_gbest = fitness[best_idx]

    Tpca = max(5, round(0.02 * Tmax))
    Trestart = max(10, round(0.04 * Tmax))
    stagnation_pca = 0
    stagnation_restart = 0

    for t in range(1, T1 + 1):
        # Standard PIO map-compass update (temporary population)
        V_temp = V * np.exp(-R * t) + rng.uniform(0, 1, (N, D)) * (x_gbest - X)
        X_temp = X + V_temp
        X_temp = np.clip(X_temp, -bound, bound)

        if use_pca:
            # Elite subset
            p_t = p_start - (p_start - p_end) * t / T1
            M = max(3, int(N * p_t))
            sorted_idx = np.argsort(fitness)
            elite = X[sorted_idx[:M]]
            mu = np.mean(elite, axis=0)
            cov = np.cov(elite, rowvar=False)
            # Eigendecomposition
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            # Sort descending
            order = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]
            # Retain components with cumulative contribution > CR
            eigenvalues = np.maximum(eigenvalues, 0)
            total_var = np.sum(eigenvalues)
            if total_var > 0:
                cum_var = np.cumsum(eigenvalues) / total_var
                m = np.searchsorted(cum_var, CR) + 1
                m = min(m, D)
            else:
                m = 1

            # Generate PCA candidates
            X_pca = np.zeros((N, D))
            for i in range(N):
                z = rng.standard_normal(m)
                candidate = x_gbest.copy()
                for k in range(m):
                    candidate += alpha * np.sqrt(eigenvalues[k]) * eigenvectors[:, k] * z[k]
                X_pca[i] = candidate
            X_pca = np.clip(X_pca, -bound, bound)

            # Compare and retain better
            fit_temp = np.array([func(x) for x in X_temp])
            fit_pca = np.array([func(x) for x in X_pca])
            use_pca_mask = fit_pca < fit_temp
            X_new = np.where(use_pca_mask[:, None], X_pca, X_temp)
            V_new = np.where(use_pca_mask[:, None], V_temp, V_temp)  # velocity for PCA candidates is ambiguous
            X = X_new
            V = V_new
            fitness = np.minimum(fit_temp, fit_pca)
        else:
            X = X_temp
            V = V_temp
            fitness = np.array([func(x) for x in X])

        # Update global best
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < f_gbest:
            x_gbest = X[best_idx].copy()
            f_gbest = fitness[best_idx]
            stagnation_pca = 0
            stagnation_restart = 0
        else:
            stagnation_pca += 1
            stagnation_restart += 1

        # PCA resampling on stagnation
        if use_pca and stagnation_pca >= Tpca:
            p_t = p_start - (p_start - p_end) * t / T1
            M = max(3, int(N * p_t))
            sorted_idx = np.argsort(fitness)
            elite = X[sorted_idx[:M]]
            mu = np.mean(elite, axis=0)
            cov = np.cov(elite, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            order = np.argsort(eigenvalues)[::-1]
            eigenvalues = np.maximum(eigenvalues[order], 0)
            eigenvectors = eigenvectors[:, order]
            total_var = np.sum(eigenvalues)
            if total_var > 0:
                cum_var = np.cumsum(eigenvalues) / total_var
                m = np.searchsorted(cum_var, CR) + 1
                m = min(m, D)
            else:
                m = 1
            for i in range(N):
                z = rng.standard_normal(m)
                candidate = x_gbest.copy()
                for k in range(m):
                    candidate += alpha * np.sqrt(eigenvalues[k]) * eigenvectors[:, k] * z[k]
                X[i] = candidate
            X = np.clip(X, -bound, bound)
            V = rng.uniform(-1, 1, (N, D)) * 0.1
            fitness = np.array([func(x) for x in X])
            stagnation_pca = 0

        # Global restart on stagnation
        if use_restart and stagnation_restart >= Trestart:
            X[0] = x_gbest.copy()
            X[1:] = rng.uniform(-bound, bound, (N - 1, D))
            V = rng.uniform(-1, 1, (N, D))
            fitness = np.array([func(x) for x in X])
            stagnation_pca = 0
            stagnation_restart = 0

    # Landmark operator
    for t in range(T1 + 1, Tmax + 1):
        sorted_idx = np.argsort(fitness)
        top_half = sorted_idx[:N // 2]
        x_center = np.mean(X[top_half], axis=0)
        X = X + rng.uniform(0, 1, (N, D)) * (x_center - X)
        X = np.clip(X, -bound, bound)
        fitness = np.array([func(x) for x in X])
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < f_gbest:
            x_gbest = X[best_idx].copy()
            f_gbest = fitness[best_idx]

    return f_gbest


# ============================================================
# Run experiments
# ============================================================
D = 100
N_RUNS = 30
functions = {'Sphere': sphere, 'Ackley': ackley}
variants = {
    'Standard PIO': {'use_pca': False, 'use_restart': False},
    'PCA-PIO w/o Restart': {'use_pca': True, 'use_restart': False},
    'PCA-PIO w/o PCA': {'use_pca': False, 'use_restart': True},
    'PCA-PIO (Full)': {'use_pca': True, 'use_restart': True},
}

results = {}

for func_name, func in functions.items():
    print(f'\n=== {func_name} (D={D}) ===')
    results[func_name] = {}
    for var_name, params in variants.items():
        t0 = time.time()
        runs = []
        for run in range(N_RUNS):
            seed = hash(f'{func_name}_{var_name}_{run}') % (2**31)
            if var_name == 'Standard PIO':
                f = run_pio(func, D, seed=seed)
            else:
                f = run_pca_pio(func, D, seed=seed, **params)
            runs.append(f)
        mean = np.mean(runs)
        std = np.std(runs)
        results[func_name][var_name] = (mean, std, runs)
        elapsed = time.time() - t0
        print(f'  {var_name:25s}: mean={mean:.4e}, std={std:.4e} ({elapsed:.1f}s)')

# ============================================================
# Save results
# ============================================================
print('\n\n=== SUMMARY (D=100) ===')
print(f'{"Function":<12} {"Variant":<25} {"Mean":<15} {"Std":<15}')
print('-' * 70)
for func_name in functions:
    for var_name in variants:
        mean, std, _ = results[func_name][var_name]
        print(f'{func_name:<12} {var_name:<25} {mean:<15.4e} {std:<15.4e}')

# Save to CSV
import csv
os.makedirs('results', exist_ok=True)
with open(os.path.join('results', 'ablation_D100_results.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Function', 'Variant', 'Mean', 'Std'] + [f'Run_{i}' for i in range(N_RUNS)])
    for func_name in functions:
        for var_name in variants:
            mean, std, runs = results[func_name][var_name]
            writer.writerow([func_name, var_name, f'{mean:.6e}', f'{std:.6e}'] + [f'{r:.6e}' for r in runs])

print('\nResults saved to ablation_D100_results.csv')
