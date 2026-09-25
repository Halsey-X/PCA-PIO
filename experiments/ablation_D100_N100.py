#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PCA-PIO D=100 Ablation Experiment with N=100 (larger population for stable covariance estimation)
4 variants x 2 functions x 10 runs first (to check timing and trend)
"""
import numpy as np
import time
import os
import csv

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
# PIO and PCA-PIO implementation (vectorized for speed)
# ============================================================
def run_pio(func, D, N=100, Tmax=500, T1=400, R=0.2, seed=None):
    rng = np.random.default_rng(seed)
    bound = 100 if func == sphere else 32
    X = rng.uniform(-bound, bound, (N, D))
    V = rng.uniform(-1, 1, (N, D))
    fitness = np.array([func(x) for x in X])
    best_idx = np.argmin(fitness)
    x_gbest = X[best_idx].copy()
    f_gbest = fitness[best_idx]

    for t in range(1, T1 + 1):
        V = V * np.exp(-R * t) + rng.uniform(0, 1, (N, D)) * (x_gbest - X)
        X = np.clip(X + V, -bound, bound)
        fitness = np.array([func(x) for x in X])
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < f_gbest:
            x_gbest = X[best_idx].copy()
            f_gbest = fitness[best_idx]

    for t in range(T1 + 1, Tmax + 1):
        sorted_idx = np.argsort(fitness)
        x_center = np.mean(X[sorted_idx[:N // 2]], axis=0)
        X = np.clip(X + rng.uniform(0, 1, (N, D)) * (x_center - X), -bound, bound)
        fitness = np.array([func(x) for x in X])
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < f_gbest:
            x_gbest = X[best_idx].copy()
            f_gbest = fitness[best_idx]
    return f_gbest


def run_pca_pio(func, D, N=100, Tmax=500, T1=400, R=0.2, CR=0.95, alpha=0.1,
                p_start=0.3, p_end=0.1, use_pca=True, use_restart=True, seed=None):
    rng = np.random.default_rng(seed)
    bound = 100 if func == sphere else 32

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

    def pca_resample(X, fitness, x_gbest, t):
        p_t = p_start - (p_start - p_end) * t / T1
        M = max(5, int(N * p_t))
        sorted_idx = np.argsort(fitness)
        elite = X[sorted_idx[:M]]
        cov = np.cov(elite, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0)
        eigenvectors = eigenvectors[:, order]
        total_var = np.sum(eigenvalues)
        if total_var > 1e-15:
            cum_var = np.cumsum(eigenvalues) / total_var
            m = min(np.searchsorted(cum_var, CR) + 1, D)
        else:
            m = 1
        X_new = np.zeros((N, D))
        for i in range(N):
            z = rng.standard_normal(m)
            candidate = x_gbest.copy()
            for k in range(m):
                candidate += alpha * np.sqrt(eigenvalues[k]) * eigenvectors[:, k] * z[k]
            X_new[i] = candidate
        return np.clip(X_new, -bound, bound)

    for t in range(1, T1 + 1):
        V_temp = V * np.exp(-R * t) + rng.uniform(0, 1, (N, D)) * (x_gbest - X)
        X_temp = np.clip(X + V_temp, -bound, bound)

        if use_pca:
            X_pca = pca_resample(X, fitness, x_gbest, t)
            fit_temp = np.array([func(x) for x in X_temp])
            fit_pca = np.array([func(x) for x in X_pca])
            use_pca_mask = fit_pca < fit_temp
            X = np.where(use_pca_mask[:, None], X_pca, X_temp)
            V = V_temp
            fitness = np.minimum(fit_temp, fit_pca)
        else:
            X = X_temp
            V = V_temp
            fitness = np.array([func(x) for x in X])

        best_idx = np.argmin(fitness)
        if fitness[best_idx] < f_gbest:
            x_gbest = X[best_idx].copy()
            f_gbest = fitness[best_idx]
            stagnation_pca = 0
            stagnation_restart = 0
        else:
            stagnation_pca += 1
            stagnation_restart += 1

        if use_pca and stagnation_pca >= Tpca:
            X = pca_resample(X, fitness, x_gbest, t)
            V = rng.uniform(-1, 1, (N, D)) * 0.1
            fitness = np.array([func(x) for x in X])
            stagnation_pca = 0

        if use_restart and stagnation_restart >= Trestart:
            X[0] = x_gbest.copy()
            X[1:] = rng.uniform(-bound, bound, (N - 1, D))
            V = rng.uniform(-1, 1, (N, D))
            fitness = np.array([func(x) for x in X])
            stagnation_pca = 0
            stagnation_restart = 0

    for t in range(T1 + 1, Tmax + 1):
        sorted_idx = np.argsort(fitness)
        x_center = np.mean(X[sorted_idx[:N // 2]], axis=0)
        X = np.clip(X + rng.uniform(0, 1, (N, D)) * (x_center - X), -bound, bound)
        fitness = np.array([func(x) for x in X])
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < f_gbest:
            x_gbest = X[best_idx].copy()
            f_gbest = fitness[best_idx]

    return f_gbest


# ============================================================
# Run experiments (N=100, D=100)
# ============================================================
D = 100
N_POP = 100
N_RUNS = 10  # Start with 10 to check timing
functions = {'Sphere': sphere, 'Ackley': ackley}
variants = {
    'Standard PIO': {'use_pca': False, 'use_restart': False},
    'PCA-PIO w/o Restart': {'use_pca': True, 'use_restart': False},
    'PCA-PIO w/o PCA': {'use_pca': False, 'use_restart': True},
    'PCA-PIO (Full)': {'use_pca': True, 'use_restart': True},
}

results = {}
print(f'=== D={D}, N={N_POP}, {N_RUNS} runs ===\n')

for func_name, func in functions.items():
    print(f'--- {func_name} ---')
    results[func_name] = {}
    for var_name, params in variants.items():
        t0 = time.time()
        runs = []
        for run in range(N_RUNS):
            seed = hash(f'N100_{func_name}_{var_name}_{run}') % (2**31)
            if var_name == 'Standard PIO':
                f = run_pio(func, D, N=N_POP, seed=seed)
            else:
                f = run_pca_pio(func, D, N=N_POP, seed=seed, **params)
            runs.append(f)
            if run == 0:
                print(f'  {var_name}: first run done in {time.time()-t0:.1f}s, val={f:.4e}')
        mean = np.mean(runs)
        std = np.std(runs)
        results[func_name][var_name] = (mean, std, runs)
        elapsed = time.time() - t0
        print(f'  {var_name:25s}: mean={mean:.4e}, std={std:.4e} ({elapsed:.1f}s total)')
    print()

# ============================================================
# Summary
# ============================================================
print('\n=== SUMMARY (D=100, N=100) ===')
print(f'{"Function":<12} {"Variant":<25} {"Mean":<15} {"Std":<15}')
print('-' * 70)
for func_name in functions:
    for var_name in variants:
        mean, std, _ = results[func_name][var_name]
        print(f'{func_name:<12} {var_name:<25} {mean:<15.4e} {std:<15.4e}')

# Save
os.makedirs('results', exist_ok=True)
with open(os.path.join('results', 'ablation_D100_N100_results.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Function', 'Variant', 'Mean', 'Std'] + [f'Run_{i}' for i in range(N_RUNS)])
    for func_name in functions:
        for var_name in variants:
            mean, std, runs = results[func_name][var_name]
            writer.writerow([func_name, var_name, f'{mean:.6e}', f'{std:.6e}'] + [f'{r:.6e}' for r in runs])

print('\nResults saved to ablation_D100_N100_results.csv')
