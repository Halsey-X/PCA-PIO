# -*- coding: utf-8 -*-
"""
Optimisers implemented for the PCA-PIO manuscript.

Five algorithms share a common interface so that every method runs under the
identical experimental protocol (same population size, iterations and the same
random seed within each paired run):

    best = run_with_dim(func, bounds, dim, seed, algorithm, N=30, Tmax=500, T1=400)

``func`` maps a (D,)-vector to a scalar fitness, ``bounds`` is the scalar
``(lo, hi)`` tuple (symmetric search range, as used in the paper) and ``dim``
the problem dimension.

The PCA-PIO implementation follows the manuscript: PCA-based anisotropic
directional search in the map--compass stage, adaptive stagnation detection
(``T_pca``), PCA resampling and adaptive global restart (``T_restart``), with a
ridge-regularised elite covariance matrix and reflect boundary handling.
Baseline methods use their standard recommended update rules and clip boundary
handling.
"""
from __future__ import annotations

import numpy as np

TOL = 1e-8  # stagnation tolerance: a generation counts as "no improvement"
             # when the best-so-far fitness does not decrease by more than TOL


def _reflect(x, lo, hi):
    """Reflect out-of-bounds coordinates back into [lo, hi] (used by PCA-PIO)."""
    x = np.asarray(x, dtype=float)
    span = hi - lo
    y = (x - lo) % (2.0 * span)
    y = np.where(y > span, 2.0 * span - y, y)
    return y + lo


def _init_pop(rng, N, D, lo, hi):
    return rng.uniform(lo, hi, (N, D))


def _eval(func, X):
    return np.array([float(func(x)) for x in X])


# ---------------------------------------------------------------------------
# Standard PIO (map-compass + landmark)
# ---------------------------------------------------------------------------
def _run_pio(func, X, lo, hi, N, Tmax, T1, R, rng):
    D = X.shape[1]
    V = rng.uniform(-1.0, 1.0, (N, D))
    fitness = _eval(func, X)
    gbest = X[np.argmin(fitness)].copy()
    f_gbest = float(np.min(fitness))

    # map-compass stage
    for t in range(1, T1 + 1):
        V = V * np.exp(-R * t) + rng.uniform(0.0, 1.0, (N, D)) * (gbest - X)
        X = np.clip(X + V, lo, hi)
        fitness = _eval(func, X)
        if float(np.min(fitness)) < f_gbest:
            f_gbest = float(np.min(fitness))
            gbest = X[np.argmin(fitness)].copy()

    # landmark stage
    for t in range(T1 + 1, Tmax + 1):
        order = np.argsort(fitness)
        center = np.mean(X[order[:N // 2]], axis=0)
        X = np.clip(X + rng.uniform(0.0, 1.0, (N, D)) * (center - X), lo, hi)
        fitness = _eval(func, X)
        if float(np.min(fitness)) < f_gbest:
            f_gbest = float(np.min(fitness))
            gbest = X[np.argmin(fitness)].copy()

    return f_gbest, gbest


def _run_cpio(func, X, lo, hi, N, Tmax, T1, R, rng, pc=0.2, scale=0.5):
    D = X.shape[1]
    V = rng.uniform(-1.0, 1.0, (N, D))
    fitness = _eval(func, X)
    gbest = X[np.argmin(fitness)].copy()
    f_gbest = float(np.min(fitness))

    for t in range(1, Tmax + 1):
        if t <= T1:
            V = V * np.exp(-R * t) + rng.uniform(0.0, 1.0, (N, D)) * (gbest - X)
            X = np.clip(X + V, lo, hi)
        else:
            order = np.argsort(fitness)
            center = np.mean(X[order[:N // 2]], axis=0)
            X = np.clip(X + rng.uniform(0.0, 1.0, (N, D)) * (center - X), lo, hi)

        # Cauchy mutation on a random subset of individuals (escape local optima)
        mask = rng.random(N) < pc
        if mask.any():
            step = scale * (hi - lo) * (0.99 ** t)
            X[mask] = np.clip(X[mask] + rng.standard_cauchy((int(mask.sum()), D)) * step, lo, hi)

        fitness = _eval(func, X)
        if float(np.min(fitness)) < f_gbest:
            f_gbest = float(np.min(fitness))
            gbest = X[np.argmin(fitness)].copy()

    return f_gbest, gbest


def _run_pca_pio(func, X, lo, hi, N, Tmax, T1, R, rng,
                 CR=0.95, alpha=0.1, p_start=0.3, p_end=0.1, ridge=1e-8,
                 use_pca=True, use_restart=True):
    D = X.shape[1]
    V = rng.uniform(-1.0, 1.0, (N, D))
    fitness = _eval(func, X)
    gbest = X[np.argmin(fitness)].copy()
    f_gbest = float(np.min(fitness))

    Tpca = max(5, round(0.02 * Tmax))
    Trestart = max(10, round(0.04 * Tmax))
    stagnation_pca = 0
    stagnation_restart = 0

    def pca_resample(cur_X, t):
        p_t = p_start - (p_start - p_end) * t / T1
        M = max(3, int(N * p_t))
        order = np.argsort(fitness)
        elite = cur_X[order[:M]]
        cov = np.cov(elite, rowvar=False) + ridge * np.eye(D)
        w, v = np.linalg.eigh(cov)
        idx = np.argsort(w)[::-1]
        w = np.maximum(w[idx], 0.0)
        v = v[:, idx]
        total = np.sum(w)
        if total > 1e-15:
            cum = np.cumsum(w) / total
            m = min(np.searchsorted(cum, CR) + 1, D)
        else:
            m = 1
        new = np.zeros_like(cur_X)
        for i in range(N):
            z = rng.standard_normal(m)
            cand = gbest.copy()
            for k in range(m):
                cand = cand + alpha * np.sqrt(w[k]) * v[:, k] * z[k]
            new[i] = cand
        return _reflect(new, lo, hi)

    for t in range(1, T1 + 1):
        V_temp = V * np.exp(-R * t) + rng.uniform(0.0, 1.0, (N, D)) * (gbest - X)
        X_temp = np.clip(X + V_temp, lo, hi)

        if use_pca:
            X_pca = pca_resample(X, t)
            fit_temp = _eval(func, X_temp)
            fit_pca = _eval(func, X_pca)
            better = fit_pca < fit_temp
            X = np.where(better[:, None], X_pca, X_temp)
            fitness = np.minimum(fit_temp, fit_pca)
        else:
            X = X_temp
            fitness = _eval(func, X)

        if float(np.min(fitness)) < f_gbest - TOL:
            f_gbest = float(np.min(fitness))
            gbest = X[np.argmin(fitness)].copy()
            stagnation_pca = 0
            stagnation_restart = 0
        else:
            stagnation_pca += 1
            stagnation_restart += 1

        if use_pca and stagnation_pca >= Tpca:
            X = pca_resample(X, t)
            V = rng.uniform(-1.0, 1.0, (N, D)) * 0.1
            fitness = _eval(func, X)
            stagnation_pca = 0

        if use_restart and stagnation_restart >= Trestart:
            X[0] = gbest.copy()
            X[1:] = rng.uniform(lo, hi, (N - 1, D))
            V = rng.uniform(-1.0, 1.0, (N, D))
            fitness = _eval(func, X)
            stagnation_pca = 0
            stagnation_restart = 0

    # landmark stage
    for t in range(T1 + 1, Tmax + 1):
        order = np.argsort(fitness)
        center = np.mean(X[order[:N // 2]], axis=0)
        X = np.clip(X + rng.uniform(0.0, 1.0, (N, D)) * (center - X), lo, hi)
        fitness = _eval(func, X)
        if float(np.min(fitness)) < f_gbest - TOL:
            f_gbest = float(np.min(fitness))
            gbest = X[np.argmin(fitness)].copy()

    return f_gbest, gbest


def _run_pso(func, X, lo, hi, N, Tmax, rng, c1=2.0, c2=2.0, w_start=0.9, w_end=0.4):
    D = X.shape[1]
    V = rng.uniform(-(hi - lo) * 0.1, (hi - lo) * 0.1, (N, D))
    fitness = _eval(func, X)
    pbest = X.copy()
    pbest_fit = fitness.copy()
    gbest = X[np.argmin(fitness)].copy()
    f_gbest = float(np.min(fitness))
    vmax = 0.2 * (hi - lo)

    for t in range(1, Tmax + 1):
        w = w_start - (w_start - w_end) * t / Tmax
        r1 = rng.random((N, D))
        r2 = rng.random((N, D))
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        V = np.clip(V, -vmax, vmax)
        X = np.clip(X + V, lo, hi)
        fitness = _eval(func, X)
        improved = fitness < pbest_fit
        pbest[improved] = X[improved]
        pbest_fit[improved] = fitness[improved]
        if float(np.min(fitness)) < f_gbest:
            f_gbest = float(np.min(fitness))
            gbest = X[np.argmin(fitness)].copy()

    return f_gbest, gbest


def _run_gwo(func, X, lo, hi, N, Tmax, rng):
    fitness = _eval(func, X)
    order = np.argsort(fitness)
    alpha, beta, delta = X[order[0]].copy(), X[order[1]].copy(), X[order[2]].copy()
    f_gbest = float(fitness[order[0]])

    for t in range(1, Tmax + 1):
        a = 2.0 - 2.0 * t / Tmax
        for i in range(N):
            xi = X[i]
            A1 = 2.0 * a * rng.random() - a
            C1 = 2.0 * rng.random()
            D_a = np.abs(C1 * alpha - xi)
            X1 = alpha - A1 * D_a
            A2 = 2.0 * a * rng.random() - a
            C2 = 2.0 * rng.random()
            D_b = np.abs(C2 * beta - xi)
            X2 = beta - A2 * D_b
            A3 = 2.0 * a * rng.random() - a
            C3 = 2.0 * rng.random()
            D_d = np.abs(C3 * delta - xi)
            X3 = delta - A3 * D_d
            X[i] = np.clip((X1 + X2 + X3) / 3.0, lo, hi)
        fitness = _eval(func, X)
        order = np.argsort(fitness)
        if float(fitness[order[0]]) < f_gbest:
            alpha, beta, delta = X[order[0]].copy(), X[order[1]].copy(), X[order[2]].copy()
            f_gbest = float(fitness[order[0]])

    return f_gbest, alpha


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------
def run_with_dim(func, bounds, dim, seed, algorithm, N=30, Tmax=500, T1=400,
                 R=0.2, return_solution=False, **kw):
    """Run one algorithm on a D-dimensional problem with a deterministic seed.

    ``bounds`` may be a scalar ``(lo, hi)`` tuple (all coordinates share the
    same range, as on the classical benchmarks) or an array-valued
    ``(lo, hi)`` pair (per-coordinate ranges, as in UAV planning). The same
    seed is used for every algorithm within a paired run to guarantee a fair
    comparison. When ``return_solution`` is True, returns ``(best_fitness,
    best_solution_vector)`` instead of just the best fitness.
    """
    if np.ndim(bounds[0]) == 0:
        lo = np.full(dim, float(bounds[0]))
        hi = np.full(dim, float(bounds[1]))
    else:
        lo = np.asarray(bounds[0], dtype=float)
        hi = np.asarray(bounds[1], dtype=float)
    rng = np.random.default_rng(seed)
    X = _init_pop(rng, N, dim, lo, hi)

    if algorithm == "PSO":
        f, sol = _run_pso(func, X, lo, hi, N, Tmax, rng, **kw)
    elif algorithm == "GWO":
        f, sol = _run_gwo(func, X, lo, hi, N, Tmax, rng)
    elif algorithm == "PIO":
        f, sol = _run_pio(func, X, lo, hi, N, Tmax, T1, R, rng)
    elif algorithm == "CPIO":
        f, sol = _run_cpio(func, X, lo, hi, N, Tmax, T1, R, rng, **kw)
    elif algorithm == "PCA-PIO":
        f, sol = _run_pca_pio(func, X, lo, hi, N, Tmax, T1, R, rng, **kw)
    else:
        raise KeyError(f"Unknown algorithm: {algorithm}")

    return (f, sol) if return_solution else f


ALGORITHMS = ["PSO", "GWO", "PIO", "CPIO", "PCA-PIO"]
