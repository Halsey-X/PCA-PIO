# -*- coding: utf-8 -*-
"""
Classical benchmark functions used in the PCA-PIO manuscript (Table 1).

The 12 functions cover unimodal, ill-conditioned and multimodal landscapes.
Each function is implemented in a vectorised (NumPy) form so that it can be
evaluated on a single D-dimensional vector or on a whole population at once.

Reproduction note: these implementations are a faithful, self-contained
re-implementation of the standard definitions used in the manuscript. Search
ranges follow the paper's Table 1.
"""
from __future__ import annotations

import numpy as np


def sphere(x):
    return np.sum(x ** 2)


def schwefel_2_22(x):
    return np.sum(np.abs(x)) + np.prod(np.abs(x))


def schwefel_1_2(x):
    # sum_i (sum_{j<=i} x_j)^2
    d = len(x)
    tri = np.cumsum(x)
    return np.sum(tri ** 2)


def schwefel_2_21(x):
    return np.max(np.abs(x))


def rosenbrock(x):
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1.0) ** 2)


def step(x):
    return np.sum(np.floor(x + 0.5) ** 2)


def quartic(x, noise_rng=None):
    # noisy quartic: sum_i i * x_i^4 + U[0,1)
    d = len(x)
    noise = 0.0
    if noise_rng is not None:
        noise = float(noise_rng.uniform(0.0, 1.0))
    return np.sum(np.arange(1, d + 1) * x ** 4) + noise


def schwefel(x):
    d = len(x)
    return 418.98288727243369 * d - np.sum(x * np.sin(np.sqrt(np.abs(x))))


def rastrigin(x):
    d = len(x)
    return 10.0 * d + np.sum(x ** 2 - 10.0 * np.cos(2.0 * np.pi * x))


def ackley(x):
    d = len(x)
    sum1 = np.sum(x ** 2)
    sum2 = np.sum(np.cos(2.0 * np.pi * x))
    return -20.0 * np.exp(-0.2 * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + 20.0 + np.e


def griewank(x):
    d = len(x)
    idx = np.arange(1, d + 1)
    return np.sum(x ** 2) / 4000.0 - np.prod(np.cos(x / np.sqrt(idx))) + 1.0


def alpine(x):
    return np.sum(np.abs(x * np.sin(x) + 0.1 * x))


# ---------------------------------------------------------------------------
# Function table (mirrors Table 1 of the manuscript)
#   name, function, type, search range (lo, hi), theoretical minimum (approx.)
# ---------------------------------------------------------------------------
FUNCTIONS = [
    dict(id="F1", name="Sphere",         fn=sphere,        typ="Unimodal",        lo=-100.0, hi=100.0,  fmin=0.0),
    dict(id="F2", name="Schwefel 2.22",  fn=schwefel_2_22, typ="Unimodal",        lo=-10.0,  hi=10.0,   fmin=0.0),
    dict(id="F3", name="Schwefel 1.2",   fn=schwefel_1_2,  typ="Unimodal",        lo=-100.0, hi=100.0,  fmin=0.0),
    dict(id="F4", name="Schwefel 2.21",  fn=schwefel_2_21, typ="Unimodal",        lo=-100.0, hi=100.0,  fmin=0.0),
    dict(id="F5", name="Rosenbrock",     fn=rosenbrock,    typ="Ill-conditioned", lo=-30.0,  hi=30.0,   fmin=0.0),
    dict(id="F6", name="Step",           fn=step,          typ="Unimodal",        lo=-100.0, hi=100.0,  fmin=0.0),
    dict(id="F7", name="Quartic",        fn=quartic,       typ="Unimodal",        lo=-1.28,  hi=1.28,   fmin=0.0),
    dict(id="F8", name="Schwefel",       fn=schwefel,      typ="Multimodal",      lo=-500.0, hi=500.0,  fmin=-418.98288727243369),
    dict(id="F9", name="Rastrigin",      fn=rastrigin,     typ="Multimodal",      lo=-5.12,  hi=5.12,   fmin=0.0),
    dict(id="F10", name="Ackley",        fn=ackley,        typ="Multimodal",      lo=-32.0,  hi=32.0,   fmin=0.0),
    dict(id="F11", name="Griewank",      fn=griewank,      typ="Multimodal",      lo=-600.0, hi=600.0,  fmin=0.0),
    dict(id="F12", name="Alpine",        fn=alpine,        typ="Multimodal",      lo=-10.0,  hi=10.0,   fmin=0.0),
]

# The five functions further evaluated at high dimension D=100 (manuscript).
HIGH_DIM_NAMES = ["Sphere", "Rosenbrock", "Rastrigin", "Ackley", "Griewank"]


def get_function(name):
    """Return the function-table dict for a benchmark by its display name."""
    for row in FUNCTIONS:
        if row["name"] == name:
            return row
    raise KeyError(f"Unknown benchmark function: {name}")
