import numpy as np


def make_circles(n=200, noise=0.10, factor=0.45, seed=42):
    rng = np.random.default_rng(seed)
    n1, n2 = n // 2, n - n // 2
    t1 = rng.uniform(0, 2 * np.pi, n1)
    t2 = rng.uniform(0, 2 * np.pi, n2)
    inner = np.c_[factor * np.cos(t1), factor * np.sin(t1)]
    outer = np.c_[np.cos(t2), np.sin(t2)]
    x = np.vstack([inner, outer])
    y = np.hstack([np.zeros(n1, dtype=int), np.ones(n2, dtype=int)])
    x += rng.normal(0, noise, x.shape)
    idx = rng.permutation(n)
    return x[idx], y[idx]


def make_xor(n=200, noise=0.15, seed=42):
    rng = np.random.default_rng(seed)
    n4 = n // 4
    centers = [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)]
    parts = [np.array(c) + rng.normal(0, noise, (n4, 2)) for c in centers]
    x = np.vstack(parts)
    y = ((x[:, 0] * x[:, 1]) < 0).astype(int)
    idx = rng.permutation(len(x))
    return x[idx], y[idx]


def make_blobs(n=200, noise=0.35, seed=42):
    rng = np.random.default_rng(seed)
    n1, n2 = n // 2, n - n // 2
    c0 = rng.normal(0, noise, (n1, 2)) + np.array([-1.2, -1.2])
    c1 = rng.normal(0, noise, (n2, 2)) + np.array([1.2,  1.2])
    x = np.vstack([c0, c1])
    y = np.hstack([np.zeros(n1, dtype=int), np.ones(n2, dtype=int)])
    idx = rng.permutation(n)
    return x[idx], y[idx]


def make_spiral(n=200, noise=0.08, turns=2.5, seed=42):
    rng = np.random.default_rng(seed)
    n1, n2 = n // 2, n - n // 2
    t1 = np.linspace(0, turns * np.pi, n1)
    t2 = np.linspace(0, turns * np.pi, n2)
    r1 = t1 / (turns * np.pi)
    r2 = t2 / (turns * np.pi)
    s1 = np.c_[r1 * np.cos(t1), r1 * np.sin(t1)]
    s2 = np.c_[r2 * np.cos(t2 + np.pi), r2 * np.sin(t2 + np.pi)]
    x = np.vstack([s1, s2])
    x += rng.normal(0, noise, x.shape)
    y = np.hstack([np.zeros(n1, dtype=int), np.ones(n2, dtype=int)])
    idx = rng.permutation(n)
    return x[idx], y[idx]


def split_data(x, y, test_ratio=0.3, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    cut = int(len(x) * (1 - test_ratio))
    tr, te = idx[:cut], idx[cut:]
    return x[tr], y[tr], x[te], y[te]


DATASETS = [
    ('Circles', make_circles),
    ('XOR',     make_xor),
    ('Blobs',   make_blobs),
    ('Spiral',  make_spiral),
]
