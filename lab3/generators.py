import numpy as np
from config import SEED, AXIS_LIMIT

def random_polynomial(seed=SEED):
    rng = np.random.default_rng(seed)
    a, b, c, d = rng.uniform(-3, 3, 4)

    def f(x):
        return a * x**3 + b * x**2 + c * x + d

    return f, (a, b, c, d)

def trig_function():
    def f(x):
        return x * np.sin(2 * np.pi * x)
    return f

def generate_noise(n, eps0, mode='uniform', sigma=None, seed=SEED):
    rng = np.random.default_rng(seed)
    if sigma is None:
        sigma = eps0 / 3
    if mode == 'uniform':
        return rng.uniform(-eps0, eps0, n)
    if mode == 'normal':
        return rng.normal(0, sigma, n)
    raise ValueError(f'Unknown noise mode: {mode}')

def generate_regression_sample(f, n, eps0, noise_mode='uniform', sigma=None, seed=SEED):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    noise = generate_noise(n, eps0, mode=noise_mode, sigma=sigma, seed=seed + 1)
    y = f(x) + noise
    return x.reshape(-1, 1), y.reshape(-1, 1)

def make_circles(n=400, noise=0.08, factor=0.35, seed=SEED):
    rng = np.random.default_rng(seed)
    n_outer = n // 2
    n_inner = n - n_outer

    theta_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    theta_inner = np.linspace(0, 2 * np.pi, n_inner, endpoint=False)

    outer = np.column_stack([np.cos(theta_outer), np.sin(theta_outer)])
    inner = factor * np.column_stack([np.cos(theta_inner), np.sin(theta_inner)])

    X = np.vstack([outer, inner])
    y = np.concatenate([np.zeros(n_outer, dtype=int), np.ones(n_inner, dtype=int)])
    if noise > 0:
        X += rng.normal(0, noise, X.shape)
    return X, y


def make_xor(n=400, noise=0.55, scale=2.2, seed=SEED):
    rng = np.random.default_rng(seed)
    centers = np.array([[scale, scale], [scale, -scale], [-scale, scale], [-scale, -scale]])
    per = n // 4
    rem = n % 4
    X_parts, y_parts = [], []
    for i, c in enumerate(centers):
        cnt = per + (1 if i < rem else 0)
        X_parts.append(c + noise * rng.standard_normal((cnt, 2)))
        y_parts.append(np.full(cnt, 0 if i in (0, 3) else 1, dtype=int))
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def make_blobs(n=400, centers=None, cluster_std=0.9, seed=SEED):
    rng = np.random.default_rng(seed)
    if centers is None:
        centers = np.array([[-2.5, -2.0], [2.3, 2.0]], dtype=float)
    else:
        centers = np.asarray(centers, dtype=float)
    n_cls = len(centers)
    base = n // n_cls
    rem = n % n_cls
    X_parts, y_parts = [], []
    for k, c in enumerate(centers):
        cnt = base + (1 if k < rem else 0)
        X_parts.append(c + cluster_std * rng.standard_normal((cnt, 2)))
        y_parts.append(np.full(cnt, k, dtype=int))
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]

def make_spiral(n=400, turns=1.5, radius=0.08, sweep=0.28, noise=0.08, seed=SEED):
    rng = np.random.default_rng(seed)
    n0 = n // 2
    n1 = n - n0
    theta0 = np.linspace(0, 2 * np.pi * turns, n0)
    theta1 = np.linspace(0, 2 * np.pi * turns, n1)
    r0 = radius + sweep * theta0
    r1 = radius + sweep * theta1
    X0 = np.column_stack([r0 * np.cos(theta0), r0 * np.sin(theta0)])
    X1 = np.column_stack([r1 * np.cos(theta1 + np.pi),
                          r1 * np.sin(theta1 + np.pi)])
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])
    if noise > 0:
        X += rng.normal(0, noise, X.shape)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]

CLS_DATASETS = [
    ('Circles', make_circles),
    ('XOR',     make_xor),
    ('Blobs',   make_blobs),
    ('Spiral',  make_spiral),
]

def train_test_split(X, y, test_ratio=0.25, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    cut = int(len(X) * (1 - test_ratio))
    return X[idx[:cut]], X[idx[cut:]], y[idx[:cut]], y[idx[cut:]]

def standardize(X_train, X_test):
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std, mean, std

def standardize_targets(y_train, y_test):
    mean = y_train.mean(axis=0, keepdims=True)
    std = y_train.std(axis=0, keepdims=True) + 1e-8
    return (y_train - mean) / std, (y_test - mean) / std, mean, std
