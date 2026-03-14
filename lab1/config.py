import numpy as np
import os

SEED = int(42)
N = int(os.environ.get('SMGMO_N', '100'))
EPS0 = float(0.3)

np.random.seed(SEED)

a, b, c, d = np.random.uniform(-3, 3, size=4)
print(f"Коэффициенты полинома: a={a:.3f}, b={b:.3f}, c={c:.3f}, d={d:.3f}")
print(f"Параметры: N={N}, ε₀={EPS0}, seed={SEED}")

FIGURE_SIZE = (15, 9)
DPI = 150
OUTPUT_FILE = 'task1_block1.png'
