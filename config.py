import numpy as np

np.random.seed(42)


N = 100

a, b, c, d = np.random.uniform(-3, 3, size=4)
print(f"Коэффициенты полинома: a={a:.3f}, b={b:.3f}, c={c:.3f}, d={d:.3f}")

FIGURE_SIZE = (15, 9)
DPI = 150
OUTPUT_FILE = 'task1_block1.png'
