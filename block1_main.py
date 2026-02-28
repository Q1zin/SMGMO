import numpy as np
import matplotlib.pyplot as plt

from config import N, FIGURE_SIZE, DPI, OUTPUT_FILE
from functions import f_poly, f_sin, generate_sample

def main():
    x_line = np.linspace(-1, 1, 500)
    
    fig, axes = plt.subplots(2, 3, figsize=FIGURE_SIZE)
    fig.suptitle('Задание 1. Блок I — Генерация выборок', fontsize=14, fontweight='bold')

    configs = [
        (f_poly, 'f = ax³+bx²+cx+d', 0.2, 'uniform'),
        (f_poly, 'f = ax³+bx²+cx+d', 0.5, 'uniform'),
        (f_poly, 'f = ax³+bx²+cx+d', 0.5, 'normal'),
        (f_sin,  'f = x·sin(2πx)',   0.2, 'uniform'),
        (f_sin,  'f = x·sin(2πx)',   0.5, 'uniform'),
        (f_sin,  'f = x·sin(2πx)',   0.5, 'normal'),
    ]
    
    for ax, (f, fname, eps0, etype) in zip(axes.flat, configs):
        x_s, y_s = generate_sample(f, N, eps0, etype)
        y_true = f(x_line)
        
        ax.plot(x_line, y_true, 'b-', linewidth=2, label='f(x)')
        ax.scatter(x_s, y_s, s=15, alpha=0.6, color='tomato', label='выборка')
        ax.fill_between(x_line, y_true - eps0, y_true + eps0, alpha=0.15, color='blue', label=f'±ε₀={eps0}')

        err_label = 'равномерная' if etype == 'uniform' else 'нормальная (правило 3σ)'
        ax.set_title(f'{fname}\nε₀={eps0}, ошибка: {err_label}', fontsize=9)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=DPI, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
