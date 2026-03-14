import numpy as np
import matplotlib.pyplot as plt
import os

from config import (SEED, EPS0)
from functions import f_poly, f_sin, generate_sample
from regression import (train_test_split, fit_polynomial, predict, mse, plot_regression_case)

def main_regression_demo():
    seed = SEED
    N = int(os.environ.get('SMGMO_N', '15'))
    eps0 = EPS0
    np.random.seed(seed)

    functions = [
        (f_poly, 'f = ax³+bx²+cx+d', eps0),
        (f_sin, 'f = x·sin(2πx)', eps0),
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Задание 1. Блок II — Полиномиальная регрессия', fontsize=13, fontweight='bold')
    
    cases = [
        (1, 'Недообучение'),
        (3, 'Норма'),
        (20, 'Переобучение'),
    ]
    
    for row, (f, fname, eps) in enumerate(functions):
        x_all, y_all = generate_sample(f, N, eps, error_type='normal')

        x_train, y_train, x_test, y_test = train_test_split(x_all, y_all, 0.2)

        for col, (degree, case_name) in enumerate(cases):
            ax = axes[row, col]
            if degree == 3 and f == f_sin:
                degree = 8
            plot_regression_case(
                ax, x_train, y_train, x_test, y_test,
                degree, f,
                title=f'{fname}\n{case_name}'
            )
    
    plt.tight_layout()
    plt.savefig('task1_block2.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_mse_vs_degree():
    seed = SEED
    N = int(os.environ.get('SMGMO_N', '15'))
    eps0 = EPS0
    np.random.seed(seed)
    
    functions = [
        (f_poly, 'f = ax³+bx²+cx+d', eps0),
        (f_sin, 'f = x·sin(2πx)', eps0),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Зависимость MSE от степени полинома', fontsize=12, fontweight='bold')

    degrees = list(range(1, 21))

    for ax, (f, fname, eps) in zip(axes, functions):
        x_all, y_all = generate_sample(f, N, eps, error_type='normal')
        x_train, y_train, x_test, y_test = train_test_split(x_all, y_all, 0.2)

        mse_trains, mse_tests = [], []
        for deg in degrees:
            coeffs = fit_polynomial(x_train, y_train, deg)
            mse_trains.append(mse(y_train, predict(x_train, coeffs)))
            mse_tests.append(mse(y_test,  predict(x_test,  coeffs)))

        best_idx = int(np.argmin(mse_tests))
        best_degree = degrees[best_idx]

        ax.plot(degrees, mse_trains, 'b-o', ms=5, label='MSE train')
        ax.plot(degrees, mse_tests, 'r-o', ms=5, label='MSE test')

        if best_degree > degrees[0]:
            ax.axvspan(degrees[0] - 0.5, best_degree - 0.5, alpha=0.1, color='orange', label='недообучение')
        
        ax.axvspan(best_degree - 0.5, best_degree + 0.5, alpha=0.25, color='green', label=f'норма (M={best_degree})')
        
        if best_degree < degrees[-1]:
            ax.axvspan(best_degree + 0.5, degrees[-1] + 0.5, alpha=0.1, color='red', label='переобучение')

        ax.set_yscale('log')
        ax.set_xlabel('Степень полинома')
        ax.set_ylabel('MSE (лог. шкала)')
        ax.set_title(fname)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('task1_block2_mse.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    main_regression_demo()

    plot_mse_vs_degree()

if __name__ == "__main__":
    main()
