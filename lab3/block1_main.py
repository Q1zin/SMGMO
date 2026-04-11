"""
Задание 3 — Блок 1: Регрессия MLP.

Перебор архитектур (активация × слои × нейроны) на двух функциях:
  1. Случайный полином 3-й степени
  2. x·sin(2πx)

Визуализация лучших моделей + демонстрация переобучения.
"""

import copy
import itertools
import numpy as np
import matplotlib.pyplot as plt

from config import SEED, REG_N, REG_EPS0, REG_EPOCHS, REG_LR, DPI
from generators import (random_polynomial, trig_function,
                         generate_regression_sample,
                         train_test_split, standardize, standardize_targets)
from mlp import (build_mlp, train_model, train_no_early_stop,
                 predict_regression, mse_np, to_tensor)


# ═══════════════════════════════════════════════════════════════════
#  Подготовка данных
# ═══════════════════════════════════════════════════════════════════

poly_f, poly_coeffs = random_polynomial(seed=7)
trig_f = trig_function()

REGRESSION_DATASETS = [
    {
        'name': 'Полином + равн. шум',
        'func': poly_f,
        'data': generate_regression_sample(poly_f, REG_N, REG_EPS0,
                                           noise_mode='uniform', seed=11),
    },
    {
        'name': 'x·sin(2πx) + норм. шум',
        'func': trig_f,
        'data': generate_regression_sample(trig_f, REG_N, 0.12,
                                           noise_mode='normal', sigma=0.05,
                                           seed=13),
    },
]


# ═══════════════════════════════════════════════════════════════════
#  Перебор архитектур
# ═══════════════════════════════════════════════════════════════════

def run_regression_experiments(datasets, activations=('sigmoid', 'tanh', 'relu'),
                               epochs=REG_EPOCHS, lr=REG_LR):
    rows = []
    best_models = {}
    archs = list(itertools.product([1, 2, 3], [1, 2, 3, 4, 5]))

    for ds in datasets:
        name = ds['name']
        x, y = ds['data']
        func = ds['func']

        X_tr, X_te, y_tr, y_te = train_test_split(x, y, seed=SEED)
        X_tr_s, X_te_s, x_m, x_s = standardize(X_tr, X_te)
        y_tr_s, y_te_s, y_m, y_s = standardize_targets(y_tr, y_te)

        X_tr_t = to_tensor(X_tr_s)
        X_te_t = to_tensor(X_te_s)
        y_tr_t = to_tensor(y_tr_s)
        y_te_t = to_tensor(y_te_s)

        for act in activations:
            best = None
            for h_layers, h_dim in archs:
                model = build_mlp(1, 1, h_layers, h_dim, act)
                hist = train_model(model, X_tr_t, y_tr_t, X_te_t, y_te_t,
                                   task='regression', epochs=epochs, lr=lr)
                pred = predict_regression(model, X_te_t, y_m, y_s)
                test_mse = mse_np(y_te, pred)

                rec = {
                    'dataset': name, 'activation': act,
                    'linear_layers': h_layers + 1,
                    'hidden_layers': h_layers, 'hidden_dim': h_dim,
                    'test_mse': test_mse, 'history': hist,
                    'model': copy.deepcopy(model), 'func': func,
                    'x_train': X_tr, 'x_test': X_te,
                    'y_train': y_tr, 'y_test': y_te,
                    'x_mean': x_m, 'x_std': x_s,
                    'y_mean': y_m, 'y_std': y_s,
                }
                rows.append(rec)
                if best is None or test_mse < best['test_mse']:
                    best = rec

            best_models[(name, act)] = best

    return rows, best_models


# ═══════════════════════════════════════════════════════════════════
#  Сводная таблица
# ═══════════════════════════════════════════════════════════════════

def print_pivot(rows, ds_name, act):
    subset = [r for r in rows
              if r['dataset'] == ds_name and r['activation'] == act]
    all_layers = sorted({r['linear_layers'] for r in subset})
    all_dims = sorted({r['hidden_dim'] for r in subset})

    col_w = 12
    header = f"{'Слоёв':<8}" + ''.join(f"{'N=' + str(d):<{col_w}}" for d in all_dims)
    print(header)
    print('-' * len(header))
    for nl in all_layers:
        line = f'{nl:<8}'
        for hd in all_dims:
            val = next((r['test_mse'] for r in subset
                        if r['linear_layers'] == nl and r['hidden_dim'] == hd), None)
            line += f'{val:<{col_w}.6f}' if val is not None else f'{"—":<{col_w}}'
        print(line)


# ═══════════════════════════════════════════════════════════════════
#  Визуализация лучших моделей
# ═══════════════════════════════════════════════════════════════════

def plot_best_regression(record):
    model = record['model']
    func = record['func']
    x_m, x_s = record['x_mean'], record['x_std']
    y_m, y_s = record['y_mean'], record['y_std']

    x_dense = np.linspace(-1, 1, 400).reshape(-1, 1)
    x_dense_s = (x_dense - x_m) / x_s
    y_dense = predict_regression(model, to_tensor(x_dense_s), y_m, y_s)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    axes[0].plot(record['history']['train_loss'], label='train loss')
    axes[0].plot(record['history']['val_loss'], label='test loss')
    axes[0].set_title(
        f"Loss | {record['activation']} | слоёв={record['linear_layers']} "
        f"| нейронов={record['hidden_dim']}")
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(record['x_train'][:, 0], record['y_train'][:, 0],
                    s=16, alpha=0.65, label='train')
    axes[1].scatter(record['x_test'][:, 0], record['y_test'][:, 0],
                    s=20, alpha=0.8, label='test')
    axes[1].plot(x_dense[:, 0], func(x_dense[:, 0]),
                 color='black', lw=2, label='истинная f(x)')
    axes[1].plot(x_dense[:, 0], y_dense[:, 0],
                 color='crimson', lw=2, label='MLP')
    axes[1].set_title(
        f"{record['dataset']} | test MSE = {record['test_mse']:.4f}")
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    plt.close()


# ═══════════════════════════════════════════════════════════════════
#  Переобучение на x·sin(2πx)
# ═══════════════════════════════════════════════════════════════════

def run_overfit_demo():
    ds = next(d for d in REGRESSION_DATASETS if 'sin' in d['name'])
    x, y = ds['data']
    func = ds['func']

    X_tr_full, X_te, y_tr_full, y_te = train_test_split(x, y, seed=SEED)

    rng = np.random.default_rng(SEED)
    small_idx = rng.choice(len(X_tr_full), size=10, replace=False)
    X_tr = X_tr_full[small_idx]
    y_tr = y_tr_full[small_idx]

    X_tr_s, X_te_s, x_m, x_s = standardize(X_tr, X_te)
    y_tr_s, y_te_s, y_m, y_s = standardize_targets(y_tr, y_te)

    X_tr_t = to_tensor(X_tr_s)
    X_te_t = to_tensor(X_te_s)
    y_tr_t = to_tensor(y_tr_s)
    y_te_t = to_tensor(y_te_s)

    model = build_mlp(1, 1, hidden_layers=8, hidden_dim=5, activation='tanh')
    hist = train_no_early_stop(model, X_tr_t, y_tr_t, X_te_t, y_te_t,
                               epochs=5000, lr=0.01)

    pred_tr = predict_regression(model, X_tr_t, y_m, y_s)
    pred_te = predict_regression(model, X_te_t, y_m, y_s)
    tr_mse = mse_np(y_tr, pred_tr)
    te_mse = mse_np(y_te, pred_te)

    best_epoch = int(np.argmin(hist['val_loss'])) + 1

    print(f'\nПереобучение на x·sin(2πx)')
    print(f'  Train объектов: {len(X_tr)}')
    print(f'  Train MSE: {tr_mse:.6f}')
    print(f'  Test  MSE: {te_mse:.6f}')
    print(f'  Лучшая эпоха по val loss: {best_epoch}')

    # ── График ──
    x_dense = np.linspace(-1, 1, 500).reshape(-1, 1)
    x_dense_s = (x_dense - x_m) / x_s
    y_dense = predict_regression(model, to_tensor(x_dense_s), y_m, y_s)

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))

    axes[0].plot(hist['train_loss'], label='train loss')
    axes[0].plot(hist['val_loss'], label='test loss')
    axes[0].axvline(best_epoch - 1, color='crimson', ls='--',
                    label='минимум test loss')
    axes[0].set_title('Переобучение: train / test loss')
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(X_tr[:, 0], y_tr[:, 0], s=28, alpha=0.85, label='train')
    axes[1].scatter(X_te[:, 0], y_te[:, 0], s=24, alpha=0.65, label='test')
    axes[1].plot(x_dense[:, 0], func(x_dense[:, 0]),
                 color='black', lw=2, label='истинная f(x)')
    axes[1].plot(x_dense[:, 0], y_dense[:, 0],
                 color='crimson', lw=2, label='MLP (переобучение)')
    axes[1].set_title('Переобученная аппроксимация x·sin(2πx)')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    plt.close()


# ═══════════════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════════════

def main():
    print(f'Коэффициенты полинома: {poly_coeffs}')
    print(f'Запуск экспериментов по регрессии (epochs={REG_EPOCHS}, lr={REG_LR})…')

    rows, best = run_regression_experiments(REGRESSION_DATASETS)

    activations = ['sigmoid', 'tanh', 'relu']
    ds_names = [d['name'] for d in REGRESSION_DATASETS]

    for ds_name in ds_names:
        print('\n' + '=' * 70)
        print(ds_name)
        print('=' * 70)
        for act in activations:
            print(f'\n  Активация: {act}  (Test MSE)')
            print('  ' + '-' * 62)
            print_pivot(rows, ds_name, act)

    print('\n\n── Лучшие модели по регрессии ──')
    for (ds_name, act), rec in best.items():
        print(f'  {ds_name} | {act}: '
              f'слоёв={rec["linear_layers"]}, нейронов={rec["hidden_dim"]}, '
              f'MSE={rec["test_mse"]:.6f}')
        plot_best_regression(rec)

    print('\n── Демонстрация переобучения ──')
    run_overfit_demo()


if __name__ == '__main__':
    main()
