"""
Задание 3 — Блок 2: Классификация MLP + оценка размера выборки + кросс-валидация.

1. Перебор архитектур (активация × слои × нейроны) на 4 датасетах.
2. Оценка минимального размера обучающей выборки для accuracy >= 90 %.
3. 5-fold кросс-валидация для выбора лучшей активации и числа слоёв.
"""

import copy
import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch

from config import (SEED, CLS_N, CLS_NOISE, CLS_EPOCHS, CLS_LR,
                    CV_K, DPI)
from generators import (CLS_DATASETS, make_circles, make_xor,
                         make_blobs, make_spiral,
                         train_test_split, standardize)
from mlp import (build_mlp, train_model, predict_classes,
                 classification_metrics, to_tensor)


# ═══════════════════════════════════════════════════════════════════
#  Подготовка датасетов
# ═══════════════════════════════════════════════════════════════════

def load_cls_datasets(n=CLS_N, noise=CLS_NOISE):
    return [
        {'name': 'Circles',
         'data': make_circles(n=n, noise=noise, seed=21)},
        {'name': 'XOR',
         'data': make_xor(n=n, noise=0.55, seed=22)},
        {'name': 'Blobs',
         'data': make_blobs(n=n, seed=23)},
        {'name': 'Spiral',
         'data': make_spiral(n=n, noise=noise, seed=24)},
    ]


# ═══════════════════════════════════════════════════════════════════
#  Перебор архитектур
# ═══════════════════════════════════════════════════════════════════

def run_classification_experiments(datasets, activations=('sigmoid', 'tanh', 'relu'),
                                   epochs=CLS_EPOCHS, lr=CLS_LR):
    rows = []
    best_models = {}
    archs = list(itertools.product([1, 2, 3], [1, 2, 3, 4, 5]))

    for ds in datasets:
        name = ds['name']
        X, y = ds['data']
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, seed=SEED)
        X_tr_s, X_te_s, x_m, x_s = standardize(X_tr, X_te)

        X_tr_t = to_tensor(X_tr_s)
        X_te_t = to_tensor(X_te_s)
        y_tr_t = to_tensor(y_tr, dtype=torch.long)
        y_te_t = to_tensor(y_te, dtype=torch.long)

        for act in activations:
            best = None
            for h_layers, h_dim in archs:
                model = build_mlp(2, 2, h_layers, h_dim, act)
                hist = train_model(model, X_tr_t, y_tr_t, X_te_t, y_te_t,
                                   task='classification', epochs=epochs, lr=lr)
                pred = predict_classes(model, X_te_t)
                metrics = classification_metrics(y_te, pred)

                rec = {
                    'dataset': name, 'activation': act,
                    'linear_layers': h_layers + 1,
                    'hidden_layers': h_layers, 'hidden_dim': h_dim,
                    'test_accuracy': metrics['accuracy'],
                    'history': hist,
                    'model': copy.deepcopy(model),
                    'X_train': X_tr, 'X_test': X_te,
                    'y_train': y_tr, 'y_test': y_te,
                    'x_mean': x_m, 'x_std': x_s,
                    'confusion_matrix': metrics['confusion_matrix'],
                }
                rows.append(rec)
                if best is None or metrics['accuracy'] > best['test_accuracy']:
                    best = rec

            best_models[(name, act)] = best

    return rows, best_models


# ═══════════════════════════════════════════════════════════════════
#  Сводная таблица (accuracy)
# ═══════════════════════════════════════════════════════════════════

def print_cls_pivot(rows, ds_name, act):
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
            val = next((r['test_accuracy'] for r in subset
                        if r['linear_layers'] == nl and r['hidden_dim'] == hd), None)
            line += f'{val:<{col_w}.4f}' if val is not None else f'{"—":<{col_w}}'
        print(line)


# ═══════════════════════════════════════════════════════════════════
#  Визуализация лучших моделей (boundary + confusion matrix)
# ═══════════════════════════════════════════════════════════════════

def plot_decision_boundary(record):
    model = record['model']
    X_tr, X_te = record['X_train'], record['X_test']
    y_tr, y_te = record['y_train'], record['y_test']
    x_m, x_s = record['x_mean'], record['x_std']

    X_all = np.vstack([X_tr, X_te])
    lo = X_all.min(axis=0) - 0.5
    hi = X_all.max(axis=0) + 0.5

    xx, yy = np.meshgrid(np.linspace(lo[0], hi[0], 250),
                         np.linspace(lo[1], hi[1], 250))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    grid_s = (grid - x_m) / x_s
    zz = predict_classes(model, to_tensor(grid_s)).reshape(xx.shape)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].plot(record['history']['train_loss'], label='train loss')
    axes[0].plot(record['history']['val_loss'], label='test loss')
    axes[0].set_title(
        f"Loss | {record['activation']} | слоёв={record['linear_layers']} "
        f"| нейронов={record['hidden_dim']}")
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].contourf(xx, yy, zz, levels=2, alpha=0.35, cmap='coolwarm')
    axes[1].scatter(X_te[y_te == 0, 0], X_te[y_te == 0, 1], s=18, label='Класс 0')
    axes[1].scatter(X_te[y_te == 1, 0], X_te[y_te == 1, 1], s=18, label='Класс 1')
    axes[1].set_title(
        f"{record['dataset']} | accuracy = {record['test_accuracy']:.3f}")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    cm = record['confusion_matrix']
    axes[2].imshow(cm, cmap='Blues')
    for i in range(2):
        for j in range(2):
            axes[2].text(j, i, str(cm[i, j]),
                         ha='center', va='center', color='black', fontsize=13)
    axes[2].set_xticks([0, 1])
    axes[2].set_yticks([0, 1])
    axes[2].set_xlabel('predicted')
    axes[2].set_ylabel('true')
    axes[2].set_title('Confusion matrix')

    plt.tight_layout()
    plt.show()
    plt.close()


# ═══════════════════════════════════════════════════════════════════
#  Часть II.1 — Оценка минимального размера выборки
# ═══════════════════════════════════════════════════════════════════

def estimate_min_train_size(datasets, activation='tanh', linear_layers=4,
                            hidden_dim=5, target_acc=0.90,
                            epochs=CLS_EPOCHS, lr=CLS_LR):
    fracs = np.arange(0.10, 0.96, 0.05)
    summary = []

    for ds in datasets:
        name = ds['name']
        X, y = ds['data']
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        best_frac = None
        for frac in fracs:
            test_ratio = 1.0 - float(frac)
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_ratio=test_ratio,
                                                       seed=SEED)
            X_tr_s, X_te_s, _, _ = standardize(X_tr, X_te)

            model = build_mlp(2, 2, linear_layers - 1, hidden_dim, activation)
            train_model(model,
                        to_tensor(X_tr_s), to_tensor(y_tr, torch.long),
                        to_tensor(X_te_s), to_tensor(y_te, torch.long),
                        task='classification', epochs=epochs, lr=lr)

            pred = predict_classes(model, to_tensor(X_te_s))
            acc = classification_metrics(y_te, pred)['accuracy']

            if best_frac is None and acc >= target_acc:
                best_frac = float(frac)

        summary.append({
            'dataset': name, 'activation': activation,
            'linear_layers': linear_layers, 'hidden_dim': hidden_dim,
            'target_acc': target_acc,
            'min_frac': best_frac,
            'min_train_size': (None if best_frac is None
                               else int(round(best_frac * len(X)))),
        })

    return summary


# ═══════════════════════════════════════════════════════════════════
#  Часть II.2 — K-fold кросс-валидация
# ═══════════════════════════════════════════════════════════════════

def kfold_indices(n, k=5, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return np.array_split(idx, k)


def cv_grid_search(X, y, activations=('sigmoid', 'tanh', 'relu'),
                   layers_grid=(2, 3, 4), hidden_dim=5,
                   k=CV_K, epochs=CLS_EPOCHS, lr=CLS_LR):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    folds = kfold_indices(len(X), k=k)

    rows = []
    best_row = None

    for act in activations:
        for nl in layers_grid:
            fold_accs = []
            for fi in range(k):
                val_idx = folds[fi]
                tr_idx = np.concatenate([folds[j] for j in range(k) if j != fi])

                X_tr, X_val = X[tr_idx], X[val_idx]
                y_tr, y_val = y[tr_idx], y[val_idx]

                X_tr_s, X_val_s, _, _ = standardize(X_tr, X_val)

                model = build_mlp(2, 2, nl - 1, hidden_dim, act)
                train_model(model,
                            to_tensor(X_tr_s), to_tensor(y_tr, torch.long),
                            to_tensor(X_val_s), to_tensor(y_val, torch.long),
                            task='classification', epochs=epochs, lr=lr)

                pred = predict_classes(model, to_tensor(X_val_s))
                fold_accs.append(classification_metrics(y_val, pred)['accuracy'])

            row = {
                'activation': act, 'linear_layers': nl,
                'hidden_dim': hidden_dim,
                'cv_mean_acc': float(np.mean(fold_accs)),
                'cv_std_acc': (float(np.std(fold_accs, ddof=1))
                               if len(fold_accs) > 1 else 0.0),
            }
            rows.append(row)
            if best_row is None or row['cv_mean_acc'] > best_row['cv_mean_acc']:
                best_row = row

    return rows, best_row


# ═══════════════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════════════

def main():
    datasets = load_cls_datasets()

    # ── 1. Визуализация датасетов ──
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle('Классификационные выборки', fontsize=14, fontweight='bold')
    for ax, ds in zip(axes.flat, datasets):
        X, y = ds['data']
        ax.scatter(X[y == 0, 0], X[y == 0, 1], s=16, alpha=0.75, label='Класс 0')
        ax.scatter(X[y == 1, 0], X[y == 1, 1], s=16, alpha=0.75, label='Класс 1')
        ax.set_title(ds['name'])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()
    plt.close()

    # ── 2. Перебор архитектур ──
    print(f'\nПеребор архитектур (epochs={CLS_EPOCHS}, lr={CLS_LR})…')
    rows, best = run_classification_experiments(datasets)

    activations = ['sigmoid', 'tanh', 'relu']
    for ds in datasets:
        print('\n' + '=' * 70)
        print(ds['name'])
        print('=' * 70)
        for act in activations:
            print(f'\n  Активация: {act}  (Test accuracy)')
            print('  ' + '-' * 62)
            print_cls_pivot(rows, ds['name'], act)

    print('\n\n── Лучшие модели по классификации ──')
    for (ds_name, act), rec in best.items():
        print(f'  {ds_name} | {act}: '
              f'слоёв={rec["linear_layers"]}, нейронов={rec["hidden_dim"]}, '
              f'acc={rec["test_accuracy"]:.4f}')
        plot_decision_boundary(rec)

    # ── 3. Минимальный размер выборки ──
    print('\n── Минимальный размер выборки для accuracy >= 90 % ──')
    size_summary = estimate_min_train_size(datasets)
    col_w = 14
    header = (f"{'Датасет':<12}{'Активация':<{col_w}}{'Слоёв':<{col_w}}"
              f"{'Нейронов':<{col_w}}{'Target':<{col_w}}"
              f"{'Min frac':<{col_w}}{'Min N':<{col_w}}")
    print(header)
    print('-' * len(header))
    for r in size_summary:
        mf = f"{r['min_frac']:.2f}" if r['min_frac'] is not None else '—'
        ms = str(r['min_train_size']) if r['min_train_size'] is not None else '—'
        print(f"{r['dataset']:<12}{r['activation']:<{col_w}}"
              f"{r['linear_layers']:<{col_w}}{r['hidden_dim']:<{col_w}}"
              f"{r['target_acc']:<{col_w}}{mf:<{col_w}}{ms:<{col_w}}")

    # ── 4. Кросс-валидация ──
    print(f'\n── {CV_K}-fold кросс-валидация ──')
    cv_summary = []
    for ds in datasets:
        name = ds['name']
        X, y = ds['data']
        details, best_row = cv_grid_search(X, y)

        cv_summary.append({
            'dataset': name,
            'best_activation': best_row['activation'],
            'best_layers': best_row['linear_layers'],
            'cv_mean_acc': best_row['cv_mean_acc'],
            'cv_std_acc': best_row['cv_std_acc'],
        })

        print(f'\n  {name}: все комбинации')
        print(f"  {'Активация':<12}{'Слоёв':<10}{'N':<8}{'Mean acc':<14}{'Std acc':<14}")
        print('  ' + '-' * 56)
        for r in sorted(details, key=lambda x: x['cv_mean_acc']):
            print(f"  {r['activation']:<12}{r['linear_layers']:<10}"
                  f"{r['hidden_dim']:<8}{r['cv_mean_acc']:<14.4f}"
                  f"{r['cv_std_acc']:<14.4f}")

    print(f'\n  Лучшие результаты {CV_K}-fold CV:')
    print(f"  {'Датасет':<12}{'Активация':<12}{'Слоёв':<10}"
          f"{'Mean acc':<14}{'Std acc':<14}")
    print('  ' + '-' * 60)
    for r in cv_summary:
        print(f"  {r['dataset']:<12}{r['best_activation']:<12}"
              f"{r['best_layers']:<10}{r['cv_mean_acc']:<14.4f}"
              f"{r['cv_std_acc']:<14.4f}")


if __name__ == '__main__':
    main()
