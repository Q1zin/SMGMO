import numpy as np
import matplotlib.pyplot as plt
import os

from config import SEED, N, DPI, OUTPUT_FILE_BLOCK2, OUTPUT_FILE_BLOCK2_CM
from generators import DATASETS, split_data
from perceptron import Perceptron, confusion_matrix, accuracy

def _env_float(name, default):
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return float(default)

def _env_int(name, default):
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return int(default)

STEP_LR = _env_float('SMGMO2_STEP_LR', 0.03)
STEP_EPOCHS = _env_int('SMGMO2_STEP_EPOCHS', 1000)
SIGMOID_LR = _env_float('SMGMO2_SIGMOID_LR', 0.03)
SIGMOID_EPOCHS = _env_int('SMGMO2_SIGMOID_EPOCHS', 1000)

ACTIVATIONS = [
    ('step', 'Ступенчатая', STEP_LR, STEP_EPOCHS),
    ('sigmoid', 'Сигмоида', SIGMOID_LR, SIGMOID_EPOCHS),
]

def train_all():
    results = {}
    for act, _name, lr, epochs in ACTIVATIONS:
        results[act] = []
        for ds_name, make_fn in DATASETS:
            x, y = make_fn(n=N, seed=SEED)
            x_tr, y_tr, x_te, y_te = split_data(x, y, seed=SEED)
            model = Perceptron(activation=act, lr=lr, n_epochs=epochs, seed=SEED)
            model.fit(x_tr, y_tr)
            y_pred = model.predict(x_te)
            results[act].append({
                'name': ds_name,
                'x': x,
                'x_te': x_te,
                'y_te': y_te,
                'model': model,
                'cm': confusion_matrix(y_te, y_pred),
                'acc': accuracy(y_te, y_pred),
            })
    return results

def show_boundaries(results):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(
        f'Задание 2. Блок II — Разделяющие границы  (N = {N})',
        fontsize=13, fontweight='bold',
    )
    colors = ['steelblue', 'tomato']

    for row, (act, act_name, *_) in enumerate(ACTIVATIONS):
        for col, entry in enumerate(results[act]):
            ax = axes[row, col]
            x_te, y_te, x = entry['x_te'], entry['y_te'], entry['x']

            for cls in (0, 1):
                m = y_te == cls
                ax.scatter(x_te[m, 0], x_te[m, 1],
                           s=18, alpha=0.55, color=colors[cls])

            x1 = np.linspace(x[:, 0].min() - 0.3, x[:, 0].max() + 0.3, 400)
            x2 = entry['model'].boundary_x2(x1)
            if x2 is not None:
                y_lo = x[:, 1].min() - 0.3
                y_hi = x[:, 1].max() + 0.3
                mask = (x2 >= y_lo) & (x2 <= y_hi)
                if mask.any():
                    ax.plot(x1[mask], x2[mask], 'k-', lw=2)

            top = entry['name'] if row == 0 else ''
            ax.set_title(f'{top}\nAcc = {entry["acc"]:.3f}', fontsize=10)
            if col == 0:
                ax.set_ylabel(act_name, fontsize=11, fontweight='bold')
            ax.set_xlabel('$x_1$')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE_BLOCK2, dpi=DPI, bbox_inches='tight')
    plt.show()

def _draw_cm(ax, cm, title, acc):
    ax.imshow(cm, interpolation='nearest', cmap='Blues', vmin=0)
    ax.set_title(f'{title}\nAcc = {acc:.3f}', fontsize=10)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Pred 0', 'Pred 1'], fontsize=8)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['True 0', 'True 1'], fontsize=8)
    vmax = cm.max() if cm.max() > 0 else 1
    for i in range(2):
        for j in range(2):
            c = 'white' if cm[i, j] > vmax / 2 else 'black'
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center',
                    color=c, fontsize=13, fontweight='bold')

def show_confusion_matrices(results):
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    fig.suptitle(
        f'Задание 2. Блок II — Матрицы ошибок  (N = {N})',
        fontsize=13, fontweight='bold',
    )

    for row, (act, act_name, *_) in enumerate(ACTIVATIONS):
        for col, entry in enumerate(results[act]):
            ax = axes[row, col]
            top = entry['name'] if row == 0 else ''
            _draw_cm(ax, entry['cm'], top, entry['acc'])
            if col == 0:
                ax.set_ylabel(act_name, fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE_BLOCK2_CM, dpi=DPI, bbox_inches='tight')
    plt.show()

    col_w = 22
    print('\n' + '=' * (10 + col_w * len(ACTIVATIONS)))
    header = f"{'Датасет':<10}"
    for _, name, *_ in ACTIVATIONS:
        header += f"  {(name + ': Acc / Время'):<{col_w - 2}}"
    print(header)
    print('-' * (10 + col_w * len(ACTIVATIONS)))
    for i, (ds_name, _) in enumerate(DATASETS):
        row_str = f'{ds_name:<10}'
        for act, *_ in ACTIVATIONS:
            e = results[act][i]
            row_str += f'  {e["acc"]:.3f} / {e["model"].train_time:.3f}s'.ljust(col_w)
        print(row_str)
    print('=' * (10 + col_w * len(ACTIVATIONS)))

def main():
    print(f'Обучение моделей (N = {N})...')
    results = train_all()
    show_boundaries(results)
    show_confusion_matrices(results)

if __name__ == '__main__':
    main()
