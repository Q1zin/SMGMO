import numpy as np
import matplotlib.pyplot as plt

from config import SEED, N, DPI, OUTPUT_FILE_BLOCK2, OUTPUT_FILE_BLOCK2_CM
from generators import DATASETS, split_data
from perceptron import Perceptron, confusion_matrix, accuracy

ACTIVATIONS = [
    ('step',    'Ступенчатая', 0.1,  300),
    ('sigmoid', 'Сигмоида',    0.5,  800),
]


# ── train all models once ──────────────────────────────────────────────
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
                'name':  ds_name,
                'x':     x,
                'x_te':  x_te,
                'y_te':  y_te,
                'model': model,
                'cm':    confusion_matrix(y_te, y_pred),
                'acc':   accuracy(y_te, y_pred),
            })
    return results


# ── figure 1: decision boundaries ─────────────────────────────────────
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

            # decision boundary: w0 + w1*x1 + w2*x2 = 0
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


# ── figure 2: confusion matrices ──────────────────────────────────────
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

    # ── console summary ──────────────────────────────────────────────
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


# ── entry point ────────────────────────────────────────────────────────
def main():
    print(f'Обучение моделей (N = {N})...')
    results = train_all()
    show_boundaries(results)
    show_confusion_matrices(results)


if __name__ == '__main__':
    main()
