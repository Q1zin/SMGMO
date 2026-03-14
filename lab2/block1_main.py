import os
import matplotlib.pyplot as plt

from config import AXIS_LIMIT, DPI, N, NOISE, OUTPUT_FILE_BLOCK1, SEED
from generators import DATASETS

COLORS = ['steelblue', 'tomato']


def _plot_single(ax, name, make_fn):
    """Draw one scatter on the given Axes."""
    x, y = make_fn(n=N, seed=SEED)
    for cls in (0, 1):
        mask = y == cls
        ax.scatter(x[mask, 0], x[mask, 1],
                   s=20, alpha=0.6, color=COLORS[cls],
                   label=f'класс {cls}')
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-AXIS_LIMIT, AXIS_LIMIT)
    ax.set_ylim(-AXIS_LIMIT, AXIS_LIMIT)


def main_single(ds_name):
    """Show and save one dataset by name."""
    entry = next((it for it in DATASETS if it[0] == ds_name), None)
    if entry is None:
        raise ValueError(f'Unknown dataset: {ds_name}')
    name, make_fn = entry
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.suptitle(
        f'Задание 2. Блок I — {name}  (N = {N}, noise = {NOISE})',
        fontsize=13, fontweight='bold',
    )
    _plot_single(ax, name, make_fn)
    plt.tight_layout()
    out = f'task2_block1_{name.lower()}.png'
    plt.savefig(out, dpi=DPI, bbox_inches='tight')
    print(f'Saved: {out}')
    plt.show()


def main_all():
    """Show all four datasets in one figure (fallback)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        f'Задание 2. Блок I — Генерация выборок  (N = {N}, noise = {NOISE})',
        fontsize=14, fontweight='bold',
    )
    for ax, (name, make_fn) in zip(axes.flat, DATASETS):
        _plot_single(ax, name, make_fn)
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE_BLOCK1, dpi=DPI, bbox_inches='tight')
    plt.show()


def main():
    ds = os.environ.get('SMGMO2_DS', '').strip()
    if ds:
        main_single(ds)
    else:
        main_all()


if __name__ == '__main__':
    main()
