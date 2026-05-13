import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (BATCH_SIZE, DATA_ROOT, DATASET_NAME, FINAL_EPOCHS,
                    MIN_DELTA, N_EPOCHS, NUM_WORKERS, PATIENCE,
                    PIN_MEMORY, SEED, TARGET_ACC, TEST_SUBSET,
                    TRAIN_SUBSET, VAL_RATIO)
from cnn import (ImageCNN, confusion_matrix, evaluate, get_device,
                 is_valid_config, make_optimizer, set_seed, train_model,
                 train_one_epoch)
from data import (load_image_dataset, make_loaders, plot_random_images,
                  split_train_val, take_subset)


BLOCK_TYPES = ['a', 'b']
CHANNELS_OPTIONS = [(16, 32), (32, 64)]
CONV_KERNELS = [3, 5]
CONV_STRIDES = [1]
POOL_KERNELS = [2]
POOL_STRIDES = [1, 2]
LEARNING_RATES = [1e-3]


def build_grid(image_size=28):
    grid = []
    config_id = 1
    for block_type, channels, conv_k, conv_s, pool_k, pool_s, lr in itertools.product(
        BLOCK_TYPES,
        CHANNELS_OPTIONS,
        CONV_KERNELS,
        CONV_STRIDES,
        POOL_KERNELS,
        POOL_STRIDES,
        LEARNING_RATES,
    ):
        if not is_valid_config(image_size, len(channels), conv_k, conv_s,
                               pool_k, pool_s):
            continue
        grid.append({
            'id': config_id,
            'block_type': block_type,
            'channels': channels,
            'conv_k': conv_k,
            'conv_s': conv_s,
            'pool_k': pool_k,
            'pool_s': pool_s,
            'lr': lr,
        })
        config_id += 1
    return grid


def short_name(cfg):
    return (f"#{cfg['id']:02d} blk{cfg['block_type']} "
            f"ch={cfg['channels']} ck={cfg['conv_k']} ps={cfg['pool_s']}")


def cfg_pretty(cfg):
    return (
        f"block={cfg['block_type']} | channels={cfg['channels']} | "
        f"conv(k={cfg['conv_k']}, s={cfg['conv_s']}) | "
        f"pool(k={cfg['pool_k']}, s={cfg['pool_s']}) | "
        f"lr={cfg['lr']}"
    )


def run_one(cfg, train_loader, val_loader, info, device):
    set_seed(SEED)
    model = ImageCNN(
        in_channels=info['in_channels'],
        num_classes=len(info['classes']),
        block_type=cfg['block_type'],
        channels=cfg['channels'],
        conv_k=cfg['conv_k'],
        conv_s=cfg['conv_s'],
        pool_k=cfg['pool_k'],
        pool_s=cfg['pool_s'],
    ).to(device)
    optimizer = make_optimizer('Adam', model.parameters(), cfg['lr'])
    criterion = nn.CrossEntropyLoss()
    n_params = sum(parameter.numel() for parameter in model.parameters())
    print(f"\n>>> {cfg_pretty(cfg)} | params={n_params:,}")
    res = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        n_epochs=N_EPOCHS,
        target_acc=TARGET_ACC,
        patience=PATIENCE,
        min_delta=MIN_DELTA,
        log_every=2,
        tag=f"#{cfg['id']:02d}",
    )
    res['cfg'] = cfg
    res['model'] = model
    res['n_params'] = n_params
    return res


def train_final_on_full(best_cfg, train_full_ds, test_ds, info, device,
                        n_epochs=FINAL_EPOCHS):
    set_seed(SEED)
    model = ImageCNN(
        in_channels=info['in_channels'],
        num_classes=len(info['classes']),
        block_type=best_cfg['block_type'],
        channels=best_cfg['channels'],
        conv_k=best_cfg['conv_k'],
        conv_s=best_cfg['conv_s'],
        pool_k=best_cfg['pool_k'],
        pool_s=best_cfg['pool_s'],
    ).to(device)

    optimizer = make_optimizer('Adam', model.parameters(), best_cfg['lr'])
    criterion = nn.CrossEntropyLoss()

    use_pin_memory = torch.cuda.is_available() if PIN_MEMORY is None else (
        PIN_MEMORY and torch.cuda.is_available()
    )
    train_loader_full = DataLoader(
        train_full_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=use_pin_memory,
    )
    test_loader_full = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=use_pin_memory,
    )

    history = {'train_loss': [], 'train_acc': []}
    print('\n' + '#' * 80)
    print(' ФИНАЛЬНОЕ ПЕРЕОБУЧЕНИЕ НА ПОЛНОМ TRAIN')
    print('#' * 80)
    print(f'  Конфиг: {cfg_pretty(best_cfg)}')
    print(f'  train size: {len(train_full_ds)} | test size: {len(test_ds)}')

    for epoch in range(1, n_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader_full, criterion, optimizer, device
        )
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        if epoch == 1 or epoch % 2 == 0 or epoch == n_epochs:
            print(f'  [final] эпоха {epoch:>3}/{n_epochs} | '
                  f'train_loss={train_loss:.4f} train_acc={train_acc:.4f}')

    test_loss, test_acc, y_true, y_pred = evaluate(
        model, test_loader_full, criterion, device, return_preds=True
    )
    return {
        'model': model,
        'history': history,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'y_true': y_true,
        'y_pred': y_pred,
        'trained_epochs': n_epochs,
    }


def print_results_table(results):
    print('\n' + '=' * 110)
    print(f"{'#':>3} | {'blk':>3} | {'channels':>10} | {'ck':>2} | {'cs':>2} | {'pk':>2} | {'ps':>2} | {'lr':>7} | {'best_val':>9} | {'ep@90':>5} | {'epochs':>6} | {'time,s':>7}")
    print('-' * 110)
    for result in sorted(results, key=lambda item: -item['best_val_acc']):
        cfg = result['cfg']
        epoch90 = result['epoch_to_target'] if result['epoch_to_target'] else '—'
        print(
            f"{cfg['id']:>3} | {cfg['block_type']:>3} | "
            f"{str(cfg['channels']):>10} | {cfg['conv_k']:>2} | "
            f"{cfg['conv_s']:>2} | {cfg['pool_k']:>2} | {cfg['pool_s']:>2} | "
            f"{cfg['lr']:>7.4f} | {result['best_val_acc']:>9.4f} | "
            f"{str(epoch90):>5} | {result['trained_epochs']:>6} | "
            f"{result['duration_sec']:>7.1f}"
        )
    print('=' * 110)


def plot_curves(history, title, target_acc=TARGET_ACC, best_epoch=None):
    _, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(13, 4.2))

    ax_loss.plot(history['train_loss'], label='train loss')
    ax_loss.plot(history['val_loss'], label='val loss')
    if best_epoch is not None:
        ax_loss.axvline(best_epoch - 1, color='crimson', ls='--',
                        label='best epoch')
    ax_loss.set_title(f'{title} — loss')
    ax_loss.set_xlabel('epoch')
    ax_loss.set_ylabel('loss')
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    ax_acc.plot(history['train_acc'], label='train acc')
    ax_acc.plot(history['val_acc'], label='val acc')
    ax_acc.axhline(target_acc, color='green', ls='--',
                   label=f'target {target_acc:.2f}')
    if best_epoch is not None:
        ax_acc.axvline(best_epoch - 1, color='crimson', ls='--',
                       label='best epoch')
    ax_acc.set_title(f'{title} — accuracy')
    ax_acc.set_xlabel('epoch')
    ax_acc.set_ylabel('accuracy')
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_confusion(cm, class_names, title):
    n_classes = len(class_names)
    _, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(cm, cmap='Blues')
    ax.set_title(title)
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_xlabel('predicted')
    ax.set_ylabel('true')
    threshold = cm.max() / 2 if cm.size else 0
    for row in range(n_classes):
        for col in range(n_classes):
            color = 'white' if cm[row, col] > threshold else 'black'
            ax.text(col, row, str(cm[row, col]), ha='center', va='center',
                    fontsize=8, color=color)
    plt.colorbar(image, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_param_compare(results, param_name, getter, target_acc=TARGET_ACC):
    groups = {}
    for result in results:
        groups.setdefault(getter(result['cfg']), []).append(result['best_val_acc'])
    keys = sorted(groups, key=str)
    means = [float(np.mean(groups[key])) for key in keys]
    stds = [float(np.std(groups[key])) for key in keys]

    _, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(keys))
    ax.bar(x, means, yerr=stds, capsize=4, color='seagreen')
    ax.axhline(target_acc, color='red', ls='--',
               label=f'target {target_acc:.2f}')
    ax.set_title(f'Влияние «{param_name}» на val_acc (mean ± std)')
    ax.set_xticks(x)
    ax.set_xticklabels([str(key) for key in keys])
    ax.set_ylabel('val_acc')
    ax.set_ylim(min(means) * 0.95 if means else 0, 1.0)
    ax.legend()
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_grid_summary(results, target_acc=TARGET_ACC):
    sorted_results = sorted(results, key=lambda result: -result['best_val_acc'])
    labels = [short_name(result['cfg']) for result in sorted_results]
    accuracies = [result['best_val_acc'] for result in sorted_results]
    epochs_to_target = [result['epoch_to_target'] or 0 for result in sorted_results]

    _, axes = plt.subplots(2, 1, figsize=(max(10, len(labels) * 0.55), 8))
    x = np.arange(len(labels))

    bars = axes[0].bar(x, accuracies, color='steelblue')
    axes[0].axhline(target_acc, color='red', ls='--',
                    label=f'target {target_acc:.2f}')
    axes[0].set_title('Best val_acc по экспериментам (по убыванию)')
    axes[0].set_ylim(min(accuracies) * 0.95 if accuracies else 0, 1.0)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=70, ha='right', fontsize=7)
    axes[0].legend()
    for bar, value in zip(bars, accuracies):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 0.002,
                     f'{value:.3f}', ha='center', fontsize=6)

    axes[1].bar(x, epochs_to_target, color='darkorange')
    axes[1].set_title(
        f'Эпоха достижения val_acc >= {target_acc:.2f} (0 = не достигнут)'
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=70, ha='right', fontsize=7)

    plt.tight_layout()
    plt.show()
    plt.close()


def main():
    set_seed(SEED)
    device = get_device()
    print('=' * 80)
    print(f' Задание 4 — Блок 1: перебор гиперпараметров CNN на {DATASET_NAME}')
    print(f' Устройство: {device}')
    print('=' * 80)

    train_full, test_full, info = load_image_dataset(DATASET_NAME, DATA_ROOT)

    train_grid = take_subset(train_full, TRAIN_SUBSET, seed=SEED)
    train_ds, val_ds = split_train_val(train_grid, VAL_RATIO, seed=SEED)
    test_ds = take_subset(test_full, TEST_SUBSET, seed=SEED + 1)

    print(f' train: {len(train_ds)} | val: {len(val_ds)} | test: {len(test_ds)}')
    train_loader, val_loader, test_loader = make_loaders(train_ds, val_ds, test_ds, batch_size=BATCH_SIZE)

    plot_random_images(train_ds, info['classes'], f'{DATASET_NAME}: примеры из train')

    grid = build_grid(image_size=info['image_size'])
    print(f'\nКонфигураций в сетке: {len(grid)}')

    results = []
    for cfg in grid:
        results.append(run_one(cfg, train_loader, val_loader, info, device))

    print_results_table(results)

    best = max(results, key=lambda result: result['best_val_acc'])
    print('\n' + '#' * 80)
    print(' ЛУЧШАЯ КОНФИГУРАЦИЯ')
    print('#' * 80)
    print(f"  {cfg_pretty(best['cfg'])}")
    print(f"  best val_acc:        {best['best_val_acc']:.4f}")
    print(f"  лучшая эпоха:        {best['best_epoch']}")
    print(f"  эпох до 90%:         {best['epoch_to_target']}")
    print(f"  параметров:          {best['n_params']:,}")
    print(f"  время обучения:      {best['duration_sec']:.1f} сек")

    test_loss, test_acc, _, _ = evaluate(
        best['model'], test_loader, nn.CrossEntropyLoss(), device,
        return_preds=True
    )
    print(f'\n  test loss:           {test_loss:.4f}')
    print(f'  test accuracy:       {test_acc:.4f}')

    final_res = train_final_on_full(best['cfg'], train_full, test_full, info,
                                    device)
    print('\n' + '=' * 80)
    print(' Финальные метрики после переобучения на полном train')
    print('=' * 80)
    print(f"  epochs:              {final_res['trained_epochs']}")
    print(f"  final test loss:     {final_res['test_loss']:.4f}")
    print(f"  final test accuracy: {final_res['test_acc']:.4f}")

    plot_grid_summary(results)
    plot_curves(best['history'], title=short_name(best['cfg']), best_epoch=best['best_epoch'])

    cm = confusion_matrix(final_res['y_true'], final_res['y_pred'], len(info['classes']))
    plot_confusion(cm, info['classes'], f'Confusion matrix — final full-train ({DATASET_NAME})')

    plot_param_compare(results, 'block_type', lambda cfg: cfg['block_type'])
    plot_param_compare(results, 'channels', lambda cfg: str(cfg['channels']))
    plot_param_compare(results, 'conv_kernel_size', lambda cfg: cfg['conv_k'])
    plot_param_compare(results, 'pool_stride', lambda cfg: cfg['pool_s'])

    reached = [result for result in results if result['epoch_to_target'] is not None]
    print('\n' + '=' * 80)
    print(' Оценка числа эпох до accuracy >= 90%')
    print('=' * 80)
    if reached:
        epochs = [result['epoch_to_target'] for result in reached]
        print(f'  достигли цели:           {len(reached)}/{len(results)}')
        print(f'  минимум эпох:            {min(epochs)}')
        print(f'  медиана:                 {int(np.median(epochs))}')
        print(f"  у лучшей конфигурации:   {best['epoch_to_target']}")
    else:
        print('  Ни одна конфигурация не достигла 90% — увеличьте N_EPOCHS.')

    return {'results': results, 'best': best, 'final_res': final_res,
            'info': info}


if __name__ == '__main__':
    main()


def cfg_pretty(cfg):
    return (f"block={cfg['block_type']} | channels={cfg['channels']} | "
            f"conv(k={cfg['conv_k']}, s={cfg['conv_s']}) | "
            f"pool(k={cfg['pool_k']}, s={cfg['pool_s']}) | "
            f"lr={cfg['lr']}")


def run_one(cfg, train_loader, val_loader, info, device):
    set_seed(SEED)
    model = ImageCNN(
        in_channels=info['in_channels'],
        num_classes=len(info['classes']),
        block_type=cfg['block_type'],
        channels=cfg['channels'],
        conv_k=cfg['conv_k'], conv_s=cfg['conv_s'],
        pool_k=cfg['pool_k'], pool_s=cfg['pool_s'],
    ).to(device)
    optimizer = make_optimizer('Adam', model.parameters(), cfg['lr'])
    criterion = nn.CrossEntropyLoss()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n>>> {cfg_pretty(cfg)} | params={n_params:,}")
    res = train_model(model, train_loader, val_loader, optimizer, criterion,
                      device, n_epochs=N_EPOCHS, target_acc=TARGET_ACC,
                      patience=PATIENCE, min_delta=MIN_DELTA, log_every=2,
                      tag=f"#{cfg['id']:02d}")
    res['cfg'] = cfg
    res['model'] = model
    res['n_params'] = n_params
    return res


def train_final_on_full(best_cfg, train_full_ds, test_ds, info, device,
                        n_epochs=FINAL_EPOCHS):
    set_seed(SEED)
    model = ImageCNN(
        in_channels=info['in_channels'],
        num_classes=len(info['classes']),
        block_type=best_cfg['block_type'],
        channels=best_cfg['channels'],
        conv_k=best_cfg['conv_k'], conv_s=best_cfg['conv_s'],
        pool_k=best_cfg['pool_k'], pool_s=best_cfg['pool_s'],
    ).to(device)

    optimizer = make_optimizer('Adam', model.parameters(), best_cfg['lr'])
    criterion = nn.CrossEntropyLoss()

    use_pin_memory = torch.cuda.is_available() if PIN_MEMORY is None else (PIN_MEMORY and torch.cuda.is_available())
    train_loader_full = DataLoader(
        train_full_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=use_pin_memory,
    )
    test_loader_full = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=use_pin_memory,
    )

    history = {'train_loss': [], 'train_acc': []}
    t0 = time.perf_counter()
    print('\n' + '#' * 80)
    print(' ФИНАЛЬНОЕ ПЕРЕОБУЧЕНИЕ НА ПОЛНОМ TRAIN')
    print('#' * 80)
    print(f'  Конфиг: {cfg_pretty(best_cfg)}')
    print(f'  train size: {len(train_full_ds)} | test size: {len(test_ds)}')

    for epoch in range(1, n_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader_full, criterion, optimizer, device)
        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        if epoch == 1 or epoch % 2 == 0 or epoch == n_epochs:
            print(f'  [final] эпоха {epoch:>3}/{n_epochs} | '
                  f'train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}')

    test_loss, test_acc, y_true, y_pred = evaluate(
        model, test_loader_full, criterion, device, return_preds=True)
    duration = time.perf_counter() - t0
    return {
        'model': model,
        'history': history,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'y_true': y_true,
        'y_pred': y_pred,
        'duration_sec': duration,
        'trained_epochs': n_epochs,
    }


def print_results_table(results):
    print('\n' + '=' * 110)
    print(f"{'#':>3} | {'blk':>3} | {'channels':>10} | {'ck':>2} | {'cs':>2} | "
          f"{'pk':>2} | {'ps':>2} | {'lr':>7} | "
          f"{'best_val':>9} | {'ep@90':>5} | {'epochs':>6} | {'time,s':>7}")
    print('-' * 110)
    for r in sorted(results, key=lambda x: -x['best_val_acc']):
        c = r['cfg']
        ep90 = r['epoch_to_target'] if r['epoch_to_target'] else '—'
        print(f"{c['id']:>3} | {c['block_type']:>3} | {str(c['channels']):>10} | "
              f"{c['conv_k']:>2} | {c['conv_s']:>2} | "
              f"{c['pool_k']:>2} | {c['pool_s']:>2} | "
              f"{c['lr']:>7.4f} | "
              f"{r['best_val_acc']:>9.4f} | {str(ep90):>5} | "
              f"{r['trained_epochs']:>6} | {r['duration_sec']:>7.1f}")
    print('=' * 110)


def plot_curves(history, title, target_acc=TARGET_ACC, best_epoch=None):
    fig, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(13, 4.2))

    ax_l.plot(history['train_loss'], label='train loss')
    ax_l.plot(history['val_loss'], label='val loss')
    if best_epoch is not None:
        ax_l.axvline(best_epoch - 1, color='crimson', ls='--',
                     label='best epoch')
    ax_l.set_title(f'{title} — loss')
    ax_l.set_xlabel('epoch'); ax_l.set_ylabel('loss')
    ax_l.legend(); ax_l.grid(True, alpha=0.3)

    ax_a.plot(history['train_acc'], label='train acc')
    ax_a.plot(history['val_acc'], label='val acc')
    ax_a.axhline(target_acc, color='green', ls='--',
                 label=f'target {target_acc:.2f}')
    if best_epoch is not None:
        ax_a.axvline(best_epoch - 1, color='crimson', ls='--',
                     label='best epoch')
    ax_a.set_title(f'{title} — accuracy')
    ax_a.set_xlabel('epoch'); ax_a.set_ylabel('accuracy')
    ax_a.legend(); ax_a.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_confusion(cm, class_names, title):
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_title(title)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_xlabel('predicted')
    ax.set_ylabel('true')
    thr = cm.max() / 2 if cm.size else 0
    for i in range(n):
        for j in range(n):
            color = 'white' if cm[i, j] > thr else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=8, color=color)
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_param_compare(results, param_name, getter, target_acc=TARGET_ACC):
    """Группирует по одному параметру и строит mean ± std accuracy."""
    groups = {}
    for r in results:
        groups.setdefault(getter(r['cfg']), []).append(r['best_val_acc'])
    keys = sorted(groups, key=str)
    means = [float(np.mean(groups[k])) for k in keys]
    stds = [float(np.std(groups[k])) for k in keys]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(keys))
    ax.bar(x, means, yerr=stds, capsize=4, color='seagreen')
    ax.axhline(target_acc, color='red', ls='--',
               label=f'target {target_acc:.2f}')
    ax.set_title(f'Влияние «{param_name}» на val_acc (mean ± std)')
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in keys])
    ax.set_ylabel('val_acc')
    ax.set_ylim(min(means) * 0.95 if means else 0, 1.0)
    ax.legend()
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_grid_summary(results, target_acc=TARGET_ACC):
    sorted_res = sorted(results, key=lambda r: -r['best_val_acc'])
    labels = [short_name(r['cfg']) for r in sorted_res]
    accs = [r['best_val_acc'] for r in sorted_res]
    eps90 = [r['epoch_to_target'] or 0 for r in sorted_res]

    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(labels) * 0.55), 8))
    x = np.arange(len(labels))

    bars = axes[0].bar(x, accs, color='steelblue')
    axes[0].axhline(target_acc, color='red', ls='--',
                    label=f'target {target_acc:.2f}')
    axes[0].set_title('Best val_acc по экспериментам (по убыванию)')
    axes[0].set_ylim(min(accs) * 0.95 if accs else 0, 1.0)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=70, ha='right', fontsize=7)
    axes[0].legend()
    for bar, val in zip(bars, accs):
        axes[0].text(bar.get_x() + bar.get_width() / 2, val + 0.002,
                     f'{val:.3f}', ha='center', fontsize=6)

    axes[1].bar(x, eps90, color='darkorange')
    axes[1].set_title(f'Эпоха достижения val_acc >= {target_acc:.2f} '
                      f'(0 = не достигнут)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=70, ha='right', fontsize=7)

    plt.tight_layout()
    plt.show()
    plt.close()

def main():
    set_seed(SEED)
    device = get_device()
    print('=' * 80)
    print(f' Задание 4 — Блок 1: перебор гиперпараметров CNN на {DATASET_NAME}')
    print(f' Устройство: {device}')
    print('=' * 80)

    train_full, test_full, info = load_image_dataset(DATASET_NAME, DATA_ROOT)

    train_grid = take_subset(train_full, TRAIN_SUBSET, seed=SEED)
    train_ds, val_ds = split_train_val(train_grid, VAL_RATIO, seed=SEED)
    test_ds = take_subset(test_full, TEST_SUBSET, seed=SEED + 1)

    print(f' train: {len(train_ds)} | val: {len(val_ds)} | test: {len(test_ds)}')
    train_loader, val_loader, test_loader = make_loaders(
        train_ds, val_ds, test_ds, batch_size=BATCH_SIZE)

    plot_random_images(train_ds, info['classes'],
                       f'{DATASET_NAME}: примеры из train')

    grid = build_grid(image_size=info['image_size'])
    print(f'\nКонфигураций в сетке: {len(grid)}')

    results = []
    for cfg in grid:
        results.append(run_one(cfg, train_loader, val_loader, info, device))

    print_results_table(results)

    best = max(results, key=lambda r: r['best_val_acc'])
    print('\n' + '#' * 80)
    print(' ЛУЧШАЯ КОНФИГУРАЦИЯ')
    print('#' * 80)
    print(f'  {cfg_pretty(best["cfg"])}')
    print(f'  best val_acc:        {best["best_val_acc"]:.4f}')
    print(f'  лучшая эпоха:        {best["best_epoch"]}')
    print(f'  эпох до 90%:         {best["epoch_to_target"]}')
    print(f'  параметров:          {best["n_params"]:,}')
    print(f'  время обучения:      {best["duration_sec"]:.1f} сек')

    test_loss, test_acc, y_true, y_pred = evaluate(
        best['model'], test_loader, nn.CrossEntropyLoss(), device,
        return_preds=True)
    print(f'\n  test loss:           {test_loss:.4f}')
    print(f'  test accuracy:       {test_acc:.4f}')

    final_res = train_final_on_full(best['cfg'], train_full, test_full,
                                    info, device)
    print('\n' + '=' * 80)
    print(' Финальные метрики после переобучения на полном train')
    print('=' * 80)
    print(f'  epochs:              {final_res["trained_epochs"]}')
    print(f'  train time:          {final_res["duration_sec"]:.1f} сек')
    print(f'  final test loss:     {final_res["test_loss"]:.4f}')
    print(f'  final test accuracy: {final_res["test_acc"]:.4f}')

    plot_grid_summary(results)
    plot_curves(best['history'], title=short_name(best['cfg']),
                best_epoch=best['best_epoch'])

    cm = confusion_matrix(final_res['y_true'], final_res['y_pred'],
                          len(info['classes']))
    plot_confusion(cm, info['classes'],
                   f'Confusion matrix — final full-train ({DATASET_NAME})')

    plot_param_compare(results, 'block_type', lambda c: c['block_type'])
    plot_param_compare(results, 'channels', lambda c: str(c['channels']))
    plot_param_compare(results, 'conv_kernel_size', lambda c: c['conv_k'])
    plot_param_compare(results, 'pool_stride', lambda c: c['pool_s'])

    reached = [r for r in results if r['epoch_to_target'] is not None]
    print('\n' + '=' * 80)
    print(' Оценка числа эпох до accuracy >= 90%')
    print('=' * 80)
    if reached:
        eps = [r['epoch_to_target'] for r in reached]
        print(f'  достигли цели:           {len(reached)}/{len(results)}')
        print(f'  минимум эпох:            {min(eps)}')
        print(f'  медиана:                 {int(np.median(eps))}')
        print(f'  у лучшей конфигурации:   {best["epoch_to_target"]}')
    else:
        print('  Ни одна конфигурация не достигла 90% — увеличьте N_EPOCHS.')


if __name__ == '__main__':
    main()
