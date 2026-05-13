import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

from config import (SEED, DATASET_NAME, DATA_ROOT, TRAIN_SUBSET, TEST_SUBSET,
                    VAL_RATIO, BATCH_SIZE, N_EPOCHS, TARGET_ACC,
                    OVERFIT_TRAIN, OVERFIT_EPOCHS)
from data import (load_image_dataset, take_subset, split_train_val,
                  make_loaders)
from cnn import (set_seed, get_device, ImageCNN, make_optimizer,
                 train_model)


BASE_CFG = {
    'block_type': 'b',
    'channels': (32, 64),
    'conv_k': 3, 'conv_s': 1,
    'pool_k': 2, 'pool_s': 2,
}


def make_model(info, device, dropout=0.0):
    return ImageCNN(
        in_channels=info['in_channels'],
        num_classes=len(info['classes']),
        block_type=BASE_CFG['block_type'],
        channels=BASE_CFG['channels'],
        conv_k=BASE_CFG['conv_k'], conv_s=BASE_CFG['conv_s'],
        pool_k=BASE_CFG['pool_k'], pool_s=BASE_CFG['pool_s'],
        dropout_rate=dropout,
    ).to(device)

OPT_VARIANTS = [
    ('Adam', 1e-3),
    ('Adam', 3e-3),
    ('AdamW', 1e-3),
    ('SGD', 1e-2),
    ('SGD', 5e-2),
]


def study_optimizers(train_loader, val_loader, info, device,
                     n_epochs=10, patience=4):
    print('=' * 80)
    print(' Сравнение оптимизаторов и learning rate')
    print('=' * 80)
    print(f' Базовая архитектура: block={BASE_CFG["block_type"]}, '
          f'channels={BASE_CFG["channels"]}, conv_k={BASE_CFG["conv_k"]}')

    results = []
    criterion = nn.CrossEntropyLoss()
    for opt_name, lr in OPT_VARIANTS:
        set_seed(SEED)
        model = make_model(info, device)
        optimizer = make_optimizer(opt_name, model.parameters(), lr)
        print(f'\n--- {opt_name} | lr={lr} ---')
        res = train_model(model, train_loader, val_loader, optimizer,
                          criterion, device, n_epochs=n_epochs,
                          target_acc=TARGET_ACC, patience=patience,
                          min_delta=1e-4, log_every=1,
                          tag=f'{opt_name}-{lr}')
        res['opt_name'] = opt_name
        res['lr'] = lr
        results.append(res)

    print('\nИтог сравнения:')
    print(f"  {'optimizer':>9} | {'lr':>7} | {'best_val':>9} | {'ep@90':>5}")
    print('  ' + '-' * 38)
    for r in results:
        ep = r['epoch_to_target'] if r['epoch_to_target'] else '—'
        print(f"  {r['opt_name']:>9} | {r['lr']:>7.4f} | "
              f"{r['best_val_acc']:>9.4f} | {str(ep):>5}")
    return results


def plot_optimizer_curves(results, target_acc=TARGET_ACC):
    fig, ax = plt.subplots(figsize=(9, 5))
    for r in results:
        epochs = np.arange(1, len(r['history']['val_acc']) + 1)
        ax.plot(epochs, r['history']['val_acc'], 'o-',
                label=f"{r['opt_name']} lr={r['lr']}")
    ax.axhline(target_acc, color='red', ls='--',
               label=f'target {target_acc:.2f}')
    ax.set_title('Сравнение оптимизаторов и learning rate (val accuracy)')
    ax.set_xlabel('epoch')
    ax.set_ylabel('val accuracy')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()
    plt.close()


def run_overfit_demo(info, device):
    print('\n' + '=' * 80)
    print(' Демонстрация переобучения на малой подвыборке')
    print('=' * 80)

    train_full, test_full, _ = load_image_dataset(DATASET_NAME, DATA_ROOT)
    train_small = take_subset(train_full, OVERFIT_TRAIN, seed=SEED)
    train_ds, val_ds = split_train_val(train_small, VAL_RATIO, seed=SEED)
    test_ds = take_subset(test_full, TEST_SUBSET, seed=SEED + 1)

    train_loader, val_loader, _ = make_loaders(
        train_ds, val_ds, test_ds, batch_size=BATCH_SIZE)

    print(f' train: {len(train_ds)} | val: {len(val_ds)}')

    criterion = nn.CrossEntropyLoss()

    print('\n--- без регуляризации (dropout=0) ---')
    set_seed(SEED)
    model_no = make_model(info, device, dropout=0.0)
    optimizer = make_optimizer('Adam', model_no.parameters(), 1e-3)
    res_no = train_model(model_no, train_loader, val_loader, optimizer,
                         criterion, device, n_epochs=OVERFIT_EPOCHS,
                         target_acc=1.0, patience=OVERFIT_EPOCHS,
                         min_delta=0.0, log_every=20, tag='no-reg')

    print('\n--- с Dropout=0.4 ---')
    set_seed(SEED)
    model_dr = make_model(info, device, dropout=0.4)
    optimizer = make_optimizer('Adam', model_dr.parameters(), 1e-3)
    res_dr = train_model(model_dr, train_loader, val_loader, optimizer,
                         criterion, device, n_epochs=OVERFIT_EPOCHS,
                         target_acc=1.0, patience=OVERFIT_EPOCHS,
                         min_delta=0.0, log_every=20, tag='dropout')

    plot_overfit_compare(res_no, res_dr)

    print('\nИтог переобучения:')
    for tag, r in [('без рег.', res_no), ('dropout 0.4', res_dr)]:
        h = r['history']
        gap = h['train_acc'][-1] - h['val_acc'][-1]
        print(f"  {tag:<14} | train_acc={h['train_acc'][-1]:.4f} | "
              f"val_acc={h['val_acc'][-1]:.4f} | gap={gap:+.4f}")


def plot_overfit_compare(res_no, res_dr):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    h1, h2 = res_no['history'], res_dr['history']
    axes[0].plot(h1['train_loss'], label='train (no-reg)')
    axes[0].plot(h1['val_loss'], label='val (no-reg)')
    axes[0].plot(h2['train_loss'], label='train (dropout)', ls='--')
    axes[0].plot(h2['val_loss'], label='val (dropout)', ls='--')
    axes[0].set_title('Loss: переобучение vs Dropout')
    axes[0].set_xlabel('epoch'); axes[0].set_ylabel('loss')
    axes[0].grid(True, alpha=0.3); axes[0].legend()

    axes[1].plot(h1['train_acc'], label='train (no-reg)')
    axes[1].plot(h1['val_acc'], label='val (no-reg)')
    axes[1].plot(h2['train_acc'], label='train (dropout)', ls='--')
    axes[1].plot(h2['val_acc'], label='val (dropout)', ls='--')
    axes[1].set_title('Accuracy: переобучение vs Dropout')
    axes[1].set_xlabel('epoch'); axes[1].set_ylabel('accuracy')
    axes[1].grid(True, alpha=0.3); axes[1].legend()

    plt.tight_layout()
    plt.show()
    plt.close()



def main():
    set_seed(SEED)
    device = get_device()
    print('=' * 80)
    print(f' Задание 4 — Блок 2: оптимизаторы и переобучение ({DATASET_NAME})')
    print(f' Устройство: {device}')
    print('=' * 80)

    train_full, test_full, info = load_image_dataset(DATASET_NAME, DATA_ROOT)
    train_full = take_subset(train_full, TRAIN_SUBSET, seed=SEED)
    train_ds, val_ds = split_train_val(train_full, VAL_RATIO, seed=SEED)
    test_ds = take_subset(test_full, TEST_SUBSET, seed=SEED + 1)
    train_loader, val_loader, _ = make_loaders(
        train_ds, val_ds, test_ds, batch_size=BATCH_SIZE)

    opt_results = study_optimizers(train_loader, val_loader, info, device,
                                   n_epochs=min(N_EPOCHS, 10))
    plot_optimizer_curves(opt_results)

    run_overfit_demo(info, device)


if __name__ == '__main__':
    main()
