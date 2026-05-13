"""
Два типа свёрточных блоков:
  блок 'a': Conv2d -> ReLU -> MaxPool
  блок 'b': Conv2d -> ReLU -> Conv2d -> ReLU -> MaxPool
"""

import copy
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn

from config import SEED


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device('mps')
    return torch.device('cpu')


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class ConvBlockA(nn.Module):
    """Conv2d -> ReLU -> MaxPool."""
    def __init__(self, in_ch, out_ch, conv_k, conv_s, pool_k, pool_s):
        super().__init__()
        pad = conv_k // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=conv_k,
                      stride=conv_s, padding=pad),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_k, stride=pool_s),
        )

    def forward(self, x):
        return self.block(x)


class ConvBlockB(nn.Module):
    """Conv2d -> ReLU -> Conv2d -> ReLU -> MaxPool."""
    def __init__(self, in_ch, out_ch, conv_k, conv_s, pool_k, pool_s):
        super().__init__()
        pad = conv_k // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=conv_k,
                      stride=conv_s, padding=pad),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=conv_k,
                      stride=1, padding=pad),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_k, stride=pool_s),
        )

    def forward(self, x):
        return self.block(x)


class ImageCNN(nn.Module):
    def __init__(self, in_channels, num_classes, block_type, channels,
                 conv_k, conv_s, pool_k, pool_s, dropout_rate=0.0):
        super().__init__()
        if block_type not in ('a', 'b'):
            raise ValueError("block_type должен быть 'a' или 'b'")
        Block = ConvBlockA if block_type == 'a' else ConvBlockB

        layers = []
        prev = in_channels
        for ch in channels:
            layers.append(Block(prev, ch, conv_k, conv_s, pool_k, pool_s))
            prev = ch
        self.features = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(prev, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def is_valid_config(image_size, num_blocks, conv_k, conv_s, pool_k, pool_s):
    """Проверяет, что после всех блоков пространственный размер >= 1."""
    s = image_size
    pad = conv_k // 2
    for _ in range(num_blocks):
        s = math.floor((s + 2 * pad - conv_k) / conv_s + 1)
        if s < pool_k:
            return False
        s = math.floor((s - pool_k) / pool_s + 1)
        if s < 1:
            return False
    return True


def make_optimizer(name, params, lr):
    if name == 'Adam':
        return torch.optim.Adam(params, lr=lr)
    if name == 'AdamW':
        return torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    if name == 'SGD':
        return torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True)
    raise ValueError(f'Неизвестный оптимизатор: {name}')


def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        total_correct += (out.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, total_correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, return_preds=False):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    y_true_all, y_pred_all = [], []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        out = model(x)
        loss = criterion(out, y)
        preds = out.argmax(dim=1)
        total_loss += loss.item() * y.size(0)
        total_correct += (preds == y).sum().item()
        total += y.size(0)
        if return_preds:
            y_true_all.extend(y.cpu().numpy())
            y_pred_all.extend(preds.cpu().numpy())
    avg_loss = total_loss / total
    avg_acc = total_correct / total
    if return_preds:
        return avg_loss, avg_acc, np.array(y_true_all), np.array(y_pred_all)
    return avg_loss, avg_acc


def train_model(model, train_loader, val_loader, optimizer, criterion, device,
                n_epochs, target_acc, patience, min_delta,
                log_every=5, tag=''):
    history = {'train_loss': [], 'val_loss': [],
               'train_acc': [], 'val_acc': []}
    best_val_acc = -1.0
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epoch_to_target = None
    bad = 0

    t0 = time.perf_counter()
    for epoch in range(1, n_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader,
                                          criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(val_acc)

        if epoch == 1 or epoch % log_every == 0 or epoch == n_epochs:
            prefix = f'[{tag}] ' if tag else ''
            print(f'  {prefix}эпоха {epoch:>3}/{n_epochs} | '
                  f'train_loss={tr_loss:.4f} val_loss={val_loss:.4f} | '
                  f'train_acc={tr_acc:.4f} val_acc={val_acc:.4f}')

        if epoch_to_target is None and val_acc >= target_acc:
            epoch_to_target = epoch

        if val_acc > best_val_acc + min_delta:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1

        if bad >= patience:
            print(f'  -> ранняя остановка на эпохе {epoch}')
            break

    model.load_state_dict(best_state)
    duration = time.perf_counter() - t0

    return {
        'history': history,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'epoch_to_target': epoch_to_target,
        'duration_sec': duration,
        'trained_epochs': len(history['train_loss']),
    }
