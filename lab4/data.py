import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from config import SEED, NORMALIZE_IMAGES, NUM_WORKERS, PIN_MEMORY


DATASET_INFO = {
    'MNIST': {
        'loader': datasets.MNIST,
        'classes': [str(i) for i in range(10)],
        'in_channels': 1,
        'image_size': 28,
    },
    'FashionMNIST': {
        'loader': datasets.FashionMNIST,
        'classes': [
            'T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot',
        ],
        'in_channels': 1,
        'image_size': 28,
    },
}

DATASET_NORMALIZATION = {
    'MNIST': {
        'mean': (0.1307,),
        'std': (0.3081,),
    },
    'FashionMNIST': {
        'mean': (0.2860,),
        'std': (0.3530,),
    },
}

def load_image_dataset(name, data_root):
    if name not in DATASET_INFO:
        raise ValueError(f'Неизвестный датасет: {name}')
    info = DATASET_INFO[name]
    tfms = [transforms.ToTensor()]
    if NORMALIZE_IMAGES and name in DATASET_NORMALIZATION:
        stats = DATASET_NORMALIZATION[name]
        tfms.append(transforms.Normalize(stats['mean'], stats['std']))
    transform = transforms.Compose(tfms)
    train = info['loader'](root=data_root, train=True,
                           download=True, transform=transform)
    test = info['loader'](root=data_root, train=False,
                          download=True, transform=transform)
    return train, test, info


def take_subset(dataset, subset_size, seed=SEED):
    rng = np.random.default_rng(seed)
    n = min(subset_size, len(dataset))
    idx = rng.choice(len(dataset), size=n, replace=False)
    return Subset(dataset, idx.tolist())


def split_train_val(dataset, val_ratio, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(dataset))
    rng.shuffle(idx)
    n_val = int(len(dataset) * val_ratio)
    n_train = len(dataset) - n_val
    return (Subset(dataset, idx[:n_train].tolist()),
            Subset(dataset, idx[n_train:].tolist()))


def make_loaders(train_ds, val_ds, test_ds, batch_size, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY):
    use_pin_memory = torch.cuda.is_available() if pin_memory is None else (pin_memory and torch.cuda.is_available())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_pin_memory)
    return train_loader, val_loader, test_loader

def plot_random_images(dataset, class_names, title,
                       rows=4, cols=4, seed=SEED):
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.2 * rows))
    fig.suptitle(title, fontsize=13)
    for ax in axes.ravel():
        i = int(rng.integers(0, len(dataset)))
        img, label = dataset[i]
        ax.imshow(img.squeeze(0), cmap='gray')
        ax.set_title(class_names[int(label)], fontsize=9)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close()
