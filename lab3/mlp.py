import copy
import numpy as np
import torch
import torch.nn as nn

from config import SEED

torch.manual_seed(SEED)
device = torch.device('cpu')

ACTIVATIONS = {
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh,
    'relu': nn.ReLU,
}

def to_tensor(array, dtype=torch.float32):
    return torch.tensor(np.asarray(array), dtype=dtype, device=device)

def build_mlp(input_dim, output_dim, hidden_layers, hidden_dim, activation):
    layers = []
    in_f = input_dim
    act_cls = ACTIVATIONS[activation]

    for _ in range(hidden_layers):
        layers.append(nn.Linear(in_f, hidden_dim))
        layers.append(act_cls())
        in_f = hidden_dim

    layers.append(nn.Linear(in_f, output_dim))
    return nn.Sequential(*layers).to(device)

def train_model(model, X_train, y_train, X_val, y_val, task='regression', epochs=300, lr=0.01):
    loss_fn = nn.MSELoss() if task == 'regression' else nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {'train_loss': [], 'val_loss': []}
    best_state = None
    best_val = float('inf')

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X_train)
        loss = loss_fn(out, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(X_val)
            val_loss = loss_fn(val_out, y_val)

        history['train_loss'].append(float(loss.item()))
        history['val_loss'].append(float(val_loss.item()))

        if val_loss.item() < best_val:
            best_val = float(val_loss.item())
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    return history

def train_no_early_stop(model, X_train, y_train, X_val, y_val, epochs=2500, lr=0.01):
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {'train_loss': [], 'val_loss': []}

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X_train)
        loss = loss_fn(out, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(X_val)
            val_loss = loss_fn(val_out, y_val)

        history['train_loss'].append(float(loss.item()))
        history['val_loss'].append(float(val_loss.item()))

    return history

def predict_regression(model, X_t, y_mean, y_std):
    model.eval()
    with torch.no_grad():
        pred = model(X_t).cpu().numpy()
    return pred * y_std + y_mean

def predict_classes(model, X_t):
    model.eval()
    with torch.no_grad():
        logits = model(X_t)
        return torch.argmax(logits, dim=1).cpu().numpy()

def mse_np(y_true, y_pred):
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def classification_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    cm = np.array([[tn, fp], [fn, tp]])
    return {'accuracy': float(acc), 'precision': float(prec), 'recall': float(rec), 'confusion_matrix': cm}
