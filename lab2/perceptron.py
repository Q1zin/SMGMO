import numpy as np
import time

class Perceptron:
    def __init__(self, activation='step', lr=0.1, n_epochs=300, seed=42):
        self.activation = activation
        self.lr = lr
        self.n_epochs = n_epochs
        self.seed = seed
        self.w = None
        self.train_time = 0.0

    def _sigmoid(self, a):
        return 1.0 / (1.0 + np.exp(-np.clip(a, -500, 500)))

    def fit(self, x, y):
        rng = np.random.default_rng(self.seed)
        xb = np.c_[np.ones(len(x)), x]
        self.w = rng.normal(0, 0.01, xb.shape[1])
        y_norm = 2 * y - 1

        t0 = time.perf_counter()

        if self.activation == 'step':
            for _ in range(self.n_epochs):
                changed = False
                for i in range(len(xb)):
                    pred = 1 if (xb[i] @ self.w) >= 0 else 0
                    if pred != y[i]:
                        self.w += self.lr * y_norm[i] * xb[i]
                        changed = True
                if not changed:
                    break
        else:
            for _ in range(self.n_epochs):
                yh = self._sigmoid(xb @ self.w)
                delta = -2.0 * (y - yh) * yh * (1.0 - yh)
                grad = (delta @ xb) / len(xb)
                self.w -= self.lr * grad

        self.train_time = time.perf_counter() - t0

    def predict(self, x):
        xb = np.c_[np.ones(len(x)), x]
        a = xb @ self.w
        if self.activation == 'step':
            return (a >= 0).astype(int)
        return (self._sigmoid(a) >= 0.5).astype(int)

    def boundary_x2(self, x1):
        w0, w1, w2 = self.w
        if abs(w2) < 1e-8:
            return None
        return -(w0 + w1 * x1) / w2

def confusion_matrix(y_true, y_pred):
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return np.array([[tn, fp], [fn, tp]])


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))
