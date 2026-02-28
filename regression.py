import numpy as np

def train_test_split(x, y, test_ratio=0.2):
    n = len(x)
    indices = np.random.permutation(n)
    split = int(n * (1 - test_ratio))
    
    train_idx = indices[:split]
    test_idx = indices[split:]
    
    return x[train_idx], y[train_idx], x[test_idx], y[test_idx]

def fit_polynomial(x_train, y_train, degree):
    coeffs = np.polyfit(x_train, y_train, degree)
    return coeffs

def predict(x, coeffs):
    return np.polyval(coeffs, x)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def plot_regression_case(ax, x_train, y_train, x_test, y_test, degree, true_f, title):
    coeffs = fit_polynomial(x_train, y_train, degree)
    
    mse_train = mse(y_train, predict(x_train, coeffs))
    mse_test = mse(y_test, predict(x_test, coeffs))
    
    x_line = np.linspace(-1, 1, 500)
    y_pred_line = predict(x_line, coeffs)
    y_true_line = true_f(x_line)
    
    ax.plot(x_line, y_true_line, 'b-', lw=2, label='истинная f(x)')
    ax.plot(x_line, y_pred_line, 'r-', lw=2, label=f'полином (deg={degree})')
    ax.scatter(x_train, y_train, s=15, alpha=0.6, color='steelblue', label='train')
    ax.scatter(x_test, y_test, s=15, alpha=0.6, color='orange', label='test')
    
    ax.set_title(f'{title}\nMSE train={mse_train:.4f}  |  MSE test={mse_test:.4f}', fontsize=9)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    y_range = np.percentile(np.abs(y_true_line), 99) * 3
    ax.set_ylim(-y_range, y_range)
    
    return mse_train, mse_test
