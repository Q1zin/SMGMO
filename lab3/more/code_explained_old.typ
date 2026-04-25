// ─────────────────────────────────────────────────────────
#set document(title: "Разбор кода лабораторной работы 3")

#set page(
  paper: "a4",
  margin: (top: 1cm, bottom: 1cm, left: 1cm, right: 1cm),
  numbering: "1",
  number-align: center,
)

#set text(font: "New Computer Modern", size: 13pt, lang: "ru")
#set par(justify: true, leading: 0.7em, first-line-indent: 1.25cm)
#set heading(numbering: "1.")

#show heading: it => {
  set text(weight: "bold")
  set par(first-line-indent: 0pt)
  v(0.4em); it; v(0.2em)
}

#show raw.where(block: true): it => {
  set text(size: 9.5pt)
  block(fill: luma(245), inset: (x: 10pt, y: 8pt), radius: 4pt, width: 100%, it)
}

#outline(title: "Содержание", indent: 1.5em)
#pagebreak()

// ═══════════════════════════════════════════════════════════
= config.py --- конфигурация экспериментов
// ═══════════════════════════════════════════════════════════

Файл хранит все настраиваемые параметры проекта в одном месте.
Если нужно поменять, например, число эпох --- меняем только здесь,
а не в десяти местах по коду.

```python
import os
```
Модуль `os` нужен для чтения переменных окружения (`os.environ`).

```python
SEED = 42
```
Глобальный сид (зерно) генератора случайных чисел.
Благодаря ему при каждом запуске данные и шум генерируются *одинаково* ---
эксперименты воспроизводимы.

```python
REG_N = int(os.environ.get('SMGMO3_REG_N', '280'))
```
Число точек в регрессионной выборке. По умолчанию = 280.\
`os.environ.get('SMGMO3_REG_N', '280')` --- пробуем прочитать
переменную окружения; если её нет, берём строку `'280'`.
Затем `int(...)` преобразует строку в число.

Зачем так: можно запустить из терминала
`SMGMO3_REG_N=500 python block1_main.py` и параметр изменится
*без правки кода*.

```python
REG_EPS0  = float(os.environ.get('SMGMO3_REG_EPS0', '0.18'))
REG_EPOCHS = int(os.environ.get('SMGMO3_REG_EPOCHS', '350'))
REG_LR     = float(os.environ.get('SMGMO3_REG_LR', '0.02'))
```
- `REG_EPS0` --- амплитуда шума для равномерного распределения ($epsilon_0$).
- `REG_EPOCHS` --- число эпох обучения при регрессии.
- `REG_LR` --- learning rate (шаг обучения) для регрессии.

```python
CLS_N      = int(os.environ.get('SMGMO3_CLS_N', '500'))
CLS_NOISE  = float(os.environ.get('SMGMO3_CLS_NOISE', '0.08'))
CLS_EPOCHS = int(os.environ.get('SMGMO3_CLS_EPOCHS', '250'))
CLS_LR     = float(os.environ.get('SMGMO3_CLS_LR', '0.03'))
```
То же, но для задач классификации:
- `CLS_N` --- число объектов (500).
- `CLS_NOISE` --- уровень шума при генерации Circles / Spiral.
- `CLS_EPOCHS` --- эпохи (250).
- `CLS_LR` --- learning rate (0.03).

```python
CV_K = int(os.environ.get('SMGMO3_CV_K', '5'))
```
Число фолдов в кросс-валидации ($K = 5$).

```python
DPI = 130
AXIS_LIMIT = 6.0
```
- `DPI` --- разрешение при сохранении графиков (130 точек на дюйм).
- `AXIS_LIMIT` --- предел осей (используется в GUI).

#pagebreak()

// ═══════════════════════════════════════════════════════════
= generators.py --- генерация данных
// ═══════════════════════════════════════════════════════════

Здесь собраны функции, которые *создают* обучающие и тестовые
выборки для регрессии и классификации.

== Импорты

```python
import numpy as np
from config import SEED, AXIS_LIMIT
```
`numpy` --- библиотека для работы с массивами и математикой.
Из конфига берём `SEED` и `AXIS_LIMIT`.

== Генерация полинома

```python
def random_polynomial(seed=SEED):
    rng = np.random.default_rng(seed)
    a, b, c, d = rng.uniform(-3, 3, 4)
```
Создаём генератор случайных чисел `rng` с фиксированным сидом.
Генерируем 4 коэффициента из равномерного распределения $[-3, 3]$.

```python
    def f(x):
        return a * x**3 + b * x**2 + c * x + d
```
Определяем внутреннюю функцию (замыкание), которая вычисляет
$f(x) = a x^3 + b x^2 + c x + d$ с захваченными коэффициентами.

```python
    return f, (a, b, c, d)
```
Возвращаем саму функцию *и* кортеж коэффициентов (чтобы можно
было напечатать их в лог).

== Тригонометрическая функция

```python
def trig_function():
    def f(x):
        return x * np.sin(2 * np.pi * x)
    return f
```
$f(x) = x sin(2 pi x)$ --- осциллирующая функция.
Коэффициентов нет, поэтому возвращаем только функцию.

== Генерация шума

```python
def generate_noise(n, eps0, mode='uniform', sigma=None, seed=SEED):
    rng = np.random.default_rng(seed)
    if sigma is None:
        sigma = eps0 / 3
```
- `n` --- сколько значений шума нужно.
- `eps0` --- амплитуда (для равномерного: от $-epsilon_0$ до $+epsilon_0$).
- `mode` --- тип распределения (`'uniform'` или `'normal'`).
- Если $sigma$ не задан, берём $sigma = epsilon_0 / 3$.

```python
    if mode == 'uniform':
        return rng.uniform(-eps0, eps0, n)
    if mode == 'normal':
        return rng.normal(0, sigma, n)
    raise ValueError(f'Unknown noise mode: {mode}')
```
Возвращаем массив шума нужного типа.
Если `mode` не распознан --- выбрасываем ошибку.

== Генерация регрессионной выборки

```python
def generate_regression_sample(f, n, eps0, noise_mode='uniform',
                                sigma=None, seed=SEED):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
```
Генерируем $n$ равномерно распределённых точек $x in [-1, 1]$.

```python
    noise = generate_noise(n, eps0, mode=noise_mode,
                            sigma=sigma, seed=seed + 1)
```
Генерируем шум (с другим сидом `seed+1`, чтобы шум
не коррелировал с позициями $x$).

```python
    y = f(x) + noise
    return x.reshape(-1, 1), y.reshape(-1, 1)
```
$y = f(x) + epsilon$ --- зашумлённые значения.\
`reshape(-1, 1)` --- превращаем одномерный массив в столбец
(матрицу $N times 1$), потому что PyTorch ожидает двумерный вход.

== Генерация Circles

```python
def make_circles(n=400, noise=0.08, factor=0.35, seed=SEED):
    rng = np.random.default_rng(seed)
    n_outer = n // 2
    n_inner = n - n_outer
```
Делим $n$ объектов пополам: половина --- внешний круг (класс 0),
половина --- внутренний (класс 1).

```python
    theta_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    theta_inner = np.linspace(0, 2 * np.pi, n_inner, endpoint=False)
```
Равномерно расставляем углы $theta$ от 0 до $2 pi$ для каждого
кольца. `endpoint=False` --- не включаем конечную точку, чтобы
точки не наложились.

```python
    outer = np.column_stack([np.cos(theta_outer),
                             np.sin(theta_outer)])
    inner = factor * np.column_stack([np.cos(theta_inner),
                                       np.sin(theta_inner)])
```
Внешний круг: радиус = 1 (единичная окружность).\
Внутренний: радиус = `factor` (0.35), то есть маленький кружок
внутри большого.

```python
    X = np.vstack([outer, inner])
    y = np.concatenate([np.zeros(n_outer, dtype=int),
                        np.ones(n_inner, dtype=int)])
```
Склеиваем координаты и метки классов.

```python
    if noise > 0:
        X += rng.normal(0, noise, X.shape)
    return X, y
```
Добавляем гауссов шум, чтобы кружки были «размазанными».

== Генерация XOR

```python
def make_xor(n=400, noise=0.55, scale=2.2, seed=SEED):
    rng = np.random.default_rng(seed)
    centers = np.array([[scale, scale], [scale, -scale],
                         [-scale, scale], [-scale, -scale]])
```
Четыре «облака» в углах квадрата.\
Диагональные пары (左上 + 右下, 右上 + 左下) принадлежат одному
классу --- это и есть XOR-паттерн.

```python
    per = n // 4
    rem = n % 4
    X_parts, y_parts = [], []
    for i, c in enumerate(centers):
        cnt = per + (1 if i < rem else 0)
        X_parts.append(c + noise * rng.standard_normal((cnt, 2)))
        y_parts.append(np.full(cnt, 0 if i in (0, 3) else 1, dtype=int))
```
Разбиваем $n$ точек поровну на 4 кластера.
`rng.standard_normal` --- нормальный шум $N(0, 1)$, домноженный
на `noise`.
Класс 0 --- кластеры 0 и 3 (левый-нижний и правый-верхний),
класс 1 --- кластеры 1 и 2.

```python
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]
```
Перемешиваем (`permutation`), чтобы объекты шли не подряд по
кластерам.

== Генерация Blobs

```python
def make_blobs(n=400, centers=None, cluster_std=0.9, seed=SEED):
```
Два гауссовых облака (самый простой случай: линейно разделимые
данные).

```python
    if centers is None:
        centers = np.array([[-2.5, -2.0], [2.3, 2.0]], dtype=float)
```
Центры по умолчанию: $(-2.5, -2)$ и $(2.3, 2)$.

Далее аналогично XOR: генерируем точки вокруг каждого центра,
присваиваем метки, перемешиваем.

== Генерация Spiral

```python
def make_spiral(n=400, turns=1.5, radius=0.08, sweep=0.28,
                noise=0.08, seed=SEED):
```
Две спирали (класс 0 и класс 1), закрученные вокруг начала
координат.

```python
    n0 = n // 2
    n1 = n - n0
    theta0 = np.linspace(0, 2 * np.pi * turns, n0)
    theta1 = np.linspace(0, 2 * np.pi * turns, n1)
```
Углы $theta$ от 0 до $2 pi times 1.5$ (полтора оборота).

```python
    r0 = radius + sweep * theta0
    r1 = radius + sweep * theta1
```
Радиус линейно растёт с углом --- получается спираль Архимеда.

```python
    X0 = np.column_stack([r0 * np.cos(theta0),
                          r0 * np.sin(theta0)])
    X1 = np.column_stack([r1 * np.cos(theta1 + np.pi),
                          r1 * np.sin(theta1 + np.pi)])
```
Вторая спираль повёрнута на $pi$ (180°) относительно первой.

Далее: склеиваем, добавляем шум, перемешиваем.

== CLS_DATASETS

```python
CLS_DATASETS = [
    ('Circles', make_circles),
    ('XOR',     make_xor),
    ('Blobs',   make_blobs),
    ('Spiral',  make_spiral),
]
```
Список пар (название, функция-генератор) --- удобно для перебора
в цикле.

== Разбиение на train / test

```python
def train_test_split(X, y, test_ratio=0.25, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    cut = int(len(X) * (1 - test_ratio))
    return X[idx[:cut]], X[idx[cut:]], y[idx[:cut]], y[idx[cut:]]
```
Перемешиваем индексы и делим: первые `cut` --- train, остальные --- test.
По умолчанию 75% / 25%.

== Стандартизация признаков

```python
def standardize(X_train, X_test):
    mean = X_train.mean(axis=0, keepdims=True)
    std  = X_train.std(axis=0, keepdims=True) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std, mean, std
```
Формула: $hat(x) = (x - mu) / sigma$.

Важно: $mu$ и $sigma$ считаются *только по train*.
Тест-часть нормализуется теми же статистиками --- иначе была бы
утечка информации.\
`+ 1e-8` --- защита от деления на ноль.

== Стандартизация целевых значений

```python
def standardize_targets(y_train, y_test):
    mean = y_train.mean(axis=0, keepdims=True)
    std  = y_train.std(axis=0, keepdims=True) + 1e-8
    return (y_train - mean) / std, (y_test - mean) / std, mean, std
```
Для регрессии полезно стандартизировать и $y$, чтобы MSE-loss
был в однородном масштабе.

#pagebreak()

// ═══════════════════════════════════════════════════════════
= mlp.py --- нейросеть
// ═══════════════════════════════════════════════════════════

Ядро проекта: построение MLP, обучение, предсказание, метрики.

== Импорты

```python
import copy
import numpy as np
import torch
import torch.nn as nn
from config import SEED
```
- `copy` --- для глубокого копирования весов модели (`deepcopy`).
- `torch` --- основная библиотека для нейросетей.
- `torch.nn` --- модули слоёв, функций потерь и т.д.

```python
torch.manual_seed(SEED)
device = torch.device('cpu')
```
Фиксируем сид PyTorch для воспроизводимости инициализации весов.
`device='cpu'` --- считаем на процессоре (без GPU).

== Словарь активаций

```python
ACTIVATIONS = {
    'sigmoid': nn.Sigmoid,
    'tanh':    nn.Tanh,
    'relu':    nn.ReLU,
}
```
Маппинг строка → класс PyTorch.
Позволяет выбирать активацию по имени: `ACTIVATIONS['relu']`
вернёт *класс* `nn.ReLU`.

== Конвертация в тензор

```python
def to_tensor(array, dtype=torch.float32):
    return torch.tensor(np.asarray(array), dtype=dtype, device=device)
```
Преобразует numpy-массив (или список) в PyTorch-тензор нужного типа.
`np.asarray` --- на случай, если передали не numpy-массив.

== Построение MLP

```python
def build_mlp(input_dim, output_dim, hidden_layers,
              hidden_dim, activation):
    layers = []
    in_f = input_dim
    act_cls = ACTIVATIONS[activation]
```
- `input_dim` --- размер входа (1 для регрессии, 2 для классификации).
- `output_dim` --- размер выхода (1 или 2).
- `hidden_layers` --- количество *скрытых* слоёв.
- `hidden_dim` --- нейронов в каждом скрытом слое.
- `act_cls` --- класс активации (например, `nn.Tanh`).

```python
    for _ in range(hidden_layers):
        layers.append(nn.Linear(in_f, hidden_dim))
        layers.append(act_cls())
        in_f = hidden_dim
```
В цикле добавляем пары: *линейный слой* + *активация*.
`nn.Linear(in_f, hidden_dim)` --- полносвязный слой: $h = W x + b$,
где $W$ --- матрица $"in\_f" times "hidden\_dim"$.
После первого слоя `in_f` становится `hidden_dim`, потому что
следующий слой принимает на вход выходы предыдущего.

```python
    layers.append(nn.Linear(in_f, output_dim))
    return nn.Sequential(*layers).to(device)
```
Последний (выходной) слой *без активации*:
- для регрессии выход --- произвольное число;
- для классификации выход --- логиты (CrossEntropyLoss сам
  применяет softmax).

`nn.Sequential` --- контейнер, который прокидывает вход через все
слои последовательно.

*Пример* для `build_mlp(1, 1, 2, 4, 'tanh')`:
```
Linear(1 → 4) → Tanh → Linear(4 → 4) → Tanh → Linear(4 → 1)
```
Итого 3 линейных слоя (2 скрытых + 1 выходной).

== Обучение с ранней остановкой

```python
def train_model(model, X_train, y_train, X_val, y_val,
                task='regression', epochs=300, lr=0.01):
```

```python
    loss_fn = (nn.MSELoss() if task == 'regression'
               else nn.CrossEntropyLoss())
```
Выбираем функцию потерь:
- *MSE* для регрессии: $L = (1/N) sum (hat(y) - y)^2$.
- *CrossEntropy* для классификации:
  $L = -(1/N) sum y log hat(y)$.

```python
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
```
Adam --- адаптивный оптимизатор.
`model.parameters()` --- все обучаемые веса $W$ и смещения $b$.

```python
    history = {'train_loss': [], 'val_loss': []}
    best_state = None
    best_val = float('inf')
```
- `history` --- будет хранить значения loss на каждой эпохе
  (для графиков).
- `best_state` --- копия весов модели с наименьшим val loss.
- `best_val` --- минимальный val loss, пока $= +infinity$.

```python
    for _ in range(epochs):
```
Основной цикл обучения. Одна *эпоха* = один проход по всем
данным.

```python
        model.train()
```
Переключаем модель в режим обучения (влияет на Dropout, BatchNorm
и т.п., хотя в нашей сети их нет).

```python
        optimizer.zero_grad()
```
*Обнуляем градиенты*. PyTorch по умолчанию *накапливает*
градиенты при каждом `backward()`. Если не обнулить ---
градиенты сложатся с предыдущей итерации.

```python
        out = model(X_train)
        loss = loss_fn(out, y_train)
```
*Forward pass*: пропускаем данные через сеть, получаем
предсказания `out` и считаем loss.

```python
        loss.backward()
```
*Backward pass* (обратное распространение ошибки):
PyTorch вычисляет $partial L / partial w$ для каждого
параметра $w$ модели.

```python
        optimizer.step()
```
*Обновление весов*: Adam корректирует каждый параметр,
используя вычисленные градиенты.

```python
        model.eval()
        with torch.no_grad():
            val_out = model(X_val)
            val_loss = loss_fn(val_out, y_val)
```
- `model.eval()` --- режим оценки.
- `torch.no_grad()` --- отключаем вычисление градиентов
  (экономим память и время, градиенты на валидации не нужны).
- Считаем loss на валидационной (тестовой) части.

```python
        history['train_loss'].append(float(loss.item()))
        history['val_loss'].append(float(val_loss.item()))
```
`.item()` извлекает скалярное значение из тензора.
Сохраняем для графика «loss по эпохам».

```python
        if val_loss.item() < best_val:
            best_val = float(val_loss.item())
            best_state = copy.deepcopy(model.state_dict())
```
*Ранняя остановка (early stopping)*:
если текущий val loss --- лучший за всё время, сохраняем
*полную копию* весов модели (`deepcopy` + `state_dict()`).

```python
    if best_state is not None:
        model.load_state_dict(best_state)
    return history
```
В конце загружаем обратно лучшие веса. Таким образом модель
не «портится» от лишних эпох.

== Обучение БЕЗ ранней остановки

```python
def train_no_early_stop(model, X_train, y_train, X_val, y_val,
                         epochs=2500, lr=0.01):
```
Аналогичный цикл, но *без сохранения лучших весов*.
Модель обучается все `epochs` эпох подряд.
Используется для *демонстрации переобучения*: мы намеренно
не останавливаем обучение, чтобы увидеть, как test loss
начинает расти.

== Предсказание для регрессии

```python
def predict_regression(model, X_t, y_mean, y_std):
    model.eval()
    with torch.no_grad():
        pred = model(X_t).cpu().numpy()
    return pred * y_std + y_mean
```
Модель предсказывает в *стандартизированном* масштабе.
Чтобы получить реальные значения, делаем обратное
преобразование: $hat(y)_"real" = hat(y)_"std" dot sigma_y + mu_y$.

== Предсказание для классификации

```python
def predict_classes(model, X_t):
    model.eval()
    with torch.no_grad():
        logits = model(X_t)
        return torch.argmax(logits, dim=1).cpu().numpy()
```
Модель выдаёт 2 числа (логита) для каждого объекта.
`argmax` по оси 1 берёт индекс максимального ---
это и есть предсказанный класс (0 или 1).

== Метрика MSE

```python
def mse_np(y_true, y_pred):
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))
```
$"MSE" = (1/N) sum (y - hat(y))^2$ --- среднеквадратичная ошибка.

== Метрики классификации

```python
def classification_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
```
`.ravel()` --- сделать одномерным (на случай, если пришла матрица).

```python
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
```
Подсчёт элементов confusion matrix:
- *TN* (True Negative): истинный = 0, предсказано = 0.
- *TP* (True Positive): истинный = 1, предсказано = 1.
- *FP* (False Positive): истинный = 0, предсказано = 1 (ошибка I рода).
- *FN* (False Negative): истинный = 1, предсказано = 0 (ошибка II рода).

```python
    acc  = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
```
- *Accuracy* = $(T P + T N) / N$.
- *Precision* = $T P / (T P + F P)$.
- *Recall* = $T P / (T P + F N)$.
- `max(..., 1)` --- защита от деления на ноль.

```python
    cm = np.array([[tn, fp], [fn, tp]])
    return {'accuracy': float(acc), 'precision': float(prec),
            'recall': float(rec), 'confusion_matrix': cm}
```
Возвращаем словарь со всеми метриками и матрицей ошибок.

#pagebreak()

// ═══════════════════════════════════════════════════════════
= block1_main.py --- регрессия
// ═══════════════════════════════════════════════════════════

Запускает все эксперименты по регрессии + демо переобучения.

== Импорты и подготовка данных

```python
import copy, itertools, os
import numpy as np
import matplotlib.pyplot as plt
```
- `itertools.product` --- для перебора всех комбинаций
  (слои × нейроны).
- `matplotlib.pyplot` --- рисование графиков.

```python
poly_f, poly_coeffs = random_polynomial(seed=7)
trig_f = trig_function()
```
Создаём две функции *при загрузке модуля*:
полином (с конкретным сидом = 7) и $x sin(2 pi x)$.

```python
REGRESSION_DATASETS = [
    {
        'name': 'Полином + равн. шум',
        'func': poly_f,
        'data': generate_regression_sample(poly_f, REG_N, REG_EPS0,
                    noise_mode='uniform', seed=11),
    },
    {
        'name': 'x·sin(2πx) + норм. шум',
        'func': trig_f,
        'data': generate_regression_sample(trig_f, REG_N, 0.12,
                    noise_mode='normal', sigma=0.05, seed=13),
    },
]
```
Два датасета. Каждый хранит:
- `name` --- название для печати.
- `func` --- «чистая» функция (для рисования истинной кривой).
- `data` --- сгенерированные точки $(x, y)$.

```python
IMG_DIR = os.path.join(os.path.dirname(
              os.path.abspath(__file__)), 'images')
```
Путь к папке `images/` рядом со скриптом.

```python
DS_SLUG = {
    'Полином + равн. шум': 'poly',
    'x·sin(2πx) + норм. шум': 'trig'
}
```
Короткие имена для файлов картинок: `reg_poly_tanh.png` и т.д.

== Перебор архитектур

```python
def run_regression_experiments(datasets, ...):
    archs = list(itertools.product([1, 2, 3], [1, 2, 3, 4, 5]))
```
Все комбинации: 1--3 скрытых слоя × 1--5 нейронов = 15 вариантов.

```python
    for ds in datasets:
        X_tr, X_te, y_tr, y_te = train_test_split(x, y, seed=SEED)
        X_tr_s, X_te_s, x_m, x_s = standardize(X_tr, X_te)
        y_tr_s, y_te_s, y_m, y_s = standardize_targets(y_tr, y_te)
```
Для каждого датасета:
1. Разбиваем на train/test (75/25).
2. Стандартизируем $x$ и $y$ *по train-части*.
3. Запоминаем $mu, sigma$ чтобы потом «развернуть» предсказания.

```python
        X_tr_t = to_tensor(X_tr_s)
        ...
```
Конвертируем numpy → тензоры PyTorch.

```python
        for act in activations:
            best = None
            for h_layers, h_dim in archs:
                model = build_mlp(1, 1, h_layers, h_dim, act)
                hist = train_model(model, ...)
                pred = predict_regression(model, X_te_t, y_m, y_s)
                test_mse = mse_np(y_te, pred)
```
Для каждой комбинации (активация × архитектура):
1. Строим MLP.
2. Обучаем с ранней остановкой.
3. Предсказываем на тесте.
4. Считаем MSE.

```python
                if best is None or test_mse < best['test_mse']:
                    best = rec
```
Запоминаем лучшую модель (с наименьшей MSE) для каждой активации.

== Вывод таблицы

```python
def print_pivot(rows, ds_name, act):
```
Печатает таблицу «Слоёв × N нейронов» для конкретного датасета
и активации. Перебирает строки, ищет нужную комбинацию через
`next(...)`.

== График лучшей регрессии

```python
def plot_best_regression(record, save_path=None):
```
Рисует 2 графика рядом:
1. *Кривые loss* --- train (синяя) и test (оранжевая) по эпохам.
2. *Аппроксимация* --- точки train/test, истинная $f(x)$ (чёрная),
   и предсказание MLP (красная).

```python
    x_dense = np.linspace(-1, 1, 400).reshape(-1, 1)
    x_dense_s = (x_dense - x_m) / x_s
    y_dense = predict_regression(model, to_tensor(x_dense_s), y_m, y_s)
```
400 равноотстоящих точек для плавной кривой предсказания.
Стандартизируем вход, прогоняем через модель, обратно
денормализуем выход.

== Демонстрация переобучения

```python
def run_overfit_demo():
```
Цель --- наглядно показать, что переобучение *существует*.

```python
    small_idx = rng.choice(len(X_tr_full), size=10, replace=False)
    X_tr = X_tr_full[small_idx]
```
Берём *всего 10 точек* из train-части --- очень мало данных.

```python
    model = build_mlp(1, 1, hidden_layers=8, hidden_dim=5,
                       activation='tanh')
```
Строим *слишком большую* сеть: 8 скрытых слоёв по 5 нейронов.
У неё тысячи параметров, а данных --- 10 точек.

```python
    hist = train_no_early_stop(model, ..., epochs=5000, lr=0.01)
```
Обучаем 5000 эпох *без ранней остановки*.
Модель «запомнит» 10 точек (train MSE $arrow$ 0),
но на тесте будет плохо (test MSE растёт).

```python
    best_epoch = int(np.argmin(hist['val_loss'])) + 1
```
`np.argmin` --- индекс минимума val loss.
Это эпоха, после которой начинается деградация.

== main()

```python
def main():
    rows, best = run_regression_experiments(REGRESSION_DATASETS)
```
Запускаем все эксперименты, получаем результаты и лучшие модели.

Далее: печатаем таблицы, рисуем графики для лучших моделей,
запускаем демо переобучения.

#pagebreak()

// ═══════════════════════════════════════════════════════════
= block2_main.py --- классификация
// ═══════════════════════════════════════════════════════════

Запускает эксперименты по классификации, оценку min N
и кросс-валидацию.

== Загрузка датасетов

```python
def load_cls_datasets(n=CLS_N, noise=CLS_NOISE):
    return [
        {'name': 'Circles', 'data': make_circles(n=n, ...)},
        {'name': 'XOR',     'data': make_xor(n=n, ...)},
        {'name': 'Blobs',   'data': make_blobs(n=n, ...)},
        {'name': 'Spiral',  'data': make_spiral(n=n, ...)},
    ]
```
4 датасета, каждый с $n = 500$ объектами.

== Перебор архитектур

```python
def run_classification_experiments(datasets, ...):
```
Аналогично регрессии, но:
- `build_mlp(2, 2, ...)` --- вход 2D, выход 2 класса.
- `y_tr_t = to_tensor(y_tr, dtype=torch.long)` --- метки классов
  должны быть *целыми числами* (`long`), потому что
  `CrossEntropyLoss` ожидает индексы, а не float.
- Метрика --- `accuracy` вместо MSE.
- Лучшая модель --- та, у которой accuracy *максимальная*
  (а не минимальная, как MSE).

== Границы решений

```python
def plot_decision_boundary(record, save_path=None):
```

```python
    xx, yy = np.meshgrid(
        np.linspace(lo[0], hi[0], 250),
        np.linspace(lo[1], hi[1], 250))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
```
Создаём *сетку* $250 times 250 = 62500$ точек, покрывающих всю
плоскость. Для каждой точки спрашиваем модель: «какой класс?»

```python
    zz = predict_classes(model, to_tensor(grid_s)).reshape(xx.shape)
```
Получаем предсказанный класс для каждой точки сетки.
`reshape` --- возвращаем обратно в форму матрицы для рисования.

```python
    axes[1].contourf(xx, yy, zz, levels=2, alpha=0.35, cmap='coolwarm')
```
`contourf` --- заливка контуров: синяя область = класс 0,
красная = класс 1. `alpha=0.35` --- полупрозрачная.

Третий график --- confusion matrix (`imshow`).

== Минимальный размер выборки

```python
def estimate_min_train_size(datasets, activation='tanh',
      linear_layers=4, hidden_dim=5, target_acc=0.90, ...):
```
Фиксируем архитектуру (tanh, 4 слоя, 5 нейронов).
Перебираем долю train от 10% до 95% с шагом 5%.

```python
    fracs = np.arange(0.10, 0.96, 0.05)
    ...
    for frac in fracs:
        test_ratio = 1.0 - float(frac)
        X_tr, X_te, ... = train_test_split(X, y, test_ratio=..., ...)
        ...
        if best_frac is None and acc >= target_acc:
            best_frac = float(frac)
```
Как только accuracy $>= 90%$ --- останавливаемся.
`best_frac` --- минимальная доля, при которой это произошло.

== K-fold кросс-валидация

```python
def kfold_indices(n, k=5, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return np.array_split(idx, k)
```
Перемешиваем $n$ индексов и разбиваем на $k$ примерно равных
частей (фолдов). `np.array_split` --- может создавать части
разного размера (если $n$ не делится на $k$).

```python
def cv_grid_search(X, y, activations=..., layers_grid=(2, 3, 4),
                    hidden_dim=5, k=CV_K, ...):
```
Для каждой комбинации (активация × число слоёв):

```python
    for act in activations:
        for nl in layers_grid:
            fold_accs = []
            for fi in range(k):
                val_idx = folds[fi]
                tr_idx = np.concatenate(
                    [folds[j] for j in range(k) if j != fi])
```
На $i$-й итерации:
- фолд $i$ --- валидация ($T'_i$);
- остальные $k-1$ фолдов --- обучение ($S'_i$).

```python
                X_tr, X_val = X[tr_idx], X[val_idx]
                ...
                model = build_mlp(2, 2, nl - 1, hidden_dim, act)
                train_model(model, ...)
                pred = predict_classes(model, ...)
                fold_accs.append(
                    classification_metrics(y_val, pred)['accuracy'])
```
Обучаем *новую* модель на $S'_i$, оцениваем на $T'_i$.
Повторяем $k$ раз.

```python
            row = {
                ...
                'cv_mean_acc': float(np.mean(fold_accs)),
                'cv_std_acc': float(np.std(fold_accs, ddof=1)),
            }
```
Усредняем accuracy по $k$ фолдам. `ddof=1` --- несмещённая
оценка стандартного отклонения.

Выбираем комбинацию с наибольшим `cv_mean_acc`.

== main()

```python
def main():
    datasets = load_cls_datasets()
```

Последовательность действий:
1. Визуализация 4 датасетов (scatter plot, сохранение в `cls_datasets.png`).
2. Перебор архитектур → таблицы accuracy.
3. Графики для лучших моделей.
4. Оценка минимального размера выборки.
5. 5-fold кросс-валидация → выбор лучшей архитектуры для каждого датасета.
