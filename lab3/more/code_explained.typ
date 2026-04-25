// ─────────────────────────────────────────────────────────
#set document(title: "Разбор кода лабораторной работы 3")

#set page(
  paper: "a4",
  margin: (x: 1.6cm, y: 1.6cm),
)
#set text(font: "New Computer Modern", size: 11pt, lang: "ru")
#set heading(numbering: "1.1.")
#set par(justify: true, leading: 0.58em)

#align(center)[
  #text(size: 18pt, weight: "bold")[Построчный разбор кода\ лабораторной работы №3]
  #v(0.3cm)
  #text(size: 11pt)[_Каждая строка каждого файла — с номером и пояснением_]
]

#v(0.5cm)

#outline(title: "Содержание", depth: 2)

#pagebreak()

// ══════════════════════════════════════════════════════════
= config.py — глобальные параметры
// ══════════════════════════════════════════════════════════

Файл содержит *18 строк*. Все гиперпараметры задаются через переменные окружения с
дефолтными значениями, чтобы можно было менять настройки без правки кода.

#table(
  columns: (auto, 1fr, 1fr),
  align: (center, left, left),
  table.header[*Строка*][*Код*][*Пояснение*],

  [1], [`import os`],
  [Импортируем модуль `os` стандартной библиотеки — нужен для чтения переменных окружения через `os.environ.get`.],

  [2], [(пустая строка)],
  [Визуальное разделение: отделяем импорт от объявлений констант.],

  [3], [`SEED = 42`],
  [Глобальный seed для воспроизводимости (`np.random`, `torch`). Число 42 — произвольное, но зато каждый запуск одинаков.],

  [4], [(пустая строка)],
  [Пустая строка-разделитель между блоками параметров.],

  [5], [`REG_N = int(os.environ.get('SMGMO3_REG_N', '280'))`],
  [Число объектов выборки для *регрессии*. Берётся из `SMGMO3_REG_N`, если задана; иначе 280. `int(...)` приводит строку к целому.],

  [6], [`REG_EPS0 = float(os.environ.get(...))`],
  [Полу-ширина равномерного шума $epsilon_0 = 0.18$ для регрессии.],

  [7], [`REG_EPOCHS = int(os.environ.get(...))`],
  [Количество эпох обучения регрессионных моделей (по умолчанию $350$).],

  [8], [`REG_LR = float(os.environ.get(...))`],
  [Learning rate для Adam-оптимизатора в задачах регрессии ($0.02$).],

  [9], [(пустая строка)],
  [Разделитель между регрессионным и классификационным блоками.],

  [10], [`CLS_N = int(os.environ.get('SMGMO3_CLS_N', '500'))`],
  [Число объектов для классификационных выборок ($500$).],

  [11], [`CLS_NOISE = float(os.environ.get(...))`],
  [Уровень шума в классификационных данных ($0.08$).],

  [12], [`CLS_EPOCHS = int(os.environ.get(...))`],
  [Кол-во эпох классификации ($250$ по умолчанию; в экспериментах мы передали $500$).],

  [13], [`CLS_LR = float(os.environ.get(...))`],
  [Learning rate классификации ($0.03$).],

  [14], [(пустая строка)],
  [Разделитель.],

  [15], [`CV_K = int(os.environ.get('SMGMO3_CV_K', '5'))`],
  [Число фолдов для $k$-fold кросс-валидации ($5$).],

  [16], [(пустая строка)],
  [Разделитель.],

  [17], [`DPI = 130`],
  [Разрешение сохраняемых PNG-изображений (точек на дюйм).],

  [18], [`AXIS_LIMIT = 6.0`],
  [Граница осей для визуализации (используется при отрисовке Grid в GUI).],
)

#pagebreak()

// ══════════════════════════════════════════════════════════
= generators.py — генераторы данных
// ══════════════════════════════════════════════════════════

Файл содержит ~130 строк. Реализует генерацию регрессионных и классификационных
выборок, а также `train_test_split` и стандартизацию.

== Строки 1–2: Импорты

#table(
  columns: (auto, 1fr, 1fr),
  align: (center, left, left),
  table.header[*Строка*][*Код*][*Пояснение*],

  [1], [`import numpy as np`],
  [Импорт NumPy — основная библиотека для работы с массивами и математическими операциями.],

  [2], [`from config import SEED, AXIS_LIMIT`],
  [Из файла `config.py` берём `SEED` (для воспроизводимости) и `AXIS_LIMIT` (граница координат).],
)

== Строки 4–11: random_polynomial()

#table(
  columns: (auto, 1fr, 1fr),
  align: (center, left, left),
  table.header[*Строка*][*Код*][*Пояснение*],

  [3], [(пустая строка)],
  [Разделитель между импортами и телом файла.],

  [4], [`def random_polynomial(seed=SEED):`],
  [Объявляем функцию, генерирующую случайный полином 3-й степени. Параметр `seed` — для воспроизводимости.],

  [5], [`    rng = np.random.default_rng(seed)`],
  [Создаём генератор случайных чисел (новый NumPy API). Каждый вызов с тем же `seed` даёт те же числа.],

  [6], [`    a, b, c, d = rng.uniform(-3, 3, 4)`],
  [Генерируем 4 случайных коэффициента из равномерного распределения $[-3; 3]$ и распаковываем в `a, b, c, d`.],

  [7], [(пустая строка)],
  [Разделитель внутри функции.],

  [8], [`    def f(x):`],
  [Объявляем вложенную функцию-замыкание (closure), которая «запоминает» коэффициенты `a, b, c, d`.],

  [9], [`        return a * x**3 + b * x**2 + c * x + d`],
  [Вычисляем полином $a x^3 + b x^2 + c x + d$. Работает поэлементно с массивами NumPy.],

  [10], [(пустая строка)],
  [Разделитель.],

  [11], [`    return f, (a, b, c, d)`],
  [Возвращаем кортеж: саму функцию `f` и набор коэффициентов `(a, b, c, d)` для вывода в отчёт.],
)

== Строки 13–16: trig_function()

#table(
  columns: (auto, 1fr, 1fr),
  align: (center, left, left),
  table.header[*Строка*][*Код*][*Пояснение*],

  [12], [(пустая строка)],
  [Разделитель.],

  [13], [`def trig_function():`],
  [Объявляем функцию-фабрику для тригонометрической функции.],

  [14], [`    def f(x):`],
  [Внутренняя функция — целевая зависимость.],

  [15], [`        return x * np.sin(2 * np.pi * x)`],
  [Формула $y = x sin(2 pi x)$. `np.pi` ≈ 3.14159. `np.sin` работает поэлементно.],

  [16], [`    return f`],
  [Возвращаем замыкание `f`.],
)

== Строки 18–27: generate_noise()

#table(
  columns: (auto, 1fr, 1fr),
  align: (center, left, left),
  table.header[*Строка*][*Код*][*Пояснение*],

  [17], [(пустая строка)],
  [Разделитель.],

  [18], [`def generate_noise(n, eps0, mode='uniform', sigma=None, seed=SEED):`],
  [Генератор шума. `n` — число точек, `eps0` — амплитуда, `mode` — тип распределения, `sigma` — σ для нормального.],

  [19], [`    rng = np.random.default_rng(seed)`],
  [Создаём локальный генератор с заданным seed.],

  [20], [`    if sigma is None:`],
  [Если σ не задана явно...],

  [21], [`        sigma = eps0 / 3`],
  [...по умолчанию берём $sigma = epsilon_0 / 3$ — правило «трёх сигм»: 99.7 % значений попадут в $[-epsilon_0; epsilon_0]$.],

  [22], [`    if mode == 'uniform':`],
  [Проверяем: нужен ли равномерный шум?],

  [23], [`        return rng.uniform(-eps0, eps0, n)`],
  [Генерируем `n` значений из $cal(U)(-epsilon_0, epsilon_0)$.],

  [24], [`    if mode == 'normal':`],
  [Проверяем: нужен ли нормальный шум?],

  [25], [`        return rng.normal(0, sigma, n)`],
  [Генерируем `n` значений из $cal(N)(0, sigma)$.],

  [26], [`    raise ValueError(f'Unknown noise mode: {mode}')`],
  [Если `mode` не 'uniform' и не 'normal' — бросаем ошибку с понятным сообщением.],
)

== Строки 28–33: generate_regression_sample()

#table(
  columns: (auto, 1fr, 1fr),
  align: (center, left, left),
  table.header[*Строка*][*Код*][*Пояснение*],

  [27], [(пустая строка)],
  [Разделитель.],

  [28], [`def generate_regression_sample(f, n, eps0, noise_mode='uniform', sigma=None, seed=SEED):`],
  [Главная функция генерации регрессионной выборки. `f` — целевая функция, `n` — число точек, `eps0` — амплитуда шума.],

  [29], [`    rng = np.random.default_rng(seed)`],
  [Генератор для x-координат.],

  [30], [`    x = rng.uniform(-1, 1, n)`],
  [Генерируем `n` случайных $x in [-1, 1]$.],

  [31], [`    noise = generate_noise(n, eps0, mode=noise_mode, sigma=sigma, seed=seed + 1)`],
  [Создаём шум. `seed + 1` — чтобы шум и x-координаты генерировались с разными seed, иначе были бы зависимы.],

  [32], [`    y = f(x) + noise`],
  [Вычисляем «зашумлённое» значение: $y = f(x) + epsilon$.],

  [33], [`    return x.reshape(-1, 1), y.reshape(-1, 1)`],
  [Возвращаем массивы формы $(n, 1)$. `reshape(-1, 1)` превращает вектор в столбец — нужно для PyTorch `Linear(1, ...)`.],
)

== Строки 35–50: make_circles()

#table(
  columns: (auto, 1fr, 1fr),
  align: (center, left, left),
  table.header[*Строка*][*Код*][*Пояснение*],

  [34], [(пустая строка)],
  [Разделитель.],

  [35], [`def make_circles(n=400, noise=0.08, factor=0.35, seed=SEED):`],
  [Генерирует два концентрических кольца (Circles). `factor` — отношение радиусов.],

  [36], [`    rng = np.random.default_rng(seed)`],
  [Генератор.],

  [37], [`    n_outer = n // 2`],
  [Половина точек — внешнее кольцо.],

  [38], [`    n_inner = n - n_outer`],
  [Остальные — внутреннее. Если `n` нечётное, внутреннее получит на 1 больше.],

  [39], [(пустая строка)],
  [Разделитель.],

  [40], [`    theta_outer = np.linspace(0, 2*np.pi, n_outer, endpoint=False)`],
  [Равномерно располагаем углы от $0$ до $2 pi$ для внешнего кольца. `endpoint=False` — точка $2 pi$ не включена (совпадает с $0$).],

  [41], [`    theta_inner = np.linspace(0, 2*np.pi, n_inner, endpoint=False)`],
  [Аналогично для внутреннего кольца.],

  [42], [(пустая строка)],
  [Разделитель.],

  [43], [`    outer = np.column_stack([np.cos(theta_outer), np.sin(theta_outer)])`],
  [Координаты точек внешнего кольца: $(cos theta, sin theta)$. `column_stack` — объединение двух столбцов в матрицу $(n_"outer" times 2)$.],

  [44], [`    inner = factor * np.column_stack([np.cos(...), np.sin(...)])`],
  [Внутреннее кольцо — те же формулы, но умноженные на `factor` (масштабирование радиуса).],

  [45], [(пустая строка)],
  [Разделитель.],

  [46], [`    X = np.vstack([outer, inner])`],
  [Склеиваем внешние и внутренние точки по вертикали → матрица $(n times 2)$.],

  [47], [`    y = np.concatenate([np.zeros(n_outer, dtype=int), np.ones(n_inner, dtype=int)])`],
  [Метки: 0 для внешнего кольца, 1 для внутреннего.],

  [48], [`    if noise > 0:`],
  [Если шум задан...],

  [49], [`        X += rng.normal(0, noise, X.shape)`],
  [...добавляем гауссов шум к каждой координате. `X.shape` гарантирует, что размер шума совпадает с размером данных.],

  [50], [`    return X, y`],
  [Возвращаем матрицу признаков $X$ и вектор меток $y$.],
)

== Строки 53–66: make_xor()

#table(
  columns: (auto, 1fr, 1fr),
  align: (center, left, left),
  table.header[*Строка*][*Код*][*Пояснение*],

  [51], [(пустая строка)],
  [Разделитель.],

  [52], [(пустая строка)],
  [Дополнительный разделитель.],

  [53], [`def make_xor(n=400, noise=0.55, scale=2.2, seed=SEED):`],
  [Генерирует XOR-подобный датасет — четыре облака в углах квадрата, противоположные пары одного класса.],

  [54], [`    rng = np.random.default_rng(seed)`],
  [Генератор.],

  [55], [`    centers = np.array([[scale, scale], [scale, -scale], [-scale, scale], [-scale, -scale]])`],
  [Четыре центра облаков. При `scale=2.2`: $(2.2, 2.2)$, $(2.2, -2.2)$, $(-2.2, 2.2)$, $(-2.2, -2.2)$.],

  [56], [`    per = n // 4`],
  [Базовое число точек на облако.],

  [57], [`    rem = n % 4`],
  [Остаток — распределяется по первым `rem` облакам.],

  [58], [`    X_parts, y_parts = [], []`],
  [Списки-аккумуляторы для частей X и y.],

  [59], [`    for i, c in enumerate(centers):`],
  [Перебираем центры с индексами.],

  [60], [`        cnt = per + (1 if i < rem else 0)`],
  [Число точек для данного облака: `per`, возможно +1 из остатка.],

  [61], [`        X_parts.append(c + noise * rng.standard_normal((cnt, 2)))`],
  [Генерируем `cnt` точек: центр + шум. `standard_normal` — $cal(N)(0; 1)$, масштабирован `noise`.],

  [62], [`        y_parts.append(np.full(cnt, 0 if i in (0, 3) else 1, dtype=int))`],
  [Метка: облака 0 и 3 (противоположные углы) → класс 0; облака 1 и 2 → класс 1. Это даёт XOR-структуру.],

  [63], [`    X = np.vstack(X_parts)`],
  [Склеиваем все части в одну матрицу.],

  [64], [`    y = np.concatenate(y_parts)`],
  [Склеиваем метки.],

  [65], [`    perm = rng.permutation(len(X))`],
  [Случайная перестановка индексов — перемешиваем данные.],

  [66], [`    return X[perm], y[perm]`],
  [Возвращаем перемешанные X и y.],
)

== Строки 69–86: make_blobs()

#table(
  columns: (auto, 1fr, 1fr),
  align: (center, left, left),
  table.header[*Строка*][*Код*][*Пояснение*],

  [67], [(пустая строка)],
  [Разделитель.],

  [68], [(пустая строка)],
  [Дополнительный разделитель.],

  [69], [`def make_blobs(n=400, centers=None, cluster_std=0.9, seed=SEED):`],
  [Генерирует два гауссовых облака. Самый простой для классификации датасет.],

  [70], [`    rng = np.random.default_rng(seed)`],
  [Генератор.],

  [71], [`    if centers is None:`],
  [Если центры не переданы...],

  [72], [`        centers = np.array([[-2.5, -2.0], [2.3, 2.0]], dtype=float)`],
  [...используем центры по умолчанию: $(-2.5, -2.0)$ и $(2.3, 2.0)$.],

  [73], [`    else:`],
  [Иначе...],

  [74], [`        centers = np.asarray(centers, dtype=float)`],
  [...приводим переданный список к NumPy-массиву.],

  [75], [`    n_cls = len(centers)`],
  [Число классов = число центров (у нас 2).],

  [76], [`    base = n // n_cls`],
  [Базовое число точек на класс.],

  [77], [`    rem = n % n_cls`],
  [Остаток.],

  [78], [`    X_parts, y_parts = [], []`],
  [Аккумуляторы.],

  [79], [`    for k, c in enumerate(centers):`],
  [Для каждого центра с индексом `k`...],

  [80], [`        cnt = base + (1 if k < rem else 0)`],
  [Число точек для данного облака.],

  [81], [`        X_parts.append(c + cluster_std * rng.standard_normal((cnt, 2)))`],
  [Точки = центр + гауссов шум с std = `cluster_std`.],

  [82], [`        y_parts.append(np.full(cnt, k, dtype=int))`],
  [Метка = индекс центра (`0` или `1`).],

  [83], [`    X = np.vstack(X_parts)`],
  [Склейка.],

  [84], [`    y = np.concatenate(y_parts)`],
  [Склейка меток.],

  [85], [`    perm = rng.permutation(len(X))`],
  [Перемешивание.],

  [86], [`    return X[perm], y[perm]`],
  [Возврат перемешанных данных.],
)

== Строки 88–103: make_spiral()

#table(
  columns: (auto, 1fr, 1fr),
  align: (center, left, left),
  table.header[*Строка*][*Код*][*Пояснение*],

  [87], [(пустая строка)],
  [Разделитель.],

  [88], [`def make_spiral(n=400, turns=1.5, radius=0.08, sweep=0.28, noise=0.08, seed=SEED):`],
  [Генерирует двуспиральный датасет. `turns` — сколько оборотов, `sweep` — скорость расширения спирали, `radius` — начальный радиус.],

  [89], [`    rng = np.random.default_rng(seed)`],
  [Генератор.],

  [90], [`    n0 = n // 2`],
  [Точек для спирали 0.],

  [91], [`    n1 = n - n0`],
  [Точек для спирали 1.],

  [92], [`    theta0 = np.linspace(0, 2*np.pi*turns, n0)`],
  [Углы от 0 до $2 pi dot "turns"$ для первой спирали. При `turns=1.5` → от 0 до $3 pi$.],

  [93], [`    theta1 = np.linspace(0, 2*np.pi*turns, n1)`],
  [То же для второй спирали.],

  [94], [`    r0 = radius + sweep * theta0`],
  [Радиус первой спирали растёт линейно с углом: $r = 0.08 + 0.28 dot theta$. Это спираль Архимеда.],

  [95], [`    r1 = radius + sweep * theta1`],
  [Радиус второй спирали (те же параметры).],

  [96], [`    X0 = np.column_stack([r0 * np.cos(theta0), r0 * np.sin(theta0)])`],
  [Полярные → декартовы координаты: $x = r cos theta$, $y = r sin theta$.],

  [97], [`    X1 = np.column_stack([r1*np.cos(theta1+np.pi), r1*np.sin(theta1+np.pi)])`],
  [Вторая спираль сдвинута на $pi$ (180°): она закручена зеркально.],

  [98], [`    X = np.vstack([X0, X1])`],
  [Склейка спиралей.],

  [99], [`    y = np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])`],
  [Метки: первая спираль — 0, вторая — 1.],

  [100], [`    if noise > 0:`],
  [Если шум задан...],

  [101], [`        X += rng.normal(0, noise, X.shape)`],
  [...добавляем гауссов шум.],

  [102], [`    perm = rng.permutation(len(X))`],
  [Перемешиваем.],

  [103], [`    return X[perm], y[perm]`],
  [Возвращаем перемешанные данные.],
)

== Строки 105–110: CLS_DATASETS

#table(
  columns: (auto, 1fr, 1fr),
  align: (center, left, left),
  table.header[*Строка*][*Код*][*Пояснение*],

  [104], [(пустая строка)],
  [Разделитель.],

  [105], [`CLS_DATASETS = [`],
  [Начало списка пар (название, функция-генератор). Используется в GUI.],

  [106], [`    ('Circles', make_circles),`],
  [Пара: имя датасета и ссылка на функцию.],

  [107], [`    ('XOR',     make_xor),`],
  [Аналогично для XOR.],

  [108], [`    ('Blobs',   make_blobs),`],
  [Аналогично для Blobs.],

  [109], [`    ('Spiral',  make_spiral),`],
  [Аналогично для Spiral.],

  [110], [`]`],
  [Конец списка.],
)

== Строки 112–116: train_test_split()

#table(
  columns: (auto, 1fr, 1fr),
  align: (center, left, left),
  table.header[*Строка*][*Код*][*Пояснение*],

  [111], [(пустая строка)],
  [Разделитель.],

  [112], [`def train_test_split(X, y, test_ratio=0.25, seed=SEED):`],
  [Разделяет выборку на обучающую и тестовую. По умолчанию 75%/25%.],

  [113], [`    rng = np.random.default_rng(seed)`],
  [Генератор для перемешивания.],

  [114], [`    idx = rng.permutation(len(X))`],
  [Случайная перестановка индексов $[0, ..., n-1]$.],

  [115], [`    cut = int(len(X) * (1 - test_ratio))`],
  [Индекс разреза: первые `cut` — обучение, остальные — тест. При 280 объектах и 0.25: $"cut" = 210$.],

  [116], [`    return X[idx[:cut]], X[idx[cut:]], y[idx[:cut]], y[idx[cut:]]`],
  [Возвращаем `X_train, X_test, y_train, y_test`. Индексация через массив `idx` — элегантная перестановка без копий.],
)

== Строки 118–121: standardize()

#table(
  columns: (auto, 1fr, 1fr),
  align: (center, left, left),
  table.header[*Строка*][*Код*][*Пояснение*],

  [117], [(пустая строка)],
  [Разделитель.],

  [118], [`def standardize(X_train, X_test):`],
  [Z-стандартизация (нормализация) признаков. Статистики считаются *только по train*.],

  [119], [`    mean = X_train.mean(axis=0, keepdims=True)`],
  [Среднее по каждому столбцу (признаку). `keepdims=True` сохраняет размерность $(1, d)$ для broadcasting.],

  [120], [`    std = X_train.std(axis=0, keepdims=True) + 1e-8`],
  [Стандартное отклонение + $10^(-8)$, чтобы не делить на 0.],

  [121], [`    return (X_train - mean) / std, (X_test - mean) / std, mean, std`],
  [Стандартизуем train и test *одними и теми же* mean/std. Возвращаем также mean и std для обратного преобразования.],
)

== Строки 123–126: standardize_targets()

#table(
  columns: (auto, 1fr, 1fr),
  align: (center, left, left),
  table.header[*Строка*][*Код*][*Пояснение*],

  [122], [(пустая строка)],
  [Разделитель.],

  [123], [`def standardize_targets(y_train, y_test):`],
  [Аналогичная Z-стандартизация, но для целевых значений (y). Нужна только в задаче регрессии.],

  [124], [`    mean = y_train.mean(axis=0, keepdims=True)`],
  [Среднее таргетов по train.],

  [125], [`    std = y_train.std(axis=0, keepdims=True) + 1e-8`],
  [Std таргетов + защита от деления на 0.],

  [126], [`    return (y_train - mean) / std, (y_test - mean) / std, mean, std`],
  [Возвращаем стандартизованные таргеты и статистики для обратного преобразования при предсказании.],
)

#pagebreak()

// ══════════════════════════════════════════════════════════
= mlp.py — многослойный перцептрон
// ══════════════════════════════════════════════════════════

Файл содержит ~120 строк. Реализует построение, обучение и метрики MLP на PyTorch.

== Строки 1–9: Импорты и глобальные настройки

#table(
  columns: (auto, 1fr, 1fr),
  align: (center, left, left),
  table.header[*Строка*][*Код*][*Пояснение*],

  [1], [`import copy`],
  [Для глубокого копирования (`copy.deepcopy`) — нужен при сохранении лучшего состояния модели.],

  [2], [`import numpy as np`],
  [NumPy — для работы с массивами при вычислении метрик.],

  [3], [`import torch`],
  [Основной фреймворк глубокого обучения.],

  [4], [`import torch.nn as nn`],
  [Модуль нейронных слоёв (`Linear`, `Sequential`, функции потерь, активации).],

  [5], [(пустая строка)],
  [Разделитель.],

  [6], [`from config import SEED`],
  [Импорт seed для фиксации случайности.],

  [7], [(пустая строка)],
  [Разделитель.],

  [8], [`torch.manual_seed(SEED)`],
  [Фиксируем seed PyTorch — все инициализации весов будут воспроизводимы.],

  [9], [`device = torch.device('cpu')`],
  [Выбираем устройство — CPU. Можно заменить на `'cuda'` для GPU, но здесь это не нужно.],
)

== Строки 11–15: Словарь активаций

#table(
  columns: (auto, 1fr, 1fr),
  align: (center, left, left),
  table.header[*Строка*][*Код*][*Пояснение*],

  [10], [(пустая строка)],
  [Разделитель.],

  [11], [`ACTIVATIONS = {`],
  [Словарь: строковое имя → класс активации PyTorch. Позволяет строить модель по имени.],

  [12], [`    'sigmoid': nn.Sigmoid,`],
  [$sigma(x) = 1/(1 + e^(-x))$. Гладкая, выходы в $(0, 1)$, склонна к «затуханию градиента».],

  [13], [`    'tanh': nn.Tanh,`],
  [$tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))$. Выходы в $(-1, 1)$, центрирована — лучше sigmoid.],

  [14], [`    'relu': nn.ReLU,`],
  [$"ReLU"(x) = max(0, x)$. Быстрая, не затухает для $x > 0$, но «мёртвые нейроны» при $x < 0$.],

  [15], [`}`],
  [Конец словаря.],
)

== Строки 17–18: to_tensor()

#table(
  columns: (auto, 1fr, 1fr),
  align: (center, left, left),
  table.header[*Строка*][*Код*][*Пояснение*],

  [16], [(пустая строка)],
  [Разделитель.],

  [17], [`def to_tensor(array, dtype=torch.float32):`],
  [Вспомогательная функция: NumPy-массив → PyTorch-тензор.],

  [18], [`    return torch.tensor(np.asarray(array), dtype=dtype, device=device)`],
  [`np.asarray` гарантирует NumPy-формат; `torch.tensor` создаёт тензор указанного типа на указанном устройстве.],
)

== Строки 20–31: build_mlp()

#table(
  columns: (auto, 1fr, 1fr),
  align: (center, left, left),
  table.header[*Строка*][*Код*][*Пояснение*],

  [19], [(пустая строка)],
  [Разделитель.],

  [20], [`def build_mlp(input_dim, output_dim, hidden_layers, hidden_dim, activation):`],
  [Строит MLP. `input_dim`/`output_dim` — размерности входа/выхода; `hidden_layers` — число скрытых слоёв; `hidden_dim` — нейронов в каждом; `activation` — строка.],

  [21], [`    layers = []`],
  [Список, в который будем последовательно добавлять слои.],

  [22], [`    in_f = input_dim`],
  [Текущая входная размерность — начинаем с `input_dim`.],

  [23], [`    act_cls = ACTIVATIONS[activation]`],
  [Получаем класс активации из словаря по имени (например, `nn.ReLU`).],

  [24], [(пустая строка)],
  [Разделитель.],

  [25], [`    for _ in range(hidden_layers):`],
  [Цикл по числу скрытых слоёв.],

  [26], [`        layers.append(nn.Linear(in_f, hidden_dim))`],
  [Добавляем линейный слой $y = W x + b$ с `in_f` входами и `hidden_dim` выходами.],

  [27], [`        layers.append(act_cls())`],
  [Добавляем функцию активации (создаём экземпляр класса).],

  [28], [`        in_f = hidden_dim`],
  [Обновляем входную размерность для следующего слоя.],

  [29], [(пустая строка)],
  [Разделитель.],

  [30], [`    layers.append(nn.Linear(in_f, output_dim))`],
  [Добавляем выходной линейный слой (без активации — «сырые» значения).],

  [31], [`    return nn.Sequential(*layers).to(device)`],
  [`nn.Sequential` объединяет все слои в единый модуль. `*layers` — распаковка списка. `.to(device)` — перенос на CPU.],
)

== Строки 33–63: train_model() — обучение с early stopping

#table(
  columns: (auto, 1fr, 1fr),
  align: (center, left, left),
  table.header[*Строка*][*Код*][*Пояснение*],

  [32], [(пустая строка)],
  [Разделитель.],

  [33], [`def train_model(model, X_train, y_train, X_val, y_val, task='regression', epochs=300, lr=0.01):`],
  [Основная функция обучения. `task` определяет функцию потерь. Возвращает историю потерь.],

  [34], [`    loss_fn = nn.MSELoss() if task == 'regression' else nn.CrossEntropyLoss()`],
  [Регрессия → MSE ($1/n sum (hat(y)_i - y_i)^2$); классификация → кросс-энтропия ($-sum y_k log hat(p)_k$).],

  [35], [`    optimizer = torch.optim.Adam(model.parameters(), lr=lr)`],
  [Оптимизатор Adam: адаптивный learning rate для каждого параметра. `model.parameters()` — все веса и сдвиги.],

  [36], [(пустая строка)],
  [Разделитель.],

  [37], [`    history = {'train_loss': [], 'val_loss': []}`],
  [Словарь для записи потерь на каждой эпохе — для графиков.],

  [38], [`    best_state = None`],
  [Будем хранить лучшее состояние модели.],

  [39], [`    best_val = float('inf')`],
  [Инициализируем лучший val loss бесконечностью — любое реальное значение будет меньше.],

  [40], [(пустая строка)],
  [Разделитель.],

  [41], [`    for _ in range(epochs):`],
  [Цикл по эпохам.],

  [42], [`        model.train()`],
  [Переводим модель в режим обучения (влияет на Dropout/BatchNorm, у нас их нет, но это хорошая практика).],

  [43], [`        optimizer.zero_grad()`],
  [Обнуляем градиенты всех параметров перед новым шагом.],

  [44], [`        out = model(X_train)`],
  [Прямой проход (forward pass): вычисляем предсказания для обучающих данных.],

  [45], [`        loss = loss_fn(out, y_train)`],
  [Считаем функцию потерь.],

  [46], [`        loss.backward()`],
  [Обратное распространение ошибки (backpropagation) — вычисляем градиенты $partial L / partial w$ для каждого параметра.],

  [47], [`        optimizer.step()`],
  [Обновляем параметры: $w <- w - "lr" dot nabla L$ (с адаптацией Adam).],

  [48], [(пустая строка)],
  [Разделитель.],

  [49], [`        model.eval()`],
  [Переводим модель в режим оценки.],

  [50], [`        with torch.no_grad():`],
  [Отключаем вычисление градиентов — экономим память и время при валидации.],

  [51], [`            val_out = model(X_val)`],
  [Прямой проход на валидационных данных.],

  [52], [`            val_loss = loss_fn(val_out, y_val)`],
  [Считаем валидационную потерю.],

  [53], [(пустая строка)],
  [Разделитель.],

  [54], [`        history['train_loss'].append(float(loss.item()))`],
  [Сохраняем train loss. `.item()` извлекает число из тензора-скаляра.],

  [55], [`        history['val_loss'].append(float(val_loss.item()))`],
  [Сохраняем val loss.],

  [56], [(пустая строка)],
  [Разделитель.],

  [57], [`        if val_loss.item() < best_val:`],
  [Если текущая валидационная потеря лучше рекорда...],

  [58], [`            best_val = float(val_loss.item())`],
  [...обновляем рекорд.],

  [59], [`            best_state = copy.deepcopy(model.state_dict())`],
  [...сохраняем deep copy всех весов. `state_dict()` — словарь \{имя_параметра: тензор\}. Это *early stopping*.],

  [60], [(пустая строка)],
  [Разделитель.],

  [61], [`    if best_state is not None:`],
  [Если в ходе обучения мы нашли хотя бы одно лучшее состояние...],

  [62], [`        model.load_state_dict(best_state)`],
  [...загружаем лучшие веса обратно в модель. Это и есть early stopping — возвращаемся к лучшей эпохе.],

  [63], [`    return history`],
  [Возвращаем историю для построения графиков.],
)

== Строки 65–86: train_no_early_stop()

#table(
  columns: (auto, 1fr, 1fr),
  align: (center, left, left),
  table.header[*Строка*][*Код*][*Пояснение*],

  [64], [(пустая строка)],
  [Разделитель.],

  [65], [`def train_no_early_stop(model, X_train, y_train, X_val, y_val, epochs=2500, lr=0.01):`],
  [Обучение *без* early stopping — для демонстрации переобучения. Модель обучается все `epochs` эпох без возврата.],

  [66], [`    loss_fn = nn.MSELoss()`],
  [Используем MSE (функция создана для регрессии в демо переобучения).],

  [67], [`    optimizer = torch.optim.Adam(model.parameters(), lr=lr)`],
  [Adam-оптимизатор.],

  [68], [`    history = {'train_loss': [], 'val_loss': []}`],
  [Словарь для истории.],

  [69], [(пустая строка)],
  [Разделитель.],

  [70], [`    for _ in range(epochs):`],
  [Цикл по эпохам.],

  [71], [`        model.train()`],
  [Режим обучения.],

  [72], [`        optimizer.zero_grad()`],
  [Обнуляем градиенты.],

  [73], [`        out = model(X_train)`],
  [Прямой проход.],

  [74], [`        loss = loss_fn(out, y_train)`],
  [Расчёт потерь.],

  [75], [`        loss.backward()`],
  [Backpropagation.],

  [76], [`        optimizer.step()`],
  [Обновление весов.],

  [77], [(пустая строка)],
  [Разделитель.],

  [78], [`        model.eval()`],
  [Режим оценки.],

  [79], [`        with torch.no_grad():`],
  [Без градиентов.],

  [80], [`            val_out = model(X_val)`],
  [Предсказания на val.],

  [81], [`            val_loss = loss_fn(val_out, y_val)`],
  [Val loss.],

  [82], [(пустая строка)],
  [Разделитель.],

  [83], [`        history['train_loss'].append(float(loss.item()))`],
  [Запись train loss.],

  [84], [`        history['val_loss'].append(float(val_loss.item()))`],
  [Запись val loss.],

  [85], [(пустая строка)],
  [Разделитель.],

  [86], [`    return history`],
  [Возвращаем историю. *Без возврата к лучшим весам* — модель в состоянии последней эпохи.],
)

== Строки 88–92: predict_regression()

#table(
  columns: (auto, 1fr, 1fr),
  align: (center, left, left),
  table.header[*Строка*][*Код*][*Пояснение*],

  [87], [(пустая строка)],
  [Разделитель.],

  [88], [`def predict_regression(model, X_t, y_mean, y_std):`],
  [Предсказание для регрессии с *обратной* стандартизацией.],

  [89], [`    model.eval()`],
  [Режим оценки.],

  [90], [`    with torch.no_grad():`],
  [Без градиентов.],

  [91], [`        pred = model(X_t).cpu().numpy()`],
  [Прямой проход → NumPy. `.cpu()` — на всякий случай (у нас CPU). `.numpy()` — конвертация.],

  [92], [`    return pred * y_std + y_mean`],
  [Обратная стандартизация: $hat(y)_"orig" = hat(y)_"std" dot sigma_y + mu_y$. Восстанавливаем исходный масштаб.],
)

== Строки 94–98: predict_classes()

#table(
  columns: (auto, 1fr, 1fr),
  align: (center, left, left),
  table.header[*Строка*][*Код*][*Пояснение*],

  [93], [(пустая строка)],
  [Разделитель.],

  [94], [`def predict_classes(model, X_t):`],
  [Предсказание классов (0 или 1).],

  [95], [`    model.eval()`],
  [Режим оценки.],

  [96], [`    with torch.no_grad():`],
  [Без градиентов.],

  [97], [`        logits = model(X_t)`],
  [Получаем «сырые» выходы (logits) — два числа на объект (для 2 классов).],

  [98], [`        return torch.argmax(logits, dim=1).cpu().numpy()`],
  [`argmax` по `dim=1` выбирает индекс максимального logit → метка класса. Результат — NumPy-массив.],
)

== Строки 100–101: mse_np()

#table(
  columns: (auto, 1fr, 1fr),
  align: (center, left, left),
  table.header[*Строка*][*Код*][*Пояснение*],

  [99], [(пустая строка)],
  [Разделитель.],

  [100], [`def mse_np(y_true, y_pred):`],
  [Вычисляет MSE на NumPy — для итоговой метрики (не для обучения).],

  [101], [`    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))`],
  [$"MSE" = 1/n sum_(i=1)^n (y_i - hat(y)_i)^2$. `np.asarray` гарантирует NumPy; `float(...)` — скаляр.],
)

== Строки 104–115: classification_metrics()

#table(
  columns: (auto, 1fr, 1fr),
  align: (center, left, left),
  table.header[*Строка*][*Код*][*Пояснение*],

  [102], [(пустая строка)],
  [Разделитель.],

  [103], [(пустая строка)],
  [Дополнительный разделитель.],

  [104], [`def classification_metrics(y_true, y_pred):`],
  [Вычисляет accuracy, precision, recall и confusion matrix.],

  [105], [`    y_true = np.asarray(y_true).ravel()`],
  [Приводим к одномерному NumPy-массиву. `.ravel()` — «разворачивает» в вектор.],

  [106], [`    y_pred = np.asarray(y_pred).ravel()`],
  [Аналогично для предсказаний.],

  [107], [`    tn = int(np.sum((y_true == 0) & (y_pred == 0)))`],
  [True Negatives: сколько объектов класса 0 правильно предсказаны как 0.],

  [108], [`    tp = int(np.sum((y_true == 1) & (y_pred == 1)))`],
  [True Positives: сколько объектов класса 1 правильно предсказаны как 1.],

  [109], [`    fp = int(np.sum((y_true == 0) & (y_pred == 1)))`],
  [False Positives: объекты класса 0, ошибочно предсказанные как 1.],

  [110], [`    fn = int(np.sum((y_true == 1) & (y_pred == 0)))`],
  [False Negatives: объекты класса 1, ошибочно предсказанные как 0.],

  [111], [`    acc = (tp + tn) / max(tp + tn + fp + fn, 1)`],
  [$"Accuracy" = ("TP" + "TN") / N$. `max(..., 1)` — защита от деления на 0.],

  [112], [`    prec = tp / max(tp + fp, 1)`],
  [$"Precision" = "TP" / ("TP" + "FP")$ — какую долю объявленных «класс 1» мы угадали.],

  [113], [`    rec = tp / max(tp + fn, 1)`],
  [$"Recall" = "TP" / ("TP" + "FN")$ — какую долю реальных «класс 1» мы нашли.],

  [114], [`    cm = np.array([[tn, fp], [fn, tp]])`],
  [Confusion matrix $2 times 2$: строки — истинные, столбцы — предсказанные.],

  [115], [`    return {'accuracy': ..., 'precision': ..., 'recall': ..., 'confusion_matrix': cm}`],
  [Возвращаем словарь метрик. `float(...)` — для JSON-совместимости.],
)

#pagebreak()

// ══════════════════════════════════════════════════════════
= block1_main.py — регрессионные эксперименты
// ══════════════════════════════════════════════════════════

Файл содержит *229 строк*. Запускает перебор архитектур MLP для двух наборов данных
(полином, тригонометрия), строит графики, демонстрирует переобучение.

== Строки 1–9: Импорты

#table(
  columns: (auto, 1fr, 1fr),
  align: (center, left, left),
  table.header[*Строка*][*Код*][*Пояснение*],

  [1], [`import copy`],
  [Глубокое копирование — нужно для `copy.deepcopy(model)` при сохранении лучшей модели.],

  [2], [`import itertools`],
  [Для `itertools.product` — декартово произведение (слои × нейроны).],

  [3], [`import os`],
  [Для работы с путями (сохранение изображений).],

  [4], [`import numpy as np`],
  [NumPy.],

  [5], [`import matplotlib.pyplot as plt`],
  [Matplotlib — построение графиков.],

  [6], [(пустая строка)],
  [Разделитель.],

  [7], [`from config import SEED, REG_N, REG_EPS0, REG_EPOCHS, REG_LR, DPI`],
  [Импортируем нужные параметры из `config.py`.],

  [8], [`from generators import (random_polynomial, trig_function, ...)`],
  [Из `generators.py` берём генераторы данных и утилиты: `generate_regression_sample`, `train_test_split`, `standardize`, `standardize_targets`.],

  [9], [`from mlp import (build_mlp, train_model, train_no_early_stop, predict_regression, mse_np, to_tensor)`],
  [Из `mlp.py` берём всё, что связано с нейросетью.],
)

== Строки 11–28: Создание данных и константы

#table(
  columns: (auto, 1fr, 1fr),
  align: (center, left, left),
  table.header[*Строка*][*Код*][*Пояснение*],

  [10], [(пустая строка)],
  [Разделитель.],

  [11], [`poly_f, poly_coeffs = random_polynomial(seed=7)`],
  [Создаём случайный полином с `seed=7`. `poly_f` — функция, `poly_coeffs` — кортеж $(a, b, c, d)$.],

  [12], [`trig_f = trig_function()`],
  [Создаём тригонометрическую функцию $x sin(2 pi x)$.],

  [13], [(пустая строка)],
  [Разделитель.],

  [14], [`REGRESSION_DATASETS = [`],
  [Начало списка словарей — каждый описывает один датасет.],

  [15], [`    {`],
  [Начало словаря для полиномиального датасета.],

  [16], [`        'name': 'Полином + равн. шум',`],
  [Имя датасета для вывода в консоль и заголовки графиков.],

  [17], [`        'func': poly_f,`],
  [Ссылка на функцию-полином (для сравнения с предсказанием на графике).],

  [18], [`        'data': generate_regression_sample(poly_f, REG_N, REG_EPS0, noise_mode='uniform', seed=11),`],
  [Генерируем 280 точек с равномерным шумом $epsilon_0 = 0.18$. `seed=11` — фиксация. Возвращает `(x, y)`.],

  [19], [`    },`],
  [Конец словаря полиномиального датасета.],

  [20], [`    {`],
  [Начало словаря для тригонометрического датасета.],

  [21], [`        'name': 'x·sin(2πx) + норм. шум',`],
  [Имя.],

  [22], [`        'func': trig_f,`],
  [Ссылка на $x sin(2 pi x)$.],

  [23], [`        'data': generate_regression_sample(trig_f, REG_N, 0.12, noise_mode='normal', sigma=0.05, seed=13),`],
  [280 точек, нормальный шум $sigma = 0.05$.],

  [24], [`    },`],
  [Конец словаря.],

  [25], [`]`],
  [Конец списка `REGRESSION_DATASETS`.],

  [26], [(пустая строка)],
  [Разделитель.],

  [27], [`IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')`],
  [Путь к папке `images/` рядом с текущим файлом. `__file__` — путь к `block1_main.py`, `abspath` — полный путь, `dirname` — директория.],

  [28], [`DS_SLUG = {'Полином + равн. шум': 'poly', 'x·sin(2πx) + норм. шум': 'trig'}`],
  [Словарь коротких имён для файлов: `reg_poly_sigmoid.png`, `reg_trig_relu.png` и т.д.],
)

== Строки 30–74: run_regression_experiments()

#table(
  columns: (auto, 1fr, 1fr),
  align: (center, left, left),
  table.header[*Строка*][*Код*][*Пояснение*],

  [29], [(пустая строка)],
  [Разделитель.],

  [30], [`def run_regression_experiments(datasets, activations=(...), epochs=REG_EPOCHS, lr=REG_LR):`],
  [Основная функция перебора. Пробует все комбинации `(датасет × активация × архитектура)`.],

  [31], [`    rows = []`],
  [Список-аккумулятор для *всех* результатов экспериментов.],

  [32], [`    best_models = {}`],
  [Словарь лучших моделей по ключу `(dataset_name, activation)`.],

  [33], [`    archs = list(itertools.product([1,2,3], [1,2,3,4,5]))`],
  [Декартово произведение: 3 варианта скрытых слоёв × 5 вариантов нейронов = 15 архитектур.],

  [34], [(пустая строка)],
  [Разделитель.],

  [35], [`    for ds in datasets:`],
  [Цикл по датасетам (полином, тригонометрия).],

  [36], [`        name = ds['name']`],
  [Извлекаем имя датасета.],

  [37], [`        x, y = ds['data']`],
  [Извлекаем данные: $(x, y)$ — массивы формы $(280, 1)$.],

  [38], [`        func = ds['func']`],
  [Целевая функция (для визуализации).],

  [39], [(пустая строка)],
  [Разделитель.],

  [40], [`        X_tr, X_te, y_tr, y_te = train_test_split(x, y, seed=SEED)`],
  [Разбиение 75/25: 210 train / 70 test.],

  [41], [`        X_tr_s, X_te_s, x_m, x_s = standardize(X_tr, X_te)`],
  [Z-стандартизация X. `x_m`, `x_s` — среднее и std (для обратного преобразования).],

  [42], [`        y_tr_s, y_te_s, y_m, y_s = standardize_targets(y_tr, y_te)`],
  [Z-стандартизация Y. `y_m`, `y_s` — для обратного преобразования предсказаний.],

  [43], [(пустая строка)],
  [Разделитель.],

  [44], [`        X_tr_t = to_tensor(X_tr_s)`],
  [Преобразуем стандартизованные X_train в тензор PyTorch.],

  [45], [`        X_te_t = to_tensor(X_te_s)`],
  [X_test → тензор.],

  [46], [`        y_tr_t = to_tensor(y_tr_s)`],
  [y_train → тензор.],

  [47], [`        y_te_t = to_tensor(y_te_s)`],
  [y_test → тензор.],

  [48], [(пустая строка)],
  [Разделитель.],

  [49], [`        for act in activations:`],
  [Цикл по активациям (sigmoid, tanh, relu).],

  [50], [`            best = None`],
  [Лучшая модель для текущей `(датасет, активация)`.],

  [51], [`            for h_layers, h_dim in archs:`],
  [Перебор: `h_layers` ∈ \{1,2,3\}, `h_dim` ∈ \{1,2,3,4,5\}.],

  [52], [`                model = build_mlp(1, 1, h_layers, h_dim, act)`],
  [Строим MLP: 1 вход, 1 выход, `h_layers` скрытых, `h_dim` нейронов, активация `act`.],

  [53], [`                hist = train_model(model, X_tr_t, y_tr_t, X_te_t, y_te_t, task='regression', ...)`],
  [Обучаем модель с early stopping. Получаем историю потерь.],

  [54], [`                pred = predict_regression(model, X_te_t, y_m, y_s)`],
  [Предсказание на тесте с обратной стандартизацией.],

  [55], [`                test_mse = mse_np(y_te, pred)`],
  [Считаем MSE между *оригинальными* y и предсказанием.],

  [56], [(пустая строка)],
  [Разделитель.],

  [57], [`                rec = {`],
  [Начало словаря-записи для текущего эксперимента.],

  [58], [`                    'dataset': name, 'activation': act,`],
  [Имя датасета и активация.],

  [59], [`                    'linear_layers': h_layers + 1,`],
  [Число *линейных* слоёв = скрытые + 1 (выходной). Если 2 скрытых → 3 линейных.],

  [60], [`                    'hidden_layers': h_layers, 'hidden_dim': h_dim,`],
  [Число скрытых слоёв и нейронов.],

  [61], [`                    'test_mse': test_mse, 'history': hist,`],
  [Метрика и история.],

  [62], [`                    'model': copy.deepcopy(model), 'func': func,`],
  [Deep copy модели и ссылка на целевую функцию.],

  [63], [`                    'x_train': X_tr, 'x_test': X_te,`],
  [Сохраняем данные для визуализации.],

  [64], [`                    'y_train': y_tr, 'y_test': y_te,`],
  [Таргеты.],

  [65], [`                    'x_mean': x_m, 'x_std': x_s,`],
  [Статистики X.],

  [66], [`                    'y_mean': y_m, 'y_std': y_s,`],
  [Статистики Y.],

  [67], [`                }`],
  [Конец записи.],

  [68], [`                rows.append(rec)`],
  [Добавляем запись в общий список.],

  [69], [`                if best is None or test_mse < best['test_mse']:`],
  [Если это первая модель или лучше предыдущей лучшей...],

  [70], [`                    best = rec`],
  [...обновляем лучшую.],

  [71], [(пустая строка)],
  [Разделитель.],

  [72], [`            best_models[(name, act)] = best`],
  [Сохраняем лучшую модель для комбинации (датасет, активация).],

  [73], [(пустая строка)],
  [Разделитель.],

  [74], [`    return rows, best_models`],
  [Возвращаем все результаты и словарь лучших.],
)

== Строки 76–90: print_pivot()

#table(
  columns: (auto, 1fr, 1fr),
  align: (center, left, left),
  table.header[*Строка*][*Код*][*Пояснение*],

  [75], [(пустая строка)],
  [Разделитель.],

  [76], [`def print_pivot(rows, ds_name, act):`],
  [Выводит сводную таблицу MSE в консоль для заданного датасета и активации.],

  [77], [`    subset = [r for r in rows if r['dataset'] == ds_name and r['activation'] == act]`],
  [Фильтруем записи: только нужный датасет + активация.],

  [78], [`    all_layers = sorted({r['linear_layers'] for r in subset})`],
  [Уникальные значения числа слоёв, отсортированные.],

  [79], [`    all_dims = sorted({r['hidden_dim'] for r in subset})`],
  [Уникальные числа нейронов.],

  [80], [(пустая строка)],
  [Разделитель.],

  [81], [`    col_w = 12`],
  [Ширина столбца (12 символов).],

  [82], [`    header = f"{'Слоёв':<8}" + ''.join(f"{'N='+str(d):<{col_w}}" for d in all_dims)`],
  [Формируем заголовок: «Слоёв» + «N=1», «N=2» и т.д.],

  [83], [`    print(header)`],
  [Печатаем заголовок.],

  [84], [`    print('-' * len(header))`],
  [Горизонтальная линия.],

  [85], [`    for nl in all_layers:`],
  [Для каждого числа слоёв...],

  [86], [`        line = f'{nl:<8}'`],
  [...начинаем строку с числа слоёв.],

  [87], [`        for hd in all_dims:`],
  [Для каждого числа нейронов...],

  [88], [`            val = next((r['test_mse'] for r in subset if ...), None)`],
  [...ищем MSE. `next(..., None)` — если не найдено, `None`.],

  [89], [`            line += f'{val:<{col_w}.6f}' if val is not None else f'{"—":<{col_w}}'`],
  [Форматируем: 6 знаков после точки или прочерк.],

  [90], [`        print(line)`],
  [Печатаем строку таблицы.],
)

== Строки 92–127: plot_best_regression()

#table(
  columns: (auto, 1fr, 1fr),
  align: (center, left, left),
  table.header[*Строка*][*Код*][*Пояснение*],

  [91], [(пустая строка)],
  [Разделитель.],

  [92], [`def plot_best_regression(record, save_path=None):`],
  [Строит два графика для лучшей модели: loss-кривые и аппроксимацию.],

  [93], [`    model = record['model']`],
  [Извлекаем обученную модель.],

  [94], [`    func = record['func']`],
  [Целевая функция.],

  [95], [`    x_m, x_s = record['x_mean'], record['x_std']`],
  [Статистики X для стандартизации.],

  [96], [`    y_m, y_s = record['y_mean'], record['y_std']`],
  [Статистики Y для обратной стандартизации.],

  [97], [(пустая строка)],
  [Разделитель.],

  [98], [`    x_dense = np.linspace(-1, 1, 400).reshape(-1, 1)`],
  [400 равномерных точек по x для плавной кривой.],

  [99], [`    x_dense_s = (x_dense - x_m) / x_s`],
  [Стандартизуем плотную сетку теми же статистиками.],

  [100], [`    y_dense = predict_regression(model, to_tensor(x_dense_s), y_m, y_s)`],
  [Предсказание MLP на плотной сетке с обратной стандартизацией.],

  [101], [(пустая строка)],
  [Разделитель.],

  [102], [`    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))`],
  [Фигура с двумя субплотами: 14×4.5 дюймов.],

  [103], [(пустая строка)],
  [Разделитель.],

  [104], [`    axes[0].plot(record['history']['train_loss'], label='train loss')`],
  [Левый график: кривая train loss.],

  [105], [`    axes[0].plot(record['history']['val_loss'], label='test loss')`],
  [Кривая val/test loss.],

  [106], [`    axes[0].set_title(f"Loss | {record['activation']} | ...")`],
  [Заголовок с параметрами модели.],

  [107], [`    axes[0].set_xlabel('epoch')`],
  [Подпись оси X.],

  [108], [`    axes[0].set_ylabel('loss')`],
  [Подпись оси Y.],

  [109], [`    axes[0].legend()`],
  [Легенда.],

  [110], [`    axes[0].grid(True, alpha=0.3)`],
  [Полупрозрачная сетка.],

  [111], [(пустая строка)],
  [Разделитель.],

  [112], [`    axes[1].scatter(record['x_train'][:,0], record['y_train'][:,0], s=16, alpha=0.65, label='train')`],
  [Правый график: точки обучающей выборки. `[:,0]` — берём первый (единственный) столбец.],

  [113], [`    axes[1].scatter(record['x_test'][:,0], record['y_test'][:,0], s=20, alpha=0.8, label='test')`],
  [Точки тестовой выборки.],

  [114], [`    axes[1].plot(x_dense[:,0], func(x_dense[:,0]), color='black', lw=2, label='истинная f(x)')`],
  [Чёрная кривая — истинная функция.],

  [115], [`    axes[1].plot(x_dense[:,0], y_dense[:,0], color='crimson', lw=2, label='MLP')`],
  [Красная кривая — предсказание MLP.],

  [116], [`    axes[1].set_title(f"... | test MSE = {record['test_mse']:.4f}")`],
  [Заголовок с MSE.],

  [117], [`    axes[1].set_xlabel('x')`],
  [Подпись X.],

  [118], [`    axes[1].set_ylabel('y')`],
  [Подпись Y.],

  [119], [`    axes[1].legend()`],
  [Легенда.],

  [120], [`    axes[1].grid(True, alpha=0.3)`],
  [Сетка.],

  [121], [(пустая строка)],
  [Разделитель.],

  [122], [`    plt.tight_layout()`],
  [Автоматическая подгонка отступов.],

  [123], [`    if save_path:`],
  [Если задан путь для сохранения...],

  [124], [`        os.makedirs(os.path.dirname(save_path), exist_ok=True)`],
  [...создаём директорию (если не существует). `exist_ok=True` — не ошибка, если уже есть.],

  [125], [`        fig.savefig(save_path, dpi=DPI, bbox_inches='tight')`],
  [...сохраняем PNG. `bbox_inches='tight'` — обрезаем лишние поля.],

  [126], [`    plt.show()`],
  [Показываем окно.],

  [127], [`    plt.close()`],
  [Закрываем фигуру — освобождаем память.],
)

== Строки 129–194: run_overfit_demo()

#table(
  columns: (auto, 1fr, 1fr),
  align: (center, left, left),
  table.header[*Строка*][*Код*][*Пояснение*],

  [128], [(пустая строка)],
  [Разделитель.],

  [129], [`def run_overfit_demo():`],
  [Демонстрация переобучения: маленькая выборка + сложная модель + много эпох.],

  [130], [`    ds = next(d for d in REGRESSION_DATASETS if 'sin' in d['name'])`],
  [Берём тригонометрический датасет (где есть 'sin' в названии).],

  [131], [`    x, y = ds['data']`],
  [Данные.],

  [132], [`    func = ds['func']`],
  [Функция.],

  [133], [(пустая строка)],
  [Разделитель.],

  [134], [`    X_tr_full, X_te, y_tr_full, y_te = train_test_split(x, y, seed=SEED)`],
  [Полное разбиение 75/25.],

  [135], [(пустая строка)],
  [Разделитель.],

  [136], [`    rng = np.random.default_rng(SEED)`],
  [Генератор для выбора подмножества.],

  [137], [`    small_idx = rng.choice(len(X_tr_full), size=10, replace=False)`],
  [Выбираем *10* случайных объектов из train. `replace=False` — без повторений.],

  [138], [`    X_tr = X_tr_full[small_idx]`],
  [Маленький train X (10 точек).],

  [139], [`    y_tr = y_tr_full[small_idx]`],
  [Маленький train y.],

  [140], [(пустая строка)],
  [Разделитель.],

  [141], [`    X_tr_s, X_te_s, x_m, x_s = standardize(X_tr, X_te)`],
  [Стандартизация (по 10 точкам — очень нестабильные статистики).],

  [142], [`    y_tr_s, y_te_s, y_m, y_s = standardize_targets(y_tr, y_te)`],
  [Стандартизация Y.],

  [143], [(пустая строка)],
  [Разделитель.],

  [144], [`    X_tr_t = to_tensor(X_tr_s)`],
  [Тензор X_train.],

  [145], [`    X_te_t = to_tensor(X_te_s)`],
  [Тензор X_test.],

  [146], [`    y_tr_t = to_tensor(y_tr_s)`],
  [Тензор y_train.],

  [147], [`    y_te_t = to_tensor(y_te_s)`],
  [Тензор y_test.],

  [148], [(пустая строка)],
  [Разделитель.],

  [149], [`    model = build_mlp(1, 1, hidden_layers=8, hidden_dim=5, activation='tanh')`],
  [Строим *избыточно сложную* модель: 8 скрытых слоёв × 5 нейронов = 9 линейных слоёв. Для 10 точек — явный перебор.],

  [150], [`    hist = train_no_early_stop(model, X_tr_t, y_tr_t, X_te_t, y_te_t, epochs=5000, lr=0.01)`],
  [Обучаем 5000 эпох *без* early stopping — нарочно даём переобучиться.],

  [151], [(пустая строка)],
  [Разделитель.],

  [152], [`    pred_tr = predict_regression(model, X_tr_t, y_m, y_s)`],
  [Предсказания на train.],

  [153], [`    pred_te = predict_regression(model, X_te_t, y_m, y_s)`],
  [Предсказания на test.],

  [154], [`    tr_mse = mse_np(y_tr, pred_tr)`],
  [Train MSE — ожидается *очень низкий* (модель «запомнила» 10 точек).],

  [155], [`    te_mse = mse_np(y_te, pred_te)`],
  [Test MSE — ожидается *высокий* ($"Test" >> "Train"$) — признак переобучения.],

  [156], [(пустая строка)],
  [Разделитель.],

  [157], [`    best_epoch = int(np.argmin(hist['val_loss'])) + 1`],
  [`argmin` находит индекс минимального val loss. `+1` — нумерация эпох с 1.],

  [158], [(пустая строка)],
  [Разделитель.],

  [159], [`    print(f'\\nПереобучение на x·sin(2πx)')`],
  [Заголовок вывода.],

  [160], [`    print(f'  Train объектов: {len(X_tr)}')`],
  [Выводим: 10.],

  [161], [`    print(f'  Train MSE: {tr_mse:.6f}')`],
  [Train MSE с 6 знаками.],

  [162], [`    print(f'  Test  MSE: {te_mse:.6f}')`],
  [Test MSE с 6 знаками.],

  [163], [`    print(f'  Лучшая эпоха по val loss: {best_epoch}')`],
  [Эпоха, когда val loss был минимален.],

  [164], [(пустая строка)],
  [Разделитель.],

  [165], [`    x_dense = np.linspace(-1, 1, 500).reshape(-1, 1)`],
  [500 точек для плавной кривой.],

  [166], [`    x_dense_s = (x_dense - x_m) / x_s`],
  [Стандартизация.],

  [167], [`    y_dense = predict_regression(model, to_tensor(x_dense_s), y_m, y_s)`],
  [Предсказание MLP на плотной сетке.],

  [168], [(пустая строка)],
  [Разделитель.],

  [169], [`    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))`],
  [Два графика.],

  [170], [(пустая строка)],
  [Разделитель.],

  [171], [`    axes[0].plot(hist['train_loss'], label='train loss')`],
  [Кривая train loss.],

  [172], [`    axes[0].plot(hist['val_loss'], label='test loss')`],
  [Кривая test loss — расходится после минимума.],

  [173], [`    axes[0].axvline(best_epoch - 1, color='crimson', ls='--', label='минимум test loss')`],
  [Вертикальная красная пунктирная линия в точке минимума val loss.],

  [174], [`    axes[0].set_title('Переобучение: train / test loss')`],
  [Заголовок.],

  [175], [`    axes[0].set_xlabel('epoch')`],
  [Подпись X.],

  [176], [`    axes[0].set_ylabel('loss')`],
  [Подпись Y.],

  [177], [`    axes[0].legend()`],
  [Легенда.],

  [178], [`    axes[0].grid(True, alpha=0.3)`],
  [Сетка.],

  [179], [(пустая строка)],
  [Разделитель.],

  [180], [`    axes[1].scatter(X_tr[:,0], y_tr[:,0], s=28, alpha=0.85, label='train')`],
  [10 обучающих точек (крупные).],

  [181], [`    axes[1].scatter(X_te[:,0], y_te[:,0], s=24, alpha=0.65, label='test')`],
  [Тестовые точки.],

  [182], [`    axes[1].plot(x_dense[:,0], func(x_dense[:,0]), color='black', lw=2, label='истинная f(x)')`],
  [Чёрная кривая — истинная функция.],

  [183], [`    axes[1].plot(x_dense[:,0], y_dense[:,0], color='crimson', lw=2, label='MLP (переобучение)')`],
  [Красная кривая — модель, которая «извивается» через 10 точек вместо правильной формы.],

  [184], [`    axes[1].set_title('Переобученная аппроксимация x·sin(2πx)')`],
  [Заголовок.],

  [185], [`    axes[1].set_xlabel('x')`],
  [Подпись X.],

  [186], [`    axes[1].set_ylabel('y')`],
  [Подпись Y.],

  [187], [`    axes[1].legend()`],
  [Легенда.],

  [188], [`    axes[1].grid(True, alpha=0.3)`],
  [Сетка.],

  [189], [(пустая строка)],
  [Разделитель.],

  [190], [`    plt.tight_layout()`],
  [Подгонка.],

  [191], [`    os.makedirs(IMG_DIR, exist_ok=True)`],
  [Создаём папку images.],

  [192], [`    fig.savefig(os.path.join(IMG_DIR, 'overfit.png'), dpi=DPI, bbox_inches='tight')`],
  [Сохраняем как `overfit.png`.],

  [193], [`    plt.show()`],
  [Показываем.],

  [194], [`    plt.close()`],
  [Закрываем.],
)

== Строки 196–229: main()

#table(
  columns: (auto, 1fr, 1fr),
  align: (center, left, left),
  table.header[*Строка*][*Код*][*Пояснение*],

  [195], [(пустая строка)],
  [Разделитель.],

  [196], [`def main():`],
  [Точка входа.],

  [197], [`    print(f'Коэффициенты полинома: {poly_coeffs}')`],
  [Выводим коэффициенты $(a, b, c, d)$ — для записи в отчёт.],

  [198], [`    print(f'Запуск экспериментов по регрессии (epochs={REG_EPOCHS}, lr={REG_LR})…')`],
  [Информационное сообщение.],

  [199], [(пустая строка)],
  [Разделитель.],

  [200], [`    rows, best = run_regression_experiments(REGRESSION_DATASETS)`],
  [Запуск всех 90 экспериментов (2 датасета × 3 активации × 15 архитектур).],

  [201], [(пустая строка)],
  [Разделитель.],

  [202], [`    activations = ['sigmoid', 'tanh', 'relu']`],
  [Список активаций для вывода.],

  [203], [`    ds_names = [d['name'] for d in REGRESSION_DATASETS]`],
  [Список имён датасетов.],

  [204], [(пустая строка)],
  [Разделитель.],

  [205], [`    for ds_name in ds_names:`],
  [Для каждого датасета...],

  [206], [`        print('\\n' + '=' * 70)`],
  [Разделительная линия из 70 символов `=`.],

  [207], [`        print(ds_name)`],
  [Имя датасета.],

  [208], [`        print('=' * 70)`],
  [Ещё линия.],

  [209], [`        for act in activations:`],
  [Для каждой активации...],

  [210], [`            print(f'\\n  Активация: {act}  (Test MSE)')`],
  [Подзаголовок.],

  [211], [`            print('  ' + '-' * 62)`],
  [Линия.],

  [212], [`            print_pivot(rows, ds_name, act)`],
  [Печатаем сводную таблицу MSE.],

  [213], [(пустая строка)],
  [Разделитель.],

  [214], [`    print('\\n\\n── Лучшие модели по регрессии ──')`],
  [Заголовок раздела.],

  [215], [`    for (ds_name, act), rec in best.items():`],
  [Перебираем лучшие модели по ключу (датасет, активация).],

  [216], [`        print(f'  {ds_name} | {act}: слоёв={rec["linear_layers"]}, нейронов={rec["hidden_dim"]}, MSE={rec["test_mse"]:.6f}')`],
  [Выводим характеристики лучшей модели.],

  [217–218], [(`slug = ...`, `plot_best_regression(rec, ...)`)],
  [Получаем короткое имя и строим/сохраняем графики лучшей модели.],

  [219], [(пустая строка)],
  [Разделитель.],

  [220], [`        slug = DS_SLUG.get(ds_name, ds_name)`],
  [Получаем `'poly'` или `'trig'` для имени файла.],

  [221], [`        plot_best_regression(rec, save_path=os.path.join(IMG_DIR, f'reg_{slug}_{act}.png'))`],
  [Строим и сохраняем: `reg_poly_sigmoid.png`, `reg_trig_relu.png` и т.д.],

  [222], [(пустая строка)],
  [Разделитель.],

  [223], [`    print('\\n── Демонстрация переобучения ──')`],
  [Заголовок.],

  [224], [`    run_overfit_demo()`],
  [Запуск демо переобучения.],

  [225], [(пустая строка)],
  [Разделитель.],

  [226], [(пустая строка)],
  [Дополнительный разделитель.],

  [227], [`if __name__ == '__main__':`],
  [Стандартная проверка: файл запущен как скрипт (а не импортирован)?],

  [228], [`    main()`],
  [Вызываем `main()`.],
)

#pagebreak()

// ══════════════════════════════════════════════════════════
= block2_main.py — классификационные эксперименты
// ══════════════════════════════════════════════════════════

Файл содержит *310 строк*. Запускает классификацию, визуализирует границы решений,
ищет минимальный размер выборки и проводит кросс-валидацию.

== Строки 1–10: Импорты

#table(
  columns: (auto, 1fr, 1fr),
  align: (center, left, left),
  table.header[*Строка*][*Код*][*Пояснение*],

  [1], [`import copy`],
  [Глубокое копирование моделей.],

  [2], [`import itertools`],
  [Декартово произведение архитектур.],

  [3], [`import os`],
  [Работа с путями.],

  [4], [`import numpy as np`],
  [NumPy.],

  [5], [`import matplotlib.pyplot as plt`],
  [Matplotlib.],

  [6], [`import torch`],
  [PyTorch — нужен для `torch.long` при создании меток.],

  [7], [(пустая строка)],
  [Разделитель.],

  [8], [`from config import (SEED, CLS_N, CLS_NOISE, CLS_EPOCHS, CLS_LR, CV_K, DPI)`],
  [Классификационные параметры из конфига.],

  [9], [`from generators import (CLS_DATASETS, make_circles, make_xor, make_blobs, make_spiral, train_test_split, standardize)`],
  [Генераторы и утилиты.],

  [10], [`from mlp import (build_mlp, train_model, predict_classes, classification_metrics, to_tensor)`],
  [Функции MLP для классификации.],
)

== Строки 12–24: load_cls_datasets()

#table(
  columns: (auto, 1fr, 1fr),
  align: (center, left, left),
  table.header[*Строка*][*Код*][*Пояснение*],

  [11], [(пустая строка)],
  [Разделитель.],

  [12], [`IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')`],
  [Путь к папке `images/`.],

  [13], [(пустая строка)],
  [Разделитель.],

  [14], [`def load_cls_datasets(n=CLS_N, noise=CLS_NOISE):`],
  [Создаёт список из 4 классификационных датасетов.],

  [15], [`    return [`],
  [Начало списка.],

  [16], [`        {'name': 'Circles',`],
  [Словарь для Circles.],

  [17], [`         'data': make_circles(n=n, noise=noise, seed=21)},`],
  [500 точек двух концентрических колец с `seed=21`.],

  [18], [`        {'name': 'XOR',`],
  [Словарь для XOR.],

  [19], [`         'data': make_xor(n=n, noise=0.55, seed=22)},`],
  [XOR с `noise=0.55` (4 облака).],

  [20], [`        {'name': 'Blobs',`],
  [Словарь для Blobs.],

  [21], [`         'data': make_blobs(n=n, seed=23)},`],
  [Два гауссовых облака.],

  [22], [`        {'name': 'Spiral',`],
  [Словарь для Spiral.],

  [23], [`         'data': make_spiral(n=n, noise=noise, seed=24)},`],
  [Двуспиральный датасет — самый сложный.],

  [24], [`    ]`],
  [Конец списка.],
)

== Строки 26–68: run_classification_experiments()

#table(
  columns: (auto, 1fr, 1fr),
  align: (center, left, left),
  table.header[*Строка*][*Код*][*Пояснение*],

  [25], [(пустая строка)],
  [Разделитель.],

  [26], [`def run_classification_experiments(datasets, activations=(...), epochs=CLS_EPOCHS, lr=CLS_LR):`],
  [Основной перебор архитектур для классификации.],

  [27], [`    rows = []`],
  [Аккумулятор всех результатов.],

  [28], [`    best_models = {}`],
  [Лучшие модели по ключу (имя, активация).],

  [29], [`    archs = list(itertools.product([1,2,3], [1,2,3,4,5]))`],
  [15 архитектур.],

  [30], [(пустая строка)],
  [Разделитель.],

  [31], [`    for ds in datasets:`],
  [Цикл по 4 датасетам.],

  [32], [`        name = ds['name']`],
  [Имя.],

  [33], [`        X, y = ds['data']`],
  [Данные: $X$ — матрица $(500 times 2)$, $y$ — вектор из 0 и 1.],

  [34], [`        X_tr, X_te, y_tr, y_te = train_test_split(X, y, seed=SEED)`],
  [Разбиение 75/25.],

  [35], [`        X_tr_s, X_te_s, x_m, x_s = standardize(X_tr, X_te)`],
  [Z-стандартизация *признаков*. Метки не стандартизуем!],

  [36], [(пустая строка)],
  [Разделитель.],

  [37], [`        X_tr_t = to_tensor(X_tr_s)`],
  [Тензор X_train.],

  [38], [`        X_te_t = to_tensor(X_te_s)`],
  [Тензор X_test.],

  [39], [`        y_tr_t = to_tensor(y_tr, dtype=torch.long)`],
  [Метки → `long` (int64). `CrossEntropyLoss` требует именно `long`.],

  [40], [`        y_te_t = to_tensor(y_te, dtype=torch.long)`],
  [Аналогично для теста.],

  [41], [(пустая строка)],
  [Разделитель.],

  [42], [`        for act in activations:`],
  [Цикл по 3 активациям.],

  [43], [`            best = None`],
  [Лучшая модель для текущей комбинации.],

  [44], [`            for h_layers, h_dim in archs:`],
  [Перебор 15 архитектур.],

  [45], [`                model = build_mlp(2, 2, h_layers, h_dim, act)`],
  [MLP: 2 входа $(x_1, x_2)$, 2 выхода (logits для 2 классов).],

  [46], [`                hist = train_model(model, X_tr_t, y_tr_t, X_te_t, y_te_t, task='classification', ...)`],
  [Обучение с `CrossEntropyLoss` и early stopping.],

  [47], [`                pred = predict_classes(model, X_te_t)`],
  [Предсказание классов на тесте.],

  [48], [`                metrics = classification_metrics(y_te, pred)`],
  [Accuracy, precision, recall, confusion matrix.],

  [49], [(пустая строка)],
  [Разделитель.],

  [50], [`                rec = {`],
  [Начало записи.],

  [51], [`                    'dataset': name, 'activation': act,`],
  [Имя и активация.],

  [52], [`                    'linear_layers': h_layers + 1,`],
  [Число линейных слоёв.],

  [53], [`                    'hidden_layers': h_layers, 'hidden_dim': h_dim,`],
  [Параметры архитектуры.],

  [54], [`                    'test_accuracy': metrics['accuracy'],`],
  [Accuracy — основная метрика.],

  [55], [`                    'history': hist,`],
  [История потерь.],

  [56], [`                    'model': copy.deepcopy(model),`],
  [Копия модели.],

  [57], [`                    'X_train': X_tr, 'X_test': X_te,`],
  [Данные.],

  [58], [`                    'y_train': y_tr, 'y_test': y_te,`],
  [Метки.],

  [59], [`                    'x_mean': x_m, 'x_std': x_s,`],
  [Статистики стандартизации.],

  [60], [`                    'confusion_matrix': metrics['confusion_matrix'],`],
  [Confusion matrix.],

  [61], [`                }`],
  [Конец записи.],

  [62], [`                rows.append(rec)`],
  [Добавление.],

  [63], [`                if best is None or metrics['accuracy'] > best['test_accuracy']:`],
  [Accuracy *больше* → лучше (в отличие от MSE).],

  [64], [`                    best = rec`],
  [Обновляем лучшую.],

  [65], [(пустая строка)],
  [Разделитель.],

  [66], [`            best_models[(name, act)] = best`],
  [Сохраняем.],

  [67], [(пустая строка)],
  [Разделитель.],

  [68], [`    return rows, best_models`],
  [Возвращаем.],
)

== Строки 70–84: print_cls_pivot()

#table(
  columns: (auto, 1fr, 1fr),
  align: (center, left, left),
  table.header[*Строка*][*Код*][*Пояснение*],

  [69], [(пустая строка)],
  [Разделитель.],

  [70], [`def print_cls_pivot(rows, ds_name, act):`],
  [Аналогична `print_pivot()` из block1, но выводит *accuracy* (4 знака) вместо MSE (6 знаков).],

  [71], [`    subset = [r for r in rows if r['dataset'] == ds_name and r['activation'] == act]`],
  [Фильтрация.],

  [72], [`    all_layers = sorted({r['linear_layers'] for r in subset})`],
  [Уникальные слои.],

  [73], [`    all_dims = sorted({r['hidden_dim'] for r in subset})`],
  [Уникальные нейроны.],

  [74], [(пустая строка)],
  [Разделитель.],

  [75], [`    col_w = 12`],
  [Ширина столбца.],

  [76], [`    header = f"{'Слоёв':<8}" + ''.join(f"{'N='+str(d):<{col_w}}" for d in all_dims)`],
  [Заголовок.],

  [77], [`    print(header)`],
  [Печать.],

  [78], [`    print('-' * len(header))`],
  [Линия.],

  [79], [`    for nl in all_layers:`],
  [Цикл по строкам.],

  [80], [`        line = f'{nl:<8}'`],
  [Начало строки.],

  [81], [`        for hd in all_dims:`],
  [Цикл по столбцам.],

  [82], [`            val = next((r['test_accuracy'] for r in subset if ...), None)`],
  [Ищем accuracy.],

  [83], [`            line += f'{val:<{col_w}.4f}' if val is not None else f'{"—":<{col_w}}'`],
  [4 знака после точки.],

  [84], [`        print(line)`],
  [Печать строки.],
)

== Строки 86–133: plot_decision_boundary()

#table(
  columns: (auto, 1fr, 1fr),
  align: (center, left, left),
  table.header[*Строка*][*Код*][*Пояснение*],

  [85], [(пустая строка)],
  [Разделитель.],

  [86], [`def plot_decision_boundary(record, save_path=None):`],
  [Строит 3 графика: loss, граница решений, confusion matrix.],

  [87], [`    model = record['model']`],
  [Модель.],

  [88], [`    X_tr, X_te = record['X_train'], record['X_test']`],
  [Данные.],

  [89], [`    y_tr, y_te = record['y_train'], record['y_test']`],
  [Метки.],

  [90], [`    x_m, x_s = record['x_mean'], record['x_std']`],
  [Статистики.],

  [91], [(пустая строка)],
  [Разделитель.],

  [92], [`    X_all = np.vstack([X_tr, X_te])`],
  [Объединяем train + test для определения диапазона осей.],

  [93], [`    lo = X_all.min(axis=0) - 0.5`],
  [Нижняя граница по каждой оси (с отступом 0.5).],

  [94], [`    hi = X_all.max(axis=0) + 0.5`],
  [Верхняя граница.],

  [95], [(пустая строка)],
  [Разделитель.],

  [96], [`    xx, yy = np.meshgrid(np.linspace(lo[0], hi[0], 250), np.linspace(lo[1], hi[1], 250))`],
  [Сетка 250×250 для заливки решающей границы. `meshgrid` создаёт координатные матрицы.],

  [97], [`    grid = np.column_stack([xx.ravel(), yy.ravel()])`],
  [«Разворачиваем» сетку в матрицу $(62500 times 2)$.],

  [98], [`    grid_s = (grid - x_m) / x_s`],
  [Стандартизуем сетку теми же mean/std, что и train.],

  [99], [`    zz = predict_classes(model, to_tensor(grid_s)).reshape(xx.shape)`],
  [Предсказываем классы → возвращаем в форму 250×250 для `contourf`.],

  [100], [(пустая строка)],
  [Разделитель.],

  [101], [`    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))`],
  [Три субплота.],

  [102], [(пустая строка)],
  [Разделитель.],

  [103], [`    axes[0].plot(record['history']['train_loss'], label='train loss')`],
  [Кривая train loss.],

  [104], [`    axes[0].plot(record['history']['val_loss'], label='test loss')`],
  [Кривая test loss.],

  [105], [`    axes[0].set_title(f"Loss | {record['activation']} | слоёв=... | нейронов=...")`],
  [Заголовок.],

  [106], [`    axes[0].set_xlabel('epoch')`],
  [Подпись X.],

  [107], [`    axes[0].set_ylabel('loss')`],
  [Подпись Y.],

  [108], [`    axes[0].legend()`],
  [Легенда.],

  [109], [`    axes[0].grid(True, alpha=0.3)`],
  [Сетка.],

  [110], [(пустая строка)],
  [Разделитель.],

  [111], [`    axes[1].contourf(xx, yy, zz, levels=2, alpha=0.35, cmap='coolwarm')`],
  [*Заливка* решающей границы. `contourf` — заполненный контурный график. `levels=2` — два класса.],

  [112], [`    axes[1].scatter(X_te[y_te==0, 0], X_te[y_te==0, 1], s=18, label='Класс 0')`],
  [Точки класса 0 на тесте.],

  [113], [`    axes[1].scatter(X_te[y_te==1, 0], X_te[y_te==1, 1], s=18, label='Класс 1')`],
  [Точки класса 1.],

  [114], [`    axes[1].set_title(f"... | accuracy = {record['test_accuracy']:.3f}")`],
  [Заголовок с accuracy.],

  [115], [`    axes[1].legend()`],
  [Легенда.],

  [116], [`    axes[1].grid(True, alpha=0.3)`],
  [Сетка.],

  [117], [(пустая строка)],
  [Разделитель.],

  [118], [`    cm = record['confusion_matrix']`],
  [Извлекаем confusion matrix $2 times 2$.],

  [119], [`    axes[2].imshow(cm, cmap='Blues')`],
  [Отображаем как цветную матрицу (тепловая карта).],

  [120], [`    for i in range(2):`],
  [Для каждой строки (0, 1)...],

  [121], [`        for j in range(2):`],
  [...для каждого столбца...],

  [122], [`            axes[2].text(j, i, str(cm[i, j]), ha='center', va='center', color='black', fontsize=13)`],
  [...пишем число (TN/FP/FN/TP) в центре ячейки.],

  [123], [`    axes[2].set_xticks([0, 1])`],
  [Метки оси X: 0 и 1.],

  [124], [`    axes[2].set_yticks([0, 1])`],
  [Метки оси Y.],

  [125], [`    axes[2].set_xlabel('predicted')`],
  [Подпись X.],

  [126], [`    axes[2].set_ylabel('true')`],
  [Подпись Y.],

  [127], [`    axes[2].set_title('Confusion matrix')`],
  [Заголовок.],

  [128], [(пустая строка)],
  [Разделитель.],

  [129], [`    plt.tight_layout()`],
  [Подгонка отступов.],

  [130], [`    if save_path:`],
  [Если задан путь...],

  [131], [`        os.makedirs(os.path.dirname(save_path), exist_ok=True)`],
  [...создаём директорию.],

  [132], [`        fig.savefig(save_path, dpi=DPI, bbox_inches='tight')`],
  [...сохраняем PNG.],

  [133], [`    plt.show()`],
  [Показываем.],

  [134], [`    plt.close()`],
  [Закрываем.],
)

== Строки 136–166: estimate_min_train_size()

#table(
  columns: (auto, 1fr, 1fr),
  align: (center, left, left),
  table.header[*Строка*][*Код*][*Пояснение*],

  [135], [(пустая строка)],
  [Разделитель.],

  [136], [`def estimate_min_train_size(datasets, activation='tanh', linear_layers=4, hidden_dim=5, target_acc=0.90, ...):`],
  [Ищет минимальную долю train, при которой accuracy ≥ 90 %. Фиксированная архитектура: tanh, 4 линейных слоя, 5 нейронов.],

  [137], [`    fracs = np.arange(0.10, 0.96, 0.05)`],
  [Доли train: 0.10, 0.15, 0.20, ..., 0.95 (18 значений).],

  [138], [`    summary = []`],
  [Аккумулятор.],

  [139], [(пустая строка)],
  [Разделитель.],

  [140], [`    for ds in datasets:`],
  [Для каждого из 4 датасетов...],

  [141], [`        name = ds['name']`],
  [Имя.],

  [142], [`        X, y = ds['data']`],
  [Данные.],

  [143], [`        X = np.asarray(X, dtype=np.float32)`],
  [Приведение типа (float32 для PyTorch).],

  [144], [`        y = np.asarray(y, dtype=np.int64)`],
  [int64 для `CrossEntropyLoss`.],

  [145], [(пустая строка)],
  [Разделитель.],

  [146], [`        best_frac = None`],
  [Минимальная доля, пока не найдена.],

  [147], [`        for frac in fracs:`],
  [Перебор от 10 % до 95 %.],

  [148], [`            test_ratio = 1.0 - float(frac)`],
  [`test_ratio` = 1 − `frac`. Например: `frac=0.10` → 90 % тест.],

  [149], [`            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_ratio=test_ratio, seed=SEED)`],
  [Разбиение с заданной долей.],

  [150], [`            X_tr_s, X_te_s, _, _ = standardize(X_tr, X_te)`],
  [Стандартизация. `_, _` — mean и std не нужны дальше.],

  [151], [(пустая строка)],
  [Разделитель.],

  [152], [`            model = build_mlp(2, 2, linear_layers-1, hidden_dim, activation)`],
  [Строим модель. `linear_layers-1` = 3 скрытых слоя.],

  [153], [`            train_model(model, to_tensor(X_tr_s), to_tensor(y_tr, torch.long), to_tensor(X_te_s), to_tensor(y_te, torch.long), task='classification', ...)`],
  [Обучаем классификатор.],

  [154], [(пустая строка)],
  [Разделитель.],

  [155], [`            pred = predict_classes(model, to_tensor(X_te_s))`],
  [Предсказания.],

  [156], [`            acc = classification_metrics(y_te, pred)['accuracy']`],
  [Accuracy.],

  [157], [(пустая строка)],
  [Разделитель.],

  [158], [`            if best_frac is None and acc >= target_acc:`],
  [Если ещё не нашли min *и* accuracy ≥ 0.90...],

  [159], [`                best_frac = float(frac)`],
  [...запоминаем `frac`.],

  [160], [(пустая строка)],
  [Разделитель.],

  [161], [`        summary.append({`],
  [Добавляем результат.],

  [162], [`            'dataset': name, 'activation': activation,`],
  [Датасет и активация.],

  [163], [`            'linear_layers': linear_layers, 'hidden_dim': hidden_dim,`],
  [Архитектура.],

  [164], [`            'target_acc': target_acc,`],
  [Целевая accuracy.],

  [165], [`            'min_frac': best_frac,`],
  [Минимальная доля (или `None` если не найдена).],

  [166], [`            'min_train_size': (None if best_frac is None else int(round(best_frac * len(X)))),`],
  [Абсолютное число объектов: $"frac" times n$.],

  [167], [`        })`],
  [Конец словаря.],

  [168], [(пустая строка)],
  [Разделитель.],

  [169], [`    return summary`],
  [Возвращаем список результатов.],
)

== Строки 171–173: kfold_indices()

#table(
  columns: (auto, 1fr, 1fr),
  align: (center, left, left),
  table.header[*Строка*][*Код*][*Пояснение*],

  [170], [(пустая строка)],
  [Разделитель.],

  [171], [`def kfold_indices(n, k=5, seed=SEED):`],
  [Генерирует $k$ фолдов: списки индексов для кросс-валидации.],

  [172], [`    rng = np.random.default_rng(seed)`],
  [Генератор.],

  [173], [`    idx = rng.permutation(n)`],
  [Перемешиваем индексы $[0, ..., n-1]$.],

  [174], [`    return np.array_split(idx, k)`],
  [`array_split` делит массив на $k$ примерно равных частей. При $n = 500$, $k = 5$ → 5 массивов по 100.],
)

== Строки 176–209: cv_grid_search()

#table(
  columns: (auto, 1fr, 1fr),
  align: (center, left, left),
  table.header[*Строка*][*Код*][*Пояснение*],

  [175], [(пустая строка)],
  [Разделитель.],

  [176], [`def cv_grid_search(X, y, activations=(...), layers_grid=(2,3,4), hidden_dim=5, k=CV_K, ...):`],
  [Grid search с $k$-fold кросс-валидацией. Перебирает (3 активации × 3 числа слоёв = 9 комбинаций).],

  [177], [`    X = np.asarray(X, dtype=np.float32)`],
  [Приведение типа.],

  [178], [`    y = np.asarray(y, dtype=np.int64)`],
  [Приведение типа.],

  [179], [`    folds = kfold_indices(len(X), k=k)`],
  [Генерируем $k$ фолдов.],

  [180], [(пустая строка)],
  [Разделитель.],

  [181], [`    rows = []`],
  [Аккумулятор.],

  [182], [`    best_row = None`],
  [Лучшая комбинация.],

  [183], [(пустая строка)],
  [Разделитель.],

  [184], [`    for act in activations:`],
  [Цикл по активациям.],

  [185], [`        for nl in layers_grid:`],
  [Цикл по числу линейных слоёв (2, 3, 4).],

  [186], [`            fold_accs = []`],
  [Список accuracy для каждого фолда.],

  [187], [`            for fi in range(k):`],
  [Перебор фолдов: `fi` — индекс валидационного фолда.],

  [188], [`                val_idx = folds[fi]`],
  [Индексы валидации — текущий фолд.],

  [189], [`                tr_idx = np.concatenate([folds[j] for j in range(k) if j != fi])`],
  [Индексы обучения — *все остальные* фолды.],

  [190], [(пустая строка)],
  [Разделитель.],

  [191], [`                X_tr, X_val = X[tr_idx], X[val_idx]`],
  [Разделение данных.],

  [192], [`                y_tr, y_val = y[tr_idx], y[val_idx]`],
  [Разделение меток.],

  [193], [(пустая строка)],
  [Разделитель.],

  [194], [`                X_tr_s, X_val_s, _, _ = standardize(X_tr, X_val)`],
  [Стандартизация — *заново для каждого фолда*.],

  [195], [(пустая строка)],
  [Разделитель.],

  [196], [`                model = build_mlp(2, 2, nl-1, hidden_dim, act)`],
  [Строим модель. `nl-1` — скрытых слоёв.],

  [197], [`                train_model(model, to_tensor(X_tr_s), to_tensor(y_tr, torch.long), ...)`],
  [Обучаем.],

  [198], [(пустая строка)],
  [Разделитель.],

  [199], [`                pred = predict_classes(model, to_tensor(X_val_s))`],
  [Предсказания на валидации.],

  [200], [`                fold_accs.append(classification_metrics(y_val, pred)['accuracy'])`],
  [Добавляем accuracy текущего фолда.],

  [201], [(пустая строка)],
  [Разделитель.],

  [202], [`            row = {`],
  [Формируем запись.],

  [203], [`                'activation': act, 'linear_layers': nl,`],
  [Параметры.],

  [204], [`                'hidden_dim': hidden_dim,`],
  [Число нейронов (5).],

  [205], [`                'cv_mean_acc': float(np.mean(fold_accs)),`],
  [Среднее accuracy по $k$ фолдам.],

  [206], [`                'cv_std_acc': (float(np.std(fold_accs, ddof=1)) if len(fold_accs) > 1 else 0.0),`],
  [Std с `ddof=1` — несмещённая оценка (делим на $k - 1$).],

  [207], [`            }`],
  [Конец записи.],

  [208], [`            rows.append(row)`],
  [Добавляем.],

  [209], [`            if best_row is None or row['cv_mean_acc'] > best_row['cv_mean_acc']:`],
  [Обновляем лучшую.],

  [210], [`                best_row = row`],
  [Запоминаем.],

  [211], [(пустая строка)],
  [Разделитель.],

  [212], [`    return rows, best_row`],
  [Возвращаем все результаты и лучшую комбинацию.],
)

== Строки 214–304: main()

#table(
  columns: (auto, 1fr, 1fr),
  align: (center, left, left),
  table.header[*Строка*][*Код*][*Пояснение*],

  [213], [(пустая строка)],
  [Разделитель.],

  [214], [`def main():`],
  [Точка входа.],

  [215], [`    datasets = load_cls_datasets()`],
  [Создаём 4 датасета.],

  [216], [(пустая строка)],
  [Разделитель.],

  [217], [`    # Визуализация датасетов`],
  [Комментарий.],

  [218], [`    fig, axes = plt.subplots(2, 2, figsize=(10, 10))`],
  [Сетка 2×2 для обзорного графика.],

  [219], [`    fig.suptitle('Классификационные выборки', fontsize=14, fontweight='bold')`],
  [Общий заголовок.],

  [220], [`    for ax, ds in zip(axes.flat, datasets):`],
  [`axes.flat` — итератор по 4 субплотам; `zip` связывает subplot↔dataset.],

  [221], [`        X, y = ds['data']`],
  [Данные.],

  [222], [`        ax.scatter(X[y==0, 0], X[y==0, 1], s=16, alpha=0.75, label='Класс 0')`],
  [Точки класса 0.],

  [223], [`        ax.scatter(X[y==1, 0], X[y==1, 1], s=16, alpha=0.75, label='Класс 1')`],
  [Точки класса 1.],

  [224], [`        ax.set_title(ds['name'])`],
  [Заголовок — имя датасета.],

  [225], [`        ax.legend(fontsize=8)`],
  [Легенда.],

  [226], [`        ax.grid(True, alpha=0.3)`],
  [Сетка.],

  [227], [`        ax.set_aspect('equal', adjustable='box')`],
  [Одинаковый масштаб по осям.],

  [228], [`    plt.tight_layout()`],
  [Подгонка.],

  [229], [`    os.makedirs(IMG_DIR, exist_ok=True)`],
  [Создание папки.],

  [230], [`    fig.savefig(os.path.join(IMG_DIR, 'cls_datasets.png'), dpi=DPI, bbox_inches='tight')`],
  [Сохранение обзорного графика.],

  [231], [`    plt.show()`],
  [Показ.],

  [232], [`    plt.close()`],
  [Закрытие.],

  [233], [(пустая строка)],
  [Разделитель.],

  [234], [`    # Перебор архитектур`],
  [Комментарий.],

  [235], [`    print(f'\\nПеребор архитектур (epochs={CLS_EPOCHS}, lr={CLS_LR})…')`],
  [Информационное сообщение.],

  [236], [`    rows, best = run_classification_experiments(datasets)`],
  [Запуск всех 180 экспериментов (4 × 3 × 15).],

  [237], [(пустая строка)],
  [Разделитель.],

  [238], [`    activations = ['sigmoid', 'tanh', 'relu']`],
  [Список для вывода.],

  [239], [`    for ds in datasets:`],
  [Для каждого датасета...],

  [240], [`        print('\\n' + '=' * 70)`],
  [Линия.],

  [241], [`        print(ds['name'])`],
  [Имя.],

  [242], [`        print('=' * 70)`],
  [Линия.],

  [243], [`        for act in activations:`],
  [Для каждой активации...],

  [244], [`            print(f'\\n  Активация: {act}  (Test accuracy)')`],
  [Подзаголовок.],

  [245], [`            print('  ' + '-' * 62)`],
  [Линия.],

  [246], [`            print_cls_pivot(rows, ds['name'], act)`],
  [Печать таблицы accuracy.],

  [247], [(пустая строка)],
  [Разделитель.],

  [248], [`    print('\\n\\n── Лучшие модели по классификации ──')`],
  [Заголовок.],

  [249], [`    for (ds_name, act), rec in best.items():`],
  [Перебор лучших моделей.],

  [250], [`        print(f'  {ds_name} | {act}: слоёв=... нейронов=... acc=...')`],
  [Вывод.],

  [251], [`        plot_decision_boundary(rec, save_path=os.path.join(IMG_DIR, f'cls_{ds_name.lower()}_{act}.png'))`],
  [Графики + сохранение.],

  [252], [(пустая строка)],
  [Разделитель.],

  [253], [`    # Минимальный размер выборки`],
  [Комментарий.],

  [254], [`    print('\\n── Минимальный размер выборки для accuracy >= 90 % ──')`],
  [Заголовок.],

  [255], [`    size_summary = estimate_min_train_size(datasets)`],
  [Запуск поиска.],

  [256], [`    col_w = 14`],
  [Ширина столбца.],

  [257], [`    header = (f"{'Датасет':<12}{'Активация':<{col_w}}{'Слоёв':<{col_w}}" ...)`],
  [Формирование заголовка с 7 столбцами.],

  [258], [`    print(header)`],
  [Печать.],

  [259], [`    print('-' * len(header))`],
  [Линия.],

  [260], [`    for r in size_summary:`],
  [Для каждого датасета...],

  [261], [`        mf = f"{r['min_frac']:.2f}" if r['min_frac'] is not None else '—'`],
  [Минимальная доля или прочерк.],

  [262], [`        ms = str(r['min_train_size']) if r['min_train_size'] is not None else '—'`],
  [Минимальное число объектов или прочерк.],

  [263], [`        print(f"...")`],
  [Форматированная строка таблицы.],

  [264], [(пустая строка)],
  [Разделитель.],

  [265], [`    # Кросс-валидация`],
  [Комментарий.],

  [266], [`    print(f'\\n── {CV_K}-fold кросс-валидация ──')`],
  [Заголовок.],

  [267], [`    cv_summary = []`],
  [Аккумулятор лучших результатов CV.],

  [268], [`    for ds in datasets:`],
  [Для каждого датасета...],

  [269], [`        name = ds['name']`],
  [Имя.],

  [270], [`        X, y = ds['data']`],
  [Данные.],

  [271], [`        details, best_row = cv_grid_search(X, y)`],
  [Запуск grid search с CV.],

  [272], [(пустая строка)],
  [Разделитель.],

  [273], [`        cv_summary.append({`],
  [Сохраняем лучший результат CV.],

  [274], [`            'dataset': name,`],
  [Датасет.],

  [275], [`            'best_activation': best_row['activation'],`],
  [Лучшая активация.],

  [276], [`            'best_layers': best_row['linear_layers'],`],
  [Лучшее число слоёв.],

  [277], [`            'cv_mean_acc': best_row['cv_mean_acc'],`],
  [Среднее accuracy.],

  [278], [`            'cv_std_acc': best_row['cv_std_acc'],`],
  [Std.],

  [279], [`        })`],
  [Конец.],

  [280], [(пустая строка)],
  [Разделитель.],

  [281], [`        print(f'\\n  {name}: все комбинации')`],
  [Заголовок для датасета.],

  [282], [`        print(f"  {'Активация':<12}{'Слоёв':<10}{'N':<8}{'Mean acc':<14}{'Std acc':<14}")`],
  [Заголовок таблицы.],

  [283], [`        print('  ' + '-' * 56)`],
  [Линия.],

  [284], [`        for r in sorted(details, key=lambda x: x['cv_mean_acc']):`],
  [Сортировка по mean acc (от худшего к лучшему).],

  [285], [`            print(f"  {r['activation']:<12}{r['linear_layers']:<10}{r['hidden_dim']:<8}{r['cv_mean_acc']:<14.4f}{r['cv_std_acc']:<14.4f}")`],
  [Печать строки.],

  [286], [(пустая строка)],
  [Разделитель.],

  [287], [`    print(f'\\n  Лучшие результаты {CV_K}-fold CV:')`],
  [Заголовок итоговой таблицы.],

  [288], [`    print(f"  {'Датасет':<12}{'Активация':<12}{'Слоёв':<10}{'Mean acc':<14}{'Std acc':<14}")`],
  [Заголовок.],

  [289], [`    print('  ' + '-' * 60)`],
  [Линия.],

  [290], [`    for r in cv_summary:`],
  [Для каждого датасета...],

  [291], [`        print(f"  {r['dataset']:<12}{r['best_activation']:<12}{r['best_layers']:<10}{r['cv_mean_acc']:<14.4f}{r['cv_std_acc']:<14.4f}")`],
  [Итоговая строка: лучшая комбинация для каждого датасета.],

  [292], [(пустая строка)],
  [Разделитель.],

  [293], [(пустая строка)],
  [Дополнительный разделитель.],

  [294], [`if __name__ == '__main__':`],
  [Проверка: файл запущен как скрипт?],

  [295], [`    main()`],
  [Вызов `main()`.],
)
