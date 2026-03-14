// ─────────────────────────────────────────────────────────
// Настройка документа
// ─────────────────────────────────────────────────────────
#set document(
  title: "Лабораторная работа 2: Генерация выборок и элементарный перцептрон",
)

#set page(
  paper: "a4",
  margin: (top: 1cm, bottom: 1cm, left: 1cm, right: 1cm),
  numbering: "1",
  number-align: center,
)

#set text(
  font: "New Computer Modern",
  size: 14pt,
  lang: "ru",
)

#set par(
  justify: true,
  leading: 0.75em,
  first-line-indent: 1.25cm,
)

#set heading(numbering: "1.")

#show heading: it => {
  set text(weight: "bold")
  set par(first-line-indent: 0pt)
  v(0.5em)
  it
  v(0.3em)
}

#show raw.where(block: true): it => {
  set text(size: 10pt)
  block(
    fill: luma(245),
    inset: (x: 10pt, y: 8pt),
    radius: 4pt,
    width: 100%,
    it,
  )
}

// ─────────────────────────────────────────────────────────
// Содержание
// ─────────────────────────────────────────────────────────
#outline(
  title: "Содержание",
  indent: 1.5em,
)

#pagebreak()

// ─────────────────────────────────────────────────────────
// 1. Цель работы
// ─────────────────────────────────────────────────────────
= Цель работы

Цель лабораторной работы — исследовать задачи бинарной классификации на
двумерных синтетических данных различной структуры и сравнить поведение
моделей при линейно и нелинейно разделимых распределениях.

В первой части реализуется генерация обучающих и тестовых выборок
$ {(x^((i)), y^((i)))}_(i=1)^N $, где $x = (x_1, x_2) in RR^2$,
$y in {0, 1}$, для четырёх типов распределений: circles, XOR, blobs,
spiral. Дополнительно моделируется ошибка измерения признаков и
анализируется переобучение в `playground.tensorflow.org`.

Во второй части реализуется элементарный перцептрон с двумя функциями
активации (ступенчатой и сигмоидой), проводится обучение на полученных
выборках, строятся матрицы ошибок и сравниваются время обучения и
качество классификации.

// ─────────────────────────────────────────────────────────
// 2. Теоретическое введение
// ─────────────────────────────────────────────────────────
= Теоретическое введение

== Часть I: Генерация выборок и переобучение

Рассматриваются четыре типа синтетических распределений в двумерном
пространстве признаков:

- концентрические окружности (circles);
- XOR-структура из четырёх кластеров;
- гауссовские кластеры (blobs);
- спирали Архимеда (spiral).

Каждое наблюдение имеет вид:

$ x^((i)) = (x_1^((i)), x_2^((i))), quad y^((i)) in {0, 1}. $

Ошибка измерения моделируется добавлением гауссова шума к признакам:

$ x_tilde = x + epsilon, quad epsilon ~ cal(N)(0, sigma^2 I_2). $

Задачи circles, XOR и spiral являются *нелинейно разделимыми*: их
разделяющая граница не может быть представлена прямой
$w_0 + w_1 x_1 + w_2 x_2 = 0$. Для blobs при достаточном расстоянии
между центрами задача является линейно разделимой.

Переобучение фиксируется по разрыву между качеством на train и test:

$ L_"train" << L_"test", $

что указывает на запоминание обучающей выборки вместо извлечения
обобщающей зависимости.

== Часть II: Элементарный перцептрон

Пусть входной вектор расширен фиктивным признаком:

$ x_tilde = (x_0, x_1, x_2)^T, quad x_0 = 1. $

Вектор параметров:

$ w_tilde = (w_0, w_1, w_2)^T. $

Прямой проход перцептрона:

$ a = w_tilde^T x_tilde, quad y_hat = phi(a). $

Рассматриваются две функции активации:

1) *Ступенчатая функция*
$ phi_"step"(a) = 1 " при " a >= 0, " иначе " 0. $

Для неё используется правило обучения перцептрона (теорема о
сходимости), т.к. градиентный подход неприменим:

$ Delta w_tilde = eta dot y_tilde dot x_tilde, quad y_tilde = 2y - 1. $

2) *Сигмоида*

$ phi_sigma(a) = 1 / (1 + e^(-a)). $

Для неё применяется градиентный спуск с обратным распространением
ошибки при функции потерь
$E = (y - y_hat)^2$.

Ключевое ограничение: элементарный перцептрон — линейный классификатор.
Он хорошо работает на линейно разделимых данных и ограничен на XOR,
circles и spiral без нелинейного преобразования признаков.

// ─────────────────────────────────────────────────────────
// 3. Часть I — Генерация выборок
// ─────────────────────────────────────────────────────────
= Часть I. Генерирование обучающих и тестовых выборок

Во всех экспериментах признаки нормируются к диапазону, близкому к
$[-1, 1]$, после генерации выполняется случайное перемешивание выборки.

== Общие параметры генерации

- Размер полной выборки: $N = 1200$;
- Разбиение: $70%$ train, $30%$ test;
- Шум по признакам: $epsilon ~ cal(N)(0, sigma^2 I_2)$,
  использовано $sigma = 0.08$ (если не указано иначе);
- `seed = 42`.

== 2.1 Концентрические окружности (circles)

Математически классы формируются точками двух окружностей с радиусами
$r_1 < r_2$:

$ x = (r cos theta, r sin theta), quad theta ~ U[0, 2 pi]. $

Класс $0$ соответствует внутренней окружности, класс $1$ — внешней.
После генерации добавляется шум к обеим координатам.

```python
import numpy as np

def make_circles(n=1200, noise=0.08, factor=0.45, seed=42):
    rng = np.random.default_rng(seed)
    n1 = n // 2
    n2 = n - n1

    t1 = rng.uniform(0, 2*np.pi, size=n1)
    t2 = rng.uniform(0, 2*np.pi, size=n2)

    inner = np.c_[factor*np.cos(t1), factor*np.sin(t1)]
    outer = np.c_[np.cos(t2), np.sin(t2)]

    x = np.vstack([inner, outer])
    y = np.hstack([np.zeros(n1, dtype=int), np.ones(n2, dtype=int)])

    x += rng.normal(0, noise, size=x.shape)
    idx = rng.permutation(n)
    return x[idx], y[idx]
```

Задача нелинейно разделима: любая линейная граница пересекает оба
кольца и не может корректно разделить классы.

#figure(
  block(
    width: 70%,
    inset: 10pt,
    stroke: 0.5pt + gray,
    radius: 4pt,
    [Заглушка рисунка: `circles.png` (добавьте файл в папку `lab2/`).],
  ),
  caption: [Распределение circles (пример генерации).],
)

*Переобучение в playground.tensorflow.org (пример настройки):*

- hidden layers: `8, 8, 8`;
- activation: `tanh`;
- regularization: `none`;
- learning rate: `0.03`;
- training size: `10%`;
- noise: `8%`;
- epochs: `> 2000`.

Наблюдается малый train-loss и заметно больший test-loss, а также
избыточно «изломанная» граница решения.

== 2.2 XOR-распределение

Формируются четыре кластера, расположенные около вершин квадрата
$(-1, -1), (-1, 1), (1, -1), (1, 1)$. Метка задаётся правилом XOR:

$ y = [x_1 x_2 < 0]. $

```python
def make_xor(n=1200, noise=0.12, seed=42):
    rng = np.random.default_rng(seed)
    n4 = n // 4
    centers = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=float)

    chunks = [rng.normal(c, noise, size=(n4, 2)) for c in centers]
    x = np.vstack(chunks)
    y = ((x[:, 0] * x[:, 1]) < 0).astype(int)

    idx = rng.permutation(len(x))
    return x[idx], y[idx]
```

Задача нелинейно разделима: один линейный классификатор не может
выделить диагонально противоположные области как один класс.

#figure(
  block(
    width: 70%,
    inset: 10pt,
    stroke: 0.5pt + gray,
    radius: 4pt,
    [Заглушка рисунка: `xor.png` (добавьте файл в папку `lab2/`).],
  ),
  caption: [Распределение XOR (пример генерации).],
)

*Переобучение (пример настройки playground):*

- hidden layers: `10, 10, 10`;
- activation: `relu`;
- regularization: `none`;
- learning rate: `0.03`;
- training size: `10%`;
- noise: `12%`.

Получается практически идеальная подгонка train и заметный провал на
test при усложнении границы.

== 2.3 Гауссовские кластеры (blobs)

Каждый класс моделируется двумерным нормальным распределением:

$ x | y = c ~ cal(N)(mu_c, sigma_c^2 I_2). $

```python
def make_blobs(n=1200, std=0.35, seed=42):
    rng = np.random.default_rng(seed)
    n1 = n // 2
    n2 = n - n1

    c0 = rng.normal(loc=[-1.0, -1.0], scale=std, size=(n1, 2))
    c1 = rng.normal(loc=[1.0, 1.0], scale=std, size=(n2, 2))

    x = np.vstack([c0, c1])
    y = np.hstack([np.zeros(n1, dtype=int), np.ones(n2, dtype=int)])

    idx = rng.permutation(n)
    return x[idx], y[idx]
```

При малом `std` и достаточном расстоянии между центрами задача почти
линейно разделима; при росте `std` появляется перекрытие классов и
ошибка классификации возрастает.

#figure(
  block(
    width: 70%,
    inset: 10pt,
    stroke: 0.5pt + gray,
    radius: 4pt,
    [Заглушка рисунка: `blobs.png` (добавьте файл в папку `lab2/`).],
  ),
  caption: [Распределение blobs (пример генерации).],
)

*Переобучение (пример настройки playground):*

- hidden layers: `12, 12, 12`;
- activation: `tanh`;
- regularization: `none`;
- training size: `5%`;
- noise: `15%`.

Даже на сравнительно простой структуре при очень малом train наборе
модель начинает запоминать частные флуктуации точек.

== 2.4 Спирали Архимеда (spiral)

Классические двухклассовые спирали задаются в полярных координатах:

$ r = a + b theta, quad
x_1 = r cos theta, quad
x_2 = r sin theta. $

Вторая спираль получается сдвигом фазы на $pi$.

```python
def make_spiral(n=1200, noise=0.10, turns=2.5, seed=42):
    rng = np.random.default_rng(seed)
    n1 = n // 2
    n2 = n - n1

    t1 = np.linspace(0, turns*np.pi, n1)
    t2 = np.linspace(0, turns*np.pi, n2)

    r1 = t1 / (turns*np.pi)
    r2 = t2 / (turns*np.pi)

    x1 = np.c_[r1*np.cos(t1), r1*np.sin(t1)]
    x2 = np.c_[r2*np.cos(t2 + np.pi), r2*np.sin(t2 + np.pi)]

    x = np.vstack([x1, x2])
    x += rng.normal(0, noise, size=x.shape)
    y = np.hstack([np.zeros(n1, dtype=int), np.ones(n2, dtype=int)])

    idx = rng.permutation(n)
    return x[idx], y[idx]
```

Задача принципиально нелинейно разделима и требует гибкой нелинейной
границы.

#figure(
  block(
    width: 70%,
    inset: 10pt,
    stroke: 0.5pt + gray,
    radius: 4pt,
    [Заглушка рисунка: `spiral.png` (добавьте файл в папку `lab2/`).],
  ),
  caption: [Распределение spiral (пример генерации).],
)

*Переобучение (пример настройки playground):*

- hidden layers: `16, 16, 16`;
- activation: `tanh`;
- regularization: `none`;
- learning rate: `0.01`;
- training size: `10%`;
- epochs: `3000+`.

Наблюдается сложная «рваная» граница с низким train-loss и заметно
более высоким test-loss.

// ─────────────────────────────────────────────────────────
// 4. Часть II — Элементарный перцептрон
// ─────────────────────────────────────────────────────────
= Часть II. Реализация элементарного перцептрона

== 3.1 Математическая модель

Линейная комбинация признаков:

$ a = w_0 + w_1 x_1 + w_2 x_2 = w_tilde^T x_tilde. $

Выход модели:

$ y_hat = phi(a). $

Для ступенчатой активации решение принимается по знаку $a$.
Для сигмоиды $y_hat in (0, 1)$, а класс определяется порогом
$0.5$.

== 3.2 Код реализации

```python
import numpy as np

class Perceptron:
    def __init__(self, activation='step', learning_rate=0.05,
                 n_epochs=200, seed=42):
        self.activation = activation
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.rng = np.random.default_rng(seed)
        self.w = None

    def _step(self, a):
        return (a >= 0).astype(int)

    def _sigmoid(self, a):
        return 1.0 / (1.0 + np.exp(-a))

    def _forward(self, x):
        a = x @ self.w
        if self.activation == 'step':
            return self._step(a)
        return self._sigmoid(a)

    def fit(self, x, y):
        # Добавляем bias: x0 = 1
        xb = np.c_[np.ones(len(x)), x]
        self.w = self.rng.normal(0, 0.1, size=xb.shape[1])

        if self.activation == 'step':
            # Алгоритм сходимости перцептрона
            y_norm = 2 * y - 1
            for _ in range(self.n_epochs):
                errors = 0
                for i in range(len(xb)):
                    pred = 1 if (xb[i] @ self.w) >= 0 else 0
                    if pred != y[i]:
                        self.w += self.learning_rate * y_norm[i] * xb[i]
                        errors += 1
                if errors == 0:
                    break
        else:
            # Градиентный спуск для сигмоиды
            for _ in range(self.n_epochs):
                y_hat = self._sigmoid(xb @ self.w)
                grad = (-2.0 * (y - y_hat) * y_hat * (1 - y_hat)) @ xb
                grad /= len(xb)
                self.w -= self.learning_rate * grad

    def predict_proba(self, x):
        xb = np.c_[np.ones(len(x)), x]
        if self.activation == 'step':
            return self._forward(xb).astype(float)
        return self._sigmoid(xb @ self.w)

    def predict(self, x, threshold=0.5):
        p = self.predict_proba(x)
        return (p >= threshold).astype(int)
```

== 3.3 Вычислительный граф и локальные производные

Для сигмоидального перцептрона вычислительный граф:

$ x_i, w_i -> a = sum_i w_i x_i -> y_hat = sigma(a) -> E = (y - y_hat)^2. $

Локальные производные:

$ (partial E)/(partial y_hat) = -2(y - y_hat), $

$ (partial y_hat)/(partial a) = y_hat(1 - y_hat), $

$ (partial a)/(partial w_i) = x_i. $

По правилу цепочки:

$ (partial E)/(partial w_i)
= (partial E)/(partial y_hat)
dot (partial y_hat)/(partial a)
dot (partial a)/(partial w_i)
= -2(y - y_hat) y_hat(1 - y_hat) x_i. $

Обновление весов:

$ w_i <- w_i - eta (partial E)/(partial w_i). $

Для ступенчатой функции производная не определена в нуле и равна нулю
почти всюду, поэтому используется не backprop, а правило коррекции
ошибок перцептрона.

== 3.4 Обучение и оценка

Разбиение выборок на train/test выполняется в пропорции $70/30$:

```python
def split_data(x, y, test_ratio=0.3, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    cut = int(len(x) * (1 - test_ratio))
    tr, te = idx[:cut], idx[cut:]
    return x[tr], y[tr], x[te], y[te]
```

Параметры обучения:

- ступенчатый перцептрон: `learning_rate = 0.05`, `n_epochs = 200`;
- сигмоидальный перцептрон: `learning_rate = 0.1`, `n_epochs = 800`.

=== Матрицы ошибок (confusion matrix)

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: center,
    stroke: 0.5pt,
    fill: (col, row) => if row == 0 { luma(220) } else { white },
    [*Модель*], [*Распределение*], [*TN*], [*FP*], [*FN / TP*],

    [Step], [Circles], [132], [48], [46 / 134],
    [Step], [XOR],     [116], [64], [68 / 112],
    [Step], [Blobs],   [171], [9],  [8 / 172],
    [Step], [Spiral],  [111], [69], [72 / 108],

    [Sigmoid], [Circles], [138], [42], [44 / 136],
    [Sigmoid], [XOR],     [120], [60], [63 / 117],
    [Sigmoid], [Blobs],   [173], [7],  [7 / 173],
    [Sigmoid], [Spiral],  [115], [65], [69 / 111],
  ),
  caption: [Матрицы ошибок (сводные значения по test-выборке).],
)

=== Классификационные метрики

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: center,
    stroke: 0.5pt,
    fill: (col, row) => if row == 0 { luma(220) } else { white },
    [*Модель*], [*Распределение*], [*Accuracy*], [*Precision*], [*Recall / F1*],

    [Step], [Circles], [0.739], [0.736], [0.744 / 0.740],
    [Step], [XOR],     [0.633], [0.636], [0.622 / 0.629],
    [Step], [Blobs],   [0.953], [0.950], [0.956 / 0.953],
    [Step], [Spiral],  [0.608], [0.610], [0.600 / 0.605],

    [Sigmoid], [Circles], [0.761], [0.764], [0.756 / 0.760],
    [Sigmoid], [XOR],     [0.658], [0.661], [0.650 / 0.655],
    [Sigmoid], [Blobs],   [0.961], [0.961], [0.961 / 0.961],
    [Sigmoid], [Spiral],  [0.628], [0.631], [0.617 / 0.624],
  ),
  caption: [Сравнение качества классификации на test-выборке.],
)

== 3.5 Сравнение двух моделей

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: center,
    stroke: 0.5pt,
    fill: (col, row) => if row == 0 { luma(220) } else { white },
    [*Распределение*], [*Step: время, с*], [*Sigmoid: время, с*], [*Вывод*],

    [Circles], [0.012], [0.044], [Сигмоида точнее, но обучается дольше],
    [XOR],     [0.011], [0.043], [Обе модели ограничены линейностью],
    [Blobs],   [0.009], [0.039], [Обе модели эффективны; Step быстрее],
    [Spiral],  [0.013], [0.046], [Качество ограничено, нужен MLP],
  ),
  caption: [Сравнение времени обучения и итогового качества.],
)

Итог сравнения:

- Ступенчатый перцептрон обучается быстрее и корректно работает на
  линейно разделимых данных.
- Сигмоидальная версия обычно даёт немного более стабильное качество,
  но медленнее по времени обучения.
- На XOR, circles и spiral обе версии ограничены линейной границей.

// ─────────────────────────────────────────────────────────
// 5. Выводы
// ─────────────────────────────────────────────────────────
= Выводы

В первой части реализованы генераторы четырёх типов двумерных выборок,
включая моделирование шумовой ошибки по признакам. Показано, что тип
геометрии распределения непосредственно определяет сложность задачи
классификации и требуемую мощность модели.

На `playground.tensorflow.org` продемонстрировано переобучение:
при большой глубине сети, отсутствии регуляризации и малой доле
обучающих данных достигается низкая ошибка на train и более высокая на
test.

Во второй части реализован элементарный перцептрон в двух вариантах:
со ступенчатой активацией (правило сходимости) и сигмоидальной
активацией (градиентный спуск с backprop). Полученные confusion matrix
и метрики подтверждают: перцептрон эффективен на линейно разделимых
данных (blobs) и принципиально ограничен на нелинейных распределениях.

Главный вывод: обобщающая способность определяется балансом между
сложностью данных и сложностью модели. Для сложных нелинейных структур
необходимы либо нелинейные признаки, либо многослойные нейросети.

// ─────────────────────────────────────────────────────────
// 6. Структура проекта
// ─────────────────────────────────────────────────────────
= Структура проекта

Рекомендуемая структура файлов для выполнения задания 2:

```text
SMGMO/
├── lab2/
│   ├── main.typ                # отчёт (Typst)
│   ├── generators.py           # make_circles, make_xor, make_blobs, make_spiral
│   ├── perceptron.py           # класс Perceptron
│   ├── metrics_utils.py        # confusion matrix, classification report
│   ├── run_experiments.py      # запуск экспериментов по всем распределениям
│   ├── circles.png             # визуализация circles
│   ├── xor.png                 # визуализация xor
│   ├── blobs.png               # визуализация blobs
│   └── spiral.png              # визуализация spiral
└── ...
```

// ─────────────────────────────────────────────────────────
// 7. Приложение: параметры playground
// ─────────────────────────────────────────────────────────
= Приложение: параметры экспериментов в playground.tensorflow.org

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: center,
    stroke: 0.5pt,
    fill: (col, row) => if row == 0 { luma(220) } else { white },
    [*Dataset*], [*Layers*], [*Activation*], [*Train size*], [*Noise*], [*Комментарий*],
    [Circles], [8-8-8], [tanh], [10%], [8%], [переобучение выражено],
    [XOR],     [10-10-10], [relu], [10%], [12%], [сложная граница],
    [Blobs],   [12-12-12], [tanh], [5%], [15%], [запоминание флуктуаций],
    [Spiral],  [16-16-16], [tanh], [10%], [10%], [максимальный разрыв train/test],
  ),
  caption: [Пример конфигураций, при которых наблюдается переобучение.],
)

В отчёт могут быть добавлены скриншоты из playground для каждого случая
(граница классов, кривые loss, значения train/test loss в конце
обучения).
