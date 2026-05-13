// ─────────────────────────────────────────────────────────
// Визуальный pipeline лабораторной работы 4
// ─────────────────────────────────────────────────────────
#set document(
  title: "Лабораторная работа 4: Pipeline обучения CNN",
)

#set page(
  paper: "a4",
  margin: (top: 1.2cm, bottom: 1.2cm, left: 1.2cm, right: 1.2cm),
  numbering: "1",
  number-align: center,
)

#set text(
  font: "New Computer Modern",
  size: 13.5pt,
  lang: "ru",
)

#set par(
  justify: true,
  leading: 0.72em,
  first-line-indent: 1.2cm,
)

#set heading(numbering: "1.")

#show heading: it => {
  set text(weight: "bold")
  set par(first-line-indent: 0pt)
  v(0.45em)
  it
  v(0.25em)
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

#let pipebox(title, body, fill: rgb("#f3f6fb")) = block(
  width: 100%,
  inset: 10pt,
  radius: 5pt,
  stroke: 0.6pt + luma(150),
  fill: fill,
  [*#title*\#body],
)

#outline(title: "Содержание", indent: 1.5em)

#pagebreak()

= Зачем нужен этот файл

Этот документ нужен как *визуальная карта всей лабораторной*. Если `main_teor.typ` объясняет теорию, а `main_code.typ` объясняет код, то здесь цель другая: показать *pipeline целиком*.

То есть ответить на вопросы:

- откуда берутся данные;
- как из них строятся `train`, `val`, `test`;
- как батч проходит через CNN;
- какие размеры тензоров получаются после каждого шага;
- где применяются гиперпараметры;
- как считается loss;
- как работает `backward` и обновление весов;
- как выбирается лучшая модель.

= Общая схема всей лабораторной

Ниже самая верхнеуровневая схема всей работы:

#grid(
  columns: (1fr, auto, 1fr, auto, 1fr, auto, 1fr),
  column-gutter: 6pt,
  row-gutter: 6pt,
  pipebox(
    "1. Конфиг",
    [Параметры из `config.py`:\датасет, batch size, epochs, patience, target accuracy.],
    fill: rgb("#edf7ed"),
  ),
  [→],
  pipebox(
    "2. Данные",
    [Загрузка `MNIST` или `FashionMNIST`, нормализация, split `train/val`, выбор `test`.],
    fill: rgb("#eef4ff"),
  ),
  [→],
  pipebox(
    "3. Модель",
    [Создание `ImageCNN` из блоков `A` или `B`, задание каналов, ядер, pooling.],
    fill: rgb("#fff4e8"),
  ),
  [→],
  pipebox(
    "4. Обучение",
    [Forward, loss, backward, optimizer step, validation, early stopping.],
    fill: rgb("#fdf0f3"),
  ),
)

После этого идёт второй уровень логики:

- в *блоке 1* лабораторная много раз повторяет шаги 3 и 4 для разных конфигураций;
- в *блоке 2* фиксирует хорошую архитектуру и отдельно исследует оптимизаторы и переобучение.

= Где какие параметры применяются

Очень полезно сразу видеть не просто список гиперпараметров, а *место их применения*.

#grid(
  columns: (1.3fr, 1.6fr, 2.3fr),
  gutter: 8pt,
  [*Параметр*], [*Где применяется*], [*Что меняет*],
  [`DATASET_NAME`], [`data.py -> load_image_dataset(...)`], [Какой датасет будет загружен: `MNIST` или `FashionMNIST`.],
  [`TRAIN_SUBSET`, `TEST_SUBSET`], [`block1_main.py`, `block2_main.py`], [Размер подвыборок для ускорения экспериментов.],
  [`VAL_RATIO`], [`data.py -> split_train_val(...)`], [Какую долю train выделить под validation.],
  [`BATCH_SIZE`], [`make_loaders(...)`, `DataLoader`], [Сколько изображений идёт в одном батче.],
  [`block_type`], [`cnn.py -> ImageCNN`], [Использовать блок `A` или `B`.],
  [`channels`], [`cnn.py -> цикл построения слоёв`], [Сколько выходных каналов у каждого блока.],
  [`conv_k`, `conv_s`], [`Conv2d`], [Размер ядра и шаг свёртки.],
  [`pool_k`, `pool_s`], [`MaxPool2d`], [Размер окна pooling и его шаг.],
  [`dropout_rate`], [`ImageCNN -> classifier`], [Сила dropout перед финальным линейным слоем.],
  [`lr`], [`make_optimizer(...)`], [Скорость изменения весов на каждом шаге.],
  [`PATIENCE`, `MIN_DELTA`], [`train_model(...)`], [Когда остановить обучение без улучшения validation.],
  [`TARGET_ACC`], [`train_model(...)`, `сводные графики`], [Контрольная цель качества, например `0.90`.],
)

= Pipeline данных

Сначала разберём именно путь данных до подачи в сеть.

== Шаг 1. Загрузка датасета

В `data.py` вызывается:

```python
train, test, info = load_image_dataset(name, data_root)
```

На этом шаге происходит:

1. выбор датасета по имени;
2. добавление `ToTensor()`;
3. при необходимости добавление `Normalize(mean, std)`;
4. загрузка обучающей и тестовой части.

Схема:

#grid(
  columns: (1fr, auto, 1fr, auto, 1fr),
  column-gutter: 6pt,
  pipebox("Имя датасета", [Например, `MNIST`.], fill: rgb("#edf7ed")),
  [→],
  pipebox("Transforms", [`ToTensor()` + опционально `Normalize(...)`.], fill: rgb("#eef4ff")),
  [→],
  pipebox("Dataset objects", [Получаем `train` и `test`, а также `info` с метаданными.], fill: rgb("#fff4e8")),
)

== Шаг 2. Уменьшение выборки

Чтобы сетка экспериментов не была слишком тяжёлой, используется `take_subset(...)`.

Схема:

#grid(
  columns: (1fr, auto, 1fr),
  column-gutter: 6pt,
  pipebox("Полный train", [Например, 60 000 изображений в `MNIST`.], fill: rgb("#eef4ff")),
  [→],
  pipebox("Train subset", [Например, 10 000 изображений для быстрого перебора конфигураций.], fill: rgb("#fdf0f3")),
)

== Шаг 3. Split train/val

Из train-подвыборки строятся две части:

- `train_ds` --- для обновления весов;
- `val_ds` --- для контроля качества и выбора модели.

Схема:

#grid(
  columns: (1fr, auto, 1fr, auto, 1fr),
  column-gutter: 6pt,
  pipebox("Train subset", [Например, 10 000 объектов.], fill: rgb("#eef4ff")),
  [→],
  pipebox("Split by VAL_RATIO", [Например, `0.2` означает 80% в train и 20% в validation.], fill: rgb("#fff4e8")),
  [→],
  pipebox("train_ds + val_ds", [Например, 8 000 для обучения и 2 000 для проверки.], fill: rgb("#edf7ed")),
)

== Шаг 4. DataLoader

После этого из датасетов создаются `DataLoader`.

Смысл:

- `train_loader` выдаёт батчи и перемешивает train;
- `val_loader` и `test_loader` выдают батчи без перемешивания.

Схема:

#grid(
  columns: (1fr, auto, 1fr, auto, 1fr),
  column-gutter: 6pt,
  pipebox("train_ds", [Сырые объекты train.], fill: rgb("#eef4ff")),
  [→],
  pipebox("DataLoader", [`batch_size`, `shuffle=True`, `num_workers`, `pin_memory`.], fill: rgb("#fff4e8")),
  [→],
  pipebox("train_loader", [Последовательность батчей для обучения.], fill: rgb("#edf7ed")),
)

= Что приходит на вход CNN

Для `MNIST` или `FashionMNIST` одно изображение имеет форму:

$ (1, 28, 28). $

Здесь:

- `1` --- число каналов, потому что изображение серое;
- `28 x 28` --- высота и ширина.

Если батч размера 128, то на вход модели приходит тензор формы:

$ (128, 1, 28, 28). $

Это один из самых важных моментов всей темы. Нужно чётко понимать: модель почти никогда не видит один объект, она обычно видит *батч объектов*.

= Pipeline одного прохода через модель

Теперь покажем, как один батч проходит через CNN.

== Общая схема forward-pass

#grid(
  columns: (1fr, auto, 1fr, auto, 1fr, auto, 1fr, auto, 1fr),
  column-gutter: 6pt,
  pipebox("Вход", [`(batch, 1, 28, 28)`], fill: rgb("#eef4ff")),
  [→],
  pipebox("Features", [Последовательность `ConvBlockA` или `ConvBlockB`.], fill: rgb("#fff4e8")),
  [→],
  pipebox("AdaptiveAvgPool2d", [Каждый канал сводится к размеру `1 x 1`.], fill: rgb("#edf7ed")),
  [→],
  pipebox("Flatten + Dropout", [Формируем вектор признаков перед классификатором.], fill: rgb("#fdf0f3")),
  [→],
  pipebox("Linear", [Получаем логиты классов.], fill: rgb("#f7f0ff")),
)

== Внутри `ConvBlockA`

Блок `A` в коде устроен так:

```python
Conv2d -> ReLU -> MaxPool2d
```

Схема:

#grid(
  columns: (1fr, auto, 1fr, auto, 1fr, auto, 1fr),
  column-gutter: 6pt,
  pipebox("Input", [Например, `(128, 1, 28, 28)`], fill: rgb("#eef4ff")),
  [→],
  pipebox("Conv2d", [Применяются фильтры. Число выходных каналов определяется `out_ch`.], fill: rgb("#fff4e8")),
  [→],
  pipebox("ReLU", [Все отрицательные отклики обнуляются.], fill: rgb("#edf7ed")),
  [→],
  pipebox("MaxPool2d", [Пространственный размер уменьшается, остаются сильные сигналы.], fill: rgb("#fdf0f3")),
)

== Внутри `ConvBlockB`

Блок `B` глубже:

```python
Conv2d -> ReLU -> Conv2d -> ReLU -> MaxPool2d
```

Схема:

#grid(
  columns: (1fr, auto, 1fr, auto, 1fr, auto, 1fr, auto, 1fr),
  column-gutter: 6pt,
  pipebox("Input", [Например, `(128, 1, 28, 28)`], fill: rgb("#eef4ff")),
  [→],
  pipebox("Conv2d #1", [Первичное выделение локальных признаков.], fill: rgb("#fff4e8")),
  [→],
  pipebox("ReLU", [Нелинейность.], fill: rgb("#edf7ed")),
  [→],
  pipebox("Conv2d #2", [Уточнение и усложнение признаков на том же числе каналов.], fill: rgb("#fdf0f3")),
  [→],
  pipebox("ReLU + Pool", [Ещё одна нелинейность и уменьшение размера.], fill: rgb("#f7f0ff")),
)

Разница в смысле такая:

- `A` извлекает признаки проще и быстрее;
- `B` извлекает признаки глубже, потому что внутри блока две свёртки.

= Как меняются размеры тензора

Это одна из самых важных частей для понимания CNN. Нужно видеть не только "какой слой стоит после какого", но и *какой размер получается после него*.

== Формула для свёртки

Если входной размер по одной координате равен `in`, а параметры свёртки равны `kernel = k`, `stride = s`, `padding = p`, то выходной размер:

$ "out" = floor(("in" + 2 p - k) / s + 1). $

== Формула для pooling

Для pooling используется та же идея:

$ "out" = floor(("in" - k) / s + 1). $

В лабораторной наиболее типичный случай такой:

- `conv_k = 3`;
- `conv_s = 1`;
- `padding = 1`;
- `pool_k = 2`;
- `pool_s = 2`.

== Пример размеров для одной популярной конфигурации

Возьмём конфиг:

```python
block_type = 'b'
channels = (32, 64)
conv_k = 3
conv_s = 1
pool_k = 2
pool_s = 2
```

Тогда pipeline размеров выглядит так:

#grid(
  columns: (1.35fr, auto, 1.35fr, auto, 1.35fr, auto, 1.35fr, auto, 1.35fr),
  column-gutter: 6pt,
  pipebox("Вход", [`(128, 1, 28, 28)`], fill: rgb("#eef4ff")),
  [→],
  pipebox("Блок 1", [`Conv -> ReLU -> Conv -> ReLU -> Pool`
`(128, 32, 14, 14)`], fill: rgb("#fff4e8")),
  [→],
  pipebox("Блок 2", [`Conv -> ReLU -> Conv -> ReLU -> Pool`
`(128, 64, 7, 7)`], fill: rgb("#edf7ed")),
  [→],
  pipebox("AdaptiveAvgPool", [`(128, 64, 1, 1)`], fill: rgb("#fdf0f3")),
  [→],
  pipebox("Flatten + Linear", [`(128, 64)` -> `(128, 10)`], fill: rgb("#f7f0ff")),
)

Что здесь произошло:

- первая свёртка с `k=3`, `s=1`, `p=1` сохраняет размер `28 x 28`;
- pooling `2 x 2` со stride `2` уменьшает размер до `14 x 14`;
- второй блок делает то же самое ещё раз и получаем `7 x 7`;
- затем каждый из 64 каналов усредняется до одной точки;
- остаётся вектор длины 64 на каждый объект;
- линейный слой переводит эти 64 признака в 10 логитов классов.

= Как строится модель из списка `channels`

Очень важно видеть, как параметр `channels` реально разворачивается в архитектуру.

== Пример 1

```python
channels = (16, 32)
```

Это означает:

- первый блок получает `in_channels = 1` и выдаёт `16` каналов;
- второй блок получает уже `16` каналов и выдаёт `32`.

Схема:

#grid(
  columns: (1fr, auto, 1fr, auto, 1fr),
  column-gutter: 6pt,
  pipebox("Input", [`1` канал], fill: rgb("#eef4ff")),
  [→],
  pipebox("Block 1", [`1 -> 16` каналов], fill: rgb("#fff4e8")),
  [→],
  pipebox("Block 2", [`16 -> 32` каналов], fill: rgb("#edf7ed")),
)

== Пример 2

```python
channels = (16, 32, 64)
```

Схема:

#grid(
  columns: (1fr, auto, 1fr, auto, 1fr, auto, 1fr),
  column-gutter: 6pt,
  pipebox("Input", [`1` канал], fill: rgb("#eef4ff")),
  [→],
  pipebox("Block 1", [`1 -> 16`], fill: rgb("#fff4e8")),
  [→],
  pipebox("Block 2", [`16 -> 32`], fill: rgb("#edf7ed")),
  [→],
  pipebox("Block 3", [`32 -> 64`], fill: rgb("#fdf0f3")),
)

То есть длина `channels` фактически задаёт *глубину модели*, а сами значения задают *ширину* по каналам.

= Где в pipeline применяется каждый архитектурный параметр

#grid(
  columns: (1.2fr, 2.4fr),
  gutter: 8pt,
  [*Параметр*], [*Как он проявляется в pipeline*],
  [`block_type`], [Выбирает схему `A` или `B`, то есть сколько свёрток внутри одного блока.],
  [`channels`], [Определяет число выходных каналов после каждого блока.],
  [`conv_k`], [Меняет размер окна, через которое фильтр "смотрит" на изображение.],
  [`conv_s`], [Меняет скорость перемещения фильтра и влияет на размер выходной карты признаков.],
  [`pool_k`], [Определяет размер окна pooling.],
  [`pool_s`], [Определяет, насколько быстро pooling уменьшает размерность.],
  [`dropout_rate`], [Определяет, какая доля признаков будет занулена перед финальным классификатором.],
)

= Pipeline одной training-итерации

Теперь разберём не только прохождение через слои, но и то, что происходит на *одном обучающем шаге*.

== Схема training-step

#grid(
  columns: (1fr, auto, 1fr, auto, 1fr, auto, 1fr, auto, 1fr, auto, 1fr),
  column-gutter: 6pt,
  pipebox("1. Батч", [Из `train_loader` приходит `(x, y)`.], fill: rgb("#eef4ff")),
  [→],
  pipebox("2. Forward", [`out = model(x)`
модель выдаёт логиты.], fill: rgb("#fff4e8")),
  [→],
  pipebox("3. Loss", [`loss = criterion(out, y)`], fill: rgb("#edf7ed")),
  [→],
  pipebox("4. Backward", [`loss.backward()`
считаются градиенты.], fill: rgb("#fdf0f3")),
  [→],
  pipebox("5. Update", [`optimizer.step()`
веса меняются.], fill: rgb("#f7f0ff")),
  [→],
  pipebox("6. Metrics", [Считаются loss и accuracy по батчу.], fill: rgb("#eef7f0")),
)

== То же самое словами

1. Берём батч изображений и истинных меток.
2. Прогоняем изображения через модель.
3. Получаем логиты классов.
4. Сравниваем логиты с истинными метками через `CrossEntropyLoss`.
5. Считаем градиенты по всем параметрам.
6. Оптимизатор обновляет параметры.

Это и есть элементарный акт обучения нейросети.

= Что именно происходит в `loss = criterion(out, y)`

В лабораторной используется `CrossEntropyLoss`.

Схема вычисления по смыслу такая:

#grid(
  columns: (1fr, auto, 1fr, auto, 1fr),
  column-gutter: 6pt,
  pipebox("Logits", [Модель выдаёт 10 сырых чисел на каждый объект.], fill: rgb("#eef4ff")),
  [→],
  pipebox("Softmax idea", [Из логитов можно получить вероятности классов.], fill: rgb("#fff4e8")),
  [→],
  pipebox("Cross-entropy", [Чем ниже вероятность правильного класса, тем больше loss.], fill: rgb("#edf7ed")),
)

Пример по одному объекту:

- если правильный класс получил вероятность `0.95`, loss маленький;
- если правильный класс получил вероятность `0.10`, loss большой.

= Pipeline одной эпохи

Одна эпоха --- это не один шаг, а *полный проход по всем батчам train-loader*.

Схема:

#grid(
  columns: (1fr, auto, 1fr, auto, 1fr),
  column-gutter: 6pt,
  pipebox("Start epoch", [Обнуляем накопители метрик.], fill: rgb("#eef4ff")),
  [→],
  pipebox("Loop over batches", [Для каждого батча повторяем forward/loss/backward/update.], fill: rgb("#fff4e8")),
  [→],
  pipebox("Epoch metrics", [Считаем средний `train_loss` и `train_acc` за эпоху.], fill: rgb("#edf7ed")),
)

После завершения train-части эпохи идёт validation.

= Pipeline validation после эпохи

После каждой эпохи модель проверяется на `val_loader`.

Схема:

#grid(
  columns: (1fr, auto, 1fr, auto, 1fr, auto, 1fr),
  column-gutter: 6pt,
  pipebox("model.eval()", [Отключаем training-режим.], fill: rgb("#eef4ff")),
  [→],
  pipebox("No grad", [Градиенты не считаются.], fill: rgb("#fff4e8")),
  [→],
  pipebox("Forward on val", [Прогоняем validation-батчи через модель.], fill: rgb("#edf7ed")),
  [→],
  pipebox("val_loss + val_acc", [Считаем качество без обновления весов.], fill: rgb("#fdf0f3")),
)

Это важный шаг, потому что именно здесь видно, учит ли модель общую закономерность, а не только запоминает train.

= Как работает early stopping

В `train_model(...)` после каждой эпохи сравнивается новая `val_acc` с лучшей предыдущей.

Схема логики:

#grid(
  columns: (1fr, auto, 1fr, auto, 1fr),
  column-gutter: 6pt,
  pipebox("Новая val_acc", [Получили качество после очередной эпохи.], fill: rgb("#eef4ff")),
  [→],
  pipebox("Сравнение с best_val_acc", [Проверяем, есть ли улучшение больше `MIN_DELTA`.], fill: rgb("#fff4e8")),
  [→],
  pipebox("Решение", [Если улучшение есть, сохраняем `best_state`, иначе увеличиваем счётчик `bad`.], fill: rgb("#edf7ed")),
)

Дальше:

- если `bad < PATIENCE`, обучение продолжается;
- если `bad >= PATIENCE`, обучение останавливается.

То есть early stopping --- это защитный фильтр от бесконечного обучения без смысла и от переобучения.

= Pipeline перебора конфигураций в блоке 1

Теперь посмотрим не на одну модель, а на *весь цикл поиска лучшей архитектуры*.

== Общая схема grid search

#grid(
  columns: (1fr, auto, 1fr, auto, 1fr, auto, 1fr, auto, 1fr),
  column-gutter: 6pt,
  pipebox("Сетка конфигов", [Формируются разные комбинации `block_type`, `channels`, `conv_k`, `pool_s` и других параметров.], fill: rgb("#eef4ff")),
  [→],
  pipebox("Config #1", [Создаём модель и обучаем.], fill: rgb("#fff4e8")),
  [→],
  pipebox("Config #2", [Повторяем всё заново с другой архитектурой.], fill: rgb("#edf7ed")),
  [→],
  pipebox("...", [Так перебираются все валидные конфигурации.], fill: rgb("#fdf0f3")),
  [→],
  pipebox("Best config", [Выбирается максимум по `best_val_acc`.], fill: rgb("#f7f0ff")),
)

== Что значит "заново"

Для каждой конфигурации происходит полный отдельный pipeline:

- создаётся новая модель;
- инициализируются новые веса;
- создаётся новый оптимизатор;
- модель обучается с нуля;
- сохраняются её история и метрики.

Это нужно для честного сравнения архитектур.

= Какой результат сохраняется для каждой конфигурации

После обучения одного конфига сохраняется словарь результата, где есть:

- `history`;
- `best_val_acc`;
- `best_epoch`;
- `epoch_to_target`;
- `duration_sec`;
- `trained_epochs`;
- `cfg`;
- `model`;
- `n_params`.

Схема:

#grid(
  columns: (1fr, auto, 1fr),
  column-gutter: 6pt,
  pipebox("Одна конфигурация", [Конкретный набор гиперпараметров CNN.], fill: rgb("#eef4ff")),
  [→],
  pipebox("Result object", [Метрики, история, обученная модель, число параметров и скорость сходимости.], fill: rgb("#edf7ed")),
)

= Финальное дообучение лучшей модели

После выбора лучшего конфига происходит ещё один отдельный pipeline.

== Схема

#grid(
  columns: (1fr, auto, 1fr, auto, 1fr, auto, 1fr),
  column-gutter: 6pt,
  pipebox("Best cfg", [Лучшая архитектура по validation.], fill: rgb("#eef4ff")),
  [→],
  pipebox("Новая модель", [Создаётся заново, а не берётся старая train/val-модель.], fill: rgb("#fff4e8")),
  [→],
  pipebox("Полный train", [Используются все доступные обучающие данные.], fill: rgb("#edf7ed")),
  [→],
  pipebox("Final test", [Считается итоговое качество на test.], fill: rgb("#fdf0f3")),
)

Почему это правильно:

- validation уже выполнила свою роль при выборе модели;
- теперь можно использовать все train-данные для максимального качества;
- итоговая оценка делается по test.

= Pipeline блока 2: оптимизаторы и переобучение

Во втором блоке pipeline немного другой.

== Часть 1. Сравнение оптимизаторов

Схема:

#grid(
  columns: (1fr, auto, 1fr, auto, 1fr, auto, 1fr),
  column-gutter: 6pt,
  pipebox("Фиксированная архитектура", [Выбирается одна хорошая CNN.], fill: rgb("#eef4ff")),
  [→],
  pipebox("Adam", [Обучение и запись `val_acc`.], fill: rgb("#fff4e8")),
  [→],
  pipebox("AdamW", [Та же архитектура, другой оптимизатор.], fill: rgb("#edf7ed")),
  [→],
  pipebox("SGD", [Та же архитектура, сравнение динамики.], fill: rgb("#fdf0f3")),
)

Идея здесь такая: архитектура фиксирована, меняется только способ обновления весов.

== Часть 2. Демонстрация переобучения

Схема:

#grid(
  columns: (1fr, auto, 1fr, auto, 1fr, auto, 1fr),
  column-gutter: 6pt,
  pipebox("Маленький train", [Берётся очень маленькая подвыборка.], fill: rgb("#eef4ff")),
  [→],
  pipebox("Модель без dropout", [Сеть легко запоминает train.], fill: rgb("#fff4e8")),
  [→],
  pipebox("Модель с dropout", [Регуляризация делает обучение устойчивее.], fill: rgb("#edf7ed")),
  [→],
  pipebox("Сравнение кривых", [Смотрим разрыв между train и val.], fill: rgb("#fdf0f3")),
)

= Полный pipeline в одном списке

Если сжать всю лабораторную в одну последовательность, получится такой маршрут.

1. Задать конфигурацию запуска.
2. Загрузить датасет и подготовить transforms.
3. Получить `train`, `val`, `test`.
4. Построить `DataLoader` для батчей.
5. Построить одну CNN-конфигурацию.
6. Прогонять батчи через `forward`.
7. Считать `CrossEntropyLoss`.
8. Делать `backward` и `optimizer.step()`.
9. После каждой эпохи считать validation-метрики.
10. Хранить лучшую модель по validation.
11. Повторить это для всех конфигураций в сетке.
12. Выбрать лучшую конфигурацию.
13. Переобучить её на полном train.
14. Посчитать итоговую test accuracy.
15. Построить графики и confusion matrix.

= Что особенно важно запомнить

Если тебе нужно быстро воспроизвести pipeline по памяти, то запомни пять ключевых уровней.

1. *Данные*: `dataset -> subset -> split -> dataloader`.
2. *Архитектура*: `input -> conv blocks -> global pooling -> linear`.
3. *Один шаг обучения*: `forward -> loss -> backward -> optimizer.step()`.
4. *Одна эпоха*: `train all batches -> validate`.
5. *Вся лабораторная*: `grid search -> best config -> final retrain -> test`.

Если эта пятиуровневая схема ясна, то и код, и теория этой лабораторной уже собираются в цельную картину.