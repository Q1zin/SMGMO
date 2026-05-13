import os

SEED = 52

# Датасет
DATASET_NAME = os.environ.get('SMGMO4_DATASET', 'MNIST')   # MNIST или FashionMNIST
DATA_ROOT = os.environ.get('SMGMO4_DATA_ROOT', './data')

# Размеры выборок
TRAIN_SUBSET = int(os.environ.get('SMGMO4_TRAIN_SUBSET', '10000'))
TEST_SUBSET = int(os.environ.get('SMGMO4_TEST_SUBSET', '2000'))
VAL_RATIO = float(os.environ.get('SMGMO4_VAL_RATIO', '0.2'))

# Обучение
BATCH_SIZE = int(os.environ.get('SMGMO4_BATCH_SIZE', '128'))
N_EPOCHS = int(os.environ.get('SMGMO4_N_EPOCHS', '20'))
PATIENCE = int(os.environ.get('SMGMO4_PATIENCE', '5'))
MIN_DELTA = float(os.environ.get('SMGMO4_MIN_DELTA', '1e-4'))
TARGET_ACC = float(os.environ.get('SMGMO4_TARGET_ACC', '0.90'))
FINAL_EPOCHS = int(os.environ.get('SMGMO4_FINAL_EPOCHS', str(N_EPOCHS)))

# DataLoader / preprocessing
NORMALIZE_IMAGES = os.environ.get('SMGMO4_NORMALIZE_IMAGES', '1') == '1'
NUM_WORKERS = int(os.environ.get('SMGMO4_NUM_WORKERS', '2'))
# None = авто (True на CUDA, False на CPU)
_pin_memory_raw = os.environ.get('SMGMO4_PIN_MEMORY', 'auto').strip().lower()
PIN_MEMORY = None if _pin_memory_raw == 'auto' else _pin_memory_raw in ('1', 'true', 'yes', 'on')

# Демонстрация переобучения
OVERFIT_TRAIN = int(os.environ.get('SMGMO4_OVERFIT_TRAIN', '300'))
OVERFIT_EPOCHS = int(os.environ.get('SMGMO4_OVERFIT_EPOCHS', '120'))

# Визуализация
DPI = 130
