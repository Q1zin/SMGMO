import os

SEED = 42
N    = int(os.environ.get('SMGMO2_N', '200'))
NOISE = float(os.environ.get('SMGMO2_NOISE', '0.8'))
AXIS_LIMIT = 6.0

DPI = 130
OUTPUT_FILE_BLOCK1    = 'task2_block1.png'
OUTPUT_FILE_BLOCK2    = 'task2_block2_boundaries.png'
OUTPUT_FILE_BLOCK2_CM = 'task2_block2_cm.png'
