import os

SEED = 42

REG_N = int(os.environ.get('SMGMO3_REG_N', '280'))
REG_EPS0 = float(os.environ.get('SMGMO3_REG_EPS0', '0.18'))
REG_EPOCHS = int(os.environ.get('SMGMO3_REG_EPOCHS', '350'))
REG_LR = float(os.environ.get('SMGMO3_REG_LR', '0.02'))

CLS_N = int(os.environ.get('SMGMO3_CLS_N', '500'))
CLS_NOISE = float(os.environ.get('SMGMO3_CLS_NOISE', '0.08'))
CLS_EPOCHS = int(os.environ.get('SMGMO3_CLS_EPOCHS', '250'))
CLS_LR = float(os.environ.get('SMGMO3_CLS_LR', '0.03'))

CV_K = int(os.environ.get('SMGMO3_CV_K', '5'))

DPI = 130
AXIS_LIMIT = 6.0
