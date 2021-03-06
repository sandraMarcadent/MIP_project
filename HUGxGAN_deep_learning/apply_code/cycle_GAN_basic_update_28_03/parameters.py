################# DATA LOADER PARAMETERS

SAVE_MAX_IMAGES = 5000
MAX_IMAGES = 0
IMG_HEIGHT = 224 
IMG_WIDTH = 224
IMG_CHANNELS = 1
DEFAULT_PATH = '/mnt/dataC/HUGxGAN/data/HUGxGAN_224.hdf5'
DATA_NAME = 'cxr8'


################ Cycle GAN PARAMETERS

# MODEL TRAINING PARAMETERS
BATCH_SIZE = 1
POOL_SIZE = 50
NGF = 32
NDF = 64

# HYPER PARAMETERS

LAMBDA_A =  10.0
LAMBDA_B =  10.0
BASE_LR = 0.0002
MAX_STEP = 100
GEN_FREQ = 1
NET_VERSION = 'pytorch'

DO_FLIPPING = 1
SKIP = True

# PRINT OPTION
SAVE_IMAGES = 5
