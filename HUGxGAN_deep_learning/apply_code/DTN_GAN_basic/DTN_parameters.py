# image input params
BATCH_SIZE = 1
IMG_HEIGHT = 500
IMG_WIDTH = 500
IMG_CHANNELS = 1
IMG_SIZE = IMG_HEIGHT * IMG_WIDTH


DO_FLIPPING = 1
POOL_SIZE = 50
NGF = 32
NDF = 64

# pre-processing parameters
GRAY_LEVELS = 255
OFF_LEVEL = 0


# Training parameters
MAX_IMAGES = 400
MAX_STEP = 200
BASE_LR = 0.0002

# model and losses
NET_VERSION = 'tensorflow'
DATA_NAME = 'cxr8'

SKIP = False
LAMBDA_A =  10.0
LAMBDA_B =  10.0


# data paths
PATH_A = "/home/sandra/project_GANs/project_GANs_data_mining/cxr8/images_001/images/*.png"
PATH_B = "/home/sandra/project_GANs/project_GANs_data_mining/cxr8/images_002/images/*.png"