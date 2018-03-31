# The number of samples per batch.
BATCH_SIZE = 1

# The height of each image.
IMG_HEIGHT = 500

# The width of each image.
IMG_WIDTH = 500

# The number of color channels per image.
IMG_CHANNELS = 1

# 

POOL_SIZE = 50
NGF = 32
NDF = 64

# pre-processing parameters
GRAY_LEVELS = 255
OFF_LEVEL = 0


MAX_IMAGES = 400

IMG_SIZE = IMG_HEIGHT * IMG_WIDTH

# Training parameters

LAMBDA_A =  10.0
LAMBDA_B =  10.0

BASE_LR = 0.0002
MAX_STEP = 200
NET_VERSION = 'tensorflow'
DATA_NAME = 'cxr8'
DO_FLIPPING = 1
SKIP = True



PATH_A = "/home/sandra/project_GANs/project_GANs_data_mining/cxr8/images_001/images/*.png"
PATH_B = "/home/sandra/project_GANs/project_GANs_data_mining/cxr8/images_002/images/*.png"