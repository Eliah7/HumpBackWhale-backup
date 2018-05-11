# imports
from network import CNN
import data_prep as dp

# constants
IMG_ROWS = 1613
IMG_COLS = 1050
IMG_CHANNELS = 3
NB_CLASSES = 4251


model = CNN.build(NB_CLASSES, IMG_ROWS, IMG_COLS, IMG_CHANNELS)

# TODO: train the tensorFlow model too




