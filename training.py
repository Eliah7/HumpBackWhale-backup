# imports
from network import CNN
import data_prep as dp
import constants as c

# constants
NB_CLASSES = 4251

# data
(X_train, y_train), (X_test, y_test) = dp.gen_training_data()
# X_train /= 255   # normalization
# X_test /= 255

# training
model = CNN.build(NB_CLASSES, c.IMG_ROWS, c.IMG_COLS, c.IMG_CHANNELS)


# TODO: Actually do the training
# TODO: Save the model in a h5 file





