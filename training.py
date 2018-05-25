# imports
from network import CNN
import data_prep as dp
import constants as c
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
import numpy as np

# constants
OUTPUT = 'model.h5'
NB_CLASSES = 4251
OPTIM = RMSprop()
NB_EPOCHS = 20
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 50
VERBOSE = 0

# data
(X_train, y_train), (X_test, y_test) = dp.gen_training_data()
X_train = np.array([i for i in X_train]).reshape(-1, c.IMG_ROWS, c.IMG_COLS, c.IMG_CHANNELS)
y_train = np.array([i for i in y_train])

X_test = np.array([i for i in X_test]).reshape(-1, c.IMG_ROWS, c.IMG_COLS, c.IMG_CHANNELS)
y_test = np.array([i for i in y_test])
# X_train /= 255   # normalization
# X_test /= 255

# training
model = CNN.build(NB_CLASSES, c.IMG_ROWS, c.IMG_COLS, c.IMG_CHANNELS)
model.compile(loss='categorical_crossentropy', optimizer=OPTIM, metrics=['accuracy'])

checkpoint = ModelCheckpoint(filepath=OUTPUT, save_best_only=True)

model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCHS, validation_split=VALIDATION_SPLIT, verbose=VERBOSE, callbacks=[checkpoint])
score = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=VERBOSE)

# TODO: Actually do the training






