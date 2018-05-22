# imports
from network import CNN
import data_prep as dp
import constants as c
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint

# constants
NB_CLASSES = 4251
OPTIM = RMSprop()
NB_EPOCHS = 20
VALIDATION_SPLIT = 0.2
VERBOSE = 0

# data
(X_train, y_train), (X_test, y_test) = dp.gen_training_data()
# X_train /= 255   # normalization
# X_test /= 255

# training
model = CNN.build(NB_CLASSES, c.IMG_ROWS, c.IMG_COLS, c.IMG_CHANNELS)
model.compile(loss='categorical_crossentropy', optimizer=OPTIM, metrics=['accuracy'])

checkpoint = ModelCheckpoint(filepath='model.h5', save_best_only=True)

model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCHS, validation_split=VALIDATION_SPLIT, verbose=VERBOSE, callbacks=[checkpoint])
score = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=VERBOSE)

# TODO: Actually do the training
# TODO: Save the model in a h5 file





