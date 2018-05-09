# imports
import os
import numpy as np
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split

# constants
DATA_DIR = 'Data'
DATA_DIR_TEST = DATA_DIR + '/test'
DATA_DIR_TRAINING = DATA_DIR + '/train'
IMG_ROWS = 1613
IMG_COLS = 1050
IMG_CHANNELS = 3


def prepare_data(kind=0):
    train_index = pd.read_csv(DATA_DIR_TRAINING + '/train.csv').set_index('Image')

    data = {}
    if kind == 0:
        """returns a dictionary of label: image_array"""
        for each_file in os.listdir(DATA_DIR_TRAINING + '/training_images'):
            for each_index in train_index.index:
                if each_file == each_index:
                    index = train_index.loc[each_index].Id
                    image = cv2.imread(DATA_DIR_TRAINING + '/training_images/' + each_file)
                    data[index] = np.array(cv2.resize(image, (IMG_ROWS, IMG_COLS)))
        return data

    elif kind == 1:
        """returns a dictionary of image_name: image_array"""
        for each_file in os.listdir(DATA_DIR_TEST + '/test_images'):
            image = cv2.imread(DATA_DIR_TEST + '/test_images/' + each_file)
            data[each_file] = np.array(cv2.resize(image, (IMG_ROWS, IMG_COLS)))
        return data

    else:
        raise Exception("Please select a valid choice, 0 => training data, 1 => test data")


def gen_label_vectors():
    train_index = pd.read_csv(DATA_DIR_TRAINING + '/train.csv').set_index('Image')
    unique_ids = train_index.drop_duplicates(subset=['Id']).get_values()
    unique_ids = unique_ids.reshape((len(unique_ids)))  # not sure if the reshape is going to work

    labels = np.zeros((len(unique_ids), ), dtype=list)

    for j, each_name in enumerate(unique_ids):
        label = np.zeros((len(unique_ids)), dtype=int)
        for i, id in enumerate(unique_ids):
            if id == each_name:
                label[i] = 1
        labels[j] = label

    indices = train_index.drop_duplicates().Id
    labels_to_array = {indices[i]: labels[i] for i in range(len(labels))}

    return labels, labels_to_array # no need for returning the labels array


def gen_training_data(test_size=0.2):
    """This method returns two tuples : (X_train, y_train), (X_test, y_test)
        It requires a test_size to use when splitting the data-set into test and training sets"""
    data = prepare_data()
    labels, labels_to_array = gen_label_vectors()

    x_data = [] # store the images
    for _, x in data.items():
        x_data.append(x)

    one_hot_labels = []
    for key in data.keys():
        for label, vec_label in labels_to_array.items():
            if key == label:
                one_hot_labels.append(vec_label)

    X_train, X_test = train_test_split(x_data, test_size=test_size)
    y_train, y_test = train_test_split(one_hot_labels, test_size=test_size)

    return (X_train, y_train), (X_test, y_test)


