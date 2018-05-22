# imports
import os
import numpy as np
import cv2
import pandas as pd
import constants as c
from sklearn.model_selection import train_test_split

# constants
DATA_DIR = 'Data'
DATA_DIR_TEST = DATA_DIR + '/test'
DATA_DIR_TRAINING = DATA_DIR + '/train'


def prepare_data(kind=0):
    train_index = pd.read_csv(DATA_DIR_TRAINING + '/train.csv').set_index('Image')

    data = []
    if kind == 0:
        """returns a list of tuples (label, image_array)"""
        
        for each_file in os.listdir(DATA_DIR_TRAINING + '/training_images'):
            index = train_index.loc[each_file].Id
            image = np.array(cv2.imread(DATA_DIR_TRAINING + '/training_images/' + each_file))
            my_tup = (index, image)
            data.append(my_tup)

        return data

    elif kind == 1:
        """returns a dictionary of image_name: image_array"""
        for each_file in os.listdir(DATA_DIR_TEST + '/test_images'):
            image = cv2.imread(DATA_DIR_TEST + '/test_images/' + each_file)
            data.append(np.array(cv2.resize(image, (c.IMG_ROWS, c.IMG_COLS))))
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
    """This method returns two tuples : (X_train, y_train), (X_test, y_test) and the number of examples
        It requires a test_size to use when splitting the data-set into test and training sets"""
    data = prepare_data()
    labels, labels_to_array = gen_label_vectors()

    x_data = np.zeros((len(data), ), dtype=list) # store the images
    i = 0
    for index, image in data:
        x_data[i] = image
        i += 1

    one_hot_labels = np.zeros((len(data), ), dtype=list)
    j = 0
    for index, image in data:
        for label, vec_label in labels_to_array.items():
            if index == label:
                one_hot_labels[j] = vec_label
                j += 1

    X_train, X_test = train_test_split(x_data, test_size=test_size)
    y_train, y_test = train_test_split(one_hot_labels, test_size=test_size)

    return (X_train, y_train), (X_test, y_test)

# TODO: Create more data by using transformations
# TODO: use os functions instead of hardcoding the directory urls


