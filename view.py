import matplotlib.pyplot as plt
from skimage.io import imshow
import pickle
import os
import numpy as np


def load_data():
    with open(os.path.join('data', 'train.pkl'), mode='rb') as train_file:
        train = pickle.load(train_file)
    with open(os.path.join('data', 'test.pkl'), mode='rb') as test_file:
        test = pickle.load(test_file)
    return train, test


if __name__ == '__main__':
    train, test = load_data()

    imshow(train['x'][10])
    plt.show()
    imshow(np.squeeze(train['y'][10].astype(np.uint8)))
    plt.show()
