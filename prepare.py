import numpy as np
import os
import progressbar
import pickle
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize

IMG_WIDTH = 128  # for faster computing on kaggle
IMG_HEIGHT = 128  # for faster computing on kaggle
IMG_CHANNELS = 3
TRAIN_PATH = 'input/stage1_train'
TEST_PATH = 'input/stage1_test'

train_image_ids = next(os.walk(TRAIN_PATH))[1]
test_image_ids = next(os.walk(TEST_PATH))[1]

X_train = np.zeros([len(train_image_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS], dtype=np.uint8)
Y_train = np.zeros([len(train_image_ids), IMG_HEIGHT, IMG_WIDTH, 1], dtype=np.bool)

# load train image
train_bar = progressbar.ProgressBar(max_value=len(train_image_ids))
train_bar.start()
for index, id_ in enumerate(train_image_ids):
    train_bar.update(index)
    path = os.path.join(TRAIN_PATH, id_)
    img = imread(os.path.join(path, 'images', id_ + '.png'))[:, :, :IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), mode='constant', preserve_range=True)
    X_train[index] = img
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',
                                      preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)
    Y_train[index] = mask
train_bar.finish()

# load test image
X_test = np.zeros((len(test_image_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
test_sizes = []
test_bar = progressbar.ProgressBar(max_value=len(test_image_ids))
test_bar.start()
for index, id_ in enumerate(test_image_ids):
    path = os.path.join(TEST_PATH, id_)
    img = imread(os.path.join(path, 'images', id_ + '.png'))[:, :, :IMG_CHANNELS]
    test_sizes.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[index] = img
    test_bar.update(index)
test_bar.finish()

# dump data

if not os.path.exists('data'):
    os.mkdir('data')

train = {'x': X_train, 'y': Y_train}
test = {'x': X_test, 'size': test_sizes}

with open(os.path.join('data', 'train.pkl'), mode='wb+') as train_file:
    pickle.dump(train, train_file)
with open(os.path.join('data', 'test.pkl'), mode='wb+') as test_file:
    pickle.dump(test, test_file)

