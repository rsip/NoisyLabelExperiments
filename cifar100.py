import numpy as np
import tensorflow as tf

import tflearn
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

import random
import argparse

def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

def onehot_labels(labels):
    return np.eye(100)[labels]

def get_images(raw):
    raw_float = np.array(raw, dtype=float)
    images = raw_float.reshape([-1, 3, 32, 32])
    images = images.transpose([0, 2, 3, 1])
    return images

def load_data(path):
    # Load data
    X = get_images(unpickle(path + '/train')['data'])
    Y = onehot_labels(unpickle(path + '/train')['fine_labels'])
    X_test = get_images(unpickle(path + '/test')['data'])
    Y_test = onehot_labels(unpickle(path + '/test')['fine_labels'])
    return X, Y, X_test, Y_test

def corrupt_train_labels(Y, p): 
    # Corrupt labels with probability p
    for i in range(len(Y)):
        if (random.random() < p):
            originalTrainingLabel = np.argmax(Y[i])
            validOtherChoices = list(range(0, originalTrainingLabel)) + list(range(originalTrainingLabel + 1, 100))
            newTrainingLabel = random.choice(validOtherChoices)
            Y[i] = np.zeros(100)
            Y[i][newTrainingLabel] = 1
            # print("Original: " + str(originalTrainingLabel) + ", New: " + str(newTrainingLabel))
    return Y

def run_network(X, Y, X_test, Y_test, log_file, p):
    # Image preprocessing and augmentation
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()
    img_aug = ImageAugmentation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_rotation(max_angle=15.)

    # Define network
    net = input_data(shape=[None, 32, 32, 3], data_preprocessing=img_prep, data_augmentation=img_aug)
    net = conv_2d(net, 16, 5, strides=1, padding='same', activation='relu', bias=True, bias_init='zeros', weights_init='uniform_scaling')
    net = max_pool_2d(net, 2, strides=None, padding='same')
    net = conv_2d(net, 25, 5, strides=1, padding='same', activation='relu', bias=True, bias_init='zeros', weights_init='uniform_scaling')
    net = max_pool_2d(net, 2, strides=None, padding='same')
    net = fully_connected(net, 256, activation='relu')
    net = fully_connected(net, 100, activation='softmax')
    net = regression(net, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.001)

    # Train and test network
    model = tflearn.DNN(net, tensorboard_verbose=0)
    model.fit(X, Y, n_epoch=50, shuffle=True, validation_set=(X_test, Y_test), show_metric=True, batch_size=100, run_id='cifar100_2layer')

    # Write test accuracy to file
    accuracy = model.evaluate(X_test, Y_test)
    log_file.write("Test accuracy for p value of " + str(p) + ": " +  str(accuracy) + "\n")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('probability', type=float, help="Probability of noisy label injection")
    args = parser.parse_args()
    p = args.probability
    print("Training with probability of label corruption " + str(p))

    # Load data
    path = 'data/cifar-100-python'
    X, Y, X_test, Y_test = load_data(path)

    # Run with noise injection probability p
    log_file = open('trainlog.txt', 'w')
    Y_noisy = corrupt_train_labels(Y, p)
    run_network(X, Y_noisy, X_test, Y_test, log_file, p)
    # for p in np.arange(0.1, 1.1, 0.1):
    #     Y_noisy = corrupt_train_labels(Y, p)
    #     run_network(X, Y_noisy, X_test, Y_test, log_file, p)
    log_file.close()

if __name__ == "__main__":
    main()
