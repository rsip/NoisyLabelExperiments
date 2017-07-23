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

import noisy_labels.models.inception as inception

def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

def onehot_labels(labels, num_classes):
    return np.eye(num_classes)[labels]

def get_images(raw):
    raw_float = np.array(raw, dtype=float)
    images = raw_float.reshape([-1, 3, 32, 32])
    images = images.transpose([0, 2, 3, 1])
    return images

def load_mnist():
    path = 'data/mnist'
    from tensorflow.examples.tutorials.mnist import mnist
    data = mnist.read_data_sets(path, one_hot=True)
    X = mnist.train.images
    Y = mnist.train.labels
    X_test = mnist.test.images
    Y_test = mnist.test.labels
    return X, Y, X_test, Y_test

def load_cifar10():
    path = 'data/cifar-10-batches-py'
    data = [unpickle(path + '/data_batch_' + str(i)) for i in range(1, 5)]
    X = np.concatenate([get_images(batch['data']) for batch in data])
    Y = np.concatenate([onehot_labels(batch['labels'], 10) for batch in data])
    X_test = get_images(unpickle(path + '/test_batch')['data'])
    Y_test = onehot_labels(unpickle(path + '/test_batch')['labels'], 10)
    return X, Y, X_test, Y_test

def load_cifar100():
    path = 'data/cifar-100-python'
    X = get_images(unpickle(path + '/train')['data'])
    Y = onehot_labels(unpickle(path + '/train')['fine_labels'], 100)
    X_test = get_images(unpickle(path + '/test')['data'])
    Y_test = onehot_labels(unpickle(path + '/test')['fine_labels'], 100)
    return X, Y, X_test, Y_test
    
def corrupt_train_labels(Y, p, num_classes): 
    # Corrupt labels with probability p
    for i in range(len(Y)):
        if (random.random() < p):
            originalTrainingLabel = np.argmax(Y[i])
            validOtherChoices = list(range(0, originalTrainingLabel)) + list(range(originalTrainingLabel + 1, num_classes))
            newTrainingLabel = random.choice(validOtherChoices)
            Y[i] = np.zeros(num_classes)
            Y[i][newTrainingLabel] = 1
            #print("Original: " + str(originalTrainingLabel) + ", New: " + str(newTrainingLabel))
    return Y

def run_network(X, Y, X_test, Y_test, p, model_type, dataset):
    # Image preprocessing and augmentation
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()
    img_aug = ImageAugmentation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_rotation(max_angle=15.)

    # Define the network
    if model_type == 'shallow_cnn':
        net = input_data(shape=[None, 32, 32, 3], data_preprocessing=img_prep, data_augmentation=img_aug)
        net = conv_2d(net, 16, 5, strides=1, padding='same', activation='relu', bias=True, bias_init='zeros', weights_init='uniform_scaling')
        net = max_pool_2d(net, 2, strides=None, padding='same')
        net = conv_2d(net, 25, 5, strides=1, padding='same', activation='relu', bias=True, bias_init='zeros', weights_init='uniform_scaling')
        net = max_pool_2d(net, 2, strides=None, padding='same')
        if dataset == 'mnist' or dataset == 'cifar10':
            net = fully_connected(net, 32, activation='relu')
            net = fully_connected(net, 10, activation='softmax')
        elif dataset == 'cifar100':
            net = fully_connected(net, 256, activation='relu')
            net = fully_connected(net, 100, activation='softmax')
        net = regression(net, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.001)

    elif model_type == 'inception_v3':
        if dataset == 'mnist' or dataset == 'cifar10':
            net = inception.get_model(10)
        elif dataset == 'cifar100':
            net = inception.get_model(100)

    else:
        return

    # Train network
    model = tflearn.DNN(net, tensorboard_verbose=0)
    model.fit(X, Y, n_epoch=50, shuffle=True, validation_set=(X_test, Y_test), show_metric=True, batch_size=100, run_id='noisylabel')

    # Output test accuracy
    accuracy = model.evaluate(X_test, Y_test)
    print("Test accuracy: " + str(accuracy))

def main():
    # Add arguments to parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--probability', type=float, required=True, help="Probability of noisy label injection")
    parser.add_argument('-m', '--model', type=str, required=True,  help="Model architecture")
    parser.add_argument('-d', '--dataset', type=str, required=True, help="Dataset to load")

    # Parse arguments
    args = parser.parse_args()
    p = args.probability
    dataset = args.dataset
    model = args.model
    print("Probability of label corruption: " + str(p))
    print("Model: " + model)
    print("Dataset: " + dataset)

    # Load data
    if dataset == 'cifar10':
        X, Y, X_test, Y_test = load_cifar10()
        Y_noisy = corrupt_train_labels(Y, p, 10)
    elif dataset == 'cifar100':
        X, Y, X_test, Y_test = load_cifar100()
        Y_noisy = corrupt_train_labels(Y, p, 100)
    else:
        return

    # Run with noise injection probability p
    run_network(X, Y_noisy, X_test, Y_test, p, model, dataset)

if __name__ == "__main__":
    main()
