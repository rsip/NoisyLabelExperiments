from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import random
import sys
import cmd_args
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

class DataSet(object):

    def __init__(self, images, labels):
        assert images.shape[0] == labels.shape[0], ('images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        # Shuffle for first epoch
        if self._epochs_completed == 0 and start == 0:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._images = self._images[perm0]
            self._labels = self._labels[perm0]
        # Go to next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get rest of examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((images_rest_part, images_new_part), axis=0), np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]

def main():
    # Parse command line arguments
    args = cmd_args.parse_args()

    # Import data
    mnist = input_data.read_data_sets("../data/mnist", one_hot=True)
    print(mnist.train.labels.shape)

    experiment_type = int(args.experiment_type)

    # Change labels to random other label with probability p
    if experiment_type == 1:
        p = float(args.change_label_prob)
        for i in range(len(mnist.train.labels)):
            if (random.random() < p):
                originalTrainingLabel = np.argmax(mnist.train.labels[i])
                validOtherChoices = list(range(0, originalTrainingLabel)) + list(range(originalTrainingLabel + 1, 10))
                newTrainingLabel = random.choice(validOtherChoices)
                mnist.train.labels[i] = np.zeros(10)
                mnist.train.labels[i][newTrainingLabel] = 1
        mnist_train = DataSet(mnist.train.images, mnist.train.labels)

    # Remove p percentage of training data
    elif experiment_type == 2: 
        p = float(args.change_label_prob)
        num = int(round(len(mnist.train.labels) * (1 - p)))
        train_x = mnist.train.images[:num, :]
        train_y = mnist.train.labels[:num, :]
        print("Size of MNIST training set = " + str(train_x.shape[0]))
        mnist_train = DataSet(train_x, train_y)

    # Change labels to single other label with probability p
    elif experiment_type == 3:
        p = float(args.change_label_prob)
        print("Change label probability of " + str(p))
        otherTrainingLabel = int(args.other_training_label)
        print("Other training label set to " + str(otherTrainingLabel))
        for i in range(len(mnist.train.labels)):
            if (random.random() < p):
                mnist.train.labels[i] = np.zeros(10)
                mnist.train.labels[i][otherTrainingLabel] = 1
        mnist_train = DataSet(mnist.train.images, mnist.train.labels)

    # With probability p, change labels to another label randomly selected from set of three closest labels (determined by confusion matrix)
    elif experiment_type == 4:
        # MNIST confusion matrix (valid other choices for experiment 4)
        confusionMatrix = [[3, 5, 8],   # Actual 0
                            [2, 5, 8],  # Actual 1
                            [3, 7, 8],  # Actual 2
                            [5, 7, 8],  # Actual 3
                            [1, 5, 9],  # Actual 4
                            [3, 6, 8],  # Actual 5
                            [0, 5, 8],  # Actual 6
                            [1, 4, 9],  # Actual 7
                            [1, 3, 5],  # Actual 8
                            [4, 5, 7]]  # Actual 9

        p = float(args.change_label_prob)
        for i in range(len(mnist.train.labels)):
            if (random.random() < p):
                originalTrainingLabel = np.argmax(mnist.train.labels[i])
                # print("Original training label = " + str(originalTrainingLabel))
                validOtherChoices = confusionMatrix[originalTrainingLabel]
                newTrainingLabel = random.choice(validOtherChoices)
                # print("New training label = " + str(newTrainingLabel))
                mnist.train.labels[i] = np.zeros(10)
                mnist.train.labels[i][newTrainingLabel] = 1
        mnist_train = DataSet(mnist.train.images, mnist.train.labels)

     # With probability p, change labels to another label randomly selected from set of three furthest labels (determined by confusion matrix)
    elif experiment_type == 5:
        # MNIST confusion matrix (valid other choices for experiment 5)
        confusionMatrix = [[1, 2, 4],   # Actual 0
                            [0, 4, 6],  # Actual 1
                            [4, 5, 9],  # Actual 2
                            [0, 4, 6],  # Actual 3
                            [0, 2, 7],  # Actual 4
                            [1, 4, 9],  # Actual 5
                            [3, 7, 9],  # Actual 6
                            [0, 6, 8],  # Actual 7
                            [0, 4, 6],  # Actual 8
                            [1, 2, 6]]  # Actual 9

        p = float(args.change_label_prob)
        for i in range(len(mnist.train.labels)):
            if (random.random() < p):
                originalTrainingLabel = np.argmax(mnist.train.labels[i])
                # print("Original training label = " + str(originalTrainingLabel))
                validOtherChoices = confusionMatrix[originalTrainingLabel]
                newTrainingLabel = random.choice(validOtherChoices)
                # print("New training label = " + str(newTrainingLabel))
                mnist.train.labels[i] = np.zeros(10)
                mnist.train.labels[i][newTrainingLabel] = 1
        mnist_train = DataSet(mnist.train.images, mnist.train.labels)
 
    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])

    # Training parameters
    stddev = 0.1

    # First convolutional layer
    W_conv1 = tf.get_variable("W_conv1", [5, 5, 1, 16], tf.float32, initializer=tf.truncated_normal_initializer(stddev=stddev))
    b_conv1 = tf.Variable(tf.constant(0.1, shape=[16], name="b_conv1"))
    
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Second convolutional layer
    W_conv2 = tf.get_variable("W_conv2", [5, 5, 16, 25], tf.float32, initializer=tf.truncated_normal_initializer(stddev=stddev))
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[25], name="b_conv2"))
    
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Fully connected layer
    W_fc1 = tf.get_variable("W_fc1", [7 * 7 * 25, 32], tf.float32, initializer=tf.truncated_normal_initializer(stddev=stddev))
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[32], name="b_fc1"))
    
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 25])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Readout layer
    W_fc2 = tf.get_variable("W_fc2", [32, 10], tf.float32, initializer=tf.truncated_normal_initializer(stddev=stddev))
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[10], name="b_fc2"))

    y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

    # Train and evaluate model
    with tf.Session() as sess:
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        sess.run(tf.global_variables_initializer())
        for i in range(20000):
            batch_x, batch_y = mnist_train.next_batch(50) 
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batch_x, y_: batch_y})
                print("step %d, training accuracy %g" % (i, train_accuracy))
            train_step.run(feed_dict={x: batch_x, y_: batch_y})

        print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

if __name__ == '__main__':
    main()
