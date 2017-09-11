import numpy as np
import tensorflow as tf
tf.reset_default_graph()

import tflearn
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, fully_connected, dropout
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression

import random
import argparse
from sklearn.metrics import confusion_matrix
import pickle, pprint

from dataloader import DataLoader
    
def rand_corrupt(Y, p, num_classes): 
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

def least_confusing(Y, p, dataset, num_classes):
    # Corrupt labels with probability p to 3 least confusable classes
    if dataset == 'cifar10':
        cm_file = open('confusion_matrices/cifar10_cm.pkl', 'rb')
    elif dataset == 'cifar100':
        cm_file = open('confusion_matrices/cifar100_cm.pkl', 'rb')
    else:
        return
    cm = pickle.load(cm_file)
    pprint.pprint(cm)
    cm_file.close()

    # Get 3 least confusable CM
    least_confusing = []
    for row in cm:
        least = np.argsort(row)[:3]
        print(least)
        least_confusing.append(least)

    # Inject noise
    for i in range(len(Y)):
        if (random.random() < p):
            originalTrainingLabel = np.argmax(Y[i])
            validOtherChoices = least_confusing[originalTrainingLabel]
            newTrainingLabel = random.choice(validOtherChoices)
            Y[i] = np.zeros(num_classes)
            Y[i][newTrainingLabel] = 1
            print("Original: " + str(originalTrainingLabel) + ", New: " + str(newTrainingLabel))
    return Y

def most_confusing(Y, p, dataset, num_classes):
    # Corrupt labels with probability p to 3 least confusable classes
    if dataset == 'cifar10':
        cm_file = open('confusion_matrices/cifar10_cm.pkl', 'rb')
    elif dataset == 'cifar100':
        cm_file = open('confusion_matrices/cifar100_cm.pkl', 'rb')
    else:
        return
    cm = pickle.load(cm_file)
    pprint.pprint(cm)
    cm_file.close()

    # Get 3 least confusable CM
    most_confusing = []
    for idx, row in enumerate(cm):
        print ('current index :' + str(idx))
        most = np.argsort(-row)[:4]
        print(most)
        if idx in most:
            rm = np.argwhere(most == idx)
            most = np.delete(most, rm)
        else:
            most = most[:3]
        print(most)
        most_confusing.append(most)

    # Inject noise
    for i in range(len(Y)):
        if (random.random() < p):
            originalTrainingLabel = np.argmax(Y[i])
            validOtherChoices = most_confusing[originalTrainingLabel]
            newTrainingLabel = random.choice(validOtherChoices)
            Y[i] = np.zeros(num_classes)
            Y[i][newTrainingLabel] = 1
            print("Original: " + str(originalTrainingLabel) + ", New: " + str(newTrainingLabel))
    return Y

def run_network(X, Y, X_test, Y_test, p, model_type, dataset):
    # Image preprocessing and augmentation
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()
    img_aug = ImageAugmentation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_rotation(max_angle=15.)

    # Run the network
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
        
        # Train network
        model = tflearn.DNN(net, tensorboard_verbose=0)
        model.fit(X, Y, n_epoch=50, shuffle=True, validation_set=(X_test, Y_test), show_metric=True, batch_size=100, run_id='noisylabel')

    elif model_type == 'inception_v3':
        if dataset == 'cifar10' or dataset == 'cifar100':
            network = input_data(shape=[None, 32, 32, 3])
        elif dataset == 'mnist':
            network = input_data(shape=[None, 28, 28, 1])
        conv1_7_7 = conv_2d(network, 64, 7, strides=2, activation='relu', name = 'conv1_7_7_s2')
        pool1_3_3 = max_pool_2d(conv1_7_7, 3,strides=2)
        pool1_3_3 = local_response_normalization(pool1_3_3)
        conv2_3_3_reduce = conv_2d(pool1_3_3, 64,1, activation='relu',name = 'conv2_3_3_reduce')
        conv2_3_3 = conv_2d(conv2_3_3_reduce, 192,3, activation='relu', name='conv2_3_3')
        conv2_3_3 = local_response_normalization(conv2_3_3)
        pool2_3_3 = max_pool_2d(conv2_3_3, kernel_size=3, strides=2, name='pool2_3_3_s2')
        inception_3a_1_1 = conv_2d(pool2_3_3, 64, 1, activation='relu', name='inception_3a_1_1')
        inception_3a_3_3_reduce = conv_2d(pool2_3_3, 96,1, activation='relu', name='inception_3a_3_3_reduce')
        inception_3a_3_3 = conv_2d(inception_3a_3_3_reduce, 128,filter_size=3,  activation='relu', name = 'inception_3a_3_3')
        inception_3a_5_5_reduce = conv_2d(pool2_3_3,16, filter_size=1,activation='relu', name ='inception_3a_5_5_reduce' )
        inception_3a_5_5 = conv_2d(inception_3a_5_5_reduce, 32, filter_size=5, activation='relu', name= 'inception_3a_5_5')
        inception_3a_pool = max_pool_2d(pool2_3_3, kernel_size=3, strides=1, )
        inception_3a_pool_1_1 = conv_2d(inception_3a_pool, 32, filter_size=1, activation='relu', name='inception_3a_pool_1_1')

        # merge the inception_3a__
        inception_3a_output = merge([inception_3a_1_1, inception_3a_3_3, inception_3a_5_5, inception_3a_pool_1_1], mode='concat', axis=3)

        inception_3b_1_1 = conv_2d(inception_3a_output, 128,filter_size=1,activation='relu', name= 'inception_3b_1_1' )
        inception_3b_3_3_reduce = conv_2d(inception_3a_output, 128, filter_size=1, activation='relu', name='inception_3b_3_3_reduce')
        inception_3b_3_3 = conv_2d(inception_3b_3_3_reduce, 192, filter_size=3,  activation='relu',name='inception_3b_3_3')
        inception_3b_5_5_reduce = conv_2d(inception_3a_output, 32, filter_size=1, activation='relu', name = 'inception_3b_5_5_reduce')
        inception_3b_5_5 = conv_2d(inception_3b_5_5_reduce, 96, filter_size=5,  name = 'inception_3b_5_5')
        inception_3b_pool = max_pool_2d(inception_3a_output, kernel_size=3, strides=1,  name='inception_3b_pool')
        inception_3b_pool_1_1 = conv_2d(inception_3b_pool, 64, filter_size=1,activation='relu', name='inception_3b_pool_1_1')

        #merge the inception_3b_*
        inception_3b_output = merge([inception_3b_1_1, inception_3b_3_3, inception_3b_5_5, inception_3b_pool_1_1], mode='concat',axis=3,name='inception_3b_output')

        pool3_3_3 = max_pool_2d(inception_3b_output, kernel_size=3, strides=2, name='pool3_3_3')
        inception_4a_1_1 = conv_2d(pool3_3_3, 192, filter_size=1, activation='relu', name='inception_4a_1_1')
        inception_4a_3_3_reduce = conv_2d(pool3_3_3, 96, filter_size=1, activation='relu', name='inception_4a_3_3_reduce')
        inception_4a_3_3 = conv_2d(inception_4a_3_3_reduce, 208, filter_size=3,  activation='relu', name='inception_4a_3_3')
        inception_4a_5_5_reduce = conv_2d(pool3_3_3, 16, filter_size=1, activation='relu', name='inception_4a_5_5_reduce')
        inception_4a_5_5 = conv_2d(inception_4a_5_5_reduce, 48, filter_size=5,  activation='relu', name='inception_4a_5_5')
        inception_4a_pool = max_pool_2d(pool3_3_3, kernel_size=3, strides=1,  name='inception_4a_pool')
        inception_4a_pool_1_1 = conv_2d(inception_4a_pool, 64, filter_size=1, activation='relu', name='inception_4a_pool_1_1')

        inception_4a_output = merge([inception_4a_1_1, inception_4a_3_3, inception_4a_5_5, inception_4a_pool_1_1], mode='concat', axis=3, name='inception_4a_output')

        inception_4b_1_1 = conv_2d(inception_4a_output, 160, filter_size=1, activation='relu', name='inception_4a_1_1')
        inception_4b_3_3_reduce = conv_2d(inception_4a_output, 112, filter_size=1, activation='relu', name='inception_4b_3_3_reduce')
        inception_4b_3_3 = conv_2d(inception_4b_3_3_reduce, 224, filter_size=3, activation='relu', name='inception_4b_3_3')
        inception_4b_5_5_reduce = conv_2d(inception_4a_output, 24, filter_size=1, activation='relu', name='inception_4b_5_5_reduce')
        inception_4b_5_5 = conv_2d(inception_4b_5_5_reduce, 64, filter_size=5,  activation='relu', name='inception_4b_5_5')

        inception_4b_pool = max_pool_2d(inception_4a_output, kernel_size=3, strides=1,  name='inception_4b_pool')
        inception_4b_pool_1_1 = conv_2d(inception_4b_pool, 64, filter_size=1, activation='relu', name='inception_4b_pool_1_1')

        inception_4b_output = merge([inception_4b_1_1, inception_4b_3_3, inception_4b_5_5, inception_4b_pool_1_1], mode='concat', axis=3, name='inception_4b_output')

        inception_4c_1_1 = conv_2d(inception_4b_output, 128, filter_size=1, activation='relu',name='inception_4c_1_1')
        inception_4c_3_3_reduce = conv_2d(inception_4b_output, 128, filter_size=1, activation='relu', name='inception_4c_3_3_reduce')
        inception_4c_3_3 = conv_2d(inception_4c_3_3_reduce, 256,  filter_size=3, activation='relu', name='inception_4c_3_3')
        inception_4c_5_5_reduce = conv_2d(inception_4b_output, 24, filter_size=1, activation='relu', name='inception_4c_5_5_reduce')
        inception_4c_5_5 = conv_2d(inception_4c_5_5_reduce, 64,  filter_size=5, activation='relu', name='inception_4c_5_5')

        inception_4c_pool = max_pool_2d(inception_4b_output, kernel_size=3, strides=1)
        inception_4c_pool_1_1 = conv_2d(inception_4c_pool, 64, filter_size=1, activation='relu', name='inception_4c_pool_1_1')

        inception_4c_output = merge([inception_4c_1_1, inception_4c_3_3, inception_4c_5_5, inception_4c_pool_1_1], mode='concat', axis=3,name='inception_4c_output')

        inception_4d_1_1 = conv_2d(inception_4c_output, 112, filter_size=1, activation='relu', name='inception_4d_1_1')
        inception_4d_3_3_reduce = conv_2d(inception_4c_output, 144, filter_size=1, activation='relu', name='inception_4d_3_3_reduce')
        inception_4d_3_3 = conv_2d(inception_4d_3_3_reduce, 288, filter_size=3, activation='relu', name='inception_4d_3_3')
        inception_4d_5_5_reduce = conv_2d(inception_4c_output, 32, filter_size=1, activation='relu', name='inception_4d_5_5_reduce')
        inception_4d_5_5 = conv_2d(inception_4d_5_5_reduce, 64, filter_size=5,  activation='relu', name='inception_4d_5_5')
        inception_4d_pool = max_pool_2d(inception_4c_output, kernel_size=3, strides=1,  name='inception_4d_pool')
        inception_4d_pool_1_1 = conv_2d(inception_4d_pool, 64, filter_size=1, activation='relu', name='inception_4d_pool_1_1')

        inception_4d_output = merge([inception_4d_1_1, inception_4d_3_3, inception_4d_5_5, inception_4d_pool_1_1], mode='concat', axis=3, name='inception_4d_output')

        inception_4e_1_1 = conv_2d(inception_4d_output, 256, filter_size=1, activation='relu', name='inception_4e_1_1')
        inception_4e_3_3_reduce = conv_2d(inception_4d_output, 160, filter_size=1, activation='relu', name='inception_4e_3_3_reduce')
        inception_4e_3_3 = conv_2d(inception_4e_3_3_reduce, 320, filter_size=3, activation='relu', name='inception_4e_3_3')
        inception_4e_5_5_reduce = conv_2d(inception_4d_output, 32, filter_size=1, activation='relu', name='inception_4e_5_5_reduce')
        inception_4e_5_5 = conv_2d(inception_4e_5_5_reduce, 128,  filter_size=5, activation='relu', name='inception_4e_5_5')
        inception_4e_pool = max_pool_2d(inception_4d_output, kernel_size=3, strides=1,  name='inception_4e_pool')
        inception_4e_pool_1_1 = conv_2d(inception_4e_pool, 128, filter_size=1, activation='relu', name='inception_4e_pool_1_1')

        inception_4e_output = merge([inception_4e_1_1, inception_4e_3_3, inception_4e_5_5,inception_4e_pool_1_1],axis=3, mode='concat')

        pool4_3_3 = max_pool_2d(inception_4e_output, kernel_size=3, strides=2, name='pool_3_3')

        inception_5a_1_1 = conv_2d(pool4_3_3, 256, filter_size=1, activation='relu', name='inception_5a_1_1')
        inception_5a_3_3_reduce = conv_2d(pool4_3_3, 160, filter_size=1, activation='relu', name='inception_5a_3_3_reduce')
        inception_5a_3_3 = conv_2d(inception_5a_3_3_reduce, 320, filter_size=3, activation='relu', name='inception_5a_3_3')
        inception_5a_5_5_reduce = conv_2d(pool4_3_3, 32, filter_size=1, activation='relu', name='inception_5a_5_5_reduce')
        inception_5a_5_5 = conv_2d(inception_5a_5_5_reduce, 128, filter_size=5,  activation='relu', name='inception_5a_5_5')
        inception_5a_pool = max_pool_2d(pool4_3_3, kernel_size=3, strides=1,  name='inception_5a_pool')
        inception_5a_pool_1_1 = conv_2d(inception_5a_pool, 128, filter_size=1,activation='relu', name='inception_5a_pool_1_1')

        inception_5a_output = merge([inception_5a_1_1, inception_5a_3_3, inception_5a_5_5, inception_5a_pool_1_1], axis=3,mode='concat')

        inception_5b_1_1 = conv_2d(inception_5a_output, 384, filter_size=1,activation='relu', name='inception_5b_1_1')
        inception_5b_3_3_reduce = conv_2d(inception_5a_output, 192, filter_size=1, activation='relu', name='inception_5b_3_3_reduce')
        inception_5b_3_3 = conv_2d(inception_5b_3_3_reduce, 384,  filter_size=3,activation='relu', name='inception_5b_3_3')
        inception_5b_5_5_reduce = conv_2d(inception_5a_output, 48, filter_size=1, activation='relu', name='inception_5b_5_5_reduce')
        inception_5b_5_5 = conv_2d(inception_5b_5_5_reduce,128, filter_size=5,  activation='relu', name='inception_5b_5_5' )
        inception_5b_pool = max_pool_2d(inception_5a_output, kernel_size=3, strides=1,  name='inception_5b_pool')
        inception_5b_pool_1_1 = conv_2d(inception_5b_pool, 128, filter_size=1, activation='relu', name='inception_5b_pool_1_1')
        inception_5b_output = merge([inception_5b_1_1, inception_5b_3_3, inception_5b_5_5, inception_5b_pool_1_1], axis=3, mode='concat')

        pool5_7_7 = avg_pool_2d(inception_5b_output, kernel_size=7, strides=1)
        pool5_7_7 = dropout(pool5_7_7, 0.4)

        if dataset == 'mnist' or dataset == 'cifar10':
            loss = fully_connected(pool5_7_7, 10, activation='softmax')
        elif dataset == 'cifar100':
            loss = fully_connected(pool5_7_7, 100, activation='softmax')

        net = regression(loss, optimizer='momentum',
                            loss='categorical_crossentropy',
                            learning_rate=0.001)

        # Train network
        model = tflearn.DNN(net, tensorboard_verbose=0)
        model.fit(X, Y, n_epoch=1000, shuffle=True, validation_set=(X_test, Y_test), show_metric=True, batch_size=100, run_id='noisylabel')

    else:
        return
        
    # Output test accuracy
    accuracy = model.evaluate(X_test, Y_test)
    print("Test accuracy: " + str(accuracy))

    # Write to file
    with open('tmp/record.txt', 'a') as file:
        file.write(str(p) + ": " + str(accuracy))

    # Compute confusion matrix
    # Y_test = np.argmax(Y_test, axis=1)
    # Y_pred = []
    # for x in X_test:
    #     y = model.predict([x])
    #     Y_pred.append(np.argmax(y[0]))
    # cm = confusion_matrix(Y_test, Y_pred)
    # if dataset == 'mnist' or dataset == 'cifar10':
    #     labels = [str(i) for i in range(10)]
    # if dataset == 'cifar100':
    #     labels = [str(i) for i in range(100)]
    # print_confusion_matrix(cm, labels)
    # cm_file = open('cm.pkl', 'wb')
    # pickle.dump(cm, cm_file)
    # cm_file.close()

# def print_confusion_matrix(cm, labels):
#     column_width = max([len(x) for x in labels] + [5])
#     empty_cell = " " * column_width
#
#     # Print header
#     print "   " + empty_cell,
#     for label in labels:
#         print "%{0}s".format(column_width) % label,
#     print
#
#     # Print rows
#     for i, label in enumerate(labels):
#         print "  %{0}s".format(column_width) % label,
#         for j in range(len(labels)):
#             cell = "%{0}.1f".format(column_width) % cm[i, j]
#             print cell,
#         print

def main():
    # Add arguments to parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--probability', type=float, required=True, help="Probability of noisy label injection")
    parser.add_argument('-m', '--model', type=str, required=True, help="Model architecture")
    parser.add_argument('-d', '--dataset', type=str, required=True, help="Dataset to load")
    parser.add_argument('-e', '--experiment', type=str, required=True, help="Noisy label experiment to run")

    # Parse arguments
    args = parser.parse_args()
    p = args.probability
    dataset = args.dataset
    model = args.model
    experiment = args.experiment
    print("Probability of label corruption: " + str(p))
    print("Model: " + model)
    print("Dataset: " + dataset)

    # Load data
    dataLoader = DataLoader()
    if dataset == 'mnist':
        X, Y, X_test, Y_test = dataLoader.load_mnist()
        num_classes = 10
    elif dataset == 'cifar10':
        X, Y, X_test, Y_test = dataLoader.load_cifar10()
        num_classes = 10
    elif dataset == 'cifar100':
        X, Y, X_test, Y_test = dataLoader.load_cifar100()
        num_classes = 100
    else:
        return

    # Inject noisy labels according to experiment type
    if experiment == 'rand_corrupt':
        Y_noisy = rand_corrupt(Y, p, num_classes)
    elif experiment == 'least_confusing':
        Y_noisy = least_confusing(Y, p, dataset, num_classes)
    elif experiment == 'most_confusing':
        Y_noisy = most_confusing(Y, p, dataset, num_classes)
    else:
        return

    # Run with noise injection probability p
    run_network(X, Y_noisy, X_test, Y_test, p, model, dataset)

if __name__ == "__main__":
    main()
