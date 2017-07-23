import numpy as np

class DataLoader():
    def unpickle(self, file):
        import cPickle
        with open(file, 'rb') as fo:
            dict = cPickle.load(fo)
        return dict

    def onehot_labels(self, labels, num_classes):
        return np.eye(num_classes)[labels]

    def get_images(self, raw):
        raw_float = np.array(raw, dtype=float)
        images = raw_float.reshape([-1, 3, 32, 32])
        images = images.transpose([0, 2, 3, 1])
        return images

    def load_mnist(self):
        path = 'data/mnist'
        from tensorflow.examples.tutorials.mnist import mnist
        data = mnist.read_data_sets(path, one_hot=True)
        X = mnist.train.images
        Y = mnist.train.labels
        X_test = mnist.test.images
        Y_test = mnist.test.labels
        return X, Y, X_test, Y_test

    def load_cifar10(self):
        path = 'data/cifar-10-batches-py'
        data = [self.unpickle(path + '/data_batch_' + str(i)) for i in range(1, 5)]
        X = np.concatenate([self.get_images(batch['data']) for batch in data])
        Y = np.concatenate([self.onehot_labels(batch['labels'], 10) for batch in data])
        X_test = self.get_images(self.unpickle(path + '/test_batch')['data'])
        Y_test = self.onehot_labels(self.unpickle(path + '/test_batch')['labels'], 10)
        return X, Y, X_test, Y_test

    def load_cifar100(self):
        path = 'data/cifar-100-python'
        X = self.get_images(self.unpickle(path + '/train')['data'])
        Y = self.onehot_labels(self.unpickle(path + '/train')['fine_labels'], 100)
        X_test = self.get_images(self.unpickle(path + '/test')['data'])
        Y_test = self.onehot_labels(self.unpickle(path + '/test')['fine_labels'], 100)
        return X, Y, X_test, Y_test