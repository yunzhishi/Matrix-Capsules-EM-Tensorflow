import os
import scipy
import numpy as np
import tensorflow as tf

from config import cfg
import data.smallNORB as norb

import logging
import daiquiri

from six.moves import cPickle as pickle

daiquiri.setup(level=logging.DEBUG)
logger = daiquiri.getLogger(__name__)


def create_inputs_norb(is_train: bool, epochs: int):

    import re
    if is_train:
        CHUNK_RE = re.compile(r"train\d+\.tfrecords")
    else:
        CHUNK_RE = re.compile(r"test\d+\.tfrecords")

    processed_dir = './data/smallNORB/'
    chunk_files = [os.path.join(processed_dir, fname)
                   for fname in os.listdir(processed_dir)
                   if CHUNK_RE.match(fname)]

    image, label = norb.read_norb_tfrecord(chunk_files, epochs)

    # TODO: is it the right order: add noise, resize, then corp?
    image = tf.image.random_brightness(image, max_delta=32. / 255.)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)

    image = tf.image.resize_images(image, [48, 48])

    # naive minimax to [0,1]
    if is_train:
        image = tf.random_crop(image, [32, 32, 1]) / 255.
    else:
        image = tf.slice(image, [8, 8, 0], [32, 32, 1])/255. #image[8:40, 8:40, 1]/255.

    image = (image-0.5)*2.

    x, y = tf.train.shuffle_batch([image, label], num_threads=cfg.num_threads, batch_size=cfg.batch_size, capacity=cfg.batch_size * 64,
                                  min_after_dequeue=cfg.batch_size * 32, allow_smaller_final_batch=False)

    return x, y


def create_inputs_mnist(is_train):
    tr_x, tr_y = load_mnist(cfg.dataset, is_train)
    data_queue = tf.train.slice_input_producer([tr_x, tr_y], capacity=64 * 8)
    x, y = tf.train.shuffle_batch(data_queue, num_threads=8, batch_size=cfg.batch_size, capacity=cfg.batch_size * 64,
                                  min_after_dequeue=cfg.batch_size * 32, allow_smaller_final_batch=False)

    return (x, y)


def load_mnist(path, is_training):
    fd = open(os.path.join(cfg.dataset, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

    fd = open(os.path.join(cfg.dataset, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.int32)

    fd = open(os.path.join(cfg.dataset, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

    fd = open(os.path.join(cfg.dataset, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.int32)

    # normalization and convert to a tensor [60000, 28, 28, 1]
    trX = tf.convert_to_tensor(trX / 255., tf.float32)
    teX = tf.convert_to_tensor(teX / 255., tf.float32)

    # => [num_samples, 10]
    # trY = tf.one_hot(trY, depth=10, axis=1, dtype=tf.float32)
    # teY = tf.one_hot(teY, depth=10, axis=1, dtype=tf.float32)

    if is_training:
        return trX, trY
    else:
        return teX, teY
    
    
def create_inputs_cifar10(is_train):
    processed_dir = './data/cifar-10-batches-py/'
    tr_x, tr_y = load_cifar10(processed_dir, is_train)
    data_queue = tf.train.slice_input_producer([tr_x, tr_y], capacity=64 * 8)
    x, y = tf.train.shuffle_batch(data_queue, num_threads=8, batch_size=cfg.batch_size, capacity=cfg.batch_size * 64, min_after_dequeue=cfg.batch_size * 32, allow_smaller_final_batch=False)
    return (x, y)

    
def load_cifar10_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
        return X, Y

    
def load_cifar10(path, is_training):
    if is_training:
        xs = []
        ys = []
        for b in range(1,6):
            f = os.path.join(path, 'data_batch_%d' % (b, ))
            X, Y = load_cifar10_batch(f)
            xs.append(X)
            ys.append(Y)
        trX = np.concatenate(xs)
        trY = np.concatenate(ys)
        del X, Y
        trX = tf.convert_to_tensor(trX / 255., tf.float32)
        return trX, trY
    else:
        teX, teY = load_cifar10_batch(os.path.join(path, 'test_batch'))
        teX = tf.convert_to_tensor(teX / 255., tf.float32)
        return teX, teY