"""
License: Apache-2.0
Author: Yunzhi Shi
E-mail: yzshi08 at utexas.edu
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from config import cfg


def margin_loss(output, y):
    """
    Margin loss for Eq.(4).
    When y_true[i, :] contains not just one `1`, this loss should work too.
    Not test yet.

    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    n_class = output.get_shape()[-1]
    y_vec = tf.one_hot(y, n_class, dtype=tf.float32)

    L = y_vec * tf.square(tf.maximum(0., 0.9 - output)) + \
        0.5 * (1 - y_vec) * tf.square(tf.maximum(0., output- 0.1))

    return tf.reduce_mean(tf.reduce_sum(L, axis=1))


def cross_entropy_loss(output, y):
    loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=output)
    regularization = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    return tf.add_n([loss] + regularization)


def squash(vectors, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large
    vector to near 1 and small vector to 0.

    Args:
        vectors: some vectors to be squashed, N-dim tensor
        axis: the axis to squash
    Returns:
        A Tensor with same shape as input vectors.
    """
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis=axis, keep_dims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm +
                                                            tf.keras.backend.epsilon())
    return scale * vectors


def build_arch(input, is_train: bool, num_classes: int):
    data_size = input.shape[1]

    with slim.arg_scope([slim.conv2d], trainable=is_train):

        with tf.variable_scope('relu_conv1') as scope:
            # Ordinary conv2d with ReLU activation.
            n_filters = 256
            kernel = 9
            stride=1
            output = slim.conv2d(input, num_outputs=n_filters,
                                 kernel_size=[kernel, kernel],
                                 stride=stride, padding='VALID',
                                 activation_fn=tf.nn.relu, scope=scope)
            data_size = (data_size - kernel//2*2) // stride
            assert output.get_shape() == [cfg.batch_size, data_size, data_size, n_filters]

        with tf.variable_scope('primary_caps') as scope:
            # Conv2d with 'dim_capsule * n_channels' filters to form n_channels
            # vectors of dimension dim_capsule.
            dim_capsule = 8
            n_channels = 32
            kernel = 9
            stride=2
            output = slim.conv2d(output, num_outputs=dim_capsule*n_channels,
                                 kernel_size=[kernel, kernel],
                                 stride=stride, padding='VALID',
                                 activation_fn=None, scope=scope)
            data_size = (data_size - kernel//2*2) // stride
            output = tf.reshape(output, shape=[cfg.batch_size,
                                               data_size*data_size*n_channels,
                                               dim_capsule])

            # Squash the capsule vectors.
            output = tf.map_fn(lambda x: squash(x), output)
            assert output.get_shape() == [cfg.batch_size, data_size*data_size*n_channels, dim_capsule]
    
        with tf.variable_scope('digit_caps') as scope:
            # Capsule layer, routing algorithm works here.
            dim_capsule = 16
            num_capsule = num_classes
            num_routing = cfg.iter_routing
            input_num_capsule = output.shape[1]
            input_dim_capsule = output.shape[2]

            # Using weights with bigger stddev helps numerical stability
            W = slim.variable('W', dtype=tf.float32,
                              shape=[num_capsule, input_num_capsule,
                                     dim_capsule, input_dim_capsule],
                              initializer=tf.keras.initializers.glorot_uniform())

            # inputs.shape=[None, input_num_capsule, input_dim_capsule]
            # inputs_expand.shape=[None, 1, input_num_capsule, input_dim_capsule]
            input_expand = tf.expand_dims(output, axis=1)

            # replicate num_capsule dimension to prepare being multiplied by w
            # inputs_tiled.shape=[none, num_capsule, input_num_capsule, input_dim_capsule]
            input_tiled = tf.tile(input_expand, [1, num_capsule, 1, 1])

            # compute `inputs * w` by scanning inputs_tiled on dimension 0.
            # x.shape=[num_capsule, input_num_capsule, input_dim_capsule]
            # w.shape=[num_capsule, input_num_capsule, dim_capsule, input_dim_capsule]
            # regard the first two dimensions as `batch` dimension,
            # then matmul: [input_dim_capsule] x [dim_capsule, input_dim_capsule]^t -> [dim_capsule].
            # inputs_hat.shape = [none, num_capsule, input_num_capsule, dim_capsule]
            input_hat = tf.map_fn(lambda x: tf.keras.backend.batch_dot(x, W, [2, 3]), input_tiled)

            # begin: routing algorithm ---------------------------------------------------------------------#
            # in forward pass, `inputs_hat_stopped` = `inputs_hat`;
            # in backward, no gradient can flow from `inputs_hat_stopped` back to `inputs_hat`.
            input_hat_stopped = tf.stop_gradient(input_hat)
            
            # the prior for coupling coefficient, initialized as zeros.
            # b.shape = [none, self.num_capsule, self.input_num_capsule].
            b = tf.zeros(shape=[tf.shape(input_hat)[0], num_capsule, input_num_capsule])

            assert num_routing > 0, 'The num_routing should be > 0.'
            for i in range(num_routing):
                # c.shape = [batch_size, num_capsule, input_num_capsule]
                c = tf.nn.softmax(b, dim=1)

                # At last iteration, use `inputs_hat` to compute `outputs`
                # in order to backpropagate gradient.
                if i == num_routing - 1:
                    # c.shape = [None, num_capsule, input_num_capsule]
                    # inputs_hat.shape = [None, num_capsule, input_num_capsule, dim_capsule]
                    # The first two dimensions as `batch` dimension,
                    # then matmal: [input_num_capsule] x [input_num_capsule, dim_capsule] -> [dim_capsule].
                    # outputs.shape=[None, num_capsule, dim_capsule]
                    output = squash(tf.keras.backend.batch_dot(c, input_hat, [2, 2]))  # [None, 10, 16]
                else:
                    # Otherwise, use `inputs_hat_stopped` to update `b`.
                    # No gradients flow on this path.
                    output = squash(tf.keras.backend.batch_dot(c, input_hat_stopped, [2, 2]))

                    # outputs.shape = [None, num_capsule, dim_capsule]
                    # inputs_hat.shape = [None, num_capsule, input_num_capsule, dim_capsule]
                    # The first two dimensions as `batch` dimension,
                    # then matmal: [dim_capsule] x [input_num_capsule, dim_capsule]^T -> [input_num_capsule].
                    # b.shape=[batch_size, num_capsule, input_num_capsule]
                    b += tf.keras.backend.batch_dot(output, input_hat_stopped, [2, 3])
            # End: Routing algorithm -----------------------------------------------------------------------#
    
        with tf.variable_scope('classify') as scope:
            """
            Compute the length of vectors. This is used to compute a Tensor that has
            the same shape with y_true in margin_loss. Using this layer as model's
            output can directly predict labels by using
            `y_pred = np.argmax(model.predict(x), 1)`.

            inputs: shape=[None, num_vectors, dim_vector]
            output: shape=[None, num_vectors]
            """
            out_caps = tf.sqrt(tf.reduce_sum(tf.square(output), axis=-1))

    return out_caps
