"""
License: Apache-2.0
Author: Suofei Zhang
E-mail: zhangsuofei at njupt.edu.cn
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import math

from config import cfg


def spread_loss(output, y, m):
    """
    # See: https://stackoverflow.com/questions/40701712/how-to-check-nan-in-gradients-in-tensorflow-when-updating
    output_check = [tf.check_numerics(output, message='NaN Found!')]
    with tf.control_dependencies(output_check):
    """
    num_class = int(output.get_shape()[-1])
    y_vec = tf.one_hot(y, num_class, dtype=tf.float32)

    output1 = tf.reshape(output, shape=[cfg.batch_size, 1, num_class])
    y_vec = tf.expand_dims(y_vec, axis=2)
    at = tf.matmul(output1, y_vec)
    """Paper eq(5)."""
    loss = tf.square(tf.maximum(0., m - (at - output1)))
    loss = tf.matmul(loss, 1. - y_vec)
    loss = tf.reduce_mean(loss)
    regularization = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    return tf.add_n([loss] + regularization)


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
    loss = tf.reduce_mean(loss)
    regularization = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    return tf.add_n([loss] + regularization)


# input should be a tensor with size as [batch_size, height, width, channels]
def kernel_tile(input, kernel, stride):
    input_shape = input.get_shape()

    tile_filter = np.zeros(shape=[kernel, kernel, input_shape[3],
                                  kernel * kernel], dtype=np.float32)
    for i in range(kernel):
        for j in range(kernel):
            tile_filter[i, j, :, i * kernel + j] = 1.0

    tile_filter_op = tf.constant(tile_filter, dtype=tf.float32)
    output = tf.nn.depthwise_conv2d(input, tile_filter_op, strides=[
                                    1, stride, stride, 1], padding='VALID')
    output_shape = output.get_shape()
    output = tf.reshape(output, shape=[int(output_shape[0]), int(
        output_shape[1]), int(output_shape[2]), int(input_shape[3]), kernel * kernel])
    output = tf.transpose(output, perm=[0, 1, 2, 4, 3])

    return output

# input should be a tensor with size as [batch_size, caps_num_i, 16]


def mat_transform(input, caps_num_c, regularizer):
    batch_size = int(input.get_shape()[0])
    caps_num_i = int(input.get_shape()[1])
    if cfg.is_mat:  # 4x4 pose matrix
        output = tf.reshape(input, shape=[batch_size, caps_num_i, 1, 4, 4])
        shape_w = [1, caps_num_i, caps_num_c, 4, 4]
    else:  # length 16 pose vector
        output = tf.reshape(input, shape=[batch_size, caps_num_i, 1, 16])
        shape_w = [1, caps_num_i, caps_num_c, 16]
    # the output of capsule is miu, the mean of a Gaussian, and activation, the sum of probabilities
    # it has no relationship with the absolute values of w and votes
    # using weights with bigger stddev helps numerical stability
    w = slim.variable('w', shape=shape_w, dtype=tf.float32,
                      initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0),
                      regularizer=regularizer)

    if cfg.is_mat:
        w = tf.tile(w, [batch_size, 1, 1, 1, 1])
        output = tf.tile(output, [1, 1, caps_num_c, 1, 1])
        votes = tf.reshape(tf.matmul(output, w), [batch_size, caps_num_i, caps_num_c, 16])
    else:
        w = tf.tile(w, [batch_size, 1, 1, 1])
        output = tf.tile(output, [1, 1, caps_num_c, 1])
        votes = tf.reshape(output*w, [batch_size, caps_num_i, caps_num_c, 16])

    return votes


def build_arch(input, coord_add, is_train: bool, num_classes: int):
    data_size = int(input.get_shape()[1])
    # xavier initialization is necessary here to provide higher stability
    #initializer = tf.truncated_normal_initializer(mean=0.0, stddev=1.0)
    # instead of initializing bias with constant 0, a truncated normal initializer is exploited here for higher stability
    bias_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)  # tf.constant_initializer(0.0)
    # The paper didnot mention any regularization, a common l2 regularizer to weights is added here
    weights_regularizer = tf.contrib.layers.l2_regularizer(5e-04)

    # weights_initializer=initializer,
    with slim.arg_scope([slim.conv2d], trainable=is_train,
                        biases_initializer=bias_initializer,
                        weights_regularizer=weights_regularizer):
        output = slim.batch_norm(input, center=False, is_training=is_train, trainable=is_train)

        with tf.variable_scope('relu_conv1') as scope:
            output = slim.conv2d(output, num_outputs=cfg.A, kernel_size=[
                                 5, 5], stride=2, padding='VALID', scope=scope)
            data_size = int(np.floor((data_size - 4) / 2))

            assert output.get_shape() == [cfg.batch_size, data_size, data_size, cfg.A]

        with tf.variable_scope('primary_caps') as scope:
            pose = slim.conv2d(output, num_outputs=cfg.B * 16,
                               kernel_size=[1, 1], stride=1, padding='VALID', scope=scope, activation_fn=None)
            activation = slim.conv2d(output, num_outputs=cfg.B, kernel_size=[
                                     1, 1], stride=1, padding='VALID', activation_fn=tf.nn.sigmoid)
            pose = tf.reshape(pose, shape=[cfg.batch_size, data_size, data_size, cfg.B, 16])
            activation = tf.reshape(
                activation, shape=[cfg.batch_size, data_size, data_size, cfg.B, 1])
            output = tf.concat([pose, activation], axis=4)
            output = tf.reshape(output, shape=[cfg.batch_size, data_size, data_size, -1])
            assert output.get_shape() == [cfg.batch_size, data_size, data_size, cfg.B * 17]

        with tf.variable_scope('conv_caps1') as scope:
            output = kernel_tile(output, 3, 2)
            data_size = int(np.floor((data_size - 2) / 2))
            output = tf.reshape(output, shape=[cfg.batch_size *
                                               data_size * data_size, 3 * 3 * cfg.B, 17])
            activation = tf.reshape(output[:, :, 16], shape=[
                                    cfg.batch_size * data_size * data_size, 3 * 3 * cfg.B, 1])

            with tf.variable_scope('v') as scope:
                votes = mat_transform(output[:, :, :16], cfg.C, weights_regularizer)

            with tf.variable_scope('routing') as scope:
                miu, activation = em_routing(votes, activation, cfg.C, weights_regularizer)

            pose = tf.reshape(miu, shape=[cfg.batch_size, data_size, data_size, cfg.C, 16])
            activation = tf.reshape(
                activation, shape=[cfg.batch_size, data_size, data_size, cfg.C, 1])
            output = tf.reshape(tf.concat([pose, activation], axis=4), [
                                cfg.batch_size, data_size, data_size, -1])

        with tf.variable_scope('conv_caps2') as scope:
            output = kernel_tile(output, 3, 1)
            data_size = int(np.floor((data_size - 2) / 1))
            output = tf.reshape(output, shape=[cfg.batch_size *
                                               data_size * data_size, 3 * 3 * cfg.C, 17])
            activation = tf.reshape(output[:, :, 16], shape=[
                                    cfg.batch_size * data_size * data_size, 3 * 3 * cfg.C, 1])

            with tf.variable_scope('v') as scope:
                votes = mat_transform(output[:, :, :16], cfg.D, weights_regularizer)

            with tf.variable_scope('routing') as scope:
                miu, activation = em_routing(votes, activation, cfg.D, weights_regularizer)

            pose = tf.reshape(miu, shape=[cfg.batch_size * data_size * data_size, cfg.D, 16])
            activation = tf.reshape(
                activation, shape=[cfg.batch_size * data_size * data_size, cfg.D, 1])

        # It is not clear from the paper that ConvCaps2 is full connected to Class Capsules, or is conv connected with kernel size of 1*1 and a global average pooling.
        # From the description in Figure 1 of the paper and the amount of parameters (310k in the paper and 316,853 in fact), I assume a conv cap plus a golbal average pooling is the design.
        with tf.variable_scope('class_caps') as scope:
            with tf.variable_scope('v') as scope:
                votes = mat_transform(pose, num_classes, weights_regularizer)

                assert votes.get_shape() == [cfg.batch_size * data_size *
                                             data_size, cfg.D, num_classes, 16]

                coord_add = np.reshape(coord_add, newshape=[data_size * data_size, 1, 1, 2])
                coord_add = np.tile(coord_add, [cfg.batch_size, cfg.D, num_classes, 1])
                coord_add_op = tf.constant(coord_add, dtype=tf.float32)

                votes = tf.concat([coord_add_op, votes], axis=3)

            with tf.variable_scope('routing') as scope:
                miu, activation = em_routing(
                    votes, activation, num_classes, weights_regularizer)

            output = tf.reshape(activation, shape=[
                                cfg.batch_size, data_size, data_size, num_classes])

        output = tf.reshape(tf.nn.avg_pool(output, ksize=[1, data_size, data_size, 1], strides=[
                            1, 1, 1, 1], padding='VALID'), shape=[cfg.batch_size, num_classes])

    return output


def accuracy(logits, labels):
    logits_idx = tf.to_int32(tf.argmax(logits, axis=1))
    logits_idx = tf.reshape(logits_idx, shape=(cfg.batch_size,))
    correct_preds = tf.equal(tf.to_int32(labels), logits_idx)
    accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))

    return accuracy


def em_routing(votes, activation, caps_num_c, regularizer):
    batch_size = int(votes.get_shape()[0])
    caps_num_i = int(activation.get_shape()[1])
    n_channels = int(votes.get_shape()[-1])

    # m-step
    r = tf.constant(np.ones([batch_size, caps_num_i, caps_num_c], dtype=np.float32) / caps_num_c)
    r = r * activation

    # tf.reshape(tf.reduce_sum(r, axis=1), shape=[batch_size, 1, caps_num_c])
    r_sum = tf.reduce_sum(r, axis=1, keep_dims=True)
    r1 = tf.reshape(r / (r_sum + cfg.epsilon),
                    shape=[batch_size, caps_num_i, caps_num_c, 1])

    miu = tf.reduce_sum(votes * r1, axis=1, keep_dims=True)
    sigma_square = tf.reduce_sum(tf.square(votes - miu) * r1, axis=1, keep_dims=True) + cfg.epsilon

    beta_v = slim.variable('beta_v', shape=[caps_num_c, n_channels], dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                           regularizer=regularizer)
    r_sum = tf.reshape(r_sum, [batch_size, caps_num_c, 1])
    cost_h = (beta_v + tf.log(tf.sqrt(tf.reshape(sigma_square,
                                                 shape=[batch_size, caps_num_c, n_channels])))) * r_sum

    beta_a = slim.variable('beta_a', shape=[caps_num_c], dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                           regularizer=regularizer)
    activation1 = tf.nn.sigmoid(cfg.ac_lambda0 * (beta_a - tf.reduce_sum(cost_h, axis=2)))

    for iters in range(cfg.iter_routing):
        # e-step

        # algorithm from paper is replaced by products of p_{ch}, which supports better numerical stability
        # p_c_h = 1 / (tf.sqrt(sigma_square)) * tf.exp(-tf.square(votes - miu) / (2 * sigma_square))
        # p_c_h = p_c_h / (tf.reduce_max(p_c_h, axis=[2, 3], keep_dims=True) / 10.0)
        # p_c = tf.reduce_prod(p_c_h, axis=3)

        p_c_h = -0.5 * tf.log(2 * math.pi * sigma_square) -tf.square(votes - miu) / (2 * sigma_square)
        p_c = tf.reduce_sum(p_c_h, axis=3)
        a1 = tf.log(tf.reshape(activation1, shape=[batch_size, 1, caps_num_c]))
        ap = p_c + a1
        sum_ap = tf.reduce_logsumexp(ap, axis=2, keep_dims=True)
        r = tf.exp(ap - sum_ap)

        # log_p_c_h = -tf.log(tf.sqrt(sigma_square)) - (tf.square(votes-miu) / (2*sigma_square))
        # log_p_c_h = log_p_c_h - tf.reduce_max(log_p_c_h, axis=[2,3], keep_dims=True) - tf.log(10.0)
        # p_c = tf.exp(tf.reduce_sum(log_p_c_h, axis=3))
        # a1 = tf.reshape(activation1, shape=[batch_size, 1, caps_num_c])
        # ap = p_c * a1
        # sum_ap = tf.reduce_sum(ap, axis=2, keep_dims=True)
        # r = ap / (sum_ap + cfg.epsilon)

        # m-step
        r = r * activation

        r_sum = tf.reduce_sum(r, axis=1,
                              keep_dims=True)  # tf.reshape(tf.reduce_sum(r, axis=1), shape=[batch_size, 1, caps_num_c])
        r1 = tf.reshape(r / (r_sum + cfg.epsilon),
                        shape=[batch_size, caps_num_i, caps_num_c, 1])

        miu = tf.reduce_sum(votes * r1, axis=1, keep_dims=True)
        sigma_square = tf.reduce_sum(tf.square(votes - miu) * r1,
                                     axis=1, keep_dims=True) + cfg.epsilon

        r_sum = tf.reshape(r_sum, [batch_size, caps_num_c, 1])
        cost_h = (beta_v + tf.log(tf.sqrt(tf.reshape(sigma_square,
                                                     shape=[batch_size, caps_num_c, n_channels])))) * r_sum

        activation1 = tf.nn.sigmoid(
            (cfg.ac_lambda0 + (iters + 1) * cfg.ac_lambda_step) * (beta_a - tf.reduce_sum(cost_h, axis=2)))

    return miu, activation1
