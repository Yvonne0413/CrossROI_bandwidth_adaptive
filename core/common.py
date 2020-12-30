#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : common.py
#   Author      : YunYang1994
#   Created date: 2019-02-28 09:56:29
#   Description :
#
#================================================================

import tensorflow as tf
from core.sparse_conv_lib import convert_mask_to_indices_custom, calc_block_params_res_block
from core.sparse_conv_lib import sparse_res_block_bottleneck, calc_block_params, sparse_conv2d_custom

def convolutional(input_data, filters_shape, trainable, name, downsample=False, activate=True, bn=True):

    with tf.variable_scope(name):
        if downsample:
            pad_h, pad_w = (filters_shape[0] - 2) // 2 + 1, (filters_shape[1] - 2) // 2 + 1
            paddings = tf.constant([[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]])
            input_data = tf.pad(input_data, paddings, 'CONSTANT')
            strides = (1, 2, 2, 1)
            padding = 'VALID'
        else:
            strides = (1, 1, 1, 1)
            padding = "SAME"

        weight = tf.get_variable(name='weight', dtype=tf.float32, trainable=True,
                                 shape=filters_shape, initializer=tf.random_normal_initializer(stddev=0.01))
        conv = tf.nn.conv2d(input=input_data, filter=weight, strides=strides, padding=padding)

        if bn:
            conv = tf.layers.batch_normalization(conv, beta_initializer=tf.zeros_initializer(),
                                                 gamma_initializer=tf.ones_initializer(),
                                                 moving_mean_initializer=tf.zeros_initializer(),
                                                 moving_variance_initializer=tf.ones_initializer(), training=trainable)
        else:
            bias = tf.get_variable(name='bias', shape=filters_shape[-1], trainable=True,
                                   dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, bias)

        if activate == True: conv = tf.nn.leaky_relu(conv, alpha=0.1)

    return conv


def residual_block(input_data, input_channel, filter_num1, filter_num2, trainable, name):

    short_cut = input_data

    with tf.variable_scope(name):
        input_data = convolutional(input_data, filters_shape=(1, 1, input_channel, filter_num1),
                                   trainable=trainable, name='conv1')
        input_data = convolutional(input_data, filters_shape=(3, 3, filter_num1,   filter_num2),
                                   trainable=trainable, name='conv2')

        residual_output = input_data + short_cut

    return residual_output



def route(name, previous_output, current_output):

    with tf.variable_scope(name):
        output = tf.concat([current_output, previous_output], axis=-1)

    return output


def upsample(input_data, name, method="deconv"):
    assert method in ["resize", "deconv"]

    if method == "resize":
        with tf.variable_scope(name):
            input_shape = tf.shape(input_data)
            output = tf.image.resize_nearest_neighbor(input_data, (input_shape[1] * 2, input_shape[2] * 2))

    if method == "deconv":
        # replace resize_nearest_neighbor with conv2d_transpose To support TensorRT optimization
        numm_filter = input_data.shape.as_list()[-1]
        output = tf.layers.conv2d_transpose(input_data, numm_filter, kernel_size=2, padding='same',
                                            strides=(2,2), kernel_initializer=tf.random_normal_initializer())

    return output

def sparse_convolutional(input_data, input_mask, block_size, tol, filters_shape, trainable, name, downsample=False, activate=True, bn=True):
    with tf.variable_scope(name):
        xsize, bsize = tf.shape(input_data), [1, block_size, block_size, 1]
        ksize = filters_shape
        strides, padding, tol, avgpool = [1, 1, 1, 1], 'SAME', tol, True
        mask = input_mask[:, :, :, 0]

        block_params = calc_block_params(xsize, bsize, ksize, strides, padding)
        ind = convert_mask_to_indices_custom(mask, block_params, tol, avgpool)

        weight = tf.get_variable(name='weight', dtype=tf.float32, trainable=True,
                                 shape=filters_shape, initializer=tf.random_normal_initializer(stddev=0.01))

        print(filters_shape)

        # conv = sparse_conv2d_custom(input_data, weight, ind, block_params, strides)
        conv = tf.nn.conv2d(input=input_data, filter=weight, strides=strides, padding=padding)

        new_size = tf.stack([xsize[0], xsize[1], xsize[2], filters_shape[-1]])
        conv = tf.reshape(conv, new_size)

        if bn:
            conv = tf.layers.batch_normalization(conv, beta_initializer=tf.zeros_initializer(),
                                                 gamma_initializer=tf.ones_initializer(),
                                                 moving_mean_initializer=tf.zeros_initializer(),
                                                 moving_variance_initializer=tf.ones_initializer(), training=trainable)
        else:
            bias = tf.get_variable(name='bias', shape=filters_shape[-1], trainable=True,
                                   dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, bias)

        if activate == True: conv = tf.nn.leaky_relu(conv, alpha=0.1)

    return conv


def sparse_residual_block(input_data, input_mask, block_size, tol, input_channel, filter_num1, filter_num2, trainable, name):
    with tf.variable_scope(name):
        xsize, bsize = tf.shape(input_data), [1, block_size, block_size, 1]
        ksize_list = [[1, 1, input_channel, filter_num1], [3, 3, filter_num1, filter_num2]]
        strides, padding, tol, avgpool = [1, 1, 1, 1], 'SAME', tol, True
        mask = input_mask[:, :, :, 0]

        block_params = calc_block_params_res_block(xsize, bsize, ksize_list, strides, padding)
        ind = convert_mask_to_indices_custom(mask, block_params, tol, avgpool)

        conv = sparse_res_block_bottleneck(
                input_data, ksize_list, ind, block_params, strides, is_training=trainable, use_var=False, data_format='NCHW')
       
        conv = tf.reshape(conv, xsize)

    return conv


def sparse_residual_block_var(input_data, input_mask, block_size, tol, n_repeat, input_channel, filter_num1, filter_num2, trainable, name):
    with tf.variable_scope(name):

        xsize, bsize = tf.shape(input_data), [1, block_size, block_size, 1]
        ksize_list = [[1, 1, input_channel, filter_num1], [3, 3, filter_num1, filter_num2]]
        strides, padding, tol, avgpool = [1, 1, 1, 1], 'SAME', tol, True
        mask = input_mask[:, :, :, 0]

        block_params = calc_block_params_res_block(xsize, bsize, ksize_list, strides, padding)
        ind = convert_mask_to_indices_custom(mask, block_params, tol, avgpool)

        xs, ys, xs_assign = [], [], []

        print(input_data.shape)

        for i in range(n_repeat):
            with tf.variable_scope('sparse_{}'.format(i)):
                xs.append( tf.Variable(tf.zeros_like(input_data), trainable=False))

        x0_init = tf.assign(xs[0], input_data)

        for i in range(n_repeat):
            print(len(ys), len(xs_assign))
            with tf.control_dependencies([x0_init] + xs + ys + xs_assign):
                with tf.variable_scope('sparse_{}'.format(i)):
                    y_ = sparse_res_block_bottleneck( xs[i], ksize_list, ind, block_params, strides, \
                                                is_training=trainable, use_var=True, data_format='NCHW')
                ys.append(y_)
                if i + 1 < n_repeat:
                    x_ = tf.assign(xs[i+1], y_)
                    xs_assign.append(x_)

        conv = ys[-1]
        conv = tf.reshape(conv, xsize)

    return conv