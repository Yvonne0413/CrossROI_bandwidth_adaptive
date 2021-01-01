#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : backbone.py
#   Author      : YunYang1994
#   Created date: 2019-02-17 11:03:35
#   Description :
#
#================================================================

import core.common as common
import tensorflow as tf


def darknet53(input_data, trainable, debug_bk=9):

    with tf.variable_scope('darknet'):

        input_data = common.convolutional(input_data, filters_shape=(3, 3,  3,  32), trainable=trainable, name='conv0')
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 32,  64),
                                          trainable=trainable, name='conv1', downsample=True)

        if debug_bk == 0: return input_data

        for i in range(1):
            input_data = common.residual_block(input_data,  64,  32, 64, trainable=trainable, name='residual%d' %(i+0))

        if debug_bk == 1: return input_data

        input_data = common.convolutional(input_data, filters_shape=(3, 3,  64, 128),
                                          trainable=trainable, name='conv4', downsample=True)

        if debug_bk == 2: return input_data

        for i in range(2):
            input_data = common.residual_block(input_data, 128,  64, 128, trainable=trainable, name='residual%d' %(i+1))

        if debug_bk == 3: return input_data

        input_data = common.convolutional(input_data, filters_shape=(3, 3, 128, 256),
                                          trainable=trainable, name='conv9', downsample=True)

        if debug_bk == 4: return input_data

        for i in range(8):
            input_data = common.residual_block(input_data, 256, 128, 256, trainable=trainable, name='residual%d' %(i+3))

        if debug_bk == 5: return input_data

        route_1 = input_data
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 256, 512),
                                          trainable=trainable, name='conv26', downsample=True)

        if debug_bk == 6: return input_data

        for i in range(8):
            input_data = common.residual_block(input_data, 512, 256, 512, trainable=trainable, name='residual%d' %(i+11))

        if debug_bk == 7: return input_data

        route_2 = input_data
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 512, 1024),
                                          trainable=trainable, name='conv43', downsample=True)

        if debug_bk == 8: return input_data

        for i in range(4):
            input_data = common.residual_block(input_data, 1024, 512, 1024, trainable=trainable, name='residual%d' %(i+19))

        return route_1, route_2, input_data



def sparse_darknet53(input_data, input_mask, trainable, debug_bk=9):

    input_data = common.convolutional(input_data, filters_shape=(3, 3,  3,  32), trainable=trainable, name='conv0')

    input_data = common.convolutional(input_data, filters_shape=(3, 3, 32,  64),
                                        trainable=trainable, name='conv1', downsample=True)

    input_mask = tf.nn.max_pool(input_mask, [1,2,2,1], [1,2,2,1], 'SAME', data_format='NHWC')

    if debug_bk == 0: return input_data

    for i in range(1):
        input_data = common.sparse_residual_block(input_data, input_mask, 21, 0.05, 64,  32, 64, trainable=trainable, name='residual%d' %(i+0))
        
    if debug_bk == 1: return input_data

    input_data = common.convolutional(input_data, filters_shape=(3, 3,  64, 128),
                                        trainable=trainable, name='conv4', downsample=True)

    input_mask = tf.nn.max_pool(input_mask, [1,2,2,1], [1,2,2,1], 'SAME', data_format='NHWC')

    if debug_bk == 2: return input_data

    for i in range(2):
        input_data = common.sparse_residual_block(input_data, input_mask, 17, 0.05, 128, 64, 128, trainable=trainable, name='residual%d' %(i+1))

    if debug_bk == 3: return input_data

    input_data = common.convolutional(input_data, filters_shape=(3, 3, 128, 256),
                                        trainable=trainable, name='conv9', downsample=True)

    input_mask = tf.nn.max_pool(input_mask, [1,2,2,1], [1,2,2,1], 'SAME', data_format='NHWC')

    if debug_bk == 4: return input_data

    for i in range(8):
        input_data = common.sparse_residual_block(input_data, input_mask, 11, 0.05, 256, 128, 256, trainable=trainable, name='residual%d' %(i+3))

    if debug_bk == 5: return input_data

    route_1 = input_data
    input_data = common.convolutional(input_data, filters_shape=(3, 3, 256, 512),
                                        trainable=trainable, name='conv26', downsample=True)

    input_mask = tf.nn.max_pool(input_mask, [1,2,2,1], [1,2,2,1], 'SAME', data_format='NHWC')

    if debug_bk == 6: return input_data

    for i in range(8):
        input_data = common.sparse_residual_block(input_data, input_mask, 9, 0.05, 512, 256, 512, trainable=trainable, name='residual%d' %(i+11))

    input_mask = tf.nn.max_pool(input_mask, [1,2,2,1], [1,2,2,1], 'SAME', data_format='NHWC')

    if debug_bk == 7: return input_data

    route_2 = input_data
    input_data = common.convolutional(input_data, filters_shape=(3, 3, 512, 1024),
                                        trainable=trainable, name='conv43', downsample=True)

    input_mask = tf.nn.max_pool(input_mask, [1,2,2,1], [1,2,2,1], 'SAME', data_format='NHWC')

    if debug_bk == 8: return input_data

    for i in range(4):
        input_data = common.sparse_residual_block(input_data, input_mask, 4, 0.05, 1024, 512, 1024, trainable=trainable, name='residual%d' %(i+19))

    return route_1, route_2, input_data



def sparse_darknet53_var(input_data, input_mask, trainable, debug_bk=9):

    with tf.variable_scope('darknet'):

        input_data = common.convolutional(input_data, filters_shape=(3, 3,  3,  32), trainable=trainable, name='conv0')
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 32,  64),
                                          trainable=trainable, name='conv1', downsample=True)

        input_mask = tf.nn.max_pool(input_mask, [1,2,2,1], [1,2,2,1], 'SAME')

        if debug_bk == 0: return input_data

        input_data = common.sparse_residual_block_var(input_data, input_mask, 21, 0.05, 1, 64, 32, 64, trainable=trainable, name='res_bk_%d' %(0))

        if debug_bk == 1: return input_data

        input_data = common.convolutional(input_data, filters_shape=(3, 3,  64, 128),
                                          trainable=trainable, name='conv4', downsample=True)

        input_mask = tf.nn.max_pool(input_mask, [1,2,2,1], [1,2,2,1], 'SAME')

        if debug_bk == 2: return input_data

        input_data = common.sparse_residual_block_var(input_data, input_mask, 17, 0.05, 2, 128, 64, 128, trainable=trainable, name='res_bk_%d' %(1))

        if debug_bk == 3: return input_data

        input_data = common.convolutional(input_data, filters_shape=(3, 3, 128, 256),
                                          trainable=trainable, name='conv9', downsample=True)

        input_mask = tf.nn.max_pool(input_mask, [1,2,2,1], [1,2,2,1], 'SAME')

        if debug_bk == 4: return input_data        

        input_data = common.sparse_residual_block_var(input_data, input_mask, 11, 0.05, 8, 256, 128, 256, trainable=trainable, name='res_bk_%d' %(2))

        if debug_bk == 5: return input_data

        route_1 = input_data
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 256, 512),
                                          trainable=trainable, name='conv26', downsample=True)

        input_mask = tf.nn.max_pool(input_mask, [1,2,2,1], [1,2,2,1], 'SAME')

        if debug_bk == 6: return input_data

        input_data = common.sparse_residual_block_var(input_data, input_mask, 9, 0.05, 8, 512, 256, 512, trainable=trainable, name='res_bk_%d' %(3))

        if debug_bk == 7: return input_data

        route_2 = input_data
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 512, 1024),
                                          trainable=trainable, name='conv43', downsample=True)

        if debug_bk == 8: return input_data

        for i in range(4):
            input_data = common.residual_block(input_data, 1024, 512, 1024, trainable=trainable, name='residual%d' %(i+19))

        return route_1, route_2, input_data



