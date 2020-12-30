import numpy as np
import tensorflow as tf

from collections import namedtuple
from core.sparse_conv_lib import calc_block_params_res_block, convert_mask_to_indices_custom

# tf.enable_eager_execution()

MaskConfig = namedtuple(
    'MaskConfig', ['xsize', 'ksize_list', 'bsize', 'strides', 'padding', 'tol', 'avgpool'])

MaskResult = namedtuple('MaskResult', ['ReduceMask', 'BlockParams'])

def darnet53_mask(input_mask):
    batch_size = input_mask.shape[0]

    result = {}

    input_mask = tf.nn.max_pool(input_mask, [1,2,2,1], [1,2,2,1], 'SAME')

    res_0_config = MaskConfig(
                    xsize= [batch_size, 208, 208, 64],
                    ksize_list= [[1, 1, 64, 32], [3, 3,  32, 64]],
                    strides=[1, 1, 1, 1],
                    padding='SAME',
                    bsize=[1, 14, 14, 1],
                    tol=1.0,
                    avgpool=True)

    res_0_block_params = calc_block_params_res_block(res_0_config.xsize, res_0_config.bsize, res_0_config.ksize_list, \
                                                    res_0_config.strides, res_0_config.padding)
    res_0_ind = convert_mask_to_indices_custom(input_mask[:,:,:,0], res_0_block_params, res_0_config.tol, res_0_config.avgpool)

    result['res_block_0'] = MaskResult(ReduceMask=res_0_ind, BlockParams=res_0_block_params)

    input_mask = tf.nn.max_pool(input_mask, [1,2,2,1], [1,2,2,1], 'SAME')

    res_1_config = MaskConfig(
                    xsize= [batch_size, 104, 104, 128],
                    ksize_list= [[1, 1, 128, 64], [3, 3, 64, 128]],
                    strides=[1, 1, 1, 1],
                    padding='SAME',
                    bsize=[1, 7, 7, 1],
                    tol=0.9,
                    avgpool=True)

    res_1_block_params = calc_block_params_res_block(res_1_config.xsize, res_1_config.bsize, res_1_config.ksize_list, \
                                                    res_1_config.strides, res_1_config.padding)
    res_1_ind = convert_mask_to_indices_custom(input_mask[:,:,:,0], res_1_block_params, res_1_config.tol, res_1_config.avgpool)

    result['res_block_1'] = MaskResult(ReduceMask=res_1_ind, BlockParams=res_1_block_params)

    input_mask = tf.nn.max_pool(input_mask, [1,2,2,1], [1,2,2,1], 'SAME')

    res_2_config = MaskConfig(
                    xsize= [batch_size, 52, 52, 256],
                    ksize_list= [[1, 1, 256, 128], [3, 3,  128, 256]],
                    strides=[1, 1, 1, 1],
                    padding='SAME',
                    bsize=[1, 4, 4, 1],
                    tol=0.9,
                    avgpool=True)

    res_2_block_params = calc_block_params_res_block(res_2_config.xsize, res_2_config.bsize, res_2_config.ksize_list, \
                                                    res_2_config.strides, res_2_config.padding)
    res_2_ind = convert_mask_to_indices_custom(input_mask[:,:,:,0], res_2_block_params, res_2_config.tol, res_2_config.avgpool)

    result['res_block_2'] = MaskResult(ReduceMask=res_2_ind, BlockParams=res_2_block_params)

    input_mask = tf.nn.max_pool(input_mask, [1,2,2,1], [1,2,2,1], 'SAME')

    res_3_config = MaskConfig(
                    xsize= [batch_size, 26, 26, 512],
                    ksize_list= [[1, 1, 512, 256], [3, 3, 256, 512]],
                    strides=[1, 1, 1, 1],
                    padding='SAME',
                    bsize=[1, 4, 4, 1],
                    tol=0.1,
                    avgpool=True)

    res_3_block_params = calc_block_params_res_block(res_3_config.xsize, res_3_config.bsize, res_3_config.ksize_list, \
                                                    res_3_config.strides, res_3_config.padding)
    res_3_ind = convert_mask_to_indices_custom(input_mask[:,:,:,0], res_3_block_params, res_3_config.tol, res_3_config.avgpool)

    result['res_block_3'] = MaskResult(ReduceMask=res_3_ind, BlockParams=res_3_block_params)

    return result