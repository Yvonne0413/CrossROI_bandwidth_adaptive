#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2018 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : video_demo.py
#   Author      : YunYang1994
#   Created date: 2018-11-30 15:56:37
#   Description :
#
#================================================================

import cv2
import os
import errno
import time
import numpy as np
import core.utils as utils
import tensorflow as tf
from PIL import Image

data_root = '../DelegationGraph/videos'
result_root = './results'
scene_name = 'S01'
cameras = ['c001', 'c002', 'c003', 'c004', 'c005']


def run_directory(scene_name, setting_name):
    setting_data_dir = data_root + '/' + scene_name + '/' + setting_name
    for camera_name in cameras:
        if setting_name == 'baseline':
            video_path = setting_data_dir + '/' + 'h264_' + camera_name + '.avi'
            mask_path = None
        else:
            video_path = setting_data_dir + '/' + 'croped_' + camera_name + '.avi'
            mask_path = setting_data_dir + '/'  + camera_name + '_mask.jpg'

        output_path = result_root + '/' + scene_name + '/' + 'det_' + camera_name + '.txt'

        run_inference(video_path, mask_path, output_path)
    

def run_inference(video_path, mask_path, output_path):

    return_elements = ["input/input_data:0", "input/input_mask:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
    pb_file         = "./yolov3_coco.pb"
    video_path      = video_path
    mask_path      = mask_path
    output_path    = output_path
    num_classes     = 80
    input_size      = 416
    graph           = tf.Graph()
    return_tensors  = utils.read_pb_return_tensors(graph, pb_file, return_elements)

    print(video_path, mask_path, output_path)

    def generate_mask_data(mask_path, batch_size):
        print(mask_path)
        mask_im = cv2.imread(mask_path)
        mask_im = np.round(utils.image_preporcess(np.copy(mask_im), [input_size, input_size])[:,:,0])
        mask_im = mask_im[np.newaxis, ..., np.newaxis]

        images_data = [mask_im for _ in range(batch_size)] 

        x = np.vstack(images_data).astype(np.float32)

        return x

    if mask_path == None:
        mask_data = np.ones((1, input_size, input_size, 1)).astype(np.float32)
    else:
        mask_data = generate_mask_data(mask_path, 1)


    if not os.path.exists(os.path.dirname(output_path)):
        try:
            os.makedirs(os.path.dirname(output_path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    # erase the detection output file.
    open(output_path, 'w').close()

    with tf.Session(graph=graph) as sess:
        vid = cv2.VideoCapture(video_path)
        frame_id = 0
        f = open(output_path, "a")
        while True:
            return_value, frame = vid.read()
            if return_value:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
            else:
                f.close()
                print ("No image!")
                return 0
            frame_size = frame.shape[:2]
            image_data = utils.image_preporcess(np.copy(frame), [input_size, input_size])
            image_data = image_data[np.newaxis, ...]
            prev_time = time.time()

            s = time.time()
            print(image_data.shape)
            pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
                [return_tensors[2], return_tensors[3], return_tensors[4]],
                        feed_dict={ return_tensors[0]: image_data,
                                    return_tensors[1]: mask_data})
            e = time.time()

            print(e - s)

            pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                        np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                        np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

            bboxes = utils.postprocess_boxes(pred_bbox, frame_size, input_size, 0.3)
            bboxes = utils.nms(bboxes, 0.45, method='nms')

            for bbox in bboxes:
                if bbox[-1] not in [2, 5, 7]:
                    continue
                if bbox[2] * bbox[3] < 6000:
                    continue
                f.write("{} {} {} {} {} {}\n".format(frame_id, bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]))

            frame_id += 1

            # image = utils.draw_bbox(frame, bboxes)
            # curr_time = time.time()
            # exec_time = curr_time - prev_time
            # result = np.asarray(image)
            # info = "time: %.2f ms" %(1000*exec_time)
            # cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
            # result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # cv2.imshow("result", result)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     f.close() 
            #     break


if __name__ == '__main__':
    for setting_name in os.listdir(data_root + '/' + scene_name):
        run_directory(scene_name, setting_name)


