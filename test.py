from numpy.lib.type_check import imag
from core.sparse_conv_perf import N_REPEAT
import tensorflow as tf
import numpy as np
from core.backbone import darknet53, sparse_darknet53, sparse_darknet53_var
from core.mask import darnet53_mask
from core.common import sparse_residual_block, residual_block, convolutional
import time, json
from core.yolov3 import YOLOV3
import cv2
import numpy as np
import core.utils as utils
import tensorflow as tf
from PIL import Image

def generate_image_data(data_path, batch_size, resolution):
    frame = cv2.imread(data_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frames = []
    for i in range(batch_size):
        frames.append(frame)

    images_data  = []
    for frame in frames:
        image_data = utils.image_preporcess(np.copy(frame), [resolution, resolution])
        image_data = image_data[np.newaxis, ...]
        images_data.append(image_data)

    x = tf.constant(np.vstack(images_data).astype(np.float32))

    return x


def generate_mask_data(mask_path, batch_size, resolution):
    mask_im = cv2.imread(mask_path)
    mask_im = np.round(utils.image_preporcess(np.copy(mask_im), [resolution, resolution])[:,:,0])
    mask_im = mask_im[np.newaxis, ..., np.newaxis]

    images_data = [mask_im for _ in range(batch_size)] 

    x = tf.constant(np.vstack(images_data).astype(np.float32))

    return x


data_root = '../DelegationGraph/videos/S01/'
result_root = './measure_speed/'
scene_name = 'S01'
cameras = ['c001', 'c002', 'c003', 'c004', 'c005']


if __name__ == '__main__':

    batch_size = 5
    resolution = 960
    repeat = 10

    Results = {}

    for setting in ['1e-06_1.0', '5e-06_1.0', '1e-05_1.0', '5e-05_1.0', '1e-04_1.0', \
					'2e-05_0.01', '2e-05_0.05', '2e-05_0.1',  '2e-05_1.0', '2e-05_10.0',\
					'nofilter' , 'baseline']:

        Results[setting] = {}

        for camera in cameras:

            Results[setting][camera] = []
 
            data_path = data_root + setting + '/' + camera + '_mask.jpg'
            mask_path = data_root + setting + '/' + camera + '_mask.jpg'

            if setting == 'baseline':
                data_path = data_root + 'nofilter' + '/' + camera + '_mask.jpg'
                mask_path = None

            x = generate_image_data(data_path, batch_size, resolution)


            if mask_path is not None:
                mask = generate_mask_data(mask_path, batch_size, resolution)
                print(mask.shape)
                model = YOLOV3(x, False, mask)
            else:
                model = YOLOV3(x, False)

            for _ in range(repeat):

                sess = tf.Session()
                sess.run(tf.global_variables_initializer())

                s = time.time()
                for i in range(30):
                    if i == 10:
                        s = time.time()
                    _, a, b, c = sess.run([x, model.pred_sbbox, model.pred_mbbox, model.pred_lbbox])
                e = time.time()

                print(batch_size,  (e - s) / 20)
                Results[setting][camera].append((e -s) / 20)

                sess.close()

            tf.reset_default_graph()
    
    np.save('measure_speed/speed_res.npy', Results)