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

def generate_image_data(batch_size):
    vid = cv2.VideoCapture("../DelegationGraph/croped_c003.mp4")

    frames = []
    for i in range(batch_size):
        vid.set(1, i)
        _, frame = vid.read()
        frames.append(frame)

    for i, frame in enumerate(frames):
        frames[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    images_data  = []
    for frame in frames:
        image_data = utils.image_preporcess(np.copy(frame), [416, 416])
        image_data = image_data[np.newaxis, ...]
        images_data.append(image_data)

    x = tf.constant(np.vstack(images_data).astype(np.float32))

    return x


def generate_mask_data(batch_size):
    mask_im = cv2.imread("../DelegationGraph/c003_mask.jpg")
    mask_im = np.round(utils.image_preporcess(np.copy(mask_im), [416, 416])[:,:,0])
    mask_im = mask_im[np.newaxis, ..., np.newaxis]

    images_data = [mask_im for _ in range(batch_size)] 

    x = tf.constant(np.vstack(images_data).astype(np.float32))

    return x

print("=======================")

batch_size = 20

x = generate_image_data(batch_size)
mask = generate_mask_data(batch_size)

model = YOLOV3(x, False, mask)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

s = time.time()
for i in range(30):
    if i == 10:
        s = time.time()
    _, a, b, c = sess.run([x, model.pred_sbbox, model.pred_mbbox, model.pred_lbbox])
    # print(a.shape, b.shape, c.shape)
e = time.time()

print(batch_size,  (e - s) / 20)

'''
Profile = {'original': [0] * 10, 'sparse': [0] * 10, 'sparse_var': [0] * 10}

for debug_bk in range(10):
    with tf.variable_scope('original{}'.format(debug_bk)):
        y = darknet53(x, False, debug_bk)

    with tf.variable_scope('sparse{}'.format(debug_bk)):
        z = sparse_darknet53(x, mask, False, debug_bk)

    # with tf.variable_scope('sparse_var{}'.format(debug_bk)):
    #     w = sparse_darknet53_var(x, mask, False, debug_bk)


    for type in ['sparse', 'original']:
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        s = time.time()
        for i in range(30):
            if i == 10:
                s = time.time()
            if type == 'sparse':
                _, z_out = sess.run([x, z])
                if isinstance(z_out, tuple):
                    z_out = z_out[-1]
                print(np.min(z_out))
            elif type == 'original':
                _, y_out = sess.run([x, y])
                if isinstance(y_out, tuple):
                    y_out = y_out[-1]
                print(np.min(y_out))
        e = time.time()

        Profile[type][debug_bk] = (e - s) / 20


print(Profile['original'])
print(Profile['sparse'])
# print(Profile['sparse_var'])

part_org = [Profile['original'][0]] + [Profile['original'][i+1] - Profile['original'][i] for i in range(9)]
part_spa = [Profile['sparse'][0]] + [Profile['sparse'][i+1] - Profile['sparse'][i] for i in range(9)]

print(part_org)
print(part_spa)
# print(part_spa_var)

json = json.dumps(Profile)
f = open("dict.json","w")
f.write(json)
f.close()
'''