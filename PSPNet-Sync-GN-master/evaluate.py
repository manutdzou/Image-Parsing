from __future__ import print_function
import argparse
import os
import sys
import time

from PIL import Image
import tensorflow as tf
import numpy as np
import cv2
from model import PSPNet
from tools import decode_labels
from image_reader import read_labeled_image_list

IMG_MEAN = np.array((72.41519599, 82.93553322, 73.18188461), dtype=np.float32) # RGB, Cityscapes.
input_size = 720

SAVE_DIR = './output/'
SNAPSHOT_DIR = './model/'

DATA_DIRECTORY = '/dfsdata/jinyi_data/DATASET/Cityscapes'
DATA_LIST_PATH = './list/eval_list.txt'

num_classes = 19
ignore_label = 255 # Don't care label


class ImageSplitter(object):
    def __init__(self, image, scale, crop_image_size, IMG_MEAN):
        height, width = image.shape[0], image.shape[1]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.float32(image)
        scaled_height, scaled_width = int(height * scale), int(width * scale)
        image = cv2.resize(image, dsize=(scaled_width, scaled_height))  # dsize should be (width, height)
        image -= IMG_MEAN

        # print image.shape  # (1024, 2048, 3) (512, 1024, 3)
        self.dif_height = scaled_height - crop_image_size
        self.dif_width = scaled_width - crop_image_size
        if self.dif_height < 0 or self.dif_width < 0:
            image = numpy_pad_image(image, self.dif_height, self.dif_width)
            scaled_height, scaled_width = image.shape[0], image.shape[1]
        # print image.shape  # (1024, 2048, 3) (864, 1024, 3)
        self.image = image
        self.crop_image_size = crop_image_size

        self.heights = decide_intersection(scaled_height, crop_image_size)
        self.widths = decide_intersection(scaled_width, crop_image_size)

        split_crops = []
        for height in self.heights:
            for width in self.widths:
                image_crop = self.image[height:height + crop_image_size, width:width + crop_image_size]
                split_crops.append(image_crop[np.newaxis, :])

        self.split_crops = np.concatenate(split_crops, axis=0)  # (n, crop_image_size, crop_image_size, 3)

    def get_split_crops(self):
        return self.split_crops

    def reassemble_crops(self, proba_crops):
        # in general, crops are probabilities of self.split_crops.
        assert proba_crops.shape[0:3] == self.split_crops.shape[0:3], \
            '%s vs %s' % (proba_crops.shape[0:3], self.split_crops.shape[0:3])
        # (n, crop_image_size, crop_image_size, num_classes) vs (n, crop_image_size, crop_image_size, 3)

        # reassemble
        reassemble = np.zeros((self.image.shape[0], self.image.shape[1], proba_crops.shape[-1]), np.float32)
        index = 0
        for height in self.heights:
            for width in self.widths:
                reassemble[height:height+self.crop_image_size, width:width+self.crop_image_size] += proba_crops[index]
                index += 1
        # print reassemble.shape

        # crop to original image
        if self.dif_height < 0 or self.dif_width < 0:
            reassemble = numpy_crop_image(reassemble, self.dif_height, self.dif_width)

        return reassemble

def numpy_crop_image(image, dif_height, dif_width):
    # (height, width, channel)
    assert len(image.shape) == 3
    if dif_height < 0:
        if dif_height % 2 == 0:
            pad_before_h = - dif_height // 2
            pad_after_h = dif_height // 2
        else:
            pad_before_h = - dif_height // 2
            pad_after_h = dif_height // 2
        image = image[pad_before_h:pad_after_h]

    if dif_width < 0:
        if dif_width % 2 == 0:
            pad_before_w = - dif_width // 2
            pad_after_w = dif_width // 2
        else:
            pad_before_w = - dif_width // 2
            pad_after_w = dif_width // 2
        image = image[:, pad_before_w:pad_after_w]

    return image

def numpy_pad_image(image, total_padding_h, total_padding_w, image_padding_value=0):
    # (height, width, channel)
    assert len(image.shape) == 3
    pad_before_w = pad_after_w = 0
    pad_before_h = pad_after_h = 0
    if total_padding_h < 0:
        if total_padding_h % 2 == 0:
            pad_before_h = pad_after_h = - total_padding_h // 2
        else:
            pad_before_h = - total_padding_h // 2
            pad_after_h = - total_padding_h // 2 + 1
    if total_padding_w < 0:
        if total_padding_w % 2 == 0:
            pad_before_w = pad_after_w = - total_padding_w // 2
        else:
            pad_before_w = - total_padding_w // 2
            pad_after_w = - total_padding_w // 2 + 1
    image_crop = np.pad(image,
                        ((pad_before_h, pad_after_h), (pad_before_w, pad_after_w), (0, 0)),
                        mode='constant', constant_values=image_padding_value)
    return image_crop

def decide_intersection(total_length, crop_length):
    stride = crop_length * 2 // 3
    times = (total_length - crop_length) // stride + 1
    cropped_starting = []
    for i in range(times):
        cropped_starting.append(stride*i)
    if total_length - cropped_starting[-1] > crop_length:
        cropped_starting.append(total_length - crop_length)
    return cropped_starting

def get_arguments():
    parser = argparse.ArgumentParser(description="Reproduced PSPNet")

    parser.add_argument("--measure-time", action="store_true",
                        help="whether to measure inference time")
    parser.add_argument("--model", type=str, default=SNAPSHOT_DIR,
                        help="Path to restore weights.")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help="Path to save output.")
    parser.add_argument("--flipped-eval", action="store_true",
                        help="whether to evaluate with flipped img.")

    return parser.parse_args()

def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def calculate_time(sess, net):
    start = time.time()
    sess.run(net.layers['data'])
    data_time = time.time() - start

    start = time.time()
    sess.run(net.layers['conv6'])
    total_time = time.time() - start

    inference_time = total_time - data_time

    time_list.append(inference_time)
    print('average inference time: {}'.format(np.mean(time_list)))

def main():
    args = get_arguments()
    print(args)

    coord = tf.train.Coordinator()

    tf.reset_default_graph()

    image_list, label_list = read_labeled_image_list(DATA_DIRECTORY, DATA_LIST_PATH)

     # Create network.
    image_batch = tf.placeholder(tf.float32, [None, input_size, input_size, 3])

    net = PSPNet({'data': [image_batch]}, is_training=False, num_classes=num_classes)

    # Which variables to load.
    restore_var = tf.global_variables()

    # Predictions.
    raw_output = net.layers['conv6'][0]

    raw_output_up = tf.image.resize_bilinear(raw_output, size=[input_size,input_size], align_corners=True)

    # mIoU
    pred_all = tf.placeholder(tf.float32, [None,None])
    raw_all = tf.placeholder(tf.float32, [None,None,None, None])
    pred_flatten = tf.reshape(pred_all, [-1,])
    raw_gt = tf.reshape(raw_all, [-1,])
    indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, num_classes - 1)), 1)
    gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
    pred_label = tf.gather(pred_flatten, indices)

    mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred_label, gt, num_classes=num_classes)

    # Set up tf session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)
    sess.run(tf.local_variables_initializer())

    restore_var = tf.global_variables()

    ckpt = tf.train.get_checkpoint_state(args.model)
    if ckpt and ckpt.model_checkpoint_path:
        loader = tf.train.Saver(var_list=restore_var)
        load_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
        load(loader, sess, ckpt.model_checkpoint_path)
    else:
        print('No checkpoint file found.')


    for step in range(len(image_list)):
        image, label = cv2.imread(image_list[step], 1), cv2.imread(label_list[step], 0)
        label = np.reshape(label, [1, label.shape[0], label.shape[1], 1])

        imgsplitter = ImageSplitter(image, 1.0, input_size, IMG_MEAN)
        feed_dict = {image_batch: imgsplitter.get_split_crops()}

        logits= sess.run(raw_output_up, feed_dict=feed_dict)
        total_logits = imgsplitter.reassemble_crops(logits)

        #mirror
        image_mirror = image[:, ::-1]
        imgsplitter_mirror = ImageSplitter(image_mirror, 1.0, input_size, IMG_MEAN)
        feed_dict = {image_batch: imgsplitter_mirror.get_split_crops()}
        logits_mirror = sess.run(raw_output_up,feed_dict=feed_dict)
        logits_mirror = imgsplitter_mirror.reassemble_crops(logits_mirror)
        total_logits += logits_mirror[:, ::-1]

        prediction = np.argmax(total_logits, axis=-1)

        #=====================================================#
        sess.run([update_op], feed_dict ={pred_all:prediction, raw_all:label})
        
        if step > 0 and args.measure_time:
            calculate_time(sess, net)

        if step % 10 == 0:
            print('Finish {0}/{1}'.format(step, len(image_list)))
            print('step {0} mIoU: {1}'.format(step, sess.run(mIoU)))

    print('step {0} mIoU: {1}'.format(step, sess.run(mIoU)))

if __name__ == '__main__':
    main()
