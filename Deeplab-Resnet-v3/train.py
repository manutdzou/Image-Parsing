"""
This code is based on DrSleep's framework: https://github.com/DrSleep/tensorflow-deeplab-resnet 
"""

from __future__ import print_function

import argparse
import os
import sys
import time

import tensorflow as tf
import numpy as np

from model import DeepLabResNetModel
from tools import decode_labels, prepare_label
from image_reader import ImageReader
from tensorflow.contrib.metrics import streaming_mean_iou

#IMG_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)
IMG_MEAN = np.array((72.41519599, 82.93553322, 73.18188461), dtype=np.float32) # RGB, Cityscapes.

BATCH_SIZE = 3
GPU_NUMS = 2
DATA_DIRECTORY = '/dfsdata/jinyi_data/DATASET/Cityscapes'
DATA_LIST_PATH = './list/train_list.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '720,720'
LEARNING_RATE = 1e-2
MOMENTUM = 0.9
NUM_CLASSES = 19
NUM_STEPS = 200001
POWER = 0.9
RANDOM_SEED = 123
WEIGHT_DECAY = 0.0001
RESTORE_FROM = './pre_train_model'
SNAPSHOT_DIR = './model/'
SAVE_NUM_IMAGES = 4
SAVE_PRED_EVERY = 40000

def get_arguments():
    parser = argparse.ArgumentParser(description="PSPnet Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--gpu_nums", type=int, default=GPU_NUMS,
                        help="Number of gpus.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--is-training", action="store_true", default=True,
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true", default=False,
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-blur", action="store_true",default=True,
                        help='random blur: brightness/saturation/constrast')
    parser.add_argument("--random-mirror", action="store_true",default=True,
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",default=True,
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-rotate", action="store_true",default=True,
                        help="random rotate")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--update-mean-var", action="store_true",default=True,
                        help="whether to get update_op from tf.Graphic_Keys")
    parser.add_argument("--train-beta-gamma", action="store_true",default=True,
                        help="whether to train beta & gamma in bn layer")
    return parser.parse_args()

def save(saver, sess, logdir, step):
   model_name = 'model.ckpt'
   checkpoint_path = os.path.join(logdir, model_name)
    
   if not os.path.exists(logdir):
      os.makedirs(logdir)
   saver.save(sess, checkpoint_path, global_step=step)
   print('The checkpoint has been created.')

def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def main():
    """Create the model and start the training."""
    args = get_arguments()
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    
    #tf.set_random_seed(args.random_seed)
    
    coord = tf.train.Coordinator()

    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # Using Poly learning rate policy 
        base_lr = tf.constant(args.learning_rate)
        step_ph = tf.placeholder(dtype=tf.float32, shape=())
        learning_rate = tf.train.exponential_decay(base_lr,
                                    step_ph,
                                    20000,
                                    0.5,
                                    staircase=True)

        tf.summary.scalar('lr', learning_rate)

        opt = tf.train.MomentumOptimizer(learning_rate, 0.9)

        #opt = tf.train.RMSPropOptimizer(learning_rate, 0.9, momentum=0.9, epsilon=1e-10)

        #opt = tf.train.AdamOptimizer(learning_rate)

        losses = []
        train_op = []

        total_batch_size = args.batch_size*args.gpu_nums

        with tf.name_scope('DeepLabResNetModel') as scope:
            with tf.name_scope("create_inputs"):
                reader = ImageReader(
                    args.data_dir,
                    args.data_list,
                    input_size,
                    args.random_blur,
                    args.random_scale,
                    args.random_mirror,
                    args.random_rotate,
                    args.ignore_label,
                    IMG_MEAN,
                    coord)
                image_batch, label_batch = reader.dequeue(total_batch_size)

                images_splits = tf.split(axis=0, num_or_size_splits=args.gpu_nums, value=image_batch)
                labels_splits = tf.split(axis=0, num_or_size_splits=args.gpu_nums, value=label_batch)
   
            net = DeepLabResNetModel({'data': images_splits}, is_training=True, num_classes=args.num_classes)
    
            raw_output_list = net.layers['fc_voc12']

            num_valide_pixel = 0
            for i in range(len(raw_output_list)):
                with tf.device('/gpu:%d' % i):
                    raw_output_up = tf.image.resize_bilinear(raw_output_list[i], size=input_size, align_corners=True)

                    tf.summary.image('images_{}'.format(i), images_splits[i]+IMG_MEAN, max_outputs = 4)
                    tf.summary.image('labels_{}'.format(i), labels_splits[i], max_outputs = 4)

                    tf.summary.image('predict_{}'.format(i), tf.cast(tf.expand_dims(tf.argmax(raw_output_up, -1),3),tf.float32), max_outputs = 4)

                    all_trainable = [v for v in tf.trainable_variables()]

                    # Predictions: ignoring all predictions with labels greater or equal than n_classes
                    raw_prediction = tf.reshape(raw_output_up, [-1, args.num_classes])
                    label_proc = prepare_label(labels_splits[i], tf.stack(raw_output_up.get_shape()[1:3]), num_classes=args.num_classes, one_hot=False) # [batch_size, h, w]
                    raw_gt = tf.reshape(label_proc, [-1,])
                    #indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, args.num_classes - 1)), 1)
                    indices = tf.where(tf.logical_and(tf.less(raw_gt, args.num_classes), tf.greater_equal(raw_gt, 0)))
                    gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
                    prediction = tf.gather(raw_prediction, indices)
                    mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(tf.argmax(tf.nn.softmax(prediction), axis=-1), gt, num_classes=args.num_classes)
                    tf.summary.scalar('mean IoU_{}'.format(i), mIoU)
                    train_op.append(update_op)
                                                                                             
                    # Pixel-wise softmax loss.
                    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
                    num_valide_pixel += tf.shape(gt)[0]
 
                    losses.append(tf.reduce_sum(loss))

            l2_losses = [args.weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]
            reduced_loss = tf.truediv(tf.reduce_sum(losses), tf.cast(num_valide_pixel, tf.float32)) + tf.add_n(l2_losses)
            tf.summary.scalar('average_loss', reduced_loss) 

        grads = tf.gradients(reduced_loss, all_trainable, colocate_gradients_with_ops=True)

        variable_averages = tf.train.ExponentialMovingAverage(0.99, step_ph)

        variables_to_average = (tf.trainable_variables() + tf.moving_average_variables())
        variables_averages_op = variable_averages.apply(variables_to_average)

        train_op = tf.group(opt.apply_gradients(zip(grads, all_trainable)), *train_op)
        
        train_op = tf.group(train_op, variables_averages_op)

        summary_op = tf.summary.merge_all()
    
        # Set up tf session and initialize variables. 
        config = tf.ConfigProto()
        config.allow_soft_placement=True
        sess = tf.Session(config=config)
        init = [tf.global_variables_initializer(),tf.local_variables_initializer()]
        sess.run(init)
        # Saver for storing checkpoints of the model.
        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=2)

        
        #restore from resnet imagenet, bised and local_step is in moving_average
        restore_var = [v for v in tf.trainable_variables() if 'fc' not in v.name]+[v for v in tf.global_variables() if ('moving_mean' in v.name or 'moving_variance' in v.name) and ('biased' not in v.name and 'local_step' not in v.name)]

        ckpt = tf.train.get_checkpoint_state(args.restore_from)
        if ckpt and ckpt.model_checkpoint_path:
            loader = tf.train.Saver(var_list=restore_var)
            load(loader, sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint file found.')

        """
        #restore from snapshot
        restore_var = tf.global_variables()

        ckpt = tf.train.get_checkpoint_state(args.snapshot_dir)
        if ckpt and ckpt.model_checkpoint_path:
            loader = tf.train.Saver(var_list=restore_var, allow_empty=True)
            load_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
            load(loader, sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint file found.')
            load_step = 0
        """
        # Start queue threads.
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        summary_writer = tf.summary.FileWriter(args.snapshot_dir, graph=sess.graph)
        # Iterate over training steps.
        for step in range(args.num_steps):
            start_time = time.time()
        
            feed_dict = {step_ph: step}
            if step % args.save_pred_every == 0 and step != 0:
                loss_value, _ = sess.run([reduced_loss, train_op], feed_dict=feed_dict)
                save(saver, sess, args.snapshot_dir, step)
            elif step%100 == 0:
                summary_str, loss_value, _, IOU = sess.run([summary_op, reduced_loss, train_op, mIoU], feed_dict=feed_dict)
                duration = time.time() - start_time
                summary_writer.add_summary(summary_str, step)
                print('step {:d} \t loss = {:.3f}, mean_IoU = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, IOU, duration))
            else:
                loss_value, _ = sess.run([reduced_loss, train_op], feed_dict=feed_dict)
        
        coord.request_stop()
        coord.join(threads)
    
if __name__ == '__main__':
    main()
