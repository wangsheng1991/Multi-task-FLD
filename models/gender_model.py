# coding=utf8

import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import sys
from utils.common import getDataFromTxt, logger, BBox, processImage
from numpy.linalg import norm
from utils.layers import full_layer, conv_layer, mse, softmaxCE, config

class CNN(object):
    def __init__(self,
        input_shape=[128, 96, 96, 1], 
        n_filter=[32, 64, 128], 
        n_hidden=[500, 500],
        n_y=30,
        receptive_field=[[3, 3], [2, 2], [2, 2]],
        pool_size=[[2, 2], [2, 2], [2, 2]],
        obj_fcn=mse):

        self._sanity_check(input_shape, n_filter, receptive_field, pool_size)

        x_shape = input_shape[:]
        x_shape[0] = None
        
        x = tf.placeholder(shape=x_shape, dtype=tf.float32)
        y = tf.placeholder(shape=(None, n_y), dtype=tf.float32)

        self.x, self.y = x, y

        # ========= CNN layers =========
        n_channel = [input_shape[-1]] + n_filter
        for i in range(len(n_channel) -1):
            filter_shape = receptive_field[i] + n_channel[i:i+2] # e.g. [5, 5, 32, 64]
            pool_shape = [1] + pool_size[i] + [1]
            print 'Filter shape (layer %d): %s' % (i, filter_shape)
            
            conv_and_filter = conv_layer(x, filter_shape, 'conv%d' % i, padding='VALID')
            print 'Shape after conv: %s' % conv_and_filter.get_shape().as_list()
            # norm1 = tf.nn.local_response_normalization(
            #    conv_and_filter, 4, bias=1.0, alpha=0.001 / 9.0,
            #    beta=0.75, name='norm%d'%i)
            pool1 = tf.nn.max_pool(
                #norm1,
                conv_and_filter,
                ksize=pool_shape,
                strides=pool_shape,
                padding='SAME',
                name='pool%d' % i)
            print 'Shape after pooling: %s' % pool1.get_shape().as_list()
            x = pool1

        # ========= Fully-connected layers =========
        dim = np.prod(x.get_shape()[1:].as_list())
        x = tf.reshape(x, [-1, dim])
        print 'Total dim after CNN: %d' % dim
        for i, n in enumerate(n_hidden):
            x = full_layer(x, n, layer_name='full%d' % i) # nonlinear=tf.nn.relu
        yhat = full_layer(x, n_y, layer_name='output', nonlinear=tf.identity)

        self.yhat = yhat
        
        self.batch_size = input_shape[0]
        self.lr = tf.placeholder(dtype=tf.float32)

        self.objective = softmaxCE(y, yhat)
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.objective)
        tf.scalar_summary(self.objective.op.name, self.objective)

        self.sess = tf.Session(config=config)
        
        correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(yhat, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def predict(self, x):
        ''' Input images (Nx96x96x1), return 30 feature predictors '''
        return self.sess.run(self.yhat, feed_dict={self.x: x})

    def train(self, train_batches, valid_set=None, 
        lr=1e-2, n_epoch=100, logdir='model2'):

        saver = tf.train.Saver(tf.all_variables())

        summary_op = tf.merge_all_summaries()

        init = tf.initialize_all_variables()
        self.sess.run(init)
        
        summary_writer = tf.train.SummaryWriter(
            logdir=logdir,
            graph=self.sess.graph)

        loss_train_record = list()
        loss_valid_record = list()
        start_time = time.gmtime()

        n_batch = train_batches.n_batch
        for i in range(n_epoch):
            loss_train_sum = 0.0
            loss_valid_sum = 0.0
            for images, _, genders, _, _, _, _ in train_batches:
                _, loss, summary_str = self.sess.run(
                    [self.optimizer, self.objective, summary_op],
                    feed_dict={
                        self.x: images,
                        self.y: genders,
                        self.lr: lr})
                loss_train_sum += loss
                if valid_set:
                    loss, acc = self.sess.run(
                        [self.objective, self.accuracy],
                        feed_dict={
                            self.x: valid_set.images,
                            self.y: valid_set.genders})
                else:
                    acc = 0.0
                    loss = 0.0
                loss_valid_sum += loss
                print('Batch accuracy: ' + str(acc))
            loss_train_sum /= n_batch
            loss_valid_sum /= n_batch

            end_time = time.mktime(time.gmtime())

            print 'Epoch %04d, %.8f, %.8f,  %0.8f| %.2f sec per epoch' % (
                i, loss_train_sum, loss_valid_sum,
                loss_train_sum/loss_valid_sum,
                (end_time - time.mktime(start_time)) / (i+1))
            loss_train_record.append(loss_train_sum)    # np.log10()
            loss_valid_record.append(loss_valid_sum)    # np.log10()
            
            if i % 1 == 0:
                ckpt = os.path.join(logdir, 'model.ckpt')
                saver.save(self.sess, ckpt) # 存的是 session
                summary_writer.add_summary(summary_str, i)  # 千萬別太常用，會超慢
            
        end_time = time.gmtime()
        print time.strftime('%H:%M:%S', start_time)
        print time.strftime('%H:%M:%S', end_time)

    def _sanity_check(self, input_shape, receptive_field, n_filter, pool_size):
        assert len(input_shape) == 4, 'Input size is confined to 2'
        assert len(receptive_field) == len(n_filter), \
            'Inconsistent argument: receptive_field (%d) & n_filter (%d)' % (
                len(receptive_field), len(n_filter))
        assert len(receptive_field) == len(pool_size), \
            'Inconsistent argument: receptive_field (%d) & n_filter (%d)' % (
                len(receptive_field), len(pool_size))

    def test(self):
        # Restore variables from disk.
        try:
            saver = tf.train.Saver()
            ckpt = os.path.join('gender_model', 'model.ckpt')
            saver.restore(self.sess, ckpt)
            print("Model restored.")
        except ValueError:
            print("Model not in model.ckpt")
            return

        src = 'upload/10.jpg'

        fd = FaceDetector()

        img = cv2.imread(src)
        gray_img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)

        bboxes = []
        landmarks = []
        for bbox in fd.detectFace(gray_img):
            f_bbox = bbox.subBBox(0.1, 0.9, 0.2, 1)
            f_face = gray_img[f_bbox.top:f_bbox.bottom+1,f_bbox.left:f_bbox.right+1]

            # Resize
            f_face = cv2.resize(f_face, (39, 39))
            f_face = f_face.reshape((39, 39, 1))
            f_face = f_face / 255.0

            F_imgs = []
            F_imgs.append(f_face)
            F_imgs = np.asarray(F_imgs)

            # Normalize
            #F_imgs = processImage(F_imgs)
            pred_val = self.sess.run(self.yhat, feed_dict={self.x: F_imgs})
            print(pred_val)

class FaceDetector(object):
    """
        class FaceDetector use VJ detector
    """

    def __init__(self):
        self.cc = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

    def detectFace(self, img):
        rects = self.cc.detectMultiScale(img, scaleFactor=1.2, minNeighbors=2, \
                    minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
        for rect in rects:
            rect[2:] += rect[:2]
            yield BBox([rect[0], rect[2], rect[1], rect[3]])
