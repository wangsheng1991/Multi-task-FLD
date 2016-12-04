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
from utils.layers import full_layer, conv_layer, mse, softmaxCE, sigmoidCE, config

"""
input_shape=[128, 96, 96, 1],
n_filter=[32, 64, 128], 
n_hidden=[500, 500],
n_y_landmark=30,
n_y_attribute=30,
receptive_field=[[3, 3], [2, 2], [2, 2]],
pool_size=[[2, 2], [2, 2], [2, 2]],
attribute="gender",
logdir="train"
"""
class SCNN(object):
    def __init__(self,
        input_shape,
        n_filter, 
        n_hidden,
        n_y_landmark,
        n_y_attribute,
        receptive_field,
        pool_size,
        attribute,
        logdir):

        print('Attribute: ' + attribute)
        print('Storing in ' + logdir)
        
        self._sanity_check(input_shape, n_filter, receptive_field, pool_size)

        x_shape = input_shape[:]
        x_shape[0] = None
        
        x = tf.placeholder(shape=x_shape, dtype=tf.float32)
        y_landmark = tf.placeholder(shape=(None, n_y_landmark), dtype=tf.float32)
        y_attribute = tf.placeholder(shape=(None, n_y_attribute), dtype=tf.float32)

        self.x, self.y_landmark, self.y_attribute = x, y_landmark, y_attribute

        # Loss
        self.objective = 0

        # ========= CNN layers =========
        n_channel = [input_shape[-1]] + n_filter
        for i in range(len(n_channel) -1):
            filter_shape = receptive_field[i] + n_channel[i:i+2] # e.g. [5, 5, 32, 64]
            pool_shape = [1] + pool_size[i] + [1]
            print 'Filter shape (layer %d): %s' % (i, filter_shape)

            # Convolutional layers
            conv_and_filter = conv_layer(x, filter_shape, 'conv%d' % i, padding='VALID')
            print 'Shape after conv: %s' % conv_and_filter.get_shape().as_list()

            # Batch normalization
            # norm1 = tf.nn.local_response_normalization(
            #    conv_and_filter, 4, bias=1.0, alpha=0.001 / 9.0,
            #    beta=0.75, name='norm%d_%d'%(i,1))

            # Pooling layer
            pool = tf.nn.max_pool(
                #norm1,
                conv_and_filter,
                ksize=pool_shape,
                strides=pool_shape,
                padding='SAME',
                name='pool%d' % i)
            print 'Shape after pooling: %s' % pool.get_shape().as_list()

            x = pool

        # ========= Fully-connected layers =========
        dim = np.prod(x.get_shape()[1:].as_list())
        x = tf.reshape(x, [-1, dim])
        print 'Total dim after CNN: %d' % dim
        for i, n in enumerate(n_hidden):
            x = full_layer(x, n, layer_name='full%d' % i) # nonlinear=tf.nn.relu
        
        yhat_1 = full_layer(x, n_y_landmark, layer_name='output_1', nonlinear=tf.identity)
        yhat_2 = full_layer(x, n_y_attribute, layer_name='output_2', nonlinear=tf.identity)

        self.yhat_1 = yhat_1
        self.yhat_2 = yhat_2

        self.batch_size = input_shape[0]
        self.lr = tf.placeholder(dtype=tf.float32)

        self.objective_1 = mse(y_landmark, yhat_1)
        if attribute == "all":
            self.objective_2 = sigmoidCE(y_attribute, yhat_2)
        else:
            self.objective_2 = softmaxCE(y_attribute, yhat_2)

        self.obj1_lambda = 200
        self.objective = self.obj1_lambda * self.objective_1 + self.objective_2
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.objective)

        tf.scalar_summary(self.objective.op.name, self.objective)

        self.sess = tf.Session(config=config)
        
        correct_pred = tf.equal(tf.argmax(y_attribute, 1), tf.argmax(yhat_2, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        self.logdir = logdir
        self.attribute = attribute
        
    def predict(self, x):
        ''' Input images (Nx96x96x1), return 30 feature predictors '''
        return self.sess.run(self.yhat, feed_dict={self.x: x})

    def train(self, train_batches, valid_set=None, 
        lr=1e-2, n_epoch=100):

        saver = tf.train.Saver(tf.all_variables())

        summary_op = tf.merge_all_summaries()

        init = tf.initialize_all_variables()
        self.sess.run(init)
        
        summary_writer = tf.train.SummaryWriter(
            logdir=self.logdir,
            graph=self.sess.graph)

        loss_train_record = list()
        loss_valid_record = list()
        start_time = time.gmtime()

        n_batch = train_batches.n_batch
        for i in range(n_epoch):
            loss_train_sum = 0.0
            loss_valid_sum = 0.0
            for images, landmarks, genders, smiles, glasses, poses, all_attr in train_batches:
                if self.attribute == "gender":
                    y_attribute = genders
                elif self.attribute == "smile":
                    y_attribute = smiles
                elif self.attribute == "glasses":
                    y_attribute = glasses
                elif self.attribute == "pose":
                    y_attribute = poses
                elif self.attribute == "all":
                    y_attribute = all_attr

                _, loss, summary_str = self.sess.run(
                    [self.optimizer, self.objective, summary_op],
                    feed_dict={
                        self.x: images,
                        self.y_landmark: landmarks,
                        self.y_attribute: y_attribute,
                        self.lr: lr})
                loss_train_sum += loss

                if valid_set:
                    if self.attribute == "gender":
                        y_attribute = valid_set.genders
                    elif self.attribute == "smile":
                        y_attribute = valid_set.smiles
                    elif self.attribute == "glasses":
                        y_attribute = valid_set.glasses
                    elif self.attribute == "pose":
                        y_attribute = valid_set.poses
                    elif self.attribute == "all":
                        y_attribute = valid_set.all_attr

                    mse, ce, loss = self.sess.run(
                        [self.objective_1, self.objective_2, self.objective],
                        feed_dict={
                            self.x: valid_set.images,
                            self.y_landmark: valid_set.landmarks,
                            self.y_attribute: y_attribute})
                else:
                    mse, softmax, loss = 0.0, 0.0, 0.0
                loss_valid_sum += loss
                print("Batch loss (mse, ce, total):", self.obj1_lambda * mse, ce, loss)
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
                ckpt = os.path.join(self.logdir, 'model.ckpt')
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

    def test2(self):
        # Restore variables from disk.
        try:
            saver = tf.train.Saver()
            ckpt = os.path.join(self.logdir, 'model.ckpt')
            saver.restore(self.sess, ckpt)
            print("Model restored.")
        except ValueError:
            print("Model not in model.ckpt")
            return

        src = 'upload/3.jpg'
        dst = 'upload/3_landmarked.jpg'

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
            pred_val = self.sess.run(self.yhat_1, feed_dict={self.x: F_imgs})
            pred_val = pred_val[0].reshape((5,2))
            pred_val = bbox.reprojectLandmark(pred_val)

            bboxes.append(bbox)
            landmarks.append(pred_val)

        for i in range(len(bboxes)):
            img = drawLandmark(img, bboxes[i], landmarks[i])

        cv2.imwrite(dst, img)

    def test(self):
        TXT = 'train/testImageList.txt'
        template = '''################## Summary #####################
        Test Number: %d
        Time Consume: %.03f s
        FPS: %.03f
        LEVEL - %d
        Mean Error:
            Left Eye       = %f
            Right Eye      = %f
            Nose           = %f
            Left Mouth     = %f
            Right Mouth    = %f
        Failure:
            Left Eye       = %f
            Right Eye      = %f
            Nose           = %f
            Left Mouth     = %f
            Right Mouth    = %f
        '''

        t = time.clock()
        # Restore variables from disk.
        try:
            saver = tf.train.Saver()
            ckpt = os.path.join(self.logdir, 'model.ckpt')
            saver.restore(self.sess, ckpt)
            print("Model restored.")
        except ValueError:
            print("Model not in model.ckpt")
            return

        data = getDataFromTxt(TXT)
        error = np.zeros((len(data), 5))
        for i in range(len(data)):
            imgPath, bbox, landmarkGt, _, _, _, _ = data[i]
            img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
            
            assert(img is not None)
            logger("process %s" % imgPath)

            # Crop
            f_bbox = bbox.subBBox(-0.05, 1.05, -0.05, 1.05)
            f_face = img[f_bbox.top:f_bbox.bottom+1,f_bbox.left:f_bbox.right+1]

            # Resize
            f_face = cv2.resize(f_face, (39, 39))
            f_face = f_face.reshape((39, 39, 1))
            f_face = f_face / 255.0
            landmarkP = self.sess.run(self.yhat_1, feed_dict={self.x: [f_face]})
            landmarkP = landmarkP.reshape((5,2))

            # real landmark
            landmarkP = bbox.reprojectLandmark(landmarkP)
            landmarkGt = bbox.reprojectLandmark(landmarkGt)
            error[i] = evaluateError(landmarkGt, landmarkP, bbox)

        t = time.clock() - t
        N = len(error)
        fps = N / t
        errorMean = error.mean(0)

        # failure
        failure = np.zeros(5)
        threshold = 0.05
        for i in range(5):
            failure[i] = float(sum(error[:, i] > threshold)) / N

        # log string
        s = template % (N, t, fps, 0, errorMean[0], errorMean[1], errorMean[2], \
            errorMean[3], errorMean[4], failure[0], failure[1], failure[2], \
            failure[3], failure[4])
        print s

        name = 'fl_' + str(self.attribute) + '_split'
        logfile = 'log/' + name + '.log'
        with open(logfile, 'w') as fd:
            fd.write(s)

        # plot error hist
        plotError(error, name)

def plotError(e, name):
    # config global plot
    plt.rc('font', size=16)
    plt.rcParams["savefig.dpi"] = 240

    fig = plt.figure(figsize=(20, 15))
    binwidth = 0.001
    yCut = np.linspace(0, 70, 100)
    xCut = np.ones(100)*0.05
    # left eye
    ax = fig.add_subplot(321)
    data = e[:, 0]
    ax.hist(data, bins=np.arange(min(data), max(data) + binwidth, binwidth), normed=1)
    ax.plot(xCut, yCut, 'r', linewidth=2)
    ax.set_title('left eye')
    # right eye
    ax = fig.add_subplot(322)
    data = e[:, 1]
    ax.hist(data, bins=np.arange(min(data), max(data) + binwidth, binwidth), normed=1)
    ax.plot(xCut, yCut, 'r', linewidth=2)
    ax.set_title('right eye')
    # nose
    ax = fig.add_subplot(323)
    data = e[:, 2]
    ax.hist(data, bins=np.arange(min(data), max(data) + binwidth, binwidth), normed=1)
    ax.plot(xCut, yCut, 'r', linewidth=2)
    ax.set_title('nose')
    # left mouth
    ax = fig.add_subplot(325)
    data = e[:, 3]
    ax.hist(data, bins=np.arange(min(data), max(data) + binwidth, binwidth), normed=1)
    ax.plot(xCut, yCut, 'r', linewidth=2)
    ax.set_title('left mouth')
    # right mouth
    ax = fig.add_subplot(326)
    data = e[:, 4]
    ax.hist(data, bins=np.arange(min(data), max(data) + binwidth, binwidth), normed=1)
    ax.plot(xCut, yCut, 'r', linewidth=2)
    ax.set_title('right mouth')

    fig.suptitle('%s'%name)
    fig.savefig('log/%s.png'%name)

def evaluateError(landmarkGt, landmarkP, bbox):
    e = np.zeros(5)
    for i in range(5):
        e[i] = norm(landmarkGt[i] - landmarkP[i])
    print(e)
    print(bbox.w)
    e = e / bbox.w
    print 'landmarkGt'
    print landmarkGt
    print 'landmarkP'
    print landmarkP
    print 'error', e
    return e

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

def drawLandmark(img, bbox, landmark):
    cv2.rectangle(img, (bbox.left, bbox.top), (bbox.right, bbox.bottom), (0,0,255), 2)
    for x, y in landmark:
        cv2.circle(img, (int(x), int(y)), 1, (0,255,0), -1)
    return img
