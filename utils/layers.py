# coding=utf8

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import time

from tensorflow.python.training import moving_averages

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
config = tf.ConfigProto(gpu_options=gpu_options)

def glorot_init(shape, constant=1): 
    """
    Initialization of network weights using Xavier Glorot's proposal
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    """
    # assert len(shape) == 2
    # fan_in, fan_out = shape
    _dim_sum = np.sum(shape)
    low = -constant*np.sqrt(6.0/_dim_sum) 
    high = constant*np.sqrt(6.0/_dim_sum)
    w = tf.random_uniform(
        # (fan_in, fan_out),
        shape, 
        minval=low, 
        maxval=high, 
        dtype=tf.float32)
    return w

def full_layer(x, fan_out, layer_name, nonlinear=tf.nn.relu):
    ''' Fully-connected layer '''
    with tf.variable_scope(layer_name):
        fan_in = x.get_shape().as_list()[-1]
        shape = (fan_in, fan_out)
        w = tf.get_variable('w', initializer=glorot_init(shape))
        b = tf.get_variable('b', initializer=tf.zeros([fan_out]))
        xw = tf.matmul(x, w)
        o = tf.add(xw, b)
        y = nonlinear(o)
    return y

def _variable_with_weight_decay(name, initializer, wd):
    var = tf.get_variable(name, initializer=initializer)
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
    return var

def conv_layer(x, shape, layer_name, nonlinear=tf.nn.relu, 
    stride=[1, 1, 1, 1], wd=0.0, padding='SAME'):
    ''' Conv layer '''
    with tf.variable_scope(layer_name) as scope:
        w = _variable_with_weight_decay(
            name='w',
            initializer=glorot_init(shape),
            wd=wd)
        c = tf.nn.conv2d(x, w, stride, padding=padding)   
        b = tf.get_variable(
            name='b',
            shape=[shape[-1]],
            initializer=tf.constant_initializer(0.1))
        xwb = tf.nn.bias_add(c, b)
        o = nonlinear(xwb, name=scope.name)
        return o

def bn_layer(x, shape, layer_name, axis, is_training=True, epsilon=1e-3, decay=0.999):
    '''
    Batch Normalization Layer
    '''
    moving_mean = tf.Variable(
        tf.zeros(shape),
        name='moving_mean',
        trainable=False)
    moving_variance = tf.Variable(
        tf.ones(shape),
        name='moving_variance',
        trainable=False)

    control_inputs = list()
    if is_training:
        # Calculate the moments based on the individual batch.
        mean, variance = tf.nn.moments(x, axis)
        update_moving_mean = moving_averages.assign_moving_average(
            moving_mean, mean, decay)
        update_moving_variance = moving_averages.assign_moving_average(
            moving_variance, variance, decay)
        control_inputs = [update_moving_mean, update_moving_variance]
    else:
        # Just use the moving_mean and moving_variance.
        mean = moving_mean
        variance = moving_variance

    # Normalize the activations.
    with tf.variable_scope(layer_name):
        beta = tf.get_variable('b', initializer=tf.ones(shape))
        gamma = tf.get_variable('g', initializer=tf.zeros(shape))

    with tf.control_dependencies(control_inputs):
        outputs = tf.nn.batch_normalization(
            x, mean, variance, beta, gamma, epsilon)
        outputs.set_shape(x.get_shape())
    # if activation:
    #   outputs = activation(outputs)
    return outputs

def mse(y, yhat, name='obj_mse'):
    with tf.variable_scope(name):
        obj = tf.sub(y, yhat)
        obj = tf.square(obj)
        obj = tf.reduce_mean(obj)
    return obj

def softmaxCE(y, yhat, name='obj_sm_ce'):
    with tf.variable_scope(name):
        obj = tf.nn.softmax_cross_entropy_with_logits(yhat, y)
        obj = tf.reduce_mean(obj)
    return obj

def sigmoidCE(y, yhat, name='obj_sig_ce'):
    with tf.variable_scope(name):
        obj = tf.nn.sigmoid_cross_entropy_with_logits(yhat, y)
        obj = tf.reduce_mean(obj)
    return obj

def error_plot(trn_loss, vld_loss, filename):
    '''
    trn_loss: training_loss_record
    vld_loss: validation_loss_record
    '''
    plt.figure()
    plt.plot(trn_loss, label='train')
    plt.plot(vld_loss, c='r', label='validation')
    plt.xlabel('mini-batch')
    plt.ylabel('loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig(filename)


def LeakyReLU(x, a=1e-2):
    return tf.nn.relu(x) + tf.mul(a, tf.nn.relu(-x))

"""
  img: a vector of 96*96 length
label: a vector of 30 targets
"""
def show_img_keypoint(img, label, truth=None):
    plt.imshow(img.reshape((96, 96)), cmap='gray')
    # plt.scatter(label[0::2] * 96, label[1::2] * 96)
    # if truth is not None:
    #     plt.scatter(truth[0::2] * 96, truth[1::2] * 96, c='r', marker='x')
    # plt.show()
    plt.scatter(label[0::2] * 48 + 48, label[1::2] * 48 + 48)
    if truth is not None:
        plt.scatter(truth[0::2] * 48 + 48, truth[1::2] * 48 + 48, 
            c='r', marker='x')
    # plt.show()

def weight_variable(shape, std=0.1):
    w_initial = tf.truncated_normal(shape, stddev=std)
    return tf.Variable(w_initial)

def bias_variable(shape):
    b_initial = tf.constant(0.1, shape=shape)
    return tf.Variable(b_initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, 
        strides=[1, 1, 1, 1], 
        padding='VALID')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='VALID')
