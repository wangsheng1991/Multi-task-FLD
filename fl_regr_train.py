# coding=utf8

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from models.regr_model import CNN
from data.mtfl_data import read_data_sets, BatchRenderer
from utils.layers import mse

FLAGS = tf.app.flags.FLAGS  
tf.app.flags.DEFINE_float('lr', 1e-2, 'learning rate')
tf.app.flags.DEFINE_float('valid', 0.2, 'fraction of validation set')
tf.app.flags.DEFINE_integer('n_epoch', 1000, 'number of epochs')
tf.app.flags.DEFINE_integer('batch_size', 128, 'batch size')
tf.app.flags.DEFINE_string('train_dir', 'trained_models/fl_regr_single_model/', 'dir to store models')

# Global settings
szImg = 39
n_x = szImg * szImg
n_y = 10

def main(args=None):
    datasets = read_data_sets()

    batches = BatchRenderer(
        datasets.train.images,
        datasets.train.landmarks,
        datasets.train.genders,
        datasets.train.smiles,
        datasets.train.glasses,
        datasets.train.poses,
        datasets.train.all_attr,
        FLAGS.batch_size)

    nn = CNN(
        input_shape=[FLAGS.batch_size, szImg, szImg, 1],
        n_filter=[20, 40, 60, 80],
        n_hidden=[120],
        n_y=n_y,
        receptive_field=[[4, 4], [3, 3], [3, 3], [2, 2]],
        pool_size=[[2, 2], [2, 2], [2, 2], [1, 1]],
        obj_fcn=mse,
        logdir=FLAGS.train_dir)

    nn.train(
        batches,
        datasets.test,
        lr=FLAGS.lr,
        n_epoch=FLAGS.n_epoch)

if __name__ == '__main__':
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    tf.app.run()
