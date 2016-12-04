# coding=utf8

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from models.split_model import SCNN
from data.mtfl_data import read_data_sets, BatchRenderer
import sys

attribute = 'gender'
logdir = 'trained_models/fl_' + attribute + '_split_model'

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('lr', 1e-2, 'learning rate')
tf.app.flags.DEFINE_float('valid', 0.2, 'fraction of validation set')
tf.app.flags.DEFINE_integer('n_epoch', 1000, 'number of epochs')
tf.app.flags.DEFINE_integer('batch_size', 128, 'batch size')
#tf.app.flags.DEFINE_string('train_dir', logdir, 'dir to store models')

# Global settings
szImg = 39
n_y_landmark = 10
n_y_attribute = 2

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

    nn = SCNN(
        input_shape=[FLAGS.batch_size, szImg, szImg, 1],
        n_filter=[20, 40, 60, 80],
        n_hidden=[120],
        n_y_landmark=n_y_landmark,
        n_y_attribute=n_y_attribute,
        receptive_field=[[4, 4], [3, 3], [3, 3], [2, 2]],
        pool_size=[[2, 2], [2, 2], [2, 2], [1, 1]],
        attribute=attribute,
        logdir=logdir)

    nn.train(
        batches,
        datasets.test,
        lr=FLAGS.lr,
        n_epoch=FLAGS.n_epoch)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        attribute = sys.argv[1]
        logdir = 'trained_models/fl_' + attribute + '_split_model'
        if attribute == 'all':
            n_y_attribute = 8
        else:
            n_y_attribute = 2
    
        if tf.gfile.Exists(logdir):
            tf.gfile.DeleteRecursively(logdir)
        tf.gfile.MakeDirs(logdir)
        tf.app.run()
    else:
        print("Please pass in argument")
