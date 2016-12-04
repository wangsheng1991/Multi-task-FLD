# coding=utf8

import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets import base
import h5py

# Batch iterator
class BatchRenderer(object):
    def __init__(self, images, landmarks, genders, smiles, glasses, poses, all_attr, batch_size, shuffle=True):
        n_sample = images.shape[0]
        self._n_batch = n_sample // batch_size
        self._batch_index = 0
        self._indices = np.arange(n_sample)
        
        self._images = images
        self._landmarks = landmarks
        self._genders = genders
        self._smiles = smiles
        self._glasses = glasses
        self._poses = poses
        self._all_attr = all_attr
        
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._n_sample = n_sample

    def __iter__(self):
        return self

    def next(self):
        if self._batch_index >= self._n_batch:
            self._batch_index = 0
            if self._shuffle:
                np.random.shuffle(self._indices)
            raise StopIteration
        else:
            i, b = self._batch_index, self._batch_size
            index = self._indices[i*b: (i+1)*b+1]
            self._batch_index += 1
            images = self._images[index]
            landmarks = self._landmarks[index]
            genders = self._genders[index]
            smiles = self._smiles[index]
            glasses = self._glasses[index]
            poses = self._poses[index]
            all_attr = self._all_attr[index]
            
            return (images, landmarks, genders, smiles, glasses, poses, all_attr)

    @property
    def n_batch(self):
        return self._n_batch


class DataSet(object):
    def __init__(self, images, landmarks, genders, smiles, glasses, poses, all_attr):
        self.images = np.array(images)
        self.landmarks = np.array(landmarks)
        self.genders = np.array(genders)
        self.smiles = np.array(smiles)
        self.glasses = np.array(glasses)
        self.poses = np.array(poses)
        self.all_attr = np.array(all_attr)

# Get facial landmark detection data from file
def read_data_set(filename):
    with h5py.File(filename, 'r') as h5:
        data = np.array(h5['data'])
        landmarks = np.array(h5['landmarks'])
        genders = np.array(h5['genders'])
        smiles = np.array(h5['smiles'])
        glasses = np.array(h5['glasses'])
        poses = np.array(h5['poses'])
        all_attr = np.array(h5['all_attr'])

        return DataSet(data, landmarks, genders, smiles, glasses, poses, all_attr)

# Get facial landmark detection data
def read_data_sets():
	TRAIN_DATA = 'train.h5'
	TEST_DATA = 'test.h5'
	val_size = 2000

	train = read_data_set(TRAIN_DATA)
	test = read_data_set(TEST_DATA)
	validation = None

	# train = train[val_size:]
	# validation = train[:val_size]
	
	return base.Datasets(train=train, validation=validation, test=test)

if __name__ == '__main__':
	d = read_data_sets()
	print(d.train.num_examples)
	print(d.validation)
	print(d.test.num_examples)