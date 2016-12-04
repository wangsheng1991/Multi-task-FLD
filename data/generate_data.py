#!/usr/bin/env python2.7
# coding: utf-8

import os
from os.path import join, exists
import cv2
import numpy as np
from common import getDataFromTxt, shuffle_in_unison_scary, logger, createDir, processImage
import h5py

def generate_data(ftxt, fname):
    data = getDataFromTxt(ftxt)
    F_imgs = []
    F_landmarks = []
    F_genders = []
    F_smiles = []
    F_glasses = []
    F_poses = []
    F_all_attr = []

    for (imgPath, bbox, landmarkGt, gender, smile, glasses, pose, all_attr) in data:
        img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
        
        assert(img is not None)
        logger("process %s" % imgPath)
        
        f_bbox = bbox
        #f_bbox = bbox.subBBox(-0.05, 1.05, -0.05, 1.05)

        f_face = img[f_bbox.top:f_bbox.bottom+1,f_bbox.left:f_bbox.right+1]
        f_face = cv2.resize(f_face, (39, 39))
        f_face = f_face.reshape((39, 39, 1))
        f_face = f_face / 255.0

        f_landmark = landmarkGt.reshape((10))

        F_imgs.append(f_face)
        F_landmarks.append(f_landmark)
        F_genders.append(gender)
        F_smiles.append(smile)
        F_glasses.append(glasses)
        F_poses.append(pose)
        F_all_attr.append(all_attr)

    F_imgs = np.asarray(F_imgs)
    F_landmarks = np.asarray(F_landmarks)
    F_genders = np.asarray(F_genders)
    F_smiles = np.asarray(F_smiles)
    F_glasses = np.asarray(F_glasses)
    F_poses = np.asarray(F_poses)
    F_all_attr = np.asarray(F_all_attr)

    shuffle_in_unison_scary(F_imgs, F_landmarks, F_genders, F_smiles, F_glasses, F_poses, F_all_attr)
    
    logger("generate %s" % fname)
    with h5py.File(fname, 'w') as h5:
        h5['data'] = F_imgs.astype(np.float32)
        h5['landmarks'] = F_landmarks.astype(np.float32)
        h5['genders'] = F_genders.astype(np.float32)
        h5['smiles'] = F_smiles.astype(np.float32)
        h5['glasses'] = F_glasses.astype(np.float32)
        h5['poses'] = F_poses.astype(np.float32)
        h5['all_attr'] = F_all_attr.astype(np.float32)
        
if __name__ == '__main__':
    # train data
    TRAIN = 'train'

    train_txt = join(TRAIN, 'merged_training.txt')
    generate_data(train_txt, 'train.h5')

    test_txt = join(TRAIN, 'merged_testing.txt')
    generate_data(test_txt, 'test.h5')

