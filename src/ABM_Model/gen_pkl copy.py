#!/usr/bin/env python

import pickle as pkl

import numpy

from PIL import Image
import cv2
import os

paths = ["../BTTR_Model/data/2019"]
labels = ["../BTTR_Model/data/2019/caption.txt"]

outFile = 'data/offline-test-2019.pkl'
outlabel = 'data/test-caption-2019.txt'
oupFp_feature = open(outFile, 'wb')
file_label = open(outlabel, 'w')
features = {}
channels = 1
sentNum = 0

for img_path in paths:
    image_path = os.path.join(os.getcwd(), img_path)
    for i in os.listdir(os.path.join(os.getcwd(), image_path)):
        print(i)
        key = str(i.split('.')[0])
        if i == "caption.txt":
            continue

        if os.path.exists(image_path + '/' + key + '.png'):
            image_file = image_path + '/' + key + '.png'
        else:
            image_file = image_path + '/' + key + '.bmp'
        print(image_file)
        im = cv2.imread(image_file)
        mat = numpy.zeros([channels, im.shape[0], im.shape[1]], dtype='uint8')
        for channel in range(channels):
            mat[channel, :, :] = im[:, :, 0]  # 3 channel -> 1 channel
        sentNum = sentNum + 1
        features[key] = mat
        if sentNum % 500 == 0:
            print('process sentences ', sentNum)

for filename in labels:
    idx = 0
    for line in open(os.path.join(os.getcwd(), filename)):
        file_label.writelines(line)
        idx += 1

file_label.close()

print('load images done. sentence number ', sentNum)

pkl.dump(features, oupFp_feature)
print('save file done')
oupFp_feature.close()
