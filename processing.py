#!/usr/bin/python
# -*- coding: utf-8 -*-

# @uthor: Makram Jandar 
#  |    \|    \ /   \   /  ] /  _] ___/ ___/    |    \ /    |
#  |  o  )  D  )     | /  / /  [(   \(   \_ |  ||  _  |   __|
#  |   _/|    /|  O  |/  / |    _]__  \__  ||  ||  |  |  |  |
#  |  |  |    \|     /   \_|   [_/  \ /  \ ||  ||  |  |  |_ |
#  |  |  |  .  \     \     |     \    \    ||  ||  |  |     |
#  |__|  |__|\_|\___/ \____|_____|\___|\___|____|__|__|___,_|
#          © Data Processing - dl, exploration, flatenning...

import h5py as h5
from numpy import array, reshape, shape
from os.path import isfile
from urllib.request import urlretrieve
import matplotlib.pyplot as plt

#""" Download ready 4 training Dataset """
# def datasetReady():
#     f = 'https://raw.githubusercontent.com/makramjandar/Object-Oriented-Deep-Neural-Networks/master/catvnoncat.h5' 
    
#     if isfile("catvnoncat.h5"):
#         print("catvnoncat.h5 file is ready for exploration...") 
#     else:
#         {urlretrieve(f, "catvnoncat.h5")}
    
#     trainF = array(h5.File("catvnoncat.h5", "r")["trainF"][:])
#     trainL = array(h5.File("catvnoncat.h5", "r")["trainL"][:])
#     testF = array(h5.File("catvnoncat.h5", "r")["testF"][:])
#     testL = array(h5.File("catvnoncat.h5", "r")["testL"][:])
#     classes = array(h5.File("catvnoncat.h5", "r")["classes"][:])
#     return trainF, trainL, testF, testL, classes

""" Download, repack and merge Andrew NG's Datasets """
def repackNgDatasets():
    f = 'catvnoncat.h5'
    urls = {
        "https://github.com/Mashimo/datascience/raw/master/datasets/":
        ["train_" + f, "test_" + f]
    }

    if isfile("./" + f):
        print("train_" + f + " and " + "test_" + f +
              " exists!! ready for exploration...")
    else:
        {urlretrieve(k + f, "./" + f) for k, v in urls.items() for f in v}
        with h5.File(f, 'w') as h:
            h.create_dataset('trainF',
                             data=array(
                                 h5.File('train_' + f, "r")["train_set_x"][:]))
            h.create_dataset(
                'trainL',
                data=array(h5.File(
                    'train_' + f, "r")["train_set_y"][:]).reshape(
                        (1,
                         array(h5.File('train_' + f,
                                       "r")["train_set_y"][:]).shape[0])))
            h.create_dataset('testF',
                             data=array(
                                 h5.File('test_' + f, "r")["test_set_x"][:]))
            h.create_dataset(
                'testL',
                data=array(h5.File('test_' + f, "r")["test_set_y"][:]).reshape(
                    (1, array(h5.File('test_' + f,
                                      "r")["test_set_y"][:]).shape[0])))
            h.create_dataset('classes',
                             data=array(
                                 h5.File('test_' + f, "r")["list_classes"][:]))

    trainF = array(h5.File(f, "r")["trainF"][:])
    trainL = array(h5.File(f, "r")["trainL"][:])
    testF = array(h5.File(f, "r")["testF"][:])
    testL = array(h5.File(f, "r")["testL"][:])
    classes = array(h5.File(f, "r")["classes"][:])
    return trainF, trainL, testF, testL, classes


""" Datasets(Images) exploration """
def exploringImages(x):
    #if x.__contains__('train'):
    fig = plt.figure(figsize=(20, 20))
    for i in range(x.shape[0]):
        fig.add_subplot(30, 30, i + 1).imshow(x[i], interpolation='nearest')
    #else:
    #    print("Only training Sets contains data to explore !!!")

""" flattening Images into vectors ready 4 training """
def flatteningImages(x, y):
    standarizationRatio = 1 / x.max()
    trainX = x.reshape(x.shape[0], -1).T * standarizationRatio
    testX = y.reshape(y.shape[0], -1).T * standarizationRatio
    print("trainX and testX shapes >> ", trainX.shape[0], trainX.shape[1], testX.shape[0], testX.shape[1])
    return trainX, testX