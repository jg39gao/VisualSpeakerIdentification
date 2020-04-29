# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 11:21:24 2020

@author: Ben Mouncer

Description - Code to link all the scripts of the project together
"""

import pic2array
import model
   
data_train = "\data_test_train_dev\\train"
data_test = "\data_test_train_dev\\test"
data_dev = "\data_test_train_dev\\dev"

# Beware this will eat memory like Google Chrome
# X Array Takes Up 13GB for the full dataset
X_train, Y_train = pic2array.faces2array_sample(data_train)
X_test, Y_test = pic2array.faces2array_sample(data_test)
X_dev, Y_dev = pic2array.faces2array_sample(data_dev)

# Use face detection on a new video, extract images of faces as np arrays
model.basic(X_train, X_test, Y_train, Y_test)
model.pca2(X_train, X_test, Y_train, Y_test)
model.cnn(X_train, X_test, X_dev, Y_train, Y_test, Y_dev)
