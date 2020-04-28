# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 11:21:24 2020

@author: Ben Mouncer

Description - Code to link all the scripts of the project together
"""

# TODO
# Apply the trained models to the new images of faces

import pic2array
import model

def main():
    
    data_train = "\data_test_train_dev\\train"
    data_test = "\data_test_train_dev\\test"
    
    # Beware this will eat memory like Google Chrome
    # X Array Takes Up 13GB for the full dataset
    X_train, Y_train = pic2array.faces2array_sample(data_train)
    X_test, Y_test = pic2array.faces2array_sample(data_test)

    # Use face detection on a new video, extract images of faces as np arrays
    model.basic(X_train, X_test, Y_train, Y_test)
    model.pca(X_train, X_test, Y_train, Y_test)
    model.cnn(X_train, X_test, Y_train, Y_test)

    # Apply the trained model to the new images of faces (TODO)


if __name__ == "__main__":
    main()
