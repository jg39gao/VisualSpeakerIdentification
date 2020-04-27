# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 11:21:24 2020

@author: Ben Mouncer

Description - Code to link all the scripts of the project together
"""

# TODO
# Modify models to take test train and not split internally
# Clean up CNN and PCA and get them working

import pic2array
import model

def main():
    
    data = "\data_test_train_dev\\train"
    # Beware this will eat memory like Google Chrome
    # X Array Takes Up 13GB for the full dataset
    
    X, Y = pic2array.faces2array(data)

    # Use face detection on a new video, extract images of faces as np arrays
    model.basic(X, Y)
    model.pca(X, Y)
    model.cnn(X, Y)

    # Apply the trained model to the new images of faces (TODO)


if __name__ == "__main__":
    main()
