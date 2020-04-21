# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 11:21:24 2020

@author: Ben Mouncer

Description - Code to link all the scripts of the project together
"""

# BM TODO:
# Test code works with full dataset
# Import the face recognition components
# Integrate the models and face recognition

import clean
import video2array
import model

def main():
    
    data_folder = "\data"
    
    # Clean the Text Files
    clean.all_txt(data_folder)
    
    # Find the Raw Training/Test Data and store in np array
    # of shape [No. pictures, 224 * 224 * 3]
    X, Y = video2array.dataset(data_folder, exclude = {"lips-speaker","lips"})   
    
    # Use face detection on a new video, extract images of faces as np arrays
    model.basic(X, Y)
    model.pca(X, Y)
    model.cnn(X, Y)

    # Apply the trained model to the new images of faces


if __name__ == "__main__":
    main()
