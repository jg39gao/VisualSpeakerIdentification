# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 17:37:57 2020
@author: George Tsoumis

"""

import glob
import os
import re
import ntpath

def all_txt(path_t = "\data"):

    file_list = glob.glob(os.path.join(os.getcwd() + path_t, "*_gt.txt"))
    #file_list = [name for name in os.listdir(path) if name.endswith("_gt.txt")]
    print(os.getcwd())
    print("%d Files Found" % len(file_list))
    
    corpus = []
    
    for file_path in file_list:
        with open(file_path) as f_input:
            corpus.append(f_input.read())
    
    for i in range (len(corpus)):
        tempfile = corpus[i]
        tempfile = re.sub("[,-]|\d|lips,speaker|head,speaker|offscreen,speaker|head|lips|\n", "", tempfile )
        if tempfile != '':
            print('error in: ', file_list[i])
            print('find the error and add another sub command below')
    
    for i in range (len(corpus)):
        corpus[i] = re.sub(r'(lips,speaker)','lips-speaker',corpus[i])
        corpus[i] = re.sub(r'(offscreen,speaker)','offscreen-speaker',corpus[i])
        corpus[i] = re.sub(r'(offscreen-speaker)','offscreen-speaker\n',corpus[i])
        corpus[i] = re.sub(r'(offscreen-speaker-speaker\n)','offscreen-speaker',corpus[i])
        corpus[i] = re.sub(r'(lips, speaker)','lips-speaker',corpus[i])
        corpus[i] = re.sub(r'(head,speaker)','head-speaker',corpus[i])
        corpus[i] = re.sub(r'(head,speake)','head-speaker',corpus[i])
        
        corpus[i] = re.sub(r'(head_speaker)','head-speaker',corpus[i])
        corpus[i] = re.sub(r'(lips_speaker)','lips-speaker',corpus[i])
        corpus[i] = re.sub(r'(offscreen_speaker)','offscreen-speaker',corpus[i])
        
        #corpus[i] = re.sub(r'\n','\'\n\'',corpus[i])
        file1 = file_list[i]
        with open(file1, "w") as text_file:
            print(corpus[i], file=text_file)
            
    for i in range (len(corpus)):
        tempfile = corpus[i]
        tempfile = re.sub("[,-]|\d|lips-speaker|head-speaker|offscreen-speaker|head|lips|\n", "", tempfile )
        if tempfile != '':
            print('error in: ', file_list[i],tempfile)
            print('find the error and add another sub command below')

 def all_txt_basic(path_t = "\data"):

    file_list = glob.glob(os.path.join(os.getcwd() + path_t, "*_gt.txt"))
    print("%d Files Found" % len(file_list))
    
    
    for file_path in file_list:
        with open(file_path) as f:
            # Read the entire file into memory
            raw_file = f.read()
            
            # Replace Rules
            raw_file = raw_file.replace("_", "-")
            
            raw_file = raw_file.replace("lips,speaker",  "lips-speaker")
            raw_file = raw_file.replace("lips, speaker", "lips-speaker")

            raw_file = raw_file.replace("head,speaker", "head-speaker")
            raw_file = raw_file.replace("head, speaker", "head-speaker")
            
            raw_file = raw_file.replace("offscreen,speaker", "offscreen-speaker")
            raw_file = raw_file.replace("offscreen, speaker", "offscreen-speaker")
            
        with open(file_path, "w") as f:
            f.write(raw_file)

def all_txt_basic(path_t = "\data"):

    file_list = glob.glob(os.path.join(os.getcwd() + path_t, "*_gt.txt"))
    print("%d Files Found" % len(file_list))
    
    
    for file_path in file_list:
        with open(file_path) as f:
            # Read the entire file into memory
            raw_file = f.read()
            
            # Replace Rules
            raw_file = raw_file.replace("_", "-")
            
            raw_file = raw_file.replace("lips,speaker",  "lips-speaker")
            raw_file = raw_file.replace("lips, speaker", "lips-speaker")
            
            raw_file = raw_file.replace("head,speaker", "head-speaker")
            raw_file = raw_file.replace("head, speaker", "head-speaker")
            
            raw_file = raw_file.replace("offscreen,speaker", "offscreen-speaker")
            raw_file = raw_file.replace("offscreen, speaker", "offscreen-speaker")
            
            raw_file = raw_file.replace("head0", "head")
            raw_file = raw_file.replace("lips0", "lips")
            
            raw_file = raw_file.replace("head1", "head")
            raw_file = raw_file.replace("lips1", "lips")
            
            raw_file = raw_file.replace("head2", "head")
            raw_file = raw_file.replace("lips2", "lips")
            
            raw_file = raw_file.replace("head3", "head")
            raw_file = raw_file.replace("lips3", "lips")
            
            raw_file = raw_file.replace("lips--speaker", "lips-speaker")
            raw_file = raw_file.replace("head--speaker", "head-speaker")
            
            raw_file = raw_file.replace("offscreen,", "offscreen-speaker,")
            raw_file = raw_file.replace("offscreen-speaker-speaker", "offscreen-speaker")
            
            raw_file = raw_file.replace("lip-speaker", "lips-speaker")
            raw_file = raw_file.replace("lips+speaker", "lips-speaker")
            
            raw_file = raw_file.replace("had,", "head,")
            raw_file = raw_file.replace("lips-speker",  "lips-speaker")
            raw_file = raw_file.replace("head-spekaer",  "head-speaker")
            raw_file = raw_file.replace("lpis-speaker", "lips-speaker")
            
            raw_file = raw_file.replace("head-speaking", "head-speaker")
            raw_file = raw_file.replace("lips-speaking", "lips-speaker")
            
            raw_file = raw_file.replace("tips", "lips")
            raw_file = raw_file.replace("offsccreen-speaker", "offscreen-speaker")
            
            raw_file = raw_file.replace("hed", "head")
            raw_file = raw_file.replace("offsereen,speaker", "offscreen-speaker")
            raw_file = raw_file.replace("offsereen-speaker", "offscreen-speaker")
            
        with open(file_path, "w") as f:
            f.write(raw_file)

def missing_head(path_t = "\data"):
    
    file_list = glob.glob(os.path.join(os.getcwd() + path_t, "*_gt.txt"))
    print("%d Files Found" % len(file_list))
    
    for file_path in file_list:
        with open(file_path) as f:
            # Read the entire file into memory
            raw_file = f.read()
            
            if ("head" not in raw_file):
                print(file_path)

def sort_classes(path = "\data_test_train_dev\\train"):
    
    # Get a list of the pictures in the path
    # Sort pics into seperate folders for the two classes
    
    file_list = glob.glob(os.path.join(os.getcwd() + path, "*.png"))
    print(os.getcwd() + path)
    print(len(file_list))
    for i in range(len(file_list)):
        
        file_name = ntpath.basename(file_list[i])
        
        if ("speaker" in file_name):
            print(file_name)
            
            os.rename(file_list[i], os.getcwd() + path + "\speaker\\" + file_name)
        else:
            print(file_name)
            os.rename(file_list[i], os.getcwd() + path + "\\non-speaker\\" + file_name)
