# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 17:37:57 2020

@author: George Tsoumis
"""

import glob
import os
import re


file_list = glob.glob(os.path.join(os.getcwd(), "*_gt.txt"))

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
