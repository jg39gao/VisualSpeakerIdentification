Prerequisite:

1. tensorflow-gpu==1.3
2. keras==2.0.8
3. imgaug
4. opencv-python
5. h5py

API:

from predict import HeadDetector

hd = HeadDetector()

hd.detect_head(inputpath, exportImg=False, outputpath = "./output/")

-inputpath: The directory of the input raw images (end with "/", ".jpg" or ".png" files only).
-exportImg: If export the annotated images or not(False as default).
-outputpath: The directory of the output annotated images (end with "/", "./output/" as default).
         The name of a output image starts with the last 3 characters before ".jpg" or ".png"

This function returns a list of numpy arrays. Each array shows the [x,y,w,h] of the boxes in this image.



You can take the "demo.py" or "demo_jupyter.ipynb" as a reference.
