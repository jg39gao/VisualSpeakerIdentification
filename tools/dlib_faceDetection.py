import sys
import dlib
import glob
import numpy as np

class dlib_face_detection:
    
    def dlib_extract_face(image_path):
        
        detector = dlib.get_frontal_face_detector()
        path = glob.glob(image_path+"/*.jpg")
        face_counts = []

        for f in path:
            
            img = dlib.load_rgb_image(f)
            dets = detector(img, 1)

            for d in dets:
                
                left, top, right, bottom = d.left(), d.top(), d.right(), d.bottom()
                face_counts.append((left, top, right, bottom))
        
        return np.array(face_counts)