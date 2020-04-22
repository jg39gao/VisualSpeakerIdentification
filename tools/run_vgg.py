from cv2 import imread
from cv2 import imshow
from cv2 import waitKey
from cv2 import destroyAllWindows
from cv2 import CascadeClassifier
from cv2 import rectangle
import numpy as np
import glob

class mtcnnFace:
    
    def extract_face(image_path):
        
        # load the photograph
        path = glob.glob(image_path+"/*.jpg")
        face_counts = []
        
        for p in path:
            
            pixels = imread(p)
            # load the pre-trained model
            classifier = CascadeClassifier('haarcascade_frontalface_default.xml')
            # perform face detection
            bboxes = classifier.detectMultiScale(pixels,1.17,6)
            # print bounding box for each detected face
            for box in bboxes:
            # extract
                x, y, width, height = box
                x2, y2 = x + width, y + height
                face_counts.append((x, y, width, height))

        return np.array(face_counts)