from PIL import Image
import face_recognition
import glob
import numpy as np

class hog_face_detection:
    
    def hog_extract_face(image_path):
        # Load the jpg file into a numpy array
        path = glob.glob(image_path+"/*.jpg")
        face_counts = []
        
        for p in path:
            image = face_recognition.load_image_file(p)
            face_locations = face_recognition.face_locations(image,number_of_times_to_upsample=2)

            for face_location in face_locations:
        
                top, right, bottom, left = face_location
                face_counts.append((left, top, right, bottom))
        
        return np.array(face_counts)