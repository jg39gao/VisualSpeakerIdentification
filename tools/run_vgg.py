from matplotlib import pyplot as plt
from mtcnn.mtcnn import MTCNN
from numpy import asarray
from PIL import Image
import os
import cv2
import glob
import numpy as np


class mtcnnFace:

    def extract_face(image_path):
        # load image and detect faces
        path = glob.glob(image_path + "/*.jpg")
        count = 0
        face_counts = []

        # Create directory
        dirName = 'images'
        try:
            # Create target Directory
            os.mkdir(dirName)
            print("Directory ", dirName, " Created ")
        except FileExistsError:
            print("Directory ", dirName, " already exists")

        for p in path:
            face_images = []
            image = plt.imread(p)
            detector = MTCNN()
            faces = detector.detect_faces(image)

            for n in range(len(faces)):
                # extract the bounding box from the requested face

                x1, y1, width, height = faces[n]['box']
                x2, y2 = x1 + width, y1 + height
                face_counts.append((x1, y1, width, height))

                # extract the face
                face_boundary = image[y1:y2, x1:x2]

                # resize pixels to the model size
                face_image = Image.fromarray(face_boundary)
                face_image = face_image.resize((224, 224))
                face_array = asarray(face_image)
                face_images.append(face_array)

        # save faces from the extracted faces
            for num in range(len(face_images)):
                count += 1
                plt.axis("off")
                cv2.imwrite(os.path.join(dirName,"image{}.jpg").format(count), cv2.cvtColor(face_images[num], cv2.COLOR_RGB2BGR))

        return np.array(face_counts)

if __name__ == '__main__':
    #input directory path
    vgg = mtcnnFace.extract_face("ROI")
    print(vgg)
