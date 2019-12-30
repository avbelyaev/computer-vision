# -*- coding: utf-8 -*-
"""face_classifier_haar_cascade

Automatically generated by Colaboratory.
"""

# !pip install face_recognition

import zipfile
import os
import cv2
import face_recognition
import numpy as np
from google.colab.patches import cv2_imshow

with zipfile.ZipFile('data 2.zip', 'r') as zip_ref:
    zip_ref.extractall('.')

# xml with classifier can be found at venv at cv2/data
clf = cv2.CascadeClassifier('/content/haarcascade_frontalface_default.xml')


class Face:
    def __init__(self, img_path: str, clazz: str = None):
        self.clazz = clazz
        self.img = cv2.imread(img_path, cv2.COLOR_BGR2GRAY)
        self.detected_face = None
        self.embedding = None
        self._detect_face()

    def _detect_face(self):
        faces = clf.detectMultiScale(self.img, scaleFactor=1.2, minNeighbors=5)
        if 0 != len(faces):
            x, y, w, h = faces[0]
            self.detected_face = self.img[y:y + w, x:x + h]
            self.embedding = face_recognition.face_encodings(self.img, [[x, y, w, h]])[0]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f'[{self.clazz}] -> embedding[{len(self.embedding)}]'


jen = Face('/content/data/Jennifer_Aniston/Jennifer_Aniston_0006.jpg', 'Jen')

print('image:')
cv2_imshow(jen.img)

print('detected face:')
cv2_imshow(jen.detected_face)

print(jen)

IMG_DATA_ROOT = 'data'

faces = []
for person in os.listdir(IMG_DATA_ROOT):
    for person_img_name in os.listdir(IMG_DATA_ROOT + "/" + person):
        img_path = IMG_DATA_ROOT + "/" + person + "/" + person_img_name
        face = Face(img_path=img_path, clazz=person)

        if face.detected_face is None:
            print('could not detect face on ', img_path)
            continue

        faces.append(face)

print('faces detected: ', len(faces))
x_train = list(map(lambda face: face.embedding, faces))
y_train = list(map(lambda face: face.clazz, faces))

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)

# image not contained in test data
test_face = Face('/content/sample.jpg')
cv2_imshow(test_face.img)

# predict class of image
clazz_predicted = knn.predict([test_face.embedding])
print(clazz_predicted)