import cv2
import numpy as np
from PIL import Image
import os

path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
ids = []

for image in os.listdir(path):
    img_path = os.path.join(path, image)
    gray_img = Image.open(img_path).convert('L')
    img_np = np.array(gray_img, 'uint8')

    id = int(image.split('.')[1])
    faces.append(img_np)
    ids.append(id)

recognizer.train(faces, np.array(ids))
recognizer.save('trainer.yml')

print("Training Complete")
