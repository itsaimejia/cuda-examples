import numpy as np
from PIL import Image
import os
import cv2
import math

def apply_contrast(image, factor):
    # Aplicamos el contraste
    image = (image - np.mean(image)) * factor + np.mean(image)

    # Ajustamos los valores para que estén dentro del rango [0, 255]
    return np.clip(image, 0, 255)


file_name = os.path.join(os.path.dirname(__file__), 'gato.png')
assert os.path.exists(file_name)
img = Image.open(file_name)
np_img = np.array(img)
width = np_img.shape[0]
height = np_img.shape[1]

img2 = np.zeros((width,height), np.uint8)

for i in range(np_img.shape[0]):
    for j in range(np_img.shape[1]):    
        b = np_img[i,j][0]
        g = np_img[i,j][1]
        r = np_img[i,j][2]  
        img2[i, j] = math.floor(sum(np_img[i, j]) / 3)


res = apply_contrast(img2, 1.5)
cv2.imwrite("contraste.png", res)
cv2.waitKey(0)