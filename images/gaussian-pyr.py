import cv2
import os
from lib_kernels import  gaussianBlur


#cargar archivo
file_name = os.path.join(os.path.dirname(__file__), 'hamster.jpg')
#leer imagen a color
img = cv2.imread(file_name)

pyramid = []

pyramid.append(img)

for i in range(2):
    img = cv2.GaussianBlur(pyramid[i],(3,3),0)
    img = cv2.pyrDown(img)

    pyramid.append(img)

for i in range(3):
    cv2.imwrite('pyr-{}.png'.format(i + 1), pyramid[i])