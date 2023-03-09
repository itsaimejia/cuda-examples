import numpy as np
import cv2
import os
import math 

file_name = os.path.join(os.path.dirname(__file__), 'gato.png')
assert os.path.exists(file_name)

img = cv2.imread(file_name)

width = img.shape[0]
height = img.shape[1]

img2 = np.zeros((width,height,1), np.uint8)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):    
        b = img[i,j][0]
        g = img[i,j][1]
        r = img[i,j][2]  
        img2[i, j] = math.floor(sum(img[i, j]) / 3)

# write image
cv2.imwrite("output.png", img2)
cv2.waitKey(0)



