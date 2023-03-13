import numpy as np
import cv2
import os
import math 
import time
from numba import jit

@jit
def img_to_grey(img):
    height, width = img.shape[0:2]
    output = np.zeros(img.shape[0:2])
    for i in range(height):
        for j in range(width):    
            b = img[i,j][0]
            g = img[i,j][1]
            r = img[i,j][2]  
            output[i, j] = math.floor(sum(img[i, j]) / 3)
    
    return output
    
file_name = os.path.join(os.path.dirname(__file__), 'gato.png')
assert os.path.exists(file_name)

img = cv2.imread(file_name)

start = round(time.time())
img2 = img_to_grey(img)
end = round(time.time())
print('Segundos: ', end - start)

# write image
cv2.imwrite("output.png", img2)
cv2.waitKey(0)



