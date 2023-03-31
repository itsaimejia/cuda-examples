import numpy as np
import cv2
import os
import time 
from lib_kernels import sobel


#cargar archivo
file_name = os.path.join(os.path.dirname(__file__), 'hamster.jpg')
assert os.path.exists(file_name)


img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
cv2.imwrite('grey.png', img)
start = time.time()
img_result = sobel(img, 'x')
end = time.time()

print(end - start)
cv2.imwrite('result.png',img_result)
