import numpy as np
import cv2
import os
<<<<<<< Updated upstream
import time
from lib.filter2d import apply_kernel
=======
from lib_kernels import sharpen, edge, sobel, prewitt, gaussianBlur, convolve2D
import time 
>>>>>>> Stashed changes


#cargar archivo
file_name = os.path.join(os.path.dirname(__file__), 'hamster.jpg')
assert os.path.exists(file_name)

<<<<<<< Updated upstream
    img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    
    
    start = time.time()
    img_result = apply_kernel(img)
    end = time.time()
    # Muestra la imagen resultante
    print('tiempo:',end - start)
    # Muestra la imagen resultante
    cv2.imwrite('result.png', img_result)
=======
>>>>>>> Stashed changes

img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)

cv2.imwrite('grey.png', img)
img_result = sobel(img, 'x')

cv2.imwrite('result.png',img_result)
