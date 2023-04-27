import cv2
import os
import numpy as np
from lib_kernels import sharpen, sobel, prewitt, gaussianBlur
import time


#cargar archivo
file_name = os.path.join(os.path.dirname(__file__), 'hamster.jpg')

#leer imagen y convertir a escala de grises
img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)

#crear imagen de escala de grises
cv2.imwrite('grey.png', img)

start = time.time()
#aplicar kernel
img_result = gaussianBlur(img)
end = time.time()
print('Sin opencv+cuda: ', end - start)
#crear imagen con kernel aplicado
cv2.imwrite('gaussian_cpu.png',img_result)