import cv2
import os
import numpy as np
from lib_kernels import sharpen, sobel, prewitt, gaussianBlur



#cargar archivo
file_name = os.path.join(os.path.dirname(__file__), 'hamster.jpg')

#leer imagen y convertir a escala de grises
img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
#crear imagen de escala de grises
cv2.imwrite('grey.png', img)

#aplicar kernel
img_result = gaussianBlur(img)

#crear imagen con kernel aplicado
cv2.imwrite('sharpen.png',img_result)
